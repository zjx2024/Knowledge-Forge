"""Structure-aware input contracts for GraphRAG Native."""

from __future__ import annotations

import json
from pathlib import Path
import asyncio

import pyarrow.parquet as pq
import pytest

from knowledge_forge.graphrag_native_inputs import (
    GraphRAGNativeInputBuilder,
    NativeInputError,
)
from knowledge_forge.graphrag_native_config import (
    GraphRAGNativePromptProfileBuilder,
    GraphRAGNativeSettingsBuilder,
)
from knowledge_forge.graphrag_native_runner import GraphRAGNativeRunner


class _Database:
    def __init__(self) -> None:
        self.documents = [{
            "id": 7, "project_id": 3, "active_revision_id": "revision-7",
            "doc_hash": "source-7", "original_filename": "source.md",
        }]
        self.revision = {
            "revision_id": "revision-7", "project_id": 3, "document_id": 7,
            "source_hash": "source-7", "status": "active",
        }

    def list_documents(self, project_id):
        assert project_id == 3
        return list(self.documents)

    def get_document_revision(self, revision_id):
        return self.revision if revision_id == "revision-7" else None


class _Service:
    def __init__(self, blocks):
        self.db = _Database()
        self.blocks = blocks

    def document_structure(self, document_id):
        assert document_id == 7
        return {"blocks": self.blocks}


def _blocks():
    return [
        {"key": "h1", "block_type": "heading", "text": "标题", "order": 1, "heading_path": ["标题"]},
        {"key": "p1", "block_type": "paragraph", "text": "普通段落。", "order": 2, "heading_path": ["标题"], "page": 1},
        {"key": "r1", "block_type": "table_row", "text": "矿物=锂；国家=甲；产量=1", "order": 3, "heading_path": ["标题"], "metadata": {"table_id": "t1", "row": 1, "headers": ["矿物", "国家", "产量"]}},
        {"key": "r2", "block_type": "table_row", "text": "矿物=钴；国家=乙；产量=2", "order": 4, "heading_path": ["标题"], "metadata": {"table_id": "t1", "row": 2, "headers": ["矿物", "国家", "产量"]}},
        {"key": "l1", "block_type": "list_item", "text": "列表项甲", "order": 5, "metadata": {"list_id": "l", "item_index": 1}},
        {"key": "l2", "block_type": "list_item", "text": "列表项乙", "order": 6, "metadata": {"list_id": "l", "item_index": 2}},
    ]


def test_native_input_is_deterministic_and_preserves_table_row_list_boundaries(tmp_path):
    first = GraphRAGNativeInputBuilder(_Service(_blocks())).build(
        project_id=3, run_root=tmp_path / "first",
    )
    second = GraphRAGNativeInputBuilder(_Service(_blocks())).build(
        project_id=3, run_root=tmp_path / "second",
    )

    assert first.source_snapshot_hash == second.source_snapshot_hash
    assert first.input_manifest_hash == second.input_manifest_hash
    assert len(first.input_documents) == 5
    texts = [item.text for item in first.input_documents]
    assert all(text.startswith("[KF_SOURCE_ID:") for text in texts)
    assert not any("产量=1\n\n矿物=钴" in text for text in texts)
    assert not any("列表项甲\n\n列表项乙" in text for text in texts)
    assert any("Section: 标题" in text for text in texts)
    assert any("Columns: 矿物 | 国家 | 产量" in text for text in texts)
    source_map = pq.read_table(first.source_map_path).to_pylist()
    assert {row["table_id"] for row in source_map if row["table_id"]} == {"t1"}
    assert {row["list_id"] for row in source_map if row["list_id"]} == {"l"}
    manifest_records = [json.loads(line) for line in first.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert {row["id"] for row in manifest_records} == {item.id for item in first.input_documents}


def test_native_input_fails_closed_when_active_document_has_no_structure_blocks(tmp_path):
    with pytest.raises(NativeInputError, match="legacy chunks"):
        GraphRAGNativeInputBuilder(_Service([])).build(project_id=3, run_root=tmp_path / "run")


def test_native_input_preflight_marks_empty_before_any_graphrag_workflow(tmp_path):
    del tmp_path
    preflight = GraphRAGNativeInputBuilder(_Service([])).preflight(project_id=3)
    assert preflight.ready is False
    assert preflight.source_snapshot_hash
    assert "legacy chunks" in preflight.reason


def test_native_input_accepts_only_empty_graphrag_init_input_directory(tmp_path):
    empty_root = tmp_path / "empty"
    (empty_root / "input").mkdir(parents=True)
    result = GraphRAGNativeInputBuilder(_Service(_blocks())).build(project_id=3, run_root=empty_root)
    assert result.documents_path.is_file()
    stale_root = tmp_path / "stale"
    (stale_root / "input").mkdir(parents=True)
    (stale_root / "input" / "old.parquet").write_text("old", encoding="utf-8")
    with pytest.raises(NativeInputError, match="空目录"):
        GraphRAGNativeInputBuilder(_Service(_blocks())).build(project_id=3, run_root=stale_root)


def test_native_input_accepts_executor_evidence_manifest_but_never_overwrites_its_own_outputs(tmp_path):
    """Candidate evidence is written before input freezing in every real run."""

    root = tmp_path / "run"
    (root / "input").mkdir(parents=True)
    (root / "kf-manifest").mkdir()
    (root / "kf-manifest" / "candidate-identity.json").write_text("{}", encoding="utf-8")
    result = GraphRAGNativeInputBuilder(_Service(_blocks())).build(project_id=3, run_root=root)
    assert result.source_map_path.is_file()
    assert result.manifest_path.is_file()

    # Isolate the manifest ownership guard from the earlier input-directory
    # guard: an existing Documents file is itself already fail-closed.
    result.documents_path.unlink()
    with pytest.raises(NativeInputError, match="拒绝覆盖旧尝试"):
        GraphRAGNativeInputBuilder(_Service(_blocks())).build(project_id=3, run_root=root)


def test_real_graphrag_311_init_then_native_input_and_config_contract(tmp_path):
    # This is deliberately CLI/config-only: no index and no model/provider call.
    runner = GraphRAGNativeRunner(timeout_seconds=30)
    runner.initialize(root=tmp_path, completion_model="test-completion", embedding_model="test-embedding", environment={"GRAPHRAG_API_KEY": "placeholder"})
    # The executor persists candidate evidence after ``init`` and before
    # freezing Documents.  Exercise that exact shared-directory lifecycle.
    (tmp_path / "kf-manifest").mkdir(parents=True, exist_ok=True)
    (tmp_path / "kf-manifest" / "candidate-identity.json").write_text("{}", encoding="utf-8")
    build = GraphRAGNativeInputBuilder(_Service(_blocks())).build(project_id=3, run_root=tmp_path)
    prompt_profile = GraphRAGNativePromptProfileBuilder.apply(
        run_root=tmp_path,
        texts=tuple(item.text for item in build.input_documents),
    )
    original_cwd = Path.cwd()
    settings = GraphRAGNativeSettingsBuilder().write_and_validate(
        run_root=tmp_path, completion_model="test-completion", embedding_model="test-embedding",
        embedding_dimension=2, api_base="http://127.0.0.1:11434/v1",
    )
    runner.config_dry_run(
        root=tmp_path, environment={"GRAPHRAG_API_KEY": "placeholder"},
    )
    assert build.documents_path.is_file()
    assert prompt_profile["language"]["language_code"] == "zh"
    assert settings["settings_path"].is_file()
    assert Path.cwd() == original_cwd


def test_actual_graphrag_311_parquet_reader_preserves_native_input_ids_and_titles(tmp_path):
    """Do not mistake a config dry-run for an input-reader compatibility test."""

    from graphrag.config.load_config import load_config
    from graphrag_input import create_input_reader
    from graphrag_storage import create_storage

    runner = GraphRAGNativeRunner(timeout_seconds=30)
    runner.initialize(root=tmp_path, completion_model="test-completion", embedding_model="test-embedding", environment={"GRAPHRAG_API_KEY": "placeholder"})
    build = GraphRAGNativeInputBuilder(_Service(_blocks())).build(project_id=3, run_root=tmp_path)
    GraphRAGNativeSettingsBuilder().write_and_validate(
        run_root=tmp_path, completion_model="test-completion", embedding_model="test-embedding",
        embedding_dimension=2, api_base="http://127.0.0.1:11434/v1",
    )
    original_cwd = Path.cwd()
    try:
        config = load_config(tmp_path)
    finally:
        # The real 3.1.1 loader may mutate cwd; this test must not hide that
        # process-wide side effect from later tests.
        import os
        os.chdir(original_cwd)
    reader = create_input_reader(config.input, create_storage(config.input_storage))

    async def collect():
        return [document async for document in reader]

    documents = asyncio.run(collect())
    assert {document.id for document in documents} == {item.id for item in build.input_documents}
    assert {document.title for document in documents} == {item.title for item in build.input_documents}
