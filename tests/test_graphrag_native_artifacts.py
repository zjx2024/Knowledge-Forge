"""Offline bundle integrity and corruption checks for GraphRAG Native."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from knowledge_forge.graphrag_native_artifacts import (
    FINAL_TABLE_COLUMNS,
    STANDARD_WORKFLOWS,
    GraphRAGNativeBundleError,
    GraphRAGNativeBundleValidator,
    describe_file,
    safe_bundle_path,
    assert_bundle_has_no_plaintext_credentials,
    redact_plaintext_credential_lines,
    write_run_manifest,
)
from knowledge_forge.graphrag_native_config import (
    GraphRAGNativePromptProfileBuilder,
)


def _write(root, relative, rows):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def _valid_bundle(root):
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "index.log").write_text("native workflow complete", encoding="utf-8")
    _write(root, "input/documents.parquet", [{"id": "doc-1", "title": "native:doc-1", "text": "input", "raw_data": "{}"}])
    (root / "kf-manifest").mkdir(parents=True, exist_ok=True)
    (root / "kf-manifest/input-documents-manifest.jsonl").write_text(
        json.dumps({"id": "doc-1", "title": "native:doc-1", "text_sha256": "fixture"}) + "\n",
        encoding="utf-8",
    )
    _write(root, "kf-manifest/source-map.parquet", [{
        "project_id": 1, "document_id": 1, "revision_id": "revision-1", "source_hash": "source",
        "input_document_id": "doc-1", "input_title": "native:doc-1", "block_ids": "[\"b1\"]",
    }])
    _write(root, "output/documents.parquet", [{
        "id": "doc-1", "human_readable_id": "1", "title": "native:doc-1", "text": "input",
        "text_unit_ids": ["tu-1"], "creation_date": "2026-01-01", "raw_data": "{}",
    }])
    _write(root, "output/text_units.parquet", [{
        "id": "tu-1", "human_readable_id": "1", "text": "input", "n_tokens": 1,
        "document_id": "doc-1", "entity_ids": ["e-1", "e-2"], "relationship_ids": ["r-1"], "covariate_ids": [],
    }])
    entities = [
        {"id": "e-1", "human_readable_id": "1", "title": "A", "type": "thing", "description": "A", "text_unit_ids": ["tu-1"], "frequency": 1, "degree": 1},
        {"id": "e-2", "human_readable_id": "2", "title": "B", "type": "thing", "description": "B", "text_unit_ids": ["tu-1"], "frequency": 1, "degree": 1},
    ]
    _write(root, "output/entities.parquet", entities)
    _write(root, "output/relationships.parquet", [{
        "id": "r-1", "human_readable_id": "1", "source": "A", "target": "B", "description": "related", "weight": 1.0, "combined_degree": 2, "text_unit_ids": ["tu-1"],
    }])
    _write(root, "output/communities.parquet", [{
        "id": "c-1", "human_readable_id": "1", "community": "1", "level": 0, "parent": "-1", "children": [], "title": "community", "entity_ids": ["e-1", "e-2"], "relationship_ids": ["r-1"], "text_unit_ids": ["tu-1"], "period": "", "size": 2,
    }])
    _write(root, "output/community_reports.parquet", [{
        "id": "cr-1", "human_readable_id": "1", "community": "1", "level": 0, "parent": "-1", "children": [], "title": "community", "summary": "summary", "full_content": "report", "rank": 1.0, "rating_explanation": "because", "findings": [], "full_content_json": json.dumps({"title": "community", "summary": "summary", "findings": [], "rating": 1.0, "rating_explanation": "because"}), "period": "", "size": 2,
    }])
    _write(root, "output/raw_entities.parquet", [{"title": "A", "type": "thing"}, {"title": "B", "type": "thing"}])
    _write(root, "output/raw_relationships.parquet", [{"source": "A", "target": "B"}])
    (root / "output/graph.graphml").write_text("<graphml/>", encoding="utf-8")
    settings = {"vector_store": {"vector_size": 2, "index_schema": {key: {"index_name": key, "vector_size": 2} for key in ("text_unit_text", "entity_description", "community_full_content")}}}
    (root / "settings.redacted.yaml").write_text(yaml.safe_dump(settings), encoding="utf-8")
    ledger = root / "kf-manifest/workflow-ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("".join(json.dumps({"workflow": name, "status": "disabled" if name == "extract_covariates" else "completed"}) + "\n" for name in STANDARD_WORKFLOWS), encoding="utf-8")
    db = lancedb.connect(str(root / "output/lancedb"))
    db.create_table("text_unit_text", pa.Table.from_pylist([{ "id": "tu-1", "vector": [0.1, 0.2]}]))
    db.create_table("entity_description", pa.Table.from_pylist([{ "id": "e-1", "vector": [0.1, 0.2]}, {"id": "e-2", "vector": [0.2, 0.3]}]))
    db.create_table("community_full_content", pa.Table.from_pylist([{ "id": "cr-1", "vector": [0.3, 0.4]}]))
    artifact_paths = [
        root / "input/documents.parquet", root / "kf-manifest/source-map.parquet",
        root / "kf-manifest/input-documents-manifest.jsonl", root / "settings.redacted.yaml", ledger,
        *(root / "output" / name for name in FINAL_TABLE_COLUMNS),
        root / "output/raw_entities.parquet", root / "output/raw_relationships.parquet", root / "output/graph.graphml",
    ]
    artifacts = [describe_file(root, path, logical_name=path.relative_to(root).as_posix()) for path in artifact_paths]
    write_run_manifest(root, identity={"run_id": "fake"}, artifacts=artifacts,
                       workflow_ledger_path="kf-manifest/workflow-ledger.jsonl",
                       settings_redacted_path="settings.redacted.yaml", lancedb_relative_path="output/lancedb")


def _refresh_manifest_artifact(root: Path, relative: str) -> None:
    """Keep the file hash valid so corruption tests reach semantic checks."""

    manifest_path = root / "kf-manifest/run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / relative
    table = pq.read_table(path) if path.suffix == ".parquet" else None
    replacement = asdict(describe_file(
        root, path, logical_name=relative,
        row_count=table.num_rows if table is not None else -1,
        schema_hash=(hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest() if table is not None else ""),
    ))
    manifest["artifacts"] = [
        replacement if row["relative_path"] == relative else row
        for row in manifest["artifacts"]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _add_project_language_profile(root: Path) -> dict:
    prompts = root / "prompts"
    prompts.mkdir()
    for name in GraphRAGNativePromptProfileBuilder.required_prompt_names:
        content = (
            "Return output in English with {input_text}."
            if name in {"extract_graph.txt", "extract_claims.txt"}
            else "native prompt"
        )
        (prompts / name).write_text(content, encoding="utf-8")
    result = GraphRAGNativePromptProfileBuilder.apply(
        run_root=root, texts=("这是中文项目正文。" * 100,),
    )
    (root / "settings.yaml").write_text("concurrent_requests: 1\n", encoding="utf-8")
    manifest_path = root / "kf-manifest/run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [
        root / "settings.yaml", result["profile_path"],
        *(root / item["relative_path"] for item in result["prompts"]),
    ]
    manifest["artifacts"].extend(
        asdict(describe_file(
            root, path, logical_name=path.relative_to(root).as_posix(),
        ))
        for path in paths
    )
    manifest["identity"]["prompt_profile_hash"] = result["prompt_profile_hash"]
    manifest["identity"]["output_language"] = "zh"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return result


def test_native_bundle_validator_accepts_complete_independent_bundle(tmp_path):
    _valid_bundle(tmp_path)
    report = GraphRAGNativeBundleValidator().validate(tmp_path)
    assert report["relationships"] == 1
    assert report["lancedb"]["entity_description"]["rows"] == 2


def test_validator_freezes_project_language_profile_and_every_prompt(tmp_path):
    _valid_bundle(tmp_path)
    profile = _add_project_language_profile(tmp_path)

    report = GraphRAGNativeBundleValidator().validate(tmp_path)

    assert report["prompt_profile"]["status"] == "validated"
    assert report["prompt_profile"]["language_code"] == "zh"
    prompt = tmp_path / "prompts/extract_graph.txt"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\ntampered", encoding="utf-8")
    _refresh_manifest_artifact(tmp_path, "prompts/extract_graph.txt")
    with pytest.raises(GraphRAGNativeBundleError, match="prompt 哈希"):
        GraphRAGNativeBundleValidator().validate(tmp_path)
    assert profile["prompt_profile_hash"]


def test_global_query_subset_reads_only_global_parquet_and_never_opens_lancedb(monkeypatch, tmp_path):
    _valid_bundle(tmp_path)
    allowed = {
        "entities.parquet", "communities.parquet", "community_reports.parquet",
    }
    original_read_table = pq.read_table
    reads: list[str] = []

    def guarded_read_table(path, *args, **kwargs):
        name = Path(path).name
        reads.append(name)
        assert name in allowed, f"Global query validation read non-Global table: {name}"
        return original_read_table(path, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    monkeypatch.setattr(lancedb, "connect", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Global must not open LanceDB")))

    report = GraphRAGNativeBundleValidator().validate_global_query_subset(tmp_path)

    assert report["scope"] == "global_query_subset"
    assert report["lancedb"]["status"] == "not_opened_for_global_query"
    assert set(reads) == allowed


def test_native_bundle_validator_rejects_missing_report_and_path_escape(tmp_path):
    _valid_bundle(tmp_path)
    report = tmp_path / "kf-manifest/run-manifest.json"
    manifest = json.loads(report.read_text(encoding="utf-8"))
    manifest["artifacts"] = [row for row in manifest["artifacts"] if row["relative_path"] != "output/community_reports.parquet"]
    report.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(GraphRAGNativeBundleError, match="关键原生 artifact"):
        GraphRAGNativeBundleValidator().validate(tmp_path)
    with pytest.raises(GraphRAGNativeBundleError, match="不安全"):
        safe_bundle_path(tmp_path, "../escape")


def test_standalone_validator_script_accepts_the_same_complete_bundle(tmp_path):
    _valid_bundle(tmp_path)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/validate_graphrag_native_bundle.py", "--root", str(tmp_path)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"relationships": 1' in completed.stdout


def test_bundle_secret_scan_rejects_real_bearer_credential(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "bad.log").write_text("Authorization: Bearer secret-token-value-123456", encoding="utf-8")
    with pytest.raises(GraphRAGNativeBundleError, match="明文凭据"):
        assert_bundle_has_no_plaintext_credentials(tmp_path)


@pytest.mark.parametrize(
    "reference",
    ("$RERANKER_API_KEY", "${RERANKER_API_KEY}", "${REDACTED}"),
)
def test_bundle_secret_scan_allows_bearer_environment_references(tmp_path, reference):
    (tmp_path / "README.md").write_text(
        f'curl -H "Authorization: Bearer {reference}" http://localhost',
        encoding="utf-8",
    )

    assert_bundle_has_no_plaintext_credentials(tmp_path)


def test_candidate_patch_redaction_removes_credential_fixture_and_keeps_diff_marker():
    content = (
        "+safe = True\n"
        '+token = "Authorization: Bearer secret-token-value-123456"\n'
        " context remains\n"
    )

    redacted, count = redact_plaintext_credential_lines(content)

    assert count == 1
    assert "secret-token-value" not in redacted
    assert "+[KF_REDACTED_CREDENTIAL_LIKE_LINE]" in redacted
    assert "+safe = True" in redacted


@pytest.mark.parametrize(
    ("relative", "mutate", "message"),
    [
        (
            "output/relationships.parquet",
            lambda rows: [{**rows[0], "target": "missing entity"}],
            "endpoint",
        ),
        (
            "output/community_reports.parquet",
            lambda rows: [{**rows[0], "summary": "tampered summary"}],
            "JSON 与 Parquet",
        ),
        (
            "output/communities.parquet",
            lambda rows: [{**rows[0], "parent": "missing-community"}],
            "parent",
        ),
        (
            "output/text_units.parquet",
            lambda rows: [rows[0], {**rows[0]}],
            "唯一",
        ),
    ],
)
def test_validator_rejects_semantic_corruption_even_when_manifest_hashes_are_recomputed(tmp_path, relative, mutate, message):
    _valid_bundle(tmp_path)
    path = tmp_path / relative
    rows = pq.read_table(path).to_pylist()
    pq.write_table(pa.Table.from_pylist(mutate(rows)), path)
    _refresh_manifest_artifact(tmp_path, relative)

    with pytest.raises(GraphRAGNativeBundleError, match=message):
        GraphRAGNativeBundleValidator().validate(tmp_path)


def test_validator_rejects_missing_manifest_and_missing_lancedb_vector(tmp_path):
    missing_manifest = tmp_path / "missing-manifest"
    _valid_bundle(missing_manifest)
    (missing_manifest / "kf-manifest/run-manifest.json").unlink()
    with pytest.raises(GraphRAGNativeBundleError, match="run-manifest"):
        GraphRAGNativeBundleValidator().validate(missing_manifest)

    missing_vector = tmp_path / "missing-vector"
    _valid_bundle(missing_vector)
    lancedb.connect(str(missing_vector / "output/lancedb")).open_table("entity_description").delete("id = 'e-1'")
    with pytest.raises(GraphRAGNativeBundleError, match="ID/row count"):
        GraphRAGNativeBundleValidator().validate(missing_vector)
