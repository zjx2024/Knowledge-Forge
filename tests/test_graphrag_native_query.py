"""Offline query protocol and 3.1.1 context-mapping contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from knowledge_forge.graphrag_native_query import (
    GraphRAGNativeQueryError,
    NativeQueryRequest,
    assert_query_response_clean,
    assert_graphrag_query_api_contract,
    load_bundle_query_config,
    query_bundle,
    serialize_context,
)


def _tables():
    entities = pd.DataFrame([{"id": "e1", "human_readable_id": "1", "title": "A"}])
    relationships = pd.DataFrame([{"id": "r1", "human_readable_id": "1", "source": "A", "target": "B"}])
    communities = pd.DataFrame([{"id": "c1", "community": "7"}])
    reports = pd.DataFrame([{"id": "cr1", "community": "7"}])
    text_units = pd.DataFrame([{"id": "tu1", "human_readable_id": "101", "document_id": "doc1"}])
    source_map = pd.DataFrame([{"input_document_id": "doc1", "project_id": 1, "document_id": 2, "revision_id": "r", "block_ids": "[\"b\"]"}])
    return entities, relationships, communities, reports, text_units, source_map


def test_query_worker_locks_actual_graphrag_311_public_api_signature():
    assert_graphrag_query_api_contract()


def test_query_config_rebinds_stale_staging_paths_to_published_bundle(tmp_path):
    import os
    from pathlib import Path

    import yaml

    from knowledge_forge.graphrag_native_config import GraphRAGNativeSettingsBuilder

    stage = tmp_path / ".staging" / "attempt"
    stage.mkdir(parents=True)
    prompts = stage / "prompts"
    prompts.mkdir()
    for name in (
        "local_search_system_prompt.txt",
        "global_search_map_system_prompt.txt",
        "global_search_reduce_system_prompt.txt",
        "global_search_knowledge_system_prompt.txt",
        "drift_search_system_prompt.txt",
        "drift_reduce_prompt.txt",
        "basic_search_system_prompt.txt",
    ):
        (prompts / name).write_text("frozen", encoding="utf-8")
    settings = GraphRAGNativeSettingsBuilder().build_mapping(
        run_root=stage, completion_model="model", embedding_model="embed",
        embedding_dimension=2, api_base="http://127.0.0.1:11434/v1",
    )
    # Model a historical bundle that captured its absolute staging URI.
    settings["vector_store"]["db_uri"] = str(stage / "output" / "lancedb")
    (stage / "settings.yaml").write_text(
        yaml.safe_dump(settings, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    published = tmp_path / "project-1" / "runs" / "run-1"
    published.parent.mkdir(parents=True)
    stage.replace(published)
    previous_key = os.environ.get("GRAPHRAG_API_KEY")
    previous_cwd = Path.cwd()
    os.environ["GRAPHRAG_API_KEY"] = "placeholder"
    try:
        config = load_bundle_query_config(published)
    finally:
        if previous_key is None:
            os.environ.pop("GRAPHRAG_API_KEY", None)
        else:
            os.environ["GRAPHRAG_API_KEY"] = previous_key

    assert Path.cwd() == previous_cwd
    assert Path(config.vector_store.db_uri) == published / "output" / "lancedb"
    assert Path(config.input_storage.base_dir) == published / "input"
    assert Path(config.output_storage.base_dir) == published / "output"
    assert Path(config.table_provider.storage["base_dir"]) == (
        published / "output"
    )
    assert Path(config.local_search.prompt) == (
        published / "prompts" / "local_search_system_prompt.txt"
    )


def test_context_short_ids_resolve_to_one_native_record_and_source_map():
    tables = _tables()
    records = serialize_context(
        {"entities": [{"id": "1"}], "relationships": [{"id": "1"}], "reports": [{"id": "7"}], "Sources": [{"id": "0"}]},
        entities=tables[0], relationships=tables[1], communities=tables[2], reports=tables[3], text_units=tables[4], source_map=tables[5],
    )
    assert [item["native_id"] for item in records] == ["e1", "r1", "cr1", "tu1"]
    assert records[-1]["source_map"]["revision_id"] == "r"


def test_context_mapping_preserves_valid_zero_short_ids():
    entities = pd.DataFrame([
        {"id": "e0", "human_readable_id": 0, "title": "Zero"},
    ])
    relationships = pd.DataFrame([
        {
            "id": "r0", "human_readable_id": 0,
            "source": "Zero", "target": "One",
        },
    ])
    communities = pd.DataFrame([{"id": "c0", "community": 0}])
    reports = pd.DataFrame([{"id": "cr0", "community": 0}])

    records = serialize_context(
        {
            "entities": [{"id": "0"}],
            "relationships": [{"id": "0"}],
            "reports": [{"id": "0"}],
        },
        entities=entities,
        relationships=relationships,
        communities=communities,
        reports=reports,
        text_units=None,
        source_map=None,
    )

    assert [item["native_id"] for item in records] == ["e0", "r0", "cr0"]


def test_disabled_claims_placeholder_may_be_empty_but_never_nonempty():
    tables = _tables()
    assert serialize_context(
        {"claims": []},
        entities=tables[0], relationships=tables[1], communities=tables[2],
        reports=tables[3], text_units=tables[4], source_map=tables[5],
    ) == []
    with pytest.raises(GraphRAGNativeQueryError, match="禁用 claims"):
        serialize_context(
            {"claims": [{"id": "1"}]},
            entities=tables[0], relationships=tables[1],
            communities=tables[2], reports=tables[3],
            text_units=tables[4], source_map=tables[5],
        )


def test_context_mapping_fails_closed_on_zero_or_ambiguous_short_id():
    tables = _tables()
    with pytest.raises(GraphRAGNativeQueryError, match="恰好映射一条"):
        serialize_context({"entities": [{"id": "missing"}]}, entities=tables[0], relationships=tables[1], communities=tables[2], reports=tables[3], text_units=tables[4], source_map=tables[5])
    duplicated = pd.concat([tables[0], tables[0]], ignore_index=True)
    with pytest.raises(GraphRAGNativeQueryError, match="恰好映射一条"):
        serialize_context({"entities": [{"id": "1"}]}, entities=duplicated, relationships=tables[1], communities=tables[2], reports=tables[3], text_units=tables[4], source_map=tables[5])


def test_basic_sources_use_documented_reset_index_position_not_native_id():
    tables = _tables()
    with pytest.raises(GraphRAGNativeQueryError, match="reset-index"):
        serialize_context({"Sources": [{"id": "tu1"}]}, entities=tables[0], relationships=tables[1], communities=tables[2], reports=tables[3], text_units=tables[4], source_map=tables[5])


def test_text_unit_position_accepts_zero_human_readable_id():
    tables = _tables()
    text_units = pd.DataFrame([
        {
            "id": "tu0", "human_readable_id": 0,
            "document_id": "doc1",
        },
    ])
    records = serialize_context(
        {"Sources": [{"id": "0"}]},
        entities=tables[0], relationships=tables[1],
        communities=tables[2], reports=tables[3],
        text_units=text_units, source_map=tables[5],
    )
    assert records[0]["native_id"] == "tu0"


@pytest.mark.parametrize("mode,answer", [
    ("global", "Exception in reduce_response"),
    ("global", "Error parsing search response json"),
    ("drift", "No answer found for query"),
    ("drift", "No follow-up actions found"),
])
def test_query_protocol_rejects_known_swallowed_graphrag_failures(mode, answer):
    with pytest.raises(GraphRAGNativeQueryError):
        assert_query_response_clean(mode=mode, answer=answer, context_data={})


def test_query_protocol_allows_an_explicit_no_data_answer_without_internal_failure_marker():
    assert_query_response_clean(mode="global", answer="No relevant information is available.", context_data={})


def test_global_query_uses_only_global_tables_and_never_calls_full_vector_validator(monkeypatch, tmp_path):
    """Global is a strict Parquet-only query mode, not a hidden Local query."""

    import asyncio
    import knowledge_forge.graphrag_native_query as module
    from tests.test_graphrag_native_artifacts import _valid_bundle

    _valid_bundle(tmp_path)
    original_table = module._table
    reads: list[str] = []

    def guarded_table(root, name):
        reads.append(name)
        assert name in {"entities.parquet", "communities.parquet", "community_reports.parquet"}
        return original_table(root, name)

    async def fake_call(_request, _root):
        return "answer", {"entities": [{"id": "1"}], "reports": [{"id": "1"}]}

    monkeypatch.setattr(module, "_table", guarded_table)
    monkeypatch.setattr(module, "_call_api", fake_call)
    monkeypatch.setattr(module, "assert_graphrag_query_api_contract", lambda: None)
    monkeypatch.setattr(
        module.GraphRAGNativeBundleValidator, "validate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Global must not run full validator/LanceDB path")),
    )
    # The fake call is async only to model the real public API; the worker
    # itself owns the event loop.
    assert asyncio.iscoroutinefunction(fake_call)

    result = query_bundle(NativeQueryRequest(run_id="fake", mode="global", question="what changed?"), tmp_path)

    assert result["mode"] == "global"
    assert [item["native_id"] for item in result["context_records"]] == ["e-1", "cr-1"]
    assert set(reads) == {"entities.parquet", "communities.parquet", "community_reports.parquet"}
