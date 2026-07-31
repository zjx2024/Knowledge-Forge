"""GraphRAG 3.1.1 settings contract tests."""

from __future__ import annotations

import yaml

import pytest

from knowledge_forge.graphrag_native_config import (
    GraphRAGNativeConfigError,
    GraphRAGNativeSettingsBuilder,
)


def _prompts(root):
    prompts = root / "prompts"
    prompts.mkdir(parents=True)
    for name in (
        "extract_graph.txt", "extract_claims.txt", "summarize_descriptions.txt",
        "community_report_graph.txt", "community_report_text.txt", "local_search_system_prompt.txt",
        "global_search_map_system_prompt.txt", "global_search_reduce_system_prompt.txt",
        "global_search_knowledge_system_prompt.txt", "drift_search_system_prompt.txt",
        "drift_reduce_prompt.txt", "basic_search_system_prompt.txt",
    ):
        (prompts / name).write_text("native prompt", encoding="utf-8")


def test_native_settings_are_typed_standard_and_secret_free(tmp_path):
    _prompts(tmp_path)
    result = GraphRAGNativeSettingsBuilder().write_and_validate(
        run_root=tmp_path, completion_model="qwen3.6:latest", embedding_model="bge-m3:latest",
        embedding_dimension=1024, api_base="http://127.0.0.1:11434/v1",
    )
    settings = yaml.safe_load(result["settings_path"].read_text(encoding="utf-8"))
    redacted = result["redacted_path"].read_text(encoding="utf-8")
    assert "workflows" not in settings
    assert settings["input"]["type"] == "parquet"
    assert settings["cluster_graph"]["use_lcc"] is False
    assert settings["cluster_graph"]["seed"] == 3735928559
    assert settings["concurrent_requests"] == 1
    assert settings["extract_claims"]["enabled"] is False
    assert all(
        item["vector_size"] == 1024
        for item in settings["vector_store"]["index_schema"].values()
    )
    assert settings["input_storage"]["base_dir"] == "input"
    assert settings["output_storage"]["base_dir"] == "output"
    assert settings["table_provider"]["storage"]["base_dir"] == "output"
    assert settings["vector_store"]["db_uri"] == "output/lancedb"
    assert "graphrag-local-placeholder" not in redacted
    assert "${REDACTED}" in redacted


def test_native_settings_refuse_assumed_embedding_dimension(tmp_path):
    with pytest.raises(GraphRAGNativeConfigError, match="实际 probe"):
        GraphRAGNativeSettingsBuilder().build_mapping(
            run_root=tmp_path, completion_model="model", embedding_model="embed",
            embedding_dimension=0, api_base="http://127.0.0.1:11434/v1",
        )
