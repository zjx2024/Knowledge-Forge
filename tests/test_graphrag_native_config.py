"""GraphRAG 3.1.1 settings contract tests."""

from __future__ import annotations

import yaml

import pytest

from knowledge_forge.graphrag_native_config import (
    GraphRAGNativeConfigError,
    GraphRAGNativePromptProfileBuilder,
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


def test_native_prompt_profile_follows_project_language_across_all_workflows(tmp_path):
    _prompts(tmp_path)
    (tmp_path / "prompts" / "extract_graph.txt").write_text(
        "Return output in English with {entity_types} and {input_text}.",
        encoding="utf-8",
    )
    (tmp_path / "prompts" / "extract_claims.txt").write_text(
        "Return output in English with {input_text}.", encoding="utf-8",
    )

    result = GraphRAGNativePromptProfileBuilder.apply(
        run_root=tmp_path,
        texts=("这是项目中文正文。" * 300, "English abstract only. " * 20),
    )

    assert result["language"]["language_code"] == "zh"
    assert result["language"]["input_unit_count"] == 2
    extract = (tmp_path / "prompts" / "extract_graph.txt").read_text(
        encoding="utf-8",
    )
    assert "Return output in English" not in extract
    assert "Return output in Chinese" in extract
    assert "{entity_types}" in extract and "{input_text}" in extract
    assert "Do not translate entity names" in extract
    assert "KF_SOURCE_ID" in extract
    for name in GraphRAGNativePromptProfileBuilder.required_prompt_names:
        prompt = (tmp_path / "prompts" / name).read_text(encoding="utf-8")
        assert prompt.count("KF_PROJECT_LANGUAGE_POLICY_START") == 1
    query = (tmp_path / "prompts" / "local_search_system_prompt.txt").read_text(
        encoding="utf-8",
    )
    assert "language used by the user's query" in query
    assert result["profile_path"].is_file()

    # Applying the same frozen project profile is deterministic and does not
    # stack another instruction block.
    repeated = GraphRAGNativePromptProfileBuilder.apply(
        run_root=tmp_path,
        texts=("这是项目中文正文。" * 300, "English abstract only. " * 20),
    )
    assert repeated["prompt_profile_hash"] == result["prompt_profile_hash"]


def test_native_prompt_profile_distinguishes_english_project(tmp_path):
    _prompts(tmp_path)
    result = GraphRAGNativePromptProfileBuilder.apply(
        run_root=tmp_path,
        texts=("This project contains an English technical manual. " * 200,),
    )

    assert result["language"]["language_code"] == "en"
    assert result["language"]["prompt_language"] == "English"


def test_native_prompt_profile_honours_explicit_project_language(tmp_path):
    _prompts(tmp_path)
    result = GraphRAGNativePromptProfileBuilder.apply(
        run_root=tmp_path,
        texts=("This source document is written in English. " * 100,),
        requested_language="zh",
    )

    assert result["language"]["language_code"] == "zh"
    assert result["language"]["detected_language_code"] == "en"
    assert result["language"]["selection_mode"] == "explicit"
    assert "Return output in Chinese" not in (
        tmp_path / "prompts" / "extract_graph.txt"
    ).read_text(encoding="utf-8")
    assert "Write every generated natural-language value in Chinese" in (
        tmp_path / "prompts" / "extract_graph.txt"
    ).read_text(encoding="utf-8")
