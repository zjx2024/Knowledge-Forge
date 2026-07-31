"""Version-locked settings construction for GraphRAG 3.1.1 Native."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


class GraphRAGNativeConfigError(RuntimeError):
    pass


class GraphRAGNativeSettingsBuilder:
    """Patch a freshly generated 3.1.1 ``graphrag init`` settings file safely."""

    version = "graphrag-native-settings-v1"
    logical_vector_targets = (
        "text_unit_text", "entity_description", "community_full_content",
    )

    @staticmethod
    def _hash(payload: Any) -> str:
        return hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @staticmethod
    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ("${REDACTED}" if "key" in key.lower() or "secret" in key.lower() or "password" in key.lower() else GraphRAGNativeSettingsBuilder._redact(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [GraphRAGNativeSettingsBuilder._redact(item) for item in value]
        return value

    def build_mapping(
        self,
        *,
        run_root: Path,
        completion_model: str,
        embedding_model: str,
        embedding_dimension: int,
        api_base: str,
    ) -> dict[str, Any]:
        if embedding_dimension <= 0:
            raise GraphRAGNativeConfigError("embedding dimension 必须先经实际 probe 确认")
        root = Path(run_root).resolve()
        api_base = str(api_base).rstrip("/")
        if not api_base:
            raise GraphRAGNativeConfigError("GraphRAG provider endpoint 不能为空")
        # These are current GraphRAG 3.1.1 keys, validated again by load_config
        # after being written.  No workflows key is emitted: Standard remains
        # entirely upstream-owned.
        return {
            "completion_models": {
                "default_completion_model": {
                    "model_provider": "openai", "model": completion_model,
                    "api_base": api_base, "auth_method": "api_key",
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "retry": {"type": "exponential_backoff"},
                },
            },
            "embedding_models": {
                "default_embedding_model": {
                    "model_provider": "openai", "model": embedding_model,
                    "api_base": api_base, "auth_method": "api_key",
                    "api_key": "${GRAPHRAG_API_KEY}",
                    "retry": {"type": "exponential_backoff"},
                },
            },
            "concurrent_requests": 1,
            "input": {
                # GraphRAG's config loader uses string.Template before YAML
                # parsing, so the regex end anchor must be escaped as ``$$``.
                "type": "parquet", "file_pattern": "documents\\.parquet$$",
                "id_column": "id", "title_column": "title", "text_column": "text",
            },
            # Bundle-local paths must remain relocatable.  The isolated CLI
            # always runs with ``run_root`` as cwd, and GraphRAG resolves these
            # paths while loading the configuration.  Persisting the staging
            # absolute path would leave a READY bundle pointing at a directory
            # that no longer exists after atomic publication.
            "input_storage": {"type": "file", "base_dir": "input"},
            "output_storage": {"type": "file", "base_dir": "output"},
            "update_output_storage": {"type": "file", "base_dir": "output-update"},
            "reporting": {"type": "file", "base_dir": "logs"},
            "cache": {"type": "json", "storage": {"type": "file", "base_dir": "cache"}},
            "chunking": {"type": "tokens", "size": 900, "overlap": 100, "encoding_model": "o200k_base", "prepend_metadata": ["id"]},
            "table_provider": {"type": "parquet", "storage": {"type": "file", "base_dir": "output"}},
            "vector_store": {
                "type": "lancedb", "db_uri": "output/lancedb",
                "vector_size": embedding_dimension,
                "index_schema": {
                    target: {"index_name": target, "vector_size": embedding_dimension}
                    for target in self.logical_vector_targets
                },
            },
            "extract_graph": {
                "completion_model_id": "default_completion_model",
                "prompt": "prompts/extract_graph.txt", "max_gleanings": 1,
            },
            "summarize_descriptions": {
                "completion_model_id": "default_completion_model",
                "prompt": "prompts/summarize_descriptions.txt", "max_length": 500,
            },
            "extract_claims": {"enabled": False, "completion_model_id": "default_completion_model", "prompt": "prompts/extract_claims.txt"},
            "cluster_graph": {"max_cluster_size": 10, "use_lcc": False, "seed": 3735928559},
            "community_reports": {
                "completion_model_id": "default_completion_model",
                "graph_prompt": "prompts/community_report_graph.txt",
                "text_prompt": "prompts/community_report_text.txt",
            },
            "snapshots": {"raw_graph": True, "graphml": True, "embeddings": False},
            "embed_text": {"embedding_model_id": "default_embedding_model"},
            "local_search": {"completion_model_id": "default_completion_model", "embedding_model_id": "default_embedding_model", "prompt": "prompts/local_search_system_prompt.txt"},
            "global_search": {"completion_model_id": "default_completion_model", "map_prompt": "prompts/global_search_map_system_prompt.txt", "reduce_prompt": "prompts/global_search_reduce_system_prompt.txt", "knowledge_prompt": "prompts/global_search_knowledge_system_prompt.txt"},
            "drift_search": {
                "completion_model_id": "default_completion_model", "embedding_model_id": "default_embedding_model",
                "prompt": "prompts/drift_search_system_prompt.txt", "reduce_prompt": "prompts/drift_reduce_prompt.txt",
                "primer_folds": 1, "drift_k_followups": 1,
            },
            "basic_search": {"completion_model_id": "default_completion_model", "embedding_model_id": "default_embedding_model", "prompt": "prompts/basic_search_system_prompt.txt"},
        }

    def write_and_validate(
        self,
        *,
        run_root: Path,
        completion_model: str,
        embedding_model: str,
        embedding_dimension: int,
        api_base: str,
    ) -> dict[str, Any]:
        root = Path(run_root).resolve()
        settings = self.build_mapping(
            run_root=root, completion_model=completion_model, embedding_model=embedding_model,
            embedding_dimension=embedding_dimension, api_base=api_base,
        )
        if "workflows" in settings:
            raise GraphRAGNativeConfigError("native settings 不得覆盖 GraphRAG Standard workflows")
        settings_path = root / "settings.yaml"
        redacted_path = root / "settings.redacted.yaml"
        settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding="utf-8")
        redacted_path.write_text(yaml.safe_dump(self._redact(settings), allow_unicode=True, sort_keys=False), encoding="utf-8")
        from graphrag.config.load_config import load_config
        # GraphRAG's LiteLLM adapter requires a non-empty API key even for a
        # local OpenAI-compatible endpoint.  Keep the placeholder in process
        # environment only; it is never written into either settings file.
        previous_key = os.environ.get("GRAPHRAG_API_KEY")
        previous_cost_map = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
        previous_cwd = Path.cwd()
        if not previous_key:
            os.environ["GRAPHRAG_API_KEY"] = "graphrag-local-placeholder"
        if previous_cost_map is None:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        try:
            # GraphRAG resolves bundle-relative storage paths against cwd.
            # Validate under the exact cwd used by the isolated CLI.
            os.chdir(root)
            config = load_config(root)
        finally:
            # GraphRAG 3.1.1's config loading can change process cwd.  The
            # web worker and test runner are long-lived, so restore it before
            # any subsequent relative-path operation.
            os.chdir(previous_cwd)
            if previous_key is None:
                os.environ.pop("GRAPHRAG_API_KEY", None)
            if previous_cost_map is None:
                os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        if str(config.input.type) != "parquet" or config.cluster_graph.use_lcc:
            raise GraphRAGNativeConfigError("GraphRAG 3.1.1 resolved config 与 native 契约不一致")
        if int(config.concurrent_requests) != 1 or int(config.vector_store.vector_size) != embedding_dimension:
            raise GraphRAGNativeConfigError("GraphRAG concurrency/vector dimension 未按 native 契约解析")
        return {
            "settings_path": settings_path,
            "redacted_path": redacted_path,
            "settings_hash": self._hash(settings),
            "resolved_vector_targets": {
                key: value.index_name for key, value in config.vector_store.index_schema.items()
                if key in self.logical_vector_targets
            },
        }
