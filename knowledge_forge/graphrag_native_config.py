"""Version-locked settings construction for GraphRAG 3.1.1 Native."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import yaml

from .knowledge_language import (
    CHINESE,
    DOCUMENT_LANGUAGE,
    ENGLISH,
    normalize_knowledge_language,
)


class GraphRAGNativeConfigError(RuntimeError):
    pass


class GraphRAGNativePromptProfileBuilder:
    """Freeze one project-wide language policy into every native prompt.

    GraphRAG 3.1.1's generated indexing prompts explicitly request English.
    Native runs instead derive one language from the complete frozen project
    input, then apply it consistently to extraction, summarization, community
    reports, and query generation.  Entity names and machine-readable control
    syntax remain source-faithful.
    """

    version = "graphrag-native-prompt-profile-v2"
    required_prompt_names = (
        "extract_graph.txt",
        "extract_claims.txt",
        "summarize_descriptions.txt",
        "community_report_graph.txt",
        "community_report_text.txt",
        "local_search_system_prompt.txt",
        "global_search_map_system_prompt.txt",
        "global_search_reduce_system_prompt.txt",
        "global_search_knowledge_system_prompt.txt",
        "drift_search_system_prompt.txt",
        "drift_reduce_prompt.txt",
        "basic_search_system_prompt.txt",
    )
    optional_prompt_names = ("question_gen_system_prompt.txt",)
    query_prompt_names = frozenset({
        "local_search_system_prompt.txt",
        "global_search_map_system_prompt.txt",
        "global_search_reduce_system_prompt.txt",
        "global_search_knowledge_system_prompt.txt",
        "drift_search_system_prompt.txt",
        "drift_reduce_prompt.txt",
        "basic_search_system_prompt.txt",
        "question_gen_system_prompt.txt",
    })
    _policy_start = "<!-- KF_PROJECT_LANGUAGE_POLICY_START -->"
    _policy_end = "<!-- KF_PROJECT_LANGUAGE_POLICY_END -->"

    @staticmethod
    def _hash(payload: Any) -> str:
        return hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @classmethod
    def detect_language(cls, texts: list[str] | tuple[str, ...]) -> dict[str, Any]:
        """Infer one deterministic project language from all frozen input.

        Counts use comparable CJK-character and Latin-word information units,
        matching the parser's bilingual-document policy.  The slightly lower
        Han threshold is intentional for Chinese technical documents, where
        API names and identifiers create many Latin tokens without changing
        the prose language.
        """

        cleaned_parts: list[str] = []
        for value in texts:
            text = re.sub(r"\[KF_SOURCE_ID:[^\]\r\n]+\]", "", str(value or ""))
            text = re.sub(r"(?m)^(?:Section|Table|Columns):\s*", "", text)
            if text.strip():
                cleaned_parts.append(text)
        joined = "\n".join(cleaned_parts)
        han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", joined))
        kana = len(re.findall(r"[\u3040-\u30ff]", joined))
        hangul = len(re.findall(r"[\uac00-\ud7af]", joined))
        latin_words = len(re.findall(r"[A-Za-z]+", joined))
        latin = math.ceil(latin_words * 1.3)

        if kana >= 8 and kana + han >= max(1, math.ceil(latin * 0.65)):
            code = "ja"
            prompt_language = "Japanese"
            display_name = "日语"
            selected_units = kana + han
        elif hangul >= 8 and hangul + han >= max(1, math.ceil(latin * 0.65)):
            code = "ko"
            prompt_language = "Korean"
            display_name = "韩语"
            selected_units = hangul + han
        elif han > 0 and han >= max(1, math.ceil(latin * 0.65)):
            code = "zh"
            prompt_language = (
                "Chinese, preserving the source project's Simplified or "
                "Traditional Chinese convention"
            )
            display_name = "中文"
            selected_units = han
        else:
            # Knowledge Forge currently evaluates Chinese and English project
            # corpora.  A Latin-script corpus defaults to English; source
            # names and terminology are still protected from translation by
            # the frozen policy below.
            code = "en"
            prompt_language = "English"
            display_name = "英语"
            selected_units = latin

        total_units = max(1, han + kana + hangul + latin)
        return {
            "schema_version": "graphrag-native-language-profile-v1",
            "language_code": code,
            "prompt_language": prompt_language,
            "display_name": display_name,
            "method": "deterministic-project-script-profile-v1",
            "confidence": round(min(1.0, selected_units / total_units), 6),
            "input_unit_count": len(cleaned_parts),
            "script_units": {
                "han": han, "kana": kana, "hangul": hangul,
                "latin": latin, "latin_words": latin_words,
            },
            "source_text_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        }

    @classmethod
    def resolve_language(
        cls,
        texts: list[str] | tuple[str, ...],
        requested_language: str = DOCUMENT_LANGUAGE,
    ) -> dict[str, Any]:
        """Resolve the persisted project preference against frozen input."""

        requested = normalize_knowledge_language(requested_language)
        detected = cls.detect_language(texts)
        if requested == DOCUMENT_LANGUAGE:
            return {
                **detected,
                "requested_language": requested,
                "selection_mode": "document_language",
                "detected_language_code": str(detected["language_code"]),
            }
        configured = {
            CHINESE: (
                "Chinese, preserving the source project's Simplified or "
                "Traditional Chinese convention",
                "中文",
            ),
            ENGLISH: ("English", "英语"),
        }
        prompt_language, display_name = configured[requested]
        return {
            **detected,
            "language_code": requested,
            "prompt_language": prompt_language,
            "display_name": display_name,
            "method": "explicit-project-knowledge-language-v1",
            "confidence": 1.0,
            "requested_language": requested,
            "selection_mode": "explicit",
            "detected_language_code": str(detected["language_code"]),
        }

    @classmethod
    def _language_policy(cls, *, prompt_name: str, prompt_language: str) -> str:
        query_rule = (
            "Answer in the language used by the user's query. If that is "
            f"ambiguous, use {prompt_language}."
            if prompt_name in cls.query_prompt_names
            else f"Write every generated natural-language value in {prompt_language}."
        )
        source_marker_rule = (
            "- A [KF_SOURCE_ID:...] marker is provenance metadata only; never "
            "emit it as an entity, relationship endpoint, type, or claim."
            if prompt_name in {"extract_graph.txt", "extract_claims.txt"}
            else ""
        )
        return (
            f"\n\n{cls._policy_start}\n"
            "Knowledge Forge project language policy (this overrides the language "
            "used by examples above):\n"
            f"- {query_rule}\n"
            "- Do not translate entity names, proper nouns, standard abbreviations, "
            "identifiers, code symbols, formulas, or quoted source text; preserve their "
            "source spelling.\n"
            "- Language rules apply only to natural-language values. Preserve JSON keys, "
            "tuple delimiters, completion markers, citation syntax, record IDs, and all "
            "other machine-readable control tokens exactly.\n"
            f"{source_marker_rule + chr(10) if source_marker_rule else ''}"
            f"{cls._policy_end}\n"
        )

    @staticmethod
    def _place_policy(*, prompt_name: str, content: str, policy: str) -> str:
        """Keep indexing ``Output:`` as the final generation boundary."""

        anchors = {
            "extract_graph.txt": "-Real Data-",
            "extract_claims.txt": "-Real Data-",
            "summarize_descriptions.txt": "-Data-",
            "community_report_graph.txt": "# Real Data",
            "community_report_text.txt": "# Real Data",
        }
        anchor = anchors.get(prompt_name)
        index = content.rfind(anchor) if anchor else -1
        if index < 0:
            return content.rstrip() + policy
        return (
            content[:index].rstrip()
            + policy
            + "\n"
            + content[index:].lstrip("\n")
        )

    @classmethod
    def apply(
        cls,
        *,
        run_root: Path,
        texts: list[str] | tuple[str, ...],
        requested_language: str = DOCUMENT_LANGUAGE,
    ) -> dict[str, Any]:
        root = Path(run_root).resolve()
        prompt_root = root / "prompts"
        if not prompt_root.is_dir() or prompt_root.is_symlink():
            raise GraphRAGNativeConfigError("GraphRAG init 未生成安全的 prompts 目录")
        missing = [
            name for name in cls.required_prompt_names
            if not (prompt_root / name).is_file() or (prompt_root / name).is_symlink()
        ]
        if missing:
            raise GraphRAGNativeConfigError(
                "GraphRAG init 缺少必需 prompt: " + ", ".join(missing)
            )

        language = cls.resolve_language(texts, requested_language)
        prompt_language = str(language["prompt_language"])
        names = list(cls.required_prompt_names) + [
            name for name in cls.optional_prompt_names
            if (prompt_root / name).is_file() and not (prompt_root / name).is_symlink()
        ]
        prompt_files: list[dict[str, Any]] = []
        existing_policy = re.compile(
            re.escape(cls._policy_start) + r".*?" + re.escape(cls._policy_end) + r"\n?",
            flags=re.DOTALL,
        )
        for name in names:
            path = prompt_root / name
            content = existing_policy.sub("", path.read_text(encoding="utf-8")).rstrip()
            if name in {"extract_graph.txt", "extract_claims.txt"}:
                content = content.replace(
                    "Return output in English",
                    f"Return output in {prompt_language}",
                )
            policy = cls._language_policy(
                prompt_name=name, prompt_language=prompt_language,
            )
            content = cls._place_policy(
                prompt_name=name, content=content, policy=policy,
            )
            path.write_text(content, encoding="utf-8")
            prompt_files.append({
                "relative_path": f"prompts/{name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })

        profile = {
            "schema_version": cls.version,
            "language": language,
            "prompts": sorted(prompt_files, key=lambda item: item["relative_path"]),
            "source_metadata_rule": (
                "KF_SOURCE_ID is provenance metadata and never a graph item"
            ),
        }
        profile_hash = cls._hash(profile)
        profile["prompt_profile_hash"] = profile_hash
        manifest_root = root / "kf-manifest"
        manifest_root.mkdir(parents=True, exist_ok=True)
        profile_path = manifest_root / "prompt-profile.json"
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return {
            "prompt_profile_hash": profile_hash,
            "profile_path": profile_path,
            "language": language,
            "prompts": profile["prompts"],
        }


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
