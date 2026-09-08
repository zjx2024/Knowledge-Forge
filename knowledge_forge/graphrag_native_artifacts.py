"""Immutable GraphRAG Native bundle manifests and offline integrity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import re
from typing import Any

import pyarrow.parquet as pq
import yaml


class GraphRAGNativeBundleError(RuntimeError):
    """A native bundle is incomplete, unsafe, or internally inconsistent."""


FINAL_TABLE_COLUMNS = {
    "documents.parquet": {"id", "human_readable_id", "title", "text", "text_unit_ids", "creation_date", "raw_data"},
    "text_units.parquet": {"id", "human_readable_id", "text", "n_tokens", "document_id", "entity_ids", "relationship_ids", "covariate_ids"},
    "entities.parquet": {"id", "human_readable_id", "title", "type", "description", "text_unit_ids", "frequency", "degree"},
    "relationships.parquet": {"id", "human_readable_id", "source", "target", "description", "weight", "combined_degree", "text_unit_ids"},
    "communities.parquet": {"id", "human_readable_id", "community", "level", "parent", "children", "title", "entity_ids", "relationship_ids", "text_unit_ids", "period", "size"},
    "community_reports.parquet": {"id", "human_readable_id", "community", "level", "parent", "children", "title", "summary", "full_content", "rank", "rating_explanation", "findings", "full_content_json", "period", "size"},
}
RAW_TABLE_COLUMNS = {
    "raw_entities.parquet": {"title", "type"},
    "raw_relationships.parquet": {"source", "target"},
}
STANDARD_WORKFLOWS = (
    "load_input_documents", "create_base_text_units", "create_final_documents", "extract_graph",
    "finalize_graph", "extract_covariates", "create_communities", "create_final_text_units",
    "create_community_reports", "generate_text_embeddings",
)


PLAINTEXT_CREDENTIAL_PATTERNS = (
    re.compile(
        r"(?i)authorization\s*[:=]\s*bearer\s+"
        r"(?!\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})(?:\s|['\"]|$))"
        r"(?!\$\{REDACTED\})\S+"
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(
        r"(?i)(api[_-]?key|password)\s*[:=]\s*['\"]"
        r"(?!\$\{)(?:sk-|[A-Za-z0-9_./-]{16,})"
    ),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_bundle_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    value = str(relative_path or "").replace("\\", "/")
    if not value or value.startswith("/") or any(part in {"", ".", ".."} for part in value.split("/")):
        raise GraphRAGNativeBundleError(f"不安全的 bundle 相对路径: {relative_path!r}")
    candidate = (root / value).resolve(strict=False)
    if root not in candidate.parents:
        raise GraphRAGNativeBundleError(f"bundle 路径逃逸: {relative_path!r}")
    return candidate


def assert_bundle_has_no_plaintext_credentials(root: Path) -> None:
    """Reject recognizable credentials while allowing `${VAR}` configuration."""

    root = Path(root).resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            continue
        if path.suffix.lower() in {".parquet", ".lance", ".zst", ".tar"}:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(content) for pattern in PLAINTEXT_CREDENTIAL_PATTERNS):
            raise GraphRAGNativeBundleError(f"bundle 检测到疑似明文凭据: {path.relative_to(root)}")


def redact_plaintext_credential_lines(content: str) -> tuple[str, int]:
    """Redact credential-like patch lines while preserving their diff marker.

    Reproducibility patches can legitimately include security-test fixtures or
    an accidental local secret.  The raw patch hash remains part of the run
    identity, but the bundle must never retain such line contents.
    """

    redacted: list[str] = []
    count = 0
    for line in str(content).splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if any(pattern.search(body) for pattern in PLAINTEXT_CREDENTIAL_PATTERNS):
            marker = body[:1] if body[:1] in {"+", "-", " "} else ""
            redacted.append(marker + "[KF_REDACTED_CREDENTIAL_LIKE_LINE]" + ending)
            count += 1
        else:
            redacted.append(line)
    return "".join(redacted), count


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return parsed if isinstance(parsed, list) else [value]
    return list(value) if hasattr(value, "tolist") else [value]


@dataclass(frozen=True)
class NativeArtifact:
    logical_name: str
    relative_path: str
    sha256: str
    byte_size: int
    media_type: str
    schema_hash: str = ""
    row_count: int = -1


def describe_file(root: Path, path: Path, *, logical_name: str, row_count: int = -1, schema_hash: str = "") -> NativeArtifact:
    root = root.resolve()
    path = path.resolve(strict=True)
    if root not in path.parents:
        raise GraphRAGNativeBundleError("artifact 不属于 run root")
    return NativeArtifact(
        logical_name=logical_name,
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path), byte_size=path.stat().st_size,
        media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        schema_hash=schema_hash, row_count=row_count,
    )


class GraphRAGNativeBundleValidator:
    """Validate a bundle from files only; it never reads PostgreSQL summaries."""

    validator_version = "graphrag-native-validator-v1"

    @staticmethod
    def _validate_prompt_profile(
        root: Path,
        manifest: dict[str, Any],
        artifact_by_path: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Validate the frozen project-language profile for new bundles.

        Historical 3.1.1 bundles predate project-language profiles and remain
        readable.  Once a run identity declares ``prompt_profile_hash``, every
        configured prompt and the settings file must be immutable manifest
        artifacts and the profile becomes fail-closed.
        """

        identity = manifest.get("identity") if isinstance(manifest.get("identity"), dict) else {}
        expected_hash = str(identity.get("prompt_profile_hash") or "")
        if not expected_hash:
            return {"status": "legacy_unprofiled", "language_code": ""}
        profile_relative = "kf-manifest/prompt-profile.json"
        profile_artifact = artifact_by_path.get(profile_relative)
        if profile_artifact is None:
            raise GraphRAGNativeBundleError("项目语言提示配置未写入 immutable artifact manifest")
        profile_path = GraphRAGNativeBundleValidator._validate_manifest_artifact(
            root, profile_artifact,
        )
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GraphRAGNativeBundleError("项目语言提示配置不是合法 JSON") from exc
        if not isinstance(profile, dict) or profile.get("schema_version") != "graphrag-native-prompt-profile-v2":
            raise GraphRAGNativeBundleError("项目语言提示配置版本不受支持")
        embedded_hash = str(profile.pop("prompt_profile_hash", "") or "")
        actual_hash = hashlib.sha256(canonical_json(profile).encode("utf-8")).hexdigest()
        if not embedded_hash or embedded_hash != expected_hash or actual_hash != expected_hash:
            raise GraphRAGNativeBundleError("项目语言提示配置哈希不一致")
        language = profile.get("language")
        if not isinstance(language, dict) or str(language.get("language_code") or "") not in {"zh", "en", "ja", "ko"}:
            raise GraphRAGNativeBundleError("项目语言提示配置缺少受支持的冻结语言")
        prompts = profile.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise GraphRAGNativeBundleError("项目语言提示配置没有 prompts")
        seen: set[str] = set()
        for item in prompts:
            if not isinstance(item, dict):
                raise GraphRAGNativeBundleError("项目语言 prompt 记录必须为对象")
            relative = str(item.get("relative_path") or "")
            if (
                not relative.startswith("prompts/")
                or relative.count("/") != 1
                or relative in seen
            ):
                raise GraphRAGNativeBundleError("项目语言 prompt 路径非法或重复")
            seen.add(relative)
            artifact = artifact_by_path.get(relative)
            if artifact is None:
                raise GraphRAGNativeBundleError(f"项目语言 prompt 未冻结为 artifact: {relative}")
            path = GraphRAGNativeBundleValidator._validate_manifest_artifact(root, artifact)
            if str(item.get("sha256") or "") != sha256_file(path):
                raise GraphRAGNativeBundleError(f"项目语言 prompt 哈希不一致: {relative}")
        settings_artifact = artifact_by_path.get("settings.yaml")
        if settings_artifact is None:
            raise GraphRAGNativeBundleError("项目语言运行缺少冻结的 settings.yaml")
        GraphRAGNativeBundleValidator._validate_manifest_artifact(root, settings_artifact)
        return {
            "status": "validated",
            "language_code": str(language["language_code"]),
            "prompt_count": len(prompts),
            "prompt_profile_hash": expected_hash,
        }

    @staticmethod
    def _validate_manifest_artifact(root: Path, artifact: dict[str, Any]) -> Path:
        """Check one immutable artifact without broadening a query's inputs."""

        relative = str(artifact.get("relative_path") or "")
        path = safe_bundle_path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise GraphRAGNativeBundleError(f"artifact 缺失或不是普通文件: {relative}")
        if int(artifact.get("byte_size") or -1) != path.stat().st_size:
            raise GraphRAGNativeBundleError(f"artifact byte size 不一致: {relative}")
        if str(artifact.get("sha256") or "") != sha256_file(path):
            raise GraphRAGNativeBundleError(f"artifact SHA-256 不一致: {relative}")
        return path

    def validate_global_query_subset(self, root: Path) -> dict[str, Any]:
        """Validate exactly the Parquet inputs permitted to Global search.

        A READY run has already passed ``validate`` before activation.  Query
        time rechecks the manifest identity and hashes/schema of the three
        Global tables, without reading TextUnits, relationships, source maps
        or LanceDB.  This keeps the mode's data-dependency contract explicit.
        """

        root = Path(root).resolve()
        if not root.is_dir() or root.is_symlink():
            raise GraphRAGNativeBundleError("bundle root 不存在或不安全")
        manifest = self._read_manifest(root)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise GraphRAGNativeBundleError("run manifest 没有 artifacts")
        by_path: dict[str, dict[str, Any]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise GraphRAGNativeBundleError("artifact manifest row 非对象")
            relative = str(artifact.get("relative_path") or "")
            if not relative or relative in by_path:
                raise GraphRAGNativeBundleError("artifact manifest path 必须唯一")
            by_path[relative] = artifact
        prompt_profile = self._validate_prompt_profile(root, manifest, by_path)
        table_names = ("entities.parquet", "communities.parquet", "community_reports.parquet")
        tables: dict[str, list[dict[str, Any]]] = {}
        for name in table_names:
            relative = "output/" + name
            artifact = by_path.get(relative)
            if artifact is None:
                raise GraphRAGNativeBundleError(f"Global query 缺少原生表: {relative}")
            path = self._validate_manifest_artifact(root, artifact)
            rows, schema_hash = self._read_table(path, FINAL_TABLE_COLUMNS[name])
            if artifact.get("schema_hash") not in {"", schema_hash}:
                raise GraphRAGNativeBundleError(f"Parquet schema hash 不一致: {name}")
            self._unique(rows, "id", name)
            tables[name] = rows
        communities = {str(row.get("community") or "") for row in tables["communities.parquet"]}
        reports = {str(row.get("community") or "") for row in tables["community_reports.parquet"]}
        if not communities or communities != reports:
            raise GraphRAGNativeBundleError("Global query communities 与 reports 不一一对应")
        for row in tables["community_reports.parquet"]:
            if not str(row.get("full_content") or "").strip():
                raise GraphRAGNativeBundleError("Global query community report full_content 不能为空")
        return {
            "validator_version": self.validator_version,
            "scope": "global_query_subset",
            "manifest_sha256": sha256_file(root / "kf-manifest" / "run-manifest.json"),
            "entities": len(tables["entities.parquet"]),
            "communities": len(tables["communities.parquet"]),
            "community_reports": len(tables["community_reports.parquet"]),
            "lancedb": {"status": "not_opened_for_global_query"},
            "prompt_profile": prompt_profile,
        }

    def _read_manifest(self, root: Path) -> dict[str, Any]:
        path = root / "kf-manifest" / "run-manifest.json"
        if not path.is_file() or path.is_symlink():
            raise GraphRAGNativeBundleError("缺少安全的 kf-manifest/run-manifest.json")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GraphRAGNativeBundleError(f"run manifest 无法读取: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != "graphrag-native-bundle-v1":
            raise GraphRAGNativeBundleError("run manifest schema_version 不受支持")
        if str(manifest.get("graphrag_version") or "") != "3.1.1":
            raise GraphRAGNativeBundleError("bundle GraphRAG 版本不是 3.1.1")
        return manifest

    @staticmethod
    def _read_table(path: Path, required: set[str]) -> tuple[list[dict[str, Any]], str]:
        try:
            table = pq.read_table(path)
        except Exception as exc:
            raise GraphRAGNativeBundleError(f"Parquet 无法完整读取: {path.name}: {exc}") from exc
        columns = set(table.column_names)
        absent = sorted(required - columns)
        if absent:
            raise GraphRAGNativeBundleError(f"{path.name} 缺少必需列: {', '.join(absent)}")
        schema_hash = hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest()
        return table.to_pylist(), schema_hash

    @staticmethod
    def _unique(rows: list[dict[str, Any]], key: str, table_name: str) -> set[str]:
        values = [str(row.get(key) or "") for row in rows]
        if not values or any(not value for value in values) or len(values) != len(set(values)):
            raise GraphRAGNativeBundleError(f"{table_name} 的 {key} 必须非空且唯一")
        return set(values)

    @staticmethod
    def _validate_workflow_ledger(root: Path, manifest: dict[str, Any]) -> None:
        relative = str(manifest.get("workflow_ledger_path") or "kf-manifest/workflow-ledger.jsonl")
        path = safe_bundle_path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise GraphRAGNativeBundleError("workflow ledger 缺失")
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        statuses = {str(row.get("workflow") or ""): str(row.get("status") or "") for row in rows}
        if set(statuses) != set(STANDARD_WORKFLOWS):
            raise GraphRAGNativeBundleError("Standard workflow ledger 不完整")
        for workflow, status in statuses.items():
            if workflow == "extract_covariates":
                if status not in {"completed", "disabled"}:
                    raise GraphRAGNativeBundleError("claims/covariates ledger 状态非法")
            elif status != "completed":
                raise GraphRAGNativeBundleError(f"workflow 未完成: {workflow}={status}")

    @staticmethod
    def _validate_logs(root: Path) -> None:
        log_root = root / "logs"
        if not log_root.exists():
            raise GraphRAGNativeBundleError("native logs 缺失")
        for path in sorted(log_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            content = path.read_text(encoding="utf-8", errors="replace").lower()
            if any(marker in content for marker in (
                "error extracting graph", "no report found for community", "error processing community",
            )):
                raise GraphRAGNativeBundleError(f"native log 指示静默 GraphRAG 错误: {path.name}")

    @staticmethod
    def _validate_lancedb(root: Path, manifest: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        settings_path = safe_bundle_path(root, str(manifest.get("settings_redacted_path") or "settings.redacted.yaml"))
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        vector_store = settings.get("vector_store") if isinstance(settings, dict) else {}
        schemas = vector_store.get("index_schema") if isinstance(vector_store, dict) else {}
        dimension = int((vector_store or {}).get("vector_size") or 0)
        if dimension <= 0 or not isinstance(schemas, dict):
            raise GraphRAGNativeBundleError("LanceDB settings 契约不完整")
        try:
            import lancedb
        except ImportError as exc:
            raise GraphRAGNativeBundleError("LanceDB validator 依赖不可用") from exc
        db_path = safe_bundle_path(root, str(manifest.get("lancedb_relative_path") or "output/lancedb"))
        if not db_path.is_dir() or db_path.is_symlink():
            raise GraphRAGNativeBundleError("native LanceDB 目录缺失或不安全")
        db = lancedb.connect(str(db_path))
        target_to_rows = {
            "text_unit_text": tables["text_units.parquet"],
            "entity_description": tables["entities.parquet"],
            "community_full_content": tables["community_reports.parquet"],
        }
        report: dict[str, Any] = {}
        for logical, expected_rows in target_to_rows.items():
            schema = schemas.get(logical)
            if not isinstance(schema, dict) or not str(schema.get("index_name") or ""):
                raise GraphRAGNativeBundleError(f"LanceDB logical target 未在 settings 声明: {logical}")
            physical = str(schema["index_name"])
            if int(schema.get("vector_size") or 0) != dimension:
                raise GraphRAGNativeBundleError(f"LanceDB target dimension 不一致: {logical}")
            try:
                table = db.open_table(physical)
                rows = table.to_arrow().to_pylist()
            except Exception as exc:
                raise GraphRAGNativeBundleError(f"LanceDB 表不可读: {physical}: {exc}") from exc
            actual_ids = {str(row.get("id") or "") for row in rows}
            expected_ids = {str(row.get("id") or "") for row in expected_rows}
            if actual_ids != expected_ids or len(rows) != len(expected_rows):
                raise GraphRAGNativeBundleError(f"LanceDB ID/row count 与 {logical} 事实源不一致")
            for row in rows:
                vector = _list_value(row.get("vector"))
                if len(vector) != dimension or any(not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in vector):
                    raise GraphRAGNativeBundleError(f"LanceDB 向量维度或有限性非法: {logical}")
            report[logical] = {"physical": physical, "rows": len(rows), "dimension": dimension}
        return report

    def validate(self, root: Path, *, validate_lancedb: bool = True) -> dict[str, Any]:
        """Validate an immutable bundle without consulting application state.

        ``global`` GraphRAG search is intentionally a Parquet-only operation.
        It still needs the full file/provenance/community integrity checks, but
        must not open LanceDB just because a caller is asking a Global-only
        question.  All other callers keep the stricter, complete validation by
        default.
        """
        root = Path(root).resolve()
        if not root.is_dir() or root.is_symlink():
            raise GraphRAGNativeBundleError("bundle root 不存在或不安全")
        assert_bundle_has_no_plaintext_credentials(root)
        manifest = self._read_manifest(root)
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise GraphRAGNativeBundleError("run manifest 没有 artifacts")
        seen_names: set[str] = set()
        seen_paths: set[str] = set()
        artifact_by_path: dict[str, dict[str, Any]] = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise GraphRAGNativeBundleError("artifact manifest row 非对象")
            name = str(artifact.get("logical_name") or "")
            relative = str(artifact.get("relative_path") or "")
            if not name or name in seen_names or relative in seen_paths:
                raise GraphRAGNativeBundleError("artifact logical_name/path 必须唯一")
            seen_names.add(name); seen_paths.add(relative)
            path = safe_bundle_path(root, relative)
            if not path.is_file() or path.is_symlink():
                raise GraphRAGNativeBundleError(f"artifact 缺失或不是普通文件: {relative}")
            if int(artifact.get("byte_size") or -1) != path.stat().st_size:
                raise GraphRAGNativeBundleError(f"artifact byte size 不一致: {relative}")
            if str(artifact.get("sha256") or "") != sha256_file(path):
                raise GraphRAGNativeBundleError(f"artifact SHA-256 不一致: {relative}")
            if path.suffix == ".parquet":
                table = pq.read_table(path)
                schema_hash = hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest()
                expected_schema = str(artifact.get("schema_hash") or "")
                expected_rows = int(artifact.get("row_count", -1))
                if expected_schema and expected_schema != schema_hash:
                    raise GraphRAGNativeBundleError(f"artifact Parquet schema hash 不一致: {relative}")
                if expected_rows >= 0 and expected_rows != table.num_rows:
                    raise GraphRAGNativeBundleError(f"artifact Parquet row count 不一致: {relative}")
            artifact_by_path[relative] = artifact
        prompt_profile = self._validate_prompt_profile(
            root, manifest, artifact_by_path,
        )
        required_paths = {
            "input/documents.parquet", "kf-manifest/source-map.parquet",
            "kf-manifest/input-documents-manifest.jsonl",
        }
        required_paths.update("output/" + name for name in FINAL_TABLE_COLUMNS)
        required_paths.update("output/" + name for name in RAW_TABLE_COLUMNS)
        required_paths.add("output/graph.graphml")
        missing = sorted(required_paths - set(artifact_by_path))
        if missing:
            raise GraphRAGNativeBundleError("manifest 缺少关键原生 artifact: " + ", ".join(missing))
        tables: dict[str, list[dict[str, Any]]] = {}
        for name, columns in FINAL_TABLE_COLUMNS.items():
            rows, schema_hash = self._read_table(root / "output" / name, columns)
            if artifact_by_path["output/" + name].get("schema_hash") not in {"", schema_hash}:
                raise GraphRAGNativeBundleError(f"Parquet schema hash 不一致: {name}")
            tables[name] = rows
        raw_tables: dict[str, list[dict[str, Any]]] = {}
        for name, columns in RAW_TABLE_COLUMNS.items():
            raw_tables[name], _ = self._read_table(root / "output" / name, columns)
        input_rows, _ = self._read_table(root / "input" / "documents.parquet", {"id", "title", "text", "raw_data"})
        source_map_rows, _ = self._read_table(root / "kf-manifest" / "source-map.parquet", {
            "project_id", "document_id", "revision_id", "source_hash", "input_document_id", "input_title", "block_ids",
        })
        input_ids = self._unique(input_rows, "id", "input/documents.parquet")
        output_document_ids = self._unique(tables["documents.parquet"], "id", "documents.parquet")
        if input_ids != output_document_ids:
            raise GraphRAGNativeBundleError("input Documents 与 output documents ID 集合不一致")
        if {str(row.get("input_document_id") or "") for row in source_map_rows} != input_ids:
            raise GraphRAGNativeBundleError("source map 未完整覆盖 native input Documents")
        text_unit_ids = self._unique(tables["text_units.parquet"], "id", "text_units.parquet")
        entity_ids = self._unique(tables["entities.parquet"], "id", "entities.parquet")
        relationship_ids = self._unique(tables["relationships.parquet"], "id", "relationships.parquet")
        community_ids = self._unique(tables["communities.parquet"], "id", "communities.parquet")
        report_ids = self._unique(tables["community_reports.parquet"], "id", "community_reports.parquet")
        del entity_ids, relationship_ids, community_ids, report_ids
        for row in tables["text_units.parquet"]:
            if str(row.get("document_id") or "") not in output_document_ids:
                raise GraphRAGNativeBundleError("TextUnit 指向不存在 document")
        entity_title_rows: dict[str, list[dict[str, Any]]] = {}
        for row in tables["entities.parquet"]:
            entity_title_rows.setdefault(str(row.get("title") or ""), []).append(row)
        if any(not title or len(rows) != 1 for title, rows in entity_title_rows.items()):
            raise GraphRAGNativeBundleError("final entity title 无法唯一解析；拒绝猜测 GraphRAG title merge")
        entity_titles = set(entity_title_rows)
        for table_name in ("entities.parquet", "relationships.parquet", "communities.parquet"):
            for row in tables[table_name]:
                for text_unit_id in _list_value(row.get("text_unit_ids")):
                    if str(text_unit_id) not in text_unit_ids:
                        raise GraphRAGNativeBundleError(f"{table_name} 指向不存在 TextUnit")
        for row in tables["relationships.parquet"]:
            if str(row.get("source") or "") not in entity_titles or str(row.get("target") or "") not in entity_titles:
                raise GraphRAGNativeBundleError("semantic relationship endpoint 无法解析为同 run entity title")
        communities = {str(row.get("community")) for row in tables["communities.parquet"]}
        reports = {str(row.get("community")) for row in tables["community_reports.parquet"]}
        if communities != reports:
            raise GraphRAGNativeBundleError("communities 与 community_reports 不一一对应")
        for row in tables["community_reports.parquet"]:
            if not str(row.get("full_content") or "").strip():
                raise GraphRAGNativeBundleError("community report full_content 不能为空")
            try:
                full_content_json = json.loads(str(row.get("full_content_json") or ""))
            except json.JSONDecodeError as exc:
                raise GraphRAGNativeBundleError("community report full_content_json 非法") from exc
            if not isinstance(full_content_json, dict):
                raise GraphRAGNativeBundleError("community report full_content_json 必须为对象")
            expected = {
                "title": str(row.get("title") or ""),
                "summary": str(row.get("summary") or ""),
                "findings": _list_value(row.get("findings")),
                "rating": row.get("rank"),
                "rating_explanation": str(row.get("rating_explanation") or ""),
            }
            if (
                str(full_content_json.get("title") or "") != expected["title"]
                or str(full_content_json.get("summary") or "") != expected["summary"]
                or _list_value(full_content_json.get("findings")) != expected["findings"]
                or full_content_json.get("rating") != expected["rating"]
                or str(full_content_json.get("rating_explanation") or "") != expected["rating_explanation"]
            ):
                raise GraphRAGNativeBundleError("community report JSON 与 Parquet 字段不一致")
        entity_ids = {str(row.get("id") or "") for row in tables["entities.parquet"]}
        relationship_ids = {str(row.get("id") or "") for row in tables["relationships.parquet"]}
        for row in tables["communities.parquet"]:
            for entity_id in _list_value(row.get("entity_ids")):
                if str(entity_id) not in entity_ids:
                    raise GraphRAGNativeBundleError("community 指向不存在 entity")
            for relationship_id in _list_value(row.get("relationship_ids")):
                if str(relationship_id) not in relationship_ids:
                    raise GraphRAGNativeBundleError("community 指向不存在 relationship")
            parent = str(row.get("parent") or "")
            if parent and parent not in {"-1", "None", "null"} and parent not in communities:
                raise GraphRAGNativeBundleError("community parent 无法解析")
            for child in _list_value(row.get("children")):
                if str(child) not in communities:
                    raise GraphRAGNativeBundleError("community child 无法解析")
                child_row = next(
                    (item for item in tables["communities.parquet"] if str(item.get("community") or "") == str(child)),
                    None,
                )
                if not child_row or str(child_row.get("parent") or "") != str(row.get("community") or ""):
                    raise GraphRAGNativeBundleError("community parent/child 层级不闭合")
        if len({(str(row.get("title") or ""), str(row.get("type") or "")) for row in raw_tables["raw_entities.parquet"]}) != len(raw_tables["raw_entities.parquet"]):
            raise GraphRAGNativeBundleError("raw_entities 聚合键重复")
        if len({(str(row.get("source") or ""), str(row.get("target") or "")) for row in raw_tables["raw_relationships.parquet"]}) != len(raw_tables["raw_relationships.parquet"]):
            raise GraphRAGNativeBundleError("raw_relationships 聚合键重复")
        self._validate_workflow_ledger(root, manifest)
        self._validate_logs(root)
        lance_report = (
            self._validate_lancedb(root, manifest, tables)
            if validate_lancedb
            else {"status": "not_opened_for_global_query"}
        )
        return {
            "validator_version": self.validator_version,
            "manifest_sha256": sha256_file(root / "kf-manifest" / "run-manifest.json"),
            "input_documents": len(input_rows), "text_units": len(tables["text_units.parquet"]),
            "entities": len(tables["entities.parquet"]), "relationships": len(tables["relationships.parquet"]),
            "communities": len(tables["communities.parquet"]), "community_reports": len(tables["community_reports.parquet"]),
            "lancedb": lance_report, "prompt_profile": prompt_profile,
        }


def write_run_manifest(root: Path, *, identity: dict[str, Any], artifacts: list[NativeArtifact], **values: Any) -> Path:
    """Write the manifest after all immutable output files are finalized."""

    root = Path(root).resolve()
    path = root / "kf-manifest" / "run-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "graphrag-native-bundle-v1",
        "graphrag_version": "3.1.1",
        "identity": identity,
        "artifacts": [asdict(item) for item in artifacts],
        **values,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return path
