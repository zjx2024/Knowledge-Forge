"""Project-wide, isolated Microsoft GraphRAG 3.1.1 executor."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from dataclasses import asdict
from typing import Any, Protocol

import pyarrow.parquet as pq

from .config import Config
from .database import JobCancellationRequested
from .graphrag_native_artifacts import (
    GraphRAGNativeBundleValidator,
    describe_file,
    redact_plaintext_credential_lines,
    safe_bundle_path,
    write_run_manifest,
)
from .graphrag_native_config import (
    GraphRAGNativePromptProfileBuilder,
    GraphRAGNativeSettingsBuilder,
)
from .graphrag_native_inputs import GraphRAGNativeInputBuilder, NativeInputPreflight
from .graphrag_native_runner import (
    GraphRAGNativeProviderDoctor,
    GraphRAGNativeRunner,
    NativeProcessProgress,
)


class GraphRAGNativeExecutorProtocol(Protocol):
    def run(self, *, job_id: str, project_id: int) -> None: ...


class NativeNoGraphUnsupported(RuntimeError):
    """Non-empty native input produced no semantic graph in Standard mode."""


class NativeBuildCancelled(RuntimeError):
    """The durable job requested cancellation before bundle publication."""


class NativeStandardWorkflowProgress:
    """Monotonic progress for GraphRAG's ordered Standard workflow.

    GraphRAG reports an independent ``completed/total`` denominator for each
    internal operation.  Treating any one of those denominators as the whole
    Standard pipeline makes the first short operation move the public job to
    the end of its execution band.  Keep the upstream signals intact, but
    place them inside stable workflow milestones for the pinned 3.1.1
    Standard pipeline.

    The weights are observability-only.  They never select a workflow, skip a
    model call, alter GraphRAG settings, or participate in bundle validation.
    Expensive model-backed extraction and reporting deliberately own most of
    the band; deterministic table transforms own only small milestones.
    """

    lower_bound = 30
    upper_bound = 71
    workflow_weights: tuple[tuple[str, float], ...] = (
        ("create_base_text_units", 0.03),
        ("create_final_documents", 0.01),
        ("extract_graph", 0.43),
        ("finalize_graph", 0.04),
        ("extract_covariates", 0.01),
        ("create_communities", 0.05),
        ("create_final_text_units", 0.03),
        ("create_community_reports", 0.30),
        ("generate_text_embeddings", 0.10),
    )
    workflow_labels = {
        "create_base_text_units": "生成文本单元",
        "create_final_documents": "整理文档索引",
        "extract_graph": "抽取实体与关系",
        "finalize_graph": "整理实体关系图",
        "extract_covariates": "整理声明与属性",
        "create_communities": "识别主题社区",
        "create_final_text_units": "关联文本与图谱",
        "create_community_reports": "生成社区报告",
        "generate_text_embeddings": "生成检索向量",
    }
    operation_labels = {
        "chunker": "生成文本单元",
        "extract_graph": "抽取实体与关系",
        "relationship_description": "生成实体与关系描述",
        "generate_embeddings": "生成检索向量",
    }

    def __init__(self, previous: int = lower_bound) -> None:
        self.previous = max(
            self.lower_bound, min(self.upper_bound, int(previous)),
        )
        self.active_workflow = ""
        self.fractions = {name: 0.0 for name, _weight in self.workflow_weights}
        self.operation_fractions: dict[tuple[str, str], float] = {}

    @staticmethod
    def _normalize_workflow(value: str) -> str:
        return "_".join(str(value or "").strip().lower().replace("-", " ").split())

    def _known_workflow(self, value: str) -> str:
        normalized = self._normalize_workflow(value)
        return normalized if normalized in self.fractions else ""

    def _event_workflow(self, event: NativeProcessProgress) -> str:
        direct = self._known_workflow(event.workflow)
        if direct:
            return direct
        # Progress records often name an inner operation (for example
        # ``chunker``) rather than the enclosing workflow.  Attribute that
        # denominator only to the workflow whose explicit start event was
        # observed; never guess from an arbitrary operation name.
        return self.active_workflow

    def weighted_fraction(self) -> float:
        return max(0.0, min(1.0, sum(
            self.fractions[name] * weight
            for name, weight in self.workflow_weights
        )))

    def _operation_fraction(self, workflow: str) -> float:
        """Aggregate known independent counters inside one workflow.

        GraphRAG reuses ``n/total`` for multiple operations.  In particular,
        ``extract_graph`` first reports text-unit extraction and then a much
        larger description-summarization ledger; community reports separately
        report each hierarchy level.  These counters must not overwrite one
        another or individually claim the complete enclosing workflow.
        """

        def value(operation: str) -> float:
            return self.operation_fractions.get((workflow, operation), 0.0)
        if workflow == "create_base_text_units":
            return value("chunker")
        if workflow == "extract_graph":
            # Fresh local runs show description summarization dominating the
            # second half of elapsed time.  Reserve it explicitly so a
            # completed text-unit extraction ledger cannot claim that graph
            # extraction itself is complete.
            return 0.25 * value("extract_graph") + 0.75 * value(
                "relationship_description"
            )
        if workflow == "create_community_reports":
            # GraphRAG emits higher hierarchy levels first and level 0 last.
            # A geometric allocation works for any hierarchy depth without
            # hard-coding the number of communities in a particular project.
            progress = 0.0
            prefix = "level_"
            suffix = "_summarize_communities"
            for (owner, operation), fraction in self.operation_fractions.items():
                if owner != workflow or not (
                    operation.startswith(prefix) and operation.endswith(suffix)
                ):
                    continue
                level_text = operation[len(prefix):-len(suffix)]
                if not level_text.isdigit():
                    continue
                level = int(level_text)
                progress += 0.65 * (0.35 ** level) * fraction
            return min(1.0, progress)
        # ``generate embeddings`` restarts its denominator for every batch
        # and field.  Without a trustworthy aggregate denominator, hold this
        # short workflow at its milestone until GraphRAG emits completion.
        return 0.0

    def update(self, event: NativeProcessProgress) -> int:
        state = str(event.workflow_state or "")
        direct = self._known_workflow(event.workflow)
        if state == "started":
            self.active_workflow = direct
        elif state == "completed":
            if direct:
                self.fractions[direct] = 1.0
                if self.active_workflow == direct:
                    self.active_workflow = ""
        elif (
            event.completed is not None
            and event.total is not None
            and int(event.total) > 0
        ):
            workflow = self._event_workflow(event)
            if workflow:
                operation = self._normalize_workflow(event.workflow)
                fraction = max(0.0, min(
                    1.0,
                    float(event.completed) / float(event.total),
                ))
                self.operation_fractions[(workflow, operation)] = max(
                    self.operation_fractions.get((workflow, operation), 0.0),
                    fraction,
                )
                self.fractions[workflow] = max(
                    self.fractions[workflow],
                    self._operation_fraction(workflow),
                )
        value = self.lower_bound + round(
            (self.upper_bound - self.lower_bound) * self.weighted_fraction(),
        )
        self.previous = max(
            self.previous, min(self.upper_bound, int(value)),
        )
        return self.previous

    def metadata(self, event: NativeProcessProgress) -> dict[str, Any]:
        workflow = self._event_workflow(event)
        operation = self._normalize_workflow(event.workflow)
        level_match = (
            operation.startswith("level_")
            and operation.endswith("_summarize_communities")
        )
        if level_match:
            level = operation[len("level_"):-len("_summarize_communities")]
            label = f"生成第 {level} 层社区报告"
        else:
            label = self.operation_labels.get(
                operation,
                self.workflow_labels.get(workflow, "GraphRAG 标准流程"),
            )
        if (
            event.completed is not None
            and event.total is not None
            and int(event.total) > 0
        ):
            detail = f"{label} {int(event.completed)}/{int(event.total)}"
        elif str(event.workflow_state or "") == "completed" and workflow:
            detail = f"{label}已完成"
        else:
            detail = f"正在{label}" if workflow else "GraphRAG 标准流程正在运行"
        workflow_fraction = self.fractions.get(workflow, 0.0)
        operation_fraction = self.operation_fractions.get(
            (workflow, operation), 0.0,
        )
        return {
            "schema_version": "graphrag-native-progress-v2",
            "phase": "standard_workflow",
            "workflow": workflow,
            "operation": operation,
            "detail": detail,
            "workflow_progress": round(workflow_fraction * 100),
            "operation_progress": round(operation_fraction * 100),
            "execution_progress": round(self.weighted_fraction() * 100),
            "overall_progress": self.previous,
        }


class GraphRAGNativeExecutor:
    """Explicit project-wide executor for the third construction strategy."""

    # A Standard workflow can legitimately run for far longer than one
    # database lease.  Refresh well before the 15-minute DB TTL; progress
    # callbacks include a 15-second heartbeat even when GraphRAG has no new
    # structural line, so this is independent of document shape and model
    # verbosity.
    _LEASE_TTL_SECONDS = 900
    _LEASE_RENEW_INTERVAL_SECONDS = 60

    def __init__(self, service: Any):
        self.service = service
        self.runner = GraphRAGNativeRunner(
            timeout_seconds=Config.graphrag_native_timeout_seconds,
        )
        self.settings = GraphRAGNativeSettingsBuilder()
        self.validator = GraphRAGNativeBundleValidator()

    @staticmethod
    def _sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def _run_relative_path(project_id: int, run_id: str) -> str:
        return f"project-{int(project_id)}/runs/{run_id}"

    @staticmethod
    def _staging_relative_path(run_id: str, attempt_id: str) -> str:
        return f".staging/{run_id}-{attempt_id}"

    @property
    def native_root(self) -> Path:
        return (Config.files_root / "graphrag_native").resolve()

    @staticmethod
    def _git_output(root: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=str(root), check=True, capture_output=True, text=True,
        )
        return completed.stdout

    def _candidate_identity(self) -> dict[str, str]:
        """Bind a run to the exact checked-out candidate, including untracked code."""

        repo = Path(__file__).resolve().parents[1]
        # A bundle is bound to the checked-out commit plus its local changes.
        # Depending on a historical remote feature branch made clean builds
        # fail after that branch was renamed or deleted.
        head = self._git_output(repo, "rev-parse", "HEAD").strip()
        base_sha = head
        diff = self._git_output(repo, "diff", "--binary", head).encode("utf-8")
        untracked = self._git_output(repo, "ls-files", "--others", "--exclude-standard").splitlines()
        untracked_manifest: list[dict[str, Any]] = []
        for relative in sorted(untracked):
            path = repo / relative
            if path.is_file() and not path.is_symlink():
                untracked_manifest.append({
                    "path": relative, "mode": path.stat().st_mode & 0o777,
                    "size": path.stat().st_size,
                    "sha256": self._sha256_bytes(path.read_bytes()),
                })
        tracked_paths = [
            "knowledge_forge/graphrag_native_executor.py",
            "knowledge_forge/graphrag_native_inputs.py",
            "knowledge_forge/graphrag_native_config.py",
            "knowledge_forge/graphrag_native_runner.py",
            "knowledge_forge/graphrag_native_artifacts.py",
            "knowledge_forge/database.py", "knowledge_forge/migrations.py",
            "knowledge_forge/service.py", "requirements.txt",
        ]
        source_manifest = []
        for relative in tracked_paths:
            path = repo / relative
            if path.is_file():
                source_manifest.append({"path": relative, "sha256": self._sha256_bytes(path.read_bytes())})
        return {
            "base_sha": base_sha,
            "head": head,
            "worktree_diff_hash": self._sha256_bytes(diff + json.dumps(untracked_manifest, sort_keys=True).encode("utf-8")),
            "code_tree_hash": self._sha256_bytes(json.dumps(source_manifest, sort_keys=True).encode("utf-8")),
        }

    @staticmethod
    def _provider_api_base() -> str:
        base = Config.ollama_base_url.rstrip("/")
        return base if base.endswith("/v1") else base + "/v1"

    def _provider_identity(self, project: dict[str, Any]) -> tuple[dict[str, str], int]:
        completion_model = str(project.get("llm_model") or Config.default_llm_model)
        embedding_model = str(project.get("embedding_model") or Config.default_embedding_model)
        completion = self.service.ollama.model_info(completion_model)
        embedding = self.service.ollama.model_info(embedding_model)
        api_key = os.getenv("GRAPHRAG_API_KEY", "graphrag-local-placeholder")
        probe = GraphRAGNativeProviderDoctor.probe_embedding(
            api_base=self._provider_api_base(), model=embedding_model, api_key=api_key,
            timeout=min(60, Config.ollama_timeout),
        )
        return {
            "completion_model_id": completion_model,
            "completion_model_digest": str(completion.get("digest") or "unknown"),
            "embedding_model_id": embedding_model,
            "embedding_model_digest": str(embedding.get("digest") or "unknown"),
        }, int(probe["dimension"])

    @staticmethod
    def _job_cancel_requested(job: dict[str, Any] | None) -> bool:
        job = job or {}
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        return bool(
            str(job.get("status") or "") == "cancelled"
            or metadata.get("cancel_requested")
        )

    def _cancellation_requested(self, job_id: str) -> bool:
        return self._job_cancel_requested(self.service.db.get_job(job_id))

    def _assert_not_cancelled(self, job_id: str) -> None:
        job = self.service.db.get_job(job_id) or {}
        if self._job_cancel_requested(job):
            metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
            raise NativeBuildCancelled(
                str(metadata.get("cancel_reason") or "GraphRAG Native 构建已取消")
            )

    def _discard_cancelled_bundle_path(self, path: Path) -> None:
        """Remove only this generated run's staging/final directory."""

        root = self.native_root
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("拒绝清理 native 根目录之外的取消产物") from exc
        if not relative.parts or resolved == root:
            raise RuntimeError("拒绝清理 GraphRAG Native 根目录")
        if path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _remove_generated_env_file(root: Path) -> None:
        """Do not retain an ``init``-generated env file in an immutable bundle.

        GraphRAG receives the API credential only through this worker's process
        environment.  Keeping the generated ``.env`` would make it too easy to
        accidentally preserve a credential in a staged or exported run.
        """

        path = root / ".env"
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("GraphRAG init 生成了不安全的 .env 路径")
            path.unlink()

    def _seed_compatible_graph_cache(
        self,
        root: Path,
        *,
        project_id: int,
        current_run: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Reuse only GraphRAG's content-addressed call cache after a late failure.

        A complete Standard workflow can still fail closed during independent
        bundle validation.  Re-running every model call after fixing such a
        validator issue adds hours without changing graph quality.  A failed
        run's cache is safe to seed when all algorithm inputs, prompt/settings
        identities and model digests match.  GraphRAG still executes its full
        upstream workflow and regenerates every output table; no prior graph
        artifact is copied or published.
        """

        list_runs = getattr(self.service.db, "list_graphrag_native_runs", None)
        if not callable(list_runs):
            return None
        identity_fields = (
            "source_snapshot_hash",
            "settings_hash",
            "prompt_profile_hash",
            "completion_model_id",
            "completion_model_digest",
            "embedding_model_id",
            "embedding_model_digest",
            "embedding_dimension",
            "graphrag_version",
        )
        candidates = list_runs(int(project_id))
        for candidate in candidates:
            if str(candidate.get("run_id") or "") == str(current_run.get("run_id") or ""):
                continue
            if (
                str(candidate.get("state") or "") != "FAILED"
                or str(candidate.get("failure_class") or "") != "GraphRAGNativeBundleError"
                or not candidate.get("execution_completed_at")
                or any(
                    str(candidate.get(field) or "") != str(current_run.get(field) or "")
                    for field in identity_fields
                )
            ):
                continue
            relative = str(candidate.get("staging_relative_path") or "")
            if not relative:
                continue
            try:
                source_root = safe_bundle_path(self.native_root, relative)
                source_cache = source_root / "cache"
                if (
                    not source_cache.is_dir()
                    or source_cache.is_symlink()
                    or source_root.resolve() == Path(root).resolve()
                ):
                    continue
                entries = sorted(source_cache.rglob("*"))
                if any(path.is_symlink() for path in entries):
                    continue
                files = [path for path in entries if path.is_file()]
                if not files:
                    continue
                destination = Path(root) / "cache"
                destination.mkdir(parents=True, exist_ok=True)
                byte_size = 0
                for source in files:
                    relative_file = source.relative_to(source_cache)
                    target = destination / relative_file
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    byte_size += source.stat().st_size
                evidence = {
                    "schema_version": "graphrag-native-cache-reuse-v1",
                    "source_run_id": str(candidate.get("run_id") or ""),
                    "reason": "compatible_failed_run_after_execution_complete",
                    "file_count": len(files),
                    "byte_size": byte_size,
                    "identity_fields": list(identity_fields),
                }
                manifest = Path(root) / "kf-manifest" / "cache-reuse.json"
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(
                    json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
                return evidence
            except (OSError, RuntimeError, ValueError):
                # Cache reuse is an optimization only.  A malformed or
                # unreadable previous cache must never weaken a fresh build.
                continue
        return None

    @staticmethod
    def _collect_artifacts(root: Path) -> list[Any]:
        artifacts: list[Any] = []
        relatives = [
            "input/documents.parquet", "kf-manifest/source-map.parquet",
            "kf-manifest/input-documents-manifest.jsonl", "settings.yaml",
            "settings.redacted.yaml", "kf-manifest/prompt-profile.json",
            "kf-manifest/workflow-ledger.jsonl", "output/documents.parquet",
            "output/text_units.parquet", "output/entities.parquet", "output/relationships.parquet",
            "output/raw_entities.parquet", "output/raw_relationships.parquet",
            "output/communities.parquet", "output/community_reports.parquet", "output/graph.graphml",
        ]
        profile_path = root / "kf-manifest" / "prompt-profile.json"
        if not profile_path.is_file() or profile_path.is_symlink():
            raise RuntimeError("GraphRAG Native 缺少冻结的项目语言提示配置")
        try:
            prompt_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("GraphRAG Native 项目语言提示配置不是合法 JSON") from exc
        for item in prompt_profile.get("prompts") or []:
            relative = str(item.get("relative_path") or "") if isinstance(item, dict) else ""
            if not relative.startswith("prompts/") or relative.count("/") != 1:
                raise RuntimeError("GraphRAG Native 项目语言提示配置包含非法路径")
            relatives.append(relative)
        for relative in relatives:
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"GraphRAG Native 缺少必需原生 artifact: {relative}")
            schema_hash = ""
            row_count = -1
            if path.suffix == ".parquet":
                table = pq.read_table(path)
                schema_hash = hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest()
                row_count = table.num_rows
            artifacts.append(describe_file(root, path, logical_name=relative, row_count=row_count, schema_hash=schema_hash))
        return artifacts

    def _write_candidate_evidence(
        self, root: Path, *, identity: dict[str, str], prompt_hash: str,
        prompt_profile: dict[str, Any] | None = None,
    ) -> None:
        """Freeze reproducible code/runtime evidence without exporting secrets."""

        manifest_root = root / "kf-manifest"
        manifest_root.mkdir(parents=True, exist_ok=True)
        repo = Path(__file__).resolve().parents[1]
        source_paths = [
            "knowledge_forge/graphrag_native_executor.py", "knowledge_forge/graphrag_native_inputs.py",
            "knowledge_forge/graphrag_native_config.py", "knowledge_forge/graphrag_native_runner.py",
            "knowledge_forge/graphrag_native_artifacts.py", "knowledge_forge/graphrag_native_query.py",
            "knowledge_forge/graphrag_native_runtime.py", "knowledge_forge/database.py",
            "knowledge_forge/migrations.py", "knowledge_forge/service.py", "knowledge_forge/web.py",
            "requirements.txt", "constraints.txt",
        ]
        source_manifest = []
        for relative in source_paths:
            path = repo / relative
            if path.is_file() and not path.is_symlink():
                source_manifest.append({
                    "path": relative, "sha256": self._sha256_bytes(path.read_bytes()),
                    "size": path.stat().st_size,
                })
        (manifest_root / "candidate-identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8",
        )
        (manifest_root / "candidate-source-manifest.json").write_text(
            json.dumps(source_manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8",
        )
        raw_patch = self._git_output(repo, "diff", "--binary", identity["base_sha"])
        patch, redacted_lines = redact_plaintext_credential_lines(raw_patch)
        patch_path = manifest_root / "candidate-worktree.patch"
        patch_path.write_text(patch, encoding="utf-8")
        (manifest_root / "candidate-worktree-redaction.json").write_text(
            json.dumps({
                "schema_version": "graphrag-native-worktree-redaction-v1",
                "raw_patch_sha256": self._sha256_bytes(raw_patch.encode("utf-8")),
                "stored_patch_sha256": self._sha256_bytes(patch.encode("utf-8")),
                "redacted_line_count": redacted_lines,
                "policy": "credential-like lines are omitted; raw content is never stored",
            }, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        archive_tar = manifest_root / "candidate-source-archive.tar"
        with tarfile.open(archive_tar, "w") as archive:
            for item in source_manifest:
                archive.add(repo / item["path"], arcname=item["path"], recursive=False)
        zstd = shutil.which("zstd")
        if not zstd:
            raise RuntimeError("GraphRAG Native 证据包需要 zstd 生成 candidate-source-archive.tar.zst")
        subprocess.run([zstd, "-q", "-f", str(archive_tar), "-o", str(manifest_root / "candidate-source-archive.tar.zst")], check=True)
        archive_tar.unlink()
        (manifest_root / "runtime-environment.json").write_text(json.dumps({
            "python": sys.version, "executable": sys.executable, "graphrag_version": "3.1.1",
        }, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        prompt_profile = prompt_profile or {}
        prompt_records = prompt_profile.get("prompts") or []
        extract_graph_hash = next((
            str(item.get("sha256") or "")
            for item in prompt_records if isinstance(item, dict)
            and item.get("relative_path") == "prompts/extract_graph.txt"
        ), "")
        (manifest_root / "prompt-manifest.json").write_text(json.dumps({
            "prompt_profile_sha256": prompt_hash,
            "extract_graph_prompt_sha256": extract_graph_hash or prompt_hash,
            "output_language": (prompt_profile.get("language") or {}).get("language_code", ""),
            "source_metadata_rule": "KF_SOURCE_ID is provenance metadata and never a graph item",
        }, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "model-metrics.jsonl").write_text(json.dumps({
            "capability": "per_call_instrumentation", "status": "unavailable",
            "reason": "GraphRAG 3.1.1 CLI has no configured run-local provider audit proxy",
        }, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def _write_post_index_evidence(root: Path, *, validation_report: dict[str, Any], migration_report: dict[str, Any]) -> None:
        manifest_root = root / "kf-manifest"
        raw_entities = pq.read_table(root / "output" / "raw_entities.parquet").to_pylist()
        title_counts: dict[str, int] = {}
        for row in raw_entities:
            title = str(row.get("title") or "")
            title_counts[title] = title_counts.get(title, 0) + 1
        (manifest_root / "title-collision-audit.json").write_text(json.dumps({
            "raw_titles_with_multiple_records": {title: count for title, count in title_counts.items() if count > 1},
            "note": "GraphRAG native aggregation is preserved; no legacy canonical/alias mapping is applied.",
        }, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        try:
            import lancedb
            database = lancedb.connect(str(root / "output" / "lancedb"))
            lancedb_manifest = {"tables": sorted(database.table_names())}
        except Exception as exc:  # validator will fail before READY if unusable
            lancedb_manifest = {"error": type(exc).__name__}
        (manifest_root / "lancedb-manifest.json").write_text(json.dumps(lancedb_manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "query-smoke.jsonl").write_text(json.dumps({"status": "not_run", "reason": "live model query is separately authorized"}, ensure_ascii=False) + "\n", encoding="utf-8")
        (manifest_root / "migration-report.json").write_text(json.dumps(migration_report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "native-isolation.json").write_text(json.dumps({
            "legacy_entry_invocations": 0,
            "forbidden": ["HugeGraph", "pgvector", "stable_entity_relation", "knowledge_forge", "graphrag_community"],
            "evidence": "native executor uses only persisted structures, GraphRAG CLI, control-plane DB, and bundle files",
        }, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "stable-knowledge-forge-invariance.json").write_text(json.dumps({
            "native_writes": ["graphrag_native_* control-plane tables only"],
            "stable_knowledge_forge_projection_mutation": 0,
        }, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "metric-summary.json").write_text(json.dumps(validation_report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "validation-report.json").write_text(json.dumps(validation_report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "audit-report.md").write_text(
            "# GraphRAG Native bundle audit\n\nIndependent validator completed before READY. "
            "This bundle contains GraphRAG-native algorithm facts only; it was not projected to HugeGraph or old relation tables.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_workflow_ledger(root: Path, index_result: Any) -> None:
        # GraphRAG CLI does not expose durable per-workflow statuses.  The
        # runner records a complete invocation; the independent validator then
        # enforces all expected output surfaces before READY.  A live executor
        # writes this ledger only after the standard CLI has succeeded.
        workflows = (
            "load_input_documents", "create_base_text_units", "create_final_documents", "extract_graph",
            "finalize_graph", "extract_covariates", "create_communities", "create_final_text_units",
            "create_community_reports", "generate_text_embeddings",
        )
        path = root / "kf-manifest" / "workflow-ledger.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"workflow": workflow, "status": "disabled" if workflow == "extract_covariates" else "completed",
             "runner_exit": int(index_result.returncode)}
            for workflow in workflows
        ]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def _require_nonempty_semantic_graph(root: Path) -> None:
        relationship_path = root / "output" / "relationships.parquet"
        entity_path = root / "output" / "entities.parquet"
        if not relationship_path.is_file() or not entity_path.is_file():
            raise NativeNoGraphUnsupported("GraphRAG Standard 未产生完整语义图输出")
        if pq.read_table(entity_path).num_rows == 0 or pq.read_table(relationship_path).num_rows == 0:
            raise NativeNoGraphUnsupported("非空输入未形成可聚类的原生语义关系图")

    @staticmethod
    def _reject_source_metadata_pollution(root: Path) -> None:
        """A provenance marker emitted as graph data invalidates the run."""

        for name in ("entities.parquet", "relationships.parquet", "raw_entities.parquet", "raw_relationships.parquet"):
            path = root / "output" / name
            if not path.is_file():
                continue
            for row in pq.read_table(path).to_pylist():
                if "[KF_SOURCE_ID:" in json.dumps(row, ensure_ascii=False):
                    raise RuntimeError("GraphRAG 将 KF_SOURCE_ID provenance metadata 抽成了图数据")

    def _record_empty_input(
        self,
        *,
        job_id: str,
        project_id: int,
        project: dict[str, Any],
        run_id: str,
        owner: str,
        attempt_id: str,
        preflight: NativeInputPreflight,
    ) -> None:
        """Persist the non-queryable EMPTY_INPUT terminal state without GraphRAG."""

        identity = self._candidate_identity()
        completion_model = str(project.get("llm_model") or Config.default_llm_model)
        embedding_model = str(project.get("embedding_model") or Config.default_embedding_model)
        marker = hashlib.sha256(
            ("empty-input|" + preflight.source_snapshot_hash + "|" + preflight.reason).encode("utf-8"),
        ).hexdigest()
        run = self.service.db.create_graphrag_native_run({
            "run_id": run_id, "project_id": project_id, **identity,
            "source_snapshot_hash": preflight.source_snapshot_hash,
            "settings_hash": marker, "prompt_profile_hash": marker,
            "completion_model_id": completion_model, "completion_model_digest": "not_probed_empty_input",
            "embedding_model_id": embedding_model, "embedding_model_digest": "not_probed_empty_input",
            "embedding_dimension": 0, "graphrag_version": "3.1.1",
            "python_version": sys.version.split()[0],
            "staging_relative_path": self._staging_relative_path(run_id, attempt_id),
        })
        self.service.db.acquire_graphrag_native_run_lease(str(run["run_id"]), owner)
        self.service.db.update_graphrag_native_run(
            str(run["run_id"]), owner=owner, state="EMPTY_INPUT",
            failure_class="NativeInputError", failure_message=preflight.reason,
        )
        release = getattr(self.service.db, "release_graphrag_native_run_lease", None)
        if release:
            release(str(run["run_id"]), owner)
        message = "GraphRAG Native 输入为空或不可渲染，未调用 GraphRAG/模型：" + preflight.reason
        self.service.db.update_job(job_id, status="failed", progress=100, log=message, error=message)

    def run(self, *, job_id: str, project_id: int) -> None:
        """Run a fresh Standard workflow without KF or HugeGraph fallback."""

        run_id = "grn_" + uuid.uuid4().hex
        attempt_id = uuid.uuid4().hex
        owner = "native-worker-" + uuid.uuid4().hex
        stage_root = self.native_root / self._staging_relative_path(run_id, attempt_id)
        final_root = self.native_root / self._run_relative_path(project_id, run_id)
        run: dict[str, Any] | None = None
        activated = False
        try:
            self._assert_not_cancelled(job_id)
            self.service.db.assert_graphrag_native_ready()
            project = self.service.db.get_project(project_id)
            if not project:
                raise LookupError("项目不存在")
            job = self.service.db.get_job(job_id) or {}
            job_metadata = (
                job.get("metadata")
                if isinstance(job.get("metadata"), dict) else {}
            )
            requested_knowledge_language = str(
                job_metadata.get("knowledge_language")
                or project.get("knowledge_language")
                or "auto"
            )
            if self.service.db.latest_active_job(project_id, "ingestion"):
                raise RuntimeError("源文档正在变化，拒绝创建 GraphRAG Native 快照")
            preflight = GraphRAGNativeInputBuilder(self.service).preflight(project_id=project_id)
            self._assert_not_cancelled(job_id)
            if not preflight.ready:
                self._record_empty_input(
                    job_id=job_id, project_id=project_id, project=project, run_id=run_id,
                    owner=owner, attempt_id=attempt_id, preflight=preflight,
                )
                return
            if stage_root.exists() or final_root.exists():
                raise RuntimeError("GraphRAG Native run 目录冲突，拒绝覆盖历史 artifact")
            stage_root.mkdir(parents=True, exist_ok=False)
            self.service.db.update_job(job_id, status="running", progress=2, log="GraphRAG Native：初始化隔离 staging run")
            provider_environment = {"GRAPHRAG_API_KEY": os.getenv("GRAPHRAG_API_KEY", "graphrag-local-placeholder")}
            models = {
                "completion": str(project.get("llm_model") or Config.default_llm_model),
                "embedding": str(project.get("embedding_model") or Config.default_embedding_model),
            }
            self.runner.initialize(
                root=stage_root, completion_model=models["completion"], embedding_model=models["embedding"],
                environment=provider_environment,
                cancel_requested=lambda: self._cancellation_requested(job_id),
            )
            self._assert_not_cancelled(job_id)
            self._remove_generated_env_file(stage_root)
            identity = self._candidate_identity()
            self.service.db.update_job(job_id, progress=10, log="GraphRAG Native：冻结 active revision 与结构化 Documents")
            inputs = GraphRAGNativeInputBuilder(self.service).build(project_id=project_id, run_root=stage_root)
            prompt_profile = GraphRAGNativePromptProfileBuilder.apply(
                run_root=stage_root,
                texts=tuple(item.text for item in inputs.input_documents),
                requested_language=str(
                    requested_knowledge_language
                ),
            )
            prompt_hash = str(prompt_profile["prompt_profile_hash"])
            output_language = str(
                (prompt_profile.get("language") or {}).get("language_code") or "en"
            )
            output_language_name = str(
                (prompt_profile.get("language") or {}).get("display_name")
                or output_language
            )
            self._write_candidate_evidence(
                stage_root, identity=identity, prompt_hash=prompt_hash,
                prompt_profile=prompt_profile,
            )
            self.service.db.update_job(
                job_id, progress=13,
                log=(
                    "GraphRAG Native：项目输出语言已冻结为"
                    f"{output_language_name}（"
                    + (
                        "项目设置"
                        if (prompt_profile.get("language") or {}).get(
                            "selection_mode"
                        ) == "explicit"
                        else "随文档语言"
                    )
                    + "）"
                ),
            )
            model_identity, dimension = self._provider_identity(project)
            self._assert_not_cancelled(job_id)
            settings = self.settings.write_and_validate(
                run_root=stage_root, completion_model=models["completion"], embedding_model=models["embedding"],
                embedding_dimension=dimension, api_base=self._provider_api_base(),
            )
            run = self.service.db.create_graphrag_native_run({
                "run_id": run_id, "project_id": project_id, **identity,
                "source_snapshot_hash": inputs.source_snapshot_hash,
                "settings_hash": settings["settings_hash"], "prompt_profile_hash": prompt_hash,
                **model_identity, "embedding_dimension": dimension, "graphrag_version": "3.1.1",
                "python_version": sys.version.split()[0],
                "staging_relative_path": self._staging_relative_path(run_id, attempt_id),
            })
            self.service.db.acquire_graphrag_native_run_lease(
                run_id, owner, ttl_seconds=self._LEASE_TTL_SECONDS,
            )
            self.service.db.replace_graphrag_native_run_documents(run_id, list(inputs.run_documents), owner=owner)
            reused_cache = self._seed_compatible_graph_cache(
                stage_root, project_id=project_id, current_run=run,
            )
            if reused_cache:
                self.service.db.update_job(
                    job_id, progress=16,
                    log=(
                        "GraphRAG Native：复用同源失败运行的模型调用缓存 "
                        f"{int(reused_cache['file_count'])} 项"
                    ),
                )
            self.service.db.update_job(job_id, progress=18, log="GraphRAG Native：验证 3.1.1 配置")
            self.runner.config_dry_run(
                root=stage_root, environment=provider_environment,
                cancel_requested=lambda: self._cancellation_requested(job_id),
            )
            self._assert_not_cancelled(job_id)
            self.service.db.update_job(job_id, progress=25, log="GraphRAG Native：验证 completion/embedding 连通性")
            self.runner.connectivity_dry_run(
                root=stage_root, environment=provider_environment,
                cancel_requested=lambda: self._cancellation_requested(job_id),
            )
            self._assert_not_cancelled(job_id)
            self.service.db.update_graphrag_native_run(run_id, owner=owner, state="RUNNING")
            self.service.db.update_job(job_id, progress=30, log="GraphRAG Native：执行 Standard workflow")
            workflow_tracker = NativeStandardWorkflowProgress()
            last_lease_renewal = time.monotonic()
            lease_failure: Exception | None = None

            def renew_native_lease(*, force: bool = False) -> bool:
                """Renew ownership while the isolated child is alive.

                A renewal failure must stop the child and fail closed.  It is
                never treated as an observability-only error because every
                subsequent write or activation depends on this ownership.
                """

                nonlocal last_lease_renewal, lease_failure
                if lease_failure is not None:
                    return False
                now = time.monotonic()
                if not force and now - last_lease_renewal < self._LEASE_RENEW_INTERVAL_SECONDS:
                    return True
                try:
                    self.service.db.renew_graphrag_native_run_lease(
                        run_id, owner, ttl_seconds=self._LEASE_TTL_SECONDS,
                    )
                except Exception as exc:
                    lease_failure = exc
                    return False
                last_lease_renewal = now
                return True

            def report_workflow_progress(event: NativeProcessProgress) -> None:
                renew_native_lease()
                if lease_failure is not None:
                    return
                # Do not revive a job that has been cancelled while the
                # isolated child is shutting down.
                current_job = self.service.db.get_job(job_id) or {}
                if (
                    str(current_job.get("status") or "") not in {"queued", "running"}
                    or self._job_cancel_requested(current_job)
                ):
                    return
                workflow_progress = workflow_tracker.update(event)
                self.service.db.update_job(
                    job_id, status="running", progress=workflow_progress,
                    log="GraphRAG Native：" + event.message,
                    metadata={
                        "native_progress": workflow_tracker.metadata(event),
                    },
                )

            index_result = self.runner.index(
                root=stage_root, environment=provider_environment,
                cancel_requested=lambda: (
                    lease_failure is not None
                    or self._cancellation_requested(job_id)
                ),
                progress_callback=report_workflow_progress,
            )
            if lease_failure is not None:
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            if not renew_native_lease(force=True):
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            self._assert_not_cancelled(job_id)
            self._write_workflow_ledger(stage_root, index_result)
            self._require_nonempty_semantic_graph(stage_root)
            self._reject_source_metadata_pollution(stage_root)
            self.service.db.update_job(
                job_id, progress=72,
                log="GraphRAG Native：知识提取完成，开始检查构建结果",
                metadata={
                    "native_progress": {
                        "schema_version": "graphrag-native-progress-v2",
                        "phase": "validation",
                        "detail": "正在检查图谱结构与问答索引",
                        "execution_progress": 100,
                        "phase_progress": 0,
                        "overall_progress": 72,
                    },
                },
            )
            self.service.db.update_graphrag_native_run(
                run_id, owner=owner, state="EXECUTION_COMPLETE",
            )
            self.service.db.update_graphrag_native_run(run_id, owner=owner, state="VALIDATING")
            artifacts = self._collect_artifacts(stage_root)
            write_run_manifest(
                stage_root, identity={
                    **identity, "run_id": run_id,
                    "source_snapshot_hash": inputs.source_snapshot_hash,
                    "prompt_profile_hash": prompt_hash,
                    "output_language": output_language,
                },
                artifacts=artifacts, workflow_ledger_path="kf-manifest/workflow-ledger.jsonl",
                settings_redacted_path="settings.redacted.yaml", lancedb_relative_path="output/lancedb",
            )
            report = self.validator.validate(stage_root)
            self._assert_not_cancelled(job_id)
            if not renew_native_lease(force=True):
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            self._write_post_index_evidence(
                stage_root, validation_report=report,
                migration_report=self.service.db.assert_graphrag_native_ready(),
            )
            self.service.db.update_job(
                job_id, progress=86,
                log="GraphRAG Native：图谱结构与问答索引已检查，正在保存审计记录",
                metadata={
                    "native_progress": {
                        "schema_version": "graphrag-native-progress-v2",
                        "phase": "validation",
                        "detail": "正在保存构建审计记录",
                        "execution_progress": 100,
                        "phase_progress": 50,
                        "overall_progress": 86,
                    },
                },
            )
            # Post-index evidence is generated after the first structural
            # report.  Run the independent validator again so its credential
            # scanner and file-safety checks cover those evidence files too;
            # READY is bound to this final validation result.
            report = self.validator.validate(stage_root)
            self._assert_not_cancelled(job_id)
            if not renew_native_lease(force=True):
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            report_hash = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
            self.service.db.update_job(
                job_id, progress=94,
                log="GraphRAG Native：最终检查通过，正在发布知识图谱成果",
                metadata={
                    "native_progress": {
                        "schema_version": "graphrag-native-progress-v2",
                        "phase": "publication",
                        "detail": "正在发布已通过检查的知识图谱成果",
                        "execution_progress": 100,
                        "phase_progress": 79,
                        "overall_progress": 94,
                    },
                },
            )
            self._assert_not_cancelled(job_id)
            final_root.parent.mkdir(parents=True, exist_ok=True)
            stage_root.replace(final_root)
            self._assert_not_cancelled(job_id)
            if not renew_native_lease(force=True):
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            self.service.db.update_graphrag_native_run(
                run_id, owner=owner, bundle_relative_path=self._run_relative_path(project_id, run_id),
                validator_version=self.validator.validator_version,
            )
            self.service.db.replace_graphrag_native_artifacts(
                run_id, [asdict(item) for item in artifacts], owner=owner,
            )
            self.service.db.activate_graphrag_native_run(
                run_id, owner=owner, validation_report_hash=report_hash,
                expected_source_snapshot_hash=inputs.source_snapshot_hash,
                job_id=job_id,
            )
            activated = True
            self.service.db.update_job(
                job_id, status="completed", progress=100,
                log="GraphRAG Native：实体关系、主题社区、摘要与问答索引已通过检查",
                metadata={
                    "run_id": run_id,
                    "source_snapshot_hash": inputs.source_snapshot_hash,
                    "output_language": output_language,
                    "native_progress": {
                        "schema_version": "graphrag-native-progress-v2",
                        "phase": "completed",
                        "detail": "知识图谱成果已发布",
                        "execution_progress": 100,
                        "phase_progress": 100,
                        "overall_progress": 100,
                    },
                },
            )
        except Exception as exc:
            current_job = self.service.db.get_job(job_id) or {}
            cancelled = bool(
                isinstance(exc, (NativeBuildCancelled, JobCancellationRequested))
                or self._job_cancel_requested(current_job)
            )
            message = (
                f"GraphRAG Native 构建已安全终止：{exc}"
                if cancelled else
                f"GraphRAG Native 构建失败，禁止回退旧流程：{exc}"
            )
            no_graph = isinstance(exc, NativeNoGraphUnsupported)
            if run:
                try:
                    terminal_state = "CANCELLED" if cancelled else "NO_GRAPH_UNSUPPORTED" if no_graph else "FAILED"
                    try:
                        self.service.db.update_graphrag_native_run(
                            str(run["run_id"]), owner=owner, state=terminal_state,
                            failure_class=type(exc).__name__, failure_message=message,
                        )
                        release = getattr(self.service.db, "release_graphrag_native_run_lease", None)
                        if release:
                            release(str(run["run_id"]), owner)
                    except RuntimeError:
                        # Lease expiry is a terminal failure too; record it
                        # without weakening the strict lease requirement for
                        # execution, artifacts, validation, or activation.
                        close_after_lease_loss = getattr(
                            self.service.db, "fail_graphrag_native_run_after_lease_loss", None,
                        )
                        if close_after_lease_loss is None:
                            raise
                        close_after_lease_loss(
                            str(run["run_id"]), owner=owner, state=terminal_state,
                            failure_class=type(exc).__name__, failure_message=message,
                        )
                except Exception:
                    pass
            if cancelled:
                if not activated:
                    for path in (stage_root, final_root):
                        try:
                            self._discard_cancelled_bundle_path(path)
                        except Exception:
                            pass
                self.service.db.update_job(
                    job_id, status="cancelled", progress=100,
                    log=message + "；未完成 bundle 未发布",
                    metadata={
                        "stage": "cancelled",
                        "cancel_acknowledged_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            else:
                self.service.db.update_job(job_id, status="failed", progress=100, log=message, error=message)
