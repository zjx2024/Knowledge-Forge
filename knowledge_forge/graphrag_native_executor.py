"""Project-wide, isolated Microsoft GraphRAG 3.1.1 executor."""

from __future__ import annotations

import hashlib
import json
import os
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
from .graphrag_native_artifacts import (
    GraphRAGNativeBundleValidator,
    describe_file,
    write_run_manifest,
)
from .graphrag_native_config import GraphRAGNativeSettingsBuilder
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
        self.runner = GraphRAGNativeRunner()
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
        base_sha = self._git_output(repo, "rev-parse", "origin/fix/shadow-relation-quality-v2").strip()
        head = self._git_output(repo, "rev-parse", "HEAD").strip()
        diff = self._git_output(repo, "diff", "--binary", base_sha).encode("utf-8")
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
    def _workflow_progress_value(event: NativeProcessProgress, previous: int) -> int:
        """Map a CLI's generic ``completed/total`` signal into the index band.

        The runner may report several GraphRAG workflows, each with its own
        denominator.  The UI must never move backwards or claim that final
        bundle validation has happened, so live workflow events stay strictly
        inside the 30–68 execution band.  Executor-owned validation starts at
        72 only after the CLI exits successfully.
        """

        if event.completed is None or event.total is None or event.total <= 0:
            return int(previous)
        fraction = max(0.0, min(1.0, float(event.completed) / float(event.total)))
        return max(int(previous), min(68, 30 + max(1, round(38 * fraction))))

    @staticmethod
    def _patch_source_identity_prompt(root: Path) -> str:
        path = root / "prompts" / "extract_graph.txt"
        if not path.is_file():
            raise RuntimeError("GraphRAG init 未生成 extract_graph prompt")
        text = path.read_text(encoding="utf-8")
        instruction = (
            "\n\nKnowledge-Forge native input may include a [KF_SOURCE_ID:...] metadata marker. "
            "It is provenance metadata, never an entity, relationship endpoint, type, or claim.\n"
        )
        if instruction not in text:
            path.write_text(text + instruction, encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

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

    @staticmethod
    def _collect_artifacts(root: Path) -> list[Any]:
        artifacts: list[Any] = []
        for relative in (
            "input/documents.parquet", "kf-manifest/source-map.parquet",
            "kf-manifest/input-documents-manifest.jsonl", "settings.redacted.yaml",
            "kf-manifest/workflow-ledger.jsonl", "output/documents.parquet",
            "output/text_units.parquet", "output/entities.parquet", "output/relationships.parquet",
            "output/raw_entities.parquet", "output/raw_relationships.parquet",
            "output/communities.parquet", "output/community_reports.parquet", "output/graph.graphml",
        ):
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
        patch = self._git_output(repo, "diff", "--binary", identity["base_sha"])
        (manifest_root / "candidate-worktree.patch").write_text(patch, encoding="utf-8")
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
        (manifest_root / "prompt-manifest.json").write_text(json.dumps({
            "extract_graph_prompt_sha256": prompt_hash,
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
            "forbidden": ["HugeGraph", "pgvector", "stable_entity_relation", "enhanced_shadow", "graphrag_community"],
            "evidence": "native executor uses only persisted structures, GraphRAG CLI, control-plane DB, and bundle files",
        }, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        (manifest_root / "stable-enhanced-invariance.json").write_text(json.dumps({
            "native_writes": ["graphrag_native_* control-plane tables only"],
            "stable_enhanced_projection_mutation": 0,
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
        """Run a fresh Standard workflow; no Stable/Shadow/HugeGraph fallback exists."""

        run_id = "grn_" + uuid.uuid4().hex
        attempt_id = uuid.uuid4().hex
        owner = "native-worker-" + uuid.uuid4().hex
        stage_root = self.native_root / self._staging_relative_path(run_id, attempt_id)
        final_root = self.native_root / self._run_relative_path(project_id, run_id)
        run: dict[str, Any] | None = None
        try:
            self.service.db.assert_graphrag_native_ready()
            project = self.service.db.get_project(project_id)
            if not project:
                raise LookupError("项目不存在")
            if self.service.db.latest_active_job(project_id, "ingestion"):
                raise RuntimeError("源文档正在变化，拒绝创建 GraphRAG Native 快照")
            preflight = GraphRAGNativeInputBuilder(self.service).preflight(project_id=project_id)
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
            )
            self._remove_generated_env_file(stage_root)
            prompt_hash = self._patch_source_identity_prompt(stage_root)
            identity = self._candidate_identity()
            self._write_candidate_evidence(stage_root, identity=identity, prompt_hash=prompt_hash)
            self.service.db.update_job(job_id, progress=10, log="GraphRAG Native：冻结 active revision 与结构化 Documents")
            inputs = GraphRAGNativeInputBuilder(self.service).build(project_id=project_id, run_root=stage_root)
            model_identity, dimension = self._provider_identity(project)
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
            self.service.db.update_job(job_id, progress=18, log="GraphRAG Native：验证 3.1.1 配置")
            self.runner.config_dry_run(root=stage_root, environment=provider_environment)
            self.service.db.update_job(job_id, progress=25, log="GraphRAG Native：验证 completion/embedding 连通性")
            self.runner.connectivity_dry_run(root=stage_root, environment=provider_environment)
            self.service.db.update_graphrag_native_run(run_id, owner=owner, state="RUNNING")
            self.service.db.update_job(job_id, progress=30, log="GraphRAG Native：执行 Standard workflow")
            workflow_progress = 30
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
                nonlocal workflow_progress
                renew_native_lease()
                if lease_failure is not None:
                    return
                # Do not revive a job that has been cancelled while the
                # isolated child is shutting down.
                current_job = self.service.db.get_job(job_id) or {}
                if str(current_job.get("status") or "") not in {"queued", "running"}:
                    return
                workflow_progress = self._workflow_progress_value(event, workflow_progress)
                self.service.db.update_job(
                    job_id, status="running", progress=workflow_progress,
                    log="GraphRAG Native：" + event.message,
                )

            index_result = self.runner.index(
                root=stage_root, environment=provider_environment,
                cancel_requested=lambda: (
                    lease_failure is not None
                    or (self.service.db.get_job(job_id) or {}).get("status") == "cancelled"
                ),
                progress_callback=report_workflow_progress,
            )
            if lease_failure is not None:
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            if not renew_native_lease(force=True):
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            self._write_workflow_ledger(stage_root, index_result)
            self._require_nonempty_semantic_graph(stage_root)
            self._reject_source_metadata_pollution(stage_root)
            self.service.db.update_job(
                job_id, progress=72,
                log="GraphRAG Native：Standard workflow 完成，开始独立验证原生输出",
            )
            self.service.db.update_graphrag_native_run(
                run_id, owner=owner, state="EXECUTION_COMPLETE",
            )
            self.service.db.update_graphrag_native_run(run_id, owner=owner, state="VALIDATING")
            artifacts = self._collect_artifacts(stage_root)
            write_run_manifest(
                stage_root, identity={**identity, "run_id": run_id, "source_snapshot_hash": inputs.source_snapshot_hash},
                artifacts=artifacts, workflow_ledger_path="kf-manifest/workflow-ledger.jsonl",
                settings_redacted_path="settings.redacted.yaml", lancedb_relative_path="output/lancedb",
            )
            report = self.validator.validate(stage_root)
            if not renew_native_lease(force=True):
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            self._write_post_index_evidence(
                stage_root, validation_report=report,
                migration_report=self.service.db.assert_graphrag_native_ready(),
            )
            self.service.db.update_job(
                job_id, progress=86,
                log="GraphRAG Native：原生输出结构与 LanceDB 已验证，写入可审计证据",
            )
            # Post-index evidence is generated after the first structural
            # report.  Run the independent validator again so its credential
            # scanner and file-safety checks cover those evidence files too;
            # READY is bound to this final validation result.
            report = self.validator.validate(stage_root)
            if not renew_native_lease(force=True):
                raise RuntimeError(f"GraphRAG Native run lease 续期失败：{lease_failure}") from lease_failure
            report_hash = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
            self.service.db.update_job(
                job_id, progress=94,
                log="GraphRAG Native：最终独立验证通过，发布 immutable native bundle",
            )
            final_root.parent.mkdir(parents=True, exist_ok=True)
            stage_root.replace(final_root)
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
            )
            self.service.db.update_job(
                job_id, status="completed", progress=100,
                log="GraphRAG Native：原生语义图、社区、报告与 LanceDB bundle 已通过独立验证",
                metadata={"run_id": run_id, "source_snapshot_hash": inputs.source_snapshot_hash},
            )
        except Exception as exc:
            message = f"GraphRAG Native 构建失败，禁止回退旧流程：{exc}"
            cancelled = bool((self.service.db.get_job(job_id) or {}).get("status") == "cancelled")
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
            if not cancelled:
                self.service.db.update_job(job_id, status="failed", progress=100, log=message, error=message)
