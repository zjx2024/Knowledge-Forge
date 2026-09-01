"""Project lifecycle, document ingestion, retrieval and graph orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import shutil
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from .chunking import (
    DOCUMENT_CHUNK_PROFILE_VERSION,
    DOCUMENT_STRUCTURE_PROFILE_VERSION,
    ProcessingResult,
    build_processing_result,
    parse_document_structure,
    validate_processing_request,
)
from .config import Config
from .construction_strategies import (
    KNOWLEDGE_FORGE_STRATEGY,
    GRAPHRAG_NATIVE_STRATEGY,
    STABLE_STRATEGY,
    normalize_strategy,
)
from .knowledge_language import (
    CHINESE,
    ENGLISH,
    normalize_knowledge_language,
    resolve_knowledge_language,
)
from .graphrag_native_executor import GraphRAGNativeExecutor
from .graphrag_native_inputs import GraphRAGNativeInputBuilder
from .graphrag_native_runtime import GraphRAGNativeRuntime
from .graphrag_community import CommunityConfig, CommunityDetectionCancelled
from .knowledge_forge_communities import (
    KnowledgeForgeCommunityService,
    enrich_community_descriptions,
)
from .community_report_runtime import (
    COMMUNITY_REPORT_PROMPT_VERSION,
    CommunityReportPolicy,
    KnowledgeForgeCommunityReportGenerator,
    build_community_report_plan,
    clean_report_prose,
    community_report_request_hash,
)
from .knowledge_forge_descriptions import (
    enrich_knowledge_forge_snapshot,
    entity_type_label,
    relation_display_label,
    relation_type_label,
)
from .database import Database, JobCancellationRequested, JobPublicationCommitted
from .unicode_safety import sanitize_unicode_data
from .entity_resolution import EntityResolver
from .entity_publishability import (
    canonical_display_surface,
    ENTITY_PUBLISHABILITY_POLICY_VERSION,
    aggregate_canonical_publishability,
    project_publishable_snapshot,
)
from .grounded_entities import (
    DatabaseGroundedStore,
    GroundedEntityResolver,
    GroundedModelGateway,
    bounded_identity_evidence,
)
from .extraction_funnel import relation_quality_funnel
from .graph_policy import (
    GRAPH_EMPTY,
    GRAPH_FAILED,
    GRAPH_POLICY_HASH,
    GRAPH_POLICY_VERSION,
    GRAPH_READY,
    GRAPH_REBUILDING,
    GRAPH_STALE,
    GraphNotReadyError,
    graph_policy_hash,
    is_current_graph,
    project_graph_policy_hash,
)
from .hugegraph_store import HugeGraphStore
from .graph_manifest import compare_graph_manifests
from .graph_explorer import build_graph_explorer_view
from .graph_retrieval import (
    GRAPH_RETRIEVAL_POLICY_VERSION,
    build_graph_entity_hops,
    graph_neighbor_keys,
    rank_graph_relations,
    select_graph_seed_entities,
)
from .graph_safety import GraphMutationContext
from .media import DocumentMediaService
from .ollama import OllamaClient
from .agent.authorization import session_key_hash
from .agent.trace import build_public_execution_trace
from .agent.workflow import KnowledgeAgentWorkflow
from .observability import recorded_job
from .policy import (
    ENTITY_EXTRACTION_POLICY_SET,
    POLICY_AXES,
    RELATION_CANDIDATE_POLICY_SET,
    GROUNDED_ENTITY_POLICY_SET,
)
from .quality import canonical_digest
from .release_gate import ALLOWED_EVALUATOR_VERSIONS, evaluate_frozen_bundle
from .relation_pipeline import RelationCandidatePipeline
from .grounded_relations import (
    LOADED_RELATION_CODE_DIGEST,
    GROUNDED_RELATION_POLICY_SET,
    GroundedRelationRuntime,
    grounded_relation_treatment_digest,
)
from .relation_replay import (
    PersistedRelationModelReplayGateway,
    RelationReplayError,
    assert_code_only_replay_compatible,
)
from .schema_evolution import SchemaEvolutionService
from .grounded_context import build_grounded_context_manifest
from .span_entities import (
    SpanFirstEntityExtractor,
    policy_relation_endpoint_candidates,
    structured_list_item_endpoint_candidates,
    structured_relation_endpoint_candidates,
    structured_section_label_endpoint_candidates,
)
from .vector_store import VectorStore, cosine_similarity, lexical_similarity


LOGGER = logging.getLogger(__name__)


def _optional_relation_confidence(metadata: dict[str, Any]) -> float | None:
    """Return a real supplied score without inventing certainty for absence."""

    value = metadata.get("confidence")
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if 0.0 <= result <= 1.0 else None


def _supported_evidence_count(
    metadata: dict[str, Any], evidence_text: object,
) -> int:
    """Count independently retained evidence records for a published fact."""

    for key in ("evidence_keys", "all_evidence_spans"):
        values = metadata.get(key)
        if isinstance(values, (list, tuple)) and values:
            return len(values)
    return 1 if str(evidence_text or "").strip() else 0


def _page_label(values: object) -> str:
    pages = sorted({
        int(value) for value in (values if isinstance(values, (list, tuple)) else [])
        if str(value).isdigit() and int(value) > 0
    })
    if not pages:
        return ""
    if len(pages) == 1:
        return f"第 {pages[0]} 页"
    if pages == list(range(pages[0], pages[-1] + 1)):
        return f"第 {pages[0]}–{pages[-1]} 页"
    shown = "、".join(str(value) for value in pages[:6])
    return f"第 {shown}{' 等' if len(pages) > 6 else ''}页"


def _chunk_source_payload(chunk: dict[str, Any] | None) -> dict[str, Any]:
    if not chunk:
        return {}
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    media_assets = []
    for raw in metadata.get("media_assets") or []:
        if not isinstance(raw, dict) or not str(raw.get("url") or "").strip():
            continue
        sources = [
            {
                "page": source.get("page"),
                "location": str(source.get("location") or ""),
            }
            for source in raw.get("sources") or [] if isinstance(source, dict)
        ]
        media_assets.append({
            "type": "image",
            "url": str(raw.get("url") or ""),
            "width": max(0, int(raw.get("width") or 0)),
            "height": max(0, int(raw.get("height") or 0)),
            "title": str(raw.get("title") or "文档图片"),
            "description": str(raw.get("description") or ""),
            "sources": sources,
        })
    content = str(chunk.get("content") or "")
    lines = [line for line in content.splitlines() if line.strip()]
    looks_tabular = bool(
        metadata.get("table")
        or str(metadata.get("chunk_role") or "") == "table_row"
        or str(metadata.get("strategy") or "") == "table"
        or "table" in set(metadata.get("block_types") or [])
        or sum(line.count("|") >= 2 for line in lines) >= 2
    )
    return {
        "document_id": int(chunk.get("document_id") or 0),
        "filename": str(chunk.get("original_filename") or ""),
        "document_url": str(chunk.get("storage_path") or ""),
        "revision_id": str(chunk.get("revision_id") or ""),
        "mime_type": str(chunk.get("mime_type") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "pages": list(metadata.get("pages") or []),
        "page_label": _page_label(metadata.get("pages") or []),
        "media_assets": media_assets[:12],
        "table": ({
            "title": str(chunk.get("section_title") or "表格内容"),
            "text": content[:12_000],
        } if looks_tabular else None),
    }


def _public_community_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the stable browser/search projection of one report item.

    In particular, do not expose the bounded model prompt, retry diagnostics or
    raw model output through the public status endpoint.
    """

    if not report:
        return None
    persisted_status = str(report.get("status") or "pending")
    status = (
        "deferred"
        if persisted_status == "skipped"
        and str(report.get("error_category") or "") == "Deferred"
        else persisted_status
    )
    input_context = (
        report.get("input_context")
        if isinstance(report.get("input_context"), dict) else {}
    )
    return {
        "report_id": str(report.get("report_id") or ""),
        "community_key": str(report.get("community_key") or ""),
        "status": status,
        "title": clean_report_prose(report.get("title"), input_context),
        "summary": clean_report_prose(report.get("summary"), input_context),
        "findings": [
            {
                "title": clean_report_prose(item.get("title"), input_context),
                "explanation": clean_report_prose(
                    item.get("explanation"), input_context,
                ),
                "entity_keys": list(item.get("entity_keys") or []),
                "relation_keys": list(item.get("relation_keys") or []),
                "sources": [],
            }
            for item in report.get("findings") or []
            if isinstance(item, dict)
        ],
        "limitations": clean_report_prose(
            report.get("limitations"), input_context,
        ),
        "content": clean_report_prose(report.get("content"), input_context),
        "cited_entity_keys": list(report.get("cited_entity_keys") or []),
        "cited_relation_keys": list(report.get("cited_relation_keys") or []),
        "embedding_status": str(report.get("embedding_status") or "pending"),
        "source_report_id": str(report.get("source_report_id") or ""),
        "completed_at": report.get("completed_at"),
        "sources": [],
    }


def _attach_public_community_report_sources(
    report: dict[str, Any], sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach exact relation provenance to the finding that cited it.

    Report-level sources remain available for compatibility and aggregate
    views.  The additive per-finding projection lets presentation and search
    clients explain each conclusion without guessing from report order.
    """

    normalized_sources = [dict(item) for item in sources if isinstance(item, dict)]
    report["sources"] = normalized_sources
    sources_by_relation: dict[str, list[dict[str, Any]]] = {}
    for source in normalized_sources:
        relation_key = str(source.get("relation_key") or "")
        if relation_key:
            sources_by_relation.setdefault(relation_key, []).append(source)
    for finding in report.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        finding["sources"] = [
            dict(source)
            for relation_key in finding.get("relation_keys") or []
            for source in sources_by_relation.get(str(relation_key), [])
        ]
    return report


def _public_community_report_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run:
        return None
    return {
        "run_id": str(run.get("run_id") or ""),
        "job_id": str(run.get("job_id") or ""),
        "status": str(run.get("status") or "not_started"),
        "expected_count": int(run.get("expected_count") or 0),
        "eligible_count": int(run.get("eligible_count") or 0),
        "completed_count": int(run.get("completed_count") or 0),
        "reused_count": int(run.get("reused_count") or 0),
        "failed_count": int(run.get("failed_count") or 0),
        "skipped_count": int(run.get("skipped_count") or 0),
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
    }


# Bump this whenever the entity extraction or identity-resolution treatment
# changes in a way that requires rematerializing the immutable namespace.  The
# public project build entrypoint compares this value with the namespace used
# by the currently advertised graph, so a code-only upgrade cannot silently
# keep serving an older entity snapshot.
GROUNDED_ENTITY_CODE_VERSION = (
    "knowledge-forge-grounded-entity-v50-bounded-layout-mention-coverage"
)
# The exact model request/response contract did not change with the v50
# deterministic occurrence recovery. Keep the previous semantic version in
# the raw extraction cache key so this code-only rebuild reuses model output.
GROUNDED_ENTITY_MODEL_EXTRACTION_CONTRACT_VERSION = (
    "knowledge-forge-grounded-entity-v40-soft-wrap-suffix-recovery"
)
GROUNDED_ENTITY_EXTRACTION_CACHE_VERSION = (
    "knowledge-forge-grounded-entity-extraction-cache-v1"
)


def _grounded_entity_freshness_policy_set() -> dict[str, str]:
    """Return every cheap entity-publication axis required by a ready graph.

    The shared entity policy intentionally excludes relation treatment axes,
    but the deterministic publishability projection is also part of the
    public entity snapshot.  Including it here means a future policy-only
    upgrade invalidates an old graph without forcing a new model extraction
    contract or relying on an unrelated relation-code digest.
    """

    return {
        **ENTITY_EXTRACTION_POLICY_SET,
        "entity_publishability_policy_version": (
            ENTITY_PUBLISHABILITY_POLICY_VERSION
        ),
    }


class IngestionCancelled(RuntimeError):
    """Raised inside a worker after its document has been deleted or cancelled."""


class TreatmentMaterializationRequired(RuntimeError):
    """Treatment snapshots cannot activate until formal materialization is implemented."""


class KnowledgeForgeService:
    INGESTION_SUBTASKS = (
        "structure",
        "chunking",
        "schema_base",
        "extraction",
        "graph_sync",
    )
    # Respect the deployment's configured application-side Ollama budget for
    # newly created projects.  ``project_token_limits`` still clamps this to
    # the selected model's verified runtime capability.
    DEFAULT_MODEL_MAX_TOKENS = max(
        8192, int(Config.ollama_default_context_budget),
    )
    DEFAULT_CHUNK_MAX_TOKENS = 512
    USER_CANCELLABLE_JOB_KINDS = frozenset({
        "content_preparation",
        "ingestion",
        "graphrag_native_build",
        "graph_sync",
        "community_report_build",
    })

    @staticmethod
    def _relation_entities_with_canonicals(
        raw_entities: list[dict[str, Any]],
        raw_to_mention: dict[str, str],
        canonical_by_mention: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Attach resolved mention and canonical identities before P1 planning."""

        resolved = []
        for raw in raw_entities:
            raw_id = str(raw.get("mention_id") or raw.get("id") or "")
            mention_id = raw_to_mention.get(raw_id)
            canonical_id = canonical_by_mention.get(str(mention_id or ""))
            if not mention_id or not canonical_id:
                continue
            resolved.append({
                **raw,
                "mention_id": mention_id,
                "mention_key": str(raw.get("mention_key") or mention_id),
                "canonical_id": canonical_id,
            })
        return resolved

    @staticmethod
    def _relation_endpoint_mention_id(
        endpoint_id: str,
        raw_to_mention: dict[str, str],
        canonical_by_mention: dict[str, str],
    ) -> str:
        """Accept relation endpoints before or after mention normalization."""

        if endpoint_id in canonical_by_mention:
            return endpoint_id
        mention_id = str(raw_to_mention.get(endpoint_id) or "")
        return mention_id if mention_id in canonical_by_mention else ""

    @staticmethod
    def _relation_run_id(
        project_id: int,
        document_id: int,
        revision_id: str,
        chunk_id: int,
        treatment_hash: str,
        input_fingerprint: str,
    ) -> str:
        material = "|".join((
            str(project_id),
            str(document_id),
            revision_id,
            str(chunk_id),
            treatment_hash,
            input_fingerprint,
        ))
        return "relshadow_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _relation_input_fingerprint(
        *,
        revision_id: str,
        chunk_id: int,
        content: str,
        schema_version: int,
        treatment_hash: str,
        candidate_map: dict[str, dict[str, Any]],
    ) -> str:
        """Bind a relation run to complete candidates, including canonical endpoints."""

        payload = {
            "revision_id": revision_id,
            "chunk_id": chunk_id,
            "chunk_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "schema_version": schema_version,
            "treatment_hash": treatment_hash,
            "candidates": [
                candidate_map[candidate_id] for candidate_id in sorted(candidate_map)
            ],
        }
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def __init__(self, database: Database | None = None):
        self.db = database or Database()
        self.ollama = OllamaClient()
        self.media = DocumentMediaService(self.ollama)
        self.vectors = VectorStore(self.db, self.ollama)
        self.schemas = SchemaEvolutionService(self.db, self.ollama)
        self.resolver = EntityResolver(self.db, self.vectors)
        self.relation_pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
        self.graph = HugeGraphStore(self.db)
        self._document_locks_guard = threading.Lock()
        self._document_locks: dict[int, threading.RLock] = {}
        self._graph_locks_guard = threading.Lock()
        self._graph_locks: dict[int, threading.RLock] = {}
        # A single worker keeps multi-file ingestion deterministic and prevents local LLM contention.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-forge")
        # Cleanup must be able to wait for an active document without blocking queued ingestion work.
        self.cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-forge-cleanup")
        # Reports enhance a published graph and must never hold the ingestion
        # queue.  One worker avoids competing local LLM calls between reports.
        self.report_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="knowledge-forge-community-reports",
        )
        self.graphrag_native = GraphRAGNativeExecutor(self)
        self.graphrag_native_runtime = GraphRAGNativeRuntime(self)
        self.knowledge_forge_communities = KnowledgeForgeCommunityService(self.db)
        self.community_report_generator = KnowledgeForgeCommunityReportGenerator(
            self.ollama,
        )
        self.agent = KnowledgeAgentWorkflow(self)
        self.agent_executor = ThreadPoolExecutor(
            max_workers=Config.agent_max_workers,
            thread_name_prefix="knowledge-forge-agent",
        )

    def _mark_graphrag_native_source_changed(self, project_id: int, reason: str) -> None:
        """Invalidate only the immutable native pointer after source mutation."""

        # The native migration is additive.  On an older database where the
        # feature cannot exist yet, ordinary upload/delete must remain usable;
        # once the control plane exists, an invalidation failure is surfaced by
        # a native build/status rather than silently queried as fresh.
        marker = getattr(self.db, "mark_graphrag_native_stale", None)
        readiness = getattr(self.db, "assert_graphrag_native_ready", None)
        if marker is None or readiness is None:
            return
        try:
            readiness()
        except RuntimeError as exc:
            # Before migration 0022 is applied no native pointer can be
            # queryable.  Do not make ordinary uploads/deletes unavailable on
            # a pre-feature database; a native build itself remains fail-closed.
            if "数据库结构不完整" in str(exc):
                return
            raise
        cancel = getattr(self.db, "cancel_project_jobs", None)
        if cancel is not None:
            cancel(int(project_id), ["graphrag_native_build"], reason)
        marker(int(project_id), reason)

    def _strategy_executor(self, strategy_key: str) -> Callable[..., None]:
        """Resolve all ingestion strategies explicitly; never use an ``else`` fallback."""

        strategy_key = normalize_strategy(strategy_key)
        registry: dict[str, Callable[..., None]] = {
            STABLE_STRATEGY: self._ingest_document,
            KNOWLEDGE_FORGE_STRATEGY: self._ingest_knowledge_forge_strategy,
            GRAPHRAG_NATIVE_STRATEGY: self._ingest_graphrag_native,
        }
        try:
            return registry[strategy_key]
        except KeyError as exc:  # normalize_strategy currently catches this first.
            raise ValueError(f"未注册的知识构建方案: {strategy_key}") from exc

    def _ingest_graphrag_native(self, job_id: str, project_id: int) -> None:
        """Execute only the isolated native strategy; it has no fallback chain."""

        self.graphrag_native.run(job_id=job_id, project_id=project_id)

    def bootstrap(self) -> None:
        for directory in (
            Config.docs_root, Config.chunks_root, Config.media_root, Config.trash_root,
            Config.files_root / "graphrag_native",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.db.bootstrap()
        for project in self.db.list_projects():
            selection = self.db.get_project_construction_strategy_selection(int(project["id"]))
            if (
                selection
                and str(selection.get("association_status") or "") == "bound"
                and str(selection.get("selected_strategy") or "") == KNOWLEDGE_FORGE_STRATEGY
            ):
                self.schemas.ensure_base(int(project["id"]))
                self.db.ensure_project_strategy_graphs(int(project["id"]))
                self.db.reconcile_knowledge_forge_schema_freshness(
                    int(project["id"]),
                )
                self.db.reconcile_knowledge_forge_entity_freshness(
                    int(project["id"]),
                    code_version=GROUNDED_ENTITY_CODE_VERSION,
                    policy_set=_grounded_entity_freshness_policy_set(),
                )
                self.db.reconcile_knowledge_forge_relation_freshness(
                    int(project["id"]),
                    code_digest=LOADED_RELATION_CODE_DIGEST,
                    treatment_digest=grounded_relation_treatment_digest(),
                )
                self._reconcile_knowledge_forge_content_freshness(project)
        try:
            self.graph.ensure_schema()
        except Exception as exc:
            LOGGER.warning("HugeGraph bootstrap deferred: %s", exc)

    def _legacy_graph_recovery_allowed(self, project_id: int) -> bool:
        """Return whether an old production Stable graph may be recovered.

        Fixed Knowledge Forge and GraphRAG Native projects own independent
        graph products and must never acquire a background Stable graph job.
        The compatibility branch is only for older narrow test doubles that
        do not expose the project-strategy control plane.
        """

        getter = getattr(
            self.db, "get_project_construction_strategy_selection", None,
        )
        if getter is None:
            return True
        selection = getter(project_id)
        return bool(
            selection
            and str(selection.get("association_status") or "") == "bound"
            and str(selection.get("selected_strategy") or "") == STABLE_STRATEGY
        )

    def recover_interrupted_jobs(self) -> int:
        recovered = 0
        recover_agent = getattr(self.db, "recover_interrupted_agent_runs", None)
        if callable(recover_agent):
            recovered += int(recover_agent() or 0)
        for job in self.db.list_recoverable_jobs():
            job_id = str(job["id"])
            metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
            if metadata.get("cancel_requested"):
                if metadata.get("publication_committed"):
                    self.db.update_job(
                        job_id,
                        status="completed",
                        progress=100,
                        log="服务恢复时确认成果已发布，任务收尾完成",
                        metadata={"stage": "completed"},
                    )
                else:
                    self._recover_cancelled_document_staging(job)
                    self._recover_cancelled_native_staging(job)
                    self._recover_cancelled_graph_staging(job)
                    cancelled_subtasks = self._cancelled_subtasks(
                        job, "服务恢复时确认任务已终止",
                    )
                    self.db.update_job(
                        job_id,
                        status="cancelled",
                        progress=100,
                        log="服务恢复时确认终止请求；未完成结果未发布",
                        metadata={
                            "stage": "cancelled",
                            "cancel_acknowledged_at": datetime.now(timezone.utc).isoformat(),
                            **(
                                {"subtasks": cancelled_subtasks}
                                if cancelled_subtasks is not None else {}
                            ),
                        },
                    )
                continue
            self.db.update_job(
                job_id,
                status="queued",
                log="服务重新启动，任务将从头恢复",
                metadata={**(job.get("metadata") or {}), "recovered_after_restart": True},
            )
            if job["kind"] == "ingestion" and job.get("document_id"):
                strategy_key = normalize_strategy(job.get("strategy_key"))
                if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
                    self.db.update_job(
                        job_id, status="failed", progress=100,
                        log="native 策略不支持文档级恢复",
                        error="GraphRAG Native 仅支持项目级全量构建",
                    )
                    continue
                worker = self._strategy_executor(strategy_key)
                self.executor.submit(worker, job_id, int(job["document_id"]))
                recovered += 1
            elif job["kind"] == "graphrag_native_build" and job.get("project_id"):
                self.executor.submit(
                    self._strategy_executor(GRAPHRAG_NATIVE_STRATEGY),
                    job_id, int(job["project_id"]),
                )
                recovered += 1
            elif job["kind"] == "content_preparation" and job.get("document_id"):
                metadata = job.get("metadata") or {}
                strategy_key = str(
                    metadata.get("strategy_key")
                    or job.get("strategy_key")
                    or KNOWLEDGE_FORGE_STRATEGY
                )
                self.executor.submit(
                    self._prepare_document_content, job_id, int(job["document_id"]),
                    strategy_key,
                )
                recovered += 1
            elif job["kind"] == "graph_sync" and job.get("project_id"):
                if not self._legacy_graph_recovery_allowed(int(job["project_id"])):
                    self.db.update_job(
                        job_id, status="failed", progress=100,
                        log="项目已绑定独立构建策略，不恢复旧 Stable 图谱任务",
                        error="固定策略项目禁止旧 Stable 图谱恢复",
                    )
                    continue
                self.executor.submit(self._sync_project_graph, job_id, int(job["project_id"]))
                recovered += 1
            elif job["kind"] == "schema_reassessment" and job.get("project_id"):
                self.executor.submit(self._reassess_project_schema, job_id, int(job["project_id"]))
                recovered += 1
            elif job["kind"] == "community_report_build" and job.get("project_id"):
                report_run_id = str(metadata.get("report_run_id") or "")
                if not report_run_id:
                    self.db.update_job(
                        job_id, status="failed", progress=100,
                        log="社区报告任务缺少运行编号", error="任务无法恢复",
                    )
                    continue
                self.report_executor.submit(
                    self._run_knowledge_forge_community_reports,
                    job_id, int(job["project_id"]), report_run_id,
                )
                recovered += 1
            elif job["kind"] == "document_cleanup" and job.get("document_id"):
                document = self.db.get_document_including_deleted(int(job["document_id"]))
                if document:
                    self.cleanup_executor.submit(self._cleanup_deleted_document, job_id, document)
                    recovered += 1
                else:
                    self.db.update_job(job_id, status="failed", progress=100,
                                       log="待清理文档记录不存在", error="文档清理无法恢复")
            else:
                self.db.update_job(job_id, status="failed", progress=100,
                                   log="任务缺少恢复所需参数", error="任务无法恢复")
        if hasattr(self.db, "list_orphaned_graph_projects"):
            for project in self.db.list_orphaned_graph_projects():
                if not self._legacy_graph_recovery_allowed(int(project["id"])):
                    continue
                try:
                    job = self.db.create_graph_rebuild_job(int(project["id"]))
                    if not job.get("_coalesced"):
                        self.executor.submit(
                            self._sync_project_graph, str(job["id"]), int(project["id"]),
                        )
                    recovered += 1
                except ValueError:
                    # A recovered mutating job already owns this project's next generation.
                    continue
        if hasattr(self.db, "cleanup_indexing_failures"):
            self.db.cleanup_indexing_failures()
        if hasattr(self.db, "cleanup_audit_metrics"):
            self.db.cleanup_audit_metrics()
        return recovered

    def _recover_cancelled_document_staging(
        self, job: dict[str, Any],
    ) -> None:
        """Discard interrupted preparation staging before acknowledging cancel."""

        if str(job.get("kind") or "") not in {
            "content_preparation", "ingestion",
        }:
            return
        abandon = getattr(self.db, "abandon_job_document_revision", None)
        if abandon is None:
            return
        revision = abandon(
            str(job["id"]), reason="服务恢复时确认预处理已终止",
        )
        if not revision:
            return
        revision_id = str(revision.get("revision_id") or "")
        revision_manifest = (
            revision.get("manifest")
            if isinstance(revision.get("manifest"), dict) else {}
        )
        project_id = int(
            revision.get("project_id") or job.get("project_id") or 0
        )
        document_hash = str(
            revision.get("content_hash")
            or revision_manifest.get("content_hash")
            or ""
        )
        project = (
            self.db.get_project(project_id)
            if project_id and hasattr(self.db, "get_project") else None
        )
        if project and document_hash and revision_id:
            project_key = Config.project_storage_key(
                project_id, str(project.get("name") or ""),
            )
            self._discard_staged_process_payload(
                Config.chunks_root
                / project_key
                / f"{document_hash[:16]}-{revision_id}.json"
            )
            self._discard_staged_media_directory(
                Config.media_root
                / project_key
                / document_hash[:16]
                / revision_id
            )
        document = None
        getter = getattr(self.db, "get_document_including_deleted", None)
        if getter is not None and revision.get("document_id") is not None:
            document = getter(int(revision["document_id"]))
        if document and not document.get("is_deleted"):
            active = bool(document.get("active_revision_id"))
            self.db.update_document(
                int(document["id"]),
                status="indexed" if active else "uploaded",
                processing_stage=(
                    str(document.get("processing_stage") or "chunk_ready")
                    if active else "uploaded"
                ),
                error_message=None,
            )
            metadata = (
                document.get("metadata")
                if isinstance(document.get("metadata"), dict) else {}
            )
            staged = str(metadata.get("chunk_file") or "")
            if staged and revision_id and revision_id in Path(staged).name:
                self._discard_staged_process_payload(Path(staged))

    def _recover_cancelled_native_staging(
        self, job: dict[str, Any],
    ) -> None:
        """Close and clean only unpublished Native attempts after restart."""

        if str(job.get("kind") or "") != "graphrag_native_build":
            return
        project_id = int(job.get("project_id") or 0)
        cancel_runs = getattr(
            self.db, "cancel_unpublished_graphrag_native_runs", None,
        )
        if not project_id or cancel_runs is None:
            return
        for run in cancel_runs(
            project_id, reason="服务恢复时确认 GraphRAG Native 构建已终止",
        ):
            for relative in {
                str(run.get("staging_relative_path") or ""),
                str(run.get("bundle_relative_path") or ""),
            }:
                if not relative:
                    continue
                try:
                    self.graphrag_native._discard_cancelled_bundle_path(
                        self.graphrag_native.native_root / relative,
                    )
                except Exception:
                    LOGGER.exception(
                        "Unable to clean cancelled GraphRAG Native path: %s",
                        relative,
                    )

    def _recover_cancelled_graph_staging(
        self, job: dict[str, Any],
    ) -> None:
        """Release graph ownership left by a cancelled worker process."""

        release = getattr(self.db, "release_job_graph_leases", None)
        if callable(release):
            release(str(job["id"]))
        kind = str(job.get("kind") or "")
        strategy_key = str(job.get("strategy_key") or "")
        project_id = int(job.get("project_id") or 0)
        if project_id and (
            kind == "graph_sync"
            or (
                kind == "ingestion"
                and normalize_strategy(strategy_key) == STABLE_STRATEGY
            )
        ):
            # A Stable writer updates the shared physical namespace in place.
            # If its process stopped before acknowledging cancellation, readers
            # must remain fenced until a later complete rebuild replaces it.
            try:
                self.db.set_graph_state(
                    project_id,
                    GRAPH_STALE,
                    error="服务恢复时确认图谱构建已终止；未完成图谱未发布",
                )
            except LookupError:
                pass
            return
        if (
            kind != "ingestion"
            or strategy_key != KNOWLEDGE_FORGE_STRATEGY
            or not project_id
        ):
            return
        graph = self.db.get_project_strategy_graph(
            project_id, KNOWLEDGE_FORGE_STRATEGY,
        )
        if not graph or str(graph.get("status") or "") != "rebuilding":
            return
        self.db.cancel_strategy_graph_rebuild(
            project_id, KNOWLEDGE_FORGE_STRATEGY,
            int(graph.get("generation") or 0),
            "服务恢复时确认图谱构建已终止；未完成图谱未发布",
        )

    @staticmethod
    def _model_type(name: str) -> str:
        normalized = name.lower()
        return "embedding" if any(token in normalized for token in ("embed", "embedding", "bge", "gte", "e5")) else "llm"

    def sync_ollama_models(self) -> list[dict[str, Any]]:
        discovered = []
        for model in self.ollama.models():
            name = str(model.get("name") or "").strip()
            if name:
                try:
                    metadata = self.ollama.model_info(name)
                except Exception as exc:
                    LOGGER.warning("Ollama model metadata unavailable for %s: %s", name, exc)
                    metadata = {"context_length": 8192, "capabilities": []}
                model_type = (
                    "embedding"
                    if "embedding" in (metadata.get("capabilities") or [])
                    else self._model_type(name)
                )
                discovered.append(self.db.upsert_model(
                    name,
                    model_type,
                    base_url=Config.ollama_base_url,
                    metadata=metadata,
                ))
        return discovered

    @staticmethod
    def _metadata(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @classmethod
    def _graph_extraction_chunks(
        cls, chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Select graph inputs while leaving every chunk in the vector store.

        Chunk construction marks duplicate parent overviews and high-confidence
        navigation/reference material as retrieval-only.  Enhanced extraction
        must honor the same persisted flag as the stable path; otherwise it
        spends model calls on duplicate or non-factual text and can publish
        graph noise even though the structure UI says the chunk is excluded.
        Legacy rows without the additive flag remain eligible.
        """

        return [
            item for item in chunks
            if cls._metadata(item.get("metadata")).get(
                "graph_extraction_enabled", True,
            ) is not False
        ]

    @classmethod
    def _schema_evaluation_chunks(
        cls, chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return factual graph inputs suitable for logical-Schema review.

        Schema discovery is part of graph construction, so retrieval-only
        navigation/reference chunks must not influence it.  The explicit
        parent-role check preserves the historical exclusion for legacy rows
        created before ``graph_extraction_enabled`` was persisted.
        """

        return [
            item for item in cls._graph_extraction_chunks(chunks)
            if cls._metadata(item.get("metadata")).get("chunk_role")
            != "parent"
        ]

    def _model_context_limit(self, model_name: str, model_type: str) -> int:
        models = {item["name"]: item for item in self.db.list_models()}
        record = models.get(model_name)
        if record and record.get("model_type") != model_type:
            raise ValueError(f"请选择有效的 {'LLM' if model_type == 'llm' else 'Embedding'} 模型")
        metadata = self._metadata(record.get("metadata")) if record else {}
        try:
            metadata = self.ollama.model_info(model_name)
            self.db.upsert_model(
                model_name,
                model_type,
                provider=(record or {}).get("provider") or "ollama",
                base_url=(record or {}).get("base_url") or Config.ollama_base_url,
                metadata=metadata,
            )
        except Exception as exc:
            if not record:
                raise ValueError(f"无法读取模型 {model_name}：{exc}") from exc
            LOGGER.warning("Using cached context limit for %s: %s", model_name, exc)
        return max(1024, int(metadata.get("context_length") or 8192))

    def project_token_limits(self, llm_model: str, embedding_model: str) -> dict[str, int]:
        llm_context = self._model_context_limit(llm_model, "llm")
        embedding_context = self._model_context_limit(embedding_model, "embedding")
        return {
            "llm_context_max": llm_context,
            "embedding_context_max": embedding_context,
            "chunk_context_max": min(llm_context, embedding_context),
        }

    def _assert_project_strategy(
        self, project_id: int, expected_strategy: str | None = None,
    ) -> str:
        """Resolve the fixed project strategy and reject cross-strategy calls."""

        checker = getattr(self.db, "assert_project_construction_strategy", None)
        if checker is None:
            # Narrow unit-test doubles created before the binding control plane
            # may not implement this persistence method.  The real Database
            # always does and therefore always fails closed.
            if expected_strategy is None:
                raise RuntimeError("项目策略校验接口不可用")
            return normalize_strategy(expected_strategy)
        return str(checker(project_id, expected_strategy))

    def create_project(
        self,
        name: str,
        description: str,
        llm_model: str = "",
        embedding_model: str = "",
        *,
        strategy_key: str,
        knowledge_language: str = CHINESE,
    ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("项目名称不能为空")
        strategy_key = normalize_strategy(strategy_key)
        knowledge_language = normalize_knowledge_language(
            knowledge_language, default=CHINESE,
        )
        if strategy_key not in {KNOWLEDGE_FORGE_STRATEGY, GRAPHRAG_NATIVE_STRATEGY}:
            raise ValueError("请选择 Knowledge Forge 或 GraphRAG Native")
        project = self.db.create_project(
            name, (description or "").strip(), llm_model or Config.default_llm_model,
            embedding_model or Config.default_embedding_model,
            strategy_key=strategy_key,
            knowledge_language=knowledge_language,
        )
        try:
            limits = self.project_token_limits(project["llm_model"], project["embedding_model"])
            project = self.db.update_project(
                int(project["id"]),
                model_max_tokens=min(self.DEFAULT_MODEL_MAX_TOKENS, limits["llm_context_max"]),
                chunk_max_tokens=min(self.DEFAULT_CHUNK_MAX_TOKENS, limits["chunk_context_max"]),
            )
        except Exception as exc:
            LOGGER.warning("Using default project token limits: %s", exc)
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
            self.schemas.ensure_base(int(project["id"]))
            self.db.ensure_project_strategy_graphs(int(project["id"]))
        return self.db.get_project(int(project["id"])) or project

    def bind_existing_project_strategy(
        self, project_id: int, strategy_key: str, *, expected_generation: int,
    ) -> dict[str, Any]:
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        row = self.db.bind_project_construction_strategy(
            project_id, strategy_key, expected_generation=expected_generation,
        )
        if str(row["selected_strategy"]) == KNOWLEDGE_FORGE_STRATEGY:
            self.schemas.ensure_base(project_id)
            self.db.ensure_project_strategy_graphs(project_id)
        return row

    def update_project_settings(
        self,
        project_id: int,
        llm_model: str,
        embedding_model: str,
        model_max_tokens: int,
        chunk_max_tokens: int,
        knowledge_language: str | None = None,
    ) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        available = {item["name"]: item["model_type"] for item in self.db.list_models()}
        if llm_model and available.get(llm_model) != "llm":
            raise ValueError("请选择有效的 LLM 模型")
        if embedding_model and available.get(embedding_model) != "embedding":
            raise ValueError("请选择有效的 Embedding 模型")
        selected_llm = llm_model or Config.default_llm_model
        selected_embedding = embedding_model or Config.default_embedding_model
        selected_knowledge_language = normalize_knowledge_language(
            (
                knowledge_language
                if knowledge_language is not None
                else project.get("knowledge_language") or "auto"
            ),
        )
        limits = self.project_token_limits(selected_llm, selected_embedding)
        if not 1024 <= model_max_tokens <= limits["llm_context_max"]:
            raise ValueError(
                f"模型 Token 上限必须在 1024 到 {limits['llm_context_max']} 之间"
            )
        if not 64 <= chunk_max_tokens <= min(model_max_tokens, limits["chunk_context_max"]):
            raise ValueError(
                "单切片 Token 上限必须在 64 到 "
                f"{min(model_max_tokens, limits['chunk_context_max'])} 之间"
            )
        return self.db.update_project(
            project_id,
            llm_model=selected_llm,
            embedding_model=selected_embedding,
            model_max_tokens=model_max_tokens,
            chunk_max_tokens=chunk_max_tokens,
            knowledge_language=selected_knowledge_language,
        )

    def update_project_models(self, project_id: int, llm_model: str, embedding_model: str) -> dict[str, Any]:
        """Compatibility wrapper for callers that only update model choices."""
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        return self.update_project_settings(
            project_id,
            llm_model,
            embedding_model,
            int(project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS),
            int(project.get("chunk_max_tokens") or self.DEFAULT_CHUNK_MAX_TOKENS),
            str(project.get("knowledge_language") or "auto"),
        )

    def save_upload(
        self,
        project_id: int,
        filename: str,
        body: bytes,
        *,
        requested_parser: str = "",
        requested_strategy: str = "",
        chunk_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if not project: raise LookupError("项目不存在")
        strategy_key = self._assert_project_strategy(project_id)
        source_name = Path(filename or "").name; suffix = Path(source_name).suffix.lower()
        requested_parser = requested_parser or "auto"
        requested_strategy = (
            "auto" if strategy_key == GRAPHRAG_NATIVE_STRATEGY
            else requested_strategy or "auto"
        )
        validate_processing_request(source_name, requested_parser, requested_strategy)
        if not body: raise ValueError("上传文件为空")
        if len(body) > Config.upload_max_bytes: raise ValueError(f"文件超过 {Config.upload_max_bytes // 1024 // 1024} MB 限制")
        tmp_path = Config.docs_root / f".upload-{uuid.uuid4().hex}"; tmp_path.write_bytes(body)
        try:
            file_hash = Config.sha256_file(tmp_path); existing = self.db.find_document_by_hash(project_id, file_hash)
            if existing and not existing.get("is_deleted"):
                raise ValueError(f"该项目已存在相同内容的文档：{existing['original_filename']}")
            project_dir = Config.docs_root / Config.project_storage_key(int(project["id"]), project["name"])
            project_dir.mkdir(parents=True, exist_ok=True)
            stored_name = f"{file_hash[:16]}_{Config.safe_name(Path(source_name).stem)}{suffix}"
            target = project_dir / stored_name; shutil.move(str(tmp_path), target)
            values = {"project_id": project_id, "doc_hash": file_hash, "original_filename": source_name,
                "stored_filename": stored_name, "storage_path": Config.file_url(target), "local_path": str(target),
                "file_size_bytes": target.stat().st_size, "mime_type": mimetypes.guess_type(source_name)[0] or "application/octet-stream",
                "requested_parser": requested_parser, "requested_chunk_strategy": requested_strategy,
                "chunk_config": chunk_config or {},
                "metadata": {"project_key": project["project_key"]}}
            if existing and existing.get("is_deleted"):
                restored = self.db.update_document(int(existing["id"]), **{**values, "status": "uploaded", "error_message": None,
                    "is_deleted": False, "deleted_at": None, "indexed_at": None,
                    "active_revision_id": None,
                    "effective_parser": None, "effective_chunk_strategy": None, "parse_quality": {},
                    "routing_reason": None, "processing_stage": "uploaded", "structure_ready_at": None,
                    "chunks_ready_at": None, "graph_ready_at": None})
                if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
                    self.db.ensure_document_strategy_builds(int(restored["id"]), reset=True)
                else:
                    self._mark_graphrag_native_source_changed(
                        project_id, "文档恢复，native source snapshot 已失效",
                    )
                return restored
            document = self.db.create_document(values)
            if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
                self.db.ensure_document_strategy_builds(int(document["id"]), reset=True)
            else:
                self._mark_graphrag_native_source_changed(
                    project_id, "新增文档，native source snapshot 已失效",
                )
            return document
        finally:
            tmp_path.unlink(missing_ok=True)

    def submit_ingestion(
        self,
        document_id: int,
        *,
        batch_id: str = "",
        strategy_key: str = STABLE_STRATEGY,
        target_document_ids: list[int] | None = None,
    ) -> str:
        strategy_key = normalize_strategy(strategy_key)
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
            raise ValueError("GraphRAG Native 仅支持项目级全量构建，请使用“构建知识”")
        document = self.db.get_document(document_id)
        if not document: raise LookupError("文档不存在")
        target_documents = [document]
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY and target_document_ids:
            target_documents = []
            for target_id in dict.fromkeys(int(value) for value in target_document_ids):
                target = self.db.get_document(target_id)
                if not target or int(target["project_id"]) != int(document["project_id"]):
                    raise ValueError("Knowledge Forge 包含不存在或不属于当前项目的文档")
                target_documents.append(target)
            if document_id not in {int(item["id"]) for item in target_documents}:
                target_documents.insert(0, document)
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
            getattr(
                self.db,
                "assert_grounded_relation_ready",
                self.db.assert_grounded_relation_ready,
            )()
        self.db.ensure_project_strategy_graphs(int(document["project_id"]))
        for target in target_documents:
            self.db.ensure_document_strategy_builds(int(target["id"]))
        build = self.db.get_document_strategy_build(document_id, strategy_key) or {}
        if build.get("status") in {"queued", "processing"}:
            raise ValueError("该文档正在使用当前方案构建")
        if strategy_key == STABLE_STRATEGY and document["status"] == "processing":
            raise ValueError("该文档正在处理")
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
            active_knowledge_forge = self.db.latest_active_job(
                int(document["project_id"]), "ingestion", KNOWLEDGE_FORGE_STRATEGY,
            )
            if active_knowledge_forge:
                raise ValueError("当前项目的 Knowledge Forge 正在构建，请等待任务完成")
            active_stable = self.db.latest_active_job(
                int(document["project_id"]), "ingestion", STABLE_STRATEGY,
            )
            if active_stable:
                raise ValueError("当前项目的文档预处理或稳定方案正在构建，请等待任务完成")
        else:
            active_knowledge_forge = self.db.latest_active_job(
                int(document["project_id"]), "ingestion", KNOWLEDGE_FORGE_STRATEGY,
            )
            if active_knowledge_forge:
                raise ValueError("Knowledge Forge 正在构建，完成后才能更新稳定基线")
            # Stable and knowledge_forge are independent construction products.
            # A Stable rebuild must not preemptively mark a ready Knowledge Forge
            # snapshot as stale; source/document mutations have their own
            # invalidation boundaries.
        batch_id = batch_id or uuid.uuid4().hex
        project = self.db.get_project(int(document["project_id"])) or {}
        job = self.db.create_job(
            "ingestion",
            int(document["project_id"]),
            document_id,
            metadata={
                "batch_id": batch_id,
                "document_name": document["original_filename"],
                "strategy_key": strategy_key,
                "target_document_ids": [int(item["id"]) for item in target_documents],
                "target_document_names": [str(item["original_filename"]) for item in target_documents],
                "knowledge_language": str(
                    project.get("knowledge_language") or "auto"
                ),
            },
            strategy_key=strategy_key,
        )
        for target in target_documents:
            self.db.update_document_strategy_build(
                int(target["id"]), strategy_key, status="queued", latest_job_id=str(job["id"]),
                error_message="", started_at=None, completed_at=None,
            )
        if strategy_key == STABLE_STRATEGY and project and project.get("graph_status") == GRAPH_READY:
            self.db.set_graph_state(
                int(document["project_id"]), GRAPH_STALE,
                error="文档重新入库后图谱需要重建",
            )
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
            cancel_project_jobs = getattr(self.db, "cancel_project_jobs", None)
            if callable(cancel_project_jobs):
                cancel_project_jobs(
                    int(document["project_id"]), ["community_report_build"],
                    "Knowledge Forge 正在创建新图谱版本，旧版本社区报告任务已停止",
                )
            self.db.mark_strategy_graph_stale(
                int(document["project_id"]), strategy_key,
                "Knowledge Forge 事实将重新计算",
            )
            self.executor.submit(self._strategy_executor(strategy_key), str(job["id"]), document_id)
        elif strategy_key == STABLE_STRATEGY:
            self.executor.submit(self._strategy_executor(strategy_key), str(job["id"]), document_id)
        else:
            raise ValueError(f"策略不支持文档级构建: {strategy_key}")
        return str(job["id"])

    def submit_project_ingestion(
        self, project_id: int, *, strategy_key: str = STABLE_STRATEGY,
    ) -> dict[str, Any]:
        """Build every pending document for one project and construction strategy."""

        strategy_key = normalize_strategy(strategy_key)
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        self._assert_project_strategy(project_id, strategy_key)
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
            return self.submit_graphrag_native_build(project_id)
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
            reconcile_entity = getattr(
                self.db,
                "reconcile_knowledge_forge_entity_freshness",
                None,
            )
            if callable(reconcile_entity):
                reconcile_entity(
                    project_id,
                    code_version=GROUNDED_ENTITY_CODE_VERSION,
                    policy_set=_grounded_entity_freshness_policy_set(),
                )
            reconcile_relation = getattr(
                self.db,
                "reconcile_knowledge_forge_relation_freshness",
                None,
            )
            if callable(reconcile_relation):
                reconcile_relation(
                    project_id,
                    code_digest=LOADED_RELATION_CODE_DIGEST,
                    treatment_digest=grounded_relation_treatment_digest(),
                )
        documents = self.db.list_documents(project_id, strategy_key)
        pending = [
            item for item in documents
            if str(item.get("strategy_status") or "uploaded")
            not in {"queued", "processing", "indexed", "deleted"}
        ]
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY and not pending:
            graph_getter = getattr(
                self.db, "get_project_strategy_graph", None,
            )
            graph = (
                graph_getter(project_id, strategy_key)
                if callable(graph_getter) else None
            )
            # A project-wide Schema/policy upgrade can invalidate the graph
            # without changing source bytes or per-document completion rows.
            # In that state the public project build action must be able to
            # recompute the current snapshot instead of reporting that no
            # documents are pending. Reuse only already indexed documents;
            # queued/failed/deleted rows retain their ordinary state path.
            if str((graph or {}).get("status") or "") in {
                GRAPH_EMPTY, GRAPH_STALE, GRAPH_FAILED,
            }:
                pending = [
                    item for item in documents
                    if str(item.get("strategy_status") or "") == "indexed"
                    and not bool(item.get("is_deleted"))
                ]
        if not pending:
            raise ValueError("当前构建方案没有待处理文档")
        pending.sort(key=lambda item: int(item["id"]))
        batch_id = uuid.uuid4().hex
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
            job_ids = [self.submit_ingestion(
                int(pending[0]["id"]), batch_id=batch_id, strategy_key=strategy_key,
                target_document_ids=[int(item["id"]) for item in pending],
            )]
        else:
            job_ids = [
                self.submit_ingestion(
                    int(item["id"]), batch_id=batch_id, strategy_key=strategy_key,
                )
                for item in pending
            ]
        return {
            "job_id": job_ids[0],
            "job_ids": job_ids,
            "batch_id": batch_id,
            "document_count": len(pending),
        }

    def submit_graphrag_native_build(self, project_id: int) -> dict[str, Any]:
        """Queue one full-snapshot native build without creating document strategy rows."""

        self.db.assert_graphrag_native_ready()
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        documents = self.db.list_documents(project_id)
        if not documents:
            raise ValueError("GraphRAG Native 需要至少一份已预处理文档")
        if not all(
            bool(item.get("active_revision_id"))
            and str(item.get("status") or "") == "indexed"
            and bool(item.get("structure_ready_at"))
            for item in documents
        ):
            raise ValueError("GraphRAG Native 只能从所有已完成预处理的当前文档创建全量快照")
        active = self.db.latest_active_job(
            project_id, "graphrag_native_build", GRAPHRAG_NATIVE_STRATEGY,
        )
        if active:
            raise ValueError("当前项目已有 GraphRAG Native 全量构建正在执行")
        job = self.db.create_job(
            "graphrag_native_build", project_id, None,
            metadata={
                "strategy_key": GRAPHRAG_NATIVE_STRATEGY,
                "scope": "project_full_snapshot",
                "document_ids": [int(item["id"]) for item in documents],
                "knowledge_language": str(
                    project.get("knowledge_language") or "auto"
                ),
            },
            strategy_key=GRAPHRAG_NATIVE_STRATEGY,
        )
        self.executor.submit(
            self._strategy_executor(GRAPHRAG_NATIVE_STRATEGY), str(job["id"]), project_id,
        )
        return {
            "job_id": str(job["id"]),
            "job_ids": [str(job["id"])],
            "document_count": len(documents),
            "scope": "project_full_snapshot",
        }

    def cancel_graphrag_native_build(self, project_id: int) -> dict[str, Any]:
        """Request cancellation; the isolated subprocess observes it safely."""

        job = self.db.latest_active_job(
            project_id, "graphrag_native_build", GRAPHRAG_NATIVE_STRATEGY,
        )
        if not job:
            raise LookupError("当前项目没有可取消的 GraphRAG Native 构建")
        result = self.request_project_job_cancellation(
            project_id, job_id=str(job["id"]), expected_kind="graphrag_native_build",
        )
        return dict(result["jobs"][0])

    def request_project_job_cancellation(
        self,
        project_id: int,
        *,
        job_id: str = "",
        batch_id: str = "",
        expected_kind: str = "",
    ) -> dict[str, Any]:
        """Request safe cancellation of preparation or graph construction.

        The durable job stays ``running`` until the worker has left its current
        non-interruptible call and acknowledged a checkpoint.  This preserves
        project-wide serialization and prevents a replacement build from
        racing an old writer that is still unwinding.
        """

        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        self._assert_project_strategy(project_id)
        expected_kind = str(expected_kind or "").strip()
        if expected_kind and expected_kind not in self.USER_CANCELLABLE_JOB_KINDS:
            raise ValueError("该任务类型不支持安全终止")
        if not job_id and not batch_id and not expected_kind:
            raise ValueError("请指定要终止的任务")

        targets: list[dict[str, Any]] = []
        if batch_id:
            targets = [
                row for row in self.db.list_ingestion_batch_jobs(str(batch_id))
                if int(row.get("project_id") or 0) == int(project_id)
                and str(row.get("status") or "") in {"queued", "running"}
                and (
                    not expected_kind
                    or str(row.get("kind") or "") == expected_kind
                )
            ]
            # The page initially stores the batch's first job id.  In the
            # single-worker preparation queue that job may finish before the
            # user clicks stop, while a later sibling is now running.  The
            # durable batch id is the scope authority; reject only when the
            # supplied id belongs to a different batch/project, not merely
            # because it is no longer active.
            if job_id and all(str(row["id"]) != str(job_id) for row in targets):
                supplied = self.db.get_job(str(job_id))
                supplied_metadata = (
                    supplied.get("metadata")
                    if supplied and isinstance(supplied.get("metadata"), dict)
                    else {}
                )
                if (
                    not supplied
                    or int(supplied.get("project_id") or 0) != int(project_id)
                    or str(supplied_metadata.get("batch_id") or "")
                    != str(batch_id)
                    or (
                        expected_kind
                        and str(supplied.get("kind") or "") != expected_kind
                    )
                ):
                    raise ValueError("任务与批次不匹配")
        elif job_id:
            row = self.db.get_job(str(job_id))
            if row:
                targets = [row]
        else:
            row = self.db.latest_active_job(project_id, expected_kind)
            if row:
                targets = [row]

        if not targets:
            raise LookupError("当前没有可终止的任务")
        requested: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        publication_error: JobPublicationCommitted | None = None
        for target in targets:
            kind = str(target.get("kind") or "")
            if int(target.get("project_id") or 0) != int(project_id):
                raise LookupError("任务不存在或不属于当前项目")
            if expected_kind and kind != expected_kind:
                raise ValueError("任务类型与页面操作不匹配")
            label = (
                "预处理" if kind == "content_preparation"
                else "GraphRAG Native 构建" if kind == "graphrag_native_build"
                else "详细社区报告" if kind == "community_report_build"
                else "图谱构建"
            )
            try:
                requested.append(self.db.request_job_cancellation(
                    str(target["id"]), project_id=project_id,
                    allowed_kinds=list(self.USER_CANCELLABLE_JOB_KINDS),
                    reason=f"用户请求安全终止{label}",
                ))
            except JobPublicationCommitted as exc:
                # Keep the job active until its own worker finishes the
                # post-publication bookkeeping.  Marking it completed here
                # could hide an interrupted tail and strand per-document
                # status rows if the process then stopped.
                if not batch_id:
                    raise
                publication_error = exc
                skipped.append({
                    "job_id": str(target["id"]),
                    "reason": "publication_committed",
                })
            except ValueError:
                # A sibling can reach its terminal state after the batch
                # snapshot above but before its row is locked.  That race must
                # not prevent cancellation of the remaining active siblings.
                current = self.db.get_job(str(target["id"]))
                if (
                    batch_id
                    and current
                    and int(current.get("project_id") or 0) == int(project_id)
                    and str(current.get("status") or "")
                    not in {"queued", "running"}
                ):
                    skipped.append({
                        "job_id": str(target["id"]),
                        "reason": "already_finished",
                    })
                    continue
                raise
        if not requested:
            if publication_error is not None:
                raise publication_error
            raise LookupError("当前没有可终止的任务")
        return {
            "status": "cancellation_requested",
            "project_id": int(project_id),
            "job_id": str(requested[0]["id"]),
            "batch_id": str(batch_id or ""),
            "requested_count": len(requested),
            "skipped_count": len(skipped),
            "skipped_jobs": skipped,
            "jobs": requested,
        }

    def graphrag_native_status(self, project_id: int) -> dict[str, Any]:
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        self.db.assert_graphrag_native_ready()
        return self.graphrag_native_runtime.status(project_id)

    def graphrag_native_graph(
        self, project_id: int, *, offset: int = 0, limit: int = 200,
        include_isolates: bool = False,
    ) -> dict[str, Any]:
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        return self.graphrag_native_runtime.graph(
            project_id, offset=offset, limit=limit, include_isolates=include_isolates,
        )

    def _current_knowledge_forge_snapshot_reference(
        self, project_id: int,
    ) -> dict[str, Any]:
        """Resolve the only snapshot that Knowledge Forge may query."""

        self._assert_project_strategy(project_id, KNOWLEDGE_FORGE_STRATEGY)
        graph = self.db.get_project_strategy_graph(
            project_id, KNOWLEDGE_FORGE_STRATEGY,
        )
        if not graph:
            raise GraphNotReadyError("Knowledge Forge 图谱尚未初始化")
        if (
            str(graph.get("status") or "") != GRAPH_READY
            or int(graph.get("generation") or 0)
            != int(graph.get("ready_generation") or 0)
        ):
            raise GraphNotReadyError(
                str(graph.get("error_message") or "Knowledge Forge 图谱尚未就绪")
            )
        policy_summary = (
            graph.get("policy_summary")
            if isinstance(graph.get("policy_summary"), dict) else {}
        )
        entity_metrics = (
            policy_summary.get("entity_metrics")
            if isinstance(policy_summary.get("entity_metrics"), dict) else {}
        )
        entity_namespace_id = str(
            policy_summary.get("entity_namespace_id") or ""
        ).strip()
        relation_run_id = str(
            entity_metrics.get("relation_v2_run_id") or ""
        ).strip()
        if not entity_namespace_id or not relation_run_id:
            raise GraphNotReadyError(
                "Knowledge Forge 图谱缺少当前实体或关系快照引用，拒绝读取旧流程数据"
            )
        namespace = self.db.get_grounded_entity_namespace(entity_namespace_id)
        if (
            not namespace
            or int(namespace.get("project_id") or 0) != int(project_id)
            or str(namespace.get("status") or "") != GRAPH_READY
            or not bool(namespace.get("snapshot_complete"))
        ):
            raise GraphNotReadyError(
                "Knowledge Forge 实体快照不完整或不属于当前项目"
            )
        relation_run = self.db.get_grounded_relation_run(relation_run_id)
        if (
            not relation_run
            or int(relation_run.get("project_id") or 0) != int(project_id)
            or str(relation_run.get("status") or "") != GRAPH_READY
        ):
            raise GraphNotReadyError(
                "Knowledge Forge 关系快照不完整或不属于当前项目"
            )
        materialization_root = (
            (relation_run.get("fingerprint_components") or {}).get(
                "materialization_root", {}
            )
            if isinstance(
                relation_run.get("fingerprint_components"), dict,
            )
            else {}
        )
        if (
            str(materialization_root.get("code_tree") or "")
            != LOADED_RELATION_CODE_DIGEST
            or str(materialization_root.get("treatment") or "")
            != grounded_relation_treatment_digest()
        ):
            preservation = policy_summary.get("relation_artifact_preservation")
            preservation = (
                dict(preservation) if isinstance(preservation, dict) else {}
            )
            validate = getattr(
                self.db, "validate_grounded_entity_relation_artifact", None,
            )
            proof = (
                validate(entity_namespace_id, relation_run_id)
                if (
                    preservation.get("mode")
                    == "frozen_artifact_reprojection_v1"
                    and str(preservation.get("source_run_id") or "")
                    == relation_run_id
                    and callable(validate)
                )
                else {}
            )
            if not bool(proof.get("valid")):
                raise GraphNotReadyError(
                    "Knowledge Forge 关系抽取策略已升级，请重新构建后再查询"
                )
        return {
            "graph": graph,
            "policy_summary": policy_summary,
            "entity_namespace_id": entity_namespace_id,
            "relation_run_id": relation_run_id,
            "namespace": namespace,
            "relation_run": relation_run,
        }

    @staticmethod
    def _community_report_run_id(values: dict[str, Any]) -> str:
        payload = json.dumps(
            values, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        return "kfcommunityreport_" + hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:40]

    def submit_knowledge_forge_community_reports(
        self,
        project_id: int,
        *,
        automatic: bool = False,
        retry_failed: bool = False,
        knowledge_language: str | None = None,
        reference: dict[str, Any] | None = None,
        snapshot: dict[str, Any] | None = None,
        community_run: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Queue reports for exactly one current, published community snapshot."""

        if not Config.knowledge_forge_community_reports_enabled:
            if automatic:
                return {"enabled": False, "status": "disabled"}
            raise ValueError("Knowledge Forge 详细社区报告已关闭")
        if automatic and not Config.knowledge_forge_community_reports_auto_build:
            return {"enabled": True, "status": "manual"}
        reference = reference or self._current_knowledge_forge_snapshot_reference(
            project_id,
        )
        policy_summary = reference.get("policy_summary") or {}
        community_run_id = str(policy_summary.get("community_run_id") or "")
        community_input_hash = str(
            policy_summary.get("community_input_hash") or ""
        )
        graph_generation = int(
            reference["graph"].get("ready_generation") or 0
        )
        entity_namespace_id = str(reference["entity_namespace_id"])
        if not community_run_id or not community_input_hash:
            raise GraphNotReadyError("当前图谱缺少社区快照引用")
        community_run = community_run or self.db.get_knowledge_forge_community_run(
            project_id,
            graph_generation,
            entity_namespace_id,
            input_hash=community_input_hash,
        )
        if not community_run or str(community_run.get("run_id") or "") != community_run_id:
            raise GraphNotReadyError("当前图谱引用的社区快照不存在或已过期")
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        model_id = str(project.get("llm_model") or Config.default_llm_model)
        embedding_model_id = str(
            project.get("embedding_model") or Config.default_embedding_model
        )
        knowledge_language = normalize_knowledge_language(
            knowledge_language
            if knowledge_language is not None
            else policy_summary.get("knowledge_language") or CHINESE,
        )
        generator = self.community_report_generator
        prompt_hash_for = getattr(generator, "prompt_hash_for", None)
        report_prompt_hash = (
            str(prompt_hash_for(knowledge_language))
            if callable(prompt_hash_for)
            else str(generator.prompt_hash)
        )
        existing = self.db.get_matching_knowledge_forge_community_report_run(
            project_id=project_id,
            community_run_id=community_run_id,
            model_id=model_id,
            prompt_hash=report_prompt_hash,
            config_hash=generator.config_hash,
        )
        if existing and (
            not retry_failed
            or str(existing.get("status") or "") in {"queued", "running"}
        ):
            existing_config = (
                existing.get("config")
                if isinstance(existing.get("config"), dict) else {}
            )
            return {
                "enabled": True,
                "status": str(existing.get("status") or "queued"),
                "run_id": str(existing["run_id"]),
                "job_id": str(existing.get("job_id") or ""),
                "community_run_id": community_run_id,
                "expected_count": int(existing.get("expected_count") or 0),
                "eligible_count": int(existing.get("eligible_count") or 0),
                "skipped_count": int(existing.get("skipped_count") or 0),
                "deferred_count": int(existing_config.get("deferred_count") or 0),
                "coalesced": True,
            }
        retry_round = (
            int(existing.get("retry_round") or 0) + 1
            if existing and retry_failed else 0
        )
        snapshot = snapshot or self.db.load_grounded_entity_snapshot(
            entity_namespace_id,
        )
        snapshot_mentions = list(snapshot.get("mentions") or [])
        if not snapshot_mentions:
            snapshot_mentions = self.db.list_grounded_entity_mentions(
                entity_namespace_id,
            )
        snapshot = project_publishable_snapshot(
            snapshot,
            mentions=snapshot_mentions,
        )
        snapshot = enrich_knowledge_forge_snapshot(
            snapshot,
            mentions=snapshot_mentions,
            knowledge_language=knowledge_language,
        )
        plan = build_community_report_plan(
            community_run=community_run,
            snapshot=snapshot,
            policy=generator.policy,
            knowledge_language=knowledge_language,
            initial_limit=(
                Config.knowledge_forge_community_report_initial_limit
            ),
        )
        identity = {
            "project_id": int(project_id),
            "community_run_id": community_run_id,
            "community_input_hash": community_input_hash,
            "input_hash": str(plan["input_hash"]),
            "model_id": model_id,
            "prompt_hash": report_prompt_hash,
            "config_hash": generator.config_hash,
            "retry_round": retry_round,
        }
        run_id = self._community_report_run_id(identity)
        job_id = uuid.uuid4().hex
        items = []
        for item in plan["items"]:
            report_id = "kfcommunityreportitem_" + hashlib.sha256(
                f"{run_id}|{item['community_key']}".encode("utf-8")
            ).hexdigest()[:40]
            items.append({
                **item,
                "report_id": report_id,
                "max_attempts": generator.policy.max_attempts,
            })
        created = self.db.create_knowledge_forge_community_report_build({
            **identity,
            "run_id": run_id,
            "job_id": job_id,
            "graph_generation": graph_generation,
            "entity_namespace_id": entity_namespace_id,
            "embedding_model_id": embedding_model_id,
            "prompt_version": COMMUNITY_REPORT_PROMPT_VERSION,
            "expected_count": int(plan["expected_count"]),
            "eligible_count": int(plan["eligible_count"]),
            "skipped_count": int(plan["skipped_count"]),
            "config": {
                **generator.policy.as_dict(),
                "initial_limit": (
                    Config.knowledge_forge_community_report_initial_limit
                ),
                "fact_eligible_count": int(plan["fact_eligible_count"]),
                "deferred_count": int(plan["deferred_count"]),
                "knowledge_language": knowledge_language,
            },
            "automatic": automatic,
        }, items)
        run = created["run"]
        job = created.get("job") or {}
        if not created.get("_coalesced"):
            self.report_executor.submit(
                self._run_knowledge_forge_community_reports,
                str(job["id"]), project_id, str(run["run_id"]),
            )
        return {
            "enabled": True,
            "status": str(run.get("status") or "queued"),
            "run_id": str(run["run_id"]),
            "job_id": str(job.get("id") or run.get("job_id") or ""),
            "community_run_id": community_run_id,
            "expected_count": int(run.get("expected_count") or 0),
            "eligible_count": int(run.get("eligible_count") or 0),
            "skipped_count": int(run.get("skipped_count") or 0),
            "deferred_count": int(plan["deferred_count"]),
            "coalesced": bool(created.get("_coalesced")),
        }

    def _schedule_knowledge_forge_community_reports(
        self,
        project_id: int,
        *,
        knowledge_language: str | None = None,
        reference: dict[str, Any] | None = None,
        snapshot: dict[str, Any] | None = None,
        community_run: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.submit_knowledge_forge_community_reports(
                project_id,
                automatic=True,
                knowledge_language=knowledge_language,
                reference=reference,
                snapshot=snapshot,
                community_run=community_run,
            )
        except Exception as exc:
            # Reports are an enhancement layer.  Their durable job/status is
            # observable, but a scheduling failure cannot revoke a graph that
            # has already passed publication checks.
            LOGGER.warning(
                "Knowledge Forge community report scheduling failed project=%s: %s",
                project_id, exc,
            )

    def _knowledge_forge_community_report_run_is_current(
        self, project_id: int, run: dict[str, Any],
    ) -> bool:
        try:
            reference = self._current_knowledge_forge_snapshot_reference(project_id)
        except (LookupError, GraphNotReadyError):
            return False
        policy = reference.get("policy_summary") or {}
        return bool(
            int(reference["graph"].get("ready_generation") or 0)
            == int(run.get("graph_generation") or 0)
            and str(reference.get("entity_namespace_id") or "")
            == str(run.get("entity_namespace_id") or "")
            and str(policy.get("community_run_id") or "")
            == str(run.get("community_run_id") or "")
            and str(policy.get("community_input_hash") or "")
            == str(run.get("community_input_hash") or "")
        )

    def _index_knowledge_forge_community_report(
        self, run: dict[str, Any], report: dict[str, Any],
    ) -> None:
        report_id = str(report["report_id"])
        content = str(report.get("content") or "").strip()
        if not content:
            self.db.set_knowledge_forge_community_report_embedding_status(
                report_id, status="failed", error="社区报告内容为空",
            )
            return
        try:
            embedding_model = str(run["embedding_model_id"])
            vector = self.ollama.embed([content], embedding_model)[0]
            self.db.upsert_embedding({
                "project_id": int(run["project_id"]),
                "object_type": "community_report",
                "object_key": report_id,
                "content": content,
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "model_name": embedding_model,
                "embedding": vector,
                "metadata": self.vectors.embedding_metadata({
                    "run_id": str(run["run_id"]),
                    "community_run_id": str(run["community_run_id"]),
                    "community_key": str(report["community_key"]),
                    "graph_generation": int(run["graph_generation"]),
                }),
            })
            self.db.set_knowledge_forge_community_report_embedding_status(
                report_id, status="succeeded",
            )
        except Exception as exc:
            # Lexical report retrieval and UI rendering remain available.
            self.db.set_knowledge_forge_community_report_embedding_status(
                report_id, status="failed", error=str(exc),
            )
            LOGGER.warning(
                "Community report embedding failed report=%s: %s", report_id, exc,
            )

    def _run_knowledge_forge_community_reports(
        self, job_id: str, project_id: int, run_id: str,
    ) -> None:
        run: dict[str, Any] | None = None
        try:
            run = self.db.start_knowledge_forge_community_report_run(run_id)
            if str(run.get("status") or "") not in {"queued", "running"}:
                return
            generator = self.community_report_generator
            run_config = (
                run.get("config")
                if isinstance(run.get("config"), dict) else {}
            )
            knowledge_language = normalize_knowledge_language(
                run_config.get("knowledge_language") or "auto",
            )
            prompt_hash_for = getattr(generator, "prompt_hash_for", None)
            runtime_prompt_hash = (
                str(prompt_hash_for(knowledge_language))
                if callable(prompt_hash_for)
                else str(generator.prompt_hash)
            )
            runtime_matches_run = (
                str(run.get("prompt_version") or "")
                == COMMUNITY_REPORT_PROMPT_VERSION
                and str(run.get("prompt_hash") or "") == runtime_prompt_hash
                and str(run.get("config_hash") or "") == generator.config_hash
            )
            if not runtime_matches_run:
                reason = "社区报告生成策略已升级，旧版本任务已停止"
                self.db.terminate_knowledge_forge_community_report_run(
                    run_id, status="superseded", reason=reason,
                )
                self.db.update_job(
                    job_id, status="cancelled", progress=100, log=reason,
                    metadata={"stage": "superseded"},
                )
                return
            self._job(
                job_id, "running", 1, "开始生成详细社区报告",
                stage="community_reports", report_run_id=run_id,
            )
            state = self.db.get_knowledge_forge_community_report_state(
                project_id=project_id,
                community_run_id=str(run["community_run_id"]),
                graph_generation=int(run["graph_generation"]),
                entity_namespace_id=str(run["entity_namespace_id"]),
                community_input_hash=str(run["community_input_hash"]),
            )
            processed = sum(
                str(item.get("status") or "")
                in {"succeeded", "reused", "failed"}
                for item in state.get("reports") or []
            )
            eligible = max(1, int(run.get("eligible_count") or 0))
            finalized: dict[str, Any] = {}
            while True:
                job = self.db.get_job(job_id)
                metadata = (
                    job.get("metadata")
                    if job and isinstance(job.get("metadata"), dict) else {}
                )
                if (
                    not job
                    or str(job.get("status") or "") in {"cancelled", "failed"}
                    or bool(metadata.get("cancel_requested"))
                ):
                    reason = str(metadata.get("cancel_reason") or "社区报告生成已终止")
                    self.db.terminate_knowledge_forge_community_report_run(
                        run_id, status="cancelled", reason=reason,
                    )
                    if job and str(job.get("status")) not in {"cancelled", "failed"}:
                        self.db.update_job(
                            job_id, status="cancelled", progress=100,
                            log=reason, metadata={
                                "stage": "cancelled",
                                "cancel_acknowledged_at": datetime.now(
                                    timezone.utc,
                                ).isoformat(),
                            },
                        )
                    return
                if not self._knowledge_forge_community_report_run_is_current(
                    project_id, run,
                ):
                    reason = "图谱版本已变化，旧版本社区报告任务已停止"
                    self.db.terminate_knowledge_forge_community_report_run(
                        run_id, status="superseded", reason=reason,
                    )
                    self.db.update_job(
                        job_id, status="cancelled", progress=100, log=reason,
                        metadata={"stage": "superseded"},
                    )
                    return
                item = self.db.claim_knowledge_forge_community_report(run_id)
                if not item:
                    finalized = self.db.finalize_knowledge_forge_community_report_run(
                        run_id,
                    )
                    if int(finalized.get("_pending") or 0) > 0:
                        run_reader = getattr(
                            self.db,
                            "get_knowledge_forge_community_report_run",
                            None,
                        )
                        if callable(run_reader):
                            run = run_reader(run_id) or run
                            eligible = max(
                                1, int(run.get("eligible_count") or 0),
                            )
                        continue
                    break
                report_id = str(item["report_id"])
                reusable = self.db.find_reusable_knowledge_forge_community_report(
                    run_id, report_id,
                )
                if reusable:
                    completed = self.db.reuse_knowledge_forge_community_report(
                        report_id, str(reusable["report_id"]),
                    )
                    self._index_knowledge_forge_community_report(run, completed)
                    processed += 1
                else:
                    item = self.db.begin_knowledge_forge_community_report_model_attempt(
                        report_id,
                    )
                    attempt_no = int(item["attempt_count"])
                    context = (
                        item.get("input_context")
                        if isinstance(item.get("input_context"), dict) else {}
                    )
                    request_hash = community_report_request_hash(
                        str(run["model_id"]), context, knowledge_language,
                    )
                    attempt_id = "kfcommunityreportattempt_" + hashlib.sha256(
                        f"{report_id}|{attempt_no}".encode("utf-8")
                    ).hexdigest()[:40]
                    self.db.record_knowledge_forge_community_report_attempt_start(
                        attempt_id=attempt_id,
                        report_id=report_id,
                        run_id=run_id,
                        attempt_no=attempt_no,
                        model_id=str(run["model_id"]),
                        prompt_version=str(run["prompt_version"]),
                        request_hash=request_hash,
                    )
                    started = time.perf_counter()
                    try:
                        generated = self.community_report_generator.generate(
                            model_id=str(run["model_id"]),
                            input_context=context,
                            knowledge_language=knowledge_language,
                        )
                        latency_ms = int((time.perf_counter() - started) * 1000)
                        self.db.complete_knowledge_forge_community_report_attempt(
                            attempt_id,
                            status="succeeded",
                            raw_response_text=generated.get("raw_response_text") or "",
                            parsed_response=generated.get("parsed_response") or {},
                            diagnostics=generated.get("diagnostics") or {},
                            latency_ms=latency_ms,
                        )
                        completed = self.db.complete_knowledge_forge_community_report(
                            report_id, {**generated, "latency_ms": latency_ms},
                        )
                        self._index_knowledge_forge_community_report(run, completed)
                        processed += 1
                    except Exception as exc:
                        latency_ms = int((time.perf_counter() - started) * 1000)
                        diagnostics = self.ollama.last_call_diagnostics
                        self.db.complete_knowledge_forge_community_report_attempt(
                            attempt_id,
                            status="failed",
                            raw_response_text=self.ollama.last_raw_response_text,
                            parsed_response=(
                                self.ollama.last_parsed_json_response
                                if isinstance(
                                    self.ollama.last_parsed_json_response, dict,
                                ) else {}
                            ),
                            diagnostics=asdict(diagnostics) if diagnostics else {},
                            latency_ms=latency_ms,
                            error_category=type(exc).__name__,
                            error_message=str(exc),
                        )
                        outcome = self.db.fail_or_retry_knowledge_forge_community_report(
                            report_id,
                            error_category=type(exc).__name__,
                            error_message=str(exc),
                        )
                        if str(outcome.get("status") or "") == "failed":
                            processed += 1
                run_reader = getattr(
                    self.db, "get_knowledge_forge_community_report_run", None,
                )
                if callable(run_reader):
                    run = run_reader(run_id) or run
                    eligible = max(1, int(run.get("eligible_count") or 0))
                progress = min(99, 1 + int(processed / eligible * 98))
                log = (
                    f"详细社区报告进度 {processed}/{int(run.get('eligible_count') or 0)}"
                    if processed == 1 or processed % 10 == 0 else ""
                )
                self._job(
                    job_id, "running", progress, log,
                    stage="community_reports", report_run_id=run_id,
                    completed_reports=processed,
                    eligible_reports=int(run.get("eligible_count") or 0),
                )
            status = str(finalized.get("status") or "failed")
            if status == "failed":
                self.db.update_job(
                    job_id, status="failed", progress=100,
                    log="详细社区报告生成失败；知识图谱与基础检索仍可使用",
                    error=str(finalized.get("error_message") or "社区报告生成失败"),
                    metadata={"stage": "failed", "report_status": status},
                )
            else:
                self._job(
                    job_id, "completed", 100,
                    (
                        "详细社区报告已生成"
                        if status == "completed"
                        else "详细社区报告已完成，部分社区将继续使用基础摘要"
                        if status == "partial"
                        else "当前没有需要生成详细报告的社区"
                    ),
                    stage="completed", report_status=status,
                    completed_reports=int(finalized.get("completed_count") or 0),
                    failed_reports=int(finalized.get("failed_count") or 0),
                    reused_reports=int(finalized.get("reused_count") or 0),
                )
        except Exception as exc:
            LOGGER.exception("Knowledge Forge community reports failed: %s", run_id)
            if run:
                try:
                    self.db.terminate_knowledge_forge_community_report_run(
                        run_id, status="failed", reason=str(exc),
                    )
                except Exception:
                    LOGGER.exception("Could not close failed community report run")
            try:
                self.db.update_job(
                    job_id, status="failed", progress=100,
                    log="详细社区报告任务异常结束；知识图谱与基础检索不受影响",
                    error=str(exc), metadata={"stage": "failed"},
                )
            except (JobCancellationRequested, LookupError):
                pass

    def _public_community_reports_with_sources(
        self, project_id: int, raw_reports: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Project report rows to browser-safe reports with exact provenance."""

        cited_chunk_ids: set[int] = set()
        report_relations: dict[str, list[dict[str, Any]]] = {}
        for report in raw_reports:
            cited = {
                str(value) for value in report.get("cited_relation_keys") or []
                if str(value)
            }
            context = (
                report.get("input_context")
                if isinstance(report.get("input_context"), dict) else {}
            )
            relationships = [
                dict(item) for item in context.get("relationships") or []
                if isinstance(item, dict) and str(item.get("key") or "") in cited
            ]
            report_relations[str(report.get("report_id") or "")] = relationships
            cited_chunk_ids.update(
                int(item.get("chunk_id") or 0) for item in relationships
                if int(item.get("chunk_id") or 0) > 0
            )
        provenance_reader = getattr(self.db, "get_chunk_provenance", None)
        provenance = (
            provenance_reader(project_id, sorted(cited_chunk_ids))
            if callable(provenance_reader) else {}
        )
        public_reports: list[dict[str, Any]] = []
        for raw in raw_reports:
            public = _public_community_report(raw)
            if public is None:
                continue
            sources = [
                {
                    "relation_key": str(item.get("key") or ""),
                    "source_title": str(item.get("source_name") or ""),
                    "predicate_label": str(item.get("relation_label") or ""),
                    "target_title": str(item.get("target_name") or ""),
                    "evidence_text": str(item.get("evidence") or ""),
                    "source_info": _chunk_source_payload(
                        provenance.get(int(item.get("chunk_id") or 0)),
                    ),
                }
                for item in report_relations.get(
                    str(raw.get("report_id") or ""), [],
                )
            ]
            public_reports.append(
                _attach_public_community_report_sources(public, sources)
            )
        return public_reports

    def knowledge_forge_community_report_status(
        self, project_id: int, *, auto_start: bool = True,
    ) -> dict[str, Any]:
        reference = self._current_knowledge_forge_snapshot_reference(project_id)
        policy = reference.get("policy_summary") or {}
        community_run_id = str(policy.get("community_run_id") or "")
        community_input_hash = str(policy.get("community_input_hash") or "")
        if not Config.knowledge_forge_community_reports_enabled:
            return {"enabled": False, "status": "disabled", "reports": []}
        if auto_start:
            self._schedule_knowledge_forge_community_reports(
                project_id, reference=reference,
            )
        state = self.db.get_knowledge_forge_community_report_state(
            project_id=project_id,
            community_run_id=community_run_id,
            graph_generation=int(reference["graph"].get("ready_generation") or 0),
            entity_namespace_id=str(reference["entity_namespace_id"]),
            community_input_hash=community_input_hash,
        )
        run = state.get("run")
        raw_reports = [dict(item) for item in state.get("reports") or []]
        public_reports = self._public_community_reports_with_sources(
            project_id, raw_reports,
        )
        public_run = _public_community_report_run(run)
        deferred_count = sum(
            str(item.get("status") or "") == "deferred"
            for item in public_reports
        )
        if public_run is not None:
            public_run["deferred_count"] = deferred_count
        return {
            "enabled": True,
            "status": str((run or {}).get("status") or "not_started"),
            "run": public_run,
            "deferred_count": deferred_count,
            "reports": public_reports,
        }

    def _activate_deferred_knowledge_forge_reports(
        self,
        *,
        project_id: int,
        run_id: str,
        community_key: str = "",
        entity_keys: list[str] | None = None,
        limit: int = 1,
    ) -> dict[str, Any]:
        activator = getattr(
            self.db, "queue_deferred_knowledge_forge_community_reports", None,
        )
        if not callable(activator):
            updated = self.db.bump_knowledge_forge_community_report_priority(
                run_id=run_id,
                community_key=community_key,
                entity_keys=entity_keys,
            )
            return {"activated": 0, "prioritized": updated, "dispatch": False}
        result = activator(
            project_id=project_id,
            run_id=run_id,
            job_id=uuid.uuid4().hex,
            community_key=community_key,
            entity_keys=entity_keys,
            limit=limit,
        )
        if result.get("dispatch"):
            job = result.get("job") or {}
            self.report_executor.submit(
                self._run_knowledge_forge_community_reports,
                str(job["id"]), project_id, run_id,
            )
        return result

    def prioritize_knowledge_forge_community_report(
        self, project_id: int, community_key: str,
    ) -> dict[str, Any]:
        key = str(community_key or "").strip()
        if not key:
            raise ValueError("缺少社区标识")
        status = self.knowledge_forge_community_report_status(
            project_id, auto_start=True,
        )
        run = status.get("run") or {}
        if not run:
            return {"ok": True, "updated": 0, "status": status["status"]}
        activation = self._activate_deferred_knowledge_forge_reports(
            project_id=project_id,
            run_id=str(run["run_id"]),
            community_key=key,
            limit=1,
        )
        current_run = activation.get("run") or {}
        return {
            "ok": True,
            "updated": int(activation.get("prioritized") or 0),
            "activated": int(activation.get("activated") or 0),
            "status": str(current_run.get("status") or status["status"]),
            "job_id": str((activation.get("job") or {}).get("id") or ""),
        }

    def knowledge_forge_graph(
        self, project_id: int, *, limit: int = 1000,
        include_isolates: bool = True,
    ) -> dict[str, Any]:
        """Return the current ready Knowledge Forge snapshot for the web viewer.

        The browser reads the same immutable entity/relation snapshot that was
        materialized to the isolated graph namespace.  It does not query
        Hubble and never falls back to Stable data.
        """

        reference = self._current_knowledge_forge_snapshot_reference(project_id)
        graph = reference["graph"]
        policy_summary = reference.get("policy_summary") or {}
        graph_knowledge_language = normalize_knowledge_language(
            policy_summary.get("knowledge_language") or CHINESE,
        )
        entity_namespace_id = str(reference["entity_namespace_id"])
        snapshot = self.db.load_grounded_entity_snapshot(entity_namespace_id)
        snapshot = project_publishable_snapshot(
            snapshot,
            mentions=self.db.list_grounded_entity_mentions(entity_namespace_id),
        )
        snapshot = enrich_knowledge_forge_snapshot(
            snapshot,
            mentions=list(snapshot["mentions"]),
            knowledge_language=graph_knowledge_language,
        )
        canonicals = list(snapshot.get("canonicals") or [])
        relations = list(snapshot.get("relation_endpoints") or [])
        limit = max(1, min(int(limit), 100_000))

        canonical_by_key = {
            str(item.get("canonical_key") or ""): item
            for item in canonicals
            if str(item.get("canonical_key") or "")
        }
        degree: dict[str, int] = {key: 0 for key in canonical_by_key}
        valid_relations: list[dict[str, Any]] = []
        connected: set[str] = set()
        for relation in relations:
            source = str(relation.get("source_canonical_key") or "")
            target = str(relation.get("target_canonical_key") or "")
            if source not in canonical_by_key or target not in canonical_by_key:
                continue
            degree[source] = degree.get(source, 0) + 1
            degree[target] = degree.get(target, 0) + 1
            connected.update((source, target))
            valid_relations.append(relation)

        ordered = sorted(
            canonicals,
            key=lambda item: (
                0 if str(item.get("canonical_key") or "") in connected else 1,
                str(item.get("primary_type") or ""),
                str(item.get("normalized_name") or item.get("name") or ""),
                str(item.get("canonical_key") or ""),
            ),
        )
        if not include_isolates:
            ordered = [
                item for item in ordered
                if str(item.get("canonical_key") or "") in connected
            ]
        selected = ordered[:limit]
        selected_keys = {
            str(item.get("canonical_key") or "") for item in selected
        }
        visible_relations = [
            item for item in valid_relations
            if str(item.get("source_canonical_key") or "") in selected_keys
            and str(item.get("target_canonical_key") or "") in selected_keys
        ][:limit]

        # Several completed community experiments may exist for the same graph
        # generation and entity namespace.  The graph browser must use the
        # immutable run referenced by the published graph pointer, not simply
        # the most recently created experiment; reports are bound to that same
        # reference and cannot otherwise be joined safely.
        active_community_run_id = str(
            policy_summary.get("community_run_id") or ""
        )
        active_community_input_hash = str(
            policy_summary.get("community_input_hash") or ""
        )
        community_run = None
        if active_community_run_id and active_community_input_hash:
            candidate = self.db.get_knowledge_forge_community_run(
                project_id,
                int(graph.get("ready_generation") or 0),
                entity_namespace_id,
                input_hash=active_community_input_hash,
            )
            if (
                candidate
                and str(candidate.get("run_id") or "")
                == active_community_run_id
            ):
                community_run = candidate
        community_items = (
            [dict(item) for item in community_run.get("communities") or []]
            if community_run else []
        )
        community_run_metadata = (
            dict(community_run.get("metadata") or {}) if community_run else {}
        )
        community_topic_associations = [
            dict(item)
            for item in community_run_metadata.get(
                "community_topic_associations", []
            )
            if isinstance(item, dict)
            and str(item.get("source_community_key") or "")
            and str(item.get("target_community_key") or "")
        ]
        raw_topic_metrics = community_run_metadata.get(
            "community_topic_association_metrics", {}
        )
        community_topic_association_metrics = (
            dict(raw_topic_metrics) if isinstance(raw_topic_metrics, dict) else {}
        )
        community_items = enrich_community_descriptions(
            community_items, snapshot=snapshot,
            knowledge_language=graph_knowledge_language,
        )
        community_items.sort(key=lambda item: (
            int(item.get("level") or 0),
            int(item.get("community") or 0),
            str(item.get("key") or ""),
        ))
        for display_index, item in enumerate(community_items):
            item["display_index"] = display_index
            item["label"] = f"社区 {display_index}"
        report_run: dict[str, Any] = {}
        reports_by_community: dict[str, dict[str, Any]] = {}
        report_scheduler = getattr(
            self, "_schedule_knowledge_forge_community_reports", None,
        )
        report_state_reader = getattr(
            self.db, "get_knowledge_forge_community_report_state", None,
        )
        if (
            community_run
            and Config.knowledge_forge_community_reports_enabled
            and callable(report_scheduler)
            and callable(report_state_reader)
        ):
            report_scheduler(
                project_id,
                reference=reference,
                snapshot=snapshot,
                community_run=community_run,
            )
            report_state = report_state_reader(
                project_id=project_id,
                community_run_id=str(community_run["run_id"]),
                graph_generation=int(graph.get("ready_generation") or 0),
                entity_namespace_id=entity_namespace_id,
                community_input_hash=str(community_run["input_hash"]),
            )
            report_run = dict(report_state.get("run") or {})
            reports_by_community = {
                str(item.get("community_key") or ""): dict(item)
                for item in report_state.get("reports") or []
                if str(item.get("community_key") or "")
            }
        for community in community_items:
            report = reports_by_community.get(str(community.get("key") or ""))
            status = str((report or {}).get("status") or (
                "pending"
                if Config.knowledge_forge_community_reports_enabled else "disabled"
            ))
            community["report_status"] = status
            public_report = _public_community_report(report)
            community["community_report"] = public_report
            community["semantic_title"] = (
                str(public_report.get("title") or "")
                if public_report
                and str(public_report.get("status") or "")
                in {"succeeded", "reused"}
                else ""
            )
        singleton_communities = [
            item for item in community_items
            if int(item.get("size") or 0) == 1
            and int(item.get("relationship_count") or 0) == 0
        ]
        if singleton_communities:
            singleton_keys = [
                str(key)
                for item in singleton_communities
                for key in (item.get("entity_keys") or [])
            ]
            community_items = [
                item for item in community_items
                if item not in singleton_communities
            ] + [{
                "key": "knowledge-forge-unconnected",
                "label": "未连接实体",
                "level": 0,
                "community": 2_147_483_647,
                "size": len(singleton_keys),
                "relationship_count": 0,
                "entity_keys": singleton_keys,
                "relationship_keys": [],
                "summary": (
                    f"{len(singleton_keys)} 个规范实体当前没有已验证关系，"
                    "因此不参与连通社区；它们仍完整保留在图中。"
                ),
                "report_status": "skipped",
                "community_report": None,
            }]
        community_by_entity: dict[str, dict[str, Any]] = {}
        for community in sorted(
            community_items,
            key=lambda item: (
                int(item.get("level") or 0),
                int(item.get("community") or 0),
                str(item.get("key") or ""),
            ),
        ):
            for canonical_key in community.get("entity_keys") or []:
                community_by_entity.setdefault(str(canonical_key), community)

        nodes: list[dict[str, Any]] = []
        groups: dict[str, dict[str, Any]] = {}
        for item in selected:
            key = str(item.get("canonical_key") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            entity_type = str(item.get("primary_type") or "GenericEntity")
            community = community_by_entity.get(key)
            group_key = str(
                community.get("key") if community else entity_type
            )
            nodes.append({
                "id": key,
                "title": str(item.get("name") or key),
                "type": entity_type_label(
                    entity_type, graph_knowledge_language,
                ),
                "type_id": entity_type,
                "description": str(metadata.get("description") or ""),
                "description_source": str(
                    metadata.get("description_source")
                    or "mentions_and_verified_facts"
                ),
                "aliases": list(item.get("aliases") or []),
                "degree": int(degree.get(key) or 0),
                "group": group_key,
            })
            group = groups.setdefault(group_key, {
                "key": group_key,
                "label": str(
                    community.get("label") if community
                    else entity_type_label(
                        entity_type, graph_knowledge_language,
                    )
                ),
                "level": int(community.get("level") or 0) if community else 0,
                "community": (
                    int(community.get("community") or 0)
                    if community else None
                ),
                "display_index": (
                    int(community.get("display_index") or 0)
                    if community else None
                ),
                "size": 0,
                "relationship_count": 0,
                "summary": str(
                    community.get("summary")
                    if community
                    else (
                        "Canonical entities of type "
                        f"{entity_type_label(entity_type, graph_knowledge_language)}"
                        if graph_knowledge_language == ENGLISH else
                        f"{entity_type_label(entity_type)}类别的规范实体"
                    )
                ),
                "description": str(
                    (community.get("description") or community.get("summary"))
                    if community
                    else (
                        "Canonical entities of type "
                        f"{entity_type_label(entity_type, graph_knowledge_language)}"
                        if graph_knowledge_language == ENGLISH else
                        f"{entity_type_label(entity_type)}类别的规范实体"
                    )
                ),
                "description_source": str(
                    (community.get("description_source")
                     or "verified_facts_and_membership")
                    if community else "entity_type"
                ),
                "report_status": str(
                    community.get("report_status") if community else "skipped"
                ),
                "semantic_title": str(
                    community.get("semantic_title") if community else ""
                ),
                "community_report": (
                    community.get("community_report") if community else None
                ),
                "evidence_sources": [],
            })
            group["size"] = int(group["size"]) + 1

        provenance_reader = getattr(self.db, "get_chunk_provenance", None)
        chunk_provenance = (
            provenance_reader(
                project_id,
                [int(item.get("chunk_id") or 0) for item in visible_relations],
            )
            if callable(provenance_reader) else {}
        )
        edges: list[dict[str, Any]] = []
        for item in visible_relations:
            source = str(item.get("source_canonical_key") or "")
            target = str(item.get("target_canonical_key") or "")
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            relation_type = str(item.get("relation_type") or "related_to")
            relation_label = relation_display_label(
                relation_type, metadata, graph_knowledge_language,
            )
            evidence_text = str(item.get("evidence_text") or "")
            confidence = _optional_relation_confidence(metadata)
            source_group = str(
                community_by_entity.get(source, {}).get("key")
                or canonical_by_key[source].get("primary_type")
                or "GenericEntity"
            )
            target_group = str(
                community_by_entity.get(target, {}).get("key")
                or canonical_by_key[target].get("primary_type")
                or "GenericEntity"
            )
            if source_group == target_group and source_group in groups:
                groups[source_group]["relationship_count"] = (
                    int(groups[source_group]["relationship_count"]) + 1
                )
            source_payload = _chunk_source_payload(
                chunk_provenance.get(int(item.get("chunk_id") or 0)),
            )
            if source_group == target_group and source_group in groups and source_payload:
                known_sources = {
                    (
                        int(value.get("document_id") or 0),
                        str(value.get("page_label") or ""),
                        str(value.get("section_title") or ""),
                    )
                    for value in groups[source_group]["evidence_sources"]
                }
                identity = (
                    int(source_payload.get("document_id") or 0),
                    str(source_payload.get("page_label") or ""),
                    str(source_payload.get("section_title") or ""),
                )
                if identity not in known_sources and len(
                    groups[source_group]["evidence_sources"]
                ) < 8:
                    groups[source_group]["evidence_sources"].append(source_payload)
            edges.append({
                "id": str(item.get("relation_key") or ""),
                "source": source,
                "target": target,
                "source_title": str(canonical_by_key[source].get("name") or source),
                "target_title": str(canonical_by_key[target].get("name") or target),
                "predicate": relation_type,
                "predicate_label": relation_label,
                "predicate_id": relation_type,
                "description": str(
                    metadata.get("display_description") or relation_label
                ),
                "evidence_text": evidence_text,
                "evidence_count": _supported_evidence_count(
                    metadata, evidence_text,
                ),
                "verification_status": "SUPPORTED",
                # Publication is a verifier-backed categorical decision, not
                # a calibrated probability. Preserve an explicit score when
                # one exists, but never turn absence into a misleading 1.0.
                "confidence": confidence,
                "document_id": int(item.get("document_id") or 0),
                "chunk_id": int(item.get("chunk_id") or 0),
                "source_info": source_payload,
            })

        edge_by_key = {
            str(item.get("id") or ""): item for item in edges
            if str(item.get("id") or "")
        }
        for group in groups.values():
            report = group.get("community_report")
            if not isinstance(report, dict):
                continue
            cited = {
                str(value) for value in report.get("cited_relation_keys") or []
                if str(value)
            }
            sources = [
                {
                    "relation_key": str(edge.get("id") or ""),
                    "source_title": str(edge.get("source_title") or ""),
                    "predicate_label": str(edge.get("predicate_label") or ""),
                    "target_title": str(edge.get("target_title") or ""),
                    "evidence_text": str(edge.get("evidence_text") or ""),
                    "source_info": dict(edge.get("source_info") or {}),
                }
                for key in sorted(cited)
                for edge in [edge_by_key.get(key)]
                if edge
            ]
            _attach_public_community_report_sources(report, sources)

        return {
            "strategy": KNOWLEDGE_FORGE_STRATEGY,
            "generation": int(graph.get("ready_generation") or 0),
            "entity_namespace_id": entity_namespace_id,
            "community_run_id": str(
                community_run.get("run_id") if community_run else ""
            ),
            "community_ready": bool(community_run),
            "community_report_status": str(
                report_run.get("status") or (
                    "not_started"
                    if Config.knowledge_forge_community_reports_enabled
                    else "disabled"
                )
            ),
            "community_report_run_id": str(report_run.get("run_id") or ""),
            "community_topic_associations": community_topic_associations,
            "community_topic_association_metrics": (
                community_topic_association_metrics
            ),
            "total_communities": int(
                community_run.get("community_count") if community_run else 0
            ),
            "total_entities": len(canonicals),
            "total_relationships": len(valid_relations),
            "nodes": nodes,
            "edges": edges,
            "groups": sorted(groups.values(), key=lambda item: str(item["label"])),
            "truncated": len(selected) < len(ordered)
            or len(visible_relations) < len(valid_relations),
        }

    def graph_explorer(
        self, project_id: int, *, strategy_key: str, mode: str = "overview",
        community_key: str = "", entity_id: str = "", query: str = "",
        community_level: str = "focus", overview_relation_scope: str = "backbone",
        limit: int = 500,
    ) -> dict[str, Any]:
        """Return one bounded view for the common commercial graph browser."""

        strategy = self._assert_project_strategy(project_id, strategy_key)
        if strategy == KNOWLEDGE_FORGE_STRATEGY:
            payload = self.knowledge_forge_graph(
                project_id, limit=100_000, include_isolates=True,
            )
        elif strategy == GRAPHRAG_NATIVE_STRATEGY:
            payload = self.graphrag_native_runtime.explorer_payload(project_id)
        else:
            raise ValueError("该项目策略不支持图谱浏览")
        return build_graph_explorer_view(
            payload, mode=mode, community_key=community_key,
            entity_id=entity_id, query=query, community_level=community_level,
            overview_relation_scope=overview_relation_scope,
            limit=limit,
        )

    def graphrag_native_communities(self, project_id: int, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        return self.graphrag_native_runtime.communities(project_id, offset=offset, limit=limit)

    def graphrag_native_community_reports(self, project_id: int, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        return self.graphrag_native_runtime.community_reports(project_id, offset=offset, limit=limit)

    def graphrag_native_source_documents(
        self,
        project_id: int,
        *,
        document_id: int | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        return GraphRAGNativeInputBuilder(self).preview(
            project_id=project_id, document_id=document_id,
            offset=offset, limit=limit,
        )

    def graphrag_native_artifact_catalog(self, project_id: int) -> dict[str, Any]:
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        return self.graphrag_native_runtime.artifact_catalog(project_id)

    def graphrag_native_artifact_rows(
        self,
        project_id: int,
        artifact_key: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        return self.graphrag_native_runtime.artifact_rows(
            project_id, artifact_key, offset=offset, limit=limit,
        )

    def graphrag_native_query(self, project_id: int, *, mode: str, question: str) -> dict[str, Any]:
        self._assert_project_strategy(project_id, GRAPHRAG_NATIVE_STRATEGY)
        result = self.graphrag_native_runtime.query(
            project_id, mode=mode, question=question,
        )
        documents: dict[int, dict[str, Any] | None] = {}
        for context in result.get("context_records") or []:
            if not isinstance(context, dict):
                continue
            source_map = (
                context.get("source_map")
                if isinstance(context.get("source_map"), dict) else {}
            )
            try:
                document_id = int(source_map.get("document_id") or 0)
            except (TypeError, ValueError):
                document_id = 0
            if not document_id:
                continue
            if document_id not in documents:
                documents[document_id] = self.db.get_document(document_id)
            document = documents[document_id]
            if (
                not document
                or int(document.get("project_id") or 0) != int(project_id)
            ):
                continue
            context["source_document"] = {
                "id": document_id,
                "original_filename": str(
                    document.get("original_filename") or f"文档 {document_id}"
                ),
                "download_path": f"/api/documents/{document_id}/download",
            }
        return result

    def _project_preparation_plan(
        self,
        project: dict[str, Any],
        strategy_key: str,
    ) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
        """Plan only stale document revisions using the worker's exact contract."""

        strategy_key = normalize_strategy(strategy_key)
        documents = self.db.list_documents(int(project["id"]))
        pending: list[tuple[dict[str, Any], str]] = []
        for document in documents:
            manifest = (
                self._native_source_revision_manifest(document, project)
                if strategy_key == GRAPHRAG_NATIVE_STRATEGY
                else self._document_revision_manifest(document, project)
            )
            if not self._shared_content_is_current(document, manifest):
                pending.append((
                    document,
                    self._document_preparation_reason(document, manifest),
                ))
                continue
            if (
                strategy_key == KNOWLEDGE_FORGE_STRATEGY
                and not (
                    int(document.get("prepared_chunk_count") or 0) > 0
                    if "prepared_chunk_count" in document
                    else bool(self.db.list_document_chunks(int(document["id"])))
                )
            ):
                pending.append((document, "artifacts_missing"))
        return documents, pending

    def project_preparation_status(self, project_id: int) -> dict[str, Any]:
        """Return the read-only incremental preparation plan used by the UI."""

        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        strategy_key = self._assert_project_strategy(project_id)
        documents, pending = self._project_preparation_plan(project, strategy_key)
        reason_counts: dict[str, int] = {}
        pending_documents = []
        for document, reason in pending:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            pending_documents.append({
                "id": int(document["id"]),
                "original_filename": str(document.get("original_filename") or ""),
                "reason": reason,
            })
        new_count = int(reason_counts.get("new_document") or 0)
        return {
            "strategy_key": strategy_key,
            "document_count": len(documents),
            "ready_count": len(documents) - len(pending),
            "pending_count": len(pending),
            "new_count": new_count,
            "refresh_count": len(pending) - new_count,
            "reason_counts": reason_counts,
            "pending_documents": pending_documents,
        }

    def submit_project_preparation(self, project_id: int) -> dict[str, Any]:
        """Prepare only stale documents according to the immutable strategy."""

        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        strategy_key = self._assert_project_strategy(project_id)
        active = self.latest_active_mutating_job(project_id)
        if active:
            raise ValueError("当前项目已有任务正在执行，请等待任务完成")
        _, planned = self._project_preparation_plan(project, strategy_key)
        pending = [document for document, _reason in planned]
        if not pending:
            raise ValueError("当前项目的文档均已完成预处理")
        pending.sort(key=lambda item: int(item["id"]))
        batch_id = uuid.uuid4().hex
        job_ids: list[str] = []
        for document in pending:
            job = self.db.create_job(
                "content_preparation", project_id, int(document["id"]),
                metadata={
                    "batch_id": batch_id,
                    "document_name": document["original_filename"],
                    "target_document_ids": [int(document["id"])],
                    "strategy_key": strategy_key,
                },
                strategy_key=strategy_key,
            )
            job_id = str(job["id"])
            job_ids.append(job_id)
            self.executor.submit(
                self._prepare_document_content, job_id, int(document["id"]), strategy_key,
            )
        return {
            "job_id": job_ids[0], "job_ids": job_ids, "batch_id": batch_id,
            "document_count": len(pending),
        }

    def submit_graph_build(self, project_id: int) -> str:
        if not self.db.get_project(project_id): raise LookupError("项目不存在")
        self._assert_project_strategy(project_id, KNOWLEDGE_FORGE_STRATEGY)
        job = self.db.create_graph_rebuild_job(project_id)
        if not job.get("_coalesced"):
            self.executor.submit(self._sync_project_graph, job["id"], project_id)
        return str(job["id"])

    def submit_schema_reassessment(self, project_id: int) -> str:
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        self._assert_project_strategy(project_id, KNOWLEDGE_FORGE_STRATEGY)
        active = self.latest_active_mutating_job(project_id)
        if active:
            raise ValueError("当前项目已有任务正在执行，请等待任务完成后再重新评估 Schema")
        job = self.db.create_job(
            "schema_reassessment",
            project_id,
            strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        )
        self.executor.submit(self._reassess_project_schema, str(job["id"]), project_id)
        return str(job["id"])

    def add_entity_cannot_link(self, project_id: int, **values: Any) -> dict[str, Any]:
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        return self.db.add_entity_cannot_link(project_id, **values)

    def merge_entities(self, project_id: int, source_entity_id: int, target_entity_id: int,
                       **values: Any) -> dict[str, Any]:
        result = self.db.manual_merge_entities(
            project_id, source_entity_id, target_entity_id, **values,
        )
        result["derived_index_failures"] = self._refresh_entity_change_indexes(
            project_id,
            result.get("affected_entity_ids") or [source_entity_id, target_entity_id],
            int((result.get("project") or {}).get("graph_generation") or 0),
        )
        result["graph_job_id"] = self.submit_graph_build(project_id)
        return result

    def split_entity_mentions(self, project_id: int, source_entity_id: int,
                              target_entity_id: int, mention_ids: list[int],
                              **values: Any) -> dict[str, Any]:
        result = self.db.split_entity_mentions(
            project_id, source_entity_id, target_entity_id, mention_ids, **values,
        )
        result["derived_index_failures"] = self._refresh_entity_change_indexes(
            project_id,
            result.get("affected_entity_ids") or [source_entity_id, target_entity_id],
            int((result.get("project") or {}).get("graph_generation") or 0),
        )
        result["graph_job_id"] = self.submit_graph_build(project_id)
        return result

    def undo_entity_resolution_event(self, project_id: int, event_id: str,
                                     **values: Any) -> dict[str, Any]:
        result = self.db.undo_entity_resolution_event(project_id, event_id, **values)
        result["derived_index_failures"] = self._refresh_entity_change_indexes(
            project_id,
            result.get("affected_entity_ids") or [],
            int((result.get("project") or {}).get("graph_generation") or 0),
        )
        result["graph_job_id"] = self.submit_graph_build(project_id)
        return result

    def _refresh_entity_change_indexes(
        self,
        project_id: int,
        entity_ids: list[int],
        graph_generation: int,
    ) -> int:
        """Rebuild bounded entity/relation vectors after assignment changes."""

        ids = sorted({int(value) for value in entity_ids})
        if not ids:
            return 0
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        model = str(project.get("embedding_model") or Config.default_embedding_model)
        self.db.retire_derived_embeddings_for_entity_change(project_id, ids, model)
        failures = 0
        blocking_payload = {"blocks_graph_generation": int(graph_generation)}
        for entity in self.db.get_entities_by_ids(project_id, ids):
            object_key = str(entity["id"])
            payload = {
                **blocking_payload,
                "name": str(entity.get("name") or ""),
                "entity_type": str(entity.get("entity_type") or "GenericEntity"),
            }
            try:
                self.vectors.upsert_entity_embedding(project_id, entity, model)
                self.db.complete_indexing_failure(
                    project_id, "entity", object_key, model,
                )
            except Exception as exc:
                failures += 1
                self.db.record_indexing_failure({
                    "project_id": project_id,
                    "object_type": "entity",
                    "object_key": object_key,
                    "model_name": model,
                    "last_error": str(exc),
                    "payload": payload,
                })
                LOGGER.warning("Entity embedding refresh deferred for %s: %s", object_key, exc)
        for relation in self.db.list_project_relations_for_entities(project_id, ids):
            object_key = str(relation["id"])
            payload = {
                **blocking_payload,
                "relation_key": str(relation.get("relation_key") or f"relation:{object_key}"),
                "source_name": str(relation.get("source_name") or ""),
                "target_name": str(relation.get("target_name") or ""),
            }
            relation_for_index = {**relation, "relation_key": payload["relation_key"]}
            try:
                self.vectors.upsert_relation_embedding(
                    project_id,
                    int(relation.get("document_id") or 0),
                    int(relation.get("chunk_id") or 0),
                    relation_for_index,
                    payload["source_name"],
                    payload["target_name"],
                    model,
                )
                self.db.complete_indexing_failure(
                    project_id, "relation", object_key, model,
                )
            except Exception as exc:
                failures += 1
                self.db.record_indexing_failure({
                    "project_id": project_id,
                    "document_id": relation.get("document_id"),
                    "revision_id": relation.get("revision_id"),
                    "chunk_id": relation.get("chunk_id"),
                    "object_type": "relation",
                    "object_key": object_key,
                    "model_name": model,
                    "last_error": str(exc),
                    "payload": payload,
                })
                LOGGER.warning("Relation embedding refresh deferred for %s: %s", object_key, exc)
        return failures

    def _grounded_relation_runtime(
        self, model_gateway: Any,
    ) -> GroundedRelationRuntime:
        """Create the isolated relation runtime with the canonical source loader."""

        def load_processing_payload(document: dict[str, Any]) -> dict[str, Any] | None:
            try:
                return self._document_process_payload(int(document["id"]))[1]
            except (LookupError, RuntimeError):
                return None

        return GroundedRelationRuntime(
            self.db,
            model_gateway,
            default_context_tokens=self.DEFAULT_MODEL_MAX_TOKENS,
            processing_payload_loader=load_processing_payload,
        )

    def _replay_code_only_grounded_relation_namespace(
        self,
        *,
        namespace_id: str,
        project_id: int,
        source_run: dict[str, Any] | None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any] | None:
        """Recover a code-drifted relation snapshot from its audited calls.

        This path is deliberately narrow: all frozen inputs must reproduce
        exactly, every persisted model response must be consumed in order, and
        the replay gateway never reaches the live model.  Returning ``None``
        means a normal, fully recomputed relation run is required.
        """

        source_id = str((source_run or {}).get("id") or "").strip()
        if not source_id:
            return None
        runtime = self._grounded_relation_runtime(self.ollama)
        previous_snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
        previous_relation_keys = {
            str(item.get("relation_key") or "")
            for item in previous_snapshot.get("relation_endpoints") or []
            if str(item.get("relation_key") or "")
        }
        expected_fact_count = int(
            ((source_run or {}).get("metrics") or {}).get(
                "facts_materialization_eligible",
                ((source_run or {}).get("metrics") or {}).get(
                    "facts_after_normalization", 0,
                ),
            ) or 0
        )
        try:
            current = runtime._context(project_id, namespace_id)["fingerprint"]
            compatibility = assert_code_only_replay_compatible(
                dict(source_run or {}), current,
            )
            replay_gateway = PersistedRelationModelReplayGateway(
                self.db,
                source_id,
                runtime_model_gateway=self.ollama,
            )
            recovered = self._grounded_relation_runtime(
                replay_gateway,
            ).run_namespace(
                namespace_id=namespace_id,
                project_id=project_id,
                progress=progress,
                # Do not cohabit with an interrupted ordinary run whose ID is
                # derived from the same frozen fingerprint.  The salt remains
                # deterministic for this source/current replay pair, while
                # its immutable input fingerprint remains unchanged.
                run_id_salt=(
                    "persisted-call-replay-v1:" + source_id
                ),
            )
            current_snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
            recovered_relation_keys = {
                str(item.get("relation_key") or "")
                for item in current_snapshot.get("relation_endpoints") or []
                if str(item.get("relation_key") or "")
            }
            recovered_fact_count = int(
                (recovered.get("metrics") or {}).get(
                    "facts_materialization_eligible",
                    (recovered.get("metrics") or {}).get(
                        "facts_after_normalization", 0,
                    ),
                ) or 0
            )
            if (
                recovered_relation_keys != previous_relation_keys
                or recovered_fact_count != expected_fact_count
            ):
                # The relation endpoint table is the only mutable namespace
                # contribution of a completed replay.  Restore it before
                # failing closed so a rejected recovery never changes the
                # stale snapshot's observable facts.
                self.db.persist_grounded_entity_relation_endpoints(
                    namespace_id,
                    list(previous_snapshot.get("relation_endpoints") or []),
                )
                mismatch = {
                    "expected_relation_keys": len(previous_relation_keys),
                    "recovered_relation_keys": len(recovered_relation_keys),
                    "expected_fact_count": expected_fact_count,
                    "recovered_fact_count": recovered_fact_count,
                }
                self.db.finish_grounded_relation_run(
                    str(recovered["run_id"]),
                    status="stale",
                    metrics={
                        **dict(recovered.get("metrics") or {}),
                        "replay_parity_mismatch": mismatch,
                    },
                    input_fingerprint=str(recovered["input_fingerprint"]),
                    error_message="persisted-call replay output parity mismatch",
                )
                raise RelationReplayError(
                    "持久化调用重放结果与冻结关系快照不一致: "
                    + json.dumps(mismatch, sort_keys=True)
                )
        except RelationReplayError as exc:
            LOGGER.info(
                "Knowledge Forge relation replay is incompatible for %s: %s",
                namespace_id, exc,
            )
            return None
        recovered["replay_compatibility"] = compatibility
        return recovered

    def _preserve_frozen_grounded_relation_artifact(
        self,
        *,
        namespace_id: str,
        project_id: int,
        source_run: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Reuse an exact audited fact snapshot when replay batching drifted.

        Persisted-call replay is the strongest recovery path, but its model
        requests are intentionally batch-shaped.  A code-only operational
        improvement may preserve all frozen inputs and published facts while
        changing the order in which candidates are grouped into requests.
        In that case, rerunning the current scheduler cannot consume the old
        calls one-for-one even though the old endpoint artifact is complete.

        This fallback never performs inference and never accepts approximate
        output: it requires code-only input compatibility and a set-equality
        proof between the namespace endpoints and the referenced run's
        eligible normalized facts.  The graph remains explicitly labelled as
        a reprojected historical artifact, rather than as a current-runtime
        re-extraction.
        """

        source_id = str((source_run or {}).get("id") or "").strip()
        if not source_id:
            return None
        try:
            runtime = self._grounded_relation_runtime(self.ollama)
            current = runtime._context(project_id, namespace_id)["fingerprint"]
            compatibility = assert_code_only_replay_compatible(
                dict(source_run or {}), current,
            )
            validate = getattr(
                self.db, "validate_grounded_entity_relation_artifact", None,
            )
            if not callable(validate):
                return None
            artifact = validate(namespace_id, source_id)
            expected_fact_count = int(
                ((source_run or {}).get("metrics") or {}).get(
                    "facts_materialization_eligible",
                    ((source_run or {}).get("metrics") or {}).get(
                        "facts_after_normalization", 0,
                    ),
                ) or 0
            )
            if (
                not bool(artifact.get("valid"))
                or int(artifact.get("eligible_facts") or 0)
                != expected_fact_count
            ):
                raise RelationReplayError(
                    "冻结关系事实与实体快照不满足精确集合一致性: "
                    + json.dumps(artifact, sort_keys=True)
                )
        except RelationReplayError as exc:
            LOGGER.info(
                "Knowledge Forge frozen relation artifact cannot be reprojected for %s: %s",
                namespace_id, exc,
            )
            return None
        return {
            "run_id": source_id,
            "input_fingerprint": str(source_run.get("input_fingerprint") or ""),
            "metrics": dict(source_run.get("metrics") or {}),
            "replay_compatibility": compatibility,
            "artifact_preservation": {
                "mode": "frozen_artifact_reprojection_v1",
                "source_run_id": source_id,
                "proof": artifact,
            },
        }

    def _run_grounded_relation_namespace(
        self,
        *,
        namespace_id: str,
        project_id: int,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Invoke the only relation runtime permitted for Knowledge Forge."""

        return self._grounded_relation_runtime(self.ollama).run_namespace(
            namespace_id=namespace_id,
            project_id=project_id,
            progress=progress,
        )

    @staticmethod
    def _grounded_relation_endpoint_fingerprint(
        relation_endpoints: list[dict[str, Any]],
    ) -> str:
        """Identify the exact endpoint set used by the entity projection."""

        payload = sorted(
            (
                str(item.get("relation_key") or ""),
                str(item.get("source_canonical_key") or ""),
                str(item.get("target_canonical_key") or ""),
            )
            for item in relation_endpoints
        )
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _audit_grounded_entity_publishability(
        self,
        namespace_id: str,
        *,
        relation_endpoints: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Recompute the cheap canonical projection from immutable support."""

        snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
        mentions = self.db.list_grounded_entity_mentions(namespace_id)
        endpoint_rows = (
            list(relation_endpoints)
            if relation_endpoints is not None
            else list(snapshot.get("relation_endpoints") or [])
        )
        rows, metrics = aggregate_canonical_publishability(
            canonicals=list(snapshot.get("canonicals") or []),
            assignments=list(snapshot.get("assignments") or []),
            mentions=mentions,
            relation_endpoints=endpoint_rows,
        )
        updated = self.db.persist_grounded_entity_publishability_audit(
            namespace_id, rows,
        )
        if updated != len(rows):
            raise RuntimeError(
                "entity publishability audit did not update every canonical"
            )
        return {
            **metrics,
            "relation_endpoint_count": len(endpoint_rows),
            "relation_endpoint_fingerprint": (
                self._grounded_relation_endpoint_fingerprint(endpoint_rows)
            ),
            "mention_filter": self.db.grounded_entity_publishability_summary(
                namespace_id
            ),
        }

    def build_grounded_entity_namespace(
        self,
        project_id: int,
        snapshot_id: str = "",
        *,
        snapshot_manifest: dict[str, Any] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Build the independent span-first, evidence-grounded entity treatment."""

        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        frozen_manifest_documents = list((snapshot_manifest or {}).get("documents") or [])
        if frozen_manifest_documents:
            documents = []
            chunks = []
            for frozen in frozen_manifest_documents:
                document = self.db.get_document(int(frozen["document_id"]))
                if not document or int(document["project_id"]) != int(project_id):
                    raise RuntimeError("snapshot 冻结文档已不存在或项目不匹配")
                document = {
                    **document,
                    "active_revision_id": str(frozen["revision_id"]),
                    "doc_hash": str(frozen["content_hash"]),
                }
                documents.append(document)
                expected_chunks = {
                    int(item["chunk_id"]): str(item["content_hash"])
                    for item in frozen.get("chunks") or []
                }
                revision_chunks = self.db.list_revision_chunks(
                    int(document["id"]), str(frozen["revision_id"]),
                )
                actual = {int(item["id"]): item for item in revision_chunks}
                if set(actual) != set(expected_chunks):
                    raise RuntimeError("snapshot 冻结 Chunk 集合与不可变修订不一致")
                for chunk_id, expected_hash in expected_chunks.items():
                    if str(actual[chunk_id].get("content_hash") or "") != expected_hash:
                        raise RuntimeError("snapshot 冻结 Chunk content hash 已改变")
                    chunks.append(actual[chunk_id])
        else:
            documents = [
                item for item in self.db.list_documents(project_id)
                if item.get("status") == "indexed" and not item.get("is_deleted")
            ]
            chunks = []
            for document in documents:
                chunks.extend(self.db.list_document_chunks(int(document["id"])))
        documents.sort(key=lambda item: int(item["id"]))
        schema_version = int(
            ((snapshot_manifest or {}).get("project") or {}).get("schema_version") or 0
        )
        active_schema = (
            self.db.get_schema_version(project_id, schema_version)
            if schema_version else self.db.get_active_schema(project_id)
        )
        if not active_schema:
            raise RuntimeError("snapshot 冻结 Schema 不存在，无法构建 entity treatment")
        # Schema metadata is part of the entity/relation treatment input.  A
        # new Schema must never reuse a ready namespace produced under an older
        # version, otherwise newly induced relation families are silently
        # absent from the rebuilt graph.
        schema_fingerprint = hashlib.sha256(
            json.dumps({
                "version": int(active_schema.get("version") or 1),
                "schema_json": active_schema.get("schema_json") or {},
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        chunks.sort(key=lambda item: (
            int(item["document_id"]), int(item.get("chunk_index") or 0), int(item["id"]),
        ))
        source_chunks = chunks
        chunks = self._graph_extraction_chunks(source_chunks)
        chunks_by_document: dict[int, list[dict[str, Any]]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(int(chunk["document_id"]), []).append(chunk)

        def load_processing_payload(document: dict[str, Any]) -> dict[str, Any] | None:
            try:
                _, payload, _ = self._document_process_payload(int(document["id"]))
                return payload
            except (LookupError, RuntimeError):
                return None

        grounded_context = build_grounded_context_manifest(
            namespace_id=f"grounded-context:{project_id}:{snapshot_id or 'live'}",
            documents=documents,
            chunks_by_document=chunks_by_document,
            payload_loader=load_processing_payload,
        )
        structure_by_chunk = {
            int(unit.structure.get("chunk_id") or 0): dict(unit.structure)
            for unit in grounded_context.context_units
            if int(unit.structure.get("chunk_id") or 0) > 0
        }
        policy_set = {
            **ENTITY_EXTRACTION_POLICY_SET,
            "entity_publishability_policy_version": (
                ENTITY_PUBLISHABILITY_POLICY_VERSION
            ),
            "grounded_context_policy_version": grounded_context.policy_version,
            "grounded_context_source_map_policy": grounded_context.source_map_policy,
            "structure_artifact_hash": grounded_context.artifact_hash,
        }
        llm_model = str(project.get("llm_model") or Config.default_llm_model)
        embedding_model = str(project.get("embedding_model") or Config.default_embedding_model)
        frozen_inputs = [{
            "document_id": int(document["id"]),
            "revision_id": str(document.get("active_revision_id") or ""),
            "content_hash": str(document.get("doc_hash") or ""),
        } for document in documents]
        frozen_chunks = [{
            "chunk_id": int(chunk["id"]),
            "document_id": int(chunk["document_id"]),
            "revision_id": str(chunk.get("revision_id") or ""),
            "content_hash": str(chunk.get("content_hash") or ""),
            "graph_extraction_enabled": (
                self._metadata(chunk.get("metadata")).get(
                    "graph_extraction_enabled", True,
                ) is not False
            ),
        } for chunk in source_chunks]
        getattr(
            self.db,
            "assert_grounded_relation_ready",
            self.db.assert_grounded_relation_ready,
        )()
        code_version = GROUNDED_ENTITY_CODE_VERSION
        config_payload = {
            "policy_set": policy_set,
            "schema_version": int(active_schema.get("version") or 1),
            "schema_fingerprint": schema_fingerprint,
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "ollama_seed": Config.ollama_seed,
            "model_max_tokens": int(project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS),
            "grounded_inventory": {
                "enabled": bool(Config.entity_grounded_inventory_enabled),
                "segment_max_chars": int(
                    Config.entity_grounded_segment_max_chars
                ),
                "max_sentences_per_segment": int(
                    Config.entity_grounded_max_sentences_per_segment
                ),
                "max_items_per_segment": int(
                    Config.entity_grounded_max_items_per_segment
                ),
                "gleaning_rounds": int(
                    Config.entity_grounded_gleaning_rounds
                ),
            },
            "lexical_cap": min(Config.entity_lexical_max_k, Config.entity_lexical_top_k),
            "vector_cap": min(Config.entity_vector_max_k, Config.entity_vector_top_k),
            "reranker_cap": Config.entity_reranker_max_k,
            "code_version": code_version,
            "grounded_context": grounded_context.fingerprint_payload(),
        }
        config_hash = hashlib.sha256(
            json.dumps(config_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        input_payload = {
            "project_id": project_id,
            "snapshot_id": snapshot_id or "live",
            "schema_version": int(active_schema.get("version") or 1),
            "schema_fingerprint": schema_fingerprint,
            "documents": frozen_inputs,
            "chunks": frozen_chunks,
            "grounded_context": grounded_context.fingerprint_payload(),
            "config_hash": config_hash,
        }
        input_fingerprint = hashlib.sha256(
            json.dumps(input_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        namespace_id = "grounded_entity_" + hashlib.sha256(
            f"{snapshot_id or 'live'}|{input_fingerprint}".encode("utf-8")
        ).hexdigest()
        namespace = self.db.create_grounded_entity_namespace(
            project_id, namespace_id, policy_set,
            [int(item["id"]) for item in documents], snapshot_id or None,
            input_fingerprint=input_fingerprint,
            model_name=llm_model,
            embedding_model=embedding_model,
            config_hash=config_hash,
            code_version=code_version,
        )
        if namespace.get("snapshot_complete") and namespace.get("status") == "ready":
            snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
            # Entity work is safely reusable, but relation code is an
            # independent treatment.  A ready entity namespace must not
            # short-circuit a newly deployed relation classifier/verifier:
            # compare the immutable relation run's code fingerprint before
            # reusing its derived endpoints.
            relation_metrics = namespace.get("metrics") or {}
            relation_run_id = str(
                relation_metrics.get("relation_v2_run_id") or ""
            ).strip()
            relation_run = (
                self.db.get_grounded_relation_run(relation_run_id)
                if relation_run_id else None
            )
            stored_code_digest = str(
                ((relation_run or {}).get("fingerprint_components") or {})
                .get("materialization_root", {})
                .get("code_tree") or ""
            )
            stored_treatment_digest = str(
                ((relation_run or {}).get("fingerprint_components") or {})
                .get("materialization_root", {})
                .get("treatment") or ""
            )
            relation_refresh_required = (
                not relation_run
                or str(relation_run.get("status") or "") != "ready"
                or stored_code_digest != LOADED_RELATION_CODE_DIGEST
                or stored_treatment_digest
                != grounded_relation_treatment_digest()
            )
            publishability_after_relations: dict[str, Any] | None = None
            if relation_refresh_required:
                relation_v2 = self._replay_code_only_grounded_relation_namespace(
                    namespace_id=namespace_id,
                    project_id=project_id,
                    source_run=relation_run,
                    progress=progress,
                )
                if relation_v2 is None:
                    relation_v2 = self._run_grounded_relation_namespace(
                        namespace_id=namespace_id,
                        project_id=project_id,
                        progress=progress,
                    )
                # Relation publication is allowed to change independently of
                # the reusable entity namespace.  Recompute the cheap entity
                # projection from the newly persisted endpoint set before
                # exposing that set: otherwise an entity that was previously
                # retrieval-only can become a relation endpoint while keeping
                # its stale ``graph_publishable=False`` audit, producing a
                # dangling edge at the publication boundary.
                persisted = self.db.load_grounded_entity_snapshot(namespace_id)
                publishability_after_relations = (
                    self._audit_grounded_entity_publishability(
                        namespace_id,
                        relation_endpoints=list(
                            persisted.get("relation_endpoints") or []
                        ),
                    )
                )
                # The audit updates canonical metadata in PostgreSQL; reload
                # so the artifact hash and subsequent graph projection are
                # computed from the same atomically coherent entity/edge view.
                persisted = self.db.load_grounded_entity_snapshot(namespace_id)
                artifact_payload = {
                    "namespace_id": namespace_id,
                    "input_fingerprint": input_fingerprint,
                    "structure_artifact_hash": grounded_context.artifact_hash,
                    "canonicals": [
                        str(item["canonical_key"])
                        for item in persisted["canonicals"]
                    ],
                    "assignments": [{
                        "mention_key": str(item["mention_key"]),
                        "canonical_key": str(item["canonical_key"]),
                    } for item in persisted["assignments"]],
                    "relation_endpoints": [
                        str(item["relation_key"])
                        for item in persisted["relation_endpoints"]
                    ],
                }
                artifact_hash = hashlib.sha256(
                    json.dumps(
                        artifact_payload, sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                refreshed_metrics = {
                    **relation_metrics,
                    "entity_publishability": {
                        **dict(
                            relation_metrics.get("entity_publishability") or {}
                        ),
                        "after_relations": publishability_after_relations,
                    },
                    "relation_endpoints": len(
                        persisted.get("relation_endpoints") or ()
                    ),
                    "relation_v2_run_id": relation_v2["run_id"],
                    "relation_v2_fingerprint": relation_v2["input_fingerprint"],
                    "relation_v2": relation_v2["metrics"],
                }
                self.db.finalize_grounded_entity_namespace(
                    namespace_id,
                    metrics=refreshed_metrics,
                    artifact_hash=artifact_hash,
                )
                snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
            # A relation-only refresh reuses the immutable entity namespace.
            # Older refresh code updated the nested v2 run metrics but left
            # this top-level count at the prior run's value, so the strategy
            # graph and job correctly reported N facts while namespace audit
            # metadata reported an obsolete M. Reconcile this derived counter
            # even when the current relation snapshot itself is reusable.
            snapshot_namespace = snapshot.get("namespace") or {}
            snapshot_metrics = dict(
                snapshot_namespace.get("metrics") or {}
            )
            entity_publishability_metrics = dict(
                snapshot_metrics.get("entity_publishability") or {}
            )
            stored_after_relations = dict(
                entity_publishability_metrics.get("after_relations") or {}
            )
            current_endpoint_fingerprint = (
                self._grounded_relation_endpoint_fingerprint(
                    list(snapshot.get("relation_endpoints") or [])
                )
            )
            if (
                str(stored_after_relations.get(
                    "relation_endpoint_fingerprint"
                ) or "")
                != current_endpoint_fingerprint
            ):
                # This also repairs a process that stopped after persisting a
                # new relation artifact but before refreshing its entity
                # projection.  The endpoint fingerprint makes the check
                # model-free on ordinary reusable builds and avoids
                # repeating bounded JSON updates when nothing changed.
                publishability_after_relations = (
                    self._audit_grounded_entity_publishability(
                        namespace_id,
                        relation_endpoints=list(
                            snapshot.get("relation_endpoints") or []
                        ),
                    )
                )
                snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
                snapshot_namespace = snapshot.get("namespace") or {}
                snapshot_metrics = dict(
                    snapshot_namespace.get("metrics") or {}
                )
                entity_publishability_metrics = dict(
                    snapshot_metrics.get("entity_publishability") or {}
                )
                stored_after_relations = dict(
                    entity_publishability_metrics.get("after_relations") or {}
                )
            relation_endpoint_count = len(
                snapshot.get("relation_endpoints") or ()
            )
            publishability_metrics_changed = bool(
                publishability_after_relations is not None
                and stored_after_relations != publishability_after_relations
            )
            if (
                int(snapshot_metrics.get("relation_endpoints") or 0)
                != relation_endpoint_count
                or publishability_metrics_changed
            ):
                self.db.finalize_grounded_entity_namespace(
                    namespace_id,
                    metrics={
                        **snapshot_metrics,
                        "entity_publishability": {
                            **entity_publishability_metrics,
                            **({
                                "after_relations": (
                                    publishability_after_relations
                                ),
                            } if publishability_after_relations is not None else {}),
                        },
                        "relation_endpoints": relation_endpoint_count,
                    },
                    artifact_hash=str(
                        snapshot_namespace.get("artifact_hash") or ""
                    ),
                )
                snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
            if progress:
                total = int(namespace.get("expected_work_items") or len(chunks))
                progress(total, total, "复用已完成的增强实体关系快照")
            return {"namespace_id": namespace_id, **snapshot}

        work_items = []
        chunk_by_id = {int(item["id"]): item for item in chunks}
        for chunk in chunks:
            chunk_fingerprint = hashlib.sha256(
                "|".join((
                    namespace_id, str(chunk["id"]), str(chunk.get("revision_id") or ""),
                    str(chunk.get("content_hash") or ""),
                )).encode("utf-8")
            ).hexdigest()
            work_items.append({
                "id": "entity_shadow_work_" + chunk_fingerprint,
                "project_id": project_id,
                "document_id": int(chunk["document_id"]),
                "revision_id": str(chunk.get("revision_id") or ""),
                "chunk_id": int(chunk["id"]),
                "input_fingerprint": chunk_fingerprint,
            })
        self.db.seed_grounded_entity_work_items(namespace_id, work_items)
        summary = self.db.grounded_entity_work_summary(namespace_id)
        retried = 0
        if int(summary.get("failed") or 0):
            retried = self.db.requeue_terminal_grounded_entity_work(namespace_id)
            summary = self.db.grounded_entity_work_summary(namespace_id)
        if progress and retried:
            progress(
                int(summary["completed"]), int(summary["expected"]),
                f"重新排队 {retried} 个先前终止失败的知识块",
            )
        if progress:
            progress(
                int(summary["completed"]), int(summary["expected"]),
                "开始抽取实体并核验关系",
            )
        extractor = SpanFirstEntityExtractor(policy_set["entity_extraction_policy_version"])
        store = DatabaseGroundedStore(self.db)
        model_gateway = GroundedModelGateway(
            self.ollama,
            llm_model=llm_model,
            context_budget_tokens=int(
                project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS
            ),
        )

        grounded_inventory_enabled = bool(
            Config.entity_grounded_inventory_enabled
            and callable(getattr(
                self.schemas, "extract_grounded_mentions", None,
            ))
        )
        extraction_model_digest = ""
        extraction_effective_context = 0
        if grounded_inventory_enabled:
            try:
                runtime = self.ollama.effective_model_context(llm_model)
                extraction_model_digest = str(runtime.digest or "").strip()
                extraction_effective_context = int(
                    runtime.effective_context_tokens or 0
                )
            except Exception as exc:
                # A missing digest disables reuse. It must never weaken or
                # block the ordinary extraction path.
                LOGGER.warning(
                    "Knowledge Forge exact extraction cache disabled: %s", exc,
                )

        def rerank(mention: dict[str, Any], candidates: list[dict[str, Any]]) -> list[float]:
            return self.resolver.reranker.score(
                {
                    "name": mention["surface"],
                    "type": mention.get("entity_type") or "GenericEntity",
                    "description": mention.get("evidence") or "",
                    "aliases": mention.get("aliases") or [],
                },
                [{
                    **candidate,
                    "entity_type": candidate.get("primary_type") or "GenericEntity",
                } for candidate in candidates],
            )

        resolver = GroundedEntityResolver(
            namespace=namespace_id,
            policy_version=policy_set["entity_resolution_policy_version"],
            store=store,
            models=model_gateway,
            reranker=rerank,
            must_links=self.db.list_grounded_entity_must_links(namespace_id),
            cannot_links=self.db.list_grounded_entity_cannot_links(namespace_id),
            lexical_cap=config_payload["lexical_cap"],
            vector_cap=config_payload["vector_cap"],
            reranker_cap=config_payload["reranker_cap"],
            reviewer_cap=Config.entity_llm_max_candidates,
            auto_merge_threshold=Config.entity_merge_threshold,
            review_threshold=Config.entity_llm_review_threshold,
            identity_profile=policy_set["entity_identity_profile"],
        )
        worker_id = f"grounded-entity:{uuid.uuid4().hex}"
        while True:
            work = self.db.claim_grounded_entity_work(namespace_id, worker_id)
            if not work:
                break
            chunk = chunk_by_id.get(int(work["chunk_id"]))
            try:
                summary = self.db.grounded_entity_work_summary(namespace_id)
                if progress:
                    chunk_label = str(
                        (chunk or {}).get("section_title")
                        or (chunk or {}).get("chunk_key")
                        or f"知识块 {work['chunk_id']}"
                    )
                    progress(
                        int(summary["completed"]), int(summary["expected"]),
                        f"正在处理 {chunk_label}（第 {int(work['attempt'])}/"
                        f"{int(work['max_attempts'])} 次）",
                    )
            except Exception as exc:
                # A cancellation checkpoint can fire immediately after a
                # durable work lease is claimed.  Release that lease before
                # propagating so a later retry does not wait for its expiry.
                self.db.fail_grounded_entity_work(
                    str(work["id"]), str(work["lease_token"]), str(exc),
                )
                raise
            if not chunk:
                status = self.db.fail_grounded_entity_work(
                    str(work["id"]), str(work["lease_token"]), "active Chunk 不存在",
                )
                summary = self.db.grounded_entity_work_summary(namespace_id)
                if progress:
                    progress(
                        int(summary["completed"]), int(summary["expected"]),
                        f"知识块不存在，已标记为 {status}",
                    )
                continue
            try:
                self.db.reset_grounded_entity_chunk_attempt(
                    namespace_id, int(chunk["id"]),
                )
                structure = structure_by_chunk.get(int(chunk["id"]), {})
                extraction_chunk = {
                    **chunk,
                    "metadata": {
                        **self._metadata(chunk.get("metadata")),
                        "structural_role": str(
                            structure.get("source_role") or "body"
                        ),
                        "structural_spans": list(
                            structure.get("structural_spans") or []
                        ),
                        "structure_policy_version": str(
                            structure.get("structure_policy_version") or ""
                        ),
                    },
                }
                grounded_inventory = grounded_inventory_enabled
                extraction_cache_status = "disabled"
                extraction_cache_key = ""
                extraction = None
                cache_request: dict[str, Any] = {}
                if grounded_inventory and extraction_model_digest:
                    cache_request = sanitize_unicode_data({
                        "cache_version": GROUNDED_ENTITY_EXTRACTION_CACHE_VERSION,
                        "project_id": int(project_id),
                        "document_id": int(chunk["document_id"]),
                        "revision_id": str(chunk.get("revision_id") or ""),
                        "chunk_id": int(chunk["id"]),
                        "chunk_key": str(chunk.get("chunk_key") or ""),
                        "content": str(chunk.get("content") or ""),
                        "metadata": extraction_chunk.get("metadata") or {},
                        "schema_fingerprint": schema_fingerprint,
                        "schema": active_schema.get("schema_json") or {},
                        "model_name": llm_model,
                        "model_digest": extraction_model_digest,
                        "effective_context_tokens": extraction_effective_context,
                        "context_budget_tokens": int(
                            project.get("model_max_tokens")
                            or self.DEFAULT_MODEL_MAX_TOKENS
                        ),
                        "max_output_tokens": int(
                            Config.entity_only_max_output_tokens
                        ),
                        "seed": int(Config.ollama_seed),
                        "segment_max_chars": int(
                            Config.entity_grounded_segment_max_chars
                        ),
                        "max_sentences_per_segment": int(
                            Config.entity_grounded_max_sentences_per_segment
                        ),
                        "max_items_per_segment": int(
                            Config.entity_grounded_max_items_per_segment
                        ),
                        "gleaning_rounds": int(
                            Config.entity_grounded_gleaning_rounds
                        ),
                        "extraction_policy_version": policy_set[
                            "entity_extraction_policy_version"
                        ],
                        "code_version": (
                            GROUNDED_ENTITY_MODEL_EXTRACTION_CONTRACT_VERSION
                        ),
                    })
                    cache_bytes = json.dumps(
                        cache_request,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    extraction_cache_key = hashlib.sha256(
                        cache_bytes
                    ).hexdigest()
                    cached = (
                        self.db.get_knowledge_forge_entity_extraction_cache(
                            project_id, extraction_cache_key,
                        )
                    )
                    if cached:
                        cached_output = cached.get("output_json")
                        cached_hash = hashlib.sha256(json.dumps(
                            cached_output,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")).hexdigest()
                        if (
                            isinstance(cached_output, dict)
                            and str(cached.get("request_fingerprint") or "")
                            == extraction_cache_key
                            and str(cached.get("output_hash") or "")
                            == cached_hash
                        ):
                            extraction = json.loads(json.dumps(
                                cached_output, ensure_ascii=False,
                            ))
                            extraction_cache_status = "hit"
                        else:
                            self.db.delete_knowledge_forge_entity_extraction_cache(
                                project_id, extraction_cache_key,
                            )
                            LOGGER.warning(
                                "Knowledge Forge discarded corrupt exact extraction cache %s",
                                extraction_cache_key,
                            )
                if extraction is None:
                    extraction = (
                        self.schemas.extract_grounded_mentions(
                            project, active_schema, extraction_chunk,
                        )
                        if grounded_inventory
                        else self.schemas.extract_entities_only(
                            project, active_schema, extraction_chunk,
                        )
                    )
                    if grounded_inventory and extraction_cache_key:
                        extraction_output_hash = hashlib.sha256(json.dumps(
                            extraction,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")).hexdigest()
                        self.db.persist_knowledge_forge_entity_extraction_cache(
                            cache_key=extraction_cache_key,
                            project_id=project_id,
                            document_id=int(chunk["document_id"]),
                            revision_id=str(chunk.get("revision_id") or ""),
                            chunk_id=int(chunk["id"]),
                            content_hash=str(chunk.get("content_hash") or ""),
                            schema_fingerprint=schema_fingerprint,
                            model_name=llm_model,
                            model_digest=extraction_model_digest,
                            extraction_policy_version=policy_set[
                                "entity_extraction_policy_version"
                            ],
                            code_version=(
                                GROUNDED_ENTITY_MODEL_EXTRACTION_CONTRACT_VERSION
                            ),
                            request_fingerprint=extraction_cache_key,
                            request_metadata={
                                key: cache_request[key]
                                for key in (
                                    "cache_version", "effective_context_tokens",
                                    "context_budget_tokens", "max_output_tokens",
                                    "seed", "segment_max_chars",
                                    "max_sentences_per_segment",
                                    "max_items_per_segment", "gleaning_rounds",
                                )
                            },
                            output=extraction,
                            output_hash=extraction_output_hash,
                        )
                        extraction_cache_status = "miss_stored"
                endpoint_coverage = {"entities": [], "_diagnostics": {}}
                if grounded_inventory:
                    endpoint_coverage = {
                        "entities": [],
                        "_diagnostics": {
                            "skipped": True,
                            "reason": "grounded_inventory_is_endpoint_complete",
                            "llm_calls_avoided": (
                                Config.entity_endpoint_gleaning_rounds + 1
                            ),
                        },
                    }
                else:
                    try:
                        endpoint_coverage = (
                            self.schemas.extract_relation_endpoint_mentions(
                                project,
                                active_schema,
                                extraction_chunk,
                                existing_entities=list(
                                    extraction.get("entities") or []
                                ),
                            )
                        )
                    except Exception as exc:
                        # This stage only improves endpoint recall.  Exact-span
                        # entities from the primary pass remain valid and the
                        # relation classifier backstop still runs.  Publication
                        # never bypasses the independent verifier.
                        LOGGER.warning(
                            "Knowledge Forge relation endpoint coverage skipped: %s",
                            exc,
                        )
                        endpoint_coverage = {
                            "entities": [],
                            "_diagnostics": {"error": str(exc)},
                        }
                raw_entities = [
                    *(extraction.get("entities") or []),
                    *(endpoint_coverage.get("entities") or []),
                    *policy_relation_endpoint_candidates(
                        str(chunk.get("content") or ""),
                        list(
                            (
                                active_schema.get("schema_json")
                                if isinstance(active_schema.get("schema_json"), dict)
                                else active_schema
                            ).get("relation_types") or []
                        ),
                    ),
                    *structured_relation_endpoint_candidates(
                        str(chunk.get("content") or ""),
                        list(structure.get("structural_spans") or []),
                    ),
                    *structured_list_item_endpoint_candidates(
                        str(chunk.get("content") or ""),
                        list(structure.get("structural_spans") or []),
                    ),
                    *structured_section_label_endpoint_candidates(
                        str(chunk.get("content") or ""),
                        list(structure.get("structural_spans") or []),
                    ),
                ]
                normalized = extractor.normalize(
                    project_scope=str(project.get("project_key") or project_id),
                    document_key=str(chunk.get("revision_id") or chunk["document_id"]),
                    chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
                    content=str(chunk.get("content") or ""),
                    raw_entities=raw_entities,
                    chunk_metadata=self._metadata(extraction_chunk.get("metadata")),
                )
                layout_mention_coverage = extractor.audit_layout_coverage(
                    content=str(chunk.get("content") or ""),
                    mentions=normalized.mentions,
                    type_assertions=normalized.type_assertions,
                    chunk_metadata=self._metadata(
                        extraction_chunk.get("metadata")
                    ),
                )
                endpoint_candidate_spans = {
                    (
                        int(item.get("mention_start")),
                        int(item.get("mention_end")),
                    )
                    for item in raw_entities
                    if item.get("endpoint_candidate") is True
                    and item.get("mention_start") is not None
                    and item.get("mention_end") is not None
                }
                identity_evidence_by_mention = {
                    item.mention_id: bounded_identity_evidence(
                        str(chunk.get("content") or ""),
                        int(item.start),
                        int(item.end),
                    ) or item.evidence
                    for item in normalized.mentions
                }
                mention_rows = [{
                    **asdict(item),
                    "evidence": identity_evidence_by_mention[item.mention_id],
                    "provenance": {
                        "input_fingerprint": work["input_fingerprint"],
                        "model_name": llm_model,
                        "config_hash": config_hash,
                        "code_version": code_version,
                        "physical_occurrence_key": item.occurrence_key,
                        "extraction_evidence": item.evidence,
                        "identity_evidence_policy": (
                            "neighboring-source-sentences-v1"
                        ),
                        "publishability_policy_version": (
                            item.publishability_policy_version
                        ),
                        "publishability_score": item.publishability_score,
                        "publishability_protected": (
                            item.publishability_protected
                        ),
                        "publishability_features": list(
                            item.publishability_features
                        ),
                        "occurrence_recovery": item.recovery_kind,
                        "endpoint_candidate": (
                            int(item.start), int(item.end)
                        ) in endpoint_candidate_spans,
                    },
                } for item in normalized.mentions]
                assertion_rows = [asdict(item) for item in normalized.type_assertions]
                sanitizer_rejected = [
                    *(extraction.get("_rejected_entities") or []),
                    *(endpoint_coverage.get("_rejected_entities") or []),
                ]
                rejected_rows = [{
                    **item,
                    "policy_version": str(
                        item.get("policy_version")
                        or policy_set["entity_extraction_policy_version"]
                    ),
                } for item in [*normalized.rejected, *sanitizer_rejected]]
                self.db.persist_grounded_entity_extraction(
                    namespace_id,
                    project_id=project_id,
                    document_id=int(chunk["document_id"]),
                    revision_id=str(chunk["revision_id"]),
                    chunk_id=int(chunk["id"]),
                    mentions=mention_rows,
                    type_assertions=assertion_rows,
                    rejected=rejected_rows,
                )
                alias_by_mention: dict[str, list[dict[str, Any]]] = {}
                for alias in normalized.aliases:
                    alias_by_mention.setdefault(alias.mention_id, []).append({
                        **asdict(alias),
                    })
                type_by_mention = {item.mention_id: item for item in normalized.type_assertions}
                raw_id_by_span = {
                    (
                        str(item.get("name") or ""),
                        int(item["mention_start"]) if item.get("mention_start") is not None else -1,
                        int(item["mention_end"]) if item.get("mention_end") is not None else -1,
                    ): str(item.get("mention_id") or item.get("id") or "")
                    for item in raw_entities
                }
                raw_to_mention: dict[str, str] = {}
                canonical_by_mention: dict[str, str] = {}
                prepared_mentions: list[tuple[Any, Any, dict[str, Any]]] = []
                for item in normalized.mentions:
                    type_assertion = type_by_mention[item.mention_id]
                    mention = {
                        "mention_id": item.mention_id,
                        "occurrence_key": item.occurrence_key,
                        "surface": item.surface,
                        "normalized_surface": item.normalized_surface,
                        "entity_type": type_assertion.entity_type,
                        "evidence": identity_evidence_by_mention[item.mention_id],
                        "confidence": item.confidence,
                        "document_id": int(chunk["document_id"]),
                        "revision_id": str(chunk["revision_id"]),
                        "chunk_id": int(chunk["id"]),
                        "start": item.start,
                        "end": item.end,
                        "policy_version": policy_set["entity_resolution_policy_version"],
                        "type_source": type_assertion.source,
                        "model_name": llm_model,
                        "aliases": [
                            alias["surface"]
                            for alias in alias_by_mention.get(item.mention_id, [])
                        ],
                        "endpoint_candidate": bool(
                            (int(item.start), int(item.end)) in endpoint_candidate_spans
                        ),
                        "recovery_kind": item.recovery_kind,
                    }
                    prepared_mentions.append((item, type_assertion, mention))

                # Identity resolution is order-sensitive because an earlier
                # exact mention can safely become a candidate for a later
                # alias/reference.  Model array order is not source order, so
                # enforce the immutable source-span order before assigning
                # canonicals.  This is structural and independent of language,
                # entity type, predicate, or document domain.
                prepared_mentions.sort(key=lambda value: (
                    bool(value[0].recovery_kind),
                    int(value[0].start), int(value[0].end),
                    str(value[0].mention_id),
                ))

                exact_results: dict[str, dict[str, Any]] = {}
                unresolved: list[dict[str, Any]] = []
                for _, _, mention in prepared_mentions:
                    exact_result = resolver.try_exact(mention)
                    if exact_result is None:
                        unresolved.append(mention)
                    else:
                        exact_results[str(mention["mention_id"])] = exact_result
                embedded: dict[str, list[float]] = {}
                if unresolved:
                    vectors = model_gateway.embed(
                        [resolver.identity_text(mention) for mention in unresolved],
                        embedding_model,
                    )
                    if len(vectors) != len(unresolved):
                        raise RuntimeError("grounded entity embedding 批次返回数量不匹配")
                    embedded = {
                        str(mention["mention_id"]): vector
                        for mention, vector in zip(unresolved, vectors)
                    }

                for item, type_assertion, mention in prepared_mentions:
                    mention_id = str(mention["mention_id"])
                    result = exact_results.get(mention_id)
                    if result is None:
                        # Earlier mentions in this chunk may have installed an exact alias.
                        result = resolver.try_exact(mention)
                    if result is None:
                        result = resolver.resolve_with_embedding(
                            mention,
                            embedding_model=embedding_model,
                            embedding=embedded[mention_id],
                        )
                    existing = next((
                        candidate for candidate in result.get("candidates") or []
                        if str(candidate["canonical_key"]) == result["canonical_key"]
                    ), None)
                    canonical = {
                        "canonical_key": result["canonical_key"],
                        # Mention surfaces remain exact source slices, including
                        # renderer-inserted soft line wraps. Canonical display
                        # names are presentation values and should not expose
                        # those PDF/OCR layout newlines in the graph UI.
                        "name": str(
                            (existing or {}).get("name")
                            or canonical_display_surface(item.surface)
                        ),
                        "normalized_name": str(
                            (existing or {}).get("normalized_name") or item.normalized_surface
                        ),
                        "primary_type": str(
                            (existing or {}).get("primary_type") or type_assertion.entity_type
                        ),
                        "aliases": sorted({
                            item.surface,
                            *((existing or {}).get("aliases") or []),
                            *mention["aliases"],
                        }),
                        "metadata": {
                            "type_assertion": asdict(type_assertion),
                            "identity_type_decoupled": True,
                            "resolution_decision": str(result.get("decision") or ""),
                            "quarantined": bool(result.get("quarantined")),
                        },
                    }
                    self.db.upsert_grounded_entity_canonical(
                        namespace_id, canonical, mention,
                        aliases=alias_by_mention.get(item.mention_id, []),
                        embedding_model=embedding_model,
                        embedding=result.get("embedding"),
                        decision=result["decision"], reason=result["reason"],
                        scores=result.get("scores") or {}, config_hash=config_hash,
                    )
                    canonical_by_mention[item.mention_id] = result["canonical_key"]
                    raw_id = raw_id_by_span.get((item.surface, item.start, item.end))
                    if raw_id:
                        raw_to_mention[raw_id] = item.mention_id

                output = {
                    "mentions": len(normalized.mentions),
                    "rejected": len(normalized.rejected),
                    "mention_publishability": dict(Counter(
                        item.filter_decision for item in normalized.mentions
                    )),
                    "layout_mentions_recovered": sum(
                        item.recovery_kind == "layout_whitespace_variant"
                        for item in normalized.mentions
                    ),
                    "publishability_rejections": dict(Counter(
                        str(item.get("reason") or "unknown")
                        for item in normalized.rejected
                    )),
                    "entity_extraction": dict(
                        extraction.get("_diagnostics") or {}
                    ),
                    "entity_extraction_cache": extraction_cache_status,
                    "endpoint_coverage_mentions": len(
                        endpoint_coverage.get("entities") or []
                    ),
                    "endpoint_coverage": dict(
                        endpoint_coverage.get("_diagnostics") or {}
                    ),
                    "layout_mention_coverage": layout_mention_coverage,
                    "relation_endpoints": 0,
                }
                output_hash = hashlib.sha256(
                    json.dumps(output, sort_keys=True).encode("utf-8")
                ).hexdigest()
                if not self.db.complete_grounded_entity_work(
                    str(work["id"]), str(work["lease_token"]), output, output_hash,
                ):
                    raise RuntimeError("grounded entity work lease 已失效")
                summary = self.db.grounded_entity_work_summary(namespace_id)
                if progress:
                    progress(
                        int(summary["completed"]), int(summary["expected"]),
                        f"知识块实体阶段完成：{output['mentions']} 个实体提及",
                    )
            except Exception as exc:
                status = self.db.fail_grounded_entity_work(
                    str(work["id"]), str(work["lease_token"]), str(exc),
                )
                summary = self.db.grounded_entity_work_summary(namespace_id)
                if progress:
                    label = "终止失败" if status == "terminal_failed" else "等待重试"
                    progress(
                        int(summary["completed"]), int(summary["expected"]),
                        f"知识块{label}：{exc}",
                    )

        publishability_before_relations = (
            self._audit_grounded_entity_publishability(namespace_id)
        )
        layout_mention_coverage = (
            self.db.grounded_entity_layout_coverage_summary(namespace_id)
        )
        relation_v2 = self._run_grounded_relation_namespace(
            namespace_id=namespace_id,
            project_id=project_id,
            progress=progress,
        )
        publishability_after_relations = (
            self._audit_grounded_entity_publishability(
                namespace_id,
                relation_endpoints=list(relation_v2.get("endpoint_rows") or []),
            )
        )
        persisted = self.db.load_grounded_entity_snapshot(namespace_id)
        mention_count = len(self.db.list_grounded_entity_mentions(namespace_id))
        canonical_count = len(persisted["canonicals"])
        artifact_payload = {
            "namespace_id": namespace_id,
            "input_fingerprint": input_fingerprint,
            "structure_artifact_hash": grounded_context.artifact_hash,
            "canonicals": [str(item["canonical_key"]) for item in persisted["canonicals"]],
            "assignments": [{
                "mention_key": str(item["mention_key"]),
                "canonical_key": str(item["canonical_key"]),
            } for item in persisted["assignments"]],
            "relation_endpoints": [str(item["relation_key"]) for item in persisted["relation_endpoints"]],
        }
        artifact_hash = hashlib.sha256(
            json.dumps(artifact_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        metrics = {
            "documents": len(documents),
            "chunks": len(chunks),
            "source_chunks": len(source_chunks),
            "retrieval_only_chunks": len(source_chunks) - len(chunks),
            "mentions": mention_count,
            "canonicals": canonical_count,
            "relation_endpoints": len(persisted["relation_endpoints"]),
            "duplicate_canonical_rate": None,
            "canonical_compression_rate": (
                0.0 if not mention_count
                else max(0.0, 1.0 - (canonical_count / mention_count))
            ),
            "relation_v2_run_id": relation_v2["run_id"],
            "relation_v2_fingerprint": relation_v2["input_fingerprint"],
            "relation_v2": relation_v2["metrics"],
            "entity_publishability": {
                "before_relations": publishability_before_relations,
                "after_relations": publishability_after_relations,
            },
            "layout_mention_coverage": layout_mention_coverage,
            "grounded_context": grounded_context.summary,
            "structure_artifact_hash": grounded_context.artifact_hash,
        }
        self.db.finalize_grounded_entity_namespace(
            namespace_id, metrics=metrics, artifact_hash=artifact_hash,
        )
        return {"namespace_id": namespace_id, **self.db.load_grounded_entity_snapshot(namespace_id)}

    def _refresh_replayed_relation_namespace(
        self,
        namespace_id: str,
        relation_v2: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach a replayed relation run to an unchanged entity namespace."""

        snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
        namespace = dict(snapshot.get("namespace") or {})
        previous_metrics = dict(namespace.get("metrics") or {})
        artifact_payload = {
            "namespace_id": namespace_id,
            "input_fingerprint": str(namespace.get("input_fingerprint") or ""),
            "structure_artifact_hash": str(
                previous_metrics.get("structure_artifact_hash") or ""
            ),
            "canonicals": [
                str(item["canonical_key"])
                for item in snapshot.get("canonicals") or []
            ],
            "assignments": [{
                "mention_key": str(item["mention_key"]),
                "canonical_key": str(item["canonical_key"]),
            } for item in snapshot.get("assignments") or []],
            "relation_endpoints": [
                str(item["relation_key"])
                for item in snapshot.get("relation_endpoints") or []
            ],
        }
        artifact_hash = hashlib.sha256(
            json.dumps(artifact_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.db.finalize_grounded_entity_namespace(
            namespace_id,
            metrics={
                **previous_metrics,
                "relation_endpoints": len(
                    snapshot.get("relation_endpoints") or ()
                ),
                "relation_v2_run_id": str(relation_v2["run_id"]),
                "relation_v2_fingerprint": str(
                    relation_v2["input_fingerprint"]
                ),
                "relation_v2": dict(relation_v2.get("metrics") or {}),
            },
            artifact_hash=artifact_hash,
        )
        return self.db.load_grounded_entity_snapshot(namespace_id)

    def recover_code_only_knowledge_forge_snapshot(
        self,
        project_id: int,
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Recover a stale Knowledge Forge graph without live model calls.

        The recovery is allowed only when the graph's referenced entity
        namespace is complete and the relation runtime can reproduce every
        frozen audited request.  It then writes a new physical graph generation
        and community partition from that recovered snapshot.
        """

        self._assert_project_strategy(project_id, KNOWLEDGE_FORGE_STRATEGY)
        graph = self.db.get_project_strategy_graph(
            project_id, KNOWLEDGE_FORGE_STRATEGY,
        )
        if not graph or str(graph.get("status") or "") != GRAPH_STALE:
            raise TreatmentMaterializationRequired(
                "当前 Knowledge Forge 图谱不是可恢复的 stale 快照"
            )
        policy_summary = graph.get("policy_summary")
        policy_summary = (
            dict(policy_summary) if isinstance(policy_summary, dict) else {}
        )
        entity_metrics = policy_summary.get("entity_metrics")
        entity_metrics = (
            dict(entity_metrics) if isinstance(entity_metrics, dict) else {}
        )
        namespace_id = str(
            policy_summary.get("entity_namespace_id") or ""
        ).strip()
        source_run_id = str(
            entity_metrics.get("relation_v2_run_id") or ""
        ).strip()
        if not namespace_id or not source_run_id:
            raise TreatmentMaterializationRequired(
                "stale 图谱缺少冻结实体或关系快照引用"
            )
        namespace = self.db.get_grounded_entity_namespace(namespace_id)
        source_run = self.db.get_grounded_relation_run(source_run_id)
        if (
            not namespace
            or int(namespace.get("project_id") or 0) != int(project_id)
            or str(namespace.get("status") or "") != GRAPH_READY
            or not bool(namespace.get("snapshot_complete"))
            or not source_run
            or int(source_run.get("project_id") or 0) != int(project_id)
        ):
            raise TreatmentMaterializationRequired(
                "stale 图谱的冻结实体或关系快照不完整"
            )
        namespace_policy = self._metadata(namespace.get("policy_set"))
        if str(
            namespace_policy.get("entity_publishability_policy_version") or ""
        ) != ENTITY_PUBLISHABILITY_POLICY_VERSION:
            raise TreatmentMaterializationRequired(
                "冻结实体快照早于当前发布资格策略，需要执行常规重建"
            )
        if progress:
            progress(0, 100, "正在校验冻结关系事实快照")
        # An exact, ready artifact is a stronger and less fragile recovery
        # source than replaying historical batch prompts: it is the actual
        # published fact set, checked against its eligible memberships.  Only
        # fall back to request-level replay if that set proof is unavailable.
        relation_v2 = self._preserve_frozen_grounded_relation_artifact(
            namespace_id=namespace_id,
            project_id=project_id,
            source_run=source_run,
        )
        if relation_v2 is None:
            if progress:
                progress(0, 100, "正在校验可重放的关系审计账本")
            relation_v2 = self._replay_code_only_grounded_relation_namespace(
                namespace_id=namespace_id,
                project_id=project_id,
                source_run=source_run,
                # Preserve the runtime's phase denominator.  The durable-job
                # wrapper performs the only 0-100 conversion; mapping twice made
                # long candidate-ledger progress appear frozen around 40%.
                progress=progress,
            )
        if relation_v2 is None:
            raise TreatmentMaterializationRequired(
                "当前关系运行与冻结审计不兼容，需要执行常规重建"
            )
        snapshot = self._refresh_replayed_relation_namespace(
            namespace_id, relation_v2,
        )
        snapshot = project_publishable_snapshot(
            snapshot,
            mentions=self.db.list_grounded_entity_mentions(namespace_id),
        )
        project = self.db.get_project(project_id)
        schema_record = self.db.get_active_schema(project_id)
        if not project or not schema_record:
            raise RuntimeError("Knowledge Forge 恢复时缺少项目或活动 Schema")
        recovery_language_setting = normalize_knowledge_language(
            project.get("knowledge_language") or "auto",
        )
        recovery_knowledge_language = resolve_knowledge_language(
            recovery_language_setting,
            [
                str(item.get("evidence_text") or item.get("surface") or "")
                for item in snapshot.get("mentions") or []
            ] + [
                str(item.get("evidence_text") or "")
                for item in snapshot.get("relation_endpoints") or []
            ],
        )
        snapshot = enrich_knowledge_forge_snapshot(
            snapshot,
            mentions=list(snapshot["mentions"]),
            knowledge_language=recovery_knowledge_language,
        )
        documents = [
            item for item in self.db.list_documents(project_id)
            if item.get("status") == "indexed"
            and item.get("active_revision_id")
            and not item.get("is_deleted")
        ]
        if not documents:
            raise RuntimeError("Knowledge Forge 恢复时没有当前已入库文档")
        state: dict[str, Any] | None = None
        lease: dict[str, Any] | None = None
        try:
            if progress:
                progress(78, 100, "正在写入恢复后的隔离图")
            state = self.db.start_strategy_graph_rebuild(
                project_id, KNOWLEDGE_FORGE_STRATEGY,
            )
            generation = int(state["generation"])
            graph_namespace_id = str(state["namespace_id"])
            metrics = dict(
                (snapshot.get("namespace") or {}).get("metrics") or {}
            )
            schema_discovery = bool(
                policy_summary.get("schema_discovery_during_build")
            )
            graph_policy = {
                "strategy": KNOWLEDGE_FORGE_STRATEGY,
                "schema_version": int(schema_record["version"]),
                "schema_discovery_during_build": schema_discovery,
                "knowledge_language_setting": recovery_language_setting,
                "knowledge_language": recovery_knowledge_language,
                "entity_policy": dict(GROUNDED_ENTITY_POLICY_SET),
                "entity_publishability_policy": (
                    ENTITY_PUBLISHABILITY_POLICY_VERSION
                ),
                "relation_policy": dict(GROUNDED_RELATION_POLICY_SET),
                "relation_v2_run_id": metrics.get("relation_v2_run_id") or "",
                "relation_v2_fingerprint": (
                    metrics.get("relation_v2_fingerprint") or ""
                ),
                "artifact_hash": (
                    (snapshot.get("namespace") or {}).get("artifact_hash") or ""
                ),
            }
            policy_hash = hashlib.sha256(json.dumps(
                graph_policy, ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")).hexdigest()
            lease = self.db.acquire_graph_lease(
                project_id,
                graph_namespace_id,
                generation,
                owner=f"strategy:{KNOWLEDGE_FORGE_STRATEGY}:replay:{uuid.uuid4().hex}",
                ttl_seconds=900,
            )
            mutation = GraphMutationContext(
                namespace_id=graph_namespace_id,
                project_id=project_id,
                generation=generation,
                policy_hash=policy_hash,
                lease_token=str(lease["lease_token"]),
            )
            chunks_by_document = {
                int(item["id"]): self.db.list_document_chunks(int(item["id"]))
                for item in documents
            }
            graph_stats = self.graph.replace_strategy_snapshot(
                project,
                documents,
                chunks_by_document,
                snapshot,
                strategy_key=KNOWLEDGE_FORGE_STRATEGY,
                mutation=mutation,
            )
            if progress:
                progress(90, 100, "正在重建图谱社区")
            community_run = self.knowledge_forge_communities.detect(
                project_id=project_id,
                graph_generation=generation,
                entity_namespace_id=namespace_id,
                snapshot=snapshot,
                config=CommunityConfig(
                    seed=42, max_cluster_size=10,
                    use_lcc=False, timeout_seconds=300,
                ),
                knowledge_language=recovery_knowledge_language,
            )
            completed = self.db.complete_strategy_graph_rebuild(
                project_id,
                KNOWLEDGE_FORGE_STRATEGY,
                generation,
                stats=graph_stats,
                policy_summary={
                    "mode": "grounded",
                    "policy_hash": policy_hash,
                    "schema_version": int(schema_record["version"]),
                    "schema_discovery_during_build": schema_discovery,
                    "knowledge_language_setting": recovery_language_setting,
                    "knowledge_language": recovery_knowledge_language,
                    "entity_namespace_id": namespace_id,
                    "entity_metrics": metrics,
                    "community_run_id": str(community_run["run_id"]),
                    "community_input_hash": str(community_run["input_hash"]),
                    "recovery": {
                        "mode": (
                            "frozen_artifact_reprojection"
                            if relation_v2.get("artifact_preservation")
                            else "persisted_model_call_replay"
                        ),
                        "source_run_id": source_run_id,
                        "compatibility": dict(
                            relation_v2.get("replay_compatibility") or {}
                        ),
                    },
                    **(
                        {
                            "relation_artifact_preservation": dict(
                                relation_v2["artifact_preservation"]
                            ),
                        }
                        if relation_v2.get("artifact_preservation")
                        else {}
                    ),
                },
                empty=not graph_stats["documents"],
            )
            if not completed:
                raise RuntimeError("Knowledge Forge 恢复图代次已过期")
            now = datetime.now(timezone.utc)
            for document in documents:
                self.db.ensure_document_strategy_builds(int(document["id"]))
                self.db.update_document_strategy_build(
                    int(document["id"]),
                    KNOWLEDGE_FORGE_STRATEGY,
                    status="indexed",
                    content_revision_id=document.get("active_revision_id"),
                    stats={
                        "entity_mentions": sum(
                            1 for mention in snapshot["mentions"]
                            if int(mention["document_id"]) == int(document["id"])
                        ),
                        "relations": sum(
                            1 for relation in snapshot.get("relation_endpoints") or []
                            if int(relation["document_id"]) == int(document["id"])
                        ),
                        "graph_generation": generation,
                    },
                    error_message="",
                    completed_at=now,
                )
            if progress:
                progress(100, 100, "已恢复当前知识图谱快照")
            self._schedule_knowledge_forge_community_reports(
                project_id,
                knowledge_language=recovery_knowledge_language,
                snapshot=snapshot,
                community_run=community_run,
            )
            return {
                "generation": generation,
                "stats": graph_stats,
                "relation_run_id": str(relation_v2["run_id"]),
                "replayed_source_run_id": source_run_id,
            }
        except Exception as exc:
            if state:
                self.db.fail_strategy_graph_rebuild(
                    project_id,
                    KNOWLEDGE_FORGE_STRATEGY,
                    int(state["generation"]),
                    str(exc),
                )
            raise
        finally:
            if lease:
                self.db.release_graph_lease(
                    str(lease["id"]), str(lease["lease_token"]),
                )

    def submit_code_only_knowledge_forge_recovery(
        self, project_id: int,
    ) -> dict[str, Any]:
        """Queue persisted-call snapshot recovery as a visible durable job."""

        self._assert_project_strategy(project_id, KNOWLEDGE_FORGE_STRATEGY)
        active = self.latest_active_mutating_job(project_id)
        if active:
            raise ValueError("当前项目已有运行中的任务，暂不能恢复图谱快照")
        job = self.db.create_job(
            "knowledge_forge_snapshot_recovery",
            project_id,
            metadata={"strategy_key": KNOWLEDGE_FORGE_STRATEGY},
            strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        )
        self.executor.submit(
            self._recover_code_only_knowledge_forge_snapshot_job,
            str(job["id"]),
            project_id,
        )
        return job

    @recorded_job("knowledge_forge_snapshot_recovery")
    def _recover_code_only_knowledge_forge_snapshot_job(
        self, job_id: str, project_id: int,
    ) -> None:
        try:
            self._job(job_id, "running", 1, "正在恢复可重放的 Knowledge Forge 快照")
            last_progress = 1

            def update(done: int, total: int, message: str) -> None:
                nonlocal last_progress
                last_progress = max(
                    last_progress,
                    max(1, min(99, int(done * 99 / max(1, total)))),
                )
                self._job(
                    job_id,
                    "running",
                    last_progress,
                    message,
                    stage="graph",
                    strategy_key=KNOWLEDGE_FORGE_STRATEGY,
                )

            recovered = self.recover_code_only_knowledge_forge_snapshot(
                project_id, progress=update,
            )
            self._job(
                job_id,
                "completed",
                100,
                "Knowledge Forge 图谱已由已审计关系快照恢复",
                stage="completed",
                graph_ready=True,
                strategy_key=KNOWLEDGE_FORGE_STRATEGY,
                recovery=recovered,
            )
        except Exception as exc:
            LOGGER.exception("Knowledge Forge persisted-call recovery failed")
            self._job(
                job_id,
                "failed",
                100,
                f"Knowledge Forge 图谱恢复失败：{exc}",
                error=str(exc),
                stage="failed",
                strategy_key=KNOWLEDGE_FORGE_STRATEGY,
            )

    def process_policy_snapshot_work(
        self, snapshot: dict[str, Any], work: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one frozen grounded work item without publishing product facts."""

        manifest = self._metadata(snapshot.get("snapshot_manifest"))
        if str(manifest.get("manifest_hash") or "") != str(snapshot.get("manifest_hash") or ""):
            raise RuntimeError("policy snapshot manifest hash 不匹配")
        project_id = int(snapshot["project_id"])
        phase = str(work["phase"])
        if phase == "entity":
            result = self.build_grounded_entity_namespace(
                project_id, str(snapshot["id"]), snapshot_manifest=manifest,
            )
            namespace = result["namespace"]
            if namespace.get("status") != "ready" or not namespace.get("snapshot_complete"):
                raise RuntimeError("grounded entity namespace 尚未完整收敛")
            return {
                "namespace_id": result["namespace_id"],
                "artifact_hash": str(namespace.get("artifact_hash") or ""),
                "metrics": namespace.get("metrics") or {},
            }

        chunk_id = int(work.get("chunk_id") or 0)
        revision_id = str(work.get("revision_id") or "")
        chunk = self.db.get_revision_chunk(chunk_id, revision_id)
        if not chunk:
            raise RuntimeError("snapshot frozen Chunk 不存在")
        expected_hash = next((
            str(item["content_hash"])
            for document in manifest.get("documents") or []
            if int(document["document_id"]) == int(work["document_id"])
            and str(document["revision_id"]) == revision_id
            for item in document.get("chunks") or []
            if int(item["chunk_id"]) == chunk_id
        ), "")
        if not expected_hash or str(chunk.get("content_hash") or "") != expected_hash:
            raise RuntimeError("snapshot frozen Chunk hash 不一致")
        if phase == "control":
            return {
                "document_id": int(work["document_id"]),
                "revision_id": revision_id,
                "chunk_id": chunk_id,
                "content_hash": expected_hash,
            }
        if phase != "relation":
            raise ValueError(f"未知 policy snapshot phase: {phase}")

        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        schema_version = int((manifest.get("project") or {}).get("schema_version") or 1)
        schema_record = self.db.get_schema_version(project_id, schema_version)
        if not schema_record:
            raise RuntimeError("snapshot frozen Schema 不存在")
        if str(snapshot.get("mode")) == "end_to_end":
            namespace = self.db.get_grounded_entity_namespace_for_snapshot(str(snapshot["id"]))
            if not namespace or namespace.get("status") != "ready":
                raise RuntimeError("relation treatment 缺少完整 grounded entity endpoint")
            mentions = self.db.list_grounded_entity_mentions(str(namespace["id"]), chunk_id=chunk_id)
            grounded_state = self.db.load_grounded_entity_snapshot(str(namespace["id"]))
            canonical_by_mention = {
                str(item["mention_key"]): str(item["canonical_key"])
                for item in grounded_state.get("assignments") or []
            }
            entities = [{
                "mention_id": str(item["mention_key"]),
                "mention_key": str(item["mention_key"]),
                "surface": str(item["surface"]),
                "name": str(item["surface"]),
                "type": str(item.get("entity_type") or "GenericEntity"),
                "evidence": str(item.get("evidence_text") or item["surface"]),
                "mention_start": int(item["evidence_start"]),
                "mention_end": int(item["evidence_end"]),
                "confidence": float(item.get("confidence") or 0.0),
                "canonical_id": canonical_by_mention[str(item["mention_key"])],
            } for item in mentions]
        else:
            entities = []
            for item in self.db.list_control_entities_for_chunk(chunk_id, revision_id):
                metadata = item.get("metadata") or {}
                start = metadata.get("mention_start")
                end = metadata.get("mention_end")
                if start is None or end is None:
                    continue
                entities.append({
                    "mention_id": str(item["id"]),
                    "mention_key": str(item.get("mention_key") or item["id"]),
                    "surface": str(item["name"]),
                    "name": str(item["name"]),
                    "type": str(item.get("entity_type") or "GenericEntity"),
                    "evidence": str(item.get("evidence_text") or item["name"]),
                    "mention_start": int(start), "mention_end": int(end),
                    "confidence": float(item.get("confidence") or 0.0),
                    "canonical_id": str(item.get("canonical_entity_id") or item["id"]),
                })
        treatment_run = self._run_versioned_relation_pipeline(
            str(snapshot["id"]), project, int(work["document_id"]), chunk,
            {"entities": entities}, schema_record, raise_on_failure=True,
        )
        control_rows = self.db.list_control_entities_for_chunk(chunk_id, revision_id)

        def control_mention(item: dict[str, Any]) -> dict[str, Any]:
            metadata = item.get("metadata") or {}
            return {
                "mention_id": str(item["id"]),
                "mention_key": str(item.get("mention_key") or item["id"]),
                "document_id": int(item["document_id"]),
                "revision_id": str(item["revision_id"]), "content_hash": expected_hash,
                "chunk_id": int(item["chunk_id"]),
                "start": metadata.get("mention_start"),
                "end": metadata.get("mention_end"),
                "type": str(item.get("entity_type") or "GenericEntity"),
                "canonical_id": str(item.get("canonical_entity_id") or item["id"]),
            }

        control_mentions = [control_mention(item) for item in control_rows]
        if str(snapshot.get("mode")) == "end_to_end":
            treatment_mentions = [{
                "mention_id": str(item.get("id") or item["mention_key"]),
                "mention_key": str(item["mention_key"]),
                "document_id": int(item["document_id"]),
                "revision_id": str(item["revision_id"]), "content_hash": expected_hash,
                "chunk_id": int(item["chunk_id"]), "start": int(item["evidence_start"]),
                "end": int(item["evidence_end"]),
                "type": str(item.get("entity_type") or "GenericEntity"),
                "canonical_id": canonical_by_mention.get(
                    str(item["mention_key"]), str(item["mention_key"]),
                ),
            } for item in mentions]
        else:
            treatment_mentions = list(control_mentions)
            canonical_by_mention = {
                str(item.get("mention_key") or item["mention_id"]): str(item["canonical_id"])
                for item in treatment_mentions
            }

        def relation_fact(
            item: dict[str, Any], *, treatment: bool,
            candidate: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            metadata = self._metadata(item.get("metadata"))
            source_mention = str(
                (candidate or {}).get("source_mention_id")
                or metadata.get("source_mention_id") or item.get("source_entity_id") or ""
            )
            target_mention = str(
                (candidate or {}).get("target_mention_id")
                or metadata.get("target_mention_id") or item.get("target_entity_id") or ""
            )
            source_mention_key = str(
                (candidate or {}).get("source_mention_key")
                or metadata.get("source_mention_key") or source_mention
            )
            target_mention_key = str(
                (candidate or {}).get("target_mention_key")
                or metadata.get("target_mention_key") or target_mention
            )
            return {
                "document_id": int(work["document_id"]), "revision_id": revision_id,
                "content_hash": expected_hash, "chunk_id": chunk_id,
                "subject_mention_id": source_mention,
                "subject_canonical_id": (
                    canonical_by_mention.get(source_mention_key, source_mention) if treatment
                    else str(item.get("source_entity_id") or source_mention)
                ),
                "predicate": str(item.get("predicate") or item.get("relation_type") or ""),
                "object_mention_id": target_mention,
                "object_canonical_id": (
                    canonical_by_mention.get(target_mention_key, target_mention) if treatment
                    else str(item.get("target_entity_id") or target_mention)
                ),
                "direction": str(item.get("direction") or metadata.get("direction") or "forward"),
                "evidence_text": str(item.get("evidence") or item.get("evidence_text") or ""),
                "evidence_start": item.get("evidence_start"),
                "evidence_end": item.get("evidence_end"),
                "snapshot_id": str(snapshot["id"]),
                "policy_hash": str(
                    snapshot.get("policy_hash") if treatment
                    else manifest.get("control_policy_hash") or project.get("active_policy_hash") or "control"
                ),
                "confidence": float(item.get("confidence") or 0.0),
            }

        control_relations = [
            relation_fact(item, treatment=False)
            for item in self.db.list_control_relations_for_chunk(chunk_id, revision_id)
        ]
        candidates = {
            str(item["candidate_id"]): item for item in treatment_run.get("candidates") or []
        }
        treatment_relations = [
            relation_fact(
                item, treatment=True, candidate=candidates.get(str(item.get("candidate_id") or "")),
            ) for item in treatment_run.get("decisions") or []
            if str(item.get("predicate") or "").upper() != "NO_RELATION"
        ]
        item_id = f"{int(work['document_id'])}:{revision_id}:{chunk_id}"
        evaluation_item = {
            "item_id": item_id,
            "control": {
                "mentions": control_mentions, "relations": control_relations,
                "candidates": control_relations, "calibration": [],
                "cost": {"tokens": 0, "model_calls": 0},
            },
            "treatment": {
                "mentions": treatment_mentions, "relations": treatment_relations,
                "candidates": [relation_fact(
                    {"predicate": "", "evidence": ""}, treatment=True, candidate=item,
                ) for item in candidates.values()],
                "calibration": [],
                "cost": {
                    "tokens": int((treatment_run.get("metrics") or {}).get("input_tokens") or 0)
                    + int((treatment_run.get("metrics") or {}).get("output_tokens") or 0),
                    "model_calls": int((treatment_run.get("metrics") or {}).get("model_calls") or 0),
                },
            },
        }
        evaluation_item["item_hash"] = canonical_digest(evaluation_item)
        return {**treatment_run, "evaluation_item": evaluation_item}

    def evaluate_policy_snapshot_artifacts(
        self,
        project_id: int,
        snapshot_id: str,
        *,
        gold_manifest: dict[str, Any],
        gold_items: list[dict[str, Any]],
        evaluator_version: str = "quality-v2",
    ) -> dict[str, Any]:
        """Compute and append an evaluation from frozen artifact item facts only."""

        status = self.db.policy_snapshot_status(snapshot_id)
        snapshot = status["snapshot"]
        if int(snapshot["project_id"]) != int(project_id):
            raise ValueError("evaluation project 与 snapshot 不一致")
        if not bool(snapshot.get("coverage_complete")):
            raise RuntimeError("snapshot relation coverage 不完整，禁止评估")
        evaluation_items = []
        for artifact in status["artifacts"]:
            payload = self._metadata(artifact.get("payload"))
            if artifact.get("kind") == "relation" and not bool(
                payload.get("coverage_complete")
            ):
                raise RuntimeError("relation artifact coverage 不完整，禁止评估")
            item = payload.get("evaluation_item")
            if isinstance(item, dict):
                evaluation_items.append(item)
        if not evaluation_items:
            raise RuntimeError("snapshot 未生成受信 evaluator 所需 evaluation_item")
        bundle_snapshot = {
            **snapshot,
            "expected_items": len(evaluation_items),
            "completed_items": len(evaluation_items),
            "artifact_root_hash": canonical_digest(sorted(
                [[str(item["item_id"]), str(item["item_hash"])] for item in evaluation_items]
            )),
        }
        evaluation = evaluate_frozen_bundle(
            snapshot_bundle={"snapshot": bundle_snapshot, "items": evaluation_items},
            gold_manifest=gold_manifest, gold_items=gold_items,
            evaluator_version=evaluator_version,
        )
        manifest_report = evaluation["manifest_report"]
        gold = self.db.persist_gold_manifest(
            project_id, gold_manifest, gold_items, manifest_report,
        )
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        row = self.db.create_quality_evaluation_run({
            "project_id": project_id, "snapshot_id": snapshot_id,
            "control_policy_hash": str(project.get("active_policy_hash") or "control"),
            "treatment_policy_hash": str(snapshot.get("policy_hash") or snapshot_id),
            "dataset": gold_manifest["dataset"], "dataset_version": gold_manifest["version"],
            "split": "combined", "manifest": gold_manifest,
            "manifest_valid": bool(manifest_report["production_valid"]),
            "metrics": evaluation["report"],
            "denominators": {
                "manifest": manifest_report,
                "control": evaluation["report"]["control"]["denominators"],
                "treatment": evaluation["report"]["treatment"]["denominators"],
            },
            "cost_metrics": {
                "control": evaluation["report"]["control"]["cost"],
                "treatment": evaluation["report"]["treatment"]["cost"],
            },
            "status": "completed", "gold_manifest_id": gold["id"],
            "evaluator_version": evaluator_version,
            "artifact_root_hash": str(snapshot.get("artifact_root_hash") or ""),
            "gold_manifest_hash": evaluation["gold_manifest_hash"],
            "gold_items_hash": evaluation["gold_items_hash"],
            "report_hash": evaluation["report_hash"],
            "item_hashes": evaluation["item_hashes"],
            "release_level": manifest_report["level"],
            "evaluation_items": [{
                "item_id": item["item_id"], "artifact_hash": item["item_hash"],
                "result": {"control": item["control"], "treatment": item["treatment"]},
            } for item in evaluation_items],
        })
        return {"evaluation": row, "report": evaluation["report"]}

    def evaluate_policy_snapshot_gate(
        self, evaluation_run_id: str, *, approved_by: str | None = None,
    ) -> dict[str, Any]:
        """Derive a gate from one immutable evaluation id; caller metrics are impossible."""

        evaluation = self.db.get_quality_evaluation_run(evaluation_run_id)
        if not evaluation:
            raise LookupError("evaluation_run 不存在")
        if evaluation["status"] != "completed":
            raise RuntimeError("evaluation_run 未完成")
        if not bool(evaluation.get("snapshot_coverage_complete")):
            raise RuntimeError("snapshot relation coverage 不完整，门禁 fail closed")
        if evaluation["evaluator_version"] not in ALLOWED_EVALUATOR_VERSIONS:
            raise RuntimeError("evaluator version 不在允许列表")
        if evaluation["release_level"] != evaluation["current_gold_level"]:
            raise RuntimeError("Gold release level 已变化，门禁 fail closed")
        if (
            evaluation["snapshot_status"] != "ready"
            or int(evaluation["snapshot_failed_items"] or 0)
            or int(evaluation["snapshot_expected_items"] or 0)
            != int(evaluation["snapshot_completed_items"] or 0)
        ):
            raise RuntimeError("snapshot artifact 不完整")
        checks = {
            "artifact root": (
                evaluation["artifact_root_hash"], evaluation["current_artifact_root_hash"]
            ),
            "Gold manifest": (
                evaluation["gold_manifest_hash"], evaluation["current_gold_manifest_hash"]
            ),
            "Gold manifest payload": (
                evaluation["gold_manifest_hash"], evaluation["recomputed_gold_manifest_hash"]
            ),
            "Gold annotations": (
                evaluation["gold_items_hash"], evaluation["current_gold_items_hash"]
            ),
            "Gold annotation payloads": (
                evaluation["gold_items_hash"], evaluation["recomputed_gold_items_hash"]
            ),
            "evaluation report": (
                evaluation["report_hash"], canonical_digest(evaluation["metrics"])
            ),
        }
        for label, (recorded, actual) in checks.items():
            if not recorded or recorded != actual:
                raise RuntimeError(f"{label} hash 不一致，门禁 fail closed")
        if list(evaluation.get("item_hashes") or []) != list(
            evaluation.get("current_evaluation_item_hashes") or []
        ) or not evaluation.get("evaluation_item_results_valid"):
            raise RuntimeError("evaluation item facts 不完整或 hash 不一致，门禁 fail closed")
        report = dict((evaluation.get("metrics") or {}).get("gate") or {})
        if not report:
            raise RuntimeError("evaluation report 缺少 gate 事实")
        production_approved = bool(report.get("production_approved"))
        failed_gates = list(report.get("failed_gates") or [])
        if production_approved and not approved_by:
            production_approved = False
            failed_gates.append("operator_approval")
        gate_report = {
            **report, "production_approved": production_approved,
            "status": "production_approved" if production_approved else "engineering_complete",
            "failed_gates": sorted(set(failed_gates)),
            "evaluation_run_id": evaluation_run_id,
            "report_hash": evaluation["report_hash"],
        }
        gate = self.db.create_policy_activation_gate({
            "project_id": evaluation["project_id"],
            "snapshot_id": evaluation["snapshot_id"],
            "evaluation_run_id": evaluation_run_id,
            "manifest_valid": bool(evaluation["manifest_valid"]),
            "engineering_complete": True,
            "production_approved": production_approved,
            "failed_gates": gate_report["failed_gates"],
            "gate_report": gate_report,
            "approved_by": approved_by if production_approved else None,
        })
        return {"evaluation": evaluation, "gate": gate, "report": gate_report}

    def activate_treatment_snapshot(self, project_id: int, snapshot_id: str) -> dict[str, Any]:
        raise TreatmentMaterializationRequired(
            "treatment snapshot 尚未正式物化，禁止激活或触发图重建"
        )

    @staticmethod
    def _project_entity_relations_gremlin(project_id: int, namespace_id: str) -> str:
        """Show every entity plus entity-to-entity edges, without provenance nodes."""
        project_id = int(project_id)
        namespace = json.dumps(str(uuid.UUID(str(namespace_id))))
        scoped = f'.has("kf_namespace_id",{namespace}).has("kf_project_id",{project_id})'
        return (
            f'g.V(){scoped}.hasLabel("KFEntity")'
            '.union('
            '__.identity(),'
            f'__.bothE("kf_related"){scoped}.dedup().otherV()'
            ').path().limit(1000)'
        )

    def hubble_project_url(
        self, project_id: int, strategy_key: str = STABLE_STRATEGY,
    ) -> str:
        """Resolve Hubble and prefill a connected entity-relationship query."""
        strategy_key = normalize_strategy(strategy_key)
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        if strategy_key == STABLE_STRATEGY:
            if not is_current_graph(project):
                raise GraphNotReadyError(
                    project.get("graph_error") or "图谱需要按当前验证策略重建"
                )
            strategy_graph = {
                "namespace_id": project["graph_namespace_id"],
                "generation": project.get("graph_generation") or 0,
                "ready_generation": project.get("graph_ready_generation") or 0,
                "status": project.get("graph_status") or "empty",
                "error_message": project.get("graph_error") or "",
            }
        else:
            strategy_graph = self.db.get_project_strategy_graph(project_id, strategy_key)
            if not strategy_graph:
                raise GraphNotReadyError("当前构建方案尚未初始化")
        if strategy_key != STABLE_STRATEGY and (
            strategy_graph.get("status") != "ready"
            or int(strategy_graph.get("ready_generation") or 0)
            != int(strategy_graph.get("generation") or 0)
        ):
            raise GraphNotReadyError(
                strategy_graph.get("error_message") or "当前实验图谱尚未就绪"
            )
        base_url = Config.hugegraph_hubble_url.rstrip("/")
        endpoint = f"{base_url}/api/v1.2/graph-connections"

        def connections() -> list[dict[str, Any]]:
            response = requests.get(endpoint, params={"page_no": 1, "page_size": 100}, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if int(payload.get("status") or 0) != 200:
                raise RuntimeError(payload.get("message") or "Hubble 图连接查询失败")
            return ((payload.get("data") or {}).get("records") or [])

        connection = next(
            (item for item in connections()
             if item.get("graph") == Config.hugegraph_graph and item.get("enabled", True)),
            None,
        )
        if not connection:
            response = requests.post(endpoint, json={
                "name": Config.hugegraph_graph,
                "graph": Config.hugegraph_graph,
                "host": Config.hugegraph_hubble_graph_host,
                "port": Config.hugegraph_port,
                "username": Config.hugegraph_user,
                "password": Config.hugegraph_password,
            }, timeout=10)
            response.raise_for_status()
            payload = response.json()
            if int(payload.get("status") or 0) != 200:
                raise RuntimeError(payload.get("message") or "Hubble 图连接创建失败")
            connection = next(
                (item for item in connections()
                 if item.get("graph") == Config.hugegraph_graph and item.get("enabled", True)),
                None,
            )
        if not connection:
            raise RuntimeError(f"Hubble 中没有可用的 {Config.hugegraph_graph} 图连接")

        gremlin = self._project_entity_relations_gremlin(
            project_id, str(strategy_graph["namespace_id"]),
        )
        return (
            f"{base_url}/graph-management/{int(connection['id'])}/data-analyze"
            f"?gremlin={quote(gremlin, safe='')}"
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._job_with_runtime(self.db.get_job(job_id))

    @staticmethod
    def _job_with_runtime(
        job: dict[str, Any] | None,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        if not job:
            return None
        current = now or datetime.now(timezone.utc)
        started = job.get("created_at")
        ended = job.get("completed_at") if job.get("status") in {"completed", "failed", "cancelled"} else current
        if isinstance(started, str):
            started = datetime.fromisoformat(started.replace("Z", "+00:00"))
        if isinstance(ended, str):
            ended = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        elapsed = 0
        if isinstance(started, datetime) and isinstance(ended, datetime):
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if ended.tzinfo is None:
                ended = ended.replace(tzinfo=timezone.utc)
            elapsed = max(0, int((ended - started).total_seconds()))
        return {**job, "elapsed_seconds": elapsed}

    def get_ingestion_batch(self, batch_id: str) -> dict[str, Any] | None:
        jobs = [self._job_with_runtime(row) for row in self.db.list_ingestion_batch_jobs(batch_id)]
        jobs = [row for row in jobs if row]
        if not jobs:
            return None
        kinds = {str(job.get("kind") or "ingestion") for job in jobs}
        batch_kind = kinds.pop() if len(kinds) == 1 else "ingestion"
        target_documents: set[int] = set()
        for job in jobs:
            metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
            target_ids = metadata.get("target_document_ids") or [job.get("document_id")]
            target_documents.update(int(value) for value in target_ids if value is not None)
        document_count = len(target_documents) or len(jobs)
        statuses = [str(row.get("status") or "queued") for row in jobs]
        cancellation_requested = any(
            bool(
                (job.get("metadata") or {}).get("cancel_requested")
                if isinstance(job.get("metadata"), dict)
                else False
            )
            for job in jobs
            if str(job.get("status") or "") in {"queued", "running"}
        )
        terminal = {"completed", "failed", "cancelled"}
        if any(status == "running" for status in statuses) or (
            any(status == "queued" for status in statuses)
            and any(status in terminal for status in statuses)
        ):
            status = "running"
        elif any(status == "queued" for status in statuses):
            status = "queued"
        elif any(status == "failed" for status in statuses):
            status = "failed"
        elif any(value == "cancelled" for value in statuses):
            status = "cancelled"
        else:
            status = "completed"

        subtasks: dict[str, dict[str, Any]] = {}
        for key in self.INGESTION_SUBTASKS:
            values = []
            for job in jobs:
                metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
                stored = metadata.get("subtasks") if isinstance(metadata.get("subtasks"), dict) else {}
                value = (
                    stored.get(key) if isinstance(stored.get(key), dict) else {
                        "status": "waiting", "progress": 0, "detail": "",
                    }
                )
                if (
                    str(job.get("status") or "") == "cancelled"
                    and str(value.get("status") or "waiting")
                    not in {"completed", "failed", "cancelled"}
                ):
                    value = {**value, "status": "cancelled"}
                values.append(value)
            step_statuses = [str(value.get("status") or "waiting") for value in values]
            if any(value == "running" for value in step_statuses):
                step_status = "running"
            elif any(value == "failed" for value in step_statuses):
                step_status = "failed"
            elif all(value == "completed" for value in step_statuses):
                step_status = "completed"
            elif all(value in {"completed", "cancelled"} for value in step_statuses):
                step_status = "cancelled"
            elif any(value == "completed" for value in step_statuses):
                step_status = "running"
            else:
                step_status = "waiting"
            completed = sum(value == "completed" for value in step_statuses)
            completed_documents = (
                document_count if len(jobs) == 1 and completed else completed
            )
            subtasks[key] = {
                "status": step_status,
                "progress": int(sum(int(value.get("progress") or 0) for value in values) / len(values)),
                # A project-wide Knowledge Forge build is one durable job that
                # internally processes many chunks/pairs/facts.  Preserve that
                # job's real stage detail instead of replacing it with the
                # misleading batch-level ``0/N 个文档`` label.
                "detail": (
                    str(values[0].get("detail") or "")
                    if len(values) == 1
                    else f"{completed_documents}/{document_count} 个文档"
                ),
            }

        log: list[str] = []
        for index, job in enumerate(jobs, start=1):
            name = str(job.get("document_name") or (job.get("metadata") or {}).get("document_name") or f"文档 {index}")
            state = {
                "queued": "排队中", "running": "执行中", "completed": "已完成",
                "failed": "失败", "cancelled": "已终止",
            }.get(str(job.get("status") or "queued"), str(job.get("status") or "queued"))
            log.append(f"===== [{index}/{len(jobs)}] {name} · {state} =====")
            log.extend(str(line) for line in (job.get("log") or []))
            if not job.get("log"):
                log.append("等待开始处理")

        starts = [job.get("created_at") for job in jobs if isinstance(job.get("created_at"), datetime)]
        elapsed = max(int(job.get("elapsed_seconds") or 0) for job in jobs)
        if starts:
            start = min(starts)
            end = datetime.now(timezone.utc)
            if all(value in terminal for value in statuses):
                completed = [job.get("completed_at") for job in jobs if isinstance(job.get("completed_at"), datetime)]
                if completed:
                    end = max(completed)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            elapsed = max(0, int((end - start).total_seconds()))
        return {
            "id": batch_id,
            "batch_id": batch_id,
            "kind": batch_kind,
            "status": status,
            "progress": int(sum(int(job.get("progress") or 0) for job in jobs) / len(jobs)),
            "elapsed_seconds": elapsed,
            "document_count": document_count,
            "jobs": jobs,
            "log": log,
            "metadata": {
                "batch_id": batch_id,
                "subtasks": subtasks,
                "cancel_requested": cancellation_requested,
            },
        }

    def latest_job(
        self, project_id: int, kind: str = "", strategy_key: str = "",
    ) -> dict[str, Any] | None:
        return self._job_with_runtime(self.db.latest_job(project_id, kind, strategy_key))

    def latest_active_job(
        self, project_id: int, kind: str = "", strategy_key: str = "",
    ) -> dict[str, Any] | None:
        return self._job_with_runtime(self.db.latest_active_job(project_id, kind, strategy_key))

    def latest_active_mutating_job(
        self, project_id: int,
    ) -> dict[str, Any] | None:
        getter = getattr(self.db, "latest_active_mutating_job", None)
        row = getter(project_id) if callable(getter) else self.db.latest_active_job(project_id)
        return self._job_with_runtime(row)

    def _document_process_payload(self, document_id: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
        document = self.db.get_document(document_id)
        if not document:
            raise LookupError("文档不存在")

        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        candidates: list[Path] = []
        if metadata.get("chunk_file"):
            candidates.append(Path(str(metadata["chunk_file"])))
        candidates.append(
            Config.chunks_root
            / Config.project_storage_key(int(document["project_id"]), str(document["project_name"]))
            / f"{document['doc_hash'][:16]}.json"
        )

        root = Config.chunks_root.resolve()
        payload_path = next(
            (
                path.resolve()
                for path in candidates
                if (root in path.resolve().parents or path.resolve() == root) and path.is_file()
            ),
            None,
        )
        if payload_path is None:
            if document.get("status") == "processing":
                raise LookupError("结构块尚未生成，请稍后刷新")
            raise LookupError("未找到该文档的结构解析过程文件，请重新解析文档")

        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"结构解析过程文件读取失败：{exc}") from exc
        return document, payload, payload_path

    @staticmethod
    def _write_process_payload(payload_path: Path, document_name: str, result: ProcessingResult) -> None:
        payload = sanitize_unicode_data({
            "document": document_name, **result.payload(),
        })
        temporary = payload_path.with_suffix(payload_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(payload_path)

    @staticmethod
    def _revision_process_payload_path(
        root: Path,
        document: dict[str, Any],
        revision: dict[str, Any],
    ) -> Path:
        """Allocate a process artifact that cannot overwrite an active revision."""

        revision_id = str(revision.get("revision_id") or "")
        if not revision_id or not all(
            character.isalnum() or character in {"-", "_"}
            for character in revision_id
        ):
            raise ValueError("内容修订编号不适合生成过程文件")
        return root / f"{str(document['doc_hash'])[:16]}-{revision_id}.json"

    @staticmethod
    def _discard_staged_process_payload(path: Path | None) -> None:
        """Remove only a generated process file inside the configured chunk root."""

        if path is None:
            return
        candidate = path.resolve()
        root = Config.chunks_root.resolve()
        if candidate == root or root not in candidate.parents:
            return
        if candidate.is_symlink():
            candidate.unlink(missing_ok=True)
            return
        if candidate.is_file():
            candidate.unlink(missing_ok=True)
        temporary = candidate.with_suffix(candidate.suffix + ".tmp")
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _discard_staged_media_directory(path: Path | None) -> None:
        """Remove only one revision-scoped generated media directory."""

        if path is None:
            return
        candidate = path.resolve()
        root = Config.media_root.resolve()
        if candidate == root or root not in candidate.parents:
            return
        if candidate.is_symlink():
            candidate.unlink(missing_ok=True)
        elif candidate.is_dir():
            shutil.rmtree(candidate)

    def _restore_unpublished_document_attempt(
        self,
        document_id: int,
        original: dict[str, Any],
        error: Exception,
        *,
        cancelled: bool,
    ) -> None:
        """Restore the prior public document projection after staging stops."""

        if not self.db.get_document(document_id):
            return
        values = {
            key: original.get(key)
            for key in (
                "indexed_at",
                "structure_ready_at",
                "chunks_ready_at",
                "graph_ready_at",
                "effective_parser",
                "effective_chunk_strategy",
                "parser_version",
                "parse_quality",
                "routing_reason",
                "metadata",
            )
        }
        if cancelled or original.get("active_revision_id"):
            values.update({
                "status": original.get("status") or (
                    "indexed" if original.get("active_revision_id") else "uploaded"
                ),
                "processing_stage": original.get("processing_stage") or (
                    "chunk_ready" if original.get("active_revision_id") else "uploaded"
                ),
                "error_message": original.get("error_message"),
            })
        else:
            values.update({
                "status": "failed",
                "processing_stage": "failed",
                "error_message": str(error),
            })
        self.db.update_document(document_id, **values)

    def document_structure(self, document_id: int) -> dict[str, Any]:
        """Return the persisted parser blocks without parsing the source again."""
        document, payload, _ = self._document_process_payload(document_id)
        blocks = payload.get("blocks")
        if not isinstance(blocks, list):
            raise RuntimeError("结构解析过程文件中没有有效的 blocks 数据")

        return {
            "document_id": document_id,
            "document_name": document["original_filename"],
            "status": document["status"],
            "plan": payload.get("plan") if isinstance(payload.get("plan"), dict) else {},
            "quality": payload.get("quality") if isinstance(payload.get("quality"), dict) else {},
            "block_count": len(blocks),
            "block_types": sorted({str(item.get("block_type") or "unknown") for item in blocks if isinstance(item, dict)}),
            "blocks": [item for item in blocks if isinstance(item, dict)],
        }

    def document_chunks(self, document_id: int) -> dict[str, Any]:
        """Return persisted retrieval chunks after the chunking stage completes."""
        document = self.db.get_document(document_id)
        if not document:
            raise LookupError("文档不存在")
        self._assert_project_strategy(
            int(document["project_id"]), KNOWLEDGE_FORGE_STRATEGY,
        )
        if not document.get("chunks_ready_at"):
            raise LookupError("知识块尚未生成，请等待第二阶段完成")
        chunks = self.db.list_document_chunks(document_id)
        return {
            "document_id": document_id,
            "document_name": document["original_filename"],
            "status": document["status"],
            "chunk_count": len(chunks),
            "chunks": chunks,
        }

    def _job(self, job_id: str, status: str | None = None, progress: int | None = None, log: str = "", **metadata: Any) -> None:
        self.db.update_job(job_id, status=status, progress=progress, log=log, metadata=metadata or None)

    @classmethod
    def _empty_ingestion_subtasks(cls) -> dict[str, dict[str, Any]]:
        return {
            key: {"status": "waiting", "progress": 0, "detail": ""}
            for key in cls.INGESTION_SUBTASKS
        }

    def _start_ingestion_job(
        self, job_id: str, document_id: int, *, mutate_graph: bool = True,
    ) -> None:
        # Production Database exposes the durable job/document read contract.
        # A handful of narrow legacy test adapters intentionally exercise only
        # mutation routing and do not implement those reads.
        if hasattr(self.db, "get_job") and hasattr(self.db, "get_document"):
            self._assert_ingestion_active(job_id, document_id)
        if mutate_graph:
            self.db.begin_graph_mutation(job_id)
        else:
            self.db.update_job(job_id, status="running")
        subtasks = self._empty_ingestion_subtasks()
        subtasks["structure"] = {"status": "running", "progress": 0, "detail": ""}
        self._job(
            job_id,
            "running",
            2,
            "===== 阶段一：结构解析 =====",
            document_id=document_id,
            stage="structure",
            active_subtask="structure",
            subtasks=subtasks,
            structure_ready=False,
            chunks_ready=False,
            graph_ready=False,
        )

    def _job_subtask(
        self,
        job_id: str,
        subtask: str,
        *,
        subtask_status: str,
        subtask_progress: int,
        detail: str = "",
        status: str | None = None,
        progress: int | None = None,
        log: str = "",
        **metadata: Any,
    ) -> None:
        job = self.db.get_job(job_id) or {}
        existing_metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        existing_subtasks = existing_metadata.get("subtasks")
        subtasks = {
            key: dict(value)
            for key, value in (
                existing_subtasks.items() if isinstance(existing_subtasks, dict) else self._empty_ingestion_subtasks().items()
            )
        }
        previous_subtask = (
            subtasks.get(subtask)
            if isinstance(subtasks.get(subtask), dict)
            else {}
        )
        bounded_subtask_progress = max(
            0, min(100, int(subtask_progress)),
        )
        # A durable job exposes one public progress timeline. Nested workers
        # may start a fresh local counter for a later phase, but that must not
        # make an already-rendered progress bar move backwards.
        if subtask_status in {"running", "completed"}:
            bounded_subtask_progress = max(
                int(previous_subtask.get("progress") or 0),
                bounded_subtask_progress,
            )
        subtasks[subtask] = {
            "status": subtask_status,
            "progress": bounded_subtask_progress,
            "detail": detail,
        }
        if progress is not None and status not in {"queued"}:
            progress = max(int(job.get("progress") or 0), int(progress))
        self._job(
            job_id,
            status,
            progress,
            log,
            active_subtask=subtask,
            subtasks=subtasks,
            **metadata,
        )

    @staticmethod
    def _knowledge_forge_extraction_progress(
        done: int,
        total: int,
        message: str,
    ) -> tuple[int, str]:
        """Map nested entity/relation phases onto one monotonic UI range."""

        done = max(0, int(done))
        total = max(1, int(total))
        message = " ".join(str(message or "").split())
        relation_phase = message.startswith((
            "Knowledge Forge 候选对规划",
            "Knowledge Forge 源事实清单",
            "Knowledge Forge 候选关系审计账本",
            "Knowledge Forge 候选补充",
            "Knowledge Forge 关系分类",
            "Knowledge Forge 关系批量终审",
            "Knowledge Forge 关系结构复核",
            "Knowledge Forge 关系冲突复核",
            "Knowledge Forge 关系验证",
        ))
        if message.startswith("Knowledge Forge 完成"):
            return 95, message.removeprefix("Knowledge Forge ").strip()
        if relation_phase:
            progress = 45 + int(min(done, total) / total * 50)
            return min(95, progress), message.removeprefix("Knowledge Forge ").strip()
        progress = 5 + int(min(done, total) / total * 40)
        detail = f"实体抽取 {min(done, total)}/{total} 个知识块"
        if "复用已完成" in message:
            detail = "已复用完整实体与关系快照"
        return min(45, progress), detail

    @staticmethod
    def _knowledge_forge_build_progress(
        *,
        extraction_progress: int = 0,
        graph_sync_progress: int = 0,
        schema_progress: int = 0,
        schema_discovery: bool = False,
    ) -> int:
        """Report only work performed by the graph build itself.

        Prepared document structure, chunks and vectors are immutable inputs,
        not build progress.  Explicit build-time Schema discovery remains a
        visible optional phase; the ordinary active-Schema snapshot has zero
        progress weight.
        """

        def bounded(value: int) -> int:
            return max(0, min(100, int(value)))
        schema_weight = 0.10 if schema_discovery else 0.0
        extraction_weight = 0.82 if schema_discovery else 0.90
        graph_weight = 0.08 if schema_discovery else 0.10
        progress = (
            bounded(schema_progress) * schema_weight
            + bounded(extraction_progress) * extraction_weight
            + bounded(graph_sync_progress) * graph_weight
        )
        return max(0, min(100, int(progress + 0.5)))

    def _knowledge_forge_build_schema(
        self,
        project: dict[str, Any],
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Freeze Schema for one build without making governance a hard gate."""

        if Config.schema_discovery_during_build:
            return self.schemas.evaluate_corpus(project, documents)
        active = dict(self.schemas.ensure_base(int(project["id"])))
        active["_evaluation"] = {
            "scope": "project_build_active_schema_snapshot",
            "reviewed": False,
            "reason": "deferred_to_explicit_schema_reassessment",
            "document_ids": [int(item["id"]) for item in documents],
        }
        return active

    def _fail_active_ingestion_subtask(
        self, job_id: str, detail: str, *, cancelled: bool = False,
    ) -> dict[str, dict[str, Any]] | None:
        job = self.db.get_job(job_id) or {}
        if cancelled:
            return self._cancelled_subtasks(job, detail)
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        active = str(metadata.get("active_subtask") or "")
        if active not in self.INGESTION_SUBTASKS:
            return None
        subtasks = metadata.get("subtasks") if isinstance(metadata.get("subtasks"), dict) else {}
        current = subtasks.get(active) if isinstance(subtasks.get(active), dict) else {}
        self._job_subtask(
            job_id,
            active,
            subtask_status="failed",
            subtask_progress=int(current.get("progress") or 0),
            detail=detail,
        )
        return None

    def _cancelled_subtasks(
        self, job: dict[str, Any], detail: str,
    ) -> dict[str, dict[str, Any]] | None:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        active = str(metadata.get("active_subtask") or "")
        if active not in self.INGESTION_SUBTASKS:
            return None
        subtasks = metadata.get("subtasks") if isinstance(metadata.get("subtasks"), dict) else {}
        current = subtasks.get(active) if isinstance(subtasks.get(active), dict) else {}
        # Cancellation acknowledgement and subtask closure are persisted
        # together in one terminal update by the caller.  A separate running-
        # state update would be intentionally suppressed by the database once
        # cancel_requested is durable.
        closed = {
            key: dict(value) for key, value in subtasks.items()
            if isinstance(value, dict)
        }
        closed[active] = {
            "status": "cancelled",
            "progress": int(current.get("progress") or 0),
            "detail": detail,
        }
        return closed

    def _document_lock(self, document_id: int) -> threading.RLock:
        with self._document_locks_guard:
            return self._document_locks.setdefault(document_id, threading.RLock())

    def _graph_lock(self, project_id: int) -> threading.RLock:
        if not hasattr(self, "_graph_locks_guard"):
            self._graph_locks_guard = threading.Lock()
            self._graph_locks = {}
        with self._graph_locks_guard:
            return self._graph_locks.setdefault(project_id, threading.RLock())

    def _release_strategy_graph_resources(
        self,
        mutation: GraphMutationContext | None,
        lease: dict[str, Any] | None,
    ) -> None:
        """Release local and durable ownership after any strategy-graph exit."""

        if mutation is not None:
            discard_manifest = getattr(
                self.graph, "discard_writer_manifest", None,
            )
            if discard_manifest is not None:
                try:
                    discard_manifest(mutation)
                except Exception:
                    LOGGER.exception(
                        "Unable to discard Knowledge Forge strategy writer manifest"
                    )
        if lease:
            try:
                self.db.release_graph_lease(
                    str(lease["id"]), str(lease["lease_token"]),
                )
            except Exception:
                LOGGER.exception(
                    "Unable to release Knowledge Forge strategy graph lease"
                )

    def _assert_job_active(self, job_id: str) -> None:
        if not hasattr(self.db, "get_job"):
            return
        job = self.db.get_job(job_id)
        metadata = job.get("metadata") if job and isinstance(job.get("metadata"), dict) else {}
        if (
            not job
            or str(job.get("status") or "") not in {"queued", "running"}
            or bool(metadata.get("cancel_requested"))
        ):
            raise IngestionCancelled(
                str(metadata.get("cancel_reason") or "任务已安全终止")
            )

    def _assert_ingestion_active(self, job_id: str, document_id: int) -> None:
        self._assert_job_active(job_id)
        if not self.db.get_document(document_id):
            raise IngestionCancelled("文档已删除")

    def _job_cancellation_requested(self, job_id: str) -> bool:
        get_job = getattr(self.db, "get_job", None)
        if not callable(get_job):
            return False
        job = get_job(job_id) or {}
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        return bool(
            str(job.get("status") or "") == "cancelled"
            or metadata.get("cancel_requested")
        )

    def _document_revision_manifest(
        self, document: dict[str, Any], project: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "content_hash": document["doc_hash"],
            "source_hash": document["doc_hash"],
            "config_hash": hashlib.sha256(json.dumps({
                "structure_profile": DOCUMENT_STRUCTURE_PROFILE_VERSION,
                "chunk_profile": DOCUMENT_CHUNK_PROFILE_VERSION,
                "schema_sampling_profile": "kf-schema-graph-eligible-v1",
                "requested_parser": document.get("requested_parser") or "auto",
                "requested_chunk_strategy": document.get("requested_chunk_strategy") or "auto",
                "chunk_max_tokens": int(
                    project.get("chunk_max_tokens") or self.DEFAULT_CHUNK_MAX_TOKENS
                ),
                "chunking": {
                    "compat_max_chars": int(Config.chunk_max_chars),
                    "compat_overlap_chars": int(Config.chunk_overlap_chars),
                    "general_target_tokens": int(Config.general_chunk_target_tokens),
                    "general_max_tokens": int(Config.general_chunk_max_tokens),
                    "general_overlap_percent": int(
                        Config.general_chunk_overlap_percent
                    ),
                    "general_coalesce_min_sections": int(
                        Config.general_coalesce_min_sections
                    ),
                    "paper_target_tokens": int(Config.paper_chunk_target_tokens),
                    "paper_max_tokens": int(Config.paper_chunk_max_tokens),
                    "paper_overlap_percent": int(
                        Config.paper_chunk_overlap_percent
                    ),
                },
                "llm_model": project.get("llm_model") or Config.default_llm_model,
                "embedding_model": (
                    project.get("embedding_model") or Config.default_embedding_model
                ),
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "policy_hash": graph_policy_hash(
                {axis: str(project.get(axis) or "") for axis in POLICY_AXES},
                schema_version=int(project.get("active_schema_version") or 1),
            ),
            "document_id": int(document["id"]),
            "project_id": int(project["id"]),
        }

    def _native_source_revision_manifest(
        self, document: dict[str, Any], project: dict[str, Any],
    ) -> dict[str, Any]:
        """Fingerprint only parser output consumed by GraphRAG Native."""

        return {
            "content_hash": document["doc_hash"],
            "source_hash": document["doc_hash"],
            "config_hash": hashlib.sha256(json.dumps({
                "profile": "graphrag_native_source_v1",
                "structure_profile": DOCUMENT_STRUCTURE_PROFILE_VERSION,
                "requested_parser": document.get("requested_parser") or "auto",
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "policy_hash": "graphrag-native-source-v1",
            "document_id": int(document["id"]),
            "project_id": int(project["id"]),
        }

    def _shared_content_is_current(
        self,
        document: dict[str, Any],
        manifest: dict[str, Any],
    ) -> bool:
        revision_id = str(document.get("active_revision_id") or "")
        if not revision_id:
            return False
        revision = self._active_document_revision_snapshot(document)
        if not revision or str(revision.get("status") or "") != "active":
            return False
        return all(
            str(revision.get(key) or "") == str(manifest.get(key) or "")
            for key in ("content_hash", "source_hash", "config_hash")
        )

    def _active_document_revision_snapshot(
        self, document: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Reuse the list query's joined revision, falling back for adapters."""

        revision_id = str(document.get("active_revision_id") or "")
        if not revision_id:
            return None
        if "active_revision_status" in document:
            return {
                "revision_id": revision_id,
                "status": document.get("active_revision_status"),
                "content_hash": document.get("active_revision_content_hash"),
                "source_hash": document.get("active_revision_source_hash"),
                "config_hash": document.get("active_revision_config_hash"),
            }
        return self.db.get_document_revision(revision_id)

    def _document_preparation_reason(
        self,
        document: dict[str, Any],
        manifest: dict[str, Any],
    ) -> str:
        """Classify why a document appears in the incremental preparation plan."""

        if not document.get("active_revision_id"):
            return "new_document"
        revision = self._active_document_revision_snapshot(document)
        if not revision or str(revision.get("status") or "") != "active":
            return "revision_unavailable"
        if any(
            str(revision.get(key) or "") != str(manifest.get(key) or "")
            for key in ("content_hash", "source_hash")
        ):
            return "source_changed"
        if str(revision.get("config_hash") or "") != str(
            manifest.get("config_hash") or ""
        ):
            return "configuration_changed"
        return "preparation_outdated"

    def _reconcile_knowledge_forge_content_freshness(
        self, project: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Invalidate a ready Knowledge Forge graph built from obsolete shared chunks."""

        project_id = int(project["id"])
        graph = self.db.get_project_strategy_graph(
            project_id, KNOWLEDGE_FORGE_STRATEGY,
        )
        if not graph or str(graph.get("status") or "") != GRAPH_READY:
            return None
        stale_documents: list[int] = []
        for document in self.db.list_documents(project_id):
            if (
                bool(document.get("is_deleted"))
                or str(document.get("status") or "") != "indexed"
                or not document.get("active_revision_id")
            ):
                continue
            manifest = self._document_revision_manifest(document, project)
            if not self._shared_content_is_current(document, manifest):
                stale_documents.append(int(document["id"]))
        if not stale_documents:
            return None
        return self.db.mark_strategy_graph_stale(
            project_id,
            KNOWLEDGE_FORGE_STRATEGY,
            (
                "文档解析或分块策略已升级，Knowledge Forge 需要重新构建"
                f"（{len(stale_documents)} 个文档）"
            ),
        )

    def _prepare_shared_content(
        self,
        job_id: str,
        document_id: int,
        *,
        mutate_graph: bool = True,
        require_reusable: bool = False,
        report_progress: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        """Prepare one strategy-neutral content revision when none is reusable."""

        document = self.db.get_document(document_id)
        if not document:
            raise IngestionCancelled("文档不存在或已删除")
        original_document = dict(document)
        project = self.db.get_project(int(document["project_id"]))
        if not project:
            raise LookupError("项目不存在")
        manifest = self._document_revision_manifest(document, project)
        self._assert_ingestion_active(job_id, document_id)
        if self._shared_content_is_current(document, manifest):
            chunks = self.db.list_document_chunks(document_id)
            if chunks:
                self._assert_ingestion_active(job_id, document_id)
                vector_count = len(chunks)
                vectors_reindexed = False
                vector_checker = getattr(
                    getattr(self, "vectors", None),
                    "document_index_is_current",
                    None,
                )
                if callable(vector_checker) and not vector_checker(
                    project, document_id, chunks,
                ):
                    vector_count = self.vectors.index_chunks(
                        project, document_id, chunks,
                    )
                    vectors_reindexed = True
                    self._assert_ingestion_active(job_id, document_id)
                if report_progress:
                    subtasks = self._empty_ingestion_subtasks()
                    subtasks["structure"] = {
                        "status": "completed", "progress": 100, "detail": "复用结构解析结果",
                    }
                    subtasks["chunking"] = {
                        "status": "completed", "progress": 100,
                        "detail": (
                            f"复用 {len(chunks)} 个知识块，更新 {vector_count} 个向量索引"
                            if vectors_reindexed else
                            f"复用 {len(chunks)} 个知识块与向量"
                        ),
                    }
                    subtasks["schema_base"] = {
                        "status": "completed", "progress": 100, "detail": "复用共享 Schema",
                    }
                    self._job(
                        job_id, "running", 20, "文档预处理结果已就绪，直接进入所选构建方案",
                        document_id=document_id, stage="graph", active_subtask="schema_base",
                        subtasks=subtasks, structure_ready=True, chunks_ready=True,
                        shared_content_reused=True,
                        embedding_index_rebuilt=vectors_reindexed,
                    )
                return document, project, chunks

        if require_reusable:
            raise RuntimeError(
                f"文档“{document['original_filename']}”的预处理结果不存在或已过期，"
                "请先完成文档预处理再构建图谱"
            )

        revision: dict[str, Any] | None = None
        revision_scope: Any = None
        payload_path: Path | None = None
        media_dir: Path | None = None
        published = False
        try:
            self._start_ingestion_job(job_id, document_id, mutate_graph=mutate_graph)
            revision = self.db.begin_document_revision(document_id, job_id, manifest)
            revision_scope = self.db.content_revision_context(
                str(revision["revision_id"]), job_id, str(revision["lease_token"]),
            )
            revision_scope.__enter__()
            source_path = Path(document["local_path"])
            plan, parsed = parse_document_structure(
                source_path,
                requested_parser=document.get("requested_parser") or "auto",
                requested_strategy=document.get("requested_chunk_strategy") or "auto",
            )
            media_dir = (
                Config.media_root
                / Config.project_storage_key(int(project["id"]), project["name"])
                / str(document["doc_hash"][:16])
                / str(revision["revision_id"])
            )

            def media_progress(index: int, total: int, message: str) -> None:
                self._assert_ingestion_active(job_id, document_id)
                self._job_subtask(
                    job_id, "structure", subtask_status="running",
                    subtask_progress=20 + int(index / max(1, total) * 55),
                    detail=f"图片内容 {index}/{total}",
                    progress=8 + int(index / max(1, total) * 4), log=message,
                )

            parsed = self.media.enrich(
                source_path, parsed, project, media_dir, progress=media_progress,
            )
            chunk_dir = (
                Config.chunks_root
                / Config.project_storage_key(int(project["id"]), project["name"])
            )
            chunk_dir.mkdir(parents=True, exist_ok=True)
            payload_path = self._revision_process_payload_path(
                chunk_dir, document, revision,
            )
            current_metadata = (
                document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
            )
            process_metadata = {
                **current_metadata, "chunk_file": str(payload_path),
                "media_dir": str(media_dir), "effective_parser": plan.effective_parser,
                "effective_strategy": plan.effective_strategy,
            }
            self._assert_ingestion_active(job_id, document_id)
            process_result = ProcessingResult(plan, parsed, [], [])
            self._write_process_payload(
                payload_path, document["original_filename"], process_result,
            )
            self._job_subtask(
                job_id, "structure", subtask_status="completed", subtask_progress=100,
                detail=f"{parsed.quality.get('block_count', 0)} 个结构块", progress=18,
                log=(f"结构解析完成：{parsed.quality.get('text_chars', 0)} 字符、"
                     f"{parsed.quality.get('block_count', 0)} 个结构块、"
                     f"{parsed.quality.get('table_count', 0)} 个表格、"
                     f"{parsed.quality.get('image_count', 0)} 张图片"),
                stage="structure_ready", structure_ready=True,
            )
            self._job_subtask(
                job_id, "chunking", subtask_status="running", subtask_progress=0,
                progress=20, log="===== 文档预处理：知识分块与向量索引 =====",
                stage="chunking",
            )
            result = build_processing_result(
                source_path, plan, parsed, source_name=document["original_filename"],
                chunk_max_tokens=int(
                    project.get("chunk_max_tokens") or self.DEFAULT_CHUNK_MAX_TOKENS
                ),
            )
            sections, chunks = result.sections, result.chunks
            section_values = [{
                **asdict(item),
                "content_hash": hashlib.sha256(item.content.encode()).hexdigest(),
                "metadata": {**item.metadata, "source": document["original_filename"]},
            } for item in sections]
            chunk_values = []
            for item in chunks:
                value = asdict(item)
                value["metadata"] = {
                    **item.metadata, "project_id": str(project["id"]),
                    "project_key": project["project_key"],
                    "document_id": str(document["id"]),
                    "document_name": document["original_filename"],
                    "document_url": document["storage_path"],
                    "section_key": item.section_key, "section_title": item.title,
                    "token_count": item.token_count, "parent_key": item.parent_key,
                }
                chunk_values.append(value)
            self._assert_ingestion_active(job_id, document_id)
            self._write_process_payload(
                payload_path, document["original_filename"], result,
            )
            stored_chunks = self.db.replace_document_content(
                document_id, section_values, chunk_values,
                revision_id=str(revision["revision_id"]), ingestion_run_id=job_id,
                lease_token=str(revision["lease_token"]), manifest=manifest,
            )
            self._job_subtask(
                job_id, "chunking", subtask_status="running", subtask_progress=60,
                detail=f"{len(chunks)} 个知识块", progress=32,
                log=f"已生成 {len(sections)} 个章节和 {len(chunks)} 个知识块",
                stage="chunk_ready", chunks_ready=True,
            )
            self._assert_ingestion_active(job_id, document_id)
            count = self.vectors.index_chunks(project, document_id, stored_chunks)
            self._job_subtask(
                job_id, "chunking", subtask_status="completed", subtask_progress=100,
                detail=f"{count} 个向量索引", progress=40,
                log=f"向量索引已生成并保存：{count} 个",
            )
            self._job_subtask(
                job_id, "schema_base", subtask_status="running", subtask_progress=10,
                progress=42,
                log="===== 文档预处理：等待项目批次 Schema 评估 =====",
                stage="schema_base",
            )
            # Shared content is parser/chunk/vector state.  Evolving the
            # project Schema here once per file made a multi-file build depend
            # on upload order and granted every file another full predicate
            # budget.  Enhanced construction evaluates all prepared targets in
            # one bounded corpus pass below.
            schema_record = self.schemas.ensure_base(int(project["id"]))
            self._job_subtask(
                job_id, "schema_base", subtask_status="completed", subtask_progress=100,
                detail="等待批次评估", progress=48,
                log="文档内容已就绪；逻辑 Schema 将在项目构建阶段统一评估",
            )
            self._assert_ingestion_active(job_id, document_id)
            revision_scope.__exit__(None, None, None)
            revision_scope = None
            document_values = {
                "status": "indexed", "indexed_at": datetime.now(timezone.utc),
                "error_message": None, "processing_stage": "chunk_ready",
                "effective_parser": plan.effective_parser,
                "effective_chunk_strategy": plan.effective_strategy,
                "parser_version": parsed.parser_version,
                "parse_quality": parsed.quality,
                "routing_reason": plan.routing_reason,
                "structure_ready_at": datetime.now(timezone.utc),
                "chunks_ready_at": datetime.now(timezone.utc),
                "graph_ready_at": None,
                "metadata": {
                    **process_metadata, "section_count": len(sections),
                    "chunk_count": len(chunks), "schema_version": schema_record["version"],
                    "shared_content_ready": True,
                    "schema_evaluation_scope": "project_document_batch",
                },
            }
            document, _ = self.db.activate_document_revision(
                document_id, str(revision["revision_id"]), job_id,
                str(revision["lease_token"]),
                reason="文档预处理结果已更新，策略图谱等待构建",
                document_values=document_values,
            )
            published = True
            project = self.db.get_project(int(project["id"])) or project
            return document, project, self.db.list_document_chunks(document_id)
        except Exception as exc:
            # The context is deliberately closed immediately before the
            # atomic publication transaction.  Cancellation can win in that
            # narrow interval, so cleanup keys off publication state rather
            # than whether the context manager is still entered.
            if revision and not published:
                self.db.fail_document_revision(
                    str(revision["revision_id"]), job_id,
                    str(revision["lease_token"]), str(exc),
                )
            cancelled = isinstance(
                exc, (IngestionCancelled, JobCancellationRequested),
            )
            if not published:
                self._restore_unpublished_document_attempt(
                    document_id, original_document, exc, cancelled=cancelled,
                )
                self._discard_staged_process_payload(payload_path)
                self._discard_staged_media_directory(media_dir)
            raise
        finally:
            if revision_scope is not None:
                revision_scope.__exit__(None, None, None)

    def _prepare_native_source(
        self, job_id: str, document_id: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Publish parser blocks without creating KF chunks, vectors, or Schema."""

        document = self.db.get_document(document_id)
        if not document:
            raise IngestionCancelled("文档不存在或已删除")
        original_document = dict(document)
        project = self.db.get_project(int(document["project_id"]))
        if not project:
            raise LookupError("项目不存在")
        self._assert_project_strategy(int(project["id"]), GRAPHRAG_NATIVE_STRATEGY)
        manifest = self._native_source_revision_manifest(document, project)
        self._assert_ingestion_active(job_id, document_id)
        if (
            self._shared_content_is_current(document, manifest)
            and bool(document.get("structure_ready_at"))
        ):
            self._assert_ingestion_active(job_id, document_id)
            preview = GraphRAGNativeInputBuilder(self).preview(
                project_id=int(project["id"]), document_id=document_id,
            )
            self._job(
                job_id, "running", 90,
                f"GraphRAG Native 源结构已就绪，复用 {preview['total']} 个结构化文档单元",
                document_id=document_id, stage="source_ready",
                structure_ready=True, chunks_ready=False, shared_content_reused=True,
                subtasks={
                    "structure": {
                        "status": "completed", "progress": 100,
                        "detail": "复用源结构",
                    },
                    "chunking": {"status": "waiting", "progress": 0, "detail": ""},
                    "schema_base": {
                        "status": "completed", "progress": 100,
                        "detail": f"{preview['total']} 个结构化文档单元",
                    },
                    "extraction": {"status": "waiting", "progress": 0, "detail": ""},
                    "graph_sync": {"status": "waiting", "progress": 0, "detail": ""},
                },
            )
            return document, project

        revision: dict[str, Any] | None = None
        payload_path: Path | None = None
        media_dir: Path | None = None
        published = False
        try:
            self._start_ingestion_job(job_id, document_id, mutate_graph=False)
            revision = self.db.begin_document_revision(document_id, job_id, manifest)
            source_path = Path(document["local_path"])
            plan, parsed = parse_document_structure(
                source_path,
                requested_parser=document.get("requested_parser") or "auto",
                requested_strategy="auto",
            )
            media_dir = (
                Config.media_root
                / Config.project_storage_key(int(project["id"]), project["name"])
                / str(document["doc_hash"][:16])
                / str(revision["revision_id"])
            )

            def media_progress(index: int, total: int, message: str) -> None:
                self._assert_ingestion_active(job_id, document_id)
                self._job_subtask(
                    job_id, "structure", subtask_status="running",
                    subtask_progress=20 + int(index / max(1, total) * 60),
                    detail=f"图片内容 {index}/{total}", progress=10, log=message,
                )

            parsed = self.media.enrich(
                source_path, parsed, project, media_dir, progress=media_progress,
            )
            process_root = (
                Config.chunks_root
                / Config.project_storage_key(int(project["id"]), project["name"])
            )
            process_root.mkdir(parents=True, exist_ok=True)
            payload_path = self._revision_process_payload_path(
                process_root, document, revision,
            )
            current_metadata = (
                document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
            )
            process_metadata = {
                **current_metadata,
                "chunk_file": str(payload_path),
                "media_dir": str(media_dir),
                "effective_parser": plan.effective_parser,
                "native_source_profile": "graphrag_native_source_v1",
                "native_source_ready": True,
            }
            process_result = ProcessingResult(plan, parsed, [], [])
            self._assert_ingestion_active(job_id, document_id)
            self._write_process_payload(
                payload_path, document["original_filename"],
                process_result,
            )
            self._job_subtask(
                job_id, "structure", subtask_status="completed", subtask_progress=100,
                detail=f"{parsed.quality.get('block_count', 0)} 个源结构块",
                progress=70,
                log=(
                    f"源结构解析完成：{parsed.quality.get('text_chars', 0)} 字符、"
                    f"{parsed.quality.get('block_count', 0)} 个结构块"
                ),
                stage="structure_ready", structure_ready=True, chunks_ready=False,
            )
            blocks = process_result.payload().get("blocks") or []
            native_units = GraphRAGNativeInputBuilder(self)._units_for_document(
                int(project["id"]), document, revision, blocks,
            )
            self._assert_ingestion_active(job_id, document_id)
            document, _ = self.db.activate_document_revision(
                document_id, str(revision["revision_id"]), job_id,
                str(revision["lease_token"]),
                reason="GraphRAG Native 源文档 revision 已更新",
                mutate_legacy_graph=False,
                document_values={
                    "status": "indexed",
                    "indexed_at": datetime.now(timezone.utc),
                    "error_message": None,
                    "processing_stage": "source_ready",
                    "effective_parser": plan.effective_parser,
                    "effective_chunk_strategy": None,
                    "parser_version": parsed.parser_version,
                    "parse_quality": parsed.quality,
                    "routing_reason": plan.routing_reason,
                    "structure_ready_at": datetime.now(timezone.utc),
                    "chunks_ready_at": None,
                    "graph_ready_at": None,
                    "metadata": process_metadata,
                },
            )
            published = True
            self._mark_graphrag_native_source_changed(
                int(project["id"]),
                "GraphRAG Native 源文档修订已更新，旧构建成果已失效",
            )
            self._job_subtask(
                job_id, "schema_base", subtask_status="completed",
                subtask_progress=100,
                detail=f"{len(native_units)} 个结构化文档单元",
                progress=95, log=f"结构化文档预检完成：{len(native_units)} 个",
            )
            return document, project
        except Exception as exc:
            if revision:
                self.db.fail_document_revision(
                    str(revision["revision_id"]), job_id,
                    str(revision["lease_token"]), str(exc),
                )
            cancelled = isinstance(
                exc, (IngestionCancelled, JobCancellationRequested),
            )
            if not published:
                self._restore_unpublished_document_attempt(
                    document_id, original_document, exc, cancelled=cancelled,
                )
                self._discard_staged_process_payload(payload_path)
                self._discard_staged_media_directory(media_dir)
            raise

    @recorded_job("content_preparation")
    def _prepare_document_content(
        self,
        job_id: str,
        document_id: int,
        strategy_key: str = KNOWLEDGE_FORGE_STRATEGY,
    ) -> None:
        """Prepare one document using the project's fixed strategy."""

        try:
            strategy_key = normalize_strategy(strategy_key)
            document = self.db.get_document(document_id)
            if not document:
                raise IngestionCancelled("文档不存在或已删除")
            self._assert_project_strategy(int(document["project_id"]), strategy_key)
            if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
                self._prepare_native_source(job_id, document_id)
                message = "GraphRAG Native 源文档准备完成，可以开始构建知识图谱"
                chunks_ready = False
            else:
                self._prepare_shared_content(job_id, document_id, mutate_graph=False)
                message = "文档预处理完成，可构建 Knowledge Forge"
                chunks_ready = True
            self._job(
                job_id, "completed", 100, message,
                stage="completed", active_subtask="", structure_ready=True,
                chunks_ready=chunks_ready, graph_ready=False,
            )
        except (IngestionCancelled, JobCancellationRequested) as exc:
            message = str(exc) or "预处理任务已安全终止"
            cancelled_subtasks = self._fail_active_ingestion_subtask(
                job_id, message, cancelled=True,
            )
            self.db.update_job(
                job_id, status="cancelled", progress=100,
                log=f"{message}；未完成预处理结果未发布",
                metadata={
                    "stage": "cancelled",
                    "cancel_acknowledged_at": datetime.now(timezone.utc).isoformat(),
                    **(
                        {"subtasks": cancelled_subtasks}
                        if cancelled_subtasks is not None else {}
                    ),
                },
            )
        except Exception as exc:
            if self._job_cancellation_requested(job_id):
                message = str(
                    ((self.db.get_job(job_id) or {}).get("metadata") or {}).get(
                        "cancel_reason"
                    ) or "预处理任务已安全终止"
                )
                cancelled_subtasks = self._fail_active_ingestion_subtask(
                    job_id, message, cancelled=True,
                )
                self.db.update_job(
                    job_id, status="cancelled", progress=100,
                    log=f"{message}；未完成预处理结果未发布",
                    metadata={
                        "stage": "cancelled",
                        "cancel_acknowledged_at": datetime.now(
                            timezone.utc,
                        ).isoformat(),
                        **(
                            {"subtasks": cancelled_subtasks}
                            if cancelled_subtasks is not None else {}
                        ),
                    },
                )
                return
            LOGGER.exception("Document preparation failed: %s", document_id)
            self._fail_active_ingestion_subtask(job_id, str(exc))
            self.db.update_job(
                job_id, status="failed", progress=100,
                log=f"预处理失败：{exc}", error=str(exc),
                metadata={"stage": "failed"},
            )

    @recorded_job("ingestion")
    def _ingest_document(self, job_id: str, document_id: int) -> None:
        revision: dict[str, Any] | None = None
        revision_scope: Any = None
        graph_mutation_started = False
        try:
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self._start_ingestion_job(job_id, document_id)
                graph_mutation_started = True
                self.db.update_document(document_id, status="processing", error_message=None,
                                        processing_stage="structure", structure_ready_at=None,
                                        chunks_ready_at=None, graph_ready_at=None)
            document = self.db.get_document(document_id)
            if not document: raise LookupError("文档不存在")
            project = self.db.get_project(int(document["project_id"]))
            if not project: raise LookupError("项目不存在")
            self._mark_graphrag_native_source_changed(
                int(project["id"]), "文档正在生成新的 active revision，native source snapshot 已失效",
            )
            revision_manifest = self._document_revision_manifest(
                document, project,
            )
            revision = self.db.begin_document_revision(document_id, job_id, revision_manifest)
            if revision.get("already_active"):
                self._job(job_id, "completed", 100, "相同入库 run 已完成，无需重复发布", stage="completed")
                return
            revision_scope = self.db.content_revision_context(
                str(revision["revision_id"]), job_id, str(revision["lease_token"]),
            )
            revision_scope.__enter__()
            source_path = Path(document["local_path"])
            plan, parsed = parse_document_structure(
                source_path,
                requested_parser=document.get("requested_parser") or "auto",
                requested_strategy=document.get("requested_chunk_strategy") or "auto",
            )
            media_dir = (
                Config.media_root
                / Config.project_storage_key(int(project["id"]), project["name"])
                / str(document["doc_hash"][:16])
            )

            def media_progress(index: int, total: int, message: str) -> None:
                self._assert_ingestion_active(job_id, document_id)
                amount = 20 + int(index / max(1, total) * 55)
                self._job_subtask(
                    job_id,
                    "structure",
                    subtask_status="running",
                    subtask_progress=amount,
                    detail=f"图片内容 {index}/{total}",
                    progress=8 + int(index / max(1, total) * 4),
                    log=message,
                )

            parsed = self.media.enrich(
                source_path,
                parsed,
                project,
                media_dir,
                progress=media_progress,
            )
            chunk_dir = Config.chunks_root / Config.project_storage_key(int(project["id"]), project["name"])
            chunk_dir.mkdir(parents=True, exist_ok=True)
            payload_path = chunk_dir / f"{document['doc_hash'][:16]}.json"
            current_metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
            process_metadata = {**current_metadata, "chunk_file": str(payload_path),
                                "media_dir": str(media_dir),
                                "effective_parser": plan.effective_parser,
                                "effective_strategy": plan.effective_strategy}
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self._write_process_payload(payload_path, document["original_filename"], ProcessingResult(plan, parsed, [], []))
                self.db.update_document(
                    document_id,
                    effective_parser=plan.effective_parser,
                    effective_chunk_strategy=plan.effective_strategy,
                    parser_version=parsed.parser_version,
                    parse_quality=parsed.quality,
                    routing_reason=plan.routing_reason,
                    processing_stage="structure_ready",
                    structure_ready_at=datetime.now(timezone.utc),
                    metadata=process_metadata,
                )
            self._job_subtask(
                job_id, "structure", subtask_status="running", subtask_progress=80,
                detail=f"{parsed.quality.get('block_count', 0)} 个结构块", progress=12,
                log=f"处理路由：{plan.effective_parser} + {plan.effective_strategy}；{plan.routing_reason}",
            )
            self._job_subtask(
                job_id, "structure", subtask_status="completed", subtask_progress=100,
                detail=f"{parsed.quality.get('block_count', 0)} 个结构块", progress=18,
                log=(f"解析质量：{parsed.quality.get('text_chars', 0)} 字符、"
                     f"{parsed.quality.get('block_count', 0)} 个结构块、"
                     f"{parsed.quality.get('table_count', 0)} 个表格、"
                     f"{parsed.quality.get('image_count', 0)} 张图片"
                     + (f"、Docling 加速器 {parsed.quality['docling_accelerator']}"
                        if parsed.quality.get("docling_accelerator") else "")
                     + (f"（{parsed.quality['docling_gpu_fallback_reason']}）"
                        if parsed.quality.get("docling_gpu_fallback_reason") else "")),
                stage="structure_ready", structure_ready=True,
            )
            self._job_subtask(
                job_id, "chunking", subtask_status="running", subtask_progress=0,
                progress=20, log="===== 阶段二：知识分块与向量索引 =====", stage="chunking",
            )

            result = build_processing_result(
                source_path,
                plan,
                parsed,
                source_name=document["original_filename"],
                chunk_max_tokens=int(project.get("chunk_max_tokens") or self.DEFAULT_CHUNK_MAX_TOKENS),
            )
            sections, chunks = result.sections, result.chunks
            section_values = [{**asdict(item), "content_hash": hashlib.sha256(item.content.encode()).hexdigest(),
                               "metadata": {**item.metadata, "source": document["original_filename"]}}
                              for item in sections]
            chunk_values = []
            for item in chunks:
                value = asdict(item); value["metadata"] = {**item.metadata, "project_id": str(project["id"]),
                    "project_key": project["project_key"], "document_id": str(document["id"]),
                    "document_name": document["original_filename"], "document_url": document["storage_path"],
                    "section_key": item.section_key, "section_title": item.title,
                    "token_count": item.token_count, "parent_key": item.parent_key}
                chunk_values.append(value)
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self._write_process_payload(payload_path, document["original_filename"], result)
                stored_chunks = self.db.replace_document_content(
                    document_id,
                    section_values,
                    chunk_values,
                    revision_id=str(revision["revision_id"]),
                    ingestion_run_id=job_id,
                    lease_token=str(revision["lease_token"]),
                    manifest=revision_manifest,
                )
                self.db.update_document(document_id, processing_stage="chunk_ready",
                                        chunks_ready_at=datetime.now(timezone.utc),
                                        metadata={**process_metadata, "section_count": len(sections),
                                                  "chunk_count": len(chunks)})
            self._job_subtask(
                job_id, "chunking", subtask_status="running", subtask_progress=60,
                detail=f"{len(chunks)} 个知识块", progress=32,
                log=f"已生成 {len(sections)} 个章节和 {len(chunks)} 个知识块，并写入 PostgreSQL",
                stage="chunk_ready", chunks_ready=True,
            )

            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                count = self.vectors.index_chunks(project, document_id, stored_chunks)
            self._job_subtask(
                job_id, "chunking", subtask_status="completed", subtask_progress=100,
                detail=f"{count} 个向量索引", progress=42, log=f"已生成并保存 {count} 个向量索引",
            )
            self.db.update_document(document_id, processing_stage="graph")
            self._job_subtask(
                job_id, "schema_base", subtask_status="running", subtask_progress=0,
                progress=44, log="===== 阶段三：Schema、实体关系与知识图谱 =====", stage="graph",
            )
            semantic_chunks = self._schema_evaluation_chunks(stored_chunks)
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                schema_record = self.schemas.evaluate_document(project, document_id, semantic_chunks)
            project = self.db.get_project(int(project["id"])) or project
            evaluation = schema_record.get("_evaluation") or {}
            sampling = evaluation.get("sampling") or {}
            if evaluation.get("reviewed"):
                schema_detail = (
                    f"已从 {sampling.get('selected_chunks', []) and len(sampling['selected_chunks']) or 0} 个代表性知识块评审；"
                    f"新增 {len(evaluation.get('accepted_entity_types') or [])} 类实体、"
                    f"{len(evaluation.get('accepted_relation_types') or [])} 类关系"
                )
            else:
                schema_detail = f"无需扩展（{evaluation.get('reason') or '未触发'}）"
            self._job_subtask(
                job_id, "schema_base", subtask_status="running", subtask_progress=55,
                detail=f"Schema v{schema_record['version']}", progress=48,
                log=f"逻辑 Schema v{schema_record['version']}：{schema_detail}",
            )

            self._job_subtask(
                job_id, "schema_base", subtask_status="completed", subtask_progress=100,
                detail=f"{len(stored_chunks)} 个基础知识块", progress=50,
                log=f"Schema 与 {len(stored_chunks)} 个基础知识块已就绪，等待关系发布后统一重建图谱",
            )
            extraction_chunks = self._graph_extraction_chunks(stored_chunks)
            self._job_subtask(
                job_id, "extraction", subtask_status="running", subtask_progress=0,
                detail=f"0/{len(extraction_chunks)}", progress=52,
            )
            resolution_session = self.resolver.start_session(project)
            total_mentions = total_relations = extraction_errors = 0
            gleaning_chunks = verified_relations = rejected_relations = 0
            verification_totals = {
                "verification_requested": 0,
                "rules_verified": 0,
                "llm_verified": 0,
                "rejected": 0,
                "uncertain": 0,
                "missing_decisions": 0,
                "verifier_failures": 0,
                "retry_count": 0,
                "quarantined": 0,
                "published": 0,
            }
            for index, chunk in enumerate(extraction_chunks, start=1):
                try:
                    self._assert_ingestion_active(job_id, document_id)
                    extraction = self.schemas.extract_chunk(project, schema_record, chunk)
                    diagnostics = extraction.get("_diagnostics") or {}
                    with self._document_lock(document_id):
                        self._assert_ingestion_active(job_id, document_id)
                        mentions, relations = self.resolver.resolve_chunk(
                            project, document_id, chunk, extraction, int(schema_record["version"]),
                            session=resolution_session, ingestion_run_id=job_id,
                        )
                    total_mentions += len(mentions); total_relations += len(relations)
                    detail = f"{len(mentions)} 个实体提及，{len(relations)} 条关系"
                    if diagnostics.get("gleaning_used"):
                        gleaning_chunks += 1
                        detail += "，已执行针对性补抽取"
                    verified_relations += int(diagnostics.get("verified") or 0)
                    rejected_relations += int(diagnostics.get("verification_rejected") or 0)
                    for key in verification_totals:
                        verification_totals[key] += int(diagnostics.get(key) or 0)
                    if diagnostics.get("verification_candidates"):
                        detail += (
                            f"，复核 {int(diagnostics['verification_candidates'])} 条候选"
                            f"（发布 {int(diagnostics.get('published') or 0)}，隔离 "
                            f"{int(diagnostics.get('quarantined') or 0)}）"
                        )
                except IngestionCancelled:
                    raise
                except Exception as exc:
                    extraction_errors += 1
                    LOGGER.warning("Chunk graph extraction failed: %s: %s", chunk.get("chunk_key"), exc)
                    detail = f"抽取失败，已保留文本和向量：{exc}"
                progress = 52 + int(index / max(1, len(extraction_chunks)) * 38)
                extraction_progress = int(index / max(1, len(extraction_chunks)) * 100)
                chunk_label = chunk.get("section_title") or chunk.get("chunk_key") or f"知识块 {index}"
                self._job_subtask(
                    job_id, "extraction",
                    subtask_status="completed" if index == len(extraction_chunks) else "running",
                    subtask_progress=extraction_progress,
                    detail=f"{index}/{len(extraction_chunks)}",
                    progress=progress,
                    log=f"图谱抽取 {index}/{len(extraction_chunks)} [{chunk_label}]：{detail}",
                )
            if not extraction_chunks:
                self._job_subtask(
                    job_id, "extraction", subtask_status="completed", subtask_progress=100,
                    detail="0/0", progress=90,
                )
            self._job_subtask(
                job_id, "graph_sync", subtask_status="running", subtask_progress=0,
                detail="正在同步", progress=92,
            )
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self.db.retire_orphan_entities(int(project["id"]))
            self._assert_ingestion_active(job_id, document_id)
            document_values = {
                "status": "indexed",
                "indexed_at": datetime.now(timezone.utc),
                "error_message": None,
                "processing_stage": "completed",
                "graph_ready_at": None,
                "metadata": {
                    **process_metadata,
                    "section_count": len(sections),
                    "chunk_count": len(chunks),
                    "effective_parser": plan.effective_parser,
                    "effective_strategy": plan.effective_strategy,
                    "schema_version": schema_record["version"],
                    "entity_mentions": total_mentions,
                    "relations": total_relations,
                    "extraction_errors": extraction_errors,
                    "gleaning_chunks": gleaning_chunks,
                    "verified_relations": verified_relations,
                    "rejected_relations": rejected_relations,
                    "relation_verification": verification_totals,
                },
            }
            if hasattr(self.db, "finalize_document_graph_facts"):
                _, graph_project = self.db.finalize_document_graph_facts(
                    document_id,
                    reason="文档事实已更新，等待当前代次图谱重建",
                    revision_id=str(revision["revision_id"]),
                    ingestion_run_id=job_id,
                    lease_token=str(revision["lease_token"]),
                    **document_values,
                )
                # Activation changes this revision from ``building`` to ``active``.
                # Subsequent graph publication updates active projections and must
                # therefore run without the now-stale building-lease context.
                revision_scope.__exit__(None, None, None)
                revision_scope = None
            else:
                self.db.update_document(document_id, **document_values)
                if hasattr(self.db, "mark_graph_stale"):
                    graph_project = self.db.mark_graph_stale(
                        int(project["id"]), "文档事实已更新，等待当前代次图谱重建",
                    )
                else:
                    graph_project = self.db.set_graph_state(
                        int(project["id"]), GRAPH_STALE,
                        error="文档事实已更新，等待当前代次图谱重建",
                    )
            target_generation = int(graph_project.get("graph_generation") or 0)
            graph_report = self._rebuild_at_quiescence(
                int(project["id"]), job_id, target_generation,
            )
            if graph_report is None:
                self._job_subtask(
                    job_id,
                    "graph_sync",
                    subtask_status="completed",
                    subtask_progress=100,
                    detail="已合并到批次最终重建",
                    progress=96,
                    log="检测到同项目后续文档，跳过本次全量清图并合并到批次最终重建",
                )
                self._job(
                    job_id,
                    "completed",
                    100,
                    "文档知识库入库完成；知识图谱将在当前批次静默点统一重建",
                    stage="completed",
                    graph_ready=False,
                    graph_rebuild_coalesced=True,
                    target_generation=target_generation,
                    extraction_errors=extraction_errors,
                    relation_verification=verification_totals,
                    published_to_hugegraph=0,
                )
                return
            self._assert_ingestion_active(job_id, document_id)
            graph_ready_at = datetime.now(timezone.utc)
            self._job_subtask(
                job_id, "graph_sync", subtask_status="completed", subtask_progress=100,
                detail=f"{graph_report['entities']} 个实体，{graph_report['relations']} 条关系",
                progress=96,
                log=f"HugeGraph 同步完成：{graph_report['entities']} 个实体，{graph_report['relations']} 条关系",
            )
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self.db.update_document(
                    document_id,
                    processing_stage="completed",
                    graph_ready_at=graph_ready_at,
                )
            completion = "文档知识库与知识图谱入库完成"
            if extraction_errors:
                completion += f"；{extraction_errors} 个知识块的图谱抽取已降级"
            completion += (
                f"；补抽取 {gleaning_chunks} 个知识块；关系复核请求 "
                f"{verification_totals['verification_requested']}，规则通过 {verification_totals['rules_verified']}，"
                f"LLM 通过 {verification_totals['llm_verified']}，拒绝 {verification_totals['rejected']}，"
                f"不确定 {verification_totals['uncertain']}，漏答 {verification_totals['missing_decisions']}，"
                f"复核失败 {verification_totals['verifier_failures']}，重试 {verification_totals['retry_count']}；"
                f"隔离 {verification_totals['quarantined']}，发布到 HugeGraph {graph_report['relations']}"
            )
            self._job(job_id, "completed", 100, completion, stage="completed", graph_ready=True,
                      extraction_errors=extraction_errors, gleaning_chunks=gleaning_chunks,
                      verified_relations=verified_relations, rejected_relations=rejected_relations,
                      relation_verification=verification_totals,
                      published_to_hugegraph=graph_report["relations"])
        except IngestionCancelled as exc:
            LOGGER.info("Document ingestion cancelled: %s: %s", document_id, exc)
            if revision:
                self.db.fail_document_revision(
                    str(revision["revision_id"]), job_id, str(revision["lease_token"]), str(exc),
                )
            job = self.db.get_job(job_id)
            if job and job.get("project_id") and graph_mutation_started:
                self.db.set_graph_state(
                    int(job["project_id"]), GRAPH_STALE,
                    error="文档任务已取消，图谱需要重新同步",
                )
            cancelled_subtasks = self._fail_active_ingestion_subtask(
                job_id, str(exc), cancelled=True,
            )
            if job and job.get("status") != "cancelled":
                self.db.update_job(
                    job_id, status="cancelled", progress=100,
                    log=f"{exc}；未完成结果未发布",
                    metadata={
                        "stage": "cancelled",
                        "cancel_acknowledged_at": datetime.now(timezone.utc).isoformat(),
                        **(
                            {"subtasks": cancelled_subtasks}
                            if cancelled_subtasks is not None else {}
                        ),
                    },
                )
        except Exception as exc:
            job = self.db.get_job(job_id)
            if revision:
                self.db.fail_document_revision(
                    str(revision["revision_id"]), job_id, str(revision["lease_token"]), str(exc),
                )
            job_metadata = (
                job.get("metadata") if job and isinstance(job.get("metadata"), dict)
                else {}
            )
            if (
                (job and job.get("status") == "cancelled")
                or job_metadata.get("cancel_requested")
                or not self.db.get_document(document_id)
            ):
                LOGGER.info("Document ingestion stopped after deletion/cancellation: %s: %s", document_id, exc)
                if job and job.get("project_id") and graph_mutation_started:
                    self.db.set_graph_state(
                        int(job["project_id"]), GRAPH_STALE,
                        error="文档任务已取消，图谱需要重新同步",
                    )
                cancelled_subtasks = self._fail_active_ingestion_subtask(
                    job_id, "文档已删除，任务已取消", cancelled=True,
                )
                if job and job.get("status") != "cancelled":
                    self.db.update_job(
                        job_id,
                        status="cancelled",
                        progress=100,
                        log="文档已删除，任务已取消",
                        metadata={
                            "stage": "cancelled",
                            "cancel_acknowledged_at": datetime.now(timezone.utc).isoformat(),
                            **(
                                {"subtasks": cancelled_subtasks}
                                if cancelled_subtasks is not None else {}
                            ),
                        },
                    )
                return
            LOGGER.exception("Document ingestion failed: %s", document_id)
            current_document = self.db.get_document(document_id)
            if job and job.get("project_id") and not (current_document or {}).get("active_revision_id"):
                self.db.set_graph_state(int(job["project_id"]), GRAPH_FAILED, error=str(exc))
            self._fail_active_ingestion_subtask(job_id, str(exc))
            self.db.update_document(
                document_id,
                status="indexed" if (current_document or {}).get("active_revision_id") else "failed",
                processing_stage="failed",
                error_message=str(exc),
            )
            self.db.update_job(job_id, status="failed", progress=100, log=f"处理失败：{exc}", error=str(exc),
                               metadata={"stage": "failed"})
        finally:
            if revision_scope is not None:
                revision_scope.__exit__(None, None, None)
            try:
                job = self.db.get_job(job_id) or {}
                if normalize_strategy(job.get("strategy_key")) == STABLE_STRATEGY:
                    current_document = self.db.get_document_including_deleted(document_id) or {}
                    job_status = str(job.get("status") or "failed")
                    status = {
                        "queued": "queued", "running": "processing",
                        "completed": "indexed", "cancelled": "cancelled",
                    }.get(job_status, "failed")
                    metadata = current_document.get("metadata") or {}
                    self.db.update_document_strategy_build(
                        document_id,
                        STABLE_STRATEGY,
                        status="deleted" if current_document.get("is_deleted") else status,
                        content_revision_id=current_document.get("active_revision_id"),
                        latest_job_id=job_id,
                        stats={
                            "section_count": int(metadata.get("section_count") or 0),
                            "chunk_count": int(metadata.get("chunk_count") or 0),
                            "entity_mentions": int(metadata.get("entity_mentions") or 0),
                            "relations": int(metadata.get("relations") or 0),
                        },
                        error_message=str(job.get("error_message") or "") if status == "failed" else "",
                        started_at=job.get("started_at"),
                        completed_at=(
                            datetime.now(timezone.utc)
                            if status in {"indexed", "failed", "cancelled", "deleted"}
                            else None
                        ),
                    )
            except Exception:
                LOGGER.exception("Unable to persist stable strategy build status")

    @recorded_job("knowledge_forge_strategy_ingestion")
    def _ingest_knowledge_forge_strategy(self, job_id: str, document_id: int) -> None:
        """Build the isolated grounded relation and entity knowledge graph."""

        strategy_key = KNOWLEDGE_FORGE_STRATEGY
        lease: dict[str, Any] | None = None
        mutation: GraphMutationContext | None = None
        state: dict[str, Any] | None = None
        job = self.db.get_job(job_id) or {}
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        requested_knowledge_language = metadata.get("knowledge_language")
        target_document_ids = [
            int(value) for value in (metadata.get("target_document_ids") or [document_id])
        ]
        schema_discovery = bool(Config.schema_discovery_during_build)
        try:
            # A queued job can be cancelled after the executor accepted it but
            # before this worker gets CPU time.  Stop before touching readiness
            # state or per-document build rows in that race.
            self._assert_ingestion_active(job_id, document_id)
            getattr(
                self.db,
                "assert_grounded_relation_ready",
                self.db.assert_grounded_relation_ready,
            )()
            prerequisite_subtasks = self._empty_ingestion_subtasks()
            prerequisite_subtasks["structure"] = {
                "status": "waiting", "progress": 0,
                "detail": "正在确认文档结构",
            }
            prerequisite_subtasks["chunking"] = {
                "status": "waiting", "progress": 0,
                "detail": "正在确认知识块与向量",
            }
            self._job(
                job_id, "running", 0,
                "===== Knowledge Forge：确认构建输入 =====",
                stage="build_prerequisites", active_subtask="structure",
                subtasks=prerequisite_subtasks,
                prerequisite_status="checking",
                prepared_document_count=0,
                prepared_chunk_count=0,
                schema_discovery_during_build=schema_discovery,
            )
            project: dict[str, Any] | None = None
            prepared_chunk_count = 0
            for index, target_id in enumerate(target_document_ids, start=1):
                document = self.db.get_document(target_id)
                if not document:
                    raise IngestionCancelled("文档不存在或已删除")
                self._job(
                    job_id, "running", 0,
                    f"===== 构建输入 [{index}/{len(target_document_ids)}] "
                    f"{document['original_filename']} =====",
                    active_document_id=target_id,
                    active_document_index=index,
                    target_document_count=len(target_document_ids),
                    prerequisite_status="checking",
                )
                self.db.update_document_strategy_build(
                    target_id, strategy_key, status="processing", latest_job_id=job_id,
                    error_message="", started_at=datetime.now(timezone.utc), completed_at=None,
                )
                document, project, chunks = self._prepare_shared_content(
                    job_id, target_id,
                    require_reusable=True,
                    report_progress=False,
                )
                prepared_chunk_count += len(chunks)
            if project is None:
                raise RuntimeError("Knowledge Forge 没有可处理文档")
            build_knowledge_language = normalize_knowledge_language(
                requested_knowledge_language
                if requested_knowledge_language is not None
                else project.get("knowledge_language") or "auto",
            )
            # Freeze all model-generated knowledge in this build to the
            # submission-time project setting.  A settings edit made while the
            # job is queued or running must only affect the next build.
            project = {
                **project,
                "knowledge_language": build_knowledge_language,
            }
            prerequisite_subtasks["structure"] = {
                "status": "completed", "progress": 100,
                "detail": f"复用 {len(target_document_ids)} 份文档结构",
            }
            prerequisite_subtasks["chunking"] = {
                "status": "completed", "progress": 100,
                "detail": f"复用 {prepared_chunk_count} 个知识块与向量",
            }
            prerequisite_subtasks["schema_base"] = {
                "status": "running", "progress": 0,
                "detail": (
                    "正在评估知识结构"
                    if schema_discovery else "正在确认当前知识结构"
                ),
            }
            self._job(
                job_id, "running", 0,
                "预处理结果已确认，开始确认本次构建使用的知识结构",
                stage="schema_base", active_subtask="schema_base",
                subtasks=prerequisite_subtasks,
                prerequisite_status="ready",
                prepared_document_count=len(target_document_ids),
                prepared_chunk_count=prepared_chunk_count,
                structure_ready=True, chunks_ready=True,
                shared_content_reused=True,
                schema_discovery_during_build=schema_discovery,
            )
            project_id = int(project["id"])
            self._assert_ingestion_active(job_id, document_id)
            eligible_documents = [
                item for item in self.db.list_documents(project_id)
                if item.get("status") == "indexed" and item.get("active_revision_id")
                and not item.get("is_deleted")
            ]
            if not eligible_documents:
                raise RuntimeError("项目尚无可用于 Knowledge Forge 的预处理结果")
            target_id_set = set(target_document_ids)
            schema_documents = [
                item for item in eligible_documents
                if int(item["id"]) in target_id_set
            ]
            if not schema_documents:
                raise RuntimeError("本次构建没有可用于批次 Schema 评估的文档")
            self._job_subtask(
                job_id, "schema_base", subtask_status="running",
                subtask_progress=35,
                detail=(
                    f"正在评估 {len(schema_documents)} 个文档"
                    if schema_discovery else "正在冻结活动 Schema"
                ),
                progress=self._knowledge_forge_build_progress(
                    schema_progress=35,
                    schema_discovery=schema_discovery,
                ),
                log=(
                    (
                        "===== Knowledge Forge：项目批次逻辑 Schema =====\n"
                        f"正在对 {len(schema_documents)} 个不可变文档修订执行一次"
                        "分层抽样、语义去重和增量发布"
                    )
                    if schema_discovery else (
                        "===== Knowledge Forge：活动逻辑 Schema 快照 =====\n"
                        "普通构建复用当前活动 Schema；开放原文关系独立发布，"
                        "新增类型由独立的 Schema 重新评估任务治理"
                    )
                ),
                stage="schema_base",
            )
            self._assert_ingestion_active(job_id, document_id)
            schema_record = self._knowledge_forge_build_schema(
                project, schema_documents,
            )
            schema_evaluation = schema_record.get("_evaluation") or {}
            for item in schema_documents:
                current_metadata = (
                    item.get("metadata")
                    if isinstance(item.get("metadata"), dict) else {}
                )
                updated_metadata = {
                    **current_metadata,
                    "schema_version": int(schema_record["version"]),
                    "schema_evaluation_scope": str(
                        schema_evaluation.get("scope")
                        or "project_build_active_schema_snapshot"
                    ),
                }
                if schema_evaluation.get("input_fingerprint"):
                    updated_metadata["schema_input_fingerprint"] = str(
                        schema_evaluation["input_fingerprint"]
                    )
                self.db.update_document(int(item["id"]), metadata=updated_metadata)
            project = {
                **(self.db.get_project(project_id) or project),
                "knowledge_language": build_knowledge_language,
            }
            sampling = schema_evaluation.get("sampling") or {}
            if schema_evaluation.get("reviewed"):
                schema_detail = (
                    f"抽样 {len(sampling.get('selected_chunks') or [])} 个知识块；"
                    f"新增 {len(schema_evaluation.get('accepted_entity_types') or [])} 类实体、"
                    f"{len(schema_evaluation.get('accepted_relation_types') or [])} 类关系"
                )
            elif schema_discovery:
                schema_detail = (
                    f"复用或无需扩展（{schema_evaluation.get('reason') or '未触发'}）"
                )
            else:
                schema_detail = "复用当前版本"
            self._job_subtask(
                job_id, "schema_base", subtask_status="completed", subtask_progress=100,
                detail=f"Schema v{schema_record['version']} · {schema_detail}",
                progress=self._knowledge_forge_build_progress(
                    schema_progress=100,
                    schema_discovery=schema_discovery,
                ),
                log=(
                    "===== Knowledge Forge：关系验证 + 实体归一 =====\n"
                    f"逻辑 Schema v{schema_record['version']} 已按项目批次冻结："
                    f"{schema_detail}。开始构建隔离实体关系快照"
                ),
                stage="graph", strategy_key=strategy_key, graph_ready=False,
                schema_version=int(schema_record["version"]),
            )
            self._job_subtask(
                job_id, "extraction", subtask_status="running", subtask_progress=0,
                detail="实体归一与关系验证处理中",
                progress=self._knowledge_forge_build_progress(
                    schema_progress=100,
                    schema_discovery=schema_discovery,
                ),
            )

            last_grounded_progress = 5

            def grounded_progress(done: int, total: int, message: str) -> None:
                nonlocal last_grounded_progress
                self._assert_ingestion_active(job_id, document_id)
                mapped_progress, detail = (
                    self._knowledge_forge_extraction_progress(
                        done, total, message,
                    )
                )
                # Defensive monotonicity protects the public progress bar if a
                # future nested phase reports a fresh local counter.
                last_grounded_progress = max(
                    last_grounded_progress, mapped_progress,
                )
                overall_progress = self._knowledge_forge_build_progress(
                    schema_progress=100,
                    extraction_progress=last_grounded_progress,
                    schema_discovery=schema_discovery,
                )
                self._job_subtask(
                    job_id, "extraction", subtask_status="running",
                    subtask_progress=last_grounded_progress,
                    detail=detail,
                    progress=overall_progress,
                    log=message,
                )

            grounded_snapshot = self.build_grounded_entity_namespace(
                project_id, progress=grounded_progress,
            )
            namespace = grounded_snapshot.get("namespace") or {}
            if namespace.get("status") != "ready" or not namespace.get("snapshot_complete"):
                raise RuntimeError("Knowledge Forge 快照未完整收敛")
            persisted_mentions = self.db.list_grounded_entity_mentions(
                grounded_snapshot["namespace_id"]
            )
            grounded_snapshot = project_publishable_snapshot(
                grounded_snapshot,
                mentions=persisted_mentions,
            )
            effective_knowledge_language = resolve_knowledge_language(
                build_knowledge_language,
                [
                    str(item.get("evidence_text") or item.get("surface") or "")
                    for item in grounded_snapshot.get("mentions") or []
                ] + [
                    str(item.get("evidence_text") or "")
                    for item in grounded_snapshot.get("relation_endpoints") or []
                ],
            )
            grounded_snapshot = enrich_knowledge_forge_snapshot(
                grounded_snapshot,
                mentions=list(grounded_snapshot["mentions"]),
                knowledge_language=effective_knowledge_language,
            )
            metrics = namespace.get("metrics") or {}
            self._job_subtask(
                job_id, "extraction", subtask_status="completed", subtask_progress=100,
                detail=(
                    f"{len(grounded_snapshot.get('canonicals') or [])} 个实体，"
                    f"{len(grounded_snapshot.get('relation_endpoints') or [])} 条关系"
                ),
                progress=self._knowledge_forge_build_progress(
                    schema_progress=100,
                    extraction_progress=100,
                    schema_discovery=schema_discovery,
                ),
                log=(
                    f"Knowledge Forge 快照完成：{len(grounded_snapshot['mentions'])} 个实体提及，"
                    f"{len(grounded_snapshot.get('canonicals') or [])} 个规范实体，"
                    f"{len(grounded_snapshot.get('relation_endpoints') or [])} 条可发布关系"
                ),
            )
            state = self.db.start_strategy_graph_rebuild(project_id, strategy_key)
            generation = int(state["generation"])
            namespace_id = str(state["namespace_id"])
            policy_hash = hashlib.sha256(json.dumps({
                "strategy": strategy_key,
                "schema_version": int(schema_record["version"]),
                "schema_discovery_during_build": schema_discovery,
                "knowledge_language_setting": build_knowledge_language,
                "knowledge_language": effective_knowledge_language,
                "entity_policy": dict(GROUNDED_ENTITY_POLICY_SET),
                "entity_publishability_policy": (
                    ENTITY_PUBLISHABILITY_POLICY_VERSION
                ),
                "relation_policy": dict(GROUNDED_RELATION_POLICY_SET),
                "relation_v2_run_id": metrics.get("relation_v2_run_id") or "",
                "relation_v2_fingerprint": metrics.get("relation_v2_fingerprint") or "",
                "artifact_hash": namespace.get("artifact_hash") or "",
            }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            lease = self.db.acquire_graph_lease(
                project_id, namespace_id, generation,
                owner=f"strategy:{strategy_key}:{job_id}", ttl_seconds=900,
            )
            mutation = GraphMutationContext(
                namespace_id=namespace_id, project_id=project_id,
                generation=generation, policy_hash=policy_hash,
                lease_token=str(lease["lease_token"]),
            )
            chunks_by_document = {
                int(item["id"]): self.db.list_document_chunks(int(item["id"]))
                for item in eligible_documents
            }
            self._job_subtask(
                job_id, "graph_sync", subtask_status="running", subtask_progress=10,
                detail="写入隔离图",
                progress=self._knowledge_forge_build_progress(
                    schema_progress=100,
                    extraction_progress=100,
                    graph_sync_progress=10,
                    schema_discovery=schema_discovery,
                ),
            )
            graph_stats = self.graph.replace_strategy_snapshot(
                project, eligible_documents, chunks_by_document, grounded_snapshot,
                strategy_key=strategy_key, mutation=mutation,
                checkpoint=lambda: self._assert_ingestion_active(
                    job_id, document_id,
                ),
            )
            self._assert_ingestion_active(job_id, document_id)
            self._job_subtask(
                job_id, "graph_sync", subtask_status="running", subtask_progress=70,
                detail="正在识别图谱社区",
                progress=self._knowledge_forge_build_progress(
                    schema_progress=100,
                    extraction_progress=100,
                    graph_sync_progress=70,
                    schema_discovery=schema_discovery,
                ),
                log="Knowledge Forge 图谱写入完成，正在基于当前规范实体和已验证关系识别社区",
            )
            community_run = self.knowledge_forge_communities.detect(
                project_id=project_id,
                graph_generation=generation,
                entity_namespace_id=str(grounded_snapshot["namespace_id"]),
                snapshot=grounded_snapshot,
                config=CommunityConfig(
                    seed=42, max_cluster_size=10,
                    use_lcc=False, timeout_seconds=300,
                ),
                cancel_requested=lambda: self._job_cancellation_requested(job_id),
                knowledge_language=effective_knowledge_language,
            )
            self._assert_ingestion_active(job_id, document_id)
            completed = self.db.complete_strategy_graph_rebuild(
                project_id, strategy_key, generation,
                stats=graph_stats,
                policy_summary={
                    "mode": "grounded",
                    "policy_hash": policy_hash,
                    "schema_version": int(schema_record["version"]),
                    "schema_discovery_during_build": schema_discovery,
                    "knowledge_language_setting": build_knowledge_language,
                    "knowledge_language": effective_knowledge_language,
                    "entity_namespace_id": grounded_snapshot["namespace_id"],
                    "entity_metrics": metrics,
                    "community_run_id": str(community_run["run_id"]),
                    "community_input_hash": str(community_run["input_hash"]),
                },
                empty=not graph_stats["documents"],
                job_id=job_id,
            )
            if not completed:
                raise RuntimeError("Knowledge Forge 图代次已过期，拒绝发布就绪状态")
            now = datetime.now(timezone.utc)
            for item in eligible_documents:
                self.db.ensure_document_strategy_builds(int(item["id"]))
                build_values: dict[str, Any] = {
                    "status": "indexed",
                    "content_revision_id": item.get("active_revision_id"),
                    "stats": {
                        "entity_mentions": sum(
                            1 for mention in grounded_snapshot["mentions"]
                            if int(mention["document_id"]) == int(item["id"])
                        ),
                        "relations": sum(
                            1 for relation in grounded_snapshot.get("relation_endpoints") or []
                            if int(relation["document_id"]) == int(item["id"])
                        ),
                        "graph_generation": generation,
                    },
                    "error_message": "",
                    "completed_at": now,
                }
                if int(item["id"]) in target_document_ids:
                    build_values["latest_job_id"] = job_id
                self.db.update_document_strategy_build(
                    int(item["id"]), strategy_key, **build_values,
                )
            self._job_subtask(
                job_id, "graph_sync", subtask_status="completed", subtask_progress=100,
                detail=f"{graph_stats['entities']} 个实体，{graph_stats['relations']} 条关系",
                progress=100,
                log=(
                    f"Knowledge Forge 图谱同步完成：{graph_stats['entities']} 个实体，"
                    f"{graph_stats['relations']} 条关系，"
                    f"{int(community_run.get('community_count') or 0)} 个社区"
                ),
            )
            self._job(
                job_id, "completed", 100, "Knowledge Forge 构建完成，当前图谱已发布",
                stage="completed", graph_ready=True, strategy_key=strategy_key,
                graph_stats=graph_stats,
                knowledge_language=effective_knowledge_language,
            )
            self._schedule_knowledge_forge_community_reports(
                project_id,
                knowledge_language=effective_knowledge_language,
                snapshot=grounded_snapshot,
                community_run=community_run,
            )
        except (
            IngestionCancelled,
            JobCancellationRequested,
            CommunityDetectionCancelled,
        ) as exc:
            if state:
                self.db.cancel_strategy_graph_rebuild(
                    int(state["project_id"]), strategy_key,
                    int(state["generation"]),
                    f"{exc}；未完成图谱未发布",
                )
            for target_id in target_document_ids:
                self.db.update_document_strategy_build(
                    target_id, strategy_key, status="cancelled", latest_job_id=job_id,
                    error_message=str(exc), completed_at=datetime.now(timezone.utc),
                )
            cancelled_subtasks = self._fail_active_ingestion_subtask(
                job_id, str(exc), cancelled=True,
            )
            self.db.update_job(
                job_id, status="cancelled", progress=100,
                log=f"{exc}；未完成图谱未发布", error=str(exc),
                metadata={
                    "stage": "cancelled",
                    "cancel_acknowledged_at": datetime.now(timezone.utc).isoformat(),
                    **(
                        {"subtasks": cancelled_subtasks}
                        if cancelled_subtasks is not None else {}
                    ),
                },
            )
        except Exception as exc:
            if self._job_cancellation_requested(job_id):
                reason = str(
                    ((self.db.get_job(job_id) or {}).get("metadata") or {}).get(
                        "cancel_reason"
                    ) or "图谱构建已安全终止"
                )
                if state:
                    self.db.cancel_strategy_graph_rebuild(
                        int(state["project_id"]), strategy_key,
                        int(state["generation"]),
                        f"{reason}；未完成图谱未发布",
                    )
                for target_id in target_document_ids:
                    self.db.update_document_strategy_build(
                        target_id, strategy_key, status="cancelled",
                        latest_job_id=job_id, error_message=reason,
                        completed_at=datetime.now(timezone.utc),
                    )
                cancelled_subtasks = self._fail_active_ingestion_subtask(
                    job_id, reason, cancelled=True,
                )
                self.db.update_job(
                    job_id, status="cancelled", progress=100,
                    log=f"{reason}；未完成图谱未发布",
                    metadata={
                        "stage": "cancelled",
                        "cancel_acknowledged_at": datetime.now(
                            timezone.utc,
                        ).isoformat(),
                        **(
                            {"subtasks": cancelled_subtasks}
                            if cancelled_subtasks is not None else {}
                        ),
                    },
                )
                return
            LOGGER.exception("Knowledge Forge strategy ingestion failed: %s", document_id)
            error_message = str(exc)
            if "禁止回退旧流程" not in error_message:
                error_message = (
                    "Knowledge Forge 构建失败，禁止回退旧流程："
                    f"{error_message}"
                )
            if state:
                self.db.fail_strategy_graph_rebuild(
                    int(state["project_id"]), strategy_key, int(state["generation"]),
                    error_message,
                )
            for target_id in target_document_ids:
                self.db.update_document_strategy_build(
                    target_id, strategy_key, status="failed", latest_job_id=job_id,
                    error_message=error_message, completed_at=datetime.now(timezone.utc),
                )
            self._fail_active_ingestion_subtask(job_id, error_message)
            self.db.update_job(
                job_id, status="failed", progress=100,
                log=f"Knowledge Forge 构建失败：{error_message}", error=error_message,
                metadata={"stage": "failed", "strategy_key": strategy_key},
            )
        finally:
            self._release_strategy_graph_resources(mutation, lease)

    @staticmethod
    def _relation_coverage_failures(
        plan: Any,
        persisted_work: list[dict[str, Any]],
        decision_count: int,
    ) -> list[str]:
        failures: list[str] = []
        terminal_failed = [
            item for item in persisted_work if item.get("status") == "terminal_failed"
        ]
        if plan.overflow:
            failures.append("overflow")
        if plan.deferred:
            failures.append("deferred")
        if any(
            token in str(item.get("error_message") or "").lower()
            for item in terminal_failed
            for token in ("missing", "response", "decision")
        ):
            failures.append("missing_response")
        if terminal_failed:
            failures.append("terminal_failure")
        if any(item.get("status") == "quarantined" for item in persisted_work):
            failures.append("quarantined")
        if int(decision_count) != int(plan.admitted_count):
            failures.append("unclassified_candidate")
        return sorted(set(failures))

    def _run_versioned_relation_pipeline(
        self,
        ingestion_run_id: str,
        project: dict[str, Any],
        document_id: int,
        chunk: dict[str, Any],
        extraction: dict[str, Any],
        schema_record: dict[str, Any],
        *,
        raise_on_failure: bool = False,
    ) -> dict[str, Any]:
        """Run P1 relation treatment in isolation; never fail or mutate the control path."""

        schema_version = int(schema_record.get("version") or 1)
        treatment_hash = graph_policy_hash(
            RELATION_CANDIDATE_POLICY_SET,
            schema_version=schema_version,
            config_summary={
                "max_candidates": Config.relation_classifier_max_candidates,
                "output_reserve": Config.relation_classifier_output_reserve,
                "safety_margin": Config.relation_classifier_safety_margin,
                "decision_wire_format": "root-candidate-map-v9-singleton-id-repair",
            },
        )
        revision_id = str(chunk.get("revision_id") or "")
        if not revision_id:
            raise ValueError("关系运行必须绑定不可变 revision_id")
        context_budget = int(
            project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS
        )
        prepared = self.relation_pipeline.prepare_relation_batch(
            context_budget_tokens=context_budget,
            project_scope=f"{project.get('project_key') or project['id']}:{revision_id}",
            chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
            content=str(chunk.get("content") or ""),
            entities=list(extraction.get("entities") or []),
            logical_schema=schema_record.get("schema_json") or {},
        )
        admitted_candidates = [
            {**asdict(item), "chunk_id": int(chunk["id"])}
            for item in prepared.plan.candidates
        ]
        overflow_candidates = [
            {**dict(item.get("candidate") or {}), "chunk_id": int(chunk["id"])}
            for item in prepared.plan.overflow
            if item.get("candidate")
        ]
        candidates = [*admitted_candidates, *overflow_candidates]
        candidate_map = {item["candidate_id"]: item for item in candidates}
        input_fingerprint = self._relation_input_fingerprint(
            revision_id=revision_id,
            chunk_id=int(chunk["id"]),
            content=str(chunk.get("content") or ""),
            schema_version=schema_version,
            treatment_hash=treatment_hash,
            candidate_map=candidate_map,
        )
        run_id = self._relation_run_id(
            int(project["id"]),
            document_id,
            revision_id,
            int(chunk["id"]),
            treatment_hash,
            input_fingerprint,
        )
        started = False
        try:
            self.db.start_relation_pipeline_run({
                "id": run_id,
                "project_id": int(project["id"]),
                "document_id": document_id,
                "chunk_id": int(chunk["id"]),
                "control_policy_hash": GRAPH_POLICY_HASH,
                "treatment_policy_hash": treatment_hash,
                "policy_set": dict(RELATION_CANDIDATE_POLICY_SET),
                "input_fingerprint": input_fingerprint,
            })
            started = True
            work_items = []
            for batch in prepared.batches:
                fingerprint = hashlib.sha256(
                    f"{input_fingerprint}|{batch.batch_id}".encode("utf-8")
                ).hexdigest()
                work_items.append({
                    "id": "relwork_" + fingerprint,
                    "revision_id": revision_id,
                    "project_id": int(project["id"]),
                    "document_id": document_id,
                    "chunk_id": int(chunk["id"]),
                    "batch_id": batch.batch_id,
                    "input_fingerprint": fingerprint,
                    "candidate_ids": list(batch.candidate_ids),
                    "request_json": {
                        "prompt": batch.prompt,
                        "candidates": [candidate_map[value] for value in batch.candidate_ids],
                    },
                    "estimated_input_tokens": batch.estimated_input_tokens,
                    "max_output_tokens": batch.output_token_reserve,
                    "max_attempts": Config.relation_max_attempts,
                })
            quarantined = []
            for item in [*prepared.plan.deferred, *prepared.plan.overflow]:
                candidate_id = str(item["candidate_id"])
                fingerprint = hashlib.sha256(
                    f"{input_fingerprint}|quarantined|{candidate_id}".encode("utf-8")
                ).hexdigest()
                quarantined.append({
                    "id": "relwork_" + fingerprint,
                    "revision_id": revision_id,
                    "project_id": int(project["id"]),
                    "document_id": document_id,
                    "chunk_id": int(chunk["id"]),
                    "batch_id": f"quarantined:{candidate_id}",
                    "input_fingerprint": fingerprint,
                    "candidate_ids": [candidate_id],
                    "request_json": {
                        "candidates": [candidate_map[candidate_id]],
                        "reason": item.get("reason") or "deferred",
                    },
                    "estimated_input_tokens": 0,
                    "max_output_tokens": 0,
                    "status": "quarantined",
                    "max_attempts": Config.relation_max_attempts,
                    "error_message": item.get("reason") or "candidate deferred",
                })
            self.db.prepare_relation_pipeline_work(
                run_id, work_items, quarantined=quarantined,
            )

            batch_map = {item.batch_id: item for item in prepared.batches}
            candidate_objects = {
                item.candidate_id: item for item in prepared.plan.candidates
            }
            owner = f"{ingestion_run_id}:{threading.get_ident()}"
            while True:
                leased = self.db.lease_relation_pipeline_work(
                    run_id, owner, lease_seconds=Config.relation_work_lease_seconds,
                )
                if not leased:
                    break
                batch = batch_map[str(leased["batch_id"])]
                selected = [
                    candidate_objects[str(candidate_id)]
                    for candidate_id in list(leased.get("candidate_ids") or [])
                ]
                try:
                    decisions, raw = self.relation_pipeline.classify_prepared_batch(
                        classifier=self.ollama,
                        model=project.get("llm_model") or Config.default_llm_model,
                        context_budget_tokens=context_budget,
                        batch=batch,
                        candidates=selected,
                        content=str(chunk.get("content") or ""),
                        logical_schema=schema_record.get("schema_json") or {},
                    )
                    self.db.settle_relation_pipeline_work(
                        str(leased["id"]), str(leased["lease_token"]),
                        output={"decisions": decisions, "raw": raw},
                    )
                except Exception as exc:
                    self.db.settle_relation_pipeline_work(
                        str(leased["id"]), str(leased["lease_token"]),
                        error_message=str(exc)[:2000],
                        retry_delay_seconds=Config.relation_retry_delay_seconds,
                    )

            persisted_work = self.db.list_relation_pipeline_work(run_id)
            open_items = [
                item for item in persisted_work
                if item["status"] in {"pending", "leased", "retryable_failed"}
            ]
            if open_items:
                raise RuntimeError("关系工作队列尚未收敛")
            decisions: list[dict[str, Any]] = []
            output_tokens = 0
            for item in persisted_work:
                output = item.get("output_json") or {}
                if isinstance(output, str):
                    output = json.loads(output)
                for decision in output.get("decisions") or []:
                    decisions.append({
                        **decision,
                        "policy_version": RELATION_CANDIDATE_POLICY_SET[
                            "relation_classifier_policy_version"
                        ],
                        "model_name": project.get("llm_model") or Config.default_llm_model,
                        "config_hash": treatment_hash,
                        "accepted_attempt_id": item.get("accepted_attempt_id"),
                        "output_hash": item.get("accepted_output_hash") or "",
                    })
                output_tokens += len(json.dumps(output, ensure_ascii=False)) // 4
            failed_items = [
                item for item in persisted_work
                if item["status"] in {"terminal_failed", "quarantined"}
            ]
            terminal_failed = sum(
                1 for item in persisted_work if item["status"] == "terminal_failed"
            )
            quarantined_count = sum(
                1 for item in persisted_work if item["status"] == "quarantined"
            )
            missing_responses = sum(
                1 for item in persisted_work
                if item["status"] == "terminal_failed"
                and any(token in str(item.get("error_message") or "").lower()
                        for token in ("missing", "response", "decision"))
            )
            coverage_failures = self._relation_coverage_failures(
                prepared.plan, persisted_work, len(decisions),
            )
            funnel_report = relation_quality_funnel(
                run_id=run_id,
                plan=prepared.plan,
                persisted_work=persisted_work,
                decisions=decisions,
            )
            if not funnel_report["coverage_complete"]:
                coverage_failures = sorted({
                    *coverage_failures,
                    *(f"funnel:{failure}" for failure in funnel_report["failures"]),
                })
            coverage_complete = not coverage_failures
            final_status = "completed"
            if any(item["status"] == "terminal_failed" for item in failed_items):
                final_status = "terminal_failed"
            elif failed_items:
                final_status = "quarantined"
            metrics = {
                **prepared.metrics,
                "classified": len(decisions),
                "model_calls": sum(int(item.get("attempt") or 0) for item in persisted_work),
                "input_tokens": sum(
                    int(item.get("estimated_input_tokens") or 0)
                    * int(item.get("attempt") or 0)
                    for item in persisted_work
                ),
                "output_tokens": output_tokens,
                "latency_ms": sum(
                    float(item.get("accepted_latency_ms") or 0) for item in persisted_work
                ),
                "retried": sum(
                    max(0, int(item.get("attempt") or 0) - 1) for item in persisted_work
                ),
                "failed": len(failed_items),
                "terminal_failed": terminal_failed,
                "quarantined": quarantined_count,
                "missing_responses": missing_responses,
                "coverage_complete": coverage_complete,
                "coverage_failures": coverage_failures,
                "funnel": funnel_report,
                "published": 0,
                "verified": sum(
                    1 for item in decisions if item.get("status") == "classified"
                ),
            }
            finished = self.db.finish_relation_pipeline_run(
                run_id,
                candidates=candidates,
                decisions=decisions,
                metrics=metrics,
                status=final_status,
                error_message=(
                    "" if not failed_items
                    else "one or more relation work items failed closed"
                ),
            )
            return {
                "run_id": run_id,
                "status": final_status,
                "artifact_hash": str(finished.get("artifact_hash") or ""),
                "metrics": metrics,
                "candidates": candidates,
                "decisions": decisions,
                "coverage_complete": coverage_complete,
                "coverage_failures": coverage_failures,
            }
        except Exception as exc:
            LOGGER.warning("Isolated relation pipeline failed without publication: %s", exc)
            if started:
                try:
                    self.db.finish_relation_pipeline_run(
                        run_id, candidates=[], decisions=[],
                        metrics={"relation_exception": type(exc).__name__,
                                 "coverage_complete": False,
                                 "coverage_failures": ["terminal_failure"]},
                        status="failed", error_message=str(exc)[:2000],
                    )
                except Exception:
                    LOGGER.exception("Unable to persist isolated relation pipeline failure")
            if raise_on_failure:
                raise
            return {
                "run_id": run_id,
                "status": "failed",
                "artifact_hash": "",
                "metrics": {"relation_exception": type(exc).__name__,
                            "coverage_complete": False,
                            "coverage_failures": ["terminal_failure"]},
                "candidates": [],
                "decisions": [],
                "coverage_complete": False,
                "coverage_failures": ["terminal_failure"],
            }

    def _rebuild_project_graph(
        self,
        project_id: int,
        *,
        include_document_id: int | None = None,
        target_generation: int | None = None,
        checkpoint: Callable[[], None] | None = None,
        publication_job_id: str = "",
    ) -> dict[str, int]:
        """Purge and deterministically rebuild one project graph, then verify counts."""

        with self._graph_lock(project_id):
            return self._rebuild_project_graph_locked(
                project_id,
                include_document_id=include_document_id,
                target_generation=target_generation,
                checkpoint=checkpoint,
                publication_job_id=publication_job_id,
            )

    def _rebuild_at_quiescence(
        self,
        project_id: int,
        job_id: str,
        target_generation: int,
    ) -> dict[str, int] | None:
        """Coalesce a queued ingestion burst into its final generation rebuild."""

        if (
            hasattr(self.db, "has_pending_ingestions")
            and self.db.has_pending_ingestions(project_id, exclude_job_id=job_id)
        ):
            return None
        if not hasattr(self.db, "get_job"):
            return self._rebuild_project_graph(
                project_id, target_generation=target_generation,
            )
        return self._rebuild_project_graph(
            project_id,
            target_generation=target_generation,
            checkpoint=lambda: self._assert_job_active(job_id),
            publication_job_id=job_id,
        )

    def _rebuild_project_graph_locked(
        self,
        project_id: int,
        *,
        include_document_id: int | None = None,
        target_generation: int | None = None,
        checkpoint: Callable[[], None] | None = None,
        publication_job_id: str = "",
    ) -> dict[str, int]:
        """Execute a rebuild while the process-local project graph lock is held."""

        def check() -> None:
            if checkpoint is not None:
                checkpoint()

        checkpoint_kwargs = (
            {"checkpoint": checkpoint} if checkpoint is not None else {}
        )
        # Real HugeGraphStore supports per-record cooperative checkpoints.
        # Narrow graph lifecycle adapters used by policy tests deliberately do
        # not; outer stage checkpoints still exercise their intended contract.
        if not isinstance(getattr(self, "graph", None), HugeGraphStore):
            checkpoint_kwargs = {}

        check()
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        fixed_generation = int(
            target_generation
            if target_generation is not None
            else project.get("graph_generation") or 0
        )
        if int(project.get("graph_generation") or 0) != fixed_generation:
            raise RuntimeError("图谱重建目标代次已过期")
        self._retry_entity_indexing_failures(project, checkpoint=checkpoint)
        self._retry_relation_indexing_failures(project, checkpoint=checkpoint)
        check()
        blocking = self.db.list_blocking_indexing_failures(project_id, fixed_generation)
        if blocking:
            raise RuntimeError(
                f"图谱代次 {fixed_generation} 仍有 {len(blocking)} 个派生索引待修复"
            )
        namespace = str(project.get("graph_namespace_id") or "")
        policy_hash = project_graph_policy_hash(project)
        # Bind a cancellable public rebuild lease to its durable job so restart
        # recovery can release that exact owner immediately.  Internal callers
        # without a job retain a unique attempt owner.
        owner = (
            f"graph-rebuild:{publication_job_id}"
            if publication_job_id
            else f"graph-rebuild:{uuid.uuid4().hex}"
        )
        lease = self.db.acquire_graph_lease(
            project_id, namespace, fixed_generation, owner,
            ttl_seconds=Config.graph_lease_seconds,
        )
        mutation = GraphMutationContext(
            namespace_id=namespace, project_id=project_id, generation=fixed_generation,
            policy_hash=policy_hash, lease_token=str(lease["lease_token"]),
        )

        def heartbeat() -> None:
            check()
            if not self.db.heartbeat_graph_lease(
                str(lease["id"]), mutation.lease_token,
                ttl_seconds=Config.graph_lease_seconds,
            ):
                raise RuntimeError("graph lease 已失效")

        try:
            documents = [
                item for item in self.db.list_documents(project_id)
                if item["status"] == "indexed" or int(item["id"]) == include_document_id
            ]
            document_ids = [int(item["id"]) for item in documents]
            expected_manifest = self.graph.expected_manifest(
                project, documents, mutation=mutation, **checkpoint_kwargs,
            )
            self.db.persist_graph_manifest(
                project_id, namespace, fixed_generation, "expected",
                mutation.lease_token, expected_manifest,
            )
            heartbeat()
            self.graph.ensure_schema()
            check()
            self.graph.delete_project(
                project, mutation=mutation, **checkpoint_kwargs,
            )
            written_relations = 0
            for document in documents:
                heartbeat()
                chunks = self.db.list_document_chunks(int(document["id"]))
                self.graph.write_document_snapshot(
                    project, document, chunks, mutation=mutation,
                    **checkpoint_kwargs,
                )
                report = self.graph.sync_document_assertions(
                    project, document, mutation=mutation,
                    **checkpoint_kwargs,
                )
                written_relations += int(report.get("relations") or 0)
                heartbeat()
            if documents:
                self.graph.prune_project_entities(
                    project, mutation=mutation, **checkpoint_kwargs,
                )
            check()
            expected_relations = self.db.count_publishable_relations(project_id, document_ids)
            graph_stats = self.graph.statistics(project, **checkpoint_kwargs)
            graph_relations = int(graph_stats.get("relations") or 0)
            if written_relations != expected_relations or graph_relations != expected_relations:
                raise RuntimeError(
                    "图谱关系一致性检查失败："
                    f"PostgreSQL={expected_relations}, written={written_relations}, "
                    f"HugeGraph={graph_relations}"
                )
            writer_manifest = self.graph.writer_manifest(project, mutation=mutation)
            actual_manifest = self.graph.actual_manifest(
                project, mutation=mutation, **checkpoint_kwargs,
            )
            comparison = compare_graph_manifests(
                expected_manifest, writer_manifest, actual_manifest,
            )
            self.db.persist_graph_manifest(
                project_id, namespace, fixed_generation, "writer",
                mutation.lease_token, writer_manifest,
            )
            self.db.persist_graph_manifest(
                project_id, namespace, fixed_generation, "actual",
                mutation.lease_token, actual_manifest,
            )
            if not comparison["complete"]:
                raise RuntimeError(
                    "图谱内容 manifest 不一致：" + ",".join(comparison["reasons"])
                )
            heartbeat()
            blocking = self.db.list_blocking_indexing_failures(project_id, fixed_generation)
            if blocking:
                raise RuntimeError(
                    f"图谱代次 {fixed_generation} 仍有 {len(blocking)} 个派生索引待修复"
                )
            completed = self.db.complete_graph_rebuild(
                project_id, fixed_generation, policy_hash=policy_hash,
                namespace_id=namespace, lease_token=mutation.lease_token,
                manifest_digest=expected_manifest["content_digest"], empty=not documents,
                stats={
                    "documents": int(graph_stats.get("documents") or 0),
                    "chunks": int(graph_stats.get("chunks") or 0),
                    "entities": int(graph_stats.get("entities") or 0),
                    "relations": graph_relations,
                },
                job_id=publication_job_id,
            )
            if not completed:
                raise RuntimeError("图谱重建完成时 generation、lease 或 manifest 已过期")
            return {
                "documents": len(documents),
                "entities": int(graph_stats.get("entities") or 0),
                "relations": graph_relations,
            }
        finally:
            discard_manifest = getattr(
                self.graph, "discard_writer_manifest", None,
            )
            if discard_manifest is not None:
                discard_manifest(mutation)
            self.db.release_graph_lease(str(lease["id"]), mutation.lease_token)

    def _retry_entity_indexing_failures(
        self,
        project: dict[str, Any],
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> int:
        """Retry assignment-derived entity vectors before graph readiness checks."""

        project_id = int(project["id"])
        model = str(project.get("embedding_model") or Config.default_embedding_model)
        failures = self.db.list_pending_indexing_failures(project_id, "entity")
        completed = 0
        for failure in failures:
            if checkpoint is not None:
                checkpoint()
            if str(failure.get("object_type") or "") != "entity":
                continue
            if str(failure.get("model_name") or "") != model:
                continue
            object_key = str(failure.get("object_key") or "")
            entities = self.db.get_entities_by_ids(
                project_id, [int(object_key)] if object_key.isdigit() else [],
            )
            if not entities:
                self.db.complete_indexing_failure(
                    project_id, "entity", object_key, model,
                )
                completed += 1
                continue
            payload = failure.get("payload") if isinstance(failure.get("payload"), dict) else {}
            try:
                self.vectors.upsert_entity_embedding(project_id, entities[0], model)
                self.db.complete_indexing_failure(
                    project_id, "entity", object_key, model,
                )
                completed += 1
            except Exception as exc:
                self.db.record_indexing_failure({
                    "project_id": project_id,
                    "object_type": "entity",
                    "object_key": object_key,
                    "model_name": model,
                    "last_error": str(exc),
                    "payload": payload,
                })
                LOGGER.warning("Entity embedding retry deferred for %s: %s", object_key, exc)
        return completed

    def _retry_relation_indexing_failures(
        self,
        project: dict[str, Any],
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> int:
        """Retry derived relation vectors without changing assertion publication."""

        project_id = int(project["id"])
        model = str(project.get("embedding_model") or Config.default_embedding_model)
        failures = self.db.list_pending_indexing_failures(project_id, "relation")
        completed = 0
        for failure in failures:
            if checkpoint is not None:
                checkpoint()
            if str(failure.get("model_name") or "") != model:
                continue
            object_key = str(failure.get("object_key") or "")
            relation_rows = self.db.get_relations_by_ids(
                project_id, [int(object_key)] if object_key.isdigit() else [],
            )
            if not relation_rows:
                # The assertion was removed or is no longer publishable. There is
                # no derived index left to repair, so retire the outbox record.
                self.db.complete_indexing_failure(
                    project_id, "relation", object_key, model,
                )
                completed += 1
                continue
            relation = relation_rows[0]
            payload = failure.get("payload") if isinstance(failure.get("payload"), dict) else {}
            relation_for_index = {
                **relation,
                "relation_key": payload.get("relation_key") or f"relation:{object_key}",
            }
            try:
                self.vectors.upsert_relation_embedding(
                    project_id,
                    int(failure.get("document_id") or 0),
                    int(failure.get("chunk_id") or 0),
                    relation_for_index,
                    str(relation.get("source_name") or payload.get("source_name") or ""),
                    str(relation.get("target_name") or payload.get("target_name") or ""),
                    model,
                )
                self.db.complete_indexing_failure(
                    project_id, "relation", object_key, model,
                )
                completed += 1
            except Exception as exc:
                self.db.record_indexing_failure({
                    "project_id": project_id,
                    "document_id": failure.get("document_id"),
                    "revision_id": failure.get("revision_id"),
                    "chunk_id": failure.get("chunk_id"),
                    "object_type": "relation",
                    "object_key": object_key,
                    "model_name": model,
                    "last_error": str(exc),
                    "payload": payload,
                })
                LOGGER.warning("Relation embedding retry deferred for %s: %s", object_key, exc)
        return completed

    @recorded_job("graph_rebuild")
    def _sync_project_graph(self, job_id: str, project_id: int) -> None:
        graph_mutation_started = False
        try:
            self._assert_job_active(job_id)
            job = self.db.begin_graph_mutation(job_id)
            graph_mutation_started = True
            target_generation = int(
                job.get("target_generation")
                if job.get("target_generation") is not None
                else (self.db.get_project(project_id) or {}).get("graph_generation") or 0
            )
            self._job(job_id, progress=5, log="开始按当前验证策略重建项目知识图谱")
            report = self._rebuild_project_graph(
                project_id,
                target_generation=target_generation,
                checkpoint=lambda: self._assert_job_active(job_id),
                publication_job_id=job_id,
            )
            self._job(
                job_id, "completed", 100,
                f"项目图谱同步完成，共处理 {report['documents']} 个文档、{report['relations']} 条关系",
                graph_ready=True,
            )
        except (IngestionCancelled, JobCancellationRequested) as exc:
            LOGGER.info("Project graph sync cancelled: %s: %s", project_id, exc)
            if graph_mutation_started:
                try:
                    self.db.set_graph_state(
                        project_id,
                        GRAPH_STALE,
                        error="图谱构建已终止，未完成结果未发布",
                    )
                except LookupError:
                    pass
            self.db.update_job(
                job_id,
                status="cancelled",
                progress=100,
                log=f"{exc}；未完成图谱未发布",
                metadata={
                    "stage": "cancelled",
                    "cancel_acknowledged_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as exc:
            if self._job_cancellation_requested(job_id):
                LOGGER.info(
                    "Project graph sync stopped after cancellation: %s: %s",
                    project_id, exc,
                )
                if graph_mutation_started:
                    try:
                        self.db.set_graph_state(
                            project_id,
                            GRAPH_STALE,
                            error="图谱构建已终止，未完成结果未发布",
                        )
                    except LookupError:
                        pass
                self.db.update_job(
                    job_id, status="cancelled", progress=100,
                    log="图谱构建已安全终止；未完成图谱未发布",
                    metadata={
                        "stage": "cancelled",
                        "cancel_acknowledged_at": datetime.now(
                            timezone.utc,
                        ).isoformat(),
                    },
                )
                return
            LOGGER.exception("Project graph sync failed: %s", project_id)
            stale_generation = "过期" in str(exc)
            try:
                self.db.set_graph_state(
                    project_id,
                    GRAPH_STALE if stale_generation else GRAPH_FAILED,
                    error=str(exc),
                )
            except LookupError:
                pass
            self.db.update_job(job_id, status="failed", progress=100, log=str(exc), error=str(exc))
            if stale_generation:
                try:
                    next_job = self.db.create_graph_rebuild_job(project_id)
                    if not next_job.get("_coalesced"):
                        self.executor.submit(
                            self._sync_project_graph, str(next_job["id"]), project_id,
                        )
                except (LookupError, ValueError):
                    LOGGER.info("A newer project mutation already owns graph recovery: %s", project_id)

    def _reassess_project_schema(self, job_id: str, project_id: int) -> None:
        self._job(
            job_id,
            "running",
            3,
            "开始基于当前项目的全部有效文档重新评估 Schema",
            stage="schema_reassessment",
        )
        try:
            project = self.db.get_project(project_id)
            if not project:
                raise LookupError("项目不存在")
            documents = [
                item for item in self.db.list_documents(project_id)
                if item.get("status") == "indexed"
            ]

            def report_progress(index: int, total: int, report: dict[str, Any]) -> None:
                self._job(
                    job_id,
                    progress=5 + int(index / max(1, total) * 90),
                    log=(
                        f"Schema 评估 {index}/{total} [{report.get('document_name') or report.get('document_id')}]："
                        f"新增 {len(report.get('accepted_entity_types') or [])} 类实体、"
                        f"{len(report.get('accepted_relation_types') or [])} 类关系"
                    ),
                    stage="schema_reassessment",
                )

            def ensure_current_snapshot() -> None:
                job = self.db.get_job(job_id)
                if not job or job.get("status") == "cancelled":
                    raise IngestionCancelled("项目文档已发生变化，Schema 重新评估任务已取消")

            schema = self.schemas.reassess_project(
                project, documents, report_progress, ensure_current_snapshot
            )
            logical = schema.get("schema_json") or {}
            self._job(
                job_id,
                "completed",
                100,
                (
                    f"Schema 重新评估完成：v{schema.get('version') or 1}，"
                    f"{len(logical.get('entity_types') or [])} 类实体，"
                    f"{len(logical.get('relation_types') or [])} 类关系"
                ),
                stage="completed",
                schema_version=int(schema.get("version") or 1),
            )
        except Exception as exc:
            job = self.db.get_job(job_id)
            if job and job.get("status") == "cancelled":
                LOGGER.info("Project schema reassessment cancelled: %s", project_id)
                return
            LOGGER.exception("Project schema reassessment failed: %s", project_id)
            self.db.update_job(
                job_id,
                status="failed",
                progress=100,
                log=f"Schema 重新评估失败：{exc}",
                error=str(exc),
                metadata={"stage": "failed"},
            )

    def delete_document(self, document_id: int) -> str:
        document = self.db.get_document(document_id)
        if not document: raise LookupError("文档不存在或已删除")
        strategy_key = self._assert_project_strategy(int(document["project_id"]))
        self.db.cancel_document_jobs(document_id)
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
            cancel_project_jobs = getattr(self.db, "cancel_project_jobs", None)
            if callable(cancel_project_jobs):
                cancel_project_jobs(
                    int(document["project_id"]),
                    ["schema_reassessment", "community_report_build"],
                    "项目文档已删除，依赖旧文档版本的后台任务已停止",
                )
        # Make the user-visible deletion immediate. The final cleanup runs after any
        # in-flight resolver call releases the per-document lock and reapplies the
        # soft-delete cascade to catch writes that were already in progress.
        self.db.soft_delete_document(document_id)
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
            self._mark_graphrag_native_source_changed(
                int(document["project_id"]), "文档删除，native source snapshot 已失效",
            )
        else:
            self.db.set_graph_state(
                int(document["project_id"]), GRAPH_STALE,
                error="项目文档已删除，图谱正在重新收敛",
            )
        document_name = str(document.get("original_filename") or f"文档 {document_id}")
        job = self.db.create_job(
            "document_cleanup", int(document["project_id"]), document_id,
            metadata={"strategy_key": strategy_key},
            strategy_key=strategy_key,
        )
        self.db.update_job(
            str(job["id"]),
            status="queued",
            progress=5,
            log="文档已软删除，正在等待活动解析步骤安全退出",
            metadata={"stage": "waiting_for_worker", "document_name": document_name},
        )
        self.cleanup_executor.submit(self._cleanup_deleted_document, str(job["id"]), document)
        return str(job["id"])

    def _cleanup_deleted_document(self, job_id: str, document: dict[str, Any]) -> None:
        document_id = int(document["id"])
        project_id = int(document["project_id"])
        strategy_key = self._assert_project_strategy(project_id)
        try:
            self._job(
                job_id,
                "running",
                10,
                "等待正在执行的解析步骤释放文档资源",
                stage="waiting_for_worker",
                document_name=str(document.get("original_filename") or f"文档 {document_id}"),
            )
            with self._document_lock(document_id):
                self._job(job_id, "running", 30, "开始清理文档派生数据", stage="cleanup")
                # Repeat the cascade after the worker exits so no late mention,
                # relation or vector contribution can survive deletion.
                self.db.soft_delete_document(document_id)
                trash_dir = Config.trash_root / Config.project_storage_key(project_id, document["project_name"])
                trash_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime('%Y%m%d%H%M%S')
                local_path = Path(document["local_path"])
                if local_path.exists():
                    shutil.move(str(local_path), trash_dir / f"{stamp}_{local_path.name}")
                metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
                process_path = Path(str(metadata.get("chunk_file") or "")) if metadata.get("chunk_file") else None
                if process_path and process_path.exists():
                    shutil.move(str(process_path), trash_dir / f"{stamp}_{process_path.name}")
                media_path = Path(str(metadata.get("media_dir") or "")) if metadata.get("media_dir") else None
                if media_path and media_path.exists():
                    shutil.move(str(media_path), trash_dir / f"{stamp}_{media_path.name}_media")
                self.db.soft_delete_document(document_id)
                self._job(job_id, "running", 80, "源文件和解析过程文件已移入回收目录", stage="cleanup")
                if (
                    strategy_key == KNOWLEDGE_FORGE_STRATEGY
                    and self.db.count_documents(project_id) == 0
                ):
                    self.schemas.activate_base(project_id)

            if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
                report = self._rebuild_project_graph(project_id)
                self._job(
                    job_id, "running", 92,
                    f"HugeGraph 已按剩余文档重建：{report['relations']} 条关系",
                    stage="cleanup",
                )
            self._job(job_id, "completed", 100, "文档及其派生数据清理完成", stage="completed")
        except Exception as exc:
            LOGGER.exception("Document cleanup failed: %s", document_id)
            if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
                try:
                    self.db.set_graph_state(project_id, GRAPH_FAILED, error=str(exc))
                except LookupError:
                    pass
            self.db.update_job(
                job_id,
                status="failed",
                progress=100,
                log=f"文档已软删除，但后台清理失败：{exc}",
                error=str(exc),
                metadata={"stage": "failed", "document_name": document.get("original_filename", "")},
            )

    def search(
        self, project_id: int, query: str, top_k: int | None = None, *,
        allow_report_scheduling: bool = True,
    ) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if not project: raise LookupError("项目不存在")
        query = (query or "").strip()
        if not query: raise ValueError("请输入检索内容")
        reference = self._current_knowledge_forge_snapshot_reference(project_id)
        namespace_id = str(reference["entity_namespace_id"])
        relation_run_id = str(reference["relation_run_id"])
        namespace = reference["namespace"]
        report_state: dict[str, Any] = {"run": None, "reports": []}
        policy_summary = reference.get("policy_summary") or {}
        graph_knowledge_language = normalize_knowledge_language(
            policy_summary.get("knowledge_language") or CHINESE,
        )
        community_run_id = str(policy_summary.get("community_run_id") or "")
        community_input_hash = str(
            policy_summary.get("community_input_hash") or ""
        )
        if (
            Config.knowledge_forge_community_reports_enabled
            and community_run_id and community_input_hash
        ):
            if allow_report_scheduling:
                self._schedule_knowledge_forge_community_reports(
                    project_id, reference=reference,
                )
            report_state = self.db.get_knowledge_forge_community_report_state(
                project_id=project_id,
                community_run_id=community_run_id,
                graph_generation=int(
                    reference["graph"].get("ready_generation") or 0
                ),
                entity_namespace_id=namespace_id,
                community_input_hash=community_input_hash,
            )
        limit = min(top_k or 20, 50)
        dense_weight = max(
            0.0, min(1.0, float(Config.retrieval_dense_weight)),
        )
        chunks = self.vectors.search(project, query, top_k)
        chunk_scores = {
            int(item["chunk_id"]): float(item.get("score") or 0.0)
            for item in chunks if item.get("chunk_id") is not None
        }

        embedding_model = str(
            namespace.get("embedding_model")
            or project.get("embedding_model")
            or Config.default_embedding_model
        )
        query_vector = self.ollama.embed([query], embedding_model)[0]
        lexical_entities = self.db.search_knowledge_forge_entities(
            namespace_id, query, limit,
        )
        vector_entities = self.db.search_grounded_entity_vector_candidates(
            namespace_id, embedding_model, query_vector, limit,
        )
        entities: dict[str, dict[str, Any]] = {}
        for item in [*lexical_entities, *vector_entities]:
            key = str(item.get("canonical_key") or "")
            if not key:
                continue
            metadata = (
                item.get("metadata")
                if isinstance(item.get("metadata"), dict) else {}
            )
            description = str(metadata.get("description") or "")
            content = " ".join([
                str(item.get("name") or ""),
                str(item.get("primary_type") or ""),
                " ".join(str(value) for value in (item.get("aliases") or [])),
                description,
            ])
            lexical_score = max(
                float(item.get("lexical_score") or 0.0),
                lexical_similarity(query, content),
            )
            vector_score = float(item.get("vector_score") or 0.0)
            existing = entities.setdefault(key, {
                "id": key,
                "canonical_key": key,
                "name": str(item.get("name") or key),
                "entity_type": str(item.get("primary_type") or "GenericEntity"),
                "entity_type_label": entity_type_label(
                    item.get("primary_type") or "GenericEntity",
                    graph_knowledge_language,
                ),
                "description": description,
                "aliases": list(item.get("aliases") or []),
                "source_count": int(item.get("source_count") or 0),
                "lexical_score": 0.0,
                "dense_score": 0.0,
                "score": 0.0,
            })
            existing["lexical_score"] = max(
                float(existing["lexical_score"]), lexical_score,
            )
            existing["dense_score"] = max(
                float(existing["dense_score"]), vector_score,
            )
            existing["score"] = (
                dense_weight * float(existing["dense_score"])
                + (1 - dense_weight)
                * float(existing["lexical_score"])
            )

        ranked_entities = sorted(
            entities.values(),
            key=lambda item: (
                float(item.get("score") or 0.0),
                int(item.get("source_count") or 0),
                str(item.get("name") or ""),
            ),
            reverse=True,
        )
        seed_entities = select_graph_seed_entities(
            query, ranked_entities, limit=min(10, limit),
        )
        seed_scores = {
            str(item["canonical_key"]): max(
                float(item.get("score") or 0.0),
                1.0 if item.get("query_surface_match") else 0.0,
            )
            for item in seed_entities if item.get("canonical_key")
        }
        current_report_run = report_state.get("run") or {}
        if allow_report_scheduling and current_report_run and seed_scores:
            self._activate_deferred_knowledge_forge_reports(
                project_id=project_id,
                run_id=str(current_report_run["run_id"]),
                entity_keys=list(seed_scores),
                limit=Config.knowledge_forge_community_report_search_activation_limit,
            )
        candidate_limit = min(100, max(40, limit * 5, len(chunk_scores) * 4))
        first_relation_rows = self.db.search_knowledge_forge_relations(
            namespace_id,
            relation_run_id,
            query,
            chunk_ids=list(chunk_scores),
            canonical_keys=list(seed_scores),
            limit=candidate_limit,
        )
        neighbor_keys = graph_neighbor_keys(
            first_relation_rows, seed_scores, limit=24,
        )
        second_relation_rows = (
            self.db.search_knowledge_forge_relations(
                namespace_id,
                relation_run_id,
                query,
                chunk_ids=[],
                canonical_keys=neighbor_keys,
                limit=candidate_limit,
            )
            if neighbor_keys else []
        )
        entity_hops = build_graph_entity_hops(
            seed_scores, first_relation_rows, second_relation_rows,
        )
        relations: list[dict[str, Any]] = []
        seen_relation_ids: set[str] = set()
        for item in [*first_relation_rows, *second_relation_rows]:
            relation_id = str(item.get("id") or "")
            if not relation_id or relation_id in seen_relation_ids:
                continue
            seen_relation_ids.add(relation_id)
            metadata = (
                item.get("metadata")
                if isinstance(item.get("metadata"), dict) else {}
            )
            content = " ".join([
                str(item.get("source_name") or ""),
                str(item.get("relation_type") or ""),
                str(metadata.get("display_relation_label") or ""),
                str(metadata.get("source_relation_expression") or ""),
                " ".join(
                    str(value)
                    for value in (
                        metadata.get("source_relation_expressions") or ()
                    )
                ),
                str(item.get("target_name") or ""),
                str(item.get("evidence_text") or ""),
            ])
            relation_surface_content = " ".join([
                str(item.get("source_name") or ""),
                str(item.get("relation_type") or ""),
                str(metadata.get("display_relation_label") or ""),
                str(metadata.get("source_relation_expression") or ""),
                " ".join(
                    str(value)
                    for value in (
                        metadata.get("source_relation_expressions") or ()
                    )
                ),
                str(item.get("target_name") or ""),
            ])
            lexical_score = max(
                float(item.get("lexical_score") or 0.0),
                lexical_similarity(query, content),
            )
            relation_surface_score = lexical_similarity(
                query, relation_surface_content,
            )
            evidence_score = float(
                chunk_scores.get(int(item.get("chunk_id") or 0), 0.0)
            )
            score = (
                dense_weight * evidence_score
                + (1 - dense_weight) * lexical_score
            )
            relations.append({
                **dict(item),
                "relation_type_label": relation_display_label(
                    item.get("relation_type"), metadata,
                    graph_knowledge_language,
                ),
                "description": str(item.get("evidence_text") or ""),
                "confidence": _optional_relation_confidence(metadata),
                "evidence_count": int(
                    item.get("evidence_count")
                    or _supported_evidence_count(
                        metadata, item.get("evidence_text"),
                    )
                ),
                "verification_status": "SUPPORTED",
                "dense_score": evidence_score,
                "lexical_score": lexical_score,
                "relation_surface_score": relation_surface_score,
                "_retrieval_surface_text": relation_surface_content,
                "_retrieval_predicate_text": relation_display_label(
                    item.get("relation_type"), metadata,
                    graph_knowledge_language,
                ),
                "score": score,
            })
        if relations:
            semantic_texts = list(dict.fromkeys(
                str(value).strip()
                for item in relations
                for value in (
                    item.get("_retrieval_surface_text"),
                    item.get("_retrieval_predicate_text"),
                )
                if str(value or "").strip()
            ))
            semantic_vectors = self.ollama.embed(
                semantic_texts, embedding_model,
            )
            semantic_by_text = dict(zip(
                semantic_texts, semantic_vectors, strict=True,
            ))
            for item in relations:
                surface_text = str(
                    item.pop("_retrieval_surface_text", "") or ""
                ).strip()
                predicate_text = str(
                    item.pop("_retrieval_predicate_text", "") or ""
                ).strip()
                surface_semantic = cosine_similarity(
                    query_vector, semantic_by_text.get(surface_text, []),
                )
                predicate_semantic = cosine_similarity(
                    query_vector, semantic_by_text.get(predicate_text, []),
                )
                item["relation_semantic_score"] = max(
                    0.0,
                    min(1.0, 0.45 * surface_semantic + 0.55 * predicate_semantic),
                )
        ranked_relations = rank_graph_relations(
            relations,
            seed_scores=seed_scores,
            entity_hops=entity_hops,
            limit=limit,
        )
        community_reports: list[dict[str, Any]] = []
        report_rows: list[dict[str, Any]] = []
        if current_report_run:
            report_rows = self.db.list_searchable_knowledge_forge_community_reports(
                str(current_report_run["run_id"]), embedding_model,
            )
            public_report_by_id = {
                str(item.get("report_id") or ""): item
                for item in self._public_community_reports_with_sources(
                    project_id, [dict(value) for value in report_rows],
                )
                if str(item.get("report_id") or "")
            }
            for item in report_rows:
                content = " ".join(filter(None, (
                    str(item.get("title") or ""),
                    str(item.get("summary") or ""),
                    str(item.get("content") or ""),
                )))
                lexical_score = lexical_similarity(query, content)
                dense_score = cosine_similarity(
                    query_vector, list(item.get("embedding") or []),
                )
                score = (
                    dense_weight * max(0.0, dense_score)
                    + (1 - dense_weight) * lexical_score
                )
                community_reports.append({
                    "report_id": str(item.get("report_id") or ""),
                    "community_key": str(item.get("community_key") or ""),
                    "title": public_report_by_id.get(
                        str(item.get("report_id") or ""), {},
                    ).get("title") or "",
                    "summary": public_report_by_id.get(
                        str(item.get("report_id") or ""), {},
                    ).get("summary") or "",
                    "findings": list(public_report_by_id.get(
                        str(item.get("report_id") or ""), {},
                    ).get("findings") or []),
                    "limitations": public_report_by_id.get(
                        str(item.get("report_id") or ""), {},
                    ).get("limitations") or "",
                    "cited_entity_keys": list(
                        item.get("cited_entity_keys") or []
                    ),
                    "cited_relation_keys": list(
                        item.get("cited_relation_keys") or []
                    ),
                    "sources": list(
                        public_report_by_id.get(
                            str(item.get("report_id") or ""), {},
                        ).get("sources") or []
                    ),
                    "lexical_score": lexical_score,
                    "dense_score": max(0.0, dense_score),
                    "score": score,
                })
            community_reports.sort(
                key=lambda item: (
                    float(item["score"]), float(item["lexical_score"]),
                    str(item["title"]),
                ),
                reverse=True,
            )
            community_reports = community_reports[
                :Config.knowledge_forge_community_report_search_top_k
            ]
        result_chunk_ids = {
            int(item.get("chunk_id") or 0)
            for item in [*chunks, *ranked_relations]
            if int(item.get("chunk_id") or 0) > 0
        }
        provenance_reader = getattr(self.db, "get_chunk_provenance", None)
        result_provenance = (
            provenance_reader(project_id, sorted(result_chunk_ids))
            if callable(provenance_reader) else {}
        )
        chunks = [
            {
                **item,
                "source_info": _chunk_source_payload(
                    result_provenance.get(int(item.get("chunk_id") or 0)),
                ),
            }
            for item in chunks
        ]
        ranked_relations = [
            {
                **item,
                "source_info": _chunk_source_payload(
                    result_provenance.get(int(item.get("chunk_id") or 0)),
                ),
            }
            for item in ranked_relations
        ]
        return {
            "query": query,
            "strategy": KNOWLEDGE_FORGE_STRATEGY,
            "graph_generation": int(
                reference["graph"].get("ready_generation") or 0
            ),
            "entity_namespace_id": namespace_id,
            "relation_run_id": relation_run_id,
            "chunks": chunks,
            "entities": ranked_entities[:limit],
            "relations": ranked_relations,
            "community_reports": community_reports,
            "community_report_retrieval": {
                "status": str(
                    current_report_run.get("status") or "not_started"
                ),
                "available_reports": len(report_rows),
                "selected_reports": len(community_reports),
                "fallback": "chunks_entities_relations",
            },
            "graph_retrieval": {
                "policy_version": GRAPH_RETRIEVAL_POLICY_VERSION,
                "seed_entity_count": len(seed_scores),
                "seed_entities": [
                    {
                        "canonical_key": str(
                            item.get("canonical_key") or ""
                        ),
                        "score": float(
                            seed_scores.get(
                                str(item.get("canonical_key") or ""), 0.0,
                            )
                        ),
                        "query_surface_match": bool(
                            item.get("query_surface_match")
                        ),
                    }
                    for item in seed_entities
                    if str(item.get("canonical_key") or "")
                ],
                "expanded_entity_count": max(0, len(entity_hops) - len(seed_scores)),
                "candidate_relation_count": len(relations),
                "maximum_hops": 2,
            },
        }

    # ------------------------------------------------------------------
    # Knowledge-agent v1 public service boundary.

    def agent_knowledge_bases(self) -> dict[str, Any]:
        """List explicit single-knowledge-base scopes and their real readiness."""

        items: list[dict[str, Any]] = []
        for project in self.db.list_projects():
            project_id = int(project["id"])
            selection = self.db.get_project_construction_strategy_selection(project_id) or {}
            if str(selection.get("association_status") or "") != "bound":
                continue
            strategy = str(selection.get("selected_strategy") or "")
            queryable = False
            status_key = "unavailable"
            status_label = "暂不可问答"
            details: dict[str, Any] = {}
            if strategy == KNOWLEDGE_FORGE_STRATEGY:
                try:
                    source = self.db.agent_project_source_snapshot(project_id)
                    queryable = bool(
                        int(source.get("chunk_count") or 0) > 0
                        and int(source.get("embedding_count") or 0) > 0
                    )
                    graph_current = False
                    try:
                        self._current_knowledge_forge_snapshot_reference(project_id)
                        graph_current = True
                    except GraphNotReadyError:
                        graph_current = False
                    graph = self.db.get_project_strategy_graph(
                        project_id, KNOWLEDGE_FORGE_STRATEGY,
                    ) or {}
                    if queryable and graph_current:
                        status_key, status_label = "ready_hybrid", "可问答 · 原文与图谱"
                    elif queryable and str(graph.get("status") or "") == GRAPH_STALE:
                        status_key, status_label = "ready_text_stale_graph", "图谱待更新 · 暂用原文"
                    elif queryable:
                        status_key, status_label = "ready_text", "可问答 · 仅原文"
                    else:
                        status_key, status_label = "needs_preparation", "需完成文档预处理"
                    details = {
                        "document_count": int(source.get("document_count") or 0),
                        "chunk_count": int(source.get("chunk_count") or 0),
                        "embedding_count": int(source.get("embedding_count") or 0),
                        "graph_current": graph_current,
                    }
                except Exception as exc:
                    details = {"error": str(exc)}
            elif strategy == GRAPHRAG_NATIVE_STRATEGY:
                try:
                    native = self.graphrag_native_runtime.status(project_id)
                    queryable = bool(native.get("fresh"))
                    if queryable:
                        status_key, status_label = "ready_native", "可问答 · GraphRAG"
                    elif native.get("run"):
                        status_key, status_label = "needs_native_rebuild", "需重新构建 GraphRAG 成果"
                    else:
                        status_key, status_label = "needs_native_build", "需构建 GraphRAG 成果"
                    details = {
                        "run_id": str((native.get("run") or {}).get("run_id") or ""),
                        "fresh": bool(native.get("fresh")),
                        "reason": str(
                            native.get("stale_reason")
                            or native.get("unavailable_reason") or ""
                        ),
                    }
                except Exception as exc:
                    details = {"error": str(exc)}
            model_context_tokens = int(
                project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS
            )
            details["model_context_tokens"] = model_context_tokens
            details["minimum_agent_context_tokens"] = Config.agent_min_context_tokens
            if queryable and model_context_tokens < Config.agent_min_context_tokens:
                queryable = False
                status_key = "context_budget_too_small"
                status_label = (
                    "需将知识库上下文上限提高到至少 "
                    f"{Config.agent_min_context_tokens:,} Token"
                )
            items.append({
                "id": project_id, "name": str(project.get("name") or ""),
                "description": str(project.get("description") or ""),
                "strategy": strategy, "queryable": queryable,
                "status": status_key, "status_label": status_label,
                "knowledge_language": str(project.get("knowledge_language") or "auto"),
                "updated_at": project.get("updated_at"), "details": details,
            })
        return {
            "items": items,
            "queryable_count": sum(1 for item in items if item["queryable"]),
            "total": len(items),
            "scope_policy": "single_project_explicit_v1",
        }

    def create_agent_thread(
        self, project_id: int, *, session_key: str,
    ) -> dict[str, Any]:
        if not Config.agent_enabled:
            raise RuntimeError("智能问答当前未启用")
        project = self.db.get_project(int(project_id))
        if not project:
            raise LookupError("知识库不存在")
        strategy = self._assert_project_strategy(int(project_id))
        readiness = next(
            (item for item in self.agent_knowledge_bases()["items"]
             if int(item["id"]) == int(project_id)),
            None,
        )
        if not readiness or not readiness["queryable"]:
            raise GraphNotReadyError(
                str((readiness or {}).get("status_label") or "知识库尚不可问答")
            )
        thread_id = "ath_" + uuid.uuid4().hex
        return self.db.create_agent_thread(
            thread_id=thread_id, project_id=int(project_id),
            strategy_key=strategy, session_key_hash=session_key_hash(session_key),
            metadata={"protocol": "knowledge-agent-v1"},
        )

    def list_agent_threads(
        self, project_id: int, *, session_key: str, limit: int = 100,
    ) -> dict[str, Any]:
        if not Config.agent_enabled:
            raise RuntimeError("智能问答当前未启用")
        project = self.db.get_project(int(project_id))
        if not project:
            raise LookupError("知识库不存在")
        rows = self.db.list_agent_threads(
            project_id=int(project_id),
            session_key_hash=session_key_hash(session_key),
            limit=limit,
        )
        return {
            "items": rows,
            "total": len(rows),
            "project_id": int(project_id),
        }

    def get_agent_thread(self, thread_id: str, *, session_key: str) -> dict[str, Any]:
        row = self.db.get_agent_thread_payload(
            str(thread_id), session_key_hash=session_key_hash(session_key),
        )
        if not row:
            raise PermissionError("问答会话不存在或无权访问")
        return row

    def archive_agent_thread(
        self, thread_id: str, *, session_key: str,
    ) -> dict[str, Any]:
        if not Config.agent_enabled:
            raise RuntimeError("智能问答当前未启用")
        archived = self.db.archive_agent_thread(
            str(thread_id), session_key_hash=session_key_hash(session_key),
        )
        try:
            self.agent.context.delete_thread(str(thread_id))
        except Exception as exc:
            LOGGER.warning(
                "Archived agent thread but checkpoint cleanup failed thread=%s error=%s",
                thread_id, type(exc).__name__,
            )
        return archived

    def submit_agent_message(
        self, thread_id: str, *, session_key: str, query: str,
        requested_mode: str = "auto",
    ) -> dict[str, Any]:
        if not Config.agent_enabled:
            raise RuntimeError("智能问答当前未启用")
        query = sanitize_unicode_data(str(query or "")).strip()
        if not query:
            raise ValueError("请输入问题")
        if len(query) > Config.agent_max_question_chars:
            raise ValueError(f"问题不能超过 {Config.agent_max_question_chars} 个字符")
        requested_mode = str(requested_mode or "auto").strip().lower()
        if requested_mode not in {"auto", "basic", "local", "global"}:
            raise ValueError("问答方式只能是自动、Basic、Local 或 Global")
        thread = self.db.get_agent_thread(
            str(thread_id), session_key_hash=session_key_hash(session_key),
        )
        if not thread:
            raise PermissionError("问答会话不存在或无权访问")
        if (
            str(thread.get("strategy_key") or "") == KNOWLEDGE_FORGE_STRATEGY
            and requested_mode != "auto"
        ):
            raise ValueError("Knowledge Forge 第一版使用固定混合检索，不接受 GraphRAG 问答方式")
        readiness = next(
            (item for item in self.agent_knowledge_bases()["items"]
             if int(item["id"]) == int(thread["project_id"])),
            None,
        )
        if not readiness or not readiness["queryable"]:
            raise GraphNotReadyError(
                str((readiness or {}).get("status_label") or "知识库尚不可问答")
            )
        run_id = "arun_" + uuid.uuid4().hex
        message_id = "amsg_" + uuid.uuid4().hex
        run = self.db.create_agent_run(
            run_id=run_id, message_id=message_id, thread_id=str(thread_id),
            session_key_hash=session_key_hash(session_key), query=query,
            requested_mode=requested_mode,
            budget={
                "candidate_limit": Config.agent_candidate_limit,
                "context_max_items": Config.agent_context_max_items,
                "context_budget_tokens": Config.agent_context_budget_tokens,
                "max_output_tokens": Config.agent_max_output_tokens,
                "max_question_chars": Config.agent_max_question_chars,
                # The primary pass is mandatory.  When the bounded
                # supplemental policy is enabled it may make at most one
                # additional adapter call and one additional reranker pass.
                # Freeze the real upper bound in the run ledger so audit data
                # never understates what the workflow is allowed to execute.
                "maximum_retrieval_calls": (
                    2 if Config.agent_supplemental_retrieval_enabled else 1
                ),
                "maximum_supplemental_attempts": (
                    1 if Config.agent_supplemental_retrieval_enabled else 0
                ),
                "maximum_context_resolution_calls": 1,
                "maximum_input_scope_calls": 1,
                "maximum_reranker_calls": (
                    2 if Config.agent_supplemental_retrieval_enabled else 1
                ),
                "maximum_answer_generations": 1,
                "maximum_citation_validation_passes": 1,
            },
        )
        self.agent_executor.submit(self.agent.run, run_id)
        return {
            "run_id": run_id, "message_id": message_id,
            "thread_id": str(thread_id), "status": str(run["status"]),
            "events_url": f"/api/agent/runs/{run_id}/events",
        }

    def get_agent_run(self, run_id: str, *, session_key: str) -> dict[str, Any]:
        row = self.db.get_agent_run_payload(
            str(run_id), session_key_hash=session_key_hash(session_key),
        )
        if not row:
            raise PermissionError("问答任务不存在或无权访问")
        row["execution_trace"] = build_public_execution_trace(row)
        return row

    def get_agent_run_trace(
        self, run_id: str, *, session_key: str,
    ) -> dict[str, Any]:
        row = self.db.get_agent_run_trace_payload(
            str(run_id), session_key_hash=session_key_hash(session_key),
        )
        if not row:
            raise PermissionError("问答任务不存在或无权访问")
        return build_public_execution_trace(row)

    def agent_events(
        self, run_id: str, *, session_key: str, after_id: int = 0,
    ) -> list[dict[str, Any]]:
        return self.db.list_agent_events(
            str(run_id), session_key_hash=session_key_hash(session_key),
            after_id=after_id,
        )

    def cancel_agent_run(self, run_id: str, *, session_key: str) -> dict[str, Any]:
        return self.db.request_agent_run_cancellation(
            str(run_id), session_key_hash=session_key_hash(session_key),
        )

    def get_agent_evidence(
        self, evidence_id: str, *, session_key: str,
    ) -> dict[str, Any]:
        row = self.db.get_agent_evidence(
            str(evidence_id), session_key_hash=session_key_hash(session_key),
        )
        if not row:
            raise PermissionError("问答证据不存在或无权访问")
        return row

    def record_agent_feedback(
        self, run_id: str, *, session_key: str, rating: int,
        comment: str = "", action: str = "answer_rating",
    ) -> dict[str, Any]:
        if int(rating) not in {-1, 0, 1}:
            raise ValueError("反馈评分只能是 -1、0 或 1")
        return self.db.record_agent_feedback(
            feedback_id="afb_" + uuid.uuid4().hex,
            run_id=str(run_id), session_key_hash=session_key_hash(session_key),
            rating=int(rating), comment=sanitize_unicode_data(str(comment or "")),
            action=str(action or "answer_rating"),
            metadata={"protocol": "knowledge-agent-v1"},
        )

    def health(self) -> dict[str, Any]:
        self.db.list_projects()
        ollama_models = len(self.ollama.models())
        graphs = self.graph.available_graphs()
        return {"status": "ok", "postgres": "reachable", "ollama_models": ollama_models,
                "hugegraph": Config.hugegraph_graph, "hugegraph_ready": Config.hugegraph_graph in graphs,
                "vector_backend": "pgvector-hnsw",
                "entity_reranker": self.resolver.reranker.diagnostics(),
                "knowledge_agent": {
                    "enabled": Config.agent_enabled,
                    "protocol": "knowledge-agent-v1",
                    "workers": Config.agent_max_workers,
                }}
