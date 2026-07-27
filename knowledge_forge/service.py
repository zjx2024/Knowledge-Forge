"""Project lifecycle, document ingestion, retrieval and graph orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from .chunking import ProcessingResult, build_processing_result, parse_document_structure, validate_processing_request
from .config import Config
from .construction_strategies import (
    ENHANCED_SHADOW_STRATEGY,
    STABLE_STRATEGY,
    normalize_strategy,
)
from .database import Database
from .entity_resolution import EntityResolver
from .entity_shadow import DatabaseShadowStore, ShadowEntityResolver, ShadowModelGateway
from .extraction_funnel import relation_shadow_funnel
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
from .graph_safety import GraphMutationContext
from .media import DocumentMediaService
from .ollama import OllamaClient
from .observability import recorded_job
from .policy import POLICY_AXES, P1_RELATION_SHADOW_POLICY_SET, P2_ENTITY_SHADOW_POLICY_SET
from .quality import canonical_digest
from .release_gate import ALLOWED_EVALUATOR_VERSIONS, evaluate_frozen_bundle
from .relation_pipeline import VersionedRelationPipeline
from .relation_shadow_runtime import (
    SHADOW_RELATION_V2_POLICY_SET,
    ShadowRelationV2Runtime,
)
from .schema_evolution import SchemaEvolutionService
from .span_entities import SpanFirstEntityExtractor
from .vector_store import VectorStore


LOGGER = logging.getLogger(__name__)


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
    DEFAULT_MODEL_MAX_TOKENS = 32768
    DEFAULT_CHUNK_MAX_TOKENS = 512

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
    def _shadow_relation_mention_id(
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
    def _relation_shadow_run_id(
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
    def _relation_shadow_input_fingerprint(
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
        self.relation_shadow = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
        self.graph = HugeGraphStore(self.db)
        self._document_locks_guard = threading.Lock()
        self._document_locks: dict[int, threading.RLock] = {}
        self._graph_locks_guard = threading.Lock()
        self._graph_locks: dict[int, threading.RLock] = {}
        # A single worker keeps multi-file ingestion deterministic and prevents local LLM contention.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-forge")
        # Cleanup must be able to wait for an active document without blocking queued ingestion work.
        self.cleanup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="knowledge-forge-cleanup")

    def bootstrap(self) -> None:
        for directory in (Config.docs_root, Config.chunks_root, Config.media_root, Config.trash_root):
            directory.mkdir(parents=True, exist_ok=True)
        self.db.bootstrap()
        for project in self.db.list_projects():
            self.schemas.ensure_base(int(project["id"]))
            self.db.ensure_project_strategy_graphs(int(project["id"]))
        try:
            self.graph.ensure_schema()
        except Exception as exc:
            LOGGER.warning("HugeGraph bootstrap deferred: %s", exc)

    def recover_interrupted_jobs(self) -> int:
        recovered = 0
        for job in self.db.list_recoverable_jobs():
            job_id = str(job["id"])
            self.db.update_job(
                job_id,
                status="queued",
                log="服务重新启动，任务将从头恢复",
                metadata={**(job.get("metadata") or {}), "recovered_after_restart": True},
            )
            if job["kind"] == "ingestion" and job.get("document_id"):
                strategy_key = normalize_strategy(job.get("strategy_key"))
                worker = (
                    self._ingest_enhanced_strategy
                    if strategy_key == ENHANCED_SHADOW_STRATEGY
                    else self._ingest_document
                )
                self.executor.submit(worker, job_id, int(job["document_id"]))
                recovered += 1
            elif job["kind"] == "content_preparation" and job.get("document_id"):
                self.executor.submit(
                    self._prepare_document_content, job_id, int(job["document_id"]),
                )
                recovered += 1
            elif job["kind"] == "graph_sync" and job.get("project_id"):
                self.executor.submit(self._sync_project_graph, job_id, int(job["project_id"]))
                recovered += 1
            elif job["kind"] == "schema_reassessment" and job.get("project_id"):
                self.executor.submit(self._reassess_project_schema, job_id, int(job["project_id"]))
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

    def create_project(self, name: str, description: str, llm_model: str = "", embedding_model: str = "") -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("项目名称不能为空")
        project = self.db.create_project(
            name, (description or "").strip(), llm_model or Config.default_llm_model,
            embedding_model or Config.default_embedding_model,
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
        self.schemas.ensure_base(int(project["id"]))
        self.db.ensure_project_strategy_graphs(int(project["id"]))
        return self.db.get_project(int(project["id"])) or project

    def update_project_settings(
        self,
        project_id: int,
        llm_model: str,
        embedding_model: str,
        model_max_tokens: int,
        chunk_max_tokens: int,
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
        source_name = Path(filename or "").name; suffix = Path(source_name).suffix.lower()
        requested_parser = requested_parser or "auto"
        requested_strategy = requested_strategy or "auto"
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
                self.db.ensure_document_strategy_builds(int(restored["id"]), reset=True)
                return restored
            document = self.db.create_document(values)
            self.db.ensure_document_strategy_builds(int(document["id"]), reset=True)
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
        document = self.db.get_document(document_id)
        if not document: raise LookupError("文档不存在")
        target_documents = [document]
        if strategy_key == ENHANCED_SHADOW_STRATEGY and target_document_ids:
            target_documents = []
            for target_id in dict.fromkeys(int(value) for value in target_document_ids):
                target = self.db.get_document(target_id)
                if not target or int(target["project_id"]) != int(document["project_id"]):
                    raise ValueError("增强实验包含不存在或不属于当前项目的文档")
                target_documents.append(target)
            if document_id not in {int(item["id"]) for item in target_documents}:
                target_documents.insert(0, document)
        if strategy_key == ENHANCED_SHADOW_STRATEGY:
            self.db.assert_shadow_relation_v2_ready()
        self.db.ensure_project_strategy_graphs(int(document["project_id"]))
        for target in target_documents:
            self.db.ensure_document_strategy_builds(int(target["id"]))
        build = self.db.get_document_strategy_build(document_id, strategy_key) or {}
        if build.get("status") in {"queued", "processing"}:
            raise ValueError("该文档正在使用当前方案构建")
        if strategy_key == STABLE_STRATEGY and document["status"] == "processing":
            raise ValueError("该文档正在处理")
        if strategy_key == ENHANCED_SHADOW_STRATEGY:
            active_enhanced = self.db.latest_active_job(
                int(document["project_id"]), "ingestion", ENHANCED_SHADOW_STRATEGY,
            )
            if active_enhanced:
                raise ValueError("当前项目的增强实验正在构建，请等待任务完成")
            active_stable = self.db.latest_active_job(
                int(document["project_id"]), "ingestion", STABLE_STRATEGY,
            )
            if active_stable:
                raise ValueError("当前项目的文档预处理或稳定方案正在构建，请等待任务完成")
        else:
            active_enhanced = self.db.latest_active_job(
                int(document["project_id"]), "ingestion", ENHANCED_SHADOW_STRATEGY,
            )
            if active_enhanced:
                raise ValueError("增强实验正在构建，完成后才能更新稳定基线")
            # Stable and enhanced_shadow are independent construction products.
            # A stable rebuild must not preemptively mark a ready enhanced
            # snapshot as stale; source/document mutations have their own
            # invalidation boundaries.
        batch_id = batch_id or uuid.uuid4().hex
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
            },
            strategy_key=strategy_key,
        )
        for target in target_documents:
            self.db.update_document_strategy_build(
                int(target["id"]), strategy_key, status="queued", latest_job_id=str(job["id"]),
                error_message="", started_at=None, completed_at=None,
            )
        project = self.db.get_project(int(document["project_id"]))
        if strategy_key == STABLE_STRATEGY and project and project.get("graph_status") == GRAPH_READY:
            self.db.set_graph_state(
                int(document["project_id"]), GRAPH_STALE,
                error="文档重新入库后图谱需要重建",
            )
        if strategy_key == ENHANCED_SHADOW_STRATEGY:
            self.db.mark_strategy_graph_stale(
                int(document["project_id"]), strategy_key,
                "增强实验事实将重新计算",
            )
            self.executor.submit(self._ingest_enhanced_strategy, str(job["id"]), document_id)
        else:
            self.executor.submit(self._ingest_document, str(job["id"]), document_id)
        return str(job["id"])

    def submit_project_ingestion(
        self, project_id: int, *, strategy_key: str = STABLE_STRATEGY,
    ) -> dict[str, Any]:
        """Build every pending document for one project and construction strategy."""

        strategy_key = normalize_strategy(strategy_key)
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        documents = self.db.list_documents(project_id, strategy_key)
        pending = [
            item for item in documents
            if str(item.get("strategy_status") or "uploaded")
            not in {"queued", "processing", "indexed", "deleted"}
        ]
        if not pending:
            raise ValueError("当前构建方案没有待处理文档")
        pending.sort(key=lambda item: int(item["id"]))
        batch_id = uuid.uuid4().hex
        if strategy_key == ENHANCED_SHADOW_STRATEGY:
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

    def submit_project_preparation(self, project_id: int) -> dict[str, Any]:
        """Prepare strategy-neutral document content without extracting graph facts."""

        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        active = self.db.latest_active_job(project_id)
        if active:
            raise ValueError("当前项目已有任务正在执行，请等待任务完成")
        pending: list[dict[str, Any]] = []
        for document in self.db.list_documents(project_id):
            manifest = self._document_revision_manifest(document, project)
            if not self._shared_content_is_current(document, manifest):
                pending.append(document)
                continue
            if not self.db.list_document_chunks(int(document["id"])):
                pending.append(document)
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
                },
            )
            job_id = str(job["id"])
            job_ids.append(job_id)
            self.executor.submit(
                self._prepare_document_content, job_id, int(document["id"]),
            )
        return {
            "job_id": job_ids[0], "job_ids": job_ids, "batch_id": batch_id,
            "document_count": len(pending),
        }

    def submit_graph_build(self, project_id: int) -> str:
        if not self.db.get_project(project_id): raise LookupError("项目不存在")
        job = self.db.create_graph_rebuild_job(project_id)
        if not job.get("_coalesced"):
            self.executor.submit(self._sync_project_graph, job["id"], project_id)
        return str(job["id"])

    def submit_schema_reassessment(self, project_id: int) -> str:
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
        active = self.db.latest_active_job(project_id)
        if active:
            raise ValueError("当前项目已有任务正在执行，请等待任务完成后再重新评估 Schema")
        job = self.db.create_job("schema_reassessment", project_id)
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

    def _run_shadow_relation_v2_namespace(
        self,
        *,
        namespace_id: str,
        project_id: int,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Invoke the only relation runtime permitted for enhanced_shadow."""

        return ShadowRelationV2Runtime(
            self.db,
            self.ollama,
            default_context_tokens=self.DEFAULT_MODEL_MAX_TOKENS,
        ).run_namespace(
            namespace_id=namespace_id,
            project_id=project_id,
            progress=progress,
        )

    def build_entity_shadow_namespace(
        self,
        project_id: int,
        snapshot_id: str = "",
        *,
        snapshot_manifest: dict[str, Any] | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Build an independent span-first P2 treatment from active raw chunks."""

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
        chunks.sort(key=lambda item: (
            int(item["document_id"]), int(item.get("chunk_index") or 0), int(item["id"]),
        ))
        policy_set = dict(P2_ENTITY_SHADOW_POLICY_SET)
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
        } for chunk in chunks]
        self.db.assert_shadow_relation_v2_ready()
        code_version = (
            "entity-shadow-p2-relation-v2-runtime-v2-predicate-metadata"
        )
        config_payload = {
            "policy_set": policy_set,
            "relation_policy_set": dict(SHADOW_RELATION_V2_POLICY_SET),
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "model_max_tokens": int(project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS),
            "lexical_cap": min(Config.entity_lexical_max_k, Config.entity_lexical_top_k),
            "vector_cap": min(Config.entity_vector_max_k, Config.entity_vector_top_k),
            "reranker_cap": Config.entity_reranker_max_k,
            "code_version": code_version,
        }
        config_hash = hashlib.sha256(
            json.dumps(config_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        input_payload = {
            "project_id": project_id,
            "snapshot_id": snapshot_id or "live",
            "documents": frozen_inputs,
            "chunks": frozen_chunks,
            "config_hash": config_hash,
        }
        input_fingerprint = hashlib.sha256(
            json.dumps(input_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        namespace_id = "entity_shadow_" + hashlib.sha256(
            f"{snapshot_id or 'live'}|{input_fingerprint}".encode("utf-8")
        ).hexdigest()
        namespace = self.db.create_entity_shadow_namespace(
            project_id, namespace_id, policy_set,
            [int(item["id"]) for item in documents], snapshot_id or None,
            input_fingerprint=input_fingerprint,
            model_name=llm_model,
            embedding_model=embedding_model,
            config_hash=config_hash,
            code_version=code_version,
        )
        if namespace.get("snapshot_complete") and namespace.get("status") == "ready":
            snapshot = self.db.load_entity_shadow_snapshot(namespace_id)
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
        self.db.seed_entity_shadow_work_items(namespace_id, work_items)
        summary = self.db.entity_shadow_work_summary(namespace_id)
        retried = 0
        if int(summary.get("failed") or 0):
            retried = self.db.requeue_terminal_entity_shadow_work(namespace_id)
            summary = self.db.entity_shadow_work_summary(namespace_id)
        if progress and retried:
            progress(
                int(summary["completed"]), int(summary["expected"]),
                f"重新排队 {retried} 个先前终止失败的知识块",
            )
        if progress:
            progress(
                int(summary["completed"]), int(summary["expected"]),
                "增强实体关系快照开始处理",
            )
        extractor = SpanFirstEntityExtractor(policy_set["entity_extraction_policy_version"])
        store = DatabaseShadowStore(self.db)
        model_gateway = ShadowModelGateway(
            self.ollama,
            llm_model=llm_model,
            context_budget_tokens=int(
                project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS
            ),
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

        resolver = ShadowEntityResolver(
            namespace=namespace_id,
            policy_version=policy_set["entity_resolution_policy_version"],
            store=store,
            models=model_gateway,
            reranker=rerank,
            must_links=self.db.list_entity_shadow_must_links(namespace_id),
            cannot_links=self.db.list_entity_shadow_cannot_links(namespace_id),
            lexical_cap=config_payload["lexical_cap"],
            vector_cap=config_payload["vector_cap"],
            reranker_cap=config_payload["reranker_cap"],
            auto_merge_threshold=Config.entity_merge_threshold,
            review_threshold=Config.entity_llm_review_threshold,
        )
        worker_id = f"entity-shadow:{uuid.uuid4().hex}"
        while True:
            work = self.db.claim_entity_shadow_work(namespace_id, worker_id)
            if not work:
                break
            chunk = chunk_by_id.get(int(work["chunk_id"]))
            summary = self.db.entity_shadow_work_summary(namespace_id)
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
            if not chunk:
                status = self.db.fail_entity_shadow_work(
                    str(work["id"]), str(work["lease_token"]), "active Chunk 不存在",
                )
                summary = self.db.entity_shadow_work_summary(namespace_id)
                if progress:
                    progress(
                        int(summary["completed"]), int(summary["expected"]),
                        f"知识块不存在，已标记为 {status}",
                    )
                continue
            try:
                self.db.reset_entity_shadow_chunk_attempt(
                    namespace_id, int(chunk["id"]),
                )
                extraction = self.schemas.extract_entities_only(project, active_schema, chunk)
                normalized = extractor.normalize(
                    project_scope=str(project.get("project_key") or project_id),
                    document_key=str(chunk.get("revision_id") or chunk["document_id"]),
                    chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
                    content=str(chunk.get("content") or ""),
                    raw_entities=extraction.get("entities") or [],
                    chunk_metadata=self._metadata(chunk.get("metadata")),
                )
                mention_rows = [{
                    **asdict(item),
                    "provenance": {
                        "input_fingerprint": work["input_fingerprint"],
                        "model_name": llm_model,
                        "config_hash": config_hash,
                            "code_version": code_version,
                    },
                } for item in normalized.mentions]
                assertion_rows = [asdict(item) for item in normalized.type_assertions]
                rejected_rows = [{
                    **item,
                    "policy_version": policy_set["entity_extraction_policy_version"],
                } for item in normalized.rejected]
                self.db.persist_entity_shadow_extraction(
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
                        **asdict(alias), "kind": "verified_alias",
                    })
                type_by_mention = {item.mention_id: item for item in normalized.type_assertions}
                raw_id_by_span = {
                    (
                        str(item.get("name") or ""),
                        int(item["mention_start"]) if item.get("mention_start") is not None else -1,
                        int(item["mention_end"]) if item.get("mention_end") is not None else -1,
                    ): str(item.get("mention_id") or item.get("id") or "")
                    for item in extraction.get("entities") or []
                }
                raw_to_mention: dict[str, str] = {}
                canonical_by_mention: dict[str, str] = {}
                prepared_mentions: list[tuple[Any, Any, dict[str, Any]]] = []
                for item in normalized.mentions:
                    type_assertion = type_by_mention[item.mention_id]
                    mention = {
                        "mention_id": item.mention_id,
                        "surface": item.surface,
                        "normalized_surface": item.normalized_surface,
                        "entity_type": type_assertion.entity_type,
                        "evidence": item.evidence,
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
                            alias["normalized_surface"]
                            for alias in alias_by_mention.get(item.mention_id, [])
                        ],
                    }
                    prepared_mentions.append((item, type_assertion, mention))

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
                        raise RuntimeError("entity shadow embedding 批次返回数量不匹配")
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
                        "name": str((existing or {}).get("name") or item.surface),
                        "normalized_name": str(
                            (existing or {}).get("normalized_name") or item.normalized_surface
                        ),
                        "primary_type": str(
                            (existing or {}).get("primary_type") or type_assertion.entity_type
                        ),
                        "aliases": sorted({
                            item.normalized_surface,
                            *((existing or {}).get("aliases") or []),
                            *mention["aliases"],
                        }),
                        "metadata": {
                            "type_assertion": asdict(type_assertion),
                            "identity_type_decoupled": True,
                        },
                    }
                    self.db.upsert_entity_shadow_canonical(
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
                    "relation_endpoints": 0,
                }
                output_hash = hashlib.sha256(
                    json.dumps(output, sort_keys=True).encode("utf-8")
                ).hexdigest()
                if not self.db.complete_entity_shadow_work(
                    str(work["id"]), str(work["lease_token"]), output, output_hash,
                ):
                    raise RuntimeError("entity shadow work lease 已失效")
                summary = self.db.entity_shadow_work_summary(namespace_id)
                if progress:
                    progress(
                        int(summary["completed"]), int(summary["expected"]),
                        f"知识块实体阶段完成：{output['mentions']} 个实体提及",
                    )
            except Exception as exc:
                status = self.db.fail_entity_shadow_work(
                    str(work["id"]), str(work["lease_token"]), str(exc),
                )
                summary = self.db.entity_shadow_work_summary(namespace_id)
                if progress:
                    label = "终止失败" if status == "terminal_failed" else "等待重试"
                    progress(
                        int(summary["completed"]), int(summary["expected"]),
                        f"知识块{label}：{exc}",
                    )

        relation_v2 = self._run_shadow_relation_v2_namespace(
            namespace_id=namespace_id,
            project_id=project_id,
            progress=progress,
        )
        persisted = self.db.load_entity_shadow_snapshot(namespace_id)
        mention_count = len(self.db.list_entity_shadow_mentions(namespace_id))
        canonical_count = len(persisted["canonicals"])
        artifact_payload = {
            "namespace_id": namespace_id,
            "input_fingerprint": input_fingerprint,
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
        }
        self.db.finalize_entity_shadow_namespace(
            namespace_id, metrics=metrics, artifact_hash=artifact_hash,
        )
        return {"namespace_id": namespace_id, **self.db.load_entity_shadow_snapshot(namespace_id)}

    def process_policy_snapshot_work(
        self, snapshot: dict[str, Any], work: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one frozen shadow work item without publishing product facts."""

        manifest = self._metadata(snapshot.get("snapshot_manifest"))
        if str(manifest.get("manifest_hash") or "") != str(snapshot.get("manifest_hash") or ""):
            raise RuntimeError("policy snapshot manifest hash 不匹配")
        project_id = int(snapshot["project_id"])
        phase = str(work["phase"])
        if phase == "entity":
            result = self.build_entity_shadow_namespace(
                project_id, str(snapshot["id"]), snapshot_manifest=manifest,
            )
            namespace = result["namespace"]
            if namespace.get("status") != "ready" or not namespace.get("snapshot_complete"):
                raise RuntimeError("entity shadow namespace 尚未完整收敛")
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
            namespace = self.db.get_entity_shadow_namespace_for_snapshot(str(snapshot["id"]))
            if not namespace or namespace.get("status") != "ready":
                raise RuntimeError("relation treatment 缺少完整 entity shadow endpoint")
            mentions = self.db.list_entity_shadow_mentions(str(namespace["id"]), chunk_id=chunk_id)
            shadow_state = self.db.load_entity_shadow_snapshot(str(namespace["id"]))
            canonical_by_mention = {
                str(item["mention_key"]): str(item["canonical_key"])
                for item in shadow_state.get("assignments") or []
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
        treatment_run = self._run_relation_shadow(
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
        elif all(value == "cancelled" for value in statuses):
            status = "cancelled"
        else:
            status = "completed"

        subtasks: dict[str, dict[str, Any]] = {}
        for key in self.INGESTION_SUBTASKS:
            values = []
            for job in jobs:
                metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
                stored = metadata.get("subtasks") if isinstance(metadata.get("subtasks"), dict) else {}
                values.append(stored.get(key) if isinstance(stored.get(key), dict) else {
                    "status": "waiting", "progress": 0, "detail": "",
                })
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
                "detail": f"{completed_documents}/{document_count} 个文档",
            }

        log: list[str] = []
        for index, job in enumerate(jobs, start=1):
            name = str(job.get("document_name") or (job.get("metadata") or {}).get("document_name") or f"文档 {index}")
            state = {
                "queued": "排队中", "running": "执行中", "completed": "已完成",
                "failed": "失败", "cancelled": "已取消",
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
            "metadata": {"batch_id": batch_id, "subtasks": subtasks},
        }

    def latest_job(
        self, project_id: int, kind: str = "", strategy_key: str = "",
    ) -> dict[str, Any] | None:
        return self._job_with_runtime(self.db.latest_job(project_id, kind, strategy_key))

    def latest_active_job(
        self, project_id: int, kind: str = "", strategy_key: str = "",
    ) -> dict[str, Any] | None:
        return self._job_with_runtime(self.db.latest_active_job(project_id, kind, strategy_key))

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
        payload = {"document": document_name, **result.payload()}
        temporary = payload_path.with_suffix(payload_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(payload_path)

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
        subtasks[subtask] = {
            "status": subtask_status,
            "progress": max(0, min(100, int(subtask_progress))),
            "detail": detail,
        }
        self._job(
            job_id,
            status,
            progress,
            log,
            active_subtask=subtask,
            subtasks=subtasks,
            **metadata,
        )

    def _fail_active_ingestion_subtask(self, job_id: str, detail: str, *, cancelled: bool = False) -> None:
        job = self.db.get_job(job_id) or {}
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        active = str(metadata.get("active_subtask") or "")
        if active not in self.INGESTION_SUBTASKS:
            return
        subtasks = metadata.get("subtasks") if isinstance(metadata.get("subtasks"), dict) else {}
        current = subtasks.get(active) if isinstance(subtasks.get(active), dict) else {}
        self._job_subtask(
            job_id,
            active,
            subtask_status="cancelled" if cancelled else "failed",
            subtask_progress=int(current.get("progress") or 0),
            detail=detail,
        )

    def _document_lock(self, document_id: int) -> threading.RLock:
        with self._document_locks_guard:
            return self._document_locks.setdefault(document_id, threading.RLock())

    def _graph_lock(self, project_id: int) -> threading.RLock:
        if not hasattr(self, "_graph_locks_guard"):
            self._graph_locks_guard = threading.Lock()
            self._graph_locks = {}
        with self._graph_locks_guard:
            return self._graph_locks.setdefault(project_id, threading.RLock())

    def _assert_ingestion_active(self, job_id: str, document_id: int) -> None:
        job = self.db.get_job(job_id)
        if not job or job.get("status") == "cancelled":
            raise IngestionCancelled("文档处理任务已取消")
        if not self.db.get_document(document_id):
            raise IngestionCancelled("文档已删除")

    def _document_revision_manifest(
        self, document: dict[str, Any], project: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "content_hash": document["doc_hash"],
            "source_hash": document["doc_hash"],
            "config_hash": hashlib.sha256(json.dumps({
                "requested_parser": document.get("requested_parser") or "auto",
                "requested_chunk_strategy": document.get("requested_chunk_strategy") or "auto",
                "chunk_max_tokens": int(
                    project.get("chunk_max_tokens") or self.DEFAULT_CHUNK_MAX_TOKENS
                ),
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

    def _shared_content_is_current(
        self,
        document: dict[str, Any],
        manifest: dict[str, Any],
    ) -> bool:
        revision_id = str(document.get("active_revision_id") or "")
        if not revision_id:
            return False
        revision = self.db.get_document_revision(revision_id)
        if not revision or str(revision.get("status") or "") != "active":
            return False
        return all(
            str(revision.get(key) or "") == str(manifest.get(key) or "")
            for key in ("content_hash", "source_hash", "config_hash")
        )

    def _prepare_shared_content(
        self,
        job_id: str,
        document_id: int,
        *,
        mutate_graph: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        """Prepare one strategy-neutral content revision when none is reusable."""

        document = self.db.get_document(document_id)
        if not document:
            raise IngestionCancelled("文档不存在或已删除")
        project = self.db.get_project(int(document["project_id"]))
        if not project:
            raise LookupError("项目不存在")
        manifest = self._document_revision_manifest(document, project)
        if self._shared_content_is_current(document, manifest):
            chunks = self.db.list_document_chunks(document_id)
            if chunks:
                subtasks = self._empty_ingestion_subtasks()
                subtasks["structure"] = {
                    "status": "completed", "progress": 100, "detail": "复用结构解析结果",
                }
                subtasks["chunking"] = {
                    "status": "completed", "progress": 100,
                    "detail": f"复用 {len(chunks)} 个知识块与向量",
                }
                subtasks["schema_base"] = {
                    "status": "completed", "progress": 100, "detail": "复用共享 Schema",
                }
                self._job(
                    job_id, "running", 20, "文档预处理结果已就绪，直接进入所选构建方案",
                    document_id=document_id, stage="graph", active_subtask="schema_base",
                    subtasks=subtasks, structure_ready=True, chunks_ready=True,
                    shared_content_reused=True,
                )
                return document, project, chunks

        revision: dict[str, Any] | None = None
        revision_scope: Any = None
        try:
            self._start_ingestion_job(job_id, document_id, mutate_graph=mutate_graph)
            self.db.update_document(
                document_id, status="processing", error_message=None,
                processing_stage="structure", structure_ready_at=None,
                chunks_ready_at=None, graph_ready_at=None,
            )
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
            payload_path = chunk_dir / f"{document['doc_hash'][:16]}.json"
            current_metadata = (
                document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
            )
            process_metadata = {
                **current_metadata, "chunk_file": str(payload_path),
                "media_dir": str(media_dir), "effective_parser": plan.effective_parser,
                "effective_strategy": plan.effective_strategy,
            }
            self._assert_ingestion_active(job_id, document_id)
            self._write_process_payload(
                payload_path, document["original_filename"],
                ProcessingResult(plan, parsed, [], []),
            )
            self.db.update_document(
                document_id, effective_parser=plan.effective_parser,
                effective_chunk_strategy=plan.effective_strategy,
                parser_version=parsed.parser_version, parse_quality=parsed.quality,
                routing_reason=plan.routing_reason, processing_stage="structure_ready",
                structure_ready_at=datetime.now(timezone.utc), metadata=process_metadata,
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
            self.db.update_document(
                document_id, processing_stage="chunk_ready",
                chunks_ready_at=datetime.now(timezone.utc),
                metadata={**process_metadata, "section_count": len(sections),
                          "chunk_count": len(chunks)},
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
            semantic_chunks = [
                item for item in stored_chunks
                if (item.get("metadata") or {}).get("chunk_role") != "parent"
            ]
            self._job_subtask(
                job_id, "schema_base", subtask_status="running", subtask_progress=10,
                progress=42, log="===== 文档预处理：逻辑 Schema =====", stage="schema_base",
            )
            schema_record = self.schemas.evaluate_document(
                project, document_id, semantic_chunks,
            )
            self._job_subtask(
                job_id, "schema_base", subtask_status="completed", subtask_progress=100,
                detail=f"Schema v{schema_record['version']}", progress=48,
                log=f"逻辑 Schema v{schema_record['version']} 已就绪",
            )
            revision_scope.__exit__(None, None, None)
            revision_scope = None
            document_values = {
                "status": "indexed", "indexed_at": datetime.now(timezone.utc),
                "error_message": None, "processing_stage": "chunk_ready",
                "metadata": {
                    **process_metadata, "section_count": len(sections),
                    "chunk_count": len(chunks), "schema_version": schema_record["version"],
                    "shared_content_ready": True,
                },
            }
            document, _ = self.db.activate_document_revision(
                document_id, str(revision["revision_id"]), job_id,
                str(revision["lease_token"]),
                reason="文档预处理结果已更新，策略图谱等待构建",
                document_values=document_values,
            )
            project = self.db.get_project(int(project["id"])) or project
            return document, project, self.db.list_document_chunks(document_id)
        except Exception as exc:
            if revision and revision_scope is not None:
                self.db.fail_document_revision(
                    str(revision["revision_id"]), job_id,
                    str(revision["lease_token"]), str(exc),
                )
            current = self.db.get_document(document_id)
            if current:
                self.db.update_document(
                    document_id,
                    status="indexed" if current.get("active_revision_id") else "failed",
                    processing_stage="failed", error_message=str(exc),
                )
            raise
        finally:
            if revision_scope is not None:
                revision_scope.__exit__(None, None, None)

    @recorded_job("content_preparation")
    def _prepare_document_content(self, job_id: str, document_id: int) -> None:
        """Build reusable structure, chunks, vectors and Schema for one document."""

        try:
            self._prepare_shared_content(job_id, document_id, mutate_graph=False)
            self._job(
                job_id, "completed", 100, "文档预处理完成，可选择任一方案构建知识",
                stage="completed", active_subtask="", structure_ready=True,
                chunks_ready=True, graph_ready=False,
            )
        except IngestionCancelled:
            self._fail_active_ingestion_subtask(
                job_id, "文档已删除，预处理任务已取消", cancelled=True,
            )
            self.db.update_job(
                job_id, status="cancelled", progress=100,
                log="文档已删除，预处理任务已取消",
                metadata={"stage": "cancelled"},
            )
        except Exception as exc:
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
        try:
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self._start_ingestion_job(job_id, document_id)
                self.db.update_document(document_id, status="processing", error_message=None,
                                        processing_stage="structure", structure_ready_at=None,
                                        chunks_ready_at=None, graph_ready_at=None)
            document = self.db.get_document(document_id)
            if not document: raise LookupError("文档不存在")
            project = self.db.get_project(int(document["project_id"]))
            if not project: raise LookupError("项目不存在")
            revision_manifest = {
                "content_hash": document["doc_hash"],
                "source_hash": document["doc_hash"],
                "config_hash": hashlib.sha256(json.dumps({
                    "requested_parser": document.get("requested_parser") or "auto",
                    "requested_chunk_strategy": document.get("requested_chunk_strategy") or "auto",
                    "chunk_max_tokens": int(project.get("chunk_max_tokens") or self.DEFAULT_CHUNK_MAX_TOKENS),
                    "llm_model": project.get("llm_model") or Config.default_llm_model,
                    "embedding_model": project.get("embedding_model") or Config.default_embedding_model,
                }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                "policy_hash": graph_policy_hash(
                    {axis: str(project.get(axis) or "") for axis in POLICY_AXES},
                    schema_version=int(project.get("active_schema_version") or 1),
                ),
                "document_id": document_id,
                "project_id": int(project["id"]),
            }
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
            semantic_chunks = [item for item in stored_chunks
                               if (item.get("metadata") or {}).get("chunk_role") != "parent"]
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
            extraction_chunks = [
                item for item in stored_chunks
                if (item.get("metadata") or {}).get("graph_extraction_enabled", True)
            ]
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
                    if Config.relation_shadow_enabled:
                        self._run_relation_shadow(
                            job_id, project, document_id, chunk, extraction, schema_record,
                        )
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
            if job and job.get("project_id"):
                self.db.set_graph_state(
                    int(job["project_id"]), GRAPH_STALE,
                    error="文档任务已取消，图谱需要重新同步",
                )
            self._fail_active_ingestion_subtask(job_id, str(exc), cancelled=True)
            if job and job.get("status") != "cancelled":
                self.db.update_job(job_id, status="cancelled", log=str(exc), metadata={"stage": "cancelled"})
        except Exception as exc:
            job = self.db.get_job(job_id)
            if revision:
                self.db.fail_document_revision(
                    str(revision["revision_id"]), job_id, str(revision["lease_token"]), str(exc),
                )
            if (job and job.get("status") == "cancelled") or not self.db.get_document(document_id):
                LOGGER.info("Document ingestion stopped after deletion/cancellation: %s: %s", document_id, exc)
                if job and job.get("project_id"):
                    self.db.set_graph_state(
                        int(job["project_id"]), GRAPH_STALE,
                        error="文档任务已取消，图谱需要重新同步",
                    )
                self._fail_active_ingestion_subtask(job_id, "文档已删除，任务已取消", cancelled=True)
                if job and job.get("status") != "cancelled":
                    self.db.update_job(
                        job_id,
                        status="cancelled",
                        log="文档已删除，任务已取消",
                        metadata={"stage": "cancelled"},
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

    @recorded_job("enhanced_strategy_ingestion")
    def _ingest_enhanced_strategy(self, job_id: str, document_id: int) -> None:
        """Build the isolated Shadow v2 relation + P2 entity comparison graph."""

        strategy_key = ENHANCED_SHADOW_STRATEGY
        lease: dict[str, Any] | None = None
        state: dict[str, Any] | None = None
        job = self.db.get_job(job_id) or {}
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        target_document_ids = [
            int(value) for value in (metadata.get("target_document_ids") or [document_id])
        ]
        try:
            self.db.assert_shadow_relation_v2_ready()
            project: dict[str, Any] | None = None
            for index, target_id in enumerate(target_document_ids, start=1):
                document = self.db.get_document(target_id)
                if not document:
                    raise IngestionCancelled("文档不存在或已删除")
                self._job(
                    job_id, "running", max(2, int((index - 1) / len(target_document_ids) * 40)),
                    f"===== 文档预处理 [{index}/{len(target_document_ids)}] "
                    f"{document['original_filename']} =====",
                    active_document_id=target_id,
                    active_document_index=index,
                    target_document_count=len(target_document_ids),
                )
                self.db.update_document_strategy_build(
                    target_id, strategy_key, status="processing", latest_job_id=job_id,
                    error_message="", started_at=datetime.now(timezone.utc), completed_at=None,
                )
                document, project, _ = self._prepare_shared_content(job_id, target_id)
            if project is None:
                raise RuntimeError("增强实验没有可处理文档")
            project_id = int(project["id"])
            self._assert_ingestion_active(job_id, document_id)
            eligible_documents = [
                item for item in self.db.list_documents(project_id)
                if item.get("status") == "indexed" and item.get("active_revision_id")
                and not item.get("is_deleted")
            ]
            if not eligible_documents:
                raise RuntimeError("项目尚无可用于增强实验的预处理结果")
            self._job_subtask(
                job_id, "schema_base", subtask_status="completed", subtask_progress=100,
                detail=f"{len(eligible_documents)} 个预处理结果", progress=50,
                log=(
                    "===== 增强实验：Shadow v2 关系 + P2 实体归一 =====\n"
                    "已冻结文档预处理结果与逻辑 Schema，开始构建隔离实体关系快照"
                ),
                stage="graph", strategy_key=strategy_key, graph_ready=False,
            )
            self._job_subtask(
                job_id, "extraction", subtask_status="running", subtask_progress=5,
                detail="Shadow v2/P2 处理中", progress=52,
            )

            def shadow_progress(done: int, total: int, message: str) -> None:
                self._assert_ingestion_active(job_id, document_id)
                ratio = done / max(1, total)
                self._job_subtask(
                    job_id, "extraction", subtask_status="running",
                    subtask_progress=max(5, min(95, 5 + int(ratio * 90))),
                    detail=f"{done}/{total} 个知识块",
                    progress=max(52, min(79, 52 + int(ratio * 27))),
                    log=message,
                )

            shadow = self.build_entity_shadow_namespace(
                project_id, progress=shadow_progress,
            )
            namespace = shadow.get("namespace") or {}
            if namespace.get("status") != "ready" or not namespace.get("snapshot_complete"):
                raise RuntimeError("增强实验快照未完整收敛")
            shadow["mentions"] = self.db.list_entity_shadow_mentions(shadow["namespace_id"])
            metrics = namespace.get("metrics") or {}
            self._job_subtask(
                job_id, "extraction", subtask_status="completed", subtask_progress=100,
                detail=(
                    f"{len(shadow.get('canonicals') or [])} 个实体，"
                    f"{len(shadow.get('relation_endpoints') or [])} 条关系"
                ),
                progress=80,
                log=(
                    f"增强实验快照完成：{len(shadow['mentions'])} 个实体提及，"
                    f"{len(shadow.get('canonicals') or [])} 个规范实体，"
                    f"{len(shadow.get('relation_endpoints') or [])} 条 v2 可物化关系"
                ),
            )
            state = self.db.start_strategy_graph_rebuild(project_id, strategy_key)
            generation = int(state["generation"])
            namespace_id = str(state["namespace_id"])
            policy_hash = hashlib.sha256(json.dumps({
                "strategy": strategy_key,
                "entity_policy": dict(P2_ENTITY_SHADOW_POLICY_SET),
                "relation_policy": dict(SHADOW_RELATION_V2_POLICY_SET),
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
                detail="写入隔离图", progress=85,
            )
            graph_stats = self.graph.replace_strategy_snapshot(
                project, eligible_documents, chunks_by_document, shadow,
                strategy_key=strategy_key, mutation=mutation,
            )
            completed = self.db.complete_strategy_graph_rebuild(
                project_id, strategy_key, generation,
                stats=graph_stats,
                policy_summary={
                    "mode": "shadow", "production_approved": False,
                    "policy_hash": policy_hash,
                    "entity_namespace_id": shadow["namespace_id"],
                    "entity_metrics": metrics,
                },
                empty=not graph_stats["documents"],
            )
            if not completed:
                raise RuntimeError("增强实验图代次已过期，拒绝发布就绪状态")
            now = datetime.now(timezone.utc)
            for item in eligible_documents:
                self.db.ensure_document_strategy_builds(int(item["id"]))
                build_values: dict[str, Any] = {
                    "status": "indexed",
                    "content_revision_id": item.get("active_revision_id"),
                    "stats": {
                        "entity_mentions": sum(
                            1 for mention in shadow["mentions"]
                            if int(mention["document_id"]) == int(item["id"])
                        ),
                        "relations": sum(
                            1 for relation in shadow.get("relation_endpoints") or []
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
                    f"增强实验隔离图同步完成：{graph_stats['entities']} 个实体，"
                    f"{graph_stats['relations']} 条关系"
                ),
            )
            self._job(
                job_id, "completed", 100, "增强实验构建完成（仍保持 shadow，不激活生产）",
                stage="completed", graph_ready=True, strategy_key=strategy_key,
                graph_stats=graph_stats, production_approved=False,
            )
        except IngestionCancelled as exc:
            for target_id in target_document_ids:
                self.db.update_document_strategy_build(
                    target_id, strategy_key, status="cancelled", latest_job_id=job_id,
                    error_message=str(exc), completed_at=datetime.now(timezone.utc),
                )
            self._fail_active_ingestion_subtask(job_id, str(exc), cancelled=True)
            self.db.update_job(job_id, status="cancelled", progress=100, log=str(exc), error=str(exc))
        except Exception as exc:
            LOGGER.exception("Enhanced strategy ingestion failed: %s", document_id)
            if state:
                self.db.fail_strategy_graph_rebuild(
                    int(state["project_id"]), strategy_key, int(state["generation"]), str(exc),
                )
            for target_id in target_document_ids:
                self.db.update_document_strategy_build(
                    target_id, strategy_key, status="failed", latest_job_id=job_id,
                    error_message=str(exc), completed_at=datetime.now(timezone.utc),
                )
            self._fail_active_ingestion_subtask(job_id, str(exc))
            self.db.update_job(
                job_id, status="failed", progress=100,
                log=f"增强实验构建失败：{exc}", error=str(exc),
                metadata={"stage": "failed", "strategy_key": strategy_key},
            )
        finally:
            if lease:
                try:
                    self.db.release_graph_lease(str(lease["id"]), str(lease["lease_token"]))
                except Exception:
                    LOGGER.exception("Unable to release enhanced strategy graph lease")

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

    def _run_relation_shadow(
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
            P1_RELATION_SHADOW_POLICY_SET,
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
            raise ValueError("relation shadow 必须绑定不可变 revision_id")
        context_budget = int(
            project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS
        )
        prepared = self.relation_shadow.prepare_shadow(
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
        input_fingerprint = self._relation_shadow_input_fingerprint(
            revision_id=revision_id,
            chunk_id=int(chunk["id"]),
            content=str(chunk.get("content") or ""),
            schema_version=schema_version,
            treatment_hash=treatment_hash,
            candidate_map=candidate_map,
        )
        run_id = self._relation_shadow_run_id(
            int(project["id"]),
            document_id,
            revision_id,
            int(chunk["id"]),
            treatment_hash,
            input_fingerprint,
        )
        started = False
        try:
            self.db.start_relation_shadow_run({
                "id": run_id,
                "project_id": int(project["id"]),
                "document_id": document_id,
                "chunk_id": int(chunk["id"]),
                "control_policy_hash": GRAPH_POLICY_HASH,
                "treatment_policy_hash": treatment_hash,
                "policy_set": dict(P1_RELATION_SHADOW_POLICY_SET),
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
                    "max_attempts": Config.relation_shadow_max_attempts,
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
                    "max_attempts": Config.relation_shadow_max_attempts,
                    "error_message": item.get("reason") or "candidate deferred",
                })
            self.db.prepare_relation_shadow_work(
                run_id, work_items, quarantined=quarantined,
            )

            batch_map = {item.batch_id: item for item in prepared.batches}
            candidate_objects = {
                item.candidate_id: item for item in prepared.plan.candidates
            }
            owner = f"{ingestion_run_id}:{threading.get_ident()}"
            while True:
                leased = self.db.lease_relation_shadow_work(
                    run_id, owner, lease_seconds=Config.relation_shadow_lease_seconds,
                )
                if not leased:
                    break
                batch = batch_map[str(leased["batch_id"])]
                selected = [
                    candidate_objects[str(candidate_id)]
                    for candidate_id in list(leased.get("candidate_ids") or [])
                ]
                try:
                    decisions, raw = self.relation_shadow.classify_prepared_batch(
                        classifier=self.ollama,
                        model=project.get("llm_model") or Config.default_llm_model,
                        context_budget_tokens=context_budget,
                        batch=batch,
                        candidates=selected,
                        content=str(chunk.get("content") or ""),
                        logical_schema=schema_record.get("schema_json") or {},
                    )
                    self.db.settle_relation_shadow_work(
                        str(leased["id"]), str(leased["lease_token"]),
                        output={"decisions": decisions, "raw": raw},
                    )
                except Exception as exc:
                    self.db.settle_relation_shadow_work(
                        str(leased["id"]), str(leased["lease_token"]),
                        error_message=str(exc)[:2000],
                        retry_delay_seconds=Config.relation_shadow_retry_delay_seconds,
                    )

            persisted_work = self.db.list_relation_shadow_work(run_id)
            open_items = [
                item for item in persisted_work
                if item["status"] in {"pending", "leased", "retryable_failed"}
            ]
            if open_items:
                raise RuntimeError("relation shadow work queue 尚未收敛")
            decisions: list[dict[str, Any]] = []
            output_tokens = 0
            for item in persisted_work:
                output = item.get("output_json") or {}
                if isinstance(output, str):
                    output = json.loads(output)
                for decision in output.get("decisions") or []:
                    decisions.append({
                        **decision,
                        "policy_version": P1_RELATION_SHADOW_POLICY_SET[
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
            funnel_report = relation_shadow_funnel(
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
            finished = self.db.finish_relation_shadow_run(
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
            LOGGER.warning("Relation shadow failed without affecting control: %s", exc)
            if started:
                try:
                    self.db.finish_relation_shadow_run(
                        run_id, candidates=[], decisions=[],
                        metrics={"shadow_exception": type(exc).__name__,
                                 "coverage_complete": False,
                                 "coverage_failures": ["terminal_failure"]},
                        status="failed", error_message=str(exc)[:2000],
                    )
                except Exception:
                    LOGGER.exception("Unable to persist isolated relation shadow failure")
            if raise_on_failure:
                raise
            return {
                "run_id": run_id,
                "status": "failed",
                "artifact_hash": "",
                "metrics": {"shadow_exception": type(exc).__name__,
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
    ) -> dict[str, int]:
        """Purge and deterministically rebuild one project graph, then verify counts."""

        with self._graph_lock(project_id):
            return self._rebuild_project_graph_locked(
                project_id,
                include_document_id=include_document_id,
                target_generation=target_generation,
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
        return self._rebuild_project_graph(
            project_id, target_generation=target_generation,
        )

    def _rebuild_project_graph_locked(
        self,
        project_id: int,
        *,
        include_document_id: int | None = None,
        target_generation: int | None = None,
    ) -> dict[str, int]:
        """Execute a rebuild while the process-local project graph lock is held."""

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
        self._retry_entity_indexing_failures(project)
        self._retry_relation_indexing_failures(project)
        blocking = self.db.list_blocking_indexing_failures(project_id, fixed_generation)
        if blocking:
            raise RuntimeError(
                f"图谱代次 {fixed_generation} 仍有 {len(blocking)} 个派生索引待修复"
            )
        namespace = str(project.get("graph_namespace_id") or "")
        policy_hash = project_graph_policy_hash(project)
        owner = f"graph-rebuild:{uuid.uuid4().hex}"
        lease = self.db.acquire_graph_lease(
            project_id, namespace, fixed_generation, owner,
            ttl_seconds=Config.graph_lease_seconds,
        )
        mutation = GraphMutationContext(
            namespace_id=namespace, project_id=project_id, generation=fixed_generation,
            policy_hash=policy_hash, lease_token=str(lease["lease_token"]),
        )

        def heartbeat() -> None:
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
                project, documents, mutation=mutation,
            )
            self.db.persist_graph_manifest(
                project_id, namespace, fixed_generation, "expected",
                mutation.lease_token, expected_manifest,
            )
            heartbeat()
            self.graph.ensure_schema()
            self.graph.delete_project(project, mutation=mutation)
            written_relations = 0
            for document in documents:
                heartbeat()
                chunks = self.db.list_document_chunks(int(document["id"]))
                self.graph.write_document_snapshot(
                    project, document, chunks, mutation=mutation,
                )
                report = self.graph.sync_document_assertions(
                    project, document, mutation=mutation,
                )
                written_relations += int(report.get("relations") or 0)
                heartbeat()
            if documents:
                self.graph.prune_project_entities(project, mutation=mutation)
            expected_relations = self.db.count_publishable_relations(project_id, document_ids)
            graph_stats = self.graph.statistics(project)
            graph_relations = int(graph_stats.get("relations") or 0)
            if written_relations != expected_relations or graph_relations != expected_relations:
                raise RuntimeError(
                    "图谱关系一致性检查失败："
                    f"PostgreSQL={expected_relations}, written={written_relations}, "
                    f"HugeGraph={graph_relations}"
                )
            writer_manifest = self.graph.writer_manifest(project, mutation=mutation)
            actual_manifest = self.graph.actual_manifest(project, mutation=mutation)
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
            )
            if not completed:
                raise RuntimeError("图谱重建完成时 generation、lease 或 manifest 已过期")
            return {
                "documents": len(documents),
                "entities": int(graph_stats.get("entities") or 0),
                "relations": graph_relations,
            }
        finally:
            self.db.release_graph_lease(str(lease["id"]), mutation.lease_token)

    def _retry_entity_indexing_failures(self, project: dict[str, Any]) -> int:
        """Retry assignment-derived entity vectors before graph readiness checks."""

        project_id = int(project["id"])
        model = str(project.get("embedding_model") or Config.default_embedding_model)
        failures = self.db.list_pending_indexing_failures(project_id, "entity")
        completed = 0
        for failure in failures:
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

    def _retry_relation_indexing_failures(self, project: dict[str, Any]) -> int:
        """Retry derived relation vectors without changing assertion publication."""

        project_id = int(project["id"])
        model = str(project.get("embedding_model") or Config.default_embedding_model)
        failures = self.db.list_pending_indexing_failures(project_id, "relation")
        completed = 0
        for failure in failures:
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
        try:
            job = self.db.begin_graph_mutation(job_id)
            target_generation = int(
                job.get("target_generation")
                if job.get("target_generation") is not None
                else (self.db.get_project(project_id) or {}).get("graph_generation") or 0
            )
            self._job(job_id, progress=5, log="开始按当前验证策略重建项目知识图谱")
            report = self._rebuild_project_graph(
                project_id, target_generation=target_generation,
            )
            self._job(
                job_id, "completed", 100,
                f"项目图谱同步完成，共处理 {report['documents']} 个文档、{report['relations']} 条关系",
                graph_ready=True,
            )
        except Exception as exc:
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
        self.db.cancel_document_jobs(document_id)
        self.db.cancel_project_jobs(
            int(document["project_id"]), ["schema_reassessment"],
            "项目文档已删除，本次 Schema 重新评估已取消，请按需重新执行",
        )
        # Make the user-visible deletion immediate. The final cleanup runs after any
        # in-flight resolver call releases the per-document lock and reapplies the
        # soft-delete cascade to catch writes that were already in progress.
        self.db.soft_delete_document(document_id)
        self.db.set_graph_state(
            int(document["project_id"]), GRAPH_STALE,
            error="项目文档已删除，图谱正在重新收敛",
        )
        document_name = str(document.get("original_filename") or f"文档 {document_id}")
        job = self.db.create_job("document_cleanup", int(document["project_id"]), document_id)
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
                if self.db.count_documents(project_id) == 0:
                    self.schemas.activate_base(project_id)

            report = self._rebuild_project_graph(project_id)
            self._job(
                job_id, "running", 92,
                f"HugeGraph 已按剩余文档重建：{report['relations']} 条关系",
                stage="cleanup",
            )
            self._job(job_id, "completed", 100, "文档及其派生数据清理完成", stage="completed")
        except Exception as exc:
            LOGGER.exception("Document cleanup failed: %s", document_id)
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

    def search(self, project_id: int, query: str, top_k: int | None = None) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if not project: raise LookupError("项目不存在")
        query = (query or "").strip()
        if not query: raise ValueError("请输入检索内容")
        limit = min(top_k or 20, 50)
        lexical = self.db.search_graph_text(project_id, query, limit)
        entity_hits = self.vectors.search_type(project, query, "entity", limit)
        relation_hits = self.vectors.search_type(project, query, "relation", limit)
        entity_scores = {int(item["object_key"]): item["score"] for item in entity_hits if str(item["object_key"]).isdigit()}
        relation_scores = {int(item["object_key"]): item["score"] for item in relation_hits if str(item["object_key"]).isdigit()}
        entities = {int(item["id"]): dict(item) for item in lexical["entities"]}
        for item in self.db.get_entities_by_ids(project_id, list(entity_scores)):
            value = entities.setdefault(int(item["id"]), dict(item)); value["score"] = entity_scores[int(item["id"])]
        relation_ids = {
            *(int(item["id"]) for item in lexical["relations"]),
            *relation_scores,
        }
        relations = {}
        for item in self.db.get_relations_by_ids(project_id, list(relation_ids)):
            value = relations.setdefault(int(item["id"]), dict(item))
            if int(item["id"]) in relation_scores:
                value["score"] = relation_scores[int(item["id"])]
        return {"query": query, "chunks": self.vectors.search(project, query, top_k),
                "entities": sorted(entities.values(), key=lambda item: item.get("score", 0), reverse=True)[:limit],
                "relations": sorted(relations.values(), key=lambda item: item.get("score", item.get("confidence", 0)), reverse=True)[:limit]}

    def health(self) -> dict[str, Any]:
        self.db.list_projects()
        ollama_models = len(self.ollama.models())
        graphs = self.graph.available_graphs()
        return {"status": "ok", "postgres": "reachable", "ollama_models": ollama_models,
                "hugegraph": Config.hugegraph_graph, "hugegraph_ready": Config.hugegraph_graph in graphs,
                "vector_backend": "pgvector-hnsw",
                "entity_reranker": self.resolver.reranker.diagnostics()}
