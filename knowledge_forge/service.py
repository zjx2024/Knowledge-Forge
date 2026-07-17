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
from typing import Any
from urllib.parse import quote

import requests

from .chunking import ProcessingResult, build_processing_result, parse_document_structure, validate_processing_request
from .config import Config
from .database import Database
from .entity_resolution import EntityResolver
from .entity_shadow import EntityShadowState
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
from .media import DocumentMediaService
from .ollama import OllamaClient
from .observability import recorded_job
from .policy import P1_RELATION_SHADOW_POLICY_SET, P2_ENTITY_SHADOW_POLICY_SET
from .quality import evaluate_release_gate, validate_gold_manifest
from .relation_pipeline import VersionedRelationPipeline
from .schema_evolution import SchemaEvolutionService
from .vector_store import VectorStore


LOGGER = logging.getLogger(__name__)


class IngestionCancelled(RuntimeError):
    """Raised inside a worker after its document has been deleted or cancelled."""


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
                self.executor.submit(self._ingest_document, job_id, int(job["document_id"]))
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
                return self.db.update_document(int(existing["id"]), **{**values, "status": "uploaded", "error_message": None,
                    "is_deleted": False, "deleted_at": None, "indexed_at": None,
                    "effective_parser": None, "effective_chunk_strategy": None, "parse_quality": {},
                    "routing_reason": None, "processing_stage": "uploaded", "structure_ready_at": None,
                    "chunks_ready_at": None, "graph_ready_at": None})
            return self.db.create_document(values)
        finally:
            tmp_path.unlink(missing_ok=True)

    def submit_ingestion(self, document_id: int) -> str:
        document = self.db.get_document(document_id)
        if not document: raise LookupError("文档不存在")
        if document["status"] == "processing": raise ValueError("该文档正在处理")
        job = self.db.create_job("ingestion", int(document["project_id"]), document_id)
        project = self.db.get_project(int(document["project_id"]))
        if project and project.get("graph_status") == GRAPH_READY:
            self.db.set_graph_state(
                int(document["project_id"]), GRAPH_STALE,
                error="文档重新入库后图谱需要重建",
            )
        self.executor.submit(self._ingest_document, job["id"], document_id)
        return str(job["id"])

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
                    "chunk_id": relation.get("chunk_id"),
                    "object_type": "relation",
                    "object_key": object_key,
                    "model_name": model,
                    "last_error": str(exc),
                    "payload": payload,
                })
                LOGGER.warning("Relation embedding refresh deferred for %s: %s", object_key, exc)
        return failures

    def build_entity_shadow_namespace(self, project_id: int, snapshot_id: str = "") -> dict[str, Any]:
        """Replay indexed documents in stable order into an isolated P2 entity namespace."""

        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        documents = [
            item for item in self.db.list_documents(project_id)
            if item.get("status") == "indexed" and not item.get("is_deleted")
        ]
        documents.sort(key=lambda item: int(item["id"]))
        material = "|".join((
            str(project_id), snapshot_id or "live",
            *(str(item["id"]) for item in documents),
            P2_ENTITY_SHADOW_POLICY_SET["entity_resolution_policy_version"],
        ))
        namespace_id = "entity_shadow_" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        self.db.create_entity_shadow_namespace(
            project_id, namespace_id, dict(P2_ENTITY_SHADOW_POLICY_SET),
            [int(item["id"]) for item in documents], snapshot_id or None,
        )
        state = EntityShadowState(
            namespace=namespace_id,
            policy_version=P2_ENTITY_SHADOW_POLICY_SET["entity_resolution_policy_version"],
        )
        mention_payloads: list[dict[str, Any]] = []
        for document in documents:
            for mention in sorted(
                self.db.list_document_mentions(int(document["id"])),
                key=lambda item: (int(item.get("chunk_id") or 0), int(item["id"])),
            ):
                mention_payloads.append({
                    "mention_id": mention.get("mention_key") or f"legacy_mention_{mention['id']}",
                    "surface": mention.get("evidence_text") or mention["mention_text"],
                    "normalized_surface": mention["normalized_name"],
                    "entity_type": mention["entity_type"],
                    "aliases": (mention.get("metadata") or {}).get("accepted_aliases") or [],
                    "document_id": int(document["id"]),
                    "chunk_id": int(mention["chunk_id"]),
                })
        state.replay(mention_payloads)
        snapshot = state.snapshot()
        payload_by_mention = {item["mention_id"]: item for item in mention_payloads}
        for assignment in snapshot["assignments"]:
            assignment.update(payload_by_mention[assignment["mention_id"]])
        self.db.persist_entity_shadow_snapshot(
            namespace_id, snapshot["canonicals"], snapshot["assignments"],
            metrics={
                "documents": len(documents), "mentions": len(mention_payloads),
                "canonicals": len(snapshot["canonicals"]),
                "duplicate_canonical_rate": None,
                "canonical_compression_rate": (
                    0.0 if not mention_payloads
                    else max(
                        0.0,
                        1.0 - (len(snapshot["canonicals"]) / len(mention_payloads)),
                    )
                ),
            },
        )
        return {"namespace_id": namespace_id, **snapshot}

    def evaluate_policy_snapshot_gate(
        self,
        project_id: int,
        snapshot_id: str,
        *,
        manifest: dict[str, Any],
        control_metrics: dict[str, float],
        treatment_metrics: dict[str, float],
        approved_by: str | None = None,
    ) -> dict[str, Any]:
        """Persist an auditable gate; model-only labels can never approve production."""

        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        manifest_report = validate_gold_manifest(manifest, require_production=False)
        gate_report = evaluate_release_gate(
            control=control_metrics, treatment=treatment_metrics,
            manifest=manifest_report,
        )
        evaluation = self.db.create_quality_evaluation_run({
            "project_id": project_id,
            "snapshot_id": snapshot_id,
            "control_policy_hash": str(project.get("active_policy_hash") or "control"),
            "treatment_policy_hash": snapshot_id,
            "dataset": manifest["dataset"],
            "dataset_version": manifest["version"],
            "split": "combined",
            "manifest": manifest,
            "manifest_valid": manifest_report["production_valid"],
            "metrics": {"control": control_metrics, "treatment": treatment_metrics},
            "denominators": manifest_report,
            "cost_metrics": {
                "control": control_metrics.get("cost"),
                "treatment": treatment_metrics.get("cost"),
            },
            "status": "completed",
        })
        gate = self.db.upsert_policy_activation_gate({
            "project_id": project_id,
            "snapshot_id": snapshot_id,
            "evaluation_run_id": evaluation["id"],
            "manifest_valid": gate_report["manifest_valid"],
            "engineering_complete": True,
            "production_approved": gate_report["production_approved"],
            "failed_gates": gate_report["failed_gates"],
            "gate_report": gate_report,
            "approved_by": approved_by if gate_report["production_approved"] else None,
        })
        return {"evaluation": evaluation, "gate": gate, "report": gate_report}

    def activate_treatment_snapshot(self, project_id: int, snapshot_id: str) -> dict[str, Any]:
        project = self.db.activate_policy_snapshot_with_gate(project_id, snapshot_id)
        rollout = self.db.record_policy_rollout(
            project_id, snapshot_id, "canary", percentage=100.0, status="active",
            reason="生产门禁通过后按项目级 canary 激活",
        )
        self.submit_graph_build(project_id)
        return {"project": project, "rollout": rollout}

    @staticmethod
    def _project_provenance_gremlin(project_id: int) -> str:
        """Show the project graph plus document paths through shared entities."""
        project_id = int(project_id)
        return (
            f'g.V().has("kf_project_id",{project_id}).union('
            f'__.bothE().has("kf_project_id",{project_id}).dedup().otherV().limit(300),'
            '__.hasLabel("KFEntity")'
            f'.where(__.inE("kf_mentions").has("kf_project_id",{project_id})'
            '.values("kf_doc_id").dedup().count().is(gte(2)))'
            f'.inE("kf_mentions").has("kf_project_id",{project_id}).outV()'
            f'.inE("kf_contains").has("kf_project_id",{project_id}).outV().limit(200)'
            ').path().limit(500)'
        )

    def hubble_project_url(self, project_id: int) -> str:
        """Resolve Hubble and prefill a cross-document provenance query."""
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        if not is_current_graph(project):
            raise GraphNotReadyError(
                project.get("graph_error") or "图谱需要按当前验证策略重建"
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

        gremlin = self._project_provenance_gremlin(project_id)
        return (
            f"{base_url}/graph-management/{int(connection['id'])}/data-analyze"
            f"?gremlin={quote(gremlin, safe='')}"
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.db.get_job(job_id)

    def latest_job(self, project_id: int, kind: str = "") -> dict[str, Any] | None:
        return self.db.latest_job(project_id, kind)

    def latest_active_job(self, project_id: int, kind: str = "") -> dict[str, Any] | None:
        return self.db.latest_active_job(project_id, kind)

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

    def _start_ingestion_job(self, job_id: str, document_id: int) -> None:
        self.db.begin_graph_mutation(job_id)
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

    @recorded_job("ingestion")
    def _ingest_document(self, job_id: str, document_id: int) -> None:
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
                stored_chunks = self.db.replace_document_content(document_id, section_values, chunk_values)
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
                "content_version": int(document.get("content_version") or 0) + 1,
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
                    **document_values,
                )
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
            if job and job.get("project_id"):
                self.db.set_graph_state(
                    int(job["project_id"]), GRAPH_FAILED, error=str(exc),
                )
            self._fail_active_ingestion_subtask(job_id, str(exc))
            self.db.update_document(document_id, status="failed", processing_stage="failed", error_message=str(exc))
            self.db.update_job(job_id, status="failed", progress=100, log=f"处理失败：{exc}", error=str(exc),
                               metadata={"stage": "failed"})

    def _run_relation_shadow(
        self,
        ingestion_run_id: str,
        project: dict[str, Any],
        document_id: int,
        chunk: dict[str, Any],
        extraction: dict[str, Any],
        schema_record: dict[str, Any],
    ) -> None:
        """Run P1 relation treatment in isolation; never fail or mutate the control path."""

        schema_version = int(schema_record.get("version") or 1)
        treatment_hash = graph_policy_hash(
            P1_RELATION_SHADOW_POLICY_SET,
            schema_version=schema_version,
            config_summary={
                "max_candidates": Config.relation_classifier_max_candidates,
                "output_reserve": Config.relation_classifier_output_reserve,
                "safety_margin": Config.relation_classifier_safety_margin,
            },
        )
        run_material = (
            f"{ingestion_run_id}|{project['id']}|{document_id}|{chunk['id']}|{treatment_hash}"
        )
        run_id = "relshadow_" + hashlib.sha256(run_material.encode("utf-8")).hexdigest()
        started = False
        try:
            self.db.start_relation_shadow_run({
                "id": run_id,
                "project_id": int(project["id"]),
                "document_id": document_id,
                "chunk_id": int(chunk["id"]),
                "control_policy_hash": GRAPH_POLICY_HASH,
                "treatment_policy_hash": treatment_hash,
                "policy_set": P1_RELATION_SHADOW_POLICY_SET,
            })
            started = True
            result = self.relation_shadow.classify_shadow(
                classifier=self.ollama,
                model=project.get("llm_model") or Config.default_llm_model,
                model_max_tokens=int(project.get("model_max_tokens") or self.DEFAULT_MODEL_MAX_TOKENS),
                project_scope=str(project.get("project_key") or project["id"]),
                chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
                content=str(chunk.get("content") or ""),
                entities=list(extraction.get("entities") or []),
                logical_schema=schema_record.get("schema_json") or {},
            )
            candidates = [asdict(item) for item in result.plan.candidates]
            self.db.finish_relation_shadow_run(
                run_id,
                candidates=candidates,
                decisions=result.decisions,
                metrics={**result.metrics, "batches": result.batches},
                status=result.status,
                error_message=result.error,
            )
        except Exception as exc:
            LOGGER.warning("Relation shadow failed without affecting control: %s", exc)
            if started:
                try:
                    self.db.finish_relation_shadow_run(
                        run_id, candidates=[], decisions=[],
                        metrics={"shadow_exception": type(exc).__name__},
                        status="failed", error_message=str(exc)[:2000],
                    )
                except Exception:
                    LOGGER.exception("Unable to persist isolated relation shadow failure")

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
        if hasattr(self.db, "list_blocking_indexing_failures"):
            blocking = self.db.list_blocking_indexing_failures(project_id, fixed_generation)
            if blocking:
                raise RuntimeError(
                    f"图谱代次 {fixed_generation} 仍有 {len(blocking)} 个派生索引待修复"
                )
        documents = [
            item for item in self.db.list_documents(project_id)
            if item["status"] == "indexed" or int(item["id"]) == include_document_id
        ]
        document_ids = [int(item["id"]) for item in documents]
        self.graph.ensure_schema()
        self.graph.delete_project(project_id)
        written_relations = 0
        written_entities = 0
        for document in documents:
            chunks = self.db.list_document_chunks(int(document["id"]))
            writer = getattr(self.graph, "write_document_snapshot", None)
            if writer:
                writer(project, document, chunks)
            else:
                self.graph.replace_document(project, document, chunks)
            report = self.graph.sync_document_assertions(project, document)
            written_relations += int(report.get("relations") or 0)
            written_entities += int(report.get("entities") or 0)
        if documents:
            self.graph.prune_project_entities(project_id)
        expected_relations = self.db.count_publishable_relations(project_id, document_ids)
        graph_stats = self.graph.statistics(project_id)
        graph_relations = int(graph_stats.get("relations") or 0)
        if written_relations != expected_relations or graph_relations != expected_relations:
            raise RuntimeError(
                "图谱关系一致性检查失败："
                f"PostgreSQL={expected_relations}, written={written_relations}, HugeGraph={graph_relations}"
            )
        if not documents:
            if hasattr(self.db, "complete_graph_rebuild"):
                completed = self.db.complete_graph_rebuild(
                    project_id,
                    fixed_generation,
                    policy_hash=project_graph_policy_hash(project),
                    empty=True,
                )
                if not completed:
                    raise RuntimeError("图谱重建完成时目标代次已过期")
            else:
                self.db.set_graph_state(project_id, GRAPH_EMPTY)
            return {"documents": 0, "entities": 0, "relations": 0}
        if hasattr(self.db, "complete_graph_rebuild"):
            completed = self.db.complete_graph_rebuild(
                project_id,
                fixed_generation,
                policy_hash=project_graph_policy_hash(project),
            )
            if not completed:
                raise RuntimeError("图谱重建完成时目标代次已过期")
        else:
            self.db.set_graph_state(project_id, GRAPH_READY, current_policy=True)
        return {
            "documents": len(documents),
            "entities": written_entities,
            "relations": graph_relations,
        }

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
                "vector_backend": "pgvector-hnsw"}
