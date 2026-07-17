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
from .hugegraph_store import HugeGraphStore
from .media import DocumentMediaService
from .ollama import OllamaClient
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
        self.graph = HugeGraphStore(self.db)
        self._document_locks_guard = threading.Lock()
        self._document_locks: dict[int, threading.RLock] = {}
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
        self.executor.submit(self._ingest_document, job["id"], document_id)
        return str(job["id"])

    def submit_graph_build(self, project_id: int) -> str:
        if not self.db.get_project(project_id): raise LookupError("项目不存在")
        job = self.db.create_job("graph_sync", project_id)
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
        if not self.db.get_project(project_id):
            raise LookupError("项目不存在")
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

    def _assert_ingestion_active(self, job_id: str, document_id: int) -> None:
        job = self.db.get_job(job_id)
        if not job or job.get("status") == "cancelled":
            raise IngestionCancelled("文档处理任务已取消")
        if not self.db.get_document(document_id):
            raise IngestionCancelled("文档已删除")

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
            self.db.update_project(int(project["id"]), graph_status="building")
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

            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self.graph.replace_document(project, document, stored_chunks)
            self._job_subtask(
                job_id, "schema_base", subtask_status="completed", subtask_progress=100,
                detail=f"{len(stored_chunks)} 个基础知识块", progress=50,
                log=f"文档与 {len(stored_chunks)} 个知识块已写入 HugeGraph",
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
                    with self._document_lock(document_id):
                        self._assert_ingestion_active(job_id, document_id)
                        mentions, relations = self.resolver.resolve_chunk(
                            project, document_id, chunk, extraction, int(schema_record["version"]),
                            session=resolution_session,
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
                self.db.retire_orphan_entities()
                graph_report = self.graph.sync_document_assertions(project, document)
                self.graph.prune_project_entities(int(project["id"]))
                self.db.update_project(int(project["id"]), graph_status="ready")
                graph_ready_at = datetime.now(timezone.utc)
            self._job_subtask(
                job_id, "graph_sync", subtask_status="completed", subtask_progress=100,
                detail=f"{graph_report['entities']} 个实体，{graph_report['relations']} 条关系",
                progress=96,
                log=f"HugeGraph 同步完成：{graph_report['entities']} 个实体，{graph_report['relations']} 条关系",
            )
            with self._document_lock(document_id):
                self._assert_ingestion_active(job_id, document_id)
                self.db.update_document(document_id, status="indexed", indexed_at=datetime.now(timezone.utc), error_message=None,
                    processing_stage="completed", graph_ready_at=graph_ready_at,
                    content_version=int(document.get("content_version") or 0) + 1,
                    metadata={**process_metadata, "section_count": len(sections), "chunk_count": len(chunks),
                              "effective_parser": plan.effective_parser, "effective_strategy": plan.effective_strategy,
                              "schema_version": schema_record["version"], "entity_mentions": total_mentions,
                              "relations": total_relations, "extraction_errors": extraction_errors,
                              "gleaning_chunks": gleaning_chunks, "verified_relations": verified_relations,
                              "rejected_relations": rejected_relations,
                              "relation_verification": verification_totals})
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
            self._fail_active_ingestion_subtask(job_id, str(exc), cancelled=True)
            job = self.db.get_job(job_id)
            if job and job.get("status") != "cancelled":
                self.db.update_job(job_id, status="cancelled", log=str(exc), metadata={"stage": "cancelled"})
        except Exception as exc:
            job = self.db.get_job(job_id)
            if (job and job.get("status") == "cancelled") or not self.db.get_document(document_id):
                LOGGER.info("Document ingestion stopped after deletion/cancellation: %s: %s", document_id, exc)
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
            self._fail_active_ingestion_subtask(job_id, str(exc))
            self.db.update_document(document_id, status="failed", processing_stage="failed", error_message=str(exc))
            self.db.update_job(job_id, status="failed", progress=100, log=f"处理失败：{exc}", error=str(exc),
                               metadata={"stage": "failed"})

    def _sync_project_graph(self, job_id: str, project_id: int) -> None:
        self._job(job_id, "running", 5, "开始同步项目知识图谱")
        try:
            project = self.db.get_project(project_id)
            if not project: raise LookupError("项目不存在")
            documents = [item for item in self.db.list_documents(project_id) if item["status"] == "indexed"]
            self.graph.ensure_schema()
            for index, document in enumerate(documents, 1):
                chunks = self.db.list_document_chunks(int(document["id"]))
                self.graph.replace_document(project, document, chunks)
                report = self.graph.sync_document_assertions(project, document)
                self._job(job_id, progress=5 + int(index / max(1, len(documents)) * 90),
                          log=f"{document['original_filename']}：{report['entities']} 个实体，{report['relations']} 条关系")
            self.db.update_project(project_id, graph_status="ready")
            self._job(job_id, "completed", 100, f"项目图谱同步完成，共处理 {len(documents)} 个文档")
        except Exception as exc:
            LOGGER.exception("Project graph sync failed: %s", project_id)
            self.db.update_job(job_id, status="failed", progress=100, log=str(exc), error=str(exc))

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
                try:
                    self.graph.delete_document(document_id)
                    self._job(job_id, "running", 55, "HugeGraph 文档数据已清理", stage="cleanup")
                except Exception as exc:
                    LOGGER.warning("HugeGraph document cleanup deferred for %s: %s", document_id, exc)
                    self._job(job_id, "running", 55, f"HugeGraph 清理失败，已记录：{exc}", stage="cleanup")

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

            try:
                self.graph.prune_project_entities(project_id)
            except Exception as exc:
                LOGGER.warning("HugeGraph orphan pruning deferred for project %s: %s", project_id, exc)
            self._job(job_id, "completed", 100, "文档及其派生数据清理完成", stage="completed")
        except Exception as exc:
            LOGGER.exception("Document cleanup failed: %s", document_id)
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
        relations = {int(item["id"]): dict(item) for item in lexical["relations"]}
        for item in self.db.get_relations_by_ids(project_id, list(relation_scores)):
            value = relations.setdefault(int(item["id"]), dict(item)); value["score"] = relation_scores[int(item["id"])]
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
