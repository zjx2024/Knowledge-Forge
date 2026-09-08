"""Dedicated HTTP service for cross-encoder reranking.

The service owns torch/Transformers and model memory.  Knowledge Forge talks to
it through a small, bounded ``query + documents`` protocol so model lifecycle,
GPU pressure and LAN access are independent from web and background jobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import logging
import math
from pathlib import Path
import signal
import threading
import time
from typing import Any
from uuid import uuid4

from .config import Config


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "knowledge-forge-rerank-v1"


class RerankerServiceError(RuntimeError):
    """A request cannot be served without weakening its contract."""


def _is_loopback(host: str) -> bool:
    value = str(host or "").strip().lower()
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class RerankerServiceSettings:
    host: str
    port: int
    model_name: str
    api_key: str
    allow_unauthenticated_lan: bool
    max_documents: int
    max_text_chars: int
    max_total_chars: int
    max_body_bytes: int
    max_pending_requests: int
    queue_timeout: int

    @classmethod
    def from_config(cls) -> "RerankerServiceSettings":
        return cls(
            host=Config.reranker_host,
            port=Config.reranker_port,
            model_name=Config.reranker_model_name,
            api_key=Config.entity_reranker_api_key,
            allow_unauthenticated_lan=(
                Config.reranker_allow_unauthenticated_lan
            ),
            max_documents=Config.reranker_max_documents,
            max_text_chars=Config.reranker_max_text_chars,
            max_total_chars=Config.reranker_max_total_chars,
            max_body_bytes=Config.reranker_max_body_bytes,
            max_pending_requests=Config.reranker_max_pending_requests,
            queue_timeout=Config.reranker_queue_timeout,
        )

    def validate(self) -> None:
        if not self.host or not 1 <= int(self.port) <= 65535:
            raise RerankerServiceError("reranker 监听地址或端口无效")
        if not self.model_name:
            raise RerankerServiceError("RERANKER_MODEL_NAME 不能为空")
        if self.api_key and len(self.api_key) < 16:
            raise RerankerServiceError("RERANKER_API_KEY 至少需要 16 个字符")
        if (
            not _is_loopback(self.host)
            and not self.api_key
            and not self.allow_unauthenticated_lan
        ):
            raise RerankerServiceError(
                "非回环 reranker 监听必须配置 RERANKER_API_KEY，或显式设置 "
                "RERANKER_ALLOW_UNAUTHENTICATED_LAN=true"
            )


class CrossEncoderRuntime:
    """One process-local model instance with serialized inference."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        gpu_min_free_mb: int | None = None,
    ) -> None:
        self.model_path = Path(
            model_path or Config.entity_reranker_model_path
        )
        self.model_name = str(
            model_name or Config.reranker_model_name
        )
        self.device_preference = str(
            device or Config.entity_reranker_device
        ).lower()
        self.batch_size = max(
            1, int(batch_size or Config.entity_reranker_batch_size)
        )
        self.max_length = max(
            64, int(max_length or Config.entity_reranker_max_length)
        )
        self.gpu_min_free_mb = max(
            0,
            int(
                gpu_min_free_mb
                if gpu_min_free_mb is not None
                else Config.entity_reranker_gpu_min_free_mb
            ),
        )
        self._lock = threading.RLock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device = "unloaded"
        self._disabled_reason = ""
        self._loaded_at = ""
        self._request_count = 0
        self._document_count = 0

    @property
    def available(self) -> bool:
        return self.model_path.is_dir() and not self._disabled_reason

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "protocol_version": PROTOCOL_VERSION,
                "status": "ok" if self.available else "degraded",
                "ready": self.available,
                "model": self.model_name,
                "model_loaded": self.loaded,
                "device": self._device,
                "loaded_at": self._loaded_at,
                "request_count": self._request_count,
                "document_count": self._document_count,
                "error": (
                    self._disabled_reason
                    or (
                        "configured model directory does not exist"
                        if not self.model_path.is_dir()
                        else ""
                    )
                ),
            }

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path.is_dir():
                self._disabled_reason = (
                    "configured model directory does not exist"
                )
                raise RerankerServiceError(self._disabled_reason)
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                selected_device = "cpu"
                if (
                    self.device_preference in {"auto", "cuda"}
                    and torch.cuda.is_available()
                ):
                    free_bytes, _ = torch.cuda.mem_get_info(0)
                    free_mb = int(free_bytes / 1024 / 1024)
                    if free_mb >= self.gpu_min_free_mb:
                        selected_device = "cuda:0"
                    elif self.device_preference == "cuda":
                        LOGGER.warning(
                            "reranker 显存不足（%s MiB），回退 CPU", free_mb,
                        )
                tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path), local_files_only=True,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    dtype=(
                        torch.float16
                        if selected_device.startswith("cuda")
                        else torch.float32
                    ),
                )
                model.eval().to(selected_device)
                self._torch = torch
                self._tokenizer = tokenizer
                self._model = model
                self._device = selected_device
                self._loaded_at = datetime.now(timezone.utc).isoformat()
                self._disabled_reason = ""
                LOGGER.info(
                    "reranker 模型已加载: %s (%s)",
                    self.model_name,
                    selected_device,
                )
            except Exception as exc:
                self._disabled_reason = str(exc)
                LOGGER.exception("reranker 模型加载失败: %s", exc)
                raise RerankerServiceError(
                    "reranker 模型加载失败"
                ) from exc

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.load()
        with self._lock:
            assert self._torch is not None
            try:
                pairs = [[query, document] for document in documents]
                scores: list[float] = []
                for start in range(0, len(pairs), self.batch_size):
                    batch = pairs[start : start + self.batch_size]
                    inputs = self._tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self._device)
                    with self._torch.inference_mode():
                        logits = self._model(
                            **inputs, return_dict=True,
                        ).logits
                    if logits.numel() != len(batch):
                        raise RerankerServiceError(
                            "模型输出数量与输入文档数量不一致"
                        )
                    values = self._torch.sigmoid(
                        logits.reshape(-1).float()
                    ).cpu().tolist()
                    scores.extend(float(value) for value in values)
                if (
                    len(scores) != len(documents)
                    or any(
                        not math.isfinite(value)
                        or value < 0.0
                        or value > 1.0
                        for value in scores
                    )
                ):
                    raise RerankerServiceError("模型返回了非法重排分数")
                self._request_count += 1
                self._document_count += len(documents)
                return scores
            except self._torch.cuda.OutOfMemoryError as exc:
                self._torch.cuda.empty_cache()
                LOGGER.warning("reranker GPU OOM: %s", exc)
                raise RerankerServiceError("reranker GPU 显存不足") from exc


class RerankerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        runtime: CrossEncoderRuntime,
        settings: RerankerServiceSettings,
    ) -> None:
        self.runtime = runtime
        self.settings = settings
        self.pending_slots = threading.BoundedSemaphore(
            settings.max_pending_requests
        )
        super().__init__(server_address, handler_class)


class RerankerRequestHandler(BaseHTTPRequestHandler):
    server: RerankerHTTPServer
    server_version = "KnowledgeForgeReranker/1"

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("reranker HTTP %s - %s", self.address_string(), format % args)

    def _json_response(
        self, status: int, payload: dict[str, Any], *, request_id: str = "",
    ) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if request_id:
            self.send_header("X-Request-ID", request_id)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(
        self, status: int, code: str, message: str, *, request_id: str,
    ) -> None:
        self._json_response(
            status,
            {
                "protocol_version": PROTOCOL_VERSION,
                "error": {"code": code, "message": message},
                "request_id": request_id,
            },
            request_id=request_id,
        )

    def _authorized(self) -> bool:
        expected = self.server.settings.api_key
        if not expected:
            return True
        authorization = str(self.headers.get("Authorization") or "")
        supplied = ""
        if authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not supplied:
            supplied = str(self.headers.get("X-API-Key") or "").strip()
        return bool(supplied) and hmac.compare_digest(supplied, expected)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        request_id = uuid4().hex
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            diagnostics = self.server.runtime.diagnostics()
            status = (
                HTTPStatus.OK
                if diagnostics["ready"]
                else HTTPStatus.SERVICE_UNAVAILABLE
            )
            self._json_response(
                status, diagnostics, request_id=request_id,
            )
            return
        if path == "/v1/models":
            if not self._authorized():
                self._error(
                    HTTPStatus.UNAUTHORIZED,
                    "unauthorized",
                    "缺少有效的 reranker 访问凭据",
                    request_id=request_id,
                )
                return
            self._json_response(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [{
                        "id": self.server.settings.model_name,
                        "object": "model",
                    }],
                },
                request_id=request_id,
            )
            return
        self._error(
            HTTPStatus.NOT_FOUND,
            "not_found",
            "接口不存在",
            request_id=request_id,
        )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        request_id = uuid4().hex
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in {"/api/rerank", "/v1/rerank"}:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "接口不存在",
                request_id=request_id,
            )
            return
        if not self._authorized():
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "unauthorized",
                "缺少有效的 reranker 访问凭据",
                request_id=request_id,
            )
            return
        try:
            payload = self._read_payload()
            query, documents, top_n = self._validate_payload(payload)
        except RerankerServiceError as exc:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_request",
                str(exc),
                request_id=request_id,
            )
            return

        acquired = self.server.pending_slots.acquire(
            timeout=self.server.settings.queue_timeout
        )
        if not acquired:
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "queue_timeout",
                "reranker 请求队列繁忙，请稍后重试",
                request_id=request_id,
            )
            return
        started = time.monotonic()
        try:
            scores = self.server.runtime.score(query, documents)
            ranked = sorted(
                (
                    {"index": index, "relevance_score": score}
                    for index, score in enumerate(scores)
                ),
                key=lambda item: (
                    -float(item["relevance_score"]), int(item["index"])
                ),
            )[:top_n]
            self._json_response(
                HTTPStatus.OK,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": request_id,
                    "model": self.server.settings.model_name,
                    "results": ranked,
                    "usage": {"documents": len(documents)},
                    "device": self.server.runtime.device,
                    "latency_ms": int(
                        (time.monotonic() - started) * 1000
                    ),
                },
                request_id=request_id,
            )
        except RerankerServiceError as exc:
            LOGGER.warning("reranker 请求失败 [%s]: %s", request_id, exc)
            self._error(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "inference_unavailable",
                str(exc),
                request_id=request_id,
            )
        except Exception:
            LOGGER.exception("reranker 未处理异常 [%s]", request_id)
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "reranker 内部错误",
                request_id=request_id,
            )
        finally:
            self.server.pending_slots.release()

    def _read_payload(self) -> dict[str, Any]:
        raw_length = str(self.headers.get("Content-Length") or "")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise RerankerServiceError("Content-Length 无效") from exc
        if length <= 0:
            raise RerankerServiceError("请求体不能为空")
        if length > self.server.settings.max_body_bytes:
            raise RerankerServiceError("请求体超过服务上限")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RerankerServiceError("请求体不是有效 JSON") from exc
        if not isinstance(payload, dict):
            raise RerankerServiceError("请求体必须是 JSON object")
        return payload

    def _validate_payload(
        self, payload: dict[str, Any],
    ) -> tuple[str, list[str], int]:
        requested_model = str(
            payload.get("model") or self.server.settings.model_name
        ).strip()
        if requested_model != self.server.settings.model_name:
            raise RerankerServiceError(
                f"模型不可用：{requested_model}"
            )
        query = payload.get("query")
        documents = payload.get("documents")
        if not isinstance(query, str) or not query.strip():
            raise RerankerServiceError("query 必须是非空字符串")
        query = query.strip()
        if len(query) > self.server.settings.max_text_chars:
            raise RerankerServiceError("query 超过字符上限")
        if not isinstance(documents, list) or not documents:
            raise RerankerServiceError("documents 必须是非空字符串数组")
        if len(documents) > self.server.settings.max_documents:
            raise RerankerServiceError("documents 数量超过服务上限")
        normalized: list[str] = []
        for document in documents:
            if not isinstance(document, str) or not document.strip():
                raise RerankerServiceError(
                    "documents 只能包含非空字符串"
                )
            value = document.strip()
            if len(value) > self.server.settings.max_text_chars:
                raise RerankerServiceError("单个 document 超过字符上限")
            normalized.append(value)
        if len(query) + sum(map(len, normalized)) > (
            self.server.settings.max_total_chars
        ):
            raise RerankerServiceError("请求文本总量超过服务上限")
        top_n = payload.get("top_n", len(normalized))
        if (
            isinstance(top_n, bool)
            or not isinstance(top_n, int)
            or not 1 <= top_n <= len(normalized)
        ):
            raise RerankerServiceError(
                "top_n 必须在 1 和 documents 数量之间"
            )
        return query, normalized, top_n


def create_server(
    *,
    settings: RerankerServiceSettings | None = None,
    runtime: CrossEncoderRuntime | None = None,
) -> RerankerHTTPServer:
    settings = settings or RerankerServiceSettings.from_config()
    settings.validate()
    runtime = runtime or CrossEncoderRuntime(
        model_name=settings.model_name,
    )
    return RerankerHTTPServer(
        (settings.host, settings.port),
        RerankerRequestHandler,
        runtime=runtime,
        settings=settings,
    )


def serve() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = RerankerServiceSettings.from_config()
    server = create_server(settings=settings)
    if Config.reranker_preload:
        server.runtime.load()

    def stop(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOGGER.info(
        "reranker HTTP 服务监听 %s:%s，模型 %s",
        settings.host,
        settings.port,
        settings.model_name,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
