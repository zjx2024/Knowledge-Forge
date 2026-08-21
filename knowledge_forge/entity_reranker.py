"""HTTP client for the dedicated entity cross-encoder reranker."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

from .config import Config
from .entity_identity import entity_identity_text


LOGGER = logging.getLogger(__name__)
HTTP_OK = 200


class RerankerHTTPError(RuntimeError):
    def __init__(self, message: str, status_code: int = 0) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


class EntityReranker:
    """Use remote relevance as one conservative entity-resolution signal.

    The HTTP service, rather than the Knowledge Forge process, owns
    Transformers, torch and model memory. A failed or malformed request is
    represented by zero scores and last_score_used_model=False so existing
    rule/vector/LLM fallback behavior remains intact.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        connect_timeout: int | None = None,
        timeout: int | None = None,
        retries: int | None = None,
        session_factory: Callable[[], requests.Session] | None = None,
    ) -> None:
        self.base_url = str(
            base_url if base_url is not None else Config.entity_reranker_url
        ).strip().rstrip("/")
        self.model = str(
            model if model is not None else Config.entity_reranker_model
        ).strip()
        self.api_key = str(
            api_key
            if api_key is not None
            else Config.entity_reranker_api_key
        ).strip()
        self.connect_timeout = max(
            1,
            int(
                connect_timeout
                if connect_timeout is not None
                else Config.entity_reranker_connect_timeout
            ),
        )
        self.timeout = max(
            self.connect_timeout,
            int(
                timeout
                if timeout is not None
                else Config.entity_reranker_timeout
            ),
        )
        self.retries = max(
            0,
            int(
                retries
                if retries is not None
                else Config.entity_reranker_retries
            ),
        )
        self._session_factory = session_factory or requests.Session
        self._thread_state = threading.local()
        self._state_lock = threading.RLock()
        self._last_fallback_reason = ""
        self._last_http_status = 0
        self._last_latency_ms = 0
        self._last_server_loaded = False
        self._last_server_device = ""
        self._last_success_at = 0.0
        self._disabled_reason = self._configuration_error()

    def _configuration_error(self) -> str:
        if not Config.entity_reranker_enabled:
            return "disabled by configuration"
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "ENTITY_RERANKER_URL is invalid"
        if not self.model:
            return "ENTITY_RERANKER_MODEL is empty"
        return ""

    @property
    def available(self) -> bool:
        return not self._disabled_reason

    @property
    def last_score_used_model(self) -> bool:
        return bool(
            getattr(self._thread_state, "last_score_used_model", False)
        )

    def _set_thread_result(self, used_model: bool) -> None:
        self._thread_state.last_score_used_model = bool(used_model)

    def _session(self) -> requests.Session:
        session = getattr(self._thread_state, "session", None)
        if session is None:
            session = self._session_factory()
            self._thread_state.session = session
        return session

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def diagnostics(self) -> dict[str, Any]:
        enabled = bool(Config.entity_reranker_enabled)
        with self._state_lock:
            return {
                "reranker_enabled": enabled,
                "reranker_transport": "http",
                "reranker_endpoint": self.base_url,
                "reranker_model": self.model,
                "reranker_loaded": self._last_server_loaded,
                "last_score_used_model": self.last_score_used_model,
                "reranker_device": self._last_server_device or "remote",
                "last_http_status": self._last_http_status,
                "last_latency_ms": self._last_latency_ms,
                "last_success_at": self._last_success_at,
                "fallback_reason": (
                    self._disabled_reason or self._last_fallback_reason
                ),
            }

    def probe(self) -> dict[str, Any]:
        """Read service readiness without loading the model in this process."""

        if not self.available:
            return {
                "status": "unavailable",
                "ready": False,
                "error": self._disabled_reason,
            }
        try:
            response = self._session().get(
                f"{self.base_url}/health",
                headers={"Accept": "application/json"},
                timeout=(self.connect_timeout, min(self.timeout, 10)),
            )
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("health response is not an object")
            with self._state_lock:
                self._last_http_status = int(response.status_code)
                self._last_server_loaded = bool(
                    payload.get("model_loaded")
                )
                self._last_server_device = str(
                    payload.get("device") or ""
                )
            return payload
        except (requests.RequestException, ValueError) as exc:
            with self._state_lock:
                self._last_fallback_reason = str(exc)
            return {
                "status": "unavailable",
                "ready": False,
                "error": str(exc),
            }

    @staticmethod
    def _pair(raw: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
        mention = entity_identity_text(
            raw["name"],
            raw["type"],
            raw.get("description") or "",
            raw.get("aliases") or [],
        )
        canonical = entity_identity_text(
            candidate["name"],
            candidate["entity_type"],
            candidate.get("description") or "",
            candidate.get("aliases") or [],
        )
        return [mention, canonical]

    @staticmethod
    def _response_error(payload: Any) -> str:
        if not isinstance(payload, dict):
            return "reranker returned a non-object response"
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        return str(error or "reranker request failed")

    def _remote_scores(
        self, query: str, documents: list[str],
    ) -> tuple[list[float], dict[str, Any], int]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            # Entity resolution needs pair/score alignment for every candidate.
            "top_n": len(documents),
        }
        response: requests.Response | None = None
        last_error: Exception | None = None
        retryable_statuses = {429, 502, 503, 504}
        for attempt in range(self.retries + 1):
            try:
                response = self._session().post(
                    f"{self.base_url}/api/rerank",
                    headers=self._headers(),
                    json=payload,
                    timeout=(self.connect_timeout, self.timeout),
                )
                if (
                    response.status_code in retryable_statuses
                    and attempt < self.retries
                ):
                    time.sleep(min(1.0, 0.2 * (2**attempt)))
                    continue
                try:
                    body = response.json()
                except ValueError as exc:
                    raise ValueError(
                        "reranker returned invalid JSON"
                    ) from exc
                if response.status_code != HTTP_OK:
                    raise RuntimeError(self._response_error(body))
                if not isinstance(body, dict):
                    raise ValueError("reranker response is not an object")
                results = body.get("results")
                if not isinstance(results, list) or len(results) != len(documents):
                    raise ValueError(
                        "reranker result count does not match candidate count"
                    )
                aligned: list[float | None] = [None] * len(documents)
                for result in results:
                    if not isinstance(result, dict):
                        raise ValueError("reranker result is not an object")
                    index = result.get("index")
                    score = result.get("relevance_score")
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 0 <= index < len(documents)
                        or aligned[index] is not None
                    ):
                        raise ValueError("reranker returned invalid result indices")
                    try:
                        numeric = float(score)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "reranker returned a non-numeric score"
                        ) from exc
                    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                        raise ValueError("reranker returned an out-of-range score")
                    aligned[index] = numeric
                if any(value is None for value in aligned):
                    raise ValueError("reranker response omitted candidate scores")
                return [float(value) for value in aligned], body, response.status_code
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                last_error = exc
                if (
                    isinstance(exc, requests.RequestException)
                    and attempt < self.retries
                ):
                    time.sleep(min(1.0, 0.2 * (2**attempt)))
                    continue
                break
        status = int(response.status_code) if response is not None else 0
        raise RerankerHTTPError(
            str(last_error or "reranker request failed"), status,
        )

    def score(
        self, raw: dict[str, Any], candidates: list[dict[str, Any]],
    ) -> list[float]:
        self._set_thread_result(False)
        if not candidates:
            return []
        if not self.available:
            return [0.0] * len(candidates)
        mention, _ = self._pair(raw, candidates[0])
        documents = [
            self._pair(raw, candidate)[1] for candidate in candidates
        ]
        started = time.monotonic()
        try:
            scores, payload, status = self._remote_scores(mention, documents)
            self._set_thread_result(True)
            with self._state_lock:
                self._last_fallback_reason = ""
                self._last_http_status = int(status)
                self._last_latency_ms = int(
                    payload.get("latency_ms")
                    or (time.monotonic() - started) * 1000
                )
                self._last_server_loaded = True
                self._last_server_device = str(payload.get("device") or "")
                self._last_success_at = time.time()
            return scores
        except RerankerHTTPError as exc:
            with self._state_lock:
                self._last_fallback_reason = str(exc)
                self._last_http_status = exc.status_code
                self._last_latency_ms = int(
                    (time.monotonic() - started) * 1000
                )
            LOGGER.warning("实体重排服务不可用，本批次回退: %s", exc)
            return [0.0] * len(candidates)
