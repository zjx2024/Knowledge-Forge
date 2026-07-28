"""Ollama clients for embeddings and schema-constrained JSON extraction."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Protocol

import requests

from .config import Config
from .observability import current_metrics, increment


LOGGER = logging.getLogger(__name__)


class OllamaBudgetError(ValueError):
    """The rendered request cannot fit the verified application budget."""


class OllamaProtocolError(RuntimeError):
    """Ollama returned an incomplete, truncated, or diagnostically unsafe result."""


class TokenEstimator(Protocol):
    name: str

    def estimate(self, text: str) -> int: ...


class ConservativeTokenEstimator:
    """Replaceable tokenizer-free fallback for mixed CJK and ASCII text."""

    name = "conservative-cjk-ascii-v2"

    def estimate(self, text: str) -> int:
        cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
        # JSON schemas and mixed technical text tokenize much more densely than
        # natural English under Qwen. Two non-CJK characters per token keeps
        # request admission conservative without changing the Ollama runner.
        return max(1, cjk + math.ceil(max(0, len(text) - cjk) / 2))


@dataclass(frozen=True)
class EffectiveModelContext:
    model: str
    model_context_tokens: int
    effective_context_tokens: int
    digest: str
    source: str


@dataclass(frozen=True)
class OllamaCallDiagnostics:
    model: str
    digest: str
    context_source: str
    effective_context_tokens: int
    context_budget_tokens: int
    max_output_tokens: int
    safety_margin_tokens: int
    estimated_input_tokens: int
    prompt_eval_count: int
    output_eval_count: int
    rendered_message_hash: str
    estimator: str


class OllamaClient:
    def __init__(
        self,
        base_url: str = "",
        timeout: int | None = None,
        *,
        token_estimator: TokenEstimator | None = None,
    ):
        self.base_url = (base_url or Config.ollama_base_url).rstrip("/")
        self.timeout = timeout or Config.ollama_timeout
        self.token_estimator = token_estimator or ConservativeTokenEstimator()
        self.last_call_diagnostics: OllamaCallDiagnostics | None = None
        self.last_raw_response_text = ""
        self.last_parsed_json_response: dict[str, Any] | list[Any] | None = None
        self._context_cache: dict[str, tuple[float, EffectiveModelContext]] = {}

    def models(self) -> list[dict[str, Any]]:
        increment("ollama_logical_calls")
        response = self._request("get", f"{self.base_url}/api/tags", timeout=min(self.timeout, 30))
        response.raise_for_status()
        return response.json().get("models") or []

    def model_info(self, model: str) -> dict[str, Any]:
        increment("ollama_logical_calls")
        response = self._request(
            "post",
            f"{self.base_url}/api/show",
            json={"model": model},
            timeout=min(self.timeout, 30),
        )
        response.raise_for_status()
        payload = response.json()
        context_lengths = [
            int(value)
            for key, value in (payload.get("model_info") or {}).items()
            if str(key).endswith(".context_length") and str(value).isdigit()
        ]
        return {
            "context_length": max(context_lengths, default=8192),
            "digest": str(payload.get("digest") or ""),
            "capabilities": payload.get("capabilities") or [],
            "details": payload.get("details") or {},
        }

    def effective_model_context(self, model: str) -> EffectiveModelContext:
        """Return the runner context when loaded, otherwise an explicit safe fallback."""

        now = time.monotonic()
        cached = self._context_cache.get(model)
        if cached and cached[0] > now:
            return cached[1]
        info = self.model_info(model)
        model_context = max(1, int(info.get("context_length") or 0))
        digest = str(info.get("digest") or "")
        effective = min(model_context, max(1, Config.ollama_effective_context_fallback))
        source = "configured_fallback"
        try:
            increment("ollama_logical_calls")
            response = self._request(
                "get", f"{self.base_url}/api/ps", timeout=min(self.timeout, 30)
            )
            response.raise_for_status()
            for runner in response.json().get("models") or []:
                runner_name = str(runner.get("name") or runner.get("model") or "")
                if not self._same_model(model, runner_name):
                    continue
                runner_context = int(runner.get("context_length") or 0)
                if runner_context > 0:
                    effective = min(model_context, runner_context)
                    source = "runner"
                digest = str(runner.get("digest") or digest)
                break
        except Exception as exc:
            LOGGER.warning(
                "Ollama runner context unavailable; using configured fallback model=%s error=%s",
                model,
                type(exc).__name__,
            )
        result = EffectiveModelContext(
            model=model,
            model_context_tokens=model_context,
            effective_context_tokens=effective,
            digest=digest,
            source=source,
        )
        self._context_cache[model] = (
            now + max(0, Config.ollama_context_cache_seconds), result,
        )
        return result

    def embed(self, texts: Iterable[str], model: str) -> list[list[float]]:
        values = [str(item) for item in texts]
        if not values:
            return []
        increment("ollama_logical_calls")
        increment("embedding_input_count", len(values))
        response = self._request(
            "post",
            f"{self.base_url}/api/embed",
            json={"model": model, "input": values, "keep_alive": Config.ollama_keep_alive},
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return [self._legacy_embedding(text, model) for text in values]
        response.raise_for_status()
        embeddings = response.json().get("embeddings") or []
        if len(embeddings) != len(values):
            raise RuntimeError(f"Ollama embedding count mismatch: expected {len(values)}, got {len(embeddings)}")
        return [[float(value) for value in vector] for vector in embeddings]

    def _legacy_embedding(self, text: str, model: str) -> list[float]:
        response = self._request(
            "post",
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text, "keep_alive": Config.ollama_keep_alive},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return [float(value) for value in response.json().get("embedding") or []]

    def json_chat(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.1,
        context_budget_tokens: int | None = None,
        max_output_tokens: int | None = None,
        safety_margin_tokens: int | None = None,
        images: Iterable[bytes] | None = None,
        ) -> dict[str, Any]:
        increment("ollama_logical_calls")
        self.last_raw_response_text = ""
        self.last_parsed_json_response = None
        self.last_call_diagnostics = None
        context_budget = int(
            context_budget_tokens or Config.ollama_default_context_budget
        )
        output_budget = int(max_output_tokens or Config.ollama_default_max_output)
        safety_margin = int(
            Config.ollama_default_safety_margin
            if safety_margin_tokens is None else safety_margin_tokens
        )
        if min(context_budget, output_budget) <= 0 or safety_margin < 0:
            raise OllamaBudgetError("Ollama token budgets must be positive")
        runtime = self.effective_model_context(model)
        verified_context = min(context_budget, runtime.effective_context_tokens)
        user_message: dict[str, Any] = {"role": "user", "content": prompt}
        encoded_images = [base64.b64encode(value).decode("ascii") for value in (images or [])]
        if encoded_images:
            user_message["images"] = encoded_images
        messages = [{"role": "system", "content": system}, user_message]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": Config.ollama_keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": output_budget,
            },
            "format": schema or "json",
        }
        parse_error: Exception | None = None
        for attempt in range(2):
            if attempt:
                increment("http_retry_count")
            estimate, rendered_hash = self._admit_request(
                body,
                verified_context_tokens=verified_context,
                max_output_tokens=output_budget,
                safety_margin_tokens=safety_margin,
                image_count=len(encoded_images),
            )
            increment("estimated_prompt_tokens", estimate)
            response = self._request(
                "post", f"{self.base_url}/api/chat", json=body, timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
            self._validate_response_protocol(payload, estimate)
            prompt_eval_count = int(payload.get("prompt_eval_count") or 0)
            self.last_call_diagnostics = OllamaCallDiagnostics(
                model=model,
                digest=runtime.digest,
                context_source=runtime.source,
                effective_context_tokens=runtime.effective_context_tokens,
                context_budget_tokens=context_budget,
                max_output_tokens=output_budget,
                safety_margin_tokens=safety_margin,
                estimated_input_tokens=estimate,
                prompt_eval_count=prompt_eval_count,
                output_eval_count=int(payload.get("eval_count") or 0),
                rendered_message_hash=rendered_hash,
                estimator=self.token_estimator.name,
            )
            LOGGER.info("Ollama request diagnostics: %s", json.dumps(
                asdict(self.last_call_diagnostics), ensure_ascii=True, sort_keys=True,
            ))
            content = str((payload.get("message") or {}).get("content") or "").strip()
            self.last_raw_response_text = content
            try:
                parsed = self._parse_json(content)
                self.last_parsed_json_response = parsed
                return parsed
            except (RuntimeError, json.JSONDecodeError) as exc:
                parse_error = exc
                if attempt:
                    break
                increment("json_repair_count")
                if content:
                    messages.append({"role": "assistant", "content": content[:12000]})
                messages.append({
                    "role": "user",
                    "content": "The previous response was not one valid JSON value. Return only the JSON object required by the schema, without analysis, markdown, or a second object.",
                })
                body["options"] = {
                    "temperature": 0,
                    "num_predict": output_budget,
                }
        raise RuntimeError(f"Ollama JSON response invalid after retry: {parse_error}")

    @staticmethod
    def _request(method: str, url: str, **kwargs: Any):
        """Count the actual HTTP boundary, including repair and legacy calls."""

        increment("ollama_http_calls")
        payload = kwargs.get("json")
        if payload is not None:
            try:
                increment("candidate_payload_bytes", len(json.dumps(payload, ensure_ascii=False).encode("utf-8")))
            except (TypeError, ValueError):
                pass
        started = time.perf_counter()
        try:
            return getattr(requests, method)(url, **kwargs)
        finally:
            metrics = current_metrics()
            if metrics is not None:
                metrics.duration("ollama_http", (time.perf_counter() - started) * 1000)

    def _admit_request(
        self,
        body: dict[str, Any],
        *,
        verified_context_tokens: int,
        max_output_tokens: int,
        safety_margin_tokens: int,
        image_count: int,
    ) -> tuple[int, str]:
        rendered = json.dumps({
            "messages": body.get("messages") or [],
            "format": body.get("format") or "json",
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        estimated = self.token_estimator.estimate(rendered)
        estimated += max(0, image_count) * max(0, Config.ollama_image_token_estimate)
        required = estimated + max_output_tokens + safety_margin_tokens
        if required > verified_context_tokens:
            raise OllamaBudgetError(
                "Ollama token budget exceeded before HTTP: "
                f"estimated_input={estimated}, max_output={max_output_tokens}, "
                f"safety_margin={safety_margin_tokens}, effective_context={verified_context_tokens}"
            )
        return estimated, hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_response_protocol(payload: dict[str, Any], estimated: int) -> None:
        if payload.get("done") is not True:
            raise OllamaProtocolError("Ollama response is incomplete")
        if str(payload.get("done_reason") or "").lower() in {"length", "max_tokens"}:
            raise OllamaProtocolError("Ollama response was truncated by output length")
        actual = int(payload.get("prompt_eval_count") or 0)
        if actual <= 0:
            raise OllamaProtocolError("Ollama response omitted prompt_eval_count")
        allowed = max(
            Config.ollama_token_variance_absolute,
            math.ceil(estimated * Config.ollama_token_variance_ratio),
        )
        if actual > estimated + allowed:
            raise OllamaProtocolError(
                "Ollama prompt token estimate underestimated actual usage: "
                f"estimated={estimated}, actual={actual}, allowed_variance={allowed}"
            )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Compatibility helper; request admission uses the replaceable estimator."""
        return ConservativeTokenEstimator().estimate(text)

    @staticmethod
    def _same_model(requested: str, runner: str) -> bool:
        def canonical(value: str) -> str:
            value = value.strip()
            return value if ":" in value else f"{value}:latest"

        return canonical(requested) == canonical(runner)

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            value = None
            for match in re.finditer(r"[\[{]", cleaned):
                try:
                    candidate, _ = decoder.raw_decode(cleaned[match.start():])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, (dict, list)):
                    value = candidate
                    break
            if value is None:
                raise RuntimeError("Ollama did not return a JSON object")
        if isinstance(value, list):
            value = {"items": value}
        if not isinstance(value, dict):
            raise RuntimeError("Ollama JSON response must be an object or array")
        return value
