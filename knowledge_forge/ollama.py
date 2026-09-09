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
from typing import Any, Callable, Iterable, Protocol

import requests

from .config import Config
from .observability import current_metrics, increment
from .unicode_safety import sanitize_unicode_data, sanitize_unicode_text


LOGGER = logging.getLogger(__name__)


class OllamaBudgetError(ValueError):
    """The rendered request cannot fit the verified application budget."""


class OllamaProtocolError(RuntimeError):
    """Ollama returned an incomplete, truncated, or diagnostically unsafe result."""


class OllamaCancelled(RuntimeError):
    """A caller closed a streaming model request at a safe cancellation point."""


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
        embedding_base_url: str = "",
        token_estimator: TokenEstimator | None = None,
        llm_provider: str | None = None,
        embedding_provider: str | None = None,
        llm_api_base_url: str | None = None,
        embedding_api_base_url: str | None = None,
        llm_api_key: str | None = None,
        embedding_api_key: str | None = None,
        llm_api_models: Iterable[str] | None = None,
        embedding_api_models: Iterable[str] | None = None,
        llm_api_context_tokens: int | None = None,
        embedding_api_context_tokens: int | None = None,
    ):
        self.base_url = (base_url or Config.ollama_base_url).rstrip("/")
        self.embedding_base_url = (
            embedding_base_url or Config.ollama_embedding_base_url
        ).rstrip("/")
        self.timeout = timeout or Config.ollama_timeout
        self.token_estimator = token_estimator or ConservativeTokenEstimator()
        self.llm_provider = self._normalize_provider(
            llm_provider or Config.llm_provider
        )
        self.embedding_provider = self._normalize_provider(
            embedding_provider or Config.embedding_provider
        )
        self.llm_api_base_url = (
            Config.llm_api_base_url if llm_api_base_url is None else llm_api_base_url
        ).rstrip("/")
        self.embedding_api_base_url = (
            Config.embedding_api_base_url
            if embedding_api_base_url is None else embedding_api_base_url
        ).rstrip("/")
        self.llm_api_key = Config.llm_api_key if llm_api_key is None else llm_api_key
        self.embedding_api_key = (
            Config.embedding_api_key
            if embedding_api_key is None else embedding_api_key
        )
        self.llm_api_models = tuple(
            str(item).strip() for item in (
                Config.llm_api_models if llm_api_models is None else llm_api_models
            ) if str(item).strip()
        )
        self.embedding_api_models = tuple(
            str(item).strip() for item in (
                Config.embedding_api_models
                if embedding_api_models is None else embedding_api_models
            ) if str(item).strip()
        )
        self.llm_api_context_tokens = max(
            1024,
            int(
                Config.llm_api_context_tokens
                if llm_api_context_tokens is None else llm_api_context_tokens
            ),
        )
        self.embedding_api_context_tokens = max(
            1024,
            int(
                Config.embedding_api_context_tokens
                if embedding_api_context_tokens is None else embedding_api_context_tokens
            ),
        )
        self.last_call_diagnostics: OllamaCallDiagnostics | None = None
        self.last_raw_response_text = ""
        self.last_parsed_json_response: dict[str, Any] | list[Any] | None = None
        self._context_cache: dict[tuple[str, str], tuple[float, EffectiveModelContext]] = {}

    @staticmethod
    def _normalize_provider(value: str) -> str:
        normalized = str(value or "ollama").strip().lower().replace("-", "_")
        if normalized in {"openai", "openai_compat", "api", "remote"}:
            return "openai_compatible"
        if normalized not in {"ollama", "openai_compatible"}:
            raise ValueError(
                "模型提供方必须是 ollama 或 openai_compatible"
            )
        return normalized

    def provider_for(self, model_type: str = "llm") -> str:
        return self.llm_provider if model_type == "llm" else self.embedding_provider

    def base_url_for(self, model_type: str = "llm") -> str:
        if self.provider_for(model_type) == "openai_compatible":
            return (
                self.llm_api_base_url
                if model_type == "llm" else self.embedding_api_base_url
            )
        return self.base_url if model_type == "llm" else self.embedding_base_url

    def api_key_for(self, model_type: str = "llm") -> str:
        return self.llm_api_key if model_type == "llm" else self.embedding_api_key

    def api_models_for(self, model_type: str = "llm") -> tuple[str, ...]:
        configured = (
            self.llm_api_models if model_type == "llm"
            else self.embedding_api_models
        )
        if configured:
            return configured
        default = Config.default_llm_model if model_type == "llm" else Config.default_embedding_model
        return (default,)

    @staticmethod
    def _openai_base_url(value: str, *, model_type: str) -> str:
        base = str(value or "").strip().rstrip("/")
        if not base:
            raise RuntimeError(
                f"{('LLM' if model_type == 'llm' else 'Embedding')} API 地址未配置"
            )
        return base if base.endswith("/v1") else f"{base}/v1"

    def _api_headers(self, model_type: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.api_key_for(model_type).strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _infer_model_type(self, model: str) -> str:
        if model in self.embedding_api_models:
            return "embedding"
        if model in self.llm_api_models:
            return "llm"
        normalized = str(model).lower()
        return (
            "embedding"
            if any(token in normalized for token in ("embed", "embedding", "bge", "gte", "e5"))
            else "llm"
        )

    def models(self) -> list[dict[str, Any]]:
        increment("ollama_logical_calls")
        if self.llm_provider == "ollama" and self.embedding_provider == "ollama":
            response = self._request(
                "get", f"{self.base_url}/api/tags", timeout=min(self.timeout, 30)
            )
            response.raise_for_status()
            return response.json().get("models") or []
        discovered: list[dict[str, Any]] = []
        for model_type in ("llm", "embedding"):
            if self.provider_for(model_type) == "openai_compatible":
                capability = "completion" if model_type == "llm" else "embedding"
                discovered.extend({
                    "name": name,
                    "provider": "openai_compatible",
                    "model_type": model_type,
                    "capabilities": [capability],
                } for name in self.api_models_for(model_type))
                continue
            base = self.base_url if model_type == "llm" else self.embedding_base_url
            response = self._request(
                "get", f"{base}/api/tags", timeout=min(self.timeout, 30)
            )
            response.raise_for_status()
            for item in response.json().get("models") or []:
                discovered.append({**item, "provider": "ollama"})
        return discovered

    def model_info(self, model: str, model_type: str | None = None) -> dict[str, Any]:
        increment("ollama_logical_calls")
        resolved_type = model_type or self._infer_model_type(model)
        if self.provider_for(resolved_type) == "openai_compatible":
            configured = (
                self.llm_api_context_tokens
                if resolved_type == "llm" else self.embedding_api_context_tokens
            )
            base = self._openai_base_url(
                self.base_url_for(resolved_type), model_type=resolved_type
            )
            digest = hashlib.sha256(
                f"openai-compatible|{base}|{resolved_type}|{model}".encode("utf-8")
            ).hexdigest()
            return {
                "context_length": configured,
                "digest": digest,
                "capabilities": [
                    "completion" if resolved_type == "llm" else "embedding"
                ],
                "details": {"provider": "openai_compatible", "base_url": base},
            }
        base = self.base_url if resolved_type == "llm" else self.embedding_base_url
        response = self._request(
            "post",
            f"{base}/api/show",
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

    def effective_model_context(
        self, model: str, model_type: str | None = None,
    ) -> EffectiveModelContext:
        """Return the runner context when loaded, otherwise an explicit safe fallback."""

        resolved_type = model_type or self._infer_model_type(model)
        now = time.monotonic()
        cache_key = (model, resolved_type)
        cached = self._context_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]
        info = self.model_info(model, resolved_type)
        model_context = max(1, int(info.get("context_length") or 0))
        digest = str(info.get("digest") or "")
        if self.provider_for(resolved_type) == "openai_compatible":
            result = EffectiveModelContext(
                model=model,
                model_context_tokens=model_context,
                effective_context_tokens=model_context,
                digest=digest,
                source="configured_api",
            )
            self._context_cache[cache_key] = (
                now + max(0, Config.ollama_context_cache_seconds), result
            )
            return result
        effective = min(model_context, max(1, Config.ollama_effective_context_fallback))
        source = "configured_fallback"
        base = self.base_url if resolved_type == "llm" else self.embedding_base_url
        try:
            increment("ollama_logical_calls")
            response = self._request(
                "get", f"{base}/api/ps", timeout=min(self.timeout, 30)
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
        self._context_cache[cache_key] = (
            now + max(0, Config.ollama_context_cache_seconds), result,
        )
        return result

    def embed(self, texts: Iterable[str], model: str) -> list[list[float]]:
        values = [sanitize_unicode_text(str(item)) for item in texts]
        if not values:
            return []
        increment("ollama_logical_calls")
        increment("embedding_input_count", len(values))
        if self.embedding_provider == "openai_compatible":
            return self._openai_embed(values, model)
        response = self._request(
            "post",
            f"{self.embedding_base_url}/api/embed",
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

    def _openai_embed(self, values: list[str], model: str) -> list[list[float]]:
        base = self._openai_base_url(
            self.embedding_api_base_url, model_type="embedding"
        )
        response = self._request(
            "post",
            f"{base}/embeddings",
            headers=self._api_headers("embedding"),
            json={"model": model, "input": values},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(values):
            raise RuntimeError(
                "OpenAI-compatible embedding count mismatch: "
                f"expected {len(values)}, got {len(data) if isinstance(data, list) else 0}"
            )
        ordered: list[list[float] | None] = [None] * len(values)
        for fallback_index, item in enumerate(data):
            if not isinstance(item, dict):
                raise RuntimeError("OpenAI-compatible embedding item is not an object")
            index = item.get("index", fallback_index)
            try:
                index = int(index)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("OpenAI-compatible embedding index is invalid") from exc
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise RuntimeError("OpenAI-compatible embedding vector is empty")
            if not 0 <= index < len(values) or ordered[index] is not None:
                raise RuntimeError("OpenAI-compatible embedding indexes are invalid")
            ordered[index] = [float(value) for value in vector]
        if any(vector is None for vector in ordered):
            raise RuntimeError("OpenAI-compatible embedding response is incomplete")
        return [vector for vector in ordered if vector is not None]

    @staticmethod
    def _openai_message_content(message: Any) -> str:
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            parts: list[str] = []
            for item in message:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                    parts.append(str(item.get("text") or ""))
            return "".join(parts)
        return ""

    def _json_chat_openai(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None,
        temperature: float,
        context_budget_tokens: int | None,
        max_output_tokens: int | None,
        safety_margin_tokens: int | None,
        images: Iterable[bytes] | None,
        cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        """Call an OpenAI-compatible endpoint using the same JSON contract as Ollama.

        The provider adapter intentionally lives behind OllamaClient so the
        extraction, reports and agent code retain one transport contract.  An
        API's ``response_format`` guarantees valid JSON only; the schema is
        therefore also included in the instruction and the existing parser
        still performs the final required-field gate.
        """

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
            raise OllamaBudgetError("API token budgets must be positive")
        runtime = self.effective_model_context(model, "llm")
        verified_context = min(context_budget, runtime.effective_context_tokens)
        encoded_images = [
            base64.b64encode(value).decode("ascii") for value in (images or [])
        ]
        system_content = sanitize_unicode_text(system)
        if schema is not None:
            system_content += (
                "\n\nReturn exactly one JSON object matching this schema. "
                "Do not include markdown or explanations:\n"
                + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            )
        user_content: str | list[dict[str, Any]] = sanitize_unicode_text(prompt)
        if encoded_images:
            user_content = [{"type": "text", "text": user_content}]
            user_content.extend({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{value}"},
            } for value in encoded_images)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": bool(cancel_check),
            "temperature": temperature,
            "max_tokens": output_budget,
            "seed": Config.ollama_seed,
            "response_format": {"type": "json_object"},
        }
        if cancel_check:
            body["stream_options"] = {"include_usage": True}
        parse_error: Exception | None = None
        last_parsed_candidate: dict[str, Any] | None = None
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
            if cancel_check and cancel_check():
                raise OllamaCancelled("API 请求已由调用方终止")
            base = self._openai_base_url(
                self.llm_api_base_url, model_type="llm"
            )
            response = self._request(
                "post", f"{base}/chat/completions",
                headers=self._api_headers("llm"),
                json=body, timeout=self.timeout,
                **({"stream": True} if cancel_check else {}),
            )
            response.raise_for_status()
            payload = (
                self._consume_openai_streaming(response, cancel_check)
                if cancel_check else response.json()
            )
            self._validate_openai_response_protocol(payload)
            usage = payload.get("usage") if isinstance(payload, dict) else {}
            usage = usage if isinstance(usage, dict) else {}
            prompt_eval_count = int(usage.get("prompt_tokens") or 0)
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
                output_eval_count=int(
                    usage.get("completion_tokens")
                    or usage.get("output_tokens") or 0
                ),
                rendered_message_hash=rendered_hash,
                estimator=self.token_estimator.name,
            )
            choices = payload.get("choices") if isinstance(payload, dict) else []
            choice = choices[0] if isinstance(choices, list) and choices else {}
            message = choice.get("message") if isinstance(choice, dict) else {}
            content = sanitize_unicode_text(
                self._openai_message_content(
                    message.get("content") if isinstance(message, dict) else ""
                )
            ).strip()
            self.last_raw_response_text = content
            try:
                parsed = self._parse_json(content, schema=schema)
                last_parsed_candidate = parsed
                self._validate_schema_shape(parsed, schema)
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
                    "content": (
                        "The previous response did not satisfy the required JSON "
                        "object contract. Return only one JSON object containing "
                        "every required schema field, with no analysis, markdown, "
                        "YAML, or second object."
                    ),
                })
        if (
            last_parsed_candidate is not None
            and self._schema_has_known_property(last_parsed_candidate, schema)
        ):
            self.last_parsed_json_response = last_parsed_candidate
            return last_parsed_candidate
        raise RuntimeError(f"OpenAI-compatible JSON response invalid after retry: {parse_error}")

    @staticmethod
    def _consume_openai_streaming(
        response: Any, cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        fragments: list[str] = []
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if cancel_check and cancel_check():
                    response.close()
                    raise OllamaCancelled("API 请求已由调用方终止")
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    item = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise OllamaProtocolError(
                        "OpenAI-compatible 流式响应包含非法 JSON"
                    ) from exc
                if not isinstance(item, dict):
                    raise OllamaProtocolError("OpenAI-compatible 流式响应不是对象")
                item_usage = item.get("usage")
                if isinstance(item_usage, dict):
                    usage = item_usage
                choice = (item.get("choices") or [{}])[0] or {}
                delta = choice.get("delta") or {}
                content = OllamaClient._openai_message_content(delta.get("content"))
                if content:
                    fragments.append(content)
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
        finally:
            response.close()
        return {
            "choices": [{
                "message": {"content": "".join(fragments)},
                "finish_reason": finish_reason or "stop",
            }],
            "usage": usage,
        }

    @staticmethod
    def _validate_openai_response_protocol(payload: dict[str, Any]) -> None:
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise OllamaProtocolError("OpenAI-compatible response has no choices")
        choice = choices[0] if isinstance(choices[0], dict) else {}
        if str(choice.get("finish_reason") or "").lower() in {"length", "max_tokens"}:
            raise OllamaProtocolError(
                "OpenAI-compatible response was truncated by output length"
            )
        content = OllamaClient._openai_message_content(
            (choice.get("message") or {}).get("content")
        )
        if not content.strip():
            raise OllamaProtocolError("OpenAI-compatible response has empty content")

    def _legacy_embedding(self, text: str, model: str) -> list[float]:
        response = self._request(
            "post",
            f"{self.embedding_base_url}/api/embeddings",
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
        cancel_check: Callable[[], bool] | None = None,
        ) -> dict[str, Any]:
        if self.llm_provider == "openai_compatible":
            return self._json_chat_openai(
                model=model,
                system=system,
                prompt=prompt,
                schema=schema,
                temperature=temperature,
                context_budget_tokens=context_budget_tokens,
                max_output_tokens=max_output_tokens,
                safety_margin_tokens=safety_margin_tokens,
                images=images,
                cancel_check=cancel_check,
            )
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
        user_message: dict[str, Any] = {
            "role": "user", "content": sanitize_unicode_text(prompt),
        }
        encoded_images = [base64.b64encode(value).decode("ascii") for value in (images or [])]
        if encoded_images:
            user_message["images"] = encoded_images
        messages = [{
            "role": "system", "content": sanitize_unicode_text(system),
        }, user_message]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": bool(cancel_check),
            "think": False,
            "keep_alive": Config.ollama_keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": output_budget,
                "seed": Config.ollama_seed,
            },
            "format": schema or "json",
        }
        parse_error: Exception | None = None
        last_parsed_candidate: dict[str, Any] | None = None
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
            if cancel_check and cancel_check():
                raise OllamaCancelled("Ollama 请求已由调用方终止")
            response = self._request(
                "post", f"{self.base_url}/api/chat", json=body,
                timeout=self.timeout, **({"stream": True} if cancel_check else {}),
            )
            response.raise_for_status()
            payload = (
                self._consume_streaming_chat(response, cancel_check)
                if cancel_check else response.json()
            )
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
            content = sanitize_unicode_text(
                str((payload.get("message") or {}).get("content") or "")
            ).strip()
            self.last_raw_response_text = content
            try:
                parsed = self._parse_json(content, schema=schema)
                last_parsed_candidate = parsed
                self._validate_schema_shape(parsed, schema)
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
                    "content": (
                        "The previous response did not satisfy the required JSON "
                        "object contract. Return only one JSON object containing "
                        "every required schema field, with no analysis, markdown, "
                        "YAML, or second object."
                    ),
                })
                body["options"] = {
                    "temperature": 0,
                    "num_predict": output_budget,
                    "seed": Config.ollama_seed,
                }
        if (
            last_parsed_candidate is not None
            and self._schema_has_known_property(
                last_parsed_candidate, schema,
            )
        ):
            # Keep a schema-compatible legacy object available to the
            # component parser.  This is deliberately narrower than accepting
            # an arbitrary extracted fragment: a response must contain at
            # least one declared property, while aliases and semantic gates
            # remain the responsibility of the caller.
            self.last_parsed_json_response = last_parsed_candidate
            return last_parsed_candidate
        raise RuntimeError(f"Ollama JSON response invalid after retry: {parse_error}")

    @staticmethod
    def _consume_streaming_chat(
        response: Any, cancel_check: Callable[[], bool],
    ) -> dict[str, Any]:
        """Collect an Ollama NDJSON response while permitting transport close."""

        fragments: list[str] = []
        final: dict[str, Any] = {}
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if cancel_check():
                    response.close()
                    raise OllamaCancelled("Ollama 请求已由调用方终止")
                if not raw:
                    continue
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                try:
                    item = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise OllamaProtocolError("Ollama 流式响应包含非法 JSON") from exc
                if not isinstance(item, dict):
                    raise OllamaProtocolError("Ollama 流式响应不是对象")
                content = str((item.get("message") or {}).get("content") or "")
                if content:
                    fragments.append(content)
                final = item
            if cancel_check():
                raise OllamaCancelled("Ollama 请求已由调用方终止")
        finally:
            response.close()
        if not final:
            raise OllamaProtocolError("Ollama 流式响应为空")
        final = dict(final)
        final["message"] = {
            **dict(final.get("message") or {}),
            "content": "".join(fragments),
        }
        return final

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
    def _parse_json(
        content: str,
        *,
        schema: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = (
                OllamaClient._repair_missing_schema_property_separators(
                    cleaned, schema=schema,
                )
            )
            repaired = OllamaClient._repair_unescaped_json_string_quotes(
                repaired
            )
            try:
                value = json.loads(repaired)
            except json.JSONDecodeError:
                yaml_like = OllamaClient._parse_key_value_object(cleaned)
                if yaml_like is not None:
                    return sanitize_unicode_data(yaml_like)
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
        return sanitize_unicode_data(value)

    @staticmethod
    def _repair_missing_schema_property_separators(
        content: str,
        *,
        schema: dict[str, Any] | str | None,
    ) -> str:
        """Close a malformed scalar before a known top-level Schema key.

        A schema-constrained local model can finish a long string value and
        start the next requested property on a new line without emitting the
        closing quote and comma.  Generic JSON substring recovery then finds
        an inner span object and loses the otherwise complete response.  This
        repair is deliberately Schema-guided: it considers only a line that
        starts with an exact top-level property declared by the request, and
        only while the JSON scanner is still inside the preceding string.
        The reconstructed object must subsequently pass ordinary JSON parsing
        and the complete top-level Schema check.
        """

        if not isinstance(schema, dict):
            return content
        properties = schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            return content
        known = {str(key) for key in properties}
        if not known or "\n" not in content:
            return content

        result: list[str] = []
        in_string = False
        escaped = False
        index = 0
        length = len(content)
        while index < length:
            char = content[index]
            if char == "\n" and in_string and not escaped:
                match = re.match(
                    r'\n([ \t]*)"([^"\\\r\n]+)"[ \t]*:',
                    content[index:],
                )
                if match is not None and match.group(2) in known:
                    result.extend(('"', ',', '\n'))
                    in_string = False
                    index += 1
                    continue
            result.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            index += 1
        return "".join(result)

    @staticmethod
    def _repair_unescaped_json_string_quotes(content: str) -> str:
        """Escape only quotes that are unambiguously inside a JSON string.

        Schema-constrained local models occasionally place a quoted phrase in
        a long rationale without escaping its double quotes.  The surrounding
        object and every typed audit field can otherwise be valid.  A quote is
        treated as a legitimate string terminator only when the next
        non-whitespace character is a JSON delimiter (colon for an object key,
        comma, closing bracket/brace, or end of input).  The repaired object is
        still parsed normally and then checked against the complete requested
        Schema; this helper never fabricates a field or accepts a fragment.
        """

        result: list[str] = []
        in_string = False
        escaped = False
        length = len(content)
        for index, char in enumerate(content):
            if not in_string:
                result.append(char)
                if char == '"':
                    in_string = True
                continue
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char != '"':
                result.append(char)
                continue
            next_index = index + 1
            while (
                next_index < length
                and content[next_index] in " \t\r\n"
            ):
                next_index += 1
            next_char = (
                content[next_index] if next_index < length else ""
            )
            if not next_char or next_char in ":,}]":
                result.append(char)
                in_string = False
            else:
                result.append('\\"')
        return "".join(result)

    @staticmethod
    def _parse_key_value_object(content: str) -> dict[str, Any] | None:
        """Recover a small line-oriented object emitted instead of JSON.

        This intentionally handles only ``key: scalar`` and JSON-array values.
        It is a transport compatibility layer for schema-constrained models
        that occasionally emit YAML-like lines; semantic aliases and type
        rules remain in the component parsers.
        """

        result: dict[str, Any] = {}
        for line in content.splitlines():
            match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*?)\s*$", line)
            if not match:
                continue
            key, raw_value = match.groups()
            if not raw_value:
                result[key] = ""
                continue
            try:
                result[key] = json.loads(raw_value)
            except json.JSONDecodeError:
                normalized = raw_value.strip().lower()
                if normalized == "true":
                    result[key] = True
                elif normalized == "false":
                    result[key] = False
                elif normalized in {"null", "none"}:
                    result[key] = None
                else:
                    result[key] = raw_value.strip().strip('"').strip("'")
        return result if len(result) >= 2 else None

    @staticmethod
    def _validate_schema_shape(
        value: dict[str, Any], schema: dict[str, Any] | str | None,
    ) -> None:
        """Reject a JSON-shaped fragment that is not the requested object.

        Some local models wrap a YAML-like answer in prose.  The permissive
        substring recovery in ``_parse_json`` can otherwise mistake an inner
        list (for example a span) for the whole response.  Checking only the
        schema's top-level required keys keeps this transport-layer guard
        generic; component parsers remain responsible for semantic validation.
        """

        if not isinstance(schema, dict):
            return
        required = schema.get("required")
        if not isinstance(required, list):
            return
        if OllamaClient._schema_accepts_items_array(value, schema):
            return
        missing = [
            str(key) for key in required
            if str(key) not in value
        ]
        if missing:
            raise RuntimeError(
                "Ollama JSON response omitted required fields: "
                + ", ".join(missing[:12])
            )

    @staticmethod
    def _schema_has_known_property(
        value: dict[str, Any], schema: dict[str, Any] | str | None,
    ) -> bool:
        if not isinstance(schema, dict):
            return True
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return True
        if OllamaClient._schema_accepts_items_array(value, schema):
            return True
        if any(str(key) in properties for key in value):
            return True
        # A line-oriented compatibility object may use parser-level aliases
        # rather than the exact schema spellings.  Require several fields so
        # an extracted inner list/object such as ``{"items": [...]}`` is not
        # mistaken for the response root.
        return len(value) >= 3 and "items" not in value

    @staticmethod
    def _schema_accepts_items_array(
        value: dict[str, Any], schema: dict[str, Any] | str | None,
    ) -> bool:
        """Allow the historical object wrapper for a top-level JSON array."""

        if not isinstance(schema, dict) or value.keys() != {"items"}:
            return False
        required = schema.get("required")
        properties = schema.get("properties")
        if not isinstance(required, list) or not isinstance(properties, dict):
            return False
        array_fields = [
            key for key in required
            if isinstance(properties.get(key), dict)
            and properties[key].get("type") == "array"
        ]
        return len(array_fields) == 1
