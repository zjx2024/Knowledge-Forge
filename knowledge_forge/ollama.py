"""Ollama clients for embeddings and schema-constrained JSON extraction."""

from __future__ import annotations

import base64
import json
import math
import re
import time
from typing import Any, Iterable

import requests

from .config import Config
from .observability import current_metrics, increment


class OllamaClient:
    def __init__(self, base_url: str = "", timeout: int | None = None):
        self.base_url = (base_url or Config.ollama_base_url).rstrip("/")
        self.timeout = timeout or Config.ollama_timeout

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
            "capabilities": payload.get("capabilities") or [],
            "details": payload.get("details") or {},
        }

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
        max_tokens: int | None = None,
        images: Iterable[bytes] | None = None,
    ) -> dict[str, Any]:
        increment("ollama_logical_calls")
        increment("estimated_prompt_tokens", self._estimate_tokens("\n".join((system, prompt))))
        self._validate_project_context_budget(system, prompt, schema, max_tokens)
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
            },
            "format": schema or "json",
        }
        parse_error: Exception | None = None
        for attempt in range(2):
            if attempt:
                increment("http_retry_count")
            response = self._request(
                "post", f"{self.base_url}/api/chat", json=body, timeout=self.timeout
            )
            response.raise_for_status()
            content = str((response.json().get("message") or {}).get("content") or "").strip()
            try:
                return self._parse_json(content)
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

    @classmethod
    def _validate_project_context_budget(
        cls,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None,
        max_tokens: int | None,
    ) -> None:
        """Enforce the project budget without changing Ollama's runner context.

        Ollama's global context is configured by OLLAMA_CONTEXT_LENGTH. Sending
        options.num_ctx here would create a differently sized runner and can
        repeatedly unload an otherwise permanently kept-alive model.
        """
        if not max_tokens:
            return
        schema_text = json.dumps(schema, ensure_ascii=False) if schema else ""
        estimated = cls._estimate_tokens("\n".join((system, prompt, schema_text)))
        if estimated > int(max_tokens):
            raise ValueError(
                f"项目请求预计需要 {estimated} tokens，超过项目上下文上限 {int(max_tokens)}；"
                "请减小输入或提高项目上下文上限"
            )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservative tokenizer-free estimate for mixed Chinese/ASCII text."""
        cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
        return max(1, cjk + math.ceil(max(0, len(text) - cjk) / 4))

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
