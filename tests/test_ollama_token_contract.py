from __future__ import annotations

import pytest

from knowledge_forge.ollama import (
    EffectiveModelContext,
    OllamaBudgetError,
    OllamaClient,
    OllamaProtocolError,
)


class Response:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def ready_client(monkeypatch, *, effective_context: int = 8192):
    client = OllamaClient("http://ollama.test")
    monkeypatch.setattr(
        client,
        "effective_model_context",
        lambda _model: EffectiveModelContext(
            model="model",
            model_context_tokens=32768,
            effective_context_tokens=effective_context,
            digest="sha256:test",
            source="runner",
        ),
    )
    return client


def successful_payload(*, prompt_eval_count: int = 8, done_reason: str = "stop"):
    return {
        "message": {"content": '{"ok": true}'},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": 4,
    }


def test_transport_maps_output_limit_without_changing_runner_context(monkeypatch):
    client = ready_client(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(successful_payload()),
    )

    client.json_chat(
        model="model",
        system="system",
        prompt="prompt",
        context_budget_tokens=4096,
        max_output_tokens=512,
        safety_margin_tokens=128,
    )

    assert calls[0]["options"]["num_predict"] == 512
    assert "num_ctx" not in calls[0]["options"]
    assert calls[0]["options"]["num_predict"] != 4096
    assert client.last_call_diagnostics is not None
    assert client.last_call_diagnostics.max_output_tokens == 512
    assert client.last_call_diagnostics.prompt_eval_count == 8
    assert len(client.last_call_diagnostics.rendered_message_hash) == 64
    assert "system" not in client.last_call_diagnostics.rendered_message_hash


@pytest.mark.parametrize(
    ("estimated_input", "allowed"),
    [(639, True), (640, True), (641, False)],
)
def test_admission_boundary_is_inclusive_and_runs_before_http(
    monkeypatch, estimated_input, allowed,
):
    client = ready_client(monkeypatch, effective_context=1024)
    monkeypatch.setattr(client.token_estimator, "estimate", lambda _text: estimated_input)
    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(
            successful_payload(prompt_eval_count=estimated_input)
        ),
    )
    kwargs = dict(
        model="model",
        system="s",
        prompt="p",
        context_budget_tokens=1024,
        max_output_tokens=256,
        safety_margin_tokens=128,
    )

    if allowed:
        assert client.json_chat(**kwargs) == {"ok": True}
        assert len(calls) == 1
    else:
        with pytest.raises(OllamaBudgetError, match="token budget"):
            client.json_chat(**kwargs)
        assert calls == []


def test_effective_context_uses_runner_not_model_maximum(monkeypatch):
    client = OllamaClient("http://ollama.test")

    def fake_request(method, url, **_kwargs):
        if url.endswith("/api/show"):
            return Response({
                "model_info": {"qwen.context_length": 32768},
                "details": {"family": "qwen"},
                "capabilities": ["completion"],
            })
        if url.endswith("/api/ps"):
            return Response({
                "models": [{
                    "name": "model:latest",
                    "model": "model:latest",
                    "digest": "sha256:runner",
                    "context_length": 4096,
                }]
            })
        raise AssertionError(url)

    monkeypatch.setattr(client, "_request", fake_request)
    context = client.effective_model_context("model:latest")

    assert context.model_context_tokens == 32768
    assert context.effective_context_tokens == 4096
    assert context.digest == "sha256:runner"
    assert context.source == "runner"


def test_missing_runner_uses_explicit_conservative_fallback(monkeypatch):
    client = OllamaClient("http://ollama.test")

    def fake_request(method, url, **_kwargs):
        if url.endswith("/api/show"):
            return Response({"model_info": {"qwen.context_length": 32768}})
        if url.endswith("/api/ps"):
            return Response({"models": []})
        raise AssertionError(url)

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr("knowledge_forge.ollama.Config.ollama_effective_context_fallback", 2048)

    context = client.effective_model_context("model")
    assert context.effective_context_tokens == 2048
    assert context.source == "configured_fallback"


def test_prompt_token_underestimate_fails_closed(monkeypatch):
    client = ready_client(monkeypatch)
    monkeypatch.setattr(client.token_estimator, "estimate", lambda _text: 100)
    monkeypatch.setattr("knowledge_forge.ollama.Config.ollama_token_variance_absolute", 10)
    monkeypatch.setattr("knowledge_forge.ollama.Config.ollama_token_variance_ratio", 0.10)
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda *_args, **_kwargs: Response(successful_payload(prompt_eval_count=200)),
    )

    with pytest.raises(OllamaProtocolError, match="prompt token estimate"):
        client.json_chat(
            model="model", system="s", prompt="p",
            context_budget_tokens=4096, max_output_tokens=256,
        )


@pytest.mark.parametrize(
    "payload",
    [
        successful_payload(done_reason="length"),
        {**successful_payload(), "done": False},
    ],
)
def test_truncated_or_partial_response_fails_closed(monkeypatch, payload):
    client = ready_client(monkeypatch)
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda *_args, **_kwargs: Response(payload),
    )

    with pytest.raises(OllamaProtocolError, match="truncated|incomplete"):
        client.json_chat(
            model="model", system="s", prompt="p",
            context_budget_tokens=4096, max_output_tokens=256,
        )


def test_invalid_json_still_fails_closed_after_bounded_repair(monkeypatch):
    client = ready_client(monkeypatch)
    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response({
            **successful_payload(),
            "message": {"content": "not-json"},
        }),
    )

    with pytest.raises(RuntimeError, match="invalid after retry"):
        client.json_chat(
            model="model", system="s", prompt="p",
            context_budget_tokens=4096, max_output_tokens=256,
        )
    assert len(calls) == 2
    assert all(call["options"]["num_predict"] == 256 for call in calls)


def test_timeout_fails_closed(monkeypatch):
    client = ready_client(monkeypatch)
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("timeout")),
    )

    with pytest.raises(TimeoutError, match="timeout"):
        client.json_chat(
            model="model", system="s", prompt="p",
            context_budget_tokens=4096, max_output_tokens=256,
        )
