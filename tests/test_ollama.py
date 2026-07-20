import base64

from knowledge_forge.ollama import EffectiveModelContext, OllamaClient


def ready(client, monkeypatch, *, context=32768):
    monkeypatch.setattr(
        client,
        "effective_model_context",
        lambda model: EffectiveModelContext(model, context, context, "test", "runner"),
    )
    return client


def test_parse_json_preserves_array_for_domain_normalization():
    assert OllamaClient._parse_json("```json\n[{\"id\": \"e1\"}]\n```") == {
        "items": [{"id": "e1"}]
    }


def test_parse_json_uses_first_complete_object_when_model_appends_content():
    content = '说明：\n{"entities": [], "relations": []}\n{"unexpected": true}'
    assert OllamaClient._parse_json(content) == {"entities": [], "relations": []}


def test_json_chat_retries_an_invalid_model_response(monkeypatch):
    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": self.content}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 10,
            }

    responses = iter([Response("没有结构化结果"), Response('{"entities": [], "relations": []}')])
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json.copy())
        return next(responses)

    monkeypatch.setattr("knowledge_forge.ollama.requests.post", fake_post)
    result = ready(OllamaClient("http://ollama.test"), monkeypatch).json_chat(
        model="test", system="system", prompt="prompt", schema={"type": "object"}
    )

    assert result == {"entities": [], "relations": []}
    assert len(calls) == 2
    assert calls[1]["options"]["temperature"] == 0


def test_json_chat_keeps_project_limit_out_of_ollama_runner_options(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": '{"ok": true}'}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 4,
            }

    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(),
    )

    ready(OllamaClient("http://ollama.test"), monkeypatch).json_chat(
        model="test", system="system", prompt="prompt",
        context_budget_tokens=32768, max_output_tokens=512,
    )

    assert "num_ctx" not in calls[0]["options"]


def test_json_chat_rejects_input_over_project_context_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda *args, **kwargs: calls.append(kwargs) or None,
    )

    try:
        ready(OllamaClient("http://ollama.test"), monkeypatch, context=8).json_chat(
            model="test",
            system="system",
            prompt="很长的项目输入" * 20,
            context_budget_tokens=8,
            max_output_tokens=1,
            safety_margin_tokens=0,
        )
    except ValueError as exc:
        assert "token budget exceeded" in str(exc)
    else:
        raise AssertionError("expected project context validation to fail")
    assert calls == []


def test_json_chat_sends_embedded_images_without_changing_runner_context(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": '{"description": "diagram"}'}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 4,
            }

    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(),
    )
    image = b"a small image payload"
    ready(OllamaClient("http://ollama.test"), monkeypatch).json_chat(
        model="vision",
        system="system",
        prompt="prompt",
        images=[image],
        context_budget_tokens=32768,
        max_output_tokens=512,
    )

    assert calls[0]["messages"][1]["images"] == [base64.b64encode(image).decode("ascii")]
    assert "num_ctx" not in calls[0]["options"]


def test_model_info_reads_ollama_context_length(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model_info": {"qwen.context_length": 262144},
                "capabilities": ["completion"],
                "details": {"family": "qwen"},
            }

    monkeypatch.setattr("knowledge_forge.ollama.requests.post", lambda *args, **kwargs: Response())
    info = OllamaClient("http://ollama.test").model_info("qwen")

    assert info["context_length"] == 262144
    assert info["capabilities"] == ["completion"]
