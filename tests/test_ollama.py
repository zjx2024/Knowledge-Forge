import base64

from knowledge_forge.ollama import OllamaClient


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
            return {"message": {"content": self.content}}

    responses = iter([Response("没有结构化结果"), Response('{"entities": [], "relations": []}')])
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json.copy())
        return next(responses)

    monkeypatch.setattr("knowledge_forge.ollama.requests.post", fake_post)
    result = OllamaClient("http://ollama.test").json_chat(
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
            return {"message": {"content": '{"ok": true}'}}

    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(),
    )

    OllamaClient("http://ollama.test").json_chat(
        model="test", system="system", prompt="prompt", max_tokens=32768
    )

    assert "num_ctx" not in calls[0]["options"]


def test_json_chat_rejects_input_over_project_context_budget(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda *args, **kwargs: calls.append(kwargs) or None,
    )

    try:
        OllamaClient("http://ollama.test").json_chat(
            model="test",
            system="system",
            prompt="很长的项目输入" * 20,
            max_tokens=8,
        )
    except ValueError as exc:
        assert "超过项目上下文上限" in str(exc)
    else:
        raise AssertionError("expected project context validation to fail")
    assert calls == []


def test_json_chat_sends_embedded_images_without_changing_runner_context(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"message": {"content": '{"description": "diagram"}'}}

    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(),
    )
    image = b"a small image payload"
    OllamaClient("http://ollama.test").json_chat(
        model="vision",
        system="system",
        prompt="prompt",
        images=[image],
        max_tokens=32768,
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
