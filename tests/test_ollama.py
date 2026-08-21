import base64

from knowledge_forge.ollama import EffectiveModelContext, OllamaClient


def ready(client, monkeypatch, *, context=32768):
    monkeypatch.setattr(
        client,
        "effective_model_context",
        lambda model: EffectiveModelContext(model, context, context, "test", "runner"),
    )
    return client


def test_embed_can_use_a_separate_runtime_without_rerouting_generation(monkeypatch):
    calls = []

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.25, 0.75]]}

    client = OllamaClient(
        "http://generation.test",
        embedding_base_url="http://embedding.test",
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda method, url, **kwargs: calls.append((method, url, kwargs))
        or Response(),
    )

    assert client.embed(["source"], "bge-m3") == [[0.25, 0.75]]
    assert calls[0][1] == "http://embedding.test/api/embed"
    assert client.base_url == "http://generation.test"


def test_parse_json_preserves_array_for_domain_normalization():
    assert OllamaClient._parse_json("```json\n[{\"id\": \"e1\"}]\n```") == {
        "items": [{"id": "e1"}]
    }


def test_parse_json_uses_first_complete_object_when_model_appends_content():
    content = '说明：\n{"entities": [], "relations": []}\n{"unexpected": true}'
    assert OllamaClient._parse_json(content) == {"entities": [], "relations": []}


def test_parse_json_repairs_unescaped_quotes_inside_a_string_value():
    content = (
        '{"verdict":"SUPPORTED","reason_code":"Judge "exact claim" now",'
        '"forward_entailment":"ENTAILED","reverse_entailment":"NOT_ENTAILED",'
        '"subject_role_supported":true,"object_role_supported":true,'
        '"supported_evidence_spans":[{"start":0,"end":12}]}'
    )

    parsed = OllamaClient._parse_json(content)

    assert parsed["verdict"] == "SUPPORTED"
    assert parsed["reason_code"] == 'Judge "exact claim" now'
    assert parsed["supported_evidence_spans"] == [{"start": 0, "end": 12}]


def test_json_chat_accepts_full_schema_after_inner_quote_repair(monkeypatch):
    malformed = (
        '{"verdict":"SUPPORTED","reason_code":"The prompt says "judge only".",'
        '"forward_entailment":"ENTAILED","reverse_entailment":"NOT_ENTAILED",'
        '"subject_role_supported":true,"object_role_supported":true,'
        '"supported_evidence_spans":[{"start":0,"end":12}]}'
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": malformed},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 10,
            }

    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(),
    )
    required = [
        "verdict", "reason_code", "forward_entailment",
        "reverse_entailment", "subject_role_supported",
        "object_role_supported", "supported_evidence_spans",
    ]
    result = ready(OllamaClient("http://ollama.test"), monkeypatch).json_chat(
        model="test",
        system="system",
        prompt="prompt",
        schema={
            "type": "object",
            "properties": {key: {} for key in required},
            "required": required,
        },
    )

    assert result["verdict"] == "SUPPORTED"
    assert result["reason_code"] == 'The prompt says "judge only".'
    assert len(calls) == 1


def test_json_chat_repairs_missing_string_close_before_known_schema_key(monkeypatch):
    malformed = (
        '{\n  "verdict": "SUPPORTED",\n'
        '  "reason_code": "The exact directed claim is supported.\n'
        '  "forward_entailment": "ENTAILED",\n'
        '  "reverse_entailment": "NOT_ENTAILED",\n'
        '  "subject_role_supported": true,\n'
        '  "object_role_supported": true,\n'
        '  "supported_evidence_spans": [{"start": 0, "end": 12}]\n}'
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": malformed},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 20,
                "eval_count": 10,
            }

    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json) or Response(),
    )
    required = [
        "verdict", "reason_code", "forward_entailment",
        "reverse_entailment", "subject_role_supported",
        "object_role_supported", "supported_evidence_spans",
    ]
    result = ready(OllamaClient("http://ollama.test"), monkeypatch).json_chat(
        model="test",
        system="system",
        prompt="prompt",
        schema={
            "type": "object",
            "properties": {key: {} for key in required},
            "required": required,
        },
    )

    assert result["verdict"] == "SUPPORTED"
    assert result["reason_code"] == "The exact directed claim is supported."
    assert result["forward_entailment"] == "ENTAILED"
    assert result["supported_evidence_spans"] == [{"start": 0, "end": 12}]
    assert len(calls) == 1


def test_schema_separator_repair_does_not_treat_unknown_quoted_line_as_key():
    malformed = (
        '{"known":"first line\n'
        '  "unknown": "still rationale"\n'
        '  "next": true}'
    )

    repaired = OllamaClient._repair_missing_schema_property_separators(
        malformed,
        schema={"type": "object", "properties": {"known": {}, "next": {}}},
    )

    assert '"unknown": "still rationale"' in repaired
    assert '\",\n  "next": true' in repaired


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


def test_json_chat_retries_a_json_fragment_missing_schema_fields(monkeypatch):
    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": self.content}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 20,
                "eval_count": 10,
            }

    responses = iter([
        Response("endpoint_a_role: actor\nspan: [1, 2]"),
        Response('{"endpoint_a_role":"actor","endpoint_b_role":"target"}'),
    ])
    calls = []

    def fake_post(url, json, timeout):
        calls.append(json.copy())
        return next(responses)

    monkeypatch.setattr("knowledge_forge.ollama.requests.post", fake_post)
    result = ready(OllamaClient("http://ollama.test"), monkeypatch).json_chat(
        model="test", system="system", prompt="prompt",
        schema={
            "type": "object",
            "required": ["endpoint_a_role", "endpoint_b_role"],
        },
    )

    assert result["endpoint_a_role"] == "actor"
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
    assert calls[0]["options"]["seed"] == 17


def test_json_chat_preserves_generation_seed_on_json_repair(monkeypatch):
    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": self.content}, "done": True,
                "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 4,
            }

    responses = iter([Response("not json"), Response('{"ok": true}')])
    calls = []
    monkeypatch.setattr(
        "knowledge_forge.ollama.requests.post",
        lambda url, json, timeout: calls.append(json.copy()) or next(responses),
    )
    ready(OllamaClient("http://ollama.test"), monkeypatch).json_chat(
        model="test", system="system", prompt="prompt", schema={"type": "object"},
    )

    assert calls[0]["options"]["seed"] == calls[1]["options"]["seed"] == 17


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
