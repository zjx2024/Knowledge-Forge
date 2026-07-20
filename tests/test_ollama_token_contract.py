from __future__ import annotations

from knowledge_forge.ollama import OllamaClient


def test_current_transport_does_not_apply_output_limit_characterization(monkeypatch):
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
        model="model", system="system", prompt="prompt", max_tokens=4096,
    )
    assert "num_ctx" not in calls[0]["options"]
    assert "num_predict" not in calls[0]["options"]
