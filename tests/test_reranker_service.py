from __future__ import annotations

import socket
import threading

import requests

from knowledge_forge.entity_reranker import EntityReranker
from knowledge_forge.reranker_service import (
    PROTOCOL_VERSION,
    RerankerServiceError,
    RerankerServiceSettings,
    create_server,
)


class FakeRuntime:
    def __init__(self, scores=None):
        self.scores = list(scores or [0.2, 0.9, 0.4])
        self.device = "cpu"
        self.calls = []

    def diagnostics(self):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "status": "ok",
            "ready": True,
            "model": "fixture-reranker",
            "model_loaded": True,
            "device": self.device,
            "error": "",
        }

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return self.scores[: len(documents)]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _settings(port: int, *, api_key: str = "") -> RerankerServiceSettings:
    return RerankerServiceSettings(
        host="127.0.0.1",
        port=port,
        model_name="fixture-reranker",
        api_key=api_key,
        allow_unauthenticated_lan=False,
        max_documents=8,
        max_text_chars=1000,
        max_total_chars=4000,
        max_body_bytes=16_384,
        max_pending_requests=4,
        queue_timeout=2,
    )


class RunningServer:
    def __init__(self, runtime=None, *, api_key=""):
        self.runtime = runtime or FakeRuntime()
        self.server = create_server(
            settings=_settings(_free_port(), api_key=api_key),
            runtime=self.runtime,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True,
        )

    @property
    def base_url(self):
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def test_service_reranks_and_returns_stable_source_indices():
    with RunningServer() as running:
        response = requests.post(
            f"{running.base_url}/api/rerank",
            json={
                "model": "fixture-reranker",
                "query": "target",
                "documents": ["first", "second", "third"],
                "top_n": 2,
            },
            timeout=2,
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["results"] == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.4},
    ]
    assert running.runtime.calls == [
        ("target", ["first", "second", "third"]),
    ]


def test_service_requires_configured_bearer_token_but_health_is_safe():
    key = "fixture-secret-at-least-16-characters"
    with RunningServer(api_key=key) as running:
        health = requests.get(f"{running.base_url}/health", timeout=2)
        denied = requests.post(
            f"{running.base_url}/v1/rerank",
            json={
                "model": "fixture-reranker",
                "query": "q",
                "documents": ["d"],
            },
            timeout=2,
        )
        accepted = requests.post(
            f"{running.base_url}/v1/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "fixture-reranker",
                "query": "q",
                "documents": ["d"],
            },
            timeout=2,
        )

    assert health.status_code == 200
    assert health.json()["ready"] is True
    assert denied.status_code == 401
    assert accepted.status_code == 200


def test_http_client_restores_input_alignment_from_ranked_results(monkeypatch):
    monkeypatch.setattr(
        "knowledge_forge.entity_reranker.Config.entity_reranker_enabled", True,
    )
    runtime = FakeRuntime([0.15, 0.92])
    key = "fixture-secret-at-least-16-characters"
    with RunningServer(runtime, api_key=key) as running:
        client = EntityReranker(
            base_url=running.base_url,
            model="fixture-reranker",
            api_key=key,
            retries=0,
            timeout=2,
        )
        scores = client.score(
            {
                "name": "WHO",
                "type": "Organization",
                "description": "health agency",
                "aliases": [],
            },
            [
                {
                    "name": "World Trade Organization",
                    "entity_type": "Organization",
                    "description": "trade body",
                    "aliases": ["WTO"],
                },
                {
                    "name": "World Health Organization",
                    "entity_type": "Organization",
                    "description": "health agency",
                    "aliases": ["WHO"],
                },
            ],
        )

    assert scores == [0.15, 0.92]
    assert client.last_score_used_model is True
    diagnostics = client.diagnostics()
    assert diagnostics["reranker_transport"] == "http"
    assert diagnostics["last_http_status"] == 200
    assert diagnostics["reranker_loaded"] is True


def test_http_client_fails_closed_to_zero_scores(monkeypatch):
    monkeypatch.setattr(
        "knowledge_forge.entity_reranker.Config.entity_reranker_enabled", True,
    )
    client = EntityReranker(
        base_url=f"http://127.0.0.1:{_free_port()}",
        model="fixture-reranker",
        retries=0,
        connect_timeout=1,
        timeout=1,
    )
    candidates = [{
        "name": "candidate",
        "entity_type": "GenericEntity",
        "description": "",
        "aliases": [],
    }]

    assert client.score(
        {
            "name": "mention",
            "type": "GenericEntity",
            "description": "",
            "aliases": [],
        },
        candidates,
    ) == [0.0]
    assert client.last_score_used_model is False
    assert client.diagnostics()["fallback_reason"]


def test_non_loopback_service_requires_auth_or_explicit_lan_opt_in():
    settings = _settings(8012)
    settings = RerankerServiceSettings(
        **{**settings.__dict__, "host": "0.0.0.0"},
    )

    try:
        settings.validate()
    except RerankerServiceError as exc:
        assert "RERANKER_API_KEY" in str(exc)
    else:
        raise AssertionError("unsafe non-loopback bind was accepted")

    RerankerServiceSettings(
        **{
            **settings.__dict__,
            "allow_unauthenticated_lan": True,
        },
    ).validate()
