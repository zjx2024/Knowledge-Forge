from __future__ import annotations

import math
import os

import pytest
import requests

from knowledge_forge.config import Config
from knowledge_forge.entity_reranker import EntityReranker
from knowledge_forge.ollama import OllamaClient


def _require_enabled(name: str) -> None:
    if os.getenv(name, "").strip().lower() not in {"1", "true", "yes", "on"}:
        pytest.skip(f"{name}=1 is required")


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(
        sum(b * b for b in right)
    )
    return numerator / denominator


@pytest.mark.live_model
def test_live_ollama_structured_output_and_embedding_contract():
    _require_enabled("RUN_LIVE_MODEL_TESTS")
    client = OllamaClient(timeout=max(120, Config.ollama_timeout))
    models = {str(item.get("name") or ""): item for item in client.models()}

    assert Config.default_llm_model in models
    assert Config.default_embedding_model in models
    assert models[Config.default_llm_model].get("digest")

    info = client.model_info(Config.default_llm_model)
    assert info["context_length"] > 0
    assert info["capabilities"]

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok"]},
            "evidence": {"type": "string"},
        },
        "required": ["status", "evidence"],
        "additionalProperties": False,
    }
    result = client.json_chat(
        model=Config.default_llm_model,
        system="Return only JSON that follows the supplied schema.",
        prompt="Set status to ok and copy this exact evidence: live-smoke-evidence",
        schema=schema,
        temperature=0,
        max_tokens=2048,
    )
    assert result == {"status": "ok", "evidence": "live-smoke-evidence"}

    texts = ["alpha entity", "beta entity"]
    batch = client.embed(texts, Config.default_embedding_model)
    first_alone = client.embed([texts[0]], Config.default_embedding_model)[0]
    assert len(batch) == len(texts)
    assert len(batch[0]) == len(batch[1]) > 0
    assert all(math.isfinite(value) for vector in batch for value in vector)
    assert _cosine(batch[0], first_alone) > 0.99999
    assert _cosine(batch[0], batch[1]) < 0.99999


@pytest.mark.live_model
def test_live_reranker_preserves_pair_score_alignment_and_finite_values():
    _require_enabled("RUN_LIVE_MODEL_TESTS")
    reranker = EntityReranker()
    if not reranker.available:
        pytest.skip("configured local reranker model is unavailable")

    mention = {
        "name": "World Health Organization",
        "type": "Organization",
        "description": "United Nations health agency",
        "aliases": ["WHO"],
    }
    candidates = [
        {
            "id": 1,
            "name": "World Health Organization",
            "entity_type": "Organization",
            "description": "United Nations health agency",
            "aliases": ["WHO"],
        },
        {
            "id": 2,
            "name": "World Trade Organization",
            "entity_type": "Organization",
            "description": "International trade body",
            "aliases": ["WTO"],
        },
    ]
    scores = reranker.score(mention, candidates)

    assert len(scores) == len(candidates)
    assert all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores)
    assert scores[0] > scores[1]


@pytest.mark.hugegraph_integration
def test_live_hugegraph_read_only_graph_discovery():
    _require_enabled("RUN_HUGEGRAPH_INTEGRATION")
    response = requests.get(
        f"http://{Config.hugegraph_host}:{Config.hugegraph_port}/graphs",
        timeout=min(30, Config.hugegraph_timeout),
    )
    response.raise_for_status()

    assert Config.hugegraph_graph in (response.json().get("graphs") or [])
