from __future__ import annotations

import json

import pytest

from knowledge_forge.config import Config
from knowledge_forge.relation_shadow_runtime import ShadowRelationV2Runtime
from knowledge_forge.relation_shadow_v2 import RETRYABLE_FAILED
from tests.test_shadow_relation_runtime_wiring import (
    _RuntimeDatabase,
    _ScriptedGateway,
)


def _keep_chunks(database: _RuntimeDatabase, chunk_ids: set[int]) -> None:
    database.chunks = [
        item for item in database.chunks if int(item["id"]) in chunk_ids
    ]
    database.mentions = [
        item for item in database.mentions if int(item["chunk_id"]) in chunk_ids
    ]
    database.assignments = [
        item for item in database.assignments
        if any(mention["mention_key"] == item["mention_key"]
               for mention in database.mentions)
    ]


def test_singleton_empty_classifier_response_retries_then_fails_closed(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 2)
    database = _RuntimeDatabase()
    _keep_chunks(database, {1})

    class EmptyClassifierGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                return {"items": []}
            return super().json_chat(**kwargs)

    with pytest.raises(RuntimeError, match="禁止回退旧流程"):
        ShadowRelationV2Runtime(
            database, EmptyClassifierGateway(),
        ).run_namespace(namespace_id="namespace-runtime", project_id=1)

    assert database.decisions == []
    assert database.endpoints == []
    assert database.finished[-1][1]["status"] == "failed"
    assert [values["status"] for _, _, values in database.classification_attempts] == [
        "RETRYABLE_EMPTY",
        "TERMINAL_FAILED",
    ]
    assert all(
        values["status"] != "ACCEPTED"
        for _, _, values in database.classification_attempts
    )


def test_missing_pair_in_batch_is_retried_by_shrinking_without_losing_pairs(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 1)
    monkeypatch.setattr(Config, "relation_classifier_max_candidates", 10)
    database = _RuntimeDatabase()
    _keep_chunks(database, {1})
    content = "NO_CASE Left1 connects Right1 and Extra1."
    database.chunks[0]["content"] = content
    database.chunks[0]["content_hash"] = "chunk-hash-1-with-extra"
    extra_start = content.index("Extra1")
    database.mentions.append({
        "mention_key": "extra-left",
        "chunk_id": 1,
        "surface": "Extra1",
        "evidence_start": extra_start,
        "evidence_end": extra_start + len("Extra1"),
    })
    database.assignments.append({
        "mention_key": "extra-left",
        "canonical_key": "canonical-1-extra",
        "entity_type": "Organization",
        "policy_version": "assignment-v1",
        "revision_id": "revision-runtime",
    })

    class OmitLastPairOnceGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.omitted_once = False

        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                payload = json.loads(kwargs["prompt"])
                response = super().json_chat(**kwargs)
                if not self.omitted_once and len(payload["pairs"]) > 1:
                    self.omitted_once = True
                    response["items"] = response["items"][:-1]
                return response
            return super().json_chat(**kwargs)

    result = ShadowRelationV2Runtime(
        database, OmitLastPairOnceGateway(),
    ).run_namespace(namespace_id="namespace-runtime", project_id=1)

    statuses = [values["status"] for _, _, values in database.classification_attempts]
    assert "RETRYABLE_EMPTY" in statuses
    assert statuses.count("ACCEPTED") == len(database.decisions)
    assert "TERMINAL_FAILED" not in statuses
    assert result["metrics"]["pair_opportunities_classified"] == len(database.decisions)
    assert database.finished[-1][1]["status"] == "ready"


def test_verifier_terminal_failure_is_not_ready_and_does_not_materialize(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 2)
    database = _RuntimeDatabase()
    _keep_chunks(database, {11})

    class TimeoutVerifierGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            if "proposer" in system or "classifier" in system:
                return super().json_chat(**kwargs)
            self.verifier_calls += 1
            raise TimeoutError("verifier timeout")

    with pytest.raises(RuntimeError, match="verifier 存在终态失败"):
        ShadowRelationV2Runtime(
            database, TimeoutVerifierGateway(),
        ).run_namespace(namespace_id="namespace-runtime", project_id=1)

    assert database.normalization is None
    assert database.endpoints == []
    assert database.finished[-1][1]["status"] == "failed"
    assert [item.verdict for item in database.verification_attempts] == [
        RETRYABLE_FAILED,
        RETRYABLE_FAILED,
    ]
