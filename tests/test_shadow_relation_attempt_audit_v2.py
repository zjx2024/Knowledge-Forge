from __future__ import annotations

from knowledge_forge.config import Config
from knowledge_forge.relation_shadow_runtime import (
    ShadowRelationV2Runtime,
    shadow_model_payload_hash,
)
from tests.test_shadow_relation_runtime_wiring import (
    _RuntimeDatabase,
    _ScriptedGateway,
)
from tests.test_shadow_relation_retry_v2 import _keep_chunks


def test_model_payload_hashes_are_domain_separated() -> None:
    payload = {"same": "payload"}

    assert shadow_model_payload_hash("request", payload) != shadow_model_payload_hash(
        "response", payload,
    )
    assert shadow_model_payload_hash("request", payload) == shadow_model_payload_hash(
        "request", {"same": "payload"},
    )


def test_classifier_attempt_audit_payloads_recompute_hashes(monkeypatch) -> None:
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 1)
    database = _RuntimeDatabase()
    _keep_chunks(database, {11})

    ShadowRelationV2Runtime(
        database, _ScriptedGateway(),
    ).run_namespace(namespace_id="namespace-runtime", project_id=1)

    accepted = [
        values for _, _, values in database.classification_attempts
        if values["status"] == "ACCEPTED"
    ]
    assert len(accepted) == 1
    attempt = accepted[0]
    assert attempt["independent_call"] is True
    assert attempt["request_payload"]["stage"] == "classifier"
    assert attempt["raw_response_text"]
    assert attempt["parsed_response"]
    assert attempt["error_type"] == ""
    assert attempt["request_hash"] == shadow_model_payload_hash(
        "request", attempt["request_payload"],
    )
    assert attempt["response_hash"] == shadow_model_payload_hash(
        "response", attempt["raw_response_text"],
    )


def test_classifier_failure_audit_records_error_without_response_hash(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 1)
    database = _RuntimeDatabase()
    _keep_chunks(database, {11})

    result = ShadowRelationV2Runtime(
        database, _ScriptedGateway(fail_classifier=True),
    ).run_namespace(namespace_id="namespace-runtime", project_id=1)

    attempt = database.classification_attempts[-1][2]
    assert attempt["status"] == "TERMINAL_FAILED"
    assert attempt["request_payload"]["stage"] == "classifier"
    assert attempt["request_hash"] == shadow_model_payload_hash(
        "request", attempt["request_payload"],
    )
    assert attempt["raw_response_text"] == ""
    assert attempt["response_hash"] == ""
    assert attempt["error_type"] == "RuntimeError"
    assert "scripted classifier failure" in attempt["error_message"]
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"
    assert database.endpoints == []
