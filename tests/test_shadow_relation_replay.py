from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from knowledge_forge.shadow_relation_replay import (
    PersistedShadowModelReplayGateway,
    ShadowReplayError,
    assert_code_only_replay_compatible,
)


def _fingerprint(code_tree: str, *, documents: str = "documents"):
    return SimpleNamespace(
        root=f"root-{code_tree}-{documents}",
        components={
            "documents": documents,
            "schema": "schema",
            "classifier": "classifier",
            "materialization_root": {
                "code_tree": code_tree,
                "policy_root": "policy",
            },
        },
    )


def test_code_only_replay_compatibility_rejects_every_other_input_change():
    source = {
        "id": "run-old",
        "status": "stale",
        "input_fingerprint": "root-old",
        "fingerprint_components": _fingerprint("old").components,
    }

    accepted = assert_code_only_replay_compatible(
        source, _fingerprint("new"),
    )
    assert accepted["changed_components"] == [
        "materialization_root.code_tree",
    ]

    with pytest.raises(ShadowReplayError, match="documents"):
        assert_code_only_replay_compatible(
            source, _fingerprint("new", documents="changed"),
        )


class _Rows:
    def __init__(self, rows):
        self.rows = rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return list(self.rows)


class _ReplayDatabase:
    def __init__(self, calls):
        self.calls = calls

    @contextmanager
    def connection(self):
        calls = self.calls

        class Connection:
            @staticmethod
            def execute(sql, _params):
                if "FROM shadow_relation_v2_runs" in sql:
                    return _Rows([{"id": "run-old", "status": "stale"}])
                if "FROM shadow_model_calls" in sql:
                    return _Rows(calls)
                raise AssertionError(sql)

        yield Connection()


class _RuntimeModel:
    @staticmethod
    def effective_model_context(_model):
        return SimpleNamespace(digest="model-digest")


def _request(system: str, prompt: str) -> dict:
    return {
        "model": "model",
        "system": system,
        "prompt": prompt,
        "schema": {"type": "object"},
        "temperature": 0.0,
        "context_budget_tokens": 1000,
        "max_output_tokens": 100,
        "safety_margin_tokens": 10,
    }


def test_persisted_gateway_replays_exact_sequence_and_never_calls_model():
    proposer = _request("relation proposer", '{"context":{"context_id":"c1"}}')
    classifier = _request(
        "relation classifier",
        '{"pairs":[{"pair_id":"p1"}]}',
    )
    classifier_retry = _request(
        "relation classifier",
        '{"pairs":[{"pair_id":"p1"}],"protocol_feedback":"retry"}',
    )
    verifier_prompt = '{"hypothesis_id":"h1"}'
    calls = [
        {
            "id": "m1", "component": "proposer", "status": "ACCEPTED",
            "attempt_no": 1, "request_payload": proposer,
            "raw_response_text": '{"proposals":[]}',
            "parsed_response": {"raw": {"proposals": []}},
            "error_type": "", "error_message": "", "created_at": 1,
        },
        {
            "id": "m2", "component": "classifier",
            "status": "RETRYABLE_MODEL_ERROR", "attempt_no": 1,
            "request_payload": classifier, "raw_response_text": "",
            "parsed_response": {}, "error_type": "OllamaProtocolError",
            "error_message": "temporary protocol failure", "created_at": 2,
        },
        {
            "id": "m3", "component": "classifier", "status": "ACCEPTED",
            "attempt_no": 2, "request_payload": classifier_retry,
            "raw_response_text": '[{"pair_id":"p1","decision":"NO_RELATION"}]',
            "parsed_response": [{"pair_id": "p1", "decision": "NO_RELATION"}],
            "error_type": "", "error_message": "", "created_at": 3,
        },
        {
            # One batch model call is audited once per pair.  An identical
            # audit membership must not become a second replayed model call.
            "id": "m3-pair-2", "component": "classifier",
            "status": "ACCEPTED", "attempt_no": 1,
            "request_payload": classifier_retry,
            "raw_response_text": '[{"pair_id":"p1","decision":"NO_RELATION"}]',
            "parsed_response": [{"pair_id": "p1", "decision": "NO_RELATION"}],
            "error_type": "", "error_message": "", "created_at": 3,
        },
        {
            # REVIEW_ONLY/REJECTED deterministic validation rows are
            # verification audit terminals, not verifier model calls.
            "id": "m-review", "component": "verifier",
            "status": "UNCERTAIN", "attempt_no": 1,
            "request_payload": {"hypothesis_id": "h-review"},
            "raw_response_text": "",
            "parsed_response": {
                "hypothesis_id": "h-review", "verdict": "UNCERTAIN",
                "reason_code": "REVIEW_ONLY",
                "supported_evidence_spans": [],
            },
            "error_type": "", "error_message": "",
            "independent_call": False, "created_at": 3,
        },
        {
            "id": "m4", "component": "verifier", "status": "SUPPORTED",
            "attempt_no": 1, "request_payload": {"hypothesis_id": "h1"},
            "raw_response_text": '{"verdict":"SUPPORTED"}',
            "parsed_response": {
                "hypothesis_id": "h1", "verdict": "SUPPORTED",
                "reason_code": "SUPPORTED",
                "forward_entailment": "ENTAILED",
                "reverse_entailment": "NOT_ENTAILED",
                "subject_role_supported": True,
                "object_role_supported": True,
                "supported_evidence_spans": [
                    {"start": 0, "end": 4},
                ],
            },
            "error_type": "", "error_message": "", "created_at": 4,
        },
    ]
    gateway = PersistedShadowModelReplayGateway(
        _ReplayDatabase(calls),
        "run-old",
        runtime_model_gateway=_RuntimeModel(),
    )

    assert gateway.json_chat(**proposer) == {"proposals": []}
    with pytest.raises(RuntimeError) as recorded:
        gateway.json_chat(**classifier)
    assert type(recorded.value).__name__ == "OllamaProtocolError"
    assert gateway.json_chat(**classifier_retry)[0]["pair_id"] == "p1"
    verifier = gateway.json_chat(
        **_request("relation verifier", verifier_prompt),
    )
    assert verifier == {
        "verdict": "SUPPORTED",
        "reason_code": "SUPPORTED",
        "forward_entailment": "ENTAILED",
        "reverse_entailment": "NOT_ENTAILED",
        "subject_role_supported": True,
        "object_role_supported": True,
        "supported_evidence_spans": [{"start": 0, "end": 4}],
    }
    gateway.assert_replay_complete()
    summary = gateway.replay_summary()
    assert summary["live_model_calls"] == 0
    assert summary["source_audit_rows"] == {
        "proposer": 1, "classifier": 3, "verifier": 2,
    }
    assert summary["source_calls"] == {
        "proposer": 1, "classifier": 2, "verifier": 1,
    }


def test_persisted_gateway_fails_closed_on_unmatched_or_unused_call():
    call = {
        "id": "m1", "component": "classifier", "status": "ACCEPTED",
        "attempt_no": 1,
        "request_payload": _request("relation classifier", '{"pairs":[]}'),
        "raw_response_text": "[]", "parsed_response": [],
        "error_type": "", "error_message": "", "created_at": 1,
    }
    gateway = PersistedShadowModelReplayGateway(
        _ReplayDatabase([call]),
        "run-old",
        runtime_model_gateway=_RuntimeModel(),
    )

    with pytest.raises(ShadowReplayError, match="无法在冻结账本中精确匹配"):
        gateway.json_chat(
            **_request("relation classifier", '{"pairs":[{"pair_id":"new"}]}'),
        )
    with pytest.raises(ShadowReplayError, match="拒绝物化"):
        gateway.assert_replay_complete()
