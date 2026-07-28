from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from knowledge_forge.config import Config
from knowledge_forge.ollama import OllamaProtocolError
from knowledge_forge.relation_shadow_runtime import ShadowRelationV2Runtime
from knowledge_forge.relation_shadow_v2 import (
    HYPOTHESES,
    NO_RELATION,
    OTHER_UNSUPPORTED,
    REJECTED,
    ShadowClassificationError,
    SUPPORTED,
    UNCERTAIN,
    StaleRelationSnapshotError,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


def _schema(action_name: str = "acts_on") -> dict:
    return logical_schema(
        predicate_policy(
            action_name,
            domain=["Organization"],
            range_=["Product"],
        ),
        predicate_policy("linked", symmetric=True),
        predicate_policy(
            "contains",
            domain=["Container"],
            range_=["Part"],
            inverse_predicate="inside",
        ),
        predicate_policy(
            "inside",
            domain=["Part"],
            range_=["Container"],
            inverse_predicate="contains",
            canonical_orientation="inverse",
        ),
        predicate_policy("association", publish_mode="review_only"),
        version="schema-runtime-v1",
    )


class _ScriptedGateway:
    def __init__(self, action_name: str = "acts_on", *, fail_classifier=False):
        self.action_name = action_name
        self.fail_classifier = fail_classifier
        self.classifier_calls = 0
        self.verifier_calls = 0

    @staticmethod
    def effective_model_context(_model):
        return SimpleNamespace(digest="scripted-model-digest")

    def json_chat(self, **kwargs):
        system = str(kwargs["system"])
        payload = json.loads(kwargs["prompt"])
        if "proposer" in system:
            return {
                "proposals": [],
                "missing_endpoint_mentions": [],
                "coverage": {"complete": True, "uncovered_cues": []},
            }
        if "classifier" in system:
            self.classifier_calls += 1
            if self.fail_classifier:
                raise RuntimeError("scripted classifier failure")
            result = []
            for pair in payload["pairs"]:
                context = pair["context"]
                common = {
                    "pair_id": pair["pair_id"],
                    "hypotheses": [],
                }
                if "NO_CASE" in context:
                    result.append({
                        **common,
                        "decision": NO_RELATION,
                        "reason_code": "NO_SEMANTIC_RELATION",
                        "evidence_spans": [{
                            "start": pair["context_span"][0],
                            "end": pair["context_span"][1],
                        }],
                    })
                    continue
                if "OTHER_CASE" in context:
                    result.append({
                        **common,
                        "decision": OTHER_UNSUPPORTED,
                        "reason_code": "SCHEMA_GAP",
                        "evidence_spans": [{
                            "start": pair["context_span"][0],
                            "end": pair["context_span"][1],
                        }],
                    })
                    continue
                left = pair["left_occurrence_id"]
                right = pair["right_occurrence_id"]
                predicate = self.action_name
                subject, object_ = left, right
                hypotheses = []
                if "INVERSE_CASE" in context:
                    predicate = "inside"
                    subject = (
                        left if pair["left_type"] == "Part" else right
                    )
                    object_ = (
                        left if pair["left_type"] == "Container" else right
                    )
                elif "SYMMETRIC_CASE" in context:
                    predicate = "linked"
                elif "REVIEW_CASE" in context:
                    predicate = "association"
                spans = [{
                    "start": pair["context_span"][0],
                    "end": pair["context_span"][1],
                }]
                if "MULTI_EVIDENCE_CASE" in context:
                    spans = [
                        {"start": pair["left_span"][0], "end": pair["left_span"][1]},
                        {"start": pair["right_span"][0], "end": pair["right_span"][1]},
                    ]
                hypotheses.append({
                    "subject_occurrence_id": subject,
                    "predicate": predicate,
                    "object_occurrence_id": object_,
                    "evidence_spans": spans,
                    "classifier_confidence": 0.9,
                })
                if "MULTI_RELATION_CASE" in context:
                    hypotheses.append({
                        "subject_occurrence_id": subject,
                        "predicate": "linked",
                        "object_occurrence_id": object_,
                        "evidence_spans": spans,
                        "classifier_confidence": 0.8,
                    })
                result.append({
                    **common,
                    "decision": HYPOTHESES,
                    "hypotheses": hypotheses,
                })
            return {"items": result}
        self.verifier_calls += 1
        context = payload["context"]["text"]
        verdict = SUPPORTED
        if "VERIFIER_REJECT_CASE" in context:
            verdict = REJECTED
        elif "VERIFIER_UNCERTAIN_CASE" in context:
            verdict = UNCERTAIN
        return {
            "hypothesis_id": payload["hypothesis_id"],
            "verdict": verdict,
            "reason_code": f"SCRIPTED_{verdict}",
            "supported_evidence_spans": (
                payload["classifier_evidence_spans"]
                if verdict == SUPPORTED else []
            ),
        }


class _RuntimeDatabase:
    def __init__(self, *, action_name="acts_on", stale=False):
        self.action_name = action_name
        self.stale = stale
        self.project_reads = 0
        self.schema_ready_checks = 0
        self.runs = []
        self.opportunities = []
        self.classification_attempts = []
        self.decisions = []
        self.validations = []
        self.verification_attempts = []
        self.normalization = None
        self.materialization_decisions = []
        self.finished = []
        self.endpoints = []
        cases = [
            ("NO_CASE", "Organization", "Product", "no-left", "no-right"),
            ("OTHER_CASE", "Organization", "Product", "other-left", "other-right"),
            (
                "VERIFIER_REJECT_CASE", "Organization", "Product",
                "reject-left", "reject-right",
            ),
            (
                "VERIFIER_UNCERTAIN_CASE", "Organization", "Product",
                "uncertain-left", "uncertain-right",
            ),
            ("INVERSE_CASE", "Part", "Container", "inverse-part", "inverse-box"),
            (
                "SYMMETRIC_CASE", "Organization", "Product",
                "symmetric-left", "symmetric-right",
            ),
            (
                "MULTI_RELATION_CASE", "Organization", "Product",
                "multi-left", "multi-right",
            ),
            (
                "SELF_LOOP_CASE", "Organization", "Product",
                "self-left", "self-right",
            ),
            (
                "MULTI_EVIDENCE_CASE", "Organization", "Product",
                "evidence-left", "evidence-right",
            ),
            (
                "REVIEW_CASE", "Organization", "Product",
                "review-left", "review-right",
            ),
            (
                "POSITIVE_CASE", "Organization", "Product",
                "positive-left", "positive-right",
            ),
        ]
        self.chunks = []
        self.mentions = []
        self.assignments = []
        for index, (tag, left_type, right_type, left_key, right_key) in enumerate(
            cases, start=1,
        ):
            left_surface = f"Left{index}"
            right_surface = f"Right{index}"
            content = f"{tag} {left_surface} connects {right_surface}."
            left_start = content.index(left_surface)
            right_start = content.index(right_surface)
            self.chunks.append({
                "id": index,
                "document_id": 1,
                "revision_id": "revision-runtime",
                "chunk_key": f"chunk-{index}",
                "chunk_index": index,
                "content": content,
                "content_hash": f"chunk-hash-{index}",
            })
            same_canonical = tag == "SELF_LOOP_CASE"
            for key, surface, start, entity_type, canonical in (
                (
                    left_key, left_surface, left_start, left_type,
                    f"canonical-{index}-left",
                ),
                (
                    right_key, right_surface, right_start, right_type,
                    (
                        f"canonical-{index}-left"
                        if same_canonical else f"canonical-{index}-right"
                    ),
                ),
            ):
                self.mentions.append({
                    "mention_key": key,
                    "chunk_id": index,
                    "surface": surface,
                    "evidence_start": start,
                    "evidence_end": start + len(surface),
                })
                self.assignments.append({
                    "mention_key": key,
                    "canonical_key": canonical,
                    "entity_type": entity_type,
                    "policy_version": "assignment-v1",
                    "revision_id": "revision-runtime",
                })

    def assert_shadow_relation_v2_ready(self):
        self.schema_ready_checks += 1
        return {"migrations": ["0014", "0015", "0016", "0017", "0018", "0019"]}

    def get_project(self, project_id):
        assert project_id == 1
        self.project_reads += 1
        return {
            "id": 1,
            "project_key": "runtime-project",
            "graph_generation": 2 if self.stale and self.project_reads > 1 else 1,
            "llm_model": "scripted-model",
            "model_max_tokens": 32768,
        }

    @staticmethod
    def list_documents(project_id):
        assert project_id == 1
        return [{
            "id": 1,
            "project_id": 1,
            "status": "indexed",
            "active_revision_id": "revision-runtime",
            "doc_hash": "document-hash-runtime",
            "is_deleted": False,
        }]

    def list_document_chunks(self, document_id):
        assert document_id == 1
        return deepcopy(self.chunks)

    def get_active_schema(self, project_id):
        assert project_id == 1
        return {"version": 1, "schema_json": _schema(self.action_name)}

    def list_entity_shadow_mentions(self, namespace_id):
        assert namespace_id == "namespace-runtime"
        return deepcopy(self.mentions)

    def load_entity_shadow_snapshot(self, namespace_id):
        assert namespace_id == "namespace-runtime"
        return {
            "namespace": {"id": namespace_id, "config_hash": "entity-config"},
            "assignments": deepcopy(self.assignments),
            "canonicals": [],
            "relation_endpoints": [],
        }

    def start_shadow_relation_v2_run(self, values):
        self.runs.append(values)
        return values

    def persist_shadow_relation_v2_opportunities(self, _run_id, values):
        self.opportunities = list(values)
        return []

    def record_shadow_relation_v2_classification_attempt(
        self, run_id, pair_id, values,
    ):
        self.classification_attempts.append((run_id, pair_id, values))
        return values

    def persist_shadow_relation_v2_classifications(self, _run_id, values):
        self.decisions.extend(values)
        return []

    def persist_shadow_relation_v2_validations(self, _run_id, values):
        self.validations.extend(values)
        return []

    def persist_shadow_relation_v2_verification_attempts(self, _run_id, values):
        self.verification_attempts.extend(values)
        return []

    def persist_shadow_relation_v2_proposed_hypotheses(
        self, _run_id, values, *, candidate_source,
    ):
        self.proposed_hypotheses = list(getattr(self, "proposed_hypotheses", []))
        self.proposed_hypotheses.extend(values)
        self.proposed_candidate_source = candidate_source
        return []

    def persist_shadow_relation_v2_normalization(
        self, _run_id, result, *, input_fingerprint, materialization_decisions,
    ):
        assert input_fingerprint
        self.normalization = result
        self.materialization_decisions = list(materialization_decisions)
        return {}

    def finish_shadow_relation_v2_run(self, run_id, **values):
        self.finished.append((run_id, values))
        return values

    def persist_entity_shadow_relation_endpoints(self, namespace_id, rows):
        assert namespace_id == "namespace-runtime"
        self.endpoints.extend(rows)


def _keep_runtime_chunks(database: _RuntimeDatabase, chunk_ids: set[int]) -> None:
    database.chunks = [
        item for item in database.chunks if int(item["id"]) in chunk_ids
    ]
    database.mentions = [
        item for item in database.mentions
        if int(item["chunk_id"]) in chunk_ids
    ]
    mention_keys = {str(item["mention_key"]) for item in database.mentions}
    database.assignments = [
        item for item in database.assignments
        if str(item["mention_key"]) in mention_keys
    ]


def test_runtime_executes_every_v2_stage_and_never_treats_classified_as_verified():
    database = _RuntimeDatabase()
    gateway = _ScriptedGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert database.schema_ready_checks == 1
    assert len(database.opportunities) == 11
    assert metrics["pair_statuses"]["REJECTED_SELF_LOOP"] == 1
    assert metrics["classification_decisions"] == {
        HYPOTHESES: 8,
        NO_RELATION: 1,
        OTHER_UNSUPPORTED: 1,
    }
    assert metrics["pair_decision_statuses"] == {
        "EXPLICIT_NO_RELATION": 1,
        "HYPOTHESES_EMITTED": 8,
        "SCHEMA_GAP": 1,
    }
    assert metrics["classified_hypotheses"] == 9
    assert metrics["verification_verdicts"] == {
        REJECTED: 1,
        SUPPORTED: 6,
        UNCERTAIN: 2,
    }
    assert metrics["verifier_supported"] == 6
    assert metrics["facts_after_normalization"] == 6
    assert metrics["facts_materialization_eligible"] == 6
    assert metrics["legacy_pipeline_calls"] == 0
    assert len(database.endpoints) == 6
    mention_keys = {str(item["mention_key"]) for item in database.mentions}
    assert all(row["source_mention_key"] in mention_keys for row in database.endpoints)
    assert all(row["target_mention_key"] in mention_keys for row in database.endpoints)
    assert all(
        not str(row["source_mention_key"]).startswith("occurrence_")
        and not str(row["target_mention_key"]).startswith("occurrence_")
        for row in database.endpoints
    )
    assert all(
        row["metadata"]["source_occurrence_key"]
        and row["metadata"]["target_occurrence_key"]
        for row in database.endpoints
    )
    assert all(
        row["metadata"]["verification_status"] == "VERIFIER_SUPPORTED"
        for row in database.endpoints
    )
    assert all(
        evidence.verifier_supported
        for evidence in database.normalization.evidence
    )
    inverse = next(
        fact for fact in database.normalization.facts
        if fact.predicate == "contains"
    )
    assert inverse.subject_canonical_id == "canonical-5-right"
    assert inverse.object_canonical_id == "canonical-5-left"
    assert any(fact.symmetric for fact in database.normalization.facts)
    assert any(len(item.evidence_spans) == 2
               for item in database.normalization.evidence)


def test_runtime_predicate_rename_with_same_metadata_keeps_stage_behavior():
    outputs = []
    for name in ("acts_on", "renamed_action"):
        database = _RuntimeDatabase(action_name=name)
        result = ShadowRelationV2Runtime(
            database, _ScriptedGateway(action_name=name),
        ).run_namespace(
            namespace_id="namespace-runtime",
            project_id=1,
        )
        outputs.append((
            result["metrics"]["classification_decisions"],
            result["metrics"]["verification_verdicts"],
            result["metrics"]["facts_after_normalization"],
            result["metrics"]["facts_materialization_eligible"],
        ))
        assert any(
            fact.predicate == name for fact in database.normalization.facts
        )
    assert outputs[0] == outputs[1]


def test_runtime_stale_fingerprint_fails_closed_without_materialization():
    database = _RuntimeDatabase(stale=True)

    with pytest.raises(StaleRelationSnapshotError):
        ShadowRelationV2Runtime(
            database, _ScriptedGateway(),
        ).run_namespace(
            namespace_id="namespace-runtime",
            project_id=1,
        )

    assert database.endpoints == []
    assert database.finished[-1][1]["status"] == "stale"


def test_runtime_classifier_failure_does_not_fallback_or_materialize():
    database = _RuntimeDatabase()

    with pytest.raises(RuntimeError, match="禁止回退旧流程"):
        ShadowRelationV2Runtime(
            database, _ScriptedGateway(fail_classifier=True),
        ).run_namespace(
            namespace_id="namespace-runtime",
            project_id=1,
        )

    assert database.endpoints == []
    assert database.finished[-1][1]["status"] == "failed"


def test_runtime_excludes_quarantined_canonicals_from_pair_planner():
    class QuarantinedDatabase(_RuntimeDatabase):
        def load_entity_shadow_snapshot(self, namespace_id):
            snapshot = super().load_entity_shadow_snapshot(namespace_id)
            snapshot["canonicals"] = [
                {
                    "canonical_key": item["canonical_key"],
                    "metadata": {"quarantined": True},
                }
                for item in snapshot["assignments"]
            ]
            return snapshot

    database = QuarantinedDatabase()
    gateway = _ScriptedGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["pair_opportunities_total"] == 0
    assert result["metrics"]["pair_opportunities_classified"] == 0
    assert gateway.classifier_calls == 0
    assert gateway.verifier_calls == 0
    assert database.endpoints == []
    assert database.finished[-1][1]["status"] == "ready"


def test_runtime_direct_proposer_enters_verifier_even_when_classifier_says_no_relation():
    class ProposerPositiveClassifierNegativeGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                left, right = payload["occurrences"]
                return {
                    "proposals": [{
                        "subject_occurrence_id": left["occurrence_id"],
                        "predicate": self.action_name,
                        "object_occurrence_id": right["occurrence_id"],
                        "evidence_spans": [{
                            "start": 0,
                            "end": len(payload["context"]["text"]),
                        }],
                        "qualifiers": [],
                        "confidence": 0.77,
                    }],
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            if "classifier" in system:
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CLASSIFIER_NO_RELATION",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                } for pair in payload["pairs"]]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = ProposerPositiveClassifierNegativeGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["classification_decisions"] == {NO_RELATION: 1}
    assert result["metrics"]["proposer_model_calls"] == 1
    assert result["metrics"]["proposals_total"] == 1
    assert result["metrics"]["proposer_hypotheses_persisted"] == 1
    assert result["metrics"]["verifier_supported"] == 1
    assert len(database.proposed_hypotheses) == 1
    assert database.proposed_candidate_source == "DIRECT_PROPOSER"
    assert len(database.endpoints) == 1


def test_runtime_proposer_truncation_does_not_abort_v2_main_chain(monkeypatch):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 2)

    class TruncatingProposerGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_calls = 0
            self.proposer_output_budgets = []

        def json_chat(self, **kwargs):
            if "proposer" in str(kwargs["system"]):
                self.proposer_calls += 1
                self.proposer_output_budgets.append(
                    int(kwargs["max_output_tokens"]),
                )
                raise OllamaProtocolError(
                    "Ollama response was truncated by output length",
                )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {11})
    gateway = TruncatingProposerGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["proposer_model_calls"] == 2
    assert result["metrics"]["proposer_failures"] == 2
    assert result["metrics"]["proposals_total"] == 0
    assert gateway.proposer_output_budgets == [2048, 4096]
    assert gateway.classifier_calls == 1
    assert gateway.verifier_calls == 1
    assert result["metrics"]["verifier_supported"] == 1
    assert database.finished[-1][1]["status"] == "ready"
    assert len(database.endpoints) == 1


def test_runtime_proposer_failure_circuit_breaker_skips_auxiliary_calls(monkeypatch):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 1)
    monkeypatch.setattr(Config, "relation_proposer_failure_circuit_breaker", 1)

    class FailingProposerGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_calls = 0

        def json_chat(self, **kwargs):
            if "proposer" in str(kwargs["system"]):
                self.proposer_calls += 1
                raise OllamaProtocolError(
                    "Ollama response was truncated by output length",
                )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1, 11})
    gateway = FailingProposerGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.proposer_calls == 1
    assert result["metrics"]["proposer_circuit_breaker_tripped"] is True
    assert result["metrics"]["proposer_skipped_by_circuit"] == 1
    assert gateway.classifier_calls == 2
    assert gateway.verifier_calls == 1
    assert result["metrics"]["verifier_supported"] == 1
    assert database.finished[-1][1]["status"] == "ready"


def test_runtime_no_relation_without_evidence_span_is_a_negative_decision():
    class NoRelationWithoutEvidenceGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                pair = json.loads(kwargs["prompt"])["pairs"][0]
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "NO_SEMANTIC_RELATION",
                }]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = NoRelationWithoutEvidenceGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["classification_decisions"] == {NO_RELATION: 1}
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"
    assert len(database.endpoints) == 0


def test_runtime_invalid_single_pair_hypothesis_becomes_unsupported_not_failure(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 2)

    class InvalidPredicateGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                pair = json.loads(kwargs["prompt"])["pairs"][0]
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": HYPOTHESES,
                    "hypotheses": [{
                        "subject_occurrence_id": "LEFT",
                        "predicate": "funded_by",
                        "object_occurrence_id": "RIGHT",
                        "evidence_spans": [{
                            "start": pair["context_span"][0],
                            "end": pair["context_span"][1],
                        }],
                        "classifier_confidence": 0.95,
                    }],
                }]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {11})
    gateway = InvalidPredicateGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 1
    assert result["metrics"]["classifier_failures"] == 0
    assert result["metrics"]["classification_decisions"] == {
        OTHER_UNSUPPORTED: 1,
    }
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"
    assert len(database.endpoints) == 0
    accepted = [
        values for _, _, values in database.classification_attempts
        if values["status"] == "ACCEPTED"
    ]
    assert len(accepted) == 1
    assert accepted[0]["independent_call"] is True


def test_runtime_recovers_single_pair_negative_raw_when_parser_reports_missing(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 1)

    class RawNegativeParserFailureGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                pair = json.loads(kwargs["prompt"])["pairs"][0]
                self.last_raw_response_text = json.dumps({
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "NO_SEMANTIC_RELATION",
                })
                raise ShadowClassificationError(
                    f"missing pair ids: ['{pair['pair_id']}']",
                )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = RawNegativeParserFailureGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 1
    assert result["metrics"]["classification_decisions"] == {NO_RELATION: 1}
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"
    accepted = [
        values for _, _, values in database.classification_attempts
        if values["status"] == "ACCEPTED"
    ]
    assert len(accepted) == 1
    assert accepted[0]["independent_call"] is False


def test_runtime_classifier_truncation_retry_expands_output_budget(monkeypatch):
    monkeypatch.setattr(Config, "relation_classifier_output_reserve", 2048)
    monkeypatch.setattr(Config, "relation_shadow_max_attempts", 3)

    class TruncatingOnceGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.classifier_output_budgets = []

        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_output_budgets.append(
                    int(kwargs["max_output_tokens"]),
                )
                if len(self.classifier_output_budgets) == 1:
                    raise OllamaProtocolError(
                        "Ollama response was truncated by output length",
                    )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    gateway = TruncatingOnceGateway()

    result = ShadowRelationV2Runtime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["classifier_failures"] == 1
    assert gateway.classifier_output_budgets[:2] == [2048, 4096]
    retry = next(
        values for _, _, values in database.classification_attempts
        if values["status"] == "RETRYABLE_TRUNCATED"
    )
    accepted_retry = next(
        values for _, _, values in database.classification_attempts
        if (
            values["status"] == "ACCEPTED"
            and values["metadata"]["max_output_tokens"] == 4096
        )
    )
    assert retry["error_message"] == "Ollama response was truncated by output length"
    assert retry["metadata"]["max_output_tokens"] == 2048
    assert accepted_retry["metadata"]["max_output_tokens"] == 4096
    assert database.finished[-1][1]["status"] == "ready"


def test_single_pair_classifier_object_is_only_shape_normalized_not_reinterpreted():
    class SingleObjectGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {
                "pair_id": "pair-1",
                "decision": NO_RELATION,
                "hypotheses": [],
            }

    runtime = ShadowRelationV2Runtime(None, SingleObjectGateway())
    result = runtime._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-1",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_single_pair_classifier_identity_is_not_bound_by_caller(monkeypatch):
    class SingleObjectGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {
                "decision": NO_RELATION,
                "hypotheses": [],
            }

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    runtime = ShadowRelationV2Runtime(None, SingleObjectGateway())
    result = runtime._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-from-caller")],
        content="",
        schema=_schema(),
    )

    assert result == {
        "decision": NO_RELATION,
        "hypotheses": [],
    }


def test_multi_pair_classifier_identity_is_never_inferred(monkeypatch):
    raw = {
        "decision": NO_RELATION,
        "hypotheses": [],
    }

    class AmbiguousObjectGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return raw

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    runtime = ShadowRelationV2Runtime(None, AmbiguousObjectGateway())
    result = runtime._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[
            SimpleNamespace(pair_identity="pair-1"),
            SimpleNamespace(pair_identity="pair-2"),
        ],
        content="",
        schema=_schema(),
    )

    assert result is raw


@pytest.mark.parametrize("hypotheses", ["missing", None])
def test_negative_classifier_null_hypotheses_is_normalized_to_empty(
    monkeypatch, hypotheses,
):
    decision = {
        "pair_id": "pair-1",
        "decision": NO_RELATION,
    }
    if hypotheses is None:
        decision["hypotheses"] = None

    class NegativeGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {"items": [decision]}

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = ShadowRelationV2Runtime(None, NegativeGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-1")],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-1",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_classifier_result_wrapper_is_shape_normalized(monkeypatch):
    class ResultWrapperGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {
                "result": [{
                    "pair_id": "pair-1",
                    "decision": NO_RELATION,
                    "hypotheses": [],
                }],
            }

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = ShadowRelationV2Runtime(None, ResultWrapperGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-1")],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-1",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_single_pair_classifier_pair_id_drift_is_not_overwritten(monkeypatch):
    class DriftedPairGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return [{
                "pair_id": "pair-drifted",
                "decision": NO_RELATION,
                "hypotheses": [],
            }]

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = ShadowRelationV2Runtime(None, DriftedPairGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-from-caller")],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-drifted",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_multi_pair_classifier_pair_id_drift_is_not_bound(monkeypatch):
    raw = [{
        "pair_id": "pair-drifted",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]

    class DriftedPairGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return raw

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = ShadowRelationV2Runtime(None, DriftedPairGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[
            SimpleNamespace(pair_identity="pair-1"),
            SimpleNamespace(pair_identity="pair-2"),
        ],
        content="",
        schema=_schema(),
    )

    assert result is not raw
    assert result == raw


def test_single_pair_empty_classifier_items_is_not_negative(monkeypatch):
    class EmptyItemsGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {"items": []}

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = ShadowRelationV2Runtime(None, EmptyItemsGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-from-caller")],
        content="",
        schema=_schema(),
    )

    assert result == []


def test_multi_pair_empty_classifier_items_is_not_negative(monkeypatch):
    class EmptyItemsGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {"items": []}

    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_shadow_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = ShadowRelationV2Runtime(None, EmptyItemsGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[
            SimpleNamespace(pair_identity="pair-1"),
            SimpleNamespace(pair_identity="pair-2"),
        ],
        content="",
        schema=_schema(),
    )

    assert result == []
