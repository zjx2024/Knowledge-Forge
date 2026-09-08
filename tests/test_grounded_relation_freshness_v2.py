from __future__ import annotations

from copy import deepcopy
import inspect

import pytest

from knowledge_forge.database import Database
from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.relation_extraction import (
    EvidenceSpan,
    NormalizedRelationFact,
    RelationFactEvidence,
    StaleRelationSnapshotError,
    assert_relation_fresh,
    build_relation_materialization_fingerprint,
    compare_relation_fingerprints,
    relation_materialization_decision,
)
from knowledge_forge.service import KnowledgeForgeService


def _components():
    return {
        "source_generation": 31,
        "documents": [{
            "document_id": 16,
            "revision_id": "revision-1",
            "content_hash": "a" * 64,
            "chunks": [
                {"chunk_id": 6117, "content_hash": "b" * 64},
                {"chunk_id": 6118, "content_hash": "c" * 64},
            ],
        }],
        "mention_occurrences": [{
            "occurrence_key": "occ-a",
            "span": [0, 5],
            "content_hash": "d" * 64,
        }],
        "entity_assignment": {
            "snapshot_id": "entity-snapshot-1",
            "version": "assignment-v1",
            "hash": "e" * 64,
        },
        "schema": {
            "version": 7,
            "hash": "f" * 64,
            "predicate_metadata_hash": "1" * 64,
        },
        "candidate": {
            "policy_version": "candidate-v2",
            "parameters": {"sentence_window": 1, "max_char_distance": 800},
            "budget": 256,
        },
        "classifier": {
            "prompt_version": "classifier-prompt-v2",
            "schema_version": "classifier-schema-v2",
            "parser_version": "classifier-parser-v2",
            "model": "qwen3.6:latest",
            "model_revision": "classifier-model-r1",
        },
        "proposer": {
            "prompt_version": "proposer-prompt-v1",
            "schema_version": "proposer-schema-v1",
            "parser_version": "proposer-parser-v1",
            "model": "qwen3.6:latest",
            "model_revision": "proposer-model-r1",
            "max_gleaning_rounds": 2,
        },
        "verifier": {
            "prompt_version": "verifier-prompt-v2",
            "schema_version": "verifier-schema-v2",
            "parser_version": "verifier-parser-v2",
            "model": "qwen3.6:latest",
            "model_revision": "verifier-model-r1",
        },
        "thresholds": {
            "classifier_confidence_analysis_only": 0.5,
            "max_attempts": 3,
        },
        "normalization": {
            "version": "normalization-v2",
            "symmetric_rule": "sort-canonical-endpoints",
            "inverse_rule": "declared-canonical-forward-only",
        },
        "publish_policy": {
            "version": "publish-v2",
            "mode_source": "versioned-schema-publish_mode",
            "unsupported": "never-publish",
        },
        "quality_gate": {
            "version": "relation-release-v2",
            "relation_precision_noninferiority_delta": 0.01,
            "confidence_method": "wilson-95",
            "minimum_gold_facts": 200,
        },
        "materialization_root": {
            "code_tree": "2" * 40,
            "policy_root": "3" * 64,
        },
    }


def _fingerprint(components=None):
    return build_relation_materialization_fingerprint(**(components or _components()))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("source_generation",), 32),
        (("documents", 0, "chunks", 0, "content_hash"), "9" * 64),
        (("entity_assignment", "version"), "assignment-v2"),
        (("schema", "predicate_metadata_hash"), "8" * 64),
        (("proposer", "prompt_version"), "proposer-prompt-v2"),
        (("proposer", "max_gleaning_rounds"), 1),
        (("classifier", "prompt_version"), "classifier-prompt-v3"),
        (("classifier", "model_revision"), "classifier-model-r2"),
        (("verifier", "prompt_version"), "verifier-prompt-v3"),
        (("verifier", "model_revision"), "verifier-model-r2"),
        (("candidate", "budget"), 128),
        (("thresholds", "max_attempts"), 4),
        (("normalization", "inverse_rule"), "changed"),
        (("materialization_root", "policy_root"), "7" * 64),
    ],
)
def test_every_behavioral_fingerprint_change_makes_snapshot_stale(path, value) -> None:
    recorded = _fingerprint()
    changed = deepcopy(_components())
    target = changed
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    current = _fingerprint(changed)

    decision = compare_relation_fingerprints(
        recorded, current, checkpoint="materialize",
    )

    assert decision.current is False
    assert decision.status == "STALE"
    assert decision.changed_components


@pytest.mark.parametrize(
    "checkpoint",
    [
        "evaluate",
        "quality_gate",
        "snapshot_completion",
        "activate",
        "materialize",
        "database_cas",
        "graph_rebuild_input",
    ],
)
def test_all_freshness_checkpoints_fail_closed(checkpoint) -> None:
    recorded = _fingerprint()
    changed = deepcopy(_components())
    changed["source_generation"] += 1

    with pytest.raises(StaleRelationSnapshotError, match=checkpoint):
        assert_relation_fresh(recorded, _fingerprint(changed), checkpoint=checkpoint)


def _fact(**overrides):
    values = {
        "fact_id": "fact-1",
        "semantic_fact_key": "fact-1",
        "project_id": 12,
        "subject_canonical_id": "entity-a",
        "predicate": "uses",
        "object_canonical_id": "entity-b",
        "symmetric": False,
        "reflexive": False,
        "review_only": False,
        "normalization_eligible": True,
        "evidence_count": 1,
    }
    values.update(overrides)
    return NormalizedRelationFact(**values)


def _evidence(**overrides):
    values = {
        "evidence_id": "evidence-1",
        "evidence_key": "evidence-1",
        "fact_id": "fact-1",
        "document_revision_id": "revision-1",
        "subject_occurrence_key": "occ-a",
        "object_occurrence_key": "occ-b",
        "evidence_spans": (EvidenceSpan(0, 20),),
        "hypothesis_id": "hypothesis-1",
        "verification_attempt_id": "attempt-1",
        "verifier_supported": True,
        "provenance": {},
    }
    values.update(overrides)
    return RelationFactEvidence(**values)


def test_only_complete_fresh_supported_specific_fact_is_materialization_eligible() -> None:
    fingerprint = _fingerprint()
    decision = relation_materialization_decision(
        _fact(),
        [_evidence()],
        recorded_fingerprint=fingerprint,
        current_fingerprint=fingerprint,
        snapshot_complete=True,
        current_assignment_and_schema=True,
        conflicts=[],
    )

    assert decision.eligible is True
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("fact", "evidence", "kwargs", "reason"),
    [
        (_fact(predicate="related_to", review_only=True), [_evidence()], {}, "PREDICATE_REVIEW_ONLY"),
        (_fact(review_only=True), [_evidence()], {}, "PREDICATE_REVIEW_ONLY"),
        (_fact(), [_evidence(verifier_supported=False)], {}, "VERIFIER_EVIDENCE_MISSING"),
        (_fact(), [], {}, "VERIFIER_EVIDENCE_MISSING"),
        (_fact(), [_evidence()], {"snapshot_complete": False}, "SNAPSHOT_INCOMPLETE"),
        (
            _fact(),
            [_evidence()],
            {"current_assignment_and_schema": False},
            "ASSIGNMENT_OR_SCHEMA_STALE",
        ),
        (_fact(), [_evidence()], {"conflicts": ["conflict-1"]}, "UNRESOLVED_CONFLICT"),
    ],
)
def test_materialization_predicate_fails_closed(fact, evidence, kwargs, reason) -> None:
    fingerprint = _fingerprint()
    values = {
        "recorded_fingerprint": fingerprint,
        "current_fingerprint": fingerprint,
        "snapshot_complete": True,
        "current_assignment_and_schema": True,
        "conflicts": [],
    }
    values.update(kwargs)

    decision = relation_materialization_decision(fact, evidence, **values)

    assert decision.eligible is False
    assert reason in decision.reasons


def test_stale_fact_cannot_materialize_even_if_other_checks_pass() -> None:
    recorded = _fingerprint()
    changed = deepcopy(_components())
    changed["source_generation"] += 1
    decision = relation_materialization_decision(
        _fact(),
        [_evidence()],
        recorded_fingerprint=recorded,
        current_fingerprint=_fingerprint(changed),
        snapshot_complete=True,
        current_assignment_and_schema=True,
        conflicts=[],
    )

    assert decision.eligible is False
    assert "STALE_FINGERPRINT" in decision.reasons


def test_database_and_service_activation_paths_require_fresh_materialization() -> None:
    database_source = inspect.getsource(Database._activate_policy_snapshot_in_connection)
    service_source = inspect.getsource(KnowledgeForgeService.activate_treatment_snapshot)

    assert "materialization_fingerprint" in database_source
    assert "current_fingerprint" in database_source
    assert "禁止激活" in service_source


def test_grounded_database_context_blocks_product_and_graph_state_writes() -> None:
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0017_shadow_relation_v2_freshness"
    )

    for table in (
        "relation_assertions",
        "canonical_relation_projections",
        "vector_embeddings",
        "indexing_failures",
        "entity_mentions",
        "entity_assignments",
        "canonical_entity_aliases",
        "projects",
    ):
        assert f" ON {table}" in migration.sql
    assert "knowledge_forge.shadow_relation_v2_run_id" in migration.sql
    assert "shadow_relation_current_fingerprints" in migration.sql
