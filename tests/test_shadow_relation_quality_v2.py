from __future__ import annotations

from knowledge_forge.quality import (
    canonical_relation_fact_key,
    evaluate_relations,
    evaluate_shadow_relation_quality,
    occurrence_relation_assertion_key,
    relation_layer_funnel,
)


def _fact(**overrides):
    value = {
        "project_scope": "project-12",
        "subject_canonical_id": "entity-a",
        "predicate": "uses",
        "object_canonical_id": "entity-b",
        "direction": "forward",
        "document_id": 16,
        "revision_id": "revision-current",
        "chunk_id": 6117,
        "subject_mention_id": "mention-a",
        "subject_start": 4,
        "subject_end": 5,
        "object_mention_id": "mention-b",
        "object_start": 11,
        "object_end": 12,
        "evidence_start": 0,
        "evidence_end": 13,
        "snapshot_id": "snapshot-a",
        "policy_hash": "policy-a",
    }
    value.update(overrides)
    return value


def test_relation_funnel_never_counts_classified_hypotheses_as_verified() -> None:
    hypotheses = [
        {"hypothesis_id": f"h-{index}", "rule_valid": True}
        for index in range(10)
    ]
    attempts = [
        {
            "hypothesis_id": f"h-{index}",
            "verdict": "SUPPORTED" if index < 2 else "REJECTED",
        }
        for index in range(10)
    ]

    report = relation_layer_funnel(
        pair_opportunities=[],
        hypotheses=hypotheses,
        verification_attempts=attempts,
        normalized_facts=[],
        published_facts=[],
    )

    assert report["hypotheses_total"] == 10
    assert report["hypotheses_rule_valid"] == 10
    assert report["hypotheses_verifier_supported"] == 2
    assert report["hypotheses_verifier_rejected"] == 8
    assert report["facts_materialization_eligible"] == 0
    assert "verified" not in report


def test_relation_funnel_counts_v2_pair_terminals_without_hiding_failures() -> None:
    report = relation_layer_funnel(
        pair_opportunities=[
            {"status": "HYPOTHESES"},
            {"status": "NO_RELATION"},
            {"status": "OTHER_UNSUPPORTED"},
            {"status": "DEFERRED_BUDGET"},
            {"status": "CLASSIFICATION_RETRYABLE_FAILED"},
            {"status": "CLASSIFICATION_PERMANENT_FAILED"},
        ],
        hypotheses=[],
        verification_attempts=[],
        normalized_facts=[],
        published_facts=[],
    )

    assert report["pair_opportunities_classified"] == 3
    assert report["pair_deferred_budget"] == 1
    assert report["pair_failed"] == 2


def test_canonical_fact_identity_ignores_snapshot_policy_run_and_internal_ids() -> None:
    left = _fact()
    right = _fact(
        snapshot_id="snapshot-b",
        policy_hash="policy-b",
        run_id="run-b",
        candidate_id="candidate-b",
        id=999,
        evidence_start=20,
        evidence_end=30,
    )

    assert canonical_relation_fact_key(left) == canonical_relation_fact_key(right)
    assert occurrence_relation_assertion_key(left) != occurrence_relation_assertion_key(right)


def test_occurrence_identity_uses_explicit_document_revision_id() -> None:
    left = _fact(revision_id="", document_revision_id="revision-a")
    right = _fact(revision_id="", document_revision_id="revision-b")

    assert canonical_relation_fact_key(left) == canonical_relation_fact_key(right)
    assert occurrence_relation_assertion_key(left) != occurrence_relation_assertion_key(right)


def test_relation_evaluation_uses_multisets_and_exposes_exact_duplicates() -> None:
    duplicate = _fact()
    report = evaluate_shadow_relation_quality(
        gold=[duplicate],
        predicted=[duplicate, dict(duplicate)],
        snapshot_current=True,
    )

    assert report["canonical"]["predicted"] == 2
    assert report["canonical"]["unique_predicted"] == 1
    assert report["canonical"]["exact_duplicates"] == 1
    assert report["canonical"]["true_positive"] == 1
    assert report["canonical"]["false_positive"] == 1


def test_reverse_predictions_are_not_hidden_by_set_conversion() -> None:
    forward = _fact()
    reverse = _fact(
        subject_canonical_id="entity-b",
        object_canonical_id="entity-a",
        subject_mention_id="mention-b",
        object_mention_id="mention-a",
    )

    report = evaluate_shadow_relation_quality(
        gold=[forward],
        predicted=[forward, reverse],
        snapshot_current=True,
    )

    assert report["canonical"]["predicted"] == 2
    assert report["canonical"]["unique_predicted"] == 2
    assert report["canonical"]["false_positive"] == 1


def test_recall_is_unavailable_without_human_gold() -> None:
    report = evaluate_shadow_relation_quality(
        gold=None,
        predicted=[_fact()],
        snapshot_current=True,
    )

    assert report["canonical"]["recall"] is None
    assert report["canonical"]["precision"] is None
    assert report["gold_status"] == "unavailable"
    assert report["production_eligible"] is False
    assert "human_gold_unavailable" in report["blocking_reasons"]


def test_stale_snapshot_cannot_supply_production_evidence() -> None:
    report = evaluate_shadow_relation_quality(
        gold=[_fact()],
        predicted=[_fact()],
        snapshot_current=False,
    )

    assert report["canonical"]["recall"] == 1.0
    assert report["production_eligible"] is False
    assert "stale_snapshot" in report["blocking_reasons"]


def test_primary_relation_evaluator_uses_occurrence_multisets_across_snapshots() -> None:
    gold = _fact(snapshot_id="gold-snapshot", policy_hash="gold-policy")
    same_fact = _fact(snapshot_id="treatment-snapshot", policy_hash="treatment-policy")

    report = evaluate_relations([gold], [same_fact, dict(same_fact)])

    assert report["strict"]["predicted"] == 2
    assert report["strict"]["true_positive"] == 1
    assert report["strict"]["false_positive"] == 1
    assert report["strict"]["exact_duplicates"] == 1
    assert report["canonical"]["true_positive"] == 1
