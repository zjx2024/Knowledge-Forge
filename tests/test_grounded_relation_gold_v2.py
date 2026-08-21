from __future__ import annotations

from copy import deepcopy

import pytest

from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.quality import (
    GoldManifestError,
    evaluate_relation_gold_run,
    evaluate_relation_release_gate_v2,
    validate_relation_gold,
)


def _manifest(**overrides):
    value = {
        "dataset": "relation-pilot",
        "version": "v1",
        "level": "release",
        "frozen_at": "2026-07-24T00:00:00Z",
        "holdout_content_hash": "a" * 64,
        "recall_universe_complete": True,
        "precision_adjudication_universe": "control-union-treatment-v1",
        "trusted_approval": {
            "approval_id": "approval-1",
            "approved_at": "2026-07-24T00:00:00Z",
            "signers": ["reviewer-a", "reviewer-b"],
        },
    }
    value.update(overrides)
    return value


def _pair(pair_id, decision, *, split="holdout", label_source="human"):
    return {
        "pair_id": pair_id,
        "dataset": "relation-pilot",
        "dataset_version": "v1",
        "split": split,
        "document_revision_id": "revision-1",
        "document_content_hash": "b" * 64,
        "left_occurrence_id": f"{pair_id}-left",
        "left_span": [0, 5],
        "left_canonical_id": f"{pair_id}-entity-left",
        "right_occurrence_id": f"{pair_id}-right",
        "right_span": [10, 15],
        "right_canonical_id": f"{pair_id}-entity-right",
        "decision": decision,
        "evidence_spans": [[0, 16]] if decision != "NO_RELATION" else [],
        "critical": False,
        "label_source": label_source,
        "annotator": "annotator-a",
        "reviewer": "reviewer-b",
        "reviewed_at": "2026-07-24T00:00:00Z",
        "adjudication_status": "adjudicated",
    }


def _fact(pair_id="pair-positive", *, predicate="uses", evidence=None):
    return {
        "pair_id": pair_id,
        "project_scope": "project-12",
        "subject_occurrence_id": f"{pair_id}-left",
        "subject_canonical_id": f"{pair_id}-entity-left",
        "predicate": predicate,
        "object_occurrence_id": f"{pair_id}-right",
        "object_canonical_id": f"{pair_id}-entity-right",
        "direction": "forward",
        "document_id": 16,
        "revision_id": "revision-1",
        "chunk_id": 6117,
        "subject_start": 0,
        "subject_end": 5,
        "object_start": 10,
        "object_end": 15,
        "evidence_start": 0,
        "evidence_end": 16,
        "evidence_text": evidence or "Alpha uses Beta.",
        "critical": False,
        "adjudication_status": "adjudicated",
    }


def _gold():
    pairs = [
        _pair("pair-positive", "HAS_RELATION"),
        _pair("pair-negative", "NO_RELATION"),
        _pair("pair-schema-gap", "OTHER_UNSUPPORTED"),
    ]
    return pairs, [_fact()]


def _run(*, facts=None, prediction_adjudications=None):
    facts = list(facts if facts is not None else [dict(_fact(), prediction_id="pred-1")])
    return {
        "mention_occurrences": [
            {"occurrence_key": "pair-positive-left"},
            {"occurrence_key": "pair-positive-right"},
        ],
        "pair_opportunities": [{"pair_identity": value["pair_id"]} for value in _gold()[0]],
        "pair_decisions": [
            {"pair_identity": "pair-positive", "decision": "HYPOTHESES"},
            {"pair_identity": "pair-negative", "decision": "NO_RELATION"},
            {"pair_identity": "pair-schema-gap", "decision": "OTHER_UNSUPPORTED"},
        ],
        "hypotheses": [{
            "pair_identity": "pair-positive",
            "subject_occurrence_key": "pair-positive-left",
            "predicate": "uses",
            "object_occurrence_key": "pair-positive-right",
            "hypothesis_id": "hypothesis-1",
        }],
        "verification_attempts": [{
            "hypothesis_id": "hypothesis-1",
            "verdict": "SUPPORTED",
        }],
        "facts": facts,
        "prediction_adjudications": (
            prediction_adjudications
            if prediction_adjudications is not None
            else [{"prediction_id": "pred-1", "correct": True}]
        ),
        "normalization_metrics": {
            "exact_duplicates_collapsed": 0,
            "symmetric_inverse_duplicates_collapsed": 0,
            "unsupported_reverse_conflicts": 0,
            "verified_reciprocal_facts": 0,
            "illegal_self_loops": 0,
        },
        "cost": {
            "candidate_calls": 0,
            "classifier_calls": 1,
            "verifier_calls": 1,
            "tokens": 100,
            "latency_ms": 10,
            "failed_calls": 0,
        },
        "all_pairs_terminal": True,
        "stale": False,
        "cannot_link_violations": 0,
    }


def test_relation_gold_requires_independent_human_adjudication() -> None:
    pairs, facts = _gold()
    pairs[0] = _pair("pair-positive", "HAS_RELATION", label_source="model")

    with pytest.raises(GoldManifestError, match="human"):
        validate_relation_gold(_manifest(), pairs, facts)


@pytest.mark.parametrize(
    ("decision", "facts", "message"),
    [
        ("NO_RELATION", [_fact()], "NO_RELATION"),
        ("OTHER_UNSUPPORTED", [_fact()], "OTHER_UNSUPPORTED"),
        ("HAS_RELATION", [], "HAS_RELATION"),
    ],
)
def test_pair_and_fact_gold_decisions_are_structurally_consistent(
    decision, facts, message,
) -> None:
    pair = _pair("pair-positive", decision)
    with pytest.raises(GoldManifestError, match=message):
        validate_relation_gold(_manifest(), [pair], facts)


def test_valid_gold_freezes_pair_and_fact_hashes() -> None:
    pairs, facts = _gold()
    report = validate_relation_gold(_manifest(), pairs, facts)

    assert report["engineering_valid"] is True
    assert report["production_valid"] is True
    assert report["pair_gold_hash"]
    assert report["fact_gold_hash"]
    assert report["positive_facts"] == 1


def test_evaluator_refuses_recall_and_precision_without_human_gold() -> None:
    report = evaluate_relation_gold_run(
        manifest_report=None,
        pair_gold=None,
        fact_gold=None,
        run=_run(),
    )

    assert report["gold_status"] == "unavailable"
    assert report["strict"]["precision"] is None
    assert report["strict"]["recall"] is None
    assert report["production_eligible"] is False


def test_unadjudicated_prediction_makes_precision_unavailable() -> None:
    pairs, facts = _gold()
    manifest = validate_relation_gold(_manifest(), pairs, facts)
    report = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(prediction_adjudications=[]),
    )

    assert report["strict"]["recall"] == 1.0
    assert report["strict"]["precision"] is None
    assert report["precision_status"] == "unavailable"
    assert "unadjudicated_predictions" in report["blocking_reasons"]


def test_multiset_evaluation_does_not_hide_duplicate_predictions() -> None:
    pairs, facts = _gold()
    manifest = validate_relation_gold(_manifest(), pairs, facts)
    duplicate_predictions = [
        dict(_fact(), prediction_id="pred-1"),
        dict(_fact(), prediction_id="pred-2"),
    ]
    report = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(
            facts=duplicate_predictions,
            prediction_adjudications=[
                {"prediction_id": "pred-1", "correct": True},
                {"prediction_id": "pred-2", "correct": True},
            ],
        ),
    )

    assert report["strict"]["predicted"] == 2
    assert report["strict"]["true_positive"] == 1
    assert report["strict"]["precision"] == 0.5
    assert report["strict"]["exact_duplicates"] == 1


def test_layered_evaluator_reports_endpoint_pair_hypothesis_and_verifier_recall() -> None:
    pairs, facts = _gold()
    manifest = validate_relation_gold(_manifest(), pairs, facts)
    report = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(),
    )

    assert report["endpoint_recall"] == 1.0
    assert report["pair_coverage"] == 1.0
    assert report["hypothesis_recall"] == 1.0
    assert report["verifier_recall"] == 1.0
    assert report["strict"]["precision"] == report["strict"]["recall"] == 1.0
    assert report["direction_accuracy"] == 1.0
    assert report["exact_evidence_coverage"] == 1.0
    assert report["pair_decisions"]["NO_RELATION"]["recall"] == 1.0
    assert report["pair_decisions"]["OTHER_UNSUPPORTED"]["recall"] == 1.0


def test_related_to_published_is_a_hard_release_failure() -> None:
    pairs, facts = _gold()
    related = dict(_fact(predicate="related_to"), prediction_id="pred-1")
    manifest = validate_relation_gold(_manifest(), pairs, facts)
    treatment = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(
            facts=[related],
            prediction_adjudications=[{"prediction_id": "pred-1", "correct": False}],
        ),
    )
    control = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(),
    )
    gate = evaluate_relation_release_gate_v2(
        control=control,
        treatment=treatment,
        manifest_report=manifest,
        release_policy={
            "version": "release-v2",
            "relation_precision_noninferiority_delta": 0.01,
            "minimum_gold_facts": 1,
            "confidence_method": "wilson-95",
        },
    )

    assert treatment["related_to"]["published"] == 1
    assert gate["production_approved"] is False
    assert "related_to_published" in gate["failed_gates"]


def test_high_recall_low_precision_treatment_cannot_pass_gate() -> None:
    pairs, facts = _gold()
    manifest = validate_relation_gold(_manifest(), pairs, facts)
    control = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(),
    )
    noisy = [
        dict(_fact(), prediction_id="pred-1"),
        dict(
            _fact(
                pair_id="pair-negative",
                predicate="uses",
            ),
            prediction_id="pred-2",
        ),
    ]
    treatment = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(
            facts=noisy,
            prediction_adjudications=[
                {"prediction_id": "pred-1", "correct": True},
                {"prediction_id": "pred-2", "correct": False},
            ],
        ),
    )
    gate = evaluate_relation_release_gate_v2(
        control=control,
        treatment=treatment,
        manifest_report=manifest,
        release_policy={
            "version": "release-v2",
            "relation_precision_noninferiority_delta": 0.01,
            "minimum_gold_facts": 1,
            "confidence_method": "wilson-95",
        },
    )

    assert treatment["strict"]["recall"] == 1.0
    assert treatment["strict"]["precision"] == 0.5
    assert gate["production_approved"] is False
    assert "strict_precision_noninferiority" in gate["failed_gates"]


def test_missing_versioned_precision_policy_fails_closed() -> None:
    pairs, facts = _gold()
    manifest = validate_relation_gold(_manifest(), pairs, facts)
    report = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pairs,
        fact_gold=facts,
        run=_run(),
    )

    gate = evaluate_relation_release_gate_v2(
        control=report,
        treatment=deepcopy(report),
        manifest_report=manifest,
        release_policy={},
    )

    assert gate["production_approved"] is False
    assert "release_policy_incomplete" in gate["failed_gates"]


def test_relation_gold_migration_separates_pair_fact_and_prediction_adjudication() -> None:
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0018_shadow_relation_v2_gold"
    )

    assert "shadow_relation_gold_pair_decisions" in migration.sql
    assert "shadow_relation_gold_facts" in migration.sql
    assert "shadow_relation_prediction_adjudications" in migration.sql
    assert "NO_RELATION" in migration.sql
    assert "OTHER_UNSUPPORTED" in migration.sql
