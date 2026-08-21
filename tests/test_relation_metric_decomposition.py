from __future__ import annotations

from copy import deepcopy

from knowledge_forge.quality import (
    evaluate_relation_gold_run,
    relation_pair_candidate_recall_at_budget,
    validate_relation_gold,
)


def _manifest() -> dict:
    return {
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


def _pair(pair_id: str, decision: str = "HAS_RELATION") -> dict:
    return {
        "pair_id": pair_id,
        "dataset": "relation-pilot",
        "dataset_version": "v1",
        "split": "holdout",
        "document_revision_id": "revision-1",
        "document_content_hash": "b" * 64,
        "left_occurrence_id": f"{pair_id}-left",
        "left_span": [0, 5],
        "left_canonical_id": f"{pair_id}-entity-left",
        "right_occurrence_id": f"{pair_id}-right",
        "right_span": [10, 15],
        "right_canonical_id": f"{pair_id}-entity-right",
        "decision": decision,
        "evidence_spans": [[0, 16]] if decision == "HAS_RELATION" else [],
        "critical": False,
        "label_source": "human",
        "annotator": "annotator-a",
        "reviewer": "reviewer-b",
        "reviewed_at": "2026-07-24T00:00:00Z",
        "adjudication_status": "adjudicated",
    }


def _fact(pair_id: str = "pair-positive", *, evidence_start=0, evidence_end=16) -> dict:
    return {
        "pair_id": pair_id,
        "project_scope": "project-13",
        "subject_occurrence_id": f"{pair_id}-left",
        "subject_canonical_id": f"{pair_id}-entity-left",
        "predicate": "uses",
        "object_occurrence_id": f"{pair_id}-right",
        "object_canonical_id": f"{pair_id}-entity-right",
        "direction": "forward",
        "document_id": 18,
        "revision_id": "revision-1",
        "chunk_id": 6200,
        "subject_start": 0,
        "subject_end": 5,
        "object_start": 10,
        "object_end": 15,
        "evidence_start": evidence_start,
        "evidence_end": evidence_end,
        "evidence_text": "Alpha uses Beta.",
        "critical": False,
        "adjudication_status": "adjudicated",
    }


def _run(prediction: dict) -> dict:
    return {
        "mention_occurrences": [
            {"occurrence_key": "pair-positive-left"},
            {"occurrence_key": "pair-positive-right"},
        ],
        "pair_opportunities": [{"pair_identity": "pair-positive"}],
        "pair_decisions": [
            {"pair_identity": "pair-positive", "decision": "HYPOTHESES"},
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
        "facts": [prediction],
        "prediction_adjudications": [{
            "prediction_id": str(prediction["prediction_id"]),
            "correct": False,
        }],
        "normalization_metrics": {},
        "all_pairs_terminal": True,
        "stale": False,
        "cannot_link_violations": 0,
    }


def test_relation_pair_candidate_recall_uses_unordered_occurrence_pairs() -> None:
    report = relation_pair_candidate_recall_at_budget(
        [("gold-left", "gold-right")],
        [("gold-right", "gold-left")],
        budget=32,
    )

    assert report["recall"] == 1.0
    assert report["true_positive_ids"] == ["gold-left|gold-right"]
    assert report["false_negative_ids"] == []


def test_direction_and_evidence_metrics_are_independent_layers() -> None:
    pair_gold = [_pair("pair-positive")]
    fact_gold = [_fact()]
    manifest = validate_relation_gold(_manifest(), pair_gold, fact_gold)
    wrong_evidence = deepcopy(_fact(evidence_start=2, evidence_end=14))
    wrong_evidence["prediction_id"] = "pred-wrong-evidence"

    report = evaluate_relation_gold_run(
        manifest_report=manifest,
        pair_gold=pair_gold,
        fact_gold=fact_gold,
        run=_run(wrong_evidence),
    )

    assert report["canonical"]["true_positive"] == 1
    assert report["strict"]["true_positive"] == 0
    assert report["direction_accuracy"] == 1.0
    assert report["exact_evidence_coverage"] == 0.0
    assert report["denominators"]["direction_matched_endpoint_predicate"] == 1
