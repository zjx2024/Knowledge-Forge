from __future__ import annotations

import pytest

from knowledge_forge.extraction_funnel import (
    ExtractionFunnel,
    FunnelInvariantError,
    relation_shadow_funnel,
)
from knowledge_forge.policy import P1_RELATION_SHADOW_POLICY_SET
from knowledge_forge.relation_pipeline import RelationCandidate, RelationPlan


def test_funnel_conserves_each_stage_and_reports_stable_terminal_ids() -> None:
    funnel = ExtractionFunnel(run_id="run-1")
    funnel.register("raw_entity_mentions", ["mention-1", "mention-2", "mention-3"])
    funnel.terminate("raw_entity_mentions", "mention-1", "admitted_mentions")
    funnel.terminate("raw_entity_mentions", "mention-2", "rejected_mentions", reason="title")
    funnel.terminate("raw_entity_mentions", "mention-3", "uncertain_mentions", reason="low_confidence")

    funnel.register("entity_resolution_candidates", ["mention-1"])
    funnel.terminate("entity_resolution_candidates", "mention-1", "resolved")

    funnel.register("bounded_relation_pairs", ["pair-1", "pair-2"])
    funnel.terminate("bounded_relation_pairs", "pair-1", "specific_relation")
    funnel.terminate("bounded_relation_pairs", "pair-2", "NO_RELATION")

    report = funnel.report()

    assert report["coverage_complete"] is True
    assert report["failures"] == []
    assert report["stages"]["raw_entity_mentions"]["conserved"] is True
    assert report["stages"]["raw_entity_mentions"]["terminal_counts"] == {
        "admitted_mentions": 1,
        "rejected_mentions": 1,
        "uncertain_mentions": 1,
    }
    assert report["stages"]["bounded_relation_pairs"]["terminal_ids"] == {
        "NO_RELATION": ["pair-2"],
        "specific_relation": ["pair-1"],
    }


def test_funnel_rejects_duplicate_or_unknown_terminal_assignment() -> None:
    funnel = ExtractionFunnel(run_id="run-2")
    funnel.register("bounded_relation_pairs", ["pair-1"])
    funnel.terminate("bounded_relation_pairs", "pair-1", "specific_relation")

    with pytest.raises(FunnelInvariantError, match="already has terminal"):
        funnel.terminate("bounded_relation_pairs", "pair-1", "NO_RELATION")
    with pytest.raises(FunnelInvariantError, match="not registered"):
        funnel.terminate("bounded_relation_pairs", "pair-2", "NO_RELATION")
    with pytest.raises(FunnelInvariantError, match="not allowed"):
        funnel.terminate("bounded_relation_pairs", "pair-1", "publishable")


def test_funnel_fails_closed_when_any_registered_item_has_no_terminal() -> None:
    funnel = ExtractionFunnel(run_id="run-3")
    funnel.register("bounded_relation_pairs", ["pair-1", "pair-2"])
    funnel.terminate("bounded_relation_pairs", "pair-1", "OTHER_UNSUPPORTED")

    report = funnel.report()

    assert report["coverage_complete"] is False
    assert report["failures"] == ["bounded_relation_pairs:missing_terminal:pair-2"]
    assert report["stages"]["bounded_relation_pairs"]["conserved"] is False
    assert report["stages"]["bounded_relation_pairs"]["missing_terminal_ids"] == ["pair-2"]


def _candidate(candidate_id: str) -> RelationCandidate:
    return RelationCandidate(
        candidate_id=candidate_id,
        source_mention_id=f"{candidate_id}-source",
        target_mention_id=f"{candidate_id}-target",
        source_mention_key=f"{candidate_id}-source-key",
        target_mention_key=f"{candidate_id}-target-key",
        source_name="Alpha",
        target_name="Beta",
        source_type="Organization",
        target_type="Product",
        allowed_predicates=("produces",),
        allowed_directions=("as_is",),
        context_id="ctx-0-0",
        sentence_start=0,
        sentence_end=0,
        context_start_char=0,
        context_end_char=20,
        source_start_char=0,
        source_end_char=5,
        target_start_char=15,
        target_end_char=19,
    )


def test_relation_shadow_funnel_accounts_for_decisions_and_failed_work() -> None:
    plan = RelationPlan(
        policy_set=P1_RELATION_SHADOW_POLICY_SET,
        raw_pairs=4,
        blocked_pairs=0,
        candidates=[_candidate("positive"), _candidate("negative"), _candidate("failed")],
        qualified_count=4,
        overflow=[{
            "candidate_id": "overflow",
            "reason": "chunk_limit",
            "candidate": {"candidate_id": "overflow"},
        }],
    )
    report = relation_shadow_funnel(
        run_id="run-4",
        plan=plan,
        persisted_work=[
            {
                "status": "completed",
                "candidate_ids": ["positive", "negative"],
                "output_json": {"decisions": [
                    {"candidate_id": "positive", "predicate": "produces"},
                    {"candidate_id": "negative", "predicate": "NO_RELATION"},
                ]},
            },
            {
                "status": "terminal_failed",
                "candidate_ids": ["failed"],
                "error_message": "missing response",
            },
            {
                "status": "quarantined",
                "candidate_ids": ["overflow"],
                "error_message": "chunk_limit",
            },
        ],
        decisions=[
            {"candidate_id": "positive", "predicate": "produces"},
            {"candidate_id": "negative", "predicate": "NO_RELATION"},
        ],
    )

    stage = report["stages"]["bounded_relation_pairs"]
    assert stage["conserved"] is True
    assert stage["terminal_counts"] == {
        "NO_RELATION": 1,
        "rejected": 1,
        "retryable_failed": 1,
        "specific_relation": 1,
    }
    assert stage["failure_reasons"] == {
        "failed": "missing response",
        "overflow": "chunk_limit",
    }


def test_relation_shadow_funnel_exposes_missing_classifier_decision() -> None:
    plan = RelationPlan(
        policy_set=P1_RELATION_SHADOW_POLICY_SET,
        raw_pairs=1,
        blocked_pairs=0,
        candidates=[_candidate("missing")],
        qualified_count=1,
    )
    report = relation_shadow_funnel(
        run_id="run-5",
        plan=plan,
        persisted_work=[{
            "status": "completed",
            "candidate_ids": ["missing"],
            "output_json": {"decisions": []},
        }],
        decisions=[],
    )

    assert report["coverage_complete"] is False
    assert report["failures"] == ["bounded_relation_pairs:missing_terminal:missing"]
