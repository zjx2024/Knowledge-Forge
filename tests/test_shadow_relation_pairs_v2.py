from __future__ import annotations

from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.relation_shadow_v2 import (
    DEFERRED_BUDGET,
    PENDING_CLASSIFICATION,
    REJECTED_SELF_LOOP,
    TYPE_INCOMPATIBLE,
    TYPE_UNRESOLVED,
    plan_pair_opportunities,
)


def _mention(
    key: str,
    start: int,
    end: int,
    *,
    surface: str,
    canonical: str | None = None,
    entity_type: str = "Organization",
):
    return {
        "mention_id": f"id-{key}",
        "occurrence_key": key,
        "surface": surface,
        "mention_start": start,
        "mention_end": end,
        "canonical_id": canonical if canonical is not None else f"canonical-{key}",
        "assignment_version": "assignment-v1",
        "type": entity_type,
    }


def _schema(*definitions):
    return {"relation_types": list(definitions) or [{
        "name": "related_to",
        "domain": ["*"],
        "range": ["*"],
    }]}


def _plan(content, mentions, **overrides):
    values = {
        "project_id": 12,
        "document_revision_id": "revision-1",
        "chunk_key": "chunk-1",
        "content": content,
        "mentions": mentions,
        "logical_schema": _schema(),
        "run_id": "run-1",
        "candidate_policy_version": "candidate-v2",
    }
    values.update(overrides)
    return plan_pair_opportunities(**values)


def test_two_occurrences_create_one_unordered_pair() -> None:
    content = "Alpha uses Beta."
    plan = _plan(content, [
        _mention("a", 0, 5, surface="Alpha"),
        _mention("b", 11, 15, surface="Beta"),
    ])

    assert plan.theoretical_pairs == 1
    assert len(plan.opportunities) == 1
    assert plan.opportunities[0].status == PENDING_CLASSIFICATION


def test_three_occurrences_create_at_most_three_pairs_not_six() -> None:
    content = "Alpha uses Beta and Gamma."
    plan = _plan(content, [
        _mention("a", 0, 5, surface="Alpha"),
        _mention("b", 11, 15, surface="Beta"),
        _mention("c", 20, 25, surface="Gamma"),
    ])

    assert plan.theoretical_pairs == 3
    assert len(plan.opportunities) == 3
    assert len({item.pair_identity for item in plan.opportunities}) == 3


def test_pair_identity_is_order_and_policy_independent_but_opportunity_is_not() -> None:
    content = "Alpha uses Beta."
    mentions = [
        _mention("a", 0, 5, surface="Alpha"),
        _mention("b", 11, 15, surface="Beta"),
    ]
    first = _plan(content, mentions).opportunities[0]
    reversed_input = _plan(content, list(reversed(mentions))).opportunities[0]
    other_policy = _plan(
        content, mentions, candidate_policy_version="candidate-v3",
    ).opportunities[0]

    assert first.pair_identity == reversed_input.pair_identity == other_policy.pair_identity
    assert first.opportunity_identity == reversed_input.opportunity_identity
    assert first.opportunity_identity != other_policy.opportunity_identity


def test_overlapping_context_discovery_does_not_duplicate_a_pair() -> None:
    content = "Alpha operates here. Beta uses Alpha."
    alpha = _mention("a", 0, 5, surface="Alpha")
    beta = _mention("b", 21, 25, surface="Beta")
    repeated_alpha = dict(alpha)
    plan = _plan(content, [alpha, beta, repeated_alpha])

    assert len(plan.opportunities) == 1


def test_same_surface_different_occurrences_are_not_collapsed() -> None:
    content = "Canada supplies Canada and Canada."
    starts = [0, 16, 27]
    plan = _plan(content, [
        _mention(f"canada-{index}", start, start + 6, surface="Canada")
        for index, start in enumerate(starts)
    ])

    assert len(plan.opportunities) == 3
    assert len({item.pair_identity for item in plan.opportunities}) == 3


def test_same_canonical_self_loop_is_recorded_and_filtered() -> None:
    content = "Alpha references Alpha."
    plan = _plan(content, [
        _mention("a-1", 0, 5, surface="Alpha", canonical="same"),
        _mention("a-2", 17, 22, surface="Alpha", canonical="same"),
    ])

    assert len(plan.opportunities) == 1
    assert plan.opportunities[0].status == REJECTED_SELF_LOOP


def test_wildcard_and_unknown_types_are_not_treated_as_incompatible() -> None:
    content = "Alpha uses Beta."
    unknown = [
        _mention("a", 0, 5, surface="Alpha", entity_type=""),
        _mention("b", 11, 15, surface="Beta", entity_type=""),
    ]
    wildcard = _plan(content, unknown).opportunities[0]
    unresolved = _plan(
        content,
        unknown,
        logical_schema=_schema({
            "name": "employs",
            "domain": ["Organization"],
            "range": ["Person"],
        }),
    ).opportunities[0]

    assert wildcard.status == PENDING_CLASSIFICATION
    assert wildcard.type_status == "compatible"
    assert unresolved.status == TYPE_UNRESOLVED
    assert unresolved.status != TYPE_INCOMPATIBLE


def test_known_incompatible_types_are_recorded_separately() -> None:
    content = "Alpha knows Beta."
    plan = _plan(
        content,
        [
            _mention("a", 0, 5, surface="Alpha", entity_type="Location"),
            _mention("b", 12, 16, surface="Beta", entity_type="Technology"),
        ],
        logical_schema=_schema({
            "name": "employs",
            "domain": ["Organization"],
            "range": ["Person"],
        }),
    )

    assert plan.opportunities[0].status == TYPE_INCOMPATIBLE


def test_budget_deferral_is_explicit_and_deterministic() -> None:
    content = "Alpha uses Beta and Gamma."
    plan = _plan(
        content,
        [
            _mention("a", 0, 5, surface="Alpha"),
            _mention("b", 11, 15, surface="Beta"),
            _mention("c", 20, 25, surface="Gamma"),
        ],
        max_pair_opportunities=1,
    )

    assert [item.status for item in plan.opportunities].count(PENDING_CLASSIFICATION) == 1
    assert [item.status for item in plan.opportunities].count(DEFERRED_BUDGET) == 2


def test_pair_ledger_migration_is_additive_and_allows_many_hypotheses() -> None:
    migration = next(
        item for item in MIGRATIONS if item.version == "0014_shadow_relation_v2_ledger"
    )

    assert "CREATE TABLE IF NOT EXISTS shadow_relation_pair_opportunities" in migration.sql
    assert "UNIQUE(run_id,candidate_policy_version,pair_identity)" in migration.sql
    assert "CREATE TABLE IF NOT EXISTS shadow_relation_hypotheses" in migration.sql
    assert "pair_opportunity_id,subject_occurrence_key,predicate" in migration.sql
    assert "object_occurrence_key,hypothesis_key" in migration.sql
    assert "direction TEXT" not in migration.sql
