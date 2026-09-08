from __future__ import annotations

from knowledge_forge.relation_extraction import (
    DEFERRED_BUDGET,
    PENDING_CLASSIFICATION,
    plan_pair_opportunities,
)
from tests.grounded_relation_fixtures import logical_schema, predicate_policy


def _schema() -> dict:
    return logical_schema(
        predicate_policy("alpha_rel", domain=["*"], range_=["*"]),
        version="schema-priority-v1",
    )


def _mention(key: str, start: int, end: int, surface: str):
    return {
        "mention_id": f"mention-{key}",
        "occurrence_key": f"occ-{key}",
        "surface": surface,
        "mention_start": start,
        "mention_end": end,
        "canonical_id": f"canonical-{key}",
        "assignment_version": "assignment-v1",
        "type": "GenericEntity",
    }


def test_budget_uses_priority_tuple_not_pair_identity_hash_order():
    content = "A links B. " + ("x" * 60) + " C."
    mentions = [
        _mention("far-left", 0, 1, "A"),
        _mention("near", 8, 9, "B"),
        _mention("far-right", len(content) - 2, len(content) - 1, "C"),
    ]
    unbounded = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
    )
    bounded = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_pair_opportunities=1,
    )
    admitted = [
        item for item in bounded.opportunities
        if item.status == PENDING_CLASSIFICATION
    ]
    deferred = [
        item for item in bounded.opportunities
        if item.status == DEFERRED_BUDGET
    ]

    assert len(admitted) == 1
    assert len(deferred) == 2
    assert admitted[0].provenance["character_distance"] == min(
        item.provenance["character_distance"]
        for item in unbounded.opportunities
    )
    assert "pair_budget" in deferred[0].provenance
