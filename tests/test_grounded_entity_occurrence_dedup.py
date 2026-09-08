from __future__ import annotations

import pytest

from knowledge_forge.relation_extraction import (
    PENDING_CLASSIFICATION,
    plan_pair_opportunities,
)
from tests.grounded_relation_fixtures import logical_schema, predicate_policy


def _schema() -> dict:
    return logical_schema(
        predicate_policy(
            "acts_on",
            domain=["Location"],
            range_=["Material"],
        ),
        version="schema-occurrence-v1",
    )


def test_pair_planner_collapses_multiple_observations_of_same_physical_occurrence():
    content = "Canada acts on lithium."
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=[
            {
                "occurrence_key": "occ-canada",
                "mention_id": "mention-canada-a",
                "surface": "Canada",
                "mention_start": 0,
                "mention_end": 6,
                "canonical_id": "canonical-canada",
                "assignment_version": "assignment-v1",
                "type": "Location",
            },
            {
                "occurrence_key": "occ-canada",
                "mention_id": "mention-canada-b",
                "surface": "Canada",
                "mention_start": 0,
                "mention_end": 6,
                "canonical_id": "canonical-canada",
                "assignment_version": "assignment-v1",
                "type": "Location",
            },
            {
                "occurrence_key": "occ-lithium",
                "mention_id": "mention-lithium",
                "surface": "lithium",
                "mention_start": 15,
                "mention_end": 22,
                "canonical_id": "canonical-lithium",
                "assignment_version": "assignment-v1",
                "type": "Material",
            },
        ],
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="policy-v1",
    )

    assert plan.duplicate_occurrences_collapsed == 1
    assert plan.theoretical_pairs == 1
    assert len(plan.opportunities) == 1
    assert plan.opportunities[0].left_occurrence_key == "occ-canada"
    assert plan.opportunities[0].left_mention_id == "mention-canada-a"
    assert plan.opportunities[0].status == PENDING_CLASSIFICATION


def test_pair_planner_rejects_same_occurrence_key_with_different_source_span():
    content = "Canada and Canada."

    with pytest.raises(ValueError, match="occurrence key collision"):
        plan_pair_opportunities(
            project_id=1,
            document_revision_id="revision-1",
            chunk_key="chunk-1",
            content=content,
            mentions=[
                {
                    "occurrence_key": "occ-canada",
                    "mention_id": "mention-canada-a",
                    "surface": "Canada",
                    "mention_start": 0,
                    "mention_end": 6,
                    "canonical_id": "canonical-canada",
                    "assignment_version": "assignment-v1",
                    "type": "Location",
                },
                {
                    "occurrence_key": "occ-canada",
                    "mention_id": "mention-canada-b",
                    "surface": "Canada",
                    "mention_start": 11,
                    "mention_end": 17,
                    "canonical_id": "canonical-canada",
                    "assignment_version": "assignment-v1",
                    "type": "Location",
                },
            ],
            logical_schema=_schema(),
            run_id="run-1",
            candidate_policy_version="policy-v1",
        )
