from __future__ import annotations

from knowledge_forge.relation_shadow_v2 import (
    FILTERED_DISTANCE,
    FILTERED_STRUCTURE_PEER,
    PENDING_CLASSIFICATION,
    plan_pair_opportunities,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


def _schema() -> dict:
    return logical_schema(
        predicate_policy("alpha_rel", domain=["*"], range_=["*"]),
        version="schema-structure-v1",
    )


def _mention(key: str, start: int, end: int, surface: str, **metadata):
    return {
        "mention_id": f"mention-{key}",
        "occurrence_key": f"occ-{key}",
        "surface": surface,
        "mention_start": start,
        "mention_end": end,
        "canonical_id": f"canonical-{key}",
        "assignment_version": "assignment-v1",
        "type": "GenericEntity",
        **metadata,
    }


def test_same_table_row_is_admitted_even_when_character_window_would_filter():
    content = "Country " + ("." * 80) + " Commodity"
    commodity_start = content.index("Commodity")
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=[
            _mention(
                "country", 0, 7, "Country",
                table_id="table-1", row_id="row-1", cell_id="cell-1",
            ),
            _mention(
                "commodity",
                commodity_start,
                commodity_start + len("Commodity"),
                "Commodity",
                table_id="table-1", row_id="row-1", cell_id="cell-2",
            ),
        ],
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_char_distance=10,
    )

    assert plan.opportunities[0].status == PENDING_CLASSIFICATION
    assert plan.opportunities[0].planning_reason == "same_table_row"


def test_neighboring_table_rows_are_explicitly_filtered_as_peers():
    content = "Canada lithium Chile cobalt"
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=[
            _mention(
                "canada", 0, 6, "Canada",
                table_id="table-1", row_id="row-1", cell_id="cell-country",
            ),
            _mention(
                "chile", 15, 20, "Chile",
                table_id="table-1", row_id="row-2", cell_id="cell-country",
            ),
        ],
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
    )

    assert plan.opportunities[0].status == FILTERED_STRUCTURE_PEER
    assert plan.opportunities[0].planning_reason == "different_table_peer_rows"


def test_plain_far_apart_pair_still_fails_distance_without_structure():
    content = "Alpha. " + ("x" * 80) + " Beta."
    beta_start = content.index("Beta")
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=[
            _mention("alpha", 0, 5, "Alpha"),
            _mention("beta", beta_start, beta_start + 4, "Beta"),
        ],
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_char_distance=10,
    )

    assert plan.opportunities[0].status == FILTERED_DISTANCE
