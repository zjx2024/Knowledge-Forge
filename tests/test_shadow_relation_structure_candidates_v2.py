from __future__ import annotations

from knowledge_forge.relation_shadow_v2 import (
    FILTERED_DISTANCE,
    FILTERED_STRUCTURE_INELIGIBLE,
    FILTERED_SCOPE_BOUNDARY,
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

    assert plan.opportunities[0].status == FILTERED_SCOPE_BOUNDARY


def test_list_item_scopes_and_parser_roles_are_hard_generic_boundaries():
    content = "Alpha Beta; Gamma Delta"
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=[
            _mention(
                "alpha", 0, 5, "Alpha", source_role="list_item",
                relation_eligible=True, scope_id="item-1", scope_kind="list_item",
                scope_start=0, scope_end=10,
            ),
            _mention(
                "beta", 6, 10, "Beta", source_role="list_item",
                relation_eligible=True, scope_id="item-1", scope_kind="list_item",
                scope_start=0, scope_end=10,
            ),
            _mention(
                "gamma", 12, 17, "Gamma", source_role="list_item",
                relation_eligible=True, scope_id="item-2", scope_kind="list_item",
                scope_start=12, scope_end=len(content),
            ),
        ],
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
    )

    by_pair = {
        frozenset((item.left_occurrence_key, item.right_occurrence_key)): item
        for item in plan.opportunities
    }
    assert by_pair[frozenset(("occ-alpha", "occ-beta"))].status == PENDING_CLASSIFICATION
    assert by_pair[frozenset(("occ-alpha", "occ-gamma"))].status == FILTERED_SCOPE_BOUNDARY

    header_plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-2",
        chunk_key="chunk-2",
        content="Alpha Beta",
        mentions=[
            _mention(
                "alpha", 0, 5, "Alpha", source_role="table_header",
                relation_eligible=False, scope_id="header", scope_kind="table_header",
                scope_start=0, scope_end=10,
            ),
            _mention(
                "beta", 6, 10, "Beta", source_role="table_header",
                relation_eligible=False, scope_id="header", scope_kind="table_header",
                scope_start=0, scope_end=10,
            ),
        ],
        logical_schema=_schema(),
        run_id="run-2",
        candidate_policy_version="candidate-v1",
    )
    assert header_plan.opportunities[0].status == FILTERED_STRUCTURE_INELIGIBLE


def test_schema_scope_and_source_role_metadata_control_admission_not_predicate_name():
    content = "Alpha. Beta."
    beta_start = content.index("Beta")
    mentions = [
        _mention("alpha", 0, 5, "Alpha", source_role="body", relation_eligible=True),
        _mention("beta", beta_start, beta_start + 4, "Beta", source_role="body", relation_eligible=True),
    ]
    strict = logical_schema(predicate_policy("alpha_rel"))
    permissive = logical_schema(predicate_policy(
        "renamed_rel",
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))

    strict_plan = plan_pair_opportunities(
        project_id=1, document_revision_id="revision-1", chunk_key="chunk-1",
        content=content, mentions=mentions, logical_schema=strict,
        run_id="run-1", candidate_policy_version="candidate-v1",
    )
    permissive_plan = plan_pair_opportunities(
        project_id=1, document_revision_id="revision-1", chunk_key="chunk-1",
        content=content, mentions=mentions, logical_schema=permissive,
        run_id="run-1", candidate_policy_version="candidate-v1",
    )

    assert strict_plan.opportunities[0].status == FILTERED_SCOPE_BOUNDARY
    assert permissive_plan.opportunities[0].status == PENDING_CLASSIFICATION
