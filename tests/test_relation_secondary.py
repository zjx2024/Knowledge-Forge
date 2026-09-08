from __future__ import annotations

import json

import pytest

from knowledge_forge.relation_secondary import (
    NO_SPECIFIC_RELATION,
    ROUTE_EVIDENCE_RECOVERY,
    ROUTE_NONE,
    ROUTE_POLARITY_REVIEW,
    ROUTE_PREDICATE_REFINEMENT,
    SECONDARY_INSUFFICIENT,
    SECONDARY_RELATION,
    parse_secondary_decision,
    parse_source_frame_polarity,
    route_priority,
    route_secondary_attempt,
    route_secondary_reason,
    secondary_recovery_prompt,
    secondary_response_schema,
    source_frame_polarity_prompt,
    source_frame_polarity_response_schema,
)
from knowledge_forge.migrations import MIGRATIONS


def test_secondary_routes_are_reason_axis_specific_and_generic():
    assert route_secondary_reason(
        "ATOMIC_FACT_INVENTORY_INCOMPLETE",
    ).route == ROUTE_EVIDENCE_RECOVERY
    assert route_secondary_reason(
        "NO_DIRECT_ATOMIC_PAIR",
    ).route == ROUTE_EVIDENCE_RECOVERY
    assert route_secondary_reason(
        "PUBLISH_MODE_REVIEW_ONLY",
    ).route == ROUTE_PREDICATE_REFINEMENT
    assert route_secondary_reason(
        "MODEL_REJECTED_REASON_OMITTED",
    ).route == "ROLE_RETRY"
    assert route_secondary_reason(
        "ORDERED_ROLE_NOT_ENTAILED",
    ).route == "ROLE_RETRY"
    assert route_secondary_reason(
        "DIRECTION_NOT_EXCLUSIVE",
    ).route == "ROLE_RETRY"
    assert route_secondary_reason(
        "SUBJECT_ROLE_MISMATCH",
    ).route == "ROLE_RETRY"
    assert route_secondary_reason(
        "INSUFFICIENT_EVIDENCE",
    ).route == ROUTE_EVIDENCE_RECOVERY
    assert route_secondary_reason(
        "ATOMIC_FRAME_RELATION_MISMATCH",
    ).route == ROUTE_PREDICATE_REFINEMENT
    assert route_secondary_reason(
        "RELATION_SEMANTICS_MISMATCH",
    ).route == ROUTE_PREDICATE_REFINEMENT
    assert route_secondary_reason("EXPLICIT_CO_OCCURRENCE").route == ROUTE_NONE
    typed_direction = route_secondary_attempt(
        "model-specific prose",
        verdict="REJECTED",
        forward_entailment="NOT_ENTAILED",
        reverse_entailment="ENTAILED",
    )
    assert typed_direction.route == "ROLE_RETRY"
    assert typed_direction.reason_code == "REVERSE_ORIENTATION_ENTAILED"


def test_secondary_candidate_idempotency_is_scoped_to_primary_hypothesis():
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0028_shadow_relation_secondary_idempotency"
    )
    assert "DROP CONSTRAINT IF EXISTS" in migration.sql
    assert "COALESCE(parent_hypothesis_id, ''::text)" in migration.sql
    assert "uq_shadow_relation_secondary_candidates_idempotency" in migration.sql
    cleanup = next(
        item for item in MIGRATIONS
        if item.version == "0029_shadow_relation_secondary_idempotency_cleanup"
    )
    assert "shadow_relation_secondary_can_secondary_pass_id_pair_opport_key" in cleanup.sql


def test_secondary_response_requires_exact_endpoints_and_frozen_spans():
    raw = {
        "decision": SECONDARY_RELATION,
        "predicate": "function_purpose",
        "subject_side": "LEFT",
        "object_side": "RIGHT",
        "evidence_spans": [{"start": 0, "end": 10}],
        "reason_code": "DIRECT_ASSERTION",
    }
    parsed = parse_secondary_decision(
        raw,
        allowed_predicates=["function_purpose", "uses"],
        context_span=(0, 10),
        left_span=(0, 2),
        right_span=(8, 10),
        content_length=10,
    )
    assert parsed.predicate == "function_purpose"
    assert parsed.evidence_spans == ((0, 10),)

    with pytest.raises(ValueError, match="not allowed"):
        parse_secondary_decision(
            {**raw, "predicate": "invented_predicate"},
            allowed_predicates=["function_purpose"],
            context_span=(0, 10),
            left_span=(0, 2),
            right_span=(8, 10),
            content_length=10,
        )

    with pytest.raises(ValueError, match="predicate lock"):
        parse_secondary_decision(
            {**raw, "predicate": "uses"},
            allowed_predicates=["function_purpose", "uses"],
            required_predicate="function_purpose",
            context_span=(0, 10),
            left_span=(0, 2),
            right_span=(8, 10),
            content_length=10,
        )


def test_secondary_negative_decision_does_not_create_a_predicate():
    parsed = parse_secondary_decision(
        {
            "decision": SECONDARY_INSUFFICIENT,
            "predicate": NO_SPECIFIC_RELATION,
            "subject_side": "",
            "object_side": "",
            "evidence_spans": [],
            "reason_code": "NO_DIRECT_ASSERTION",
        },
        allowed_predicates=["uses"],
        context_span=(0, 10),
        left_span=(0, 2),
        right_span=(8, 10),
        content_length=10,
    )
    assert parsed.predicate == NO_SPECIFIC_RELATION

    echoed = parse_secondary_decision(
        {
            "decision": "NO_RELATION",
            "predicate": "uses",
            "subject_side": "",
            "object_side": "",
            "evidence_spans": [],
            "reason_code": "NO_DIRECT_ASSERTION",
        },
        allowed_predicates=["uses"],
        context_span=(100, 110),
        left_span=(100, 102),
        right_span=(108, 110),
        content_length=110,
    )
    assert echoed.predicate == NO_SPECIFIC_RELATION


def test_secondary_maps_relative_evidence_coordinates_back_to_document():
    parsed = parse_secondary_decision(
        {
            "decision": SECONDARY_RELATION,
            "predicate": "uses",
            "subject_side": "LEFT",
            "object_side": "RIGHT",
            "evidence_spans": [{"span": [0, 10], "text": "local context"}],
            "reason_code": "DIRECT_ASSERTION",
        },
        allowed_predicates=["uses"],
        context_span=(100, 110),
        left_span=(100, 102),
        right_span=(108, 110),
        content_length=110,
    )
    assert parsed.evidence_spans == ((100, 110),)


def test_secondary_accepts_legacy_relation_wrapper_only_with_exact_endpoints():
    parsed = parse_secondary_decision(
        {
            "relation": {
                "predicate": "uses",
                "subject": {"span": [8, 10], "surface": "右端点"},
                "object": {"span": [0, 2], "surface": "左端点"},
            },
        },
        allowed_predicates=["uses"],
        context_span=(0, 10),
        left_span=(0, 2),
        right_span=(8, 10),
        content_length=10,
    )
    assert parsed.decision == SECONDARY_RELATION
    assert parsed.subject_side == "RIGHT"
    assert parsed.object_side == "LEFT"
    assert parsed.evidence_spans == ((0, 10),)

    with pytest.raises(ValueError, match="lacks endpoint roles"):
        parse_secondary_decision(
            {"relation": "uses"},
            allowed_predicates=["uses"],
            context_span=(0, 10),
            left_span=(0, 2),
            right_span=(8, 10),
            content_length=10,
        )


def test_secondary_normalizes_exact_endpoint_surfaces_in_side_fields():
    parsed = parse_secondary_decision(
        {
            "decision": SECONDARY_RELATION,
            "predicate": "uses",
            "subject_side": "suite",
            "object_side": "clause",
            "evidence_spans": [[0, 10]],
            "reason_code": "DIRECT_ASSERTION",
        },
        allowed_predicates=["uses"],
        context_span=(0, 10),
        left_span=(0, 5),
        right_span=(6, 10),
        content_length=10,
        left_surface="suite",
        right_surface="clause",
    )
    assert parsed.subject_side == "LEFT"
    assert parsed.object_side == "RIGHT"


def test_secondary_normalizes_endpoint_objects_and_recovers_endpoint_evidence():
    parsed = parse_secondary_decision(
        {
            "decision": SECONDARY_RELATION,
            "predicate": "uses",
            "subject_side": {"span": [0, 5], "surface": "suite"},
            "object_side": {"span": [6, 10], "surface": "clause"},
            # The model returned only the lexical cue. Exact endpoint spans
            # may be added as transport evidence; the independent verifier
            # still decides whether the relation itself is supported.
            "evidence_spans": [{"span": [3, 4], "surface": "uses"}],
            "reason_code": "DIRECT_ASSERTION",
        },
        allowed_predicates=["uses"],
        context_span=(0, 10),
        left_span=(0, 5),
        right_span=(6, 10),
        content_length=10,
    )
    assert parsed.subject_side == "LEFT"
    assert parsed.object_side == "RIGHT"
    assert (0, 5) in parsed.evidence_spans
    assert (6, 10) in parsed.evidence_spans


def test_secondary_prompt_withholds_primary_verdict_and_keeps_structure_contract():
    prompt = secondary_recovery_prompt(
        route=route_secondary_reason("PUBLISH_MODE_REVIEW_ONLY"),
        content="函数用于发送消息。",
        context_span=(0, 9),
        left_span=(0, 2),
        right_span=(5, 7),
        left_surface="函数",
        right_surface="消息",
        left_type="Function",
        right_type="Concept",
        allowed_predicate_metadata=[{
            "name": "function_purpose",
            "semantic_definition": "function has purpose",
            "direction_description": "function to purpose",
            "publish_mode": "auto",
        }],
        structural_contract={"hierarchical_parent_child": False},
    )
    payload = json.loads(prompt)
    assert "primary_verdict" not in payload
    assert payload["route"]["name"] == ROUTE_PREDICATE_REFINEMENT
    assert payload["structural_contract"]["hierarchical_parent_child"] is False
    assert secondary_response_schema(["function_purpose"])["$id"]


def test_role_retry_prompt_requires_fresh_endpoint_orientation():
    prompt = secondary_recovery_prompt(
        route=route_secondary_reason("AUXILIARY_ROLE_AUDIT_UNAVAILABLE"),
        content="The platform supports tracking operations.",
        context_span=(0, 44),
        left_span=(4, 12),
        right_span=(22, 41),
        left_surface="platform",
        right_surface="tracking operations",
        left_type="Technology",
        right_type="Event",
        allowed_predicate_metadata=[{
            "name": "supports",
            "semantic_definition": "A source provides capability to a target.",
            "direction_description": "supporting source to supported target",
            "publish_mode": "auto",
        }],
        structural_contract={"hierarchical_parent_child": False},
        orientation_only=True,
        preferred_predicate="supports",
    )
    payload = json.loads(prompt)
    assert "不得复制一次候选的端点顺序" in payload["instruction"]
    assert "supports" in payload["instruction"]
    assert "必须只审计该谓词" in payload["instruction"]
    assert payload["preferred_predicate"] == "supports"
    assert payload["route"]["name"] == "ROLE_RETRY"


def test_source_frame_polarity_contract_is_narrow_and_source_grounded():
    content = "Alice did not stop because she saw Rabbit."
    frame_text = "she saw Rabbit"
    frame_start = content.index(frame_text)
    frame_span = (frame_start, frame_start + len(frame_text))
    payload = json.loads(source_frame_polarity_prompt(
        content=content,
        context_span=(0, len(content)),
        frame_span=frame_span,
        frame_text=frame_text,
        open_relation="saw",
    ))

    assert payload["exact_frame"]["text"] == frame_text
    assert "matrix, sibling, or nearby predicate" in payload["instruction"]
    schema = source_frame_polarity_response_schema()
    assert list(schema["properties"]) == ["result"]
    assert schema["required"] == ["result"]
    assert parse_source_frame_polarity({"result": "ASSERTED"}) == "ASSERTED"
    assert parse_source_frame_polarity({"polarity": "NEGATED"}) == "NEGATED"
    assert route_priority(ROUTE_POLARITY_REVIEW) < route_priority(
        ROUTE_PREDICATE_REFINEMENT
    )


def test_source_frame_polarity_parser_and_prompt_fail_closed():
    content = "A did not stop; A moved."
    frame_start = content.index("A moved")
    with pytest.raises(ValueError, match="does not match"):
        source_frame_polarity_prompt(
            content=content,
            context_span=(0, len(content)),
            frame_span=(frame_start, frame_start + len("A moved")),
            frame_text="A halted",
            open_relation="moved",
        )
    with pytest.raises(ValueError, match="invalid"):
        parse_source_frame_polarity({"result": "SUPPORTED"})
    with pytest.raises(ValueError, match="conflict"):
        parse_source_frame_polarity({
            "result": "ASSERTED", "polarity": "NEGATED",
        })
