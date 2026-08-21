from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from knowledge_forge.relation_shadow_v2 import (
    ALLOW_EXPLICIT_NEGATIVE_RELATION,
    bounded_relation_context,
    FILTERED_DISTANCE,
    FILTERED_STRUCTURE_INELIGIBLE,
    FILTERED_SCOPE_BOUNDARY,
    FILTERED_STRUCTURE_PEER,
    PENDING_CLASSIFICATION,
    normalize_predicate_schema,
    parse_source_frame_schema_bindings,
    plan_pair_opportunities,
    source_frame_projection_hypotheses,
    source_frame_schema_recall_shortlist,
    batch_verifier_prompt,
    batch_verifier_response_schema,
    batch_verifier_decision_is_source_grounded,
    conflicting_source_hypothesis_groups,
    deterministic_parser_structure_terminal_attempt,
    deterministic_source_structure_schema_bindings,
    coordinated_source_projection_recovery_attempt,
    parse_batch_verifier_response,
    source_frame_schema_binding_response_schema,
    source_frame_schema_binding_prompt,
    source_batch_focused_review,
    source_projection_competition_risks,
    source_projection_competition_components,
    source_projection_coordinated_audit_components,
    source_projection_coordinated_recovery_plan,
    source_projection_group_anchor_claim,
    source_projection_group_anchor_audit_response_schema,
    parse_source_projection_group_anchor_audit,
    source_projection_group_anchor_audit_terminal_attempt,
    source_projection_group_audit_prompt,
    source_projection_group_audit_response_schema,
    source_projection_group_audit_terminal_attempt,
    parse_source_projection_group_audit,
    source_role_audit_conflict_is_terminal,
    source_projection_role_audit_prompt,
    source_projection_role_audit_response_schema,
    source_projection_role_audit_terminal_attempt,
    source_structure_binding_audit_prompt,
    source_structure_binding_audit_response_schema,
    parse_source_structure_binding_audit,
    source_structure_binding_audit_terminal_attempt,
    source_predicate_conflict_prompt,
    source_predicate_conflict_response_schema,
    parse_source_predicate_conflict,
    parse_source_projection_role_audit,
    parser_structure_claim_intent,
    ShadowClassificationError,
    structural_projection_hypotheses,
    supported_for_normalization,
    VerificationAttempt,
    SHADOW_RELATION_VERIFIER_PARSER_VERSION,
    SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
    SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
    augment_grounded_open_fact_inventory,
    validate_hypothesis,
)
from knowledge_forge.shadow_context import (
    occurrence_scope,
    structural_spans_for_chunk,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


def _schema() -> dict:
    return logical_schema(
        predicate_policy("alpha_rel", domain=["*"], range_=["*"]),
        version="schema-structure-v1",
    )


def test_source_frame_schema_binding_projects_only_frozen_endpoint_mentions():
    content = "Alpha connects Beta."
    mentions = [
        _mention("alpha", 0, 5, "Alpha"),
        _mention("beta", 15, 19, "Beta"),
    ]
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_char_distance=100,
    )
    assert len(plan.opportunities) == 1
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "connects",
            "arguments": ["Alpha", "Beta"],
            "roles": ["source", "target"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Alpha", "Beta"],
            "assertion_status": "ASSERTED",
        }],
    }
    schema = _schema()
    binding_prompt = source_frame_schema_binding_prompt(
        inventory, logical_schema=schema,
    )
    assert "Alpha" in binding_prompt and "alpha_rel" in binding_prompt
    assert source_frame_schema_binding_response_schema(schema)["additionalProperties"] is False
    bindings = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "alpha_rel",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory=inventory,
        logical_schema=schema,
    )
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        bindings,
        occurrences=[
            {"occurrence_key": "occ-alpha", "surface": "Alpha"},
            {"occurrence_key": "occ-beta", "surface": "Beta"},
        ],
        opportunities_by_pair={
            plan.opportunities[0].pair_identity: plan.opportunities[0],
        },
        logical_schema=schema,
        classifier_model_revision="model-v1",
    )
    assert len(hypotheses) == 1
    assert hypotheses[0].predicate == "alpha_rel"
    assert hypotheses[0].subject_occurrence_key == "occ-alpha"
    assert hypotheses[0].object_occurrence_key == "occ-beta"
    assert hypotheses[0].predicate_metadata["candidate_source"] == (
        "ATOMIC_FRAME_SCHEMA_PROJECTION"
    )
    assert hypotheses[0].predicate_metadata[
        "source_frame_assertion_status"
    ] == "ASSERTED"
    assert hypotheses[0].predicate_metadata[
        "source_frame_explicit_negation"
    ] is False

    recovery = source_batch_focused_review(
        hypotheses[0], {"verdict": "REJECTED"}, logical_schema=schema,
    )
    assert recovery == {
        "kind": "recovery",
        "reason": "BATCH_REJECTED_DIRECT_SOURCE_BINDING",
        "priority": 5,
        "source_role_audit": True,
    }
    direct_confirmation = source_batch_focused_review(
        hypotheses[0],
        {"verdict": "SUPPORTED"},
        logical_schema=schema,
    )
    assert direct_confirmation == {
        "kind": "confirmation",
        "reason": "BATCH_SUPPORTED_DIRECT_SOURCE_BINDING",
        "priority": 1,
        "source_role_audit": True,
        "role_audit_only": True,
    }
    shortlist_hypothesis = replace(
        hypotheses[0],
        predicate_metadata={
            **hypotheses[0].predicate_metadata,
            "source_binding_source": "DETERMINISTIC_SEMANTIC_SHORTLIST",
        },
    )
    confirmation = source_batch_focused_review(
        shortlist_hypothesis,
        {"verdict": "SUPPORTED"},
        logical_schema=schema,
    )
    assert confirmation == {
        "kind": "confirmation",
        "reason": "BATCH_SUPPORTED_RECALL_SHORTLIST",
        "priority": 2,
        "source_role_audit": True,
        "role_audit_only": True,
    }


def test_batch_supported_nary_and_competing_projections_require_confirmation():
    content = "Alpha moves Item from Store."
    mentions = [
        _mention("alpha", 0, 5, "Alpha"),
        _mention("store", 22, 27, "Store"),
    ]
    schema = _schema()
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_char_distance=100,
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "moves from",
            "arguments": ["Alpha", "Store"],
            "argument_occurrence_keys": ["occ-alpha", "occ-store"],
            "roles": ["grammatical_subject", "role_complement"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Alpha", "Store"],
            "assertion_status": "ASSERTED",
            "parser_inferred_source_syntax": "ordered_three_role_construction",
            "evidence_span": [0, len(content)],
            "evidence_text": content,
        }],
    }
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        {"coverage_complete": True, "bindings": [{
            "frame_index": 0,
            "predicate": "alpha_rel",
            "subject_index": 0,
            "object_index": 1,
        }]},
        occurrences=[
            {"occurrence_key": "occ-alpha", "surface": "Alpha"},
            {"occurrence_key": "occ-store", "surface": "Store"},
        ],
        opportunities_by_pair={
            plan.opportunities[0].pair_identity: plan.opportunities[0],
        },
        logical_schema=schema,
        classifier_model_revision="model-v1",
    )[0]
    nary_review = source_batch_focused_review(
        hypothesis, {"verdict": "SUPPORTED"}, logical_schema=schema,
    )
    assert nary_review == {
        "kind": "confirmation",
        "reason": "BATCH_SUPPORTED_ACTOR_COMPLEMENT_PROJECTION",
        "priority": 0,
        "source_role_audit": True,
        "role_audit_only": True,
    }

    # One parser-proved three-role frame licenses additive binary projections.
    # Actor->complement and theme->complement share an endpoint and evidence,
    # but they are not competing attempts to fill the same source role.
    actor_projection = replace(
        hypothesis,
        hypothesis_id="actor-to-complement",
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_subject_argument_index": 0,
            "source_object_argument_index": 2,
            "source_subject_canonical_id": "alpha",
            "source_object_canonical_id": "store",
        },
    )
    theme_projection = replace(
        actor_projection,
        hypothesis_id="theme-to-complement",
        subject_occurrence_key="occ-item",
        predicate_metadata={
            **actor_projection.predicate_metadata,
            "source_frame_roles": ["action_theme", "role_complement"],
            "source_subject_argument_index": 1,
            "source_subject_surface": "Item",
            "source_subject_canonical_id": "item",
        },
    )
    assert source_projection_competition_risks([
        actor_projection, theme_projection,
    ]) == frozenset()
    carrier_schema = logical_schema(predicate_policy(
        "source_assertion",
        semantic_definition=(
            "The exact open source expression relates subject to object."
        ),
        open_relation_policy={
            "enabled": True,
            "fallback_only": True,
            "allowed_candidate_sources": [
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            ],
            "display_source_expression": True,
            "identity_source_expression": True,
            "require_exact_source_expression": True,
        },
    ))
    carrier_theme_projection = replace(
        theme_projection,
        predicate="source_assertion",
        predicate_metadata={
            **theme_projection.predicate_metadata,
            "candidate_source": "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            "source_binding_source": "OPEN_EXPRESSION_CARRIER_FALLBACK",
            "source_subject_argument_index": 0,
            "source_object_argument_index": 1,
        },
    )
    assert source_batch_focused_review(
        carrier_theme_projection,
        {"verdict": "REJECTED"},
        logical_schema=carrier_schema,
    ) == {
        "kind": "recovery",
        "reason": "BATCH_REJECTED_PARSER_THEME_COMPLEMENT_PROJECTION",
        "priority": -1,
        "source_role_audit": False,
        "mandatory": True,
    }

    competitor = replace(
        hypothesis,
        hypothesis_id="competing-hypothesis",
        subject_occurrence_key="occ-nearby-participant",
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_syntax_origin": "",
            "source_subject_surface": "Nearby participant",
        },
    )
    base = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_syntax_origin": "",
        },
    )
    risks = source_projection_competition_risks([base, competitor])
    assert risks == frozenset({
        base.hypothesis_id, competitor.hypothesis_id,
    })
    competition_review = source_batch_focused_review(
        base,
        {"verdict": "SUPPORTED"},
        logical_schema=schema,
        source_competition_risk=True,
    )
    assert competition_review == {
        "kind": "confirmation",
        "reason": "BATCH_SUPPORTED_COMPETING_SOURCE_PROJECTION",
        "priority": 1,
        "source_role_audit": True,
        "role_audit_only": True,
    }
    competing_object = replace(
        base,
        hypothesis_id="competing-object-hypothesis",
        object_occurrence_key="occ-nearby-object",
        predicate_metadata={
            **base.predicate_metadata,
            "source_frame_relation": "different nearby phrase",
            "source_object_surface": "Nearby object",
            "source_object_canonical_id": "nearby-object",
        },
    )
    assert source_projection_competition_risks([
        base, competing_object,
    ]) == frozenset()
    parser_member_a = replace(
        hypothesis,
        hypothesis_id="parser-member-a",
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_object_surface": "Member A",
            "source_object_canonical_id": "member-a",
        },
    )
    parser_member_b = replace(
        parser_member_a,
        hypothesis_id="parser-member-b",
        object_occurrence_key="occ-member-b",
        predicate_metadata={
            **parser_member_a.predicate_metadata,
            "source_object_surface": "Member B",
            "source_object_canonical_id": "member-b",
        },
    )
    assert source_projection_competition_risks([
        parser_member_a, parser_member_b,
    ]) == frozenset()
    model_member_a = replace(
        base,
        hypothesis_id="model-coordinated-member-a",
        object_occurrence_key="occ-model-member-a",
        predicate_metadata={
            **base.predicate_metadata,
            "source_syntax_origin": "",
            "source_object_surface": "Member A",
            "source_object_canonical_id": "model-member-a",
            "source_frame_other_participants": ["Member B"],
        },
    )
    model_member_b = replace(
        model_member_a,
        hypothesis_id="model-coordinated-member-b",
        object_occurrence_key="occ-model-member-b",
        predicate_metadata={
            **model_member_a.predicate_metadata,
            "source_object_surface": "Member B",
            "source_object_canonical_id": "model-member-b",
            "source_frame_other_participants": ["Member A"],
        },
    )
    assert source_projection_competition_risks([
        model_member_a, model_member_b,
    ]) == frozenset()
    assert source_projection_coordinated_audit_components([
        (model_member_a, plan.opportunities[0]),
        (model_member_b, plan.opportunities[0]),
    ]) == ((
        *sorted({
            model_member_a.hypothesis_id,
            model_member_b.hypothesis_id,
        }),
    ),)
    anchor, anchor_opportunity = source_projection_group_anchor_claim([
        (model_member_b, plan.opportunities[0]),
        (model_member_a, plan.opportunities[0]),
    ])
    assert anchor.hypothesis_id == model_member_a.hypothesis_id
    assert anchor_opportunity == plan.opportunities[0]
    assert source_projection_group_anchor_audit_response_schema()[
        "additionalProperties"
    ] is False
    assert parse_source_projection_group_anchor_audit({
        "projection_status": "CONFLICT",
    }) == "CONFLICT"
    anchor_terminal = source_projection_group_anchor_audit_terminal_attempt(
        model_member_a,
        result="CONFLICT",
        verifier_model_revision="model-v1",
    )
    assert anchor_terminal.verdict == "REJECTED"
    assert anchor_terminal.reason_code == (
        "SOURCE_PROJECTION_GROUP_ANCHOR_CONFLICT"
    )
    unrelated_member = replace(
        model_member_b,
        hypothesis_id="unrelated-nearby-member",
        predicate_metadata={
            **model_member_b.predicate_metadata,
            "source_object_surface": "Member C",
            "source_object_canonical_id": "model-member-c",
            "source_frame_other_participants": [],
        },
    )
    assert source_projection_competition_risks([
        model_member_a, unrelated_member,
    ]) == frozenset({
        model_member_a.hypothesis_id,
        unrelated_member.hypothesis_id,
    })
    assert source_projection_coordinated_audit_components([
        (model_member_a, plan.opportunities[0]),
        (unrelated_member, plan.opportunities[0]),
    ]) == ()
    with pytest.raises(ValueError):
        source_projection_group_anchor_claim([
            (model_member_a, plan.opportunities[0]),
            (unrelated_member, plan.opportunities[0]),
        ])

    # A large connected component may be a star rather than a clique.  A
    # stable 16-member slice can contain the articulation member while its
    # two-member tail contains unrelated leaves.  The tail must remain on the
    # ordinary per-claim audit path instead of being presented as a jointly
    # coordinated group.
    star_surfaces = [f"Leaf {index:02d}" for index in range(1, 18)]
    star_hub = replace(
        model_member_a,
        hypothesis_id="star-00-hub",
        object_occurrence_key="occ-star-hub",
        predicate_metadata={
            **model_member_a.predicate_metadata,
            "source_object_surface": "Hub",
            "source_object_canonical_id": "star-hub",
            "source_frame_other_participants": star_surfaces,
        },
    )
    star_leaves = [
        replace(
            model_member_a,
            hypothesis_id=f"star-{index:02d}-leaf",
            object_occurrence_key=f"occ-star-leaf-{index:02d}",
            predicate_metadata={
                **model_member_a.predicate_metadata,
                "source_object_surface": surface,
                "source_object_canonical_id": f"star-leaf-{index:02d}",
                "source_frame_other_participants": ["Hub"],
            },
        )
        for index, surface in enumerate(star_surfaces, start=1)
    ]
    star_claims = [
        (item, plan.opportunities[0])
        for item in [star_hub, *star_leaves]
    ]
    star_components = source_projection_coordinated_audit_components(
        star_claims
    )
    assert star_components == (tuple(
        ["star-00-hub", *[
            f"star-{index:02d}-leaf" for index in range(1, 16)
        ]]
    ),)
    source_projection_group_anchor_claim([
        next(value for value in star_claims if value[0].hypothesis_id == item)
        for item in star_components[0]
    ])

    recovery_member_a = replace(
        model_member_a,
        hypothesis_id="recovery-member-a",
        object_occurrence_key="occ-recovery-member-a",
        predicate_metadata={
            **model_member_a.predicate_metadata,
            "source_object_surface": "Member A",
            "source_object_canonical_id": "recovery-member-a",
            "source_frame_other_participants": ["Member B", "Member C"],
            "source_frame_evidence_text": (
                "Member A, Member B, and Member C"
            ),
        },
    )
    recovery_member_b = replace(
        recovery_member_a,
        hypothesis_id="recovery-member-b",
        object_occurrence_key="occ-recovery-member-b",
        predicate_metadata={
            **recovery_member_a.predicate_metadata,
            "source_object_surface": "Member B",
            "source_object_canonical_id": "recovery-member-b",
            "source_frame_other_participants": ["Member A", "Member C"],
        },
    )
    recovery_member_c = replace(
        recovery_member_a,
        hypothesis_id="recovery-member-c",
        object_occurrence_key="occ-recovery-member-c",
        predicate_metadata={
            **recovery_member_a.predicate_metadata,
            "source_object_surface": "Member C",
            "source_object_canonical_id": "recovery-member-c",
            "source_frame_other_participants": ["Member A", "Member B"],
        },
    )

    def ordinary_attempt(item, verdict):
        supported = verdict == "SUPPORTED"
        return VerificationAttempt(
            attempt_id=f"attempt-{item.hypothesis_id}",
            hypothesis_id=item.hypothesis_id,
            attempt_no=1,
            verdict=verdict,
            reason_code=(
                "SUPPORTED" if supported
                else "MODEL_UNCERTAIN" if verdict == "UNCERTAIN"
                else "MODEL_REJECTED"
            ),
            supported_evidence_spans=(
                item.evidence_spans if supported else ()
            ),
            independent_call=True,
            verifier_prompt_version=SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
            verifier_schema_version=SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
            verifier_parser_version=SHADOW_RELATION_VERIFIER_PARSER_VERSION,
            verifier_model_revision="model-v1",
            forward_entailment=(
                "ENTAILED" if supported else "NOT_ENTAILED"
            ),
            reverse_entailment="NOT_ENTAILED",
            subject_role_supported=supported,
            object_role_supported=supported,
            normalized_subject_occurrence_key=item.subject_occurrence_key,
            normalized_object_occurrence_key=item.object_occurrence_key,
        )

    recovery_claims = [
        (item, plan.opportunities[0])
        for item in [
            recovery_member_a, recovery_member_b, recovery_member_c,
        ]
    ]
    attempts = {
        recovery_member_a.hypothesis_id: ordinary_attempt(
            recovery_member_a, "SUPPORTED"
        ),
        recovery_member_b.hypothesis_id: ordinary_attempt(
            recovery_member_b, "UNCERTAIN"
        ),
        recovery_member_c.hypothesis_id: ordinary_attempt(
            recovery_member_c, "UNCERTAIN"
        ),
    }
    recovery_plan = source_projection_coordinated_recovery_plan(
        recovery_claims, attempts
    )
    assert set(recovery_plan) == {
        recovery_member_b.hypothesis_id,
        recovery_member_c.hypothesis_id,
    }
    assert all(
        value["anchor_hypothesis_id"]
        == recovery_member_a.hypothesis_id
        and value["varying_slot"] == "object"
        for value in recovery_plan.values()
    )
    recovered = coordinated_source_projection_recovery_attempt(
        recovery_member_b,
        anchor_attempt=attempts[recovery_member_a.hypothesis_id],
        member_hypothesis_ids=recovery_plan[
            recovery_member_b.hypothesis_id
        ]["member_hypothesis_ids"],
        varying_slot="object",
        coordination_grounding=recovery_plan[
            recovery_member_b.hypothesis_id
        ]["coordination_grounding"],
        attempt_no=2,
    )
    assert recovered.reason_code == (
        "SUPPORTED_COORDINATED_SOURCE_PROJECTION"
    )
    assert recovered.object_context_role == "coordinated_source_argument"
    assert supported_for_normalization(recovered) is True
    # A definite ordinary semantic rejection is authoritative even when
    # another member of the same source list was supported.
    rejected_member_attempts = {
        **attempts,
        recovery_member_b.hypothesis_id: ordinary_attempt(
            recovery_member_b, "REJECTED"
        ),
    }
    assert source_projection_coordinated_recovery_plan(
        recovery_claims, rejected_member_attempts,
    ) == {}
    assert source_projection_coordinated_recovery_plan(
        recovery_claims,
        {
            key: replace(value, verdict="REJECTED",
                         supported_evidence_spans=())
            for key, value in attempts.items()
        },
    ) == {}

    prefix_member_a = replace(
        recovery_member_a,
        hypothesis_id="prefix-member-a",
        object_occurrence_key="occ-prefix-member-a",
        predicate_metadata={
            **recovery_member_a.predicate_metadata,
            "source_object_surface": "Member A",
            "source_object_canonical_id": "prefix-member-a",
            "source_frame_other_participants": [],
            "source_frame_evidence_span": [0, len("Member A")],
            "source_frame_evidence_text": "Member A",
        },
    )
    prefix_b_text = "Member A, Member B"
    prefix_member_b = replace(
        prefix_member_a,
        hypothesis_id="prefix-member-b",
        object_occurrence_key="occ-prefix-member-b",
        predicate_metadata={
            **prefix_member_a.predicate_metadata,
            "source_object_surface": "Member B",
            "source_object_canonical_id": "prefix-member-b",
            "source_frame_other_participants": ["Member A"],
            "source_frame_evidence_span": [0, len(prefix_b_text)],
            "source_frame_evidence_text": prefix_b_text,
        },
    )
    prefix_c_text = "Member A, Member B, and Member C"
    prefix_member_c = replace(
        prefix_member_a,
        hypothesis_id="prefix-member-c",
        object_occurrence_key="occ-prefix-member-c",
        predicate_metadata={
            **prefix_member_a.predicate_metadata,
            "source_object_surface": "Member C",
            "source_object_canonical_id": "prefix-member-c",
            "source_frame_other_participants": ["Member A", "Member B"],
            "source_frame_evidence_span": [0, len(prefix_c_text)],
            "source_frame_evidence_text": prefix_c_text,
        },
    )
    prefix_claims = [
        (item, plan.opportunities[0])
        for item in [prefix_member_a, prefix_member_b, prefix_member_c]
    ]
    prefix_attempts = {
        prefix_member_a.hypothesis_id: ordinary_attempt(
            prefix_member_a, "SUPPORTED"
        ),
        prefix_member_b.hypothesis_id:
            source_projection_role_audit_terminal_attempt(
                prefix_member_b,
                result="CONFLICT",
                verifier_model_revision="model-v1",
            ),
        prefix_member_c.hypothesis_id: ordinary_attempt(
            prefix_member_c, "UNCERTAIN"
        ),
    }
    prefix_plan = source_projection_coordinated_recovery_plan(
        prefix_claims, prefix_attempts,
    )
    assert set(prefix_plan) == {
        prefix_member_b.hypothesis_id,
        prefix_member_c.hypothesis_id,
    }
    assert prefix_plan[prefix_member_b.hypothesis_id][
        "coordination_grounding"
    ]["surface_sequence"] == ["Member A", "Member B", "Member C"]

    clause_text = "Member A and Member B, Member C performs another action"
    clause_claims = []
    for item, surface in (
        (recovery_member_a, "Member A"),
        (recovery_member_b, "Member B"),
        (recovery_member_c, "Member C"),
    ):
        clause_claims.append((replace(
            item,
            predicate_metadata={
                **item.predicate_metadata,
                "source_frame_evidence_span": [0, len(clause_text)],
                "source_frame_evidence_text": clause_text,
                "source_object_surface": surface,
            },
        ), plan.opportunities[0]))
    assert source_projection_coordinated_recovery_plan(
        clause_claims,
        {
            clause_claims[0][0].hypothesis_id: ordinary_attempt(
                clause_claims[0][0], "SUPPORTED"
            ),
            clause_claims[1][0].hypothesis_id: ordinary_attempt(
                clause_claims[1][0], "UNCERTAIN"
            ),
            clause_claims[2][0].hypothesis_id: ordinary_attempt(
                clause_claims[2][0], "UNCERTAIN"
            ),
        },
    ) == {}

    unavailable_prefix_attempts = dict(prefix_attempts)
    unavailable_prefix_attempts[prefix_member_b.hypothesis_id] = (
        source_projection_role_audit_terminal_attempt(
            prefix_member_b,
            result="UNAVAILABLE",
            verifier_model_revision="model-v1",
        )
    )
    assert source_projection_coordinated_recovery_plan(
        prefix_claims, unavailable_prefix_attempts,
    ) == {}

    # A narrower nested frame can be connected to an outer inventory through
    # model participant hints.  It belongs to the joint-audit component, but
    # must neither poison nor join the exact-span coordination cohort used for
    # deterministic recovery.
    recovery_member_a_with_nested_hint = replace(
        recovery_member_a,
        predicate_metadata={
            **recovery_member_a.predicate_metadata,
            "source_frame_other_participants": [
                "Member B", "Member C", "Member A detail",
            ],
        },
    )
    nested_span_bridge = replace(
        recovery_member_a,
        hypothesis_id="nested-span-bridge",
        object_occurrence_key="occ-nested-span-bridge",
        predicate_metadata={
            **recovery_member_a.predicate_metadata,
            "source_object_surface": "Member A detail",
            "source_object_canonical_id": "nested-span-bridge",
            "source_frame_other_participants": ["Member A"],
            "source_frame_evidence_span": [5, 20],
            "source_frame_evidence_text": "Member A detail",
        },
    )
    nested_bridge_claims = [
        (item, plan.opportunities[0])
        for item in [
            recovery_member_a_with_nested_hint,
            recovery_member_b,
            recovery_member_c,
            nested_span_bridge,
        ]
    ]
    assert len(source_projection_coordinated_audit_components(
        nested_bridge_claims
    )[0]) == 4
    nested_bridge_attempts = {
        **attempts,
        nested_span_bridge.hypothesis_id: ordinary_attempt(
            nested_span_bridge, "REJECTED"
        ),
    }
    nested_bridge_plan = source_projection_coordinated_recovery_plan(
        nested_bridge_claims, nested_bridge_attempts,
    )
    assert set(nested_bridge_plan) == {
        recovery_member_b.hypothesis_id,
        recovery_member_c.hypothesis_id,
    }
    assert all(
        value["member_hypothesis_ids"] == [
            "recovery-member-a",
            "recovery-member-b",
            "recovery-member-c",
        ]
        for value in nested_bridge_plan.values()
    )

    # A model participant list is not itself coordination proof.  Nested noun
    # phrases can expose the same fixed endpoint and reciprocal participant
    # hints while the source words between them express ordinary dependency,
    # not a coordinate connector.  A supported outer argument must not be
    # propagated to the nested one.
    nested_environment = replace(
        recovery_member_a,
        hypothesis_id="nested-environment",
        object_occurrence_key="occ-nested-environment",
        predicate_metadata={
            **recovery_member_a.predicate_metadata,
            "source_object_surface": "Environment",
            "source_object_canonical_id": "nested-environment",
            "source_frame_other_participants": ["Software"],
            "source_frame_evidence_text": (
                "Alpha data comes from Environment information about Software"
            ),
        },
    )
    nested_software = replace(
        nested_environment,
        hypothesis_id="nested-software",
        object_occurrence_key="occ-nested-software",
        predicate_metadata={
            **nested_environment.predicate_metadata,
            "source_object_surface": "Software",
            "source_object_canonical_id": "nested-software",
            "source_frame_other_participants": ["Environment"],
        },
    )
    nested_claims = [
        (nested_environment, plan.opportunities[0]),
        (nested_software, plan.opportunities[0]),
    ]
    assert source_projection_coordinated_audit_components(
        nested_claims
    )
    assert source_projection_coordinated_recovery_plan(
        nested_claims,
        {
            nested_environment.hypothesis_id: ordinary_attempt(
                nested_environment, "SUPPORTED"
            ),
            nested_software.hypothesis_id: ordinary_attempt(
                nested_software, "REJECTED"
            ),
        },
    ) == {}

    reverse = replace(
        base,
        hypothesis_id="reverse-hypothesis",
        subject_occurrence_key=base.object_occurrence_key,
        object_occurrence_key=base.subject_occurrence_key,
        predicate_metadata={
            **base.predicate_metadata,
            "source_subject_surface": "Store",
            "source_object_surface": "Alpha",
            "source_subject_canonical_id": base.predicate_metadata[
                "source_object_canonical_id"
            ],
            "source_object_canonical_id": base.predicate_metadata[
                "source_subject_canonical_id"
            ],
            "source_subject_argument_index": 1,
            "source_object_argument_index": 0,
        },
    )
    assert source_projection_competition_risks(
        [base, reverse], logical_schema=schema,
    ) == frozenset()
    competition = {
        base.hypothesis_id: (
            competitor.hypothesis_id, competing_object.hypothesis_id,
        ),
        competitor.hypothesis_id: (base.hypothesis_id,),
        competing_object.hypothesis_id: (base.hypothesis_id,),
    }
    assert source_projection_competition_components(competition) == ((
        *sorted({
            base.hypothesis_id,
            competitor.hypothesis_id,
            competing_object.hypothesis_id,
        }),
    ),)
    group_claims = [
        (base, plan.opportunities[0]),
        (reverse, plan.opportunities[0]),
    ]
    group_payload = json.loads(source_projection_group_audit_prompt(
        group_claims,
        content=content,
        logical_schema=schema,
    ))
    group_ids = [
        value["hypothesis_id"] for value in group_payload["candidates"]
    ]
    assert group_ids == sorted({
        base.hypothesis_id, reverse.hypothesis_id,
    })
    for semantic_guard in (
        "relation-meaning equivalence",
        "argument-0 to argument-1",
        "provider or other governing role",
        "implication",
    ):
        assert semantic_guard in group_payload["instruction"]
    group_schema = source_projection_group_audit_response_schema(group_ids)
    assert group_schema["required"] == group_ids
    assert set(group_schema["properties"]) == set(group_ids)
    group_raw = {"statuses": [{
        "hypothesis_id": value,
        "projection_status": (
            "ALIGNED" if value == base.hypothesis_id else "CONFLICT"
        ),
    } for value in group_ids]}
    assert parse_source_projection_group_audit(
        group_raw, hypothesis_ids=group_ids,
    )[base.hypothesis_id] == "ALIGNED"
    keyed_group_raw = {
        value: (
            "ALIGNED" if value == base.hypothesis_id else "CONFLICT"
        )
        for value in group_ids
    }
    assert parse_source_projection_group_audit(
        keyed_group_raw, hypothesis_ids=group_ids,
    ) == parse_source_projection_group_audit(
        group_raw, hypothesis_ids=group_ids,
    )
    with pytest.raises(ShadowClassificationError):
        parse_source_projection_group_audit(
            {"statuses": group_raw["statuses"][:1]},
            hypothesis_ids=group_ids,
        )
    group_terminal = source_projection_group_audit_terminal_attempt(
        base,
        result="CONFLICT",
        verifier_model_revision="model-v1",
    )
    assert group_terminal.verdict == "REJECTED"
    assert group_terminal.reason_code == "SOURCE_PROJECTION_GROUP_CONFLICT"
    assert not supported_for_normalization(group_terminal)
    audit_payload = json.loads(source_projection_role_audit_prompt(
        hypothesis,
        plan.opportunities[0],
        content=content,
        logical_schema=schema,
        competing_projections=[competitor, competing_object],
    ))
    assert audit_payload["source"] == content
    for semantic_guard in (
        "relation-meaning equivalence",
        "argument-0 to argument-1",
        "provider or other governing role",
        "implication",
    ):
        assert semantic_guard in audit_payload["instruction"]
    assert "Nearby participant" in audit_payload["candidate"][
        "other_participants"
    ]
    assert "Nearby object" in audit_payload["candidate"][
        "other_participants"
    ]
    assert {
        (
            value["subject"], value["object"],
            value["subject_argument_index"],
            value["object_argument_index"],
        )
        for value in audit_payload["competing_bindings"]
    } == {
        ("Nearby participant", "Store", 0, 1),
        ("Alpha", "Nearby object", 0, 1),
    }
    assert list(
        source_projection_role_audit_response_schema()["properties"]
    ) == ["projection_status"]
    assert parse_source_projection_role_audit({
        "projection_status": "ALIGNED",
    }) == "ALIGNED"
    assert source_role_audit_conflict_is_terminal({
        "reason": "BATCH_SUPPORTED_COMPETING_SOURCE_PROJECTION",
    })
    assert source_role_audit_conflict_is_terminal({
        "reason": "BATCH_SUPPORTED_RECALL_SHORTLIST",
    })
    terminal = source_projection_role_audit_terminal_attempt(
        hypothesis,
        result="CONFLICT",
        verifier_model_revision="model-v1",
    )
    assert terminal.verdict == "REJECTED"
    assert terminal.reason_code == "SOURCE_PROJECTION_ROLE_CONFLICT"


def test_reordered_open_relation_binding_is_a_mandatory_role_audit():
    content = "Platform provides Capability."
    schema = _schema()
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-reordered",
        chunk_key="chunk-reordered",
        content=content,
        mentions=[
            _mention("platform", 0, 8, "Platform"),
            _mention("capability", 18, 28, "Capability"),
        ],
        logical_schema=schema,
        run_id="run-reordered",
        candidate_policy_version="candidate-reordered",
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "provides",
            "arguments": ["Platform", "Capability"],
            "argument_occurrence_keys": [
                "occ-platform", "occ-capability",
            ],
            "roles": ["subject", "object"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Platform", "Capability"],
            "assertion_status": "ASSERTED",
            "evidence_span": [0, len(content)],
            "evidence_text": content,
        }],
    }
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        {"coverage_complete": True, "bindings": [{
            "frame_index": 0,
            "predicate": "alpha_rel",
            "subject_index": 1,
            "object_index": 0,
        }]},
        occurrences=[
            {"occurrence_key": "occ-platform", "surface": "Platform"},
            {"occurrence_key": "occ-capability", "surface": "Capability"},
        ],
        opportunities_by_pair={
            plan.opportunities[0].pair_identity: plan.opportunities[0],
        },
        logical_schema=schema,
        classifier_model_revision="model-v1",
    )[0]

    review = source_batch_focused_review(
        hypothesis, {"verdict": "SUPPORTED"}, logical_schema=schema,
    )
    assert review == {
        "kind": "confirmation",
        "reason": "BATCH_SUPPORTED_REORDERED_OPEN_ORIENTATION",
        "priority": -1,
        "source_role_audit": True,
        "mandatory": True,
    }


def test_schema_declared_source_predicate_conflict_is_contrastively_bounded():
    content = "Alpha connects Beta."
    mentions = [
        _mention("alpha", 0, 5, "Alpha"),
        _mention("beta", 15, 19, "Beta"),
    ]
    schema = logical_schema(
        predicate_policy(
            "topical_link",
            semantic_definition="The subject content is about the object.",
            conflicts_with=["functional_use"],
        ),
        predicate_policy(
            "functional_use",
            semantic_definition="The subject is used to accomplish the object.",
            conflicts_with=["topical_link"],
        ),
    )
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_char_distance=100,
    )
    opportunity = plan.opportunities[0]
    inventory = {"frames": [{
        "relation": "connects",
        "arguments": ["Alpha", "Beta"],
        "roles": ["grammatical_subject", "direct_argument"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Alpha", "Beta"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["occ-alpha", "occ-beta"],
        "evidence_span": [0, len(content)],
        "evidence_text": content,
        "deterministic_source_syntax": "intervening_predicate_construction",
    }]}
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [
            {
                "frame_index": 0, "predicate": "topical_link",
                "subject_index": 0, "object_index": 1,
            },
            {
                "frame_index": 0, "predicate": "functional_use",
                "subject_index": 0, "object_index": 1,
            },
        ]},
        occurrences=[
            {
                "occurrence_key": "occ-alpha", "surface": "Alpha",
                "canonical_id": "alpha", "type": "GenericEntity",
            },
            {
                "occurrence_key": "occ-beta", "surface": "Beta",
                "canonical_id": "beta", "type": "GenericEntity",
            },
        ],
        opportunities_by_pair={opportunity.pair_identity: opportunity},
        logical_schema=schema,
        classifier_model_revision="model-v1",
    )
    groups = conflicting_source_hypothesis_groups(
        hypotheses, logical_schema=schema,
    )
    assert len(groups) == 1
    payload = __import__("json").loads(source_predicate_conflict_prompt(
        groups[0], opportunity, content=content, logical_schema=schema,
    ))
    assert {item["relation"] for item in payload["candidates"]} == {
        "topical_link", "functional_use",
    }
    response_schema = source_predicate_conflict_response_schema(
        groups[0], opportunity, content=content,
    )
    assert response_schema["properties"]["relations"]["maxItems"] == 1
    assert parse_source_predicate_conflict(
        {"relations": [{
            "relation": "topical_link",
        }]},
        hypotheses=groups[0],
        opportunity=opportunity,
        content=content,
    ) == "topical_link"
    with pytest.raises(ShadowClassificationError):
        parse_source_predicate_conflict(
            {"relations": [
                {
                    "relation": "topical_link",
                },
                {
                    "relation": "functional_use",
                },
            ]},
            hypotheses=groups[0],
            opportunity=opportunity,
            content=content,
        )


def test_source_binding_indexes_compact_binary_frames_and_maps_back_to_inventory():
    inventory = {
        "coverage_complete": True,
        "frames": [
            {
                "relation": "is",
                "arguments": ["A state"],
                "roles": ["subject"],
                "argument_kinds": ["STATE"],
                "entity_heads": [""],
                "assertion_status": "ASSERTED",
            },
            {
                "relation": "is",
                "arguments": ["Noise", "Value"],
                "roles": ["subject", "object"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Noise", "Value"],
                "assertion_status": "ASSERTED",
            },
            {
                "relation": "contains",
                "arguments": ["Alpha", "Beta"],
                "roles": ["subject", "object"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Alpha", "Beta"],
                "assertion_status": "ASSERTED",
            },
            {
                "relation": "is used for",
                "arguments": ["A statement"],
                "roles": ["subject"],
                "argument_kinds": ["ENTITY"],
                "entity_heads": ["statement"],
                "assertion_status": "ASSERTED",
            },
            {
                "relation": "links",
                "arguments": ["Gamma", "Delta"],
                "roles": ["subject", "object"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Gamma", "Delta"],
                "assertion_status": "ASSERTED",
            },
        ],
    }
    schema = _schema()
    prompt = source_frame_schema_binding_prompt(inventory, logical_schema=schema)
    payload = __import__("json").loads(prompt)
    assert [item["frame_index"] for item in payload["frames"]] == [0, 1]
    assert payload["frames"][0]["relation"] == "contains"
    assert payload["frames"][1]["relation"] == "links"
    parsed = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 1,
                "predicate": "alpha_rel",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory=inventory,
        logical_schema=schema,
    )
    assert parsed["bindings"][0]["frame_index"] == 4


def test_source_binding_excludes_strict_structures_and_disabled_review_queue():
    inventory = {
        "coverage_complete": True,
        "frames": [
            {
                "relation": "assigns",
                "arguments": ["Row", "Value"],
                "roles": ["row_key", "column_value"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Row", "Value"],
                "assertion_status": "ASSERTED",
                "deterministic_source_structure": "table_header_assignment",
                "structural_role_label": "status",
            },
            {
                "relation": "connects",
                "arguments": ["Alpha", "Beta"],
                "roles": ["subject", "object"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Alpha", "Beta"],
                "assertion_status": "ASSERTED",
            },
        ],
    }
    schema = logical_schema(
        predicate_policy("alpha_rel"),
        predicate_policy(
            "review_rel",
            publish_mode="review_only",
            structural_evidence_kinds=["table_header_assignment"],
            structural_role_labels=["status"],
        ),
    )
    payload = json.loads(source_frame_schema_binding_prompt(
        inventory,
        logical_schema=schema,
        include_review_only=False,
    ))
    assert [item["relation"] for item in payload["frames"]] == ["connects"]
    assert [item["name"] for item in payload["predicates"]] == ["alpha_rel"]
    assert source_frame_schema_binding_response_schema(
        schema,
        include_review_only=False,
    )["properties"]["bindings"]["items"]["properties"]["predicate"][
        "enum"
    ] == ["alpha_rel"]
    assert deterministic_source_structure_schema_bindings(
        inventory,
        logical_schema=schema,
        include_review_only=False,
    )["bindings"] == []

    parsed = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "review_rel",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory=inventory,
        logical_schema=schema,
        allow_partial=True,
        include_review_only=False,
    )
    assert parsed["bindings"] == []
    assert parsed["dropped_bindings"][0]["reason"] == (
        "source Schema binding selected disabled review candidate"
    )


def test_source_binding_partial_mode_keeps_valid_items_and_audits_invalid_items():
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "connects",
            "arguments": ["Alpha", "Beta"],
            "roles": ["subject", "object"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Alpha", "Beta"],
            "assertion_status": "ASSERTED",
        }],
    }
    parsed = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [
                {
                    "frame_index": 0,
                    "predicate": "alpha_rel",
                    "subject_index": 0,
                    "object_index": 1,
                },
                {
                    "frame_index": 0,
                    "predicate": "alpha_rel",
                    "subject_index": 0,
                    "object_index": 2,
                },
            ],
        },
        inventory=inventory,
        logical_schema=_schema(),
        allow_partial=True,
    )
    assert parsed["coverage_complete"] is False
    assert len(parsed["bindings"]) == 1
    assert parsed["dropped_bindings"] == [{
        "item_index": 1,
        "reason": "source Schema binding selected unknown argument",
    }]


def test_bounded_relation_context_preserves_absolute_endpoints_and_marks_omissions():
    content = "A" * 1000 + "LEFT" + "B" * 2200 + "RIGHT" + "C" * 1000
    window = bounded_relation_context(
        content,
        (0, len(content)),
        focus_spans=((1000, 1004), (3204, 3209)),
        max_chars=256,
    )
    assert window["truncated"] is True
    assert window["source_span"] == [0, len(content)]
    assert window["visible_segments"] == [[0, 128], [544, 800]] or len(
        window["visible_segments"]
    ) == 2
    assert "LEFT" in window["text"]
    assert "RIGHT" in window["text"]
    assert "OMITTED_SOURCE" in window["text"]


def test_source_binding_prompt_explicitly_tests_role_based_reverse_direction():
    prompt = source_frame_schema_binding_prompt(
        {
            "coverage_complete": True,
            "frames": [{
                "relation": "contains",
                "arguments": ["Report", "Dataset"],
                "roles": ["whole", "member"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Report", "Dataset"],
                "assertion_status": "ASSERTED",
            }],
        },
        logical_schema=_schema(),
    )
    assert "role-based orientation" in prompt
    assert "whole, container, owner, or source" in prompt


def test_source_binding_prompt_preserves_nary_syntax_projection_roles():
    payload = __import__("json").loads(source_frame_schema_binding_prompt(
        {
            "coverage_complete": True,
            "frames": [{
                "relation": "transfers to",
                "arguments": ["Payload", "Destination"],
                "roles": ["action_theme", "role_complement"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Payload", "Destination"],
                "assertion_status": "ASSERTED",
                "deterministic_source_syntax": (
                    "ordered_three_role_construction"
                ),
            }],
        },
        logical_schema=_schema(),
    ))
    assert payload["frames"][0]["source_syntax_role_projection"] == (
        "action_theme->role_complement"
    )
    assert "Never copy the governing verb's Schema predicate" in payload[
        "instruction"
    ]


def test_source_binding_recall_shortlist_adds_one_contrastive_candidate():
    schema = logical_schema(
        predicate_policy(
            "links_to",
            semantic_definition=(
                "The linking artifact links to an endpoint."
            ),
            positive_examples=["Map links Service."],
        ),
        predicate_policy(
            "mapped_to",
            semantic_definition=(
                "The subject item is mapped to the object requirement."
            ),
            positive_examples=[
                "Service -> Requirement: "
                "The map links a Service to a Requirement."
            ],
        ),
    )
    inventory = {"frames": [{
        "relation": "links to",
        "arguments": ["Service", "Requirement"],
        "roles": ["action_theme", "role_complement"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Service", "Requirement"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["service", "requirement"],
        "evidence_text": "The map links a Service to a Requirement.",
        "deterministic_source_syntax": "ordered_three_role_construction",
    }]}
    result = source_frame_schema_recall_shortlist(
        inventory,
        [{
            "frame_index": 0,
            "predicate": "links_to",
            "subject_index": 0,
            "object_index": 1,
        }],
        occurrences=[
            {
                "occurrence_key": "service",
                "surface": "Service",
                "type": "GenericEntity",
                "canonical_id": "service",
            },
            {
                "occurrence_key": "requirement",
                "surface": "Requirement",
                "type": "GenericEntity",
                "canonical_id": "requirement",
            },
        ],
        logical_schema=schema,
    )
    assert len(result["bindings"]) == 1
    assert result["bindings"][0]["predicate"] == "mapped_to"
    assert result["bindings"][0]["binding_source"] == (
        "DETERMINISTIC_SEMANTIC_SHORTLIST"
    )


def test_source_binding_recall_shortlist_never_broadens_strict_structure():
    inventory = {"frames": [{
        "relation": "column role: Value",
        "arguments": ["Row", "Value"],
        "roles": ["row_key", "column_value"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Row", "Value"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["row", "value"],
        "evidence_text": "| Row | Value |",
        "deterministic_source_structure": "table_header_assignment",
    }]}
    result = source_frame_schema_recall_shortlist(
        inventory,
        [],
        occurrences=[
            {
                "occurrence_key": "row", "surface": "Row",
                "type": "GenericEntity", "canonical_id": "row",
            },
            {
                "occurrence_key": "value", "surface": "Value",
                "type": "GenericEntity", "canonical_id": "value",
            },
        ],
        logical_schema=_schema(),
    )
    assert result["bindings"] == []
    assert result["frames_considered"] == 0


def test_parser_proved_table_binding_uses_narrow_structural_audit():
    content = "Regions | Compilation"
    mentions = [
        _mention("regions", 0, 7, "Regions"),
        _mention("compilation", 10, 21, "Compilation"),
    ]
    schema = logical_schema(predicate_policy(
        "derived_from",
        semantic_definition="The subject data set is derived from the object source.",
        direction_description="data set to source",
        structural_evidence_kinds=["table_header_assignment"],
        structural_role_labels=["Source"],
    ))
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_char_distance=100,
    )
    inventory = {"coverage_complete": True, "frames": [{
        "relation": "column role: Source",
        "arguments": ["Regions", "Compilation"],
        "argument_occurrence_keys": ["occ-regions", "occ-compilation"],
        "roles": ["row_key", "column_value"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Regions", "Compilation"],
        "assertion_status": "ASSERTED",
        "evidence_span": [0, len(content)],
        "evidence_text": content,
        "deterministic_source_structure": "table_header_assignment",
        "structural_role_label": "Source",
    }]}
    bindings = deterministic_source_structure_schema_bindings(
        inventory=inventory,
        logical_schema=schema,
    )
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        bindings,
        occurrences=[
            {"occurrence_key": "occ-regions", "surface": "Regions"},
            {"occurrence_key": "occ-compilation", "surface": "Compilation"},
        ],
        opportunities_by_pair={
            plan.opportunities[0].pair_identity: plan.opportunities[0],
        },
        logical_schema=schema,
        classifier_model_revision="model-v1",
    )[0]
    prompt = json.loads(source_structure_binding_audit_prompt(
        hypothesis,
        plan.opportunities[0],
        content=content,
        logical_schema=schema,
    ))
    assert prompt["source"]["structure_kind"] == "table_header_assignment"
    assert prompt["source"]["role_label"] == "Source"
    assert list(
        source_structure_binding_audit_response_schema()["properties"]
    ) == ["structure_status"]
    assert parse_source_structure_binding_audit({
        "structure_status": "ALIGNED",
    }) == "ALIGNED"
    attempt = source_structure_binding_audit_terminal_attempt(
        hypothesis,
        plan.opportunities[0],
        content=content,
        logical_schema=schema,
        result="ALIGNED",
        verifier_model_revision="model-v1",
    )
    assert attempt.verdict == "SUPPORTED"
    assert attempt.forward_entailment == "ENTAILED"
    assert attempt.reverse_entailment == "NOT_ENTAILED"
    validation = validate_hypothesis(
        hypothesis,
        plan.opportunities[0],
        content=content,
        logical_schema=schema,
    )
    deterministic = deterministic_parser_structure_terminal_attempt(
        hypothesis,
        validation,
        plan.opportunities[0],
        content=content,
        logical_schema=schema,
    )
    assert deterministic.verdict == "SUPPORTED"
    assert deterministic.reason_code == (
        "SUPPORTED_PARSER_STRUCTURE_CONTRACT"
    )
    assert deterministic.model_request_payload["model_call"] is False
    assert supported_for_normalization(deterministic) is True


def test_parser_proved_table_role_uses_open_carrier_when_schema_is_old():
    content = "Regions | Compilation"
    mentions = [
        _mention("regions", 0, 7, "Regions"),
        _mention("compilation", 10, 21, "Compilation"),
    ]
    carrier_policy = predicate_policy(
        "source_assertion",
        semantic_definition=(
            "The exact open source expression relates subject to object."
        ),
        open_relation_policy={
            "enabled": True,
            "fallback_only": True,
            "allowed_candidate_sources": [
                "ATOMIC_FRAME_SCHEMA_PROJECTION",
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            ],
            "display_source_expression": True,
            "identity_source_expression": True,
            "require_exact_source_expression": True,
        },
    )
    schema = logical_schema(carrier_policy)
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v1",
        max_char_distance=100,
    )
    inventory = {"coverage_complete": True, "frames": [{
        "relation": "column role: Source",
        "arguments": ["Regions", "Compilation"],
        "argument_occurrence_keys": ["occ-regions", "occ-compilation"],
        "roles": ["row_key", "column_value"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Regions", "Compilation"],
        "assertion_status": "ASSERTED",
        "evidence_span": [0, len(content)],
        "evidence_text": content,
        "deterministic_source_structure": "table_header_assignment",
        "structural_role_label": "Source",
    }]}
    bindings = deterministic_source_structure_schema_bindings(
        inventory=inventory,
        logical_schema=schema,
    )
    assert bindings["bindings"] == [{
        "frame_index": 0,
        "predicate": "source_assertion",
        "subject_index": 0,
        "object_index": 1,
        "binding_source": (
            "DETERMINISTIC_SOURCE_STRUCTURE_OPEN_CARRIER"
        ),
        "source_structure_kind": "table_header_assignment",
    }]
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        bindings,
        occurrences=[
            {"occurrence_key": "occ-regions", "surface": "Regions"},
            {
                "occurrence_key": "occ-compilation",
                "surface": "Compilation",
            },
        ],
        opportunities_by_pair={
            plan.opportunities[0].pair_identity: plan.opportunities[0],
        },
        logical_schema=schema,
        classifier_model_revision="model-v1",
    )[0]
    validation = validate_hypothesis(
        hypothesis,
        plan.opportunities[0],
        content=content,
        logical_schema=schema,
    )
    attempt = deterministic_parser_structure_terminal_attempt(
        hypothesis,
        validation,
        plan.opportunities[0],
        content=content,
        logical_schema=schema,
    )
    assert parser_structure_claim_intent(
        hypothesis,
        plan.opportunities[0],
        logical_schema=schema,
    ) is True
    assert attempt.verdict == "SUPPORTED"
    assert attempt.reason_code == "SUPPORTED_PARSER_STRUCTURE_CONTRACT"


def test_collective_shortlist_only_retrieves_reciprocal_schema_relation():
    schema = logical_schema(
        predicate_policy(
            "acts_on_topic",
            semantic_definition="A participant acts on a shared topic.",
            positive_cues=["agree on"],
        ),
        predicate_policy(
            "mutual_alignment",
            symmetric=True,
            semantic_definition=(
                "Two participants mutually align with one another."
            ),
            positive_examples=["Recipient and Agency agree together."],
        ),
    )
    inventory = {"frames": [{
        "relation": "collective participants in agree on",
        "arguments": ["Recipient", "Agency"],
        "roles": ["collective_participant", "collective_participant"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Recipient", "Agency"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["recipient", "agency"],
        "evidence_text": "Recipient and Agency agree on the terms.",
        "deterministic_source_syntax": (
            "coordinated_collective_relation_candidate"
        ),
    }]}
    result = source_frame_schema_recall_shortlist(
        inventory,
        [],
        occurrences=[
            {
                "occurrence_key": "recipient", "surface": "Recipient",
                "type": "GenericEntity", "canonical_id": "recipient",
            },
            {
                "occurrence_key": "agency", "surface": "Agency",
                "type": "GenericEntity", "canonical_id": "agency",
            },
        ],
        logical_schema=schema,
        minimum_score=0.0,
    )
    assert [item["predicate"] for item in result["bindings"]] == [
        "mutual_alignment",
    ]


def test_compiled_role_contract_shortlist_recovers_inverse_projection():
    schema = logical_schema(predicate_policy(
        "member_of",
        semantic_definition=(
            "The subject is a contained member and the object is its container."
        ),
        positive_examples=["Container contains Member."],
        source_role_pairs=["action_theme->grammatical_subject"],
    ))
    inventory = {"frames": [{
        "relation": "contains",
        "arguments": ["Container", "Member"],
        "roles": ["grammatical_subject", "action_theme"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Container", "Member"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["container", "member"],
        "evidence_text": "Container contains Member.",
        "deterministic_source_syntax": "ordered_three_role_construction",
    }]}
    result = source_frame_schema_recall_shortlist(
        inventory,
        [],
        occurrences=[
            {
                "occurrence_key": "container", "surface": "Container",
                "type": "GenericEntity", "canonical_id": "container",
            },
            {
                "occurrence_key": "member", "surface": "Member",
                "type": "GenericEntity", "canonical_id": "member",
            },
        ],
        logical_schema=schema,
        minimum_score=0.0,
    )
    assert len(result["bindings"]) == 1
    assert result["bindings"][0]["predicate"] == "member_of"
    assert result["bindings"][0]["subject_index"] == 1
    assert result["bindings"][0]["object_index"] == 0


def test_source_binding_parser_rejects_model_override_of_strict_structure():
    inventory = {"frames": [{
        "relation": "column role: Value",
        "arguments": ["Row", "Value"],
        "roles": ["row_key", "column_value"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Row", "Value"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["row", "value"],
        "evidence_text": "| Row | Value |",
        "deterministic_source_structure": "table_header_assignment",
        "structural_role_label": "Value",
    }]}
    result = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "alpha_rel",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory=inventory,
        logical_schema=_schema(),
        allow_partial=True,
    )
    assert result["bindings"] == []
    assert result["dropped_bindings"][0]["reason"] == (
        "source Schema binding selected unknown frame"
    )


def test_source_binding_parser_enforces_compiled_semantic_role_pair():
    inventory = {"frames": [{
        "relation": "transfers to",
        "arguments": ["Actor", "Destination"],
        "roles": ["grammatical_subject", "role_complement"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Actor", "Destination"],
        "assertion_status": "ASSERTED",
        "deterministic_source_syntax": "ordered_three_role_construction",
    }]}
    schema = logical_schema(predicate_policy(
        "acts_on_theme",
        source_role_pairs=["grammatical_subject->action_theme"],
    ))
    result = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "acts_on_theme",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory=inventory,
        logical_schema=schema,
        allow_partial=True,
    )
    assert result["bindings"] == []
    assert result["dropped_bindings"][0]["reason"] == (
        "source Schema binding violates semantic source-role contract"
    )


def test_source_binding_role_contract_applies_to_parser_audited_model_frame():
    inventory = {"frames": [{
        "relation": "contains",
        "arguments": ["Catalog", "Topic"],
        "roles": ["subject", "object"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Catalog", "Topic"],
        "assertion_status": "ASSERTED",
        "parser_inferred_source_roles": [
            "grammatical_subject", "role_complement",
        ],
        "parser_inferred_source_syntax": (
            "ordered_three_role_construction"
        ),
    }]}
    schema = logical_schema(predicate_policy(
        "member_of",
        source_role_pairs=["action_theme->grammatical_subject"],
    ))
    result = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "member_of",
                "subject_index": 1,
                "object_index": 0,
            }],
        },
        inventory=inventory,
        logical_schema=schema,
        allow_partial=True,
    )
    assert result["bindings"] == []
    assert result["dropped_bindings"][0]["reason"] == (
        "source Schema binding violates semantic source-role contract"
    )


def test_source_binding_parser_rejects_negated_frame_for_positive_predicate():
    inventory = {"frames": [{
        "relation": "is not",
        "arguments": ["Neighbor", "Relation"],
        "roles": ["grammatical_subject", "direct_argument"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Neighbor", "Relation"],
        "assertion_status": "ASSERTED",
        "deterministic_source_syntax": "intervening_predicate_construction",
    }]}
    result = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "alpha_rel",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory=inventory,
        logical_schema=_schema(),
        allow_partial=True,
    )
    assert result["bindings"] == []
    assert result["dropped_bindings"][0]["reason"] == (
        "source Schema binding violates source-negation policy"
    )


def test_source_binding_negation_policy_allows_explicit_negative_relation():
    inventory = {"frames": [{
        "relation": "is not the same as",
        "arguments": ["Alpha", "Beta"],
        "roles": ["grammatical_subject", "direct_argument"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Alpha", "Beta"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["alpha", "beta"],
        "evidence_text": "Alpha is not the same as Beta.",
        "deterministic_source_syntax": "intervening_predicate_construction",
    }]}
    schema = logical_schema(predicate_policy(
        "different_from",
        symmetric=True,
        semantic_definition="The endpoints are explicitly different.",
        positive_examples=["Alpha is not the same as Beta."],
        negation_policy=ALLOW_EXPLICIT_NEGATIVE_RELATION,
    ))
    parsed = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "different_from",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory=inventory,
        logical_schema=schema,
    )
    assert len(parsed["bindings"]) == 1
    prompt = __import__("json").loads(source_frame_schema_binding_prompt(
        inventory, logical_schema=schema,
    ))
    assert prompt["frames"][0]["explicit_source_negation"] is True
    assert prompt["predicates"][0]["negation_policy"] == (
        ALLOW_EXPLICIT_NEGATIVE_RELATION
    )


def test_source_binding_shortlist_never_maps_negation_to_positive_predicate():
    inventory = {"frames": [{
        "relation": "is not",
        "arguments": ["Neighbor", "Relation"],
        "roles": ["grammatical_subject", "direct_argument"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Neighbor", "Relation"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["neighbor", "relation"],
        "evidence_text": "A Neighbor is not a Relation.",
        "deterministic_source_syntax": "intervening_predicate_construction",
    }]}
    result = source_frame_schema_recall_shortlist(
        inventory,
        [],
        occurrences=[
            {
                "occurrence_key": "neighbor", "surface": "Neighbor",
                "type": "GenericEntity", "canonical_id": "neighbor",
            },
            {
                "occurrence_key": "relation", "surface": "Relation",
                "type": "GenericEntity", "canonical_id": "relation",
            },
        ],
        logical_schema=_schema(),
        minimum_score=0.0,
    )
    assert result["bindings"] == []


def test_source_binding_gleaning_uses_compact_indexes_and_excludes_found_binding():
    inventory = {
        "coverage_complete": True,
        "frames": [
            {
                "relation": "is available",
                "arguments": ["Status"],
                "roles": ["state"],
                "argument_kinds": ["STATE"],
                "entity_heads": [""],
                "assertion_status": "ASSERTED",
            },
            {
                "relation": "contains",
                "arguments": ["Report", "Dataset"],
                "roles": ["whole", "member"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Report", "Dataset"],
                "assertion_status": "ASSERTED",
            },
        ],
    }
    payload = __import__("json").loads(source_frame_schema_binding_prompt(
        inventory,
        logical_schema=_schema(),
        already_found_bindings=[{
            # Runtime/parser coordinates point at the original inventory.
            "frame_index": 1,
            "predicate": "alpha_rel",
            "subject_index": 0,
            "object_index": 1,
        }],
    ))
    assert payload["frames"][0]["frame_index"] == 0
    assert payload["already_found_bindings"] == [{
        "frame_index": 0,
        "predicate": "alpha_rel",
        "subject_index": 0,
        "object_index": 1,
    }]
    assert "MUST NOT be repeated" in payload["instruction"]
    assert "both argument directions" in payload["instruction"]


def test_source_frame_batch_verifier_requires_exact_decision_ids():
    content = "Alpha connects Beta."
    mentions = [
        _mention("alpha", 0, 5, "Alpha"),
        _mention("beta", 15, 19, "Beta"),
    ]
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-batch",
        chunk_key="chunk-batch",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        run_id="run-batch",
        candidate_policy_version="candidate-v1",
        max_char_distance=100,
    )
    opportunity = plan.opportunities[0]
    binding = parse_source_frame_schema_bindings(
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "alpha_rel",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        inventory={
            "coverage_complete": True,
            "frames": [{
                "relation": "connects",
                "arguments": ["Alpha", "Beta"],
                "roles": ["source", "target"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Alpha", "Beta"],
                "assertion_status": "ASSERTED",
            }],
        },
        logical_schema=_schema(),
    )
    hypothesis = source_frame_projection_hypotheses(
        {
            "coverage_complete": True,
            "frames": [{
                "relation": "connects",
                "arguments": ["Alpha", "Beta"],
                "roles": ["source", "target"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Alpha", "Beta"],
                "assertion_status": "ASSERTED",
            }],
        },
        binding,
        occurrences=[
            {"occurrence_key": "occ-alpha", "surface": "Alpha"},
            {"occurrence_key": "occ-beta", "surface": "Beta"},
        ],
        opportunities_by_pair={opportunity.pair_identity: opportunity},
        logical_schema=_schema(),
        classifier_model_revision="model-v1",
    )[0]
    prompt = __import__("json").loads(batch_verifier_prompt(
        [(hypothesis, opportunity)],
        content=content,
        logical_schema=_schema(),
    ))
    assert len(prompt["claims"]) == 1
    assert "instruction" not in prompt["claims"][0]
    assert prompt["claims"][0]["context_ref"] == "CTX-1"
    assert prompt["claims"][0]["relation_policy_ref"] == "RP-1"
    assert len(prompt["context_catalog"]) == 1
    assert len(prompt["relation_policy_catalog"]) == 1
    assert prompt["output_contract"]["root_key"] == "decisions"
    assert "verifications" in prompt["output_contract"]["forbidden_root_keys"]
    schema = batch_verifier_response_schema([
        hypothesis.hypothesis_id
    ])
    assert schema["properties"]["decisions"]["minItems"] == 1
    assert (
        schema["properties"]["decisions"]["items"]["properties"]
        ["reason_code"]["maxLength"]
    ) == 160
    decision = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "verdict": "SUPPORTED",
        "reason_code": "DIRECT",
        "forward_entailment": "ENTAILED",
        "reverse_entailment": "NOT_ENTAILED",
        "subject_role_supported": True,
        "object_role_supported": True,
        "supported_evidence_spans": [{"start": 0, "end": len(content)}],
    }
    parsed = parse_batch_verifier_response(
        {"decisions": [decision]},
        hypothesis_ids=[hypothesis.hypothesis_id],
    )
    assert parsed[hypothesis.hypothesis_id]["verdict"] == "SUPPORTED"
    keyed = parse_batch_verifier_response(
        {"decisions": {
            hypothesis.hypothesis_id: {
                key: value for key, value in decision.items()
                if key != "hypothesis_id"
            },
        }},
        hypothesis_ids=[hypothesis.hypothesis_id],
    )
    assert keyed == parsed
    with pytest.raises(ShadowClassificationError):
        parse_batch_verifier_response(
            {"decisions": {
                hypothesis.hypothesis_id: {
                    **decision,
                    "hypothesis_id": "different-hypothesis",
                },
            }},
            hypothesis_ids=[hypothesis.hypothesis_id],
        )
    assert batch_verifier_decision_is_source_grounded(
        hypothesis,
        opportunity,
        parsed[hypothesis.hypothesis_id],
        content=content,
        logical_schema=_schema(),
    )
    assert not batch_verifier_decision_is_source_grounded(
        hypothesis,
        opportunity,
        {
            **parsed[hypothesis.hypothesis_id],
            "supported_evidence_spans": [{"start": 0, "end": 5}],
        },
        content=content,
        logical_schema=_schema(),
    )
    assert batch_verifier_decision_is_source_grounded(
        hypothesis,
        opportunity,
        {
            **parsed[hypothesis.hypothesis_id],
            "verdict": "REJECTED",
            "supported_evidence_spans": [],
        },
        content=content,
        logical_schema=_schema(),
    )
    assert not batch_verifier_decision_is_source_grounded(
        hypothesis,
        opportunity,
        {
            **parsed[hypothesis.hypothesis_id],
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "ENTAILED",
        },
        content=content,
        logical_schema=_schema(),
    )
    with pytest.raises(ShadowClassificationError):
        parse_batch_verifier_response(
            {"decisions": [decision]},
            hypothesis_ids=[hypothesis.hypothesis_id, "missing-hypothesis"],
        )
    partial = parse_batch_verifier_response(
        {"decisions": [decision]},
        hypothesis_ids=[hypothesis.hypothesis_id, "missing-hypothesis"],
        allow_partial=True,
    )
    assert set(partial) == {hypothesis.hypothesis_id}

    legacy_negative = parse_batch_verifier_response(
        {"verifications": [{
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "REJECTED",
            "entailment": "NOT_ENTAILED",
            "evidence_spans": [],
        }]},
        hypothesis_ids=[hypothesis.hypothesis_id],
        allow_partial=True,
    )
    assert legacy_negative[hypothesis.hypothesis_id] == {
        "verdict": "REJECTED",
        "entailment": "NOT_ENTAILED",
        "evidence_spans": [],
        "supported_evidence_spans": [],
        "forward_entailment": "NOT_ENTAILED",
        "reason_code": "MODEL_BATCH_REJECTED_LEGACY_FIELDS",
        "reverse_entailment": "UNCERTAIN",
        "subject_role_supported": False,
        "object_role_supported": False,
    }
    assert parse_batch_verifier_response(
        {"verdicts": [{
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "REJECTED",
            "entailment": "NOT_ENTAILED",
            "evidence_spans": [],
        }]},
        hypothesis_ids=[hypothesis.hypothesis_id],
        allow_partial=True,
    )[hypothesis.hypothesis_id]["verdict"] == "REJECTED"
    legacy_supported = parse_batch_verifier_response(
        {"verifications": [{
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "SUPPORTED",
            "entailment": "ENTAILED",
            "evidence_spans": [{"start": 0, "end": len(content)}],
        }]},
        hypothesis_ids=[hypothesis.hypothesis_id],
        allow_partial=True,
    )
    assert legacy_supported == {}


def test_source_projection_expands_explicit_mentions_inside_coordinated_argument():
    content = "A compound statement contains one or more clauses."
    compound_start = content.index("compound statement")
    clauses_start = content.index("clauses")
    mentions = [
        _mention(
            "compound", compound_start,
            compound_start + len("compound statement"), "compound statement",
        ),
        _mention(
            "clauses", clauses_start,
            clauses_start + len("clauses"), "clauses",
        ),
    ]
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-2",
        chunk_key="chunk-2",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        run_id="run-2",
        candidate_policy_version="candidate-v1",
        max_char_distance=100,
    )
    inventory = {
        "coverage_complete": False,
        "frames": [{
            "relation": "contains",
            "arguments": ["A compound statement", "one or more clauses"],
            "roles": ["subject", "object"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["statement", "clauses"],
            "assertion_status": "ASSERTED",
        }],
    }
    bindings = {"bindings": [{
        "frame_index": 0,
        "predicate": "alpha_rel",
        "subject_index": 0,
        "object_index": 1,
    }]}
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        bindings,
        occurrences=[
            {"occurrence_key": "occ-compound", "surface": "compound statement"},
            {"occurrence_key": "occ-clauses", "surface": "clauses"},
        ],
        opportunities_by_pair={
            plan.opportunities[0].pair_identity: plan.opportunities[0],
        },
        logical_schema=_schema(),
        classifier_model_revision="model-v1",
    )
    assert len(hypotheses) == 1


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


def test_hierarchical_list_parent_child_bypasses_distance_but_siblings_do_not():
    content = "System X\n— Module A\n— " + ("padding " * 20) + "Module B"
    parent_start = content.index("System X")
    first_start = content.index("Module A")
    second_start = content.index("Module B")
    group = "list-group-1"
    parent_scope = "scope-parent"
    first_scope = "scope-first"
    second_scope = "scope-second"
    mentions = [
        _mention(
            "parent", parent_start, parent_start + len("System X"), "System X",
            source_role="list_item", relation_eligible=True,
            scope_id=parent_scope, scope_kind="list_item",
            scope_start=0, scope_end=8, list_group_id=group, list_depth=0,
            list_parent_scope_id="",
        ),
        _mention(
            "first", first_start, first_start + len("Module A"), "Module A",
            source_role="list_item", relation_eligible=True,
            scope_id=first_scope, scope_kind="list_item",
            scope_start=9, scope_end=18, list_group_id=group, list_depth=1,
            list_parent_scope_id=parent_scope,
        ),
        _mention(
            "second", second_start, second_start + len("Module B"), "Module B",
            source_role="list_item", relation_eligible=True,
            scope_id=second_scope, scope_kind="list_item",
            scope_start=19, scope_end=len(content), list_group_id=group,
            list_depth=1, list_parent_scope_id=parent_scope,
        ),
    ]
    schema = logical_schema(predicate_policy(
        "alpha_rel", domain=["*"], range_=["*"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    plan = plan_pair_opportunities(
        project_id=1, document_revision_id="revision-list", chunk_key="chunk-list",
        content=content, mentions=mentions, logical_schema=schema,
        run_id="run-list", candidate_policy_version="candidate-list",
        max_char_distance=10,
    )
    by_pair = {
        frozenset((item.left_occurrence_key, item.right_occurrence_key)): item
        for item in plan.opportunities
    }

    parent_child = by_pair[frozenset(("occ-parent", "occ-first"))]
    sibling = by_pair[frozenset(("occ-first", "occ-second"))]
    assert parent_child.status == PENDING_CLASSIFICATION
    assert parent_child.planning_reason == "hierarchical_list_parent_child"
    assert sibling.status == FILTERED_DISTANCE


def test_structural_projection_is_independent_of_classifier_decision():
    content = "System X\n— Module A"
    parent_start = content.index("System X")
    child_start = content.index("Module A")
    carrier_policy = predicate_policy(
        "source_assertion",
        semantic_definition=(
            "The exact open source expression relates subject to object."
        ),
        open_relation_policy={
            "enabled": True,
            "fallback_only": True,
            "allowed_candidate_sources": [
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
                "ATOMIC_FRAME_SCHEMA_PROJECTION",
            ],
            "display_source_expression": True,
            "identity_source_expression": True,
            "require_exact_source_expression": True,
        },
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    )
    schema = logical_schema(
        predicate_policy(
            "part_of", domain=["*"], range_=["*"],
            structural_evidence_kinds=["list_parent_child"],
            scope_policy={
                "require_same_atomic_scope": False,
                "allow_cross_scope_with_explicit_evidence": True,
            },
        ),
        carrier_policy,
    )
    plan = plan_pair_opportunities(
        project_id=1, document_revision_id="revision-list", chunk_key="chunk-list",
        content=content,
        mentions=[
            _mention(
                "parent", parent_start, parent_start + len("System X"), "System X",
                source_role="list_item", relation_eligible=True,
                scope_id="scope-parent", scope_kind="list_item",
                scope_start=0, scope_end=8, list_group_id="group", list_depth=0,
                list_parent_scope_id="",
            ),
            _mention(
                "child", child_start, child_start + len("Module A"), "Module A",
                source_role="list_item", relation_eligible=True,
                scope_id="scope-child", scope_kind="list_item",
                scope_start=9, scope_end=len(content), list_group_id="group",
                list_depth=1, list_parent_scope_id="scope-parent",
            ),
        ],
        logical_schema=schema,
        run_id="run-list", candidate_policy_version="candidate-list",
    )
    opportunity = plan.opportunities[0]
    projected = structural_projection_hypotheses(
        opportunity,
        schema=normalize_predicate_schema(schema),
        classifier_policy_version="classifier-v1",
        classifier_model_revision="model-v1",
    )

    assert len(projected) == 1
    assert projected[0].predicate == "part_of"
    assert projected[0].subject_occurrence_key == "occ-child"
    assert projected[0].object_occurrence_key == "occ-parent"
    assert projected[0].predicate_metadata["structural_event_projection"] is True

    accepted = validate_hypothesis(
        projected[0], opportunity, content=content, logical_schema=schema,
    )
    reversed_hypothesis = replace(
        projected[0],
        hypothesis_id="reversed-structural-hypothesis",
        subject_occurrence_key=projected[0].object_occurrence_key,
        object_occurrence_key=projected[0].subject_occurrence_key,
    )
    rejected = validate_hypothesis(
        reversed_hypothesis, opportunity,
        content=content, logical_schema=schema,
    )

    assert accepted.status == "PASS"
    assert rejected.status == "REJECTED"
    assert rejected.reason_code == "STRUCTURAL_ROLE_ORIENTATION_MISMATCH"
    assert parser_structure_claim_intent(
        projected[0], opportunity, logical_schema=schema,
    ) is True
    terminal = deterministic_parser_structure_terminal_attempt(
        projected[0],
        accepted,
        opportunity,
        content=content,
        logical_schema=schema,
    )
    assert terminal.verdict == "SUPPORTED"
    assert terminal.reason_code == "SUPPORTED_PARSER_STRUCTURE_CONTRACT"
    assert terminal.independent_call is True
    assert supported_for_normalization(terminal) is True

    carrier_schema = schema
    carrier_spec = normalize_predicate_schema(carrier_schema)[
        "source_assertion"
    ]
    child_canonical = (
        opportunity.left_canonical_id
        if projected[0].subject_occurrence_key
        == opportunity.left_occurrence_key
        else opportunity.right_canonical_id
    )
    parent_canonical = (
        opportunity.left_canonical_id
        if projected[0].object_occurrence_key
        == opportunity.left_occurrence_key
        else opportunity.right_canonical_id
    )
    carrier_hypothesis = replace(
        projected[0],
        hypothesis_id="open-carrier-structural-hypothesis",
        predicate="source_assertion",
        predicate_metadata={
            **asdict(carrier_spec),
            "candidate_source": (
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION"
            ),
            "source_frame_index": 0,
            "source_frame_relation": "structural child of",
            "source_frame_assertion_status": "ASSERTED",
            "source_frame_explicit_negation": False,
            "source_frame_evidence_span": [0, len(content)],
            "source_frame_evidence_text": content,
            "source_structure_kind": "list_parent_child",
            "source_syntax_origin": "",
            "source_subject_argument_index": 0,
            "source_object_argument_index": 1,
            "source_subject_surface": "Module A",
            "source_object_surface": "System X",
            "source_subject_canonical_id": child_canonical,
            "source_object_canonical_id": parent_canonical,
            "source_frame_other_participants": [],
            "source_binding_source": (
                "OPEN_EXPRESSION_CARRIER_FALLBACK"
            ),
            "source_frame_roles": [
                "structural_child", "structural_parent",
            ],
            "source_resolved_reference_endpoints": [],
        },
    )
    carrier_validation = validate_hypothesis(
        carrier_hypothesis,
        opportunity,
        content=content,
        logical_schema=carrier_schema,
    )
    assert carrier_validation.status == "PASS"
    assert parser_structure_claim_intent(
        carrier_hypothesis,
        opportunity,
        logical_schema=carrier_schema,
    ) is True
    carrier_terminal = deterministic_parser_structure_terminal_attempt(
        carrier_hypothesis,
        carrier_validation,
        opportunity,
        content=content,
        logical_schema=carrier_schema,
    )
    assert carrier_terminal.verdict == "SUPPORTED"
    assert carrier_terminal.model_request_payload["model_call"] is False

    reversed_terminal = deterministic_parser_structure_terminal_attempt(
        reversed_hypothesis,
        rejected,
        opportunity,
        content=content,
        logical_schema=schema,
    )
    assert reversed_terminal.verdict == "REJECTED"
    assert supported_for_normalization(reversed_terminal) is False


def test_standalone_section_label_parent_child_bypasses_distance_generically():
    content = "Requirements\n\nSR-01 The service shall check intent before approval."
    child_start = content.index("SR-01")
    schema = logical_schema(predicate_policy(
        "alpha_rel", domain=["*"], range_=["*"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-label",
        chunk_key="chunk-label",
        content=content,
        mentions=[
            _mention(
                "label", 0, len("Requirements"), "Requirements",
                source_role="body", relation_eligible=True,
                scope_id="scope-label", scope_kind="sentence",
                scope_start=0, scope_end=len("Requirements"),
            ),
            _mention(
                "child", child_start, child_start + len("SR-01"), "SR-01",
                source_role="body", relation_eligible=True,
                scope_id="scope-child", scope_kind="sentence",
                scope_start=child_start, scope_end=len(content),
            ),
        ],
        logical_schema=schema,
        run_id="run-label",
        candidate_policy_version="candidate-label",
        max_char_distance=5,
    )

    opportunity = plan.opportunities[0]
    assert opportunity.status == PENDING_CLASSIFICATION
    assert opportunity.planning_reason == "hierarchical_section_label_parent_child"
    assert opportunity.provenance["hierarchical_section_label_parent_child"] is True


def test_adjacent_flattened_figure_labels_have_no_section_parent_contract():
    content = "Page8\nCoreModule\nTaskR1\nTaskR2"
    surfaces = ["Page8", "CoreModule", "TaskR1", "TaskR2"]
    mentions = []
    for index, surface in enumerate(surfaces):
        start = content.index(surface)
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", start)
        if line_end < 0:
            line_end = len(content)
        mentions.append(_mention(
            str(index), start, start + len(surface), surface,
            source_role="body", relation_eligible=True,
            scope_id=f"scope-{index}", scope_kind="sentence",
            scope_start=line_start, scope_end=line_end,
        ))
    schema = logical_schema(predicate_policy(
        "alpha_rel", domain=["*"], range_=["*"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))

    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-flat-figure",
        chunk_key="chunk-flat-figure",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-flat-figure",
        candidate_policy_version="candidate-flat-figure",
        max_char_distance=1,
    )

    assert not any(
        item.planning_reason == "hierarchical_section_label_parent_child"
        or item.provenance["hierarchical_section_label_parent_child"] is True
        for item in plan.opportunities
    )


def test_section_label_does_not_govern_later_contiguous_prose_lines():
    content = (
        "Article 12\n"
        "The applicant submits a report.\n"
        "Safety report supports the decision."
    )
    child_start = content.index("Safety report")
    schema = logical_schema(predicate_policy(
        "part_of", domain=["*"], range_=["*"],
        structural_evidence_kinds=["section_label_parent_child"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-non-direct-heading-child",
        chunk_key="chunk-non-direct-heading-child",
        content=content,
        mentions=[
            _mention(
                "label", 0, len("Article 12"), "Article 12",
                source_role="body", relation_eligible=True,
                scope_id="scope-label", scope_kind="sentence",
                scope_start=0, scope_end=len("Article 12"),
            ),
            _mention(
                "later", child_start, child_start + len("Safety report"),
                "Safety report", source_role="body", relation_eligible=True,
                scope_id="scope-later", scope_kind="sentence",
                scope_start=child_start, scope_end=len(content),
            ),
        ],
        logical_schema=schema,
        run_id="run-non-direct-heading-child",
        candidate_policy_version="candidate-non-direct-heading-child",
        max_char_distance=5,
    )

    opportunity = plan.opportunities[0]
    assert opportunity.planning_reason != (
        "hierarchical_section_label_parent_child"
    )
    assert opportunity.provenance[
        "hierarchical_section_label_parent_child"
    ] is False


def test_stable_identifier_does_not_skip_intervening_prose_or_heading():
    content = (
        "Engineering Process Inventory\n"
        "The lifecycle includes design and implementation.\n"
        "Requirements\n"
        "SR-01 The service shall validate an intent."
    )
    parent = "Engineering Process Inventory"
    child = "SR-01"
    child_start = content.index(child)
    schema = logical_schema(predicate_policy(
        "part_of", domain=["*"], range_=["*"],
        structural_evidence_kinds=["section_label_parent_child"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-intervening-heading",
        chunk_key="chunk-intervening-heading",
        content=content,
        mentions=[
            _mention(
                "parent", 0, len(parent), parent,
                source_role="body", relation_eligible=True,
                scope_id="scope-parent", scope_kind="sentence",
                scope_start=0, scope_end=len(parent),
            ),
            _mention(
                "child", child_start, child_start + len(child), child,
                source_role="body", relation_eligible=True,
                scope_id="scope-child", scope_kind="sentence",
                scope_start=child_start, scope_end=len(content),
                inline_label_candidate=True,
            ),
        ],
        logical_schema=schema,
        run_id="run-intervening-heading",
        candidate_policy_version="candidate-intervening-heading",
        max_char_distance=5,
    )

    opportunity = plan.opportunities[0]
    assert opportunity.planning_reason != (
        "hierarchical_section_label_parent_child"
    )
    assert opportunity.provenance[
        "hierarchical_section_label_parent_child"
    ] is False


def test_long_colon_terminated_prose_lead_in_is_not_a_section_label():
    content = (
        "The following example places a suite inside an `if` statement:\n\n"
        "if ready:\n"
        "    start()"
    )
    parent = "The following example places a suite inside an `if` statement:"
    parent_start = content.index(parent)
    child_start = content.index("start()")
    schema = logical_schema(predicate_policy(
        "alpha_rel", domain=["*"], range_=["*"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-prose-lead-in",
        chunk_key="chunk-prose-lead-in",
        content=content,
        mentions=[
            _mention(
                "lead-in", parent_start, parent_start + len(parent), parent,
                source_role="body", relation_eligible=True,
                scope_id="scope-lead-in", scope_kind="sentence",
                scope_start=parent_start, scope_end=parent_start + len(parent),
            ),
            _mention(
                "code", child_start, child_start + len("start()"), "start()",
                source_role="body", relation_eligible=True,
                scope_id="scope-code", scope_kind="sentence",
                scope_start=child_start, scope_end=len(content),
            ),
        ],
        logical_schema=schema,
        run_id="run-prose-lead-in",
        candidate_policy_version="candidate-prose-lead-in",
        max_char_distance=5,
    )

    opportunity = plan.opportunities[0]
    assert opportunity.planning_reason != "hierarchical_section_label_parent_child"


def test_pdf_symbol_hierarchy_generates_every_direct_child_without_sibling_edges():
    content = (
        "\uf0a8 Core Platform\n"
        "— Partition Management\n"
        "— Task Management\n"
        "— Health Monitoring\n"
        "— Multicore Support\n"
        "\uf0d8 Spin Lock\n"
        "\uf0d8 Atomic Operation"
    )
    spans = structural_spans_for_chunk(content=content, blocks=[])
    surfaces = [
        "Core Platform", "Partition Management", "Task Management",
        "Health Monitoring", "Multicore Support", "Spin Lock",
        "Atomic Operation",
    ]
    mentions = []
    for index, surface in enumerate(surfaces):
        start = content.index(surface)
        profile = occurrence_scope(
            content=content,
            mention_start=start,
            mention_end=start + len(surface),
            structure={"structural_spans": spans},
            context_kind="structured_chunk",
        )
        scope = profile["occurrence_scope"]
        mentions.append(_mention(
            str(index), start, start + len(surface), surface,
            source_role=profile["source_role"],
            relation_eligible=profile["relation_eligible"],
            scope_id=scope["scope_id"],
            scope_kind=scope["scope_kind"],
            scope_start=scope["start"],
            scope_end=scope["end"],
            list_group_id=scope.get("list_group_id", ""),
            list_depth=scope.get("list_depth", 0),
            list_parent_scope_id=scope.get("list_parent_scope_id", ""),
        ))

    inventory = augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=mentions,
    )
    edges = {
        tuple(frame["arguments"])
        for frame in inventory["frames"]
        if frame.get("deterministic_source_structure")
        == "list_parent_child"
    }

    assert edges == {
        ("Partition Management", "Core Platform"),
        ("Task Management", "Core Platform"),
        ("Health Monitoring", "Core Platform"),
        ("Multicore Support", "Core Platform"),
        ("Spin Lock", "Multicore Support"),
        ("Atomic Operation", "Multicore Support"),
    }


def test_indented_same_marker_hierarchy_plans_every_child_and_no_sibling_edge():
    content = (
        "— Parent A\n"
        "  — Child A1\n"
        "  — Child A2\n"
        "— Parent B\n"
        "  — Child B1\n"
        "  — Child B2\n"
        "  — Child B3"
    )
    spans = structural_spans_for_chunk(content=content, blocks=[])
    surfaces = [
        "Parent A", "Child A1", "Child A2",
        "Parent B", "Child B1", "Child B2", "Child B3",
    ]
    mentions = []
    surface_by_occurrence = {}
    for index, surface in enumerate(surfaces):
        start = content.index(surface)
        profile = occurrence_scope(
            content=content,
            mention_start=start,
            mention_end=start + len(surface),
            structure={"structural_spans": spans},
            context_kind="structured_chunk",
        )
        scope = profile["occurrence_scope"]
        mention = _mention(
            str(index), start, start + len(surface), surface,
            source_role=profile["source_role"],
            relation_eligible=profile["relation_eligible"],
            scope_id=scope["scope_id"],
            scope_kind=scope["scope_kind"],
            scope_start=scope["start"],
            scope_end=scope["end"],
            list_group_id=scope.get("list_group_id", ""),
            list_depth=scope.get("list_depth", 0),
            list_parent_scope_id=scope.get("list_parent_scope_id", ""),
        )
        mentions.append(mention)
        surface_by_occurrence[mention["occurrence_key"]] = surface

    schema = logical_schema(predicate_policy(
        "part_of", domain=["*"], range_=["*"],
        structural_evidence_kinds=["list_parent_child"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    plan = plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-indented-list",
        chunk_key="chunk-indented-list",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-indented-list",
        candidate_policy_version="candidate-indented-list",
        max_char_distance=1,
    )
    hierarchy = {
        (
            surface_by_occurrence[
                item.provenance["structural_child_occurrence_key"]
            ],
            surface_by_occurrence[
                item.provenance["structural_parent_occurrence_key"]
            ],
        )
        for item in plan.opportunities
        if item.planning_reason == "hierarchical_list_parent_child"
    }

    assert hierarchy == {
        ("Child A1", "Parent A"),
        ("Child A2", "Parent A"),
        ("Child B1", "Parent B"),
        ("Child B2", "Parent B"),
        ("Child B3", "Parent B"),
    }
    assert not any(
        item.planning_reason == "hierarchical_list_parent_child"
        and {
            surface_by_occurrence[item.left_occurrence_key],
            surface_by_occurrence[item.right_occurrence_key],
        } == {"Child B1", "Child B2"}
        for item in plan.opportunities
    )
