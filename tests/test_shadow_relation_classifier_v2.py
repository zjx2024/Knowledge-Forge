from __future__ import annotations

from dataclasses import replace
import json

import pytest

from knowledge_forge.relation_shadow_v2 import (
    CLASSIFICATION_FAILED,
    EvidenceSpan,
    HYPOTHESES,
    NO_RELATION,
    OTHER_UNSUPPORTED,
    PairClassificationDecision,
    ShadowClassificationError,
    classifier_prompt,
    classifier_response_schema,
    normalize_predicate_schema,
    parse_classifier_response,
    plan_pair_opportunities,
    validate_hypothesis,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


CONTENT = "Acme uses Alloy."


def _mentions():
    return [
        {
            "mention_id": "mention-acme",
            "occurrence_key": "occ-acme",
            "surface": "Acme",
            "mention_start": 0,
            "mention_end": 4,
            "canonical_id": "entity-acme",
            "assignment_version": "assign-v1",
            "type": "Organization",
        },
        {
            "mention_id": "mention-alloy",
            "occurrence_key": "occ-alloy",
            "surface": "Alloy",
            "mention_start": 10,
            "mention_end": 15,
            "canonical_id": "entity-alloy",
            "assignment_version": "assign-v1",
            "type": "Product",
        },
    ]


def _schema():
    return logical_schema(
        predicate_policy(
            "uses",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition="The organization applies the product.",
            inverse_predicate="used_by",
            positive_examples=["Acme applies Alloy."],
            negative_examples=["Alloy applies Acme."],
        ),
        predicate_policy(
            "used_by",
            domain=["Product"],
            range_=["Organization"],
            semantic_definition="The product is applied by the organization.",
            inverse_predicate="uses",
            canonical_orientation="inverse",
        ),
        predicate_policy(
            "supplies",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition="The organization supplies the product.",
        ),
        predicate_policy(
            "related_to",
            publish_mode="review_only",
            semantic_definition="Generic semantic relation.",
        ),
        predicate_policy(
            "partners_with",
            domain=["Organization"],
            range_=["Organization"],
            symmetric=True,
            semantic_definition="Mutual partnership.",
        ),
        version="test-schema-v7",
    )


def _opportunity(schema=None, content=CONTENT, mentions=None):
    return plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions or _mentions(),
        logical_schema=schema or _schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v2",
    ).opportunities[0]


def _decision(pair_id, *, decision=HYPOTHESES, hypotheses=None, **extra):
    return {
        "pair_id": pair_id,
        "decision": decision,
        "hypotheses": hypotheses or [],
        **extra,
    }


def _hypothesis(subject="occ-acme", predicate="uses", object_="occ-alloy", **extra):
    return {
        "subject_occurrence_id": subject,
        "predicate": predicate,
        "object_occurrence_id": object_,
        "evidence_spans": [{"start": 0, "end": len(CONTENT)}],
        "classifier_confidence": 0.82,
        **extra,
    }


@pytest.mark.parametrize("reason", ["SAME_LIST_ONLY", "CONTACT_BLOCK_COOCCURRENCE"])
def test_structural_cooccurrence_can_be_explicit_no_relation(reason) -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(
            pair.pair_identity,
            decision=NO_RELATION,
            reason_code=reason,
            evidence_spans=[{"start": 0, "end": len(CONTENT)}],
        )]},
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert parsed[0].decision == NO_RELATION
    assert parsed[0].hypotheses == ()


def test_no_relation_defaults_missing_reason_and_evidence() -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(
            pair.pair_identity,
            decision=NO_RELATION,
        )]},
        content=CONTENT,
        logical_schema=_schema(),
    )
    assert parsed[0].decision == NO_RELATION
    assert parsed[0].hypotheses == ()
    assert parsed[0].reason_code == "NO_RELATION_UNSPECIFIED"
    assert [(span.start, span.end) for span in parsed[0].evidence_spans] == [
        pair.context_span,
    ]


def test_no_relation_lexical_recovery_requires_pair_local_cue() -> None:
    content = "Platform supports Alpha and Beta."
    schema = logical_schema(predicate_policy(
        "supports",
        optional_soft_cues=["supports"],
        semantic_definition="The subject supports the object.",
    ))
    mentions = []
    for surface in ("Platform", "Alpha", "Beta"):
        start = content.index(surface)
        mentions.append({
            "mention_id": surface.lower(),
            "occurrence_key": surface.lower(),
            "surface": surface,
            "mention_start": start,
            "mention_end": start + len(surface),
            "canonical_id": "entity-" + surface.lower(),
            "assignment_version": "assign-v1",
            "type": "GenericEntity",
        })
    plan = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-pair-local-recovery",
        chunk_key="chunk-pair-local-recovery",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-pair-local-recovery",
        candidate_policy_version="candidate-pair-local-recovery-v1",
    )
    by_pair = {
        frozenset((item.left_occurrence_key, item.right_occurrence_key)): item
        for item in plan.opportunities
    }

    platform_alpha = by_pair[frozenset(("platform", "alpha"))]
    alpha_beta = by_pair[frozenset(("alpha", "beta"))]
    assert platform_alpha.provenance["pair_local_soft_cue_matches"]
    assert not alpha_beta.provenance["pair_local_soft_cue_matches"]

    recovered = parse_classifier_response(
        [platform_alpha],
        {"decisions": [_decision(
            platform_alpha.pair_identity,
            decision=NO_RELATION,
        )]},
        content=content,
        logical_schema=schema,
    )[0]
    sibling = parse_classifier_response(
        [alpha_beta],
        {"decisions": [_decision(
            alpha_beta.pair_identity,
            decision=NO_RELATION,
        )]},
        content=content,
        logical_schema=schema,
    )[0]

    assert recovered.decision == HYPOTHESES
    assert len(recovered.hypotheses) == 2
    assert all(
        item.predicate_metadata["candidate_source"]
        == "LEXICAL_CUE_RECOVERY"
        for item in recovered.hypotheses
    )
    assert sibling.decision == NO_RELATION
    assert sibling.hypotheses == ()


def test_no_relation_lexical_recovery_does_not_cross_later_comma_clause() -> None:
    content = (
        "标注准确性差异导致模型语义理解出现偏差，"
        "学者从噪声处理方面提出方法。"
    )
    schema = logical_schema(predicate_policy(
        "causes",
        optional_soft_cues=["导致"],
        semantic_definition="主体直接导致客体。",
    ))

    def mention(key: str, surface: str) -> dict:
        start = content.index(surface)
        return {
            "mention_id": key,
            "occurrence_key": key,
            "surface": surface,
            "mention_start": start,
            "mention_end": start + len(surface),
            "canonical_id": "entity-" + key,
            "assignment_version": "assign-v1",
            "type": "GenericEntity",
        }

    pair = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-clause-local-recovery",
        chunk_key="chunk-clause-local-recovery",
        content=content,
        mentions=[
            mention("label-difference", "标注准确性差异"),
            mention("noise-handling", "噪声处理"),
        ],
        logical_schema=schema,
        run_id="run-clause-local-recovery",
        candidate_policy_version="candidate-clause-local-recovery-v1",
    ).opportunities[0]
    assert pair.provenance["pair_local_soft_cue_matches"]

    decision = parse_classifier_response(
        [pair],
        {"decisions": [_decision(pair.pair_identity, decision=NO_RELATION)]},
        content=content,
        logical_schema=schema,
    )[0]

    assert decision.decision == NO_RELATION
    assert decision.hypotheses == ()


@pytest.mark.parametrize(
    "content",
    (
        "标注差异导致语义偏差。",
        "在复杂场景中，标注差异导致语义偏差。",
    ),
)
def test_no_relation_lexical_recovery_keeps_direct_clause_cue(
    content: str,
) -> None:
    schema = logical_schema(predicate_policy(
        "causes",
        optional_soft_cues=["导致"],
        semantic_definition="主体直接导致客体。",
    ))
    mentions = []
    for key, surface in (("difference", "标注差异"), ("deviation", "语义偏差")):
        start = content.index(surface)
        mentions.append({
            "mention_id": key,
            "occurrence_key": key,
            "surface": surface,
            "mention_start": start,
            "mention_end": start + len(surface),
            "canonical_id": "entity-" + key,
            "assignment_version": "assign-v1",
            "type": "GenericEntity",
        })
    pair = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-direct-recovery",
        chunk_key="chunk-direct-recovery",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-direct-recovery",
        candidate_policy_version="candidate-direct-recovery-v1",
    ).opportunities[0]

    decision = parse_classifier_response(
        [pair],
        {"decisions": [_decision(pair.pair_identity, decision=NO_RELATION)]},
        content=content,
        logical_schema=schema,
    )[0]

    assert decision.decision == HYPOTHESES
    assert {item.predicate for item in decision.hypotheses} == {"causes"}


def test_legacy_opportunity_recomputes_lexical_recovery_between_endpoints_only() -> None:
    content = "Platform supports Alpha and Beta."
    schema = logical_schema(predicate_policy(
        "supports",
        optional_soft_cues=["supports"],
        semantic_definition="The subject supports the object.",
    ))
    mentions = []
    for surface in ("Platform", "Alpha", "Beta"):
        start = content.index(surface)
        mentions.append({
            "mention_id": surface.lower(),
            "occurrence_key": surface.lower(),
            "surface": surface,
            "mention_start": start,
            "mention_end": start + len(surface),
            "canonical_id": "entity-" + surface.lower(),
            "assignment_version": "assign-v1",
            "type": "GenericEntity",
        })
    plan = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-legacy-local-recovery",
        chunk_key="chunk-legacy-local-recovery",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-legacy-local-recovery",
        candidate_policy_version="candidate-legacy-local-recovery-v1",
    )
    by_pair = {
        frozenset((item.left_occurrence_key, item.right_occurrence_key)): item
        for item in plan.opportunities
    }
    platform_alpha = by_pair[frozenset(("platform", "alpha"))]
    alpha_beta = by_pair[frozenset(("alpha", "beta"))]
    # Simulate a persisted opportunity from before pair-local cue provenance
    # was introduced. Immutable offsets still permit a safe local recompute.
    platform_alpha = replace(
        platform_alpha,
        provenance={
            **platform_alpha.provenance,
            "pair_local_soft_cue_matches": [],
        },
    )
    alpha_beta = replace(
        alpha_beta,
        provenance={
            **alpha_beta.provenance,
            "pair_local_soft_cue_matches": [],
        },
    )

    recovered = parse_classifier_response(
        [platform_alpha],
        {"decisions": [_decision(
            platform_alpha.pair_identity,
            decision=NO_RELATION,
        )]},
        content=content,
        logical_schema=schema,
    )[0]
    sibling = parse_classifier_response(
        [alpha_beta],
        {"decisions": [_decision(
            alpha_beta.pair_identity,
            decision=NO_RELATION,
        )]},
        content=content,
        logical_schema=schema,
    )[0]

    assert recovered.decision == HYPOTHESES
    assert sibling.decision == NO_RELATION


def test_no_relation_lexical_recovery_never_crosses_atomic_scopes() -> None:
    content = "Platform\nsupports\nAlpha"
    schema = logical_schema(predicate_policy(
        "supports",
        optional_soft_cues=["supports"],
        semantic_definition="The subject supports the object.",
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    mentions = []
    for surface in ("Platform", "Alpha"):
        start = content.index(surface)
        mentions.append({
            "mention_id": surface.lower(),
            "occurrence_key": surface.lower(),
            "surface": surface,
            "mention_start": start,
            "mention_end": start + len(surface),
            "canonical_id": "entity-" + surface.lower(),
            "assignment_version": "assign-v1",
            "type": "GenericEntity",
        })
    pair = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-cross-scope-recovery",
        chunk_key="chunk-cross-scope-recovery",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-cross-scope-recovery",
        candidate_policy_version="candidate-cross-scope-recovery-v1",
    ).opportunities[0]
    assert pair.provenance["same_atomic_scope"] is False
    assert pair.provenance["pair_local_soft_cue_matches"]

    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(
            pair.pair_identity,
            decision=NO_RELATION,
        )]},
        content=content,
        logical_schema=schema,
    )[0]

    assert parsed.decision == NO_RELATION
    assert parsed.hypotheses == ()


def test_clear_schema_gap_is_other_unsupported_not_related_to() -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(
            pair.pair_identity,
            decision=OTHER_UNSUPPORTED,
            evidence_spans=[{"start": 0, "end": len(CONTENT)}],
            reason_code="SCHEMA_GAP",
        )]},
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert parsed[0].decision == OTHER_UNSUPPORTED
    assert parsed[0].hypotheses == ()


def test_other_unsupported_reason_is_normalized_to_schema_gap() -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(
            pair.pair_identity,
            decision=OTHER_UNSUPPORTED,
            reason_code="schema does not contain this predicate",
        )]},
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert parsed[0].decision == OTHER_UNSUPPORTED
    assert parsed[0].reason_code == "SCHEMA_GAP"
    assert parsed[0].hypotheses == ()
    assert [(span.start, span.end) for span in parsed[0].evidence_spans] == [
        pair.context_span,
    ]


def test_terminal_fail_closed_decision_is_not_mislabeled_as_schema_gap() -> None:
    decision = PairClassificationDecision(
        pair_identity="pair-1",
        decision=OTHER_UNSUPPORTED,
        hypotheses=(),
        evidence_spans=(EvidenceSpan(start=0, end=1),),
        reason_code="CLASSIFIER_TERMINAL_PROTOCOL_FAILURE",
    )

    assert decision.decision_status == CLASSIFICATION_FAILED


def test_batch_schema_outside_predicate_fails_closed() -> None:
    content = "Acme uses Alloy and Bolt."
    mentions = [
        {
            "mention_id": "mention-acme",
            "occurrence_key": "occ-acme",
            "surface": "Acme",
            "mention_start": 0,
            "mention_end": 4,
            "canonical_id": "entity-acme",
            "assignment_version": "assign-v1",
            "type": "Organization",
        },
        {
            "mention_id": "mention-alloy",
            "occurrence_key": "occ-alloy",
            "surface": "Alloy",
            "mention_start": 10,
            "mention_end": 15,
            "canonical_id": "entity-alloy",
            "assignment_version": "assign-v1",
            "type": "Product",
        },
        {
            "mention_id": "mention-bolt",
            "occurrence_key": "occ-bolt",
            "surface": "Bolt",
            "mention_start": 20,
            "mention_end": 24,
            "canonical_id": "entity-bolt",
            "assignment_version": "assign-v1",
            "type": "Product",
        },
    ]
    plan = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v2",
    )
    pairs = [
        item for item in plan.admitted
        if item.left_occurrence_key == "occ-acme"
    ][:2]
    assert len(pairs) == 2

    response = {"decisions": [
        _decision(
            pairs[0].pair_identity,
            decision=NO_RELATION,
            reason_code="CO_OCCURRENCE_ONLY",
        ),
        _decision(
            pairs[1].pair_identity,
            hypotheses=[_hypothesis(
                subject="occ-acme",
                predicate="schema_outside_predicate",
                object_="occ-bolt",
            )],
        ),
    ]}
    with pytest.raises(ShadowClassificationError, match="predicate"):
        parse_classifier_response(
            pairs,
            response,
            content=content,
            logical_schema=_schema(),
        )

    recovered = parse_classifier_response(
        pairs,
        response,
        content=content,
        logical_schema=_schema(),
        allow_partial=True,
    )

    assert [item.pair_identity for item in recovered] == [
        pairs[0].pair_identity,
    ]
    assert recovered[0].decision == NO_RELATION


def test_partial_classifier_recovery_withholds_duplicate_and_unknown_members() -> None:
    pair = _opportunity()
    valid = _decision(pair.pair_identity, decision=NO_RELATION)

    assert parse_classifier_response(
        [pair],
        {"decisions": [valid, valid]},
        content=CONTENT,
        logical_schema=_schema(),
        allow_partial=True,
    ) == ()
    assert parse_classifier_response(
        [pair],
        {"decisions": [_decision("unknown", decision=NO_RELATION)]},
        content=CONTENT,
        logical_schema=_schema(),
        allow_partial=True,
    ) == ()


def test_one_pair_can_propose_two_concrete_hypotheses() -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(pair.pair_identity, hypotheses=[
            _hypothesis(predicate="uses"),
            _hypothesis(predicate="supplies"),
        ])]},
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert [item.predicate for item in parsed[0].hypotheses] == ["uses", "supplies"]
    assert all(item.status == "CLASSIFIED" for item in parsed[0].hypotheses)
    assert all(not hasattr(item, "direction") for item in parsed[0].hypotheses)


def test_ordered_endpoint_roles_are_the_only_direction_source() -> None:
    pair = _opportunity()
    forward = parse_classifier_response(
        [pair],
        {"decisions": [_decision(
            pair.pair_identity, hypotheses=[_hypothesis()],
        )]},
        content=CONTENT,
        logical_schema=_schema(),
    )[0].hypotheses[0]
    reverse = parse_classifier_response(
        [pair],
        {"decisions": [_decision(pair.pair_identity, hypotheses=[
            _hypothesis(
                subject="occ-alloy",
                predicate="used_by",
                object_="occ-acme",
            ),
        ])]},
        content=CONTENT,
        logical_schema=_schema(),
    )[0].hypotheses[0]

    assert (forward.subject_occurrence_key, forward.object_occurrence_key) == (
        "occ-acme", "occ-alloy",
    )
    assert (reverse.subject_occurrence_key, reverse.object_occurrence_key) == (
        "occ-alloy", "occ-acme",
    )


def test_structural_left_right_roles_bind_to_exact_frozen_occurrences() -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(pair.pair_identity, hypotheses=[
            _hypothesis(subject="RIGHT", predicate="used_by", object_="LEFT"),
        ])]},
        content=CONTENT,
        logical_schema=_schema(),
    )[0].hypotheses[0]

    assert parsed.subject_occurrence_key == pair.right_occurrence_key
    assert parsed.object_occurrence_key == pair.left_occurrence_key


def test_reverse_type_compatible_orientation_reaches_independent_role_verification() -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(pair.pair_identity, hypotheses=[
            _hypothesis(subject="RIGHT", predicate="uses", object_="LEFT"),
        ])]},
        content=CONTENT,
        logical_schema=_schema(),
    )[0].hypotheses[0]

    validation = validate_hypothesis(
        parsed,
        pair,
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert validation.status == "PASS"
    assert validation.reason_code == (
        "REVERSE_TYPE_COMPATIBLE_REQUIRES_ROLE_VERIFICATION"
    )
    assert validation.type_orientation == "reverse"


def test_legacy_direction_must_agree_with_endpoint_roles() -> None:
    pair = _opportunity()
    with pytest.raises(ShadowClassificationError, match="direction"):
        parse_classifier_response(
            [pair],
            {"decisions": [_decision(pair.pair_identity, hypotheses=[
                _hypothesis(direction="reversed"),
            ])]},
            content=CONTENT,
            logical_schema=_schema(),
        )


def test_symmetric_schema_is_metadata_not_a_second_direction_field() -> None:
    content = "Acme partners Globex."
    mentions = _mentions()
    mentions[1] = {
        **mentions[1],
        "mention_id": "mention-globex",
        "occurrence_key": "occ-globex",
        "surface": "Globex",
        "mention_start": 14,
        "mention_end": 20,
        "canonical_id": "entity-globex",
        "type": "Organization",
    }
    pair = _opportunity(content=content, mentions=mentions)
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(pair.pair_identity, hypotheses=[{
            "subject_occurrence_id": "occ-globex",
            "predicate": "partners_with",
            "object_occurrence_id": "occ-acme",
            "evidence_spans": [{"start": 0, "end": len(content)}],
            "classifier_confidence": 0.9,
        }])]},
        content=content,
        logical_schema=_schema(),
    )

    assert parsed[0].hypotheses[0].predicate_metadata["symmetric"] is True
    assert not hasattr(parsed[0].hypotheses[0], "direction")


def test_publish_mode_metadata_controls_review_only() -> None:
    schema = normalize_predicate_schema(_schema())
    related = schema["related_to"]

    assert related.review_only is True
    assert related.publishable is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda pair: _hypothesis(subject="occ-drifted"),
            "occurrence",
        ),
        (
            lambda pair: _hypothesis(
                evidence_spans=[{"start": 0, "end": len(CONTENT) + 1}],
            ),
            "evidence",
        ),
    ],
)
def test_classifier_parser_rejects_unknown_predicate_endpoint_drift_and_bad_span(
    mutator, message,
) -> None:
    pair = _opportunity()
    with pytest.raises(ShadowClassificationError, match=message):
        parse_classifier_response(
            [pair],
            {"decisions": [_decision(
                pair.pair_identity,
                hypotheses=[mutator(pair)],
            )]},
            content=CONTENT,
            logical_schema=_schema(),
        )


def test_classifier_parser_rejects_schema_outside_predicate() -> None:
    pair = _opportunity()
    with pytest.raises(ShadowClassificationError, match="predicate"):
        parse_classifier_response(
            [pair],
            {"decisions": [_decision(
                pair.pair_identity,
                hypotheses=[_hypothesis(predicate="invented")],
            )]},
            content=CONTENT,
            logical_schema=_schema(),
        )


def test_classifier_parser_rejects_duplicate_missing_and_unknown_pair_ids() -> None:
    pair = _opportunity()
    valid = _decision(pair.pair_identity, decision=NO_RELATION)
    with pytest.raises(ShadowClassificationError, match="duplicate"):
        parse_classifier_response(
            [pair], {"decisions": [valid, valid]},
            content=CONTENT, logical_schema=_schema(),
        )
    with pytest.raises(ShadowClassificationError, match="missing"):
        parse_classifier_response(
            [pair], {"decisions": []},
            content=CONTENT, logical_schema=_schema(),
        )
    with pytest.raises(ShadowClassificationError, match="unknown"):
        parse_classifier_response(
            [pair], {"decisions": [_decision("other", decision=NO_RELATION)]},
            content=CONTENT, logical_schema=_schema(),
        )


def test_classifier_parser_repairs_single_negative_abbreviated_pair_id_only() -> None:
    pair = _opportunity()
    abbreviated = pair.pair_identity[:40] + pair.pair_identity[-6:]
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(abbreviated, decision=NO_RELATION)]},
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert parsed[0].pair_identity == pair.pair_identity
    assert parsed[0].decision == NO_RELATION

    with pytest.raises(ShadowClassificationError, match="unknown"):
        parse_classifier_response(
            [pair],
            {"decisions": [_decision(
                abbreviated,
                decision=HYPOTHESES,
                hypotheses=[_hypothesis()],
            )]},
            content=CONTENT,
            logical_schema=_schema(),
        )


@pytest.mark.parametrize("raw", ["not-json", {"decisions": "broken"}, None])
def test_invalid_or_unrepairable_classifier_output_fails_closed(raw) -> None:
    pair = _opportunity()
    with pytest.raises(ShadowClassificationError):
        parse_classifier_response(
            [pair], raw, content=CONTENT, logical_schema=_schema(),
        )


def test_prompt_and_json_schema_version_the_generic_contract() -> None:
    pair = _opportunity()
    prompt = classifier_prompt([pair], content=CONTENT, logical_schema=_schema())
    schema = classifier_response_schema([pair])

    assert all(term in prompt for term in (
        "NO_RELATION",
        "OTHER_UNSUPPORTED",
        "semantic_definition",
        "direction_description",
        "optional_soft_cues",
        "不能作为关系、方向或发布的证据",
    ))
    assert all(term not in prompt for term in ("同一列表", "电话号码"))
    assert schema["properties"]["pair_id"]["type"] == "string"
    assert schema["properties"]["decision"]["enum"] == [
        NO_RELATION,
        OTHER_UNSUPPORTED,
        HYPOTHESES,
    ]
    payload = json.loads(prompt)
    assert payload["pairs"][0]["left_occurrence_id"] == "LEFT"
    assert payload["pairs"][0]["right_occurrence_id"] == "RIGHT"
    assert "predicates" not in payload
    assert payload["protocol_feedback"] == ""
    pair_payload = payload["pairs"][0]
    allowed_predicates = {
        metadata["name"] for metadata in pair.allowed_predicate_metadata
    }
    assert set(pair_payload["allowed_predicates"]) == allowed_predicates
    assert set(payload["predicate_catalog"]) == allowed_predicates
    assert "allowed_predicate_metadata" not in pair_payload
    endpoint_schema = (
        schema["properties"]["hypotheses"]["items"]["properties"]
    )
    assert endpoint_schema["subject_occurrence_id"]["enum"] == ["LEFT", "RIGHT"]
    assert endpoint_schema["object_occurrence_id"]["enum"] == ["LEFT", "RIGHT"]
    assert set(endpoint_schema["predicate"]["enum"]) == allowed_predicates
    assert "direction" not in endpoint_schema


def test_classifier_prompt_omits_pair_ineligible_global_predicates() -> None:
    schema = logical_schema(
        predicate_policy(
            "alpha_rel",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition="A generic allowed relation.",
        ),
        predicate_policy(
            "beta_rel",
            domain=["Location"],
            range_=["Location"],
            semantic_definition="A generic relation ineligible for this pair.",
        ),
        version="test-schema-pair-local-v1",
    )
    pair = _opportunity(schema=schema)

    prompt = classifier_prompt([pair], content=CONTENT, logical_schema=schema)
    response_schema = classifier_response_schema([pair])
    payload = json.loads(prompt)
    pair_payload = payload["pairs"][0]
    predicate_schema = (
        response_schema["properties"]["hypotheses"]["items"]["properties"]
        ["predicate"]
    )

    assert "alpha_rel" in pair_payload["allowed_predicates"]
    assert "beta_rel" not in pair_payload["allowed_predicates"]
    assert "beta_rel" not in prompt
    assert "beta_rel" not in predicate_schema["enum"]
    assert "predicates" not in payload


def test_classifier_prompt_deduplicates_predicate_catalog_across_pairs() -> None:
    pair = _opportunity()
    second = _opportunity(content=CONTENT, mentions=[
        {**_mentions()[0], "occurrence_key": "occ-acme-2"},
        {**_mentions()[1], "occurrence_key": "occ-alloy-2"},
    ])

    payload = json.loads(classifier_prompt(
        [pair, second], content=CONTENT, logical_schema=_schema(),
    ))

    assert len(payload["pairs"]) == 2
    assert set(payload["predicate_catalog"]) == {
        metadata["name"] for metadata in pair.allowed_predicate_metadata
    }
    assert all(
        "allowed_predicate_metadata" not in item
        for item in payload["pairs"]
    )
    assert len(payload["context_catalog"]) == 1
    assert {
        item["context_ref"] for item in payload["pairs"]
    } == {"CTX-1"}
    assert all("context" not in item for item in payload["pairs"])


def test_multi_pair_schema_uses_decisions_array_wrapper() -> None:
    pair = _opportunity()
    second = _opportunity(content=CONTENT, mentions=[
        {**_mentions()[0], "occurrence_key": "occ-acme-2"},
        {**_mentions()[1], "occurrence_key": "occ-alloy-2"},
    ])
    schema = classifier_response_schema([pair, second])

    decisions_schema = schema["properties"]["decisions"]
    assert decisions_schema["minItems"] == decisions_schema["maxItems"] == 2
    assert decisions_schema["items"]["required"] == [
        "pair_id",
        "decision",
        "hypotheses",
    ]
