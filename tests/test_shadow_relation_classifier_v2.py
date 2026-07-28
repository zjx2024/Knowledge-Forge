from __future__ import annotations

import json

import pytest

from knowledge_forge.relation_shadow_v2 import (
    HYPOTHESES,
    NO_RELATION,
    OTHER_UNSUPPORTED,
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


def test_batch_schema_outside_predicate_only_downgrades_that_pair() -> None:
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

    parsed = parse_classifier_response(
        pairs,
        {"decisions": [
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
        ]},
        content=content,
        logical_schema=_schema(),
    )

    assert [item.decision for item in parsed] == [
        NO_RELATION,
        OTHER_UNSUPPORTED,
    ]
    assert parsed[1].reason_code == "SCHEMA_GAP"
    assert parsed[1].hypotheses == ()
    assert [(span.start, span.end) for span in parsed[1].evidence_spans] == [
        pairs[1].context_span,
    ]


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


def test_type_incompatible_orientation_is_classified_then_rejected_by_validation() -> None:
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

    assert validation.status == "REJECTED"
    assert validation.reason_code == "TYPE_INCOMPATIBLE"


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


def test_classifier_parser_downgrades_schema_outside_predicate_to_gap() -> None:
    pair = _opportunity()
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [_decision(
            pair.pair_identity,
            hypotheses=[_hypothesis(predicate="invented")],
        )]},
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert parsed[0].decision == OTHER_UNSUPPORTED
    assert parsed[0].reason_code == "SCHEMA_GAP"
    assert parsed[0].hypotheses == ()


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
    endpoint_schema = (
        schema["properties"]["hypotheses"]["items"]["properties"]
    )
    assert endpoint_schema["subject_occurrence_id"]["enum"] == ["LEFT", "RIGHT"]
    assert endpoint_schema["object_occurrence_id"]["enum"] == ["LEFT", "RIGHT"]
    assert "direction" not in endpoint_schema


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
