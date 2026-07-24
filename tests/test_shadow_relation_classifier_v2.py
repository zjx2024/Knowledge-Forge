from __future__ import annotations

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
)


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
    return {
        "version": 7,
        "relation_types": [
            {
                "name": "uses",
                "domain": ["Organization"],
                "range": ["Product"],
                "description": "The organization uses the product.",
                "positive_examples": ["Acme uses Alloy."],
                "negative_examples": ["Alloy uses Acme."],
                "inverse_of": "used_by",
                "canonical_forward": "uses",
            },
            {
                "name": "used_by",
                "domain": ["Product"],
                "range": ["Organization"],
                "description": "Inverse of uses.",
                "inverse_of": "uses",
                "canonical_forward": "uses",
            },
            {
                "name": "supplies",
                "domain": ["Organization"],
                "range": ["Product"],
                "description": "The organization supplies the product.",
            },
            {
                "name": "related_to",
                "domain": ["*"],
                "range": ["*"],
                "description": "Generic semantic relation.",
            },
            {
                "name": "partners_with",
                "domain": ["Organization"],
                "range": ["Organization"],
                "symmetric": True,
                "description": "Mutual partnership.",
            },
        ],
    }


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
        )]},
        content=CONTENT,
        logical_schema=_schema(),
    )

    assert parsed[0].decision == NO_RELATION
    assert parsed[0].hypotheses == ()


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


def test_related_to_is_shadow_review_only_even_if_active_schema_is_not() -> None:
    schema = normalize_predicate_schema(_schema())
    related = schema["related_to"]

    assert related.review_only is True
    assert related.publishable is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda pair: _hypothesis(predicate="invented"), "predicate"),
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


@pytest.mark.parametrize("raw", ["not-json", {"decisions": "broken"}, None])
def test_invalid_or_unrepairable_classifier_output_fails_closed(raw) -> None:
    pair = _opportunity()
    with pytest.raises(ShadowClassificationError):
        parse_classifier_response(
            [pair], raw, content=CONTENT, logical_schema=_schema(),
        )


def test_prompt_and_json_schema_version_the_contract_and_hard_negatives() -> None:
    pair = _opportunity()
    prompt = classifier_prompt([pair], content=CONTENT, logical_schema=_schema())
    schema = classifier_response_schema([pair])

    assert all(term in prompt for term in (
        "NO_RELATION",
        "OTHER_UNSUPPORTED",
        "同一列表",
        "电话号码",
        "供应",
        "生产",
        "A 使用 B",
        "A 依赖 B",
    ))
    assert schema["minItems"] == schema["maxItems"] == 1
    assert "direction" not in schema["items"]["properties"]["hypotheses"]["items"]["properties"]
