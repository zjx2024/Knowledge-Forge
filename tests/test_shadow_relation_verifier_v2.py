from __future__ import annotations

import pytest

from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.relation_shadow_v2 import (
    HYPOTHESES,
    RETRYABLE_FAILED,
    REVIEW_ONLY,
    SUPPORTED,
    UNCERTAIN,
    parse_classifier_response,
    plan_pair_opportunities,
    supported_for_normalization,
    validate_hypothesis,
    verifier_prompt,
    verifier_response_schema,
    verify_hypothesis,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


def _mention(key, surface, start, *, entity_type="*", canonical=None):
    return {
        "mention_id": f"mention-{key}",
        "occurrence_key": f"occ-{key}",
        "surface": surface,
        "mention_start": start,
        "mention_end": start + len(surface),
        "canonical_id": canonical or f"entity-{key}",
        "assignment_version": "assignment-v1",
        "type": entity_type,
    }


def _schema(*predicates):
    return logical_schema(*predicates)


def _predicate(name, *, domain=None, range_=None, **metadata):
    return predicate_policy(
        name,
        domain=domain,
        range_=range_,
        **metadata,
    )


def _classified(content, mentions, schema, *, predicate, subject, object_, spans=None):
    pair = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v2",
    ).opportunities[0]
    decisions = parse_classifier_response(
        [pair],
        {"decisions": [{
            "pair_id": pair.pair_identity,
            "decision": HYPOTHESES,
            "hypotheses": [{
                "subject_occurrence_id": subject,
                "predicate": predicate,
                "object_occurrence_id": object_,
                "evidence_spans": spans or [{"start": 0, "end": len(content)}],
                "classifier_confidence": 0.99,
            }],
        }]},
        content=content,
        logical_schema=schema,
        classifier_policy_version="classifier-v2",
    )
    return pair, decisions[0].hypotheses[0]


def _supported(hypothesis):
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "verdict": "SUPPORTED",
        "reason_code": "DIRECTLY_ENTAILED",
        "supported_evidence_spans": [
            {"start": value.start, "end": value.end}
            for value in hypothesis.evidence_spans
        ],
    }


def test_semantic_non_entailment_is_decided_by_independent_verifier() -> None:
    content = "Automotive industry obtains battery material supplies."
    mentions = [
        _mention("auto", "Automotive industry", 0),
        _mention("battery", "battery material", 28),
    ]
    schema = _schema(_predicate("produces"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="produces",
        subject="occ-auto",
        object_="occ-battery",
    )
    calls = []
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )

    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args, **_kwargs: (
            calls.append(True)
            or {
                "hypothesis_id": hypothesis.hypothesis_id,
                "verdict": "REJECTED",
                "reason_code": "SEMANTICALLY_NOT_ENTAILED",
                "supported_evidence_spans": [],
            }
        ),
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == "PASS"
    assert attempt.verdict == "REJECTED"
    assert attempt.reason_code == "SEMANTICALLY_NOT_ENTAILED"
    assert calls == [True]


@pytest.mark.parametrize(
    ("content", "predicate"),
    [
        ("A uses B.", "uses"),
        ("A depends_on B.", "depends_on"),
    ],
)
def test_direction_semantics_are_decided_by_verifier(content, predicate) -> None:
    mentions = [_mention("a", "A", 0), _mention("b", "B", len(content) - 2)]
    schema = _schema(_predicate(predicate))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate=predicate,
        subject="occ-b",
        object_="occ-a",
    )

    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "REJECTED",
            "reason_code": "DIRECTION_NOT_ENTAILED",
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == "PASS"
    assert attempt.verdict == "REJECTED"
    assert attempt.reason_code == "DIRECTION_NOT_ENTAILED"


def test_list_cooccurrence_is_rejected_by_independent_verifier() -> None:
    content = "Materials: cobalt, lithium."
    mentions = [_mention("cobalt", "cobalt", 11), _mention("lithium", "lithium", 19)]
    schema = _schema(_predicate("supplies"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="supplies",
        subject="occ-cobalt",
        object_="occ-lithium",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(hypothesis, pair, content=content, logical_schema=schema),
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda _prompt, _schema: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "REJECTED",
            "reason_code": "COOCCURRENCE_ONLY",
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == "REJECTED"
    assert attempt.independent_call is True


def test_related_to_contact_cooccurrence_is_review_only_and_never_verified() -> None:
    content = "Contact: Alice; phone: 555-0100."
    mentions = [_mention("alice", "Alice", 9), _mention("phone", "555-0100", 23)]
    schema = _schema(_predicate(
        "related_to", publish_mode="review_only",
    ))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="related_to",
        subject="occ-alice",
        object_="occ-phone",
    )

    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: pytest.fail("review-only hypothesis called verifier"),
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == REVIEW_ONLY
    assert attempt.verdict == UNCERTAIN
    assert supported_for_normalization(attempt) is False


def test_evidence_must_cover_both_exact_occurrences() -> None:
    content = "Alpha uses Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 11)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="uses",
        subject="occ-alpha",
        object_="occ-beta",
        spans=[{"start": 0, "end": 5}],
    )

    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )

    assert validation.status == "REJECTED"
    assert validation.reason_code == "EVIDENCE_ENDPOINT_MISMATCH"


def test_evidence_for_same_name_other_occurrence_is_rejected() -> None:
    content = "Canada and Canada uses lithium."
    mentions = [
        _mention("canada-first", "Canada", 0, canonical="canada-1"),
        _mention("canada-second", "Canada", 11, canonical="canada-2"),
        _mention("lithium", "lithium", 23),
    ]
    schema = _schema(_predicate("uses"))
    plan = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v2",
        sentence_window=2,
    )
    pair = next(
        item for item in plan.opportunities
        if {item.left_occurrence_key, item.right_occurrence_key}
        == {"occ-canada-second", "occ-lithium"}
    )
    hypothesis = parse_classifier_response(
        [pair],
        {"decisions": [{
            "pair_id": pair.pair_identity,
            "decision": HYPOTHESES,
            "hypotheses": [{
                    "subject_occurrence_id": "occ-canada-second",
                    "predicate": "uses",
                    "object_occurrence_id": "occ-lithium",
                    "evidence_spans": [
                        {"start": 0, "end": 6},
                        {"start": 23, "end": 30},
                    ],
                    "classifier_confidence": 0.9,
            }],
        }]},
        content=content,
        logical_schema=schema,
    )[0].hypotheses[0]
    validation = validate_hypothesis(
        hypothesis,
        pair,
        content=content,
        logical_schema=schema,
    )

    assert validation.status == "REJECTED"
    assert validation.reason_code == "EVIDENCE_ENDPOINT_MISMATCH"


def test_unknown_specific_type_is_review_only_but_wildcard_can_reach_verifier() -> None:
    content = "Alpha uses Beta."
    mentions = [
        _mention("alpha", "Alpha", 0, entity_type=""),
        _mention("beta", "Beta", 11, entity_type=""),
    ]
    wildcard_schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content,
        mentions,
        wildcard_schema,
        predicate="uses",
        subject="occ-alpha",
        object_="occ-beta",
    )
    specific_schema = _schema(_predicate(
        "uses", domain=["Organization"], range_=["Product"],
    ))

    unresolved = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=specific_schema,
    )
    wildcard = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=wildcard_schema,
    )
    attempt = verify_hypothesis(
        hypothesis,
        wildcard,
        pair,
        content=content,
        logical_schema=wildcard_schema,
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert unresolved.status == REVIEW_ONLY
    assert unresolved.reason_code == "TYPE_UNRESOLVED"
    assert wildcard.status == "PASS"
    assert attempt.verdict == SUPPORTED


@pytest.mark.parametrize(
    ("content", "verdict", "reason"),
    [
        ("Alpha does not use Beta.", "REJECTED", "NEGATED"),
        ("Alpha might use Beta.", "UNCERTAIN", "HYPOTHETICAL"),
    ],
)
def test_negation_and_modality_are_decided_by_verifier(
    content, verdict, reason,
) -> None:
    beta_start = content.index("Beta")
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", beta_start)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="uses",
        subject="occ-alpha",
        object_="occ-beta",
    )

    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": verdict,
            "reason_code": reason,
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == "PASS"
    assert attempt.verdict == verdict
    assert attempt.reason_code == reason


@pytest.mark.parametrize(
    "verifier",
    [
        lambda *_args: (_ for _ in ()).throw(TimeoutError("timeout")),
        lambda *_args: "not-json",
        lambda *_args: {"hypothesis_id": "missing"},
    ],
)
def test_verifier_timeout_missing_or_invalid_output_is_retryable_failed(verifier) -> None:
    content = "Alpha uses Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 11)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="uses",
        subject="occ-alpha",
        object_="occ-beta",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(hypothesis, pair, content=content, logical_schema=schema),
        pair,
        content=content,
        logical_schema=schema,
        verifier=verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == RETRYABLE_FAILED
    assert supported_for_normalization(attempt) is False


def test_verifier_correlation_is_caller_bound_and_model_cannot_invent_an_id():
    content = "Alpha uses Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 11)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="uses",
        subject="occ-alpha",
        object_="occ-beta",
    )
    response_contract = verifier_response_schema(hypothesis.hypothesis_id)
    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda _prompt, contract: {
            "verdict": "SUPPORTED",
            "reason_code": "DIRECTLY_ENTAILED",
            "supported_evidence_spans": [{"start": 0, "end": len(content)}],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert "hypothesis_id" not in response_contract["properties"]
    assert "hypothesis_id" not in response_contract["required"]
    assert attempt.hypothesis_id == hypothesis.hypothesis_id
    assert attempt.verdict == SUPPORTED
    assert attempt.independent_call is True


def test_same_pair_hypotheses_are_verified_independently() -> None:
    content = "Alpha uses and supplies Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 24)]
    schema = _schema(_predicate("uses"), _predicate("supplies"))
    pair = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v2",
    ).opportunities[0]
    hypotheses = parse_classifier_response(
        [pair],
        {"decisions": [{
            "pair_id": pair.pair_identity,
            "decision": HYPOTHESES,
            "hypotheses": [
                {
                    "subject_occurrence_id": "occ-alpha",
                    "predicate": predicate,
                    "object_occurrence_id": "occ-beta",
                    "evidence_spans": [{"start": 0, "end": len(content)}],
                    "classifier_confidence": 0.9,
                }
                for predicate in ("uses", "supplies")
            ],
        }]},
        content=content,
        logical_schema=schema,
    )[0].hypotheses

    attempts = [
        verify_hypothesis(
            hypothesis,
            validate_hypothesis(
                hypothesis, pair, content=content, logical_schema=schema,
            ),
            pair,
            content=content,
            logical_schema=schema,
            verifier=(
                (lambda *_args, item=hypothesis: _supported(item))
                if hypothesis.predicate == "uses"
                else lambda *_args, item=hypothesis: {
                    "hypothesis_id": item.hypothesis_id,
                    "verdict": "REJECTED",
                    "reason_code": "NOT_ENTAILED",
                    "supported_evidence_spans": [],
                }
            ),
            verifier_model_revision="verifier-model-r1",
        )
        for hypothesis in hypotheses
    ]

    assert [item.verdict for item in attempts] == ["SUPPORTED", "REJECTED"]
    assert [supported_for_normalization(item) for item in attempts] == [True, False]


def test_verifier_prompt_contains_complete_directed_fact_and_exact_occurrences() -> None:
    content = "Alpha uses Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 11)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="uses",
        subject="occ-alpha",
        object_="occ-beta",
    )

    prompt = verifier_prompt(
        hypothesis, pair, content=content, logical_schema=schema,
    )

    assert all(value in prompt for value in (
        "occ-alpha", "uses", "occ-beta", "Alpha uses Beta.",
    ))
    assert "classifier_confidence" not in prompt


def test_verification_attempt_migration_is_append_only_and_versioned() -> None:
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0015_shadow_relation_v2_verification"
    )

    assert "CREATE TABLE IF NOT EXISTS shadow_relation_verification_attempts" in migration.sql
    assert "UNIQUE(hypothesis_id,attempt_no)" in migration.sql
    assert "verifier_prompt_version" in migration.sql
    assert "verifier_model_revision" in migration.sql
