from __future__ import annotations

import json
from dataclasses import replace

import pytest

from knowledge_forge.config import Config
from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.relation_extraction import (
    EvidenceSpan,
    HYPOTHESES,
    PASS,
    REJECTED,
    RETRYABLE_FAILED,
    REVIEW_ONLY,
    GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    SUPPORTED,
    RelationClassificationError,
    UNCERTAIN,
    atomic_fact_inventory_contains_pair,
    atomic_fact_inventory_prompt,
    atomic_fact_inventory_response_schema,
    atomic_frame_schema_alignment_prompt,
    atomic_frame_schema_alignment_response_schema,
    parse_atomic_fact_inventory,
    parse_atomic_frame_schema_alignment,
    parse_classifier_response,
    parse_verifier_directional_entailment,
    parse_verifier_exact_argument,
    parse_verifier_semantic_contrast,
    parse_verifier_role_alignment,
    plan_pair_opportunities,
    normalize_verified_hypotheses,
    supported_for_normalization,
    source_frame_polarity_review_eligible,
    validate_hypothesis,
    verifier_prompt,
    verifier_exact_argument_prompt,
    verifier_exact_argument_response_schema,
    verifier_role_alignment_prompt,
    verifier_role_alignment_response_schema,
    verifier_semantic_contrast_prompt,
    verifier_role_probe_prompt,
    verifier_role_probe_response_schema,
    verifier_response_schema,
    verify_hypothesis,
    verify_hypothesis_lean,
)
from tests.grounded_relation_fixtures import logical_schema, predicate_policy


def _mention(
    key, surface, start, *, entity_type="*", canonical=None,
    endpoint_candidate=False,
):
    return {
        "mention_id": f"mention-{key}",
        "occurrence_key": f"occ-{key}",
        "surface": surface,
        "mention_start": start,
        "mention_end": start + len(surface),
        "canonical_id": canonical or f"entity-{key}",
        "assignment_version": "assignment-v1",
        "type": entity_type,
        "endpoint_candidate": endpoint_candidate,
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
                "evidence_spans": spans or [{
                    "start": pair.context_span[0],
                    "end": pair.context_span[1],
                }],
                "classifier_confidence": 0.99,
            }],
        }]},
        content=content,
        logical_schema=schema,
        classifier_policy_version="classifier-v2",
    )
    return pair, decisions[0].hypotheses[0]


def _supported(hypothesis, *, symmetric=False):
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "verdict": "SUPPORTED",
        "reason_code": "DIRECTLY_ENTAILED",
        "forward_entailment": "ENTAILED",
        "reverse_entailment": "ENTAILED" if symmetric else "NOT_ENTAILED",
        "subject_role_supported": True,
        "object_role_supported": True,
        "supported_evidence_spans": [
            {"start": value.start, "end": value.end}
            for value in hypothesis.evidence_spans
        ],
    }


def _role_probe(*_args):
    return {
        "subject_role": "source endpoint role",
        "object_role": "target endpoint role",
    }


def _role_alignment_probe(prompt, *_args):
    payload = json.loads(prompt)
    if payload.get("symmetric"):
        return {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "ALIGNED",
            "exact_argument_orientation": "SYMMETRIC",
        }
    return {
        "forward_alignment": "ALIGNED",
        "reverse_alignment": "MISALIGNED",
        "exact_argument_orientation": "FORWARD",
    }


def _uncertain_role_alignment_probe(prompt, *_args):
    return {
        "forward_alignment": "UNCERTAIN",
        "reverse_alignment": "UNCERTAIN",
        "exact_argument_orientation": "NEITHER",
    }


def test_lean_verifier_uses_one_ordered_call_and_never_invokes_auxiliaries() -> None:
    content = "A uses B."
    mentions = [_mention("a", "A", 0), _mention("b", "B", 7)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content, mentions, schema,
        predicate="uses", subject="occ-a", object_="occ-b",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    calls = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("lean verifier invoked a legacy auxiliary")

    attempt = verify_hypothesis_lean(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        role_probe=forbidden,
        role_alignment_probe=forbidden,
        atomic_fact_probe=forbidden,
        atomic_frame_alignment_probe=forbidden,
        verifier=lambda *_args: calls.append(True) or _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert calls == [True]
    assert attempt.forward_role_alignment == "FINAL_VERIFIER"


def test_lean_verifier_rejects_reverse_only_result_without_rewriting_claim() -> None:
    content = "A uses B."
    mentions = [_mention("a", "A", 0), _mention("b", "B", 7)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content, mentions, schema,
        predicate="uses", subject="occ-a", object_="occ-b",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    response = _supported(hypothesis)
    response["forward_entailment"] = "NOT_ENTAILED"
    response["reverse_entailment"] = "ENTAILED"

    attempt = verify_hypothesis_lean(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "REVERSE_ONLY_CLAIM_REQUIRES_OWN_CANDIDATE"


@pytest.mark.parametrize("misspelling", ["ENTAILD", "ENTAILDED"])
def test_lean_verifier_repairs_unambiguous_single_edit_entailment_enum(
    misspelling: str,
) -> None:
    content = "A uses B."
    mentions = [_mention("a", "A", 0), _mention("b", "B", 7)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content, mentions, schema,
        predicate="uses", subject="occ-a", object_="occ-b",
    )
    response = _supported(hypothesis)
    response["forward_entailment"] = misspelling

    attempt = verify_hypothesis_lean(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.forward_entailment == "ENTAILED"


def test_lean_verifier_uses_independent_secondary_orientation_only_as_corroboration() -> None:
    content = "A uses B."
    mentions = [_mention("a", "A", 0), _mention("b", "B", 7)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content, mentions, schema,
        predicate="uses", subject="occ-a", object_="occ-b",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    ambiguous_reverse = _supported(hypothesis)
    ambiguous_reverse["reverse_entailment"] = "ENTAILED"

    without_audit = verify_hypothesis_lean(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: ambiguous_reverse,
        verifier_model_revision="verifier-model-r1",
    )
    with_audit = verify_hypothesis_lean(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: ambiguous_reverse,
        verifier_model_revision="verifier-model-r1",
        independent_orientation_audit="FORWARD",
    )

    assert without_audit.verdict == REJECTED
    assert without_audit.reason_code == "DIRECTION_NOT_EXCLUSIVE"
    assert with_audit.verdict == SUPPORTED
    assert with_audit.forward_entailment == "ENTAILED"
    assert with_audit.reverse_entailment == "NOT_ENTAILED"

    no_forward = dict(ambiguous_reverse)
    no_forward["forward_entailment"] = "NOT_ENTAILED"
    assert verify_hypothesis_lean(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: no_forward,
        verifier_model_revision="verifier-model-r1",
        independent_orientation_audit="FORWARD",
    ).verdict == REJECTED


def test_lean_verifier_keeps_malformed_final_response_retryable() -> None:
    content = "A uses B."
    mentions = [_mention("a", "A", 0), _mention("b", "B", 7)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content, mentions, schema,
        predicate="uses", subject="occ-a", object_="occ-b",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )

    attempt = verify_hypothesis_lean(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: {"verdict": "SUPPORTED"},
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == RETRYABLE_FAILED
    assert attempt.reason_code == "VERIFIER_CALL_OR_RESPONSE_FAILED"


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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
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
        attempt_no=Config.relation_max_attempts,
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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "REJECTED",
            "reason_code": "DIRECTION_NOT_ENTAILED",
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda _prompt, _schema: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "REJECTED",
            "reason_code": "COOCCURRENCE_ONLY",
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
    )

    assert attempt.verdict == "REJECTED"
    assert attempt.independent_call is True


def test_related_to_contact_cooccurrence_is_review_only_and_never_verified() -> None:
    content = "Contact: Alice, phone: 555-0100."
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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: pytest.fail("review-only hypothesis called verifier"),
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
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


def test_unknown_specific_type_reaches_verifier_but_wildcard_remains_supported() -> None:
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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert unresolved.status == "PASS"
    assert unresolved.reason_code == "TYPE_UNRESOLVED_REQUIRES_VERIFICATION"
    assert wildcard.status == "PASS"
    assert attempt.verdict == SUPPORTED


def test_endpoint_projection_unknown_type_reaches_verifier_after_exact_span_gate() -> None:
    content = "工业机器人应用于汽车制造生产线。"
    schema = _schema(_predicate(
        "applies_to", domain=["Product"], range_=["Organization"],
        optional_soft_cues=["应用于"],
    ))
    mentions = [
        _mention("robot", "工业机器人", 0, entity_type="Product"),
        _mention(
            "line", "汽车制造生产线", 8, entity_type="",
            endpoint_candidate=True,
        ),
    ]
    pair, hypothesis = _classified(
        content, mentions, schema, predicate="applies_to",
        subject="occ-robot", object_="occ-line",
        spans=[(0, len(content) - 1)],
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    assert validation.status == PASS
    assert validation.reason_code == "TYPE_UNRESOLVED_ENDPOINT_REQUIRES_VERIFICATION"


def test_unknown_type_does_not_bypass_review_only_publish_mode() -> None:
    content = "Alpha uses Beta."
    mentions = [
        _mention("alpha", "Alpha", 0, entity_type=""),
        _mention("beta", "Beta", 11, entity_type=""),
    ]
    schema = _schema(_predicate(
        "uses", domain=["Organization"], range_=["Product"],
        publish_mode="review_only",
    ))
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

    assert validation.status == REVIEW_ONLY
    assert validation.reason_code == "PUBLISH_MODE_REVIEW_ONLY"


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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": verdict,
            "reason_code": reason,
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == RETRYABLE_FAILED
    assert supported_for_normalization(attempt) is False


def test_supported_claim_without_exact_endpoint_evidence_is_rejected() -> None:
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
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "verdict": "ENTAILED",
            "reason_code": "DIRECTLY_ENTAILED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{"start": 0, "end": 5}],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "SUPPORTED_EVIDENCE_MISSING_ENDPOINT"
    assert attempt.independent_call is True
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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda _prompt, contract: {
            "verdict": "SUPPORTED",
            "reason_code": "DIRECTLY_ENTAILED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            # Common structured-output drift collapses one span's outer
            # array. Bounds and exact endpoint coverage remain fully checked.
            "supported_evidence_spans": [0, len(content)],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert "hypothesis_id" not in response_contract["properties"]
    assert "hypothesis_id" not in response_contract["required"]
    assert {
        "forward_entailment", "reverse_entailment",
        "subject_role_supported", "object_role_supported",
    }.issubset(response_contract["required"])
    assert response_contract["properties"]["reason_code"]["maxLength"] == 160
    assert "orientation marked ENTAILED" in response_contract["properties"][
        "subject_role_supported"
    ]["description"]
    assert "NOT_ENTAILED orientation does not make this false" in (
        response_contract["properties"]["object_role_supported"]["description"]
    )
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
            role_probe=_role_probe,
            role_alignment_probe=_role_alignment_probe,
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
            attempt_no=Config.relation_max_attempts,
        )
        for hypothesis in hypotheses
    ]

    assert [item.verdict for item in attempts] == ["SUPPORTED", "REJECTED"]
    assert [supported_for_normalization(item) for item in attempts] == [True, False]


def test_verifier_may_expand_evidence_within_frozen_context_unit() -> None:
    content = "According to the report, Alpha uses Beta in production."
    mentions = [_mention("alpha", "Alpha", 25), _mention("beta", "Beta", 36)]
    schema = _schema(_predicate("uses"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="uses",
        subject="occ-alpha",
        object_="occ-beta",
        spans=[{"start": 25, "end": 40}],
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
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": "SUPPORTED",
            "reason_code": "DIRECTLY_ENTAILED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{"start": 0, "end": len(content)}],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == PASS
    assert attempt.verdict == SUPPORTED
    assert attempt.supported_evidence_spans == (EvidenceSpan(0, len(content)),)


@pytest.mark.parametrize(
    ("response_overrides", "reason_code"),
    [
        ({"forward_entailment": "NOT_ENTAILED"}, "DIRECTION_NOT_EXCLUSIVE"),
        ({"subject_role_supported": False}, "ORDERED_ROLE_NOT_ENTAILED"),
        ({"object_role_supported": False}, "ORDERED_ROLE_NOT_ENTAILED"),
        ({"reverse_entailment": "ENTAILED"}, "DIRECTION_NOT_EXCLUSIVE"),
    ],
)
def test_supported_verdict_cannot_bypass_ordered_role_gate(
    response_overrides, reason_code,
) -> None:
    content = "Alpha acts on Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 14)]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )
    response = _supported(hypothesis)
    response.update(response_overrides)

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == reason_code
    assert supported_for_normalization(attempt) is False


def test_symmetric_predicate_requires_bidirectional_entailment() -> None:
    content = "Alpha is linked with Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 21)]
    schema = _schema(_predicate("alpha_rel", symmetric=True))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )

    rejected = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )
    supported = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: _supported(hypothesis, symmetric=True),
        verifier_model_revision="verifier-model-r1",
    )

    assert rejected.verdict == REJECTED
    assert rejected.reason_code == "DIRECTION_NOT_EXCLUSIVE"
    assert supported.verdict == SUPPORTED


@pytest.mark.parametrize(
    ("forward", "reverse", "subject_role", "object_role"),
    [
        (True, False, True, True),
        ("true", "false", "yes", "true"),
    ],
)
def test_boolean_entailment_protocol_variant_is_normalized_without_bypassing_gate(
    forward, reverse, subject_role, object_role,
):
    content = "Alpha acts on Beta."
    mentions = [_mention("alpha", "Alpha", 0), _mention("beta", "Beta", 14)]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )
    response = _supported(hypothesis)
    response.update({
        "forward_entailment": forward,
        "reverse_entailment": reverse,
        "subject_role_supported": subject_role,
        "object_role_supported": object_role,
    })

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.forward_entailment == "ENTAILED"
    assert attempt.reverse_entailment == "NOT_ENTAILED"


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
        hypothesis,
        pair,
        content=content,
        logical_schema=schema,
        blind_role_audit=_role_probe(),
    )

    assert all(value in prompt for value in (
        "occ-alpha", "uses", "occ-beta", "Alpha uses Beta.",
    ))
    assert "classifier_confidence" not in prompt


def test_verifier_prompt_exposes_exact_multi_role_projection_contract() -> None:
    content = "Actor sends Payload to Destination."
    mentions = [
        _mention("payload", "Payload", content.index("Payload")),
        _mention("destination", "Destination", content.index("Destination")),
    ]
    schema = _schema(_predicate(
        "delivered_to",
        semantic_definition=(
            "The subject is an action theme and the object is its destination."
        ),
        open_relation_policy={
            "enabled": True,
            "allowed_candidate_sources": [
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            ],
        },
    ))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="delivered_to",
        subject="occ-payload",
        object_="occ-destination",
    )
    hypothesis = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_binding_source": "MODEL_SOURCE_BINDING",
            "source_frame_relation": "sends to",
            "source_frame_roles": ["action_theme", "role_complement"],
            "source_syntax_origin": "ordered_three_role_construction",
            "source_frame_assertion_status": "ASSERTED",
            "source_frame_explicit_negation": False,
            "source_frame_evidence_span": [0, len(content)],
            "source_frame_evidence_text": content,
            "source_resolved_reference_endpoints": [{
                "frame_role": "subject",
                "proxy_occurrence_key": "occ-payload",
                "reference_text": "it",
                "reference_span": [content.index("Payload"), content.index("Payload") + 7],
                "reference_role": "grammatical_subject",
                "antecedent_occurrence_key": "occ-actor",
                "antecedent_surface": "Actor",
                "antecedent_span": [0, 5],
                "antecedent_canonical_id": "actor",
                "resolution_source": "DISCOURSE_ROLE_ADJUDICATOR",
            }],
        },
    )

    payload = json.loads(verifier_prompt(
        hypothesis,
        pair,
        content=content,
        logical_schema=schema,
        blind_role_audit={},
    ))
    contract = payload["source_frame_candidate"]["projection_contract"]
    assert contract["projection_kind"] == "theme_to_connector_complement"
    assert contract["subject_source_role"] == "action_theme"
    assert "grammatical actor is a third role" in contract["excluded_role"]
    assert "omitted third role" in payload["instruction"]
    relation_definition = payload["ordered_claims"]["forward"][
        "relation_definition"
    ]
    assert "theme_to_connector_complement" in relation_definition
    assert "subject is the action theme" in relation_definition
    assert "grammatical actor is a third role" in relation_definition
    assert "must not also be required to be the grammatical actor" in (
        payload["instruction"]
    )
    assert "swapped orientation marked NOT_ENTAILED" in payload["instruction"]
    assert payload["source_frame_candidate"]["assertion_status"] == "ASSERTED"
    assert payload["source_frame_candidate"][
        "explicit_relation_negation"
    ] is False
    assert payload["source_frame_candidate"][
        "resolved_reference_endpoints"
    ][0]["antecedent_surface"] == "Actor"
    assert payload["source_frame_candidate"]["frame_evidence"] == {
        "span": [0, len(content)], "text": content,
    }
    assert "Negation scope is predicate-local" in payload["instruction"]
    assert "exact evidence proxy" in payload["instruction"]


def test_negation_scope_rejection_routes_only_source_asserted_role_bound_frame() -> None:
    content = "Alice did not stop because she saw Rabbit."
    frame_text = "she saw Rabbit"
    frame_start = content.index(frame_text)
    frame_span = [frame_start, frame_start + len(frame_text)]
    mentions = [
        _mention("she", "she", frame_start),
        _mention("rabbit", "Rabbit", content.index("Rabbit")),
    ]
    schema = _schema(_predicate("observes"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="observes",
        subject="occ-she",
        object_="occ-rabbit",
    )
    hypothesis = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "candidate_source": "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            "source_frame_relation": "saw",
            "source_frame_assertion_status": "ASSERTED",
            "source_frame_explicit_negation": False,
            "source_frame_evidence_span": frame_span,
            "source_frame_evidence_text": frame_text,
        },
    )
    attempt = verify_hypothesis_lean(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        verifier=lambda *_args: {
            "verdict": "REJECTED",
            "reason_code": "negation_governs_frame",
            "forward_entailment": "NOT_ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert source_frame_polarity_review_eligible(hypothesis, attempt)
    assert not source_frame_polarity_review_eligible(
        hypothesis, replace(attempt, reason_code="PREDICATE_SEMANTICS_MISMATCH"),
    )
    assert not source_frame_polarity_review_eligible(
        replace(
            hypothesis,
            predicate_metadata={
                **hypothesis.predicate_metadata,
                "source_frame_explicit_negation": True,
            },
        ),
        attempt,
    )
    assert not source_frame_polarity_review_eligible(
        hypothesis, replace(attempt, subject_role_supported=False),
    )


def test_asserted_polarity_audit_narrows_second_verifier_to_exact_frame() -> None:
    content = "Alice did not stop because she saw Rabbit."
    frame_text = "she saw Rabbit"
    frame_start = content.index(frame_text)
    frame_span = [frame_start, frame_start + len(frame_text)]
    mentions = [
        _mention("she", "she", frame_start),
        _mention("rabbit", "Rabbit", content.index("Rabbit")),
    ]
    schema = _schema(_predicate("observes"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="observes",
        subject="occ-she",
        object_="occ-rabbit",
    )
    hypothesis = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_frame_relation": "saw",
            "source_frame_evidence_span": frame_span,
            "source_frame_evidence_text": frame_text,
            "source_frame_polarity_audit": "ASSERTED",
            "source_frame_polarity_audit_version": "polarity-v1",
        },
    )

    payload = json.loads(verifier_prompt(
        hypothesis,
        pair,
        content=content,
        logical_schema=schema,
        blind_role_audit={},
    ))

    assert payload["context"]["text"] == frame_text
    assert payload["context"]["start"] == frame_span[0]
    assert payload["context"]["end"] == frame_span[1]
    assert payload["context"]["immutable_source_span"] == list(
        pair.context_span
    )
    assert payload["source_frame_candidate"][
        "independent_polarity_audit"
    ]["result"] == "ASSERTED"
    assert "does not confirm the Schema predicate" in payload["instruction"]

    tampered = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_frame_evidence_text": "tampered",
        },
    )
    fallback = json.loads(verifier_prompt(
        tampered,
        pair,
        content=content,
        logical_schema=schema,
        blind_role_audit={},
    ))
    assert fallback["context"]["text"] != "tampered"


def test_role_probe_is_predicate_blind_and_final_verifier_is_conclusion_independent() -> None:
    content = "Supplier Alpha delivers equipment to Customer Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )

    probe = verifier_role_probe_prompt(
        hypothesis, pair, content=content,
    )
    final = verifier_prompt(
        hypothesis,
        pair,
        content=content,
        logical_schema=schema,
        blind_role_audit={
            "subject_role": "supplier",
            "object_role": "recipient",
        },
    )

    assert "alpha_rel" not in probe
    assert "semantic_definition" not in probe
    probe_payload = json.loads(probe)
    assert "endpoint_a" in probe_payload
    assert "endpoint_b" in probe_payload
    assert "subject" not in probe_payload
    assert "object" not in probe_payload
    assert "standalone role phrase" in probe_payload["instruction"]
    assert verifier_role_probe_response_schema()["required"] == [
        "endpoint_a_complete_argument",
        "endpoint_b_complete_argument",
        "endpoint_a_argument_start", "endpoint_a_argument_end",
        "endpoint_b_argument_start", "endpoint_b_argument_end",
        "direct_relation_present",
        "subject_role", "object_role", "expressed_relation",
    ]
    assert '"subject_role":"supplier"' not in final
    assert '"object_role":"recipient"' not in final
    assert "blind_context_role_audit" not in json.loads(final)

    alignment = verifier_role_alignment_prompt(
        hypothesis,
        logical_schema=schema,
        blind_role_audit={
            "subject_role": "supplier",
            "object_role": "recipient",
        },
        opportunity=pair,
        content=content,
    )
    assert "alpha_rel" not in alignment
    alignment_payload = json.loads(alignment)
    assert "context_roles" not in alignment_payload
    assert alignment_payload["source_context"]["text"] == content
    assert alignment_payload["endpoint_a"]["surface"] == "Alpha"
    assert alignment_payload["endpoint_b"]["surface"] == "Beta"
    assert alignment_payload["predicate_blind_role_audit"][
        "endpoint_a_contextual_role"
    ] == "supplier"
    assert alignment_payload["predicate_blind_role_audit"][
        "endpoint_b_contextual_role"
    ] == "recipient"
    assert set(alignment_payload["required_roles"]) == {
        "relation_source_role", "relation_target_role",
    }
    assert verifier_role_alignment_response_schema()["required"] == [
        "endpoint_a_vs_source", "endpoint_b_vs_target",
        "endpoint_b_vs_source", "endpoint_a_vs_target",
        "forward_relation_vs_definition", "reverse_relation_vs_definition",
        "semantic_contrast",
    ]


def test_role_alignment_derives_direction_from_four_independent_matches() -> None:
    result = parse_verifier_role_alignment({
        "endpoint_a_vs_source": "NO",
        "endpoint_b_vs_target": "MISMATCH",
        "endpoint_b_vs_source": "MATCH",
        "endpoint_a_vs_target": "YES",
    })

    assert result == {
        "forward_alignment": "MISALIGNED",
        "reverse_alignment": "ALIGNED",
    }


def test_relation_semantics_override_role_similarity_without_name_rules() -> None:
    mismatched = parse_verifier_role_alignment({
        "endpoint_a_vs_source": "MATCH",
        "endpoint_b_vs_target": "MATCH",
        "endpoint_b_vs_source": "MISMATCH",
        "endpoint_a_vs_target": "MISMATCH",
        "forward_relation_vs_definition": "MISMATCH",
        "reverse_relation_vs_definition": "MISMATCH",
        "semantic_contrast": "NEGATIVE_EXAMPLE",
    })
    supported = parse_verifier_role_alignment({
        "endpoint_a_vs_source": "MATCH",
        "endpoint_b_vs_target": "UNCERTAIN",
        "endpoint_b_vs_source": "MISMATCH",
        "endpoint_a_vs_target": "UNCERTAIN",
        "forward_relation_vs_definition": "MATCH",
        "reverse_relation_vs_definition": "MISMATCH",
        "semantic_contrast": "DEFINITION",
    })

    assert mismatched == {
        "forward_alignment": "MISALIGNED",
        "reverse_alignment": "MISALIGNED",
    }
    assert supported == {
        "forward_alignment": "ALIGNED",
        "reverse_alignment": "MISALIGNED",
    }


def test_role_alignment_accepts_generic_model_field_aliases() -> None:
    result = parse_verifier_role_alignment({
        "endpoint_a_role_vs_relation_source_role": "MATCH",
        "endpoint_b_role_vs_relation_target_role": "MATCH",
        "endpoint_b_role_vs_relation_source_role": "MISMATCH",
        "endpoint_a_role_vs_relation_target_role": "MISMATCH",
        "semantic_contrast": "DEFINITION",
    })

    assert result == {
        "forward_alignment": "ALIGNED",
        "reverse_alignment": "MISALIGNED",
    }


def test_independent_semantic_mismatch_rejects_abbreviated_role_response() -> None:
    result = parse_verifier_role_alignment({
        "endpoint_a_role": "actor",
        "endpoint_b_role": "provider of a third entity",
        "match": False,
        "semantic_contrast": "NEITHER",
    })

    assert result == {
        "forward_alignment": "MISALIGNED",
        "reverse_alignment": "MISALIGNED",
    }


def test_semantic_contrast_prompt_is_compact_and_predicate_blind() -> None:
    alignment = json.dumps({
        "hypothesis_id": "hypothesis-1",
        "source_context": {
            "start": 0,
            "end": 33,
            "text": "Endpoint A transfers Endpoint B.",
        },
        "endpoint_a": {"surface": "Endpoint A", "span": [0, 10]},
        "endpoint_b": {"surface": "Endpoint B", "span": [21, 31]},
        "required_relation_semantics": {
            "semantic_definition": "The source creates the target.",
            "direction_description": "From creator to output.",
            "positive_examples": ["A creates B."],
            "negative_examples": ["A transfers B."],
        },
        "comparison_questions": {"should_not_be_copied": True},
    })

    payload = json.loads(verifier_semantic_contrast_prompt(alignment))

    assert set(payload) == {
        "instruction", "contract_version", "source_context", "endpoint_a",
        "endpoint_b", "required_relation_semantics",
    }
    assert "hypothesis-1" not in json.dumps(payload)
    assert parse_verifier_semantic_contrast({
        "semantic_contrast": "negative example",
    }) == "NEGATIVE_EXAMPLE"


def test_exact_argument_prompt_binds_source_and_exact_endpoint_surfaces() -> None:
    content = "Factory Alpha manufactures Device Beta."
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        [
            _mention("alpha", "Alpha", content.index("Alpha")),
            _mention("beta", "Beta", content.index("Beta")),
        ],
        schema,
        predicate="alpha_rel",
        subject="occ-beta",
        object_="occ-alpha",
    )
    alignment = verifier_role_alignment_prompt(
        hypothesis,
        logical_schema=schema,
        blind_role_audit={
            "subject_role": "created output",
            "object_role": "creator",
            "expressed_relation": "Alpha creates Beta.",
        },
        opportunity=pair,
        content=content,
    )

    payload = json.loads(verifier_exact_argument_prompt(
        alignment, direction="FORWARD",
    ))

    assert payload["source_text"] == content
    assert payload["first_endpoint"]["surface"] == "Beta"
    assert payload["second_endpoint"]["surface"] == "Alpha"
    assert payload["first_endpoint"]["context_span"] == [
        content.index("Beta"), content.index("Beta") + len("Beta")
    ]
    assert payload["first_endpoint"]["required_role"]
    assert payload["first_endpoint"]["contextual_role"] == "created output"
    assert payload["second_endpoint"]["contextual_role"] == "creator"
    assert "alpha_rel" not in json.dumps(payload)
    assert verifier_exact_argument_response_schema()["required"] == [
        "direction_supported",
    ]

    reverse = json.loads(verifier_exact_argument_prompt(
        alignment, direction="REVERSE",
    ))
    assert reverse["first_endpoint"]["surface"] == "Alpha"
    assert reverse["second_endpoint"]["surface"] == "Beta"
    assert reverse["first_endpoint"]["contextual_role"] == "creator"
    assert reverse["second_endpoint"]["contextual_role"] == "created output"


def test_exact_argument_requires_predicate_entailment_not_only_role_shape() -> None:
    assert parse_verifier_exact_argument({
        "a_matches_source": True,
        "b_matches_target": True,
        "b_matches_source": False,
        "a_matches_target": False,
        "forward_predicate_entailed": False,
        "reverse_predicate_entailed": False,
        "orientation": "NEITHER",
    }) == "NEITHER"

    with pytest.raises(RelationClassificationError):
        parse_verifier_exact_argument({
            "a_is_complete_argument_mention": False,
            "b_is_complete_argument_mention": True,
            "a_matches_source": True,
            "b_matches_target": True,
            "b_matches_source": False,
            "a_matches_target": False,
            "forward_predicate_entailed": True,
            "reverse_predicate_entailed": False,
            "orientation": "FORWARD",
        })


def test_chinese_exact_argument_prompt_compares_roles_not_field_order() -> None:
    content = "装置由部件甲、部件乙和控制器组成。"
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        [
            _mention("first", "部件甲", content.index("部件甲")),
            _mention("second", "部件乙", content.index("部件乙")),
        ],
        schema,
        predicate="alpha_rel",
        subject="occ-first",
        object_="occ-second",
    )
    alignment = verifier_role_alignment_prompt(
        hypothesis,
        logical_schema=schema,
        blind_role_audit={
            "subject_role": "并列成员",
            "object_role": "并列成员",
            "expressed_relation": "两个端点属于同一列举。",
        },
        opportunity=pair,
        content=content,
    )

    prompt = json.loads(verifier_exact_argument_prompt(alignment))
    assert "分别比较每个端点" in prompt["instruction"]
    assert "不要按词序或字段位置匹配" in prompt["instruction"]


def test_exact_argument_wire_contract_is_one_boolean_entailment_field() -> None:
    schema = verifier_exact_argument_response_schema()

    assert schema["required"] == ["direction_supported"]
    assert set(schema["properties"]) == {"direction_supported"}
    assert parse_verifier_directional_entailment({
        "direction_supported": True,
    }) is True
    assert parse_verifier_directional_entailment({
        "direction_supported": False,
    }) is False
    # Frozen scripted fixtures retain their former three-boolean shape.
    assert parse_verifier_directional_entailment({
        "source_role": True,
        "target_role": True,
        "direct_relation": True,
    }) is True
    assert parse_verifier_directional_entailment({
        "source_role": True,
        "target_role": False,
        "direct_relation": True,
    }) is False
    assert parse_verifier_directional_entailment({
        "source_role_satisfied": True,
        "target_role_satisfied": True,
        "entailment": True,
    }) is True

    with pytest.raises(RelationClassificationError):
        parse_verifier_directional_entailment({"direct_relation": True})


def test_exact_argument_orientation_remains_independent_of_role_alignment() -> None:
    reversed_result = parse_verifier_role_alignment({
        "forward_alignment": "ALIGNED",
        "reverse_alignment": "MISALIGNED",
        "semantic_contrast": "DEFINITION",
        "exact_argument_orientation": "REVERSE",
    })
    unrelated_result = parse_verifier_role_alignment({
        "forward_alignment": "ALIGNED",
        "reverse_alignment": "MISALIGNED",
        "semantic_contrast": "DEFINITION",
        "exact_argument_orientation": "NEITHER",
    })

    assert reversed_result == {
        "forward_alignment": "ALIGNED",
        "reverse_alignment": "MISALIGNED",
    }
    assert unrelated_result == reversed_result


def test_exact_argument_conflict_is_a_publication_veto_after_final_verification() -> None:
    content = "Supplier Alpha delivers equipment to Customer Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=lambda *_args: {
            "subject_role": "supplier",
            "object_role": "recipient",
        },
        role_alignment_probe=lambda *_args: {
            "alignment": "UNCERTAIN",
            "exact_argument_orientation": "FORWARD",
        },
        verifier=lambda *_args: {
            "verdict": REJECTED,
            "reason_code": "ORDERED_FACT_NOT_SUPPORTED",
            "forward_entailment": "NOT_ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": False,
            "object_role_supported": False,
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "ORDERED_FACT_NOT_SUPPORTED"
    assert attempt.subject_context_role == "supplier"
    assert attempt.object_context_role == "recipient"
    assert attempt.subject_role_alignment == "UNCERTAIN"
    assert attempt.object_role_alignment == "UNCERTAIN"
    assert attempt.direction_normalization == "FORWARD"


def test_exact_claim_can_publish_with_exact_signal_when_semantic_alignment_is_uncertain() -> None:
    content = "System Alpha explicitly connects Platform Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=lambda *_args: {
            "subject_role": "participant",
            "object_role": "participant",
        },
        role_alignment_probe=lambda *_args: {
            "alignment": "UNCERTAIN",
            "exact_argument_orientation": "FORWARD",
        },
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.reason_code == "ORDERED_FACT_SUPPORTED"
    assert attempt.forward_role_alignment == "UNCERTAIN"
    assert attempt.direction_normalization == "FORWARD"


def test_semantic_role_signal_can_confirm_when_compact_exact_audit_misses() -> None:
    content = "System Alpha applies a component supplied by Vendor Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )
    final_calls = 0

    def final_verifier(*_args):
        nonlocal final_calls
        final_calls += 1
        return {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=lambda *_args: {
            "subject_role": "not independently preclassified",
            "object_role": "not independently preclassified",
        },
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "MISALIGNED",
            "exact_argument_orientation": "NEITHER",
        },
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.reason_code == "ORDERED_FACT_SUPPORTED"
    assert attempt.forward_role_alignment == "ALIGNED"
    assert attempt.supported_evidence_spans
    assert final_calls == 1


def test_two_independent_direction_conflicts_veto_publication() -> None:
    content = "System Alpha applies a component supplied by Vendor Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content, mentions, schema, predicate="alpha_rel",
        subject="occ-alpha", object_="occ-beta",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=lambda *_args: {
            "subject_role": "unrelated participant",
            "object_role": "unrelated participant",
        },
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "MISALIGNED",
            "exact_argument_orientation": "NEITHER",
        },
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "BROAD_FALSE_POSITIVE",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "ROLE_ALIGNMENT_NOT_CONFIRMED"


def test_valid_expanded_argument_span_vetoes_shortened_endpoint() -> None:
    content = "Failure Alpha outage causes Beta interruption."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content, mentions, schema, predicate="alpha_rel",
        subject="occ-alpha", object_="occ-beta",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=lambda *_args: {
            "endpoint_a_complete_argument": False,
            "endpoint_b_complete_argument": False,
            "endpoint_a_argument_start": content.index("Alpha"),
            "endpoint_a_argument_end": content.index(" outage") + len(" outage"),
            "endpoint_b_argument_start": content.index("Beta"),
            "endpoint_b_argument_end": len(content) - 1,
            "direct_relation_present": True,
            "subject_role": "complete causing event",
            "object_role": "complete resulting event",
            "expressed_relation": "one event causes the other",
        },
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "MISALIGNED",
            "exact_argument_orientation": "FORWARD",
        },
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "MODEL_EXPANDED_ARGUMENTS",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "INCOMPLETE_ENDPOINT_ARGUMENT"


def test_auxiliary_probe_protocol_failure_does_not_skip_final_or_publish() -> None:
    content = "System Alpha explicitly connects Platform Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )
    final_calls = 0

    def broken_probe(*_args):
        raise RelationClassificationError("auxiliary response malformed")

    def final_verifier(*_args):
        nonlocal final_calls
        final_calls += 1
        return {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=broken_probe,
        role_alignment_probe=broken_probe,
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
    )

    assert attempt.verdict == UNCERTAIN
    assert attempt.reason_code == "AUXILIARY_ROLE_AUDIT_UNAVAILABLE"
    assert attempt.subject_context_role == "source endpoint A role unresolved"
    assert attempt.forward_role_alignment == "UNCERTAIN"
    assert final_calls == 1


def test_direct_proposer_can_retain_complete_final_verdict_when_auxiliary_is_unavailable() -> None:
    content = "System Alpha explicitly connects Platform Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, classified = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )
    hypothesis = replace(
        classified,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
        predicate_metadata={
            **classified.predicate_metadata,
            "candidate_source": "DIRECT_PROPOSER",
        },
    )

    def broken_probe(*_args):
        raise RelationClassificationError("auxiliary response malformed")

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=broken_probe,
        role_alignment_probe=broken_probe,
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        },
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.reason_code == (
        "SUPPORTED_DIRECT_PROPOSER_AUXILIARY_UNAVAILABLE_FALLBACK"
    )


def test_final_verifier_disagreement_gets_one_bounded_nonpublishing_retry() -> None:
    content = "System Alpha explicitly connects Platform Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content, mentions, schema, predicate="alpha_rel",
        subject="occ-alpha", object_="occ-beta",
    )

    def rejected(*_args):
        return {
            "verdict": REJECTED,
            "reason_code": "ORDERED_FACT_NOT_SUPPORTED",
            "forward_entailment": "NOT_ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": False,
            "object_role_supported": False,
            "supported_evidence_spans": [],
        }

    first = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=rejected,
        verifier_model_revision="verifier-model-r1",
        attempt_no=1,
    )
    terminal = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=rejected,
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
    )

    assert first.verdict == RETRYABLE_FAILED
    assert first.reason_code == "VERIFIER_AUXILIARY_DISAGREEMENT_RETRY"
    assert supported_for_normalization(first) is False
    assert terminal.verdict == REJECTED
    assert supported_for_normalization(terminal) is False


def test_authoritative_verifier_preserves_the_directed_hypothesis_claim() -> None:
    content = "Factory Alpha manufactures Device Beta."
    mentions = [
        _mention(
            "alpha", "Alpha", content.index("Alpha"),
            entity_type="Organization",
        ),
        _mention(
            "beta", "Beta", content.index("Beta"),
            entity_type="Product",
        ),
    ]
    schema = _schema(_predicate(
        "alpha_rel",
        domain=["Organization"],
        range_=["Product"],
    ))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-beta",
        object_="occ-alpha",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )

    def verifier(prompt, _schema):
        payload = json.loads(prompt)
        assert payload["ordered_claims"]["forward"] == {
            "subject_surface": "Beta",
            "relation_definition": "Fixture relation semantics.",
            "object_surface": "Alpha",
        }
        assert "alpha_rel" not in prompt
        return {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{"start": 0, "end": len(content)}],
        }

    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        role_probe=lambda *_args: {
            "subject_role": "manufacturer",
            "object_role": "manufactured output",
        },
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "MISALIGNED",
            "exact_argument_orientation": "FORWARD",
        },
        verifier=verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == PASS
    assert validation.type_orientation == "reverse"
    assert attempt.verdict == SUPPORTED
    assert attempt.direction_normalization == "FORWARD"
    assert attempt.normalized_subject_occurrence_key == "occ-beta"
    assert attempt.normalized_object_occurrence_key == "occ-alpha"

    normalized = normalize_verified_hypotheses(
        project_id=12,
        opportunities={pair.pair_identity: pair},
        hypotheses=[hypothesis],
        verification_attempts=[attempt],
        logical_schema=schema,
    )
    assert len(normalized.facts) == 1
    assert normalized.facts[0].subject_canonical_id == "entity-beta"
    assert normalized.facts[0].object_canonical_id == "entity-alpha"
    assert normalized.evidence[0].provenance["direction_normalization"] == (
        "FORWARD"
    )


def test_verification_attempt_migration_is_append_only_and_versioned() -> None:
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0015_shadow_relation_v2_verification"
    )

    assert "CREATE TABLE IF NOT EXISTS shadow_relation_verification_attempts" in migration.sql
    assert "UNIQUE(hypothesis_id,attempt_no)" in migration.sql
    assert "verifier_prompt_version" in migration.sql
    assert "verifier_model_revision" in migration.sql


def test_atomic_fact_inventory_is_predicate_blind_source_grounded_and_unordered() -> None:
    content = "Alpha uses Beta and Gamma."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, _hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )

    prompt = atomic_fact_inventory_prompt(pair, content=content)
    payload = json.loads(prompt)
    assert payload["context"]["text"] == content
    assert "endpoint" not in payload
    assert "alpha_rel" not in prompt
    assert atomic_fact_inventory_response_schema()["additionalProperties"] is False

    inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "items": [
            {
                "subject_surface": "Alpha",
                "relation_text": "uses",
                "object_surface": "Beta",
            },
            {
                "subject_surface": "Alpha",
                "relation_text": "uses",
                "object_surface": "Gamma",
            },
        ],
    }, context_text=content)
    assert atomic_fact_inventory_contains_pair(inventory, "Alpha", "Beta")
    assert atomic_fact_inventory_contains_pair(inventory, "Beta", "Alpha")
    assert not atomic_fact_inventory_contains_pair(inventory, "Beta", "Gamma")

    peer_inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": ["Alpha", "Beta", "Gamma"],
            "roles": ["whole", "part1", "part2"],
            "argument_kinds": ["ENTITY", "ENTITY", "ENTITY"],
            "entity_heads": ["Alpha", "Beta", "Gamma"],
            "assertion_status": "ASSERTED",
        }],
    }, context_text=content)
    assert atomic_fact_inventory_contains_pair(
        peer_inventory, "Alpha", "Beta",
    )
    assert not atomic_fact_inventory_contains_pair(
        peer_inventory, "Beta", "Gamma",
    )

    qualified_inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [{
            "relation": "uses",
            "arguments": ["Alpha", "qualified Beta"],
            "roles": ["actor", "resource"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Alpha", "Beta"],
            "assertion_status": "ASSERTED",
        }],
    }, context_text="Alpha uses qualified Beta.")
    assert atomic_fact_inventory_contains_pair(
        qualified_inventory, "Alpha", "Beta",
    )

    representative_inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [{
            "relation": "forms",
            "arguments": ["Committee", "Organization representatives"],
            "roles": ["whole", "member"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Committee", "representatives"],
            "assertion_status": "ASSERTED",
        }],
    }, context_text="Committee is formed by Organization representatives.")
    assert not atomic_fact_inventory_contains_pair(
        representative_inventory, "Committee", "Organization",
    )

    state_inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [{
            "relation": "causes",
            "arguments": ["Alpha", "Beta unavailable"],
            "roles": ["cause", "effect"],
            "argument_kinds": ["EVENT", "STATE"],
            "entity_heads": ["", ""],
            "assertion_status": "ASSERTED",
        }],
    }, context_text="Alpha causes Beta unavailable.")
    assert not atomic_fact_inventory_contains_pair(
        state_inventory, "Alpha", "Beta",
    )

    negated_inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [{
            "relation": "does not use",
            "arguments": ["Alpha", "Beta"],
            "roles": ["actor", "resource"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Alpha", "Beta"],
            "assertion_status": "NEGATED",
        }],
    }, context_text="Alpha does not use Beta.")
    assert not atomic_fact_inventory_contains_pair(
        negated_inventory, "Alpha", "Beta",
    )

    with pytest.raises(
        RelationClassificationError,
        match="not copied from source",
    ):
        parse_atomic_fact_inventory({
            "coverage_complete": True,
            "items": [{
                "subject_surface": "Alpha",
                "relation_text": "uses",
                "object_surface": "Invented",
            }],
        }, context_text=content)


def test_atomic_inventory_pads_omitted_trailing_non_entity_heads_only() -> None:
    inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [{
            "relation": "is used for",
            "arguments": ["The `if` statement", "conditional execution"],
            "roles": ["instrument", "purpose"],
            "argument_kinds": ["ENTITY", "EVENT"],
            "entity_heads": ["`if` statement"],
            "assertion_status": "ASSERTED",
        }],
    }, context_text="The `if` statement is used for conditional execution")
    assert inventory["frames"][0]["entity_heads"] == ["`if` statement", ""]

    with pytest.raises(RelationClassificationError, match="frame is incomplete"):
        parse_atomic_fact_inventory({
            "coverage_complete": True,
            "frames": [{
                "relation": "uses",
                "arguments": ["Alpha", "Beta"],
                "roles": ["actor", "resource"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Alpha"],
                "assertion_status": "ASSERTED",
            }],
        }, context_text="Alpha uses Beta")


def test_atomic_inventory_ignores_unary_frames_and_non_entity_heads() -> None:
    inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [
            {
                "relation": "begins",
                "arguments": ["Alice"],
                "roles": ["experiencer"],
                "argument_kinds": ["ENTITY"],
                "entity_heads": ["Alice"],
                "assertion_status": "ASSERTED",
            },
            {
                "relation": "establishes",
                "arguments": ["An HTTP client", "a connection", "to an HTTP server"],
                "roles": ["agent", "theme", "goal"],
                "argument_kinds": ["ENTITY", "EVENT", "ENTITY"],
                "entity_heads": ["client", "connection", "server"],
                "assertion_status": "ASSERTED",
            },
        ],
    }, context_text="An HTTP client establishes a connection to an HTTP server.")
    assert len(inventory["frames"]) == 1
    assert inventory["frames"][0]["entity_heads"] == [
        "client", "", "server",
    ]


def test_atomic_inventory_accepts_casefolded_source_copies_only() -> None:
    inventory = parse_atomic_fact_inventory({
        "coverage_complete": True,
        "frames": [{
            "relation": "uses",
            "arguments": ["the http client", "a connection"],
            "roles": ["actor", "theme"],
            "argument_kinds": ["ENTITY", "EVENT"],
            "entity_heads": ["CLIENT", ""],
            "assertion_status": "ASSERTED",
        }],
    }, context_text="The HTTP client establishes a connection.")
    assert inventory["frames"][0]["arguments"] == [
        "the http client", "a connection",
    ]
    assert inventory["frames"][0]["entity_heads"] == ["CLIENT", ""]


def test_atomic_inventory_does_not_match_entity_head_inside_word() -> None:
    with pytest.raises(
        RelationClassificationError,
        match="entity head is not copied from its argument",
    ):
        parse_atomic_fact_inventory({
            "coverage_complete": True,
            "frames": [{
                "relation": "uses",
                "arguments": ["Alpha", "gift"],
                "roles": ["actor", "theme"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Alpha", "if"],
                "assertion_status": "ASSERTED",
            }],
        }, context_text="Alpha uses gift.")


def test_atomic_frame_schema_alignment_maps_roles_and_is_predicate_blind() -> None:
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": ["Factory Base", "Assembly Workshop"],
            "roles": ["whole", "part"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Factory Base", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    metadata["subject_role_description"] = (
        "the component contained by a whole"
    )
    metadata["object_role_description"] = (
        "the whole containing the component"
    )
    schema = _schema(metadata)
    prompt = atomic_frame_schema_alignment_prompt(
        inventory,
        subject_surface="Factory Base",
        object_surface="Assembly Workshop",
        subject_type="Location",
        object_type="Organization",
        predicate="alpha_rel",
        logical_schema=schema,
    )
    assert "alpha_rel" not in prompt
    assert "candidate_subject" not in json.loads(prompt)
    assert "candidate_object" not in json.loads(prompt)
    assert json.loads(prompt)["semantic_frames"][0]["entity_types"] == [
        "Location", "Organization",
    ]
    assert atomic_frame_schema_alignment_response_schema()["required"] == [
        "coverage_complete", "alignments",
    ]
    raw = {
        "coverage_complete": True,
        "alignments": [{
            "frame_index": 0,
            "subject_index": 1,
            "object_index": 0,
        }],
    }
    assert parse_atomic_frame_schema_alignment(
        raw,
        inventory=inventory,
        subject_surface="Factory Base",
        object_surface="Assembly Workshop",
    ) == "REVERSED"
    assert parse_atomic_frame_schema_alignment(
        raw,
        inventory=inventory,
        subject_surface="Assembly Workshop",
        object_surface="Factory Base",
    ) == "ALIGNED"
    assert parse_atomic_frame_schema_alignment(
        {
            "coverage_complete": True,
            "alignments": [],
        },
        inventory=inventory,
        subject_surface="Assembly Workshop",
        object_surface="Factory Base",
    ) == "NOT_ALIGNED"

    renamed_metadata = _predicate(
        "renamed_relation",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    renamed_metadata["subject_role_description"] = (
        "the component contained by a whole"
    )
    renamed_metadata["object_role_description"] = (
        "the whole containing the component"
    )
    renamed = _schema(renamed_metadata)
    renamed_prompt = atomic_frame_schema_alignment_prompt(
        inventory,
        subject_surface="Factory Base",
        object_surface="Assembly Workshop",
        subject_type="Location",
        object_type="Organization",
        predicate="renamed_relation",
        logical_schema=renamed,
    )
    assert json.loads(renamed_prompt) == json.loads(prompt)


def test_authoritative_verifier_is_not_pre_reversed_by_positional_audit() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    metadata["subject_role_description"] = "component"
    metadata["object_role_description"] = "containing whole"
    schema = _schema(metadata)
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-workshop",
        object_="occ-base",
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": ["Factory Base", "Assembly Workshop"],
            "roles": ["whole", "part"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Factory Base", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }

    def final_verifier(prompt, *_args):
        ordered = json.loads(prompt)["ordered_claims"]
        assert ordered["forward"]["subject_surface"] == "Assembly Workshop"
        assert ordered["forward"]["object_surface"] == "Factory Base"
        return {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [{
                "frame_index": 0,
                "subject_index": 2,
                "object_index": 0,
            }],
        },
        role_probe=_role_probe,
        # Deliberately wrong positional signal: it must remain non-binding.
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "MISALIGNED",
            "exact_argument_orientation": "FORWARD",
        },
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.normalized_subject_occurrence_key == "occ-workshop"
    assert attempt.normalized_object_occurrence_key == "occ-base"
    assert attempt.direction_normalization == "FORWARD"


def test_conflicting_frame_alignment_does_not_reverse_role_confirmed_claim() -> None:
    content = "Assembly Workshop uses Force Sensor."
    mentions = [
        _mention("workshop", "Assembly Workshop", content.index("Assembly")),
        _mention("sensor", "Force Sensor", content.index("Force")),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source actively uses the target as a resource.",
        direction_description="Orient from the active user to the resource.",
    )
    metadata["subject_role_description"] = "active user"
    metadata["object_role_description"] = "used resource"
    schema = _schema(metadata)
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-workshop",
        object_="occ-sensor",
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            # Deliberately reversed argument order: this is an auxiliary
            # frame signal and must not override an independently confirmed
            # directed verifier claim.
            "relation": "uses",
            "arguments": ["Force Sensor", "Assembly Workshop"],
            "roles": ["resource", "actor"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Force Sensor", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }
    verifier_calls = []

    def final_verifier(prompt, *_args):
        verifier_calls.append(json.loads(prompt)["ordered_claims"])
        return _supported(hypothesis)

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [{
                "frame_index": 0,
                "subject_index": 1,
                "object_index": 0,
            }],
        },
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.normalized_subject_occurrence_key == "occ-workshop"
    assert attempt.normalized_object_occurrence_key == "occ-sensor"
    assert attempt.direction_normalization == "FORWARD"
    assert len(verifier_calls) == 1


def test_direct_proposer_multi_argument_frame_checks_swapped_direction() -> None:
    content = (
        "Factory Base contains Robot Workshop, Storage Workshop, and Quality Center."
    )
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention("storage", "Storage Workshop", content.index("Storage Workshop")),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    metadata["subject_role_description"] = "component"
    metadata["object_role_description"] = "containing whole"
    schema = _schema(metadata)
    pair, classified = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-storage",
    )
    hypothesis = replace(
        classified,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": [
                "Factory Base", "Robot Workshop", "Storage Workshop",
                "Quality Center",
            ],
            "roles": ["whole", "part", "part", "part"],
            "argument_kinds": ["ENTITY", "ENTITY", "ENTITY", "ENTITY"],
            "entity_heads": [
                "Factory Base", "Robot Workshop", "Storage Workshop",
                "Quality Center",
            ],
            "assertion_status": "ASSERTED",
        }],
    }
    verifier_claims = []

    def final_verifier(prompt, *_args):
        ordered = json.loads(prompt)["ordered_claims"]
        verifier_claims.append((
            ordered["forward"]["subject_surface"],
            ordered["forward"]["object_surface"],
        ))
        return _supported(hypothesis)

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [{
                "frame_index": 0,
                "subject_index": 2,
                "object_index": 0,
            }],
        },
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.normalized_subject_occurrence_key == "occ-storage"
    assert attempt.normalized_object_occurrence_key == "occ-base"
    assert attempt.direction_normalization == "REVERSED"
    assert verifier_claims == [
        ("Factory Base", "Storage Workshop"),
        ("Storage Workshop", "Factory Base"),
    ]


def test_direct_proposer_corrects_simple_reversed_frame_binding() -> None:
    content = "Assembly Workshop uses Force Sensor."
    mentions = [
        _mention("workshop", "Assembly Workshop", content.index("Assembly")),
        _mention("sensor", "Force Sensor", content.index("Force")),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source actively uses the target as a resource.",
        direction_description="Orient from the active user to the resource.",
    )
    metadata["subject_role_description"] = "active user"
    metadata["object_role_description"] = "used resource"
    schema = _schema(metadata)
    pair, classified = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-sensor",
        object_="occ-workshop",
    )
    hypothesis = replace(
        classified,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "uses",
            "arguments": ["Assembly Workshop", "Force Sensor"],
            "roles": ["actor", "resource"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Assembly Workshop", "Force Sensor"],
            "assertion_status": "ASSERTED",
        }],
    }
    verifier_claims = []

    def final_verifier(prompt, *_args):
        ordered = json.loads(prompt)["ordered_claims"]
        verifier_claims.append((
            ordered["forward"]["subject_surface"],
            ordered["forward"]["object_surface"],
        ))
        return _supported(hypothesis)

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [{
                "frame_index": 0,
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        role_probe=_role_probe,
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "ALIGNED",
            "exact_argument_orientation": "REVERSE",
        },
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.normalized_subject_occurrence_key == "occ-workshop"
    assert attempt.normalized_object_occurrence_key == "occ-sensor"
    assert verifier_claims == [
        ("Force Sensor", "Assembly Workshop"),
        ("Assembly Workshop", "Force Sensor"),
    ]


def test_direct_proposer_keeps_exact_verifier_fact_when_inventory_misses_pair() -> None:
    content = "Lab depends on Research Institute providing capability."
    mentions = [
        _mention("lab", "Lab", content.index("Lab")),
        _mention(
            "institute", "Research Institute", content.index("Research Institute"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, classified = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-lab",
        object_="occ-institute",
    )
    hypothesis = replace(
        classified,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    )
    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: {
            "coverage_complete": True,
            "frames": [{
                "relation": "provides",
                "arguments": ["Lab", "capability"],
                "roles": ["requester", "resource"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Lab", "capability"],
                "assertion_status": "ASSERTED",
            }],
        },
        role_probe=lambda *_args: {
            "endpoint_a_complete_argument": True,
            "endpoint_b_complete_argument": True,
            "endpoint_a_argument_start": 0,
            "endpoint_a_argument_end": 3,
            "endpoint_b_argument_start": 10,
            "endpoint_b_argument_end": 27,
            "direct_relation_present": True,
            "subject_role": "source endpoint role",
            "object_role": "target endpoint role",
            "expressed_relation": "the source directly relates to the target",
        },
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "ALIGNED",
            "exact_argument_orientation": "REVERSE",
        },
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.supported_evidence_spans


def test_direct_proposer_multi_argument_rejects_unconfirmed_direction_conflict() -> None:
    content = "Factory Base contains Robot Workshop, Storage Workshop, and Quality Center."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention("storage", "Storage Workshop", content.index("Storage Workshop")),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    metadata["subject_role_description"] = "component"
    metadata["object_role_description"] = "containing whole"
    schema = _schema(metadata)
    pair, classified = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-storage",
    )
    hypothesis = replace(
        classified,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": [
                "Factory Base", "Robot Workshop", "Storage Workshop",
                "Quality Center",
            ],
            "roles": ["whole", "part", "part", "part"],
            "argument_kinds": ["ENTITY"] * 4,
            "entity_heads": [
                "Factory Base", "Robot Workshop", "Storage Workshop",
                "Quality Center",
            ],
            "assertion_status": "ASSERTED",
        }],
    }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [],
        },
        role_probe=_role_probe,
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "ALIGNED",
            "exact_argument_orientation": "REVERSE",
        },
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "EXACT_ARGUMENT_DIRECTION_CONFLICT"


def test_direct_proposer_rejects_direct_pair_when_direction_audit_conflicts() -> None:
    content = "Assembly Workshop uses Force Sensor."
    mentions = [
        _mention("workshop", "Assembly Workshop", content.index("Assembly")),
        _mention("sensor", "Force Sensor", content.index("Force")),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source actively uses the target as a resource.",
        direction_description="Orient from the active user to the resource.",
    )
    metadata["subject_role_description"] = "active user"
    metadata["object_role_description"] = "used resource"
    schema = _schema(metadata)
    pair, classified = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-workshop",
        object_="occ-sensor",
    )
    hypothesis = replace(
        classified,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "uses",
            "arguments": ["Assembly Workshop", "Force Sensor"],
            "roles": ["actor", "resource"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Assembly Workshop", "Force Sensor"],
            "assertion_status": "ASSERTED",
        }],
    }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        role_probe=_role_probe,
        # Deliberately contradictory auxiliary output. The ordered verifier
        # remains authoritative when the direct proposer supplied exact
        # endpoints and independently supported evidence.
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "ALIGNED",
            "exact_argument_orientation": "REVERSE",
        },
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "EXACT_ARGUMENT_DIRECTION_CONFLICT"


def test_direct_proposer_rejects_reverse_pair_when_source_frame_order_conflicts() -> None:
    content = "Assembly Workshop uses Force Sensor."
    mentions = [
        _mention("workshop", "Assembly Workshop", content.index("Assembly")),
        _mention("sensor", "Force Sensor", content.index("Force")),
    ]
    schema = _schema(_predicate(
        "alpha_rel",
        semantic_definition="The source actively uses the target as a resource.",
        direction_description="Orient from the active user to the resource.",
    ))
    pair, classified = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-workshop",
        object_="occ-sensor",
    )
    hypothesis = replace(
        classified,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    )
    reverse_inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "uses",
            "arguments": ["Force Sensor", "Assembly Workshop"],
            "roles": ["resource", "actor"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Force Sensor", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: reverse_inventory,
        role_probe=_role_probe,
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "ALIGNED",
            "exact_argument_orientation": "REVERSE",
        },
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
        attempt_no=Config.relation_max_attempts,
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "EXACT_ARGUMENT_DIRECTION_CONFLICT"


def test_supported_claim_is_vetoed_by_frame_schema_relation_mismatch() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    metadata["subject_role_description"] = "component"
    metadata["object_role_description"] = "containing whole"
    schema = _schema(metadata)
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": ["Factory Base", "Assembly Workshop"],
            "roles": ["whole", "part"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Factory Base", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [],
        },
        role_probe=_role_probe,
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "ALIGNED",
            "exact_argument_orientation": "REVERSE",
        },
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "ATOMIC_FRAME_RELATION_MISMATCH"


def test_reversed_frame_binding_requires_supported_corrected_verifier() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    metadata["subject_role_description"] = "component"
    metadata["object_role_description"] = "containing whole"
    schema = _schema(metadata)
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": ["Factory Base", "Assembly Workshop"],
            "roles": ["whole", "part"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Factory Base", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }
    ordered_claims = []

    def verifier(prompt, *_args):
        ordered = json.loads(prompt)["ordered_claims"]
        ordered_claims.append((
            ordered["forward"]["subject_surface"],
            ordered["forward"]["object_surface"],
        ))
        return {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [{
                "frame_index": 0,
                "subject_index": 1,
                "object_index": 0,
            }],
        },
        role_probe=_role_probe,
        role_alignment_probe=_uncertain_role_alignment_probe,
        verifier=verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert ordered_claims == [
        ("Factory Base", "Assembly Workshop"),
        ("Assembly Workshop", "Factory Base"),
    ]
    assert attempt.verdict == SUPPORTED
    assert attempt.normalized_subject_occurrence_key == "occ-workshop"
    assert attempt.normalized_object_occurrence_key == "occ-base"


def test_reversed_frame_binding_does_not_publish_when_recheck_rejects() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    metadata = _predicate(
        "alpha_rel",
        semantic_definition="The source is a component of the target.",
        direction_description="Orient from component to containing whole.",
    )
    metadata["subject_role_description"] = "component"
    metadata["object_role_description"] = "containing whole"
    schema = _schema(metadata)
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    calls = 0

    def verifier(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "verdict": SUPPORTED,
                "reason_code": "ORDERED_FACT_SUPPORTED",
                "forward_entailment": "ENTAILED",
                "reverse_entailment": "NOT_ENTAILED",
                "subject_role_supported": True,
                "object_role_supported": True,
                "supported_evidence_spans": [
                    {"start": 0, "end": len(content)},
                ],
            }
        return {
            "verdict": REJECTED,
            "reason_code": "ORDERED_FACT_REJECTED",
            "forward_entailment": "NOT_ENTAILED",
            "reverse_entailment": "ENTAILED",
            "subject_role_supported": False,
            "object_role_supported": False,
            "supported_evidence_spans": [],
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: {
            "coverage_complete": True,
            "frames": [{
                "relation": "contains",
                "arguments": ["Factory Base", "Assembly Workshop"],
                "roles": ["whole", "part"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Factory Base", "Assembly Workshop"],
                "assertion_status": "ASSERTED",
            }],
        },
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [{
                "frame_index": 0,
                "subject_index": 1,
                "object_index": 0,
            }],
        },
        role_probe=_role_probe,
        role_alignment_probe=_uncertain_role_alignment_probe,
        verifier=verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert calls == 2
    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "ATOMIC_FRAME_DIRECTION_CONFLICT"


def test_complete_atomic_inventory_vetoes_peer_pair_after_supported_verifier() -> None:
    content = "Cabinet contains Alpha, Beta, and Gamma."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )
    final_calls = 0

    def final_verifier(*_args):
        nonlocal final_calls
        final_calls += 1
        return {
            "verdict": SUPPORTED,
            "reason_code": "ORDERED_FACT_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{"start": 0, "end": len(content)}],
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: {
            "coverage_complete": True,
            "items": [
                {
                    "subject_surface": "Cabinet",
                    "relation_text": "contains",
                    "object_surface": "Alpha",
                },
                {
                    "subject_surface": "Cabinet",
                    "relation_text": "contains",
                    "object_surface": "Beta",
                },
            ],
        },
        role_probe=lambda *_args: {
            "subject_role": "first peer",
            "object_role": "second peer",
        },
        role_alignment_probe=lambda *_args: {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "MISALIGNED",
            "exact_argument_orientation": "FORWARD",
        },
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert final_calls == 1
    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "NO_DIRECT_ATOMIC_PAIR"


def test_incomplete_atomic_inventory_is_auxiliary_and_final_verifier_runs() -> None:
    content = "Alpha connects Beta."
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha")),
        _mention("beta", "Beta", content.index("Beta")),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-alpha",
        object_="occ-beta",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    final_calls = 0

    def final_verifier(*_args):
        nonlocal final_calls
        final_calls += 1
        return _supported(hypothesis)

    common = dict(
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: {
            "coverage_complete": False,
            "items": [],
        },
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )
    retry = verify_hypothesis(
        hypothesis, validation, pair, attempt_no=1, **common,
    )
    terminal = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        attempt_no=Config.relation_max_attempts,
        **common,
    )

    assert retry.verdict in {RETRYABLE_FAILED, SUPPORTED}
    assert terminal.verdict == SUPPORTED
    assert final_calls >= 1


def test_hierarchical_list_structure_uses_structural_evidence_path() -> None:
    content = "System Alpha\n- Child Component"
    mentions = [
        _mention("system", "System Alpha", 0),
        _mention("child", "Child Component", content.index("Child")),
    ]
    for mention, scope_id, parent_scope_id in (
        (mentions[0], "list:1:0", ""),
        (mentions[1], "list:1:1", "list:1:0"),
    ):
        mention.update({
            "source_role": "list_item",
            "list_group_id": "list:1",
            "scope_id": scope_id,
            "list_parent_scope_id": parent_scope_id,
            "list_depth": 0 if not parent_scope_id else 1,
            "relation_eligible": True,
        })
    schema = _schema(_predicate(
        "part_of",
        structural_evidence_kinds=["list_parent_child"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="part_of",
        subject="occ-child",
        object_="occ-system",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    atomic_calls = 0
    final_calls = 0

    def atomic_probe(*_args):
        nonlocal atomic_calls
        atomic_calls += 1
        raise AssertionError("list structure must not require sentence inventory")

    def final_verifier(*_args, **_kwargs):
        nonlocal final_calls
        final_calls += 1
        return _supported(hypothesis)

    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=atomic_probe,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=final_verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == PASS
    assert attempt.verdict == SUPPORTED
    assert atomic_calls == 0
    assert final_calls == 1


def test_rejected_structural_edge_gets_narrow_adjudication_retry() -> None:
    content = "Preamble.\nSystem Alpha\n- Child Component"
    mentions = [
        _mention("system", "System Alpha", content.index("System")),
        _mention("child", "Child Component", content.index("Child")),
    ]
    for mention, scope_id, parent_scope_id in (
        (mentions[0], "list:1:0", ""),
        (mentions[1], "list:1:1", "list:1:0"),
    ):
        mention.update({
            "source_role": "list_item",
            "list_group_id": "list:1",
            "scope_id": scope_id,
            "list_parent_scope_id": parent_scope_id,
            "list_depth": 0 if not parent_scope_id else 1,
            "relation_eligible": True,
        })
    schema = _schema(_predicate(
        "part_of",
        structural_evidence_kinds=["list_parent_child"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="part_of",
        subject="occ-child",
        object_="occ-system",
    )
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    calls = []

    def verifier(prompt, _schema):
        payload = json.loads(prompt)
        calls.append(payload)
        if len(calls) == 1:
            return {
                "hypothesis_id": hypothesis.hypothesis_id,
                "verdict": "REJECTED",
                "reason_code": "ORDERED_ROLE_NOT_ENTAILED",
                "forward_entailment": "NOT_ENTAILED",
                "reverse_entailment": "UNCERTAIN",
                "subject_role_supported": False,
                "object_role_supported": False,
                "supported_evidence_spans": [],
            }
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "decision": SUPPORTED,
            "binding": "CHILD_TO_PARENT",
            "schema_compatible": True,
            "evidence_spans": [
                # The compact structural protocol may return offsets relative
                # to the displayed frozen context rather than document text.
                {
                    "span": [0, pair.context_span[1] - pair.context_span[0]],
                    "text": content[pair.context_span[0]:pair.context_span[1]],
                },
            ],
            "reason_code": "STRUCTURAL_CHILD_PARENT_CONFIRMED",
        }

    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: pytest.fail(
            "list structure must not require sentence inventory"
        ),
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=verifier,
        verifier_model_revision="verifier-model-r1",
    )

    assert validation.status == PASS
    assert attempt.verdict == SUPPORTED
    assert len(calls) == 2
    assert calls[1]["instruction"].startswith(
        "This is an independent structural-evidence adjudication."
    )
    assert calls[1]["structural_binding"]["child_occurrence_key"] == (
        "occ-child"
    )
    assert calls[1]["structural_binding"]["parent_occurrence_key"] == (
        "occ-system"
    )
    assert calls[1]["adjudication_mode"] == "STRUCTURAL_SOURCE_ASSERTION"


def test_section_label_structure_requires_child_line_anchor() -> None:
    content = (
        "Requirements\n"
        "SR-01 The service shall check a flight intent.\n"
        "SR-02 The USS shall submit an operation intent."
    )
    mentions = [
        _mention("requirements", "Requirements", 0, endpoint_candidate=True),
        _mention(
            "sr01", "SR-01", content.index("SR-01"), endpoint_candidate=True,
        ),
        _mention("sr02", "SR-02", content.index("SR-02"), endpoint_candidate=True),
        _mention("uss", "USS", content.index("USS"), endpoint_candidate=True),
    ]
    mentions[0].update({
        "source_role": "body", "scope_id": "scope-heading",
        "scope_kind": "sentence", "scope_start": 0, "scope_end": 13,
    })
    mentions[1].update({
        "source_role": "body", "scope_id": "scope-sr01",
        "scope_kind": "sentence", "scope_start": 13, "scope_end": 60,
    })
    mentions[2].update({
        "source_role": "body", "scope_id": "scope-sr02",
        "scope_kind": "sentence", "scope_start": 60, "scope_end": len(content),
    })
    mentions[3].update({
        "source_role": "body", "scope_id": "scope-sr02",
        "scope_kind": "sentence", "scope_start": 60, "scope_end": len(content),
    })
    schema = _schema(_predicate(
        "part_of",
        structural_evidence_kinds=["section_label_parent_child"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    plan = plan_pair_opportunities(
        project_id=12,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-1",
        candidate_policy_version="candidate-v2",
    )
    pairs = {
        frozenset((item.left_occurrence_key, item.right_occurrence_key)): item
        for item in plan.opportunities
    }
    requirements = "occ-requirements"
    sr02 = "occ-sr02"
    uss = "occ-uss"
    assert pairs[frozenset((requirements, sr02))].provenance[
        "hierarchical_section_label_parent_child"
    ] is True
    assert pairs[frozenset((requirements, uss))].provenance[
        "hierarchical_section_label_parent_child"
    ] is False


def test_direct_atomic_pair_recovers_missing_final_evidence() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": ["Factory Base", "Assembly Workshop"],
            "roles": ["whole", "part"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Factory Base", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": True,
            "alignments": [{
                "frame_index": 0,
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "DIRECT_ASSERTION",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.supported_evidence_spans == (
        EvidenceSpan(start=0, end=len(content)),
    )


def test_validated_classifier_evidence_recovers_missing_final_evidence() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
        spans=[(0, len(content))],
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "DIRECT_ASSERTION",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.supported_evidence_spans == (
        EvidenceSpan(start=0, end=len(content)),
    )


def test_exact_direction_audit_recovers_omitted_final_entailment_fields() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    response = _supported(hypothesis)
    response.pop("forward_entailment")
    response.pop("reverse_entailment")

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.forward_entailment == "ENTAILED"
    assert attempt.reverse_entailment == "NOT_ENTAILED"


def test_legacy_entailment_and_evidence_aliases_use_independent_orientation() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "verdict": "SUPPORTED",
            "entailment": "ENTAILED",
            "evidence_spans": [
                {"start": 0, "end": len(content)},
            ],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.forward_entailment == "ENTAILED"
    assert attempt.reverse_entailment == "NOT_ENTAILED"
    assert attempt.supported_evidence_spans


def test_exact_direction_audit_recovers_omitted_final_role_fields() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    response = _supported(hypothesis)
    response.pop("subject_role_supported")
    response.pop("object_role_supported")

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.subject_role_supported is True
    assert attempt.object_role_supported is True


def test_role_alignment_recovers_omitted_final_role_fields_without_exact_flag() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    response = _supported(hypothesis)
    response.pop("subject_role_supported")
    response.pop("object_role_supported")

    def role_alignment_without_exact_flag(*_args):
        return {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "MISALIGNED",
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=role_alignment_without_exact_flag,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED


def test_role_alignment_recovers_omitted_final_entailment_fields_without_exact_flag() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    response = _supported(hypothesis)
    response.pop("forward_entailment")
    response.pop("reverse_entailment")

    def role_alignment_without_exact_flag(*_args):
        return {
            "forward_alignment": "ALIGNED",
            "reverse_alignment": "MISALIGNED",
        }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=role_alignment_without_exact_flag,
        verifier=lambda *_args: response,
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED
    assert attempt.forward_entailment == "ENTAILED"
    assert attempt.reverse_entailment == "NOT_ENTAILED"


def test_atomic_inventory_protocol_failure_is_auxiliary_only() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: {
            "coverage_complete": True,
            "frames": [{"malformed": True}],
        },
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: _supported(hypothesis),
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED


def test_direct_atomic_pair_survives_auxiliary_frame_protocol_failure() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )
    inventory = {
        "coverage_complete": True,
        "frames": [{
            "relation": "contains",
            "arguments": ["Factory Base", "Assembly Workshop"],
            "roles": ["whole", "part"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Factory Base", "Assembly Workshop"],
            "assertion_status": "ASSERTED",
        }],
    }

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        atomic_fact_probe=lambda *_args: inventory,
        atomic_frame_alignment_probe=lambda *_args: {
            "coverage_complete": False,
            "alignments": [],
        },
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "DIRECT_ASSERTION",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{
                "start": 0, "end": len(content),
            }],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == SUPPORTED


def test_unconfirmed_reverse_final_verdict_is_not_materialized() -> None:
    content = "Factory Base contains Assembly Workshop."
    mentions = [
        _mention("base", "Factory Base", content.index("Factory Base")),
        _mention(
            "workshop", "Assembly Workshop",
            content.index("Assembly Workshop"),
        ),
    ]
    schema = _schema(_predicate("alpha_rel"))
    pair, hypothesis = _classified(
        content,
        mentions,
        schema,
        predicate="alpha_rel",
        subject="occ-base",
        object_="occ-workshop",
    )

    attempt = verify_hypothesis(
        hypothesis,
        validate_hypothesis(
            hypothesis, pair, content=content, logical_schema=schema,
        ),
        pair,
        content=content,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: {
            "verdict": SUPPORTED,
            "reason_code": "WRONG_DIRECTION",
            "forward_entailment": "NOT_ENTAILED",
            "reverse_entailment": "ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{
                "start": 0, "end": len(content),
            }],
        },
        verifier_model_revision="verifier-model-r1",
    )

    assert attempt.verdict == REJECTED
    assert attempt.reason_code == "DIRECTION_CORROBORATION_MISSING"
