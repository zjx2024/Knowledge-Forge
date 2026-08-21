from __future__ import annotations

import ast
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from knowledge_forge import relation_shadow_v2 as v2
from knowledge_forge import entity_shadow, shadow_context, span_entities
from knowledge_forge.relation_shadow_runtime import ShadowRelationV2Runtime
from knowledge_forge.schema_evolution import BASE_SCHEMA
from knowledge_forge.shadow_schema_overlay import apply_shadow_relation_schema_overlay
from knowledge_forge.service import KnowledgeForgeService
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


CONTENT = "Left interacts with Right."
MENTIONS = [
    {
        "mention_id": "mention-left",
        "occurrence_key": "occ-left",
        "surface": "Left",
        "mention_start": 0,
        "mention_end": 4,
        "canonical_id": "entity-z-left",
        "assignment_version": "assignment-v1",
        "type": "SourceType",
    },
    {
        "mention_id": "mention-right",
        "occurrence_key": "occ-right",
        "surface": "Right",
        "mention_start": 20,
        "mention_end": 25,
        "canonical_id": "entity-a-right",
        "assignment_version": "assignment-v1",
        "type": "TargetType",
    },
]


def _role_probe(*_args):
    return {
        "subject_role": "source endpoint role",
        "object_role": "target endpoint role",
    }


def _role_alignment_probe(prompt, *_args):
    payload = json.loads(prompt)
    symmetric = bool(payload.get("symmetric"))
    return {
        "forward_alignment": "ALIGNED",
        "reverse_alignment": "ALIGNED" if symmetric else "MISALIGNED",
        "exact_argument_orientation": "SYMMETRIC" if symmetric else "FORWARD",
    }


def _plan(schema: dict, *, content: str = CONTENT):
    return v2.plan_pair_opportunities(
        project_id=7,
        document_revision_id="revision-policy",
        chunk_key="chunk-policy",
        content=content,
        mentions=MENTIONS,
        logical_schema=schema,
        run_id="run-policy",
        candidate_policy_version="candidate-policy-v1",
    ).opportunities[0]


def _classified(schema: dict, predicate: str):
    pair = _plan(schema)
    hypothesis = v2.parse_classifier_response(
        [pair],
        [{
            "pair_id": pair.pair_identity,
            "decision": v2.HYPOTHESES,
            "hypotheses": [{
                "subject_occurrence_id": "occ-left",
                "predicate": predicate,
                "object_occurrence_id": "occ-right",
                "evidence_spans": [{"start": 0, "end": len(CONTENT)}],
                "classifier_confidence": 0.9,
            }],
        }],
        content=CONTENT,
        logical_schema=schema,
        classifier_model_revision="fixture-classifier",
    )[0].hypotheses[0]
    return pair, hypothesis


def _run_supported(schema: dict, predicate: str):
    pair, hypothesis = _classified(schema, predicate)
    validation = v2.validate_hypothesis(
        hypothesis, pair, content=CONTENT, logical_schema=schema,
    )
    calls: list[str] = []
    attempt = v2.verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=CONTENT,
        logical_schema=schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda _prompt, _contract: (
            calls.append(hypothesis.hypothesis_id)
            or {
                "hypothesis_id": hypothesis.hypothesis_id,
                "verdict": v2.SUPPORTED,
                "reason_code": "FIXTURE_SUPPORTED",
                "forward_entailment": "ENTAILED",
                "reverse_entailment": (
                    "ENTAILED"
                    if v2.normalize_predicate_schema(schema)[predicate].symmetric
                    else "NOT_ENTAILED"
                ),
                "subject_role_supported": True,
                "object_role_supported": True,
                "supported_evidence_spans": [
                    {"start": 0, "end": len(CONTENT)}
                ],
            }
        ),
        verifier_model_revision="fixture-verifier",
    )
    normalized = v2.normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=[hypothesis],
        verification_attempts=[attempt],
        logical_schema=schema,
    )
    return pair, validation, attempt, normalized, calls


def test_predicate_rename_keeps_structure_direction_verification_and_publish_behavior():
    outcomes = []
    for name in ("alpha_rel", "renamed_rel"):
        schema = logical_schema(predicate_policy(
            name,
            domain=["SourceType"],
            range_=["TargetType"],
            semantic_definition="The source interacts with the target.",
            direction_description="Orient from source to target.",
        ))
        pair, validation, attempt, normalized, calls = _run_supported(
            schema, name,
        )
        fact = normalized.facts[0]
        outcomes.append({
            "pair_status": pair.status,
            "validation": validation.status,
            "verdict": attempt.verdict,
            "subject": fact.subject_canonical_id,
            "object": fact.object_canonical_id,
            "symmetric": fact.symmetric,
            "review_only": fact.review_only,
            "eligible": fact.normalization_eligible,
            "verifier_calls": len(calls),
        })
    assert outcomes[0] == outcomes[1]


def test_same_name_publish_mode_metadata_changes_verifier_and_publish_behavior():
    auto_schema = logical_schema(predicate_policy(
        "alpha_rel",
        domain=["SourceType"],
        range_=["TargetType"],
        publish_mode="auto",
    ))
    review_schema = logical_schema(predicate_policy(
        "alpha_rel",
        domain=["SourceType"],
        range_=["TargetType"],
        publish_mode="review_only",
    ))
    pair, hypothesis = _classified(review_schema, "alpha_rel")
    review_validation = v2.validate_hypothesis(
        hypothesis, pair, content=CONTENT, logical_schema=review_schema,
    )
    review_attempt = v2.verify_hypothesis(
        hypothesis,
        review_validation,
        pair,
        content=CONTENT,
        logical_schema=review_schema,
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda *_args: pytest.fail("review-only called verifier"),
        verifier_model_revision="fixture-verifier",
    )
    _, auto_validation, auto_attempt, auto_normalized, calls = _run_supported(
        auto_schema, "alpha_rel",
    )
    assert review_validation.status == v2.REVIEW_ONLY
    assert review_attempt.independent_call is False
    assert review_attempt.verdict == v2.UNCERTAIN
    assert auto_validation.status == v2.PASS
    assert auto_attempt.verdict == v2.SUPPORTED
    assert calls
    assert auto_normalized.facts[0].normalization_eligible is True


def test_same_name_inverse_and_symmetric_metadata_change_normalization():
    inverse_schema = logical_schema(
        predicate_policy(
            "alpha_rel",
            domain=["SourceType"],
            range_=["TargetType"],
            inverse_predicate="beta_rel",
            canonical_orientation="inverse",
        ),
        predicate_policy(
            "beta_rel",
            domain=["TargetType"],
            range_=["SourceType"],
            inverse_predicate="alpha_rel",
        ),
    )
    _, _, _, inverse_normalized, _ = _run_supported(
        inverse_schema, "alpha_rel",
    )
    inverse_fact = inverse_normalized.facts[0]
    assert (
        inverse_fact.predicate,
        inverse_fact.subject_canonical_id,
        inverse_fact.object_canonical_id,
    ) == ("beta_rel", "entity-a-right", "entity-z-left")

    symmetric_schema = logical_schema(predicate_policy(
        "alpha_rel",
        domain=["*"],
        range_=["*"],
        symmetric=True,
    ))
    _, _, _, symmetric_normalized, _ = _run_supported(
        symmetric_schema, "alpha_rel",
    )
    symmetric_fact = symmetric_normalized.facts[0]
    assert symmetric_fact.symmetric is True
    assert (
        symmetric_fact.subject_canonical_id,
        symmetric_fact.object_canonical_id,
    ) == ("entity-a-right", "entity-z-left")


def test_fictional_predicates_support_multi_relation_without_known_names():
    schema = logical_schema(
        predicate_policy(
            "alpha_rel",
            domain=["SourceType"],
            range_=["TargetType"],
        ),
        predicate_policy(
            "beta_rel",
            domain=["SourceType"],
            range_=["TargetType"],
        ),
    )
    pair = _plan(schema)
    hypotheses = v2.parse_classifier_response(
        [pair],
        [{
            "pair_id": pair.pair_identity,
            "decision": v2.HYPOTHESES,
            "hypotheses": [{
                "subject_occurrence_id": "occ-left",
                "predicate": predicate,
                "object_occurrence_id": "occ-right",
                "evidence_spans": [{"start": 0, "end": len(CONTENT)}],
                "classifier_confidence": 0.9,
            } for predicate in ("alpha_rel", "beta_rel")],
        }],
        content=CONTENT,
        logical_schema=schema,
    )[0].hypotheses
    attempts = []
    for hypothesis in hypotheses:
        validation = v2.validate_hypothesis(
            hypothesis, pair, content=CONTENT, logical_schema=schema,
        )
        attempts.append(v2.verify_hypothesis(
            hypothesis,
            validation,
            pair,
            content=CONTENT,
            logical_schema=schema,
            role_probe=_role_probe,
            role_alignment_probe=_role_alignment_probe,
            verifier=lambda _prompt, _contract, item=hypothesis: {
                "hypothesis_id": item.hypothesis_id,
                "verdict": v2.SUPPORTED,
                "reason_code": "FIXTURE_SUPPORTED",
                "forward_entailment": "ENTAILED",
                "reverse_entailment": "NOT_ENTAILED",
                "subject_role_supported": True,
                "object_role_supported": True,
                "supported_evidence_spans": [
                    {"start": 0, "end": len(CONTENT)}
                ],
            },
            verifier_model_revision="fixture-verifier",
        ))
    normalized = v2.normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=hypotheses,
        verification_attempts=attempts,
        logical_schema=schema,
    )
    assert {fact.predicate for fact in normalized.facts} == {
        "alpha_rel", "beta_rel",
    }


def test_unknown_predicate_and_missing_critical_metadata_fail_closed():
    complete = logical_schema(predicate_policy("alpha_rel"))
    pair = _plan(complete)
    with pytest.raises(v2.ShadowClassificationError, match="predicate"):
        v2.parse_classifier_response(
            [pair],
            [{
                "pair_id": pair.pair_identity,
                "decision": v2.HYPOTHESES,
                "hypotheses": [{
                    "subject_occurrence_id": "occ-left",
                    "predicate": "unknown_rel",
                    "object_occurrence_id": "occ-right",
                    "evidence_spans": [{"start": 0, "end": len(CONTENT)}],
                    "classifier_confidence": 0.9,
                }],
            }],
            content=CONTENT,
            logical_schema=complete,
        )

    for missing in (
        "semantic_definition",
        "domain_types",
        "range_types",
        "direction_description",
        "symmetric",
        "inverse_predicate",
        "canonical_orientation",
        "publish_mode",
        "allowed_source_roles",
        "scope_policy",
        "positive_examples",
        "negative_examples",
    ):
        raw = predicate_policy("alpha_rel")
        raw.pop(missing)
        with pytest.raises(ValueError, match="missing required fields"):
            v2.normalize_predicate_schema(logical_schema(raw))


def test_soft_cues_are_schema_configured_nonbinding_hints_only():
    without_cue = logical_schema(predicate_policy(
        "alpha_rel",
        domain=["SourceType"],
        range_=["TargetType"],
    ))
    with_cue = logical_schema(predicate_policy(
        "alpha_rel",
        domain=["SourceType"],
        range_=["TargetType"],
        optional_soft_cues=["interacts with"],
    ))
    plain = _plan(without_cue)
    hinted = _plan(with_cue)
    assert plain.pair_identity == hinted.pair_identity
    assert plain.status == hinted.status == v2.PENDING_CLASSIFICATION
    assert plain.planning_reason == hinted.planning_reason
    assert plain.provenance["soft_cue_matches"] == []
    assert hinted.provenance["soft_cue_matches"]


def test_every_versioned_predicate_policy_field_enters_fingerprint_payload():
    payload = v2.predicate_policy_fingerprint_payload(logical_schema(
        predicate_policy("alpha_rel"),
        version="schema-policy-fingerprint-v1",
    ))["alpha_rel"]
    assert {
        "name",
        "semantic_definition",
        "domain_types",
        "range_types",
        "direction_description",
        "subject_role_description",
        "object_role_description",
        "symmetric",
        "inverse_predicate",
        "canonical_orientation",
        "publish_mode",
        "reciprocal_allowed",
        "allowed_context_kinds",
        "required_evidence_roles",
        "positive_examples",
        "negative_examples",
        "positive_cues",
        "negative_cues",
        "anti_patterns",
        "optional_soft_cues",
        "qualifier_schema",
        "identity_qualifier_keys",
        "conflicts_with",
        "negation_policy",
        "modality_policy",
        "reflexive",
        "schema_version",
    } <= set(payload)

    changed = deepcopy(logical_schema(
        predicate_policy("alpha_rel"),
        version="schema-policy-fingerprint-v1",
    ))
    changed["relation_types"][0]["optional_soft_cues"] = ["fixture hint"]
    assert v2.predicate_policy_fingerprint_payload(changed) != {
        "alpha_rel": payload,
    }


def test_enhanced_schema_overlay_supplies_complete_policy_as_data():
    enhanced = apply_shadow_relation_schema_overlay(BASE_SCHEMA)
    normalized = v2.normalize_predicate_schema(enhanced)
    assert set(normalized) == {
        item["name"] for item in enhanced["relation_types"]
    }


def test_enhanced_v2_production_control_flow_has_no_concrete_predicate_or_cue():
    modules = (
        Path(v2.__file__),
        Path(inspect.getfile(ShadowRelationV2Runtime)),
        Path(shadow_context.__file__),
        Path(entity_shadow.__file__),
        Path(span_entities.__file__),
    )
    forbidden = {
        "produces", "produce", "uses", "depends_on", "depends on",
        "related_to", "related to", "supply", "supplier", "contact",
        "供应", "生产", "联系人", "锂", "钴",
    }
    violations: list[tuple[str, int, str]] = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            conditions = []
            if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                conditions.append(node.test)
            elif isinstance(node, ast.comprehension):
                conditions.extend(node.ifs)
            elif isinstance(node, ast.Match):
                conditions.extend(
                    case.guard for case in node.cases if case.guard is not None
                )
            for condition in conditions:
                for literal in ast.walk(condition):
                    if not (
                        isinstance(literal, ast.Constant)
                        and isinstance(literal.value, str)
                    ):
                        continue
                    value = literal.value.casefold()
                    for token in forbidden:
                        if token.casefold() in value:
                            violations.append(
                                (path.name, literal.lineno, literal.value)
                            )
    assert violations == []

    enhanced_source = inspect.getsource(
        KnowledgeForgeService._ingest_knowledge_forge_strategy
    )
    entity_source = inspect.getsource(
        KnowledgeForgeService.build_grounded_entity_namespace
    )
    runtime_source = inspect.getsource(ShadowRelationV2Runtime)
    combined = "\n".join((enhanced_source, entity_source, runtime_source))
    assert "_run_relation_shadow(" not in combined
    assert "relation_shadow.prepare_shadow" not in combined
    assert "relation-classifier-p1-shadow-v1" not in combined
    assert "relation-candidate-p1-shadow-v4" not in combined
