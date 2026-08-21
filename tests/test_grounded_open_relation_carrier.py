from __future__ import annotations

import json
from dataclasses import replace

import pytest

from knowledge_forge.relation_extraction import (
    PASS,
    GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
    EvidenceSpan,
    RelationFreshnessDecision,
    RelationMaterializationDecision,
    VerificationAttempt,
    classifier_prompt,
    classifier_response_schema,
    normalize_verified_hypotheses,
    plan_pair_opportunities,
    proposer_prompt,
    proposer_response_schema,
    source_frame_open_carrier_bindings,
    source_frame_projection_hypotheses,
    source_relation_display_expression,
    source_batch_focused_review,
    validate_hypothesis,
)
from knowledge_forge.relation_runtime import GroundedRelationRuntime
from tests.grounded_relation_fixtures import logical_schema, predicate_policy


OPEN_POLICY = {
    "enabled": True,
    "fallback_only": True,
    "display_source_expression": True,
    "identity_source_expression": True,
    "require_exact_source_expression": True,
    "allowed_candidate_sources": [
        "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
        "ATOMIC_FRAME_SCHEMA_PROJECTION",
    ],
}


def test_rendered_edge_transport_uses_only_inner_predicate_label():
    assert source_relation_display_expression("--calls-->") == "calls"
    assert source_relation_display_expression("--调用-->") == "调用"


def _schema():
    return logical_schema(
        predicate_policy("related", publish_mode="review_only"),
        predicate_policy("supplies"),
        predicate_policy(
            "grounded_carrier",
            open_relation_policy=OPEN_POLICY,
        ),
        version="open-carrier-test-v1",
    )


def _source_fixture(*relations: str):
    content = "Alpha provides and maintains Beta."
    mentions = [
        {
            "mention_id": "m-alpha",
            "occurrence_key": "occ-alpha",
            "surface": "Alpha",
            "mention_start": content.index("Alpha"),
            "mention_end": content.index("Alpha") + len("Alpha"),
            "canonical_id": "entity-alpha",
            "assignment_version": "assignment-v1",
            "type": "GenericEntity",
            "source_role": "body",
            "relation_eligible": True,
        },
        {
            "mention_id": "m-beta",
            "occurrence_key": "occ-beta",
            "surface": "Beta",
            "mention_start": content.index("Beta"),
            "mention_end": content.index("Beta") + len("Beta"),
            "canonical_id": "entity-beta",
            "assignment_version": "assignment-v1",
            "type": "GenericEntity",
            "source_role": "body",
            "relation_eligible": True,
        },
    ]
    schema = _schema()
    pair = plan_pair_opportunities(
        project_id=7,
        document_revision_id="revision-open",
        chunk_key="chunk-open",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-open",
        candidate_policy_version="candidate-open-v1",
    ).opportunities[0]
    inventory = {
        "frames": [
            {
                "relation": relation,
                "arguments": ["Alpha", "Beta"],
                "roles": ["subject", "object"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": ["Alpha", "Beta"],
                "assertion_status": "ASSERTED",
                "argument_occurrence_keys": ["occ-alpha", "occ-beta"],
                "evidence_span": [0, len(content)],
                "evidence_text": content,
            }
            for relation in relations
        ],
    }
    return schema, content, mentions, pair, inventory


def _attempt(hypothesis, *, verdict="SUPPORTED"):
    return VerificationAttempt(
        attempt_id="attempt-" + hypothesis.hypothesis_id,
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=1,
        verdict=verdict,
        reason_code=(
            "DIRECTLY_ENTAILED"
            if verdict == "SUPPORTED" else "NOT_ENTAILED"
        ),
        supported_evidence_spans=(
            hypothesis.evidence_spans if verdict == "SUPPORTED" else ()
        ),
        independent_call=True,
        verifier_prompt_version="verifier-v1",
        verifier_schema_version="verifier-schema-v1",
        verifier_parser_version="verifier-parser-v1",
        verifier_model_revision="model-r1",
    )


def test_open_carrier_is_fallback_only_and_precise_binding_wins():
    schema, _content, _mentions, _pair, inventory = _source_fixture("provides")

    fallback = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )
    behind_review_only = source_frame_open_carrier_bindings(
        inventory,
        [{
            "frame_index": 0,
            "predicate": "related",
            "subject_index": 0,
            "object_index": 1,
        }],
        logical_schema=schema,
        max_bindings=8,
    )
    behind_precise = source_frame_open_carrier_bindings(
        inventory,
        [{
            "frame_index": 0,
            "predicate": "supplies",
            "subject_index": 0,
            "object_index": 1,
        }],
        logical_schema=schema,
        max_bindings=8,
    )

    assert [item["predicate"] for item in fallback["bindings"]] == [
        "grounded_carrier"
    ]
    assert len(behind_review_only["bindings"]) == 1
    assert len(behind_precise["bindings"]) == 1
    assert behind_precise["parallel_precise_binding"] == 1
    assert behind_precise["skipped_precise_binding"] == 0


def test_open_carrier_rejects_bare_negative_and_caps_without_hiding_audit():
    schema, _content, _mentions, _pair, inventory = _source_fixture(
        "is", "not provides", "maintains", "provides",
    )

    result = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=1,
    )

    assert len(result["bindings"]) == 1
    assert result["eligible_frames"] == 2
    assert result["skipped_low_information"] == 2
    assert result["bindings_capped"] == 1


def test_open_carrier_budget_prioritizes_parser_proved_structure():
    schema, _content, _mentions, _pair, inventory = _source_fixture(
        "maintains", "contains",
    )
    inventory["frames"][1]["deterministic_source_structure"] = (
        "list_parent_child"
    )

    result = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=1,
    )

    assert [item["frame_index"] for item in result["bindings"]] == [1]
    assert result["eligible_frames"] == 2
    assert result["bindings_capped"] == 1


def test_open_carrier_rejects_identifier_style_explicit_negation():
    schema, _content, _mentions, _pair, inventory = _source_fixture("is_not")

    result = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )

    assert result["bindings"] == []
    assert result["eligible_frames"] == 0


def test_open_carrier_rejects_model_paraphrase_of_possessive_topology():
    schema, _content, _mentions, _pair, inventory = _source_fixture("contained")
    inventory["frames"][0]["parser_inferred_source_roles"] = [
        "possessor", "possessed",
    ]
    inventory["frames"][0]["parser_inferred_source_syntax"] = (
        "possessive_reference_construction"
    )

    result = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )

    assert result["bindings"] == []
    assert result["eligible_frames"] == 0


def test_open_carrier_rejects_relation_wording_not_copied_from_source():
    schema, _content, _mentions, _pair, inventory = _source_fixture("包含")

    result = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )

    assert result["bindings"] == []
    assert result["eligible_frames"] == 0
    assert result["skipped_ungrounded_expression"] == 1


def test_bare_function_word_frame_cannot_be_promoted_to_precise_relation():
    schema, _content, mentions, pair, inventory = _source_fixture("is")
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [{
            "frame_index": 0,
            "predicate": "supplies",
            "subject_index": 0,
            "object_index": 1,
            "binding_source": "MODEL_SOURCE_BINDING",
        }]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )

    assert hypotheses == ()


def test_possessive_topology_without_schema_role_contract_is_not_published():
    schema, _content, mentions, pair, inventory = _source_fixture(
        "possessive association"
    )
    inventory["frames"][0]["roles"] = ["possessor", "possessed"]
    inventory["frames"][0]["deterministic_source_syntax"] = (
        "possessive_reference_construction"
    )
    precise_binding = {
        "frame_index": 0,
        "predicate": "supplies",
        "subject_index": 0,
        "object_index": 1,
        "binding_source": "MODEL_SOURCE_BINDING",
    }
    carrier = source_frame_open_carrier_bindings(
        inventory, [precise_binding], logical_schema=schema, max_bindings=8,
    )["bindings"]
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [precise_binding, *carrier]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )

    assert carrier == []
    assert hypotheses == ()


def test_pairwise_classifier_positive_requires_mandatory_independent_review():
    schema, _content, mentions, pair, inventory = _source_fixture("provides")
    source_hypothesis = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [{
            "frame_index": 0,
            "predicate": "supplies",
            "subject_index": 0,
            "object_index": 1,
            "binding_source": "MODEL_SOURCE_BINDING",
        }]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )[0]
    classifier_metadata = dict(source_hypothesis.predicate_metadata)
    classifier_metadata.pop("candidate_source", None)
    classifier_hypothesis = replace(
        source_hypothesis,
        classifier_prompt_version=GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
        predicate_metadata=classifier_metadata,
    )

    assert source_batch_focused_review(
        classifier_hypothesis,
        {"verdict": "SUPPORTED"},
        logical_schema=schema,
    ) == {
        "kind": "confirmation",
        "reason": "BATCH_SUPPORTED_PAIRWISE_CLASSIFIER",
        "priority": -2,
        "mandatory": True,
    }
    assert source_batch_focused_review(
        classifier_hypothesis,
        {"verdict": "REJECTED"},
        logical_schema=schema,
    ) is None


def test_model_only_nary_binding_requires_mandatory_role_audit():
    schema, _content, mentions, pair, inventory = _source_fixture("provides")
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [{
            "frame_index": 0,
            "predicate": "supplies",
            "subject_index": 0,
            "object_index": 1,
            "binding_source": "MODEL_SOURCE_BINDING",
        }]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )[0]
    hypothesis = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_frame_other_participants": ["Gamma"],
            "source_syntax_origin": "",
        },
    )

    assert source_batch_focused_review(
        hypothesis,
        {"verdict": "SUPPORTED"},
        logical_schema=schema,
    ) == {
        "kind": "confirmation",
        "reason": "BATCH_SUPPORTED_MODEL_ONLY_NARY_BINDING",
        "priority": -1,
        "source_role_audit": True,
        "role_audit_only": True,
        "mandatory": True,
    }


def test_open_carrier_is_invisible_to_generic_model_candidate_stages():
    schema, content, mentions, pair, _inventory = _source_fixture("provides")

    classifier_contract = classifier_response_schema([pair])
    predicate_enum = classifier_contract["properties"]["hypotheses"][
        "items"
    ]["properties"]["predicate"]["enum"]
    classifier_payload = json.loads(classifier_prompt(
        [pair], content=content, logical_schema=schema,
    ))
    proposer_contract = proposer_response_schema(
        mentions=mentions,
        logical_schema=schema,
        content_length=len(content),
    )
    proposer_enum = proposer_contract["properties"]["proposals"][
        "items"
    ]["properties"]["predicate"]["enum"]
    proposer_payload = json.loads(proposer_prompt(
        context_id="ctx-open",
        content=content,
        mentions=mentions,
        logical_schema=schema,
    ))

    assert "grounded_carrier" not in predicate_enum
    assert "grounded_carrier" not in classifier_payload["predicate_catalog"]
    assert "grounded_carrier" not in classifier_payload["pairs"][0][
        "allowed_predicates"
    ]
    assert "grounded_carrier" not in proposer_enum
    assert "grounded_carrier" not in proposer_payload["predicates"]


def test_open_carrier_requires_exact_grounded_source_contract():
    schema, content, mentions, pair, inventory = _source_fixture("provides")
    binding = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        binding,
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )[0]

    accepted = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=schema,
    )
    forged_source = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "candidate_source": "DIRECT_PROPOSER",
        },
    )
    forged_text = replace(
        hypothesis,
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_frame_evidence_text": "different bytes",
        },
    )

    assert accepted.status == PASS
    assert validate_hypothesis(
        forged_source, pair, content=content, logical_schema=schema,
    ).reason_code == "OPEN_RELATION_SOURCE_NOT_ALLOWED"
    assert validate_hypothesis(
        forged_text, pair, content=content, logical_schema=schema,
    ).reason_code == "OPEN_RELATION_EVIDENCE_TEXT_MISMATCH"
    assert source_batch_focused_review(
        hypothesis,
        {"verdict": "REJECTED"},
        logical_schema=schema,
    ) == {
        "kind": "recovery",
        "reason": "BATCH_REJECTED_GROUNDED_OPEN_RELATION",
        "priority": 3,
        "source_role_audit": False,
    }
    assert source_batch_focused_review(
        hypothesis,
        {"verdict": "SUPPORTED"},
        logical_schema=schema,
    ) is None


def test_distinct_source_expressions_on_one_pair_remain_distinct_facts():
    schema, content, mentions, pair, inventory = _source_fixture(
        "provides", "maintains",
    )
    binding = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        binding,
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )

    assert len(hypotheses) == 2
    assert all(
        validate_hypothesis(
            item, pair, content=content, logical_schema=schema,
        ).status == PASS
        for item in hypotheses
    )
    normalized = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=hypotheses,
        verification_attempts=[_attempt(item) for item in hypotheses],
        logical_schema=schema,
    )

    assert len(normalized.facts) == 2
    assert len({item.semantic_fact_key for item in normalized.facts}) == 2
    assert {
        item.provenance["source_relation_expression"]
        for item in normalized.evidence
    } == {"provides", "maintains"}


def test_precise_supported_preserves_carrier_until_publication_gate():
    schema, _content, mentions, pair, inventory = _source_fixture("provides")
    precise_binding = {
        "frame_index": 0,
        "predicate": "supplies",
        "subject_index": 0,
        "object_index": 1,
        "binding_source": "MODEL_SOURCE_BINDING",
    }
    carrier = source_frame_open_carrier_bindings(
        inventory, [precise_binding], logical_schema=schema, max_bindings=8,
    )["bindings"]
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [precise_binding, *carrier]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )
    precise = next(item for item in hypotheses if item.predicate == "supplies")
    open_fact = next(
        item for item in hypotheses if item.predicate == "grounded_carrier"
    )

    normalized_supported = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=hypotheses,
        verification_attempts=[_attempt(precise), _attempt(open_fact)],
        logical_schema=schema,
    )
    normalized_rejected = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=hypotheses,
        verification_attempts=[
            _attempt(precise, verdict="REJECTED"),
            _attempt(open_fact),
        ],
        logical_schema=schema,
    )

    assert {item.predicate for item in normalized_supported.facts} == {
        "supplies", "grounded_carrier",
    }
    assert normalized_supported.metrics[
        "open_carrier_facts_suppressed_by_precise"
    ] == 0
    assert [item.predicate for item in normalized_rejected.facts] == [
        "grounded_carrier"
    ]
    assert normalized_rejected.metrics[
        "open_carrier_facts_suppressed_by_precise"
    ] == 0


def test_precise_source_expression_does_not_remove_publication_fallback():
    schema, _content, mentions, pair, inventory = _source_fixture(
        "provides", "provides",
    )
    precise_binding = {
        "frame_index": 1,
        "predicate": "supplies",
        "subject_index": 1,
        "object_index": 0,
        "binding_source": "MODEL_SOURCE_BINDING",
    }
    carrier_binding = next(
        item for item in source_frame_open_carrier_bindings(
            inventory, [], logical_schema=schema, max_bindings=8,
        )["bindings"]
        if item["frame_index"] == 0
    )
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [precise_binding, carrier_binding]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )
    precise = next(item for item in hypotheses if item.predicate == "supplies")
    carrier = next(
        item for item in hypotheses if item.predicate == "grounded_carrier"
    )

    normalized = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=hypotheses,
        verification_attempts=[_attempt(item) for item in hypotheses],
        logical_schema=schema,
    )

    assert {item.predicate for item in normalized.facts} == {
        "supplies", "grounded_carrier",
    }
    assert normalized.metrics[
        "open_carrier_evidence_suppressed_by_source_expression"
    ] == 0

    different_expression_carrier = replace(
        carrier,
        hypothesis_id="different-source-expression-carrier",
        predicate_metadata={
            **carrier.predicate_metadata,
            "source_frame_relation": "maintains",
        },
    )
    distinct = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=[precise, different_expression_carrier],
        verification_attempts=[
            _attempt(precise), _attempt(different_expression_carrier),
        ],
        logical_schema=schema,
    )
    assert {item.predicate for item in distinct.facts} == {
        "supplies", "grounded_carrier",
    }
    assert distinct.metrics[
        "open_carrier_evidence_suppressed_by_source_expression"
    ] == 0


def test_parser_structure_suppresses_overlapping_inverse_open_carrier():
    schema, content, mentions, pair, inventory = _source_fixture("provides")
    inventory["frames"].append({
        "relation": "structural child of",
        "arguments": ["Beta", "Alpha"],
        "roles": ["structural_child", "structural_parent"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Beta", "Alpha"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["occ-beta", "occ-alpha"],
        "evidence_span": [0, len(content)],
        "evidence_text": content,
        "deterministic_source_structure": "list_parent_child",
    })
    carrier = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )["bindings"]
    carrier = [item for item in carrier if item["frame_index"] == 0]
    precise_binding = {
        "frame_index": 1,
        "predicate": "supplies",
        "subject_index": 0,
        "object_index": 1,
        "binding_source": "DETERMINISTIC_SOURCE_STRUCTURE",
    }
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [precise_binding, *carrier]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )

    assert {item.predicate for item in hypotheses} == {
        "supplies", "grounded_carrier",
    }
    normalized = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=hypotheses,
        verification_attempts=[_attempt(item) for item in hypotheses],
        logical_schema=schema,
    )

    assert [item.predicate for item in normalized.facts] == ["supplies"]
    assert normalized.metrics[
        "open_carrier_evidence_suppressed_by_structure"
    ] == 1
    assert normalized.metrics[
        "open_carrier_facts_suppressed_by_precise"
    ] == 1


def test_verified_open_parser_contract_suppresses_ordinary_open_paraphrase():
    schema, content, mentions, pair, inventory = _source_fixture("provides")
    inventory["frames"].append({
        "relation": "parenthetical detail of",
        "arguments": ["Beta", "Alpha"],
        "roles": ["parenthetical_detail", "parenthetical_head"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Beta", "Alpha"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["occ-beta", "occ-alpha"],
        "evidence_span": [0, len(content)],
        "evidence_text": content,
        "deterministic_source_structure": "parenthetical_enumeration",
        "parallel_open_carrier_policy": (
            "suppress_same_pair_overlapping_open_carriers"
        ),
    })
    bindings = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )
    hypotheses = source_frame_projection_hypotheses(
        inventory,
        bindings,
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )
    structural = next(
        item for item in hypotheses
        if item.predicate_metadata.get("source_structure_kind")
        == "parenthetical_enumeration"
    )
    ordinary = next(item for item in hypotheses if item is not structural)
    structural_attempt = replace(
        _attempt(structural),
        reason_code="SUPPORTED_PARSER_STRUCTURE_CONTRACT",
        verifier_model_revision="deterministic:parser-contract-v1",
    )

    normalized = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=hypotheses,
        verification_attempts=[_attempt(ordinary), structural_attempt],
        logical_schema=schema,
    )

    assert len(normalized.facts) == 1
    assert [
        item.provenance["source_relation_expression"]
        for item in normalized.evidence
    ] == ["parenthetical detail of"]
    assert normalized.metrics[
        "open_carrier_evidence_suppressed_by_structure"
    ] == 1
    assert normalized.metrics[
        "open_carrier_facts_suppressed_by_precise"
    ] == 1

    unscoped_metadata = dict(structural.predicate_metadata)
    unscoped_metadata.pop("source_parallel_open_carrier_policy", None)
    unscoped = replace(
        structural,
        hypothesis_id="unscoped-parser-contract",
        predicate_metadata=unscoped_metadata,
    )
    unscoped_attempt = replace(
        structural_attempt,
        attempt_id="attempt-unscoped-parser-contract",
        hypothesis_id=unscoped.hypothesis_id,
    )
    normalized_unscoped = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=[ordinary, unscoped],
        verification_attempts=[_attempt(ordinary), unscoped_attempt],
        logical_schema=schema,
    )
    assert len(normalized_unscoped.facts) == 2
    assert normalized_unscoped.metrics[
        "open_carrier_evidence_suppressed_by_structure"
    ] == 0


def test_materialized_open_edge_displays_exact_relation_not_carrier_id():
    schema, content, mentions, pair, inventory = _source_fixture("provides")
    pair = replace(pair, provenance={
        **pair.provenance,
        "chunk_id": 1,
        "document_id": 2,
    })
    binding = source_frame_open_carrier_bindings(
        inventory, [], logical_schema=schema, max_bindings=8,
    )
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        binding,
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )[0]
    normalization = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=[hypothesis],
        verification_attempts=[_attempt(hypothesis)],
        logical_schema=schema,
    )
    fact = normalization.facts[0]
    decision = RelationMaterializationDecision(
        fact_id=fact.fact_id,
        eligible=True,
        reasons=(),
        freshness=RelationFreshnessDecision(
            checkpoint="materialize",
            current=True,
            status="CURRENT",
            recorded_root="root",
            current_root="root",
            changed_components=(),
        ),
    )

    rows = GroundedRelationRuntime(None, None)._endpoint_rows(
        run_id="run-open",
        normalization=normalization,
        materialization_decisions=[decision],
        opportunities={pair.pair_identity: pair},
        hypotheses={hypothesis.hypothesis_id: hypothesis},
        chunks={
            1: {
                "document_id": 2,
                "revision_id": "revision-open",
                "content": content,
            },
        },
        eligible_assignments={
            "m-alpha": {"canonical_key": "entity-alpha"},
            "m-beta": {"canonical_key": "entity-beta"},
        },
        logical_schema=schema,
    )

    assert rows[0]["relation_type"] == "grounded_carrier"
    assert rows[0]["metadata"]["display_relation_label"] == "provides"
    assert rows[0]["metadata"]["source_relation_expression"] == "provides"
    assert rows[0]["metadata"]["normalization_status"] == "OPEN_GROUNDED"
    assert rows[0]["metadata"]["relation_semantics"]["layer"] == (
        "open_source_expression"
    )
    assert rows[0]["metadata"]["publication_integrity"][
        "source_endpoint_eligible"
    ] is True

    with pytest.raises(
        RuntimeError,
        match="publication endpoint is not entity-publishability eligible",
    ):
        GroundedRelationRuntime(None, None)._endpoint_rows(
            run_id="run-open",
            normalization=normalization,
            materialization_decisions=[decision],
            opportunities={pair.pair_identity: pair},
            hypotheses={hypothesis.hypothesis_id: hypothesis},
            chunks={
                1: {
                    "document_id": 2,
                    "revision_id": "revision-open",
                    "content": content,
                },
            },
            eligible_assignments={
                "m-alpha": {"canonical_key": "entity-alpha"},
            },
        )


def test_materialized_schema_edge_uses_canonical_label_and_keeps_raw_expression():
    schema, content, mentions, pair, inventory = _source_fixture("provides")
    pair = replace(pair, provenance={
        **pair.provenance,
        "chunk_id": 1,
        "document_id": 2,
    })
    precise_binding = {
        "frame_index": 0,
        "predicate": "supplies",
        "subject_index": 0,
        "object_index": 1,
        "binding_source": "MODEL_SOURCE_BINDING",
    }
    hypothesis = source_frame_projection_hypotheses(
        inventory,
        {"bindings": [precise_binding]},
        occurrences=mentions,
        opportunities_by_pair={pair.pair_identity: pair},
        logical_schema=schema,
        classifier_model_revision="model-r1",
    )[0]
    normalization = normalize_verified_hypotheses(
        project_id=7,
        opportunities={pair.pair_identity: pair},
        hypotheses=[hypothesis],
        verification_attempts=[_attempt(hypothesis)],
        logical_schema=schema,
    )
    fact = normalization.facts[0]
    decision = RelationMaterializationDecision(
        fact_id=fact.fact_id,
        eligible=True,
        reasons=(),
        freshness=RelationFreshnessDecision(
            checkpoint="materialize",
            current=True,
            status="CURRENT",
            recorded_root="root",
            current_root="root",
            changed_components=(),
        ),
    )

    rows = GroundedRelationRuntime(None, None)._endpoint_rows(
        run_id="run-schema",
        normalization=normalization,
        materialization_decisions=[decision],
        opportunities={pair.pair_identity: pair},
        hypotheses={hypothesis.hypothesis_id: hypothesis},
        chunks={
            1: {
                "document_id": 2,
                "revision_id": "revision-open",
                "content": content,
            },
        },
        eligible_assignments={
            "m-alpha": {"canonical_key": "entity-alpha"},
            "m-beta": {"canonical_key": "entity-beta"},
        },
        logical_schema=schema,
    )

    assert rows[0]["relation_type"] == "supplies"
    assert rows[0]["metadata"]["source_relation_expression"] == "provides"
    assert "display_relation_label" not in rows[0]["metadata"]
    assert rows[0]["metadata"]["relation_semantics"]["layer"] == (
        "schema_predicate"
    )
    assert rows[0]["metadata"]["relation_semantics"]["publication"][
        "eligible"
    ] is True
