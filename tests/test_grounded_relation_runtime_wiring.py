from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from knowledge_forge.config import Config
from knowledge_forge.ollama import OllamaProtocolError
from knowledge_forge import relation_extraction as v2
from knowledge_forge.relation_runtime import (
    CLASSIFIER_ACCEPTED,
    CLASSIFIER_RETRYABLE_PROTOCOL,
    GROUNDED_RELATION_POLICY_SET,
    GroundedRelationRuntime,
    _apply_parser_endpoint_exclusions,
    _graph_extraction_chunks,
    _source_first_pair_backstop_required,
    relation_runtime_treatment_digest,
)
from knowledge_forge.evidence_context import (
    build_evidence_context_manifest,
    structural_spans_for_chunk,
)
from knowledge_forge.relation_extraction import (
    HYPOTHESES,
    NO_RELATION,
    OTHER_UNSUPPORTED,
    REJECTED,
    RelationClassificationError,
    SUPPORTED,
    UNCERTAIN,
    StaleRelationSnapshotError,
)
from tests.grounded_relation_fixtures import logical_schema, predicate_policy


def test_relation_runtime_honors_retrieval_only_chunk_metadata():
    chunks = [
        {"id": 1, "metadata": {}},
        {"id": 2, "metadata": {"graph_extraction_enabled": True}},
        {"id": 3, "metadata": {"graph_extraction_enabled": False}},
        {"id": 4, "metadata": '{"graph_extraction_enabled": false}'},
    ]

    assert [
        item["id"] for item in _graph_extraction_chunks(chunks)
    ] == [1, 2]


def test_grounded_reference_keeps_exact_antecedent_evidence_envelope():
    content = "Core Service\nIt includes Task."
    proxy_start = content.index("It")
    task_start = content.index("Task")
    assertion_span = [proxy_start, len(content)]
    mentions = [
        {
            "mention_key": "core", "mention_id": "core",
            "occurrence_key": "core", "surface": "Core Service",
            "mention_start": 0, "mention_end": len("Core Service"),
            "canonical_id": "canonical-core",
            "assignment_version": "assignment-v1", "type": "Technology",
        },
        {
            "mention_key": "proxy", "mention_id": "proxy",
            "occurrence_key": "proxy", "surface": "It",
            "mention_start": proxy_start,
            "mention_end": proxy_start + len("It"),
            "canonical_id": "canonical-core",
            "assignment_version": "assignment-v1", "type": "Technology",
        },
        {
            "mention_key": "task", "mention_id": "task",
            "occurrence_key": "task", "surface": "Task",
            "mention_start": task_start,
            "mention_end": task_start + len("Task"),
            "canonical_id": "canonical-task",
            "assignment_version": "assignment-v1", "type": "GenericEntity",
        },
    ]
    schema = logical_schema(predicate_policy(
        "part_of", domain=["*"], range_=["*"],
        scope_policy={
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        },
    ))
    frame = {
        "relation": "includes",
        "arguments": ["It", "Task"],
        "roles": ["subject", "object"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["It", "Task"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["proxy", "task"],
        "evidence_span": assertion_span,
        "evidence_text": content[assertion_span[0]:assertion_span[1]],
        "resolved_reference_endpoints": [{
            "role": "subject", "occurrence_key": "proxy",
            "reference_text": "It", "reference_span": [
                proxy_start, proxy_start + len("It"),
            ],
            "antecedent_occurrence_key": "core",
        }],
    }
    opportunity = GroundedRelationRuntime._grounded_open_fact_opportunity(
        run_id="run-reference-envelope",
        project_id=1,
        chunk={
            "id": 1, "document_id": 1,
            "revision_id": "revision-reference-envelope",
            "chunk_key": "chunk-reference-envelope", "content": content,
            "content_hash": "reference-envelope-hash",
        },
        frame=frame,
        mentions=mentions,
        logical_schema=schema,
    )

    assert opportunity is not None
    assert opportunity.provenance["required_atomic_scope"] == assertion_span
    reference = opportunity.provenance[
        "resolved_reference_endpoints"
    ][0]
    assert reference["antecedent_span"] == [0, len("Core Service")]
    assert opportunity.context_span == (0, len(content))
    assert v2._spans_stay_within_required_atomic_scope(
        (v2.EvidenceSpan(0, len(content)),), opportunity,
    ) is True
    assert v2._spans_stay_within_required_atomic_scope(
        (v2.EvidenceSpan(0, len(content) + 1),), opportunity,
    ) is False


def test_list_item_head_excludes_parenthetical_details_from_parent_hierarchy():
    content = "\uf0a8 Core Platform\n— Monitoring (Core、Partition)"
    spans = structural_spans_for_chunk(content=content, blocks=[])
    raw_mentions = []
    for key, surface in (
        ("root", "Core Platform"),
        ("item", "Monitoring"),
        ("detail-a", "Core"),
        ("detail-b", "Partition"),
    ):
        start = content.index(
            surface,
            content.index("(") if key.startswith("detail-") else 0,
        )
        raw_mentions.append({
            "mention_key": key,
            "surface": surface,
            "mention_start": start,
            "mention_end": start + len(surface),
            "canonical_id": f"canonical-{key}",
            "assignment_version": "assignment-v1",
            "type": "GenericEntity",
        })
    chunk = {
        "id": 1,
        "revision_id": "revision-list-head",
        "content": content,
        "metadata": {},
        "shadow_structure": {"structural_spans": spans},
        "shadow_context_kind": "structured_chunk",
        "shadow_context_quality": "structured_chunk_mapped",
        "shadow_source_map": [],
    }
    mentions = GroundedRelationRuntime._mentions_for_context(
        chunk, raw_mentions,
    )
    by_surface = {item["surface"]: item for item in mentions}
    assert by_surface["Core Platform"]["list_scope_head"] is True
    assert by_surface["Monitoring"]["list_scope_head"] is True
    assert by_surface["Core"]["list_scope_head"] is False
    assert by_surface["Partition"]["list_scope_head"] is False

    inventory = v2.augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=mentions,
    )
    hierarchy = {
        tuple(frame["arguments"])
        for frame in inventory["frames"]
        if frame.get("deterministic_source_structure")
        == "list_parent_child"
    }
    assert hierarchy == {("Monitoring", "Core Platform")}
    parenthetical = {
        tuple(frame["arguments"])
        for frame in inventory["frames"]
        if frame.get("deterministic_source_structure")
        == v2.PARENTHETICAL_ENUMERATION_STRUCTURE_KIND
    }
    assert parenthetical == {
        ("Core", "Monitoring"),
        ("Partition", "Monitoring"),
    }
    assert all(
        frame.get("parallel_open_carrier_policy")
        == v2.STRUCTURE_PARALLEL_OPEN_CARRIER_SUPPRESSION
        for frame in inventory["frames"]
        if frame.get("deterministic_source_structure")
        == v2.PARENTHETICAL_ENUMERATION_STRUCTURE_KIND
    )

    schema = logical_schema(
        predicate_policy(
            "part_of",
            domain=["*"],
            range_=["*"],
            structural_evidence_kinds=["list_parent_child"],
            scope_policy={
                "require_same_atomic_scope": False,
                "allow_cross_scope_with_explicit_evidence": True,
            },
        ),
        predicate_policy(
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
        ),
    )
    plan = v2.plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-list-head",
        chunk_key="chunk-list-head",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-list-head",
        candidate_policy_version="candidate-list-head",
        max_char_distance=1,
    )
    structural_pairs = {
        frozenset((item.left_occurrence_key, item.right_occurrence_key))
        for item in plan.opportunities
        if item.planning_reason == "hierarchical_list_parent_child"
    }
    assert structural_pairs == {frozenset((
        by_surface["Monitoring"]["occurrence_key"],
        by_surface["Core Platform"]["occurrence_key"],
    ))}

    carrier_bindings = v2.source_frame_open_carrier_bindings(
        inventory,
        [],
        logical_schema=schema,
        max_bindings=32,
    )
    hypotheses = v2.source_frame_projection_hypotheses(
        inventory,
        carrier_bindings,
        occurrences=mentions,
        opportunities_by_pair={
            item.pair_identity: item for item in plan.opportunities
        },
        logical_schema=schema,
        classifier_model_revision="fixture-model",
    )
    detail_hypotheses = [
        item for item in hypotheses
        if item.predicate_metadata.get("source_structure_kind")
        == v2.PARENTHETICAL_ENUMERATION_STRUCTURE_KIND
    ]
    assert all(
        item.predicate_metadata.get(
            "source_parallel_open_carrier_policy"
        ) == v2.STRUCTURE_PARALLEL_OPEN_CARRIER_SUPPRESSION
        for item in detail_hypotheses
    )
    assert {
        (
            item.predicate_metadata["source_subject_surface"],
            item.predicate_metadata["source_object_surface"],
        )
        for item in detail_hypotheses
    } == {("Core", "Monitoring"), ("Partition", "Monitoring")}
    opportunity_by_pair = {
        item.pair_identity: item for item in plan.opportunities
    }
    for hypothesis in detail_hypotheses:
        opportunity = opportunity_by_pair[hypothesis.pair_identity]
        validation = v2.validate_hypothesis(
            hypothesis,
            opportunity,
            content=content,
            logical_schema=schema,
        )
        assert validation.status == v2.PASS
        assert v2.parser_structure_claim_intent(
            hypothesis,
            opportunity,
            logical_schema=schema,
        ) is True
        terminal = v2.deterministic_parser_structure_terminal_attempt(
            hypothesis,
            validation,
            opportunity,
            content=content,
            logical_schema=schema,
        )
        assert terminal.verdict == v2.SUPPORTED
        assert terminal.model_request_payload["model_call"] is False

    hypothesis = detail_hypotheses[0]
    opportunity = opportunity_by_pair[hypothesis.pair_identity]
    malformed = replace(
        hypothesis,
        hypothesis_id="malformed-parenthetical-contract",
        predicate_metadata={
            **hypothesis.predicate_metadata,
            "source_structure_contract": {
                **hypothesis.predicate_metadata["source_structure_contract"],
                "bracket_span": [0, len(content)],
            },
        },
    )
    malformed_validation = v2.validate_hypothesis(
        malformed,
        opportunity,
        content=content,
        logical_schema=schema,
    )
    malformed_terminal = v2.deterministic_parser_structure_terminal_attempt(
        malformed,
        malformed_validation,
        opportunity,
        content=content,
        logical_schema=schema,
    )
    assert malformed_terminal.verdict == v2.REJECTED
    assert malformed_terminal.reason_code == (
        "PARSER_STRUCTURE_CONTRACT_INVALID"
    )


def test_source_first_pair_backstop_keeps_only_independent_signals():
    content = "Alpha connects Beta."
    schema = logical_schema(
        predicate_policy(
            "acts_on", domain=["Organization"], range_=["Product"],
        ),
    )
    opportunity = v2.plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-backstop",
        chunk_key="chunk-backstop",
        content=content,
        mentions=[
            {
                "occurrence_key": "alpha",
                "mention_id": "alpha",
                "surface": "Alpha",
                "mention_start": 0,
                "mention_end": 5,
                "canonical_id": "canonical-alpha",
                "assignment_version": "assignment-v1",
                "type": "Organization",
            },
            {
                "occurrence_key": "beta",
                "mention_id": "beta",
                "surface": "Beta",
                "mention_start": 15,
                "mention_end": 19,
                "canonical_id": "canonical-beta",
                "assignment_version": "assignment-v1",
                "type": "Product",
            },
        ],
        logical_schema=schema,
        run_id="run-backstop",
        candidate_policy_version="candidate-backstop-v1",
    ).opportunities[0]

    assert opportunity.provenance["structure_admission"] == "same_atomic_scope"
    assert not _source_first_pair_backstop_required(opportunity)
    excluded = _apply_parser_endpoint_exclusions(opportunity, {"alpha"})
    assert excluded.status == v2.FILTERED_STRUCTURE_INELIGIBLE
    assert excluded.planning_reason == (
        "parser_role_label_is_not_relation_endpoint"
    )
    assert excluded.provenance["parser_excluded_relation_endpoint_keys"] == [
        "alpha"
    ]
    assert _source_first_pair_backstop_required(replace(
        opportunity,
        provenance={
            **opportunity.provenance,
            "source_repair_backstop": True,
        },
    ))
    assert _source_first_pair_backstop_required(replace(
        opportunity,
        provenance={
            **opportunity.provenance,
            "pair_local_soft_cue_matches": [
                {"predicate": "acts_on", "cue": "connects"},
            ],
        },
    ))
    assert not _source_first_pair_backstop_required(replace(
        opportunity,
        provenance={
            **opportunity.provenance,
            "same_atomic_scope": False,
            "pair_local_soft_cue_matches": [
                {"predicate": "acts_on", "cue": "connects"},
            ],
        },
    ))
    for structure in (
        "same_cell",
        "same_table_row",
        "hierarchical_list_parent_child",
        "hierarchical_section_label_parent_child",
    ):
        assert _source_first_pair_backstop_required(replace(
            opportunity,
            provenance={
                **opportunity.provenance,
                "structure_admission": structure,
            },
        ))


def test_source_first_soft_cue_backstop_is_pair_local_not_sentence_wide():
    content = "Platform supports Alpha and Beta."
    schema = logical_schema(predicate_policy(
        "supports", optional_soft_cues=["supports"],
    ))
    surfaces = ["Platform", "Alpha", "Beta"]
    mentions = []
    for surface in surfaces:
        start = content.index(surface)
        mentions.append({
            "occurrence_key": surface.lower(),
            "mention_id": surface.lower(),
            "surface": surface,
            "mention_start": start,
            "mention_end": start + len(surface),
            "canonical_id": "canonical-" + surface.lower(),
            "assignment_version": "assignment-v1",
            "type": "GenericEntity",
        })
    plan = v2.plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-local-cue",
        chunk_key="chunk-local-cue",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id="run-local-cue",
        candidate_policy_version="candidate-local-cue-v1",
    )
    by_pair = {
        frozenset((item.left_occurrence_key, item.right_occurrence_key)): item
        for item in plan.opportunities
    }

    platform_alpha = by_pair[frozenset(("platform", "alpha"))]
    alpha_beta = by_pair[frozenset(("alpha", "beta"))]
    assert platform_alpha.provenance["pair_local_soft_cue_matches"]
    assert not alpha_beta.provenance["pair_local_soft_cue_matches"]
    assert _source_first_pair_backstop_required(platform_alpha)
    assert not _source_first_pair_backstop_required(alpha_beta)


def test_grounded_open_fact_uses_entity_types_as_soft_verifier_priors():
    content = "Alpha connects Beta."
    schema = logical_schema(
        predicate_policy(
            "acts_on", domain=["Organization"], range_=["Product"],
        ),
        version="schema-soft-source-types-v1",
    )
    mentions = [
        {
            "occurrence_key": "alpha",
            "mention_id": "alpha",
            "surface": "Alpha",
            "mention_start": 0,
            "mention_end": 5,
            "canonical_id": "canonical-alpha",
            "assignment_version": "assignment-v1",
            "type": "Person",
        },
        {
            "occurrence_key": "beta",
            "mention_id": "beta",
            "surface": "Beta",
            "mention_start": 15,
            "mention_end": 19,
            "canonical_id": "canonical-beta",
            "assignment_version": "assignment-v1",
            "type": "Location",
        },
    ]
    frame = {
        "relation": "connects",
        "arguments": ["Alpha", "Beta"],
        "roles": ["subject", "object"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Alpha", "Beta"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["alpha", "beta"],
        "evidence_span": [0, len(content)],
        "evidence_text": content,
    }
    opportunity = GroundedRelationRuntime._grounded_open_fact_opportunity(
        run_id="run-soft-source-types",
        project_id=1,
        chunk={
            "id": 1,
            "document_id": 1,
            "revision_id": "revision-soft-source-types",
            "chunk_key": "chunk-soft-source-types",
            "content": content,
            "content_hash": "content-soft-source-types",
        },
        frame=frame,
        mentions=mentions,
        logical_schema=schema,
    )

    assert opportunity is not None
    assert opportunity.status == v2.PENDING_CLASSIFICATION
    assert opportunity.type_status == "unresolved"
    assert opportunity.left_type == opportunity.right_type == ""
    assert opportunity.provenance["source_endpoint_types"] == [
        "Person", "Location",
    ]
    assert opportunity.provenance["source_type_constraint_mode"] == (
        "soft_source_fact_requires_verification"
    )
    assert {
        item["name"] for item in opportunity.allowed_predicate_metadata
    } == {"acts_on"}

    hypothesis = v2.source_frame_projection_hypotheses(
        {"coverage_complete": True, "frames": [frame]},
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "acts_on",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        occurrences=mentions,
        opportunities_by_pair={opportunity.pair_identity: opportunity},
        logical_schema=schema,
        classifier_model_revision="model-soft-source-types",
    )[0]
    validation = v2.validate_hypothesis(
        hypothesis, opportunity, content=content, logical_schema=schema,
    )
    assert validation.status == v2.PASS
    assert validation.reason_code == "TYPE_UNRESOLVED_REQUIRES_VERIFICATION"


def test_grounded_source_frame_uses_its_exact_cross_line_evidence_scope():
    content = "Item | Purpose\nAlpha | Beta"
    alpha_start = content.index("Alpha")
    beta_start = content.index("Beta")
    schema = logical_schema(
        predicate_policy("acts_on"),
        version="schema-grounded-frame-scope-v1",
    )
    mentions = [
        {
            "occurrence_key": "alpha",
            "mention_id": "alpha",
            "surface": "Alpha",
            "mention_start": alpha_start,
            "mention_end": alpha_start + len("Alpha"),
            "canonical_id": "canonical-alpha",
            "assignment_version": "assignment-v1",
            "type": "GenericEntity",
        },
        {
            "occurrence_key": "beta",
            "mention_id": "beta",
            "surface": "Beta",
            "mention_start": beta_start,
            "mention_end": beta_start + len("Beta"),
            "canonical_id": "canonical-beta",
            "assignment_version": "assignment-v1",
            "type": "GenericEntity",
        },
    ]
    frame = {
        "relation": "column role: Purpose",
        "arguments": ["Alpha", "Beta"],
        "roles": ["row_key", "column_value"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["Alpha", "Beta"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["alpha", "beta"],
        "evidence_span": [0, len(content)],
        "evidence_text": content,
        "deterministic_source_structure": "table_header_assignment",
        "structural_role_label": "Purpose",
    }
    opportunity = GroundedRelationRuntime._grounded_open_fact_opportunity(
        run_id="run-grounded-frame-scope",
        project_id=1,
        chunk={
            "id": 1,
            "document_id": 1,
            "revision_id": "revision-grounded-frame-scope",
            "chunk_key": "chunk-grounded-frame-scope",
            "content": content,
            "content_hash": "content-grounded-frame-scope",
        },
        frame=frame,
        mentions=mentions,
        logical_schema=schema,
    )

    assert opportunity is not None
    assert opportunity.provenance["required_atomic_scope"] == [
        0, len(content),
    ]
    assert opportunity.provenance["source_frame_atomic_scope_override"] is True
    hypothesis = v2.source_frame_projection_hypotheses(
        {"coverage_complete": True, "frames": [frame]},
        {
            "coverage_complete": True,
            "bindings": [{
                "frame_index": 0,
                "predicate": "acts_on",
                "subject_index": 0,
                "object_index": 1,
            }],
        },
        occurrences=mentions,
        opportunities_by_pair={opportunity.pair_identity: opportunity},
        logical_schema=schema,
        classifier_model_revision="model-grounded-frame-scope",
    )[0]
    validation = v2.validate_hypothesis(
        hypothesis,
        opportunity,
        content=content,
        logical_schema=schema,
    )
    assert validation.status == v2.PASS
    assert validation.reason_code == "DETERMINISTIC_VALIDATION_PASSED"


def test_relation_occurrence_keys_follow_source_blocks_without_cross_block_collisions():
    mention = {
        "mention_key": "mention-1",
        "mention_start": 2,
        "mention_end": 6,
        "provenance": {"physical_occurrence_key": "physical-1"},
    }
    first_context = {
        "revision_id": "revision-1",
        "id": 101,
        "shadow_source_map": [{
            "rendered_start": 0,
            "rendered_end": 20,
            "anchor_kind": "structured_chunk",
            "block_key": "source-block-a",
            "block_local_start": 40,
            "block_local_end": 60,
        }],
    }
    overlapping_context = {
        "revision_id": "revision-1",
        "id": 102,
        "shadow_source_map": [{
            "rendered_start": 0,
            "rendered_end": 20,
            "anchor_kind": "structured_chunk",
            "block_key": "source-block-a",
            "block_local_start": 40,
            "block_local_end": 60,
        }],
    }
    unrelated_context = {
        "revision_id": "revision-1",
        "id": 103,
        "shadow_source_map": [{
            "rendered_start": 0,
            "rendered_end": 20,
            "anchor_kind": "structured_chunk",
            "block_key": "source-block-b",
            "block_local_start": 40,
            "block_local_end": 60,
        }],
    }

    first = GroundedRelationRuntime._relation_occurrence_key(
        mention, first_context,
    )
    overlapping = GroundedRelationRuntime._relation_occurrence_key(
        mention, overlapping_context,
    )
    unrelated = GroundedRelationRuntime._relation_occurrence_key(
        mention, unrelated_context,
    )

    assert first == overlapping
    assert first != unrelated


def test_exact_argument_retry_keeps_the_alignment_contract_valid_json():
    class Gateway:
        prompt = ""

        def json_chat(self, **kwargs):
            self.prompt = kwargs["prompt"]
            payload = json.loads(self.prompt)
            entailed = payload["first_endpoint"]["surface"] == "甲"
            candidate_source = payload["first_endpoint"]
            candidate_target = payload["second_endpoint"]
            result = {
                "source_role": entailed,
                "target_role": entailed,
                "direct_relation": entailed,
                "source_argument_start": -1,
                "source_argument_end": -1,
                "source_argument_surface": "",
                "target_argument_start": -1,
                "target_argument_end": -1,
                "target_argument_surface": "",
            }
            if entailed:
                for label, candidate in (
                    ("source", candidate_source), ("target", candidate_target),
                ):
                    result[f"{label}_argument_start"] = candidate["absolute_span"][0]
                    result[f"{label}_argument_end"] = candidate["absolute_span"][1]
                    result[f"{label}_argument_surface"] = candidate["surface"]
            return result

    gateway = Gateway()
    runtime = GroundedRelationRuntime(None, gateway)
    alignment = json.dumps({
        "source_context": {"text": "甲使用乙。", "start": 0, "end": 5},
        "endpoint_a": {"surface": "甲", "span": [0, 1]},
        "endpoint_b": {"surface": "乙", "span": [3, 4]},
        "required_roles": {
            "relation_source_role": "使用者",
            "relation_target_role": "被使用对象",
        },
        "required_relation_semantics": {
            "semantic_definition": "主体使用客体。",
        },
        "context_roles": {
            "endpoint_a_role": "主体",
            "endpoint_b_role": "客体",
        },
    }, ensure_ascii=False)

    result = runtime._verify_exact_argument_alignment(
        model="test",
        context_budget_tokens=8192,
        prompt=alignment,
        attempt_no=2,
    )

    assert result == {
        "exact_argument_orientation": "FORWARD",
        "exact_argument_forward_supported": True,
        "exact_argument_reverse_supported": False,
        "exact_argument_protocol_errors": {},
    }
    assert json.loads(gateway.prompt)["protocol_retry"]


def test_final_verifier_retry_keeps_the_claim_contract_valid_json():
    class Gateway:
        prompt = ""

        def json_chat(self, **kwargs):
            self.prompt = kwargs["prompt"]
            payload = json.loads(self.prompt)
            assert payload["hypothesis_id"] == "hypothesis-1"
            return {"verdict": "UNCERTAIN"}

    gateway = Gateway()
    runtime = GroundedRelationRuntime(None, gateway)
    response = runtime._verify(
        model="test",
        context_budget_tokens=8192,
        prompt=json.dumps({"hypothesis_id": "hypothesis-1"}),
        response_schema={"type": "object"},
        attempt_no=2,
    )

    assert response == {"verdict": "UNCERTAIN"}
    retry = json.loads(gateway.prompt)["protocol_retry"]
    assert retry["attempt"] == 2
    assert "Re-evaluate" in retry["instruction"]


def test_atomic_publication_audit_uses_exact_keyed_fail_closed_contract():
    candidates = [{
        "candidate_id": "fact-1",
        "subject": "客户端数据",
        "source_expression": "满足",
        "canonical_predicate": "depends_on",
        "object": "独立同分布",
        "evidence": "客户端数据并不满足独立同分布。",
        "relation_layer": "schema_predicate",
        "schema_contract": {
            "definition": "主体依赖客体才能运行或存在。",
            "subject_role": "依赖方",
            "object_role": "被依赖对象",
            "direction": "依赖方指向被依赖对象",
        },
        "source_order_matches_proposed_fact": True,
        "resolved_reference_grounding": False,
    }]
    payload = json.loads(v2.atomic_publication_audit_prompt(candidates))
    assert payload["candidates"][0]["candidate_id"] == "fact-1"
    assert payload["candidates"][0]["schema_contract"]["definition"]
    schema = v2.atomic_publication_audit_response_schema(["fact-1"])
    assert schema["required"] == ["fact-1"]
    assert v2.parse_atomic_publication_audit(
        {"fact-1": "CONFLICT"}, candidate_ids=["fact-1"],
    ) == {"fact-1": "CONFLICT"}
    with pytest.raises(v2.RelationClassificationError):
        v2.parse_atomic_publication_audit(
            {"fact-1": "ACCEPT"}, candidate_ids=["fact-1"],
        )
    with pytest.raises(v2.RelationClassificationError):
        v2.parse_atomic_publication_audit(
            {"fact-1": "ALIGNED", "extra": "ALIGNED"},
            candidate_ids=["fact-1"],
        )

    direct_open = {
        **candidates[0],
        "relation_layer": "open_source_expression",
        "canonical_predicate": "source_assertion",
        "source_expression": "使用",
        "subject": "客户端",
        "object": "本地数据",
        "evidence": "客户端直接使用本地数据训练模型。",
        "schema_contract": {},
    }
    assert v2.atomic_publication_audit_requirement(direct_open) == (
        True, "open_source_semantic_role_confirmation",
    )
    assert v2.atomic_publication_audit_requirement({
        **direct_open,
        "evidence": "本地数据由客户端使用。",
    }) == (True, "ordered_source_geometry_missing")
    assert v2.atomic_publication_audit_requirement({
        **direct_open,
        "resolved_reference_grounding": True,
    }) == (True, "resolved_reference_role_mapping")
    assert v2.atomic_publication_audit_requirement({
        **direct_open,
        "object": "客户端",
        "evidence": "框架提升完整标注客户端权重。",
        "subject": "框架",
        "source_expression": "提升",
        "object_containing_mentions": ["客户端权重"],
    }) == (True, "endpoint_nested_in_source_mention")
    assert v2.atomic_publication_audit_requirement({
        **direct_open,
        "subject": "框架",
        "source_expression": "区分",
        "object": "可靠",
        "evidence": "框架区分可靠与低质量数据源。",
    }) == (True, "short_coordinated_endpoint")
    assert v2.atomic_publication_deterministic_rejection_reason({
        **direct_open,
        "subject": "框架",
        "subject_entity_type": "Technology",
        "source_expression": "区分",
        "object": "可靠",
        "object_entity_type": "GenericEntity",
        "evidence": "框架区分可靠与低质量数据源。",
    }) == "coordinated_elliptical_object_endpoint"
    assert v2.atomic_publication_deterministic_rejection_reason({
        **direct_open,
        "subject": "框架",
        "subject_entity_type": "Technology",
        "source_expression": "支持",
        "object": "数据库",
        "object_entity_type": "Technology",
        "evidence": "框架支持数据库和分布式文件系统。",
    }) == ""


def test_atomic_publication_audit_exposes_complete_reference_phrase_context():
    candidate = {
        "candidate_id": "fact-reference",
        "subject": "集中式学习",
        "source_expression": "难以充分利用",
        "canonical_predicate": "source_assertion",
        "object": "各级医疗机构",
        "evidence": "集中式学习难以充分利用这些分散数据进行模型训练。",
        "relation_layer": "open_source_expression",
        "schema_contract": {},
        "source_order_matches_proposed_fact": True,
        "resolved_reference_grounding": True,
        "resolved_reference_endpoints": [{
            "role": "object",
            "reference_text": "这些",
            "antecedent_surface": "各级医疗机构",
            "reference_span": [10, 12],
            "antecedent_span": [0, 6],
            "local_reference_context": (
                "集中式学习难以充分利用【这些】分散数据进行模型训练。"
            ),
        }],
    }

    payload = json.loads(v2.atomic_publication_audit_prompt([candidate]))

    reference = payload["candidates"][0]["resolved_reference_endpoints"][0]
    assert "【这些】分散数据" in reference[
        "local_reference_context"
    ]
    assert "complete phrase" in payload["instruction"]


def test_atomic_publication_audit_withholds_conflict_and_skips_structure():
    class Database:
        def __init__(self):
            self.calls = []

        def record_grounded_relation_source_model_call(self, run_id, payload):
            self.calls.append((run_id, payload))

    class Gateway:
        def json_chat(self, **kwargs):
            payload = json.loads(kwargs["prompt"])
            return {
                item["candidate_id"]: "CONFLICT"
                for item in payload["candidates"]
            }

    def row(fact_id, *, structure=""):
        return {
            "relation_key": fact_id,
            "relation_type": "depends_on",
            "evidence_text": "甲满足乙。",
            "_publication_audit_candidate": {
                "candidate_id": fact_id,
                "subject": "甲",
                "source_expression": "满足",
                "canonical_predicate": "depends_on",
                "object": "乙",
                "evidence": "甲满足乙。",
                "relation_layer": "schema_predicate",
                "schema_contract": {
                    "definition": "甲依赖乙。",
                    "subject_role": "依赖方",
                    "object_role": "被依赖对象",
                    "direction": "甲到乙",
                },
                "source_order_matches_proposed_fact": True,
                "resolved_reference_grounding": False,
            },
            "metadata": {
                "relation_semantics": {
                    "layer": "schema_predicate",
                    "source_structure_kind": structure,
                    "publication": {"eligible": True, "reason": "grounded"},
                },
            },
        }

    database = Database()
    unanchored = row("fact-unanchored")
    unanchored["_publication_audit_candidate"].update({
        "relation_candidate_origin": "source_projection",
        "schema_lexical_anchor_applicable": True,
        "schema_lexical_anchor": False,
    })
    elliptical = row("fact-elliptical")
    elliptical["_publication_audit_candidate"].update({
        "subject": "框架",
        "subject_entity_type": "Technology",
        "source_expression": "区分",
        "object": "可靠",
        "object_entity_type": "GenericEntity",
        "evidence": "框架区分可靠与低质量数据源。",
        "relation_layer": "open_source_expression",
        "canonical_predicate": "source_assertion",
        "schema_contract": {},
    })
    elliptical["metadata"]["relation_semantics"].update({
        "layer": "open_source_expression",
    })
    output, metrics = GroundedRelationRuntime(
        database, Gateway(),
    )._run_atomic_publication_audit(
        run_id="run-1",
        rows=[
            row("fact-model"),
            row("fact-structure", structure="table_row"),
            unanchored,
            elliptical,
        ],
        context={"model": "test", "model_revision": "r1"},
        context_budget_tokens=8192,
    )

    by_id = {item["relation_key"]: item for item in output}
    model_publication = by_id["fact-model"]["metadata"][
        "relation_semantics"
    ]["publication"]
    assert model_publication == {
        "eligible": False,
        "reason": "atomic_publication_role_conflict",
        "policy": v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION,
    }
    assert by_id["fact-structure"]["metadata"][
        "atomic_publication_audit"
    ]["status"] == "DETERMINISTIC_STRUCTURE_CONTRACT"
    assert by_id["fact-unanchored"]["metadata"][
        "relation_semantics"
    ]["publication"] == {
        "eligible": False,
        "reason": "schema_projection_without_lexical_anchor",
        "policy": v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION,
    }
    assert by_id["fact-unanchored"]["metadata"][
        "atomic_publication_audit"
    ]["status"] == "SCHEMA_LEXICAL_ANCHOR_MISSING"
    assert by_id["fact-elliptical"]["metadata"][
        "relation_semantics"
    ]["publication"]["reason"] == (
        "coordinated_elliptical_object_endpoint"
    )
    assert by_id["fact-elliptical"]["metadata"][
        "atomic_publication_audit"
    ]["status"] == "DETERMINISTIC_CONFLICT"
    assert metrics["candidate_facts"] == 1
    assert metrics["conflicts"] == 1
    assert metrics["deterministic_role_conflicts"] == 1
    assert metrics["deterministic_structure_contracts"] == 1
    assert metrics["schema_projections_without_lexical_anchor"] == 1
    assert database.calls[0][1]["component"] == "atomic_publication_audit"


def test_atomic_publication_audit_flushes_before_renderer_hard_limit():
    class Database:
        def __init__(self):
            self.calls = []

        def record_grounded_relation_source_model_call(self, run_id, payload):
            self.calls.append((run_id, payload))

    class Gateway:
        def json_chat(self, **kwargs):
            payload = json.loads(kwargs["prompt"])
            assert 1 <= len(payload["candidates"]) <= 16
            return {
                item["candidate_id"]: "ALIGNED"
                for item in payload["candidates"]
            }

    rows = []
    for index in range(17):
        fact_id = f"fact-{index:02d}"
        rows.append({
            "relation_key": fact_id,
            "relation_type": "acts_on",
            "evidence_text": "甲连接乙。",
            "_publication_audit_candidate": {
                "candidate_id": fact_id,
                "subject": "甲",
                "source_expression": "连接",
                "canonical_predicate": "acts_on",
                "object": "乙",
                "evidence": "甲连接乙。",
                "relation_layer": "schema_predicate",
                "schema_contract": {
                    "definition": "主体作用于客体。",
                    "subject_role": "作用方",
                    "object_role": "承受方",
                    "direction": "作用方指向承受方",
                },
                "source_order_matches_proposed_fact": True,
                "resolved_reference_grounding": False,
            },
            "metadata": {
                "relation_semantics": {
                    "layer": "schema_predicate",
                    "publication": {"eligible": True, "reason": "grounded"},
                },
            },
        })

    database = Database()
    output, metrics = GroundedRelationRuntime(
        database, Gateway(),
    )._run_atomic_publication_audit(
        run_id="run-large-audit",
        rows=rows,
        context={"model": "test", "model_revision": "r1"},
        context_budget_tokens=8192,
    )

    assert len(output) == 17
    assert metrics["candidate_facts"] == 17
    assert metrics["aligned"] == 17
    assert metrics["model_calls"] == 2
    assert len(database.calls) == 2


def _schema(action_name: str = "acts_on") -> dict:
    return logical_schema(
        predicate_policy(
            action_name,
            domain=["Organization"],
            range_=["Product"],
        ),
        predicate_policy(
            "linked", symmetric=True,
            semantic_definition="Symmetric fixture relation semantics.",
        ),
        predicate_policy(
            "contains",
            domain=["Container"],
            range_=["Part"],
            inverse_predicate="inside",
        ),
        predicate_policy(
            "inside",
            domain=["Part"],
            range_=["Container"],
            inverse_predicate="contains",
            canonical_orientation="inverse",
        ),
        predicate_policy("association", publish_mode="review_only"),
        version="schema-runtime-v1",
    )


class _ScriptedGateway:
    def __init__(self, action_name: str = "acts_on", *, fail_classifier=False):
        self.action_name = action_name
        self.fail_classifier = fail_classifier
        self.classifier_calls = 0
        self.role_probe_calls = 0
        self.role_alignment_calls = 0
        self.semantic_contrast_calls = 0
        self.atomic_fact_inventory_calls = 0
        self.exact_argument_calls = 0
        self.verifier_calls = 0

    @staticmethod
    def effective_model_context(_model):
        return SimpleNamespace(digest="scripted-model-digest")

    def json_chat(self, **kwargs):
        system = str(kwargs["system"])
        payload = json.loads(kwargs["prompt"])
        if "Audit direct ordered semantic roles" in system:
            return {
                item["candidate_id"]: "ALIGNED"
                for item in payload["candidates"]
            }
        if "proposer" in system:
            return {
                "proposals": [],
                "missing_endpoint_mentions": [],
                # The default fixture exercises the independent pair
                # classifier fallback. A complete atomic-scope response is
                # covered by dedicated proposer tests below.
                "coverage": {"complete": False, "uncovered_cues": []},
            }
        if "classifier" in system:
            self.classifier_calls += 1
            if self.fail_classifier:
                raise RuntimeError("scripted classifier failure")
            result = []
            for pair in payload["pairs"]:
                context = payload["context_catalog"][
                    pair["context_ref"]
                ]["text"]
                common = {
                    "pair_id": pair["pair_id"],
                    "hypotheses": [],
                }
                if "NO_CASE" in context:
                    result.append({
                        **common,
                        "decision": NO_RELATION,
                        "reason_code": "NO_SEMANTIC_RELATION",
                        "evidence_spans": [{
                            "start": pair["context_span"][0],
                            "end": pair["context_span"][1],
                        }],
                    })
                    continue
                if "OTHER_CASE" in context:
                    result.append({
                        **common,
                        "decision": OTHER_UNSUPPORTED,
                        "reason_code": "SCHEMA_GAP",
                        "evidence_spans": [{
                            "start": pair["context_span"][0],
                            "end": pair["context_span"][1],
                        }],
                    })
                    continue
                left = pair["left_occurrence_id"]
                right = pair["right_occurrence_id"]
                predicate = self.action_name
                subject, object_ = left, right
                hypotheses = []
                if "INVERSE_CASE" in context:
                    predicate = "inside"
                    subject = (
                        left if pair["left_type"] == "Part" else right
                    )
                    object_ = (
                        left if pair["left_type"] == "Container" else right
                    )
                elif "SYMMETRIC_CASE" in context:
                    predicate = "linked"
                elif "REVIEW_CASE" in context:
                    predicate = "association"
                spans = [{
                    "start": pair["context_span"][0],
                    "end": pair["context_span"][1],
                }]
                if "MULTI_EVIDENCE_CASE" in context:
                    spans = [
                        {"start": pair["left_span"][0], "end": pair["left_span"][1]},
                        {"start": pair["right_span"][0], "end": pair["right_span"][1]},
                    ]
                hypotheses.append({
                    "subject_occurrence_id": subject,
                    "predicate": predicate,
                    "object_occurrence_id": object_,
                    "evidence_spans": spans,
                    "classifier_confidence": 0.9,
                })
                if "MULTI_RELATION_CASE" in context:
                    hypotheses.append({
                        "subject_occurrence_id": subject,
                        "predicate": "linked",
                        "object_occurrence_id": object_,
                        "evidence_spans": spans,
                        "classifier_confidence": 0.8,
                    })
                result.append({
                    **common,
                    "decision": HYPOTHESES,
                    "hypotheses": hypotheses,
                })
            return {"items": result}
        if "semantic contrast judge" in system:
            self.semantic_contrast_calls += 1
            return {"semantic_contrast": "DEFINITION"}
        if "semantic-frame to versioned-Schema role mapper" in system:
            alignments = []
            symmetric = bool(payload["schema_relation"].get("symmetric"))
            for frame in payload["semantic_frames"]:
                if len(frame["arguments"]) < 2:
                    continue
                alignments.append({
                    "frame_index": frame["frame_index"],
                    "subject_index": 0,
                    "object_index": 1,
                })
                if symmetric:
                    alignments.append({
                        "frame_index": frame["frame_index"],
                        "subject_index": 1,
                        "object_index": 0,
                    })
            return {
                "coverage_complete": True,
                "alignments": alignments,
            }
        if (
            "atomic proposition analyst" in system
            or "原子事实分析器" in system
        ):
            self.atomic_fact_inventory_calls += 1
            context = str((payload.get("context") or {}).get("text") or "")
            left = re.search(r"Left\d+", context)
            right = re.search(r"Right\d+", context)
            items = []
            if left and right:
                items.append({
                    "subject_surface": left.group(0),
                    "relation_text": "connects",
                    "object_surface": right.group(0),
                })
            return {"coverage_complete": True, "items": items}
        if (
            "directional textual-entailment verifier" in system
            or "semantic-role mapping auditor" in system
        ):
            self.exact_argument_calls += 1
            source_text = str(payload.get("source_text") or "")
            candidate_source = str(
                (payload.get("first_endpoint") or {}).get("surface") or ""
            )
            if (
                "SYMMETRIC_CASE" in source_text
                or str(payload.get("definition") or "").startswith("Symmetric")
            ):
                entailed = True
            else:
                entailed = candidate_source.startswith("Left")
            result = {
                "source_role": entailed,
                "target_role": entailed,
                "direct_relation": entailed,
                "source_argument_start": -1,
                "source_argument_end": -1,
                "source_argument_surface": "",
                "target_argument_start": -1,
                "target_argument_end": -1,
                "target_argument_surface": "",
            }
            if entailed:
                for label in ("source", "target"):
                    candidate = payload[
                        "first_endpoint" if label == "source" else "second_endpoint"
                    ]
                    result[f"{label}_argument_start"] = candidate["absolute_span"][0]
                    result[f"{label}_argument_end"] = candidate["absolute_span"][1]
                    result[f"{label}_argument_surface"] = candidate["surface"]
            return result
        if "role alignment judge" in system:
            self.role_alignment_calls += 1
            symmetric = bool(payload.get("symmetric"))
            return {
                "endpoint_a_vs_source": "MATCH",
                "endpoint_b_vs_target": "MATCH",
                "endpoint_b_vs_source": "MATCH" if symmetric else "MISMATCH",
                "endpoint_a_vs_target": "MATCH" if symmetric else "MISMATCH",
                "forward_relation_vs_definition": "MATCH",
                "reverse_relation_vs_definition": (
                    "MATCH" if symmetric else "MISMATCH"
                ),
                "semantic_contrast": "DEFINITION",
            }
        if "role analyst" in system:
            self.role_probe_calls += 1
            endpoint_a = payload["endpoint_a"]
            endpoint_b = payload["endpoint_b"]
            return {
                "endpoint_a_complete_argument": True,
                "endpoint_b_complete_argument": True,
                "endpoint_a_argument_start": endpoint_a["span"][0],
                "endpoint_a_argument_end": endpoint_a["span"][1],
                "endpoint_b_argument_start": endpoint_b["span"][0],
                "endpoint_b_argument_end": endpoint_b["span"][1],
                "direct_relation_present": True,
                "subject_role": "scripted source endpoint role",
                "object_role": "scripted target endpoint role",
                "expressed_relation": "Left endpoint directly acts on Right endpoint.",
            }
        if system == "Audit exact source roles. Return only JSON.":
            return {"projection_status": "ALIGNED"}
        if system == "Audit exact source roles jointly. Return only JSON.":
            return {
                item["hypothesis_id"]: "ALIGNED"
                for item in payload["candidates"]
            }
        if isinstance(payload.get("claims"), list):
            self.verifier_calls += 1
            decisions = []
            for claim in payload["claims"]:
                context = payload["context_catalog"][
                    claim["context_ref"]
                ]["text"]
                relation_policy = payload["relation_policy_catalog"][
                    claim["relation_policy_ref"]
                ]
                verdict = SUPPORTED
                if "VERIFIER_REJECT_CASE" in context:
                    verdict = REJECTED
                elif "VERIFIER_UNCERTAIN_CASE" in context:
                    verdict = UNCERTAIN
                decisions.append({
                    "hypothesis_id": claim["hypothesis_id"],
                    "verdict": verdict,
                    "reason_code": f"SCRIPTED_{verdict}",
                    "forward_entailment": (
                        "ENTAILED" if verdict == SUPPORTED else "UNCERTAIN"
                    ),
                    "reverse_entailment": (
                        "ENTAILED"
                        if (
                            verdict == SUPPORTED
                            and relation_policy["symmetric"]
                        )
                        else "NOT_ENTAILED"
                        if verdict == SUPPORTED else "UNCERTAIN"
                    ),
                    "subject_role_supported": verdict == SUPPORTED,
                    "object_role_supported": verdict == SUPPORTED,
                    "supported_evidence_spans": (
                        claim["classifier_evidence_spans"]
                        if verdict == SUPPORTED else []
                    ),
                })
            return {"decisions": decisions}
        self.verifier_calls += 1
        context = payload["context"]["text"]
        verdict = SUPPORTED
        if "VERIFIER_REJECT_CASE" in context:
            verdict = REJECTED
        elif "VERIFIER_UNCERTAIN_CASE" in context:
            verdict = UNCERTAIN
        return {
            "hypothesis_id": payload["hypothesis_id"],
            "verdict": verdict,
            "reason_code": f"SCRIPTED_{verdict}",
            "forward_entailment": (
                "ENTAILED" if verdict == SUPPORTED else "UNCERTAIN"
            ),
            "reverse_entailment": (
                "ENTAILED"
                if (
                    verdict == SUPPORTED
                    and payload["relation_policy"]["symmetric"]
                )
                else "NOT_ENTAILED" if verdict == SUPPORTED else "UNCERTAIN"
            ),
            "subject_role_supported": verdict == SUPPORTED,
            "object_role_supported": verdict == SUPPORTED,
            "supported_evidence_spans": (
                payload["classifier_evidence_spans"]
                if verdict == SUPPORTED else []
            ),
        }


class _RuntimeDatabase:
    def __init__(self, *, action_name="acts_on", stale=False):
        self.action_name = action_name
        self.stale = stale
        self.project_reads = 0
        self.schema_ready_checks = 0
        self.runs = []
        self.opportunities = []
        self.classification_attempts = []
        self.decisions = []
        self.validations = []
        self.verification_attempts = []
        self.normalization = None
        self.materialization_decisions = []
        self.finished = []
        self.endpoints = []
        self.source_contexts = []
        self.proposal_attempts = []
        cases = [
            ("NO_CASE", "Organization", "Product", "no-left", "no-right"),
            ("OTHER_CASE", "Organization", "Product", "other-left", "other-right"),
            (
                "VERIFIER_REJECT_CASE", "Organization", "Product",
                "reject-left", "reject-right",
            ),
            (
                "VERIFIER_UNCERTAIN_CASE", "Organization", "Product",
                "uncertain-left", "uncertain-right",
            ),
            ("INVERSE_CASE", "Part", "Container", "inverse-part", "inverse-box"),
            (
                "SYMMETRIC_CASE", "Organization", "Product",
                "symmetric-left", "symmetric-right",
            ),
            (
                "MULTI_RELATION_CASE", "Organization", "Product",
                "multi-left", "multi-right",
            ),
            (
                "SELF_LOOP_CASE", "Organization", "Product",
                "self-left", "self-right",
            ),
            (
                "MULTI_EVIDENCE_CASE", "Organization", "Product",
                "evidence-left", "evidence-right",
            ),
            (
                "REVIEW_CASE", "Organization", "Product",
                "review-left", "review-right",
            ),
            (
                "POSITIVE_CASE", "Organization", "Product",
                "positive-left", "positive-right",
            ),
        ]
        self.chunks = []
        self.mentions = []
        self.assignments = []
        for index, (tag, left_type, right_type, left_key, right_key) in enumerate(
            cases, start=1,
        ):
            left_surface = f"Left{index}"
            right_surface = f"Right{index}"
            content = f"{tag} {left_surface} connects {right_surface}."
            left_start = content.index(left_surface)
            right_start = content.index(right_surface)
            self.chunks.append({
                "id": index,
                "document_id": 1,
                "revision_id": "revision-runtime",
                "chunk_key": f"chunk-{index}",
                "chunk_index": index,
                "content": content,
                "content_hash": f"chunk-hash-{index}",
            })
            same_canonical = tag == "SELF_LOOP_CASE"
            for key, surface, start, entity_type, canonical in (
                (
                    left_key, left_surface, left_start, left_type,
                    f"canonical-{index}-left",
                ),
                (
                    right_key, right_surface, right_start, right_type,
                    (
                        f"canonical-{index}-left"
                        if same_canonical else f"canonical-{index}-right"
                    ),
                ),
            ):
                self.mentions.append({
                    "mention_key": key,
                    "chunk_id": index,
                    "surface": surface,
                    "evidence_start": start,
                    "evidence_end": start + len(surface),
                })
                self.assignments.append({
                    "mention_key": key,
                    "canonical_key": canonical,
                    "entity_type": entity_type,
                    "policy_version": "assignment-v1",
                    "revision_id": "revision-runtime",
                })

    def assert_grounded_relation_ready(self):
        self.schema_ready_checks += 1
        return {"migrations": ["0014", "0015", "0016", "0017", "0018", "0019"]}

    def get_project(self, project_id):
        assert project_id == 1
        self.project_reads += 1
        return {
            "id": 1,
            "project_key": "runtime-project",
            "graph_generation": 2 if self.stale and self.project_reads > 1 else 1,
            "llm_model": "scripted-model",
            "model_max_tokens": 32768,
        }

    @staticmethod
    def list_documents(project_id):
        assert project_id == 1
        return [{
            "id": 1,
            "project_id": 1,
            "status": "indexed",
            "active_revision_id": "revision-runtime",
            "doc_hash": "document-hash-runtime",
            "is_deleted": False,
        }]

    def list_document_chunks(self, document_id):
        assert document_id == 1
        return deepcopy(self.chunks)

    def get_active_schema(self, project_id):
        assert project_id == 1
        return {"version": 1, "schema_json": _schema(self.action_name)}

    def list_grounded_entity_mentions(self, namespace_id):
        assert namespace_id == "namespace-runtime"
        return deepcopy(self.mentions)

    def load_grounded_entity_snapshot(self, namespace_id):
        assert namespace_id == "namespace-runtime"
        return {
            "namespace": {
                "id": namespace_id,
                "config_hash": "entity-config",
                "policy_set": {
                    "structure_artifact_hash": getattr(
                        self, "structure_artifact_hash", "",
                    ),
                },
            },
            "assignments": deepcopy(self.assignments),
            "canonicals": [],
            "relation_endpoints": [],
        }

    def start_grounded_relation_run(self, values):
        self.runs.append(values)
        return values

    def persist_grounded_relation_source_contexts(self, run_id, values):
        self.source_contexts.extend((run_id, item) for item in values)
        return values

    def record_grounded_relation_proposal_attempt(self, run_id, values):
        self.proposal_attempts.append((run_id, values))
        return values

    def persist_grounded_relation_opportunities(
        self,
        _run_id,
        values,
        *,
        mark_classifying=True,
        include_persisted_rows=True,
    ):
        self.opportunities = list(getattr(self, "opportunities", []))
        self.opportunities.extend(values)
        return []

    def record_grounded_relation_classification_attempt(
        self, run_id, pair_id, values,
    ):
        self.classification_attempts.append((run_id, pair_id, values))
        return values

    def persist_grounded_relation_classifications(self, _run_id, values):
        self.decisions.extend(values)
        return []

    def persist_grounded_relation_validations(self, _run_id, values):
        self.validations.extend(values)
        return []

    def persist_grounded_relation_verification_attempts(
        self, _run_id, values, *, terminal=None,
    ):
        self.verification_attempts.extend(values)
        self.verification_terminal_flags = list(
            getattr(self, "verification_terminal_flags", [])
        )
        self.verification_terminal_flags.extend(
            (
                item.verdict != "RETRYABLE_FAILED"
                if terminal is None else bool(terminal)
            )
            for item in values
        )
        return []

    def persist_grounded_relation_proposed_hypotheses(
        self, _run_id, values, *, candidate_source,
    ):
        self.proposed_hypotheses = list(getattr(self, "proposed_hypotheses", []))
        self.proposed_hypotheses.extend(values)
        self.proposed_candidate_source = candidate_source
        return []

    def persist_grounded_relation_normalization(
        self, _run_id, result, *, input_fingerprint, materialization_decisions,
    ):
        assert input_fingerprint
        self.normalization = result
        self.materialization_decisions = list(materialization_decisions)
        return {}

    def finish_grounded_relation_run(self, run_id, **values):
        self.finished.append((run_id, values))
        return values

    def persist_grounded_entity_relation_endpoints(self, namespace_id, rows):
        assert namespace_id == "namespace-runtime"
        self.endpoints.extend(rows)


def _keep_runtime_chunks(database: _RuntimeDatabase, chunk_ids: set[int]) -> None:
    database.chunks = [
        item for item in database.chunks if int(item["id"]) in chunk_ids
    ]
    database.mentions = [
        item for item in database.mentions
        if int(item["chunk_id"]) in chunk_ids
    ]
    mention_keys = {str(item["mention_key"]) for item in database.mentions}
    database.assignments = [
        item for item in database.assignments
        if str(item["mention_key"]) in mention_keys
    ]


def test_runtime_fingerprint_uses_process_loaded_code_without_rereading_files(
    monkeypatch,
):
    database = _RuntimeDatabase()
    runtime = GroundedRelationRuntime(database, _ScriptedGateway())
    first = runtime._context(1, "namespace-runtime")["fingerprint"]

    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("freshness check reread source file"),
        ),
    )
    second = runtime._context(1, "namespace-runtime")["fingerprint"]

    assert second.root == first.root
    assert (
        second.components["materialization_root"]["code_tree"]
        == first.components["materialization_root"]["code_tree"]
    )
    assert second.components["materialization_root"]["treatment"]


def test_runtime_treatment_digest_includes_deployment_thresholds(monkeypatch):
    before = relation_runtime_treatment_digest()

    monkeypatch.setattr(
        Config,
        "relation_verifier_batch_size",
        int(Config.relation_verifier_batch_size) + 1,
    )

    assert relation_runtime_treatment_digest() != before


def test_runtime_policy_fingerprints_open_fact_contracts():
    assert GROUNDED_RELATION_POLICY_SET["open_fact_prompt_version"] == (
        v2.GROUNDED_RELATION_OPEN_FACT_PROMPT_VERSION
    )
    assert GROUNDED_RELATION_POLICY_SET["open_fact_parser_version"] == (
        v2.GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION
    )

    context = GroundedRelationRuntime(
        _RuntimeDatabase(), _ScriptedGateway(),
    )._context(1, "namespace-runtime")
    assert context["fingerprint"].components["proposer"][
        "open_fact_parser_version"
    ] == v2.GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION
    assert GROUNDED_RELATION_POLICY_SET[
        "coordinated_recovery_policy_version"
    ] == v2.GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION
    assert context["fingerprint"].components["verifier"][
        "coordinated_recovery_policy_version"
    ] == v2.GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION


def test_persisted_call_replay_bypasses_cross_run_model_cache():
    class DatabaseWithCache(_RuntimeDatabase):
        @staticmethod
        def find_reusable_relation_model_call(**_kwargs):
            return {"id": "must-not-be-used"}

    replay_gateway = _ScriptedGateway()
    replay_gateway.replay_provenance = {"mode": "persisted-replay"}
    replay_runtime = GroundedRelationRuntime(
        DatabaseWithCache(), replay_gateway,
    )
    assert replay_runtime._reusable_model_call_lookup() is None

    ordinary_runtime = GroundedRelationRuntime(
        DatabaseWithCache(), _ScriptedGateway(),
    )
    assert callable(ordinary_runtime._reusable_model_call_lookup())


def test_runtime_uses_same_structured_payload_as_entity_stage():
    database = _RuntimeDatabase()
    blocks = [{
        "key": f"block-{item['id']}",
        "block_type": "paragraph",
        "text": item["content"],
    } for item in database.chunks]
    payload = {"parser_version": "parser-test", "blocks": blocks}
    manifest = build_evidence_context_manifest(
        namespace_id="relation-shadow:1:namespace-runtime",
        documents=database.list_documents(1),
        chunks_by_document={1: database.list_document_chunks(1)},
        payload_loader=lambda _document: payload,
    )
    database.structure_artifact_hash = manifest.artifact_hash

    context = GroundedRelationRuntime(
        database,
        _ScriptedGateway(),
        processing_payload_loader=lambda _document: payload,
    )._context(1, "namespace-runtime")

    assert context["shadow_context"].summary["legacy_chunk_local_units"] == 0
    assert {
        item["shadow_context_quality"] for item in context["relation_chunks"]
    } == {"structured_chunk_mapped"}


def test_runtime_rejects_entity_relation_structure_hash_mismatch():
    database = _RuntimeDatabase()
    database.structure_artifact_hash = "stale-structure-hash"

    with pytest.raises(RuntimeError, match="结构上下文不一致"):
        GroundedRelationRuntime(
            database,
            _ScriptedGateway(),
            processing_payload_loader=lambda _document: {
                "parser_version": "parser-test",
                "blocks": [{
                    "key": "block-1",
                    "block_type": "paragraph",
                    "text": database.chunks[0]["content"],
                }],
            },
        )._context(1, "namespace-runtime")


def test_classifier_preflight_uses_gateway_estimator_and_runtime_context():
    class CharacterEstimator:
        @staticmethod
        def estimate(value):
            return len(value)

    gateway = SimpleNamespace(
        token_estimator=CharacterEstimator(),
        effective_model_context=lambda _model: SimpleNamespace(
            effective_context_tokens=5000,
        ),
    )
    runtime = GroundedRelationRuntime(object(), gateway)

    assert runtime._classifier_request_fits(
        model="test-model",
        context_budget_tokens=5000,
        max_output_tokens=128,
        prompt="short",
        response_schema={"type": "object"},
    )
    assert not runtime._classifier_request_fits(
        model="test-model",
        context_budget_tokens=5000,
        max_output_tokens=128,
        prompt="x" * 10000,
        response_schema={"type": "object"},
    )


def test_runtime_executes_every_v2_stage_and_never_treats_classified_as_verified():
    database = _RuntimeDatabase()
    gateway = _ScriptedGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert database.schema_ready_checks == 1
    assert len(database.opportunities) == 11
    assert metrics["pair_statuses"]["REJECTED_SELF_LOOP"] == 1
    assert metrics["classification_decisions"] == {
        HYPOTHESES: 8,
        NO_RELATION: 1,
        OTHER_UNSUPPORTED: 1,
    }
    assert metrics["pair_decision_statuses"] == {
        "EXPLICIT_NO_RELATION": 1,
        "HYPOTHESES_EMITTED": 8,
        "SCHEMA_GAP": 1,
    }
    assert metrics["classified_hypotheses"] == 9
    assert metrics["batch_verifier_calls"] == 1
    assert metrics["batch_verifier_decisions"] == 8
    assert metrics["batch_verifier_failures"] == 0
    assert metrics["batch_verifier_partial_batches"] == 0
    assert metrics["batch_verifier_partial_decisions"] == 0
    assert metrics["batch_verifier_fallbacks"] == 0
    assert metrics["verification_mode"] == "lean"
    assert metrics["atomic_fact_inventory_model_calls"] == 0
    assert metrics["atomic_fact_inventory_cached_contexts"] == 0
    assert metrics["atomic_fact_inventory_cache_hits"] == 0
    assert metrics["atomic_fact_inventory_incomplete"] == 0
    assert metrics["atomic_fact_inventory_failures"] == 0
    assert metrics["atomic_fact_inventory_vetoes"] == 0
    assert metrics["verification_verdicts"] == {
        REJECTED: 1,
        SUPPORTED: 6,
        UNCERTAIN: 2,
    }
    assert metrics["verifier_supported"] == 6
    assert metrics["facts_after_normalization"] == 6
    assert metrics["facts_predicate_publication_suppressed"] == 6
    assert metrics["facts_materialization_eligible"] == 0
    assert metrics["legacy_pipeline_calls"] == 0
    assert len(database.endpoints) == 0
    mention_keys = {str(item["mention_key"]) for item in database.mentions}
    assert all(row["source_mention_key"] in mention_keys for row in database.endpoints)
    assert all(row["target_mention_key"] in mention_keys for row in database.endpoints)
    assert all(
        not str(row["source_mention_key"]).startswith("occurrence_")
        and not str(row["target_mention_key"]).startswith("occurrence_")
        for row in database.endpoints
    )
    assert all(
        row["metadata"]["source_occurrence_key"]
        and row["metadata"]["target_occurrence_key"]
        for row in database.endpoints
    )
    assert all(
        row["metadata"]["verification_status"] == "VERIFIER_SUPPORTED"
        for row in database.endpoints
    )
    assert all(
        evidence.verifier_supported
        for evidence in database.normalization.evidence
    )
    inverse = next(
        fact for fact in database.normalization.facts
        if fact.predicate == "contains"
    )
    assert inverse.subject_canonical_id == "canonical-5-right"
    assert inverse.object_canonical_id == "canonical-5-left"
    assert any(fact.symmetric for fact in database.normalization.facts)
    assert any(len(item.evidence_spans) == 2
               for item in database.normalization.evidence)


def test_runtime_source_first_skips_redundant_proposer_and_pair_classifier(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 1)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)

    class SourceFirstDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            content = (
                "POSITIVE_CASE Left11 connects Right11 while Third11 waits."
            )
            self.chunks[0].update({
                "content": content,
                "content_hash": "source-first-backstop-content-hash",
            })
            for mention in self.mentions:
                start = content.index(str(mention["surface"]))
                mention["evidence_start"] = start
                mention["evidence_end"] = start + len(
                    str(mention["surface"])
                )
            third_start = content.index("Third11")
            self.mentions.append({
                "mention_key": "positive-third",
                "chunk_id": 11,
                "surface": "Third11",
                "evidence_start": third_start,
                "evidence_end": third_start + len("Third11"),
            })
            self.assignments.append({
                "mention_key": "positive-third",
                "canonical_key": "canonical-11-third",
                "entity_type": "Organization",
                "policy_version": "assignment-v1",
                "revision_id": "revision-runtime",
            })
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class SourceFirstGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_calls = 0

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                if "coverage_audit" in payload:
                    return {"facts": []}
                return {"facts": [{
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0001",
                    "evidence": str(payload["source"]).split(" while ", 1)[0],
                }, {
                    # One malformed primary row makes the initial inventory
                    # incomplete. The explicit recovery audit above repairs
                    # the final-union completeness state without repeating the
                    # already accepted fact.
                    "subject_id": "UNKNOWN",
                    "predicate": "invalid",
                    "object_id": "M0002",
                    "evidence": str(payload["source"]),
                }]}
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": 0,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    }],
                }
            if "proposer" in system:
                self.proposer_calls += 1
            return super().json_chat(**kwargs)

    database = SourceFirstDatabase()
    gateway = SourceFirstGateway()
    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert gateway.proposer_calls == 0
    assert gateway.classifier_calls == 0
    assert metrics["source_first_primary_contexts"] == 1
    assert metrics["proposer_scopes_skipped_by_source_first"] == 1
    assert metrics["source_frame_projection_hypotheses"] == 1
    assert metrics["source_inventory_audited_contexts"] == 1
    assert metrics["source_inventory_complete_contexts"] == 1
    assert metrics["source_open_fact_gleaning_rounds_completed"] == 1
    assert metrics["source_first_classifier_backstop_pairs"] == 0
    assert metrics["source_first_classifier_pairs_skipped"] == 2
    assert metrics["pair_opportunities_classified"] == 0
    assert metrics["facts_after_normalization"] == 1
    assert sum(
        item.status == v2.SOURCE_FIRST_BACKSTOP_SKIPPED
        for item in database.opportunities
    ) == 2
    assert database.source_model_calls


def test_runtime_suppresses_reverse_only_omission_recovery(monkeypatch):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 1)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_reference_adjudication_enabled", False)

    class ReverseRecoveryDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            content = (
                "POSITIVE_CASE Left11 connects Right11 while Third11 waits."
            )
            self.chunks[0].update({
                "content": content,
                "content_hash": "reverse-recovery-content-hash",
            })
            for mention in self.mentions:
                start = content.index(str(mention["surface"]))
                mention["evidence_start"] = start
                mention["evidence_end"] = start + len(
                    str(mention["surface"])
                )
            third_start = content.index("Third11")
            self.mentions.append({
                "mention_key": "positive-third",
                "chunk_id": 11,
                "surface": "Third11",
                "evidence_start": third_start,
                "evidence_end": third_start + len("Third11"),
            })
            self.assignments.append({
                "mention_key": "positive-third",
                "canonical_key": "canonical-11-third",
                "entity_type": "Organization",
                "policy_version": "assignment-v1",
                "revision_id": "revision-runtime",
            })
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class ReverseRecoveryGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                evidence = str(payload["source"]).split(" while ", 1)[0]
                if "coverage_audit" in payload:
                    return {"facts": [{
                        "subject_id": "M0001",
                        "predicate": "connects",
                        "object_id": "M0000",
                        "evidence": evidence,
                    }]}
                return {"facts": [{
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0001",
                    "evidence": evidence,
                }, {
                    "subject_id": "UNKNOWN",
                    "predicate": "invalid",
                    "object_id": "M0002",
                    "evidence": str(payload["source"]),
                }]}
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": 0,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    }],
                }
            return super().json_chat(**kwargs)

    database = ReverseRecoveryDatabase()
    result = GroundedRelationRuntime(
        database, ReverseRecoveryGateway(),
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["source_open_fact_reverse_recoveries_suppressed"] == 1
    assert metrics["source_open_facts_recovered"] == 0
    assert metrics["facts_after_normalization"] == 1
    assert database.normalization.metrics[
        "unsupported_reverse_conflicts"
    ] == 0


def test_runtime_routes_partial_source_repair_to_exact_pair(monkeypatch):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 1)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_reference_adjudication_enabled", False)
    monkeypatch.setattr(
        Config, "relation_source_binding_recall_alternatives", 0,
    )

    class RepairDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            content = (
                "POSITIVE_CASE Left11 connects Right11 while Third11 waits."
            )
            self.chunks[0].update({
                "content": content,
                "content_hash": "source-repair-routing-content-hash",
            })
            for mention in self.mentions:
                start = content.index(str(mention["surface"]))
                mention["evidence_start"] = start
                mention["evidence_end"] = start + len(
                    str(mention["surface"])
                )
            third_start = content.index("Third11")
            self.mentions.append({
                "mention_key": "positive-third",
                "chunk_id": 11,
                "surface": "Third11",
                "evidence_start": third_start,
                "evidence_end": third_start + len("Third11"),
            })
            self.assignments.append({
                "mention_key": "positive-third",
                "canonical_key": "canonical-11-third",
                "entity_type": "Organization",
                "policy_version": "assignment-v1",
                "revision_id": "revision-runtime",
            })
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class RepairGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                if "coverage_audit" in payload:
                    return {"facts": [{
                        "subject_id": "M0001",
                        "predicate": "waits near",
                        "object_id": "M0002",
                        "evidence": "not copied from source",
                    }]}
                return {"facts": [{
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0001",
                    "evidence": str(payload["source"]).split(" while ", 1)[0],
                }]}
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": 0,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    }],
                }
            return super().json_chat(**kwargs)

    database = RepairDatabase()
    gateway = RepairGateway()
    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["source_inventory_audited_contexts"] == 1
    assert metrics["source_inventory_complete_contexts"] == 0
    assert metrics["source_repair_classifier_backstop_pairs"] == 1
    assert metrics["source_first_classifier_backstop_pairs"] == 1
    assert metrics["source_first_classifier_pairs_skipped"] == 1
    assert metrics["pair_opportunities_classified"] == 1
    assert gateway.classifier_calls == 1
    assert sum(
        item.status == v2.SOURCE_FIRST_BACKSTOP_SKIPPED
        for item in database.opportunities
    ) == 1
    assert sum(
        item.provenance.get("source_repair_backstop") is True
        for item in database.opportunities
    ) == 1


def test_runtime_failed_source_omission_pass_keeps_pair_classifier(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 1)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_reference_adjudication_enabled", False)

    class OmissionFailureDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class OmissionFailureGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.source_inventory_requests = 0

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            if "source-grounded open fact-frame extractor" in system:
                self.source_inventory_requests += 1
                if self.source_inventory_requests == 1:
                    return {"facts": []}
                raise RuntimeError("scripted omission pass failure")
            if "source-frame to Schema binding judge" in system:
                return {"coverage_complete": True, "bindings": []}
            return super().json_chat(**kwargs)

    database = OmissionFailureDatabase()
    gateway = OmissionFailureGateway()
    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert gateway.source_inventory_requests == 3
    assert metrics["source_open_fact_gleaning_failures"] == 2
    assert metrics["source_inventory_audited_contexts"] == 0
    assert metrics["source_inventory_complete_contexts"] == 0
    assert metrics["source_first_classifier_pairs_skipped"] == 0
    assert metrics["pair_opportunities_classified"] == 1
    assert gateway.classifier_calls == 1


def test_runtime_parser_proved_table_binding_uses_deterministic_terminal(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(
        Config, "relation_source_binding_recall_alternatives", 0,
    )

    structure_schema = logical_schema(
        predicate_policy(
            "derived_from",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition=(
                "The subject row item is derived from the object source."
            ),
            direction_description="row item to source",
            structural_evidence_kinds=["table_header_assignment"],
            structural_role_labels=["Source"],
        ),
        version="schema-runtime-structure-v1",
    )

    class StructureDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            content = (
                "| Item | Source |\n"
                "| --- | --- |\n"
                "| Left11 | Right11 |\n"
            )
            self.chunks[0].update({
                "content": content,
                "content_hash": "structure-content-hash",
            })
            for mention in self.mentions:
                surface = str(mention["surface"])
                start = content.index(surface)
                mention["evidence_start"] = start
                mention["evidence_end"] = start + len(surface)
            self.source_model_calls = []

        @staticmethod
        def get_active_schema(project_id):
            assert project_id == 1
            return {"version": 1, "schema_json": structure_schema}

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class StructureGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.structure_audit_calls = 0

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            if "source-grounded open fact-frame extractor" in system:
                return {"resolved_references": [], "facts": []}
            if "source-frame to Schema binding judge" in system:
                return {"coverage_complete": True, "bindings": []}
            if "Audit exact parser-proved source structure" in system:
                self.structure_audit_calls += 1
                return {"structure_status": "ALIGNED"}
            return super().json_chat(**kwargs)

    database = StructureDatabase()
    gateway = StructureGateway()
    result = GroundedRelationRuntime(
        database, gateway,
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["source_structure_bindings_total"] == 1
    assert metrics["source_structure_audit_calls"] == 0
    assert metrics["source_structure_audit_aligned"] == 1
    assert metrics["source_structure_audit_conflicts"] == 0
    assert metrics["source_structure_audit_uncertain"] == 0
    assert metrics["source_structure_audit_failures"] == 0
    assert metrics["deterministic_structure_contract_checks"] == 1
    assert metrics["deterministic_structure_contract_supported"] == 1
    assert metrics["deterministic_structure_contract_rejected"] == 0
    assert metrics["facts_after_normalization"] == 1
    assert gateway.structure_audit_calls == 0
    terminal = database.verification_attempts[-1]
    assert terminal.verdict == SUPPORTED
    assert terminal.verifier_prompt_version == (
        v2.GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION
    )
    assert not any(
        payload["component"] == "source_structure_binding_audit"
        for _run_id, payload in database.source_model_calls
    )


def test_runtime_contrastively_suppresses_schema_conflicting_source_binding(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)

    conflict_schema = logical_schema(
        predicate_policy(
            "topical_link",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition="The subject content is about the object.",
            conflicts_with=["functional_use"],
        ),
        predicate_policy(
            "functional_use",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition="The subject is used to accomplish the object.",
            conflicts_with=["topical_link"],
        ),
        version="schema-runtime-conflict-v1",
    )

    class ConflictDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            self.source_model_calls = []

        @staticmethod
        def get_active_schema(project_id):
            assert project_id == 1
            return {"version": 1, "schema_json": conflict_schema}

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class ConflictGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                return {"facts": [{
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0001",
                    "evidence": str(payload["source"]),
                }]}
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [
                        {
                            "frame_index": 0,
                            "predicate": "topical_link",
                            "subject_index": 0,
                            "object_index": 1,
                        },
                        {
                            "frame_index": 0,
                            "predicate": "functional_use",
                            "subject_index": 0,
                            "object_index": 1,
                        },
                    ],
                }
            if "contrastive relation-policy judge" in system:
                assert {
                    item["relation"] for item in payload["candidates"]
                } == {"topical_link", "functional_use"}
                return {"relations": [{"relation": "topical_link"}]}
            return super().json_chat(**kwargs)

    database = ConflictDatabase()
    result = GroundedRelationRuntime(
        database, ConflictGateway(),
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["source_conflict_groups"] == 1
    assert metrics["source_conflict_model_calls"] == 1
    assert metrics["source_conflict_failures"] == 0
    assert metrics["source_conflict_suppressed"] == 1
    assert metrics["facts_after_normalization"] == 1
    assert database.normalization.facts[0].predicate == "topical_link"
    assert any(
        attempt.reason_code == "SCHEMA_CONFLICT_ALTERNATIVE_SELECTED"
        and attempt.verdict == REJECTED
        for attempt in database.verification_attempts
    )
    assert any(
        payload["component"] == "source_conflict"
        for _run_id, payload in database.source_model_calls
    )


def test_runtime_materializes_reference_fact_against_frozen_antecedent(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)

    class ReferenceDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            content = "Left11 waited. Then it connects Right11."
            self.chunks[0].update({
                "content": content,
                "content_hash": "reference-content-hash",
            })
            starts = {
                "positive-left": content.index("Left11"),
                "positive-right": content.index("Right11"),
            }
            for mention in self.mentions:
                start = starts[str(mention["mention_key"])]
                mention["evidence_start"] = start
                mention["evidence_end"] = start + len(
                    str(mention["surface"])
                )
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class ReferenceGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                assert payload["reference_candidates"] == [{
                    "id": "R0001",
                    "surface": "it",
                    "start": payload["source"].index("it connects"),
                    "end": payload["source"].index("it connects") + 2,
                }]
                return {
                    "resolved_references": [{
                        "reference_id": "R0001",
                        "antecedent_id": "M0000",
                    }],
                    "facts": [{
                        "subject_id": "R0001",
                        "predicate": "connects",
                        "object_id": "M0001",
                        "evidence": "Then it connects Right11.",
                    }],
                }
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": 0,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    }],
                }
            if (
                "directional textual-entailment verifier" in system
                or "semantic-role mapping auditor" in system
            ):
                entailed = payload["first_endpoint"]["surface"] == "it"
                result = {
                    "source_role": entailed,
                    "target_role": entailed,
                    "direct_relation": entailed,
                    "source_argument_start": -1,
                    "source_argument_end": -1,
                    "source_argument_surface": "",
                    "target_argument_start": -1,
                    "target_argument_end": -1,
                    "target_argument_surface": "",
                }
                if entailed:
                    for label, endpoint_name in (
                        ("source", "first_endpoint"),
                        ("target", "second_endpoint"),
                    ):
                        endpoint = payload[endpoint_name]
                        result[f"{label}_argument_start"] = (
                            endpoint["absolute_span"][0]
                        )
                        result[f"{label}_argument_end"] = (
                            endpoint["absolute_span"][1]
                        )
                        result[f"{label}_argument_surface"] = endpoint[
                            "surface"
                        ]
                return result
            return super().json_chat(**kwargs)

    database = ReferenceDatabase()
    result = GroundedRelationRuntime(
        database, ReferenceGateway(),
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["source_reference_proxies_accepted"] == 1
    assert metrics["source_reference_proxies_active"] == 1
    assert metrics["source_reference_bindings_dropped"] == 0
    assert metrics["source_reference_binding_conflicts"] == 0
    opportunity = next(
        item for item in database.opportunities
        if item.provenance.get("resolved_reference_endpoints")
    )
    assert any(
        value.startswith("relation-reference-")
        for value in (
            opportunity.left_occurrence_key,
            opportunity.right_occurrence_key,
        )
    )
    assert "positive-left" in {
        opportunity.left_mention_id, opportunity.right_mention_id,
    }
    assert database.endpoints
    endpoint = database.endpoints[0]
    assert endpoint["source_mention_key"] == "positive-left"
    assert endpoint["metadata"]["resolved_reference_endpoints"][0][
        "reference_text"
    ] == "it"


def test_runtime_independently_resolves_uncovered_reference_before_syntax_recovery(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(
        Config, "relation_source_binding_recall_alternatives", 0,
    )

    class ReferenceDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            content = "Left11 waited. Then it connects Right11."
            self.chunks[0].update({
                "content": content,
                "content_hash": "independent-reference-content-hash",
            })
            starts = {
                "positive-left": content.index("Left11"),
                "positive-right": content.index("Right11"),
            }
            for mention in self.mentions:
                start = starts[str(mention["mention_key"])]
                mention["evidence_start"] = start
                mention["evidence_end"] = start + len(
                    str(mention["surface"])
                )
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class ReferenceGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                # The fact extractor deliberately misses the reference.  The
                # independent discourse-role pass must recover it without a
                # second entity extraction or a synthetic antecedent mention.
                return {"resolved_references": [], "facts": []}
            if "predicate-blind coreference adjudicator" in system:
                reference = payload["references"][0]
                return {
                    "coverage_complete": True,
                    "resolutions": [{
                        "reference_id": reference["reference_id"],
                        "antecedent_id": reference["candidates"][0][
                            "antecedent_id"
                        ],
                        "unambiguous": True,
                    }],
                }
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": 0,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    }],
                }
            if (
                "directional textual-entailment verifier" in system
                or "semantic-role mapping auditor" in system
            ):
                result = {
                    "source_role": True,
                    "target_role": True,
                    "direct_relation": True,
                }
                for label, endpoint_name in (
                    ("source", "first_endpoint"),
                    ("target", "second_endpoint"),
                ):
                    endpoint = payload[endpoint_name]
                    result[f"{label}_argument_start"] = endpoint[
                        "absolute_span"
                    ][0]
                    result[f"{label}_argument_end"] = endpoint[
                        "absolute_span"
                    ][1]
                    result[f"{label}_argument_surface"] = endpoint["surface"]
                return result
            return super().json_chat(**kwargs)

    database = ReferenceDatabase()
    result = GroundedRelationRuntime(
        database, ReferenceGateway(),
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["source_reference_adjudication_targets"] == 1
    assert metrics["source_reference_adjudication_calls"] == 1
    assert metrics["source_reference_adjudication_failures"] == 0
    assert metrics[
        "source_reference_adjudication_proxies_accepted"
    ] == 1
    assert metrics["source_deterministic_frames_recovered"] == 1
    assert metrics["facts_after_normalization"] == 1
    assert any(
        payload["component"] == "source_reference"
        for _run_id, payload in database.source_model_calls
    )
    assert database.endpoints[0]["source_mention_key"] == "positive-left"
    assert database.endpoints[0]["metadata"][
        "resolved_reference_endpoints"
    ][0]["reference_text"] == "it"


def test_runtime_adds_bounded_semantic_source_binding_shortlist(monkeypatch):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_reference_adjudication_enabled", False)
    monkeypatch.setattr(
        Config, "relation_source_binding_recall_alternatives", 1,
    )
    monkeypatch.setattr(
        Config, "relation_source_binding_recall_min_score", 0.18,
    )

    shortlist_schema = logical_schema(
        predicate_policy(
            "acts_on",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition="The subject acts on the object.",
        ),
        predicate_policy(
            "connects_to",
            domain=["Organization"],
            range_=["Product"],
            semantic_definition=(
                "The grammatical subject directly connects to the object."
            ),
            direction_description=(
                "From the connecting grammatical subject to the connected object."
            ),
            positive_examples=["Left11 connects Right11."],
            optional_soft_cues=["connects", "connects to"],
            source_role_pairs=[
                "grammatical_subject->direct_argument",
            ],
        ),
        version="schema-runtime-shortlist-v1",
    )

    class ShortlistDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            self.source_model_calls = []

        @staticmethod
        def get_active_schema(project_id):
            assert project_id == 1
            return {"version": 1, "schema_json": shortlist_schema}

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class ShortlistGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                return {"facts": [{
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0001",
                    "evidence": str(payload["source"]),
                }]}
            if "source-frame to Schema binding judge" in system:
                # Simulate the characteristic enumerative omission: one valid
                # active predicate is returned and the other is never compared.
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": 0,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    }],
                }
            if system == "Audit exact source roles. Return only JSON.":
                return {"projection_status": "ALIGNED"}
            return super().json_chat(**kwargs)

    database = ShortlistDatabase()
    result = GroundedRelationRuntime(
        database, ShortlistGateway(),
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["source_recall_shortlist_bindings"] == 1
    assert metrics["source_recall_shortlist_frames_considered"] >= 1
    assert metrics["source_recall_shortlist_candidates_considered"] >= 1
    assert metrics["focused_review_eligible"] == 2
    assert metrics["focused_review_selected"] == 2
    assert metrics["focused_review_confirmations"] == 2
    assert metrics["focused_review_batch_positive_confirmed"] == 2
    assert metrics["source_role_audit_calls"] == 2
    assert metrics["focused_review_model_calls"] == 2
    assert metrics["focused_verifier_model_calls"] == 0
    assert metrics["verifier_model_calls"] == metrics["batch_verifier_calls"] == 1
    assert {
        fact.predicate for fact in database.normalization.facts
    } == {"acts_on", "connects_to"}
    recovered = next(
        item for item in database.proposed_hypotheses
        if item.predicate == "connects_to"
    )
    assert recovered.predicate_metadata[
        "source_binding_source"
    ] == "DETERMINISTIC_SEMANTIC_SHORTLIST"
    assert recovered.predicate_metadata[
        "source_binding_shortlist_score"
    ] >= 0.18


@pytest.mark.parametrize("anchor_status", ["ALIGNED", "CONFLICT"])
def test_runtime_jointly_audits_explicit_coordinated_source_members(
    monkeypatch, anchor_status,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(
        Config, "relation_reference_adjudication_enabled", False,
    )
    monkeypatch.setattr(
        Config, "relation_source_binding_recall_alternatives", 0,
    )

    class CoordinatedDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            content = "POSITIVE_CASE Left11 connects Right11 and Third11."
            self.chunks[0].update({
                "content": content,
                "content_hash": "coordinated-source-content-hash",
            })
            for mention in self.mentions:
                surface = str(mention["surface"])
                start = content.index(surface)
                mention["evidence_start"] = start
                mention["evidence_end"] = start + len(surface)
            third_start = content.index("Third11")
            self.mentions.append({
                "mention_key": "positive-third",
                "chunk_id": 11,
                "surface": "Third11",
                "evidence_start": third_start,
                "evidence_end": third_start + len("Third11"),
            })
            self.assignments.append({
                "mention_key": "positive-third",
                "canonical_key": "canonical-11-third",
                "entity_type": "Product",
                "policy_version": "assignment-v1",
                "revision_id": "revision-runtime",
            })
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class CoordinatedGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.group_audit_calls = 0
            self.group_anchor_audit_calls = 0
            self.individual_role_audit_calls = 0

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                evidence = str(payload["source"])
                return {"facts": [{
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0001",
                    "evidence": evidence,
                }, {
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0002",
                    "evidence": evidence,
                }]}
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": frame_index,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    } for frame_index in range(2)],
                }
            if system == (
                "Audit exact source roles jointly. Return only JSON."
            ):
                self.group_audit_calls += 1
                return {
                    item["hypothesis_id"]: "ALIGNED"
                    for item in payload["candidates"]
                }
            if system == "Audit exact source roles. Return only JSON.":
                self.group_anchor_audit_calls += 1
                assert list(payload) == [
                    "instruction", "source", "candidate",
                    "competing_bindings", "required_roles",
                ]
                assert "coordination_contract" not in payload
                return {"projection_status": anchor_status}
            return super().json_chat(**kwargs)

    database = CoordinatedDatabase()
    gateway = CoordinatedGateway()
    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime", project_id=1,
    )

    metrics = result["metrics"]
    aligned = anchor_status == "ALIGNED"
    assert gateway.group_anchor_audit_calls == 1
    assert gateway.group_audit_calls == (1 if aligned else 0)
    assert gateway.individual_role_audit_calls == 0
    assert metrics["source_group_anchor_audit_calls"] == 1
    assert metrics["source_group_anchor_audit_claims"] == 2
    assert metrics["source_group_anchor_audit_aligned"] == (
        2 if aligned else 0
    )
    assert metrics["source_group_anchor_audit_conflicts"] == (
        0 if aligned else 2
    )
    assert metrics["source_group_audit_calls"] == (1 if aligned else 0)
    assert metrics["source_group_audit_claims"] == (2 if aligned else 0)
    assert metrics["source_group_audit_aligned"] == (2 if aligned else 0)
    assert metrics["source_role_audit_calls"] == (2 if aligned else 1)
    assert metrics["source_role_audit_claims"] == (3 if aligned else 1)
    assert metrics["source_role_audit_aligned"] == (2 if aligned else 0)
    assert metrics["source_role_audit_conflicts"] == (0 if aligned else 2)
    assert metrics["focused_review_model_calls"] == (2 if aligned else 1)
    assert metrics["focused_verifier_model_calls"] == 0
    assert len(database.normalization.facts) == (2 if aligned else 0)
    assert {fact.predicate for fact in database.normalization.facts} == (
        {"acts_on"} if aligned else set()
    )


@pytest.mark.parametrize("role_audit_status", ["ALIGNED", "CONFLICT"])
def test_runtime_focused_review_gates_batch_rejected_direct_source_binding(
    monkeypatch, role_audit_status,
):
    monkeypatch.setattr(Config, "relation_open_fact_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_source_binding_gleaning_rounds", 0)
    monkeypatch.setattr(Config, "relation_reference_adjudication_enabled", False)
    monkeypatch.setattr(
        Config, "relation_source_binding_recall_alternatives", 0,
    )
    monkeypatch.setattr(Config, "relation_verifier_focused_review_max", 4)

    class FocusedDatabase(_RuntimeDatabase):
        def __init__(self):
            super().__init__()
            _keep_runtime_chunks(self, {11})
            self.source_model_calls = []

        def record_grounded_relation_source_model_call(
            self, run_id, payload,
        ):
            self.source_model_calls.append((run_id, payload))
            return payload

    class FocusedGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "source-grounded open fact-frame extractor" in system:
                return {"facts": [{
                    "subject_id": "M0000",
                    "predicate": "connects",
                    "object_id": "M0001",
                    "evidence": str(payload["source"]),
                }]}
            if "source-frame to Schema binding judge" in system:
                return {
                    "coverage_complete": True,
                    "bindings": [{
                        "frame_index": 0,
                        "predicate": "acts_on",
                        "subject_index": 0,
                        "object_index": 1,
                    }],
                }
            if isinstance(payload.get("claims"), list):
                return {"decisions": [{
                    "hypothesis_id": claim["hypothesis_id"],
                    "verdict": REJECTED,
                    "reason_code": "SCRIPTED_BATCH_REJECT",
                    "forward_entailment": "NOT_ENTAILED",
                    "reverse_entailment": "NOT_ENTAILED",
                    "subject_role_supported": False,
                    "object_role_supported": False,
                    "supported_evidence_spans": [],
                } for claim in payload["claims"]]}
            if system == "Audit exact source roles. Return only JSON.":
                return {"projection_status": role_audit_status}
            return super().json_chat(**kwargs)

    database = FocusedDatabase()
    result = GroundedRelationRuntime(
        database, FocusedGateway(),
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["focused_review_eligible"] == 1
    assert metrics["focused_review_selected"] == 1
    assert metrics["focused_review_recoveries"] == 1
    assert metrics["source_role_audit_calls"] == 1
    assert metrics["source_role_audit_claims"] == 1
    if role_audit_status == "ALIGNED":
        assert metrics["source_role_audit_aligned"] == 1
        assert metrics["source_role_audit_conflicts"] == 0
        assert metrics["focused_review_model_calls"] == 2
        assert metrics["focused_verifier_model_calls"] == 1
        assert metrics["focused_review_batch_negative_recovered"] == 1
        assert metrics["focused_review_batch_negative_retained"] == 0
        assert metrics["facts_after_normalization"] == 1
    else:
        assert metrics["source_role_audit_aligned"] == 0
        assert metrics["source_role_audit_conflicts"] == 1
        assert metrics["focused_review_model_calls"] == 1
        assert metrics["focused_verifier_model_calls"] == 0
        assert metrics["focused_review_batch_negative_recovered"] == 0
        assert metrics["focused_review_batch_negative_retained"] == 1
        assert metrics["facts_after_normalization"] == 0
    attempt = database.verification_attempts[-1]
    assert attempt.model_request_payload["stage"] == (
        "focused_verifier" if role_audit_status == "ALIGNED"
        else "source_projection_role_audit"
    )
    assert attempt.model_request_payload["focused_review"]["kind"] == "recovery"


def test_runtime_persists_and_uses_shadow_context_units():
    database = _RuntimeDatabase()

    GroundedRelationRuntime(database, _ScriptedGateway()).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert len(database.source_contexts) == len(database.chunks)
    assert all(
        item["payload"]["structure_quality"] == "legacy_chunk_local"
        for _, item in database.source_contexts
    )
    assert all(
        str(item.chunk_key).startswith("shadowctx_")
        for item in database.opportunities
    )
    assert all(
        item.provenance["source_context_quality"] == "legacy_chunk_local"
        for item in database.opportunities
    )


def test_runtime_recovers_only_non_publishable_legacy_batch_members():
    class LegacyBatchGateway(_ScriptedGateway):
        last_raw_response_text = ""
        last_parsed_json_response = None
        last_call_diagnostics = None

        def json_chat(self, **kwargs):
            payload = json.loads(kwargs["prompt"])
            if isinstance(payload.get("claims"), list):
                self.verifier_calls += 1
                legacy = {"verifications": [{
                    "hypothesis_id": claim["hypothesis_id"],
                    "verdict": "REJECTED",
                    "entailment": "NOT_ENTAILED",
                    "evidence_spans": [],
                } for claim in payload["claims"]]}
                self.last_parsed_json_response = legacy
                self.last_raw_response_text = json.dumps(legacy)
                raise RuntimeError("legacy batch root alias")
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    gateway = LegacyBatchGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    metrics = result["metrics"]
    assert metrics["batch_verifier_calls"] == 1
    assert metrics["batch_verifier_partial_batches"] == 1
    assert metrics["batch_verifier_partial_decisions"] == 8
    assert metrics["batch_verifier_fallbacks"] == 0
    assert metrics["batch_verifier_failures"] == 0
    assert metrics["verification_verdicts"] == {
        REJECTED: 8,
        UNCERTAIN: 1,
    }
    assert metrics["facts_after_normalization"] == 0
    assert len(database.endpoints) == 0


def test_runtime_retains_single_verifier_transport_audit(monkeypatch):
    monkeypatch.setattr(Config, "relation_verifier_batch_max_chars", 1)
    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {11})
    gateway = _ScriptedGateway()

    GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    attempt = database.verification_attempts[-1]
    assert attempt.model_request_payload["stage"] == "verifier"
    assert attempt.model_request_payload["schema"]["required"]
    assert attempt.raw_response_text
    assert attempt.parsed_response["verdict"] == SUPPORTED
    assert attempt.token_usage == {}


def test_runtime_predicate_rename_with_same_metadata_keeps_stage_behavior():
    outputs = []
    for name in ("acts_on", "renamed_action"):
        database = _RuntimeDatabase(action_name=name)
        result = GroundedRelationRuntime(
            database, _ScriptedGateway(action_name=name),
        ).run_namespace(
            namespace_id="namespace-runtime",
            project_id=1,
        )
        outputs.append((
            result["metrics"]["classification_decisions"],
            result["metrics"]["verification_verdicts"],
            result["metrics"]["facts_after_normalization"],
            result["metrics"]["facts_materialization_eligible"],
        ))
        assert any(
            fact.predicate == name for fact in database.normalization.facts
        )
    assert outputs[0] == outputs[1]


def test_runtime_stale_fingerprint_fails_closed_without_materialization():
    database = _RuntimeDatabase(stale=True)

    with pytest.raises(StaleRelationSnapshotError):
        GroundedRelationRuntime(
            database, _ScriptedGateway(),
        ).run_namespace(
            namespace_id="namespace-runtime",
            project_id=1,
        )

    assert database.endpoints == []
    assert database.finished[-1][1]["status"] == "stale"


def test_runtime_classifier_failure_does_not_fallback_or_materialize():
    database = _RuntimeDatabase()

    result = GroundedRelationRuntime(
        database, _ScriptedGateway(fail_classifier=True),
    ).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert database.endpoints == []
    assert result["metrics"]["classification_decisions"] == {
        OTHER_UNSUPPORTED: len(database.decisions),
    }
    assert len(database.decisions) > 0
    assert all(
        decision.decision == OTHER_UNSUPPORTED
        and decision.hypotheses == ()
        for decision in database.decisions
    )
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"


def test_runtime_excludes_quarantined_canonicals_from_pair_planner():
    class QuarantinedDatabase(_RuntimeDatabase):
        def load_grounded_entity_snapshot(self, namespace_id):
            snapshot = super().load_grounded_entity_snapshot(namespace_id)
            snapshot["canonicals"] = [
                {
                    "canonical_key": item["canonical_key"],
                    "metadata": {"quarantined": True},
                }
                for item in snapshot["assignments"]
            ]
            return snapshot

    database = QuarantinedDatabase()
    gateway = _ScriptedGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["pair_opportunities_total"] == 0
    assert result["metrics"]["pair_opportunities_classified"] == 0
    assert gateway.classifier_calls == 0
    assert gateway.verifier_calls == 0
    assert database.endpoints == []
    assert database.finished[-1][1]["status"] == "ready"


def test_runtime_local_subgraph_enters_verifier_without_duplicate_pair_classification():
    class ProposerPositiveClassifierNegativeGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                left, right = payload["occurrences"]
                return {
                    "proposals": [{
                        "subject_occurrence_id": left["occurrence_id"],
                        "predicate": self.action_name,
                        "object_occurrence_id": right["occurrence_id"],
                        "evidence_spans": [{
                            "start": 0,
                            "end": len(payload["context"]["text"]),
                        }],
                        "qualifiers": [],
                        "confidence": 0.77,
                    }],
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            if "classifier" in system:
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CLASSIFIER_NO_RELATION",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                } for pair in payload["pairs"]]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = ProposerPositiveClassifierNegativeGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["classification_decisions"] == {}
    assert gateway.classifier_calls == 0
    assert result["metrics"]["proposer_model_calls"] == 1
    assert result["metrics"]["proposals_total"] == 1
    assert result["metrics"]["proposer_hypotheses_persisted"] == 1
    assert result["metrics"]["verifier_supported"] == 1
    assert len(database.proposed_hypotheses) == 1
    assert database.proposed_candidate_source == "DIRECT_PROPOSER"
    assert len(database.endpoints) == 1


def test_runtime_complete_atomic_scope_still_runs_independent_classifier():
    class CompleteScopeGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                left, right, _third = payload["occurrences"]
                return {
                    "proposals": [{
                        "subject_occurrence_id": left["occurrence_id"],
                        "predicate": self.action_name,
                        "object_occurrence_id": right["occurrence_id"],
                        "evidence_spans": [payload["required_evidence_span"]],
                        "qualifiers": [],
                        "confidence": 0.8,
                    }],
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    content = "NO_CASE Left1 connects Right1 and Extra1."
    database.chunks[0]["content"] = content
    extra_start = content.index("Extra1")
    database.mentions.append({
        "mention_key": "extra-third",
        "chunk_id": 1,
        "surface": "Extra1",
        "evidence_start": extra_start,
        "evidence_end": extra_start + len("Extra1"),
    })
    database.assignments.append({
        "mention_key": "extra-third",
        "canonical_key": "canonical-1-extra",
        "entity_type": "Product",
        "policy_version": "assignment-v1",
        "revision_id": "revision-runtime",
    })
    gateway = CompleteScopeGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    # Proposer coverage is not an independent negative decision.  The
    # classifier must still inspect every admitted pair it did not propose.
    assert gateway.classifier_calls >= 1
    assert result["metrics"]["proposer_scopes_complete"] == 1
    assert result["metrics"]["proposer_coverage_no_relation"] == 0
    assert result["metrics"]["classification_decisions"]
    assert len(database.classification_attempts) >= 1
    assert all(
        values["status"] == "ACCEPTED"
        and values["independent_call"] is True
        and values["metadata"]["chunk_id"] == 1
        for _run_id, _pair_id, values in database.classification_attempts
    )


def test_runtime_incomplete_atomic_scope_keeps_pair_classifier_backstop():
    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = _ScriptedGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["proposer_scopes_complete"] == 0
    assert result["metrics"]["proposer_coverage_no_relation"] == 0
    assert result["metrics"]["pair_opportunities_classifier_model"] == 1
    assert gateway.classifier_calls == 1


def test_runtime_local_subgraph_promotes_pair_deferred_only_by_budget(
    monkeypatch,
):
    """A processing cap must not discard an explicit, still-verified relation."""

    monkeypatch.setattr(Config, "relation_max_candidates_per_chunk", 0)

    class BudgetDeferredGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                left, right = payload["occurrences"]
                return {
                    "proposals": [{
                        "subject_occurrence_id": left["occurrence_id"],
                        "predicate": self.action_name,
                        "object_occurrence_id": right["occurrence_id"],
                        "evidence_spans": [{
                            "start": 0,
                            "end": len(payload["context"]["text"]),
                        }],
                        "qualifiers": [],
                        "confidence": 0.77,
                    }],
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = BudgetDeferredGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 0
    assert result["metrics"]["pair_opportunities_local_subgraph"] == 1
    assert result["metrics"]["proposer_hypotheses_persisted"] == 1
    assert result["metrics"]["verifier_supported"] == 1
    assert len(database.endpoints) == 1
    opportunity = database.opportunities[0]
    assert opportunity.status == v2.PENDING_CLASSIFICATION
    assert (
        opportunity.provenance["planner_status_before_local_subgraph"]
        == v2.DEFERRED_BUDGET
    )


def test_runtime_direct_proposer_cannot_create_a_pair_missing_from_planner(monkeypatch):
    monkeypatch.setattr(
        v2,
        "plan_pair_opportunities",
        lambda **_kwargs: v2.PairOpportunityPlan(
            opportunities=(),
            theoretical_pairs=1,
            duplicate_occurrences_collapsed=0,
        ),
    )

    class DirectOnlyGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                left, right = payload["occurrences"]
                return {
                    "proposals": [{
                        "subject_occurrence_id": left["occurrence_id"],
                        "predicate": self.action_name,
                        "object_occurrence_id": right["occurrence_id"],
                        "evidence_spans": [{
                            "start": 0,
                            "end": len(payload["context"]["text"]),
                        }],
                        "qualifiers": [],
                        "confidence": 0.77,
                    }],
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            if "classifier" in system:
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CLASSIFIER_NO_RELATION",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                } for pair in payload["pairs"]]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = DirectOnlyGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert database.opportunities == []
    assert result["metrics"]["pair_opportunities_classified"] == 0
    assert gateway.classifier_calls == 0
    assert result["metrics"]["proposer_hypotheses_persisted"] == 0
    assert result["metrics"]["verifier_supported"] == 0
    assert database.endpoints == []


def test_runtime_proposer_retry_then_relation_only_gleaning(monkeypatch):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)

    class RetryThenGleanGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_payloads = []

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                self.proposer_payloads.append(payload)
                if len(self.proposer_payloads) == 1:
                    raise OllamaProtocolError(
                        "Ollama response was truncated by output length",
                    )
                if payload["pass_no"] == 0:
                    left, right = payload["occurrences"]
                    return {
                        "proposals": [{
                            "subject_occurrence_id": left["occurrence_id"],
                            "predicate": self.action_name,
                            "object_occurrence_id": right["occurrence_id"],
                            "evidence_spans": [{
                                "start": 0,
                                "end": len(payload["context"]["text"]),
                            }],
                            "qualifiers": [],
                            "confidence": 0.77,
                        }],
                        "missing_endpoint_mentions": [],
                        "coverage": {"complete": False, "uncovered_cues": []},
                    }
                return {
                    "proposals": [],
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            if "classifier" in system:
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CLASSIFIER_NO_RELATION",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                } for pair in payload["pairs"]]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = RetryThenGleanGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert [item["pass_no"] for item in gateway.proposer_payloads] == [
        0, 0, 1,
    ]
    assert gateway.proposer_payloads[1]["already_found_keys"] == []
    assert gateway.proposer_payloads[2]["already_found_keys"]
    assert result["metrics"]["proposer_model_calls"] == 3
    assert result["metrics"]["proposer_failures"] == 1
    assert result["metrics"]["proposals_total"] == 1
    assert result["metrics"]["proposer_terminal_statuses"] == {
        "SUCCESS_WITH_PROPOSALS": 1,
    }
    assert [
        item["status"] for _, item in database.proposal_attempts
    ] == [
        "RETRYABLE_FAILED", "ACCEPTED", "ACCEPTED",
    ]
    assert [
        (item["metadata"]["pass_no"], item["metadata"]["retry_no"])
        for _, item in database.proposal_attempts
    ] == [(0, 1), (0, 2), (1, 1)]
    assert all(item["request_hash"] for _, item in database.proposal_attempts)
    assert len(database.proposed_hypotheses) == 1
    assert len(database.endpoints) == 1


def test_runtime_proposer_retry_sends_protocol_feedback_after_hard_failure(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)

    class InvalidThenCorrectedProposerGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_payloads = []

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                self.proposer_payloads.append(payload)
                if len(self.proposer_payloads) == 1:
                    occurrence = payload["occurrences"][0]
                    return {
                        "proposals": [{
                            "subject_occurrence_id": "invented-occurrence",
                            "predicate": self.action_name,
                            "object_occurrence_id": occurrence["occurrence_id"],
                            "evidence_spans": [{
                                "start": 0,
                                "end": len(payload["context"]["text"]),
                            }],
                            "qualifiers": [],
                            "confidence": 0.77,
                        }],
                        "missing_endpoint_mentions": [],
                        "coverage": {"complete": True, "uncovered_cues": []},
                    }
                left, right = payload["occurrences"]
                return {
                    "proposals": [{
                        "subject_occurrence_id": left["occurrence_id"],
                        "predicate": self.action_name,
                        "object_occurrence_id": right["occurrence_id"],
                        "evidence_spans": [{
                            "start": 0,
                            "end": len(payload["context"]["text"]),
                        }],
                        "qualifiers": [],
                        "confidence": 0.77,
                    }],
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            if "classifier" in system:
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CLASSIFIER_NO_RELATION",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                } for pair in payload["pairs"]]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = InvalidThenCorrectedProposerGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert len(gateway.proposer_payloads) >= 1
    assert gateway.proposer_payloads[0]["protocol_feedback"] == ""
    assert [
        values["status"] for _, values in database.proposal_attempts
    ][0] == "ACCEPTED"
    first_schema = database.proposal_attempts[0][1]["request_payload"]["schema"]
    first_properties = (
        first_schema["properties"]["proposals"]["items"]["properties"]
    )
    assert first_properties["subject_occurrence_id"]["enum"] == [
        "no-left",
        "no-right",
    ]
    assert first_properties["predicate"]["enum"]
    assert (
        first_schema["properties"]["proposals"]["maxItems"]
        == Config.relation_proposer_max_proposals
    )
    rejected = (
        database.proposal_attempts[0][1]["parsed_response"]["coverage"]
        ["_parser_rejected_proposals"]
    )
    assert rejected == [{
        "index": 0,
        "reason": "endpoint_not_in_occurrence_universe",
    }]
    assert result["metrics"]["proposer_failures"] == 0
    assert result["metrics"]["proposals_total"] == 1
    assert result["metrics"]["proposer_hypotheses_persisted"] == 1
    assert result["metrics"]["verifier_supported"] == 1
    assert len(database.endpoints) == 1


def test_runtime_proposer_malformed_candidates_are_isolated_not_fatal(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)
    monkeypatch.setattr(Config, "relation_proposer_failure_circuit_breaker", 1)

    class MalformedCandidatesGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_payloads = []

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                self.proposer_payloads.append(payload)
                left, right = payload["occurrences"]
                return {
                    "proposals": [
                        {
                            "subject_occurrence_id": "invented-occurrence",
                            "predicate": self.action_name,
                            "object_occurrence_id": right["occurrence_id"],
                            "evidence_spans": "bad-spans",
                            "qualifiers": "bad-qualifiers",
                            "confidence": 3.0,
                        },
                        {
                            "subject_occurrence_id": left["occurrence_id"],
                            "predicate": self.action_name,
                            "object_occurrence_id": right["occurrence_id"],
                            "evidence_spans": "bad-spans",
                            "qualifiers": "bad-qualifiers",
                            "confidence": 3.0,
                        },
                    ],
                    "missing_endpoint_mentions": "bad-audit-field",
                    "coverage": "bad-coverage",
                }
            if "classifier" in system:
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CLASSIFIER_NO_RELATION",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                } for pair in payload["pairs"]]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = MalformedCandidatesGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["proposer_failures"] == 0
    assert result["metrics"]["proposer_circuit_breaker_tripped"] is False
    assert result["metrics"]["proposer_skipped_by_circuit"] == 0
    assert result["metrics"]["proposals_total"] == 1
    assert len(database.proposed_hypotheses) == 1
    accepted_attempt = database.proposal_attempts[0][1]
    assert accepted_attempt["status"] == "ACCEPTED"
    proposal = accepted_attempt["parsed_response"]["proposals"][0]
    assert proposal["evidence_spans"] == [{
        "start": 0,
        "end": len(database.chunks[0]["content"]),
    }]
    assert proposal["qualifiers"] == []
    assert proposal["confidence"] == 1.0
    coverage = accepted_attempt["parsed_response"]["coverage"]
    assert coverage["_parser_rejected_proposals"] == [{
        "index": 0,
        "reason": "endpoint_not_in_occurrence_universe",
    }]
    assert "coverage_type_defaulted" in coverage["_parser_repairs"]
    assert "missing_endpoint_mentions_type_defaulted" in (
        coverage["_parser_repairs"]
    )
    assert "proposal_1_evidence_spans_defaulted_to_context" in (
        coverage["_parser_repairs"]
    )
    assert "proposal_1_qualifiers_defaulted" in coverage["_parser_repairs"]
    assert "proposal_1_confidence_clamped" in coverage["_parser_repairs"]


def test_runtime_proposer_bad_top_level_proposals_shape_is_empty_success(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)
    monkeypatch.setattr(Config, "relation_proposer_failure_circuit_breaker", 1)

    class BadTopLevelProposalsGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            payload = json.loads(kwargs["prompt"])
            if "proposer" in system:
                return {
                    "proposals": "not-array",
                    "missing_endpoint_mentions": [],
                    "coverage": {"complete": True, "uncovered_cues": []},
                }
            if "classifier" in system:
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CLASSIFIER_NO_RELATION",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                } for pair in payload["pairs"]]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = BadTopLevelProposalsGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["proposer_failures"] == 0
    assert result["metrics"]["proposals_total"] == 0
    assert result["metrics"]["proposer_circuit_breaker_tripped"] is False
    assert database.proposal_attempts[0][1]["status"] == "ACCEPTED"
    assert database.proposal_attempts[0][1]["metadata"]["proposal_count"] == 0
    assert "proposals_type_defaulted" in (
        database.proposal_attempts[0][1]["parsed_response"]["coverage"]
        ["_parser_repairs"]
    )


def test_runtime_proposer_truncation_does_not_abort_v2_main_chain(monkeypatch):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)

    class TruncatingProposerGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_calls = 0
            self.proposer_output_budgets = []

        def json_chat(self, **kwargs):
            if "proposer" in str(kwargs["system"]):
                self.proposer_calls += 1
                self.proposer_output_budgets.append(
                    int(kwargs["max_output_tokens"]),
                )
                raise OllamaProtocolError(
                    "Ollama response was truncated by output length",
                )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {11})
    gateway = TruncatingProposerGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["proposer_model_calls"] == 2
    assert result["metrics"]["proposer_failures"] == 2
    assert result["metrics"]["proposals_total"] == 0
    assert gateway.proposer_output_budgets == [4096, 8192]
    assert gateway.classifier_calls == 1
    # The surviving pairwise-classifier positive receives one independent
    # focused confirmation in addition to the batch verifier.
    assert gateway.verifier_calls == 2
    assert result["metrics"]["verifier_supported"] == 1
    assert database.finished[-1][1]["status"] == "ready"
    # A verified pair-classifier result remains auditable, but without an
    # explicit source relation frame it is not projected into the public graph.
    assert result["metrics"]["facts_predicate_publication_suppressed"] == 1
    assert len(database.endpoints) == 0


def test_runtime_proposer_failure_circuit_breaker_skips_auxiliary_calls(monkeypatch):
    monkeypatch.setattr(Config, "relation_max_attempts", 1)
    monkeypatch.setattr(Config, "relation_proposer_failure_circuit_breaker", 1)

    class FailingProposerGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.proposer_calls = 0

        def json_chat(self, **kwargs):
            if "proposer" in str(kwargs["system"]):
                self.proposer_calls += 1
                raise OllamaProtocolError(
                    "Ollama response was truncated by output length",
                )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1, 11})
    gateway = FailingProposerGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.proposer_calls == 1
    assert result["metrics"]["proposer_circuit_breaker_tripped"] is True
    assert result["metrics"]["proposer_skipped_by_circuit"] == 1
    assert gateway.classifier_calls == 2
    # Source-first proposer failure does not weaken the precision gate on a
    # surviving pairwise-classifier positive.
    assert gateway.verifier_calls == 2
    assert result["metrics"]["verifier_supported"] == 1
    assert database.finished[-1][1]["status"] == "ready"


def test_runtime_no_relation_without_evidence_span_is_a_negative_decision():
    class NoRelationWithoutEvidenceGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                pair = json.loads(kwargs["prompt"])["pairs"][0]
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "NO_SEMANTIC_RELATION",
                }]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = NoRelationWithoutEvidenceGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["classification_decisions"] == {NO_RELATION: 1}
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"
    assert len(database.endpoints) == 0


def test_runtime_classifier_retry_sends_protocol_feedback_and_recovers(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)

    class InvalidThenCorrectedGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.classifier_payloads = []

        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                payload = json.loads(kwargs["prompt"])
                self.classifier_payloads.append(payload)
                pair = payload["pairs"][0]
                if self.classifier_calls == 1:
                    return {"items": [{
                        "pair_id": pair["pair_id"],
                        "decision": HYPOTHESES,
                        "hypotheses": [{
                            "subject_occurrence_id": "LEFT",
                            "predicate": "schema_outside_relation",
                            "object_occurrence_id": "RIGHT",
                            "evidence_spans": [{
                                "start": pair["context_span"][0],
                                "end": pair["context_span"][1],
                            }],
                            "classifier_confidence": 0.95,
                        }],
                    }]}
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "CORRECTED_AFTER_PROTOCOL_FEEDBACK",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                }]}

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = InvalidThenCorrectedGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 2
    assert len(gateway.classifier_payloads) == 2
    assert gateway.classifier_payloads[0]["protocol_feedback"] == ""
    assert "predicates" not in gateway.classifier_payloads[0]
    assert "predicates" not in gateway.classifier_payloads[1]
    assert "schema_outside_relation" not in (
        gateway.classifier_payloads[1]["pairs"][0]["allowed_predicates"]
    )
    assert "Previous classifier response violated the v2 contract" in (
        gateway.classifier_payloads[1]["protocol_feedback"]
    )
    assert "predicate only from that pair's allowed_predicates" in (
        gateway.classifier_payloads[1]["protocol_feedback"]
    )
    assert result["metrics"]["classifier_failures"] == 1
    assert result["metrics"]["classification_decisions"] == {NO_RELATION: 1}
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"
    assert len(database.endpoints) == 0
    assert [
        values["status"] for _, _, values in database.classification_attempts
    ] == ["RETRYABLE_PROTOCOL", "ACCEPTED"]


def test_runtime_invalid_single_pair_hypothesis_fails_closed(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)

    class InvalidPredicateGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                pair = json.loads(kwargs["prompt"])["pairs"][0]
                return {"items": [{
                    "pair_id": pair["pair_id"],
                    "decision": HYPOTHESES,
                    "hypotheses": [{
                        "subject_occurrence_id": "LEFT",
                        "predicate": "funded_by",
                        "object_occurrence_id": "RIGHT",
                        "evidence_spans": [{
                            "start": pair["context_span"][0],
                            "end": pair["context_span"][1],
                        }],
                        "classifier_confidence": 0.95,
                    }],
                }]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {11})
    gateway = InvalidPredicateGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 2
    assert result["metrics"]["classification_decisions"] == {
        OTHER_UNSUPPORTED: 1,
    }
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert len(database.decisions) == 1
    decision = database.decisions[0]
    assert decision.pair_identity == database.opportunities[0].pair_identity
    assert decision.decision == OTHER_UNSUPPORTED
    assert decision.hypotheses == ()
    assert decision.evidence_spans[0].start == (
        database.opportunities[0].context_span[0]
    )
    assert decision.evidence_spans[0].end == (
        database.opportunities[0].context_span[1]
    )
    assert decision.reason_code == "CLASSIFIER_TERMINAL_PROTOCOL_FAILURE"
    assert database.finished[-1][1]["status"] == "ready"
    assert len(database.endpoints) == 0
    statuses = [
        values["status"] for _, _, values in database.classification_attempts
    ]
    assert statuses == ["RETRYABLE_PROTOCOL", "TERMINAL_FAILED"]
    assert all(status != "ACCEPTED" for status in statuses)
    assert all(
        values["error_type"] == "RelationClassificationError"
        for _, _, values in database.classification_attempts
    )


def test_runtime_does_not_recover_single_pair_negative_raw_parser_failure(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 1)

    class RawNegativeParserFailureGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                pair = json.loads(kwargs["prompt"])["pairs"][0]
                self.last_raw_response_text = json.dumps({
                    "pair_id": pair["pair_id"],
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "NO_SEMANTIC_RELATION",
                })
                raise RelationClassificationError(
                    f"missing pair ids: ['{pair['pair_id']}']",
                )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = RawNegativeParserFailureGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 1
    assert result["metrics"]["classification_decisions"] == {
        OTHER_UNSUPPORTED: 1,
    }
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert len(database.decisions) == 1
    decision = database.decisions[0]
    assert decision.pair_identity == database.opportunities[0].pair_identity
    assert decision.decision == OTHER_UNSUPPORTED
    assert decision.hypotheses == ()
    assert decision.evidence_spans[0].start == (
        database.opportunities[0].context_span[0]
    )
    assert decision.evidence_spans[0].end == (
        database.opportunities[0].context_span[1]
    )
    assert decision.reason_code == "CLASSIFIER_TERMINAL_PROTOCOL_FAILURE"
    assert database.finished[-1][1]["status"] == "ready"
    statuses = [
        values["status"] for _, _, values in database.classification_attempts
    ]
    assert statuses == ["TERMINAL_FAILED"]
    assert all(status != "ACCEPTED" for status in statuses)


def test_runtime_single_pair_classifier_pair_id_drift_fails_closed(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 1)

    class DriftedPairGateway(_ScriptedGateway):
        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_calls += 1
                return {"items": [{
                    "pair_id": "relpair_drifted_from_another_prompt",
                    "decision": NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "NO_SEMANTIC_RELATION",
                }]}
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    gateway = DriftedPairGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 1
    assert result["metrics"]["classification_decisions"] == {
        OTHER_UNSUPPORTED: 1,
    }
    assert result["metrics"]["classified_hypotheses"] == 0
    assert result["metrics"]["verification_attempts"] == 0
    assert result["metrics"]["facts_after_normalization"] == 0
    assert database.finished[-1][1]["status"] == "ready"
    assert database.decisions[0].reason_code == (
        "CLASSIFIER_TERMINAL_PROTOCOL_FAILURE"
    )
    assert database.classification_attempts[0][2]["status"] == (
        "TERMINAL_FAILED"
    )
    assert database.classification_attempts[0][2]["error_type"] == (
        "RelationClassificationError"
    )
    assert len(database.endpoints) == 0


def test_runtime_classifier_truncation_retry_expands_output_budget(monkeypatch):
    monkeypatch.setattr(Config, "relation_classifier_output_reserve", 2048)
    monkeypatch.setattr(Config, "relation_max_attempts", 3)

    class TruncatingOnceGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.classifier_output_budgets = []

        def json_chat(self, **kwargs):
            if "classifier" in str(kwargs["system"]):
                self.classifier_output_budgets.append(
                    int(kwargs["max_output_tokens"]),
                )
                if len(self.classifier_output_budgets) == 1:
                    raise OllamaProtocolError(
                        "Ollama response was truncated by output length",
                    )
            return super().json_chat(**kwargs)

    database = _RuntimeDatabase()
    gateway = TruncatingOnceGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert result["metrics"]["classifier_failures"] == 1
    assert gateway.classifier_output_budgets[:2] == [2048, 4096]
    retry = next(
        values for _, _, values in database.classification_attempts
        if values["status"] == "RETRYABLE_TRUNCATED"
    )
    accepted_retry = next(
        values for _, _, values in database.classification_attempts
        if (
            values["status"] == "ACCEPTED"
            and values["metadata"]["max_output_tokens"] == 4096
        )
    )
    assert retry["error_message"] == "Ollama response was truncated by output length"
    assert retry["metadata"]["max_output_tokens"] == 2048
    assert accepted_retry["metadata"]["max_output_tokens"] == 4096
    assert database.finished[-1][1]["status"] == "ready"


def test_runtime_classifier_salvages_valid_batch_members_and_retries_only_invalid(
    monkeypatch,
):
    monkeypatch.setattr(Config, "relation_max_attempts", 2)

    class OneInvalidMemberGateway(_ScriptedGateway):
        def __init__(self):
            super().__init__()
            self.injected = False

        def json_chat(self, **kwargs):
            system = str(kwargs["system"])
            result = super().json_chat(**kwargs)
            if "classifier" not in system or self.injected:
                return result
            payload = json.loads(kwargs["prompt"])
            if len(payload.get("pairs") or []) < 2:
                return result
            self.injected = True
            pair = payload["pairs"][1]
            result["items"][1] = {
                "pair_id": pair["pair_id"],
                "decision": HYPOTHESES,
                "hypotheses": [{
                    "subject_occurrence_id": "LEFT",
                    "predicate": "not_allowed_for_this_pair",
                    "object_occurrence_id": "RIGHT",
                    "evidence_spans": [{
                        "start": pair["context_span"][0],
                        "end": pair["context_span"][1],
                    }],
                    "classifier_confidence": 0.9,
                }],
            }
            return result

    database = _RuntimeDatabase()
    _keep_runtime_chunks(database, {1})
    content = "NO_CASE Left1 connects Right1 and Extra1."
    database.chunks[0]["content"] = content
    database.chunks[0]["content_hash"] = "chunk-hash-partial-classifier"
    extra_start = content.index("Extra1")
    database.mentions.append({
        "mention_key": "no-extra",
        "chunk_id": 1,
        "surface": "Extra1",
        "evidence_start": extra_start,
        "evidence_end": extra_start + len("Extra1"),
    })
    database.assignments.append({
        "mention_key": "no-extra",
        "canonical_key": "canonical-extra",
        "entity_type": "Product",
        "policy_version": "assignment-v1",
        "revision_id": "revision-runtime",
    })
    gateway = OneInvalidMemberGateway()

    result = GroundedRelationRuntime(database, gateway).run_namespace(
        namespace_id="namespace-runtime",
        project_id=1,
    )

    assert gateway.classifier_calls == 2
    assert result["metrics"]["classifier_failures"] == 1
    assert result["metrics"]["classifier_partial_batches"] == 1
    assert result["metrics"]["classifier_partial_decisions"] == 2
    assert result["metrics"]["classifier_partial_fallbacks"] == 1
    partial_attempts = [
        values for _, _, values in database.classification_attempts
        if values["metadata"].get("partial_batch_recovery")
    ]
    assert [item["status"] for item in partial_attempts].count(
        CLASSIFIER_ACCEPTED
    ) == 2
    assert [item["status"] for item in partial_attempts].count(
        CLASSIFIER_RETRYABLE_PROTOCOL
    ) == 1
    assert len(database.decisions) == 3
    assert all(item.decision == NO_RELATION for item in database.decisions)
    assert database.finished[-1][1]["status"] == "ready"


def test_single_pair_classifier_object_is_only_shape_normalized_not_reinterpreted():
    class SingleObjectGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {
                "pair_id": "pair-1",
                "decision": NO_RELATION,
                "hypotheses": [],
            }

    runtime = GroundedRelationRuntime(None, SingleObjectGateway())
    result = runtime._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-1",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_single_pair_classifier_identity_is_not_bound_by_caller(monkeypatch):
    class SingleObjectGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {
                "decision": NO_RELATION,
                "hypotheses": [],
            }

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    runtime = GroundedRelationRuntime(None, SingleObjectGateway())
    result = runtime._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-from-caller")],
        content="",
        schema=_schema(),
    )

    assert result == {
        "decision": NO_RELATION,
        "hypotheses": [],
    }


def test_multi_pair_classifier_identity_is_never_inferred(monkeypatch):
    raw = {
        "decision": NO_RELATION,
        "hypotheses": [],
    }

    class AmbiguousObjectGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return raw

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    runtime = GroundedRelationRuntime(None, AmbiguousObjectGateway())
    result = runtime._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[
            SimpleNamespace(pair_identity="pair-1"),
            SimpleNamespace(pair_identity="pair-2"),
        ],
        content="",
        schema=_schema(),
    )

    assert result is raw


@pytest.mark.parametrize("hypotheses", ["missing", None])
def test_negative_classifier_null_hypotheses_is_normalized_to_empty(
    monkeypatch, hypotheses,
):
    decision = {
        "pair_id": "pair-1",
        "decision": NO_RELATION,
    }
    if hypotheses is None:
        decision["hypotheses"] = None

    class NegativeGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {"items": [decision]}

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = GroundedRelationRuntime(None, NegativeGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-1")],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-1",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_classifier_result_wrapper_is_shape_normalized(monkeypatch):
    class ResultWrapperGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {
                "result": [{
                    "pair_id": "pair-1",
                    "decision": NO_RELATION,
                    "hypotheses": [],
                }],
            }

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = GroundedRelationRuntime(None, ResultWrapperGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-1")],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-1",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_single_pair_classifier_pair_id_drift_is_not_overwritten(monkeypatch):
    class DriftedPairGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return [{
                "pair_id": "pair-drifted",
                "decision": NO_RELATION,
                "hypotheses": [],
            }]

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = GroundedRelationRuntime(None, DriftedPairGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-from-caller")],
        content="",
        schema=_schema(),
    )

    assert result == [{
        "pair_id": "pair-drifted",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]


def test_multi_pair_classifier_pair_id_drift_is_not_bound(monkeypatch):
    raw = [{
        "pair_id": "pair-drifted",
        "decision": NO_RELATION,
        "hypotheses": [],
    }]

    class DriftedPairGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return raw

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = GroundedRelationRuntime(None, DriftedPairGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[
            SimpleNamespace(pair_identity="pair-1"),
            SimpleNamespace(pair_identity="pair-2"),
        ],
        content="",
        schema=_schema(),
    )

    assert result is not raw
    assert result == raw


def test_single_pair_empty_classifier_items_is_not_negative(monkeypatch):
    class EmptyItemsGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {"items": []}

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = GroundedRelationRuntime(None, EmptyItemsGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[SimpleNamespace(pair_identity="pair-from-caller")],
        content="",
        schema=_schema(),
    )

    assert result == []


def test_multi_pair_empty_classifier_items_is_not_negative(monkeypatch):
    class EmptyItemsGateway:
        @staticmethod
        def json_chat(**_kwargs):
            return {"items": []}

    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_prompt",
        lambda *_args, **_kwargs: "{}",
    )
    monkeypatch.setattr(
        "knowledge_forge.relation_runtime.v2.classifier_response_schema",
        lambda *_args, **_kwargs: {},
    )
    result = GroundedRelationRuntime(None, EmptyItemsGateway())._classify(
        model="fixture",
        context_budget_tokens=4096,
        opportunities=[
            SimpleNamespace(pair_identity="pair-1"),
            SimpleNamespace(pair_identity="pair-2"),
        ],
        content="",
        schema=_schema(),
    )

    assert result == []
