from __future__ import annotations

import json

import pytest

from knowledge_forge import relation_shadow_v2 as v2
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


CONTENT = "Acme uses Alloy."
MENTIONS = [
    {
        "occurrence_key": "occ-acme",
        "mention_id": "mention-acme",
        "surface": "Acme",
        "mention_start": 0,
        "mention_end": 4,
        "canonical_id": "entity-acme",
        "assignment_version": "assignment-v1",
        "type": "Organization",
    },
    {
        "occurrence_key": "occ-alloy",
        "mention_id": "mention-alloy",
        "surface": "Alloy",
        "mention_start": 10,
        "mention_end": 15,
        "canonical_id": "entity-alloy",
        "assignment_version": "assignment-v1",
        "type": "Product",
    },
]


def _schema() -> dict:
    return logical_schema(
        predicate_policy(
            "alpha_rel",
            domain=["Organization"],
            range_=["Product"],
        ),
        version="schema-proposer-v1",
    )


def _response(**proposal_overrides):
    proposal = {
        "subject_occurrence_id": "occ-acme",
        "predicate": "alpha_rel",
        "object_occurrence_id": "occ-alloy",
        "evidence_spans": [{"start": 0, "end": len(CONTENT)}],
        "qualifiers": [],
        "confidence": 0.8,
    }
    proposal.update(proposal_overrides)
    return {
        "proposals": [proposal],
        "missing_endpoint_mentions": [],
        "coverage": {"complete": True, "uncovered_cues": []},
    }


def test_proposer_prompt_uses_occurrence_universe_and_predicate_metadata():
    prompt = json.loads(v2.proposer_prompt(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
    ))

    assert prompt["occurrences"][0]["occurrence_id"] == "occ-acme"
    assert prompt["predicates"]["alpha_rel"]["semantic_definition"]
    assert "positive_examples" not in prompt["predicates"]["alpha_rel"]
    assert "negative_examples" not in prompt["predicates"]["alpha_rel"]
    assert "conflicts_with" not in prompt["predicates"]["alpha_rel"]
    assert prompt["protocol_feedback"] == ""
    assert prompt["required_evidence_span"] == {"start": 0, "end": len(CONTENT)}
    assert "Do not invent endpoints" in prompt["instruction"]
    assert "coverage as an object" in prompt["instruction"]


def test_proposer_prompt_carries_retry_protocol_feedback():
    prompt = json.loads(v2.proposer_prompt(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        protocol_feedback="coverage must be an object",
    ))

    assert prompt["protocol_feedback"] == "coverage must be an object"


def test_proposer_prompt_focuses_one_atomic_scope_with_absolute_offsets():
    content = "Ignore this. Acme uses Alloy. Ignore that."
    start = content.index("Acme")
    end = content.index(" Ignore that")
    mentions = [
        {**MENTIONS[0], "mention_start": start, "mention_end": start + 4},
        {
            **MENTIONS[1],
            "mention_start": content.index("Alloy"),
            "mention_end": content.index("Alloy") + 5,
        },
    ]

    prompt = json.loads(v2.proposer_prompt(
        context_id="ctx-atomic",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        required_evidence_span=(start, end),
    ))
    schema = v2.proposer_response_schema(
        mentions=mentions,
        logical_schema=_schema(),
        content_length=len(content),
        required_evidence_span=(start, end),
    )

    assert prompt["context"] == {
        "context_id": "ctx-atomic",
        "text": content[start:end],
        "start": start,
        "end": end,
    }
    assert prompt["required_evidence_span"] == {
        "start": start, "end": end,
    }
    span_schema = schema["properties"]["proposals"]["items"][
        "properties"
    ]["evidence_spans"]["items"]["properties"]
    assert span_schema["start"]["minimum"] == start
    assert span_schema["end"]["maximum"] == end


def test_proposer_prompt_expands_coordinated_arguments_without_peer_edges():
    content = "系统连接模块甲、模块乙和模块丙。"
    prompt = json.loads(v2.proposer_prompt(
        context_id="ctx-list",
        content=content,
        mentions=[],
        logical_schema=_schema(),
    ))

    instruction = prompt["instruction"]
    assert "分别展开到每个并列成员" in instruction
    assert "不得把并列成员彼此连接" in instruction
    assert "事件、状态、条件端点" in instruction


def test_proposer_response_schema_constrains_runtime_universe():
    schema = v2.proposer_response_schema(
        mentions=MENTIONS,
        logical_schema=_schema(),
        content_length=len(CONTENT),
        max_proposals=7,
    )
    proposal_schema = schema["properties"]["proposals"]
    properties = proposal_schema["items"]["properties"]

    assert proposal_schema["maxItems"] == 7
    assert properties["subject_occurrence_id"]["enum"] == [
        "occ-acme",
        "occ-alloy",
    ]
    assert properties["object_occurrence_id"]["enum"] == [
        "occ-acme",
        "occ-alloy",
    ]
    assert properties["predicate"]["enum"] == ["alpha_rel"]
    span_properties = properties["evidence_spans"]["items"]["properties"]
    assert span_properties["start"]["maximum"] == len(CONTENT) - 1
    assert span_properties["end"]["maximum"] == len(CONTENT)


def test_proposer_response_becomes_candidate_hypothesis_for_same_verifier_path():
    batch = v2.parse_proposer_response(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw=_response(),
        proposer_model_revision="fixture-proposer",
    )
    proposal = batch.proposals[0]
    plan = v2.plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-1",
        chunk_key="chunk-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        run_id="run-1",
        candidate_policy_version="candidate-v1",
    )
    hypothesis = v2.proposal_to_hypothesis(
        proposal, plan.opportunities[0], _schema(),
    )

    assert hypothesis.status == "PROPOSED"
    assert hypothesis.subject_occurrence_key == "occ-acme"
    assert hypothesis.predicate == "alpha_rel"
    assert hypothesis.classifier_prompt_version == v2.SHADOW_RELATION_PROPOSER_PROMPT_VERSION
    assert v2.validate_hypothesis(
        hypothesis,
        plan.opportunities[0],
        content=CONTENT,
        logical_schema=_schema(),
    ).status == v2.PASS


def test_legacy_nested_proposer_envelope_is_strictly_normalized():
    batch = v2.parse_proposer_response(
        context_id="ctx-legacy",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw={
            "proposals": [{
                "subject_endpoint": {
                    "occurrence_id": "occ-acme",
                    "canonical_id": "entity-acme",
                    "surface": "Acme",
                },
                "predicate": "alpha_rel",
                "object_endpoint": {
                    "occurrence_id": "occ-alloy",
                    "canonical_id": "entity-alloy",
                    "surface": "Alloy",
                },
                "evidence_span": {"start": 0, "end": len(CONTENT)},
            }],
        },
        proposer_model_revision="legacy-fixture",
    )

    assert len(batch.proposals) == 1
    assert batch.proposals[0].subject_occurrence_key == "occ-acme"
    assert batch.proposals[0].object_occurrence_key == "occ-alloy"
    assert batch.proposals[0].predicate == "alpha_rel"
    assert batch.proposals[0].evidence_spans == (
        v2.EvidenceSpan(0, len(CONTENT)),
    )
    assert "legacy_nested_endpoint_proposals_normalized" in (
        batch.coverage["_parser_repairs"]
    )


def test_broad_proposer_evidence_is_narrowed_to_shared_atomic_scope() -> None:
    content = "Acme uses Alloy. Gamma observes Delta."
    mentions = [dict(item) for item in MENTIONS]
    batch = v2.parse_proposer_response(
        context_id="ctx-atomic",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        raw={
            "proposals": [{
                **_response()["proposals"][0],
                "evidence_spans": [{"start": 0, "end": len(content)}],
            }],
            "missing_endpoint_mentions": [],
            "coverage": {"complete": True, "uncovered_cues": []},
        },
    )
    plan = v2.plan_pair_opportunities(
        project_id=1,
        document_revision_id="revision-atomic",
        chunk_key="chunk-atomic",
        content=content,
        mentions=mentions,
        logical_schema=_schema(),
        run_id="run-atomic",
        candidate_policy_version="candidate-v1",
    )

    hypothesis = v2.proposal_to_hypothesis(
        batch.proposals[0], plan.opportunities[0], _schema(),
    )

    assert hypothesis.evidence_spans == (
        v2.EvidenceSpan(
            *plan.opportunities[0].provenance["required_atomic_scope"],
        ),
    )


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"subject_occurrence_id": "invented"}, "endpoint_not_in_occurrence_universe"),
        ({"predicate": "invented_rel"}, "predicate_not_in_schema"),
    ],
)
def test_proposer_drops_unsafe_candidate_without_failing_context(
    overrides, reason,
):
    batch = v2.parse_proposer_response(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw=_response(**overrides),
    )

    assert batch.proposals == ()
    assert batch.coverage["complete"] is False
    assert batch.coverage["_parser_rejected_proposals"] == [{
        "index": 0,
        "reason": reason,
    }]


def test_proposer_non_decision_fields_are_safely_repaired():
    batch = v2.parse_proposer_response(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw=_response(qualifiers="not-array", confidence=2.0),
    )

    proposal = batch.proposals[0]
    assert proposal.qualifiers == ()
    assert proposal.confidence == 1.0
    assert "proposal_0_qualifiers_defaulted" in batch.coverage["_parser_repairs"]
    assert "proposal_0_confidence_clamped" in batch.coverage["_parser_repairs"]


def test_proposer_top_level_proposals_shape_defaults_to_no_proposals():
    raw = _response()
    raw["proposals"] = "not-array"
    batch = v2.parse_proposer_response(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw=raw,
    )

    assert batch.proposals == ()
    assert "proposals_type_defaulted" in batch.coverage["_parser_repairs"]


def test_proposer_audit_only_fields_are_safely_defaulted():
    raw = _response()
    raw["missing_endpoint_mentions"] = None
    raw["coverage"] = []

    batch = v2.parse_proposer_response(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw=raw,
    )

    assert len(batch.proposals) == 1
    assert batch.missing_endpoint_mentions == ()
    assert batch.coverage["complete"] is False
    assert batch.coverage["uncovered_cues"] == []
    assert "missing_endpoint_mentions_defaulted" in (
        batch.coverage["_parser_repairs"]
    )
    assert "coverage_type_defaulted" in batch.coverage["_parser_repairs"]


@pytest.mark.parametrize(
    "evidence_spans",
    [
        {"start": 0, "end": len(CONTENT)},
        [{"start": "bad", "end": len(CONTENT)}],
        [{"start": 5, "end": 9}],
        [],
    ],
)
def test_proposer_evidence_shape_errors_fall_back_to_full_context(
    evidence_spans,
):
    batch = v2.parse_proposer_response(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw=_response(evidence_spans=evidence_spans),
    )

    proposal = batch.proposals[0]
    assert [(span.start, span.end) for span in proposal.evidence_spans] == [
        (0, len(CONTENT)),
    ]
    assert any(
        item.startswith("proposal_0_evidence")
        for item in batch.coverage["_parser_repairs"]
    )


def test_missing_endpoint_mentions_are_audit_only_not_hypotheses():
    batch = v2.parse_proposer_response(
        context_id="ctx-1",
        content=CONTENT,
        mentions=MENTIONS,
        logical_schema=_schema(),
        raw={
            "proposals": [],
            "missing_endpoint_mentions": [{
                "surface": "Missing Endpoint",
                "source_anchor": "ctx-1",
            }],
            "coverage": {"complete": False, "uncovered_cues": ["uses"]},
        },
    )

    assert batch.proposals == ()
    assert batch.missing_endpoint_mentions[0]["surface"] == "Missing Endpoint"
