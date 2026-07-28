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
    assert "Do not invent endpoints" in prompt["instruction"]


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


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"subject_occurrence_id": "invented"}, "endpoint"),
        ({"predicate": "invented_rel"}, "predicate"),
        ({"evidence_spans": [{"start": 5, "end": 9}]}, "evidence"),
        ({"confidence": 2.0}, "confidence"),
    ],
)
def test_proposer_fail_closed_on_invented_endpoint_predicate_evidence_or_confidence(
    overrides, message,
):
    with pytest.raises(v2.ShadowClassificationError, match=message):
        v2.parse_proposer_response(
            context_id="ctx-1",
            content=CONTENT,
            mentions=MENTIONS,
            logical_schema=_schema(),
            raw=_response(**overrides),
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
