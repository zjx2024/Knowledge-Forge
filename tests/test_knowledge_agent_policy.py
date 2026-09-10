"""Layered policy contracts and deterministic rail behavior."""

from __future__ import annotations

from knowledge_forge.agent.policy import (
    LAYERED_POLICY_PROTOCOL,
    execution_policy_decision,
    input_policy_decision,
    output_policy_decision,
    record_policy_decision,
    retrieval_policy_decision,
)
from knowledge_forge.agent.scope import QueryScopeDecision
from knowledge_forge.agent.sufficiency import SufficiencyResult


def _scope(route: str = "knowledge") -> QueryScopeDecision:
    return QueryScopeDecision(
        route=route,
        confidence=0.92,
        method="strong_retrieval_signals",
        reason_code=(
            "strong_cross_encoder_alignment"
            if route == "knowledge" else "model_detected_out_of_scope"
        ),
    )


def _sufficiency(state: str = "sufficient") -> SufficiencyResult:
    return SufficiencyResult(
        state=state,
        score=0.82,
        aspect_coverage=0.75,
        relevance=0.91,
        semantic_relevance=0.72,
        reranker_used_model=True,
        provenance=1.0,
        independent_support=0.75,
        graph_connectivity=0.8,
        conflict_penalty=0.0,
        reasons=(),
    )


def test_execution_rail_is_strategy_scoped_and_fail_closed():
    allowed = execution_policy_decision(
        tool_name="retrieve.knowledge_forge",
        strategy="knowledge_forge",
        project_id=7,
    )
    denied = execution_policy_decision(
        tool_name="retrieve.graphrag_native",
        strategy="knowledge_forge",
        project_id=7,
    )

    assert allowed.action == "allow"
    assert allowed.allows_continue is True
    assert denied.action == "deny"
    assert denied.allows_continue is False


def test_input_rail_routes_non_knowledge_without_treating_it_as_allowed_knowledge():
    decision = input_policy_decision(_scope("out_of_scope"))

    assert decision.action == "route"
    assert decision.route == "out_of_scope"
    assert decision.details["knowledge_answer_allowed"] is False


def test_retrieval_rail_separates_grounded_qualified_and_refused_routes():
    grounded = retrieval_policy_decision(
        scope=_scope(), sufficiency=_sufficiency(),
        selected_count=6, provenance_valid_count=18,
    )
    qualified = retrieval_policy_decision(
        scope=_scope(), sufficiency=_sufficiency("conflict"),
        selected_count=5, provenance_valid_count=12,
    )
    refused = retrieval_policy_decision(
        scope=_scope("out_of_scope"), sufficiency=_sufficiency("insufficient"),
        selected_count=0, provenance_valid_count=8,
    )

    assert grounded.action == "allow"
    assert grounded.route == "grounded_knowledge_answer"
    assert qualified.action == "qualify"
    assert qualified.route == "qualified_knowledge_answer"
    assert refused.action == "refuse"
    assert refused.route == "out_of_scope"


def test_output_rail_requires_citations_only_for_knowledge_claims():
    grounded = output_policy_decision(
        route="knowledge", sufficiency_state="sufficient",
        answer_text="有依据的回答。", claim_count=2, citation_count=2,
    )
    uncited = output_policy_decision(
        route="knowledge", sufficiency_state="sufficient",
        answer_text="没有引用的回答。", claim_count=1, citation_count=0,
    )
    capability = output_policy_decision(
        route="agent_meta", sufficiency_state="not_required",
        answer_text="我是知识库问答智能体。", claim_count=0, citation_count=0,
    )
    uncertain = output_policy_decision(
        route="uncertain", sufficiency_state="insufficient",
        answer_text="当前无法可靠判断问题范围。", claim_count=0, citation_count=0,
    )
    degraded = output_policy_decision(
        route="service_degraded", sufficiency_state="insufficient",
        answer_text="模型服务暂时不可用。", claim_count=0, citation_count=0,
    )

    assert grounded.action == "allow"
    assert uncited.action == "deny"
    assert capability.action == "allow"
    assert uncertain.action == "allow"
    assert degraded.action == "allow"


def test_policy_ledger_upserts_stable_checkpoints():
    metadata: dict = {}
    first = execution_policy_decision(
        tool_name="rerank.cross_encoder_http",
        strategy="knowledge_forge", project_id=7,
    )
    record_policy_decision(metadata, first)
    record_policy_decision(metadata, first)

    control = metadata["policy_control"]
    assert control["protocol"] == LAYERED_POLICY_PROTOCOL
    assert len(control["decisions"]) == 1
    assert control["latest_layer"] == "execution"
