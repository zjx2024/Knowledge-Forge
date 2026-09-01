"""Layered, auditable policy decisions for the knowledge agent.

The project borrows the useful lifecycle separation of programmable rails
without depending on a second orchestration runtime.  Every decision is small,
deterministic where possible, and safe to persist in the public execution
ledger.  Domain facts, model prompts and hidden reasoning never belong here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from ..unicode_safety import sanitize_unicode_data, sanitize_unicode_text
from .scope import QueryScopeDecision
from .sufficiency import SufficiencyResult


LAYERED_POLICY_PROTOCOL = "knowledge-agent-layered-policy-v1"

POLICY_LAYERS = frozenset({
    "access", "input", "dialog", "retrieval", "execution", "output",
})
POLICY_ACTIONS = frozenset({
    "allow", "route", "clarify", "qualify", "refuse", "deny", "skip",
})

_COMMON_TOOLS = frozenset({
    "context.resolve.langgraph",
    "scope.route.policy",
    "scope.evidence_alignment",
    "plan.retrieval.policy",
    "rerank.cross_encoder_http",
    "answer.ollama_json",
    "answer.refusal_policy",
    "answer.scope_policy",
})
_STRATEGY_TOOLS = {
    "knowledge_forge": frozenset({
        "retrieve.knowledge_forge",
        "retrieve.knowledge_forge.supplemental",
    }),
    "graphrag_native": frozenset({
        "retrieve.graphrag_native",
        "retrieve.graphrag_native.supplemental",
    }),
}


@dataclass(frozen=True)
class PolicyDecision:
    """One bounded decision made at a named lifecycle checkpoint."""

    layer: str
    checkpoint: str
    action: str
    reason_code: str
    route: str = ""
    method: str = "deterministic_policy"
    confidence: float = 1.0
    model_used: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    protocol: str = LAYERED_POLICY_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != LAYERED_POLICY_PROTOCOL:
            raise ValueError("invalid layered-policy protocol")
        if self.layer not in POLICY_LAYERS:
            raise ValueError("invalid layered-policy layer")
        if self.action not in POLICY_ACTIONS:
            raise ValueError("invalid layered-policy action")
        if not str(self.checkpoint or "").strip():
            raise ValueError("layered-policy checkpoint is required")
        if not str(self.reason_code or "").strip():
            raise ValueError("layered-policy reason code is required")

    @property
    def allows_continue(self) -> bool:
        return self.action in {"allow", "route", "qualify", "skip"}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = max(0.0, min(1.0, float(self.confidence)))
        return sanitize_unicode_data(payload)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyDecision":
        details = value.get("details")
        return cls(
            layer=str(value.get("layer") or ""),
            checkpoint=str(value.get("checkpoint") or ""),
            action=str(value.get("action") or ""),
            reason_code=str(value.get("reason_code") or ""),
            route=str(value.get("route") or ""),
            method=str(value.get("method") or "deterministic_policy"),
            confidence=float(value.get("confidence") or 0.0),
            model_used=bool(value.get("model_used")),
            details=dict(details) if isinstance(details, Mapping) else {},
            protocol=str(value.get("protocol") or LAYERED_POLICY_PROTOCOL),
        )


def record_policy_decision(
    metadata: dict[str, Any], decision: PolicyDecision,
) -> dict[str, Any]:
    """Upsert a decision without discarding decisions from other layers.

    A checkpoint is stable for single-decision rails.  Execution checkpoints
    include the tool name, so the policy ledger keeps the latest verdict for
    each allowed tool; individual invocations remain authoritative in the
    separate ``agent_tool_calls`` ledger.
    """

    existing = metadata.get("policy_control")
    control = dict(existing) if isinstance(existing, Mapping) else {}
    rows = [
        dict(item) for item in (control.get("decisions") or [])
        if isinstance(item, Mapping)
        and str(item.get("checkpoint") or "") != decision.checkpoint
    ]
    rows.append(decision.to_dict())
    control.update({
        "protocol": LAYERED_POLICY_PROTOCOL,
        "decisions": rows[-32:],
        "latest_layer": decision.layer,
        "latest_action": decision.action,
        "latest_route": decision.route,
    })
    metadata["policy_control"] = sanitize_unicode_data(control)
    return metadata


def access_policy_decision(
    *, project_id: int, strategy: str, authorized: bool,
) -> PolicyDecision:
    return PolicyDecision(
        layer="access",
        checkpoint="access.project_scope",
        action="allow" if authorized else "deny",
        route="selected_knowledge_base" if authorized else "blocked",
        reason_code=(
            "project_strategy_and_session_authorized"
            if authorized else "project_access_not_authorized"
        ),
        details={
            "project_id": max(0, int(project_id)),
            "strategy": sanitize_unicode_text(str(strategy or ""))[:80],
            "project_isolation": True,
        },
    )


def input_policy_decision(scope: QueryScopeDecision) -> PolicyDecision:
    return PolicyDecision(
        layer="input",
        checkpoint="input.question_route",
        action="allow" if scope.route == "knowledge" else "route",
        route=scope.route,
        reason_code=scope.reason_code,
        method=scope.method,
        confidence=scope.confidence,
        model_used=scope.model_used,
        details={
            "knowledge_answer_allowed": scope.route == "knowledge",
            "general_knowledge_fallback": False,
        },
    )


def dialog_policy_decision(
    *, dependency: str, branch: str, context_used: bool,
    model_used: bool, skipped_reason: str = "",
) -> PolicyDecision:
    if skipped_reason:
        return PolicyDecision(
            layer="dialog",
            checkpoint="dialog.context_route",
            action="skip",
            route=branch or "skipped",
            reason_code=skipped_reason,
            details={"context_used": False},
        )
    ambiguous = dependency == "ambiguous" or branch == "clarify"
    return PolicyDecision(
        layer="dialog",
        checkpoint="dialog.context_route",
        action="clarify" if ambiguous else "allow",
        route="clarify" if ambiguous else "knowledge",
        reason_code=(
            "ambiguous_conversation_reference"
            if ambiguous else
            "conversation_context_resolved"
            if context_used else "question_is_context_independent"
        ),
        method="langgraph_context_resolution",
        model_used=bool(model_used),
        details={
            "dependency": dependency,
            "context_used": bool(context_used),
        },
    )


def skipped_policy_decision(
    layer: str, *, checkpoint: str, route: str, reason_code: str,
) -> PolicyDecision:
    return PolicyDecision(
        layer=layer,
        checkpoint=checkpoint,
        action="skip",
        route=route,
        reason_code=reason_code,
    )


def retrieval_policy_decision(
    *, scope: QueryScopeDecision, sufficiency: SufficiencyResult,
    selected_count: int, provenance_valid_count: int,
) -> PolicyDecision:
    details = {
        "selected_evidence": max(0, int(selected_count)),
        "provenance_valid_candidates": max(0, int(provenance_valid_count)),
        "sufficiency_state": sufficiency.state,
        "sufficiency_score": round(float(sufficiency.score), 6),
        "scope_route": scope.route,
    }
    if scope.route != "knowledge":
        return PolicyDecision(
            layer="retrieval",
            checkpoint="retrieval.evidence_gate",
            action="refuse",
            route=scope.route,
            reason_code=scope.reason_code,
            method=scope.method,
            confidence=scope.confidence,
            model_used=scope.model_used,
            details=details,
        )
    if sufficiency.state == "insufficient":
        return PolicyDecision(
            layer="retrieval",
            checkpoint="retrieval.evidence_gate",
            action="refuse",
            route="evidence_insufficient",
            reason_code="retrieved_evidence_is_insufficient",
            details=details,
        )
    if sufficiency.state in {"retryable", "ambiguous", "conflict"}:
        return PolicyDecision(
            layer="retrieval",
            checkpoint="retrieval.evidence_gate",
            action="qualify",
            route="qualified_knowledge_answer",
            reason_code=f"evidence_{sufficiency.state}",
            details=details,
        )
    return PolicyDecision(
        layer="retrieval",
        checkpoint="retrieval.evidence_gate",
        action="allow",
        route="grounded_knowledge_answer",
        reason_code="scope_and_evidence_gate_passed",
        confidence=max(scope.confidence, sufficiency.score),
        details=details,
    )


def execution_policy_decision(
    *, tool_name: str, strategy: str, project_id: int,
) -> PolicyDecision:
    name = sanitize_unicode_text(str(tool_name or "")).strip()
    strategy_name = sanitize_unicode_text(str(strategy or "")).strip()
    allowed = name in _COMMON_TOOLS or name in _STRATEGY_TOOLS.get(
        strategy_name, frozenset(),
    )
    return PolicyDecision(
        layer="execution",
        checkpoint=f"execution.tool:{name or 'missing'}",
        action="allow" if allowed else "deny",
        route="read_only_internal_tool" if allowed else "blocked_tool",
        reason_code=(
            "tool_allowed_for_selected_strategy"
            if allowed else "tool_not_allowed_for_selected_strategy"
        ),
        details={
            "tool_name": name[:160],
            "strategy": strategy_name[:80],
            "project_id": max(0, int(project_id)),
            "read_only_knowledge_access": True,
        },
    )


def output_policy_decision(
    *, route: str, sufficiency_state: str, answer_text: str,
    claim_count: int, citation_count: int,
) -> PolicyDecision:
    route_name = sanitize_unicode_text(str(route or "")).strip()
    state = sanitize_unicode_text(str(sufficiency_state or "")).strip()
    claims = max(0, int(claim_count))
    citations = max(0, int(citation_count))
    has_text = bool(sanitize_unicode_text(str(answer_text or "")).strip())
    no_fact_routes = {
        "agent_meta", "out_of_scope", "unsupported", "clarify",
        "controlled_refusal",
    }
    if route_name in no_fact_routes:
        allowed = has_text and claims == 0 and citations == 0
        reason = (
            "non_fact_policy_response_is_clean"
            if allowed else "non_fact_response_contains_unverified_claims"
        )
    elif state == "insufficient":
        allowed = has_text and claims == 0 and citations == 0
        reason = (
            "evidence_refusal_is_clean"
            if allowed else "insufficient_answer_contains_unverified_claims"
        )
    else:
        allowed = has_text and claims > 0 and citations > 0
        reason = (
            "grounded_claims_and_citations_validated"
            if allowed else "grounded_answer_failed_output_contract"
        )
    return PolicyDecision(
        layer="output",
        checkpoint="output.answer_gate",
        action="allow" if allowed else "deny",
        route=route_name or "knowledge",
        reason_code=reason,
        details={
            "sufficiency_state": state,
            "claim_count": claims,
            "citation_count": citations,
            "has_answer_text": has_text,
        },
    )
