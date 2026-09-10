"""Bounded evidence workflow with a durable LangGraph context preflight."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import Any

from ..config import Config
from ..construction_strategies import (
    GRAPHRAG_NATIVE_STRATEGY,
    KNOWLEDGE_FORGE_STRATEGY,
)
from ..database import JobCancellationRequested
from ..ollama import ConservativeTokenEstimator, OllamaCancelled
from .answer import generate_answer, refusal_payload
from .citations import validate_and_render
from .context import ConversationContextGraph
from .contracts import EvidenceItem, RetrievalResult
from .fusion import (
    apply_final_ranking,
    context_selection_capacity,
    deduplicate_evidence,
    relevance_filtered_context_pool,
    reciprocal_rank_fuse,
    select_context,
)
from .planning import (
    RetrievalPlan,
    apply_retrieval_plan,
    plan_retrieval,
    plan_supplemental_retrieval,
)
from .policy import (
    PolicyDecision,
    access_policy_decision,
    dialog_policy_decision,
    execution_policy_decision,
    input_policy_decision,
    output_policy_decision,
    record_policy_decision,
    retrieval_policy_decision,
    skipped_policy_decision,
)
from .rerank import AgentReranker
from .retrieval import GraphRAGNativeRetriever, KnowledgeForgeRetriever
from .scope import (
    QueryScopeDecision,
    agent_capability_payload,
    assess_query_scope,
    preflight_query_scope,
    scope_boundary_payload,
)
from .sufficiency import assess_sufficiency


LOGGER = logging.getLogger(__name__)


class AgentCancelled(RuntimeError):
    pass


@dataclass
class RankingOutcome:
    pool: list[EvidenceItem]
    ranked: list[EvidenceItem]
    selected: list[EvidenceItem]
    rerank: Any
    plan_execution: dict[str, object]
    context_plan_execution: dict[str, object]
    evidence_token_budget: int
    max_output_tokens: int
    question_tokens: int
    prompt_overhead_tokens: int
    draft_tokens: int
    context_pool_count: int
    context_relevance_floor: float
    model_context_limit: int


class KnowledgeAgentWorkflow:
    """Resolve context, then execute at most one retrieval and answer pass."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.db = service.db
        self.ollama = service.ollama
        self.reranker = AgentReranker()
        self.context = ConversationContextGraph(service)
        self.adapters = {
            KNOWLEDGE_FORGE_STRATEGY: KnowledgeForgeRetriever(service),
            GRAPHRAG_NATIVE_STRATEGY: GraphRAGNativeRetriever(service),
        }

    def _cancelled(self, run_id: str) -> bool:
        return self.db.agent_run_cancel_requested(run_id)

    def _assert_active(self, run_id: str) -> None:
        if self._cancelled(run_id):
            raise AgentCancelled("用户已终止本次问答")

    def _stage(
        self, run_id: str, project_id: int, sequence: int, name: str,
        *, status: str = "running", input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None, error: str = "",
    ) -> None:
        self.db.update_agent_run(run_id, stage=name)
        self.db.record_agent_step(
            run_id=run_id, project_id=project_id, sequence=sequence,
            step_type=name, status=status, input_summary=input_summary,
            output_summary=output_summary, error_message=error,
        )
        self.db.append_agent_event(run_id, "progress", {
            "stage": name, "status": status, **(output_summary or {}),
        })

    @staticmethod
    def _record_policy(
        run_metadata: dict[str, Any], decision: PolicyDecision,
    ) -> None:
        record_policy_decision(run_metadata, decision)

    def _start_controlled_tool(
        self,
        *,
        run_id: str,
        project_id: int,
        strategy: str,
        tool_name: str,
        request_summary: dict[str, Any],
        run_metadata: dict[str, Any],
    ) -> str:
        """Enforce the strategy-scoped tool allowlist before persistence."""

        decision = execution_policy_decision(
            tool_name=tool_name,
            strategy=strategy,
            project_id=project_id,
        )
        self._record_policy(run_metadata, decision)
        if not decision.allows_continue:
            self.db.update_agent_run(run_id, metadata=run_metadata)
            raise PermissionError(
                f"内部工具不在当前知识库构建方式的允许范围内：{tool_name}"
            )
        call_id = "atc_" + uuid.uuid4().hex
        self.db.start_agent_tool_call(
            call_id=call_id,
            run_id=run_id,
            project_id=project_id,
            tool_name=tool_name,
            request_summary=request_summary,
        )
        return call_id

    def _complete_scope_response(
        self,
        *,
        run_id: str,
        project_id: int,
        strategy: str,
        sequence: int,
        payload: dict[str, Any],
        scope: QueryScopeDecision,
        effective_mode: str,
        sufficiency: dict[str, Any],
        run_metadata: dict[str, Any],
    ) -> None:
        """Publish a policy response that intentionally has no citations."""

        sequence += 1
        self._stage(
            run_id, project_id, sequence, "answering",
            input_summary={"scope_route": scope.route, "evidence_required": False},
        )
        call_id = self._start_controlled_tool(
            run_id=run_id, project_id=project_id, strategy=strategy,
            tool_name="answer.scope_policy",
            request_summary={"scope_route": scope.route},
            run_metadata=run_metadata,
        )
        validated = validate_and_render(
            payload, [], sufficiency_state="insufficient",
        )
        answer_json = {
            **validated.answer_json,
            "sufficiency_state": str(sufficiency.get("state") or "insufficient"),
            "query_scope": scope.to_dict(),
        }
        self.db.finish_agent_tool_call(
            call_id, status="completed",
            response_summary={
                "scope_route": scope.route,
                "claim_count": 0,
                "citation_count": 0,
            },
        )
        self._stage(
            run_id, project_id, sequence, "answering", status="completed",
            output_summary={
                "scope_route": scope.route,
                "claim_count": 0,
                "citation_count": 0,
            },
        )
        output_decision = output_policy_decision(
            route=scope.route,
            sufficiency_state=str(sufficiency.get("state") or "insufficient"),
            answer_text=validated.answer_text,
            claim_count=len(validated.answer_json.get("claims") or []),
            citation_count=len(validated.citations),
        )
        self._record_policy(run_metadata, output_decision)
        if not output_decision.allows_continue:
            raise RuntimeError("范围响应没有通过输出管控")
        self.db.update_agent_run(
            run_id, sufficiency=sufficiency, metadata=run_metadata,
        )
        message_id = "amsg_" + uuid.uuid4().hex
        chunks = [
            validated.answer_text[index:index + 48]
            for index in range(0, len(validated.answer_text), 48)
        ] or [""]
        stream_events = [{
            "event_type": "answer_start",
            "payload": {
                "message_id": message_id,
                "mode": effective_mode,
                "sufficiency": sufficiency,
                "query_scope": scope.to_dict(),
            },
        }, *({
            "event_type": "answer_delta", "payload": {"delta": chunk},
        } for chunk in chunks), {
            "event_type": "sources", "payload": {"items": []},
        }, {
            "event_type": "answer_metadata",
            "payload": {
                "limitations": answer_json.get("limitations") or [],
                "follow_up_questions": answer_json.get("follow_up_questions") or [],
                "query_scope": scope.to_dict(),
            },
        }]
        self.db.complete_agent_run(
            run_id=run_id, message_id=message_id,
            answer_text=validated.answer_text,
            answer_json=answer_json,
            citations=[], effective_mode=effective_mode,
            stream_events=stream_events,
        )

    def _rank_and_select(
        self,
        *,
        run_id: str,
        project_id: int,
        strategy: str,
        project: dict[str, Any],
        query: str,
        original_query: str,
        evidence: list[EvidenceItem],
        retrieval_limit: int,
        final_limit: int,
        context_max_items: int,
        configured_context_budget: int,
        max_output_tokens: int,
        candidate_draft: str,
        plan: RetrievalPlan | None,
        attempt: int,
        run_metadata: dict[str, Any],
    ) -> RankingOutcome:
        candidates = deduplicate_evidence(evidence)
        candidates = [item for item in candidates if item.provenance_valid]
        for item in candidates:
            item.selected = False
            item.metadata.pop("context_content", None)
            item.metadata.pop("context_token_cost", None)
        pool = reciprocal_rank_fuse(candidates)
        if plan is not None:
            candidates, plan_execution = apply_retrieval_plan(
                pool,
                plan=plan,
                candidate_limit=max(1, int(retrieval_limit)),
            )
        else:
            candidates = list(pool[:max(1, int(retrieval_limit))])
            plan_execution = {
                "activation": "merged",
                "candidate_count": len(candidates),
            }

        call_id = self._start_controlled_tool(
            run_id=run_id, project_id=project_id, strategy=strategy,
            tool_name="rerank.cross_encoder_http",
            request_summary={
                "candidate_count": len(candidates),
                "attempt": max(1, int(attempt)),
            },
            run_metadata=run_metadata,
        )
        try:
            rerank = self.reranker.rerank(query, candidates)
            for item in candidates:
                item.metadata["reranker_used_model"] = bool(rerank.used_model)
                item.metadata["retrieval_attempt"] = max(1, int(attempt))
            self.db.finish_agent_tool_call(
                call_id,
                status="completed" if rerank.used_model else "degraded",
                response_summary={**rerank.to_dict(), "attempt": max(1, int(attempt))},
                error_message=rerank.fallback_reason,
            )
        except Exception as exc:
            try:
                self.db.finish_agent_tool_call(
                    call_id, status="failed", error_message=str(exc),
                )
            finally:
                raise

        ranked = apply_final_ranking(candidates)[:max(1, int(final_limit))]
        context_pool, context_relevance_floor = relevance_filtered_context_pool(
            ranked, reranker_used_model=rerank.used_model,
        )
        model_context_limit = max(
            1024,
            int(project.get("model_max_tokens") or Config.ollama_default_context_budget),
        )
        max_output_tokens = min(
            max(128, int(max_output_tokens)),
            max(256, model_context_limit // 4),
        )
        token_estimator = ConservativeTokenEstimator()
        question_tokens = token_estimator.estimate(query)
        if query != original_query:
            question_tokens += token_estimator.estimate(original_query)
        draft_tokens = (
            token_estimator.estimate(candidate_draft[:1600])
            if candidate_draft.strip() else 0
        )
        prompt_overhead_tokens = 2048 + draft_tokens
        evidence_token_budget = min(
            max(256, int(configured_context_budget)),
            max(
                512,
                model_context_limit
                - max_output_tokens
                - question_tokens
                - prompt_overhead_tokens,
            ),
        )
        context_candidates = context_pool
        context_plan_execution: dict[str, object] = {
            "activation": "not_applied",
            "candidate_count": len(context_pool),
        }
        if plan is not None and plan.activation == "active" and context_pool:
            feasible_context_items = context_selection_capacity(
                context_pool,
                requested_max_items=context_max_items,
            )
            context_candidates, context_plan_execution = apply_retrieval_plan(
                context_pool,
                plan=plan,
                candidate_limit=feasible_context_items,
            )
        selected = select_context(
            context_candidates,
            token_budget=evidence_token_budget,
            max_items=context_max_items,
        )
        return RankingOutcome(
            pool=pool,
            ranked=ranked,
            selected=selected,
            rerank=rerank,
            plan_execution=plan_execution,
            context_plan_execution=context_plan_execution,
            evidence_token_budget=evidence_token_budget,
            max_output_tokens=max_output_tokens,
            question_tokens=question_tokens,
            prompt_overhead_tokens=prompt_overhead_tokens,
            draft_tokens=draft_tokens,
            context_pool_count=len(context_pool),
            context_relevance_floor=context_relevance_floor,
            model_context_limit=model_context_limit,
        )

    def run(self, run_id: str) -> None:
        active_tool = ""
        sequence = 0
        try:
            run = self.db.start_agent_run(run_id)
            if str(run.get("status") or "") != "running":
                return
            project_id = int(run["project_id"])
            original_query = str(run["query"])
            query = original_query
            strategy = str(run["strategy_key"])
            run_metadata = (
                dict(run.get("metadata") or {})
                if isinstance(run.get("metadata"), dict) else {}
            )
            budget = run.get("budget") if isinstance(run.get("budget"), dict) else {}
            candidate_limit = max(
                1, min(100, int(budget.get("candidate_limit") or Config.agent_candidate_limit)),
            )
            context_max_items = max(
                1, min(30, int(budget.get("context_max_items") or Config.agent_context_max_items)),
            )
            configured_context_budget = max(
                256, int(budget.get("context_budget_tokens") or Config.agent_context_budget_tokens),
            )
            max_output_tokens = max(
                128, int(budget.get("max_output_tokens") or Config.agent_max_output_tokens),
            )

            sequence = 1
            self._stage(
                run_id, project_id, sequence, "authorizing",
                input_summary={"project_id": project_id, "strategy": strategy},
            )
            self._assert_active(run_id)
            project = self.db.get_project(project_id)
            if not project:
                raise LookupError("知识库不存在")
            bound = self.service._assert_project_strategy(project_id, strategy)
            if bound != strategy or strategy not in self.adapters:
                raise ValueError("知识库构建方式与问答适配器不一致")
            self._record_policy(
                run_metadata,
                access_policy_decision(
                    project_id=project_id,
                    strategy=strategy,
                    authorized=True,
                ),
            )
            self.db.update_agent_run(run_id, metadata=run_metadata)
            self._stage(
                run_id, project_id, sequence, "authorizing", status="completed",
                output_summary={"authorized": True},
            )

            sequence = 2
            self._stage(
                run_id, project_id, sequence, "resolving_context",
                input_summary={"thread_id": str(run.get("thread_id") or "")},
            )
            self._assert_active(run_id)
            active_tool = self._start_controlled_tool(
                run_id=run_id, project_id=project_id, strategy=strategy,
                tool_name="context.resolve.langgraph",
                request_summary={
                    "question_chars": len(original_query),
                    "checkpoint_scope": "thread",
                },
                run_metadata=run_metadata,
            )
            resolution = self.context.resolve(
                run=run, project=project,
                cancel_check=lambda: self._cancelled(run_id),
            )
            self._assert_active(run_id)
            try:
                preflight_scope = QueryScopeDecision.from_dict(
                    resolution.scope_preflight or {},
                )
            except (TypeError, ValueError):
                preflight_scope = preflight_query_scope(
                    resolution.standalone_question or original_query,
                )
            try:
                input_decision = PolicyDecision.from_dict(
                    resolution.input_policy or {},
                )
            except (TypeError, ValueError):
                input_decision = input_policy_decision(preflight_scope)
            try:
                dialog_decision = PolicyDecision.from_dict(
                    resolution.dialog_policy or {},
                )
            except (TypeError, ValueError):
                dialog_decision = (
                    skipped_policy_decision(
                        "dialog",
                        checkpoint="dialog.context_route",
                        route="agent_meta",
                        reason_code="agent_interaction_does_not_require_conversation_resolution",
                    )
                    if preflight_scope.route == "agent_meta" else
                    dialog_policy_decision(
                        dependency=resolution.dependency,
                        branch=resolution.branch,
                        context_used=resolution.context_used,
                        model_used=resolution.model_used,
                    )
                )
            self._record_policy(run_metadata, input_decision)
            self._record_policy(run_metadata, dialog_decision)
            run_metadata["context_resolution"] = resolution.to_dict()
            run_metadata["scope_preflight"] = preflight_scope.to_dict()
            self.db.update_agent_run(run_id, metadata=run_metadata)
            self.db.finish_agent_tool_call(
                active_tool,
                status="degraded" if resolution.model_error else "completed",
                response_summary={
                    "dependency": resolution.dependency,
                    "context_used": resolution.context_used,
                    "model_used": bool(
                        resolution.model_used or input_decision.model_used
                    ),
                    "context_model_used": resolution.model_used,
                    "input_scope_model_used": input_decision.model_used,
                    "history_message_count": resolution.history_message_count,
                    "history_truncated": resolution.history_truncated,
                    "standalone_question_changed": (
                        resolution.standalone_question != original_query
                    ),
                    "branch": resolution.branch,
                    "question_route": resolution.question_route,
                    "input_action": input_decision.action,
                    "dialog_action": dialog_decision.action,
                },
                error_message=resolution.model_error,
            )
            active_tool = ""
            self._stage(
                run_id, project_id, sequence, "resolving_context",
                status="degraded" if resolution.model_error else "completed",
                output_summary={
                    "dependency": resolution.dependency,
                    "context_used": resolution.context_used,
                    "model_used": bool(
                        resolution.model_used or input_decision.model_used
                    ),
                    "branch": resolution.branch,
                    "question_route": resolution.question_route,
                },
                error=resolution.model_error,
            )
            if resolution.needs_clarification:
                clarification = resolution.clarification_question or (
                    "我还不能唯一确定你所指的对象，请明确写出对象名称。"
                )
                raw_clarification = {
                    "answer": clarification,
                    "answerability": "insufficient",
                    "claims": [],
                    "limitations": [
                        "本轮问题包含无法唯一确定的指代，尚未执行知识库检索。",
                    ],
                    "follow_up_questions": [clarification],
                }
                validated = validate_and_render(
                    raw_clarification, [], sufficiency_state="insufficient",
                )
                clarification_sufficiency = {
                    "state": "ambiguous",
                    "score": 0.0,
                    "reasons": ["需要用户明确指代对象"],
                }
                self._record_policy(
                    run_metadata,
                    skipped_policy_decision(
                        "retrieval",
                        checkpoint="retrieval.evidence_gate",
                        route="clarify",
                        reason_code="dialog_clarification_required_before_retrieval",
                    ),
                )
                self._record_policy(
                    run_metadata,
                    skipped_policy_decision(
                        "execution",
                        checkpoint="execution.tool:none",
                        route="clarify",
                        reason_code="dialog_clarification_requires_no_internal_tool",
                    ),
                )
                output_decision = output_policy_decision(
                    route="clarify",
                    sufficiency_state="ambiguous",
                    answer_text=validated.answer_text,
                    claim_count=len(validated.answer_json.get("claims") or []),
                    citation_count=len(validated.citations),
                )
                self._record_policy(run_metadata, output_decision)
                if not output_decision.allows_continue:
                    raise RuntimeError("澄清响应没有通过输出管控")
                run_metadata["query_scope"] = preflight_scope.to_dict()
                self.db.update_agent_run(
                    run_id,
                    sufficiency=clarification_sufficiency,
                    metadata=run_metadata,
                )
                message_id = "amsg_" + uuid.uuid4().hex
                self.db.complete_agent_run(
                    run_id=run_id, message_id=message_id,
                    answer_text=validated.answer_text,
                    answer_json={
                        **validated.answer_json,
                        "clarification_required": True,
                    },
                    citations=[], effective_mode="clarification",
                    stream_events=[
                        {
                            "event_type": "answer_start",
                            "payload": {
                                "message_id": message_id,
                                "mode": "clarification",
                                "sufficiency": clarification_sufficiency,
                            },
                        },
                        {
                            "event_type": "answer_delta",
                            "payload": {"delta": validated.answer_text},
                        },
                        {"event_type": "sources", "payload": {"items": []}},
                        {
                            "event_type": "answer_metadata",
                            "payload": {
                                "limitations": validated.answer_json.get("limitations") or [],
                                "follow_up_questions": [clarification],
                                "clarification_required": True,
                            },
                        },
                    ],
                )
                return
            query = resolution.standalone_question

            if preflight_scope.route in {"agent_meta", "out_of_scope"}:
                agent_meta_route = preflight_scope.route == "agent_meta"
                sequence += 1
                self._stage(
                    run_id, project_id, sequence, "checking_scope",
                    input_summary={"phase": "preflight"},
                )
                active_tool = self._start_controlled_tool(
                    run_id=run_id, project_id=project_id, strategy=strategy,
                    tool_name="scope.route.policy",
                    request_summary={"phase": "preflight"},
                    run_metadata=run_metadata,
                )
                run_metadata["query_scope"] = preflight_scope.to_dict()
                self._record_policy(
                    run_metadata,
                    skipped_policy_decision(
                        "retrieval",
                        checkpoint="retrieval.evidence_gate",
                        route=preflight_scope.route,
                        reason_code=(
                            "agent_interaction_does_not_require_knowledge_retrieval"
                            if agent_meta_route else
                            "input_scope_rejected_before_knowledge_retrieval"
                        ),
                    ),
                )
                self.db.update_agent_run(run_id, metadata=run_metadata)
                self.db.finish_agent_tool_call(
                    active_tool, status="completed",
                    response_summary=preflight_scope.to_dict(),
                )
                active_tool = ""
                self._stage(
                    run_id, project_id, sequence, "checking_scope",
                    status="completed",
                    output_summary={
                        "route": preflight_scope.route,
                        "method": preflight_scope.method,
                        "model_used": False,
                    },
                )
                self._complete_scope_response(
                    run_id=run_id,
                    project_id=project_id,
                    strategy=strategy,
                    sequence=sequence,
                    payload=(
                        agent_capability_payload(project, original_query)
                        if agent_meta_route else
                        scope_boundary_payload(
                            project, original_query, route="out_of_scope",
                        )
                    ),
                    scope=preflight_scope,
                    effective_mode=(
                        "agent_capabilities" if agent_meta_route
                        else "out_of_scope"
                    ),
                    sufficiency=(
                        {
                            "state": "not_required",
                            "score": 1.0,
                            "reasons": ["智能体角色与能力问题无需知识库检索"],
                        }
                        if agent_meta_route else
                        {
                            "state": "insufficient",
                            "score": 0.0,
                            "reasons": ["问题明显不属于当前所选知识库"],
                        }
                    ),
                    run_metadata=run_metadata,
                )
                return

            plan_activation = (
                "active" if Config.agent_dynamic_retrieval_enabled else "observe"
            )
            active_tool = self._start_controlled_tool(
                run_id=run_id, project_id=project_id, strategy=strategy,
                tool_name="plan.retrieval.policy",
                request_summary={
                    "query_chars": len(query),
                    "strategy": strategy,
                    "candidate_limit": candidate_limit,
                    "activation": plan_activation,
                },
                run_metadata=run_metadata,
            )
            retrieval_plan = plan_retrieval(
                query,
                strategy=strategy,
                candidate_limit=candidate_limit,
                requested_mode=str(run.get("requested_mode") or "auto"),
                referenced_terms=resolution.referenced_terms,
                knowledge_scope=resolution.knowledge_scope,
                activation=plan_activation,
            )
            run_metadata["retrieval_plan"] = retrieval_plan.to_dict()
            self.db.update_agent_run(run_id, metadata=run_metadata)
            self.db.finish_agent_tool_call(
                active_tool,
                status="completed",
                response_summary=retrieval_plan.to_dict(),
            )
            active_tool = ""

            sequence = 3
            self._stage(run_id, project_id, sequence, "retrieving")
            active_tool = self._start_controlled_tool(
                run_id=run_id, project_id=project_id, strategy=strategy,
                tool_name=f"retrieve.{strategy}",
                request_summary={
                    "query_chars": len(query),
                    "requested_mode": str(run.get("requested_mode") or "auto"),
                    "candidate_limit": candidate_limit,
                    "plan_protocol": retrieval_plan.protocol,
                    "plan_activation": retrieval_plan.activation,
                },
                run_metadata=run_metadata,
            )
            adapter = self.adapters[strategy]
            if strategy == GRAPHRAG_NATIVE_STRATEGY:
                retrieval: RetrievalResult = adapter.retrieve(
                    run_id=run_id, project=project, query=query,
                    candidate_limit=candidate_limit,
                    requested_mode=str(run.get("requested_mode") or "auto"),
                    plan=retrieval_plan,
                )
            else:
                retrieval = adapter.retrieve(
                    run_id=run_id, project=project, query=query,
                    candidate_limit=candidate_limit,
                    plan=retrieval_plan,
                )
            self._assert_active(run_id)
            self.db.finish_agent_tool_call(
                active_tool, status="completed",
                response_summary={
                    "mode": retrieval.mode,
                    "candidate_count": len(retrieval.evidence),
                    "snapshot": retrieval.snapshot_ref.identifier,
                },
            )
            active_tool = ""
            self.db.update_agent_run(
                run_id, snapshot_ref=retrieval.snapshot_ref.to_dict(),
                metadata={
                    **run_metadata,
                    "retrieval_diagnostics": retrieval.diagnostics,
                },
            )
            self._stage(
                run_id, project_id, sequence, "retrieving", status="completed",
                output_summary={
                    "mode": retrieval.mode,
                    "candidates": len(retrieval.evidence),
                    "graph_used": retrieval.snapshot_ref.graph_available,
                },
            )

            sequence = 4
            self._stage(run_id, project_id, sequence, "ranking")
            ranking = self._rank_and_select(
                run_id=run_id,
                project_id=project_id,
                strategy=strategy,
                project=project,
                query=query,
                original_query=original_query,
                evidence=retrieval.evidence,
                retrieval_limit=candidate_limit,
                final_limit=candidate_limit,
                context_max_items=context_max_items,
                configured_context_budget=configured_context_budget,
                max_output_tokens=max_output_tokens,
                candidate_draft=retrieval.candidate_answer,
                plan=retrieval_plan,
                attempt=1,
                run_metadata=run_metadata,
            )
            ranked = ranking.ranked
            selected = ranking.selected
            rerank = ranking.rerank
            candidate_draft = retrieval.candidate_answer
            self._stage(
                run_id, project_id, sequence, "ranking", status="completed",
                output_summary={
                    "attempt": 1,
                    "candidates": len(ranked),
                    "selected": len(selected),
                    "plan_execution": ranking.plan_execution,
                    "context_plan_execution": (
                        ranking.context_plan_execution
                    ),
                },
            )

            sequence = 5
            self._stage(run_id, project_id, sequence, "assessing")
            sufficiency = assess_sufficiency(
                query, selected, reranker_used_model=rerank.used_model,
            )
            self.db.update_agent_run(run_id, sufficiency=sufficiency.to_dict())
            self._stage(
                run_id, project_id, sequence, "assessing", status="completed",
                output_summary={
                    "attempt": 1,
                    "state": sufficiency.state,
                    "score": round(sufficiency.score, 4),
                    "reasons": list(sufficiency.reasons),
                },
            )

            supplemental = plan_supplemental_retrieval(
                query,
                strategy=strategy,
                primary_plan=retrieval_plan,
                sufficiency=sufficiency,
                maximum_candidates=min(
                    Config.agent_supplemental_candidate_limit,
                    max(4, candidate_limit // 2),
                ),
            )
            if not Config.agent_supplemental_retrieval_enabled:
                supplemental = replace(
                    supplemental,
                    execute=False,
                    candidate_limit=0,
                    plan=None,
                    reason_codes=("supplemental_retrieval_disabled",),
                )
            supplemental_diagnostics: dict[str, Any] = supplemental.to_dict()
            supplemental_diagnostics.update({
                "attempted": False,
                "source": "none",
                "new_candidate_count": 0,
            })

            if supplemental.execute and supplemental.plan is not None:
                sequence += 1
                self._stage(
                    run_id, project_id, sequence, "supplementing",
                    input_summary={
                        "candidate_limit": supplemental.candidate_limit,
                        "reason_codes": list(supplemental.reason_codes),
                    },
                )
                primary_ids = {item.evidence_id for item in ranked}
                remaining_pool = [
                    item for item in ranking.pool
                    if item.evidence_id not in primary_ids
                ]
                supplemental_candidates, pool_execution = apply_retrieval_plan(
                    remaining_pool,
                    plan=supplemental.plan,
                    candidate_limit=supplemental.candidate_limit,
                )
                supplemental_source = "existing_recall_pool"
                supplemental_retrieval: RetrievalResult | None = None
                needs_second_adapter_call = bool(
                    strategy == GRAPHRAG_NATIVE_STRATEGY
                    or (
                        len(supplemental_candidates) < min(
                            4, supplemental.candidate_limit,
                        )
                        and supplemental.query != query
                    )
                )
                if needs_second_adapter_call:
                    active_tool = self._start_controlled_tool(
                        run_id=run_id, project_id=project_id,
                        strategy=strategy,
                        tool_name=f"retrieve.{strategy}.supplemental",
                        request_summary={
                            "attempt": 2,
                            "query_chars": len(supplemental.query),
                            "candidate_limit": supplemental.candidate_limit,
                            "mode": supplemental.plan.native_mode,
                        },
                        run_metadata=run_metadata,
                    )
                    if strategy == GRAPHRAG_NATIVE_STRATEGY:
                        supplemental_retrieval = adapter.retrieve(
                            run_id=run_id,
                            project=project,
                            query=supplemental.query,
                            candidate_limit=supplemental.candidate_limit,
                            requested_mode=supplemental.plan.native_mode,
                            plan=supplemental.plan,
                        )
                    else:
                        supplemental_retrieval = adapter.retrieve(
                            run_id=run_id,
                            project=project,
                            query=supplemental.query,
                            candidate_limit=supplemental.candidate_limit,
                            plan=supplemental.plan,
                        )
                    if (
                        supplemental_retrieval.snapshot_ref.to_dict()
                        != retrieval.snapshot_ref.to_dict()
                    ):
                        raise RuntimeError(
                            "补充检索期间知识库快照发生变化，已停止本轮回答"
                        )
                    supplemental_pool = reciprocal_rank_fuse(
                        [
                            item for item in deduplicate_evidence(
                                supplemental_retrieval.evidence
                            )
                            if item.provenance_valid
                        ]
                    )
                    from_second, second_execution = apply_retrieval_plan(
                        supplemental_pool,
                        plan=supplemental.plan,
                        candidate_limit=supplemental.candidate_limit,
                    )
                    supplemental_candidates = deduplicate_evidence([
                        *supplemental_candidates,
                        *[
                            item for item in from_second
                            if item.evidence_id not in primary_ids
                        ],
                    ])[:supplemental.candidate_limit]
                    supplemental_source = (
                        "alternate_native_mode"
                        if strategy == GRAPHRAG_NATIVE_STRATEGY
                        else "alternate_query"
                    )
                    if (
                        supplemental_retrieval.candidate_answer.strip()
                        and supplemental_candidates
                    ):
                        candidate_draft = supplemental_retrieval.candidate_answer
                    self.db.finish_agent_tool_call(
                        active_tool,
                        status="completed",
                        response_summary={
                            "attempt": 2,
                            "mode": supplemental_retrieval.mode,
                            "candidate_count": len(
                                supplemental_retrieval.evidence
                            ),
                            "new_candidate_count": len(
                                supplemental_candidates
                            ),
                            "plan_execution": second_execution,
                        },
                    )
                    active_tool = ""

                supplemental_diagnostics.update({
                    "attempted": True,
                    "source": supplemental_source,
                    "new_candidate_count": len(supplemental_candidates),
                    "pool_execution": pool_execution,
                })
                self._stage(
                    run_id, project_id, sequence, "supplementing",
                    status="completed",
                    output_summary={
                        "source": supplemental_source,
                        "new_candidates": len(supplemental_candidates),
                    },
                )

                if supplemental_candidates:
                    sequence += 1
                    self._stage(run_id, project_id, sequence, "reranking")
                    ranking = self._rank_and_select(
                        run_id=run_id,
                        project_id=project_id,
                        strategy=strategy,
                        project=project,
                        query=query,
                        original_query=original_query,
                        evidence=[*ranked, *supplemental_candidates],
                        retrieval_limit=min(
                            100, len(ranked) + len(supplemental_candidates),
                        ),
                        final_limit=candidate_limit,
                        context_max_items=context_max_items,
                        configured_context_budget=configured_context_budget,
                        max_output_tokens=max_output_tokens,
                        candidate_draft=candidate_draft,
                        plan=retrieval_plan,
                        attempt=2,
                        run_metadata=run_metadata,
                    )
                    ranked = ranking.ranked
                    selected = ranking.selected
                    rerank = ranking.rerank
                    self._stage(
                        run_id, project_id, sequence, "reranking",
                        status="completed",
                        output_summary={
                            "candidates": len(ranked),
                            "selected": len(selected),
                        },
                    )
                    sequence += 1
                    self._stage(run_id, project_id, sequence, "reassessing")
                    sufficiency = assess_sufficiency(
                        query,
                        selected,
                        reranker_used_model=rerank.used_model,
                    )
                    self.db.update_agent_run(
                        run_id, sufficiency=sufficiency.to_dict(),
                    )
                    self._stage(
                        run_id, project_id, sequence, "reassessing",
                        status="completed",
                        output_summary={
                            "attempt": 2,
                            "state": sufficiency.state,
                            "score": round(sufficiency.score, 4),
                            "reasons": list(sufficiency.reasons),
                        },
                    )

            sequence += 1
            self._stage(
                run_id, project_id, sequence, "checking_scope",
                input_summary={
                    "phase": "evidence_alignment",
                    "evidence_count": len(selected),
                },
            )
            active_tool = self._start_controlled_tool(
                run_id=run_id, project_id=project_id, strategy=strategy,
                tool_name="scope.evidence_alignment",
                request_summary={
                    "phase": "evidence_alignment",
                    "evidence_count": len(selected),
                    "sufficiency_state": sufficiency.state,
                },
                run_metadata=run_metadata,
            )
            scope_decision = assess_query_scope(
                ollama=self.ollama,
                project=project,
                query=query,
                evidence=selected or ranked[:8],
                sufficiency=sufficiency,
                knowledge_scope=resolution.knowledge_scope,
                cancel_check=lambda: self._cancelled(run_id),
            )
            self._assert_active(run_id)
            run_metadata["query_scope"] = scope_decision.to_dict()
            self.db.update_agent_run(run_id, metadata=run_metadata)
            self.db.finish_agent_tool_call(
                active_tool,
                status="degraded" if scope_decision.model_error else "completed",
                response_summary=scope_decision.to_dict(),
                error_message=scope_decision.model_error,
            )
            active_tool = ""
            self._stage(
                run_id, project_id, sequence, "checking_scope",
                status="degraded" if scope_decision.model_error else "completed",
                output_summary={
                    "route": scope_decision.route,
                    "confidence": round(scope_decision.confidence, 4),
                    "method": scope_decision.method,
                    "model_used": scope_decision.model_used,
                    "matched_evidence": len(scope_decision.matched_evidence_ids),
                },
                error=scope_decision.model_error,
            )

            scope_response_payload: dict[str, Any] | None = None
            scope_response_sufficiency: dict[str, Any] | None = None
            if not scope_decision.allows_knowledge_answer:
                for item in ranked:
                    item.selected = False
                selected = []
                if scope_decision.route == "agent_meta":
                    scope_response_payload = agent_capability_payload(
                        project, original_query,
                    )
                    scope_response_sufficiency = {
                        "state": "not_required",
                        "score": 1.0,
                        "reasons": ["智能体角色与能力问题无需知识库回答"],
                    }
                else:
                    scope_reason = (
                        "问题不属于当前知识库范围"
                        if scope_decision.route == "out_of_scope"
                        else "当前知识库没有直接支持该问题的内容"
                        if scope_decision.route == "unsupported" else
                        "当前问题范围无法可靠确认"
                        if scope_decision.route == "uncertain" else
                        "模型服务暂时不可用，无法完成可靠的范围与回答判断"
                    )
                    sufficiency = replace(
                        sufficiency,
                        state="insufficient",
                        reasons=tuple(dict.fromkeys((
                            *sufficiency.reasons, scope_reason,
                        ))),
                    )
                    scope_response_payload = scope_boundary_payload(
                        project, original_query, route=scope_decision.route,
                    )
                    scope_response_sufficiency = sufficiency.to_dict()

            retrieval_decision = retrieval_policy_decision(
                scope=scope_decision,
                sufficiency=sufficiency,
                selected_count=len(selected),
                provenance_valid_count=len(ranked),
            )
            self._record_policy(run_metadata, retrieval_decision)

            retrieval_summary = {
                "strategy": strategy,
                "mode": retrieval.mode,
                "retrieved": len(retrieval.evidence),
                "provenance_valid": len(ranked),
                "selected": len(selected),
                "reranker": rerank.to_dict(),
                "evidence_types": sorted({
                    item.evidence_type for item in selected
                }),
                "document_count": len({
                    ref.document_id
                    for item in selected
                    for ref in item.source_refs
                }),
                "context_budget_tokens": ranking.evidence_token_budget,
                "max_output_tokens": ranking.max_output_tokens,
                "candidate_draft_tokens": ranking.draft_tokens,
                "context_pool": ranking.context_pool_count,
                "context_relevance_floor": (
                    ranking.context_relevance_floor
                ),
                "retrieval_plan": retrieval_plan.to_dict(),
                "retrieval_plan_execution": ranking.plan_execution,
                "context_plan_execution": ranking.context_plan_execution,
                "supplemental_retrieval": supplemental_diagnostics,
                "query_scope": scope_decision.to_dict(),
                "policy_decision": retrieval_decision.to_dict(),
            }
            run_metadata["retrieval_diagnostics"] = retrieval.diagnostics
            run_metadata["supplemental_retrieval"] = supplemental_diagnostics
            self.db.update_agent_run(
                run_id,
                retrieval_summary=retrieval_summary,
                metadata=run_metadata,
                sufficiency=sufficiency.to_dict(),
            )
            self.db.persist_agent_evidence(
                run_id, [item.to_record(run_id=run_id) for item in ranked],
            )
            self.db.append_agent_event(run_id, "evidence", {
                "selected_count": len(selected),
                "candidate_count": len(ranked),
                "items": [{
                    "evidence_id": item.evidence_id,
                    "type": item.evidence_type,
                    "title": item.title,
                    "rank": item.rank,
                } for item in selected],
            })

            if scope_response_payload is not None:
                adapter.assert_current(project_id, retrieval.snapshot_ref)
                self._assert_active(run_id)
                self._complete_scope_response(
                    run_id=run_id,
                    project_id=project_id,
                    strategy=strategy,
                    sequence=sequence,
                    payload=scope_response_payload,
                    scope=scope_decision,
                    effective_mode=(
                        "agent_capabilities"
                        if scope_decision.route == "agent_meta"
                        else scope_decision.route
                    ),
                    sufficiency=(
                        scope_response_sufficiency
                        or {"state": "insufficient", "score": 0.0, "reasons": []}
                    ),
                    run_metadata=run_metadata,
                )
                return

            max_output_tokens = ranking.max_output_tokens
            evidence_token_budget = ranking.evidence_token_budget
            question_tokens = ranking.question_tokens
            prompt_overhead_tokens = ranking.prompt_overhead_tokens
            model_context_limit = ranking.model_context_limit

            sequence += 1
            self._stage(run_id, project_id, sequence, "answering")
            self._assert_active(run_id)
            model_name = str(project.get("llm_model") or Config.default_llm_model)
            self.db.update_agent_run(run_id, model_name=model_name)
            active_tool = self._start_controlled_tool(
                run_id=run_id, project_id=project_id, strategy=strategy,
                tool_name="answer.ollama_json" if sufficiency.state != "insufficient" else "answer.refusal_policy",
                request_summary={
                    "model": model_name,
                    "selected_evidence": len(selected),
                    "sufficiency_state": sufficiency.state,
                },
                run_metadata=run_metadata,
            )
            try:
                raw_answer = generate_answer(
                    ollama=self.ollama, project=project, query=query,
                    original_query=original_query,
                    selected=selected, sufficiency=sufficiency,
                    candidate_draft=candidate_draft,
                    cancel_check=lambda: self._cancelled(run_id),
                    context_budget_tokens=min(
                        model_context_limit,
                        evidence_token_budget + max_output_tokens
                        + question_tokens + prompt_overhead_tokens,
                    ),
                    max_output_tokens=max_output_tokens,
                )
            except (AgentCancelled, OllamaCancelled, JobCancellationRequested):
                raise
            except Exception as exc:
                # A transport/protocol/model failure is a service state, not
                # evidence that the question is unsupported.  Publish a
                # retryable, citation-free response so the UI and replay API
                # expose the distinction consistently.
                error_message = str(exc).strip()[:1000]
                self.db.finish_agent_tool_call(
                    active_tool,
                    status="degraded",
                    response_summary={
                        "reason_code": "answer_service_unavailable",
                        "component": "ollama",
                    },
                    error_message=error_message,
                )
                active_tool = ""
                degraded_scope = QueryScopeDecision(
                    route="service_degraded",
                    confidence=0.0,
                    method="answer_generation_service",
                    reason_code="answer_service_unavailable",
                    model_used=True,
                    model_error=error_message,
                    retryable=True,
                    degraded_components=("ollama",),
                )
                degraded_sufficiency = replace(
                    sufficiency,
                    state="insufficient",
                    reasons=tuple(dict.fromkeys((
                        *sufficiency.reasons,
                        "模型服务暂时不可用，未生成知识库事实回答",
                    ))),
                )
                run_metadata["answer_scope"] = scope_decision.to_dict()
                run_metadata["query_scope"] = degraded_scope.to_dict()
                run_metadata["service_degraded"] = degraded_scope.to_dict()
                self.db.update_agent_run(
                    run_id,
                    metadata=run_metadata,
                    sufficiency=degraded_sufficiency.to_dict(),
                )
                self._stage(
                    run_id, project_id, sequence, "answering",
                    status="degraded",
                    output_summary={
                        "route": "service_degraded",
                        "component": "ollama",
                    },
                    error=error_message,
                )
                adapter.assert_current(project_id, retrieval.snapshot_ref)
                self._assert_active(run_id)
                self._complete_scope_response(
                    run_id=run_id,
                    project_id=project_id,
                    strategy=strategy,
                    sequence=sequence,
                    payload=scope_boundary_payload(
                        project, original_query, route="service_degraded",
                    ),
                    scope=degraded_scope,
                    effective_mode="service_degraded",
                    sufficiency=degraded_sufficiency.to_dict(),
                    run_metadata=run_metadata,
                )
                return
            if (
                str(raw_answer.get("answerability") or "") == "insufficient"
                and sufficiency.state != "insufficient"
            ):
                sufficiency = replace(
                    sufficiency,
                    state="insufficient",
                    reasons=tuple(dict.fromkeys((
                        *sufficiency.reasons,
                        "答案生成校验确认现有证据不包含所请求事实",
                    ))),
                )
                self.db.update_agent_run(
                    run_id, sufficiency=sufficiency.to_dict(),
                )
                raw_answer = refusal_payload(
                    sufficiency, project=project, query=query,
                )
            self._assert_active(run_id)
            self.db.finish_agent_tool_call(
                active_tool, status="completed",
                response_summary={
                    "claim_count": len(raw_answer.get("claims") or []),
                    "refused": sufficiency.state == "insufficient",
                },
            )
            active_tool = ""
            self._stage(
                run_id, project_id, sequence, "answering", status="completed",
                output_summary={"claim_count": len(raw_answer.get("claims") or [])},
            )

            sequence += 1
            self._stage(run_id, project_id, sequence, "validating_citations")
            validated = validate_and_render(
                raw_answer, selected, sufficiency_state=sufficiency.state,
            )
            output_decision = output_policy_decision(
                route=(
                    "controlled_refusal"
                    if sufficiency.state == "insufficient" else "knowledge"
                ),
                sufficiency_state=sufficiency.state,
                answer_text=validated.answer_text,
                claim_count=len(validated.answer_json.get("claims") or []),
                citation_count=len(validated.citations),
            )
            self._record_policy(run_metadata, output_decision)
            if not output_decision.allows_continue:
                raise RuntimeError("回答没有通过引用完整性校验与输出管控")
            adapter.assert_current(project_id, retrieval.snapshot_ref)
            self._assert_active(run_id)
            self._stage(
                run_id, project_id, sequence, "validating_citations",
                status="completed",
                output_summary={
                    "claims": len(validated.answer_json.get("claims") or []),
                    "citations": len(validated.citations),
                    "policy_action": output_decision.action,
                },
            )
            self.db.update_agent_run(run_id, metadata=run_metadata)
            message_id = "amsg_" + uuid.uuid4().hex
            chunks = [
                validated.answer_text[index:index + 48]
                for index in range(0, len(validated.answer_text), 48)
            ] or [""]
            stream_events = [{
                "event_type": "answer_start",
                "payload": {
                    "message_id": message_id,
                    "mode": retrieval.mode,
                    "sufficiency": sufficiency.to_dict(),
                },
            }, *({
                "event_type": "answer_delta", "payload": {"delta": chunk},
            } for chunk in chunks), {
                "event_type": "sources",
                "payload": {"items": validated.source_cards},
            }, {
                "event_type": "answer_metadata",
                "payload": {
                    "limitations": validated.answer_json.get("limitations") or [],
                    "follow_up_questions": validated.answer_json.get("follow_up_questions") or [],
                },
            }]
            # Snapshot freshness is checked immediately before this single
            # transaction publishes answer, citations and replayable SSE data.
            adapter.assert_current(project_id, retrieval.snapshot_ref)
            self.db.complete_agent_run(
                run_id=run_id, message_id=message_id,
                answer_text=validated.answer_text,
                answer_json=validated.answer_json,
                citations=validated.citations,
                effective_mode=retrieval.mode,
                stream_events=stream_events,
            )
        except (AgentCancelled, OllamaCancelled, JobCancellationRequested) as exc:
            if active_tool:
                try:
                    self.db.finish_agent_tool_call(
                        active_tool, status="cancelled", error_message=str(exc),
                    )
                except Exception:
                    pass
            try:
                self.db.terminate_agent_run(run_id, status="cancelled", error_message=str(exc))
            except Exception:
                LOGGER.exception("Failed to persist agent cancellation: %s", run_id)
        except Exception as exc:
            LOGGER.exception("Knowledge-agent run failed: %s", run_id)
            cancelled = False
            try:
                cancelled = self._cancelled(run_id)
            except Exception:
                pass
            if active_tool:
                try:
                    self.db.finish_agent_tool_call(
                        active_tool, status="cancelled" if cancelled else "failed",
                        error_message=str(exc),
                    )
                except Exception:
                    pass
            try:
                if sequence:
                    run = self.db.get_agent_run(run_id)
                    if run:
                        self.db.record_agent_step(
                            run_id=run_id, project_id=int(run["project_id"]),
                            sequence=sequence, step_type=str(run.get("stage") or "failed"),
                            status="cancelled" if cancelled else "failed",
                            error_message=str(exc),
                        )
                self.db.terminate_agent_run(
                    run_id, status="cancelled" if cancelled else "failed",
                    error_message=("用户已终止本次问答" if cancelled else str(exc)),
                )
            except Exception:
                LOGGER.exception("Failed to persist agent failure: %s", run_id)
