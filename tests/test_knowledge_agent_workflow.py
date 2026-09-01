"""One-pass workflow tests: bounded execution, persistence and cancellation."""

from __future__ import annotations

import re

from knowledge_forge.agent.contracts import (
    EvidenceItem,
    RetrievalResult,
    SnapshotRef,
    SourceRef,
)
from knowledge_forge.agent.context import ContextResolution
from knowledge_forge.agent.policy import input_policy_decision
from knowledge_forge.agent.rerank import RerankDiagnostics
from knowledge_forge.agent.planning import plan_retrieval
from knowledge_forge.agent.scope import QueryScopeDecision
from knowledge_forge.agent.workflow import KnowledgeAgentWorkflow


class _WorkflowDatabase:
    def __init__(self) -> None:
        self.run = {
            "id": "agent-run-1",
            "thread_id": "agent-thread-1",
            "project_id": 51,
            "query": "系统的任务管理有什么作用？",
            "strategy_key": "knowledge_forge",
            "requested_mode": "auto",
            "status": "queued",
            "stage": "queued",
            "budget": {
                "candidate_limit": 20,
                "context_max_items": 8,
                "context_budget_tokens": 4096,
                "max_output_tokens": 900,
            },
        }
        self.steps: list[dict] = []
        self.events: list[tuple[str, dict]] = []
        self.tools: dict[str, dict] = {}
        self.evidence: list[dict] = []
        self.completed: dict | None = None
        self.terminated: tuple[str, str] | None = None
        self.cancel_requested = False

    def start_agent_run(self, run_id):
        assert run_id == self.run["id"]
        if self.cancel_requested:
            return {**self.run, "status": "cancelled"}
        self.run.update(status="running", stage="authorizing")
        return dict(self.run)

    def update_agent_run(self, run_id, **values):
        assert run_id == self.run["id"]
        self.run.update({key: value for key, value in values.items() if value is not None})
        return dict(self.run)

    def record_agent_step(self, **values):
        self.steps.append(values)
        return values

    def append_agent_event(self, run_id, event_type, payload):
        assert run_id == self.run["id"]
        self.events.append((event_type, payload or {}))
        return {"event_type": event_type, "payload": payload or {}}

    def agent_run_cancel_requested(self, run_id):
        assert run_id == self.run["id"]
        return self.cancel_requested

    def get_project(self, project_id):
        assert project_id == 51
        return {
            "id": 51,
            "llm_model": "qwen3.6:latest",
            "knowledge_language": "zh-CN",
            "model_max_tokens": 8192,
        }

    def start_agent_tool_call(self, **values):
        self.tools[values["call_id"]] = {**values, "status": "running"}
        return self.tools[values["call_id"]]

    def finish_agent_tool_call(self, call_id, **values):
        self.tools[call_id].update(values)
        return self.tools[call_id]

    def persist_agent_evidence(self, run_id, rows):
        assert run_id == self.run["id"]
        self.evidence.extend(rows)
        return len(rows)

    def complete_agent_run(self, **values):
        self.completed = values
        self.run.update(status="completed", stage="completed")
        return dict(self.run)

    def terminate_agent_run(self, run_id, *, status, error_message=""):
        assert run_id == self.run["id"]
        self.terminated = (status, error_message)
        self.run.update(status=status, stage=status, error_message=error_message)
        return dict(self.run)

    def get_agent_run(self, run_id):
        assert run_id == self.run["id"]
        return dict(self.run)


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def json_chat(self, **values):
        self.calls += 1
        evidence_keys = re.findall(r'<evidence key="([^"]+)"', values["prompt"])
        assert evidence_keys == ["E1"]
        assert values["cancel_check"]() is False
        assert values["max_output_tokens"] == 900
        return {
            "answer": "模型草稿",
            "answerability": "supported",
            "claims": [{
                "text": "任务管理模块负责创建并调度任务。",
                "citations": evidence_keys,
            }],
            "limitations": [],
            "follow_up_questions": [],
        }


class _EmptyClaimsModel:
    def __init__(self, answerability: str) -> None:
        self.answerability = answerability
        self.calls = 0

    def json_chat(self, **_values):
        self.calls += 1
        return {
            "answer": "现有证据不足。",
            "answerability": self.answerability,
            "claims": [],
            "limitations": ["没有形成可引用结论"],
            "follow_up_questions": [],
        }


class _Service:
    def __init__(self) -> None:
        self.db = _WorkflowDatabase()
        self.ollama = _Model()

    def _assert_project_strategy(self, project_id, expected=None):
        assert project_id == 51
        assert expected in {None, "knowledge_forge"}
        return "knowledge_forge"


def _result(run_id: str, *, evidence: bool = True) -> RetrievalResult:
    snapshot = SnapshotRef(
        strategy="knowledge_forge", identifier="kf:5:ns:rr",
        source_hash="source-v5", generation=5, graph_available=True,
    )
    items = []
    if evidence:
        items.append(EvidenceItem.create(
            run_id=run_id, evidence_type="relation", project_id=51,
            strategy="knowledge_forge", snapshot_ref=snapshot,
            record_key="relation-8", title="任务管理 · 使用 · 调度器",
            content="任务管理模块负责创建任务，并使用调度器执行周期任务。",
            entity_keys=["task-management", "scheduler"],
            relation_keys=["relation-8"],
            source_refs=[SourceRef.from_dict({
                "document_id": 23,
                "filename": "system-guide.pdf",
                "revision_id": "revision-v5",
                "pages": [12],
                "page_label": "第 12 页",
                "section_title": "任务管理",
            })],
            retrieval_channels=["relation_graph"],
            raw_scores={"relation_graph": 0.96},
        ))
    return RetrievalResult(
        strategy="knowledge_forge", mode="hybrid",
        snapshot_ref=snapshot, evidence=items,
        diagnostics={"graph_used": True},
    )


class _Adapter:
    def __init__(self, db: _WorkflowDatabase, *, evidence: bool = True, cancel=False):
        self.db = db
        self.evidence = evidence
        self.cancel = cancel
        self.current_checks = 0
        self.queries: list[str] = []

    def retrieve(self, *, run_id, query, **_kwargs):
        self.queries.append(query)
        result = _result(run_id, evidence=self.evidence)
        if self.cancel:
            self.db.cancel_requested = True
        return result

    def assert_current(self, project_id, snapshot):
        assert project_id == 51 and snapshot.source_hash == "source-v5"
        self.current_checks += 1


class _RecoveringAdapter(_Adapter):
    """The alternate lexical surface finds evidence missed by pass one."""

    def retrieve(self, *, run_id, query, **_kwargs):
        self.queries.append(query)
        return _result(run_id, evidence=len(self.queries) > 1)


class _Reranker:
    def rerank(self, _query, items):
        for item in items:
            item.rerank_score = 0.97
        return RerankDiagnostics(True, "bge-reranker-v2-m3", len(items), 3)


class _ContextResolver:
    def __init__(
        self, *, standalone_question: str = "系统的任务管理有什么作用？",
        dependency: str = "independent", clarification: str = "",
        scope_route: str = "",
    ) -> None:
        self.standalone_question = standalone_question
        self.dependency = dependency
        self.clarification = clarification
        self.scope_route = scope_route

    def resolve(self, *, run, project, cancel_check):
        assert run["thread_id"] == "agent-thread-1"
        assert project["id"] == 51
        assert cancel_check() is False
        scope = None
        if self.scope_route:
            scope = QueryScopeDecision(
                route=self.scope_route,
                confidence=0.99,
                method="model_input_scope",
                reason_code=(
                    "model_detected_out_of_scope_before_retrieval"
                    if self.scope_route == "out_of_scope" else
                    "model_confirmed_possible_knowledge_scope"
                ),
                model_used=True,
            )
        return ContextResolution(
            original_question=str(run["query"]),
            standalone_question=self.standalone_question,
            dependency=self.dependency,
            resolved_references=(),
            referenced_terms=(),
            clarification_question=self.clarification,
            conversation_summary="",
            recent_messages=(),
            history_message_count=0,
            history_truncated=False,
            context_used=self.dependency == "resolved",
            model_used=False,
            model_error="",
            branch=(
                "clarify" if self.dependency == "ambiguous" else
                self.scope_route if self.scope_route else "ready"
            ),
            question_route=self.scope_route or "knowledge",
            scope_preflight=scope.to_dict() if scope else None,
            input_policy=(
                input_policy_decision(scope).to_dict() if scope else None
            ),
        )


def _workflow(
    *, evidence: bool = True, cancel: bool = False,
    context: _ContextResolver | None = None,
):
    service = _Service()
    workflow = KnowledgeAgentWorkflow(service)
    adapter = _Adapter(service.db, evidence=evidence, cancel=cancel)
    workflow.adapters = {"knowledge_forge": adapter}
    workflow.reranker = _Reranker()
    workflow.context = context or _ContextResolver()
    return service, workflow, adapter


def test_workflow_runs_one_retrieval_one_generation_and_atomic_citation_publish():
    service, workflow, adapter = _workflow()

    workflow.run("agent-run-1")

    assert service.db.terminated is None
    assert service.db.completed is not None
    assert service.ollama.calls == 1
    assert adapter.current_checks == 2
    assert len(service.db.evidence) == 1
    assert service.db.evidence[0]["selected"] is True
    assert len(service.db.completed["citations"]) == 1
    assert "〔1〕" in service.db.completed["answer_text"]
    assert [step["sequence"] for step in service.db.steps if step["status"] == "completed"] == [1, 2, 3, 4, 5, 6, 7, 8]
    event_types = [event_type for event_type, _payload in service.db.events]
    assert "evidence" in event_types
    assert service.db.run["metadata"]["retrieval_plan"]["activation"] == "active"
    plan_calls = [
        value for value in service.db.tools.values()
        if value["tool_name"] == "plan.retrieval.policy"
    ]
    assert len(plan_calls) == 1
    assert plan_calls[0]["status"] == "completed"
    assert service.db.run["retrieval_summary"]["supplemental_retrieval"]["execute"] is False
    policy = service.db.run["metadata"]["policy_control"]
    assert policy["protocol"] == "knowledge-agent-layered-policy-v1"
    assert {item["layer"] for item in policy["decisions"]} == {
        "access", "input", "dialog", "retrieval", "execution", "output",
    }
    assert next(
        item for item in policy["decisions"] if item["layer"] == "retrieval"
    )["action"] == "allow"
    assert next(
        item for item in policy["decisions"] if item["layer"] == "output"
    )["action"] == "allow"


def test_workflow_publishes_deterministic_refusal_without_model_call():
    service, workflow, adapter = _workflow(evidence=False)

    workflow.run("agent-run-1")

    assert service.db.completed is not None
    assert service.ollama.calls == 0
    assert adapter.current_checks == 1
    assert service.db.completed["citations"] == []
    assert "没有找到能直接支持" in service.db.completed["answer_text"]
    assert service.db.completed["answer_json"]["sufficiency_state"] == "insufficient"
    assert len(adapter.queries) == 2
    supplemental_calls = [
        value for value in service.db.tools.values()
        if value["tool_name"] == "retrieve.knowledge_forge.supplemental"
    ]
    assert len(supplemental_calls) == 1


def test_workflow_recovers_with_exactly_one_supplemental_retrieval():
    service, workflow, _adapter = _workflow(evidence=False)
    adapter = _RecoveringAdapter(service.db, evidence=False)
    workflow.adapters = {"knowledge_forge": adapter}

    workflow.run("agent-run-1")

    assert service.db.terminated is None
    assert service.db.completed is not None
    assert len(adapter.queries) == 2
    assert adapter.queries[0] != adapter.queries[1]
    assert service.ollama.calls == 1
    assert len(service.db.evidence) == 1
    supplemental = service.db.run["retrieval_summary"]["supplemental_retrieval"]
    assert supplemental["attempted"] is True
    assert supplemental["source"] == "alternate_query"
    assert supplemental["new_candidate_count"] == 1
    supplemental_calls = [
        value for value in service.db.tools.values()
        if value["tool_name"] == "retrieve.knowledge_forge.supplemental"
    ]
    assert len(supplemental_calls) == 1


def test_workflow_accepts_model_declared_insufficient_only_as_a_refusal():
    service, workflow, adapter = _workflow()
    model = _EmptyClaimsModel("insufficient")
    workflow.ollama = model

    workflow.run("agent-run-1")

    assert model.calls == 1
    assert service.db.completed is not None
    assert service.db.terminated is None
    assert service.db.completed["citations"] == []
    assert service.db.completed["answer_json"]["sufficiency_state"] == "insufficient"
    assert "没有找到足够" in service.db.completed["answer_text"]
    assert adapter.current_checks == 2


def test_workflow_fails_closed_when_model_claims_cannot_be_cited():
    service, workflow, adapter = _workflow()
    model = _EmptyClaimsModel("supported")
    workflow.ollama = model

    workflow.run("agent-run-1")

    assert model.calls == 1
    assert service.db.completed is None
    assert service.db.terminated is not None
    assert service.db.terminated[0] == "failed"
    assert "引用完整性校验" in service.db.terminated[1]
    assert adapter.current_checks == 0


def test_workflow_cancellation_stops_before_rerank_and_generation():
    service, workflow, adapter = _workflow(cancel=True)

    workflow.run("agent-run-1")

    assert service.db.completed is None
    assert service.db.terminated is not None
    assert service.db.terminated[0] == "cancelled"
    assert service.ollama.calls == 0
    assert adapter.current_checks == 0


def test_workflow_uses_standalone_context_question_for_retrieval_and_answer():
    context = _ContextResolver(
        standalone_question="任务管理模块使用什么组件调度周期任务？",
        dependency="resolved",
    )
    service, workflow, adapter = _workflow(context=context)

    workflow.run("agent-run-1")

    assert service.db.completed is not None
    assert adapter.queries == ["任务管理模块使用什么组件调度周期任务？"]
    assert "context_resolution" in service.db.run["metadata"]
    assert service.db.run["metadata"]["context_resolution"]["context_used"] is True


def test_dynamic_plan_is_preserved_in_the_final_answer_context():
    service, workflow, _adapter = _workflow()
    snapshot = SnapshotRef(
        strategy="knowledge_forge", identifier="kf:5:ns:rr",
        source_hash="source-v5", generation=5, graph_available=True,
    )
    source = SourceRef.from_dict({
        "document_id": 23,
        "filename": "system-guide.pdf",
        "revision_id": "revision-v5",
        "pages": [12],
    })
    evidence = []
    for index in range(12):
        evidence.append(EvidenceItem.create(
            run_id="agent-run-1", evidence_type="relation", project_id=51,
            strategy="knowledge_forge", snapshot_ref=snapshot,
            record_key=f"relation-{index}", title=f"关系 {index}",
            content=f"关系证据 {index}", source_refs=[source],
            retrieval_channels=["relation_graph", "relation_lexical"],
            raw_scores={"relation_graph": 1 - index / 100, "relation_lexical": 0.8},
        ))
        evidence.append(EvidenceItem.create(
            run_id="agent-run-1", evidence_type="chunk", project_id=51,
            strategy="knowledge_forge", snapshot_ref=snapshot,
            record_key=f"chunk-{index}", title=f"原文 {index}",
            content=f"图 2 的原文说明 {index}", source_refs=[source],
            retrieval_channels=["chunk_dense", "visual_anchor"],
            raw_scores={"chunk_dense": 0.8, "visual_anchor": 1 - index / 100},
        ))
    plan = plan_retrieval(
        "图 2 展示了什么？", strategy="knowledge_forge",
        candidate_limit=20, activation="active",
    )

    outcome = workflow._rank_and_select(
        run_id="agent-run-1", project_id=51,
        strategy="knowledge_forge", run_metadata={},
        project=service.db.get_project(51),
        query="图 2 展示了什么？", original_query="图 2 展示了什么？",
        evidence=evidence, retrieval_limit=20, final_limit=20,
        context_max_items=12, configured_context_budget=4096,
        max_output_tokens=900, candidate_draft="", plan=plan, attempt=1,
    )

    selected_types = [item.evidence_type for item in outcome.selected]
    assert len(selected_types) == 6
    assert selected_types.count("chunk") >= 2
    assert selected_types.count("relation") >= 1
    assert sum(
        outcome.context_plan_execution["reserved_by_family"].values()
    ) == 6


def test_workflow_asks_for_clarification_without_retrieval_or_answer_model():
    context = _ContextResolver(
        standalone_question="它有什么作用？",
        dependency="ambiguous",
        clarification="你指的是任务管理模块还是调度器？",
    )
    service, workflow, adapter = _workflow(context=context)

    workflow.run("agent-run-1")

    assert service.db.terminated is None
    assert service.db.completed is not None
    assert adapter.queries == []
    assert service.ollama.calls == 0
    assert service.db.completed["effective_mode"] == "clarification"
    assert "任务管理模块还是调度器" in service.db.completed["answer_text"]
    assert service.db.completed["citations"] == []
    policy = service.db.run["metadata"]["policy_control"]["decisions"]
    assert next(item for item in policy if item["layer"] == "dialog")["action"] == "clarify"
    assert next(item for item in policy if item["layer"] == "retrieval")["action"] == "skip"
    assert next(item for item in policy if item["layer"] == "output")["action"] == "allow"


def test_workflow_answers_agent_capability_question_without_retrieval():
    context = _ContextResolver(
        standalone_question="你能做什么",
        dependency="independent",
    )
    service, workflow, adapter = _workflow(context=context)
    service.db.run["query"] = "你能做什么"

    workflow.run("agent-run-1")

    assert service.db.terminated is None
    assert service.db.completed is not None
    assert adapter.queries == []
    assert adapter.current_checks == 0
    assert service.ollama.calls == 0
    assert service.db.completed["effective_mode"] == "agent_capabilities"
    assert service.db.completed["citations"] == []
    assert "知识库问答智能体" in service.db.completed["answer_text"]
    assert "不会回答当前知识库范围以外的问题" in service.db.completed["answer_text"]
    assert service.db.run["metadata"]["query_scope"]["route"] == "agent_meta"
    policy = service.db.run["metadata"]["policy_control"]["decisions"]
    assert next(item for item in policy if item["layer"] == "input")["action"] == "route"
    assert next(item for item in policy if item["layer"] == "dialog")["action"] == "skip"
    assert next(item for item in policy if item["layer"] == "retrieval")["action"] == "skip"


def test_workflow_skips_retrieval_for_confident_semantic_input_scope_rejection():
    context = _ContextResolver(
        standalone_question="今天是几月几日？",
        dependency="independent",
        scope_route="out_of_scope",
    )
    service, workflow, adapter = _workflow(context=context)
    service.db.run["query"] = "今天是几月几日？"

    workflow.run("agent-run-1")

    assert service.db.terminated is None
    assert service.db.completed is not None
    assert adapter.queries == []
    assert adapter.current_checks == 0
    assert service.ollama.calls == 0
    assert service.db.completed["effective_mode"] == "out_of_scope"
    assert service.db.completed["citations"] == []
    assert "不会调用通用知识作答" in service.db.completed["answer_text"]
    assert service.db.run["metadata"]["query_scope"]["route"] == "out_of_scope"
    tool_names = [item["tool_name"] for item in service.db.tools.values()]
    assert "retrieve.knowledge_forge" not in tool_names
    assert "rerank.cross_encoder_http" not in tool_names
    policy = service.db.run["metadata"]["policy_control"]["decisions"]
    assert next(item for item in policy if item["layer"] == "input")["action"] == "route"
    assert next(item for item in policy if item["layer"] == "retrieval")["action"] == "skip"


def test_workflow_refuses_semantically_unrelated_question_after_scope_check():
    class WeakReranker:
        def rerank(self, _query, items):
            for item in items:
                item.rerank_score = 0.01
            return RerankDiagnostics(True, "bge-reranker-v2-m3", len(items), 3)

    class ScopeOnlyModel:
        def __init__(self) -> None:
            self.calls = 0

        def json_chat(self, **values):
            self.calls += 1
            assert "只判断" in values["prompt"]
            return {
                "route": "out_of_scope",
                "confidence": 0.99,
                "matched_evidence_indices": [],
            }

    context = _ContextResolver(
        standalone_question="法国首都是什么？",
        dependency="independent",
    )
    service, workflow, adapter = _workflow(context=context)
    service.db.run["query"] = "法国首都是什么？"
    model = ScopeOnlyModel()
    service.ollama = model
    workflow.ollama = model
    workflow.reranker = WeakReranker()

    workflow.run("agent-run-1")

    assert service.db.terminated is None
    assert service.db.completed is not None
    assert model.calls == 1
    assert service.db.completed["effective_mode"] == "out_of_scope"
    assert service.db.completed["citations"] == []
    assert "不会调用通用知识作答" in service.db.completed["answer_text"]
    assert "巴黎" not in service.db.completed["answer_text"]
    assert all(not row["selected"] for row in service.db.evidence)
    assert service.db.run["metadata"]["query_scope"]["route"] == "out_of_scope"
