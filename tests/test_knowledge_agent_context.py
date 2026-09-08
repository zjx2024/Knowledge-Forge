"""LangGraph-backed multi-turn context resolution tests."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from knowledge_forge.agent.context import ConversationContextGraph
from knowledge_forge.config import Config
from knowledge_forge.ollama import ConservativeTokenEstimator


class _ContextDatabase:
    def __init__(self) -> None:
        self.histories: dict[str, list[dict]] = {}
        self.scopes: dict[str, dict] = {}

    def agent_conversation_context(self, run_id: str, *, limit: int = 24):
        rows = list(self.histories.get(run_id) or [])
        selected = rows[-limit:]
        return {
            "thread_id": "ath_context",
            "project_id": 7,
            "items": selected,
            "total": len(rows),
            "truncated": len(rows) > len(selected),
            "knowledge_scope": self.scopes.get(run_id) or {},
        }


class _ContextModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def json_chat(self, **values):
        self.calls.append(values)
        prompt = values["prompt"]
        if "哪一个更稳定" in prompt:
            return {
                "dependency": "ambiguous",
                "standalone_question": "哪一个更稳定？",
                "resolved_references": [],
                "referenced_terms": ["方法甲", "方法乙"],
                "clarification_question": "你想比较方法甲还是方法乙的稳定性？",
                "updated_summary": "讨论了方法甲与方法乙。",
                "reason": "存在两个候选对象",
            }
        return {
            "dependency": "resolved",
            "standalone_question": "FedAvg 在非独立同分布数据下有哪些局限？",
            "resolved_references": [{
                "expression": "它", "referent": "FedAvg",
            }],
            "referenced_terms": ["FedAvg", "非独立同分布数据"],
            "clarification_question": "",
            "updated_summary": "此前讨论了 FedAvg 的基本机制。",
            "reason": "上一轮唯一主题是 FedAvg",
        }


class _FailingContextModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def json_chat(self, **values):
        self.calls.append(values)
        raise RuntimeError("context model unavailable")


class _ContextService:
    def __init__(self) -> None:
        self.db = _ContextDatabase()
        self.ollama = _ContextModel()


def _run(run_id: str, question: str) -> dict:
    return {
        "id": run_id,
        "thread_id": "ath_context",
        "project_id": 7,
        "query": question,
    }


PROJECT = {
    "id": 7,
    "llm_model": "qwen3.6:latest",
    "knowledge_language": "zh",
    "model_max_tokens": 8192,
}


def _message(index: int, role: str, content: str) -> dict:
    return {
        "id": f"m{index:02d}",
        "run_id": f"r{(index + 1) // 2}",
        "role": role,
        "content": content,
        "created_at": f"2026-08-28T10:{index:02d}:00+00:00",
    }


def test_first_independent_turn_skips_context_model():
    service = _ContextService()
    saver = InMemorySaver()
    graph = ConversationContextGraph(
        service, checkpointer_factory=lambda: nullcontext(saver),
    )

    result = graph.resolve(
        run=_run("run-1", "FedAvg 的聚合过程是什么？"), project=PROJECT,
    )

    assert result.dependency == "independent"
    assert result.standalone_question == "FedAvg 的聚合过程是什么？"
    assert result.context_used is False
    assert result.model_used is False
    assert service.ollama.calls == []


def test_follow_up_is_rewritten_to_a_standalone_retrieval_question():
    service = _ContextService()
    service.db.histories["run-2"] = [
        _message(1, "user", "FedAvg 的聚合过程是什么？"),
        _message(2, "assistant", "FedAvg 对客户端更新进行加权平均。"),
    ]
    saver = InMemorySaver()
    graph = ConversationContextGraph(
        service, checkpointer_factory=lambda: nullcontext(saver),
    )

    result = graph.resolve(
        run=_run("run-2", "那它在非独立同分布数据下有什么局限？"),
        project=PROJECT,
    )

    assert result.dependency == "resolved"
    assert result.standalone_question == "FedAvg 在非独立同分布数据下有哪些局限？"
    assert result.context_used is True
    assert result.model_used is True
    assert result.resolved_references == ({"expression": "它", "referent": "FedAvg"},)
    assert len(service.ollama.calls) == 1


def test_ambiguous_follow_up_routes_to_clarification():
    service = _ContextService()
    service.db.histories["run-3"] = [
        _message(1, "user", "比较方法甲与方法乙。"),
        _message(2, "assistant", "两种方法的目标和代价不同。"),
    ]
    saver = InMemorySaver()
    graph = ConversationContextGraph(
        service, checkpointer_factory=lambda: nullcontext(saver),
    )

    result = graph.resolve(
        run=_run("run-3", "哪一个更稳定？"), project=PROJECT,
    )

    assert result.needs_clarification is True
    assert result.branch == "clarify"
    assert "方法甲还是方法乙" in result.clarification_question


def test_rolling_summary_survives_across_graph_invocations(monkeypatch):
    service = _ContextService()
    service.db.histories["run-4"] = [
        _message(1, "user", "介绍 FedAvg。"),
        _message(2, "assistant", "FedAvg 聚合客户端更新。"),
        _message(3, "user", "它有哪些假设？"),
        _message(4, "assistant", "假设参与方可以完成本地训练。"),
    ]
    service.db.histories["run-5"] = [
        *service.db.histories["run-4"],
        _message(5, "user", "它有什么局限？"),
        _message(6, "assistant", "数据异质性会影响收敛。"),
    ]
    monkeypatch.setattr(
        "knowledge_forge.agent.context.Config.agent_context_recent_token_ratio", 0.01,
    )
    monkeypatch.setattr(
        "knowledge_forge.agent.context.Config.agent_context_min_recent_turns", 1,
    )
    monkeypatch.setattr(
        "knowledge_forge.agent.context.Config.agent_context_recent_max_messages", 2,
    )
    saver = InMemorySaver()
    graph = ConversationContextGraph(
        service, checkpointer_factory=lambda: nullcontext(saver),
    )

    first = graph.resolve(
        run=_run("run-4", "那它在非独立同分布数据下有什么局限？"),
        project=PROJECT,
    )
    second = graph.resolve(
        run=_run("run-5", "它还有哪些限制？"), project=PROJECT,
    )

    assert first.conversation_summary == "此前讨论了 FedAvg 的基本机制。"
    assert second.conversation_summary == "此前讨论了 FedAvg 的基本机制。"
    assert "此前讨论了 FedAvg 的基本机制" in service.ollama.calls[-1]["prompt"]


def test_reference_without_any_history_fails_closed_to_clarification():
    service = _ContextService()
    saver = InMemorySaver()
    graph = ConversationContextGraph(
        service, checkpointer_factory=lambda: nullcontext(saver),
    )

    result = graph.resolve(run=_run("run-6", "它是什么？"), project=PROJECT)

    assert result.needs_clarification is True
    assert result.model_used is False
    assert service.ollama.calls == []


def test_agent_capability_question_is_routed_before_history_or_model_use():
    service = _ContextService()
    service.db.histories["run-agent-meta"] = [
        _message(1, "user", "介绍 FedAvg。"),
        _message(2, "assistant", "FedAvg 聚合客户端更新。"),
    ]
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    result = graph.resolve(
        run=_run("run-agent-meta", "你能做什么？"), project=PROJECT,
    )

    assert result.question_route == "agent_meta"
    assert result.branch == "agent_meta"
    assert result.context_used is False
    assert result.input_policy["action"] == "route"
    assert result.dialog_policy["action"] == "skip"
    assert service.ollama.calls == []


def test_semantic_input_scope_routes_clear_external_question_before_retrieval():
    service = _ContextService()
    service.db.scopes["run-current-date"] = {
        "document_count": 1,
        "documents": [{
            "document_id": 3,
            "filename": "联邦学习在医学领域中的应用.pdf",
        }],
    }

    def semantic_router(**values):
        service.ollama.calls.append(values)
        assert "检索前" in values["prompt"]
        return {
            "route": "out_of_scope",
            "request_kind": "external_or_general",
            "profile_alignment": "none",
            "confidence": 0.99,
        }

    service.ollama.json_chat = semantic_router
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )
    project = {
        **PROJECT,
        "name": "联邦学习资料",
        "description": "医学领域中的联邦学习研究资料",
    }

    result = graph.resolve(
        run=_run("run-current-date", "今天是几月几日？"), project=project,
    )

    assert result.question_route == "out_of_scope"
    assert result.branch == "out_of_scope"
    assert result.scope_preflight["route"] == "out_of_scope"
    assert result.scope_preflight["model_used"] is True
    assert result.input_policy["action"] == "route"
    assert len(service.ollama.calls) == 1


def test_first_turn_with_an_explicit_deictic_noun_still_reaches_retrieval():
    service = _ContextService()
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    result = graph.resolve(
        run=_run("run-explicit", "该系统的任务管理功能是什么？"),
        project=PROJECT,
    )

    assert result.dependency == "independent"
    assert result.standalone_question == "该系统的任务管理功能是什么？"
    assert result.needs_clarification is False


def test_first_turn_pronoun_uses_an_antecedent_from_the_same_question():
    service = _ContextService()

    def intra_turn_response(**_values):
        return {
            "dependency": "independent",
            "standalone_question": (
                "联邦学习的核心策略是什么，它的基本流程包括哪些步骤？"
            ),
            "resolved_references": [],
            "referenced_terms": ["联邦学习"],
            "clarification_question": "",
            "updated_summary": "",
            "reason": "指代对象已在同一句前半部分出现",
        }

    service.ollama.json_chat = intra_turn_response
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )
    question = "联邦学习的核心策略是什么，它的基本流程包括哪些步骤？"

    result = graph.resolve(
        run=_run("run-intra-turn", question), project=PROJECT,
    )

    assert result.dependency == "independent"
    assert result.standalone_question == question
    assert result.needs_clarification is False
    assert result.model_used is True


def test_single_document_scope_resolves_document_and_figure_references():
    service = _ContextService()
    service.db.histories["run-document-scope"] = [
        _message(1, "user", "介绍联邦学习。"),
        _message(2, "assistant", "联邦学习支持多方协同训练。"),
    ]
    service.db.scopes["run-document-scope"] = {
        "document_count": 1,
        "documents": [{
            "document_id": 19,
            "filename": "federated-learning.pdf",
        }],
        "document_list_truncated": False,
    }

    def scoped_response(**values):
        assert "federated-learning.pdf" in values["prompt"]
        assert '"document_count":1' in values["prompt"]
        return {
            "dependency": "resolved",
            "standalone_question": "federated-learning.pdf 中的图1展示了什么？",
            "resolved_references": [{
                "expression": "这篇文档中的图1",
                "referent": "federated-learning.pdf 中的图1",
            }],
            "referenced_terms": ["federated-learning.pdf", "图1"],
            "clarification_question": "",
            "updated_summary": "",
            "reason": "当前知识库只有一个活动文档",
        }

    service.ollama.json_chat = scoped_response
    result = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    ).resolve(
        run=_run("run-document-scope", "这篇文档中的图1展示了什么？"),
        project=PROJECT,
    )

    assert result.dependency == "resolved"
    assert result.needs_clarification is False
    assert result.knowledge_scope["document_count"] == 1


def test_common_local_model_schema_aliases_are_normalized():
    service = _ContextService()
    service.db.histories["run-7"] = [
        _message(1, "user", "FedAvg 是什么？"),
        _message(2, "assistant", "FedAvg 聚合客户端更新。"),
    ]

    def alias_response(**_values):
        return {
            "category": "resolved",
            "standalone_question": "FedAvg 有哪些局限？",
            "resolved_references": [{
                "original_expression": "它",
                "resolved_entity": "FedAvg",
            }],
            "updated_summary": "",
        }

    service.ollama.json_chat = alias_response
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    result = graph.resolve(
        run=_run("run-7", "它有哪些局限？"), project=PROJECT,
    )

    assert result.dependency == "resolved"
    assert result.standalone_question == "FedAvg 有哪些局限？"
    assert result.resolved_references == ({
        "expression": "它", "referent": "FedAvg",
    },)


def test_local_model_question_type_alias_is_normalized():
    service = _ContextService()
    service.db.histories["run-question-type"] = [
        _message(1, "user", "介绍 FedAvg。"),
        _message(2, "assistant", "FedAvg 聚合客户端更新。"),
    ]

    def alias_response(**_values):
        return {
            "question_type": "resolved",
            "standalone_question": "FedAvg 有哪些局限？",
            "resolved_references": [{"expression": "它", "referent": "FedAvg"}],
            "updated_summary": "",
        }

    service.ollama.json_chat = alias_response
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    result = graph.resolve(
        run=_run("run-question-type", "它有哪些局限？"), project=PROJECT,
    )

    assert result.dependency == "resolved"
    assert result.standalone_question == "FedAvg 有哪些局限？"
    assert result.needs_clarification is False


def test_singular_reference_to_multiple_candidates_requires_clarification():
    service = _ContextService()
    service.db.histories["run-8"] = [
        _message(1, "user", "比较 FedAvg 和 FedProx。"),
        _message(2, "assistant", "两种方法的机制不同。"),
    ]

    def unsafe_plural_response(**_values):
        return {
            "category": "resolved",
            "standalone_question": "FedAvg 和 FedProx 的稳定性如何？",
            "resolved_references": [{
                "original_expression": "它",
                "resolved_object": "FedAvg 和 FedProx",
            }],
            "updated_summary": "",
        }

    service.ollama.json_chat = unsafe_plural_response
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    result = graph.resolve(
        run=_run("run-8", "它的稳定性如何？"), project=PROJECT,
    )

    assert result.needs_clarification is True
    assert result.standalone_question == "它的稳定性如何？"


def test_context_model_failure_does_not_guess_from_domain_word_lists():
    service = _ContextService()
    service.ollama = _FailingContextModel()
    service.db.histories["run-model-failure"] = [
        _message(1, "user", "介绍已有的任务调度策略。"),
        _message(2, "assistant", "列出了若干调度策略。"),
    ]
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    for question in ("应该如何配置任务调度？", "其他方法有哪些？"):
        result = graph.resolve(
            run=_run("run-model-failure", question), project=PROJECT,
        )

        assert result.needs_clarification is True
        assert result.model_used is False
        assert "补写为不依赖上一轮指代的完整问题" in result.clarification_question
        assert "RuntimeError" in result.model_error


def test_no_history_pronoun_guard_does_not_confuse_common_words():
    service = _ContextService()
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    for question in ("应该如何配置任务调度？", "其他方法有哪些？", "其次要做什么准备？"):
        result = graph.resolve(
            run=_run(f"run-{question}", question), project=PROJECT,
        )

        assert result.dependency == "independent"
        assert result.needs_clarification is False
        assert result.model_used is False


def test_short_history_is_kept_by_tokens_instead_of_fixed_message_count():
    service = _ContextService()
    service.db.histories["run-short-history"] = [
        _message(index, "user" if index % 2 else "assistant", f"短消息 {index}")
        for index in range(1, 21)
    ]
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    loaded = graph._load_context({
        "run_id": "run-short-history",
        "original_question": "继续说明。",
        "model_context_tokens": 262144,
    })

    assert len(loaded["recent_messages"]) == 20
    assert loaded["messages_to_compact"] == []
    assert loaded["token_budget_triggered"] is False
    assert loaded["context_limit_tokens"] == 8192
    assert loaded["context_ceiling_tokens"] == 32768
    assert loaded["context_budget_reason"] == "minimum"
    assert loaded["estimated_prompt_tokens"] <= loaded["context_input_limit_tokens"]


def test_longer_history_expands_context_before_compacting():
    service = _ContextService()
    service.db.histories["run-expanded-history"] = [
        _message(
            index,
            "user" if index % 2 else "assistant",
            (f"第 {index} 条关于联邦学习训练过程的说明。" * 30),
        )
        for index in range(1, 25)
    ]
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    loaded = graph._load_context({
        "run_id": "run-expanded-history",
        "original_question": "它与上一种机制有什么区别？",
        "model_context_tokens": 262144,
    })

    assert 8192 < loaded["context_limit_tokens"] <= 32768
    assert loaded["context_limit_tokens"] % 4096 == 0
    assert loaded["context_ceiling_tokens"] == 32768
    assert loaded["context_budget_reason"] == "expanded"
    assert len(loaded["recent_messages"]) == 24
    assert loaded["messages_to_compact"] == []
    assert loaded["token_budget_triggered"] is False
    assert loaded["estimated_prompt_tokens"] <= loaded["context_input_limit_tokens"]


def test_very_long_history_stops_at_dynamic_ceiling_then_compacts():
    service = _ContextService()
    service.db.histories["run-capped-history"] = [
        _message(
            index,
            "user" if index % 2 else "assistant",
            (f"第 {index} 条关于联邦学习训练过程的详细说明。" * 100),
        )
        for index in range(1, 41)
    ]
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    loaded = graph._load_context({
        "run_id": "run-capped-history",
        "original_question": "它与上一种机制有什么区别？",
        "model_context_tokens": 262144,
    })

    assert loaded["context_limit_tokens"] == 32768
    assert loaded["context_required_tokens"] > loaded["context_ceiling_tokens"]
    assert loaded["context_budget_reason"] == "capped"
    assert loaded["token_budget_triggered"] is True
    assert (
        loaded["messages_to_compact"]
        or loaded["pending_compaction_message_count"] > 0
        or loaded["message_content_truncated"]
    )
    assert loaded["estimated_prompt_tokens"] <= loaded["context_input_limit_tokens"]


def test_runtime_context_is_intersected_with_project_limit_and_audited():
    service = _ContextService()
    service.db.histories["run-runtime-context"] = [
        _message(1, "user", "介绍 FedAvg。"),
        _message(2, "assistant", "FedAvg 聚合客户端更新。"),
    ]
    service.ollama.effective_model_context = lambda _model: SimpleNamespace(
        effective_context_tokens=65536,
        source="runner",
    )
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )
    project = {**PROJECT, "model_max_tokens": 262144}

    result = graph.resolve(
        run=_run("run-runtime-context", "它有哪些局限？"), project=project,
    )

    assert result.configured_model_context_tokens == 262144
    assert result.model_context_tokens == 65536
    assert result.model_context_source == "runner"
    assert result.context_ceiling_tokens == 32768
    assert result.context_limit_tokens == 8192
    assert result.context_budget_reason == "minimum"


def test_long_history_compacts_complete_turns_inside_model_token_budget():
    service = _ContextService()
    service.db.histories["run-long-history"] = [
        _message(
            index,
            "user" if index % 2 else "assistant",
            (f"第 {index} 条关于联邦学习训练过程的说明。" * 30),
        )
        for index in range(1, 25)
    ]
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    loaded = graph._load_context({
        "run_id": "run-long-history",
        "original_question": "它与上一种机制有什么区别？",
        "model_context_tokens": 8192,
    })

    recent_ids = [item["id"] for item in loaded["recent_messages"]]
    compact_ids = [item["id"] for item in loaded["messages_to_compact"]]
    assert recent_ids[-2:] == ["m23", "m24"]
    assert len(recent_ids) % 2 == 0
    assert compact_ids[:2] == ["m01", "m02"]
    assert len(compact_ids) % 2 == 0
    assert loaded["token_budget_triggered"] is True
    assert loaded["recent_message_tokens"] > 0
    assert loaded["compaction_message_tokens"] > 0
    assert loaded["estimated_prompt_tokens"] <= loaded["context_input_limit_tokens"]
    assert loaded["context_retry_reserve_tokens"] == (
        2 * Config.agent_context_resolution_max_output_tokens
        + Config.agent_context_resolution_repair_overhead_tokens
    )
    assert (
        loaded["context_input_limit_tokens"]
        + Config.agent_context_resolution_max_output_tokens
        + Config.ollama_default_safety_margin
        + loaded["context_retry_reserve_tokens"]
        == loaded["context_limit_tokens"]
    )


def test_generated_rolling_summary_is_hard_limited_by_tokens(monkeypatch):
    service = _ContextService()
    service.db.histories["run-summary-limit"] = [
        _message(index, "user" if index % 2 else "assistant", f"消息 {index}")
        for index in range(1, 7)
    ]

    def long_summary_response(**values):
        service.ollama.calls.append(values)
        return {
            "dependency": "resolved",
            "standalone_question": "FedAvg 有哪些局限？",
            "resolved_references": [{"expression": "它", "referent": "FedAvg"}],
            "referenced_terms": ["FedAvg"],
            "clarification_question": "",
            "updated_summary": "联邦学习上下文" * 2000,
            "reason": "唯一指代",
        }

    service.ollama.json_chat = long_summary_response
    monkeypatch.setattr(
        "knowledge_forge.agent.context.Config.agent_context_recent_token_ratio", 0.01,
    )
    monkeypatch.setattr(
        "knowledge_forge.agent.context.Config.agent_context_min_recent_turns", 1,
    )
    graph = ConversationContextGraph(
        service,
        checkpointer_factory=lambda: nullcontext(InMemorySaver()),
    )

    result = graph.resolve(
        run=_run("run-summary-limit", "它有哪些局限？"), project=PROJECT,
    )

    estimator = ConservativeTokenEstimator()
    assert estimator.estimate(result.conversation_summary) <= Config.agent_context_summary_max_tokens
    assert result.summary_tokens <= Config.agent_context_summary_max_tokens
    assert result.estimated_prompt_tokens <= result.context_input_limit_tokens
