"""Public, safe execution-trace projection for knowledge-agent runs."""

from __future__ import annotations

import json

from knowledge_forge.agent.trace import (
    EXECUTION_TRACE_PROTOCOL,
    build_public_execution_trace,
)


def _completed_payload() -> dict:
    return {
        "id": "arun_" + "1" * 32,
        "status": "completed",
        "stage": "completed",
        "strategy_key": "knowledge_forge",
        "query": "它为什么能缓解数据异质性？",
        "created_at": "2026-08-31T10:00:00+00:00",
        "started_at": "2026-08-31T10:00:01+00:00",
        "completed_at": "2026-08-31T10:00:09+00:00",
        "metadata": {
            "context_resolution": {
                "original_question": "它为什么能缓解数据异质性？",
                "standalone_question": "双层聚合机制为什么能缓解数据异质性？",
                "resolved_references": [{
                    "expression": "它", "referent": "双层聚合机制",
                }],
                "history_message_count": 4,
                "branch": "ready",
            },
            "retrieval_plan": {
                "intent": "relation_explanation",
                "family_limits": {
                    "text": 12, "entity": 8, "relation": 16,
                    "community": 4,
                },
                "native_mode": "local",
            },
            "policy_control": {
                "protocol": "knowledge-agent-layered-policy-v1",
                "decisions": [{
                    "layer": "access", "action": "allow",
                    "route": "selected_knowledge_base",
                    "reason_code": "project_strategy_and_session_authorized",
                }, {
                    "layer": "input", "action": "allow",
                    "route": "knowledge",
                    "reason_code": "ordinary_knowledge_question",
                }, {
                    "layer": "dialog", "action": "allow",
                    "route": "knowledge",
                    "reason_code": "conversation_context_resolved",
                }, {
                    "layer": "retrieval", "action": "allow",
                    "route": "grounded_knowledge_answer",
                    "reason_code": "scope_and_evidence_gate_passed",
                }, {
                    "layer": "execution", "action": "allow",
                    "route": "read_only_internal_tool",
                    "reason_code": "tool_allowed_for_selected_strategy",
                }, {
                    "layer": "output", "action": "allow",
                    "route": "knowledge",
                    "reason_code": "grounded_claims_and_citations_validated",
                }],
            },
        },
        "retrieval_summary": {
            "selected": 8,
            "supplemental_retrieval": {
                "attempted": True, "new_candidate_count": 3,
            },
        },
        "sufficiency": {
            "state": "sufficient", "score": 0.84,
            "reasons": ["核心结论有多个来源支持"],
        },
        "evidence_summaries": [{
            "evidence_id": "ev_" + "8" * 32,
            "evidence_type": "chunk",
            "title": "双层聚合机制",
            "excerpt": "双层聚合分别处理模型间与客户端间的异质性。",
            "filename": "federated-learning.pdf",
            "page_label": "第 8 页",
            "section_title": "3.2 双层聚合",
            "rank": 1,
            "citation_index": 1,
        }, {
            "evidence_id": "ev_" + "9" * 32,
            "evidence_type": "relation",
            "title": "双层聚合机制 → 缓解 → 数据异质性",
            "excerpt": "关系由同页原文支持。",
            "filename": "federated-learning.pdf",
            "page_label": "第 8 页",
            "rank": 2,
            "citation_index": None,
        }],
        "steps": [{
            "sequence": 1,
            "step_type": "authorizing",
            "status": "completed",
            "input_summary": {"secret": "RAW_LEDGER_SECRET"},
            "output_summary": {"authorized": True},
            "error_message": "RAW_LEDGER_SECRET",
            "started_at": "2026-08-31T10:00:01+00:00",
            "completed_at": "2026-08-31T10:00:02+00:00",
        }, {
            "sequence": 4,
            "step_type": "ranking",
            "status": "completed",
            "output_summary": {"candidates": 30, "selected": 8},
            "started_at": "2026-08-31T10:00:04+00:00",
            "completed_at": "2026-08-31T10:00:05+00:00",
        }],
        "tool_calls": [{
            "id": "atc_" + "2" * 32,
            "tool_name": "plan.retrieval.policy",
            "status": "completed",
            "request_summary": {"prompt": "RAW_LEDGER_SECRET"},
            "response_summary": {
                "intent": "relation_explanation",
                "scope": "local",
                "family_limits": {
                    "text": 12, "entity": 8, "relation": 16,
                    "community": 4,
                },
                "native_mode": "local",
                "scratchpad": "RAW_LEDGER_SECRET",
            },
            "started_at": "2026-08-31T10:00:02+00:00",
            "completed_at": "2026-08-31T10:00:03+00:00",
        }],
    }


def test_public_trace_explains_decisions_steps_and_tools_without_raw_payloads():
    trace = build_public_execution_trace(_completed_payload())

    assert trace["protocol"] == EXECUTION_TRACE_PROTOCOL
    assert trace["status"] == "completed"
    assert trace["duration_ms"] == 8000
    assert trace["step_count"] == 2
    assert trace["tool_count"] == 1
    assert any(item["title"] == "问题理解" for item in trace["decisions"])
    assert any(item["title"] == "分层管控" for item in trace["decisions"])
    assert any(
        "检索证据：通过" in item["detail"]
        for item in trace["decisions"] if item["title"] == "分层管控"
    )
    assert any(
        "双层聚合机制" in item["detail"] for item in trace["decisions"]
    )
    assert any(
        item["kind"] == "tool"
        and item["technical_name"] == "plan.retrieval.policy"
        for item in trace["items"]
    )
    planning_item = next(
        item for item in trace["items"]
        if item["technical_name"] == "plan.retrieval.policy"
    )
    assert "LOCAL" not in planning_item["detail"]
    assert {item["label"] for item in planning_item["details"]} >= {
        "问题类型", "检索范围", "证据配额",
    }
    ranking_item = next(
        item for item in trace["items"]
        if item["technical_name"] == "ranking"
    )
    assert ranking_item["evidence_title"] == "重排后进入回答上下文的证据"
    assert len(ranking_item["evidence"]) == 2
    assert ranking_item["evidence"][0]["citation_index"] == 1
    assert ranking_item["evidence"][1]["cited"] is False
    assert trace["selected_evidence_count"] == 2
    assert trace["cited_evidence_count"] == 1
    serialized = json.dumps(trace, ensure_ascii=False)
    assert "RAW_LEDGER_SECRET" not in serialized
    assert "request_summary" not in serialized
    assert "response_summary" not in serialized
    assert "scratchpad" not in serialized
    assert "隐藏推理" in trace["disclosure"]


def test_public_trace_only_labels_native_local_for_graphrag_runs():
    payload = _completed_payload()
    payload["strategy_key"] = "graphrag_native"

    trace = build_public_execution_trace(payload)

    planning_item = next(
        item for item in trace["items"]
        if item["technical_name"] == "plan.retrieval.policy"
    )
    assert "查询方式 LOCAL" in planning_item["detail"]


def test_answer_steps_expose_expandable_safe_facts_without_hidden_reasoning():
    trace = build_public_execution_trace({
        "status": "completed",
        "stage": "completed",
        "strategy_key": "knowledge_forge",
        "model_name": "qwen3.6:latest",
        "started_at": "2026-08-31T10:00:00+00:00",
        "completed_at": "2026-08-31T10:00:03+00:00",
        "sufficiency": {"state": "sufficient", "score": 0.84},
        "evidence_summaries": [{
            "evidence_id": "ev_" + "5" * 32,
            "evidence_type": "chunk",
            "title": "联邦学习原理",
            "excerpt": "客户端在本地训练，只交换模型更新。",
            "filename": "manual.pdf",
            "page_label": "第 3 页",
            "rank": 1,
            "citation_index": 1,
        }],
        "steps": [{
            "sequence": 6,
            "step_type": "answering",
            "status": "completed",
            "output_summary": {"claim_count": 3},
            "started_at": "2026-08-31T10:00:00+00:00",
            "completed_at": "2026-08-31T10:00:02+00:00",
        }],
        "tool_calls": [{
            "id": "atc_" + "4" * 32,
            "tool_name": "answer.ollama_json",
            "status": "completed",
            "request_summary": {
                "model": "qwen3.6:latest",
                "selected_evidence": 6,
                "sufficiency_state": "sufficient",
                "prompt": "NEVER_PUBLIC_PROMPT",
            },
            "response_summary": {
                "claim_count": 3,
                "scratchpad": "NEVER_PUBLIC_REASONING",
            },
            "started_at": "2026-08-31T10:00:00+00:00",
            "completed_at": "2026-08-31T10:00:02+00:00",
        }],
    })

    answering = next(item for item in trace["items"] if item["kind"] == "step")
    generation = next(item for item in trace["items"] if item["kind"] == "tool")
    assert {item["label"] for item in answering["details"]} >= {
        "回答模型", "形成陈述", "生成约束",
    }
    assert {item["label"] for item in generation["details"]} >= {
        "模型", "读取依据", "充分性状态", "形成陈述", "是否拒答",
    }
    assert generation["evidence_title"] == "回答阶段读取的具体证据"
    assert generation["evidence"][0]["title"] == "联邦学习原理"
    assert generation["evidence"][0]["location"] == "manual.pdf · 第 3 页"
    serialized = json.dumps(trace, ensure_ascii=False)
    assert "NEVER_PUBLIC_PROMPT" not in serialized
    assert "NEVER_PUBLIC_REASONING" not in serialized


def test_public_trace_explains_scope_uncertainty_and_service_degradation():
    trace = build_public_execution_trace({
        "status": "completed",
        "stage": "completed",
        "strategy_key": "knowledge_forge",
        "query_scope": {},
        "metadata": {
            "query_scope": {
                "route": "service_degraded",
                "method": "service_degraded",
                "reason_code": "scope_service_unavailable",
                "confidence": 0.0,
                "model_used": True,
                "retryable": True,
            },
        },
        "retrieval_summary": {"selected": 0},
        "sufficiency": {"state": "insufficient", "score": 0.0},
        "steps": [{
            "sequence": 5,
            "step_type": "checking_scope",
            "status": "degraded",
            "output_summary": {"route": "service_degraded"},
            "started_at": "2026-08-31T10:00:00+00:00",
            "completed_at": "2026-08-31T10:00:01+00:00",
        }],
        "tool_calls": [{
            "id": "atc_" + "6" * 32,
            "tool_name": "scope.evidence_alignment",
            "status": "degraded",
            "request_summary": {},
            "response_summary": {
                "route": "service_degraded",
                "method": "service_degraded",
            },
            "started_at": "2026-08-31T10:00:00+00:00",
            "completed_at": "2026-08-31T10:00:01+00:00",
        }],
    })

    assert any(
        item["title"] == "回答范围"
        and "模型服务暂时不可用" in item["detail"]
        for item in trace["decisions"]
    )
    scope_tool = next(
        item for item in trace["items"]
        if item["technical_name"] == "scope.evidence_alignment"
    )
    assert "范围判断模型暂时不可用" in scope_tool["detail"]


def test_running_trace_has_current_stage_and_bounded_safe_fallback_text():
    trace = build_public_execution_trace({
        "status": "running",
        "stage": "retrieving",
        "created_at": "2026-08-31T10:00:00+00:00",
        "started_at": "2026-08-31T10:00:01+00:00",
        "steps": [{
            "sequence": 3, "step_type": "retrieving", "status": "running",
            "started_at": "2026-08-31T10:00:02+00:00",
        }],
        "tool_calls": [{
            "id": "atc_" + "3" * 32,
            "tool_name": "vendor.unrecognized.internal_tool",
            "status": "running",
            "request_summary": {"filesystem_path": "/secret/private/path"},
            "started_at": "2026-08-31T10:00:03+00:00",
        }],
    })

    assert trace["current_stage"] == "retrieving"
    assert trace["current_label"] == "检索知识"
    assert trace["items"][1]["title"] == "内部工具调用"
    assert trace["items"][1]["detail"] == "执行受控内部工具"
    assert "/secret/private/path" not in json.dumps(trace, ensure_ascii=False)
