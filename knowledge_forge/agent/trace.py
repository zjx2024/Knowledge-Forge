"""Safe, deterministic projection of a knowledge-agent execution ledger.

The workflow persists operational steps and tool calls for recovery and audit.
This module converts that internal ledger into a user-facing explanation of
what the system did and why it selected a retrieval path.  It deliberately
does not expose prompts, model scratchpads, hidden chain-of-thought, raw tool
payloads, filesystem paths, or unvalidated candidate answers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ..unicode_safety import sanitize_unicode_text


EXECUTION_TRACE_PROTOCOL = "knowledge-agent-execution-trace-v2"

_STAGE_TITLES = {
    "queued": "等待开始",
    "authorizing": "确认知识范围",
    "resolving_context": "理解并路由问题",
    "checking_scope": "确认问题属于知识库范围",
    "retrieving": "检索知识",
    "ranking": "融合并重排候选",
    "assessing": "判断证据是否充分",
    "supplementing": "补充检索",
    "reranking": "重排补充证据",
    "reassessing": "再次判断证据",
    "answering": "组织回答",
    "validating_citations": "核验回答引用",
    "completed": "回答完成",
    "failed": "执行失败",
    "cancelled": "已停止",
}

_TOOL_TITLES = {
    "context.resolve.langgraph": "LangGraph 问题路由与语境解析",
    "scope.route.policy": "识别智能体交互问题",
    "scope.evidence_alignment": "核验问题与知识库的关联",
    "plan.retrieval.policy": "动态检索规划",
    "retrieve.knowledge_forge": "Knowledge Forge 混合检索",
    "retrieve.knowledge_forge.supplemental": "Knowledge Forge 补充检索",
    "retrieve.graphrag_native": "GraphRAG 查询",
    "retrieve.graphrag_native.supplemental": "GraphRAG 补充查询",
    "rerank.cross_encoder_http": "语义重排",
    "answer.ollama_json": "基于证据生成回答",
    "answer.refusal_policy": "证据不足处理",
    "answer.scope_policy": "回答范围处理",
}

_INTENT_LABELS = {
    "fact_lookup": "事实查找",
    "entity_explanation": "实体说明",
    "relation_explanation": "关系或机制说明",
    "procedure": "流程与步骤",
    "comparison": "对象比较",
    "global_summary": "全局总结",
    "source_lookup": "来源定位",
    "visual_lookup": "图表定位",
    "ambiguous": "范围待确认",
}

_FAMILY_LABELS = {
    "text": "原文",
    "entity": "实体",
    "relation": "关系",
    "community": "社区报告",
}

_SUFFICIENCY_LABELS = {
    "sufficient": "证据充分",
    "retryable": "存在可补充的证据缺口",
    "insufficient": "证据不足",
    "ambiguous": "问题范围不明确",
    "conflict": "证据存在冲突",
    "not_required": "无需知识证据",
}

_QUERY_SCOPE_LABELS = {
    "knowledge": "属于当前知识库",
    "agent_meta": "智能体角色或能力问题",
    "out_of_scope": "不属于当前知识库",
    "unsupported": "当前知识库没有直接支持",
    "uncertain": "无法可靠判断问题范围",
    "service_degraded": "模型服务异常",
}

_POLICY_LAYER_LABELS = {
    "access": "知识范围",
    "input": "问题输入",
    "dialog": "多轮语境",
    "retrieval": "检索证据",
    "execution": "工具执行",
    "output": "回答输出",
}

_POLICY_ACTION_LABELS = {
    "allow": "通过",
    "route": "已路由",
    "clarify": "需要澄清",
    "qualify": "限定回答",
    "refuse": "不作答",
    "deny": "已阻止",
    "skip": "无需执行",
}

_POLICY_ROUTE_LABELS = {
    "selected_knowledge_base": "当前知识库",
    "knowledge": "知识检索",
    "agent_meta": "角色与能力说明",
    "clarify": "请求澄清",
    "grounded_knowledge_answer": "有依据回答",
    "qualified_knowledge_answer": "限定范围回答",
    "evidence_insufficient": "证据不足",
    "out_of_scope": "知识库范围外",
    "unsupported": "知识库未直接覆盖",
    "uncertain": "问题范围未确认",
    "service_degraded": "模型服务异常",
    "controlled_refusal": "证据不足说明",
    "read_only_internal_tool": "受控内部工具",
}

_POLICY_REASON_LABELS = {
    "project_strategy_and_session_authorized": "已限定到当前知识库及其构建方式",
    "agent_role_or_capability_question": "问题仅询问智能体角色、能力或使用方式",
    "ordinary_knowledge_question": "按普通知识问题继续处理",
    "model_confirmed_possible_knowledge_scope": "本地模型确认问题可能属于当前知识库",
    "model_detected_agent_meta_before_retrieval": "检索前判定为智能体角色或能力问题",
    "model_detected_out_of_scope_before_retrieval": "检索前判定问题不属于当前知识库",
    "short_topic_requires_evidence_check": "短主题先检索，再根据实际证据判断是否相关",
    "input_scope_not_confident_to_skip_retrieval": "范围判断把握不足，保留知识检索以避免漏答",
    "knowledge_scope_profile_unavailable": "知识库范围资料不足，保留知识检索",
    "input_scope_model_unavailable_or_invalid": "检索前范围判断异常，保留检索并由后续证据门禁复核",
    "agent_interaction_does_not_require_conversation_resolution": "角色与能力说明不需要读取会话语境",
    "question_is_context_independent": "本轮问题可以独立理解",
    "conversation_context_resolved": "已结合当前会话解析指代或省略",
    "ambiguous_conversation_reference": "指代对象无法唯一确定",
    "dialog_clarification_required_before_retrieval": "澄清前不执行知识检索",
    "dialog_clarification_requires_no_internal_tool": "澄清响应不需要调用知识工具",
    "agent_interaction_does_not_require_knowledge_retrieval": "角色与能力说明不需要知识检索",
    "input_scope_rejected_before_knowledge_retrieval": "问题已在检索前确认不属于当前知识库",
    "scope_and_evidence_gate_passed": "问题范围与证据充分性均已通过",
    "retrieved_evidence_is_insufficient": "检索证据不足以支持可靠回答",
    "strong_cross_encoder_alignment": "语义重排确认问题与证据直接相关",
    "strong_semantic_and_term_alignment": "语义相似度与问题要点覆盖均达到门槛",
    "model_confirmed_evidence_alignment": "本地模型确认问题与证据直接相关",
    "model_detected_agent_meta_question": "本地模型判定为智能体角色或能力问题",
    "model_detected_out_of_scope": "本地模型判定问题不属于当前知识库",
    "model_found_no_direct_support": "本地模型未找到能够直接支持问题的证据",
    "no_knowledge_evidence": "没有取得可用于回答的知识证据",
    "scope_model_unavailable_or_invalid": "范围核验异常，已按失败关闭处理",
    "scope_model_low_confidence": "范围模型置信度不足，需要澄清",
    "scope_model_uncertain": "范围模型无法可靠判断问题归属",
    "scope_service_unavailable": "范围判断模型暂时不可用",
    "answer_service_unavailable": "回答模型暂时不可用，请稍后重试",
    "evidence_retryable": "证据只覆盖部分问题要点",
    "evidence_ambiguous": "现有证据支持的解释范围有限",
    "evidence_conflict": "检索证据之间存在冲突",
    "tool_allowed_for_selected_strategy": "工具属于当前构建方式的只读允许范围",
    "tool_not_allowed_for_selected_strategy": "工具不属于当前构建方式的允许范围",
    "grounded_claims_and_citations_validated": "回答陈述和引用均通过校验",
    "evidence_refusal_is_clean": "证据不足说明没有夹带未经核验的事实",
    "non_fact_policy_response_is_clean": "策略说明没有夹带知识事实",
    "grounded_answer_failed_output_contract": "回答缺少有效陈述或引用",
    "non_fact_response_contains_unverified_claims": "策略说明包含未经核验的事实",
    "insufficient_answer_contains_unverified_claims": "证据不足时生成了未经核验的事实",
}

_SCOPE_LABELS = {
    "local": "局部事实范围",
    "multi_entity": "多对象比较范围",
    "global": "全局主题范围",
}

_STRATEGY_LABELS = {
    "knowledge_forge": "Knowledge Forge 混合检索",
    "graphrag_native": "GraphRAG Native",
}

_EVIDENCE_TYPE_LABELS = {
    "chunk": "原文",
    "entity": "实体",
    "relation": "关系",
    "community_report": "社区报告",
    "native_context": "GraphRAG 上下文",
}

_REASON_LABELS = {
    "short_topic_exact_lexical_support": "短主题在已召回证据中有直接词面依据",
    "short_topic_requires_evidence_check": "短主题先检索，再根据实际证据判断是否相关",
    "default_fact_lookup": "没有检测到更强的特殊问法，采用通用事实检索计划",
    "safe_family_floors": "每类证据都保留最低召回额度，避免单一路径漏召回",
    "balanced_fallback": "问题类型不明确，使用均衡检索配额",
    "empty_query": "问题内容为空",
    "explicit_native_mode": "保留用户显式选择的 GraphRAG 查询方式",
    "single_document_summary_uses_local": "问题只涉及单篇文档，使用 Local 避免不必要的全局汇总",
    "primary_plan_disallows_retry": "首次计划不允许再次检索",
    "low_aspect_coverage": "首次证据没有覆盖足够的问题要点",
    "weak_graph_connectivity": "首次关系证据连接不足",
    "global_scope_report_gap": "全局问题缺少足够的社区报告",
    "limited_independent_support": "独立来源支持不足",
    "supplemental_once": "本轮最多执行一次补充检索",
}

_COMMUNITY_NAVIGATION_REASONS = {
    "baseline_relation_recall_is_sufficient": "首次关系召回已经充分，无需扩展相邻社区",
    "no_reliable_seed_community": "没有可靠的种子社区，未执行社区扩展",
    "no_eligible_links": "没有符合来源与质量门槛的社区联系",
    "disabled": "社区导航已关闭",
}


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _clip(value: object, limit: int = 260) -> str:
    text = sanitize_unicode_text(str(value or "")).strip()
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"


def _integer(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _percent(value: object, *, digits: int = 1) -> str:
    number = max(0.0, min(1.0, _number(value))) * 100.0
    return f"{number:.{max(0, int(digits))}f}%"


def _yes_no(value: object) -> str:
    return "是" if bool(value) else "否"


def _detail(label: object, value: object) -> dict[str, str] | None:
    clean_label = _clip(label, 80)
    clean_value = _clip(value, 500)
    if not clean_label or not clean_value:
        return None
    return {"label": clean_label, "value": clean_value}


def _details(*values: dict[str, str] | None) -> list[dict[str, str]]:
    return [value for value in values if value]


def _family_phrase(value: object, *, include_zero: bool = False) -> str:
    counts = _mapping(value)
    parts: list[str] = []
    for key, label in _FAMILY_LABELS.items():
        count = _integer(counts.get(key))
        if count or include_zero:
            parts.append(f"{label} {count}")
    return "、".join(parts)


def _evidence_type_phrase(value: object) -> str:
    if not isinstance(value, (list, tuple, set)):
        return ""
    return "、".join(dict.fromkeys(
        _EVIDENCE_TYPE_LABELS.get(str(item), str(item))
        for item in value if str(item or "").strip()
    ))


def _reason_label(reason: object) -> str:
    value = _clip(reason, 120)
    if not value:
        return ""
    if value.startswith("cue:"):
        intent = value.split(":", 1)[1]
        return f"问题中检测到{_INTENT_LABELS.get(intent, intent)}问法"
    if value.startswith("sufficiency:"):
        state = value.split(":", 1)[1]
        return f"首次检查结果为{_SUFFICIENCY_LABELS.get(state, state)}"
    if value.startswith("alternate_native_mode:"):
        mode = value.split(":", 1)[1].upper()
        return f"补充检索改用 GraphRAG {mode}"
    return _REASON_LABELS.get(value, value.replace("_", " "))


def _reason_phrase(values: object) -> str:
    if not isinstance(values, (list, tuple, set)):
        return ""
    return "；".join(dict.fromkeys(
        label for label in (_reason_label(item) for item in values) if label
    ))


def _policy_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = _mapping(run.get("metadata"))
    control = _mapping(metadata.get("policy_control"))
    return [
        dict(item) for item in (control.get("decisions") or [])
        if isinstance(item, Mapping)
        and str(item.get("layer") or "") in _POLICY_LAYER_LABELS
        and str(item.get("action") or "") in _POLICY_ACTION_LABELS
    ]


def _policy_reason(value: object) -> str:
    reason = _clip(value, 140)
    if not reason:
        return ""
    return _POLICY_REASON_LABELS.get(reason, _reason_label(reason))


def _policy_summary(run: dict[str, Any]) -> str:
    rows = _policy_rows(run)
    if not rows:
        return ""
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(str(row.get("layer") or ""), []).append(row)
    parts: list[str] = []
    for layer in ("access", "input", "dialog", "retrieval", "execution", "output"):
        values = by_layer.get(layer) or []
        if not values:
            continue
        if layer == "execution":
            active = [
                item for item in values
                if str(item.get("action") or "") in {"allow", "deny"}
            ]
            denied = sum(str(item.get("action") or "") == "deny" for item in active)
            action = (
                "无需执行" if not active else
                "已阻止" if denied else f"已校验 {len(active)} 项工具"
            )
        else:
            action = _POLICY_ACTION_LABELS.get(
                str(values[-1].get("action") or ""), "已处理",
            )
        parts.append(f"{_POLICY_LAYER_LABELS[layer]}：{action}")
    return "；".join(parts)


def _policy_details(run: dict[str, Any]) -> list[dict[str, str]]:
    rows = _policy_rows(run)
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(str(row.get("layer") or ""), []).append(row)
    details: list[dict[str, str]] = []
    for layer in ("access", "input", "dialog", "retrieval", "execution", "output"):
        values = by_layer.get(layer) or []
        if not values:
            continue
        if layer == "execution":
            active = [
                item for item in values
                if str(item.get("action") or "") in {"allow", "deny"}
            ]
            denied = sum(str(item.get("action") or "") == "deny" for item in active)
            value = (
                "本分支不需要调用知识工具"
                if not active else
                f"校验 {len(active)} 次内部工具调用"
                + (f"，阻止 {denied} 次" if denied else "，均在允许范围内")
            )
        else:
            row = values[-1]
            action = _POLICY_ACTION_LABELS.get(
                str(row.get("action") or ""), "已处理",
            )
            route = _POLICY_ROUTE_LABELS.get(
                str(row.get("route") or ""), _clip(row.get("route"), 80),
            )
            reason = _policy_reason(row.get("reason_code"))
            value = action
            if route:
                value += f"，去向：{route}"
            if reason:
                value += f"；{reason}"
        detail = _detail(_POLICY_LAYER_LABELS[layer], value)
        if detail:
            details.append(detail)
    return details


def _strategy_label(value: object) -> str:
    key = str(value or "")
    return _STRATEGY_LABELS.get(key, key or "当前知识库检索")


def _datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str) and value.strip():
        try:
            result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if result.tzinfo is None:
        return result.replace(tzinfo=timezone.utc)
    return result


def _iso(value: object) -> str:
    parsed = _datetime(value)
    return parsed.isoformat() if parsed else ""


def _duration_ms(started: object, completed: object, *, running: bool = False) -> int:
    start = _datetime(started)
    if not start:
        return 0
    end = _datetime(completed)
    if end is None and running:
        end = datetime.now(timezone.utc)
    if end is None:
        return 0
    return max(0, int((end - start).total_seconds() * 1000))


def _count_phrase(output: dict[str, Any]) -> str:
    candidates = _integer(
        output.get("candidates")
        or output.get("candidate_count")
        or output.get("retrieved")
    )
    selected = _integer(output.get("selected") or output.get("selected_count"))
    if candidates and selected:
        return f"处理 {candidates} 条候选，保留 {selected} 条回答依据"
    if candidates:
        return f"获得 {candidates} 条候选"
    if selected:
        return f"保留 {selected} 条回答依据"
    return ""


def _context_detail(run: dict[str, Any]) -> str:
    metadata = _mapping(run.get("metadata"))
    context = _mapping(metadata.get("context_resolution"))
    original = _clip(context.get("original_question") or run.get("query"), 180)
    standalone = _clip(context.get("standalone_question"), 240)
    references = []
    for item in context.get("resolved_references") or []:
        row = _mapping(item)
        expression = _clip(row.get("expression"), 60)
        referent = _clip(row.get("referent"), 100)
        if expression and referent:
            references.append(f"“{expression}”→“{referent}”")
    if str(context.get("branch") or "") == "clarify":
        return "检测到无法唯一确定的指代，本轮先请求用户澄清，不执行知识检索"
    parts: list[str] = []
    if references:
        parts.append("解析指代：" + "；".join(references[:4]))
    if standalone and standalone != original:
        parts.append("用于检索的问题：" + standalone)
    elif context:
        parts.append("本轮问题可直接作为独立检索问题")
    history_count = _integer(context.get("history_message_count"))
    if history_count:
        parts.append(f"参考同一会话最近 {history_count} 条消息理解语境")
    return "；".join(parts) or "检查本轮问题及允许使用的会话语境"


def _sufficiency_detail(output: dict[str, Any], run: dict[str, Any]) -> str:
    value = output or _mapping(run.get("sufficiency"))
    state = str(value.get("state") or "")
    score = _number(value.get("score"))
    detail = _SUFFICIENCY_LABELS.get(state, "已完成证据检查")
    if score:
        detail += f"（评分 {score:.2f}）"
    reasons = [_clip(item, 90) for item in value.get("reasons") or [] if _clip(item, 90)]
    if reasons:
        detail += "；" + "；".join(reasons[:3])
    return detail


def _stage_detail(stage: str, step: dict[str, Any], run: dict[str, Any]) -> str:
    output = _mapping(step.get("output_summary"))
    status = str(step.get("status") or "")
    if stage == "authorizing":
        return (
            "已确认当前会话只能读取所选知识库及其当前有效成果"
            if status == "completed" else "正在核对知识库、构建策略和访问范围"
        )
    if stage == "resolving_context":
        return _context_detail(run)
    if stage == "checking_scope":
        route = str(output.get("route") or "")
        label = _QUERY_SCOPE_LABELS.get(route, "正在确认回答范围")
        if route == "knowledge":
            return f"已确认问题与当前知识库中的证据直接相关（{label}）"
        return f"范围门禁判定为{label}；不会用无关证据生成答案"
    if stage == "retrieving":
        counts = _count_phrase(output)
        mode = _clip(output.get("mode"), 30)
        return "；".join(filter(None, (f"使用 {mode} 检索" if mode else "执行多路知识检索", counts)))
    if stage in {"ranking", "reranking"}:
        return _count_phrase(output) or "合并重复结果，并按相关性、来源完整性和证据类型重新排序"
    if stage in {"assessing", "reassessing"}:
        return _sufficiency_detail(output, run)
    if stage == "supplementing":
        count = _integer(output.get("new_candidates"))
        source = str(output.get("source") or "")
        source_label = {
            "existing_recall_pool": "首次召回的剩余候选",
            "alternate_query": "一次保守查询改写",
            "alternate_native_mode": "另一种 GraphRAG 查询方式",
        }.get(source, "证据缺口")
        return f"根据证据缺口从{source_label}补充 {count} 条候选"
    if stage == "answering":
        route = str(output.get("scope_route") or "")
        if route:
            return (
                "返回智能体角色与能力说明，不执行知识库事实回答"
                if route == "agent_meta" else
                "模型服务暂时不可用，未生成知识库事实回答"
                if route == "service_degraded" else
                "问题范围无法可靠确认，未生成知识库事实回答"
                if route == "uncertain" else
                "说明当前知识库的回答边界，不使用通用知识补答"
            )
        claims = _integer(output.get("claim_count"))
        return f"仅依据选中证据组织 {claims} 项可核验陈述" if claims else "仅依据选中证据组织回答"
    if stage == "validating_citations":
        claims = _integer(output.get("claims"))
        citations = _integer(output.get("citations"))
        return f"核验 {claims} 项陈述与 {citations} 个引用，移除无依据内容"
    return _STAGE_TITLES.get(stage, "执行受控问答步骤")


def _family_budget(value: object) -> str:
    return _family_phrase(value)


def _tool_detail(
    name: str, tool: dict[str, Any], run: dict[str, Any],
) -> str:
    request = _mapping(tool.get("request_summary"))
    response = _mapping(tool.get("response_summary"))
    if name == "context.resolve.langgraph":
        history = _integer(response.get("history_message_count"))
        changed = bool(response.get("standalone_question_changed"))
        route = _POLICY_ROUTE_LABELS.get(
            str(response.get("question_route") or ""), "知识检索",
        )
        if changed:
            return f"使用 {history} 条历史消息完成指代或省略补全；路由至{route}"
        return (
            f"检查会话状态；本轮无需改写；路由至{route}"
            if not history else
            f"检查 {history} 条历史消息；未改变原问题；路由至{route}"
        )
    if name in {"scope.route.policy", "scope.evidence_alignment"}:
        route = str(response.get("route") or "")
        label = _QUERY_SCOPE_LABELS.get(route, "范围未确认")
        if str(tool.get("status") or "") == "degraded":
            return "范围判断模型暂时不可用，已记录为服务异常"
        method = str(response.get("method") or "")
        method_label = {
            "deterministic_interaction_intent": "确定性交互意图规则",
            "strong_retrieval_signals": "强检索相关信号",
            "model_evidence_alignment": "结构化语义核验",
            "no_evidence": "无可用证据",
            "fail_closed": "失败关闭策略",
        }.get(method, "受控范围策略")
        return f"使用{method_label}判定：{label}"
    if name == "plan.retrieval.policy":
        intent = _INTENT_LABELS.get(str(response.get("intent") or ""), "通用查询")
        mode = str(response.get("native_mode") or "").upper()
        budget = _family_budget(response.get("family_limits"))
        parts = [f"识别为{intent}"]
        # ``native_mode`` is retained on the shared internal plan contract so
        # the GraphRAG adapter can consume it.  It is not a Knowledge Forge
        # query mode and must not be presented as though KF selected Local,
        # Basic or Global.
        if str(run.get("strategy_key") or "") == "graphrag_native" and mode:
            parts.append(f"查询方式 {mode}")
        if budget:
            parts.append("候选预算：" + budget)
        return "；".join(parts)
    if name.startswith("retrieve."):
        mode = _clip(response.get("mode") or request.get("mode"), 30)
        count = _integer(response.get("candidate_count"))
        parts = [f"执行 {mode} 查询" if mode else "执行知识检索"]
        if count:
            parts.append(f"返回 {count} 条带来源候选")
        return "；".join(parts)
    if name == "rerank.cross_encoder_http":
        count = _integer(response.get("candidate_count") or request.get("candidate_count"))
        model = _clip(response.get("model"), 70)
        latency = _integer(response.get("latency_ms"))
        if response.get("used_model"):
            parts = [f"使用 {model or '语义重排模型'}比较 {count} 条候选"]
            if latency:
                parts.append(f"模型耗时 {latency} ms")
            return "；".join(parts)
        return f"重排服务降级，使用可审计的融合排序处理 {count} 条候选"
    if name == "answer.ollama_json":
        if str(tool.get("status") or "") == "degraded":
            return "回答模型服务暂时不可用，未生成知识库事实回答"
        model = _clip(request.get("model"), 70)
        selected = _integer(request.get("selected_evidence"))
        claims = _integer(response.get("claim_count"))
        return f"使用 {model or '当前知识库模型'}读取 {selected} 条证据，生成 {claims} 项可核验陈述"
    if name == "answer.refusal_policy":
        return "证据不足，按确定性策略直接拒答，未调用回答模型补充事实"
    if name == "answer.scope_policy":
        route = str(response.get("scope_route") or request.get("scope_route") or "")
        return (
            "按统一模板说明智能体角色与能力，未执行知识检索"
            if route == "agent_meta" else
            "说明模型服务异常，未生成知识库事实回答"
            if route == "service_degraded" else
            "说明问题范围未确认，未生成知识库事实回答"
            if route == "uncertain" else
            "按统一范围策略说明不作答，未使用通用知识补充"
        )
    return "执行受控内部工具"


def _stage_facts(
    stage: str, step: dict[str, Any], run: dict[str, Any],
) -> list[dict[str, str]]:
    output = _mapping(step.get("output_summary"))
    metadata = _mapping(run.get("metadata"))
    context = _mapping(metadata.get("context_resolution"))
    retrieval = _mapping(run.get("retrieval_summary"))
    if stage == "authorizing":
        return _details(
            _detail("知识范围", "仅限当前选择的知识库"),
            _detail("构建策略", _strategy_label(run.get("strategy_key"))),
            _detail("访问校验", "已通过" if output.get("authorized") else "进行中"),
        )
    if stage == "resolving_context":
        input_scope = _mapping(metadata.get("scope_preflight"))
        return _details(
            _detail("问题依赖", {
                "independent": "可独立理解",
                "resolved": "已结合会话解析",
                "ambiguous": "仍存在歧义",
            }.get(str(context.get("dependency") or ""), "正在判断")),
            _detail("使用历史", (
                f"{_integer(context.get('history_message_count'))} 条消息"
                if context.get("context_used") else "未使用"
            )),
            _detail("语境模型", (
                "已调用一次结构化解析"
                if context.get("model_used") else "未调用"
            )),
            _detail("输入范围模型", (
                "已调用一次结构化分类"
                if input_scope.get("model_used") else "未调用"
            )),
            _detail("问题去向", _POLICY_ROUTE_LABELS.get(
                str(output.get("question_route") or context.get("question_route") or ""),
                "知识检索",
            )),
            _detail("检索问题", context.get("standalone_question")),
        )
    if stage == "checking_scope":
        return _details(
            _detail("范围结论", _QUERY_SCOPE_LABELS.get(
                str(output.get("route") or ""), "正在判断",
            )),
            _detail("判断方式", {
                "deterministic_interaction_intent": "确定性交互意图规则",
                "strong_retrieval_signals": "强检索相关信号直接通过",
                "model_evidence_alignment": "本地模型结构化语义核验",
                "no_evidence": "没有可用证据，按边界策略处理",
                "fail_closed": "核验异常，按失败关闭策略处理",
            }.get(str(output.get("method") or ""), "受控范围策略")),
            _detail("调用范围模型", _yes_no(output.get("model_used"))),
            _detail("匹配依据", f"{_integer(output.get('matched_evidence'))} 条"),
            _detail("置信度", _percent(output.get("confidence"))),
        )
    if stage == "retrieving":
        return _details(
            _detail("检索方式", output.get("mode") or retrieval.get("mode")),
            _detail("召回候选", f"{_integer(output.get('candidates') or retrieval.get('retrieved'))} 条"),
            _detail("图谱参与", _yes_no(output.get("graph_used"))),
        )
    if stage in {"ranking", "reranking"}:
        plan_execution = _mapping(output.get("plan_execution"))
        context_execution = _mapping(output.get("context_plan_execution"))
        return _details(
            _detail("排序轮次", f"第 {_integer(output.get('attempt')) or 1} 轮"),
            _detail("进入排序", f"{_integer(output.get('candidates'))} 条"),
            _detail("进入回答上下文", f"{_integer(output.get('selected'))} 条"),
            _detail("排序后候选组成", _family_phrase(plan_execution.get("selected_by_family"))),
            _detail("最终依据组成", _family_phrase(context_execution.get("selected_by_family"))),
        )
    if stage in {"assessing", "reassessing"}:
        return _details(
            _detail("检查结果", _SUFFICIENCY_LABELS.get(
                str(output.get("state") or ""), "正在检查",
            )),
            _detail("综合评分", _percent(output.get("score"))),
            _detail("触发原因", _reason_phrase(output.get("reasons")) or "未触发风险或拒答规则"),
        )
    if stage == "supplementing":
        return _details(
            _detail("补充来源", {
                "existing_recall_pool": "首次召回池中的剩余候选",
                "alternate_query": "保守改写后的查询",
                "alternate_native_mode": "另一种 GraphRAG 查询方式",
            }.get(str(output.get("source") or ""), "证据缺口")),
            _detail("新增候选", f"{_integer(output.get('new_candidates'))} 条"),
        )
    if stage == "answering":
        route = str(output.get("scope_route") or "")
        if route:
            return _details(
                _detail("处理方式", (
                    "说明智能体角色与能力"
                    if route == "agent_meta" else "说明知识库回答边界"
                )),
                _detail("知识事实生成", "未执行"),
                _detail("引用依据", "不需要"),
            )
        return _details(
            _detail("回答模型", run.get("model_name") or "当前知识库模型"),
            _detail("形成陈述", f"{_integer(output.get('claim_count'))} 项"),
            _detail("生成约束", "只读取已选证据，不把会话历史当作事实来源"),
        )
    if stage == "validating_citations":
        return _details(
            _detail("核验陈述", f"{_integer(output.get('claims'))} 项"),
            _detail("有效引用", f"{_integer(output.get('citations'))} 个"),
            _detail("发布规则", "未知引用、无来源陈述和非法声明不会发布"),
        )
    return _details(
        _detail("处理阶段", _STAGE_TITLES.get(stage, "受控问答步骤")),
        _detail("执行状态", {
            "queued": "等待开始", "running": "执行中",
            "completed": "已完成", "degraded": "已降级完成",
            "failed": "失败", "cancelled": "已停止",
        }.get(str(step.get("status") or ""), "处理中")),
    )


def _tool_facts(
    name: str, tool: dict[str, Any], run: dict[str, Any],
) -> list[dict[str, str]]:
    request = _mapping(tool.get("request_summary"))
    response = _mapping(tool.get("response_summary"))
    if name == "context.resolve.langgraph":
        return _details(
            _detail("输入规模", f"问题 {_integer(request.get('question_chars'))} 字符"),
            _detail("状态范围", "当前会话"),
            _detail("历史消息", f"{_integer(response.get('history_message_count'))} 条"),
            _detail("问题是否改写", _yes_no(response.get("standalone_question_changed"))),
            _detail("调用语境解析模型", _yes_no(
                response.get("context_model_used")
            )),
            _detail("调用输入范围模型", _yes_no(
                response.get("input_scope_model_used")
            )),
            _detail("输入层处理", _POLICY_ACTION_LABELS.get(
                str(response.get("input_action") or ""), "已处理",
            )),
            _detail("对话层处理", _POLICY_ACTION_LABELS.get(
                str(response.get("dialog_action") or ""), "已处理",
            )),
            _detail("问题去向", _POLICY_ROUTE_LABELS.get(
                str(response.get("question_route") or ""), "知识检索",
            )),
        )
    if name in {"scope.route.policy", "scope.evidence_alignment"}:
        return _details(
            _detail("范围结论", _QUERY_SCOPE_LABELS.get(
                str(response.get("route") or ""), "未确认",
            )),
            _detail("判断方式", {
                "deterministic_interaction_intent": "确定性交互意图规则",
                "strong_retrieval_signals": "Reranker / Embedding 强相关信号",
                "model_evidence_alignment": "本地模型结构化语义核验",
                "no_evidence": "未检索到证据",
                "fail_closed": "失败关闭",
            }.get(str(response.get("method") or ""), "受控范围策略")),
            _detail("调用模型", _yes_no(response.get("model_used"))),
            _detail("直接相关证据", f"{len(response.get('matched_evidence_ids') or [])} 条"),
            _detail("置信度", _percent(response.get("confidence"))),
        )
    if name == "plan.retrieval.policy":
        return _details(
            _detail("策略", _strategy_label(request.get("strategy"))),
            _detail("候选上限", f"{_integer(request.get('candidate_limit'))} 条"),
            _detail("问题类型", _INTENT_LABELS.get(
                str(response.get("intent") or ""), "通用查询",
            )),
            _detail("检索范围", _SCOPE_LABELS.get(
                str(response.get("scope") or ""), str(response.get("scope") or ""),
            )),
            _detail("路由置信度", _percent(response.get("confidence"))),
            _detail("证据配额", _family_phrase(response.get("family_limits"))),
            _detail("选择依据", _reason_phrase(response.get("reason_codes"))),
        )
    if name.startswith("retrieve."):
        native = str(run.get("strategy_key") or "") == "graphrag_native"
        return _details(
            _detail("请求方式", (
                request.get("requested_mode") or request.get("mode") or "自动"
                if native else "Knowledge Forge 动态混合检索"
            )),
            _detail("候选上限", f"{_integer(request.get('candidate_limit'))} 条"),
            _detail("实际方式", response.get("mode")),
            _detail("返回候选", f"{_integer(response.get('candidate_count'))} 条"),
            _detail("新增候选", (
                f"{_integer(response.get('new_candidate_count'))} 条"
                if "new_candidate_count" in response else ""
            )),
        )
    if name == "rerank.cross_encoder_http":
        return _details(
            _detail("排序轮次", f"第 {_integer(response.get('attempt') or request.get('attempt')) or 1} 轮"),
            _detail("比较候选", f"{_integer(response.get('candidate_count') or request.get('candidate_count'))} 条"),
            _detail("模型", response.get("model") or "语义重排服务"),
            _detail("使用模型", _yes_no(response.get("used_model"))),
            _detail("模型耗时", (
                f"{_integer(response.get('latency_ms'))} ms"
                if _integer(response.get("latency_ms")) else ""
            )),
            _detail("降级原因", response.get("fallback_reason")),
        )
    if name in {"answer.ollama_json", "answer.refusal_policy"}:
        return _details(
            _detail("模型", request.get("model") or "未调用回答模型"),
            _detail("读取依据", f"{_integer(request.get('selected_evidence'))} 条"),
            _detail("充分性状态", _SUFFICIENCY_LABELS.get(
                str(request.get("sufficiency_state") or ""),
                str(request.get("sufficiency_state") or ""),
            )),
            _detail("形成陈述", f"{_integer(response.get('claim_count'))} 项"),
            _detail("是否拒答", _yes_no(response.get("refused"))),
            _detail("执行结果", (
                "模型服务暂时不可用，未生成回答"
                if str(tool.get("status") or "") == "degraded" else ""
            )),
        )
    if name == "answer.scope_policy":
        route = str(response.get("scope_route") or request.get("scope_route") or "")
        return _details(
            _detail("范围结论", _QUERY_SCOPE_LABELS.get(route, route)),
            _detail("事实回答模型", "未调用"),
            _detail("形成知识陈述", "0 项"),
            _detail("引用", "0 个"),
            _detail("执行结果", (
                "模型服务暂时不可用，请稍后重试"
                if route == "service_degraded" else
                "需要补充对象或场景后重试"
                if route == "uncertain" else ""
            )),
        )
    return _details(
        _detail("内部工具", name or "受控内部工具"),
        _detail("执行状态", {
            "queued": "等待开始", "running": "执行中",
            "completed": "已完成", "degraded": "已降级完成",
            "failed": "失败", "cancelled": "已停止",
        }.get(str(tool.get("status") or ""), "处理中")),
    )


def _evidence_cards(
    run: dict[str, Any], *, cited_only: bool = False,
) -> list[dict[str, Any]]:
    """Project bounded evidence rows into clickable, browser-safe summaries."""

    result: list[dict[str, Any]] = []
    for value in run.get("evidence_summaries") or []:
        row = _mapping(value)
        evidence_id = _clip(row.get("evidence_id"), 100)
        if not evidence_id:
            continue
        citation_index = _integer(row.get("citation_index"))
        if cited_only and not citation_index:
            continue
        location = " · ".join(filter(None, (
            _clip(row.get("filename"), 160),
            _clip(row.get("page_label"), 80),
            _clip(row.get("section_title"), 180),
        )))
        result.append({
            "evidence_id": evidence_id,
            "evidence_type": _clip(row.get("evidence_type"), 60),
            "type_label": _EVIDENCE_TYPE_LABELS.get(
                str(row.get("evidence_type") or ""), "知识依据",
            ),
            "title": _clip(row.get("title") or "知识依据", 180),
            "excerpt": _clip(row.get("excerpt"), 500),
            "location": location,
            "rank": _integer(row.get("rank")),
            "citation_index": citation_index,
            "cited": bool(citation_index),
        })
    return result[:24]


def _item_evidence(
    *, kind: str, technical_name: str,
    selected: list[dict[str, Any]], cited: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    if kind == "step":
        labels = {
            "retrieving": "本次召回中最终保留的证据",
            "ranking": "重排后进入回答上下文的证据",
            "reranking": "再次重排后进入回答上下文的证据",
            "assessing": "用于充分性判断的证据",
            "supplementing": "补充检索后最终保留的证据",
            "reassessing": "用于再次判断充分性的证据",
            "checking_scope": "用于确认问题范围的证据",
            "answering": "交给回答模型的具体证据",
            "validating_citations": "回答最终引用的具体证据",
        }
        title = labels.get(technical_name, "")
    else:
        if technical_name.startswith("retrieve."):
            title = "该次召回中最终保留的证据"
        elif technical_name == "rerank.cross_encoder_http":
            title = "重排后进入回答上下文的证据"
        elif technical_name == "scope.evidence_alignment":
            title = "用于确认问题范围的证据"
        elif technical_name in {"answer.ollama_json", "answer.refusal_policy"}:
            title = "回答阶段读取的具体证据"
        else:
            title = ""
    if not title:
        return "", []
    if technical_name == "validating_citations":
        return title, cited
    return title, selected


def _decisions(run: dict[str, Any]) -> list[dict[str, str]]:
    metadata = _mapping(run.get("metadata"))
    context = _mapping(metadata.get("context_resolution"))
    query_scope = _mapping(metadata.get("query_scope"))
    plan = _mapping(metadata.get("retrieval_plan"))
    retrieval = _mapping(run.get("retrieval_summary"))
    sufficiency = _mapping(run.get("sufficiency"))
    result: list[dict[str, str]] = []
    policy_summary = _policy_summary(run)
    if policy_summary:
        result.append({
            "title": "分层管控",
            "detail": policy_summary,
        })
    if context:
        result.append({"title": "问题理解", "detail": _context_detail(run)})
    if query_scope:
        route = str(query_scope.get("route") or "")
        result.append({
            "title": "回答范围",
            "detail": (
                f"范围门禁判定为{_QUERY_SCOPE_LABELS.get(route, '未确认')}；"
                + (
                    "仅在当前知识库内组织有依据的回答"
                    if route == "knowledge" else
                    "当前无法可靠确认范围，需要补充对象或场景"
                    if route == "uncertain" else
                    "模型服务暂时不可用，未生成事实回答"
                    if route == "service_degraded" else
                    "不会用无关证据或通用知识补答"
                )
            ),
        })
    if plan:
        intent = _INTENT_LABELS.get(str(plan.get("intent") or ""), "通用查询")
        strategy = "GraphRAG" if str(run.get("strategy_key") or "") == "graphrag_native" else "Knowledge Forge"
        mode = str(plan.get("native_mode") or "").upper()
        detail = f"按{intent}问题分配检索预算，使用 {strategy}"
        if strategy == "GraphRAG" and mode:
            detail += f" {mode}"
        budget = _family_budget(plan.get("family_limits"))
        if budget:
            detail += "；候选预算：" + budget
        result.append({"title": "检索选择", "detail": detail})
    supplemental = _mapping(retrieval.get("supplemental_retrieval"))
    if supplemental.get("attempted"):
        count = _integer(supplemental.get("new_candidate_count"))
        result.append({
            "title": "补充检索",
            "detail": f"首次证据存在缺口，执行唯一一次补充尝试并取得 {count} 条候选",
        })
    if sufficiency:
        selected = _integer(retrieval.get("selected"))
        detail = _sufficiency_detail(sufficiency, run)
        if selected:
            detail += f"；最终使用 {selected} 条依据"
        result.append({"title": "证据判断", "detail": detail})
    return result


def _analysis_sections(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a detailed, deterministic work record from persisted summaries."""

    metadata = _mapping(run.get("metadata"))
    context = _mapping(metadata.get("context_resolution"))
    query_scope = _mapping(metadata.get("query_scope"))
    retrieval = _mapping(run.get("retrieval_summary"))
    diagnostics = _mapping(metadata.get("retrieval_diagnostics"))
    plan = _mapping(metadata.get("retrieval_plan") or retrieval.get("retrieval_plan"))
    sufficiency = _mapping(run.get("sufficiency"))
    supplemental = _mapping(
        retrieval.get("supplemental_retrieval")
        or metadata.get("supplemental_retrieval")
    )
    budget = _mapping(run.get("budget"))
    steps = [dict(item) for item in run.get("steps") or [] if isinstance(item, Mapping)]
    tools = [dict(item) for item in run.get("tool_calls") or [] if isinstance(item, Mapping)]
    step_by_type = {
        str(item.get("step_type") or ""): item for item in steps
    }
    tool_by_name = {
        str(item.get("tool_name") or ""): item for item in tools
    }
    sections: list[dict[str, Any]] = []

    def add(
        key: str, title: str, summary: str,
        details: list[dict[str, str]], status: str = "completed",
    ) -> None:
        clean_summary = _clip(summary, 700)
        if not clean_summary and not details:
            return
        sections.append({
            "key": key,
            "title": title,
            "summary": clean_summary,
            "details": details,
            "status": status or "completed",
        })

    policy_summary = _policy_summary(run)
    policy_details = _policy_details(run)
    if policy_summary or policy_details:
        add(
            "policy",
            "分层管控",
            "本轮按知识范围、问题输入、多轮语境、检索证据、工具执行和回答输出逐层处理。",
            policy_details,
        )

    if context:
        original = _clip(context.get("original_question") or run.get("query"), 260)
        standalone = _clip(context.get("standalone_question"), 320)
        references = []
        for item in context.get("resolved_references") or []:
            row = _mapping(item)
            expression = _clip(row.get("expression"), 60)
            referent = _clip(row.get("referent"), 120)
            if expression and referent:
                references.append(f"“{expression}”指向“{referent}”")
        knowledge_scope = _mapping(context.get("knowledge_scope"))
        documents = [
            _clip(_mapping(item).get("filename"), 140)
            for item in knowledge_scope.get("documents") or []
            if _clip(_mapping(item).get("filename"), 140)
        ]
        history_count = _integer(context.get("history_message_count"))
        dependency = str(context.get("dependency") or "")
        if dependency == "independent":
            summary = "本轮问题可以独立理解，不需要借助历史回答补全后再检索。"
        elif dependency == "resolved":
            summary = "系统结合当前会话解析了省略或指代，并生成独立检索问题。"
        else:
            summary = "系统检查了会话语境；仍无法唯一确定的对象会先请求澄清。"
        add(
            "context", "理解问题", summary,
            _details(
                _detail("用户原问题", original),
                _detail("实际检索问题", standalone),
                _detail("指代解析", "；".join(references)),
                _detail("使用的历史消息", f"{history_count} 条" if history_count else "未使用"),
                _detail("语境处理方式", (
                    "调用一次结构化语境模型"
                    if context.get("model_used")
                    else "确定性判断；未调用语境模型"
                )),
                _detail("知识范围", (
                    f"{_integer(knowledge_scope.get('document_count'))} 份文档"
                    + ("：" + "、".join(documents[:6]) if documents else "")
                )),
                _detail("语境 Token 窗口", (
                    f"本轮 {_integer(context.get('context_limit_tokens')):,}；"
                    f"模型有效上限 {_integer(context.get('model_context_tokens')):,}"
                    if _integer(context.get("context_limit_tokens")) else ""
                )),
                _detail("历史压缩", (
                    "已按 Token 预算压缩较早消息"
                    if context.get("token_budget_triggered")
                    else "未触发"
                )),
            ),
            str(step_by_type.get("resolving_context", {}).get("status") or "completed"),
        )

    if query_scope:
        route = str(query_scope.get("route") or "")
        method = str(query_scope.get("method") or "")
        add(
            "scope", "确认回答范围",
            (
                f"系统判定本轮问题{_QUERY_SCOPE_LABELS.get(route, '范围未确认')}。"
                + (
                    "后续回答只能使用当前知识库证据。"
                    if route == "knowledge" else
                    "本轮不会生成知识库事实答案，也不会使用通用知识补充。"
                )
            ),
            _details(
                _detail("范围结论", _QUERY_SCOPE_LABELS.get(route, route)),
                _detail("判断方式", {
                    "deterministic_interaction_intent": "确定性交互意图规则",
                    "strong_retrieval_signals": "强检索相关信号直接通过",
                    "model_evidence_alignment": "本地模型结构化语义核验",
                    "no_evidence": "未取得可用知识证据",
                    "fail_closed": "核验异常后失败关闭",
                    "service_degraded": "范围判断服务异常",
                }.get(method, "受控范围策略")),
                _detail("调用范围模型", _yes_no(query_scope.get("model_used"))),
                _detail("直接相关证据", f"{len(query_scope.get('matched_evidence_ids') or [])} 条"),
                _detail("置信度", _percent(query_scope.get("confidence"))),
                _detail("回答边界", (
                    "只允许依据当前知识库回答"
                    if route == "knowledge" else
                    "暂不生成事实回答，待服务恢复后重试"
                    if route == "service_degraded" else
                    "暂不生成事实回答，请补充问题范围"
                    if route == "uncertain" else
                    "不回答知识库范围外的问题"
                )),
            ),
            str(step_by_type.get("checking_scope", {}).get("status") or "completed"),
        )

    if plan:
        intent = _INTENT_LABELS.get(str(plan.get("intent") or ""), "通用查询")
        scope = _SCOPE_LABELS.get(
            str(plan.get("scope") or ""), str(plan.get("scope") or "") or "默认范围",
        )
        strategy = _strategy_label(run.get("strategy_key"))
        mode = str(plan.get("native_mode") or "").upper()
        mode_phrase = f"，查询方式为 {mode}" if str(run.get("strategy_key")) == "graphrag_native" and mode else ""
        add(
            "plan", "制定检索计划",
            f"系统将问题识别为{intent}，在{scope}内使用 {strategy}{mode_phrase}；各证据类型均保留安全下限。",
            _details(
                _detail("问题类型", intent),
                _detail("检索范围", scope),
                _detail("路由置信度", _percent(plan.get("confidence"))),
                _detail("证据候选配额", _family_phrase(plan.get("family_limits"))),
                _detail("识别出的对象", "、".join(
                    _clip(item, 100) for item in plan.get("referenced_terms") or []
                    if _clip(item, 100)
                )),
                _detail("计划依据", _reason_phrase(plan.get("reason_codes"))),
                _detail("允许补充检索", _yes_no(plan.get("retry_allowed"))),
            ),
            str(tool_by_name.get("plan.retrieval.policy", {}).get("status") or "completed"),
        )

    retrieval_step = step_by_type.get("retrieving", {})
    retrieval_output = _mapping(retrieval_step.get("output_summary"))
    if retrieval or diagnostics or retrieval_output:
        primary_probe = _mapping(diagnostics.get("primary_probe"))
        channel_counts = _family_phrase(primary_probe.get("counts_by_family"), include_zero=True)
        source_snapshot = _mapping(diagnostics.get("source_snapshot"))
        candidate_count = _integer(
            retrieval.get("retrieved") or diagnostics.get("candidate_count")
            or retrieval_output.get("candidates")
        )
        dropped = _integer(diagnostics.get("dropped_without_source"))
        graph_used = bool(
            diagnostics.get("graph_used") or retrieval_output.get("graph_used")
        )
        summary = (
            f"使用 {retrieval.get('mode') or retrieval_output.get('mode') or '当前'} 检索取得 {candidate_count} 条候选"
        )
        if channel_counts:
            summary += f"；各通道原始候选为 {channel_counts}，通道之间可能重叠"
        if dropped:
            summary += f"；{dropped} 条缺少可追溯来源的结果被淘汰"
        summary += "。"
        navigation = _mapping(diagnostics.get("community_navigation"))
        media = _mapping(diagnostics.get("page_media_enrichment"))
        visual = _mapping(diagnostics.get("numbered_visual_recall"))
        native_budget = _mapping(diagnostics.get("native_query_budget"))
        navigation_status = str(navigation.get("status") or "")
        navigation_detail = ""
        if navigation_status:
            navigation_detail = _COMMUNITY_NAVIGATION_REASONS.get(
                str(navigation.get("reason") or ""),
                _reason_label(navigation.get("reason")),
            )
            if navigation_status in {"completed", "used", "expanded"}:
                navigation_detail = (
                    f"扩展 {_integer(navigation.get('factual_links_used'))} 条事实联系和 "
                    f"{_integer(navigation.get('topic_links_used'))} 条内容导航联系，"
                    f"恢复 {_integer(navigation.get('new_relation_count'))} 条关系候选"
                )
        add(
            "retrieval", "执行多路召回", summary,
            _details(
                _detail("实际检索方式", retrieval.get("mode") or retrieval_output.get("mode")),
                _detail("召回候选总数", f"{candidate_count} 条"),
                _detail("各通道原始候选", channel_counts),
                _detail("当前来源规模", (
                    f"{_integer(source_snapshot.get('document_count'))} 份文档、"
                    f"{_integer(source_snapshot.get('chunk_count'))} 个知识块、"
                    f"{_integer(source_snapshot.get('embedding_count'))} 个向量"
                    if source_snapshot else ""
                )),
                _detail("知识图谱", "参与关系召回" if graph_used else "本轮未使用或不可用"),
                _detail("社区导航", navigation_detail),
                _detail("图片与表格来源", (
                    f"附加 {_integer(media.get('attached_assets'))} 个媒体资源，覆盖 "
                    f"{_integer(media.get('attached_references'))} 个来源引用"
                    if media else ""
                )),
                _detail("编号图表恢复", (
                    "本轮问题不需要"
                    if visual.get("status") == "not_needed"
                    else (
                        f"匹配 {_integer(visual.get('matched_chunks'))} 个知识块，"
                        f"补入 {_integer(visual.get('added_chunks'))} 个候选"
                        if visual else ""
                    )
                )),
                _detail("来源门禁淘汰", f"{dropped} 条" if dropped else "0 条"),
                _detail("GraphRAG 查询预算", (
                    f"单批 {_integer(native_budget.get('global_batch_context_tokens')):,} Token；"
                    f"模型窗口 {_integer(native_budget.get('model_context_tokens')):,} Token"
                    if native_budget else ""
                )),
                _detail("GraphRAG 候选答案", (
                    "只作为不可引用草稿，最终答案仍需重新依据来源生成"
                    if diagnostics.get("candidate_answer_retained_as_draft_only") else ""
                )),
            ),
            str(retrieval_step.get("status") or "completed"),
        )

    ranking_step = step_by_type.get("reranking") or step_by_type.get("ranking") or {}
    ranking_output = _mapping(ranking_step.get("output_summary"))
    reranker = _mapping(retrieval.get("reranker"))
    if ranking_output or retrieval:
        plan_execution = _mapping(
            retrieval.get("retrieval_plan_execution") or ranking_output.get("plan_execution")
        )
        context_execution = _mapping(
            retrieval.get("context_plan_execution") or ranking_output.get("context_plan_execution")
        )
        compared = _integer(reranker.get("candidate_count") or ranking_output.get("candidates"))
        context_pool = _integer(retrieval.get("context_pool"))
        selected = _integer(retrieval.get("selected") or ranking_output.get("selected"))
        summary = (
            f"候选去重并经 RRF 融合后，{compared} 条进入语义重排"
            + (f"，{context_pool} 条通过相关性下限" if context_pool else "")
            + f"，最终选择 {selected} 条作为回答依据。"
        )
        add(
            "ranking", "融合、重排与上下文选择", summary,
            _details(
                _detail("融合方法", "去重后按多通道排名执行 RRF 融合"),
                _detail("语义重排", (
                    f"{reranker.get('model') or '内部 reranker'} 比较 {compared} 条候选；"
                    f"耗时 {_integer(reranker.get('latency_ms'))} ms"
                    if reranker.get("used_model") else
                    "Reranker 未成功调用，使用确定性融合排序降级"
                )),
                _detail("可用候选组成", _family_phrase(plan_execution.get("available_by_family"))),
                _detail("进入最终排序的组成", _family_phrase(plan_execution.get("selected_by_family"))),
                _detail("回答上下文组成", _family_phrase(context_execution.get("selected_by_family"), include_zero=True)),
                _detail("配额自动回填", (
                    f"候选阶段 { _integer(plan_execution.get('backfilled')) } 条；"
                    f"上下文阶段 { _integer(context_execution.get('backfilled')) } 条"
                )),
                _detail("最终依据", f"{selected} 条，来自 {_integer(retrieval.get('document_count'))} 份文档"),
                _detail("证据 Token 预算", f"{_integer(retrieval.get('context_budget_tokens')):,} Token"),
            ),
            str(ranking_step.get("status") or "completed"),
        )

    assessment_step = step_by_type.get("reassessing") or step_by_type.get("assessing") or {}
    if sufficiency:
        state = str(sufficiency.get("state") or "")
        attempted = bool(supplemental.get("attempted"))
        if attempted:
            supplemental_summary = (
                f"首次证据存在缺口，已执行一次补充检索并新增 "
                f"{_integer(supplemental.get('new_candidate_count'))} 条候选"
            )
        else:
            supplemental_summary = "证据已达到回答门槛，不执行额外检索"
        add(
            "assessment", "检查证据充分性与缺口",
            f"综合判断为{_SUFFICIENCY_LABELS.get(state, state or '已检查')}，评分 {_percent(sufficiency.get('score'))}；{supplemental_summary}。",
            _details(
                _detail("综合评分", _percent(sufficiency.get("score"))),
                _detail("Reranker 相关度", _percent(sufficiency.get("relevance"))),
                _detail("Embedding 语义相关度", _percent(sufficiency.get("semantic_relevance"))),
                _detail("问题要点覆盖", _percent(sufficiency.get("aspect_coverage"))),
                _detail("独立支持", _percent(sufficiency.get("independent_support"))),
                _detail("关系连通性", _percent(sufficiency.get("graph_connectivity"))),
                _detail("来源完整性", _percent(sufficiency.get("provenance"))),
                _detail("冲突惩罚", _percent(sufficiency.get("conflict_penalty"))),
                _detail("判断原因", _reason_phrase(sufficiency.get("reasons")) or "未触发拒答、歧义或冲突规则"),
                _detail("补充检索", supplemental_summary),
                _detail("补充检索依据", _reason_phrase(supplemental.get("reason_codes"))),
                _detail("补充查询", (
                    supplemental.get("query")
                    if attempted and supplemental.get("query") != run.get("query") else ""
                )),
            ),
            str(assessment_step.get("status") or "completed"),
        )

    answer_step = step_by_type.get("answering", {})
    validation_step = step_by_type.get("validating_citations", {})
    answer_output = _mapping(answer_step.get("output_summary"))
    validation_output = _mapping(validation_step.get("output_summary"))
    answer_tool = (
        tool_by_name.get("answer.ollama_json")
        or tool_by_name.get("answer.refusal_policy")
        or tool_by_name.get("answer.scope_policy") or {}
    )
    answer_response = _mapping(answer_tool.get("response_summary"))
    if answer_output or validation_output or answer_tool:
        claims = _integer(validation_output.get("claims") or answer_output.get("claim_count"))
        citations = _integer(validation_output.get("citations"))
        selected_by_family = _mapping(
            _mapping(retrieval.get("context_plan_execution")).get("selected_by_family")
        )
        scope_route = str(answer_response.get("scope_route") or "")
        refused = bool(
            answer_response.get("refused")
            or scope_route in {
                "out_of_scope", "unsupported", "uncertain", "service_degraded",
            }
        )
        direct_scope = bool(scope_route)
        add(
            "answer", "生成回答并核验引用",
            (
                (
                    "系统返回智能体角色与能力说明，没有执行知识库事实生成。"
                    if scope_route == "agent_meta" else
                    "模型服务暂时不可用，系统没有生成知识库事实回答。"
                    if scope_route == "service_degraded" else
                    "问题范围无法可靠确认，系统没有生成知识库事实回答。"
                    if scope_route == "uncertain" else
                    "系统说明当前知识库的回答边界，没有使用通用知识补答。"
                )
                if direct_scope else
                "证据不足，系统按拒答策略返回限制说明，没有让模型补写项目事实。"
                if refused else
                f"回答模型基于最终证据形成 {claims} 项陈述；服务端核验并发布 {citations} 个有效引用。"
            ),
            _details(
                _detail("回答模型", (
                    "未调用事实回答模型"
                    if direct_scope else run.get("model_name") or "当前知识库模型"
                )),
                _detail("实际问答方式", run.get("effective_mode") or retrieval.get("mode")),
                _detail("读取依据", f"{_integer(retrieval.get('selected'))} 条"),
                _detail("依据类型", _evidence_type_phrase(retrieval.get("evidence_types"))),
                _detail("依据组成", _family_phrase(selected_by_family, include_zero=True)),
                _detail("可核验陈述", f"{claims} 项"),
                _detail("有效引用", f"{citations} 个"),
                _detail("回答输出上限", f"{_integer(retrieval.get('max_output_tokens') or budget.get('max_output_tokens')):,} Token"),
                _detail("引用门禁", (
                    "本轮不生成知识陈述，因此不需要引用"
                    if direct_scope else "移除未知引用、无依据陈述和非法声明后才发布"
                )),
                _detail("是否拒答", _yes_no(refused)),
            ),
            str(validation_step.get("status") or answer_step.get("status") or "completed"),
        )

    if budget:
        add(
            "bounds", "本轮运行边界",
            "该流程按提交时固化的预算执行，不会无限检索、无限调用模型或自行扩大知识库范围。",
            _details(
                _detail("知识库范围", "1 个用户明确选择的知识库"),
                _detail("检索调用上限", f"{_integer(budget.get('maximum_retrieval_calls'))} 次"),
                _detail("补充检索上限", f"{_integer(budget.get('maximum_supplemental_attempts'))} 次"),
                _detail("检索前范围分类上限", f"{_integer(budget.get('maximum_input_scope_calls'))} 次"),
                _detail("Reranker 调用上限", f"{_integer(budget.get('maximum_reranker_calls'))} 次"),
                _detail("回答生成上限", f"{_integer(budget.get('maximum_answer_generations'))} 次"),
                _detail("引用核验上限", f"{_integer(budget.get('maximum_citation_validation_passes'))} 次"),
                _detail("快照规则", "回答发布前再次确认知识库快照未变化"),
            ),
            str(run.get("status") or "completed"),
        )

    return sections


def build_public_execution_trace(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the public execution summary for one authorized run payload."""

    run = dict(payload)
    status = str(run.get("status") or "queued")
    running = status in {"queued", "running"}
    selected_evidence = _evidence_cards(run)
    cited_evidence = _evidence_cards(run, cited_only=True)
    items: list[dict[str, Any]] = []
    steps = [dict(item) for item in run.get("steps") or [] if isinstance(item, Mapping)]
    tools = [dict(item) for item in run.get("tool_calls") or [] if isinstance(item, Mapping)]
    for step in steps:
        stage = str(step.get("step_type") or "")
        item_status = str(step.get("status") or "running")
        evidence_title, evidence = _item_evidence(
            kind="step", technical_name=stage,
            selected=selected_evidence, cited=cited_evidence,
        )
        items.append({
            "kind": "step",
            "key": f"step:{_integer(step.get('sequence'))}:{stage}",
            "title": _STAGE_TITLES.get(stage, "执行问答步骤"),
            "detail": _clip(_stage_detail(stage, step, run), 360),
            "details": _stage_facts(stage, step, run),
            "evidence_title": evidence_title,
            "evidence": evidence,
            "status": item_status,
            "technical_name": stage,
            "started_at": _iso(step.get("started_at")),
            "completed_at": _iso(step.get("completed_at")),
            "duration_ms": _duration_ms(
                step.get("started_at"), step.get("completed_at"),
                running=item_status == "running",
            ),
        })
    for tool in tools:
        name = str(tool.get("tool_name") or "")
        item_status = str(tool.get("status") or "running")
        evidence_title, evidence = _item_evidence(
            kind="tool", technical_name=name,
            selected=selected_evidence, cited=cited_evidence,
        )
        items.append({
            "kind": "tool",
            "key": "tool:" + _clip(tool.get("id"), 100),
            "title": _TOOL_TITLES.get(name, "内部工具调用"),
            "detail": _clip(_tool_detail(name, tool, run), 360),
            "details": _tool_facts(name, tool, run),
            "evidence_title": evidence_title,
            "evidence": evidence,
            "status": item_status,
            "technical_name": name,
            "started_at": _iso(tool.get("started_at")),
            "completed_at": _iso(tool.get("completed_at")),
            "duration_ms": _duration_ms(
                tool.get("started_at"), tool.get("completed_at"),
                running=item_status == "running",
            ),
        })
    items.sort(key=lambda item: (
        item.get("started_at") or "9999",
        0 if item.get("kind") == "step" else 1,
        item.get("key") or "",
    ))
    started_at = run.get("started_at") or run.get("created_at")
    completed_at = run.get("completed_at")
    return {
        "protocol": EXECUTION_TRACE_PROTOCOL,
        "status": status,
        "current_stage": str(run.get("stage") or status),
        "current_label": _STAGE_TITLES.get(
            str(run.get("stage") or status), "正在处理"
        ),
        "started_at": _iso(started_at),
        "completed_at": _iso(completed_at),
        "duration_ms": _duration_ms(started_at, completed_at, running=running),
        "step_count": len(steps),
        "tool_count": len(tools),
        "selected_evidence_count": len(selected_evidence),
        "cited_evidence_count": len(cited_evidence),
        "decisions": _decisions(run),
        "items": items,
        "disclosure": (
            "这里展示可审计的处理步骤、检索决策和工具调用；"
            "不包含模型隐藏推理、系统提示词或未验证草稿。"
        ),
    }


__all__ = ["EXECUTION_TRACE_PROTOCOL", "build_public_execution_trace"]
