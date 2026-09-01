"""Deterministic, auditable retrieval planning for the knowledge agent.

The planner intentionally does not infer domain facts.  It only identifies
generic question shape and allocates a bounded recall budget across the
retrieval families already owned by the selected knowledge strategy.  A low
confidence analysis falls back to a balanced plan instead of disabling a
channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable

from ..construction_strategies import GRAPHRAG_NATIVE_STRATEGY
from ..unicode_safety import sanitize_unicode_text


RETRIEVAL_PLAN_PROTOCOL = "knowledge-agent-retrieval-plan-v1"
RETRIEVAL_FAMILIES = ("text", "entity", "relation", "community")
QUERY_INTENTS = frozenset({
    "fact_lookup",
    "entity_explanation",
    "relation_explanation",
    "procedure",
    "comparison",
    "global_summary",
    "source_lookup",
    "visual_lookup",
    "ambiguous",
})
QUERY_SCOPES = frozenset({"local", "multi_entity", "global"})


_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("visual_lookup", re.compile(
        r"图(?:片|表|像|示|中|里|上|下|\s*\d+)|表(?:格|中|里|上|下|\s*\d+)|"
        r"示意图|流程图|曲线|截图|插图|"
        r"\b(?:figure|fig\.?|table|chart|diagram|image|illustration)\b",
        re.IGNORECASE,
    )),
    ("source_lookup", re.compile(
        r"原文|出处|来源|哪一页|第几页|章节|条款|引用|依据|文档中|"
        r"\b(?:source|citation|reference|page|section|clause|document)\b",
        re.IGNORECASE,
    )),
    ("global_summary", re.compile(
        r"总体|整体|全局|概览|综述|总结|主要主题|共同趋势|核心内容|"
        r"跨(?:领域|文档|章节)|"
        r"\b(?:overall|global|overview|summari[sz]e|main themes?|trend|across documents?)\b",
        re.IGNORECASE,
    )),
    ("comparison", re.compile(
        r"比较|对比|区别|差异|异同|优缺点|相同点|不同点|"
        r"(?:与|和|跟).{0,40}(?:相比|比较|区别|差异)|"
        r"\b(?:compare|comparison|versus|vs\.?|difference|similarit|pros and cons)\b",
        re.IGNORECASE,
    )),
    ("procedure", re.compile(
        r"如何|怎么|怎样|步骤|流程|过程|先后|操作方法|实施|执行|"
        r"\b(?:how to|steps?|procedure|workflow|process|implement|configure|operate)\b",
        re.IGNORECASE,
    )),
    ("relation_explanation", re.compile(
        r"为什么|原因|影响|作用|关系|联系|依赖|导致|促进|制约|机制|原理|"
        r"如何(?:影响|作用于|关联)|"
        r"\b(?:why|relationship|relate|depend|cause|effect|impact|mechanism|interact)\b",
        re.IGNORECASE,
    )),
    ("entity_explanation", re.compile(
        r"是什么|指什么|含义|定义|介绍|说明|有哪些(?:组成|部分|类型|类别|功能)|"
        r"\b(?:what is|define|definition|meaning|describe|explain|components?|types?)\b",
        re.IGNORECASE,
    )),
)


_FAMILY_WEIGHTS: dict[str, dict[str, float]] = {
    "fact_lookup": {
        "text": 0.55, "entity": 0.15, "relation": 0.22, "community": 0.08,
    },
    "entity_explanation": {
        "text": 0.40, "entity": 0.30, "relation": 0.22, "community": 0.08,
    },
    "relation_explanation": {
        "text": 0.28, "entity": 0.20, "relation": 0.42, "community": 0.10,
    },
    "procedure": {
        "text": 0.48, "entity": 0.12, "relation": 0.32, "community": 0.08,
    },
    "comparison": {
        "text": 0.35, "entity": 0.20, "relation": 0.35, "community": 0.10,
    },
    "global_summary": {
        "text": 0.20, "entity": 0.10, "relation": 0.20, "community": 0.50,
    },
    "source_lookup": {
        "text": 0.70, "entity": 0.10, "relation": 0.15, "community": 0.05,
    },
    "visual_lookup": {
        "text": 0.65, "entity": 0.10, "relation": 0.20, "community": 0.05,
    },
    "ambiguous": {
        "text": 0.45, "entity": 0.20, "relation": 0.25, "community": 0.10,
    },
}


def _clean_terms(values: Iterable[object] | None) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        sanitize_unicode_text(str(value)).strip()
        for value in (values or ())
        if sanitize_unicode_text(str(value or "")).strip()
    ))


def _allocate_budget(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation with a safe floor for every family."""

    total = max(len(RETRIEVAL_FAMILIES), int(total))
    allocation = {family: 1 for family in RETRIEVAL_FAMILIES}
    remaining = total - len(RETRIEVAL_FAMILIES)
    if remaining <= 0:
        return allocation
    weight_total = sum(max(0.0, float(weights.get(family, 0.0))) for family in RETRIEVAL_FAMILIES)
    normalized = {
        family: (
            max(0.0, float(weights.get(family, 0.0))) / weight_total
            if weight_total > 0 else 1.0 / len(RETRIEVAL_FAMILIES)
        )
        for family in RETRIEVAL_FAMILIES
    }
    exact = {family: remaining * normalized[family] for family in RETRIEVAL_FAMILIES}
    for family in RETRIEVAL_FAMILIES:
        allocation[family] += int(math.floor(exact[family]))
    residue = total - sum(allocation.values())
    for family in sorted(
        RETRIEVAL_FAMILIES,
        key=lambda value: (exact[value] - math.floor(exact[value]), normalized[value], value),
        reverse=True,
    )[:residue]:
        allocation[family] += 1
    return allocation


def _scale_execution_budget(
    total: int, configured: dict[str, int],
) -> dict[str, int]:
    """Scale a plan to a smaller execution stage without changing its intent.

    Retrieval plans are normally created for the broad candidate pool.  The
    final answer context is deliberately much smaller, so reusing those raw
    counts would let the first family consume every slot.  This helper keeps
    the same proportions, preserves a one-slot floor when the stage has room
    for all families, and deterministically chooses the strongest families
    when it does not.
    """

    bounded_total = max(1, int(total))
    if bounded_total >= len(RETRIEVAL_FAMILIES):
        return _allocate_budget(bounded_total, {
            family: max(0.0, float(configured.get(family, 0)))
            for family in RETRIEVAL_FAMILIES
        })
    allocation = {family: 0 for family in RETRIEVAL_FAMILIES}
    ordered = sorted(
        RETRIEVAL_FAMILIES,
        key=lambda family: (
            int(configured.get(family, 0)),
            -RETRIEVAL_FAMILIES.index(family),
        ),
        reverse=True,
    )
    for family in ordered[:bounded_total]:
        allocation[family] = 1
    return allocation


@dataclass(frozen=True)
class RetrievalPlan:
    protocol: str
    activation: str
    intent: str
    scope: str
    confidence: float
    family_limits: dict[str, int]
    native_mode: str
    retry_allowed: bool
    referenced_terms: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.protocol != RETRIEVAL_PLAN_PROTOCOL:
            raise ValueError("unknown retrieval-plan protocol")
        if self.activation not in {"observe", "active"}:
            raise ValueError("unknown retrieval-plan activation")
        if self.intent not in QUERY_INTENTS or self.scope not in QUERY_SCOPES:
            raise ValueError("invalid retrieval-plan intent or scope")
        if set(self.family_limits) != set(RETRIEVAL_FAMILIES):
            raise ValueError("retrieval-plan families are incomplete")
        if any(int(value) < 1 for value in self.family_limits.values()):
            raise ValueError("retrieval-plan family limits must be positive")
        if self.native_mode not in {"basic", "local", "global"}:
            raise ValueError("invalid native retrieval mode")

    @property
    def candidate_limit(self) -> int:
        return sum(int(value) for value in self.family_limits.values())

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupplementalDecision:
    execute: bool
    candidate_limit: int
    query: str
    plan: RetrievalPlan | None
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["plan"] = self.plan.to_dict() if self.plan else None
        return value


def plan_retrieval(
    query: str,
    *,
    strategy: str,
    candidate_limit: int,
    requested_mode: str = "auto",
    referenced_terms: Iterable[object] | None = None,
    knowledge_scope: dict[str, Any] | None = None,
    activation: str = "observe",
) -> RetrievalPlan:
    """Classify generic query shape and produce a bounded safe plan."""

    text = sanitize_unicode_text(str(query or "")).strip()
    matches = [
        intent for intent, pattern in _CUE_PATTERNS if pattern.search(text)
    ]
    if not text:
        intent = "ambiguous"
        confidence = 0.0
        reasons = ["empty_query", "balanced_fallback"]
    elif matches:
        # Patterns are ordered by evidence specificity.  Multiple competing
        # shapes lower confidence but never remove any retrieval family.
        intent = matches[0]
        confidence = 0.90 if len(matches) == 1 else 0.72
        reasons = [f"cue:{value}" for value in matches[:4]]
    else:
        intent = "fact_lookup"
        confidence = 0.55
        reasons = ["default_fact_lookup", "safe_family_floors"]

    if intent == "global_summary":
        scope = "global"
    elif intent == "comparison":
        scope = "multi_entity"
    else:
        scope = "local"

    requested = str(requested_mode or "auto").strip().lower()
    if strategy == GRAPHRAG_NATIVE_STRATEGY and requested in {"basic", "local", "global"}:
        native_mode = requested
        reasons.append("explicit_native_mode")
    elif intent == "global_summary":
        scope_details = (
            knowledge_scope if isinstance(knowledge_scope, dict) else {}
        )
        document_count = max(0, int(scope_details.get("document_count") or 0))
        document_names = [
            sanitize_unicode_text(str(value)).strip()
            for value in (scope_details.get("document_names") or [])
            if sanitize_unicode_text(str(value or "")).strip()
        ]
        lowered_query = text.casefold()
        named_documents = {
            name for name in document_names if name.casefold() in lowered_query
        }
        single_document_scope = bool(
            document_count == 1 or len(named_documents) == 1
        )
        if strategy == GRAPHRAG_NATIVE_STRATEGY and single_document_scope:
            # Native Global is a map/reduce over community reports. A summary
            # constrained to one document is covered by Local without paying
            # that project-wide fan-out cost. Explicit user modes were handled
            # above and remain authoritative.
            native_mode = "local"
            reasons.append("single_document_summary_uses_local")
        else:
            native_mode = "global"
    elif intent in {"source_lookup", "visual_lookup"}:
        native_mode = "basic"
    else:
        native_mode = "local"

    return RetrievalPlan(
        protocol=RETRIEVAL_PLAN_PROTOCOL,
        activation=activation,
        intent=intent,
        scope=scope,
        confidence=max(0.0, min(1.0, confidence)),
        family_limits=_allocate_budget(
            max(len(RETRIEVAL_FAMILIES), int(candidate_limit)),
            _FAMILY_WEIGHTS[intent],
        ),
        native_mode=native_mode,
        retry_allowed=bool(text and intent != "ambiguous"),
        referenced_terms=_clean_terms(referenced_terms),
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def evidence_family(item: Any) -> str:
    """Map either strategy's stable evidence contract to one plan family."""

    evidence_type = str(getattr(item, "evidence_type", "") or "")
    if evidence_type == "chunk":
        return "text"
    if evidence_type == "entity":
        return "entity"
    if evidence_type == "relation":
        return "relation"
    if evidence_type == "community_report":
        return "community"
    metadata = getattr(item, "metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    native_table = str(metadata.get("native_table") or "").lower()
    if native_table in {"entities", "entity"}:
        return "entity"
    if native_table in {"relationships", "relationship", "relations", "relation"}:
        return "relation"
    if native_table in {"reports", "community_reports", "communities", "community"}:
        return "community"
    return "text"


def apply_retrieval_plan(
    ranked_items: list[Any],
    *,
    plan: RetrievalPlan,
    candidate_limit: int,
) -> tuple[list[Any], dict[str, object]]:
    """Reserve intent-aware family slots, then backfill every unused slot.

    Backfilling is a quality invariant: a stale KF graph, a sparse community
    index, or a native mode that returns only one table can never shrink recall
    merely because the planner reserved slots for unavailable families.
    """

    limit = max(1, int(candidate_limit))
    available = {
        family: sum(1 for item in ranked_items if evidence_family(item) == family)
        for family in RETRIEVAL_FAMILIES
    }
    if plan.activation != "active":
        selected = list(ranked_items[:limit])
        selected_counts = {
            family: sum(1 for item in selected if evidence_family(item) == family)
            for family in RETRIEVAL_FAMILIES
        }
        return selected, {
            "activation": "observe",
            "available_by_family": available,
            "selected_by_family": selected_counts,
            "backfilled": 0,
            "candidate_count": len(selected),
        }

    execution_limits = _scale_execution_budget(limit, plan.family_limits)
    selected: list[Any] = []
    selected_ids: set[str] = set()
    for family in RETRIEVAL_FAMILIES:
        family_limit = max(0, int(execution_limits[family]))
        if family_limit <= 0:
            continue
        for item in ranked_items:
            if evidence_family(item) != family:
                continue
            identity = str(getattr(item, "evidence_id", "") or id(item))
            if identity in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(identity)
            family_limit -= 1
            if family_limit <= 0 or len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    reserved_count = len(selected)
    if len(selected) < limit:
        for item in ranked_items:
            identity = str(getattr(item, "evidence_id", "") or id(item))
            if identity in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(identity)
            if len(selected) >= limit:
                break

    original_order = {
        str(getattr(item, "evidence_id", "") or id(item)): index
        for index, item in enumerate(ranked_items)
    }
    selected.sort(key=lambda item: original_order[
        str(getattr(item, "evidence_id", "") or id(item))
    ])
    selected_counts = {
        family: sum(1 for item in selected if evidence_family(item) == family)
        for family in RETRIEVAL_FAMILIES
    }
    return selected, {
        "activation": "active",
        "available_by_family": available,
        "reserved_by_family": execution_limits,
        "selected_by_family": selected_counts,
        "backfilled": max(0, len(selected) - reserved_count),
        "candidate_count": len(selected),
    }


def diagnose_primary_payload(
    payload: dict[str, Any],
    *,
    plan: RetrievalPlan,
    graph_available: bool,
) -> dict[str, object]:
    """Inspect an existing KF first pass before any optional expansion.

    The probe is deliberately model-free and storage-free.  It only counts
    rows and reads the seed diagnostics already returned by the normal search.
    """

    counts = {
        "text": len(payload.get("chunks") or []),
        "entity": len(payload.get("entities") or []),
        "relation": len(payload.get("relations") or []),
        "community": len(payload.get("community_reports") or []),
    }
    graph_state = payload.get("graph_retrieval")
    graph_state = graph_state if isinstance(graph_state, dict) else {}
    seed_rows = [
        item for item in graph_state.get("seed_entities") or []
        if isinstance(item, dict)
    ]
    strongest_seed = max(
        (float(item.get("score") or 0.0) for item in seed_rows),
        default=0.0,
    )
    reliable_seed_count = sum(
        bool(item.get("query_surface_match"))
        or float(item.get("score") or 0.0) >= max(0.48, strongest_seed - 0.06)
        for item in seed_rows
    )
    relation_floor = min(4, max(1, int(plan.family_limits["relation"])))
    reasons: list[str] = []
    if graph_available and counts["relation"] < relation_floor:
        reasons.append("sparse_primary_relations")
    if graph_available and reliable_seed_count > 1:
        reasons.append("multiple_reliable_entity_seeds")
    if graph_available and counts["community"] > 0 and reliable_seed_count == 0:
        reasons.append("community_report_seed_available")
    recommended = bool(
        graph_available
        and plan.retry_allowed
        and reasons
        and (reliable_seed_count > 0 or counts["community"] > 0)
    )
    return {
        "protocol": "knowledge-agent-primary-probe-v1",
        "counts_by_family": counts,
        "reliable_seed_count": reliable_seed_count,
        "relation_floor": relation_floor,
        "expansion_recommended": recommended,
        "expansion_reasons": reasons,
        "model_calls": 0,
        "embedding_calls": 0,
    }


_GENERIC_QUESTION_WORDS = re.compile(
    r"请问|请(?:介绍|说明|解释|概括|总结)?|能否|是否|为什么|为何|如何|怎样|怎么|"
    r"是什么|有什么|哪些|哪种|多少|吗|呢|么|"
    r"\b(?:please|could you|would you|what|which|why|how|is|are|does|do)\b",
    re.IGNORECASE,
)


def supplemental_keyword_query(query: str) -> str:
    """Produce a conservative second lexical surface without adding facts."""

    original = sanitize_unicode_text(str(query or "")).strip()
    candidate = _GENERIC_QUESTION_WORDS.sub(" ", original)
    candidate = re.sub(r"[？?！!。；;，,：:\s]+", " ", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = " ".join(
        token for token in candidate.split() if token not in {"是"}
    )
    candidate = re.sub(r"是$", "", candidate).strip()
    material = re.sub(r"[^a-zA-Z0-9\u3400-\u9fff]", "", candidate)
    original_normalized = re.sub(r"\s+", " ", original).strip("？?！!。；;，,：: ")
    if len(material) < 2 or candidate.casefold() == original_normalized.casefold():
        return original
    return candidate


def plan_supplemental_retrieval(
    query: str,
    *,
    strategy: str,
    primary_plan: RetrievalPlan,
    sufficiency: Any,
    maximum_candidates: int,
) -> SupplementalDecision:
    """Plan no more than one evidence-gap-driven supplemental attempt."""

    state = str(getattr(sufficiency, "state", "") or "")
    if not primary_plan.retry_allowed:
        return SupplementalDecision(
            False, 0, str(query), None, ("primary_plan_disallows_retry",),
        )
    if state not in {"insufficient", "retryable"}:
        return SupplementalDecision(
            False, 0, str(query), None, (f"sufficiency:{state or 'unknown'}",),
        )

    limit = max(4, min(32, int(maximum_candidates)))
    aspect_coverage = float(getattr(sufficiency, "aspect_coverage", 0.0) or 0.0)
    graph_connectivity = float(getattr(sufficiency, "graph_connectivity", 0.0) or 0.0)
    independent_support = float(getattr(sufficiency, "independent_support", 0.0) or 0.0)
    weights = dict(_FAMILY_WEIGHTS[primary_plan.intent])
    reasons = [f"sufficiency:{state}"]
    if aspect_coverage < 0.50:
        weights["text"] += 0.30
        reasons.append("low_aspect_coverage")
    if graph_connectivity < 0.75 and primary_plan.intent in {
        "relation_explanation", "procedure", "comparison",
    }:
        weights["relation"] += 0.35
        weights["entity"] += 0.10
        reasons.append("weak_graph_connectivity")
    if primary_plan.scope == "global":
        weights["community"] += 0.40
        reasons.append("global_scope_report_gap")
    if independent_support < 0.75:
        weights["text"] += 0.15
        weights["entity"] += 0.05
        reasons.append("limited_independent_support")

    explicit_native_mode = "explicit_native_mode" in primary_plan.reason_codes
    if strategy == GRAPHRAG_NATIVE_STRATEGY and not explicit_native_mode:
        native_mode = {
            "local": "basic",
            "basic": "local",
            "global": "local",
        }[primary_plan.native_mode]
        reasons.append(f"alternate_native_mode:{native_mode}")
    else:
        native_mode = primary_plan.native_mode

    plan = RetrievalPlan(
        protocol=RETRIEVAL_PLAN_PROTOCOL,
        activation="active",
        intent=primary_plan.intent,
        scope=primary_plan.scope,
        confidence=primary_plan.confidence,
        family_limits=_allocate_budget(limit, weights),
        native_mode=native_mode,
        retry_allowed=False,
        referenced_terms=primary_plan.referenced_terms,
        reason_codes=tuple(dict.fromkeys((*primary_plan.reason_codes, *reasons, "supplemental_once"))),
    )
    return SupplementalDecision(
        True,
        limit,
        supplemental_keyword_query(query),
        plan,
        tuple(dict.fromkeys(reasons)),
    )
