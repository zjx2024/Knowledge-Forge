"""Evidence sufficiency scoring and fail-closed answer policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from ..config import Config
from .contracts import EvidenceItem


STOPWORDS = frozenset({
    "什么", "什么是", "是什么", "何为", "是何", "定义", "含义", "简介",
    "如何", "为什么", "是否", "哪些", "这个", "那个", "请问", "介绍", "说明", "解释",
    "the", "a", "an", "is", "are", "what", "how", "why", "which", "please",
})


def terms(text: str) -> set[str]:
    lowered = str(text or "").lower()
    # Remove generic question/introductory surfaces before making Chinese
    # bigrams.  Otherwise strings such as “什么是草莓” create artificial
    # tokens (“么是”, “是草”) and understate topic coverage.
    normalized = lowered
    for stopword in sorted(STOPWORDS, key=len, reverse=True):
        if re.fullmatch(r"[a-z0-9_]+", stopword):
            normalized = re.sub(
                rf"(?<![a-z0-9_]){re.escape(stopword)}(?![a-z0-9_])",
                " ", normalized,
            )
        else:
            normalized = normalized.replace(stopword, " ")
    words = {
        value for value in re.findall(
            r"[a-z0-9_][a-z0-9_.:/+-]{1,}|[\u3400-\u9fff]{2,}",
            normalized,
        )
        if value not in STOPWORDS
    }
    # Keep the compound token for ranking, but also expose its meaningful
    # Latin subtokens so an abbreviation remains visible in “IOP-FL” and
    # “FL-QSAR”.  This is language- and domain-independent.
    for value in tuple(words):
        if re.fullmatch(r"[a-z0-9_][a-z0-9_.:/+-]{1,}", value):
            words.update(
                part for part in re.split(r"[^a-z0-9]+", value)
                if len(part) >= 2 and part not in STOPWORDS
            )
    cjk = re.sub(r"[^\u3400-\u9fff]", "", normalized)
    words.update(cjk[index:index + 2] for index in range(max(0, len(cjk) - 1)))
    return {value for value in words if value and value not in STOPWORDS}


@dataclass(frozen=True)
class SufficiencyResult:
    state: str
    score: float
    aspect_coverage: float
    relevance: float
    semantic_relevance: float
    reranker_used_model: bool
    provenance: float
    independent_support: float
    graph_connectivity: float
    conflict_penalty: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _conflict_penalty(items: list[EvidenceItem]) -> float:
    positive: set[tuple[str, ...]] = set()
    negative: set[tuple[str, ...]] = set()
    markers = ("不", "无", "未", "不能", "not ", "no ", "without ")
    for item in items:
        if not item.relation_keys and item.evidence_type != "relation":
            continue
        key = tuple(sorted(item.entity_keys)) + tuple(sorted(item.relation_keys))
        if not key:
            continue
        lowered = item.content.lower()
        (negative if any(marker in lowered for marker in markers) else positive).add(key)
    return 0.25 if positive & negative else 0.0


def _semantic_relevance(items: list[EvidenceItem]) -> float:
    """Return the strongest absolute embedding-like retrieval score.

    RRF and native rank weights are deliberately excluded: they describe
    ordering within a candidate set, not whether that set matches the query.
    """

    return max((
        float(score)
        for item in items
        for channel, score in item.raw_scores.items()
        if "dense" in channel.lower() or "semantic" in channel.lower()
    ), default=0.0)


def assess_sufficiency(
    query: str,
    items: list[EvidenceItem],
    *,
    reranker_used_model: bool = False,
) -> SufficiencyResult:
    query_terms = terms(query)
    evidence_terms = set().union(*(terms(item.title + " " + item.content) for item in items)) if items else set()
    aspect_coverage = (
        len(query_terms & evidence_terms) / len(query_terms) if query_terms else 0.0
    )
    relevance = max((item.rerank_score for item in items), default=0.0)
    if relevance <= 0 and items:
        relevance = max(item.fused_score for item in items)
    semantic_relevance = _semantic_relevance(items)
    provenance = (
        sum(item.provenance_score for item in items) / len(items) if items else 0.0
    )
    documents = {ref.document_id for item in items for ref in item.source_refs if ref.document_id > 0}
    types = {item.evidence_type for item in items}
    independent_support = min(1.0, 0.50 * min(2, len(documents)) / 2 + 0.50 * min(2, len(types)) / 2)
    graph_connectivity = 1.0 if any(
        item.evidence_type in {"relation", "community_report"}
        or item.relation_keys for item in items
    ) else (0.45 if items else 0.0)
    conflict = _conflict_penalty(items)
    score = max(0.0, min(1.0,
        0.30 * aspect_coverage
        + 0.25 * relevance
        + 0.20 * provenance
        + 0.15 * independent_support
        + 0.10 * graph_connectivity
        - conflict
    ))
    reasons: list[str] = []
    if not items:
        reasons.append("没有检索到可引用证据")
    if items and provenance < 0.75:
        reasons.append("部分候选缺少可核验来源")
    strong_semantic_match = bool(
        (reranker_used_model and relevance >= 0.50)
        or semantic_relevance >= 0.55
    )
    if aspect_coverage < 0.25 and not strong_semantic_match:
        reasons.append("证据对问题关键方面的覆盖不足")
    jointly_weak_match = bool(
        items
        and reranker_used_model
        and relevance < Config.agent_weak_rerank_threshold
        and semantic_relevance < Config.agent_weak_semantic_threshold
        and aspect_coverage < Config.agent_weak_aspect_threshold
    )
    if jointly_weak_match:
        reasons.append("多路检索均未找到与问题充分相关的证据")
    if conflict:
        reasons.append("检索证据之间存在明显冲突")
    if conflict:
        state = "conflict"
    elif (
        not items
        or not any(item.provenance_valid for item in items)
        or jointly_weak_match
        or score < 0.28
    ):
        state = "insufficient"
    elif score < 0.42:
        state = "retryable"
    elif len(query_terms) <= 1 and aspect_coverage < 0.8:
        state = "ambiguous"
    else:
        state = "sufficient"
    return SufficiencyResult(
        state=state, score=score, aspect_coverage=aspect_coverage,
        relevance=max(0.0, min(1.0, relevance)),
        semantic_relevance=max(0.0, min(1.0, semantic_relevance)),
        reranker_used_model=bool(reranker_used_model), provenance=provenance,
        independent_support=independent_support,
        graph_connectivity=graph_connectivity,
        conflict_penalty=conflict, reasons=tuple(reasons),
    )
