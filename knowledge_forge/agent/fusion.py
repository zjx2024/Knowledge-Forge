"""Deterministic candidate deduplication, RRF fusion and context selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..ollama import ConservativeTokenEstimator
from .contracts import EvidenceItem


RRF_K = 60
TYPE_WEIGHTS = {
    "chunk": 1.0,
    "entity": 0.8,
    "relation": 1.1,
    "community_report": 0.9,
    "native_context": 1.0,
    "derived_answer": 0.4,
}
TYPE_CONTEXT_LIMITS = {
    "chunk": 6,
    "entity": 3,
    "relation": 6,
    "community_report": 2,
    "native_context": 8,
    "derived_answer": 1,
}
DOCUMENT_CONTEXT_LIMIT = 6


def context_selection_capacity(
    items: list[EvidenceItem], *, requested_max_items: int,
) -> int:
    """Return the number of slots that can survive document diversity caps."""

    requested = max(1, int(requested_max_items))
    documents = {
        ref.document_id
        for item in items
        for ref in item.source_refs
        if ref.document_id > 0
    }
    if not documents:
        return min(requested, max(1, len(items)))
    return min(
        requested,
        max(1, len(documents) * DOCUMENT_CONTEXT_LIMIT),
        max(1, len(items)),
    )


def relevance_filtered_context_pool(
    items: list[EvidenceItem], *, reranker_used_model: bool,
) -> tuple[list[EvidenceItem], float]:
    """Drop obvious model-ranked noise before spending the prompt budget.

    The relative floor is intentionally very low so secondary/multi-hop facts
    remain available.  Filtering is disabled when the cross-encoder degraded
    or when even its best score is weak, because fallback/RRF scores are not
    calibrated absolute relevance.
    """

    if not items or not reranker_used_model:
        return items, 0.0
    strongest = max(item.rerank_score for item in items)
    if strongest < 0.10:
        return items, 0.0
    floor = max(0.001, strongest * 0.01)
    filtered = [item for item in items if item.rerank_score >= floor]
    return (filtered or items[:1]), floor


def _bounded_context(
    item: EvidenceItem, estimator: ConservativeTokenEstimator, max_tokens: int,
) -> tuple[str, int]:
    """Fit a candidate without mutating its persisted/auditable full text."""

    max_tokens = max(64, int(max_tokens))
    title_cost = estimator.estimate(item.title + "\n") + 48
    available = max(16, max_tokens - title_cost)
    content = item.content
    if estimator.estimate(content) <= available:
        return content, title_cost + estimator.estimate(content)
    low, high = 1, len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if estimator.estimate(content[:middle]) <= available:
            low = middle
        else:
            high = middle - 1
    excerpt = content[:low].rstrip()
    if low < len(content):
        excerpt += "…"
    return excerpt, title_cost + estimator.estimate(excerpt)


def deduplicate_evidence(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    merged: dict[str, EvidenceItem] = {}
    for item in items:
        stable_key = str(item.metadata.get("stable_key") or item.evidence_id)
        existing = merged.get(stable_key)
        if existing is None:
            merged[stable_key] = item
            continue
        existing.retrieval_channels = list(dict.fromkeys(
            [*existing.retrieval_channels, *item.retrieval_channels]
        ))
        for channel, score in item.raw_scores.items():
            existing.raw_scores[channel] = max(
                existing.raw_scores.get(channel, float("-inf")), float(score),
            )
        source_keys = {
            (ref.document_id, ref.revision_id, ref.pages, ref.section_title)
            for ref in existing.source_refs
        }
        for ref in item.source_refs:
            key = (ref.document_id, ref.revision_id, ref.pages, ref.section_title)
            if key not in source_keys:
                existing.source_refs.append(ref)
                source_keys.add(key)
        existing.entity_keys = list(dict.fromkeys([*existing.entity_keys, *item.entity_keys]))
        existing.relation_keys = list(dict.fromkeys([*existing.relation_keys, *item.relation_keys]))
        existing.provenance_valid = bool(
            existing.source_refs and all(ref.valid for ref in existing.source_refs)
        )
        existing.provenance_score = max(existing.provenance_score, item.provenance_score)
    return list(merged.values())


def reciprocal_rank_fuse(items: list[EvidenceItem], *, k: int = RRF_K) -> list[EvidenceItem]:
    channel_items: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in items:
        for channel in item.retrieval_channels:
            channel_items[channel].append(item)
    fused = defaultdict(float)
    for channel, candidates in channel_items.items():
        ranked = sorted(
            candidates,
            key=lambda item: (
                float(item.raw_scores.get(channel, 0.0)),
                item.provenance_score,
                item.evidence_id,
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked, 1):
            fused[item.evidence_id] += TYPE_WEIGHTS[item.evidence_type] / (max(1, k) + rank)
    maximum = max(fused.values(), default=1.0)
    for item in items:
        item.fused_score = fused[item.evidence_id] / maximum if maximum else 0.0
    return sorted(
        items,
        key=lambda item: (item.fused_score, item.provenance_score, item.evidence_id),
        reverse=True,
    )


def apply_final_ranking(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Combine model relevance with fusion, provenance and source diversity."""

    seen_documents: set[int] = set()
    ranked: list[tuple[float, EvidenceItem]] = []
    for item in sorted(
        items,
        key=lambda value: (
            0.50 * value.rerank_score
            + 0.25 * value.fused_score
            + 0.15 * value.provenance_score,
            value.evidence_id,
        ),
        reverse=True,
    ):
        documents = {ref.document_id for ref in item.source_refs if ref.document_id > 0}
        diversity = 1.0 if documents - seen_documents else 0.25
        score = (
            0.50 * max(0.0, min(1.0, item.rerank_score))
            + 0.25 * max(0.0, min(1.0, item.fused_score))
            + 0.15 * max(0.0, min(1.0, item.provenance_score))
            + 0.10 * diversity
        )
        item.metadata["final_score"] = score
        ranked.append((score, item))
        seen_documents.update(documents)
    ranked.sort(key=lambda pair: (pair[0], pair[1].evidence_id), reverse=True)
    for rank, (_score, item) in enumerate(ranked, 1):
        item.rank = rank
    return [item for _score, item in ranked]


def select_context(
    items: list[EvidenceItem], *, token_budget: int, max_items: int,
) -> list[EvidenceItem]:
    """Greedily maximize relevance while penalising duplicate source material."""

    estimator = ConservativeTokenEstimator()
    selected: list[EvidenceItem] = []
    used_tokens = 0
    covered_documents: set[int] = set()
    covered_entities: set[str] = set()
    document_counts: dict[int, int] = defaultdict(int)
    type_counts: dict[str, int] = defaultdict(int)
    remaining = list(items)
    while remaining and len(selected) < max(1, max_items):
        choices: list[tuple[float, int, EvidenceItem, str]] = []
        remaining_budget = max(0, token_budget - used_tokens)
        per_item_budget = min(
            remaining_budget,
            max(256, token_budget // max(2, min(max_items, 6))),
        )
        for item in remaining:
            documents = {ref.document_id for ref in item.source_refs if ref.document_id > 0}
            if type_counts[item.evidence_type] >= TYPE_CONTEXT_LIMITS[item.evidence_type]:
                continue
            if documents and all(
                document_counts[document_id] >= DOCUMENT_CONTEXT_LIMIT
                for document_id in documents
            ):
                continue
            excerpt, tokens = _bounded_context(item, estimator, per_item_budget)
            if used_tokens + tokens > max(1, token_budget):
                continue
            novelty = 0.0
            if documents - covered_documents:
                novelty += 0.12
            if set(item.entity_keys) - covered_entities:
                novelty += 0.06
            redundancy = 0.08 if documents and documents <= covered_documents else 0.0
            utility = float(item.metadata.get("final_score") or 0.0) + novelty - redundancy
            # Context length is already bounded twice (per item and globally).
            # Dividing utility by sqrt(tokens) made a short, weak fragment beat
            # a highly relevant report or relationship, then document limits
            # could lock the useful evidence out entirely.  Keep a small cost
            # penalty for compactness without allowing length to dominate
            # relevance.
            cost_fraction = min(1.0, tokens / max(1.0, per_item_budget))
            selection_score = utility - 0.04 * cost_fraction
            choices.append((selection_score, utility, -tokens, item, excerpt))
        if not choices:
            break
        choices.sort(
            key=lambda value: (value[0], value[1], value[2], value[3].evidence_id),
            reverse=True,
        )
        _selection_score, _utility, negative_tokens, item, excerpt = choices[0]
        tokens = -negative_tokens
        item.selected = True
        item.metadata["context_content"] = excerpt
        item.metadata["context_token_cost"] = tokens
        selected.append(item)
        used_tokens += tokens
        item_documents = {
            ref.document_id for ref in item.source_refs if ref.document_id > 0
        }
        covered_documents.update(item_documents)
        for document_id in item_documents:
            document_counts[document_id] += 1
        type_counts[item.evidence_type] += 1
        covered_entities.update(item.entity_keys)
        remaining.remove(item)
    return selected
