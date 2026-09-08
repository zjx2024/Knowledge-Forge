"""Deterministic project-level context for high-recall relation proposals.

This module borrows the useful *global context* idea from GraphRAG without
importing any GraphRAG runtime or treating summaries as facts.  The returned
payload is a bounded, immutable prior: it can help a model choose which local
occurrences deserve attention, but it is never sufficient evidence for
publishing a relation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Mapping, Sequence

from .quality import canonical_digest


MACRO_CONTEXT_VERSION = "enhanced-shadow-macro-context-v2-focus-only"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _row(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_name(
    canonical_key: str,
    canonical_by_key: Mapping[str, Mapping[str, Any]],
) -> str:
    record = _row(canonical_by_key.get(canonical_key))
    return _text(record.get("name")) or canonical_key


def build_project_macro_context(
    *,
    documents: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    mentions: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    canonicals: Sequence[Mapping[str, Any]],
    max_entities: int = 48,
    max_peers: int = 4,
) -> dict[str, Any]:
    """Build a stable, bounded project-wide entity/co-occurrence prior.

    Counts are derived from the frozen entity assignment and source chunk
    universe.  Co-occurrence is counted at most once per source chunk so a
    repeated mention cannot dominate the global view.  No predicate or domain
    vocabulary is inferred here.
    """

    canonical_by_key = {
        _text(item.get("canonical_key")): _row(item)
        for item in canonicals
        if _text(item.get("canonical_key"))
    }
    assignment_by_mention = {
        _text(item.get("mention_key")): _row(item)
        for item in assignments
        if _text(item.get("mention_key"))
    }
    active_mentions: list[tuple[str, int, int]] = []
    for mention in mentions:
        mention_key = _text(mention.get("mention_key"))
        assignment = assignment_by_mention.get(mention_key)
        canonical_key = _text((assignment or {}).get("canonical_key"))
        if not canonical_key:
            continue
        try:
            document_id = int(mention.get("document_id") or assignment.get("document_id") or 0)
            chunk_id = int(mention.get("chunk_id") or assignment.get("chunk_id") or 0)
        except (TypeError, ValueError):
            continue
        if chunk_id <= 0:
            continue
        active_mentions.append((canonical_key, document_id, chunk_id))

    mention_count = Counter(item[0] for item in active_mentions)
    chunks_by_entity: dict[str, set[int]] = defaultdict(set)
    documents_by_entity: dict[str, set[int]] = defaultdict(set)
    entities_by_chunk: dict[int, set[str]] = defaultdict(set)
    for canonical_key, document_id, chunk_id in active_mentions:
        chunks_by_entity[canonical_key].add(chunk_id)
        if document_id > 0:
            documents_by_entity[canonical_key].add(document_id)
        entities_by_chunk[chunk_id].add(canonical_key)

    cooccurrence = Counter()
    for entity_ids in entities_by_chunk.values():
        for left, right in combinations(sorted(entity_ids), 2):
            cooccurrence[(left, right)] += 1

    def entity_sort_key(canonical_key: str) -> tuple[int, str, str]:
        return (
            -int(mention_count[canonical_key]),
            _canonical_name(canonical_key, canonical_by_key),
            canonical_key,
        )

    selected_keys = sorted(mention_count, key=entity_sort_key)[:max(1, int(max_entities))]
    selected_set = set(selected_keys)
    peer_map: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (left, right), count in cooccurrence.items():
        if left in selected_set:
            peer_map[left].append((right, int(count)))
        if right in selected_set:
            peer_map[right].append((left, int(count)))

    entities: list[dict[str, Any]] = []
    for canonical_key in selected_keys:
        record = canonical_by_key.get(canonical_key) or {}
        peers = sorted(
            peer_map.get(canonical_key, []),
            key=lambda item: (-item[1], _canonical_name(item[0], canonical_by_key), item[0]),
        )[:max(0, int(max_peers))]
        entities.append({
            "canonical_id": canonical_key,
            "name": _canonical_name(canonical_key, canonical_by_key),
            "type": _text(record.get("primary_type")) or "GenericEntity",
            "mention_count": int(mention_count[canonical_key]),
            "chunk_count": len(chunks_by_entity[canonical_key]),
            "document_count": len(documents_by_entity[canonical_key]),
            "cooccurring_entities": [
                {
                    "canonical_id": peer,
                    "name": _canonical_name(peer, canonical_by_key),
                    "chunk_count": count,
                }
                for peer, count in peers
            ],
        })

    payload = {
        "version": MACRO_CONTEXT_VERSION,
        "entity_count": len(mention_count),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "cooccurrence_pair_count": len(cooccurrence),
        "entities": entities,
    }
    return {
        **payload,
        "hash": canonical_digest(payload),
    }


def select_macro_context(
    macro_context: Mapping[str, Any] | None,
    canonical_ids: Sequence[str],
    *,
    max_entities: int = 16,
    max_peers: int = 0,
) -> dict[str, Any]:
    """Select only project statistics for entities in one local context.

    Global high-frequency entities and unrelated co-occurrence neighborhoods
    are useful for project navigation and query-time summaries, but they are
    actively harmful in a local fact-extraction prompt: they consume most of
    the input budget and can look like relation evidence despite a textual
    warning.  The extraction prior therefore contains only frozen focus
    entities, and peers are opt-in for non-extraction callers.
    """

    source = _row(macro_context)
    all_entities = [
        _row(item) for item in source.get("entities") or []
        if _text(item.get("canonical_id"))
    ]
    focus = {_text(value) for value in canonical_ids if _text(value)}
    focus_entities = [
        item for item in all_entities
        if _text(item.get("canonical_id")) in focus
    ]
    focus_entities.sort(key=lambda item: (
        0 if _text(item.get("canonical_id")) in focus else 1,
        -int(item.get("mention_count") or 0),
        _text(item.get("name")),
        _text(item.get("canonical_id")),
    ))
    selected: list[dict[str, Any]] = []
    for item in focus_entities:
        if len(selected) >= max(1, int(max_entities)):
            break
        canonical_id = _text(item.get("canonical_id"))
        peers = [
            _row(peer) for peer in item.get("cooccurring_entities") or []
        ][:max(0, int(max_peers))]
        selected.append({
            "canonical_id": canonical_id,
            "name": _text(item.get("name")),
            "type": _text(item.get("type")),
            "mention_count": int(item.get("mention_count") or 0),
            "chunk_count": int(item.get("chunk_count") or 0),
            "document_count": int(item.get("document_count") or 0),
            "cooccurring_entities": peers,
        })

    return {
        "version": _text(source.get("version")) or MACRO_CONTEXT_VERSION,
        "hash": _text(source.get("hash")),
        "focus_entity_ids": sorted(focus),
        "entities": selected,
        "project_entity_count": int(source.get("entity_count") or 0),
        "project_document_count": int(source.get("document_count") or 0),
        "project_chunk_count": int(source.get("chunk_count") or 0),
    }
