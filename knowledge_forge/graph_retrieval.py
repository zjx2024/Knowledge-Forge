"""Bounded, project-local graph context selection for Knowledge Forge search.

The production graph is immutable per ready snapshot.  This module contains
only deterministic ranking and traversal helpers; persistence and snapshot
freshness remain owned by ``database.py`` and ``service.py``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .entity_identity import normalize_name


GRAPH_RETRIEVAL_POLICY_VERSION = (
    "kf-local-graph-context-v3-semantic-fact-diversity"
)


def _relation_fact_signature(
    relation: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """Identify one semantic fact for bounded-context diversification.

    A pair of entities can have several independently supported relations.
    Treating the unordered endpoint pair alone as a duplicate key hides those
    facts whenever the context budget is smaller than the graph.  Only rows
    with the same endpoints, source relation meaning, and evidence are
    alternates; a different predicate/expression or evidence passage remains
    eligible for the primary ranked context.
    """

    endpoints = sorted((
        str(relation.get("source_canonical_key") or ""),
        str(relation.get("target_canonical_key") or ""),
    ))
    metadata = (
        relation.get("metadata")
        if isinstance(relation.get("metadata"), Mapping) else {}
    )

    def normalized(value: Any) -> str:
        return " ".join(str(value or "").split()).casefold()

    relation_meaning = normalized(
        metadata.get("display_relation_label")
        or metadata.get("source_relation_expression")
        or relation.get("relation_type")
    )
    evidence = normalized(
        relation.get("evidence_text") or relation.get("description")
    )
    return endpoints[0], endpoints[1], relation_meaning, evidence


def _query_surface_match(query: str, entity: Mapping[str, Any]) -> bool:
    query_key = normalize_name(query)
    if not query_key:
        return False
    surfaces = [entity.get("name"), *(entity.get("aliases") or ())]
    for surface in surfaces:
        surface_key = normalize_name(str(surface or ""))
        # Single-character CJK/pronoun matches are too ambiguous to make a
        # useful graph seed.  They can still be reached from another seed.
        if len(surface_key) >= 2 and surface_key in query_key:
            return True
    return False


def select_graph_seed_entities(
    query: str,
    entities: Iterable[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Select semantic graph entry points with exact query surfaces first.

    GraphRAG local search maps a bounded set of semantically relevant entities
    and then fans out through their relationships.  Knowledge Forge already
    computes lexical and embedding scores, so seed selection can stay fully
    deterministic and avoid an additional model call at query time.
    """

    selected = []
    for entity in entities:
        item = dict(entity)
        item["query_surface_match"] = _query_surface_match(query, item)
        selected.append(item)
    selected.sort(
        key=lambda item: (
            bool(item.get("query_surface_match")),
            float(item.get("score") or 0.0),
            float(item.get("lexical_score") or 0.0),
            float(item.get("dense_score") or 0.0),
            int(item.get("source_count") or 0),
            str(item.get("canonical_key") or item.get("id") or ""),
        ),
        reverse=True,
    )
    return selected[: max(1, min(20, int(limit)))]


def graph_neighbor_keys(
    relations: Iterable[Mapping[str, Any]],
    seed_keys: Iterable[str],
    *,
    limit: int = 24,
) -> list[str]:
    """Rank one-hop neighbors, favouring nodes shared by multiple seeds."""

    seeds = {str(value) for value in seed_keys if str(value)}
    links: dict[str, set[str]] = defaultdict(set)
    relevance: dict[str, float] = defaultdict(float)
    for relation in relations:
        source = str(relation.get("source_canonical_key") or "")
        target = str(relation.get("target_canonical_key") or "")
        if source in seeds and target and target not in seeds:
            links[target].add(source)
            relevance[target] = max(
                relevance[target], float(relation.get("lexical_score") or 0.0),
            )
        if target in seeds and source and source not in seeds:
            links[source].add(target)
            relevance[source] = max(
                relevance[source], float(relation.get("lexical_score") or 0.0),
            )
    ranked = sorted(
        links,
        key=lambda key: (len(links[key]), relevance[key], key),
        reverse=True,
    )
    return ranked[: max(1, min(50, int(limit)))]


def build_graph_entity_hops(
    seed_keys: Iterable[str],
    first_hop_relations: Iterable[Mapping[str, Any]],
    second_hop_relations: Iterable[Mapping[str, Any]] = (),
) -> dict[str, int]:
    """Derive shortest undirected distances for the bounded candidate graph."""

    hops = {str(value): 0 for value in seed_keys if str(value)}

    def expand(rows: Iterable[Mapping[str, Any]], maximum: int) -> None:
        # A small fixed-point pass handles rows whose database order does not
        # follow path order while keeping traversal strictly bounded.
        for _ in range(maximum + 1):
            changed = False
            for relation in rows:
                source = str(relation.get("source_canonical_key") or "")
                target = str(relation.get("target_canonical_key") or "")
                if source in hops and hops[source] < maximum and target not in hops:
                    hops[target] = hops[source] + 1
                    changed = True
                if target in hops and hops[target] < maximum and source not in hops:
                    hops[source] = hops[target] + 1
                    changed = True
            if not changed:
                break

    expand(list(first_hop_relations), 1)
    expand(list(second_hop_relations), 2)
    return hops


def rank_graph_relations(
    relations: Iterable[Mapping[str, Any]],
    *,
    seed_scores: Mapping[str, float],
    entity_hops: Mapping[str, int],
    limit: int,
) -> list[dict[str, Any]]:
    """Rank graph context using GraphRAG-style network locality and diversity.

    Relationships between mapped entities are preferred, followed by seed
    boundary edges and then two-hop expansion. One high-ranked row per
    semantic-fact signature is emitted before equivalent alternates. This
    prevents duplicate/reverse projections from consuming the context without
    suppressing distinct facts that happen to connect the same entity pair.
    """

    seeds = {str(key) for key in seed_scores}
    unique: dict[str, dict[str, Any]] = {}
    for relation in relations:
        item = dict(relation)
        key = str(item.get("id") or "")
        if key and key not in unique:
            unique[key] = item

    outside_links: dict[str, set[str]] = defaultdict(set)
    for item in unique.values():
        source = str(item.get("source_canonical_key") or "")
        target = str(item.get("target_canonical_key") or "")
        if source in seeds and target not in seeds:
            outside_links[target].add(source)
        if target in seeds and source not in seeds:
            outside_links[source].add(target)

    scope_rank = {
        "in_network": 4,
        "seed_boundary": 3,
        "expanded": 2,
        "text_evidence": 1,
    }
    ranked: list[dict[str, Any]] = []
    for item in unique.values():
        source = str(item.get("source_canonical_key") or "")
        target = str(item.get("target_canonical_key") or "")
        source_seed = source in seeds
        target_seed = target in seeds
        if source_seed and target_seed:
            scope = "in_network"
            network_score = 1.0
        elif source_seed or target_seed:
            scope = "seed_boundary"
            network_score = 0.88
        elif source in entity_hops or target in entity_hops:
            scope = "expanded"
            network_score = 0.66
        else:
            scope = "text_evidence"
            network_score = 0.0

        known_hops = [
            int(entity_hops[key]) for key in (source, target)
            if key in entity_hops
        ]
        graph_hop = 1 + min(known_hops) if known_hops else None
        graph_hop = min(2, graph_hop) if graph_hop is not None else None
        seed_relevance = max(
            float(seed_scores.get(source) or 0.0),
            float(seed_scores.get(target) or 0.0),
        )
        outside = target if source_seed and not target_seed else source
        shared_links = len(outside_links.get(outside, set()))
        shared_score = min(1.0, shared_links / 3.0)
        text_score = float(item.get("score") or 0.0)
        lexical_score = float(item.get("lexical_score") or 0.0)
        relation_surface_score = float(
            item.get("relation_surface_score") or 0.0
        )
        relation_semantic_score = float(
            item.get("relation_semantic_score") or 0.0
        )
        graph_score = (
            0.22 * network_score
            + 0.12 * seed_relevance
            + 0.36 * relation_semantic_score
            + 0.15 * relation_surface_score
            + 0.08 * text_score
            + 0.02 * lexical_score
            + 0.05 * shared_score
        )
        item.update({
            "text_score": text_score,
            "score": graph_score,
            "graph_score": graph_score,
            "graph_hop": graph_hop,
            "network_scope": scope,
            "shared_seed_links": shared_links,
            "retrieval_policy_version": GRAPH_RETRIEVAL_POLICY_VERSION,
            "_scope_rank": scope_rank[scope],
        })
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            float(item.get("graph_score") or 0.0),
            int(item["_scope_rank"]),
            int(item.get("shared_seed_links") or 0),
            int(item.get("evidence_count") or 0),
            str(item.get("id") or ""),
        ),
        reverse=True,
    )

    primary: list[dict[str, Any]] = []
    alternates: list[dict[str, Any]] = []
    seen_facts: set[tuple[str, str, str, str]] = set()
    for item in ranked:
        signature = _relation_fact_signature(item)
        target = primary if signature not in seen_facts else alternates
        target.append(item)
        seen_facts.add(signature)

    result = primary + alternates
    for item in result:
        item.pop("_scope_rank", None)
    return result[: max(1, min(50, int(limit)))]
