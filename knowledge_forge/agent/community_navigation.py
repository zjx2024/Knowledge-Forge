"""Bounded community-guided recall for the Knowledge Forge agent.

Community aggregates and semantic topic links are routing metadata, never
answer evidence.  A route is useful only when it resolves back to a verified
relation assertion in the current immutable graph snapshot.  This module
therefore plans with lightweight community topology, then re-reads every
selected relation through the normal evidence-gated database query.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from ..vector_store import lexical_similarity


COMMUNITY_NAVIGATION_POLICY_VERSION = "kf-agent-community-navigation-v1"
_MIN_ENTITY_SEED_SCORE = 0.48
_MIN_REPORT_SEED_SCORE = 0.50
_SPARSE_RELATION_THRESHOLD = 4
_MAX_SEED_COMMUNITIES = 3


def _text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _relation_text(row: dict[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    return " ".join(filter(None, (
        _text(row.get("source_name")),
        _text(metadata.get("display_relation_label")),
        _text(metadata.get("source_relation_expression")),
        _text(row.get("relation_type")),
        _text(row.get("target_name")),
        _text(row.get("evidence_text")),
    )))


def _base_diagnostics(status: str) -> dict[str, Any]:
    return {
        "policy_version": COMMUNITY_NAVIGATION_POLICY_VERSION,
        "status": status,
        "trigger_reasons": [],
        "seed_communities": [],
        "selected_links": [],
        "candidate_relation_count": 0,
        "verified_relation_count": 0,
        "new_relation_count": 0,
        "factual_links_used": 0,
        "topic_links_used": 0,
        # A permanent audit invariant: navigation metadata may never be
        # serialized as an EvidenceItem.
        "synthetic_evidence_count": 0,
    }


class KnowledgeForgeCommunityNavigator:
    """Resolve community routes to a small set of verified relation rows."""

    def __init__(self, database: Any) -> None:
        self.database = database

    @staticmethod
    def _leaf_communities(
        community_run: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        communities = {
            _text(item.get("key")): dict(item)
            for item in community_run.get("communities") or []
            if isinstance(item, dict)
            and _text(item.get("key"))
            and int(item.get("level") or 0) == 0
        }
        community_by_entity: dict[str, str] = {}
        for key, community in sorted(communities.items()):
            for entity_key in community.get("entity_keys") or []:
                entity_key = _text(entity_key)
                if entity_key:
                    community_by_entity.setdefault(entity_key, key)
        return communities, community_by_entity

    @staticmethod
    def _seed_communities(
        payload: dict[str, Any], *,
        communities: dict[str, dict[str, Any]],
        community_by_entity: dict[str, str],
        maximum: int,
    ) -> list[dict[str, Any]]:
        graph_state = payload.get("graph_retrieval")
        graph_state = graph_state if isinstance(graph_state, dict) else {}
        seed_rows = [
            dict(item) for item in graph_state.get("seed_entities") or []
            if isinstance(item, dict)
        ]
        if not seed_rows:
            seed_rows = [
                {
                    "canonical_key": item.get("canonical_key"),
                    "score": item.get("score"),
                    "query_surface_match": False,
                }
                for item in (payload.get("entities") or [])[:_MAX_SEED_COMMUNITIES]
                if isinstance(item, dict)
            ]

        exact_rows = [
            item for item in seed_rows if bool(item.get("query_surface_match"))
        ]
        if exact_rows:
            # If the question names entities explicitly, unrelated semantic
            # neighbours must not turn into extra community seeds.  A query
            # naming two entities still retains both exact communities.
            seed_rows = exact_rows
        else:
            strongest = max(
                (float(item.get("score") or 0.0) for item in seed_rows),
                default=0.0,
            )
            relative_floor = max(_MIN_ENTITY_SEED_SCORE, strongest - 0.06)
            seed_rows = [
                item for item in seed_rows
                if float(item.get("score") or 0.0) >= relative_floor
            ]

        selected: dict[str, dict[str, Any]] = {}
        for item in seed_rows:
            entity_key = _text(
                item.get("canonical_key") or item.get("id")
            )
            community_key = community_by_entity.get(entity_key, "")
            if community_key not in communities:
                continue
            score = max(0.0, min(1.0, float(item.get("score") or 0.0)))
            exact = bool(item.get("query_surface_match"))
            if not exact and score < _MIN_ENTITY_SEED_SCORE:
                continue
            current = selected.setdefault(community_key, {
                "community_key": community_key,
                "score": 0.0,
                "sources": [],
                "entity_keys": [],
            })
            current["score"] = max(float(current["score"]), 1.0 if exact else score)
            if "entity" not in current["sources"]:
                current["sources"].append("entity")
            if entity_key and entity_key not in current["entity_keys"]:
                current["entity_keys"].append(entity_key)

        # Community reports are a useful macro-level entry point when entity
        # mapping finds no defensible seed.  They do not override a concrete
        # entity seed and must themselves clear an absolute relevance floor.
        if not selected:
            reports = sorted(
                [
                    dict(item) for item in payload.get("community_reports") or []
                    if isinstance(item, dict)
                ],
                key=lambda item: (
                    float(item.get("score") or 0.0),
                    float(item.get("dense_score") or 0.0),
                    _text(item.get("community_key")),
                ),
                reverse=True,
            )
            for item in reports:
                community_key = _text(item.get("community_key"))
                score = max(
                    0.0,
                    min(1.0, float(item.get("score") or 0.0)),
                )
                if community_key not in communities or score < _MIN_REPORT_SEED_SCORE:
                    continue
                selected[community_key] = {
                    "community_key": community_key,
                    "score": score,
                    "sources": ["community_report"],
                    "entity_keys": [],
                }
                break

        return sorted(
            selected.values(),
            key=lambda item: (
                float(item["score"]), _text(item["community_key"]),
            ),
            reverse=True,
        )[: max(1, min(_MAX_SEED_COMMUNITIES, int(maximum)))]

    @staticmethod
    def _association_candidates(
        community_run: dict[str, Any], seeds: list[dict[str, Any]],
        communities: dict[str, dict[str, Any]], query: str,
    ) -> list[dict[str, Any]]:
        metadata = community_run.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        associations = [
            dict(item)
            for item in metadata.get("community_topic_associations") or []
            if isinstance(item, dict)
        ]
        result: list[dict[str, Any]] = []
        for seed in seeds:
            seed_key = _text(seed.get("community_key"))
            for association in associations:
                left = _text(association.get("source_community_key"))
                right = _text(association.get("target_community_key"))
                if seed_key == left:
                    target = right
                    target_match_field = "target_entity_key"
                    target_name_field = "target_entity_name"
                elif seed_key == right:
                    target = left
                    target_match_field = "source_entity_key"
                    target_name_field = "source_entity_name"
                else:
                    continue
                if target not in communities or target == seed_key:
                    continue
                matches = [
                    dict(item)
                    for item in association.get("representative_entity_matches") or []
                    if isinstance(item, dict)
                ]
                representative_keys = list(dict.fromkeys(
                    _text(item.get(target_match_field)) for item in matches
                    if _text(item.get(target_match_field))
                ))
                representative_names = " ".join(
                    _text(item.get(target_name_field)) for item in matches
                    if _text(item.get(target_name_field))
                )
                association_score = max(
                    0.0, min(1.0, float(association.get("score") or 0.0)),
                )
                lexical = lexical_similarity(query, representative_names)
                score = min(0.90,
                    0.45 * float(seed.get("score") or 0.0)
                    + 0.40 * association_score
                    + 0.15 * lexical
                )
                result.append({
                    "id": _text(association.get("id"))
                    or f"topic:{seed_key}:{target}",
                    "kind": "topic_navigation",
                    "seed_community_key": seed_key,
                    "target_community_key": target,
                    "score": score,
                    "representative_entity_keys": representative_keys,
                    "association_score": association_score,
                    "rows": [],
                })
        return result

    def expand(
        self, *, project_id: int, graph_generation: int,
        namespace_id: str, relation_run_id: str,
        community_run_id: str, community_input_hash: str,
        query: str, payload: dict[str, Any],
        max_communities: int, max_relations: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        diagnostics = _base_diagnostics("not_available")
        if not community_run_id or not community_input_hash:
            diagnostics["reason"] = "current_snapshot_has_no_community_run"
            return [], diagnostics
        community_run = self.database.get_knowledge_forge_community_run(
            int(project_id), int(graph_generation), str(namespace_id),
            input_hash=str(community_input_hash),
        )
        if (
            not community_run
            or _text(community_run.get("run_id")) != _text(community_run_id)
        ):
            diagnostics["reason"] = "community_run_is_not_current"
            return [], diagnostics

        communities, community_by_entity = self._leaf_communities(community_run)
        if not communities:
            diagnostics["reason"] = "community_run_has_no_leaf_communities"
            return [], diagnostics
        seeds = self._seed_communities(
            payload, communities=communities,
            community_by_entity=community_by_entity,
            maximum=max_communities,
        )
        diagnostics["seed_communities"] = [
            {
                "community_key": item["community_key"],
                "score": round(float(item["score"]), 6),
                "sources": list(item["sources"]),
            }
            for item in seeds
        ]
        if not seeds:
            diagnostics["status"] = "not_triggered"
            diagnostics["reason"] = "no_reliable_community_seed"
            return [], diagnostics

        relation_count = len(payload.get("relations") or [])
        trigger_reasons: list[str] = []
        if len(seeds) > 1:
            trigger_reasons.append("multiple_seed_communities")
        if relation_count < _SPARSE_RELATION_THRESHOLD:
            trigger_reasons.append("sparse_relation_recall")
        if any("community_report" in item["sources"] for item in seeds):
            trigger_reasons.append("macro_report_seed")
        diagnostics["trigger_reasons"] = trigger_reasons
        if not trigger_reasons:
            diagnostics["status"] = "not_triggered"
            diagnostics["reason"] = "baseline_relation_recall_is_sufficient"
            return [], diagnostics

        loader = getattr(
            self.database,
            "list_agent_knowledge_forge_community_relation_candidates",
            None,
        )
        if not callable(loader):
            diagnostics["reason"] = "community_relation_index_is_unavailable"
            return [], diagnostics

        factual_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for seed in seeds:
            seed_key = _text(seed["community_key"])
            members = [
                _text(value) for value in communities[seed_key].get("entity_keys") or []
                if _text(value)
            ]
            rows = loader(
                namespace_id, relation_run_id, members,
                scope="cross", limit=max(40, int(max_relations) * 12),
            )
            for raw in rows:
                row = dict(raw)
                source_community = community_by_entity.get(
                    _text(row.get("source_canonical_key")), "",
                )
                target_community = community_by_entity.get(
                    _text(row.get("target_canonical_key")), "",
                )
                if (
                    not source_community or not target_community
                    or source_community == target_community
                    or seed_key not in {source_community, target_community}
                ):
                    continue
                other = (
                    target_community
                    if source_community == seed_key else source_community
                )
                pair = tuple(sorted((seed_key, other)))
                candidate = factual_by_pair.setdefault(pair, {
                    "id": f"fact:{pair[0]}:{pair[1]}",
                    "kind": "factual_bridge",
                    "seed_community_key": seed_key,
                    "target_community_key": other,
                    "seed_score": float(seed["score"]),
                    "score": 0.0,
                    "rows": [],
                })
                candidate["seed_score"] = max(
                    float(candidate["seed_score"]), float(seed["score"]),
                )
                candidate["rows"].append(row)

        factual_candidates: list[dict[str, Any]] = []
        seed_keys = {_text(item["community_key"]) for item in seeds}
        for candidate in factual_by_pair.values():
            rows = candidate["rows"]
            best_lexical = max(
                (lexical_similarity(query, _relation_text(row)) for row in rows),
                default=0.0,
            )
            strength = min(1.0, math.log1p(len(rows)) / math.log1p(8))
            target_bonus = (
                0.10 if candidate["target_community_key"] in seed_keys else 0.0
            )
            candidate["score"] = min(1.0,
                0.50 * float(candidate["seed_score"])
                + 0.25 * strength + 0.15 * best_lexical + target_bonus
            )
            factual_candidates.append(candidate)

        factual_pairs = {
            tuple(sorted((
                _text(item["seed_community_key"]),
                _text(item["target_community_key"]),
            )))
            for item in factual_candidates
        }
        topic_candidates = [
            item for item in self._association_candidates(
                community_run, seeds, communities, query,
            )
            if tuple(sorted((
                _text(item["seed_community_key"]),
                _text(item["target_community_key"]),
            ))) not in factual_pairs
        ]
        link_candidates = sorted(
            [*factual_candidates, *topic_candidates],
            key=lambda item: (
                _text(item.get("target_community_key")) in seed_keys,
                item.get("kind") == "factual_bridge",
                float(item.get("score") or 0.0),
                _text(item.get("id")),
            ),
            reverse=True,
        )

        selected_links: list[dict[str, Any]] = []
        selected_pairs: set[tuple[str, str]] = set()
        selected_targets: set[str] = set()
        for candidate in link_candidates:
            pair = tuple(sorted((
                _text(candidate["seed_community_key"]),
                _text(candidate["target_community_key"]),
            )))
            target = _text(candidate["target_community_key"])
            if pair in selected_pairs:
                continue
            if (
                target not in seed_keys
                and target not in selected_targets
                and len(selected_targets) >= max(1, int(max_communities))
            ):
                continue
            selected_links.append(candidate)
            selected_pairs.add(pair)
            if target not in seed_keys:
                selected_targets.add(target)
            if len(selected_links) >= max(1, int(max_communities)):
                break

        relation_routes: list[dict[str, Any]] = []
        for link in selected_links:
            kind = _text(link["kind"])
            if kind == "factual_bridge":
                rows = list(link.get("rows") or [])
                representative_keys: set[str] = set()
            else:
                target = _text(link["target_community_key"])
                members = [
                    _text(value)
                    for value in communities[target].get("entity_keys") or []
                    if _text(value)
                ]
                rows = loader(
                    namespace_id, relation_run_id, members,
                    scope="internal", limit=max(40, int(max_relations) * 12),
                )
                representative_keys = {
                    _text(value)
                    for value in link.get("representative_entity_keys") or []
                    if _text(value)
                }
            for row in rows:
                relation_key = _text(row.get("id") or row.get("relation_key"))
                if not relation_key:
                    continue
                lexical = lexical_similarity(query, _relation_text(row))
                representative_bonus = 0.0
                if representative_keys & {
                    _text(row.get("source_canonical_key")),
                    _text(row.get("target_canonical_key")),
                }:
                    representative_bonus = 0.12
                route_score = min(1.0,
                    0.76 * float(link.get("score") or 0.0)
                    + 0.24 * lexical + representative_bonus
                )
                relation_routes.append({
                    "relation_key": relation_key,
                    "score": route_score,
                    "kind": kind,
                    "link_id": _text(link["id"]),
                    "seed_community_key": _text(link["seed_community_key"]),
                    "target_community_key": _text(link["target_community_key"]),
                })

        # Keep one fact per selected link before filling the remaining budget.
        ordered_routes = sorted(
            relation_routes,
            key=lambda item: (
                float(item["score"]),
                item["kind"] == "factual_bridge",
                item["relation_key"],
            ),
            reverse=True,
        )
        selected_routes: list[dict[str, Any]] = []
        seen_relations: set[str] = set()
        covered_links: set[str] = set()
        for require_new_link in (True, False):
            for route in ordered_routes:
                if route["relation_key"] in seen_relations:
                    continue
                is_new_link = route["link_id"] not in covered_links
                if require_new_link != is_new_link:
                    continue
                selected_routes.append(route)
                seen_relations.add(route["relation_key"])
                covered_links.add(route["link_id"])
                if len(selected_routes) >= max(1, int(max_relations)):
                    break
            if len(selected_routes) >= max(1, int(max_relations)):
                break

        diagnostics["candidate_relation_count"] = len(seen_relations)
        if not selected_routes:
            diagnostics["status"] = "not_triggered"
            diagnostics["reason"] = "selected_communities_have_no_verified_fact_candidates"
            return [], diagnostics

        exact_rows = self.database.search_knowledge_forge_relations(
            namespace_id, relation_run_id, query,
            relation_keys=[item["relation_key"] for item in selected_routes],
            limit=len(selected_routes),
        )
        exact_by_id = {
            _text(item.get("id")): dict(item)
            for item in exact_rows if _text(item.get("id"))
        }
        baseline_ids = {
            _text(item.get("id")) for item in payload.get("relations") or []
            if isinstance(item, dict) and _text(item.get("id"))
        }
        verified: list[dict[str, Any]] = []
        used_link_ids: set[str] = set()
        for route in selected_routes:
            row = exact_by_id.get(route["relation_key"])
            if row is None:
                continue
            row["community_navigation_score"] = float(route["score"])
            row["community_navigation"] = {
                key: value for key, value in route.items()
                if key != "score"
            }
            row["score"] = max(
                float(row.get("score") or 0.0), float(route["score"]),
            )
            verified.append(row)
            used_link_ids.add(route["link_id"])

        used_links = [
            item for item in selected_links if _text(item["id"]) in used_link_ids
        ]
        diagnostics.update({
            "status": "expanded" if verified else "not_triggered",
            "reason": "verified_relations_added" if verified
            else "candidate_relations_failed_evidence_gate",
            "selected_links": [{
                "id": _text(item["id"]),
                "kind": _text(item["kind"]),
                "seed_community_key": _text(item["seed_community_key"]),
                "target_community_key": _text(item["target_community_key"]),
                "score": round(float(item.get("score") or 0.0), 6),
            } for item in used_links],
            "verified_relation_count": len(verified),
            "new_relation_count": sum(
                _text(item.get("id")) not in baseline_ids for item in verified
            ),
            "factual_links_used": sum(
                _text(item.get("kind")) == "factual_bridge" for item in used_links
            ),
            "topic_links_used": sum(
                _text(item.get("kind")) == "topic_navigation" for item in used_links
            ),
        })
        return verified, diagnostics
