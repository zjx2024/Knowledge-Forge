"""Strategy-neutral, bounded graph exploration views.

The browser never receives an unbounded full graph by default.  Both
Knowledge Forge and GraphRAG adapters normalize into the same node/edge/group
contract, then this module creates a community overview, a bounded community
subgraph, an entity neighbourhood, or explicit search results.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


# A graph scene is an exploration surface, not a database dump.  These bounds
# keep an initial scene legible on an ordinary laptop display while every
# group remains available through drill-down and search.
_OVERVIEW_NODE_CAP = 42
# The explicit "all relationship communities" mode is allowed to be denser
# than the orientation map.  It remains bounded for pathological projects, but
# ordinary projects can see every community that contains at least one fact.
_OVERVIEW_ALL_RELATION_COMMUNITY_NODE_CAP = 400
_OVERVIEW_BACKBONE_EDGE_CAP = 32
_RELATION_SCENE_NODE_CAP = 48
_RELATION_SCENE_EDGE_CAP = 60
_COMMUNITY_FOCUS_NODE_CAP = 120
_COMMUNITY_FOCUS_EDGE_CAP = 190
_COMMUNITY_EXPANDED_NODE_CAP = 320
_COMMUNITY_EXPANDED_EDGE_CAP = 560
_COMMUNITY_FULL_NODE_CAP = 2_000
_COMMUNITY_FULL_EDGE_CAP = 4_000
_ENTITY_NEIGHBOURHOOD_NODE_CAP = 120
_ENTITY_NEIGHBOURHOOD_EDGE_CAP = 160


def _text(value: object) -> str:
    return str(value or "").strip()


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return float(default)


def _normalized(payload: dict[str, Any]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    nodes = [dict(item) for item in payload.get("nodes") or [] if isinstance(item, dict)]
    edges = [dict(item) for item in payload.get("edges") or [] if isinstance(item, dict)]
    groups = [dict(item) for item in payload.get("groups") or [] if isinstance(item, dict)]
    node_ids = {_text(item.get("id")) for item in nodes}
    edges = [
        item for item in edges
        if _text(item.get("source")) in node_ids
        and _text(item.get("target")) in node_ids
    ]
    return nodes, edges, groups


def _group_index(
    nodes: list[dict[str, Any]], groups: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    by_key = {
        _text(item.get("key")): item
        for item in groups if _text(item.get("key"))
    }
    entity_group: dict[str, str] = {}
    for node in nodes:
        node_id = _text(node.get("id"))
        key = _text(node.get("group")) or _text(node.get("type_id")) or "unassigned"
        entity_group[node_id] = key
        by_key.setdefault(key, {
            "key": key,
            "label": _text(node.get("type")) or "未分组实体",
            "summary": "按实体类别临时分组；当前图谱尚未生成社区结果。",
            "description": "按实体类别临时分组；当前图谱尚未生成社区结果。",
            "description_source": "entity_type_fallback",
            "size": 0,
            "relationship_count": 0,
        })
    counts: dict[str, int] = defaultdict(int)
    for key in entity_group.values():
        counts[key] += 1
    for key, count in counts.items():
        by_key[key]["size"] = max(int(_number(by_key[key].get("size"))), count)
    return by_key, entity_group


def _degrees(edges: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for edge in edges:
        result[_text(edge.get("source"))] += 1
        result[_text(edge.get("target"))] += 1
    return result


def _edge_rank(
    edge: dict[str, Any], degrees: dict[str, int],
) -> tuple[float, float, float, str, str, str]:
    """Rank an edge by its structural usefulness to an exploration scene."""

    source = _text(edge.get("source"))
    target = _text(edge.get("target"))
    source_degree = degrees.get(source, 0)
    target_degree = degrees.get(target, 0)
    return (
        -float(max(source_degree, target_degree)),
        -float(source_degree + target_degree),
        -_number(edge.get("weight"), 1),
        _text(edge.get("source_title")) or source,
        _text(edge.get("target_title")) or target,
        _text(edge.get("id")),
    )


def _scene_subset(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *,
    node_limit: int, edge_limit: int, max_visible_fanout: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose a deterministic, structurally useful bounded entity scene.

    We seed the most connected entities and then grow through structural links.
    That retains inspectable neighbourhoods without using domain terms or
    predicate-specific rules.
    """

    if not nodes:
        return [], []
    by_id = {_text(item.get("id")): item for item in nodes}
    degrees = _degrees(edges)
    node_limit = max(1, min(int(node_limit), len(nodes)))
    edge_limit = max(1, int(edge_limit))
    if len(nodes) <= node_limit:
        selected_ids = set(by_id)
    else:
        ranked_nodes = sorted(
            by_id,
            key=lambda key: (
                -degrees.get(key, _number(by_id[key].get("degree"))),
                _text(by_id[key].get("title")), key,
            ),
        )
        # Keep several independent entry points before growing links; without
        # this one dense component can monopolize the initial graph scene.
        seed_count = min(node_limit, max(8, node_limit // 5))
        selected_ids = set(ranked_nodes[:seed_count])
        for edge in sorted(edges, key=lambda item: _edge_rank(item, degrees)):
            source = _text(edge.get("source"))
            target = _text(edge.get("target"))
            if source not in by_id or target not in by_id:
                continue
            additions = {source, target}.difference(selected_ids)
            if len(selected_ids) + len(additions) > node_limit:
                continue
            selected_ids.update(additions)
            if len(selected_ids) >= node_limit:
                break
        if len(selected_ids) < node_limit:
            selected_ids.update(ranked_nodes[:node_limit])

    candidate_edges = [
        item for item in edges
        if _text(item.get("source")) in selected_ids
        and _text(item.get("target")) in selected_ids
    ]
    candidate_edges.sort(key=lambda item: _edge_rank(item, degrees))
    # Cap the number of visible links touching any one entity.  A few hubs
    # otherwise obscure every neighbouring relation in a force-directed scene.
    # The scene is intentionally a representative map; the inspector and
    # entity-neighbour view retain the complete auditable neighbourhood.
    fanout_cap = max(5, min(14, int(edge_limit ** 0.5)))
    if max_visible_fanout is not None:
        fanout_cap = max(5, int(max_visible_fanout))
    selected_edges: list[dict[str, Any]] = []
    visible_incident: dict[str, int] = defaultdict(int)
    covered: set[str] = set()
    selected_edge_ids: set[str] = set()
    for edge in candidate_edges:
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        introduces_node = source not in covered or target not in covered
        if not introduces_node:
            continue
        if visible_incident[source] >= fanout_cap or visible_incident[target] >= fanout_cap:
            continue
        selected_edges.append(edge)
        selected_edge_ids.add(_text(edge.get("id")))
        visible_incident[source] += 1
        visible_incident[target] += 1
        covered.update((source, target))
        if len(selected_edges) >= edge_limit:
            break
    for edge in candidate_edges:
        if _text(edge.get("id")) in selected_edge_ids:
            continue
        source = _text(edge.get("source"))
        target = _text(edge.get("target"))
        if visible_incident[source] >= fanout_cap or visible_incident[target] >= fanout_cap:
            continue
        selected_edges.append(edge)
        visible_incident[source] += 1
        visible_incident[target] += 1
        if len(selected_edges) >= edge_limit:
            break
    # A scene must not contain unconnected decoration: it consumes label and
    # layout budget without explaining any visible fact.  Nodes omitted here
    # remain available through community selection, search and neighbourhood
    # drill-down; the response reports the visible count separately.
    visible_ids = {
        endpoint
        for edge in selected_edges
        for endpoint in (_text(edge.get("source")), _text(edge.get("target")))
    }
    if not visible_ids and len(nodes) <= node_limit:
        # A relationship-free community is still a valid result, but display
        # one deterministic entity rather than an arbitrary cloud of dots.
        ranked_ids = sorted(
            by_id,
            key=lambda key: (-degrees.get(key, _number(by_id[key].get("degree"))), key),
        )
        visible_ids = set(ranked_ids[:1])
    selected_nodes = [item for item in nodes if _text(item.get("id")) in visible_ids]
    return selected_nodes, selected_edges


def _group_rank(
    group: dict[str, Any], cross_degree: int = 0,
) -> tuple[int, int, int, str, str]:
    return (
        -int(_number(group.get("relationship_count"))),
        -int(cross_degree),
        -int(_number(group.get("size"))),
        _text(group.get("label")),
        _text(group.get("key")),
    )


def _community_report_preview(value: object) -> dict[str, Any] | None:
    """Return the bounded report metadata needed by an overview card.

    Full findings, source excerpts and media metadata belong to the selected
    community view.  Repeating those fields for every directory entry makes a
    project overview needlessly large and can leave the previous canvas frame
    visible while the browser parses the response.
    """

    if not isinstance(value, dict):
        return None
    preview_fields = (
        "report_id", "community_key", "status", "title", "summary", "rating",
    )
    return {
        field: value.get(field)
        for field in preview_fields
        if field in value
    }


def _overview_group(group: dict[str, Any]) -> dict[str, Any]:
    """Project a community into the lightweight overview-directory shape."""

    projected = {
        key: value for key, value in group.items()
        if key not in {
            "community_report", "entity_keys", "relationship_keys",
            "evidence_sources",
        }
    }
    if "community_report" in group:
        projected["community_report"] = _community_report_preview(
            group.get("community_report")
        )
    return projected


def _community_overview(
    payload: dict[str, Any], *, limit: int, relation_scope: str = "backbone",
) -> dict[str, Any]:
    nodes, edges, groups = _normalized(payload)
    by_group, entity_group = _group_index(nodes, groups)
    internal: dict[str, int] = defaultdict(int)
    cross: dict[tuple[str, str], dict[str, Any]] = {}
    for edge in edges:
        source_group = entity_group.get(_text(edge.get("source")), "unassigned")
        target_group = entity_group.get(_text(edge.get("target")), "unassigned")
        if source_group == target_group:
            internal[source_group] += 1
            continue
        # The overview is a topology map, not a direction audit.  Merge both
        # directions so a pair of communities is represented by one link.
        pair = tuple(sorted((source_group, target_group)))
        overview_source, overview_target = pair
        aggregate = cross.setdefault(pair, {
            "id": f"community:{overview_source}<->{overview_target}",
            "source": f"community:{overview_source}",
            "target": f"community:{overview_target}",
            "label": "社区间关系",
            "description": "两个社区之间实体关系的汇总",
            "weight": 0,
            "count": 0,
            "directed": False,
            "kind": "community_relation",
        })
        aggregate["weight"] = int(aggregate["weight"]) + 1
        aggregate["count"] = int(aggregate["count"]) + 1

    measured_groups: list[dict[str, Any]] = []
    for source_group in by_group.values():
        group = dict(source_group)
        key = _text(group.get("key"))
        group["cross_relationship_count"] = sum(
            int(value["count"]) for pair, value in cross.items() if key in pair
        )
        group["relationship_count"] = max(
            int(_number(group.get("relationship_count"))), internal.get(key, 0),
        )
        measured_groups.append(group)

    # Automatically synthesized type buckets are an implementation fallback,
    # not real communities.  Do not present a relationship-free bucket as a
    # community merely because an isolated entity was not assigned by the
    # clustering algorithm.  A connected fallback bucket remains visible so
    # no relationship-bearing part of a graph becomes unreachable.
    fallback_isolated_groups = [
        item for item in measured_groups
        if _text(item.get("description_source")) == "entity_type_fallback"
        and int(item.get("cross_relationship_count") or 0) == 0
        and int(_number(item.get("relationship_count"))) == 0
    ]
    directory_groups = [
        item for item in measured_groups if item not in fallback_isolated_groups
    ]

    # The directory is a ranked navigation list.  Measure relationship totals
    # before sorting; otherwise values synthesized from the graph would be
    # displayed as descending while their order was based on stale metadata.
    ordered_groups = sorted(
        directory_groups,
        key=lambda item: _group_rank(
            item, int(item.get("cross_relationship_count") or 0),
        ),
    )

    # Relationship-free groups (normally the consolidated isolated entities)
    # remain in the directory but do not occupy topology-map space.
    mapped_groups = [
        item for item in ordered_groups
        if int(item.get("cross_relationship_count") or 0) > 0
    ]
    internal_only_groups = [
        item for item in ordered_groups
        if item not in mapped_groups and int(_number(item.get("relationship_count"))) > 0
    ]
    isolated_groups = [
        item for item in ordered_groups
        if item not in mapped_groups and item not in internal_only_groups
    ]
    requested_scope = str(relation_scope or "backbone").strip().lower()
    include_internal_only = requested_scope in {
        "all_communities", "relationship_communities", "all_relationship_communities",
    }
    map_candidates = (
        [
            item for item in ordered_groups
            if int(item.get("cross_relationship_count") or 0) > 0
            or int(_number(item.get("relationship_count"))) > 0
        ]
        if include_internal_only else mapped_groups
    )
    overview_cap = min(
        (
            _OVERVIEW_ALL_RELATION_COMMUNITY_NODE_CAP
            if include_internal_only else _OVERVIEW_NODE_CAP
        ),
        max(8, int(limit)),
    )
    visible_groups = map_candidates[:overview_cap]
    collapsed_groups = map_candidates[overview_cap:]
    visible_keys = {_text(item.get("key")) for item in visible_groups}
    collapsed_keys = {_text(item.get("key")) for item in collapsed_groups}

    overview_nodes: list[dict[str, Any]] = []
    for index, group in enumerate(visible_groups):
        key = _text(group.get("key"))
        overview_nodes.append({
            "id": f"community:{key}",
            "title": _text(group.get("label")) or f"社区 {index + 1}",
            "type": "社区",
            "kind": "community",
            "group": key,
            "community_key": key,
            "description": _text(group.get("description") or group.get("summary")),
            "description_source": _text(
                group.get("description_source") or "community_summary"
            ),
            "member_count": int(_number(group.get("size"))),
            "relation_count": max(
                int(_number(group.get("relationship_count"))), internal.get(key, 0),
            ),
            "degree": sum(
                int(item["count"]) for pair, item in cross.items() if key in pair
            ),
            "size": int(_number(group.get("size"))),
            "report_status": _text(group.get("report_status")),
            "community_report": _community_report_preview(
                group.get("community_report")
            ),
        })

    rollup_key = "community:other"
    if collapsed_groups:
        collapsed_entities = sum(int(_number(item.get("size"))) for item in collapsed_groups)
        collapsed_relationships = sum(
            int(_number(item.get("relationship_count"))) for item in collapsed_groups
        )
        overview_nodes.append({
            "id": rollup_key,
            "title": f"其他 {len(collapsed_groups)} 个社区",
            "type": "社区集合",
            "kind": "community_rollup",
            "group": "__collapsed_communities__",
            "description": (
                f"{len(collapsed_groups)} 个较小的连通社区已折叠到此概览节点；"
                "可通过社区选择器或右侧目录进入任一社区。"
            ),
            "member_count": collapsed_entities,
            "relation_count": collapsed_relationships,
            "degree": 0,
            "size": collapsed_entities,
            "collapsed_community_count": len(collapsed_groups),
        })

    overview_edges: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, edge in cross.items():
        left, right = pair
        left_visible = left in visible_keys
        right_visible = right in visible_keys
        left_collapsed = left in collapsed_keys
        right_collapsed = right in collapsed_keys
        if not (left_visible or left_collapsed) or not (right_visible or right_collapsed):
            continue
        source = f"community:{left}" if left_visible else rollup_key
        target = f"community:{right}" if right_visible else rollup_key
        if source == target:
            continue
        aggregate_key = tuple(sorted((source, target)))
        aggregate = overview_edges.setdefault(aggregate_key, {
            "id": f"{aggregate_key[0]}<->{aggregate_key[1]}",
            "source": aggregate_key[0],
            "target": aggregate_key[1],
            "label": "社区间关系",
            "description": "两个可见社区范围之间实体关系的汇总",
            "weight": 0,
            "count": 0,
            "directed": False,
            "kind": "community_relation",
        })
        aggregate["weight"] = int(aggregate["weight"]) + int(edge["count"])
        aggregate["count"] = int(aggregate["count"]) + int(edge["count"])

    all_overview_edges = sorted(
        overview_edges.values(), key=lambda item: (-int(item["count"]), item["id"]),
    )
    if include_internal_only:
        visible_overview_edges = all_overview_edges
        normalized_scope = "all_communities"
    elif requested_scope in {"all", "complete"}:
        visible_overview_edges = all_overview_edges
        normalized_scope = "all"
    else:
        # A community map must communicate its main bridges before its every
        # bridge.  Start from strongest links and give each community at least
        # one chance to appear; then fill the remaining bounded edge budget.
        normalized_scope = "backbone"
        backbone: list[dict[str, Any]] = []
        covered: set[str] = set()
        selected_ids: set[str] = set()
        for edge in all_overview_edges:
            source = _text(edge.get("source"))
            target = _text(edge.get("target"))
            if source in covered and target in covered:
                continue
            backbone.append(edge)
            selected_ids.add(_text(edge.get("id")))
            covered.update((source, target))
            if len(backbone) >= _OVERVIEW_BACKBONE_EDGE_CAP:
                break
        for edge in all_overview_edges:
            if _text(edge.get("id")) in selected_ids:
                continue
            if len(backbone) >= _OVERVIEW_BACKBONE_EDGE_CAP:
                break
            backbone.append(edge)
        visible_overview_edges = backbone

    isolated_entities = sum(
        int(_number(item.get("size")))
        for item in isolated_groups + fallback_isolated_groups
    )
    return {
        **{key: value for key, value in payload.items() if key not in {"nodes", "edges", "groups"}},
        "mode": "overview",
        "nodes": overview_nodes,
        "edges": visible_overview_edges,
        "groups": [_overview_group(item) for item in ordered_groups],
        "total_entities": int(payload.get("total_entities") or len(nodes)),
        "total_relationships": int(payload.get("total_relationships") or len(edges)),
        "total_communities": len(ordered_groups),
        "visible_communities": len(visible_groups),
        "collapsed_communities": len(collapsed_groups),
        "internal_only_communities": len(internal_only_groups),
        "visible_internal_only_communities": sum(
            int(item.get("cross_relationship_count") or 0) == 0
            and int(_number(item.get("relationship_count"))) > 0
            for item in visible_groups
        ),
        "hidden_internal_only_communities": sum(
            int(item.get("cross_relationship_count") or 0) == 0
            and int(_number(item.get("relationship_count"))) > 0
            for item in collapsed_groups
        ) if include_internal_only else len(internal_only_groups),
        "isolated_entities": isolated_entities,
        "overview_relation_scope": normalized_scope,
        "total_cross_community_relationships": len(all_overview_edges),
        "visible_cross_community_relationships": len(visible_overview_edges),
        "truncated": bool(
            collapsed_groups or (internal_only_groups and not include_internal_only)
            or len(visible_overview_edges) < len(all_overview_edges)
        ),
    }


def _community_stage(level: str, *, limit: int) -> tuple[str, int, int, int | None]:
    """Map a semantic community zoom level to generic scene limits."""

    normalized = str(level or "focus").strip().lower()
    if normalized in {"full", "complete"}:
        return (
            "full",
            min(limit, _COMMUNITY_FULL_NODE_CAP),
            _COMMUNITY_FULL_EDGE_CAP,
            None,
        )
    if normalized in {"expanded", "expand", "detail"}:
        return (
            "expanded",
            min(limit, _COMMUNITY_EXPANDED_NODE_CAP),
            _COMMUNITY_EXPANDED_EDGE_CAP,
            24,
        )
    return (
        "focus",
        min(limit, _COMMUNITY_FOCUS_NODE_CAP),
        _COMMUNITY_FOCUS_EDGE_CAP,
        None,
    )


def _bounded_subgraph(
    payload: dict[str, Any], *, group_key: str, level: str, limit: int,
) -> dict[str, Any]:
    nodes, edges, groups = _normalized(payload)
    by_group, entity_group = _group_index(nodes, groups)
    selected = [
        item for item in nodes
        if entity_group.get(_text(item.get("id"))) == group_key
    ]
    total = len(selected)
    all_edges = [
        item for item in edges
        if entity_group.get(_text(item.get("source"))) == group_key
        and entity_group.get(_text(item.get("target"))) == group_key
    ]
    scene_level, node_limit, edge_limit, fanout_cap = _community_stage(level, limit=limit)
    if scene_level == "full" and len(selected) <= node_limit and len(all_edges) <= edge_limit:
        visible_nodes = selected
        visible_edges = all_edges
    else:
        visible_nodes, visible_edges = _scene_subset(
            selected,
            all_edges,
            node_limit=node_limit,
            edge_limit=edge_limit,
            max_visible_fanout=fanout_cap,
        )
    group = by_group.get(group_key) or {"key": group_key, "label": "社区"}
    return {
        **{key: value for key, value in payload.items() if key not in {"nodes", "edges", "groups"}},
        "mode": "community",
        "focus": group,
        "nodes": visible_nodes,
        "edges": visible_edges,
        "groups": [group],
        "total_entities": total,
        "total_relationships": len(all_edges),
        "visible_entities": len(visible_nodes),
        "visible_relationships": len(visible_edges),
        "community_level": scene_level,
        "available_community_levels": ["focus", "expanded", "full"],
        "truncated": total > len(visible_nodes) or len(all_edges) > len(visible_edges),
    }


def _entity_neighbourhood(
    payload: dict[str, Any], *, entity_id: str, limit: int,
) -> dict[str, Any]:
    nodes, edges, groups = _normalized(payload)
    by_id = {_text(item.get("id")): item for item in nodes}
    if entity_id not in by_id:
        raise LookupError("图谱中不存在该实体")
    incident = [
        item for item in edges
        if entity_id in {_text(item.get("source")), _text(item.get("target"))}
    ]
    degrees = _degrees(edges)
    incident.sort(key=lambda item: _edge_rank(item, degrees))
    incident = incident[:min(limit, _ENTITY_NEIGHBOURHOOD_EDGE_CAP)]
    ids = {entity_id}
    for edge in incident:
        ids.update((_text(edge.get("source")), _text(edge.get("target"))))
    visible_nodes, visible_edges = _scene_subset(
        [item for item in nodes if _text(item.get("id")) in ids],
        incident,
        node_limit=min(limit, _ENTITY_NEIGHBOURHOOD_NODE_CAP),
        edge_limit=_ENTITY_NEIGHBOURHOOD_EDGE_CAP,
    )
    return {
        **{key: value for key, value in payload.items() if key not in {"nodes", "edges", "groups"}},
        "mode": "neighbours",
        "focus": by_id[entity_id],
        "nodes": visible_nodes,
        "edges": visible_edges,
        "groups": groups,
        "total_entities": len(ids),
        "total_relationships": len(incident),
        "visible_entities": len(visible_nodes),
        "visible_relationships": len(visible_edges),
        "truncated": len(ids) > len(visible_nodes) or len(incident) > len(visible_edges),
    }


def _search(
    payload: dict[str, Any], *, query: str, limit: int,
) -> dict[str, Any]:
    nodes, edges, groups = _normalized(payload)
    needle = query.casefold().strip()
    if not needle:
        return _community_overview(payload, limit=limit, relation_scope="backbone")
    matches = [
        item for item in nodes
        if needle in " ".join((
            _text(item.get("title")), _text(item.get("type")),
            _text(item.get("description")), " ".join(item.get("aliases") or []),
        )).casefold()
    ]
    matches.sort(key=lambda item: (
        0 if _text(item.get("title")).casefold().startswith(needle) else 1,
        -_number(item.get("degree")), _text(item.get("title")),
    ))
    # Keep search exploratory rather than returning a pile of disconnected
    # labels: matched entities are seeds and their strongest direct links fill
    # the remaining bounded view.  This is topology-only and independent of
    # relation names or domain vocabulary.
    seeds = matches[:max(1, min(len(matches), max(1, limit // 4)))]
    ids = {_text(item.get("id")) for item in seeds}
    incident = [
        item for item in edges
        if _text(item.get("source")) in ids or _text(item.get("target")) in ids
    ]
    incident.sort(key=lambda item: _edge_rank(item, _degrees(edges)))
    visible_edges: list[dict[str, Any]] = []
    for edge in incident:
        source, target = _text(edge.get("source")), _text(edge.get("target"))
        additions = {source, target} - ids
        if len(ids) + len(additions) > limit:
            continue
        ids.update(additions)
        visible_edges.append(edge)
    selected_nodes, visible_edges = _scene_subset(
        [item for item in nodes if _text(item.get("id")) in ids],
        visible_edges,
        node_limit=min(limit, _RELATION_SCENE_NODE_CAP),
        edge_limit=_RELATION_SCENE_EDGE_CAP,
    )
    return {
        **{key: value for key, value in payload.items() if key not in {"nodes", "edges", "groups"}},
        "mode": "search",
        "query": query,
        "nodes": selected_nodes,
        "edges": visible_edges,
        "groups": groups,
        "total_entities": len(selected_nodes),
        "total_relationships": len(incident),
        "visible_entities": len(selected_nodes),
        "visible_relationships": len(visible_edges),
        "matched_entities": len(matches),
        "seed_entities": len(seeds),
        "truncated": len(matches) > len(seeds) or len(selected_nodes) >= limit,
    }


def _relation_backbone(payload: dict[str, Any], *, limit: int) -> dict[str, Any]:
    """Return a bounded entity-level view of verified relationships.

    Community overview is useful for orientation, but it becomes misleading
    when most entities are isolated or communities are tiny.  This view keeps
    the actual entity endpoints and strongest verified links visible so the
    browser opens on facts a user can inspect.
    """

    nodes, edges, groups = _normalized(payload)
    connected = {
        _text(edge.get("source")) for edge in edges
    } | {
        _text(edge.get("target")) for edge in edges
    }
    connected_nodes = [
        item for item in nodes if _text(item.get("id")) in connected
    ]
    ranked_edges = sorted(edges, key=lambda item: _edge_rank(item, _degrees(edges)))
    selected_nodes, selected_edges = _scene_subset(
        connected_nodes,
        ranked_edges,
        node_limit=min(limit, _RELATION_SCENE_NODE_CAP),
        edge_limit=_RELATION_SCENE_EDGE_CAP,
    )
    return {
        **{key: value for key, value in payload.items() if key not in {"nodes", "edges", "groups"}},
        "mode": "relations",
        "nodes": selected_nodes,
        "edges": selected_edges,
        "groups": groups,
        "connected_entities": len(connected),
        "visible_entities": len(selected_nodes),
        "visible_relationships": len(selected_edges),
        "total_entities": int(payload.get("total_entities") or len(nodes)),
        "total_relationships": int(payload.get("total_relationships") or len(edges)),
        "truncated": len(selected_nodes) < len(nodes) or len(selected_edges) < len(ranked_edges),
    }


def build_graph_explorer_view(
    payload: dict[str, Any], *, mode: str = "overview",
    community_key: str = "", entity_id: str = "", query: str = "",
    community_level: str = "focus", overview_relation_scope: str = "backbone",
    limit: int = 500,
) -> dict[str, Any]:
    """Return a bounded, strategy-neutral graph view."""

    limit = max(20, min(int(limit), 2000))
    normalized_mode = str(mode or "overview").strip().lower()
    if normalized_mode == "community":
        if not str(community_key).strip():
            raise ValueError("community 模式缺少社区标识")
        return _bounded_subgraph(
            payload,
            group_key=str(community_key),
            level=community_level,
            limit=limit,
        )
    if normalized_mode in {"neighbours", "neighbors", "entity"}:
        if not str(entity_id).strip():
            raise ValueError("实体邻域模式缺少实体标识")
        return _entity_neighbourhood(payload, entity_id=str(entity_id), limit=limit)
    if normalized_mode == "search":
        return _search(payload, query=str(query), limit=limit)
    if normalized_mode in {"relations", "backbone", "network"}:
        return _relation_backbone(payload, limit=limit)
    if normalized_mode != "overview":
        raise ValueError("不支持的图谱浏览模式")
    return _community_overview(
        payload, limit=limit, relation_scope=overview_relation_scope,
    )
