from __future__ import annotations

import pytest

from knowledge_forge.graph_explorer import build_graph_explorer_view


def _payload():
    return {
        "strategy": "fixture",
        "total_entities": 5,
        "total_relationships": 4,
        "nodes": [
            {"id": "a", "title": "Alpha", "group": "g1", "degree": 2},
            {"id": "b", "title": "Beta", "group": "g1", "degree": 2},
            {"id": "c", "title": "Gamma", "group": "g2", "degree": 2},
            {"id": "d", "title": "Delta", "group": "g2", "degree": 1},
            {"id": "e", "title": "Unconnected", "group": "g3", "degree": 0},
        ],
        "edges": [
            {"id": "ab", "source": "a", "target": "b", "weight": 1},
            {"id": "bc", "source": "b", "target": "c", "weight": 2},
            {"id": "ca", "source": "c", "target": "a", "weight": 3},
            {"id": "cd", "source": "c", "target": "d", "weight": 1},
        ],
        "groups": [
            {"key": "g1", "label": "社区甲", "size": 2},
            {"key": "g2", "label": "社区乙", "size": 2},
            {"key": "g3", "label": "孤立社区", "size": 1},
        ],
    }


def test_overview_aggregates_bidirectional_cross_group_edges_once():
    view = build_graph_explorer_view(_payload(), mode="overview")

    assert view["mode"] == "overview"
    # The topology map intentionally excludes relationship-free groups; they
    # remain discoverable in the complete community directory returned by the
    # same view.
    assert {item["group"] for item in view["nodes"]} == {"g1", "g2"}
    assert len(view["edges"]) == 1
    assert view["edges"][0]["count"] == 2
    assert view["edges"][0]["kind"] == "community_relation"
    assert view["edges"][0].get("evidence_text") is None
    assert view["edges"][0]["directed"] is False
    assert next(item for item in view["groups"] if item["key"] == "g3")["size"] == 1
    assert view["isolated_entities"] == 1


def test_overview_explains_cross_community_fact_aggregate_with_sources():
    payload = _payload()
    cross_edges = {
        edge["id"]: edge for edge in payload["edges"] if edge["id"] in {"bc", "ca"}
    }
    cross_edges["bc"].update({
        "predicate_label": "依赖",
        "evidence_text": "Beta 依赖 Gamma。",
        "evidence_count": 1,
        "document_id": 10,
        "source_info": {
            "document_id": 10,
            "filename": "alpha.pdf",
            "page_label": "第 3 页",
            "media_assets": [{"url": "/media/a.png", "title": "图 1"}],
        },
    })
    cross_edges["ca"].update({
        "predicate_label": "使用",
        "evidence_text": "Gamma 使用 Alpha。",
        "evidence_count": 1,
        "document_id": 11,
        "source_info": {
            "document_id": 11,
            "filename": "beta.docx",
            "page_label": "第 7 页",
        },
    })
    payload["edges"].append({
        "id": "bd", "source": "b", "target": "d",
        "predicate_label": "依赖", "evidence_text": "Beta 也依赖 Delta。",
        "document_id": 10,
        "source_info": {
            "document_id": 10, "filename": "alpha.pdf", "page_label": "第 4 页",
        },
    })

    view = build_graph_explorer_view(
        payload, mode="overview", overview_relation_scope="all",
    )
    edge = view["edges"][0]

    assert edge["label"] == "主要为依赖 · 2/3"
    assert edge["predicate_counts"] == [
        {"label": "依赖", "count": 2},
        {"label": "使用", "count": 1},
    ]
    assert edge["predicate_type_count"] == 2
    assert edge["entity_pair_count"] == 3
    assert edge["document_count"] == 2
    assert edge["evidence_count"] == 3
    assert edge["direction_counts"] == {
        "source_to_target": 2, "target_to_source": 1,
    }
    assert {item["predicate_label"] for item in edge["relation_examples"]} == {
        "依赖", "使用",
    }
    assert any(
        item["source_info"].get("media_assets")
        for item in edge["relation_examples"]
    )
    assert not any(key.startswith("_") for key in edge)


def test_community_and_neighbour_views_are_bounded_and_exact():
    community = build_graph_explorer_view(
        _payload(), mode="community", community_key="g2", limit=20,
    )
    neighbour = build_graph_explorer_view(
        _payload(), mode="neighbours", entity_id="c", limit=20,
    )

    assert {item["id"] for item in community["nodes"]} == {"c", "d"}
    assert [item["id"] for item in community["edges"]] == ["cd"]
    assert {item["id"] for item in neighbour["nodes"]} == {"a", "b", "c", "d"}
    assert {item["id"] for item in neighbour["edges"]} == {"bc", "ca", "cd"}


def test_relation_backbone_opens_on_verified_entity_endpoints():
    view = build_graph_explorer_view(_payload(), mode="relations", limit=4)

    assert view["mode"] == "relations"
    assert {item["id"] for item in view["nodes"]} == {"a", "b", "c", "d"}
    assert {item["id"] for item in view["edges"]} == {"ca", "bc", "ab", "cd"}
    assert view["connected_entities"] == 4


def test_search_is_explicit_and_unknown_entity_fails_closed():
    empty = build_graph_explorer_view(_payload(), mode="search", query="missing")
    found = build_graph_explorer_view(_payload(), mode="search", query="gamma")

    assert empty["nodes"] == []
    assert {item["id"] for item in found["nodes"]} == {"a", "b", "c", "d"}
    assert found["matched_entities"] == 1
    assert {item["id"] for item in found["edges"]} == {"bc", "ca", "cd"}
    with pytest.raises(LookupError):
        build_graph_explorer_view(
            _payload(), mode="neighbours", entity_id="not-present",
        )


def test_strategy_neutral_view_does_not_depend_on_predicate_names():
    left = _payload()
    right = _payload()
    for edge, label in zip(left["edges"], ["alpha_rel"] * 4):
        edge["label"] = label
    for edge, label in zip(right["edges"], ["renamed_rel"] * 4):
        edge["label"] = label

    left_view = build_graph_explorer_view(left, mode="overview")
    right_view = build_graph_explorer_view(right, mode="overview")

    assert [item["count"] for item in left_view["edges"]] == [
        item["count"] for item in right_view["edges"]
    ]


def test_overview_collapses_low_priority_communities_without_losing_directory():
    community_count = 48
    nodes = []
    edges = []
    groups = []
    for index in range(community_count):
        key = f"g{index:02d}"
        left = f"{key}-a"
        right = f"{key}-b"
        nodes.extend([
            {"id": left, "title": left, "group": key, "degree": 1},
            {"id": right, "title": right, "group": key, "degree": 1},
        ])
        edges.append({"id": f"{key}-internal", "source": left, "target": right})
        groups.append({"key": key, "label": f"社区 {index:02d}", "size": 2})
    # The chain makes every community connected at the overview level while
    # preserving a generic fixture (no domain labels or predicate names).
    for index in range(community_count - 1):
        edges.append({
            "id": f"cross-{index}", "source": f"g{index:02d}-b",
            "target": f"g{index + 1:02d}-a",
        })

    view = build_graph_explorer_view({
        "nodes": nodes,
        "edges": edges,
        "groups": groups,
        "total_entities": len(nodes),
        "total_relationships": len(edges),
    }, mode="overview", limit=500)

    assert view["visible_communities"] == 42
    assert view["collapsed_communities"] == 6
    assert view["internal_only_communities"] == 0
    assert len(view["groups"]) == community_count
    assert any(item["kind"] == "community_rollup" for item in view["nodes"])
    assert len(view["nodes"]) == 43


def test_overview_defaults_to_a_bounded_bridge_backbone_with_full_opt_in():
    nodes = []
    edges = []
    groups = []
    for index in range(40):
        key = f"g{index:02d}"
        nodes.extend([
            {"id": f"{key}-a", "title": f"{key}-a", "group": key},
            {"id": f"{key}-b", "title": f"{key}-b", "group": key},
        ])
        groups.append({"key": key, "label": f"社区 {index:02d}", "size": 2})
        if index:
            edges.append({
                "id": f"bridge-{index}", "source": f"g{index - 1:02d}-b",
                "target": f"{key}-a", "weight": 1,
            })

    payload = {"nodes": nodes, "edges": edges, "groups": groups}
    backbone = build_graph_explorer_view(payload, mode="overview")
    complete = build_graph_explorer_view(
        payload, mode="overview", overview_relation_scope="all",
    )

    assert backbone["overview_relation_scope"] == "backbone"
    assert backbone["visible_cross_community_relationships"] == len(backbone["edges"])
    assert len(backbone["edges"]) == 32
    assert complete["overview_relation_scope"] == "all"
    assert complete["total_cross_community_relationships"] == len(complete["edges"])
    assert len(complete["edges"]) == 39
    assert len(backbone["edges"]) < len(complete["edges"])


def test_overview_directory_is_sorted_by_measured_relationship_count():
    payload = {
        "nodes": [
            {"id": "a", "title": "a", "group": "g1"},
            {"id": "b", "title": "b", "group": "g1"},
            {"id": "c", "title": "c", "group": "g1"},
            {"id": "d", "title": "d", "group": "g2"},
            {"id": "e", "title": "e", "group": "g2"},
        ],
        "edges": [
            {"id": "ab", "source": "a", "target": "b"},
            {"id": "bc", "source": "b", "target": "c"},
            {"id": "ca", "source": "c", "target": "a"},
            {"id": "de", "source": "d", "target": "e"},
        ],
        "groups": [
            {"key": "g2", "label": "乙", "size": 2},
            {"key": "g1", "label": "甲", "size": 3},
        ],
    }

    view = build_graph_explorer_view(payload, mode="overview")

    assert [group["key"] for group in view["groups"]] == ["g1", "g2"]
    assert [group["relationship_count"] for group in view["groups"]] == [3, 1]


def test_bounded_scene_omits_unconnected_decoration_from_canvas():
    payload = _payload()
    payload["nodes"].append({"id": "f", "title": "Unused", "group": "g2", "degree": 0})
    payload["groups"][1]["size"] = 3

    view = build_graph_explorer_view(payload, mode="community", community_key="g2", limit=20)

    assert {item["id"] for item in view["nodes"]} == {"c", "d"}
    assert {item["id"] for item in view["edges"]} == {"cd"}
    assert view["total_entities"] == 3


def test_overview_keeps_internal_only_communities_in_directory_not_topology():
    # g1 and g2 have a cross-community relation, while g3 is relationship
    # free.  Add a fourth community that has an internal fact only.
    payload = _payload()
    payload["nodes"].extend([
        {"id": "f", "title": "Foxtrot", "group": "g4", "degree": 1},
        {"id": "g", "title": "Golf", "group": "g4", "degree": 1},
    ])
    payload["edges"].append({"id": "fg", "source": "f", "target": "g"})
    payload["groups"].append({"key": "g4", "label": "内部社区", "size": 2})

    view = build_graph_explorer_view(payload, mode="overview")

    assert "g4" not in {item["group"] for item in view["nodes"]}
    assert next(item for item in view["groups"] if item["key"] == "g4")["relationship_count"] == 1
    assert view["internal_only_communities"] == 1


def test_all_relationship_communities_scope_includes_internal_only_groups():
    payload = _payload()
    payload["nodes"].extend([
        {"id": "f", "title": "Foxtrot", "group": "g4", "degree": 1},
        {"id": "g", "title": "Golf", "group": "g4", "degree": 1},
    ])
    payload["edges"].append({"id": "fg", "source": "f", "target": "g"})
    payload["groups"].append({
        "key": "g4", "label": "内部社区", "size": 2,
        "community_report": {"status": "succeeded", "summary": "报告"},
        "report_status": "succeeded",
    })

    view = build_graph_explorer_view(
        payload,
        mode="overview",
        overview_relation_scope="all_communities",
    )

    assert view["overview_relation_scope"] == "all_communities"
    assert {item["group"] for item in view["nodes"]} == {"g1", "g2", "g4"}
    assert view["visible_internal_only_communities"] == 1
    assert view["hidden_internal_only_communities"] == 0
    internal_node = next(item for item in view["nodes"] if item["group"] == "g4")
    assert internal_node["community_report"]["summary"] == "报告"


def test_all_community_scope_adds_topic_navigation_without_changing_fact_views():
    payload = _payload()
    payload["nodes"].extend([
        {"id": "f", "title": "Foxtrot", "group": "g4", "degree": 1},
        {"id": "g", "title": "Golf", "group": "g4", "degree": 1},
    ])
    payload["edges"].append({"id": "fg", "source": "f", "target": "g"})
    payload["groups"].append({
        "key": "g4", "label": "内部社区", "size": 2,
        "relationship_count": 1,
    })
    payload["community_topic_associations"] = [{
        "id": "topic-g2-g4",
        "source_community_key": "g2",
        "target_community_key": "g4",
        "score": 0.91,
        "threshold": 0.82,
        "shared_document_count": 1,
        "lexical_support": True,
        "selection_reasons": ["mutual_nearest"],
        "representative_entity_matches": [{
            "source_entity_key": "c", "source_entity_name": "Gamma",
            "target_entity_key": "f", "target_entity_name": "Foxtrot",
            "score": 0.93,
        }],
        "kind": "community_topic_association",
    }]
    payload["community_topic_association_metrics"] = {
        "policy_version": "fixture-v1",
    }

    backbone = build_graph_explorer_view(payload, mode="overview")
    factual = build_graph_explorer_view(
        payload, mode="overview", overview_relation_scope="all",
    )
    complete = build_graph_explorer_view(
        payload, mode="overview", overview_relation_scope="all_communities",
    )

    assert all(
        edge["kind"] != "community_topic_association"
        for edge in backbone["edges"] + factual["edges"]
    )
    topic = next(
        edge for edge in complete["edges"]
        if edge["kind"] == "community_topic_association"
    )
    assert {topic["source"], topic["target"]} == {
        "community:g2", "community:g4",
    }
    assert topic["directed"] is False
    assert topic["label"] == "内容相近 91%"
    assert topic["evidence_text"] == ""
    assert "count" not in topic
    assert topic["selection_reasons"] == ["mutual_nearest"]
    assert topic["representative_entity_matches"][0]["source_entity_name"] == "Gamma"
    assert topic["shared_document_count"] == 1
    assert topic["lexical_support"] is True
    assert complete["total_cross_community_relationships"] == 1
    assert complete["visible_cross_community_relationships"] == 1
    assert complete["total_topic_associations"] == 1
    assert complete["visible_topic_associations"] == 1
    assert complete["community_topic_association_metrics"]["policy_version"] == (
        "fixture-v1"
    )


def test_topic_navigation_does_not_draw_a_second_line_over_a_factual_bridge():
    payload = _payload()
    payload["community_topic_associations"] = [{
        "source_community_key": "g1",
        "target_community_key": "g2",
        "score": 0.99,
    }]

    view = build_graph_explorer_view(
        payload, mode="overview", overview_relation_scope="all_communities",
    )

    assert len(view["edges"]) == 1
    assert view["edges"][0]["kind"] == "community_relation"
    assert view["visible_topic_associations"] == 0


def test_overview_does_not_present_isolated_type_fallback_as_a_community():
    payload = _payload()
    payload["nodes"].append({
        "id": "unassigned", "title": "未归属实体", "type": "概念",
        "type_id": "concept", "degree": 0,
    })

    view = build_graph_explorer_view(payload, mode="overview")

    assert view["total_communities"] == 3
    assert {item["key"] for item in view["groups"]} == {"g1", "g2", "g3"}
    assert not any(
        str(item["key"]).startswith("type:") for item in view["groups"]
    )
    assert view["isolated_entities"] == 2


def test_overview_uses_report_preview_but_community_detail_keeps_full_report():
    payload = _payload()
    report = {
        "report_id": "report-1",
        "community_key": "g1",
        "status": "succeeded",
        "title": "主题报告",
        "summary": "供目录展示的摘要",
        "content": "完整报告正文",
        "findings": [{"title": "结论", "explanation": "完整解释"}],
        "sources": [{"evidence_text": "完整来源"}],
        "rating": 8.5,
    }
    payload["groups"][0].update({
        "community_report": report,
        "evidence_sources": [{"evidence_text": "群组来源"}],
        "entity_keys": ["a", "b"],
        "relationship_keys": ["ab"],
    })

    overview = build_graph_explorer_view(payload, mode="overview")
    detail = build_graph_explorer_view(
        payload, mode="community", community_key="g1",
    )

    overview_group = next(item for item in overview["groups"] if item["key"] == "g1")
    overview_node = next(item for item in overview["nodes"] if item["group"] == "g1")
    assert overview_group["community_report"] == {
        "report_id": "report-1",
        "community_key": "g1",
        "status": "succeeded",
        "title": "主题报告",
        "summary": "供目录展示的摘要",
        "rating": 8.5,
    }
    assert overview_node["community_report"] == overview_group["community_report"]
    assert "evidence_sources" not in overview_group
    assert "entity_keys" not in overview_group
    assert detail["focus"]["community_report"]["content"] == "完整报告正文"
    assert detail["focus"]["community_report"]["findings"][0]["title"] == "结论"
    assert detail["focus"]["evidence_sources"][0]["evidence_text"] == "群组来源"


def test_community_expansion_has_explicit_stages_and_can_return_complete_facts():
    payload = {
        "nodes": [
            {"id": f"n{index}", "title": f"节点 {index}", "group": "g1"}
            for index in range(140)
        ],
        "edges": [
            {"id": f"e{index}", "source": f"n{index}", "target": f"n{index + 1}"}
            for index in range(139)
        ],
        "groups": [{"key": "g1", "label": "扩展社区", "size": 140}],
    }

    focus = build_graph_explorer_view(
        payload, mode="community", community_key="g1", community_level="focus", limit=2_000,
    )
    expanded = build_graph_explorer_view(
        payload, mode="community", community_key="g1", community_level="expanded", limit=2_000,
    )
    complete = build_graph_explorer_view(
        payload, mode="community", community_key="g1", community_level="full", limit=2_000,
    )

    assert focus["community_level"] == "focus"
    assert focus["visible_entities"] <= 120
    assert expanded["community_level"] == "expanded"
    assert expanded["visible_entities"] >= focus["visible_entities"]
    assert complete["community_level"] == "full"
    assert {item["id"] for item in complete["nodes"]} == {f"n{index}" for index in range(140)}
    assert {item["id"] for item in complete["edges"]} == {f"e{index}" for index in range(139)}
    assert complete["truncated"] is False
