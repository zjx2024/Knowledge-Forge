from __future__ import annotations

from knowledge_forge.span_entities import (
    SpanFirstEntityExtractor,
    entity_boundary_reason,
    layout_equivalent_surface_occurrences,
    policy_relation_endpoint_candidates,
    structured_list_item_endpoint_candidates,
    structured_relation_endpoint_candidates,
    structured_section_label_endpoint_candidates,
)


def test_layout_coverage_finds_cjk_renderer_whitespace_variant():
    content = "联邦学习用于训练；FedDIS 是基于联邦 学习的框架。"
    start = content.index("联邦 学习")

    assert layout_equivalent_surface_occurrences(
        content, "联邦学习",
    ) == [(start, start + len("联邦 学习"))]


def test_layout_coverage_preserves_semantic_latin_word_spaces():
    content = "New York and NewYork are separate source spellings."

    assert layout_equivalent_surface_occurrences(content, "New York") == []


def test_layout_coverage_does_not_cross_structural_cjk_newline():
    content = "任务管理\n知识图谱"

    assert layout_equivalent_surface_occurrences(content, "管理知识") == []


def test_layout_coverage_audit_reuses_structure_and_publishability_gates():
    content = (
        "# 主题 分析\n"
        "章节ID: 主题 分析\n"
        "正文包含主题分析，并说明主题 分析的作用。"
    )
    seed_start = content.index("主题分析")
    extractor = SpanFirstEntityExtractor("entity-layout-audit-test")
    normalized = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[{
            "name": "主题分析",
            "type": "Technology",
            "mention_start": seed_start,
            "mention_end": seed_start + len("主题分析"),
            "confidence": 0.9,
        }],
        chunk_metadata={
            "chunk_role": "child",
            "structural_spans": [{
                "start": 2,
                "end": 7,
                "source_role": "heading",
            }],
        },
    )

    audit = extractor.audit_layout_coverage(
        content=content,
        mentions=normalized.mentions,
        type_assertions=normalized.type_assertions,
        chunk_metadata={
            "chunk_role": "child",
            "structural_spans": [{
                "start": 2,
                "end": 7,
                "source_role": "heading",
            }],
        },
    )

    assert audit["candidate_count"] == 2
    assert audit["eligible_count"] == 0
    assert audit["rejected_count"] == 2
    assert audit["recovered_count"] == 1
    assert audit["rejection_reasons"] == {
        "renderer_metadata_line": 1,
        "structural_node": 1,
    }
    assert [item.surface for item in normalized.mentions] == [
        "主题分析", "主题 分析",
    ]
    recovered = normalized.mentions[1]
    assert recovered.normalized_surface == "主题分析"
    assert recovered.recovery_kind == "layout_whitespace_variant"
    assert normalized.type_assertions[1].source == (
        "deterministic_layout_repeat"
    )


def test_layout_repeat_recovery_does_not_create_nested_partial_mention():
    content = "联邦学习可用于训练，联邦 学习框架负责协调。"
    exact_start = content.index("联邦学习")
    framework_start = content.index("联邦 学习框架")
    extractor = SpanFirstEntityExtractor("entity-layout-recovery-test")

    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[
            {
                "name": "联邦学习",
                "type": "Technology",
                "mention_start": exact_start,
                "mention_end": exact_start + len("联邦学习"),
            },
            {
                "name": "联邦 学习框架",
                "type": "Product",
                "mention_start": framework_start,
                "mention_end": framework_start + len("联邦 学习框架"),
            },
        ],
        chunk_metadata={},
    )

    assert [(item.surface, item.recovery_kind) for item in result.mentions] == [
        ("联邦学习", ""),
        ("联邦 学习框架", ""),
    ]


def test_layout_repeat_does_not_expand_short_generic_seed():
    content = "模型用于训练，模 型负责生成结果。"
    seed_start = content.index("模型")
    extractor = SpanFirstEntityExtractor("entity-layout-recovery-test")

    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[{
            "name": "模型",
            "type": "Product",
            "mention_start": seed_start,
            "mention_end": seed_start + len("模型"),
        }],
        chunk_metadata={},
    )
    audit = extractor.audit_layout_coverage(
        content=content,
        mentions=result.mentions,
        type_assertions=result.type_assertions,
        chunk_metadata={},
    )

    assert [item.surface for item in result.mentions] == ["模型"]
    assert audit["eligible_count"] == 0
    assert audit["rejection_reasons"] == {
        "short_low_information_layout_seed": 1,
    }


def test_structured_relation_line_adds_exact_endpoints_but_not_predicate():
    content = "- Component A --arbitrary_label--> Container B"
    rows = structured_relation_endpoint_candidates(content, [{
        "start": 0,
        "end": len(content),
        "source_role": "list_item",
        "scope_id": "line-1",
    }])

    assert [(item["name"], item["mention_start"], item["mention_end"])
            for item in rows] == [
        ("Component A", 2, 13),
        ("Container B", 35, 46),
    ]
    assert all(item["type"] == "GenericEntity" for item in rows)
    assert all("arbitrary_label" not in item["name"] for item in rows)


def test_structured_endpoint_parser_requires_parser_owned_eligible_span():
    content = "A --alpha_rel--> B"

    assert structured_relation_endpoint_candidates(content, []) == []
    assert structured_relation_endpoint_candidates(content, [{
        "start": 0, "end": len(content), "source_role": "heading",
    }]) == []


def test_structured_list_items_add_exact_marker_free_endpoint_candidates():
    content = "\uf0a8 System X\n— Module A\n Nested B"
    spans = [
        {
            "start": start, "end": end, "source_role": "list_item",
            "scope_id": f"item-{index}", "list_marker": marker,
        }
        for index, (start, end, marker) in enumerate([
            (0, len("\uf0a8 System X"), "\uf0a8"),
            (len("\uf0a8 System X\n"), len(content) - len("\n Nested B"), "—"),
            (len(content) - len(" Nested B"), len(content), ""),
        ])
    ]

    rows = structured_list_item_endpoint_candidates(content, spans)

    assert [item["surface"] for item in rows] == [
        "System X", "Module A", "Nested B",
    ]
    assert all(item["endpoint_candidate"] is True for item in rows)
    assert all(
        content[item["mention_start"]:item["mention_end"]] == item["surface"]
        for item in rows
    )


def test_structured_section_label_adds_exact_body_label_candidate_only():
    content = "Requirements\n\nSR-01 shall be checked."
    label_end = len("Requirements")
    rows = structured_section_label_endpoint_candidates(content, [{
        "start": 0,
        "end": label_end,
        "source_role": "body",
        "scope_id": "label-scope",
    }])

    assert [(item["surface"], item["mention_start"], item["mention_end"])
            for item in rows] == [("Requirements", 0, label_end)]
    assert rows[0]["endpoint_candidate"] is True
    assert rows[0]["section_label_candidate"] is True


def test_structured_section_label_recovers_inline_identifier_from_body_line():
    content = "Requirements\nSR-01 The service shall check a flight intent."
    body_start = content.index("SR-01")
    rows = structured_section_label_endpoint_candidates(content, [{
        "start": 0,
        "end": len(content),
        "source_role": "body",
        "scope_id": "body-scope",
    }])

    assert any(
        item["surface"] == "SR-01"
        and item["mention_start"] == body_start
        and item["mention_end"] == body_start + len("SR-01")
        and item["inline_label_candidate"] is True
        for item in rows
    )


def test_policy_relation_endpoint_candidates_recover_exact_cue_frame_endpoints():
    content = "甲公司位于上海。甲公司使用乙系统。"
    rows = policy_relation_endpoint_candidates(
        content,
        [
            {"name": "located_in", "optional_soft_cues": ["位于"]},
            {"name": "uses", "optional_soft_cues": ["使用"]},
        ],
    )
    spans = {
        (row["mention_start"], row["mention_end"], row["surface"])
        for row in rows
    }
    assert (0, 3, "甲公司") in spans
    assert (5, 7, "上海") in spans
    assert any(surface == "乙系统" for _start, _end, surface in spans)
    assert all(
        {
            "mention_start", "mention_end", "evidence_text",
            "evidence_start", "evidence_end",
        } <= set(row)
        for row in rows
    )
    assert all(
        content[row["mention_start"]:row["mention_end"]]
        == row["surface"]
        for row in rows
    )


def test_exact_endpoint_projection_survives_broad_clause_boundary_gate():
    content = "工业机器人应用于汽车制造生产线。"
    rows = policy_relation_endpoint_candidates(
        content, [{"name": "alpha_rel", "optional_soft_cues": ["应用于"]}],
    )
    exact = next(row for row in rows if row["surface"] == "汽车制造生产线")
    broad = {
        "name": content,
        "type": "GenericEntity",
        "mention_start": 0,
        "mention_end": len(content) - 1,
    }
    assert entity_boundary_reason(content, exact, peers=[broad, *rows]) == ""
    assert any(row["surface"] == "工业机器人" for row in rows)


def test_policy_cue_does_not_match_ascii_word_fragment_and_splits_coordination():
    content = (
        "The platform implements a plan. "
        "The platform is a system that supports analysis and tracking."
    )
    rows = policy_relation_endpoint_candidates(
        content,
        [
            {"name": "is_a", "optional_soft_cues": ["is"]},
            {"name": "supports", "optional_soft_cues": ["supports"]},
        ],
    )
    surfaces = {row["surface"] for row in rows}
    assert "implements" not in surfaces
    assert "platform" in surfaces
    assert "system" in surfaces
    assert "analysis" in surfaces
    assert "tracking" in surfaces


def test_policy_cue_projects_prepositional_object_tail():
    content = "This report contains digital data sets for aquifers, geology, and regions."
    rows = policy_relation_endpoint_candidates(
        content,
        [{"name": "contains", "optional_soft_cues": ["contains"]}],
    )
    surfaces = {row["surface"] for row in rows}
    assert "report" in surfaces
    assert "aquifers" in surfaces
    assert "geology" in surfaces
    assert "regions" in surfaces


def test_endpoint_projection_still_rejects_qualified_outer_and_residual_tail():
    content = "甲组织使用乙组织提供的乙技术。工业机器人应用于汽车制造生产线。"
    rows = policy_relation_endpoint_candidates(
        content,
        [
            {"name": "uses", "optional_soft_cues": ["使用"]},
            {"name": "alpha_rel", "optional_soft_cues": ["应用于"]},
            {"name": "other_rel", "optional_soft_cues": ["生产"]},
        ],
    )
    typed = {
        "name": "乙技术", "type": "Technology",
        "mention_start": content.index("乙技术"),
        "mention_end": content.index("乙技术") + len("乙技术"),
    }
    qualified = next(row for row in rows if row["surface"] == "乙组织提供的乙技术")
    tail = next(row for row in rows if row["surface"] == "线")
    assert entity_boundary_reason(
        content, qualified, peers=[typed],
    ) == "nested_non_atomic_surface"
    assert entity_boundary_reason(content, tail, peers=rows) == "partial_context_fragment"


def test_endpoint_projection_keeps_complete_endpoint_over_nested_residual():
    content = "工业机器人应用于汽车制造生产线"
    complete = {
        "name": "汽车制造生产线",
        "type": "GenericEntity",
        "mention_start": 7,
        "mention_end": len(content),
        "endpoint_candidate": True,
    }
    residual = {
        "name": "线",
        "type": "GenericEntity",
        "mention_start": len(content) - 1,
        "mention_end": len(content),
        "endpoint_candidate": True,
    }
    assert entity_boundary_reason(content, complete, [complete, residual]) == ""
    assert entity_boundary_reason(
        content, residual, [complete, residual],
    ) == "partial_context_fragment"


def test_endpoint_projection_rejects_composite_table_row_surface():
    content = "| USS | submits operation intent | SR-02 |"
    row = {
        "name": content,
        "type": "GenericEntity",
        "mention_start": 0,
        "mention_end": len(content),
        "endpoint_candidate": True,
    }
    assert entity_boundary_reason(content, row, [row]) == (
        "composite_structured_endpoint"
    )


def test_span_first_mentions_are_exact_stable_and_type_decoupled():
    content = "Acme Corp released Model-X."
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    raw = [{
        "name": "Acme Corp", "type": "Organization", "evidence_text": "Acme Corp",
        "evidence_start": 0, "evidence_end": 9, "confidence": 0.9,
    }]
    first = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=raw, chunk_metadata={},
    )
    changed_type = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=[{**raw[0], "type": "Product"}], chunk_metadata={},
    )
    assert first.mentions[0].mention_id == changed_type.mentions[0].mention_id
    assert first.mentions[0].occurrence_key == changed_type.mentions[0].occurrence_key
    assert first.mentions[0].surface == "Acme Corp"
    assert first.type_assertions[0].entity_type == "Organization"
    assert changed_type.type_assertions[0].entity_type == "Product"


def test_span_first_rejects_generated_names_and_audits_structural_nodes():
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content="1. Introduction",
        raw_entities=[
            {"name": "invented", "type": "Concept", "evidence_text": "invented", "evidence_start": 0, "evidence_end": 8},
            {"name": "1", "type": "GenericEntity", "evidence_text": "1", "evidence_start": 0, "evidence_end": 1},
        ],
        chunk_metadata={"structural_role": "table_of_contents"},
    )
    assert result.mentions == []
    assert {item["reason"] for item in result.rejected} == {"invalid_exact_span", "structural_node"}


def test_alias_requires_its_own_exact_span():
    content = "International Business Machines (IBM) announced a product."
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=[{
            "name": "International Business Machines", "type": "Organization",
            "evidence_text": "International Business Machines", "evidence_start": 0,
            "evidence_end": 31, "aliases": ["IBM", "Big Blue"],
            "alias_spans": [{"surface": "IBM", "start": 33, "end": 36}],
        }], chunk_metadata={},
    )
    assert [item.surface for item in result.aliases] == ["IBM"]


def test_mention_span_is_independent_from_sentence_evidence_span():
    content = "IBM announced a product."
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=[{
            "name": "IBM", "type": "Organization",
            "mention_start": 0, "mention_end": 3,
            "evidence_text": content, "evidence_start": 0, "evidence_end": len(content),
        }],
        chunk_metadata={},
    )
    assert result.mentions[0].surface == "IBM"
    assert (result.mentions[0].start, result.mentions[0].end) == (0, 3)
    assert result.mentions[0].evidence == content


def test_one_exact_model_mention_recovers_all_repeated_source_occurrences():
    content = "New York links New York; New York remains distinct by source position."
    surface = "New York"
    starts = []
    cursor = 0
    while True:
        start = content.find(surface, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + len(surface)
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=[{
            "name": surface, "type": "Location", "mention_start": starts[0],
            "mention_end": starts[0] + len(surface), "evidence_text": surface,
            "evidence_start": starts[0],
            "evidence_end": starts[0] + len(surface),
        }],
        chunk_metadata={},
    )

    assert len(result.mentions) == 3
    assert len({item.mention_id for item in result.mentions}) == 3
    assert [(item.start, item.end) for item in result.mentions] == [
        (start, start + len(surface)) for start in starts
    ]


def test_repeat_recovery_does_not_create_nested_partial_mention():
    content = "城市甲负责协调。城市甲产业园承载实验室。"
    city_start = content.index("城市甲")
    park_start = content.index("城市甲产业园")
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[
            {
                "name": "城市甲", "type": "Location",
                "mention_start": city_start,
                "mention_end": city_start + len("城市甲"),
            },
            {
                "name": "城市甲产业园", "type": "Location",
                "mention_start": park_start,
                "mention_end": park_start + len("城市甲产业园"),
            },
        ],
        chunk_metadata={},
    )

    assert [(item.surface, item.start, item.end) for item in result.mentions] == [
        ("城市甲", city_start, city_start + len("城市甲")),
        ("城市甲产业园", park_start, park_start + len("城市甲产业园")),
    ]


def test_coordinated_distinct_item_repairs_nested_short_model_span():
    content = "健康监控分为核心级、多分区级和分区级。"
    nested_start = content.index("分区级")
    standalone_start = content.rindex("分区级")
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[
            {
                "name": "分区级", "type": "GenericEntity",
                "mention_start": nested_start,
                "mention_end": nested_start + len("分区级"),
                "confidence": 0.8,
            },
            {
                "name": "分区级", "type": "GenericEntity",
                "mention_start": standalone_start,
                "mention_end": standalone_start + len("分区级"),
                "confidence": 0.8,
            },
        ],
        chunk_metadata={},
    )

    assert [(item.surface, item.start, item.end) for item in result.mentions] == [
        ("分区级", standalone_start, standalone_start + len("分区级")),
        (
            "多分区级", content.index("多分区级"),
            content.index("多分区级") + len("多分区级"),
        ),
    ]
    assert any(
        item["reason"] == "nested_coordinated_item_fragment"
        for item in result.rejected
    )


def test_coordinated_distinct_item_repairs_parenthesized_final_sibling():
    content = "— 健康监控（核心级、多分区级、分区级）\n— 配置数据管理"
    nested_start = content.index("分区级")
    standalone_start = content.rindex("分区级")
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[
            {
                "name": "分区级", "type": "GenericEntity",
                "mention_start": nested_start,
                "mention_end": nested_start + len("分区级"),
                "confidence": 0.8,
            },
            {
                "name": "分区级", "type": "GenericEntity",
                "mention_start": standalone_start,
                "mention_end": standalone_start + len("分区级"),
                "confidence": 0.8,
            },
        ],
        chunk_metadata={},
    )

    assert [(item.surface, item.start, item.end) for item in result.mentions] == [
        ("分区级", standalone_start, standalone_start + len("分区级")),
        (
            "多分区级", content.index("多分区级"),
            content.index("多分区级") + len("多分区级"),
        ),
    ]
    assert any(
        item["reason"] == "nested_coordinated_item_fragment"
        for item in result.rejected
    )


def test_coordinated_distinct_recovery_requires_middle_item_boundaries():
    content = "城市甲和城市甲产业园共同参与。"
    start = content.index("城市甲")
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[{
            "name": "城市甲", "type": "Location",
            "mention_start": start, "mention_end": start + len("城市甲"),
        }],
        chunk_metadata={},
    )

    assert [(item.surface, item.start) for item in result.mentions] == [
        ("城市甲", start),
        ("城市甲", content.index("城市甲产业园")),
    ]


def test_generic_qualified_and_residual_fragments_are_rejected_but_typed_mentions_remain():
    content = "甲组织使用乙技术。丙设备应用于丁生产线。"
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    alpha = content.index("甲组织")
    beta = content.index("乙技术")
    gamma = content.index("丙设备")
    qualified = content.index("丙设备应用于丁生产")
    tail = content.index("线")
    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=[
            {
                "name": "甲组织", "type": "Organization",
                "mention_start": alpha, "mention_end": alpha + len("甲组织"),
            },
            {
                "name": "乙技术", "type": "Technology",
                "mention_start": beta, "mention_end": beta + len("乙技术"),
            },
            {
                "name": "丙设备", "type": "Product",
                "mention_start": gamma, "mention_end": gamma + len("丙设备"),
            },
            {
                "name": "丙设备应用于丁生产", "type": "GenericEntity",
                "mention_start": qualified,
                "mention_end": qualified + len("丙设备应用于丁生产"),
            },
            {
                "name": "线", "type": "GenericEntity",
                "mention_start": tail, "mention_end": tail + 1,
            },
        ],
        chunk_metadata={},
    )

    assert {item.surface for item in result.mentions} == {"甲组织", "乙技术", "丙设备"}
    assert {item["reason"] for item in result.rejected} >= {
        "nested_non_atomic_surface", "partial_context_fragment",
    }


def test_complete_proposition_argument_is_not_removed_by_nested_entity_gate():
    content = "设备数据采集中断导致告警。"
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=[
            {
                "name": "设备数据采集", "type": "Technology",
                "mention_start": 0, "mention_end": 6,
            },
            {
                "name": "设备数据采集中断", "type": "Event",
                "proposition_argument": True,
                "mention_start": 0, "mention_end": 8,
            },
        ],
        chunk_metadata={},
    )

    assert {item.surface for item in result.mentions} == {
        "设备数据采集", "设备数据采集中断",
    }


def test_same_source_anchor_deduplicates_physical_occurrence_across_overlapping_chunks():
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    raw = [{
        "name": "Canada",
        "type": "Location",
        "mention_start": 0,
        "mention_end": 6,
        "evidence_text": "Canada",
        "evidence_start": 0,
        "evidence_end": 6,
    }]
    left = extractor.normalize(
        project_scope="p",
        document_key="revision-1",
        chunk_key="chunk-left",
        content="Canada produced lithium.",
        raw_entities=raw,
        chunk_metadata={"source_anchor": "block-7", "chunk_source_start": 100},
    )
    right = extractor.normalize(
        project_scope="p",
        document_key="revision-1",
        chunk_key="chunk-right",
        content="Canada produced lithium.",
        raw_entities=raw,
        chunk_metadata={"source_anchor": "block-7", "chunk_source_start": 100},
    )

    assert left.mentions[0].mention_id != right.mentions[0].mention_id
    assert left.mentions[0].occurrence_key == right.mentions[0].occurrence_key


def test_scalar_temporal_contact_and_structural_noise_are_rejected():
    content = (
        "2024e 2025 年 5 月 12% +1 202 555 0123 domestic foreign 44.8 million "
        "220,000 metric tons Unit"
    )
    spans = {
        surface: (content.index(surface), content.index(surface) + len(surface))
        for surface in (
            "2024e",
            "2025 年 5 月",
            "12%",
            "+1 202 555 0123",
            "domestic",
            "foreign",
            "44.8 million",
            "220,000 metric tons",
            "Unit",
        )
    }
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[
            {
                "name": surface,
                "type": "Unit" if surface == "Unit" else "GenericEntity",
                "mention_start": start,
                "mention_end": end,
                "evidence_text": surface,
                "evidence_start": start,
                "evidence_end": end,
            }
            for surface, (start, end) in spans.items()
        ],
        chunk_metadata={},
    )

    assert result.mentions == []
    reasons = {item["reason"] for item in result.rejected}
    assert "scalar_or_temporal_noise" in reasons
    assert "non_referential_modifier" in reasons


def test_parser_mapped_table_header_is_rejected_but_table_row_stays_available():
    content = "Header Value\nBody Item"
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-structure-gate-v2")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[
            {
                "name": "Header", "type": "Concept",
                "mention_start": 0, "mention_end": 6,
            },
            {
                "name": "Body", "type": "Concept",
                "mention_start": 13, "mention_end": 17,
            },
        ],
        chunk_metadata={
            "structural_spans": [
                {"start": 0, "end": 12, "source_role": "table_header"},
                {"start": 13, "end": len(content), "source_role": "table_row"},
            ],
        },
    )

    assert [item.surface for item in result.mentions] == ["Body"]
    assert [item["reason"] for item in result.rejected] == ["structural_node"]


def test_zero_confidence_is_preserved_for_mention_type_and_alias():
    content = "International Business Machines (IBM)"
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[{
            "name": "International Business Machines",
            "type": "Organization",
            "mention_start": 0,
            "mention_end": 31,
            "evidence_text": "International Business Machines",
            "evidence_start": 0,
            "evidence_end": 31,
            "confidence": 0,
            "alias_spans": [{
                "surface": "IBM",
                "start": 33,
                "end": 36,
                "confidence": 0,
                "kind": "explicit_acronym",
            }],
        }],
        chunk_metadata={},
    )

    assert result.mentions[0].confidence == 0
    assert result.type_assertions[0].confidence == 0
    assert result.aliases[0].confidence == 0
    assert result.aliases[0].kind == "explicit_acronym"


def test_generic_boundary_rejects_function_words_and_incomplete_ascii_tokens():
    content = "The `if` statement is used for conditional execution."
    assert entity_boundary_reason(content, {
        "name": "for", "type": "GenericEntity",
        "mention_start": 27, "mention_end": 30,
    }) == "non_referential_function_word"
    assert entity_boundary_reason(content, {
        "name": "cond", "type": "GenericEntity",
        "mention_start": 31, "mention_end": 35,
    }) == "nonterminal_ascii_surface"


def test_generic_boundary_rejects_chinese_personal_pronoun_as_entity():
    content = "林岚来到北站。她把仪器交给周启。"
    start = content.index("她")
    raw = {
        "name": "她", "type": "Person",
        "mention_start": start, "mention_end": start + 1,
    }

    assert entity_boundary_reason(content, raw) == (
        "non_referential_function_word"
    )


def test_generic_boundary_rejects_enumeration_tail_as_entity():
    content = "系统提供任务管理、多核支持等功能。"
    start = content.index("等功能")
    raw = {
        "name": "等功能", "type": "GenericEntity",
        "mention_start": start, "mention_end": start + len("等功能"),
        "endpoint_candidate": True,
    }

    assert entity_boundary_reason(content, raw) == (
        "non_referential_enumeration_tail"
    )


def test_entity_span_repair_removes_sentence_punctuation_but_keeps_word_surface():
    content = "The `if` statement is used for conditional execution."
    extractor = SpanFirstEntityExtractor("entity-extraction-p2-shadow-v1")
    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c", content=content,
        raw_entities=[{
            "name": "conditional execution.", "type": "Event",
            "mention_start": 31, "mention_end": len(content),
            "evidence_text": content,
        }],
        chunk_metadata={},
    )
    assert [item.surface for item in result.mentions] == ["conditional execution"]
    assert (result.mentions[0].start, result.mentions[0].end) == (31, 52)


def test_grounded_inventory_keeps_nested_referential_noun_phrases():
    content = "The ground-water regions data set supports analysis."
    inner_start = content.index("ground-water regions")
    inner_end = inner_start + len("ground-water regions")
    outer_start = inner_start
    outer_end = outer_start + len("ground-water regions data set")
    inner = {
        "name": content[inner_start:inner_end],
        "type": "GenericEntity",
        "mention_start": inner_start,
        "mention_end": inner_end,
        "endpoint_candidate": True,
        "grounded_semantic_mention": True,
    }
    outer = {
        "name": content[outer_start:outer_end],
        "type": "GenericEntity",
        "mention_start": outer_start,
        "mention_end": outer_end,
        "endpoint_candidate": True,
        "grounded_semantic_mention": True,
    }

    assert entity_boundary_reason(content, inner, [inner, outer]) == ""
    assert entity_boundary_reason(content, outer, [inner, outer]) == ""


def test_grounded_inventory_repairs_entity_split_by_pdf_soft_wrap():
    content = (
        "主要包括任务管理、任务间通信、中断/异常\n"
        "管理、时钟管理和多核支持等功能。"
    )
    start = content.index("中断/异常")
    end = content.index("管理、时钟") + len("管理")
    extractor = SpanFirstEntityExtractor("entity-soft-wrap-v1")

    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c",
        content=content,
        raw_entities=[{
            "name": "中断/异常 管理",
            "type": "Technology",
            "mention_start": start,
            "mention_end": end,
            "endpoint_candidate": True,
            "grounded_semantic_mention": True,
        }],
        chunk_metadata={},
    )

    assert [item.surface for item in result.mentions] == [
        "中断/异常\n管理",
    ]
    assert result.rejected == []


def test_soft_wrap_coordination_restores_full_item_from_short_suffix():
    content = (
        "主要包括任务管理、任务间通信、中断/异常\n"
        "管理、时钟管理和多核支持等功能。"
    )
    short_start = content.index("管理、时钟")
    extractor = SpanFirstEntityExtractor("entity-soft-wrap-suffix-v1")

    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c",
        content=content,
        raw_entities=[{
            "name": "管理", "type": "GenericEntity",
            "mention_start": short_start,
            "mention_end": short_start + len("管理"),
            "endpoint_candidate": True,
        }],
        chunk_metadata={},
    )

    assert [item.surface for item in result.mentions] == [
        "中断/异常\n管理",
    ]
    assert any(
        item["reason"] == "nested_coordinated_item_fragment"
        for item in result.rejected
    )


def test_complete_grounded_compound_wins_over_nested_cue_fragment():
    content = "该内核提供存储管理和多核支持等功能。"
    full_start = content.index("多核支持")
    fragment_start = content.index("多核")
    extractor = SpanFirstEntityExtractor("entity-compound-v1")

    result = extractor.normalize(
        project_scope="p", document_key="d", chunk_key="c",
        content=content,
        raw_entities=[
            {
                "name": "多核支持", "type": "GenericEntity",
                "mention_start": full_start,
                "mention_end": full_start + len("多核支持"),
                "endpoint_candidate": True,
                "grounded_semantic_mention": True,
            },
            {
                "name": "多核", "type": "GenericEntity",
                "mention_start": fragment_start,
                "mention_end": fragment_start + len("多核"),
                "endpoint_candidate": True,
            },
        ],
        chunk_metadata={},
    )

    assert [item.surface for item in result.mentions] == ["多核支持"]
