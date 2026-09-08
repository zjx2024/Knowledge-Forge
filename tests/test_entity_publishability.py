from __future__ import annotations

from contextlib import contextmanager

from knowledge_forge.database import Database
from knowledge_forge.entity_publishability import (
    ACCEPTED,
    CONDITIONAL,
    REJECTED,
    aggregate_canonical_publishability,
    bibliography_entry_context,
    bibliography_reference_spans,
    canonical_display_surface,
    classify_mention_publishability,
    project_publishable_snapshot,
)
from knowledge_forge.relation_semantics import (
    RELATION_SEMANTIC_PROJECTION_VERSION,
    converge_relation_display_labels,
    normalized_relation_family_key,
    open_relation_display_expression,
    relation_publication_decision,
    relation_semantic_projection,
    source_expression_matches_schema_cues,
    suppress_exact_open_schema_duplicates,
)
from knowledge_forge.span_entities import SpanFirstEntityExtractor


def test_bibliographic_identifier_is_rejected_even_if_endpoint_candidate():
    content = "DOI： 10.19678/j.issn.1000-3428.0260167"
    decision = classify_mention_publishability(
        surface=content,
        entity_type="GenericEntity",
        content=content,
        start=0,
        end=len(content),
        chunk_metadata={"structural_role": "body"},
        raw={"endpoint_candidate": True},
    )

    assert decision.decision == REJECTED
    assert decision.reason == "document_metadata_identifier"

    label_decision = classify_mention_publishability(
        surface="DOI",
        entity_type="GenericEntity",
        content=content,
        start=0,
        end=3,
        chunk_metadata={"structural_role": "body"},
        raw={"endpoint_candidate": True},
    )
    assert label_decision.decision == REJECTED
    assert label_decision.reason == "document_metadata_identifier"


def test_standalone_doi_url_is_rejected_as_document_locator():
    content = "https://doi.org/10.19678/j.issn.1000-3428.0260167"
    decision = classify_mention_publishability(
        surface=content,
        entity_type="GenericEntity",
        content=content,
        start=0,
        end=len(content),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )

    assert decision.decision == REJECTED
    assert decision.reason == "standalone_external_identifier"

    serials = "ISSN 1000-3428, CN 31-1289/TP"
    serial_surface = "CN 31-1289/TP"
    serial_start = serials.index(serial_surface)
    serial_decision = classify_mention_publishability(
        surface=serial_surface,
        entity_type="GenericEntity",
        content=serials,
        start=serial_start,
        end=serial_start + len(serial_surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert serial_decision.decision == REJECTED
    assert serial_decision.reason == "standalone_external_identifier"


def test_publication_identifier_inside_prose_remains_source_provenance():
    content = "The registered publication uses ISSN 2096-4188 for its online edition."
    surface = "ISSN 2096-4188"
    start = content.index(surface)

    decision = classify_mention_publishability(
        surface=surface,
        entity_type="GenericEntity",
        content=content,
        start=start,
        end=start + len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )

    assert decision.decision == REJECTED
    assert decision.reason == "external_identifier_is_provenance"

    cn_surface = "CN 31-1289/TP"
    cn_content = f"The print edition is registered as {cn_surface}."
    cn_start = cn_content.index(cn_surface)
    cn_decision = classify_mention_publishability(
        surface=cn_surface,
        entity_type="GenericEntity",
        content=cn_content,
        start=cn_start,
        end=cn_start + len(cn_surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert cn_decision.decision == REJECTED
    assert cn_decision.reason == "external_identifier_is_provenance"


def test_bibliography_context_filters_citations_not_numbered_requirements():
    citation = "[12] Zhang Q. A method[J]. Journal, 2024, 29(1): 10-18."
    requirement = "[12] Scheduler initializes the timer before dispatch."
    assert bibliography_entry_context(citation) is True
    assert bibliography_entry_context(requirement) is False

    surface = "A method"
    start = citation.index(surface)
    decision = classify_mention_publishability(
        surface=surface,
        entity_type="TechnologyOrMethod",
        content=citation,
        start=start,
        end=start + len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert decision.decision == REJECTED
    assert decision.reason == "bibliography_entry_context"

    wrapped = (
        "正文结论。\n参考文献\n[1] Zhang Q.\nA method[J].\n"
        "Journal, 2024, 29(1): 10-18."
    )
    spans = bibliography_reference_spans(wrapped)
    assert spans == ((wrapped.index("参考文献"), len(wrapped)),)


def test_document_metadata_label_is_filtered_but_value_is_retained():
    content = "关键词：联邦学习；隐私保护"
    label = classify_mention_publishability(
        surface="关键词",
        entity_type="GenericEntity",
        content=content,
        start=0,
        end=3,
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    value_start = content.index("联邦学习")
    value = classify_mention_publishability(
        surface="联邦学习",
        entity_type="TechnologyOrMethod",
        content=content,
        start=value_start,
        end=value_start + len("联邦学习"),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert label.decision == REJECTED
    assert label.reason == "document_metadata_field_label"
    assert value.decision == ACCEPTED


def test_embedded_and_bracketed_metadata_labels_do_not_bypass_filter():
    fixtures = (
        ("【关键词】联邦学习；隐私保护", "关键词"),
        ("【中图分类号】R197.323；TP393.0", "中图分类号"),
        ("通信作者：张三, Email: author@example.org", "Email"),
    )
    for content, surface in fixtures:
        start = content.index(surface)
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=content,
            start=start,
            end=start + len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == REJECTED
        assert decision.reason == "document_metadata_field_label"


def test_identifier_shaped_metadata_value_is_provenance_only_in_field_context():
    content = "项目资助（SZSM202311031）"
    surface = "SZSM202311031"
    start = content.index(surface)
    metadata_value = classify_mention_publishability(
        surface=surface,
        entity_type="GenericEntity",
        content=content,
        start=start,
        end=start + len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    ordinary_content = "控制器 SZSM202311031 执行诊断任务"
    ordinary_start = ordinary_content.index(surface)
    ordinary_code = classify_mention_publishability(
        surface=surface,
        entity_type="ProductOrMaterial",
        content=ordinary_content,
        start=ordinary_start,
        end=ordinary_start + len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )

    assert metadata_value.decision == REJECTED
    assert metadata_value.reason == "document_metadata_identifier"
    assert ordinary_code.decision == ACCEPTED


def test_typed_function_word_is_rejected_and_cjk_display_wrap_is_repaired():
    decision = classify_mention_publishability(
        surface="在",
        entity_type="TechnologyOrMethod",
        content="数据在本地处理",
        start=2,
        end=3,
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert decision.decision == REJECTED
    assert decision.reason == "non_referential_function_word"
    assert canonical_display_surface("各参与方在本地数据集训练模 型") == (
        "各参与方在本地数据集训练模型"
    )


def test_page_only_renderer_heading_is_rejected():
    content = "# 第 39 页"
    surface = "第 39 页"
    start = content.index(surface)

    decision = classify_mention_publishability(
        surface=surface,
        entity_type="GenericEntity",
        content=content,
        start=start,
        end=start + len(surface),
        chunk_metadata={"section_title": surface},
        raw={},
    )

    assert decision.decision == REJECTED
    assert decision.reason == "renderer_page_heading"


def test_media_layout_phrase_is_rejected_but_domain_phrase_is_not():
    content = (
        "图片描述: The image displays a mostly blank white page with a "
        "single horizontal line near the top."
    )
    layout = "mostly blank white page"
    layout_start = content.index(layout)
    rejected = classify_mention_publishability(
        surface=layout,
        entity_type="GenericEntity",
        content=content,
        start=layout_start,
        end=layout_start + len(layout),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    useful_content = "图片描述: The diagram connects Controller-A to Valve-B."
    useful = "Controller-A"
    useful_start = useful_content.index(useful)
    accepted = classify_mention_publishability(
        surface=useful,
        entity_type="ProductOrMaterial",
        content=useful_content,
        start=useful_start,
        end=useful_start + len(useful),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )

    assert rejected.decision == REJECTED
    assert rejected.reason == "media_layout_description"
    assert accepted.decision == ACCEPTED

    image_start = content.index("image")
    renderer_label = classify_mention_publishability(
        surface="image",
        entity_type="GenericEntity",
        content=content,
        start=image_start,
        end=image_start + len("image"),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert renderer_label.decision == REJECTED
    assert renderer_label.reason == "media_renderer_label"


def test_renderer_image_text_keeps_useful_ocr_content():
    content = "图片文字: Controller-A 连接 Valve-B"
    surface = "Controller-A"
    start = content.index(surface)

    decision = classify_mention_publishability(
        surface=surface,
        entity_type="ProductOrMaterial",
        content=content,
        start=start,
        end=start + len(surface),
        chunk_metadata={"structural_role": "image"},
        raw={"endpoint_candidate": True},
    )

    assert decision.decision == ACCEPTED


def test_renderer_generated_media_headings_are_not_graph_entities():
    for surface in ("图片中明确事实", "图片中明确关系", "图片文字"):
        content = f"{surface}:\n- Controller-A --connects--> Valve-B"
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="Event",
            content=content,
            start=0,
            end=len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == REJECTED
        assert decision.reason == "media_renderer_label"


def test_renderer_visual_encoding_tokens_are_rejected_only_in_media_chunks():
    content = (
        "图片中明确关系:\n- C2 --represented_by--> Black square markers\n"
        "- x_axis --plotted_on--> Chart"
    )
    for surface in ("Black square markers", "x_axis", "Chart"):
        start = content.index(surface)
        media = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=content,
            start=start,
            end=start + len(surface),
            chunk_metadata={"block_types": ["image"]},
            raw={"endpoint_candidate": True},
        )
        ordinary = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=f"The configuration key is {surface}.",
            start=len("The configuration key is "),
            end=len("The configuration key is ") + len(surface),
            chunk_metadata={"block_types": ["paragraph"]},
            raw={"endpoint_candidate": True},
        )
        assert media.decision == REJECTED
        assert media.reason == "media_visual_encoding_token"
        assert ordinary.decision != REJECTED


def test_renderer_chart_series_and_line_tokens_are_media_only_noise():
    content = (
        "图片描述: A line chart contains Series C2, shown as a red line."
    )
    for surface in ("line chart", "Series C2", "red line"):
        start = content.index(surface)
        media = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=content,
            start=start,
            end=start + len(surface),
            chunk_metadata={"block_types": ["image"]},
            raw={"endpoint_candidate": True},
        )
        ordinary_content = f"The glossary defines {surface}."
        ordinary_start = ordinary_content.index(surface)
        ordinary = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=ordinary_content,
            start=ordinary_start,
            end=ordinary_start + len(surface),
            chunk_metadata={"block_types": ["paragraph"]},
            raw={"endpoint_candidate": True},
        )
        assert media.decision == REJECTED
        assert media.reason == "media_visual_encoding_token"
        assert ordinary.decision != REJECTED


def test_equation_locators_and_discretionary_hyphen_fragments_are_rejected():
    for surface in ("式( )", "式 ( 11 )", "Equation 4"):
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=surface,
            start=0,
            end=len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == REJECTED
        assert decision.reason == "document_equation_locator"

    content = "Uncertainty-aware Collabo⁃\nSeg improves stability."
    surface = "Uncertainty-aware Collabo⁃"
    decision = classify_mention_publishability(
        surface=surface,
        entity_type="Product",
        content=content,
        start=0,
        end=len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert decision.decision == REJECTED
    assert decision.reason == "soft_wrap_truncated_surface"

    # A converter may split the block immediately after the discretionary
    # hyphen, leaving no newline or right-hand continuation in this chunk.
    terminal_surface = "Uncertainty-aware Collabo⁃"
    terminal = classify_mention_publishability(
        surface=terminal_surface,
        entity_type="Product",
        content=terminal_surface,
        start=0,
        end=len(terminal_surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert terminal.decision == REJECTED
    assert terminal.reason == "soft_wrap_truncated_surface"

    complete_identifier = "COVID-19"
    complete = classify_mention_publishability(
        surface=complete_identifier,
        entity_type="GenericEntity",
        content=complete_identifier,
        start=0,
        end=len(complete_identifier),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert complete.decision != REJECTED


def test_figure_caption_plot_scaffold_is_not_an_entity():
    content = "Figure 4 Plot of average Dice versus lambda."
    start = content.index("Plot")
    decision = classify_mention_publishability(
        surface="Plot",
        entity_type="GenericEntity",
        content=content,
        start=start,
        end=start + len("Plot"),
        chunk_metadata={"block_types": ["paragraph"]},
        raw={"endpoint_candidate": True},
    )

    assert decision.decision == REJECTED
    assert decision.reason == "figure_caption_scaffold"


def test_unresolved_deictic_references_are_not_published_as_entities():
    for surface in (
        "二者", "三者", "前者", "双方", "以上这些方法", "各客户端间",
        "these", "both",
    ):
        content = f"甲与乙中，{surface}形成联系。"
        start = content.index(surface)
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=content,
            start=start,
            end=start + len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == REJECTED
        assert decision.reason == "unresolved_deictic_reference"


def test_source_locators_citation_fragments_and_formula_operators_are_filtered():
    fixtures = (
        ("Figure 4", "Figure 4  Plot of average score", "document_figure_table_locator"),
        ("20 - 23 ]", "[ 20 - 23 ] data differences", "citation_layout_token"),
        ("a r g m i n", "w = a r g m i n f(w)", "ocr_spaced_formula_operator"),
        ("‖ ‖ a -b", "其中， ‖ ‖ a -b 表示欧氏距离", "formula_expression_surface"),
        ("j", "∑ j = 1 k W i , j = 1", "formula_index_symbol"),
    )
    for surface, content, reason in fixtures:
        start = content.index(surface)
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="GenericEntity",
            content=content,
            start=start,
            end=start + len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == REJECTED
        assert decision.reason == reason


def test_source_author_self_reference_and_soft_wrap_suffix_are_not_entities():
    for surface in ("本文", "this study"):
        content = f"{surface} proposes a method"
        result = SpanFirstEntityExtractor("test").normalize(
            project_scope="p",
            document_key="d",
            chunk_key="c",
            content=content,
            raw_entities=[{
                "surface": surface,
                "type": "Organization",
                "mention_start": 0,
                "mention_end": len(surface),
                "endpoint_candidate": True,
            }],
            chunk_metadata={},
        )
        assert result.mentions == []
        assert result.rejected[0]["reason"] == (
            "non_referential_function_word"
            if surface == "本文" else "non_referential_context_placeholder"
        )

    content = "分布式机器学习方\n法，支持多个站点"
    start = content.index("法")
    result = SpanFirstEntityExtractor("test").normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[{
            "surface": "法", "type": "GenericEntity",
            "mention_start": start, "mention_end": start + 1,
            "endpoint_candidate": True,
        }],
        chunk_metadata={},
    )
    assert result.mentions == []
    assert result.rejected[0]["reason"] == "partial_context_fragment"


def test_renderer_field_words_in_ordinary_prose_are_not_filtered():
    content = "Chunk Manager schedules transfer operations."
    surface = "Chunk Manager"
    decision = classify_mention_publishability(
        surface=surface,
        entity_type="ProductOrMaterial",
        content=content,
        start=0,
        end=len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )

    assert decision.decision == ACCEPTED


def test_list_table_and_relation_endpoints_are_recall_protected():
    content = "ACoreOSMP653 核心操作系统\n— 分区管理\n— 任务管理"
    surface = "分区管理"
    start = content.index(surface)
    decision = classify_mention_publishability(
        surface=surface,
        entity_type="TechnologyOrMethod",
        content=content,
        start=start,
        end=start + len(surface),
        chunk_metadata={
            "structural_spans": [{
                "start": content.index("— 分区管理"),
                "end": content.index("— 分区管理") + len("— 分区管理"),
                "source_role": "list_item",
            }],
        },
        raw={"endpoint_candidate": True},
    )

    assert decision.decision == ACCEPTED
    assert decision.protected is True
    assert "eligible_source_structure" in decision.features


def test_short_unconnected_generic_is_conditional_not_destroyed():
    decision = classify_mention_publishability(
        surface="熵",
        entity_type="GenericEntity",
        content="熵",
        start=0,
        end=1,
        chunk_metadata={},
        raw={},
    )

    assert decision.decision == CONDITIONAL
    assert decision.reason == "short_generic_surface"


def test_finite_clauses_are_withheld_independently_of_model_type_or_format():
    fixtures = (
        ("服务可以处理记录并输出结果", "GenericEntity", {}),
        ("未来需建立统一的数据标准", "TechnologyOrMethod", {}),
        ("也能有效地进行模型训练", "Event", {}),
        ("仅通过共享参数来完成训练", "Event", {}),
        ("实现了安全的数据共享", "Event", {}),
        ("The service can process records securely", "ProductOrMaterial", {}),
    )
    for surface, entity_type, metadata in fixtures:
        decision = classify_mention_publishability(
            surface=surface,
            entity_type=entity_type,
            content=surface,
            start=0,
            end=len(surface),
            chunk_metadata=metadata,
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == CONDITIONAL, surface
        assert decision.reason == "clause_like_surface", surface

    list_clause = "- 系统必须保留审计日志"
    surface = "系统必须保留审计日志"
    start = list_clause.index(surface)
    decision = classify_mention_publishability(
        surface=surface,
        entity_type="GenericEntity",
        content=list_clause,
        start=start,
        end=start + len(surface),
        chunk_metadata={"structural_role": "list_item"},
        raw={"endpoint_candidate": True},
    )
    assert decision.decision == CONDITIONAL
    assert decision.reason == "clause_like_surface"


def test_continuation_clauses_are_withheld_but_noun_phrases_remain_entities():
    content = "调整参数，并结合适当的访问策略"
    clause = "并结合适当的访问策略"
    start = content.index(clause)
    decision = classify_mention_publishability(
        surface=clause,
        entity_type="GenericEntity",
        content=content,
        start=start,
        end=start + len(clause),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert decision.decision == CONDITIONAL
    assert decision.reason == "clause_like_surface"

    for surface, entity_type in (
        ("设备数据采集中断", "Event"),
        ("访问控制策略", "TechnologyOrMethod"),
        ("并行计算的优化策略", "TechnologyOrMethod"),
        ("按需加载机制", "TechnologyOrMethod"),
        ("用户所需资源", "GenericEntity"),
        ("Data Processing Service for Records", "ProductOrMaterial"),
    ):
        content = "前项，" + surface
        start = content.index(surface)
        decision = classify_mention_publishability(
            surface=surface,
            entity_type=entity_type,
            content=content,
            start=start,
            end=start + len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == ACCEPTED, surface


def test_finite_relational_frames_and_soft_wrap_subclauses_are_withheld():
    for surface in (
        "联邦学习通常分为横向联邦学习",
        "频繁的模型更新传输受限于设备数据量与通信延迟差异",
    ):
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="TechnologyOrMethod",
            content=surface,
            start=0,
            end=len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == CONDITIONAL, surface
        assert decision.reason == "clause_like_surface", surface

    content = "联邦学习在确保数据传\n输和模型训练安全的同时实现共享"
    surface = "联邦学习在确保数据传\n输"
    decision = classify_mention_publishability(
        surface=surface,
        entity_type="TechnologyOrMethod",
        content=content,
        start=0,
        end=len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True},
    )
    assert decision.decision == CONDITIONAL
    assert decision.reason == "clause_like_surface"


def test_long_finite_claims_are_relations_not_entity_nodes():
    fixtures = (
        "现有方法主要通过全局协同来建模客户端关系",
        "邻域聚合通过协作图强化局部协作以适应数据异质性",
        "低时间开销可避免因计算延迟导致的协作卡顿",
    )
    for surface in fixtures:
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="Event",
            content=surface,
            start=0,
            end=len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True, "proposition_argument": True},
        )
        assert decision.decision == CONDITIONAL
        assert decision.reason == "clause_like_surface"


def test_nominalized_finite_frames_are_not_mistaken_for_assertions():
    for surface in (
        "将任务分为多个阶段的方法",
        "受限于带宽的通信机制",
        "系统在设备中的部署方案",
    ):
        decision = classify_mention_publishability(
            surface=surface,
            entity_type="TechnologyOrMethod",
            content=surface,
            start=0,
            end=len(surface),
            chunk_metadata={},
            raw={"endpoint_candidate": True},
        )
        assert decision.decision == ACCEPTED, surface


def test_explicit_proposition_argument_keeps_complete_event_identity():
    surface = "系统可以在故障后恢复"
    decision = classify_mention_publishability(
        surface=surface,
        entity_type="Event",
        content=surface,
        start=0,
        end=len(surface),
        chunk_metadata={},
        raw={"endpoint_candidate": True, "proposition_argument": True},
    )
    assert decision.decision == ACCEPTED


def test_span_normalizer_applies_publishability_before_persistence():
    content = "DOI： 10.19678/j.issn.1000-3428.0260167"
    extractor = SpanFirstEntityExtractor("entity-extraction-test")
    result = extractor.normalize(
        project_scope="p",
        document_key="d",
        chunk_key="c",
        content=content,
        raw_entities=[{
            "name": content,
            "type": "GenericEntity",
            "mention_start": 0,
            "mention_end": len(content),
            "endpoint_candidate": True,
        }],
        chunk_metadata={"structural_role": "body"},
    )

    assert result.mentions == []
    assert result.rejected[0]["reason"] == "document_metadata_identifier"


def test_canonical_audit_promotes_only_verified_relation_backed_conditionals():
    canonicals = [
        {"canonical_key": "a", "metadata": {}},
        {"canonical_key": "b", "metadata": {}},
        {"canonical_key": "c", "metadata": {}},
        {"canonical_key": "d", "metadata": {}},
    ]
    mentions = [
        {
            "mention_key": "ma", "filter_decision": "accepted",
            "document_id": 1, "chunk_id": 1, "provenance": {},
        },
        {
            "mention_key": "mb", "filter_decision": "conditional",
            "filter_reason": "clause_like_surface",
            "document_id": 1, "chunk_id": 2, "provenance": {},
        },
        {
            "mention_key": "mc", "filter_decision": "conditional",
            "filter_reason": "short_generic_surface",
            "document_id": 1, "chunk_id": 3, "provenance": {},
        },
        {
            "mention_key": "md", "filter_decision": "accepted",
            "document_id": 1, "chunk_id": 4, "provenance": {},
        },
    ]
    assignments = [
        {"mention_key": "ma", "canonical_key": "a"},
        {"mention_key": "mb", "canonical_key": "b"},
        {"mention_key": "mc", "canonical_key": "c"},
        {"mention_key": "md", "canonical_key": "d"},
    ]
    endpoints = [{
        "relation_key": "r1",
        "source_canonical_key": "a",
        "target_canonical_key": "c",
    }]

    rows, metrics = aggregate_canonical_publishability(
        canonicals=canonicals,
        assignments=assignments,
        mentions=mentions,
        relation_endpoints=endpoints,
    )
    by_key = {row["canonical_key"]: row["audit"] for row in rows}

    assert by_key["a"]["decision"] == "accepted"
    assert by_key["b"]["decision"] == "withheld"
    assert by_key["b"]["graph_publishable"] is False
    assert by_key["c"]["decision"] == "relation_backed"
    assert by_key["c"]["graph_publishable"] is True
    assert by_key["d"]["retrieval_eligible"] is True
    assert by_key["d"]["graph_publishable"] is False
    assert metrics["relation_backed_promotions"] == 1

    snapshot = project_publishable_snapshot({
        "canonicals": [
            {**item, "metadata": {"entity_publishability": by_key[item["canonical_key"]]}}
            for item in canonicals
        ],
        "assignments": assignments,
        "relation_endpoints": endpoints,
    }, mentions=mentions)
    assert {item["canonical_key"] for item in snapshot["canonicals"]} == {"a", "c"}
    assert {item["mention_key"] for item in snapshot["mentions"]} == {"ma", "mc"}


def test_relation_projection_groups_inflections_without_guessing_synonyms():
    assert normalized_relation_family_key("checks") == "check"
    assert normalized_relation_family_key("check") == "check"

    open_projection = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="checks",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_evidence_text": "Alpha checks Beta.",
        },
    )
    precise_projection = relation_semantic_projection(
        relation_type="uses",
        source_expression="uses",
        open_relation_carrier=False,
        predicate_metadata={"publish_mode": "auto"},
    )

    assert open_projection["family_key"] == "open:check"
    assert open_projection["strict_inference_eligible"] is False
    assert precise_projection["family_key"] == "schema:use"
    assert precise_projection["strict_inference_eligible"] is True


def test_relation_projection_rejects_broken_open_labels_and_converges_variants():
    for expression in (
        "的特点及", "尤 在", "在保护", "是...的某种策略", "willingness",
        "粗且", "中来", "上均优于", "对其", "other", "other on", "Plot",
    ):
        projection = relation_semantic_projection(
            relation_type="grounded_carrier",
            source_expression=expression,
            open_relation_carrier=True,
        )
        assert relation_publication_decision(projection)["eligible"] is False
    for expression in ("出版地", "应用于", "作为"):
        projection = relation_semantic_projection(
            relation_type="grounded_carrier",
            source_expression=expression,
            open_relation_carrier=True,
            predicate_metadata={
                "source_frame_evidence_text": f"甲{expression}乙。",
            },
        )
        assert relation_publication_decision(projection)["eligible"] is True

    translated = relation_semantic_projection(
        relation_type="grounded_carrier",
        source_expression="contains",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_evidence_text": "医学图像数据包含隐私信息。",
        },
    )
    assert translated["publication"]["eligible"] is False
    assert translated["publication"]["reason"] == (
        "open_relation_expression_not_source_grounded"
    )

    cross_atomic = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="has value",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_atomic_scope_override": True,
            "source_structure_kind": "",
            "source_syntax_origin": "",
        },
    )
    assert cross_atomic["publication"] == {
        "eligible": False,
        "reason": "unsupported_cross_atomic_open_relation",
        "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
    }
    structured_cross_atomic = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="has value",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_atomic_scope_override": True,
            "source_structure_kind": "table_column_assignment",
        },
    )
    assert structured_cross_atomic["publication"]["eligible"] is True

    assert normalized_relation_family_key("authored") == "author"
    assert normalized_relation_family_key("is author of") == "author|of"
    assert normalized_relation_family_key("has author") == "author"
    rows = []
    for label, source_type, target_type in (
        ("authored", "Person", "Document"),
        ("has authored", "Person", "Document"),
        ("has author", "Document", "Person"),
    ):
        projection = relation_semantic_projection(
            relation_type="grounded_carrier",
            source_expression=label,
            open_relation_carrier=True,
        )
        rows.append({
            "relation_key": label,
            "metadata": {
                "display_relation_label": label,
                "source_relation_expression": label,
                "relation_semantics": projection,
                "publication_integrity": {
                    "source_entity_type": source_type,
                    "target_entity_type": target_type,
                },
            },
        })
    converged, metrics = converge_relation_display_labels(rows)
    assert [
        row["metadata"]["display_relation_label"] for row in converged[:2]
    ] == ["authored", "authored"]
    assert converged[2]["metadata"]["display_relation_label"] == "has author"
    assert converged[1]["metadata"]["source_relation_expression"] == (
        "has authored"
    )
    assert metrics["groups_converged"] == 1

    assert open_relation_display_expression("适用于各") == "适用于"

    achieve_rows = []
    for label in ("is to achieve", "achieve of"):
        projection = relation_semantic_projection(
            relation_type="grounded_carrier",
            source_expression=label,
            open_relation_carrier=True,
        )
        achieve_rows.append({
            "relation_key": label,
            "metadata": {
                "display_relation_label": label,
                "relation_semantics": projection,
                "publication_integrity": {
                    "source_entity_type": "Technology",
                    "target_entity_type": "Event",
                },
            },
        })
    achieve_rows, achieve_metrics = converge_relation_display_labels(
        achieve_rows
    )
    assert {
        row["metadata"]["display_relation_label"] for row in achieve_rows
    } == {"is to achieve", "achieve of"}
    assert achieve_metrics["groups_converged"] == 0


def test_schema_publication_requires_grounded_projection_and_type_contract():
    ungrounded = relation_semantic_projection(
        relation_type="located_in",
        source_expression="",
        open_relation_carrier=False,
        predicate_metadata={
            "source_relation_projection": False,
            "relation_candidate_origin": "pair_classifier",
            "source_entity_type": "Person",
            "target_entity_type": "Location",
            "schema_domain_types": ["*"],
            "schema_range_types": ["Location"],
        },
    )
    mismatch = relation_semantic_projection(
        relation_type="located_in",
        source_expression="deployed",
        open_relation_carrier=False,
        predicate_metadata={
            "source_relation_projection": True,
            "source_frame_evidence_text": "The server deployed a graph network.",
            "source_subject_surface": "server",
            "source_object_surface": "graph network",
            "source_entity_type": "Location",
            "target_entity_type": "Technology",
            "schema_domain_types": ["*"],
            "schema_range_types": ["Location"],
        },
    )
    grounded = relation_semantic_projection(
        relation_type="uses",
        source_expression="uses",
        open_relation_carrier=False,
        predicate_metadata={
            "source_relation_projection": True,
            "source_frame_evidence_text": "The system uses a database.",
            "source_subject_surface": "system",
            "source_object_surface": "database",
            "source_entity_type": "Product",
            "target_entity_type": "Technology",
            "schema_domain_types": ["*"],
            "schema_range_types": ["*"],
        },
    )

    assert ungrounded["publication"]["reason"] == (
        "schema_relation_without_grounded_source_expression"
    )
    assert mismatch["publication"]["reason"] == (
        "schema_endpoint_type_contract_mismatch"
    )
    assert grounded["publication"] == {
        "eligible": True,
        "reason": "grounded_schema_predicate",
        "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
    }


def test_schema_and_open_publication_share_boundary_and_media_guards():
    fixtures = (
        ("通过在", {}, "incomplete_functional_prefix_relation_expression"),
        ("过程中由于", {}, "truncated_prepositional_relation_expression"),
        ("其他", {}, "non_relational_marker_expression"),
        ("典型", {}, "non_relational_marker_expression"),
        ("进行", {}, "non_relational_marker_expression"),
        ("还是", {}, "non_relational_marker_expression"),
        ("如包含", {}, "non_relational_marker_expression"),
        ("之间", {}, "non_relational_marker_expression"),
        ("between", {}, "function_only_relation_expression"),
        ("中，客户端需要进行", {}, "sentence_shaped_relation_expression"),
        ("Figure 4 Plot of", {}, "figure_caption_relation_scaffold"),
        (
            "is plotted using",
            {"source_media_derived": True},
            "media_visual_encoding_relation",
        ),
    )
    for expression, extra, reason in fixtures:
        evidence = f"甲{expression}乙"
        projection = relation_semantic_projection(
            relation_type="uses",
            source_expression=expression,
            open_relation_carrier=False,
            predicate_metadata={
                "source_relation_projection": True,
                "source_frame_evidence_text": evidence,
                "source_subject_surface": "甲",
                "source_object_surface": "乙",
                "source_entity_type": "GenericEntity",
                "target_entity_type": "GenericEntity",
                "schema_domain_types": ["*"],
                "schema_range_types": ["*"],
                **extra,
            },
        )
        assert projection["publication"]["eligible"] is False
        assert projection["publication"]["reason"] == reason

    cross_atomic = relation_semantic_projection(
        relation_type="uses",
        source_expression="uses",
        open_relation_carrier=False,
        predicate_metadata={
            "source_relation_projection": True,
            "source_frame_evidence_text": "Alpha uses Beta.",
            "source_frame_atomic_scope_override": True,
            "source_entity_type": "Product",
            "target_entity_type": "Technology",
            "schema_domain_types": ["*"],
            "schema_range_types": ["*"],
        },
    )
    assert cross_atomic["publication"]["reason"] == (
        "unsupported_cross_atomic_schema_relation"
    )

    unscoped = relation_semantic_projection(
        relation_type="uses",
        source_expression="trains",
        open_relation_carrier=False,
        predicate_metadata={
            "source_relation_projection": True,
            "source_frame_evidence_text": "A tool trains a model.",
            "source_pair_same_atomic_scope": False,
            "source_entity_type": "Technology",
            "target_entity_type": "Product",
            "schema_domain_types": ["*"],
            "schema_range_types": ["*"],
        },
    )
    assert unscoped["publication"]["reason"] == (
        "source_projection_outside_atomic_scope"
    )


def test_schema_lexical_anchor_uses_only_versioned_cues():
    assert source_expression_matches_schema_cues(
        "会导致", ["causes", "导致"],
    )
    assert source_expression_matches_schema_cues(
        "is used by", ["uses"],
    )
    assert not source_expression_matches_schema_cues(
        "用于调控", ["uses", "使用"],
    )
    assert not source_expression_matches_schema_cues(
        "满足", ["depends on", "依赖"],
    )


def test_exact_open_duplicate_of_grounded_schema_edge_is_withheld():
    def row(key: str, layer: str, relation_type: str) -> dict:
        projection = {
            "layer": layer,
            "source_expression": "导致",
            "publication": {
                "eligible": True,
                "reason": "test",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            },
        }
        return {
            "relation_key": key,
            "source_canonical_key": "a",
            "target_canonical_key": "b",
            "relation_type": relation_type,
            "metadata": {
                "source_relation_expression": "导致",
                "relation_semantics": projection,
            },
        }

    rows, count = suppress_exact_open_schema_duplicates([
        row("schema", "schema_predicate", "causes"),
        row("open", "open_source_expression", "source_assertion"),
    ])

    assert count == 1
    assert rows[0]["metadata"]["relation_semantics"]["publication"][
        "eligible"
    ] is True
    assert rows[1]["metadata"]["relation_semantics"]["publication"] == {
        "eligible": False,
        "reason": "open_relation_duplicate_of_schema_source_expression",
        "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
    }


def test_relation_projection_repairs_locative_boundary_without_lexical_rules():
    projection = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="中采样",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_evidence_text": "从数据集中采样批次数据。",
            "source_subject_surface": "数据集",
            "source_object_surface": "批次数据",
            "source_syntax_origin": "intervening_predicate_construction",
        },
    )
    lexical = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="上报",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_evidence_text": "终端上报状态。",
            "source_subject_surface": "终端",
            "source_object_surface": "状态",
            "source_syntax_origin": "intervening_predicate_construction",
        },
    )
    soft_wrap_modal = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="支持",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_evidence_text": "系统应 支持功能。",
            "source_subject_surface": "系统",
            "source_object_surface": "功能",
            "source_syntax_origin": "intervening_predicate_construction",
        },
    )
    unicode_endpoint = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="取值范围",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_evidence_text": "λ 取值范围为 [0,1]。",
            "source_subject_surface": "λ 取值",
            "source_object_surface": "[0,1]",
        },
    )

    assert projection["source_expression"] == "中采样"
    assert projection["display_expression"] == "采样"
    assert projection["publication"]["eligible"] is True
    assert lexical["display_expression"] == "上报"
    assert lexical["publication"]["eligible"] is True
    assert soft_wrap_modal["publication"]["eligible"] is True
    assert unicode_endpoint["publication"]["eligible"] is True


def test_relation_projection_rejects_generic_boundary_and_renderer_artifacts():
    fixtures = (
        (
            "造用于优化",
            {
                "source_frame_evidence_text": (
                    "利用数据量占比、模型相似度与不确定性等因素来构 "
                    "造用于优化协作图的目标函数。"
                ),
                "source_subject_surface": "不确定性",
                "source_object_surface": "协作图",
                "source_syntax_origin": "intervening_predicate_construction",
            },
            "soft_wrap_relation_boundary_fragment",
        ),
        (
            "方法在",
            {
                "source_frame_evidence_text": "不确定性计算方法在医疗场景中的优势。",
                "source_subject_surface": "不确定性计算",
                "source_object_surface": "医疗场景",
                "source_syntax_origin": "intervening_predicate_construction",
            },
            "nominal_locative_fragment_relation_expression",
        ),
        (
            "可避免因",
            {
                "source_frame_evidence_text": "低时间开销可避免因部分客户端延迟造成卡顿。",
                "source_subject_surface": "低时间开销",
                "source_object_surface": "部分客户端",
                "source_syntax_origin": "intervening_predicate_construction",
            },
            "dangling_argument_marker_relation_expression",
        ),
        (
            "最大占用量",
            {
                "source_frame_evidence_text": "GPU 显存是计算过程中的最大占用量。",
                "source_subject_surface": "GPU 显存",
                "source_object_surface": "最大占用量",
            },
            "relation_expression_contains_endpoint_surface",
        ),
        (
            "cos",
            {
                "source_frame_evidence_text": "W cos(θ_i, θ_j) 是正则化项。",
                "source_subject_surface": "W",
                "source_object_surface": "θ_i",
                "source_syntax_origin": "intervening_predicate_construction",
            },
            "formula_operator_relation_expression",
        ),
    )
    for expression, metadata, expected_reason in fixtures:
        projection = relation_semantic_projection(
            relation_type="source_assertion",
            source_expression=expression,
            open_relation_carrier=True,
            predicate_metadata=metadata,
        )
        assert projection["publication"] == {
            "eligible": False,
            "reason": expected_reason,
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }


def test_rendered_edge_transport_is_reduced_to_its_semantic_label():
    projection = relation_semantic_projection(
        relation_type="source_assertion",
        source_expression="--calls-->",
        open_relation_carrier=True,
        predicate_metadata={
            "source_frame_evidence_text": "Service A --calls--> Service B",
            "source_subject_surface": "Service A",
            "source_object_surface": "Service B",
        },
    )

    assert projection["source_expression"] == "--calls-->"
    assert projection["display_expression"] == "calls"
    # The public runtime normalizes this transport syntax before projection;
    # direct raw projections still fail closed as a defence in depth.
    assert projection["publication"]["eligible"] is False
    assert projection["publication"]["reason"] == (
        "renderer_transport_relation_expression"
    )


def test_vector_candidate_sql_keeps_json_path_during_sql_composition():
    class FakeResult:
        def fetchall(self):
            return []

    class FakeConnection:
        def __init__(self):
            self.statements = []

        def execute(self, statement, _parameters=None):
            self.statements.append(
                statement.as_string()
                if hasattr(statement, "as_string") else str(statement)
            )
            return FakeResult()

    connection = FakeConnection()

    class TestDatabase(Database):
        def __init__(self):
            pass

        @contextmanager
        def connection(self):
            yield connection

        def _ensure_grounded_entity_hnsw_index(
            self, _connection, _model_name, _dimension,
        ):
            return None

    rows = TestDatabase().search_grounded_entity_vector_candidates(
        "namespace", "embedding-model", [1.0, 0.0], 5,
    )

    assert rows == []
    assert any(
        "{entity_publishability,retrieval_eligible}" in statement
        for statement in connection.statements
    )
