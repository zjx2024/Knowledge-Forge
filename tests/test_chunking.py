import pytest

from knowledge_forge.chunking import (
    Section,
    build_processing_result,
    extract_plain_sections,
    parse_document_structure,
    process_document,
    route_processing,
    split_sections,
    validate_processing_request,
)


def test_general_sentence_segmentation_uses_shared_exact_spans():
    from knowledge_forge.chunking import _segment_spans

    content = (
        "Version v1.2.3 costs 3.14 USD. Visit https://example.com/a.b and a.b@example.com. "
        "Dr. Smith said ready... 然后继续……完成！分块: 正文。"
    )
    spans = _segment_spans(content)
    assert "".join(content[start:end] for start, end, _ in spans) == content
    assert all(content[start:end] == value for start, end, value in spans)
    assert [value for _, _, value in spans] == [
        "Version v1.2.3 costs 3.14 USD. ",
        "Visit https://example.com/a.b and a.b@example.com. ",
        "Dr. Smith said ready... ",
        "然后继续……",
        "完成！",
        "分块: 正文。",
    ]


def test_structure_parsing_can_be_inspected_before_chunking(tmp_path):
    source = tmp_path / "staged.md"
    source.write_text("# 第一章\n\n正文内容。", encoding="utf-8")

    plan, parsed = parse_document_structure(source)

    assert plan.effective_parser == "native_text"
    assert [block.block_type for block in parsed.blocks] == ["heading", "paragraph"]

    result = build_processing_result(source, plan, parsed, source_name=source.name)
    assert result.chunks
    assert result.chunks[0].section_key == parsed.blocks[1].section_key


def test_split_keeps_heading_metadata():
    chunks = split_sections(
        [Section("s1", "第一章", 1, "甲。" * 100)],
        max_chars=180,
        overlap_chars=20,
        source_name="manual.txt",
    )

    assert len(chunks) > 1
    assert all(chunk.title == "第一章" for chunk in chunks)
    assert all("来源文档: manual.txt" in chunk.content for chunk in chunks)


def test_markdown_heading_sections(tmp_path):
    source = tmp_path / "sample.md"
    source.write_text("# 总则\n正文\n\n## 子节\n内容", encoding="utf-8")

    sections = extract_plain_sections(source)

    assert [item.title for item in sections] == ["总则", "子节"]


def test_incompatible_manual_parser_is_rejected_before_ingestion():
    with pytest.raises(ValueError, match="不能处理"):
        validate_processing_request("manual.docx", "spreadsheet", "general")


def test_csv_auto_routes_to_spreadsheet_table_chunks(tmp_path):
    source = tmp_path / "industry.csv"
    source.write_text("企业,产品,地区\n甲公司,芯片,上海\n乙公司,设备,苏州\n", encoding="utf-8")

    result = process_document(source, source_name=source.name)

    assert result.plan.effective_parser == "spreadsheet"
    assert result.plan.effective_strategy == "table"
    assert len(result.chunks) == 2
    assert "企业: 甲公司" in result.chunks[0].content
    assert result.chunks[0].metadata["chunk_role"] == "table_row"


def test_xlsx_preserves_sheet_headers_in_each_row_chunk(tmp_path):
    from openpyxl import Workbook

    source = tmp_path / "catalog.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "产品目录"
    sheet.append(["型号", "名称", "状态"])
    sheet.append(["A-01", "控制器", "在产"])
    workbook.save(source)

    result = process_document(source, source_name=source.name)

    assert result.plan.effective_parser == "spreadsheet"
    assert len(result.chunks) == 1
    assert "型号: A-01" in result.chunks[0].content
    assert result.chunks[0].metadata["table"]["sheet"] == "产品目录"


def test_general_strategy_builds_parent_and_child_chunks(tmp_path):
    source = tmp_path / "manual.md"
    source.write_text("# 操作说明\n\n" + "这是一个较长的操作步骤。" * 500, encoding="utf-8")

    result = process_document(source, requested_strategy="general", source_name=source.name)

    parents = [item for item in result.chunks if item.metadata.get("chunk_role") == "parent"]
    children = [item for item in result.chunks if item.metadata.get("chunk_role") == "child"]
    assert len(parents) == 1
    assert len(children) > 1
    assert all(item.parent_key == parents[0].key for item in children)
    assert parents[0].metadata["graph_extraction_enabled"] is False


def test_project_chunk_token_limit_is_a_hard_bound(tmp_path):
    source = tmp_path / "bounded.md"
    source.write_text("# 说明\n\n" + "知识图谱需要稳定的项目级分块上限。" * 300, encoding="utf-8")

    result = process_document(
        source,
        requested_strategy="general",
        source_name=source.name,
        chunk_max_tokens=128,
    )

    assert len(result.chunks) > 1
    assert all(chunk.token_count <= 128 for chunk in result.chunks)
    assert any(chunk.metadata.get("token_limit_split") for chunk in result.chunks)


def test_paper_references_remain_searchable_but_skip_graph_extraction(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text(
        "# 摘要\n研究内容。\n# 引言\n背景。\n# 研究方法\n方法。\n"
        "# 实验结果\n结果。\n# 参考文献\n[1] Example, 2026.",
        encoding="utf-8",
    )

    plan = route_processing(source)
    result = process_document(source, requested_strategy="paper", source_name=source.name)

    assert plan.effective_strategy == "paper"
    references = [item for item in result.chunks if item.metadata.get("paper_part") == "references"]
    assert references
    assert references[0].metadata["graph_extraction_enabled"] is False
