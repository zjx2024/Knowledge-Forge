from knowledge_forge.sentence_spans import (
    split_relation_scopes,
    split_source_sentences,
)


def test_relation_scopes_merge_latin_pdf_soft_wrap_but_keep_numbered_lines():
    content = (
        "2.\n"
        "These processes are requirements capture, conceptual\n"
        "design, implementation and production transition.\n"
        "3.\n"
    )

    scopes = [text for _start, _end, text in split_relation_scopes(content)]

    assert any(
        "conceptual\ndesign" in text
        and "production transition" in text
        for text in scopes
    )
    assert not any("2.\nThese processes" in text for text in scopes)
    assert not any("transition.\n3." in text for text in scopes)


def test_relation_scopes_merge_long_cjk_pdf_wrap_but_keep_figure_labels():
    prose = (
        "主要包括任务管理、任务间通信、中断/异常\n"
        "管理、时钟管理和多核支持等功能。"
    )
    figure = "应用任务R1\n应用任务R2\n应用任务R3"

    prose_scopes = split_relation_scopes(prose)
    figure_scopes = split_relation_scopes(figure)

    assert len(prose_scopes) == 1
    assert prose_scopes[0][2] == prose
    assert [text.strip() for _start, _end, text in figure_scopes] == [
        "应用任务R1", "应用任务R2", "应用任务R3",
    ]


def test_relation_scopes_never_merge_lists_tables_or_indented_code():
    content = "Heading:\n- first item\n| A | B |\n    code()"

    scopes = [text.strip() for _start, _end, text in split_relation_scopes(content)]

    assert scopes == ["Heading:", "- first item", "| A | B |", "code()"]


def test_source_sentences_keep_pdf_wrapped_clause_intact_with_exact_offsets():
    content = (
        "没有对多中心关系进行建模并利用多中心关系进行深度协作，导致\n"
        "共性知识挖掘不充分。\n- 新列表项"
    )

    spans = split_source_sentences(content)

    assert "导致\n共性知识" in spans[0][2]
    assert spans[0][2].endswith("。")
    assert spans[1][2].strip() == "- 新列表项"
    assert all(content[start:end] == value for start, end, value in spans)
