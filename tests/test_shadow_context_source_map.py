from __future__ import annotations

from knowledge_forge.shadow_context import (
    build_shadow_context_manifest,
    build_context_units_for_document,
    occurrence_scope,
    structural_spans_for_chunk,
)


def test_context_identity_and_manifest_are_runtime_namespace_invariant():
    document = {
        "id": 7,
        "active_revision_id": "revision-7",
        "doc_hash": "document-hash",
    }
    chunk = {
        "id": 17,
        "document_id": 7,
        "revision_id": "revision-7",
        "chunk_key": "chunk-17",
        "content": "Alpha acts on Beta.",
    }
    payload = {
        "parser_version": "parser-test",
        "blocks": [{
            "key": "block-1",
            "block_type": "paragraph",
            "text": chunk["content"],
        }],
    }

    manifests = [
        build_shadow_context_manifest(
            namespace_id=namespace_id,
            documents=[document],
            chunks_by_document={7: [chunk]},
            payload_loader=lambda _document: payload,
        )
        for namespace_id in ("entity-stage", "relation-stage")
    ]

    assert manifests[0].artifact_hash == manifests[1].artifact_hash
    assert (
        manifests[0].context_units[0].context_id
        == manifests[1].context_units[0].context_id
    )
    assert (
        manifests[0].context_units[0].namespace_id
        != manifests[1].context_units[0].namespace_id
    )


def test_context_units_map_rendered_text_to_parser_blocks() -> None:
    payload = {
        "parser_version": "parser-v1",
        "blocks": [{
            "key": "block-p1",
            "block_type": "paragraph",
            "text": "Alpha uses Beta.",
            "order": 1,
            "section_key": "s1",
            "section_title": "Overview",
            "heading_path": ["Overview"],
            "page": 2,
            "metadata": {"block_local_start": 100, "block_local_end": 116},
        }],
    }

    units = build_context_units_for_document(
        namespace_id="ns",
        document={"id": 1, "active_revision_id": "rev-1"},
        chunks=[],
        processing_payload=payload,
    )

    assert len(units) == 1
    unit = units[0]
    assert unit.context_kind == "paragraph"
    assert unit.rendered_text == "Alpha uses Beta."
    assert unit.source_block_keys == ("block-p1",)
    assert unit.structure_quality == "structured"
    source = unit.source_map[0]
    assert (source.rendered_start, source.rendered_end) == (0, 16)
    assert (source.block_local_start, source.block_local_end) == (100, 116)
    assert source.block_key == "block-p1"
    assert source.page == 2


def test_legacy_payload_without_blocks_falls_back_to_chunk_local_source_map():
    units = build_context_units_for_document(
        namespace_id="ns",
        document={"id": 1, "active_revision_id": "rev-1"},
        chunks=[{
            "id": 10,
            "revision_id": "rev-1",
            "chunk_key": "chunk-10",
            "section_title": "Legacy",
            "content": "Legacy chunk text.",
        }],
        processing_payload={"blocks": []},
    )

    assert len(units) == 1
    unit = units[0]
    assert unit.context_kind == "legacy_chunk"
    assert unit.structure_quality == "legacy_chunk_local"
    assert unit.source_map[0].anchor_kind == "legacy_chunk_local"
    assert unit.source_map[0].block_key == "chunk-10"
    assert unit.structure["source_role"] == "body"
    assert unit.structure["relation_eligible"] is True


def test_parser_owned_header_role_is_ineligible_even_if_model_claims_body() -> None:
    profile = occurrence_scope(
        content="Quarterly Summary",
        mention_start=0,
        mention_end=9,
        structure={"block_type": "paragraph"},
        context_kind="paragraph",
        metadata={"structural_role": "page_header"},
    )

    assert profile["source_role"] == "page_header"
    assert profile["relation_eligible"] is False
    assert profile["occurrence_scope"]["scope_kind"] == "sentence"


def test_chunk_mapped_context_keeps_table_headers_out_of_table_row_scopes() -> None:
    table = "| Region | Value |\n| --- | --- |\n| North | Item |"
    units = build_context_units_for_document(
        namespace_id="ns",
        document={"id": 1, "active_revision_id": "rev-1"},
        chunks=[{
            "id": 10,
            "revision_id": "rev-1",
            "chunk_key": "chunk-10",
            "section_title": "Data",
            "content": "Source\n" + table,
        }],
        processing_payload={
            "parser_version": "parser-v1",
            "blocks": [{
                "key": "table-1",
                "block_type": "table",
                "text": table,
                "order": 1,
                "metadata": {},
            }],
        },
    )

    unit = units[0]
    content = unit.rendered_text
    header = occurrence_scope(
        content=content,
        mention_start=content.index("Region"),
        mention_end=content.index("Region") + len("Region"),
        structure=unit.structure,
        context_kind=unit.context_kind,
    )
    row = occurrence_scope(
        content=content,
        mention_start=content.index("North"),
        mention_end=content.index("North") + len("North"),
        structure=unit.structure,
        context_kind=unit.context_kind,
    )

    assert header["source_role"] == "table_header"
    assert header["relation_eligible"] is False
    assert row["source_role"] == "table_row"
    assert row["relation_eligible"] is True


def test_pipe_table_without_markdown_delimiter_keeps_first_data_row() -> None:
    table = (
        "| Product | Source | Purpose |\n"
        "| Separator | Resin | Assembly |\n"
        "| Coating | Powder | Protection |"
    )
    units = build_context_units_for_document(
        namespace_id="ns",
        document={"id": 1, "active_revision_id": "rev-1"},
        chunks=[{
            "id": 10,
            "revision_id": "rev-1",
            "chunk_key": "chunk-10",
            "content": table,
        }],
        processing_payload={
            "parser_version": "parser-v1",
            "blocks": [{
                "key": "table-1",
                "block_type": "table",
                "text": table,
                "order": 1,
                "metadata": {},
            }],
        },
    )

    spans = units[0].structure["structural_spans"]
    assert [item["source_role"] for item in spans] == [
        "table_header", "table_row", "table_row",
    ]
    separator_start = table.index("Separator")
    profile = occurrence_scope(
        content=table,
        mention_start=separator_start,
        mention_end=separator_start + len("Separator"),
        structure=units[0].structure,
        context_kind=units[0].context_kind,
    )
    assert profile["source_role"] == "table_row"
    assert profile["relation_eligible"] is True


def test_structured_chunk_uses_sentence_scopes_and_recovers_split_list_lines() -> None:
    content = (
        "Alpha is documented. Beta is separate.\n"
        "- Left endpoint --alpha_rel--> Right endpoint"
    )
    units = build_context_units_for_document(
        namespace_id="ns",
        document={"id": 1, "active_revision_id": "rev-1"},
        chunks=[{
            "id": 10,
            "revision_id": "rev-1",
            "chunk_key": "chunk-10",
            "content": content,
        }],
        # The parser block is intentionally larger than this chunk, which is
        # how a chunk boundary can prevent whole-block exact matching.
        processing_payload={
            "parser_version": "parser-v1",
            "blocks": [{
                "key": "large-block",
                "block_type": "image",
                "text": content + "\ntrailing parser content",
            }],
        },
    )

    unit = units[0]
    left = content.index("Left endpoint")
    profile = occurrence_scope(
        content=content,
        mention_start=left,
        mention_end=left + len("Left endpoint"),
        structure=unit.structure,
        context_kind=unit.context_kind,
    )
    alpha = occurrence_scope(
        content=content,
        mention_start=0,
        mention_end=len("Alpha"),
        structure=unit.structure,
        context_kind=unit.context_kind,
    )

    assert profile["source_role"] == "list_item"
    assert profile["occurrence_scope"]["scope_kind"] == "list_item"
    assert alpha["source_role"] == "body"
    assert alpha["occurrence_scope"]["scope_kind"] == "sentence"


def test_unicode_pdf_list_markers_preserve_hierarchy_and_parent_scope():
    content = "\uf0a8 System X\n— Module A\n— Module B\n Nested C\n— Module D"
    spans = structural_spans_for_chunk(content=content, blocks=[])

    assert [item["source_role"] for item in spans] == [
        "list_item", "list_item", "list_item", "list_item", "list_item",
    ]
    assert [item["list_depth"] for item in spans] == [0, 1, 1, 2, 1]
    assert spans[1]["list_parent_scope_id"] == spans[0]["scope_id"]
    assert spans[3]["list_parent_scope_id"] == spans[2]["scope_id"]
    assert spans[4]["list_parent_scope_id"] == spans[0]["scope_id"]

    parent_start = content.index("System X")
    child_start = content.index("Module A")
    parent = occurrence_scope(
        content=content, mention_start=parent_start,
        mention_end=parent_start + len("System X"),
        structure={"structural_spans": spans}, context_kind="structured_chunk",
    )
    child = occurrence_scope(
        content=content, mention_start=child_start,
        mention_end=child_start + len("Module A"),
        structure={"structural_spans": spans}, context_kind="structured_chunk",
    )
    assert child["occurrence_scope"]["list_parent_scope_id"] == (
        parent["occurrence_scope"]["scope_id"]
    )


def test_reused_pdf_marker_keeps_every_indented_child_at_the_same_depth():
    content = (
        "— Parent A\n"
        "  — Child A1\n"
        "  — Child A2\n"
        "— Parent B\n"
        "  — Child B1\n"
        "  — Child B2\n"
        "  — Child B3"
    )
    spans = structural_spans_for_chunk(content=content, blocks=[])

    assert [item["list_depth"] for item in spans] == [0, 1, 1, 0, 1, 1, 1]
    assert [item["list_parent_scope_id"] for item in spans] == [
        "",
        spans[0]["scope_id"],
        spans[0]["scope_id"],
        "",
        spans[3]["scope_id"],
        spans[3]["scope_id"],
        spans[3]["scope_id"],
    ]


def test_pdf_page_navigation_is_transparent_to_continued_list_hierarchy():
    content = (
        "\uf0a8 System X\n"
        "— Module A\n"
        "— Multicore\n"
        " Lock\n"
        "## Page 15\n"
        "11\n"
        " Atomic operation\n"
        "— Logging\n"
        "\uf0a8 Components\n"
        "— Runtime library"
    )
    spans = structural_spans_for_chunk(content=content, blocks=[])

    assert [item["list_depth"] for item in spans] == [
        0, 1, 1, 2, 2, 1, 0, 1,
    ]
    assert spans[1]["list_parent_scope_id"] == spans[0]["scope_id"]
    assert spans[3]["list_parent_scope_id"] == spans[2]["scope_id"]
    assert spans[4]["list_parent_scope_id"] == spans[2]["scope_id"]
    assert spans[5]["list_parent_scope_id"] == spans[0]["scope_id"]
    assert spans[7]["list_parent_scope_id"] == spans[6]["scope_id"]


def test_pdf_list_hierarchy_continues_across_exact_parser_block_boundary():
    first_page = (
        "\uf0a8 System X\n"
        "— Module A\n"
        "— Multicore\n"
        " Lock\n"
    )
    navigation = "## Page 15\n11\n"
    second_page = (
        " Atomic operation\n"
        "— Logging\n"
        "\uf0a8 Components\n"
        "— Runtime library"
    )
    content = first_page + navigation + second_page
    # The first block is deliberately too large to map into this chunk while
    # the second page maps exactly.  List parentage must follow the rendered
    # source across that parser-block split.
    spans = structural_spans_for_chunk(
        content=content,
        blocks=[
            {
                "key": "first-page-large",
                "block_type": "paragraph",
                "text": "prefix outside chunk\n" + first_page,
            },
            {
                "key": "second-page",
                "block_type": "paragraph",
                "text": second_page,
            },
        ],
    )

    assert [item["list_depth"] for item in spans] == [
        0, 1, 1, 2, 2, 1, 0, 1,
    ]
    assert spans[3]["list_parent_scope_id"] == spans[2]["scope_id"]
    assert spans[4]["list_parent_scope_id"] == spans[2]["scope_id"]
    assert spans[5]["list_parent_scope_id"] == spans[0]["scope_id"]
    assert spans[7]["list_parent_scope_id"] == spans[6]["scope_id"]
    assert {item["scope_id"].split(":", 1)[0] for item in spans} == {"chunk"}


def test_semantic_heading_still_terminates_list_hierarchy():
    content = (
        "\uf0a8 System X\n"
        "— Module A\n"
        "## Installation\n"
        "— Preparation\n"
        "— Deployment"
    )
    spans = structural_spans_for_chunk(content=content, blocks=[])

    assert [item["list_depth"] for item in spans] == [0, 1, 0, 0]
    assert spans[1]["list_parent_scope_id"] == spans[0]["scope_id"]
    assert spans[2]["list_parent_scope_id"] == ""
    assert spans[3]["list_parent_scope_id"] == ""


def test_semicolon_separates_atomic_relation_scopes_without_semantic_rules() -> None:
    content = "事件甲影响事件乙；事件乙影响事件丙。"
    first = occurrence_scope(
        content=content,
        mention_start=content.index("事件甲"),
        mention_end=content.index("事件甲") + len("事件甲"),
        structure={"block_type": "paragraph"},
        context_kind="paragraph",
    )
    last = occurrence_scope(
        content=content,
        mention_start=content.index("事件丙"),
        mention_end=content.index("事件丙") + len("事件丙"),
        structure={"block_type": "paragraph"},
        context_kind="paragraph",
    )

    assert first["occurrence_scope"]["scope_kind"] == "clause"
    assert last["occurrence_scope"]["scope_kind"] == "clause"
    assert (
        first["occurrence_scope"]["scope_id"]
        != last["occurrence_scope"]["scope_id"]
    )
