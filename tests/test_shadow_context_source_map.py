from __future__ import annotations

from knowledge_forge.shadow_context import build_context_units_for_document


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
