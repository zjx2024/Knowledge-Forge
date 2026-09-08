from __future__ import annotations

from knowledge_forge.evidence_context import build_context_units_for_document


def test_table_row_context_preserves_header_row_cell_source_metadata():
    payload = {
        "parser_version": "parser-v1",
        "blocks": [{
            "key": "table-1-row-2",
            "block_type": "table_row",
            "text": "Canada | 220,000 metric tons | 2024e",
            "order": 1,
            "section_key": "s1",
            "section_title": "Production table",
            "page": 3,
            "metadata": {
                "table_id": "table-1",
                "row_id": "row-2",
                "cell_id": "cell-2-2",
                "header_path": ["Country", "Mine production", "2024e"],
                "cell_role": "value",
                "unit": "metric tons",
                "estimated": True,
            },
        }],
    }

    unit = build_context_units_for_document(
        namespace_id="ns",
        document={"id": 1, "active_revision_id": "rev-1"},
        chunks=[],
        processing_payload=payload,
    )[0]

    assert unit.context_kind == "table_row"
    assert unit.structure["table_id"] == "table-1"
    assert unit.structure["row_id"] == "row-2"
    assert unit.structure["cell_id"] == "cell-2-2"
    assert unit.structure["header_path"] == [
        "Country", "Mine production", "2024e",
    ]
    assert unit.structure["unit"] == "metric tons"
    assert unit.structure["estimated"] is True
    assert unit.source_map[0].table_id == "table-1"
    assert unit.source_map[0].row_id == "row-2"
    assert unit.source_map[0].cell_id == "cell-2-2"


def test_neighboring_table_rows_remain_separate_context_units():
    payload = {
        "blocks": [
            {
                "key": "row-canada",
                "block_type": "table_row",
                "text": "Canada | 220,000",
                "order": 1,
                "metadata": {"table_id": "t1", "row_id": "r-canada"},
            },
            {
                "key": "row-chile",
                "block_type": "table_row",
                "text": "Chile | 240,000",
                "order": 2,
                "metadata": {"table_id": "t1", "row_id": "r-chile"},
            },
        ],
    }

    units = build_context_units_for_document(
        namespace_id="ns",
        document={"id": 1, "active_revision_id": "rev-1"},
        chunks=[],
        processing_payload=payload,
    )

    assert [unit.rendered_text for unit in units] == [
        "Canada | 220,000",
        "Chile | 240,000",
    ]
    assert [unit.structure["row_id"] for unit in units] == [
        "r-canada",
        "r-chile",
    ]
