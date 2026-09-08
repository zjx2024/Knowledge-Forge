from __future__ import annotations

import json

from knowledge_forge.chunking import (
    DocumentBlock,
    ParsedDocument,
    sanitize_parsed_document,
)
from knowledge_forge.database import Database
from knowledge_forge.ollama import OllamaClient
from knowledge_forge.unicode_safety import (
    sanitize_unicode_data,
    sanitize_unicode_text,
    unsafe_unicode_code_unit_count,
)


def test_unicode_sanitizer_preserves_valid_supplementary_characters() -> None:
    mathematical_bold_a = "\U0001d400"
    assert sanitize_unicode_text(mathematical_bold_a) == mathematical_bold_a
    assert sanitize_unicode_text("\ud835\udc00") == mathematical_bold_a


def test_unicode_sanitizer_repairs_pdf_nul_and_isolated_surrogate() -> None:
    raw = "K\x001 and input \ud835(x)"
    repaired = sanitize_unicode_text(raw)
    assert repaired == "K 1 and input \ufffd(x)"
    assert unsafe_unicode_code_unit_count(raw) == 2
    assert unsafe_unicode_code_unit_count(repaired) == 0
    assert repaired.encode("utf-8")


def test_recursive_unicode_sanitizer_preserves_container_types() -> None:
    repaired = sanitize_unicode_data({"items": ["a\x00b", ("\ud835",)]})
    assert repaired == {"items": ["a b", ("\ufffd",)]}
    assert isinstance(repaired["items"][1], tuple)


def test_parsed_document_records_generic_unicode_repair_warning() -> None:
    parsed = ParsedDocument(
        blocks=[DocumentBlock(
            "b1", "paragraph", "value\x00\ud835", 1, "s1", "Title",
            metadata={"caption": "bad\ud835"},
        )],
        parser_type="pdf",
        parser_version="1",
        quality={"warnings": []},
    )
    repaired = sanitize_parsed_document(parsed)
    assert repaired.blocks[0].text == "value \ufffd"
    assert repaired.blocks[0].metadata["caption"] == "bad\ufffd"
    assert repaired.quality["unicode_sanitized_code_units"] == 3
    assert "3 个非法 Unicode 码元" in repaired.quality["warnings"][0]


def test_database_json_boundary_never_emits_invalid_utf8() -> None:
    encoded = Database._json({"text": "zero\x00surrogate\ud835"})
    assert json.loads(encoded) == {"text": "zero surrogate\ufffd"}
    assert encoded.encode("utf-8")


def test_ollama_json_boundary_repairs_escaped_isolated_surrogate() -> None:
    parsed = OllamaClient._parse_json(r'{"name":"input \ud835","note":"a\u0000b"}')
    assert parsed == {"name": "input \ufffd", "note": "a b"}
