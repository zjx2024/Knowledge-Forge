from __future__ import annotations

from knowledge_forge.span_entities import SpanFirstEntityExtractor


def test_span_first_mentions_are_exact_stable_and_type_independent():
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
    assert first.mentions[0].surface == "Acme Corp"
    assert first.type_assertions[0].entity_type == "Organization"


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
