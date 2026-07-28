from __future__ import annotations

from knowledge_forge.span_entities import SpanFirstEntityExtractor


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


def test_three_repeated_ascii_multiword_mentions_keep_distinct_occurrence_ids():
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
            "name": surface, "type": "Location", "mention_start": start,
            "mention_end": start + len(surface), "evidence_text": surface,
            "evidence_start": start, "evidence_end": start + len(surface),
        } for start in starts],
        chunk_metadata={},
    )

    assert len(result.mentions) == 3
    assert len({item.mention_id for item in result.mentions}) == 3
    assert [(item.start, item.end) for item in result.mentions] == [
        (start, start + len(surface)) for start in starts
    ]


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
        "2024e 12% +1 202 555 0123 domestic foreign 44.8 million "
        "220,000 metric tons Unit"
    )
    spans = {
        surface: (content.index(surface), content.index(surface) + len(surface))
        for surface in (
            "2024e",
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
