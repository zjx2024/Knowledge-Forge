from knowledge_forge.macro_context import (
    MACRO_CONTEXT_VERSION,
    build_project_macro_context,
    select_macro_context,
)
from knowledge_forge.relation_extraction import proposer_prompt
from tests.grounded_relation_fixtures import logical_schema, predicate_policy


def _inputs():
    documents = [{"id": 1}, {"id": 2}]
    chunks = [
        {"id": 10, "document_id": 1},
        {"id": 11, "document_id": 1},
        {"id": 20, "document_id": 2},
    ]
    canonicals = [
        {"canonical_key": "a", "name": "系统", "primary_type": "Product"},
        {"canonical_key": "b", "name": "模块", "primary_type": "Component"},
        {"canonical_key": "c", "name": "工具", "primary_type": "Tool"},
    ]
    assignments = [
        {"mention_key": "m1", "canonical_key": "a", "document_id": 1, "chunk_id": 10},
        {"mention_key": "m2", "canonical_key": "b", "document_id": 1, "chunk_id": 10},
        {"mention_key": "m3", "canonical_key": "a", "document_id": 1, "chunk_id": 10},
        {"mention_key": "m4", "canonical_key": "c", "document_id": 1, "chunk_id": 11},
        {"mention_key": "m5", "canonical_key": "a", "document_id": 2, "chunk_id": 20},
        {"mention_key": "m6", "canonical_key": "c", "document_id": 2, "chunk_id": 20},
    ]
    mentions = [
        {"mention_key": "m1", "document_id": 1, "chunk_id": 10},
        {"mention_key": "m2", "document_id": 1, "chunk_id": 10},
        {"mention_key": "m3", "document_id": 1, "chunk_id": 10},
        {"mention_key": "m4", "document_id": 1, "chunk_id": 11},
        {"mention_key": "m5", "document_id": 2, "chunk_id": 20},
        {"mention_key": "m6", "document_id": 2, "chunk_id": 20},
    ]
    return documents, chunks, mentions, assignments, canonicals


def test_macro_context_is_deterministic_and_counts_chunk_level_cooccurrence():
    payload = build_project_macro_context(
        documents=_inputs()[0],
        chunks=_inputs()[1],
        mentions=_inputs()[2],
        assignments=_inputs()[3],
        canonicals=_inputs()[4],
        max_entities=3,
    )
    assert payload["version"] == MACRO_CONTEXT_VERSION
    assert payload["hash"]
    assert payload["entity_count"] == 3
    system = payload["entities"][0]
    assert system["name"] == "系统"
    assert system["mention_count"] == 3
    assert {item["name"] for item in system["cooccurring_entities"]} == {"模块", "工具"}
    again = build_project_macro_context(
        documents=_inputs()[0], chunks=_inputs()[1], mentions=_inputs()[2],
        assignments=_inputs()[3], canonicals=_inputs()[4], max_entities=3,
    )
    assert again == payload


def test_macro_selection_marks_global_context_as_non_evidence_payload():
    source = build_project_macro_context(
        documents=_inputs()[0], chunks=_inputs()[1], mentions=_inputs()[2],
        assignments=_inputs()[3], canonicals=_inputs()[4], max_entities=3,
    )
    selected = select_macro_context(source, ["b"], max_entities=2)
    assert selected["focus_entity_ids"] == ["b"]
    assert selected["entities"][0]["canonical_id"] == "b"
    assert [item["canonical_id"] for item in selected["entities"]] == ["b"]
    assert selected["entities"][0]["cooccurring_entities"] == []
    assert selected["hash"] == source["hash"]
    assert selected["project_entity_count"] == 3


def test_proposer_prompt_carries_bounded_macro_prior_separately_from_source():
    prompt = proposer_prompt(
        context_id="scope-1",
        content="系统包含模块。",
        mentions=[
            {"occurrence_key": "a", "surface": "系统", "type": "Product", "mention_start": 0, "mention_end": 2},
            {"occurrence_key": "b", "surface": "模块", "type": "Component", "mention_start": 4, "mention_end": 6},
        ],
        logical_schema=logical_schema(
            predicate_policy("alpha_rel", domain=["Product"], range_=["Component"]),
            version="macro-prompt-v1",
        ),
        required_evidence_span=(0, len("系统包含模块。")),
        macro_context={"version": MACRO_CONTEXT_VERSION, "hash": "macro-hash", "entities": []},
    )
    import json

    payload = json.loads(prompt)
    assert payload["macro_context"]["hash"] == "macro-hash"
    assert "macro_context" in payload
    assert payload["context"]["text"] == "系统包含模块。"
