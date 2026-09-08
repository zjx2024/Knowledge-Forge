import json
import re

import pytest

from knowledge_forge.config import Config
from knowledge_forge.schema_evolution import BASE_SCHEMA, SchemaEvolutionService, _RELATION_CUES
from knowledge_forge.vector_store import _terms


def _legacy_candidates(service, content, entities, logical):
    sentences = [
        (match.start(), match.end(), match.group(0).strip())
        for match in re.finditer(r"[^\n。！？.!?]+(?:[。！？.!?]+|$)", content)
        if match.group(0).strip()
    ]
    locations = {}
    for entity in entities:
        start = entity.get("evidence_start")
        if start is None:
            continue
        sentence_index = next(
            (index for index, (left, right, _) in enumerate(sentences) if left <= int(start) < right),
            -1,
        )
        if sentence_index >= 0:
            locations[entity["id"]] = sentence_index
    definitions = logical.get("relation_types") or []
    candidates = []
    seen = set()
    for source in entities:
        source_sentence = locations.get(source["id"])
        if source_sentence is None:
            continue
        for target in entities:
            if source["id"] == target["id"]:
                continue
            target_sentence = locations.get(target["id"])
            if target_sentence is None or abs(source_sentence - target_sentence) > 1:
                continue
            pair = (source["id"], target["id"])
            if pair in seen:
                continue
            left, right = sorted((source_sentence, target_sentence))
            context = " ".join(sentences[index][2] for index in range(left, right + 1))
            if source_sentence != target_sentence and (
                len(context) > 500 or not _RELATION_CUES.search(context)
            ):
                continue
            allowed = [
                definition["name"]
                for definition in definitions
                if service._compatible_relation(definition, source["type"], target["type"])
            ]
            if not allowed:
                continue
            seen.add(pair)
            candidates.append(
                {
                    "source": source["id"],
                    "source_name": source["name"],
                    "target": target["id"],
                    "target_name": target["name"],
                    "allowed_relation_types": allowed,
                    "context": context,
                }
            )
    return candidates


def _entities(content):
    values = [
        ("e1", "Alpha", "Organization"),
        ("e2", "Beta", "Product"),
        ("e3", "Gamma", "Location"),
        ("e4", "Delta", "Organization"),
        ("e5", "Epsilon", "Product"),
    ]
    return [
        {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "evidence_start": content.index(name),
        }
        for entity_id, name, entity_type in values
    ]


def test_bucketed_relation_candidates_are_byte_equivalent_to_p02_order():
    content = (
        "Alpha produces Beta. "
        "Beta is used in Gamma! "
        "This unrelated sentence separates the evidence. "
        "Delta produces Epsilon."
    )
    service = SchemaEvolutionService(None, None)
    entities = _entities(content)

    expected = _legacy_candidates(service, content, entities, BASE_SCHEMA)
    actual = service._relation_candidates(content, entities, BASE_SCHEMA)

    assert json.dumps(actual, ensure_ascii=False, separators=(",", ":")) == json.dumps(
        expected, ensure_ascii=False, separators=(",", ":")
    )


class _SequentialOllama:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def json_chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.mark.parametrize(
    "primary,content",
    [
        (
            {
                "entities": [
                    {"id": "e1", "name": "Alpha", "type": "Organization", "evidence": "Alpha"},
                    {"id": "e2", "name": "Beta", "type": "Product", "evidence": "Beta"},
                ],
                "relations": [
                    {"source": "e1", "target": "e2", "type": "produces", "evidence": "Alpha produces Beta"}
                ],
            },
            "Alpha produces Beta.",
        ),
        (
            {"entities": [{"id": "e1", "name": "Alpha", "type": "Organization", "evidence": "Alpha"}], "relations": []},
            "Alpha produces products.",
        ),
        (
            {
                "entities": [
                    {"id": "e1", "name": "Alpha", "type": "Organization", "evidence": "Alpha"},
                    {"id": "e2", "name": "Beta", "type": "Product", "evidence": "Beta"},
                ],
                "relations": [],
            },
            "Alpha and Beta are mentioned.",
        ),
    ],
)
def test_relation_candidates_are_not_built_when_cheap_gleaning_gate_fails(monkeypatch, primary, content):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", True)
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    service = SchemaEvolutionService(None, _SequentialOllama([primary]))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("candidate generation must stay behind the cheap gate")

    monkeypatch.setattr(service, "_relation_candidates", unexpected)
    result = service.extract_chunk(
        {"llm_model": "test"}, {"schema_json": BASE_SCHEMA}, {"content": content}
    )

    assert result["_diagnostics"]["relation_candidates"] == 0


def test_empty_high_information_gleaning_does_not_build_entity_pairs(monkeypatch):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", True)
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    rich = (
        "A supplier produces components and an enterprise depends on those components. "
        "The enterprise uses the product in a manufacturing process. "
    ) * 8
    ollama = _SequentialOllama([
        {"entities": [], "relations": []},
        {"entities": [], "relations": []},
    ])
    service = SchemaEvolutionService(None, ollama)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("zero-entity gleaning must not enumerate pairs")

    monkeypatch.setattr(service, "_relation_candidates", unexpected)
    result = service.extract_chunk(
        {"llm_model": "test"}, {"schema_json": BASE_SCHEMA}, {"content": rich}
    )

    assert len(ollama.calls) == 2
    assert result["_diagnostics"]["gleaning_used"] is True
    assert result["_diagnostics"]["relation_candidates"] == 0


def test_streaming_term_counter_remains_exact_for_mixed_text():
    def legacy_terms(text):
        normalized = re.sub(r"\s+", "", (text or "").lower())
        latin = re.findall(r"[a-z0-9_]{2,}", normalized)
        chinese = [
            normalized[index : index + 2]
            for index in range(max(0, len(normalized) - 1))
            if "\u4e00" <= normalized[index] <= "\u9fff"
        ]
        from collections import Counter

        return Counter(latin + chinese)

    for text in ("Alpha alpha_2 产业链供应链A1", "纯中文", "latin only", "", "中A文"):
        assert _terms(text) == legacy_terms(text)
