from knowledge_forge.schema_evolution import (
    BASE_SCHEMA,
    ENTITY_ONLY_EXTRACTION_FORMAT,
    SchemaEvolutionService,
)
from knowledge_forge.config import Config
from knowledge_forge.ollama import OllamaProtocolError


def _chunk(key, section, text, vector):
    return {
        "id": len(key),
        "chunk_key": key,
        "content": text,
        "metadata": {"section_key": section, "section_title": section},
        "vector": vector,
    }


RICH_INDUSTRY = (
    "机器人产业链包括上游零部件企业、中游整机制造企业和下游应用企业。"
    "核心企业使用伺服系统和减速器，并依赖供应商提供关键部件。"
) * 5
RICH_SUPPLY = (
    "供应商向制造企业供应控制器和传感器，制造企业生产工业机器人产品。"
    "企业之间形成稳定供应关系，产品最终用于汽车和电子制造。"
) * 5


class FakeDatabase:
    def __init__(self, chunks=None, prior_documents=0, nearest_score=0.2):
        self.active = {"version": 1, "schema_json": BASE_SCHEMA}
        self.created = None
        self.chunks = chunks or []
        self.prior_documents = prior_documents
        self.nearest_score = nearest_score

    def ensure_base_schema(self, project_id, schema):
        return self.active

    def create_schema_version(self, project_id, schema, report, documents, created_by="llm"):
        self.created = (project_id, schema, report, documents, created_by)
        return {"version": 2, "schema_json": schema}

    def activate_base_schema(self, project_id, schema):
        self.active = {"version": 1, "schema_json": schema}
        return self.active

    def list_document_chunks(self, document_id):
        return self.chunks

    def list_document_embeddings(self, document_id, object_type, model_name):
        return [
            {"object_key": item["chunk_key"], "embedding": item["vector"]}
            for item in self.chunks
        ]

    def count_prior_indexed_documents(self, project_id, document_id):
        return self.prior_documents

    def nearest_embedding_score(self, *args, **kwargs):
        return self.nearest_score


class FakeOllama:
    def __init__(self):
        self.calls = 0

    def json_chat(self, **kwargs):
        self.calls += 1
        return {
            "summary": "产业链文档需要产业节点",
            "entity_type_candidates": [
                {
                    "name": "Industry",
                    "description": "Industry sector",
                    "confidence": 0.92,
                    "evidence_chunks": ["industry_p001", "supply_p001"],
                },
                {
                    "name": "bad-name",
                    "description": "invalid",
                    "confidence": 0.99,
                    "evidence_chunks": ["industry_p001", "supply_p001"],
                },
            ],
            "relation_type_candidates": [
                {
                    "name": "supplies",
                    "description": "Supply relation",
                    "source_types": ["Organization"],
                    "target_types": ["Organization"],
                    "semantic_definition": "One organization supplies another.",
                    "domain_types": ["Organization"],
                    "range_types": ["Organization"],
                    "direction_description": "Orient from supplier to recipient.",
                    "symmetric": False,
                    "inverse_predicate": None,
                    "canonical_orientation": "forward",
                    "publish_mode": "auto",
                    "reflexive": False,
                    "positive_examples": [],
                    "negative_examples": [],
                    "optional_soft_cues": ["supplies"],
                    "confidence": 0.91,
                    "evidence_chunks": ["industry_p001", "supply_p001"],
                }
            ],
        }


def test_schema_evolution_is_additive_and_evidence_gated():
    chunks = [
        _chunk("industry_p001", "industry", RICH_INDUSTRY, [1.0, 0.0]),
        _chunk("supply_p001", "supply", RICH_SUPPLY, [0.0, 1.0]),
    ]
    database = FakeDatabase(chunks)
    service = SchemaEvolutionService(database, FakeOllama())
    result = service.evaluate_document(
        {"id": 3, "llm_model": "test", "embedding_model": "embed"}, 7, chunks
    )
    assert result["version"] == 2
    assert result["_evaluation"]["reviewed"] is True
    schema = database.created[1]
    entity_names = {item["name"] for item in schema["entity_types"]}
    relation_names = {item["name"] for item in schema["relation_types"]}
    assert "Industry" in entity_names
    assert "bad-name" not in entity_names
    assert "supplies" in relation_names
    assert len(schema["entity_types"]) == len(BASE_SCHEMA["entity_types"]) + 1
    assert database.created[2]["policy"] == "representative-sampling+evidence-gated+additive-only"


def test_representative_sampling_filters_toc_and_spans_sections():
    toc = _chunk(
        "toc_p001",
        "toc",
        "目录\n第一章 ........................ 1\n第二章 ........................ 3\n第三章 ........................ 8",
        [0.9, 0.1],
    )
    chunks = [
        toc,
        _chunk("industry_p001", "industry", RICH_INDUSTRY, [1.0, 0.0]),
        _chunk("supply_p001", "supply", RICH_SUPPLY, [0.0, 1.0]),
    ]
    database = FakeDatabase(chunks)
    samples, diagnostics = SchemaEvolutionService(database, FakeOllama()).select_representative_chunks(
        {"id": 3, "embedding_model": "embed"}, 7, chunks
    )
    keys = {_chunk["chunk_key"] for _chunk in samples}
    assert keys == {"industry_p001", "supply_p001"}
    assert diagnostics["filtered_chunks"] == 1
    assert diagnostics["selected_sections"] == 2


def test_representative_sampling_scales_for_large_documents_with_a_ceiling(monkeypatch):
    monkeypatch.setattr(Config, "schema_sample_base_chunks", 6)
    monkeypatch.setattr(Config, "schema_sample_ceiling_chunks", 20)
    monkeypatch.setattr(Config, "schema_sample_base_chars", 6000)
    monkeypatch.setattr(Config, "schema_sample_ceiling_chars", 30000)
    monkeypatch.setattr(Config, "schema_sample_chars_per_chunk", 1200)
    chunks = [
        _chunk(
            f"section_{index:03d}_p001",
            f"section_{index:03d}",
            RICH_INDUSTRY if index % 2 else RICH_SUPPLY,
            [1.0, float(index % 7) / 7],
        )
        for index in range(100)
    ]
    database = FakeDatabase(chunks)

    samples, diagnostics = SchemaEvolutionService(database, FakeOllama()).select_representative_chunks(
        {"id": 3, "embedding_model": "embed"}, 7, chunks
    )

    assert 6 < diagnostics["target_chunks"] <= 20
    assert len(samples) == diagnostics["target_chunks"]
    assert diagnostics["target_chars"] <= 30000
    assert diagnostics["selected_chars"] <= diagnostics["target_chars"]
    assert diagnostics["selected_sections"] > 6


def test_similar_document_skips_schema_llm_review():
    chunks = [
        _chunk("industry_p001", "industry", RICH_INDUSTRY, [1.0, 0.0]),
        _chunk("supply_p001", "supply", RICH_SUPPLY, [0.0, 1.0]),
    ]
    database = FakeDatabase(chunks, prior_documents=2, nearest_score=0.95)
    ollama = FakeOllama()
    result = SchemaEvolutionService(database, ollama).evaluate_document(
        {"id": 3, "llm_model": "test", "embedding_model": "embed"}, 7, chunks
    )
    assert result["version"] == 1
    assert result["_evaluation"]["reason"] == "covered_by_active_schema"
    assert ollama.calls == 0
    assert database.created is None


class FlatExtractionOllama:
    def json_chat(self, **kwargs):
        return {"items": [
            {"id": "e1", "type": "Organization", "name": "星河机器人公司"},
            {"id": "e2", "type": "Location", "name": "上海"},
            {"id": "e1", "relation": "located_in", "target_id": "e2"},
        ]}


def test_extract_chunk_accepts_flat_model_output(monkeypatch):
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    service = SchemaEvolutionService(None, FlatExtractionOllama())
    result = service.extract_chunk(
        {"llm_model": "test"},
        {"schema_json": BASE_SCHEMA},
        {"content": "星河机器人公司位于上海。"},
    )
    assert [item["name"] for item in result["entities"]] == ["星河机器人公司", "上海"]
    relation = result["relations"][0]
    assert {key: relation[key] for key in ("source", "target", "type", "description", "confidence")} == {
        "source": "e1", "target": "e2", "type": "located_in", "description": "", "confidence": 0.5
    }
    assert relation["evidence_text"] == ""
    assert relation["verification_status"] == "uncertain"


class LabeledConfidenceOllama:
    def json_chat(self, **kwargs):
        return {
            "entities": [
                {"id": "e1", "name": "Alpha", "type": "Organization", "description": "", "aliases": [], "confidence": "high"},
                {"id": "e2", "name": "Beta", "type": "Product", "description": "", "aliases": [], "confidence": "中"},
            ],
            "relations": [
                {"source": "e1", "target": "e2", "type": "produces", "description": "", "confidence": "85%"},
            ],
        }


def test_extract_chunk_normalizes_labeled_and_percentage_confidence(monkeypatch):
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    result = SchemaEvolutionService(None, LabeledConfidenceOllama()).extract_chunk(
        {"llm_model": "test"},
        {"schema_json": BASE_SCHEMA},
        {"content": "Alpha produces Beta."},
    )
    assert [item["confidence"] for item in result["entities"]] == [0.9, 0.6]
    assert result["relations"][0]["confidence"] == 0.85


class FailingOllama:
    def json_chat(self, **kwargs):
        raise RuntimeError("invalid model output")


def test_schema_review_failure_keeps_active_schema():
    chunks = [
        _chunk("industry_p001", "industry", RICH_INDUSTRY, [1.0, 0.0]),
        _chunk("supply_p001", "supply", RICH_SUPPLY, [0.0, 1.0]),
    ]
    database = FakeDatabase(chunks)
    result = SchemaEvolutionService(database, FailingOllama()).evaluate_document(
        {"id": 3, "llm_model": "test", "embedding_model": "embed"}, 7, chunks
    )
    assert result["version"] == database.active["version"]
    assert result["_evaluation"]["reviewed"] is True
    assert result["_evaluation"]["batch_errors"]
    assert database.created is None


class HallucinatedEntityOllama:
    def json_chat(self, **kwargs):
        return {
            "entities": [{
                "id": "e1", "name": "不存在的公司", "type": "Organization",
                "description": "", "aliases": [], "evidence": "不存在的公司", "confidence": 0.99,
            }],
            "relations": [],
        }


def test_extract_chunk_rejects_entity_without_source_evidence(monkeypatch):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", False)
    result = SchemaEvolutionService(None, HallucinatedEntityOllama()).extract_chunk(
        {"llm_model": "test"}, {"schema_json": BASE_SCHEMA}, {"content": "本文只介绍星河机器人公司。"}
    )
    assert result["entities"] == []
    assert result["_diagnostics"]["rejected_entities"] == 1


def test_entity_only_schema_requires_integer_mention_offsets_and_calls_only_json_chat_once():
    entity_schema = ENTITY_ONLY_EXTRACTION_FORMAT["properties"]["entities"]["items"]
    assert {"mention_start", "mention_end"} <= set(entity_schema["required"])
    assert set(entity_schema["required"]) == {
        "id", "name", "type", "mention_start", "mention_end", "confidence",
    }
    assert entity_schema["properties"]["mention_start"] == {"type": "integer"}
    assert entity_schema["properties"]["mention_end"] == {"type": "integer"}
    assert (
        ENTITY_ONLY_EXTRACTION_FORMAT["properties"]["entities"]["maxItems"]
        == Config.entity_only_max_items_per_segment
    )

    class SpyOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"entities": [{
                "id": "e1",
                "name": "Alpha",
                "type": "Organization",
                "description": "",
                "aliases": [],
                "evidence": "Alpha",
                "evidence_start": 0,
                "evidence_end": 5,
                "mention_start": 0,
                "mention_end": 5,
                "confidence": 0.9,
            }]}

    ollama = SpyOllama()
    service = SchemaEvolutionService(None, ollama)
    service.extract_chunk = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("extract_chunk must not be called")
    )
    service._glean_extraction = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("gleaning must not be called")
    )
    service._verify_relations = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("relation verifier must not be called")
    )

    result = service.extract_entities_only(
        {"llm_model": "test"},
        {"schema_json": BASE_SCHEMA},
        {"content": "Alpha works."},
    )

    assert len(ollama.calls) == 1
    assert ollama.calls[0]["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT
    assert ollama.calls[0]["max_output_tokens"] == Config.entity_only_max_output_tokens
    assert "Return an empty entities array only" in ollama.calls[0]["system"]
    assert result["relations"] == []
    assert result["_diagnostics"]["llm_calls"] == 1
    assert result["_diagnostics"]["verification_llm_calls"] == 0


def test_entity_only_accepts_compact_items_protocol_without_relaxing_span_validation():
    class CompactItemsOllama:
        def json_chat(self, **_kwargs):
            return {
                "items": [
                    {
                        "name": "United States",
                        "type": "Location",
                        "mention_start": 0,
                        "mention_end": 13,
                    },
                    {
                        "name": "cobalt",
                        "type": "Product",
                        "mention_start": 47,
                        "mention_end": 53,
                    },
                    {
                        "name": "wrong span",
                        "type": "Product",
                        "mention_start": 14,
                        "mention_end": 19,
                    },
                ]
            }

    source = "United States produces about 1 million tons of cobalt."
    service = SchemaEvolutionService.__new__(SchemaEvolutionService)
    service.ollama = CompactItemsOllama()
    result = service.extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert [(item["id"], item["name"]) for item in result["entities"]] == [
        ("entity_0001", "United States"),
        ("entity_0002", "cobalt"),
    ]
    assert result["entities"][0]["confidence"] == 0.5
    assert result["_diagnostics"]["rejected_entities"] == 1


def test_entity_only_splits_recoverable_overflow_and_restores_source_offsets():
    class OverflowThenSplitOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            source = kwargs["prompt"].split("Source text:\n", 1)[1]
            self.calls.append(source)
            if len(self.calls) == 1:
                raise OllamaProtocolError("Ollama response was truncated by output length")
            if "Beta" not in source:
                return {"items": [{
                    "name": "Alpha",
                    "type": "Person",
                    "mention_start": source.index("Alpha"),
                    "mention_end": source.index("Alpha") + 5,
                }]}
            return {"items": [{
                "name": "Paris",
                "type": "Location",
                "mention_start": source.index("Paris"),
                "mention_end": source.index("Paris") + 5,
            }]}

    source = ("Alpha works for Acme. " * 20) + ("Beta visits Paris. " * 20)
    ollama = OverflowThenSplitOllama()
    service = SchemaEvolutionService(None, ollama)
    result = service.extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert len(ollama.calls) >= 3
    assert all(len(call) < len(source) for call in ollama.calls[1:])
    assert result["_diagnostics"]["segmented_fallback"] is True
    assert result["_diagnostics"]["llm_calls"] == 3
    assert [(item["name"], item["mention_start"], item["mention_end"]) for item in result["entities"]] == [
        ("Alpha", 0, 5),
        ("Paris", source.index("Paris", len(source) // 2), source.index("Paris", len(source) // 2) + 5),
    ]


def test_entity_only_short_overflow_uses_bounded_json_fallback():
    class ShortOverflowOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise OllamaProtocolError("Ollama response was truncated by output length")
            return {"items": [{
                "name": "Paris",
                "type": "Location",
                "mention_start": 0,
                "mention_end": 5,
            }]}

    ollama = ShortOverflowOllama()
    service = SchemaEvolutionService(None, ollama)
    result = service.extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": "Paris.", "metadata": {}},
    )

    assert len(ollama.calls) == 2
    assert ollama.calls[0]["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT
    assert ollama.calls[1]["schema"] is None
    assert ollama.calls[1]["temperature"] == 0
    assert result["entities"][0]["name"] == "Paris"
    assert result["_diagnostics"]["llm_calls"] == 2


def test_entity_only_medium_overflow_splits_before_unstructured_fallback():
    class MediumOverflowOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            source = kwargs["prompt"].split("Source text:\n", 1)[1]
            self.calls.append(kwargs)
            if len(source) > 80:
                raise OllamaProtocolError("Ollama response was truncated by output length")
            name = "Chile" if "Chile" in source else "Argentina"
            start = source.index(name)
            return {"items": [{
                "name": name,
                "type": "Location",
                "mention_start": start,
                "mention_end": start + len(name),
            }]}

    source = (
        "Projects in Chile lower EV battery recycling costs. "
        "Other import sources include Argentina."
    )
    ollama = MediumOverflowOllama()
    result = SchemaEvolutionService(None, ollama).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert len(ollama.calls) == 3
    assert all(call["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT for call in ollama.calls)
    assert {item["name"] for item in result["entities"]} == {"Chile", "Argentina"}
    assert result["_diagnostics"]["segmented_fallback"] is True


def test_entity_only_proactively_segments_large_source_and_preserves_offsets(monkeypatch):
    monkeypatch.setattr(Config, "entity_only_segment_max_chars", 256)

    class SegmentedOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            source = kwargs["prompt"].split("Source text:\n", 1)[1]
            self.calls.append(source)
            name = "Alpha" if "Alpha" in source else "Paris"
            start = source.index(name)
            return {"items": [{
                "name": name,
                "type": "Organization" if name == "Alpha" else "Location",
                "mention_start": start,
                "mention_end": start + len(name),
            }]}

    source = ("Alpha builds systems. " * 20) + ("Paris hosts teams. " * 20)
    ollama = SegmentedOllama()
    result = SchemaEvolutionService(None, ollama).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert len(ollama.calls) >= 2
    assert all(len(call) <= 256 for call in ollama.calls)
    assert result["_diagnostics"]["segmented_fallback"] is True
    assert all(
        source[item["mention_start"]:item["mention_end"]] == item["name"]
        for item in result["entities"]
    )


def test_entity_only_splits_again_when_schema_item_cap_is_reached(monkeypatch):
    monkeypatch.setattr(Config, "entity_only_segment_max_chars", 4096)
    monkeypatch.setattr(Config, "entity_only_max_items_per_segment", 2)

    class CappedOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            source = kwargs["prompt"].split("Source text:\n", 1)[1]
            self.calls.append(source)
            if len(self.calls) == 1:
                return {"items": [
                    {"name": "Alpha", "type": "Organization", "mention_start": 0, "mention_end": 5},
                    {"name": "Alpha", "type": "Organization", "mention_start": 0, "mention_end": 5},
                ]}
            rows = []
            for name, entity_type in (("Alpha", "Organization"), ("Paris", "Location")):
                if name not in source:
                    continue
                start = source.index(name)
                rows.append({
                    "name": name,
                    "type": entity_type,
                    "mention_start": start,
                    "mention_end": start + len(name),
                })
            return {"items": rows}

    source = ("Alpha builds systems. " * 20) + ("Paris hosts teams. " * 20)
    ollama = CappedOllama()
    result = SchemaEvolutionService(None, ollama).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert len(ollama.calls) >= 3
    assert all(len(call) < len(source) for call in ollama.calls[1:])
    assert result["_diagnostics"]["segmented_fallback"] is True
    assert {item["name"] for item in result["entities"]} == {"Alpha", "Paris"}


class SequentialExtractionOllama:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def json_chat(self, **kwargs):
        self.calls += 1
        return self.responses.pop(0)


def test_informative_empty_first_pass_gets_one_targeted_second_pass(monkeypatch):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", True)
    monkeypatch.setattr(Config, "extraction_verification_enabled", False)
    core_start = RICH_INDUSTRY.index("核心企业")
    servo_start = RICH_INDUSTRY.index("伺服系统")
    relation_start = RICH_INDUSTRY.index("核心企业使用伺服系统")
    relation_text = "核心企业使用伺服系统"
    ollama = SequentialExtractionOllama([
        {"entities": [], "relations": []},
        {
            "entities": [
                {"id": "e1", "name": "核心企业", "type": "Organization", "description": "", "aliases": [],
                 "evidence": "核心企业", "mention_start": core_start,
                 "mention_end": core_start + len("核心企业"), "confidence": 0.9},
                {"id": "e2", "name": "伺服系统", "type": "Product", "description": "", "aliases": [],
                 "evidence": "伺服系统", "mention_start": servo_start,
                 "mention_end": servo_start + len("伺服系统"), "confidence": 0.9},
            ],
            "relations": [{"source": "e1", "target": "e2", "type": "uses", "description": "",
                           "evidence": relation_text, "evidence_start": relation_start,
                           "evidence_end": relation_start + len(relation_text), "confidence": 0.9}],
        },
    ])
    result = SchemaEvolutionService(None, ollama).extract_chunk(
        {"llm_model": "test"}, {"schema_json": BASE_SCHEMA},
        {"content": RICH_INDUSTRY, "metadata": {"section_title": "产业链"}},
    )
    assert ollama.calls == 2
    assert len(result["entities"]) == 2
    assert len(result["relations"]) == 1
    assert result["_diagnostics"]["gleaning_used"] is True


class RejectingRelationOllama:
    def __init__(self):
        self.calls = 0

    def json_chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return {
                "entities": [
                    {"id": "e1", "name": "Alpha", "type": "Organization", "description": "", "aliases": [],
                     "evidence": "Alpha", "confidence": 0.9},
                    {"id": "e2", "name": "Beta", "type": "Product", "description": "", "aliases": [],
                     "evidence": "Beta", "confidence": 0.9},
                ],
                "relations": [{"source": "e1", "target": "e2", "type": "produces", "description": "",
                               "evidence": "Alpha and Beta", "confidence": 0.5}],
            }
        marker = "Candidate relations:\n"
        payload = kwargs["prompt"].split(marker, 1)[1].split("\n\nSOURCE TEXT:", 1)[0]
        candidate = __import__("json").loads(payload)[0]
        return {"decisions": [{
            "candidate_id": candidate["candidate_id"], "verdict": "insufficient_evidence",
            "corrected_type": "produces", "orientation": "as_is",
            "evidence": "Alpha and Beta", "reason": "co-occurrence only",
        }]}


def test_uncertain_relation_can_be_rejected_by_batch_verifier(monkeypatch):
    monkeypatch.setattr(Config, "extraction_gleaning_enabled", False)
    monkeypatch.setattr(Config, "extraction_verification_enabled", True)
    ollama = RejectingRelationOllama()
    result = SchemaEvolutionService(None, ollama).extract_chunk(
        {"llm_model": "test"}, {"schema_json": BASE_SCHEMA}, {"content": "Alpha and Beta are discussed."}
    )
    assert ollama.calls == 2
    assert result["relations"][0]["verification_status"] == "uncertain"
    assert result["_diagnostics"]["verification_rejected"] == 1


def test_manual_reassessment_rebuilds_from_base_schema():
    chunks = [
        _chunk("industry_p001", "industry", RICH_INDUSTRY, [1.0, 0.0]),
        _chunk("supply_p001", "supply", RICH_SUPPLY, [0.0, 1.0]),
    ]
    database = FakeDatabase(chunks)
    result = SchemaEvolutionService(database, FakeOllama()).reassess_project(
        {"id": 3, "active_schema_version": 5, "llm_model": "test", "embedding_model": "embed"},
        [{"id": 7, "original_filename": "industry.pdf"}],
    )
    assert result["version"] == 2
    assert database.created[4] == "manual"
    assert database.created[2]["previous_active_version"] == 5
