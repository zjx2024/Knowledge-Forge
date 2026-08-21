import json

from knowledge_forge.schema_evolution import (
    BASE_SCHEMA,
    ENTITY_ONLY_EXTRACTION_FORMAT,
    GROUNDED_MENTION_INVENTORY_FORMAT,
    OPEN_RELATION_BATCH_INDUCTION_FORMAT,
    OPEN_RELATION_INDUCTION_FORMAT,
    PROPOSITION_ARGUMENT_EXTRACTION_FORMAT,
    SCHEMA_EVIDENCE_GROUNDING_FORMAT,
    SCHEMA_POLICY_REPAIR_FORMAT,
    SCHEMA_POLICY_NOVELTY_FORMAT,
    SCHEMA_POLICY_VERIFICATION_FORMAT,
    SCHEMA_DISCOVERY_PROTOCOL_VERSION,
    SchemaEvolutionService,
    _normalize_open_relation_expression,
    _relation_expression_family_key,
    _source_relation_identifier,
    _strip_endpoint_suffix_from_identifier,
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


def test_open_relation_family_normalizes_modality_and_role_carriers():
    assert _normalize_open_relation_expression(
        "has responsibility to check"
    ) == "check"
    assert _normalize_open_relation_expression("shall check") == "check"
    assert _normalize_open_relation_expression("负责检查") == "检查"
    assert _relation_expression_family_key(
        "has responsibility to checks"
    ) == _relation_expression_family_key("check")
    assert _source_relation_identifier("is") == ""
    assert _source_relation_identifier("has") == ""
    assert _source_relation_identifier("is linked to") == "linked_to"
    assert _strip_endpoint_suffix_from_identifier(
        "receives_from_person", [], ("Person",),
    ) == "receives_from"
    assert _strip_endpoint_suffix_from_identifier(
        "arrives_at_location", [], ("Location",),
    ) == "arrives_at"


def test_schema_consolidation_collapses_only_source_provable_suffix_duplicate():
    observation = {
        "chunk_key": "network_p001",
        "subject_surface": "resource",
        "object_surface": "URI",
        "evidence_quote": "A resource is identified by a URI, not a connection.",
        "relation_expression": "identified by",
    }
    overqualified = {
        **observation,
        "relation_expression": "identified by URI not connection",
    }
    review = SchemaEvolutionService(
        None, FakeOllama(),
    )._consolidate_candidates(
        {"llm_model": "test"},
        BASE_SCHEMA,
        [{
            "entity_type_candidates": [],
            "relation_type_candidates": [
                {
                    "name": "identified_by",
                    "confidence": 0.96,
                    "evidence_chunks": ["network_p001"],
                    "evidence_quotes": [observation["evidence_quote"]],
                    "_induction_observations": [observation],
                },
                {
                    "name": "identified_by_URI_not_connection",
                    "confidence": 0.97,
                    "evidence_chunks": ["network_p001"],
                    "evidence_quotes": [observation["evidence_quote"]],
                    "_induction_observations": [overqualified],
                },
            ],
        }],
    )

    assert [
        item["name"] for item in review["relation_type_candidates"]
    ] == ["identified_by"]
    assert review["consolidation_diagnostics"] == {
        "source_overqualified_duplicates_collapsed": 1,
    }


def test_schema_discovery_adds_source_structure_without_mechanical_predicate_names():
    class StructuralSchemaOllama:
        def __init__(self):
            self.groups = []

        def json_chat(self, **kwargs):
            system = kwargs["system"]
            if "record every explicit binary semantic observation" in system:
                return {"items": [
                    {
                        "subject": "Alpha Service",
                        "relation_expression": "checks",
                        "object": "flight intent",
                        "evidence": (
                            "| Alpha Service | checks flight intent | SR-01 |"
                        ),
                        "assertion_status": "AFFIRMED",
                    },
                    {
                        "subject": "Alpha Service",
                        "relation_expression": "shall check",
                        "object": "flight intent",
                        "evidence": (
                            "SR-01 Alpha Service shall check flight intent."
                        ),
                        "assertion_status": "AFFIRMED",
                    },
                ]}
            if "Induce reusable logical knowledge graph relation families" in system:
                request = json.loads(kwargs["prompt"])
                groups = request["validated_observation_groups"]
                self.groups.extend(groups)
                rows = []
                for group in groups:
                    observations = group["observations"]
                    kinds = {
                        item.get("structural_evidence_kind")
                        for item in observations
                    }
                    if "table_header_assignment" in kinds:
                        name = "mapped_reference"
                    elif "line_identifier_binding" in kinds:
                        name = "governs_topic"
                    else:
                        name = "model_check_name"
                    rows.append({
                        "observation_group_id": group["observation_group_id"],
                        "name": name,
                        "description": "表示原文明示的有向语义角色。",
                    })
                return {"rows": rows}
            return {
                "summary": "", "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    text = (
        "| Service | Responsibility | Requirement |\n"
        "| Alpha Service | checks flight intent | SR-01 |\n\n"
        "Requirements\n\n"
        "SR-01 Alpha Service shall check flight intent."
    )
    sample = _chunk("structure_1", "section", text, [1.0, 0.0])
    sample.update({
        "_schema_text": text,
        "_schema_section": "section",
        "_schema_order": 0,
    })
    gateway = StructuralSchemaOllama()
    batches, errors = SchemaEvolutionService(
        None, gateway,
    )._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192},
        BASE_SCHEMA, [sample],
    )

    assert errors == []
    candidates = [
        item for batch in batches
        for item in batch.get("relation_type_candidates") or []
    ]
    by_name = {item["name"]: item for item in candidates}
    assert "mapped_reference" in by_name
    assert "column_role_requirement" not in by_name
    assert "label_binds_topic" not in by_name
    assert any(
        item.get("structural_evidence_kind")
        == "table_header_assignment"
        and item["subject_surface"] == "Alpha Service"
        and item["object_surface"] == "SR-01"
        for item in by_name["mapped_reference"]["_induction_observations"]
    )
    # BASE_SCHEMA already declares both identifier/topic and section-label
    # parent/child support, so neither observation consumes an additive slot.
    assert not any(
        item.get("structural_evidence_kind") in {
            "line_identifier_binding", "section_label_parent_child",
        }
        for group in gateway.groups for item in group["observations"]
    )
    scopes_topic = next(
        item for item in BASE_SCHEMA["relation_types"]
        if item["name"] == "scopes_topic"
    )
    assert scopes_topic["structural_evidence_kinds"] == [
        "line_identifier_binding"
    ]


def test_schema_discovery_recovers_nominal_table_columns_without_model_facts():
    class TableSchemaOllama:
        def json_chat(self, **kwargs):
            system = kwargs["system"]
            if "record every explicit binary semantic observation" in system:
                return {"items": []}
            if "Induce reusable logical knowledge graph relation families" in system:
                request = json.loads(kwargs["prompt"])
                return {"rows": [{
                    "observation_group_id": group["observation_group_id"],
                    "name": (
                        "source_role"
                        if group["relation_expression"].endswith("Source")
                        else "purpose_role"
                    ),
                    "description": "表示表头明示的有向语义角色。",
                } for group in request["validated_observation_groups"]]}
            return {
                "summary": "", "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    text = (
        "Data set | Source | Purpose\n"
        "Principal aquifers | National Atlas | aquifer mapping\n"
        "Surficial geology | State compilations | geologic context\n"
        "Ground-water regions | USGS compilation | regional analysis"
    )
    sample = _chunk("table_1", "section", text, [1.0, 0.0])
    sample.update({
        "_schema_text": text,
        "_schema_section": "section",
        "_schema_order": 0,
    })
    batches, errors = SchemaEvolutionService(
        None, TableSchemaOllama(),
    )._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192},
        BASE_SCHEMA, [sample],
    )

    assert errors == []
    candidates = [
        item for batch in batches
        for item in batch.get("relation_type_candidates") or []
    ]
    observations = {
        item["name"]: item["_induction_observations"]
        for item in candidates
    }
    assert len(observations["source_role"]) == 3
    assert len(observations["purpose_role"]) == 3
    assert all(
        item.get("semantic_naming_required") is True
        and item.get("observation_origin") == "source_structure"
        for values in observations.values() for item in values
    )


class FakeDatabase:
    def __init__(self, chunks=None, prior_documents=0, nearest_score=0.2):
        self.active = {"version": 1, "schema_json": BASE_SCHEMA}
        self.created = None
        self.chunks = chunks or []
        self.prior_documents = prior_documents
        self.nearest_score = nearest_score
        self.embedding_revision_id = None

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

    def list_document_embeddings(
        self, document_id, object_type, model_name, *, revision_id=None,
    ):
        self.embedding_revision_id = revision_id
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
        if kwargs.get("schema") is SCHEMA_POLICY_VERIFICATION_FORMAT:
            return {"verdict": "SUPPORTED", "reason": "角色与方向一致。"}
        if kwargs.get("schema") is SCHEMA_POLICY_NOVELTY_FORMAT:
            return {
                "verdict": "NOVEL",
                "reason": "现有关系不能表示这组端点角色与语义。",
                "overlapping_relation": None,
                "exact_role_alignment": False,
                "core_contract_equivalent": False,
                "orientation_alignment": "none",
            }
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
                    "description": "一个组织向另一个组织供应资源。",
                    "source_types": ["Organization"],
                    "target_types": ["Organization"],
                    "semantic_definition": "一个组织向另一个组织供应资源。",
                    "domain_types": ["Organization"],
                    "range_types": ["Organization"],
                    "direction_description": "从供应方组织指向接收方组织。",
                    "subject_role_description": "提供资源的供应方组织。",
                    "object_role_description": "接收资源的组织。",
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


def test_existing_project_gets_runtime_open_capability_additively():
    legacy = json.loads(json.dumps(BASE_SCHEMA, ensure_ascii=False))
    legacy["version"] = "legacy-schema-v1"
    legacy["relation_types"] = [
        item for item in legacy["relation_types"]
        if not bool((item.get("open_relation_policy") or {}).get("enabled"))
    ]
    original_relations = json.loads(json.dumps(
        legacy["relation_types"], ensure_ascii=False,
    ))
    database = FakeDatabase()
    database.active = {
        "version": 4,
        "schema_json": legacy,
        "source_document_ids": [7, 8],
    }

    upgraded = SchemaEvolutionService(database, FakeOllama()).ensure_base(3)

    assert upgraded["version"] == 2
    assert database.created is not None
    project_id, schema, report, documents, created_by = database.created
    assert project_id == 3
    assert schema["relation_types"][:len(original_relations)] == original_relations
    added = schema["relation_types"][len(original_relations):]
    assert len(added) == 1
    assert (added[0]["open_relation_policy"] or {})["fallback_only"] is True
    assert report["policy"] == "additive-runtime-capability-upgrade"
    assert documents == [7, 8]
    assert created_by == "system"


class CorpusDatabase(FakeDatabase):
    def __init__(self, chunks_by_document):
        all_chunks = [
            item for values in chunks_by_document.values() for item in values
        ]
        super().__init__(all_chunks)
        self.chunks_by_document = chunks_by_document
        self.create_calls = []
        self.evaluations = []

    def list_document_chunks(self, document_id):
        return list(self.chunks_by_document.get(document_id, []))

    def list_document_embeddings(
        self, document_id, object_type, model_name, *, revision_id=None,
    ):
        return [{
            "object_key": item["chunk_key"],
            "embedding": item["vector"],
        } for item in self.chunks_by_document.get(document_id, [])]

    def create_schema_version(
        self, project_id, schema, report, documents, created_by="llm",
    ):
        self.create_calls.append((
            project_id, schema, report, documents, created_by,
        ))
        self.created = self.create_calls[-1]
        return {
            "version": 2,
            "schema_json": schema,
            "compatibility_report": report,
        }

    def record_schema_evaluation(self, *args, **kwargs):
        self.evaluations.append((args, kwargs))


def test_project_document_batch_publishes_at_most_one_schema_version(monkeypatch):
    monkeypatch.setattr(Config, "schema_corpus_review_batch_size", 4)
    monkeypatch.setattr(Config, "schema_relation_coverage_rounds", 0)
    monkeypatch.setattr(Config, "schema_relation_compound_audit_enabled", False)
    industry = _chunk(
        "industry_p001", "industry", RICH_INDUSTRY, [1.0, 0.0],
    )
    supply = _chunk(
        "supply_p001", "supply", RICH_SUPPLY, [0.0, 1.0],
    )
    industry.update({"document_id": 7, "revision_id": "revision-7"})
    supply.update({"document_id": 8, "revision_id": "revision-8"})
    database = CorpusDatabase({7: [industry], 8: [supply]})
    result = SchemaEvolutionService(database, FakeOllama()).evaluate_corpus(
        {"id": 3, "llm_model": "test", "embedding_model": "embed"},
        [
            {"id": 7, "active_revision_id": "revision-7", "doc_hash": "h7"},
            {"id": 8, "active_revision_id": "revision-8", "doc_hash": "h8"},
        ],
    )

    assert result["version"] == 2
    assert len(database.create_calls) == 1
    assert database.create_calls[0][3] == [7, 8]
    assert result["_evaluation"]["scope"] == "project_document_batch"
    assert result["_evaluation"]["sampling"]["selected_documents"] == 2
    assert len(database.evaluations) == 2

    database.active = {
        "version": result["version"],
        "schema_json": result["schema_json"],
        "compatibility_report": database.create_calls[0][2],
    }
    reused = SchemaEvolutionService(database, FakeOllama()).evaluate_corpus(
        {"id": 3, "llm_model": "test", "embedding_model": "embed"},
        [
            {"id": 7, "active_revision_id": "revision-7", "doc_hash": "h7"},
            {"id": 8, "active_revision_id": "revision-8", "doc_hash": "h8"},
        ],
    )
    assert reused["version"] == 2
    assert reused["_evaluation"]["reason"] == (
        "identical_batch_already_evaluated"
    )
    assert len(database.create_calls) == 1


def test_project_document_batch_sampling_has_one_global_ceiling(monkeypatch):
    monkeypatch.setattr(Config, "schema_sample_base_chunks", 2)
    monkeypatch.setattr(Config, "schema_sample_ceiling_chunks", 6)
    monkeypatch.setattr(Config, "schema_sample_base_chars", 1000)
    monkeypatch.setattr(Config, "schema_sample_ceiling_chars", 5000)
    monkeypatch.setattr(Config, "schema_sample_chars_per_chunk", 800)
    chunks_by_document = {}
    documents = []
    for document_id in range(1, 11):
        chunks = []
        for chunk_index in range(2):
            chunk = _chunk(
                f"d{document_id}_p{chunk_index}",
                f"section-{chunk_index}",
                RICH_INDUSTRY + str(document_id),
                [float(document_id), float(chunk_index + 1)],
            )
            chunk.update({
                "document_id": document_id,
                "revision_id": f"revision-{document_id}",
            })
            chunks.append(chunk)
        chunks_by_document[document_id] = chunks
        documents.append({
            "id": document_id,
            "active_revision_id": f"revision-{document_id}",
            "doc_hash": f"hash-{document_id}",
        })
    samples, diagnostics = SchemaEvolutionService(
        CorpusDatabase(chunks_by_document), FakeOllama(),
    ).select_representative_corpus_chunks(
        {"id": 3, "embedding_model": "embed"}, documents,
    )

    assert len(samples) == 6
    assert diagnostics["target_chunks"] == 6
    assert diagnostics["selected_documents"] == 6
    assert diagnostics["selected_chars"] <= 5000


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


def test_representative_sampling_keeps_short_semantic_chinese_sections():
    chunks = [
        _chunk(
            "short_a_p001", "short_a",
            "甲公司向乙工厂供应控制器，乙工厂将控制器安装到装配线。", [1.0, 0.0],
        ),
        _chunk(
            "short_b_p001", "short_b",
            "物流中心把成品运送至示范园区，园区运营方负责设备巡检。", [0.0, 1.0],
        ),
    ]
    samples, diagnostics = SchemaEvolutionService(
        FakeDatabase(chunks), FakeOllama(),
    ).select_representative_chunks(
        {"id": 3, "embedding_model": "embed"}, 7, chunks,
    )
    assert {_chunk["chunk_key"] for _chunk in samples} == {
        "short_a_p001", "short_b_p001",
    }
    assert diagnostics["filtered_chunks"] == 0


def test_representative_sampling_reads_vectors_from_staged_revision():
    chunks = [
        {
            **_chunk(
                "staged_p001", "staged",
                "甲公司向乙工厂供应控制器，并明确记录该供应关系。", [1.0, 0.0],
            ),
            "revision_id": "revision-staged",
        },
    ]
    database = FakeDatabase(chunks)

    _samples, diagnostics = SchemaEvolutionService(
        database, FakeOllama(),
    ).select_representative_chunks(
        {"id": 3, "embedding_model": "embed"}, 7, chunks,
    )

    assert database.embedding_revision_id == "revision-staged"
    assert diagnostics["embedding_revision_scope"] == "revision-staged"
    assert diagnostics["embedding_coverage"] == 1


def test_schema_evaluation_audit_records_no_change():
    class AuditDatabase(FakeDatabase):
        def __init__(self, chunks):
            super().__init__(chunks)
            self.evaluations = []

        def record_schema_evaluation(self, *args, **kwargs):
            self.evaluations.append((args, kwargs))

    class EmptyReviewOllama:
        def json_chat(self, **kwargs):
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    chunks = [_chunk("short_a_p001", "short_a", RICH_INDUSTRY, [1.0, 0.0])]
    database = AuditDatabase(chunks)
    result = SchemaEvolutionService(database, EmptyReviewOllama()).evaluate_document(
        {"id": 3, "llm_model": "test", "embedding_model": "embed"}, 7, chunks,
    )
    assert result["version"] == 1
    assert database.evaluations[0][1]["outcome"] == "no_change"


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


def test_schema_review_splits_a_truncated_batch_without_weakening_gate(monkeypatch):
    monkeypatch.setattr(Config, "schema_review_batch_size", 4)
    class SplitOnTruncationOllama(FakeOllama):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def json_chat(self, **kwargs):
            size = str(kwargs["prompt"]).count("[short_")
            self.batch_sizes.append(size)
            if size > 1:
                raise OllamaProtocolError(
                    "Ollama response was truncated by output length"
                )
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    chunks = [
        _chunk(
            f"short_{index}_p001", f"short_{index}",
            f"甲公司向乙工厂供应第{index}类设备，并由乙工厂负责安装和维护。",
            [float(index), 1.0],
        )
        for index in range(4)
    ]
    ollama = SplitOnTruncationOllama()
    result = SchemaEvolutionService(FakeDatabase(chunks), ollama).evaluate_document(
        {"id": 3, "llm_model": "test", "embedding_model": "embed"}, 7, chunks,
    )
    assert result["_evaluation"]["batch_errors"] == []
    assert ollama.batch_sizes[0] == 4
    # Each recovered singleton receives the ordinary review plus the configured
    # number of bounded, deduplicated open-observation coverage rounds.
    assert ollama.batch_sizes.count(1) == 4 * (
        1 + Config.schema_relation_coverage_rounds
    )


def test_schema_relation_coverage_audit_recovers_a_family_missed_by_broad_review():
    class CoverageOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            self.calls.append(kwargs)
            if "record every explicit binary semantic observation" in kwargs["system"]:
                return {"items": [
                    {
                        "subject": "甲公司",
                        "relation_expression": "向……提供",
                        "object": "设备",
                        "evidence": "甲公司向乙工厂提供设备。",
                        "assertion_status": "AFFIRMED",
                    },
                    {
                        "subject": "丙公司",
                        "relation_expression": "向……提供",
                        "object": "材料",
                        "evidence": "丙公司向丁工厂提供材料。",
                        "assertion_status": "AFFIRMED",
                    },
                ]}
            if "Induce reusable logical knowledge graph relation families" not in kwargs["system"]:
                return {
                    "summary": "",
                    "entity_type_candidates": [],
                    "relation_type_candidates": [],
                }
            return {
                "induced_relation_family": {
                    "name": "alpha_rel",
                    "description": "主体向客体提供一种资源。",
                    "semantic_definition": "主体向客体提供一种资源。",
                    "source_types": ["Organization"],
                    "target_types": ["Product"],
                    "domain_types": ["Organization"],
                    "range_types": ["Product"],
                    "direction_description": "从提供资源的主体指向被提供的资源。",
                    "subject_role_description": "提供资源的主体。",
                    "object_role_description": "被提供的资源。",
                    "symmetric": False,
                    "inverse_predicate": None,
                    "canonical_orientation": "forward",
                    "publish_mode": "auto",
                    "reflexive": False,
                    "positive_examples": ["甲方明确向乙方提供设备。"],
                    "negative_examples": ["甲方与设备只是在同段出现。"],
                    "optional_soft_cues": [],
                    "evidence_chunks": ["chunk_1"],
                    "evidence_quotes": [
                        "甲公司向乙工厂提供设备。",
                        "丙公司向丁工厂提供材料。",
                    ],
                    "confidence": 0.99,
                },
            }

    sample = _chunk(
        "chunk_1", "section",
        "甲公司向乙工厂提供设备。丙公司向丁工厂提供材料。", [1.0, 0.0],
    )
    sample.update({
        "_schema_text": sample["content"],
        "_schema_section": "section",
        "_schema_order": 0,
    })
    ollama = CoverageOllama()
    batches, errors = SchemaEvolutionService(None, ollama)._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192}, BASE_SCHEMA, [sample],
    )

    assert errors == []
    assert len(ollama.calls) == 2 + Config.schema_relation_coverage_rounds
    assert batches[-1]["entity_type_candidates"] == []
    assert [
        item["name"] for item in batches[-1]["relation_type_candidates"]
    ] == ["alpha_rel"]


def test_schema_relation_coverage_can_induce_a_clear_single_observation():
    class SingletonCoverageOllama:
        def json_chat(self, **kwargs):
            if "record every explicit binary semantic observation" in kwargs["system"]:
                return {"observations": [{
                    "subject": "甲组织",
                    "relation_expression": "执行",
                    "object": "乙任务",
                    "evidence": "甲组织执行乙任务。",
                    "assertion_status": "AFFIRMED",
                }]}
            if "Induce reusable logical knowledge graph relation families" in kwargs["system"]:
                return {"induced_relation_family": {
                    "name": "alpha_rel",
                    "description": "主体明确执行客体。",
                    "semantic_definition": "主体对客体执行明确动作。",
                    "source_types": ["Organization"],
                    "target_types": ["Event"],
                    "domain_types": ["Organization"],
                    "range_types": ["Event"],
                    "direction_description": "从执行方指向被执行的任务。",
                    "subject_role_description": "执行动作的主体。",
                    "object_role_description": "被执行的任务。",
                    "symmetric": False,
                    "inverse_predicate": None,
                    "canonical_orientation": "forward",
                    "publish_mode": "auto",
                    "reflexive": False,
                    "positive_examples": ["甲组织执行乙任务。"],
                    "negative_examples": ["甲组织与乙任务只是在同段出现。"],
                    "optional_soft_cues": [],
                    "evidence_chunks": ["singleton_1"],
                    "evidence_quotes": ["甲组织执行乙任务。"],
                    "confidence": 0.99,
                }}
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    sample = _chunk(
        "singleton_1", "section", "甲组织执行乙任务。", [1.0, 0.0],
    )
    sample.update({
        "_schema_text": sample["content"],
        "_schema_section": "section",
        "_schema_order": 0,
    })
    batches, errors = SchemaEvolutionService(
        None, SingletonCoverageOllama(),
    )._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192},
        BASE_SCHEMA,
        [sample],
    )

    assert errors == []
    assert [
        item["name"] for item in batches[-1]["relation_type_candidates"]
    ] == ["alpha_rel"]


def test_open_relation_induction_retries_truncation_with_bounded_relation_only_schema():
    class RecoveringInductionOllama:
        def __init__(self):
            self.induction_calls = []

        def json_chat(self, **kwargs):
            if "record every explicit binary semantic observation" in kwargs["system"]:
                return {"items": [
                    {
                        "subject": "甲组织",
                        "relation_expression": "执行协调",
                        "object": "乙组织",
                        "evidence": "甲组织执行协调乙组织。",
                        "assertion_status": "AFFIRMED",
                    },
                    {
                        "subject": "丙组织",
                        "relation_expression": "执行协调",
                        "object": "丁组织",
                        "evidence": "丙组织执行协调丁组织。",
                        "assertion_status": "AFFIRMED",
                    },
                ]}
            if "Induce reusable logical knowledge graph relation families" in kwargs["system"]:
                self.induction_calls.append(kwargs)
                if len(self.induction_calls) == 1:
                    raise OllamaProtocolError(
                        "Ollama response was truncated by output length"
                    )
                return {
                    "summary": "归纳出可复用的协调关系。",
                    "entity_type_candidates": [],
                    "relation_type_candidates": [{
                        "name": "alpha_rel",
                        "description": "主体明确协调客体。",
                        "source_types": ["Organization"],
                        "target_types": ["Organization"],
                        "semantic_definition": "主体对客体承担明确协调职责。",
                        "domain_types": ["Organization"],
                        "range_types": ["Organization"],
                        "direction_description": "从协调方指向被协调方。",
                        "subject_role_description": "实施协调的组织。",
                        "object_role_description": "被协调的组织。",
                        "symmetric": False,
                        "inverse_predicate": None,
                        "canonical_orientation": "forward",
                        "publish_mode": "auto",
                        "reflexive": False,
                        "positive_examples": ["甲组织执行协调乙组织。"],
                        "negative_examples": ["两个组织只是在同段出现。"],
                        "optional_soft_cues": [],
                        "confidence": 0.95,
                        "evidence_chunks": ["chunk_1"],
                        "evidence_quotes": [
                            "甲组织执行协调乙组织。",
                            "丙组织执行协调丁组织。",
                        ],
                    }],
                }
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    sample = _chunk(
        "chunk_1", "section",
        "甲组织执行协调乙组织。丙组织执行协调丁组织。", [1.0, 0.0],
    )
    sample.update({
        "_schema_text": sample["content"],
        "_schema_section": "section",
        "_schema_order": 0,
    })
    ollama = RecoveringInductionOllama()
    batches, errors = SchemaEvolutionService(None, ollama)._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192}, BASE_SCHEMA, [sample],
    )

    assert errors == []
    assert len(ollama.induction_calls) == 2
    assert all(
        call["schema"] is OPEN_RELATION_BATCH_INDUCTION_FORMAT
        and call["max_output_tokens"] <= 8192
        for call in ollama.induction_calls
    )
    assert OPEN_RELATION_INDUCTION_FORMAT["properties"][
        "entity_type_candidates"
    ]["maxItems"] == 0
    assert OPEN_RELATION_INDUCTION_FORMAT["properties"][
        "relation_type_candidates"
    ]["maxItems"] == 1
    assert [
        item["name"] for batch in batches
        for item in batch.get("relation_type_candidates") or []
    ] == ["alpha_rel"]


def test_open_relation_omission_audit_recovers_theme_to_third_party_from_aliases():
    class TransferCoverageOllama:
        def __init__(self):
            self.observation_calls = 0

        def json_chat(self, **kwargs):
            if "record every explicit binary semantic observation" in kwargs["system"]:
                self.observation_calls += 1
                if self.observation_calls == 1:
                    return {"items": [
                        {
                            "subject": "Sender",
                            "relation_expression": "submits",
                            "object": "item",
                            "evidence": "Sender submits an item to Receiver.",
                            "assertion_status": "AFFIRMED",
                        },
                        {
                            "subject": "Sender",
                            "relation_expression": "submit",
                            "object": "an item to Receiver",
                            "evidence": "Sender submits an item to Receiver.",
                            "assertion_status": "AFFIRMED",
                        },
                    ]}
                # Models sometimes copy the previously supplied internal field
                # names instead of the requested source transport aliases.
                return {"observations": [
                    {
                        "subject_surface": "Sender",
                        "relation_expression": "submits to",
                        "object_surface": "item",
                        "evidence_quote": "Sender submits an item to Receiver.",
                        "assertion_status": "AFFIRMED",
                    },
                    {
                        "subject_surface": "Sender",
                        "relation_expression": "submits to",
                        "object_surface": "Receiver",
                        "evidence_quote": "Sender submits an item to Receiver.",
                        "assertion_status": "AFFIRMED",
                    },
                ]}
            if "Induce reusable logical knowledge graph relation families" in kwargs["system"]:
                request = json.loads(kwargs["prompt"])
                return {"rows": [
                    {
                        "observation_group_id": group["observation_group_id"],
                        "name": (
                            "submitted_to_receiver"
                            if any(
                                item.get("ordered_role_projection")
                                for item in group["observations"]
                            )
                            else "submits"
                        ),
                        "description": "表示原文明示的有向关系。",
                    }
                    for group in request["validated_observation_groups"]
                ]}
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    text = "Sender submits an item to Receiver."
    sample = _chunk("transfer_1", "section", text, [1.0, 0.0])
    sample.update({
        "_schema_text": text,
        "_schema_section": "section",
        "_schema_order": 0,
    })
    batches, errors = SchemaEvolutionService(
        None, TransferCoverageOllama(),
    )._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192},
        BASE_SCHEMA,
        [sample],
    )

    assert errors == []
    relations = [
        item for batch in batches
        for item in batch.get("relation_type_candidates") or []
    ]
    submitted_to = next(
        item for item in relations if item["name"] == "submitted_to"
    )
    submits = next(item for item in relations if item["name"] == "submits")
    assert submits["source_role_pairs"] == [
        "grammatical_subject->action_theme",
    ]
    assert submitted_to["source_role_pairs"] == [
        "action_theme->role_complement",
    ]
    assert [
        (
            item["subject_surface"],
            item["object_surface"],
            item.get("ordered_role_projection"),
        )
        for item in submitted_to["_induction_observations"]
    ] == [("item", "Receiver", "theme_to_third_party")]
    assert not any(
        item["subject_surface"] == "Sender"
        for item in submitted_to["_induction_observations"]
    )


def test_open_relation_compound_audit_splits_explicit_container_connection():
    class CompoundCoverageOllama:
        def __init__(self):
            self.coverage_calls = 0
            self.compound_calls = 0

        def json_chat(self, **kwargs):
            system = kwargs["system"]
            if "record every explicit binary semantic observation" in system:
                self.coverage_calls += 1
                if self.coverage_calls == 1:
                    return {"items": [{
                        "subject": "index",
                        "relation_expression": "links",
                        "object": "a module to a requirement",
                        "evidence": "The index links a module to a requirement.",
                        "assertion_status": "AFFIRMED",
                    }]}
                return {"items": []}
            if "unresolved compound-endpoint observations" in system:
                self.compound_calls += 1
                evidence = "The index links a module to a requirement."
                return {"items": [
                    {
                        "subject": "index", "relation_expression": "links",
                        "object": "module", "evidence": evidence,
                        "assertion_status": "AFFIRMED",
                    },
                    {
                        "subject": "index", "relation_expression": "links",
                        "object": "requirement", "evidence": evidence,
                        "assertion_status": "AFFIRMED",
                    },
                    {
                        "subject": "module",
                        "relation_expression": "maps to",
                        "object": "requirement", "evidence": evidence,
                        "assertion_status": "AFFIRMED",
                    },
                ]}
            if "Induce reusable logical knowledge graph relation families" in system:
                request = json.loads(kwargs["prompt"])
                return {"rows": [{
                    "observation_group_id": group["observation_group_id"],
                    "name": group["relation_expression"].replace(" ", "_"),
                    "description": "表示原文明示的连接关系。",
                } for group in request["validated_observation_groups"]]}
            return {
                "summary": "", "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    text = "The index links a module to a requirement."
    sample = _chunk("compound_1", "section", text, [1.0, 0.0])
    sample.update({
        "_schema_text": text,
        "_schema_section": "section",
        "_schema_order": 0,
    })
    gateway = CompoundCoverageOllama()
    batches, errors = SchemaEvolutionService(
        None, gateway,
    )._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192},
        BASE_SCHEMA, [sample],
    )

    assert errors == []
    assert gateway.compound_calls == 1
    observations = [
        observation
        for batch in batches
        for relation in batch.get("relation_type_candidates") or []
        for observation in relation.get("_induction_observations") or []
    ]
    assert not any(
        item["object_surface"] == "a module to a requirement"
        for item in observations
    )
    assert {
        (item["subject_surface"], item["object_surface"])
        for item in observations
    } == {
        ("index", "module"),
        ("index", "requirement"),
        ("module", "requirement"),
    }
    contracts = {
        relation["name"]: set(relation.get("source_role_pairs") or [])
        for batch in batches
        for relation in batch.get("relation_type_candidates") or []
    }
    assert contracts["links"] == {
        "grammatical_subject->action_theme",
        "grammatical_subject->role_complement",
    }
    assert contracts["maps_to"] == {
        "action_theme->role_complement",
    }


def test_schema_review_keeps_flattened_single_relation_candidate():
    value = {
        "name": "alpha_rel",
        "semantic_definition": "主体以明确方式作用于客体。",
        "domain_types": ["Organization"],
        "range_types": ["Product"],
        "direction_description": "从作用方指向承受方。",
    }
    payload = SchemaEvolutionService._candidate_payload(
        value, fallback_evidence_chunks=["chunk_1"],
    )
    assert payload["relation_type_candidates"] == [{
        **value, "evidence_chunks": ["chunk_1"],
    }]


def test_schema_review_normalizes_model_relation_transport_aliases():
    relation = {
        "name": "alpha_rel",
        "description": "主体向客体提供资源。",
        "semantic_definition": "主体向客体提供资源。",
        "source_types": ["Organization"],
        "target_types": ["Product"],
        "direction": "forward",
    }
    candidates = SchemaEvolutionService._candidate_payload(
        {"relation_candidates": [relation]},
        fallback_evidence_chunks=["chunk_1"],
    )["relation_type_candidates"]
    assert candidates[0]["name"] == "alpha_rel"
    assert candidates[0]["direction_description"] == "forward"
    assert candidates[0]["domain_types"] == ["Organization"]
    assert candidates[0]["range_types"] == ["Product"]

    family = SchemaEvolutionService._candidate_payload(
        {"candidate_relation_family": relation},
        fallback_evidence_chunks=["chunk_1"],
    )["relation_type_candidates"]
    assert family[0]["name"] == "alpha_rel"
    assert family[0]["evidence_chunks"] == ["chunk_1"]


def test_schema_candidate_grounding_recovers_exact_source_quotes():
    class GroundingOllama:
        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_EVIDENCE_GROUNDING_FORMAT:
                return {
                    "evidence_quotes": [
                        "甲公司向乙公司提供设备。",
                        "丙公司向丁公司提供材料。",
                    ],
                    "confidence": 0.97,
                }
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [{
                    "name": "alpha_rel",
                    "description": "主体向客体提供资源。",
                    "source_types": ["Organization"],
                    "target_types": ["Organization"],
                    "semantic_definition": "主体向客体提供资源。",
                    "domain_types": ["Organization"],
                    "range_types": ["Organization"],
                    "direction_description": "从提供方指向接收方。",
                    "subject_role_description": "提供资源的组织。",
                    "object_role_description": "接收资源的组织。",
                    "symmetric": False,
                    "inverse_predicate": None,
                    "canonical_orientation": "forward",
                    "publish_mode": "auto",
                    "reflexive": False,
                    "positive_examples": ["A provides B."],
                    "negative_examples": ["A and B merely co-occur."],
                    "optional_soft_cues": [],
                    "evidence_chunks": ["[chunk_1]"],
                }],
            }

    sample = _chunk(
        "chunk_1", "section",
        "甲公司向乙公司提供设备。丙公司向丁公司提供材料。", [1.0, 0.0],
    )
    sample.update({
        "_schema_text": sample["content"],
        "_schema_section": "section",
        "_schema_order": 0,
    })
    service = SchemaEvolutionService(None, GroundingOllama())

    batches, errors = service._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192}, BASE_SCHEMA, [sample],
    )

    candidate = batches[0]["relation_type_candidates"][0]
    assert errors == []
    assert candidate["evidence_chunks"] == ["chunk_1"]
    assert candidate["evidence_quotes"] == [
        "甲公司向乙公司提供设备。", "丙公司向丁公司提供材料。",
    ]
    assert candidate["confidence"] == 0.97


def test_schema_candidate_grounding_accepts_top_level_array_transport_only_with_exact_quotes():
    class ArrayGroundingOllama:
        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_EVIDENCE_GROUNDING_FORMAT:
                return {"items": [
                    "甲公司通过中间货物向乙公司提供设备。",
                    "丙公司通过中间服务向丁公司提供支持。",
                ]}
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [{
                    "name": "alpha_rel",
                    "description": "主体向客体提供资源。",
                    "source_types": ["Organization"],
                    "target_types": ["Organization"],
                    "semantic_definition": "主体向客体提供资源。",
                    "domain_types": ["Organization"],
                    "range_types": ["Organization"],
                    "direction_description": "从提供方指向接收方。",
                    "subject_role_description": "提供资源的组织。",
                    "object_role_description": "接收资源的组织。",
                    "symmetric": False,
                    "inverse_predicate": None,
                    "canonical_orientation": "forward",
                    "publish_mode": "auto",
                    "reflexive": False,
                    "positive_examples": ["A provides B."],
                    "negative_examples": ["A and B merely co-occur."],
                    "optional_soft_cues": [],
                    "evidence_chunks": ["chunk_1"],
                    "evidence_quotes": [],
                }],
            }

    sample = _chunk(
        "chunk_1", "section",
        "甲公司通过中间货物向乙公司提供设备。丙公司通过中间服务向丁公司提供支持。",
        [1.0, 0.0],
    )
    sample.update({
        "_schema_text": sample["content"],
        "_schema_section": "section",
        "_schema_order": 0,
    })
    service = SchemaEvolutionService(None, ArrayGroundingOllama())

    batches, errors = service._discover_candidates(
        {"llm_model": "test", "model_max_tokens": 8192}, BASE_SCHEMA, [sample],
    )

    candidate = batches[0]["relation_type_candidates"][0]
    assert errors == []
    assert candidate["evidence_quotes"] == [
        "甲公司通过中间货物向乙公司提供设备。",
        "丙公司通过中间服务向丁公司提供支持。",
    ]
    supported, _ = service._evidence_supported(candidate, {"chunk_1": sample})
    assert supported is True
    assert candidate["confidence"] == 0.96


def test_schema_policy_verifier_rejects_ambiguous_binary_roles():
    class PolicyOllama:
        def json_chat(self, **kwargs):
            assert kwargs["schema"] is SCHEMA_POLICY_VERIFICATION_FORMAT
            return {
                "decision": "REJECTED",
                "reasoning": "宾语角色同时表示接收者和被转移物。",
            }

    candidate = {
        "name": "alpha_rel",
        "evidence_chunks": ["chunk_1"],
        "object_role_description": "接收者或被转移的物品。",
    }
    sample = _chunk(
        "chunk_1", "section", "甲方向乙方交付设备。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    review = {
        "entity_type_candidates": [],
        "relation_type_candidates": [candidate],
    }

    verified, errors = SchemaEvolutionService(
        None, PolicyOllama(),
    )._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192}, review, [sample],
    )

    assert errors == []
    assert verified["relation_type_candidates"][0]["_policy_verification"] == {
        "verdict": "REJECTED",
        "reason": "宾语角色同时表示接收者和被转移物。",
    }


def test_schema_policy_verifier_fails_closed_on_unknown_transport_shape():
    class PolicyOllama:
        def json_chat(self, **kwargs):
            return {"result": "looks fine"}

    review = {
        "entity_type_candidates": [],
        "relation_type_candidates": [{"name": "alpha_rel"}],
    }
    verified, errors = SchemaEvolutionService(
        None, PolicyOllama(),
    )._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192}, review, [],
    )

    assert errors == []
    assert verified["relation_type_candidates"][0]["_policy_verification"]["verdict"] == "UNCERTAIN"


def _complete_relation_candidate(name: str, semantic_definition: str) -> dict:
    return {
        "name": name,
        "description": semantic_definition,
        "source_types": ["Organization"],
        "target_types": ["Product"],
        "semantic_definition": semantic_definition,
        "domain_types": ["Organization"],
        "range_types": ["Product"],
        "direction_description": "从执行动作的组织指向承受动作的产品。",
        "subject_role_description": "执行动作的组织。",
        "object_role_description": "承受动作的产品。",
        "symmetric": False,
        "inverse_predicate": None,
        "canonical_orientation": "forward",
        "publish_mode": "auto",
        "reflexive": False,
        "positive_examples": ["甲公司检查设备。"],
        "negative_examples": ["甲公司与设备仅在同一段落出现。"],
        "optional_soft_cues": [],
        "confidence": 0.98,
        "evidence_chunks": ["chunk_1"],
        "evidence_quotes": ["甲公司检查设备。"],
    }


def test_schema_accepts_only_resolved_non_self_source_binding_conflicts():
    sample = _chunk("chunk_1", "section", "甲公司检查设备。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    accepted_candidate = _complete_relation_candidate(
        "inspects", "组织对产品执行检查动作。",
    )
    accepted_candidate.update({
        "conflicts_with": ["supports", "supports"],
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "角色与证据一致。",
        },
    })
    unknown_candidate = _complete_relation_candidate(
        "audits", "组织对产品执行审计动作。",
    )
    unknown_candidate.update({
        "conflicts_with": ["missing_policy"],
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "角色与证据一致。",
        },
    })
    self_candidate = _complete_relation_candidate(
        "reviews", "组织对产品执行复核动作。",
    )
    self_candidate.update({
        "conflicts_with": ["reviews"],
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "角色与证据一致。",
        },
    })

    updated, _, accepted, rejected = SchemaEvolutionService(
        None, FakeOllama(),
    )._apply_candidates(
        BASE_SCHEMA,
        {
            "entity_type_candidates": [],
            "relation_type_candidates": [
                accepted_candidate, unknown_candidate, self_candidate,
            ],
        },
        [sample],
    )

    assert [item["name"] for item in accepted] == ["inspects"]
    assert accepted[0]["conflicts_with"] == ["supports"]
    assert next(
        item for item in updated["relation_types"]
        if item["name"] == "inspects"
    )["conflicts_with"] == ["supports"]
    reasons = {
        item["name"]: item["reasons"] for item in rejected["details"]
    }
    assert reasons["audits"] == ["unknown_conflicting_predicate"]
    assert reasons["reviews"] == ["self_conflicting_predicate"]


def test_schema_language_gate_accepts_english_project_presentation_text():
    sample = _chunk(
        "chunk_1", "section", "The organization inspects the product.",
        [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    candidate = _complete_relation_candidate(
        "inspects", "An organization inspects a product.",
    )
    candidate.update({
        "description": "An organization inspects a product.",
        "semantic_definition": "A directed inspection relation from an organization to a product.",
        "direction_description": "From the inspecting organization to the inspected product.",
        "subject_role_description": "The organization performing the inspection.",
        "object_role_description": "The product receiving the inspection.",
        "positive_examples": ["The organization inspects the product."],
        "negative_examples": ["The two entities merely co-occur."],
        "evidence_quotes": ["The organization inspects the product."],
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "Roles match the evidence.",
        },
    })
    service = SchemaEvolutionService(None, FakeOllama())

    _, _, accepted, rejected = service._apply_candidates(
        BASE_SCHEMA,
        {"entity_type_candidates": [], "relation_type_candidates": [candidate]},
        [sample],
        knowledge_language="en",
    )
    _, _, chinese_accepted, chinese_rejected = service._apply_candidates(
        BASE_SCHEMA,
        {"entity_type_candidates": [], "relation_type_candidates": [candidate]},
        [sample],
        knowledge_language="zh",
    )

    assert [item["name"] for item in accepted] == ["inspects"]
    assert rejected["relation_types"] == 0
    assert chinese_accepted == []
    assert "invalid_user_facing_language" in chinese_rejected["details"][0]["reasons"]


def test_schema_persists_explicit_negative_semantics_and_possessive_role_contract():
    sample = _chunk("chunk_1", "section", "甲公司的设备没有故障。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    candidate = _complete_relation_candidate(
        "lacks_fault", "组织拥有的产品处于明确无故障状态。",
    )
    candidate.update({
        "positive_examples": ["甲公司的设备没有故障。"],
        "evidence_quotes": ["甲公司的设备没有故障。"],
        "source_role_pairs": ["possessor->possessed"],
        "negation_policy": "allow_explicit_negative_relation",
        "modality_policy": "require_asserted",
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "否定状态、角色和证据一致。",
        },
    })

    updated, _, accepted, rejected = SchemaEvolutionService(
        None, FakeOllama(),
    )._apply_candidates(
        BASE_SCHEMA,
        {"entity_type_candidates": [], "relation_type_candidates": [candidate]},
        [sample],
    )

    assert rejected["relation_types"] == 0
    assert accepted[0]["source_role_pairs"] == ["possessor->possessed"]
    assert accepted[0]["negation_policy"] == "allow_explicit_negative_relation"
    stored = next(
        item for item in updated["relation_types"]
        if item["name"] == "lacks_fault"
    )
    assert stored["source_role_pairs"] == ["possessor->possessed"]
    assert stored["negation_policy"] == "allow_explicit_negative_relation"
    assert stored["modality_policy"] == "require_asserted"


def test_schema_rejects_unknown_polarity_and_modality_contracts():
    sample = _chunk("chunk_1", "section", "甲公司检查设备。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    candidate = _complete_relation_candidate(
        "invalid_contract", "组织对产品执行检查动作。",
    )
    candidate.update({
        "negation_policy": "infer_from_name",
        "modality_policy": "allow_hypothetical",
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "角色与证据一致。",
        },
    })

    _, _, accepted, rejected = SchemaEvolutionService(
        None, FakeOllama(),
    )._apply_candidates(
        BASE_SCHEMA,
        {"entity_type_candidates": [], "relation_type_candidates": [candidate]},
        [sample],
    )

    assert accepted == []
    assert rejected["details"][0]["reasons"] == [
        "invalid_negation_policy", "invalid_modality_policy",
    ]


def test_schema_removes_conflict_link_when_same_review_peer_is_rejected():
    sample = _chunk("chunk_1", "section", "甲公司检查设备。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    first = _complete_relation_candidate(
        "first_interpretation", "组织对产品执行检查动作。",
    )
    first.update({
        "conflicts_with": ["second_interpretation"],
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "角色与证据一致。",
        },
    })
    second = _complete_relation_candidate(
        "second_interpretation", "组织对产品执行观察动作。",
    )
    second.update({
        "conflicts_with": ["first_interpretation"],
        "_policy_verification": {
            "verdict": "REJECTED", "reason": "角色与证据不一致。",
        },
    })

    updated, _, accepted, rejected = SchemaEvolutionService(
        None, FakeOllama(),
    )._apply_candidates(
        BASE_SCHEMA,
        {
            "entity_type_candidates": [],
            "relation_type_candidates": [first, second],
        },
        [sample],
    )

    assert accepted == []
    assert not {
        "first_interpretation", "second_interpretation",
    } & {item["name"] for item in updated["relation_types"]}
    reasons = {
        item["name"]: item["reasons"] for item in rejected["details"]
    }
    assert reasons["second_interpretation"] == [
        "policy_verification_rejected",
    ]
    assert reasons["first_interpretation"] == [
        "dangling_conflicting_predicate",
    ]


def test_schema_novelty_gate_rejects_renamed_existing_semantics():
    class NoveltyOllama:
        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_POLICY_VERIFICATION_FORMAT:
                return {"verdict": "SUPPORTED", "reason": "角色与证据一致。"}
            assert kwargs["schema"] is SCHEMA_POLICY_NOVELTY_FORMAT
            return {
                "verdict": "REDUNDANT",
                "reason": "候选只是现有关系的改名，端点角色和核心契约相同。",
                "overlapping_relation": "existing_rel",
                "exact_role_alignment": True,
                "core_contract_equivalent": True,
                "orientation_alignment": "same",
            }

    sample = _chunk("chunk_1", "section", "甲公司检查设备。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    candidate = _complete_relation_candidate(
        "alpha_rel", "组织对产品执行检查动作。",
    )
    current = {
        **BASE_SCHEMA,
        "relation_types": [{
            **candidate,
            "name": "existing_rel",
        }],
    }
    review = {"entity_type_candidates": [], "relation_type_candidates": [candidate]}
    service = SchemaEvolutionService(None, NoveltyOllama())

    verified, errors = service._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192},
        review, [sample], current,
    )
    _, _, accepted, rejected = service._apply_candidates(
        BASE_SCHEMA, verified, [sample],
    )

    assert errors == []
    assert accepted == []
    assert rejected["details"][0]["reasons"] == ["semantic_redundancy"]
    assert rejected["details"][0]["policy_novelty"]["overlapping_relation"] == "existing_rel"


def test_schema_novelty_gate_accepts_arbitrary_distinct_predicate():
    class NoveltyOllama:
        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_POLICY_VERIFICATION_FORMAT:
                return {"verdict": "SUPPORTED", "reason": "角色与证据一致。"}
            assert kwargs["schema"] is SCHEMA_POLICY_NOVELTY_FORMAT
            return {
                "verdict": "NOVEL",
                "reason": "现有关系不能表达该动作的稳定端点角色。",
                "overlapping_relation": None,
                "exact_role_alignment": False,
                "core_contract_equivalent": False,
                "orientation_alignment": "none",
            }

    sample = _chunk("chunk_1", "section", "甲公司检查设备。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    candidate = _complete_relation_candidate(
        "beta_rel", "组织对产品执行检查动作。",
    )
    review = {"entity_type_candidates": [], "relation_type_candidates": [candidate]}
    service = SchemaEvolutionService(None, NoveltyOllama())

    verified, errors = service._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192},
        review, [sample], BASE_SCHEMA,
    )
    _, _, accepted, rejected = service._apply_candidates(
        BASE_SCHEMA, verified, [sample],
    )

    assert errors == []
    assert [item["name"] for item in accepted] == ["beta_rel"]
    assert rejected["relation_types"] == 0


def test_grounded_relation_batch_gates_accept_compact_transport_shapes():
    class CompactBatchOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            self.calls.append(kwargs["system"])
            if "批量独立核验器" in kwargs["system"]:
                assert kwargs["schema"]["properties"]["decisions"][
                    "items"
                ]["type"] == "string"
                return {
                    "candidate_0_decision": "ACCEPTED",
                    "candidate_0_reason": "第一项角色与证据一致。",
                    "candidate_1_decision": "PASS",
                    "candidate_1_reason": "第二项角色与证据一致。",
                }
            if "批量独立语义去重核验器" in kwargs["system"]:
                assert kwargs["schema"]["properties"]["decisions"][
                    "items"
                ]["type"] == "string"
                # Ollama may preserve a root-array transport as ``items``.
                return {"items": ["NOVEL", "NOVEL"]}
            raise AssertionError("compact batch result must avoid single-item calls")

    sample = _chunk(
        "chunk_1", "section", "甲公司检查设备。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    candidates = []
    for name in ("alpha_rel", "beta_rel"):
        candidate = _complete_relation_candidate(
            name, "组织对产品执行明确检查动作。",
        )
        candidate["_induction_observations"] = [{
            "chunk_key": "chunk_1",
            "subject_surface": "甲公司",
            "relation_expression": "检查",
            "object_surface": "设备",
            "evidence_quote": "甲公司检查设备。",
            "assertion_status": "AFFIRMED",
        }]
        candidates.append(candidate)
    gateway = CompactBatchOllama()

    verified, errors = SchemaEvolutionService(
        None, gateway,
    )._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192},
        {
            "entity_type_candidates": [],
            "relation_type_candidates": candidates,
        },
        [sample], BASE_SCHEMA,
    )

    assert errors == []
    assert len(gateway.calls) == 2
    assert all(
        item["_policy_verification"]["verdict"] == "SUPPORTED"
        and item["_policy_verification"]["verification_mode"]
        == "batch_flattened"
        and item["_policy_novelty"]["verdict"] == "NOVEL"
        and item["_policy_novelty"]["verification_mode"]
        == "batch_positional"
        for item in verified["relation_type_candidates"]
    )


def test_grounded_relation_partial_batch_routes_only_failures_to_targeted_recheck():
    class PartialBatchOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            self.calls.append(kwargs["system"])
            if "批量独立核验器" in kwargs["system"]:
                batch_calls = sum(
                    "批量独立核验器" in value for value in self.calls
                )
                if batch_calls == 1:
                    # A one-item response is occasionally unwrapped by local
                    # constrained-output models even though a batch was requested.
                    return {
                        "candidate_index": 0,
                        "decision": "REJECTED",
                        "reason": "第一项固定端点角色与证据不一致。",
                    }
                return {"result": "incomplete batch transport"}
            if kwargs["schema"] is SCHEMA_POLICY_VERIFICATION_FORMAT:
                candidate = json.loads(kwargs["prompt"])["candidate"]
                if candidate["name"] == "alpha_rel":
                    return {
                        "verdict": "REJECTED",
                        "reason": "单项复核确认第一项角色不一致。",
                    }
                # A one-item response is occasionally unwrapped by local
                return {
                    "verdict": "SUPPORTED",
                    "reason": "单项复核确认第二项角色与证据一致。",
                }
            if kwargs["schema"] is SCHEMA_POLICY_REPAIR_FORMAT:
                return {
                    "repairable": False,
                    "reason": "冻结证据不能修复第一项角色。",
                }
            assert kwargs["schema"] is SCHEMA_POLICY_NOVELTY_FORMAT
            return {
                "verdict": "NOVEL",
                "reason": "当前 Schema 没有相同角色契约。",
                "overlapping_relation": None,
                "exact_role_alignment": False,
                "core_contract_equivalent": False,
                "orientation_alignment": "none",
            }

    sample = _chunk(
        "chunk_1", "section", "甲公司检查设备。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    candidates = []
    for name in ("alpha_rel", "beta_rel"):
        candidate = _complete_relation_candidate(
            name, "组织对产品执行明确检查动作。",
        )
        candidate["_induction_observations"] = [{
            "chunk_key": "chunk_1",
            "subject_surface": "甲公司",
            "relation_expression": "检查",
            "object_surface": "设备",
            "evidence_quote": "甲公司检查设备。",
            "assertion_status": "AFFIRMED",
        }]
        candidates.append(candidate)

    gateway = PartialBatchOllama()
    verified, errors = SchemaEvolutionService(
        None, gateway,
    )._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192},
        {
            "entity_type_candidates": [],
            "relation_type_candidates": candidates,
        },
        [sample], BASE_SCHEMA,
    )

    policies = [
        item["_policy_verification"]
        for item in verified["relation_type_candidates"]
    ]
    assert errors == []
    assert policies[0]["verdict"] == "REJECTED"
    assert policies[0]["reason"] == "单项复核确认第一项角色不一致。"
    assert policies[1]["verdict"] == "SUPPORTED"
    assert policies[1]["reason"] == "单项复核确认第二项角色与证据一致。"
    assert all(
        item["_policy_batch_verification"]["verification_mode"].startswith(
            "batch"
        )
        for item in verified["relation_type_candidates"]
    )
    assert verified["relation_type_candidates"][0]["_policy_repair"][
        "repairable"
    ] is False
    assert verified["relation_type_candidates"][1]["_policy_novelty"][
        "verdict"
    ] == "NOVEL"


def test_schema_novelty_protocol_error_retries_with_larger_budget():
    class RetryingNoveltyOllama:
        def __init__(self):
            self.novelty_budgets = []

        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_POLICY_VERIFICATION_FORMAT:
                return {"verdict": "SUPPORTED", "reason": "角色与证据一致。"}
            assert kwargs["schema"] is SCHEMA_POLICY_NOVELTY_FORMAT
            self.novelty_budgets.append(kwargs["max_output_tokens"])
            if len(self.novelty_budgets) == 1:
                raise OllamaProtocolError("truncated fixture")
            return {
                "verdict": "NOVEL",
                "reason": "现有关系不能表达该动作的稳定端点角色。",
                "overlapping_relation": None,
                "exact_role_alignment": False,
                "core_contract_equivalent": False,
                "orientation_alignment": "none",
            }

    sample = _chunk("chunk_1", "section", "甲公司检查设备。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    candidate = _complete_relation_candidate(
        "gamma_rel", "组织对产品执行检查动作。",
    )
    gateway = RetryingNoveltyOllama()
    service = SchemaEvolutionService(None, gateway)

    verified, errors = service._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"entity_type_candidates": [], "relation_type_candidates": [candidate]},
        [sample],
        BASE_SCHEMA,
    )
    _, _, accepted, rejected = service._apply_candidates(
        BASE_SCHEMA, verified, [sample],
    )

    assert errors == []
    assert gateway.novelty_budgets == [1024, 2048]
    assert [item["name"] for item in accepted] == ["gamma_rel"]
    assert rejected["relation_types"] == 0


def test_schema_novelty_gate_deduplicates_candidates_within_same_review():
    class SameReviewOllama:
        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_POLICY_VERIFICATION_FORMAT:
                return {"verdict": "SUPPORTED", "reason": "角色与证据一致。"}
            assert kwargs["schema"] is SCHEMA_POLICY_NOVELTY_FORMAT
            request = json.loads(kwargs["prompt"])
            current_names = {
                item["name"] for item in request["current_relation_policies"]
            }
            if "alpha_rel" not in current_names:
                return {
                    "verdict": "NOVEL",
                    "reason": "现有关系不能表达该动作契约。",
                    "overlapping_relation": None,
                    "exact_role_alignment": False,
                    "core_contract_equivalent": False,
                    "orientation_alignment": "none",
                }
            return {
                "verdict": "REDUNDANT",
                "reason": "本轮较早候选已有相同端点角色与核心契约。",
                "overlapping_relation": "alpha_rel",
                "exact_role_alignment": True,
                "core_contract_equivalent": True,
                "orientation_alignment": "same",
            }

    sample = _chunk("chunk_1", "section", "甲公司检查设备。", [1.0, 0.0])
    sample["_schema_text"] = sample["content"]
    review = {
        "entity_type_candidates": [],
        "relation_type_candidates": [
            _complete_relation_candidate(
                "beta_rel", "组织对产品执行检查动作。",
            ),
            _complete_relation_candidate(
                "alpha_rel", "组织对产品执行检查动作。",
            ),
        ],
    }
    service = SchemaEvolutionService(None, SameReviewOllama())

    verified, errors = service._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192},
        review, [sample], BASE_SCHEMA,
    )
    _, _, accepted, rejected = service._apply_candidates(
        BASE_SCHEMA, verified, [sample],
    )

    assert errors == []
    assert [item["name"] for item in accepted] == ["alpha_rel"]
    assert rejected["details"][0]["name"] == "beta_rel"
    assert rejected["details"][0]["reasons"] == ["semantic_redundancy"]


def test_schema_policy_repair_compiles_one_binary_projection_then_reverifies():
    class RepairingPolicyOllama:
        def __init__(self):
            self.verification_calls = 0
            self.repair_calls = 0

        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_POLICY_REPAIR_FORMAT:
                self.repair_calls += 1
                original = json.loads(kwargs["prompt"])["candidate"]
                repaired = dict(original)
                repaired.update({
                    "description": "主体向客体交付一种物品。",
                    "semantic_definition": "主体向客体交付一种物品。",
                    "source_types": ["Organization"],
                    "target_types": ["Product"],
                    "domain_types": ["Organization"],
                    "range_types": ["Product"],
                    "direction_description": "从交付物品的主体指向被交付的物品。",
                    "subject_role_description": "执行交付动作的主体。",
                    "object_role_description": "被交付的物品。",
                    "positive_examples": ["甲方向乙方交付设备。"],
                    "negative_examples": ["甲方仅与设备同时出现。"],
                })
                return {
                    "repairable": True,
                    "reason": "可从三元事件中固定主体与物品这一组端点。",
                    "candidate": repaired,
                }
            assert kwargs["schema"] is SCHEMA_POLICY_VERIFICATION_FORMAT
            self.verification_calls += 1
            candidate = json.loads(kwargs["prompt"])["candidate"]
            if "或" in candidate["object_role_description"]:
                return {"verdict": "REJECTED", "reason": "宾语角色不唯一。"}
            return {"verdict": "SUPPORTED", "reason": "二元角色与方向明确。"}

    candidate = {
        "name": "alpha_rel",
        "description": "主体向接收者或物品交付资源。",
        "source_types": ["Organization"],
        "target_types": ["Organization", "Product"],
        "semantic_definition": "主体向接收者或物品交付资源。",
        "domain_types": ["Organization"],
        "range_types": ["Organization", "Product"],
        "direction_description": "从主体指向接收者或物品。",
        "subject_role_description": "执行交付的主体。",
        "object_role_description": "接收者或被交付的物品。",
        "symmetric": False,
        "inverse_predicate": None,
        "canonical_orientation": "forward",
        "publish_mode": "auto",
        "reflexive": False,
        "positive_examples": ["甲方向乙方交付设备。"],
        "negative_examples": [],
        "optional_soft_cues": [],
        "evidence_chunks": ["chunk_1"],
        "evidence_quotes": ["甲方向乙方交付设备。"],
    }
    sample = _chunk(
        "chunk_1", "section", "甲方向乙方交付设备。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    review = {
        "entity_type_candidates": [],
        "relation_type_candidates": [candidate],
    }
    ollama = RepairingPolicyOllama()

    verified, errors = SchemaEvolutionService(
        None, ollama,
    )._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192}, review, [sample],
    )

    repaired = verified["relation_type_candidates"][0]
    assert errors == []
    assert ollama.repair_calls == 1
    assert ollama.verification_calls == 2
    assert repaired["name"] == "alpha_rel"
    assert repaired["evidence_chunks"] == ["chunk_1"]
    assert repaired["evidence_quotes"] == ["甲方向乙方交付设备。"]
    assert repaired["confidence"] == Config.schema_review_single_evidence_confidence
    assert repaired["target_types"] == ["Product"]
    assert repaired["object_role_description"] == "被交付的物品。"
    assert repaired["_policy_verification"]["verdict"] == "SUPPORTED"
    assert repaired["_policy_repair"]["reverification"]["verdict"] == "SUPPORTED"


def test_schema_policy_compiles_incomplete_transport_before_activation():
    class CompilingOllama:
        def __init__(self):
            self.repair_calls = 0
            self.verification_calls = 0

        def json_chat(self, **kwargs):
            if kwargs["schema"] is SCHEMA_POLICY_REPAIR_FORMAT:
                self.repair_calls += 1
                original = json.loads(kwargs["prompt"])["candidate"]
                return {
                    "repairable": True,
                    "reason": "可由冻结证据补齐结构字段。",
                    "candidate": {
                        **original,
                        "description": "主体向客体提供一种资源。",
                        "source_types": ["Organization"],
                        "target_types": ["Product"],
                        "semantic_definition": "主体向客体提供一种资源。",
                        "domain_types": ["Organization"],
                        "range_types": ["Product"],
                        "direction_description": "从资源提供者指向被提供的资源。",
                        "subject_role_description": "提供资源的主体。",
                        "object_role_description": "被提供的资源。",
                        "symmetric": False,
                        "inverse_predicate": None,
                        "canonical_orientation": "forward",
                        "publish_mode": "auto",
                        "reflexive": False,
                        "positive_examples": ["甲方向乙方提供设备。"],
                        "negative_examples": ["甲方与设备只在同一段出现。"],
                        "optional_soft_cues": [],
                    },
                }
            self.verification_calls += 1
            return {"verdict": "SUPPORTED", "reason": "语义与证据一致。"}

    candidate = {
        "name": "alpha_rel",
        "description": "主体向客体提供一种资源。",
        "semantic_definition": "主体向客体提供一种资源。",
        "domain_types": ["Organization"],
        "range_types": ["Product"],
        "confidence": 0.98,
        "evidence_chunks": ["chunk_1"],
        "evidence_quotes": ["甲方向乙方提供设备。"],
    }
    sample = _chunk(
        "chunk_1", "section", "甲方向乙方提供设备。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    ollama = CompilingOllama()

    verified, errors = SchemaEvolutionService(
        None, ollama,
    )._verify_relation_policies(
        {"llm_model": "test", "model_max_tokens": 8192},
        {
            "entity_type_candidates": [],
            "relation_type_candidates": [candidate],
        },
        [sample],
    )

    compiled = verified["relation_type_candidates"][0]
    assert errors == []
    assert ollama.repair_calls == 1
    assert ollama.verification_calls == 2
    assert compiled["direction_description"] == "从资源提供者指向被提供的资源。"
    assert compiled["symmetric"] is False
    assert compiled["_policy_verification"]["verdict"] == "SUPPORTED"
    assert any(
        value.startswith("missing:")
        for value in compiled["_policy_repair"]["input_structure_issues"]
    )


def test_schema_consolidation_does_not_silently_erase_valid_candidates():
    class EmptyConsolidationOllama:
        def json_chat(self, **kwargs):
            return {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [],
            }

    candidate = {
        "name": "alpha_rel",
        "description": "主体向客体提供资源。",
        "semantic_definition": "主体向客体提供资源。",
        "domain_types": ["Organization"],
        "range_types": ["Product"],
        "evidence_chunks": ["chunk_1"],
    }
    result = SchemaEvolutionService(
        FakeDatabase(), EmptyConsolidationOllama(),
    )._consolidate_candidates(
        {"llm_model": "test"}, BASE_SCHEMA,
        [{
            "summary": "", "entity_type_candidates": [],
            "relation_type_candidates": [candidate],
        }],
    )
    assert result["relation_type_candidates"][0]["name"] == "alpha_rel"
    assert result["relation_type_candidates"][0]["source_types"] == ["Organization"]


def test_schema_consolidation_skips_generative_pass_for_grounded_open_relations():
    class ForbiddenConsolidationOllama:
        def json_chat(self, **kwargs):
            raise AssertionError(
                "grounded open-relation candidates must not be generatively "
                "re-consolidated"
            )

    relation_candidates = []
    for name, subject_role, object_role in (
        ("submits", "提交内容的主体", "被提交的内容"),
        ("submitted_to", "被提交的内容", "接收内容的主体"),
    ):
        relation_candidates.append({
            "name": name,
            "description": f"{subject_role}指向{object_role}。",
            "semantic_definition": f"{subject_role}与{object_role}之间的有向关系。",
            "domain_types": ["Entity"],
            "range_types": ["Entity"],
            "evidence_chunks": ["chunk_1"],
            "evidence_quotes": ["甲方向乙方提交丙材料。"],
            "_induction_observations": [{
                "subject": "甲方" if name == "submits" else "丙材料",
                "relation_expression": name,
                "object": "丙材料" if name == "submits" else "乙方",
                "chunk_key": "chunk_1",
                "evidence_quote": "甲方向乙方提交丙材料。",
            }],
        })

    result = SchemaEvolutionService(
        FakeDatabase(), ForbiddenConsolidationOllama(),
    )._consolidate_candidates(
        {"llm_model": "test"}, BASE_SCHEMA,
        [{
            "summary": "",
            "entity_type_candidates": [],
            "relation_type_candidates": relation_candidates,
        }],
    )

    assert [
        item["name"] for item in result["relation_type_candidates"]
    ] == ["submits", "submitted_to"]
    assert all(
        item.get("_induction_observations")
        for item in result["relation_type_candidates"]
    )


def test_schema_consolidation_is_deterministic_for_mixed_candidate_sources():
    class ForbiddenConsolidationOllama:
        def json_chat(self, **kwargs):
            raise AssertionError("candidate consolidation must not call a model")

    result = SchemaEvolutionService(
        FakeDatabase(), ForbiddenConsolidationOllama(),
    )._consolidate_candidates(
        {"llm_model": "test"}, BASE_SCHEMA,
        [
            {
                "summary": "",
                "entity_type_candidates": [{
                    "name": "NewEntityType",
                    "description": "一种新实体。",
                    "confidence": 0.9,
                    "evidence_chunks": ["chunk_1"],
                }],
                "relation_type_candidates": [],
            },
            {
                "summary": "",
                "entity_type_candidates": [],
                "relation_type_candidates": [{
                    "name": "alpha_rel",
                    "evidence_chunks": ["chunk_1"],
                    "_induction_observations": [{"chunk_key": "chunk_1"}],
                }],
            },
        ],
    )

    assert [
        item["name"] for item in result["entity_type_candidates"]
    ] == ["NewEntityType"]
    assert [
        item["name"] for item in result["relation_type_candidates"]
    ] == ["alpha_rel"]


def test_schema_consolidation_restores_partially_omitted_exact_identifiers():
    class PartialConsolidationOllama:
        def json_chat(self, **kwargs):
            first = json.loads(kwargs["prompt"].split(
                "Candidate batches:\n", 1,
            )[1])[0]["relation_type_candidates"][0]
            return {
                "summary": "只返回了一个候选",
                "entity_type_candidates": [],
                "relation_type_candidates": [first],
            }

    candidates = [{
        "summary": "",
        "entity_type_candidates": [],
        "relation_type_candidates": [
            {
                "name": name,
                "description": "主体向客体提供资源。",
                "semantic_definition": "主体向客体提供资源。",
                "domain_types": ["Organization"],
                "range_types": ["Product"],
                "evidence_chunks": ["chunk_1"],
            }
            for name in ("alpha_rel", "beta_rel")
        ],
    }]

    result = SchemaEvolutionService(
        FakeDatabase(), PartialConsolidationOllama(),
    )._consolidate_candidates(
        {"llm_model": "test"}, BASE_SCHEMA, candidates,
    )

    assert {
        item["name"] for item in result["relation_type_candidates"]
    } == {"alpha_rel", "beta_rel"}


def test_schema_candidate_audit_replay_requires_same_document_and_exact_quote():
    class ReplayDatabase(FakeDatabase):
        def list_schema_evaluations(self, project_id, *, document_id=None):
            assert project_id == 3
            assert document_id == 7
            return [{
                "evaluation": {
                    "schema_discovery_protocol_version": (
                        SCHEMA_DISCOVERY_PROTOCOL_VERSION
                    ),
                    "proposed_relation_types": [
                        {
                            "name": "alpha_rel",
                            "description": "主体向客体交付设备。",
                            "evidence_chunks": ["chunk_1"],
                            "evidence_quotes": ["甲方向乙方交付设备。"],
                            "_policy_verification": {"verdict": "REJECTED"},
                        },
                        {
                            "name": "beta_rel",
                            "description": "没有当前证据。",
                            "evidence_chunks": ["chunk_1"],
                            "evidence_quotes": ["原文中不存在的句子。"],
                        },
                    ],
                },
            }, {
                # Exact text alone must not replay a policy drafted by an
                # obsolete discovery/role protocol.
                "evaluation": {
                    "proposed_relation_types": [{
                        "name": "stale_rel",
                        "description": "旧协议候选。",
                        "evidence_chunks": ["chunk_1"],
                        "evidence_quotes": ["甲方向乙方交付设备。"],
                    }],
                },
            }]

    sample = _chunk(
        "chunk_1", "section", "甲方向乙方交付设备。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]

    batch = SchemaEvolutionService(
        ReplayDatabase(), FakeOllama(),
    )._historical_relation_candidate_batch(3, 7, [sample])

    assert batch is not None
    assert [item["name"] for item in batch["relation_type_candidates"]] == [
        "alpha_rel",
    ]
    assert "_policy_verification" not in batch["relation_type_candidates"][0]


def test_two_exact_examples_support_one_chunk_schema_evidence():
    sample = _chunk(
        "chunk_1", "section",
        "甲公司向乙公司提供设备。丙公司向丁公司提供材料。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    candidate = {
        "evidence_chunks": ["chunk_1"],
        "positive_examples": ["甲公司向乙公司提供设备。", "丙公司向丁公司提供材料。"],
    }
    supported, evidence = SchemaEvolutionService._evidence_supported(
        candidate, {"chunk_1": sample},
    )
    assert supported is True
    assert evidence == ["chunk_1"]
    assert candidate["confidence"] == 0.96


def test_exact_source_examples_can_supply_missing_transport_confidence():
    sample = _chunk(
        "chunk_1", "section",
        "甲公司向乙公司提供设备。丙公司向丁公司提供材料。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    candidate = {
        "name": "alpha_rel",
        "description": "通用关系。",
        "source_types": ["Organization"],
        "target_types": ["Organization"],
        "semantic_definition": "主体向客体提供物品。",
        "domain_types": ["Organization"],
        "range_types": ["Organization"],
        "direction_description": "从提供者指向接收者。",
        "subject_role_description": "提供物品的组织。",
        "object_role_description": "接收物品的组织。",
        "symmetric": False,
        "inverse_predicate": None,
        "canonical_orientation": "forward",
        "publish_mode": "auto",
        "reflexive": False,
        "positive_examples": [
            "甲公司向乙公司提供设备。", "丙公司向丁公司提供材料。",
        ],
        "negative_examples": ["甲公司和乙公司同时出现。"],
        "optional_soft_cues": [],
        "evidence_chunks": ["chunk_1"],
        "_policy_verification": {
            "verdict": "SUPPORTED", "reason": "角色与方向一致。",
        },
    }
    service = SchemaEvolutionService(None, FakeOllama())

    _, _, accepted, rejected = service._apply_candidates(
        BASE_SCHEMA,
        {"entity_type_candidates": [], "relation_type_candidates": [candidate]},
        [sample],
    )

    assert [item["name"] for item in accepted] == ["alpha_rel"]
    assert accepted[0]["confidence"] == 0.96
    assert rejected["relation_types"] == 0


def test_single_chunk_model_confidence_without_exact_quote_is_not_schema_evidence():
    sample = _chunk(
        "chunk_1", "section", "甲公司向乙工厂供应控制器。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    candidate = {
        "confidence": 0.99,
        "evidence_chunks": ["chunk_1"],
        "evidence_quotes": ["A supplies B."],
        "positive_examples": ["A supplies B."],
    }

    supported, evidence = SchemaEvolutionService._evidence_supported(
        candidate, {"chunk_1": sample},
    )

    assert supported is False
    assert evidence == ["chunk_1"]


def test_exact_evidence_quote_is_independent_of_generic_policy_examples():
    sample = _chunk(
        "chunk_1", "section", "甲公司向乙工厂供应控制器。", [1.0, 0.0],
    )
    sample["_schema_text"] = sample["content"]
    candidate = {
        "confidence": 0.97,
        "evidence_chunks": ["chunk_1"],
        "evidence_quotes": ["甲公司向乙工厂供应控制器。"],
        "positive_examples": ["供应方向接收方提供物品。"],
    }

    supported, evidence = SchemaEvolutionService._evidence_supported(
        candidate, {"chunk_1": sample},
    )

    assert supported is True
    assert evidence == ["chunk_1"]


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


def test_entity_only_schema_requires_offsets_and_runs_isolated_proposition_pass():
    entity_schema = ENTITY_ONLY_EXTRACTION_FORMAT["properties"]["entities"]["items"]
    assert {"mention_start", "mention_end"} <= set(entity_schema["required"])
    assert set(entity_schema["required"]) == {
        "id", "name", "type", "mention_start", "mention_end", "confidence",
        "argument_complete", "standalone_referential",
        "complete_mention_start", "complete_mention_end",
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

    assert len(ollama.calls) == 2
    assert ollama.calls[0]["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT
    assert ollama.calls[1]["schema"] is PROPOSITION_ARGUMENT_EXTRACTION_FORMAT
    assert ollama.calls[0]["max_output_tokens"] == Config.entity_only_max_output_tokens
    assert "Return an empty entities array only" in ollama.calls[0]["system"]
    assert "standalone_referential" in ollama.calls[0]["system"]
    assert "complete explicit proposition arguments" in ollama.calls[1]["system"]
    assert result["relations"] == []
    assert result["_diagnostics"]["llm_calls"] == 2
    assert result["_diagnostics"]["proposition_gleaning_calls"] == 1
    assert result["_diagnostics"]["verification_llm_calls"] == 0


def test_grounded_mention_inventory_uses_bounded_calls_and_source_boundaries():
    source = (
        "The gateway connects to a client. "
        "The `if` statement checks ready and calls start()."
    )

    class GroundedInventoryOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"items": [
                {
                    "surface": "The gateway",
                    "evidence": "The gateway connects to a client.",
                    "kind": "ENTITY",
                },
                {
                    "surface": "a client",
                    "evidence": "The gateway connects to a client.",
                    "kind": "ENTITY",
                },
                {
                    # Formatting marks are source syntax, not an alias or a
                    # fuzzy match. The deterministic grounder restores them.
                    "surface": "if statement",
                    "evidence": "The `if` statement checks ready and calls start().",
                    "kind": "ENTITY",
                },
                {
                    "surface": "ready",
                    "evidence": "The `if` statement checks ready and calls start().",
                    "kind": "EVENT_OR_STATE",
                },
                {
                    # Complete an immediately adjacent balanced empty call.
                    "surface": "start(",
                    "evidence": "The `if` statement checks ready and calls start().",
                    "kind": "ENTITY",
                },
            ]}

    ollama = GroundedInventoryOllama()
    result = SchemaEvolutionService(None, ollama).extract_grounded_mentions(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert len(ollama.calls) == 2
    assert ollama.calls[0]["schema"] is GROUNDED_MENTION_INVENTORY_FORMAT
    assert "omission-recovery pass" in ollama.calls[1]["system"]
    assert [item["name"] for item in result["entities"]] == [
        "gateway", "client", "`if` statement", "ready", "start()",
    ]
    assert all(
        source[item["mention_start"]:item["mention_end"]] == item["name"]
        for item in result["entities"]
    )
    assert all(item["endpoint_candidate"] for item in result["entities"])
    assert result["_diagnostics"]["grounded_mention_inventory"] is True
    assert result["_diagnostics"]["llm_calls"] == 2
    assert result["_diagnostics"]["primary_llm_calls"] == 1
    assert result["_diagnostics"]["grounded_gleaning_calls"] == 1
    assert result["_diagnostics"]["legacy_proposition_calls_avoided"] == 1


def test_grounded_mention_inventory_recovers_exact_repeated_occurrences():
    source = (
        "The controller checks an intent. "
        "The controller then approves the intent."
    )

    class RepresentativeOnlyOllama:
        def json_chat(self, **_kwargs):
            return {"items": [
                {
                    "surface": "controller",
                    "evidence": "The controller checks an intent.",
                    "kind": "ENTITY",
                },
                {
                    "surface": "intent",
                    "evidence": "The controller checks an intent.",
                    "kind": "ENTITY",
                },
            ]}

    result = SchemaEvolutionService(
        None, RepresentativeOnlyOllama(),
    ).extract_grounded_mentions(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert [item["name"] for item in result["entities"]] == [
        "controller", "intent", "controller", "intent",
    ]
    assert [item["mention_start"] for item in result["entities"]] == [
        source.index("controller"),
        source.index("intent"),
        source.rindex("controller"),
        source.rindex("intent"),
    ]
    assert result["_diagnostics"][
        "deterministic_occurrences_recovered"
    ] == 2


def test_entity_only_repairs_model_declared_complete_argument_span() -> None:
    source = "Gateway interruption causes data collection interruption."
    short_start = source.index("data collection")
    complete_end = source.index(".")

    class CompleteArgumentOllama:
        def json_chat(self, **_kwargs):
            return {"entities": [{
                "id": "event-1",
                "name": "data collection",
                "type": "Event",
                "mention_start": short_start,
                "mention_end": short_start + len("data collection"),
                "confidence": 0.9,
                "argument_complete": False,
                "complete_mention_start": short_start,
                "complete_mention_end": complete_end,
            }]}

    result = SchemaEvolutionService(None, CompleteArgumentOllama()).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert result["entities"][0]["name"] == "data collection interruption"
    assert result["entities"][0]["mention_start"] == short_start
    assert result["entities"][0]["mention_end"] == complete_end
    assert result["entities"][0]["argument_span_repaired"] is True


def test_entity_sanitizer_rejects_model_declared_non_referential_placeholder():
    service = SchemaEvolutionService(FakeDatabase(), FakeOllama())
    result = service._sanitize_extraction(
        {
            "entities": [{
                "id": "placeholder-1",
                "name": "该系统",
                "type": "Product",
                "mention_start": 0,
                "mention_end": 3,
                "standalone_referential": False,
            }],
            "relations": [],
        },
        "该系统发生故障。",
        BASE_SCHEMA,
    )

    assert result["entities"] == []
    assert result["_rejected_entities"][0]["reason"] == (
        "non_referential_placeholder"
    )


def test_entity_sanitizer_recovers_common_noun_from_over_conservative_hint():
    service = SchemaEvolutionService(FakeDatabase(), FakeOllama())
    source = "An HTTP client establishes a connection to an HTTP server."
    result = service._sanitize_extraction(
        {
            "entities": [
                {
                    "id": "client-1", "name": "client", "type": "GenericEntity",
                    "mention_start": source.index("client"),
                    "mention_end": source.index("client") + len("client"),
                    "standalone_referential": False,
                },
                {
                    "id": "placeholder-2", "name": "the following scene",
                    "type": "GenericEntity",
                    "mention_start": 0, "mention_end": 0,
                    "standalone_referential": False,
                },
            ],
            "relations": [],
        },
        source,
        BASE_SCHEMA,
    )
    assert [item["name"] for item in result["entities"]] == ["client"]
    assert result["entities"][0]["standalone_referential"] is True
    assert any(
        item["reason"] == "non_referential_placeholder"
        and item["placeholder_reason"] == "non_referential_context_placeholder"
        for item in result["_rejected_entities"]
    )


def test_entity_sanitizer_preserves_grounded_soft_wrap_provenance():
    source = (
        "主要包括任务管理、任务间通信、中断/异常\n"
        "管理、时钟管理和多核支持等功能。"
    )
    start = source.index("中断/异常")
    end = source.index("管理、时钟") + len("管理")
    service = SchemaEvolutionService(FakeDatabase(), FakeOllama())

    result = service._sanitize_extraction(
        {
            "entities": [{
                "id": "wrapped-1",
                "name": "中断/异常 管理",
                "type": "Technology",
                "mention_start": start,
                "mention_end": end,
                "endpoint_candidate": True,
                "grounded_semantic_mention": True,
            }],
            "relations": [],
        },
        source,
        BASE_SCHEMA,
    )

    assert result["_diagnostics"]["rejected_entities"] == 0
    assert result["entities"][0]["name"] == "中断/异常 管理"
    assert result["entities"][0]["endpoint_candidate"] is True
    assert result["entities"][0]["grounded_semantic_mention"] is True
    assert result["entities"][0]["soft_wrap_surface_repaired"] is True


def test_entity_sanitizer_aligns_repeated_surface_to_unique_nearest_exact_span():
    content = "甲基地包含车间。车间位于甲基地，仓库也位于甲基地。"
    spans = [
        (index, index + len("甲基地"))
        for index in range(len(content))
        if content.startswith("甲基地", index)
    ]
    intended = spans[1]
    service = SchemaEvolutionService(FakeDatabase(), FakeOllama())

    result = service._sanitize_extraction(
        {
            "entities": [{
                "id": "place-1",
                "name": "甲基地",
                "type": "Location",
                "mention_start": intended[0] - 1,
                "mention_end": intended[1] - 1,
                "confidence": 0.8,
            }],
            "relations": [],
        },
        content,
        BASE_SCHEMA,
    )

    assert result["_diagnostics"]["rejected_entities"] == 0
    assert result["entities"][0]["mention_start"] == intended[0]
    assert result["entities"][0]["mention_end"] == intended[1]


def test_entity_sanitizer_keeps_equally_near_repeated_surface_ambiguous():
    content = "甲基地与甲基地"
    service = SchemaEvolutionService(FakeDatabase(), FakeOllama())

    result = service._sanitize_extraction(
        {
            "entities": [{
                "id": "place-1",
                "name": "甲基地",
                "type": "Location",
                "mention_start": 2,
                "mention_end": 5,
                "confidence": 0.8,
            }],
            "relations": [],
        },
        content,
        BASE_SCHEMA,
    )

    assert result["entities"] == []
    assert result["_rejected_entities"][0]["reason"] == (
        "ambiguous_surface_without_exact_span"
    )


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
    original = Config.entity_only_max_sentences_per_segment
    Config.entity_only_max_sentences_per_segment = 100
    try:
        result = service.extract_entities_only(
            {"llm_model": "test", "model_max_tokens": 8192},
            {"schema_json": BASE_SCHEMA},
            {"content": source, "metadata": {}},
        )
    finally:
        Config.entity_only_max_sentences_per_segment = original

    assert len(ollama.calls) >= 3
    assert all(len(call) < len(source) for call in ollama.calls[1:])
    assert result["_diagnostics"]["segmented_fallback"] is True
    assert result["_diagnostics"]["llm_calls"] == 5
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

    assert len(ollama.calls) == 3
    assert ollama.calls[0]["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT
    assert ollama.calls[1]["schema"] is None
    assert ollama.calls[1]["temperature"] == 0
    assert ollama.calls[2]["schema"] is PROPOSITION_ARGUMENT_EXTRACTION_FORMAT
    assert result["entities"][0]["name"] == "Paris"
    assert result["_diagnostics"]["llm_calls"] == 3


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

    assert len(ollama.calls) == 5
    assert sum(
        call["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT for call in ollama.calls
    ) == 3
    assert sum(
        call["schema"] is PROPOSITION_ARGUMENT_EXTRACTION_FORMAT
        for call in ollama.calls
    ) == 2
    assert {item["name"] for item in result["entities"]} == {"Chile", "Argentina"}
    assert result["_diagnostics"]["segmented_fallback"] is True


def test_entity_only_segments_short_multi_statement_source_for_recall(monkeypatch):
    monkeypatch.setattr(Config, "entity_only_segment_max_chars", 4096)
    monkeypatch.setattr(Config, "entity_only_max_sentences_per_segment", 2)

    class SentenceBatchOllama:
        def __init__(self):
            self.calls = []

        def json_chat(self, **kwargs):
            source = kwargs["prompt"].split("Source text:\n", 1)[1]
            self.calls.append(source)
            rows = []
            for name in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon"):
                if name in source:
                    start = source.index(name)
                    rows.append({
                        "name": name,
                        "type": "Organization",
                        "mention_start": start,
                        "mention_end": start + len(name),
                    })
            return {"items": rows}

    source = (
        "Alpha operates. Beta supplies. Gamma designs. "
        "Delta audits. Epsilon delivers."
    )
    ollama = SentenceBatchOllama()
    result = SchemaEvolutionService(None, ollama).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert len(ollama.calls) == 6
    assert {item["name"] for item in result["entities"]} == {
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
    }
    assert result["_diagnostics"]["segmented_fallback"] is True


def test_relation_endpoint_coverage_returns_only_exact_entity_mentions():
    class EndpointOllama:
        def json_chat(self, **kwargs):
            assert "Do not return relations" in kwargs["system"]
            return {"entities": [
                {
                    "id": "missing-endpoint",
                    "name": "Beta",
                    "type": "Organization",
                    "mention_start": 15,
                    "mention_end": 19,
                    "confidence": 0.9,
                },
                {
                    "id": "invented",
                    "name": "Invented",
                    "type": "Organization",
                    "mention_start": 0,
                    "mention_end": 8,
                    "confidence": 0.9,
                },
            ]}

    result = SchemaEvolutionService(None, EndpointOllama()).extract_relation_endpoint_mentions(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": "Alpha operates Beta.", "metadata": {}},
        existing_entities=[{
            "name": "Alpha", "type": "Organization",
            "mention_start": 0, "mention_end": 5,
        }],
    )

    assert [(item["name"], item["mention_start"], item["mention_end"])
            for item in result["entities"]] == [("Beta", 15, 19)]
    assert result["relations"] == []
    assert result["_diagnostics"]["endpoint_coverage"] is True


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


def test_entity_only_prompt_requires_complete_event_and_state_arguments():
    class CapturingOllama:
        def __init__(self):
            self.systems = []

        def json_chat(self, **kwargs):
            self.systems.append(kwargs["system"])
            return {"items": []}

    ollama = CapturingOllama()
    SchemaEvolutionService(None, ollama).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": "控制器失效导致流程中断。", "metadata": {}},
    )

    assert "complete exact source span" in ollama.systems[0]
    assert "trigger or state head" in ollama.systems[0]
    assert "attached negation or modality" in ollama.systems[0]
    assert "完整事件、状态、条件、故障或变化" in ollama.systems[1]
    assert "否定和情态" in ollama.systems[1]
    assert "普通主谓宾事实" in ollama.systems[1]


def test_entity_only_focused_pass_recovers_complete_repeated_proposition_arguments():
    source = "设备数据采集中断导致告警。随后设备数据采集中断再次触发告警。"

    class PropositionArgumentOllama:
        def json_chat(self, **kwargs):
            if kwargs["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT:
                start = source.index("设备数据采集")
                return {"items": [{
                    "name": "设备数据采集",
                    "type": "Technology",
                    "mention_start": start,
                    "mention_end": start + len("设备数据采集"),
                }]}
            return {"arguments": [{
                "name": "设备数据采集中断",
                "type": "Event",
                "argument_complete": True,
                "proposition_argument": True,
            }]}

    result = SchemaEvolutionService(
        None, PropositionArgumentOllama(),
    ).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    complete = [
        item for item in result["entities"]
        if item["name"] == "设备数据采集中断"
    ]
    assert len(complete) == 2
    assert [item["mention_start"] for item in complete] == [
        source.index("设备数据采集中断"),
        source.rindex("设备数据采集中断"),
    ]
    assert result["_diagnostics"]["proposition_gleaning_calls"] == 1
    assert result["relations"] == []


def test_entity_only_optional_proposition_shape_failure_does_not_fail_snapshot():
    source = "The recipient must meet the requirement."

    class LegacyArgumentOllama:
        def json_chat(self, **kwargs):
            if kwargs["schema"] is ENTITY_ONLY_EXTRACTION_FORMAT:
                start = source.index("recipient")
                return {"entities": [{
                    "name": "recipient",
                    "type": "Organization",
                    "mention_start": start,
                    "mention_end": start + len("recipient"),
                }]}
            raise RuntimeError("Ollama JSON response omitted required fields: entities")

    result = SchemaEvolutionService(
        None, LegacyArgumentOllama(),
    ).extract_entities_only(
        {"llm_model": "test", "model_max_tokens": 8192},
        {"schema_json": BASE_SCHEMA},
        {"content": source, "metadata": {}},
    )

    assert [item["name"] for item in result["entities"]] == ["recipient"]
    assert result["_diagnostics"]["proposition_gleaning_failures"] == 2
    assert result["relations"] == []


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
