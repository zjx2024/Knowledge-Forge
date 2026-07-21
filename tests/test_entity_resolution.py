from knowledge_forge.entity_resolution import (
    EntityResolver,
    credible_aliases,
    identifiers_conflict,
    lexical_name_similarity,
    names_are_compatible,
    normalize_name,
    types_are_compatible,
)
from knowledge_forge.entity_identity import acronym_match, entity_identity_text, name_pair_features
from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.entity_reranker import EntityReranker
from knowledge_forge.observability import metrics_scope


def test_normalize_name_unifies_spacing_case_and_punctuation():
    assert normalize_name(" Knowledge-Forge ") == "knowledgeforge"
    assert normalize_name("上海 浦东（新区）") == "上海浦东新区"


def test_name_compatibility_is_conservative():
    assert names_are_compatible("星河机器人公司", "星河机器人")
    assert names_are_compatible("Knowledge Forge", "knowledge-forge")
    assert not names_are_compatible("星河机器人公司", "凌云科技公司")


def test_name_features_cover_general_acronym_and_format_variants():
    assert acronym_match("NIST", "National Institute of Standards and Technology")
    assert acronym_match("USGS", "U.S. Geological Survey")
    assert name_pair_features("CVE-2024-1234", "CVE 2024 1234")["exact"] == 1.0
    assert not acronym_match("CAT", "Category")
    assert "上下文" not in entity_identity_text("NIST", "Organization", "标准机构", ["美国国家标准与技术研究院"])


def test_unrelated_model_aliases_are_rejected():
    assert credible_aliases("凌云科技公司", ["凌云科技", "星河机器人公司"]) == ["凌云科技"]


def test_generic_entity_can_be_promoted_but_specific_type_conflicts_stay_separate():
    assert types_are_compatible("GenericEntity", "Product")
    assert types_are_compatible("Product", "Product")
    assert not types_are_compatible("Organization", "Product")


def test_lexical_score_and_identifier_conflicts_are_conservative():
    assert lexical_name_similarity("电动机控制单元", "电机控制单元") > 0.75
    assert identifiers_conflict("控制器 V2", "控制器 V3")
    assert not identifiers_conflict("控制器 V2", "二代控制器 V2")


def test_reranker_diagnostics_expose_runtime_fallback_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "entity_reranker_enabled", True)
    monkeypatch.setattr(Config, "entity_reranker_model_path", str(tmp_path))
    reranker = EntityReranker()
    reranker._last_fallback_reason = "simulated batch failure"

    diagnostics = reranker.diagnostics()

    assert diagnostics["reranker_enabled"] is True
    assert diagnostics["reranker_loaded"] is False
    assert diagnostics["last_score_used_model"] is False
    assert diagnostics["reranker_device"] == "cpu"
    assert diagnostics["fallback_reason"] == "simulated batch failure"


class FakeOllama:
    def __init__(self):
        self.chat_calls = 0
        self.chat_kwargs = []

    def embed(self, values, model):
        assert model == "embed"
        return [[1.0, 0.0] for _ in values]

    def json_chat(self, **kwargs):
        self.chat_calls += 1
        self.chat_kwargs.append(kwargs)
        return {
            "decision": "merge",
            "candidate_id": 7,
            "confidence": 0.96,
            "reason": "名称不同，但类型、描述和上下文表明是同一组织",
        }


class FakeVectors:
    def __init__(self):
        self.ollama = FakeOllama()
        self.entity_updates = []
        self.relation_updates = []

    def upsert_entity_embedding(self, project_id, entity, model):
        self.entity_updates.append((project_id, entity["id"], model))

    def upsert_relation_embedding(self, *args):
        self.relation_updates.append(args)


class FakeReranker:
    available = True

    def __init__(self, value=0.70):
        self.value = value

    def score(self, raw, candidates):
        return [self.value for _ in candidates]


def test_reranker_metrics_distinguish_model_score_from_runtime_fallback():
    class RuntimeFallbackReranker:
        available = True
        last_score_used_model = False

        @staticmethod
        def score(_raw, candidates):
            return [0.0] * len(candidates)

    resolver = EntityResolver(
        FakeResolutionDatabase(), FakeVectors(), reranker=RuntimeFallbackReranker(),
    )
    candidates = [{
        **resolver.db.entity,
        "resolution_scores": {"combined": 0.8, "lexical": 0.8, "vector": 0.8},
    }]

    with metrics_scope("test") as metrics:
        resolver._rerank_candidates({"name": "IBM", "type": "Organization"}, candidates)

    counters = metrics.snapshot()["counters"]
    assert counters["entity_reranker_fallback"] == 1
    assert "entity_reranker_used" not in counters


class FakeResolutionDatabase:
    def __init__(self, name="IBM", entity_type="Organization", population=1):
        self.entity = {
            "id": 7,
            "project_id": 3,
            "entity_key": "existing",
            "name": name,
            "normalized_name": normalize_name(name),
            "entity_type": entity_type,
            "aliases": [],
            "normalized_aliases": [],
            "description": "国际信息技术企业",
            "source_count": 4,
        }
        self.decision = None
        self.population = population
        self.lexical_limit = None
        self.vector_limit = None
        self.relation_candidates = []
        self.verification_events = []
        self.indexing_failures = []
        self.completed_indexing_failures = []
        self.blocked_ids = []

    def count_canonical_entities(self, *args):
        return self.population

    def find_exact_entity_candidates(self, *args):
        return []

    def blocked_entity_candidate_ids(self, _project_id, _mention_key, candidate_ids):
        return [value for value in candidate_ids if value in self.blocked_ids]

    def search_entity_name_candidates(self, *args):
        self.lexical_limit = args[3]
        return []

    def search_entity_vector_candidates(self, *args):
        self.vector_limit = args[4]
        return [{**self.entity, "vector_score": 0.95}]

    def merge_canonical_entity_evidence(self, entity_id, aliases, normalized_aliases, description):
        self.entity = {
            **self.entity,
            "aliases": list(dict.fromkeys([*self.entity["aliases"], *aliases])),
            "normalized_aliases": list(dict.fromkeys([*self.entity["normalized_aliases"], *normalized_aliases])),
            "description": description if len(description) > len(self.entity["description"]) else self.entity["description"],
        }
        return self.entity

    def update_canonical_entity_type(self, entity_id, entity_type):
        self.entity["entity_type"] = entity_type
        return self.entity

    def upsert_canonical_entity(self, values):
        return {"id": 8, **values, "source_count": 0}

    def add_entity_mention(self, values):
        return {"id": 11, **values}

    def add_entity_resolution_decision(self, values):
        self.decision = values
        return {"id": 12, **values}

    def add_relation_assertion(self, values):
        return {"id": 13, **values}

    def upsert_relation_candidate(self, values):
        self.relation_candidates.append(values)
        return values

    def add_relation_verification_event(self, values):
        self.verification_events.append(values)
        return values

    def persist_relation_audit_chain(self, candidates, events, assertions):
        self.relation_candidates.extend(candidates)
        self.verification_events.extend(events)
        return [{"id": 13 + index, **values} for index, values in enumerate(assertions)]

    def record_indexing_failure(self, values):
        self.indexing_failures.append(values)
        return values

    def complete_indexing_failure(self, project_id, object_type, object_key, model_name):
        self.completed_indexing_failures.append(
            (project_id, object_type, str(object_key), model_name)
        )


def test_dense_candidate_is_sent_to_llm_and_resolution_is_audited():
    database = FakeResolutionDatabase()
    vectors = FakeVectors()
    resolver = EntityResolver(database, vectors, reranker=FakeReranker())
    project = {"id": 3, "embedding_model": "embed", "llm_model": "chat"}
    mentions, relations = resolver.resolve_chunk(
        project,
        5,
        {"id": 9, "content": "国际商业机器公司发布了新产品。"},
        {
            "entities": [{
                "id": "e1", "name": "国际商业机器公司", "type": "Organization",
                "description": "一家信息技术企业", "aliases": [], "confidence": 0.9,
            }],
            "relations": [],
        },
        1,
    )

    assert len(mentions) == 1
    assert relations == []
    assert mentions[0]["canonical_entity_id"] == 7
    assert database.decision["decision"] == "llm_merge"
    assert database.decision["candidate_entity_ids"] == [7]
    assert vectors.ollama.chat_calls == 1
    assert vectors.ollama.chat_kwargs[0]["context_budget_tokens"] == 32768
    assert vectors.ollama.chat_kwargs[0]["max_output_tokens"] == Config.entity_review_max_output_tokens
    assert "max_tokens" not in vectors.ollama.chat_kwargs[0]
    assert "国际商业机器公司" in database.entity["aliases"]


def test_reranker_relevance_alone_does_not_trigger_automatic_merge():
    database = FakeResolutionDatabase("Microsoft Azure")
    vectors = FakeVectors()
    resolver = EntityResolver(database, vectors, reranker=FakeReranker(0.99))
    resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "content": "Microsoft develops cloud products including Azure."},
        {"entities": [{"id": "e1", "name": "Microsoft", "type": "Organization",
                        "description": "technology company", "aliases": [], "confidence": 0.9}], "relations": []},
        1,
    )

    assert database.decision["decision"] == "llm_merge"
    assert vectors.ollama.chat_calls == 1


def test_acronym_with_vector_and_reranker_evidence_can_rule_merge():
    database = FakeResolutionDatabase("National Institute of Standards and Technology")
    vectors = FakeVectors()
    resolver = EntityResolver(database, vectors, reranker=FakeReranker(0.99))
    resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "content": "NIST publishes standards."},
        {"entities": [{"id": "e1", "name": "NIST", "type": "Organization",
                        "description": "standards institute", "aliases": [], "confidence": 0.9}], "relations": []},
        1,
    )

    assert database.decision["decision"] == "rule_merge"
    assert vectors.ollama.chat_calls == 0


def test_relation_candidates_and_events_are_audited_but_only_publishable_relations_are_vectorized(monkeypatch):
    database = FakeResolutionDatabase()
    vectors = FakeVectors()
    resolver = EntityResolver(database, vectors, reranker=FakeReranker())
    entities = {
        "e1": {"id": 21, "name": "Alpha", "entity_type": "Organization"},
        "e2": {"id": 22, "name": "Beta", "entity_type": "Product"},
    }
    monkeypatch.setattr(
        resolver, "_resolve_one",
        lambda session, raw, chunk, version: (entities[raw["id"]], {"decision": "test", "candidate_ids": [], "scores": {}}),
    )
    base = {
        "source": "e1", "target": "e2", "type": "produces", "description": "",
        "confidence": 0, "evidence_text": "Alpha produces Beta.", "candidate_ordinal": 0,
        "_original_candidate": {
            "source": "e1", "source_name": "Alpha", "target": "e2", "target_name": "Beta",
            "type": "produces", "evidence_text": "Alpha produces Beta.", "confidence": 0,
        },
    }
    extraction = {
        "entities": [
            {"id": "e1", "name": "Alpha", "type": "Organization", "description": "", "aliases": [], "confidence": 0},
            {"id": "e2", "name": "Beta", "type": "Product", "description": "", "aliases": [], "confidence": 0},
        ],
        "relations": [
            {**base, "candidate_id": "c-verified", "verification_status": "llm_verified"},
            {**base, "candidate_id": "c-missing", "candidate_ordinal": 1,
             "verification_status": "verification_missing"},
        ],
        "_verification_events": [
            {"candidate_id": "c-verified", "attempt": 1, "model_name": "chat",
             "prompt_version": "p1", "schema_version": 1, "request_json": {}, "raw_response": {},
             "verdict": "supported", "reason": "ok", "error_category": ""},
        ],
    }
    mentions, relations = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5, {"id": 9, "content": "Alpha produces Beta."}, extraction, 1,
    )
    assert len(mentions) == 2
    assert len(relations) == 2
    assert [item["candidate_id"] for item in database.relation_candidates] == ["c-verified", "c-missing"]
    assert database.verification_events[0]["candidate_id"] == "c-verified"
    assert len(vectors.relation_updates) == 1
    assert relations[0]["confidence"] == 0


def test_relation_embedding_failure_records_retryable_outbox_without_downgrade(monkeypatch):
    database = FakeResolutionDatabase()
    vectors = FakeVectors()
    vectors.upsert_relation_embedding = lambda *args: (_ for _ in ()).throw(RuntimeError("embed down"))
    resolver = EntityResolver(database, vectors, reranker=FakeReranker())
    entities = {
        "e1": {"id": 21, "name": "Alpha", "entity_type": "Organization"},
        "e2": {"id": 22, "name": "Beta", "entity_type": "Product"},
    }
    monkeypatch.setattr(
        resolver, "_resolve_one",
        lambda session, raw, chunk, version: (
            entities[raw["id"]], {"decision": "test", "candidate_ids": [], "scores": {}},
        ),
    )
    extraction = {
        "entities": [
            {"id": "e1", "name": "Alpha", "type": "Organization", "description": "", "aliases": []},
            {"id": "e2", "name": "Beta", "type": "Product", "description": "", "aliases": []},
        ],
        "relations": [{
            "source": "e1", "target": "e2", "type": "produces", "description": "Alpha produces Beta",
            "confidence": 0.9, "evidence_text": "Alpha produces Beta.", "candidate_id": "c1",
            "candidate_ordinal": 0, "verification_status": "llm_verified",
            "metadata": {"verification_policy_version": "relation-verifier-p0-v1"},
            "_original_candidate": {
                "source": "e1", "source_name": "Alpha", "target": "e2", "target_name": "Beta",
                "type": "produces", "evidence_text": "Alpha produces Beta.", "confidence": 0.9,
            },
        }],
    }

    _, relations = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "content": "Alpha produces Beta."},
        extraction,
        1,
    )

    assert relations[0]["verification_status"] == "llm_verified"
    assert relations[0]["indexing_pending"] is True
    assert database.indexing_failures[0]["object_type"] == "relation"
    assert database.indexing_failures[0]["last_error"] == "embed down"


def test_candidate_recall_grows_sublinearly_and_respects_hard_caps():
    database = FakeResolutionDatabase(population=10000)
    resolver = EntityResolver(database, FakeVectors(), reranker=FakeReranker())
    session = resolver.start_session({"id": 3, "embedding_model": "embed", "llm_model": "chat"})

    resolver._retrieve_candidates(
        session,
        {"name": "International Business Machines", "type": "Organization", "description": "", "aliases": []},
        {"id": 9, "content": "International Business Machines publishes a report."},
    )

    assert Config.entity_lexical_top_k < database.lexical_limit <= Config.entity_lexical_max_k
    assert Config.entity_vector_top_k < database.vector_limit <= Config.entity_vector_max_k


def test_formal_exact_and_lexical_queries_are_alias_table_driven_and_bounded():
    exact_sql, exact_params = Database.build_exact_entity_candidates_query(
        3, "Organization", "ibm", 10,
    )
    lexical_sql, lexical_params = Database.build_lexical_entity_candidates_query(
        3, "Organization", "internationalbusinessmachines", 20, 0.45,
    )

    assert "canonical_entity_aliases" in exact_sql
    assert "normalized_aliases" not in exact_sql
    assert "canonical_entity_redirects" in exact_sql
    assert exact_params[-1] == 10
    assert "canonical_entity_aliases" in lexical_sql
    assert "unnest" not in lexical_sql
    assert lexical_params[-1] == 20


def test_unique_exact_hit_skips_embedding_reranker_and_llm():
    database = FakeResolutionDatabase()
    database.find_exact_entity_candidates = lambda *_args: [dict(database.entity)]
    vectors = FakeVectors()
    resolver = EntityResolver(
        database, vectors,
        reranker=FakeReranker(),
    )

    mentions, _ = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "revision_id": "rev-1", "content": "IBM publishes standards."},
        {"entities": [{
            "id": "e1", "name": "IBM", "type": "Organization", "description": "",
            "aliases": [], "evidence_text": "IBM", "evidence_start": 0, "evidence_end": 3,
        }], "relations": []},
        1,
    )

    assert mentions[0]["canonical_entity_id"] == 7
    assert vectors.ollama.chat_calls == 0
    assert vectors.entity_updates == []


def test_multiple_exact_hits_are_adjudicated_and_never_pick_first_implicitly():
    database = FakeResolutionDatabase()
    first = dict(database.entity)
    second = {**first, "id": 9, "entity_key": "second", "name": "IBM Research"}
    database.find_exact_entity_candidates = lambda *_args: [first, second]
    database.merge_canonical_entity_evidence = lambda entity_id, *_args: (
        second if entity_id == 9 else first
    )
    vectors = FakeVectors()
    vectors.ollama.json_chat = lambda **_kwargs: {
        "decision": "MATCH", "candidate_id": 9, "confidence": 0.99, "reason": "context",
    }
    resolver = EntityResolver(database, vectors, reranker=FakeReranker(0.8))

    mentions, _ = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "revision_id": "rev-1", "content": "IBM Research published the paper."},
        {"entities": [{
            "id": "e1", "name": "IBM", "type": "Organization",
            "description": "research division", "aliases": [],
            "evidence_text": "IBM", "evidence_start": 0, "evidence_end": 3,
        }], "relations": []},
        1,
    )

    assert mentions[0]["canonical_entity_id"] == 9
    assert database.decision["candidate_entity_ids"] == [7, 9]


def test_cannot_link_filters_exact_candidate_before_any_model_adjudication():
    database = FakeResolutionDatabase()
    database.find_exact_entity_candidates = lambda *_args: [dict(database.entity)]
    database.blocked_ids = [7]
    vectors = FakeVectors()
    resolver = EntityResolver(database, vectors, reranker=FakeReranker())

    mentions, _ = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "revision_id": "rev-1", "content": "IBM publishes standards."},
        {"entities": [{
            "id": "e1", "name": "IBM", "type": "Organization", "description": "",
            "aliases": [], "evidence_text": "IBM", "evidence_start": 0,
            "evidence_end": 3,
        }], "relations": []},
        1,
    )

    assert mentions[0]["canonical_entity_id"] == 8
    assert vectors.ollama.chat_calls == 0
    assert database.decision["decision"] == "new_no_candidate"


def test_cannot_link_filters_ambiguous_exact_candidate_before_rerank_and_llm():
    database = FakeResolutionDatabase()
    second = {**database.entity, "id": 9, "entity_key": "second"}
    database.find_exact_entity_candidates = lambda *_args: [dict(database.entity), second]
    database.blocked_ids = [7, 9]
    vectors = FakeVectors()
    resolver = EntityResolver(database, vectors, reranker=FakeReranker())

    mentions, _ = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"}, 5,
        {"id": 9, "revision_id": "rev-1", "content": "IBM publishes standards."},
        {"entities": [{
            "id": "e1", "name": "IBM", "type": "Organization", "description": "",
            "aliases": [], "evidence_text": "IBM", "evidence_start": 0,
            "evidence_end": 3,
        }], "relations": []}, 1,
    )

    assert mentions[0]["canonical_entity_id"] == 8
    assert vectors.ollama.chat_calls == 0
    assert database.decision["candidate_entity_ids"] == []


def test_cannot_link_is_rechecked_after_llm_match_before_merge():
    database = FakeResolutionDatabase()
    second = {**database.entity, "id": 9, "entity_key": "second"}
    database.find_exact_entity_candidates = lambda *_args: [dict(database.entity), second]
    checks = 0

    def blocked_after_llm(_project_id, _mention_key, candidate_ids):
        nonlocal checks
        checks += 1
        return [7] if checks >= 2 and 7 in candidate_ids else []

    database.blocked_entity_candidate_ids = blocked_after_llm
    vectors = FakeVectors()
    resolver = EntityResolver(database, vectors, reranker=FakeReranker())

    mentions, _ = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"}, 5,
        {"id": 9, "revision_id": "rev-1", "content": "IBM publishes standards."},
        {"entities": [{
            "id": "e1", "name": "IBM", "type": "Organization", "description": "",
            "aliases": [], "evidence_text": "IBM", "evidence_start": 0,
            "evidence_end": 3,
        }], "relations": []}, 1,
    )

    assert checks == 2
    assert mentions[0]["canonical_entity_id"] == 8
    assert database.decision["decision"] == "new_after_llm"


def test_new_same_name_mentions_use_position_identity_and_retry_is_idempotent():
    first = {
        "project_id": 3, "document_id": 5, "chunk_id": 9, "revision_id": "rev-1",
        "canonical_entity_id": 11, "mention_text": "Apple", "entity_type": "Organization",
        "evidence_text": "Apple", "evidence_start": 0, "evidence_end": 5,
    }
    second = {**first, "evidence_start": 20, "evidence_end": 25}

    first_key = Database._stable_mention_key(first)
    second_key = Database._stable_mention_key(second)

    assert first_key != second_key
    assert first_key == Database._stable_mention_key(dict(first))


def test_mention_identity_prefers_exact_mention_span_inside_shared_evidence():
    first = {
        "project_id": 3, "document_id": 5, "chunk_id": 9, "revision_id": "rev-1",
        "canonical_entity_id": 11, "mention_text": "Apple", "entity_type": "Organization",
        "evidence_text": "Apple acquired Apple.", "evidence_start": 0, "evidence_end": 21,
        "mention_start": 0, "mention_end": 5,
    }
    second = {**first, "mention_start": 15, "mention_end": 20}

    assert Database._stable_mention_key(first) != Database._stable_mention_key(second)
