from __future__ import annotations

import inspect
from contextlib import contextmanager

from knowledge_forge.database import Database
from knowledge_forge.entity_shadow import (
    ShadowEntityResolver,
    ShadowModelGateway,
    bounded_identity_evidence,
)
from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.service import KnowledgeForgeService


def test_exact_shadow_candidate_query_reads_persisted_evidence_text_column():
    statements = []

    class Result:
        def fetchall(self):
            return []

    class Connection:
        def execute(self, statement, params):
            statements.append((statement, params))
            return Result()

    @contextmanager
    def connection():
        yield Connection()

    database = object.__new__(Database)
    database.connection = connection

    assert database.find_exact_entity_shadow_candidates("ns", "甲", 3) == []
    statement, params = statements[0]
    assert "mention.evidence_text" in statement
    assert "mention.evidence\n" not in statement
    assert params == ("ns", "甲", 3)


def test_shadow_canonical_aliases_preserve_display_surfaces():
    aliases = Database._entity_shadow_display_aliases(
        {
            "name": "Compound statements",
            "aliases": ["Compound statements"],
        },
        {"surface": "compound statement"},
        [{"surface": "compound-statement construct"}],
    )

    assert aliases == [
        "Compound statements",
        "compound statement",
        "compound-statement construct",
    ]


class _CountingModels:
    def __init__(self):
        self.embedding_calls = 0
        self.review_calls = 0

    def embed(self, texts, _model):
        self.embedding_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def review(self, *_args, **_kwargs):
        self.review_calls += 1
        return None


class _MergingModels(_CountingModels):
    def __init__(self, canonical_key):
        super().__init__()
        self.canonical_key = canonical_key

    def review(self, *_args, **_kwargs):
        self.review_calls += 1
        return {
            "decision": "merge",
            "canonical_key": self.canonical_key,
            "reason": "two exact contexts identify the same entity",
        }


class _CapturingMergingModels(_MergingModels):
    def __init__(self, canonical_key):
        super().__init__(canonical_key)
        self.reviewed_candidates = []

    def review(self, _mention, candidates):
        self.reviewed_candidates.append(list(candidates))
        return super().review(_mention, candidates)


class _FailingReviewModels(_CountingModels):
    def review(self, *_args, **_kwargs):
        self.review_calls += 1
        raise RuntimeError("Ollama response was truncated by output length")


class _ShadowCandidates:
    def __init__(self, exact=None, lexical=None, vector=None):
        self.exact = exact or []
        self.lexical = lexical or []
        self.vector = vector or []
        self.candidate_rows = []

    def find_exact(self, *_args):
        return list(self.exact)

    def search_lexical(self, *_args, **_kwargs):
        return list(self.lexical)

    def search_vector(self, *_args, **_kwargs):
        return list(self.vector)

    def persist_candidates(self, rows):
        self.candidate_rows.extend(rows)

    def find_canonical(self, _namespace, canonical_key):
        return next(
            (row for row in [*self.exact, *self.lexical, *self.vector]
             if row["canonical_key"] == canonical_key),
            None,
        )


def test_singleton_exact_identity_review_repairs_omitted_candidate_key():
    class Ollama:
        def json_chat(self, **_kwargs):
            return {"decision": "merge"}

    gateway = ShadowModelGateway(
        Ollama(), llm_model="fixture", context_budget_tokens=8192,
    )
    result = gateway.review(
        {"surface": "共享实验室", "entity_type": "Organization", "evidence": ""},
        [{
            "canonical_key": "canonical-lab",
            "name": "共享实验室",
            "normalized_name": "共享实验室",
            "primary_type": "Product",
            "aliases": ["共享实验室"],
            "mention_evidence": "",
        }],
    )

    assert result["canonical_key"] == "canonical-lab"


def test_singleton_exact_identity_review_accepts_bounded_result_alias():
    class Ollama:
        def json_chat(self, **_kwargs):
            return {"result": "merge"}

    gateway = ShadowModelGateway(
        Ollama(), llm_model="fixture", context_budget_tokens=8192,
    )
    result = gateway.review(
        {"surface": "共享实验室", "entity_type": "Organization", "evidence": ""},
        [{
            "canonical_key": "canonical-lab",
            "name": "共享实验室",
            "normalized_name": "共享实验室",
            "primary_type": "Product",
            "aliases": ["共享实验室"],
            "mention_evidence": "",
        }],
    )

    assert result["decision"] == "merge"
    assert result["canonical_key"] == "canonical-lab"


def test_identity_evidence_includes_neighboring_source_sentences_without_leaving_source():
    content = (
        "An extended client establishes a connection.\n"
        "The client sends requests.\n"
        "A different server handles responses."
    )
    start = content.index("client", content.index("The client"))

    evidence = bounded_identity_evidence(content, start, start + len("client"))

    assert evidence == content
    assert "extended client" in evidence
    assert "different server" in evidence


def _mention(**overrides):
    value = {
        "mention_id": "m1",
        "surface": "IBM",
        "normalized_surface": "ibm",
        "entity_type": "Organization",
        "aliases": [],
        "document_id": 1,
        "revision_id": "rev-1",
        "chunk_id": 10,
        "start": 0,
        "end": 3,
    }
    value.update(overrides)
    return value


def test_entity_shadow_pipeline_reads_active_chunks_not_formal_mentions():
    source = inspect.getsource(KnowledgeForgeService.build_grounded_entity_namespace)
    assert "list_document_chunks" in source
    assert "build_grounded_context_manifest" in source
    assert "SpanFirstEntityExtractor" in source
    assert "structured_relation_endpoint_candidates" in source
    assert "schemas.extract_entities_only" in source
    assert "schemas.extract_chunk" not in source
    assert "list_document_mentions" not in source
    assert '"relation_policy_set"' not in source
    assert '"ollama_seed": Config.ollama_seed' in source
    assert 'snapshot_metrics.get("relation_endpoints")' in source
    assert '"relation_endpoints": relation_endpoint_count' in source


def test_ready_entity_namespace_reaudits_publishability_after_relation_refresh():
    source = inspect.getsource(KnowledgeForgeService.build_grounded_entity_namespace)
    refresh = source[
        source.index("if relation_refresh_required:"):
        source.index("# A relation-only refresh reuses the immutable entity namespace.")
    ]

    assert refresh.index("_run_grounded_relation_namespace") < refresh.index(
        "_audit_grounded_entity_publishability"
    )
    assert 'persisted.get("relation_endpoints")' in refresh
    assert '"after_relations": publishability_after_relations' in refresh
    assert refresh.count("load_entity_shadow_snapshot") >= 2


def test_ready_entity_namespace_repairs_interrupted_projection_by_endpoint_hash():
    source = inspect.getsource(KnowledgeForgeService.build_grounded_entity_namespace)

    assert "current_endpoint_fingerprint" in source
    assert "relation_endpoint_fingerprint" in source
    assert "publishability_metrics_changed" in source


def test_relation_endpoint_projection_fingerprint_is_order_independent_but_directed():
    first = [
        {
            "relation_key": "r2",
            "source_canonical_key": "b",
            "target_canonical_key": "c",
        },
        {
            "relation_key": "r1",
            "source_canonical_key": "a",
            "target_canonical_key": "b",
        },
    ]
    reordered = list(reversed(first))
    reversed_edge = [
        first[0],
        {
            **first[1],
            "source_canonical_key": "b",
            "target_canonical_key": "a",
        },
    ]

    fingerprint = KnowledgeForgeService._grounded_relation_endpoint_fingerprint
    assert fingerprint(first) == fingerprint(reordered)
    assert fingerprint(first) != fingerprint(reversed_edge)


def test_entity_shadow_honors_retrieval_only_chunk_metadata():
    chunks = [
        {"id": 1, "metadata": {}},
        {"id": 2, "metadata": {"graph_extraction_enabled": True}},
        {"id": 3, "metadata": {"graph_extraction_enabled": False}},
        {"id": 4, "metadata": '{"graph_extraction_enabled": false}'},
    ]

    selected = KnowledgeForgeService._graph_extraction_chunks(chunks)

    assert [item["id"] for item in selected] == [1, 2]


def test_entity_shadow_pipeline_requeues_terminal_work_on_explicit_retry():
    source = inspect.getsource(KnowledgeForgeService.build_grounded_entity_namespace)

    assert "requeue_terminal_entity_shadow_work" in source
    assert source.index("requeue_terminal_entity_shadow_work") < source.index("while True")


def test_entity_shadow_claim_recovers_expired_final_attempt_without_consuming_another_attempt():
    source = inspect.getsource(Database.claim_entity_shadow_work)

    assert "(status IN ('pending','retryable_failed') AND attempt<max_attempts)" in source
    assert "OR (status='leased' AND lease_expires_at<NOW())" in source
    assert "CASE WHEN work.status='leased' THEN work.attempt" in source


def test_entity_shadow_migration_persists_independent_audit_and_work_state():
    sql = next(
        item.sql for item in MIGRATIONS if item.version == "0003_entity_shadow_pipeline"
    )
    for table in (
        "entity_shadow_work_items",
        "entity_shadow_mentions",
        "entity_shadow_type_assertions",
        "entity_shadow_aliases",
        "entity_shadow_candidates",
        "entity_shadow_embeddings",
        "entity_shadow_must_links",
        "entity_shadow_events",
        "entity_shadow_relation_endpoints",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "gin(normalized_alias gin_trgm_ops)" in sql
    assert "input_fingerprint" in sql
    assert "snapshot_complete" in sql


def test_grounded_entity_extraction_cache_is_exact_input_and_provenance_bound():
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0032_knowledge_forge_entity_extraction_cache"
    )

    assert "knowledge_forge_entity_extraction_cache" in migration.sql
    assert "project_id BIGINT NOT NULL" in migration.sql
    assert "document_id BIGINT NOT NULL" in migration.sql
    assert "revision_id TEXT NOT NULL" in migration.sql
    assert "chunk_id BIGINT NOT NULL" in migration.sql
    assert "model_digest TEXT NOT NULL" in migration.sql
    assert "request_fingerprint TEXT NOT NULL" in migration.sql
    assert "output_hash TEXT NOT NULL" in migration.sql
    assert migration.startup_safe is True

    source = inspect.getsource(
        KnowledgeForgeService.build_grounded_entity_namespace
    )
    for required in (
        "schema_fingerprint",
        "model_digest",
        "effective_context_tokens",
        "context_budget_tokens",
        "max_output_tokens",
        "seed",
        "segment_max_chars",
        "max_sentences_per_segment",
        "max_items_per_segment",
        "gleaning_rounds",
        "entity_extraction_policy_version",
        "code_version",
    ):
        assert f'"{required}"' in source
    assert source.index(
        "get_knowledge_forge_entity_extraction_cache"
    ) < source.index("schemas.extract_grounded_mentions")
    assert "delete_knowledge_forge_entity_extraction_cache" in source


def test_project_activity_repair_is_bounded_to_maintenance_batches():
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0033_project_activity_timestamp_repair"
    )

    assert migration.startup_safe is True
    assert "HAVING COUNT(*)>1" in migration.sql
    assert "graph_error='图谱需要按当前策略集合和代次重建'" in migration.sql
    assert "MAX(j.updated_at)" in migration.sql
    assert "MAX(d.updated_at)" in migration.sql
    assert "MAX(s.updated_at)" in migration.sql


def test_exact_alias_hit_skips_embedding_reranker_and_llm():
    store = _ShadowCandidates(exact=[{
        "canonical_key": "c1",
        "name": "International Business Machines",
        "normalized_name": "international business machines",
        "primary_type": "Organization",
        "aliases": ["ibm"],
    }])
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=models,
        reranker=lambda *_args: (_ for _ in ()).throw(AssertionError("reranker called")),
    )

    result = resolver.resolve(_mention())

    assert result["canonical_key"] == "c1"
    assert result["decision"] == "exact_alias"
    assert models.embedding_calls == 0
    assert models.review_calls == 0


def test_exact_canonical_name_profile_does_not_merge_an_alias_like_near_match():
    candidate = {
        "canonical_key": "c-base",
        "name": "Module Base",
        "normalized_name": "module base",
        "primary_type": "Technology",
        "aliases": ["Module Variant"],
        "lexical_score": 1.0,
        "vector_score": 1.0,
    }
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns",
        policy_version="p2-evidence",
        store=_ShadowCandidates(exact=[candidate], lexical=[candidate], vector=[candidate]),
        models=models,
        reranker=lambda *_args: (_ for _ in ()).throw(AssertionError("reranker called")),
        identity_profile="exact_canonical_name_only",
    )

    result = resolver.resolve(_mention(
        surface="Module Variant",
        normalized_surface="module variant",
        entity_type="Technology",
    ))

    assert result["decision"] == "new_canonical_exact_name_policy"
    assert result["canonical_key"] != "c-base"
    assert models.review_calls == 0


def test_multilingual_exact_alias_and_ambiguous_homonym_stay_model_free():
    exact = [{
        "canonical_key": "c-cn",
        "name": "国际商业机器公司",
        "normalized_name": "国际商业机器公司",
        "primary_type": "Organization",
        "aliases": ["ibm", "国际商业机器公司"],
    }]
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2",
        store=_ShadowCandidates(exact=exact), models=models,
        reranker=lambda *_args: [],
    )

    result = resolver.resolve(_mention(surface="国际商业机器公司", normalized_surface="国际商业机器公司"))

    assert result["canonical_key"] == "c-cn"
    assert result["decision"] == "exact_alias"
    assert models.embedding_calls == models.review_calls == 0

    ambiguous_store = _ShadowCandidates(exact=[
        {**exact[0], "canonical_key": "c-city", "name": "苹果市"},
        {**exact[0], "canonical_key": "c-company", "name": "苹果公司"},
    ])
    ambiguous = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=ambiguous_store, models=models,
        reranker=lambda *_args: [],
    ).resolve(_mention(surface="苹果", normalized_surface="苹果"))
    assert ambiguous["decision"] == "quarantined_ambiguous_exact"
    assert ambiguous["canonical_key"].startswith("shadow_entity_")
    assert ambiguous["quarantined"] is True
    assert models.embedding_calls == models.review_calls == 0


def test_duplicate_exact_name_and_type_converge_deterministically():
    rows = [{
        "canonical_key": key,
        "name": "Shared Component",
        "normalized_name": "shared component",
        "primary_type": "Technology",
        "aliases": ["shared component"],
    } for key in ("canonical-b", "canonical-a")]
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns",
        policy_version="p2-name-type",
        store=_ShadowCandidates(exact=rows),
        models=models,
        reranker=lambda *_args: [],
        identity_profile="exact_canonical_name_only",
    )

    result = resolver.resolve(_mention(
        surface="Shared Component",
        normalized_surface="shared component",
        entity_type="Technology",
    ))

    assert result["canonical_key"] == "canonical-a"
    assert result["decision"] == "exact_name_type_convergence"
    assert models.embedding_calls == models.review_calls == 0


def test_exact_name_cross_type_conflict_requires_context_review():
    candidate = {
        "canonical_key": "c-platform",
        "name": "星云平台",
        "normalized_name": "星云平台",
        "primary_type": "Technology",
        "aliases": ["星云平台"],
        "mention_evidence": "机器人通过星云平台完成巡检。",
    }
    models = _MergingModels("c-platform")
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2",
        store=_ShadowCandidates(exact=[candidate]), models=models,
        reranker=lambda *_args: [],
    )

    result = resolver.resolve(_mention(
        surface="星云平台",
        normalized_surface="星云平台",
        entity_type="Product",
        evidence="星云平台为机器人提供视觉识别能力。",
    ))

    assert result["canonical_key"] == "c-platform"
    assert result["decision"] == "exact_name_cross_type_review"
    assert models.review_calls == 1
    assert models.embedding_calls == 0


def test_exact_name_cross_type_review_failure_keeps_entities_separate():
    candidate = {
        "canonical_key": "c-apple-company",
        "name": "苹果",
        "normalized_name": "苹果",
        "primary_type": "Organization",
        "aliases": ["苹果"],
        "mention_evidence": "苹果公司发布了年度报告。",
    }
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2",
        store=_ShadowCandidates(exact=[candidate]), models=models,
        reranker=lambda *_args: [],
    )

    result = resolver.resolve(_mention(
        surface="苹果",
        normalized_surface="苹果",
        entity_type="Product",
        evidence="果农采摘苹果并装箱运输。",
    ))

    assert result["canonical_key"] != "c-apple-company"
    assert result["decision"] == "new_canonical"
    assert models.review_calls == 1

def test_hard_identifier_and_cannot_link_win_over_similarity():
    candidates = [{
        "canonical_key": "model-2",
        "name": "Model 2",
        "normalized_name": "model 2",
        "primary_type": "Product",
        "aliases": [],
        "lexical_score": 0.99,
        "vector_score": 0.99,
    }]
    store = _ShadowCandidates(lexical=candidates, vector=candidates)
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=models,
        reranker=lambda _mention, rows: [0.99 for _ in rows],
        cannot_links={frozenset(("m2", "model-2"))},
        reranker_cap=3,
    )

    result = resolver.resolve(_mention(mention_id="m2", surface="Model 1", normalized_surface="model 1"))

    assert result["decision"] == "new_canonical"
    assert result["canonical_key"] != "model-2"
    assert any(
        "identifier_conflict" in item["filter_reason"]
        for item in store.candidate_rows
    )


def test_cjk_near_edit_names_remain_separate_despite_maximum_similarity():
    candidate = {
        "canonical_key": "c-intra-partition",
        "name": "分区内通信管理",
        "normalized_name": "分区内通信管理",
        "primary_type": "GenericEntity",
        "aliases": [],
        "lexical_score": 1.0,
        "vector_score": 1.0,
    }
    store = _ShadowCandidates(lexical=[candidate], vector=[candidate])
    models = _MergingModels("c-intra-partition")
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=models,
        reranker=lambda _mention, rows: [1.0 for _ in rows],
    )

    result = resolver.resolve(_mention(
        surface="分区间通信管理",
        normalized_surface="分区间通信管理",
        entity_type="GenericEntity",
    ))

    assert result["decision"] == "new_canonical"
    assert result["canonical_key"] != "c-intra-partition"
    assert models.review_calls == 0
    assert any(
        "cjk_near_edit_ambiguous" in item["filter_reason"]
        for item in store.candidate_rows
    )


def test_numeric_identifier_extension_is_not_a_similarity_alias():
    candidate = {
        "canonical_key": "c-mips-family",
        "name": "MIPS",
        "normalized_name": "mips",
        "primary_type": "Technology",
        "aliases": [],
        "lexical_score": 1.0,
        "vector_score": 1.0,
    }
    store = _ShadowCandidates(lexical=[candidate], vector=[candidate])
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store,
        models=_MergingModels("c-mips-family"),
        reranker=lambda _mention, rows: [1.0 for _ in rows],
    )

    result = resolver.resolve(_mention(
        surface="mips64", normalized_surface="mips64",
        entity_type="Technology",
    ))

    assert result["decision"] == "new_canonical"
    assert result["canonical_key"] != "c-mips-family"
    assert any(
        "identifier_conflict" in item["filter_reason"]
        for item in store.candidate_rows
    )


def test_shadow_type_gate_blocks_material_product_embedding_merge():
    candidates = [{
        "canonical_key": "c-carbonate",
        "name": "lithium carbonate",
        "normalized_name": "lithiumcarbonate",
        "primary_type": "Product",
        "aliases": ["lithium compounds"],
        "lexical_score": 1.0,
        "vector_score": 1.0,
    }]
    store = _ShadowCandidates(lexical=candidates, vector=candidates)
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=models,
        reranker=lambda _mention, rows: [1.0 for _ in rows],
        reranker_cap=3,
    )

    result = resolver.resolve(_mention(
        surface="lithium compounds",
        normalized_surface="lithiumcompounds",
        entity_type="Material",
    ))

    assert result["decision"] == "new_canonical"
    assert result["canonical_key"] != "c-carbonate"
    assert any(
        "type_incompatible" in item["filter_reason"]
        for item in store.candidate_rows
    )


def test_unresolved_type_is_missing_information_not_a_hard_type_conflict():
    candidate = {
        "canonical_key": "c-service",
        "name": "Service",
        "normalized_name": "service",
        "primary_type": "Organization",
        "aliases": ["service"],
    }
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=_ShadowCandidates(exact=[candidate]),
        models=models, reranker=lambda *_args: [],
    )

    result = resolver.resolve(_mention(
        surface="service",
        normalized_surface="service",
        entity_type="GenericEntity",
    ))

    assert result["canonical_key"] == "c-service"
    assert result["decision"] == "exact_alias"
    assert models.embedding_calls == models.review_calls == 0


def test_singular_plural_name_variant_merges_without_identity_model_guessing():
    candidate = {
        "canonical_key": "c-compound",
        "name": "Compound statements",
        "normalized_name": "compoundstatements",
        "primary_type": "GenericEntity",
        "aliases": [],
        "lexical_score": 0.97,
        "vector_score": 0.65,
    }
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2",
        store=_ShadowCandidates(lexical=[candidate], vector=[candidate]),
        models=models,
        reranker=lambda _mention, rows: [0.0 for _ in rows],
    )

    result = resolver.resolve(_mention(
        surface="compound statement",
        normalized_surface="compoundstatement",
        entity_type="Technology",
    ))

    assert result["canonical_key"] == "c-compound"
    assert result["decision"] == "bounded_inflectional_alias"
    assert models.review_calls == 0


def test_generic_nested_source_spans_remain_separate_without_identity_review():
    candidate = {
        "canonical_key": "c-platform",
        "name": "platform",
        "normalized_name": "platform",
        "primary_type": "Product",
        "aliases": [],
        "lexical_score": 1.0,
        "vector_score": 1.0,
    }
    store = _ShadowCandidates(lexical=[candidate], vector=[candidate])
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store,
        models=models,
        reranker=lambda _mention, rows: [1.0 for _ in rows],
    )

    result = resolver.resolve(_mention(
        surface="cloud-ready research platform",
        normalized_surface="cloud-ready research platform",
        entity_type="Product",
    ))

    assert result["decision"] == "new_canonical"
    assert result["canonical_key"] != "c-platform"
    assert all(
        "nested_identity_ambiguous" in item["filter_reason"]
        for item in store.candidate_rows
    )
    assert models.review_calls == 0


def test_nested_short_name_can_merge_only_after_explicit_context_review():
    candidate = {
        "canonical_key": "c-operator",
        "name": "值班员周启",
        "normalized_name": "值班员周启",
        "primary_type": "Person",
        "aliases": [],
        "lexical_score": 1.0,
        "vector_score": 1.0,
        "mention_evidence": "值班员周启将温度计交给巡检员。",
    }
    store = _ShadowCandidates(lexical=[candidate], vector=[candidate])
    models = _MergingModels("c-operator")
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=models,
        reranker=lambda _mention, rows: [1.0 for _ in rows],
    )

    result = resolver.resolve(_mention(
        surface="周启",
        normalized_surface="周启",
        entity_type="Person",
        evidence="周启随后填写异常报告。",
    ))

    assert result["canonical_key"] == "c-operator"
    assert result["decision"] == "bounded_nested_identity_review_merge"
    assert models.review_calls == 1


def test_nested_identity_review_is_pairwise_even_with_high_scoring_distractors():
    target = {
        "canonical_key": "c-inspector-alice",
        "name": "Inspector Alice",
        "normalized_name": "inspectoralice",
        "primary_type": "Person",
        "aliases": [],
        "lexical_score": 0.75,
        "vector_score": 0.99,
    }
    distractors = [{
        "canonical_key": f"c-distractor-{index}",
        "name": f"related object {index}",
        "normalized_name": f"relatedobject{index}",
        "primary_type": "GenericEntity",
        "aliases": [],
        "lexical_score": 0.70 - index / 100,
        "vector_score": 0.95 - index / 100,
    } for index in range(8)]
    rows = [target, *distractors]
    models = _CapturingMergingModels("c-inspector-alice")
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2",
        store=_ShadowCandidates(lexical=rows, vector=rows),
        models=models,
        reranker=lambda _mention, candidates: [
            1.0 if item["canonical_key"] == "c-http-client" else 0.8
            for item in candidates
        ],
        reranker_cap=9,
        reviewer_cap=5,
    )

    result = resolver.resolve(_mention(
        surface="Alice",
        normalized_surface="alice",
        entity_type="Person",
        endpoint_candidate=True,
    ))

    assert result["canonical_key"] == "c-inspector-alice"
    assert result["decision"] == "bounded_nested_identity_review_merge"
    assert len(models.reviewed_candidates) == 1
    assert [item["canonical_key"] for item in models.reviewed_candidates[0]] == [
        "c-inspector-alice",
    ]


def test_generic_nested_capability_names_do_not_merge_after_model_review():
    candidate = {
        "canonical_key": "c-partition-service",
        "name": "分区管理服务",
        "normalized_name": "分区管理服务",
        "primary_type": "GenericEntity",
        "aliases": [],
        "lexical_score": 1.0,
        "vector_score": 1.0,
    }
    store = _ShadowCandidates(lexical=[candidate], vector=[candidate])
    models = _MergingModels("c-partition-service")
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=models,
        reranker=lambda _mention, rows: [1.0 for _ in rows],
    )

    result = resolver.resolve(_mention(
        surface="分区管理", normalized_surface="分区管理",
        entity_type="GenericEntity",
    ))

    assert result["decision"] == "new_canonical"
    assert result["canonical_key"] != "c-partition-service"
    assert models.review_calls == 0
    assert any(
        "nested_identity_ambiguous" in item["filter_reason"]
        for item in store.candidate_rows
    )


def test_hard_must_link_is_audited_but_cannot_override_identifier_conflict():
    candidate = {
        "canonical_key": "model-2", "name": "Model 2",
        "normalized_name": "model 2", "primary_type": "Product", "aliases": [],
    }
    models = _CountingModels()
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2",
        store=_ShadowCandidates(exact=[candidate]), models=models,
        reranker=lambda *_args: [], must_links={"m1": "model-2"},
    )

    result = resolver.resolve(_mention(surface="Model 1", normalized_surface="model 1"))

    assert result["decision"] == "uncertain_constraint_conflict"
    assert result["canonical_key"] != "model-2"
    assert models.embedding_calls == models.review_calls == 0


def test_shadow_candidate_reranking_is_hard_bounded():
    rows = [{
        "canonical_key": f"c{index}",
        "name": f"Acme Variant {chr(65 + index)}",
        "normalized_name": f"acme variant {chr(65 + index)}",
        "primary_type": "Organization",
        "aliases": [],
        "lexical_score": 0.7,
    } for index in range(20)]
    store = _ShadowCandidates(lexical=rows)
    seen = []
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=_CountingModels(),
        reranker=lambda _mention, selected: seen.extend(selected) or [0.1 for _ in selected],
        reranker_cap=4,
    )

    resolver.resolve(_mention(surface="Acme", normalized_surface="acme"))

    assert len(seen) == 4
    assert len([item for item in store.candidate_rows if item["stage"] == "retrieval"]) <= 20
    assert len([item for item in store.candidate_rows if item["stage"] == "reranker"]) <= 4


def test_optional_llm_review_failure_fails_closed_to_new_canonical():
    candidate = {
        "canonical_key": "c-acme",
        "name": "Acme Holdings",
        "normalized_name": "acme holdings",
        "primary_type": "Organization",
        "aliases": [],
        "lexical_score": 0.80,
        "vector_score": 0.80,
    }
    models = _FailingReviewModels()
    resolver = ShadowEntityResolver(
        namespace="ns",
        policy_version="p2",
        store=_ShadowCandidates(lexical=[candidate], vector=[candidate]),
        models=models,
        reranker=lambda _mention, _rows: [0.80],
        auto_merge_threshold=0.99,
        review_threshold=0.60,
    )

    result = resolver.resolve(_mention(surface="Acme", normalized_surface="acme"))

    assert result["decision"] == "new_canonical"
    assert result["canonical_key"] != "c-acme"
    assert models.review_calls == 1


def test_candidate_recall_keeps_gold_inside_bounded_retrieval_audit():
    rows = [{
        "canonical_key": "gold",
        "name": "International Business Machines",
        "normalized_name": "international business machines",
        "primary_type": "Organization",
        "aliases": ["ibm"],
        "lexical_score": 0.91,
    }, {
        "canonical_key": "distractor",
        "name": "International Business Methods",
        "normalized_name": "international business methods",
        "primary_type": "Organization",
        "aliases": [],
        "lexical_score": 0.82,
    }]
    store = _ShadowCandidates(lexical=rows)
    resolver = ShadowEntityResolver(
        namespace="ns", policy_version="p2", store=store, models=_CountingModels(),
        reranker=lambda _mention, selected: [0.95 if row["canonical_key"] == "gold" else 0.1
                                              for row in selected],
        lexical_cap=5, vector_cap=5, reranker_cap=2,
    )

    resolver.resolve(_mention(surface="IBM Corp", normalized_surface="ibm corp"))

    retrieved = [
        row for row in store.candidate_rows
        if row["stage"] == "retrieval"
    ]
    assert "gold" in {row["canonical_key"] for row in retrieved}
    assert all(row["rank"] <= row["cap"] == 10 for row in retrieved)
