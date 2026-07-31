from __future__ import annotations

import inspect

from knowledge_forge.database import Database
from knowledge_forge.entity_shadow import ShadowEntityResolver
from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.service import KnowledgeForgeService


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

    def search_lexical(self, *_args):
        return list(self.lexical)

    def search_vector(self, *_args):
        return list(self.vector)

    def persist_candidates(self, rows):
        self.candidate_rows.extend(rows)

    def find_canonical(self, _namespace, canonical_key):
        return next(
            (row for row in [*self.exact, *self.lexical, *self.vector]
             if row["canonical_key"] == canonical_key),
            None,
        )


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
    source = inspect.getsource(KnowledgeForgeService.build_entity_shadow_namespace)
    assert "list_document_chunks" in source
    assert "build_shadow_context_manifest" in source
    assert "SpanFirstEntityExtractor" in source
    assert "schemas.extract_entities_only" in source
    assert "schemas.extract_chunk" not in source
    assert "list_document_mentions" not in source


def test_entity_shadow_pipeline_requeues_terminal_work_on_explicit_retry():
    source = inspect.getsource(KnowledgeForgeService.build_entity_shadow_namespace)

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
        "name": f"Acme {index}",
        "normalized_name": f"acme {index}",
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
