from __future__ import annotations

from contextlib import contextmanager

import pytest

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.entity_identity import normalize_name
from knowledge_forge.entity_resolution import EntityResolver
from knowledge_forge.observability import metrics_scope


class _Ollama:
    def __init__(self):
        self.embedding_batches = []

    def embed(self, values, model):
        self.embedding_batches.append((list(values), model))
        return [[1.0, 0.0] for _ in values]

    def json_chat(self, **_kwargs):
        raise AssertionError("exact and no-candidate fixtures must not call the LLM")


class _Vectors:
    def __init__(self):
        self.ollama = _Ollama()
        self.entity_updates = []

    def upsert_entity_embedding(self, project_id, entity, model):
        self.entity_updates.append((project_id, entity["id"], model, None))

    def upsert_entity_embedding_with_vector(self, project_id, entity, model, vector):
        self.entity_updates.append((project_id, entity["id"], model, list(vector)))

    def upsert_relation_embedding(self, *_args):
        pass


class _BatchDatabase:
    def __init__(self, exact=True):
        self.exact = exact
        self.batch_exact_calls = []
        self.persist_calls = []
        self.next_id = 20

    def find_exact_entity_candidates_batch(self, project_id, requests):
        self.batch_exact_calls.append((project_id, list(requests)))
        if not self.exact:
            return {tuple(value): [] for value in requests}
        entity = {
            "id": 7,
            "project_id": project_id,
            "entity_key": "alpha",
            "name": "Alpha",
            "normalized_name": "alpha",
            "entity_type": "Organization",
            "aliases": [],
            "normalized_aliases": [],
            "description": "",
            "source_count": 1,
        }
        return {tuple(value): [entity] for value in requests}

    def find_exact_entity_candidates(self, *_args):
        raise AssertionError("per-mention exact lookup must not run after batch prefetch")

    def count_canonical_entities(self, *_args):
        return 0

    def search_entity_name_candidates(self, *_args):
        return []

    def search_entity_vector_candidates(self, *_args):
        return []

    def upsert_canonical_entity(self, values):
        self.next_id += 1
        return {"id": self.next_id, "source_count": 0, **values}

    def merge_canonical_entity_evidence(self, entity_id, aliases, normalized_aliases, description):
        return {
            "id": entity_id,
            "name": "Alpha",
            "normalized_name": "alpha",
            "entity_type": "Organization",
            "aliases": aliases,
            "normalized_aliases": normalized_aliases,
            "description": description,
            "source_count": 1,
        }

    def persist_entity_resolution_batch(self, mentions, decisions):
        self.persist_calls.append((list(mentions), list(decisions)))
        return [{"id": 100 + index, **value} for index, value in enumerate(mentions)]

    def persist_relation_audit_chain(self, _candidates, _events, assertions):
        return [{"id": 200 + index, **value} for index, value in enumerate(assertions)]


def _raw(local_id="e1"):
    return {
        "id": local_id,
        "name": "Alpha",
        "type": "Organization",
        "description": "",
        "aliases": [],
        "confidence": 0.9,
        "evidence_text": "Alpha",
        "evidence_start": 0,
        "evidence_end": 5,
    }


def test_chunk_exact_candidates_and_resolution_persistence_are_batched():
    database = _BatchDatabase(exact=True)
    vectors = _Vectors()
    resolver = EntityResolver(database, vectors)

    mentions, relations = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "content": "Alpha and Alpha."},
        {"entities": [_raw("e1"), _raw("e2")], "relations": []},
        1,
    )

    assert relations == []
    assert len(mentions) == 2
    assert database.batch_exact_calls == [(3, [("Organization", "alpha")])]
    assert len(database.persist_calls) == 1
    assert len(database.persist_calls[0][0]) == 2
    assert len(database.persist_calls[0][1]) == 2
    assert vectors.ollama.embedding_batches == []


def test_unresolved_duplicate_mentions_share_one_embedding_batch_and_vector():
    database = _BatchDatabase(exact=False)
    vectors = _Vectors()
    resolver = EntityResolver(database, vectors)

    mentions, _ = resolver.resolve_chunk(
        {"id": 3, "embedding_model": "embed", "llm_model": "chat"},
        5,
        {"id": 9, "content": "Alpha and Alpha."},
        {"entities": [_raw("e1"), _raw("e2")], "relations": []},
        1,
    )

    assert len(mentions) == 2
    assert len(vectors.ollama.embedding_batches) == 1
    assert len(vectors.ollama.embedding_batches[0][0]) == 1
    assert vectors.entity_updates == [(3, 21, "embed", [1.0, 0.0])]


class _Result:
    def __init__(self, value=None):
        self.value = value

    def fetchone(self):
        return self.value or {"schema": "public"}


class _IndexConnection:
    def __init__(self, fail_first=False):
        self.statements = []
        self.fail_first = fail_first

    def execute(self, statement, params=None):
        rendered = str(statement)
        self.statements.append((rendered, params))
        if "CREATE INDEX" in rendered and self.fail_first:
            self.fail_first = False
            raise RuntimeError("index build failed")
        if "current_schema" in rendered:
            return _Result({"schema": "public"})
        return _Result()


def test_hnsw_ensure_caches_only_successful_database_schema_model_dimension():
    Database._reset_hnsw_cache_for_tests()
    connection = _IndexConnection()
    database = Database()

    database._ensure_hnsw_index(connection, "embed", 2)
    database._ensure_hnsw_index(connection, "embed", 2)

    assert sum("CREATE INDEX" in statement for statement, _ in connection.statements) == 1


def test_hnsw_failed_build_is_retried_and_degraded_dimension_is_observable(monkeypatch):
    Database._reset_hnsw_cache_for_tests()
    connection = _IndexConnection(fail_first=True)
    database = Database()
    with pytest.raises(RuntimeError, match="index build failed"):
        database._ensure_hnsw_index(connection, "embed", 2)
    database._ensure_hnsw_index(connection, "embed", 2)
    assert sum("CREATE INDEX" in statement for statement, _ in connection.statements) == 2

    monkeypatch.setattr(database.config, "pgvector_hnsw_max_dimensions", 1)
    with metrics_scope("test") as metrics:
        database._ensure_hnsw_index(connection, "too-wide", 2)
    assert metrics.snapshot()["counters"]["hnsw_degraded_exact_scan"] == 1


def test_source_document_ddl_and_orphan_refresh_are_project_scoped():
    assert "CREATE TABLE IF NOT EXISTS canonical_entity_documents" in SCHEMA_SQL
    assert "UNIQUE(canonical_entity_id,document_id)" in SCHEMA_SQL
    assert "idx_canonical_entity_documents_project" in SCHEMA_SQL

    class Connection:
        def __init__(self):
            self.calls = []

        def execute(self, statement, params=None):
            self.calls.append((str(statement), params))
            return _Result()

    connection = Connection()
    Database._retire_orphan_entities(connection, project_id=3, entity_ids=[7, 8])
    combined = "\n".join(statement for statement, _ in connection.calls)
    assert "project_id=%s" in combined
    assert "id=ANY(%s)" in combined
    assert all(params and params[0] == 3 for _, params in connection.calls)
