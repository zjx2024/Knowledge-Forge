from __future__ import annotations

import os

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge.database import Database
from knowledge_forge.service import KnowledgeForgeService


class _FakeOllama:
    def __init__(self):
        self.embedding_calls = 0
        self.embedding_batch_sizes = []
        self.review_calls = 0

    def embed(self, texts, _model):
        self.embedding_calls += 1
        self.embedding_batch_sizes.append(len(texts))
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

    def json_chat(self, **_kwargs):
        self.review_calls += 1
        return {"decision": "separate", "canonical_key": "", "reason": "test"}


class _FakeSchemas:
    def __init__(self):
        self.calls = 0

    def extract_chunk(self, _project, _schema, chunk):
        self.calls += 1
        content = chunk["content"]
        if content.startswith("International"):
            full = "International Business Machines"
            alias = "IBM"
            product = "Model X"
            return {
                "entities": [
                    {
                        "id": "e1", "name": full, "type": "Organization",
                        "mention_start": content.index(full),
                        "mention_end": content.index(full) + len(full),
                        "evidence_text": content, "confidence": 0.95,
                        "alias_spans": [{
                            "surface": alias, "start": content.index(alias),
                            "end": content.index(alias) + len(alias), "confidence": 0.95,
                        }],
                    },
                    {
                        "id": "e2", "name": product, "type": "Product",
                        "mention_start": content.index(product),
                        "mention_end": content.index(product) + len(product),
                        "evidence_text": content, "confidence": 0.9,
                    },
                ],
                "relations": [{
                    "candidate_id": "r1", "source": "e1", "target": "e2",
                    "type": "produces", "evidence_text": content,
                    "evidence_start": 0, "evidence_end": len(content),
                    "verification_status": "uncertain",
                }],
            }
        name = "IBM"
        return {
            "entities": [{
                "id": "e1", "name": name, "type": "Organization",
                "mention_start": content.index(name),
                "mention_end": content.index(name) + len(name),
                "evidence_text": content, "confidence": 0.9,
            }],
            "relations": [],
        }

    def extract_entities_only(self, project, schema, chunk):
        result = self.extract_chunk(project, schema, chunk)
        return {**result, "relations": []}


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _seed_document(conn, project_id: int, ordinal: int, content: str) -> int:
    document_id = int(conn.execute(
        """INSERT INTO documents(
           project_id,doc_hash,original_filename,stored_filename,storage_path,
           local_path,file_size_bytes,status)
           VALUES(%s,%s,%s,%s,%s,%s,%s,'uploaded') RETURNING id""",
        (
            project_id, f"doc-hash-{ordinal}", f"doc-{ordinal}.txt", f"doc-{ordinal}.txt",
            f"/doc-{ordinal}", f"/doc-{ordinal}", len(content),
        ),
    ).fetchone()["id"])
    revision_id = f"revision-{document_id}"
    conn.execute(
        """INSERT INTO document_content_revisions(
           revision_id,project_id,document_id,ingestion_run_id,content_version,
           content_hash,source_hash,config_hash,policy_hash,manifest_hash,manifest,
           status,activated_at)
           VALUES(%s,%s,%s,%s,1,%s,%s,'config','policy',%s,'{}'::jsonb,'active',NOW())""",
        (
            revision_id, project_id, document_id, f"run-{document_id}",
            f"content-{ordinal}", f"source-{ordinal}", f"manifest-{ordinal}",
        ),
    )
    conn.execute(
        """UPDATE documents SET active_revision_id=%s,status='indexed',indexed_at=NOW()
           WHERE id=%s""",
        (revision_id, document_id),
    )
    section_id = int(conn.execute(
        """INSERT INTO sections(
           project_id,document_id,revision_id,section_key,title,level,content,content_hash)
           VALUES(%s,%s,%s,%s,%s,1,%s,%s) RETURNING id""",
        (
            project_id, document_id, revision_id, f"section-{ordinal}",
            f"Section {ordinal}", content, f"section-hash-{ordinal}",
        ),
    ).fetchone()["id"])
    conn.execute(
        """INSERT INTO chunks(
           project_id,document_id,revision_id,section_id,chunk_key,chunk_index,
           content,content_hash,metadata)
           VALUES(%s,%s,%s,%s,%s,0,%s,%s,'{}'::jsonb)""",
        (
            project_id, document_id, revision_id, section_id, f"chunk-{ordinal}",
            content, f"chunk-hash-{ordinal}",
        ),
    )
    return document_id


@pytest.mark.integration
def test_independent_entity_shadow_is_restart_safe_and_product_zero_write():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        project_id = int(conn.execute(
            """INSERT INTO projects(
               project_key,name,llm_model,embedding_model,active_schema_version)
               VALUES('entity-shadow','Entity Shadow','fake-llm','fake-embed',1)
               RETURNING id"""
        ).fetchone()["id"])
        conn.execute(
            """INSERT INTO graph_schemas(
               project_id,version,status,schema_json,source_document_ids,created_by,applied_at)
               VALUES(%s,1,'active',%s::jsonb,'{}','test',NOW())""",
            (project_id, '{"entity_types":[],"relation_types":[]}'),
        )
        _seed_document(
            conn, project_id, 1,
            "International Business Machines (IBM) produces Model X.",
        )
        _seed_document(conn, project_id, 2, "IBM expanded its operations.")

    formal_tables = (
        "canonical_entities", "entity_mentions", "entity_assignments",
        "vector_embeddings", "relation_assertions", "canonical_relation_projections",
    )
    with database.connection() as conn:
        before = {
            table: int(conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"])
            for table in formal_tables
        }

    service = KnowledgeForgeService(database)
    fake_ollama = _FakeOllama()
    fake_schemas = _FakeSchemas()
    service.ollama = fake_ollama
    service.schemas = fake_schemas
    service.resolver.reranker._disabled_reason = "test fixture"

    first = service.build_entity_shadow_namespace(project_id)
    first_model_calls = (fake_schemas.calls, fake_ollama.embedding_calls, fake_ollama.review_calls)
    second = service.build_entity_shadow_namespace(project_id)

    assert first["namespace_id"] == second["namespace_id"]
    assert (fake_schemas.calls, fake_ollama.embedding_calls, fake_ollama.review_calls) == first_model_calls
    assert fake_schemas.calls == 2
    assert fake_ollama.embedding_calls == 1
    assert fake_ollama.embedding_batch_sizes == [2]
    assignments = first["assignments"]
    ibm_assignments = [
        item for item in assignments
        if (item.get("metadata") or {}).get("evidence_start") in (0,)
    ]
    assert len({item["canonical_key"] for item in ibm_assignments}) == 1
    assert first["relation_endpoints"] == []

    with database.connection() as conn:
        after = {
            table: int(conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"])
            for table in formal_tables
        }
        namespace = conn.execute(
            "SELECT * FROM entity_shadow_namespaces WHERE id=%s",
            (first["namespace_id"],),
        ).fetchone()
        assert namespace["status"] == "ready"
        assert namespace["snapshot_complete"] is True
        assert namespace["expected_work_items"] == namespace["completed_work_items"] == 2
        assert namespace["failed_work_items"] == 0
    assert after == before
