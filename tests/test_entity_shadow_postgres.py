from __future__ import annotations

import json
import os
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge.database import Database
from knowledge_forge.service import KnowledgeForgeService
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


class _FakeOllama:
    def __init__(self):
        self.embedding_calls = 0
        self.embedding_batch_sizes = []
        self.review_calls = 0

    def embed(self, texts, _model):
        self.embedding_calls += 1
        self.embedding_batch_sizes.append(len(texts))
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]

    def json_chat(self, **kwargs):
        if "classifier" in str(kwargs.get("system") or ""):
            payload = json.loads(kwargs["prompt"])
            return {"items": [{
                "pair_id": pair["pair_id"],
                "decision": "NO_RELATION",
                "hypotheses": [],
                "reason_code": "FIXTURE_NO_RELATION",
            } for pair in payload["pairs"]]}
        if "verifier" in str(kwargs.get("system") or ""):
            raise AssertionError("NO_RELATION fixture must not call verifier")
        self.review_calls += 1
        return {"decision": "separate", "canonical_key": "", "reason": "test"}

    def effective_model_context(self, _model):
        return SimpleNamespace(digest="sha256:entity-shadow-postgres-fixture")


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
            (
                project_id,
                json.dumps({
                    "entity_types": [],
                    **logical_schema(predicate_policy("fixture_rel")),
                }),
            ),
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


@pytest.mark.integration
def test_failed_entity_shadow_attempt_removes_only_its_partial_artifacts():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        project_id = int(conn.execute(
            """INSERT INTO projects(
               project_key,name,llm_model,embedding_model,active_schema_version)
               VALUES('entity-shadow-retry','Entity Shadow Retry','fake-llm','fake-embed',1)
               RETURNING id"""
        ).fetchone()["id"])
        first_document = _seed_document(conn, project_id, 1, "Shared entity appears first.")
        retry_document = _seed_document(conn, project_id, 2, "Shared and stale entities.")
        first_chunk = conn.execute(
            "SELECT * FROM chunks WHERE document_id=%s", (first_document,),
        ).fetchone()
        retry_chunk = conn.execute(
            "SELECT * FROM chunks WHERE document_id=%s", (retry_document,),
        ).fetchone()

    namespace_id = "entity-shadow-retry-namespace"
    database.create_entity_shadow_namespace(
        project_id, namespace_id, {"mode": "shadow"},
        [first_document, retry_document],
    )
    database.seed_entity_shadow_work_items(namespace_id, [{
        "id": "entity-shadow-retry-work",
        "project_id": project_id,
        "document_id": retry_document,
        "revision_id": retry_chunk["revision_id"],
        "chunk_id": retry_chunk["id"],
        "input_fingerprint": "retry-fingerprint",
    }])
    leased = database.claim_entity_shadow_work(namespace_id, "retry-test")
    assert leased is not None

    with database.connection() as conn:
        conn.execute(
            """INSERT INTO entity_shadow_mentions(
               namespace_id,mention_key,project_id,document_id,revision_id,chunk_id,
               surface,normalized_surface,evidence_text,evidence_start,evidence_end,
               confidence,extraction_policy_version)
               VALUES
               (%s,'mention-old',%s,%s,%s,%s,'Shared','shared','Shared',0,6,0.9,'v1'),
               (%s,'mention-retry',%s,%s,%s,%s,'Shared','shared','Shared',0,6,0.9,'v1'),
               (%s,'mention-stale',%s,%s,%s,%s,'Stale','stale','Stale',11,16,0.8,'v1')""",
            (
                namespace_id, project_id, first_document,
                first_chunk["revision_id"], first_chunk["id"],
                namespace_id, project_id, retry_document,
                retry_chunk["revision_id"], retry_chunk["id"],
                namespace_id, project_id, retry_document,
                retry_chunk["revision_id"], retry_chunk["id"],
            ),
        )
        conn.execute(
            """INSERT INTO entity_shadow_canonicals(
               namespace_id,canonical_key,name,normalized_name,primary_type,source_count)
               VALUES(%s,'canonical-shared','Shared','shared','Concept',2),
                     (%s,'canonical-orphan','Stale','stale','Concept',1)""",
            (namespace_id, namespace_id),
        )
        conn.execute(
            """INSERT INTO entity_shadow_assignments(
               namespace_id,mention_key,canonical_key,document_id,chunk_id,revision_id,
               entity_type,policy_version)
               VALUES
               (%s,'mention-old','canonical-shared',%s,%s,%s,'Concept','v1'),
               (%s,'mention-retry','canonical-shared',%s,%s,%s,'Concept','v1'),
               (%s,'mention-stale','canonical-orphan',%s,%s,%s,'Concept','v1')""",
            (
                namespace_id, first_document, first_chunk["id"], first_chunk["revision_id"],
                namespace_id, retry_document, retry_chunk["id"], retry_chunk["revision_id"],
                namespace_id, retry_document, retry_chunk["id"], retry_chunk["revision_id"],
            ),
        )
        conn.execute(
            """INSERT INTO entity_shadow_aliases(
               namespace_id,canonical_key,mention_key,alias,normalized_alias,
               alias_kind,confidence,policy_version)
               VALUES
               (%s,'canonical-shared','mention-old','Shared','shared','mention',0.9,'v1'),
               (%s,'canonical-shared','mention-retry','Shared','shared','mention',0.9,'v1'),
               (%s,'canonical-orphan','mention-stale','Stale','stale','mention',0.8,'v1')""",
            (namespace_id, namespace_id, namespace_id),
        )

    status = database.fail_entity_shadow_work(
        str(leased["id"]), str(leased["lease_token"]), "simulated retry",
    )

    assert status == "retryable_failed"
    with database.connection() as conn:
        mentions = conn.execute(
            """SELECT mention_key FROM entity_shadow_mentions
               WHERE namespace_id=%s ORDER BY mention_key""",
            (namespace_id,),
        ).fetchall()
        assignments = conn.execute(
            """SELECT mention_key,canonical_key FROM entity_shadow_assignments
               WHERE namespace_id=%s ORDER BY mention_key""",
            (namespace_id,),
        ).fetchall()
        canonicals = conn.execute(
            """SELECT canonical_key,source_count FROM entity_shadow_canonicals
               WHERE namespace_id=%s ORDER BY canonical_key""",
            (namespace_id,),
        ).fetchall()
    assert mentions == [{"mention_key": "mention-old"}]
    assert assignments == [{
        "mention_key": "mention-old",
        "canonical_key": "canonical-shared",
    }]
    assert canonicals == [{
        "canonical_key": "canonical-shared",
        "source_count": 1,
    }]
