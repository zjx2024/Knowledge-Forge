from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from knowledge_forge.database import SCHEMA_SQL


def _nodes(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from _nodes(child)


def _explain(conn, statement: str, params=()):
    row = conn.execute(
        "EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON) " + statement, params,
    ).fetchone()
    payload = next(iter(row.values())) if isinstance(row, dict) else row[0]
    return payload[0]["Plan"]


def _assert_selective_index_plan(plan, *, expected_fragment: str = ""):
    nodes = list(_nodes(plan))
    scans = [node for node in nodes if "Scan" in str(node.get("Node Type") or "")]
    assert scans
    assert not any(node.get("Node Type") == "Seq Scan" for node in scans)
    if expected_fragment:
        assert any(expected_fragment in str(node.get("Index Name") or "") for node in scans)
    assert int(plan.get("Actual Rows") or 0) <= 50


@pytest.mark.integration
def test_postgres_10k_and_100k_candidate_plans_use_bounded_indexes():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    schema_name = f"kf_plans_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        try:
            conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema_name)))
            conn.execute(SCHEMA_SQL)
            project_id = conn.execute(
                "INSERT INTO projects(project_key,name) VALUES('plans','Plans') RETURNING id"
            ).fetchone()["id"]
            document_id = conn.execute(
                """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
                   storage_path,local_path,file_size_bytes)
                   VALUES(%s,'plans-doc','plans.txt','plans.txt','/plans','/plans',1) RETURNING id""",
                (project_id,),
            ).fetchone()["id"]
            for target in (10_000, 100_000):
                existing = conn.execute(
                    "SELECT COUNT(*) AS count FROM canonical_entities WHERE project_id=%s",
                    (project_id,),
                ).fetchone()["count"]
                conn.execute(
                    """INSERT INTO canonical_entities(
                       project_id,entity_key,name,normalized_name,entity_type,source_count)
                       SELECT %s,'entity-'||value,'Entity '||value,'entity'||value,
                              'GenericEntity',1
                       FROM generate_series(%s::bigint,%s::bigint) value""",
                    (project_id, existing + 1, target),
                )
                conn.execute(
                    """INSERT INTO canonical_entity_aliases(
                       project_id,canonical_entity_id,entity_type,alias,normalized_alias,
                       alias_kind,policy_version)
                       SELECT project_id,id,entity_type,name,normalized_name,'canonical_name','plan-test'
                       FROM canonical_entities WHERE project_id=%s
                       ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind) DO NOTHING""",
                    (project_id,),
                )
                conn.execute(
                    """INSERT INTO canonical_entity_documents(project_id,canonical_entity_id,document_id)
                       SELECT project_id,id,%s FROM canonical_entities WHERE project_id=%s
                       ON CONFLICT(canonical_entity_id,document_id) DO NOTHING""",
                    (document_id, project_id),
                )
                rare_alias = f"rarealiastarget{target}"
                conn.execute(
                    """INSERT INTO canonical_entity_aliases(
                       project_id,canonical_entity_id,entity_type,alias,normalized_alias,
                       alias_kind,policy_version)
                       SELECT project_id,id,entity_type,%s,%s,'observed','plan-test'
                       FROM canonical_entities WHERE project_id=%s AND normalized_name=%s
                       ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind) DO NOTHING""",
                    (rare_alias, rare_alias, project_id, f"entity{target}"),
                )
                conn.execute("ANALYZE canonical_entities")
                conn.execute("ANALYZE canonical_entity_aliases")
                conn.execute("ANALYZE canonical_entity_documents")

                exact_name = _explain(
                    conn,
                    """SELECT id FROM canonical_entities WHERE project_id=%s
                       AND normalized_name=%s AND is_deleted=FALSE LIMIT 10""",
                    (project_id, f"entity{target}"),
                )
                exact_alias = _explain(
                    conn,
                    """SELECT canonical_entity_id FROM canonical_entity_aliases
                       WHERE project_id=%s AND entity_type='GenericEntity'
                         AND normalized_alias=%s AND active=TRUE LIMIT 10""",
                    (project_id, f"entity{target}"),
                )
                fuzzy_alias = _explain(
                    conn,
                    """SELECT canonical_entity_id FROM canonical_entity_aliases
                       WHERE project_id=%s AND normalized_alias ILIKE %s AND active=TRUE
                       ORDER BY similarity(normalized_alias,%s) DESC LIMIT 10""",
                    (project_id, f"%{rare_alias}%", rare_alias),
                )
                sources = _explain(
                    conn,
                    """SELECT COUNT(*) FROM canonical_entity_documents
                       WHERE project_id=%s AND canonical_entity_id=(
                         SELECT id FROM canonical_entities WHERE project_id=%s
                         AND normalized_name=%s AND is_deleted=FALSE)
                       AND is_deleted=FALSE""",
                    (project_id, project_id, f"entity{target}"),
                )
                _assert_selective_index_plan(exact_name, expected_fragment="idx_entities_normalized")
                _assert_selective_index_plan(exact_alias, expected_fragment="idx_entity_alias_exact")
                if target >= 100_000:
                    _assert_selective_index_plan(
                        fuzzy_alias, expected_fragment="idx_entity_alias_trgm",
                    )
                else:
                    assert int(fuzzy_alias.get("Actual Rows") or 0) <= 10
                assert not any(node.get("Node Type") == "Seq Scan" for node in _nodes(sources))
        finally:
            conn.execute("SET search_path TO public")
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))


@pytest.mark.integration
def test_postgres_hnsw_entity_and_relation_top_k_plan():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    schema_name = f"kf_hnsw_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        try:
            conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema_name)))
            conn.execute(SCHEMA_SQL)
            project_id = conn.execute(
                "INSERT INTO projects(project_key,name) VALUES('vectors','Vectors') RETURNING id"
            ).fetchone()["id"]
            conn.execute(
                """INSERT INTO vector_embeddings(
                   project_id,object_type,object_key,content,content_hash,model_name,
                   embedding,embedding_vector,dimension)
                   SELECT %s,CASE WHEN value%%2=0 THEN 'entity' ELSE 'relation' END,
                          value::text,'item '||value,value::text,'plan-embed',
                          ARRAY[1,0,0,0,0,0,0,(value%%97)::double precision/97],
                          ('[1,0,0,0,0,0,0,'||((value%%97)::double precision/97)||']')::vector,8
                   FROM generate_series(1,100000) value""",
                (project_id,),
            )
            conn.execute(
                """CREATE INDEX plan_vectors_hnsw ON vector_embeddings
                   USING hnsw ((embedding_vector::vector(8)) vector_cosine_ops)
                   WHERE model_name='plan-embed' AND dimension=8 AND is_deleted=FALSE"""
            )
            conn.execute("ANALYZE vector_embeddings")
            for object_type in ("entity", "relation"):
                plan = _explain(
                    conn,
                    """SELECT id FROM vector_embeddings WHERE project_id=%s
                       AND object_type=%s AND model_name='plan-embed' AND dimension=8
                       AND is_deleted=FALSE ORDER BY embedding_vector::vector(8)
                       <=> '[1,0,0,0,0,0,0,0.5]'::vector LIMIT 10""",
                    (project_id, object_type),
                )
                assert any("plan_vectors_hnsw" in str(node.get("Index Name") or "")
                           for node in _nodes(plan))
                assert int(plan.get("Actual Rows") or 0) <= 10
        finally:
            conn.execute("SET search_path TO public")
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))
