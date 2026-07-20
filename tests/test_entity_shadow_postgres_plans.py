from __future__ import annotations

import os
import json

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge.database import Database


def _nodes(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from _nodes(child)


def _explain(conn, statement: str, params=()):
    row = conn.execute(
        "EXPLAIN (ANALYZE,BUFFERS,FORMAT JSON) " + statement,
        params,
    ).fetchone()
    payload = next(iter(row.values())) if isinstance(row, dict) else row[0]
    return payload[0]["Plan"]


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


@pytest.mark.integration
def test_shadow_10k_and_100k_candidates_use_trigram_exact_and_hnsw_indexes():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES('shadow-plan','Shadow Plan') RETURNING id"
        ).fetchone()["id"])
        document_id = int(conn.execute(
            """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
               storage_path,local_path,file_size_bytes,status)
               VALUES(%s,'shadow-plan-doc','plan.txt','plan.txt','/plan','/plan',1,'indexed')
               RETURNING id""",
            (project_id,),
        ).fetchone()["id"])
        revision_id = f"shadow-plan-revision-{document_id}"
        conn.execute(
            """INSERT INTO document_content_revisions(
               revision_id,project_id,document_id,ingestion_run_id,content_version,
               content_hash,source_hash,config_hash,policy_hash,manifest_hash,manifest,
               status,activated_at)
               VALUES(%s,%s,%s,'shadow-plan-run',1,'content','source','config','policy',
                      'manifest','{}'::jsonb,'active',NOW())""",
            (revision_id, project_id, document_id),
        )
        conn.execute(
            "UPDATE documents SET active_revision_id=%s WHERE id=%s",
            (revision_id, document_id),
        )
        section_id = int(conn.execute(
            """INSERT INTO sections(project_id,document_id,revision_id,section_key,title,
               level,content,content_hash)
               VALUES(%s,%s,%s,'plan-section','Plan',1,'Acme','section') RETURNING id""",
            (project_id, document_id, revision_id),
        ).fetchone()["id"])
        chunk_id = int(conn.execute(
            """INSERT INTO chunks(project_id,document_id,revision_id,section_id,chunk_key,
               chunk_index,content,content_hash)
               VALUES(%s,%s,%s,%s,'plan-chunk',0,'Acme','chunk') RETURNING id""",
            (project_id, document_id, revision_id, section_id),
        ).fetchone()["id"])
        namespace_id = "shadow-plan-namespace"
        conn.execute(
            """INSERT INTO entity_shadow_namespaces(id,project_id,policy_set,status)
               VALUES(%s,%s,'{}'::jsonb,'building')""",
            (namespace_id, project_id),
        )
        conn.execute(
            """INSERT INTO entity_shadow_mentions(
               namespace_id,mention_key,project_id,document_id,revision_id,chunk_id,
               surface,normalized_surface,evidence_text,evidence_start,evidence_end,
               confidence,extraction_policy_version)
               VALUES(%s,'plan-mention',%s,%s,%s,%s,'Acme','acme','Acme',0,4,1,'plan')""",
            (namespace_id, project_id, document_id, revision_id, chunk_id),
        )

        for target in (10_000, 100_000):
            existing = int(conn.execute(
                "SELECT COUNT(*) AS count FROM entity_shadow_canonicals WHERE namespace_id=%s",
                (namespace_id,),
            ).fetchone()["count"])
            conn.execute(
                """INSERT INTO entity_shadow_canonicals(
                   namespace_id,canonical_key,name,normalized_name,primary_type)
                   SELECT %s,'canonical-'||value,'Entity '||value,'entity'||value,
                          'GenericEntity'
                   FROM generate_series(%s::bigint,%s::bigint) value""",
                (namespace_id, existing + 1, target),
            )
            conn.execute(
                """INSERT INTO entity_shadow_aliases(
                   namespace_id,canonical_key,mention_key,alias,normalized_alias,
                   alias_kind,confidence,policy_version)
                   SELECT namespace_id,canonical_key,'plan-mention',name,normalized_name,
                          'canonical_name',1,'plan'
                   FROM entity_shadow_canonicals WHERE namespace_id=%s
                   ON CONFLICT DO NOTHING""",
                (namespace_id,),
            )
            rare_alias = f"rarealiastarget{target}"
            conn.execute(
                """INSERT INTO entity_shadow_aliases(
                   namespace_id,canonical_key,mention_key,alias,normalized_alias,
                   alias_kind,confidence,policy_version)
                   VALUES(%s,%s,'plan-mention',%s,%s,'observed',1,'plan')
                   ON CONFLICT DO NOTHING""",
                (namespace_id, f"canonical-{target}", rare_alias, rare_alias),
            )
            conn.execute("ANALYZE entity_shadow_canonicals")
            conn.execute("ANALYZE entity_shadow_aliases")
            rare = f"entity{target}"
            exact = _explain(
                conn,
                """SELECT canonical.canonical_key FROM entity_shadow_aliases alias
                   JOIN entity_shadow_canonicals canonical
                     ON canonical.namespace_id=alias.namespace_id
                    AND canonical.canonical_key=alias.canonical_key
                   WHERE alias.namespace_id=%s AND alias.normalized_alias=%s
                   ORDER BY canonical.canonical_key LIMIT 10""",
                (namespace_id, rare),
            )
            assert any(
                "idx_entity_shadow_alias_exact" in str(node.get("Index Name") or "")
                for node in _nodes(exact)
            )
            assert int(exact.get("Actual Rows") or 0) <= 10
            if target == 100_000:
                fuzzy_name = rare_alias + "x"
                fuzzy = _explain(
                    conn,
                    """SELECT canonical.canonical_key,
                       MAX(similarity(alias.normalized_alias,%s)) AS lexical_score
                       FROM entity_shadow_aliases alias
                       JOIN entity_shadow_canonicals canonical
                         ON canonical.namespace_id=alias.namespace_id
                        AND canonical.canonical_key=alias.canonical_key
                       WHERE alias.namespace_id=%s AND alias.normalized_alias %% %s
                       GROUP BY canonical.namespace_id,canonical.canonical_key
                       ORDER BY lexical_score DESC,canonical.canonical_key LIMIT 20""",
                    (fuzzy_name, namespace_id, fuzzy_name),
                )
                assert any(
                    "idx_entity_shadow_alias_trgm" in str(node.get("Index Name") or "")
                    for node in _nodes(fuzzy)
                ), json.dumps(fuzzy, indent=2)
                assert int(fuzzy.get("Actual Rows") or 0) <= 20

        conn.execute(
            """INSERT INTO entity_shadow_embeddings(
               namespace_id,canonical_key,model_name,embedding,embedding_vector,
               dimension,content_hash,policy_version)
               SELECT namespace_id,canonical_key,'plan-embed',
                      ARRAY[1,0,0,0,0,0,0,(substring(canonical_key from '[0-9]+')::integer%%97)::double precision/97],
                      ('[1,0,0,0,0,0,0,'||
                       ((substring(canonical_key from '[0-9]+')::integer%%97)::double precision/97)||']')::vector,
                      8,canonical_key,'plan'
               FROM entity_shadow_canonicals WHERE namespace_id=%s""",
            (namespace_id,),
        )
        database._ensure_entity_shadow_hnsw_index(conn, "plan-embed", 8)
        conn.execute("ANALYZE entity_shadow_embeddings")
        vector = _explain(
            conn,
            """SELECT canonical.canonical_key FROM entity_shadow_embeddings embedding
               JOIN entity_shadow_canonicals canonical
                 ON canonical.namespace_id=embedding.namespace_id
                AND canonical.canonical_key=embedding.canonical_key
               WHERE embedding.namespace_id=%s AND embedding.model_name='plan-embed'
                 AND embedding.dimension=8
               ORDER BY embedding.embedding_vector::vector(8)
                 <=> '[1,0,0,0,0,0,0,0.5]'::vector LIMIT 30""",
            (namespace_id,),
        )
        assert any(
            "idx_entity_shadow_hnsw_" in str(node.get("Index Name") or "")
            for node in _nodes(vector)
        )
        assert int(vector.get("Actual Rows") or 0) <= 30
