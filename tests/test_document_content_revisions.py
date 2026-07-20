from __future__ import annotations

from contextlib import contextmanager
import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from knowledge_forge.database import Database, SCHEMA_SQL
from scripts.postgres_dependency_graph import collect_dependency_graph


def _content(version: int):
    return ([{
        "key": "section", "title": "Section", "level": 1,
        "content": f"content {version}", "content_hash": f"section-{version}",
    }], [{
        "key": "chunk", "section_key": "section", "content": f"content {version}",
        "content_hash": f"chunk-{version}", "part_index": 1,
    }])


def test_current_reingestion_path_is_destructive_characterization():
    source = __import__("inspect").getsource(Database.replace_document_content)
    assert "DELETE FROM chunks WHERE document_id=%s" in source
    assert "DELETE FROM relation_assertions WHERE document_id=%s" in source
    assert "document_content_revisions" not in SCHEMA_SQL


@pytest.mark.integration
def test_current_reingestion_is_blocked_by_immutable_shadow_fk_characterization():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    schema_name = f"kf_readiness_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        try:
            admin.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema_name)))
            admin.execute(SCHEMA_SQL)
            project_id = int(admin.execute(
                "INSERT INTO projects(project_key,name) VALUES('readiness','Readiness') RETURNING id"
            ).fetchone()["id"])
            document_id = int(admin.execute(
                """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
                   storage_path,local_path,file_size_bytes)
                   VALUES(%s,'hash','a.txt','a.txt','/a','/a',1) RETURNING id""",
                (project_id,),
            ).fetchone()["id"])

            @contextmanager
            def connection():
                with psycopg.connect(dsn, row_factory=dict_row) as scoped:
                    scoped.execute(
                        sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema_name))
                    )
                    yield scoped

            database = Database()
            database.connection = connection
            sections, chunks = _content(1)
            inserted = database.replace_document_content(document_id, sections, chunks)
            chunk_id = int(inserted[0]["id"])
            admin.execute(
                """INSERT INTO relation_shadow_runs(
                       id,project_id,document_id,chunk_id,control_policy_hash,
                       treatment_policy_hash,policy_set)
                   VALUES('audit-run',%s,%s,%s,'control','treatment','{}')""",
                (project_id, document_id, chunk_id),
            )
            edges = collect_dependency_graph(admin)
            assert any(
                edge["referencing_table"] == "relation_shadow_runs"
                and edge["referenced_table"] == "chunks"
                and edge["on_delete"] in {"NO ACTION", "RESTRICT"}
                for edge in edges
            )

            for version in (2, 3):
                with pytest.raises(psycopg.errors.ForeignKeyViolation):
                    next_sections, next_chunks = _content(version)
                    database.replace_document_content(document_id, next_sections, next_chunks)
            assert admin.execute(
                "SELECT COUNT(*) AS count FROM relation_shadow_runs WHERE id='audit-run'"
            ).fetchone()["count"] == 1
            assert admin.execute(
                "SELECT content_hash FROM chunks WHERE id=%s", (chunk_id,)
            ).fetchone()["content_hash"] == "chunk-1"
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))
