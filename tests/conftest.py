"""Isolation fixtures shared by the real PostgreSQL integration tests."""

from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg import sql


@pytest.fixture(autouse=True)
def _reset_runner_owned_postgres_schema(request):
    """Reset only the sentinel-bound schema provisioned by the E2E runner.

    The restricted test role deliberately cannot create arbitrary schemas.
    Reusing its one random schema is safe only when every integration test
    starts from empty product tables, while the admin-owned sentinel remains
    intact for final cleanup verification.
    """

    if request.node.get_closest_marker("integration") is None:
        yield
        return
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN", "").strip()
    schema = os.getenv("KNOWLEDGE_FORGE_TEST_SCHEMA", "").strip()
    if not dsn or not schema.startswith("kf_e2e_"):
        yield
        return

    with psycopg.connect(dsn, autocommit=True) as connection:
        preserved = {"kf_e2e_sentinel", "schema_migrations"}
        tables = [
            str(row[0]) for row in connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname=current_schema()"
            ).fetchall()
            if str(row[0]) not in preserved
        ]
        if tables:
            connection.execute(
                sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                    sql.SQL(",").join(sql.Identifier(name) for name in tables)
                )
            )
    yield


@pytest.fixture
def active_document_revision():
    """Create the explicit active revision required by the migrated write guards."""

    def create(connection, project_id: int, document_id: int) -> str:
        revision_id = f"test-revision-{document_id}-{uuid.uuid4().hex[:12]}"
        ingestion_run_id = f"test-run-{document_id}-{uuid.uuid4().hex[:12]}"
        connection.execute(
            """INSERT INTO document_content_revisions(
               revision_id,project_id,document_id,ingestion_run_id,content_version,
               content_hash,source_hash,config_hash,policy_hash,manifest_hash,manifest,
               status,activated_at)
               VALUES(%s,%s,%s,%s,1,'test-content','test-source','test-config',
                      'test-policy','test-manifest','{}'::jsonb,'active',NOW())""",
            (revision_id, project_id, document_id, ingestion_run_id),
        )
        connection.execute(
            "UPDATE documents SET active_revision_id=%s,content_version=1 WHERE id=%s",
            (revision_id, document_id),
        )
        return revision_id

    return create
