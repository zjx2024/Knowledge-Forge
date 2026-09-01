from __future__ import annotations

import os

import psycopg
import pytest

from knowledge_forge.postgres_test_safety import (
    PostgresIsolationError,
    canonical_postgres_identity,
    validate_isolated_identity,
)


def test_postgres_identity_never_contains_password():
    identity = canonical_postgres_identity(
        host="127.0.0.1", port=55432, database="fixtures", user="runner",
        schema="kf_e2e_0123456789abcdef",
    )
    assert identity == "postgresql://runner@127.0.0.1:55432/fixtures/kf_e2e_0123456789abcdef"
    assert "password" not in identity


def test_postgres_test_identity_requires_random_schema_and_known_distinct_production():
    test = "postgresql://runner@127.0.0.1:5432/db/kf_e2e_0123456789abcdef"
    production = "postgresql://app@127.0.0.1:5432/db/public"
    validate_isolated_identity(test, production, run_id="0123456789abcdef")
    with pytest.raises(PostgresIsolationError, match="production"):
        validate_isolated_identity(test, "", run_id="0123456789abcdef")
    with pytest.raises(PostgresIsolationError, match="equal"):
        validate_isolated_identity(test, test, run_id="0123456789abcdef")
    with pytest.raises(PostgresIsolationError, match="schema"):
        validate_isolated_identity(
            "postgresql://runner@127.0.0.1:5432/db/public", production,
            run_id="0123456789abcdef",
        )


@pytest.mark.integration
def test_disposable_postgres_role_is_schema_bounded_and_sentinel_is_admin_owned():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    expected_schema = os.getenv("KNOWLEDGE_FORGE_TEST_SCHEMA", "")
    expected_run_id = os.getenv("KNOWLEDGE_FORGE_TEST_RUN_ID", "")
    with psycopg.connect(dsn) as connection:
        row = connection.execute(
            "SELECT current_user, current_schema(), "
            "has_database_privilege(current_user, current_database(), 'CREATE'), "
            "has_schema_privilege(current_user, 'public', 'CREATE')"
        ).fetchone()
        assert row[1] == expected_schema
        assert row[2] is False
        assert row[3] is False
        checkpoint_schema = connection.execute(
            "SELECT pg_get_userbyid(nspowner) FROM pg_namespace "
            "WHERE nspname='knowledge_agent'"
        ).fetchone()
        assert checkpoint_schema == (row[0],)
        sentinel = connection.execute(
            "SELECT run_id, schema_name, role_name, config_hash, expires_at "
            "FROM kf_e2e_sentinel"
        ).fetchone()
        assert sentinel[0] == expected_run_id
        assert sentinel[1] == expected_schema
        assert sentinel[2] == row[0]
        owner = connection.execute(
            "SELECT pg_get_userbyid(c.relowner) FROM pg_class c "
            "JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname='kf_e2e_sentinel'"
        ).fetchone()[0]
        assert owner != row[0]
