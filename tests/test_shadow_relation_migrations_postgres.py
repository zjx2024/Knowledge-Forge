from __future__ import annotations

import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from knowledge_forge.database import (
    Database,
    SCHEMA_SQL,
    SHADOW_RELATION_V2_REQUIRED_MIGRATIONS,
    SHADOW_RELATION_V2_REQUIRED_TABLE_COLUMNS,
)
from knowledge_forge.migrations import MIGRATIONS, MigrationRunner


V2_INDEXES = {
    "idx_shadow_relation_v2_runs_project",
    "idx_shadow_relation_pairs_run_status",
    "idx_shadow_relation_pairs_cross_policy",
    "idx_shadow_relation_hypotheses_pair",
    "idx_shadow_relation_verification_verdict",
    "idx_shadow_relation_facts_endpoints",
    "idx_shadow_relation_evidence_run",
    "idx_shadow_relation_classification_attempts_pair",
}

V2_UNIQUE_COLUMN_SETS = {
    "shadow_relation_pair_opportunities": {
        ("run_id", "candidate_policy_version", "pair_identity"),
    },
    "shadow_relation_hypotheses": {
        (
            "pair_opportunity_id",
            "subject_occurrence_key",
            "predicate",
            "object_occurrence_key",
            "hypothesis_key",
        ),
    },
    "shadow_relation_verification_attempts": {
        ("hypothesis_id", "attempt_no"),
    },
    "shadow_relation_normalized_facts": {
        ("project_id", "semantic_fact_key"),
    },
    "shadow_relation_fact_evidence": {
        ("run_id", "fact_id", "evidence_key"),
    },
    "shadow_relation_classification_attempts": {
        ("pair_opportunity_id", "attempt_no"),
    },
}

V2_FOREIGN_KEY_MINIMUMS = {
    "shadow_relation_v2_runs": 1,
    "shadow_relation_pair_opportunities": 2,
    "shadow_relation_hypotheses": 1,
    "shadow_relation_verification_attempts": 1,
    "shadow_relation_normalized_facts": 1,
    "shadow_relation_fact_memberships": 2,
    "shadow_relation_fact_evidence": 4,
    "shadow_relation_fact_collapse_audit": 3,
    "shadow_relation_fact_conflicts": 1,
    "shadow_relation_classification_attempts": 1,
    "shadow_relation_hypothesis_validations": 1,
}


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _isolated_admin_dsn() -> str:
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_ADMIN_DSN", "").strip()
    run_id = os.getenv("KNOWLEDGE_FORGE_TEST_RUN_ID", "").strip()
    if not dsn or not run_id:
        pytest.skip("sentinel-bound disposable PostgreSQL admin DSN is absent")
    values = conninfo_to_dict(dsn)
    if values.get("dbname") != f"kf_e2e_db_{run_id}":
        raise AssertionError("migration test admin DSN is not the disposable database")
    return dsn


def _schema_dsn(admin_dsn: str, schema_name: str) -> str:
    values = conninfo_to_dict(admin_dsn)
    values["options"] = (
        f"-c search_path={schema_name},public -c statement_timeout=120000"
    )
    return make_conninfo(**values)


def _assert_v2_catalog(conn) -> dict[str, object]:
    migration_rows = conn.execute(
        """SELECT version,checksum,status
           FROM schema_migrations
           WHERE version=ANY(%s)
           ORDER BY version""",
        (list(SHADOW_RELATION_V2_REQUIRED_MIGRATIONS),),
    ).fetchall()
    expected_checksums = {
        item.version: item.checksum
        for item in MIGRATIONS
        if item.version in SHADOW_RELATION_V2_REQUIRED_MIGRATIONS
    }
    assert [str(row["version"]) for row in migration_rows] == list(
        SHADOW_RELATION_V2_REQUIRED_MIGRATIONS
    )
    assert all(row["status"] == "applied" for row in migration_rows)
    assert {
        str(row["version"]): str(row["checksum"]) for row in migration_rows
    } == expected_checksums

    actual_columns = {
        str(row["table_name"]): set(row["columns"])
        for row in conn.execute(
            """SELECT table_name,
                      array_agg(column_name::text ORDER BY ordinal_position) AS columns
               FROM information_schema.columns
               WHERE table_schema=current_schema() AND table_name=ANY(%s)
               GROUP BY table_name""",
            (list(SHADOW_RELATION_V2_REQUIRED_TABLE_COLUMNS),),
        ).fetchall()
    }
    assert set(actual_columns) == set(SHADOW_RELATION_V2_REQUIRED_TABLE_COLUMNS)
    for table, required in SHADOW_RELATION_V2_REQUIRED_TABLE_COLUMNS.items():
        assert required <= actual_columns[table]

    indexes = {
        str(row["indexname"])
        for row in conn.execute(
            """SELECT indexname FROM pg_indexes
               WHERE schemaname=current_schema()"""
        ).fetchall()
    }
    assert V2_INDEXES <= indexes

    unique_sets: dict[str, set[tuple[str, ...]]] = {}
    rows = conn.execute(
        """SELECT table_name,constraint_name
           FROM information_schema.table_constraints
           WHERE constraint_schema=current_schema()
             AND constraint_type IN ('UNIQUE','PRIMARY KEY')
             AND table_name=ANY(%s)""",
        (list(V2_UNIQUE_COLUMN_SETS),),
    ).fetchall()
    for row in rows:
        columns = tuple(
            str(item["column_name"])
            for item in conn.execute(
                """SELECT column_name
                   FROM information_schema.key_column_usage
                   WHERE constraint_schema=current_schema()
                     AND table_name=%s AND constraint_name=%s
                   ORDER BY ordinal_position""",
                (row["table_name"], row["constraint_name"]),
            ).fetchall()
        )
        unique_sets.setdefault(str(row["table_name"]), set()).add(columns)
    for table, expected in V2_UNIQUE_COLUMN_SETS.items():
        assert expected <= unique_sets.get(table, set())

    foreign_keys = {
        str(row["table_name"]): int(row["count"])
        for row in conn.execute(
            """SELECT table_name,COUNT(*) AS count
               FROM information_schema.table_constraints
               WHERE constraint_schema=current_schema()
                 AND constraint_type='FOREIGN KEY'
                 AND table_name=ANY(%s)
               GROUP BY table_name""",
            (list(V2_FOREIGN_KEY_MINIMUMS),),
        ).fetchall()
    }
    for table, minimum in V2_FOREIGN_KEY_MINIMUMS.items():
        assert foreign_keys.get(table, 0) >= minimum

    readiness = {
        "migration_count": len(migration_rows),
        "table_count": len(actual_columns),
        "index_count": len(V2_INDEXES),
        "foreign_key_count": sum(foreign_keys.values()),
        "unique_contract_count": sum(
            len(value) for value in V2_UNIQUE_COLUMN_SETS.values()
        ),
    }
    return readiness


@pytest.mark.integration
def test_empty_isolated_database_applies_complete_v2_migration_chain():
    admin_dsn = _isolated_admin_dsn()
    schema_name = "shadow_v2_empty_" + uuid.uuid4().hex
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
    try:
        database = _database(_schema_dsn(admin_dsn, schema_name))
        with database.connection() as conn:
            existing = conn.execute(
                """SELECT COUNT(*) AS count FROM information_schema.tables
                   WHERE table_schema=current_schema()"""
            ).fetchone()
            assert int(existing["count"]) == 0
        database.bootstrap()
        with database.connection() as conn:
            evidence = _assert_v2_catalog(conn)
            assert evidence["migration_count"] == 6
        assert database.assert_shadow_relation_v2_ready()["migrations"] == list(
            SHADOW_RELATION_V2_REQUIRED_MIGRATIONS
        )
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )


@pytest.mark.integration
def test_isolated_0013_to_0019_compatibility_upgrade_and_catalog_contract():
    dsn = _isolated_admin_dsn()
    schema_name = "shadow_v2_upgrade_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            conn.execute(
                sql.SQL("SET search_path TO {},public").format(
                    sql.Identifier(schema_name)
                )
            )
            conn.execute(SCHEMA_SQL)
            MigrationRunner(MIGRATIONS[:13]).apply(conn)
            applied = conn.execute(
                """SELECT version FROM schema_migrations
                   WHERE status='applied' ORDER BY version"""
            ).fetchall()
            assert [row["version"] for row in applied] == [
                item.version for item in MIGRATIONS[:13]
            ]
            assert conn.execute(
                """SELECT to_regclass(
                     current_schema()||'.shadow_relation_v2_runs'
                   ) AS name"""
            ).fetchone()["name"] is None

            MigrationRunner().apply(conn)
            evidence = _assert_v2_catalog(conn)
            assert evidence == {
                "migration_count": 6,
                "table_count": len(SHADOW_RELATION_V2_REQUIRED_TABLE_COLUMNS),
                "index_count": 8,
                "foreign_key_count": sum(
                    int(row["count"])
                    for row in conn.execute(
                        """SELECT COUNT(*) AS count
                           FROM information_schema.table_constraints
                           WHERE constraint_schema=current_schema()
                             AND constraint_type='FOREIGN KEY'
                             AND table_name=ANY(%s)
                           GROUP BY table_name""",
                        (list(V2_FOREIGN_KEY_MINIMUMS),),
                    ).fetchall()
                ),
                "unique_contract_count": 6,
            }
            assert evidence["foreign_key_count"] >= sum(
                V2_FOREIGN_KEY_MINIMUMS.values()
            )
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(
                    sql.Identifier(schema_name)
                )
            )
