"""Static contracts for the additive GraphRAG Native migration."""

from __future__ import annotations

import hashlib
import json
import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

from knowledge_forge.database import (
    Database,
    GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS,
    GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS,
    SCHEMA_SQL,
)
from knowledge_forge.migrations import MIGRATIONS, MigrationRunner


NATIVE_INDEXES = {
    "idx_graphrag_native_runs_project_state",
    "idx_graphrag_native_runs_lease",
    "idx_graphrag_native_run_documents_project",
    "idx_graphrag_native_artifacts_run",
    "idx_graphrag_native_project_state_active",
    "idx_graphrag_native_queries_project_run",
}

NATIVE_MINIMUM_FOREIGN_KEYS = {
    "graphrag_native_runs": 2,
    "graphrag_native_run_documents": 3,
    "graphrag_native_artifacts": 1,
    "graphrag_native_project_state": 2,
    "graphrag_native_query_runs": 2,
}


def _isolated_admin_dsn() -> str:
    """Use only the runner-created disposable database, never production."""

    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_ADMIN_DSN", "").strip()
    run_id = os.getenv("KNOWLEDGE_FORGE_TEST_RUN_ID", "").strip()
    if not dsn or not run_id:
        pytest.skip("sentinel-bound disposable PostgreSQL admin DSN is absent")
    values = conninfo_to_dict(dsn)
    if values.get("dbname") != f"kf_e2e_db_{run_id}":
        raise AssertionError("native migration test admin DSN is not the disposable database")
    return dsn


def _schema_dsn(admin_dsn: str, schema_name: str) -> str:
    values = conninfo_to_dict(admin_dsn)
    values["options"] = f"-c search_path={schema_name},public -c statement_timeout=120000"
    return make_conninfo(**values)


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _catalog_evidence(conn) -> dict[str, object]:
    expected_versions = (
        *GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS,
        "0023_project_construction_strategy_selection",
        "0024_project_strategy_binding",
    )
    migration_rows = conn.execute(
        """SELECT version,checksum,status FROM schema_migrations
           WHERE version=ANY(%s) ORDER BY version""",
        (list(expected_versions),),
    ).fetchall()
    expected_checksums = {
        item.version: item.checksum for item in MIGRATIONS if item.version in expected_versions
    }
    assert [str(row["version"]) for row in migration_rows] == list(expected_versions)
    assert all(row["status"] == "applied" for row in migration_rows)
    assert {str(row["version"]): str(row["checksum"]) for row in migration_rows} == expected_checksums

    columns = {
        str(row["table_name"]): set(row["columns"])
        for row in conn.execute(
            """SELECT table_name,array_agg(column_name::text ORDER BY ordinal_position) AS columns
               FROM information_schema.columns
               WHERE table_schema=current_schema() AND table_name=ANY(%s)
               GROUP BY table_name""",
            (list(GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS),),
        ).fetchall()
    }
    assert set(columns) == set(GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS)
    for table, required_columns in GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS.items():
        assert required_columns <= columns[table]

    indexes = {
        str(row["indexname"])
        for row in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname=current_schema()"
        ).fetchall()
    }
    assert NATIVE_INDEXES <= indexes
    foreign_keys = {
        str(row["table_name"]): int(row["count"])
        for row in conn.execute(
            """SELECT table_name,COUNT(*) AS count
               FROM information_schema.table_constraints
               WHERE constraint_schema=current_schema() AND constraint_type='FOREIGN KEY'
                 AND table_name=ANY(%s) GROUP BY table_name""",
            (list(NATIVE_MINIMUM_FOREIGN_KEYS),),
        ).fetchall()
    }
    for table, minimum in NATIVE_MINIMUM_FOREIGN_KEYS.items():
        assert foreign_keys.get(table, 0) >= minimum
    return {
        "migration_count": len(migration_rows), "table_count": len(columns),
        "index_count": len(NATIVE_INDEXES), "foreign_key_count": sum(foreign_keys.values()),
    }


def _legacy_strategy_fingerprint(conn, project_id: int) -> str:
    rows = conn.execute(
        """SELECT strategy_key,namespace_id,generation,ready_generation,status,policy_summary
           FROM project_strategy_graphs WHERE project_id=%s ORDER BY strategy_key""",
        (project_id,),
    ).fetchall()
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_native_control_plane_is_one_new_immutable_migration():
    assert GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS == (
        "0022_graphrag_native_control_plane",
    )
    migration = next(
        item for item in MIGRATIONS if item.version == GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS[0]
    )
    assert "graphrag_native_runs" in migration.sql
    assert "graphrag_native_project_state" in migration.sql
    assert "canonical_entities" not in migration.sql
    assert "relation_assertions" not in migration.sql
    assert "vector_embeddings" not in migration.sql
    assert "project_strategy_graphs" not in migration.sql


def test_native_control_plane_contract_includes_only_control_metadata_tables():
    assert set(GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS) == {
        "graphrag_native_runs",
        "graphrag_native_run_documents",
        "graphrag_native_artifacts",
        "graphrag_native_project_state",
        "graphrag_native_query_runs",
    }
    assert {"source_snapshot_hash", "validation_report_hash", "lease_owner"} <= (
        GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS["graphrag_native_runs"]
    )


def test_project_selection_pointer_is_a_separate_additive_migration():
    migration = next(item for item in MIGRATIONS if item.version == "0023_project_construction_strategy_selection")
    assert "project_construction_strategy_state" in migration.sql
    assert "graphrag_native" in migration.sql
    assert "graphrag_native_runs" not in migration.sql
    assert "project_strategy_graphs" not in migration.sql
    binding = next(item for item in MIGRATIONS if item.version == "0024_project_strategy_binding")
    assert binding.startup_safe is True
    assert "association_status" in binding.sql
    assert "bound_at" in binding.sql
    assert "graphrag_native_runs" not in binding.sql


@pytest.mark.integration
def test_native_migrations_install_on_an_empty_isolated_schema():
    admin_dsn = _isolated_admin_dsn()
    schema_name = "native_empty_" + uuid.uuid4().hex
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
    try:
        database = _database(_schema_dsn(admin_dsn, schema_name))
        with database.connection() as conn:
            existing = conn.execute(
                "SELECT COUNT(*) AS count FROM information_schema.tables WHERE table_schema=current_schema()"
            ).fetchone()
            assert int(existing["count"]) == 0
        database.bootstrap()
        with database.connection() as conn:
            evidence = _catalog_evidence(conn)
        assert evidence["migration_count"] == 3
        assert database.assert_graphrag_native_ready()["migrations"] == list(
            GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS
        )
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))


@pytest.mark.integration
def test_native_migrations_upgrade_existing_stable_and_enhanced_state_without_mutation():
    """0022-0026 upgrade a pre-native schema without rewriting legacy graph state."""

    admin_dsn = _isolated_admin_dsn()
    schema_name = "native_upgrade_" + uuid.uuid4().hex
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
    try:
        schema_dsn = _schema_dsn(admin_dsn, schema_name)
        with psycopg.connect(schema_dsn, row_factory=dict_row) as conn:
            Database._enable_vector_extensions(conn)
            conn.execute(SCHEMA_SQL)
            MigrationRunner(MIGRATIONS[:21]).apply(conn)
            conn.commit()
        database = _database(schema_dsn)
        project = database.create_project("native migration fixture", "", "llm", "embed")
        with database.connection() as conn:
            # Seed only the two strategy rows understood by the frozen
            # pre-native schema.  Calling the current helper before applying
            # later migrations would incorrectly ask that historical CHECK
            # constraint to accept the current `knowledge_forge` key.
            conn.execute(
                """INSERT INTO project_strategy_graphs(
                   project_id,strategy_key,namespace_id,status,policy_summary)
                   VALUES
                   (%s,'stable',%s,'empty',%s::jsonb),
                   (%s,'enhanced_shadow',%s,'empty',%s::jsonb)""",
                (
                    project["id"], project["graph_namespace_id"],
                    json.dumps({"mode": "stable", "production_approved": True}),
                    project["id"], str(uuid.uuid4()),
                    json.dumps({"mode": "shadow", "production_approved": False}),
                ),
            )
            before = _legacy_strategy_fingerprint(conn, int(project["id"]))
            assert conn.execute(
                "SELECT to_regclass(current_schema()||'.graphrag_native_runs') AS table_name"
            ).fetchone()["table_name"] is None
            assert conn.execute(
                "SELECT COUNT(*) AS count FROM project_strategy_graphs WHERE project_id=%s",
                (project["id"],),
            ).fetchone()["count"] == 2

            native_upgrade = MigrationRunner(MIGRATIONS[:26])
            assert native_upgrade.apply(conn) == [
                "0022_graphrag_native_control_plane",
                "0023_project_construction_strategy_selection",
                "0024_project_strategy_binding",
                "0025_knowledge_forge_communities",
                "0026_schema_evaluation_audit",
            ]
            assert native_upgrade.apply(conn) == []
            evidence = _catalog_evidence(conn)
            assert evidence["migration_count"] == 3
            assert _legacy_strategy_fingerprint(conn, int(project["id"])) == before
            selection = conn.execute(
                """SELECT selected_strategy,association_status,bound_at
                   FROM project_construction_strategy_state WHERE project_id=%s""",
                (project["id"],),
            ).fetchone()
            assert selection == {
                "selected_strategy": "stable",
                "association_status": "unassigned",
                "bound_at": None,
            }
            assert conn.execute(
                "SELECT COUNT(*) AS count FROM graphrag_native_runs WHERE project_id=%s",
                (project["id"],),
            ).fetchone()["count"] == 0
            assert conn.execute(
                "SELECT COUNT(*) AS count FROM graphrag_native_project_state WHERE project_id=%s",
                (project["id"],),
            ).fetchone()["count"] == 1
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))


@pytest.mark.integration
def test_project_strategy_is_transactionally_bound_once_and_database_guarded():
    admin_dsn = _isolated_admin_dsn()
    schema_name = "strategy_binding_" + uuid.uuid4().hex
    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
    try:
        database = _database(_schema_dsn(admin_dsn, schema_name))
        database.bootstrap()
        native = database.create_project(
            "native bound", "", "llm", "embed", strategy_key="graphrag_native",
        )
        selection = database.get_project_construction_strategy_selection(int(native["id"]))
        assert selection
        assert selection["selected_strategy"] == "graphrag_native"
        assert selection["association_status"] == "bound"
        assert selection["bound_at"] is not None
        assert database.assert_project_construction_strategy(
            int(native["id"]), "graphrag_native",
        ) == "graphrag_native"
        with pytest.raises(ValueError, match="不一致"):
            database.assert_project_construction_strategy(
                int(native["id"]), "enhanced_shadow",
            )
        with database.connection() as conn:
            with pytest.raises(psycopg.errors.RaiseException, match="immutable"):
                conn.execute(
                    """UPDATE project_construction_strategy_state
                       SET selected_strategy='enhanced_shadow'
                       WHERE project_id=%s""",
                    (native["id"],),
                )

        legacy = database.create_project("legacy unassigned", "", "llm", "embed")
        before = database.get_project_construction_strategy_selection(int(legacy["id"]))
        assert before and before["association_status"] == "unassigned"
        bound = database.bind_project_construction_strategy(
            int(legacy["id"]), "enhanced_shadow",
            expected_generation=int(before["generation"]),
        )
        assert bound["association_status"] == "bound"
        with pytest.raises(RuntimeError, match="已绑定"):
            database.bind_project_construction_strategy(
                int(legacy["id"]), "graphrag_native",
                expected_generation=int(bound["generation"]),
            )
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))
