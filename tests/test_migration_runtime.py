from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from knowledge_forge.migrations import Migration, MigrationRunner


def _dsn() -> str:
    value = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not value:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    return value


def _connect(dsn: str, schema: str):
    conn = psycopg.connect(dsn, row_factory=dict_row)
    conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
    conn.commit()
    return conn


@pytest.mark.integration
def test_migration_plan_apply_idempotency_failure_resume_and_tamper():
    dsn = _dsn()
    schema = "migration_" + uuid.uuid4().hex[:12]
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        first = Migration(
            "9001_probe", "create probe",
            "CREATE TABLE migration_probe(id INTEGER PRIMARY KEY);"
            "INSERT INTO migration_probe VALUES(1);",
        )
        runner = MigrationRunner((first,))
        with _connect(dsn, schema) as conn:
            before = conn.execute(
                "SELECT COUNT(*) AS count FROM information_schema.tables WHERE table_schema=%s",
                (schema,),
            ).fetchone()["count"]
            assert runner.plan(conn)[0]["status"] == "pending"
            after = conn.execute(
                "SELECT COUNT(*) AS count FROM information_schema.tables WHERE table_schema=%s",
                (schema,),
            ).fetchone()["count"]
            assert before == after == 0
            assert runner.apply(conn) == ["9001_probe"]
            assert runner.apply(conn) == []
            assert conn.execute("SELECT COUNT(*) AS count FROM migration_probe").fetchone()["count"] == 1

        transient = Migration(
            "9002_resume", "transient failure",
            "DO $$ BEGIN IF current_setting('knowledge_forge.allow_resume',true) IS DISTINCT FROM 'yes' "
            "THEN RAISE EXCEPTION 'transient'; END IF; END $$;"
            "CREATE TABLE migration_resume_probe(id INTEGER PRIMARY KEY);",
        )
        resumable = MigrationRunner((first, transient))
        with _connect(dsn, schema) as conn:
            with pytest.raises(RuntimeError, match="9002_resume"):
                resumable.apply(conn)
            failed = conn.execute(
                "SELECT status,error_message FROM schema_migrations WHERE version='9002_resume'"
            ).fetchone()
            assert failed["status"] == "failed"
            assert "transient" in failed["error_message"]
            conn.execute("SET knowledge_forge.allow_resume='yes'")
            conn.commit()
            assert resumable.apply(conn) == ["9002_resume"]
            assert resumable.status(conn)[1]["status"] == "applied"

            tampered = MigrationRunner((Migration("9001_probe", "changed", "SELECT 2"),))
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                tampered.plan(conn)
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.integration
def test_two_migration_processes_are_serialized_and_apply_exactly_once():
    dsn = _dsn()
    schema = "migration_concurrent_" + uuid.uuid4().hex[:10]
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    migration = Migration(
        "9100_concurrent", "concurrent probe",
        "CREATE TABLE concurrent_probe(id INTEGER PRIMARY KEY);"
        "INSERT INTO concurrent_probe VALUES(1);"
        "SELECT pg_sleep(0.15);",
    )

    def apply_once():
        with _connect(dsn, schema) as conn:
            return MigrationRunner((migration,)).apply(conn)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [future.result(timeout=10) for future in (pool.submit(apply_once), pool.submit(apply_once))]
        assert sorted(len(result) for result in results) == [0, 1]
        with _connect(dsn, schema) as conn:
            assert conn.execute("SELECT COUNT(*) AS count FROM concurrent_probe").fetchone()["count"] == 1
            assert conn.execute(
                "SELECT COUNT(*) AS count FROM schema_migrations WHERE version='9100_concurrent' AND status='applied'"
            ).fetchone()["count"] == 1
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.mark.integration
def test_startup_safe_filter_nontransactional_mode_and_resumable_backfill():
    dsn = _dsn()
    schema = "migration_modes_" + uuid.uuid4().hex[:10]
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        heavy = Migration(
            "9200_heavy", "operator-only migration",
            "CREATE TABLE heavy_probe(id INTEGER PRIMARY KEY);",
        )
        safe = Migration(
            "9201_safe", "startup-safe migration",
            "CREATE TABLE safe_probe(id INTEGER PRIMARY KEY);",
            startup_safe=True,
        )
        concurrent_index = Migration(
            "9202_concurrent_index", "nontransactional index",
            "CREATE INDEX CONCURRENTLY idx_safe_probe_id ON safe_probe(id);",
            transactional=False,
        )
        runner = MigrationRunner((heavy, safe, concurrent_index))
        with _connect(dsn, schema) as conn:
            assert runner.apply(conn, startup_safe_only=True) == ["9201_safe"]
            assert conn.execute("SELECT to_regclass('heavy_probe') AS name").fetchone()["name"] is None
            assert conn.execute("SELECT to_regclass('safe_probe') AS name").fetchone()["name"] == "safe_probe"
            assert runner.status(conn)[0]["status"] == "pending"

            assert runner.apply(conn) == ["9200_heavy", "9202_concurrent_index"]
            mode = conn.execute(
                "SELECT status,execution_mode FROM schema_migrations WHERE version='9202_concurrent_index'"
            ).fetchone()
            assert mode == {"status": "applied", "execution_mode": "nontransactional"}
            assert conn.execute("SELECT to_regclass('idx_safe_probe_id') AS name").fetchone()["name"] == "idx_safe_probe_id"

            conn.execute("CREATE TABLE backfill_probe(id INTEGER PRIMARY KEY,done BOOLEAN NOT NULL DEFAULT FALSE)")
            conn.execute("INSERT INTO backfill_probe(id) SELECT generate_series(1,4)")
            conn.execute(
                """INSERT INTO schema_migrations(
                       version,description,checksum,status,execution_mode,progress,applied_at)
                   VALUES('9203_backfill','resumable backfill','test','applied','transactional','{}',NOW())"""
            )
            conn.commit()

            calls = 0

            def fetch_batch(connection, size):
                return connection.execute(
                    "SELECT id FROM backfill_probe WHERE done=FALSE ORDER BY id LIMIT %s",
                    (size,),
                ).fetchall()

            def fail_second_batch(connection, rows):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated backfill interruption")
                with connection.cursor() as cursor:
                    cursor.executemany(
                        "UPDATE backfill_probe SET done=TRUE WHERE id=%s",
                        [(row["id"],) for row in rows],
                    )

            with pytest.raises(RuntimeError, match="simulated backfill interruption"):
                MigrationRunner.run_batched_backfill(
                    conn,
                    version="9203_backfill",
                    fetch_batch=fetch_batch,
                    apply_batch=fail_second_batch,
                    batch_size=2,
                )
            conn.rollback()
            progress = conn.execute(
                "SELECT progress FROM schema_migrations WHERE version='9203_backfill'"
            ).fetchone()["progress"]
            assert progress == {"processed": 2}

            assert MigrationRunner.run_batched_backfill(
                conn,
                version="9203_backfill",
                fetch_batch=fetch_batch,
                apply_batch=lambda connection, rows: [
                    connection.execute(
                        "UPDATE backfill_probe SET done=TRUE WHERE id=%s",
                        (row["id"],),
                    )
                    for row in rows
                ],
                batch_size=2,
            ) == 4
            assert conn.execute(
                "SELECT progress FROM schema_migrations WHERE version='9203_backfill'"
            ).fetchone()["progress"] == {"processed": 4, "complete": True}
    finally:
        with psycopg.connect(dsn, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
