from pathlib import Path

import pytest

from knowledge_forge.database import SCHEMA_SQL
from knowledge_forge.migrations import Migration, MigrationRunner


def test_startup_bootstrap_is_additive_and_has_no_legacy_column_drop():
    assert "schema_migrations" not in SCHEMA_SQL
    assert "DROP COLUMN" not in SCHEMA_SQL.upper()


class _PlanCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _ReadOnlyPlanConnection:
    def __init__(self, table_exists: bool, rows=()):
        self.table_exists = table_exists
        self.rows = list(rows)
        self.statements: list[str] = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.statements.append(sql)
        forbidden = ("CREATE ", "ALTER ", "DROP ", "INSERT ", "UPDATE ", "DELETE ")
        assert not sql.lstrip().upper().startswith(forbidden), sql
        if "to_regclass" in sql:
            return _PlanCursor([{"table_name": "schema_migrations" if self.table_exists else None}])
        return _PlanCursor(self.rows)


def test_migration_plan_is_zero_write_for_empty_and_existing_databases():
    migration = Migration("9999_test", "test", "SELECT 1")
    runner = MigrationRunner((migration,))

    empty = _ReadOnlyPlanConnection(False)
    assert runner.plan(empty)[0]["status"] == "pending"
    assert len(empty.statements) == 1

    existing = _ReadOnlyPlanConnection(
        True,
        [{"version": migration.version, "checksum": migration.checksum, "status": "applied"}],
    )
    assert runner.plan(existing)[0]["status"] == "applied"
    assert all("schema_migrations" in statement for statement in existing.statements)


def test_migration_checksum_tamper_fails_even_in_plan_mode():
    migration = Migration("9999_test", "test", "SELECT 1")
    conn = _ReadOnlyPlanConnection(
        True,
        [{"version": migration.version, "checksum": "tampered", "status": "applied"}],
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        MigrationRunner((migration,)).plan(conn)


def test_migration_cli_exposes_plan_status_apply_and_explicit_cleanup():
    source = Path("scripts/manage_migrations.py").read_text(encoding="utf-8")
    assert all(command in source for command in ("plan", "status", "apply", "cleanup-legacy-columns"))
    assert "--backup-confirmation" in source
    assert "--startup-safe-only" in source
