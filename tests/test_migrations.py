from pathlib import Path

import pytest

from knowledge_forge.database import SCHEMA_SQL
from knowledge_forge.migrations import MIGRATIONS, Migration, MigrationRunner


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


def test_mention_occurrence_migration_replaces_payload_uniqueness_idempotently():
    migration = next(
        item for item in MIGRATIONS if item.version == "0008_mention_occurrence_identity"
    )

    assert "ADD COLUMN IF NOT EXISTS mention_key" in migration.sql
    assert "DROP INDEX IF EXISTS uq_entity_mentions_revision_payload" in migration.sql
    assert "DROP INDEX IF EXISTS uq_resolution_revision_key" in migration.sql
    assert "DROP INDEX IF EXISTS idx_entity_alias_exact" in migration.sql
    assert "uq_resolution_revision_mention" in migration.sql
    assert "canonical_entity_aliases(project_id,normalized_alias,entity_type" in migration.sql
    assert "WHERE mention_key<>'' AND is_deleted=FALSE" in migration.sql
    assert "md5(" in migration.sql
    assert "digest(" not in migration.sql


def test_project_language_migration_preserves_legacy_auto_then_defaults_new_to_chinese():
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0034_project_knowledge_language"
    )

    assert "ADD COLUMN IF NOT EXISTS knowledge_language" in migration.sql
    assert "DEFAULT 'auto'" in migration.sql
    assert "ALTER COLUMN knowledge_language SET DEFAULT 'zh'" in migration.sql
    assert "knowledge_language IN ('auto','zh','en')" in migration.sql
    assert "knowledge_language TEXT NOT NULL DEFAULT 'zh'" in SCHEMA_SQL
