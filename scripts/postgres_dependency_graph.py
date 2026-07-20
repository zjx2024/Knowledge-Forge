#!/usr/bin/env python3
"""Print the PostgreSQL foreign-key dependency graph used by readiness tests."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row


DEPENDENCY_SQL = """
SELECT
    child.relname AS referencing_table,
    constraint_row.conname AS constraint_name,
    parent.relname AS referenced_table,
    pg_get_constraintdef(constraint_row.oid, true) AS definition,
    CASE constraint_row.confdeltype
        WHEN 'a' THEN 'NO ACTION'
        WHEN 'r' THEN 'RESTRICT'
        WHEN 'c' THEN 'CASCADE'
        WHEN 'n' THEN 'SET NULL'
        WHEN 'd' THEN 'SET DEFAULT'
        ELSE constraint_row.confdeltype::text
    END AS on_delete
FROM pg_constraint constraint_row
JOIN pg_class child ON child.oid = constraint_row.conrelid
JOIN pg_class parent ON parent.oid = constraint_row.confrelid
JOIN pg_namespace namespace_row ON namespace_row.oid = child.relnamespace
WHERE constraint_row.contype = 'f'
  AND namespace_row.nspname = current_schema()
ORDER BY referenced_table, referencing_table, constraint_name
"""


READINESS_TABLES = {
    "documents",
    "chunks",
    "entity_mentions",
    "relation_assertions",
    "mention_relation_assertions",
    "canonical_relation_projections",
    "entity_assignments",
    "entity_type_assertions",
    "relation_shadow_runs",
    "policy_snapshots",
    "policy_snapshot_documents",
    "indexing_failures",
    "relation_verification_events",
}


def collect_dependency_graph(connection: psycopg.Connection) -> list[dict[str, Any]]:
    """Return relevant FK edges without mutating the target database."""

    rows = connection.execute(DEPENDENCY_SQL).fetchall()
    return [
        dict(row)
        for row in rows
        if row["referencing_table"] in READINESS_TABLES
        or row["referenced_table"] in READINESS_TABLES
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN", ""))
    args = parser.parse_args()
    if not args.dsn:
        parser.error("--dsn or KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is required")
    with psycopg.connect(args.dsn, row_factory=dict_row) as connection:
        print(json.dumps(collect_dependency_graph(connection), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
