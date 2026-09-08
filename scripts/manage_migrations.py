#!/usr/bin/env python3
"""Plan, inspect, apply, and explicitly clean up Knowledge Forge migrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_forge.config import Config  # noqa: E402
from knowledge_forge.database import Database, SCHEMA_SQL  # noqa: E402
from knowledge_forge.migrations import MigrationRunner  # noqa: E402


LEGACY_COLUMNS = (
    ("projects", "ragflow_dataset_id"),
    ("projects", "ragflow_dataset_name"),
    ("projects", "default_parser"),
    ("projects", "default_chunk_strategy"),
    ("documents", "ragflow_document_id"),
    ("chunks", "ragflow_chunk_id"),
    ("graph_jobs", "ragflow_task_id"),
)


def _connect():
    return psycopg.connect(**Config.postgres, row_factory=dict_row)


def _print(rows) -> None:
    print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))


def _cleanup_legacy_columns(args: argparse.Namespace) -> int:
    if not args.execute:
        _print([{"table": table, "column": column, "action": "would_drop"}
                for table, column in LEGACY_COLUMNS])
        return 0
    if not args.backup_confirmation or len(args.backup_confirmation.strip()) < 8:
        raise SystemExit("--backup-confirmation 必须提供可追溯的备份标识")
    if not args.confirm_destructive_cleanup:
        raise SystemExit("必须同时提供 --confirm-destructive-cleanup")
    with _connect() as conn:
        for table, column in LEGACY_COLUMNS:
            relation = conn.execute("SELECT to_regclass(%s) AS name", (table,)).fetchone()
            if not relation or not relation["name"]:
                continue
            dependency = conn.execute(
                """SELECT COUNT(*) AS count
                   FROM pg_depend dependency
                   JOIN pg_attribute attribute
                     ON attribute.attrelid=dependency.refobjid
                    AND attribute.attnum=dependency.refobjsubid
                   WHERE dependency.refobjid=to_regclass(%s)
                     AND attribute.attname=%s
                     AND dependency.deptype NOT IN ('a','i')""",
                (table, column),
            ).fetchone()
            if int((dependency or {}).get("count") or 0):
                raise RuntimeError(f"{table}.{column} 仍被数据库对象引用，拒绝删除")
            conn.execute(
                sql.SQL("ALTER TABLE {} DROP COLUMN IF EXISTS {}").format(
                    sql.Identifier(table), sql.Identifier(column),
                )
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge Forge migration management")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="只读显示待执行迁移")
    subparsers.add_parser("status", help="只读显示迁移状态")
    apply_parser = subparsers.add_parser("apply", help="执行版本化迁移")
    apply_parser.add_argument("--startup-safe-only", action="store_true")
    apply_parser.add_argument("--initialize-empty", action="store_true")
    cleanup = subparsers.add_parser(
        "cleanup-legacy-columns", help="显式清理已确认无引用的旧列",
    )
    cleanup.add_argument("--execute", action="store_true")
    cleanup.add_argument("--backup-confirmation", default="")
    cleanup.add_argument("--confirm-destructive-cleanup", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "cleanup-legacy-columns":
        return _cleanup_legacy_columns(args)
    runner = MigrationRunner()
    with _connect() as conn:
        if args.command in {"plan", "status"}:
            _print(runner.plan(conn))
            return 0
        if args.initialize_empty:
            exists = conn.execute("SELECT to_regclass('projects') AS name").fetchone()
            if not exists or not exists["name"]:
                Database._enable_vector_extensions(conn)
                conn.execute(SCHEMA_SQL)
                conn.commit()
        applied = runner.apply(conn, startup_safe_only=args.startup_safe_only)
        _print({"applied": applied, "status": runner.status(conn)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
