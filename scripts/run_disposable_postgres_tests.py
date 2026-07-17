#!/usr/bin/env python3
"""Create a disposable PostgreSQL database, run pytest, and always remove it."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

from knowledge_forge.config import Config


def main() -> int:
    database_name = f"knowledge_forge_test_{uuid.uuid4().hex[:12]}"
    maintenance = {**Config.postgres, "dbname": Config.postgres_maintenance_db}
    with psycopg.connect(**maintenance, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database_name)))
    test_dsn = make_conninfo(**{**Config.postgres, "dbname": database_name})
    with psycopg.connect(test_dsn, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    environment = {**os.environ, "KNOWLEDGE_FORGE_TEST_POSTGRES_DSN": test_dsn}
    command = [sys.executable, "-m", "pytest", "-q", *(sys.argv[1:] or [])]
    print(f"Created disposable PostgreSQL database: {database_name}")
    try:
        return subprocess.run(command, env=environment, check=False).returncode
    finally:
        with psycopg.connect(**maintenance, autocommit=True) as conn:
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s",
                (database_name,),
            )
            conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database_name)))
        print(f"Removed disposable PostgreSQL database: {database_name}")


if __name__ == "__main__":
    raise SystemExit(main())
