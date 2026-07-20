#!/usr/bin/env python3
"""Verify native PostgreSQL backup/restore against disposable synthetic databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_forge.config import Config  # noqa: E402
from knowledge_forge.database import Database, SCHEMA_SQL  # noqa: E402
from knowledge_forge.migrations import MigrationRunner  # noqa: E402


DATABASE_PREFIX = "knowledge_forge_backup_"
SENTINEL_TABLE = "kf_backup_sentinel"


def _canonical(value):
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _manifest(conn: psycopg.Connection) -> dict:
    queries = {
        "projects": "SELECT project_key,name,status,metadata FROM projects ORDER BY project_key",
        "documents": """SELECT doc_hash,original_filename,status,active_revision_id,content_version
                        FROM documents ORDER BY doc_hash""",
        "revisions": """SELECT revision_id,content_hash,source_hash,status,manifest_hash
                         FROM document_content_revisions ORDER BY revision_id""",
        "audit": "SELECT run_id,kind,metrics FROM operation_metrics ORDER BY run_id",
        "migrations": """SELECT version,checksum,status,execution_mode,progress
                          FROM schema_migrations ORDER BY version""",
    }
    result: dict[str, dict[str, object]] = {}
    for name, query in queries.items():
        rows = [_canonical(dict(row)) for row in conn.execute(query).fetchall()]
        payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        result[name] = {
            "count": len(rows),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }
    return result


def _create_database(conn: psycopg.Connection, name: str) -> None:
    if not name.startswith(DATABASE_PREFIX):
        raise RuntimeError("Refusing to create a database outside the disposable prefix")
    conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def _drop_database(conn: psycopg.Connection, name: str, run_id: str) -> None:
    if not name.startswith(DATABASE_PREFIX):
        raise RuntimeError("Refusing to drop a database outside the disposable prefix")
    database_config = {**Config.postgres, "dbname": name}
    with psycopg.connect(**database_config, row_factory=dict_row) as target:
        sentinel = target.execute(
            sql.SQL("SELECT run_id FROM {} LIMIT 1").format(sql.Identifier(SENTINEL_TABLE))
        ).fetchone()
        if not sentinel or sentinel["run_id"] != run_id:
            raise RuntimeError(f"Backup sentinel mismatch; refusing cleanup of {name}")
    conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s", (name,))
    conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(name)))


def _tool_command(tool: str, database: str, docker_container: str) -> list[str]:
    args = [tool, "-U", Config.postgres["user"]]
    if tool == "pg_dump":
        args.extend(["--format=custom", "--no-owner", "--no-privileges", database])
    else:
        args.extend(["--exit-on-error", "--no-owner", "--no-privileges", "-d", database])
    if docker_container:
        return ["docker", "exec", *( ["-i"] if tool == "pg_restore" else [] ), docker_container, *args]
    binary = shutil.which(tool)
    if not binary:
        raise RuntimeError(f"{tool} is not installed; use --docker-container or install PostgreSQL client tools")
    args[0] = binary
    args[1:1] = ["-h", str(Config.postgres["host"]), "-p", str(Config.postgres["port"])]
    return args


def _run_dump(database: str, destination: Path, docker_container: str) -> None:
    environment = {**os.environ, "PGPASSWORD": str(Config.postgres["password"])}
    with destination.open("wb") as output:
        subprocess.run(
            _tool_command("pg_dump", database, docker_container),
            check=True,
            stdout=output,
            env=environment,
        )


def _run_restore(database: str, source: Path, docker_container: str) -> None:
    environment = {**os.environ, "PGPASSWORD": str(Config.postgres["password"])}
    with source.open("rb") as input_stream:
        subprocess.run(
            _tool_command("pg_restore", database, docker_container),
            check=True,
            stdin=input_stream,
            env=environment,
        )


def _seed_source(database: str, run_id: str) -> dict:
    config = {**Config.postgres, "dbname": database}
    with psycopg.connect(**config, row_factory=dict_row) as conn:
        Database._enable_vector_extensions(conn)
        conn.execute(SCHEMA_SQL)
        project_id = conn.execute(
            """INSERT INTO projects(project_key,name,metadata)
               VALUES(%s,%s,%s::jsonb) RETURNING id""",
            (f"backup-{run_id}", "Synthetic backup fixture", '{"fixture":true}'),
        ).fetchone()["id"]
        conn.execute(
            """INSERT INTO documents(
                   project_id,doc_hash,original_filename,stored_filename,storage_path,
                   local_path,file_size_bytes,mime_type,status,metadata)
               VALUES(%s,%s,'fixture.txt','fixture.txt','synthetic://fixture','/tmp/fixture',
                      7,'text/plain','indexed','{"synthetic":true}'::jsonb)""",
            (project_id, hashlib.sha256(b"fixture").hexdigest()),
        )
        conn.execute(
            """INSERT INTO operation_metrics(run_id,kind,project_id,metrics)
               VALUES(%s,'backup_fixture',%s,'{"redacted":true}'::jsonb)""",
            (run_id, project_id),
        )
        conn.commit()
        MigrationRunner().apply(conn)
        conn.execute(
            sql.SQL("CREATE TABLE {}(run_id TEXT PRIMARY KEY)").format(sql.Identifier(SENTINEL_TABLE))
        )
        conn.execute(
            sql.SQL("INSERT INTO {}(run_id) VALUES(%s)").format(sql.Identifier(SENTINEL_TABLE)),
            (run_id,),
        )
        conn.commit()
        return _manifest(conn)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docker-container", default="",
        help="Run pg_dump/pg_restore inside this PostgreSQL container",
    )
    args = parser.parse_args(argv)
    run_id = uuid.uuid4().hex
    source_database = DATABASE_PREFIX + "source_" + run_id[:10]
    restored_database = DATABASE_PREFIX + "restore_" + run_id[:10]
    maintenance = {**Config.postgres, "dbname": Config.postgres_maintenance_db}
    created: list[str] = []
    with tempfile.TemporaryDirectory(prefix="knowledge-forge-backup-") as temporary:
        archive = Path(temporary) / "synthetic.backup"
        try:
            with psycopg.connect(**maintenance, autocommit=True) as conn:
                _create_database(conn, source_database)
                created.append(source_database)
                _create_database(conn, restored_database)
                created.append(restored_database)
            expected = _seed_source(source_database, run_id)
            _run_dump(source_database, archive, args.docker_container)
            _run_restore(restored_database, archive, args.docker_container)
            with psycopg.connect(
                **{**Config.postgres, "dbname": restored_database}, row_factory=dict_row,
            ) as restored:
                actual = _manifest(restored)
            if actual != expected:
                raise RuntimeError(
                    "Restored manifest does not match source: "
                    + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
                )
            print(json.dumps({
                "status": "passed",
                "source_database": source_database,
                "restored_database": restored_database,
                "manifest": expected,
            }, ensure_ascii=False, indent=2))
            return 0
        finally:
            with psycopg.connect(**maintenance, autocommit=True) as conn:
                for name in reversed(created):
                    _drop_database(conn, name, run_id)


if __name__ == "__main__":
    raise SystemExit(main())
