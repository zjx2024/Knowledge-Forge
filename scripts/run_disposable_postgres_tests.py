#!/usr/bin/env python3
"""Run pytest in a sentinel-bound random schema owned by a restricted role."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg.conninfo import conninfo_to_dict, make_conninfo

from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.postgres_test_safety import (
    cleanup_disposable_schema,
    inspect_postgres_identity,
    provision_disposable_schema,
)


def main() -> int:
    production_dsn = make_conninfo(**Config.postgres)
    production_identity, server_version, pgvector_version = inspect_postgres_identity(
        production_dsn, schema="public",
    )
    maintenance_dsn = make_conninfo(
        **{**Config.postgres, "dbname": Config.postgres_maintenance_db}
    )
    context = provision_disposable_schema(
        maintenance_dsn, production_identity=production_identity,
    )
    test_postgres = conninfo_to_dict(context.test_dsn)

    class DisposableConfig:
        postgres = test_postgres
        postgres_maintenance_db = context.database

    # The runner owns the database lifecycle, so every test selection starts
    # from the same fully migrated schema instead of depending on test order.
    Database(DisposableConfig).bootstrap()
    environment = {
        **os.environ,
        "KNOWLEDGE_FORGE_TEST_POSTGRES_DSN": context.test_dsn,
        # Migration-contract tests need to create additional schemas, but only
        # inside this sentinel-bound random disposable database.
        "KNOWLEDGE_FORGE_TEST_POSTGRES_ADMIN_DSN": context.target_admin_dsn,
        "KNOWLEDGE_FORGE_TEST_SCHEMA": context.schema,
        "KNOWLEDGE_FORGE_TEST_RUN_ID": context.run_id,
        "KNOWLEDGE_FORGE_TEST_POSTGRES_IDENTITY": context.identity,
        "KNOWLEDGE_FORGE_PRODUCTION_POSTGRES_IDENTITY": production_identity,
    }
    report_file = tempfile.NamedTemporaryFile(
        prefix="kf-e2e-pytest-", suffix=".xml", delete=False,
    )
    report_file.close()
    command = [
        sys.executable, "-m", "pytest", "-q",
        f"--junitxml={report_file.name}", *(sys.argv[1:] or []),
    ]
    print(f"PostgreSQL server: {server_version}")
    print(f"pgvector: {pgvector_version}")
    print(f"Armed disposable PostgreSQL schema: {context.safe_report()}")
    try:
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode:
            return completed.returncode
        root = ElementTree.parse(report_file.name).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
        if tests == 0 or tests == skipped:
            print("Refusing a zero-test or all-skipped PostgreSQL run", file=sys.stderr)
            return 3
        return 0
    finally:
        Path(report_file.name).unlink(missing_ok=True)
        outcome = cleanup_disposable_schema(context)
        print(f"PostgreSQL cleanup: {outcome} ({context.schema})")


if __name__ == "__main__":
    raise SystemExit(main())
