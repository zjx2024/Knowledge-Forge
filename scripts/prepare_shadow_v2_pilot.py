#!/usr/bin/env python3
"""Prepare only an explicitly named isolated Shadow v2 pilot database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge.database import Database
from knowledge_forge.relation_shadow_v2 import normalize_predicate_schema
from knowledge_forge.schema_evolution import BASE_SCHEMA


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)
    database_name = str(params.get("dbname") or "")
    if not database_name.startswith("knowledge_forge_shadow_v2_pilot_"):
        raise RuntimeError(
            "refusing non-pilot database: expected "
            "knowledge_forge_shadow_v2_pilot_*"
        )

    class PilotConfig:
        postgres = params
        postgres_maintenance_db = database_name

    return Database(PilotConfig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument(
        "--project-id", type=int, action="append", required=True,
    )
    args = parser.parse_args()
    database = _database(args.dsn)
    readiness = database.assert_shadow_relation_v2_ready()
    superseded_runs: list[str] = []
    with database.connection() as conn:
        rows = conn.execute(
            """UPDATE shadow_relation_v2_runs
               SET status='failed',completed_at=NOW(),
                   error_message='superseded by predicate-policy gate refreeze',
                   metrics=metrics||'{"superseded":true}'::jsonb
               WHERE status IN ('running','verifying','normalizing')
                 AND id LIKE 'shadow_relation_v2_legacy_replay_%'
               RETURNING id"""
        ).fetchall()
        superseded_runs = [str(row["id"]) for row in rows]

    schema_versions: dict[int, int] = {}
    for project_id in sorted(set(args.project_id)):
        active = database.get_active_schema(project_id)
        if not active:
            raise RuntimeError(f"project {project_id} has no active Schema")
        try:
            normalize_predicate_schema(active.get("schema_json") or {})
        except ValueError:
            documents = [
                int(item["id"]) for item in database.list_documents(project_id)
                if item.get("status") == "indexed" and not item.get("is_deleted")
            ]
            created = database.create_schema_version(
                project_id,
                BASE_SCHEMA,
                {
                    "summary": (
                        "Isolated Shadow v2 predicate-policy compatibility "
                        "upgrade; no production database mutation"
                    ),
                    "policy": "shadow-v2-predicate-policy-fail-closed",
                },
                sorted(documents),
                created_by="manual",
            )
            schema_versions[project_id] = int(created["version"])
        else:
            schema_versions[project_id] = int(active["version"])
        current = database.get_active_schema(project_id)
        normalize_predicate_schema(current.get("schema_json") or {})

    print(json.dumps({
        "database": conninfo_to_dict(args.dsn)["dbname"],
        "migrations": readiness["migrations"],
        "superseded_runs": superseded_runs,
        "active_schema_versions": schema_versions,
        "predicate_policy_version": BASE_SCHEMA["version"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
