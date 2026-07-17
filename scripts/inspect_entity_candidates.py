"""Inspect project-scoped entity candidates without changing stored data."""

from __future__ import annotations

import argparse
import json

from knowledge_forge.database import Database
from knowledge_forge.entity_resolution import EntityResolver
from knowledge_forge.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_id", type=int)
    parser.add_argument("name")
    parser.add_argument("entity_type")
    parser.add_argument("--description", default="")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    database = Database()
    project = database.get_project(args.project_id)
    if not project:
        raise SystemExit(f"project {args.project_id} not found")
    resolver = EntityResolver(database, VectorStore(database))
    session = resolver.start_session(project)
    candidates, _ = resolver._retrieve_candidates(  # noqa: SLF001 - diagnostic utility
        session,
        {"name": args.name, "type": args.entity_type, "description": args.description, "aliases": []},
        {"content": ""},
    )
    print(json.dumps([
        {
            "id": item["id"],
            "name": item["name"],
            "type": item["entity_type"],
            "scores": item["resolution_scores"],
        }
        for item in candidates[: max(1, args.limit)]
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
