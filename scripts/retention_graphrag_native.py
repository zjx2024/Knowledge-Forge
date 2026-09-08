#!/usr/bin/env python3
"""List (or explicitly remove) safe GraphRAG Native retention candidates.

The default is dry-run.  ``--apply`` is deliberately one run at a time and
requires an exact ``--confirm-run-id``; it never deletes PostgreSQL rows,
active runs, last READY runs, or currently leased runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge.config import Config  # noqa: E402
from knowledge_forge.database import Database  # noqa: E402
from knowledge_forge.graphrag_native_artifacts import safe_bundle_path  # noqa: E402
from knowledge_forge.graphrag_native_retention import classify_retention_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative GraphRAG Native run retention")
    parser.add_argument("--project-id", required=True, type=int)
    parser.add_argument("--apply", action="store_true", help="remove exactly one already-listed terminal bundle")
    parser.add_argument("--confirm-run-id", default="", help="must exactly name the candidate passed to --apply")
    args = parser.parse_args()

    database = Database()
    database.assert_graphrag_native_ready()
    state = database.get_graphrag_native_project_state(args.project_id) or {}
    plan = classify_retention_candidates(
        database.list_graphrag_native_runs(args.project_id),
        active_run_id=str(state.get("active_run_id") or ""),
    )
    if not args.apply:
        print(json.dumps({"mode": "dry_run", **plan}, ensure_ascii=False, default=str, indent=2))
        return 0

    candidate_ids = {
        str(entry["run"].get("run_id") or ""): entry["run"]
        for entry in plan["candidates"]
    }
    target = candidate_ids.get(str(args.confirm_run_id))
    if not target:
        parser.error("--apply 只接受本次 dry-run 列出的 terminal non-active non-leased run，并且必须提供精确 --confirm-run-id")
    relative = str(target.get("bundle_relative_path") or target.get("staging_relative_path") or "")
    native_root = (Config.files_root / "graphrag_native").resolve()
    path = safe_bundle_path(native_root, relative)
    # A failed attempt may never have reached a final bundle.  Refuse a path
    # that is unexpectedly not a real directory instead of broadening scope.
    if not path.is_dir() or path.is_symlink():
        parser.error("候选 run 没有可安全删除的本地目录；未执行删除")
    shutil.rmtree(path)
    print(json.dumps({"mode": "applied", "deleted_run_id": args.confirm_run_id, "deleted_path": relative}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
