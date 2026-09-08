#!/usr/bin/env python3
"""Online control-plane freshness check for a single native run."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_forge.database import Database  # noqa: E402
from knowledge_forge.graphrag_native_runtime import GraphRAGNativeRuntime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    database = Database()
    database.assert_graphrag_native_ready()
    run = database.get_graphrag_native_run(args.run_id)
    if not run or int(run["project_id"]) != args.project_id:
        raise SystemExit("run 不属于指定项目")
    active = database.get_active_graphrag_native_run(args.project_id)
    if not active or str(active["run_id"]) != args.run_id:
        raise SystemExit("run 不是当前项目 active native pointer，拒绝把旧 bundle 当作 fresh")
    runtime = GraphRAGNativeRuntime(SimpleNamespace(db=database))
    runtime.current_run(args.project_id, validate_bundle=False)
    print("fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
