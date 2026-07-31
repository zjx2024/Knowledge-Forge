#!/usr/bin/env python3
"""Safely export one immutable ready GraphRAG Native run and revalidate it."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_forge.config import Config  # noqa: E402
from knowledge_forge.database import Database  # noqa: E402
from knowledge_forge.graphrag_native_artifacts import GraphRAGNativeBundleValidator, safe_bundle_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit("导出目录必须不存在或为空，拒绝覆盖")
    database = Database()
    database.assert_graphrag_native_ready()
    run = database.get_graphrag_native_run(args.run_id)
    if not run or int(run["project_id"]) != args.project_id:
        raise SystemExit("run 不属于指定项目")
    if str(run.get("state") or "") != "READY":
        raise SystemExit("只能导出 READY 的 immutable native run")
    source = safe_bundle_path((Config.files_root / "graphrag_native").resolve(), str(run.get("bundle_relative_path") or ""))
    GraphRAGNativeBundleValidator().validate(source)
    if destination.exists():
        destination.rmdir()
    shutil.copytree(source, destination, symlinks=False)
    report = GraphRAGNativeBundleValidator().validate(destination)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
