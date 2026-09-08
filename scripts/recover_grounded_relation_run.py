#!/usr/bin/env python3
"""Recover a code-only stale Knowledge Forge relation run without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge.database import Database
from knowledge_forge.ollama import OllamaClient
from knowledge_forge.relation_runtime import GroundedRelationRuntime
from knowledge_forge.service import KnowledgeForgeService
from knowledge_forge.relation_replay import (
    PersistedRelationModelReplayGateway,
    assert_code_only_replay_compatible,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute exact persisted-call replay; otherwise validate only.",
    )
    args = parser.parse_args()

    database = Database()
    source_run = database.get_grounded_relation_run(args.source_run_id)
    if not source_run:
        raise RuntimeError("恢复源关系运行不存在")
    project_id = int(source_run["project_id"])
    namespace_id = str(source_run.get("snapshot_id") or "")
    namespace = database.get_grounded_entity_namespace(namespace_id)
    if not namespace or int(namespace["project_id"]) != project_id:
        raise RuntimeError("恢复源实体快照不存在或项目不匹配")
    work = database.grounded_entity_work_summary(namespace_id)
    if (
        int(work.get("expected") or 0) <= 0
        or int(work.get("completed") or 0) != int(work.get("expected") or 0)
        or int(work.get("pending") or 0)
        or int(work.get("failed") or 0)
    ):
        raise RuntimeError("实体快照工作账本未完整完成，拒绝关系恢复")

    ollama = OllamaClient()
    replay = PersistedRelationModelReplayGateway(
        database,
        args.source_run_id,
        runtime_model_gateway=ollama,
    )
    runtime = GroundedRelationRuntime(database, replay)
    current_context = runtime._context(project_id, namespace_id)
    compatibility = assert_code_only_replay_compatible(
        source_run, current_context["fingerprint"],
    )
    report = {
        "compatible": True,
        "project_id": project_id,
        "namespace_id": namespace_id,
        "entity_work": work,
        "fingerprint": compatibility,
        "replay": replay.replay_summary(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if not args.execute:
        return 0

    service = KnowledgeForgeService(database)
    service.ollama = replay

    def progress(done: int, total: int, message: str) -> None:
        print(f"[{done}/{total}] {message}", flush=True)

    result = service.build_grounded_entity_namespace(
        project_id,
        progress=progress,
    )
    if str(result.get("namespace_id") or "") != namespace_id:
        raise RuntimeError("恢复过程解析到了不同实体快照")
    ready = result.get("namespace") or {}
    if str(ready.get("status") or "") != "ready" or not ready.get(
        "snapshot_complete"
    ):
        raise RuntimeError("恢复后实体快照未就绪")
    print(json.dumps({
        "recovered": True,
        "project_id": project_id,
        "namespace_id": namespace_id,
        "relation_run_id": (
            (ready.get("metrics") or {}).get("relation_v2_run_id")
        ),
        "canonicals": len(result.get("canonicals") or []),
        "relations": len(result.get("relation_endpoints") or []),
        "replay": replay.replay_summary(),
    }, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
