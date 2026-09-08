#!/usr/bin/env python3
"""Forward safe live GraphRAG workflow progress for already-running jobs.

This is an operator-only bridge for a run that began before the in-process
runner gained live callbacks.  It reads only GraphRAG's structural log records,
never document/model payloads, and exits when no Native build remains active.
It does not start, cancel, restart, or otherwise alter GraphRAG subprocesses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Executing a script by path puts ``scripts/`` rather than the repository root
# on ``sys.path``.  Keep this bridge runnable both from a shell and a transient
# systemd unit without relying on ambient PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.graphrag_native_executor import GraphRAGNativeExecutor
from knowledge_forge.graphrag_native_runner import GraphRAGNativeRunner


def active_native_jobs(database: Database) -> list[dict]:
    with database.connection() as conn:
        return conn.execute(
            """SELECT * FROM jobs
               WHERE kind='graphrag_native_build' AND status='running'
               ORDER BY created_at,id""",
        ).fetchall()


def current_staging_root(database: Database, project_id: int) -> Path | None:
    run = next(
        (item for item in database.list_graphrag_native_runs(project_id)
         if str(item.get("state") or "") == "RUNNING"),
        None,
    )
    if not run:
        return None
    native_root = (Config.files_root / "graphrag_native").resolve()
    candidate = (native_root / str(run.get("staging_relative_path") or "")).resolve()
    if native_root not in candidate.parents or candidate.is_symlink() or not candidate.is_dir():
        return None
    return candidate


def observe_once(database: Database, offsets: dict[tuple[str, Path], int]) -> int:
    """Persist only newly observed safe workflow events; return active count."""

    jobs = active_native_jobs(database)
    active_keys: set[tuple[str, Path]] = set()
    for job in jobs:
        root = current_staging_root(database, int(job["project_id"]))
        if root is None:
            continue
        key = (str(job["id"]), root)
        active_keys.add(key)
        offset, events = GraphRAGNativeRunner._progress_events_from_log(
            root, offsets.get(key, 0), stage="index", elapsed_seconds=0,
        )
        offsets[key] = offset
        for event in events:
            latest = database.get_job(str(job["id"])) or {}
            if str(latest.get("status") or "") != "running":
                break
            progress = GraphRAGNativeExecutor._workflow_progress_value(
                event, int(latest.get("progress") or 30),
            )
            database.update_job(
                str(job["id"]), status="running", progress=progress,
                log="GraphRAG Native：" + event.message,
            )
    for key in list(offsets):
        if key not in active_keys:
            offsets.pop(key, None)
    return len(jobs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--idle-exit-seconds", type=float, default=45.0)
    args = parser.parse_args()
    interval = max(0.5, args.poll_seconds)
    idle_limit = max(interval, args.idle_exit_seconds)
    database = Database()
    offsets: dict[tuple[str, Path], int] = {}
    idle_since: float | None = None
    while True:
        active_count = observe_once(database, offsets)
        now = time.monotonic()
        if active_count:
            idle_since = None
        elif idle_since is None:
            idle_since = now
        elif now - idle_since >= idle_limit:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
