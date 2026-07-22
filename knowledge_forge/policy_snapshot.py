"""Immutable, resumable policy snapshot orchestration.

The orchestrator deliberately depends on a small persistence protocol.  The
PostgreSQL implementation lives in :mod:`knowledge_forge.database`; tests can
use an in-memory store without weakening lease or artifact semantics in
production.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence
import uuid

from .database import Database
from .policy import P1_RELATION_SHADOW_POLICY_SET, P2_ENTITY_SHADOW_POLICY_SET, normalize_policy_set


class SnapshotManifestError(ValueError):
    """The requested snapshot cannot be represented as a frozen input set."""


@dataclass(frozen=True)
class GitProvenance:
    commit: str
    tree: str
    dirty: bool
    tracked_diff_hash: str = ""
    untracked_code: tuple[str, ...] = ()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def collect_git_provenance(repo_root: str | Path = ".") -> GitProvenance:
    """Collect release-relevant Git lineage without reading file contents."""

    repo = Path(repo_root).resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    tracked_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=repo, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    relevant_suffixes = {
        ".py", ".sql", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg", ".sh",
    }
    untracked = []
    dirty = bool(tracked_diff)
    for line in status.splitlines():
        if not line:
            continue
        state, path = line[:2], line[3:]
        if state == "??":
            if Path(path).suffix.lower() in relevant_suffixes:
                untracked.append(path)
                dirty = True
        else:
            dirty = True
    return GitProvenance(
        commit=commit,
        tree=tree,
        dirty=dirty,
        tracked_diff_hash=hashlib.sha256(tracked_diff).hexdigest() if tracked_diff else "",
        untracked_code=tuple(sorted(untracked)),
    )


def _normalize_documents(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for source in documents:
        document_id = int(source.get("document_id") or source.get("id") or 0)
        revision_id = str(source.get("revision_id") or source.get("active_revision_id") or "")
        content_hash = str(source.get("content_hash") or source.get("doc_hash") or "")
        if not document_id or not revision_id or not content_hash:
            raise SnapshotManifestError(
                "snapshot document requires document_id, active revision_id and content_hash"
            )
        chunks = []
        for item in source.get("chunks") or []:
            chunk_id = int(item.get("chunk_id") or item.get("id") or 0)
            chunk_hash = str(item.get("content_hash") or "")
            if not chunk_id or not chunk_hash:
                raise SnapshotManifestError("snapshot chunk requires chunk_id and content_hash")
            chunks.append({"chunk_id": chunk_id, "content_hash": chunk_hash})
        chunks.sort(key=lambda item: (item["chunk_id"], item["content_hash"]))
        normalized.append({
            "document_id": document_id,
            "revision_id": revision_id,
            "content_hash": content_hash,
            "chunks": chunks,
        })
    normalized.sort(key=lambda item: (item["document_id"], item["revision_id"]))
    if len({item["document_id"] for item in normalized}) != len(normalized):
        raise SnapshotManifestError("snapshot document list contains duplicate document_id")
    return normalized


def build_snapshot_manifest(
    *,
    project: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    mode: str,
    policy_set: Mapping[str, Any],
    models: Mapping[str, Any],
    config: Mapping[str, Any],
    prompts: Mapping[str, Any],
    gold_manifest: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    metric_formulas: Mapping[str, Any],
    created_by: str,
    git: GitProvenance,
    created_at: str | None = None,
    release: bool = False,
) -> dict[str, Any]:
    """Build a canonical snapshot manifest whose identity freezes all inputs."""

    if mode not in {"relation_only", "end_to_end"}:
        raise SnapshotManifestError("mode must be relation_only or end_to_end")
    if not int(project.get("id") or 0) or not str(project.get("project_key") or ""):
        raise SnapshotManifestError("snapshot requires a persisted project namespace")
    if release and git.dirty:
        raise SnapshotManifestError("release snapshot requires a clean tracked worktree")
    if release and bool(gold_manifest.get("pilot", True)):
        raise SnapshotManifestError("release snapshot requires independently approved release Gold")
    normalized_documents = _normalize_documents(documents)
    normalized_policy = normalize_policy_set(policy_set)
    pilot = bool(gold_manifest.get("pilot", True) or git.dirty or not release)
    payload: dict[str, Any] = {
        "manifest_version": "policy-snapshot-manifest-v1",
        "mode": mode,
        "project": {
            "id": int(project["id"]),
            "project_key": str(project["project_key"]),
            "schema_version": int(project.get("active_schema_version") or 1),
        },
        "documents": normalized_documents,
        "corpus_hash": _digest(normalized_documents),
        "policy_set": normalized_policy,
        "policy_hash": _digest(normalized_policy),
        "models": dict(models),
        "model_config_hash": _digest(dict(models)),
        "config": dict(config),
        "config_hash": _digest(dict(config)),
        "prompts": dict(prompts),
        "prompt_hash": _digest(dict(prompts)),
        "gold_manifest": dict(gold_manifest),
        "gold_manifest_hash": _digest(dict(gold_manifest)),
        "gate_thresholds": dict(thresholds),
        "metric_formulas": dict(metric_formulas),
        "evaluator_hash": _digest({
            "metric_formulas": dict(metric_formulas),
            "thresholds": dict(thresholds),
        }),
        "calibration_universe_version": str(
            gold_manifest.get("calibration_universe_version") or "unspecified"
        ),
        "git": {**asdict(git), "untracked_code": list(git.untracked_code)},
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "created_by": created_by or "unknown",
        "input_count": {
            "documents": len(normalized_documents),
            "chunks": sum(len(item["chunks"]) for item in normalized_documents),
        },
        "pilot": pilot,
        "release_requested": bool(release),
    }
    seed_hash = _digest(payload)
    payload["snapshot_id"] = f"snapshot_{seed_hash[:32]}"
    payload["manifest_hash"] = _digest(payload)
    return payload


class PolicySnapshotOrchestrator:
    """Drive phase-ordered snapshot work through short DB-backed leases."""

    def __init__(self, store: Any):
        self.store = store

    @staticmethod
    def _work_items(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
        snapshot_id = str(manifest["snapshot_id"])
        items: list[dict[str, Any]] = []

        def add(phase: str, phase_order: int, document: Mapping[str, Any], chunk: Mapping[str, Any]):
            identity = {
                "snapshot_id": snapshot_id,
                "phase": phase,
                "document_id": int(document["document_id"]),
                "revision_id": str(document["revision_id"]),
                "chunk_id": int(chunk["chunk_id"]),
                "content_hash": str(chunk["content_hash"]),
            }
            fingerprint = _digest(identity)
            items.append({
                "id": f"snapshot_work_{fingerprint[:40]}",
                "phase": phase,
                "phase_order": phase_order,
                "project_id": int(manifest["project"]["id"]),
                "document_id": identity["document_id"],
                "revision_id": identity["revision_id"],
                "chunk_id": identity["chunk_id"],
                "input_hash": fingerprint,
            })

        documents = list(manifest.get("documents") or [])
        for document in documents:
            for chunk in document.get("chunks") or []:
                add("control", 10, document, chunk)
        if manifest["mode"] == "end_to_end":
            identity = {
                "snapshot_id": snapshot_id,
                "phase": "entity",
                "documents": [
                    (item["document_id"], item["revision_id"], item["content_hash"])
                    for item in documents
                ],
            }
            fingerprint = _digest(identity)
            items.append({
                "id": f"snapshot_work_{fingerprint[:40]}",
                "phase": "entity", "phase_order": 20,
                "project_id": int(manifest["project"]["id"]),
                "document_id": None, "revision_id": None, "chunk_id": None,
                "input_hash": fingerprint,
            })
        for document in documents:
            for chunk in document.get("chunks") or []:
                add("relation", 30, document, chunk)
        return items

    def create(
        self, project_id: int, manifest: Mapping[str, Any], policy_set: Mapping[str, Any],
    ) -> dict[str, Any]:
        if int(manifest["project"]["id"]) != int(project_id):
            raise SnapshotManifestError("manifest project does not match requested project")
        snapshot = self.store.create_policy_snapshot(
            project_id, policy_set,
            snapshot_id=str(manifest["snapshot_id"]),
            mode=str(manifest["mode"]),
            manifest_hash=str(manifest["manifest_hash"]),
            snapshot_manifest=dict(manifest),
            expected_documents=int(manifest["input_count"]["documents"]),
            expected_items=len(self._work_items(manifest)),
            pilot=bool(manifest["pilot"]),
            created_by=str(manifest["created_by"]),
        )
        self.store.seed_policy_snapshot_work_items(snapshot["id"], self._work_items(manifest))
        return snapshot

    def resume(
        self,
        snapshot_id: str,
        processor: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        *,
        owner: str,
        max_items: int | None = None,
        lease_seconds: int = 300,
    ) -> dict[str, Any]:
        processed = 0
        while max_items is None or processed < max_items:
            status = self.store.policy_snapshot_status(snapshot_id)
            snapshot = status["snapshot"]
            if snapshot["status"] in {"ready", "failed", "cancelled"}:
                return status
            work = self.store.claim_policy_snapshot_work(
                snapshot_id, owner, lease_seconds=lease_seconds,
            )
            if not work:
                break
            try:
                output = dict(processor(snapshot, work))
                artifact = {
                    "kind": work["phase"],
                    "work_id": work["id"],
                    "input_hash": work["input_hash"],
                    "payload": output,
                }
                self.store.settle_policy_snapshot_work(
                    work["id"], work["lease_token"], artifact=artifact,
                )
            except Exception as exc:
                self.store.settle_policy_snapshot_work(
                    work["id"], work["lease_token"], artifact=None,
                    error_message=f"{type(exc).__name__}: {exc}"[:2000],
                )
            processed += 1
        status = self.store.policy_snapshot_status(snapshot_id)
        open_items = [
            item for item in status["work_items"]
            if item["status"] in {"pending", "leased", "retryable_failed"}
        ]
        failed = [item for item in status["work_items"] if item["status"] == "terminal_failed"]
        if not open_items and not failed and status["snapshot"]["status"] == "building":
            self.store.finalize_policy_snapshot(snapshot_id)
        return self.store.policy_snapshot_status(snapshot_id)

    def status(self, snapshot_id: str) -> dict[str, Any]:
        return self.store.policy_snapshot_status(snapshot_id)

    def cancel(self, snapshot_id: str) -> dict[str, Any]:
        return self.store.request_policy_snapshot_cancel(snapshot_id)


def _load_json(path: str | None, fallback: Mapping[str, Any]) -> dict[str, Any]:
    if not path:
        return dict(fallback)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build and resume immutable policy snapshots")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--project-id", type=int, required=True)
    build.add_argument("--mode", choices=("relation_only", "end_to_end"), required=True)
    build.add_argument("--policy-set", default="treatment-v1")
    build.add_argument("--gold-manifest")
    build.add_argument("--created-by", default="cli")
    build.add_argument("--release", action="store_true")
    for name in ("resume", "status", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("--snapshot-id", required=True)
    args = parser.parse_args()
    database = Database()
    database.bootstrap()
    orchestrator = PolicySnapshotOrchestrator(database)
    if args.command == "build":
        project = database.get_project(args.project_id)
        if not project:
            raise SystemExit("project not found")
        policy = (
            P2_ENTITY_SHADOW_POLICY_SET
            if args.mode == "end_to_end" else P1_RELATION_SHADOW_POLICY_SET
        )
        manifest = build_snapshot_manifest(
            project=project,
            documents=database.list_policy_snapshot_inputs(args.project_id),
            mode=args.mode,
            policy_set=policy,
            models={
                "llm": project.get("llm_model") or "",
                "embedding": project.get("embedding_model") or "",
            },
            config={}, prompts={},
            gold_manifest=_load_json(args.gold_manifest, {"pilot": True}),
            thresholds={}, metric_formulas={}, created_by=args.created_by,
            git=collect_git_provenance(Path(__file__).resolve().parents[1]),
            release=args.release,
        )
        result = orchestrator.create(args.project_id, manifest, policy)
    elif args.command == "status":
        result = orchestrator.status(args.snapshot_id)
    elif args.command == "cancel":
        result = orchestrator.cancel(args.snapshot_id)
    else:
        # Resume is intentionally explicit about the processor.  The service
        # callback performs only shadow writes and is imported lazily.
        from .service import KnowledgeForgeService

        service = KnowledgeForgeService()
        result = orchestrator.resume(
            args.snapshot_id, service.process_policy_snapshot_work,
            owner=f"cli:{uuid.uuid4().hex}",
        )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests
    raise SystemExit(_cli())
