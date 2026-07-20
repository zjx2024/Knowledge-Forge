import os

import psycopg
import pytest

from knowledge_forge.database import Database
from knowledge_forge.config import Config
from knowledge_forge.policy import P2_ENTITY_SHADOW_POLICY_SET
from knowledge_forge.policy_snapshot import (
    GitProvenance,
    PolicySnapshotOrchestrator,
    SnapshotManifestError,
    build_snapshot_manifest,
)


def _documents():
    return [
        {
            "document_id": 2,
            "revision_id": "revision-b",
            "content_hash": "hash-b",
            "chunks": [{"chunk_id": 20, "content_hash": "chunk-b"}],
        },
        {
            "document_id": 1,
            "revision_id": "revision-a",
            "content_hash": "hash-a",
            "chunks": [{"chunk_id": 10, "content_hash": "chunk-a"}],
        },
    ]


def test_manifest_is_canonical_immutable_and_dirty_release_is_rejected():
    git = GitProvenance(
        commit="abc123", tree="tree123", dirty=True,
        tracked_diff_hash="diff123", untracked_code=("scratch.py",),
    )
    first = build_snapshot_manifest(
        project={"id": 7, "project_key": "project-7", "active_schema_version": 3},
        documents=_documents(), mode="end_to_end",
        policy_set=P2_ENTITY_SHADOW_POLICY_SET,
        models={"llm": "llm@digest", "embedding": "embed@digest"},
        config={"candidate_cap": 20}, prompts={"entity": "entity-v1"},
        gold_manifest={"id": "pilot-gold", "pilot": True},
        thresholds={"candidate_recall": 0.99}, metric_formulas={"b3": "v1"},
        created_by="test", git=git, created_at="2026-07-20T00:00:00+00:00",
    )
    second = build_snapshot_manifest(
        project={"id": 7, "project_key": "project-7", "active_schema_version": 3},
        documents=list(reversed(_documents())), mode="end_to_end",
        policy_set=P2_ENTITY_SHADOW_POLICY_SET,
        models={"embedding": "embed@digest", "llm": "llm@digest"},
        config={"candidate_cap": 20}, prompts={"entity": "entity-v1"},
        gold_manifest={"pilot": True, "id": "pilot-gold"},
        thresholds={"candidate_recall": 0.99}, metric_formulas={"b3": "v1"},
        created_by="test", git=git, created_at="2026-07-20T00:00:00+00:00",
    )
    assert first["manifest_hash"] == second["manifest_hash"]
    assert [item["document_id"] for item in first["documents"]] == [1, 2]
    assert first["pilot"] is True
    assert first["git"]["untracked_code"] == ["scratch.py"]

    with pytest.raises(SnapshotManifestError, match="clean"):
        build_snapshot_manifest(
            project={"id": 7, "project_key": "project-7", "active_schema_version": 3},
            documents=_documents(), mode="relation_only",
            policy_set=P2_ENTITY_SHADOW_POLICY_SET, models={}, config={}, prompts={},
            gold_manifest={"pilot": False}, thresholds={}, metric_formulas={},
            created_by="test", git=git, release=True,
            created_at="2026-07-20T00:00:00+00:00",
        )


class MemorySnapshotStore:
    def __init__(self):
        self.snapshot = None
        self.items = []
        self.artifacts = {}
        self.cancelled = False

    def create_policy_snapshot(self, project_id, policy_set, **kwargs):
        self.snapshot = {
            "id": kwargs["snapshot_id"], "project_id": project_id,
            "mode": kwargs["mode"], "status": "building",
            "snapshot_manifest": kwargs["snapshot_manifest"],
            "manifest_hash": kwargs["manifest_hash"],
        }
        return self.snapshot

    def seed_policy_snapshot_work_items(self, snapshot_id, items):
        self.items = [{**item, "snapshot_id": snapshot_id, "status": "pending"} for item in items]
        return self.items

    def claim_policy_snapshot_work(self, snapshot_id, owner, **_):
        if self.cancelled:
            return None
        phase_order = {"control": 10, "entity": 20, "relation": 30}
        for item in sorted(self.items, key=lambda row: (phase_order[row["phase"]], row["id"])):
            previous_open = any(
                phase_order[other["phase"]] < phase_order[item["phase"]]
                and other["status"] != "completed" for other in self.items
            )
            if item["status"] == "pending" and not previous_open:
                item.update(status="leased", lease_token=f"lease:{item['id']}")
                return dict(item)
        return None

    def settle_policy_snapshot_work(self, work_id, lease_token, *, artifact, error_message=""):
        item = next(row for row in self.items if row["id"] == work_id)
        assert item["lease_token"] == lease_token
        if error_message:
            item["status"] = "pending"
            return item
        current = self.artifacts.get(work_id)
        if current is not None and current != artifact:
            raise RuntimeError("immutable artifact hash conflict")
        self.artifacts[work_id] = artifact
        item["status"] = "completed"
        return item

    def finalize_policy_snapshot(self, snapshot_id):
        assert all(item["status"] == "completed" for item in self.items)
        self.snapshot["status"] = "ready"
        return self.snapshot

    def policy_snapshot_status(self, snapshot_id):
        return {"snapshot": self.snapshot, "work_items": self.items, "artifacts": self.artifacts}


def test_orchestrator_orders_phases_resumes_and_keeps_artifacts_immutable():
    store = MemorySnapshotStore()
    orchestrator = PolicySnapshotOrchestrator(store)
    manifest = build_snapshot_manifest(
        project={"id": 7, "project_key": "project-7", "active_schema_version": 1},
        documents=_documents(), mode="end_to_end", policy_set=P2_ENTITY_SHADOW_POLICY_SET,
        models={}, config={}, prompts={}, gold_manifest={"pilot": True}, thresholds={},
        metric_formulas={}, created_by="test",
        git=GitProvenance("abc", "tree", False, "", ()),
        created_at="2026-07-20T00:00:00+00:00",
    )
    snapshot = orchestrator.create(7, manifest, P2_ENTITY_SHADOW_POLICY_SET)
    seen = []

    def processor(_snapshot, work):
        seen.append(work["phase"])
        return {"work_id": work["id"], "phase": work["phase"]}

    completed = orchestrator.resume(snapshot["id"], processor, owner="worker-a")
    assert completed["snapshot"]["status"] == "ready"
    assert seen == ["control", "control", "entity", "relation", "relation"]
    assert len(completed["artifacts"]) == 5

    # A restart is idempotent and cannot overwrite a previously accepted artifact.
    assert orchestrator.resume(snapshot["id"], processor, owner="worker-b")["snapshot"]["status"] == "ready"
    with pytest.raises(RuntimeError, match="immutable"):
        store.settle_policy_snapshot_work(
            store.items[0]["id"], store.items[0]["lease_token"], artifact={"changed": True},
        )


@pytest.mark.integration
def test_disposable_postgres_snapshot_leases_cas_and_artifact_immutability():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = Database()
    database.config = type("TestConfig", (), {
        "postgres": psycopg.conninfo.conninfo_to_dict(dsn),
        "postgres_maintenance_db": Config.postgres_maintenance_db,
        "pgvector_hnsw_max_dimensions": Config.pgvector_hnsw_max_dimensions,
        "pgvector_hnsw_m": Config.pgvector_hnsw_m,
        "pgvector_hnsw_ef_construction": Config.pgvector_hnsw_ef_construction,
    })
    database.bootstrap()
    project = database.create_project("snapshot", "", None, None)
    document = database.create_document({
        "project_id": project["id"], "doc_hash": "doc", "original_filename": "doc.txt",
        "stored_filename": "doc.txt", "storage_path": "/doc", "local_path": "/doc",
        "file_size_bytes": 3, "mime_type": "text/plain",
    })
    revision = database.begin_document_revision(
        document["id"], "run-1",
        {
            "content_hash": "doc", "source_hash": "doc",
            "config_hash": "config", "policy_hash": "policy",
        },
    )
    database.replace_document_content(
        document["id"],
        [{
            "key": "s", "title": "S", "level": 1, "content": "Alpha",
            "content_hash": "section",
        }],
        [{
            "key": "c", "section_key": "s", "content": "Alpha",
            "content_hash": "chunk", "part_index": 1,
        }],
        revision_id=revision["revision_id"], ingestion_run_id="run-1",
        lease_token=revision["lease_token"],
    )
    database.activate_document_revision(
        document["id"], revision["revision_id"], "run-1", revision["lease_token"],
        reason="snapshot fixture",
        document_values={"status": "indexed", "indexed_at": "2026-07-20T00:00:00+00:00"},
    )
    inputs = database.list_policy_snapshot_inputs(project["id"])
    manifest = build_snapshot_manifest(
        project=project, documents=inputs, mode="relation_only",
        policy_set=P2_ENTITY_SHADOW_POLICY_SET, models={}, config={}, prompts={},
        gold_manifest={"pilot": True}, thresholds={}, metric_formulas={}, created_by="test",
        git=GitProvenance("abc", "tree", False, "", ()),
        created_at="2026-07-20T00:00:00+00:00",
    )
    orchestrator = PolicySnapshotOrchestrator(database)
    snapshot = orchestrator.create(project["id"], manifest, P2_ENTITY_SHADOW_POLICY_SET)
    leased = database.claim_policy_snapshot_work(snapshot["id"], "worker-a", lease_seconds=1)
    assert leased and leased["lease_token"]
    assert database.claim_policy_snapshot_work(snapshot["id"], "worker-b", lease_seconds=1) is None
    assert database.heartbeat_policy_snapshot_work(
        leased["id"], leased["lease_token"], lease_seconds=5,
    ) is True
    database.settle_policy_snapshot_work(
        leased["id"], leased["lease_token"], artifact={"stable": True},
    )
    with pytest.raises(RuntimeError, match="immutable"):
        database.settle_policy_snapshot_work(
            leased["id"], leased["lease_token"], artifact={"stable": False},
        )

    # The next phase becomes claimable only after control completes.  An expired
    # worker loses CAS ownership and a replacement can resume the same item.
    relation = database.claim_policy_snapshot_work(snapshot["id"], "worker-a", lease_seconds=30)
    assert relation and relation["phase"] == "relation"
    with database.connection() as conn:
        conn.execute(
            "UPDATE policy_snapshot_work_items SET lease_expires_at=NOW()-INTERVAL '1 second' WHERE id=%s",
            (relation["id"],),
        )
    replacement = database.claim_policy_snapshot_work(snapshot["id"], "worker-b", lease_seconds=30)
    assert replacement and replacement["id"] == relation["id"]
    with pytest.raises(RuntimeError, match="stale"):
        database.settle_policy_snapshot_work(
            relation["id"], relation["lease_token"], artifact={"winner": "old"},
        )
    database.settle_policy_snapshot_work(
        replacement["id"], replacement["lease_token"], artifact={"winner": "new"},
    )
    ready = database.finalize_policy_snapshot(snapshot["id"])
    assert ready["status"] == "ready"
    assert ready["completed_items"] == ready["expected_items"] == 2
    assert ready["artifact_root_hash"]

    cancelled_manifest = build_snapshot_manifest(
        project=project, documents=inputs, mode="relation_only",
        policy_set=P2_ENTITY_SHADOW_POLICY_SET, models={}, config={"cancel": True}, prompts={},
        gold_manifest={"pilot": True}, thresholds={}, metric_formulas={}, created_by="test",
        git=GitProvenance("abc", "tree", False, "", ()),
        created_at="2026-07-20T00:00:01+00:00",
    )
    cancelled = orchestrator.create(
        project["id"], cancelled_manifest, P2_ENTITY_SHADOW_POLICY_SET,
    )
    database.request_policy_snapshot_cancel(cancelled["id"])
    assert database.claim_policy_snapshot_work(cancelled["id"], "worker-c") is None
    cancelled_status = database.policy_snapshot_status(cancelled["id"])
    assert cancelled_status["snapshot"]["status"] == "cancelled"
    assert {item["status"] for item in cancelled_status["work_items"]} == {"cancelled"}
