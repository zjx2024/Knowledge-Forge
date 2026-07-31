"""Disposable-PostgreSQL lifecycle tests for the native run control plane."""

from __future__ import annotations

import os
import uuid

import pytest
from psycopg.conninfo import conninfo_to_dict

from knowledge_forge.database import Database


def _database() -> Database:
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN", "").strip()
    if not dsn:
        pytest.skip("sentinel-bound disposable PostgreSQL DSN is absent")
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _project(database: Database, suffix: str) -> dict:
    return database.create_project(
        f"native-{suffix}", "", "qwen3.6:latest", "bge-m3:latest",
    )


def _active_revision(database: Database, project_id: int, suffix: str) -> tuple[dict, dict]:
    document = database.create_document({
        "project_id": project_id,
        "doc_hash": ("a" * 56) + suffix[:8],
        "original_filename": f"{suffix}.txt",
        "stored_filename": f"{suffix}.txt",
        "storage_path": f"/documents/{suffix}.txt",
        "local_path": f"/documents/{suffix}.txt",
        "file_size_bytes": 1,
        "mime_type": "text/plain",
    })
    revision = database.begin_document_revision(
        int(document["id"]), "ingestion-" + suffix,
        {
            "content_hash": document["doc_hash"], "source_hash": document["doc_hash"],
            "config_hash": "config-" + suffix, "policy_hash": "policy-" + suffix,
        },
    )
    with database.connection() as conn:
        conn.execute(
            "UPDATE document_content_revisions SET status='active' WHERE revision_id=%s",
            (revision["revision_id"],),
        )
        conn.execute(
            """UPDATE documents SET active_revision_id=%s,status='indexed',
                   structure_ready_at=NOW() WHERE id=%s""",
            (revision["revision_id"], document["id"]),
        )
    return database.get_document(int(document["id"])), database.get_document_revision(revision["revision_id"])


def _run_values(project_id: int, run_id: str, source_hash: str) -> dict:
    digest = "b" * 64
    return {
        "run_id": run_id,
        "project_id": project_id,
        "base_sha": "a" * 40,
        "worktree_diff_hash": digest,
        "code_tree_hash": digest,
        "source_snapshot_hash": source_hash,
        "settings_hash": digest,
        "prompt_profile_hash": digest,
        "completion_model_id": "test-completion",
        "completion_model_digest": digest,
        "embedding_model_id": "test-embedding",
        "embedding_model_digest": digest,
        "embedding_dimension": 8,
        "graphrag_version": "3.1.1",
        "python_version": "3.12",
        "staging_relative_path": f".staging/{run_id}-attempt",
    }


@pytest.mark.integration
def test_native_control_plane_fresh_catalog_and_fenced_activation():
    database = _database()
    assert database.assert_graphrag_native_ready()["migrations"] == [
        "0022_graphrag_native_control_plane",
    ]
    project = _project(database, uuid.uuid4().hex)
    document, revision = _active_revision(database, int(project["id"]), uuid.uuid4().hex)
    source_snapshot_hash = "c" * 64
    run_id = "native_" + uuid.uuid4().hex
    run = database.create_graphrag_native_run(
        _run_values(int(project["id"]), run_id, source_snapshot_hash),
    )
    assert run["state"] == "PENDING"
    owner = "worker-1"
    database.acquire_graphrag_native_run_lease(run_id, owner)
    database.replace_graphrag_native_run_documents(run_id, [{
        "project_id": project["id"], "document_id": document["id"],
        "revision_id": revision["revision_id"], "source_hash": revision["source_hash"],
        "structure_hash": "d" * 64, "input_document_count": 1,
        "input_manifest_hash": "e" * 64,
    }], owner=owner)
    database.update_graphrag_native_run(run_id, owner=owner, state="VALIDATING")
    ready = database.activate_graphrag_native_run(
        run_id, owner=owner, validation_report_hash="f" * 64,
        expected_source_snapshot_hash=source_snapshot_hash,
    )
    assert ready["state"] == "READY"
    state = database.get_graphrag_native_project_state(int(project["id"]))
    assert state["active_run_id"] == run_id
    # Native activation does not bind a legacy project.  Its one-time explicit
    # CAS binding is independent from the native active-run pointer and is
    # immutable afterwards.
    selection = database.get_project_construction_strategy_selection(int(project["id"]))
    assert selection["selected_strategy"] == "stable"
    assert selection["association_status"] == "unassigned"
    selected = database.select_project_construction_strategy(
        int(project["id"]), "graphrag_native", expected_generation=int(selection["generation"]),
    )
    assert selected["selected_strategy"] == "graphrag_native"
    assert selected["association_status"] == "bound"
    assert database.get_active_graphrag_native_run(int(project["id"]))["run_id"] == run_id
    with pytest.raises(RuntimeError, match="不能回滚或切换"):
        database.rollback_project_construction_strategy(
            int(project["id"]), "stable",
            expected_generation=int(selected["generation"]),
        )
    assert database.get_active_graphrag_native_run(int(project["id"]))["run_id"] == run_id
    replacement_id = "native_" + uuid.uuid4().hex
    database.create_graphrag_native_run(
        _run_values(int(project["id"]), replacement_id, source_snapshot_hash),
    )
    database.acquire_graphrag_native_run_lease(replacement_id, "worker-2")
    database.replace_graphrag_native_run_documents(replacement_id, [{
        "project_id": project["id"], "document_id": document["id"],
        "revision_id": revision["revision_id"], "source_hash": revision["source_hash"],
        "structure_hash": "d" * 64, "input_document_count": 1,
        "input_manifest_hash": "e" * 64,
    }], owner="worker-2")
    database.update_graphrag_native_run(replacement_id, owner="worker-2", state="VALIDATING")
    database.activate_graphrag_native_run(
        replacement_id, owner="worker-2", validation_report_hash="a" * 64,
        expected_source_snapshot_hash=source_snapshot_hash,
    )
    restored = database.rollback_graphrag_native_active_run(
        int(project["id"]), run_id, expected_active_run_id=replacement_id,
    )
    assert restored["active_run_id"] == run_id
    assert (
        database.get_project_construction_strategy_selection(
            int(project["id"]),
        )["selected_strategy"]
        == "graphrag_native"
    )
    query = database.create_graphrag_native_query_run(
        query_run_id="query_" + uuid.uuid4().hex, project_id=int(project["id"]),
        run_id=run_id, query_mode="basic", question_hash="9" * 64, model_id="test-completion",
    )
    completed = database.complete_graphrag_native_query_run(
        query["query_run_id"], status="READY", latency_ms=1, context_reference_count=2,
        token_usage={"status": "unavailable"},
    )
    assert completed["status"] == "READY"
    transient_id = "native_" + uuid.uuid4().hex
    database.create_graphrag_native_run(
        _run_values(int(project["id"]), transient_id, source_snapshot_hash),
    )
    database.acquire_graphrag_native_run_lease(transient_id, "release-test")
    released = database.release_graphrag_native_run_lease(transient_id, "release-test")
    assert released["lease_owner"] == ""
    assert released["lease_expires_at"] is None


@pytest.mark.integration
def test_native_control_plane_rejects_cross_project_revision_and_late_generation():
    database = _database()
    left = _project(database, uuid.uuid4().hex)
    right = _project(database, uuid.uuid4().hex)
    left_document, left_revision = _active_revision(database, int(left["id"]), uuid.uuid4().hex)
    right_document, right_revision = _active_revision(database, int(right["id"]), uuid.uuid4().hex)
    source_hash = "1" * 64
    run_one_id = "native_" + uuid.uuid4().hex
    run_one = database.create_graphrag_native_run(_run_values(int(left["id"]), run_one_id, source_hash))
    database.acquire_graphrag_native_run_lease(run_one_id, "one")
    with pytest.raises(ValueError, match="跨项目"):
        database.replace_graphrag_native_run_documents(run_one_id, [{
            "project_id": right["id"], "document_id": right_document["id"],
            "revision_id": right_revision["revision_id"], "source_hash": right_revision["source_hash"],
            "structure_hash": "2" * 64, "input_document_count": 1,
            "input_manifest_hash": "3" * 64,
        }], owner="one")
    database.replace_graphrag_native_run_documents(run_one_id, [{
        "project_id": left["id"], "document_id": left_document["id"],
        "revision_id": left_revision["revision_id"], "source_hash": left_revision["source_hash"],
        "structure_hash": "2" * 64, "input_document_count": 1,
        "input_manifest_hash": "3" * 64,
    }], owner="one")
    run_two_id = "native_" + uuid.uuid4().hex
    database.create_graphrag_native_run(_run_values(int(left["id"]), run_two_id, source_hash))
    database.update_graphrag_native_run(run_one_id, owner="one", state="VALIDATING")
    with pytest.raises(RuntimeError, match="generation"):
        database.activate_graphrag_native_run(
            run_one_id, owner="one", validation_report_hash="4" * 64,
            expected_source_snapshot_hash=source_hash,
        )
    assert run_one["generation"] == 1
    assert database.mark_graphrag_native_stale(int(left["id"]), "source updated") >= 1
    with pytest.raises(RuntimeError, match="过期|lease"):
        database.update_graphrag_native_run(
            run_one_id, owner="one", state="EXECUTION_COMPLETE",
        )


@pytest.mark.integration
def test_native_activation_rechecks_current_revisions_not_only_service_invalidation():
    database = _database()
    project = _project(database, uuid.uuid4().hex)
    document, revision = _active_revision(database, int(project["id"]), uuid.uuid4().hex)
    source_snapshot_hash = "7" * 64
    run_id = "native_" + uuid.uuid4().hex
    database.create_graphrag_native_run(_run_values(int(project["id"]), run_id, source_snapshot_hash))
    database.acquire_graphrag_native_run_lease(run_id, "worker")
    database.replace_graphrag_native_run_documents(run_id, [{
        "project_id": project["id"], "document_id": document["id"],
        "revision_id": revision["revision_id"], "source_hash": revision["source_hash"],
        "structure_hash": "8" * 64, "input_document_count": 1,
        "input_manifest_hash": "9" * 64,
    }], owner="worker")
    replacement = database.begin_document_revision(
        int(document["id"]), "new-content-" + uuid.uuid4().hex,
        {"content_hash": "e" * 64, "source_hash": "e" * 64, "config_hash": "cfg", "policy_hash": "pol"},
    )
    database.activate_document_revision(
        int(document["id"]), replacement["revision_id"], replacement["ingestion_run_id"],
        replacement["lease_token"], reason="test source replacement",
    )
    database.update_graphrag_native_run(run_id, owner="worker", state="VALIDATING")

    with pytest.raises(RuntimeError, match="source revision"):
        database.activate_graphrag_native_run(
            run_id, owner="worker", validation_report_hash="a" * 64,
            expected_source_snapshot_hash=source_snapshot_hash,
        )


def test_native_relative_path_validation_fails_closed():
    with pytest.raises(ValueError, match="不安全路径"):
        Database._validate_graphrag_native_relative_path("../escape", field="test")
    with pytest.raises(ValueError, match="相对路径"):
        Database._validate_graphrag_native_relative_path("/absolute", field="test")
