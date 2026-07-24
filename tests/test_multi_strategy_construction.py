from __future__ import annotations

from concurrent.futures import Future
import json
import os
import uuid

import pytest
from psycopg.conninfo import conninfo_to_dict

from knowledge_forge.construction_strategies import (
    ENHANCED_SHADOW_STRATEGY,
    STABLE_STRATEGY,
    normalize_strategy,
    strategy_options,
)
from knowledge_forge.database import Database
from knowledge_forge.migrations import MULTI_STRATEGY_CONSTRUCTION_SQL
from knowledge_forge.policy import POLICY_AXES
from knowledge_forge.policy import P1_RELATION_SHADOW_POLICY_SET
from knowledge_forge.service import KnowledgeForgeService
from knowledge_forge import web
from knowledge_forge.web import _construction_controls, _document_rows, _strategy_statistics


class _ImmediateRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def submit(self, callback, *args):
        self.calls.append((callback.__name__, args))
        future = Future()
        future.set_result(None)
        return future


class _StrategyDatabase:
    def __init__(self) -> None:
        self.document = {
            "id": 11,
            "project_id": 7,
            "status": "indexed",
            "active_revision_id": "revision-11",
            "original_filename": "source.pdf",
            "doc_hash": "source-hash",
            "requested_parser": "auto",
            "requested_chunk_strategy": "auto",
        }
        self.builds = {
            STABLE_STRATEGY: {"status": "indexed"},
            ENHANCED_SHADOW_STRATEGY: {"status": "indexed"},
        }
        self.active: dict[str, dict] = {}
        self.updated: list[tuple[int, str, dict]] = []
        self.invalidated: list[tuple[int, str]] = []
        self.stale: list[tuple[int, str, str]] = []
        self.jobs = {"job-enhanced_shadow": {"id": "job-enhanced_shadow", "metadata": {}}}
        self.revision: dict | None = None
        self.chunks = [{"id": 101, "document_id": 11, "content": "shared"}]

    def get_document(self, document_id):
        return self.document if document_id == 11 else None

    def ensure_project_strategy_graphs(self, project_id):
        return []

    def ensure_document_strategy_builds(self, document_id, reset=False):
        return []

    def latest_active_job(self, project_id, kind="", strategy_key=""):
        return self.active.get(strategy_key)

    def get_document_strategy_build(self, document_id, strategy_key):
        return self.builds[strategy_key]

    def create_job(self, kind, project_id, document_id, metadata, strategy_key):
        return {"id": f"job-{strategy_key}", "status": "queued"}

    def update_document_strategy_build(self, document_id, strategy_key, **values):
        self.updated.append((document_id, strategy_key, values))
        self.builds[strategy_key] = {**self.builds[strategy_key], **values}
        return self.builds[strategy_key]

    def get_project(self, project_id):
        return {
            "id": project_id,
            "project_key": "project-7",
            "graph_status": "ready",
            "active_schema_version": 1,
            "chunk_max_tokens": 512,
            "llm_model": "qwen3.6:latest",
            "embedding_model": "bge-m3:latest",
            **{axis: f"test-{axis}" for axis in POLICY_AXES},
        }

    def get_document_revision(self, revision_id):
        return self.revision if revision_id == "revision-11" else None

    def list_document_chunks(self, document_id):
        return list(self.chunks) if document_id == 11 else []

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def update_job(self, job_id, *, status=None, progress=None, log="", error="", metadata=None):
        job = self.jobs.setdefault(job_id, {"id": job_id, "metadata": {}})
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if metadata:
            job["metadata"] = {**job.get("metadata", {}), **metadata}
        return job

    def set_graph_state(self, project_id, status, **values):
        return {"id": project_id, "graph_status": status, **values}

    def invalidate_enhanced_strategy_for_document(self, document_id, *, reason):
        self.invalidated.append((document_id, reason))

    def mark_strategy_graph_stale(self, project_id, strategy_key, reason):
        self.stale.append((project_id, strategy_key, reason))
        return {"project_id": project_id, "strategy_key": strategy_key, "status": "stale"}


def _service(database: _StrategyDatabase) -> KnowledgeForgeService:
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = database
    service.executor = _ImmediateRecorder()
    return service


def test_strategy_registry_is_stable_and_rejects_unknown_values():
    assert [item.key for item in strategy_options()] == [
        STABLE_STRATEGY,
        ENHANCED_SHADOW_STRATEGY,
    ]
    assert normalize_strategy(None) == STABLE_STRATEGY
    with pytest.raises(ValueError, match="未知知识构建方案"):
        normalize_strategy("typo")


def test_stable_rebuild_invalidates_enhanced_snapshot_before_dispatch():
    database = _StrategyDatabase()
    service = _service(database)

    job_id = service.submit_ingestion(11, strategy_key=STABLE_STRATEGY)

    assert job_id == "job-stable"
    assert database.invalidated == [(11, "稳定基线开始重新构建，增强实验结果已失效")]
    assert service.executor.calls == [("_ingest_document", (job_id, 11))]


def test_enhanced_build_is_project_scoped_and_cannot_overlap_stable_work():
    database = _StrategyDatabase()
    database.active[STABLE_STRATEGY] = {"id": "stable-running", "status": "running"}
    service = _service(database)

    with pytest.raises(ValueError, match="文档预处理或稳定方案正在构建"):
        service.submit_ingestion(11, strategy_key=ENHANCED_SHADOW_STRATEGY)

    assert service.executor.calls == []
    assert database.updated == []


def test_stable_documents_can_queue_sequentially_but_cannot_overlap_enhanced_work():
    database = _StrategyDatabase()
    database.active[STABLE_STRATEGY] = {"id": "other-stable-queued", "status": "queued"}
    service = _service(database)

    assert service.submit_ingestion(11, strategy_key=STABLE_STRATEGY) == "job-stable"

    database = _StrategyDatabase()
    database.active[ENHANCED_SHADOW_STRATEGY] = {
        "id": "enhanced-running", "status": "running",
    }
    service = _service(database)
    with pytest.raises(ValueError, match="增强实验正在构建"):
        service.submit_ingestion(11, strategy_key=STABLE_STRATEGY)
    assert service.executor.calls == []
    assert database.invalidated == []


def test_enhanced_build_can_prepare_shared_content_without_stable_build():
    database = _StrategyDatabase()
    database.builds[STABLE_STRATEGY] = {"status": "uploaded"}
    database.document["status"] = "uploaded"
    database.document["active_revision_id"] = None
    service = _service(database)

    job_id = service.submit_ingestion(11, strategy_key=ENHANCED_SHADOW_STRATEGY)

    assert job_id == "job-enhanced_shadow"
    assert service.executor.calls == [("_ingest_enhanced_strategy", (job_id, 11))]
    assert database.updated[-1][2]["status"] == "queued"


def test_enhanced_project_job_marks_every_target_document_queued():
    class BatchDatabase(_StrategyDatabase):
        def __init__(self):
            super().__init__()
            self.second_document = {
                **self.document,
                "id": 12,
                "original_filename": "second.pdf",
                "doc_hash": "second-hash",
            }
            self.created_metadata = None

        def get_document(self, document_id):
            if document_id == 12:
                return self.second_document
            return super().get_document(document_id)

        def create_job(self, kind, project_id, document_id, metadata, strategy_key):
            self.created_metadata = metadata
            return super().create_job(kind, project_id, document_id, metadata, strategy_key)

    database = BatchDatabase()
    database.builds[ENHANCED_SHADOW_STRATEGY] = {"status": "uploaded"}
    service = _service(database)

    job_id = service.submit_ingestion(
        11,
        strategy_key=ENHANCED_SHADOW_STRATEGY,
        target_document_ids=[11, 12],
    )

    queued_targets = [
        document_id for document_id, strategy, values in database.updated
        if strategy == ENHANCED_SHADOW_STRATEGY and values.get("status") == "queued"
    ]
    assert queued_targets == [11, 12]
    assert database.created_metadata["target_document_ids"] == [11, 12]
    assert service.executor.calls == [("_ingest_enhanced_strategy", (job_id, 11))]


def test_shared_content_reuse_does_not_depend_on_stable_strategy_status():
    database = _StrategyDatabase()
    database.builds[STABLE_STRATEGY] = {"status": "uploaded"}
    service = _service(database)
    manifest = service._document_revision_manifest(
        database.document, database.get_project(7),
    )
    database.revision = {
        "revision_id": "revision-11",
        "status": "active",
        **manifest,
    }

    document, project, chunks = service._prepare_shared_content(
        "job-enhanced_shadow", 11,
    )

    assert document["id"] == 11
    assert project["id"] == 7
    assert chunks == database.chunks
    assert database.jobs["job-enhanced_shadow"]["metadata"]["shared_content_reused"] is True


def test_document_rows_show_selected_strategy_without_overwriting_shared_readiness():
    document = {
        "id": 11,
        "original_filename": "source.pdf",
        "file_size_bytes": 1024,
        "status": "indexed",
        "active_revision_id": "revision-11",
        "strategy_status": "uploaded",
        "strategy_error_message": "",
        "processing_stage": "completed",
        "structure_ready_at": "2026-01-01",
        "chunks_ready_at": "2026-01-01",
        "strategy_updated_at": "2026-01-01",
        "updated_at": "2026-01-01",
    }

    html = _document_rows([document], ENHANCED_SHADOW_STRATEGY)

    assert "等待增强实验构建" in html
    assert 'data-ingest="11"' not in html
    assert "构建知识" not in html
    assert 'data-view-structure="11" disabled' not in html
    assert 'data-view-chunks="11" disabled' not in html


def test_construction_controls_render_together_outside_upload_form():
    html = _construction_controls(
        project_id=7,
        strategy_key=ENHANCED_SHADOW_STRATEGY,
        strategy_select='<option value="enhanced_shadow">增强实验</option>',
        latest_job_id="job-7",
        latest_batch_id="batch-7",
        active_job_id="",
    )

    assert '<label><span>构建方案</span><select' in html
    assert 'id="prepare-project-documents"' in html
    assert 'id="build-project-knowledge"' in html
    assert 'id="view-ingestion-log"' in html
    assert html.count('class="construction-controls"') == 1


def test_enhanced_statistics_do_not_show_deleted_snapshot_for_empty_project(monkeypatch):
    class EmptyProjectDatabase:
        @staticmethod
        def count_documents(project_id):
            assert project_id == 7
            return 0

    monkeypatch.setattr(web, "SERVICE", type("Service", (), {"db": EmptyProjectDatabase()})())

    result = _strategy_statistics(
        7,
        ENHANCED_SHADOW_STRATEGY,
        {"stats": {"documents": 2, "chunks": 15, "entities": 266, "relations": 9}},
    )

    assert result == {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}


def test_enhanced_document_without_shared_content_can_start_build():
    document = {
        "id": 12,
        "original_filename": "new-source.pdf",
        "file_size_bytes": 1024,
        "status": "uploaded",
        "active_revision_id": None,
        "strategy_status": "uploaded",
        "strategy_error_message": "",
        "processing_stage": "uploaded",
        "structure_ready_at": None,
        "chunks_ready_at": None,
        "strategy_updated_at": "2026-01-01",
        "updated_at": "2026-01-01",
    }

    html = _document_rows([document], ENHANCED_SHADOW_STRATEGY)

    assert "等待文档预处理与增强实验构建" in html
    assert 'data-ingest="12"' not in html


class _ProjectBuildDatabase:
    def __init__(self) -> None:
        self.documents = [
            {"id": 12, "strategy_status": "failed"},
            {"id": 11, "strategy_status": "uploaded"},
            {"id": 13, "strategy_status": "indexed"},
        ]

    def get_project(self, project_id):
        return {"id": project_id} if project_id == 7 else None

    def list_documents(self, project_id, strategy_key):
        assert project_id == 7
        return list(self.documents)


class _PreparationDatabase:
    def __init__(self) -> None:
        self.documents = [{
            "id": 21,
            "project_id": 7,
            "original_filename": "pending.pdf",
            "doc_hash": "pending-hash",
            "requested_parser": "auto",
            "requested_chunk_strategy": "auto",
            "active_revision_id": None,
        }]

    def get_project(self, project_id):
        return {
            "id": project_id,
            "name": "test",
            "project_key": "test-7",
            "active_schema_version": 1,
            "chunk_max_tokens": 512,
            "llm_model": "qwen3.6:latest",
            "embedding_model": "bge-m3:latest",
            **{axis: f"test-{axis}" for axis in POLICY_AXES},
        }

    def latest_active_job(self, project_id, kind="", strategy_key=""):
        return None

    def list_documents(self, project_id, strategy_key=STABLE_STRATEGY):
        return list(self.documents)

    def list_document_chunks(self, document_id):
        return []

    def create_job(self, kind, project_id, document_id, *, metadata, strategy_key=STABLE_STRATEGY):
        assert kind == "content_preparation"
        return {"id": "prepare-21"}


def test_project_stable_build_queues_all_pending_documents_in_one_batch():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = _ProjectBuildDatabase()
    calls = []

    def submit(document_id, **values):
        calls.append((document_id, values))
        return f"job-{document_id}"

    service.submit_ingestion = submit
    result = service.submit_project_ingestion(7, strategy_key=STABLE_STRATEGY)

    assert [item[0] for item in calls] == [11, 12]
    assert len({item[1]["batch_id"] for item in calls}) == 1
    assert all("target_document_ids" not in item[1] for item in calls)
    assert result["job_ids"] == ["job-11", "job-12"]
    assert result["document_count"] == 2


def test_project_enhanced_build_uses_one_project_job_with_all_pending_documents():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = _ProjectBuildDatabase()
    calls = []

    def submit(document_id, **values):
        calls.append((document_id, values))
        return "job-enhanced"

    service.submit_ingestion = submit
    result = service.submit_project_ingestion(
        7, strategy_key=ENHANCED_SHADOW_STRATEGY,
    )

    assert len(calls) == 1
    assert calls[0][0] == 11
    assert calls[0][1]["target_document_ids"] == [11, 12]
    assert result["job_ids"] == ["job-enhanced"]
    assert result["document_count"] == 2


def test_project_preparation_queues_strategy_neutral_content_job():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = _PreparationDatabase()
    service.executor = _ImmediateRecorder()

    result = service.submit_project_preparation(7)

    assert result["job_ids"] == ["prepare-21"]
    assert result["document_count"] == 1
    assert service.executor.calls == [("_prepare_document_content", ("prepare-21", 21))]


def test_strategy_neutral_preparation_does_not_begin_graph_mutation():
    class JobDatabase:
        def __init__(self):
            self.mutations = []
            self.updates = []

        def begin_graph_mutation(self, job_id):
            self.mutations.append(job_id)

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))
            return {"id": job_id, **values}

    database = JobDatabase()
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = database

    service._start_ingestion_job("prepare-21", 21, mutate_graph=False)

    assert database.mutations == []
    assert database.updates[0][1]["status"] == "running"


def test_shadow_policy_constants_are_converted_before_json_serialization():
    payload = {"relation_policy_set": dict(P1_RELATION_SHADOW_POLICY_SET)}

    assert json.loads(json.dumps(payload))["relation_policy_set"] == dict(
        P1_RELATION_SHADOW_POLICY_SET
    )


def test_multi_strategy_migration_contains_bounded_strategy_contract():
    sql = MULTI_STRATEGY_CONSTRUCTION_SQL
    assert "CREATE TABLE IF NOT EXISTS document_strategy_builds" in sql
    assert "PRIMARY KEY(document_id,strategy_key)" in sql
    assert "CREATE TABLE IF NOT EXISTS project_strategy_graphs" in sql
    assert "PRIMARY KEY(project_id,strategy_key)" in sql
    assert "strategy_key IN ('stable','enhanced_shadow')" in sql
    assert sql.count("SELECT p.id,'enhanced_shadow'") == 1
    assert "digest(" not in sql


@pytest.mark.integration
def test_disposable_postgres_keeps_strategy_states_consistent_on_invalidate_and_delete():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    postgres_params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = postgres_params
        postgres_maintenance_db = postgres_params["dbname"]

    database = Database(DisposableConfig)
    database.bootstrap()
    suffix = uuid.uuid4().hex
    project = database.create_project(f"strategy-{suffix}", "", None, None)
    documents = []
    for index in range(2):
        document = database.create_document({
            "project_id": project["id"],
            "doc_hash": f"hash-{suffix}-{index}",
            "original_filename": f"source-{index}.txt",
            "stored_filename": f"source-{index}.txt",
            "storage_path": f"/source-{index}.txt",
            "local_path": f"/source-{index}.txt",
            "file_size_bytes": 10,
            "mime_type": "text/plain",
        })
        database.ensure_document_strategy_builds(int(document["id"]))
        documents.append(document)
    database.ensure_project_strategy_graphs(int(project["id"]))

    with database.connection() as connection:
        connection.execute(
            """UPDATE document_strategy_builds SET status='indexed'
               WHERE project_id=%s AND strategy_key=%s""",
            (project["id"], ENHANCED_SHADOW_STRATEGY),
        )
        connection.execute(
            """UPDATE project_strategy_graphs SET status='ready',generation=4,
                      ready_generation=4,
                      stats='{"documents":2,"chunks":15,"entities":266,"relations":9}'::jsonb
               WHERE project_id=%s AND strategy_key=%s""",
            (project["id"], ENHANCED_SHADOW_STRATEGY),
        )

    database.invalidate_enhanced_strategy_for_document(
        int(documents[0]["id"]), reason="stable revision changed",
    )
    with database.connection() as connection:
        builds = connection.execute(
            """SELECT document_id,status,content_revision_id
               FROM document_strategy_builds
               WHERE project_id=%s AND strategy_key=%s ORDER BY document_id""",
            (project["id"], ENHANCED_SHADOW_STRATEGY),
        ).fetchall()
    assert [row["status"] for row in builds] == ["uploaded", "uploaded"]
    assert all(row["content_revision_id"] is None for row in builds)
    graph = database.get_project_strategy_graph(
        int(project["id"]), ENHANCED_SHADOW_STRATEGY,
    )
    assert graph and graph["status"] == "stale"
    assert graph["generation"] == 5
    assert graph["ready_generation"] == 4

    deleted = database.soft_delete_document(int(documents[0]["id"]))
    assert deleted and deleted["status"] == "deleted"
    with database.connection() as connection:
        deleted_builds = connection.execute(
            """SELECT strategy_key,status FROM document_strategy_builds
               WHERE document_id=%s ORDER BY strategy_key""",
            (documents[0]["id"],),
        ).fetchall()
        graph_states = connection.execute(
            """SELECT strategy_key,status FROM project_strategy_graphs
               WHERE project_id=%s ORDER BY strategy_key""",
            (project["id"],),
        ).fetchall()
    assert {row["status"] for row in deleted_builds} == {"deleted"}
    assert {row["status"] for row in graph_states} == {"stale"}

    deleted = database.soft_delete_document(int(documents[1]["id"]))
    assert deleted and deleted["status"] == "deleted"
    with database.connection() as connection:
        empty_graph_states = connection.execute(
            """SELECT strategy_key,status,stats FROM project_strategy_graphs
               WHERE project_id=%s ORDER BY strategy_key""",
            (project["id"],),
        ).fetchall()
    assert {row["status"] for row in empty_graph_states} == {"stale"}
    assert all(row["stats"] == {} for row in empty_graph_states)
