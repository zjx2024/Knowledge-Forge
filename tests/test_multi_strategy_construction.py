from __future__ import annotations

from concurrent.futures import Future
import json
import os
import re
import uuid

import pytest
from psycopg.conninfo import conninfo_to_dict

from knowledge_forge.construction_strategies import (
    GRAPHRAG_NATIVE_STRATEGY,
    KNOWLEDGE_FORGE_STRATEGY,
    LEGACY_KNOWLEDGE_FORGE_STRATEGY_KEY,
    STABLE_STRATEGY,
    normalize_strategy,
    strategy_options,
)
from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.migrations import MULTI_STRATEGY_CONSTRUCTION_SQL
from knowledge_forge.policy import ENTITY_EXTRACTION_POLICY_SET, POLICY_AXES
from knowledge_forge.policy import RELATION_CANDIDATE_POLICY_SET
from knowledge_forge.service import (
    GROUNDED_ENTITY_CODE_VERSION,
    _grounded_entity_freshness_policy_set,
    KnowledgeForgeService,
)
from knowledge_forge import web
from knowledge_forge.web import (
    _construction_controls,
    _document_rows,
    _knowledge_forge_graph_update_notice,
    _knowledge_forge_pending_graph_documents,
    _preparation_update_notice,
    _schema_reassessment_available,
    _strategy_statistics,
)


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
            KNOWLEDGE_FORGE_STRATEGY: {"status": "indexed"},
        }
        self.active: dict[str, dict] = {}
        self.updated: list[tuple[int, str, dict]] = []
        self.invalidated: list[tuple[int, str]] = []
        self.stale: list[tuple[int, str, str]] = []
        self.grounded_relation_readiness_checks = 0
        self.jobs = {
            "job-knowledge_forge": {
                "id": "job-knowledge_forge", "status": "queued", "metadata": {},
            },
        }
        self.revision: dict | None = None
        self.chunks = [{"id": 101, "document_id": 11, "content": "shared"}]

    def get_document(self, document_id):
        return self.document if document_id == 11 else None

    def ensure_project_strategy_graphs(self, project_id):
        return []

    def assert_grounded_relation_ready(self):
        self.grounded_relation_readiness_checks += 1
        return {"migrations": [f"00{value}" for value in range(14, 20)]}

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

    def invalidate_knowledge_forge_for_document(self, document_id, *, reason):
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
        KNOWLEDGE_FORGE_STRATEGY,
        GRAPHRAG_NATIVE_STRATEGY,
    ]
    assert normalize_strategy(None) == STABLE_STRATEGY
    assert [item.key for item in strategy_options(include_hidden=False)] == [
        KNOWLEDGE_FORGE_STRATEGY,
        GRAPHRAG_NATIVE_STRATEGY,
    ]
    with pytest.raises(ValueError, match="未知知识构建方案"):
        normalize_strategy("typo")
    assert normalize_strategy(LEGACY_KNOWLEDGE_FORGE_STRATEGY_KEY) == KNOWLEDGE_FORGE_STRATEGY


def test_schema_reassessment_is_owned_by_idle_knowledge_forge_projects():
    assert _schema_reassessment_available(KNOWLEDGE_FORGE_STRATEGY)
    assert not _schema_reassessment_available(
        KNOWLEDGE_FORGE_STRATEGY, "schema-job-running",
    )
    assert not _schema_reassessment_available(STABLE_STRATEGY)
    assert not _schema_reassessment_available(GRAPHRAG_NATIVE_STRATEGY)


def test_schema_reassessment_job_keeps_knowledge_forge_strategy_identity():
    database = _StrategyDatabase()
    created: list[tuple[str, int, str]] = []

    def create_job(
        kind, project_id, document_id=None, *, metadata=None,
        strategy_key=STABLE_STRATEGY,
    ):
        created.append((kind, project_id, strategy_key))
        return {"id": "schema-job", "status": "queued", "metadata": {}}

    database.create_job = create_job
    service = _service(database)

    assert service.submit_schema_reassessment(7) == "schema-job"
    assert created == [
        ("schema_reassessment", 7, KNOWLEDGE_FORGE_STRATEGY),
    ]
    assert service.executor.calls == [
        ("_reassess_project_schema", ("schema-job", 7)),
    ]


def test_stable_rebuild_preserves_ready_enhanced_snapshot_before_dispatch():
    database = _StrategyDatabase()
    service = _service(database)

    job_id = service.submit_ingestion(11, strategy_key=STABLE_STRATEGY)

    assert job_id == "job-stable"
    assert database.invalidated == []
    assert database.builds[KNOWLEDGE_FORGE_STRATEGY]["status"] == "indexed"
    assert service.executor.calls == [("_ingest_document", (job_id, 11))]


def test_enhanced_build_is_project_scoped_and_cannot_overlap_stable_work():
    database = _StrategyDatabase()
    database.active[STABLE_STRATEGY] = {"id": "stable-running", "status": "running"}
    service = _service(database)

    with pytest.raises(ValueError, match="文档预处理或稳定方案正在构建"):
        service.submit_ingestion(11, strategy_key=KNOWLEDGE_FORGE_STRATEGY)

    assert service.executor.calls == []
    assert database.updated == []


def test_stable_documents_can_queue_sequentially_but_cannot_overlap_enhanced_work():
    database = _StrategyDatabase()
    database.active[STABLE_STRATEGY] = {"id": "other-stable-queued", "status": "queued"}
    service = _service(database)

    assert service.submit_ingestion(11, strategy_key=STABLE_STRATEGY) == "job-stable"

    database = _StrategyDatabase()
    database.active[KNOWLEDGE_FORGE_STRATEGY] = {
        "id": "enhanced-running", "status": "running",
    }
    service = _service(database)
    with pytest.raises(ValueError, match="Knowledge Forge 正在构建"):
        service.submit_ingestion(11, strategy_key=STABLE_STRATEGY)
    assert service.executor.calls == []
    assert database.invalidated == []


def test_enhanced_build_can_prepare_shared_content_without_stable_build():
    database = _StrategyDatabase()
    database.builds[STABLE_STRATEGY] = {"status": "uploaded"}
    database.document["status"] = "uploaded"
    database.document["active_revision_id"] = None
    service = _service(database)

    job_id = service.submit_ingestion(11, strategy_key=KNOWLEDGE_FORGE_STRATEGY)

    assert job_id == "job-knowledge_forge"
    assert database.grounded_relation_readiness_checks == 1
    assert service.executor.calls == [("_ingest_knowledge_forge_strategy", (job_id, 11))]
    assert database.updated[-1][2]["status"] == "queued"


def test_enhanced_entry_fails_closed_when_shadow_v2_migrations_are_missing():
    class MissingMigrationDatabase(_StrategyDatabase):
        def assert_grounded_relation_ready(self):
            raise RuntimeError("Shadow relation v2 数据库结构不完整")

    database = MissingMigrationDatabase()
    service = _service(database)

    with pytest.raises(RuntimeError, match="数据库结构不完整"):
        service.submit_ingestion(11, strategy_key=KNOWLEDGE_FORGE_STRATEGY)

    assert service.executor.calls == []
    assert database.updated == []
    assert database.stale == []


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

        def get_project(self, project_id):
            return {
                **super().get_project(project_id),
                "knowledge_language": "en",
            }

        def create_job(self, kind, project_id, document_id, metadata, strategy_key):
            self.created_metadata = metadata
            return super().create_job(kind, project_id, document_id, metadata, strategy_key)

    database = BatchDatabase()
    database.builds[KNOWLEDGE_FORGE_STRATEGY] = {"status": "uploaded"}
    service = _service(database)

    job_id = service.submit_ingestion(
        11,
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        target_document_ids=[11, 12],
    )

    queued_targets = [
        document_id for document_id, strategy, values in database.updated
        if strategy == KNOWLEDGE_FORGE_STRATEGY and values.get("status") == "queued"
    ]
    assert queued_targets == [11, 12]
    assert database.created_metadata["target_document_ids"] == [11, 12]
    assert database.created_metadata["knowledge_language"] == "en"
    assert service.executor.calls == [("_ingest_knowledge_forge_strategy", (job_id, 11))]


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
        "job-knowledge_forge", 11,
    )

    assert document["id"] == 11
    assert project["id"] == 7
    assert chunks == database.chunks
    assert database.jobs["job-knowledge_forge"]["metadata"]["shared_content_reused"] is True


def test_shared_content_reuse_observes_cancellation_before_reporting_success():
    database = _StrategyDatabase()
    service = _service(database)
    manifest = service._document_revision_manifest(
        database.document, database.get_project(7),
    )
    database.revision = {
        "revision_id": "revision-11",
        "status": "active",
        **manifest,
    }
    database.jobs["job-knowledge_forge"]["status"] = "running"
    database.jobs["job-knowledge_forge"]["metadata"]["cancel_requested"] = True

    with pytest.raises(RuntimeError, match="安全终止"):
        service._prepare_shared_content("job-knowledge_forge", 11)

    assert "shared_content_reused" not in database.jobs[
        "job-knowledge_forge"
    ]["metadata"]


def test_graph_build_can_require_an_existing_preparation_result():
    database = _StrategyDatabase()
    service = _service(database)

    with pytest.raises(RuntimeError, match="请先完成文档预处理"):
        service._prepare_shared_content(
            "job-knowledge_forge", 11,
            require_reusable=True,
            report_progress=False,
        )


def test_graph_build_reuses_preparation_without_publishing_fake_progress():
    database = _StrategyDatabase()
    service = _service(database)
    manifest = service._document_revision_manifest(
        database.document, database.get_project(7),
    )
    database.revision = {
        "revision_id": "revision-11",
        "status": "active",
        **manifest,
    }
    before = dict(database.jobs["job-knowledge_forge"])

    document, _project, chunks = service._prepare_shared_content(
        "job-knowledge_forge", 11,
        require_reusable=True,
        report_progress=False,
    )

    assert document["id"] == 11
    assert chunks == database.chunks
    assert database.jobs["job-knowledge_forge"] == before


def test_shared_content_rebuilds_only_stale_embedding_index():
    database = _StrategyDatabase()
    service = _service(database)
    manifest = service._document_revision_manifest(
        database.document, database.get_project(7),
    )
    database.revision = {
        "revision_id": "revision-11",
        "status": "active",
        **manifest,
    }

    class StaleVectors:
        index_calls = 0

        @staticmethod
        def document_index_is_current(project, document_id, chunks):
            return False

        def index_chunks(self, project, document_id, chunks):
            self.index_calls += 1
            return len(chunks)

    service.vectors = StaleVectors()

    document, _project, chunks = service._prepare_shared_content(
        "job-knowledge_forge", 11,
    )

    assert document["id"] == 11
    assert chunks == database.chunks
    assert service.vectors.index_calls == 1
    metadata = database.jobs["job-knowledge_forge"]["metadata"]
    assert metadata["shared_content_reused"] is True
    assert metadata["embedding_index_rebuilt"] is True
    assert "更新 1 个向量索引" in metadata["subtasks"]["chunking"]["detail"]


def test_document_revision_fingerprint_tracks_chunking_configuration(monkeypatch):
    database = _StrategyDatabase()
    service = _service(database)
    document = database.document
    project = database.get_project(7)

    before = service._document_revision_manifest(document, project)["config_hash"]
    monkeypatch.setattr(
        Config,
        "general_coalesce_min_sections",
        int(Config.general_coalesce_min_sections) + 1,
    )
    after = service._document_revision_manifest(document, project)["config_hash"]

    assert before != after


def test_schema_review_uses_only_graph_eligible_non_parent_chunks():
    chunks = [
        {"id": 1, "metadata": {}},
        {"id": 2, "metadata": {"chunk_role": "child"}},
        {"id": 3, "metadata": {"graph_extraction_enabled": False}},
        {"id": 4, "metadata": {
            "chunk_role": "parent", "graph_extraction_enabled": True,
        }},
        {"id": 5, "metadata": json.dumps({
            "chunk_role": "child", "graph_extraction_enabled": False,
        })},
    ]

    selected = KnowledgeForgeService._schema_evaluation_chunks(chunks)

    assert [item["id"] for item in selected] == [1, 2]


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

    html = _document_rows([document], KNOWLEDGE_FORGE_STRATEGY)

    assert "预处理已就绪，等待图谱构建" in html
    assert 'data-ingest="11"' not in html
    assert "构建知识" not in html
    assert 'data-view-structure="11" disabled' not in html
    assert 'data-view-chunks="11" disabled' not in html


def test_construction_controls_render_together_outside_upload_form():
    html = _construction_controls(
        project_id=7,
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        strategy_select='<option value="knowledge_forge">Knowledge Forge</option>',
        latest_job_id="job-7",
        latest_batch_id="batch-7",
        active_job_id="",
    )

    assert 'class="fixed-strategy"' not in html
    assert "<select" not in html
    assert 'id="prepare-project-documents"' in html
    assert 'id="build-project-knowledge"' in html
    assert 'id="view-ingestion-log"' in html
    assert html.count('class="construction-controls"') == 1


def test_construction_controls_can_be_split_between_preparation_and_graph_pages():
    common = {
        "project_id": 7,
        "strategy_key": KNOWLEDGE_FORGE_STRATEGY,
        "latest_job_id": "job-7",
        "latest_batch_id": "batch-7",
        "active_job_id": "",
    }

    preparation = _construction_controls(**common, scope="preparation")
    graph = _construction_controls(**common, scope="graph")

    assert 'id="prepare-project-documents"' in preparation
    assert 'id="view-preparation-log"' in preparation
    assert 'id="build-project-knowledge"' not in preparation
    assert 'id="view-ingestion-log"' not in preparation
    assert 'id="prepare-project-documents"' not in graph
    assert 'id="view-preparation-log"' not in graph
    assert 'id="build-project-knowledge"' in graph
    assert 'id="view-ingestion-log"' in graph


def test_active_preparation_and_graph_controls_render_safe_stop_actions():
    common = {
        "project_id": 7,
        "strategy_key": KNOWLEDGE_FORGE_STRATEGY,
        "latest_job_id": "job-7",
        "latest_batch_id": "batch-7",
        "active_job_id": "job-7",
        "active_batch_id": "batch-7",
    }

    preparation = _construction_controls(
        **common,
        active_job_kind="content_preparation",
        scope="preparation",
    )
    graph = _construction_controls(
        **common,
        active_job_kind="ingestion",
        scope="graph",
    )

    assert "终止预处理" in preparation
    assert 'data-cancel-scope="preparation"' in preparation
    assert 'data-batch="batch-7"' in preparation
    assert "终止构建" in graph
    assert 'data-cancel-scope="graph"' in graph
    assert 'data-kind="ingestion"' in graph
    assert re.search(r'data-cancel-scope="preparation"[^>]* hidden', preparation) is None
    assert re.search(r'data-cancel-scope="graph"[^>]* hidden', graph) is None


def test_construction_controls_disable_build_until_preprocessed():
    html = _construction_controls(
        project_id=7,
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        strategy_select='<option value="knowledge_forge">Knowledge Forge</option>',
        latest_job_id="",
        latest_batch_id="",
        active_job_id="",
        build_ready=False,
    )

    button = re.search(r'<button[^>]+id="build-project-knowledge"[^>]*>', html)
    assert button
    assert 'class="button primary"' in button.group(0)
    assert "disabled" in button.group(0)
    assert "请先预处理全部文档" in button.group(0)


def test_construction_controls_enable_build_after_preprocessing():
    html = _construction_controls(
        project_id=7,
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        strategy_select='<option value="knowledge_forge">Knowledge Forge</option>',
        latest_job_id="",
        latest_batch_id="",
        active_job_id="",
        build_ready=True,
    )

    button = re.search(r'<button[^>]+id="build-project-knowledge"[^>]*>', html)
    assert button
    assert 'class="button primary"' in button.group(0)
    assert "disabled" not in button.group(0)
    assert "基于已完成预处理" in button.group(0)


def test_construction_controls_disable_preparation_without_pending_documents():
    html = _construction_controls(
        project_id=7,
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        latest_job_id="",
        latest_batch_id="",
        active_job_id="",
        preparation_pending=False,
        scope="preparation",
    )

    button = re.search(r'<button[^>]+id="prepare-project-documents"[^>]*>', html)
    assert button
    assert "disabled" in button.group(0)
    assert "当前没有待处理文档" in button.group(0)


def test_pending_graph_documents_use_strategy_revision_not_upload_timestamp():
    graph_state = {"status": "ready", "generation": 4, "ready_generation": 4}
    documents = [
        {
            "id": 11,
            "status": "indexed",
            "strategy_status": "indexed",
            "active_revision_id": "revision-11",
            "strategy_revision_id": "revision-11",
        },
        {
            "id": 12,
            "status": "indexed",
            "strategy_status": "uploaded",
            "active_revision_id": "revision-12",
            "strategy_revision_id": None,
        },
        {
            "id": 13,
            "status": "indexed",
            "strategy_status": "indexed",
            "active_revision_id": "revision-13-new",
            "strategy_revision_id": "revision-13-old",
        },
    ]

    pending = _knowledge_forge_pending_graph_documents(documents, graph_state)

    assert [item["id"] for item in pending] == [12, 13]
    assert _knowledge_forge_pending_graph_documents(
        documents, {"status": "empty", "ready_generation": 0},
    ) == []


def test_graph_update_notice_guides_preparation_then_rebuild():
    documents = [{
        "id": 12,
        "original_filename": "new-source.pdf",
        "status": "uploaded",
        "strategy_status": "uploaded",
        "active_revision_id": None,
        "strategy_revision_id": None,
    }]
    graph_state = {"status": "ready", "generation": 4, "ready_generation": 4}

    preparation_notice = _knowledge_forge_graph_update_notice(
        documents, graph_state, build_ready=False,
    )
    documents[0].update(
        status="indexed",
        active_revision_id="revision-12",
        structure_ready_at="now",
    )
    rebuild_notice = _knowledge_forge_graph_update_notice(
        documents, graph_state, build_ready=True,
    )

    assert "有 1 份文档尚未纳入当前知识图谱" in preparation_notice
    assert "请先完成文档预处理，再重新构建" in preparation_notice
    assert 'data-graph-update-action="prepare"' in preparation_notice
    assert "重新构建后生效" in rebuild_notice
    assert 'data-graph-update-action="build"' in rebuild_notice


def test_enhanced_statistics_do_not_show_deleted_snapshot_for_empty_project(monkeypatch):
    class EmptyProjectDatabase:
        @staticmethod
        def count_documents(project_id):
            assert project_id == 7
            return 0

    monkeypatch.setattr(web, "SERVICE", type("Service", (), {"db": EmptyProjectDatabase()})())

    result = _strategy_statistics(
        7,
        KNOWLEDGE_FORGE_STRATEGY,
        {"stats": {"documents": 2, "chunks": 15, "entities": 266, "relations": 9}},
    )

    assert result == {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}


def test_enhanced_statistics_hide_stale_entity_relation_snapshot(monkeypatch):
    class StaleProjectDatabase:
        @staticmethod
        def count_documents(project_id):
            assert project_id == 7
            return 1

        @staticmethod
        def statistics(project_id):
            assert project_id == 7
            return {"documents": 1, "chunks": 8, "entities": 99, "relations": 42}

    monkeypatch.setattr(web, "SERVICE", type("Service", (), {"db": StaleProjectDatabase()})())

    result = _strategy_statistics(
        7,
        KNOWLEDGE_FORGE_STRATEGY,
        {
            "status": "stale",
            "generation": 5,
            "ready_generation": 4,
            "stats": {"documents": 2, "chunks": 15, "entities": 266, "relations": 9},
        },
    )

    assert result == {"documents": 1, "chunks": 8, "entities": 0, "relations": 0}


def test_enhanced_statistics_keep_ready_current_snapshot(monkeypatch):
    class ReadyProjectDatabase:
        @staticmethod
        def count_documents(project_id):
            assert project_id == 7
            return 1

        @staticmethod
        def statistics(project_id):
            assert project_id == 7
            return {"documents": 1, "chunks": 8, "entities": 99, "relations": 42}

    monkeypatch.setattr(web, "SERVICE", type("Service", (), {"db": ReadyProjectDatabase()})())

    result = _strategy_statistics(
        7,
        KNOWLEDGE_FORGE_STRATEGY,
        {
            "status": "ready",
            "generation": 4,
            "ready_generation": 4,
            "stats": {"documents": 2, "chunks": 15, "entities": 266, "relations": 9},
        },
    )

    assert result == {"documents": 2, "chunks": 15, "entities": 266, "relations": 9}


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

    html = _document_rows([document], KNOWLEDGE_FORGE_STRATEGY)

    assert "等待文档预处理与图谱构建" in html
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

    def get_project_strategy_graph(self, project_id, strategy_key):
        return {"status": "ready"}


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
        7, strategy_key=KNOWLEDGE_FORGE_STRATEGY,
    )

    assert len(calls) == 1
    assert calls[0][0] == 11
    assert calls[0][1]["target_document_ids"] == [11, 12]
    assert result["job_ids"] == ["job-enhanced"]
    assert result["document_count"] == 2


def test_stale_enhanced_graph_requeues_indexed_documents_after_policy_change():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = _ProjectBuildDatabase()
    service.db.documents = [
        {"id": 11, "strategy_status": "indexed"},
        {"id": 12, "strategy_status": "indexed"},
    ]
    service.db.get_project_strategy_graph = (
        lambda _project_id, _strategy_key: {"status": "stale"}
    )
    calls = []

    def submit(document_id, **values):
        calls.append((document_id, values))
        return "job-enhanced-rebuild"

    service.submit_ingestion = submit

    result = service.submit_project_ingestion(
        7, strategy_key=KNOWLEDGE_FORGE_STRATEGY,
    )

    assert calls == [(
        11,
        {
            "batch_id": result["batch_id"],
            "strategy_key": KNOWLEDGE_FORGE_STRATEGY,
            "target_document_ids": [11, 12],
        },
    )]
    assert result["document_count"] == 2


def test_ready_enhanced_graph_is_requeued_when_relation_treatment_changes():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)

    class RelationFreshnessDatabase(_ProjectBuildDatabase):
        def __init__(self):
            super().__init__()
            self.documents = [
                {"id": 11, "strategy_status": "indexed"},
                {"id": 12, "strategy_status": "indexed"},
            ]
            self.graph_status = "ready"
            self.reconciled = []

        def reconcile_knowledge_forge_relation_freshness(
            self, project_id, *, code_digest, treatment_digest,
        ):
            self.reconciled.append((
                project_id, code_digest, treatment_digest,
            ))
            self.graph_status = "stale"
            return {"status": "stale"}

        def get_project_strategy_graph(self, project_id, strategy_key):
            return {"status": self.graph_status}

    service.db = RelationFreshnessDatabase()
    calls = []

    def submit(document_id, **values):
        calls.append((document_id, values))
        return "job-relation-policy-refresh"

    service.submit_ingestion = submit
    result = service.submit_project_ingestion(
        7, strategy_key=KNOWLEDGE_FORGE_STRATEGY,
    )

    assert len(service.db.reconciled) == 1
    assert service.db.reconciled[0][0] == 7
    assert all(service.db.reconciled[0][1:])
    assert calls == [(
        11,
        {
            "batch_id": result["batch_id"],
            "strategy_key": KNOWLEDGE_FORGE_STRATEGY,
            "target_document_ids": [11, 12],
        },
    )]
    assert result["document_count"] == 2


def test_ready_enhanced_graph_is_requeued_when_entity_treatment_changes():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)

    class EntityFreshnessDatabase(_ProjectBuildDatabase):
        def __init__(self):
            super().__init__()
            self.documents = [
                {"id": 11, "strategy_status": "indexed"},
                {"id": 12, "strategy_status": "indexed"},
            ]
            self.graph_status = "ready"
            self.reconciled = []

        def reconcile_knowledge_forge_entity_freshness(
            self, project_id, *, code_version, policy_set,
        ):
            self.reconciled.append((project_id, code_version, policy_set))
            self.graph_status = "stale"
            return {"status": "stale"}

        def get_project_strategy_graph(self, project_id, strategy_key):
            return {"status": self.graph_status}

    service.db = EntityFreshnessDatabase()
    calls = []

    def submit(document_id, **values):
        calls.append((document_id, values))
        return "job-entity-policy-refresh"

    service.submit_ingestion = submit
    result = service.submit_project_ingestion(
        7, strategy_key=KNOWLEDGE_FORGE_STRATEGY,
    )

    assert service.db.reconciled == [(
        7,
        GROUNDED_ENTITY_CODE_VERSION,
        _grounded_entity_freshness_policy_set(),
    )]
    assert calls == [(
        11,
        {
            "batch_id": result["batch_id"],
            "strategy_key": KNOWLEDGE_FORGE_STRATEGY,
            "target_document_ids": [11, 12],
        },
    )]
    assert result["document_count"] == 2


def test_enhanced_build_freezes_active_schema_without_running_governance(monkeypatch):
    monkeypatch.setattr(Config, "schema_discovery_during_build", False)
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)

    class Schemas:
        def ensure_base(self, project_id):
            assert project_id == 7
            return {"version": 4, "schema_json": {"relation_types": []}}

        def evaluate_corpus(self, _project, _documents):
            raise AssertionError("ordinary build must not run Schema discovery")

    service.schemas = Schemas()
    result = service._knowledge_forge_build_schema(
        {"id": 7}, [{"id": 11}, {"id": 12}],
    )

    assert result["version"] == 4
    assert result["_evaluation"] == {
        "scope": "project_build_active_schema_snapshot",
        "reviewed": False,
        "reason": "deferred_to_explicit_schema_reassessment",
        "document_ids": [11, 12],
    }


def test_enhanced_build_can_explicitly_enable_schema_discovery(monkeypatch):
    monkeypatch.setattr(Config, "schema_discovery_during_build", True)
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)

    class Schemas:
        def evaluate_corpus(self, project, documents):
            assert project["id"] == 7
            assert [item["id"] for item in documents] == [11]
            return {"version": 5, "schema_json": {"relation_types": []}}

    service.schemas = Schemas()

    assert service._knowledge_forge_build_schema(
        {"id": 7}, [{"id": 11}],
    )["version"] == 5


def test_ready_enhanced_graph_is_invalidated_by_stale_shared_content(monkeypatch):
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)

    class DatabaseStub:
        def __init__(self):
            self.stale = []

        def get_project_strategy_graph(self, project_id, strategy_key):
            return {
                "project_id": project_id,
                "strategy_key": strategy_key,
                "status": "ready",
            }

        def list_documents(self, project_id):
            return [{
                "id": 11,
                "project_id": project_id,
                "status": "indexed",
                "active_revision_id": "revision-11",
                "is_deleted": False,
            }]

        def mark_strategy_graph_stale(self, project_id, strategy_key, reason):
            self.stale.append((project_id, strategy_key, reason))
            return {"status": "stale"}

    service.db = DatabaseStub()
    monkeypatch.setattr(
        service, "_document_revision_manifest",
        lambda _document, _project: {"config_hash": "current"},
    )
    monkeypatch.setattr(
        service, "_shared_content_is_current",
        lambda _document, _manifest: False,
    )

    result = service._reconcile_knowledge_forge_content_freshness({"id": 7})

    assert result == {"status": "stale"}
    assert service.db.stale[0][:2] == (7, KNOWLEDGE_FORGE_STRATEGY)
    assert "1 个文档" in service.db.stale[0][2]


def test_project_preparation_queues_knowledge_forge_content_job():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = _PreparationDatabase()
    service.db.assert_project_construction_strategy = (
        lambda _project_id, _expected=None: KNOWLEDGE_FORGE_STRATEGY
    )
    service.executor = _ImmediateRecorder()

    result = service.submit_project_preparation(7)

    assert result["job_ids"] == ["prepare-21"]
    assert result["document_count"] == 1
    assert service.executor.calls == [
        (
            "_prepare_document_content",
            ("prepare-21", 21, KNOWLEDGE_FORGE_STRATEGY),
        )
    ]


def test_project_preparation_status_and_submission_reuse_current_old_documents():
    class IncrementalPreparationDatabase(_PreparationDatabase):
        def __init__(self):
            super().__init__()
            self.documents = [
                {
                    **self.documents[0],
                    "id": 21,
                    "original_filename": "existing.pdf",
                    "doc_hash": "existing-hash",
                    "active_revision_id": "revision-existing",
                    "prepared_chunk_count": 4,
                },
                {
                    **self.documents[0],
                    "id": 22,
                    "original_filename": "new.pdf",
                    "doc_hash": "new-hash",
                    "active_revision_id": None,
                    "prepared_chunk_count": 0,
                },
            ]
            self.created_document_ids = []

        def create_job(
            self, kind, project_id, document_id, *, metadata,
            strategy_key=STABLE_STRATEGY,
        ):
            assert kind == "content_preparation"
            self.created_document_ids.append(document_id)
            return {"id": f"prepare-{document_id}"}

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = IncrementalPreparationDatabase()
    service.db.assert_project_construction_strategy = (
        lambda _project_id, _expected=None: KNOWLEDGE_FORGE_STRATEGY
    )
    service.executor = _ImmediateRecorder()
    existing = service.db.documents[0]
    manifest = service._document_revision_manifest(
        existing, service.db.get_project(7),
    )
    existing.update({
        "active_revision_status": "active",
        "active_revision_content_hash": manifest["content_hash"],
        "active_revision_source_hash": manifest["source_hash"],
        "active_revision_config_hash": manifest["config_hash"],
    })

    status = service.project_preparation_status(7)
    result = service.submit_project_preparation(7)

    assert status["document_count"] == 2
    assert status["ready_count"] == 1
    assert status["pending_count"] == 1
    assert status["new_count"] == 1
    assert status["refresh_count"] == 0
    assert status["pending_documents"] == [{
        "id": 22,
        "original_filename": "new.pdf",
        "reason": "new_document",
    }]
    assert result["job_ids"] == ["prepare-22"]
    assert service.db.created_document_ids == [22]
    assert service.executor.calls == [(
        "_prepare_document_content",
        ("prepare-22", 22, KNOWLEDGE_FORGE_STRATEGY),
    )]


def test_preparation_notice_distinguishes_new_and_outdated_documents():
    html = _preparation_update_notice(
        {
            "pending_count": 3,
            "new_count": 1,
            "refresh_count": 2,
            "pending_documents": [
                {"id": 1, "original_filename": "new.pdf"},
                {"id": 2, "original_filename": "changed.pdf"},
                {"id": 3, "original_filename": "missing.pdf"},
            ],
        },
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
    )

    assert "有 3 份文档等待预处理" in html
    assert "新文档 1 份，需要更新 2 份" in html
    assert 'title="new.pdf、changed.pdf、missing.pdf"' in html


def test_project_preparation_queues_native_source_job_without_chunk_contract():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = _PreparationDatabase()
    service.db.assert_project_construction_strategy = (
        lambda _project_id, _expected=None: GRAPHRAG_NATIVE_STRATEGY
    )
    service.executor = _ImmediateRecorder()
    service._mark_graphrag_native_source_changed = lambda *_args: None

    result = service.submit_project_preparation(7)

    assert result["job_ids"] == ["prepare-21"]
    assert service.executor.calls == [
        (
            "_prepare_document_content",
            ("prepare-21", 21, GRAPHRAG_NATIVE_STRATEGY),
        )
    ]


def test_native_source_readiness_does_not_require_knowledge_forge_chunks():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = _PreparationDatabase()
    service.db.documents[0].update({
        "active_revision_id": "native-revision",
        "structure_ready_at": "now",
    })
    service.db.assert_project_construction_strategy = (
        lambda _project_id, _expected=None: GRAPHRAG_NATIVE_STRATEGY
    )
    service._shared_content_is_current = lambda *_args: True

    with pytest.raises(ValueError, match="均已完成预处理"):
        service.submit_project_preparation(7)


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


def test_grounded_policy_constants_are_converted_before_json_serialization():
    payload = {"relation_policy_set": dict(RELATION_CANDIDATE_POLICY_SET)}

    assert json.loads(json.dumps(payload))["relation_policy_set"] == dict(
        RELATION_CANDIDATE_POLICY_SET
    )


def test_multi_strategy_migration_contains_bounded_strategy_contract():
    sql = MULTI_STRATEGY_CONSTRUCTION_SQL
    assert "CREATE TABLE IF NOT EXISTS document_strategy_builds" in sql
    assert "PRIMARY KEY(document_id,strategy_key)" in sql
    assert "CREATE TABLE IF NOT EXISTS project_strategy_graphs" in sql
    assert "PRIMARY KEY(project_id,strategy_key)" in sql
    # Historical migration content must not be rewritten: 0030 performs the
    # durable rename for existing and newly bootstrapped databases.
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
            (project["id"], KNOWLEDGE_FORGE_STRATEGY),
        )
        connection.execute(
            """UPDATE project_strategy_graphs SET status='ready',generation=4,
                      ready_generation=4,
                      stats='{"documents":2,"chunks":15,"entities":266,"relations":9}'::jsonb
               WHERE project_id=%s AND strategy_key=%s""",
            (project["id"], KNOWLEDGE_FORGE_STRATEGY),
        )

    database.invalidate_knowledge_forge_for_document(
        int(documents[0]["id"]), reason="stable revision changed",
    )
    with database.connection() as connection:
        builds = connection.execute(
            """SELECT document_id,status,content_revision_id
               FROM document_strategy_builds
               WHERE project_id=%s AND strategy_key=%s ORDER BY document_id""",
            (project["id"], KNOWLEDGE_FORGE_STRATEGY),
        ).fetchall()
    assert [row["status"] for row in builds] == ["uploaded", "uploaded"]
    assert all(row["content_revision_id"] is None for row in builds)
    graph = database.get_project_strategy_graph(
        int(project["id"]), KNOWLEDGE_FORGE_STRATEGY,
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
