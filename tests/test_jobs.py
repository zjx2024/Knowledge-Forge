from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.graph_policy import GRAPH_STALE
from knowledge_forge.graph_safety import GraphMutationContext
from knowledge_forge.service import KnowledgeForgeService


class FakeCursor:
    def __init__(self, row):
        self.row = row
        self.query = ""
        self.params = []

    def execute(self, query, params):
        self.query = query
        self.params = params
        return self

    def fetchone(self):
        return self.row


class FakeDatabase(Database):
    def __init__(self, row):
        self.cursor = FakeCursor(row)

    @contextmanager
    def connection(self):
        yield self.cursor


def test_latest_active_job_ignores_completed_jobs_and_prefers_running():
    expected = {"id": "running-job", "status": "running"}
    database = FakeDatabase(expected)

    result = database.latest_active_job(7, "ingestion")

    assert result == expected
    assert "status IN ('queued','running')" in database.cursor.query
    assert "CASE WHEN status='running' THEN 0 ELSE 1 END" in database.cursor.query
    assert database.cursor.params == [7, "ingestion"]


def test_creating_project_job_marks_project_as_recently_active():
    class Result:
        def __init__(self, row=None):
            self.row = row

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self):
            self.calls = []

        def execute(self, query, params):
            self.calls.append((query, params))
            if "INSERT INTO jobs" in query:
                return Result({"id": "job-1", "project_id": 7})
            return Result()

    class JobDatabase(Database):
        def __init__(self):
            self.conn = Connection()

        @contextmanager
        def connection(self):
            yield self.conn

    database = JobDatabase()

    job = database.create_job("ingestion", 7, 11, metadata={"batch_id": "batch-1"})

    assert job == {"id": "job-1", "project_id": 7}
    assert len(database.conn.calls) == 2
    assert "INSERT INTO jobs" in database.conn.calls[0][0]
    assert "UPDATE projects SET updated_at=NOW()" in database.conn.calls[1][0]
    assert database.conn.calls[1][1] == (7,)


def test_project_parser_defaults_are_not_reintroduced_or_destructively_dropped():
    assert "DROP COLUMN IF EXISTS default_parser" not in SCHEMA_SQL
    assert "DROP COLUMN IF EXISTS default_chunk_strategy" not in SCHEMA_SQL
    assert "ADD COLUMN IF NOT EXISTS default_parser" not in SCHEMA_SQL
    assert "ADD COLUMN IF NOT EXISTS default_chunk_strategy" not in SCHEMA_SQL


def test_project_token_settings_are_additive_and_legacy_cleanup_is_explicit():
    assert "model_max_tokens INTEGER NOT NULL DEFAULT 32768" in SCHEMA_SQL
    assert "chunk_max_tokens INTEGER NOT NULL DEFAULT 512" in SCHEMA_SQL
    assert "DROP COLUMN" not in SCHEMA_SQL.upper()


def test_document_pipeline_stages_are_persisted_and_backfilled():
    assert "processing_stage TEXT NOT NULL DEFAULT 'uploaded'" in SCHEMA_SQL
    assert "structure_ready_at TIMESTAMPTZ" in SCHEMA_SQL
    assert "chunks_ready_at TIMESTAMPTZ" in SCHEMA_SQL
    assert "graph_ready_at TIMESTAMPTZ" in SCHEMA_SQL
    assert "EXISTS(SELECT 1 FROM chunks" in SCHEMA_SQL


def test_extraction_evidence_and_verification_columns_are_migrated():
    assert "entity_mentions ADD COLUMN IF NOT EXISTS evidence_text" in SCHEMA_SQL
    assert "relation_assertions ADD COLUMN IF NOT EXISTS evidence_text" in SCHEMA_SQL
    assert "relation_assertions ADD COLUMN IF NOT EXISTS verification_status" in SCHEMA_SQL


def test_restart_recovers_all_durable_job_kinds_and_orphaned_graph_generation():
    class ImmediateExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, callback, *args):
            self.calls.append((callback.__name__, args))

    class RecoveryDatabase:
        def __init__(self):
            self.updates = []
            self.cleanup_calls = 0

        def list_recoverable_jobs(self):
            return [
                {"id": "ingest", "kind": "ingestion", "document_id": 11, "metadata": {}},
                {"id": "graph", "kind": "graph_sync", "project_id": 7, "metadata": {}},
                {"id": "schema", "kind": "schema_reassessment", "project_id": 7, "metadata": {}},
                {"id": "cleanup", "kind": "document_cleanup", "document_id": 12, "metadata": {}},
                {"id": "invalid", "kind": "ingestion", "metadata": {}},
            ]

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

        def get_document_including_deleted(self, document_id):
            return {"id": document_id, "project_id": 7}

        def list_orphaned_graph_projects(self):
            return [{"id": 8}]

        def create_graph_rebuild_job(self, project_id):
            return {"id": f"graph-{project_id}", "project_id": project_id}

        def cleanup_indexing_failures(self):
            self.cleanup_calls += 1

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = RecoveryDatabase()
    service.executor = ImmediateExecutor()
    service.cleanup_executor = ImmediateExecutor()
    service._ingest_document = lambda *_args: None
    service._sync_project_graph = lambda *_args: None
    service._reassess_project_schema = lambda *_args: None
    service._cleanup_deleted_document = lambda *_args: None

    recovered = service.recover_interrupted_jobs()

    assert recovered == 5
    assert service.executor.calls == [
        ("<lambda>", ("ingest", 11)),
        ("<lambda>", ("graph", 7)),
        ("<lambda>", ("schema", 7)),
        ("<lambda>", ("graph-8", 8)),
    ]
    assert service.cleanup_executor.calls == [("<lambda>", ("cleanup", {"id": 12, "project_id": 7}))]
    reset = {job_id: values for job_id, values in service.db.updates if values.get("status") == "queued"}
    assert set(reset) == {"ingest", "graph", "schema", "cleanup", "invalid"}
    assert all(values["metadata"]["recovered_after_restart"] is True for values in reset.values())
    assert any(job_id == "invalid" and values.get("status") == "failed"
               for job_id, values in service.db.updates)
    assert service.db.cleanup_calls == 1


def test_restart_does_not_create_stable_graph_job_for_fixed_strategy_project():
    class ImmediateExecutor:
        def __init__(self):
            self.calls = []

        def submit(self, callback, *args):
            self.calls.append((callback.__name__, args))

    class FixedStrategyRecoveryDatabase:
        def __init__(self):
            self.created = []

        def list_recoverable_jobs(self):
            return []

        def list_orphaned_graph_projects(self):
            return [{"id": 8}]

        def get_project_construction_strategy_selection(self, project_id):
            assert project_id == 8
            return {
                "project_id": 8,
                "association_status": "bound",
                "selected_strategy": "knowledge_forge",
            }

        def create_graph_rebuild_job(self, project_id):
            self.created.append(project_id)
            raise AssertionError("fixed strategy must not create a Stable graph job")

        def cleanup_indexing_failures(self):
            return None

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = FixedStrategyRecoveryDatabase()
    service.executor = ImmediateExecutor()

    assert service.recover_interrupted_jobs() == 0
    assert service.db.created == []
    assert service.executor.calls == []


def test_restart_acknowledges_pending_cancellation_without_resubmitting():
    class NoSubmitExecutor:
        def submit(self, *_args):
            raise AssertionError("cancel-requested job must not be resubmitted")

    class CancellationRecoveryDatabase:
        def __init__(self):
            self.updates = []

        def list_recoverable_jobs(self):
            return [{
                "id": "stopping", "kind": "content_preparation",
                "project_id": 7, "document_id": 11,
                "metadata": {"cancel_requested": True},
            }]

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

        def list_orphaned_graph_projects(self):
            return []

        def cleanup_indexing_failures(self):
            return None

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = CancellationRecoveryDatabase()
    service.executor = NoSubmitExecutor()

    assert service.recover_interrupted_jobs() == 0
    assert service.db.updates[0][1]["status"] == "cancelled"
    assert service.db.updates[0][1]["metadata"]["stage"] == "cancelled"


def test_restart_finishes_job_when_publication_preceded_cancel_tail():
    class NoSubmitExecutor:
        def submit(self, *_args):
            raise AssertionError("published job must not be rerun")

    class PublishedRecoveryDatabase:
        def __init__(self):
            self.updates = []

        def list_recoverable_jobs(self):
            return [{
                "id": "published", "kind": "content_preparation",
                "project_id": 7, "document_id": 11,
                "metadata": {
                    "cancel_requested": True,
                    "publication_committed": True,
                },
            }]

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

        def list_orphaned_graph_projects(self):
            return []

        def cleanup_indexing_failures(self):
            return None

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = PublishedRecoveryDatabase()
    service.executor = NoSubmitExecutor()

    assert service.recover_interrupted_jobs() == 0
    assert service.db.updates[0][1]["status"] == "completed"
    assert service.db.updates[0][1]["metadata"]["stage"] == "completed"


def test_restart_abandons_unpublished_document_revision_before_cancel_ack():
    class NoSubmitExecutor:
        def submit(self, *_args):
            raise AssertionError("cancel-requested job must not be rerun")

    class RecoveryDatabase:
        def __init__(self):
            self.updates = []
            self.abandoned = []

        def list_recoverable_jobs(self):
            return [{
                "id": "staged", "kind": "content_preparation",
                "project_id": 7, "document_id": 11,
                "metadata": {"cancel_requested": True},
            }]

        def abandon_job_document_revision(self, job_id, *, reason):
            self.abandoned.append((job_id, reason))
            return {
                "revision_id": "revision_staged",
                "project_id": 7,
                "document_id": 11,
                "content_hash": "a" * 64,
                "manifest": {},
            }

        def get_project(self, project_id):
            return {"id": project_id, "name": "test"}

        def get_document_including_deleted(self, document_id):
            return {
                "id": document_id,
                "active_revision_id": None,
                "processing_stage": "structure",
                "metadata": {},
                "is_deleted": False,
            }

        def update_document(self, document_id, **values):
            self.updates.append((f"document-{document_id}", values))

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

        def list_orphaned_graph_projects(self):
            return []

        def cleanup_indexing_failures(self):
            return None

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = RecoveryDatabase()
    service.executor = NoSubmitExecutor()

    assert service.recover_interrupted_jobs() == 0
    assert service.db.abandoned[0][0] == "staged"
    restored = dict(service.db.updates)["document-11"]
    assert restored["status"] == "uploaded"
    assert restored["processing_stage"] == "uploaded"
    assert dict(service.db.updates)["staged"]["status"] == "cancelled"


def test_restart_closes_and_cleans_unpublished_native_attempt(tmp_path):
    class NoSubmitExecutor:
        def submit(self, *_args):
            raise AssertionError("cancel-requested native job must not be rerun")

    class RecoveryDatabase:
        def __init__(self):
            self.updates = []
            self.closed = []

        def list_recoverable_jobs(self):
            return [{
                "id": "native-staged", "kind": "graphrag_native_build",
                "project_id": 7,
                "metadata": {"cancel_requested": True},
            }]

        def cancel_unpublished_graphrag_native_runs(self, project_id, *, reason):
            self.closed.append((project_id, reason))
            return [{
                "run_id": "grn_staged",
                "staging_relative_path": ".staging/grn_staged-attempt",
                "bundle_relative_path": "",
            }]

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

        def list_orphaned_graph_projects(self):
            return []

        def cleanup_indexing_failures(self):
            return None

    native_root = tmp_path / "graphrag_native"
    staging = native_root / ".staging" / "grn_staged-attempt"
    staging.mkdir(parents=True)
    (staging / "partial.txt").write_text("partial", encoding="utf-8")
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = RecoveryDatabase()
    service.executor = NoSubmitExecutor()

    def discard(path):
        assert Path(path) == staging
        import shutil
        shutil.rmtree(path)

    service.graphrag_native = SimpleNamespace(
        native_root=native_root,
        _discard_cancelled_bundle_path=discard,
    )

    assert service.recover_interrupted_jobs() == 0
    assert service.db.closed[0][0] == 7
    assert not staging.exists()
    assert dict(service.db.updates)["native-staged"]["status"] == "cancelled"


def test_restart_releases_cancelled_strategy_graph_ownership_before_ack():
    class NoSubmitExecutor:
        def submit(self, *_args):
            raise AssertionError("cancel-requested graph job must not be rerun")

    class RecoveryDatabase:
        def __init__(self):
            self.updates = []
            self.released = []
            self.cancelled_graphs = []

        def list_recoverable_jobs(self):
            return [{
                "id": "graph-staged", "kind": "ingestion",
                "project_id": 7, "document_id": 11,
                "strategy_key": "knowledge_forge",
                "metadata": {"cancel_requested": True},
            }]

        def abandon_job_document_revision(self, *_args, **_kwargs):
            return None

        def release_job_graph_leases(self, job_id):
            self.released.append(job_id)
            return 1

        def get_project_strategy_graph(self, project_id, strategy_key):
            return {
                "project_id": project_id, "strategy_key": strategy_key,
                "status": "rebuilding", "generation": 4,
            }

        def cancel_strategy_graph_rebuild(
            self, project_id, strategy_key, generation, reason,
        ):
            self.cancelled_graphs.append(
                (project_id, strategy_key, generation, reason),
            )

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

        def list_orphaned_graph_projects(self):
            return []

        def cleanup_indexing_failures(self):
            return None

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = RecoveryDatabase()
    service.executor = NoSubmitExecutor()

    assert service.recover_interrupted_jobs() == 0
    assert service.db.released == ["graph-staged"]
    assert service.db.cancelled_graphs[0][:3] == (
        7, "knowledge_forge", 4,
    )
    assert dict(service.db.updates)["graph-staged"]["status"] == "cancelled"


def test_restart_fences_cancelled_shared_graph_writer_before_ack():
    class NoSubmitExecutor:
        def submit(self, *_args):
            raise AssertionError("cancel-requested graph job must not be rerun")

    class RecoveryDatabase:
        def __init__(self):
            self.updates = []
            self.released = []
            self.states = []

        def list_recoverable_jobs(self):
            return [{
                "id": "stable-graph-staged", "kind": "graph_sync",
                "project_id": 7, "document_id": None,
                "strategy_key": "stable",
                "metadata": {"cancel_requested": True},
            }]

        def release_job_graph_leases(self, job_id):
            self.released.append(job_id)
            return 1

        def set_graph_state(self, project_id, status, **values):
            self.states.append((project_id, status, values))

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

        def list_orphaned_graph_projects(self):
            return []

        def cleanup_indexing_failures(self):
            return None

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = RecoveryDatabase()
    service.executor = NoSubmitExecutor()

    assert service.recover_interrupted_jobs() == 0
    assert service.db.released == ["stable-graph-staged"]
    assert service.db.states == [(
        7,
        GRAPH_STALE,
        {"error": "服务恢复时确认图谱构建已终止；未完成图谱未发布"},
    )]
    assert dict(service.db.updates)["stable-graph-staged"]["status"] == "cancelled"


def test_preparation_publication_race_cleans_revision_after_context_exit():
    """The implementation must not strand a building revision at its CAS."""

    import inspect

    source = inspect.getsource(KnowledgeForgeService._prepare_shared_content)
    cleanup = source.index("if revision and not published:")
    failure = source.index("self.db.fail_document_revision(", cleanup)

    assert failure > cleanup
    assert "if revision and revision_scope is not None:" not in source[cleanup:failure]


def test_cancel_acknowledgement_can_close_only_the_active_subtask():
    row = {
        "id": "stopping", "status": "running", "progress": 63,
        "log": ["work", "stop requested"], "error_message": None,
        "metadata": {
            "cancel_requested": True,
            "cancel_reason": "用户请求安全终止图谱构建",
            "stage": "cancelling",
            "active_subtask": "extraction",
            "subtasks": {
                "extraction": {
                    "status": "running", "progress": 43,
                    "detail": "实体抽取 227/233 个知识块",
                },
                "graph_sync": {
                    "status": "waiting", "progress": 0, "detail": "",
                },
            },
        },
    }
    database = FakeDatabase(row)

    database.update_job(
        "stopping", status="cancelled", progress=100,
        log="已安全终止；未完成结果未发布",
        metadata={
            "stage": "cancelled",
            "cancel_acknowledged_at": "2026-08-13T12:00:00+00:00",
            "active_subtask": "untrusted-change",
            "subtasks": {
                "extraction": {
                    "status": "cancelled", "progress": 43,
                    "detail": "用户请求安全终止图谱构建",
                },
                "graph_sync": {
                    "status": "completed", "progress": 100,
                    "detail": "must not be accepted",
                },
            },
        },
    )

    params = database.cursor.params
    import json
    persisted = json.loads(params[4])
    assert params[0] == "cancelled"
    assert persisted["stage"] == "cancelled"
    assert persisted["cancel_acknowledged_at"]
    assert persisted["active_subtask"] == "extraction"
    assert persisted["subtasks"]["extraction"]["status"] == "cancelled"
    assert persisted["subtasks"]["graph_sync"]["status"] == "waiting"


def test_cancelled_subtask_snapshot_closes_the_running_step_without_db_progress_write():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    job = {
        "metadata": {
            "active_subtask": "extraction",
            "subtasks": {
                "extraction": {
                    "status": "running", "progress": 43, "detail": "227/233",
                },
                "graph_sync": {
                    "status": "waiting", "progress": 0, "detail": "",
                },
            },
        },
    }

    closed = service._cancelled_subtasks(job, "用户请求安全终止图谱构建")

    assert closed["extraction"] == {
        "status": "cancelled", "progress": 43,
        "detail": "用户请求安全终止图谱构建",
    }
    assert closed["graph_sync"]["status"] == "waiting"


def test_queued_graph_sync_cancellation_does_not_stale_an_untouched_graph():
    class QueuedCancellationDatabase:
        def __init__(self):
            self.states = []
            self.updates = []

        def get_job(self, job_id):
            assert job_id == "queued-graph"
            return {
                "id": job_id, "project_id": 7, "status": "cancelled",
                "metadata": {
                    "cancel_requested": True,
                    "cancel_reason": "用户请求安全终止图谱构建",
                },
            }

        def begin_graph_mutation(self, _job_id):
            raise AssertionError("cancelled queued job must not acquire graph ownership")

        def set_graph_state(self, project_id, status, **values):
            self.states.append((project_id, status, values))

        def update_job(self, job_id, **values):
            self.updates.append((job_id, values))

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = QueuedCancellationDatabase()

    service._sync_project_graph("queued-graph", 7)

    assert service.db.states == []
    assert service.db.updates[-1][1]["status"] == "cancelled"


def test_knowledge_forge_strategy_finalizer_discards_manifest_and_releases_lease():
    """The actual strategy worker owns both in-process and durable leases."""

    mutation = GraphMutationContext(
        namespace_id="00000000-0000-0000-0000-000000000007",
        project_id=7,
        generation=3,
        policy_hash="policy",
        lease_token="lease-token",
    )

    class CancellationDatabase:
        def __init__(self):
            self.released = []

        def release_graph_lease(self, lease_id, lease_token):
            self.released.append((lease_id, lease_token))

    class Graph:
        def __init__(self):
            self.discarded = []

        def discard_writer_manifest(self, value):
            self.discarded.append(value)

    database = CancellationDatabase()
    graph = Graph()
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = database
    service.graph = graph

    service._release_strategy_graph_resources(
        mutation,
        {"id": "lease-id", "lease_token": "lease-token"},
    )

    assert graph.discarded == [mutation]
    assert database.released == [("lease-id", "lease-token")]

    # Keep the ownership helper wired into the real worker's unconditional
    # finalizer instead of leaving it as an unused utility.
    import inspect
    worker_source = inspect.getsource(
        KnowledgeForgeService._ingest_knowledge_forge_strategy,
    )
    assert "self._release_strategy_graph_resources(mutation, lease)" in worker_source
