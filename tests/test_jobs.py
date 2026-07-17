from contextlib import contextmanager

from knowledge_forge.database import Database, SCHEMA_SQL
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


def test_project_parser_defaults_are_removed_by_schema_migration():
    assert "DROP COLUMN IF EXISTS default_parser" in SCHEMA_SQL
    assert "DROP COLUMN IF EXISTS default_chunk_strategy" in SCHEMA_SQL
    assert "ADD COLUMN IF NOT EXISTS default_parser" not in SCHEMA_SQL
    assert "ADD COLUMN IF NOT EXISTS default_chunk_strategy" not in SCHEMA_SQL


def test_project_token_settings_and_ragflow_cleanup_are_migrated():
    assert "model_max_tokens INTEGER NOT NULL DEFAULT 32768" in SCHEMA_SQL
    assert "chunk_max_tokens INTEGER NOT NULL DEFAULT 512" in SCHEMA_SQL
    assert "DROP COLUMN IF EXISTS ragflow_dataset_id" in SCHEMA_SQL
    assert "DROP COLUMN IF EXISTS ragflow_document_id" in SCHEMA_SQL
    assert "DROP COLUMN IF EXISTS ragflow_chunk_id" in SCHEMA_SQL
    assert "DROP COLUMN IF EXISTS ragflow_task_id" in SCHEMA_SQL


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
