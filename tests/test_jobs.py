from contextlib import contextmanager

from knowledge_forge.database import Database, SCHEMA_SQL


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
