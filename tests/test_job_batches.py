from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from knowledge_forge.database import Database
from knowledge_forge.service import KnowledgeForgeService


class BatchCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = []

    def execute(self, query, params):
        self.query = query
        self.params = list(params)
        return self

    def fetchall(self):
        return self.rows


class BatchDatabase(Database):
    def __init__(self, rows):
        self.cursor = BatchCursor(rows)

    @contextmanager
    def connection(self):
        yield self.cursor


def test_ingestion_batch_query_returns_every_document_in_submission_order():
    rows = [{"id": "first"}, {"id": "second"}]
    database = BatchDatabase(rows)

    result = database.list_ingestion_batch_jobs("batch-1")

    assert result == rows
    assert "j.kind IN ('ingestion','content_preparation')" in database.cursor.query
    assert "j.metadata->>'batch_id'=%s" in database.cursor.query
    assert "ORDER BY j.created_at,j.id" in database.cursor.query
    assert database.cursor.params == ["batch-1"]


class CancellingCursor:
    def __init__(self):
        self.updates = []

    def execute(self, query, params):
        if query.lstrip().startswith("SELECT id,log,metadata FROM jobs"):
            assert params == (41,)
            return BatchCursor([
                {"id": "queued", "log": ["queued"], "metadata": {}},
                {"id": "running", "log": [], "metadata": {"batch_id": "batch-1"}},
            ])
        self.updates.append((query, params))
        return self


class CancellingDatabase(Database):
    def __init__(self):
        self.cursor = CancellingCursor()

    @contextmanager
    def connection(self):
        yield self.cursor


def test_cancel_document_jobs_updates_every_active_job_and_returns_count():
    database = CancellingDatabase()

    count = database.cancel_document_jobs(41, "deleted")

    assert count == 2
    assert len(database.cursor.updates) == 2
    for query, params in database.cursor.updates:
        assert "status='cancelled'" in query
        assert "completed_at=NOW()" in query
        assert params[2] in {"queued", "running"}
        assert "deleted" in params[0]
        assert '"cancelled_by_document_delete": true' in params[1]


class ServiceBatchDatabase:
    def __init__(self, rows):
        self.rows = rows

    def list_ingestion_batch_jobs(self, batch_id):
        assert batch_id == "batch-1"
        return self.rows


def _job(job_id, name, status, created_at, *, completed_at=None, progress=0, log=None, subtasks=None, kind="ingestion"):
    return {
        "id": job_id,
        "kind": kind,
        "status": status,
        "progress": progress,
        "created_at": created_at,
        "completed_at": completed_at,
        "document_name": name,
        "log": log or [],
        "metadata": {"batch_id": "batch-1", "subtasks": subtasks or {}},
    }


def test_batch_snapshot_keeps_completed_document_logs_while_next_document_runs():
    started = datetime.now(timezone.utc) - timedelta(minutes=4)
    completed_steps = {
        key: {"status": "completed", "progress": 100, "detail": ""}
        for key in KnowledgeForgeService.INGESTION_SUBTASKS
    }
    running_steps = {
        "structure": {"status": "completed", "progress": 100, "detail": ""},
        "chunking": {"status": "running", "progress": 50, "detail": ""},
    }
    rows = [
        _job(
            "first", "first.pdf", "completed", started,
            completed_at=started + timedelta(minutes=2), progress=100,
            log=["first started", "first completed"], subtasks=completed_steps,
        ),
        _job(
            "second", "second.pdf", "running", started + timedelta(seconds=1),
            progress=30, log=["second started"], subtasks=running_steps,
        ),
    ]
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = ServiceBatchDatabase(rows)

    batch = service.get_ingestion_batch("batch-1")

    assert batch is not None
    assert batch["status"] == "running"
    assert batch["document_count"] == 2
    assert batch["elapsed_seconds"] >= 239
    assert "===== [1/2] first.pdf · 已完成 =====" in batch["log"]
    assert "first completed" in batch["log"]
    assert "===== [2/2] second.pdf · 执行中 =====" in batch["log"]
    assert "second started" in batch["log"]
    assert batch["metadata"]["subtasks"]["structure"]["status"] == "completed"


def test_preparation_batch_preserves_its_job_kind():
    started = datetime.now(timezone.utc)
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = ServiceBatchDatabase([
        _job(
            "prepare", "source.pdf", "running", started,
            progress=30, log=["预处理中"], kind="content_preparation",
            subtasks={
                "structure": {"status": "completed", "progress": 100, "detail": ""},
                "chunking": {"status": "running", "progress": 40, "detail": ""},
            },
        ),
    ])

    batch = service.get_ingestion_batch("batch-1")

    assert batch["kind"] == "content_preparation"
    assert batch["metadata"]["subtasks"]["chunking"]["status"] == "running"


def test_batch_does_not_finish_while_any_document_is_queued():
    started = datetime.now(timezone.utc) - timedelta(seconds=10)
    rows = [
        _job("first", "first.pdf", "completed", started,
             completed_at=started + timedelta(seconds=5), progress=100, log=["done"]),
        _job("second", "second.pdf", "queued", started, progress=0),
    ]
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = ServiceBatchDatabase(rows)

    batch = service.get_ingestion_batch("batch-1")

    assert batch is not None
    assert batch["status"] == "running"
    assert "等待开始处理" in batch["log"]
