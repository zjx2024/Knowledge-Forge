from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

from knowledge_forge.database import Database, JobPublicationCommitted
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

    def get_project(self, project_id):
        return {"id": project_id} if project_id == 7 else None

    def get_job(self, job_id):
        return next((row for row in self.rows if row["id"] == job_id), None)


def _job(
    job_id, name, status, created_at, *, completed_at=None, progress=0,
    log=None, subtasks=None, kind="ingestion", metadata=None,
):
    return {
        "id": job_id,
        "kind": kind,
        "status": status,
        "progress": progress,
        "created_at": created_at,
        "completed_at": completed_at,
        "document_name": name,
        "log": log or [],
        "metadata": {
            "batch_id": "batch-1", "subtasks": subtasks or {},
            **(metadata or {}),
        },
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


def test_batch_propagates_cancellation_request_and_terminal_partial_cancel():
    started = datetime.now(timezone.utc)
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = ServiceBatchDatabase([
        _job(
            "running", "first.pdf", "running", started, progress=42,
            kind="content_preparation",
            subtasks={
                "structure": {"status": "completed", "progress": 100},
                "chunking": {"status": "running", "progress": 50},
            },
            metadata={"cancel_requested": True},
        ),
        _job(
            "queued", "second.pdf", "cancelled", started, progress=100,
            kind="content_preparation",
        ),
    ])

    stopping = service.get_ingestion_batch("batch-1")

    assert stopping["status"] == "running"
    assert stopping["metadata"]["cancel_requested"] is True
    assert stopping["metadata"]["subtasks"]["chunking"]["status"] == "running"

    service.db.rows[0]["status"] = "cancelled"
    service.db.rows[0]["metadata"]["subtasks"]["chunking"]["status"] = "cancelled"
    stopped = service.get_ingestion_batch("batch-1")

    assert stopped["status"] == "cancelled"
    assert stopped["metadata"]["cancel_requested"] is False
    assert stopped["metadata"]["subtasks"]["chunking"]["status"] == "cancelled"


def test_batch_cancellation_accepts_completed_lead_job_id_and_stops_active_siblings():
    started = datetime.now(timezone.utc)
    rows = [
        {
            **_job(
                "first", "first.pdf", "completed", started,
                completed_at=started, progress=100,
                kind="content_preparation",
            ),
            "project_id": 7,
        },
        {
            **_job(
                "second", "second.pdf", "running", started, progress=35,
                kind="content_preparation",
            ),
            "project_id": 7,
        },
        {
            **_job(
                "third", "third.pdf", "queued", started,
                kind="content_preparation",
            ),
            "project_id": 7,
        },
    ]

    class CancellationDatabase(ServiceBatchDatabase):
        def __init__(self, values):
            super().__init__(values)
            self.cancelled = []

        def assert_project_construction_strategy(self, project_id, expected=None):
            assert project_id == 7
            return "knowledge_forge"

        def request_job_cancellation(
            self, job_id, *, project_id, allowed_kinds, reason,
        ):
            row = self.get_job(job_id)
            self.cancelled.append(job_id)
            return {**row, "metadata": {**row["metadata"], "cancel_requested": True}}

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = CancellationDatabase(rows)

    result = service.request_project_job_cancellation(
        7,
        job_id="first",
        batch_id="batch-1",
        expected_kind="content_preparation",
    )

    assert service.db.cancelled == ["second", "third"]
    assert result["requested_count"] == 2


def test_batch_cancellation_skips_published_tail_and_stops_other_siblings():
    started = datetime.now(timezone.utc)
    rows = [
        {
            **_job(
                "published", "first.pdf", "running", started, progress=99,
                kind="content_preparation",
            ),
            "project_id": 7,
        },
        {
            **_job(
                "queued", "second.pdf", "queued", started,
                kind="content_preparation",
            ),
            "project_id": 7,
        },
    ]

    class PublicationRaceDatabase(ServiceBatchDatabase):
        def __init__(self, values):
            super().__init__(values)
            self.cancelled = []

        def assert_project_construction_strategy(self, project_id, expected=None):
            assert project_id == 7
            return "knowledge_forge"

        def request_job_cancellation(
            self, job_id, *, project_id, allowed_kinds, reason,
        ):
            if job_id == "published":
                raise JobPublicationCommitted("任务成果已经发布")
            self.cancelled.append(job_id)
            row = self.get_job(job_id)
            return {**row, "metadata": {**row["metadata"], "cancel_requested": True}}

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = PublicationRaceDatabase(rows)

    result = service.request_project_job_cancellation(
        7,
        job_id="published",
        batch_id="batch-1",
        expected_kind="content_preparation",
    )

    assert service.db.cancelled == ["queued"]
    assert result["requested_count"] == 1
    assert result["skipped_count"] == 1
    assert result["skipped_jobs"] == [
        {"job_id": "published", "reason": "publication_committed"},
    ]


def test_single_job_cancellation_rejects_post_publication_tail():
    started = datetime.now(timezone.utc)
    rows = [{
        **_job(
            "published", "only.pdf", "running", started, progress=99,
            kind="content_preparation",
        ),
        "project_id": 7,
    }]

    class PublicationRaceDatabase(ServiceBatchDatabase):
        def assert_project_construction_strategy(self, project_id, expected=None):
            return "knowledge_forge"

        def request_job_cancellation(self, *args, **kwargs):
            raise JobPublicationCommitted("任务成果已经发布")

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = PublicationRaceDatabase(rows)

    with pytest.raises(JobPublicationCommitted, match="成果已经发布"):
        service.request_project_job_cancellation(
            7,
            job_id="published",
            expected_kind="content_preparation",
        )


def test_cancellation_without_any_target_is_rejected_instead_of_guessing():
    class CancellationDatabase(ServiceBatchDatabase):
        def assert_project_construction_strategy(self, project_id, expected=None):
            assert project_id == 7
            return "knowledge_forge"

        def latest_active_job(self, *_args, **_kwargs):
            raise AssertionError("an unscoped request must not select an arbitrary job")

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = CancellationDatabase([])

    with pytest.raises(ValueError, match="请指定要终止的任务"):
        service.request_project_job_cancellation(7)


def test_single_project_wide_job_preserves_real_subtask_unit_detail():
    started = datetime.now(timezone.utc)
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = ServiceBatchDatabase([
        _job(
            "project-build", "project.pdf", "running", started,
            progress=64, log=["关系分类处理中"],
            subtasks={
                "extraction": {
                    "status": "running", "progress": 72,
                    "detail": "关系分类 9/15",
                },
            },
        ),
    ])

    batch = service.get_ingestion_batch("batch-1")

    assert batch["metadata"]["subtasks"]["extraction"]["detail"] == (
        "关系分类 9/15"
    )


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
