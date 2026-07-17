import threading

import pytest

from knowledge_forge.config import Config
from knowledge_forge.service import IngestionCancelled, KnowledgeForgeService


class FakeDatabase:
    def __init__(self, document, job=None, remaining_documents=0):
        self.document = document
        self.job = job or {"id": "job-1", "status": "running"}
        self.cancelled = False
        self.soft_deleted = False
        self.remaining_documents = remaining_documents
        self.cancelled_project_jobs = []
        self.jobs = {self.job["id"]: self.job}
        self.soft_delete_calls = 0
        self.graph_states = []

    def get_document(self, document_id):
        return None if self.soft_deleted else self.document

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def get_document_including_deleted(self, document_id):
        return self.document

    def cancel_document_jobs(self, document_id):
        self.job["status"] = "cancelled"
        self.cancelled = True
        return 1

    def soft_delete_document(self, document_id):
        self.soft_delete_calls += 1
        self.soft_deleted = True

    def create_job(self, kind, project_id=None, document_id=None):
        job = {
            "id": "cleanup-job",
            "kind": kind,
            "project_id": project_id,
            "document_id": document_id,
            "status": "queued",
            "progress": 0,
            "log": [],
            "metadata": {},
        }
        self.jobs[job["id"]] = job
        return job

    def update_job(self, job_id, *, status=None, progress=None, log="", error="", metadata=None):
        job = self.jobs[job_id]
        if status is not None:
            job["status"] = status
        if progress is not None:
            job["progress"] = progress
        if log:
            job.setdefault("log", []).append(log)
        if error:
            job["error_message"] = error
        if metadata:
            job["metadata"] = {**job.get("metadata", {}), **metadata}
        return job

    def count_documents(self, project_id):
        return self.remaining_documents

    def set_graph_state(self, project_id, status, **values):
        self.graph_states.append((project_id, status, values))
        return {"id": project_id, "graph_status": status, **values}

    def get_project(self, project_id):
        return {"id": project_id, "project_key": "test"}

    def list_documents(self, project_id):
        return []

    def count_publishable_relations(self, project_id, document_ids=None):
        return 0

    def list_pending_indexing_failures(self, project_id, object_type="relation"):
        return []

    def cancel_project_jobs(self, project_id, kinds, reason):
        self.cancelled_project_jobs.append((project_id, kinds, reason))
        return 1


class FakeSchemas:
    def __init__(self):
        self.activated = []

    def activate_base(self, project_id):
        self.activated.append(project_id)


class FakeGraph:
    def __init__(self):
        self.deleted = []
        self.pruned = []
        self.calls = []

    def delete_document(self, document_id):
        self.deleted.append(document_id)

    def ensure_schema(self):
        self.calls.append("schema")

    def delete_project(self, project_id):
        self.calls.append(("purge", project_id))

    def statistics(self, project_id):
        return {"relations": 0}

    def prune_project_entities(self, project_id):
        self.pruned.append(project_id)


class DeferredExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run_all(self):
        while self.calls:
            function, args = self.calls.pop(0)
            function(*args)


def make_service(database):
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = database
    service.graph = FakeGraph()
    service.schemas = FakeSchemas()
    service._document_locks_guard = threading.Lock()
    service._document_locks = {}
    service.cleanup_executor = DeferredExecutor()
    return service


def test_delete_document_cancels_work_and_moves_source_and_process_files(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    process = tmp_path / "process.json"
    source.write_text("source", encoding="utf-8")
    process.write_text("{}", encoding="utf-8")
    database = FakeDatabase({
        "id": 8,
        "project_id": 9,
        "project_name": "test",
        "local_path": str(source),
        "metadata": {"chunk_file": str(process)},
    })
    service = make_service(database)
    monkeypatch.setattr(Config, "trash_root", tmp_path / "trash")

    job_id = service.delete_document(8)

    assert database.cancelled is True
    assert database.soft_deleted is True
    assert database.cancelled_project_jobs[0][0:2] == (9, ["schema_reassessment"])
    assert job_id == "cleanup-job"
    assert service.graph.deleted == []
    service.cleanup_executor.run_all()
    assert service.graph.deleted == []
    assert service.graph.calls == ["schema", ("purge", 9)]
    assert service.graph.pruned == []
    assert service.schemas.activated == [9]
    assert not source.exists()
    assert not process.exists()
    assert len(list((tmp_path / "trash").rglob("*source.pdf"))) == 1
    assert len(list((tmp_path / "trash").rglob("*process.json"))) == 1
    assert database.soft_delete_calls == 3
    assert database.jobs[job_id]["status"] == "completed"


def test_delete_document_keeps_active_schema_while_project_has_documents(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_text("source", encoding="utf-8")
    database = FakeDatabase({
        "id": 8,
        "project_id": 9,
        "project_name": "test",
        "local_path": str(source),
        "metadata": {},
    }, remaining_documents=1)
    service = make_service(database)
    monkeypatch.setattr(Config, "trash_root", tmp_path / "trash")

    service.delete_document(8)
    service.cleanup_executor.run_all()

    assert service.schemas.activated == []


def test_delete_returns_before_active_document_lock_is_released(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_text("source", encoding="utf-8")
    database = FakeDatabase({
        "id": 8,
        "project_id": 9,
        "project_name": "test",
        "original_filename": "source.pdf",
        "local_path": str(source),
        "metadata": {},
    })
    service = make_service(database)
    monkeypatch.setattr(Config, "trash_root", tmp_path / "trash")

    lock = service._document_lock(8)
    lock.acquire()
    try:
        job_id = service.delete_document(8)
    finally:
        lock.release()

    assert job_id == "cleanup-job"
    assert database.soft_deleted is True
    assert service.graph.deleted == []
    service.cleanup_executor.run_all()
    assert service.graph.deleted == []
    assert service.graph.calls == ["schema", ("purge", 9)]


def test_cancelled_job_is_rejected_before_worker_writes():
    database = FakeDatabase({"id": 8}, {"id": "job-1", "status": "cancelled"})
    service = make_service(database)

    with pytest.raises(IngestionCancelled):
        service._assert_ingestion_active("job-1", 8)
