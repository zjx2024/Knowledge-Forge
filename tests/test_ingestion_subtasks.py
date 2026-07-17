from knowledge_forge.service import KnowledgeForgeService


class FakeJobDatabase:
    def __init__(self):
        self.job = {
            "id": "job-1",
            "status": "queued",
            "progress": 0,
            "metadata": {"preserved": True},
        }

    def get_job(self, job_id):
        assert job_id == "job-1"
        return self.job

    def update_job(self, job_id, *, status=None, progress=None, log="", metadata=None, **_kwargs):
        assert job_id == "job-1"
        if status is not None:
            self.job["status"] = status
        if progress is not None:
            self.job["progress"] = progress
        if metadata:
            self.job["metadata"] = {**self.job["metadata"], **metadata}
        return self.job

    def begin_graph_mutation(self, job_id):
        assert job_id == "job-1"
        self.job["status"] = "running"
        return self.job


def service_with_fake_job():
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = FakeJobDatabase()
    return service


def test_start_ingestion_initializes_five_independent_subtasks():
    service = service_with_fake_job()

    service._start_ingestion_job("job-1", 9)

    metadata = service.db.job["metadata"]
    assert metadata["active_subtask"] == "structure"
    assert list(metadata["subtasks"]) == list(service.INGESTION_SUBTASKS)
    assert metadata["subtasks"]["structure"]["status"] == "running"
    assert metadata["subtasks"]["chunking"]["status"] == "waiting"


def test_subtask_update_preserves_completed_steps():
    service = service_with_fake_job()
    service._start_ingestion_job("job-1", 9)
    service._job_subtask(
        "job-1",
        "structure",
        subtask_status="completed",
        subtask_progress=100,
        detail="12 个结构块",
    )
    service._job_subtask(
        "job-1",
        "chunking",
        subtask_status="running",
        subtask_progress=60,
        detail="18 个知识块",
    )

    subtasks = service.db.job["metadata"]["subtasks"]
    assert subtasks["structure"] == {
        "status": "completed",
        "progress": 100,
        "detail": "12 个结构块",
    }
    assert subtasks["chunking"]["progress"] == 60
    assert service.db.job["metadata"]["active_subtask"] == "chunking"
