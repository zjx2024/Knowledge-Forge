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


def test_subtask_and_job_progress_never_move_backwards():
    service = service_with_fake_job()
    service._start_ingestion_job("job-1", 9)
    service._job_subtask(
        "job-1",
        "extraction",
        subtask_status="running",
        subtask_progress=27,
        detail="关系分类 1/4",
        progress=63,
    )
    service._job_subtask(
        "job-1",
        "extraction",
        subtask_status="running",
        subtask_progress=5,
        detail="关系验证 1/8 个关系假设",
        progress=52,
    )

    assert service.db.job["progress"] == 63
    assert service.db.job["metadata"]["subtasks"]["extraction"] == {
        "status": "running",
        "progress": 27,
        "detail": "关系验证 1/8 个关系假设",
    }


def test_knowledge_forge_nested_progress_is_phase_aware_and_monotonic():
    entity_end = KnowledgeForgeService._knowledge_forge_extraction_progress(
        15, 15, "知识块实体阶段完成：24 个实体提及",
    )
    planner_start = KnowledgeForgeService._knowledge_forge_extraction_progress(
        1, 60, "Knowledge Forge 候选对规划 1/15",
    )
    source_inventory = KnowledgeForgeService._knowledge_forge_extraction_progress(
        16, 60, "Knowledge Forge 源事实清单 1/15",
    )
    candidate_ledger = KnowledgeForgeService._knowledge_forge_extraction_progress(
        30, 60, "Knowledge Forge 候选关系审计账本 128/1024",
    )
    proposer = KnowledgeForgeService._knowledge_forge_extraction_progress(
        16, 60, "Knowledge Forge 候选补充 1/15 完成",
    )
    classifier = KnowledgeForgeService._knowledge_forge_extraction_progress(
        31, 60, "Knowledge Forge 关系分类 1/15",
    )
    verifier = KnowledgeForgeService._knowledge_forge_extraction_progress(
        46, 60, "Knowledge Forge 关系验证 1/12 个关系假设",
    )
    batch_verifier = KnowledgeForgeService._knowledge_forge_extraction_progress(
        33, 44, "Knowledge Forge 关系批量终审 44/272",
    )
    structure_review = KnowledgeForgeService._knowledge_forge_extraction_progress(
        38, 44, "Knowledge Forge 关系结构复核 121/322 个关系假设",
    )
    completed = KnowledgeForgeService._knowledge_forge_extraction_progress(
        60, 60, "Knowledge Forge 完成：20 个关系假设，8 个规范事实",
    )

    assert entity_end == (45, "实体抽取 15/15 个知识块")
    assert planner_start[0] >= entity_end[0]
    assert source_inventory[0] > planner_start[0]
    assert source_inventory[1] == "源事实清单 1/15"
    assert candidate_ledger == (70, "候选关系审计账本 128/1024")
    assert proposer[0] > planner_start[0]
    assert classifier[0] > proposer[0]
    assert verifier[0] > classifier[0]
    assert batch_verifier == (82, "关系批量终审 44/272")
    assert structure_review == (88, "关系结构复核 121/322 个关系假设")
    assert completed[0] == 95
    assert "候选对规划 1/15" in planner_start[1]
    assert "个文档" not in planner_start[1]


def test_knowledge_forge_build_progress_excludes_reused_preparation():
    progress = KnowledgeForgeService._knowledge_forge_build_progress

    assert progress() == 0
    assert progress(schema_progress=100) == 0
    assert progress(extraction_progress=5) == 5
    assert progress(extraction_progress=45) == 41
    assert progress(extraction_progress=100) == 90
    assert progress(extraction_progress=100, graph_sync_progress=10) == 91
    assert progress(extraction_progress=100, graph_sync_progress=70) == 97
    assert progress(extraction_progress=100, graph_sync_progress=100) == 100


def test_explicit_build_time_schema_discovery_has_its_own_weight():
    progress = KnowledgeForgeService._knowledge_forge_build_progress

    assert progress(schema_progress=35, schema_discovery=True) == 4
    assert progress(schema_progress=100, schema_discovery=True) == 10
    assert progress(
        schema_progress=100,
        extraction_progress=100,
        schema_discovery=True,
    ) == 92
    assert progress(
        schema_progress=100,
        extraction_progress=100,
        graph_sync_progress=100,
        schema_discovery=True,
    ) == 100
