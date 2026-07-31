"""Characterization tests for the isolated third construction strategy."""

from __future__ import annotations

from concurrent.futures import Future

import pytest

from knowledge_forge.construction_strategies import (
    ENHANCED_SHADOW_STRATEGY,
    GRAPHRAG_NATIVE_STRATEGY,
    STABLE_STRATEGY,
    normalize_strategy,
)
from knowledge_forge.service import KnowledgeForgeService


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def submit(self, callback, *args):
        self.calls.append((callback.__name__, args))
        future = Future()
        future.set_result(None)
        return future


class _NativeDatabase:
    def __init__(self) -> None:
        self.project = {"id": 7}
        self.documents = [
            {
                "id": 11,
                "project_id": 7,
                "status": "indexed",
                "active_revision_id": "revision-11",
                "structure_ready_at": "2026-07-28T00:00:00Z",
                "original_filename": "one.pdf",
            },
            {
                "id": 12,
                "project_id": 7,
                "status": "indexed",
                "active_revision_id": "revision-12",
                "structure_ready_at": "2026-07-28T00:00:00Z",
                "original_filename": "two.pdf",
            },
        ]
        self.created: list[dict] = []

    @staticmethod
    def assert_graphrag_native_ready():
        return {"migrations": ["0022_graphrag_native_control_plane"]}

    def get_project(self, project_id):
        return self.project if project_id == 7 else None

    def list_documents(self, project_id, strategy_key=STABLE_STRATEGY):
        assert project_id == 7
        assert strategy_key == STABLE_STRATEGY
        return list(self.documents)

    def latest_active_job(self, project_id, kind="", strategy_key=""):
        assert project_id == 7
        assert kind == "graphrag_native_build"
        assert strategy_key == GRAPHRAG_NATIVE_STRATEGY
        return None

    def create_job(self, kind, project_id, document_id, *, metadata, strategy_key):
        self.created.append({
            "kind": kind, "project_id": project_id, "document_id": document_id,
            "metadata": metadata, "strategy_key": strategy_key,
        })
        return {"id": "native-job"}


class _NativeExecutor:
    def run(self, *, job_id: str, project_id: int) -> None:
        raise AssertionError("worker must be queued, not run in this characterization test")


def _service(database: _NativeDatabase) -> KnowledgeForgeService:
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = database
    service.executor = _Recorder()
    service.graphrag_native = _NativeExecutor()
    return service


def test_native_is_registered_and_unknown_strategy_fails_closed():
    assert normalize_strategy(GRAPHRAG_NATIVE_STRATEGY) == GRAPHRAG_NATIVE_STRATEGY
    with pytest.raises(ValueError, match="未知知识构建方案"):
        normalize_strategy("native-typo")


def test_native_rejects_document_scoped_graph_build_without_fallback():
    service = _service(_NativeDatabase())

    with pytest.raises(ValueError, match="仅支持项目级全量构建"):
        service.submit_ingestion(11, strategy_key=GRAPHRAG_NATIVE_STRATEGY)

    assert service.executor.calls == []


def test_native_project_build_queues_only_native_executor_and_no_document_build_rows():
    database = _NativeDatabase()
    service = _service(database)

    result = service.submit_project_ingestion(7, strategy_key=GRAPHRAG_NATIVE_STRATEGY)

    assert result == {
        "job_id": "native-job", "job_ids": ["native-job"],
        "document_count": 2, "scope": "project_full_snapshot",
    }
    assert database.created == [{
        "kind": "graphrag_native_build", "project_id": 7, "document_id": None,
        "metadata": {
            "strategy_key": GRAPHRAG_NATIVE_STRATEGY,
            "scope": "project_full_snapshot",
            "document_ids": [11, 12],
        },
        "strategy_key": GRAPHRAG_NATIVE_STRATEGY,
    }]
    assert service.executor.calls == [("_ingest_graphrag_native", ("native-job", 7))]


def test_native_project_build_requires_shared_preprocessing():
    database = _NativeDatabase()
    database.documents[1] = {**database.documents[1], "structure_ready_at": None}
    service = _service(database)

    with pytest.raises(ValueError, match="已完成预处理"):
        service.submit_project_ingestion(7, strategy_key=GRAPHRAG_NATIVE_STRATEGY)

    assert database.created == []
    assert service.executor.calls == []


def test_strategy_executor_registry_has_no_non_native_fallback(monkeypatch):
    service = _service(_NativeDatabase())
    stable = object()
    enhanced = object()
    native = object()
    monkeypatch.setattr(service, "_ingest_document", stable)
    monkeypatch.setattr(service, "_ingest_enhanced_strategy", enhanced)
    monkeypatch.setattr(service, "_ingest_graphrag_native", native)

    assert service._strategy_executor(STABLE_STRATEGY) is stable
    assert service._strategy_executor(ENHANCED_SHADOW_STRATEGY) is enhanced
    assert service._strategy_executor(GRAPHRAG_NATIVE_STRATEGY) is native
