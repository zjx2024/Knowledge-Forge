from __future__ import annotations

import inspect
from concurrent.futures import Future

from knowledge_forge.construction_strategies import (
    KNOWLEDGE_FORGE_STRATEGY,
    STABLE_STRATEGY,
)
from knowledge_forge.policy import POLICY_AXES
from knowledge_forge.relation_runtime import GroundedRelationRuntime
from knowledge_forge.service import KnowledgeForgeService


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
        self.grounded_relation_readiness_checks = 0
        self.invalidated: list[tuple[int, str]] = []
        self.updated: list[tuple[int, str, dict]] = []
        self.graph_states: list[tuple[int, str, dict]] = []

    def get_document(self, document_id):
        return self.document if document_id == 11 else None

    @staticmethod
    def ensure_project_strategy_graphs(_project_id):
        return []

    @staticmethod
    def ensure_document_strategy_builds(_document_id, reset=False):
        return []

    def assert_grounded_relation_ready(self):
        self.grounded_relation_readiness_checks += 1
        return {"migrations": ["0014", "0015", "0016", "0017", "0018", "0019"]}

    def latest_active_job(self, _project_id, kind="", strategy_key=""):
        return self.active.get(strategy_key)

    def get_document_strategy_build(self, _document_id, strategy_key):
        return self.builds[strategy_key]

    @staticmethod
    def create_job(_kind, _project_id, _document_id, metadata, strategy_key):
        return {"id": f"job-{strategy_key}", "status": "queued"}

    def update_document_strategy_build(self, document_id, strategy_key, **values):
        self.updated.append((document_id, strategy_key, values))
        self.builds[strategy_key] = {**self.builds[strategy_key], **values}
        return self.builds[strategy_key]

    def invalidate_knowledge_forge_for_document(self, document_id, *, reason):
        self.invalidated.append((document_id, reason))

    def set_graph_state(self, project_id, status, **values):
        self.graph_states.append((project_id, status, values))
        return {"id": project_id, "graph_status": status, **values}

    @staticmethod
    def get_project(project_id):
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


def _service(database: _StrategyDatabase) -> KnowledgeForgeService:
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = database
    service.executor = _ImmediateRecorder()
    return service


def test_stable_submit_does_not_touch_knowledge_forge_runtime_or_state() -> None:
    database = _StrategyDatabase()
    service = _service(database)

    assert service.submit_ingestion(11, strategy_key=STABLE_STRATEGY) == "job-stable"

    assert database.grounded_relation_readiness_checks == 0
    assert database.invalidated == []
    assert database.updated[-1][1] == STABLE_STRATEGY
    assert service.executor.calls == [("_ingest_document", ("job-stable", 11))]


def test_knowledge_forge_entry_does_not_call_legacy_p1_or_stable_extractors() -> None:
    enhanced_source = inspect.getsource(
        KnowledgeForgeService._ingest_knowledge_forge_strategy
    )
    entity_source = inspect.getsource(
        KnowledgeForgeService.build_grounded_entity_namespace
    )
    runtime_source = inspect.getsource(GroundedRelationRuntime)
    combined = "\n".join((enhanced_source, entity_source, runtime_source))

    assert "_run_relation_shadow(" not in combined
    assert "RelationCandidatePipeline" not in runtime_source
    assert "extract_chunk(" not in combined
    assert "_relation_candidates(" not in combined
    assert "_verify_relations(" not in combined
    assert "_apply_relation_decision(" not in combined
