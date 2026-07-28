from __future__ import annotations

import os

import pandas as pd
import psycopg
import pytest
from psycopg.rows import dict_row

from knowledge_forge.community_reports import (
    CommunityReportConfig,
    CommunityReportService,
    GraphRAGCommunityReportGateway,
)
from knowledge_forge.global_search import (
    GraphRAGGlobalSearchGateway,
    GlobalSearchConfig,
    GlobalSearchService,
    GlobalSearchUnavailable,
)
from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.database import Database
from knowledge_forge.graph_policy import GRAPH_POLICY_HASH, GRAPH_POLICY_VERSION


def test_reports_and_global_search_migration_is_additive_and_snapshot_bound():
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0011_graphrag_reports_global_search"
    )
    sql = migration.sql.upper()
    assert "CREATE TABLE GRAPHRAG_COMMUNITY_REPORT_RUNS" in sql
    assert "CREATE TABLE GRAPHRAG_COMMUNITY_REPORTS" in sql
    assert "CREATE TABLE GRAPHRAG_GLOBAL_SEARCH_RUNS" in sql
    assert "DETECTION_RUN_ID" in sql
    assert "GRAPH_GENERATION" in sql
    assert "PROMPT_HASH" in sql
    assert "MODEL_CONFIG_HASH" in sql
    assert "GRAPHRAG_VERSION" in sql
    assert "REPORT_INPUT_HASH" in sql
    assert "SELECTED_REPORT_IDS" in sql
    assert "DROP " not in sql


class MemoryRepository:
    def __init__(self, *, communities=None, current=True):
        self.community = {
            "run": {
                "run_id": "detect-1", "project_id": 7, "graph_generation": 3,
                "input_hash": "graph-input", "status": "completed", "current": current,
            },
            "current": current,
            "communities": communities if communities is not None else [{
                "stable_community_id": "community-1", "level": 0, "title": "C1",
                "metadata": {},
            }],
            "members": [
                {"stable_community_id": "community-1", "canonical_entity_id": 11,
                 "entity_key": "e-11", "name": "Alpha", "entity_type": "Organization"},
                {"stable_community_id": "community-1", "canonical_entity_id": 12,
                 "entity_key": "e-12", "name": "Beta", "entity_type": "Product"},
            ],
            "edges": [{
                "stable_edge_id": "edge-1", "source_entity_id": 11,
                "target_entity_id": 12, "relation_types": ["uses"],
                "assertion_ids": [101], "text_unit_ids": ["chunk:r:c"],
            }],
        }
        self.report_runs = {}
        self.report_items = {}
        self.active_report_run_id = None
        self.search_runs = {}

    def show_graphrag_community_run(self, run_id):
        assert run_id == "detect-1"
        return self.community

    def start_community_report_run(self, values):
        self.report_runs.setdefault(values["run_id"], {**values, "status": "running"})
        return self.report_runs[values["run_id"]]

    def list_community_report_items(self, run_id):
        return list(self.report_items.get(run_id, {}).values())

    def upsert_community_report_item(self, run_id, item):
        self.report_items.setdefault(run_id, {})[item["community_id"]] = dict(item)
        return item

    def finalize_community_report_run(self, run_id, *, status, metrics, error=""):
        self.report_runs[run_id].update(status=status, metrics=metrics, error_message=error)
        if status == "succeeded":
            self.active_report_run_id = run_id
        return self.report_runs[run_id]

    def show_community_report_run(self, run_id):
        run = dict(self.report_runs[run_id])
        run["current"] = bool(
            self.community["current"] and run_id == self.active_report_run_id
        )
        return {
            "run": run,
            "current": run["current"],
            "reports": self.list_community_report_items(run_id),
        }

    def get_active_community_report_run(self, project_id):
        if self.active_report_run_id is None:
            return None
        run = self.report_runs[self.active_report_run_id]
        return run if run["project_id"] == project_id else None

    def start_global_search_run(self, values):
        self.search_runs.setdefault(values["run_id"], {**values, "status": "running"})
        return self.search_runs[values["run_id"]]

    def finalize_global_search_run(
        self, run_id, *, status, result, error="", error_category="", latency_ms=0,
    ):
        self.search_runs[run_id].update(
            status=status, result=result, error_message=error,
            error_category=error_category, latency_ms=latency_ms,
        )
        return self.search_runs[run_id]


class FakeReportGateway:
    model_id = "fake-report-model"

    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def generate(self, context, config):
        self.calls += 1
        entity_ids = [999] if self.invalid else [11, 12]
        return {
            "title": "Alpha and Beta",
            "summary": "Alpha uses Beta.",
            "findings": [{
                "summary": "Usage", "explanation": "Supported by edge-1.",
                "entity_ids": entity_ids, "relation_ids": ["edge-1"],
            }],
            "usage": {"tokens": 20},
        }


class RateLimitError(RuntimeError):
    status_code = 429


class FailingReportGateway(FakeReportGateway):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    def generate(self, context, config):
        self.calls += 1
        if isinstance(self.failure, Exception):
            raise self.failure
        return self.failure


class FakeSearchGateway:
    model_id = "fake-global-model"

    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def search(self, query, reports, config):
        self.calls += 1
        if self.fail:
            raise TimeoutError("controlled map failure")
        return {
            "answer": "Alpha uses Beta.",
            "report_ids": [reports[0]["report_id"]],
            "citations": [{"report_id": reports[0]["report_id"], "finding": 0}],
            "map_outputs": [{"report_id": reports[0]["report_id"], "score": 90}],
            "reduce_output": "Alpha uses Beta.",
            "usage": {"tokens": 30},
        }


def test_reports_are_snapshot_bound_citation_checked_and_resume_without_repeat_calls():
    repository = MemoryRepository()
    gateway = FakeReportGateway()
    service = CommunityReportService(repository, gateway)
    config = CommunityReportConfig(prompt="summarize", max_input_tokens=1000)

    first = service.run("detect-1", config)
    second = service.run("detect-1", config)

    assert first["status"] == "succeeded"
    assert second["run_id"] == first["run_id"]
    assert gateway.calls == 1
    shown = repository.show_community_report_run(first["run_id"])
    assert shown["current"] is True
    assert shown["reports"][0]["entity_ids"] == [11, 12]
    assert shown["reports"][0]["relation_ids"] == ["edge-1"]


def test_invalid_report_citation_fails_closed_and_never_becomes_active():
    repository = MemoryRepository()
    result = CommunityReportService(repository, FakeReportGateway(invalid=True)).run(
        "detect-1", CommunityReportConfig(prompt="summarize")
    )
    assert result["status"] == "terminal_failed"
    assert repository.active_report_run_id is None


@pytest.mark.parametrize("payload", ["not-json", {"title": "missing fields"}])
def test_invalid_report_output_is_terminal_and_never_active(payload):
    repository = MemoryRepository()
    gateway = FailingReportGateway(payload)
    service = CommunityReportService(repository, gateway)
    result = service.run(
        "detect-1", CommunityReportConfig(prompt="summarize")
    )
    repeated = service.run("detect-1", CommunityReportConfig(prompt="summarize"))
    assert result["status"] == "terminal_failed"
    assert repeated["status"] == "terminal_failed"
    assert gateway.calls == 1
    assert repository.active_report_run_id is None


def test_rate_limit_is_retryable_and_can_resume():
    repository = MemoryRepository()
    gateway = FailingReportGateway(RateLimitError("slow down"))
    service = CommunityReportService(repository, gateway)
    config = CommunityReportConfig(prompt="summarize")

    first = service.run("detect-1", config)
    first_status = first["status"]
    gateway.failure = FakeReportGateway().generate(
        repository.community, config,
    )
    second = service.run("detect-1", config)

    assert first_status == "retryable_failed"
    assert second["status"] == "succeeded"
    assert gateway.calls == 2


def test_retryable_report_becomes_terminal_without_another_call_after_attempt_budget():
    repository = MemoryRepository()
    gateway = FailingReportGateway(RateLimitError("slow down"))
    service = CommunityReportService(repository, gateway)
    config = CommunityReportConfig(prompt="summarize", max_attempts=1)

    first = service.run("detect-1", config)
    first_status = first["status"]
    second = service.run("detect-1", config)
    item = repository.list_community_report_items(second["run_id"])[0]

    assert first_status == "retryable_failed"
    assert second["status"] == "terminal_failed"
    assert item["status"] == "terminal_failed"
    assert item["error_category"] == "max_attempts"
    assert gateway.calls == 1


def test_persistence_interruption_is_recorded_and_resume_retries_model_once():
    class InterruptedRepository(MemoryRepository):
        def __init__(self):
            super().__init__()
            self.interrupt_once = True

        def upsert_community_report_item(self, run_id, item):
            if item["status"] == "succeeded" and self.interrupt_once:
                self.interrupt_once = False
                raise ConnectionError("controlled persistence interruption")
            return super().upsert_community_report_item(run_id, item)

    repository = InterruptedRepository()
    gateway = FakeReportGateway()
    service = CommunityReportService(repository, gateway)

    first = service.run("detect-1", CommunityReportConfig(prompt="summarize"))
    first_status = first["status"]
    second = service.run("detect-1", CommunityReportConfig(prompt="summarize"))

    assert first_status == "retryable_failed"
    assert second["status"] == "succeeded"
    assert gateway.calls == 2


def test_empty_graph_completes_zero_report_run_without_model_call():
    repository = MemoryRepository(communities=[])
    gateway = FakeReportGateway()
    result = CommunityReportService(repository, gateway).run(
        "detect-1", CommunityReportConfig(prompt="summarize")
    )
    assert result["status"] == "succeeded"
    assert result["metrics"]["expected"] == 0
    assert gateway.calls == 0
    with pytest.raises(GlobalSearchUnavailable, match="empty"):
        GlobalSearchService(repository, FakeSearchGateway()).search(
            7, "question", GlobalSearchConfig(enabled=True),
        )


def test_global_search_requires_opt_in_and_current_complete_report_snapshot():
    repository = MemoryRepository()
    report = CommunityReportService(repository, FakeReportGateway()).run(
        "detect-1", CommunityReportConfig(prompt="summarize")
    )
    gateway = FakeSearchGateway()
    service = GlobalSearchService(repository, gateway)

    with pytest.raises(GlobalSearchUnavailable, match="opt-in"):
        service.search(7, "What is connected?", GlobalSearchConfig(enabled=False))
    repository.community["current"] = False
    with pytest.raises(GlobalSearchUnavailable, match="stale"):
        service.search(7, "What is connected?", GlobalSearchConfig(enabled=True))
    repository.community["current"] = True
    result = service.search(7, "What is connected?", GlobalSearchConfig(enabled=True))
    repeated = service.search(7, "What is connected?", GlobalSearchConfig(enabled=True))

    assert report["status"] == "succeeded"
    assert result["status"] == "succeeded"
    assert repeated["run_id"] == result["run_id"]
    assert result["result"]["citations"][0]["report_id"]
    assert gateway.calls == 1


def test_global_search_map_failure_is_persisted_and_never_success():
    repository = MemoryRepository()
    CommunityReportService(repository, FakeReportGateway()).run(
        "detect-1", CommunityReportConfig(prompt="summarize")
    )
    result = GlobalSearchService(repository, FakeSearchGateway(fail=True)).search(
        7, "question", GlobalSearchConfig(enabled=True)
    )
    assert result["status"] == "retryable_failed"
    assert not result.get("result", {}).get("answer")


def test_global_search_terminal_output_is_not_retried_for_immutable_input():
    class MalformedSearchGateway(FakeSearchGateway):
        def search(self, query, reports, config):
            self.calls += 1
            return {"answer": "", "report_ids": [], "citations": []}

    repository = MemoryRepository()
    CommunityReportService(repository, FakeReportGateway()).run(
        "detect-1", CommunityReportConfig(prompt="summarize")
    )
    gateway = MalformedSearchGateway()
    service = GlobalSearchService(repository, gateway)

    first = service.search(7, "question", GlobalSearchConfig(enabled=True))
    repeated = service.search(7, "question", GlobalSearchConfig(enabled=True))

    assert first["status"] == repeated["status"] == "terminal_failed"
    assert gateway.calls == 1


def test_global_search_rejects_general_knowledge_and_invalid_token_cap():
    repository = MemoryRepository()
    CommunityReportService(repository, FakeReportGateway()).run(
        "detect-1", CommunityReportConfig(prompt="summarize")
    )
    service = GlobalSearchService(repository, FakeSearchGateway())

    with pytest.raises(ValueError, match="general knowledge"):
        service.search(7, "question", GlobalSearchConfig(
            enabled=True, allow_general_knowledge=True,
        ))
    with pytest.raises(ValueError, match="max_context_tokens"):
        service.search(7, "question", GlobalSearchConfig(
            enabled=True, max_context_tokens=0,
        ))


def test_graphrag_report_adapter_calls_verified_311_workflow(monkeypatch):
    calls = []

    async def fake_create_reports(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame([{
            "title": "Alpha and Beta",
            "summary": "Alpha uses Beta.",
            "findings": [{"summary": "Usage", "explanation": "Supported."}],
        }])

    monkeypatch.setattr(
        "graphrag.index.workflows.create_community_reports.create_community_reports",
        fake_create_reports,
    )
    context = {
        "community": {
            "stable_community_id": "community-1", "level": 0, "title": "C1",
        },
        "entities": MemoryRepository().community["members"],
        "relationships": MemoryRepository().community["edges"],
        "allowed_entity_ids": [11, 12],
        "allowed_relation_ids": ["edge-1"],
    }
    gateway = GraphRAGCommunityReportGateway(
        model_id="fake-model", model=object(), tokenizer=object(),
    )

    result = gateway.generate(context, CommunityReportConfig(prompt="summarize"))

    assert gateway.graphrag_version == "3.1.1"
    assert len(calls) == 1
    assert calls[0]["claims_input"] is None
    assert calls[0]["communities"].iloc[0]["entity_ids"] == [11, 12]
    assert result["findings"][0]["entity_ids"] == [11, 12]
    assert result["findings"][0]["relation_ids"] == ["edge-1"]


def test_graphrag_global_adapter_calls_verified_311_api(monkeypatch):
    calls = []

    async def fake_global_search(**kwargs):
        calls.append(kwargs)
        return "Alpha uses Beta.", {"reports": pd.DataFrame([{"community": 0}])}

    monkeypatch.setattr("graphrag.api.query.global_search", fake_global_search)
    repository = MemoryRepository()
    repository.community["communities"][0].update({
        "graphrag_community": 0,
        "parent_stable_community_id": None,
        "size": 2,
    })
    reports = [{
        "report_id": "report-1", "community_id": "community-1", "level": 0,
        "title": "C1", "summary": "Alpha uses Beta.", "findings": [],
    }]
    class CopyableGlobalSearchConfig:
        def __init__(self, max_context_tokens=12_000):
            self.max_context_tokens = max_context_tokens

        def model_copy(self, *, update):
            return CopyableGlobalSearchConfig(**{
                "max_context_tokens": self.max_context_tokens,
                **update,
            })

    class CopyableGraphConfig:
        def __init__(self, global_search=None):
            self.global_search = global_search or CopyableGlobalSearchConfig()

        def model_copy(self, *, update):
            return CopyableGraphConfig(global_search=update["global_search"])

    gateway = GraphRAGGlobalSearchGateway(
        model_id="fake-global", graph_config=CopyableGraphConfig(),
        detection_snapshot=repository.community,
    )

    result = gateway.search(
        "What is connected?", reports,
        GlobalSearchConfig(enabled=True, max_context_tokens=777),
    )

    assert gateway.graphrag_version == "3.1.1"
    assert len(calls) == 1
    assert calls[0]["dynamic_community_selection"] is False
    assert calls[0]["config"].global_search.max_context_tokens == 777
    assert result["report_ids"] == ["report-1"]
    assert result["citations"] == [{"report_id": "report-1"}]


@pytest.mark.integration
def test_reports_and_global_search_persist_resume_and_fail_closed_when_stale():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")

    class DisposableConfig:
        postgres = psycopg.conninfo.conninfo_to_dict(dsn)

    database = Database(DisposableConfig)
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        project = connection.execute(
            """INSERT INTO projects(
               project_key,name,graph_status,graph_generation,graph_ready_generation,
               graph_policy_version,graph_policy_hash,active_policy_hash)
               VALUES('report-search-test','report-search-test','ready',1,1,%s,%s,%s)
               RETURNING *""",
            (GRAPH_POLICY_VERSION, GRAPH_POLICY_HASH, GRAPH_POLICY_HASH),
        ).fetchone()
        entities = [connection.execute(
            """INSERT INTO canonical_entities(
               project_id,entity_key,name,normalized_name,entity_type)
               VALUES(%s,%s,%s,%s,'Concept') RETURNING *""",
            (project["id"], f"entity-{ordinal}", f"Entity {ordinal}", f"entity {ordinal}"),
        ).fetchone() for ordinal in (1, 2)]
        connection.execute(
            """INSERT INTO graphrag_community_runs(
               run_id,project_id,graph_generation,graph_policy_hash,active_policy_hash,
               input_hash,graphrag_version,adapter_version,weighting_strategy,seed,
               max_cluster_size,use_lcc,status,node_count,community_count,hierarchy_levels)
               VALUES('detect-report-test',%s,1,%s,%s,'input-1','3.1.1','test',
                      'publishable_assertion_count_v1',42,10,FALSE,'completed',2,1,1)""",
            (project["id"], GRAPH_POLICY_HASH, GRAPH_POLICY_HASH),
        )
        connection.execute(
            """INSERT INTO graphrag_communities(
               run_id,project_id,stable_community_id,graphrag_output_id,
               graphrag_community,level,title,size)
               VALUES('detect-report-test',%s,'community-1','output-1',0,0,'C1',2)""",
            (project["id"],),
        )
        for entity in entities:
            connection.execute(
                """INSERT INTO graphrag_community_members(
                   run_id,project_id,stable_community_id,canonical_entity_id)
                   VALUES('detect-report-test',%s,'community-1',%s)""",
                (project["id"], entity["id"]),
            )
        connection.commit()

    class SnapshotReportGateway:
        model_id = "fake-report"
        graphrag_version = "3.1.1"

        def __init__(self):
            self.calls = 0

        def generate(self, context, _config):
            self.calls += 1
            return {
                "title": "Fixture",
                "summary": "Two fixture entities.",
                "findings": [{
                    "summary": "Members", "explanation": "Snapshot members.",
                    "entity_ids": context["allowed_entity_ids"], "relation_ids": [],
                }],
            }

    class SnapshotSearchGateway:
        model_id = "fake-search"
        graphrag_version = "3.1.1"
        prompt_hash = "fake-prompt"

        def search(self, _query, reports, _config):
            report_id = reports[0]["report_id"]
            return {
                "answer": "Fixture answer.", "report_ids": [report_id],
                "citations": [{"report_id": report_id}], "usage": {"tokens": 5},
            }

    report_gateway = SnapshotReportGateway()
    report_service = CommunityReportService(database, report_gateway)
    first = report_service.run(
        "detect-report-test", CommunityReportConfig(prompt="summarize"),
    )
    repeated = report_service.run(
        "detect-report-test", CommunityReportConfig(prompt="summarize"),
    )

    class FlakyReportGateway(SnapshotReportGateway):
        model_id = "fake-report-flaky"

        def generate(self, context, config):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("controlled interruption")
            return {
                "title": "Recovered fixture", "summary": "Recovered.",
                "findings": [{
                    "summary": "Members", "explanation": "Snapshot members.",
                    "entity_ids": context["allowed_entity_ids"], "relation_ids": [],
                }],
            }

    flaky_gateway = FlakyReportGateway()
    flaky_service = CommunityReportService(database, flaky_gateway)
    interrupted = flaky_service.run(
        "detect-report-test", CommunityReportConfig(prompt="recover"),
    )
    assert interrupted["status"] == "retryable_failed"
    assert database.get_active_community_report_run(project["id"])["run_id"] == first["run_id"]
    recovered = flaky_service.run(
        "detect-report-test", CommunityReportConfig(prompt="recover"),
    )
    assert recovered["status"] == "succeeded"
    assert recovered["run_id"] != first["run_id"]
    assert flaky_gateway.calls == 2

    search_service = GlobalSearchService(database, SnapshotSearchGateway())
    search = search_service.search(
        project["id"], "fixture?", GlobalSearchConfig(enabled=True),
    )

    assert first["status"] == repeated["status"] == "succeeded"
    assert first["active"] is True
    assert report_gateway.calls == 1
    assert search["status"] == "succeeded"
    assert search["selected_report_ids"]
    with pytest.raises(ValueError, match="不存在的 entity"):
        database.upsert_community_report_item(first["run_id"], {
            "report_id": "invalid-report", "community_id": "community-1",
            "level": 0, "status": "terminal_failed", "attempt": 2,
            "entity_ids": [999999], "relation_ids": [], "input_hash": "invalid",
        })
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        persisted_report = connection.execute(
            "SELECT * FROM graphrag_community_reports WHERE run_id=%s",
            (recovered["run_id"],),
        ).fetchone()
        persisted_search = connection.execute(
            "SELECT * FROM graphrag_global_search_runs WHERE run_id=%s",
            (search["run_id"],),
        ).fetchone()
        assert persisted_report["status"] == "succeeded"
        assert persisted_report["attempt"] == 2
        assert persisted_search["usage"] == {"tokens": 5}
        other_project = connection.execute(
            "INSERT INTO projects(project_key,name) VALUES('report-other','report-other') RETURNING id",
        ).fetchone()
        connection.commit()

    with pytest.raises(RuntimeError, match="跨项目"):
        database.start_global_search_run({
            "run_id": "global-cross-project",
            "project_id": other_project["id"],
            "report_run_id": recovered["run_id"],
            "query_hash": "q", "report_input_hash": "reports",
            "prompt_hash": "prompt", "model_config_hash": "model-config",
            "model_id": "fake-search", "graphrag_version": "3.1.1",
            "config": {},
        })

    race_run = database.start_global_search_run({
        "run_id": "global-stale-before-finalize",
        "project_id": project["id"], "report_run_id": recovered["run_id"],
        "query_hash": "race-q", "report_input_hash": "reports",
        "prompt_hash": "prompt", "model_config_hash": "model-config",
        "model_id": "fake-search", "graphrag_version": "3.1.1", "config": {},
    })
    assert race_run["status"] == "running"
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        connection.execute(
            "UPDATE projects SET graph_generation=2 WHERE id=%s", (project["id"],),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="stale"):
        database.finalize_global_search_run(
            race_run["run_id"], status="succeeded",
            result={
                "answer": "must not persist",
                "report_ids": [recovered["run_id"]],
                "citations": [{"report_id": recovered["run_id"]}],
            },
        )

    with pytest.raises(GlobalSearchUnavailable, match="stale"):
        search_service.search(project["id"], "fixture?", GlobalSearchConfig(enabled=True))
