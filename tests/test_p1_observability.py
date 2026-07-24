import json

from benchmarks.build_annotation_bundle import build_review_queue
from benchmarks.benchmark_kg_construction import relation_scenario

from knowledge_forge import database as database_module
from knowledge_forge.database import Database
from knowledge_forge.observability import (
    InstrumentedConnection,
    current_metrics,
    increment,
    metrics_scope,
    timed,
)
from knowledge_forge.ollama import EffectiveModelContext, OllamaClient


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_metrics_scope_is_nested_fail_open_and_contains_no_payload():
    assert current_metrics() is None
    with metrics_scope("ingestion") as outer:
        increment("chunk_count", 2)
        with metrics_scope("ignored") as nested:
            assert nested is outer
            with timed("primary"):
                increment("entity_count", 3)
    snapshot = outer.snapshot()
    assert snapshot["kind"] == "ingestion"
    assert snapshot["counters"] == {"chunk_count": 2, "entity_count": 3}
    assert snapshot["durations_ms"]["primary"] >= 0
    assert "content" not in json.dumps(snapshot)
    assert current_metrics() is None


def test_ollama_transport_counts_logical_http_inputs_and_repairs(monkeypatch):
    replies = iter([
        Response({"embeddings": [[1.0, 0.0], [0.0, 1.0]]}),
        Response({"message": {"content": "not-json"}, "done": True,
                  "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 3}),
        Response({"message": {"content": '{"ok": true}'}, "done": True,
                  "done_reason": "stop", "prompt_eval_count": 20, "eval_count": 3}),
    ])
    monkeypatch.setattr("knowledge_forge.ollama.requests.post", lambda *args, **kwargs: next(replies))
    client = OllamaClient("http://ollama.test")
    monkeypatch.setattr(
        client,
        "effective_model_context",
        lambda model: EffectiveModelContext(model, 8192, 8192, "test", "runner"),
    )

    with metrics_scope("transport") as metrics:
        assert len(client.embed(["one", "two"], "embed")) == 2
        assert client.json_chat(
            model="llm", system="s", prompt="p",
            context_budget_tokens=8192, max_output_tokens=256,
        ) == {"ok": True}

    counters = metrics.snapshot()["counters"]
    assert counters["ollama_logical_calls"] == 2
    assert counters["ollama_http_calls"] == 3
    assert counters["embedding_input_count"] == 2
    assert counters["json_repair_count"] == 1
    assert counters["http_retry_count"] == 1


class RawResult:
    def __init__(self, value=None):
        self.value = value or {"value": 1}

    def fetchone(self):
        return self.value


class RawConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((statement, params))
        return RawResult()


def test_database_proxy_counts_the_real_execute_boundary():
    raw = RawConnection()
    with metrics_scope("db") as metrics:
        result = InstrumentedConnection(raw).execute("SELECT %s", (1,))
        assert result.fetchone() == {"value": 1}
    assert raw.calls == [("SELECT %s", (1,))]
    assert metrics.snapshot()["counters"]["db_statement_count"] == 1


def test_column_discovery_unwraps_instrumented_database_connection(monkeypatch):
    class ColumnConnection(RawConnection):
        def execute(self, statement, params=None):
            self.calls.append((statement, params))
            return RawResult({"present": True})

    raw = ColumnConnection()
    monkeypatch.setattr(database_module.psycopg, "Connection", ColumnConnection)

    with metrics_scope("db") as metrics:
        assert Database._column_exists(
            InstrumentedConnection(raw), "relation_assertions", "revision_id",
        )

    assert raw.calls[0][1] == ("relation_assertions", "revision_id")
    assert metrics.snapshot()["counters"]["db_statement_count"] == 1


def test_characterization_matrix_keeps_dense_pair_and_document_denominators():
    assert relation_scenario(0, 0)["raw_directed_pairs_per_chunk"] == 0
    assert relation_scenario(20, 3) == {
        "documents": 3,
        "entities_per_chunk": 20,
        "raw_directed_pairs_per_chunk": 380,
        "legacy_full_rebuild_requests": 3,
        "coalesced_target_rebuild_requests": 1,
    }
    assert relation_scenario(100, 10)["raw_directed_pairs_per_chunk"] == 9900


def test_annotation_bundle_is_large_enough_but_explicitly_not_human_gold():
    rows = build_review_queue()
    assert 50 <= len(rows) <= 100
    assert {row["split"] for row in rows} == {"dev", "holdout_candidate"}
    assert all(row["label_source"] == "synthetic_candidate_unreviewed" for row in rows)
    assert all(row["review_status"] == "pending_human_confirmation" for row in rows)
