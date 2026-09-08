"""Migration and isolated PostgreSQL persistence checks for agent V1."""

from __future__ import annotations

import os
import uuid

import pytest
from psycopg.conninfo import conninfo_to_dict

from knowledge_forge.agent.authorization import session_key_hash
from knowledge_forge.database import Database
from knowledge_forge.migrations import MIGRATIONS


AGENT_TABLES = {
    "agent_threads",
    "agent_runs",
    "agent_messages",
    "agent_run_steps",
    "agent_tool_calls",
    "agent_evidence",
    "agent_claim_citations",
    "agent_run_events",
    "agent_feedback",
}


def test_agent_schema_is_one_additive_startup_safe_migration():
    migration = next(item for item in MIGRATIONS if item.version == "0035_knowledge_agent_v1")

    assert migration.startup_safe is True
    assert all(f"CREATE TABLE IF NOT EXISTS {table}" in migration.sql for table in AGENT_TABLES)
    assert "REFERENCES projects(id)" in migration.sql
    assert "FOREIGN KEY(run_id,project_id)" in migration.sql
    assert "session_key_hash" in migration.sql
    assert "cancellation_requested" in migration.sql
    assert "budget JSONB" in migration.sql
    assert "source_refs JSONB" in migration.sql
    assert "canonical_entities" not in migration.sql
    assert "relation_assertions" not in migration.sql
    assert "vector_embeddings" not in migration.sql
    assert "graphrag_native_runs" not in migration.sql


def test_context_runtime_uses_an_isolated_additive_schema():
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0036_knowledge_agent_context_runtime"
    )

    assert migration.startup_safe is True
    assert "CREATE SCHEMA IF NOT EXISTS knowledge_agent" in migration.sql
    assert "agent_messages remains authoritative" in migration.sql


def _database() -> Database:
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN", "").strip()
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]
        default_embedding_model = "bge-m3:latest"

    return Database(DisposableConfig)


@pytest.mark.integration
def test_agent_run_evidence_citations_and_capability_are_transactionally_persisted():
    database = _database()
    database.bootstrap()
    suffix = uuid.uuid4().hex
    session = "agent_session_" + suffix
    owner_hash = session_key_hash(session)
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            ("agent-e2e-" + suffix, "Agent E2E " + suffix),
        ).fetchone()["id"])
    thread_id = "ath_" + uuid.uuid4().hex
    run_id = "arun_" + uuid.uuid4().hex
    user_message_id = "amsg_" + uuid.uuid4().hex
    assistant_message_id = "amsg_" + uuid.uuid4().hex
    evidence_id = "ev_" + uuid.uuid4().hex
    database.create_agent_thread(
        thread_id=thread_id, project_id=project_id,
        strategy_key="knowledge_forge", session_key_hash=owner_hash,
    )
    database.create_agent_run(
        run_id=run_id, message_id=user_message_id, thread_id=thread_id,
        session_key_hash=owner_hash, query="任务管理是什么？",
        requested_mode="auto", budget={"candidate_limit": 40},
    )
    assert database.start_agent_run(run_id)["status"] == "running"
    database.record_agent_step(
        run_id=run_id, project_id=project_id, sequence=1,
        step_type="authorizing", status="completed",
        output_summary={"authorized": True},
    )
    tool_call_id = "atc_" + uuid.uuid4().hex
    database.start_agent_tool_call(
        call_id=tool_call_id, run_id=run_id, project_id=project_id,
        tool_name="retrieve.knowledge_forge",
        request_summary={"candidate_limit": 40},
    )
    database.finish_agent_tool_call(
        tool_call_id, status="completed",
        response_summary={"candidate_count": 12},
    )
    with pytest.raises(RuntimeError, match="正在生成回答"):
        database.archive_agent_thread(
            thread_id, session_key_hash=owner_hash,
        )
    database.persist_agent_evidence(run_id, [{
        "evidence_id": evidence_id,
        "run_id": run_id,
        "project_id": project_id,
        "evidence_type": "chunk",
        "strategy_key": "knowledge_forge",
        "snapshot_ref": {
            "strategy": "knowledge_forge", "identifier": "kf-source:test",
            "source_hash": "source", "generation": 0,
            "graph_available": False, "details": {},
        },
        "title": "任务管理",
        "content": "任务管理负责创建和调度任务。",
        "entity_keys": [], "relation_keys": [],
        "source_refs": [{
            "document_id": 1, "filename": "manual.pdf",
            "document_url": "/api/documents/1/download",
        }],
        "retrieval_channels": ["chunk_dense"],
        "raw_scores": {"chunk_dense": 0.9},
        "fused_score": 1.0, "rerank_score": 0.95,
        "provenance_score": 1.0, "provenance_valid": True,
        "rank": 1, "selected": True, "metadata": {},
    }])
    database.complete_agent_run(
        run_id=run_id, message_id=assistant_message_id,
        answer_text="任务管理负责创建和调度任务〔1〕。",
        answer_json={"claims": [{
            "text": "任务管理负责创建和调度任务。",
            "evidence_ids": [evidence_id], "status": "supported",
        }]},
        citations=[{
            "claim_index": 0,
            "claim_text": "任务管理负责创建和调度任务。",
            "evidence_id": evidence_id, "citation_index": 1,
            "status": "supported",
        }],
        effective_mode="text_only",
        stream_events=[{"event_type": "answer_delta", "payload": {"delta": "answer"}}],
    )

    payload = database.get_agent_run_payload(run_id, session_key_hash=owner_hash)
    assert payload is not None
    assert payload["status"] == "completed"
    assert payload["budget"]["candidate_limit"] == 40
    assert len(payload["evidence"]) == 1
    assert len(payload["citations"]) == 1
    assert len(payload["messages"]) == 2
    trace_payload = database.get_agent_run_trace_payload(
        run_id, session_key_hash=owner_hash,
    )
    assert trace_payload is not None
    assert len(trace_payload["steps"]) == 1
    assert len(trace_payload["tool_calls"]) == 1
    assert len(trace_payload["evidence_summaries"]) == 1
    assert trace_payload["evidence_summaries"][0]["evidence_id"] == evidence_id
    assert trace_payload["evidence_summaries"][0]["excerpt"] == "任务管理负责创建和调度任务。"
    assert trace_payload["evidence_summaries"][0]["filename"] == "manual.pdf"
    assert trace_payload["evidence_summaries"][0]["citation_index"] == 1
    assert "evidence" not in trace_payload
    assert "messages" not in trace_payload
    assert database.get_agent_run_trace_payload(
        run_id, session_key_hash="0" * 64,
    ) is None
    history = database.list_agent_threads(
        project_id=project_id, session_key_hash=owner_hash,
    )
    assert [item["id"] for item in history] == [thread_id]
    assert history[0]["title"] == "任务管理是什么？"
    assert history[0]["message_count"] == 2
    assert history[0]["last_message_preview"].startswith("任务管理负责")
    assert history[0]["active_run_id"] == ""
    assert database.list_agent_threads(
        project_id=project_id, session_key_hash="0" * 64,
    ) == []
    assert database.get_agent_run_payload(
        run_id, session_key_hash="0" * 64,
    ) is None
    followup_run_id = "arun_" + uuid.uuid4().hex
    database.create_agent_run(
        run_id=followup_run_id,
        message_id="amsg_" + uuid.uuid4().hex,
        thread_id=thread_id, session_key_hash=owner_hash,
        query="它还负责什么？", requested_mode="auto",
        budget={"candidate_limit": 40},
    )
    context = database.agent_conversation_context(followup_run_id, limit=8)
    assert context["thread_id"] == thread_id
    assert context["project_id"] == project_id
    assert context["total"] == 2
    assert [item["role"] for item in context["items"]] == ["user", "assistant"]
    assert all(item["run_id"] == run_id for item in context["items"])
    database.terminate_agent_run(
        followup_run_id, status="cancelled", error_message="test cleanup",
    )
    archived = database.archive_agent_thread(
        thread_id, session_key_hash=owner_hash,
    )
    assert archived["status"] == "archived"
    assert database.list_agent_threads(
        project_id=project_id, session_key_hash=owner_hash,
    ) == []
    assert database.get_agent_thread_payload(
        thread_id, session_key_hash=owner_hash,
    ) is None
    assert database.get_agent_run_payload(
        run_id, session_key_hash=owner_hash,
    ) is None
    assert database.get_agent_run(run_id) is not None
