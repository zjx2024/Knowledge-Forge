from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
import uuid

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.migrations import MIGRATIONS, MigrationRunner
from knowledge_forge.policy import P1_RELATION_SHADOW_POLICY_SET
from knowledge_forge.relation_pipeline import (
    RelationBatchValidationError,
    RelationCandidate,
    RelationPlan,
    VersionedRelationPipeline,
    _sentences,
)


def candidate() -> RelationCandidate:
    return RelationCandidate(
        candidate_id="candidate-1",
        source_mention_id="mention-a",
        target_mention_id="mention-b",
        source_name="Alpha",
        target_name="Beta",
        source_type="Organization",
        target_type="Product",
        allowed_predicates=("produces",),
        allowed_directions=("as_is",),
        context_id="ctx-0-0",
        sentence_start=0,
        sentence_end=0,
        context_start_char=0,
        context_end_char=38,
        source_start_char=0,
        source_end_char=5,
        target_start_char=15,
        target_end_char=19,
    )


def _database_for_dsn(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    database = Database(DisposableConfig)

    @contextmanager
    def connection():
        with psycopg.connect(dsn, row_factory=dict_row) as scoped:
            revision_context = database._revision_context.get()
            if revision_context:
                scoped.execute(
                    "SELECT set_config('knowledge_forge.revision_id',%s,TRUE),"
                    "set_config('knowledge_forge.ingestion_run_id',%s,TRUE),"
                    "set_config('knowledge_forge.revision_lease_token',%s,TRUE)",
                    (
                        revision_context["revision_id"],
                        revision_context["ingestion_run_id"],
                        revision_context["lease_token"],
                    ),
                )
            yield scoped

    database.connection = connection
    return database


def test_relation_shadow_has_durable_leased_work_queue_and_attempt_audit():
    migration_sql = "\n".join(item.sql for item in MIGRATIONS)
    combined = SCHEMA_SQL + migration_sql
    assert "CREATE TABLE IF NOT EXISTS relation_shadow_work_items" in combined
    assert "CREATE TABLE IF NOT EXISTS relation_shadow_attempts" in combined
    for state in (
        "pending", "leased", "completed", "retryable_failed", "terminal_failed", "quarantined",
    ):
        assert state in combined
    for column in (
        "lease_owner", "lease_token", "lease_expires_at", "attempt", "next_attempt_at",
        "accepted_output_hash", "input_fingerprint",
    ):
        assert column in combined


def test_sentence_offsets_preserve_decimal_versions_urls_email_abbreviations_and_ellipsis():
    content = (
        'Version v1.2.3 costs 3.14 USD. Visit https://example.com/a.b or e-mail a.b@example.com. '
        'Dr. Smith said “ready...” Then继续……完成！分块: 只是正文。'
    )
    values = _sentences(content)
    assert "".join(text for _, _, text in values) == content
    assert [text for _, _, text in values] == [
        "Version v1.2.3 costs 3.14 USD. ",
        "Visit https://example.com/a.b or e-mail a.b@example.com. ",
        "Dr. Smith said “ready...” ",
        "Then继续……",
        "完成！",
        "分块: 只是正文。",
    ]
    assert all(content[start:end] == text for start, end, text in values)


def test_duplicate_evidence_requires_exact_offsets_and_hash():
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    content = "Alpha produces Beta. Alpha produces Beta."
    evidence = "Alpha produces Beta."
    second_start = content.rindex(evidence)
    decision = pipeline.validate_decisions(
        [candidate()],
        [{
            "candidate_id": "candidate-1",
            "predicate": "produces",
            "direction": "as_is",
            "evidence": evidence,
            "evidence_start": second_start,
            "evidence_end": second_start + len(evidence),
            "reason": "explicit",
        }],
        content=content,
        logical_schema={"relation_types": [{"name": "produces"}]},
    )[0]
    assert decision.evidence_start == second_start
    assert decision.evidence_end == second_start + len(evidence)
    assert decision.evidence_hash == hashlib.sha256(evidence.encode("utf-8")).hexdigest()

    with pytest.raises(RelationBatchValidationError, match="ambiguous"):
        pipeline.validate_decisions(
            [candidate()],
            [{
                "candidate_id": "candidate-1", "predicate": "produces",
                "direction": "as_is", "evidence": evidence, "reason": "explicit",
            }],
            content=content,
            logical_schema={"relation_types": [{"name": "produces"}]},
        )


def test_evidence_offsets_are_half_open_and_must_match_source_exactly():
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    content = "Alpha produces Beta."
    raw = [{
        "candidate_id": "candidate-1", "predicate": "produces", "direction": "as_is",
        "evidence": content, "evidence_start": 0, "evidence_end": len(content) - 1,
        "reason": "bad end",
    }]
    with pytest.raises(RelationBatchValidationError, match="exact source span"):
        pipeline.validate_decisions(
            [candidate()], raw, content=content,
            logical_schema={"relation_types": [{"name": "produces"}]},
        )


def test_long_unrelated_chunk_prefix_does_not_consume_candidate_batch_budget():
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    content = ("Unrelated background sentence. " * 300) + "Alpha produces Beta."
    alpha_start = content.index("Alpha")
    beta_start = content.index("Beta")
    prepared = pipeline.prepare_shadow(
        context_budget_tokens=2048, project_scope="project:revision",
        chunk_key="long-chunk", content=content,
        entities=[
            {
                "id": "alpha", "name": "Alpha", "type": "Organization",
                "evidence_text": "Alpha", "evidence_start": alpha_start,
                "evidence_end": alpha_start + 5,
            },
            {
                "id": "beta", "name": "Beta", "type": "Product",
                "evidence_text": "Beta", "evidence_start": beta_start,
                "evidence_end": beta_start + 4,
            },
        ],
        logical_schema={
            "relation_types": [{
                "name": "produces", "source_types": ["Organization"],
                "target_types": ["Product"],
            }],
        },
    )
    assert prepared.plan.admitted_count == 1
    assert prepared.plan.deferred == []
    assert len(prepared.batches) == 1
    payload = __import__("json").loads(prepared.batches[0].prompt)
    assert payload["source_contexts"] == [{
        "context_id": prepared.plan.candidates[0].context_id,
        "start": prepared.plan.candidates[0].context_start_char,
        "end": prepared.plan.candidates[0].context_end_char,
        "text": "Alpha produces Beta.",
    }]


def test_dense_candidates_are_streamed_across_batches_without_loss():
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    content = "A relates B relates C relates D relates E relates F."
    names = ["A", "B", "C", "D", "E", "F"]
    entities = []
    cursor = 0
    for name in names:
        start = content.index(name, cursor)
        cursor = start + 1
        entities.append({
            "id": name, "name": name, "type": "GenericEntity",
            "evidence_text": name, "evidence_start": start, "evidence_end": start + 1,
        })
    plan = pipeline.plan(
        project_scope="project:revision", chunk_key="dense", content=content,
        entities=entities,
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
    )
    batches = pipeline.build_batches(
        plan, content=content,
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
        candidate_budget_tokens=2048, max_candidates_per_batch=3,
    )
    planned = [item.candidate_id for item in plan.candidates]
    batched = [candidate_id for batch in batches for candidate_id in batch.candidate_ids]
    assert len(planned) == 15
    assert batched == planned
    assert plan.deferred == []


def test_relation_candidate_limits_are_deterministic_and_conserve_overflow(monkeypatch):
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_targets_per_source", 2)
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_candidates_per_chunk", 5)
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_char_distance", 384)
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    content = "A links B links C links D links E links F."
    entities = []
    cursor = 0
    for name in "ABCDEF":
        start = content.index(name, cursor)
        cursor = start + 1
        entities.append({
            "id": name, "name": name, "type": "GenericEntity",
            "evidence_text": name, "evidence_start": start, "evidence_end": start + 1,
        })
    plan = pipeline.plan(
        project_scope="project:revision", chunk_key="bounded", content=content,
        entities=entities,
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
    )

    assert len(plan.candidates) == 5
    assert plan.qualified_count == len(plan.candidates) + len(plan.overflow)
    assert all(item["reason"] in {"per_source_limit", "chunk_limit"} for item in plan.overflow)
    assert [item.candidate_id for item in plan.candidates] == sorted(
        (item.candidate_id for item in plan.candidates),
        key=lambda candidate_id: next(
            candidate.priority_key for candidate in plan.candidates
            if candidate.candidate_id == candidate_id
        ),
    )


@pytest.mark.parametrize(("count", "expected_batches"), [(20, 1), (50, 2), (100, 4)])
def test_relation_batches_conserve_20_50_100_candidates_with_hard_batch_cap(
    count, expected_batches,
):
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    values = [
        RelationCandidate(
            candidate_id=f"candidate-{index}", source_mention_id=f"source-{index}",
            target_mention_id=f"target-{index}", source_name="A", target_name="B",
            source_type="GenericEntity", target_type="GenericEntity",
            allowed_predicates=("related_to",), allowed_directions=("as_is",),
            context_id="ctx-0-0", sentence_start=0, sentence_end=0,
            context_start_char=0, context_end_char=12, source_start_char=0,
            source_end_char=1, target_start_char=10, target_end_char=11,
        )
        for index in range(count)
    ]
    plan = RelationPlan(
        policy_set=dict(pipeline.policy_set), raw_pairs=count,
        blocked_pairs=0, candidates=values, qualified_count=count,
    )

    batches = pipeline.build_batches(
        plan, content="A relates B.",
        logical_schema={"relation_types": [{"name": "related_to"}]},
        candidate_budget_tokens=100_000, max_candidates_per_batch=100,
    )

    assert len(batches) == expected_batches
    assert all(len(batch.candidate_ids) <= 32 for batch in batches)
    assert [item for batch in batches for item in batch.candidate_ids] == [
        item.candidate_id for item in values
    ]


def test_relation_char_distance_blocks_far_pairs_before_model_calls(monkeypatch):
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_char_distance", 32)
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    content = "Alpha " + ("x" * 100) + " Beta."
    beta = content.index("Beta")
    plan = pipeline.plan(
        project_scope="project:revision", chunk_key="far", content=content,
        entities=[
            {"id": "a", "name": "Alpha", "type": "GenericEntity",
             "evidence_text": "Alpha", "evidence_start": 0, "evidence_end": 5},
            {"id": "b", "name": "Beta", "type": "GenericEntity",
             "evidence_text": "Beta", "evidence_start": beta, "evidence_end": beta + 4},
        ],
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
    )
    assert plan.candidates == []
    assert plan.qualified_count == 0


@pytest.mark.integration
def test_relation_shadow_queue_recovers_leases_retries_and_late_workers_without_product_writes():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database_for_dsn(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"relation-shadow-{suffix}", f"Relation Shadow {suffix}"),
        ).fetchone()["id"])
        document_id = int(conn.execute(
            """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
               storage_path,local_path,file_size_bytes)
               VALUES(%s,%s,'shadow.txt','shadow.txt','/shadow','/shadow',1) RETURNING id""",
            (project_id, f"shadow-{suffix}"),
        ).fetchone()["id"])
    run_id = f"ingestion-{suffix}"
    manifest = {
        "content_hash": f"content-{suffix}", "source_hash": f"source-{suffix}",
        "config_hash": "config", "policy_hash": "policy",
    }
    revision = database.begin_document_revision(document_id, run_id, manifest)
    chunks = database.replace_document_content(
        document_id,
        [{
            "key": "section", "title": "Section", "level": 1,
            "content": "Alpha produces Beta.", "content_hash": "section-hash",
        }],
        [{
            "key": "chunk", "section_key": "section", "part_index": 1,
            "content": "Alpha produces Beta.", "content_hash": "chunk-hash",
        }],
        revision_id=revision["revision_id"], ingestion_run_id=run_id,
        lease_token=revision["lease_token"], manifest=manifest,
    )
    chunk_id = int(chunks[0]["id"])
    shadow_run_id = f"shadow-run-{suffix}"
    work = [
        {
            "id": f"work-{suffix}-{index}", "revision_id": revision["revision_id"],
            "project_id": project_id, "document_id": document_id, "chunk_id": chunk_id,
            "batch_id": f"batch-{index}", "input_fingerprint": f"fingerprint-{index}",
            "candidate_ids": [f"candidate-{index}"], "request_json": {"batch": index},
            "estimated_input_tokens": 10, "max_output_tokens": 10, "max_attempts": 3,
        }
        for index in (1, 2)
    ]

    with database.content_revision_context(
        revision["revision_id"], run_id, revision["lease_token"],
    ):
        database.start_relation_shadow_run({
            "id": shadow_run_id, "project_id": project_id, "document_id": document_id,
            "chunk_id": chunk_id, "control_policy_hash": "control",
            "treatment_policy_hash": "treatment", "policy_set": P1_RELATION_SHADOW_POLICY_SET,
            "input_fingerprint": f"run-fingerprint-{suffix}",
        })
        database.prepare_relation_shadow_work(shadow_run_id, work)
        with database.connection() as conn:
            product_before = {
                table: int(conn.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE project_id=%s", (project_id,),
                ).fetchone()["count"])
                for table in (
                    "relation_assertions", "relation_candidates",
                    "mention_relation_assertions", "canonical_relation_projections",
                    "vector_embeddings", "jobs",
                )
            }

        expired = database.lease_relation_shadow_work(shadow_run_id, "worker-expired")
        assert expired and expired["batch_id"] == "batch-1"
        concurrent = database.lease_relation_shadow_work(shadow_run_id, "worker-concurrent")
        assert concurrent and concurrent["batch_id"] == "batch-2"
        assert concurrent["id"] != expired["id"]
        with database.connection() as conn:
            conn.execute(
                "UPDATE relation_shadow_work_items SET lease_expires_at=NOW()-INTERVAL '1 second' WHERE id=%s",
                (expired["id"],),
            )
        recovered = database.lease_relation_shadow_work(shadow_run_id, "worker-recovery")
        assert recovered and recovered["id"] == expired["id"]
        accepted = database.settle_relation_shadow_work(
            recovered["id"], recovered["lease_token"],
            output={"decisions": [{"candidate_id": "candidate-1"}]},
        )
        assert accepted["status"] == "completed"
        late = database.settle_relation_shadow_work(
            expired["id"], expired["lease_token"],
            output={"decisions": [{"candidate_id": "candidate-1", "late": True}]},
        )
        assert late["accepted_output_hash"] == accepted["accepted_output_hash"]

        failed = database.settle_relation_shadow_work(
            concurrent["id"], concurrent["lease_token"], error_message="temporary model outage",
        )
        assert failed["status"] == "retryable_failed"
        retried = database.lease_relation_shadow_work(shadow_run_id, "worker-retry")
        assert retried and retried["attempt"] == 2
        database.settle_relation_shadow_work(
            retried["id"], retried["lease_token"],
            output={"decisions": [{"candidate_id": "candidate-2"}]},
        )

        rows = database.list_relation_shadow_work(shadow_run_id)
        assert [row["status"] for row in rows] == ["completed", "completed"]
        assert all(float(row["accepted_latency_ms"] or 0) >= 0 for row in rows)
        result = database.finish_relation_shadow_run(
            shadow_run_id, candidates=[], decisions=[],
            metrics={"classified": 2, "model_calls": 4},
        )
        repeated = database.finish_relation_shadow_run(
            shadow_run_id, candidates=[], decisions=[],
            metrics={"classified": 2, "model_calls": 4},
        )
        assert result["artifact_hash"] == repeated["artifact_hash"]
        assert result["snapshot_complete"] is True

        terminal_run_id = f"terminal-shadow-run-{suffix}"
        database.start_relation_shadow_run({
            "id": terminal_run_id, "project_id": project_id, "document_id": document_id,
            "chunk_id": chunk_id, "control_policy_hash": "control",
            "treatment_policy_hash": "treatment", "policy_set": P1_RELATION_SHADOW_POLICY_SET,
            "input_fingerprint": f"terminal-fingerprint-{suffix}",
        })
        terminal_work = {
            **work[0], "id": f"terminal-work-{suffix}", "batch_id": "terminal-batch",
            "input_fingerprint": f"terminal-work-fingerprint-{suffix}", "max_attempts": 1,
        }
        database.prepare_relation_shadow_work(terminal_run_id, [terminal_work])
        terminal_lease = database.lease_relation_shadow_work(terminal_run_id, "terminal-worker")
        terminal = database.settle_relation_shadow_work(
            terminal_lease["id"], terminal_lease["lease_token"], error_message="permanent failure",
        )
        assert terminal["status"] == "terminal_failed"
        with pytest.raises(ValueError, match="failed work item"):
            database.finish_relation_shadow_run(
                terminal_run_id, candidates=[], decisions=[], metrics={}, status="completed",
            )
        terminal_result = database.finish_relation_shadow_run(
            terminal_run_id, candidates=[], decisions=[], metrics={}, status="terminal_failed",
        )
        assert terminal_result["snapshot_complete"] is True
        assert terminal_result["failed_work_items"] == 1

        with database.connection() as conn:
            attempts = conn.execute(
                """SELECT attempt_no,status FROM relation_shadow_attempts
                   WHERE work_item_id=%s ORDER BY attempt_no""",
                (expired["id"],),
            ).fetchall()
            assert attempts == [
                {"attempt_no": 1, "status": "superseded"},
                {"attempt_no": 2, "status": "accepted"},
            ]
            product_after = {
                table: int(conn.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE project_id=%s", (project_id,),
                ).fetchone()["count"])
                for table in (
                    "relation_assertions", "relation_candidates",
                    "mention_relation_assertions", "canonical_relation_projections",
                    "vector_embeddings", "jobs",
                )
            }
        assert product_after == product_before
