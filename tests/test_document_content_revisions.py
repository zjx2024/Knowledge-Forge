from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import uuid

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge.database import (
    Database,
    RevisionManifestMismatch,
    SCHEMA_SQL,
    StaleContentRevision,
)
from knowledge_forge.migrations import MIGRATIONS, MigrationRunner
from knowledge_forge.policy import CONTROL_POLICY_SET


def _database_for_dsn(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _content(version: int):
    return ([{
        "key": "section", "title": "Section", "level": 1,
        "content": f"content {version}", "content_hash": f"section-{version}",
    }], [{
        "key": "chunk", "section_key": "section", "content": f"content {version}",
        "content_hash": f"chunk-{version}", "part_index": 1,
    }])


def _manifest(version: int) -> dict[str, str]:
    return {
        "content_hash": f"content-{version}", "source_hash": "source",
        "config_hash": "config", "policy_hash": "policy",
    }


def test_reingestion_path_is_revisioned_and_non_destructive():
    source = __import__("inspect").getsource(Database.replace_document_content)
    assert "DELETE FROM chunks WHERE document_id=%s" not in source
    assert "DELETE FROM relation_assertions WHERE document_id=%s" not in source
    assert "document_content_revisions" in MIGRATIONS[0].sql
    assert "schema_migrations" in __import__("inspect").getsource(MigrationRunner)


def test_revision_audit_retention_cleanup_is_opt_in():
    from knowledge_forge.config import Config

    assert Config.indexing_failure_cleanup_enabled is False


@pytest.mark.integration
def test_revisioned_reingestion_preserves_audit_and_rejects_stale_workers():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database_for_dsn(dsn)
    database.bootstrap()
    with psycopg.connect(dsn, row_factory=dict_row) as setup:
        project_id = int(setup.execute(
            "INSERT INTO projects(project_key,name) VALUES('readiness','Readiness') RETURNING id"
        ).fetchone()["id"])
        document_id = int(setup.execute(
            """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
               storage_path,local_path,file_size_bytes)
               VALUES(%s,'hash','a.txt','a.txt','/a','/a',1) RETURNING id""",
            (project_id,),
        ).fetchone()["id"])

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

    def publish(version: int, run_id: str):
        revision = database.begin_document_revision(document_id, run_id, _manifest(version))
        sections, chunks = _content(version)
        inserted = database.replace_document_content(
            document_id, sections, chunks,
            revision_id=revision["revision_id"], ingestion_run_id=run_id,
            lease_token=revision["lease_token"], manifest=_manifest(version),
        )
        database.activate_document_revision(
            document_id, revision["revision_id"], run_id, revision["lease_token"],
            reason="test", document_values={"status": "indexed"},
        )
        return revision, inserted[0]

    first, first_chunk = publish(1, "run-1")
    with database.content_revision_context(first["revision_id"], "run-1", first["lease_token"]):
        # The revision is already active, so a late write with its old lease is rejected.
        with pytest.raises(psycopg.errors.OperationalError):
            with database.connection() as conn:
                conn.execute(
                    "INSERT INTO relation_shadow_runs(id,project_id,document_id,chunk_id,control_policy_hash,treatment_policy_hash,policy_set) VALUES('late',%s,%s,%s,'c','t','{}')",
                    (project_id, document_id, first_chunk["id"]),
                )

    # Historical audit written without a worker context remains immutable and queryable.
    with database.connection() as conn:
        conn.execute(
            """INSERT INTO relation_shadow_runs(
               id,project_id,document_id,chunk_id,revision_id,control_policy_hash,
               treatment_policy_hash,policy_set,status)
               VALUES('audit-run',%s,%s,%s,%s,'control','treatment','{}','completed')""",
            (project_id, document_id, first_chunk["id"], first["revision_id"]),
        )
        before = conn.execute(
            "SELECT row_to_json(r)::text AS payload FROM relation_shadow_runs r WHERE id='audit-run'"
        ).fetchone()["payload"]
        before_digest = hashlib.sha256(before.encode()).hexdigest()

    for version in (2, 3):
        publish(version, f"run-{version}")

    revisions = database.list_document_revisions(document_id)
    assert [row["status"] for row in revisions[:3]] == ["active", "retired", "retired"]
    assert len({row["revision_id"] for row in revisions}) == 3
    with database.connection() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE document_id=%s", (document_id,)).fetchone()["n"] == 3
        after = conn.execute(
            "SELECT row_to_json(r)::text AS payload FROM relation_shadow_runs r WHERE id='audit-run'"
        ).fetchone()["payload"]
        assert hashlib.sha256(after.encode()).hexdigest() == before_digest
        with pytest.raises(psycopg.errors.OperationalError):
            conn.execute(
                "UPDATE chunks SET content='late mutation' WHERE id=%s",
                (first_chunk["id"],),
            )
        conn.rollback()
    assert [row["content"] for row in database.list_document_chunks(document_id)] == ["content 3"]

    snapshot = database.create_policy_snapshot(
        project_id, dict(CONTROL_POLICY_SET), expected_documents=1,
        snapshot_manifest={"purpose": "revision-freeze"},
    )
    with database.connection() as conn:
        frozen_revision = conn.execute(
            "SELECT revision_id FROM policy_snapshot_documents WHERE snapshot_id=%s",
            (snapshot["id"],),
        ).fetchone()["revision_id"]
    assert frozen_revision == revisions[0]["revision_id"]

    pending = database.begin_document_revision(document_id, "run-activation-fails", _manifest(7))
    with pytest.raises(psycopg.errors.UndefinedColumn):
        database.activate_document_revision(
            document_id, pending["revision_id"], "run-activation-fails",
            pending["lease_token"], reason="rollback", document_values={"not_a_column": "x"},
        )
    with database.connection() as conn:
        assert conn.execute(
            "SELECT active_revision_id FROM documents WHERE id=%s", (document_id,),
        ).fetchone()["active_revision_id"] == frozen_revision
    database.fail_document_revision(
        pending["revision_id"], "run-activation-fails", pending["lease_token"], "expected",
    )

    idempotent = database.begin_document_revision(document_id, "run-3", _manifest(3))
    assert idempotent["already_active"] is True
    with pytest.raises(RevisionManifestMismatch):
        database.begin_document_revision(document_id, "run-3", _manifest(99))

    failed = database.begin_document_revision(document_id, "run-failed", _manifest(4))
    database.fail_document_revision(
        failed["revision_id"], "run-failed", failed["lease_token"], "chunk failed",
    )
    assert database.list_document_revisions(document_id)[0]["status"] == "failed"
    with database.connection() as conn:
        active = conn.execute(
            "SELECT active_revision_id FROM documents WHERE id=%s", (document_id,)
        ).fetchone()["active_revision_id"]
        assert active != failed["revision_id"]

    older = database.begin_document_revision(document_id, "run-older", _manifest(5))
    newer = database.begin_document_revision(document_id, "run-newer", _manifest(6))
    with pytest.raises(StaleContentRevision):
        database.activate_document_revision(
            document_id, older["revision_id"], "run-older", older["lease_token"],
            reason="late", document_values={},
        )
    database.fail_document_revision(
        older["revision_id"], "run-older", older["lease_token"], "superseded",
    )
    database.fail_document_revision(
        newer["revision_id"], "run-newer", newer["lease_token"], "test cleanup",
    )

    with database.connection() as conn:
        audit_before = conn.execute(
            "SELECT jsonb_agg(to_jsonb(c) ORDER BY c.id)::text AS payload "
            "FROM chunks c WHERE document_id=%s",
            (document_id,),
        ).fetchone()["payload"]
        generation_before = int(conn.execute(
            "SELECT graph_generation FROM projects WHERE id=%s", (project_id,),
        ).fetchone()["graph_generation"])
    deleted = database.soft_delete_document(document_id)
    assert deleted and deleted["is_deleted"] is True
    assert database.list_document_chunks(document_id) == []
    assert database.soft_delete_document(document_id) is None
    with database.connection() as conn:
        audit_after = conn.execute(
            "SELECT jsonb_agg(to_jsonb(c) ORDER BY c.id)::text AS payload "
            "FROM chunks c WHERE document_id=%s",
            (document_id,),
        ).fetchone()["payload"]
        assert audit_after == audit_before
        assert conn.execute(
            "SELECT graph_generation FROM projects WHERE id=%s", (project_id,),
        ).fetchone()["graph_generation"] == generation_before + 1
        assert conn.execute(
            "SELECT revision_id FROM policy_snapshot_documents WHERE snapshot_id=%s",
            (snapshot["id"],),
        ).fetchone()["revision_id"] == frozen_revision


@pytest.mark.integration
def test_bootstrap_is_restart_safe_after_revision_migration():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database_for_dsn(dsn)
    database.bootstrap()
    database.bootstrap()
    with database.connection() as conn:
        rows = conn.execute(
            """SELECT version,COUNT(*) AS count FROM schema_migrations
               GROUP BY version ORDER BY version"""
        ).fetchall()
        assert rows == [
            {"version": "0001_document_content_revisions", "count": 1},
            {"version": "0002_relation_shadow_work_queue", "count": 1},
                {"version": "0003_entity_shadow_pipeline", "count": 1},
                {"version": "0004_policy_snapshot_orchestrator", "count": 1},
                {"version": "0005_release_gate_artifacts", "count": 1},
                {"version": "0006_graph_namespace_manifest", "count": 1},
                {"version": "0007_indexing_failure_status", "count": 1},
            ]
        assert conn.execute(
            "SELECT to_regclass('uq_chunks_revision_key') AS index_name"
        ).fetchone()["index_name"] == "uq_chunks_revision_key"


@pytest.mark.integration
def test_revision_composite_foreign_keys_reject_cross_scope_facts():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        if not conn.execute("SELECT to_regclass('schema_migrations') AS table_name").fetchone()["table_name"]:
            conn.execute(SCHEMA_SQL)
        MigrationRunner().apply(conn)
        projects = [int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"p-{index}", f"P {index}"),
        ).fetchone()["id"]) for index in (1, 2)]
        documents = [int(conn.execute(
            """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
               storage_path,local_path,file_size_bytes) VALUES(%s,%s,%s,%s,'/x','/x',1) RETURNING id""",
            (projects[index], f"h-{index}", f"{index}.txt", f"{index}.txt"),
        ).fetchone()["id"]) for index in (0, 1)]
        revisions = []
        for index in (0, 1):
            revision_id = f"revision-{index}"
            conn.execute(
                """INSERT INTO document_content_revisions(
                   revision_id,project_id,document_id,ingestion_run_id,content_version,
                   content_hash,source_hash,config_hash,policy_hash,manifest_hash,manifest,status,
                   lease_token,lease_expires_at)
                   VALUES(%s,%s,%s,%s,1,'c','s','x','p',%s,'{}','building',%s,NOW()+INTERVAL '1 hour')""",
                (revision_id, projects[index], documents[index], f"run-{index}", f"m-{index}", f"lease-{index}"),
            )
            revisions.append(revision_id)
        conn.execute(
            "SELECT set_config('knowledge_forge.revision_id',%s,TRUE),"
            "set_config('knowledge_forge.ingestion_run_id','run-0',TRUE),"
            "set_config('knowledge_forge.revision_lease_token','lease-0',TRUE)",
            (revisions[0],),
        )
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            conn.execute(
                """INSERT INTO sections(project_id,document_id,revision_id,section_key,title,content,content_hash)
                   VALUES(%s,%s,%s,'bad','bad','bad','bad')""",
                (projects[1], documents[0], revisions[0]),
            )
        conn.rollback()


@pytest.mark.integration
def test_revision_lineage_guards_cover_all_document_fact_tables():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    expected_trigger_tables = {
        "sections", "chunks", "vector_embeddings", "indexing_failures",
        "entity_mentions", "canonical_entity_documents",
        "entity_resolution_decisions", "entity_type_assertions", "entity_assignments",
        "relation_assertions", "canonical_relation_projections", "relation_candidates",
        "mention_relation_assertions", "relation_verification_batches",
        "relation_verification_events", "relation_shadow_runs",
        "relation_shadow_decisions", "relation_shadow_work_items",
        "entity_shadow_assignments",
    }
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        guarded = {
            row["table_name"] for row in conn.execute(
                """SELECT event_object_table AS table_name
                   FROM information_schema.triggers
                   WHERE trigger_schema=current_schema()
                     AND action_statement ILIKE '%kf_guard_revision_write%'"""
            ).fetchall()
        }
        assert guarded == expected_trigger_tables
        lineage_tables = {
            row["table_name"] for row in conn.execute(
                """SELECT DISTINCT conrelid::regclass::text AS table_name
                   FROM pg_constraint
                   WHERE contype='f' AND confrelid='document_content_revisions'::regclass"""
            ).fetchall()
        }
        assert {
            "sections", "chunks", "vector_embeddings", "indexing_failures",
            "entity_mentions", "canonical_entity_documents",
            "entity_resolution_decisions", "entity_type_assertions",
            "relation_assertions", "relation_candidates", "mention_relation_assertions",
            "relation_verification_batches", "relation_verification_events",
            "relation_shadow_runs", "relation_shadow_work_items",
            "policy_snapshot_documents",
        } <= lineage_tables


def test_restore_path_requires_a_new_active_revision():
    source = __import__("inspect").getsource(
        __import__("knowledge_forge.service", fromlist=["KnowledgeForgeService"]).KnowledgeForgeService.save_upload
    )
    assert '"active_revision_id": None' in source


@pytest.mark.integration
def test_manual_merge_only_changes_active_revision_assignments():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database_for_dsn(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"merge-{suffix}", f"Merge {suffix}"),
        ).fetchone()["id"])
        document_id = int(conn.execute(
            """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
               storage_path,local_path,file_size_bytes)
               VALUES(%s,%s,'merge.txt','merge.txt','/merge','/merge',1) RETURNING id""",
            (project_id, f"hash-{suffix}"),
        ).fetchone()["id"])
        entities = conn.execute(
            """INSERT INTO canonical_entities(
                   project_id,entity_key,name,normalized_name,entity_type)
               VALUES(%s,%s,'Alpha','alpha','Concept'),
                     (%s,%s,'Beta','beta','Concept')
               RETURNING id,entity_key""",
            (project_id, f"alpha-{suffix}", project_id, f"beta-{suffix}"),
        ).fetchall()
    entity_ids = {str(row["entity_key"]): int(row["id"]) for row in entities}
    alpha_id = entity_ids[f"alpha-{suffix}"]
    beta_id = entity_ids[f"beta-{suffix}"]

    def publish(version: int) -> tuple[dict, dict]:
        run_id = f"merge-run-{suffix}-{version}"
        revision = database.begin_document_revision(document_id, run_id, _manifest(version))
        sections, chunks = _content(version)
        chunk = database.replace_document_content(
            document_id, sections, chunks,
            revision_id=revision["revision_id"], ingestion_run_id=run_id,
            lease_token=revision["lease_token"], manifest=_manifest(version),
        )[0]
        with database.content_revision_context(
            revision["revision_id"], run_id, revision["lease_token"],
        ):
            database.persist_entity_resolution_batch(
                [
                    {
                        "project_id": project_id, "document_id": document_id,
                        "chunk_id": chunk["id"], "canonical_entity_id": alpha_id,
                        "mention_text": "Alpha", "normalized_name": "alpha",
                        "entity_type": "Concept", "evidence_text": "Alpha",
                        "evidence_start": 0, "evidence_end": 5,
                    },
                    {
                        "project_id": project_id, "document_id": document_id,
                        "chunk_id": chunk["id"], "canonical_entity_id": beta_id,
                        "mention_text": "Beta", "normalized_name": "beta",
                        "entity_type": "Concept", "evidence_text": "Beta",
                        "evidence_start": 6, "evidence_end": 10,
                    },
                ],
                [],
            )
        database.activate_document_revision(
            document_id, revision["revision_id"], run_id, revision["lease_token"],
            reason="merge revision test", document_values={"status": "indexed"},
        )
        return revision, chunk

    first, _ = publish(21)
    second, _ = publish(22)
    with database.connection() as conn:
        old_before = conn.execute(
            """SELECT jsonb_agg(to_jsonb(a) ORDER BY a.id)::text AS payload
               FROM entity_assignments a JOIN entity_mentions m ON m.id=a.mention_id
               WHERE m.revision_id=%s""",
            (first["revision_id"],),
        ).fetchone()["payload"]

    database.manual_merge_entities(
        project_id, alpha_id, beta_id, reason="active revision only",
        policy_version="revision-test",
    )

    with database.connection() as conn:
        old_after = conn.execute(
            """SELECT jsonb_agg(to_jsonb(a) ORDER BY a.id)::text AS payload
               FROM entity_assignments a JOIN entity_mentions m ON m.id=a.mention_id
               WHERE m.revision_id=%s""",
            (first["revision_id"],),
        ).fetchone()["payload"]
        current_targets = conn.execute(
            """SELECT DISTINCT a.canonical_entity_id
               FROM entity_assignments a
               JOIN entity_mentions m ON m.id=a.mention_id
               JOIN documents d ON d.id=m.document_id AND d.active_revision_id=m.revision_id
               WHERE a.project_id=%s AND a.active=TRUE""",
            (project_id,),
        ).fetchall()
        assert old_after == old_before
        assert {int(row["canonical_entity_id"]) for row in current_targets} == {beta_id}
        assert conn.execute(
            """SELECT COUNT(*) AS count FROM canonical_relation_projections projection
               JOIN relation_assertions assertion ON assertion.id=projection.assertion_id
               WHERE assertion.project_id=%s AND assertion.revision_id<>%s""",
            (project_id, second["revision_id"]),
        ).fetchone()["count"] == 0
