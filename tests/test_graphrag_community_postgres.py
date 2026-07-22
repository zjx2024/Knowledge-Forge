from __future__ import annotations

import os

import psycopg
import pytest
from psycopg.rows import dict_row

from knowledge_forge.database import Database
from knowledge_forge.graphrag_community import CommunityConfig, CommunityService
from knowledge_forge.graph_policy import GRAPH_POLICY_HASH, GRAPH_POLICY_VERSION
from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.relation_verification import RELATION_VERIFICATION_POLICY_VERSION


def test_graphrag_community_migration_is_additive_and_isolated():
    migration = next(
        item for item in MIGRATIONS if item.version == "0010_graphrag_communities"
    )
    sql = migration.sql.upper()
    assert "CREATE TABLE GRAPHRAG_COMMUNITY_RUNS" in sql
    assert "CREATE TABLE GRAPHRAG_COMMUNITIES" in sql
    assert "CREATE TABLE GRAPHRAG_COMMUNITY_MEMBERS" in sql
    assert "CREATE TABLE GRAPHRAG_COMMUNITY_EDGES" in sql
    assert "DROP " not in sql
    assert "UPDATE PROJECTS" not in sql


class RecordingRows:
    def __init__(self, *, one=None, rows=None):
        self.one = one
        self.rows = rows or []

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.rows


class SnapshotConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        text = str(statement)
        self.calls.append((text, tuple(params or ())))
        if text.startswith("SET TRANSACTION"):
            return RecordingRows()
        if "FROM projects" in text:
            return RecordingRows(one={
                "id": 7,
                "graph_status": "ready",
                "graph_generation": 4,
                "graph_ready_generation": 4,
                "graph_policy_version": "graph-policy-v1",
                "graph_policy_hash": "policy",
                "active_policy_hash": "policy",
                "active_schema_version": 1,
                "entity_extraction_policy_version": "entity-extraction-p0-v1",
                "entity_resolution_policy_version": "entity-resolution-p0-v1",
                "relation_candidate_policy_version": "relation-candidate-p0-v1",
                "relation_classifier_policy_version": "relation-classifier-p0-v1",
                "verification_policy_version": "relation-verifier-p0-v1",
                "schema_policy_version": "schema-policy-p0-v1",
            })
        return RecordingRows(rows=[])


def test_control_snapshot_query_reuses_fail_closed_active_revision_semantics(monkeypatch):
    connection = SnapshotConnection()
    database = Database()

    class Context:
        def __enter__(self):
            return connection

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(database, "connection", lambda: Context())
    monkeypatch.setattr(
        "knowledge_forge.database.is_current_graph", lambda _project: True
    )

    snapshot = database.export_graphrag_control_snapshot(7)

    assert snapshot.project_id == 7
    relation_query, params = connection.calls[-1]
    normalized_query = " ".join(relation_query.split())
    assert "documents d ON d.id=r.document_id AND d.active_revision_id=r.revision_id" in normalized_query
    assert "canonical_relation_projections" in relation_query
    assert "COALESCE(projection.publishable,TRUE)=TRUE" in relation_query
    assert "metadata->>'verification_policy_version'" in relation_query
    assert "entity_shadow" not in relation_query
    assert "relation_shadow" not in relation_query
    assert params == (
        7,
        "manual_verified",
        ["llm_verified", "rules_verified"],
        RELATION_VERIFICATION_POLICY_VERSION,
    )


@pytest.mark.integration
def test_detect_persists_shadow_without_mutating_control_graph(active_document_revision):
    dsn = os.environ["KNOWLEDGE_FORGE_TEST_POSTGRES_DSN"]

    class DisposableConfig:
        postgres = psycopg.conninfo.conninfo_to_dict(dsn)

    database = Database(DisposableConfig)
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        project = connection.execute(
            """INSERT INTO projects(
               project_key,name,graph_status,graph_generation,graph_ready_generation,
               graph_policy_version,graph_policy_hash,active_policy_hash)
               VALUES('community-test','community-test','ready',1,1,%s,%s,%s)
               RETURNING *""",
            (GRAPH_POLICY_VERSION, GRAPH_POLICY_HASH, GRAPH_POLICY_HASH),
        ).fetchone()
        document = connection.execute(
            """INSERT INTO documents(
               project_id,doc_hash,original_filename,stored_filename,storage_path,
               local_path,file_size_bytes,status)
               VALUES(%s,'hash','fixture.txt','fixture.txt','http://example/fixture.txt',
                      '/tmp/fixture.txt',1,'indexed') RETURNING *""",
            (project["id"],),
        ).fetchone()
        revision_id = active_document_revision(connection, project["id"], document["id"])
        chunk = connection.execute(
            """INSERT INTO chunks(
               project_id,document_id,chunk_key,content,content_hash,revision_id)
               VALUES(%s,%s,'chunk-1','fixture','chunk-hash',%s) RETURNING *""",
            (project["id"], document["id"], revision_id),
        ).fetchone()
        entities = []
        for ordinal in range(1, 7):
            entities.append(connection.execute(
                """INSERT INTO canonical_entities(
                   project_id,entity_key,name,normalized_name,entity_type)
                   VALUES(%s,%s,%s,%s,'Concept') RETURNING *""",
                (project["id"], f"entity-{ordinal}", f"Entity {ordinal}", f"entity {ordinal}"),
            ).fetchone())
        assertion_ids = []
        ordinal = 0
        for source, target, repeats in (
            (0, 1, 3), (1, 2, 3), (0, 2, 3),
            (3, 4, 3), (4, 5, 3), (3, 5, 3), (2, 3, 1),
        ):
            for _ in range(repeats):
                ordinal += 1
                assertion = connection.execute(
                    """INSERT INTO relation_assertions(
                       project_id,document_id,chunk_id,source_entity_id,target_entity_id,
                       relation_key,relation_type,verification_status,revision_id,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,'related_to','rules_verified',%s,%s::jsonb)
                       RETURNING id""",
                    (
                        project["id"], document["id"], chunk["id"],
                        entities[source]["id"], entities[target]["id"], f"relation-{ordinal}",
                        revision_id,
                        '{"verification_policy_version":"relation-verifier-p0-v1"}',
                    ),
                ).fetchone()
                assertion_ids.append(assertion["id"])
        connection.commit()

    before = database.export_graphrag_control_snapshot(project["id"])
    run = CommunityService(database).detect(
        project["id"], CommunityConfig(seed=42, max_cluster_size=10)
    )
    after = database.export_graphrag_control_snapshot(project["id"])
    shown = CommunityService(database).show(run["run_id"])

    assert run["status"] == "completed"
    assert run["community_count"] > 0
    assert shown["current"] is True
    assert shown["communities"]
    assert shown["members"]
    assert shown["edges"]
    assert sorted(item["assertion_id"] for item in before.relations) == assertion_ids
    assert before == after

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        assert connection.execute("SELECT COUNT(*) AS count FROM entity_shadow_namespaces").fetchone()["count"] == 0
        assert connection.execute("SELECT COUNT(*) AS count FROM relation_shadow_runs").fetchone()["count"] == 0
        persisted = connection.execute(
            "SELECT COUNT(*) AS count FROM graphrag_community_edges WHERE run_id=%s",
            (run["run_id"],),
        ).fetchone()["count"]
        assert persisted == 7
        member_count = connection.execute(
            "SELECT COUNT(*) AS count FROM graphrag_community_members WHERE run_id=%s",
            (run["run_id"],),
        ).fetchone()["count"]

    retried = database.finalize_graphrag_community_run(
        run_id=run["run_id"],
        snapshot=before,
        input_hash=run["input_hash"],
        status=run["status"],
        communities=[],
        edges=[],
        metadata={},
    )

    assert retried["status"] == "completed"
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        assert connection.execute(
            "SELECT COUNT(*) AS count FROM graphrag_community_edges WHERE run_id=%s",
            (run["run_id"],),
        ).fetchone()["count"] == persisted
        assert connection.execute(
            "SELECT COUNT(*) AS count FROM graphrag_community_members WHERE run_id=%s",
            (run["run_id"],),
        ).fetchone()["count"] == member_count


@pytest.mark.integration
def test_detect_marks_changed_generation_superseded(monkeypatch):
    dsn = os.environ["KNOWLEDGE_FORGE_TEST_POSTGRES_DSN"]

    class DisposableConfig:
        postgres = psycopg.conninfo.conninfo_to_dict(dsn)

    database = Database(DisposableConfig)
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        project = connection.execute(
            """INSERT INTO projects(
               project_key,name,graph_status,graph_generation,graph_ready_generation,
               graph_policy_version,graph_policy_hash,active_policy_hash)
               VALUES('empty-community','empty-community','ready',1,1,%s,%s,%s)
               RETURNING *""",
            (GRAPH_POLICY_VERSION, GRAPH_POLICY_HASH, GRAPH_POLICY_HASH),
        ).fetchone()
        connection.commit()

    original = database.create_graphrag_community_run

    def mutate_after_create(**values):
        row = original(**values)
        with database.connection() as connection:
            connection.execute(
                "UPDATE projects SET graph_generation=graph_generation+1 WHERE id=%s",
                (project["id"],),
            )
        return row

    monkeypatch.setattr(database, "create_graphrag_community_run", mutate_after_create)
    run = CommunityService(database).detect(project["id"], CommunityConfig())

    assert run["status"] == "superseded"
    assert CommunityService(database).show(run["run_id"])["current"] is False
