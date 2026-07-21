from __future__ import annotations

import json
import os
from typing import Any
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge.database import SCHEMA_SQL


def test_merge_split_schema_keeps_redirects_projection_and_reversible_payload():
    assert "CREATE TABLE IF NOT EXISTS canonical_entity_redirects" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS canonical_relation_projections" in SCHEMA_SQL
    assert "event_type TEXT NOT NULL" in SCHEMA_SQL
    assert "before_state JSONB NOT NULL" in SCHEMA_SQL
    assert "reversed_by_event_id" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS mention_relation_assertions" in SCHEMA_SQL
    assert "source_mention_key TEXT NOT NULL" in SCHEMA_SQL
    assert "target_mention_key TEXT NOT NULL" in SCHEMA_SQL


def test_relation_assertions_are_not_rewritten_by_merge_migration():
    merge_section = SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS entity_resolution_events", 1)[1]
    assert "UPDATE relation_assertions SET source_entity_id" not in merge_section
    assert "UPDATE relation_assertions SET target_entity_id" not in merge_section


def test_product_relation_queries_use_rebuildable_projection():
    from knowledge_forge.database import Database

    for method_name in ("list_project_relations", "list_document_relations",
                        "search_graph_text", "get_relations_by_ids"):
        source = __import__("inspect").getsource(getattr(Database, method_name))
        assert "canonical_relation_projections" in source
        assert "COALESCE(projection.source_entity_id,r.source_entity_id)" in source
        assert "COALESCE(projection.target_entity_id,r.target_entity_id)" in source


def test_relation_projection_uses_active_mention_assignments_before_redirects():
    from knowledge_forge.database import Database

    source = __import__("inspect").getsource(Database._refresh_relation_projections)
    assert "mention_relation_assertions" in source
    assert "source_assignment.canonical_entity_id" in source
    assert "target_assignment.canonical_entity_id" in source
    assert source.index("source_assignment.canonical_entity_id") < source.index(
        "source_redirect.target_entity_id"
    )


def test_merge_split_and_undo_refresh_assignment_derived_projections():
    from knowledge_forge.database import Database

    for method_name in (
        "manual_merge_entities",
        "split_entity_mentions",
        "undo_entity_resolution_event",
    ):
        source = __import__("inspect").getsource(getattr(Database, method_name))
        assert "_refresh_entity_document_projections" in source
        assert "_refresh_assignment_aliases" in source
        assert "_refresh_relation_projections" in source


class _DerivedIndexDb:
    def __init__(self, *, entity_failure: bool = False, relation_failure: bool = False):
        self.entity_failure = entity_failure
        self.relation_failure = relation_failure
        self.retired: list[tuple[int, list[int]]] = []
        self.completed: list[tuple[int, str, str, str]] = []
        self.failures: list[dict[str, Any]] = []

    def get_project(self, project_id: int) -> dict[str, Any]:
        return {"id": project_id, "embedding_model": "embed-v1"}

    def manual_merge_entities(self, project_id: int, source: int, target: int, **values: Any):
        return {"event_id": "merge-1", "affected_entity_ids": [source, target]}

    def split_entity_mentions(
        self, project_id: int, source: int, target: int, mentions: list[int], **values: Any,
    ):
        return {"event_id": "split-1", "affected_entity_ids": [source, target]}

    def undo_entity_resolution_event(self, project_id: int, event_id: str, **values: Any):
        return {"event_id": "undo-1", "affected_entity_ids": [11, 12]}

    def retire_derived_embeddings_for_entity_change(
        self, project_id: int, entity_ids: list[int], model_name: str,
    ) -> None:
        self.retired.append((project_id, entity_ids))

    def get_entities_by_ids(self, project_id: int, ids: list[int]):
        return [
            {
                "id": entity_id,
                "name": f"Entity {entity_id}",
                "entity_type": "Concept",
                "description": "",
                "aliases": [],
                "entity_key": f"entity-{entity_id}",
            }
            for entity_id in ids
        ]

    def list_project_relations_for_entities(self, project_id: int, ids: list[int]):
        return [{
            "id": 71,
            "document_id": 3,
            "chunk_id": 5,
            "relation_key": "relation-71",
            "relation_type": "uses",
            "description": "",
            "source_name": "Entity 11",
            "target_name": "Entity 12",
        }]

    def complete_indexing_failure(
        self, project_id: int, object_type: str, object_key: str, model: str,
    ) -> None:
        self.completed.append((project_id, object_type, object_key, model))

    def record_indexing_failure(self, values: dict[str, Any]) -> None:
        self.failures.append(values)


class _DerivedIndexVectors:
    def __init__(self, db: _DerivedIndexDb):
        self.db = db
        self.entities: list[int] = []
        self.relations: list[int] = []

    def upsert_entity_embedding(self, project_id: int, entity: dict[str, Any], model: str):
        if self.db.entity_failure:
            raise RuntimeError("entity vector unavailable")
        self.entities.append(int(entity["id"]))

    def upsert_relation_embedding(
        self, project_id: int, document_id: int, chunk_id: int,
        relation: dict[str, Any], source_name: str, target_name: str, model: str,
    ):
        if self.db.relation_failure:
            raise RuntimeError("relation vector unavailable")
        self.relations.append(int(relation["id"]))


def _derived_index_service(db: _DerivedIndexDb):
    from knowledge_forge.service import KnowledgeForgeService

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = db
    service.vectors = _DerivedIndexVectors(db)
    service.submit_graph_build = lambda project_id: f"graph-{project_id}"
    return service


def test_manual_merge_refreshes_affected_vectors_before_submitting_graph_rebuild():
    db = _DerivedIndexDb()
    service = _derived_index_service(db)

    result = service.merge_entities(9, 11, 12, reason="same entity", policy_version="p2")

    assert db.retired == [(9, [11, 12])]
    assert service.vectors.entities == [11, 12]
    assert service.vectors.relations == [71]
    assert result["graph_job_id"] == "graph-9"
    assert result["derived_index_failures"] == 0


def test_split_and_undo_persist_vector_failures_without_skipping_graph_rebuild():
    db = _DerivedIndexDb(entity_failure=True, relation_failure=True)
    service = _derived_index_service(db)

    split = service.split_entity_mentions(
        9, 11, 12, [101], reason="different entity", policy_version="p2",
    )
    undo = service.undo_entity_resolution_event(9, "merge-1", reason="reviewed")

    assert split["graph_job_id"] == "graph-9"
    assert undo["graph_job_id"] == "graph-9"
    assert split["derived_index_failures"] == 3
    assert undo["derived_index_failures"] == 3
    assert {item["object_type"] for item in db.failures} == {"entity", "relation"}
    assert all(item["model_name"] == "embed-v1" for item in db.failures)


def test_graph_generation_cannot_become_ready_with_blocking_derived_index_outbox():
    from knowledge_forge.service import KnowledgeForgeService

    class GateDb:
        def get_project(self, project_id: int):
            return {"id": project_id, "graph_generation": 4, "embedding_model": "embed-v1"}

        def list_pending_indexing_failures(self, project_id: int, object_type: str):
            return []

        def list_blocking_indexing_failures(self, project_id: int, generation: int):
            return [{"object_type": "entity", "object_key": "11"}]

        def list_documents(self, project_id: int):
            raise AssertionError("graph IO must not start while derived indexes are blocked")

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = GateDb()
    service.vectors = object()

    with pytest.raises(RuntimeError, match="派生索引待修复"):
        service._rebuild_project_graph_locked(9, target_generation=4)


@pytest.mark.integration
def test_postgres_merge_undo_split_rebuilds_provenance_and_relation_endpoints(
    active_document_revision,
):
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    from knowledge_forge.database import Database

    managed_schema = os.getenv("KNOWLEDGE_FORGE_TEST_SCHEMA", "").strip()
    schema_name = managed_schema or f"kf_merge_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        if not managed_schema:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        try:
            conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema_name)))
            if not managed_schema:
                conn.execute(SCHEMA_SQL)
            project_id = int(conn.execute(
                """INSERT INTO projects(project_key,name,active_policy_hash)
                   VALUES('p-merge','P merge','policy-v1') RETURNING id"""
            ).fetchone()["id"])
            document_id = int(conn.execute(
                """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
                   storage_path,local_path,file_size_bytes,status)
                   VALUES(%s,'doc-hash','doc.txt','doc.txt','/doc','/doc',10,'indexed') RETURNING id""",
                (project_id,),
            ).fetchone()["id"])
            if managed_schema:
                active_document_revision(conn, project_id, document_id)
            chunk_id = int(conn.execute(
                """INSERT INTO chunks(project_id,document_id,chunk_key,content,content_hash)
                   VALUES(%s,%s,'chunk-1','Alpha uses Beta','chunk-hash') RETURNING id""",
                (project_id, document_id),
            ).fetchone()["id"])
            entity_rows = conn.execute(
                """INSERT INTO canonical_entities(
                       project_id,entity_key,name,normalized_name,entity_type)
                   VALUES(%s,'alpha','Alpha','alpha','Concept'),
                         (%s,'beta','Beta','beta','Concept') RETURNING id,entity_key""",
                (project_id, project_id),
            ).fetchall()
            entity_ids = {str(item["entity_key"]): int(item["id"]) for item in entity_rows}
            alpha_id, beta_id = entity_ids["alpha"], entity_ids["beta"]
            mention_rows = conn.execute(
                """INSERT INTO entity_mentions(
                       project_id,document_id,chunk_id,canonical_entity_id,mention_text,
                       normalized_name,entity_type,mention_key)
                   VALUES(%s,%s,%s,%s,'Alpha','alpha','Concept','mention-alpha'),
                         (%s,%s,%s,%s,'Beta','beta','Concept','mention-beta')
                   RETURNING id,canonical_entity_id,mention_key""",
                (
                    project_id, document_id, chunk_id, alpha_id,
                    project_id, document_id, chunk_id, beta_id,
                ),
            ).fetchall()
            mentions_by_key = {str(item["mention_key"]): item for item in mention_rows}
            for mention in mention_rows:
                conn.execute(
                    """INSERT INTO entity_assignments(
                           project_id,mention_id,canonical_entity_id,assignment_generation,
                           policy_version,reason)
                       VALUES(%s,%s,%s,1,'p2','initial')""",
                    (project_id, mention["id"], mention["canonical_entity_id"]),
                )
            conn.execute(
                """INSERT INTO canonical_entity_documents(
                       project_id,canonical_entity_id,document_id)
                   VALUES(%s,%s,%s),(%s,%s,%s)""",
                (project_id, alpha_id, document_id, project_id, beta_id, document_id),
            )
            schema_json = {
                "entity_types": [{"name": "Concept"}],
                "relation_types": [{
                    "name": "uses", "source_types": ["Concept"],
                    "target_types": ["Concept"],
                }],
            }
            conn.execute(
                """INSERT INTO graph_schemas(project_id,version,status,schema_json)
                   VALUES(%s,1,'active',%s::jsonb)""",
                (project_id, json.dumps(schema_json)),
            )
            candidate_id = "candidate-1"
            conn.execute(
                """INSERT INTO relation_candidates(
                       candidate_id,project_id,document_id,chunk_id,candidate_ordinal,
                       source_mention_id,source_name,target_mention_id,target_name,
                       proposed_relation_type,status)
                   VALUES(%s,%s,%s,%s,1,'mention-alpha','Alpha','mention-beta','Beta',
                          'uses','manual_verified')""",
                (candidate_id, project_id, document_id, chunk_id),
            )
            relation_id = int(conn.execute(
                """INSERT INTO relation_assertions(
                       project_id,document_id,chunk_id,source_entity_id,target_entity_id,
                       relation_key,relation_type,verification_status)
                   VALUES(%s,%s,%s,%s,%s,'relation-1','uses','manual_verified') RETURNING id""",
                (project_id, document_id, chunk_id, alpha_id, beta_id),
            ).fetchone()["id"])
            conn.execute(
                """INSERT INTO mention_relation_assertions(
                       id,project_id,document_id,chunk_id,candidate_id,canonical_assertion_id,
                       source_mention_key,target_mention_key,predicate,evidence_text,
                       extraction_policy_version,verification_policy_version)
                   VALUES('fact-1',%s,%s,%s,%s,%s,'mention-alpha','mention-beta','uses',
                          'Alpha uses Beta','p2','p2')""",
                (project_id, document_id, chunk_id, candidate_id, relation_id),
            )
            Database._refresh_relation_projections(conn, project_id, "policy-v1", 0)

            config_values = conninfo_to_dict(dsn)
            config_values["options"] = f"-c search_path={schema_name},public"

            class TestConfig:
                postgres = config_values

            database = Database(TestConfig)
            merged = database.manual_merge_entities(
                project_id, alpha_id, beta_id, reason="same", policy_version="p2",
            )
            projection = conn.execute(
                "SELECT * FROM canonical_relation_projections WHERE assertion_id=%s",
                (relation_id,),
            ).fetchone()
            alpha = conn.execute(
                "SELECT source_count,is_deleted FROM canonical_entities WHERE id=%s",
                (alpha_id,),
            ).fetchone()
            assert projection["source_entity_id"] == beta_id
            assert projection["target_entity_id"] == beta_id
            assert projection["publishable"] is False
            assert alpha == {"source_count": 0, "is_deleted": True}

            database.undo_entity_resolution_event(
                project_id, merged["event_id"], reason="reviewed",
            )
            projection = conn.execute(
                "SELECT * FROM canonical_relation_projections WHERE assertion_id=%s",
                (relation_id,),
            ).fetchone()
            alpha = conn.execute(
                "SELECT source_count,is_deleted FROM canonical_entities WHERE id=%s",
                (alpha_id,),
            ).fetchone()
            assert (projection["source_entity_id"], projection["target_entity_id"]) == (
                alpha_id, beta_id,
            )
            assert projection["publishable"] is True
            assert alpha == {"source_count": 1, "is_deleted": False}

            database.split_entity_mentions(
                project_id, alpha_id, beta_id,
                [int(mentions_by_key["mention-alpha"]["id"])],
                reason="different assignment", policy_version="p2",
            )
            projection = conn.execute(
                "SELECT * FROM canonical_relation_projections WHERE assertion_id=%s",
                (relation_id,),
            ).fetchone()
            assert (projection["source_entity_id"], projection["target_entity_id"]) == (
                beta_id, beta_id,
            )
            assert projection["quarantine_reason"] == "self relation after entity assignment change"
        finally:
            if not managed_schema:
                conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema_name)))
