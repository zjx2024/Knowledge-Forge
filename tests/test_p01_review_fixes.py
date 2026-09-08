from contextlib import contextmanager
import os
import uuid

import pytest
import psycopg
from psycopg import sql

from knowledge_forge.construction_strategies import KNOWLEDGE_FORGE_STRATEGY
from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.graph_policy import GraphNotReadyError
from knowledge_forge.graph_manifest import build_graph_manifest
from knowledge_forge.graph_policy import GRAPH_POLICY_VERSION
from knowledge_forge.relation_verification import PUBLISHABLE_RELATION_STATUSES
from knowledge_forge.relation_runtime import (
    LOADED_RELATION_CODE_DIGEST,
    relation_runtime_treatment_digest,
)
from knowledge_forge.service import KnowledgeForgeService


class Rows:
    def __init__(self, *, one=None, rows=None, rowcount=0):
        self._one = one or {}
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._rows


class SearchDatabase:
    def __init__(self):
        self.snapshot_calls = []

    def get_project(self, project_id):
        return {"id": project_id, "embedding_model": "embed"}

    def assert_project_construction_strategy(self, project_id, expected):
        assert project_id == 4
        assert expected == KNOWLEDGE_FORGE_STRATEGY
        return expected

    def get_project_strategy_graph(self, project_id, strategy):
        assert project_id == 4
        assert strategy == KNOWLEDGE_FORGE_STRATEGY
        return {
            "status": "ready", "generation": 3, "ready_generation": 3,
            "policy_summary": {
                "entity_namespace_id": "namespace-current",
                "entity_metrics": {"relation_v2_run_id": "relation-run-current"},
            },
        }

    def get_grounded_entity_namespace(self, namespace_id):
        assert namespace_id == "namespace-current"
        return {
            "id": namespace_id, "project_id": 4, "status": "ready",
            "snapshot_complete": True, "embedding_model": "embed",
        }

    def get_grounded_relation_run(self, run_id):
        assert run_id == "relation-run-current"
        return {
            "id": run_id, "project_id": 4, "status": "ready",
            "fingerprint_components": {
                "materialization_root": {
                    "code_tree": LOADED_RELATION_CODE_DIGEST,
                    "treatment": relation_runtime_treatment_digest(),
                },
            },
        }

    def search_knowledge_forge_entities(self, namespace_id, query, limit):
        self.snapshot_calls.append(("entities", namespace_id, query))
        return [{
            "canonical_key": "entity-current", "name": "Alpha",
            "primary_type": "Thing", "aliases": [], "metadata": {},
            "source_count": 2, "lexical_score": 1.0,
        }]

    def search_grounded_entity_vector_candidates(
        self, namespace_id, model_name, query_vector, limit,
    ):
        self.snapshot_calls.append(("entity-vectors", namespace_id, model_name))
        return [{
            "canonical_key": "entity-current", "name": "Alpha",
            "primary_type": "Thing", "aliases": [], "metadata": {},
            "source_count": 2, "vector_score": 0.9,
        }]

    def search_knowledge_forge_relations(
        self, namespace_id, relation_run_id, query, *, chunk_ids,
        canonical_keys, limit,
    ):
        self.snapshot_calls.append((
            "relations", namespace_id, relation_run_id, tuple(chunk_ids),
            tuple(canonical_keys),
        ))
        return [{
            "id": "fact-current", "source_name": "Alpha",
            "source_canonical_key": "entity-current",
            "relation_type": "connected_to", "target_name": "Beta",
            "target_canonical_key": "entity-beta",
            "evidence_text": "Alpha connects to Beta.", "chunk_id": 7,
            "metadata": {"confidence": 0.91}, "evidence_count": 1,
            "evidence_records": [{
                "verification_attempt_id": "supported-verification",
            }],
            "lexical_score": 1.0, "original_filename": "current.pdf",
            "storage_path": "/files/current.pdf",
        }]

    def search_graph_text(self, *_args, **_kwargs):
        raise AssertionError("Knowledge Forge search must not read legacy graph text")

    def get_entities_by_ids(self, *_args, **_kwargs):
        raise AssertionError("Knowledge Forge search must not read legacy entities")

    def get_relations_by_ids(self, *_args, **_kwargs):
        raise AssertionError("Knowledge Forge search must not read legacy relations")


class SearchVectors:
    def search(self, project, query, top_k):
        return [{
            "chunk_id": 7, "object_key": "chunk-current",
            "content": "Alpha connects to Beta.", "score": 0.8,
        }]


class SearchOllama:
    def embed(self, texts, model):
        assert model == "embed"
        if texts == ["Alpha"]:
            return [[0.1, 0.2]]
        assert any("Alpha" in text and "Beta" in text for text in texts)
        return [[0.1, 0.2] for _ in texts]


def test_service_search_reads_only_current_knowledge_forge_snapshot():
    service = object.__new__(KnowledgeForgeService)
    service.db = SearchDatabase()
    service.vectors = SearchVectors()
    service.ollama = SearchOllama()

    result = service.search(4, "Alpha")

    assert result["strategy"] == KNOWLEDGE_FORGE_STRATEGY
    assert result["graph_generation"] == 3
    assert result["entity_namespace_id"] == "namespace-current"
    assert result["relation_run_id"] == "relation-run-current"
    assert result["entities"][0]["canonical_key"] == "entity-current"
    assert result["relations"][0]["id"] == "fact-current"
    assert result["relations"][0]["verification_status"] == "SUPPORTED"
    assert result["relations"][0]["evidence_count"] == 1
    assert ("entities", "namespace-current", "Alpha") in service.db.snapshot_calls
    assert (
        "relations", "namespace-current", "relation-run-current", (7,),
        ("entity-current",),
    ) in service.db.snapshot_calls
    assert result["relations"][0]["network_scope"] == "seed_boundary"
    assert result["relations"][0]["graph_hop"] == 1
    assert result["graph_retrieval"]["maximum_hops"] == 2


def test_service_search_fails_closed_when_knowledge_forge_graph_is_stale():
    database = SearchDatabase()
    database.get_project_strategy_graph = lambda *_args: {
        "status": "stale", "generation": 4, "ready_generation": 3,
        "error_message": "source changed",
    }
    service = object.__new__(KnowledgeForgeService)
    service.db = database
    service.vectors = SearchVectors()
    service.ollama = SearchOllama()

    with pytest.raises(GraphNotReadyError, match="source changed"):
        service.search(4, "Alpha")


def test_service_search_fails_closed_for_cross_project_relation_run():
    database = SearchDatabase()
    database.get_grounded_relation_run = lambda *_args: {
        "id": "relation-run-current", "project_id": 99, "status": "ready",
    }
    service = object.__new__(KnowledgeForgeService)
    service.db = database
    service.vectors = SearchVectors()
    service.ollama = SearchOllama()

    with pytest.raises(GraphNotReadyError, match="关系快照不完整"):
        service.search(4, "Alpha")


def test_service_search_fails_closed_for_obsolete_relation_treatment():
    database = SearchDatabase()
    database.get_grounded_relation_run = lambda *_args: {
        "id": "relation-run-current",
        "project_id": 4,
        "status": "ready",
        "fingerprint_components": {
            "materialization_root": {
                "code_tree": LOADED_RELATION_CODE_DIGEST,
                "treatment": "obsolete-treatment",
            },
        },
    }
    service = object.__new__(KnowledgeForgeService)
    service.db = database
    service.vectors = SearchVectors()
    service.ollama = SearchOllama()

    with pytest.raises(GraphNotReadyError, match="抽取策略已升级"):
        service.search(4, "Alpha")


class QueryRecorder:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        return Rows(one={
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "relations": 0,
            "relation_candidates": 0,
        })


def database_with_connection(connection):
    database = Database()

    @contextmanager
    def connect():
        yield connection

    database.connection = connect
    return database


def test_default_queries_share_allowlist_while_audit_queries_are_explicit():
    connection = QueryRecorder()
    database = database_with_connection(connection)

    database.list_project_relations(1)
    database.list_document_relations(2)
    database.search_graph_text(1, "x")
    database.get_relations_by_ids(1, [1, 2])
    database.statistics(1)
    database.list_all_project_relations(1)
    database.list_all_document_relations(2)
    database.list_relation_candidates(1)

    default_statements = [call[0] for call in connection.calls[:6]]
    assert all("verification_status=ANY" in statement for statement in (
        default_statements[0],
        default_statements[1],
        default_statements[3],
        default_statements[4],
        default_statements[5],
    ))
    assert "relation_candidates" in default_statements[5]
    audit_statements = [call[0] for call in connection.calls[-3:]]
    assert all("verification_status=ANY" not in statement for statement in audit_statements)


def test_json_audit_values_preserve_non_none_falsy_values():
    assert Database._json([]) == "[]"
    assert Database._json(False) == "false"
    assert Database._json(0) == "0"
    assert Database._json("") == '""'
    assert Database._json(None) == "{}"


class MigrationConnection:
    def __init__(self):
        self.assertion_status = "rules_verified"
        self.candidate_status = "rules_verified"
        self.vector_active = True

    def execute(self, statement, params=None):
        text = str(statement)
        if "UPDATE relation_assertions" in text:
            changed = self.assertion_status == "rules_verified"
            if changed:
                self.assertion_status = "legacy_unverified"
            return Rows(rowcount=int(changed))
        if "UPDATE relation_candidates" in text:
            changed = self.candidate_status == "rules_verified"
            if changed:
                self.candidate_status = "legacy_unverified"
            return Rows(rowcount=int(changed))
        if "UPDATE vector_embeddings v" in text:
            changed = self.vector_active and self.assertion_status not in PUBLISHABLE_RELATION_STATUSES
            if changed:
                self.vector_active = False
            return Rows(rowcount=int(changed))
        raise AssertionError(text)


def test_legacy_policy_migration_is_conservative_and_idempotent():
    connection = MigrationConnection()
    first = Database._migrate_relation_verification_policy(connection)
    second = Database._migrate_relation_verification_policy(connection)

    assert first == {
        "assertions_quarantined": 1,
        "candidates_quarantined": 1,
        "vectors_retired": 1,
    }
    assert second == {
        "assertions_quarantined": 0,
        "candidates_quarantined": 0,
        "vectors_retired": 0,
    }
    assert connection.assertion_status == "legacy_unverified"
    assert connection.vector_active is False


class DowngradeConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        if "INSERT INTO relation_assertions" in str(statement):
            return Rows(one={"id": 9, "project_id": 1, "verification_status": "uncertain"})
        return Rows()


def test_assertion_downgrade_retires_existing_relation_vector():
    connection = DowngradeConnection()
    database = database_with_connection(connection)
    database.add_relation_assertion({
        "project_id": 1, "document_id": 2, "chunk_id": 3,
        "source_entity_id": 4, "target_entity_id": 5, "relation_key": "r1",
        "relation_type": "uses", "verification_status": "uncertain",
    })
    assert any(
        "UPDATE vector_embeddings SET is_deleted=TRUE" in statement
        and params == (1, "9")
        for statement, params in connection.calls
    )


class TransactionConnection:
    def __init__(self):
        self.pending = []

    def execute(self, statement, params=None):
        text = str(statement)
        if "INSERT INTO relation_candidates" in text:
            self.pending.append("candidate")
            return Rows(one={"candidate_id": params[0]})
        if "INSERT INTO relation_verification_events" in text:
            raise RuntimeError("injected event failure")
        raise AssertionError(text)


def test_chunk_relation_audit_chain_rolls_back_on_later_write_failure():
    connection = TransactionConnection()
    database = Database()
    committed = []

    @contextmanager
    def connect():
        snapshot = list(connection.pending)
        try:
            yield connection
        except Exception:
            connection.pending[:] = snapshot
            raise
        else:
            committed[:] = connection.pending

    database.connection = connect
    candidate = {
        "candidate_id": "c1", "project_id": 1, "document_id": 2, "chunk_id": 3,
        "candidate_ordinal": 0, "source_mention_id": "e1", "source_name": "Alpha",
        "target_mention_id": "e2", "target_name": "Beta", "proposed_relation_type": "uses",
    }
    event = {
        "candidate_id": "c1", "project_id": 1, "document_id": 2, "chunk_id": 3,
        "attempt": 1, "model_name": "m", "prompt_version": "p", "schema_version": 1,
    }
    with pytest.raises(RuntimeError, match="injected event failure"):
        database.persist_relation_audit_chain([candidate], [event], [])
    assert connection.pending == []
    assert committed == []


class GraphRebuildDatabase:
    def __init__(self):
        self.updated = False
        self.states = []
        self.namespace = str(uuid.UUID(int=1))
        self.lease = None

    def get_project(self, project_id):
        return {
            "id": project_id, "project_key": "p1",
            "graph_namespace_id": self.namespace, "graph_generation": 0,
            "active_schema_version": 1,
        }

    def list_documents(self, project_id):
        return [{"id": 2, "status": "indexed", "original_filename": "a.txt"}]

    def list_document_chunks(self, document_id):
        return []

    def update_project(self, project_id, **values):
        self.updated = values.get("graph_status") == "ready"

    def begin_graph_mutation(self, job_id):
        return {"id": job_id, "project_id": 1, "status": "running"}

    def count_publishable_relations(self, project_id, document_ids=None):
        return 1

    def list_pending_indexing_failures(self, project_id, object_type="relation"):
        return []

    def list_blocking_indexing_failures(self, project_id, graph_generation):
        return []

    def acquire_graph_lease(self, project_id, namespace_id, generation, owner, **kwargs):
        self.lease = {"id": "lease-id", "lease_token": uuid.uuid4().hex}
        return dict(self.lease)

    def heartbeat_graph_lease(self, lease_id, lease_token, **kwargs):
        return bool(self.lease and lease_token == self.lease["lease_token"])

    def release_graph_lease(self, lease_id, lease_token):
        return True

    def persist_graph_manifest(self, project_id, namespace_id, generation, role, lease_token, manifest):
        return manifest

    def complete_graph_rebuild(self, project_id, generation, **kwargs):
        self.updated = True
        return self.get_project(project_id)

    def update_job(self, job_id, **kwargs):
        return {"id": job_id, **kwargs}

    def set_graph_state(self, project_id, status, **values):
        self.states.append((status, values))
        self.updated = status == "ready" and values.get("current_policy") is True
        return {"id": project_id, "graph_status": status,
                "graph_policy_version": GRAPH_POLICY_VERSION if values.get("current_policy") else ""}


class GraphRebuildStore:
    def __init__(self):
        self.edges = {"legacy-edge"}
        self.calls = []

    def ensure_schema(self):
        self.calls.append("schema")

    @staticmethod
    def _manifest(project, mutation):
        return build_graph_manifest(
            [], namespace_id=mutation.namespace_id, project_id=project["id"],
            graph_generation=mutation.generation, policy_hash=mutation.policy_hash,
        )

    def expected_manifest(self, project, documents, *, mutation):
        return self._manifest(project, mutation)

    def writer_manifest(self, project, *, mutation):
        return self._manifest(project, mutation)

    def actual_manifest(self, project, *, mutation):
        return self._manifest(project, mutation)

    def delete_project(self, project, *, mutation):
        self.calls.append("purge")
        self.edges.clear()

    def write_document_snapshot(self, project, document, chunks, *, mutation):
        self.calls.append("document")

    def sync_document_assertions(self, project, document, *, mutation):
        self.calls.append("relations")
        self.edges.add("published-edge")
        return {"entities": 2, "relations": 1}

    def prune_project_entities(self, project, *, mutation):
        self.calls.append("prune")

    def statistics(self, project):
        return {"relations": len(self.edges)}


def test_explicit_project_graph_rebuild_purges_stale_edges_before_sync():
    service = object.__new__(KnowledgeForgeService)
    service.db = GraphRebuildDatabase()
    service.graph = GraphRebuildStore()
    service._job = lambda *args, **kwargs: None

    service._sync_project_graph("job-1", 1)

    assert service.graph.calls == ["schema", "purge", "document", "relations", "prune"]
    assert service.graph.edges == {"published-edge"}
    assert service.db.updated is True


@pytest.mark.integration
def test_disposable_postgres_schema_and_policy_migration(active_document_revision):
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    managed_schema = os.getenv("KNOWLEDGE_FORGE_TEST_SCHEMA", "").strip()
    schema_name = managed_schema or f"kf_p01_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        if not managed_schema:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        try:
            conn.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema_name)))
            if not managed_schema:
                conn.execute(SCHEMA_SQL)
                conn.execute(SCHEMA_SQL)
            project_id = conn.execute(
                "INSERT INTO projects(project_key,name) VALUES('p','P') RETURNING id"
            ).fetchone()[0]
            document_id = conn.execute(
                """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
                storage_path,local_path,file_size_bytes) VALUES(%s,'h','a.txt','a.txt','/a','/a',1)
                RETURNING id""",
                (project_id,),
            ).fetchone()[0]
            revision_id = None
            if managed_schema:
                revision_id = active_document_revision(conn, project_id, document_id)
            chunk_id = conn.execute(
                """INSERT INTO chunks(project_id,document_id,chunk_key,content,content_hash)
                VALUES(%s,%s,'c','Alpha uses Beta','ch') RETURNING id""",
                (project_id, document_id),
            ).fetchone()[0]
            entity_ids = [
                conn.execute(
                    """INSERT INTO canonical_entities(project_id,entity_key,name,normalized_name,entity_type)
                    VALUES(%s,%s,%s,%s,'GenericEntity') RETURNING id""",
                    (project_id, key, name, name.casefold()),
                ).fetchone()[0]
                for key, name in (("e1", "Alpha"), ("e2", "Beta"))
            ]
            assertion_id = conn.execute(
                """INSERT INTO relation_assertions(project_id,document_id,chunk_id,source_entity_id,
                target_entity_id,relation_key,relation_type,confidence,verification_status,metadata)
                VALUES(%s,%s,%s,%s,%s,'r','uses',0,'rules_verified','{}') RETURNING id""",
                (project_id, document_id, chunk_id, *entity_ids),
            ).fetchone()[0]
            conn.execute(
                """INSERT INTO vector_embeddings(project_id,document_id,chunk_id,object_type,object_key,
                content,content_hash,model_name,embedding,embedding_vector,dimension)
                VALUES(%s,%s,%s,'relation',%s,'Alpha uses Beta','vh','embed',ARRAY[0,0],
                '[0,0]'::vector,2)""",
                (project_id, document_id, chunk_id, str(assertion_id)),
            )
            conn.execute(
                """INSERT INTO relation_candidates(candidate_id,project_id,document_id,chunk_id,
                candidate_ordinal,source_mention_id,source_name,target_mention_id,target_name,
                proposed_relation_type,status,metadata) VALUES('c',%s,%s,%s,0,'e1','Alpha',
                'e2','Beta','uses','rules_verified','[]')""",
                (project_id, document_id, chunk_id),
            )
            assert conn.execute(
                "SELECT confidence FROM relation_assertions WHERE id=%s", (assertion_id,)
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT jsonb_typeof(metadata) FROM relation_candidates WHERE candidate_id='c'"
            ).fetchone()[0] == "array"
            assert conn.execute(
                "SELECT COUNT(*) FROM relation_assertions WHERE verification_status=ANY(%s)",
                (sorted(PUBLISHABLE_RELATION_STATUSES),),
            ).fetchone()[0] == 1
            Database._migrate_relation_verification_policy(conn)
            assert conn.execute(
                "SELECT verification_status FROM relation_assertions WHERE id=%s", (assertion_id,)
            ).fetchone()[0] == "legacy_unverified"
            assert conn.execute(
                "SELECT is_deleted FROM vector_embeddings WHERE object_key=%s", (str(assertion_id),)
            ).fetchone()[0] is True
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                if managed_schema:
                    conn.execute(
                        """INSERT INTO chunks(
                           project_id,document_id,revision_id,chunk_key,content,content_hash)
                           VALUES(%s,999999,%s,'bad','bad','bad')""",
                        (project_id, revision_id),
                    )
                else:
                    conn.execute(
                        """INSERT INTO chunks(project_id,document_id,chunk_key,content,content_hash)
                        VALUES(%s,999999,'bad','bad','bad')""",
                        (project_id,),
                    )
        finally:
            if not managed_schema:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))
