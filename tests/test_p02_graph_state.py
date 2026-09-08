from contextlib import contextmanager
from http import HTTPStatus
import os
import uuid

import pytest
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.graph_manifest import build_graph_manifest
from knowledge_forge.graph_policy import (
    GRAPH_FAILED,
    GRAPH_POLICY_HASH,
    GRAPH_POLICY_VERSION,
    GRAPH_READY,
    GRAPH_REBUILDING,
    GRAPH_STALE,
    GraphNotReadyError,
)
from knowledge_forge.relation_verification import (
    RELATION_VERIFICATION_POLICY_VERSION,
    is_publishable_relation,
)
from knowledge_forge.service import KnowledgeForgeService
from knowledge_forge import web


class Rows:
    def __init__(self, *, one=None, rows=None, rowcount=0):
        self._one = one
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._rows


def database_with_connection(connection):
    database = Database()

    @contextmanager
    def connect():
        yield connection

    database.connection = connect
    return database


@pytest.mark.parametrize("status", ["rules_verified", "llm_verified"])
def test_automatic_relation_publication_requires_current_policy(status):
    assert not is_publishable_relation(status)
    assert not is_publishable_relation(status, {"verification_policy_version": "old"})
    assert is_publishable_relation(
        status,
        {"verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION},
    )


def test_manual_relation_has_explicit_version_independent_publication_policy():
    assert is_publishable_relation("manual_verified")
    assert not is_publishable_relation("unknown_verified")


class ProductQueryRecorder:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), tuple(params or ())))
        return Rows(rows=[])


def test_product_query_fails_closed_even_before_migration_runs():
    connection = ProductQueryRecorder()
    database = database_with_connection(connection)

    database.list_project_relations(7)

    statement, params = connection.calls[0]
    assert "metadata->>'verification_policy_version'" in statement
    assert params == (
        7,
        "manual_verified",
        ["llm_verified", "rules_verified"],
        RELATION_VERIFICATION_POLICY_VERSION,
    )


class GraphMigrationConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), tuple(params or ())))
        return Rows(rowcount=2)


def test_old_nonempty_graphs_are_marked_stale_but_empty_projects_are_excluded():
    connection = GraphMigrationConnection()

    assert Database._migrate_graph_policy(connection) == 2
    statement, params = connection.calls[0]
    assert "graph_status<>%s" in statement
    assert "graph_policy_version<>%s" in statement
    assert "updated_at" not in statement
    assert params[0] == GRAPH_STALE
    assert params[-3:] == ("empty", GRAPH_POLICY_VERSION, GRAPH_POLICY_HASH)


def test_legacy_graph_view_routes_to_inline_workspace_without_hubble(monkeypatch):
    class NotReadyService:
        @staticmethod
        def _assert_project_strategy(project_id, expected_strategy=None):
            return expected_strategy

    monkeypatch.setattr(web, "SERVICE", NotReadyService())
    handler = object.__new__(web.AppHandler)
    handler.path = "/graph/view?project=7"
    response = {}
    handler._redirect = lambda target, project_id=None: response.update(
        target=target, project_id=project_id,
    )
    handler._project = lambda: 7

    handler.do_GET()

    assert response == {
        "target": "/projects/7/knowledge-forge#query",
        "project_id": 7,
    }


class RebuildSubmissionConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        text = str(statement)
        self.calls.append((text, tuple(params or ())))
        if "SELECT id,graph_generation FROM projects" in text:
            return Rows(one={"id": 7, "graph_generation": 3})
        if "SELECT id,status,target_generation FROM jobs" in text or "SELECT id FROM jobs" in text:
            return Rows(one=None)
        if "INSERT INTO jobs" in text:
            return Rows(one={
                "id": params[0], "project_id": 7, "status": "queued",
                "target_generation": params[2],
            })
        if "UPDATE projects" in text:
            return Rows(rowcount=1)
        raise AssertionError(text)


def test_graph_rebuild_submission_atomically_enters_rebuilding():
    connection = RebuildSubmissionConnection()
    job = database_with_connection(connection).create_graph_rebuild_job(7)

    assert job["project_id"] == 7
    assert job["target_generation"] == 3
    update = next(call for call in connection.calls if "UPDATE projects" in call[0])
    assert update[1] == (GRAPH_REBUILDING, 7)


class LifecycleDatabase:
    def __init__(self, *, expected_relations=2):
        self.project = {
            "id": 7,
            "project_key": "project-7",
            "graph_namespace_id": str(uuid.UUID(int=7)),
            "graph_generation": 0,
            "graph_ready_generation": 0,
            "active_schema_version": 1,
            "embedding_model": "embed",
            "graph_status": GRAPH_STALE,
            "graph_policy_version": "",
            "graph_error": "",
        }
        self.documents = [
            {"id": 11, "status": "indexed", "original_filename": "a.txt"},
            {"id": 12, "status": "indexed", "original_filename": "b.txt"},
        ]
        self.expected_relations = expected_relations
        self.jobs = {"job-1": {"id": "job-1", "project_id": 7, "status": "queued"}}
        self.state_history = []
        self.lease = None
        self.manifests = {}

    def get_project(self, project_id):
        return dict(self.project) if project_id == 7 else None

    def begin_graph_mutation(self, job_id):
        self.jobs[job_id]["status"] = "running"
        self.set_graph_state(7, GRAPH_REBUILDING)
        return self.jobs[job_id]

    def update_job(self, job_id, *, status=None, progress=None, log="", error="", metadata=None):
        job = self.jobs[job_id]
        if status is not None:
            job["status"] = status
        if error:
            job["error_message"] = error
        return job

    def set_graph_state(self, project_id, status, *, error="", current_policy=False):
        self.project.update({
            "graph_status": status,
            "graph_policy_version": GRAPH_POLICY_VERSION if current_policy else "",
            "graph_error": error,
        })
        self.state_history.append((status, current_policy, error))
        return dict(self.project)

    def list_documents(self, project_id):
        return list(self.documents)

    def list_document_chunks(self, document_id):
        return [{"id": document_id * 10, "content": "content"}]

    def count_publishable_relations(self, project_id, document_ids=None):
        return self.expected_relations

    def list_pending_indexing_failures(self, project_id, object_type="relation"):
        return []

    def list_blocking_indexing_failures(self, project_id, graph_generation):
        return []

    def acquire_graph_lease(self, project_id, namespace_id, generation, owner, **kwargs):
        self.lease = {"id": "lease-id", "lease_token": uuid.uuid4().hex}
        return dict(self.lease)

    def assert_graph_lease(self, project_id, namespace_id, generation, lease_token):
        return bool(self.lease and lease_token == self.lease["lease_token"])

    def heartbeat_graph_lease(self, lease_id, lease_token, **kwargs):
        return self.assert_graph_lease(7, self.project["graph_namespace_id"], 0, lease_token)

    def release_graph_lease(self, lease_id, lease_token):
        return True

    def persist_graph_manifest(self, project_id, namespace_id, generation, role, lease_token, manifest):
        self.manifests[role] = manifest
        return manifest

    def complete_graph_rebuild(self, project_id, generation, **kwargs):
        if generation != self.project["graph_generation"]:
            return None
        self.project.update({
            "graph_status": GRAPH_READY,
            "graph_ready_generation": generation,
            "graph_policy_version": GRAPH_POLICY_VERSION,
            "graph_policy_hash": kwargs["policy_hash"],
            "graph_error": "",
        })
        return dict(self.project)


class LifecycleGraph:
    def __init__(self, fail_at=""):
        self.fail_at = fail_at
        self.edges = set()
        self.document_writes = 0

    def ensure_schema(self):
        return None

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
        self.edges.clear()
        self.document_writes = 0
        if self.fail_at == "purge":
            raise RuntimeError("purge failed")

    def write_document_snapshot(self, project, document, chunks, *, mutation):
        self.document_writes += 1
        if self.fail_at == "after_first_document" and self.document_writes == 2:
            raise RuntimeError("second document failed")

    def sync_document_assertions(self, project, document, *, mutation):
        self.edges.add(int(document["id"]))
        return {"entities": 2, "relations": 1}

    def prune_project_entities(self, project, *, mutation):
        return 0

    def statistics(self, project):
        count = len(self.edges)
        if self.fail_at == "final_check":
            count += 1
        return {"relations": count}


def lifecycle_service(fail_at="", *, expected_relations=2):
    service = object.__new__(KnowledgeForgeService)
    service.db = LifecycleDatabase(expected_relations=expected_relations)
    service.graph = LifecycleGraph(fail_at)
    service.vectors = object()
    return service


@pytest.mark.parametrize("fail_at", ["purge", "after_first_document", "final_check"])
def test_rebuild_failure_never_exposes_ready_and_retry_converges(fail_at):
    service = lifecycle_service(fail_at)

    service._sync_project_graph("job-1", 7)

    assert service.db.project["graph_status"] == GRAPH_FAILED
    with pytest.raises(GraphNotReadyError):
        service.hubble_project_url(7)

    service.graph.fail_at = ""
    service.db.jobs["job-1"]["status"] = "queued"
    service._sync_project_graph("job-1", 7)

    assert service.db.project["graph_status"] == GRAPH_READY
    assert service.db.project["graph_policy_version"] == GRAPH_POLICY_VERSION
    assert service.graph.edges == {11, 12}


def test_postgres_hugegraph_relation_count_mismatch_fails_rebuild():
    service = lifecycle_service(expected_relations=3)

    service._sync_project_graph("job-1", 7)

    assert service.db.project["graph_status"] == GRAPH_FAILED
    assert "一致性检查失败" in service.db.project["graph_error"]


class SearchEmbeddingRecorder:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), tuple(params or ())))
        if "set_config" in str(statement):
            return Rows()
        return Rows(rows=[])


def test_relation_vector_top_k_sql_filters_policy_and_orphans_before_limit():
    connection = SearchEmbeddingRecorder()
    database = database_with_connection(connection)

    database.search_embeddings(7, "relation", "embed", [1.0, 0.0], "uses", 5)

    statement, params = connection.calls[-1]
    first_limit = statement.index("LIMIT")
    assert "EXISTS (SELECT 1 FROM relation_assertions" in statement[:first_limit]
    assert "r.id::text=v0.object_key" in statement[:first_limit]
    assert statement.count("verification_policy_version") == 2
    assert params.count(RELATION_VERIFICATION_POLICY_VERSION) == 2


class RetryDatabase(LifecycleDatabase):
    def __init__(self):
        super().__init__(expected_relations=0)
        self.failures = [{
            "project_id": 7,
            "document_id": 11,
            "chunk_id": 110,
            "object_key": "91",
            "model_name": "embed",
            "payload": {"relation_key": "r91", "source_name": "Alpha", "target_name": "Beta"},
        }]
        self.completed = []

    def list_pending_indexing_failures(self, project_id, object_type="relation"):
        return list(self.failures)

    def get_relations_by_ids(self, project_id, ids):
        return [{
            "id": 91,
            "relation_type": "uses",
            "description": "Alpha uses Beta",
            "source_name": "Alpha",
            "target_name": "Beta",
        }] if ids == [91] else []

    def complete_indexing_failure(self, project_id, object_type, object_key, model_name):
        self.completed.append((project_id, object_type, object_key, model_name))
        self.failures.clear()

    def record_indexing_failure(self, values):
        self.failures = [values]


class RetryVectors:
    def __init__(self):
        self.calls = []

    def upsert_relation_embedding(self, *args):
        self.calls.append(args)


def test_relation_indexing_outbox_is_retried_without_changing_assertion_state():
    service = object.__new__(KnowledgeForgeService)
    service.db = RetryDatabase()
    service.vectors = RetryVectors()

    assert service._retry_relation_indexing_failures(service.db.project) == 1
    assert len(service.vectors.calls) == 1
    assert service.db.completed == [(7, "relation", "91", "embed")]


@pytest.mark.integration
def test_p02_real_postgres_policy_migration_and_audit_transactions(active_document_revision):
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    managed_schema = os.getenv("KNOWLEDGE_FORGE_TEST_SCHEMA", "").strip()
    schema_name = managed_schema or f"kf_p02_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True, row_factory=dict_row) as admin:
        if not managed_schema:
            admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        try:
            admin.execute(
                sql.SQL("SET search_path TO {},public").format(sql.Identifier(schema_name))
            )
            if not managed_schema:
                admin.execute(SCHEMA_SQL)
                admin.execute(SCHEMA_SQL)
            project_id = admin.execute(
                "INSERT INTO projects(project_key,name) VALUES('p02','P02') RETURNING id"
            ).fetchone()["id"]
            document_id = admin.execute(
                """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
                   storage_path,local_path,file_size_bytes)
                   VALUES(%s,'hash','a.txt','a.txt','/a','/a',1) RETURNING id""",
                (project_id,),
            ).fetchone()["id"]
            if managed_schema:
                active_document_revision(admin, project_id, document_id)
            chunk_id = admin.execute(
                """INSERT INTO chunks(project_id,document_id,chunk_key,content,content_hash)
                   VALUES(%s,%s,'chunk','Alpha uses Beta','chunk-hash') RETURNING id""",
                (project_id, document_id),
            ).fetchone()["id"]
            entity_ids = [
                admin.execute(
                    """INSERT INTO canonical_entities(project_id,entity_key,name,normalized_name,entity_type)
                       VALUES(%s,%s,%s,%s,'GenericEntity') RETURNING id""",
                    (project_id, key, name, name.casefold()),
                ).fetchone()["id"]
                for key, name in (("alpha", "Alpha"), ("beta", "Beta"))
            ]

            assertion_ids = {}
            for index, (status, metadata) in enumerate((
                ("rules_verified", {}),
                ("llm_verified", {"verification_policy_version": "old"}),
                ("manual_verified", {}),
                ("llm_verified", {
                    "verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION,
                }),
            ), start=1):
                assertion_ids[index] = admin.execute(
                    """INSERT INTO relation_assertions(project_id,document_id,chunk_id,
                       source_entity_id,target_entity_id,relation_key,relation_type,
                       verification_status,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,'uses',%s,%s::jsonb) RETURNING id""",
                    (
                        project_id, document_id, chunk_id, *entity_ids,
                        f"migration-{index}", status, Database._json(metadata),
                    ),
                ).fetchone()["id"]
                admin.execute(
                    """INSERT INTO vector_embeddings(project_id,document_id,chunk_id,object_type,
                       object_key,content,content_hash,model_name,embedding,embedding_vector,dimension)
                       VALUES(%s,%s,%s,'relation',%s,'Alpha uses Beta',%s,'embed',
                       ARRAY[1,0],'[1,0]'::vector,2)""",
                    (project_id, document_id, chunk_id, str(assertion_ids[index]), f"v-{index}"),
                )
            admin.execute(
                """INSERT INTO vector_embeddings(project_id,document_id,chunk_id,object_type,
                   object_key,content,content_hash,model_name,embedding,embedding_vector,dimension)
                   VALUES(%s,%s,%s,'relation','999999','orphan','orphan-v','embed',
                   ARRAY[1,0],'[1,0]'::vector,2)""",
                (project_id, document_id, chunk_id),
            )
            for candidate_id, status, metadata in (
                ("legacy-rule", "rules_verified", {}),
                ("legacy-llm", "llm_verified", {"verification_policy_version": "old"}),
            ):
                admin.execute(
                    """INSERT INTO relation_candidates(candidate_id,project_id,document_id,chunk_id,
                       candidate_ordinal,source_mention_id,source_name,target_mention_id,target_name,
                       proposed_relation_type,status,metadata)
                       VALUES(%s,%s,%s,%s,0,'e1','Alpha','e2','Beta','uses',%s,%s::jsonb)""",
                    (candidate_id, project_id, document_id, chunk_id, status, Database._json(metadata)),
                )

            def transaction_connection():
                @contextmanager
                def connect():
                    with psycopg.connect(dsn, row_factory=dict_row) as connection:
                        connection.execute(
                            sql.SQL("SET search_path TO {},public").format(
                                sql.Identifier(schema_name)
                            )
                        )
                        yield connection

                return connect

            database = Database()
            database.connection = transaction_connection()
            pre_migration_hits = database.search_embeddings(
                project_id, "relation", "embed", [1.0, 0.0], "Alpha", 1,
            )
            assert {row["object_key"] for row in pre_migration_hits} == {
                str(assertion_ids[3]), str(assertion_ids[4]),
            }

            first = Database._migrate_relation_verification_policy(admin)
            second = Database._migrate_relation_verification_policy(admin)
            assert first == {
                "assertions_quarantined": 2,
                "candidates_quarantined": 2,
                "vectors_retired": 3,
            }
            assert second == {
                "assertions_quarantined": 0,
                "candidates_quarantined": 0,
                "vectors_retired": 0,
            }
            statuses = admin.execute(
                "SELECT relation_key,verification_status,metadata FROM relation_assertions"
            ).fetchall()
            status_map = {row["relation_key"]: row for row in statuses}
            assert status_map["migration-1"]["verification_status"] == "legacy_unverified"
            assert status_map["migration-2"]["verification_status"] == "legacy_unverified"
            assert status_map["migration-3"]["verification_status"] == "manual_verified"
            assert status_map["migration-4"]["verification_status"] == "llm_verified"
            assert status_map["migration-1"]["metadata"]["legacy_verification_status"] == "rules_verified"
            active_vector_keys = {
                row["object_key"] for row in admin.execute(
                    "SELECT object_key FROM vector_embeddings WHERE is_deleted=FALSE"
                ).fetchall()
            }
            assert active_vector_keys == {
                str(assertion_ids[3]), str(assertion_ids[4]),
            }
            failed_run = "run-rollback"
            failed_candidate = {
                "candidate_id": "audit-rollback", "project_id": project_id,
                "document_id": document_id, "chunk_id": chunk_id,
                "candidate_ordinal": 0, "source_mention_id": "e1", "source_name": "Alpha",
                "target_mention_id": "e2", "target_name": "Beta",
                "proposed_relation_type": "uses", "status": "llm_verified",
                "ingestion_run_id": failed_run,
            }
            failed_event = {
                "candidate_id": "audit-rollback", "project_id": project_id,
                "document_id": document_id, "chunk_id": chunk_id, "attempt": 1,
                "model_name": "llm", "prompt_version": "p0", "schema_version": 1,
                "ingestion_run_id": failed_run,
            }
            failed_assertion = {
                "project_id": project_id, "document_id": document_id, "chunk_id": chunk_id,
                "source_entity_id": 999999999, "target_entity_id": entity_ids[1],
                "relation_key": "audit-rollback", "relation_type": "uses",
                "verification_status": "llm_verified", "ingestion_run_id": failed_run,
                "metadata": {
                    "verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION,
                },
            }
            with pytest.raises(psycopg.errors.ForeignKeyViolation):
                database.persist_relation_audit_chain(
                    [failed_candidate], [failed_event], [failed_assertion],
                )
            assert admin.execute(
                "SELECT COUNT(*) AS count FROM relation_candidates WHERE candidate_id='audit-rollback'"
            ).fetchone()["count"] == 0
            assert admin.execute(
                "SELECT COUNT(*) AS count FROM relation_verification_events WHERE ingestion_run_id=%s",
                (failed_run,),
            ).fetchone()["count"] == 0
            assert admin.execute(
                "SELECT COUNT(*) AS count FROM relation_assertions WHERE relation_key='audit-rollback'"
            ).fetchone()["count"] == 0

            success_run = "run-success"
            success_candidate = {
                **failed_candidate,
                "candidate_id": "audit-success",
                "ingestion_run_id": success_run,
            }
            success_event = {
                **failed_event,
                "candidate_id": "audit-success",
                "ingestion_run_id": success_run,
            }
            success_assertion = {
                **failed_assertion,
                "source_entity_id": entity_ids[0],
                "relation_key": "audit-success",
                "ingestion_run_id": success_run,
            }
            database.persist_relation_audit_chain(
                [success_candidate], [success_event], [success_assertion],
            )
            recorded_runs = {
                admin.execute(
                    "SELECT ingestion_run_id FROM relation_candidates WHERE candidate_id='audit-success'"
                ).fetchone()["ingestion_run_id"],
                admin.execute(
                    "SELECT ingestion_run_id FROM relation_verification_events WHERE candidate_id='audit-success'"
                ).fetchone()["ingestion_run_id"],
                admin.execute(
                    "SELECT ingestion_run_id FROM relation_assertions WHERE relation_key='audit-success'"
                ).fetchone()["ingestion_run_id"],
            }
            assert recorded_runs == {success_run}
        finally:
            if not managed_schema:
                admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name)))
