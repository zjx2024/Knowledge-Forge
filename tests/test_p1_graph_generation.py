import threading
from contextlib import contextmanager
import uuid

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.graph_manifest import build_graph_manifest
from knowledge_forge.graph_policy import (
    GRAPH_POLICY_HASH,
    GRAPH_READY,
    GRAPH_STALE,
    is_current_graph,
)
from knowledge_forge.service import KnowledgeForgeService


def test_generation_schema_and_active_rebuild_uniqueness_are_additive():
    assert "graph_generation BIGINT NOT NULL DEFAULT 0" in SCHEMA_SQL
    assert "graph_ready_generation BIGINT NOT NULL DEFAULT 0" in SCHEMA_SQL
    assert "target_generation BIGINT" in SCHEMA_SQL
    assert "uq_jobs_active_graph_sync" in SCHEMA_SQL
    assert "next_attempt_at TIMESTAMPTZ" in SCHEMA_SQL


def test_graph_readiness_requires_matching_generation_and_policy_hash():
    base = {
        "graph_status": GRAPH_READY,
        "graph_policy_version": "relation-graph-p0-v1",
        "graph_policy_hash": GRAPH_POLICY_HASH,
        "graph_generation": 3,
        "graph_ready_generation": 3,
    }
    assert is_current_graph(base)
    assert not is_current_graph({**base, "graph_ready_generation": 2})
    assert not is_current_graph({**base, "graph_policy_hash": ""})


class GenerationDatabase:
    def __init__(self, generation=4, pending=0):
        self.project = {
            "id": 7,
            "project_key": "p7",
            "graph_namespace_id": str(uuid.UUID(int=7)),
            "graph_generation": generation,
            "graph_status": GRAPH_STALE,
            "active_schema_version": 1,
        }
        self.pending = pending
        self.completed = []
        self.lease = None
        self.manifests = {}

    def get_project(self, project_id):
        return dict(self.project)

    def list_documents(self, project_id):
        return [
            {"id": 11, "status": "indexed", "original_filename": "a.txt", "storage_path": "a"},
            {"id": 12, "status": "indexed", "original_filename": "b.txt", "storage_path": "b"},
        ]

    def list_document_chunks(self, document_id):
        return [{"id": document_id * 10, "chunk_key": str(document_id)}]

    def count_publishable_relations(self, project_id, document_ids=None):
        return 2

    def list_pending_indexing_failures(self, project_id, object_type="relation", **kwargs):
        return []

    def list_blocking_indexing_failures(self, project_id, graph_generation):
        return []

    def acquire_graph_lease(self, project_id, namespace_id, generation, owner, **kwargs):
        self.lease = {"id": "lease-id", "lease_token": "lease-token"}
        return dict(self.lease)

    def assert_graph_lease(self, project_id, namespace_id, generation, lease_token):
        return bool(self.lease and lease_token == self.lease["lease_token"])

    def heartbeat_graph_lease(self, lease_id, lease_token, **kwargs):
        return self.assert_graph_lease(7, self.project["graph_namespace_id"], 4, lease_token)

    def release_graph_lease(self, lease_id, lease_token):
        return True

    def persist_graph_manifest(self, project_id, namespace_id, generation, role, lease_token, manifest):
        self.manifests[role] = manifest
        return manifest

    def complete_graph_rebuild(self, project_id, target_generation, **kwargs):
        if self.project["graph_generation"] != target_generation:
            self.project["graph_status"] = GRAPH_STALE
            return None
        self.project.update({
            "graph_status": GRAPH_READY,
            "graph_ready_generation": target_generation,
            "graph_policy_version": "relation-graph-p0-v1",
            "graph_policy_hash": kwargs["policy_hash"],
        })
        self.completed.append(target_generation)
        return dict(self.project)


class GenerationGraph:
    def __init__(self):
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
        self.calls.append(("purge", project["id"]))

    def write_document_snapshot(self, project, document, chunks, *, mutation):
        self.calls.append(("write", document["id"]))
        return {"documents": 1, "chunks": len(chunks)}

    def sync_document_assertions(self, project, document, *, mutation):
        self.calls.append(("relations", document["id"]))
        return {"entities": 1, "relations": 1}

    def prune_project_entities(self, project, *, mutation):
        self.calls.append(("prune", project["id"]))

    def statistics(self, project):
        self.calls.append(("count", project["id"]))
        return {"relations": 2}


def generation_service(generation=4):
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = GenerationDatabase(generation)
    service.graph = GenerationGraph()
    service.vectors = object()
    service._graph_locks_guard = threading.Lock()
    service._graph_locks = {}
    return service


def test_full_rebuild_uses_one_schema_one_project_purge_and_no_document_purge():
    service = generation_service()
    result = service._rebuild_project_graph(7, target_generation=4)
    assert result["relations"] == 2
    assert service.graph.calls.count("schema") == 1
    assert service.graph.calls.count(("purge", 7)) == 1
    assert [call for call in service.graph.calls if call[0] == "write"] == [
        ("write", 11), ("write", 12),
    ]
    assert service.db.completed == [4]


def test_generation_change_during_rebuild_never_becomes_ready():
    service = generation_service()
    original = service.graph.statistics

    def mutate(project):
        service.db.project["graph_generation"] = 5
        return original(project)

    service.graph.statistics = mutate
    try:
        service._rebuild_project_graph(7, target_generation=4)
    except RuntimeError as exc:
        assert "过期" in str(exc)
    else:
        raise AssertionError("stale rebuild unexpectedly succeeded")
    assert service.db.project["graph_status"] == GRAPH_STALE
    assert service.db.completed == []


def test_ten_queued_ingestions_coalesce_to_one_quiescent_rebuild():
    class PendingDatabase:
        def has_pending_ingestions(self, project_id, *, exclude_job_id=""):
            return int(exclude_job_id.split("-")[-1]) < 10

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = PendingDatabase()
    rebuilds = []
    service._rebuild_project_graph = lambda project_id, target_generation=None: (
        rebuilds.append((project_id, target_generation)) or {"relations": 0}
    )

    reports = [
        service._rebuild_at_quiescence(7, f"job-{index}", index)
        for index in range(1, 11)
    ]

    assert reports[:9] == [None] * 9
    assert reports[9] == {"relations": 0}
    assert rebuilds == [(7, 10)]


class CoalescingConnection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        text = str(statement)
        params = tuple(params or ())
        self.calls.append((text, params))
        if "SELECT id,graph_generation FROM projects" in text:
            return Row(one={"id": 7, "graph_generation": 8})
        if "SELECT id,status,target_generation FROM jobs" in text:
            return Row(one={"id": "existing", "status": "queued", "target_generation": 7})
        if "UPDATE jobs SET target_generation" in text:
            return Row(one={"id": "existing", "status": "queued", "target_generation": params[0]})
        raise AssertionError(text)


class Row:
    def __init__(self, *, one=None, rows=None, rowcount=0):
        self.one = one
        self.rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.rows


def database_for(connection):
    database = Database()

    @contextmanager
    def connect():
        yield connection

    database.connection = connect
    return database


def test_queued_graph_rebuild_is_coalesced_to_latest_generation():
    connection = CoalescingConnection()
    result = database_for(connection).create_graph_rebuild_job(7)

    assert result["id"] == "existing"
    assert result["target_generation"] == 8
    assert result["_coalesced"] is True
    assert not any("INSERT INTO jobs" in statement for statement, _ in connection.calls)


def test_outbox_poll_is_due_only_and_bounded():
    class OutboxConnection:
        def __init__(self):
            self.calls = []

        def execute(self, statement, params=None):
            self.calls.append((str(statement), tuple(params or ())))
            return Row(rows=[])

    connection = OutboxConnection()
    database_for(connection).list_pending_indexing_failures(7, limit=25)

    statement, params = connection.calls[0]
    assert "failure.status='pending'" in statement
    assert "failure.next_attempt_at<=NOW()" in statement
    assert "LIMIT %s" in statement
    assert params == (7, "relation", 25)


def test_document_embedding_query_qualifies_joined_columns():
    class EmbeddingConnection:
        def __init__(self):
            self.calls = []

        def execute(self, statement, params=None):
            self.calls.append((str(statement), tuple(params or ())))
            return Row(rows=[])

    connection = EmbeddingConnection()
    database_for(connection).list_document_embeddings(8, "chunk", "embed")

    statement, params = connection.calls[0]
    assert "SELECT v.id,v.project_id,v.document_id,v.chunk_id" in statement
    assert "ORDER BY v.chunk_id,v.id" in statement
    assert params == (8, "chunk", "embed")
