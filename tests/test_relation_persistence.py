from contextlib import contextmanager
import uuid

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.hugegraph_store import HugeGraphStore
from knowledge_forge.graph_policy import project_graph_policy_hash
from knowledge_forge.graph_safety import GraphMutationContext


class Result:
    def __init__(self, one=None, rows=None):
        self.one = one or {
            "id": 1,
            "project_id": 1,
            "verification_status": "unverified",
        }
        self.rows = rows or []

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.rows


class RecordingConnection:
    def __init__(self, rows=None):
        self.calls = []
        self.rows = rows or []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), params))
        return Result(rows=self.rows)


def database_with_connection(connection):
    database = Database()

    @contextmanager
    def connect():
        yield connection

    database.connection = connect
    return database


def test_relation_audit_ddl_is_idempotent_indexed_and_event_insert_is_append_only():
    assert "CREATE TABLE IF NOT EXISTS relation_candidates" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS relation_verification_events" in SCHEMA_SQL
    assert "idx_relation_candidates_project_status" in SCHEMA_SQL
    assert "idx_relation_candidates_document" in SCHEMA_SQL
    assert "idx_relation_verification_candidate" in SCHEMA_SQL
    assert "idx_relation_verification_scope" in SCHEMA_SQL
    event_block = SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS relation_verification_events", 1)[1]
    assert "ON CONFLICT" not in event_block.split("CREATE INDEX", 1)[0]


def test_confidence_zero_is_preserved_for_mentions_candidates_and_relations():
    connection = RecordingConnection()
    database = database_with_connection(connection)
    database.add_entity_mention({
        "project_id": 1, "document_id": 2, "revision_id": "revision-2",
        "chunk_id": 3, "canonical_entity_id": 4,
        "mention_text": "Alpha", "normalized_name": "alpha", "entity_type": "Organization",
        "confidence": 0,
    })
    mention_insert = connection.calls[0]
    assert mention_insert[1][10] == 0

    connection.calls.clear()
    database.upsert_relation_candidate({
        "candidate_id": "c1", "project_id": 1, "document_id": 2, "chunk_id": 3,
        "candidate_ordinal": 0, "source_mention_id": "e1", "source_name": "Alpha",
        "target_mention_id": "e2", "target_name": "Beta", "proposed_relation_type": "uses",
        "model_confidence": 0, "status": "uncertain",
    })
    assert connection.calls[0][1][13] == 0

    connection.calls.clear()
    database.add_relation_assertion({
        "project_id": 1, "document_id": 2, "chunk_id": 3,
        "source_entity_id": 4, "target_entity_id": 5, "relation_key": "r1",
        "relation_type": "uses", "confidence": 0,
    })
    assert connection.calls[0][1][8] == 0


def test_publishable_relation_query_uses_explicit_status_allowlist():
    connection = RecordingConnection()
    database = database_with_connection(connection)
    database.list_document_relations(9)
    statement, params = connection.calls[0]
    assert "verification_status=ANY" in statement
    assert params[1] == "manual_verified"
    assert set(params[2]) == {"rules_verified", "llm_verified"}
    assert params[3] == "relation-verifier-p0-v1"


class Graph:
    def __init__(self):
        self.edges = []

    def addEdge(self, label, source, target, properties):
        self.edges.append((label, source, target, properties))
        return None


class Client:
    def __init__(self):
        self.instance = Graph()

    def graph(self):
        return self.instance


class GraphDatabase:
    def __init__(self):
        self.default_listing_called = False

    def list_document_mentions(self, document_id):
        return []

    def list_document_relations(self, document_id):
        self.default_listing_called = True
        return [{
            "id": 7,
            "source_entity_id": 11,
            "target_entity_id": 12,
            "chunk_id": 3,
            "relation_type": "uses",
            "description": "Alpha uses Beta",
            "confidence": 0.9,
            "verification_status": "llm_verified",
        }]

    def list_all_document_relations(self, document_id):
        return [
            *self.list_document_relations(document_id),
            {"id": 8, "verification_status": "rejected"},
        ]

    def set_relation_graph_id(self, relation_id, edge_id):
        pass

    def assert_graph_lease(self, project_id, namespace_id, generation, lease_token):
        return True


def test_hugegraph_sync_requests_only_publishable_relations(monkeypatch):
    database = GraphDatabase()
    store = HugeGraphStore(database)
    client = Client()
    monkeypatch.setattr(store, "client", lambda: client)
    namespace = str(uuid.UUID(int=1))
    project = {
        "id": 1, "project_key": "p1", "active_schema_version": 1,
        "graph_namespace_id": namespace, "graph_generation": 3,
    }
    report = store.sync_document_assertions(
        project, {"id": 2}, mutation=GraphMutationContext(
            namespace_id=namespace, project_id=1, generation=3,
            policy_hash=project_graph_policy_hash(project), lease_token="lease",
        ),
    )
    assert database.default_listing_called is True
    assert report["relations"] == 1
    assert len(client.instance.edges) == 1
