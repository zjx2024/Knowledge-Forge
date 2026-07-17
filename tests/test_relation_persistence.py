from contextlib import contextmanager

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.hugegraph_store import HugeGraphStore


class Result:
    def __init__(self, one=None, rows=None):
        self.one = one or {"id": 1}
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
        "project_id": 1, "document_id": 2, "chunk_id": 3, "canonical_entity_id": 4,
        "mention_text": "Alpha", "normalized_name": "alpha", "entity_type": "Organization",
        "confidence": 0,
    })
    mention_insert = connection.calls[0]
    assert mention_insert[1][8] == 0

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
    database.list_document_relations(9, publishable_only=True)
    statement, params = connection.calls[0]
    assert "verification_status=ANY" in statement
    assert set(params[1]) == {"rules_verified", "llm_verified", "manual_verified"}


class Graph:
    pass


class Client:
    def graph(self):
        return Graph()


class GraphDatabase:
    def __init__(self):
        self.publishable_only = None

    def list_document_mentions(self, document_id):
        return []

    def list_document_relations(self, document_id, *, publishable_only=False):
        self.publishable_only = publishable_only
        return []


def test_hugegraph_sync_requests_only_publishable_relations(monkeypatch):
    database = GraphDatabase()
    store = HugeGraphStore(database)
    monkeypatch.setattr(store, "client", lambda: Client())
    report = store.sync_document_assertions(
        {"id": 1, "project_key": "p1", "active_schema_version": 1}, {"id": 2},
    )
    assert database.publishable_only is True
    assert report["relations"] == 0
