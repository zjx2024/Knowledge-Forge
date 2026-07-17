from contextlib import contextmanager

from knowledge_forge.database import Database, SCHEMA_SQL


class _Result:
    def __init__(self, row=None):
        self.row = row or {"id": 1, "project_id": 3, "verification_status": "uncertain"}

    def fetchone(self):
        return dict(self.row)


class _Connection:
    def __init__(self):
        self.calls = []
        self.next_id = 10

    def execute(self, statement, params=None):
        rendered = str(statement)
        self.calls.append((rendered, params))
        self.next_id += 1
        return _Result({"id": self.next_id, "project_id": 3, "verification_status": "uncertain"})


def _database(connection):
    database = Database()

    @contextmanager
    def connect():
        yield connection

    database.connection = connect
    return database


def _candidate(candidate_id):
    return {
        "candidate_id": candidate_id,
        "project_id": 3,
        "document_id": 5,
        "chunk_id": 9,
        "candidate_ordinal": 0,
        "source_mention_id": "e1",
        "source_name": "Alpha",
        "target_mention_id": "e2",
        "target_name": "Beta",
        "proposed_relation_type": "uses",
        "status": "uncertain",
        "ingestion_run_id": "run-1",
    }


def _event(candidate_id):
    return {
        "candidate_id": candidate_id,
        "project_id": 3,
        "document_id": 5,
        "chunk_id": 9,
        "attempt": 1,
        "model_name": "test",
        "prompt_version": "p0",
        "schema_version": 1,
        "ingestion_run_id": "run-1",
        "request_json": {"candidates": ["c1", "c2"]},
        "raw_response": {"decisions": ["c1", "c2"]},
        "verdict": "reject",
        "reason": "fixture",
    }


def test_relation_verification_batch_ddl_preserves_legacy_event_columns():
    assert "CREATE TABLE IF NOT EXISTS relation_verification_batches" in SCHEMA_SQL
    assert "ALTER TABLE relation_verification_events ADD COLUMN IF NOT EXISTS batch_id" in SCHEMA_SQL
    event_block = SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS relation_verification_events", 1)[1]
    assert "request_json JSONB" in event_block
    assert "raw_response JSONB" in event_block


def test_shared_verifier_payload_is_persisted_once_and_events_reference_batch():
    connection = _Connection()
    database = _database(connection)

    database.persist_relation_audit_chain(
        [_candidate("c1"), _candidate("c2")],
        [_event("c1"), _event("c2")],
        [],
    )

    batch_calls = [call for call in connection.calls if "INSERT INTO relation_verification_batches" in call[0]]
    event_calls = [call for call in connection.calls if "INSERT INTO relation_verification_events" in call[0]]
    assert len(batch_calls) == 1
    assert len(event_calls) == 2
    assert batch_calls[0][1][9] == '{"candidates": ["c1", "c2"]}'
    assert all("batch_id" in statement for statement, _ in event_calls)
    assert all(params[9] == "{}" and params[10] is None for _, params in event_calls)
