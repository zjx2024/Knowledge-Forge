from contextlib import contextmanager

from knowledge_forge.database import Database


class _Rows:
    def fetchall(self):
        return []


class _Connection:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        self.calls.append((str(statement), tuple(params or ())))
        return _Rows()


def test_document_chunk_preview_prefers_latest_building_revision():
    connection = _Connection()
    database = Database()

    @contextmanager
    def connect():
        yield connection

    database.connection = connect
    database.list_document_chunks(17)

    statement, params = connection.calls[0]
    assert "revision.status='building'" in statement
    assert "ORDER BY revision.content_version DESC" in statement
    assert "d.active_revision_id" in statement
    assert params == (17,)
