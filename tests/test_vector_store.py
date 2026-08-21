from knowledge_forge.vector_store import VectorStore, cosine_similarity, lexical_similarity
from knowledge_forge.config import Config


def test_cosine_similarity_handles_direction_and_mismatch():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0


def test_lexical_similarity_supports_chinese_bigrams():
    assert lexical_similarity("工业机器人", "甲公司生产工业机器人") > 0.7
    assert lexical_similarity("工业机器人", "天气预报") == 0.0


class FakeOllama:
    def embed(self, values, model):
        assert values == ["工业机器人"]
        assert model == "bge-m3:latest"
        return [[1.0, 0.0]]


class FakeDatabase:
    def __init__(self):
        self.call = None

    def search_embeddings(self, project_id, object_type, model_name, query_vector, query, limit):
        self.call = (project_id, object_type, model_name, query_vector, query, limit)
        return [
            {"id": 1, "content": "甲公司生产工业机器人", "dense_score": 0.8},
            {"id": 2, "content": "工业设备", "dense_score": 0.9},
        ]


def test_search_uses_database_vector_candidates_and_hybrid_reranking():
    database = FakeDatabase()
    store = VectorStore(database, FakeOllama())
    results = store.search_type(
        {"id": 7, "embedding_model": "bge-m3:latest"}, "工业机器人", "chunk", 2
    )

    assert database.call == (7, "chunk", "bge-m3:latest", [1.0, 0.0], "工业机器人", 2)
    assert results[0]["id"] == 1
    assert "embedding" not in results[0]


class IndexOllama:
    def embed(self, values, model):
        assert values == ["first", "second"]
        assert model == "bge-m3:latest"
        return [[1.0, 0.0], [0.0, 1.0]]


class IndexDatabase:
    def __init__(self):
        self.records = []

    def replace_document_embeddings(self, document_id, records):
        assert document_id == 9
        self.records = [
            {**item, "dimension": len(item["embedding"])} for item in records
        ]

    def list_document_embeddings(
        self, document_id, object_type, model_name, *, revision_id=None,
    ):
        assert (document_id, object_type, model_name, revision_id) == (
            9, "chunk", "bge-m3:latest", "revision-1",
        )
        return self.records


def test_chunk_index_tracks_runtime_profile_without_reparsing(monkeypatch):
    monkeypatch.setattr(Config, "ollama_embedding_runtime_profile", "gpu-test-v1")
    database = IndexDatabase()
    store = VectorStore(database, IndexOllama())
    chunks = [
        {
            "id": 1, "chunk_key": "chunk-1", "content": "first",
            "content_hash": "hash-1", "revision_id": "revision-1",
            "section_id": 11,
        },
        {
            "id": 2, "chunk_key": "chunk-2", "content": "second",
            "content_hash": "hash-2", "revision_id": "revision-1",
            "section_id": 12,
        },
    ]
    project = {"id": 7, "embedding_model": "bge-m3:latest"}

    assert store.index_chunks(project, 9, chunks) == 2
    assert all(
        item["revision_id"] == "revision-1"
        and item["metadata"]["embedding_runtime_profile"] == "gpu-test-v1"
        for item in database.records
    )
    assert store.document_index_is_current(project, 9, chunks) is True

    database.records[0]["metadata"]["embedding_runtime_profile"] = "cpu-old-v1"
    assert store.document_index_is_current(project, 9, chunks) is False
