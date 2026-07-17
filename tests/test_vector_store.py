from knowledge_forge.vector_store import VectorStore, cosine_similarity, lexical_similarity


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
