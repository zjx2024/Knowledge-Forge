from benchmarks.knowledge_forge_hybrid_rag_v1.validate import validate


def test_hybrid_rag_question_contract_is_complete_and_bilingual():
    report = validate()

    assert report == {
        "status": "valid",
        "questions": 24,
        "languages": {"en": 12, "zh-CN": 12},
        "task_types": {
            "aggregation": 8,
            "multi_hop": 6,
            "single_fact": 5,
            "vector_fallback": 5,
        },
        "documents_covered": 10,
        "positive_labels_referenced": 55,
        "interpretation_labels_referenced": 0,
        "negative_labels_referenced": 8,
    }
