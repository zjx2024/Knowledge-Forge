from benchmarks.knowledge_forge_long_document_stress_v1.evaluate import evaluate


def test_long_document_processing_and_base_call_schedule_scale_near_linearly():
    report = evaluate()

    assert report["status"] == "pass"
    assert report["quality_gold"] is False
    assert report["source_documents"] == 10
    assert all(report["checks"].values())
