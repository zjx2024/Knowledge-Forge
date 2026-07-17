from knowledge_forge.value_parsing import coerce_confidence


def test_coerce_confidence_accepts_common_model_formats():
    assert coerce_confidence("high") == 0.9
    assert coerce_confidence("低") == 0.3
    assert coerce_confidence("75%") == 0.75
    assert coerce_confidence("75") == 0.75
    assert coerce_confidence("0.82") == 0.82


def test_coerce_confidence_clamps_and_falls_back():
    assert coerce_confidence(120) == 1.0
    assert coerce_confidence(-1) == 0.0
    assert coerce_confidence("unknown", 0.4) == 0.4
    assert coerce_confidence("nan", 0.4) == 0.4
