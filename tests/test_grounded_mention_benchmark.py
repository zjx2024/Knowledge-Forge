from benchmarks.evaluate_grounded_mentions import score_mentions


def test_grounded_mention_scorer_separates_exact_and_overlap_boundaries():
    gold = [
        {"surface": "Alpha system", "span": [0, 12], "type": "Product"},
        {"surface": "Beta", "span": [23, 27], "type": "Organization"},
    ]
    predicted = [
        {"surface": "Alpha", "start": 0, "end": 5, "type": "Product"},
        {"surface": "Beta", "start": 23, "end": 27, "type": "Person"},
        {"surface": "Gamma", "start": 32, "end": 37, "type": "Product"},
    ]

    result = score_mentions(gold, predicted)

    assert result["exact"]["hits"] == 1
    assert result["exact"]["precision"] == 1 / 3
    assert result["exact"]["recall"] == 1 / 2
    assert result["overlap"]["hits"] == 2
    assert result["overlap"]["precision"] == 2 / 3
    assert result["overlap"]["recall"] == 1
    assert result["type_accuracy_on_exact"] == 0
    assert result["strict_typed"]["micro"]["hits"] == 0
    assert result["strict_typed"]["micro"]["f1"] == 0
    assert result["strict_typed"]["macro_f1"] == 0
    assert set(result["strict_typed"]["by_type"]) == {
        "Organization", "Person", "Product",
    }
