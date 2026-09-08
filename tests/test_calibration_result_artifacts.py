from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_relation_upper_bound_result_is_arithmetically_consistent_and_scoped():
    result = json.loads((
        ROOT / "benchmarks/results/knowledge_forge_relation_upper_bound_2026-08-11.json"
    ).read_text(encoding="utf-8"))
    rows = result["documents"]
    aggregate = result["aggregate"]

    gold = sum(item["gold_unique_facts"] for item in rows)
    hits = sum(item["exact_hits"] for item in rows)
    supported = sum(item["supported_unique_facts"] for item in rows)
    leaks = sum(item["hard_negative_leaks"] for item in rows)
    for item in rows:
        assert len(item["all_label_misses"]) == (
            item["gold_unique_facts"] - item["exact_hits"]
        )
        assert len(item["non_gold_supported"]) == (
            item["non_gold_supported_facts"]
        )
        assert item["supported_unique_facts"] == (
            item["exact_hits"] + item["non_gold_supported_facts"]
        )

    assert aggregate["gold_unique_facts"] == gold == 78
    assert aggregate["exact_hits"] == hits == 76
    assert aggregate["supported_unique_facts"] == supported == 78
    assert aggregate["hard_negative_leaks"] == leaks == 0
    assert aggregate["recall"] == round(hits / gold, 10)
    assert aggregate["precision"] == round(hits / supported, 10)
    primary = aggregate["primary_core_direct"]
    core_gold = sum(item["core_direct_gold_unique_facts"] for item in rows)
    core_hits = sum(item["core_direct_exact_hits"] for item in rows)
    interpretation = sum(item["interpretation_unique_facts"] for item in rows)
    interpretation_hits = sum(
        item["interpretation_exact_hits"] for item in rows
    )
    non_gold = sum(item["non_gold_supported_facts"] for item in rows)
    assert primary["gold_unique_facts"] == core_gold == 69
    assert primary["exact_hits"] == core_hits == 68
    assert primary["interpretation_unique_facts"] == interpretation == 9
    assert primary["interpretation_exact_hits"] == interpretation_hits == 8
    assert primary["non_gold_supported_facts"] == non_gold == 2
    assert primary["precision_evaluated_facts"] == core_hits + non_gold
    assert primary["precision"] == round(
        core_hits / (core_hits + non_gold), 10
    )
    assert primary["recall"] == round(core_hits / core_gold, 10)
    assert result["scope"]["end_to_end_public_job"] is False
    default_delta = result["current_default_difference"]
    assert default_delta["counterfactual_core_recall_lower_bound_if_both_unique_and_required"] == round(
        (core_hits - default_delta["observed_extra_bindings_from_second_round"])
        / core_gold,
        10,
    )
    assert default_delta[
        "counterfactual_minimum_document_core_recall"
    ] == round((11 - 2) / 11, 10)
    assert default_delta["status"] == (
        "requires_end_to_end_runtime_confirmation"
    )
