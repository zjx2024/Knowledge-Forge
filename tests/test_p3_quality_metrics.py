from __future__ import annotations

import json
from pathlib import Path

from knowledge_forge.quality import (
    GoldManifestError,
    b3_metrics,
    calibration_metrics,
    entity_candidate_recall_at_k,
    evaluate_entity_mentions,
    evaluate_relations,
    paired_discordance,
    pairwise_clustering_metrics,
    validate_gold_manifest,
    relation_pair_candidate_recall_at_budget,
    wilson_interval,
)
from knowledge_forge.release_gate import _policy_metrics


def test_manifest_rejects_synthetic_or_model_only_production_gate():
    manifest = {
        "dataset": "fixture", "version": "1", "frozen_at": "2026-01-01T00:00:00Z",
        "items": [{
            "id": "dev-1", "content_sha256": "a" * 64, "split": "dev",
            "label_source": "synthetic", "reviewer": "", "reviewed_at": "",
        }],
    }
    result = validate_gold_manifest(manifest, require_production=False)
    assert result["engineering_valid"] is True
    assert result["production_valid"] is False
    try:
        validate_gold_manifest(manifest, require_production=True)
    except GoldManifestError as exc:
        assert "human" in str(exc).lower()
    else:
        raise AssertionError("synthetic manifest must not authorize production")


def test_repository_engineering_bundle_hashes_are_valid_but_not_production_gold():
    root = Path(__file__).parent / "fixtures" / "kg_gold"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = validate_gold_manifest(manifest, bundle_root=root)
    assert result["engineering_valid"] is True
    assert result["production_valid"] is False


def test_manifest_rejects_dev_holdout_overlap_and_hash_drift():
    manifest = {
        "dataset": "fixture", "version": "1", "frozen_at": "2026-01-01T00:00:00Z",
        "items": [
            {"id": "same", "content_sha256": "a" * 64, "split": "dev",
             "label_source": "human", "reviewer": "r", "reviewed_at": "2026-01-01"},
            {"id": "same", "content_sha256": "b" * 64, "split": "holdout",
             "label_source": "human", "reviewer": "r", "reviewed_at": "2026-01-01"},
        ],
    }
    try:
        validate_gold_manifest(manifest)
    except GoldManifestError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("split overlap must be rejected")


def test_entity_span_and_type_metrics_are_exact_and_overlap_aware():
    gold = [
        {"document_id": "d", "start": 0, "end": 4, "type": "Organization"},
        {"document_id": "d", "start": 10, "end": 14, "type": "Product"},
    ]
    predicted = [
        {"document_id": "d", "start": 0, "end": 4, "type": "Organization"},
        {"document_id": "d", "start": 11, "end": 14, "type": "Organization"},
    ]
    metrics = evaluate_entity_mentions(gold, predicted)
    assert metrics["exact_span"]["f1"] == 0.5
    assert metrics["overlap"]["f1"] == 1.0
    assert metrics["type_macro_f1"] < 1.0


def test_clustering_relation_and_calibration_metrics_have_explicit_denominators():
    gold_clusters = {"m1": "a", "m2": "a", "m3": "b"}
    predicted = {"m1": "x", "m2": "x", "m3": "y"}
    assert pairwise_clustering_metrics(gold_clusters, predicted)["f1"] == 1.0
    assert b3_metrics(gold_clusters, predicted)["f1"] == 1.0

    gold_relations = [{"source": "m1", "predicate": "uses", "target": "m3",
                       "direction": "forward", "evidence_start": 2, "evidence_end": 8}]
    predicted_relations = [dict(gold_relations[0], confidence=0.9)]
    relation = evaluate_relations(gold_relations, predicted_relations)
    assert relation["strict"]["f1"] == 1.0
    assert relation["evidence_coverage"] == 1.0
    calibration = calibration_metrics([1, 0], [0.9, 0.2], bins=5)
    assert calibration["count"] == 2
    assert 0 <= calibration["brier"] <= 1
    assert 0 <= calibration["ece"] <= 1


def test_paired_discordance_and_wilson_interval_are_reported_without_fake_precision():
    report = paired_discordance([True, False, False, True], [True, True, False, False])
    assert report == {"both_correct": 1, "control_only": 1, "treatment_only": 1,
                      "both_wrong": 1, "count": 4}
    lower, upper = wilson_interval(9, 10)
    assert 0 <= lower < 0.9 < upper <= 1


def test_entity_and_relation_candidate_recall_have_distinct_names_and_sample_ids():
    entity = entity_candidate_recall_at_k(
        [("m1", "m2"), ("m3", "m4")],
        [("m2", "m1")],
        k=10,
    )
    relation = relation_pair_candidate_recall_at_budget(
        [("m1", "m2"), ("m2", "m3")],
        [("m1", "m2")],
        budget=32,
    )

    assert entity["metric"] == "entity_candidate_recall_at_k"
    assert entity["true_positive_ids"] == ["m1|m2"]
    assert entity["false_negative_ids"] == ["m3|m4"]
    assert entity["k"] == 10
    assert relation["metric"] == "relation_pair_candidate_recall_at_budget"
    assert relation["true_positive_ids"] == ["m1|m2"]
    assert relation["false_negative_ids"] == ["m2|m3"]
    assert relation["budget"] == 32


def test_release_evaluator_consumes_must_cannot_hard_negative_and_both_candidate_universes():
    mentions = [
        {"mention_id": "m1", "document_id": "d", "revision_id": "r", "chunk_id": "c",
         "start": 0, "end": 2, "type": "Person", "canonical_id": "alice"},
        {"mention_id": "m2", "document_id": "d", "revision_id": "r", "chunk_id": "c",
         "start": 3, "end": 5, "type": "Person", "canonical_id": "alice"},
        {"mention_id": "m3", "document_id": "d", "revision_id": "r", "chunk_id": "c",
         "start": 6, "end": 8, "type": "Person", "canonical_id": "bob"},
    ]
    relation = {
        "document_id": "d", "revision_id": "r", "chunk_id": "c",
        "subject_mention_id": "m1", "subject_canonical_id": "alice",
        "predicate": "knows", "object_mention_id": "m3", "object_canonical_id": "bob",
        "direction": "forward", "evidence_text": "A knows B", "evidence_start": 0,
        "evidence_end": 9, "snapshot_id": "s", "policy_hash": "p",
    }
    gold = [{
        "item_id": "item-1",
        "mentions": mentions,
        "must_links": [["m1", "m2"]],
        "cannot_links": [["m1", "m3"]],
        "hard_negatives": [["m2", "m3"]],
        "relations": [relation],
    }]
    predictions = [{
        "item_id": "item-1",
        "treatment": {
            "mentions": mentions,
            "entity_candidates": [["m1", "m2"]],
            "relation_candidates": [["m1", "m3"]],
            "relations": [relation],
        },
    }]

    report = _policy_metrics(gold, predictions, "treatment")

    assert "candidate_recall" not in report
    assert report["entity_candidate_recall_at_k"]["recall"] == 1.0
    assert report["relation_pair_candidate_recall_at_budget"]["recall"] == 1.0
    assert report["entity_constraints"]["confusion_matrix"] == {
        "true_positive": 1,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
    }
    assert report["entity_constraints"]["denominators"] == {
        "must_links": 1,
        "cannot_links": 1,
        "hard_negatives": 1,
    }
