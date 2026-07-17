from __future__ import annotations

import json
from pathlib import Path

from knowledge_forge.quality import (
    GoldManifestError,
    b3_metrics,
    calibration_metrics,
    evaluate_entity_mentions,
    evaluate_relations,
    paired_discordance,
    pairwise_clustering_metrics,
    validate_gold_manifest,
    wilson_interval,
)


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
