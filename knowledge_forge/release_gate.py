"""Artifact-derived quality evaluation and release-gate command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .database import Database
from .quality import (
    b3_metrics,
    calibration_metrics,
    candidate_recall,
    canonical_digest,
    evaluate_entity_mentions,
    evaluate_relations,
    evaluate_release_gate,
    paired_discordance,
    pairwise_clustering_metrics,
    validate_gold_annotations,
    validate_gold_manifest,
)


class ArtifactEvaluationError(RuntimeError):
    """A frozen evaluation input is missing, mutable, or hash-inconsistent."""


ALLOWED_EVALUATOR_VERSIONS = {"quality-v2"}


def _mention_identity(item: dict[str, Any]) -> str:
    return "|".join(str(value) for value in (
        item.get("document_id"), item.get("revision_id") or item.get("content_hash"),
        item.get("chunk_id"), item.get("start"), item.get("end"),
    ))


def _policy_metrics(gold_items: list[dict[str, Any]], predictions: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    gold_mentions: list[dict[str, Any]] = []
    predicted_mentions: list[dict[str, Any]] = []
    gold_relations: list[dict[str, Any]] = []
    predicted_relations: list[dict[str, Any]] = []
    gold_clusters: dict[str, str] = {}
    predicted_clusters: dict[str, str] = {}
    gold_candidate_links: list[tuple[str, str]] = []
    predicted_candidate_links: list[tuple[str, str]] = []
    calibration_labels: list[int] = []
    calibration_probabilities: list[float] = []
    item_correct: list[bool] = []
    total_tokens = total_calls = 0
    predictions_by_id = {str(item["item_id"]): item for item in predictions}

    for gold_item in gold_items:
        item_id = str(gold_item["item_id"])
        prediction = predictions_by_id[item_id][policy]
        current_gold_mentions = list(gold_item.get("mentions") or [])
        current_predictions = list(prediction.get("mentions") or [])
        gold_mentions.extend(current_gold_mentions)
        predicted_mentions.extend(current_predictions)
        for mention in current_gold_mentions:
            gold_clusters[_mention_identity(mention)] = str(
                mention.get("canonical_id") or mention.get("mention_id") or _mention_identity(mention)
            )
        for mention in current_predictions:
            predicted_clusters[_mention_identity(mention)] = str(
                mention.get("canonical_id") or mention.get("mention_id") or _mention_identity(mention)
            )
        current_gold_relations = list(gold_item.get("relations") or [])
        current_predicted_relations = list(prediction.get("relations") or [])
        gold_relations.extend(current_gold_relations)
        predicted_relations.extend(current_predicted_relations)
        for relation in current_gold_relations:
            gold_candidate_links.append((
                str(relation.get("subject_mention_id") or relation.get("source")),
                str(relation.get("object_mention_id") or relation.get("target")),
            ))
        for candidate in prediction.get("candidates") or current_predicted_relations:
            predicted_candidate_links.append((
                str(candidate.get("subject_mention_id") or candidate.get("source_mention_id") or candidate.get("source")),
                str(candidate.get("object_mention_id") or candidate.get("target_mention_id") or candidate.get("target")),
            ))
        probabilities = {
            str(item["key"]): float(item.get("probability") or 0.0)
            for item in prediction.get("calibration") or []
        }
        for item in gold_item.get("calibration") or []:
            calibration_labels.append(int(item["label"]))
            calibration_probabilities.append(probabilities.get(str(item["key"]), 0.0))
        cost = prediction.get("cost") or {}
        total_tokens += int(cost.get("tokens") or 0)
        total_calls += int(cost.get("model_calls") or 0)
        item_correct.append(
            evaluate_entity_mentions(current_gold_mentions, current_predictions)["exact_span"]["f1"] == 1.0
            and evaluate_relations(current_gold_relations, current_predicted_relations)["strict"]["f1"] == 1.0
        )

    mention_report = evaluate_entity_mentions(gold_mentions, predicted_mentions)
    pairwise = pairwise_clustering_metrics(gold_clusters, predicted_clusters)
    b3 = b3_metrics(gold_clusters, predicted_clusters)
    relation_report = evaluate_relations(gold_relations, predicted_relations)
    candidates = candidate_recall(gold_candidate_links, predicted_candidate_links)
    invalid_evidence = sum(
        relation.get("evidence_start") is None or relation.get("evidence_end") is None
        or not (relation.get("evidence_text") or relation.get("evidence"))
        for relation in predicted_relations
    )
    return {
        "entity_mentions": mention_report,
        "pairwise": pairwise,
        "b3": b3,
        "relations": relation_report,
        "candidate_recall": candidates,
        "calibration": calibration_metrics(calibration_labels, calibration_probabilities),
        "cost": {"tokens": total_tokens, "model_calls": total_calls},
        "safety": {
            "invalid_evidence_publication": invalid_evidence,
            "critical_errors": invalid_evidence,
        },
        "item_correct": item_correct,
        "denominators": {
            "gold_mentions": len(gold_mentions), "predicted_mentions": len(predicted_mentions),
            "gold_relations": len(gold_relations), "predicted_relations": len(predicted_relations),
            "calibration_items": len(calibration_labels), "items": len(gold_items),
        },
    }


def evaluate_frozen_bundle(
    *, snapshot_bundle: dict[str, Any], gold_manifest: dict[str, Any],
    gold_items: list[dict[str, Any]], evaluator_version: str,
) -> dict[str, Any]:
    """Derive every metric from immutable item facts and verify their Merkle-like root."""

    if evaluator_version not in ALLOWED_EVALUATOR_VERSIONS:
        raise ArtifactEvaluationError("evaluator version is not allowed")
    snapshot = snapshot_bundle.get("snapshot") or {}
    if snapshot.get("status") != "ready" or int(snapshot.get("failed_items") or 0):
        raise ArtifactEvaluationError("snapshot is not complete and ready")
    artifacts = list(snapshot_bundle.get("items") or [])
    if (
        int(snapshot.get("expected_items") or 0) != int(snapshot.get("completed_items") or 0)
        or len(artifacts) != int(snapshot.get("expected_items") or 0)
    ):
        raise ArtifactEvaluationError("artifact item universe is not complete")
    item_ids = [str(item.get("item_id") or "") for item in artifacts]
    if not all(item_ids) or len(set(item_ids)) != len(item_ids):
        raise ArtifactEvaluationError("artifact item ids are missing or duplicated")
    verified_hashes: list[list[str]] = []
    for item in sorted(artifacts, key=lambda value: str(value["item_id"])):
        expected = str(item.get("item_hash") or "")
        payload = {key: value for key, value in item.items() if key != "item_hash"}
        if not expected or canonical_digest(payload) != expected:
            raise ArtifactEvaluationError(f"artifact item hash mismatch: {item['item_id']}")
        verified_hashes.append([str(item["item_id"]), expected])
    if canonical_digest(verified_hashes) != str(snapshot.get("artifact_root_hash") or ""):
        raise ArtifactEvaluationError("artifact root hash mismatch")
    manifest_report = validate_gold_annotations(
        gold_manifest, gold_items,
        validate_gold_manifest(gold_manifest, require_production=False),
    )
    manifest_ids = {str(item["id"]) for item in gold_manifest["items"]}
    gold_ids = {str(item.get("item_id") or "") for item in gold_items}
    if manifest_ids != gold_ids or gold_ids != set(item_ids):
        raise ArtifactEvaluationError("Gold, manifest and artifact universes differ")
    gold_items_hash = canonical_digest(sorted(gold_items, key=lambda item: str(item["item_id"])))
    control = _policy_metrics(gold_items, artifacts, "control")
    treatment = _policy_metrics(gold_items, artifacts, "treatment")
    gate_report = evaluate_release_gate(
        control={
            "entity_f1": float(control["entity_mentions"]["exact_span"]["f1"]),
            "b3_f1": float(control["b3"]["f1"]),
            "relation_f1": float(control["relations"]["strict"]["f1"]),
            "candidate_recall": float(control["candidate_recall"]["recall"]),
            "critical_errors": int(control["safety"]["critical_errors"]),
            "cost": float(control["cost"]["tokens"]),
        },
        treatment={
            "entity_f1": float(treatment["entity_mentions"]["exact_span"]["f1"]),
            "b3_f1": float(treatment["b3"]["f1"]),
            "relation_f1": float(treatment["relations"]["strict"]["f1"]),
            "candidate_recall": float(treatment["candidate_recall"]["recall"]),
            "critical_errors": int(treatment["safety"]["critical_errors"]),
            "cost": float(treatment["cost"]["tokens"]),
        },
        manifest=manifest_report,
    )
    report = {
        "evaluator_version": evaluator_version,
        "snapshot_id": str(snapshot.get("id") or ""),
        "snapshot_manifest_hash": str(snapshot.get("manifest_hash") or ""),
        "artifact_root_hash": str(snapshot.get("artifact_root_hash") or ""),
        "gold_manifest_hash": canonical_digest(gold_manifest),
        "gold_items_hash": gold_items_hash,
        "manifest": manifest_report,
        "control": control,
        "treatment": treatment,
        "paired": paired_discordance(control["item_correct"], treatment["item_correct"]),
        "gate": gate_report,
    }
    return {
        "status": "completed", "snapshot_id": report["snapshot_id"],
        "evaluator_version": evaluator_version,
        "manifest_report": manifest_report, "gate_report": gate_report,
        "artifact_root_hash": report["artifact_root_hash"],
        "gold_manifest_hash": report["gold_manifest_hash"],
        "gold_items_hash": gold_items_hash,
        "item_hashes": [value[1] for value in verified_hashes],
        "report": report, "report_hash": canonical_digest(report),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a persisted evaluation and derive its gate")
    parser.add_argument("--evaluation-run-id", required=True)
    parser.add_argument("--approved-by")
    args = parser.parse_args(list(argv) if argv is not None else None)
    from .service import KnowledgeForgeService

    result = KnowledgeForgeService(Database()).evaluate_policy_snapshot_gate(
        args.evaluation_run_id, approved_by=args.approved_by,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
