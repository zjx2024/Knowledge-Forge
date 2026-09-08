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
    canonical_digest,
    entity_candidate_recall_at_k,
    evaluate_entity_pair_constraints,
    evaluate_entity_mentions,
    evaluate_relations,
    evaluate_release_gate,
    paired_discordance,
    pairwise_clustering_metrics,
    relation_pair_candidate_recall_at_budget,
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


def _pair(value: Any, *, relation: bool = False) -> tuple[str, str] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return str(value[0]), str(value[1])
    if not isinstance(value, dict):
        return None
    if relation:
        left_keys = ("subject_mention_id", "source_mention_id", "source", "left")
        right_keys = ("object_mention_id", "target_mention_id", "target", "right")
    else:
        left_keys = ("left_mention_id", "left", "source_mention_id", "source")
        right_keys = ("right_mention_id", "right", "target_mention_id", "target")
    left = next((value.get(key) for key in left_keys if value.get(key) is not None), None)
    right = next((value.get(key) for key in right_keys if value.get(key) is not None), None)
    if left is None or right is None:
        return None
    return str(left), str(right)


def _scoped_pair(item_id: str, value: Any, *, relation: bool = False) -> tuple[str, str] | None:
    pair = _pair(value, relation=relation)
    if pair is None:
        return None
    return f"{item_id}:{pair[0]}", f"{item_id}:{pair[1]}"


def _policy_metrics(gold_items: list[dict[str, Any]], predictions: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    gold_mentions: list[dict[str, Any]] = []
    predicted_mentions: list[dict[str, Any]] = []
    gold_relations: list[dict[str, Any]] = []
    predicted_relations: list[dict[str, Any]] = []
    gold_clusters: dict[str, str] = {}
    predicted_clusters: dict[str, str] = {}
    assignment_by_mention: dict[str, str] = {}
    must_links: list[tuple[str, str]] = []
    cannot_links: list[tuple[str, str]] = []
    hard_negatives: list[tuple[str, str]] = []
    gold_entity_candidate_links: list[tuple[str, str]] = []
    predicted_entity_candidate_links: list[tuple[str, str]] = []
    gold_relation_candidate_links: list[tuple[str, str]] = []
    predicted_relation_candidate_links: list[tuple[str, str]] = []
    entity_candidate_limits: list[int] = []
    relation_candidate_budgets: list[int] = []
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
            mention_id = str(mention.get("mention_id") or _mention_identity(mention))
            assignment_by_mention[f"{item_id}:{mention_id}"] = str(
                mention.get("canonical_id") or mention_id
            )
        for field, target in (
            ("must_links", must_links),
            ("cannot_links", cannot_links),
            ("entity_hard_negatives", hard_negatives),
        ):
            values = gold_item.get(field)
            if values is None and field == "entity_hard_negatives":
                values = gold_item.get("hard_negatives")
            for value in values or []:
                pair = _scoped_pair(item_id, value)
                if pair is not None:
                    target.append(pair)
        gold_entity_candidate_links.extend(
            pair for value in gold_item.get("must_links") or []
            if (pair := _scoped_pair(item_id, value)) is not None
        )
        predicted_entity_candidate_links.extend(
            pair for value in prediction.get("entity_candidates") or []
            if (pair := _scoped_pair(item_id, value)) is not None
        )
        if prediction.get("entity_candidate_k") is not None:
            entity_candidate_limits.append(int(prediction["entity_candidate_k"]))
        current_gold_relations = list(gold_item.get("relations") or [])
        current_predicted_relations = list(prediction.get("relations") or [])
        gold_relations.extend(current_gold_relations)
        predicted_relations.extend(current_predicted_relations)
        for relation in current_gold_relations:
            pair = _scoped_pair(item_id, relation, relation=True)
            if pair is not None:
                gold_relation_candidate_links.append(pair)
        relation_candidates = (
            prediction.get("relation_candidates")
            or prediction.get("candidates")
            or current_predicted_relations
        )
        for candidate in relation_candidates:
            pair = _scoped_pair(item_id, candidate, relation=True)
            if pair is not None:
                predicted_relation_candidate_links.append(pair)
        if prediction.get("relation_candidate_budget") is not None:
            relation_candidate_budgets.append(int(prediction["relation_candidate_budget"]))
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
    entity_candidates = entity_candidate_recall_at_k(
        gold_entity_candidate_links,
        predicted_entity_candidate_links,
        k=max(entity_candidate_limits) if entity_candidate_limits else None,
    )
    relation_candidates = relation_pair_candidate_recall_at_budget(
        gold_relation_candidate_links,
        predicted_relation_candidate_links,
        budget=max(relation_candidate_budgets) if relation_candidate_budgets else None,
    )
    entity_constraints = evaluate_entity_pair_constraints(
        must_links=must_links,
        cannot_links=cannot_links,
        hard_negatives=hard_negatives,
        assignments=assignment_by_mention,
    )
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
        "entity_candidate_recall_at_k": entity_candidates,
        "relation_pair_candidate_recall_at_budget": relation_candidates,
        "entity_constraints": entity_constraints,
        "calibration": calibration_metrics(calibration_labels, calibration_probabilities),
        "cost": {"tokens": total_tokens, "model_calls": total_calls},
        "safety": {
            "invalid_evidence_publication": invalid_evidence,
            "incomplete_entity_constraint_coverage": int(
                not entity_constraints["coverage_complete"]
            ),
            "critical_errors": invalid_evidence + int(
                not entity_constraints["coverage_complete"]
            ),
        },
        "item_correct": item_correct,
        "denominators": {
            "gold_mentions": len(gold_mentions), "predicted_mentions": len(predicted_mentions),
            "gold_relations": len(gold_relations), "predicted_relations": len(predicted_relations),
            "must_links": len(must_links), "cannot_links": len(cannot_links),
            "hard_negatives": len(hard_negatives),
            "entity_candidate_pairs": len(predicted_entity_candidate_links),
            "relation_candidate_pairs": len(predicted_relation_candidate_links),
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
            "entity_candidate_recall_at_k": float(
                control["entity_candidate_recall_at_k"]["recall"]
            ),
            "relation_pair_candidate_recall_at_budget": float(
                control["relation_pair_candidate_recall_at_budget"]["recall"]
            ),
            "critical_errors": int(control["safety"]["critical_errors"]),
            "cost": float(control["cost"]["tokens"]),
        },
        treatment={
            "entity_f1": float(treatment["entity_mentions"]["exact_span"]["f1"]),
            "b3_f1": float(treatment["b3"]["f1"]),
            "relation_f1": float(treatment["relations"]["strict"]["f1"]),
            "entity_candidate_recall_at_k": float(
                treatment["entity_candidate_recall_at_k"]["recall"]
            ),
            "relation_pair_candidate_recall_at_budget": float(
                treatment["relation_pair_candidate_recall_at_budget"]["recall"]
            ),
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
