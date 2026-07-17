"""Deterministic P3 quality metrics and release-gate validation."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


class GoldManifestError(ValueError):
    """The annotation manifest cannot be used for the requested gate."""


def _prf(true_positive: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else float(gold == 0)
    recall = true_positive / gold if gold else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive, "predicted": predicted, "gold": gold,
        "precision": precision, "recall": recall, "f1": f1,
    }


def validate_gold_manifest(
    manifest: dict[str, Any], *, require_production: bool = False,
    bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    if not manifest.get("dataset") or not manifest.get("version") or not manifest.get("frozen_at"):
        raise GoldManifestError("manifest requires dataset, version and frozen_at")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise GoldManifestError("manifest items must be a non-empty array")
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    production_valid = True
    root = Path(bundle_root) if bundle_root else None
    for item in items:
        item_id = str(item.get("id") or "")
        split = str(item.get("split") or "")
        digest = str(item.get("content_sha256") or "")
        if not item_id or split not in {"dev", "holdout"} or len(digest) != 64:
            raise GoldManifestError("each item requires id, dev/holdout split and sha256")
        if item_id in seen_ids and seen_ids[item_id] != split:
            raise GoldManifestError("dev/holdout overlap detected by item id")
        if digest in seen_hashes and seen_hashes[digest] != split:
            raise GoldManifestError("dev/holdout overlap detected by content hash")
        seen_ids[item_id], seen_hashes[digest] = split, split
        if root and item.get("path"):
            path = root / str(item["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise GoldManifestError(f"content hash drift: {item_id}")
        human = str(item.get("label_source") or "").lower() in {
            "human", "human_reviewed", "independent_human",
        }
        production_valid &= bool(human and item.get("reviewer") and item.get("reviewed_at"))
    result = {
        "engineering_valid": True,
        "production_valid": bool(production_valid),
        "items": len(items),
        "dev_items": sum(item.get("split") == "dev" for item in items),
        "holdout_items": sum(item.get("split") == "holdout" for item in items),
    }
    if require_production and not result["production_valid"]:
        raise GoldManifestError("production gate requires independently reviewed human gold")
    return result


def _span_key(item: dict[str, Any]) -> tuple[Any, int, int]:
    return item.get("document_id"), int(item["start"]), int(item["end"])


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("document_id") == right.get("document_id")
        and max(int(left["start"]), int(right["start"]))
        < min(int(left["end"]), int(right["end"]))
    )


def evaluate_entity_mentions(
    gold: list[dict[str, Any]], predicted: list[dict[str, Any]],
) -> dict[str, Any]:
    exact = len(set(map(_span_key, gold)) & set(map(_span_key, predicted)))
    unmatched = set(range(len(gold)))
    overlap = 0
    type_pairs: list[tuple[str, str]] = []
    for prediction in predicted:
        match = next((index for index in sorted(unmatched) if _overlaps(prediction, gold[index])), None)
        if match is None:
            continue
        unmatched.remove(match)
        overlap += 1
        type_pairs.append((str(gold[match].get("type") or ""), str(prediction.get("type") or "")))
    types = sorted({str(item.get("type") or "") for item in gold + predicted})
    type_f1 = []
    for entity_type in types:
        tp = sum(left == entity_type and right == entity_type for left, right in type_pairs)
        predicted_count = sum(str(item.get("type") or "") == entity_type for item in predicted)
        gold_count = sum(str(item.get("type") or "") == entity_type for item in gold)
        type_f1.append(float(_prf(tp, predicted_count, gold_count)["f1"]))
    return {
        "exact_span": _prf(exact, len(predicted), len(gold)),
        "overlap": _prf(overlap, len(predicted), len(gold)),
        "type_macro_f1": sum(type_f1) / len(type_f1) if type_f1 else 1.0,
    }


def _positive_pairs(assignments: dict[str, str]) -> set[tuple[str, str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for mention, cluster in assignments.items():
        groups[str(cluster)].append(str(mention))
    return {
        (members[left], members[right])
        for members in groups.values() for left in range(len(members))
        for right in range(left + 1, len(members))
    }


def pairwise_clustering_metrics(gold: dict[str, str], predicted: dict[str, str]) -> dict[str, Any]:
    mentions = set(gold) & set(predicted)
    gold_pairs = _positive_pairs({key: gold[key] for key in mentions})
    predicted_pairs = _positive_pairs({key: predicted[key] for key in mentions})
    return _prf(len(gold_pairs & predicted_pairs), len(predicted_pairs), len(gold_pairs))


def b3_metrics(gold: dict[str, str], predicted: dict[str, str]) -> dict[str, Any]:
    mentions = sorted(set(gold) & set(predicted))
    if not mentions:
        return {"mentions": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    gold_groups: dict[str, set[str]] = defaultdict(set)
    predicted_groups: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        gold_groups[gold[mention]].add(mention)
        predicted_groups[predicted[mention]].add(mention)
    precision = sum(
        len(gold_groups[gold[m]] & predicted_groups[predicted[m]]) / len(predicted_groups[predicted[m]])
        for m in mentions
    ) / len(mentions)
    recall = sum(
        len(gold_groups[gold[m]] & predicted_groups[predicted[m]]) / len(gold_groups[gold[m]])
        for m in mentions
    ) / len(mentions)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"mentions": len(mentions), "precision": precision, "recall": recall, "f1": f1}


def candidate_recall(gold_links: Iterable[tuple[str, str]], candidates: Iterable[tuple[str, str]]) -> dict[str, Any]:
    gold = {tuple(sorted(pair)) for pair in gold_links}
    recalled = {tuple(sorted(pair)) for pair in candidates}
    return {"gold": len(gold), "candidates": len(recalled),
            "recall": len(gold & recalled) / len(gold) if gold else 1.0}


def _relation_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("source"), item.get("predicate"), item.get("target"),
        item.get("direction") or "forward", int(item.get("evidence_start") or 0),
        int(item.get("evidence_end") or 0),
    )


def evaluate_relations(gold: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    gold_keys, predicted_keys = set(map(_relation_key, gold)), set(map(_relation_key, predicted))
    precise_evidence = sum(
        item.get("evidence_start") is not None and item.get("evidence_end") is not None
        for item in predicted
    )
    return {
        "strict": _prf(len(gold_keys & predicted_keys), len(predicted_keys), len(gold_keys)),
        "evidence_coverage": precise_evidence / len(predicted) if predicted else 1.0,
    }


def calibration_metrics(labels: list[int], probabilities: list[float], bins: int = 10) -> dict[str, Any]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be aligned")
    if not labels:
        return {"count": 0, "brier": 0.0, "ece": 0.0, "bins": []}
    values = [min(1.0, max(0.0, float(value))) for value in probabilities]
    brier = sum((probability - int(label)) ** 2 for label, probability in zip(labels, values)) / len(labels)
    reliability = []
    ece = 0.0
    for index in range(max(1, bins)):
        lower, upper = index / bins, (index + 1) / bins
        selected = [offset for offset, value in enumerate(values)
                    if lower <= value < upper or (index == bins - 1 and value == 1.0)]
        if not selected:
            continue
        confidence = sum(values[offset] for offset in selected) / len(selected)
        accuracy = sum(int(labels[offset]) for offset in selected) / len(selected)
        ece += len(selected) / len(labels) * abs(confidence - accuracy)
        reliability.append({"lower": lower, "upper": upper, "count": len(selected),
                            "confidence": confidence, "accuracy": accuracy})
    return {"count": len(labels), "brier": brier, "ece": ece, "bins": reliability}


def paired_discordance(control: list[bool], treatment: list[bool]) -> dict[str, int]:
    if len(control) != len(treatment):
        raise ValueError("paired outcomes must be aligned")
    return {
        "both_correct": sum(left and right for left, right in zip(control, treatment)),
        "control_only": sum(left and not right for left, right in zip(control, treatment)),
        "treatment_only": sum(not left and right for left, right in zip(control, treatment)),
        "both_wrong": sum(not left and not right for left, right in zip(control, treatment)),
        "count": len(control),
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evaluate_release_gate(
    *, control: dict[str, float], treatment: dict[str, float], manifest: dict[str, Any],
) -> dict[str, Any]:
    failed: list[str] = []
    if not manifest.get("production_valid"):
        failed.append("human_gold")
    if treatment.get("critical_errors", 0) > 0:
        failed.append("critical_errors")
    if treatment.get("candidate_recall", 0) < 0.99:
        failed.append("candidate_recall")
    for metric, tolerance in (("entity_f1", 0.005), ("b3_f1", 0.005), ("relation_f1", 0.01)):
        if treatment.get(metric, 0) + tolerance < control.get(metric, 0):
            failed.append(f"{metric}_non_inferiority")
    benefit = bool(
        treatment.get("relation_f1", 0) > control.get("relation_f1", 0)
        or treatment.get("b3_f1", 0) > control.get("b3_f1", 0)
        or treatment.get("cost", math.inf) < control.get("cost", math.inf)
    )
    if not benefit:
        failed.append("material_benefit")
    approved = not failed
    return {
        "status": "production_approved" if approved else "engineering_complete",
        "production_approved": approved, "failed_gates": failed,
        "manifest_valid": bool(manifest.get("production_valid")),
    }
