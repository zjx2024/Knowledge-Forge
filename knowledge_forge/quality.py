"""Deterministic P3 quality metrics and release-gate validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


class GoldManifestError(ValueError):
    """The annotation manifest cannot be used for the requested gate."""


RELEASE_GOLD_MINIMUMS = {
    "items": 200,
    "domains": 2,
    "entity_mentions": 500,
    "must_links": 150,
    "cannot_links": 150,
    "positive_relations": 200,
    "hard_negatives": 300,
}


def canonical_json(value: Any) -> str:
    """Serialize an immutable evaluation fact identically across processes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _first_value(item: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        value = item.get(name)
        if value is not None and value != "":
            return value
    return default


def canonical_relation_fact_key(item: dict[str, Any]) -> tuple[str, ...]:
    """Return a cross-run semantic identity without provenance or internal IDs."""

    project_scope = _first_value(
        item, "gold_universe", "project_scope", "project_key", "project_id",
    )
    subject = _first_value(
        item, "subject_canonical_id", "source_canonical_id", "subject", "source",
    )
    predicate = _first_value(item, "canonical_predicate", "predicate", "relation_type")
    object_ = _first_value(
        item, "object_canonical_id", "target_canonical_id", "object", "target",
    )
    direction = str(_first_value(item, "canonical_direction", "direction", default="forward"))
    if direction.lower() in {"reversed", "reverse"}:
        subject, object_ = object_, subject
        direction = "forward"
    elif direction.lower() == "symmetric":
        subject, object_ = sorted((str(subject), str(object_)))
        direction = "symmetric"
    return tuple(map(str, (project_scope, subject, predicate, object_, direction)))


def unordered_relation_endpoint_predicate_key(item: dict[str, Any]) -> tuple[str, ...]:
    """Return an unordered endpoint+predicate identity for pair-recall metrics."""

    project_scope = _first_value(
        item, "gold_universe", "project_scope", "project_key", "project_id",
    )
    subject = _first_value(
        item, "subject_canonical_id", "source_canonical_id", "subject", "source",
    )
    predicate = _first_value(item, "canonical_predicate", "predicate", "relation_type")
    object_ = _first_value(
        item, "object_canonical_id", "target_canonical_id", "object", "target",
    )
    left, right = sorted((str(subject), str(object_)))
    return tuple(map(str, (project_scope, left, right, predicate)))


def occurrence_relation_assertion_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Return a strict occurrence/evidence identity layered over the semantic fact."""

    return (
        *canonical_relation_fact_key(item),
        _first_value(item, "gold_document_key", "document_id"),
        _first_value(
            item, "document_revision_id", "revision_id", "content_hash",
        ),
        _first_value(item, "chunk_id"),
        _first_value(item, "subject_occurrence_id", "subject_mention_id", "source_mention_id"),
        _first_value(item, "subject_start", "source_start", "source_mention_start"),
        _first_value(item, "subject_end", "source_end", "source_mention_end"),
        _first_value(item, "object_occurrence_id", "object_mention_id", "target_mention_id"),
        _first_value(item, "object_start", "target_start", "target_mention_start"),
        _first_value(item, "object_end", "target_end", "target_mention_end"),
        _first_value(item, "gold_evidence_id", "evidence_start"),
        _first_value(item, "evidence_end"),
        _first_value(item, "evidence_hash", "evidence_text", "evidence"),
    )


def _multiset_relation_metrics(
    gold: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    *,
    key: Any,
) -> dict[str, Any]:
    gold_counts = Counter(key(item) for item in gold)
    predicted_counts = Counter(key(item) for item in predicted)
    true_positive = sum(
        min(count, predicted_counts.get(identity, 0))
        for identity, count in gold_counts.items()
    )
    predicted_total = sum(predicted_counts.values())
    gold_total = sum(gold_counts.values())
    precision = true_positive / predicted_total if predicted_total else float(not gold_total)
    recall = true_positive / gold_total if gold_total else float(not predicted_total)
    return {
        "true_positive": true_positive,
        "false_positive": predicted_total - true_positive,
        "false_negative": gold_total - true_positive,
        "predicted": predicted_total,
        "gold": gold_total,
        "unique_predicted": len(predicted_counts),
        "unique_gold": len(gold_counts),
        "exact_duplicates": sum(max(0, count - 1) for count in predicted_counts.values()),
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def evaluate_relation_quality(
    *,
    gold: list[dict[str, Any]] | None,
    predicted: list[dict[str, Any]],
    snapshot_current: bool,
) -> dict[str, Any]:
    """Evaluate relation output without treating output volume as Gold quality."""

    blocking_reasons = [] if snapshot_current else ["stale_snapshot"]
    if gold is None:
        blocking_reasons.append("human_gold_unavailable")
        unavailable = {
            "true_positive": None,
            "false_positive": None,
            "false_negative": None,
            "predicted": len(predicted),
            "gold": None,
            "unique_predicted": len({
                canonical_relation_fact_key(item) for item in predicted
            }),
            "unique_gold": None,
            "exact_duplicates": sum(
                max(0, count - 1)
                for count in Counter(
                    canonical_relation_fact_key(item) for item in predicted
                ).values()
            ),
            "precision": None,
            "recall": None,
            "f1": None,
        }
        return {
            "gold_status": "unavailable",
            "canonical": unavailable,
            "occurrence": {
                **unavailable,
                "unique_predicted": len({
                    occurrence_relation_assertion_key(item) for item in predicted
                }),
            },
            "production_eligible": False,
            "blocking_reasons": sorted(set(blocking_reasons)),
        }
    return {
        "gold_status": "available",
        "canonical": _multiset_relation_metrics(
            gold, predicted, key=canonical_relation_fact_key,
        ),
        "occurrence": _multiset_relation_metrics(
            gold, predicted, key=occurrence_relation_assertion_key,
        ),
        "production_eligible": not blocking_reasons,
        "blocking_reasons": sorted(set(blocking_reasons)),
    }


def relation_layer_funnel(
    *,
    pair_opportunities: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    verification_attempts: list[dict[str, Any]],
    normalized_facts: list[dict[str, Any]],
    published_facts: list[dict[str, Any]],
) -> dict[str, int]:
    """Report distinct pair, hypothesis, verifier, fact and publication layers."""

    pair_statuses = Counter(
        str(item.get("terminal_status") or item.get("status") or "").upper()
        for item in pair_opportunities
    )
    verdicts = Counter(
        str(item.get("verdict") or item.get("status") or "").upper()
        for item in verification_attempts
    )
    return {
        "pair_opportunities_total": len(pair_opportunities),
        "pair_opportunities_classified": sum(
            pair_statuses[value]
            for value in (
                "HYPOTHESES",
                "HAS_RELATION",
                "NO_RELATION",
                "OTHER_UNSUPPORTED",
                "CLASSIFIED",
            )
        ),
        "pair_no_relation": pair_statuses["NO_RELATION"],
        "pair_other_unsupported": pair_statuses["OTHER_UNSUPPORTED"],
        "pair_deferred_budget": pair_statuses["DEFERRED_BUDGET"],
        "pair_failed": sum(
            pair_statuses[value]
            for value in (
                "FAILED",
                "CLASSIFICATION_RETRYABLE_FAILED",
                "CLASSIFICATION_PERMANENT_FAILED",
            )
        ),
        "hypotheses_total": len(hypotheses),
        "hypotheses_rule_valid": sum(
            bool(item.get("rule_valid")) for item in hypotheses
        ),
        "hypotheses_verifier_supported": verdicts["SUPPORTED"],
        "hypotheses_verifier_rejected": verdicts["REJECTED"] + verdicts["CONTRADICTED"],
        "hypotheses_verifier_uncertain": (
            verdicts["UNCERTAIN"] + verdicts["INSUFFICIENT_EVIDENCE"]
        ),
        "hypotheses_retryable_failed": (
            verdicts["RETRYABLE_FAILED"] + verdicts["FAILED"]
        ),
        "facts_after_normalization": len(normalized_facts),
        "facts_duplicate_collapsed": sum(
            int(item.get("duplicate_collapsed") or 0) for item in normalized_facts
        ),
        "facts_review_only": sum(
            bool(item.get("review_only")) for item in normalized_facts
        ),
        "facts_materialization_eligible": sum(
            bool(item.get("materialization_eligible")) for item in normalized_facts
        ),
        "facts_published": len(published_facts),
    }


def validate_relation_gold(
    manifest: dict[str, Any],
    pair_decisions: list[dict[str, Any]],
    fact_positives: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate frozen human relation Gold with separate pair and fact labels."""

    for field in (
        "dataset", "version", "frozen_at", "holdout_content_hash",
        "precision_adjudication_universe",
    ):
        if not manifest.get(field):
            raise GoldManifestError(f"relation Gold manifest missing {field}")
    if len(str(manifest["holdout_content_hash"])) != 64:
        raise GoldManifestError("relation Gold holdout hash is invalid")
    if not pair_decisions:
        raise GoldManifestError("relation Gold requires pair decisions")
    pair_map: dict[str, dict[str, Any]] = {}
    human_valid = True
    for item in pair_decisions:
        pair_id = str(item.get("pair_id") or "")
        if not pair_id or pair_id in pair_map:
            raise GoldManifestError("relation Gold pair ids are missing or duplicated")
        if str(item.get("dataset") or "") != str(manifest["dataset"]) or str(
            item.get("dataset_version") or ""
        ) != str(manifest["version"]):
            raise GoldManifestError("relation Gold dataset/version mismatch")
        if item.get("split") not in {"dev", "holdout"}:
            raise GoldManifestError("relation Gold split must be dev or holdout")
        decision = str(item.get("decision") or "")
        if decision not in {"NO_RELATION", "OTHER_UNSUPPORTED", "HAS_RELATION"}:
            raise GoldManifestError("relation Gold pair decision is invalid")
        for side in ("left", "right"):
            span = item.get(f"{side}_span")
            if (
                not item.get(f"{side}_occurrence_id")
                or not item.get(f"{side}_canonical_id")
                or not isinstance(span, list)
                or len(span) != 2
                or int(span[0]) < 0
                or int(span[1]) <= int(span[0])
            ):
                raise GoldManifestError("relation Gold occurrence/span is invalid")
        human = str(item.get("label_source") or "").lower() in {
            "human", "human_reviewed", "independent_human",
        }
        adjudicated = (
            str(item.get("adjudication_status") or "") == "adjudicated"
            and bool(item.get("annotator"))
            and bool(item.get("reviewer"))
            and bool(item.get("reviewed_at"))
        )
        human_valid = human_valid and human and adjudicated
        if not human or not adjudicated:
            raise GoldManifestError(
                "relation Gold requires independent human adjudication"
            )
        pair_map[pair_id] = item

    facts_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in fact_positives:
        pair_id = str(fact.get("pair_id") or "")
        if pair_id not in pair_map:
            raise GoldManifestError("relation fact Gold references unknown pair")
        if pair_map[pair_id]["decision"] != "HAS_RELATION":
            raise GoldManifestError(
                f"{pair_map[pair_id]['decision']} pair cannot contain positive fact"
            )
        if not all(
            fact.get(field) not in {None, ""}
            for field in (
                "subject_occurrence_id", "subject_canonical_id", "predicate",
                "object_occurrence_id", "object_canonical_id",
                "evidence_start", "evidence_end", "adjudication_status",
            )
        ):
            raise GoldManifestError("relation fact Gold is incomplete")
        if (
            int(fact["evidence_start"]) < 0
            or int(fact["evidence_end"]) <= int(fact["evidence_start"])
            or fact.get("adjudication_status") != "adjudicated"
        ):
            raise GoldManifestError("relation fact Gold evidence/adjudication is invalid")
        facts_by_pair[pair_id].append(fact)
    for pair_id, pair in pair_map.items():
        facts = facts_by_pair.get(pair_id, [])
        if pair["decision"] == "HAS_RELATION" and not facts:
            raise GoldManifestError("HAS_RELATION pair requires positive fact Gold")
        if pair["decision"] == "NO_RELATION" and facts:
            raise GoldManifestError("NO_RELATION pair cannot contain positive fact Gold")
        if pair["decision"] == "OTHER_UNSUPPORTED" and facts:
            raise GoldManifestError(
                "OTHER_UNSUPPORTED pair cannot contain current-Schema fact Gold"
            )
    approval = manifest.get("trusted_approval") or {}
    production_valid = bool(
        manifest.get("level") == "release"
        and manifest.get("recall_universe_complete")
        and human_valid
        and approval.get("approval_id")
        and approval.get("approved_at")
        and len(set(approval.get("signers") or [])) >= 2
    )
    return {
        "engineering_valid": True,
        "production_valid": production_valid,
        "dataset": str(manifest["dataset"]),
        "version": str(manifest["version"]),
        "pair_gold_hash": canonical_digest(sorted(
            pair_decisions, key=lambda item: str(item["pair_id"]),
        )),
        "fact_gold_hash": canonical_digest(sorted(
            fact_positives,
            key=lambda item: (
                str(item["pair_id"]),
                canonical_relation_fact_key(item),
                occurrence_relation_assertion_key(item),
            ),
        )),
        "pairs": len(pair_decisions),
        "positive_facts": len(fact_positives),
        "recall_universe_complete": bool(
            manifest.get("recall_universe_complete")
        ),
        "precision_adjudication_universe": str(
            manifest["precision_adjudication_universe"]
        ),
    }


def _unavailable_relation_gold_report(predicted: int) -> dict[str, Any]:
    unavailable = {
        "true_positive": None,
        "false_positive": None,
        "false_negative": None,
        "predicted": predicted,
        "gold": None,
        "unique_predicted": None,
        "unique_gold": None,
        "exact_duplicates": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    return {
        "gold_status": "unavailable",
        "precision_status": "unavailable",
        "strict": unavailable,
        "canonical": dict(unavailable),
        "endpoint_recall": None,
        "pair_coverage": None,
        "hypothesis_recall": None,
        "verifier_recall": None,
        "production_eligible": False,
        "blocking_reasons": ["human_gold_unavailable"],
    }


def _decision_prf(
    gold_by_pair: Mapping[str, str],
    predicted_by_pair: Mapping[str, str],
    decision: str,
) -> dict[str, Any]:
    gold_ids = {
        pair_id for pair_id, value in gold_by_pair.items() if value == decision
    }
    predicted_ids = {
        pair_id for pair_id, value in predicted_by_pair.items() if value == decision
    }
    return _prf(
        len(gold_ids & predicted_ids),
        len(predicted_ids),
        len(gold_ids),
    )


def evaluate_relation_gold_run(
    *,
    manifest_report: dict[str, Any] | None,
    pair_gold: list[dict[str, Any]] | None,
    fact_gold: list[dict[str, Any]] | None,
    run: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate endpoint→pair→hypothesis→verifier→fact layers on frozen Gold."""

    predictions = list(run.get("facts") or [])
    if manifest_report is None or pair_gold is None or fact_gold is None:
        return _unavailable_relation_gold_report(len(predictions))
    if not manifest_report.get("engineering_valid"):
        raise GoldManifestError("relation Gold manifest is not engineering-valid")
    gold_facts = list(fact_gold)
    strict_raw = _multiset_relation_metrics(
        gold_facts, predictions, key=occurrence_relation_assertion_key,
    )
    canonical_raw = _multiset_relation_metrics(
        gold_facts, predictions, key=canonical_relation_fact_key,
    )
    prediction_ids = [str(item.get("prediction_id") or "") for item in predictions]
    adjudications = {
        str(item.get("prediction_id") or ""): bool(item.get("correct"))
        for item in run.get("prediction_adjudications") or []
        if item.get("prediction_id")
    }
    unadjudicated = [
        prediction_id for prediction_id in prediction_ids
        if not prediction_id or prediction_id not in adjudications
    ]
    precision_available = not unadjudicated

    def final_metrics(raw: dict[str, Any]) -> dict[str, Any]:
        result = dict(raw)
        if not precision_available:
            result["precision"] = None
            result["f1"] = None
            return result
        predicted_total = int(raw["predicted"])
        true_positive = int(raw["true_positive"])
        precision = (
            true_positive / predicted_total
            if predicted_total else float(not raw["gold"])
        )
        result["precision"] = precision
        result["f1"] = (
            2 * precision * float(raw["recall"])
            / (precision + float(raw["recall"]))
            if precision + float(raw["recall"]) else 0.0
        )
        return result

    strict = final_metrics(strict_raw)
    canonical = final_metrics(canonical_raw)
    direction_denominator = sum(
        min(count, Counter(
            unordered_relation_endpoint_predicate_key(item)
            for item in predictions
        ).get(identity, 0))
        for identity, count in Counter(
            unordered_relation_endpoint_predicate_key(item)
            for item in gold_facts
        ).items()
    )
    direction_true_positive = canonical_raw["true_positive"]
    positive_pair_ids = {str(item["pair_id"]) for item in gold_facts}
    gold_occurrences = {
        str(item[field])
        for item in gold_facts
        for field in ("subject_occurrence_id", "object_occurrence_id")
    }
    predicted_occurrences = {
        str(
            item.get("occurrence_key")
            or item.get("mention_key")
            or item.get("mention_id")
            or ""
        )
        for item in run.get("mention_occurrences") or []
    }
    pair_ids = {
        str(item.get("pair_identity") or item.get("pair_id") or "")
        for item in run.get("pair_opportunities") or []
    }
    gold_hypotheses = {
        (
            str(item["pair_id"]),
            str(item["subject_occurrence_id"]),
            str(item["predicate"]),
            str(item["object_occurrence_id"]),
        )
        for item in gold_facts
    }
    predicted_hypothesis_values = {
        (
            str(item.get("pair_identity") or item.get("pair_id") or ""),
            str(item.get("subject_occurrence_key")
                or item.get("subject_occurrence_id") or ""),
            str(item.get("predicate") or ""),
            str(item.get("object_occurrence_key")
                or item.get("object_occurrence_id") or ""),
        ): str(item.get("hypothesis_id") or "")
        for item in run.get("hypotheses") or []
    }
    supported_hypothesis_ids = {
        str(item.get("hypothesis_id") or "")
        for item in run.get("verification_attempts") or []
        if str(item.get("verdict") or "").upper() == "SUPPORTED"
    }
    predicted_pair_decisions = {
        str(item.get("pair_identity") or item.get("pair_id") or ""): (
            "HAS_RELATION"
            if str(item.get("decision") or "") == "HYPOTHESES"
            else str(item.get("decision") or "")
        )
        for item in run.get("pair_decisions") or []
    }
    gold_pair_decisions = {
        str(item["pair_id"]): str(item["decision"]) for item in pair_gold
    }
    predicates = sorted({
        str(item.get("predicate") or "") for item in gold_facts + predictions
        if item.get("predicate")
    })
    per_predicate = {}
    for predicate in predicates:
        per_predicate[predicate] = _multiset_relation_metrics(
            [item for item in gold_facts if item.get("predicate") == predicate],
            [item for item in predictions if item.get("predicate") == predicate],
            key=canonical_relation_fact_key,
        )
    related_classified = sum(
        str(item.get("predicate") or "") == "related_to"
        for item in run.get("hypotheses") or []
    )
    related_review = sum(
        str(item.get("predicate") or "") == "related_to"
        for item in run.get("review_facts") or []
    )
    related_published = sum(
        str(item.get("predicate") or "") == "related_to"
        for item in (run.get("published_facts") or predictions)
    )
    invalid_evidence = sum(
        item.get("evidence_start") is None
        or item.get("evidence_end") is None
        or not (item.get("evidence_text") or item.get("evidence"))
        for item in predictions
    )
    normalization = dict(run.get("normalization_metrics") or {})
    blocking_reasons = []
    if unadjudicated:
        blocking_reasons.append("unadjudicated_predictions")
    if run.get("stale"):
        blocking_reasons.append("stale_snapshot")
    endpoint_matches = len(gold_occurrences & predicted_occurrences)
    hypothesis_matches = gold_hypotheses & set(predicted_hypothesis_values)
    verifier_matches = sum(
        bool(predicted_hypothesis_values[value] in supported_hypothesis_ids)
        for value in hypothesis_matches
    )
    return {
        "gold_status": "available",
        "precision_status": "available" if precision_available else "unavailable",
        "strict": strict,
        "canonical": canonical,
        "endpoint_recall": (
            endpoint_matches / len(gold_occurrences)
            if gold_occurrences else 1.0
        ),
        "pair_coverage": (
            len(positive_pair_ids & pair_ids) / len(positive_pair_ids)
            if positive_pair_ids else 1.0
        ),
        "hypothesis_recall": (
            len(hypothesis_matches) / len(gold_hypotheses)
            if gold_hypotheses else 1.0
        ),
        "verifier_recall": (
            verifier_matches / len(gold_hypotheses)
            if gold_hypotheses else 1.0
        ),
        "predicate_metrics": per_predicate,
        "direction_accuracy": (
            direction_true_positive / direction_denominator
            if direction_denominator else float(not gold_facts)
        ),
        "exact_evidence_coverage": strict_raw["recall"],
        "pair_decisions": {
            decision: _decision_prf(
                gold_pair_decisions, predicted_pair_decisions, decision,
            )
            for decision in ("NO_RELATION", "OTHER_UNSUPPORTED", "HAS_RELATION")
        },
        "related_to": {
            "classified": related_classified,
            "review": related_review,
            "published": related_published,
        },
        "normalization": normalization,
        "safety": {
            "stale_published": int(bool(run.get("stale") and predictions)),
            "unverified_published": int(run.get("unverified_published") or 0),
            "invalid_evidence_published": invalid_evidence,
            "related_to_published": related_published,
            "other_unsupported_published": sum(
                str(item.get("predicate") or "") == "OTHER_UNSUPPORTED"
                for item in predictions
            ),
            "cannot_link_violations": int(
                run.get("cannot_link_violations") or 0
            ),
            "all_pairs_terminal": bool(run.get("all_pairs_terminal")),
        },
        "cost": dict(run.get("cost") or {}),
        "production_eligible": bool(
            manifest_report.get("production_valid")
            and precision_available
            and not blocking_reasons
        ),
        "blocking_reasons": blocking_reasons,
        "denominators": {
            "gold_pairs": len(pair_gold),
            "gold_positive_facts": len(gold_facts),
            "predicted_facts": len(predictions),
            "unadjudicated_predictions": len(unadjudicated),
            "direction_matched_endpoint_predicate": direction_denominator,
        },
    }


def evaluate_relation_release_gate_v2(
    *,
    control: dict[str, Any],
    treatment: dict[str, Any],
    manifest_report: dict[str, Any],
    release_policy: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed on safety, precision and actual Gold-recall improvement."""

    failed: set[str] = set()
    required_policy = {
        "version",
        "relation_precision_noninferiority_delta",
        "minimum_gold_facts",
        "confidence_method",
    }
    if any(
        key not in release_policy or release_policy.get(key) in {None, ""}
        for key in required_policy
    ):
        failed.add("release_policy_incomplete")
    if not manifest_report.get("production_valid"):
        failed.add("human_gold")
    minimum = int(release_policy.get("minimum_gold_facts") or 0)
    if int(treatment.get("denominators", {}).get("gold_positive_facts") or 0) < minimum:
        failed.add("minimum_gold_facts")
    if treatment.get("strict", {}).get("precision") is None:
        failed.add("strict_precision_unavailable")
    else:
        delta = float(
            release_policy.get("relation_precision_noninferiority_delta") or 0
        )
        control_precision = control.get("strict", {}).get("precision")
        if (
            control_precision is None
            or float(treatment["strict"]["precision"]) + delta
            < float(control_precision)
        ):
            failed.add("strict_precision_noninferiority")
    hard_safety = {
        "stale_published": "stale_published",
        "unverified_published": "unverified_published",
        "invalid_evidence_published": "invalid_evidence_published",
        "related_to_published": "related_to_published",
        "other_unsupported_published": "other_unsupported_published",
        "cannot_link_violations": "cannot_link_violations",
    }
    safety = treatment.get("safety") or {}
    for field, gate in hard_safety.items():
        if int(safety.get(field) or 0):
            failed.add(gate)
    if not safety.get("all_pairs_terminal"):
        failed.add("pair_ledger_incomplete")
    normalization = treatment.get("normalization") or {}
    for field, gate in (
        ("exact_duplicates_remaining", "exact_duplicates"),
        (
            "symmetric_inverse_duplicates_remaining",
            "symmetric_inverse_duplicates",
        ),
        ("unsupported_reverse_conflicts", "unsupported_reverse_conflicts"),
        ("illegal_self_loops", "illegal_self_loops"),
    ):
        if int(normalization.get(field) or 0):
            failed.add(gate)
    if float(treatment.get("pair_coverage") or 0) < float(
        control.get("pair_coverage") or 0
    ):
        failed.add("pair_coverage_regression")
    if int(treatment.get("strict", {}).get("true_positive") or 0) < int(
        control.get("strict", {}).get("true_positive") or 0
    ):
        failed.add("baseline_gold_fact_retention")
    control_recall = control.get("strict", {}).get("recall")
    treatment_recall = treatment.get("strict", {}).get("recall")
    goal_met = bool(
        control_recall is not None
        and treatment_recall is not None
        and float(treatment_recall) > float(control_recall)
    )
    if not goal_met:
        failed.add("gold_recall_not_improved")
    engineering_failures = {
        "stale_published",
        "unverified_published",
        "invalid_evidence_published",
        "related_to_published",
        "other_unsupported_published",
        "cannot_link_violations",
        "pair_ledger_incomplete",
        "exact_duplicates",
        "symmetric_inverse_duplicates",
        "unsupported_reverse_conflicts",
        "illegal_self_loops",
    }
    engineering_complete = not bool(failed & engineering_failures)
    production_approved = not failed
    return {
        "status": (
            "production_approved"
            if production_approved else "engineering_complete"
        ),
        "engineering_complete": engineering_complete,
        "goal_met": goal_met,
        "production_approved": production_approved,
        "failed_gates": sorted(failed),
        "release_policy_version": str(release_policy.get("version") or ""),
    }


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
    level = str(manifest.get("level") or ("pilot" if manifest.get("pilot", True) else "release"))
    if level not in {"pilot", "release"}:
        raise GoldManifestError("manifest level must be pilot or release")
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    seen_documents: dict[str, str] = {}
    seen_chunks: dict[tuple[str, str, str], str] = {}
    production_reasons: list[str] = []
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
        document_id = str(item.get("document_id") or "")
        revision_id = str(item.get("revision_id") or "")
        chunk_id = str(item.get("chunk_id") or "")
        if document_id and document_id in seen_documents and seen_documents[document_id] != split:
            raise GoldManifestError("dev/holdout overlap detected by document id")
        chunk_key = (document_id, revision_id, chunk_id)
        if all(chunk_key) and chunk_key in seen_chunks and seen_chunks[chunk_key] != split:
            raise GoldManifestError("dev/holdout overlap detected by chunk lineage")
        seen_ids[item_id], seen_hashes[digest] = split, split
        if document_id:
            seen_documents[document_id] = split
        if all(chunk_key):
            seen_chunks[chunk_key] = split
        if root and item.get("path"):
            path = root / str(item["path"])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise GoldManifestError(f"content hash drift: {item_id}")
        human = str(item.get("label_source") or "").lower() in {
            "human", "human_reviewed", "independent_human",
        }
        reviewers = item.get("reviewers") or ([item["reviewer"]] if item.get("reviewer") else [])
        reviewers = {str(value).strip() for value in reviewers if str(value).strip()}
        adjudicated = bool(item.get("adjudication") and item.get("adjudicated_at"))
        if not human:
            production_reasons.append(f"{item_id}:non_human_label")
        if len(reviewers) < 2 and not adjudicated:
            production_reasons.append(f"{item_id}:independent_review")
        if not item.get("reviewed_at"):
            production_reasons.append(f"{item_id}:reviewed_at")
    holdout_items = sum(item.get("split") == "holdout" for item in items)
    if not holdout_items:
        production_reasons.append("empty_holdout")
    for field in ("schema_compatibility", "policy_compatibility", "calibration_universe_version"):
        if not manifest.get(field):
            production_reasons.append(f"missing_{field}")
    approval = manifest.get("trusted_approval") or {}
    if (
        str(approval.get("method") or "") not in {"trusted_signature", "approval_service"}
        or not approval.get("approval_id") or not approval.get("approved_at")
        or len(set(approval.get("signers") or [])) < 2
    ):
        production_reasons.append("trusted_approval")
    label_counts = manifest.get("label_counts") or {}
    observed = {
        "items": len(items),
        "domains": len({str(item.get("domain") or "") for item in items if item.get("domain")}),
        **{key: int(label_counts.get(key) or 0) for key in RELEASE_GOLD_MINIMUMS if key not in {"items", "domains"}},
    }
    for key, minimum in RELEASE_GOLD_MINIMUMS.items():
        if observed[key] < minimum:
            production_reasons.append(f"minimum_{key}")
    production_valid = level == "release" and not production_reasons
    result = {
        "engineering_valid": True,
        "production_valid": bool(production_valid),
        "level": level,
        "production_reasons": sorted(set(production_reasons)),
        "manifest_hash": canonical_digest(manifest),
        "items": len(items),
        "dev_items": sum(item.get("split") == "dev" for item in items),
        "holdout_items": holdout_items,
        "observed_minimums": observed,
    }
    if require_production and not result["production_valid"]:
        raise GoldManifestError("production gate requires independently reviewed human gold")
    return result


def validate_gold_annotations(
    manifest: dict[str, Any], annotations: list[dict[str, Any]],
    manifest_report: dict[str, Any] | None = None, *, require_production: bool = False,
) -> dict[str, Any]:
    """Validate release minimums from frozen labels instead of caller claims."""

    report = dict(manifest_report or validate_gold_manifest(manifest))
    manifest_items = {str(item["id"]): item for item in manifest.get("items") or []}
    annotation_items = {str(item.get("item_id") or ""): item for item in annotations}
    if set(manifest_items) != set(annotation_items) or "" in annotation_items:
        raise GoldManifestError("Gold manifest and annotation universes differ")

    observed = {
        "items": len(annotations),
        "domains": len({
            str(item.get("domain") or "") for item in manifest_items.values()
            if item.get("domain")
        }),
        "entity_mentions": sum(len(item.get("mentions") or []) for item in annotations),
        "must_links": sum(len(item.get("must_links") or []) for item in annotations),
        "cannot_links": sum(len(item.get("cannot_links") or []) for item in annotations),
        "positive_relations": sum(len(item.get("relations") or []) for item in annotations),
        "hard_negatives": sum(
            len(item.get("hard_negatives") or [])
            + sum(int(value.get("label") or 0) == 0 for value in item.get("calibration") or [])
            for item in annotations
        ),
    }
    reasons = set(report.get("production_reasons") or [])
    declared = manifest.get("label_counts") or {}
    if report.get("level") == "release" and not declared:
        reasons.add("missing_label_counts")
    for key in RELEASE_GOLD_MINIMUMS:
        if key in {"items", "domains"}:
            continue
        if key in declared and int(declared.get(key) or 0) != observed[key]:
            reasons.add(f"label_count_mismatch_{key}")
    for key, minimum in RELEASE_GOLD_MINIMUMS.items():
        if observed[key] < minimum:
            reasons.add(f"minimum_{key}")

    for item_id, annotation in annotation_items.items():
        source = str(
            annotation.get("label_source")
            or manifest_items[item_id].get("label_source") or ""
        ).lower()
        if source not in {"human", "human_reviewed", "independent_human"}:
            reasons.add(f"{item_id}:non_human_annotation")

    report.update({
        "observed_minimums": observed,
        "production_reasons": sorted(reasons),
        "production_valid": report.get("level") == "release" and not reasons,
        "annotations_hash": canonical_digest(
            sorted(annotations, key=lambda item: str(item.get("item_id") or ""))
        ),
    })
    if require_production and not report["production_valid"]:
        raise GoldManifestError("production gate requires complete independently reviewed Gold labels")
    return report


def _span_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item.get("document_id"), item.get("revision_id") or item.get("content_hash"),
        item.get("chunk_id"), int(item["start"]), int(item["end"]),
    )


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("document_id") == right.get("document_id")
        and (left.get("revision_id") or left.get("content_hash"))
        == (right.get("revision_id") or right.get("content_hash"))
        and left.get("chunk_id") == right.get("chunk_id")
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
    """Detection-aware pairwise score over the augmented mention universe."""

    gold_mentions, predicted_mentions = set(gold), set(predicted)
    universe = gold_mentions | predicted_mentions
    augmented_gold = {
        mention: gold.get(mention, f"__spurious_gold__:{mention}") for mention in universe
    }
    augmented_predicted = {
        mention: predicted.get(mention, f"__missing_prediction__:{mention}") for mention in universe
    }
    gold_pairs = _positive_pairs(augmented_gold)
    predicted_pairs = _positive_pairs(augmented_predicted)
    missed = len(gold_mentions - predicted_mentions)
    spurious = len(predicted_mentions - gold_mentions)
    report = _prf(
        len(gold_pairs & predicted_pairs), len(predicted_pairs) + spurious,
        len(gold_pairs) + missed,
    )
    matched = gold_mentions & predicted_mentions
    matched_gold = _positive_pairs({key: gold[key] for key in matched})
    matched_predicted = _positive_pairs({key: predicted[key] for key in matched})
    report.update({
        "universe_mentions": len(universe), "matched_mentions": len(matched),
        "missed_mentions": missed, "spurious_mentions": spurious,
        "matched_only": _prf(
            len(matched_gold & matched_predicted), len(matched_predicted), len(matched_gold),
        ),
    })
    return report


def b3_metrics(gold: dict[str, str], predicted: dict[str, str]) -> dict[str, Any]:
    gold_mentions, predicted_mentions = set(gold), set(predicted)
    mentions = sorted(gold_mentions | predicted_mentions)
    if not mentions:
        return {"mentions": 0, "universe_mentions": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    gold_groups: dict[str, set[str]] = defaultdict(set)
    predicted_groups: dict[str, set[str]] = defaultdict(set)
    for mention in mentions:
        gold_groups[gold.get(mention, f"__spurious_gold__:{mention}")].add(mention)
        predicted_groups[predicted.get(mention, f"__missing_prediction__:{mention}")].add(mention)
    precision = sum(
        0.0 if m not in gold_mentions else
        len(gold_groups[gold[m]] & predicted_groups[predicted.get(m, f"__missing_prediction__:{m}")])
        / len(predicted_groups[predicted.get(m, f"__missing_prediction__:{m}")])
        for m in mentions
    ) / len(mentions)
    recall = sum(
        0.0 if m not in predicted_mentions else
        len(gold_groups[gold.get(m, f"__spurious_gold__:{m}")] & predicted_groups[predicted[m]])
        / len(gold_groups[gold.get(m, f"__spurious_gold__:{m}")])
        for m in mentions
    ) / len(mentions)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = gold_mentions & predicted_mentions
    matched_only = b3_metrics(
        {key: gold[key] for key in matched}, {key: predicted[key] for key in matched},
    ) if matched != set(mentions) else None
    return {
        "mentions": len(mentions), "universe_mentions": len(mentions),
        "matched_mentions": len(matched), "missed_mentions": len(gold_mentions - predicted_mentions),
        "spurious_mentions": len(predicted_mentions - gold_mentions),
        "precision": precision, "recall": recall, "f1": f1,
        "matched_only": matched_only or {
            "mentions": len(mentions), "precision": precision, "recall": recall, "f1": f1,
        },
    }


def candidate_recall(gold_links: Iterable[tuple[str, str]], candidates: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Compatibility wrapper for legacy callers; new reports use domain-specific names."""

    gold = {tuple(sorted(pair)) for pair in gold_links}
    recalled = {tuple(sorted(pair)) for pair in candidates}
    return {"gold": len(gold), "candidates": len(recalled),
            "recall": len(gold & recalled) / len(gold) if gold else 1.0}


def _candidate_recall_details(
    gold_links: Iterable[tuple[str, str]],
    candidates: Iterable[tuple[str, str]],
    *,
    directed: bool,
    metric: str,
    limit_name: str,
    limit: int | None,
) -> dict[str, Any]:
    def normalize(pair: tuple[str, str]) -> tuple[str, str]:
        left, right = str(pair[0]), str(pair[1])
        return (left, right) if directed else tuple(sorted((left, right)))

    gold = {normalize(pair) for pair in gold_links}
    recalled = {normalize(pair) for pair in candidates}
    matched = gold & recalled
    missing = gold - recalled
    separator = "->" if directed else "|"
    lower, upper = wilson_interval(len(matched), len(gold))
    return {
        "metric": metric,
        limit_name: int(limit) if limit is not None else None,
        "gold": len(gold),
        "candidates": len(recalled),
        "true_positive": len(matched),
        "false_negative": len(missing),
        "recall": len(matched) / len(gold) if gold else 1.0,
        "recall_wilson_95": [lower, upper],
        "true_positive_ids": [separator.join(pair) for pair in sorted(matched)],
        "false_negative_ids": [separator.join(pair) for pair in sorted(missing)],
    }


def entity_candidate_recall_at_k(
    gold_links: Iterable[tuple[str, str]],
    candidates: Iterable[tuple[str, str]],
    *,
    k: int | None = None,
) -> dict[str, Any]:
    """Recall of must-link entity pairs inside the bounded normalization candidates."""

    return _candidate_recall_details(
        gold_links,
        candidates,
        directed=False,
        metric="entity_candidate_recall_at_k",
        limit_name="k",
        limit=k,
    )


def relation_pair_candidate_recall_at_budget(
    gold_links: Iterable[tuple[str, str]],
    candidates: Iterable[tuple[str, str]],
    *,
    budget: int | None = None,
) -> dict[str, Any]:
    """Recall of unordered Gold endpoint pairs inside the bounded pair plan."""

    return _candidate_recall_details(
        gold_links,
        candidates,
        directed=False,
        metric="relation_pair_candidate_recall_at_budget",
        limit_name="budget",
        limit=budget,
    )


def evaluate_entity_pair_constraints(
    *,
    must_links: Iterable[tuple[str, str]],
    cannot_links: Iterable[tuple[str, str]],
    hard_negatives: Iterable[tuple[str, str]],
    assignments: dict[str, str],
) -> dict[str, Any]:
    """Evaluate positive and negative pair labels against canonical assignments."""

    must = {tuple(sorted((str(left), str(right)))) for left, right in must_links}
    cannot = {tuple(sorted((str(left), str(right)))) for left, right in cannot_links}
    hard = {tuple(sorted((str(left), str(right)))) for left, right in hard_negatives}
    negatives = cannot | hard
    overlap = must & negatives
    if overlap:
        raise GoldManifestError(f"pair appears in positive and negative Gold: {sorted(overlap)}")

    tp_ids: list[str] = []
    fn_ids: list[str] = []
    tn_ids: list[str] = []
    fp_ids: list[str] = []
    unresolved_ids: list[str] = []

    def same_cluster(pair: tuple[str, str]) -> bool | None:
        left, right = pair
        if left not in assignments or right not in assignments:
            return None
        return str(assignments[left]) == str(assignments[right])

    for pair in sorted(must):
        verdict = same_cluster(pair)
        pair_id = "|".join(pair)
        if verdict is None:
            unresolved_ids.append(pair_id)
        elif verdict:
            tp_ids.append(pair_id)
        else:
            fn_ids.append(pair_id)
    for pair in sorted(negatives):
        verdict = same_cluster(pair)
        pair_id = "|".join(pair)
        if verdict is None:
            unresolved_ids.append(pair_id)
        elif verdict:
            fp_ids.append(pair_id)
        else:
            tn_ids.append(pair_id)

    precision_total = len(tp_ids) + len(fp_ids)
    recall_total = len(tp_ids) + len(fn_ids)
    precision_lower, precision_upper = wilson_interval(len(tp_ids), precision_total)
    recall_lower, recall_upper = wilson_interval(len(tp_ids), recall_total)
    review_queue = [
        {"pair_id": pair_id, "reason": reason}
        for reason, values in (
            ("must_link_not_merged", fn_ids),
            ("negative_pair_merged", fp_ids),
            ("missing_assignment", unresolved_ids),
        )
        for pair_id in values
    ]
    return {
        "confusion_matrix": {
            "true_positive": len(tp_ids),
            "true_negative": len(tn_ids),
            "false_positive": len(fp_ids),
            "false_negative": len(fn_ids),
        },
        "denominators": {
            "must_links": len(must),
            "cannot_links": len(cannot),
            "hard_negatives": len(hard),
        },
        "sample_ids": {
            "true_positive": tp_ids,
            "true_negative": tn_ids,
            "false_positive": fp_ids,
            "false_negative": fn_ids,
            "unresolved": sorted(unresolved_ids),
        },
        "precision": len(tp_ids) / precision_total if precision_total else 1.0,
        "recall": len(tp_ids) / recall_total if recall_total else float(not must),
        "precision_wilson_95": [precision_lower, precision_upper],
        "recall_wilson_95": [recall_lower, recall_upper],
        "coverage_complete": not unresolved_ids,
        "review_queue": review_queue,
    }


def _relation_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return occurrence_relation_assertion_key(item)


def evaluate_relations(gold: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    strict = _multiset_relation_metrics(gold, predicted, key=_relation_key)
    canonical = _multiset_relation_metrics(
        gold, predicted, key=canonical_relation_fact_key,
    )
    precise_evidence = sum(
        item.get("evidence_start") is not None and item.get("evidence_end") is not None
        for item in predicted
    )
    return {
        "strict": strict,
        "canonical": canonical,
        "evidence_coverage": precise_evidence / len(predicted) if predicted else 1.0,
        "evidence_exactness": (
            strict["true_positive"] / precise_evidence
            if precise_evidence else float(not gold)
        ),
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
    legacy_recall = treatment.get("candidate_recall")
    entity_candidate_recall = treatment.get(
        "entity_candidate_recall_at_k", legacy_recall if legacy_recall is not None else 0,
    )
    relation_candidate_recall = treatment.get(
        "relation_pair_candidate_recall_at_budget",
        legacy_recall if legacy_recall is not None else 0,
    )
    if entity_candidate_recall < 0.99:
        failed.append("entity_candidate_recall_at_k")
    if relation_candidate_recall < 0.99:
        failed.append("relation_pair_candidate_recall_at_budget")
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
