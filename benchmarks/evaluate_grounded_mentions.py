#!/usr/bin/env python3
"""Evaluate the production grounded-mention protocol on frozen corpora.

This runner measures mention inventory coverage before canonical resolution.
Canonical B3 is intentionally absent here: it must be scored from a persisted
runtime namespace, not approximated with surface-string clustering.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_forge.config import Config
from knowledge_forge.ollama import OllamaClient
from knowledge_forge.policy import ENTITY_SHADOW_POLICY_SET
from knowledge_forge.schema_evolution import BASE_SCHEMA, SchemaEvolutionService
from knowledge_forge.shadow_context import structural_spans_for_chunk
from knowledge_forge.span_entities import (
    SpanFirstEntityExtractor,
    policy_relation_endpoint_candidates,
    structured_list_item_endpoint_candidates,
    structured_relation_endpoint_candidates,
    structured_section_label_endpoint_candidates,
)


DEFAULT_ROOTS = (
    PROJECT_ROOT / "benchmarks" / "knowledge_forge_cross_domain_v1",
    PROJECT_ROOT / "benchmarks" / "knowledge_forge_cross_domain_zh_v1",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _logical_schema(
    assertions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    schema = json.loads(json.dumps(BASE_SCHEMA, ensure_ascii=False))
    known = {
        str(item.get("name") or "")
        for item in schema.get("relation_types") or ()
    }
    for predicate in sorted({
        str(item["predicate"]) for item in assertions
    } - known):
        schema["relation_types"].append({
            "name": predicate,
            "description": "Frozen calibration relation type.",
            "domain_types": ["*"],
            "range_types": ["*"],
            "optional_soft_cues": [],
        })
    return schema


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _summarize_type_counts(
    counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    by_type: dict[str, dict[str, int | float]] = {}
    for entity_type, values in sorted(counts.items()):
        true_positive = int(values.get("true_positive") or 0)
        predicted = int(values.get("predicted") or 0)
        gold = int(values.get("gold") or 0)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / gold if gold else 0.0
        by_type[entity_type] = {
            "true_positive": true_positive,
            "predicted": predicted,
            "gold": gold,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            ),
        }
    total_true_positive = sum(
        int(item["true_positive"]) for item in by_type.values()
    )
    total_predicted = sum(int(item["predicted"]) for item in by_type.values())
    total_gold = sum(int(item["gold"]) for item in by_type.values())
    micro_precision = (
        total_true_positive / total_predicted if total_predicted else 0.0
    )
    micro_recall = total_true_positive / total_gold if total_gold else 0.0
    return {
        "micro": {
            "hits": total_true_positive,
            "predicted_mentions": total_predicted,
            "gold_mentions": total_gold,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": (
                2 * micro_precision * micro_recall
                / (micro_precision + micro_recall)
                if micro_precision + micro_recall else 0.0
            ),
        },
        "macro_f1": (
            sum(float(item["f1"]) for item in by_type.values())
            / len(by_type)
            if by_type else 1.0
        ),
        "by_type": by_type,
    }


def score_mentions(
    gold: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return exact and one-to-one overlap mention metrics."""

    gold_by_span = {
        (int(item["span"][0]), int(item["span"][1])): item
        for item in gold
    }
    predicted_by_span = {
        (int(item["start"]), int(item["end"])): item
        for item in predicted
    }
    exact_spans = set(gold_by_span) & set(predicted_by_span)
    exact_hits = len(exact_spans)

    candidates: list[tuple[float, int, tuple[int, int], tuple[int, int]]] = []
    for gold_span in gold_by_span:
        for predicted_span in predicted_by_span:
            intersection = _overlap(gold_span, predicted_span)
            if not intersection:
                continue
            union = (
                gold_span[1] - gold_span[0]
                + predicted_span[1] - predicted_span[0]
                - intersection
            )
            candidates.append((
                intersection / max(1, union), intersection,
                gold_span, predicted_span,
            ))
    candidates.sort(reverse=True)
    matched_gold: set[tuple[int, int]] = set()
    matched_predicted: set[tuple[int, int]] = set()
    for _iou, _intersection, gold_span, predicted_span in candidates:
        if gold_span in matched_gold or predicted_span in matched_predicted:
            continue
        matched_gold.add(gold_span)
        matched_predicted.add(predicted_span)

    def metrics(hits: int) -> dict[str, Any]:
        precision = hits / len(predicted_by_span) if predicted_by_span else 0.0
        recall = hits / len(gold_by_span) if gold_by_span else 0.0
        return {
            "hits": hits,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            ),
        }

    type_hits = sum(
        str(gold_by_span[span].get("type") or "GenericEntity")
        == str(predicted_by_span[span].get("type") or "GenericEntity")
        for span in exact_spans
    )
    entity_types = sorted({
        str(item.get("type") or "GenericEntity")
        for item in [*gold, *predicted]
    })
    type_counts = {
        entity_type: {
            "true_positive": sum(
                str(gold_by_span[span].get("type") or "GenericEntity")
                == entity_type
                and str(
                    predicted_by_span[span].get("type")
                    or "GenericEntity"
                ) == entity_type
                for span in exact_spans
            ),
            "predicted": sum(
                str(item.get("type") or "GenericEntity") == entity_type
                for item in predicted
            ),
            "gold": sum(
                str(item.get("type") or "GenericEntity") == entity_type
                for item in gold
            ),
        }
        for entity_type in entity_types
    }
    return {
        "gold_mentions": len(gold_by_span),
        "predicted_mentions": len(predicted_by_span),
        "exact": metrics(exact_hits),
        "overlap": metrics(len(matched_gold)),
        "type_accuracy_on_exact": (
            type_hits / exact_hits if exact_hits else None
        ),
        "exact_type_hits": type_hits,
        "strict_typed": _summarize_type_counts(type_counts),
        "missed_gold": [
            {
                "surface": str(gold_by_span[span]["surface"]),
                "span": list(span),
                "type": str(gold_by_span[span].get("type") or ""),
            }
            for span in sorted(set(gold_by_span) - exact_spans)
        ],
        "non_gold_predictions": [
            {
                "surface": str(predicted_by_span[span]["surface"]),
                "span": list(span),
                "type": str(predicted_by_span[span].get("type") or ""),
            }
            for span in sorted(set(predicted_by_span) - exact_spans)
        ],
    }


def _evaluate_document(
    *,
    client: OllamaClient,
    model: str,
    dataset_root: Path,
    document: dict[str, Any],
    gold_mentions: list[dict[str, Any]],
    logical_schema: dict[str, Any],
) -> dict[str, Any]:
    content = (dataset_root / str(document["path"])).read_text(
        encoding="utf-8"
    )
    service = SchemaEvolutionService(None, client)  # type: ignore[arg-type]
    extraction = service.extract_grounded_mentions(
        {
            "id": 1,
            "project_key": "mention-calibration",
            "llm_model": model,
            "model_max_tokens": int(Config.ollama_default_context_budget),
        },
        {"version": 1, "schema_json": logical_schema},
        {
            "id": 1,
            "document_id": 1,
            "revision_id": str(document["id"]),
            "chunk_key": str(document["id"]),
            "content": content,
            "metadata": {"structural_role": "body"},
        },
    )
    structural_spans = structural_spans_for_chunk(
        content=content,
        blocks=(),
    )
    relation_types = list(logical_schema.get("relation_types") or ())
    raw_entities = [
        *(extraction.get("entities") or ()),
        *policy_relation_endpoint_candidates(content, relation_types),
        *structured_relation_endpoint_candidates(content, structural_spans),
        *structured_list_item_endpoint_candidates(content, structural_spans),
        *structured_section_label_endpoint_candidates(content, structural_spans),
    ]
    normalized = SpanFirstEntityExtractor(
        ENTITY_SHADOW_POLICY_SET["entity_extraction_policy_version"]
    ).normalize(
        project_scope="mention-calibration",
        document_key=str(document["id"]),
        chunk_key=str(document["id"]),
        content=content,
        raw_entities=raw_entities,
        chunk_metadata={
            "structural_role": "body",
            "structural_spans": structural_spans,
        },
    )
    type_by_mention = {
        item.mention_id: item.entity_type
        for item in normalized.type_assertions
    }
    predictions = [
        {
            **asdict(item),
            "type": type_by_mention.get(item.mention_id, "GenericEntity"),
        }
        for item in normalized.mentions
    ]
    metrics = score_mentions(gold_mentions, predictions)
    return {
        "document_id": str(document["id"]),
        "language": str(document.get("language") or "en"),
        "domain": str(document.get("domain") or ""),
        "metrics": metrics,
        "diagnostics": dict(extraction.get("_diagnostics") or {}),
        "normalized_rejections": Counter(
            str(item.get("reason") or "unknown")
            for item in normalized.rejected
        ),
    }


def _micro(documents: list[dict[str, Any]], key: str) -> dict[str, Any]:
    gold = sum(int(item["metrics"]["gold_mentions"]) for item in documents)
    predicted = sum(
        int(item["metrics"]["predicted_mentions"]) for item in documents
    )
    hits = sum(int(item["metrics"][key]["hits"]) for item in documents)
    precision = hits / predicted if predicted else 0.0
    recall = hits / gold if gold else 0.0
    return {
        "hits": hits,
        "gold_mentions": gold,
        "predicted_mentions": predicted,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        ),
    }


def _aggregate_typed(documents: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for document in documents:
        by_type = document["metrics"]["strict_typed"]["by_type"]
        for entity_type, values in by_type.items():
            target = counts.setdefault(str(entity_type), {
                "true_positive": 0,
                "predicted": 0,
                "gold": 0,
            })
            for field in ("true_positive", "predicted", "gold"):
                target[field] += int(values[field])
    return _summarize_type_counts(counts)


def _bucket_metrics(
    documents: list[dict[str, Any]], key: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in sorted({str(item[key]) for item in documents}):
        selected = [item for item in documents if str(item[key]) == value]
        result[value] = {
            "documents": len(selected),
            "exact": _micro(selected, "exact"),
            "overlap": _micro(selected, "overlap"),
            "strict_typed": _aggregate_typed(selected),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", action="append", type=Path, default=[],
        help="Frozen corpus root; repeat for bilingual evaluation.",
    )
    parser.add_argument("--documents", nargs="*", default=[])
    parser.add_argument("--model", default=Config.default_llm_model)
    parser.add_argument("--gleaning-rounds", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    roots = [value.resolve() for value in args.dataset_root] or list(
        DEFAULT_ROOTS
    )
    selected = set(str(value) for value in args.documents)
    client = OllamaClient(timeout=Config.ollama_timeout)
    original_rounds = int(Config.entity_grounded_gleaning_rounds)
    Config.entity_grounded_gleaning_rounds = max(
        0, min(1, int(args.gleaning_rounds))
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    available: set[str] = set()
    try:
        for root in roots:
            manifest = _read(root / "manifest.json")
            entities = _read(
                root / str(manifest["annotations"]["entities"])
            )["documents"]
            assertion_payload = _read(
                root / str(manifest["annotations"]["assertions"])
            )
            logical_schema = _logical_schema(assertion_payload["relations"])
            for document in manifest["documents"]:
                document_id = str(document["id"])
                available.add(document_id)
                if selected and document_id not in selected:
                    continue
                results.append(_evaluate_document(
                    client=client,
                    model=str(args.model),
                    dataset_root=root,
                    document=document,
                    gold_mentions=list(entities[document_id]),
                    logical_schema=logical_schema,
                ))
    finally:
        Config.entity_grounded_gleaning_rounds = original_rounds
    missing = selected - available
    if missing:
        raise SystemExit("unknown document ids: " + ", ".join(sorted(missing)))
    type_hits = sum(
        int(item["metrics"]["exact_type_hits"]) for item in results
    )
    exact_hits = sum(
        int(item["metrics"]["exact"]["hits"]) for item in results
    )
    payload = {
        "model": str(args.model),
        "dataset_roots": [str(value) for value in roots],
        "gleaning_rounds": max(0, min(1, int(args.gleaning_rounds))),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "documents": results,
        "aggregate": {
            "exact": _micro(results, "exact"),
            "overlap": _micro(results, "overlap"),
            "strict_typed": _aggregate_typed(results),
            "type_accuracy_on_exact": (
                type_hits / exact_hits if exact_hits else None
            ),
            "canonical_b3": None,
            "canonical_b3_status": "requires_persisted_runtime_namespace",
            "llm_calls": sum(
                int(item["diagnostics"].get("llm_calls") or 0)
                for item in results
            ),
        },
        "by_language": _bucket_metrics(results, "language"),
        "by_domain": _bucket_metrics(results, "domain"),
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=dict)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
