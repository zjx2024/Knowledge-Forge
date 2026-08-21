#!/usr/bin/env python3
"""Audit the frozen English and Chinese calibration corpora."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.knowledge_forge_long_document_stress_v1.evaluate import (
    evaluate as evaluate_long_document_stress,
)


CORPORA = (
    ROOT / "knowledge_forge_cross_domain_v1",
    ROOT / "knowledge_forge_cross_domain_zh_v1",
)
HYBRID_RAG_ROOT = ROOT / "knowledge_forge_hybrid_rag_v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit() -> dict[str, Any]:
    document_count = 0
    document_bytes: list[int] = []
    languages: Counter[str] = Counter()
    domains: set[str] = set()
    structures: set[str] = set()
    mentions = 0
    canonical_entities = 0
    positives = 0
    core_positives = 0
    unique_facts: set[tuple[str, str, str, str]] = set()
    core_unique_facts: set[tuple[str, str, str, str]] = set()
    negatives = 0
    predicates: Counter[str] = Counter()
    production_eligible = True
    corpus_rows: list[dict[str, Any]] = []
    interpretation_assertion_ids: set[str] = set()

    for corpus in CORPORA:
        manifest = _read(corpus / "manifest.json")
        entities = _read(corpus / manifest["annotations"]["entities"])[
            "documents"
        ]
        assertions = _read(corpus / manifest["annotations"]["assertions"])
        interpretation_ids: set[str] = set()
        tier_path = corpus / "annotations" / "adjudication_tiers_2026_08_11.json"
        if tier_path.is_file():
            interpretation_ids = {
                str(item["assertion_id"])
                for item in _read(tier_path)["interpretation_band"]
            }
        interpretation_ids.update(
            str(item["assertion_id"])
            for item in assertions["relations"]
            if str(item.get("adjudication_tier") or "core_direct")
            != "core_direct"
        )
        interpretation_assertion_ids.update(interpretation_ids)
        corpus_core = 0
        for item in assertions["relations"]:
            positives += 1
            predicates[str(item["predicate"])] += 1
            fact = (
                str(item["document_id"]),
                str(item["subject_canonical_id"]),
                str(item["predicate"]),
                str(item["object_canonical_id"]),
            )
            unique_facts.add(fact)
            is_core = (
                str(item.get("assertion_id") or "") not in interpretation_ids
                and str(item.get("adjudication_tier") or "core_direct")
                == "core_direct"
            )
            if is_core:
                core_positives += 1
                corpus_core += 1
                core_unique_facts.add(fact)
        negatives += len(assertions.get("negative_claims") or ())
        corpus_mentions = sum(len(rows) for rows in entities.values())
        corpus_canonicals = sum(
            len({str(item["canonical_id"]) for item in rows})
            for rows in entities.values()
        )
        mentions += corpus_mentions
        canonical_entities += corpus_canonicals
        production_eligible = production_eligible and bool(
            manifest.get("production_gate_eligible")
        )
        for document in manifest["documents"]:
            content = (corpus / document["path"]).read_text(encoding="utf-8")
            document_count += 1
            document_bytes.append(len(content.encode("utf-8")))
            language = str(
                document.get("language")
                or manifest.get("language")
                or "en"
            )
            languages[language] += 1
            domains.add(str(document["domain"]))
            structures.update(str(value) for value in document["structure"])
        corpus_rows.append({
            "dataset": str(manifest["dataset"]),
            "version": str(manifest["version"]),
            "documents": len(manifest["documents"]),
            "entity_mentions": corpus_mentions,
            "canonical_entities": corpus_canonicals,
            "positive_assertions": len(assertions["relations"]),
            "core_positive_assertions": corpus_core,
            "hard_negatives": len(assertions.get("negative_claims") or ()),
            "production_gate_eligible": bool(
                manifest.get("production_gate_eligible")
            ),
        })

    singleton_predicates = sorted(
        predicate for predicate, count in predicates.items() if count == 1
    )
    hybrid_questions: list[dict[str, Any]] = []
    hybrid_manifest: dict[str, Any] = {}
    if (HYBRID_RAG_ROOT / "manifest.json").is_file():
        hybrid_manifest = _read(HYBRID_RAG_ROOT / "manifest.json")
        hybrid_questions = _read(
            HYBRID_RAG_ROOT / hybrid_manifest["questions_path"]
        )
    hybrid_languages = Counter(
        str(item["language"]) for item in hybrid_questions
    )
    hybrid_task_types = Counter(
        str(item["task_type"]) for item in hybrid_questions
    )
    hybrid_positive_labels = {
        str(value)
        for item in hybrid_questions
        for value in item.get("required_assertion_ids") or ()
    }
    hybrid_interpretation_labels = sorted(
        hybrid_positive_labels & interpretation_assertion_ids
    )
    hybrid_ready = bool(
        len(hybrid_questions) >= 24
        and len(hybrid_languages) >= 2
        and {
            "single_fact", "multi_hop", "aggregation", "vector_fallback",
        }.issubset(hybrid_task_types)
        and not hybrid_interpretation_labels
    )
    try:
        long_document_report = evaluate_long_document_stress()
    except Exception as exc:  # The audit must report a gap, not hide it.
        long_document_report = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    long_document_ready = bool(
        long_document_report.get("status") == "pass"
        and all(
            bool(value)
            for value in (long_document_report.get("checks") or {}).values()
        )
    )
    gaps = []
    if not long_document_ready:
        gaps.append("no_long_document_performance_corpus")
    if not production_eligible:
        gaps.append("no_independent_human_production_gold")
    if len(core_unique_facts) < 200:
        gaps.append("below_optional_200_fact_independent_gold_target")
    if not hybrid_ready:
        gaps.append("no_downstream_hybrid_rag_question_set")
    engineering_adequate = bool(
        document_count >= 10
        and len(languages) >= 2
        and len(domains) >= 8
        and len(structures) >= 10
        and core_positives >= 120
        and negatives >= 24
        and long_document_ready
    )
    return {
        "status": (
            "engineering_calibration_adequate"
            if engineering_adequate else "calibration_expansion_needed"
        ),
        "corpora": corpus_rows,
        "totals": {
            "documents": document_count,
            "languages": dict(sorted(languages.items())),
            "domains": len(domains),
            "structures": len(structures),
            "document_bytes_min": min(document_bytes, default=0),
            "document_bytes_max": max(document_bytes, default=0),
            "entity_mentions": mentions,
            "canonical_entities": canonical_entities,
            "positive_assertions": positives,
            "core_positive_assertions": core_positives,
            "unique_facts": len(unique_facts),
            "core_unique_facts": len(core_unique_facts),
            "hard_negatives": negatives,
            "relation_types": len(predicates),
            "singleton_relation_types": len(singleton_predicates),
            "singleton_relation_type_ratio": (
                len(singleton_predicates) / len(predicates)
                if predicates else 0.0
            ),
        },
        "adequacy": {
            "engineering_calibration": engineering_adequate,
            "production_release_gold": production_eligible,
            "downstream_question_set": hybrid_ready,
            "long_document_scaling": long_document_ready,
            "dataset_expansion_required_now": not engineering_adequate,
        },
        "downstream": {
            "dataset": str(hybrid_manifest.get("dataset") or ""),
            "questions": len(hybrid_questions),
            "languages": dict(sorted(hybrid_languages.items())),
            "task_types": dict(sorted(hybrid_task_types.items())),
            "positive_labels": len(hybrid_positive_labels),
            "interpretation_labels_in_hard_gate": (
                hybrid_interpretation_labels
            ),
        },
        "long_document": long_document_report,
        "gaps": gaps,
        "expansion_policy": {
            "bulk_expansion_recommended_now": False,
            "reason": (
                "The bilingual calibration set is large and varied enough for "
                "engineering regression; more single-expert labels would add "
                "volume without independent adjudication."
            ),
            "reopen_only_if": [
                "language_or_structure_bucket_core_recall_below_0.70",
                "reproducible_hard_safety_failure",
                "downstream_hybrid_retrieval_deficit_caused_by_graph_extraction",
                "new_document_family_not_covered_by_existing_buckets",
            ],
            "optional_independent_gold_target_facts": 200,
        },
        "metric_policy": {
            "primary": "core_direct_micro_precision_recall_f1",
            "secondary": "all_label_interpretation_band",
            "always_gate": [
                "invalid_evidence", "explicit_negative_leak",
                "unsupported_reverse", "related_to_publication",
            ],
            "do_not_use_alone": [
                "macro_f1_over_singleton_predicates", "aggregate_exact_match",
            ],
        },
    }


def main() -> int:
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
