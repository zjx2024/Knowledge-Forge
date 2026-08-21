#!/usr/bin/env python3
"""Score the public Knowledge Forge search route on frozen RAG questions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _contains_evidence(text: Any, evidence: Any) -> bool:
    content = _normalized(text)
    expected = _normalized(evidence)
    return bool(
        content and expected
        and (expected in content or content in expected)
    )


def _label_evidence(row: Mapping[str, Any]) -> str:
    return str(row.get("evidence_text") or row.get("evidence") or "")


def load_benchmark() -> dict[str, Any]:
    manifest = _load(ROOT / "manifest.json")
    questions = _load(ROOT / str(manifest["questions_path"]))
    assertions: dict[str, dict[str, Any]] = {}
    negatives: dict[str, dict[str, Any]] = {}
    canonical_surfaces: dict[str, set[str]] = defaultdict(set)
    for relative in manifest["source_corpora"]:
        corpus = (ROOT / str(relative)).resolve()
        corpus_manifest = _load(corpus / "manifest.json")
        annotation = _load(
            corpus / str(corpus_manifest["annotations"]["assertions"])
        )
        for row in annotation["relations"]:
            assertions[str(row["assertion_id"])] = row
        for row in annotation["negative_claims"]:
            negatives[str(row["claim_id"])] = row
        entity_documents = _load(
            corpus / str(corpus_manifest["annotations"]["entities"])
        )["documents"]
        for mentions in entity_documents.values():
            for mention in mentions:
                canonical_surfaces[str(mention["canonical_id"])].add(
                    str(mention["surface"])
                )
    return {
        "manifest": manifest,
        "questions": questions,
        "assertions": assertions,
        "negatives": negatives,
        "canonical_surfaces": canonical_surfaces,
    }


def _endpoint_matches(
    runtime_name: Any,
    canonical_id: str,
    label_surface: Any,
    canonical_surfaces: Mapping[str, set[str]],
) -> bool:
    name = _normalized(runtime_name)
    candidates = {
        _normalized(label_surface),
        *(
            _normalized(value)
            for value in canonical_surfaces.get(canonical_id, set())
        ),
    }
    return bool(name and any(
        value and (value == name or value in name or name in value)
        for value in candidates
    ))


def _relation_matches(
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    canonical_surfaces: Mapping[str, set[str]],
    *,
    require_exact_predicate: bool = True,
) -> bool:
    if (
        require_exact_predicate
        and _normalized(row.get("relation_type")) != _normalized(
            label.get("predicate")
        )
    ):
        return False
    if not _endpoint_matches(
        row.get("source_name"),
        str(label.get("subject_canonical_id") or label.get("subject") or ""),
        label.get("subject_surface"),
        canonical_surfaces,
    ):
        return False
    if not _endpoint_matches(
        row.get("target_name"),
        str(label.get("object_canonical_id") or label.get("object") or ""),
        label.get("object_surface"),
        canonical_surfaces,
    ):
        return False
    evidence = row.get("evidence_text") or row.get("description") or ""
    return _contains_evidence(evidence, _label_evidence(label))


def score_question(
    question: Mapping[str, Any],
    response: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    chunks = list(response.get("chunks") or ())[:top_k]
    relations = list(response.get("relations") or ())[:top_k]
    assertions = benchmark["assertions"]
    negatives = benchmark["negatives"]
    surfaces = benchmark["canonical_surfaces"]

    positive_rows = [
        assertions[str(value)]
        for value in question.get("required_assertion_ids") or ()
    ]
    negative_rows = [
        negatives[str(value)]
        for value in question.get("required_negative_claim_ids") or ()
    ]
    vector_positive_hits = [
        any(
            _contains_evidence(chunk.get("content"), _label_evidence(row))
            for chunk in chunks
        )
        for row in positive_rows
    ]
    vector_negative_hits = [
        any(
            _contains_evidence(chunk.get("content"), _label_evidence(row))
            for chunk in chunks
        )
        for row in negative_rows
    ]
    graph_grounded_positive_hits = [
        any(
            _relation_matches(
                relation,
                row,
                surfaces,
                require_exact_predicate=False,
            )
            for relation in relations
        )
        for row in positive_rows
    ]
    graph_exact_predicate_hits = [
        any(
            _relation_matches(relation, row, surfaces)
            for relation in relations
        )
        for row in positive_rows
    ]
    negative_relation_leaks = [
        row for row in negative_rows
        if any(
            _relation_matches(relation, row, surfaces)
            for relation in relations
        )
    ]
    vector_success = bool(
        all(vector_positive_hits) and all(vector_negative_hits)
    )
    generic_relation_publications = sum(
        _normalized(relation.get("relation_type"))
        in {"related_to", "relation"}
        for relation in relations
    )
    graph_success = bool(
        positive_rows
        and all(graph_grounded_positive_hits)
        and not negative_rows
        and not negative_relation_leaks
        and not generic_relation_publications
    )
    hybrid_success = bool(
        all(
            vector_hit or graph_hit
            for vector_hit, graph_hit in zip(
                vector_positive_hits,
                graph_grounded_positive_hits,
                strict=True,
            )
        )
        and all(vector_negative_hits)
        and not negative_relation_leaks
        and not generic_relation_publications
    )
    return {
        "question_id": str(question["question_id"]),
        "language": str(question["language"]),
        "task_type": str(question["task_type"]),
        "retrieval_expectation": str(question["retrieval_expectation"]),
        "vector_success": vector_success,
        "graph_success": graph_success,
        "hybrid_success": hybrid_success,
        "vector_positive_hits": sum(vector_positive_hits),
        "graph_positive_hits": sum(graph_grounded_positive_hits),
        "graph_exact_predicate_positive_hits": sum(
            graph_exact_predicate_hits
        ),
        "required_positive_facts": len(positive_rows),
        "vector_negative_evidence_hits": sum(vector_negative_hits),
        "required_negative_facts": len(negative_rows),
        "negative_relation_leaks": len(negative_relation_leaks),
        "generic_relation_publications": generic_relation_publications,
        "vector_context_chars": sum(
            len(str(chunk.get("content") or "")) for chunk in chunks
        ),
        "graph_context_chars": sum(
            len(str(
                relation.get("evidence_text")
                or relation.get("description") or ""
            ))
            for relation in relations
        ),
    }


def _bucket(
    rows: list[dict[str, Any]], key: str,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for value in sorted({str(row[key]) for row in rows}):
        group = [row for row in rows if str(row[key]) == value]
        result[value] = {
            "questions": len(group),
            "vector": sum(bool(row["vector_success"]) for row in group),
            "graph": sum(bool(row["graph_success"]) for row in group),
            "hybrid": sum(bool(row["hybrid_success"]) for row in group),
        }
    return result


def summarize(
    rows: list[dict[str, Any]], manifest: Mapping[str, Any],
) -> dict[str, Any]:
    total = len(rows)
    vector_hits = sum(bool(row["vector_success"]) for row in rows)
    graph_hits = sum(bool(row["graph_success"]) for row in rows)
    hybrid_hits = sum(bool(row["hybrid_success"]) for row in rows)
    leaks = sum(int(row["negative_relation_leaks"]) for row in rows)
    generic_publications = sum(
        int(row.get("generic_relation_publications") or 0) for row in rows
    )
    grounded_graph_facts = sum(
        int(row.get("graph_positive_hits") or 0) for row in rows
    )
    exact_predicate_graph_facts = sum(
        int(row.get("graph_exact_predicate_positive_hits") or 0)
        for row in rows
    )
    graph_suitable = [
        row for row in rows
        if row["retrieval_expectation"] == "graph_preferred"
    ]
    suitable_vector = sum(
        bool(row["vector_success"]) for row in graph_suitable
    )
    suitable_graph = sum(
        bool(row["graph_success"]) for row in graph_suitable
    )
    suitable_total = max(1, len(graph_suitable))
    graph_gain_rate = (suitable_graph - suitable_vector) / suitable_total
    suitable_required_facts = sum(
        int(row.get("required_positive_facts") or 0)
        for row in graph_suitable
    )
    suitable_graph_fact_hits = sum(
        int(row.get("graph_positive_hits") or 0)
        for row in graph_suitable
    )
    suitable_graph_fact_coverage = (
        suitable_graph_fact_hits / suitable_required_facts
        if suitable_required_facts else 0.0
    )
    suitable_vector_chars = sum(
        int(row.get("vector_context_chars") or 0)
        for row in graph_suitable
    )
    suitable_graph_chars = sum(
        int(row.get("graph_context_chars") or 0)
        for row in graph_suitable
    )
    graph_context_ratio = (
        suitable_graph_chars / suitable_vector_chars
        if suitable_vector_chars else None
    )
    same_accuracy_less_text = bool(
        suitable_graph == suitable_vector
        and suitable_graph > 0
        and sum(row["graph_context_chars"] for row in graph_suitable)
        < sum(row["vector_context_chars"] for row in graph_suitable)
    )
    acceptance = manifest["retrieval_acceptance"]
    rates = {
        "vector": vector_hits / total if total else 0.0,
        "graph": graph_hits / total if total else 0.0,
        "hybrid": hybrid_hits / total if total else 0.0,
        "graph_suitable_vector": suitable_vector / suitable_total,
        "graph_suitable_graph": suitable_graph / suitable_total,
        "graph_suitable_gain": graph_gain_rate,
        "graph_suitable_fact_coverage": suitable_graph_fact_coverage,
    }
    grounded_context_value = bool(
        suitable_graph_fact_coverage
        >= float(acceptance["minimum_graph_preferred_fact_coverage"])
        and graph_context_ratio is not None
        and graph_context_ratio
        <= float(acceptance["maximum_graph_context_ratio_vs_vector"])
    )
    checks = {
        "hybrid_minimum": (
            rates["hybrid"]
            >= float(acceptance["minimum_hybrid_evidence_hit_rate"])
        ),
        "hybrid_vector_noninferiority": (
            rates["hybrid"] + float(
                acceptance["maximum_hybrid_regression_rate_vs_vector"]
            ) >= rates["vector"]
        ),
        "graph_adds_value": (
            graph_gain_rate
            >= float(acceptance["minimum_graph_suitable_gain_rate"])
            or same_accuracy_less_text
            or grounded_context_value
        ),
        "negative_relation_safety": (
            leaks <= int(acceptance["maximum_negative_relation_leaks"])
        ),
        "generic_relation_safety": generic_publications == 0,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "questions": total,
        "hits": {
            "vector": vector_hits,
            "graph": graph_hits,
            "hybrid": hybrid_hits,
        },
        "rates": {key: round(value, 4) for key, value in rates.items()},
        "negative_relation_leaks": leaks,
        "generic_relation_publications": generic_publications,
        "predicate_normalization": {
            "grounded_endpoint_evidence_hits": grounded_graph_facts,
            "exact_predicate_hits": exact_predicate_graph_facts,
            "exact_rate_on_grounded_hits": (
                exact_predicate_graph_facts / grounded_graph_facts
                if grounded_graph_facts else None
            ),
            "status": "diagnostic_for_open_project_schema",
        },
        "same_graph_accuracy_with_less_text": same_accuracy_less_text,
        "graph_suitable_facts": {
            "hits": suitable_graph_fact_hits,
            "required": suitable_required_facts,
            "coverage": round(suitable_graph_fact_coverage, 4),
            "graph_context_chars": suitable_graph_chars,
            "vector_context_chars": suitable_vector_chars,
            "context_ratio": (
                round(graph_context_ratio, 4)
                if graph_context_ratio is not None else None
            ),
        },
        "checks": checks,
        "by_language": _bucket(rows, "language"),
        "by_task_type": _bucket(rows, "task_type"),
    }


def evaluate_project(base_url: str, project_id: int) -> dict[str, Any]:
    benchmark = load_benchmark()
    manifest = benchmark["manifest"]
    top_k = int(manifest["retrieval_acceptance"]["top_k"])
    rows = []
    # This is an acceptance probe for the explicitly supplied Knowledge Forge
    # endpoint.  Environment proxies can otherwise intercept loopback URLs
    # (urllib does not consistently honour every NO_PROXY spelling) and turn a
    # healthy local route into a misleading proxy-side 502.
    opener = build_opener(ProxyHandler({}))
    for question in benchmark["questions"]:
        url = (
            base_url.rstrip("/") + "/api/search?"
            + urlencode({
                "project": project_id,
                "q": question["question"],
                "top_k": top_k,
            })
        )
        with opener.open(url, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows.append(score_question(
            question, payload, benchmark, top_k=top_k,
        ))
    summary = summarize(rows, manifest)
    return {
        **summary,
        "project_id": project_id,
        "base_url": base_url,
        "top_k": top_k,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate_project(args.base_url, args.project_id)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
