#!/usr/bin/env python3
"""Score a Knowledge Forge graph against the expert calibration corpus.

The complete assertion universe is the primary metric.  The older 43-pair
slice remains in the report only for historical A/B comparisons; its local
NO_RELATION cases must not be interpreted as graph-global negatives when the
same canonical pair has a supported fact at another occurrence.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener


DEFAULT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEFAULT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.calibration_policy import (  # noqa: E402
    partition_assertions,
    unique_fact_rows,
)


ROOT = DEFAULT_ROOT


def _read(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("`", "").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def _alias_norm(value: object) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", " ", _norm(value)).strip()


def _conservative_alias_overlap(left: str, right: str) -> bool:
    """Match a nested alias only when it is long enough to be meaningful."""

    left_value = _alias_norm(left)
    right_value = _alias_norm(right)
    if not left_value or not right_value or left_value == right_value:
        return left_value == right_value
    shorter, longer = sorted(
        (left_value, right_value), key=lambda value: (len(value), value)
    )
    contains_cjk = bool(re.search(r"[\u3400-\u9fff]", shorter))
    compact = re.sub(r"\W+", "", shorter, flags=re.UNICODE)
    if len(compact) < (2 if contains_cjk else 4):
        return False
    if contains_cjk:
        return shorter in longer
    return bool(re.search(
        rf"(?<!\w){re.escape(shorter)}(?!\w)", longer,
        flags=re.UNICODE,
    ))


def _contains_evidence(actual: object, expected: object) -> bool:
    actual_value = _norm(actual)
    expected_value = _norm(expected)
    return bool(
        actual_value
        and expected_value
        and (
            actual_value in expected_value
            or expected_value in actual_value
        )
    )


def _surface(document_id: str, start: int, end: int) -> str:
    manifest = _read("manifest.json")
    item = next(row for row in manifest["documents"] if row["id"] == document_id)
    text = (ROOT / item["path"]).read_text(encoding="utf-8")
    return text[start:end]


def _endpoint_names(graph: dict) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for node in graph.get("nodes") or []:
        values = {node.get("title"), node.get("label"), node.get("name")}
        values.update(node.get("aliases") or [])
        names[str(node.get("id"))] = {
            _norm(value) for value in values if _norm(value)
        }
    return names


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Frozen calibration dataset root (defaults to the English v1 "
            "corpus)."
        ),
    )
    parser.add_argument("--project-id", type=int)
    parser.add_argument("--graph-json", type=Path)
    parser.add_argument(
        "--document-ids",
        default="",
        help=(
            "Comma-separated runtime document IDs belonging to this dataset. "
            "Use when one project contains several benchmark corpora so edges "
            "from the other corpus are not counted as false positives."
        ),
    )
    args = parser.parse_args()
    ROOT = args.dataset_root.expanduser().resolve()
    if not (ROOT / "manifest.json").is_file():
        raise SystemExit(f"dataset manifest not found: {ROOT / 'manifest.json'}")
    if bool(args.project_id) == bool(args.graph_json):
        raise SystemExit("provide exactly one of --project-id or --graph-json")
    if args.graph_json:
        graph = json.loads(args.graph_json.read_text(encoding="utf-8"))
    else:
        url = (
            f"http://127.0.0.1:8011/api/projects/{args.project_id}/"
            "knowledge-forge/graph?limit=100000&include_isolates=1"
        )
        request = Request(url, headers={"User-Agent": "KnowledgeForgeBenchmark/2.0"})
        with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
            graph = json.load(response)

    manifest = _read("manifest.json")
    annotation_paths = manifest["annotations"]
    entity_payload = _read(str(annotation_paths["entities"]))
    assertion_payload = _read(str(annotation_paths["assertions"]))
    complete_assertions = list(assertion_payload.get("relations") or [])
    assertion_bands = partition_assertions(ROOT, complete_assertions)
    core_assertions = assertion_bands["core_direct"]
    interpretation_assertions = assertion_bands["interpretation_band"]
    negative_claims = list(assertion_payload.get("negative_claims") or [])
    legacy_pairs = (
        _read(str(annotation_paths["pairs"]))
        if annotation_paths.get("pairs") else []
    )
    legacy_facts = (
        _read(str(annotation_paths["facts"]))
        if annotation_paths.get("facts") else []
    )
    legacy_unsupported = (
        _read(str(annotation_paths["unsupported_facts"]))
        if annotation_paths.get("unsupported_facts") else []
    )

    nodes = _endpoint_names(graph)
    all_graph_edges = list(graph.get("edges") or [])
    selected_document_ids = {
        int(value.strip())
        for value in str(args.document_ids or "").split(",")
        if value.strip()
    }
    edges = [
        edge for edge in all_graph_edges
        if not selected_document_ids
        or int(edge.get("document_id") or 0) in selected_document_ids
    ]

    mention_rows: list[dict] = []
    canonical_surfaces: dict[str, set[str]] = defaultdict(set)
    surface_to_canonicals: dict[str, set[str]] = defaultdict(set)
    for document_id, rows in entity_payload.get("documents", {}).items():
        for raw in rows:
            mention = {**raw, "document_id": document_id}
            mention_rows.append(mention)
            surface = _norm(mention.get("surface"))
            canonical_id = str(mention.get("canonical_id") or "")
            if surface and canonical_id:
                canonical_surfaces[canonical_id].add(surface)
                surface_to_canonicals[surface].add(canonical_id)

    graph_entity_names = set().union(*nodes.values()) if nodes else set()
    entity_mention_hits = [
        mention for mention in mention_rows
        if _norm(mention.get("surface")) in graph_entity_names
    ]
    gold_canonical_ids = set(canonical_surfaces)
    hit_canonical_ids = {
        str(mention["canonical_id"]) for mention in entity_mention_hits
    }

    def edge_names(edge: dict) -> tuple[set[str], set[str]]:
        source = set(nodes.get(str(edge.get("source")), set()))
        target = set(nodes.get(str(edge.get("target")), set()))
        source.add(_norm(edge.get("source_title")))
        target.add(_norm(edge.get("target_title")))
        return source - {""}, target - {""}

    def canonical_candidates(names: set[str]) -> set[str]:
        exact: set[str] = set()
        for value in names:
            exact.update(surface_to_canonicals.get(value) or set())
        if exact:
            return exact
        fuzzy: set[str] = set()
        for value in names:
            for surface, canonical_ids in surface_to_canonicals.items():
                if _conservative_alias_overlap(value, surface):
                    fuzzy.update(canonical_ids)
        # Ambiguous containment (for example "requirements") is not an alias
        # decision.  The evaluator stays conservative instead of guessing.
        return fuzzy if len(fuzzy) == 1 else set()

    alias_aligned_canonical_ids: set[str] = set()
    for graph_name in graph_entity_names:
        alias_aligned_canonical_ids.update(
            canonical_candidates({graph_name})
        )

    edge_analysis: list[dict] = []
    for edge in edges:
        source_names, target_names = edge_names(edge)
        edge_analysis.append({
            "edge": edge,
            "source_names": source_names,
            "target_names": target_names,
            "source_canonicals": canonical_candidates(source_names),
            "target_canonicals": canonical_candidates(target_names),
            "predicate": _norm(edge.get("predicate_id") or edge.get("predicate")),
        })

    def assertion_evidence(assertion: dict) -> str:
        copied = str(assertion.get("evidence_text") or "")
        if copied:
            return copied
        span = assertion.get("evidence_span")
        document_id = str(assertion.get("document_id") or "")
        if (
            document_id
            and isinstance(span, (list, tuple))
            and len(span) == 2
            and all(isinstance(value, int) for value in span)
            and 0 <= int(span[0]) < int(span[1])
        ):
            return _surface(document_id, int(span[0]), int(span[1]))
        return ""

    def analysis_matches_assertion(
        analysis: dict,
        assertion: dict,
        *,
        exact_predicate: bool,
        require_evidence: bool,
        allow_reverse_endpoint: bool = False,
    ) -> bool:
        subject = str(assertion["subject_canonical_id"])
        object_ = str(assertion["object_canonical_id"])
        predicate = _norm(assertion.get("predicate"))
        directed_endpoint = (
            subject in analysis["source_canonicals"]
            and object_ in analysis["target_canonicals"]
        )
        reverse_endpoint = bool(
            allow_reverse_endpoint
            and subject in analysis["target_canonicals"]
            and object_ in analysis["source_canonicals"]
        )
        if not (directed_endpoint or reverse_endpoint):
            return False
        if exact_predicate and analysis["predicate"] != predicate:
            return False
        if not require_evidence:
            return True
        edge = analysis["edge"]
        return _contains_evidence(
            edge.get("evidence_text") or edge.get("description"),
            assertion_evidence(assertion),
        )

    def assertion_matches(
        assertion: dict,
        *,
        exact_predicate: bool,
        require_evidence: bool = False,
        allow_reverse_endpoint: bool = False,
    ) -> list[dict]:
        return [
            analysis["edge"] for analysis in edge_analysis
            if analysis_matches_assertion(
                analysis,
                assertion,
                exact_predicate=exact_predicate,
                require_evidence=require_evidence,
                allow_reverse_endpoint=allow_reverse_endpoint,
            )
        ]

    def score_assertions(
        assertions: list[dict], *, deduplicate: bool,
    ) -> list[dict]:
        selected = (
            list(unique_fact_rows(assertions).values())
            if deduplicate else assertions
        )
        result = []
        for assertion in selected:
            endpoint_edges = assertion_matches(
                assertion, exact_predicate=False,
            )
            exact_edges = assertion_matches(assertion, exact_predicate=True)
            grounded_endpoint_edges = assertion_matches(
                assertion,
                exact_predicate=False,
                require_evidence=True,
            )
            grounded_unordered_endpoint_edges = assertion_matches(
                assertion,
                exact_predicate=False,
                require_evidence=True,
                allow_reverse_endpoint=True,
            )
            grounded_exact_edges = assertion_matches(
                assertion,
                exact_predicate=True,
                require_evidence=True,
            )
            result.append({
                "assertion": assertion,
                "endpoint_match": bool(endpoint_edges),
                "exact_match": bool(exact_edges),
                "grounded_endpoint_match": bool(grounded_endpoint_edges),
                "grounded_unordered_endpoint_match": bool(
                    grounded_unordered_endpoint_edges
                ),
                "grounded_exact_match": bool(grounded_exact_edges),
                "edges": exact_edges[:3] or endpoint_edges[:3],
            })
        return result

    assertion_results = score_assertions(
        complete_assertions, deduplicate=False,
    )
    # A graph stores semantic facts, while the annotation universe can retain
    # more than one evidence occurrence for the same fact.  Deduplicate only
    # for graph-level recall; occurrence counts remain visible by structure.
    unique_results = score_assertions(
        complete_assertions, deduplicate=True,
    )
    core_unique_results = score_assertions(
        core_assertions, deduplicate=True,
    )
    interpretation_unique_results = score_assertions(
        interpretation_assertions, deduplicate=True,
    )

    def predicates_by_pair(
        assertions: list[dict],
    ) -> dict[tuple[str, str], set[str]]:
        result: dict[tuple[str, str], set[str]] = defaultdict(set)
        for assertion in assertions:
            result[(
                str(assertion["subject_canonical_id"]),
                str(assertion["object_canonical_id"]),
            )].add(_norm(assertion["predicate"]))
        return result

    expected_predicates_by_pair = predicates_by_pair(complete_assertions)
    core_predicates_by_pair = predicates_by_pair(core_assertions)
    interpretation_predicates_by_pair = predicates_by_pair(
        interpretation_assertions
    )

    exact_supported_edges = []
    endpoint_supported_edges = []
    unordered_endpoint_supported_edges = []
    endpoint_unsupported_edges = []
    unsupported_edges = []
    core_exact_supported_edges = []
    core_evaluated_unsupported_edges = []
    interpretation_only_edges = []
    for analysis in edge_analysis:
        expected: set[str] = set()
        core_expected: set[str] = set()
        interpretation_expected: set[str] = set()
        for source in analysis["source_canonicals"]:
            for target in analysis["target_canonicals"]:
                expected.update(expected_predicates_by_pair.get((source, target), set()))
                core_expected.update(
                    core_predicates_by_pair.get((source, target), set())
                )
                interpretation_expected.update(
                    interpretation_predicates_by_pair.get(
                        (source, target), set()
                    )
                )
        endpoint_supported = any(
            analysis_matches_assertion(
                analysis,
                assertion,
                exact_predicate=False,
                require_evidence=True,
            )
            for assertion in complete_assertions
        )
        unordered_endpoint_supported = any(
            analysis_matches_assertion(
                analysis,
                assertion,
                exact_predicate=False,
                require_evidence=True,
                allow_reverse_endpoint=True,
            )
            for assertion in complete_assertions
        )
        exact_supported = any(
            analysis_matches_assertion(
                analysis,
                assertion,
                exact_predicate=True,
                require_evidence=True,
            )
            for assertion in complete_assertions
        )
        core_exact_supported = any(
            analysis_matches_assertion(
                analysis,
                assertion,
                exact_predicate=True,
                require_evidence=True,
            )
            for assertion in core_assertions
        )
        interpretation_exact_supported = any(
            analysis_matches_assertion(
                analysis,
                assertion,
                exact_predicate=True,
                require_evidence=True,
            )
            for assertion in interpretation_assertions
        )
        audit = {
            "id": analysis["edge"].get("id"),
            "source": analysis["edge"].get("source_title"),
            "predicate": analysis["edge"].get("predicate"),
            "target": analysis["edge"].get("target_title"),
            "document_id": analysis["edge"].get("document_id"),
            "expected_predicates": sorted(expected),
            "core_expected_predicates": sorted(core_expected),
            "interpretation_expected_predicates": sorted(
                interpretation_expected
            ),
            "source_mapped": bool(analysis["source_canonicals"]),
            "target_mapped": bool(analysis["target_canonicals"]),
            "evidence_present": bool(
                analysis["edge"].get("evidence_text")
                or analysis["edge"].get("description")
            ),
        }
        if endpoint_supported:
            endpoint_supported_edges.append(audit)
        else:
            endpoint_unsupported_edges.append(audit)
        if unordered_endpoint_supported:
            unordered_endpoint_supported_edges.append(audit)
        if exact_supported:
            exact_supported_edges.append(audit)
        else:
            unsupported_edges.append(audit)
        if core_exact_supported:
            core_exact_supported_edges.append(audit)
        elif interpretation_exact_supported:
            # A plausible but interpretation-sensitive edge is neither a
            # required hit nor a false positive in the primary score.
            interpretation_only_edges.append(audit)
        else:
            core_evaluated_unsupported_edges.append(audit)

    negative_leaks = []
    for claim in negative_claims:
        hits = assertion_matches(
            claim,
            exact_predicate=True,
            require_evidence=True,
        )
        if hits:
            negative_leaks.append({
                "claim_id": claim["claim_id"],
                "reason_code": claim["reason_code"],
                "edges": hits[:3],
            })

    # A positive relation must not evade an explicit-negation test merely by
    # changing its predicate name (for example ``instance_of`` to an open
    # carrier). Match the same directed endpoints and frozen evidence, then
    # exempt only an edge that is itself an exact, annotated negative-state
    # assertion such as ``lacks_content``. This remains evidence-local and does
    # not turn the benchmark into a closed-world all-pairs test.
    polarity_negative_leaks = []
    for claim in negative_claims:
        if str(claim.get("reason_code") or "") != "EXPLICIT_NEGATION":
            continue
        hits = []
        for analysis in edge_analysis:
            if not analysis_matches_assertion(
                analysis,
                claim,
                exact_predicate=False,
                require_evidence=True,
            ):
                continue
            edge_is_annotated_negative_state = any(
                analysis_matches_assertion(
                    analysis,
                    assertion,
                    exact_predicate=True,
                    require_evidence=True,
                )
                for assertion in complete_assertions
            )
            if not edge_is_annotated_negative_state:
                hits.append(analysis["edge"])
        if hits:
            polarity_negative_leaks.append({
                "claim_id": claim["claim_id"],
                "reason_code": claim["reason_code"],
                "edges": hits[:3],
            })

    def score_by_structure(
        results: list[dict],
    ) -> dict[str, dict[str, int | float | None]]:
        output: dict[str, dict[str, int | float | None]] = {}
        structures = {
            str(item["assertion"]["structure"]) for item in results
        }
        for structure in sorted(structures):
            values = [
                item for item in results
                if str(item["assertion"]["structure"]) == structure
            ]
            endpoint_hits = sum(
                bool(item["endpoint_match"]) for item in values
            )
            exact_hits = sum(bool(item["exact_match"]) for item in values)
            grounded_endpoint_hits = sum(
                bool(item["grounded_endpoint_match"]) for item in values
            )
            grounded_exact_hits = sum(
                bool(item["grounded_exact_match"]) for item in values
            )
            output[structure] = {
                "assertions": len(values),
                "endpoint_hits": endpoint_hits,
                "endpoint_recall": _ratio(endpoint_hits, len(values)),
                "exact_predicate_hits": exact_hits,
                "exact_predicate_recall": _ratio(exact_hits, len(values)),
                "grounded_endpoint_hits": grounded_endpoint_hits,
                "grounded_endpoint_recall": _ratio(
                    grounded_endpoint_hits, len(values)
                ),
                "grounded_exact_predicate_hits": grounded_exact_hits,
                "grounded_exact_predicate_recall": _ratio(
                    grounded_exact_hits, len(values)
                ),
            }
        return output

    by_structure = score_by_structure(assertion_results)
    core_by_structure = score_by_structure(
        score_assertions(core_assertions, deduplicate=False)
    )

    by_schema: dict[str, dict[str, int | float | None]] = {}
    for schema_status in ("base", "extension"):
        values = [
            item for item in assertion_results
            if item["assertion"]["schema_status"] == schema_status
        ]
        endpoint_hits = sum(bool(item["endpoint_match"]) for item in values)
        exact_hits = sum(bool(item["exact_match"]) for item in values)
        by_schema[schema_status] = {
            "assertions": len(values),
            "endpoint_hits": endpoint_hits,
            "endpoint_recall": _ratio(endpoint_hits, len(values)),
            "exact_predicate_hits": exact_hits,
            "exact_predicate_recall": _ratio(exact_hits, len(values)),
        }

    # Historical anchor metrics retained to compare with earlier experiments.
    def pair_endpoints(pair: dict) -> tuple[str, str]:
        document_id = pair["document_revision_id"].removesuffix("@v1")
        return (
            _norm(_surface(document_id, *pair["left_span"])),
            _norm(_surface(document_id, *pair["right_span"])),
        )

    def legacy_matching(pair: dict, *, allow_reverse: bool = False) -> list[dict]:
        left, right = pair_endpoints(pair)
        result = []
        for edge in edges:
            source, target = edge_names(edge)
            if (
                (left in source and right in target)
                or (allow_reverse and right in source and left in target)
            ):
                result.append(edge)
        return result

    legacy_pair_map = {item["pair_id"]: item for item in legacy_pairs}
    legacy_exact_hits = 0
    for fact in legacy_facts:
        pair = legacy_pair_map[fact["pair_id"]]
        legacy_exact_hits += bool([
            edge for edge in legacy_matching(pair)
            if _norm(edge.get("predicate_id") or edge.get("predicate"))
            == _norm(fact["predicate"])
        ])
    legacy_extension_matches = [
        pair for pair in legacy_pairs
        if pair["decision"] == "OTHER_UNSUPPORTED" and legacy_matching(pair)
    ]
    legacy_no_relation_collisions = [
        {"pair_id": pair["pair_id"], "edges": legacy_matching(pair, allow_reverse=True)[:3]}
        for pair in legacy_pairs
        if pair["decision"] == "NO_RELATION"
        and legacy_matching(pair, allow_reverse=True)
    ]

    unique_endpoint_hits = sum(bool(item["endpoint_match"]) for item in unique_results)
    unique_exact_hits = sum(bool(item["exact_match"]) for item in unique_results)
    unique_grounded_endpoint_hits = sum(
        bool(item["grounded_endpoint_match"]) for item in unique_results
    )
    unique_grounded_unordered_endpoint_hits = sum(
        bool(item["grounded_unordered_endpoint_match"])
        for item in unique_results
    )
    unique_grounded_exact_hits = sum(
        bool(item["grounded_exact_match"]) for item in unique_results
    )
    core_endpoint_hits = sum(
        bool(item["endpoint_match"]) for item in core_unique_results
    )
    core_exact_hits = sum(
        bool(item["exact_match"]) for item in core_unique_results
    )
    core_grounded_endpoint_hits = sum(
        bool(item["grounded_endpoint_match"])
        for item in core_unique_results
    )
    core_grounded_exact_hits = sum(
        bool(item["grounded_exact_match"])
        for item in core_unique_results
    )
    interpretation_endpoint_hits = sum(
        bool(item["endpoint_match"])
        for item in interpretation_unique_results
    )
    interpretation_exact_hits = sum(
        bool(item["exact_match"])
        for item in interpretation_unique_results
    )
    interpretation_grounded_endpoint_hits = sum(
        bool(item["grounded_endpoint_match"])
        for item in interpretation_unique_results
    )
    interpretation_grounded_exact_hits = sum(
        bool(item["grounded_exact_match"])
        for item in interpretation_unique_results
    )
    core_evaluated_edge_count = (
        len(core_exact_supported_edges)
        + len(core_evaluated_unsupported_edges)
    )
    report = {
        "status": "calibration_only",
        "dataset": str(manifest.get("dataset") or ""),
        "dataset_root": str(ROOT),
        "graph_nodes": len(graph.get("nodes") or []),
        "graph_edges": len(edges),
        "graph_edges_before_document_slice": len(all_graph_edges),
        "runtime_document_ids": sorted(selected_document_ids),
        "entity_mentions": len(mention_rows),
        "canonical_entities": len(gold_canonical_ids),
        "entity_mention_surface_recall": _ratio(
            len(entity_mention_hits), len(mention_rows)
        ),
        "canonical_entity_surface_coverage": _ratio(
            len(hit_canonical_ids), len(gold_canonical_ids)
        ),
        "canonical_entity_conservative_alias_coverage": _ratio(
            len(alias_aligned_canonical_ids & gold_canonical_ids),
            len(gold_canonical_ids),
        ),
        "canonical_entities_aligned_by_surface_or_unique_alias": len(
            alias_aligned_canonical_ids & gold_canonical_ids
        ),
        "entity_mentions_unmatched": [
            {
                "mention_id": mention.get("mention_id"),
                "surface": mention.get("surface"),
                "document_id": mention.get("document_id"),
            }
            for mention in mention_rows if mention not in entity_mention_hits
        ],
        "complete_assertion_occurrences": len(complete_assertions),
        "complete_semantic_facts": len(unique_results),
        "complete_semantic_fact_endpoint_hits": unique_endpoint_hits,
        "complete_semantic_fact_endpoint_recall": _ratio(
            unique_endpoint_hits, len(unique_results)
        ),
        "complete_semantic_fact_exact_predicate_hits": unique_exact_hits,
        "complete_semantic_fact_exact_predicate_recall": _ratio(
            unique_exact_hits, len(unique_results)
        ),
        "complete_semantic_fact_grounded_endpoint_hits": (
            unique_grounded_endpoint_hits
        ),
        "complete_semantic_fact_grounded_endpoint_recall": _ratio(
            unique_grounded_endpoint_hits, len(unique_results)
        ),
        "complete_semantic_fact_grounded_unordered_endpoint_association_hits": (
            unique_grounded_unordered_endpoint_hits
        ),
        "complete_semantic_fact_grounded_unordered_endpoint_association_recall": _ratio(
            unique_grounded_unordered_endpoint_hits, len(unique_results)
        ),
        "complete_semantic_fact_grounded_exact_predicate_hits": (
            unique_grounded_exact_hits
        ),
        "complete_semantic_fact_grounded_exact_predicate_recall": _ratio(
            unique_grounded_exact_hits, len(unique_results)
        ),
        "primary_core_direct": {
            "precision_requires_matching_evidence": True,
            "assertion_occurrences": len(core_assertions),
            "semantic_facts": len(core_unique_results),
            "semantic_fact_endpoint_hits": core_endpoint_hits,
            "semantic_fact_endpoint_recall": _ratio(
                core_endpoint_hits, len(core_unique_results)
            ),
            "semantic_fact_exact_predicate_hits": core_exact_hits,
            "semantic_fact_exact_predicate_recall": _ratio(
                core_exact_hits, len(core_unique_results)
            ),
            "semantic_fact_grounded_endpoint_hits": (
                core_grounded_endpoint_hits
            ),
            "semantic_fact_grounded_endpoint_recall": _ratio(
                core_grounded_endpoint_hits, len(core_unique_results)
            ),
            "semantic_fact_grounded_exact_predicate_hits": (
                core_grounded_exact_hits
            ),
            "semantic_fact_grounded_exact_predicate_recall": _ratio(
                core_grounded_exact_hits, len(core_unique_results)
            ),
            "graph_edges_evaluated_for_precision": (
                core_evaluated_edge_count
            ),
            "graph_edges_with_exact_supported_predicate": len(
                core_exact_supported_edges
            ),
            "graph_exact_fact_precision": _ratio(
                len(core_exact_supported_edges),
                core_evaluated_edge_count,
            ),
            "interpretation_only_edges_ignored_for_precision": len(
                interpretation_only_edges
            ),
            "recall_by_structure": core_by_structure,
        },
        "interpretation_band_sensitivity": {
            "assertion_occurrences": len(interpretation_assertions),
            "semantic_facts": len(interpretation_unique_results),
            "semantic_fact_endpoint_hits": interpretation_endpoint_hits,
            "semantic_fact_endpoint_recall": _ratio(
                interpretation_endpoint_hits,
                len(interpretation_unique_results),
            ),
            "semantic_fact_exact_predicate_hits": interpretation_exact_hits,
            "semantic_fact_exact_predicate_recall": _ratio(
                interpretation_exact_hits,
                len(interpretation_unique_results),
            ),
            "semantic_fact_grounded_endpoint_hits": (
                interpretation_grounded_endpoint_hits
            ),
            "semantic_fact_grounded_endpoint_recall": _ratio(
                interpretation_grounded_endpoint_hits,
                len(interpretation_unique_results),
            ),
            "semantic_fact_grounded_exact_predicate_hits": (
                interpretation_grounded_exact_hits
            ),
            "semantic_fact_grounded_exact_predicate_recall": _ratio(
                interpretation_grounded_exact_hits,
                len(interpretation_unique_results),
            ),
        },
        "all_label_sensitivity": {
            "precision_requires_matching_evidence": True,
            "assertion_occurrences": len(complete_assertions),
            "semantic_facts": len(unique_results),
            "semantic_fact_endpoint_hits": unique_endpoint_hits,
            "semantic_fact_endpoint_recall": _ratio(
                unique_endpoint_hits, len(unique_results)
            ),
            "semantic_fact_exact_predicate_hits": unique_exact_hits,
            "semantic_fact_exact_predicate_recall": _ratio(
                unique_exact_hits, len(unique_results)
            ),
            "semantic_fact_grounded_endpoint_hits": (
                unique_grounded_endpoint_hits
            ),
            "semantic_fact_grounded_endpoint_recall": _ratio(
                unique_grounded_endpoint_hits, len(unique_results)
            ),
            "semantic_fact_grounded_exact_predicate_hits": (
                unique_grounded_exact_hits
            ),
            "semantic_fact_grounded_exact_predicate_recall": _ratio(
                unique_grounded_exact_hits, len(unique_results)
            ),
            "graph_exact_fact_precision": _ratio(
                len(exact_supported_edges), len(edges)
            ),
        },
        "recall_by_schema_status": by_schema,
        "occurrence_surface_proxy_by_structure": by_structure,
        "graph_edges_with_supported_endpoint_pair": len(endpoint_supported_edges),
        "graph_endpoint_pair_precision": _ratio(
            len(endpoint_supported_edges), len(edges)
        ),
        "graph_edges_with_supported_unordered_endpoint_association": len(
            unordered_endpoint_supported_edges
        ),
        "graph_grounded_unordered_endpoint_association_precision": _ratio(
            len(unordered_endpoint_supported_edges), len(edges)
        ),
        "unsupported_endpoint_edges": endpoint_unsupported_edges,
        "graph_edges_with_exact_supported_predicate": len(exact_supported_edges),
        "graph_exact_fact_precision": _ratio(len(exact_supported_edges), len(edges)),
        "predicate_specific_negative_claims": len(negative_claims),
        "predicate_specific_negative_leaks": len(negative_leaks),
        "negative_claim_leaks": negative_leaks,
        "polarity_endpoint_negative_claims": sum(
            1 for claim in negative_claims
            if str(claim.get("reason_code") or "") == "EXPLICIT_NEGATION"
        ),
        "polarity_endpoint_negative_leaks": len(polarity_negative_leaks),
        "polarity_endpoint_negative_claim_leaks": polarity_negative_leaks,
        "unsupported_or_wrong_predicate_edges": unsupported_edges,
        "unmatched_assertions": [
            {
                "assertion_id": item["assertion"]["assertion_id"],
                "subject": item["assertion"]["subject_canonical_id"],
                "predicate": item["assertion"]["predicate"],
                "object": item["assertion"]["object_canonical_id"],
                "endpoint_match": item["endpoint_match"],
            }
            for item in assertion_results if not item["exact_match"]
        ],
        "historical_anchor": {
            "pair_slice_size": len(legacy_pairs),
            "base_facts": len(legacy_facts),
            "base_exact_hits": legacy_exact_hits,
            "base_exact_recall": _ratio(legacy_exact_hits, len(legacy_facts)),
            "extension_facts": len(legacy_unsupported),
            "extension_pair_hits": len(legacy_extension_matches),
            "extension_pair_recall": _ratio(
                len(legacy_extension_matches), len(legacy_unsupported)
            ),
            "local_no_relation_surface_collisions_non_gating": (
                legacy_no_relation_collisions
            ),
        },
        "production_gate_eligible": bool(
            manifest.get("production_gate_eligible")
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
