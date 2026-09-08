from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.calibration_policy import (
    partition_assertions,
    unique_fact_rows,
)
from benchmarks.knowledge_forge_cross_domain_v1.score_graph import (
    _conservative_alias_overlap,
)


ROOT = Path(__file__).resolve().parents[1]
SCORER = (
    ROOT
    / "benchmarks/knowledge_forge_cross_domain_v1/score_graph.py"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _complete_gold_graph(dataset_root: Path) -> dict:
    manifest = _load(dataset_root / "manifest.json")
    annotation_paths = manifest["annotations"]
    entities = _load(
        dataset_root / str(annotation_paths["entities"])
    )["documents"]
    assertions = _load(
        dataset_root / str(annotation_paths["assertions"])
    )["relations"]
    title_by_canonical: dict[str, str] = {}
    for mentions in entities.values():
        for mention in mentions:
            title_by_canonical.setdefault(
                str(mention["canonical_id"]), str(mention["surface"])
            )
    facts = unique_fact_rows(assertions)
    return {
        "nodes": [
            {"id": canonical_id, "title": title}
            for canonical_id, title in sorted(title_by_canonical.items())
        ],
        "edges": [
            {
                "id": f"e{index}",
                "source": fact[0],
                "predicate_id": fact[1],
                "target": fact[2],
                "document_id": row["document_id"],
                "evidence_text": row["evidence_text"],
            }
            for index, (fact, row) in enumerate(sorted(facts.items()))
        ],
    }


def _score(
    tmp_path: Path,
    dataset_root: Path,
    graph: dict | None = None,
) -> dict:
    graph_path = tmp_path / f"{dataset_root.name}-graph.json"
    graph_path.write_text(
        json.dumps(
            graph or _complete_gold_graph(dataset_root),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCORER),
            "--dataset-root",
            str(dataset_root),
            "--graph-json",
            str(graph_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("relative", "core_facts", "interpretation_facts", "all_facts"),
    [
        ("knowledge_forge_cross_domain_v1", 69, 9, 78),
        ("knowledge_forge_cross_domain_zh_v1", 61, 2, 63),
    ],
)
def test_graph_scorer_separates_primary_and_interpretation_bands(
    tmp_path: Path,
    relative: str,
    core_facts: int,
    interpretation_facts: int,
    all_facts: int,
):
    dataset_root = ROOT / "benchmarks" / relative

    report = _score(tmp_path, dataset_root)

    primary = report["primary_core_direct"]
    sensitivity = report["interpretation_band_sensitivity"]
    assert primary["semantic_facts"] == core_facts
    assert primary["semantic_fact_exact_predicate_recall"] == 1.0
    assert primary["semantic_fact_grounded_endpoint_recall"] == 1.0
    assert primary[
        "semantic_fact_grounded_exact_predicate_recall"
    ] == 1.0
    assert primary["graph_exact_fact_precision"] == 1.0
    assert primary[
        "interpretation_only_edges_ignored_for_precision"
    ] == interpretation_facts
    assert sensitivity["semantic_facts"] == interpretation_facts
    assert sensitivity["semantic_fact_exact_predicate_recall"] == 1.0
    assert report["all_label_sensitivity"]["semantic_facts"] == all_facts


def test_calibration_policy_rejects_unknown_sidecar_assertion(tmp_path: Path):
    tier_path = tmp_path / "annotations"
    tier_path.mkdir()
    (tier_path / "adjudication_tiers_2026_08_11.json").write_text(
        json.dumps({
            "interpretation_band": [{"assertion_id": "missing"}],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown assertions"):
        partition_assertions(tmp_path, [{
            "assertion_id": "known",
            "subject_canonical_id": "a",
            "predicate": "p",
            "object_canonical_id": "b",
        }])


def test_graph_scorer_requires_matching_evidence_for_fact_credit(
    tmp_path: Path,
):
    dataset_root = (
        ROOT / "benchmarks/knowledge_forge_cross_domain_v1"
    )
    graph = _complete_gold_graph(dataset_root)
    target = next(
        edge for edge in graph["edges"]
        if edge["source"] == "py:if_statement"
        and edge["predicate_id"] == "used_for"
        and edge["target"] == "py:conditional_execution"
    )
    target["evidence_text"] = "Unrelated evidence."

    report = _score(tmp_path, dataset_root, graph)
    primary = report["primary_core_direct"]

    assert primary["semantic_fact_exact_predicate_recall"] == 1.0
    assert primary["semantic_fact_grounded_exact_predicate_hits"] == 68
    assert primary["semantic_fact_grounded_exact_predicate_recall"] == 68 / 69
    assert primary["graph_exact_fact_precision"] == 68 / 69


def test_graph_scorer_detects_cross_predicate_explicit_negation_leak(
    tmp_path: Path,
):
    dataset_root = (
        ROOT / "benchmarks/knowledge_forge_cross_domain_v1"
    )
    graph = _complete_gold_graph(dataset_root)
    graph["edges"].append({
        "id": "negated-fact-with-open-predicate",
        "source": "nasa:utm_software",
        "predicate_id": "source_assertion",
        "target": "nasa:operational_system",
        "document_id": "engineering-nasa-utm-requirements-excerpt",
        "evidence_text": (
            "UTM TCL 2 is a research prototype. It is a cloud-ready "
            "research platform, not an operational system."
        ),
    })

    report = _score(tmp_path, dataset_root, graph)

    # Predicate-specific scoring alone cannot see this leak because the model
    # renamed ``instance_of`` to an open carrier.  The polarity audit must use
    # the frozen evidence_span from the negative annotation and still reject
    # the same directed endpoint claim.
    assert report["predicate_specific_negative_leaks"] == 0
    assert report["polarity_endpoint_negative_claims"] == 8
    assert report["polarity_endpoint_negative_leaks"] == 1
    assert [
        item["claim_id"]
        for item in report["polarity_endpoint_negative_claim_leaks"]
    ] == ["nasa-n01"]


def test_graph_scorer_reports_grounded_inverse_endpoint_as_association(
    tmp_path: Path,
):
    dataset_root = (
        ROOT / "benchmarks/knowledge_forge_cross_domain_v1"
    )
    graph = _complete_gold_graph(dataset_root)
    target = next(
        edge for edge in graph["edges"]
        if edge["source"] == "net:resource"
        and edge["predicate_id"] == "is_target_of"
        and edge["target"] == "net:http_request"
    )
    target.update({
        "source": "net:http_request",
        "predicate_id": "targets",
        "target": "net:resource",
    })

    report = _score(tmp_path, dataset_root, graph)

    assert report["complete_semantic_fact_grounded_endpoint_recall"] == 77 / 78
    assert report[
        "complete_semantic_fact_grounded_unordered_endpoint_association_recall"
    ] == 1.0
    assert report[
        "graph_grounded_unordered_endpoint_association_precision"
    ] == 1.0


def test_graph_scorer_alias_overlap_is_nested_and_boundary_safe():
    assert _conservative_alias_overlap("sister", "her sister on the bank")
    assert _conservative_alias_overlap("软件需求", "核心软件需求模块")
    assert not _conservative_alias_overlap("if", "if statement")
    assert not _conservative_alias_overlap("bank", "banking")
