from __future__ import annotations

import csv
import json
from pathlib import Path

from knowledge_forge.quality import evaluate_shadow_relation_quality


ARTIFACT_ROOT = Path(
    "artifacts/shadow-v2/project-13/2026-07-27-native-v2"
)


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (ARTIFACT_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_native_v2_exported_artifact_bundle_has_reconciled_ledgers() -> None:
    summary = json.loads(
        (ARTIFACT_ROOT / "run-summary.json").read_text(encoding="utf-8")
    )
    facts = _csv_rows("normalized-facts-with-evidence.csv")
    hypotheses = _csv_rows("hypotheses.csv")
    verifications = _csv_rows("verifications.csv")
    pair_ledger = _csv_rows("pair-ledger.csv")

    counts = summary["counts"]
    assert counts["pair_opportunities"] == len(pair_ledger)
    assert counts["hypotheses"] == len(hypotheses)
    assert counts["verification_attempts"] == len(verifications)
    assert counts["normalized_facts"] == len(facts) == 33

    fact_check = summary["fact_supported_evidence_check"]
    assert fact_check == {
        "evidence_rows": 33,
        "facts": 33,
        "supported_facts": 33,
    }
    assert {row["verifier_verdict"] for row in facts} == {"SUPPORTED"}


def test_supported_verification_ratio_is_not_a_semantic_quality_gate() -> None:
    facts = _csv_rows("normalized-facts-with-evidence.csv")
    report = evaluate_shadow_relation_quality(
        gold=None,
        predicted=facts,
        snapshot_current=True,
    )

    assert report["gold_status"] == "unavailable"
    assert report["canonical"]["predicted"] == 33
    assert report["canonical"]["precision"] is None
    assert report["canonical"]["recall"] is None
    assert report["production_eligible"] is False
    assert "human_gold_unavailable" in report["blocking_reasons"]


def test_artifact_response_quality_distinguishes_no_relation_from_empty_response() -> None:
    response_quality = json.loads(
        (ARTIFACT_ROOT / "response-quality.json").read_text(encoding="utf-8")
    )
    pair_status = {
        row["status"]: int(row["count"])
        for row in _csv_rows("pair-status-summary.csv")
    }

    assert pair_status["NO_RELATION"] == 691
    assert response_quality["observable_empty_model_response_count"] == 0
    assert response_quality["classifier"]["error_attempts"] == 0
    assert response_quality["classifier"]["missing_response_hash"] == 0
    assert response_quality["verifier"]["error_attempts"] == 0
    assert response_quality["verifier"]["missing_response_hash"] == 0
