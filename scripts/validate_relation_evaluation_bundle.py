#!/usr/bin/env python3
"""Independently validate a Knowledge Forge relation evaluation bundle.

The exporter writes summaries for convenience; this validator deliberately
recomputes the core audit checks from raw CSV/JSONL files so a stale or forged
summary cannot make an incomplete bundle look valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "run-summary.json",
    "structure-contexts.jsonl",
    "source-maps.jsonl",
    "model-calls.jsonl",
    "pair-ledger.csv",
    "classifier-attempts.jsonl",
    "hypotheses.csv",
    "verifications.csv",
    "normalized-facts-with-evidence.csv",
    "fact-evidence.jsonl",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{line_number}: invalid JSONL") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: record must be object")
            rows.append(value)
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def validate_relation_evaluation_bundle(bundle_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    for name in REQUIRED_FILES:
        path = bundle_dir / name
        if not path.exists():
            failures.append(f"missing_file:{name}")

    if failures:
        return {"ok": False, "failures": failures, "counts": {}}

    run = _read_json(bundle_dir / "run-summary.json")
    source_contexts = _read_jsonl(bundle_dir / "structure-contexts.jsonl")
    source_maps = _read_jsonl(bundle_dir / "source-maps.jsonl")
    model_calls = _read_jsonl(bundle_dir / "model-calls.jsonl")
    pairs = _read_csv(bundle_dir / "pair-ledger.csv")
    classifier_attempts = _read_jsonl(bundle_dir / "classifier-attempts.jsonl")
    hypotheses = _read_csv(bundle_dir / "hypotheses.csv")
    verifications = _read_csv(bundle_dir / "verifications.csv")
    facts = _read_csv(bundle_dir / "normalized-facts-with-evidence.csv")
    evidence = _read_jsonl(bundle_dir / "fact-evidence.jsonl")

    if pairs and not source_contexts:
        failures.append("pair_ledger_without_source_contexts")
    if source_contexts and not source_maps:
        failures.append("source_contexts_without_source_maps")
    if facts and not evidence:
        failures.append("facts_without_fact_evidence")
    if hypotheses and not verifications:
        failures.append("hypotheses_without_verifications")
    if pairs and not classifier_attempts:
        failures.append("pairs_without_classifier_attempts")
    if (pairs or classifier_attempts or hypotheses) and not model_calls:
        failures.append("model_ledgers_without_model_calls")

    pair_ids = {str(row.get("id") or "") for row in pairs}
    pair_ids.discard("")
    for row in hypotheses:
        pair_identity = str(row.get("pair_identity") or "")
        if not pair_identity:
            failures.append(f"hypothesis_missing_pair_identity:{row.get('id')}")

    latest_verification_by_hypothesis: dict[str, dict[str, Any]] = {}
    for row in verifications:
        key = str(row.get("hypothesis_key") or "")
        if not key:
            failures.append(f"verification_missing_hypothesis_key:{row.get('id')}")
            continue
        current = latest_verification_by_hypothesis.get(key)
        try:
            attempt_no = int(row.get("attempt_no") or 0)
            current_attempt_no = int((current or {}).get("attempt_no") or 0)
        except ValueError:
            failures.append(f"verification_bad_attempt_no:{row.get('id')}")
            continue
        if current is None or attempt_no > current_attempt_no:
            latest_verification_by_hypothesis[key] = row

    supported_verification_ids = {
        str(row.get("id") or "")
        for row in latest_verification_by_hypothesis.values()
        if str(row.get("verdict") or "") == "SUPPORTED"
        and _truthy(row.get("independent_call"))
        and str(row.get("supported_evidence_spans") or "").strip()
        and str(row.get("supported_evidence_spans") or "").strip() != "[]"
    }
    supported_verification_ids.discard("")

    for row in evidence:
        verification_attempt_id = str(row.get("verification_attempt_id") or "")
        if verification_attempt_id not in supported_verification_ids:
            failures.append(
                "fact_evidence_without_supported_independent_verification:"
                f"{row.get('id') or row.get('evidence_key') or verification_attempt_id}"
            )
        if not str(row.get("evidence_spans") or "").strip():
            failures.append(
                f"fact_evidence_missing_spans:{row.get('id') or row.get('evidence_key')}"
            )

    eligible_facts = [
        row for row in facts
        if str(row.get("membership_status") or "") == "ELIGIBLE"
    ]
    evidence_fact_ids = {str(row.get("fact_id") or "") for row in evidence}
    for row in eligible_facts:
        if str(row.get("id") or "") not in evidence_fact_ids:
            failures.append(f"eligible_fact_without_evidence:{row.get('id')}")

    accepted_empty = [
        str(row.get("id") or "")
        for row in classifier_attempts
        if str(row.get("status") or "") in {"ACCEPTED", "accepted"}
        and not str(row.get("raw_response_text") or "").strip()
    ]
    if accepted_empty:
        failures.append("accepted_classifier_empty_response:" + ",".join(accepted_empty))

    if str(run.get("status") or "") == "ready":
        retryable_latest = [
            str(row.get("id") or "")
            for row in latest_verification_by_hypothesis.values()
            if str(row.get("verdict") or "") == "RETRYABLE_FAILED"
        ]
        if retryable_latest:
            failures.append(
                "ready_run_with_terminal_verifier_failure:"
                + ",".join(retryable_latest)
            )

    counts = {
        "source_contexts": len(source_contexts),
        "source_maps": len(source_maps),
        "pairs": len(pairs),
        "model_calls": len(model_calls),
        "classifier_attempts": len(classifier_attempts),
        "hypotheses": len(hypotheses),
        "verifications": len(verifications),
        "facts": len(facts),
        "fact_evidence": len(evidence),
        "eligible_facts": len(eligible_facts),
    }
    return {"ok": not failures, "failures": failures, "counts": counts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a Knowledge Forge relation evaluation bundle",
    )
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args(argv)
    result = validate_relation_evaluation_bundle(args.bundle_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
