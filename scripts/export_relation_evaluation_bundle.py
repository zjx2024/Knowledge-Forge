#!/usr/bin/env python3
"""Export an auditable Knowledge Forge relation evaluation bundle.

This script is intentionally read-only.  It reconstructs the grounded relation
ledger from PostgreSQL and writes deterministic JSON/CSV artifacts plus a
machine-readable audit summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from knowledge_forge.relation_runtime import relation_model_payload_hash


CSV_COLUMNS = {
    "mentions.csv": [
        "namespace_id", "mention_key", "project_id", "document_id", "revision_id",
        "chunk_id", "surface", "normalized_surface", "evidence_start",
        "evidence_end", "filter_decision", "filter_reason",
    ],
    "mention-appearances.csv": [
        "namespace_id", "mention_key", "occurrence_id", "occurrence_key",
        "mapping_policy_version",
    ],
    "occurrences.csv": [
        "id", "namespace_id", "project_id", "document_id", "document_revision_id",
        "chunk_id", "occurrence_key", "physical_occurrence_key", "surface",
        "normalized_surface", "evidence_start", "evidence_end", "source_context_id",
    ],
    "entities.csv": [
        "namespace_id", "canonical_key", "name", "normalized_name", "primary_type",
        "source_count", "generation",
    ],
    "resolution-outcomes.csv": [
        "namespace_id", "source_canonical_key", "target_canonical_key",
        "redirect_policy_version", "reason_code",
    ],
    "assignments.csv": [
        "occurrence_id", "namespace_id", "canonical_key", "assignment_version",
        "status", "confidence", "decision_reason",
    ],
    "proposals.csv": [
        "id", "proposal_attempt_id", "pair_opportunity_id", "hypothesis_id",
        "proposal_key", "predicate", "subject_occurrence_key",
        "object_occurrence_key", "evidence_spans", "qualifiers",
        "confidence", "status", "reason_code",
    ],
    "pair-contexts.csv": [
        "pair_opportunity_id", "context_id", "context_role", "payload",
    ],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _rows(conn: psycopg.Connection, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def _table_exists(conn: psycopg.Connection, table: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s) AS name", (table,)).fetchone()
    return bool(row and row.get("name"))


def _optional_rows(
    conn: psycopg.Connection,
    table: str,
    query: str,
    params: tuple[Any, ...],
) -> list[dict[str, Any]]:
    if not _table_exists(conn, table):
        return []
    return _rows(conn, query, params)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: (
                    _canonical_json(row.get(column))
                    if isinstance(row.get(column), (dict, list, tuple))
                    else row.get(column)
                )
                for column in columns
            })


def _count_by(rows: Iterable[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [
        {field: key, "count": count}
        for key, count in sorted(Counter(str(row.get(field) or "") for row in rows).items())
    ]


def _duplicates(rows: Iterable[dict[str, Any]], field: str) -> list[str]:
    counts = Counter(str(row.get(field) or "") for row in rows)
    return sorted(key for key, count in counts.items() if key and count > 1)


def export_relation_evaluation_bundle(
    conn: psycopg.Connection,
    *,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    run = conn.execute(
        "SELECT * FROM shadow_relation_v2_runs WHERE id=%s",
        (run_id,),
    ).fetchone()
    if not run:
        raise LookupError(f"Knowledge Forge relation run not found: {run_id}")
    run = dict(run)
    pairs = _rows(
        conn,
        """SELECT * FROM shadow_relation_pair_opportunities
           WHERE run_id=%s ORDER BY pair_identity""",
        (run_id,),
    )
    classifier_attempts = _rows(
        conn,
        """SELECT attempt.*,pair.pair_identity
           FROM shadow_relation_classification_attempts attempt
           JOIN shadow_relation_pair_opportunities pair
             ON pair.id=attempt.pair_opportunity_id
           WHERE pair.run_id=%s
           ORDER BY pair.pair_identity,attempt.attempt_no""",
        (run_id,),
    )
    hypotheses = _rows(
        conn,
        """SELECT hypothesis.*,pair.pair_identity
           FROM shadow_relation_hypotheses hypothesis
           JOIN shadow_relation_pair_opportunities pair
             ON pair.id=hypothesis.pair_opportunity_id
           WHERE pair.run_id=%s ORDER BY hypothesis.hypothesis_key""",
        (run_id,),
    )
    validations = _rows(
        conn,
        """SELECT validation.*,hypothesis.hypothesis_key
           FROM shadow_relation_hypothesis_validations validation
           JOIN shadow_relation_hypotheses hypothesis
             ON hypothesis.id=validation.hypothesis_id
           JOIN shadow_relation_pair_opportunities pair
             ON pair.id=hypothesis.pair_opportunity_id
           WHERE pair.run_id=%s ORDER BY hypothesis.hypothesis_key""",
        (run_id,),
    )
    verifications = _rows(
        conn,
        """SELECT verification.*,hypothesis.hypothesis_key
           FROM shadow_relation_verification_attempts verification
           JOIN shadow_relation_hypotheses hypothesis
             ON hypothesis.id=verification.hypothesis_id
           JOIN shadow_relation_pair_opportunities pair
             ON pair.id=hypothesis.pair_opportunity_id
           WHERE pair.run_id=%s
           ORDER BY hypothesis.hypothesis_key,verification.attempt_no""",
        (run_id,),
    )
    facts = _rows(
        conn,
        """SELECT fact.*,membership.status AS membership_status,
                  membership.input_fingerprint
           FROM shadow_relation_fact_memberships membership
           JOIN shadow_relation_normalized_facts fact
             ON fact.id=membership.fact_id
           WHERE membership.run_id=%s ORDER BY fact.semantic_fact_key""",
        (run_id,),
    )
    evidence = _rows(
        conn,
        """SELECT evidence.*,fact.semantic_fact_key
           FROM shadow_relation_fact_evidence evidence
           JOIN shadow_relation_normalized_facts fact ON fact.id=evidence.fact_id
           WHERE evidence.run_id=%s ORDER BY evidence.evidence_key""",
        (run_id,),
    )
    conflicts = _rows(
        conn,
        """SELECT * FROM shadow_relation_fact_conflicts
           WHERE run_id=%s ORDER BY id""",
        (run_id,),
    )
    collapse = _rows(
        conn,
        """SELECT * FROM shadow_relation_fact_collapse_audit
           WHERE run_id=%s ORDER BY id""",
        (run_id,),
    )
    source_contexts = _optional_rows(
        conn,
        "shadow_source_contexts",
        """SELECT * FROM shadow_source_contexts
           WHERE run_id=%s ORDER BY context_identity,id""",
        (run_id,),
    )
    source_maps = _optional_rows(
        conn,
        "shadow_context_source_maps",
        """SELECT source_map.*
           FROM shadow_context_source_maps source_map
           JOIN shadow_source_contexts context ON context.id=source_map.context_id
           WHERE context.run_id=%s
           ORDER BY source_map.context_id,source_map.source_kind,
                    source_map.source_record_id,source_map.source_start""",
        (run_id,),
    )
    mentions = _optional_rows(
        conn,
        "entity_shadow_mentions",
        """SELECT mention.*
           FROM entity_shadow_mentions mention
           WHERE EXISTS (
             SELECT 1 FROM entity_shadow_mention_occurrence_map mapping
             JOIN entity_shadow_occurrences occurrence
               ON occurrence.id=mapping.occurrence_id
             JOIN shadow_source_contexts context
               ON context.id=occurrence.source_context_id
             WHERE context.run_id=%s
               AND mapping.namespace_id=mention.namespace_id
               AND mapping.mention_key=mention.mention_key
           )
           ORDER BY mention.namespace_id,mention.mention_key""",
        (run_id,),
    )
    occurrences = _optional_rows(
        conn,
        "entity_shadow_occurrences",
        """SELECT occurrence.*
           FROM entity_shadow_occurrences occurrence
           JOIN shadow_source_contexts context
             ON context.id=occurrence.source_context_id
           WHERE context.run_id=%s
           ORDER BY occurrence.namespace_id,occurrence.occurrence_key""",
        (run_id,),
    )
    mention_appearances = _optional_rows(
        conn,
        "entity_shadow_mention_occurrence_map",
        """SELECT mapping.*,occurrence.occurrence_key
           FROM entity_shadow_mention_occurrence_map mapping
           JOIN entity_shadow_occurrences occurrence
             ON occurrence.id=mapping.occurrence_id
           JOIN shadow_source_contexts context
             ON context.id=occurrence.source_context_id
           WHERE context.run_id=%s
           ORDER BY mapping.namespace_id,mapping.mention_key""",
        (run_id,),
    )
    assignments = _optional_rows(
        conn,
        "entity_shadow_occurrence_assignments",
        """SELECT assignment.*
           FROM entity_shadow_occurrence_assignments assignment
           JOIN entity_shadow_occurrences occurrence
             ON occurrence.id=assignment.occurrence_id
           JOIN shadow_source_contexts context
             ON context.id=occurrence.source_context_id
           WHERE context.run_id=%s
           ORDER BY assignment.namespace_id,assignment.canonical_key,
                    assignment.occurrence_id""",
        (run_id,),
    )
    entities = _optional_rows(
        conn,
        "entity_shadow_canonicals",
        """SELECT DISTINCT canonical.*
           FROM entity_shadow_canonicals canonical
           JOIN entity_shadow_occurrence_assignments assignment
             ON assignment.namespace_id=canonical.namespace_id
            AND assignment.canonical_key=canonical.canonical_key
           JOIN entity_shadow_occurrences occurrence
             ON occurrence.id=assignment.occurrence_id
           JOIN shadow_source_contexts context
             ON context.id=occurrence.source_context_id
           WHERE context.run_id=%s
           ORDER BY canonical.namespace_id,canonical.canonical_key""",
        (run_id,),
    )
    resolution_outcomes = _optional_rows(
        conn,
        "entity_shadow_canonical_redirects",
        """SELECT DISTINCT redirect.*
           FROM entity_shadow_canonical_redirects redirect
           JOIN entity_shadow_occurrence_assignments assignment
             ON assignment.namespace_id=redirect.namespace_id
            AND assignment.canonical_key IN (
                redirect.source_canonical_key,redirect.target_canonical_key
            )
           JOIN entity_shadow_occurrences occurrence
             ON occurrence.id=assignment.occurrence_id
           JOIN shadow_source_contexts context
             ON context.id=occurrence.source_context_id
           WHERE context.run_id=%s
           ORDER BY redirect.namespace_id,redirect.source_canonical_key""",
        (run_id,),
    )
    model_calls = _optional_rows(
        conn,
        "shadow_model_calls",
        """SELECT * FROM shadow_model_calls
           WHERE run_id=%s ORDER BY component,idempotency_key,attempt_no,id""",
        (run_id,),
    )
    proposal_attempts = _optional_rows(
        conn,
        "shadow_relation_proposal_attempts",
        """SELECT * FROM shadow_relation_proposal_attempts
           WHERE run_id=%s ORDER BY context_id,attempt_no,id""",
        (run_id,),
    )
    proposals = _optional_rows(
        conn,
        "shadow_relation_proposals",
        """SELECT proposal.*
           FROM shadow_relation_proposals proposal
           JOIN shadow_relation_proposal_attempts attempt
             ON attempt.id=proposal.proposal_attempt_id
           WHERE attempt.run_id=%s
           ORDER BY proposal.proposal_attempt_id,proposal.proposal_key""",
        (run_id,),
    )
    pair_contexts = _optional_rows(
        conn,
        "shadow_relation_pair_contexts",
        """SELECT pair_context.*
           FROM shadow_relation_pair_contexts pair_context
           JOIN shadow_relation_pair_opportunities pair
             ON pair.id=pair_context.pair_opportunity_id
           WHERE pair.run_id=%s
           ORDER BY pair_context.pair_opportunity_id,
                    pair_context.context_role,pair_context.context_id""",
        (run_id,),
    )
    evidence_refs = _optional_rows(
        conn,
        "shadow_relation_evidence_refs",
        """SELECT * FROM shadow_relation_evidence_refs
           WHERE run_id=%s ORDER BY fact_evidence_id,source_kind,source_record_id""",
        (run_id,),
    )

    latest_verification_by_hypothesis: dict[str, dict[str, Any]] = {}
    for row in verifications:
        current = latest_verification_by_hypothesis.get(str(row["hypothesis_id"]))
        if current is None or int(row["attempt_no"]) > int(current["attempt_no"]):
            latest_verification_by_hypothesis[str(row["hypothesis_id"])] = row
    supported_attempt_ids = {
        str(row["id"])
        for row in latest_verification_by_hypothesis.values()
        if row.get("verdict") == "SUPPORTED"
        and bool(row.get("independent_call"))
        and bool(row.get("supported_evidence_spans"))
    }
    evidence_attempt_ids = {str(row["verification_attempt_id"]) for row in evidence}
    facts_with_evidence = {
        str(row["fact_id"]) for row in evidence
        if str(row["verification_attempt_id"]) in supported_attempt_ids
    }
    fact_rows_with_evidence = [
        {
            **fact,
            "evidence_count": sum(
                1 for row in evidence if str(row["fact_id"]) == str(fact["id"])
            ),
        }
        for fact in facts
    ]
    hash_mismatches = []
    for attempt in classifier_attempts:
        request_payload = attempt.get("request_payload") or {}
        raw_response_text = str(attempt.get("raw_response_text") or "")
        if attempt.get("request_hash"):
            recomputed = relation_model_payload_hash("request", request_payload)
            if recomputed != str(attempt["request_hash"]):
                hash_mismatches.append({
                    "attempt_id": attempt["id"],
                    "kind": "request",
                    "expected": attempt["request_hash"],
                    "actual": recomputed,
                })
        if attempt.get("response_hash") and raw_response_text:
            recomputed = relation_model_payload_hash("response", raw_response_text)
            if recomputed != str(attempt["response_hash"]):
                hash_mismatches.append({
                    "attempt_id": attempt["id"],
                    "kind": "response",
                    "expected": attempt["response_hash"],
                    "actual": recomputed,
                })

    pair_status_summary = _count_by(pairs, "status")
    verification_verdict_summary = _count_by(
        latest_verification_by_hypothesis.values(), "verdict",
    )
    audit_failures = []
    for name, values in {
        "pair_id": _duplicates(pairs, "id"),
        "pair_identity": _duplicates(pairs, "pair_identity"),
        "hypothesis_id": _duplicates(hypotheses, "id"),
        "hypothesis_key": _duplicates(hypotheses, "hypothesis_key"),
        "verification_id": _duplicates(verifications, "id"),
        "fact_id": _duplicates(facts, "id"),
        "evidence_key": _duplicates(evidence, "evidence_key"),
    }.items():
        if values:
            audit_failures.append({"check": f"duplicate_{name}", "values": values})
    missing_supported = sorted(evidence_attempt_ids - supported_attempt_ids)
    if missing_supported:
        audit_failures.append({
            "check": "fact_evidence_without_supported_verification",
            "values": missing_supported,
        })
    eligible_fact_ids = {
        str(row["id"]) for row in facts
        if str(row.get("membership_status") or "") == "ELIGIBLE"
    }
    missing_fact_evidence = sorted(eligible_fact_ids - facts_with_evidence)
    if missing_fact_evidence:
        audit_failures.append({
            "check": "eligible_fact_without_supported_evidence",
            "values": missing_fact_evidence,
        })
    if hash_mismatches:
        audit_failures.append({
            "check": "classification_attempt_hash_mismatch",
            "values": hash_mismatches,
        })
    accepted_empty_responses = [
        row["id"] for row in classifier_attempts
        if str(row.get("status")) in {"ACCEPTED", "accepted"}
        and not str(row.get("raw_response_text") or "")
    ]
    if accepted_empty_responses:
        audit_failures.append({
            "check": "accepted_classifier_empty_response",
            "values": accepted_empty_responses,
        })
    terminal_failures = [
        row for row in latest_verification_by_hypothesis.values()
        if row.get("verdict") == "RETRYABLE_FAILED"
    ]
    if terminal_failures and str(run.get("status")) == "ready":
        audit_failures.append({
            "check": "ready_run_with_terminal_verifier_failure",
            "values": [row["id"] for row in terminal_failures],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "structure-contexts.jsonl", source_contexts)
    _write_jsonl(output_dir / "source-maps.jsonl", source_maps)
    _write_csv(output_dir / "mentions.csv", mentions, CSV_COLUMNS["mentions.csv"])
    _write_csv(
        output_dir / "mention-appearances.csv",
        mention_appearances,
        CSV_COLUMNS["mention-appearances.csv"],
    )
    _write_csv(output_dir / "occurrences.csv", occurrences, CSV_COLUMNS["occurrences.csv"])
    _write_csv(output_dir / "entities.csv", entities, CSV_COLUMNS["entities.csv"])
    _write_csv(
        output_dir / "resolution-outcomes.csv",
        resolution_outcomes,
        CSV_COLUMNS["resolution-outcomes.csv"],
    )
    _write_csv(output_dir / "assignments.csv", assignments, CSV_COLUMNS["assignments.csv"])
    _write_jsonl(output_dir / "model-calls.jsonl", model_calls)
    _write_jsonl(output_dir / "proposal-attempts.jsonl", proposal_attempts)
    _write_csv(output_dir / "proposals.csv", proposals, CSV_COLUMNS["proposals.csv"])
    _write_csv(output_dir / "pair-contexts.csv", pair_contexts, CSV_COLUMNS["pair-contexts.csv"])
    _write_json(output_dir / "freeze-manifest.json", {
        "run_id": run_id,
        "project_id": run.get("project_id"),
        "snapshot_id": run.get("snapshot_id"),
        "input_fingerprint": run.get("input_fingerprint"),
        "fingerprint_components": run.get("fingerprint_components") or {},
    })
    _write_json(output_dir / "run-summary.json", run)
    _write_csv(
        output_dir / "pair-ledger.csv",
        pairs,
        [
            "id", "pair_identity", "document_revision_id", "chunk_id",
            "left_occurrence_key", "right_occurrence_key", "status",
            "planning_reason", "type_status", "context_id", "source_kind",
            "source_record_id", "pipeline_generation", "provenance",
        ],
    )
    _write_csv(output_dir / "pair-status-summary.csv", pair_status_summary, ["status", "count"])
    _write_jsonl(output_dir / "classifier-attempts.jsonl", classifier_attempts)
    _write_csv(
        output_dir / "hypotheses.csv",
        hypotheses,
        [
            "id", "hypothesis_key", "pair_identity", "predicate",
            "subject_occurrence_key", "object_occurrence_key", "status",
            "classifier_policy_version", "classifier_model_revision",
            "source_kind", "source_record_id", "qualifiers", "metadata",
        ],
    )
    _write_csv(
        output_dir / "verifications.csv",
        verifications,
        [
            "id", "hypothesis_key", "attempt_no", "verdict", "reason_code",
            "independent_call", "supported_evidence_spans",
            "verifier_prompt_version", "verifier_model_revision",
        ],
    )
    _write_csv(
        output_dir / "verification-verdict-summary.csv",
        verification_verdict_summary,
        ["verdict", "count"],
    )
    _write_csv(
        output_dir / "normalized-facts-with-evidence.csv",
        fact_rows_with_evidence,
        [
            "id", "semantic_fact_key", "subject_canonical_id", "predicate",
            "object_canonical_id", "is_review_only", "membership_status",
            "qualifiers", "audit_quality", "evidence_count",
        ],
    )
    _write_jsonl(output_dir / "fact-evidence.jsonl", evidence)
    qualifiers = [
        {
            "source_table": "shadow_relation_hypotheses",
            "source_id": row["id"],
            "qualifiers": row.get("qualifiers"),
        }
        for row in hypotheses
        if row.get("qualifiers") not in ({}, None)
    ] + [
        {
            "source_table": "shadow_relation_normalized_facts",
            "source_id": row["id"],
            "qualifiers": row.get("qualifiers"),
        }
        for row in facts
        if row.get("qualifiers") not in ({}, None)
    ] + [
        {
            "source_table": "shadow_relation_fact_evidence",
            "source_id": row["id"],
            "qualifiers": row.get("qualifiers"),
        }
        for row in evidence
        if row.get("qualifiers") not in ({}, None)
    ] + [
        {
            "source_table": "shadow_relation_proposals",
            "source_id": row["id"],
            "qualifiers": row.get("qualifiers"),
        }
        for row in proposals
        if row.get("qualifiers") not in ({}, None)
    ] + [
        {
            "source_table": "shadow_relation_evidence_refs",
            "source_id": row["id"],
            "qualifiers": row.get("qualifiers"),
        }
        for row in evidence_refs
        if row.get("qualifiers") not in ({}, None)
    ]
    _write_jsonl(output_dir / "qualifiers.jsonl", qualifiers)
    _write_csv(
        output_dir / "conflicts.csv",
        conflicts,
        ["id", "pair_identity", "hypothesis_id", "reason_code", "metadata"],
    )
    _write_jsonl(output_dir / "collapse-audit.jsonl", collapse)
    response_quality = {
        "classification_attempts": len(classifier_attempts),
        "hash_recompute_mismatches": hash_mismatches,
        "empty_response_attempts": sum(
            1 for row in classifier_attempts
            if not str(row.get("raw_response_text") or "")
        ),
        "accepted_empty_responses": sum(
            1 for row in classifier_attempts
            if str(row.get("status")) in {"ACCEPTED", "accepted"}
            and not str(row.get("raw_response_text") or "")
        ),
    }
    metric_summary = {
        "pairs": len(pairs),
        "classifier_attempts": len(classifier_attempts),
        "hypotheses": len(hypotheses),
        "validations": len(validations),
        "verification_attempts": len(verifications),
        "terminal_verifications": len(latest_verification_by_hypothesis),
        "facts": len(facts),
        "fact_evidence": len(evidence),
        "source_contexts": len(source_contexts),
        "source_maps": len(source_maps),
        "occurrences": len(occurrences),
        "model_calls": len(model_calls),
        "proposal_attempts": len(proposal_attempts),
        "proposals": len(proposals),
        "pair_contexts": len(pair_contexts),
        "evidence_refs": len(evidence_refs),
        "conflicts": len(conflicts),
        "supported_verification_for_fact_evidence_ratio": (
            1.0 if not evidence else (
                sum(
                    1 for row in evidence
                    if str(row["verification_attempt_id"]) in supported_attempt_ids
                ) / len(evidence)
            )
        ),
        "audit_passed": not audit_failures,
    }
    _write_json(output_dir / "response-quality.json", response_quality)
    _write_json(output_dir / "metric-summary.json", metric_summary)
    audit_report = [
        "# Knowledge Forge relation audit report",
        "",
        f"- run_id: `{run_id}`",
        f"- run_status: `{run.get('status')}`",
        f"- pairs: {len(pairs)}",
        f"- hypotheses: {len(hypotheses)}",
        f"- facts: {len(facts)}",
        f"- fact evidence: {len(evidence)}",
        f"- supported evidence ratio: {metric_summary['supported_verification_for_fact_evidence_ratio']:.6f}",
        f"- audit passed: {str(not audit_failures).lower()}",
    ]
    if audit_failures:
        audit_report.extend(["", "## Failures", ""])
        audit_report.extend(
            f"- {item['check']}: {len(item.get('values') or [])} issue(s)"
            for item in audit_failures
        )
    (output_dir / "audit-report.md").write_text(
        "\n".join(audit_report) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "metric_summary": metric_summary,
        "response_quality": response_quality,
        "audit_failures": audit_failures,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="Isolated PostgreSQL DSN")
    parser.add_argument("--run-id", required=True, help="shadow_relation_v2_runs.id")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv or sys.argv[1:]))
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(
            f"output directory is not empty; pass --overwrite: {output_dir}"
        )
    with psycopg.connect(args.dsn, row_factory=dict_row) as conn:
        result = export_relation_evaluation_bundle(
            conn,
            run_id=args.run_id,
            output_dir=output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not result["audit_failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
