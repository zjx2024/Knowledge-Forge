#!/usr/bin/env python3
"""Replay the frozen 176 legacy hypotheses through v2 validation and verifier."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge import relation_shadow_v2 as v2
from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.ollama import OllamaClient
from knowledge_forge.quality import canonical_digest
from knowledge_forge.relation_shadow_runtime import (
    SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION,
    SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
    ShadowRelationV2Runtime,
)


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class ReplayConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(ReplayConfig)


def _terminal_counts(attempts: list[v2.VerificationAttempt]) -> Counter:
    terminal: dict[str, v2.VerificationAttempt] = {}
    for item in attempts:
        previous = terminal.get(item.hypothesis_id)
        if previous is None or item.attempt_no > previous.attempt_no:
            terminal[item.hypothesis_id] = item
    return Counter(item.verdict for item in terminal.values())


class _HistoricalSnapshotDatabase:
    """Expose only the frozen historical revision universe to fingerprinting."""

    def __init__(
        self,
        database: Database,
        source_revisions: list[dict[str, Any]],
        source_chunks: list[dict[str, Any]],
    ):
        self.database = database
        self.revisions = {
            int(item["document_id"]): dict(item)
            for item in source_revisions
        }
        self.chunks_by_document: dict[int, list[dict[str, Any]]] = {}
        for chunk in source_chunks:
            self.chunks_by_document.setdefault(
                int(chunk["document_id"]), []
            ).append(dict(chunk))

    def __getattr__(self, name: str):
        return getattr(self.database, name)

    def list_documents(self, project_id: int):
        documents = []
        for document_id, revision in sorted(self.revisions.items()):
            document = self.database.get_document(document_id)
            if not document or int(document["project_id"]) != int(project_id):
                raise RuntimeError("frozen legacy document no longer matches project")
            documents.append({
                **document,
                "status": "indexed",
                "is_deleted": False,
                "active_revision_id": str(revision["revision_id"]),
                "doc_hash": str(revision["content_hash"]),
            })
        return documents

    def list_document_chunks(self, document_id: int):
        return list(self.chunks_by_document.get(int(document_id), []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    legacy = freeze["frozen_legacy_176"]
    project_id = int(legacy["namespace"]["project_id"])
    namespace_id = str(legacy["namespace"]["id"])
    frozen_endpoints = list(legacy["endpoints"])
    if len(frozen_endpoints) != 176:
        raise RuntimeError("legacy replay requires the original frozen 176 hypotheses")
    database = _database(args.dsn)
    database.assert_shadow_relation_v2_ready()
    gateway = OllamaClient()
    runtime = ShadowRelationV2Runtime(database, gateway)
    historical_database = _HistoricalSnapshotDatabase(
        database,
        list(legacy["source_revisions"]),
        list(legacy["source_chunks"]),
    )
    historical_runtime = ShadowRelationV2Runtime(
        historical_database, gateway,
    )
    context = historical_runtime._context(project_id, namespace_id)
    fingerprint = context["fingerprint"]
    run_id = "shadow_relation_v2_legacy_replay_" + hashlib.sha256(
        (
            str(freeze["freeze_payload_sha256"])
            + "|176|replay-parser-filter-v2|"
            + fingerprint.root
        ).encode("utf-8")
    ).hexdigest()
    database.start_shadow_relation_v2_run({
        "id": run_id,
        "project_id": project_id,
        "snapshot_id": namespace_id,
        "input_fingerprint": fingerprint.root,
        "fingerprint_components": {
            **fingerprint.components,
            "replay": {
                "kind": "frozen-legacy-hypotheses",
                "count": 176,
                "freeze_payload_sha256": freeze["freeze_payload_sha256"],
            },
        },
        "source_generation": int(
            context["project"].get("graph_generation") or 0
        ),
    })

    assignment_by_mention = context["assignment_by_mention"]
    mentions_by_chunk: dict[int, list[dict[str, Any]]] = {}
    for mention in context["mentions"]:
        assignment = assignment_by_mention[str(mention["mention_key"])]
        mentions_by_chunk.setdefault(int(mention["chunk_id"]), []).append({
            "occurrence_key": str(mention["mention_key"]),
            "mention_id": str(mention["mention_key"]),
            "surface": str(mention["surface"]),
            "mention_start": int(mention["evidence_start"]),
            "mention_end": int(mention["evidence_end"]),
            "canonical_id": str(assignment["canonical_key"]),
            "assignment_version": str(
                assignment.get("policy_version") or namespace_id
            ),
            "type": str(assignment.get("entity_type") or "GenericEntity"),
        })
    planned_by_occurrences: dict[tuple[str, str], v2.PairOpportunity] = {}
    for chunk in context["chunks"]:
        plan = v2.plan_pair_opportunities(
            project_id=project_id,
            document_revision_id=str(chunk["revision_id"]),
            chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
            content=str(chunk.get("content") or ""),
            mentions=mentions_by_chunk.get(int(chunk["id"]), []),
            logical_schema=context["schema"],
            run_id=run_id,
            candidate_policy_version=
                SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
            max_pair_opportunities=
                Config.relation_max_candidates_per_chunk,
            sentence_window=Config.relation_sentence_window,
            max_char_distance=Config.relation_max_char_distance,
        )
        for item in plan.opportunities:
            annotated = replace(item, provenance={
                **item.provenance,
                "document_id": int(chunk["document_id"]),
                "chunk_id": int(chunk["id"]),
                "content_hash": str(chunk.get("content_hash") or ""),
                "legacy_replay": True,
            })
            key = tuple(sorted((
                str(item.left_occurrence_key),
                str(item.right_occurrence_key),
            )))
            if key in planned_by_occurrences:
                raise RuntimeError(f"duplicate planned occurrence pair: {key}")
            planned_by_occurrences[key] = annotated

    endpoints_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for endpoint in frozen_endpoints:
        key = tuple(sorted((
            str(endpoint["source_mention_key"]),
            str(endpoint["target_mention_key"]),
        )))
        endpoints_by_pair.setdefault(key, []).append(endpoint)
    missing_pairs = sorted(set(endpoints_by_pair) - set(planned_by_occurrences))
    if missing_pairs:
        raise RuntimeError(
            f"{len(missing_pairs)} frozen legacy pairs are absent from v2 planner"
        )
    opportunities = [
        planned_by_occurrences[key] for key in sorted(endpoints_by_pair)
    ]
    database.persist_shadow_relation_v2_opportunities(run_id, opportunities)
    admitted_opportunities = [
        item for item in opportunities
        if item.status == v2.PENDING_CLASSIFICATION
    ]
    planner_excluded_hypotheses = sum(
        len(endpoints_by_pair[tuple(sorted((
            item.left_occurrence_key,
            item.right_occurrence_key,
        )))])
        for item in opportunities
        if item.status != v2.PENDING_CLASSIFICATION
    )

    decisions: list[v2.PairClassificationDecision] = []
    derived_evidence_spans = 0
    structural_rejections: list[dict[str, str]] = []
    for index, opportunity in enumerate(admitted_opportunities, start=1):
        chunk = context["chunk_by_id"][
            int(opportunity.provenance["chunk_id"])
        ]
        content = str(chunk.get("content") or "")
        raw_hypotheses = []
        for endpoint in sorted(
            endpoints_by_pair[tuple(sorted((
                opportunity.left_occurrence_key,
                opportunity.right_occurrence_key,
            )))],
            key=lambda item: str(item["relation_key"]),
        ):
            evidence_text = str(endpoint.get("evidence_text") or "")
            evidence_start = content.find(evidence_text) if evidence_text else -1
            if evidence_start >= 0:
                evidence_end = evidence_start + len(evidence_text)
                if not (
                    opportunity.context_span[0] <= evidence_start
                    and evidence_end <= opportunity.context_span[1]
                ):
                    evidence_start, evidence_end = opportunity.context_span
            else:
                evidence_start, evidence_end = opportunity.context_span
            derived_evidence_spans += 1
            raw_hypothesis = {
                "subject_occurrence_id":
                    str(endpoint["source_mention_key"]),
                "predicate": str(endpoint["relation_type"]),
                "object_occurrence_id":
                    str(endpoint["target_mention_key"]),
                "evidence_spans": [{
                    "start": int(evidence_start),
                    "end": int(evidence_end),
                }],
                "classifier_confidence": float(
                    (endpoint.get("metadata") or {}).get("model_confidence")
                    or 0.5
                ),
            }
            probe = [{
                "pair_id": opportunity.pair_identity,
                "decision": v2.HYPOTHESES,
                "hypotheses": [raw_hypothesis],
            }]
            try:
                v2.parse_classifier_response(
                    [opportunity],
                    probe,
                    content=content,
                    logical_schema=context["schema"],
                    classifier_policy_version=
                        SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION,
                    classifier_model_revision=context["model_revision"],
                )
            except v2.ShadowClassificationError as exc:
                structural_rejections.append({
                    "relation_key": str(endpoint["relation_key"]),
                    "pair_identity": opportunity.pair_identity,
                    "reason": str(exc),
                })
            else:
                raw_hypotheses.append(raw_hypothesis)
        if raw_hypotheses:
            raw = [{
                "pair_id": opportunity.pair_identity,
                "decision": v2.HYPOTHESES,
                "hypotheses": raw_hypotheses,
            }]
        else:
            raw = [{
                "pair_id": opportunity.pair_identity,
                "decision": v2.OTHER_UNSUPPORTED,
                "reason_code": "SCHEMA_GAP",
                "evidence_spans": [{
                    "start": int(opportunity.context_span[0]),
                    "end": int(opportunity.context_span[1]),
                }],
                "hypotheses": [],
            }]
        request_hash = canonical_digest({
            "kind": "legacy-replay",
            "pair_id": opportunity.pair_identity,
            "frozen_relation_keys": sorted(
                str(item["relation_key"])
                for item in endpoints_by_pair[tuple(sorted((
                    opportunity.left_occurrence_key,
                    opportunity.right_occurrence_key,
                )))]
            ),
        })
        database.record_shadow_relation_v2_classification_attempt(
            run_id,
            opportunity.pair_identity,
            {
                "attempt_no": 1,
                "status": "accepted",
                "request_hash": request_hash,
                "response_hash": canonical_digest(raw),
                "classifier_prompt_version": "legacy-frozen-replay-v1",
                "classifier_schema_version":
                    v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
                "classifier_parser_version":
                    v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
                "classifier_model_revision": context["model_revision"],
                "metadata": {
                    "freeze_payload_sha256":
                        freeze["freeze_payload_sha256"],
                    "not_a_native_classifier_call": True,
                },
            },
        )
        parsed = v2.parse_classifier_response(
            [opportunity],
            raw,
            content=content,
            logical_schema=context["schema"],
            classifier_policy_version=
                SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION,
            classifier_model_revision=context["model_revision"],
        )
        database.persist_shadow_relation_v2_classifications(run_id, parsed)
        decisions.extend(parsed)
        if index % 20 == 0 or index == len(admitted_opportunities):
            print(
                "legacy replay classification "
                f"{index}/{len(admitted_opportunities)}"
            )
    hypotheses = [
        hypothesis
        for decision in decisions
        for hypothesis in decision.hypotheses
    ]
    if (
        len(hypotheses)
        + planner_excluded_hypotheses
        + len(structural_rejections)
        != 176
    ):
        raise RuntimeError(
            "legacy replay failed to account for all frozen hypotheses: "
            f"classified={len(hypotheses)},"
            f"planner_excluded={planner_excluded_hypotheses},"
            f"structural_rejected={len(structural_rejections)}"
        )
    opportunity_by_pair = {
        item.pair_identity: item for item in opportunities
    }
    validations = []
    for hypothesis in hypotheses:
        opportunity = opportunity_by_pair[hypothesis.pair_identity]
        chunk = context["chunk_by_id"][
            int(opportunity.provenance["chunk_id"])
        ]
        validations.append(v2.validate_hypothesis(
            hypothesis,
            opportunity,
            content=str(chunk.get("content") or ""),
            logical_schema=context["schema"],
        ))
    database.persist_shadow_relation_v2_validations(run_id, validations)
    validation_by_hypothesis = {
        item.hypothesis_id: item for item in validations
    }

    attempts: list[v2.VerificationAttempt] = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        opportunity = opportunity_by_pair[hypothesis.pair_identity]
        content = str(context["chunk_by_id"][
            int(opportunity.provenance["chunk_id"])
        ].get("content") or "")
        validation = validation_by_hypothesis[hypothesis.hypothesis_id]
        for attempt_no in range(1, Config.relation_shadow_max_attempts + 1):
            attempt = v2.verify_hypothesis(
                hypothesis,
                validation,
                opportunity,
                content=content,
                logical_schema=context["schema"],
                verifier=lambda prompt, response_schema: runtime._verify(
                    model=context["model"],
                    context_budget_tokens=int(
                        context["project"].get("model_max_tokens")
                        or runtime.default_context_tokens
                    ),
                    prompt=prompt,
                    response_schema=response_schema,
                ),
                verifier_model_revision=context["model_revision"],
                attempt_no=attempt_no,
            )
            database.persist_shadow_relation_v2_verification_attempts(
                run_id, [attempt],
            )
            attempts.append(attempt)
            if (
                attempt.verdict != v2.RETRYABLE_FAILED
                or validation.status != v2.PASS
            ):
                break
        if index % 20 == 0 or index == len(hypotheses):
            print(f"legacy replay verifier {index}/{len(hypotheses)}")

    normalization = v2.normalize_verified_hypotheses(
        project_id=project_id,
        opportunities=opportunity_by_pair,
        hypotheses=hypotheses,
        verification_attempts=attempts,
        logical_schema=context["schema"],
    )
    current_fingerprint = historical_runtime._context(
        project_id, namespace_id,
    )["fingerprint"]
    hypothesis_by_id = {
        str(item.hypothesis_id): item for item in hypotheses
    }
    evidence_pairs_by_fact: dict[str, set[str]] = {}
    for evidence in normalization.evidence:
        hypothesis = hypothesis_by_id[str(evidence.hypothesis_id)]
        evidence_pairs_by_fact.setdefault(
            evidence.fact_id, set(),
        ).add(hypothesis.pair_identity)
    materialization_decisions = [
        v2.relation_materialization_decision(
            fact,
            normalization.evidence,
            recorded_fingerprint=fingerprint,
            current_fingerprint=current_fingerprint,
            snapshot_complete=True,
            current_assignment_and_schema=(
                fingerprint.root == current_fingerprint.root
            ),
            conflicts=[
                conflict for conflict in normalization.conflicts
                if conflict.pair_identity
                in evidence_pairs_by_fact.get(fact.fact_id, set())
            ],
        )
        for fact in normalization.facts
    ]
    database.persist_shadow_relation_v2_normalization(
        run_id,
        normalization,
        input_fingerprint=fingerprint.root,
        materialization_decisions=materialization_decisions,
    )
    verdict_counts = _terminal_counts(attempts)
    metrics = {
        "mode": "frozen-legacy-176-replay",
        "frozen_hypotheses": 176,
        "planner_admitted_hypotheses": len(hypotheses),
        "planner_excluded_hypotheses": planner_excluded_hypotheses,
        "structural_rejected_hypotheses": len(structural_rejections),
        "structural_rejection_reasons": dict(Counter(
            item["reason"] for item in structural_rejections
        )),
        "planner_statuses": dict(Counter(
            item.status for item in opportunities
        )),
        "pair_ledger": len(opportunities),
        "derived_evidence_spans": derived_evidence_spans,
        "validation": dict(Counter(item.status for item in validations)),
        "verifier_terminal": dict(verdict_counts),
        "normalized_facts": len(normalization.facts),
        "normalized_evidence": len(normalization.evidence),
        "eligible_facts": sum(
            int(item.eligible) for item in materialization_decisions
        ),
        "legacy_pipeline_calls": 0,
        "native_classifier_calls": 0,
        "production_approved": False,
    }
    database.finish_shadow_relation_v2_run(
        run_id,
        status="ready",
        metrics=metrics,
        input_fingerprint=fingerprint.root,
    )
    result = {
        "run_id": run_id,
        "project_id": project_id,
        "namespace_id": namespace_id,
        "input_fingerprint": fingerprint.root,
        "metrics": metrics,
        "formal_graph_writes": 0,
        "activation_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
