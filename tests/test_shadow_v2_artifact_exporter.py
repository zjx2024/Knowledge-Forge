from __future__ import annotations

import json
import os
import uuid
import hashlib

import pytest
from psycopg.conninfo import conninfo_to_dict

from knowledge_forge.database import Database
from knowledge_forge.relation_shadow_runtime import shadow_model_payload_hash
from knowledge_forge.relation_shadow_v2 import (
    HYPOTHESES,
    PASS,
    SUPPORTED,
    RelationFreshnessDecision,
    RelationMaterializationDecision,
    normalize_verified_hypotheses,
    parse_classifier_response,
    plan_pair_opportunities,
    validate_hypothesis,
    verify_hypothesis,
)
from scripts.export_shadow_v2_evaluation_bundle import export_shadow_v2_bundle
from tests.shadow_v2_fixtures import (
    logical_schema,
    persist_pair_source_context,
    predicate_policy,
)


def _role_probe(*_args):
    return {
        "subject_role": "source endpoint role",
        "object_role": "target endpoint role",
    }


def _role_alignment_probe(*_args):
    return {"subject_role": "ALIGNED", "object_role": "ALIGNED"}


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _schema() -> dict:
    return logical_schema(
        predicate_policy(
            "uses",
            domain=["Organization"],
            range_=["Product"],
        ),
        version="schema-exporter-v1",
    )


@pytest.mark.integration
def test_shadow_v2_artifact_exporter_exports_and_audits_ready_run(tmp_path):
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"shadow-v2-export-{suffix}", f"Shadow v2 Export {suffix}"),
        ).fetchone()["id"])

    content = "Alpha uses Beta."
    run_id = f"shadow-v2-export-run-{suffix}"
    fingerprint = f"shadow-v2-export-fingerprint-{suffix}"
    plan = plan_pair_opportunities(
        project_id=project_id,
        document_revision_id=f"revision-{suffix}",
        chunk_key=f"chunk-{suffix}",
        content=content,
        mentions=[
            {
                "occurrence_key": "occ-alpha",
                "mention_id": "mention-alpha",
                "surface": "Alpha",
                "mention_start": 0,
                "mention_end": 5,
                "canonical_id": "entity-alpha",
                "assignment_version": "assignment-v1",
                "type": "Organization",
            },
            {
                "occurrence_key": "occ-beta",
                "mention_id": "mention-beta",
                "surface": "Beta",
                "mention_start": 11,
                "mention_end": 15,
                "canonical_id": "entity-beta",
                "assignment_version": "assignment-v1",
                "type": "Product",
            },
        ],
        logical_schema=_schema(),
        run_id=run_id,
        candidate_policy_version="candidate-v2",
    )
    pair = plan.admitted[0]
    decisions = parse_classifier_response(
        plan.admitted,
        [{
            "pair_id": pair.pair_identity,
            "decision": HYPOTHESES,
            "hypotheses": [{
                "subject_occurrence_id": "occ-alpha",
                "predicate": "uses",
                "object_occurrence_id": "occ-beta",
                "evidence_spans": [{"start": 0, "end": len(content)}],
                "classifier_confidence": 0.9,
            }],
        }],
        content=content,
        logical_schema=_schema(),
        classifier_model_revision="classifier-r1",
    )
    hypothesis = decisions[0].hypotheses[0]
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=_schema(),
    )
    assert validation.status == PASS
    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=_schema(),
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda _prompt, _schema: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": SUPPORTED,
            "reason_code": "EXACT_TRIPLE_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{"start": 0, "end": len(content)}],
        },
        verifier_model_revision="verifier-r1",
    )
    normalization = normalize_verified_hypotheses(
        project_id=project_id,
        opportunities={pair.pair_identity: pair},
        hypotheses=[hypothesis],
        verification_attempts=[attempt],
        logical_schema=_schema(),
    )
    materialization = [
        RelationMaterializationDecision(
            fact_id=normalization.facts[0].fact_id,
            eligible=True,
            reasons=(),
            freshness=RelationFreshnessDecision(
                checkpoint="materialize",
                current=True,
                status="CURRENT",
                recorded_root=fingerprint,
                current_root=fingerprint,
                changed_components=(),
            ),
        )
    ]

    database.start_shadow_relation_v2_run({
        "id": run_id,
        "project_id": project_id,
        "input_fingerprint": fingerprint,
        "fingerprint_components": {"test": suffix},
        "source_generation": 1,
    })
    persist_pair_source_context(database, run_id, pair, content)
    database.persist_shadow_relation_v2_opportunities(run_id, plan.opportunities)
    request_payload = {"stage": "classifier", "pair_ids": [pair.pair_identity]}
    raw_response_text = json.dumps({"decisions": []}, separators=(",", ":"))
    database.record_shadow_relation_v2_classification_attempt(
        run_id,
        pair.pair_identity,
        {
            "attempt_no": 1,
            "status": "ACCEPTED",
            "request_hash": shadow_model_payload_hash("request", request_payload),
            "response_hash": shadow_model_payload_hash("response", raw_response_text),
            "classifier_prompt_version": hypothesis.classifier_prompt_version,
            "classifier_schema_version": hypothesis.classifier_schema_version,
            "classifier_parser_version": hypothesis.classifier_parser_version,
            "classifier_model_revision": hypothesis.classifier_model_revision,
            "model_name": "test-model",
            "request_payload": request_payload,
            "raw_response_text": raw_response_text,
            "parsed_response": {"decisions": []},
            "token_usage": {},
            "error_type": "",
            "independent_call": True,
        },
    )
    database.persist_shadow_relation_v2_classifications(run_id, decisions)
    database.persist_shadow_relation_v2_validations(run_id, [validation])
    database.persist_shadow_relation_v2_verification_attempts(run_id, [attempt])
    database.persist_shadow_relation_v2_normalization(
        run_id,
        normalization,
        input_fingerprint=fingerprint,
        materialization_decisions=materialization,
    )
    database.finish_shadow_relation_v2_run(
        run_id,
        status="ready",
        metrics={"pair_opportunities_total": 1, "facts_materialization_eligible": 1},
        input_fingerprint=fingerprint,
    )

    output = tmp_path / "bundle"
    with database.connection() as conn:
        result = export_shadow_v2_bundle(conn, run_id=run_id, output_dir=output)

    assert result["audit_failures"] == []
    assert result["metric_summary"]["facts"] == 1
    assert result["metric_summary"]["fact_evidence"] == 1
    assert result["metric_summary"]["supported_verification_for_fact_evidence_ratio"] == 1.0
    assert result["response_quality"]["hash_recompute_mismatches"] == []
    for filename in (
        "freeze-manifest.json",
        "run-summary.json",
        "structure-contexts.jsonl",
        "source-maps.jsonl",
        "mentions.csv",
        "mention-appearances.csv",
        "occurrences.csv",
        "entities.csv",
        "resolution-outcomes.csv",
        "assignments.csv",
        "model-calls.jsonl",
        "proposal-attempts.jsonl",
        "proposals.csv",
        "pair-ledger.csv",
        "pair-contexts.csv",
        "classifier-attempts.jsonl",
        "hypotheses.csv",
        "verifications.csv",
        "normalized-facts-with-evidence.csv",
        "fact-evidence.jsonl",
        "qualifiers.jsonl",
        "conflicts.csv",
        "pair-status-summary.csv",
        "verification-verdict-summary.csv",
        "response-quality.json",
        "metric-summary.json",
        "audit-report.md",
    ):
        assert (output / filename).exists()
    assert "audit passed: true" in (output / "audit-report.md").read_text(
        encoding="utf-8",
    )
    second_output = tmp_path / "bundle-repeat"
    with database.connection() as conn:
        repeat = export_shadow_v2_bundle(conn, run_id=run_id, output_dir=second_output)
    assert repeat["audit_failures"] == []

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for filename in (
        "freeze-manifest.json",
        "pair-ledger.csv",
        "hypotheses.csv",
        "verifications.csv",
        "normalized-facts-with-evidence.csv",
        "metric-summary.json",
    ):
        assert digest(output / filename) == digest(second_output / filename)
