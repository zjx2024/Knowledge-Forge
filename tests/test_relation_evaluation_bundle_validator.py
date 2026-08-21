from __future__ import annotations

import json

from scripts.validate_relation_evaluation_bundle import validate_relation_evaluation_bundle


def _write_minimal_bundle(path):
    path.mkdir()
    (path / "run-summary.json").write_text(
        json.dumps({"id": "run-1", "status": "ready"}) + "\n",
        encoding="utf-8",
    )
    (path / "structure-contexts.jsonl").write_text(
        json.dumps({"id": "ctx-1", "context_identity": "ctx-1"}) + "\n",
        encoding="utf-8",
    )
    (path / "source-maps.jsonl").write_text(
        json.dumps({
            "id": "map-1",
            "context_id": "ctx-1",
            "source_kind": "legacy_chunk_local",
            "source_record_id": "chunk-1",
            "source_start": 0,
            "source_end": 20,
        }) + "\n",
        encoding="utf-8",
    )
    (path / "model-calls.jsonl").write_text(
        json.dumps({
            "id": "model-call-1",
            "run_id": "run-1",
            "component": "classifier",
            "status": "ACCEPTED",
            "idempotency_key": "pair-1:1",
            "attempt_no": 1,
        }) + "\n",
        encoding="utf-8",
    )
    (path / "pair-ledger.csv").write_text(
        "id,pair_identity,document_revision_id,chunk_id,left_occurrence_key,"
        "right_occurrence_key,status,planning_reason,type_status,context_id,"
        "source_kind,source_record_id,pipeline_generation,provenance\n"
        "pair-row-1,pair-1,rev-1,1,left,right,PENDING_CLASSIFICATION,"
        "same_sentence,compatible,ctx-1,v2_pair_planner,ctx-1,1,{}\n",
        encoding="utf-8",
    )
    (path / "classifier-attempts.jsonl").write_text(
        json.dumps({
            "id": "attempt-1",
            "pair_identity": "pair-1",
            "attempt_no": 1,
            "status": "ACCEPTED",
            "raw_response_text": "{\"decisions\":[]}",
        }) + "\n",
        encoding="utf-8",
    )
    (path / "hypotheses.csv").write_text(
        "id,hypothesis_key,pair_identity,predicate,subject_occurrence_key,"
        "object_occurrence_key,status,classifier_policy_version,"
        "classifier_model_revision,source_kind,source_record_id,qualifiers,"
        "metadata\n"
        "hyp-1,hyp-key-1,pair-1,acts_on,left,right,CLASSIFIED,policy,model,"
        "classifier,pair-1,{},{}\n",
        encoding="utf-8",
    )
    (path / "verifications.csv").write_text(
        "id,hypothesis_key,attempt_no,verdict,reason_code,independent_call,"
        "supported_evidence_spans,verifier_prompt_version,"
        "verifier_model_revision\n"
        "ver-1,hyp-key-1,1,SUPPORTED,SUPPORTED_BY_TEXT,true,"
        "\"[{\"\"start\"\":0,\"\"end\"\":20}]\",prompt,model\n",
        encoding="utf-8",
    )
    (path / "normalized-facts-with-evidence.csv").write_text(
        "id,semantic_fact_key,subject_canonical_id,predicate,"
        "object_canonical_id,is_review_only,membership_status,qualifiers,"
        "audit_quality,evidence_count\n"
        "fact-1,fact-key-1,canon-left,acts_on,canon-right,false,ELIGIBLE,"
        "{},complete,1\n",
        encoding="utf-8",
    )
    (path / "fact-evidence.jsonl").write_text(
        json.dumps({
            "id": "evidence-1",
            "fact_id": "fact-1",
            "verification_attempt_id": "ver-1",
            "evidence_spans": [{"start": 0, "end": 20}],
        }) + "\n",
        encoding="utf-8",
    )


def test_relation_bundle_validator_accepts_complete_minimal_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    _write_minimal_bundle(bundle)

    result = validate_relation_evaluation_bundle(bundle)

    assert result["ok"] is True
    assert result["counts"]["facts"] == 1
    assert result["counts"]["source_contexts"] == 1


def test_relation_bundle_validator_rejects_missing_source_contexts(tmp_path):
    bundle = tmp_path / "bundle"
    _write_minimal_bundle(bundle)
    (bundle / "structure-contexts.jsonl").write_text("", encoding="utf-8")

    result = validate_relation_evaluation_bundle(bundle)

    assert result["ok"] is False
    assert "pair_ledger_without_source_contexts" in result["failures"]


def test_relation_bundle_validator_rejects_empty_model_calls(tmp_path):
    bundle = tmp_path / "bundle"
    _write_minimal_bundle(bundle)
    (bundle / "model-calls.jsonl").write_text("", encoding="utf-8")

    result = validate_relation_evaluation_bundle(bundle)

    assert result["ok"] is False
    assert "model_ledgers_without_model_calls" in result["failures"]


def test_relation_bundle_validator_rejects_fact_evidence_without_supported_verifier(
    tmp_path,
):
    bundle = tmp_path / "bundle"
    _write_minimal_bundle(bundle)
    (bundle / "verifications.csv").write_text(
        "id,hypothesis_key,attempt_no,verdict,reason_code,independent_call,"
        "supported_evidence_spans,verifier_prompt_version,"
        "verifier_model_revision\n"
        "ver-1,hyp-key-1,1,INSUFFICIENT_EVIDENCE,NO_TEXT,true,[],prompt,model\n",
        encoding="utf-8",
    )

    result = validate_relation_evaluation_bundle(bundle)

    assert result["ok"] is False
    assert any(
        failure.startswith("fact_evidence_without_supported_independent_verification")
        for failure in result["failures"]
    )
