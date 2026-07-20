from __future__ import annotations

import inspect
import os

import psycopg
import pytest

from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.quality import (
    GoldManifestError,
    b3_metrics,
    canonical_digest,
    evaluate_relations,
    pairwise_clustering_metrics,
    validate_gold_annotations,
    validate_gold_manifest,
)
from knowledge_forge.release_gate import ArtifactEvaluationError, evaluate_frozen_bundle
from knowledge_forge.service import KnowledgeForgeService


def _pilot_manifest() -> dict:
    return {
        "manifest_version": "kg-gold-v2",
        "dataset": "pilot-fixture",
        "version": "1",
        "level": "pilot",
        "frozen_at": "2026-07-20T00:00:00Z",
        "schema_compatibility": ["schema-v1"],
        "policy_compatibility": ["policy-v1"],
        "calibration_universe_version": "calibration-v1",
        "items": [{
            "id": "item-1", "document_id": "doc-1", "revision_id": "rev-1",
            "chunk_id": "chunk-1", "content_sha256": "a" * 64, "split": "dev",
            "domain": "fixture", "label_source": "human_reviewed",
            "reviewers": ["alice"], "reviewed_at": "2026-07-20T00:00:00Z",
        }],
    }


def _lineage_relation(**overrides) -> dict:
    value = {
        "document_id": "doc-1", "revision_id": "rev-1", "content_hash": "a" * 64,
        "chunk_id": "chunk-1", "subject_mention_id": "m1",
        "subject_canonical_id": "c1", "predicate": "uses",
        "object_mention_id": "m2", "object_canonical_id": "c2",
        "direction": "forward", "evidence_text": "Alpha uses Beta",
        "evidence_start": 10, "evidence_end": 25,
        "snapshot_id": "snapshot-1", "policy_hash": "policy-v1",
    }
    value.update(overrides)
    return value


def _bundle() -> tuple[dict, list[dict], dict]:
    manifest = _pilot_manifest()
    gold = [{
        "item_id": "item-1",
        "mentions": [
            {"mention_id": "m1", "document_id": "doc-1", "revision_id": "rev-1",
             "content_hash": "a" * 64, "chunk_id": "chunk-1", "start": 0, "end": 5,
             "type": "Organization", "canonical_id": "c1"},
            {"mention_id": "m2", "document_id": "doc-1", "revision_id": "rev-1",
             "content_hash": "a" * 64, "chunk_id": "chunk-1", "start": 11, "end": 15,
             "type": "Product", "canonical_id": "c2"},
        ],
        "relations": [_lineage_relation()],
        "calibration": [{"key": "r1", "label": 1}],
    }]
    prediction = {
        "item_id": "item-1",
        "control": {
            "mentions": gold[0]["mentions"], "relations": gold[0]["relations"],
            "calibration": [{"key": "r1", "probability": 0.9}],
            "cost": {"tokens": 100, "model_calls": 1},
        },
        "treatment": {
            "mentions": gold[0]["mentions"], "relations": gold[0]["relations"],
            "calibration": [{"key": "r1", "probability": 0.95}],
            "cost": {"tokens": 90, "model_calls": 1},
        },
    }
    prediction["item_hash"] = canonical_digest(prediction)
    snapshot = {
        "id": "snapshot-1", "status": "ready", "pilot": True,
        "manifest_hash": "b" * 64,
        "artifact_root_hash": canonical_digest([["item-1", prediction["item_hash"]]]),
        "expected_items": 1, "completed_items": 1, "failed_items": 0,
    }
    return manifest, gold, {"snapshot": snapshot, "items": [prediction]}


def test_gate_api_only_accepts_persisted_evaluation_run_id():
    signature = inspect.signature(KnowledgeForgeService.evaluate_policy_snapshot_gate)
    assert list(signature.parameters) == ["self", "evaluation_run_id", "approved_by"]
    assert "control_metrics" not in signature.parameters
    assert "treatment_metrics" not in signature.parameters


def test_detection_aware_clustering_penalizes_missing_and_spurious_mentions():
    gold = {"m1": "a", "m2": "a", "m3": "b"}
    perfect = {"m1": "x", "m2": "x", "m3": "y"}
    incomplete = {"m1": "x", "extra": "x"}
    assert pairwise_clustering_metrics(gold, perfect)["f1"] == 1.0
    assert b3_metrics(gold, perfect)["f1"] == 1.0
    pairwise = pairwise_clustering_metrics(gold, incomplete)
    b3 = b3_metrics(gold, incomplete)
    assert pairwise["f1"] < 1.0
    assert pairwise["missed_mentions"] == 2
    assert pairwise["spurious_mentions"] == 1
    assert b3["f1"] < 1.0
    assert b3["universe_mentions"] == 4


@pytest.mark.parametrize("change", [
    {"direction": "reverse"},
    {"evidence_start": 11},
    {"evidence_text": "Alpha uses Gamma"},
    {"revision_id": "rev-2"},
    {"chunk_id": "chunk-2"},
    {"subject_canonical_id": "wrong"},
    {"snapshot_id": "snapshot-2"},
])
def test_strict_relation_requires_exact_lineage_direction_endpoints_and_evidence(change):
    report = evaluate_relations([_lineage_relation()], [_lineage_relation(**change)])
    assert report["strict"]["true_positive"] == 0
    assert report["strict"]["f1"] == 0.0


def test_pilot_bundle_is_auditable_but_cannot_approve_production():
    manifest, gold, artifacts = _bundle()
    evaluation = evaluate_frozen_bundle(
        snapshot_bundle=artifacts, gold_manifest=manifest, gold_items=gold,
        evaluator_version="quality-v2",
    )
    assert evaluation["status"] == "completed"
    assert evaluation["manifest_report"]["production_valid"] is False
    assert evaluation["gate_report"]["production_approved"] is False
    assert "human_gold" in evaluation["gate_report"]["failed_gates"]
    assert evaluation["report_hash"] == canonical_digest(evaluation["report"])
    assert evaluation["item_hashes"] == [artifacts["items"][0]["item_hash"]]


def test_evaluator_fails_closed_for_missing_or_tampered_artifacts():
    manifest, gold, artifacts = _bundle()
    artifacts["snapshot"]["artifact_root_hash"] = "0" * 64
    with pytest.raises(ArtifactEvaluationError, match="artifact root"):
        evaluate_frozen_bundle(
            snapshot_bundle=artifacts, gold_manifest=manifest, gold_items=gold,
            evaluator_version="quality-v2",
        )
    manifest, gold, artifacts = _bundle()
    artifacts["items"] = []
    with pytest.raises(ArtifactEvaluationError, match="complete"):
        evaluate_frozen_bundle(
            snapshot_bundle=artifacts, gold_manifest=manifest, gold_items=gold,
            evaluator_version="quality-v2",
        )


def test_release_manifest_requires_holdout_two_reviewers_and_trusted_approval():
    manifest = _pilot_manifest()
    manifest["level"] = "release"
    manifest["pilot"] = False
    with pytest.raises(GoldManifestError):
        validate_gold_manifest(manifest, require_production=True)


def test_gold_minimums_are_derived_from_annotations_not_manifest_claims():
    manifest, gold_items, _ = _bundle()
    manifest["label_counts"] = {
        "entity_mentions": 500,
        "must_links": 150,
        "cannot_links": 150,
        "positive_relations": 200,
        "hard_negatives": 300,
    }
    report = validate_gold_annotations(
        manifest, gold_items, validate_gold_manifest(manifest),
    )
    assert report["observed_minimums"]["entity_mentions"] == 2
    assert report["observed_minimums"]["positive_relations"] == 1
    assert report["observed_minimums"]["hard_negatives"] == 0
    assert "label_count_mismatch_entity_mentions" in report["production_reasons"]


@pytest.mark.integration
def test_disposable_postgres_quality_facts_are_append_only_and_gate_revalidates_hashes():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    params = psycopg.conninfo.conninfo_to_dict(dsn)

    class TestConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]
        pgvector_hnsw_max_dimensions = Config.pgvector_hnsw_max_dimensions
        pgvector_hnsw_m = Config.pgvector_hnsw_m
        pgvector_hnsw_ef_construction = Config.pgvector_hnsw_ef_construction

    database = Database(TestConfig)
    database.bootstrap()
    project = database.create_project("quality", "", None, None)
    manifest, gold_items, artifacts = _bundle()
    snapshot = artifacts["snapshot"]
    with database.connection() as conn:
        conn.execute(
            """INSERT INTO policy_snapshots(
               id,project_id,policy_set,policy_hash,status,source_generation,
               expected_documents,snapshot_manifest,mode,manifest_hash,artifact_root_hash,
               expected_items,completed_items,failed_items,pilot)
               VALUES(%s,%s,'{}'::jsonb,'treatment','ready',0,0,'{}'::jsonb,
                      'relation_only',%s,%s,1,1,0,TRUE)""",
            (snapshot["id"], project["id"], snapshot["manifest_hash"], snapshot["artifact_root_hash"]),
        )
    evaluation = evaluate_frozen_bundle(
        snapshot_bundle=artifacts, gold_manifest=manifest, gold_items=gold_items,
        evaluator_version="quality-v2",
    )
    validation = evaluation["manifest_report"]
    gold = database.persist_gold_manifest(project["id"], manifest, gold_items, validation)

    def persist(
        run_id: str, artifact_root: str, *, gold_hash: str | None = None,
        report_hash: str | None = None,
    ):
        return database.create_quality_evaluation_run({
            "id": run_id, "project_id": project["id"], "snapshot_id": snapshot["id"],
            "control_policy_hash": "control", "treatment_policy_hash": "treatment",
            "dataset": manifest["dataset"], "dataset_version": manifest["version"],
            "split": "combined", "manifest": manifest, "manifest_valid": False,
            "metrics": evaluation["report"], "denominators": validation,
            "cost_metrics": {}, "status": "completed", "gold_manifest_id": gold["id"],
            "evaluator_version": "quality-v2", "artifact_root_hash": artifact_root,
            "gold_manifest_hash": gold_hash or evaluation["gold_manifest_hash"],
            "gold_items_hash": evaluation["gold_items_hash"],
            "report_hash": report_hash or evaluation["report_hash"],
            "item_hashes": evaluation["item_hashes"],
            "release_level": "pilot", "evaluation_items": [{
                "item_id": "item-1", "artifact_hash": artifacts["items"][0]["item_hash"],
                "result": {"strict": True},
            }],
        })

    persisted = persist("evaluation-good", snapshot["artifact_root_hash"])
    service = KnowledgeForgeService(database)
    gate = service.evaluate_policy_snapshot_gate(persisted["id"])
    assert gate["report"]["production_approved"] is False
    assert gate["gate"]["evaluation_run_id"] == persisted["id"]
    with database.connection() as conn:
        with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState):
            conn.execute(
                "UPDATE quality_evaluation_runs SET report_hash='tampered' WHERE id=%s",
                (persisted["id"],),
            )
    bad = persist("evaluation-bad", "0" * 64)
    with pytest.raises(RuntimeError, match="artifact root"):
        service.evaluate_policy_snapshot_gate(bad["id"])
    bad_gold = persist(
        "evaluation-bad-gold", snapshot["artifact_root_hash"], gold_hash="1" * 64,
    )
    with pytest.raises(RuntimeError, match="Gold manifest"):
        service.evaluate_policy_snapshot_gate(bad_gold["id"])
    bad_report = persist(
        "evaluation-bad-report", snapshot["artifact_root_hash"], report_hash="2" * 64,
    )
    with pytest.raises(RuntimeError, match="evaluation report"):
        service.evaluate_policy_snapshot_gate(bad_report["id"])
