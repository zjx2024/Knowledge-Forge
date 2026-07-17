from __future__ import annotations

import inspect

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.quality import evaluate_release_gate


def test_quality_and_rollout_schema_is_additive_and_auditable():
    for table in ("quality_evaluation_runs", "policy_activation_gates", "policy_rollouts"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in SCHEMA_SQL
    assert "production_approved BOOLEAN NOT NULL DEFAULT FALSE" in SCHEMA_SQL
    assert "rollback_snapshot_id" in SCHEMA_SQL


def test_release_gate_stays_engineering_complete_without_human_gold():
    result = evaluate_release_gate(
        control={"entity_f1": 0.9, "b3_f1": 0.9, "relation_f1": 0.9,
                 "candidate_recall": 1.0, "critical_errors": 0, "cost": 100},
        treatment={"entity_f1": 0.91, "b3_f1": 0.91, "relation_f1": 0.91,
                   "candidate_recall": 1.0, "critical_errors": 0, "cost": 80},
        manifest={"engineering_valid": True, "production_valid": False},
    )
    assert result["status"] == "engineering_complete"
    assert result["production_approved"] is False
    assert "human_gold" in result["failed_gates"]


def test_activation_path_requires_persisted_production_gate():
    source = inspect.getsource(Database.activate_policy_snapshot_with_gate)
    assert "production_approved=TRUE" in source
    assert "manifest_valid=TRUE" in source
    assert "FOR UPDATE" in source


def test_rollout_order_is_controlled_and_reversible():
    source = inspect.getsource(Database.record_policy_rollout)
    assert "control" in source
    assert "shadow" in source
    assert "canary" in source
    assert "rollback_snapshot_id" in source
