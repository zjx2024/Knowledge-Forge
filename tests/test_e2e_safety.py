from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import uuid

import pytest

from knowledge_forge.e2e_safety import (
    E2ESafetyError,
    PerformanceObservation,
    PostgresCleanupPlan,
    REQUIRED_E2E_SCENARIOS,
    build_preflight_manifest,
    evaluate_performance_gates,
    issue_execution_token,
    verify_postgres_sentinel_record,
    verify_execution_token,
)


def _manifest(**updates):
    values = {
        "run_id": f"run-{uuid.uuid4().hex}",
        "postgres_identity": "postgresql://test@127.0.0.1:55432/kf_e2e/schema_a",
        "production_postgres_identity": "postgresql://prod@10.0.0.9:5432/knowledge_forge/public",
        "graph_identity": "http://127.0.0.1:18080/kf_disposable",
        "production_graph_identity": "http://10.0.0.9:8080/knowledge_forge",
        "graph_namespace": str(uuid.uuid4()),
        "api_identity": "http://127.0.0.1:18011",
        "model_identities": {
            "llm": {"endpoint": "http://127.0.0.1:11435", "model": "fixture", "digest": "sha256:a", "context": 8192},
            "embedding": {"endpoint": "http://127.0.0.1:11435", "model": "fixture-embed", "digest": "sha256:b", "context": 8192},
            "reranker": {"endpoint": "local://fixture", "model": "fixture-reranker", "digest": "sha256:c", "context": 384},
        },
        "git_commit": "a" * 40,
        "git_tree": "clean",
        "policy_hash": "b" * 64,
        "config_hash": "c" * 64,
        "postgres_sentinel": True,
        "graph_sentinel": True,
        "integration_enabled": True,
        "destructive_enabled": True,
        "api_bind_warning": "",
        "limits": {"documents": 10, "reranker_top_k": 10, "max_context": 8192},
        "postgres_server_version": "PostgreSQL 16.14",
        "pgvector_version": "0.8.5",
        "production_model_endpoints": {
            "llm": "http://10.0.0.9:11434",
            "embedding": "http://10.0.0.9:11434",
            "reranker": "local://production-reranker",
        },
    }
    values.update(updates)
    return build_preflight_manifest(**values)


def test_preflight_manifest_redacts_secrets_and_lists_every_required_scenario():
    manifest = _manifest()
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    assert "password" not in serialized.lower()
    assert "secret" not in serialized.lower()
    assert set(REQUIRED_E2E_SCENARIOS).issubset(manifest["scenarios"])
    assert manifest["postgres_server_version"] == "PostgreSQL 16.14"
    assert manifest["pgvector_version"] == "0.8.5"
    assert manifest["safe_to_arm"] is True


def test_execution_token_is_short_lived_and_bound_to_entire_environment():
    now = datetime(2026, 7, 20, tzinfo=timezone.utc)
    manifest = _manifest()
    token = issue_execution_token(manifest, secret="local-arm-key", now=now, ttl_seconds=120)
    verified = verify_execution_token(
        token, current_manifest=manifest, secret="local-arm-key",
        now=now + timedelta(seconds=30),
    )
    assert verified["run_id"] == manifest["run_id"]

    changed = {**manifest, "api_identity": "http://127.0.0.1:18012"}
    with pytest.raises(E2ESafetyError, match="environment"):
        verify_execution_token(
            token, current_manifest=changed, secret="local-arm-key",
            now=now + timedelta(seconds=30),
        )
    with pytest.raises(E2ESafetyError, match="expired"):
        verify_execution_token(
            token, current_manifest=manifest, secret="local-arm-key",
            now=now + timedelta(seconds=121),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"production_postgres_identity": ""}, "production PostgreSQL"),
        ({"postgres_identity": "same", "production_postgres_identity": "same"}, "PostgreSQL"),
        ({"graph_identity": "same", "production_graph_identity": "same"}, "HugeGraph"),
        ({"graph_namespace": "default"}, "namespace"),
        ({"postgres_sentinel": False}, "PostgreSQL sentinel"),
        ({"graph_sentinel": False}, "HugeGraph sentinel"),
        ({"destructive_enabled": False}, "destructive"),
        ({"postgres_server_version": ""}, "PostgreSQL server version"),
        ({"pgvector_version": ""}, "pgvector version"),
        ({"production_model_endpoints": {}}, "production endpoint"),
        ({
            "production_model_endpoints": {
                "llm": "http://127.0.0.1:11435",
                "embedding": "http://10.0.0.9:11434",
                "reranker": "local://production-reranker",
            },
        }, "llm test and production endpoints"),
    ],
)
def test_preflight_fails_closed_when_isolation_evidence_is_missing(updates, message):
    manifest = _manifest(**updates)
    assert manifest["safe_to_arm"] is False
    assert any(message in blocker for blocker in manifest["blockers"])


def test_cleanup_plan_can_only_target_owned_random_schema_and_role():
    run_id = uuid.uuid4().hex
    plan = PostgresCleanupPlan(
        schema=f"kf_e2e_{run_id}", role=f"kf_e2e_role_{run_id}",
        run_id=run_id, sentinel_verified=True,
    )
    assert plan.operations() == (
        ("drop_schema", f"kf_e2e_{run_id}"),
        ("drop_role", f"kf_e2e_role_{run_id}"),
    )
    with pytest.raises(E2ESafetyError):
        PostgresCleanupPlan(
            schema="public", role=f"kf_e2e_role_{run_id}",
            run_id=run_id, sentinel_verified=True,
        ).operations()


def test_postgres_sentinel_must_match_actual_identity_config_and_expiry():
    now = datetime(2026, 7, 20, tzinfo=timezone.utc)
    run_id = "a" * 20
    record = {
        "run_id": run_id,
        "identity": "postgresql://test@127.0.0.1:5432/kf_e2e_db/kf_e2e_aaaaaaaaaaaaaaaaaaaa",
        "schema_name": "kf_e2e_aaaaaaaaaaaaaaaaaaaa",
        "role_name": "kf_e2e_role_aaaaaaaaaaaaaaaaaaaa",
        "config_hash": "b" * 64,
        "expires_at": now + timedelta(minutes=5),
    }
    assert verify_postgres_sentinel_record(
        record, run_id=record["run_id"], identity=record["identity"],
        schema=record["schema_name"], role=record["role_name"],
        config_hash=record["config_hash"], now=now,
    )
    assert not verify_postgres_sentinel_record(
        {**record, "identity": "postgresql://other"},
        run_id=record["run_id"], identity=record["identity"],
        schema=record["schema_name"], role=record["role_name"],
        config_hash=record["config_hash"], now=now,
    )
    assert not verify_postgres_sentinel_record(
        {**record, "expires_at": now - timedelta(seconds=1)},
        run_id=record["run_id"], identity=record["identity"],
        schema=record["schema_name"], role=record["role_name"],
        config_hash=record["config_hash"], now=now,
    )
    with pytest.raises(E2ESafetyError):
        PostgresCleanupPlan(
            schema=f"kf_e2e_{run_id}", role="postgres",
            run_id=run_id, sentinel_verified=True,
        ).operations()
    with pytest.raises(E2ESafetyError):
        PostgresCleanupPlan(
            schema=f"kf_e2e_{run_id}", role=f"kf_e2e_role_{run_id}",
            run_id=run_id, sentinel_verified=False,
        ).operations()


def test_performance_gate_uses_operations_and_rejects_hidden_backlog_or_full_scan():
    healthy = PerformanceObservation(
        context_occurrences=1, request_tokens=7000, verified_context=8192,
        exact_hit_embedding_calls=0, exact_hit_reranker_calls=0,
        exact_hit_llm_calls=0, reranker_pairs=80, mentions=10,
        configured_top_k=8, project_full_scan=False,
        relation_filter_before_limit=True, rebuilds_for_ten_documents=1,
        bounded_queues=True, deferred_terminal=0,
    )
    assert evaluate_performance_gates(healthy).passed is True
    broken = PerformanceObservation(
        **{**healthy.__dict__, "context_occurrences": 2,
           "exact_hit_embedding_calls": 1, "reranker_pairs": 81,
           "project_full_scan": True, "relation_filter_before_limit": False,
           "rebuilds_for_ten_documents": 10, "deferred_terminal": 2}
    )
    report = evaluate_performance_gates(broken)
    assert report.passed is False
    assert len(report.failures) >= 7
