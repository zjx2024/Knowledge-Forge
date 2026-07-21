"""Fail-closed identities, arm tokens and performance gates for isolated E2E runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import hmac
import json
import re
from typing import Any, Mapping
from uuid import UUID


class E2ESafetyError(RuntimeError):
    """An E2E operation could not prove that it owns an isolated target."""


REQUIRED_E2E_SCENARIOS = (
    "upload_1_3_10_documents",
    "bilingual_table_dense_repeated_url_version_decimal",
    "span_first_relation_only_end_to_end_shadow",
    "three_reingestions",
    "chunk_failure_resume",
    "worker_restart",
    "concurrent_reingestion_out_of_order",
    "delete_restore_reingest",
    "outbox_embedding_reranker_model_timeout",
    "graph_partial_failure",
    "graph_equal_counts_wrong_content",
    "graph_lease_expiry_retry",
    "graph_409_rebuild_manifest_ready",
    "retrieval_list_statistics_id_graph_consistency",
    "control_shadow_no_pollution",
    "ten_document_rebuild_coalescing",
    "namespace_cleanup_idempotent",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _valid_uuid(value: str) -> bool:
    try:
        UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def build_preflight_manifest(
    *, run_id: str, postgres_identity: str, production_postgres_identity: str,
    graph_identity: str, production_graph_identity: str, graph_namespace: str,
    api_identity: str, model_identities: Mapping[str, Mapping[str, Any]],
    git_commit: str, git_tree: str, policy_hash: str, config_hash: str,
    postgres_sentinel: bool, graph_sentinel: bool,
    integration_enabled: bool, destructive_enabled: bool,
    api_bind_warning: str, limits: Mapping[str, int],
    postgres_server_version: str = "", pgvector_version: str = "",
    production_model_endpoints: Mapping[str, str] | None = None,
    scenarios: tuple[str, ...] = REQUIRED_E2E_SCENARIOS,
) -> dict[str, Any]:
    """Build a redacted read-only report and fail-closed arm decision."""

    blockers: list[str] = []
    if not run_id:
        blockers.append("run_id is missing")
    if not postgres_identity:
        blockers.append("test PostgreSQL identity is unknown")
    if not production_postgres_identity:
        blockers.append("production PostgreSQL identity is unknown")
    elif postgres_identity == production_postgres_identity:
        blockers.append("PostgreSQL test and production identities are equal")
    if not graph_identity:
        blockers.append("test HugeGraph identity is unknown")
    if not production_graph_identity:
        blockers.append("production HugeGraph identity is unknown")
    elif graph_identity == production_graph_identity:
        blockers.append("HugeGraph test and production identities are equal")
    if graph_identity.rstrip("/").endswith("/knowledge_forge"):
        blockers.append("default production HugeGraph graph is forbidden")
    if not _valid_uuid(graph_namespace):
        blockers.append("HugeGraph namespace must be a UUID")
    if not postgres_sentinel:
        blockers.append("PostgreSQL sentinel is not verified")
    if not postgres_server_version:
        blockers.append("PostgreSQL server version is unknown")
    if not pgvector_version:
        blockers.append("pgvector version is unknown")
    if not graph_sentinel:
        blockers.append("HugeGraph sentinel is not verified")
    if not integration_enabled:
        blockers.append("integration switch is disabled")
    if not destructive_enabled:
        blockers.append("destructive switch is disabled")
    if api_bind_warning:
        blockers.append(f"API bind/auth is unsafe: {api_bind_warning}")
    production_model_endpoints = dict(production_model_endpoints or {})
    for role in ("llm", "embedding", "reranker"):
        identity = dict(model_identities.get(role) or {})
        missing = [key for key in ("endpoint", "model", "digest", "context") if not identity.get(key)]
        if missing:
            blockers.append(f"{role} identity is incomplete: {','.join(missing)}")
        production_endpoint = str(production_model_endpoints.get(role) or "").rstrip("/")
        test_endpoint = str(identity.get("endpoint") or "").rstrip("/")
        if not production_endpoint:
            blockers.append(f"{role} production endpoint is unknown")
        elif test_endpoint and test_endpoint == production_endpoint:
            blockers.append(f"{role} test and production endpoints are equal")

    report = {
        "version": 1,
        "run_id": run_id,
        "postgres_identity": postgres_identity,
        "production_postgres_identity": production_postgres_identity,
        "graph_identity": graph_identity,
        "production_graph_identity": production_graph_identity,
        "graph_namespace": graph_namespace,
        "api_identity": api_identity,
        "model_identities": {key: dict(value) for key, value in model_identities.items()},
        "git_commit": git_commit,
        "git_tree": git_tree,
        "policy_hash": policy_hash,
        "config_hash": config_hash,
        "sentinels": {"postgres": bool(postgres_sentinel), "hugegraph": bool(graph_sentinel)},
        "flags": {
            "integration": bool(integration_enabled),
            "destructive": bool(destructive_enabled),
        },
        "api_bind_warning": api_bind_warning,
        "limits": {key: int(value) for key, value in limits.items()},
        "postgres_server_version": postgres_server_version,
        "pgvector_version": pgvector_version,
        "production_model_endpoints": production_model_endpoints,
        "scenarios": list(scenarios),
        "blockers": blockers,
        "safe_to_arm": not blockers,
    }
    report["manifest_hash"] = _sha256(report)
    return report


def verify_postgres_sentinel_record(
    record: Mapping[str, Any], *, run_id: str, identity: str, schema: str,
    role: str, config_hash: str, now: datetime | None = None,
) -> bool:
    """Validate data actually read from the random-schema sentinel table."""

    now = now or datetime.now(timezone.utc)
    expiry = record.get("expires_at")
    if isinstance(expiry, str):
        try:
            expiry = datetime.fromisoformat(expiry)
        except ValueError:
            return False
    if not isinstance(expiry, datetime) or expiry.tzinfo is None:
        return False
    expected = {
        "run_id": run_id, "identity": identity, "schema_name": schema,
        "role_name": role, "config_hash": config_hash,
    }
    return all(str(record.get(key) or "") == str(value) for key, value in expected.items()) and expiry > now


def issue_execution_token(
    manifest: Mapping[str, Any], *, secret: str,
    now: datetime | None = None, ttl_seconds: int = 300,
) -> str:
    if not manifest.get("safe_to_arm"):
        raise E2ESafetyError("preflight is not safe to arm")
    if len(secret) < 12:
        raise E2ESafetyError("E2E arm secret must contain at least 12 characters")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise E2ESafetyError("token issue time must be timezone-aware")
    ttl_seconds = max(1, min(int(ttl_seconds), 900))
    payload = {
        "version": 1,
        "run_id": manifest.get("run_id"),
        "environment_hash": _sha256(manifest),
        "issued_at": int(now.timestamp()),
        "expires_at": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    encoded = _urlsafe_encode(_canonical(payload))
    signature = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    return f"{encoded}.{_urlsafe_encode(signature)}"


def verify_execution_token(
    token: str, *, current_manifest: Mapping[str, Any], secret: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    try:
        encoded, supplied_signature = token.split(".", 1)
        payload = json.loads(_urlsafe_decode(encoded))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise E2ESafetyError("invalid execution token") from exc
    expected = hmac.new(secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256).digest()
    try:
        supplied = _urlsafe_decode(supplied_signature)
    except (ValueError, TypeError) as exc:
        raise E2ESafetyError("invalid execution token signature") from exc
    if not hmac.compare_digest(expected, supplied):
        raise E2ESafetyError("invalid execution token signature")
    now = now or datetime.now(timezone.utc)
    if int(payload.get("expires_at") or 0) < int(now.timestamp()):
        raise E2ESafetyError("execution token expired")
    if payload.get("run_id") != current_manifest.get("run_id"):
        raise E2ESafetyError("execution token environment changed: run_id")
    if payload.get("environment_hash") != _sha256(current_manifest):
        raise E2ESafetyError("execution token environment changed")
    if not current_manifest.get("safe_to_arm"):
        raise E2ESafetyError("current environment is no longer safe to arm")
    return payload


@dataclass(frozen=True)
class PostgresCleanupPlan:
    schema: str
    role: str
    run_id: str
    sentinel_verified: bool

    def operations(self) -> tuple[tuple[str, str], ...]:
        if not self.sentinel_verified:
            raise E2ESafetyError("cleanup requires a reverified PostgreSQL sentinel")
        if not re.fullmatch(r"[a-f0-9]{16,64}", self.run_id):
            raise E2ESafetyError("cleanup run_id is not a bounded random identifier")
        if self.schema != f"kf_e2e_{self.run_id}":
            raise E2ESafetyError("cleanup schema is outside the owned namespace")
        if self.role != f"kf_e2e_role_{self.run_id}":
            raise E2ESafetyError("cleanup role is outside the owned namespace")
        return (("drop_schema", self.schema), ("drop_role", self.role))


@dataclass(frozen=True)
class PerformanceObservation:
    context_occurrences: int
    request_tokens: int
    verified_context: int
    exact_hit_embedding_calls: int
    exact_hit_reranker_calls: int
    exact_hit_llm_calls: int
    reranker_pairs: int
    mentions: int
    configured_top_k: int
    project_full_scan: bool
    relation_filter_before_limit: bool
    rebuilds_for_ten_documents: int
    bounded_queues: bool
    deferred_terminal: int


@dataclass(frozen=True)
class PerformanceGateReport:
    passed: bool
    failures: tuple[str, ...]
    observation: dict[str, Any]


def evaluate_performance_gates(value: PerformanceObservation) -> PerformanceGateReport:
    failures: list[str] = []
    if value.context_occurrences != 1:
        failures.append("relation prompt context must occur exactly once")
    if value.request_tokens > value.verified_context:
        failures.append("request exceeds verified model context")
    if any((value.exact_hit_embedding_calls, value.exact_hit_reranker_calls, value.exact_hit_llm_calls)):
        failures.append("exact entity hit invoked model retrieval or review")
    if value.reranker_pairs > value.mentions * value.configured_top_k:
        failures.append("reranker pair count exceeds M*K")
    if value.project_full_scan:
        failures.append("entity retrieval performed a project-wide scan")
    if not value.relation_filter_before_limit:
        failures.append("relation vector filtering happened after LIMIT")
    if value.rebuilds_for_ten_documents > 1:
        failures.append("ten-document ingestion was not coalesced")
    if not value.bounded_queues:
        failures.append("cache, queue, lease or outbox is unbounded")
    if value.deferred_terminal:
        failures.append("deferred work did not reach zero or a declared terminal state")
    return PerformanceGateReport(
        passed=not failures, failures=tuple(failures), observation=asdict(value),
    )
