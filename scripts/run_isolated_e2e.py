#!/usr/bin/env python3
"""Preflight and execute the isolated Knowledge Forge E2E verification matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg
from psycopg.conninfo import conninfo_to_dict
import requests
from psycopg import sql
from psycopg.rows import dict_row
from pyhugegraph.client import PyHugeClient

from knowledge_forge.config import Config
from knowledge_forge.e2e_safety import (
    build_preflight_manifest,
    issue_execution_token,
    verify_postgres_sentinel_record,
    verify_execution_token,
)
from knowledge_forge.graph_safety import GraphEndpointIdentity, HugeGraphTestGuard


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_TESTS = (
    "tests/test_document_content_revisions.py",
    "tests/test_relation_shadow_recovery.py",
    "tests/test_entity_shadow_independence.py",
    "tests/test_policy_snapshot_orchestrator.py",
    "tests/test_release_gate_artifacts.py",
    "tests/test_graph_manifest_validation.py",
    "tests/test_p1_graph_generation.py",
    "tests/test_p02_graph_state.py",
    "tests/test_p2_merge_split.py",
    "tests/test_ollama_token_contract.py",
    "tests/test_p1_postgres_plans.py",
    "tests/test_entity_shadow_postgres_plans.py",
    "tests/test_hugegraph_safety_guard.py",
    "tests/test_e2e_safety.py",
)


def _output(*command: str) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _postgres_identity(dsn: str, schema: str) -> tuple[str, str, str]:
    if not dsn:
        return "", "", ""
    values = conninfo_to_dict(dsn)
    safe = {
        "host": values.get("host") or "localhost",
        "port": int(values.get("port") or 5432),
        "user": values.get("user") or "",
        "dbname": values.get("dbname") or "",
    }
    pgvector = ""
    try:
        with psycopg.connect(dsn) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            row = connection.execute(
                "SELECT current_user, current_database(), inet_server_addr()::text, "
                "inet_server_port(), version()"
            ).fetchone()
            safe.update(user=row[0], dbname=row[1], host=row[2] or safe["host"], port=row[3])
            version = connection.execute(
                "SELECT extversion FROM pg_extension WHERE extname='vector'"
            ).fetchone()
            pgvector = str(version[0]) if version else ""
    except Exception:
        return "", "", ""
    return (
        f"postgresql://{safe['user']}@{safe['host']}:{safe['port']}/{safe['dbname']}/{schema}",
        str(row[4]),
        pgvector,
    )


def _postgres_sentinel_verified(
    dsn: str, *, schema: str, run_id: str, identity: str,
) -> bool:
    config_hash = os.getenv("E2E_POSTGRES_SENTINEL_CONFIG_HASH", "").strip()
    role = os.getenv("E2E_POSTGRES_ROLE", "").strip()
    if not all((dsn, schema, run_id, identity, config_hash, role)):
        return False
    try:
        with psycopg.connect(dsn, row_factory=dict_row) as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            record = connection.execute(sql.SQL(
                "SELECT run_id,identity,schema_name,role_name,config_hash,expires_at "
                "FROM {}.kf_e2e_sentinel"
            ).format(sql.Identifier(schema))).fetchone()
    except psycopg.Error:
        return False
    return bool(record) and verify_postgres_sentinel_record(
        record, run_id=run_id, identity=identity, schema=schema, role=role,
        config_hash=config_hash,
    )


def _configured_production_postgres_identity() -> str:
    explicit = os.getenv("KNOWLEDGE_FORGE_PRODUCTION_POSTGRES_IDENTITY", "").strip()
    if explicit:
        return explicit
    values = Config.postgres
    return (
        f"postgresql://{values['user']}@{values['host']}:{values['port']}/"
        f"{values['dbname']}/public"
    )


def _ollama_models() -> dict[str, dict[str, Any]]:
    endpoint = os.getenv("E2E_OLLAMA_BASE_URL", "").rstrip("/")
    requested = {
        "llm": os.getenv("E2E_LLM_MODEL", ""),
        "embedding": os.getenv("E2E_EMBEDDING_MODEL", ""),
        "reranker": os.getenv("E2E_RERANKER_MODEL", ""),
    }
    identities: dict[str, dict[str, Any]] = {}
    tags: dict[str, dict[str, Any]] = {}
    running: dict[str, dict[str, Any]] = {}
    if endpoint:
        try:
            response = requests.get(f"{endpoint}/api/tags", timeout=5)
            response.raise_for_status()
            tags = {str(item.get("name")): item for item in response.json().get("models") or []}
            ps_response = requests.get(f"{endpoint}/api/ps", timeout=5)
            ps_response.raise_for_status()
            running = {str(item.get("name")): item for item in ps_response.json().get("models") or []}
        except Exception:
            tags = {}; running = {}
    for role, model in requested.items():
        tag = tags.get(model) or {}
        active = running.get(model) or {}
        shown: dict[str, Any] = {}
        if endpoint and model:
            try:
                show_response = requests.post(
                    f"{endpoint}/api/show", json={"model": model}, timeout=5,
                )
                show_response.raise_for_status()
                shown = show_response.json()
            except Exception:
                shown = {}
        model_info = shown.get("model_info") or {}
        context = int(active.get("context_length") or 0)
        if not context:
            context = max(
                (
                    int(value)
                    for key, value in model_info.items()
                    if str(key).endswith(".context_length") and str(value).isdigit()
                ),
                default=0,
            )
        identities[role] = {
            "endpoint": endpoint,
            "model": model,
            "digest": str(
                tag.get("digest") or active.get("digest") or shown.get("digest") or ""
            ),
            "context": context,
        }
    return identities


def _hugegraph_identity() -> tuple[str, bool]:
    host = os.getenv("E2E_HUGEGRAPH_HOST", "").strip()
    port = int(os.getenv("E2E_HUGEGRAPH_PORT", "8080"))
    graph = os.getenv("E2E_HUGEGRAPH_GRAPH", "").strip()
    if not host or not graph:
        return "", False
    identity = f"http://{host}:{port}/{graph}"
    user = os.getenv("E2E_HUGEGRAPH_USER", "")
    password = os.getenv("E2E_HUGEGRAPH_PASSWORD", "")
    auth = (user, password) if user else None
    try:
        response = requests.get(f"http://{host}:{port}/graphs", timeout=5, auth=auth)
        response.raise_for_status()
        exists = graph in (response.json().get("graphs") or [])
    except Exception:
        exists = False
    if not exists:
        return "", False
    namespace = os.getenv("E2E_GRAPH_NAMESPACE", "").strip()
    run_id = os.getenv("E2E_RUN_ID", "").strip()
    token = os.getenv("E2E_GRAPH_SENTINEL", "").strip()
    if not all((namespace, run_id, token)):
        return identity, False
    try:
        client = PyHugeClient(
            ip=host, port=str(port), graph=graph, user=user, pwd=password, timeout=5,
        )
        vertex = client.graph().getVertexById(f"kf:test-sentinel:{namespace}")
        properties = dict(vertex.properties or {}) if vertex else {}
        guard = HugeGraphTestGuard(
            identity=GraphEndpointIdentity("http", host, port, graph),
            namespace_id=namespace, run_id=run_id, destructive=True,
            config_hash=str(properties.get("kf_test_config_hash") or ""),
            expires_at=str(properties.get("kf_test_expires_at") or ""),
        )
        sentinel = {
            "purpose": properties.get("kf_test_purpose"),
            "run_id": properties.get("kf_test_run_id"),
            "namespace_id": properties.get("kf_test_namespace_id"),
            "physical_identity": properties.get("kf_test_physical_identity"),
            "config_hash": properties.get("kf_test_config_hash"),
            "expires_at": properties.get("kf_test_expires_at"),
        }
        guard.authorize(token, read_sentinel=lambda: sentinel)
        return identity, True
    except Exception:
        return identity, False


def _api_identity() -> tuple[str, str]:
    endpoint = os.getenv("E2E_API_BASE_URL", "").rstrip("/")
    if not endpoint:
        return "", "endpoint is missing"
    parsed = urlparse(endpoint)
    warning = ""
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("E2E_API_AUTH_TOKEN"):
        warning = "non-loopback endpoint has no E2E_API_AUTH_TOKEN"
    try:
        response = requests.get(f"{endpoint}/", timeout=5)
        response.raise_for_status()
    except Exception:
        warning = warning or "API read-only health probe failed"
    return endpoint, warning


def collect_manifest() -> dict[str, Any]:
    run_id = os.getenv("E2E_RUN_ID", "").strip()
    schema = os.getenv("E2E_POSTGRES_SCHEMA", "").strip()
    target_dsn = os.getenv("KNOWLEDGE_FORGE_E2E_POSTGRES_DSN", "").strip()
    postgres_identity, postgres_server_version, pgvector_version = _postgres_identity(
        target_dsn, schema,
    )
    postgres_sentinel = _postgres_sentinel_verified(
        target_dsn, schema=schema, run_id=run_id, identity=postgres_identity,
    )
    graph_identity, graph_sentinel = _hugegraph_identity()
    api_identity, api_warning = _api_identity()
    config_snapshot = {
        "postgres": postgres_identity,
        "graph": graph_identity,
        "namespace": os.getenv("E2E_GRAPH_NAMESPACE", ""),
        "api": api_identity,
        "models": _ollama_models(),
        "pgvector": pgvector_version,
        "limits": {
            "documents": int(os.getenv("E2E_MAX_DOCUMENTS", "10")),
            "reranker_top_k": Config.entity_reranker_top_k,
            "max_context": Config.ollama_default_context_budget,
        },
    }
    policy_snapshot = {
        "relation_shadow": Config.relation_shadow_enabled,
        "entity_merge_threshold": Config.entity_merge_threshold,
        "schema_auto_apply": Config.schema_auto_apply,
    }
    git_commit = _output("git", "rev-parse", "HEAD")
    git_tree = "dirty" if _output("git", "status", "--porcelain") else "clean"
    return build_preflight_manifest(
        run_id=run_id,
        postgres_identity=postgres_identity,
        production_postgres_identity=_configured_production_postgres_identity(),
        graph_identity=graph_identity,
        production_graph_identity=os.getenv(
            "KNOWLEDGE_FORGE_PRODUCTION_HUGEGRAPH_IDENTITY",
            f"http://{Config.hugegraph_host}:{Config.hugegraph_port}/{Config.hugegraph_graph}",
        ),
        graph_namespace=os.getenv("E2E_GRAPH_NAMESPACE", ""),
        api_identity=api_identity,
        model_identities=config_snapshot["models"],
        git_commit=git_commit, git_tree=git_tree,
        policy_hash=_hash(policy_snapshot), config_hash=_hash(config_snapshot),
        postgres_sentinel=postgres_sentinel,
        postgres_server_version=postgres_server_version,
        pgvector_version=pgvector_version,
        production_model_endpoints={
            role: os.getenv(
                f"E2E_PRODUCTION_{role.upper()}_ENDPOINT",
                Config.ollama_base_url if role in {"llm", "embedding"} else Config.entity_reranker_model_path,
            )
            for role in ("llm", "embedding", "reranker")
        },
        graph_sentinel=graph_sentinel,
        integration_enabled=os.getenv("RUN_HUGEGRAPH_INTEGRATION", "false").lower() in {"1", "true"},
        destructive_enabled=os.getenv("RUN_HUGEGRAPH_DESTRUCTIVE_TESTS", "false").lower() in {"1", "true"},
        api_bind_warning=api_warning, limits=config_snapshot["limits"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--arm-token", default=os.getenv("E2E_EXECUTION_TOKEN", ""))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    manifest = collect_manifest()
    report: dict[str, Any] = {"preflight": manifest}
    secret = os.getenv("E2E_ARM_SECRET", "")
    if args.preflight_only:
        if manifest["safe_to_arm"] and secret:
            report["execution_token"] = issue_execution_token(manifest, secret=secret)
        elif manifest["safe_to_arm"]:
            report["preflight"]["blockers"].append("E2E_ARM_SECRET is missing")
            report["preflight"]["safe_to_arm"] = False
        text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        print(text)
        if args.report:
            args.report.write_text(text + "\n", encoding="utf-8")
        return 0 if report["preflight"]["safe_to_arm"] else 2

    if not secret or not args.arm_token:
        print("E2E_ARM_SECRET and --arm-token are required", file=sys.stderr)
        return 2
    verify_execution_token(args.arm_token, current_manifest=manifest, secret=secret)
    command = [sys.executable, "-m", "pytest", "-q", "-ra", "--strict-markers", *SCENARIO_TESTS]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    report["execution"] = {
        "command": command, "returncode": completed.returncode,
        "scenario_tests": list(SCENARIO_TESTS),
    }
    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
