#!/usr/bin/env python3
"""Read-only HugeGraph preflight and guarded arm/cleanup for isolated test graphs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from pyhugegraph.client import PyHugeClient

from knowledge_forge.graph_safety import (
    GraphEndpointIdentity,
    GraphSafetyError,
    HugeGraphTestGuard,
)


def _client(args) -> PyHugeClient:
    with contextlib.redirect_stderr(io.StringIO()):
        return PyHugeClient(
            ip=args.host, port=str(args.port), graph=args.graph,
            user=args.user, pwd=args.password, timeout=args.timeout,
        )


def _headers(args) -> dict[str, str]:
    return {"Accept": "application/json"}


def _auth(args) -> tuple[str, str] | None:
    return (args.user, args.password) if args.user else None


def _base(args) -> str:
    return f"http://{args.host}:{args.port}"


def _endpoint_payload(args) -> dict[str, Any]:
    graphs = requests.get(
        f"{_base(args)}/graphs", headers=_headers(args), timeout=args.timeout,
        auth=_auth(args),
    )
    graphs.raise_for_status()
    if args.graph not in (graphs.json().get("graphs") or []):
        raise GraphSafetyError(f"HugeGraph graph does not exist: {args.graph}")
    version = "verified"
    try:
        response = requests.get(
            f"{_base(args)}/versions", headers=_headers(args), timeout=args.timeout,
            auth=_auth(args),
        )
        if response.ok:
            version = json.dumps(response.json(), sort_keys=True)
    except requests.RequestException:
        pass
    return {"version": version, "graph": args.graph}


def _production_identity(args) -> GraphEndpointIdentity | None:
    value = args.production_identity or os.getenv("KNOWLEDGE_FORGE_PRODUCTION_HUGEGRAPH_IDENTITY", "")
    if not value:
        return None
    from urllib.parse import urlparse
    parsed = urlparse(value)
    graph = parsed.path.strip("/")
    return GraphEndpointIdentity(parsed.scheme, parsed.hostname or "", parsed.port or 80, graph)


def _guard(args, *, destructive: bool) -> HugeGraphTestGuard:
    return HugeGraphTestGuard.preflight(
        identity=GraphEndpointIdentity("http", args.host, args.port, args.graph),
        production_identity=_production_identity(args), namespace_id=args.namespace,
        run_id=args.run_id or "read-only-preflight",
        integration_enabled=(not destructive) or os.getenv("RUN_HUGEGRAPH_INTEGRATION") == "1",
        destructive_enabled=(not destructive) or os.getenv("RUN_HUGEGRAPH_DESTRUCTIVE_TESTS") == "1",
        endpoint_payload=_endpoint_payload(args), destructive=destructive,
    )


def _empty_counts(args) -> tuple[int, int]:
    counts = []
    for kind in ("vertices", "edges"):
        response = requests.get(
            f"{_base(args)}/graphs/{quote(args.graph, safe='')}/graph/{kind}",
            params={"limit": 1}, headers=_headers(args), timeout=args.timeout,
            auth=_auth(args),
        )
        response.raise_for_status()
        payload = response.json()
        counts.append(len(payload.get(kind) or payload.get("data") or []))
    return int(counts[0]), int(counts[1])


def _ensure_sentinel_schema(client: PyHugeClient) -> None:
    schema = client.schema()
    for name in (
        "kf_test_purpose", "kf_test_run_id", "kf_test_namespace_id",
        "kf_test_physical_identity", "kf_test_config_hash", "kf_test_expires_at",
    ):
        schema.propertyKey(name).asText().ifNotExist().create()
    schema.vertexLabel("KFTestSentinel").properties(
        "kf_test_purpose", "kf_test_run_id", "kf_test_namespace_id",
        "kf_test_physical_identity", "kf_test_config_hash", "kf_test_expires_at",
    ).useCustomizeStringId().ifNotExist().create()


def _sentinel_id(namespace: str) -> str:
    return f"kf:test-sentinel:{namespace}"


def _read_sentinel(args) -> dict[str, Any] | None:
    try:
        vertex = _client(args).graph().getVertexById(_sentinel_id(args.namespace))
    except Exception:
        return None
    if not vertex:
        return None
    properties = dict(vertex.properties or {})
    return {
        "purpose": properties.get("kf_test_purpose"),
        "run_id": properties.get("kf_test_run_id"),
        "namespace_id": properties.get("kf_test_namespace_id"),
        "physical_identity": properties.get("kf_test_physical_identity"),
        "config_hash": properties.get("kf_test_config_hash"),
        "expires_at": properties.get("kf_test_expires_at"),
    }


def _write_sentinel(args, value: dict[str, Any]) -> None:
    client = _client(args); _ensure_sentinel_schema(client)
    client.graph().addVertex("KFTestSentinel", {
        "kf_test_purpose": value["purpose"],
        "kf_test_run_id": value["run_id"],
        "kf_test_namespace_id": value["namespace_id"],
        "kf_test_physical_identity": value["physical_identity"],
        "kf_test_config_hash": value["config_hash"],
        "kf_test_expires_at": value["expires_at"],
    }, id=_sentinel_id(args.namespace))


def _delete_namespace(args, namespace: str) -> None:
    graph = _client(args).graph()
    edge_ids: list[str] = []
    vertex_ids: list[str] = []
    for label in ("kf_related", "kf_mentions", "kf_contains"):
        page = None
        while True:
            result = graph.getEdgeByPage(
                label=label, properties={"kf_namespace_id": namespace}, limit=500, page=page,
            )
            edges, next_page = result if isinstance(result, tuple) else (result, None)
            edge_ids.extend(str(edge.id) for edge in edges or [])
            if not next_page:
                break
            page = next_page
    for label in ("KFChunk", "KFDocument", "KFEntity"):
        page = None
        while True:
            result = graph.getVertexByPage(
                label=label, properties={"kf_namespace_id": namespace}, limit=500, page=page,
            )
            vertices, next_page = result if isinstance(result, tuple) else (result, None)
            vertex_ids.extend(str(vertex.id) for vertex in vertices or [])
            if not next_page:
                break
            page = next_page
    # Pagination tokens can be invalidated by mutation.  Collect and deduplicate
    # the entire owned namespace before performing the first destructive call.
    for identifier in dict.fromkeys(edge_ids):
        graph.removeEdgeById(identifier)
    for identifier in dict.fromkeys(vertex_ids):
        graph.removeVertexById(identifier)
    graph.removeVertexById(_sentinel_id(namespace))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "arm", "cleanup"):
        command = commands.add_parser(name)
        command.add_argument("--host", required=True)
        command.add_argument("--port", type=int, default=8080)
        command.add_argument("--graph", required=True)
        command.add_argument("--namespace", required=True)
        command.add_argument("--run-id", default="")
        command.add_argument("--sentinel", default="")
        command.add_argument("--production-identity", default="")
        command.add_argument("--user", default=os.getenv("HUGEGRAPH_USER", ""))
        command.add_argument("--password", default=os.getenv("HUGEGRAPH_PASSWORD", ""))
        command.add_argument("--timeout", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            guard = _guard(args, destructive=False)
            print(json.dumps({
                "physical_identity": guard.identity.canonical(),
                "namespace": guard.namespace_id, "mutation_count": 0,
                "destructive": False, "status": "read_only_verified",
            }, indent=2, sort_keys=True))
            return 0
        if not args.run_id:
            raise GraphSafetyError("--run-id is required")
        if args.command == "arm":
            guard = _guard(args, destructive=True)
            token = guard.arm(
                read_counts=lambda: _empty_counts(args),
                write_sentinel=lambda value: _write_sentinel(args, value),
                read_sentinel=lambda: _read_sentinel(args),
            )
            print(json.dumps({"sentinel": token, "run_id": args.run_id,
                              "namespace": args.namespace}, indent=2, sort_keys=True))
            return 0
        if not args.sentinel:
            raise GraphSafetyError("--sentinel is required")
        _guard(args, destructive=True)
        stored = _read_sentinel(args) or {}
        guard = HugeGraphTestGuard(
            identity=GraphEndpointIdentity("http", args.host, args.port, args.graph),
            namespace_id=args.namespace, run_id=args.run_id, destructive=True,
            config_hash=str(stored.get("config_hash") or ""),
            expires_at=str(stored.get("expires_at") or ""),
        )
        guard.cleanup(
            args.sentinel, read_sentinel=lambda: _read_sentinel(args),
            delete_namespace=lambda namespace: _delete_namespace(args, namespace),
        )
        print(json.dumps({"status": "cleaned", "namespace": args.namespace}, indent=2))
        return 0
    except (GraphSafetyError, requests.RequestException) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
