"""Graph publication policy and fail-closed project state helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from .policy import (
    CONTROL_POLICY_SET,
    POLICY_AXES,
    PUBLISH_PREDICATE_VERSION,
    normalize_policy_set,
)


GRAPH_POLICY_VERSION = "relation-graph-p0-v1"


def graph_policy_hash(
    policy_set: Mapping[str, object] = CONTROL_POLICY_SET,
    *,
    schema_version: int = 1,
    config_summary: Mapping[str, object] | None = None,
) -> str:
    """Return the deterministic product graph policy-set fingerprint."""

    payload = {
        "graph_policy": GRAPH_POLICY_VERSION,
        "policy_set": normalize_policy_set(policy_set),
        "publish_predicate": PUBLISH_PREDICATE_VERSION,
        "schema_version": int(schema_version),
        "config": dict(config_summary or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


GRAPH_POLICY_HASH = graph_policy_hash()


def project_policy_set(project: Mapping[str, object] | None) -> dict[str, str]:
    values = project or {}
    return normalize_policy_set({
        axis: values.get(axis) or CONTROL_POLICY_SET[axis]
        for axis in POLICY_AXES
    })


def project_graph_policy_hash(project: Mapping[str, object] | None) -> str:
    values = project or {}
    return graph_policy_hash(
        project_policy_set(values),
        schema_version=int(values.get("active_schema_version") or 1),
    )

GRAPH_EMPTY = "empty"
GRAPH_STALE = "stale"
GRAPH_REBUILDING = "rebuilding"
GRAPH_READY = "ready"
GRAPH_FAILED = "failed"

GRAPH_STATES = frozenset({
    GRAPH_EMPTY,
    GRAPH_STALE,
    GRAPH_REBUILDING,
    GRAPH_READY,
    GRAPH_FAILED,
})


class GraphNotReadyError(RuntimeError):
    """Raised when a caller tries to consume a non-current project graph."""


def is_current_graph(project: dict[str, object] | None) -> bool:
    """Return true only for a fully rebuilt graph under the current policy."""

    return bool(
        project
        and project.get("graph_status") == GRAPH_READY
        and project.get("graph_policy_version") == GRAPH_POLICY_VERSION
        and project.get("graph_policy_hash") == project_graph_policy_hash(project)
        and int(project.get("graph_generation") or 0)
        == int(project.get("graph_ready_generation") or -1)
    )
