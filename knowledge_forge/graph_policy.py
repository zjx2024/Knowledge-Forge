"""Graph publication policy and fail-closed project state helpers."""

from __future__ import annotations

import hashlib
import json

from .relation_verification import RELATION_VERIFICATION_POLICY_VERSION


GRAPH_POLICY_VERSION = "relation-graph-p0-v1"


def graph_policy_hash() -> str:
    """Return the deterministic product graph policy-set fingerprint."""

    payload = {
        "graph_policy": GRAPH_POLICY_VERSION,
        "relation_verification": RELATION_VERIFICATION_POLICY_VERSION,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


GRAPH_POLICY_HASH = graph_policy_hash()

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
        and project.get("graph_policy_hash") == GRAPH_POLICY_HASH
        and int(project.get("graph_generation") or 0)
        == int(project.get("graph_ready_generation") or -1)
    )
