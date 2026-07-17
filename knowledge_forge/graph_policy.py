"""Graph publication policy and fail-closed project state helpers."""

from __future__ import annotations


GRAPH_POLICY_VERSION = "relation-graph-p0-v1"

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
    )
