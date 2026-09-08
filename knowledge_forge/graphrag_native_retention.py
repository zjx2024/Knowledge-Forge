"""Conservative retention planning for immutable GraphRAG Native runs.

This module only classifies terminal runs.  It does not mutate PostgreSQL,
legacy graph state, or bundles; the operator script is responsible for an
explicit, one-run-at-a-time physical deletion after a second confirmation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


REMOVABLE_TERMINAL_STATES = frozenset({
    "FAILED", "CANCELLED", "STALE", "SUPERSEDED", "EMPTY_INPUT",
    "NO_GRAPH_UNSUPPORTED", "COMPLETED_WITH_FAILURES",
})


def _as_utc(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def classify_retention_candidates(
    runs: list[dict[str, Any]], *, active_run_id: str | None, now: datetime | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return dry-run candidates and protected rows with explicit reasons.

    The last READY run is protected even when it is not the active pointer, so
    an operator always has a rollback-capable native bundle.  A live lease is
    protected regardless of terminal state.
    """

    now = now or datetime.now(timezone.utc)
    ready = [row for row in runs if str(row.get("state") or "") == "READY"]
    last_ready = max(
        ready,
        key=lambda row: _as_utc(row.get("ready_at") or row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc),
        default=None,
    )
    last_ready_id = str(last_ready.get("run_id") or "") if last_ready else ""
    candidates: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for row in runs:
        item = dict(row)
        run_id = str(item.get("run_id") or "")
        state = str(item.get("state") or "")
        lease = _as_utc(item.get("lease_expires_at"))
        reasons: list[str] = []
        if run_id and run_id == str(active_run_id or ""):
            reasons.append("active_run")
        if run_id and run_id == last_ready_id:
            reasons.append("last_ready_run")
        if lease and lease > now:
            reasons.append("live_lease")
        if state not in REMOVABLE_TERMINAL_STATES:
            reasons.append("nonremovable_state:" + (state or "unknown"))
        if reasons:
            protected.append({"run": item, "reasons": reasons})
        else:
            candidates.append({"run": item, "reasons": ["terminal_nonactive_unleased"]})
    return {"candidates": candidates, "protected": protected}
