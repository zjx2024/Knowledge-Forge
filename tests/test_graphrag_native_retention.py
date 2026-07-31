"""Retention never proposes an active, last-ready, or leased native run."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from knowledge_forge.graphrag_native_retention import classify_retention_candidates


def test_retention_dry_run_keeps_active_last_ready_and_live_lease():
    now = datetime(2026, 7, 28, tzinfo=timezone.utc)
    plan = classify_retention_candidates([
        {"run_id": "active", "state": "STALE", "created_at": now - timedelta(days=3)},
        {"run_id": "last-ready", "state": "READY", "ready_at": now - timedelta(days=1)},
        {"run_id": "leased", "state": "FAILED", "lease_expires_at": now + timedelta(minutes=5)},
        {"run_id": "old-failed", "state": "FAILED", "created_at": now - timedelta(days=5)},
        {"run_id": "running", "state": "RUNNING", "created_at": now - timedelta(days=2)},
    ], active_run_id="active", now=now)

    assert [item["run"]["run_id"] for item in plan["candidates"]] == ["old-failed"]
    reasons = {item["run"]["run_id"]: set(item["reasons"]) for item in plan["protected"]}
    assert "active_run" in reasons["active"]
    assert "last_ready_run" in reasons["last-ready"]
    assert "live_lease" in reasons["leased"]
    assert "nonremovable_state:RUNNING" in reasons["running"]
