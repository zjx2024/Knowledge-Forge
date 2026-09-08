"""Deterministic, externally sorted HugeGraph content manifests."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
import unicodedata
from typing import Any, Iterable

from .graph_safety import validate_namespace_id


GRAPH_MANIFEST_PROTOCOL = "kf-graph-manifest-v1"


class DuplicateGraphLogicalId(ValueError):
    """A generation attempted to publish the same logical graph fact twice."""


class GraphManifestAccumulator:
    """Disk-backed logical fact set for bounded writer-side manifest accounting."""

    def __init__(
        self, *, namespace_id: str, project_id: int, graph_generation: int,
        policy_hash: str,
    ) -> None:
        self.namespace_id = validate_namespace_id(namespace_id)
        self.project_id = int(project_id)
        self.graph_generation = int(graph_generation)
        self.policy_hash = str(policy_hash)
        self._directory = tempfile.TemporaryDirectory(prefix="kf-graph-manifest-")
        self._conn = sqlite3.connect(f"{self._directory.name}/manifest.sqlite3")
        self._conn.execute(
            """CREATE TABLE facts(
               logical_key TEXT PRIMARY KEY,kind TEXT NOT NULL,label TEXT NOT NULL,
               payload BLOB NOT NULL)"""
        )

    @staticmethod
    def _prepare(raw: dict[str, Any]) -> tuple[str, str, str, bytes]:
        record = _normalize(raw)
        kind = str(record.get("kind") or "")
        label = str(record.get("label") or "")
        logical_id = str(record.get("logical_id") or "")
        if kind not in {"vertex", "edge"} or not label or not logical_id:
            raise ValueError("graph manifest fact requires kind, label and logical_id")
        return f"{kind}\x1f{label}\x1f{logical_id}", kind, label, _encoded(record)

    def add(self, raw: dict[str, Any], *, replace: bool = False) -> None:
        logical_key, kind, label, payload = self._prepare(raw)
        statement = (
            "INSERT OR REPLACE INTO facts(logical_key,kind,label,payload) VALUES(?,?,?,?)"
            if replace else
            "INSERT INTO facts(logical_key,kind,label,payload) VALUES(?,?,?,?)"
        )
        try:
            self._conn.execute(statement, (logical_key, kind, label, payload))
        except sqlite3.IntegrityError as exc:
            raise DuplicateGraphLogicalId(logical_key) from exc

    def manifest(self) -> dict[str, Any]:
        self._conn.commit()
        digest = hashlib.sha256()
        total = int(self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
        counts = {
            f"{kind}:{label}": int(count)
            for kind, label, count in self._conn.execute(
                "SELECT kind,label,COUNT(*) FROM facts GROUP BY kind,label ORDER BY kind,label"
            )
        }
        for logical_key, payload in self._conn.execute(
            "SELECT logical_key,payload FROM facts ORDER BY logical_key"
        ):
            key_bytes = logical_key.encode("utf-8")
            payload_bytes = bytes(payload)
            digest.update(len(key_bytes).to_bytes(8, "big")); digest.update(key_bytes)
            digest.update(len(payload_bytes).to_bytes(8, "big")); digest.update(payload_bytes)
        return {
            "protocol": GRAPH_MANIFEST_PROTOCOL,
            "namespace_id": self.namespace_id,
            "project_id": self.project_id,
            "graph_generation": self.graph_generation,
            "policy_hash": self.policy_hash,
            "fact_count": total,
            "counts": counts,
            "content_digest": digest.hexdigest(),
        }

    def close(self) -> None:
        self._conn.close()
        self._directory.cleanup()


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("graph manifest forbids non-finite floats")
        return float(repr(value))
    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise TypeError(f"unsupported graph manifest value: {type(value).__name__}")


def _encoded(value: Any) -> bytes:
    return json.dumps(
        _normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def build_graph_manifest(
    records: Iterable[dict[str, Any]], *, namespace_id: str, project_id: int,
    graph_generation: int, policy_hash: str,
) -> dict[str, Any]:
    """Build one order-independent manifest without retaining the whole graph in RAM."""

    accumulator = GraphManifestAccumulator(
        namespace_id=namespace_id, project_id=project_id,
        graph_generation=graph_generation, policy_hash=policy_hash,
    )
    try:
        for record in records:
            accumulator.add(record)
        return accumulator.manifest()
    finally:
        accumulator.close()


def compare_graph_manifests(
    expected: dict[str, Any], writer_reported: dict[str, Any], actual: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    for name, value in (("writer", writer_reported), ("actual", actual)):
        for key in (
            "protocol", "namespace_id", "project_id", "graph_generation",
            "policy_hash", "fact_count", "counts", "content_digest",
        ):
            if value.get(key) != expected.get(key):
                reasons.append(f"{name}:{key}")
    return {"complete": not reasons, "reasons": reasons}
