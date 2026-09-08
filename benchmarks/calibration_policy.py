"""Shared uncertainty policy for the frozen calibration corpora.

The calibration labels are useful engineering references, but they are not
independently adjudicated human Gold.  Directly entailed facts are therefore
the primary score, while interpretation-sensitive facts remain a non-gating
sensitivity band.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


INTERPRETATION_TIER_PATH = (
    Path("annotations") / "adjudication_tiers_2026_08_11.json"
)


def assertion_fact_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return the graph-level directed fact key for an assertion row."""

    return (
        str(row["subject_canonical_id"]),
        str(row["predicate"]),
        str(row["object_canonical_id"]),
    )


def interpretation_assertion_ids(
    dataset_root: Path,
    assertions: Iterable[dict[str, Any]],
) -> set[str]:
    """Resolve interpretation-band IDs from inline and sidecar metadata."""

    rows = list(assertions)
    known_ids = {str(row["assertion_id"]) for row in rows}
    interpretation_ids = {
        str(row["assertion_id"])
        for row in rows
        if str(row.get("adjudication_tier") or "core_direct")
        != "core_direct"
    }
    tier_path = dataset_root / INTERPRETATION_TIER_PATH
    if tier_path.is_file():
        payload = json.loads(tier_path.read_text(encoding="utf-8"))
        interpretation_ids.update(
            str(item["assertion_id"])
            for item in payload.get("interpretation_band") or ()
        )
    unknown = interpretation_ids - known_ids
    if unknown:
        raise ValueError(
            "interpretation tier references unknown assertions: "
            + ", ".join(sorted(unknown))
        )
    return interpretation_ids


def partition_assertions(
    dataset_root: Path,
    assertions: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Split assertions into the primary and sensitivity evaluation bands."""

    rows = list(assertions)
    interpretation_ids = interpretation_assertion_ids(dataset_root, rows)
    return {
        "core_direct": [
            row for row in rows
            if str(row["assertion_id"]) not in interpretation_ids
        ],
        "interpretation_band": [
            row for row in rows
            if str(row["assertion_id"]) in interpretation_ids
        ],
        "all_labels": rows,
    }


def unique_fact_rows(
    assertions: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Deduplicate evidence occurrences while retaining one representative."""

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in assertions:
        result.setdefault(assertion_fact_key(row), row)
    return result
