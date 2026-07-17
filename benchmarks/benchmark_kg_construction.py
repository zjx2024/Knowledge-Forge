"""Deterministic operation-count benchmark; it does not call external services."""

from __future__ import annotations

import argparse
import json


def relation_scenario(entity_count: int, documents: int) -> dict[str, int]:
    # Characterization denominator only. It intentionally does not claim that
    # all pairs are semantically eligible or should be sent to a model.
    directed_pairs = max(0, entity_count * (entity_count - 1))
    return {
        "documents": documents,
        "entities_per_chunk": entity_count,
        "raw_directed_pairs_per_chunk": directed_pairs,
        "legacy_full_rebuild_requests": documents,
        "coalesced_target_rebuild_requests": 1 if documents else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [relation_scenario(entity_count, documents) for documents in (1, 3, 10) for entity_count in (0, 1, 2, 20, 50, 100)]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
