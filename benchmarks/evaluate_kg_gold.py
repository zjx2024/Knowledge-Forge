#!/usr/bin/env python3
"""Evaluate a prediction bundle without treating model labels as human Gold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge.quality import (
    b3_metrics, evaluate_entity_mentions, evaluate_relations,
    pairwise_clustering_metrics, validate_gold_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--require-production", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    gold = json.loads((args.bundle / "annotations.json").read_text(encoding="utf-8"))
    predicted = json.loads(args.predictions.read_text(encoding="utf-8"))
    manifest_report = validate_gold_manifest(
        manifest, require_production=args.require_production, bundle_root=args.bundle,
    )
    report = {
        "manifest": manifest_report,
        "mentions": evaluate_entity_mentions(gold.get("mentions", []), predicted.get("mentions", [])),
        "pairwise": pairwise_clustering_metrics(gold.get("clusters", {}), predicted.get("clusters", {})),
        "b3": b3_metrics(gold.get("clusters", {}), predicted.get("clusters", {})),
        "relations": evaluate_relations(gold.get("relations", []), predicted.get("relations", [])),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
