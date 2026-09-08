#!/usr/bin/env python3
"""Evaluate immutable snapshot item artifacts against a frozen Gold bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge.database import Database
from knowledge_forge.quality import canonical_digest
from knowledge_forge.release_gate import evaluate_frozen_bundle


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _database_bundle(snapshot_id: str) -> dict:
    status = Database().policy_snapshot_status(snapshot_id)
    items = []
    for artifact in status["artifacts"]:
        payload = artifact.get("payload") or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        if isinstance(payload.get("evaluation_item"), dict):
            items.append(payload["evaluation_item"])
    if not items:
        raise RuntimeError("snapshot contains no evaluation_item artifacts")
    snapshot = {
        **status["snapshot"],
        "expected_items": len(items), "completed_items": len(items), "failed_items": 0,
        "artifact_root_hash": canonical_digest(sorted(
            [[str(item["item_id"]), str(item["item_hash"])] for item in items]
        )),
    }
    return {"snapshot": snapshot, "items": items}


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot-id")
    source.add_argument("--snapshot-bundle", type=Path)
    parser.add_argument("--gold-manifest", required=True, type=Path)
    parser.add_argument("--gold-items", type=Path)
    parser.add_argument("--evaluator-version", default="quality-v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = _read(args.gold_manifest)
    gold_path = args.gold_items
    if gold_path is None:
        configured = manifest.get("annotations_path")
        gold_path = (
            args.gold_manifest.parent / str(configured)
            if configured else args.gold_manifest.with_name("annotations.json")
        )
    gold_payload = _read(gold_path)
    gold_items = gold_payload.get("items") if isinstance(gold_payload, dict) else gold_payload
    if not isinstance(gold_items, list):
        raise ValueError("Gold annotations must be an array or {'items': [...]} object")
    bundle = _database_bundle(args.snapshot_id) if args.snapshot_id else _read(args.snapshot_bundle)
    result = evaluate_frozen_bundle(
        snapshot_bundle=bundle, gold_manifest=manifest, gold_items=gold_items,
        evaluator_version=args.evaluator_version,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
