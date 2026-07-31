#!/usr/bin/env python3
"""Validate an immutable GraphRAG Native bundle without PostgreSQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_forge.graphrag_native_artifacts import GraphRAGNativeBundleError, GraphRAGNativeBundleValidator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(GraphRAGNativeBundleValidator().validate(Path(args.root)), ensure_ascii=False, sort_keys=True))
        return 0
    except GraphRAGNativeBundleError as exc:
        print(json.dumps({"error": str(exc), "failure_class": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
