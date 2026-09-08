#!/usr/bin/env python3
"""Operator-safe GraphRAG Native dependency doctor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge.graphrag_native_runner import GraphRAGNativeProviderDoctor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    if not args.config_only:
        parser.error("this safe doctor currently requires --config-only")
    print(json.dumps(GraphRAGNativeProviderDoctor.config_only(args.python), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
