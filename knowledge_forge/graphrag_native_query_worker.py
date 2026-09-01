"""One-request subprocess JSON protocol for GraphRAG Native queries."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from .graphrag_native_query import (
    QUERY_PROTOCOL_VERSION,
    GraphRAGNativeQueryError,
    NativeQueryRequest,
    query_bundle,
)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict) or payload.get("protocol_version") != QUERY_PROTOCOL_VERSION:
            raise GraphRAGNativeQueryError("native query worker protocol version 不匹配")
        request = NativeQueryRequest(
            run_id=str(payload.get("run_id") or ""), mode=str(payload.get("mode") or ""),
            question=str(payload.get("question") or ""),
            response_type=str(payload.get("response_type") or "Multiple Paragraphs"),
            community_level=int(payload.get("community_level") or 2),
            model_context_tokens=max(
                0, int(payload.get("model_context_tokens") or 0)
            ),
        )
        print(json.dumps(query_bundle(request, Path(str(payload.get("bundle_root") or ""))), ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"protocol_version": QUERY_PROTOCOL_VERSION, "error": str(exc), "failure_class": type(exc).__name__}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
