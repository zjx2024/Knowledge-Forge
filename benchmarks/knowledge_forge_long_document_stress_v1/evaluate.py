#!/usr/bin/env python3
"""Evaluate deterministic 1x/10x document-processing scalability."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_forge.chunking import process_document
from knowledge_forge.config import Config


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_documents(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for relative in manifest["source_corpora"]:
        corpus = (ROOT / str(relative)).resolve()
        corpus_manifest = _read(corpus / "manifest.json")
        for document in corpus_manifest["documents"]:
            rows.append((
                str(document["id"]),
                (corpus / str(document["path"])).read_text(encoding="utf-8"),
            ))
    if len(rows) < 10:
        raise AssertionError("long-document stress requires ten source documents")
    return rows


def _render(rows: list[tuple[str, str]], repetitions: int) -> str:
    parts = ["# Knowledge Forge deterministic scale fixture"]
    for repetition in range(1, repetitions + 1):
        for document_id, content in rows:
            parts.extend((
                f"# Copy {repetition}: {document_id}",
                content.strip(),
            ))
    return "\n\n".join(parts) + "\n"


def _measure(content: str, path: Path) -> dict[str, Any]:
    path.write_text(content, encoding="utf-8")
    started = time.perf_counter()
    result = process_document(
        path,
        requested_parser="native_text",
        requested_strategy="general",
        source_name=path.name,
    )
    elapsed = time.perf_counter() - started
    graph_chunks = [
        chunk for chunk in result.chunks
        if chunk.metadata.get("graph_extraction_enabled", True) is not False
    ]
    entity_passes = len(graph_chunks) * (
        1 + int(Config.entity_grounded_gleaning_rounds)
    )
    relation_source_passes = len(graph_chunks) * (
        2
        + int(Config.relation_open_fact_gleaning_rounds)
        + int(Config.relation_source_binding_gleaning_rounds)
    )
    return {
        "bytes": len(content.encode("utf-8")),
        "blocks": len(result.parsed.blocks),
        "sections": len(result.sections),
        "retrieval_chunks": len(result.chunks),
        "graph_chunks": len(graph_chunks),
        "graph_tokens": sum(chunk.token_count for chunk in graph_chunks),
        "scheduled_entity_passes": entity_passes,
        "scheduled_relation_source_passes": relation_source_passes,
        "scheduled_base_model_passes": entity_passes + relation_source_passes,
        "processing_seconds": round(elapsed, 6),
    }


def evaluate() -> dict[str, Any]:
    manifest = _read(ROOT / "manifest.json")
    rows = _source_documents(manifest)
    measurements: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="kf-long-stress-") as directory:
        temp_root = Path(directory)
        for scale in manifest["scales"]:
            repetitions = int(scale)
            measurements[f"{repetitions}x"] = _measure(
                _render(rows, repetitions),
                temp_root / f"stress-{repetitions}x.md",
            )

    small = measurements["1x"]
    large = measurements["10x"]
    byte_growth = large["bytes"] / small["bytes"]
    graph_growth = large["graph_chunks"] / max(1, small["graph_chunks"])
    call_growth = (
        large["scheduled_base_model_passes"]
        / max(1, small["scheduled_base_model_passes"])
    )
    acceptance = manifest["acceptance"]
    graph_relative_growth = graph_growth / byte_growth
    call_relative_growth = call_growth / byte_growth
    checks = {
        "large_document_size": (
            large["bytes"] >= int(acceptance["minimum_large_document_bytes"])
        ),
        "graph_chunk_scaling": (
            graph_relative_growth
            <= float(acceptance[
                "maximum_graph_chunk_growth_vs_source_growth"
            ])
        ),
        "scheduled_base_call_scaling": (
            call_relative_growth
            <= float(acceptance[
                "maximum_scheduled_base_call_growth_vs_source_growth"
            ])
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "quality_gold": False,
        "source_documents": len(rows),
        "measurements": measurements,
        "growth": {
            "bytes": round(byte_growth, 4),
            "graph_chunks": round(graph_growth, 4),
            "scheduled_base_model_passes": round(call_growth, 4),
            "graph_chunks_relative_to_bytes": round(graph_relative_growth, 4),
            "scheduled_calls_relative_to_bytes": round(call_relative_growth, 4),
        },
        "checks": checks,
        "scope_note": (
            "Parser/chunker and deterministic base-pass schedule only; real model "
            "latency, retries and verifier calls require HTTP/Ollama evaluation."
        ),
    }


def main() -> int:
    report = evaluate()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
