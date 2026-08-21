"""Rebuild active pgvector rows after changing the Embedding runtime.

The operation is resumable: each bounded database batch is atomic and rows
already stamped with the configured runtime profile are skipped on the next
invocation.  Source documents, parser artifacts, chunks and graph facts are
never modified.
"""

from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_forge.config import Config  # noqa: E402
from knowledge_forge.database import Database  # noqa: E402
from knowledge_forge.ollama import OllamaClient  # noqa: E402


def _batches(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild active vectors with the configured Embedding runtime",
    )
    parser.add_argument("--project-id", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=max(1, int(Config.entity_embedding_batch_size)),
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")

    database = Database()
    profile = Config.ollama_embedding_runtime_profile
    rows = database.list_active_embeddings_for_runtime_reindex(
        profile,
        project_id=arguments.project_id,
    )
    summary = {
        "runtime_profile": profile,
        "embedding_endpoint": Config.ollama_embedding_base_url,
        "project_id": arguments.project_id,
        "pending_vectors": len(rows),
        "models": sorted({str(item["model_name"]) for item in rows}),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if arguments.dry_run or not rows:
        return

    client = OllamaClient()
    completed = 0
    for model_name, grouped in groupby(rows, key=lambda item: str(item["model_name"])):
        model_rows = list(grouped)
        for batch in _batches(model_rows, arguments.batch_size):
            vectors = client.embed(
                [str(item.get("content") or "") for item in batch],
                model_name,
            )
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch for {model_name}: "
                    f"expected {len(batch)}, got {len(vectors)}"
                )
            records = []
            for row, vector in zip(batch, vectors, strict=True):
                old_dimension = int(row.get("dimension") or 0)
                if old_dimension and old_dimension != len(vector):
                    raise RuntimeError(
                        f"Embedding dimension changed for {model_name}: "
                        f"expected {old_dimension}, got {len(vector)}"
                    )
                records.append({
                    "id": int(row["id"]),
                    "model_name": model_name,
                    "embedding": vector,
                })
            updated = database.replace_embedding_runtime_batch(records, profile)
            if updated != len(records):
                raise RuntimeError(
                    f"Embedding persistence mismatch: expected {len(records)}, got {updated}"
                )
            completed += updated
            print(
                json.dumps({
                    "model": model_name,
                    "completed": completed,
                    "total": len(rows),
                }, ensure_ascii=False),
                flush=True,
            )


if __name__ == "__main__":
    main()
