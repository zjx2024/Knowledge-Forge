"""pgvector-backed embedding persistence and hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .config import Config
from .database import Database
from .entity_identity import entity_identity_text
from .ollama import OllamaClient


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(value * value for value in left)) * math.sqrt(sum(value * value for value in right))
    return dot / norm if norm else 0.0


def _terms(text: str) -> Counter[str]:
    normalized = re.sub(r"\s+", "", (text or "").lower())
    latin = re.findall(r"[a-z0-9_]{2,}", normalized)
    terms = Counter(latin)
    terms.update(
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
        if "\u4e00" <= normalized[index] <= "\u9fff"
    )
    return terms


def lexical_similarity(query: str, content: str) -> float:
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    content_terms = _terms(content)
    matched = sum(min(count, content_terms.get(term, 0)) for term, count in query_terms.items())
    return matched / sum(query_terms.values())


class VectorStore:
    def __init__(self, database: Database, ollama: OllamaClient | None = None):
        self.db = database
        self.ollama = ollama or OllamaClient()

    def index_chunks(self, project: dict[str, Any], document_id: int, chunks: list[dict[str, Any]]) -> int:
        model = project.get("embedding_model") or Config.default_embedding_model
        vectors = self.ollama.embed([item["content"] for item in chunks], model)
        records = []
        for chunk, vector in zip(chunks, vectors):
            records.append({
                "project_id": int(project["id"]),
                "document_id": document_id,
                "chunk_id": int(chunk["id"]),
                "object_type": "chunk",
                "object_key": str(chunk["chunk_key"]),
                "content": chunk["content"],
                "content_hash": chunk["content_hash"],
                "model_name": model,
                "embedding": vector,
                "metadata": {"section_id": chunk.get("section_id")},
            })
        self.db.replace_document_embeddings(document_id, records)
        return len(records)

    def upsert_entity_embedding(self, project_id: int, entity: dict[str, Any], model: str) -> None:
        content = entity_identity_text(
            entity["name"], entity["entity_type"], entity.get("description") or "",
            entity.get("aliases") or [],
        )
        vector = self.ollama.embed([content], model)[0]
        self.db.upsert_embedding({
            "project_id": project_id,
            "document_id": None,
            "chunk_id": None,
            "object_type": "entity",
            "object_key": str(entity["id"]),
            "content": content,
            "content_hash": entity.get("entity_key") or str(entity["id"]),
            "model_name": model,
            "embedding": vector,
            "metadata": {"entity_type": entity["entity_type"]},
        })

    def upsert_relation_embedding(
        self,
        project_id: int,
        document_id: int,
        chunk_id: int,
        relation: dict[str, Any],
        source_name: str,
        target_name: str,
        model: str,
    ) -> None:
        content = f"{source_name} {relation['relation_type']} {target_name}\n{relation.get('description') or ''}".strip()
        vector = self.ollama.embed([content], model)[0]
        self.db.upsert_embedding({"project_id": project_id, "document_id": document_id, "chunk_id": chunk_id,
            "object_type": "relation", "object_key": str(relation["id"]), "content": content,
            "content_hash": relation["relation_key"], "model_name": model, "embedding": vector,
            "metadata": {"relation_type": relation["relation_type"]}})

    def search(self, project: dict[str, Any], query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return self.search_type(project, query, "chunk", top_k)

    def search_type(self, project: dict[str, Any], query: str, object_type: str, top_k: int | None = None) -> list[dict[str, Any]]:
        model = project.get("embedding_model") or Config.default_embedding_model
        query_vector = self.ollama.embed([query], model)[0]
        result_limit = max(1, min(top_k or Config.retrieval_top_k, 50))
        rows = self.db.search_embeddings(
            int(project["id"]), object_type, model, query_vector, query, result_limit
        )
        dense_weight = max(0.0, min(1.0, Config.retrieval_dense_weight))
        results = []
        for row in rows:
            dense = max(0.0, float(row.get("dense_score") or 0.0))
            lexical = lexical_similarity(query, row["content"])
            item = dict(row)
            item.pop("embedding", None)
            item.update({"dense_score": dense, "lexical_score": lexical, "score": dense_weight * dense + (1 - dense_weight) * lexical})
            results.append(item)
        results.sort(key=lambda item: (item["score"], item["dense_score"]), reverse=True)
        return results[:result_limit]
