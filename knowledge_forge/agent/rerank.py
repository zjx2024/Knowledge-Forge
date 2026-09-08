"""Bounded client for the existing internal cross-encoder HTTP service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import requests

from ..config import Config
from .contracts import EvidenceItem


@dataclass(frozen=True)
class RerankDiagnostics:
    used_model: bool
    model: str
    candidate_count: int
    latency_ms: int
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentReranker:
    def __init__(self) -> None:
        self.base_url = Config.entity_reranker_url.rstrip("/")
        self.model = Config.entity_reranker_model
        self.api_key = Config.entity_reranker_api_key

    def rerank(
        self, query: str, items: list[EvidenceItem],
    ) -> RerankDiagnostics:
        if not items:
            return RerankDiagnostics(False, self.model, 0, 0, "no candidates")
        documents = [
            (item.title + "\n" + item.content)[:Config.reranker_max_text_chars]
            for item in items[:Config.reranker_max_documents]
        ]
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        started = time.monotonic()
        try:
            response = requests.post(
                f"{self.base_url}/api/rerank",
                headers=headers,
                json={
                    "model": self.model, "query": str(query),
                    "documents": documents, "top_n": len(documents),
                },
                timeout=(
                    Config.entity_reranker_connect_timeout,
                    Config.entity_reranker_timeout,
                ),
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") if isinstance(payload, dict) else None
            if not isinstance(results, list) or len(results) != len(documents):
                raise ValueError("reranker 返回数量与候选不一致")
            aligned: list[float | None] = [None] * len(documents)
            for value in results:
                index = value.get("index") if isinstance(value, dict) else None
                score = value.get("relevance_score") if isinstance(value, dict) else None
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(aligned):
                    raise ValueError("reranker 返回非法候选编号")
                numeric = float(score)
                if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                    raise ValueError("reranker 返回非法相关度")
                if aligned[index] is not None:
                    raise ValueError("reranker 重复返回候选编号")
                aligned[index] = numeric
            if any(value is None for value in aligned):
                raise ValueError("reranker 遗漏候选")
            for item, score in zip(items, aligned, strict=False):
                item.rerank_score = float(score)
            return RerankDiagnostics(
                True, str(payload.get("model") or self.model), len(documents),
                int(payload.get("latency_ms") or (time.monotonic() - started) * 1000),
            )
        except (requests.RequestException, ValueError, TypeError) as exc:
            # RRF is a deterministic, inspectable degradation path.  A model
            # outage must not turn into arbitrary order or ungrounded output.
            for item in items:
                item.rerank_score = item.fused_score
            return RerankDiagnostics(
                False, self.model, len(documents),
                int((time.monotonic() - started) * 1000), str(exc)[:500],
            )
