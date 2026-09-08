"""Stable data contracts shared by both first-party retrieval strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable

from ..unicode_safety import sanitize_unicode_data, sanitize_unicode_text


EVIDENCE_TYPES = frozenset({
    "chunk", "entity", "relation", "community_report",
    "native_context", "derived_answer",
})
STRATEGIES = frozenset({"knowledge_forge", "graphrag_native"})


def _clean_list(values: Iterable[object] | None) -> list[str]:
    return list(dict.fromkeys(
        sanitize_unicode_text(str(value)).strip()
        for value in (values or []) if str(value or "").strip()
    ))


def canonical_json(value: Any) -> str:
    return json.dumps(
        sanitize_unicode_data(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class SnapshotRef:
    """An immutable query-time view of one project's published knowledge."""

    strategy: str
    identifier: str
    source_hash: str
    generation: int = 0
    graph_available: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError("unknown knowledge strategy")
        if not self.identifier.strip() or not self.source_hash.strip():
            raise ValueError("snapshot identifier and source hash are required")
        if self.generation < 0:
            raise ValueError("snapshot generation cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return sanitize_unicode_data(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SnapshotRef":
        return cls(
            strategy=str(value.get("strategy") or ""),
            identifier=str(value.get("identifier") or ""),
            source_hash=str(value.get("source_hash") or ""),
            generation=max(0, int(value.get("generation") or 0)),
            graph_available=bool(value.get("graph_available")),
            details=dict(value.get("details") or {}),
        )


@dataclass(frozen=True)
class SourceRef:
    """A browser-safe pointer back to an exact source location."""

    document_id: int
    filename: str
    document_url: str
    revision_id: str = ""
    section_title: str = ""
    pages: tuple[int, ...] = ()
    page_label: str = ""
    quote: str = ""
    media_assets: tuple[dict[str, Any], ...] = ()
    table: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return (
            self.document_id > 0
            and bool(self.filename.strip())
            and bool(self.revision_id.strip())
        )

    def to_dict(self) -> dict[str, Any]:
        return sanitize_unicode_data(asdict(self))

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, quote: str = "") -> "SourceRef":
        document_id = int(value.get("document_id") or 0)
        pages: list[int] = []
        for raw in value.get("pages") or ([value.get("page")] if value.get("page") else []):
            try:
                page = int(raw)
            except (TypeError, ValueError):
                continue
            if page > 0:
                pages.append(page)
        document_url = str(value.get("document_url") or value.get("download_path") or "")
        if document_id > 0:
            document_url = f"/api/documents/{document_id}/download"
        return cls(
            document_id=document_id,
            filename=sanitize_unicode_text(str(value.get("filename") or value.get("original_filename") or "")),
            document_url=document_url,
            revision_id=str(value.get("revision_id") or ""),
            section_title=sanitize_unicode_text(str(value.get("section_title") or value.get("location") or "")),
            pages=tuple(sorted(set(pages))),
            page_label=sanitize_unicode_text(str(value.get("page_label") or "")),
            quote=sanitize_unicode_text(str(quote or value.get("quote") or ""))[:4000],
            media_assets=tuple(
                sanitize_unicode_data(item) for item in (value.get("media_assets") or [])
                if isinstance(item, dict)
            )[:12],
            table=(sanitize_unicode_data(value.get("table")) if isinstance(value.get("table"), dict) else None),
            metadata=sanitize_unicode_data(dict(value.get("metadata") or {})),
        )


@dataclass
class EvidenceItem:
    """One normalized, provenance-bearing retrieval candidate."""

    evidence_id: str
    evidence_type: str
    project_id: int
    strategy: str
    snapshot_ref: SnapshotRef
    title: str
    content: str
    entity_keys: list[str] = field(default_factory=list)
    relation_keys: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)
    retrieval_channels: list[str] = field(default_factory=list)
    raw_scores: dict[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    rerank_score: float = 0.0
    provenance_score: float = 0.0
    provenance_valid: bool = False
    rank: int = 0
    selected: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError("unknown evidence type")
        if self.strategy not in STRATEGIES or self.project_id <= 0:
            raise ValueError("invalid evidence project or strategy")
        if self.snapshot_ref.strategy != self.strategy:
            raise ValueError("evidence and snapshot strategies differ")
        self.title = sanitize_unicode_text(str(self.title or "")).strip()
        self.content = sanitize_unicode_text(str(self.content or "")).strip()
        if not self.content:
            raise ValueError("evidence content is required")
        self.entity_keys = _clean_list(self.entity_keys)
        self.relation_keys = _clean_list(self.relation_keys)
        self.retrieval_channels = _clean_list(self.retrieval_channels)
        self.raw_scores = {
            str(key): float(value) for key, value in self.raw_scores.items()
            if str(key) and isinstance(value, (int, float))
        }
        self.provenance_valid = bool(
            self.source_refs and all(ref.valid for ref in self.source_refs)
        )
        self.provenance_score = max(
            0.0, min(1.0, float(self.provenance_score or (1.0 if self.provenance_valid else 0.0))),
        )

    @staticmethod
    def identity(
        *, run_id: str, project_id: int, snapshot_ref: SnapshotRef,
        evidence_type: str, record_key: str, content: str,
    ) -> tuple[str, str]:
        stable_material = canonical_json({
            "project_id": project_id,
            "snapshot": snapshot_ref.to_dict(),
            "type": evidence_type,
            "record": record_key,
            "content_hash": hashlib.sha256(
                sanitize_unicode_text(content).encode("utf-8")
            ).hexdigest(),
        })
        stable_key = hashlib.sha256(stable_material.encode("utf-8")).hexdigest()
        evidence_id = "ev_" + hashlib.sha256(
            f"{run_id}:{stable_key}".encode("utf-8")
        ).hexdigest()[:32]
        return evidence_id, stable_key

    @classmethod
    def create(
        cls, *, run_id: str, evidence_type: str, project_id: int,
        strategy: str, snapshot_ref: SnapshotRef, record_key: str,
        title: str, content: str, **values: Any,
    ) -> "EvidenceItem":
        evidence_id, stable_key = cls.identity(
            run_id=run_id, project_id=project_id,
            snapshot_ref=snapshot_ref, evidence_type=evidence_type,
            record_key=str(record_key), content=content,
        )
        metadata = dict(values.pop("metadata", {}) or {})
        metadata.update({"stable_key": stable_key, "record_key": str(record_key)})
        return cls(
            evidence_id=evidence_id, evidence_type=evidence_type,
            project_id=project_id, strategy=strategy,
            snapshot_ref=snapshot_ref, title=title, content=content,
            metadata=metadata, **values,
        )

    def to_record(self, *, run_id: str) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "run_id": run_id,
            "project_id": self.project_id,
            "evidence_type": self.evidence_type,
            "strategy_key": self.strategy,
            "snapshot_ref": self.snapshot_ref.to_dict(),
            "title": self.title,
            "content": self.content,
            "entity_keys": self.entity_keys,
            "relation_keys": self.relation_keys,
            "source_refs": [value.to_dict() for value in self.source_refs],
            "retrieval_channels": self.retrieval_channels,
            "raw_scores": self.raw_scores,
            "fused_score": self.fused_score,
            "rerank_score": self.rerank_score,
            "provenance_score": self.provenance_score,
            "provenance_valid": self.provenance_valid,
            "rank": self.rank,
            "selected": self.selected,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    strategy: str
    mode: str
    snapshot_ref: SnapshotRef
    evidence: list[EvidenceItem]
    candidate_answer: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)
