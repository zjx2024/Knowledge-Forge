"""Span-first entity normalization for isolated P2 treatment runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .entity_identity import normalize_name
from .value_parsing import coerce_confidence


_PURE_NUMBER = re.compile(r"^[\W_]*\d+(?:[.\-]\d+)*[\W_]*$")
_STRUCTURAL_ROLES = frozenset({
    "table_of_contents", "header", "footer", "page_number", "index_entry",
    "empty_cell", "reference_heading",
})


@dataclass(frozen=True)
class SpanMention:
    mention_id: str
    surface: str
    normalized_surface: str
    start: int
    end: int
    evidence: str
    confidence: float
    extraction_policy_version: str


@dataclass(frozen=True)
class TypeAssertion:
    mention_id: str
    entity_type: str
    confidence: float
    evidence: str
    policy_version: str
    source: str = "model"


@dataclass(frozen=True)
class AliasAssertion:
    mention_id: str
    surface: str
    normalized_surface: str
    start: int
    end: int
    confidence: float


@dataclass
class SpanExtractionResult:
    mentions: list[SpanMention] = field(default_factory=list)
    type_assertions: list[TypeAssertion] = field(default_factory=list)
    aliases: list[AliasAssertion] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)


class SpanFirstEntityExtractor:
    """Accept model output only when it points at an exact source span."""

    def __init__(self, policy_version: str):
        self.policy_version = policy_version

    @staticmethod
    def _exact_span(content: str, surface: str, start: Any, end: Any) -> tuple[int, int] | None:
        try:
            left, right = int(start), int(end)
        except (TypeError, ValueError):
            return None
        if left < 0 or right <= left or right > len(content) or content[left:right] != surface:
            return None
        return left, right

    @staticmethod
    def _structural(surface: str, metadata: dict[str, Any]) -> bool:
        role = str(metadata.get("structural_role") or metadata.get("chunk_role") or "").lower()
        return role in _STRUCTURAL_ROLES or bool(_PURE_NUMBER.fullmatch(surface.strip()))

    def normalize(
        self,
        *,
        project_scope: str,
        document_key: str,
        chunk_key: str,
        content: str,
        raw_entities: Iterable[dict[str, Any]],
        chunk_metadata: dict[str, Any],
    ) -> SpanExtractionResult:
        result = SpanExtractionResult()
        seen: set[str] = set()
        for raw in raw_entities:
            surface = str(raw.get("surface") or raw.get("name") or "")
            span = self._exact_span(
                content,
                surface,
                raw.get("mention_start", raw.get("evidence_start")),
                raw.get("mention_end", raw.get("evidence_end")),
            )
            if span is None:
                result.rejected.append({"raw": dict(raw), "reason": "invalid_exact_span"})
                continue
            if self._structural(surface, chunk_metadata):
                result.rejected.append({"raw": dict(raw), "reason": "structural_node"})
                continue
            left, right = span
            entity_type = str(raw.get("type") or "GenericEntity")
            material = "\x1f".join((
                project_scope, document_key, chunk_key, str(left), str(right),
                entity_type, surface,
            ))
            mention_id = "mention_" + hashlib.sha256(material.encode("utf-8")).hexdigest()
            if mention_id in seen:
                continue
            seen.add(mention_id)
            confidence = coerce_confidence(raw.get("confidence"), 0.5)
            mention = SpanMention(
                mention_id=mention_id,
                surface=surface,
                normalized_surface=normalize_name(surface),
                start=left,
                end=right,
                evidence=str(raw.get("evidence_text") or surface),
                confidence=confidence,
                extraction_policy_version=self.policy_version,
            )
            result.mentions.append(mention)
            result.type_assertions.append(TypeAssertion(
                mention_id=mention_id,
                entity_type=entity_type,
                confidence=confidence,
                evidence=surface,
                policy_version=self.policy_version,
            ))
            for alias in raw.get("alias_spans") or []:
                alias_surface = str(alias.get("surface") or "")
                alias_span = self._exact_span(
                    content, alias_surface, alias.get("start"), alias.get("end"),
                )
                if alias_span is None:
                    continue
                result.aliases.append(AliasAssertion(
                    mention_id=mention_id,
                    surface=alias_surface,
                    normalized_surface=normalize_name(alias_surface),
                    start=alias_span[0],
                    end=alias_span[1],
                    confidence=coerce_confidence(alias.get("confidence"), confidence),
                ))
        return result
