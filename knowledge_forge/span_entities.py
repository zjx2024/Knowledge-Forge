"""Span-first entity normalization for isolated P2 treatment runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .entity_identity import normalize_name
from .value_parsing import coerce_confidence


_PURE_NUMBER = re.compile(r"^[\W_]*\d+(?:[.\-]\d+)*[\W_]*$")
_YEAR_OR_ESTIMATE = re.compile(r"^(?:18|19|20)\d{2}e?$", re.IGNORECASE)
_PERCENTAGE = re.compile(r"^[+-]?\d+(?:\.\d+)?\s*%$")
_PHONE = re.compile(r"^\+?\d[\d\s().-]{6,}\d$")
_AMOUNT_OR_UNIT = re.compile(
    r"^[+-]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|tons?|tonnes?|metric\s+tons?|mt|kg|g|lbs?|usd|us\$|\\$|"
    r"million|billion|thousand|万吨|吨|千克|美元|亿元?)?$",
    re.IGNORECASE,
)
_NOISE_ENTITY_TYPES = frozenset({
    "Date", "Year", "Time", "Number", "Quantity", "Amount", "Money",
    "Currency", "Percentage", "Percent", "Phone", "PhoneNumber", "Unit",
})
_NON_REFERENTIAL_MODIFIERS = frozenset({
    "domestic", "foreign", "estimated", "estimate", "free", "total", "subtotal",
})
_STRUCTURAL_ROLES = frozenset({
    "table_of_contents", "header", "footer", "page_number", "index_entry",
    "empty_cell", "reference_heading", "table_header", "unit_header",
    "heading", "title", "page_header", "page_footer", "table_container",
    "caption", "footnote",
})


@dataclass(frozen=True)
class SpanMention:
    mention_id: str
    occurrence_key: str
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
    kind: str = "explicit_alias"


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
    def _source_anchor(
        *,
        document_key: str,
        chunk_key: str,
        surface: str,
        left: int,
        right: int,
        chunk_metadata: dict[str, Any],
        raw: dict[str, Any],
    ) -> str:
        anchor = str(
            raw.get("source_anchor")
            or raw.get("block_key")
            or chunk_metadata.get("source_anchor")
            or chunk_metadata.get("block_key")
            or f"chunk:{chunk_key}"
        )
        try:
            base = int(
                raw.get("source_base")
                if raw.get("source_base") is not None
                else chunk_metadata.get("source_start")
                if chunk_metadata.get("source_start") is not None
                else chunk_metadata.get("chunk_source_start")
                if chunk_metadata.get("chunk_source_start") is not None
                else 0
            )
        except (TypeError, ValueError):
            base = 0
        source_left = int(raw.get("source_start") if raw.get("source_start") is not None else base + left)
        source_right = int(raw.get("source_end") if raw.get("source_end") is not None else base + right)
        material = "\x1f".join((
            document_key, anchor, str(source_left), str(source_right), surface,
        ))
        return "occurrence_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _reject_reason(
        surface: str,
        metadata: dict[str, Any],
        raw: dict[str, Any],
        *,
        start: int,
        end: int,
    ) -> str:
        stripped = surface.strip()
        normalized = normalize_name(stripped)
        # Parser-owned source structure wins over a model-supplied label.  A
        # model must not be able to reclassify a page header/table header as
        # body text simply by emitting a different structural_role.
        role = str(
            metadata.get("structural_role")
            or metadata.get("source_role")
            or metadata.get("chunk_role")
            or metadata.get("cell_role")
            or raw.get("structural_role")
            or ""
        ).lower().replace("-", "_").replace(" ", "_")
        block_types = {
            str(value).strip().lower().replace("-", "_")
            for value in metadata.get("block_types") or []
            if str(value).strip()
        }
        if not role and len(block_types) == 1:
            role = next(iter(block_types))
        entity_type = str(raw.get("type") or "GenericEntity")
        for structural_span in metadata.get("structural_spans") or []:
            if not isinstance(structural_span, dict):
                continue
            try:
                span_start = int(structural_span["start"])
                span_end = int(structural_span["end"])
            except (KeyError, TypeError, ValueError):
                continue
            span_role = str(
                structural_span.get("source_role") or ""
            ).lower().replace("-", "_")
            if span_start <= start < end <= span_end and span_role in _STRUCTURAL_ROLES:
                return "structural_node"
        if role in _STRUCTURAL_ROLES:
            return "structural_node"
        if entity_type in _NOISE_ENTITY_TYPES:
            return "scalar_or_temporal_noise"
        if not normalized:
            return "empty_normalized_surface"
        if normalized in _NON_REFERENTIAL_MODIFIERS:
            return "non_referential_modifier"
        if (
            _PURE_NUMBER.fullmatch(stripped)
            or _YEAR_OR_ESTIMATE.fullmatch(stripped)
            or _PERCENTAGE.fullmatch(stripped)
            or _PHONE.fullmatch(stripped)
            or _AMOUNT_OR_UNIT.fullmatch(stripped)
        ):
            return "scalar_or_temporal_noise"
        return ""

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
            reject_reason = self._reject_reason(
                surface,
                chunk_metadata,
                raw,
                start=span[0],
                end=span[1],
            )
            if reject_reason:
                result.rejected.append({"raw": dict(raw), "reason": reject_reason})
                continue
            left, right = span
            entity_type = str(raw.get("type") or "GenericEntity")
            material = "\x1f".join((
                project_scope, document_key, chunk_key, str(left), str(right),
                surface,
            ))
            mention_id = "mention_" + hashlib.sha256(material.encode("utf-8")).hexdigest()
            if mention_id in seen:
                continue
            seen.add(mention_id)
            confidence = coerce_confidence(raw.get("confidence"), 0.5)
            occurrence_key = self._source_anchor(
                document_key=document_key,
                chunk_key=chunk_key,
                surface=surface,
                left=left,
                right=right,
                chunk_metadata=chunk_metadata,
                raw=raw,
            )
            mention = SpanMention(
                mention_id=mention_id,
                occurrence_key=occurrence_key,
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
                    kind=str(alias.get("kind") or "explicit_alias"),
                ))
        return result
