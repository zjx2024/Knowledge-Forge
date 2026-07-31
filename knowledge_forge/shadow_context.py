"""Enhanced-only structured context units for Shadow v2 relation work."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .quality import canonical_json
from .sentence_spans import split_sentences


SHADOW_CONTEXT_POLICY_VERSION = "shadow-context-source-map-v2"
SHADOW_CONTEXT_SOURCE_MAP_POLICY = "block-or-legacy-chunk-local-v2"
KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION = "knowledge-forge-structure-gate-v1"

# These are parser roles, rather than business concepts.  They deliberately
# decide only whether a source is suitable for factual extraction; relation
# semantics, direction and publication still come exclusively from Schema.
_SOURCE_ROLE_ALIASES = {
    "paragraph": "body",
    "body": "body",
    "text": "body",
    "sentence": "body",
    "clause": "body",
    "apposition": "body",
    "list": "list_item",
    "list_item": "list_item",
    "list-item": "list_item",
    "table_row": "table_row",
    "table-row": "table_row",
    "row": "table_row",
    "table_cell": "table_cell",
    "table-cell": "table_cell",
    "cell": "table_cell",
    "heading": "heading",
    "title": "heading",
    "header": "page_header",
    "page_header": "page_header",
    "page-header": "page_header",
    "footer": "page_footer",
    "page_footer": "page_footer",
    "page-footer": "page_footer",
    "page_number": "page_number",
    "page-number": "page_number",
    "table": "table_container",
    "table_header": "table_header",
    "table-header": "table_header",
    "caption": "caption",
    "footnote": "footnote",
    "reference_heading": "reference_heading",
    "table_of_contents": "table_of_contents",
    "index_entry": "index_entry",
}
_RELATION_ELIGIBLE_SOURCE_ROLES = frozenset({
    "body", "list_item", "table_row", "table_cell",
})
_LIST_ITEM_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)]|[A-Za-z][.)])\s+\S")


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def source_role_for_structure(
    structure: Mapping[str, Any] | None = None,
    *,
    context_kind: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Return a conservative parser-role classification without semantic cues."""

    structure = structure if isinstance(structure, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw = _first_text(
        structure.get("source_role"),
        structure.get("structural_role"),
        metadata.get("structural_role"),
        metadata.get("source_role"),
        metadata.get("cell_role"),
        context_kind,
        structure.get("block_type"),
        metadata.get("context_kind"),
    ).lower().replace(" ", "_")
    return _SOURCE_ROLE_ALIASES.get(raw, "unknown")


def relation_eligible_source_role(role: str) -> bool:
    return str(role or "") in _RELATION_ELIGIBLE_SOURCE_ROLES


def structural_profile(
    *,
    content: str,
    structure: Mapping[str, Any] | None = None,
    context_kind: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe source eligibility and deterministic atomic evidence scopes.

    The profile has no knowledge of predicates or domains.  A body unit is
    divided into sentences; a list item, table row and table cell stay atomic.
    Older chunk-local source has sentence scopes too, so its lower structural
    fidelity never silently opens a cross-sentence relation path.
    """

    role = source_role_for_structure(
        structure, context_kind=context_kind, metadata=metadata,
    )
    normalized_kind = str(context_kind or "").strip().lower().replace("-", "_")
    atomic_kind = (
        role if role in {"list_item", "table_row", "table_cell"}
        else "sentence"
    )
    scopes: list[dict[str, Any]] = []
    if atomic_kind == "sentence":
        for index, (start, end, _text) in enumerate(split_sentences(content)):
            scopes.append({
                "scope_id": f"sentence:{index}:{start}:{end}",
                "scope_kind": "sentence",
                "start": int(start),
                "end": int(end),
            })
    if not scopes and content:
        scopes.append({
            "scope_id": f"{atomic_kind}:0:{len(content)}",
            "scope_kind": atomic_kind,
            "start": 0,
            "end": len(content),
        })
    return {
        "policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
        "source_role": role,
        "relation_eligible": relation_eligible_source_role(role),
        "context_kind": normalized_kind or "legacy_chunk",
        "atomic_scope_kind": atomic_kind,
        "scopes": scopes,
    }


def occurrence_scope(
    *,
    content: str,
    mention_start: int,
    mention_end: int,
    structure: Mapping[str, Any] | None = None,
    context_kind: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the one exact atomic scope containing an occurrence.

    An occurrence that crosses parser/sentence scope is marked ineligible
    rather than guessed into a neighboring scope.
    """

    profile = structural_profile(
        content=content,
        structure=structure,
        context_kind=context_kind,
        metadata=metadata,
    )
    start, end = int(mention_start), int(mention_end)
    structure = structure if isinstance(structure, Mapping) else {}
    structural_spans = structure.get("structural_spans") or []
    for raw_scope in structural_spans:
        if not isinstance(raw_scope, Mapping):
            continue
        try:
            scope_start = int(raw_scope["start"])
            scope_end = int(raw_scope["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if scope_start <= start < end <= scope_end:
            role = str(raw_scope.get("source_role") or "unknown")
            return {
                **profile,
                "source_role": role,
                "relation_eligible": relation_eligible_source_role(role),
                "occurrence_scope": {
                    "scope_id": str(
                        raw_scope.get("scope_id")
                        or f"mapped:{scope_start}:{scope_end}"
                    ),
                    "scope_kind": str(raw_scope.get("scope_kind") or role),
                    "start": scope_start,
                    "end": scope_end,
                },
            }
    if any(
        isinstance(raw_scope, Mapping)
        and int(raw_scope.get("start") or -1) < end
        and int(raw_scope.get("end") or -1) > start
        for raw_scope in structural_spans
    ):
        return {
            **profile,
            "relation_eligible": False,
            "occurrence_scope": {
                "scope_id": "invalid:crosses-structural-span",
                "scope_kind": "invalid",
                "start": start,
                "end": end,
            },
        }
    for scope in profile["scopes"]:
        if int(scope["start"]) <= start < end <= int(scope["end"]):
            return {**profile, "occurrence_scope": dict(scope)}
    return {
        **profile,
        "relation_eligible": False,
        "occurrence_scope": {
            "scope_id": "invalid:span",
            "scope_kind": "invalid",
            "start": start,
            "end": end,
        },
    }


def _line_spans(text: str, offset: int) -> Iterable[tuple[int, int, str]]:
    cursor = 0
    for line in text.splitlines(keepends=True):
        line_end = cursor + len(line)
        body_length = len(line.rstrip("\r\n"))
        if body_length:
            stripped_end = cursor + body_length
            yield offset + cursor, offset + stripped_end, line[:body_length]
        cursor = line_end
    if not text:
        return


def _table_or_list_scopes(
    *,
    block: Mapping[str, Any],
    block_start: int,
    block_end: int,
    block_key: str,
) -> list[dict[str, Any]]:
    """Turn layout rows/items into auditable source spans without domain rules."""

    text = str(block.get("text") or "")
    role = source_role_for_structure(
        {"block_type": block.get("block_type")},
        metadata=_metadata(block.get("metadata")),
    )
    spans: list[dict[str, Any]] = []
    lines = list(_line_spans(text, block_start))
    if role == "table_container":
        for index, (start, end, _line) in enumerate(lines):
            row_role = "table_header" if index < 2 else "table_row"
            spans.append({
                "start": start,
                "end": end,
                "source_role": row_role,
                "scope_id": f"{block_key}:table-row:{index}",
                "scope_kind": row_role,
            })
        return spans
    list_lines = [line for line in lines if _LIST_ITEM_LINE.match(line[2])]
    if list_lines:
        for index, (start, end, _line) in enumerate(list_lines):
            spans.append({
                "start": start,
                "end": end,
                "source_role": "list_item",
                "scope_id": f"{block_key}:list-item:{index}",
                "scope_kind": "list_item",
            })
        return spans
    spans.append({
        "start": block_start,
        "end": block_end,
        "source_role": role,
        "scope_id": f"{block_key}:{role}:0",
        "scope_kind": role,
    })
    return spans


def structural_spans_for_chunk(
    *,
    content: str,
    blocks: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map parser blocks to a persisted retrieval chunk only when exact.

    The map intentionally does not use fuzzy text matching.  Unmapped content
    stays body/sentence-scoped; an ambiguous parser block cannot be guessed
    into another region and thereby alter factual extraction.
    """

    spans: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for index, block in enumerate(blocks, start=1):
        text = str(block.get("text") or "")
        if not text:
            continue
        start_at = 0
        match_start = -1
        while True:
            candidate = content.find(text, start_at)
            if candidate < 0:
                break
            candidate_end = candidate + len(text)
            if not any(
                candidate < end and candidate_end > start
                for start, end in occupied
            ):
                match_start = candidate
                occupied.append((candidate, candidate_end))
                break
            start_at = candidate + 1
        if match_start < 0:
            continue
        block_key = _block_key(block, index)
        spans.extend(_table_or_list_scopes(
            block=block,
            block_start=match_start,
            block_end=match_start + len(text),
            block_key=block_key,
        ))
    return sorted(
        spans,
        key=lambda item: (int(item["start"]), int(item["end"]), str(item["scope_id"])),
    )


@dataclass(frozen=True)
class SourceMapEntry:
    rendered_start: int
    rendered_end: int
    anchor_kind: str
    block_key: str
    block_local_start: int
    block_local_end: int
    table_id: str = ""
    row_id: str = ""
    cell_id: str = ""
    page: int | None = None


@dataclass(frozen=True)
class ContextUnit:
    context_id: str
    namespace_id: str
    document_revision_id: str
    context_kind: str
    rendered_text: str
    rendered_hash: str
    source_block_keys: tuple[str, ...]
    source_map: tuple[SourceMapEntry, ...]
    structure: dict[str, Any] = field(default_factory=dict)
    structure_quality: str = "structured"


@dataclass(frozen=True)
class ShadowContextManifest:
    policy_version: str
    source_map_policy: str
    artifact_hash: str
    context_units: tuple[ContextUnit, ...]
    summary: dict[str, Any]

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "source_map_policy": self.source_map_policy,
            "artifact_hash": self.artifact_hash,
            "summary": self.summary,
        }


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _block_key(block: Mapping[str, Any], order: int) -> str:
    return str(block.get("key") or block.get("block_key") or f"block-{order:04d}")


def _metadata(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _context_kind(block: Mapping[str, Any]) -> str:
    metadata = _metadata(block.get("metadata"))
    kind = str(
        metadata.get("context_kind")
        or metadata.get("role")
        or block.get("block_type")
        or "paragraph"
    ).strip().lower()
    if kind in {"table-row", "table row", "row"}:
        return "table_row"
    if kind in {"list-item", "list item"}:
        return "list_item"
    return kind or "paragraph"


def _source_entry(
    *,
    rendered_text: str,
    block: Mapping[str, Any],
    block_key: str,
) -> SourceMapEntry:
    metadata = _metadata(block.get("metadata"))
    return SourceMapEntry(
        rendered_start=0,
        rendered_end=len(rendered_text),
        anchor_kind=str(metadata.get("anchor_kind") or _context_kind(block)),
        block_key=block_key,
        block_local_start=int(metadata.get("block_local_start") or 0),
        block_local_end=int(
            metadata.get("block_local_end")
            if metadata.get("block_local_end") is not None
            else len(str(block.get("text") or ""))
        ),
        table_id=str(metadata.get("table_id") or ""),
        row_id=str(metadata.get("row_id") or ""),
        cell_id=str(metadata.get("cell_id") or ""),
        page=(
            int(block["page"])
            if block.get("page") is not None and str(block.get("page")).isdigit()
            else None
        ),
    )


def _block_structure(block: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata(block.get("metadata"))
    base = {
        "block_key": _block_key(block, int(block.get("order") or 0)),
        "block_type": str(block.get("block_type") or ""),
        "section_key": str(block.get("section_key") or ""),
        "section_title": str(block.get("section_title") or ""),
        "heading_path": list(block.get("heading_path") or []),
        "page": block.get("page"),
        "table_id": str(metadata.get("table_id") or ""),
        "row_id": str(metadata.get("row_id") or ""),
        "cell_id": str(metadata.get("cell_id") or ""),
        "header_path": list(metadata.get("header_path") or metadata.get("headers") or []),
        "cell_role": str(metadata.get("cell_role") or ""),
        "unit": str(metadata.get("unit") or ""),
        "estimated": bool(metadata.get("estimated") or False),
        "parser_version": str(payload.get("parser_version") or ""),
        "chunk_id": int(
            metadata.get("chunk_id")
            or metadata.get("source_chunk_id")
            or block.get("chunk_id")
            or 0
        ),
    }
    role = source_role_for_structure(base, metadata=metadata)
    return {
        **base,
        "source_role": role,
        "relation_eligible": relation_eligible_source_role(role),
        "structure_policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
    }


def _unit(
    *,
    namespace_id: str,
    revision_id: str,
    context_kind: str,
    rendered_text: str,
    source_entry: SourceMapEntry,
    source_block_keys: tuple[str, ...],
    structure: dict[str, Any],
    structure_quality: str,
) -> ContextUnit:
    rendered_hash = _digest(rendered_text)
    context_id = "shadowctx_" + _digest({
        "namespace_id": namespace_id,
        "revision_id": revision_id,
        "context_kind": context_kind,
        "rendered_hash": rendered_hash,
        "source_block_keys": source_block_keys,
        "source_map": [asdict(source_entry)],
        "source_role": str(structure.get("source_role") or ""),
        "relation_eligible": bool(structure.get("relation_eligible")),
        "structure_policy_version": str(
            structure.get("structure_policy_version") or ""
        ),
        "policy": SHADOW_CONTEXT_POLICY_VERSION,
    })
    return ContextUnit(
        context_id=context_id,
        namespace_id=namespace_id,
        document_revision_id=revision_id,
        context_kind=context_kind,
        rendered_text=rendered_text,
        rendered_hash=rendered_hash,
        source_block_keys=source_block_keys,
        source_map=(source_entry,),
        structure=structure,
        structure_quality=structure_quality,
    )


def _chunk_mapped_units(
    *,
    namespace_id: str,
    revision_id: str,
    chunks: Iterable[Mapping[str, Any]],
    blocks: Iterable[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> tuple[ContextUnit, ...]:
    block_values = [item for item in blocks if isinstance(item, Mapping)]
    units: list[ContextUnit] = []
    for chunk in chunks:
        rendered_text = str(chunk.get("content") or "")
        if not rendered_text.strip():
            continue
        chunk_id = int(chunk.get("id") or 0)
        chunk_key = str(chunk.get("chunk_key") or chunk_id or "chunk")
        spans = structural_spans_for_chunk(
            content=rendered_text,
            blocks=block_values,
        )
        source_entry = SourceMapEntry(
            rendered_start=0,
            rendered_end=len(rendered_text),
            anchor_kind="structured_chunk",
            block_key=chunk_key,
            block_local_start=0,
            block_local_end=len(rendered_text),
        )
        units.append(_unit(
            namespace_id=namespace_id,
            revision_id=str(chunk.get("revision_id") or revision_id),
            context_kind="structured_chunk",
            rendered_text=rendered_text,
            source_entry=source_entry,
            source_block_keys=tuple(sorted({
                str(item["scope_id"]).split(":", 1)[0]
                for item in spans
            })) or (chunk_key,),
            structure={
                "chunk_id": chunk_id,
                "chunk_key": chunk_key,
                "section_title": str(chunk.get("section_title") or ""),
                "block_type": "chunk",
                "source_role": "body",
                "relation_eligible": True,
                "structural_spans": spans,
                "parser_version": str(payload.get("parser_version") or ""),
                "structure_policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
            },
            structure_quality="structured_chunk_mapped",
        ))
    return tuple(units)


def build_context_units_for_document(
    *,
    namespace_id: str,
    document: Mapping[str, Any],
    chunks: Iterable[Mapping[str, Any]],
    processing_payload: Mapping[str, Any] | None,
) -> tuple[ContextUnit, ...]:
    """Build enhanced-only ContextUnits from persisted structure payloads.

    When historical documents lack parser blocks/source maps, fall back to
    chunk-local units and explicitly mark the degraded quality.
    """

    payload = processing_payload if isinstance(processing_payload, Mapping) else {}
    revision_id = str(document.get("active_revision_id") or document.get("revision_id") or "")
    chunk_values = list(chunks)
    blocks = [
        item for item in payload.get("blocks") or []
        if isinstance(item, Mapping) and str(item.get("text") or "").strip()
    ]
    if blocks:
        mapped = _chunk_mapped_units(
            namespace_id=namespace_id,
            revision_id=revision_id,
            chunks=chunk_values,
            blocks=blocks,
            payload=payload,
        )
        if mapped:
            return mapped
        units = []
        for index, block in enumerate(blocks, start=1):
            rendered_text = str(block.get("text") or "")
            block_key = _block_key(block, index)
            source_entry = _source_entry(
                rendered_text=rendered_text, block=block, block_key=block_key,
            )
            units.append(_unit(
                namespace_id=namespace_id,
                revision_id=revision_id,
                context_kind=_context_kind(block),
                rendered_text=rendered_text,
                source_entry=source_entry,
                source_block_keys=(block_key,),
                structure=_block_structure(block, payload),
                structure_quality="structured",
            ))
        return tuple(units)

    fallback_units = []
    for chunk in chunk_values:
        rendered_text = str(chunk.get("content") or "")
        if not rendered_text.strip():
            continue
        chunk_key = str(chunk.get("chunk_key") or chunk.get("id") or "chunk")
        source_entry = SourceMapEntry(
            rendered_start=0,
            rendered_end=len(rendered_text),
            anchor_kind="legacy_chunk_local",
            block_key=chunk_key,
            block_local_start=0,
            block_local_end=len(rendered_text),
            page=None,
        )
        fallback_units.append(_unit(
            namespace_id=namespace_id,
            revision_id=str(chunk.get("revision_id") or revision_id),
            context_kind="legacy_chunk",
            rendered_text=rendered_text,
            source_entry=source_entry,
            source_block_keys=(chunk_key,),
            structure={
                "chunk_id": int(chunk.get("id") or 0),
                "chunk_key": chunk_key,
                "section_title": str(chunk.get("section_title") or ""),
                "source_map_quality": "legacy_chunk_local",
                "source_role": "body",
                "relation_eligible": True,
                "structure_policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
            },
            structure_quality="legacy_chunk_local",
        ))
    return tuple(fallback_units)


def build_shadow_context_manifest(
    *,
    namespace_id: str,
    documents: Iterable[Mapping[str, Any]],
    chunks_by_document: Mapping[int, Iterable[Mapping[str, Any]]],
    payload_loader: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
) -> ShadowContextManifest:
    """Build a deterministic structure/source-map manifest for enhanced Shadow."""

    all_units: list[ContextUnit] = []
    document_summaries = []
    for document in sorted(documents, key=lambda item: int(item["id"])):
        document_id = int(document["id"])
        chunks = list(chunks_by_document.get(document_id) or [])
        payload = payload_loader(document)
        units = build_context_units_for_document(
            namespace_id=namespace_id,
            document=document,
            chunks=chunks,
            processing_payload=payload,
        )
        all_units.extend(units)
        document_summaries.append({
            "document_id": document_id,
            "revision_id": str(document.get("active_revision_id") or ""),
            "content_hash": str(document.get("doc_hash") or ""),
            "parser_version": str((payload or {}).get("parser_version") or ""),
            "block_count": len((payload or {}).get("blocks") or []),
            "context_unit_count": len(units),
            "structure_quality": sorted({
                unit.structure_quality for unit in units
            }),
        })
    units_tuple = tuple(sorted(all_units, key=lambda item: item.context_id))
    artifact_payload = {
        "policy_version": SHADOW_CONTEXT_POLICY_VERSION,
        "source_map_policy": SHADOW_CONTEXT_SOURCE_MAP_POLICY,
        "documents": document_summaries,
        "contexts": [{
            "context_id": unit.context_id,
            "revision_id": unit.document_revision_id,
            "kind": unit.context_kind,
            "rendered_hash": unit.rendered_hash,
            "source_block_keys": list(unit.source_block_keys),
            "source_map": [asdict(item) for item in unit.source_map],
            "structure_quality": unit.structure_quality,
            "source_role": str(unit.structure.get("source_role") or ""),
            "relation_eligible": bool(unit.structure.get("relation_eligible")),
            "structure_policy_version": str(
                unit.structure.get("structure_policy_version") or ""
            ),
        } for unit in units_tuple],
    }
    summary = {
        "documents": len(document_summaries),
        "context_units": len(units_tuple),
        "source_map_entries": sum(len(unit.source_map) for unit in units_tuple),
        "legacy_chunk_local_units": sum(
            1 for unit in units_tuple
            if unit.structure_quality == "legacy_chunk_local"
        ),
        "context_kinds": sorted({unit.context_kind for unit in units_tuple}),
        "relation_eligible_contexts": sum(
            int(bool(unit.structure.get("relation_eligible")))
            for unit in units_tuple
        ),
        "relation_ineligible_contexts": sum(
            int(not bool(unit.structure.get("relation_eligible")))
            for unit in units_tuple
        ),
    }
    return ShadowContextManifest(
        policy_version=SHADOW_CONTEXT_POLICY_VERSION,
        source_map_policy=SHADOW_CONTEXT_SOURCE_MAP_POLICY,
        artifact_hash=_digest(artifact_payload),
        context_units=units_tuple,
        summary=summary,
    )
