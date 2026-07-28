"""Enhanced-only structured context units for Shadow v2 relation work."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .quality import canonical_json


SHADOW_CONTEXT_POLICY_VERSION = "shadow-context-source-map-v1"
SHADOW_CONTEXT_SOURCE_MAP_POLICY = "block-or-legacy-chunk-local-v1"


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
    return {
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
    blocks = [
        item for item in payload.get("blocks") or []
        if isinstance(item, Mapping) and str(item.get("text") or "").strip()
    ]
    if blocks:
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
    for chunk in chunks:
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
    }
    return ShadowContextManifest(
        policy_version=SHADOW_CONTEXT_POLICY_VERSION,
        source_map_policy=SHADOW_CONTEXT_SOURCE_MAP_POLICY,
        artifact_hash=_digest(artifact_payload),
        context_units=units_tuple,
        summary=summary,
    )
