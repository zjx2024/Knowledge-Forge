"""Deterministic, structure-aware Documents input for GraphRAG Native.

The native strategy deliberately reads persisted parser blocks rather than
legacy chunks.  Parser blocks are source artefacts; chunks, canonical entities,
and formal relation records are not native inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class NativeInputError(RuntimeError):
    """A source snapshot cannot be rendered safely as native GraphRAG input."""


class NativeInputService(Protocol):
    db: Any

    def document_structure(self, document_id: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class NativeInputDocument:
    id: str
    title: str
    text: str
    raw_data: str


@dataclass(frozen=True)
class NativeSourceMap:
    project_id: int
    document_id: int
    revision_id: str
    source_hash: str
    input_document_id: str
    input_title: str
    block_ids: str
    page: int | None
    heading_path: str
    table_id: str
    row: int | None
    column: str
    header_path: str
    list_id: str
    list_item: int | None
    source_start: int | None
    source_end: int | None
    rendered_text_start: int
    rendered_text_end: int


@dataclass(frozen=True)
class NativeInputBuild:
    project_id: int
    source_snapshot_hash: str
    input_manifest_hash: str
    documents_path: Path
    source_map_path: Path
    manifest_path: Path
    input_documents: tuple[NativeInputDocument, ...]
    source_maps: tuple[NativeSourceMap, ...]
    run_documents: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class NativeInputPreflight:
    """Source-only snapshot used to record a terminal EMPTY_INPUT attempt."""

    project_id: int
    source_snapshot_hash: str
    ready: bool
    reason: str = ""


class GraphRAGNativeInputBuilder:
    """Render active document revisions into official GraphRAG Parquet input."""

    schema_version = "graphrag-native-input-v1"

    def __init__(self, service: NativeInputService):
        self.service = service

    @staticmethod
    def _opaque_id(*parts: object) -> str:
        material = "|".join(str(part) for part in parts)
        return "grn_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _metadata(block: dict[str, Any]) -> dict[str, Any]:
        metadata = block.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _block_identity(block: dict[str, Any]) -> str:
        return str(block.get("key") or block.get("id") or "")

    @classmethod
    def _structure_hash(cls, blocks: list[dict[str, Any]]) -> str:
        normalized = [
            {
                "key": cls._block_identity(block),
                "type": str(block.get("block_type") or ""),
                "text": str(block.get("text") or ""),
                "order": int(block.get("order") or 0),
                "heading_path": list(block.get("heading_path") or []),
                "page": block.get("page"),
                "metadata": cls._metadata(block),
            }
            for block in blocks
        ]
        return _digest(normalized)

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None and str(value) != "" else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _render_unit(
        cls,
        *,
        project_id: int,
        document: dict[str, Any],
        revision: dict[str, Any],
        unit_kind: str,
        blocks: list[dict[str, Any]],
        ordinal: int,
    ) -> tuple[NativeInputDocument, NativeSourceMap]:
        if not blocks:
            raise NativeInputError("native input unit 不能为空")
        first = blocks[0]
        metadata = cls._metadata(first)
        block_ids = [cls._block_identity(block) for block in blocks]
        if not all(block_ids):
            raise NativeInputError("结构块缺少稳定 key，拒绝构建 native 输入")
        input_id = cls._opaque_id(
            project_id, document["id"], revision["revision_id"], unit_kind, ordinal, *block_ids,
        )
        # The source identity is deliberately recognizable by the native prompt
        # and is not a domain entity.  It keeps otherwise identical text from
        # separate source documents from receiving an identical TextUnit hash.
        source_identity = f"[KF_SOURCE_ID:{input_id}]"
        heading_path = [str(value) for value in (first.get("heading_path") or []) if str(value)]
        body = "\n\n".join(str(block.get("text") or "").strip() for block in blocks).strip()
        if not body:
            raise NativeInputError("结构块没有可用于 native 输入的文本")
        # Keep parser structure in the rendered document itself.  GraphRAG's
        # standard Parquet reader treats ``raw_data`` as provenance only, so
        # headings and table headers must be repeated in the text that reaches
        # extraction.  This is structural rendering, never a domain/predicate
        # special case.
        context_lines: list[str] = []
        if heading_path:
            context_lines.append("Section: " + " > ".join(heading_path))
        if unit_kind == "table_row":
            table_title = str(metadata.get("table_title") or metadata.get("table_name") or "").strip()
            headers = metadata.get("header_path") or metadata.get("headers") or []
            if isinstance(headers, str):
                headers = [headers]
            if table_title:
                context_lines.append("Table: " + table_title)
            if isinstance(headers, (list, tuple)) and any(str(item).strip() for item in headers):
                context_lines.append("Columns: " + " | ".join(str(item).strip() for item in headers if str(item).strip()))
        text = source_identity + "\n" + "\n".join([*context_lines, body])
        title = "native:" + input_id
        raw_data = {
            "schema_version": cls.schema_version,
            "project_id": int(project_id),
            "document_id": int(document["id"]),
            "revision_id": str(revision["revision_id"]),
            "human_title": str(document.get("original_filename") or ""),
            "unit_kind": unit_kind,
            "block_ids": block_ids,
            "heading_path": heading_path,
        }
        map_row = NativeSourceMap(
            project_id=int(project_id),
            document_id=int(document["id"]),
            revision_id=str(revision["revision_id"]),
            source_hash=str(revision.get("source_hash") or document.get("doc_hash") or ""),
            input_document_id=input_id,
            input_title=title,
            block_ids=_canonical_json(block_ids),
            page=cls._as_int(first.get("page")),
            heading_path=_canonical_json(heading_path),
            table_id=str(metadata.get("table_id") or metadata.get("table") or ""),
            row=cls._as_int(metadata.get("row") or metadata.get("row_index")),
            column=str(metadata.get("column") or ""),
            header_path=_canonical_json(metadata.get("header_path") or metadata.get("headers") or []),
            list_id=str(metadata.get("list_id") or ""),
            list_item=cls._as_int(metadata.get("list_item") or metadata.get("item_index")),
            source_start=cls._as_int(metadata.get("source_start")),
            source_end=cls._as_int(metadata.get("source_end")),
            rendered_text_start=len(source_identity) + 1,
            rendered_text_end=len(text),
        )
        return NativeInputDocument(input_id, title, text, _canonical_json(raw_data)), map_row

    @classmethod
    def _units_for_document(
        cls, project_id: int, document: dict[str, Any], revision: dict[str, Any], blocks: list[dict[str, Any]],
    ) -> list[tuple[NativeInputDocument, NativeSourceMap]]:
        usable = [
            block for block in sorted(blocks, key=lambda item: (int(item.get("order") or 0), cls._block_identity(item)))
            if isinstance(block, dict) and str(block.get("text") or "").strip()
        ]
        if not usable:
            raise NativeInputError(f"文档 {document['id']} 没有可用结构块，拒绝回退 legacy chunks")
        rendered: list[tuple[NativeInputDocument, NativeSourceMap]] = []
        paragraph_group: list[dict[str, Any]] = []
        ordinal = 0

        def flush_paragraphs() -> None:
            nonlocal ordinal, paragraph_group
            if paragraph_group:
                rendered.append(cls._render_unit(
                    project_id=project_id, document=document, revision=revision,
                    unit_kind="section", blocks=paragraph_group, ordinal=ordinal,
                ))
                ordinal += 1
                paragraph_group = []

        for block in usable:
            kind = str(block.get("block_type") or "paragraph").lower()
            metadata = cls._metadata(block)
            is_table_row = kind == "table_row" or bool(metadata.get("table_id") and metadata.get("row") is not None)
            is_list_item = kind in {"list_item", "list"} or bool(metadata.get("list_id"))
            if kind == "heading":
                flush_paragraphs()
                continue
            if is_table_row or is_list_item or kind in {"image", "ocr"}:
                flush_paragraphs()
                rendered.append(cls._render_unit(
                    project_id=project_id, document=document, revision=revision,
                    unit_kind="table_row" if is_table_row else "list_item" if is_list_item else "media",
                    blocks=[block], ordinal=ordinal,
                ))
                ordinal += 1
                continue
            paragraph_group.append(block)
        flush_paragraphs()
        if not rendered:
            raise NativeInputError(f"文档 {document['id']} 无法生成 native 输入单元")
        return rendered

    def preflight(self, *, project_id: int) -> NativeInputPreflight:
        """Check whether a complete source snapshot can render input at all.

        This does not create directories or import GraphRAG.  It exists so a
        zero/structurally empty project has a durable ``EMPTY_INPUT`` run
        state before any model or GraphRAG workflow is invoked.
        """

        documents = sorted(
            self.service.db.list_documents(project_id), key=lambda item: int(item["id"]),
        )
        source_rows: list[dict[str, Any]] = []
        if not documents:
            return NativeInputPreflight(
                project_id=int(project_id),
                source_snapshot_hash=_digest({"schema_version": self.schema_version, "documents": []}),
                ready=False, reason="项目没有可冻结的源文档",
            )
        for document in documents:
            revision_id = str(document.get("active_revision_id") or "")
            if not revision_id:
                return NativeInputPreflight(
                    project_id=int(project_id),
                    source_snapshot_hash=_digest({"schema_version": self.schema_version, "document_id": document["id"], "revision": "missing"}),
                    ready=False, reason=f"文档 {document['id']} 没有 active revision",
                )
            revision = self.service.db.get_document_revision(revision_id)
            if not revision or str(revision.get("status") or "") != "active":
                return NativeInputPreflight(
                    project_id=int(project_id),
                    source_snapshot_hash=_digest({"schema_version": self.schema_version, "document_id": document["id"], "revision": revision_id}),
                    ready=False, reason=f"文档 {document['id']} active revision 不可用",
                )
            if int(revision.get("project_id") or 0) != int(project_id) or int(revision.get("document_id") or 0) != int(document["id"]):
                return NativeInputPreflight(
                    project_id=int(project_id),
                    source_snapshot_hash=_digest({"schema_version": self.schema_version, "document_id": document["id"], "revision": revision_id}),
                    ready=False, reason="active revision 与项目/文档范围不一致",
                )
            structure = self.service.document_structure(int(document["id"]))
            blocks = structure.get("blocks") if isinstance(structure.get("blocks"), list) else []
            source_rows.append({
                "document_id": int(document["id"]), "revision_id": revision_id,
                "source_hash": str(revision.get("source_hash") or ""),
                "structure_hash": self._structure_hash(blocks),
            })
            try:
                self._units_for_document(int(project_id), document, revision, blocks)
            except NativeInputError as exc:
                return NativeInputPreflight(
                    project_id=int(project_id),
                    source_snapshot_hash=_digest({"schema_version": self.schema_version, "documents": source_rows}),
                    ready=False, reason=str(exc),
                )
        return NativeInputPreflight(
            project_id=int(project_id),
            source_snapshot_hash=_digest({"schema_version": self.schema_version, "documents": source_rows}),
            ready=True,
        )

    def preview(
        self,
        *,
        project_id: int,
        document_id: int | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Render Native Documents in memory without creating a run bundle."""

        documents = sorted(
            self.service.db.list_documents(project_id), key=lambda item: int(item["id"]),
        )
        if document_id is not None:
            documents = [
                item for item in documents if int(item["id"]) == int(document_id)
            ]
            if not documents:
                raise LookupError("文档不存在或不属于当前项目")
        rendered: list[dict[str, Any]] = []
        for document in documents:
            revision_id = str(document.get("active_revision_id") or "")
            if not revision_id:
                raise NativeInputError(f"文档 {document['id']} 没有 active revision")
            revision = self.service.db.get_document_revision(revision_id)
            if not revision or str(revision.get("status") or "") != "active":
                raise NativeInputError(f"文档 {document['id']} active revision 不可用")
            structure = self.service.document_structure(int(document["id"]))
            blocks = structure.get("blocks") if isinstance(structure.get("blocks"), list) else []
            units = self._units_for_document(
                int(project_id), document, revision, blocks,
            )
            for native_document, source_map in units:
                rendered.append({
                    **asdict(native_document),
                    "source_map": asdict(source_map),
                    "source_filename": str(document.get("original_filename") or ""),
                })
        offset = max(0, int(offset))
        limit = min(500, max(1, int(limit)))
        return {
            "project_id": int(project_id),
            "document_id": int(document_id) if document_id is not None else None,
            "total": len(rendered),
            "offset": offset,
            "limit": limit,
            "items": rendered[offset:offset + limit],
        }

    def build(self, *, project_id: int, run_root: Path) -> NativeInputBuild:
        """Freeze all current active revisions into deterministic Parquet and source maps."""

        run_root = Path(run_root).resolve()
        input_root = run_root / "input"
        manifest_root = run_root / "kf-manifest"
        # GraphRAG 3.1.1 ``init`` pre-creates an empty input directory.  It is
        # a valid bootstrap state; any existing content is an old attempt and
        # must not be overwritten.
        if input_root.exists() and (input_root.is_symlink() or any(input_root.iterdir())):
            raise NativeInputError("native staging input 目录不是 GraphRAG init 的空目录")
        input_root.mkdir(parents=True, exist_ok=True)
        # The executor writes candidate provenance before freezing Documents.
        # ``kf-manifest`` is therefore a shared staging directory, while this
        # builder exclusively owns only its two input artifacts.  Allowing the
        # directory itself makes the sequence re-entrant; allowing either
        # owned artifact to pre-exist would instead risk mixing attempts, so
        # it remains a fail-closed error.
        if manifest_root.exists() and (manifest_root.is_symlink() or not manifest_root.is_dir()):
            raise NativeInputError("native staging kf-manifest 不是安全目录")
        manifest_root.mkdir(parents=True, exist_ok=True)
        for owned_name in ("source-map.parquet", "input-documents-manifest.jsonl"):
            owned_path = manifest_root / owned_name
            if owned_path.exists() or owned_path.is_symlink():
                raise NativeInputError(
                    f"native staging kf-manifest 已含本步骤产物 {owned_name}，拒绝覆盖旧尝试"
                )
        documents = sorted(
            self.service.db.list_documents(project_id), key=lambda item: int(item["id"]),
        )
        if not documents:
            raise NativeInputError("项目没有可冻结的源文档")
        all_documents: list[NativeInputDocument] = []
        source_maps: list[NativeSourceMap] = []
        run_documents: list[dict[str, Any]] = []
        for document_ordinal, document in enumerate(documents):
            revision_id = str(document.get("active_revision_id") or "")
            if not revision_id:
                raise NativeInputError(f"文档 {document['id']} 没有 active revision")
            revision = self.service.db.get_document_revision(revision_id)
            if not revision or str(revision.get("status") or "") != "active":
                raise NativeInputError(f"文档 {document['id']} active revision 不可用")
            if int(revision.get("project_id") or 0) != int(project_id) or int(revision.get("document_id") or 0) != int(document["id"]):
                raise NativeInputError("active revision 与项目/文档范围不一致")
            structure = self.service.document_structure(int(document["id"]))
            blocks = structure.get("blocks") if isinstance(structure.get("blocks"), list) else []
            structure_hash = self._structure_hash(blocks)
            units = self._units_for_document(project_id, document, revision, blocks)
            all_documents.extend(item[0] for item in units)
            source_maps.extend(item[1] for item in units)
            run_documents.append({
                "project_id": project_id,
                "document_id": int(document["id"]),
                "revision_id": revision_id,
                "source_hash": str(revision.get("source_hash") or ""),
                "structure_hash": structure_hash,
                "input_document_count": len(units),
                "input_manifest_hash": "",  # populated after the complete manifest is frozen
                "ordinal": document_ordinal,
            })
        ids = [item.id for item in all_documents]
        titles = [item.title for item in all_documents]
        if len(ids) != len(set(ids)) or len(titles) != len(set(titles)):
            raise NativeInputError("native input ID 或 title 重复，拒绝静默去重")
        document_rows = [asdict(item) for item in all_documents]
        source_map_rows = [asdict(item) for item in source_maps]
        manifest_records = [
            {
                "id": row["id"], "title": row["title"],
                "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                "raw_data": json.loads(row["raw_data"]),
            }
            for row in document_rows
        ]
        input_manifest_hash = _digest(manifest_records)
        for item in run_documents:
            item["input_manifest_hash"] = input_manifest_hash
        source_snapshot_hash = _digest({
            "schema_version": self.schema_version,
            "documents": [
                {
                    "document_id": item["document_id"], "revision_id": item["revision_id"],
                    "source_hash": item["source_hash"], "structure_hash": item["structure_hash"],
                }
                for item in run_documents
            ],
            "input_manifest_hash": input_manifest_hash,
        })
        documents_path = input_root / "documents.parquet"
        source_map_path = manifest_root / "source-map.parquet"
        manifest_path = manifest_root / "input-documents-manifest.jsonl"
        pq.write_table(pa.Table.from_pylist(document_rows), documents_path, compression="zstd")
        pq.write_table(pa.Table.from_pylist(source_map_rows), source_map_path, compression="zstd")
        manifest_path.write_text(
            "".join(_canonical_json(item) + "\n" for item in manifest_records), encoding="utf-8",
        )
        return NativeInputBuild(
            project_id=project_id, source_snapshot_hash=source_snapshot_hash,
            input_manifest_hash=input_manifest_hash, documents_path=documents_path,
            source_map_path=source_map_path, manifest_path=manifest_path,
            input_documents=tuple(all_documents), source_maps=tuple(source_maps),
            run_documents=tuple(run_documents),
        )
