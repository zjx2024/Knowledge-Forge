"""Read/query facade for immutable GraphRAG Native bundles.

The facade is deliberately separate from the executor.  It owns no graph
mutation and talks only to the native control-plane records plus the selected
run's file bundle; Stable/Enhanced/HugeGraph/pgvector are out of scope.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import uuid

import pyarrow.parquet as pq

from .config import Config
from .graphrag_native_artifacts import GraphRAGNativeBundleValidator, safe_bundle_path
from .graphrag_native_query import QUERY_MODES, QUERY_PROTOCOL_VERSION


class GraphRAGNativeUnavailable(RuntimeError):
    """Native bundle is absent, stale, incomplete, or not queryable."""


class GraphRAGNativeStale(GraphRAGNativeUnavailable):
    """Current document revisions differ from the immutable native snapshot."""


def _native_list(value: Any) -> list[str]:
    """Normalize Arrow/JSON list values without accepting scalar garbage."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def _native_page_label(pages: list[int]) -> str:
    values = sorted({int(page) for page in pages if int(page) > 0})
    if not values:
        return ""
    if len(values) == 1:
        return f"第 {values[0]} 页"
    if values == list(range(values[0], values[-1] + 1)):
        return f"第 {values[0]}–{values[-1]} 页"
    shown = "、".join(str(value) for value in values[:6])
    return f"第 {shown}{' 等' if len(values) > 6 else ''}页"


def _native_evidence_text(value: Any, *, limit: int = 1600) -> str:
    """Remove transport-only headers from a GraphRAG TextUnit excerpt."""

    lines = str(value or "").replace("\r\n", "\n").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and (
        re.fullmatch(r"id:\s*\S+\.?", lines[0].strip(), flags=re.IGNORECASE)
        or re.fullmatch(r"\[KF_SOURCE_ID:[^\]]+\]", lines[0].strip())
        or lines[0].strip().lower().startswith("section: ")
    ):
        lines.pop(0)
    text = "\n".join(lines).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _native_citation_ids(value: Any, table: str) -> list[str]:
    """Resolve GraphRAG's stable ``[Data: Table (ids)]`` citations."""

    result: list[str] = []
    pattern = re.compile(rf"\b{re.escape(table)}\s*\(([^)]*)\)", re.IGNORECASE)
    for match in pattern.finditer(str(value or "")):
        result.extend(re.findall(r"\d+", match.group(1)))
    return list(dict.fromkeys(result))


def _native_report_prose(value: Any) -> str:
    """Hide machine citation syntax after it has been projected to source cards."""

    text = re.sub(r"\s*\[Data:\s*[^\]]+\]", "", str(value or ""), flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", text).strip()


def _native_relation_label(description: Any) -> str:
    text = re.sub(r"\s+", " ", str(description or "")).strip()
    if not text:
        return "语义关系"
    sentence = re.split(r"(?<=[。！？.!?])\s+", text, maxsplit=1)[0]
    return sentence if len(sentence) <= 42 else sentence[:41].rstrip() + "…"


class GraphRAGNativeRuntime:
    VIEWABLE_ARTIFACTS = {
        "documents": "input/documents.parquet",
        "source-map": "kf-manifest/source-map.parquet",
        "text-units": "output/text_units.parquet",
        "entities": "output/entities.parquet",
        "relationships": "output/relationships.parquet",
        "communities": "output/communities.parquet",
        "community-reports": "output/community_reports.parquet",
    }

    def __init__(self, service: Any, *, query_timeout_seconds: int = 300) -> None:
        self.service = service
        self.query_timeout_seconds = max(1, int(query_timeout_seconds))
        self.validator = GraphRAGNativeBundleValidator()

    @property
    def native_root(self) -> Path:
        return (Config.files_root / "graphrag_native").resolve()

    def _bundle_root(self, run: dict[str, Any]) -> Path:
        relative = str(run.get("bundle_relative_path") or "")
        if not relative:
            raise GraphRAGNativeUnavailable("GraphRAG Native run 尚未产生可查询 bundle")
        return safe_bundle_path(self.native_root, relative)

    def _assert_fresh(self, project_id: int, run: dict[str, Any]) -> None:
        rows = self.service.db.list_graphrag_native_run_documents(str(run["run_id"]))
        current = {int(row["id"]): row for row in self.service.db.list_documents(project_id)}
        frozen_ids = {int(row["document_id"]) for row in rows}
        if not rows or frozen_ids != set(current):
            raise GraphRAGNativeStale("native source snapshot 与当前项目文档集合不同")
        for row in rows:
            document = current[int(row["document_id"])]
            revision_id = str(document.get("active_revision_id") or "")
            if revision_id != str(row["revision_id"]):
                raise GraphRAGNativeStale("native source snapshot 已被新 revision 取代")
            revision = self.service.db.get_document_revision(revision_id)
            if not revision or str(revision.get("source_hash") or "") != str(row["source_hash"]):
                raise GraphRAGNativeStale("native source hash 已变化")

    def current_run(self, project_id: int, *, validate_bundle: bool = False) -> tuple[dict[str, Any], Path]:
        run = self.service.db.get_active_graphrag_native_run(project_id)
        if not run:
            raise GraphRAGNativeUnavailable("当前项目没有已验证的 GraphRAG Native run")
        if str(run.get("state") or "") == "STALE":
            raise GraphRAGNativeStale("GraphRAG Native run 已过期，需要重新构建")
        if str(run.get("state") or "") != "READY":
            raise GraphRAGNativeUnavailable("当前 GraphRAG Native run 尚未 READY")
        state = self.service.db.get_graphrag_native_project_state(project_id) or {}
        desired = str(state.get("desired_source_snapshot_hash") or "")
        if desired and desired != str(run.get("source_snapshot_hash") or ""):
            raise GraphRAGNativeStale("当前 native source/model/config 快照已由后续 run 取代")
        self._assert_fresh(project_id, run)
        root = self._bundle_root(run)
        if validate_bundle:
            self.validator.validate(root)
        return run, root

    def status(self, project_id: int) -> dict[str, Any]:
        state = self.service.db.get_graphrag_native_project_state(project_id) or {}
        run = self.service.db.get_active_graphrag_native_run(project_id)
        payload = {"project_id": int(project_id), "state": state, "run": run, "fresh": False}
        if run:
            try:
                self.current_run(project_id, validate_bundle=False)
                payload["fresh"] = True
            except GraphRAGNativeStale as exc:
                payload["stale_reason"] = str(exc)
            except GraphRAGNativeUnavailable as exc:
                payload["unavailable_reason"] = str(exc)
        return payload

    @staticmethod
    def _records(root: Path, name: str) -> list[dict[str, Any]]:
        return GraphRAGNativeRuntime._artifact_records(root, f"output/{name}")

    @staticmethod
    def _artifact_records(root: Path, relative: str) -> list[dict[str, Any]]:
        path = safe_bundle_path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise GraphRAGNativeUnavailable(f"native artifact 缺少 {relative}")
        return pq.read_table(path).to_pylist()

    def _native_source_context(
        self,
        project_id: int,
        source_maps: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Project immutable source-map rows to browser-only provenance.

        Graph facts continue to come exclusively from the native bundle.  The
        PostgreSQL document row supplies only the human filename/download URL;
        optional persisted parser blocks restore thumbnails and table previews.
        """

        documents = {
            int(item["id"]): item
            for item in self.service.db.list_documents(project_id)
            if int(item.get("id") or 0) > 0
        }
        structures: dict[int, dict[str, dict[str, Any]]] = {}
        structure_reader = getattr(self.service, "document_structure", None)
        if callable(structure_reader):
            for document_id in sorted({
                int(item.get("document_id") or 0) for item in source_maps
                if int(item.get("document_id") or 0) > 0
            }):
                try:
                    payload = structure_reader(document_id)
                except Exception:
                    continue
                structures[document_id] = {
                    str(block.get("key") or block.get("id") or ""): block
                    for block in payload.get("blocks") or []
                    if isinstance(block, dict)
                }

        result: dict[str, dict[str, Any]] = {}
        for row in source_maps:
            input_document_id = str(row.get("input_document_id") or "")
            document_id = int(row.get("document_id") or 0)
            if not input_document_id or document_id <= 0:
                continue
            document = documents.get(document_id, {})
            block_map = structures.get(document_id, {})
            blocks = [
                block_map[key] for key in _native_list(row.get("block_ids"))
                if key in block_map
            ]
            pages: set[int] = set()
            for raw_page in [row.get("page"), *(block.get("page") for block in blocks)]:
                try:
                    page = int(raw_page)
                except (TypeError, ValueError):
                    continue
                if page > 0:
                    pages.add(page)
            headings = _native_list(row.get("heading_path"))
            media_assets: list[dict[str, Any]] = []
            table_rows: list[str] = []
            for block in blocks:
                metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
                block_type = str(block.get("block_type") or "")
                if block_type == "image" and str(metadata.get("url") or "").strip():
                    media_assets.append({
                        "type": "image",
                        "url": str(metadata.get("url") or ""),
                        "width": int(metadata.get("width") or 0),
                        "height": int(metadata.get("height") or 0),
                        "title": str(
                            metadata.get("title") or block.get("section_title")
                            or "文档图片"
                        ),
                        "description": str(
                            metadata.get("description") or metadata.get("caption") or ""
                        ),
                        "sources": list(metadata.get("sources") or []),
                    })
                if block_type in {"table", "table_row"}:
                    text = str(block.get("text") or "").strip()
                    if text:
                        table_rows.append(text)
            storage_path = str(document.get("storage_path") or "").strip()
            result[input_document_id] = {
                "document_id": document_id,
                "revision_id": str(row.get("revision_id") or ""),
                "filename": str(
                    document.get("original_filename") or row.get("input_title") or ""
                ),
                "document_url": storage_path or f"/api/documents/{document_id}/download",
                "mime_type": str(document.get("mime_type") or ""),
                "section_title": headings[-1] if headings else "",
                "pages": sorted(pages),
                "page_label": _native_page_label(sorted(pages)),
                "media_assets": media_assets[:12],
                "table": ({
                    "title": headings[-1] if headings else "表格内容",
                    "text": "\n".join(table_rows)[:12_000],
                } if table_rows else None),
            }
        return result

    def graph(
        self, project_id: int, *, offset: int = 0, limit: int = 200,
        include_isolates: bool = False,
    ) -> dict[str, Any]:
        run, root = self.current_run(project_id, validate_bundle=True)
        offset = max(0, int(offset)); limit = min(1000, max(1, int(limit)))
        entities = self._records(root, "entities.parquet")
        relationships = self._records(root, "relationships.parquet")
        selected = relationships[offset:offset + limit]
        titles = {str(row.get("source") or "") for row in selected} | {str(row.get("target") or "") for row in selected}
        # The audit-table default remains deliberately compact, while a graph
        # browser can ask for every entity.  Otherwise isolated entities would
        # disappear merely because they have no relationship row in this page.
        nodes = entities if include_isolates else [
            row for row in entities if str(row.get("title") or "") in titles
        ]
        return {
            "run_id": run["run_id"], "offset": offset, "limit": limit,
            "total_relationships": len(relationships), "total_entities": len(entities),
            "nodes": nodes,
            "edges": selected, "truncated": offset + limit < len(relationships),
        }

    def explorer_payload(self, project_id: int) -> dict[str, Any]:
        """Normalize a fresh immutable bundle for the shared graph explorer."""

        run, root = self.current_run(project_id, validate_bundle=True)
        entities = self._records(root, "entities.parquet")
        relationships = self._records(root, "relationships.parquet")
        communities = self._records(root, "communities.parquet")
        reports = self._records(root, "community_reports.parquet")
        text_units = self._records(root, "text_units.parquet")
        source_maps = self._artifact_records(root, "kf-manifest/source-map.parquet")
        source_context = self._native_source_context(project_id, source_maps)
        text_unit_by_id = {
            str(item.get("id") or ""): item for item in text_units
            if str(item.get("id") or "")
        }
        report_by_key = {
            (
                str(item.get("level") if item.get("level") is not None else 0),
                str(item.get("community") if item.get("community") is not None else ""),
            ): item
            for item in reports
        }
        title_to_id: dict[str, str] = {}
        nodes: list[dict[str, Any]] = []
        for item in entities:
            entity_id = str(item.get("id") or item.get("title") or "").strip()
            if not entity_id:
                continue
            title = str(item.get("title") or entity_id)
            title_to_id[title] = entity_id
            native_description = str(item.get("description") or "").strip()
            nodes.append({
                "id": entity_id,
                "title": title,
                "type": str(item.get("type") or "实体"),
                "type_id": str(item.get("type") or "entity"),
                "description": (
                    native_description
                    or f"“{title}”是 GraphRAG 从原文文本单元中识别的实体。"
                ),
                "description_source": (
                    "native_entity_description"
                    if native_description else "native_entity_title_fallback"
                ),
                "degree": int(item.get("degree") or 0),
                "frequency": int(item.get("frequency") or 0),
                "text_unit_ids": list(item.get("text_unit_ids") or []),
                "group": "",
            })
        node_by_id = {str(item["id"]): item for item in nodes}

        def source_record(
            text_unit_id: str,
            *,
            relation_key: str = "",
            source_title: str = "",
            predicate_label: str = "",
            target_title: str = "",
        ) -> dict[str, Any] | None:
            text_unit = text_unit_by_id.get(str(text_unit_id))
            if not text_unit:
                return None
            return {
                "relation_key": relation_key,
                "source_title": source_title,
                "predicate_label": predicate_label,
                "target_title": target_title,
                "evidence_text": _native_evidence_text(text_unit.get("text")),
                "text_unit_id": str(text_unit_id),
                "source_kind": "relationship" if relation_key else "text_unit",
                "source_info": dict(source_context.get(
                    str(text_unit.get("document_id") or ""), {}
                )),
            }

        relation_source_by_id: dict[str, dict[str, Any]] = {}
        relation_id_by_human: dict[str, str] = {}
        edges: list[dict[str, Any]] = []
        for index, item in enumerate(relationships):
            source_title = str(item.get("source") or "")
            target_title = str(item.get("target") or "")
            source = title_to_id.get(source_title, source_title)
            target = title_to_id.get(target_title, target_title)
            if source not in node_by_id or target not in node_by_id:
                continue
            relation_id = str(item.get("id") or f"native-edge-{index}")
            if item.get("human_readable_id") is not None:
                relation_id_by_human[str(item["human_readable_id"])] = relation_id
            text_unit_ids = [
                str(value) for value in item.get("text_unit_ids") or [] if str(value)
            ]
            sources = [
                record
                for text_unit_id in text_unit_ids
                if (record := source_record(
                    text_unit_id,
                    relation_key=relation_id,
                    source_title=source_title,
                    predicate_label="语义关系",
                    target_title=target_title,
                )) is not None
            ]
            evidence_parts = list(dict.fromkeys(
                str(source.get("evidence_text") or "") for source in sources
                if str(source.get("evidence_text") or "")
            ))
            evidence_text = "\n\n".join(evidence_parts[:2])
            if len(evidence_text) > 1600:
                evidence_text = evidence_text[:1600].rstrip() + "…"
            description = str(item.get("description") or "")
            edge = {
                "id": relation_id,
                "source": source,
                "target": target,
                "source_title": source_title,
                "target_title": target_title,
                "label": _native_relation_label(description),
                "description": description,
                "weight": float(item.get("weight") or 1.0),
                "evidence_count": len(text_unit_ids),
                "evidence_text": evidence_text,
                "source_info": dict(sources[0].get("source_info") or {}) if sources else {},
                "evidence_sources": sources,
                "text_unit_ids": text_unit_ids,
                "directed": True,
            }
            edges.append(edge)
            if sources:
                relation_source_by_id[relation_id] = sources[0]

        entity_id_by_human = {
            str(item["human_readable_id"]): str(item.get("id") or "")
            for item in entities if item.get("human_readable_id") is not None
        }
        entity_by_id = {
            str(item.get("id") or ""): item for item in entities
            if str(item.get("id") or "")
        }

        def report_payload(report: dict[str, Any]) -> dict[str, Any] | None:
            if not report:
                return None
            report_sources: list[dict[str, Any]] = []
            seen_sources: set[tuple[str, str]] = set()
            findings: list[dict[str, Any]] = []
            for finding in report.get("findings") or []:
                if not isinstance(finding, dict):
                    continue
                explanation = str(finding.get("explanation") or "")
                relation_keys = [
                    relation_id_by_human[value]
                    for value in _native_citation_ids(explanation, "Relationships")
                    if value in relation_id_by_human
                ]
                entity_keys = [
                    entity_id_by_human[value]
                    for value in _native_citation_ids(explanation, "Entities")
                    if value in entity_id_by_human
                ]
                sources = [
                    dict(relation_source_by_id[key]) for key in relation_keys
                    if key in relation_source_by_id
                ]
                covered_text_units = {
                    str(item.get("text_unit_id") or "") for item in sources
                }
                for entity_id in entity_keys:
                    entity = entity_by_id.get(entity_id, {})
                    for text_unit_id in entity.get("text_unit_ids") or []:
                        if str(text_unit_id) in covered_text_units:
                            continue
                        item = source_record(
                            str(text_unit_id), source_title=str(entity.get("title") or entity_id),
                            predicate_label="实体文本依据",
                        )
                        if item:
                            item["relation_key"] = f"entity:{entity_id}"
                            sources.append(item)
                            covered_text_units.add(str(text_unit_id))
                        break
                for source in sources:
                    key = (
                        str(source.get("relation_key") or ""),
                        str(source.get("text_unit_id") or ""),
                    )
                    if key not in seen_sources:
                        seen_sources.add(key)
                        report_sources.append(dict(source))
                findings.append({
                    "title": _native_report_prose(
                        finding.get("summary") or finding.get("title") or "结论"
                    ),
                    "explanation": _native_report_prose(explanation),
                    "entity_keys": entity_keys,
                    "relation_keys": relation_keys,
                    "sources": sources,
                })
            return {
                "report_id": str(report.get("id") or ""),
                "community_key": (
                    f"{int(report.get('level') or 0)}:{report.get('community')}"
                ),
                "status": "succeeded",
                "title": _native_report_prose(report.get("title")),
                "summary": _native_report_prose(report.get("summary")),
                "findings": findings,
                "limitations": "",
                "content": str(report.get("full_content") or ""),
                "rating": report.get("rank"),
                "rating_explanation": _native_report_prose(
                    report.get("rating_explanation")
                ),
                "sources": report_sources,
            }

        groups: list[dict[str, Any]] = []
        # Level zero is the most useful default partition for browsing.  Keep
        # hierarchical metadata for the detail panel without placing the same
        # entity into several simultaneous layout clusters.
        level_zero = [item for item in communities if int(item.get("level") or 0) == 0]
        selected_communities = level_zero or communities
        for display_index, item in enumerate(sorted(
            selected_communities,
            key=lambda value: (int(value.get("level") or 0), int(value.get("community") or 0)),
        )):
            level = int(item.get("level") or 0)
            community_value = (
                item.get("community") if item.get("community") is not None
                else item.get("id") if item.get("id") is not None
                else display_index
            )
            community = str(community_value)
            key = f"{level}:{community}"
            report = report_by_key.get((str(level), community), {})
            public_report = report_payload(report)
            entity_ids = [str(value) for value in item.get("entity_ids") or []]
            relation_ids = [str(value) for value in item.get("relationship_ids") or []]
            report_summary = _native_report_prose(report.get("summary"))
            evidence_sources = [
                dict(relation_source_by_id[value]) for value in relation_ids
                if value in relation_source_by_id
            ][:12]
            groups.append({
                "key": key,
                "label": str(report.get("title") or item.get("title") or f"社区 {display_index + 1}"),
                "summary": report_summary,
                "description": str(
                    report_summary
                    or "当前社区暂无自动生成的文字报告。"
                ),
                "description_source": (
                    "native_community_report"
                    if report.get("full_content") or report.get("summary")
                    else "native_community_fallback"
                ),
                "level": level,
                "community": community,
                "display_index": display_index,
                "size": int(item.get("size") or len(entity_ids)),
                "relationship_count": len(relation_ids),
                "entity_keys": entity_ids,
                "relationship_keys": relation_ids,
                "parent": item.get("parent"),
                "children": list(item.get("children") or []),
                "report_status": "succeeded" if public_report else "unavailable",
                "community_report": public_report,
                "evidence_sources": evidence_sources,
            })
            for entity_id in entity_ids:
                if entity_id in node_by_id and not node_by_id[entity_id].get("group"):
                    node_by_id[entity_id]["group"] = key
        for node in nodes:
            if not str(node.get("group") or ""):
                node["group"] = f"type:{node.get('type_id') or 'entity'}"

        return {
            "strategy": "graphrag_native",
            "run_id": str(run["run_id"]),
            "community_report_status": "completed",
            "total_entities": len(nodes),
            "total_relationships": len(edges),
            "total_communities": len(groups),
            "nodes": nodes,
            "edges": edges,
            "groups": groups,
        }

    def communities(self, project_id: int, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        run, root = self.current_run(project_id, validate_bundle=True)
        rows = self._records(root, "communities.parquet")
        offset = max(0, int(offset)); limit = min(500, max(1, int(limit)))
        return {"run_id": run["run_id"], "total": len(rows), "items": rows[offset:offset + limit]}

    def community_reports(self, project_id: int, *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        run, root = self.current_run(project_id, validate_bundle=True)
        rows = self._records(root, "community_reports.parquet")
        offset = max(0, int(offset)); limit = min(500, max(1, int(limit)))
        return {"run_id": run["run_id"], "total": len(rows), "items": rows[offset:offset + limit]}

    def artifact_catalog(self, project_id: int) -> dict[str, Any]:
        run, _ = self.current_run(project_id, validate_bundle=True)
        records = self.service.db.list_graphrag_native_artifacts(str(run["run_id"]))
        return {
            "run_id": str(run["run_id"]),
            "graphrag_version": str(run.get("graphrag_version") or ""),
            "artifacts": records,
            "viewable": [
                {"key": key, "logical_name": value}
                for key, value in self.VIEWABLE_ARTIFACTS.items()
            ],
        }

    def artifact_rows(
        self,
        project_id: int,
        artifact_key: str,
        *,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        run, root = self.current_run(project_id, validate_bundle=True)
        relative = self.VIEWABLE_ARTIFACTS.get(str(artifact_key))
        if not relative:
            raise ValueError("该 GraphRAG Native 产物不支持网页逐行查看")
        path = safe_bundle_path(root, relative)
        if not path.is_file() or path.is_symlink():
            raise GraphRAGNativeUnavailable(f"native artifact 缺少 {relative}")
        table = pq.read_table(path)
        total = int(table.num_rows)
        offset = max(0, int(offset))
        limit = min(500, max(1, int(limit)))
        rows = table.slice(offset, limit).to_pylist()
        return {
            "run_id": str(run["run_id"]),
            "artifact": str(artifact_key),
            "logical_name": relative,
            "columns": list(table.column_names),
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": rows,
        }

    def query(self, project_id: int, *, mode: str, question: str) -> dict[str, Any]:
        if mode not in QUERY_MODES or not str(question).strip():
            raise ValueError("native query 的 mode 或问题非法")
        # The worker performs a mode-specific immutable-bundle check.  In
        # particular Global must not trigger the full validator here because
        # that would open LanceDB before the worker can enforce its Parquet-only
        # contract.
        run, root = self.current_run(project_id, validate_bundle=False)
        query_run_id = "grnq_" + uuid.uuid4().hex
        question_hash = hashlib.sha256(str(question).encode("utf-8")).hexdigest()
        model_id = str(run.get("completion_model_id") or "")
        self.service.db.create_graphrag_native_query_run(
            query_run_id=query_run_id, project_id=project_id, run_id=str(run["run_id"]),
            query_mode=mode, question_hash=question_hash, model_id=model_id,
        )
        started = time.monotonic()
        payload = {
            "protocol_version": QUERY_PROTOCOL_VERSION, "bundle_root": str(root),
            "run_id": str(run["run_id"]), "mode": mode, "question": str(question),
        }
        project = self.service.db.get_project(project_id)
        payload["model_context_tokens"] = max(
            0, int((project or {}).get("model_max_tokens") or 0)
        )
        try:
            environment = dict(os.environ)
            # GraphRAG's OpenAI adapter validates a non-empty API key even
            # when the configured local provider does not require one.  The
            # placeholder is process-only and never persisted in a bundle.
            environment.setdefault("GRAPHRAG_API_KEY", "graphrag-local-placeholder")
            environment.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            completed = subprocess.run(
                [sys.executable, "-m", "knowledge_forge.graphrag_native_query_worker"],
                input=json.dumps(payload, ensure_ascii=False), text=True, capture_output=True,
                cwd=str(Path(__file__).resolve().parents[1]), env=environment,
                timeout=self.query_timeout_seconds, check=False,
            )
            if completed.returncode != 0:
                try:
                    worker_error = json.loads(completed.stdout)
                except (TypeError, json.JSONDecodeError):
                    worker_error = {}
                failure_class = str(
                    worker_error.get("failure_class") or "WorkerProcessError"
                )
                detail = str(worker_error.get("error") or "").strip()
                if detail:
                    detail = detail.replace(str(root), "当前知识包")
                    detail = " ".join(detail.split())[:500]
                    raise GraphRAGNativeUnavailable(
                        f"GraphRAG 查询失败（{failure_class}）：{detail}"
                    )
                raise GraphRAGNativeUnavailable(
                    f"GraphRAG 查询进程异常退出（{failure_class}）"
                )
            try:
                response = json.loads(completed.stdout)
            except (TypeError, json.JSONDecodeError) as exc:
                raise GraphRAGNativeUnavailable(
                    "GraphRAG 查询进程返回了无法解析的结果"
                ) from exc
            if not isinstance(response, dict) or response.get("protocol_version") != QUERY_PROTOCOL_VERSION or response.get("error"):
                raise GraphRAGNativeUnavailable(str(response.get("error") if isinstance(response, dict) else "native query worker 返回坏 JSON"))
            if str(response.get("run_id") or "") != str(run["run_id"]):
                raise GraphRAGNativeUnavailable("native query worker 返回跨 run 结果")
            self.service.db.complete_graphrag_native_query_run(
                query_run_id, status="READY", latency_ms=int((time.monotonic() - started) * 1000),
                context_reference_count=int(response.get("context_reference_count") or 0),
                token_usage=response.get("model_usage") if isinstance(response.get("model_usage"), dict) else {},
            )
            response["query_run_id"] = query_run_id
            return response
        except Exception as exc:
            self.service.db.complete_graphrag_native_query_run(
                query_run_id, status="FAILED", latency_ms=int((time.monotonic() - started) * 1000),
                failure_class=type(exc).__name__,
            )
            raise
