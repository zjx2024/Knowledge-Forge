"""GraphRAG Native query adapter over one immutable validated bundle."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pyarrow.parquet as pq

from ...construction_strategies import GRAPHRAG_NATIVE_STRATEGY
from ...graphrag_native_artifacts import safe_bundle_path
from ..contracts import EvidenceItem, RetrievalResult, SnapshotRef, SourceRef, canonical_json
from ..planning import RetrievalPlan


GLOBAL_PATTERNS = re.compile(
    r"总体|整体|全局|概览|综述|主要主题|共同趋势|跨(?:领域|文档|章节)|"
    r"\boverall\b|\bglobal\b|\boverview\b|\bmain themes?\b|\btrend",
    re.IGNORECASE,
)


def choose_native_mode(question: str, requested: str = "auto") -> str:
    requested = str(requested or "auto").strip().lower()
    if requested in {"basic", "local", "global"}:
        return requested
    if requested != "auto":
        raise ValueError("GraphRAG 问答方式只能是自动、Basic、Local 或 Global")
    return "global" if GLOBAL_PATTERNS.search(str(question or "")) else "local"


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item or "")]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item or "")]
        return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]
    return [str(value)]


def _record_content(table: str, record: dict[str, Any]) -> tuple[str, str]:
    title = str(
        record.get("title") or record.get("entity") or record.get("name")
        or record.get("community") or f"GraphRAG {table}"
    )
    fields = []
    for key in (
        "text", "description", "summary", "full_content", "content",
        "relationship", "findings",
    ):
        value = record.get(key)
        if value in (None, "", [], {}):
            continue
        fields.append(value if isinstance(value, str) else canonical_json(value))
    if not fields:
        fields.append(canonical_json({
            key: value for key, value in record.items()
            if key not in {"id", "rank", "weight"}
        }))
    return title, "\n".join(str(value) for value in fields)[:12_000]


class GraphRAGNativeRetriever:
    strategy = GRAPHRAG_NATIVE_STRATEGY

    def __init__(self, service: Any) -> None:
        self.service = service

    def _snapshot(self, project_id: int) -> tuple[SnapshotRef, dict[str, Any], Any]:
        run, root = self.service.graphrag_native_runtime.current_run(
            project_id, validate_bundle=False,
        )
        manifest_sha256 = str(run.get("manifest_sha256") or "")
        manifest_path = safe_bundle_path(root, "kf-manifest/run-manifest.json")
        if manifest_path.is_file() and not manifest_path.is_symlink():
            computed_manifest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if manifest_sha256 and manifest_sha256 != computed_manifest:
                raise RuntimeError("GraphRAG Native Bundle 清单摘要与控制记录不一致")
            manifest_sha256 = computed_manifest
        if not manifest_sha256:
            raise RuntimeError("GraphRAG Native Bundle 缺少可验证的运行清单摘要")
        source_hash = str(run.get("source_snapshot_hash") or "")
        snapshot = SnapshotRef(
            strategy=self.strategy,
            identifier=f"graphrag-native:{run['run_id']}",
            source_hash=source_hash,
            graph_available=True,
            details={
                "run_id": str(run["run_id"]),
                "graphrag_version": str(run.get("graphrag_version") or ""),
                "bundle_manifest_sha256": manifest_sha256,
            },
        )
        return snapshot, run, root

    @staticmethod
    def _source_rows_for_context(
        context: dict[str, Any], *, tables: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        direct = context.get("source_map")
        if isinstance(direct, dict):
            return [direct]
        table = str(context.get("table") or "")
        native_id = str(context.get("native_id") or "")
        raw_rows = tables.get(table) or []
        raw = next((row for row in raw_rows if str(row.get("id") or "") == native_id), None)
        text_unit_ids: set[str] = set()
        if raw:
            text_unit_ids.update(_list(raw.get("text_unit_ids")))
        if table in {"reports", "communities"}:
            community_id = str((raw or {}).get("community") or (context.get("record") or {}).get("id") or "")
            community = next(
                (row for row in tables.get("communities", []) if str(row.get("community") or "") == community_id),
                None,
            )
            if community:
                text_unit_ids.update(_list(community.get("text_unit_ids")))
                relation_ids = set(_list(community.get("relationship_ids")))
                for relation in tables.get("relationships", []):
                    if str(relation.get("id") or "") in relation_ids:
                        text_unit_ids.update(_list(relation.get("text_unit_ids")))
        input_document_ids: set[str] = set()
        for text_unit in tables.get("text_units", []):
            if str(text_unit.get("id") or "") not in text_unit_ids:
                continue
            input_document_ids.update(_list(text_unit.get("document_ids")))
            input_document_ids.update(_list(text_unit.get("document_id")))
        return [
            row for row in tables.get("source_map", [])
            if str(row.get("input_document_id") or "") in input_document_ids
        ]

    def retrieve(
        self, *, run_id: str, project: dict[str, Any], query: str,
        candidate_limit: int, requested_mode: str = "auto",
        plan: RetrievalPlan | None = None,
    ) -> RetrievalResult:
        project_id = int(project["id"])
        snapshot, native_run, root = self._snapshot(project_id)
        mode = (
            plan.native_mode
            if plan is not None and plan.activation == "active"
            else choose_native_mode(query, requested_mode)
        )
        response = self.service.graphrag_native_query(
            project_id, mode=mode, question=query,
        )
        if str(response.get("run_id") or "") != str(native_run["run_id"]):
            raise RuntimeError("GraphRAG 查询结果不属于当前 Native Bundle")
        run_manifest = str(snapshot.details.get("bundle_manifest_sha256") or "")
        response_manifest = str(response.get("bundle_manifest_sha256") or "")
        if run_manifest and response_manifest and run_manifest != response_manifest:
            raise RuntimeError("GraphRAG 查询结果与当前 Native Bundle 清单不一致")
        if response_manifest and not run_manifest:
            details = dict(snapshot.details)
            details["bundle_manifest_sha256"] = response_manifest
            snapshot = SnapshotRef(
                strategy=snapshot.strategy,
                identifier=snapshot.identifier,
                source_hash=snapshot.source_hash,
                generation=snapshot.generation,
                graph_available=True,
                details=details,
            )
        paths = {
            "entities": "output/entities.parquet",
            "relationships": "output/relationships.parquet",
            "communities": "output/communities.parquet",
            "reports": "output/community_reports.parquet",
            "text_units": "output/text_units.parquet",
            "source_map": "kf-manifest/source-map.parquet",
        }
        tables: dict[str, list[dict[str, Any]]] = {}
        for name, relative in paths.items():
            path = safe_bundle_path(root, relative)
            tables[name] = pq.read_table(path).to_pylist() if path.is_file() else []
        contexts = list(response.get("context_records") or [])[:max(1, candidate_limit)]
        all_source_rows: list[dict[str, Any]] = []
        source_rows_by_context: list[list[dict[str, Any]]] = []
        for context in contexts:
            rows = self._source_rows_for_context(context, tables=tables)
            source_rows_by_context.append(rows)
            all_source_rows.extend(rows)
        source_context = self.service.graphrag_native_runtime._native_source_context(
            project_id,
            list({str(row.get("input_document_id") or ""): row for row in all_source_rows}.values()),
        ) if all_source_rows else {}
        items: list[EvidenceItem] = []
        dropped_without_source = 0
        for index, (context, source_rows) in enumerate(zip(contexts, source_rows_by_context, strict=True)):
            record = context.get("record") if isinstance(context.get("record"), dict) else {}
            table = str(context.get("table") or "context")
            native_id = str(context.get("native_id") or context.get("context_id") or index)
            title, content = _record_content(table, record)
            refs: list[SourceRef] = []
            for row in source_rows:
                payload = source_context.get(str(row.get("input_document_id") or "")) or {
                    "document_id": row.get("document_id"),
                    "revision_id": row.get("revision_id"),
                    "pages": [row.get("page")] if row.get("page") else [],
                    "section_title": " / ".join(_list(row.get("heading_path"))),
                }
                payload = {**payload, "revision_id": str(row.get("revision_id") or payload.get("revision_id") or "")}
                ref = SourceRef.from_dict(payload, quote=content[:3000])
                if ref.valid:
                    refs.append(ref)
            refs = list({
                (ref.document_id, ref.revision_id, ref.pages, ref.section_title): ref
                for ref in refs
            }.values())
            if not refs:
                dropped_without_source += 1
                continue
            score = float(record.get("score") or record.get("weight") or 1.0 / (index + 1))
            items.append(EvidenceItem.create(
                run_id=run_id, evidence_type="native_context", project_id=project_id,
                strategy=self.strategy, snapshot_ref=snapshot,
                record_key=f"{table}:{native_id}", title=title, content=content,
                entity_keys=[native_id] if table == "entities" else [],
                relation_keys=[native_id] if table == "relationships" else [],
                source_refs=refs,
                retrieval_channels=[f"native_{mode}_{table}"],
                raw_scores={f"native_{mode}_{table}": max(0.0, score)},
                metadata={
                    "native_table": table,
                    "native_id": native_id,
                    "context_id": str(context.get("context_id") or ""),
                },
            ))
        return RetrievalResult(
            strategy=self.strategy, mode=mode, snapshot_ref=snapshot,
            evidence=items, candidate_answer=str(response.get("answer") or ""),
            diagnostics={
                "native_query_run_id": str(response.get("query_run_id") or ""),
                "context_count": len(contexts),
                "candidate_count": len(items),
                "dropped_without_source": dropped_without_source,
                "candidate_answer_retained_as_draft_only": True,
                "native_query_budget": dict(response.get("query_budget") or {}),
                "retrieval_plan_mode_applied": bool(
                    plan is not None and plan.activation == "active"
                ),
            },
        )

    def assert_current(self, project_id: int, snapshot: SnapshotRef) -> None:
        current, _run, _root = self._snapshot(project_id)
        if (
            current.identifier != snapshot.identifier
            or current.source_hash != snapshot.source_hash
            or str(current.details.get("bundle_manifest_sha256") or "")
            != str(snapshot.details.get("bundle_manifest_sha256") or "")
        ):
            raise RuntimeError("问答期间 GraphRAG Native Bundle 发生变化，未发布旧快照回答")
