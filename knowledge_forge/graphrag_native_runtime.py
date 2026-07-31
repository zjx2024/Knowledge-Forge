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
        path = root / "output" / name
        if not path.is_file() or path.is_symlink():
            raise GraphRAGNativeUnavailable(f"native artifact 缺少 {name}")
        return pq.read_table(path).to_pylist()

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
