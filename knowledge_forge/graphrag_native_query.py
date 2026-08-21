"""Version-locked, file-only query adapter for GraphRAG Native bundles.

This module intentionally has no Database, pgvector, HugeGraph, stable, or
Knowledge Forge extraction dependency. Its input is a validated immutable native bundle
and its output follows the small JSON-safe protocol consumed by the service.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from .graphrag_native_artifacts import GraphRAGNativeBundleValidator, safe_bundle_path


QUERY_PROTOCOL_VERSION = "graphrag-native-query-v1"
QUERY_MODES = frozenset({"local", "global", "drift", "basic"})


class GraphRAGNativeQueryError(RuntimeError):
    """A bundle/query protocol violation that must never fall back."""


@dataclass(frozen=True)
class NativeQueryRequest:
    run_id: str
    mode: str
    question: str
    response_type: str = "Multiple Paragraphs"
    community_level: int = 2


def _table(root: Path, name: str) -> pd.DataFrame:
    path = root / "output" / name
    if not path.is_file() or path.is_symlink():
        raise GraphRAGNativeQueryError(f"native query 缺少安全表: {name}")
    return pq.read_table(path).to_pandas()


def assert_graphrag_query_api_contract() -> None:
    """Reject a GraphRAG upgrade before a query can silently change meaning."""

    if version("graphrag") != "3.1.1":
        raise GraphRAGNativeQueryError("native query 只接受 graphrag==3.1.1")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from graphrag import api

    expected = {
        "local_search": ("config", "entities", "communities", "community_reports", "text_units", "relationships", "covariates", "community_level", "response_type", "query", "callbacks", "verbose"),
        "global_search": ("config", "entities", "communities", "community_reports", "community_level", "dynamic_community_selection", "response_type", "query", "callbacks", "verbose"),
        "drift_search": ("config", "entities", "communities", "community_reports", "text_units", "relationships", "community_level", "response_type", "query", "callbacks", "verbose"),
        "basic_search": ("config", "text_units", "response_type", "query", "callbacks", "verbose"),
    }
    for name, fields in expected.items():
        function = getattr(api, name, None)
        if function is None or tuple(inspect.signature(function).parameters) != fields:
            raise GraphRAGNativeQueryError(f"GraphRAG 3.1.1 query API 契约不匹配: {name}")


def _rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).to_dict("records")
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _one(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], what: str) -> dict[str, Any]:
    matching = [row for row in rows if predicate(row)]
    if len(matching) != 1:
        raise GraphRAGNativeQueryError(f"native context {what} 必须恰好映射一条，实际 {len(matching)} 条")
    return matching[0]


def _context_id(value: Any) -> str:
    """Preserve GraphRAG's valid zero-valued human-readable identifiers."""

    return "" if value is None else str(value)


def _source_reference(text_unit: dict[str, Any], source_map: list[dict[str, Any]]) -> dict[str, Any]:
    document_id = str(text_unit.get("document_id") or "")
    entries = [row for row in source_map if str(row.get("input_document_id") or "") == document_id]
    if len(entries) != 1:
        raise GraphRAGNativeQueryError("TextUnit 无法唯一映射到同 run source map")
    row = entries[0]
    return {
        "input_document_id": document_id,
        "project_id": row.get("project_id"), "document_id": row.get("document_id"),
        "revision_id": row.get("revision_id"), "block_ids": row.get("block_ids"),
        "page": row.get("page"), "heading_path": row.get("heading_path"),
    }


def serialize_context(
    context_data: Any,
    *,
    entities: pd.DataFrame | None,
    relationships: pd.DataFrame | None,
    communities: pd.DataFrame | None,
    reports: pd.DataFrame | None,
    text_units: pd.DataFrame | None,
    source_map: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Resolve GraphRAG short IDs to one immutable native ID each.

    3.1.1 emits short IDs in callback context.  This serializer preserves that
    callback ID and independently resolves it through the documented adapter
    rules; it never guesses from titles or accepts a partial match.
    """

    tables = {
        "entities": _rows(entities), "relationships": _rows(relationships),
        "communities": _rows(communities), "reports": _rows(reports),
        "text_units": _rows(text_units), "source_map": _rows(source_map),
    }
    if not isinstance(context_data, dict):
        raise GraphRAGNativeQueryError("GraphRAG query context 不是受支持的表字典")
    result: list[dict[str, Any]] = []
    aliases = {
        "entities": "entities", "relationships": "relationships",
        "communities": "communities", "reports": "reports",
        "community_reports": "reports", "sources": "text_units",
        "text_units": "text_units", "claims": "claims",
    }
    for original_table, value in context_data.items():
        table = aliases.get(str(original_table).lower())
        if not table:
            raise GraphRAGNativeQueryError(f"未知 GraphRAG context table: {original_table}")
        records = _rows(value)
        if table == "claims":
            if records:
                raise GraphRAGNativeQueryError(
                    "当前 native bundle 已禁用 claims，但查询返回了非空 claims"
                )
            continue
        for record in records:
            context_id = _context_id(record.get("id"))
            if not context_id:
                raise GraphRAGNativeQueryError(f"{original_table} context 缺少 id")
            source_ref: dict[str, Any] | None = None
            if table in {"entities", "relationships"}:
                native = _one(
                    tables[table],
                    lambda row: _context_id(row.get("human_readable_id")) == context_id,
                    table,
                )
            elif table in {"communities", "reports"}:
                native = _one(
                    tables[table],
                    lambda row: _context_id(row.get("community")) == context_id,
                    table,
                )
            else:
                # ``read_text_units`` in GraphRAG 3.1.1 resets the final
                # DataFrame index and calls that zero-based position short_id.
                try:
                    position = int(context_id)
                except ValueError as exc:
                    raise GraphRAGNativeQueryError("TextUnit context id 不是 3.1.1 reset-index 位置") from exc
                if position < 0 or position >= len(tables["text_units"]):
                    raise GraphRAGNativeQueryError("TextUnit context reset-index 位置越界")
                native = tables["text_units"][position]
                if not _context_id(native.get("id")) or not _context_id(
                    native.get("human_readable_id")
                ):
                    raise GraphRAGNativeQueryError("TextUnit position 与 final human-readable/native ID 契约不完整")
                source_ref = _source_reference(native, tables["source_map"])
            result.append({
                "table": table, "context_id": context_id,
                "native_id": str(native.get("id") or ""),
                "record": record, "source_map": source_ref,
            })
    return result


def _answer_text(answer: Any) -> str:
    if isinstance(answer, str):
        return answer
    return json.dumps(answer, ensure_ascii=False, sort_keys=True)


def assert_query_response_clean(*, mode: str, answer: Any, context_data: Any) -> None:
    """Fail closed on GraphRAG 3.1.1's known swallowed search failures.

    Public search APIs can turn an internal map/reduce/action exception into a
    superficially successful empty response.  The adapter has no separate
    proof of a legal DRIFT termination boundary, therefore an emitted
    ``No follow-up actions found`` is treated as protocol failure rather than
    guessed to be a valid completion.  A normal no-data answer that does not
    contain these internal markers remains a valid, explicit answer.
    """

    rendered = "\n".join((
        _answer_text(answer),
        json.dumps(context_data, ensure_ascii=False, sort_keys=True, default=str),
    )).lower()
    markers = (
        "exception in _map_response_single_batch",
        "error parsing search response json",
        "exception in reduce_response",
        "exception in _asearch",
        "no answer found for query",
    )
    if any(marker in rendered for marker in markers):
        raise GraphRAGNativeQueryError("GraphRAG query 返回静默内部失败标记")
    if mode == "drift" and "no follow-up actions found" in rendered:
        raise GraphRAGNativeQueryError("DRIFT 未证明合法终止边界，拒绝 No follow-up actions found")


def _bundle_prompt_path(root: Path, configured: str | None) -> str | None:
    """Resolve one frozen query prompt inside the published bundle.

    Older bundles persisted paths relative to their staging cwd, while newer
    bundles are intentionally relocatable.  Query execution must use the
    current immutable bundle in both cases and must never read an arbitrary
    host path left in ``settings.yaml``.
    """

    if not configured:
        return None
    path = Path(str(configured))
    parts = path.parts
    try:
        prompt_index = parts.index("prompts")
    except ValueError as exc:
        raise GraphRAGNativeQueryError(
            f"GraphRAG query prompt 不属于 bundle prompts: {path.name}"
        ) from exc
    relative = Path(*parts[prompt_index:])
    resolved = safe_bundle_path(root, relative.as_posix())
    if not resolved.is_file() or resolved.is_symlink():
        raise GraphRAGNativeQueryError(f"GraphRAG query prompt 缺失: {relative.as_posix()}")
    return str(resolved)


def load_bundle_query_config(root: Path) -> Any:
    """Load 3.1.1 settings and rebind every query dependency to this bundle."""

    from graphrag.config.load_config import load_config

    root = Path(root).resolve()
    previous_cwd = Path.cwd()
    try:
        # GraphRAG resolves bundle-relative paths against cwd and its 3.1.1
        # loader may itself leave cwd changed. Query workers are long-lived,
        # so contain that global side effect exactly as the build validator
        # does.
        os.chdir(root)
        config = load_config(
            root,
            cli_overrides={
                "input_storage": {
                    "base_dir": str(safe_bundle_path(root, "input")),
                },
                "output_storage": {
                    "base_dir": str(safe_bundle_path(root, "output")),
                },
                "update_output_storage": {
                    "base_dir": str(safe_bundle_path(root, "output-update")),
                },
                "reporting": {
                    "base_dir": str(safe_bundle_path(root, "logs")),
                },
                "cache": {
                    "storage": {
                        "base_dir": str(safe_bundle_path(root, "cache")),
                    },
                },
                "table_provider": {
                    "storage": {
                        "base_dir": str(safe_bundle_path(root, "output")),
                    },
                },
                "vector_store": {
                    "db_uri": str(safe_bundle_path(root, "output/lancedb")),
                },
            },
        )
    finally:
        os.chdir(previous_cwd)
    prompt_fields = {
        "local_search": ("prompt",),
        "global_search": ("map_prompt", "reduce_prompt", "knowledge_prompt"),
        "drift_search": ("prompt", "reduce_prompt"),
        "basic_search": ("prompt",),
    }
    for section_name, fields in prompt_fields.items():
        section = getattr(config, section_name)
        for field in fields:
            value = getattr(section, field, None)
            if value:
                setattr(section, field, _bundle_prompt_path(root, str(value)))
    return config


async def _call_api(request: NativeQueryRequest, root: Path) -> tuple[Any, Any]:
    from graphrag.api import basic_search, drift_search, global_search, local_search

    config = load_bundle_query_config(root)
    params = {"config": config, "response_type": request.response_type, "query": request.question}
    if request.mode == "global":
        # GraphRAG Global is deliberately Parquet-only.  Do not even load
        # TextUnits/relationships here: that makes a future adapter regression
        # observable and guarantees that this path has no vector-store need.
        entities = _table(root, "entities.parquet")
        communities = _table(root, "communities.parquet")
        reports = _table(root, "community_reports.parquet")
        return await global_search(
            **params, entities=entities, communities=communities, community_reports=reports,
            community_level=request.community_level, dynamic_community_selection=False,
        )
    if request.mode == "local":
        entities = _table(root, "entities.parquet")
        communities = _table(root, "communities.parquet")
        reports = _table(root, "community_reports.parquet")
        text_units = _table(root, "text_units.parquet")
        relationships = _table(root, "relationships.parquet")
        return await local_search(
            **params, entities=entities, communities=communities, community_reports=reports,
            text_units=text_units, relationships=relationships, covariates=None,
            community_level=request.community_level,
        )
    if request.mode == "drift":
        entities = _table(root, "entities.parquet")
        communities = _table(root, "communities.parquet")
        reports = _table(root, "community_reports.parquet")
        text_units = _table(root, "text_units.parquet")
        relationships = _table(root, "relationships.parquet")
        return await drift_search(
            **params, entities=entities, communities=communities, community_reports=reports,
            text_units=text_units, relationships=relationships, community_level=request.community_level,
        )
    if request.mode == "basic":
        text_units = _table(root, "text_units.parquet")
        return await basic_search(**params, text_units=text_units)
    raise GraphRAGNativeQueryError("不支持的 native query mode")


def query_bundle(request: NativeQueryRequest, root: Path) -> dict[str, Any]:
    """Run one isolated 3.1.1 query against a validated bundle."""

    if request.mode not in QUERY_MODES or not str(request.question).strip():
        raise GraphRAGNativeQueryError("native query mode 或问题非法")
    root = Path(root).resolve()
    manifest_path = safe_bundle_path(root, "kf-manifest/run-manifest.json")
    # Global has a deliberately narrow data contract: it rechecks exactly its
    # three native Parquet inputs and their manifest hashes, rather than
    # opening LanceDB or reading Local/DRIFT-only tables.
    validator = GraphRAGNativeBundleValidator()
    if request.mode == "global":
        validator.validate_global_query_subset(root)
    else:
        validator.validate(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_run_id = str((manifest.get("identity") or {}).get("run_id") or "")
    if manifest_run_id != request.run_id:
        raise GraphRAGNativeQueryError("query run_id 与 immutable bundle 不一致")
    assert_graphrag_query_api_contract()
    started = time.monotonic()
    answer, context_data = asyncio.run(_call_api(request, root))
    assert_query_response_clean(mode=request.mode, answer=answer, context_data=context_data)
    answer_text = _answer_text(answer)
    warnings = []
    # Only materialize tables that a mode is permitted to use.  Context
    # serialization remains fail-closed if GraphRAG unexpectedly returns a
    # table outside that mode's contract.
    if request.mode == "global":
        context_records = serialize_context(
            context_data, entities=_table(root, "entities.parquet"), relationships=None,
            communities=_table(root, "communities.parquet"), reports=_table(root, "community_reports.parquet"),
            text_units=None, source_map=None,
        )
    elif request.mode == "basic":
        context_records = serialize_context(
            context_data, entities=None, relationships=None, communities=None, reports=None,
            text_units=_table(root, "text_units.parquet"),
            source_map=pq.read_table(root / "kf-manifest" / "source-map.parquet").to_pandas(),
        )
    else:
        context_records = serialize_context(
            context_data, entities=_table(root, "entities.parquet"),
            relationships=_table(root, "relationships.parquet"), communities=_table(root, "communities.parquet"),
            reports=_table(root, "community_reports.parquet"), text_units=_table(root, "text_units.parquet"),
            source_map=pq.read_table(root / "kf-manifest" / "source-map.parquet").to_pandas(),
        )
    return {
        "protocol_version": QUERY_PROTOCOL_VERSION, "answer": answer_text,
        "mode": request.mode, "run_id": request.run_id, "context_records": context_records,
        "context_reference_count": len(context_records), "answer_references": [],
        "model_usage": {"status": "unavailable"},
        "latency_ms": int((time.monotonic() - started) * 1000), "warnings": warnings,
        "bundle_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
