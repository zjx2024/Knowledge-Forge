"""Explicit opt-in Global Search shadow over current community reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import asyncio
import hashlib
from importlib.metadata import version
import json
import time
from typing import Any, Protocol

import pandas as pd

from .database import Database


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class GlobalSearchUnavailable(RuntimeError):
    """A complete current report snapshot is not available (HTTP 409/503 equivalent)."""


@dataclass(frozen=True)
class GlobalSearchConfig:
    enabled: bool = False
    allow_general_knowledge: bool = False
    max_context_tokens: int = 12_000
    response_type: str = "Multiple Paragraphs"

    def validate(self) -> None:
        if self.allow_general_knowledge:
            raise ValueError("global shadow does not allow general knowledge")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens 必须大于 0")
        if not self.response_type.strip():
            raise ValueError("response_type 不能为空")


class GlobalSearchGateway(Protocol):
    model_id: str
    graphrag_version: str
    prompt_hash: str

    def search(
        self, query: str, reports: list[dict[str, Any]], config: GlobalSearchConfig,
    ) -> dict[str, Any]: ...


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class GraphRAGGlobalSearchGateway:
    """Adapter over ``graphrag.api.query.global_search`` from GraphRAG 3.1.1."""

    graphrag_version = version("graphrag")
    prompt_hash = "graphrag-configured-global-prompts"

    def __init__(self, *, model_id: str, graph_config, detection_snapshot: dict[str, Any]):
        self.model_id = model_id
        self.graph_config = graph_config
        self.detection_snapshot = detection_snapshot

    def _frames(
        self, reports: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        snapshot = self.detection_snapshot
        communities = list(snapshot.get("communities") or [])
        members = list(snapshot.get("members") or [])
        community_numbers = {
            str(item["stable_community_id"]): int(item["graphrag_community"])
            for item in communities
        }
        member_ids: dict[str, list[int]] = {}
        for item in members:
            member_ids.setdefault(str(item["stable_community_id"]), []).append(
                int(item["canonical_entity_id"])
            )
        entity_rows: dict[int, dict[str, Any]] = {}
        for index, item in enumerate(members):
            entity_id = int(item["canonical_entity_id"])
            entity_rows.setdefault(entity_id, {
                "id": entity_id,
                "human_readable_id": index,
                "title": str(item.get("entity_key") or item.get("name") or ""),
                "type": str(item.get("entity_type") or "Entity"),
                "description": str(item.get("description") or item.get("name") or ""),
                "degree": 0,
                "description_embedding": None,
                "text_unit_ids": list(item.get("text_unit_ids") or []),
            })
        community_rows = []
        for item in communities:
            stable_id = str(item["stable_community_id"])
            metadata = dict(item.get("metadata") or {})
            parent = item.get("parent_stable_community_id")
            children = [
                community_numbers[child] for child in metadata.get("children") or []
                if child in community_numbers
            ]
            community_rows.append({
                "id": stable_id,
                "community": community_numbers[stable_id],
                "level": int(item.get("level") or 0),
                "parent": community_numbers.get(str(parent), -1) if parent else -1,
                "children": children,
                "title": str(item.get("title") or stable_id),
                "entity_ids": sorted(member_ids.get(stable_id, [])),
                "relationship_ids": list(metadata.get("relationship_ids") or []),
                "text_unit_ids": list(metadata.get("text_unit_ids") or []),
                "period": item.get("period"),
                "size": int(item.get("size") or len(member_ids.get(stable_id, []))),
            })
        report_rows = []
        for item in reports:
            stable_id = str(item["community_id"])
            graph_community = community_numbers[stable_id]
            findings = list(item.get("findings") or [])
            report_rows.append({
                "id": str(item["report_id"]),
                "human_readable_id": graph_community,
                "community": graph_community,
                "level": int(item.get("level") or 0),
                "parent": -1,
                "children": [],
                "title": str(item.get("title") or stable_id),
                "summary": str(item.get("summary") or ""),
                "full_content": json.dumps({
                    "title": item.get("title"), "summary": item.get("summary"),
                    "findings": findings,
                }, ensure_ascii=False),
                "rank": 1.0,
                "rating_explanation": "Knowledge Forge snapshot-bound report",
                "findings": findings,
                "full_content_json": json.dumps(findings, ensure_ascii=False),
                "period": None,
                "size": len(member_ids.get(stable_id, [])),
            })
        return (
            pd.DataFrame(list(entity_rows.values())),
            pd.DataFrame(community_rows),
            pd.DataFrame(report_rows),
        )

    def search(
        self, query: str, reports: list[dict[str, Any]], config: GlobalSearchConfig,
    ) -> dict[str, Any]:
        from graphrag.api.query import global_search

        entities, communities, community_reports = self._frames(reports)
        global_config = self.graph_config.global_search.model_copy(update={
            "max_context_tokens": config.max_context_tokens,
        })
        graph_config = self.graph_config.model_copy(update={
            "global_search": global_config,
        })
        answer, context = asyncio.run(global_search(
            config=graph_config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=None,
            dynamic_community_selection=False,
            response_type=config.response_type,
            query=query,
            callbacks=None,
            verbose=False,
        ))
        report_ids = sorted(str(item["report_id"]) for item in reports)
        return {
            "answer": answer if isinstance(answer, str) else json.dumps(
                _json_safe(answer), ensure_ascii=False,
            ),
            "report_ids": report_ids,
            "citations": [{"report_id": report_id} for report_id in report_ids],
            "map_outputs": _json_safe(context),
            "reduce_output": _json_safe(answer),
            "usage": {},
        }


class GlobalSearchService:
    def __init__(self, repository, gateway: GlobalSearchGateway):
        self.repository = repository
        self.gateway = gateway

    def search(
        self, project_id: int, query: str, config: GlobalSearchConfig,
    ) -> dict[str, Any]:
        if not config.enabled:
            raise GlobalSearchUnavailable("global shadow requires explicit opt-in")
        config.validate()
        if not query.strip():
            raise ValueError("query 不能为空")
        active = self.repository.get_active_community_report_run(project_id)
        if not active:
            raise GlobalSearchUnavailable("no complete active community report run")
        snapshot = self.repository.show_community_report_run(active["run_id"])
        if not snapshot.get("current") or snapshot["run"].get("status") != "succeeded":
            raise GlobalSearchUnavailable("community report snapshot is stale or incomplete")
        reports = list(snapshot.get("reports") or [])
        if not reports:
            raise GlobalSearchUnavailable("active report snapshot is empty")
        allowed_report_ids = {str(item["report_id"]) for item in reports}
        query_hash = _digest(query.strip())
        report_input_hash = _digest([{
            "report_id": item["report_id"], "input_hash": item.get("input_hash"),
            "output_hash": item.get("output_hash"),
        } for item in reports])
        prompt_hash = str(getattr(self.gateway, "prompt_hash", "gateway-managed"))
        model_config_hash = _digest({
            "model_id": self.gateway.model_id, **asdict(config),
        })
        graphrag_version = str(getattr(
            self.gateway, "graphrag_version", version("graphrag"),
        ))
        run_id = "global-search:" + _digest([
            project_id, active["run_id"], query_hash, report_input_hash,
            self.gateway.model_id, asdict(config),
        ])[:40]
        run = self.repository.start_global_search_run({
            "run_id": run_id, "project_id": project_id,
            "report_run_id": active["run_id"], "query_hash": query_hash,
            "report_input_hash": report_input_hash, "prompt_hash": prompt_hash,
            "model_config_hash": model_config_hash, "model_id": self.gateway.model_id,
            "graphrag_version": graphrag_version,
            "config": asdict(config),
        })
        if run.get("status") in {"succeeded", "terminal_failed"}:
            return run
        started = time.perf_counter()
        try:
            result = self.gateway.search(query.strip(), reports, config)
            report_ids = {str(item) for item in result.get("report_ids") or []}
            citation_ids = {
                str(item.get("report_id")) for item in result.get("citations") or []
            }
            if not report_ids or report_ids.difference(allowed_report_ids):
                raise ValueError("global search selected unknown or no report IDs")
            if citation_ids.difference(report_ids) or not citation_ids:
                raise ValueError("global search citations do not resolve to selected reports")
            if not str(result.get("answer") or "").strip():
                raise ValueError("global search returned an empty answer")
            return self.repository.finalize_global_search_run(
                run_id, status="succeeded", result=result,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except (TimeoutError, ConnectionError) as exc:
            return self.repository.finalize_global_search_run(
                run_id, status="retryable_failed", result={}, error=str(exc)[:1000],
                error_category=type(exc).__name__,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        except Exception as exc:
            return self.repository.finalize_global_search_run(
                run_id, status="terminal_failed", result={}, error=str(exc)[:1000],
                error_category=type(exc).__name__,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit opt-in GraphRAG Global Search shadow")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="显式执行 global shadow 查询")
    search.add_argument("project_id", type=int)
    search.add_argument("query")
    search.add_argument("--graphrag-root", required=True)
    show = commands.add_parser("show", help="只读查看一个 global search run")
    show.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database = Database()
    database.bootstrap()
    if args.command == "show":
        print(json.dumps(
            database.show_global_search_run(args.run_id), ensure_ascii=False,
            default=str,
        ))
        return 0

    active = database.get_active_community_report_run(args.project_id)
    if not active:
        raise GlobalSearchUnavailable("no complete active community report run")
    detection = database.show_graphrag_community_run(active["detection_run_id"])
    from graphrag.config.load_config import load_config

    graph_config = load_config(args.graphrag_root)
    gateway = GraphRAGGlobalSearchGateway(
        model_id=graph_config.global_search.completion_model_id,
        graph_config=graph_config,
        detection_snapshot=detection,
    )
    result = GlobalSearchService(database, gateway).search(
        args.project_id, args.query, GlobalSearchConfig(enabled=True),
    )
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result.get("status") == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
