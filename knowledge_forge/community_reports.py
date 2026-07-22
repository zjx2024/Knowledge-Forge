"""Snapshot-bound GraphRAG community report shadow orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import asyncio
from datetime import date, datetime
from decimal import Decimal
import hashlib
from importlib.metadata import version
import json
import time
from typing import Any, Protocol

import pandas as pd

from .database import Database


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (set, tuple)):
        return sorted(value)
    raise TypeError(f"unsupported snapshot value: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=_json_default,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 429, 502, 503, 504}:
        return True
    name = type(exc).__name__.casefold().replace("_", "")
    return any(token in name for token in (
        "ratelimit", "serviceunavailable", "apiconnection", "temporarilyunavailable",
    ))


@dataclass(frozen=True)
class CommunityReportConfig:
    prompt: str
    max_input_tokens: int = 8_000
    max_report_tokens: int = 2_000
    max_attempts: int = 3

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("community report prompt 不能为空")
        if self.max_input_tokens < 1 or self.max_report_tokens < 1:
            raise ValueError("community report token budget 必须大于 0")
        if self.max_attempts < 1:
            raise ValueError("community report max_attempts 必须大于 0")


class CommunityReportGateway(Protocol):
    model_id: str
    graphrag_version: str

    def generate(
        self, context: dict[str, Any], config: CommunityReportConfig,
    ) -> dict[str, Any]: ...


class GraphRAGCommunityReportGateway:
    """Adapter over GraphRAG 3.1.1's supported report workflow function.

    GraphRAG's report schema does not carry Knowledge Forge entity/relation IDs.
    The adapter therefore attaches the exact input-community IDs to every finding.
    This is resolvable provenance, but it is deliberately not treated as human-Gold
    claim-level support by the release gate.
    """

    graphrag_version = version("graphrag")

    def __init__(self, *, model_id: str, model, tokenizer, callbacks=None):
        from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.callbacks = callbacks or NoopWorkflowCallbacks()

    @staticmethod
    def _frames(context: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        members = list(context["entities"])
        edges = list(context["relationships"])
        degrees = {int(item["canonical_entity_id"]): 0 for item in members}
        member_by_id = {int(item["canonical_entity_id"]): item for item in members}
        for edge in edges:
            degrees[int(edge["source_entity_id"])] += 1
            degrees[int(edge["target_entity_id"])] += 1
        entity_columns = [
            "id", "human_readable_id", "title", "type", "description", "degree",
            "text_unit_ids",
        ]
        entities = pd.DataFrame([{
            "id": int(item["canonical_entity_id"]),
            "human_readable_id": index,
            "title": str(item.get("entity_key") or item.get("name") or ""),
            "type": str(item.get("entity_type") or "Entity"),
            "description": str(item.get("description") or item.get("name") or ""),
            "degree": degrees[int(item["canonical_entity_id"])],
            "text_unit_ids": list(item.get("text_unit_ids") or []),
        } for index, item in enumerate(members)], columns=entity_columns)
        relationship_columns = [
            "id", "human_readable_id", "source", "target", "description", "weight",
            "combined_degree", "text_unit_ids",
        ]
        relationships = pd.DataFrame([{
            "id": str(item["stable_edge_id"]),
            "human_readable_id": index,
            "source": str(member_by_id[int(item["source_entity_id"])].get("entity_key")
                          or member_by_id[int(item["source_entity_id"])].get("name") or ""),
            "target": str(member_by_id[int(item["target_entity_id"])].get("entity_key")
                          or member_by_id[int(item["target_entity_id"])].get("name") or ""),
            "description": "; ".join(map(str, item.get("relation_types") or [])),
            "weight": float(item.get("weight") or 1.0),
            "combined_degree": (
                degrees[int(item["source_entity_id"])]
                + degrees[int(item["target_entity_id"])]
            ),
            "text_unit_ids": list(item.get("text_unit_ids") or []),
        } for index, item in enumerate(edges)], columns=relationship_columns)
        community = context["community"]
        communities = pd.DataFrame([{
            "id": str(community["stable_community_id"]),
            "community": 0,
            "level": int(community.get("level") or 0),
            "parent": -1,
            "children": [],
            "title": str(community.get("title") or community["stable_community_id"]),
            "entity_ids": sorted(member_by_id),
            "relationship_ids": sorted(str(item["stable_edge_id"]) for item in edges),
            "text_unit_ids": sorted({
                str(text_unit)
                for item in edges for text_unit in item.get("text_unit_ids") or []
            }),
            "period": None,
            "size": len(members),
        }])
        return relationships, entities, communities

    def generate(
        self, context: dict[str, Any], config: CommunityReportConfig,
    ) -> dict[str, Any]:
        from graphrag.config.enums import AsyncType
        from graphrag.index.workflows.create_community_reports import (
            create_community_reports,
        )

        relationships, entities, communities = self._frames(context)
        frame = asyncio.run(create_community_reports(
            relationships=relationships,
            entities=entities,
            communities=communities,
            claims_input=None,
            callbacks=self.callbacks,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=config.prompt,
            max_input_length=config.max_input_tokens,
            max_report_length=config.max_report_tokens,
            num_threads=1,
            async_type=AsyncType.AsyncIO,
        ))
        if frame.empty:
            raise RuntimeError("GraphRAG did not return a community report")
        row = frame.iloc[0].to_dict()
        findings = row.get("findings") or []
        if hasattr(findings, "tolist"):
            findings = findings.tolist()
        entity_ids = list(context["allowed_entity_ids"])
        relation_ids = list(context["allowed_relation_ids"])
        return {
            "title": str(row.get("title") or ""),
            "summary": str(row.get("summary") or ""),
            "findings": [{
                "summary": str(item.get("summary") or ""),
                "explanation": str(item.get("explanation") or ""),
                "entity_ids": entity_ids,
                "relation_ids": relation_ids,
            } for item in findings],
            "usage": {},
        }


def _community_context(snapshot: dict[str, Any], community: dict[str, Any]) -> dict[str, Any]:
    community_id = str(community["stable_community_id"])
    members = [
        dict(item) for item in snapshot.get("members") or []
        if str(item["stable_community_id"]) == community_id
    ]
    member_ids = {int(item["canonical_entity_id"]) for item in members}
    edges = [
        dict(item) for item in snapshot.get("edges") or []
        if int(item["source_entity_id"]) in member_ids
        and int(item["target_entity_id"]) in member_ids
    ]
    return {
        "community": dict(community),
        "entities": members,
        "relationships": edges,
        "allowed_entity_ids": sorted(member_ids),
        "allowed_relation_ids": sorted(str(item["stable_edge_id"]) for item in edges),
    }


def _validate_report(payload: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("community report 必须是 JSON object")
    title = str(payload.get("title") or "").strip()
    summary = str(payload.get("summary") or "").strip()
    findings = payload.get("findings")
    if not title or not summary or not isinstance(findings, list):
        raise ValueError("community report 缺少 title/summary/findings")
    allowed_entities = set(context["allowed_entity_ids"])
    allowed_relations = set(context["allowed_relation_ids"])
    entity_ids: set[int] = set()
    relation_ids: set[str] = set()
    normalized_findings: list[dict[str, Any]] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"community finding {index} 不是 object")
        current_entities = {int(item) for item in finding.get("entity_ids") or []}
        current_relations = {str(item) for item in finding.get("relation_ids") or []}
        if current_entities.difference(allowed_entities):
            raise ValueError("community report 引用了跨社区或不存在的 entity")
        if current_relations.difference(allowed_relations):
            raise ValueError("community report 引用了跨快照或不存在的 relation")
        if not current_entities and not current_relations:
            raise ValueError("community finding 缺少可解析 citation")
        entity_ids.update(current_entities)
        relation_ids.update(current_relations)
        normalized_findings.append({
            "summary": str(finding.get("summary") or "").strip(),
            "explanation": str(finding.get("explanation") or "").strip(),
            "entity_ids": sorted(current_entities),
            "relation_ids": sorted(current_relations),
        })
    return {
        "title": title,
        "summary": summary,
        "findings": normalized_findings,
        "entity_ids": sorted(entity_ids),
        "relation_ids": sorted(relation_ids),
        "usage": dict(payload.get("usage") or {}),
    }


class CommunityReportService:
    """Build reports in shadow and activate only one complete immutable run."""

    def __init__(self, repository, gateway: CommunityReportGateway):
        self.repository = repository
        self.gateway = gateway

    def run(self, detection_run_id: str, config: CommunityReportConfig) -> dict[str, Any]:
        config.validate()
        snapshot = self.repository.show_graphrag_community_run(detection_run_id)
        detection = snapshot["run"]
        if not snapshot.get("current") or detection.get("status") not in {
            "completed", "completed_empty",
        }:
            raise RuntimeError("community detection snapshot stale or incomplete")
        prompt_hash = _digest(config.prompt)
        model_config_hash = _digest({"model_id": self.gateway.model_id, **asdict(config)})
        graphrag_version = str(getattr(
            self.gateway, "graphrag_version", version("graphrag"),
        ))
        input_hash = _digest({
            "detection_run_id": detection_run_id,
            "community_input_hash": detection["input_hash"],
            "communities": snapshot.get("communities") or [],
        })
        run_id = "report:" + _digest([
            detection["project_id"], detection_run_id, input_hash,
            prompt_hash, model_config_hash,
        ])[:40]
        run = self.repository.start_community_report_run({
            "run_id": run_id,
            "project_id": int(detection["project_id"]),
            "graph_generation": int(detection["graph_generation"]),
            "detection_run_id": detection_run_id,
            "community_input_hash": detection["input_hash"],
            "input_hash": input_hash,
            "prompt_hash": prompt_hash,
            "model_config_hash": model_config_hash,
            "model_id": self.gateway.model_id,
            "graphrag_version": graphrag_version,
            "config": asdict(config),
            "expected": len(snapshot.get("communities") or []),
        })
        persisted = {
            str(item["community_id"]): item
            for item in self.repository.list_community_report_items(run_id)
        }
        completed = sum(item.get("status") == "succeeded" for item in persisted.values())
        failures: list[str] = []
        for community in snapshot.get("communities") or []:
            community_id = str(community["stable_community_id"])
            previous = persisted.get(community_id)
            if previous and previous.get("status") == "succeeded":
                continue
            if previous and previous.get("status") == "terminal_failed":
                failures.append(
                    f"{community_id}:terminal:"
                    f"{previous.get('error_category') or 'terminal_failed'}"
                )
                continue
            context = _community_context(snapshot, community)
            attempt = int((previous or {}).get("attempt") or 0) + 1
            if attempt > config.max_attempts:
                failures.append(f"{community_id}:terminal:max_attempts")
                self.repository.upsert_community_report_item(run_id, {
                    "report_id": (
                        (previous or {}).get("report_id")
                        or "community-report:" + _digest([run_id, community_id])[:40]
                    ),
                    "community_id": community_id,
                    "level": int(community["level"]),
                    "attempt": int((previous or {}).get("attempt") or config.max_attempts),
                    "input_hash": (
                        (previous or {}).get("input_hash") or _digest(context)
                    ),
                    "status": "terminal_failed",
                    "error_category": "max_attempts",
                    "error": "community report retry budget exhausted",
                })
                continue
            base_item = {
                "report_id": "community-report:" + _digest([run_id, community_id])[:40],
                "community_id": community_id,
                "level": int(community["level"]),
                "attempt": attempt,
                "input_hash": _digest(context),
            }
            self.repository.upsert_community_report_item(run_id, {
                **base_item, "status": "running",
            })
            started = time.perf_counter()
            try:
                payload = _validate_report(self.gateway.generate(context, config), context)
                item = {
                    **base_item,
                    "status": "succeeded",
                    "output_hash": _digest(payload),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    **payload,
                }
                self.repository.upsert_community_report_item(run_id, item)
                completed += 1
            except Exception as exc:
                retryable = _is_retryable_error(exc)
                terminal = "retryable" if retryable else "terminal"
                failures.append(f"{community_id}:{terminal}:{type(exc).__name__}")
                self.repository.upsert_community_report_item(run_id, {
                    **base_item,
                    "status": "retryable_failed" if retryable else "terminal_failed",
                    "error_category": type(exc).__name__, "error": str(exc)[:1000],
                })
        expected = len(snapshot.get("communities") or [])
        status = "succeeded" if completed == expected and not failures else (
            "retryable_failed" if failures and all(":retryable:" in item for item in failures)
            else "terminal_failed"
        )
        return self.repository.finalize_community_report_run(
            run_id,
            status=status,
            metrics={"expected": expected, "completed": completed, "failed": len(failures)},
            error="; ".join(failures),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG community reports shadow")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="显式生成一个 detection snapshot 的社区报告")
    run.add_argument("detection_run_id")
    run.add_argument("--graphrag-root", required=True)
    show = commands.add_parser("show", help="只读查看一个 community report run")
    show.add_argument("run_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database = Database()
    database.bootstrap()
    if args.command == "show":
        print(_json(database.show_community_report_run(args.run_id)))
        return 0

    from graphrag.config.load_config import load_config
    from graphrag_llm.completion import create_completion

    graph_config = load_config(args.graphrag_root)
    report_config = graph_config.community_reports
    model_config = graph_config.get_completion_model_config(
        report_config.completion_model_id,
    )
    model = create_completion(model_config)
    gateway = GraphRAGCommunityReportGateway(
        model_id=report_config.completion_model_id,
        model=model,
        tokenizer=model.tokenizer,
    )
    result = CommunityReportService(database, gateway).run(
        args.detection_run_id,
        CommunityReportConfig(
            prompt=report_config.resolved_prompts().graph_prompt,
            max_input_tokens=report_config.max_input_length,
            max_report_tokens=report_config.max_length,
        ),
    )
    print(_json(result))
    return 0 if result.get("status") == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
