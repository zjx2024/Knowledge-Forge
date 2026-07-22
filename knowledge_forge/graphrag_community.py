"""Shadow GraphRAG BYOG community detection over the publishable control graph."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable
import uuid

import pandas as pd

from .database import Database


ADAPTER_VERSION = "knowledge-forge-byog-v1"
WEIGHTING_STRATEGY = "publishable_assertion_count_v1"
DESCRIPTION_LIMIT = 4000
REQUIRED_COMMUNITY_COLUMNS = frozenset({
    "id", "community", "level", "parent", "title", "entity_ids",
    "relationship_ids", "text_unit_ids", "period", "size",
})


@dataclass(frozen=True)
class CommunityConfig:
    seed: int = 42
    max_cluster_size: int = 10
    use_lcc: bool = False
    timeout_seconds: int = 300

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "max_cluster_size": int(self.max_cluster_size),
            "use_lcc": bool(self.use_lcc),
            "timeout_seconds": int(self.timeout_seconds),
            "weighting_strategy": WEIGHTING_STRATEGY,
            "adapter_version": ADAPTER_VERSION,
        }

    def validate(self) -> None:
        if self.max_cluster_size < 1:
            raise ValueError("max_cluster_size 必须大于 0")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds 必须大于 0")
        if self.use_lcc:
            raise ValueError("社区检测必须保持 use_lcc=false 以覆盖所有非连通分量")


@dataclass(frozen=True)
class ControlGraphSnapshot:
    project_id: int
    graph_generation: int
    graph_policy_hash: str
    active_policy_hash: str
    relations: tuple[dict[str, Any], ...]
    entities: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class BYOGSnapshot:
    project_id: int
    input_hash: str
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    metrics: dict[str, int]


@dataclass(frozen=True)
class GraphRAGResult:
    communities: list[dict[str, Any]]
    log: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_edge_id(project_id: int, source_id: int, target_id: int) -> str:
    return "kf_edge:" + _digest([project_id, source_id, target_id])[:32]


def stable_community_id(
    project_id: int,
    input_hash: str,
    level: int,
    canonical_entity_ids: Iterable[int],
) -> str:
    """Return the business identifier independent of GraphRAG UUIDs/integers."""

    members = sorted({int(item) for item in canonical_entity_ids})
    return "kf_community:" + _digest([int(project_id), input_hash, int(level), members])[:40]


def _text_unit_id(row: dict[str, Any]) -> str:
    revision_id = str(row.get("revision_id") or "")
    chunk_key = str(row.get("chunk_key") or "")
    if not revision_id or not chunk_key:
        raise ValueError("正式关系缺少 active revision 或 chunk_key")
    return f"chunk:{revision_id}:{chunk_key}"


def _description(entity: dict[str, Any]) -> str:
    parts = [
        f"Name: {entity['name']}",
        f"Type: {entity['entity_type']}",
    ]
    if entity.get("description"):
        parts.append(str(entity["description"]))
    return "\n".join(parts)[:DESCRIPTION_LIMIT]


def build_byog_snapshot(
    snapshot: ControlGraphSnapshot,
    config: CommunityConfig,
) -> BYOGSnapshot:
    """Normalize one immutable control-graph snapshot into GraphRAG BYOG tables."""

    config.validate()
    entity_map: dict[int, dict[str, Any]] = {}
    edge_groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    excluded_self_loops = 0

    for row in sorted(snapshot.entities, key=lambda item: int(item["id"])):
        entity_map[int(row["id"])] = {
            "id": int(row["id"]),
            "entity_key": str(row["entity_key"]),
            "name": str(row["name"]),
            "entity_type": str(row["entity_type"]),
            "description": str(row.get("description") or ""),
            "text_unit_ids": sorted(map(str, row.get("text_unit_ids") or [])),
        }

    for row in sorted(snapshot.relations, key=lambda item: int(item["assertion_id"])):
        source_id = int(row["source_entity_id"])
        target_id = int(row["target_entity_id"])
        for prefix, entity_id in (("source", source_id), ("target", target_id)):
            candidate = {
                "id": entity_id,
                "entity_key": str(row[f"{prefix}_entity_key"]),
                "name": str(row[f"{prefix}_name"]),
                "entity_type": str(row[f"{prefix}_type"]),
                "description": str(row.get(f"{prefix}_description") or ""),
            }
            existing = entity_map.get(entity_id)
            if existing is not None and any(
                existing.get(key) != candidate.get(key)
                for key in ("id", "entity_key", "name", "entity_type", "description")
            ):
                raise ValueError(f"实体 {entity_id} 在同一快照中信息不一致")
            if existing is None:
                entity_map[entity_id] = candidate
        if source_id == target_id:
            excluded_self_loops += 1
            continue
        endpoint = tuple(sorted((source_id, target_id)))
        edge_groups.setdefault(endpoint, []).append(row)

    referenced_ids = {entity_id for endpoint in edge_groups for entity_id in endpoint}
    isolated_entity_ids = set(entity_map).difference(referenced_ids)
    title_values = [item["entity_key"] for item in entity_map.values()]
    if len(title_values) != len(set(title_values)):
        raise ValueError("entity_key 在项目内不唯一，不能作为 GraphRAG title")

    degrees = {entity_id: 0 for entity_id in entity_map}
    assertion_frequency = {entity_id: 0 for entity_id in entity_map}
    for (source_id, target_id), rows in edge_groups.items():
        degrees[source_id] += 1
        degrees[target_id] += 1
        assertion_frequency[source_id] += len(rows)
        assertion_frequency[target_id] += len(rows)

    incident_text_units: dict[int, set[str]] = {
        entity_id: set(map(str, entity.get("text_unit_ids") or []))
        for entity_id, entity in entity_map.items()
    }
    edges: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for human_id, ((source_id, target_id), rows) in enumerate(sorted(edge_groups.items())):
        text_unit_ids = sorted({_text_unit_id(row) for row in rows})
        assertion_ids = sorted({int(row["assertion_id"]) for row in rows})
        chunk_ids = sorted({int(row["chunk_id"]) for row in rows})
        relation_types = sorted({str(row["relation_type"]) for row in rows})
        descriptions = sorted({
            str(row.get("relation_description") or "").strip()
            for row in rows if str(row.get("relation_description") or "").strip()
        })
        description = "; ".join([
            "Types: " + ", ".join(relation_types),
            *descriptions,
        ])[:DESCRIPTION_LIMIT]
        weight = float(len(assertion_ids))
        stable_edge = _stable_edge_id(snapshot.project_id, source_id, target_id)
        provenance = {
            "assertion_ids": assertion_ids,
            "chunk_ids": chunk_ids,
            "relation_types": relation_types,
            "text_unit_ids": text_unit_ids,
        }
        edge = {
            "stable_edge_id": stable_edge,
            "source_entity_id": source_id,
            "target_entity_id": target_id,
            "weight": weight,
            **provenance,
            "provenance_hash": _digest(provenance),
        }
        edges.append(edge)
        relationships.append({
            "id": stable_edge,
            "human_readable_id": human_id,
            "source": entity_map[source_id]["entity_key"],
            "target": entity_map[target_id]["entity_key"],
            "description": description,
            "weight": weight,
            "combined_degree": degrees[source_id] + degrees[target_id],
            "text_unit_ids": text_unit_ids,
        })
        incident_text_units[source_id].update(text_unit_ids)
        incident_text_units[target_id].update(text_unit_ids)

    entities: list[dict[str, Any]] = []
    for human_id, (entity_id, entity) in enumerate(sorted(entity_map.items())):
        entities.append({
            "id": f"kf_entity:{entity_id}",
            "human_readable_id": human_id,
            "title": entity["entity_key"],
            "type": entity["entity_type"],
            "description": _description(entity),
            "text_unit_ids": sorted(incident_text_units[entity_id]),
            "frequency": assertion_frequency[entity_id],
            "degree": degrees[entity_id],
        })

    metrics = {
        "node_count": len(entities),
        "aggregated_edge_count": len(relationships),
        "excluded_self_loops": excluded_self_loops,
        "excluded_isolated_entities": 0,
        "isolated_entity_count": len(isolated_entity_ids),
    }
    hash_payload = {
        "project_id": snapshot.project_id,
        "graph_generation": snapshot.graph_generation,
        "graph_policy_hash": snapshot.graph_policy_hash,
        "active_policy_hash": snapshot.active_policy_hash,
        "config": config.as_dict(),
        "entities": entities,
        "relationships": relationships,
        "edges": edges,
        "metrics": metrics,
    }
    result = BYOGSnapshot(
        project_id=snapshot.project_id,
        input_hash=_digest(hash_payload),
        entities=entities,
        relationships=relationships,
        edges=edges,
        metrics=metrics,
    )
    validate_byog_snapshot(result)
    return result


def validate_byog_snapshot(snapshot: BYOGSnapshot) -> None:
    entity_titles = [str(item["title"]) for item in snapshot.entities]
    if len(entity_titles) != len(set(entity_titles)):
        raise ValueError("GraphRAG entity title 必须唯一")
    titles = set(entity_titles)
    for relationship in snapshot.relationships:
        if relationship.get("source") not in titles or relationship.get("target") not in titles:
            raise ValueError("GraphRAG 关系存在悬空端点")
        weight = float(relationship.get("weight", 0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("GraphRAG 关系权重必须为有限正数")
        if relationship.get("source") == relationship.get("target"):
            raise ValueError("GraphRAG 输入不得包含自环")


def _list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


class GraphRAGCommunityRunner:
    """Execute the supported GraphRAG CLI/Parquet BYOG boundary."""

    def __init__(self, workspace_root: Path | str | None = None):
        self.workspace_root = Path(workspace_root) if workspace_root else None

    @staticmethod
    def _command(workspace: Path) -> list[str]:
        executable = Path(sys.executable).with_name("graphrag")
        if not executable.is_file():
            raise RuntimeError("当前虚拟环境缺少 graphrag CLI")
        return [
            str(executable), "index", "--root", str(workspace),
            "--skip-validation", "--no-cache",
        ]

    @staticmethod
    def _settings(config: CommunityConfig) -> str:
        use_lcc = "true" if config.use_lcc else "false"
        return (
            "workflows:\n"
            "  - create_communities\n"
            "output_storage:\n"
            "  type: file\n"
            "  base_dir: output\n"
            "table_provider:\n"
            "  type: parquet\n"
            "cluster_graph:\n"
            f"  max_cluster_size: {config.max_cluster_size}\n"
            f"  use_lcc: {use_lcc}\n"
            f"  seed: {config.seed}\n"
        )

    def run(self, snapshot: BYOGSnapshot, config: CommunityConfig) -> GraphRAGResult:
        config.validate()
        validate_byog_snapshot(snapshot)
        if not snapshot.relationships:
            return GraphRAGResult(
                communities=self._append_singletons(snapshot, []),
                log="empty edge set; deterministic singleton communities",
            )
        if self.workspace_root:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(
            prefix="kf-graphrag-community-",
            dir=str(self.workspace_root) if self.workspace_root else None,
        ))
        try:
            output = workspace / "output"
            output.mkdir()
            pd.DataFrame(snapshot.entities).to_parquet(output / "entities.parquet", index=False)
            pd.DataFrame(snapshot.relationships).to_parquet(
                output / "relationships.parquet", index=False,
            )
            (workspace / "settings.yaml").write_text(self._settings(config), encoding="utf-8")
            completed = subprocess.run(
                self._command(workspace),
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                check=False,
            )
            combined_log = "\n".join((completed.stdout, completed.stderr)).replace(
                str(workspace), "<workspace>"
            )[-12000:]
            if completed.returncode:
                raise RuntimeError(
                    f"GraphRAG create_communities 失败(exit={completed.returncode}): "
                    f"{combined_log}"
                )
            result_path = output / "communities.parquet"
            if not result_path.is_file():
                raise RuntimeError("GraphRAG 未生成 communities.parquet")
            frame = pd.read_parquet(result_path)
            missing = REQUIRED_COMMUNITY_COLUMNS.difference(frame.columns)
            if missing:
                raise RuntimeError(f"GraphRAG communities schema 缺少字段: {sorted(missing)}")
            communities = self._append_singletons(
                snapshot,
                self._normalize_communities(snapshot, frame.to_dict("records")),
            )
            return GraphRAGResult(communities=communities, log=combined_log)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    @staticmethod
    def _append_singletons(
        snapshot: BYOGSnapshot,
        communities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        covered = {
            int(entity_id)
            for community in communities
            for entity_id in community["canonical_entity_ids"]
        }
        all_entity_ids = {
            int(str(entity["id"]).split(":", 1)[1]) for entity in snapshot.entities
        }
        next_community = max(
            (int(item["graphrag_community"]) for item in communities), default=-1,
        ) + 1
        by_id = {
            int(str(entity["id"]).split(":", 1)[1]): entity
            for entity in snapshot.entities
        }
        additions: list[dict[str, Any]] = []
        for offset, entity_id in enumerate(sorted(all_entity_ids - covered)):
            entity = by_id[entity_id]
            additions.append({
                "stable_community_id": stable_community_id(
                    snapshot.project_id, snapshot.input_hash, 0, [entity_id],
                ),
                "graphrag_output_id": f"singleton:{entity_id}",
                "graphrag_community": next_community + offset,
                "level": 0,
                "parent_stable_community_id": None,
                "title": str(entity["title"]),
                "size": 1,
                "period": "",
                "canonical_entity_ids": [entity_id],
                "relationship_ids": [],
                "text_unit_ids": sorted(map(str, entity.get("text_unit_ids") or [])),
                "children": [],
            })
        return sorted(
            [*communities, *additions],
            key=lambda item: (
                item["level"], item["graphrag_community"], item["stable_community_id"],
            ),
        )

    @staticmethod
    def _normalize_communities(
        snapshot: BYOGSnapshot,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_entity_id = {item["id"]: int(str(item["id"]).split(":", 1)[1])
                        for item in snapshot.entities}
        prepared: list[dict[str, Any]] = []
        community_map: dict[int, str] = {}
        for row in rows:
            entity_ids = sorted({
                by_entity_id[str(item)] for item in _list_value(row.get("entity_ids"))
                if str(item) in by_entity_id
            })
            if not entity_ids:
                raise RuntimeError("GraphRAG community 未包含可识别的 Knowledge Forge 实体")
            if int(row["size"]) != len(entity_ids):
                raise RuntimeError("GraphRAG community size 与成员数量不一致")
            level = int(row["level"])
            stable_id = stable_community_id(
                snapshot.project_id, snapshot.input_hash, level, entity_ids,
            )
            graph_community = int(row["community"])
            if graph_community in community_map and community_map[graph_community] != stable_id:
                raise RuntimeError("GraphRAG community 整数标识发生冲突")
            community_map[graph_community] = stable_id
            prepared.append({
                "stable_community_id": stable_id,
                "graphrag_output_id": str(row["id"]),
                "graphrag_community": graph_community,
                "level": level,
                "raw_parent": int(row["parent"]),
                "title": str(row["title"]),
                "size": int(row["size"]),
                "period": str(row.get("period") or ""),
                "canonical_entity_ids": entity_ids,
                "relationship_ids": sorted(map(str, _list_value(row.get("relationship_ids")))),
                "text_unit_ids": sorted(map(str, _list_value(row.get("text_unit_ids")))),
                "children": sorted(map(int, _list_value(row.get("children")))),
            })
        known_relationships = {str(item["id"]) for item in snapshot.relationships}
        known_communities = set(community_map)
        for item in prepared:
            parent = item.pop("raw_parent")
            if parent == -1:
                item["parent_stable_community_id"] = None
            elif parent in community_map:
                item["parent_stable_community_id"] = community_map[parent]
            else:
                raise RuntimeError(f"GraphRAG community parent {parent} 不存在")
            unknown_relationships = set(item["relationship_ids"]).difference(
                known_relationships
            )
            if unknown_relationships:
                raise RuntimeError("GraphRAG community 引用了未知关系")
            if set(item["children"]).difference(known_communities):
                raise RuntimeError("GraphRAG community 引用了未知子社区")
        return sorted(
            prepared,
            key=lambda item: (
                item["level"], item["graphrag_community"], item["stable_community_id"],
            ),
        )


class CommunityService:
    def __init__(self, database: Database | None = None, runner: GraphRAGCommunityRunner | None = None):
        self.database = database or Database()
        self.runner = runner or GraphRAGCommunityRunner()

    def detect(self, project_id: int, config: CommunityConfig) -> dict[str, Any]:
        control = self.database.export_graphrag_control_snapshot(project_id)
        byog = build_byog_snapshot(control, config)
        run_id = uuid.uuid4().hex
        self.database.create_graphrag_community_run(
            run_id=run_id,
            snapshot=control,
            input_hash=byog.input_hash,
            config=config.as_dict(),
            metrics=byog.metrics,
            graphrag_version=version("graphrag"),
            adapter_version=ADAPTER_VERSION,
            weighting_strategy=WEIGHTING_STRATEGY,
        )
        try:
            result = self.runner.run(byog, config)
            if byog.relationships and not result.communities:
                raise RuntimeError("非空正式图没有生成任何 GraphRAG community")
            try:
                refreshed = self.database.export_graphrag_control_snapshot(project_id)
            except RuntimeError:
                return self.database.supersede_graphrag_community_run(
                    run_id, "正式图 generation/policy 在检测期间失去 current 状态"
                )
            refreshed_byog = build_byog_snapshot(refreshed, config)
            if refreshed_byog.input_hash != byog.input_hash:
                return self.database.supersede_graphrag_community_run(
                    run_id, "正式图输入在检测期间发生变化"
                )
            status = "completed" if result.communities else "completed_empty"
            return self.database.finalize_graphrag_community_run(
                run_id=run_id,
                snapshot=control,
                input_hash=byog.input_hash,
                status=status,
                communities=result.communities,
                edges=byog.edges,
                metadata={"graphrag_log_tail": result.log[-2000:]},
            )
        except Exception as exc:
            self.database.fail_graphrag_community_run(run_id, str(exc)[-4000:])
            raise

    def status(self, run_id: str) -> dict[str, Any]:
        run = self.database.get_graphrag_community_run(run_id)
        return self._check_current_input(run)

    def show(self, run_id: str) -> dict[str, Any]:
        result = self.database.show_graphrag_community_run(run_id)
        run = self._check_current_input(result["run"])
        result["run"] = run
        result["current"] = run["current"]
        return result

    def _check_current_input(self, run: dict[str, Any]) -> dict[str, Any]:
        result = dict(run)
        if not result.get("current"):
            return result
        config_value = result.get("config") or {}
        config = CommunityConfig(
            seed=int(config_value.get("seed", 42)),
            max_cluster_size=int(config_value.get("max_cluster_size", 10)),
            use_lcc=bool(config_value.get("use_lcc", False)),
            timeout_seconds=int(config_value.get("timeout_seconds", 300)),
        )
        try:
            snapshot = self.database.export_graphrag_control_snapshot(
                int(result["project_id"])
            )
            result["current"] = (
                build_byog_snapshot(snapshot, config).input_hash == result["input_hash"]
            )
        except (LookupError, RuntimeError, ValueError):
            result["current"] = False
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shadow GraphRAG community detection")
    subparsers = parser.add_subparsers(dest="command", required=True)
    detect = subparsers.add_parser("detect", help="显式检测一个项目的正式图社区")
    detect.add_argument("--project-id", type=int, required=True)
    detect.add_argument("--seed", type=int, default=42)
    detect.add_argument("--max-cluster-size", type=int, default=10)
    detect.add_argument("--timeout", type=int, default=300)
    for command in ("status", "show"):
        item = subparsers.add_parser(command, help=f"只读 {command} 一个社区运行")
        item.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service = CommunityService()
    if args.command == "detect":
        output = service.detect(
            args.project_id,
            CommunityConfig(
                seed=args.seed,
                max_cluster_size=args.max_cluster_size,
                use_lcc=False,
                timeout_seconds=args.timeout,
            ),
        )
    elif args.command == "status":
        output = service.status(args.run_id)
    else:
        output = service.show(args.run_id)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
