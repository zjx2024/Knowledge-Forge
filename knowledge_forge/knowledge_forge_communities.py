"""Community clustering for the immutable Knowledge Forge strategy snapshot."""

from __future__ import annotations

import hashlib
import json
from typing import Any
import uuid

from .construction_strategies import ENHANCED_SHADOW_STRATEGY
from .graphrag_community import (
    ADAPTER_VERSION,
    BYOGSnapshot,
    CommunityConfig,
    GraphRAGCommunityRunner,
)


STRATEGY_COMMUNITY_ADAPTER_VERSION = ADAPTER_VERSION + "-strategy-snapshot-v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def build_knowledge_forge_byog(
    *,
    project_id: int,
    graph_generation: int,
    entity_namespace_id: str,
    snapshot: dict[str, Any],
    config: CommunityConfig,
) -> tuple[BYOGSnapshot, dict[int, str], dict[str, dict[str, Any]]]:
    """Adapt text-keyed strategy canonicals to GraphRAG's supported BYOG input."""

    config.validate()
    canonicals = sorted(
        list(snapshot.get("canonicals") or []),
        key=lambda item: str(item.get("canonical_key") or ""),
    )
    canonical_by_key = {
        str(item.get("canonical_key") or ""): item
        for item in canonicals if str(item.get("canonical_key") or "")
    }
    id_by_key = {
        key: index for index, key in enumerate(canonical_by_key, start=1)
    }
    key_by_id = {value: key for key, value in id_by_key.items()}

    relation_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    excluded_self_loops = 0
    for relation in sorted(
        list(snapshot.get("relation_endpoints") or []),
        key=lambda item: str(item.get("relation_key") or ""),
    ):
        source = str(relation.get("source_canonical_key") or "")
        target = str(relation.get("target_canonical_key") or "")
        if source not in canonical_by_key or target not in canonical_by_key:
            raise ValueError("Knowledge Forge 社区输入存在悬空关系端点")
        if source == target:
            excluded_self_loops += 1
            continue
        relation_groups.setdefault(tuple(sorted((source, target))), []).append(relation)

    degree = {key: 0 for key in canonical_by_key}
    incident_units: dict[str, set[str]] = {key: set() for key in canonical_by_key}
    relationships: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for human_id, ((source, target), rows) in enumerate(sorted(relation_groups.items())):
        degree[source] += 1
        degree[target] += 1
        relationship_id = "kf_strategy_edge:" + _digest([
            project_id, entity_namespace_id, source, target,
            sorted(str(row.get("relation_key") or "") for row in rows),
        ])[:32]
        relation_types = sorted({
            str(row.get("relation_type") or "") for row in rows
            if str(row.get("relation_type") or "")
        })
        text_unit_ids = sorted({
            f"chunk:{row.get('revision_id')}:{row.get('chunk_id')}"
            for row in rows
        })
        incident_units[source].update(text_unit_ids)
        incident_units[target].update(text_unit_ids)
        evidence = sorted({
            str(row.get("evidence_text") or "").strip()
            for row in rows if str(row.get("evidence_text") or "").strip()
        })
        description = "; ".join([
            "Types: " + ", ".join(relation_types),
            *evidence,
        ])[:4000]
        weight = float(len(rows))
        relationships.append({
            "id": relationship_id,
            "human_readable_id": human_id,
            "source": source,
            "target": target,
            "description": description,
            "weight": weight,
            "combined_degree": degree[source] + degree[target],
            "text_unit_ids": text_unit_ids,
        })
        edges.append({
            "stable_edge_id": relationship_id,
            "source_entity_id": id_by_key[source],
            "target_entity_id": id_by_key[target],
            "weight": weight,
            "relation_keys": sorted(
                str(row.get("relation_key") or "") for row in rows
            ),
            "relation_types": relation_types,
            "text_unit_ids": text_unit_ids,
        })

    entities: list[dict[str, Any]] = []
    for human_id, (key, item) in enumerate(canonical_by_key.items()):
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        description = "\n".join(filter(None, [
            f"Name: {item.get('name') or key}",
            f"Type: {item.get('primary_type') or 'GenericEntity'}",
            str(metadata.get("description") or ""),
        ]))[:4000]
        entities.append({
            "id": f"kf_entity:{id_by_key[key]}",
            "human_readable_id": human_id,
            "title": key,
            "type": str(item.get("primary_type") or "GenericEntity"),
            "description": description,
            "text_unit_ids": sorted(incident_units[key]),
            "frequency": int(item.get("source_count") or 0),
            "degree": int(degree[key]),
        })

    input_hash = _digest({
        "project_id": int(project_id),
        "strategy": ENHANCED_SHADOW_STRATEGY,
        "graph_generation": int(graph_generation),
        "entity_namespace_id": str(entity_namespace_id),
        "config": config.as_dict(),
        "canonicals": [
            {
                "key": key,
                "name": canonical_by_key[key].get("name"),
                "type": canonical_by_key[key].get("primary_type"),
            }
            for key in canonical_by_key
        ],
        "relationships": relationships,
    })
    result = BYOGSnapshot(
        project_id=int(project_id),
        input_hash=input_hash,
        entities=entities,
        relationships=relationships,
        edges=edges,
        metrics={
            "node_count": len(entities),
            "aggregated_edge_count": len(relationships),
            "excluded_self_loops": excluded_self_loops,
            "excluded_isolated_entities": 0,
            "isolated_entity_count": sum(1 for value in degree.values() if value == 0),
        },
    )
    return result, key_by_id, canonical_by_key


def _browser_communities(
    communities: list[dict[str, Any]],
    *,
    key_by_id: dict[int, str],
    canonical_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in communities:
        entity_keys = [
            key_by_id[int(entity_id)]
            for entity_id in item.get("canonical_entity_ids") or []
            if int(entity_id) in key_by_id
        ]
        names = [
            str(canonical_by_key[key].get("name") or key) for key in entity_keys
        ]
        relationship_keys = list(item.get("relationship_ids") or [])
        level = int(item.get("level") or 0)
        number = int(item.get("graphrag_community") or 0)
        result.append({
            "key": str(item["stable_community_id"]),
            "label": f"社区 {number} · L{level}",
            "level": level,
            "community": number,
            "parent": item.get("parent_stable_community_id"),
            "size": len(entity_keys),
            "relationship_count": len(relationship_keys),
            "entity_keys": entity_keys,
            "relationship_keys": relationship_keys,
            "summary": (
                f"包含 {len(entity_keys)} 个规范实体、"
                f"{len(relationship_keys)} 条内部聚合关系。"
                + (f" 代表成员：{'、'.join(names[:6])}。" if names else "")
            ),
        })
    return sorted(
        result, key=lambda item: (
            int(item["level"]), int(item["community"]), str(item["key"]),
        ),
    )


class KnowledgeForgeCommunityService:
    """Run and persist clustering bound to one ready strategy generation."""

    def __init__(self, database, runner: GraphRAGCommunityRunner | None = None):
        self.database = database
        self.runner = runner or GraphRAGCommunityRunner()

    def detect(
        self,
        *,
        project_id: int,
        graph_generation: int,
        entity_namespace_id: str,
        snapshot: dict[str, Any],
        config: CommunityConfig | None = None,
    ) -> dict[str, Any]:
        config = config or CommunityConfig()
        byog, key_by_id, canonical_by_key = build_knowledge_forge_byog(
            project_id=project_id,
            graph_generation=graph_generation,
            entity_namespace_id=entity_namespace_id,
            snapshot=snapshot,
            config=config,
        )
        existing = self.database.get_knowledge_forge_community_run(
            project_id, graph_generation, entity_namespace_id,
            input_hash=byog.input_hash,
        )
        if existing:
            return existing
        created = self.database.create_knowledge_forge_community_run(
            run_id="kfcommunity_" + uuid.uuid4().hex,
            project_id=project_id,
            graph_generation=graph_generation,
            entity_namespace_id=entity_namespace_id,
            input_hash=byog.input_hash,
            node_count=byog.metrics["node_count"],
            edge_count=byog.metrics["aggregated_edge_count"],
            config={
                **config.as_dict(),
                "adapter_version": STRATEGY_COMMUNITY_ADAPTER_VERSION,
            },
        )
        run_id = str(created["run_id"])
        if str(created.get("status") or "") in {"completed", "completed_empty"}:
            return created
        try:
            raw = self.runner.run(byog, config)
            communities = _browser_communities(
                raw.communities,
                key_by_id=key_by_id,
                canonical_by_key=canonical_by_key,
            )
            status = "completed" if communities else "completed_empty"
            return self.database.complete_knowledge_forge_community_run(
                run_id,
                status=status,
                communities=communities,
                hierarchy_levels=len({
                    int(item["level"]) for item in communities
                }),
                metadata={
                    "graphrag_log_tail": raw.log[-2000:],
                    "input_metrics": byog.metrics,
                },
            )
        except Exception as exc:
            self.database.fail_knowledge_forge_community_run(run_id, str(exc))
            raise
