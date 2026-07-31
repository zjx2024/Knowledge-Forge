"""Strategy-owned community clustering over Knowledge Forge snapshots."""

from __future__ import annotations

from knowledge_forge.graphrag_community import CommunityConfig, GraphRAGResult
from knowledge_forge.knowledge_forge_communities import (
    KnowledgeForgeCommunityService,
    build_knowledge_forge_byog,
)


def _snapshot():
    return {
        "canonicals": [
            {
                "canonical_key": "entity-a", "name": "甲",
                "primary_type": "Alpha", "source_count": 2, "metadata": {},
            },
            {
                "canonical_key": "entity-b", "name": "乙",
                "primary_type": "Beta", "source_count": 1, "metadata": {},
            },
            {
                "canonical_key": "entity-c", "name": "丙",
                "primary_type": "Beta", "source_count": 1, "metadata": {},
            },
        ],
        "relation_endpoints": [
            {
                "relation_key": "r1", "source_canonical_key": "entity-a",
                "target_canonical_key": "entity-b", "relation_type": "alpha_rel",
                "revision_id": "rev-1", "chunk_id": 1,
                "evidence_text": "甲与乙存在关系。",
            },
            {
                "relation_key": "r2", "source_canonical_key": "entity-b",
                "target_canonical_key": "entity-a", "relation_type": "beta_rel",
                "revision_id": "rev-1", "chunk_id": 2,
                "evidence_text": "乙也与甲存在关系。",
            },
        ],
    }


def test_strategy_snapshot_adapter_is_deterministic_and_preserves_isolates():
    first, first_keys, _ = build_knowledge_forge_byog(
        project_id=5, graph_generation=2, entity_namespace_id="namespace",
        snapshot=_snapshot(), config=CommunityConfig(),
    )
    reversed_snapshot = _snapshot()
    reversed_snapshot["canonicals"].reverse()
    reversed_snapshot["relation_endpoints"].reverse()
    second, second_keys, _ = build_knowledge_forge_byog(
        project_id=5, graph_generation=2, entity_namespace_id="namespace",
        snapshot=reversed_snapshot, config=CommunityConfig(),
    )

    assert first.input_hash == second.input_hash
    assert first_keys == second_keys
    assert len(first.entities) == 3
    assert len(first.relationships) == 1
    assert first.relationships[0]["weight"] == 2.0
    assert first.metrics["isolated_entity_count"] == 1


def test_strategy_community_service_persists_browser_ready_memberships():
    class Repository:
        def __init__(self):
            self.run = None

        def get_knowledge_forge_community_run(self, *_args, **_kwargs):
            return None

        def create_knowledge_forge_community_run(self, **values):
            self.run = {**values, "status": "running"}
            return self.run

        def complete_knowledge_forge_community_run(
            self, run_id, *, status, communities, hierarchy_levels, metadata,
        ):
            self.run.update({
                "run_id": run_id, "status": status,
                "communities": communities,
                "community_count": len(communities),
                "hierarchy_levels": hierarchy_levels,
                "metadata": metadata,
            })
            return self.run

        def fail_knowledge_forge_community_run(self, *_args):
            raise AssertionError("successful clustering must not fail")

    class Runner:
        def run(self, snapshot, _config):
            return GraphRAGResult(
                communities=[
                    {
                        "stable_community_id": "community-main",
                        "graphrag_community": 0, "level": 0,
                        "parent_stable_community_id": None,
                        "canonical_entity_ids": [1, 2],
                        "relationship_ids": [snapshot.relationships[0]["id"]],
                    },
                    {
                        "stable_community_id": "community-isolate",
                        "graphrag_community": 1, "level": 0,
                        "parent_stable_community_id": None,
                        "canonical_entity_ids": [3],
                        "relationship_ids": [],
                    },
                ],
                log="ok",
            )

    repository = Repository()
    result = KnowledgeForgeCommunityService(repository, Runner()).detect(
        project_id=5, graph_generation=2, entity_namespace_id="namespace",
        snapshot=_snapshot(),
    )

    assert result["status"] == "completed"
    assert result["community_count"] == 2
    assert result["communities"][0]["entity_keys"] == ["entity-a", "entity-b"]
    assert result["communities"][0]["relationship_count"] == 1
    assert result["communities"][1]["entity_keys"] == ["entity-c"]
