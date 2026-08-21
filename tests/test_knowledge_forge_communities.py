"""Strategy-owned community clustering over Knowledge Forge snapshots."""

from __future__ import annotations

import pytest

from knowledge_forge.graphrag_community import (
    CommunityConfig,
    CommunityDetectionCancelled,
    GraphRAGResult,
)
from knowledge_forge.knowledge_forge_communities import (
    COMMUNITY_NARRATIVE_VERSION,
    KnowledgeForgeCommunityService,
    build_knowledge_forge_byog,
    enrich_community_descriptions,
)
from knowledge_forge.knowledge_forge_descriptions import (
    enrich_knowledge_forge_snapshot,
    relation_display_label,
    relation_type_label,
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
    assert result["communities"][0]["description"]
    assert result["communities"][0]["description_source"] == (
        "verified_graph_theme_narrative"
    )
    assert "该社区主要介绍" in result["communities"][0]["description"]
    assert "该社区包含" not in result["communities"][0]["description"]
    assert result["communities"][1]["entity_keys"] == ["entity-c"]


def test_community_description_explains_the_topic_instead_of_repeating_counts():
    snapshot = {
        "canonicals": [
            {
                "canonical_key": "platform", "name": "系统平台",
                "primary_type": "Product", "source_count": 3,
            },
            {
                "canonical_key": "task", "name": "任务管理",
                "primary_type": "Method", "source_count": 1,
            },
            {
                "canonical_key": "storage", "name": "存储管理",
                "primary_type": "Method", "source_count": 1,
            },
            {
                "canonical_key": "timer", "name": "定时器管理",
                "primary_type": "Method", "source_count": 1,
            },
        ],
        "relation_endpoints": [
            {
                "relation_key": "task-platform",
                "source_canonical_key": "task",
                "target_canonical_key": "platform",
                "relation_type": "structural_relation",
                "evidence_text": (
                    "系统平台的主要能力包括任务管理、存储管理和定时器管理。"
                ),
                "metadata": {"display_relation_label": "主要包括"},
            },
            {
                "relation_key": "storage-platform",
                "source_canonical_key": "storage",
                "target_canonical_key": "platform",
                "relation_type": "structural_relation",
                "evidence_text": (
                    "系统平台的主要能力包括任务管理、存储管理和定时器管理。"
                ),
                "metadata": {"display_relation_label": "主要包括"},
            },
            {
                "relation_key": "timer-platform",
                "source_canonical_key": "timer",
                "target_canonical_key": "platform",
                "relation_type": "structural_relation",
                "evidence_text": (
                    "系统平台的主要能力包括任务管理、存储管理和定时器管理。"
                ),
                "metadata": {"display_relation_label": "主要包括"},
            },
        ],
    }
    old = [{
        "key": "community-1",
        "entity_keys": ["task", "storage", "platform", "timer"],
        "summary": "该社区包含 4 个规范实体、3 条已验证关系。",
        "description_source": "verified_facts_and_membership",
    }]

    result = enrich_community_descriptions(old, snapshot=snapshot)
    description = result[0]["description"]

    assert description.startswith("该社区主要围绕“系统平台”展开")
    assert all(name in description for name in ("任务管理", "存储管理", "定时器管理"))
    assert "主要包括" in description
    assert "可核对的原文要点" in description
    assert "4 个规范实体" not in description
    assert result[0]["summary"] == description
    assert result[0]["topic_entities"][0] == "系统平台"
    assert result[0]["description_version"] == COMMUNITY_NARRATIVE_VERSION


def test_community_description_for_isolate_is_honest_about_missing_relations():
    result = enrich_community_descriptions(
        [{"key": "isolate", "entity_keys": ["entity-c"]}],
        snapshot=_snapshot(),
    )

    assert result[0]["description"].startswith("该社区主要收录“丙”这一主题")
    assert "尚无已验证的内部关系" in result[0]["description"]


def test_strategy_community_cancellation_is_persisted_and_propagated():
    class Repository:
        def __init__(self):
            self.failed = None

        def get_knowledge_forge_community_run(self, *_args, **_kwargs):
            return None

        def create_knowledge_forge_community_run(self, **values):
            return {**values, "status": "running"}

        def fail_knowledge_forge_community_run(self, run_id, error):
            self.failed = (run_id, error)

    class Runner:
        def run(self, _snapshot, _config, *, cancel_requested):
            assert cancel_requested()
            raise CommunityDetectionCancelled("cancelled at safe checkpoint")

    repository = Repository()
    with pytest.raises(CommunityDetectionCancelled, match="safe checkpoint"):
        KnowledgeForgeCommunityService(repository, Runner()).detect(
            project_id=5,
            graph_generation=2,
            entity_namespace_id="namespace",
            snapshot=_snapshot(),
            cancel_requested=lambda: True,
        )

    assert repository.failed is not None
    assert "safe checkpoint" in repository.failed[1]


def test_grounded_descriptions_cover_entities_relations_and_communities():
    snapshot = _snapshot()
    snapshot["assignments"] = [
        {"mention_key": "mention-a", "canonical_key": "entity-a"},
        {"mention_key": "mention-b", "canonical_key": "entity-b"},
    ]
    enriched = enrich_knowledge_forge_snapshot(
        snapshot,
        mentions=[
            {
                "mention_key": "mention-a", "document_id": 1, "chunk_id": 1,
                "evidence_start": 0, "evidence_text": "甲负责一项明确工作。",
            },
            {
                "mention_key": "mention-b", "document_id": 1, "chunk_id": 1,
                "evidence_start": 3, "evidence_text": "乙是该工作的对象。",
            },
        ],
    )

    canonical = {item["canonical_key"]: item for item in enriched["canonicals"]}
    assert "原文信息：甲负责一项明确工作" in canonical["entity-a"]["metadata"]["description"]
    assert "已验证联系" in canonical["entity-a"]["metadata"]["description"]
    assert "已验证关系" in enriched["relation_endpoints"][0]["metadata"]["display_description"]

    byog, _, _ = build_knowledge_forge_byog(
        project_id=5, graph_generation=2, entity_namespace_id="namespace",
        snapshot=enriched, config=CommunityConfig(),
    )
    assert "原文信息" in byog.entities[0]["description"]
    assert "已验证关系" in byog.relationships[0]["description"]


def test_grounded_descriptions_are_chinese_and_avoid_duplicate_punctuation():
    snapshot = _snapshot()
    snapshot["assignments"] = [
        {"mention_key": "mention-a", "canonical_key": "entity-a"},
    ]
    snapshot["canonicals"][0]["primary_type"] = "Organization"
    snapshot["relation_endpoints"][0]["relation_type"] = "uses"
    snapshot["relation_endpoints"][0]["metadata"] = {
        "description": "经核验的补充说明。",
    }
    enriched = enrich_knowledge_forge_snapshot(
        snapshot,
        mentions=[{
            "mention_key": "mention-a", "document_id": 1, "chunk_id": 1,
            "evidence_start": 0, "evidence_text": "甲。",
        }],
    )

    entity = enriched["canonicals"][0]
    relation = enriched["relation_endpoints"][0]
    assert "组织机构" in entity["metadata"]["description"]
    assert "原文信息：甲" not in entity["metadata"]["description"]
    assert relation["metadata"]["display_relation_label"] == "使用"
    assert "经核验的补充说明。" in relation["metadata"]["display_description"]
    assert "。。" not in relation["metadata"]["display_description"]


def test_grounded_descriptions_and_community_narrative_follow_english_policy():
    snapshot = _snapshot()
    snapshot["assignments"] = [
        {"mention_key": "mention-a", "canonical_key": "entity-a"},
    ]
    snapshot["canonicals"][0]["primary_type"] = "Organization"
    snapshot["relation_endpoints"][0]["relation_type"] = "uses"
    snapshot["relation_endpoints"][0]["evidence_text"] = (
        "Alpha explicitly uses Beta."
    )
    enriched = enrich_knowledge_forge_snapshot(
        snapshot,
        mentions=[{
            "mention_key": "mention-a", "document_id": 1, "chunk_id": 1,
            "evidence_start": 0, "evidence_text": "Alpha performs the work.",
        }],
        knowledge_language="en",
    )

    entity = enriched["canonicals"][0]
    relation = enriched["relation_endpoints"][0]
    assert "is identified in the source documents as organization" in entity[
        "metadata"
    ]["description"]
    assert "Verified connections:" in entity["metadata"]["description"]
    assert relation["metadata"]["display_relation_label"] == "uses"
    assert "The verified relation" in relation["metadata"]["display_description"]
    assert relation["metadata"]["description_language"] == "en"

    communities = enrich_community_descriptions(
        [{"key": "main", "entity_keys": ["entity-a", "entity-b"]}],
        snapshot=enriched,
        knowledge_language="en",
    )
    assert communities[0]["description"].startswith(
        "This community describes the knowledge connection"
    )
    assert "The graph primarily uses" in communities[0]["description"]
    assert communities[0]["description_language"] == "en"

    byog, _, _ = build_knowledge_forge_byog(
        project_id=5, graph_generation=2, entity_namespace_id="namespace",
        snapshot=enriched, config=CommunityConfig(), knowledge_language="en",
    )
    assert "The verified relation" in byog.relationships[0]["description"]


def test_relation_display_labels_never_expose_machine_predicate_names():
    assert relation_type_label("supply_to") == "供应给"
    assert relation_type_label("executes_task") == "执行任务"
    assert relation_type_label("source_assertion") == "原文直接关系"
    assert relation_type_label("fictional_relation") == "Fictional relation"
    assert "_" not in relation_type_label("fictional_relation")
    assert relation_type_label("supply_to", "en") == "supplies to"


def test_open_relation_carrier_has_a_safe_label_without_source_expression():
    assert relation_display_label(
        "stable_internal_id",
        {"open_relation_carrier": True},
    ) == "原文直接关系"


def test_precise_schema_relation_does_not_display_model_source_paraphrase():
    assert relation_display_label(
        "uses",
        {
            "source_relation_expression": "引入于",
            "open_relation_carrier": False,
        },
    ) == "使用"


def test_grounded_source_relation_label_survives_description_enrichment():
    snapshot = _snapshot()
    snapshot["relation_endpoints"] = [{
        "relation_key": "open-1",
        "source_canonical_key": "entity-a",
        "target_canonical_key": "entity-b",
        "relation_type": "internal_open_carrier",
        "revision_id": "rev-1",
        "chunk_id": 1,
        "evidence_text": "甲明确负责乙。",
        "metadata": {
            "display_relation_label": "负责",
            "source_relation_expression": "负责",
            "canonical_relation_type": "internal_open_carrier",
            "normalization_status": "OPEN_GROUNDED",
        },
    }]

    relation = enrich_knowledge_forge_snapshot(
        snapshot, mentions=[],
    )["relation_endpoints"][0]

    assert relation["metadata"]["display_relation_label"] == "负责"
    assert "关系为“负责”" in relation["metadata"]["display_description"]
    assert "internal_open_carrier" not in relation["metadata"][
        "display_description"
    ]
