from __future__ import annotations

from dataclasses import replace
import math

import pytest

from knowledge_forge.graphrag_community import (
    CommunityConfig,
    ControlGraphSnapshot,
    GraphRAGCommunityRunner,
    build_byog_snapshot,
    stable_community_id,
    validate_byog_snapshot,
)


def relation(
    assertion_id: int,
    source_id: int,
    source_key: str,
    target_id: int,
    target_key: str,
    *,
    chunk_id: int | None = None,
    revision_id: str = "revision-a",
    relation_type: str = "related_to",
    description: str = "",
) -> dict:
    chunk_id = assertion_id if chunk_id is None else chunk_id
    return {
        "assertion_id": assertion_id,
        "document_id": 11,
        "chunk_id": chunk_id,
        "chunk_key": f"chunk-{chunk_id}",
        "revision_id": revision_id,
        "relation_type": relation_type,
        "relation_description": description,
        "source_entity_id": source_id,
        "source_entity_key": source_key,
        "source_name": f"name-{source_id}",
        "source_type": "Concept",
        "source_description": f"description-{source_id}",
        "target_entity_id": target_id,
        "target_entity_key": target_key,
        "target_name": f"name-{target_id}",
        "target_type": "Concept",
        "target_description": f"description-{target_id}",
    }


def snapshot(*relations: dict) -> ControlGraphSnapshot:
    return ControlGraphSnapshot(
        project_id=7,
        graph_generation=3,
        graph_policy_hash="graph-policy",
        active_policy_hash="active-policy",
        relations=tuple(relations),
    )


def test_byog_contract_is_stable_and_aggregates_reverse_duplicate_edges():
    graph = snapshot(
        relation(3, 2, "entity-b", 1, "entity-a", relation_type="uses"),
        relation(1, 1, "entity-a", 2, "entity-b", relation_type="depends_on"),
        relation(2, 1, "entity-a", 2, "entity-b", description="evidence"),
        relation(4, 1, "entity-a", 1, "entity-a"),
    )

    byog = build_byog_snapshot(graph, CommunityConfig())

    assert [item["id"] for item in byog.entities] == ["kf_entity:1", "kf_entity:2"]
    assert [item["human_readable_id"] for item in byog.entities] == [0, 1]
    assert [item["title"] for item in byog.entities] == ["entity-a", "entity-b"]
    assert all(isinstance(item["text_unit_ids"], list) for item in byog.entities)
    assert all(item["degree"] == 1 for item in byog.entities)

    assert len(byog.relationships) == 1
    edge = byog.relationships[0]
    assert edge["source"] == "entity-a"
    assert edge["target"] == "entity-b"
    assert edge["weight"] == 3.0
    assert edge["combined_degree"] == 2
    assert edge["text_unit_ids"] == [
        "chunk:revision-a:chunk-1",
        "chunk:revision-a:chunk-2",
        "chunk:revision-a:chunk-3",
    ]
    assert byog.edges[0]["assertion_ids"] == [1, 2, 3]
    assert byog.edges[0]["relation_types"] == ["depends_on", "related_to", "uses"]
    assert byog.metrics["excluded_self_loops"] == 1
    assert byog.metrics["excluded_isolated_entities"] == 0
    validate_byog_snapshot(byog)


def test_byog_hash_and_ids_do_not_depend_on_database_order():
    rows = (
        relation(1, 1, "entity-a", 2, "entity-b"),
        relation(2, 2, "entity-b", 3, "entity-c"),
    )
    first = build_byog_snapshot(snapshot(*rows), CommunityConfig(seed=42))
    second = build_byog_snapshot(snapshot(*reversed(rows)), CommunityConfig(seed=42))

    assert first.input_hash == second.input_hash
    assert first.entities == second.entities
    assert first.relationships == second.relationships
    assert first.edges == second.edges
    assert stable_community_id(7, first.input_hash, 1, [3, 1, 2]) == stable_community_id(
        7, second.input_hash, 1, [2, 3, 1]
    )
    assert stable_community_id(7, first.input_hash, 1, [1, 2]) != stable_community_id(
        7, first.input_hash, 2, [1, 2]
    )


def test_config_changes_are_part_of_input_hash():
    graph = snapshot(relation(1, 1, "entity-a", 2, "entity-b"))
    first = build_byog_snapshot(graph, CommunityConfig(seed=42))
    second = build_byog_snapshot(graph, CommunityConfig(seed=99))
    assert first.input_hash != second.input_hash


def test_empty_graph_is_a_completed_empty_candidate_without_graphrag_input():
    byog = build_byog_snapshot(snapshot(), CommunityConfig())
    assert byog.entities == []
    assert byog.relationships == []
    assert byog.edges == []
    assert byog.metrics == {
        "node_count": 0,
        "aggregated_edge_count": 0,
        "excluded_self_loops": 0,
        "excluded_isolated_entities": 0,
    }
    validate_byog_snapshot(byog)


def test_self_loop_only_graph_counts_the_excluded_isolated_entity():
    byog = build_byog_snapshot(
        snapshot(relation(1, 1, "entity-a", 1, "entity-a")), CommunityConfig()
    )

    assert byog.entities == []
    assert byog.relationships == []
    assert byog.metrics == {
        "node_count": 0,
        "aggregated_edge_count": 0,
        "excluded_self_loops": 1,
        "excluded_isolated_entities": 1,
    }


def test_validation_rejects_dangling_nonfinite_and_nonpositive_edges():
    byog = build_byog_snapshot(
        snapshot(relation(1, 1, "entity-a", 2, "entity-b")), CommunityConfig()
    )

    dangling = replace(
        byog,
        relationships=[{**byog.relationships[0], "target": "missing"}],
    )
    with pytest.raises(ValueError, match="悬空"):
        validate_byog_snapshot(dangling)

    for weight in (0.0, -1.0, math.inf, math.nan):
        invalid = replace(
            byog,
            relationships=[{**byog.relationships[0], "weight": weight}],
        )
        with pytest.raises(ValueError, match="权重"):
            validate_byog_snapshot(invalid)


def test_real_graphrag_fixture_is_deterministic_without_external_models(tmp_path):
    internal_repeats = 4
    relations = []
    assertion_id = 1
    for source, source_key, target, target_key in (
        (1, "a-1", 2, "a-2"),
        (2, "a-2", 3, "a-3"),
        (1, "a-1", 3, "a-3"),
        (4, "b-1", 5, "b-2"),
        (5, "b-2", 6, "b-3"),
        (4, "b-1", 6, "b-3"),
    ):
        for _ in range(internal_repeats):
            relations.append(relation(assertion_id, source, source_key, target, target_key))
            assertion_id += 1
    relations.append(relation(assertion_id, 3, "a-3", 4, "b-1"))
    byog = build_byog_snapshot(snapshot(*relations), CommunityConfig(seed=42))
    runner = GraphRAGCommunityRunner(workspace_root=tmp_path)

    first = runner.run(byog, CommunityConfig(seed=42, max_cluster_size=10))
    second = runner.run(byog, CommunityConfig(seed=42, max_cluster_size=10))

    def normalized(result):
        return sorted(
            (
                int(item["level"]),
                tuple(sorted(item["canonical_entity_ids"])),
                tuple(sorted(item["relationship_ids"])),
                int(item["size"]),
            )
            for item in result.communities
        )

    assert normalized(first) == normalized(second)
    assert {item["stable_community_id"] for item in first.communities} == {
        item["stable_community_id"] for item in second.communities
    }
    assert len(byog.entities) == 6
    assert len(byog.relationships) == 7
    assert first.communities
    assert all(item["stable_community_id"] for item in first.communities)
    assert all(
        item["parent_stable_community_id"] is None
        or item["parent_stable_community_id"]
        in {community["stable_community_id"] for community in first.communities}
        for item in first.communities
    )
    assert not list(tmp_path.iterdir())


def test_real_graphrag_processes_every_disconnected_component(tmp_path):
    byog = build_byog_snapshot(
        snapshot(
            relation(1, 1, "component-a-1", 2, "component-a-2"),
            relation(2, 3, "component-b-1", 4, "component-b-2"),
        ),
        CommunityConfig(seed=42),
    )

    result = GraphRAGCommunityRunner(workspace_root=tmp_path).run(
        byog, CommunityConfig(seed=42, max_cluster_size=10)
    )

    covered_entities = {
        entity_id
        for community in result.communities
        for entity_id in community["canonical_entity_ids"]
    }
    assert covered_entities == {1, 2, 3, 4}
    assert not list(tmp_path.iterdir())


def test_graphrag_settings_enable_only_community_detection():
    settings = GraphRAGCommunityRunner._settings(
        CommunityConfig(seed=42, max_cluster_size=17, use_lcc=False)
    )
    assert "create_communities" in settings
    assert "create_community_reports" not in settings
    assert "generate_text_embeddings" not in settings
    assert "global" not in settings.lower()
    assert "use_lcc: false" in settings
    assert "max_cluster_size: 17" in settings
    assert "seed: 42" in settings


def test_largest_component_mode_is_rejected_to_preserve_disconnected_graphs():
    with pytest.raises(ValueError, match="use_lcc=false"):
        CommunityConfig(use_lcc=True).validate()


def test_runner_failure_cleans_only_its_workspace(monkeypatch, tmp_path):
    byog = build_byog_snapshot(
        snapshot(relation(1, 1, "entity-a", 2, "entity-b")), CommunityConfig()
    )
    sentinel = tmp_path / "keep-me.txt"
    sentinel.write_text("user file", encoding="utf-8")

    class Failed:
        returncode = 2
        stdout = "failed stdout"
        stderr = "failed stderr"

    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: Failed())
    with pytest.raises(RuntimeError, match="create_communities 失败"):
        GraphRAGCommunityRunner(workspace_root=tmp_path).run(byog, CommunityConfig())

    assert sentinel.read_text(encoding="utf-8") == "user file"
    assert [item.name for item in tmp_path.iterdir()] == ["keep-me.txt"]


def test_service_failure_is_persisted_and_never_completed():
    graph = snapshot(relation(1, 1, "entity-a", 2, "entity-b"))

    class FailedDatabase:
        def __init__(self):
            self.failed = None
            self.created = None

        def export_graphrag_control_snapshot(self, _project_id):
            return graph

        def create_graphrag_community_run(self, **values):
            self.created = values
            return {"run_id": values["run_id"], "status": "running"}

        def fail_graphrag_community_run(self, run_id, error):
            self.failed = {"run_id": run_id, "error": error, "status": "failed"}
            return self.failed

    class FailedRunner:
        def run(self, *_args, **_kwargs):
            raise RuntimeError("controlled failure")

    database = FailedDatabase()
    from knowledge_forge.graphrag_community import CommunityService

    with pytest.raises(RuntimeError, match="controlled failure"):
        CommunityService(database, FailedRunner()).detect(7, CommunityConfig())

    assert database.created["run_id"] == database.failed["run_id"]
    assert database.failed["status"] == "failed"
    assert "controlled failure" in database.failed["error"]
