from __future__ import annotations

import os
import uuid

import psycopg
import pytest

from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.graph_manifest import (
    DuplicateGraphLogicalId,
    build_graph_manifest,
    compare_graph_manifests,
)
from knowledge_forge.migrations import MIGRATIONS


def _records():
    return [
        {"kind": "vertex", "label": "KFEntity", "logical_id": "entity:a",
         "properties": {"kf_name": "Alpha", "kf_entity_type": "Organization"}},
        {"kind": "vertex", "label": "KFEntity", "logical_id": "entity:b",
         "properties": {"kf_name": "Beta", "kf_entity_type": "Product"}},
        {"kind": "edge", "label": "kf_related", "logical_id": "relation:1",
         "source": "entity:a", "target": "entity:b", "direction": "out-in",
         "properties": {"kf_relation_type": "uses"}},
    ]


def _manifest(records=None):
    return build_graph_manifest(
        records or _records(), namespace_id=str(uuid.UUID(int=1)), project_id=7,
        graph_generation=3, policy_hash="policy-v1",
    )


def _replace_one_fact(rows):
    rows.pop(0)
    rows.append({
        "kind": "vertex", "label": "KFEntity", "logical_id": "entity:c",
        "properties": {"kf_name": "Gamma", "kf_entity_type": "Organization"},
    })


@pytest.mark.parametrize("mutator", [
    lambda rows: rows[2].update(target="entity:a"),
    lambda rows: rows[2].update(direction="in-out"),
    lambda rows: rows[2]["properties"].update(kf_relation_type="owns"),
    lambda rows: rows[0].update(label="KFChunk"),
    _replace_one_fact,
])
def test_equal_counts_with_wrong_graph_content_never_match(mutator):
    rows = _records()
    mutator(rows)
    expected, actual = _manifest(), _manifest(rows)
    comparison = compare_graph_manifests(expected, actual, actual)
    assert comparison["complete"] is False
    assert comparison["reasons"]


def test_manifest_is_order_independent_and_duplicate_ids_fail_closed():
    assert _manifest(_records()) == _manifest(list(reversed(_records())))
    with pytest.raises(DuplicateGraphLogicalId):
        _manifest([*_records(), _records()[0]])


def test_graph_namespace_manifest_and_lease_are_versioned_migration_facts():
    migration = next(item for item in MIGRATIONS if item.version == "0006_graph_namespace_manifest")
    assert "graph_namespace_id" in migration.sql
    assert "graph_content_manifests" in migration.sql
    assert "graph_leases" in migration.sql

    strategy_migration = next(
        item for item in MIGRATIONS
        if item.version == "0013_strategy_graph_lease_namespace"
    )
    assert "DROP CONSTRAINT IF EXISTS graph_leases_project_id_namespace_id_fkey" in strategy_migration.sql
    assert "DROP CONSTRAINT IF EXISTS graph_content_manifests_project_id_namespace_id_fkey" in strategy_migration.sql


@pytest.mark.integration
def test_disposable_postgres_graph_lease_is_exclusive_recoverable_and_generation_scoped():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    params = psycopg.conninfo.conninfo_to_dict(dsn)

    class TestConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]
        pgvector_hnsw_max_dimensions = Config.pgvector_hnsw_max_dimensions
        pgvector_hnsw_m = Config.pgvector_hnsw_m
        pgvector_hnsw_ef_construction = Config.pgvector_hnsw_ef_construction

    database = Database(TestConfig)
    database.bootstrap()
    project = database.create_project("graph-lease", "", None, None)
    database.ensure_project_strategy_graphs(project["id"])
    first = database.acquire_graph_lease(
        project["id"], project["graph_namespace_id"], 0, "worker-a", ttl_seconds=30,
    )
    manifest = build_graph_manifest(
        [], namespace_id=project["graph_namespace_id"], project_id=project["id"],
        graph_generation=0, policy_hash="policy-v1",
    )
    database.persist_graph_manifest(
        project["id"], project["graph_namespace_id"], 0, "expected",
        first["lease_token"], manifest,
    )
    with pytest.raises(ValueError, match="scope"):
        database.persist_graph_manifest(
            project["id"], project["graph_namespace_id"], 0, "writer",
            first["lease_token"], {**manifest, "project_id": project["id"] + 1},
        )
    with pytest.raises(RuntimeError, match="lease"):
        database.acquire_graph_lease(
            project["id"], project["graph_namespace_id"], 0, "worker-b", ttl_seconds=30,
        )
    assert database.assert_graph_lease(
        project["id"], project["graph_namespace_id"], 0, first["lease_token"],
    )
    database.release_graph_lease(first["id"], first["lease_token"])
    second = database.acquire_graph_lease(
        project["id"], project["graph_namespace_id"], 0, "worker-b", ttl_seconds=30,
    )
    assert second["lease_token"] != first["lease_token"]
    for role in ("expected", "writer", "actual"):
        database.persist_graph_manifest(
            project["id"], project["graph_namespace_id"], 0, role,
            second["lease_token"], manifest,
        )
    completed = database.complete_graph_rebuild(
        project["id"], 0, policy_hash="policy-v1",
        namespace_id=project["graph_namespace_id"],
        lease_token=second["lease_token"],
        manifest_digest=manifest["content_digest"], empty=True,
        stats={"documents": 0, "chunks": 0, "entities": 0, "relations": 0},
    )
    assert completed is not None
    stable_graph = database.get_project_strategy_graph(project["id"], "stable")
    assert stable_graph and stable_graph["status"] == "empty"
    assert stable_graph["ready_generation"] == 0
    assert stable_graph["stats"] == {
        "documents": 0, "chunks": 0, "entities": 0, "relations": 0,
    }

    strategy_graph = next(
        item for item in database.ensure_project_strategy_graphs(project["id"])
        if item["strategy_key"] == "knowledge_forge"
    )
    strategy_graph = database.start_strategy_graph_rebuild(
        project["id"], "knowledge_forge",
    )
    strategy_lease = database.acquire_graph_lease(
        project["id"], strategy_graph["namespace_id"], strategy_graph["generation"],
        "strategy:knowledge_forge:job-recovery", ttl_seconds=30,
    )
    assert strategy_lease["namespace_id"] == strategy_graph["namespace_id"]
    assert database.release_job_graph_leases("unrelated-job") == 0
    assert database.assert_graph_lease(
        project["id"], strategy_graph["namespace_id"],
        strategy_graph["generation"], strategy_lease["lease_token"],
    )
    assert database.release_job_graph_leases("job-recovery") == 1
    assert not database.assert_graph_lease(
        project["id"], strategy_graph["namespace_id"],
        strategy_graph["generation"], strategy_lease["lease_token"],
    )

    rebuild_lease = database.acquire_graph_lease(
        project["id"], project["graph_namespace_id"], 0,
        "graph-rebuild:stable-job-recovery", ttl_seconds=30,
    )
    assert database.release_job_graph_leases("stable-job-recovery") == 1
    assert not database.assert_graph_lease(
        project["id"], project["graph_namespace_id"], 0,
        rebuild_lease["lease_token"],
    )
