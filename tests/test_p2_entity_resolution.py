from __future__ import annotations

from knowledge_forge.database import SCHEMA_SQL
from knowledge_forge.entity_shadow import EntityShadowState


def test_p2_schema_has_indexed_aliases_assignments_cannot_links_and_shadow_namespace():
    for table in (
        "canonical_entity_aliases", "entity_type_assertions", "entity_assignments",
        "entity_cannot_links", "entity_resolution_events", "entity_shadow_namespaces",
        "entity_shadow_canonicals", "entity_shadow_assignments",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in SCHEMA_SQL
    assert "USING gin(normalized_alias gin_trgm_ops)" in SCHEMA_SQL


def test_shadow_resolution_is_deterministic_stateful_and_isolated():
    mentions = [
        {"mention_id": "m1", "surface": "International Business Machines", "normalized_surface": "international business machines", "entity_type": "Organization"},
        {"mention_id": "m2", "surface": "IBM", "normalized_surface": "ibm", "entity_type": "Organization", "aliases": ["international business machines"]},
    ]
    first = EntityShadowState(namespace="ns", policy_version="entity-resolution-p2-shadow-v1")
    second = EntityShadowState(namespace="ns", policy_version="entity-resolution-p2-shadow-v1")
    first.replay(mentions)
    second.replay(mentions)
    assert first.snapshot() == second.snapshot()
    assert len(first.snapshot()["canonicals"]) == 1
    assert all(item["namespace"] == "ns" for item in first.snapshot()["assignments"])


def test_cannot_link_wins_before_exact_or_alias_merge():
    state = EntityShadowState(namespace="ns", policy_version="v")
    state.add_cannot_link("m1", "m2", "identifier conflict")
    state.replay([
        {"mention_id": "m1", "surface": "Model 1", "normalized_surface": "model 1", "entity_type": "Product"},
        {"mention_id": "m2", "surface": "Model 1", "normalized_surface": "model 1", "entity_type": "Product"},
    ])
    assert len(state.snapshot()["canonicals"]) == 2


def test_cross_type_exact_is_bounded_review_not_automatic_merge():
    state = EntityShadowState(namespace="ns", policy_version="v")
    state.replay([
        {"mention_id": "m1", "surface": "Mercury", "normalized_surface": "mercury", "entity_type": "Organization"},
        {"mention_id": "m2", "surface": "Mercury", "normalized_surface": "mercury", "entity_type": "Product"},
    ])
    snapshot = state.snapshot()
    assert len(snapshot["canonicals"]) == 2
    assert snapshot["review_candidates"] == [{
        "mention_id": "m2", "canonical_id": snapshot["canonicals"][0]["id"],
        "reason": "cross_type_exact_alias",
    }]
