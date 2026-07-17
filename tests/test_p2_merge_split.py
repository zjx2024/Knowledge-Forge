from __future__ import annotations

from knowledge_forge.database import SCHEMA_SQL


def test_merge_split_schema_keeps_redirects_projection_and_reversible_payload():
    assert "CREATE TABLE IF NOT EXISTS canonical_entity_redirects" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS canonical_relation_projections" in SCHEMA_SQL
    assert "event_type TEXT NOT NULL" in SCHEMA_SQL
    assert "before_state JSONB NOT NULL" in SCHEMA_SQL
    assert "reversed_by_event_id" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS mention_relation_assertions" in SCHEMA_SQL
    assert "source_mention_key TEXT NOT NULL" in SCHEMA_SQL
    assert "target_mention_key TEXT NOT NULL" in SCHEMA_SQL


def test_relation_assertions_are_not_rewritten_by_merge_migration():
    merge_section = SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS entity_resolution_events", 1)[1]
    assert "UPDATE relation_assertions SET source_entity_id" not in merge_section
    assert "UPDATE relation_assertions SET target_entity_id" not in merge_section


def test_product_relation_queries_use_rebuildable_projection():
    from knowledge_forge.database import Database

    for method_name in ("list_project_relations", "list_document_relations",
                        "search_graph_text", "get_relations_by_ids"):
        source = __import__("inspect").getsource(getattr(Database, method_name))
        assert "canonical_relation_projections" in source
        assert "COALESCE(projection.source_entity_id,r.source_entity_id)" in source
        assert "COALESCE(projection.target_entity_id,r.target_entity_id)" in source
