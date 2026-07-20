from knowledge_forge.database import SCHEMA_SQL


def test_current_relation_shadow_has_no_durable_work_queue_characterization():
    assert "CREATE TABLE IF NOT EXISTS relation_shadow_runs" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS relation_work_items" not in SCHEMA_SQL
