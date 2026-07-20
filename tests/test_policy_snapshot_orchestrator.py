from knowledge_forge.database import SCHEMA_SQL


def test_current_snapshot_state_has_no_leased_item_orchestrator_characterization():
    assert "CREATE TABLE IF NOT EXISTS policy_snapshots" in SCHEMA_SQL
    assert "snapshot_work_items" not in SCHEMA_SQL
    assert "lease_token" not in SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS policy_snapshots", 1)[1].split(";", 1)[0]
