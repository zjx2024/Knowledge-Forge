from knowledge_forge.database import SCHEMA_SQL


def test_current_graph_state_has_no_content_manifest_characterization():
    assert "graph_content_manifests" not in SCHEMA_SQL
    assert "graph_leases" not in SCHEMA_SQL
