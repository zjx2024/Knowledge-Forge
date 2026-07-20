from knowledge_forge.database import SCHEMA_SQL


def test_current_bootstrap_contains_destructive_ddl_characterization():
    assert "schema_migrations" not in SCHEMA_SQL
    assert "DROP COLUMN IF EXISTS ragflow_chunk_id" in SCHEMA_SQL
