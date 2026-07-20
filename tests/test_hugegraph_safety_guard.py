import inspect

from knowledge_forge.hugegraph_store import HugeGraphStore


def test_current_destructive_scope_uses_database_integer_characterization():
    source = inspect.getsource(HugeGraphStore.delete_project)
    assert "project_id" in source
    assert "GRAPH_NAMESPACE_ID" not in source
