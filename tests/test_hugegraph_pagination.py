from types import SimpleNamespace

import pytest

from knowledge_forge.config import Config
from knowledge_forge.hugegraph_store import HugeGraphStore


class FakePagedGraph:
    def __init__(self):
        self.edge_calls = []
        self.vertex_calls = []

    def getEdgeByPage(self, **kwargs):
        self.edge_calls.append(kwargs)
        page = kwargs.get("page")
        if page is None:
            return [SimpleNamespace(id="e1"), SimpleNamespace(id="e2")], "edge-page-2"
        if page == "edge-page-2":
            return [SimpleNamespace(id="e3")], None
        raise AssertionError(f"unexpected edge page: {page}")

    def getVertexByPage(self, **kwargs):
        self.vertex_calls.append(kwargs)
        page = kwargs.get("page")
        if page is None:
            return [SimpleNamespace(id="v1")], "vertex-page-2"
        if page == "vertex-page-2":
            return [SimpleNamespace(id="v2"), SimpleNamespace(id="v3")], None
        raise AssertionError(f"unexpected vertex page: {page}")


def test_hugegraph_iterators_follow_all_cursor_pages():
    graph = FakePagedGraph()

    edges = list(HugeGraphStore._iter_edges(graph, label="kf_related", properties={"kf_project_id": 3}))
    vertices = list(HugeGraphStore._iter_vertices(graph, "KFEntity", {"kf_project_id": 3}))

    assert [item.id for item in edges] == ["e1", "e2", "e3"]
    assert [item.id for item in vertices] == ["v1", "v2", "v3"]
    assert [call["page"] for call in graph.edge_calls] == [None, "edge-page-2"]
    assert [call["page"] for call in graph.vertex_calls] == [None, "vertex-page-2"]
    assert all(call["limit"] == Config.hugegraph_page_size for call in graph.edge_calls)
    assert all(call["limit"] == Config.hugegraph_page_size for call in graph.vertex_calls)


class RepeatedCursorGraph:
    def __init__(self):
        self.calls = 0

    def getEdgeByPage(self, **kwargs):
        self.calls += 1
        return [SimpleNamespace(id=f"e{self.calls}")], "same-page"


def test_hugegraph_iterator_fails_closed_on_repeated_cursor():
    graph = RepeatedCursorGraph()
    with pytest.raises(RuntimeError, match="repeated cursor"):
        list(HugeGraphStore._iter_edges(graph, label="kf_related", properties={}))
    assert graph.calls == 2
