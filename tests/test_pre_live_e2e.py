from pathlib import Path
import importlib.util
import subprocess
import sys
from types import SimpleNamespace


def test_repository_exposes_single_isolated_e2e_runner():
    runner = Path("scripts/run_isolated_e2e.py")
    assert runner.exists()
    completed = subprocess.run(
        [sys.executable, str(runner), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0
    assert "--preflight-only" in completed.stdout
    assert "--execute" in completed.stdout


def test_hugegraph_guard_cli_is_available_and_read_only_by_default():
    guard = Path("scripts/hugegraph_test_guard.py")
    assert guard.exists()
    completed = subprocess.run(
        [sys.executable, str(guard), "--help"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0
    assert "preflight" in completed.stdout
    assert "arm" in completed.stdout
    assert "cleanup" in completed.stdout


def test_hugegraph_cleanup_collects_all_pages_before_any_mutation(monkeypatch):
    path = Path("scripts/hugegraph_test_guard.py")
    spec = importlib.util.spec_from_file_location("kf_hugegraph_guard_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    events = []

    class Graph:
        def getEdgeByPage(self, *, label, properties, limit, page):
            events.append(("read_edge", label, page))
            if page is None:
                return ([SimpleNamespace(id=f"{label}-1")], "next")
            return ([SimpleNamespace(id=f"{label}-2")], None)

        def getVertexByPage(self, *, label, properties, limit, page):
            events.append(("read_vertex", label, page))
            if page is None:
                return ([SimpleNamespace(id=f"{label}-1")], "next")
            return ([SimpleNamespace(id=f"{label}-2")], None)

        def removeEdgeById(self, identifier):
            events.append(("delete_edge", identifier))

        def removeVertexById(self, identifier):
            events.append(("delete_vertex", identifier))

    graph = Graph()
    monkeypatch.setattr(
        module, "_client", lambda _args: SimpleNamespace(graph=lambda: graph),
    )
    module._delete_namespace(SimpleNamespace(namespace="unused"), "namespace-a")
    first_delete = next(index for index, event in enumerate(events) if event[0].startswith("delete"))
    assert first_delete == 12
    assert all(event[0].startswith("read") for event in events[:first_delete])
    assert sum(event[0] == "delete_edge" for event in events) == 6
    assert sum(event[0] == "delete_vertex" for event in events) == 7
