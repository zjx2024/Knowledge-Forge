from __future__ import annotations

from types import SimpleNamespace
import uuid

import pytest

from knowledge_forge.graph_safety import (
    GraphEndpointIdentity,
    GraphMutationContext,
    GraphSafetyError,
    HugeGraphTestGuard,
)
from knowledge_forge.hugegraph_store import HugeGraphStore


class MutationRecorder:
    def __init__(self):
        self.mutations = 0

    def removeEdgeById(self, _identifier):
        self.mutations += 1

    def removeVertexById(self, _identifier):
        self.mutations += 1


def test_vertex_ids_and_purge_scope_use_stable_namespace_not_database_id():
    namespace = str(uuid.uuid4())
    assert HugeGraphStore.document_vertex_id(namespace, 17) == f"kf:document:{namespace}:17"
    assert HugeGraphStore.chunk_vertex_id(namespace, 19) == f"kf:chunk:{namespace}:19"
    assert HugeGraphStore.entity_vertex_id(namespace, 23) == f"kf:entity:{namespace}:23"


def test_missing_or_mismatched_mutation_context_causes_zero_graph_mutations(monkeypatch):
    namespace = str(uuid.uuid4())
    store = HugeGraphStore(SimpleNamespace())
    graph = MutationRecorder()
    monkeypatch.setattr(store, "client", lambda: SimpleNamespace(graph=lambda: graph))
    project = {"id": 7, "graph_namespace_id": namespace, "graph_generation": 3}

    with pytest.raises(GraphSafetyError):
        store.delete_project(project, mutation=None)
    with pytest.raises(GraphSafetyError):
        store.delete_project(project, mutation=GraphMutationContext(
            namespace_id=str(uuid.uuid4()), project_id=7, generation=3,
            policy_hash="p", lease_token="lease",
        ))
    assert graph.mutations == 0


def test_destructive_live_guard_requires_double_switch_physical_isolation_and_sentinel():
    namespace = str(uuid.uuid4())
    identity = GraphEndpointIdentity("http", "127.0.0.1", 8080, "disposable_graph")
    production = GraphEndpointIdentity("http", "127.0.0.1", 8080, "knowledge_forge")
    with pytest.raises(GraphSafetyError):
        HugeGraphTestGuard.preflight(
            identity=identity, production_identity=production, namespace_id=namespace,
            run_id="run-1", integration_enabled=True, destructive_enabled=False,
            endpoint_payload={"version": "1.5.0", "graph": "disposable_graph"},
        )

    armed = HugeGraphTestGuard.preflight(
        identity=identity, production_identity=production, namespace_id=namespace,
        run_id="run-1", integration_enabled=True, destructive_enabled=True,
        endpoint_payload={"version": "1.5.0", "graph": "disposable_graph"},
    )
    sentinel = armed.sentinel()
    assert sentinel["purpose"] == "test"
    assert sentinel["namespace_id"] == namespace
    assert armed.verify_sentinel(sentinel)
    assert not armed.verify_sentinel({**sentinel, "run_id": "other"})


def test_read_only_preflight_never_invokes_mutation_transport():
    calls = []
    identity = GraphEndpointIdentity("http", "127.0.0.1", 8080, "readonly_graph")
    guard = HugeGraphTestGuard.preflight(
        identity=identity,
        production_identity=GraphEndpointIdentity("http", "127.0.0.1", 8080, "prod"),
        namespace_id=str(uuid.uuid4()), run_id="read-only", integration_enabled=True,
        destructive_enabled=False, destructive=False,
        endpoint_payload={"version": "1.5.0", "graph": "readonly_graph"},
        mutation_transport=lambda *_args, **_kwargs: calls.append(True),
    )
    assert guard.destructive is False
    assert calls == []


def test_destructive_guard_arms_only_empty_graph_and_revalidates_cleanup_scope():
    namespace = str(uuid.uuid4())
    guard = HugeGraphTestGuard.preflight(
        identity=GraphEndpointIdentity("http", "127.0.0.1", 8080, "disposable_graph"),
        production_identity=GraphEndpointIdentity("http", "127.0.0.1", 8080, "prod"),
        namespace_id=namespace, run_id="run-2", integration_enabled=True,
        destructive_enabled=True,
        endpoint_payload={"version": "1.5.0", "graph": "disposable_graph"},
    )
    stored = {}
    token = guard.arm(
        read_counts=lambda: (0, 0), write_sentinel=lambda value: stored.update(value),
        read_sentinel=lambda: dict(stored),
    )
    deleted = []
    guard.cleanup(
        token, read_sentinel=lambda: dict(stored),
        delete_namespace=lambda value: deleted.append(value),
    )
    assert deleted == [namespace]

    with pytest.raises(GraphSafetyError):
        guard.authorize("wrong", read_sentinel=lambda: dict(stored))
    with pytest.raises(GraphSafetyError):
        guard.arm(
            read_counts=lambda: (1, 0), write_sentinel=lambda _value: None,
            read_sentinel=lambda: {},
        )
