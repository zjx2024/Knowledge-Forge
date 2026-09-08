import uuid

from knowledge_forge.graph_safety import GraphMutationContext
from knowledge_forge.hugegraph_store import HugeGraphStore


class _Label:
    def __init__(self, properties):
        self.properties = list(properties)


class _Builder:
    def __init__(self):
        self.added = []
        self.nullable = []
        self.append_calls = 0

    def properties(self, *names):
        self.added.extend(names)
        return self

    def nullableKeys(self, *names):
        self.nullable.extend(names)
        return self

    def append(self):
        self.append_calls += 1


class _Schema:
    def __init__(self):
        self.vertices = {"KFDocument": _Label(["kf_project_id"])}
        self.edges = {"kf_related": _Label(["kf_project_id"])}
        self.builders = {}

    def getVertexLabel(self, label):
        return self.vertices[label]

    def getEdgeLabel(self, label):
        return self.edges[label]

    def vertexLabel(self, label):
        return self.builders.setdefault(("vertex", label), _Builder())

    def edgeLabel(self, label):
        return self.builders.setdefault(("edge", label), _Builder())


def test_old_vertex_label_appends_only_missing_properties_as_nullable():
    schema = _Schema()

    missing = HugeGraphStore._append_missing_label_properties(
        schema,
        label_type="vertex",
        label="KFDocument",
        properties=["kf_project_id", "kf_namespace_id", "kf_graph_generation"],
    )

    builder = schema.builders[("vertex", "KFDocument")]
    assert missing == ["kf_namespace_id", "kf_graph_generation"]
    assert builder.added == missing
    assert builder.nullable == missing
    assert builder.append_calls == 1


def test_current_edge_label_does_not_issue_schema_update():
    schema = _Schema()
    schema.edges["kf_related"] = _Label(["kf_project_id", "kf_namespace_id"])

    missing = HugeGraphStore._append_missing_label_properties(
        schema,
        label_type="edge",
        label="kf_related",
        properties=["kf_project_id", "kf_namespace_id"],
    )

    assert missing == []
    assert ("edge", "kf_related") not in schema.builders


def test_every_edge_label_has_namespace_project_and_document_indexes():
    indexes = {
        (kind, label, prop)
        for _name, kind, label, prop in HugeGraphStore.INDEX_DEFINITIONS
    }

    for label in ("kf_contains", "kf_mentions", "kf_related"):
        assert ("E", label, "kf_namespace_id") in indexes
        assert ("E", label, "kf_project_id") in indexes
        assert ("E", label, "kf_doc_id") in indexes


class _ManifestDatabase:
    def assert_graph_lease(self, *_args, **_kwargs):
        return True

    def list_document_chunks(self, _document_id):
        return []

    def list_document_mentions(self, _document_id):
        return []

    def list_document_relations(self, _document_id):
        return []


def test_expected_manifest_is_strategy_scoped():
    namespace = str(uuid.uuid4())
    mutation = GraphMutationContext(
        namespace_id=namespace,
        project_id=7,
        generation=3,
        policy_hash="policy",
        lease_token="lease",
    )
    document = {
        "id": 11,
        "original_filename": "source.pdf",
        "storage_path": "http://example.invalid/source.pdf",
    }
    store = HugeGraphStore(_ManifestDatabase())
    project = {
        "id": 7,
        "project_key": "project-7",
        "graph_namespace_id": namespace,
        "graph_generation": 3,
        "active_schema_version": 1,
    }

    stable = store.expected_manifest(project, [document], mutation=mutation)
    enhanced = store.expected_manifest(
        {**project, "_strategy_key": "knowledge_forge"},
        [document],
        mutation=mutation,
    )

    assert stable["fact_count"] == enhanced["fact_count"] == 1
    assert stable["content_digest"] != enhanced["content_digest"]
