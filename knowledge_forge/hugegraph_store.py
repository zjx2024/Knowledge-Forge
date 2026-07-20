"""Dedicated HugeGraph physical schema and idempotent graph writer."""

from __future__ import annotations

import contextlib
import io
import logging
from datetime import datetime, timezone
from collections.abc import Iterator
from typing import Any

import requests
from pyhugegraph.client import PyHugeClient

from .config import Config
from .database import Database
from .graph_manifest import GraphManifestAccumulator, build_graph_manifest
from .graph_safety import GraphMutationContext, GraphSafetyError


# A missing customized vertex is the normal first-write path. The SDK logs that
# 404 at ERROR before returning control, so keep expected upsert probes quiet.
logging.getLogger("client").setLevel(logging.CRITICAL)


class HugeGraphStore:
    def __init__(self, database: Database):
        self.db = database
        self._writer_records: dict[tuple[str, int], GraphManifestAccumulator] = {}

    def client(self) -> PyHugeClient:
        with contextlib.redirect_stderr(io.StringIO()):
            return PyHugeClient(
                ip=Config.hugegraph_host,
                port=str(Config.hugegraph_port),
                graph=Config.hugegraph_graph,
                user=Config.hugegraph_user,
                pwd=Config.hugegraph_password,
                timeout=Config.hugegraph_timeout,
            )

    def available_graphs(self) -> list[str]:
        response = requests.get(f"http://{Config.hugegraph_host}:{Config.hugegraph_port}/graphs", timeout=Config.hugegraph_timeout)
        response.raise_for_status()
        return response.json().get("graphs") or []

    def ensure_schema(self) -> dict[str, Any]:
        if Config.hugegraph_graph not in self.available_graphs():
            raise RuntimeError(
                f"HugeGraph physical graph '{Config.hugegraph_graph}' does not exist. "
                "Install deploy/hugegraph/knowledge_forge.properties and run init-store.sh first."
            )
        client = self.client(); schema = client.schema()
        props = {
            "kf_namespace_id": "text", "kf_graph_generation": "int", "kf_policy_hash": "text",
            "kf_project_id": "int", "kf_project_key": "text", "kf_doc_id": "int", "kf_chunk_id": "int",
            "kf_entity_id": "int", "kf_mention_id": "int", "kf_assertion_id": "int", "kf_name": "text",
            "kf_normalized_name": "text", "kf_entity_type": "text", "kf_relation_type": "text",
            "kf_description": "text", "kf_source_url": "text", "kf_schema_version": "int",
            "kf_confidence": "double", "kf_updated_at": "text",
        }
        for name, kind in props.items():
            builder = schema.propertyKey(name)
            if kind == "int": builder.asInt()
            elif kind == "double": builder.asDouble()
            else: builder.asText()
            builder.ifNotExist().create()

        common = [
            "kf_namespace_id", "kf_graph_generation", "kf_policy_hash",
            "kf_project_id", "kf_project_key", "kf_updated_at",
        ]
        document_props = [*common, "kf_doc_id", "kf_name", "kf_source_url", "kf_schema_version"]
        chunk_props = [*common, "kf_doc_id", "kf_chunk_id", "kf_name", "kf_source_url", "kf_schema_version"]
        entity_props = [*common, "kf_entity_id", "kf_name", "kf_normalized_name", "kf_entity_type", "kf_description", "kf_schema_version"]
        for label, values, nullable in [
            ("KFDocument", document_props, ["kf_source_url"]),
            ("KFChunk", chunk_props, ["kf_name", "kf_source_url"]),
            ("KFEntity", entity_props, ["kf_description"]),
        ]:
            schema.vertexLabel(label).properties(*values).nullableKeys(*nullable).useCustomizeStringId().ifNotExist().create()

        edge_common = [
            "kf_namespace_id", "kf_graph_generation", "kf_policy_hash",
            "kf_project_id", "kf_project_key", "kf_doc_id", "kf_updated_at",
        ]
        schema.edgeLabel("kf_contains").sourceLabel("KFDocument").targetLabel("KFChunk").properties(*edge_common).nullableKeys("kf_doc_id").ifNotExist().create()
        mention_props = [*edge_common, "kf_chunk_id", "kf_mention_id", "kf_confidence", "kf_description"]
        schema.edgeLabel("kf_mentions").sourceLabel("KFChunk").targetLabel("KFEntity").properties(*mention_props).nullableKeys("kf_description").multiTimes().sortKeys("kf_mention_id").ifNotExist().create()
        relation_props = [*edge_common, "kf_chunk_id", "kf_assertion_id", "kf_relation_type", "kf_description", "kf_confidence", "kf_schema_version"]
        schema.edgeLabel("kf_related").sourceLabel("KFEntity").targetLabel("KFEntity").properties(*relation_props).nullableKeys("kf_description").multiTimes().sortKeys("kf_assertion_id").ifNotExist().create()

        indexes = [
            ("kfDocumentByNamespace", "V", "KFDocument", "kf_namespace_id"),
            ("kfDocumentByProject", "V", "KFDocument", "kf_project_id"),
            ("kfDocumentByDocument", "V", "KFDocument", "kf_doc_id"),
            ("kfChunkByNamespace", "V", "KFChunk", "kf_namespace_id"),
            ("kfChunkByProject", "V", "KFChunk", "kf_project_id"),
            ("kfChunkByDocument", "V", "KFChunk", "kf_doc_id"),
            ("kfEntityByNamespace", "V", "KFEntity", "kf_namespace_id"),
            ("kfEntityByProject", "V", "KFEntity", "kf_project_id"),
            ("kfEntityByName", "V", "KFEntity", "kf_normalized_name"),
            ("kfRelationByNamespace", "E", "kf_related", "kf_namespace_id"),
            ("kfRelationByProject", "E", "kf_related", "kf_project_id"),
            ("kfRelationByDocument", "E", "kf_related", "kf_doc_id"),
        ]
        for name, on, label, prop in indexes:
            builder = schema.indexLabel(name)
            (builder.onV(label) if on == "V" else builder.onE(label)).by(prop).secondary().ifNotExist().create()
        return {"graph": Config.hugegraph_graph, "vertex_labels": 3, "edge_labels": 3}

    @staticmethod
    def document_vertex_id(namespace_id: str, document_id: int) -> str:
        return f"kf:document:{namespace_id}:{document_id}"

    @staticmethod
    def chunk_vertex_id(namespace_id: str, chunk_id: int) -> str:
        return f"kf:chunk:{namespace_id}:{chunk_id}"

    @staticmethod
    def entity_vertex_id(namespace_id: str, entity_id: int) -> str:
        return f"kf:entity:{namespace_id}:{entity_id}"

    def _validate_mutation(
        self, project: dict[str, Any], mutation: GraphMutationContext | None,
    ) -> GraphMutationContext:
        if mutation is None:
            raise GraphSafetyError("HugeGraph mutation context is required")
        mutation.validate_for(project)
        if not self.db.assert_graph_lease(
            mutation.project_id, mutation.namespace_id, mutation.generation,
            mutation.lease_token,
        ):
            raise GraphSafetyError("PostgreSQL graph lease is stale or expired")
        return mutation

    @staticmethod
    def _manifest_properties(properties: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in properties.items()
            if key != "kf_updated_at"
        }

    def _record_writer(self, mutation: GraphMutationContext, record: dict[str, Any]) -> None:
        key = (mutation.namespace_id, mutation.generation)
        accumulator = self._writer_records.get(key)
        if accumulator is None:
            accumulator = GraphManifestAccumulator(
                namespace_id=mutation.namespace_id, project_id=mutation.project_id,
                graph_generation=mutation.generation, policy_hash=mutation.policy_hash,
            )
            self._writer_records[key] = accumulator
        accumulator.add(record, replace=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _upsert_vertex(graph, label: str, vertex_id: str, properties: dict[str, Any]) -> Any:
        try:
            current = graph.getVertexById(vertex_id)
        except Exception:
            current = None
        return graph.appendVertex(vertex_id, properties) if current else graph.addVertex(label, properties, id=vertex_id)

    @staticmethod
    def _iter_edges(graph, **conditions: Any) -> Iterator[Any]:
        """Read every matching edge through HugeGraph's cursor pagination."""
        page = None
        seen_pages: set[str] = set()
        while True:
            result = graph.getEdgeByPage(
                **conditions,
                limit=Config.hugegraph_page_size,
                page=page,
            )
            if not result:
                return
            if isinstance(result, tuple):
                items, next_page = result
            else:
                items, next_page = result, None
            yield from items or []
            if not next_page:
                return
            marker = str(next_page)
            if marker in seen_pages:
                raise RuntimeError(f"HugeGraph edge pagination repeated cursor: {marker}")
            seen_pages.add(marker)
            page = next_page

    @staticmethod
    def _iter_vertices(graph, label: str, properties: dict[str, Any]) -> Iterator[Any]:
        """Read every matching vertex through HugeGraph's cursor pagination."""
        page = None
        seen_pages: set[str] = set()
        while True:
            result = graph.getVertexByPage(
                label=label,
                properties=properties,
                limit=Config.hugegraph_page_size,
                page=page,
            )
            if not result:
                return
            if isinstance(result, tuple):
                items, next_page = result
            else:
                items, next_page = result, None
            yield from items or []
            if not next_page:
                return
            marker = str(next_page)
            if marker in seen_pages:
                raise RuntimeError(f"HugeGraph vertex pagination repeated cursor: {marker}")
            seen_pages.add(marker)
            page = next_page

    def replace_document(
        self, project: dict[str, Any], document: dict[str, Any], chunks: list[dict[str, Any]],
        *, mutation: GraphMutationContext,
    ) -> dict[str, int]:
        self._validate_mutation(project, mutation)
        self.ensure_schema()
        graph = self.client().graph()
        self._purge_document_graph(
            graph, mutation.namespace_id, int(document["id"]), keep_document=True,
        )
        return self._write_document_snapshot(graph, project, document, chunks, mutation)

    def write_document_snapshot(
        self,
        project: dict[str, Any],
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
        *, mutation: GraphMutationContext,
    ) -> dict[str, int]:
        """Write into an already-purged full rebuild without schema/purge repetition."""

        self._validate_mutation(project, mutation)
        return self._write_document_snapshot(
            self.client().graph(), project, document, chunks, mutation,
        )

    def _write_document_snapshot(
        self,
        graph,
        project: dict[str, Any],
        document: dict[str, Any],
        chunks: list[dict[str, Any]],
        mutation: GraphMutationContext,
    ) -> dict[str, int]:
        project_id, document_id = int(project["id"]), int(document["id"])
        namespace = mutation.namespace_id
        common = {
            "kf_namespace_id": namespace, "kf_graph_generation": mutation.generation,
            "kf_policy_hash": mutation.policy_hash, "kf_project_id": project_id,
            "kf_project_key": project["project_key"], "kf_updated_at": self._now(),
        }
        doc_vertex_id = self.document_vertex_id(namespace, document_id)
        doc_properties = {**common, "kf_doc_id": document_id,
            "kf_name": document["original_filename"], "kf_source_url": document["storage_path"],
            "kf_schema_version": int(project.get("active_schema_version") or 1)}
        self._upsert_vertex(graph, "KFDocument", doc_vertex_id, doc_properties)
        self._record_writer(mutation, {"kind": "vertex", "label": "KFDocument",
            "logical_id": doc_vertex_id, "properties": self._manifest_properties(doc_properties)})
        for chunk in chunks:
            chunk_id = int(chunk["id"]); vertex_id = self.chunk_vertex_id(namespace, chunk_id)
            chunk_properties = {**common, "kf_doc_id": document_id,
                "kf_chunk_id": chunk_id, "kf_name": str(chunk.get("chunk_key") or ""),
                "kf_source_url": document["storage_path"],
                "kf_schema_version": int(project.get("active_schema_version") or 1)}
            self._upsert_vertex(graph, "KFChunk", vertex_id, chunk_properties)
            self._record_writer(mutation, {"kind": "vertex", "label": "KFChunk",
                "logical_id": vertex_id, "properties": self._manifest_properties(chunk_properties)})
            edge_properties = {**common, "kf_doc_id": document_id}
            graph.addEdge("kf_contains", doc_vertex_id, vertex_id, edge_properties)
            self._record_writer(mutation, {"kind": "edge", "label": "kf_contains",
                "logical_id": f"contains:{doc_vertex_id}:{vertex_id}", "source": doc_vertex_id,
                "target": vertex_id, "direction": "out-in",
                "properties": self._manifest_properties(edge_properties)})
        return {"documents": 1, "chunks": len(chunks)}

    def sync_document_assertions(
        self, project: dict[str, Any], document: dict[str, Any],
        *, mutation: GraphMutationContext,
    ) -> dict[str, int]:
        self._validate_mutation(project, mutation)
        graph = self.client().graph(); project_id, document_id = int(project["id"]), int(document["id"])
        namespace = mutation.namespace_id
        common = {"kf_namespace_id": namespace, "kf_graph_generation": mutation.generation,
            "kf_policy_hash": mutation.policy_hash, "kf_project_id": project_id,
            "kf_project_key": project["project_key"], "kf_doc_id": document_id,
            "kf_updated_at": self._now()}
        mentions = self.db.list_document_mentions(document_id)
        relations = self.db.list_document_relations(document_id)
        entities = {int(item["canonical_entity_id"]): item for item in mentions}
        for entity_id, mention in entities.items():
            vertex_id = self.entity_vertex_id(namespace, entity_id)
            entity_properties = {"kf_namespace_id": namespace,
                "kf_graph_generation": mutation.generation, "kf_policy_hash": mutation.policy_hash,
                "kf_project_id": project_id, "kf_project_key": project["project_key"],
                "kf_entity_id": entity_id, "kf_name": mention["name"], "kf_normalized_name": mention["normalized_name"],
                "kf_entity_type": mention["entity_type"], "kf_description": mention.get("description") or "",
                "kf_schema_version": int(project.get("active_schema_version") or 1), "kf_updated_at": self._now()}
            self._upsert_vertex(graph, "KFEntity", vertex_id, entity_properties)
            self._record_writer(mutation, {"kind": "vertex", "label": "KFEntity",
                "logical_id": vertex_id, "properties": self._manifest_properties(entity_properties)})
            self.db.set_entity_graph_id(entity_id, vertex_id)
        for mention in mentions:
            source = self.chunk_vertex_id(namespace, int(mention["chunk_id"]))
            target = self.entity_vertex_id(namespace, int(mention["canonical_entity_id"]))
            properties = {**common,
                "kf_chunk_id": int(mention["chunk_id"]), "kf_mention_id": int(mention["id"]),
                "kf_confidence": float(mention["confidence"]), "kf_description": mention.get("description") or ""}
            edge = graph.addEdge("kf_mentions", source, target, properties)
            self._record_writer(mutation, {"kind": "edge", "label": "kf_mentions",
                "logical_id": f"mention:{mention['id']}", "source": source, "target": target,
                "direction": "out-in", "properties": self._manifest_properties(properties)})
            if edge: self.db.set_mention_graph_id(int(mention["id"]), str(edge.id))
        for relation in relations:
            source = self.entity_vertex_id(namespace, int(relation["source_entity_id"]))
            target = self.entity_vertex_id(namespace, int(relation["target_entity_id"]))
            properties = {**common,
                "kf_chunk_id": int(relation["chunk_id"]), "kf_assertion_id": int(relation["id"]),
                "kf_relation_type": relation["relation_type"], "kf_description": relation.get("description") or "",
                "kf_confidence": float(relation["confidence"]),
                "kf_schema_version": int(project.get("active_schema_version") or 1)}
            edge = graph.addEdge("kf_related", source, target, properties)
            self._record_writer(mutation, {"kind": "edge", "label": "kf_related",
                "logical_id": f"relation:{relation['id']}", "source": source, "target": target,
                "direction": "out-in", "properties": self._manifest_properties(properties)})
            if edge: self.db.set_relation_graph_id(int(relation["id"]), str(edge.id))
        return {"entities": len(entities), "mentions": len(mentions), "relations": len(relations)}

    def delete_document(
        self, project: dict[str, Any], document_id: int, *, mutation: GraphMutationContext,
    ) -> None:
        self._validate_mutation(project, mutation)
        graph = self.client().graph(); self._purge_document_graph(
            graph, mutation.namespace_id, document_id, keep_document=False,
        )

    def prune_project_entities(
        self, project: dict[str, Any], *, mutation: GraphMutationContext,
    ) -> int:
        self._validate_mutation(project, mutation)
        project_id = int(project["id"])
        active_ids = {int(item["id"]) for item in self.db.list_canonical_entities(project_id)}
        graph = self.client().graph(); removed = 0
        for vertex in self._iter_vertices(graph, "KFEntity", {"kf_namespace_id": mutation.namespace_id}):
            entity_id = int(vertex.properties.get("kf_entity_id") or 0)
            if entity_id not in active_ids:
                graph.removeVertexById(vertex.id); removed += 1
        return removed

    def delete_project(
        self, project: dict[str, Any], *, mutation: GraphMutationContext | None,
    ) -> None:
        mutation = self._validate_mutation(project, mutation)
        graph = self.client().graph()
        namespace = mutation.namespace_id
        key = (namespace, mutation.generation)
        previous = self._writer_records.pop(key, None)
        if previous is not None:
            previous.close()
        self._writer_records[key] = GraphManifestAccumulator(
            namespace_id=namespace, project_id=mutation.project_id,
            graph_generation=mutation.generation, policy_hash=mutation.policy_hash,
        )
        for label in ("kf_related", "kf_mentions", "kf_contains"):
            for edge in self._iter_edges(graph, label=label, properties={"kf_namespace_id": namespace}):
                graph.removeEdgeById(edge.id)
        for label in ("KFChunk", "KFDocument", "KFEntity"):
            for vertex in self._iter_vertices(graph, label, {"kf_namespace_id": namespace}):
                graph.removeVertexById(vertex.id)

    def _purge_document_graph(
        self, graph, namespace_id: str, document_id: int, keep_document: bool,
    ) -> None:
        for label in ("kf_related", "kf_mentions", "kf_contains"):
            for edge in self._iter_edges(graph, label=label, properties={
                "kf_namespace_id": namespace_id, "kf_doc_id": document_id,
            }):
                graph.removeEdgeById(edge.id)
        for vertex in self._iter_vertices(graph, "KFChunk", {
            "kf_namespace_id": namespace_id, "kf_doc_id": document_id,
        }):
            graph.removeVertexById(vertex.id)
        if not keep_document:
            for vertex in self._iter_vertices(graph, "KFDocument", {
                "kf_namespace_id": namespace_id, "kf_doc_id": document_id,
            }):
                graph.removeVertexById(vertex.id)

    def statistics(self, project: dict[str, Any]) -> dict[str, int]:
        graph = self.client().graph()
        namespace = str(project["graph_namespace_id"])
        result = {}
        for key, label in (("documents", "KFDocument"), ("chunks", "KFChunk"), ("entities", "KFEntity")):
            result[key] = sum(1 for _ in self._iter_vertices(graph, label, {"kf_namespace_id": namespace}))
        result["relations"] = sum(
            1 for _ in self._iter_edges(graph, label="kf_related", properties={"kf_namespace_id": namespace})
        )
        return result

    def expected_manifest(
        self, project: dict[str, Any], documents: list[dict[str, Any]],
        *, mutation: GraphMutationContext,
    ) -> dict[str, Any]:
        """Derive the exact generation manifest from current PostgreSQL facts."""

        self._validate_mutation(project, mutation)
        namespace = mutation.namespace_id
        project_id = int(project["id"])
        records: dict[str, dict[str, Any]] = {}

        def add(record: dict[str, Any]) -> None:
            records["|".join(str(record.get(key) or "") for key in (
                "kind", "label", "logical_id",
            ))] = record

        for document in documents:
            document_id = int(document["id"])
            common = {
                "kf_namespace_id": namespace, "kf_graph_generation": mutation.generation,
                "kf_policy_hash": mutation.policy_hash, "kf_project_id": project_id,
                "kf_project_key": project["project_key"],
            }
            doc_id = self.document_vertex_id(namespace, document_id)
            add({"kind": "vertex", "label": "KFDocument", "logical_id": doc_id,
                "properties": {**common, "kf_doc_id": document_id,
                    "kf_name": document["original_filename"],
                    "kf_source_url": document["storage_path"],
                    "kf_schema_version": int(project.get("active_schema_version") or 1)}})
            for chunk in self.db.list_document_chunks(document_id):
                chunk_id = int(chunk["id"])
                chunk_vertex = self.chunk_vertex_id(namespace, chunk_id)
                add({"kind": "vertex", "label": "KFChunk", "logical_id": chunk_vertex,
                    "properties": {**common, "kf_doc_id": document_id,
                        "kf_chunk_id": chunk_id, "kf_name": str(chunk.get("chunk_key") or ""),
                        "kf_source_url": document["storage_path"],
                        "kf_schema_version": int(project.get("active_schema_version") or 1)}})
                add({"kind": "edge", "label": "kf_contains",
                    "logical_id": f"contains:{doc_id}:{chunk_vertex}", "source": doc_id,
                    "target": chunk_vertex, "direction": "out-in",
                    "properties": {**common, "kf_doc_id": document_id}})
            mentions = self.db.list_document_mentions(document_id)
            for mention in mentions:
                entity_id = int(mention["canonical_entity_id"])
                entity_vertex = self.entity_vertex_id(namespace, entity_id)
                add({"kind": "vertex", "label": "KFEntity", "logical_id": entity_vertex,
                    "properties": {**common, "kf_entity_id": entity_id,
                        "kf_name": mention["name"],
                        "kf_normalized_name": mention["normalized_name"],
                        "kf_entity_type": mention["entity_type"],
                        "kf_description": mention.get("description") or "",
                        "kf_schema_version": int(project.get("active_schema_version") or 1)}})
                source = self.chunk_vertex_id(namespace, int(mention["chunk_id"]))
                properties = {**common, "kf_doc_id": document_id,
                    "kf_chunk_id": int(mention["chunk_id"]),
                    "kf_mention_id": int(mention["id"]),
                    "kf_confidence": float(mention["confidence"]),
                    "kf_description": mention.get("description") or ""}
                add({"kind": "edge", "label": "kf_mentions",
                    "logical_id": f"mention:{mention['id']}", "source": source,
                    "target": entity_vertex, "direction": "out-in", "properties": properties})
            for relation in self.db.list_document_relations(document_id):
                source = self.entity_vertex_id(namespace, int(relation["source_entity_id"]))
                target = self.entity_vertex_id(namespace, int(relation["target_entity_id"]))
                add({"kind": "edge", "label": "kf_related",
                    "logical_id": f"relation:{relation['id']}", "source": source,
                    "target": target, "direction": "out-in",
                    "properties": {**common, "kf_doc_id": document_id,
                        "kf_chunk_id": int(relation["chunk_id"]),
                        "kf_assertion_id": int(relation["id"]),
                        "kf_relation_type": relation["relation_type"],
                        "kf_description": relation.get("description") or "",
                        "kf_confidence": float(relation["confidence"]),
                        "kf_schema_version": int(project.get("active_schema_version") or 1)}})
        return build_graph_manifest(
            records.values(), namespace_id=namespace, project_id=project_id,
            graph_generation=mutation.generation, policy_hash=mutation.policy_hash,
        )

    def writer_manifest(
        self, project: dict[str, Any], *, mutation: GraphMutationContext,
    ) -> dict[str, Any]:
        self._validate_mutation(project, mutation)
        key = (mutation.namespace_id, mutation.generation)
        accumulator = self._writer_records.get(key)
        if accumulator is None:
            raise RuntimeError("writer manifest accumulator is missing")
        try:
            return accumulator.manifest()
        finally:
            accumulator.close()
            self._writer_records.pop(key, None)

    def actual_manifest(
        self, project: dict[str, Any], *, mutation: GraphMutationContext,
    ) -> dict[str, Any]:
        """Read the namespace through bounded cursor pages and hash exact graph facts."""

        self._validate_mutation(project, mutation)
        graph = self.client().graph()
        namespace = mutation.namespace_id
        def records():
            for label in ("KFDocument", "KFChunk", "KFEntity"):
                for vertex in self._iter_vertices(graph, label, {"kf_namespace_id": namespace}):
                    yield {
                        "kind": "vertex", "label": label, "logical_id": str(vertex.id),
                        "properties": self._manifest_properties(dict(vertex.properties or {})),
                    }
            for label in ("kf_contains", "kf_mentions", "kf_related"):
                for edge in self._iter_edges(
                    graph, label=label, properties={"kf_namespace_id": namespace},
                ):
                    properties = dict(edge.properties or {})
                    if label == "kf_contains":
                        logical_id = f"contains:{edge.outV}:{edge.inV}"
                    elif label == "kf_mentions":
                        logical_id = f"mention:{properties.get('kf_mention_id')}"
                    else:
                        logical_id = f"relation:{properties.get('kf_assertion_id')}"
                    yield {
                        "kind": "edge", "label": label, "logical_id": logical_id,
                        "source": str(edge.outV), "target": str(edge.inV),
                        "direction": "out-in",
                        "properties": self._manifest_properties(properties),
                    }
        return build_graph_manifest(
            records(), namespace_id=namespace, project_id=int(project["id"]),
            graph_generation=mutation.generation, policy_hash=mutation.policy_hash,
        )
