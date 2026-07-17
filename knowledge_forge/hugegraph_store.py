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


# A missing customized vertex is the normal first-write path. The SDK logs that
# 404 at ERROR before returning control, so keep expected upsert probes quiet.
logging.getLogger("client").setLevel(logging.CRITICAL)


class HugeGraphStore:
    def __init__(self, database: Database):
        self.db = database

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

        common = ["kf_project_id", "kf_project_key", "kf_updated_at"]
        document_props = [*common, "kf_doc_id", "kf_name", "kf_source_url", "kf_schema_version"]
        chunk_props = [*common, "kf_doc_id", "kf_chunk_id", "kf_name", "kf_source_url", "kf_schema_version"]
        entity_props = [*common, "kf_entity_id", "kf_name", "kf_normalized_name", "kf_entity_type", "kf_description", "kf_schema_version"]
        for label, values, nullable in [
            ("KFDocument", document_props, ["kf_source_url"]),
            ("KFChunk", chunk_props, ["kf_name", "kf_source_url"]),
            ("KFEntity", entity_props, ["kf_description"]),
        ]:
            schema.vertexLabel(label).properties(*values).nullableKeys(*nullable).useCustomizeStringId().ifNotExist().create()

        edge_common = ["kf_project_id", "kf_project_key", "kf_doc_id", "kf_updated_at"]
        schema.edgeLabel("kf_contains").sourceLabel("KFDocument").targetLabel("KFChunk").properties(*edge_common).nullableKeys("kf_doc_id").ifNotExist().create()
        mention_props = [*edge_common, "kf_chunk_id", "kf_mention_id", "kf_confidence", "kf_description"]
        schema.edgeLabel("kf_mentions").sourceLabel("KFChunk").targetLabel("KFEntity").properties(*mention_props).nullableKeys("kf_description").multiTimes().sortKeys("kf_mention_id").ifNotExist().create()
        relation_props = [*edge_common, "kf_chunk_id", "kf_assertion_id", "kf_relation_type", "kf_description", "kf_confidence", "kf_schema_version"]
        schema.edgeLabel("kf_related").sourceLabel("KFEntity").targetLabel("KFEntity").properties(*relation_props).nullableKeys("kf_description").multiTimes().sortKeys("kf_assertion_id").ifNotExist().create()

        indexes = [
            ("kfDocumentByProject", "V", "KFDocument", "kf_project_id"),
            ("kfDocumentByDocument", "V", "KFDocument", "kf_doc_id"),
            ("kfChunkByProject", "V", "KFChunk", "kf_project_id"),
            ("kfChunkByDocument", "V", "KFChunk", "kf_doc_id"),
            ("kfEntityByProject", "V", "KFEntity", "kf_project_id"),
            ("kfEntityByName", "V", "KFEntity", "kf_normalized_name"),
            ("kfRelationByProject", "E", "kf_related", "kf_project_id"),
            ("kfRelationByDocument", "E", "kf_related", "kf_doc_id"),
        ]
        for name, on, label, prop in indexes:
            builder = schema.indexLabel(name)
            (builder.onV(label) if on == "V" else builder.onE(label)).by(prop).secondary().ifNotExist().create()
        return {"graph": Config.hugegraph_graph, "vertex_labels": 3, "edge_labels": 3}

    @staticmethod
    def document_vertex_id(project_id: int, document_id: int) -> str:
        return f"kf:document:{project_id}:{document_id}"

    @staticmethod
    def chunk_vertex_id(project_id: int, chunk_id: int) -> str:
        return f"kf:chunk:{project_id}:{chunk_id}"

    @staticmethod
    def entity_vertex_id(project_id: int, entity_id: int) -> str:
        return f"kf:entity:{project_id}:{entity_id}"

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
                logging.warning("HugeGraph edge pagination returned a repeated cursor; stopping at %s", marker)
                return
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
                logging.warning("HugeGraph vertex pagination returned a repeated cursor; stopping at %s", marker)
                return
            seen_pages.add(marker)
            page = next_page

    def replace_document(self, project: dict[str, Any], document: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, int]:
        self.ensure_schema(); graph = self.client().graph()
        project_id, document_id = int(project["id"]), int(document["id"])
        self._purge_document_graph(graph, document_id, keep_document=True)
        common = {"kf_project_id": project_id, "kf_project_key": project["project_key"], "kf_updated_at": self._now()}
        doc_vertex_id = self.document_vertex_id(project_id, document_id)
        self._upsert_vertex(graph, "KFDocument", doc_vertex_id, {**common, "kf_doc_id": document_id,
            "kf_name": document["original_filename"], "kf_source_url": document["storage_path"],
            "kf_schema_version": int(project.get("active_schema_version") or 1)})
        for chunk in chunks:
            chunk_id = int(chunk["id"]); vertex_id = self.chunk_vertex_id(project_id, chunk_id)
            self._upsert_vertex(graph, "KFChunk", vertex_id, {**common, "kf_doc_id": document_id,
                "kf_chunk_id": chunk_id, "kf_name": str(chunk.get("chunk_key") or ""),
                "kf_source_url": document["storage_path"], "kf_schema_version": int(project.get("active_schema_version") or 1)})
            graph.addEdge("kf_contains", doc_vertex_id, vertex_id, {**common, "kf_doc_id": document_id})
        return {"documents": 1, "chunks": len(chunks)}

    def sync_document_assertions(self, project: dict[str, Any], document: dict[str, Any]) -> dict[str, int]:
        graph = self.client().graph(); project_id, document_id = int(project["id"]), int(document["id"])
        common = {"kf_project_id": project_id, "kf_project_key": project["project_key"], "kf_doc_id": document_id, "kf_updated_at": self._now()}
        mentions = self.db.list_document_mentions(document_id)
        relations = self.db.list_document_relations(document_id, publishable_only=True)
        entities = {int(item["canonical_entity_id"]): item for item in mentions}
        for entity_id, mention in entities.items():
            vertex_id = self.entity_vertex_id(project_id, entity_id)
            self._upsert_vertex(graph, "KFEntity", vertex_id, {"kf_project_id": project_id, "kf_project_key": project["project_key"],
                "kf_entity_id": entity_id, "kf_name": mention["name"], "kf_normalized_name": mention["normalized_name"],
                "kf_entity_type": mention["entity_type"], "kf_description": mention.get("description") or "",
                "kf_schema_version": int(project.get("active_schema_version") or 1), "kf_updated_at": self._now()})
            self.db.set_entity_graph_id(entity_id, vertex_id)
        for mention in mentions:
            edge = graph.addEdge("kf_mentions", self.chunk_vertex_id(project_id, int(mention["chunk_id"])),
                self.entity_vertex_id(project_id, int(mention["canonical_entity_id"])), {**common,
                "kf_chunk_id": int(mention["chunk_id"]), "kf_mention_id": int(mention["id"]),
                "kf_confidence": float(mention["confidence"]), "kf_description": mention.get("description") or ""})
            if edge: self.db.set_mention_graph_id(int(mention["id"]), str(edge.id))
        for relation in relations:
            edge = graph.addEdge("kf_related", self.entity_vertex_id(project_id, int(relation["source_entity_id"])),
                self.entity_vertex_id(project_id, int(relation["target_entity_id"])), {**common,
                "kf_chunk_id": int(relation["chunk_id"]), "kf_assertion_id": int(relation["id"]),
                "kf_relation_type": relation["relation_type"], "kf_description": relation.get("description") or "",
                "kf_confidence": float(relation["confidence"]), "kf_schema_version": int(project.get("active_schema_version") or 1)})
            if edge: self.db.set_relation_graph_id(int(relation["id"]), str(edge.id))
        return {"entities": len(entities), "mentions": len(mentions), "relations": len(relations)}

    def delete_document(self, document_id: int) -> None:
        graph = self.client().graph(); self._purge_document_graph(graph, document_id, keep_document=False)

    def prune_project_entities(self, project_id: int) -> int:
        active_ids = {int(item["id"]) for item in self.db.list_canonical_entities(project_id)}
        graph = self.client().graph(); removed = 0
        for vertex in self._iter_vertices(graph, "KFEntity", {"kf_project_id": project_id}):
            entity_id = int(vertex.properties.get("kf_entity_id") or 0)
            if entity_id not in active_ids:
                graph.removeVertexById(vertex.id); removed += 1
        return removed

    def delete_project(self, project_id: int) -> None:
        graph = self.client().graph()
        for edge in self._iter_edges(graph, label="kf_related", properties={"kf_project_id": project_id}):
            graph.removeEdgeById(edge.id)
        for label in ("KFChunk", "KFDocument", "KFEntity"):
            for vertex in self._iter_vertices(graph, label, {"kf_project_id": project_id}):
                graph.removeVertexById(vertex.id)

    def _purge_document_graph(self, graph, document_id: int, keep_document: bool) -> None:
        for edge in self._iter_edges(graph, label="kf_related", properties={"kf_doc_id": document_id}):
            graph.removeEdgeById(edge.id)
        for vertex in self._iter_vertices(graph, "KFChunk", {"kf_doc_id": document_id}):
            graph.removeVertexById(vertex.id)
        if not keep_document:
            for vertex in self._iter_vertices(graph, "KFDocument", {"kf_doc_id": document_id}):
                graph.removeVertexById(vertex.id)

    def statistics(self, project_id: int) -> dict[str, int]:
        graph = self.client().graph()
        result = {}
        for key, label in (("documents", "KFDocument"), ("chunks", "KFChunk"), ("entities", "KFEntity")):
            result[key] = sum(1 for _ in self._iter_vertices(graph, label, {"kf_project_id": project_id}))
        result["relations"] = sum(
            1 for _ in self._iter_edges(graph, label="kf_related", properties={"kf_project_id": project_id})
        )
        return result
