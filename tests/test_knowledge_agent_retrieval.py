"""Strategy-adapter tests for the unified, provenance-bearing evidence layer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from knowledge_forge.agent.retrieval.graphrag_native import (
    GraphRAGNativeRetriever,
    choose_native_mode,
)
from knowledge_forge.agent.community_navigation import (
    KnowledgeForgeCommunityNavigator,
)
from knowledge_forge.agent.retrieval.knowledge_forge import KnowledgeForgeRetriever
from knowledge_forge.agent.planning import plan_retrieval
from knowledge_forge.graph_policy import GraphNotReadyError


class _KnowledgeForgeDatabase:
    def __init__(self) -> None:
        self.source_hash = "current-source"

    def agent_project_source_snapshot(self, project_id):
        assert project_id == 31
        return {
            "project_id": 31,
            "source_hash": self.source_hash,
            "document_count": 1,
            "chunk_count": 1,
            "embedding_count": 1,
            "embedding_model": "bge-m3:latest",
        }

    def get_chunk_provenance(self, project_id, chunk_ids):
        assert project_id == 31 and chunk_ids == [91]
        return {91: {
            "id": 91,
            "document_id": 13,
            "revision_id": "revision-current",
            "content": "受理机构应在五个工作日内审查申请材料。",
            "metadata": {"pages": [4]},
            "section_title": "审查期限",
            "original_filename": "regulation.pdf",
            "storage_path": "/private/source/regulation.pdf",
        }}


class _VectorSearch:
    def search(self, project, query, top_k):
        assert project["id"] == 31
        assert query == "审查需要多久？"
        assert top_k == 10
        return [{
            "chunk_id": 91,
            "content": "受理机构应在五个工作日内审查申请材料。",
            "dense_score": 0.88,
            "lexical_score": 0.5,
        }]


class _KnowledgeForgeService:
    def __init__(self) -> None:
        self.db = _KnowledgeForgeDatabase()
        self.vectors = _VectorSearch()
        self.search_called = False

    def _current_knowledge_forge_snapshot_reference(self, project_id):
        assert project_id == 31
        raise GraphNotReadyError("graph is stale")

    def search(self, *_args, **_kwargs):  # pragma: no cover - failure sentinel
        self.search_called = True
        raise AssertionError("stale graph must not enter graph-backed search")


def test_knowledge_forge_adapter_degrades_to_current_text_without_stale_graph():
    service = _KnowledgeForgeService()
    adapter = KnowledgeForgeRetriever(service)

    result = adapter.retrieve(
        run_id="run-kf", project={"id": 31},
        query="审查需要多久？", candidate_limit=10,
    )

    assert result.mode == "text_only"
    assert result.snapshot_ref.graph_available is False
    assert result.snapshot_ref.source_hash == "current-source"
    assert service.search_called is False
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.provenance_valid is True
    assert evidence.source_refs[0].document_url == "/api/documents/13/download"
    assert evidence.source_refs[0].page_label == "第 4 页"
    assert evidence.source_refs[0].revision_id == "revision-current"


def test_knowledge_forge_adapter_rechecks_source_snapshot_before_publish():
    service = _KnowledgeForgeService()
    adapter = KnowledgeForgeRetriever(service)
    result = adapter.retrieve(
        run_id="run-kf", project={"id": 31},
        query="审查需要多久？", candidate_limit=10,
    )

    adapter.assert_current(31, result.snapshot_ref)
    service.db.source_hash = "new-source"
    with pytest.raises(RuntimeError, match="内容发生变化"):
        adapter.assert_current(31, result.snapshot_ref)


class _PageMediaDatabase(_KnowledgeForgeDatabase):
    def get_chunk_provenance(self, project_id, chunk_ids):
        row = super().get_chunk_provenance(project_id, chunk_ids)[91]
        row["content"] = "图 1 展示了受理流程。"
        return {91: row}

    def agent_page_media_assets(self, project_id, document_ids):
        assert project_id == 31 and document_ids == [13]
        return [{
            "document_id": 13,
            "revision_id": "revision-current",
            "metadata": {
                "pages": [4],
                "media_assets": [{
                    "url": "http://files.test/media/flow.png",
                    "local_path": "/private/media/flow.png",
                    "width": 800,
                    "height": 500,
                    "title": "受理流程图",
                    "description": "图 1 展示受理流程。",
                    "knowledge_relevance": "substantive",
                    "sources": [{"page": 4, "location": "Im0.png"}],
                }],
            },
        }]


class _PageMediaService(_KnowledgeForgeService):
    def __init__(self) -> None:
        super().__init__()
        self.db = _PageMediaDatabase()
        self.vectors = _PageMediaVectorSearch()


class _PageMediaVectorSearch(_VectorSearch):
    def search(self, project, query, top_k):
        rows = super().search(project, query, top_k)
        rows[0]["content"] = "图 1 展示了受理流程。"
        return rows


def test_knowledge_forge_adapter_attaches_visual_from_exact_source_page():
    result = KnowledgeForgeRetriever(_PageMediaService()).retrieve(
        run_id="run-kf-media", project={"id": 31},
        query="审查需要多久？", candidate_limit=10,
    )

    source = result.evidence[0].source_refs[0]
    assert [item["title"] for item in source.media_assets] == ["受理流程图"]
    assert "local_path" not in source.media_assets[0]
    assert result.diagnostics["page_media_enrichment"] == {
        "status": "attached", "attached_references": 1,
        "attached_assets": 1,
    }


class _VisualAnchorDatabase(_KnowledgeForgeDatabase):
    def agent_visual_reference_chunks(
        self, project_id, patterns, *, query, limit,
    ):
        assert project_id == 31
        assert "%图%2%" in patterns
        assert "图 2" in query
        assert limit == 12
        return [{
            "id": 92,
            "document_id": 13,
            "revision_id": "revision-current",
            "content": (
                "图 2 展示审批流程：提交申请后由受理机构审查，"
                "审查通过后作出决定。"
            ),
            "metadata": {"pages": [5]},
            "section_title": "审批流程",
            "original_filename": "regulation.pdf",
            "storage_path": "/private/source/regulation.pdf",
            "mime_type": "application/pdf",
        }]


class _VisualAnchorVectorSearch:
    def search(self, project, query, top_k):
        assert project["id"] == 31
        assert query == "图 2 展示了什么？"
        assert top_k == 10
        return [{
            "chunk_id": 91,
            "content": "受理机构应在五个工作日内审查申请材料。",
            "dense_score": 0.88,
            "lexical_score": 0.5,
        }]


class _VisualAnchorService(_KnowledgeForgeService):
    def __init__(self) -> None:
        super().__init__()
        self.db = _VisualAnchorDatabase()
        self.vectors = _VisualAnchorVectorSearch()


def test_knowledge_forge_adapter_adds_exact_numbered_visual_source_chunk():
    result = KnowledgeForgeRetriever(_VisualAnchorService()).retrieve(
        run_id="run-kf-visual-anchor", project={"id": 31},
        query="图 2 展示了什么？", candidate_limit=10,
        plan=plan_retrieval(
            "图 2 展示了什么？", strategy="knowledge_forge",
            candidate_limit=10, activation="active",
        ),
    )

    exact = [
        item for item in result.evidence
        if item.metadata.get("chunk_id") == 92
    ]
    assert len(exact) == 1
    assert "visual_anchor" in exact[0].retrieval_channels
    assert exact[0].source_refs[0].pages == (5,)
    assert result.diagnostics["numbered_visual_recall"] == {
        "status": "matched",
        "matched_chunks": 1,
        "added_chunks": 1,
        "anchors": ["figure:2"],
    }


class _CommunityNavigationDatabase:
    def __init__(self) -> None:
        self.candidate_calls = []
        self.exact_relation_keys = []

    @staticmethod
    def community_run():
        return {
            "run_id": "community-current",
            "communities": [
                {
                    "key": "community-a", "level": 0,
                    "entity_keys": ["a", "b"],
                },
                {
                    "key": "community-b", "level": 0,
                    "entity_keys": ["c", "d"],
                },
                {
                    "key": "community-c", "level": 0,
                    "entity_keys": ["e", "f"],
                },
            ],
            "metadata": {
                "community_topic_associations": [{
                    "id": "topic-a-c",
                    "source_community_key": "community-a",
                    "target_community_key": "community-c",
                    "score": 0.94,
                    "representative_entity_matches": [{
                        "source_entity_key": "a",
                        "source_entity_name": "甲模块",
                        "target_entity_key": "e",
                        "target_entity_name": "相近主题模块",
                        "score": 0.95,
                    }],
                }],
            },
        }

    def get_knowledge_forge_community_run(
        self, project_id, generation, namespace_id, *, input_hash,
    ):
        assert (project_id, generation, namespace_id, input_hash) == (
            31, 2, "namespace-current", "community-input-current",
        )
        return self.community_run()

    def list_agent_knowledge_forge_community_relation_candidates(
        self, namespace_id, relation_run_id, entity_keys, *, scope, limit,
    ):
        assert (namespace_id, relation_run_id) == (
            "namespace-current", "relation-current",
        )
        self.candidate_calls.append((tuple(entity_keys), scope, limit))
        if scope == "cross" and "a" in entity_keys:
            return [{
                "id": "relation-bridge",
                "source_canonical_key": "a", "source_name": "甲模块",
                "relation_type": "supports",
                "target_canonical_key": "c", "target_name": "乙模块",
                "evidence_text": "甲模块为乙模块提供支持。", "metadata": {},
            }]
        if scope == "internal" and "e" in entity_keys:
            return [{
                "id": "relation-topic-fact",
                "source_canonical_key": "e", "source_name": "相近主题模块",
                "relation_type": "includes",
                "target_canonical_key": "f", "target_name": "主题子项",
                "evidence_text": "相近主题模块包括主题子项。", "metadata": {},
            }]
        return []

    def search_knowledge_forge_relations(
        self, namespace_id, relation_run_id, query, *, relation_keys, limit,
        **_kwargs,
    ):
        assert query == "甲模块与其他主题有什么联系？"
        self.exact_relation_keys = list(relation_keys)
        rows = {
            "relation-bridge": {
                "id": "relation-bridge", "document_id": 13,
                "revision_id": "revision-current", "chunk_id": 201,
                "source_canonical_key": "a", "source_name": "甲模块",
                "relation_type": "supports",
                "target_canonical_key": "c", "target_name": "乙模块",
                "evidence_text": "甲模块为乙模块提供支持。",
                "lexical_score": 0.4, "metadata": {},
            },
            "relation-topic-fact": {
                "id": "relation-topic-fact", "document_id": 13,
                "revision_id": "revision-current", "chunk_id": 202,
                "source_canonical_key": "e", "source_name": "相近主题模块",
                "relation_type": "includes",
                "target_canonical_key": "f", "target_name": "主题子项",
                "evidence_text": "相近主题模块包括主题子项。",
                "lexical_score": 0.2, "metadata": {},
            },
        }
        return [rows[key] for key in relation_keys if key in rows][:limit]


def _community_navigation_payload(*, relation_count=0):
    return {
        "entities": [],
        "relations": [
            {"id": f"baseline-{index}"} for index in range(relation_count)
        ],
        "community_reports": [],
        "graph_retrieval": {
            "seed_entities": [{
                "canonical_key": "a", "score": 1.0,
                "query_surface_match": True,
            }],
        },
    }


def test_community_navigation_resolves_fact_and_topic_routes_to_verified_relations():
    database = _CommunityNavigationDatabase()
    navigator = KnowledgeForgeCommunityNavigator(database)

    rows, diagnostics = navigator.expand(
        project_id=31, graph_generation=2,
        namespace_id="namespace-current", relation_run_id="relation-current",
        community_run_id="community-current",
        community_input_hash="community-input-current",
        query="甲模块与其他主题有什么联系？",
        payload=_community_navigation_payload(),
        max_communities=3, max_relations=8,
    )

    assert {item["id"] for item in rows} == {
        "relation-bridge", "relation-topic-fact",
    }
    assert set(database.exact_relation_keys) == {
        "relation-bridge", "relation-topic-fact",
    }
    assert diagnostics["status"] == "expanded"
    assert diagnostics["factual_links_used"] == 1
    assert diagnostics["topic_links_used"] == 1
    assert diagnostics["synthetic_evidence_count"] == 0
    assert all(item.get("community_navigation") for item in rows)


def test_community_navigation_does_not_expand_a_well_supported_local_query():
    database = _CommunityNavigationDatabase()
    rows, diagnostics = KnowledgeForgeCommunityNavigator(database).expand(
        project_id=31, graph_generation=2,
        namespace_id="namespace-current", relation_run_id="relation-current",
        community_run_id="community-current",
        community_input_hash="community-input-current",
        query="甲模块与其他主题有什么联系？",
        payload=_community_navigation_payload(relation_count=4),
        max_communities=3, max_relations=8,
    )

    assert rows == []
    assert database.candidate_calls == []
    assert diagnostics["status"] == "not_triggered"
    assert diagnostics["reason"] == "baseline_relation_recall_is_sufficient"


class _GraphReadyCommunityDatabase(_CommunityNavigationDatabase):
    def agent_project_source_snapshot(self, project_id):
        return {
            "project_id": project_id, "source_hash": "current-source",
            "document_count": 1, "chunk_count": 2, "embedding_count": 2,
            "embedding_model": "bge-m3:latest",
        }

    def get_chunk_provenance(self, project_id, chunk_ids):
        assert project_id == 31
        return {
            chunk_id: {
                "id": chunk_id, "document_id": 13,
                "revision_id": "revision-current",
                "content": (
                    "甲模块为乙模块提供支持。" if chunk_id == 201
                    else "相近主题模块包括主题子项。"
                ),
                "metadata": {"pages": [4 if chunk_id == 201 else 5]},
                "section_title": "模块关系",
                "original_filename": "architecture.pdf",
                "storage_path": "/private/source/architecture.pdf",
            }
            for chunk_id in chunk_ids
        }


class _GraphReadyCommunityService:
    def __init__(self):
        self.db = _GraphReadyCommunityDatabase()

    def _current_knowledge_forge_snapshot_reference(self, project_id):
        assert project_id == 31
        return {
            "graph": {"ready_generation": 2},
            "policy_summary": {
                "community_run_id": "community-current",
                "community_input_hash": "community-input-current",
                "knowledge_language": "zh",
            },
            "entity_namespace_id": "namespace-current",
            "relation_run_id": "relation-current",
            "namespace": {"embedding_model": "bge-m3:latest"},
        }

    def search(self, project_id, query, *, top_k, allow_report_scheduling):
        assert (project_id, top_k, allow_report_scheduling) == (31, 10, False)
        assert query == "甲模块与其他主题有什么联系？"
        return {
            "chunks": [], "entities": [], "relations": [],
            "community_reports": [],
            "graph_retrieval": _community_navigation_payload()["graph_retrieval"],
        }


def test_knowledge_forge_adapter_uses_only_source_backed_facts_from_community_routes():
    plan = plan_retrieval(
        "甲模块与其他主题有什么联系？",
        strategy="knowledge_forge", candidate_limit=10,
        activation="active",
    )
    result = KnowledgeForgeRetriever(_GraphReadyCommunityService()).retrieve(
        run_id="run-community", project={"id": 31},
        query="甲模块与其他主题有什么联系？", candidate_limit=10,
        plan=plan,
    )

    assert {item.evidence_type for item in result.evidence} == {"relation"}
    assert {item.metadata["record_key"] for item in result.evidence} == {
        "relation-bridge", "relation-topic-fact",
    }
    assert all(item.provenance_valid for item in result.evidence)
    assert all(
        "community_navigation" in item.retrieval_channels
        for item in result.evidence
    )
    assert all("内容相近" not in item.content for item in result.evidence)
    assert result.diagnostics["community_navigation"][
        "synthetic_evidence_count"
    ] == 0
    assert result.diagnostics["primary_probe"]["expansion_recommended"] is True


class _NativeRuntime:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_id = "native-run-1"

    def current_run(self, project_id, *, validate_bundle):
        assert project_id == 41 and validate_bundle is False
        return ({
            "run_id": self.run_id,
            "source_snapshot_hash": "native-source",
            "graphrag_version": "3.1.1",
            "manifest_sha256": "manifest-a",
        }, self.root)

    def _native_source_context(self, project_id, source_maps):
        assert project_id == 41
        assert source_maps[0]["input_document_id"] == "input-1"
        return {"input-1": {
            "document_id": 17,
            "filename": "architecture.md",
            "revision_id": "native-revision",
            "pages": [2],
            "page_label": "第 2 页",
            "section_title": "Task management",
            "media_assets": [],
            "table": None,
        }}


class _NativeService:
    def __init__(self, root: Path) -> None:
        self.graphrag_native_runtime = _NativeRuntime(root)

    def graphrag_native_query(self, project_id, *, mode, question):
        assert (project_id, mode, question) == (41, "basic", "What handles tasks?")
        return {
            "run_id": "native-run-1",
            "query_run_id": "native-query-1",
            "bundle_manifest_sha256": "manifest-a",
            "answer": "A candidate draft that is not evidence.",
            "context_records": [{
                "table": "text_units",
                "context_id": "0",
                "native_id": "text-unit-1",
                "record": {
                    "id": "0",
                    "text": "The task manager creates and schedules tasks.",
                },
                "source_map": {
                    "input_document_id": "input-1",
                    "document_id": 17,
                    "revision_id": "native-revision",
                    "page": 2,
                },
            }],
        }


def _write_native_tables(root: Path) -> None:
    (root / "output").mkdir(parents=True)
    (root / "kf-manifest").mkdir(parents=True)
    tables = {
        "entities.parquet": [],
        "relationships.parquet": [],
        "communities.parquet": [],
        "community_reports.parquet": [],
        "text_units.parquet": [{
            "id": "text-unit-1", "human_readable_id": 1,
            "text": "The task manager creates and schedules tasks.",
            "document_ids": ["input-1"],
        }],
    }
    for filename, rows in tables.items():
        table = pa.Table.from_pylist(rows) if rows else pa.table({"id": pa.array([], type=pa.string())})
        pq.write_table(table, root / "output" / filename)
    pq.write_table(pa.Table.from_pylist([{
        "input_document_id": "input-1",
        "document_id": 17,
        "revision_id": "native-revision",
        "page": 2,
    }]), root / "kf-manifest" / "source-map.parquet")


def test_graphrag_adapter_uses_context_as_evidence_and_answer_only_as_draft(tmp_path):
    _write_native_tables(tmp_path)
    service = _NativeService(tmp_path)
    adapter = GraphRAGNativeRetriever(service)

    result = adapter.retrieve(
        run_id="run-native", project={"id": 41},
        query="What handles tasks?", candidate_limit=10,
        requested_mode="basic",
    )

    assert result.mode == "basic"
    assert result.candidate_answer.startswith("A candidate draft")
    assert len(result.evidence) == 1
    assert result.evidence[0].evidence_type == "native_context"
    assert "candidate draft" not in result.evidence[0].content.lower()
    assert result.evidence[0].source_refs[0].document_id == 17
    assert result.evidence[0].source_refs[0].page_label == "第 2 页"
    assert result.diagnostics["candidate_answer_retained_as_draft_only"] is True


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("请概括全部文档的主要主题", "global"),
        ("What are the overall trends?", "global"),
        ("任务管理和调度器是什么关系？", "local"),
    ],
)
def test_native_auto_mode_is_bounded_and_deterministic(question, expected):
    assert choose_native_mode(question) == expected
    assert choose_native_mode(question, requested="basic") == "basic"


def test_graphrag_adapter_refuses_a_context_from_another_active_run(tmp_path):
    _write_native_tables(tmp_path)
    service = _NativeService(tmp_path)
    service.graphrag_native_query = lambda *_args, **_kwargs: {
        "run_id": "other-native-run", "context_records": [], "answer": "",
    }

    with pytest.raises(RuntimeError, match="不属于当前 Native Bundle"):
        GraphRAGNativeRetriever(service).retrieve(
            run_id="run-native", project={"id": 41}, query="question",
            candidate_limit=10, requested_mode="local",
        )


def test_graphrag_adapter_refuses_a_context_from_another_bundle_manifest(tmp_path):
    _write_native_tables(tmp_path)
    service = _NativeService(tmp_path)
    original = service.graphrag_native_query

    def mismatched(*args, **kwargs):
        payload = original(*args, **kwargs)
        payload["bundle_manifest_sha256"] = "manifest-b"
        return payload

    service.graphrag_native_query = mismatched
    with pytest.raises(RuntimeError, match="清单不一致"):
        GraphRAGNativeRetriever(service).retrieve(
            run_id="run-native", project={"id": 41},
            query="What handles tasks?", candidate_limit=10,
            requested_mode="basic",
        )
