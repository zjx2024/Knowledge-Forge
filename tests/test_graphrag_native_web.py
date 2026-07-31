"""Native UI/API routing characterization without Hubble or legacy search."""

from __future__ import annotations

import json
import re
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from knowledge_forge import web
from knowledge_forge.construction_strategies import (
    ENHANCED_SHADOW_STRATEGY,
    GRAPHRAG_NATIVE_STRATEGY,
)
from knowledge_forge.graph_policy import GraphNotReadyError
from knowledge_forge.graphrag_native_runtime import GraphRAGNativeStale
from knowledge_forge.schema_evolution import BASE_SCHEMA
from knowledge_forge.service import KnowledgeForgeService


class _Database:
    def list_projects(self): return [{"id": 1, "name": "native", "llm_model": "llm", "embedding_model": "embed", "model_max_tokens": 32768, "chunk_max_tokens": 512}]
    def list_models(self): return []
    def ensure_project_strategy_graphs(self, _project_id): return []
    def list_documents(self, _project_id, *_args): return [{"id": 1, "project_id": 1, "original_filename": "doc.pdf", "file_size_bytes": 1, "status": "indexed", "active_revision_id": "r", "structure_ready_at": "now", "chunks_ready_at": "now", "updated_at": "now"}]
    def count_documents(self, _project_id): return 1
    def statistics(self, _project_id): return {"documents": 1, "chunks": 2}
    def list_graphrag_native_artifacts(self, _run_id): return [{"logical_name": "output/entities.parquet", "row_count": 2}, {"logical_name": "output/relationships.parquet", "row_count": 1}, {"logical_name": "output/text_units.parquet", "row_count": 3}]
    def get_active_schema(self, _project_id): return None
    def get_project_construction_strategy_selection(self, _project_id):
        return {
            "project_id": 1, "selected_strategy": GRAPHRAG_NATIVE_STRATEGY,
            "association_status": "bound", "generation": 1,
        }


class _Service:
    def __init__(self): self.db = _Database()
    def graphrag_native_status(self, _project_id): return {"fresh": True, "run": {"run_id": "run-1", "generation": 1, "graphrag_version": "3.1.1"}}
    def latest_job(self, *_args): return None
    def latest_active_job(self, *_args): return None
    def hubble_project_url(self, *_args): raise AssertionError("native graph view must not call Hubble")
    def _assert_project_strategy(self, _project_id, expected=None):
        assert expected in {None, GRAPHRAG_NATIVE_STRATEGY}
        return GRAPHRAG_NATIVE_STRATEGY


class _KnowledgeForgeDatabase(_Database):
    def get_project_construction_strategy_selection(self, _project_id):
        return {
            "project_id": 1, "selected_strategy": ENHANCED_SHADOW_STRATEGY,
            "association_status": "bound", "generation": 1,
        }

    def get_project_strategy_graph(self, _project_id, strategy_key):
        assert strategy_key == ENHANCED_SHADOW_STRATEGY
        return {
            "status": "ready", "generation": 1, "ready_generation": 1,
            "stats": {"documents": 1, "chunks": 2, "entities": 3, "relations": 1},
        }

    def get_active_schema(self, _project_id):
        return {
            "version": 1,
            "schema_json": {"entity_types": ["Thing"], "relation_types": ["rel"]},
        }


class _KnowledgeForgeService(_Service):
    def __init__(self): self.db = _KnowledgeForgeDatabase()

    def _assert_project_strategy(self, _project_id, expected=None):
        assert expected in {None, ENHANCED_SHADOW_STRATEGY}
        return ENHANCED_SHADOW_STRATEGY


class _KnowledgeForgeHttpService(_KnowledgeForgeService):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, int]] = []

    def knowledge_forge_graph(self, project_id, *, limit, include_isolates=True):
        self.calls.append(("knowledge-forge-graph", project_id))
        return {
            "strategy": ENHANCED_SHADOW_STRATEGY,
            "generation": 2,
            "total_entities": 2,
            "total_relationships": 1,
            "nodes": [
                {"id": "a", "title": "甲", "type": "Thing", "group": "Thing"},
                {"id": "b", "title": "乙", "type": "Thing", "group": "Thing"},
            ],
            "edges": [{
                "id": "r", "source": "a", "target": "b",
                "source_title": "甲", "target_title": "乙",
                "predicate": "关联", "evidence_text": "甲与乙有关。",
            }],
            "groups": [{"key": "Thing", "label": "Thing", "size": 2}],
            "truncated": False,
        }

    def search(self, project_id, query, top_k=None):
        self.calls.append(("knowledge-forge-search", project_id))
        assert query == "甲"
        return {
            "query": query,
            "strategy": ENHANCED_SHADOW_STRATEGY,
            "graph_generation": 2,
            "entity_namespace_id": "namespace-current",
            "relation_run_id": "relation-run-current",
            "chunks": [],
            "entities": [{
                "canonical_key": "a", "name": "甲", "entity_type": "Thing",
                "description": "当前图谱实体",
            }],
            "relations": [{
                "id": "fact-current", "source_name": "甲",
                "relation_type": "关联", "target_name": "乙",
                "description": "甲与乙有关。",
                "verification_status": "SUPPORTED", "evidence_count": 1,
                "storage_path": "/documents/current.pdf",
                "original_filename": "current.pdf",
            }],
        }


class _HttpService(_Service):
    """A narrow service double exercised through a real localhost HTTP server.

    This is deliberately not a direct handler invocation: it proves the
    request parsing, response statuses, JSON encoding and the public Native
    UI contract without reaching a running user's service or a model.
    """

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[str, int]] = []

    def submit_graphrag_native_build(self, project_id):
        self.calls.append(("build", project_id))
        return {"job_id": "native-job", "job_ids": ["native-job"], "scope": "project_full_snapshot"}

    def cancel_graphrag_native_build(self, project_id):
        self.calls.append(("cancel", project_id))
        return {"id": "native-job", "status": "cancelled"}

    def graphrag_native_graph(self, project_id, *, offset, limit, include_isolates=False):
        self.calls.append(("graph", project_id))
        return {
            "run_id": "run-1", "offset": offset, "limit": limit,
            "total_relationships": 1,
            "nodes": [{"title": "alpha"}, {"title": "beta"}],
            "edges": [{"source": "alpha", "target": "beta", "description": "rel", "weight": 1.0, "text_unit_ids": ["tu-1"]}],
            "truncated": False,
        }

    def graphrag_native_communities(self, project_id, *, offset, limit):
        self.calls.append(("communities", project_id))
        return {"run_id": "run-1", "total": 1, "items": [{"community": "0"}]}

    def graphrag_native_community_reports(self, project_id, *, offset, limit):
        self.calls.append(("reports", project_id))
        return {"run_id": "run-1", "total": 1, "items": [{"community": "0", "summary": "native"}]}

    def graphrag_native_query(self, project_id, *, mode, question):
        self.calls.append(("query", project_id))
        assert mode == "global"
        assert question == "What is connected?"
        return {
            "run_id": "run-1", "mode": mode, "answer": "native answer",
            "context_reference_count": 2,
            "context_records": [
                {
                    "table": "reports", "native_id": "report-1",
                    "record": {
                        "id": "0", "title": "供应链社区",
                        "summary": "社区摘要",
                    },
                },
                {
                    "table": "text_units", "native_id": "tu-1",
                    "record": {"id": "0", "text": "Alpha relates to Beta."},
                    "source_map": {"page": 2, "heading_path": ["章节一"]},
                    "source_document": {
                        "original_filename": "doc.pdf",
                        "download_path": "/api/documents/1/download",
                    },
                },
            ],
        }

    def graphrag_native_source_documents(
        self, project_id, *, document_id=None, offset=0, limit=100,
    ):
        self.calls.append(("source-documents", project_id))
        return {
            "project_id": project_id, "document_id": document_id, "total": 1,
            "offset": offset, "limit": limit,
            "items": [{"id": "native-document", "text": "Alpha relates to Beta."}],
        }

    def graphrag_native_artifact_catalog(self, project_id):
        self.calls.append(("artifacts", project_id))
        return {
            "run_id": "run-1",
            "artifacts": [{"logical_name": "output/text_units.parquet", "row_count": 1}],
        }

    def graphrag_native_artifact_rows(
        self, project_id, artifact_key, *, offset=0, limit=100,
    ):
        self.calls.append(("artifact-rows", project_id))
        return {
            "run_id": "run-1", "artifact": artifact_key,
            "logical_name": "output/text_units.parquet", "total": 1,
            "offset": offset, "limit": limit, "items": [{"id": "tu-1"}],
        }

    def search(self, *_args, **_kwargs):
        raise AssertionError("Native UI/API must not call legacy search")


def _http_json(url: str, *, method: str = "GET", form: dict[str, str] | None = None):
    data = urlencode(form).encode("utf-8") if form is not None else None
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_native_page_uses_native_build_route_and_not_hubble(monkeypatch):
    monkeypatch.setattr(web, "SERVICE", _Service())

    page = web.render_page(1, GRAPHRAG_NATIVE_STRATEGY)

    assert "/graphrag-native/build" in page
    assert "GraphRAG Native" in page
    assert "Hubble" not in page
    assert 'id="native-search-form"' in page
    assert 'id="search-form"' not in page
    assert "nativeSubtaskLabels" in web.SCRIPT
    assert "验证并发布原生产物" in web.SCRIPT
    # Native owns its visible construction contract.  In particular, the
    # first paint must not reuse the Stable task panel or Schema controls.
    assert "GraphRAG Native 原生构建" in page
    assert "准备源文档" in page
    assert "构建原生 bundle" in page
    assert "查看原生构建日志" in page
    assert page.index('id="build-project-knowledge"') < page.index(
        'id="view-native-artifacts"'
    )
    assert "GraphRAG Native 原生语义图" in page
    assert 'id="embedded-graph-browser"' in page
    assert 'data-ready="true"' in page
    assert "Native Documents" in page
    assert "预处理分块" not in page
    assert "新建项目" not in page
    assert '<a class="brand-link" href="/" aria-label="返回项目首页">' in page
    assert ">返回项目首页</a>" not in page
    assert "源文档解析" in page
    assert "Knowledge Forge Chunk、pgvector 或知识结构定义" in page
    assert 'id="view-native-artifacts"' in page
    assert "GraphRAG 3.1.1 · 独立 native bundle" not in page
    assert ">源文档</span>" in page
    assert ">TextUnits</span>" in page
    assert ">原生实体</span>" in page
    assert ">原生关系</span>" in page
    assert "源文档已准备" in page
    assert "验证并发布原生产物" in page
    assert '<span>HugeGraph 同步</span>' not in page
    assert "重新评估知识结构</button>" not in page
    # A completed run is available from the explicit build-log button, but its
    # old progress bar must not remain on the main page.
    assert 'id="native-runtime-details"' not in page
    assert "最近一次原生构建" not in page
    assert "查看原生构建日志" in page
    assert 'id="native-runtime-log"' not in page
    assert 'id="task-log"' not in page
    assert "生成 TextUnits、语义图与社区" in page
    assert "window.KF={projectId:1,strategyKey:\"graphrag_native\",taskProfile:\"graphrag_native\"" in page
    # Mode selection must not consume the flex row and collapse the question
    # input into a tiny blank rectangle.
    assert ".search-form select{width:160px;min-width:160px;flex:0 0 160px}" in web.CSS
    assert ".upload-form input,.search-form input{flex:1;min-width:0}" in web.CSS


def test_both_strategy_workspaces_use_four_section_sidebar_navigation(monkeypatch):
    monkeypatch.setattr(web, "SERVICE", _KnowledgeForgeService())
    knowledge_forge_page = web.render_page(1, ENHANCED_SHADOW_STRATEGY)

    monkeypatch.setattr(web, "SERVICE", _Service())
    native_page = web.render_page(1, GRAPHRAG_NATIVE_STRATEGY)

    for page in (knowledge_forge_page, native_page):
        assert 'class="workspace-layout"' in page
        assert 'class="workspace-sidebar"' in page
        assert 'data-workspace-target="documents"' in page
        assert 'data-workspace-target="build"' in page
        assert 'data-workspace-target="query"' in page
        assert 'data-workspace-target="settings"' in page
        assert 'data-workspace-panel="documents"' in page
        assert 'data-workspace-panel="build"' in page
        assert 'data-workspace-panel="query"' in page
        assert 'data-workspace-panel="settings"' in page
        assert 'id="project-settings-form"' in page
        assert 'id="document-upload-form"' in page
        assert 'form="document-upload-form"' in page
        assert 'id="project-settings-dialog"' not in page
        assert "项目策略已固定" not in page
        assert 'class="workspace-project-title"' in page
        assert 'class="panel project-panel"' not in page
        assert 'class="panel workspace-view"' not in page
        assert "activateWorkspaceView" in web.SCRIPT
        document_panel = page[
            page.index('<section id="workspace-documents"'):
            page.index('<section id="workspace-build"')
        ]
        settings_panel = page[page.index('<section id="workspace-settings"'):]
        assert 'id="processing-selects"' not in document_panel
        assert 'id="processing-selects"' in settings_panel

    assert ">文档预处理</b>" in knowledge_forge_page
    assert ">图谱与查询</b>" in knowledge_forge_page
    assert ">源文档准备</b>" in native_page
    assert ">语义图谱与问答</b>" in native_page
    assert 'id="search-form"' in knowledge_forge_page
    assert "联合检索" in knowledge_forge_page
    assert "重新评估知识结构" in knowledge_forge_page
    assert "<h2>知识结构定义</h2>" in knowledge_forge_page
    knowledge_document_panel = knowledge_forge_page[
        knowledge_forge_page.index('<section id="workspace-documents"'):
        knowledge_forge_page.index('<section id="workspace-build"')
    ]
    knowledge_build_panel = knowledge_forge_page[
        knowledge_forge_page.index('<section id="workspace-build"'):
        knowledge_forge_page.index('<section id="workspace-query"')
    ]
    assert "查看结构块列表" not in knowledge_document_panel
    assert "查看知识块列表" not in knowledge_document_panel
    assert "查看结构块列表" in knowledge_build_panel
    assert "查看知识块列表" in knowledge_build_panel
    assert 'class="schema-chip"' not in knowledge_build_panel
    assert knowledge_build_panel.index("<h2>知识结构定义</h2>") < knowledge_build_panel.index(
        "重新评估知识结构"
    )
    assert 'id="native-search-form"' not in knowledge_forge_page
    assert 'id="native-search-form"' in native_page
    assert "GraphRAG 知识问答" in native_page
    assert "详细问答（Local）" in native_page
    assert "全局总结（Global）" in native_page
    assert "主题深入分析（DRIFT）" in native_page
    assert "原文问答（Basic）" in native_page
    assert "输入需要基于当前知识库回答的问题" in native_page
    assert "重新评估知识结构" not in native_page
    assert 'id="search-form"' not in native_page
    for page in (knowledge_forge_page, native_page):
        build_panel = page[
            page.index('<section id="workspace-build"'):
            page.index('<section id="workspace-query"')
        ]
        # Preparation progress belongs directly below the preparation header,
        # before definition/artifact lists, rather than at the page bottom.
        assert build_panel.index('data-task-slot="preparation"') < build_panel.index(
            "process-artifacts"
        )


def test_preparation_updates_only_counters_owned_by_each_strategy():
    assert web._updating_statistic_keys(
        ENHANCED_SHADOW_STRATEGY, "content_preparation",
    ) == {"chunks"}
    assert web._updating_statistic_keys(
        GRAPHRAG_NATIVE_STRATEGY, "content_preparation",
    ) == set()
    assert web._updating_statistic_keys(
        ENHANCED_SHADOW_STRATEGY, "ingestion",
    ) == {"entities", "relations"}
    assert web._updating_statistic_keys(
        GRAPHRAG_NATIVE_STRATEGY, "graphrag_native_build",
    ) == {"chunks", "entities", "relations"}
    assert "updatingStatisticKeys(kind)" in web.SCRIPT
    assert (
        "kind==='content_preparation')return "
        "window.KF.taskProfile==='graphrag_native'?[]:['chunks']"
    ) in web.SCRIPT
    assert "if(kind==='ingestion')return['entities','relations']" in web.SCRIPT
    assert (
        "knowledgeBuildHidden=job.kind==='ingestion'"
        "&&window.KF.strategyKey==='enhanced_shadow'"
    ) in web.SCRIPT


def test_task_monitor_cancels_stale_pollers_and_clamps_visual_progress():
    assert "const generation=++activeMonitorGeneration" in web.SCRIPT
    assert "if(generation!==activeMonitorGeneration)return" in web.SCRIPT
    assert "renderSubtasks(job,monitorKey)" in web.SCRIPT
    assert "Math.max(renderedSubtaskProgress[key]||0,reportedProgress)" in (
        web.SCRIPT
    )


def test_job_log_opens_at_latest_line_and_live_updates_while_open():
    assert "function scrollJobLogToBottom()" in web.SCRIPT
    assert "jobHistory.scrollTop=jobHistory.scrollHeight" in web.SCRIPT
    assert web.SCRIPT.count("scrollJobLogToBottom();") == 1
    assert "function refreshJobLog(" in web.SCRIPT
    assert "setTimeout(()=>refreshJobLog(id,batch,generation),1000)" in (
        web.SCRIPT
    )
    assert "jobLogFollow=jobLogNearBottom()" in web.SCRIPT
    assert "stopJobLogPolling();jobDialog?.close()" in web.SCRIPT


def test_strategy_pages_separate_preparation_from_graph_actions(monkeypatch):
    monkeypatch.setattr(web, "SERVICE", _KnowledgeForgeService())
    knowledge_forge_page = web.render_page(1, ENHANCED_SHADOW_STRATEGY)

    monkeypatch.setattr(web, "SERVICE", _Service())
    native_page = web.render_page(1, GRAPHRAG_NATIVE_STRATEGY)

    for page in (knowledge_forge_page, native_page):
        preparation = page.split('id="workspace-build"', 1)[1].split(
            'id="workspace-query"', 1,
        )[0]
        graph_query = page.split('id="workspace-query"', 1)[1].split(
            'id="workspace-settings"', 1,
        )[0]
        assert 'id="prepare-project-documents"' in preparation
        assert 'id="view-preparation-log"' in preparation
        assert re.search(
            r'class="button primary" id="prepare-project-documents"',
            preparation,
        )
        assert re.search(
            r'class="section-heading workspace-view-heading[^"]*".*?'
            r'class="preparation-heading-actions".*?'
            r'id="prepare-project-documents".*?id="view-preparation-log"',
            preparation,
            re.S,
        )
        assert 'id="build-project-knowledge"' not in preparation
        assert 'id="view-knowledge-graph"' not in preparation
        assert 'data-stat="entities"' not in preparation
        assert 'data-stat="relations"' not in preparation
        assert 'id="build-project-knowledge"' in graph_query
        assert 'id="view-ingestion-log"' in graph_query
        assert re.search(
            r'class="button primary" id="build-project-knowledge"',
            graph_query,
        )
        assert 'id="view-knowledge-graph"' not in graph_query
        assert 'id="embedded-graph-browser"' in graph_query
        assert graph_query.index("search-form") < graph_query.index(
            'id="embedded-graph-browser"'
        )
        assert 'data-stat="entities"' in graph_query
        assert 'data-stat="relations"' in graph_query
        assert '<span class="workspace-view-kicker">构建</span>' not in page

    assert "文档处理产物" in knowledge_forge_page
    assert "源文档准备产物" in native_page
    document_panel = native_page.split('id="workspace-documents"', 1)[1].split(
        'id="workspace-build"', 1,
    )[0]
    assert "查看源结构" not in document_panel
    assert "查看 Native Documents" not in document_panel


def test_knowledge_structure_descriptions_are_localized_for_chinese_ui():
    panel = web._knowledge_structure_details(
        {"version": 1, "schema_json": BASE_SCHEMA},
        project_id=9,
        reassessment_available=True,
    )

    assert "自然人、人物或角色" in panel
    assert "主体是客体的组成部分或部件" in panel
    assert "方向从“部分”指向“整体”" in panel
    assert "Natural person or role" not in panel
    assert "The subject is a component or part of the object." not in panel
    assert 'data-reassess-schema="9"' in panel


def test_knowledge_forge_graph_reads_ready_snapshot_without_hubble():
    class SnapshotDatabase:
        def get_project_strategy_graph(self, project_id, strategy_key):
            assert project_id == 9
            assert strategy_key == ENHANCED_SHADOW_STRATEGY
            return {
                "status": "ready", "generation": 3, "ready_generation": 3,
                "policy_summary": {
                    "entity_namespace_id": "namespace-1",
                    "entity_metrics": {"relation_v2_run_id": "relation-run-1"},
                },
            }

        def get_entity_shadow_namespace(self, namespace_id):
            assert namespace_id == "namespace-1"
            return {
                "id": namespace_id, "project_id": 9, "status": "ready",
                "snapshot_complete": True, "embedding_model": "embed",
            }

        def get_shadow_relation_v2_run(self, run_id):
            assert run_id == "relation-run-1"
            return {
                "id": run_id, "project_id": 9, "status": "ready",
            }

        def load_entity_shadow_snapshot(self, namespace_id):
            assert namespace_id == "namespace-1"
            return {
                "canonicals": [
                    {
                        "canonical_key": "a", "name": "甲",
                        "normalized_name": "甲", "primary_type": "Thing",
                        "aliases": ["A"], "metadata": {"description": "甲实体"},
                    },
                    {
                        "canonical_key": "b", "name": "乙",
                        "normalized_name": "乙", "primary_type": "Thing",
                        "aliases": [], "metadata": {},
                    },
                    {
                        "canonical_key": "c", "name": "孤立实体",
                        "normalized_name": "孤立实体", "primary_type": "Other",
                        "aliases": [], "metadata": {},
                    },
                ],
                "relation_endpoints": [{
                    "relation_key": "r1", "source_canonical_key": "a",
                    "target_canonical_key": "b", "relation_type": "关联",
                    "evidence_text": "甲与乙有关。", "document_id": 1,
                    "chunk_id": 2, "metadata": {"confidence": 0.9},
                }],
            }

        def get_knowledge_forge_community_run(
            self, project_id, graph_generation, entity_namespace_id, **_kwargs,
        ):
            return {
                "run_id": "community-run", "community_count": 2,
                "communities": [
                    {
                        "key": "community-main", "label": "社区 0 · L0",
                        "level": 0, "community": 0, "size": 2,
                        "relationship_count": 1,
                        "entity_keys": ["a", "b"],
                        "summary": "两个相连实体。",
                    },
                    {
                        "key": "community-single", "label": "社区 1 · L0",
                        "level": 0, "community": 1, "size": 1,
                        "relationship_count": 0,
                        "entity_keys": ["c"],
                        "summary": "单点社区。",
                    },
                ],
            }

    service = SimpleNamespace(
        db=SnapshotDatabase(),
        _assert_project_strategy=lambda project_id, strategy: (
            project_id == 9 and strategy == ENHANCED_SHADOW_STRATEGY
        ),
    )
    service._current_knowledge_forge_snapshot_reference = (
        lambda project_id: KnowledgeForgeService
        ._current_knowledge_forge_snapshot_reference(service, project_id)
    )
    graph = KnowledgeForgeService.knowledge_forge_graph(
        service, 9, include_isolates=True,
    )

    assert graph["total_entities"] == 3
    assert graph["total_relationships"] == 1
    assert {node["title"] for node in graph["nodes"]} == {"甲", "乙", "孤立实体"}
    assert graph["edges"][0]["evidence_text"] == "甲与乙有关。"
    assert graph["edges"][0]["source_title"] == "甲"
    assert graph["community_ready"] is True
    assert graph["total_communities"] == 2
    assert {group["label"] for group in graph["groups"]} == {
        "社区 0 · L0", "未连接实体",
    }


def test_embedded_graph_loading_and_interaction_contract():
    loading = web._embedded_graph_browser(
        ENHANCED_SHADOW_STRATEGY, ready=False, active=True,
    )
    ready = web._embedded_graph_browser(
        GRAPHRAG_NATIVE_STRATEGY, ready=True, active=False,
    )

    assert "图谱正在构建" in loading
    assert "构建完成后将在这里自动加载" in loading
    assert "正在加载图谱" in ready
    assert 'data-graph-filter' in ready
    assert 'data-apply-graph-filter' in ready
    assert 'placeholder="筛选实体名称、类别、说明或别名"' in ready
    assert 'class="button primary" type="button" data-apply-graph-filter' in ready
    assert ready.index("data-graph-group-filter") < ready.index("data-graph-filter")
    assert ready.index("data-graph-filter") < ready.index("data-apply-graph-filter")
    assert 'role="tablist" aria-label="图谱浏览方式"' in ready
    assert 'class="embedded-graph-tab is-active"' in ready
    assert 'role="tab" aria-selected="true"' in ready
    assert 'role="tabpanel"' in ready
    assert "classList.toggle('is-active',active)" in web.SCRIPT
    assert "setAttribute('aria-selected',String(active))" in web.SCRIPT
    assert "data-graph-filter]').addEventListener('input'" not in web.SCRIPT
    assert "data-graph-network]').addEventListener('click'" in web.SCRIPT
    assert "data-graph-zoom-out" in ready
    assert "data-graph-zoom-in" in ready
    assert "data-graph-fit" in ready
    assert "function updateViewport()" in web.SCRIPT
    assert "label:'孤立实体'" in web.SCRIPT
    assert "graph-relation-card" in web.CSS
    assert "q('[data-graph-community-status]').hidden=hasSelection" in web.SCRIPT
    assert "q('[data-graph-groups]').hidden=hasSelection" in web.SCRIPT
    assert "/knowledge-forge/graph?limit=5000" in web.SCRIPT


def test_common_home_owns_creation_and_routes_bound_project_to_native_workspace(monkeypatch):
    monkeypatch.setattr(web, "SERVICE", _Service())

    page = web.render_home_page()

    assert "新建项目" in page
    assert 'name="strategy_key"' in page
    assert "Knowledge Forge" in page
    assert "GraphRAG Native" in page
    assert 'href="/projects/1/graphrag-native"' in page
    assert 'id="build-project-knowledge"' not in page
    assert "让分散的资料成为可检索、可连接、可追溯的知识资产" in page
    assert "精确治理型" in page
    assert "实体归一" in page
    assert "开放探索型" in page
    assert "Leiden 社区" in page
    assert "项目创建后不能切换构建方式" in page
    assert '<section class="panel strategy-overview"' in page
    assert 'id="create-strategy-key"' in page
    assert 'id="strategy-choice-help"' in page

    # Project selection is opened from the header instead of being pushed
    # below the long product introduction. Creation stays inside that list.
    header = page[page.index('<header class="home-header">'):page.index("</header>")]
    project_dialog = page[
        page.index('<dialog id="project-list-dialog"'):
        page.index('<div class="project-grid">')
    ]
    assert 'id="open-create-project"' not in header
    assert 'id="open-project-list"' in header
    assert 'id="open-create-project"' in project_dialog
    assert '<section class="panel project-list-panel"' not in page


def test_native_runtime_overview_keeps_verbose_events_in_on_demand_dialog():
    panel = web._native_runtime_details({
        "id": "run-job", "status": "running", "progress": 47,
        "log": [
            "GraphRAG Native：执行 Standard workflow",
            "GraphRAG Native：GraphRAG extract graph：4/9",
            "GraphRAG Native：GraphRAG workflow 完成：extract_graph",
            "GraphRAG Native 构建失败，禁止回退旧流程：raw model response must not render",
            "unrelated raw document content must not render",
        ],
    })

    assert "最近一次原生构建" in panel
    assert "47%" in panel
    assert "执行 Standard workflow" not in panel
    assert "GraphRAG extract graph：4/9" not in panel
    assert "GraphRAG workflow 完成：extract_graph" not in panel
    assert "raw model response" not in panel
    assert "raw document content" not in panel
    assert 'data-job="run-job"' in panel


def test_legacy_native_graph_link_redirects_to_inline_workspace(monkeypatch):
    monkeypatch.setattr(web, "SERVICE", _Service())
    handler = object.__new__(web.AppHandler)
    handler.path = "/graph/view?project=1&strategy=graphrag_native"
    captured = {}
    handler._redirect = lambda target, project_id=None: captured.update(
        target=target, project_id=project_id,
    )

    handler.do_GET()

    assert captured == {
        "target": "/projects/1/graphrag-native#query",
        "project_id": 1,
    }


def test_legacy_stable_graph_parameter_cannot_escape_fixed_knowledge_forge_strategy(
    monkeypatch,
):
    class FixedKnowledgeForgeService:
        def _assert_project_strategy(self, project_id, expected=None):
            assert project_id == 1
            assert expected == "enhanced_shadow"
            return "enhanced_shadow"

        def hubble_project_url(self, *_args):
            raise AssertionError("stable graph route must not reach Hubble")

    monkeypatch.setattr(web, "SERVICE", FixedKnowledgeForgeService())
    handler = object.__new__(web.AppHandler)
    handler.path = "/graph/view?project=1&strategy=stable"
    handler._project = lambda: 1
    captured = {}
    handler._redirect = lambda target, *_args: captured.update(target=target)

    handler.do_GET()

    assert captured["target"].startswith("/?kind=error&message=")


def test_statistics_fail_closed_for_unassigned_project(monkeypatch):
    class UnassignedService:
        db = SimpleNamespace(get_project=lambda _project_id: {"id": 1})

        def _assert_project_strategy(self, *_args):
            raise ValueError("项目尚未绑定知识图谱策略")

    monkeypatch.setattr(web, "SERVICE", UnassignedService())
    handler = object.__new__(web.AppHandler)
    handler.path = "/api/projects/1/statistics"
    captured = {}
    handler._json = lambda status, payload: captured.update(
        status=status, payload=payload,
    )

    handler.do_GET()

    assert captured == {
        "status": 409,
        "payload": {"error": "项目尚未绑定知识图谱策略"},
    }


def test_native_web_public_endpoints_work_over_real_http_without_legacy_fallback(monkeypatch):
    """Browser-facing Native page and APIs work as one isolated HTTP flow."""

    service = _HttpService()
    monkeypatch.setattr(web, "SERVICE", service)
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.AppHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/projects/1/graphrag-native", timeout=3) as response:
            page = response.read().decode("utf-8")
            assert response.status == 200
        assert "GraphRAG Native" in page
        assert 'id="build-project-knowledge"' in page
        assert 'id="native-search-form"' in page
        assert "GraphRAG 知识问答" in page
        assert "生成回答" in page
        assert "Hubble" not in page

        status, build = _http_json(base + "/api/projects/1/graphrag-native/build", method="POST")
        assert status == 202
        assert build["job_id"] == "native-job"

        status, native_status = _http_json(base + "/api/projects/1/graphrag-native/status")
        assert status == 200 and native_status["fresh"] is True
        status, graph = _http_json(base + "/api/projects/1/graphrag-native/graph?limit=10")
        assert status == 200 and graph["edges"][0]["source"] == "alpha"
        status, communities = _http_json(base + "/api/projects/1/graphrag-native/communities")
        assert status == 200 and communities["items"][0]["community"] == "0"
        status, reports = _http_json(base + "/api/projects/1/graphrag-native/community-reports")
        assert status == 200 and reports["items"][0]["summary"] == "native"
        status, source_documents = _http_json(
            base + "/api/projects/1/graphrag-native/source-documents?document_id=1"
        )
        assert status == 200 and source_documents["items"][0]["id"] == "native-document"
        status, artifact = _http_json(
            base + "/api/projects/1/graphrag-native/artifacts/text-units"
        )
        assert status == 200 and artifact["items"][0]["id"] == "tu-1"

        status, query = _http_json(
            base + "/api/projects/1/graphrag-native/query", method="POST",
            form={"mode": "global", "question": "What is connected?"},
        )
        assert status == 200 and query["answer"] == "native answer"
        assert query["context_records"][0]["record"]["title"] == "供应链社区"
        assert "回答依据" in web.SCRIPT
        assert "nativeContextCard" in web.SCRIPT
        assert "查看源文档" in web.SCRIPT

        try:
            _http_json(base + "/api/search?project=1&strategy=graphrag_native&q=forbidden")
        except HTTPError as exc:
            assert exc.code == 400
            error = json.loads(exc.read().decode("utf-8"))
            assert "不会回退旧检索" in error["error"]
        else:  # pragma: no cover - protects the no-fallback contract
            raise AssertionError("native strategy unexpectedly reached legacy search")
        assert ("build", 1) in service.calls
        assert ("query", 1) in service.calls
        assert ("graph", 1) in service.calls
        assert ("source-documents", 1) in service.calls
        assert ("artifact-rows", 1) in service.calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_knowledge_forge_inline_graph_api_works_over_real_http(monkeypatch):
    service = _KnowledgeForgeHttpService()
    monkeypatch.setattr(web, "SERVICE", service)
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.AppHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/projects/1/knowledge-forge#query", timeout=3) as response:
            page = response.read().decode("utf-8")
        assert "Knowledge Forge 知识图谱" in page
        assert 'placeholder="输入关键词、实体名称或检索描述"' in page
        assert "输入概念、实体、事件或问题" not in page
        assert 'id="embedded-graph-browser"' in page
        assert 'target="_blank"' not in page.split(
            'id="embedded-graph-browser"', 1,
        )[0].split('id="workspace-query"', 1)[1]

        status, graph = _http_json(
            base + "/api/projects/1/knowledge-forge/graph"
            "?limit=100&include_isolates=1"
        )
        assert status == 200
        assert graph["nodes"][0]["title"] == "甲"
        assert graph["edges"][0]["evidence_text"] == "甲与乙有关。"
        status, search = _http_json(
            base + "/api/search?project=1&q=" + urlencode({"q": "甲"})[2:]
        )
        assert status == 200
        assert search["entity_namespace_id"] == "namespace-current"
        assert search["relation_run_id"] == "relation-run-current"
        assert search["relations"][0]["verification_status"] == "SUPPORTED"
        assert search["relations"][0]["evidence_count"] == 1
        assert ("knowledge-forge-graph", 1) in service.calls
        assert ("knowledge-forge-search", 1) in service.calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_knowledge_forge_search_fails_closed_over_real_http(monkeypatch):
    service = _KnowledgeForgeHttpService()
    service.search = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        GraphNotReadyError("当前 Knowledge Forge 图谱已过期")
    )
    monkeypatch.setattr(web, "SERVICE", service)
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.AppHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(HTTPError) as captured:
            _http_json(base + "/api/search?project=1&q=stale")
        assert captured.value.code == 409
        error = json.loads(captured.value.read().decode("utf-8"))
        assert "图谱已过期" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
