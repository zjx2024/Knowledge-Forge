"""Dependency-light management UI for Knowledge Forge."""

from __future__ import annotations

import cgi
import html
import json
import logging
import mimetypes
import re
import threading
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .config import Config
from .chunking import CHUNK_STRATEGY_OPTIONS, PARSER_OPTIONS, docling_available, validate_processing_request
from .construction_strategies import (
    ENHANCED_SHADOW_STRATEGY,
    GRAPHRAG_NATIVE_STRATEGY,
    STABLE_STRATEGY,
    normalize_strategy,
    strategy_options,
)
from .graph_policy import GraphNotReadyError, is_current_graph
from .service import KnowledgeForgeService, TreatmentMaterializationRequired
from .graphrag_native_runtime import GraphRAGNativeStale, GraphRAGNativeUnavailable
from .runtime_safety import validate_runtime_config


LOGGER = logging.getLogger(__name__)
SERVICE: KnowledgeForgeService | None = None


def _esc(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def _time(value: object) -> str:
    return _esc(value).replace("+00:00", " UTC")[:19]


def _model_options(models: list[dict], kind: str, selected: str = "") -> str:
    options = []
    for model in models:
        if model["model_type"] != kind: continue
        name = str(model["name"]); chosen = " selected" if name == selected else ""
        options.append(f'<option value="{_esc(name)}"{chosen}>{_esc(name)}</option>')
    if not options:
        fallback = Config.default_embedding_model if kind == "embedding" else Config.default_llm_model
        options.append(f'<option value="{_esc(fallback)}">{_esc(fallback)}</option>')
    return "".join(options)


def _named_options(options: dict[str, str], selected: str = "", *, include_auto: bool = True) -> str:
    return "".join(
        f'<option value="{_esc(key)}"{" selected" if key == selected else ""}>{_esc(label)}</option>'
        for key, label in options.items() if include_auto or key != "auto"
    )


def _notice(kind: str, text: str) -> str:
    if not text: return ""
    good = kind == "ok"
    return f'<div class="notice {"success" if good else "error"}"><b>{"✓" if good else "!"}</b>{_esc(text)}</div>'


STATUS_NAMES = {
    "uploaded": "待构建", "queued": "排队中", "processing": "处理中",
    "indexed": "已入库", "failed": "失败", "cancelled": "已取消", "deleted": "已删除",
}
STAGE_NAMES = {
    "uploaded": "等待处理", "structure": "结构解析中", "structure_ready": "结构解析完成",
    "chunking": "知识分块中", "chunk_ready": "知识分块完成", "graph": "知识图谱构建中",
    "completed": "全部完成", "failed": "处理失败",
}


# Presentation is deliberately strategy-owned.  A construction strategy is not
# just a different endpoint: it may have a different source contract, runtime,
# artifacts and viewer.  Keeping the copy in one declarative map prevents a new
# strategy from accidentally inheriting Stable/HugeGraph terminology in a
# shared template.
STRATEGY_UI = {
    STABLE_STRATEGY: {
        "workspace_title": "知识库构建",
        "workspace_description": "上传后执行章节切分、向量索引、知识结构评估、实体归一与图谱写入。",
        "preparation_workspace_title": "文档预处理",
        "preparation_workspace_description": "解析源文档并生成后续构建可复用的结构块、知识块和向量索引。",
        "preparation_navigation_label": "文档预处理",
        "preparation_navigation_hint": "解析、分块与过程产物",
        "query_navigation_label": "图谱与查询",
        "query_navigation_hint": "构建、浏览与检索",
        "preparation_label": "预处理文档",
        "preparation_log_label": "查看预处理日志",
        "preparation_title": "生成结构块、Chunk 分片、向量索引和知识结构定义，不抽取实体关系，也不更新知识图谱",
        "build_label": "构建知识",
        "build_ready_title": "基于已完成预处理的文档抽取实体关系并构建知识图谱",
        "build_not_ready_title": "请先预处理文档，完成结构块、Chunk、向量和知识结构定义后再构建知识",
        "job_log_label": "查看知识构建日志",
        "viewer_label": "查看知识图谱",
        "task_title": "知识库构建",
        "task_profile": "entity_graph",
        "task_labels": {
            "structure": "结构解析",
            "chunking": "分块及向量索引",
            "schema_base": "知识结构定义和基础图数据",
            "extraction": "逐块抽取实体与关系",
            "graph_sync": "HugeGraph 同步",
        },
        "task_descriptions": {
            "structure": "读取源文档并恢复标题、正文、表格等结构。",
            "chunking": "生成可检索知识块并写入向量索引。",
            "schema_base": "确认项目允许的实体类别与关系规则。",
            "extraction": "从知识块识别实体、关系及其原文证据。",
            "graph_sync": "发布当前项目可浏览的知识图谱。",
        },
        "stat_labels": {"documents": "文档", "chunks": "知识块", "entities": "实体", "relations": "关系"},
        "prepared_stage": "已完成预处理，等待知识构建",
        "pending_stage": "等待文档预处理与知识构建",
        "chunk_action_label": "Chunk 分片",
        "show_schema_controls": True,
    },
    ENHANCED_SHADOW_STRATEGY: {
        "workspace_title": "Knowledge Forge 构建",
        "workspace_description": "复用已冻结的预处理结果与知识结构定义，执行实体归一、关系验证与隔离知识图谱构建。",
        "preparation_workspace_title": "Knowledge Forge 文档预处理",
        "preparation_workspace_description": "解析源文档，生成结构块、知识块、向量索引和知识结构定义，供后续图谱构建复用。",
        "preparation_navigation_label": "文档预处理",
        "preparation_navigation_hint": "解析、分块与过程产物",
        "query_navigation_label": "图谱与查询",
        "query_navigation_hint": "构建、浏览与检索",
        "preparation_label": "预处理文档",
        "preparation_log_label": "查看预处理日志",
        "preparation_title": "生成供 Knowledge Forge 复用的结构块、Chunk 分片、向量索引和知识结构定义",
        "build_label": "构建 Knowledge Forge",
        "build_ready_title": "基于已完成预处理的冻结结果构建 Knowledge Forge 图谱",
        "build_not_ready_title": "请先预处理文档，完成可复用的结构块、Chunk、向量和知识结构定义后再构建 Knowledge Forge",
        "job_log_label": "查看 Knowledge Forge 构建日志",
        "viewer_label": "查看 Knowledge Forge 图谱",
        "task_title": "Knowledge Forge 构建",
        "task_profile": "entity_graph",
        "task_labels": {
            "structure": "准备文档内容",
            "chunking": "生成知识块与向量",
            "schema_base": "确认知识结构定义",
            "extraction": "识别实体并验证关系",
            "graph_sync": "发布 Knowledge Forge 图谱",
        },
        "task_descriptions": {
            "structure": "读取或复用文档结构，确保后续处理使用同一版本。",
            "chunking": "生成可检索知识块和向量；已有结果会直接复用。",
            "schema_base": "冻结实体类别、关系语义、方向和发布规则。",
            "extraction": "识别实体并归一别名，生成候选关系后逐条核验原文证据。",
            "graph_sync": "把规范实体和通过验证的关系发布到当前项目图谱。",
        },
        "stat_labels": {"documents": "文档", "chunks": "知识块", "entities": "规范实体", "relations": "已验证关系"},
        "prepared_stage": "预处理已就绪，等待 Knowledge Forge 构建",
        "pending_stage": "等待文档预处理与 Knowledge Forge 构建",
        "chunk_action_label": "预处理 Chunk",
        "show_schema_controls": True,
    },
    GRAPHRAG_NATIVE_STRATEGY: {
        "workspace_title": "GraphRAG Native 原生构建",
        "workspace_description": "冻结已准备的源文档后，由 GraphRAG 3.1.1 独立生成 TextUnits、语义图、社区、报告和 LanceDB bundle；不调用 pgvector、实体关系管线或 HugeGraph。",
        "preparation_workspace_title": "GraphRAG Native 源文档准备",
        "preparation_workspace_description": "解析并冻结原生工作流使用的结构化 Documents；TextUnits、语义图和社区在下一页构建。",
        "preparation_navigation_label": "源文档准备",
        "preparation_navigation_hint": "解析、冻结与过程产物",
        "query_navigation_label": "语义图谱与问答",
        "query_navigation_hint": "构建、浏览与问答",
        "preparation_label": "准备源文档",
        "preparation_log_label": "查看源文档准备日志",
        "preparation_title": "解析并冻结结构化源文档；不会生成 Knowledge Forge Chunk、pgvector 或知识结构定义。",
        "build_label": "构建原生 bundle",
        "build_ready_title": "从冻结的源文档构建独立 GraphRAG Native 语义图、社区、报告与 LanceDB bundle，不调用 HugeGraph",
        "build_not_ready_title": "请先准备全部源文档，生成可冻结的结构化 Documents 后再构建原生 bundle",
        "job_log_label": "查看原生构建日志",
        "viewer_label": "查看原生语义图",
        "task_title": "GraphRAG Native 构建",
        "task_profile": "graphrag_native",
        "task_labels": {
            "structure": "准备原生构建输入",
            "chunking": "检查 GraphRAG 配置",
            "schema_base": "检查模型服务",
            "extraction": "生成 TextUnits、语义图与社区",
            "graph_sync": "验证并发布原生产物",
        },
        "task_descriptions": {
            "structure": "冻结本次使用的源文档版本，创建隔离运行目录。",
            "chunking": "检查 GraphRAG 版本、设置和提示模板是否完整。",
            "schema_base": "确认文本生成与向量模型可用且维度一致。",
            "extraction": "运行 GraphRAG 标准流程，生成 TextUnits、实体关系、社区、报告和 LanceDB。",
            "graph_sync": "独立校验全部产物，通过后发布不可变原生 bundle。",
        },
        "stat_labels": {"documents": "源文档", "chunks": "TextUnits", "entities": "原生实体", "relations": "原生关系"},
        "prepared_stage": "源文档已就绪，等待 GraphRAG Native 构建",
        "pending_stage": "等待源文档准备",
        "chunk_action_label": "Native Documents",
        "show_schema_controls": False,
    },
}


def _strategy_ui(strategy_key: str) -> dict[str, object]:
    """Return the complete presentation contract for a known strategy."""

    return STRATEGY_UI[normalize_strategy(strategy_key)]


def _schema_reassessment_available(strategy_key: str, active_job_id: str = "") -> bool:
    """Only an idle Knowledge Forge project owns the logical Schema workflow."""

    return (
        normalize_strategy(strategy_key) == ENHANCED_SHADOW_STRATEGY
        and not bool(active_job_id)
    )


def _updating_statistic_keys(strategy_key: str, job_kind: str) -> set[str]:
    """Return only counters that the active stage can actually change."""

    strategy_key = normalize_strategy(strategy_key)
    job_kind = str(job_kind or "")
    if job_kind == "content_preparation":
        # Knowledge Forge preparation creates reusable chunks, but it does not
        # extract entities or relations. Native source preparation creates
        # neither TextUnits nor graph records.
        return {"chunks"} if strategy_key != GRAPHRAG_NATIVE_STRATEGY else set()
    if job_kind == "ingestion":
        # Knowledge Forge ingestion consumes a frozen chunk set. Only its
        # canonical entities and verified relations change during this stage.
        return {"entities", "relations"}
    if job_kind == "graphrag_native_build":
        # Native TextUnits are not source-preparation artifacts; GraphRAG
        # creates them together with its semantic graph and communities.
        return {"chunks", "entities", "relations"}
    if job_kind == "graph_sync":
        return {"entities", "relations"}
    if job_kind == "document_cleanup":
        return {"chunks", "entities", "relations"}
    return set()


def _task_steps_html(strategy_key: str) -> str:
    """Render initial task rows from the selected strategy's own contract.

    JavaScript updates the same rows while a job is running.  Rendering them
    here as well avoids a first paint that describes a Native run as a legacy
    HugeGraph ingestion.
    """

    ui = _strategy_ui(strategy_key)
    labels = ui["task_labels"]
    descriptions = ui["task_descriptions"]
    assert isinstance(labels, dict)
    assert isinstance(descriptions, dict)
    return "".join(
        '<div class="task-step" data-subtask="{key}"><div class="task-step-head">'
        '<div><span>{label}</span><small>{description}</small></div>'
        '<b data-subtask-status>等待</b></div>'
        '<div class="progress"><i data-subtask-bar></i></div></div>'.format(
            key=_esc(key), label=_esc(labels[key]),
            description=_esc(descriptions[key]),
        )
        for key in ("structure", "chunking", "schema_base", "extraction", "graph_sync")
    )


def _native_runtime_details(job: dict | None) -> str:
    """Show Native progress only while a run is active.

    Completed and failed runs remain available through the explicit build-log
    button, so stale progress bars never look like a current operation.
    """

    job_id = str((job or {}).get("id") or "")
    status = str((job or {}).get("status") or "")
    if status not in {"queued", "running"}:
        return ""
    status_name = {
        "queued": "排队中", "running": "执行中",
    }.get(status, "暂无运行记录")
    progress = min(100, max(0, int((job or {}).get("progress") or 0)))
    return f"""<section id="native-runtime-details" class="workspace-section native-runtime-details"
      data-native-job="{_esc(job_id)}">
      <div class="section-heading"><div><h2>最近一次原生构建</h2>
        <p>主页面只显示状态和总进度；需要排查时再打开完整日志。</p></div>
        <button class="button secondary" type="button" data-view-job-log data-job="{_esc(job_id)}"{' disabled' if not job_id else ''}>查看完整原生运行日志</button>
      </div>
      <div class="native-runtime-status"><b id="native-runtime-status">{_esc(status_name)}</b><span id="native-runtime-progress-text">{progress}%</span></div>
      <div class="progress native-runtime-progress"><i id="native-runtime-progress" style="width:{progress}%"></i></div>
    </section>"""


def _strategy_statistics(
    project_id: int, strategy_key: str, graph_state: dict | None,
) -> dict[str, int]:
    if SERVICE.db.count_documents(project_id) == 0:
        return {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}
    if strategy_key == STABLE_STRATEGY:
        return SERVICE.db.statistics(project_id)
    if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
        document_count = SERVICE.db.count_documents(project_id)
        try:
            status = SERVICE.graphrag_native_status(project_id)
        except Exception:
            return {"documents": document_count, "chunks": 0, "entities": 0, "relations": 0}
        run = status.get("run") if isinstance(status, dict) else None
        counts = {"entities": 0, "relations": 0, "chunks": 0}
        if run and status.get("fresh"):
            for artifact in SERVICE.db.list_graphrag_native_artifacts(str(run["run_id"])):
                name = str(artifact.get("logical_name") or "")
                if name == "output/entities.parquet": counts["entities"] = int(artifact.get("row_count") or 0)
                elif name == "output/relationships.parquet": counts["relations"] = int(artifact.get("row_count") or 0)
                elif name == "output/text_units.parquet": counts["chunks"] = int(artifact.get("row_count") or 0)
        return {"documents": document_count, **counts}
    current = SERVICE.db.statistics(project_id)
    graph_status = str((graph_state or {}).get("status") or "")
    graph_current = (
        graph_status == "ready"
        and int((graph_state or {}).get("generation") or 0)
        == int((graph_state or {}).get("ready_generation") or 0)
    )
    if not graph_current:
        return {
            "documents": int(current.get("documents") or 0),
            "chunks": int(current.get("chunks") or 0),
            "entities": 0,
            "relations": 0,
        }
    stats = (graph_state or {}).get("stats") or {}
    return {
        "documents": int(stats.get("documents") or 0),
        "chunks": int(stats.get("chunks") or 0),
        "entities": int(stats.get("entities") or 0),
        "relations": int(stats.get("relations") or 0),
    }


def _document_rows(documents: list[dict], strategy_key: str) -> str:
    rows: list[str] = []
    ui = _strategy_ui(strategy_key)
    for item in documents:
        document_id = int(item["id"])
        stable_ready = bool(item.get("active_revision_id")) and item.get("status") == "indexed"
        if strategy_key == STABLE_STRATEGY:
            status = str(item.get("strategy_status") or "uploaded")
            stage = STAGE_NAMES.get(
                item.get("processing_stage"), item.get("processing_stage") or "等待构建",
            )
        else:
            # Native and experimental strategies consume the stable prepared
            # revision as input.  Their per-document lifecycle must not be
            # presented as the Stable ingestion lifecycle.
            status = str(item.get("strategy_status") or ("indexed" if stable_ready else "uploaded"))
            stage = str(ui["prepared_stage"] if stable_ready else ui["pending_stage"])
        error = "" if strategy_key == GRAPHRAG_NATIVE_STRATEGY else str(item.get("strategy_error_message") or "")
        status_label = STATUS_NAMES.get(status, status)
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
            status_label = "源文档已准备" if stable_ready else "待准备"
        # Strategy-owned process artifacts live on the second workspace page;
        # document management is limited to source-file lifecycle actions.
        process_actions = ""
        rows.append(
            f"""<tr><td><strong>{_esc(item['original_filename'])}</strong></td>
            <td>{item['file_size_bytes']/1024:.1f} KB</td><td>
              <span class="status status-{_esc(status)}" data-document-status="{document_id}">{_esc(status_label)}</span>
              <small data-stage-label="{document_id}">{_esc(stage)}</small>
              {f'<small>{_esc(item.get("effective_parser"))} · {_esc(item.get("effective_chunk_strategy"))}</small>' if item.get('effective_parser') else ''}
              {f'<small class="failure">{_esc(error)}</small>' if error else ''}
            </td><td>{_time(item.get('strategy_updated_at') or item['updated_at'])}</td>
            <td class="actions">{process_actions}
              <a class="button small secondary" href="/api/documents/{document_id}/download">下载</a>
              <button class="icon-button danger" title="删除文档" data-delete="{document_id}">×</button>
            </td></tr>"""
        )
    return "".join(rows) or '<tr><td colspan="5" class="empty">该项目尚未上传文档。</td></tr>'


def _schema_item(item: object) -> dict[str, object]:
    """Normalize legacy string entries and current metadata-rich definitions."""

    if isinstance(item, dict):
        return item
    return {"name": str(item or ""), "description": ""}


def _schema_type_names(value: object) -> str:
    values = value if isinstance(value, list) else []
    names = [str(item) for item in values if str(item).strip()]
    return "任意类型" if not names or names == ["*"] else "、".join(names)


SCHEMA_UI_ZH = {
    "Fallback concept or named object": "通用概念，或无法归入更具体类别的命名对象。",
    "Natural person or role": "自然人、人物或角色。",
    "Organization, institution or enterprise": "组织、机构或企业。",
    "Physical or administrative location": "物理地点或行政区域。",
    "Product, device, component or material": "产品、设备、组件或材料。",
    "Technology, method, process or standard": "技术、方法、流程或标准。",
    "Event, activity, state change or incident": "事件、活动、状态变化或事故。",
    "General semantic relation.": "主体与客体之间存在可由原文证据支持的通用语义关联。",
    "The subject is a component or part of the object.": "主体是客体的组成部分或部件。",
    "The subject is a member of or owned by the object.": "主体是客体的成员，或归客体所有。",
    "The subject is physically or administratively located in the object.": "主体在物理位置或行政区划上位于客体之内。",
    "The subject applies or consumes the object.": "主体使用、采用或消耗客体。",
    "The subject creates or outputs the object.": "主体创建、生产或输出客体。",
    "The subject requires the object for operation or existence.": "主体的运行或存在依赖客体。",
    "The subject brings about the object.": "主体导致或促成客体所表示的结果。",
    "The subject is semantically related to the object; no inverse role is implied.": "主体与客体存在语义关联；该关系不隐含反向角色。",
    "Orient from the part as subject to the whole as object.": "方向从“部分”指向“整体”：部分作为主体，整体作为客体。",
    "Orient from the member or owned item to the owning object.": "方向从“成员或被拥有对象”指向“所属或拥有方”。",
    "Orient from the located subject to the containing location.": "方向从“被定位对象”指向“其所在位置”。",
    "Orient from the user as subject to the used object.": "方向从“使用方”指向“被使用对象”。",
    "Orient from the producer as subject to the produced output.": "方向从“生产或创建方”指向“产出对象”。",
    "Orient from the dependent subject to the dependency object.": "方向从“依赖方”指向“被依赖对象”。",
    "Orient from the cause as subject to the resulting object.": "方向从“原因”指向“由其导致的结果”。",
}


def _schema_ui_zh(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return SCHEMA_UI_ZH.get(text, text or fallback)


def _knowledge_structure_details(
    schema: dict | None,
    *,
    project_id: int,
    reassessment_available: bool,
) -> str:
    """Render the active logical schema as an explicit user-facing definition."""

    reassess_button = (
        f"""<button class="button primary" type="button"
          data-reassess-schema="{_esc(project_id)}"
          {'disabled' if not reassessment_available else ''}>重新评估知识结构</button>"""
    )
    if not schema:
        return """<div class="workspace-section knowledge-structure">
          <div class="section-heading"><div><h2>知识结构定义</h2>
            <p>预处理文档后会在这里展示实体类别、关系类别、方向和发布规则。</p></div>
            {button}</div>
          <div class="empty-state compact-empty">尚未生成知识结构定义。</div>
        </div>""".format(button=reassess_button)
    data = schema.get("schema_json") if isinstance(schema.get("schema_json"), dict) else {}
    entities = [_schema_item(item) for item in data.get("entity_types", [])]
    relations = [_schema_item(item) for item in data.get("relation_types", [])]
    entity_cards = "".join(
        f"""<article class="definition-card"><div class="definition-card-head">
          <strong>{_esc(item.get("name") or "未命名类别")}</strong><span>实体类别</span></div>
          <p>{_esc(_schema_ui_zh(item.get("description_zh") or item.get("description"), "暂无类别说明"))}</p></article>"""
        for item in entities
    ) or '<div class="empty-state compact-empty">当前版本未定义实体类别。</div>'
    publish_labels = {"auto": "自动发布", "review_only": "仅供审核", "disabled": "停用"}
    relation_cards = "".join(
        f"""<article class="definition-card relation-definition"><div class="definition-card-head">
          <strong>{_esc(item.get("name") or "未命名关系")}</strong>
          <span>{_esc(publish_labels.get(str(item.get("publish_mode") or ""), item.get("publish_mode") or "发布规则未设置"))}</span></div>
          <p>{_esc(_schema_ui_zh(item.get("semantic_definition_zh") or item.get("semantic_definition") or item.get("description"), "暂无关系说明"))}</p>
          <dl><div><dt>允许方向</dt><dd>{_esc(_schema_type_names(item.get("domain_types") or item.get("source_types")))}
            → {_esc(_schema_type_names(item.get("range_types") or item.get("target_types")))}</dd></div>
          <div><dt>方向说明</dt><dd>{_esc(_schema_ui_zh(item.get("direction_description_zh") or item.get("direction_description"), "未设置"))}</dd></div>
          <div><dt>结构属性</dt><dd>{'对称关系' if bool(item.get("symmetric")) else '有向关系'}</dd></div></dl></article>"""
        for item in relations
    ) or '<div class="empty-state compact-empty">当前版本未定义关系类别。</div>'
    return f"""<div class="workspace-section knowledge-structure">
      <div class="section-heading"><div>
        <div class="knowledge-structure-title"><h2>知识结构定义</h2>
          <span class="schema-version-badge">当前版本 v{_esc(schema.get("version") or "")}</span></div>
        <p>定义项目允许识别的实体类别、关系语义、端点类型与方向；构建时据此约束抽取和验证。</p></div>
        {reassess_button}</div>
      <section class="definition-group"><div class="definition-group-heading"><h3>实体类别</h3><span>{len(entities)} 类</span></div>
        <div class="definition-grid entity-definition-grid">{entity_cards}</div></section>
      <section class="definition-group"><div class="definition-group-heading"><h3>关系类别</h3><span>{len(relations)} 类</span></div>
        <div class="definition-grid relation-definition-grid">{relation_cards}</div></section>
    </div>"""


def _knowledge_forge_process_artifacts(documents: list[dict]) -> str:
    rows: list[str] = []
    for item in documents:
        document_id = int(item["id"])
        quality = item.get("parse_quality") if isinstance(item.get("parse_quality"), dict) else {}
        block_count = int(quality.get("block_count") or 0)
        chunk_count = int(item.get("prepared_chunk_count") or 0)
        parser = str(item.get("effective_parser") or "等待预处理")
        chunk_strategy = str(item.get("effective_chunk_strategy") or "等待预处理")
        rows.append(
            f"""<tr><td><strong>{_esc(item['original_filename'])}</strong>
              <small>{_esc(parser)} · {_esc(chunk_strategy)}</small></td>
            <td><b class="artifact-count">{block_count}</b><small>结构块</small></td>
            <td><b class="artifact-count">{chunk_count}</b><small>知识块</small></td>
            <td class="artifact-actions">
              <button class="button small secondary" type="button" data-view-structure="{document_id}"
                {"disabled" if not item.get("structure_ready_at") else ""}>查看结构块列表</button>
              <button class="button small secondary" type="button" data-view-chunks="{document_id}"
                {"disabled" if not item.get("chunks_ready_at") else ""}>查看知识块列表</button>
            </td></tr>"""
        )
    body = "".join(rows) or '<tr><td colspan="4" class="empty">上传并预处理文档后，这里会集中展示结构块和知识块。</td></tr>'
    return f"""<div class="workspace-section process-artifacts">
      <div class="section-heading"><div><h2>文档处理产物</h2>
        <p>集中查看每份文档的结构块与知识块，无需返回文档管理页面。</p></div></div>
      <div class="table-wrap"><table class="artifact-table"><thead><tr><th>文档与处理方式</th>
        <th>结构解析</th><th>知识分块</th><th>查看列表</th></tr></thead><tbody>{body}</tbody></table></div>
    </div>"""


def _graphrag_native_process_artifacts(documents: list[dict]) -> str:
    """Render prepared Native inputs beside the preparation controls."""

    rows: list[str] = []
    for item in documents:
        document_id = int(item["id"])
        quality = item.get("parse_quality") if isinstance(item.get("parse_quality"), dict) else {}
        block_count = int(quality.get("block_count") or 0)
        parser = str(item.get("effective_parser") or "等待准备")
        ready = bool(item.get("structure_ready_at"))
        rows.append(
            f"""<tr><td><strong>{_esc(item['original_filename'])}</strong>
              <small>{_esc(parser)}</small></td>
            <td><b class="artifact-count">{block_count}</b><small>源结构块</small></td>
            <td><span class="status {'status-indexed' if ready else ''}">{'已冻结' if ready else '待准备'}</span></td>
            <td class="artifact-actions">
              <button class="button small secondary" type="button" data-view-structure="{document_id}"
                {"disabled" if not ready else ""}>查看源结构</button>
              <button class="button small secondary" type="button" data-view-native-documents="{document_id}"
                {"disabled" if not ready else ""}>查看 Native Documents</button>
            </td></tr>"""
        )
    body = "".join(rows) or '<tr><td colspan="4" class="empty">上传并准备源文档后，这里会展示 GraphRAG Native 的冻结输入。</td></tr>'
    return f"""<div class="workspace-section process-artifacts native-process-artifacts">
      <div class="section-heading"><div><h2>源文档准备产物</h2>
        <p>集中查看源结构与冻结的 Native Documents；TextUnits 和图谱产物在原生图谱页查看。</p></div></div>
      <div class="table-wrap"><table class="artifact-table"><thead><tr><th>文档与解析方式</th>
        <th>结构解析</th><th>准备状态</th><th>查看过程文件</th></tr></thead><tbody>{body}</tbody></table></div>
    </div>"""


def _documents_ready_for_knowledge_build(documents: list[dict]) -> bool:
    return bool(documents) and all(
        bool(item.get("active_revision_id")) and bool(item.get("structure_ready_at"))
        and str(item.get("status") or "") == "indexed"
        for item in documents
    )


def _construction_controls(
    *,
    project_id: int,
    strategy_key: str,
    strategy_select: str = "",
    latest_job_id: str,
    latest_batch_id: str,
    active_job_id: str,
    build_ready: bool = True,
    scope: str = "all",
    after_build: str = "",
) -> str:
    ui = _strategy_ui(strategy_key)
    if scope not in {"all", "preparation", "graph"}:
        raise ValueError(f"unknown construction control scope: {scope}")
    disabled = " disabled" if active_job_id else ""
    build_disabled = " disabled" if active_job_id or not build_ready else ""
    build_title = str(ui["build_ready_title"] if build_ready else ui["build_not_ready_title"])
    preparation_controls = ""
    if scope in {"all", "preparation"}:
        preparation_controls = f"""
      <button class="button primary" id="prepare-project-documents" data-prepare-project="{_esc(project_id)}" type="button"
        title="{_esc(ui['preparation_title'])}"{disabled}>{_esc(ui['preparation_label'])}</button>
      <button class="button secondary" id="view-preparation-log" data-view-job-log data-job="{_esc(latest_job_id)}"
        data-batch="{_esc(latest_batch_id)}"{' disabled' if not latest_job_id else ''}>{_esc(ui['preparation_log_label'])}</button>"""
    graph_controls = ""
    if scope in {"all", "graph"}:
        graph_controls = f"""
      <button class="button primary" id="build-project-knowledge" data-build-project="{_esc(project_id)}" type="button"
        title="{_esc(build_title)}"{build_disabled}>{_esc(ui['build_label'])}</button>
      {after_build}
      <button class="button secondary" id="view-ingestion-log" data-view-job-log data-job="{_esc(latest_job_id)}"
        data-batch="{_esc(latest_batch_id)}"{' disabled' if not latest_job_id else ''}>{_esc(ui['job_log_label'])}</button>"""
    return f"""<div class="construction-controls">
      {preparation_controls}
      {graph_controls}
    </div>"""


def _workspace_path(project_id: int, strategy_key: str) -> str:
    strategy_key = normalize_strategy(strategy_key)
    slug = (
        "graphrag-native"
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY
        else "knowledge-forge"
    )
    return f"/projects/{int(project_id)}/{slug}"


def _project_binding(project_id: int) -> dict[str, Any] | None:
    getter = getattr(SERVICE.db, "get_project_construction_strategy_selection", None)
    if getter is None:
        return None
    return getter(int(project_id))


def render_home_page(message: tuple[str, str] | None = None) -> str:
    """Common project creation/selection page; workspaces are strategy-owned."""

    projects, models = SERVICE.db.list_projects(), SERVICE.db.list_models()
    cards: list[str] = []
    for project in projects:
        project_id = int(project["id"])
        selection = _project_binding(project_id) or {}
        status = str(selection.get("association_status") or "unassigned")
        strategy_key = str(selection.get("selected_strategy") or STABLE_STRATEGY)
        if status == "bound" and strategy_key in {
            ENHANCED_SHADOW_STRATEGY, GRAPHRAG_NATIVE_STRATEGY,
        }:
            strategy = _strategy_ui(strategy_key)
            action = (
                f'<a class="button primary" href="{_workspace_path(project_id, strategy_key)}">'
                f'进入 {_esc(strategy["workspace_title"])}</a>'
            )
            badge = _esc(
                next(
                    item.label
                    for item in strategy_options(include_hidden=False)
                    if item.key == strategy_key
                )
            )
        else:
            generation = int(selection.get("generation") or 0)
            choices = "".join(
                f'<option value="{_esc(item.key)}">{_esc(item.label)}</option>'
                for item in strategy_options(include_hidden=False)
            )
            action = f"""<form method="post" action="/projects/{project_id}/strategy" class="bind-form">
              <input type="hidden" name="expected_generation" value="{generation}">
              <select name="strategy_key" aria-label="选择 {_esc(project['name'])} 的知识图谱策略" required>{choices}</select>
              <button class="button primary" type="submit">确认并进入</button>
            </form>"""
            badge = "待确认策略"
        cards.append(f"""<article class="project-card"><div><h3>{_esc(project['name'])}</h3>
          <p>{_esc(project.get('description') or '暂无项目说明')}</p>
          <span class="strategy-badge">{badge}</span></div>{action}</article>""")
    choices = "".join(
        f'<option value="{_esc(item.key)}">{_esc(item.label)}</option>'
        for item in strategy_options(include_hidden=False)
    )
    notice = _notice(*message) if message else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>Knowledge Forge 项目</title><style>{CSS}</style></head><body>
      <main class="shell home-shell"><header class="home-header"><div><h1>Knowledge Forge</h1>
      <p>通用知识库与知识图谱构建平台</p></div>
      <button class="button primary" type="button" id="open-project-list">开始工作</button></header>
      {notice}
      <section class="panel home-hero" aria-labelledby="home-introduction-title">
        <div class="hero-layout"><div class="hero-copy">
          <span class="hero-kicker">从原始文档到可解释知识</span>
          <h2 id="home-introduction-title">让分散的资料成为可检索、可连接、可追溯的知识资产</h2>
          <p>Knowledge Forge 面向需要长期积累和使用文档知识的团队。你可以导入 PDF、Word、表格、Markdown
          和文本资料，由系统完成结构解析、语义组织、实体关系构建与知识查询。</p>
          <p>平台提供两条彼此独立的构建路线：一条强调精确语义、实体归一和证据核验；另一条采用
          GraphRAG 原生工作流，强调主题发现、关系网络与社区摘要。每个项目固定选择一种路线，
          让数据边界和运行结果始终清晰。</p>
        </div>
        <div class="hero-points" aria-label="平台能力">
          <div><b>多格式解析</b><span>保留章节、表格、页码和来源结构</span></div>
          <div><b>知识可追溯</b><span>实体、关系和回答可回看原文依据</span></div>
          <div><b>双构建引擎</b><span>按业务目标选择精确治理或开放探索</span></div>
          <div><b>本地化运行</b><span>使用本机 PostgreSQL、Ollama 与图数据</span></div>
        </div></div>
        <ol class="home-flow" aria-label="知识构建流程">
          <li><b>1</b><span><strong>导入资料</strong><small>汇集项目文档</small></span></li>
          <li><b>2</b><span><strong>解析结构</strong><small>理解章节与内容边界</small></span></li>
          <li><b>3</b><span><strong>构建知识</strong><small>形成实体、关系与主题网络</small></span></li>
          <li><b>4</b><span><strong>检索探索</strong><small>查询、审计并持续迭代</small></span></li>
        </ol>
      </section>
      <section class="panel strategy-overview" aria-labelledby="strategy-overview-title">
        <div class="section-heading strategy-heading"><div><span class="section-kicker">创建项目前先做选择</span>
          <h2 id="strategy-overview-title">两种知识构建方式，分别适合什么场景？</h2>
          <p>它们共享同一个项目入口，但准备、构建、存储和查询链路互不混用。下面的差异会直接影响最终图谱的形态。</p>
        </div></div>
        <div class="strategy-intro-grid">
          <article class="strategy-explainer strategy-kf">
            <div class="strategy-card-head"><span class="strategy-mark">KF</span><div>
              <span class="strategy-tag">精确治理型</span><h3>Knowledge Forge</h3>
            </div></div>
            <p class="strategy-lead">围绕项目 Schema 构建受控实体和关系，强调“这是谁、是什么关系、方向是否正确、证据在哪里”。</p>
            <div class="strategy-path">结构块 → Chunk 与向量 → Schema → 实体归一 → 关系核验 → 隔离图谱</div>
            <dl class="strategy-details">
              <div><dt>得到什么</dt><dd>规范实体、明确谓词与方向、关系事实、精确证据跨度，以及可审计的处理记录。</dd></div>
              <div><dt>主要特点</dt><dd>实体别名归一，关系经过分类和独立核验；可用项目 Schema 约束类型、方向和发布方式。</dd></div>
              <div><dt>查询与存储</dt><dd>PostgreSQL、pgvector 与独立知识图谱协同，适合语义检索和实体关系探索。</dd></div>
            </dl>
            <div class="strategy-fit"><b>更适合</b><ul>
              <li>业务术语和关系含义需要长期稳定</li>
              <li>在意实体去重、关系方向和原文证据</li>
              <li>要建设可治理、可复核的领域知识图谱</li>
            </ul></div>
            <p class="strategy-tradeoff"><b>需要知道：</b>质量门禁和模型步骤更多，构建通常更慢；严格门禁可能舍弃证据不足的候选关系。</p>
          </article>
          <article class="strategy-explainer strategy-native">
            <div class="strategy-card-head"><span class="strategy-mark">GR</span><div>
              <span class="strategy-tag">开放探索型</span><h3>GraphRAG Native</h3>
            </div></div>
            <p class="strategy-lead">采用 GraphRAG 3.1.1 原生工作流，从 TextUnits 生成语义实体、描述型关系、社区和社区报告。</p>
            <div class="strategy-path">Native Documents → TextUnits → 原生实体关系 → 社区 → 报告与 LanceDB</div>
            <dl class="strategy-details">
              <div><dt>得到什么</dt><dd>原生语义网络、关系描述、Leiden 社区、社区摘要、GraphML 和可独立审计的 bundle。</dd></div>
              <div><dt>主要特点</dt><dd>更关注文档中“哪些概念彼此相关、形成哪些主题群”，支持跨文档的整体结构发现。</dd></div>
              <div><dt>查询与存储</dt><dd>使用 GraphRAG 的 Local、Global、DRIFT、Basic 查询及 LanceDB，不依赖 pgvector 或 HugeGraph。</dd></div>
            </dl>
            <div class="strategy-fit"><b>更适合</b><ul>
              <li>希望快速探索一批资料的主题与联系</li>
              <li>重视社区发现、全局摘要和开放式问答</li>
              <li>希望直接使用标准 GraphRAG 原生产物</li>
            </ul></div>
            <p class="strategy-tradeoff"><b>需要知道：</b>关系主要是自然语言语义描述，不受项目谓词 Schema 严格约束；相近实体也可能保留为不同节点。</p>
          </article>
        </div>
        <div class="choice-guide"><b>快速判断</b>
          <span>需要精确实体、受控关系和审计证据，选择 <strong>Knowledge Forge</strong>。</span>
          <span>需要快速理解主题网络、社区和全局脉络，选择 <strong>GraphRAG Native</strong>。</span>
          <em>项目创建后不能切换构建方式；如果两种结果都需要，请分别创建两个项目。</em>
        </div>
      </section>
      <dialog id="project-list-dialog" class="project-list-dialog" aria-labelledby="project-list-title">
      <div class="dialog-heading project-list-heading"><div><span class="section-kicker">开始工作</span><h2 id="project-list-title">项目</h2>
      <p>选择已有项目继续工作，或创建一个绑定知识构建方式的新项目。</p></div>
      <div class="dialog-title-actions"><button class="button primary" type="button" id="open-create-project">新建项目</button>
      <button class="icon-button" type="button" data-close-project-list aria-label="关闭项目列表">×</button></div></div>
      <div class="project-grid">{''.join(cards) or '<p class="empty">尚无项目，请先创建。</p>'}</div></dialog>
      <dialog id="create-project-dialog"><div class="dialog-heading"><div><h2>新建项目</h2>
      <p>请选择最符合当前目标的知识构建方式；项目创建后不可切换。</p></div>
      <button class="icon-button" type="button" data-close-create-project>×</button></div>
      <form action="/projects" method="post" class="form-grid">
        <label>项目名称<input name="name" required maxlength="100"></label>
        <label>知识构建方式<select id="create-strategy-key" name="strategy_key" required>{choices}</select></label>
        <p class="wide strategy-choice-help" id="strategy-choice-help"></p>
        <label>LLM<select name="llm_model">{_model_options(models,'llm')}</select></label>
        <label>Embedding<select name="embedding_model">{_model_options(models,'embedding')}</select></label>
        <label class="wide">项目说明<textarea name="description" rows="2"></textarea></label>
        <div class="form-actions dialog-actions">
          <button class="button secondary" type="button" data-close-create-project>取消</button>
          <button class="button primary" type="submit">创建项目</button></div></form></dialog>
      </main><script>
      const projectDialog=document.querySelector('#project-list-dialog');
      const dialog=document.querySelector('#create-project-dialog');
      const strategySelect=document.querySelector('#create-strategy-key');
      const strategyHelp=document.querySelector('#strategy-choice-help');
      const strategyDescriptions={{
        enhanced_shadow:'适合需要实体归一、受控关系、方向核验和原文证据的领域知识图谱。',
        graphrag_native:'适合快速发现主题网络、社区结构，并使用 GraphRAG 原生查询和产物。'
      }};
      function updateStrategyHelp(){{
        if(strategyHelp)strategyHelp.textContent=strategyDescriptions[strategySelect?.value]||'';
      }}
      document.querySelector('#open-project-list')?.addEventListener('click',()=>projectDialog.showModal());
      document.querySelectorAll('[data-close-project-list]').forEach(
        button=>button.addEventListener('click',()=>projectDialog.close()));
      projectDialog?.addEventListener('click',event=>{{
        if(event.target===projectDialog)projectDialog.close();
      }});
      document.querySelector('#open-create-project')?.addEventListener('click',()=>{{
        projectDialog.close();dialog.showModal();
      }});
      strategySelect?.addEventListener('change',updateStrategyHelp);updateStrategyHelp();
      document.querySelectorAll('[data-close-create-project]').forEach(
        button=>button.addEventListener('click',()=>dialog.close()));
      </script></body></html>"""


def _embedded_graph_browser(
    strategy_key: str,
    *,
    ready: bool,
    active: bool,
) -> str:
    native = strategy_key == GRAPHRAG_NATIVE_STRATEGY
    title = "GraphRAG Native 原生语义图" if native else "Knowledge Forge 知识图谱"
    description = (
        "浏览原生实体、关系、社区与 TextUnit 证据；图谱来自当前 fresh native bundle。"
        if native else
        "浏览规范实体、已验证关系、社区及原文证据；图谱直接读取当前已发布快照，不依赖 Hubble。"
    )
    group_label = "全部社区"
    if ready:
        initial = (
            '<div class="embedded-graph-loading" data-graph-loading>'
            '<span class="embedded-graph-spinner" aria-hidden="true"></span>'
            '<div><b>正在加载图谱</b><p>正在读取实体、关系和分组信息...</p></div></div>'
        )
    elif active:
        initial = (
            '<div class="embedded-graph-loading">'
            '<span class="embedded-graph-spinner" aria-hidden="true"></span>'
            '<div><b>图谱正在构建</b><p>构建完成后将在这里自动加载，无需打开其他页面。</p></div></div>'
        )
    else:
        initial = (
            '<div class="empty-state embedded-graph-empty">'
            '完成一次图谱构建后，实体、关系、分组和证据会自动显示在这里。</div>'
        )
    viewer = "" if ready else " hidden"
    return f"""<section id="embedded-graph-browser" class="workspace-section embedded-graph-browser"
      data-strategy="{_esc(strategy_key)}" data-ready="{str(ready).lower()}" data-active="{str(active).lower()}">
      <div class="section-heading"><div><h2>{_esc(title)}</h2><p>{_esc(description)}</p></div>
        <div class="embedded-graph-tabs" role="tablist" aria-label="图谱浏览方式">
          <button id="embedded-graph-network-tab" class="embedded-graph-tab is-active"
            type="button" role="tab" aria-selected="true" aria-controls="embedded-graph-network-panel"
            data-graph-tab="network">图谱连接</button>
          <button id="embedded-graph-audit-tab" class="embedded-graph-tab"
            type="button" role="tab" aria-selected="false" aria-controls="embedded-graph-audit-panel"
            tabindex="-1" data-graph-tab="audit">关系审计表</button>
        </div></div>
      <div data-graph-status>{initial}</div>
      <div class="embedded-graph-viewer"{viewer}>
        <div class="embedded-graph-controls">
          <select data-graph-group-filter aria-label="图谱分组"><option value="">{_esc(group_label)}</option></select>
          <input data-graph-filter placeholder="筛选实体名称、类别、说明或别名" aria-label="图谱筛选条件">
          <button class="button primary" type="button" data-apply-graph-filter>筛选</button>
          <label><input data-toggle-graph-isolates type="checkbox" checked> 显示孤立实体</label>
          <div class="embedded-graph-zoom" role="group" aria-label="图谱缩放">
            <button class="button secondary" type="button" data-graph-zoom-out aria-label="缩小图谱">−</button>
            <span data-graph-zoom-value aria-live="polite">100%</span>
            <button class="button secondary" type="button" data-graph-zoom-in aria-label="放大图谱">＋</button>
            <button class="button secondary" type="button" data-graph-fit>适应画布</button>
          </div>
          <button class="button secondary" type="button" data-reset-graph>重置视图</button>
        </div>
        <div id="embedded-graph-network-panel" role="tabpanel"
          aria-labelledby="embedded-graph-network-tab" data-graph-panel="network">
          <div class="embedded-graph-layout">
            <div class="embedded-graph-canvas">
              <svg data-graph-network viewBox="0 0 1200 720" role="img">
                <defs><marker id="embedded-graph-arrow" markerWidth="7" markerHeight="7"
                  refX="22" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z"></path></marker></defs>
                <g data-graph-group-labels></g><g data-graph-edges></g><g data-graph-nodes></g>
              </svg>
              <p class="embedded-graph-count" data-graph-count></p>
            </div>
            <aside class="embedded-graph-sidebar">
              <div data-graph-details><h3>浏览说明</h3><p>点击实体或关系查看详情；点击图中空白处恢复完整视图。</p></div>
              <div data-graph-community-status class="embedded-graph-community-status"></div>
              <div data-graph-groups class="embedded-graph-groups"></div>
            </aside>
          </div>
        </div>
        <div id="embedded-graph-audit-panel" role="tabpanel"
          aria-labelledby="embedded-graph-audit-tab" data-graph-panel="audit" hidden>
          <p class="muted" data-graph-audit-summary></p>
          <div class="table-wrap embedded-graph-audit"><table><thead><tr>
            <th>源实体</th><th>关系</th><th>目标实体</th><th>强度</th><th>证据</th>
          </tr></thead><tbody data-graph-audit-rows></tbody></table></div>
        </div>
      </div>
    </section>"""


def render_page(
    project_id: int | None,
    strategy_key: str = ENHANCED_SHADOW_STRATEGY,
    message: tuple[str, str] | None = None,
) -> str:
    try:
        strategy_key = normalize_strategy(strategy_key)
    except ValueError:
        strategy_key = ENHANCED_SHADOW_STRATEGY
    visible_strategy_keys = {
        item.key for item in strategy_options(include_hidden=False)
    }
    if strategy_key not in visible_strategy_keys:
        strategy_key = ENHANCED_SHADOW_STRATEGY
    ui = _strategy_ui(strategy_key)
    projects, models = SERVICE.db.list_projects(), SERVICE.db.list_models()
    current = next((item for item in projects if item["id"] == project_id), None)
    if current:
        binding = _project_binding(int(current["id"]))
        if binding is not None:
            if str(binding.get("association_status") or "") != "bound":
                raise ValueError("项目尚未绑定知识图谱策略")
            if str(binding.get("selected_strategy") or "") != strategy_key:
                raise ValueError("工作区与项目绑定的知识图谱策略不一致")
        if strategy_key == ENHANCED_SHADOW_STRATEGY:
            SERVICE.db.ensure_project_strategy_graphs(int(current["id"]))
    native_status = None
    native_run = None
    if current and strategy_key == GRAPHRAG_NATIVE_STRATEGY:
        try:
            native_status = SERVICE.graphrag_native_status(int(current["id"]))
        except Exception as exc:
            native_status = {"fresh": False, "unavailable_reason": str(exc)}
        native_run = native_status.get("run") if isinstance(native_status, dict) else None
        graph_state = {
            "status": "ready" if native_run and native_status.get("fresh") else "stale" if native_run else "empty",
            "generation": int((native_run or {}).get("generation") or 0),
            "ready_generation": int((native_run or {}).get("generation") or 0),
            "native_run": native_run,
        }
    else:
        graph_state = SERVICE.db.get_project_strategy_graph(int(current["id"]), strategy_key) if current else None
    documents = SERVICE.db.list_documents(int(current["id"])) if current and strategy_key == GRAPHRAG_NATIVE_STRATEGY else (SERVICE.db.list_documents(int(current["id"]), strategy_key) if current else [])
    statistics = _strategy_statistics(int(current["id"]), strategy_key, graph_state) if current else {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}
    schema = SERVICE.db.get_active_schema(int(current["id"])) if current else None
    def newest(*jobs):
        values = [item for item in jobs if item]
        return max(
            values,
            key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""),
            default=None,
        )

    native_latest_job = (
        SERVICE.latest_job(int(current["id"]), "graphrag_native_build", strategy_key)
        if current and strategy_key == GRAPHRAG_NATIVE_STRATEGY else None
    )
    latest_preparation_job = (
        SERVICE.latest_job(int(current["id"]), "content_preparation", strategy_key)
        if current else None
    )
    latest_build_job = (
        native_latest_job
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY
        else (SERVICE.latest_job(int(current["id"]), "ingestion", strategy_key) if current else None)
    )
    strategy_active_job = newest(
        SERVICE.latest_active_job(int(current["id"]), "graphrag_native_build", strategy_key) if current and strategy_key == GRAPHRAG_NATIVE_STRATEGY else (SERVICE.latest_active_job(int(current["id"]), "ingestion", strategy_key) if current else None),
        SERVICE.latest_active_job(int(current["id"]), "content_preparation", strategy_key) if current else None,
    )
    # The backend serializes Schema reassessment, graph publication and
    # document cleanup against every other project mutation.  Use the same
    # project-wide view here so a reload cannot re-enable controls while one
    # of those jobs is still active.
    active_job = newest(
        strategy_active_job,
        SERVICE.latest_active_job(int(current["id"])) if current else None,
    )
    preparation_job_id = str(latest_preparation_job["id"]) if latest_preparation_job else ""
    preparation_batch_id = str((latest_preparation_job.get("metadata") or {}).get("batch_id") or "") if latest_preparation_job else ""
    build_job_id = str(latest_build_job["id"]) if latest_build_job else ""
    build_batch_id = str((latest_build_job.get("metadata") or {}).get("batch_id") or "") if latest_build_job else ""
    active_job_id = str(active_job["id"]) if active_job else ""
    active_job_kind = str(active_job["kind"]) if active_job_id else ""
    active_batch_id = str((active_job.get("metadata") or {}).get("batch_id") or "") if active_job_kind in {"ingestion", "content_preparation"} else ""
    schema_reassessment_available = bool(current) and _schema_reassessment_available(
        strategy_key, active_job_id,
    )
    processing_options = (
        f"""<div class="workspace-section settings-processing">
          <div class="section-heading"><div><h2>文档解析与分块</h2>
            <p>这些选项应用于下一次上传；文件仍在“文档管理”中选择和上传。</p></div></div>
          <div class="processing-options settings-processing-options">
            <div id="processing-selects" class="form-grid compact">
              <label>解析器<select name="requested_parser" form="document-upload-form">{_named_options(PARSER_OPTIONS,'auto')}</select></label>
              <label>分块策略<select name="requested_strategy" form="document-upload-form">{_named_options(CHUNK_STRATEGY_OPTIONS,'auto')}</select></label>
            </div>
            <p id="processing-hint" class="muted">选择文件后会检查解析器和分块策略的兼容性。</p>
          </div>
        </div>"""
        if strategy_key == ENHANCED_SHADOW_STRATEGY
        else f"""<div class="workspace-section settings-processing">
          <div class="section-heading"><div><h2>源文档解析</h2>
            <p>解析器应用于下一次上传；TextUnit 参数由 GraphRAG Native 工作流固定管理。</p></div></div>
          <div class="processing-options settings-processing-options">
            <div id="processing-selects" class="form-grid compact">
              <label>解析器<select name="requested_parser" form="document-upload-form">{_named_options(PARSER_OPTIONS,'auto')}</select></label>
              <input type="hidden" name="requested_strategy" value="auto" form="document-upload-form">
              <div class="native-config-note"><b>TextUnit 设置</b><span>GraphRAG 3.1.1 在原生构建内使用固定、可审计的 900 tokens / 100 overlap / o200k_base。</span></div>
            </div>
            <p id="processing-hint" class="muted">这里只生成源结构；不会生成 Knowledge Forge Chunk、pgvector 或知识结构定义。</p>
          </div>
        </div>"""
    )
    rows = _document_rows(documents, strategy_key)
    knowledge_structure_section = (
        _knowledge_structure_details(
            schema,
            project_id=int(current["id"]) if current else 0,
            reassessment_available=schema_reassessment_available,
        ) if strategy_key == ENHANCED_SHADOW_STRATEGY else ""
    )
    process_artifacts_section = (
        _knowledge_forge_process_artifacts(documents)
        if strategy_key == ENHANCED_SHADOW_STRATEGY
        else _graphrag_native_process_artifacts(documents)
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY else ""
    )
    current_block = ""
    graph_query_actions = ""
    embedded_graph_section = ""
    settings_section = ""
    if current:
        graph_status = str((graph_state or {}).get("status") or "empty")
        graph_data_ready = (
            (
                is_current_graph(current)
                if strategy_key == STABLE_STRATEGY
                else graph_status == "ready" and int((graph_state or {}).get("generation") or 0)
                == int((graph_state or {}).get("ready_generation") or 0)
            )
        )
        graph_enabled = graph_data_ready and not active_job_id
        graph_retry = "" if strategy_key != STABLE_STRATEGY or graph_enabled or active_job_id else (
            f'<button class="button primary" type="button" data-rebuild-graph="{_esc(current["id"])}">重建知识图谱</button>'
        )
        native_artifact_control = (
            f"""<button class="button secondary" type="button" id="view-native-artifacts"
              {'disabled' if not native_run or not (native_status or {}).get('fresh') else ''}>查看原生产物</button>"""
            if strategy_key == GRAPHRAG_NATIVE_STRATEGY else ""
        )
        preparation_controls = _construction_controls(
            project_id=int(current["id"]),
            strategy_key=strategy_key,
            latest_job_id=preparation_job_id,
            latest_batch_id=preparation_batch_id,
            active_job_id=active_job_id,
            build_ready=_documents_ready_for_knowledge_build(documents),
            scope="preparation",
        )
        build_controls = _construction_controls(
            project_id=int(current["id"]),
            strategy_key=strategy_key,
            latest_job_id=build_job_id,
            latest_batch_id=build_batch_id,
            active_job_id=active_job_id,
            build_ready=_documents_ready_for_knowledge_build(documents),
            scope="graph",
            after_build=native_artifact_control,
        )
        strategy_metadata = ""
        stat_labels = ui["stat_labels"]
        assert isinstance(stat_labels, dict)
        updating_statistics = _updating_statistic_keys(
            strategy_key, active_job_kind,
        )
        current_block = f"""<div class="preparation-heading-actions">
          {preparation_controls}
        </div>"""
        graph_query_actions = f"""<div class="workspace-section graph-query-actions">
          <div class="section-heading"><div><h2>{_esc(ui['workspace_title'])}</h2>
            <p>{_esc(ui['workspace_description'])}</p></div></div>
          <div class="stats"><div><b data-stat="documents">{statistics['documents']}</b><span>{_esc(stat_labels['documents'])}</span></div>
            <div><b data-stat="chunks"{' class="is-updating"' if 'chunks' in updating_statistics else ''}>{statistics['chunks']}</b><span>{_esc(stat_labels['chunks'])}</span></div>
            <div><b data-stat="entities"{' class="is-updating"' if 'entities' in updating_statistics else ''}>{statistics['entities']}</b><span>{_esc(stat_labels['entities'])}</span></div>
            <div><b data-stat="relations"{' class="is-updating"' if 'relations' in updating_statistics else ''}>{statistics['relations']}</b><span>{_esc(stat_labels['relations'])}</span></div>
          </div>
          <div class="stats-actions">{build_controls}{strategy_metadata}
            <div class="graph-actions">{graph_retry}</div>
          </div>
        </div>"""
        if strategy_key in {ENHANCED_SHADOW_STRATEGY, GRAPHRAG_NATIVE_STRATEGY}:
            embedded_graph_section = _embedded_graph_browser(
                strategy_key, ready=graph_data_ready, active=bool(active_job_id),
            )
        chunk_setting = (
            f"""<label>单切片 Token 上限
            <input name="chunk_max_tokens" type="number" min="64" step="1" required value="{int(current.get('chunk_max_tokens') or 512)}">
            <small id="chunk-token-limit-hint" class="field-help">正在读取 LLM 与 Embedding 的共同上限...</small></label>"""
            if strategy_key == ENHANCED_SHADOW_STRATEGY
            else f"""<input name="chunk_max_tokens" type="hidden" value="{int(current.get('chunk_max_tokens') or 512)}">
            <div class="native-config-note"><b>GraphRAG Native TextUnit</b>
            <span>由 GraphRAG 3.1.1 在运行内生成：900 tokens，100 overlap，o200k_base；不读取 Knowledge Forge 单切片设置。</span></div>"""
        )
        settings_section = f"""
        <div class="workspace-settings"><div class="section-heading"><div><h2>项目设置</h2>
          <p>设置当前项目使用的模型与上下文预算；不会修改 Ollama 全局上下文。</p></div>
          <button id="sync-models" class="icon-button refresh-button" type="button"
            title="从 Ollama 同步可用模型" aria-label="从 Ollama 同步可用模型">↻</button></div>
        <p id="model-sync-status" class="model-sync-status" role="status" hidden></p>
        <form id="project-settings-form" method="post" action="/projects/{current['id']}/models" class="form-grid compact">
          <label>LLM<select name="llm_model">{_model_options(models,'llm',current.get('llm_model') or '')}</select></label>
          <label>Embedding<select name="embedding_model">{_model_options(models,'embedding',current.get('embedding_model') or '')}</select></label>
          <label>项目上下文 Token 上限
            <input name="model_max_tokens" type="number" min="1024" step="1" required value="{int(current.get('model_max_tokens') or 32768)}">
            <small id="model-token-limit-hint" class="field-help">正在读取当前 LLM 的上下文上限...</small></label>
          {chunk_setting}
          <div class="form-actions"><button class="button primary" type="submit">保存项目设置</button></div></form>
        {processing_options}</div>"""
    job_hidden = "" if active_job_id else " hidden"
    notice = _notice(*message) if message else ""
    task_box = f"""<div id="task-box" class="task-box"{job_hidden} data-job="{_esc(active_job_id)}"
      data-batch="{_esc(active_batch_id)}" data-kind="{_esc(active_job_kind)}">
      <div class="task-head"><b id="task-title">{_esc(ui['task_title'])}</b><span><span id="task-status">等待</span>
        <span id="task-duration" class="task-duration">用时 00:00</span>
        <button id="active-job-log" class="button small task-log-button" type="button"
          data-view-job-log data-job="{_esc(active_job_id)}" data-batch="{_esc(active_batch_id)}">查看详细日志</button></span></div>
      <div id="task-steps" class="task-steps">{_task_steps_html(strategy_key)}</div>
    </div>"""
    preparation_task_box = (
        task_box if active_job_kind in {"", "content_preparation"} else ""
    )
    graph_task_box = (
        task_box if active_job_kind not in {"", "content_preparation"} else ""
    )
    preparation_task_slot = f'<div data-task-slot="preparation">{preparation_task_box}</div>'
    graph_task_slot = f'<div data-task-slot="graph">{graph_task_box}</div>'
    native_runtime_section = (
        _native_runtime_details(native_latest_job)
        if current and strategy_key == GRAPHRAG_NATIVE_STRATEGY else ""
    )
    retrieval_section = graph_query_actions + graph_task_slot + native_runtime_section + """<div class="workspace-section" {visible}><div class="section-heading"><div><h2>联合检索</h2><p>融合当前文档的向量召回，以及当前图谱版本的规范实体和已验证关系；不会读取旧基线图谱。</p></div></div>
      <form id="search-form" class="search-form"><input name="query" placeholder="输入关键词、实体名称或检索描述" required><button class="button primary">检索</button></form><div id="search-results"></div>
      </div>{graph}""".format(visible="" if current else "hidden", graph=embedded_graph_section)
    if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
        retrieval_section = graph_query_actions + graph_task_slot + native_runtime_section + """<div class="workspace-section" {visible}><div class="section-heading"><div><h2>GraphRAG 知识问答</h2><p>基于当前有效的 GraphRAG 知识包生成回答；不会调用其他知识库策略或旧检索接口。</p></div></div>
          <form id="native-search-form" class="search-form native-query-form"><select name="mode" aria-label="问答方式" aria-describedby="native-query-mode-help"><option value="local">详细问答（Local）</option><option value="global">全局总结（Global）</option><option value="drift">主题深入分析（DRIFT）</option><option value="basic">原文问答（Basic）</option></select><input name="question" placeholder="输入需要基于当前知识库回答的问题" aria-label="知识库问题" required><button class="button primary">生成回答</button></form>
          <p id="native-query-mode-help" class="native-query-mode-help" data-native-query-mode-help>详细问答会综合相关实体、关系、社区报告和原文片段，适合询问具体对象及其联系。</p><div id="native-search-results"></div>
          </div>{graph}""".format(visible="" if current else "hidden", graph=embedded_graph_section)
    strategy_label = next(
        item.label for item in strategy_options(include_hidden=False)
        if item.key == strategy_key
    )
    query_navigation_label = str(ui["query_navigation_label"])
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{_esc(current['name'] if current else 'Knowledge Forge')} · Knowledge Forge</title><style>{CSS}</style></head>
    <body><main class="shell workspace-shell">
    <header class="workspace-brand-header"><a class="brand-link" href="/" aria-label="返回项目首页"><h1>Knowledge Forge</h1>
      <p>通用知识库与知识图谱构建平台</p></a></header>{notice}
    <div class="workspace-layout">
      <aside class="workspace-sidebar" aria-label="项目工作区导航">
        <div class="workspace-project-summary"><div class="workspace-project-title">
          <h2>{_esc(current['name'] if current else '项目不存在')}</h2>
          <span class="workspace-strategy-badge">{_esc(strategy_label)}</span>
        </div></div>
        <nav class="workspace-nav" aria-label="项目功能">
          <button class="workspace-nav-item" type="button" data-workspace-target="documents"
            aria-controls="workspace-documents" aria-selected="true"><span>01</span><b>文档管理</b><small>上传与文件列表</small></button>
          <button class="workspace-nav-item" type="button" data-workspace-target="build"
            aria-controls="workspace-build" aria-selected="false"><span>02</span><b>{_esc(ui['preparation_navigation_label'])}</b><small>{_esc(ui['preparation_navigation_hint'])}</small></button>
          <button class="workspace-nav-item" type="button" data-workspace-target="query"
            aria-controls="workspace-query" aria-selected="false"><span>03</span><b>{_esc(query_navigation_label)}</b><small>{_esc(ui['query_navigation_hint'])}</small></button>
          <button class="workspace-nav-item" type="button" data-workspace-target="settings"
            aria-controls="workspace-settings" aria-selected="false"><span>04</span><b>项目设置</b><small>模型与上下文</small></button>
        </nav>
      </aside>
      <div class="workspace-content">
        <section id="workspace-documents" class="workspace-view" data-workspace-panel="documents"
          role="tabpanel" aria-label="文档管理">
          <div class="section-heading workspace-view-heading"><div><span class="workspace-view-kicker">文档</span>
            <h2>上传与管理项目资料</h2><p>上传源文件并查看处理状态；解析与分块方式统一在项目设置中调整。</p></div></div>
          <form id="document-upload-form" action="/upload" method="post" enctype="multipart/form-data" class="upload-form processing-form">
            <input type="hidden" name="project_id" value="{_esc(current['id'] if current else '')}">
            <input type="hidden" name="strategy_key" value="{_esc(strategy_key)}">
            <div class="upload-line"><input id="upload-files" type="file" name="files" multiple
              accept=".docx,.pdf,.md,.markdown,.txt,.xlsx,.xls,.csv" required>
              <button class="button primary" type="submit">上传文件</button></div>
            <p class="upload-help">支持多文件上传，单个文件最大 {Config.upload_max_bytes // (1024 * 1024)} MB。</p>
          </form>
          <div class="workspace-section" {'' if current else 'hidden'}><div class="section-heading"><div>
            <h2>项目文档</h2><p>源文件、处理状态和来源链接</p></div></div>
            <div class="table-wrap"><table><thead><tr><th>文件</th><th>大小</th><th>状态</th><th>更新时间</th>
              <th class="actions">操作</th></tr></thead><tbody>{rows}</tbody></table></div>
          </div>
        </section>
        <section id="workspace-build" class="workspace-view" data-workspace-panel="build"
          role="tabpanel" aria-label="{_esc(ui['preparation_navigation_label'])}" hidden>
          <div class="section-heading workspace-view-heading preparation-view-heading"><div>
            <h2>{_esc(ui['preparation_workspace_title'])}</h2><p>{_esc(ui['preparation_workspace_description'])}</p></div>
            {current_block}</div>
          {preparation_task_slot}
          {knowledge_structure_section}
          {process_artifacts_section}
        </section>
        <section id="workspace-query" class="workspace-view" data-workspace-panel="query"
          role="tabpanel" aria-label="{_esc(query_navigation_label)}" hidden>
          {retrieval_section}
        </section>
        <section id="workspace-settings" class="workspace-view" data-workspace-panel="settings"
          role="tabpanel" aria-label="项目设置" hidden>
          {settings_section}
        </section>
      </div>
    </div>
    <dialog id="job-log-dialog" class="log-dialog"><div class="dialog-heading"><div><h2>任务日志</h2><p id="job-log-summary">正在读取最近一次任务...</p></div>
      <button class="icon-button" type="button" data-close-job-log aria-label="关闭">×</button></div>
      <div id="job-log-error" class="job-log-error" hidden></div><pre id="job-history-log" class="history-log"></pre>
      <div class="dialog-actions"><button class="button secondary" type="button" data-close-job-log>关闭</button></div>
    </dialog>
    <dialog id="delete-progress-dialog" class="delete-progress-dialog">
      <div class="dialog-heading"><div><h2>删除文档</h2><p id="delete-progress-message">正在提交删除请求...</p></div></div>
      <div class="delete-progress-body" role="status" aria-live="polite">
        <span id="delete-progress-spinner" class="delete-spinner" aria-hidden="true"></span>
        <span id="delete-progress-result" class="delete-result" hidden aria-hidden="true"></span>
        <div><b id="delete-progress-title">正在取消相关任务</b><p id="delete-progress-detail">请稍候，不要重复点击删除。</p></div>
      </div>
      <div class="progress delete-progress-bar"><i id="delete-progress-bar"></i></div>
      <div class="dialog-actions"><button id="delete-progress-close" class="button secondary" type="button" disabled>关闭</button></div>
    </dialog>
    <dialog id="structure-dialog" class="structure-dialog"><div class="dialog-heading"><div><h2>{'源文档结构' if strategy_key == GRAPHRAG_NATIVE_STRATEGY else '文档结构块'}</h2><p id="structure-summary">正在读取解析结果...</p></div>
      <button class="icon-button" type="button" data-close-structure aria-label="关闭">×</button></div>
      <div id="structure-error" class="job-log-error" hidden></div>
      <div id="structure-tools" class="structure-tools" hidden>
        <input id="structure-search" type="search" placeholder="检索结构块正文、章节或标题路径">
        <select id="structure-type" aria-label="按结构块类型筛选"><option value="">全部类型</option></select>
      </div>
      <div id="structure-list" class="structure-list"></div>
      <div id="structure-pager" class="structure-pager" hidden><span id="structure-range"></span><div>
        <button id="structure-prev" class="button small secondary" type="button">上一页</button>
        <button id="structure-next" class="button small secondary" type="button">下一页</button></div></div>
      <div class="dialog-actions"><button class="button secondary" type="button" data-close-structure>关闭</button></div>
    </dialog>
    <dialog id="chunk-dialog" class="structure-dialog"><div class="dialog-heading"><div><h2>Chunk 分片</h2><p id="chunk-summary">正在读取分块结果...</p></div>
      <button class="icon-button" type="button" data-close-chunks aria-label="关闭">×</button></div>
      <div id="chunk-error" class="job-log-error" hidden></div>
      <div id="chunk-tools" class="structure-tools" hidden>
        <input id="chunk-search" type="search" placeholder="检索知识块正文、章节或编号">
        <select id="chunk-role" aria-label="按知识块角色筛选"><option value="">全部知识块</option><option value="parent">父级摘要块</option><option value="child">检索子块</option><option value="table_row">表格行块</option></select>
      </div>
      <div id="chunk-list" class="structure-list"></div>
      <div id="chunk-pager" class="structure-pager" hidden><span id="chunk-range"></span><div>
        <button id="chunk-prev" class="button small secondary" type="button">上一页</button>
        <button id="chunk-next" class="button small secondary" type="button">下一页</button></div></div>
      <div class="dialog-actions"><button class="button secondary" type="button" data-close-chunks>关闭</button></div>
    </dialog>
    <dialog id="native-artifact-dialog" class="structure-dialog"><div class="dialog-heading"><div>
      <h2 id="native-artifact-title">GraphRAG Native 过程产物</h2>
      <p id="native-artifact-summary">正在读取...</p></div>
      <button class="icon-button" type="button" data-close-native-artifacts aria-label="关闭">×</button></div>
      <div id="native-artifact-error" class="job-log-error" hidden></div>
      <div id="native-artifact-tools" class="structure-tools">
        <select id="native-artifact-kind" aria-label="选择 GraphRAG Native 产物">
          <option value="documents">冻结 Documents</option>
          <option value="source-map">Source Map</option>
          <option value="text-units">TextUnits</option>
          <option value="entities">原生实体</option>
          <option value="relationships">原生关系</option>
          <option value="communities">社区</option>
          <option value="community-reports">社区报告</option>
        </select>
      </div>
      <pre id="native-artifact-content" class="artifact-content"></pre>
      <div class="dialog-actions"><button class="button secondary" type="button" data-close-native-artifacts>关闭</button></div>
    </dialog>
    </main><script>window.KF={{projectId:{int(current['id']) if current else 0},strategyKey:{json.dumps(strategy_key)},taskProfile:{json.dumps(ui['task_profile'])},taskLabels:{json.dumps(ui['task_labels'], ensure_ascii=False)},activeJob:{str(bool(active_job_id)).lower()},activeJobKind:{json.dumps(active_job_kind)},docling:{str(docling_available()).lower()}}};{SCRIPT}</script></body></html>"""


def _render_native_graph_view(project_id: int) -> str:
    """Render an application-native graph explorer; never open Hubble."""

    payload = SERVICE.graphrag_native_status(project_id)
    run = payload.get("run") if isinstance(payload, dict) else None
    if not run or not payload.get("fresh"):
        raise GraphRAGNativeUnavailable(
            str(payload.get("stale_reason") or payload.get("unavailable_reason") or "没有可查看的 GraphRAG Native run")
        )
    # This browser intentionally reads only the immutable native bundle APIs.
    # It has no physical-graph dependency and keeps the pre-existing relation
    # table as an audit view alongside the visual explorer.
    script = r"""
const PROJECT_ID=__PROJECT_ID__;
const NS='http://www.w3.org/2000/svg';
const colors=['#2677d8','#b54cda','#d85c49','#119c87','#d18b18','#5974d9','#c45379','#1d9aa7','#7b8b3d','#8b5e3c'];
const state={graph:null,communities:[],reports:[],nodeById:new Map(),titleToNode:new Map(),communityByNode:new Map(),communityMembershipByNode:new Map(),selectedCommunity:'',query:'',showIsolates:true,selectedNodeId:'',selectedEdgeIndex:-1,positions:new Map()};
const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const text=value=>String(value??'').replace(/\s+/g,' ').trim();
const short=(value,length=180)=>{const valueText=text(value);return valueText.length>length?valueText.slice(0,length-1)+'…':valueText};
const communityKey=item=>String(item.level??'?')+':'+String(item.community??item.human_readable_id??item.id??'unknown');
const communityLabel=item=>'社区 '+String(item.community??item.human_readable_id??item.id??'—')+' · L'+String(item.level??0);
const nodeCommunity=node=>state.communityByNode.get(String(node.id))||null;
const nodeMatchesCommunity=(node,key)=>!key||Boolean(state.communityMembershipByNode.get(String(node.id))?.has(key));
const colorFor=key=>colors[Math.abs([...String(key)].reduce((sum,char)=>sum+char.charCodeAt(0),0))%colors.length];
const createSvg=(tag,attrs={})=>{const element=document.createElementNS(NS,tag);for(const [name,value] of Object.entries(attrs))element.setAttribute(name,String(value));return element};

function reportsFor(community){
  return state.reports.filter(report=>String(report.community??report.id??'')===String(community.community??community.id??'')&&String(report.level??'')===String(community.level??''));
}
function matchingNodes(){
  const all=state.graph?.nodes||[];
  const query=state.query.toLowerCase();
  const matching=node=>!query||[node.title,node.type,node.description].some(value=>String(value??'').toLowerCase().includes(query));
  const inCommunity=node=>nodeMatchesCommunity(node,state.selectedCommunity);
  return all.filter(node=>inCommunity(node)&&matching(node)&&(
    state.showIsolates||Number(node.degree||0)>0
  ));
}
function edgeNodes(edge){
  return [state.titleToNode.get(String(edge.source??'')),state.titleToNode.get(String(edge.target??''))];
}
function visibleGraph(){
  let nodes=matchingNodes();
  let ids=new Set(nodes.map(node=>String(node.id)));
  let edges=(state.graph?.edges||[]).filter(edge=>{
    const [source,target]=edgeNodes(edge);
    return source&&target&&ids.has(String(source.id))&&ids.has(String(target.id));
  });
  // A text search remains useful for topology: retain the direct neighbours
  // of matching nodes even when only one endpoint contains the query.
  if(state.query){
    const seed=new Set(nodes.map(node=>String(node.id)));
    for(const edge of state.graph?.edges||[]){
      const [source,target]=edgeNodes(edge);
      if(source&&target&&(seed.has(String(source.id))||seed.has(String(target.id)))){
        ids.add(String(source.id));ids.add(String(target.id));
      }
    }
    nodes=(state.graph?.nodes||[]).filter(node=>ids.has(String(node.id))&&nodeMatchesCommunity(node,state.selectedCommunity));
    ids=new Set(nodes.map(node=>String(node.id)));
    edges=(state.graph?.edges||[]).filter(edge=>{const [source,target]=edgeNodes(edge);return source&&target&&ids.has(String(source.id))&&ids.has(String(target.id));});
  }
  return {nodes,edges};
}
function layout(nodes){
  const groups=new Map();
  for(const node of nodes){
    const community=nodeCommunity(node);
    const key=community?.key||'unassigned';
    if(!groups.has(key))groups.set(key,{key,label:community?.label||'未归入社区',nodes:[]});
    groups.get(key).nodes.push(node);
  }
  const ordered=[...groups.values()].sort((left,right)=>left.label.localeCompare(right.label));
  const columns=Math.max(1,Math.ceil(Math.sqrt(ordered.length)));
  const rows=Math.max(1,Math.ceil(ordered.length/columns));
  const width=1200,height=760,cellWidth=width/columns,cellHeight=height/rows;
  const positions=new Map(),groupLabels=[];
  ordered.forEach((group,index)=>{
    const column=index%columns,row=Math.floor(index/columns);
    const left=column*cellWidth,top=row*cellHeight;
    const count=group.nodes.length,innerColumns=Math.max(1,Math.ceil(Math.sqrt(count)));
    const innerRows=Math.max(1,Math.ceil(count/innerColumns));
    group.nodes.sort((leftNode,rightNode)=>String(leftNode.title??'').localeCompare(String(rightNode.title??''))).forEach((node,nodeIndex)=>{
      const localColumn=nodeIndex%innerColumns,localRow=Math.floor(nodeIndex/innerColumns);
      positions.set(String(node.id),{
        x:left+(localColumn+0.5)*(cellWidth/innerColumns),
        y:top+30+(localRow+0.5)*((cellHeight-40)/innerRows),key:group.key,
      });
    });
    groupLabels.push({x:left+12,y:top+19,label:group.label,key:group.key,count});
  });
  state.positions=positions;
  return groupLabels;
}
function renderGraph(){
  const {nodes,edges}=visibleGraph();
  const svg=$('native-network'),groups=$('community-groups'),edgeLayer=$('edge-layer'),nodeLayer=$('node-layer');
  groups.replaceChildren();edgeLayer.replaceChildren();nodeLayer.replaceChildren();
  const labels=layout(nodes);
  for(const group of labels){
    const label=createSvg('text',{x:group.x,y:group.y,class:'community-label',fill:colorFor(group.key)});
    label.textContent=group.label+' ('+group.count+')';groups.append(label);
  }
  const selectedNode=state.selectedNodeId;
  const neighbourIds=new Set();
  if(selectedNode){
    for(const edge of edges){const [source,target]=edgeNodes(edge);if(String(source.id)===selectedNode)neighbourIds.add(String(target.id));if(String(target.id)===selectedNode)neighbourIds.add(String(source.id));}
  }
  edges.forEach((edge,index)=>{
    const [source,target]=edgeNodes(edge),sourcePoint=state.positions.get(String(source.id)),targetPoint=state.positions.get(String(target.id));
    if(!sourcePoint||!targetPoint)return;
    const related=!selectedNode||String(source.id)===selectedNode||String(target.id)===selectedNode;
    const line=createSvg('line',{x1:sourcePoint.x,y1:sourcePoint.y,x2:targetPoint.x,y2:targetPoint.y,class:'native-edge'+(related?'':' dim'),"marker-end":'url(#native-arrow)',"data-edge":index});
    line.style.strokeWidth=String(Math.min(5,1+Math.log1p(Number(edge.weight)||1)));
    const title=createSvg('title');title.textContent=String(edge.source??'')+' → '+String(edge.target??'')+'\n'+String(edge.description??'');line.append(title);
    line.addEventListener('click',event=>{event.stopPropagation();state.selectedEdgeIndex=index;state.selectedNodeId='';renderGraph();renderDetails();});edgeLayer.append(line);
  });
  for(const node of nodes){
    const point=state.positions.get(String(node.id));if(!point)continue;
    const relevant=!selectedNode||String(node.id)===selectedNode||neighbourIds.has(String(node.id));
    const group=createSvg('g',{class:'native-node'+(relevant?'':' dim')+(String(node.id)===selectedNode?' selected':''),transform:'translate('+point.x+' '+point.y+')'});
    const circle=createSvg('circle',{r:Math.min(17,7+Math.log1p(Number(node.degree)||0)*3),fill:colorFor(point.key)});
    const title=createSvg('title');title.textContent=String(node.title??'')+'\n'+String(node.type??'')+' · degree '+String(node.degree??0);circle.append(title);
    const label=createSvg('text',{x:0,y:25,class:'node-label'});label.textContent=short(node.title,28);
    group.append(circle,label);group.addEventListener('click',event=>{event.stopPropagation();state.selectedNodeId=String(node.id);state.selectedEdgeIndex=-1;renderGraph();renderDetails();});nodeLayer.append(group);
  }
  svg.setAttribute('aria-label','GraphRAG Native 实体关系图：'+nodes.length+' 个实体，'+edges.length+' 条关系');
  $('network-count').textContent='当前显示 '+nodes.length+' / '+String(state.graph?.total_entities??nodes.length)+' 个实体，'+edges.length+' / '+String(state.graph?.total_relationships??edges.length)+' 条关系';
  renderAudit(edges);
}
function evidenceList(ids){
  const values=Array.isArray(ids)?ids:[];
  return values.length?'<code>'+esc(values.slice(0,6).join(', '))+(values.length>6?' …':'')+'</code>':'—';
}
function renderDetails(){
  const panel=$('selection-details');
  if(state.selectedEdgeIndex>=0){
    const edge=visibleGraph().edges[state.selectedEdgeIndex];
    if(!edge){panel.innerHTML='<p>请选择一条关系或实体。</p>';return;}
    panel.innerHTML='<h3>关系详情</h3><dl><dt>源实体</dt><dd>'+esc(edge.source)+'</dd><dt>目标实体</dt><dd>'+esc(edge.target)+'</dd><dt>权重</dt><dd>'+esc(edge.weight)+'</dd><dt>关系描述</dt><dd>'+esc(edge.description)+'</dd><dt>TextUnit 证据</dt><dd>'+evidenceList(edge.text_unit_ids)+'</dd></dl>';
    return;
  }
  const node=state.nodeById.get(state.selectedNodeId);
  if(!node){panel.innerHTML='<h3>浏览说明</h3><p>圆点是 GraphRAG 原生实体，箭头是原生关系。颜色按 GraphRAG 社区划分；点击实体或关系可查看描述和 TextUnit 证据。</p>';return;}
  const community=nodeCommunity(node);
  const neighbours=visibleGraph().edges.filter(edge=>String(edge.source)===String(node.title)||String(edge.target)===String(node.title));
  panel.innerHTML='<h3>'+esc(node.title)+'</h3><dl><dt>类型</dt><dd>'+esc(node.type||'—')+'</dd><dt>度数</dt><dd>'+esc(node.degree||0)+'</dd><dt>社区</dt><dd>'+esc(community?.label||'未归入社区')+'</dd><dt>实体描述</dt><dd>'+esc(short(node.description,800)||'—')+'</dd></dl><h4>相邻关系 ('+neighbours.length+')</h4><ol class="neighbour-list">'+neighbours.slice(0,12).map(edge=>'<li><button type="button" data-detail-edge="'+esc(edge.id)+'">'+esc(edge.source)+' → '+esc(edge.target)+'</button></li>').join('')+'</ol>';
  for(const button of panel.querySelectorAll('[data-detail-edge]'))button.addEventListener('click',()=>{
    const index=visibleGraph().edges.findIndex(edge=>String(edge.id)===button.dataset.detailEdge);state.selectedEdgeIndex=index;state.selectedNodeId='';renderGraph();renderDetails();
  });
}
function renderCommunities(){
  const box=$('community-list');
  box.innerHTML=state.communities.map(community=>{
    const report=reportsFor(community)[0]||{};
    const active=communityKey(community)===state.selectedCommunity?' active':'';
    return '<button type="button" class="community-card'+active+'" data-community="'+esc(communityKey(community))+'"><span class="community-dot" style="background:'+colorFor(communityKey(community))+'"></span><b>'+esc(communityLabel(community))+'</b><small>'+esc(community.size??(community.entity_ids||[]).length)+' 个实体 · '+esc((community.relationship_ids||[]).length)+' 条关系</small><em>'+esc(short(report.summary??report.full_content??report.description??'暂无社区摘要',220))+'</em></button>';
  }).join('')||'<p>当前 bundle 未提供社区记录。</p>';
  for(const button of box.querySelectorAll('[data-community]'))button.addEventListener('click',()=>{
    state.selectedCommunity=state.selectedCommunity===button.dataset.community?'':button.dataset.community;$('community-filter').value=state.selectedCommunity;state.selectedNodeId='';state.selectedEdgeIndex=-1;renderGraph();renderDetails();renderCommunities();
  });
}
function renderAudit(edges){
  $('audit-summary').textContent='显示 '+edges.length+' / '+String(state.graph?.total_relationships??edges.length)+' 条原生关系；表格用于审计，网络图用于探索连接。';
  $('audit-rows').innerHTML=edges.map(edge=>'<tr><td>'+esc(edge.source)+'</td><td>'+esc(edge.description)+'</td><td>'+esc(edge.target)+'</td><td>'+esc(edge.weight)+'</td><td>'+evidenceList(edge.text_unit_ids)+'</td></tr>').join('');
}
function renderControls(){
  const select=$('community-filter');select.innerHTML='<option value="">全部社区</option>'+state.communities.map(community=>'<option value="'+esc(communityKey(community))+'">'+esc(communityLabel(community))+'</option>').join('');select.value=state.selectedCommunity;
}
function setTab(tab){
  $('network-panel').hidden=tab!=='network';$('audit-panel').hidden=tab!=='audit';
  for(const button of document.querySelectorAll('[data-native-tab]'))button.classList.toggle('active',button.dataset.nativeTab===tab);
}
async function load(){
  try{
    const base='/api/projects/'+PROJECT_ID+'/graphrag-native/';
    const [graphResponse,communitiesResponse,reportsResponse]=await Promise.all([fetch(base+'graph?limit=1000&include_isolates=1'),fetch(base+'communities?limit=500'),fetch(base+'community-reports?limit=500')]);
    const [graph,communities,reports]=await Promise.all([graphResponse.json(),communitiesResponse.json(),reportsResponse.json()]);
    if(!graphResponse.ok)throw Error(graph.error||'读取原生图失败');if(!communitiesResponse.ok)throw Error(communities.error||'读取社区失败');if(!reportsResponse.ok)throw Error(reports.error||'读取社区报告失败');
    state.graph=graph;state.communities=(communities.items||[]).slice().sort((left,right)=>Number(left.level??0)-Number(right.level??0)||Number(left.community??0)-Number(right.community??0));state.reports=reports.items||[];
    for(const node of graph.nodes||[]){state.nodeById.set(String(node.id),node);if(!state.titleToNode.has(String(node.title??'')))state.titleToNode.set(String(node.title??''),node);}
    for(const community of state.communities){const entry={key:communityKey(community),label:communityLabel(community),raw:community};for(const entityId of community.entity_ids||[]){const nodeId=String(entityId);if(!state.communityMembershipByNode.has(nodeId))state.communityMembershipByNode.set(nodeId,new Map());state.communityMembershipByNode.get(nodeId).set(entry.key,entry);if(!state.communityByNode.has(nodeId))state.communityByNode.set(nodeId,entry);}}
    renderControls();renderGraph();renderDetails();renderCommunities();$('loading').hidden=true;$('viewer').hidden=false;
  }catch(error){$('loading').textContent='无法读取原生 bundle：'+(error.message||error);}
}
for(const button of document.querySelectorAll('[data-native-tab]'))button.addEventListener('click',()=>setTab(button.dataset.nativeTab));
$('native-network').addEventListener('click',()=>{if(!state.selectedNodeId&&state.selectedEdgeIndex<0)return;state.selectedNodeId='';state.selectedEdgeIndex=-1;renderGraph();renderDetails();});
$('apply-filter').addEventListener('click',()=>{state.query=$('entity-filter').value;state.selectedNodeId='';state.selectedEdgeIndex=-1;renderGraph();renderDetails();});
$('community-filter').addEventListener('change',event=>{state.selectedCommunity=event.target.value;state.selectedNodeId='';state.selectedEdgeIndex=-1;renderGraph();renderDetails();renderCommunities();});
$('toggle-isolates').addEventListener('change',event=>{state.showIsolates=event.target.checked;renderGraph();renderDetails();});
$('reset-graph').addEventListener('click',()=>{state.selectedCommunity='';state.query='';state.selectedNodeId='';state.selectedEdgeIndex=-1;$('entity-filter').value='';$('community-filter').value='';renderGraph();renderDetails();renderCommunities();});
load();
"""
    return (
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>GraphRAG Native 图谱浏览</title>
<style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#172033;background:#f4f7fb}body{margin:0}.page{max-width:1480px;margin:0 auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:start;flex-wrap:wrap}.top h1{margin:0;font-size:25px}.top p{color:#536981;margin:.4rem 0 0}.tabs,.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.tab,button{font:inherit}.tab,.controls button{border:1px solid #c5d4e5;border-radius:6px;background:#fff;padding:8px 12px;color:#294764;cursor:pointer}.tab.active{background:#2677d8;border-color:#2677d8;color:#fff}.controls{margin:18px 0}.controls input,.controls select{border:1px solid #b9cade;border-radius:6px;padding:9px 10px;font:inherit;background:#fff}.controls input{min-width:260px;flex:1}.controls label{display:flex;align-items:center;gap:6px;color:#435b75}.network-layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:16px}.card{background:#fff;border:1px solid #d7e2ee;border-radius:9px;box-shadow:0 8px 20px rgba(25,53,84,.05)}.graph-card{min-height:740px;overflow:auto}.graph-card svg{display:block;width:100%;min-width:880px;min-height:720px;background:linear-gradient(145deg,#fbfdff,#eef5fb)}.community-label{font-size:13px;font-weight:700}.native-edge{stroke:#7f94aa;stroke-opacity:.72;cursor:pointer}.native-edge.dim,.native-node.dim{opacity:.12}.native-node{cursor:pointer}.native-node circle{stroke:#fff;stroke-width:2}.native-node.selected circle{stroke:#142b45;stroke-width:4}.node-label{font-size:10px;text-anchor:middle;fill:#273e57;pointer-events:none}.side{padding:17px;min-height:280px}.side h3{margin:0 0 12px}.side h4{margin:18px 0 8px}.side dl{margin:0;display:grid;grid-template-columns:88px 1fr;gap:8px;font-size:13px}.side dt{color:#64778a;font-weight:700}.side dd{margin:0;overflow-wrap:anywhere;line-height:1.45}.side code,.audit code{display:inline-block;max-width:100%;overflow-wrap:anywhere;font-size:11px;color:#465c73}.neighbour-list{padding-left:18px;margin:0}.neighbour-list button{padding:0;border:0;background:transparent;color:#1766b5;text-align:left;cursor:pointer;font-size:12px}.summary{color:#60758b;font-size:13px;margin:0 0 12px}.communities{margin-top:18px;padding:17px}.communities h2,.audit h2{font-size:18px;margin:0 0 6px}.community-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:10px}.community-card{border:1px solid #d6e2ed;background:#fff;border-radius:7px;padding:12px;text-align:left;cursor:pointer;color:#243d57}.community-card.active{border-color:#2677d8;box-shadow:0 0 0 2px #d8eaff}.community-card b,.community-card small,.community-card em{display:block}.community-card small{color:#6a7f94;margin:5px 0}.community-card em{font-size:12px;color:#526a81;font-style:normal;line-height:1.4}.community-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}.audit{margin-top:18px;padding:17px;overflow:auto}.audit table{border-collapse:collapse;width:100%;min-width:980px}.audit th,.audit td{padding:10px;border-bottom:1px solid #e0e8f0;vertical-align:top;text-align:left}.audit th{background:#f5f8fc}.audit td:nth-child(2){max-width:460px;line-height:1.35}@media(max-width:900px){.page{padding:16px}.network-layout{grid-template-columns:1fr}.graph-card{min-height:520px}.controls input{min-width:180px}.top{flex-direction:column}.graph-card svg{min-width:760px}}
</style></head><body><main class="page"><header class="top"><div><h1>GraphRAG Native 原生图谱浏览</h1><p>run: <code>"""
        + _esc(run["run_id"])
        + """</code> · GraphRAG """
        + _esc(run.get("graphrag_version"))
        + """ · 仅读取当前 fresh native bundle，不调用 HugeGraph 或旧检索。</p></div><div class="tabs"><button class="tab active" type="button" data-native-tab="network">图谱连接</button><button class="tab" type="button" data-native-tab="audit">关系审计表</button></div></header><p id="loading">正在加载原生实体、关系与社区…</p><section id="viewer" hidden><div class="controls"><input id="entity-filter" type="search" placeholder="输入实体、类型或描述"><button id="apply-filter" type="button">检索</button><select id="community-filter" aria-label="按社区筛选"></select><label><input id="toggle-isolates" type="checkbox" checked>显示孤立实体</label><button id="reset-graph" type="button">重置视图</button></div><section id="network-panel"><div class="network-layout"><div class="card graph-card"><svg id="native-network" viewBox="0 0 1200 760" role="img"><defs><marker id="native-arrow" markerWidth="8" markerHeight="8" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#7f94aa"></path></marker></defs><g id="community-groups"></g><g id="edge-layer"></g><g id="node-layer"></g></svg></div><aside id="selection-details" class="card side"></aside></div><p id="network-count" class="summary"></p><section class="card communities"><h2>GraphRAG 社区</h2><p class="summary">社区由原生 Leiden 聚类及社区报告生成；点击卡片可筛选图中的成员和关系。</p><div id="community-list" class="community-grid"></div></section></section><section id="audit-panel" class="card audit" hidden><h2>原生关系审计表</h2><p id="audit-summary" class="summary"></p><table><thead><tr><th>源实体</th><th>关系描述</th><th>目标实体</th><th>weight</th><th>TextUnit 证据</th></tr></thead><tbody id="audit-rows"></tbody></table></section></section></main><script>"""
        + script.replace("__PROJECT_ID__", str(int(project_id)))
        + """</script></body></html>"""
    )


class AppHandler(BaseHTTPRequestHandler):
    server_version = "KnowledgeForge/0.2"

    def log_message(self, fmt: str, *args) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def _project(self) -> int | None:
        value = parse_qs(urlparse(self.path).query).get("project", [""])[0]
        if value.isdigit(): return int(value)
        cookie = SimpleCookie(self.headers.get("Cookie")); item = cookie.get("knowledge_forge_project")
        return int(item.value) if item and item.value.isdigit() else None

    def _respond(self, status: int, body: str | bytes, content_type: str = "text/html; charset=utf-8", cookie: str = "") -> None:
        payload = body.encode() if isinstance(body, str) else body
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(payload)))
        if cookie: self.send_header("Set-Cookie", cookie)
        self.end_headers(); self.wfile.write(payload)

    def _json(self, status: int, data: Any) -> None:
        self._respond(status, json.dumps(data, ensure_ascii=False, default=str), "application/json; charset=utf-8")

    def _redirect(self, target: str, project_id: int | None = None) -> None:
        self.send_response(HTTPStatus.SEE_OTHER); self.send_header("Location", target)
        if project_id: self.send_header("Set-Cookie", f"knowledge_forge_project={project_id}; Path=/; SameSite=Lax")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            params = parse_qs(parsed.query)
            text = params.get("message", [""])[0]
            requested = params.get("project", [""])[0]
            if requested.isdigit():
                selection = _project_binding(int(requested)) or {}
                if str(selection.get("association_status") or "") == "bound":
                    self._redirect(
                        _workspace_path(
                            int(requested), str(selection["selected_strategy"]),
                        ),
                        int(requested),
                    )
                    return
            self._respond(
                200,
                render_home_page(
                    (params.get("kind", [""])[0], text) if text else None,
                ),
            )
            return
        match = re.fullmatch(
            r"/projects/(\d+)/(knowledge-forge|graphrag-native)", parsed.path,
        )
        if match:
            project_id = int(match.group(1))
            strategy_key = (
                GRAPHRAG_NATIVE_STRATEGY
                if match.group(2) == "graphrag-native"
                else ENHANCED_SHADOW_STRATEGY
            )
            try:
                SERVICE._assert_project_strategy(project_id, strategy_key)
                params = parse_qs(parsed.query)
                text = params.get("message", [""])[0]
                self._respond(
                    200, render_page(
                        project_id, strategy_key,
                        (params.get("kind", [""])[0], text) if text else None,
                    ),
                    cookie=f"knowledge_forge_project={project_id}; Path=/; SameSite=Lax",
                )
            except LookupError as exc:
                self._respond(404, str(exc), "text/plain; charset=utf-8")
            except Exception as exc:
                self._redirect(f"/?kind=error&message={quote(str(exc))}")
            return
        if parsed.path.startswith("/api/jobs/"):
            job = SERVICE.get_job(parsed.path.rsplit("/", 1)[-1]); self._json(200 if job else 404, job or {"error": "任务不存在"}); return
        if parsed.path.startswith("/api/job-batches/"):
            batch = SERVICE.get_ingestion_batch(parsed.path.rsplit("/", 1)[-1])
            self._json(200 if batch else 404, batch or {"error": "任务批次不存在"}); return
        match = re.fullmatch(r"/api/projects/(\d+)/statistics", parsed.path)
        if match:
            project_id = int(match.group(1))
            try:
                if not SERVICE.db.get_project(project_id):
                    raise LookupError("项目不存在")
                strategy_key = SERVICE._assert_project_strategy(project_id)
                graph_state = (
                    None if strategy_key == GRAPHRAG_NATIVE_STRATEGY
                    else SERVICE.db.get_project_strategy_graph(project_id, strategy_key)
                )
                self._json(200, _strategy_statistics(project_id, strategy_key, graph_state))
            except LookupError as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                self._json(409, {"error": str(exc)})
            return
        match = re.fullmatch(
            r"/api/projects/(\d+)/graphrag-native/source-documents", parsed.path,
        )
        if match:
            params = parse_qs(parsed.query)
            try:
                document_value = params.get("document_id", [""])[0]
                self._json(200, SERVICE.graphrag_native_source_documents(
                    int(match.group(1)),
                    document_id=int(document_value) if document_value.isdigit() else None,
                    offset=int(params.get("offset", ["0"])[0]),
                    limit=int(params.get("limit", ["100"])[0]),
                ))
            except LookupError as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                self._json(409, {"error": str(exc)})
            return
        match = re.fullmatch(
            r"/api/projects/(\d+)/graphrag-native/artifacts(?:/([a-z-]+))?",
            parsed.path,
        )
        if match:
            params = parse_qs(parsed.query)
            try:
                if match.group(2):
                    data = SERVICE.graphrag_native_artifact_rows(
                        int(match.group(1)), match.group(2),
                        offset=int(params.get("offset", ["0"])[0]),
                        limit=int(params.get("limit", ["100"])[0]),
                    )
                else:
                    data = SERVICE.graphrag_native_artifact_catalog(int(match.group(1)))
                self._json(200, data)
            except LookupError as exc:
                self._json(404, {"error": str(exc)})
            except (GraphRAGNativeStale, ValueError) as exc:
                self._json(409, {"error": str(exc)})
            except GraphRAGNativeUnavailable as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                self._json(503, {"error": str(exc)})
            return
        match = re.fullmatch(
            r"/api/projects/(\d+)/knowledge-forge/graph", parsed.path,
        )
        if match:
            params = parse_qs(parsed.query)
            try:
                include_isolates = str(
                    params.get("include_isolates", ["1"])[0]
                ).lower() in {"1", "true", "yes", "on"}
                self._json(200, SERVICE.knowledge_forge_graph(
                    int(match.group(1)),
                    limit=int(params.get("limit", ["1000"])[0]),
                    include_isolates=include_isolates,
                ))
            except LookupError as exc:
                self._json(404, {"error": str(exc)})
            except GraphNotReadyError as exc:
                self._json(409, {"error": str(exc)})
            except (TypeError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(503, {"error": str(exc)})
            return
        match = re.fullmatch(r"/api/projects/(\d+)/graphrag-native/(status|graph|communities|community-reports)", parsed.path)
        if match:
            project_id, resource = int(match.group(1)), match.group(2)
            params = parse_qs(parsed.query)
            try:
                if resource == "status":
                    data = SERVICE.graphrag_native_status(project_id)
                elif resource == "graph":
                    include_isolates = str(params.get("include_isolates", [""])[0]).lower() in {"1", "true", "yes", "on"}
                    data = SERVICE.graphrag_native_graph(
                        project_id, offset=int(params.get("offset", ["0"])[0]),
                        limit=int(params.get("limit", ["200"])[0]),
                        include_isolates=include_isolates,
                    )
                elif resource == "communities":
                    data = SERVICE.graphrag_native_communities(project_id, offset=int(params.get("offset", ["0"])[0]), limit=int(params.get("limit", ["100"])[0]))
                else:
                    data = SERVICE.graphrag_native_community_reports(project_id, offset=int(params.get("offset", ["0"])[0]), limit=int(params.get("limit", ["100"])[0]))
                self._json(200, data)
            except LookupError as exc:
                self._json(404, {"error": str(exc)})
            except GraphRAGNativeStale as exc:
                self._json(409, {"error": str(exc)})
            except GraphRAGNativeUnavailable as exc:
                self._json(404, {"error": str(exc)})
            except Exception as exc:
                self._json(503, {"error": str(exc)})
            return
        match = re.fullmatch(r"/api/documents/(\d+)/structure", parsed.path)
        if match:
            try: self._json(200, SERVICE.document_structure(int(match.group(1))))
            except LookupError as exc: self._json(404, {"error": str(exc)})
            except Exception as exc: self._json(500, {"error": str(exc)})
            return
        match = re.fullmatch(r"/api/documents/(\d+)/chunks", parsed.path)
        if match:
            try: self._json(200, SERVICE.document_chunks(int(match.group(1))))
            except LookupError as exc: self._json(404, {"error": str(exc)})
            except Exception as exc: self._json(500, {"error": str(exc)})
            return
        match = re.fullmatch(r"/api/documents/(\d+)/download", parsed.path)
        if match:
            self._download_document(int(match.group(1))); return
        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            try:
                if params.get("strategy", [STABLE_STRATEGY])[0] == GRAPHRAG_NATIVE_STRATEGY:
                    raise ValueError("GraphRAG Native 必须使用 /graphrag-native/query，不会回退旧检索")
                self._json(200, SERVICE.search(int(params.get("project", ["0"])[0]), params.get("q", [""])[0]))
            except GraphNotReadyError as exc:
                self._json(409, {"error": str(exc)})
            except Exception as exc: self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/api/models/limits":
            params = parse_qs(parsed.query)
            try:
                self._json(200, SERVICE.project_token_limits(
                    params.get("llm_model", [""])[0],
                    params.get("embedding_model", [""])[0],
                ))
            except Exception as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/graph/view":
            try:
                params = parse_qs(parsed.query)
                project_id = int(params.get("project", ["0"])[0])
                strategy_key = params.get(
                    "strategy", [ENHANCED_SHADOW_STRATEGY],
                )[0]
                if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
                    SERVICE._assert_project_strategy(
                        project_id, GRAPHRAG_NATIVE_STRATEGY,
                    )
                    self._redirect(
                        f"/projects/{project_id}/graphrag-native#query", project_id,
                    )
                    return
                SERVICE._assert_project_strategy(
                    project_id, ENHANCED_SHADOW_STRATEGY,
                )
                if strategy_key != ENHANCED_SHADOW_STRATEGY:
                    raise ValueError("当前项目只允许打开 Knowledge Forge 图谱")
                self._redirect(
                    f"/projects/{project_id}/knowledge-forge#query", project_id,
                )
            except Exception as exc:
                self._redirect(f"/?kind=error&message={quote(str(exc))}", self._project())
            return
        if parsed.path == "/health":
            try: self._json(200, SERVICE.health())
            except Exception as exc: self._json(503, {"status": "error", "error": str(exc)})
            return
        self._respond(404, "Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/projects":
                data = self._form()
                project = SERVICE.create_project(
                    data.get("name", ""), data.get("description", ""),
                    data.get("llm_model", ""), data.get("embedding_model", ""),
                    strategy_key=data.get("strategy_key", ""),
                )
                self._redirect(
                    _workspace_path(int(project["id"]), data.get("strategy_key", "")),
                    int(project["id"]),
                )
                return
            match = re.fullmatch(r"/projects/(\d+)/strategy", parsed.path)
            if match:
                data = self._form()
                row = SERVICE.bind_existing_project_strategy(
                    int(match.group(1)), data.get("strategy_key", ""),
                    expected_generation=int(data.get("expected_generation", "0")),
                )
                self._redirect(
                    _workspace_path(int(match.group(1)), str(row["selected_strategy"])),
                    int(match.group(1)),
                )
                return
            match = re.fullmatch(r"/projects/(\d+)/models", parsed.path)
            if match:
                project_id = int(match.group(1))
                SERVICE._assert_project_strategy(project_id)
                data = self._form(); project = SERVICE.update_project_settings(
                    project_id, data.get("llm_model", ""), data.get("embedding_model", ""),
                    int(data.get("model_max_tokens", "0")), int(data.get("chunk_max_tokens", "0")))
                strategy_key = SERVICE._assert_project_strategy(int(project["id"]))
                self._redirect(
                    _workspace_path(int(project["id"]), strategy_key)
                    + f"?kind=ok&message={quote('项目设置已更新')}#settings",
                    int(project["id"]),
                )
                return
            if parsed.path == "/upload": self._upload(); return
            if parsed.path == "/api/models/sync":
                models = SERVICE.sync_ollama_models(); self._json(200, {"ok": True, "count": len(models), "models": models}); return
            match = re.fullmatch(r"/api/projects/(\d+)/graphrag-native/(build|cancel|query)", parsed.path)
            if match:
                project_id, action = int(match.group(1)), match.group(2)
                try:
                    if action == "build":
                        self._json(202, {"ok": True, **SERVICE.submit_graphrag_native_build(project_id)})
                    elif action == "cancel":
                        self._json(202, {"ok": True, "job": SERVICE.cancel_graphrag_native_build(project_id)})
                    else:
                        data = self._form()
                        self._json(200, SERVICE.graphrag_native_query(
                            project_id, mode=str(data.get("mode") or "local"), question=str(data.get("question") or ""),
                        ))
                except LookupError as exc:
                    self._json(404, {"error": str(exc)})
                except GraphRAGNativeStale as exc:
                    self._json(409, {"error": str(exc)})
                except (GraphRAGNativeUnavailable, ValueError) as exc:
                    self._json(409, {"error": str(exc)})
                except Exception as exc:
                    self._json(503, {"error": str(exc)})
                return
            match = re.fullmatch(r"/api/documents/(\d+)/ingest", parsed.path)
            if match:
                strategy_key = parse_qs(parsed.query).get("strategy", [STABLE_STRATEGY])[0]
                self._json(202, {
                    "ok": True,
                    "job_id": SERVICE.submit_ingestion(
                        int(match.group(1)), strategy_key=strategy_key,
                    ),
                })
                return
            match = re.fullmatch(r"/api/projects/(\d+)/ingest", parsed.path)
            if match:
                strategy_key = parse_qs(parsed.query).get("strategy", [STABLE_STRATEGY])[0]
                result = SERVICE.submit_project_ingestion(
                    int(match.group(1)), strategy_key=strategy_key,
                )
                self._json(202, {"ok": True, **result})
                return
            match = re.fullmatch(r"/api/projects/(\d+)/knowledge-forge/build", parsed.path)
            if match:
                result = SERVICE.submit_project_ingestion(
                    int(match.group(1)), strategy_key=ENHANCED_SHADOW_STRATEGY,
                )
                self._json(202, {"ok": True, **result})
                return
            match = re.fullmatch(
                r"/api/projects/(\d+)/(knowledge-forge|graphrag-native)/prepare",
                parsed.path,
            )
            if match:
                expected = (
                    GRAPHRAG_NATIVE_STRATEGY
                    if match.group(2) == "graphrag-native"
                    else ENHANCED_SHADOW_STRATEGY
                )
                SERVICE._assert_project_strategy(int(match.group(1)), expected)
                result = SERVICE.submit_project_preparation(int(match.group(1)))
                self._json(202, {"ok": True, **result})
                return
            match = re.fullmatch(r"/api/projects/(\d+)/prepare", parsed.path)
            if match:
                result = SERVICE.submit_project_preparation(int(match.group(1)))
                self._json(202, {"ok": True, **result})
                return
            match = re.fullmatch(r"/api/documents/(\d+)/delete", parsed.path)
            if match:
                job_id = SERVICE.delete_document(int(match.group(1)))
                self._json(202, {"ok": True, "job_id": job_id})
                return
            match = re.fullmatch(r"/api/projects/(\d+)/graph", parsed.path)
            if match: self._json(202, {"ok": True, "job_id": SERVICE.submit_graph_build(int(match.group(1)))}); return
            match = re.fullmatch(r"/api/projects/(\d+)/schema/reassess", parsed.path)
            if match: self._json(202, {"ok": True, "job_id": SERVICE.submit_schema_reassessment(int(match.group(1)))}); return
            match = re.fullmatch(r"/api/projects/(\d+)/treatment/([^/]+)/activate", parsed.path)
            if match:
                result = SERVICE.activate_treatment_snapshot(
                    int(match.group(1)), unquote(match.group(2)),
                )
                self._json(200, {"ok": True, "result": result})
                return
            self._json(404, {"error": "接口不存在"})
        except TreatmentMaterializationRequired as exc:
            self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
        except Exception as exc:
            LOGGER.exception("Request failed: %s", self.path)
            if self.path.startswith("/api/"): self._json(400, {"ok": False, "error": str(exc)})
            else: self._redirect(f"/?kind=error&message={quote(str(exc))}", self._project())

    def _form(self) -> dict[str, str]:
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8", "ignore")
        return {key: values[0] for key, values in parse_qs(raw).items()}

    def _upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"): raise ValueError("上传请求格式错误")
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type})
        project_id = int(form.getfirst("project_id", "0")); entries = form["files"] if "files" in form else []
        if not isinstance(entries, list): entries = [entries]
        entries = [item for item in entries if getattr(item, "filename", "")]
        project = SERVICE.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        requested_parser = form.getfirst("requested_parser", "auto") or "auto"
        requested_strategy = form.getfirst("requested_strategy", "auto") or "auto"
        for item in entries:
            validate_processing_request(item.filename, requested_parser, requested_strategy)
        uploaded = [SERVICE.save_upload(project_id, item.filename, item.file.read(),
                    requested_parser=requested_parser, requested_strategy=requested_strategy) for item in entries]
        if not uploaded: raise ValueError("请至少选择一个文件")
        strategy_key = SERVICE._assert_project_strategy(project_id)
        names = "、".join(str(document["original_filename"]) for document in uploaded)
        self._redirect(
            f"{_workspace_path(project_id, strategy_key)}?kind=ok&message="
            f"{quote('已上传：'+names)}",
            project_id,
        )

    def _download_document(self, document_id: int) -> None:
        document = SERVICE.db.get_document(document_id)
        if not document:
            self.send_error(404, "Document not found"); return
        root = Config.docs_root.resolve()
        path = Path(document["local_path"]).resolve()
        if (root not in path.parents and path != root) or not path.is_file():
            self.send_error(404, "File not found"); return
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(str(document['original_filename']))}")
        self.end_headers()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                self.wfile.write(block)


class FilesHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        root = Config.files_root.resolve(); candidate = (root / unquote(urlparse(self.path).path).lstrip("/")).resolve()
        if (root not in candidate.parents and candidate != root) or not candidate.is_file(): self.send_error(404); return
        body = candidate.read_bytes(); self.send_response(200); self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def serve() -> None:
    global SERVICE
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    for warning in validate_runtime_config(Config):
        LOGGER.warning("RUNTIME SAFETY: %s", warning)
    if SERVICE is None:
        SERVICE = KnowledgeForgeService()
    SERVICE.bootstrap()
    recovered = SERVICE.recover_interrupted_jobs()
    if recovered:
        LOGGER.info("Recovered %s interrupted background jobs", recovered)
    file_server = ThreadingHTTPServer((Config.upload_host, 8010), FilesHandler)
    threading.Thread(target=file_server.serve_forever, daemon=True, name="knowledge-files").start()
    server = ThreadingHTTPServer((Config.upload_host, Config.upload_port), AppHandler)
    LOGGER.info("Knowledge Forge UI: http://%s:%s", Config.service_host, Config.upload_port)
    try: server.serve_forever()
    finally: file_server.shutdown(); file_server.server_close(); server.server_close()


CSS = r"""
:root{font-family:Arial,"Microsoft YaHei",sans-serif;color:#13243b;background:#f3f7fb}*{box-sizing:border-box}body{margin:0}.shell{max-width:1180px;margin:30px auto;padding:0 20px 40px}header,.section-heading,.project-picker,.header-actions,.section-actions,.task-head,.dialog-heading{display:flex;align-items:center;gap:12px}header,.section-heading,.task-head,.dialog-heading{justify-content:space-between}header{margin-bottom:18px}.brand-link{display:block;color:inherit;text-decoration:none;border-radius:6px}.brand-link:hover h1{color:#1769c2}.brand-link:focus-visible{outline:3px solid #9cc6f4;outline-offset:5px}h1,h2{margin:0}h1{font-size:27px}h2{font-size:19px}p{color:#5d7088;margin:7px 0 0;line-height:1.5}.panel{background:#fff;border:1px solid #d5e0ed;border-radius:8px;padding:22px;margin-bottom:16px;box-shadow:0 10px 24px rgba(26,61,101,.05)}.project-panel{background:#fff}.project-heading{margin:20px 0 4px}.project-heading h2{font-size:23px}.workspace-section{border-top:1px solid #dce5ee;margin-top:22px;padding-top:22px}.button,.icon-button{border:0;background:#30445f;color:#fff;padding:9px 14px;border-radius:6px;font-size:14px;cursor:pointer;text-decoration:none;white-space:nowrap}.button:hover{background:#20324c}.button.primary{background:#2677d8}.button.secondary{background:#fff;color:#263f60;border:1px solid #c4d2e1}.button.small{padding:7px 10px}.button:disabled{cursor:not-allowed;opacity:1}.button.primary:disabled{background:#b8c7d8;color:#eef3f8}.button.secondary:disabled{background:#eef2f6;color:#98a5b5;border-color:#d8e0e9}.icon-button{width:34px;height:34px;padding:0;font-size:22px}.danger{background:#df3038}.header-actions{justify-content:flex-end}label{color:#354963;font-size:14px;display:flex;flex-direction:column;gap:7px}input,select,textarea{width:100%;border:1px solid #bfd0e2;border-radius:6px;padding:9px 10px;color:#172b47;background:#fff;font:inherit}select{min-width:240px}.project-picker{margin:0}.project-picker form{margin:0}.project-picker select{width:min(430px,100%)}.project-picker select,.project-picker .button{font-size:16px;min-height:42px}.project-meta{display:flex;align-items:center;flex-wrap:wrap;gap:20px;color:#58708c;font-size:13px;padding:8px 0}.project-meta b{color:#263f60}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}.stats div{border-left:3px solid #2f7bd4;background:#f8fbff;padding:10px 14px}.stats b{display:block;font-size:20px}.stats span{font-size:12px;color:#6d7f93}.stats-actions{display:flex;align-items:center;justify-content:flex-end;gap:8px;margin-top:12px}summary{color:#1769c2;font-weight:700;cursor:pointer}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px;margin-top:14px;max-width:850px}.wide,.form-actions{grid-column:1/-1}.compact{max-width:700px}.settings-toolbar{display:flex;justify-content:flex-end;margin-top:14px}.refresh-button{background:#fff;color:#263f60;border:1px solid #c4d2e1;font-size:21px;line-height:1}.refresh-button:hover{background:#edf4fb}.upload-form,.search-form{display:flex;gap:10px;align-items:center;margin-top:17px}.upload-form input,.search-form input{flex:1;min-width:0}.search-form select{width:160px;min-width:160px;flex:0 0 160px}.native-query-form select{width:190px;min-width:190px;flex-basis:190px}.native-query-mode-help{min-height:20px}.native-query-answer{margin-top:18px;padding:16px;border:1px solid #cbdcec;border-radius:8px;background:#f8fbff}.native-query-answer-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.native-query-answer-head h3{margin:0;color:#234568}.native-query-answer-head span{padding:4px 8px;border-radius:12px;background:#e3f0fd;color:#2365a7;font-size:12px}.native-query-answer>p{white-space:pre-wrap;color:#29445f}.native-query-evidence{margin-top:18px}.native-query-evidence>h3{margin:0}.native-context-list{display:grid;gap:9px;margin-top:10px}.native-context-card{padding:12px;border:1px solid #d7e2ed;border-radius:7px;background:#fff}.native-context-card>div{display:flex;align-items:center;gap:8px}.native-context-card>div span{padding:3px 6px;border-radius:4px;background:#eaf3fc;color:#24629d;font-size:11px}.native-context-card>div b{font-size:13px}.native-context-card p{display:-webkit-box;overflow:hidden;-webkit-line-clamp:4;-webkit-box-orient:vertical;color:#49627d;font-size:12px;white-space:pre-wrap}.native-context-card small{display:block;margin-top:7px;color:#6b7e93;font-size:11px}.native-context-card small a{color:#1769c2}.task-box{margin-top:18px;background:#101c2e;color:#d9e8fb;padding:14px;border-radius:6px}.task-box pre{margin:10px 0 0;max-height:260px;overflow:auto;white-space:pre-wrap;font:13px/1.5 monospace}.progress{height:5px;background:#263951;margin-top:10px}.progress i{display:block;height:100%;width:0;background:#4fa3ff;transition:width .25s}.schema-chip{display:inline-flex;align-items:center;color:#315a83;background:#eaf3fc;padding:6px 9px;border-radius:4px;font-size:13px;line-height:1}.notice{position:fixed;top:18px;right:18px;z-index:1100;max-width:min(430px,calc(100vw - 36px));padding:12px 14px;border-radius:6px;display:flex;gap:9px;align-items:center;box-shadow:0 12px 30px rgba(20,42,70,.2);transition:opacity .25s,transform .25s}.notice.toast-hide{opacity:0;transform:translateY(-8px);pointer-events:none}.notice b{display:grid;place-items:center;flex:0 0 20px;width:20px;height:20px;border-radius:50%;color:#fff}.notice.success{background:#effaf2;color:#176a35;border:1px solid #9fdbaf}.notice.success b{background:#23a453}.notice.error{background:#fff3f3;color:#9c262d;border:1px solid #f0b1b4}.notice.error b{background:#d4383f}dialog{width:min(720px,calc(100% - 28px));border:1px solid #cbd8e6;border-radius:8px;padding:22px;color:#13243b;box-shadow:0 22px 60px rgba(15,35,60,.24)}dialog::backdrop{background:rgba(20,36,58,.46)}dialog .form-grid{max-width:none}.dialog-actions{display:flex;justify-content:flex-end;gap:10px}.log-dialog{width:min(900px,calc(100% - 28px))}.history-log{min-height:220px;max-height:60vh;overflow:auto;padding:16px;border-radius:6px;background:#101c2e;color:#d9e8fb;white-space:pre-wrap;font:13px/1.55 monospace}.job-log-error{margin:14px 0;padding:10px 12px;border:1px solid #f0b1b4;border-radius:6px;background:#fff3f3;color:#9c262d}.table-wrap{overflow:auto;border:1px solid #d6e1ec;border-radius:6px;margin-top:16px}table{width:100%;border-collapse:collapse;min-width:980px}th,td{padding:12px;text-align:left;border-bottom:1px solid #e1e9f1;vertical-align:middle}th{background:#f6f9fc;color:#39516e;font-size:14px}td small{display:block;color:#6b7e93;margin-top:5px;max-width:390px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.actions{position:sticky;right:0;background:#fff;white-space:nowrap;text-align:center;min-width:390px}.actions .button,.actions .icon-button{display:inline-flex;align-items:center;justify-content:center;vertical-align:middle;height:34px}.actions>*+*{margin-left:6px}th.actions{background:#f6f9fc;text-align:center}.status{display:inline-block;padding:4px 8px;border-radius:14px;background:#e8edf3;color:#42566f;font-size:13px}.status-indexed{background:#e8f7ec;color:#20773b}.status-failed{background:#fff0f0;color:#b72d35}.status-processing{background:#fff7df;color:#926a00}.failure{color:#b72d35!important}.empty{text-align:center;color:#74869b;padding:30px}.result-group{margin-top:18px}.result-card{border-top:1px solid #dce5ee;padding:12px 2px}.result-card h3{font-size:15px;margin:0 0 7px}.result-card p{font-size:14px;color:#354963;white-space:pre-wrap}.score{font-size:12px;color:#1b6bc1}.relation{font-weight:700;color:#315a83}.muted{color:#718398;font-size:13px}
.task-head>span{display:flex;align-items:center;gap:12px}.task-duration{color:#9eb3cc;font-variant-numeric:tabular-nums}.task-log-button{border:1px solid #50647e;background:#1a2a3f;color:#d9e8fb}.task-log-button:hover{background:#263b55}.project-heading{margin:0 0 18px}.project-heading h2{font-size:24px}.project-heading .project-picker select,.project-heading .project-picker .button{font-size:17px}
.native-runtime-details{background:#fbfdff}.native-runtime-status{display:flex;justify-content:space-between;align-items:center;margin-top:14px;color:#315a83;font-size:14px}.native-runtime-status b{color:#203f63}.native-runtime-progress{margin-top:8px;background:#dce8f5}.native-runtime-progress i{background:#4ca36f}
.stats-actions{justify-content:flex-start;flex-wrap:nowrap;overflow-x:auto;padding-bottom:2px}.schema-actions,.graph-actions{display:flex;align-items:center;gap:8px;flex:0 0 auto}.stats-actions .schema-chip{padding:9px 12px;font-size:16px;font-weight:700}
.dialog-title-actions{display:flex;align-items:center;gap:8px}.model-sync-status{margin:14px 0 0;font-size:14px}.model-sync-status.success{color:#20773b}.model-sync-status.failure{color:#b72d35}.field-help{display:block;min-height:18px;color:#6b7e93;font-size:12px;line-height:1.5}.field-help.failure{color:#b72d35}.construction-controls{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex:1 1 auto;flex-wrap:nowrap;margin:0}.construction-controls form{margin:0}.construction-controls label{flex-direction:row;align-items:center;gap:8px;white-space:nowrap}.construction-controls select{min-width:160px;padding:8px 10px}
.delete-progress-dialog{width:min(520px,calc(100% - 28px))}.delete-progress-body{display:flex;align-items:center;gap:16px;margin:22px 0 16px;padding:16px;border:1px solid #d6e1ec;border-radius:6px;background:#f8fbff}.delete-progress-body p{margin-top:4px}.delete-spinner{flex:0 0 30px;width:30px;height:30px;border:3px solid #cfdae7;border-top-color:#2677d8;border-radius:50%;animation:delete-spin .8s linear infinite}.delete-result{display:grid;place-items:center;flex:0 0 30px;width:30px;height:30px;border-radius:50%;color:#fff;font-size:20px;font-weight:700}.delete-result.success{background:#23a453}.delete-result.failure{background:#d4383f}.delete-progress-bar{margin:0 0 18px}.delete-progress-bar i{background:#2677d8}.deleting-row{opacity:.48;pointer-events:none}@keyframes delete-spin{to{transform:rotate(360deg)}}
.processing-form{display:flex;flex-direction:column;align-items:stretch}.upload-line{display:flex;align-items:center;gap:10px}.upload-line input{flex:1}.upload-help{margin:7px 0 0;color:#6b7e93;font-size:13px}.processing-options{border:1px solid #d7e2ed;border-radius:6px;padding:10px 12px;background:#f8fbff}.processing-options summary{font-size:14px}.processing-options .form-grid{margin-top:12px;max-width:none}.processing-options .muted{margin-top:10px}
.task-steps{display:grid;gap:13px;margin-top:15px}.task-step{display:grid;grid-template-columns:minmax(260px,360px) minmax(180px,1fr);align-items:center;gap:16px}.task-step-head{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:13px}.task-step-head>div{min-width:0}.task-step-head span{display:block;color:#d9e8fb;font-weight:700}.task-step-head small{display:block;margin-top:4px;color:#8fa5bd;font-size:11px;line-height:1.4}.task-step-head b{color:#8fa5bd;font-weight:400;white-space:nowrap}.task-step .progress{margin:0;height:7px;border-radius:5px;overflow:hidden}.task-step.running .task-step-head b{color:#72b8ff}.task-step.running .task-step-head span{color:#fff}.task-step.completed .task-step-head b{color:#75d69a}.task-step.completed .progress i{background:#38ae68}.task-step.failed .task-step-head b,.task-step.cancelled .task-step-head b{color:#ff9297}.task-step.failed .progress i,.task-step.cancelled .progress i{background:#df5960}.stats b.is-updating::after{content:"";display:inline-block;width:12px;height:12px;margin-left:8px;border:2px solid #c6d5e5;border-top-color:#2677d8;border-radius:50%;animation:stat-spin .8s linear infinite;vertical-align:1px}.stats b.is-updating+span::after{content:" · 更新中";color:#2677d8}@keyframes stat-spin{to{transform:rotate(360deg)}}
.disabled-link{opacity:.45;pointer-events:none;cursor:not-allowed}.status-queued{background:#eef3ff;color:#315a83}.status-cancelled{background:#f1f3f5;color:#667789}.chunk-badges{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:9px}.chunk-badge{padding:3px 7px;border-radius:4px;background:#eef3f8;color:#496079;font-size:12px}.chunk-badge.graph{background:#e8f7ec;color:#20773b}.chunk-badge.skipped{background:#fff3df;color:#8b6400}
.structure-dialog{width:min(980px,calc(100% - 28px))}.structure-tools{display:grid;grid-template-columns:minmax(0,1fr) 210px;gap:10px;margin:16px 0}.structure-tools select{min-width:0}.structure-list{max-height:58vh;overflow:auto;border:1px solid #d6e1ec;border-radius:6px}.structure-row{padding:14px 16px;border-bottom:1px solid #e1e9f1}.structure-row:last-child{border-bottom:0}.structure-row-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:9px}.structure-row-meta{display:flex;align-items:center;flex-wrap:wrap;gap:7px;color:#637891;font-size:12px}.block-type{display:inline-block;padding:3px 7px;border-radius:4px;background:#eaf3fc;color:#245d94;font-weight:700}.structure-text{margin:0;color:#243a55;white-space:pre-wrap;word-break:break-word;font:14px/1.65 Arial,"Microsoft YaHei",sans-serif}.structure-metadata{margin-top:10px}.structure-metadata summary{font-size:12px;font-weight:400}.structure-metadata pre{margin:8px 0 0;padding:10px;background:#f6f9fc;border-radius:4px;overflow:auto;font-size:12px}.structure-pager{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:14px 0}.structure-pager>div{display:flex;gap:8px}
.home-shell{max-width:1240px}.home-header{padding:2px 3px 0}.home-header p{margin-top:4px}.home-header>.button{min-width:102px}.home-hero{position:relative;overflow:hidden;padding:30px;background:linear-gradient(135deg,#fff 0%,#f4f9ff 58%,#e9f5ff 100%);border-color:#bfd5ec}.home-hero:after{content:"";position:absolute;right:-95px;top:-120px;width:300px;height:300px;border:46px solid rgba(38,119,216,.055);border-radius:50%;pointer-events:none}.hero-layout{position:relative;z-index:1;display:grid;grid-template-columns:minmax(0,1.45fr) minmax(310px,.8fr);gap:34px;align-items:start}.hero-kicker,.section-kicker{display:inline-block;color:#1769c2;font-size:12px;font-weight:800;letter-spacing:.08em}.hero-copy h2{max-width:760px;margin-top:9px;color:#173a62;font-size:30px;line-height:1.25}.hero-copy p{max-width:780px;color:#465e79;font-size:15px}.hero-points{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.hero-points div{min-height:104px;padding:15px;border:1px solid rgba(159,190,222,.75);border-radius:8px;background:rgba(255,255,255,.82);box-shadow:0 8px 20px rgba(31,79,126,.06)}.hero-points b,.hero-points span{display:block}.hero-points b{margin-bottom:7px;color:#245d94;font-size:14px}.hero-points span{color:#60758d;font-size:12px;line-height:1.55}.home-flow{position:relative;z-index:1;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:26px 0 0;padding:17px 0 0;border-top:1px solid #d5e4f2;list-style:none}.home-flow li{display:flex;align-items:center;gap:10px;padding:0 10px}.home-flow li+li{border-left:1px solid #d5e4f2}.home-flow li>b{display:grid;place-items:center;flex:0 0 30px;width:30px;height:30px;border-radius:50%;background:#2677d8;color:#fff;font-size:13px}.home-flow span,.home-flow strong,.home-flow small{display:block}.home-flow strong{color:#263f60;font-size:14px}.home-flow small{margin-top:3px;color:#718398}.strategy-overview{padding:24px;background:#fff}.strategy-heading{align-items:flex-end;margin-bottom:17px}.strategy-heading h2{margin-top:7px;font-size:24px}.strategy-heading p{max-width:760px}.strategy-intro-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px}.strategy-explainer{display:flex;flex-direction:column;padding:23px;border:1px solid #ccdaea;border-top:4px solid #2677d8;border-radius:9px;background:#fff;box-shadow:0 8px 20px rgba(24,58,98,.045)}.strategy-native{border-top-color:#159b8a}.strategy-card-head{display:flex;align-items:center;gap:12px}.strategy-card-head h3{margin:3px 0 0;color:#183b63;font-size:22px}.strategy-mark{display:grid;place-items:center;flex:0 0 44px;width:44px;height:44px;border-radius:9px;background:#e7f1fd;color:#1769c2;font-size:14px;font-weight:900}.strategy-native .strategy-mark{background:#e3f5f1;color:#08796c}.strategy-tag{color:#6a7e94;font-size:12px;font-weight:700}.strategy-lead{min-height:68px;margin-top:16px;color:#344e6b}.strategy-path{margin-top:13px;padding:11px 12px;border-radius:6px;background:#eef5fc;color:#245d94;font-size:12px;font-weight:700;line-height:1.55}.strategy-native .strategy-path{background:#eaf7f5;color:#08796c}.strategy-details{display:grid;gap:0;margin:17px 0 0}.strategy-details div{display:grid;grid-template-columns:78px minmax(0,1fr);gap:10px;padding:10px 0;border-top:1px solid #e2e9f1}.strategy-details dt{color:#263f60;font-size:13px;font-weight:800}.strategy-details dd{margin:0;color:#5d7088;font-size:13px;line-height:1.55}.strategy-fit{margin-top:auto;padding:14px 15px;border-radius:7px;background:#f7faff}.strategy-fit>b{color:#274c75;font-size:13px}.strategy-fit ul{margin:8px 0 0;padding-left:18px;color:#4c647f;font-size:13px;line-height:1.7}.strategy-tradeoff{margin-top:13px;padding-left:11px;border-left:3px solid #e0a02b;color:#65768a;font-size:12px}.strategy-tradeoff b{color:#8d6417}.choice-guide{display:grid;grid-template-columns:auto repeat(2,minmax(0,1fr));gap:9px 18px;align-items:start;margin-top:17px;padding:16px 18px;border:1px solid #cbdbea;border-radius:8px;background:#f8fbff;color:#49627e;font-size:13px;line-height:1.55}.choice-guide>b{color:#183b63}.choice-guide strong{color:#1769c2}.choice-guide em{grid-column:2/-1;color:#8a681f;font-style:normal}.project-list-dialog{width:min(1100px,calc(100% - 28px));height:min(82vh,880px);max-height:none;overflow:hidden}.project-list-dialog[open]{display:flex;flex-direction:column}.project-list-dialog .project-grid{flex:1;min-height:0;overflow:auto;padding:2px 4px 4px}.project-list-heading{flex:0 0 auto;align-items:flex-end}.project-list-heading h2{margin-top:6px;font-size:23px}.project-list-heading .button{min-width:96px}.strategy-choice-help{margin:0;padding:10px 12px;border:1px solid #d6e3f0;border-radius:6px;background:#f6faff;color:#49627e;font-size:13px}.project-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:18px}.project-card{display:flex;flex-direction:column;justify-content:space-between;gap:18px;padding:18px;border:1px solid #d6e1ec;border-radius:8px;background:#fbfdff}.project-card h3{margin:0;color:#203b5b}.strategy-badge,.fixed-strategy{display:inline-flex;margin-top:10px;padding:6px 9px;border-radius:5px;background:#eaf3fc;color:#245d94;font-size:13px;font-weight:700}.bind-form{display:flex;align-items:center;gap:8px}.bind-form select{min-width:0}.native-config-note{display:flex;flex-direction:column;gap:6px;padding:10px 12px;border:1px solid #d6e1ec;border-radius:6px;background:#fbfdff;color:#5d7088;font-size:13px}.native-config-note b{color:#284a70}.artifact-content{min-height:260px;max-height:58vh;overflow:auto;padding:14px;border-radius:6px;background:#101c2e;color:#d9e8fb;white-space:pre-wrap;word-break:break-word;font:12px/1.55 monospace}
.workspace-shell{max-width:1440px}.workspace-brand-header{margin-bottom:16px}.workspace-layout{display:grid;grid-template-columns:280px minmax(0,1fr);align-items:stretch;border:1px solid #d2deeb;border-radius:10px;background:#fff;box-shadow:0 10px 24px rgba(26,61,101,.06);overflow:hidden}.workspace-sidebar{min-width:0;border-right:1px solid #dce5ee;background:#f9fbfd}.workspace-project-summary{padding:21px 19px 18px;border-bottom:1px solid #dce5ee;background:linear-gradient(145deg,#f9fcff,#edf5fd)}.workspace-project-title{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:start;gap:9px}.workspace-project-summary h2{margin:0;color:#183b63;font-size:20px;line-height:1.3;overflow-wrap:anywhere}.workspace-strategy-badge{display:inline-flex;padding:5px 8px;border-radius:5px;background:#dcecff;color:#1769c2;font-size:12px;font-weight:800}.workspace-nav{display:grid;gap:5px;padding:10px}.workspace-nav-item{display:grid;grid-template-columns:34px minmax(0,1fr);grid-template-rows:auto auto;column-gap:10px;width:100%;padding:12px 11px;border:1px solid transparent;border-radius:7px;background:transparent;color:#415a75;text-align:left;cursor:pointer}.workspace-nav-item>span{grid-row:1/3;display:grid;place-items:center;align-self:center;width:30px;height:30px;border-radius:7px;background:#edf3f9;color:#657b93;font-size:11px;font-weight:800}.workspace-nav-item>b{align-self:end;color:#284765;font-size:14px}.workspace-nav-item>small{align-self:start;margin-top:3px;color:#7a8da1;font-size:11px}.workspace-nav-item:hover{background:#f5f9fd;border-color:#dbe6f1}.workspace-nav-item[aria-selected=true]{border-color:#bad4f1;background:#eaf4ff}.workspace-nav-item[aria-selected=true]>span{background:#2677d8;color:#fff}.workspace-nav-item[aria-selected=true]>b{color:#1769c2}.workspace-nav-item:focus-visible{outline:3px solid #9cc6f4;outline-offset:1px}.workspace-content{min-width:0;background:#fff}.workspace-view{min-height:560px;margin:0;padding:24px}.workspace-view[hidden]{display:none}.workspace-view-kicker{display:inline-block;margin-bottom:7px;color:#2677d8;font-size:12px;font-weight:800;letter-spacing:.08em}.workspace-view-heading{align-items:flex-end}.preparation-view-heading{align-items:flex-start}.workspace-view-heading h2{font-size:23px}.workspace-view>.workspace-section:first-child{margin-top:0;padding-top:0;border-top:0}.workspace-settings .form-grid{max-width:900px;margin-top:22px}.workspace-settings .form-actions{display:flex;justify-content:flex-end}.workspace-settings .form-actions .button{min-width:140px}.workspace-settings .native-config-note{min-height:82px;justify-content:center}.settings-processing{margin-top:30px}.settings-processing-options{margin-top:16px;padding:16px}.settings-processing-options .form-grid{margin-top:0}.workspace-content .graph-summary{margin-top:18px}.workspace-content .native-runtime-details{margin-top:22px}.workspace-content .task-box{margin-top:20px}.workspace-content .stats-actions{align-items:center;flex-wrap:wrap;overflow:visible}.workspace-content .construction-controls{justify-content:flex-start;flex:1 1 560px;flex-wrap:wrap}.preparation-heading-actions{flex:0 0 auto}.workspace-content .preparation-heading-actions .construction-controls{justify-content:flex-end;flex:0 0 auto;flex-wrap:nowrap}.workspace-content .schema-actions,.workspace-content .graph-actions{flex-wrap:wrap}.knowledge-structure-title{display:flex;align-items:center;gap:9px}.schema-version-badge{display:inline-flex;padding:3px 7px;border-radius:12px;background:#eaf3fc;color:#315a83;font-size:11px;font-weight:800}.definition-group{margin-top:20px}.definition-group-heading{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}.definition-group-heading h3{margin:0;color:#234568;font-size:16px}.definition-group-heading span{color:#6b7e93;font-size:12px}.definition-grid{display:grid;gap:10px}.entity-definition-grid{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}.relation-definition-grid{grid-template-columns:repeat(auto-fit,minmax(290px,1fr))}.definition-card{padding:13px 14px;border:1px solid #d7e2ed;border-radius:7px;background:#fbfdff}.definition-card-head{display:flex;align-items:center;justify-content:space-between;gap:10px}.definition-card-head strong{color:#1f456c;overflow-wrap:anywhere}.definition-card-head span{flex:0 0 auto;padding:3px 6px;border-radius:4px;background:#eaf3fc;color:#315a83;font-size:11px}.definition-card p{margin-top:7px;font-size:12px}.definition-card dl{margin:10px 0 0}.definition-card dl>div{display:grid;grid-template-columns:64px minmax(0,1fr);gap:8px;padding-top:7px;border-top:1px solid #e3eaf1;font-size:11px;line-height:1.45}.definition-card dt{color:#536a83;font-weight:700}.definition-card dd{margin:0;color:#314a65}.empty-state{padding:22px;border:1px dashed #cbd8e6;border-radius:7px;color:#708196;text-align:center}.compact-empty{padding:15px}.artifact-table{min-width:760px}.artifact-table .artifact-count{display:block;color:#1f5f9f;font-size:19px}.artifact-table td>small{margin-top:3px}.artifact-table .artifact-actions{white-space:nowrap}.artifact-table .artifact-actions>*+*{margin-left:6px}
.embedded-graph-browser{padding-bottom:2px}.embedded-graph-tabs{display:inline-flex;gap:2px;padding:3px;border:1px solid #c7d5e5;border-radius:8px;background:#eaf0f7}.embedded-graph-tab{border:0;border-radius:5px;background:transparent;color:#5b6f86;padding:8px 14px;font:inherit;font-size:14px;cursor:pointer}.embedded-graph-tab:hover{color:#174f8d;background:#f5f9fd}.embedded-graph-tab.is-active{background:#fff;color:#1769c2;font-weight:700;box-shadow:0 1px 4px rgba(31,63,99,.18)}.embedded-graph-tab:focus-visible{outline:2px solid #2677d8;outline-offset:2px}.embedded-graph-loading{display:flex;align-items:center;justify-content:center;gap:14px;min-height:130px;margin-top:16px;border:1px dashed #cbd8e6;border-radius:8px;background:#f8fbff}.embedded-graph-loading p{margin-top:4px}.embedded-graph-spinner{width:28px;height:28px;border:3px solid #cfdae7;border-top-color:#2677d8;border-radius:50%;animation:stat-spin .8s linear infinite}.embedded-graph-empty{margin-top:16px}.embedded-graph-controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:16px 0}.embedded-graph-controls>input{min-width:260px;flex:1}.embedded-graph-controls>select{min-width:160px;width:auto}.embedded-graph-controls>label{flex-direction:row;align-items:center;gap:6px;white-space:nowrap}.embedded-graph-controls>label input{width:auto}.embedded-graph-zoom{display:inline-flex;align-items:center;gap:5px;padding:3px;border:1px solid #d0dce8;border-radius:7px;background:#f7faff}.embedded-graph-zoom .button{min-width:34px;padding:6px 9px}.embedded-graph-zoom span{min-width:46px;color:#49627d;font-size:12px;text-align:center}.embedded-graph-layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:13px}.embedded-graph-canvas,.embedded-graph-sidebar{border:1px solid #d5e0ed;border-radius:8px;background:#fbfdff}.embedded-graph-canvas{min-width:0;overflow:auto}.embedded-graph-canvas svg{display:block;width:100%;min-width:760px;min-height:620px;background:linear-gradient(145deg,#fbfdff,#eef5fb)}.embedded-graph-canvas marker path{fill:#8298ae}.embedded-graph-edge{stroke:#8398ad;stroke-opacity:.72;cursor:pointer}.embedded-graph-edge.is-dim,.embedded-graph-node.is-dim{opacity:.1}.embedded-graph-edge.is-selected{stroke:#ec7e20;stroke-opacity:1}.embedded-graph-node{cursor:pointer;transition:opacity .15s}.embedded-graph-node circle{stroke:#fff;stroke-width:2}.embedded-graph-node.is-selected circle{stroke:#ec7e20;stroke-width:4}.embedded-graph-node text{font-size:12px;font-weight:600;fill:#29445f;text-anchor:middle;paint-order:stroke;stroke:#fbfdff;stroke-width:3px;stroke-linejoin:round;pointer-events:none}.embedded-graph-group-label{font-size:14px;font-weight:800;paint-order:stroke;stroke:#fbfdff;stroke-width:4px}.embedded-graph-count{margin:0;padding:8px 12px;border-top:1px solid #d5e0ed;font-size:12px}.embedded-graph-sidebar{padding:15px;max-height:665px;overflow:auto}.embedded-graph-sidebar h3{margin:0 0 9px;color:#234568;font-size:16px}.embedded-graph-sidebar h4{margin:16px 0 8px}.embedded-graph-sidebar p{font-size:13px}.embedded-graph-sidebar dl{margin:0}.embedded-graph-sidebar dl>div{padding:7px 0;border-top:1px solid #e1e9f1}.embedded-graph-sidebar dt{color:#6a7f96;font-size:11px}.embedded-graph-sidebar dd{margin:3px 0 0;color:#29445f;font-size:12px;white-space:pre-wrap;word-break:break-word}.graph-entity-heading{display:flex;align-items:center;gap:9px;margin-bottom:10px}.graph-entity-heading h3{margin:0}.graph-entity-dot{display:inline-block;flex:0 0 auto;width:17px;height:17px;border:2px solid #fff;border-radius:50%;box-shadow:0 0 0 1px #aebfd0}.graph-relation-card{display:grid;grid-template-columns:minmax(0,1fr) 74px minmax(0,1fr);align-items:center;gap:7px;width:100%;margin:8px 0;padding:9px;border:1px solid #d7e2ed;border-radius:7px;background:#fff;color:#29445f;text-align:left}.graph-relation-card.button-card{cursor:pointer}.graph-relation-card.button-card:hover{border-color:#82b4e9;background:#f4f9ff}.graph-mini-node{display:flex;align-items:center;gap:6px;min-width:0;font-size:11px;font-weight:700}.graph-mini-node span:last-child{overflow:hidden;text-overflow:ellipsis}.graph-mini-relation{position:relative;color:#315d89;font-size:10px;text-align:center;overflow:hidden;text-overflow:ellipsis}.graph-mini-relation::before{content:"";position:absolute;left:0;right:0;top:50%;border-top:2px solid #91a6ba;z-index:0}.graph-mini-relation b{position:relative;z-index:1;padding:1px 4px;background:#fff;font-weight:600}.graph-mini-relation::after{content:"›";position:absolute;right:-1px;top:calc(50% - 11px);color:#718aa3;font-size:18px}.embedded-graph-community-status{margin-top:18px;padding-top:15px;border-top:1px solid #dce5ee}.embedded-graph-community-status p{margin-top:5px;font-size:12px}.embedded-graph-groups{display:grid;gap:7px;margin-top:10px}.embedded-graph-group-card{display:grid;grid-template-columns:10px minmax(0,1fr);gap:2px 8px;width:100%;padding:9px;border:1px solid #d7e2ed;border-radius:6px;background:#fff;color:#29445f;text-align:left;cursor:pointer}.embedded-graph-group-card.is-active{border-color:#2677d8;background:#eef6ff}.embedded-graph-group-card i{grid-row:1/3;width:9px;height:9px;margin-top:4px;border-radius:50%}.embedded-graph-group-card b{font-size:12px}.embedded-graph-group-card small{color:#70849a}.embedded-graph-group-card span{grid-column:2;margin-top:3px;color:#60758d;font-size:11px;line-height:1.4}.embedded-graph-audit table{min-width:850px}.embedded-graph-audit td{font-size:12px;vertical-align:top}.embedded-graph-audit td:last-child{max-width:360px;white-space:pre-wrap;word-break:break-word}
@media(max-width:980px){.workspace-layout{grid-template-columns:1fr}.workspace-sidebar{border-right:0;border-bottom:1px solid #dce5ee}.workspace-project-summary{padding:15px 16px}.workspace-nav{grid-template-columns:repeat(4,minmax(160px,1fr));overflow-x:auto}.workspace-nav-item{min-width:160px}.workspace-view{min-height:0}.embedded-graph-layout{grid-template-columns:1fr}.embedded-graph-sidebar{max-height:none}}
@media(max-width:760px){.shell{margin:16px auto;padding:0 10px 24px}header,.section-heading,.project-picker,.upload-form,.search-form{align-items:stretch;flex-direction:column}.project-picker label{min-width:0}.form-grid,.structure-tools{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.construction-controls,.stats-actions,.schema-actions,.graph-actions{align-items:stretch;flex-direction:column}.construction-controls{flex-wrap:wrap}.construction-controls form,.construction-controls .button,.stats-actions .button{width:100%}.construction-controls label{align-items:stretch;flex-direction:column}.construction-controls select{min-width:0}.search-form select{width:100%;min-width:0;flex:auto}.stats-actions .schema-chip{justify-content:center}.panel{padding:16px}.task-step{grid-template-columns:1fr;gap:5px}.home-hero{padding:20px}.hero-layout,.strategy-intro-grid{grid-template-columns:1fr}.hero-copy h2{font-size:25px}.hero-points{margin-top:2px}.home-flow{grid-template-columns:repeat(2,minmax(0,1fr));gap:15px 0}.home-flow li:nth-child(3){border-left:0}.strategy-overview{padding-top:14px}.strategy-heading h2{font-size:22px}.strategy-lead{min-height:0}.choice-guide{grid-template-columns:1fr}.choice-guide em{grid-column:auto}.project-list-heading .button{width:100%}.workspace-nav{padding:8px}.workspace-view-heading h2{font-size:21px}.workspace-settings .form-actions .button{width:100%}}
@media(max-width:440px){.hero-points,.home-flow{grid-template-columns:1fr}.home-flow li+li{border-left:0}.home-flow li{padding:5px 2px}.strategy-details div{grid-template-columns:1fr;gap:4px}.project-grid{grid-template-columns:1fr}}
"""

SCRIPT = r"""
const box=document.querySelector('#task-box'),taskStatus=document.querySelector('#task-status'),taskDuration=document.querySelector('#task-duration'),taskSteps=document.querySelector('#task-steps');
const esc=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const ingestionLogButton=document.querySelector('#view-ingestion-log'),preparationLogButton=document.querySelector('#view-preparation-log'),taskTitle=document.querySelector('#task-title');
const stageLabels={structure:'结构解析中',structure_ready:'结构解析完成',chunking:'知识分块中',chunk_ready:'知识分块完成',graph:'知识图谱构建中',completed:'全部完成',failed:'处理失败'};
const subtaskLabels={structure:'结构解析',chunking:'分块及向量索引',schema_base:'知识结构定义和基础图数据',extraction:'逐块抽取实体与关系',graph_sync:'HugeGraph 同步'};
// Native is a separate GraphRAG Standard pipeline.  It must never inherit
// Stable's HugeGraph/pgvector/entity-relation stage names in the UI.
const nativeSubtaskLabels=window.KF.taskProfile==='graphrag_native'?window.KF.taskLabels:{structure:'准备原生构建输入',chunking:'检查 GraphRAG 配置',schema_base:'检查模型服务',extraction:'生成 TextUnits、语义图与社区',graph_sync:'验证并发布原生产物'};
const nativePreparationLabels={structure:'解析并冻结源文档结构',chunking:'',schema_base:'验证 Native Documents',extraction:'',graph_sync:''};
const taskStatusLabels={queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',cancelled:'已取消'};
const clampProgress=value=>Math.max(0,Math.min(100,Math.round(Number(value)||0)));
function labelsForJob(job){if(job.kind==='graphrag_native_build')return nativeSubtaskLabels;if(job.kind==='content_preparation'&&window.KF.taskProfile==='graphrag_native')return nativePreparationLabels;return subtaskLabels}
function legacySubtasks(job){const p=clampProgress(job.progress),step=(status,progress,detail='')=>({status,progress:clampProgress(progress),detail});const result={structure:step('waiting',0),chunking:step('waiting',0),schema_base:step('waiting',0),extraction:step('waiting',0),graph_sync:step('waiting',0)};if(job.kind==='graph_sync'){result.graph_sync=step(job.status==='completed'?'completed':job.status==='failed'?'failed':'running',p);return result}if(job.kind==='schema_reassessment'){result.schema_base=step(job.status==='completed'?'completed':job.status==='failed'?'failed':'running',p,'重新评估');return result}result.structure=p>=18?step('completed',100):step('running',p/18*100);if(p>=20)result.chunking=p>=42?step('completed',100):step('running',(p-20)/22*100);if(p>=44)result.schema_base=p>=50?step('completed',100):step('running',(p-44)/6*100);if(p>=52)result.extraction=p>=90?step('completed',100):step('running',(p-52)/38*100);if(p>=92)result.graph_sync=p>=96?step('completed',100):step('running',(p-92)/4*100);if(job.status==='completed')for(const key of Object.keys(result))result[key]=step('completed',100,result[key].detail);return result}
function nativeSubtasks(job){const p=clampProgress(job.progress),step=(status,progress,detail='')=>({status,progress:clampProgress(progress),detail}),result={structure:step('waiting',0),chunking:step('waiting',0),schema_base:step('waiting',0),extraction:step('waiting',0),graph_sync:step('waiting',0)};if(p<10)result.structure=step('running',p*10);else result.structure=step('completed',100,'已冻结 active revisions');if(p>=10&&p<18)result.chunking=step('running',(p-10)*12.5);else if(p>=18)result.chunking=step('completed',100);if(p>=18&&p<25)result.schema_base=step('running',(p-18)*14.3);else if(p>=25)result.schema_base=step('completed',100);if(p>=30&&p<75)result.extraction=step('running',Math.max(5,(p-30)*2.2),'Standard workflow');else if(p>=75)result.extraction=step('completed',100);if(p>=75&&p<100)result.graph_sync=step('running',(p-75)*4,'验证 bundle');if(job.status==='completed')for(const key of Object.keys(result))result[key]=step('completed',100,result[key].detail);if(job.status==='failed'||job.status==='cancelled'){const active=['graph_sync','extraction','schema_base','chunking','structure'].find(key=>result[key].status==='running')||'structure';result[active]=step(job.status,p,result[active].detail)}return result}
function jobSubtasks(job){if(job.kind==='graphrag_native_build')return nativeSubtasks(job);const stored=job.metadata?.subtasks;if(!stored||typeof stored!=='object')return legacySubtasks(job);return {...legacySubtasks({kind:job.kind,status:'queued',progress:0}),...stored}}
let activeMonitorGeneration=0,renderedMonitorKey='',renderedSubtaskProgress={};
function renderSubtasks(job,monitorKey=''){if(monitorKey&&renderedMonitorKey!==monitorKey){renderedMonitorKey=monitorKey;renderedSubtaskProgress={}}const subtasks=jobSubtasks(job),labels=labelsForJob(job);if(taskStatus)taskStatus.textContent=taskStatusLabels[job.status]||job.status||'等待';for(const row of taskSteps?.querySelectorAll('[data-subtask]')||[]){const key=row.dataset.subtask,title=row.querySelector('.task-step-head span');if(title)title.textContent=labels[key]||key;const preparationHidden=job.kind==='content_preparation'&&(['extraction','graph_sync'].includes(key)||(window.KF.taskProfile==='graphrag_native'&&key==='chunking')),knowledgeBuildHidden=job.kind==='ingestion'&&window.KF.strategyKey==='enhanced_shadow'&&['structure','chunking','schema_base'].includes(key);row.hidden=job.kind==='document_cleanup'||(job.kind==='schema_reassessment'&&key!=='schema_base')||preparationHidden||knowledgeBuildHidden;const data=subtasks[key]||{status:'waiting',progress:0,detail:''},status=data.status||'waiting',reportedProgress=clampProgress(data.progress),progress=monitorKey&&['running','completed'].includes(status)?Math.max(renderedSubtaskProgress[key]||0,reportedProgress):reportedProgress,label=row.querySelector('[data-subtask-status]'),bar=row.querySelector('[data-subtask-bar]');if(monitorKey&&['running','completed'].includes(status))renderedSubtaskProgress[key]=progress;row.className='task-step '+status;if(label){if(status==='completed')label.textContent='已完成'+(data.detail?' · '+data.detail:'');else if(status==='failed')label.textContent='失败'+(data.detail?' · '+data.detail:'');else if(status==='cancelled')label.textContent='已取消';else if(status==='running')label.textContent=progress+'%'+(data.detail?' · '+data.detail:'');else label.textContent='等待'}if(bar)bar.style.width=progress+'%'}}
function formatDuration(seconds){const value=Math.max(0,Math.floor(Number(seconds)||0)),hours=Math.floor(value/3600),minutes=Math.floor(value%3600/60),secs=value%60;return(hours?String(hours).padStart(2,'0')+':':'')+String(minutes).padStart(2,'0')+':'+String(secs).padStart(2,'0')}
let lastStatisticsRefresh=0;
async function refreshStatistics(force=false){if(!window.KF.projectId||(!force&&Date.now()-lastStatisticsRefresh<2000))return;lastStatisticsRefresh=Date.now();try{const response=await fetch('/api/projects/'+window.KF.projectId+'/statistics'),data=await response.json();if(!response.ok)return;for(const key of ['documents','chunks','entities','relations']){const target=document.querySelector('[data-stat="'+key+'"]');if(target)target.textContent=Number(data[key]||0).toLocaleString()}}catch(_error){}}
function updatingStatisticKeys(kind){if(kind==='content_preparation')return window.KF.taskProfile==='graphrag_native'?[]:['chunks'];if(kind==='ingestion')return['entities','relations'];if(['graphrag_native_build','document_cleanup'].includes(kind))return['chunks','entities','relations'];if(kind==='graph_sync')return['entities','relations'];return[]}
function setStatisticsUpdating(updating,kind=''){const active=new Set(updating?updatingStatisticKeys(kind):[]);for(const key of ['chunks','entities','relations']){const target=document.querySelector('[data-stat="'+key+'"]');if(!target)continue;const changing=active.has(key);target.classList.toggle('is-updating',changing);target.setAttribute('aria-label',changing?target.textContent+'，仍在更新':target.textContent)}}
function applyJobReadiness(job){const metadata=job.metadata||{},documentId=job.document_id||metadata.document_id;if(documentId){const structure=document.querySelector('[data-view-structure="'+documentId+'"]'),chunks=document.querySelector('[data-view-chunks="'+documentId+'"]'),label=document.querySelector('[data-stage-label="'+documentId+'"]'),status=document.querySelector('[data-document-status="'+documentId+'"]');if(structure&&metadata.structure_ready)structure.disabled=false;if(chunks&&metadata.chunks_ready)chunks.disabled=false;if(label&&metadata.stage)label.textContent=stageLabels[metadata.stage]||metadata.stage;if(status){let state=job.status==='completed'?'indexed':job.status==='failed'?'failed':job.status==='running'?'processing':'uploaded',text={uploaded:'已上传',processing:'处理中',indexed:'已入库',failed:'失败'}[state];if(job.kind==='content_preparation'){const native=window.KF.taskProfile==='graphrag_native';state=job.status==='failed'?'failed':job.status==='running'?'processing':'uploaded';text=job.status==='completed'?(native?'源文档已准备':'预处理完成'):job.status==='failed'?(native?'源文档准备失败':'预处理失败'):job.status==='running'?(native?'准备源文档中':'预处理中'):(native?'等待源文档准备':'等待预处理')}status.className='status status-'+state;status.textContent=text||job.status}}}
function monitor(id,kind='',batch=''){if(!id)return;const generation=++activeMonitorGeneration,monitorKey=(batch?'batch:':'job:')+(batch||id),preparation=kind==='content_preparation',slot=document.querySelector('[data-task-slot="'+(preparation?'preparation':'graph')+'"]');if(slot&&box.parentElement!==slot)slot.append(box);box.hidden=false;box.dataset.job=id;box.dataset.batch=batch;box.dataset.kind=kind;setStatisticsUpdating(true,kind);const target=preparation?preparationLogButton:['ingestion','graphrag_native_build'].includes(kind)?ingestionLogButton:null,activeLogButton=document.querySelector('#active-job-log');for(const button of [target,activeLogButton])if(button){button.dataset.job=id;button.dataset.batch=batch;button.disabled=false}if(taskTitle)taskTitle.textContent=kind==='graphrag_native_build'?'GraphRAG Native 构建':kind==='graph_sync'?'知识图谱同步':kind==='schema_reassessment'?'知识结构重新评估':kind==='document_cleanup'?'文档清理':kind==='content_preparation'?(window.KF.taskProfile==='graphrag_native'?'准备 GraphRAG Native 源文档':'文档预处理'):'知识库构建';const tick=async()=>{if(generation!==activeMonitorGeneration)return;try{const endpoint=batch?'/api/job-batches/'+batch:'/api/jobs/'+id,res=await fetch(endpoint),job=await res.json();if(generation!==activeMonitorGeneration)return;if(!res.ok)throw new Error(job.error||'任务状态读取失败');renderSubtasks(job,monitorKey);updateNativeRuntime(job);setStatisticsUpdating(!['completed','failed','cancelled'].includes(job.status),job.kind||kind);if(taskDuration)taskDuration.textContent='用时 '+formatDuration(job.elapsed_seconds);for(const item of job.jobs||[job])applyJobReadiness(item);await refreshStatistics();if(generation!==activeMonitorGeneration)return;if(['completed','failed','cancelled'].includes(job.status)){setStatisticsUpdating(false,job.kind||kind);setTimeout(()=>{if(generation===activeMonitorGeneration)location.reload()},900);return}}catch(error){if(generation!==activeMonitorGeneration)return;if(taskStatus)taskStatus.textContent='状态读取失败'}if(generation===activeMonitorGeneration)setTimeout(tick,800)};tick()}
if(box?.dataset.job)monitor(box.dataset.job,box.dataset.kind||'',box.dataset.batch||'');
document.querySelector('#build-project-knowledge')?.addEventListener('click',async event=>{const button=event.currentTarget,native=window.KF.strategyKey==='graphrag_native',operation=native?'原生构建':'知识构建';if(fileInput?.files?.length){alert('请先上传当前已选择的文件，再执行'+operation+'。');return}button.disabled=true;const original=button.textContent;button.textContent='提交中...';try{const endpoint=native?'/api/projects/'+button.dataset.buildProject+'/graphrag-native/build':'/api/projects/'+button.dataset.buildProject+'/knowledge-forge/build',response=await fetch(endpoint,{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||operation+'提交失败');if(!native)for(const status of document.querySelectorAll('[data-document-status]')){if(!status.classList.contains('status-indexed')){status.className='status status-queued';status.textContent='排队中'}}monitor(data.job_id,native?'graphrag_native_build':'ingestion',data.batch_id||'')}catch(error){alert(error.message||operation+'提交失败');button.disabled=false;button.textContent=original}});
document.querySelector('#prepare-project-documents')?.addEventListener('click',async event=>{const button=event.currentTarget;if(fileInput?.files?.length){alert('请先上传当前已选择的文件，再执行预处理。');return}button.disabled=true;const original=button.textContent;button.textContent='提交中...';try{const slug=window.KF.strategyKey==='graphrag_native'?'graphrag-native':'knowledge-forge',response=await fetch('/api/projects/'+button.dataset.prepareProject+'/'+slug+'/prepare',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'文档预处理提交失败');monitor(data.job_id,'content_preparation',data.batch_id||'')}catch(error){alert(error.message||'文档预处理提交失败');button.disabled=false;button.textContent=original}});
const deleteDialog=document.querySelector('#delete-progress-dialog'),deleteSpinner=document.querySelector('#delete-progress-spinner'),deleteResult=document.querySelector('#delete-progress-result'),deleteTitle=document.querySelector('#delete-progress-title'),deleteMessage=document.querySelector('#delete-progress-message'),deleteDetail=document.querySelector('#delete-progress-detail'),deleteBar=document.querySelector('#delete-progress-bar'),deleteClose=document.querySelector('#delete-progress-close');let deletePollTimer=null;
function updateDeleteProgress(progress,title,detail){deleteBar.style.width=clampProgress(progress)+'%';deleteTitle.textContent=title;deleteDetail.textContent=detail||''}
function finishDeleteProgress(success,title,detail){deleteSpinner.hidden=true;deleteResult.hidden=false;deleteResult.className='delete-result '+(success?'success':'failure');deleteResult.textContent=success?'✓':'!';updateDeleteProgress(100,title,detail);deleteClose.disabled=false;if(success)setTimeout(()=>location.reload(),1300)}
async function pollDeleteJob(jobId){try{const response=await fetch('/api/jobs/'+jobId),job=await response.json();if(!response.ok)throw new Error(job.error||'删除任务不存在');const latest=(job.log||[]).at(-1)||'后台清理正在进行';updateDeleteProgress(job.progress||10,job.status==='queued'?'正在等待解析步骤退出':'正在清理派生数据',latest);if(job.status==='completed'){finishDeleteProgress(true,'删除完成','文档、知识库贡献、图谱数据和过程文件已完成清理。');return}if(job.status==='failed'||job.status==='cancelled'){finishDeleteProgress(false,'后台清理未完成',job.error_message||latest);return}deletePollTimer=setTimeout(()=>pollDeleteJob(jobId),700)}catch(error){finishDeleteProgress(false,'删除状态读取失败',error.message||'请刷新页面确认文档状态')}}
document.querySelectorAll('[data-delete]').forEach(button=>button.onclick=async()=>{if(!confirm('确认软删除该文档及其向量、实体提及和关系贡献吗？'))return;button.disabled=true;deleteSpinner.hidden=false;deleteResult.hidden=true;deleteResult.className='delete-result';deleteClose.disabled=true;deleteMessage.textContent='正在提交删除请求...';updateDeleteProgress(3,'正在取消相关任务','请稍候，不要重复点击删除。');deleteDialog?.showModal();try{const response=await fetch('/api/documents/'+button.dataset.delete+'/delete',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'删除失败');button.closest('tr')?.classList.add('deleting-row');deleteMessage.textContent='文档已从当前项目移除，正在完成后台清理';updateDeleteProgress(8,'文档已软删除','正在等待活动解析步骤安全退出。');pollDeleteJob(data.job_id)}catch(error){button.disabled=false;finishDeleteProgress(false,'删除失败',error.message||'删除请求执行失败')}});
deleteClose?.addEventListener('click',()=>location.reload());deleteDialog?.addEventListener('cancel',event=>{if(deleteClose.disabled)event.preventDefault()});
document.querySelector('[data-reassess-schema]')?.addEventListener('click',async event=>{if(!confirm('将基于当前项目全部已入库文档，从基础知识结构重新评估实体类别和关系约束。该操作不会自动重跑已有文档的实体抽取，是否继续？'))return;const button=event.currentTarget;button.disabled=true;try{const response=await fetch('/api/projects/'+button.dataset.reassessSchema+'/schema/reassess',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'知识结构重新评估提交失败');monitor(data.job_id,'schema_reassessment')}catch(error){alert(error.message||'知识结构重新评估提交失败');button.disabled=false}});
document.querySelector('[data-rebuild-graph]')?.addEventListener('click',async event=>{const button=event.currentTarget;button.disabled=true;try{const response=await fetch('/api/projects/'+button.dataset.rebuildGraph+'/graph',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'图谱重建提交失败');monitor(data.job_id,'graph_sync')}catch(error){alert(error.message||'图谱重建提交失败');button.disabled=false}});
const createDialog=document.querySelector('#create-project-dialog');
document.querySelector('#open-create-project')?.addEventListener('click',()=>createDialog?.showModal());
document.querySelectorAll('[data-close-create-project]').forEach(button=>button.addEventListener('click',()=>createDialog?.close()));
createDialog?.addEventListener('click',event=>{if(event.target===createDialog)createDialog.close()});
const settingsForm=document.querySelector('#project-settings-form'),llmSelect=settingsForm?.querySelector('select[name="llm_model"]'),embeddingSelect=settingsForm?.querySelector('select[name="embedding_model"]'),modelTokenInput=settingsForm?.querySelector('input[name="model_max_tokens"]'),chunkTokenInput=settingsForm?.querySelector('input[name="chunk_max_tokens"]'),modelTokenHint=document.querySelector('#model-token-limit-hint'),chunkTokenHint=document.querySelector('#chunk-token-limit-hint');
let tokenLimits=null;
function applyTokenLimits(clamp=false){if(!tokenLimits||!modelTokenInput||!chunkTokenInput)return;const llmMax=Number(tokenLimits.llm_context_max),embeddingMax=Number(tokenLimits.embedding_context_max);modelTokenInput.max=String(llmMax);if(clamp&&Number(modelTokenInput.value)>llmMax)modelTokenInput.value=String(llmMax);const chunkMax=Math.min(Number(modelTokenInput.value)||llmMax,llmMax,embeddingMax);chunkTokenInput.max=String(chunkMax);if(clamp&&Number(chunkTokenInput.value)>chunkMax)chunkTokenInput.value=String(chunkMax);if(modelTokenHint){modelTokenHint.textContent='当前 LLM 支持的最大上下文：'+llmMax.toLocaleString()+' Token';modelTokenHint.classList.remove('failure')}if(chunkTokenHint){chunkTokenHint.textContent='当前模型组合允许的最大切片：'+chunkMax.toLocaleString()+' Token（Embedding 上限 '+embeddingMax.toLocaleString()+'）';chunkTokenHint.classList.remove('failure')}}
async function refreshTokenLimits(clamp=false){if(!llmSelect||!embeddingSelect)return;if(modelTokenHint)modelTokenHint.textContent='正在读取当前 LLM 的上下文上限...';if(chunkTokenHint)chunkTokenHint.textContent='正在读取 LLM 与 Embedding 的共同上限...';try{const query=new URLSearchParams({llm_model:llmSelect.value,embedding_model:embeddingSelect.value});const response=await fetch('/api/models/limits?'+query),data=await response.json();if(!response.ok)throw new Error(data.error||'模型上限读取失败');tokenLimits=data;applyTokenLimits(clamp)}catch(error){if(modelTokenHint){modelTokenHint.textContent=error.message||'模型上限读取失败';modelTokenHint.classList.add('failure')}if(chunkTokenHint){chunkTokenHint.textContent='请先确认 Ollama 模型可用';chunkTokenHint.classList.add('failure')}}}
function replaceModelOptions(select,models,kind){if(!select)return;const current=select.value,names=models.filter(model=>model.model_type===kind).map(model=>model.name);if(current&&!names.includes(current))names.unshift(current);select.replaceChildren(...names.map(name=>new Option(name,name,name===current,name===current)))}
document.querySelector('#sync-models')?.addEventListener('click',async e=>{const button=e.currentTarget,status=document.querySelector('#model-sync-status');button.disabled=true;if(status){status.hidden=true;status.className='model-sync-status'}try{const r=await fetch('/api/models/sync',{method:'POST'}),d=await r.json();if(!r.ok)throw new Error(d.error||'同步失败');replaceModelOptions(llmSelect,d.models||[],'llm');replaceModelOptions(embeddingSelect,d.models||[],'embedding');await refreshTokenLimits(true);if(status){status.textContent='已同步 '+d.count+' 个 Ollama 模型';status.classList.add('success');status.hidden=false}}catch(error){if(status){status.textContent=error.message||'同步失败';status.classList.add('failure');status.hidden=false}}finally{button.disabled=false}});
llmSelect?.addEventListener('change',()=>refreshTokenLimits(true));embeddingSelect?.addEventListener('change',()=>refreshTokenLimits(true));modelTokenInput?.addEventListener('input',()=>applyTokenLimits(true));
const workspaceViews=new Set(['documents','build','query','settings']);
const workspaceNavigation=[...document.querySelectorAll('[data-workspace-target]')];
const workspacePanels=[...document.querySelectorAll('[data-workspace-panel]')];
function activateWorkspaceView(view,{updateHistory=false}={}){
  const selected=workspaceViews.has(view)?view:'documents';
  for(const button of workspaceNavigation){
    const active=button.dataset.workspaceTarget===selected;
    button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1;
  }
  for(const panel of workspacePanels)panel.hidden=panel.dataset.workspacePanel!==selected;
  if(selected==='settings'&&!tokenLimits)refreshTokenLimits(false);
  if(updateHistory){
    const url=new URL(location.href);url.hash=selected;
    history.pushState({workspaceView:selected},'',url);
  }
}
workspaceNavigation.forEach(button=>button.addEventListener('click',()=>activateWorkspaceView(
  button.dataset.workspaceTarget,{updateHistory:true},
)));
window.addEventListener('popstate',()=>activateWorkspaceView(location.hash.slice(1)));
const initialWorkspaceView=workspaceViews.has(location.hash.slice(1))
  ?location.hash.slice(1):(window.KF.activeJob?(window.KF.activeJobKind==='content_preparation'?'build':'query'):'documents');
activateWorkspaceView(initialWorkspaceView);
const processingForm=document.querySelector('.processing-form'),processingSelects=document.querySelector('#processing-selects'),processingHint=document.querySelector('#processing-hint'),fileInput=document.querySelector('#upload-files');
const parserSupport={native_docx:['.docx'],native_text:['.md','.markdown','.txt'],plaintext_pdf:['.pdf'],docling:['.pdf','.docx'],spreadsheet:['.xlsx','.xls','.csv']};
const strategySupport={default:['.docx','.pdf','.md','.markdown','.txt'],general:['.docx','.pdf','.md','.markdown','.txt','.xlsx','.xls','.csv'],paper:['.docx','.pdf','.md','.markdown','.txt'],table:['.docx','.pdf','.xlsx','.xls','.csv']};
function selectedSuffixes(){return [...(fileInput?.files||[])].map(file=>{const name=file.name.toLowerCase();return name.includes('.')?name.slice(name.lastIndexOf('.')):''})}
function refreshSelect(select,support,suffixes){let available=0;for(const option of select?.options||[]){option.disabled=option.value!=='auto'&&(suffixes.some(suffix=>!support[option.value]?.includes(suffix))||(option.value==='docling'&&!window.KF.docling));if(!option.disabled)available++}if(select?.selectedOptions[0]?.disabled)select.value='auto';return available}
function refreshProcessing(){const selects=processingSelects?.querySelectorAll('select')||[];const suffixes=selectedSuffixes();refreshSelect(selects[0],parserSupport,suffixes);refreshSelect(selects[1],strategySupport,suffixes);const parser=selects[0]?.value||'auto',strategy=selects[1]?.value||'auto';if(processingHint)processingHint.textContent=!suffixes.length?'选择文件后会检查兼容性。':parser==='auto'&&strategy==='auto'?'系统将根据文件类型、文本层和文档结构自动选择解析器与分块策略。':'当前选择可用于所选文件。'}
processingSelects?.querySelectorAll('select').forEach(select=>select.addEventListener('change',refreshProcessing));fileInput?.addEventListener('change',refreshProcessing);refreshProcessing();
const jobDialog=document.querySelector('#job-log-dialog'),jobHistory=document.querySelector('#job-history-log'),jobSummary=document.querySelector('#job-log-summary'),jobError=document.querySelector('#job-log-error');
function scrollJobLogToBottom(){requestAnimationFrame(()=>{if(jobHistory)jobHistory.scrollTop=jobHistory.scrollHeight})}
let jobLogPollTimer=null,jobLogGeneration=0,jobLogFollow=true;
function jobLogNearBottom(){return !jobHistory||jobHistory.scrollHeight-jobHistory.scrollTop-jobHistory.clientHeight<36}
function stopJobLogPolling(){jobLogGeneration++;if(jobLogPollTimer){clearTimeout(jobLogPollTimer);jobLogPollTimer=null}}
const jobStatus={queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',cancelled:'已取消'};
const nativeRuntimeDetails=document.querySelector('#native-runtime-details'),nativeRuntimeStatus=document.querySelector('#native-runtime-status'),nativeRuntimeProgressText=document.querySelector('#native-runtime-progress-text'),nativeRuntimeProgress=document.querySelector('#native-runtime-progress'),nativeRuntimeLogButton=nativeRuntimeDetails?.querySelector('[data-view-job-log]');
function updateNativeRuntime(job){if(!nativeRuntimeDetails||job.kind!=='graphrag_native_build')return;const progress=clampProgress(job.progress);nativeRuntimeDetails.dataset.nativeJob=job.id||'';if(nativeRuntimeStatus)nativeRuntimeStatus.textContent=jobStatus[job.status]||job.status||'未知状态';if(nativeRuntimeProgressText)nativeRuntimeProgressText.textContent=progress+'%';if(nativeRuntimeProgress)nativeRuntimeProgress.style.width=progress+'%';if(nativeRuntimeLogButton){nativeRuntimeLogButton.dataset.job=job.id||'';nativeRuntimeLogButton.disabled=!job.id}}
function jobStageSummary(job){const key=job.metadata?.active_subtask,step=jobSubtasks(job)[key];if(!key||!step)return'';const suffix=step.status==='completed'?'已完成':step.status==='running'?clampProgress(step.progress)+'%':step.status==='failed'?'失败':step.status==='cancelled'?'已取消':'等待';const labels=labelsForJob(job);return(labels[key]||key)+' · '+suffix+(step.detail?' · '+step.detail:'')}
async function refreshJobLog(id,batch,generation,firstLoad=false){try{const endpoint=batch?'/api/job-batches/'+batch:'/api/jobs/'+id,response=await fetch(endpoint),job=await response.json();if(generation!==jobLogGeneration||!jobDialog?.open)return;if(!response.ok)throw new Error(job.error||'任务不存在');const keepFollowing=firstLoad||jobLogFollow||jobLogNearBottom(),degraded=Number(job.metadata?.extraction_errors||0),stageSummary=jobStageSummary(job),count=Number(job.document_count||0);jobSummary.textContent=(jobStatus[job.status]||job.status)+(count?' · '+count+' 个文档':'')+' · 用时 '+formatDuration(job.elapsed_seconds)+(stageSummary?' · '+stageSummary:'')+(degraded?' · '+degraded+' 个知识块已降级':'');jobHistory.textContent=(job.log||[]).join('\n')||'该任务暂无日志。';jobError.hidden=true;if(job.error_message){jobError.textContent=job.error_message;jobError.hidden=false}if(keepFollowing){jobLogFollow=true;scrollJobLogToBottom();}if(['queued','running'].includes(job.status))jobLogPollTimer=setTimeout(()=>refreshJobLog(id,batch,generation),1000)}catch(error){if(generation!==jobLogGeneration||!jobDialog?.open)return;jobSummary.textContent='日志读取失败';jobError.textContent=error.message||'任务不存在';jobError.hidden=false}}
function showJobLog(id,batch=''){if(!id)return;stopJobLogPolling();const generation=jobLogGeneration;jobLogFollow=true;jobDialog?.showModal();jobSummary.textContent='正在读取任务日志...';jobHistory.textContent='';jobError.hidden=true;refreshJobLog(id,batch,generation,true)}
document.querySelectorAll('[data-view-job-log]').forEach(button=>button.addEventListener('click',()=>showJobLog(button.dataset.job,button.dataset.batch||'')));
jobHistory?.addEventListener('scroll',()=>{jobLogFollow=jobLogNearBottom()});
document.querySelectorAll('[data-close-job-log]').forEach(button=>button.addEventListener('click',()=>{stopJobLogPolling();jobDialog?.close()}));
jobDialog?.addEventListener('click',event=>{if(event.target===jobDialog){stopJobLogPolling();jobDialog.close()}});
const structureDialog=document.querySelector('#structure-dialog'),structureSummary=document.querySelector('#structure-summary'),structureError=document.querySelector('#structure-error'),structureTools=document.querySelector('#structure-tools'),structureList=document.querySelector('#structure-list'),structureSearch=document.querySelector('#structure-search'),structureType=document.querySelector('#structure-type'),structurePager=document.querySelector('#structure-pager'),structureRange=document.querySelector('#structure-range'),structurePrev=document.querySelector('#structure-prev'),structureNext=document.querySelector('#structure-next');
const structureTypeNames={heading:'标题',paragraph:'正文段落',table:'表格',table_row:'表格行'};let structureData=null,structurePage=1;const structurePageSize=20;
function filteredStructureBlocks(){if(!structureData)return[];const query=(structureSearch.value||'').trim().toLowerCase(),type=structureType.value;return structureData.blocks.filter(block=>{if(type&&block.block_type!==type)return false;if(!query)return true;return [block.text,block.section_title,(block.heading_path||[]).join(' / '),block.page].some(value=>String(value??'').toLowerCase().includes(query))})}
function renderStructureBlocks(){const blocks=filteredStructureBlocks(),pages=Math.max(1,Math.ceil(blocks.length/structurePageSize));structurePage=Math.min(Math.max(1,structurePage),pages);const start=(structurePage-1)*structurePageSize,visible=blocks.slice(start,start+structurePageSize);structureList.innerHTML=visible.map(block=>{const path=(block.heading_path||[]).filter(Boolean).join(' / '),metadata=block.metadata&&Object.keys(block.metadata).length?'<details class="structure-metadata"><summary>查看元数据</summary><pre>'+esc(JSON.stringify(block.metadata,null,2))+'</pre></details>':'';return '<article class="structure-row"><div class="structure-row-head"><div class="structure-row-meta"><strong>#'+esc(block.order)+'</strong><span class="block-type">'+esc(structureTypeNames[block.block_type]||block.block_type||'未知')+'</span>'+(block.page?'<span>第 '+esc(block.page)+' 页</span>':'')+'<span>层级 '+esc(block.level||1)+'</span><span>章节：'+esc(block.section_title||'未命名')+'</span></div></div>'+(path?'<div class="structure-row-meta">标题路径：'+esc(path)+'</div>':'')+'<p class="structure-text">'+esc(block.text||'')+'</p>'+metadata+'</article>'}).join('')||'<div class="empty">没有符合条件的结构块。</div>';structureRange.textContent=blocks.length?'第 '+(start+1)+'-'+Math.min(start+structurePageSize,blocks.length)+' 条，共 '+blocks.length+' 条':'共 0 条';structurePrev.disabled=structurePage<=1;structureNext.disabled=structurePage>=pages;structurePager.hidden=false}
async function showStructure(documentId){structureDialog?.showModal();structureSummary.textContent='正在读取解析结果...';structureError.hidden=true;structureTools.hidden=true;structurePager.hidden=true;structureList.innerHTML='<div class="empty">正在加载结构块...</div>';structureSearch.value='';structureType.replaceChildren(new Option('全部类型',''));structureData=null;try{const response=await fetch('/api/documents/'+documentId+'/structure'),data=await response.json();if(!response.ok)throw new Error(data.error||'结构块读取失败');structureData=data;const parser=data.plan?.effective_parser||'未知解析器',strategy=data.plan?.effective_strategy||'未知策略';structureSummary.textContent=data.document_name+' · '+parser+' · '+strategy+' · '+data.block_count+' 个结构块';for(const type of data.block_types||[])structureType.add(new Option(structureTypeNames[type]||type,type));structureTools.hidden=false;structurePage=1;renderStructureBlocks()}catch(error){structureSummary.textContent='结构块读取失败';structureError.textContent=error.message||'结构块读取失败';structureError.hidden=false;structureList.innerHTML=''}}
document.querySelectorAll('[data-view-structure]').forEach(button=>button.addEventListener('click',()=>showStructure(button.dataset.viewStructure)));
structureSearch?.addEventListener('input',()=>{structurePage=1;renderStructureBlocks()});structureType?.addEventListener('change',()=>{structurePage=1;renderStructureBlocks()});structurePrev?.addEventListener('click',()=>{structurePage--;renderStructureBlocks()});structureNext?.addEventListener('click',()=>{structurePage++;renderStructureBlocks()});document.querySelectorAll('[data-close-structure]').forEach(button=>button.addEventListener('click',()=>structureDialog?.close()));structureDialog?.addEventListener('click',event=>{if(event.target===structureDialog)structureDialog.close()});
const chunkDialog=document.querySelector('#chunk-dialog'),chunkSummary=document.querySelector('#chunk-summary'),chunkError=document.querySelector('#chunk-error'),chunkTools=document.querySelector('#chunk-tools'),chunkList=document.querySelector('#chunk-list'),chunkSearch=document.querySelector('#chunk-search'),chunkRole=document.querySelector('#chunk-role'),chunkPager=document.querySelector('#chunk-pager'),chunkRange=document.querySelector('#chunk-range'),chunkPrev=document.querySelector('#chunk-prev'),chunkNext=document.querySelector('#chunk-next');let chunkData=null,chunkPage=1;const chunkPageSize=12;
function filteredChunks(){if(!chunkData)return[];const query=(chunkSearch.value||'').trim().toLowerCase(),role=chunkRole.value;return chunkData.chunks.filter(chunk=>{const metadata=chunk.metadata||{},chunkRoleValue=metadata.chunk_role||'child';if(role&&chunkRoleValue!==role)return false;if(!query)return true;return [chunk.chunk_key,chunk.section_title,chunk.content].some(value=>String(value??'').toLowerCase().includes(query))})}
function renderChunks(){const chunks=filteredChunks(),pages=Math.max(1,Math.ceil(chunks.length/chunkPageSize));chunkPage=Math.min(Math.max(1,chunkPage),pages);const start=(chunkPage-1)*chunkPageSize,visible=chunks.slice(start,start+chunkPageSize);chunkList.innerHTML=visible.map((chunk,index)=>{const metadata=chunk.metadata||{},role=metadata.chunk_role||'child',enabled=metadata.graph_extraction_enabled!==false,parent=metadata.parent_key||'';return '<article class="structure-row"><div class="structure-row-head"><div class="structure-row-meta"><strong>#'+(start+index+1)+'</strong><span class="block-type">'+esc(chunk.section_title||'未命名章节')+'</span><span>'+esc(chunk.chunk_key)+'</span></div></div><div class="chunk-badges"><span class="chunk-badge">'+esc(role==='parent'?'父级摘要块':role==='table_row'?'表格行块':'检索子块')+'</span><span class="chunk-badge">'+esc(metadata.token_count||0)+' Token</span><span class="chunk-badge '+(enabled?'graph':'skipped')+'">'+(enabled?'参与图谱抽取':'不参与图谱抽取')+'</span>'+(parent?'<span class="chunk-badge">父块：'+esc(parent)+'</span>':'')+'</div><p class="structure-text">'+esc(chunk.content||'')+'</p><details class="structure-metadata"><summary>查看元数据</summary><pre>'+esc(JSON.stringify(metadata,null,2))+'</pre></details></article>'}).join('')||'<div class="empty">没有符合条件的知识块。</div>';chunkRange.textContent=chunks.length?'第 '+(start+1)+'-'+Math.min(start+chunkPageSize,chunks.length)+' 条，共 '+chunks.length+' 条':'共 0 条';chunkPrev.disabled=chunkPage<=1;chunkNext.disabled=chunkPage>=pages;chunkPager.hidden=false}
async function showChunks(documentId){chunkDialog?.showModal();chunkSummary.textContent='正在读取分块结果...';chunkError.hidden=true;chunkTools.hidden=true;chunkPager.hidden=true;chunkList.innerHTML='<div class="empty">正在加载知识块...</div>';chunkSearch.value='';chunkRole.value='';chunkData=null;try{const response=await fetch('/api/documents/'+documentId+'/chunks'),data=await response.json();if(!response.ok)throw new Error(data.error||'知识块读取失败');chunkData=data;chunkSummary.textContent=data.document_name+' · '+data.chunk_count+' 个知识块';chunkTools.hidden=false;chunkPage=1;renderChunks()}catch(error){chunkSummary.textContent='知识块读取失败';chunkError.textContent=error.message||'知识块读取失败';chunkError.hidden=false;chunkList.innerHTML=''}}
document.querySelectorAll('[data-view-chunks]').forEach(button=>button.addEventListener('click',()=>showChunks(button.dataset.viewChunks)));chunkSearch?.addEventListener('input',()=>{chunkPage=1;renderChunks()});chunkRole?.addEventListener('change',()=>{chunkPage=1;renderChunks()});chunkPrev?.addEventListener('click',()=>{chunkPage--;renderChunks()});chunkNext?.addEventListener('click',()=>{chunkPage++;renderChunks()});document.querySelectorAll('[data-close-chunks]').forEach(button=>button.addEventListener('click',()=>chunkDialog?.close()));chunkDialog?.addEventListener('click',event=>{if(event.target===chunkDialog)chunkDialog.close()});
const nativeArtifactDialog=document.querySelector('#native-artifact-dialog'),nativeArtifactTitle=document.querySelector('#native-artifact-title'),nativeArtifactSummary=document.querySelector('#native-artifact-summary'),nativeArtifactError=document.querySelector('#native-artifact-error'),nativeArtifactKind=document.querySelector('#native-artifact-kind'),nativeArtifactContent=document.querySelector('#native-artifact-content');
async function loadNativeArtifact(kind){nativeArtifactError.hidden=true;nativeArtifactContent.textContent='正在读取...';try{const response=await fetch('/api/projects/'+window.KF.projectId+'/graphrag-native/artifacts/'+encodeURIComponent(kind)+'?limit=100'),data=await response.json();if(!response.ok)throw new Error(data.error||'原生产物读取失败');nativeArtifactTitle.textContent='GraphRAG Native 过程产物';nativeArtifactSummary.textContent=data.logical_name+' · 显示 '+data.items.length+' / '+data.total+' 条 · run '+data.run_id;nativeArtifactContent.textContent=JSON.stringify(data.items,null,2)}catch(error){nativeArtifactSummary.textContent='读取失败';nativeArtifactError.textContent=error.message||'原生产物读取失败';nativeArtifactError.hidden=false;nativeArtifactContent.textContent=''}}
async function showNativeDocuments(documentId){nativeArtifactDialog?.showModal();nativeArtifactKind.closest('.structure-tools').hidden=true;nativeArtifactError.hidden=true;nativeArtifactContent.textContent='正在生成 Native Documents 预览...';try{const response=await fetch('/api/projects/'+window.KF.projectId+'/graphrag-native/source-documents?document_id='+encodeURIComponent(documentId)+'&limit=100'),data=await response.json();if(!response.ok)throw new Error(data.error||'Native Documents 读取失败');nativeArtifactTitle.textContent='Native Documents';nativeArtifactSummary.textContent='显示 '+data.items.length+' / '+data.total+' 条；这是 GraphRAG 的冻结输入，不是 Knowledge Forge Chunk。';nativeArtifactContent.textContent=JSON.stringify(data.items,null,2)}catch(error){nativeArtifactSummary.textContent='读取失败';nativeArtifactError.textContent=error.message||'Native Documents 读取失败';nativeArtifactError.hidden=false;nativeArtifactContent.textContent=''}}
document.querySelectorAll('[data-view-native-documents]').forEach(button=>button.addEventListener('click',()=>showNativeDocuments(button.dataset.viewNativeDocuments)));
document.querySelector('#view-native-artifacts')?.addEventListener('click',()=>{nativeArtifactDialog?.showModal();nativeArtifactKind.closest('.structure-tools').hidden=false;nativeArtifactKind.value='documents';loadNativeArtifact('documents')});
nativeArtifactKind?.addEventListener('change',()=>loadNativeArtifact(nativeArtifactKind.value));
document.querySelectorAll('[data-close-native-artifacts]').forEach(button=>button.addEventListener('click',()=>nativeArtifactDialog?.close()));nativeArtifactDialog?.addEventListener('click',event=>{if(event.target===nativeArtifactDialog)nativeArtifactDialog.close()});
const toast=document.querySelector('.notice');if(toast){setTimeout(()=>toast.classList.add('toast-hide'),4200);setTimeout(()=>toast.remove(),4550);const url=new URL(location.href);url.searchParams.delete('kind');url.searchParams.delete('message');history.replaceState({},'',url.pathname+url.search)}
document.querySelector('#search-form')?.addEventListener('submit',async e=>{e.preventDefault();const target=document.querySelector('#search-results'),q=new FormData(e.currentTarget).get('query');target.innerHTML='<p class="muted">正在检索...</p>';const r=await fetch('/api/search?project='+window.KF.projectId+'&q='+encodeURIComponent(q)),d=await r.json();if(!r.ok){target.innerHTML='<p class="failure">'+esc(d.error)+'</p>';return}let h='<div class="result-group"><h2>知识块</h2>';for(const x of d.chunks)h+='<div class="result-card"><h3>'+esc(x.section_title||x.object_key)+' <span class="score">'+Number(x.score).toFixed(3)+'</span></h3><p>'+esc(x.content)+'</p><a href="'+esc(x.storage_path)+'" target="_blank">'+esc(x.original_filename)+'</a></div>';h+='</div><div class="result-group"><h2>实体</h2>';for(const x of d.entities)h+='<div class="result-card"><h3>'+esc(x.name)+' <span class="muted">'+esc(x.entity_type)+'</span></h3><p>'+esc(x.description)+'</p></div>';h+='</div><div class="result-group"><h2>关系</h2>';for(const x of d.relations)h+='<div class="result-card"><h3>'+esc(x.source_name)+' <span class="relation">'+esc(x.relation_type)+'</span> '+esc(x.target_name)+'</h3><p>'+esc(x.description)+'</p><a href="'+esc(x.storage_path)+'" target="_blank">'+esc(x.original_filename)+'</a></div>';target.innerHTML=h});
const nativeQueryForm=document.querySelector('#native-search-form'),nativeModeHelp=document.querySelector('[data-native-query-mode-help]');
const nativeModeInfo={
  local:{label:'详细问答',help:'综合相关实体、关系、社区报告和原文片段，适合询问具体对象及其联系。'},
  global:{label:'全局总结',help:'综合各社区报告回答全局性问题，适合总结整个资料集的主题、趋势和格局。'},
  drift:{label:'主题深入分析',help:'从相关社区出发继续追问和扩展证据，适合分析复杂主题及其相互影响。'},
  basic:{label:'原文问答',help:'主要依据最相关的 TextUnit 原文片段回答，适合查找具体事实或原文依据。'},
};
function updateNativeModeHelp(){const mode=nativeQueryForm?.querySelector('[name="mode"]')?.value||'local';if(nativeModeHelp)nativeModeHelp.textContent=nativeModeInfo[mode]?.help||''}
function nativeContextCard(context){
  const record=context.record||{},source=context.source_document||{},sourceMap=context.source_map||{},table=String(context.table||'');
  const typeLabels={entities:'实体',relationships:'关系',reports:'社区报告',communities:'社区',text_units:'原文片段'};
  let title=context.native_id||'上下文',excerpt='';
  if(table==='entities'){title=record.entity||record.title||title;excerpt=record.description||''}
  else if(table==='relationships'){title=(record.source||'源实体')+' → '+(record.target||'目标实体');excerpt=record.description||''}
  else if(table==='reports'||table==='communities'){title=record.title||('社区 '+(record.id??context.context_id??''));excerpt=record.summary||record.content||record.full_content||record.description||''}
  else if(table==='text_units'){title=source.original_filename||'原文片段';excerpt=record.text||''}
  const location=[sourceMap.page?'第 '+sourceMap.page+' 页':'',Array.isArray(sourceMap.heading_path)?sourceMap.heading_path.filter(Boolean).join(' / '):sourceMap.heading_path||''].filter(Boolean).join(' · ');
  const sourceLink=String(source.download_path||'').startsWith('/api/documents/')?'<a href="'+esc(source.download_path)+'" target="_blank" rel="noopener">查看源文档</a>':'';
  return '<article class="native-context-card"><div><span>'+esc(typeLabels[table]||table||'上下文')+'</span><b>'+esc(title)+'</b></div>'+(excerpt?'<p>'+esc(String(excerpt).slice(0,700))+'</p>':'')+((location||sourceLink)?'<small>'+esc(location)+(location&&sourceLink?' · ':'')+sourceLink+'</small>':'')+'</article>';
}
nativeQueryForm?.querySelector('[name="mode"]')?.addEventListener('change',updateNativeModeHelp);updateNativeModeHelp();
nativeQueryForm?.addEventListener('submit',async e=>{
  e.preventDefault();const target=document.querySelector('#native-search-results'),data=new FormData(e.currentTarget),mode=String(data.get('mode')||'local');
  target.innerHTML='<p class="muted">正在使用“'+esc(nativeModeInfo[mode]?.label||mode)+'”生成回答...</p>';
  try{
    const r=await fetch('/api/projects/'+window.KF.projectId+'/graphrag-native/query',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams(data)}),d=await r.json();
    if(!r.ok)throw Error(d.error||'知识问答失败');
    const contexts=Array.isArray(d.context_records)?d.context_records:[];
    target.innerHTML='<article class="native-query-answer"><div class="native-query-answer-head"><h3>'+esc(nativeModeInfo[d.mode]?.label||d.mode||'知识问答')+'</h3><span>'+esc(contexts.length)+' 条回答依据</span></div><p>'+esc(d.answer||'未生成回答')+'</p></article><section class="native-query-evidence"><h3>回答依据</h3><p class="muted">以下内容是 GraphRAG 本次实际使用的检索上下文，用于核查回答来源；它们不是模型生成的引用编号。</p><div class="native-context-list">'+(contexts.map(nativeContextCard).join('')||'<div class="empty">本次回答没有返回可展示的上下文。</div>')+'</div></section>';
  }catch(error){target.innerHTML='<p class="failure">'+esc(error.message||'知识问答失败')+'</p>'}
});

{
const graphRoot=document.querySelector('#embedded-graph-browser');
if(graphRoot&&graphRoot.dataset.ready==='true'){
  const q=selector=>graphRoot.querySelector(selector);
  const NS='http://www.w3.org/2000/svg';
  const colors=['#2677d8','#b54cda','#d85c49','#119c87','#d18b18','#5974d9','#c45379','#1d9aa7','#7b8b3d','#8b5e3c'];
  const graphState={nodes:[],edges:[],groups:[],nodeById:new Map(),query:'',group:'',showIsolates:true,selectedNode:'',selectedEdge:null,positions:new Map(),totalEntities:0,totalRelationships:0,totalCommunities:0,communityReady:false,zoom:1,worldWidth:1200,worldHeight:720};
  const clean=value=>String(value??'').replace(/\s+/g,' ').trim();
  const short=(value,size=52)=>{const result=clean(value);return result.length>size?result.slice(0,size-1)+'…':result};
  const colorFor=value=>colors[Math.abs([...String(value)].reduce((sum,char)=>sum+char.charCodeAt(0),0))%colors.length];
  const svg=(name,attributes={})=>{const element=document.createElementNS(NS,name);for(const [key,value] of Object.entries(attributes))element.setAttribute(key,String(value));return element};
  const groupLabel=group=>clean(group.label||group.key||'未分组');
  const edgeText=edge=>clean(edge.predicate||edge.description||'相关');
  const evidenceText=edge=>{
    if(clean(edge.evidence_text))return clean(edge.evidence_text);
    const ids=Array.isArray(edge.text_unit_ids)?edge.text_unit_ids:[];
    return ids.length?'TextUnit：'+ids.join(', '):'—';
  };
  function normalizeNative(graph,communities,reports){
    const nodes=(graph.nodes||[]).map((node,index)=>({...node,id:String(node.id||node.title||index),title:clean(node.title||node.name||node.id),group:'未分组'}));
    const nodeById=new Map(nodes.map(node=>[String(node.id),node]));
    const byTitle=new Map(nodes.map(node=>[String(node.title),node]));
    const reportByKey=new Map((reports||[]).map(item=>[String(item.level??0)+':'+String(item.community??item.id??''),item]));
    const groups=[];
    for(const item of communities||[]){
      const key=String(item.level??0)+':'+String(item.community??item.id??'');
      const report=reportByKey.get(key)||{};
      const group={key,label:'社区 '+String(item.community??item.id??'—')+' · L'+String(item.level??0),size:Number(item.size||(item.entity_ids||[]).length||0),relationship_count:Number((item.relationship_ids||[]).length||0),summary:clean(report.summary||report.full_content||report.description||'暂无社区摘要')};
      groups.push(group);
      for(const entityId of item.entity_ids||[]){
        const node=nodeById.get(String(entityId));
        if(node&&node.group==='未分组')node.group=key;
      }
    }
    const edges=(graph.edges||[]).map((edge,index)=>{
      const source=byTitle.get(String(edge.source??''))||nodeById.get(String(edge.source??''));
      const target=byTitle.get(String(edge.target??''))||nodeById.get(String(edge.target??''));
      return {...edge,id:String(edge.id||index),source_id:String(source?.id||''),target_id:String(target?.id||''),source_title:clean(source?.title||edge.source),target_title:clean(target?.title||edge.target),predicate:clean(edge.description||'相关')};
    }).filter(edge=>edge.source_id&&edge.target_id);
    if(nodes.some(node=>node.group==='未分组'))groups.push({key:'未分组',label:'未归入社区',size:nodes.filter(node=>node.group==='未分组').length,relationship_count:0,summary:'当前原生产物没有为这些实体分配社区。'});
    return {nodes,edges,groups,totalEntities:Number(graph.total_entities||nodes.length),totalRelationships:Number(graph.total_relationships||edges.length),totalCommunities:groups.length,communityReady:true};
  }
  function normalizeKnowledgeForge(graph){
    return {
      nodes:(graph.nodes||[]).map(node=>({...node,id:String(node.id),title:clean(node.title||node.name||node.id),group:String(node.group||node.type||'其他')})),
      edges:(graph.edges||[]).map((edge,index)=>({...edge,id:String(edge.id||index),source_id:String(edge.source),target_id:String(edge.target),source_title:clean(edge.source_title||edge.source),target_title:clean(edge.target_title||edge.target)})),
      groups:graph.groups||[],
      totalEntities:Number(graph.total_entities||(graph.nodes||[]).length),
      totalRelationships:Number(graph.total_relationships||(graph.edges||[]).length),
      totalCommunities:Number(graph.total_communities||0),
      communityReady:Boolean(graph.community_ready),
    };
  }
  function visibleGraph(){
    const term=graphState.query.toLowerCase();
    const matches=node=>(!graphState.group||String(node.group)===graphState.group)&&(!term||[node.title,node.type,node.description,(node.aliases||[]).join(' ')].some(value=>String(value??'').toLowerCase().includes(term)))&&(graphState.showIsolates||Number(node.degree||0)>0);
    let nodes=graphState.nodes.filter(matches);
    let ids=new Set(nodes.map(node=>String(node.id)));
    if(term){
      const seeds=new Set(ids);
      for(const edge of graphState.edges){
        if(seeds.has(edge.source_id)||seeds.has(edge.target_id)){ids.add(edge.source_id);ids.add(edge.target_id)}
      }
      nodes=graphState.nodes.filter(node=>ids.has(String(node.id))&&(!graphState.group||String(node.group)===graphState.group));
      ids=new Set(nodes.map(node=>String(node.id)));
    }
    const edges=graphState.edges.filter(edge=>ids.has(edge.source_id)&&ids.has(edge.target_id));
    return {nodes,edges};
  }
  function layout(nodes,edges){
    const connectedIds=new Set();
    for(const edge of edges){connectedIds.add(edge.source_id);connectedIds.add(edge.target_id)}
    const connected=nodes.filter(node=>connectedIds.has(String(node.id)));
    const isolates=nodes.filter(node=>!connectedIds.has(String(node.id)));
    const grouped=new Map();
    for(const node of connected){const key=String(node.group||'其他');if(!grouped.has(key))grouped.set(key,[]);grouped.get(key).push(node)}
    const entries=[...grouped.entries()].sort((left,right)=>left[0].localeCompare(right[0]));
    const isolateColumns=isolates.length?Math.max(4,Math.ceil(Math.sqrt(isolates.length*2))):0;
    const width=Math.max(1200,isolateColumns*135),connectedHeight=connected.length?600:80;
    const columns=Math.max(1,Math.ceil(Math.sqrt(Math.max(1,entries.length)))),rows=Math.max(1,Math.ceil(Math.max(1,entries.length)/columns));
    const isolateRows=isolates.length?Math.ceil(isolates.length/isolateColumns):0;
    const height=Math.max(720,connectedHeight+(isolates.length?70+isolateRows*78:0));
    const cellWidth=width/columns,cellHeight=connectedHeight/rows,labels=[];
    graphState.positions.clear();
    entries.forEach(([key,items],groupIndex)=>{
      const column=groupIndex%columns,row=Math.floor(groupIndex/columns),left=column*cellWidth,top=row*cellHeight;
      const innerColumns=Math.max(1,Math.ceil(Math.sqrt(items.length))),innerRows=Math.max(1,Math.ceil(items.length/innerColumns));
      items.sort((a,b)=>a.title.localeCompare(b.title)).forEach((node,index)=>{
        graphState.positions.set(String(node.id),{x:left+(index%innerColumns+.5)*(cellWidth/innerColumns),y:top+32+(Math.floor(index/innerColumns)+.5)*((cellHeight-40)/innerRows)});
      });
      labels.push({key,label:groupLabel(graphState.groups.find(group=>String(group.key)===key)||{key}),count:items.length,x:left+12,y:top+20});
    });
    if(isolates.length){
      const top=connected.length?connectedHeight+38:48;
      labels.push({key:'__isolates__',label:'孤立实体',count:isolates.length,x:16,y:top-16});
      isolates.sort((a,b)=>a.title.localeCompare(b.title)).forEach((node,index)=>{
        graphState.positions.set(String(node.id),{
          x:(index%isolateColumns+.5)*(width/isolateColumns),
          y:top+(Math.floor(index/isolateColumns)+.5)*78,
        });
      });
    }
    graphState.worldWidth=width;graphState.worldHeight=height;
    return labels;
  }
  function updateViewport(){
    const network=q('[data-graph-network]'),zoom=Math.max(.6,Math.min(3,graphState.zoom));
    const width=graphState.worldWidth/zoom,height=graphState.worldHeight/zoom;
    let centerX=graphState.worldWidth/2,centerY=graphState.worldHeight/2;
    if(graphState.selectedNode&&graphState.positions.has(graphState.selectedNode)){
      const point=graphState.positions.get(graphState.selectedNode);centerX=point.x;centerY=point.y;
    }else if(graphState.selectedEdge){
      const source=graphState.positions.get(graphState.selectedEdge.source_id),target=graphState.positions.get(graphState.selectedEdge.target_id);
      if(source&&target){centerX=(source.x+target.x)/2;centerY=(source.y+target.y)/2}
    }
    const x=Math.max(0,Math.min(graphState.worldWidth-width,centerX-width/2));
    const y=Math.max(0,Math.min(graphState.worldHeight-height,centerY-height/2));
    network.setAttribute('viewBox',[x,y,width,height].join(' '));
    q('[data-graph-zoom-value]').textContent=Math.round(zoom*100)+'%';
  }
  function setZoom(value){graphState.zoom=Math.max(.6,Math.min(3,Number(value)||1));updateViewport()}
  function entityBadge(node,title){
    return '<span class="graph-mini-node"><span class="graph-entity-dot" style="background:'+colorFor(node?.group||'其他')+'"></span><span>'+esc(title||node?.title||'未知实体')+'</span></span>';
  }
  function relationCard(edge,clickable=false){
    const source=graphState.nodeById.get(edge.source_id),target=graphState.nodeById.get(edge.target_id);
    const tag=clickable?'button':'div',attribute=clickable?' type="button" data-detail-edge="'+esc(edge.id)+'"':'';
    return '<'+tag+' class="graph-relation-card'+(clickable?' button-card':'')+'"'+attribute+'>'+entityBadge(source,edge.source_title)+'<span class="graph-mini-relation"><b>'+esc(short(edgeText(edge),24))+'</b></span>'+entityBadge(target,edge.target_title)+'</'+tag+'>';
  }
  function renderDetails(){
    const panel=q('[data-graph-details]');
    const hasSelection=Boolean(graphState.selectedEdge||graphState.selectedNode);
    q('[data-graph-community-status]').hidden=hasSelection;
    q('[data-graph-groups]').hidden=hasSelection;
    if(graphState.selectedEdge){
      const edge=graphState.selectedEdge;
      panel.innerHTML='<h3>关系详情</h3>'+relationCard(edge)+'<dl><div><dt>关系说明</dt><dd>'+esc(edgeText(edge))+'</dd></div><div><dt>强度</dt><dd>'+esc(edge.confidence??edge.weight??'—')+'</dd></div><div><dt>原文证据</dt><dd>'+esc(evidenceText(edge))+'</dd></div></dl>';
      return;
    }
    const node=graphState.nodeById.get(graphState.selectedNode);
    if(!node){panel.innerHTML='<h3>浏览说明</h3><p>圆点表示实体，箭头表示关系，颜色表示社区或实体类别。点击实体或关系查看详情；点击图中空白处恢复完整视图。</p>';return}
    const adjacent=graphState.edges.filter(edge=>edge.source_id===String(node.id)||edge.target_id===String(node.id));
    const group=graphState.groups.find(item=>String(item.key)===String(node.group));
    panel.innerHTML='<div class="graph-entity-heading"><span class="graph-entity-dot" style="background:'+colorFor(node.group)+'"></span><h3>'+esc(node.title)+'</h3></div><dl><div><dt>实体类别</dt><dd>'+esc(node.type||'—')+'</dd></div><div><dt>连接数</dt><dd>'+esc(node.degree??adjacent.length)+'</dd></div><div><dt>所属分组</dt><dd>'+esc(groupLabel(group||{key:node.group}))+'</dd></div><div><dt>实体说明</dt><dd>'+esc(node.description||'—')+'</dd></div><div><dt>别名</dt><dd>'+esc((node.aliases||[]).join('、')||'—')+'</dd></div></dl><h4>相邻关系（'+adjacent.length+'）</h4>'+(adjacent.length?adjacent.slice(0,20).map(edge=>relationCard(edge,true)).join(''):'<p>暂无相邻关系</p>');
    panel.querySelectorAll('[data-detail-edge]').forEach(button=>button.addEventListener('click',()=>{graphState.selectedEdge=graphState.edges.find(edge=>edge.id===button.dataset.detailEdge)||null;graphState.selectedNode='';renderGraph();renderDetails()}));
  }
  function renderGroups(){
    const box=q('[data-graph-groups]');
    const status=q('[data-graph-community-status]');
    if(status){
      if(graphState.communityReady){
        status.innerHTML='<h3>社区聚类</h3><p>当前图谱包含 '+esc(graphState.totalCommunities||graphState.groups.length)+' 个原始社区。颜色、分区和下方卡片均按社区划分；未连接实体会合并显示。</p>';
      }else{
        status.innerHTML='<h3>社区尚未生成</h3><p class="failure">当前图谱版本没有对应的社区聚类结果。下方仅按实体类别临时布局，不代表社区；重新构建后将自动生成社区。</p>';
      }
    }
    box.innerHTML=graphState.groups.map(group=>'<button type="button" class="embedded-graph-group-card'+(graphState.group===String(group.key)?' is-active':'')+'" data-embedded-group="'+esc(group.key)+'"><i style="background:'+colorFor(group.key)+'"></i><b>'+esc(groupLabel(group))+'</b><small>'+esc(group.size??0)+' 个实体 · '+esc(group.relationship_count??0)+' 条关系</small><span>'+esc(short(group.summary||'',120))+'</span></button>').join('');
    box.querySelectorAll('[data-embedded-group]').forEach(button=>button.addEventListener('click',()=>{graphState.group=graphState.group===button.dataset.embeddedGroup?'':button.dataset.embeddedGroup;q('[data-graph-group-filter]').value=graphState.group;clearSelection();renderAll()}));
  }
  function renderAudit(edges){
    q('[data-graph-audit-summary]').textContent='显示 '+edges.length+' / '+graphState.totalRelationships+' 条关系；点击“图谱连接”可查看网络结构。';
    q('[data-graph-audit-rows]').innerHTML=edges.map(edge=>'<tr><td>'+esc(edge.source_title)+'</td><td>'+esc(edgeText(edge))+'</td><td>'+esc(edge.target_title)+'</td><td>'+esc(edge.confidence??edge.weight??'—')+'</td><td>'+esc(evidenceText(edge))+'</td></tr>').join('');
  }
  function renderGraph(){
    const {nodes,edges}=visibleGraph(),labels=layout(nodes,edges);
    const labelLayer=q('[data-graph-group-labels]'),edgeLayer=q('[data-graph-edges]'),nodeLayer=q('[data-graph-nodes]');
    labelLayer.replaceChildren();edgeLayer.replaceChildren();nodeLayer.replaceChildren();
    for(const item of labels){const label=svg('text',{x:item.x,y:item.y,class:'embedded-graph-group-label',fill:colorFor(item.key)});label.textContent=item.label+' ('+item.count+')';labelLayer.append(label)}
    const selected=graphState.selectedNode,neighbours=new Set();
    if(selected)for(const edge of edges){if(edge.source_id===selected)neighbours.add(edge.target_id);if(edge.target_id===selected)neighbours.add(edge.source_id)}
    const selectedEdgeIds=new Set(graphState.selectedEdge?[graphState.selectedEdge.source_id,graphState.selectedEdge.target_id]:[]);
    for(const edge of edges){
      const source=graphState.positions.get(edge.source_id),target=graphState.positions.get(edge.target_id);if(!source||!target)continue;
      const related=graphState.selectedEdge?graphState.selectedEdge.id===edge.id:(!selected||edge.source_id===selected||edge.target_id===selected);
      const line=svg('line',{x1:source.x,y1:source.y,x2:target.x,y2:target.y,class:'embedded-graph-edge'+(related?'':' is-dim')+(graphState.selectedEdge?.id===edge.id?' is-selected':''),'marker-end':'url(#embedded-graph-arrow)'});
      line.style.strokeWidth=String(Math.min(5,1+Math.log1p(Number(edge.weight||edge.confidence)||1)));const title=svg('title');title.textContent=edge.source_title+' → '+edge.target_title+'\n'+edgeText(edge);line.append(title);
      line.addEventListener('click',event=>{event.stopPropagation();graphState.selectedEdge=edge;graphState.selectedNode='';renderGraph();renderDetails()});edgeLayer.append(line);
    }
    for(const node of nodes){
      const point=graphState.positions.get(String(node.id));if(!point)continue;
      const relevant=graphState.selectedEdge?selectedEdgeIds.has(String(node.id)):(!selected||String(node.id)===selected||neighbours.has(String(node.id)));
      const group=svg('g',{class:'embedded-graph-node'+(relevant?'':' is-dim')+(String(node.id)===selected?' is-selected':''),transform:'translate('+point.x+' '+point.y+')'});
      const circle=svg('circle',{r:Math.min(20,9+Math.log1p(Number(node.degree)||0)*3),fill:colorFor(node.group)}),title=svg('title');title.textContent=node.title+'\n'+String(node.type||'')+' · '+String(node.degree||0)+' 个连接';circle.append(title);
      const label=svg('text',{x:0,y:29});label.textContent=short(node.title,20);group.append(circle,label);group.addEventListener('click',event=>{event.stopPropagation();graphState.selectedNode=String(node.id);graphState.selectedEdge=null;renderGraph();renderDetails()});nodeLayer.append(group);
    }
    q('[data-graph-count]').textContent='当前显示 '+nodes.length+' / '+graphState.totalEntities+' 个实体，'+edges.length+' / '+graphState.totalRelationships+' 条关系';
    updateViewport();
    renderAudit(edges);
  }
  function renderAll(){renderGraph();renderDetails();renderGroups()}
  function clearSelection(){graphState.selectedNode='';graphState.selectedEdge=null}
  function setTab(name){q('[data-graph-panel="network"]').hidden=name!=='network';q('[data-graph-panel="audit"]').hidden=name!=='audit';graphRoot.querySelectorAll('[data-graph-tab]').forEach(button=>{const active=button.dataset.graphTab===name;button.classList.toggle('is-active',active);button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1})}
  async function loadEmbeddedGraph(){
    try{
      let normalized;
      if(graphRoot.dataset.strategy==='graphrag_native'){
        const base='/api/projects/'+window.KF.projectId+'/graphrag-native/';
        const responses=await Promise.all([fetch(base+'graph?limit=1000&include_isolates=1'),fetch(base+'communities?limit=500'),fetch(base+'community-reports?limit=500')]);
        const payloads=await Promise.all(responses.map(response=>response.json()));
        const failed=responses.findIndex(response=>!response.ok);if(failed>=0)throw Error(payloads[failed].error||'原生图谱读取失败');
        normalized=normalizeNative(payloads[0],payloads[1].items||[],payloads[2].items||[]);
      }else{
        const response=await fetch('/api/projects/'+window.KF.projectId+'/knowledge-forge/graph?limit=5000&include_isolates=1'),payload=await response.json();
        if(!response.ok)throw Error(payload.error||'Knowledge Forge 图谱读取失败');normalized=normalizeKnowledgeForge(payload);
      }
      Object.assign(graphState,normalized);graphState.nodeById=new Map(graphState.nodes.map(node=>[String(node.id),node]));
      const degreeByNode=new Map(graphState.nodes.map(node=>[String(node.id),0]));
      for(const edge of graphState.edges){degreeByNode.set(edge.source_id,(degreeByNode.get(edge.source_id)||0)+1);degreeByNode.set(edge.target_id,(degreeByNode.get(edge.target_id)||0)+1)}
      graphState.nodes=graphState.nodes.map(node=>({...node,degree:degreeByNode.get(String(node.id))||0}));
      graphState.nodeById=new Map(graphState.nodes.map(node=>[String(node.id),node]));
      const select=q('[data-graph-group-filter]'),defaultText=graphState.communityReady?'全部社区':'社区尚未生成（按实体类别）';
      select.replaceChildren(new Option(defaultText,''),...graphState.groups.map(group=>new Option(groupLabel(group),String(group.key))));
      q('[data-graph-status]').hidden=true;q('.embedded-graph-viewer').hidden=false;renderAll();
    }catch(error){q('[data-graph-status]').innerHTML='<div class="empty-state failure">图谱加载失败：'+esc(error.message||error)+'</div>'}
  }
  graphRoot.querySelectorAll('[data-graph-tab]').forEach(button=>button.addEventListener('click',()=>setTab(button.dataset.graphTab)));
  q('[data-apply-graph-filter]').addEventListener('click',()=>{graphState.query=q('[data-graph-filter]').value;clearSelection();renderAll()});
  q('[data-graph-filter]').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();q('[data-apply-graph-filter]').click()}});
  q('[data-graph-group-filter]').addEventListener('change',event=>{graphState.group=event.target.value;clearSelection();renderAll()});
  q('[data-toggle-graph-isolates]').addEventListener('change',event=>{graphState.showIsolates=event.target.checked;clearSelection();renderAll()});
  q('[data-graph-zoom-out]').addEventListener('click',()=>setZoom(graphState.zoom/1.25));
  q('[data-graph-zoom-in]').addEventListener('click',()=>setZoom(graphState.zoom*1.25));
  q('[data-graph-fit]').addEventListener('click',()=>setZoom(1));
  q('[data-reset-graph]').addEventListener('click',()=>{graphState.query='';graphState.group='';graphState.showIsolates=true;graphState.zoom=1;clearSelection();q('[data-graph-filter]').value='';q('[data-graph-group-filter]').value='';q('[data-toggle-graph-isolates]').checked=true;renderAll()});
  q('[data-graph-network]').addEventListener('click',()=>{clearSelection();renderGraph();renderDetails()});
  loadEmbeddedGraph();
}
}
"""


if __name__ == "__main__":
    serve()
