"""Dependency-light management UI for Knowledge Forge."""

from __future__ import annotations

import cgi
import html
import json
import logging
import mimetypes
import re
import time
import threading
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from .config import Config
from .chunking import CHUNK_STRATEGY_OPTIONS, PARSER_OPTIONS, docling_available, validate_processing_request
from .construction_strategies import (
    KNOWLEDGE_FORGE_STRATEGY,
    GRAPHRAG_NATIVE_STRATEGY,
    STABLE_STRATEGY,
    normalize_strategy,
    strategy_options,
)
from .graph_policy import GraphNotReadyError, is_current_graph
from .service import KnowledgeForgeService, TreatmentMaterializationRequired
from .database import JobPublicationCommitted
from .graphrag_native_runtime import GraphRAGNativeStale, GraphRAGNativeUnavailable
from .runtime_safety import validate_runtime_config
from .knowledge_forge_descriptions import (
    is_open_relation_carrier,
    relation_type_label,
)
from .knowledge_language import (
    CHINESE,
    DOCUMENT_LANGUAGE,
    KNOWLEDGE_LANGUAGE_OPTIONS,
    normalize_knowledge_language,
)
from .agent.authorization import new_session_key, validate_session_key


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


def _available_models() -> list[dict]:
    """Read deployment-appropriate models while keeping old service doubles usable."""
    getter = getattr(SERVICE, "available_models", None)
    return getter() if callable(getter) else SERVICE.db.list_models()


def _named_options(options: dict[str, str], selected: str = "", *, include_auto: bool = True) -> str:
    return "".join(
        f'<option value="{_esc(key)}"{" selected" if key == selected else ""}>{_esc(label)}</option>'
        for key, label in options.items() if include_auto or key != "auto"
    )


def _processing_method_label(item: dict, *, include_chunking: bool) -> str:
    """Present persisted parser/chunk keys using the same labels as settings."""

    parser = str(item.get("effective_parser") or "").strip()
    chunking = str(item.get("effective_chunk_strategy") or "").strip()
    labels = [PARSER_OPTIONS.get(parser, parser)] if parser else []
    if include_chunking and chunking:
        labels.append(CHUNK_STRATEGY_OPTIONS.get(chunking, chunking))
    return " · ".join(label for label in labels if label)


def _knowledge_language_options(selected: str = CHINESE) -> str:
    selected = normalize_knowledge_language(selected, default=CHINESE)
    return _named_options(KNOWLEDGE_LANGUAGE_OPTIONS, selected)


def _notice(kind: str, text: str) -> str:
    if not text: return ""
    good = kind == "ok"
    return f'<div class="notice {"success" if good else "error"}"><b>{"✓" if good else "!"}</b>{_esc(text)}</div>'


STATUS_NAMES = {
    "uploaded": "待构建", "queued": "排队中", "processing": "处理中",
    "indexed": "已入库", "failed": "失败", "cancelled": "已终止", "deleted": "已删除",
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
        "preparation_title": "生成结构块、知识块、向量索引和知识结构定义，不抽取实体关系，也不更新知识图谱",
        "build_label": "构建知识图谱",
        "build_ready_title": "基于已完成预处理的文档抽取实体关系并构建知识图谱",
        "build_not_ready_title": "请先预处理文档，完成结构块、知识块、向量和知识结构定义后再构建知识",
        "job_log_label": "查看图谱构建日志",
        "viewer_label": "查看知识图谱",
        "task_title": "知识库构建",
        "task_profile": "entity_graph",
        "task_labels": {
            "structure": "解析文档结构",
            "chunking": "生成知识块与向量",
            "schema_base": "确认知识结构",
            "extraction": "识别实体与关系",
            "graph_sync": "发布知识图谱",
        },
        "task_descriptions": {
            "structure": "读取源文档并恢复标题、正文、表格等结构。",
            "chunking": "生成可检索知识块并写入向量索引。",
            "schema_base": "确认知识库允许的实体类别与关系规则。",
            "extraction": "从知识块识别实体、关系及其原文证据。",
            "graph_sync": "发布当前知识库可浏览的知识图谱。",
        },
        "stat_labels": {"documents": "文档", "chunks": "知识块", "entities": "实体", "relations": "关系"},
        "prepared_stage": "已完成预处理，等待知识构建",
        "pending_stage": "等待文档预处理与知识构建",
        "chunk_action_label": "知识块",
        "show_schema_controls": True,
    },
    KNOWLEDGE_FORGE_STRATEGY: {
        "workspace_title": "Knowledge Forge 构建",
        "workspace_description": "复用已准备的文档和知识结构，完成实体归一、关系核验并发布可追溯的知识图谱。",
        "preparation_workspace_title": "Knowledge Forge 文档预处理",
        "preparation_workspace_description": "解析源文档，生成结构块、知识块、向量索引和知识结构，供后续图谱构建复用。",
        "preparation_navigation_label": "文档预处理",
        "preparation_navigation_hint": "解析、分块与过程产物",
        "query_navigation_label": "图谱与查询",
        "query_navigation_hint": "构建、浏览与检索",
        "preparation_label": "预处理文档",
        "preparation_log_label": "查看预处理日志",
        "preparation_title": "生成供后续图谱构建复用的结构块、知识块、向量索引和知识结构",
        "build_label": "构建知识图谱",
        "build_ready_title": "基于已完成预处理的结果抽取实体关系并构建知识图谱",
        "build_not_ready_title": "请先预处理全部文档，完成可复用的结构块、知识块、向量和知识结构后再构建图谱",
        "job_log_label": "查看图谱构建日志",
        "viewer_label": "查看知识图谱",
        "task_title": "Knowledge Forge 构建",
        "task_profile": "entity_graph",
        "task_labels": {
            "structure": "复用文档结构",
            "chunking": "复用知识块与向量",
            "schema_base": "确认知识结构",
            "extraction": "抽取并核验关系",
            "graph_sync": "发布知识图谱",
        },
        "task_descriptions": {
            "structure": "读取或复用文档结构，确保后续处理使用同一版本。",
            "chunking": "生成可检索知识块和向量；已有结果会直接复用。",
            "schema_base": "确认实体类别、关系语义、方向和发布规则。",
            "extraction": "识别实体并归一别名，生成候选关系后逐条核验原文证据。",
            "graph_sync": "把规范实体和通过验证的关系发布到当前知识库图谱。",
        },
        "stat_labels": {"documents": "文档", "chunks": "知识块", "entities": "规范实体", "relations": "已验证关系"},
        "prepared_stage": "预处理已就绪，等待图谱构建",
        "pending_stage": "等待文档预处理与图谱构建",
        "chunk_action_label": "知识块",
        "show_schema_controls": True,
    },
    GRAPHRAG_NATIVE_STRATEGY: {
        "workspace_title": "GraphRAG Native 知识图谱构建",
        "workspace_description": "从已准备的文档中识别实体和关系，并生成主题社区与摘要。构建完成后可直接浏览图谱或进行知识问答。",
        "preparation_workspace_title": "GraphRAG Native 源文档准备",
        "preparation_workspace_description": "解析源文档并固定本次构建使用的结构化版本；文本单元、知识图谱和社区在下一页生成。",
        "preparation_navigation_label": "源文档准备",
        "preparation_navigation_hint": "解析、版本固定与过程产物",
        "query_navigation_label": "语义图谱与问答",
        "query_navigation_hint": "构建、浏览与问答",
        "preparation_label": "准备源文档",
        "preparation_log_label": "查看源文档准备日志",
        "preparation_title": "解析并固定本次使用的结构化源文档；不会生成 Knowledge Forge 知识块、向量索引或知识结构定义。",
        "build_label": "构建知识图谱",
        "build_ready_title": "从已准备的源文档构建实体关系、主题社区和问答索引",
        "build_not_ready_title": "请先准备全部源文档，生成本次构建使用的结构化文档后再构建知识图谱",
        "job_log_label": "查看知识图谱构建日志",
        "viewer_label": "查看知识图谱",
        "task_title": "GraphRAG Native 构建",
        "task_profile": "graphrag_native",
        "task_labels": {
            "structure": "准备构建输入",
            "chunking": "检查 GraphRAG 配置",
            "schema_base": "检查模型服务",
            "extraction": "生成文本单元、知识图谱与社区",
            "graph_sync": "验证并发布知识图谱成果",
        },
        "task_descriptions": {
            "structure": "固定本次构建使用的源文档版本，并创建独立运行目录。",
            "chunking": "检查 GraphRAG 版本、设置和提示模板是否完整。",
            "schema_base": "确认文本生成与向量模型可用且维度一致。",
            "extraction": "运行 GraphRAG 标准流程，生成文本单元、实体关系、社区、报告和向量检索数据。",
            "graph_sync": "校验全部构建成果，通过后发布知识图谱。",
        },
        "stat_labels": {"documents": "文档", "chunks": "文本单元", "entities": "实体", "relations": "关系"},
        "prepared_stage": "等待知识图谱构建",
        "pending_stage": "需要先准备源文档",
        "chunk_action_label": "结构化文档",
        "show_schema_controls": False,
    },
}


def _strategy_ui(strategy_key: str) -> dict[str, object]:
    """Return the complete presentation contract for a known strategy."""

    return STRATEGY_UI[normalize_strategy(strategy_key)]


def _schema_reassessment_available(strategy_key: str, active_job_id: str = "") -> bool:
    """Only an idle Knowledge Forge project owns the logical Schema workflow."""

    return (
        normalize_strategy(strategy_key) == KNOWLEDGE_FORGE_STRATEGY
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
        '<div><span>{label}</span><small>{description}</small>'
        '<small class="task-step-detail" data-subtask-detail hidden></small></div>'
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
      <div class="section-heading"><div><h2>当前知识图谱构建</h2>
        <p>主页面只显示状态和总进度；需要排查时再打开完整日志。</p></div>
        <button class="button secondary" type="button" data-view-job-log data-job="{_esc(job_id)}"{' disabled' if not job_id else ''}>查看完整构建日志</button>
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
        processing_method = _processing_method_label(
            item, include_chunking=strategy_key != GRAPHRAG_NATIVE_STRATEGY,
        )
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
            status_label = "已准备" if stable_ready else "待准备"
        # Strategy-owned process artifacts live on the second workspace page;
        # document management is limited to source-file lifecycle actions.
        process_actions = ""
        rows.append(
            f"""<tr><td data-label="文件"><strong>{_esc(item['original_filename'])}</strong></td>
            <td data-label="大小">{item['file_size_bytes']/1024:.1f} KB</td><td data-label="状态">
              <span class="status status-{_esc(status)}" data-document-status="{document_id}">{_esc(status_label)}</span>
              <small data-stage-label="{document_id}">{_esc(stage)}</small>
              {f'<small>{_esc(processing_method)}</small>' if processing_method else ''}
              {f'<small class="failure">{_esc(error)}</small>' if error else ''}
            </td><td data-label="更新时间">{_time(item.get('strategy_updated_at') or item['updated_at'])}</td>
            <td class="actions" data-label="操作">{process_actions}
              <a class="button small secondary" href="/api/documents/{document_id}/download">下载</a>
              <button class="icon-button danger" title="删除文档" data-delete="{document_id}">×</button>
            </td></tr>"""
        )
    return "".join(rows) or '<tr class="document-empty-row"><td colspan="5" class="empty">该知识库尚未上传文档。</td></tr>'


def _schema_item(item: object) -> dict[str, object]:
    """Normalize legacy string entries and current metadata-rich definitions."""

    if isinstance(item, dict):
        return item
    return {"name": str(item or ""), "description": ""}


SCHEMA_NAME_ZH = {
    "GenericEntity": "通用实体",
    "Person": "人物",
    "Organization": "组织机构",
    "Location": "地点",
    "Product": "产品或材料",
    "Technology": "技术或方法",
    "Event": "事件",
    "related_to": "相关",
    "part_of": "组成部分",
    "belongs_to": "隶属或归属",
    "located_in": "位于",
    "uses": "使用",
    "produces": "生产或创建",
    "depends_on": "依赖",
    "causes": "导致",
    "executes_task": "执行任务",
    "supply_to": "供应给",
    "supports": "支持",
    "manages": "管理",
    "transports": "运输",
    "member_of": "成员属于",
    "represents": "代表",
}


def _schema_name_zh(value: object) -> str:
    text = str(value or "").strip()
    if text in SCHEMA_NAME_ZH:
        return SCHEMA_NAME_ZH[text]
    # Machine identifiers may be retained in Schema/Policy for stable
    # fingerprints, but they must never leak into the Chinese presentation.
    # Preserve explicitly Chinese custom labels; otherwise use a neutral UI
    # label instead of exposing snake_case or English.
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text):
        return text
    return "自定义实体类别" if text else "未命名类别"


def _schema_relation_name_zh(item: dict[str, object]) -> str:
    """Return a Chinese relation label without exposing machine identifiers."""

    if is_open_relation_carrier(item):
        return "原文直接关系"
    name = str(item.get("name") or "").strip()
    if name in SCHEMA_NAME_ZH:
        return SCHEMA_NAME_ZH[name]
    explicit = str(
        item.get("display_name_zh") or item.get("label_zh") or ""
    ).strip()
    if explicit:
        return explicit
    semantic = str(
        item.get("semantic_definition_zh")
        or item.get("semantic_definition")
        or item.get("description_zh")
        or item.get("description")
        or ""
    ).strip()
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", semantic):
        return semantic.split("。", 1)[0][:18] or relation_type_label(name)
    return relation_type_label(name)


def _schema_relation_semantics_zh(item: dict[str, object]) -> str:
    """Describe the generic source-grounded fallback in product language."""

    if is_open_relation_carrier(item):
        return (
            "保留原文中明确表达、但尚未归入更具体标准关系的有向事实。"
            "图谱边优先显示原文实际谓词；只有精确端点、连续证据和方向核验均通过时才会发布。"
        )
    return _schema_ui_zh(
        item.get("semantic_definition_zh")
        or item.get("semantic_definition")
        or item.get("description"),
        "暂无关系说明",
    )


def _schema_relation_direction_zh(item: dict[str, object]) -> str:
    """Return a user-facing direction rule for every logical relation."""

    if is_open_relation_carrier(item):
        return "从原文实际谓词的完整主体指向其直接客体，严格保持原文论元顺序。"
    return _schema_ui_zh(
        item.get("direction_description_zh")
        or item.get("direction_description"),
        "未设置",
    )


def _schema_type_names(value: object) -> str:
    values = value if isinstance(value, list) else []
    names = [str(item) for item in values if str(item).strip()]
    return (
        "任意类型" if not names or names == ["*"]
        else "、".join(_schema_name_zh(item) for item in names)
    )


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
          <strong>{_esc(_schema_name_zh(item.get("name")))}</strong></div>
          <p>{_esc(_schema_ui_zh(item.get("description_zh") or item.get("description"), "暂无类别说明"))}</p></article>"""
        for item in entities
    ) or '<div class="empty-state compact-empty">当前版本未定义实体类别。</div>'
    publish_labels = {"auto": "自动发布", "review_only": "仅供审核", "disabled": "停用"}
    relation_cards = "".join(
        f"""<article class="definition-card relation-definition"><div class="definition-card-head">
          <strong>{_esc(_schema_relation_name_zh(item))}</strong>
          <span>{_esc(publish_labels.get(str(item.get("publish_mode") or ""), item.get("publish_mode") or "发布规则未设置"))}</span></div>
          <p>{_esc(_schema_relation_semantics_zh(item))}</p>
          <dl><div><dt>允许方向</dt><dd>{_esc(_schema_type_names(item.get("domain_types") or item.get("source_types")))}
            → {_esc(_schema_type_names(item.get("range_types") or item.get("target_types")))}</dd></div>
          <div><dt>方向说明</dt><dd>{_esc(_schema_relation_direction_zh(item))}</dd></div>
          <div><dt>结构属性</dt><dd>{'对称关系' if bool(item.get("symmetric")) else '有向关系'}</dd></div></dl></article>"""
        for item in relations
    ) or '<div class="empty-state compact-empty">当前版本未定义关系类别。</div>'
    return f"""<div class="workspace-section knowledge-structure">
      <div class="section-heading"><div>
        <div class="knowledge-structure-title"><h2>知识结构定义</h2>
          <span class="schema-version-badge">当前版本 v{_esc(schema.get("version") or "")}</span></div>
        <p>定义知识库允许识别的实体类别、关系语义、连接范围与方向；构建时据此约束抽取和验证。</p></div>
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
        processing_method = _processing_method_label(item, include_chunking=True)
        rows.append(
            f"""<tr><td data-label="文档"><strong>{_esc(item['original_filename'])}</strong>
              <small>{_esc(processing_method or '等待预处理')}</small></td>
            <td data-label="结构解析"><b class="artifact-count">{block_count}</b><small>结构块</small></td>
            <td data-label="知识分块"><b class="artifact-count">{chunk_count}</b><small>知识块</small></td>
            <td class="artifact-actions" data-label="查看列表">
              <button class="button small secondary" type="button" data-view-structure="{document_id}"
                {"disabled" if not item.get("structure_ready_at") else ""}>查看结构块列表</button>
              <button class="button small secondary" type="button" data-view-chunks="{document_id}"
                {"disabled" if not item.get("chunks_ready_at") else ""}>查看知识块列表</button>
            </td></tr>"""
        )
    body = "".join(rows) or '<tr class="artifact-empty-row"><td colspan="4" class="empty">上传并预处理文档后，这里会集中展示结构块和知识块。</td></tr>'
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
        parser = _processing_method_label(item, include_chunking=False)
        ready = bool(item.get("structure_ready_at"))
        rows.append(
            f"""<tr><td data-label="文档"><strong>{_esc(item['original_filename'])}</strong>
              <small>{_esc(parser or '等待准备')}</small></td>
            <td data-label="结构解析"><b class="artifact-count">{block_count}</b><small>源结构块</small></td>
            <td data-label="准备状态"><span class="status {'status-indexed' if ready else ''}">{'已准备' if ready else '待准备'}</span></td>
            <td class="artifact-actions" data-label="查看过程文件">
              <button class="button small secondary" type="button" data-view-structure="{document_id}"
                {"disabled" if not ready else ""}>查看源结构</button>
              <button class="button small secondary" type="button" data-view-native-documents="{document_id}"
                {"disabled" if not ready else ""}>查看结构化文档</button>
            </td></tr>"""
        )
    body = "".join(rows) or '<tr class="artifact-empty-row"><td colspan="4" class="empty">上传并准备源文档后，这里会展示 GraphRAG 本次使用的构建输入。</td></tr>'
    return f"""<div class="workspace-section process-artifacts native-process-artifacts">
      <div class="section-heading"><div><h2>源文档准备产物</h2>
        <p>集中查看源结构与本次构建使用的结构化文档；文本单元和图谱产物在“语义图谱与问答”页查看。</p></div></div>
      <div class="table-wrap"><table class="artifact-table"><thead><tr><th>文档与解析方式</th>
        <th>结构解析</th><th>准备状态</th><th>查看过程文件</th></tr></thead><tbody>{body}</tbody></table></div>
    </div>"""


def _documents_ready_for_knowledge_build(
    documents: list[dict], strategy_key: str,
) -> bool:
    return bool(documents) and all(
        bool(item.get("active_revision_id")) and bool(item.get("structure_ready_at"))
        and (
            strategy_key != KNOWLEDGE_FORGE_STRATEGY
            or bool(item.get("chunks_ready_at"))
        )
        and str(item.get("status") or "") == "indexed"
        for item in documents
    )


def _fallback_preparation_status(
    documents: list[dict], strategy_key: str,
) -> dict[str, Any]:
    """Compatibility plan for narrow UI adapters without the service method."""

    pending_documents = []
    for document in documents:
        new_document = not bool(document.get("active_revision_id"))
        artifacts_missing = (
            not bool(document.get("structure_ready_at"))
            or (
                strategy_key == KNOWLEDGE_FORGE_STRATEGY
                and not bool(document.get("chunks_ready_at"))
            )
        )
        if new_document or artifacts_missing:
            pending_documents.append({
                "id": document.get("id"),
                "original_filename": document.get("original_filename") or "",
                "reason": "new_document" if new_document else "artifacts_missing",
            })
    new_count = sum(
        1 for item in pending_documents if item["reason"] == "new_document"
    )
    return {
        "document_count": len(documents),
        "ready_count": len(documents) - len(pending_documents),
        "pending_count": len(pending_documents),
        "new_count": new_count,
        "refresh_count": len(pending_documents) - new_count,
        "pending_documents": pending_documents,
    }


def _preparation_update_notice(
    preparation_status: dict[str, Any],
    *,
    strategy_key: str,
    active_job_id: str = "",
    active_job_kind: str = "",
) -> str:
    pending_count = int(preparation_status.get("pending_count") or 0)
    if pending_count <= 0:
        return ""
    new_count = min(
        pending_count, max(0, int(preparation_status.get("new_count") or 0)),
    )
    refresh_count = pending_count - new_count
    process_name = (
        "源文档准备"
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY else "预处理"
    )
    if new_count == pending_count:
        summary = f"有 {pending_count} 份新文档等待{process_name}"
        detail = ""
    elif refresh_count == pending_count:
        summary = f"有 {pending_count} 份文档的{process_name}结果需要更新"
        detail = "只重新处理内容、配置或产物已变化的文档"
    else:
        summary = f"有 {pending_count} 份文档等待{process_name}"
        detail = f"新文档 {new_count} 份，需要更新 {refresh_count} 份"
    state = ""
    if active_job_id and active_job_kind == "content_preparation":
        detail = f"正在处理这 {pending_count} 份文档"
        state = f'<span class="graph-update-state">{_esc(process_name)}中</span>'
    elif active_job_id:
        detail = "当前任务结束后可开始处理"
    pending_documents = preparation_status.get("pending_documents") or []
    filenames = "、".join(
        str(item.get("original_filename") or f"文档 {item.get('id')}")
        for item in pending_documents[:5] if isinstance(item, dict)
    )
    if len(pending_documents) > 5:
        filenames += f"等 {pending_count} 份文档"
    detail_markup = f"<small>{_esc(detail)}</small>" if detail else ""
    return f"""<div class="graph-update-notice preparation-update-notice"
      role="status" title="{_esc(filenames)}">
      <span class="graph-update-icon" aria-hidden="true">!</span>
      <span class="graph-update-copy"><b>{_esc(summary)}</b>{detail_markup}</span>
      {state}
    </div>"""


def _knowledge_forge_pending_graph_documents(
    documents: list[dict], graph_state: dict | None,
) -> list[dict]:
    """Return documents whose current revision is absent from the published graph.

    Uploading or re-preparing a document intentionally leaves the last complete
    project graph available. The strategy build row is therefore the durable
    inclusion marker; timestamps alone cannot distinguish a harmless metadata
    edit from a new document revision.
    """

    if int((graph_state or {}).get("ready_generation") or 0) <= 0:
        return []
    pending: list[dict] = []
    for document in documents:
        if document.get("is_deleted"):
            continue
        strategy_status = str(
            document.get("strategy_status") or document.get("status") or "uploaded"
        )
        active_revision_id = str(document.get("active_revision_id") or "")
        # list_documents(..., strategy_key) always supplies this key. The key
        # check keeps older presentation test doubles and third-party adapters
        # backwards compatible while production remains revision-exact.
        revision_mismatch = (
            "strategy_revision_id" in document
            and str(document.get("strategy_revision_id") or "")
            != active_revision_id
        )
        if strategy_status != "indexed" or revision_mismatch:
            pending.append(document)
    return pending


def _knowledge_forge_graph_update_notice(
    documents: list[dict],
    graph_state: dict | None,
    *,
    build_ready: bool,
    active_job_id: str = "",
    active_job_kind: str = "",
) -> str:
    pending = _knowledge_forge_pending_graph_documents(documents, graph_state)
    if not pending:
        return ""
    count = len(pending)
    filenames = "、".join(
        str(item.get("original_filename") or f"文档 {item.get('id')}")
        for item in pending[:5]
    )
    if len(pending) > 5:
        filenames += f"等 {count} 份文档"
    summary = f"有 {count} 份文档尚未纳入当前知识图谱"
    if active_job_id and active_job_kind == "ingestion":
        detail = "正在生成包含这些文档的新图谱"
        action = '<span class="graph-update-state">构建中</span>'
    elif active_job_id and active_job_kind == "content_preparation":
        detail = "正在预处理，完成后可重新构建"
        action = '<span class="graph-update-state">预处理中</span>'
    elif active_job_id:
        detail = "当前任务结束后可重新构建"
        action = ""
    elif build_ready:
        detail = "重新构建后生效"
        action = (
            '<button class="graph-update-action" type="button" '
            'data-graph-update-action="build">重新构建</button>'
        )
    else:
        detail = "请先完成文档预处理，再重新构建"
        action = (
            '<button class="graph-update-action" type="button" '
            'data-graph-update-action="prepare">去预处理</button>'
        )
    return f"""<div class="graph-update-notice" role="status" title="{_esc(filenames)}">
      <span class="graph-update-icon" aria-hidden="true">!</span>
      <span class="graph-update-copy"><b>{_esc(summary)}</b><small>{_esc(detail)}</small></span>
      {action}
    </div>"""


def _construction_controls(
    *,
    project_id: int,
    strategy_key: str,
    strategy_select: str = "",
    latest_job_id: str,
    latest_batch_id: str,
    active_job_id: str,
    active_job_kind: str = "",
    active_batch_id: str = "",
    preparation_pending: bool = True,
    build_ready: bool = True,
    scope: str = "all",
    after_build: str = "",
) -> str:
    ui = _strategy_ui(strategy_key)
    if scope not in {"all", "preparation", "graph"}:
        raise ValueError(f"unknown construction control scope: {scope}")
    preparation_disabled = (
        " disabled" if active_job_id or not preparation_pending else ""
    )
    build_disabled = " disabled" if active_job_id or not build_ready else ""
    build_title = str(ui["build_ready_title"] if build_ready else ui["build_not_ready_title"])
    preparation_controls = ""
    if scope in {"all", "preparation"}:
        preparation_active = (
            active_job_kind == "content_preparation" and bool(active_job_id)
        )
        preparation_controls = f"""
      <button class="button primary" id="prepare-project-documents" data-prepare-project="{_esc(project_id)}" type="button"
        title="{_esc(ui['preparation_title'] if preparation_pending else '当前没有待处理文档')}"{preparation_disabled}>{_esc(ui['preparation_label'])}</button>
      <button class="button safe-stop" type="button" data-cancel-project-job
        data-cancel-scope="preparation" data-job="{_esc(active_job_id if preparation_active else '')}"
        data-batch="{_esc(active_batch_id if preparation_active else '')}"
        data-kind="content_preparation"{' ' if preparation_active else ' hidden'}>终止预处理</button>
      <button class="button secondary" id="view-preparation-log" data-view-job-log data-job="{_esc(latest_job_id)}"
        data-batch="{_esc(latest_batch_id)}"{' disabled' if not latest_job_id else ''}>{_esc(ui['preparation_log_label'])}</button>"""
    graph_controls = ""
    if scope in {"all", "graph"}:
        graph_active = (
            active_job_kind in {"ingestion", "graphrag_native_build", "graph_sync"}
            and bool(active_job_id)
        )
        graph_controls = f"""
      <button class="button primary" id="build-project-knowledge" data-build-project="{_esc(project_id)}" type="button"
        title="{_esc(build_title)}"{build_disabled}>{_esc(ui['build_label'])}</button>
      <button class="button safe-stop" type="button" data-cancel-project-job
        data-cancel-scope="graph" data-job="{_esc(active_job_id if graph_active else '')}"
        data-batch="{_esc(active_batch_id if graph_active else '')}"
        data-kind="{_esc(active_job_kind if graph_active else '')}"{' ' if graph_active else ' hidden'}>终止构建</button>
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
    """Common knowledge-base creation/selection page; workspaces are strategy-owned."""

    projects, models = SERVICE.db.list_projects(), _available_models()
    cards: list[str] = []
    for project in projects:
        project_id = int(project["id"])
        selection = _project_binding(project_id) or {}
        status = str(selection.get("association_status") or "unassigned")
        strategy_key = str(selection.get("selected_strategy") or STABLE_STRATEGY)
        if status == "bound" and strategy_key in {
            KNOWLEDGE_FORGE_STRATEGY, GRAPHRAG_NATIVE_STRATEGY,
        }:
            action = (
                f'<a class="button primary" href="{_workspace_path(project_id, strategy_key)}">'
                '打开知识库</a>'
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
              <select name="strategy_key" aria-label="选择 {_esc(project['name'])} 的知识构建方式" required>{choices}</select>
              <button class="button primary" type="submit">选择并打开</button>
            </form>"""
            badge = "待选择构建方式"
        cards.append(f"""<article class="project-card" data-project-card
          data-project-search="{_esc(' '.join((str(project.get('name') or ''), str(project.get('description') or ''), badge)))}"><div><h3>{_esc(project['name'])}</h3>
          <p>{_esc(project.get('description') or '暂无知识库说明')}</p>
          <span class="strategy-badge">{badge}</span></div>{action}</article>""")
    choices = "".join(
        f'<option value="{_esc(item.key)}">{_esc(item.label)}</option>'
        for item in strategy_options(include_hidden=False)
    )
    notice = _notice(*message) if message else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>Knowledge Forge · 通用知识库与知识图谱</title><style>{CSS}</style></head><body>
      <main class="shell home-shell"><header class="home-header"><div><h1>Knowledge Forge</h1>
      <p>通用知识库与知识图谱构建平台</p></div>
      <nav class="home-primary-actions" aria-label="主要功能">
        <button class="button primary" type="button" id="open-project-list">知识库</button>
        <a class="button secondary" href="/chat">智能问答</a>
      </nav></header>
      {notice}
      <section class="panel home-hero" aria-labelledby="home-introduction-title">
        <div class="hero-layout"><div class="hero-copy">
          <span class="hero-kicker">从文档建设知识，再让知识参与回答</span>
          <h2 id="home-introduction-title">把分散文档建设成可检索、可连接、可追溯的知识库</h2>
          <p>导入 PDF、Word、表格、Markdown 和文本资料，系统保留章节、页码、图表和来源，
          再完成知识块索引、实体关系组织与知识图谱构建。</p>
          <p>多个知识库在同一入口中独立管理；需要使用知识时，可进入智能问答工作台选择相应知识范围。</p>
          <div class="hero-actions">
            <button class="button primary" type="button" data-open-knowledge-bases>管理知识库</button>
            <a class="button secondary" href="/chat">进入智能问答</a>
          </div>
        </div>
        <div class="hero-points" aria-label="平台能力">
          <div><b>多格式解析</b><span>保留章节、表格、页码和来源结构</span></div>
          <div><b>结构化知识</b><span>组织知识块、实体、关系、社区与图谱</span></div>
          <div><b>原文可追溯</b><span>检索结果和图谱事实均可回看来源</span></div>
          <div><b>私有化部署</b><span>模型、向量与图数据保留在自有环境</span></div>
        </div></div>
        <ol class="home-flow" aria-label="知识构建流程">
          <li><b>1</b><span><strong>导入文档</strong><small>集中管理多格式资料</small></span></li>
          <li><b>2</b><span><strong>文档预处理</strong><small>解析结构并建立向量索引</small></span></li>
          <li><b>3</b><span><strong>构建知识库</strong><small>形成实体、关系与主题网络</small></span></li>
          <li><b>4</b><span><strong>检索与问答</strong><small>依据原文和图谱使用知识</small></span></li>
        </ol>
      </section>
      <section class="panel product-overview" aria-labelledby="product-overview-title">
        <div class="section-heading"><div><span class="section-kicker">两项核心能力</span>
          <h2 id="product-overview-title">建设知识，也使用知识</h2>
          <p>知识库负责沉淀和治理资料，智能问答负责在授权范围内检索、组合并引用这些知识。</p>
        </div></div>
        <div class="product-entry-grid">
          <article class="product-entry-card knowledge-entry"><div class="product-entry-index">01</div><div>
            <div class="product-entry-heading"><h3>知识库</h3></div>
            <p>创建和管理多个知识库，完成文档预处理、向量索引、实体关系抽取、社区报告和图谱浏览。</p>
            <button class="text-action" type="button" data-open-knowledge-bases>查看知识库 <b aria-hidden="true">→</b></button>
          </div></article>
          <article class="product-entry-card chat-product-entry"><div class="product-entry-index">02</div><div>
            <div class="product-entry-heading"><h3>智能问答</h3></div>
            <p>选择一个知识库后提问，系统结合原文与图谱内容生成回答、标注引用，并在依据不足时明确说明。</p>
            <a class="text-action" href="/chat">进入问答工作台 <b aria-hidden="true">→</b></a>
          </div></article>
        </div>
      </section>
      <section class="panel strategy-overview" aria-labelledby="strategy-overview-title">
        <div class="section-heading strategy-heading"><div><span class="section-kicker">新建知识库时选择</span>
          <h2 id="strategy-overview-title">两种构建方式，服务不同的知识目标</h2>
          <p>每个知识库固定使用一种构建方式。选择只影响该知识库的处理、存储和查询链路，不影响其他知识库。</p>
        </div></div>
        <div class="strategy-intro-grid">
          <article class="strategy-explainer strategy-kf">
            <div class="strategy-card-head"><span class="strategy-mark">KF</span><div>
              <h3>Knowledge Forge</h3>
            </div></div>
            <p class="strategy-lead">在整个知识库范围内归一实体、核验关系与方向，并让图谱事实保留可回看的原文依据。</p>
            <ul class="strategy-summary-list">
              <li><b>得到</b><span>规范实体、明确关系、主题社区、详细报告与来源证据</span></li>
              <li><b>适合</b><span>需要细粒度关系、实体去重和关系审计的知识库</span></li>
              <li><b>特点</b><span>质量检查更完整，构建时间通常更长</span></li>
            </ul>
          </article>
          <article class="strategy-explainer strategy-native">
            <div class="strategy-card-head"><span class="strategy-mark">GR</span><div>
              <h3>GraphRAG Native</h3>
            </div></div>
            <p class="strategy-lead">采用 GraphRAG 3.1.1 标准工作流，从文本单元生成语义实体、描述型关系、社区与社区报告。</p>
            <ul class="strategy-summary-list">
              <li><b>得到</b><span>语义网络、Leiden 社区、社区报告与原生查询成果</span></li>
              <li><b>适合</b><span>快速理解跨文档主题、社区结构和整体联系</span></li>
              <li><b>特点</b><span>保留标准 GraphRAG 成果，关系语义更加开放</span></li>
            </ul>
          </article>
        </div>
        <div class="choice-guide"><b>快速判断</b>
          <span>重视实体归一、明确关系和可核对的原文依据，选择 <strong>Knowledge Forge</strong>。</span>
          <span>重视社区发现、全局主题和标准 GraphRAG 查询，选择 <strong>GraphRAG Native</strong>。</span>
          <em>需要同时保留两种结果时，请分别创建两个知识库。</em>
        </div>
      </section>
      <dialog id="project-list-dialog" class="project-list-dialog" aria-labelledby="project-list-title">
      <div class="dialog-heading project-list-heading"><div><span class="section-kicker">知识库</span><h2 id="project-list-title">选择知识库</h2>
      <p>选择已有知识库继续管理，或创建一个绑定知识构建方式的新知识库。</p></div>
      <div class="dialog-title-actions"><button class="button primary" type="button" id="open-create-project">新建知识库</button>
      <button class="icon-button" type="button" data-close-project-list aria-label="关闭知识库列表">×</button></div></div>
      <div class="project-list-tools"><input id="project-list-search" type="search"
        placeholder="按知识库名称、说明或构建方式查找" aria-label="查找知识库"><span id="project-list-count" aria-live="polite"></span></div>
      <div class="project-grid">{''.join(cards) or '<p class="empty">尚无知识库，请先创建。</p>'}</div></dialog>
      <dialog id="create-project-dialog"><div class="dialog-heading"><div><h2>新建知识库</h2>
      <p>请选择最符合当前目标的知识构建方式；知识库创建后不可切换。</p></div>
      <button class="icon-button" type="button" data-close-create-project>×</button></div>
      <form action="/projects" method="post" class="form-grid">
        <label>知识库名称<input name="name" required maxlength="100"></label>
        <label>知识构建方式<select id="create-strategy-key" name="strategy_key" required>{choices}</select></label>
        <p class="wide strategy-choice-help" id="strategy-choice-help"></p>
        <label>知识语言<select name="knowledge_language" required>{_knowledge_language_options(CHINESE)}</select>
          <small class="field-help">控制后续构建生成的说明、关系描述和社区报告；专名、缩写与代码保留原文。</small></label>
        <label>LLM<select name="llm_model">{_model_options(models,'llm',Config.default_llm_model)}</select></label>
        <label>Embedding<select name="embedding_model">{_model_options(models,'embedding')}</select></label>
        <label class="wide">知识库说明<textarea name="description" rows="2"></textarea></label>
        <div class="form-actions dialog-actions">
          <button class="button secondary" type="button" data-close-create-project>取消</button>
          <button class="button primary" type="submit">创建知识库</button></div></form></dialog>
      </main><script>
      const projectDialog=document.querySelector('#project-list-dialog');
      const dialog=document.querySelector('#create-project-dialog');
      const strategySelect=document.querySelector('#create-strategy-key');
      const strategyHelp=document.querySelector('#strategy-choice-help');
      const strategyDescriptions={{
        knowledge_forge:'适合同时需要全局主题脉络、实体归一、受控关系、方向核验和原文证据的领域知识图谱。',
        graphrag_native:'适合快速发现主题网络、社区结构，并使用 GraphRAG 问答和构建成果。'
      }};
      function updateStrategyHelp(){{
        if(strategyHelp)strategyHelp.textContent=strategyDescriptions[strategySelect?.value]||'';
      }}
      const projectSearch=document.querySelector('#project-list-search');
      const projectCount=document.querySelector('#project-list-count');
      function filterProjects(){{
        const query=String(projectSearch?.value||'').trim().toLowerCase();
        let visible=0;
        document.querySelectorAll('[data-project-card]').forEach(card=>{{
          const matches=!query||String(card.dataset.projectSearch||'').toLowerCase().includes(query);
          card.hidden=!matches;if(matches)visible++;
        }});
        if(projectCount)projectCount.textContent=query?'找到 '+visible+' 个知识库':'';
      }}
      document.querySelectorAll('#open-project-list,[data-open-knowledge-bases]').forEach(button=>
        button.addEventListener('click',()=>{{
          projectDialog.showModal();projectSearch?.focus();filterProjects();
        }}));
      if(location.hash==='#knowledge-bases'){{
        projectDialog.showModal();projectSearch?.focus();filterProjects();
      }}
      projectSearch?.addEventListener('input',filterProjects);
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


def render_chat_page() -> str:
    """Top-level single-knowledge-base question answering workspace."""

    template = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>智能问答 · Knowledge Forge</title><style>__CSS__</style></head><body>
      <main class="shell home-shell agent-shell"><header class="home-header"><div>
        <h1>Knowledge Forge</h1><p>通用知识库与知识图谱构建平台</p></div>
        <nav class="home-primary-actions" aria-label="主要功能">
          <a class="button secondary" href="/#knowledge-bases">知识库</a>
          <a class="button primary" href="/chat" aria-current="page">智能问答</a>
        </nav></header>
        <section class="panel agent-workbench" aria-labelledby="agent-title">
          <header class="agent-heading"><div><span class="section-kicker">智能问答</span>
            <h2 id="agent-title">基于当前知识库回答</h2>
            <p>选择一个知识库后即可提问。回答会标注实际引用的来源；依据不足时会直接说明。</p>
          </div></header>
          <section class="agent-scope-bar" aria-label="问答知识范围">
            <label><span>知识库</span><select id="agent-project"><option value="">正在读取知识库…</option></select></label>
            <label id="agent-mode-field" hidden><span>问答方式</span>
              <select id="agent-mode"><option value="auto">自动选择（推荐）</option><option value="basic">原文检索（Basic）</option>
                <option value="local">实体与关系（Local）</option><option value="global">全局主题（Global）</option></select></label>
          </section>
          <div class="agent-layout">
            <aside class="agent-thread-pane" aria-label="历史会话">
              <div class="agent-thread-heading"><div><h3>历史会话</h3>
                <span id="agent-thread-count">选择知识库</span></div>
                <button class="button secondary" id="agent-new-thread" type="button" disabled>新对话</button></div>
              <div class="agent-thread-list" id="agent-thread-list">
                <p class="agent-thread-empty">选择知识库后显示历史会话。</p>
              </div>
            </aside>
            <section class="agent-conversation" aria-label="问答对话">
              <div class="agent-message-list" id="agent-messages" aria-live="polite">
                <div class="agent-empty" id="agent-empty"><span>KF</span><h3>选择知识库后开始提问</h3>
                  <p>回答只使用当前选择的知识库。生成回答后，可在右侧查看引用的原文和图谱依据。</p></div>
              </div>
              <div class="agent-composer">
                <textarea id="agent-query" rows="3" maxlength="4000" disabled placeholder="输入需要基于知识库回答的问题；Shift + Enter 换行"></textarea>
                <div class="agent-composer-footer"><span id="agent-composer-hint">先选择知识库</span>
                  <div><button class="button secondary agent-stop" id="agent-stop" type="button" hidden>停止回答</button>
                    <button class="button primary" id="agent-send" type="button" disabled>发送</button></div></div>
              </div>
            </section>
            <aside class="agent-source-pane" aria-label="本次回答来源">
              <div class="agent-source-heading"><h3>回答依据</h3>
                <span id="agent-source-count">0</span></div>
              <label class="agent-source-turn" id="agent-source-turn-field" hidden>
                <span>查看回答</span><select id="agent-source-turn"></select></label>
              <div class="agent-source-content" id="agent-source-content">
                <p id="agent-source-empty">生成回答后，这里会列出实际使用的原文和图谱依据。</p>
                <div id="agent-sources" class="agent-source-list"></div>
              </div>
            </aside>
          </div>
        </section>
      </main>
      <dialog id="agent-evidence-dialog" class="agent-evidence-dialog"><div class="dialog-heading"><div>
        <h2 id="agent-evidence-title">来源详情</h2><p id="agent-evidence-subtitle"></p></div>
        <button class="icon-button" id="agent-evidence-close" type="button" aria-label="关闭">×</button></div>
        <div class="agent-evidence-body" id="agent-evidence-body"></div></dialog>
      <script>
      (()=>{
        const $=selector=>document.querySelector(selector);
        const state={bases:[],base:null,threads:[],turns:[],threadId:'',selectedTurnId:'',runId:'',stream:null,traceTimer:null,busy:false,loadingThreads:false,loadGeneration:0,sourceSwitchToken:0,lastQuery:'',assistant:null};
        const projectSelect=$('#agent-project'),modeField=$('#agent-mode-field'),modeSelect=$('#agent-mode');
        const messages=$('#agent-messages'),empty=$('#agent-empty'),query=$('#agent-query'),send=$('#agent-send'),stop=$('#agent-stop');
        const threadList=$('#agent-thread-list'),threadCount=$('#agent-thread-count'),newThread=$('#agent-new-thread');
        const hint=$('#agent-composer-hint'),sourcePane=$('.agent-source-pane'),sourceContent=$('#agent-source-content'),sourceList=$('#agent-sources'),sourceEmpty=$('#agent-source-empty'),sourceCount=$('#agent-source-count');
        const sourceTurnField=$('#agent-source-turn-field'),sourceTurn=$('#agent-source-turn');
        const dialog=$('#agent-evidence-dialog'),dialogTitle=$('#agent-evidence-title'),dialogSubtitle=$('#agent-evidence-subtitle'),dialogBody=$('#agent-evidence-body');
        const text=(tag,value,className='')=>{const node=document.createElement(tag);if(className)node.className=className;node.textContent=String(value??'');return node};
        const formatType=value=>({chunk:'原文知识块',entity:'实体依据',relation:'关系依据',community_report:'社区报告',native_context:'GraphRAG 上下文'}[value]||'知识依据');
        const threadKey=id=>'knowledge-forge.agent.thread.'+id;
        const selectedProjectKey='knowledge-forge.agent.project';
        const evidenceKey=id=>'knowledge-forge.agent.evidence.'+id;
        function syncRoute(){const url=new URL(location.href);if(state.base?.id)url.searchParams.set('project',String(state.base.id));else url.searchParams.delete('project');if(state.threadId)url.searchParams.set('thread',state.threadId);else url.searchParams.delete('thread');history.replaceState(null,'',url.pathname+url.search)}
        function setBusy(value){state.busy=Boolean(value);const locked=state.busy||state.loadingThreads;send.disabled=locked||!state.base?.queryable||!query.value.trim();stop.hidden=!state.busy;projectSelect.disabled=locked;modeSelect.disabled=locked;query.disabled=locked||!state.base?.queryable;newThread.disabled=locked||!state.base?.queryable;threadList.querySelectorAll('button').forEach(button=>button.disabled=locked||button.dataset.blocked==='true');hint.textContent=state.busy?'正在检索并核验依据…':state.loadingThreads?'正在读取历史会话…':state.base?.queryable?'回答只使用当前所选知识库':'先选择可问答的知识库'}
        function resetSourceTransition(){state.sourceSwitchToken+=1;sourceContent.getAnimations().forEach(animation=>animation.cancel());for(const name of ['height','overflow','opacity','transform','will-change'])sourceContent.style.removeProperty(name)}
        function resetTurns(){state.turns=[];state.selectedTurnId='';sourceTurn.replaceChildren();sourceTurnField.hidden=true;resetSourceTransition();renderSources([],'生成回答后，这里会列出实际使用的原文和图谱依据。')}
        function stopTracePolling(){if(state.traceTimer){clearInterval(state.traceTimer);state.traceTimer=null}}
        function clearConversation(){stopTracePolling();messages.replaceChildren(empty);empty.hidden=Boolean(state.base?.queryable);state.assistant=null;resetTurns()}
        function createExecutionTrace(runId='',live=false){
          const root=document.createElement('details');root.className='agent-execution-trace';root.dataset.runId=runId;root.open=Boolean(live);
          const summary=document.createElement('summary'),label=text('span','执行过程','agent-trace-summary-label'),status=text('span',live?'等待开始':'展开查看','agent-trace-summary-status');summary.append(label,status);
          const body=document.createElement('div');body.className='agent-trace-body';body.append(text('p',live?'正在记录本轮处理步骤与工具调用…':'展开后读取本轮处理步骤与工具调用。','agent-trace-placeholder'));
          root.append(summary,body);const view={root,summary,label,status,body,data:null,loading:false,loaded:false};
          root.addEventListener('toggle',()=>{if(!root.open)return;const turn=state.turns.find(item=>item.runId===root.dataset.runId);if(turn)refreshExecutionTrace(turn)});
          return view;
        }
        function addMessage(role,content,answerData=null,actionContext={}){empty.hidden=true;const card=document.createElement('article');card.className='agent-message '+role;const roleLabel=role==='user'?'你':actionContext.turnNumber?'Knowledge Forge · 第 '+actionContext.turnNumber+' 轮':'Knowledge Forge';card.append(text('span',roleLabel,'agent-message-role'));let trace=null;if(role==='assistant'){trace=createExecutionTrace(actionContext.runId||'',Boolean(actionContext.liveTrace));card.append(trace.root)}const body=text('div',content,'agent-message-body');card.append(body);if(role==='assistant'){const actions=document.createElement('div');actions.className='agent-message-actions';const copy=text('button','复制');copy.type='button';copy.addEventListener('click',async()=>{await navigator.clipboard.writeText(body.textContent||'');copy.textContent='已复制';setTimeout(()=>copy.textContent='复制',1200)});const retry=text('button','重新检索');retry.type='button';retry.addEventListener('click',()=>{query.value=actionContext.query||state.lastQuery||'';submitQuestion()});const up=text('button','有帮助');up.type='button';up.addEventListener('click',()=>sendFeedback(1,up,actionContext.runId));const down=text('button','需改进');down.type='button';down.addEventListener('click',()=>sendFeedback(-1,down,actionContext.runId));actions.append(copy,retry,up,down);card.append(actions);if(answerData?.source_cards)card.dataset.sources=JSON.stringify(answerData.source_cards)}messages.append(card);messages.scrollTop=messages.scrollHeight;return {card,body,trace}}
        async function sendFeedback(rating,button,runId=''){const target=runId||state.runId;if(!target)return;try{const response=await fetch('/api/agent/runs/'+target+'/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rating})});if(!response.ok)throw Error();button.textContent='已记录';button.disabled=true}catch(_error){button.textContent='提交失败'}}
        function formatTraceDuration(value){const milliseconds=Math.max(0,Number(value)||0);if(!milliseconds)return '';if(milliseconds<1000)return milliseconds+' ms';const seconds=milliseconds/1000;if(seconds<60)return (seconds<10?seconds.toFixed(1):Math.round(seconds))+' 秒';const minutes=Math.floor(seconds/60),rest=Math.round(seconds%60);return minutes+' 分'+(rest?rest+' 秒':'')}
        function traceStatusLabel(value){return ({queued:'等待开始',running:'执行中',completed:'已完成',degraded:'已降级完成',failed:'失败',cancelled:'已停止'}[value]||'处理中')}
        function appendTraceFacts(target,items){const facts=Array.isArray(items)?items.filter(item=>item?.label&&item?.value):[];if(!facts.length)return false;const list=document.createElement('dl');list.className='agent-trace-facts';for(const item of facts){list.append(text('dt',item.label),text('dd',item.value))}target.append(list);return true}
        function appendTraceEvidence(target,title,items){const evidence=Array.isArray(items)?items.filter(item=>item?.evidence_id):[];if(!evidence.length)return false;const section=document.createElement('section');section.className='agent-trace-evidence';section.append(text('h5',title||'本步使用的具体证据'));const list=document.createElement('div');for(const item of evidence){const card=document.createElement('button');card.type='button';card.className='agent-trace-evidence-card';const head=document.createElement('span');head.className='agent-trace-evidence-head';const marker=Number(item.citation_index||0)?'〔'+item.citation_index+'〕 ':'';head.append(text('b',marker+(item.title||'知识依据')),text('small',(item.type_label||formatType(item.evidence_type))+(item.cited?' · 最终引用':' · 生成依据')));card.append(head);if(item.location)card.append(text('span',item.location,'agent-trace-evidence-location'));if(item.excerpt)card.append(text('p',item.excerpt));card.addEventListener('click',event=>{event.stopPropagation();openEvidence(item.evidence_id)});list.append(card)}section.append(list);target.append(section);return true}
        function renderExecutionTrace(view,data){
          if(!view||!data)return;view.data=data;view.loaded=true;view.root.classList.toggle('is-running',['queued','running'].includes(data.status));view.root.classList.toggle('is-failed',data.status==='failed');
          const counts=[];if(Number(data.step_count||0))counts.push(data.step_count+' 个步骤');if(Number(data.tool_count||0))counts.push(data.tool_count+' 次工具调用');const duration=formatTraceDuration(data.duration_ms);
          if(['queued','running'].includes(data.status)){view.status.textContent=(data.current_label||'正在处理')+(counts.length?' · '+counts.join(' · '):'')}else{view.status.textContent=(counts.length?counts.join(' · '):traceStatusLabel(data.status))+(duration?' · '+duration:'')}
          const expandedItems=new Set([...view.body.querySelectorAll('.agent-trace-item-detail[open]')].map(node=>node.dataset.traceKey));
          view.body.replaceChildren();
          if(['queued','running'].includes(data.status)){const current=document.createElement('div');current.className='agent-trace-current';current.append(text('i',''),text('span',data.current_label||'正在处理'));view.body.append(current)}
          const decisions=Array.isArray(data.decisions)?data.decisions:[];
          if(decisions.length){const section=document.createElement('section');section.className='agent-trace-decisions';section.append(text('h4','本轮处理摘要'));const grid=document.createElement('div');for(const item of decisions){const card=document.createElement('div');card.append(text('b',item.title||'处理决定'),text('p',item.detail||''));grid.append(card)}section.append(grid);view.body.append(section)}
          const items=Array.isArray(data.items)?data.items:[];
          if(items.length){const section=document.createElement('section');section.className='agent-trace-timeline';section.append(text('h4','步骤与工具调用'));const list=document.createElement('ol');for(const item of items){const row=document.createElement('li');row.className='agent-trace-item '+(item.kind==='tool'?'is-tool ':'is-step ')+'status-'+(item.status||'running');const marker=text('span','', 'agent-trace-marker');marker.setAttribute('aria-hidden','true');const facts=Array.isArray(item.details)?item.details.filter(fact=>fact?.label&&fact?.value):[],evidence=Array.isArray(item.evidence)?item.evidence.filter(value=>value?.evidence_id):[],expandable=Boolean(facts.length||evidence.length),content=document.createElement(expandable?'details':'div'),head=document.createElement(expandable?'summary':'div');head.className='agent-trace-item-head';const heading=document.createElement('span');heading.className='agent-trace-item-heading';heading.append(text('strong',item.title||'执行步骤'));if(item.detail)heading.append(text('p',item.detail));if(item.kind==='tool'&&item.technical_name)heading.append(text('code',item.technical_name));const meta=document.createElement('span');meta.className='agent-trace-item-meta';meta.append(text('small',traceStatusLabel(item.status)));const itemDuration=formatTraceDuration(item.duration_ms);if(itemDuration)meta.append(text('small',itemDuration));if(expandable)meta.append(text('i','展开','agent-trace-expand-label'));head.append(heading,meta);content.append(head);if(expandable){content.className='agent-trace-item-detail';content.dataset.traceKey=item.key||'';content.open=expandedItems.has(item.key||'');const panel=document.createElement('div');panel.className='agent-trace-item-panel';appendTraceFacts(panel,facts);appendTraceEvidence(panel,item.evidence_title,evidence);content.append(panel)}row.append(marker,content);list.append(row)}section.append(list);view.body.append(section)}
          if(!decisions.length&&!items.length)view.body.append(text('p','执行台账正在建立，请稍候…','agent-trace-placeholder'));
          view.body.append(text('p',data.disclosure||'这里展示可审计的处理摘要，不包含模型隐藏推理或未验证草稿。','agent-trace-disclosure'));
        }
        async function refreshExecutionTrace(turn,{force=false}={}){if(!turn?.runId||!turn.trace||(!force&&turn.trace.settled))return;const view=turn.trace;if(view.loading){if(force)view.pendingRefresh=true;return}view.loading=true;view.root.dataset.runId=turn.runId;try{const response=await fetch('/api/agent/runs/'+encodeURIComponent(turn.runId)+'/trace'),data=await response.json();if(!response.ok)throw Error(data.error||'执行过程读取失败');if(!view.root.isConnected)return;renderExecutionTrace(view,data);turn.status=data.status||turn.status;view.settled=['completed','failed','cancelled'].includes(data.status);if(view.settled&&turn.runId===state.runId)stopTracePolling()}catch(error){if(view.root.isConnected&&!view.loaded){view.status.textContent='暂时无法读取';view.body.replaceChildren(text('p',error.message||'执行过程读取失败','agent-trace-placeholder'))}}finally{view.loading=false;if(view.pendingRefresh){view.pendingRefresh=false;setTimeout(()=>refreshExecutionTrace(turn,{force:true}),0)}}}
        function startTracePolling(turn){stopTracePolling();if(!turn?.runId||!turn.trace)return;refreshExecutionTrace(turn,{force:true});if(['completed','failed','cancelled'].includes(turn.status))return;state.traceTimer=setInterval(()=>refreshExecutionTrace(turn,{force:true}),1200)}
        function sourceCardMedia(refs){const assets=[],seen=new Set();for(const ref of refs||[]){for(const asset of ref.media_assets||[]){const url=String(asset?.url||'').trim();if(!url||seen.has(url))continue;seen.add(url);assets.push(asset)}}if(!assets.length)return null;const gallery=document.createElement('span');gallery.className='agent-source-card-media';for(const asset of assets.slice(0,2)){const image=document.createElement('img');image.src=asset.url;image.alt=asset.description||asset.title||'原文图片';image.loading='lazy';gallery.append(image)}if(assets.length>2)gallery.append(text('small','另有 '+(assets.length-2)+' 张'));return gallery}
        function renderSources(items,emptyMessage='本轮回答没有可展示的依据。'){sourceList.replaceChildren();sourceCount.textContent=String(items?.length||0);sourceEmpty.hidden=Boolean(items?.length);sourceEmpty.textContent=emptyMessage;for(const item of items||[]){const button=document.createElement('button');button.type='button';button.className='agent-source-card';const head=document.createElement('span');head.className='agent-source-card-head';head.append(text('b','〔'+item.citation_index+'〕 '+(item.title||'来源')),text('small',formatType(item.evidence_type)));button.append(head);const refs=item.source_refs||[],ref=refs[0]||{},location=text('span',[ref.filename,ref.page_label,ref.section_title].filter(Boolean).join(' · ')||'查看来源位置');location.className='agent-source-card-location';button.append(location);const media=sourceCardMedia(refs);if(media)button.append(media);button.append(text('p',item.excerpt||''));button.addEventListener('click',()=>openEvidence(item.evidence_id));sourceList.append(button)}}
        function displayTurnSources(turn,{animate=false}={}){const emptyMessage=['queued','running'].includes(turn.status)?'本轮回答的依据正在生成。':turn.status==='failed'?'本轮回答未生成可用依据。':'本轮回答没有可展示的依据。',items=turn.sourceCards||[],reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;resetSourceTransition();const token=state.sourceSwitchToken;renderSources(items,emptyMessage);if(!animate||reduced)return;sourceContent.style.willChange='opacity, transform';const transition=sourceContent.animate([{opacity:.72,transform:'translateY(3px)'},{opacity:1,transform:'translateY(0)'}],{duration:170,easing:'cubic-bezier(.2,.75,.25,1)'});transition.finished.catch(()=>{}).then(()=>{if(token===state.sourceSwitchToken)sourceContent.style.removeProperty('will-change')})}
        function turnLabel(turn,index){const question=String(turn.query||'本轮问题').replace(/\\s+/g,' ').trim();return '第 '+(index+1)+' 轮 · '+(question.length>28?question.slice(0,28)+'…':question)}
        function renderTurnPicker(){sourceTurnField.hidden=!state.turns.length;sourceTurn.replaceChildren(...state.turns.map((turn,index)=>new Option(turnLabel(turn,index),turn.runId)));if(state.selectedTurnId)sourceTurn.value=state.selectedTurnId}
        function bindTurnCard(turn){if(turn.trace)turn.trace.root.dataset.runId=turn.runId;if(!turn.card||turn.card.dataset.turnBound==='true')return;turn.card.dataset.turnBound='true';turn.card.dataset.runId=turn.runId;turn.card.classList.add('agent-turn-selectable');turn.card.addEventListener('click',event=>{if(!event.target.closest('button,details,a,input,select'))selectTurn(turn.runId)})}
        function upsertTurn(values){if(!values?.runId)return null;let turn=state.turns.find(item=>item.runId===values.runId);if(turn)Object.assign(turn,values);else{turn={runId:values.runId,query:values.query||'',sourceCards:values.sourceCards||[],status:values.status||'completed',card:values.card||null,trace:values.trace||null};state.turns.push(turn)}bindTurnCard(turn);renderTurnPicker();return turn}
        function selectTurn(runId,{persist=true,animate=true}={}){const turn=state.turns.find(item=>item.runId===runId);if(!turn)return false;const changed=Boolean(state.selectedTurnId&&state.selectedTurnId!==turn.runId);state.selectedTurnId=turn.runId;renderTurnPicker();messages.querySelectorAll('.agent-message.assistant').forEach(card=>card.classList.toggle('evidence-selected',card.dataset.runId===turn.runId));displayTurnSources(turn,{animate:animate&&changed});if(changed&&animate)sourcePane.scrollTo({top:0,behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});if(persist&&state.threadId)localStorage.setItem(evidenceKey(state.threadId),turn.runId);return true}
        function restoreTurnSelection(){const stored=state.threadId?localStorage.getItem(evidenceKey(state.threadId))||'':'';const explicit=Boolean(stored&&state.turns.some(turn=>turn.runId===stored));const target=explicit?stored:(state.turns.at(-1)?.runId||'');if(target)selectTurn(target,{animate:false});return explicit}
        async function openEvidence(evidenceId){dialogTitle.textContent='来源详情';dialogSubtitle.textContent='正在读取…';dialogBody.replaceChildren();dialog.showModal();try{const response=await fetch('/api/agent/evidence/'+encodeURIComponent(evidenceId)),data=await response.json();if(!response.ok)throw Error(data.error||'来源读取失败');dialogTitle.textContent=data.title||'来源详情';dialogSubtitle.textContent=formatType(data.evidence_type);dialogBody.append(text('p',data.content,'agent-evidence-content'));for(const ref of data.source_refs||[]){const section=document.createElement('section');section.className='agent-evidence-source';section.append(text('h3',ref.filename||'原文件'));section.append(text('p',[ref.page_label,ref.section_title].filter(Boolean).join(' · ')||'文件来源'));if(ref.quote)section.append(text('blockquote',ref.quote));if(ref.document_url){const link=text('a','打开原文件');link.href=ref.document_url;link.target='_blank';link.rel='noopener';section.append(link)}if(ref.table?.text){section.append(text('h4',ref.table.title||'表格'),text('pre',ref.table.text))}for(const asset of ref.media_assets||[]){if(!asset.url)continue;const link=document.createElement('a');link.href=asset.url;link.target='_blank';link.rel='noopener';link.className='agent-evidence-image';const image=document.createElement('img');image.src=asset.url;image.alt=asset.description||asset.title||'原文图片';link.append(image);section.append(link)}dialogBody.append(section)}}catch(error){dialogSubtitle.textContent='读取失败';dialogBody.append(text('p',error.message||'来源读取失败','failure'))}}
        function formatThreadTime(value){const date=new Date(value||'');if(Number.isNaN(date.getTime()))return '';const now=new Date(),sameDay=date.toDateString()===now.toDateString();if(sameDay)return date.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'});if(date.getFullYear()===now.getFullYear())return date.toLocaleDateString('zh-CN',{month:'2-digit',day:'2-digit'});return date.toLocaleDateString('zh-CN',{year:'numeric',month:'2-digit',day:'2-digit'})}
        function renderThreadList(message=''){
          threadList.replaceChildren();
          threadCount.textContent=state.base?(state.threads.length+' 个会话'):'选择知识库';
          if(message||!state.base||!state.threads.length){threadList.append(text('p',message||(!state.base?'选择知识库后显示历史会话。':'暂无历史会话。'),'agent-thread-empty'));return}
          for(const item of state.threads){
            const row=document.createElement('div');row.className='agent-thread-row';
            const button=document.createElement('button');button.type='button';button.className='agent-thread-item'+(String(item.id)===state.threadId?' active':'');button.dataset.threadId=item.id;button.setAttribute('aria-current',String(item.id)===state.threadId?'true':'false');
            const title=text('strong',String(item.title||'').trim()||'新对话');button.append(title);
            const preview=String(item.last_message_preview||'').replace(/\\s+/g,' ').trim();if(preview&&preview!==String(item.title||'').trim())button.append(text('span',preview,'agent-thread-preview'));
            const meta=document.createElement('small');const count=Number(item.message_count||0);meta.append(text('span',formatThreadTime(item.updated_at)||'刚刚'));if(count)meta.append(text('span',count+' 条消息'));if(item.active_run_id)meta.append(text('span','回答中','agent-thread-running'));button.append(meta);
            button.disabled=state.busy||state.loadingThreads;button.addEventListener('click',()=>{if(button.dataset.threadId!==state.threadId)loadThread(button.dataset.threadId)});
            const remove=text('button','删除','agent-thread-delete');remove.type='button';remove.dataset.deleteThread=item.id;remove.dataset.blocked=String(Boolean(item.active_run_id));remove.setAttribute('aria-label','删除会话：'+(String(item.title||'').trim()||'新对话'));remove.title=item.active_run_id?'回答生成期间不能删除，请先停止回答':'删除会话';remove.disabled=state.busy||state.loadingThreads||Boolean(item.active_run_id);remove.addEventListener('click',()=>deleteThread(item.id));row.append(button,remove);threadList.append(row);
          }
        }
        async function deleteThread(threadId){if(state.busy||state.loadingThreads)return;const item=state.threads.find(thread=>thread.id===threadId);if(!item)return;const title=String(item.title||'').trim()||'新对话';if(!confirm('删除会话“'+title+'”？\\n\\n该会话将从历史列表移除。'))return;state.loadingThreads=true;renderThreadList();setBusy(false);let nextId='';try{const response=await fetch('/api/agent/threads/'+encodeURIComponent(threadId),{method:'DELETE'}),data=await response.json();if(!response.ok)throw Error(data.error||'会话删除失败');state.threads=state.threads.filter(thread=>thread.id!==threadId);localStorage.removeItem(evidenceKey(threadId));if(localStorage.getItem(threadKey(state.base.id))===threadId)localStorage.removeItem(threadKey(state.base.id));if(state.threadId===threadId){state.threadId='';state.runId='';clearConversation();nextId=state.threads[0]?.id||''}syncRoute()}
          catch(error){alert(error.message||'会话删除失败')}
          finally{state.loadingThreads=false;renderThreadList();setBusy(false)}
          if(nextId)await loadThread(nextId)
        }
        function renderHistory(history,runs=[]){clearConversation();state.lastQuery='';const runStatus=new Map(runs.map(run=>[run.id,run.status]));let previousQuery='',turnNumber=0;for(const item of history){if(item.role==='user'){previousQuery=item.content;state.lastQuery=item.content;addMessage('user',item.content)}else if(item.role==='assistant'){turnNumber+=1;const answer=item.metadata?.answer||{},runId=item.run_id||'',message=addMessage('assistant',item.content,answer,{runId,query:previousQuery,turnNumber});upsertTurn({runId,query:previousQuery,sourceCards:answer.source_cards||[],status:runStatus.get(runId)||'completed',card:message.card,trace:message.trace})}}return restoreTurnSelection()}
        async function fetchThreadList(){const response=await fetch('/api/agent/threads?project_id='+encodeURIComponent(state.base.id)),data=await response.json();if(!response.ok)throw Error(data.error||'历史会话读取失败');state.threads=Array.isArray(data.items)?data.items:[];renderThreadList();return state.threads}
        async function refreshThreadList(){if(!state.base?.id)return;try{await fetchThreadList()}catch(_error){}}
        async function createThread(){
          if(!state.base?.queryable||state.busy)return '';
          state.loadingThreads=true;renderThreadList('正在新建会话…');setBusy(state.busy);
          try{const response=await fetch('/api/agent/threads',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project_id:state.base.id})}),data=await response.json();if(!response.ok)throw Error(data.error||'无法创建问答会话');const item={...data.thread,message_count:0,last_message_preview:'',active_run_id:''};state.threads=[item,...state.threads.filter(existing=>existing.id!==item.id)];state.threadId=item.id;state.runId='';localStorage.setItem(threadKey(state.base.id),state.threadId);clearConversation();syncRoute();return state.threadId}
          finally{state.loadingThreads=false;renderThreadList();setBusy(false)}
        }
        async function loadThread(threadId){
          if(!threadId||state.busy)return;const generation=++state.loadGeneration,baseId=String(state.base?.id||'');state.loadingThreads=true;renderThreadList();setBusy(false);let activeRun=null;
          try{const response=await fetch('/api/agent/threads/'+encodeURIComponent(threadId)),payload=await response.json();if(!response.ok)throw Error(payload.error||'会话读取失败');if(generation!==state.loadGeneration||baseId!==String(state.base?.id||''))return;state.threadId=String(threadId);state.runId='';localStorage.setItem(threadKey(state.base.id),state.threadId);syncRoute();const explicitTurn=renderHistory(payload.messages||[],payload.runs||[]);activeRun=(payload.runs||[]).slice().reverse().find(item=>['queued','running'].includes(item.status))||null;if(activeRun){state.runId=activeRun.id;state.lastQuery=activeRun.query||state.lastQuery;state.assistant=addMessage('assistant','正在恢复本次回答…',null,{runId:activeRun.id,query:state.lastQuery,turnNumber:state.turns.length+1,liveTrace:true});upsertTurn({runId:activeRun.id,query:state.lastQuery,sourceCards:[],status:activeRun.status,card:state.assistant.card,trace:state.assistant.trace});if(!explicitTurn)selectTurn(activeRun.id,{animate:false})}}
          catch(error){if(generation===state.loadGeneration){state.threadId='';localStorage.removeItem(threadKey(state.base.id));clearConversation();renderThreadList(error.message||'会话读取失败')}}
          finally{if(generation===state.loadGeneration){state.loadingThreads=false;renderThreadList();setBusy(Boolean(activeRun));if(activeRun)connectEvents('/api/agent/runs/'+activeRun.id+'/events')}}
        }
        async function ensureThread(){if(state.threadId)return state.threadId;return createThread()}
        async function loadThreads(preferredThread=''){
          if(!state.base?.queryable)return;state.loadingThreads=true;renderThreadList('正在读取历史会话…');setBusy(false);
          try{await fetchThreadList();const stored=localStorage.getItem(threadKey(state.base.id))||'',selected=state.threads.some(item=>item.id===preferredThread)?preferredThread:state.threads.some(item=>item.id===stored)?stored:(state.threads[0]?.id||'');state.loadingThreads=false;if(selected)await loadThread(selected);else await createThread()}
          catch(error){state.loadingThreads=false;state.threads=[];renderThreadList(error.message||'历史会话读取失败');setBusy(false)}
        }
        function updateCurrentThread(queryText){const item=state.threads.find(thread=>thread.id===state.threadId);if(!item)return;if(!String(item.title||'').trim())item.title=queryText.slice(0,80);item.last_message_preview=queryText;item.message_count=Number(item.message_count||0)+1;item.updated_at=new Date().toISOString();state.threads=[item,...state.threads.filter(thread=>thread.id!==item.id)];renderThreadList()}
        function terminal(kind,payload={}){state.stream?.close();state.stream=null;stopTracePolling();state.busy=false;const turn=upsertTurn({runId:state.runId,status:kind});if(turn){refreshExecutionTrace(turn,{force:true});if(state.selectedTurnId===turn.runId)selectTurn(turn.runId,{animate:false})}setBusy(false);if(kind==='failed'&&state.assistant){state.assistant.body.textContent='本次问答失败：'+(payload.error||'请稍后重试');state.assistant.card.classList.add('failed')}else if(kind==='cancelled'&&state.assistant){state.assistant.body.textContent=state.assistant.body.textContent||'本次回答已停止。';state.assistant.card.classList.add('cancelled')}messages.scrollTop=messages.scrollHeight;refreshThreadList()}
        function connectEvents(url){state.stream?.close();const activeTurn=state.turns.find(item=>item.runId===state.runId);if(activeTurn)startTracePolling(activeTurn);const stream=new EventSource(url);state.stream=stream;for(const name of ['queued','status','progress','evidence','cancellation_requested'])stream.addEventListener(name,event=>{const data=JSON.parse(event.data||'{}');const turn=state.turns.find(item=>item.runId===state.runId);if(turn)refreshExecutionTrace(turn,{force:true});if(name==='progress')hint.textContent=({authorizing:'正在确认知识范围…',resolving_context:'正在理解本轮语境…',retrieving:'正在检索证据…',ranking:'正在融合与重排…',assessing:'正在判断证据是否充分…',supplementing:'正在补充检索…',reranking:'正在重排补充证据…',reassessing:'正在再次检查证据…',answering:'正在组织回答…',validating_citations:'正在核验引用…'}[data.stage]||'正在处理…')});stream.addEventListener('answer_start',event=>{const data=JSON.parse(event.data||'{}');state.assistant.body.textContent='';state.assistant.card.dataset.messageId=data.message_id||''});stream.addEventListener('answer_delta',event=>{const data=JSON.parse(event.data||'{}');state.assistant.body.textContent+=(data.delta||'');messages.scrollTop=messages.scrollHeight});stream.addEventListener('sources',event=>{const data=JSON.parse(event.data||'{}'),turn=upsertTurn({runId:state.runId,sourceCards:data.items||[],status:'running'});state.assistant.card.dataset.sources=JSON.stringify(data.items||[]);if(turn&&state.selectedTurnId===turn.runId)selectTurn(turn.runId,{animate:false})});stream.addEventListener('completed',event=>terminal('completed',JSON.parse(event.data||'{}')));stream.addEventListener('failed',event=>terminal('failed',JSON.parse(event.data||'{}')));stream.addEventListener('cancelled',event=>terminal('cancelled',JSON.parse(event.data||'{}')));stream.addEventListener('error',event=>{if(event.data)terminal('failed',JSON.parse(event.data||'{}'))});stream.onerror=async()=>{if(!state.busy)return;try{const response=await fetch('/api/agent/runs/'+state.runId),run=await response.json();if(['completed','failed','cancelled'].includes(run.status))terminal(run.status,{error:run.error_message})}catch(_error){}}}
        function selectedBaseStrategy(){return String(state.base?.strategy||state.base?.strategy_key||'')}
        function selectedBaseIsNative(){return selectedBaseStrategy()==='graphrag_native'}
        async function submitQuestion(){const value=query.value.trim();if(!value||state.busy||state.loadingThreads||!state.base?.queryable)return;let threadId='';try{threadId=await ensureThread()}catch(error){hint.textContent=error.message||'无法创建问答会话';return}if(!threadId)return;const turnContext={query:value,runId:'',turnNumber:state.turns.length+1,liveTrace:true};state.lastQuery=value;addMessage('user',value);state.assistant=addMessage('assistant','正在检索可核验依据…',null,turnContext);query.value='';setBusy(true);try{const response=await fetch('/api/agent/threads/'+threadId+'/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:value,mode:selectedBaseIsNative()?modeSelect.value:'auto'})}),data=await response.json();if(!response.ok)throw Error(data.error||'问题提交失败');state.runId=data.run_id;turnContext.runId=data.run_id;upsertTurn({runId:data.run_id,query:value,sourceCards:[],status:'queued',card:state.assistant.card,trace:state.assistant.trace});selectTurn(data.run_id);updateCurrentThread(value);connectEvents(data.events_url)}catch(error){state.assistant.body.textContent='问题提交失败：'+(error.message||'请稍后重试');state.assistant.card.classList.add('failed');if(state.assistant.trace){state.assistant.trace.status.textContent='提交失败';state.assistant.trace.body.replaceChildren(text('p','任务没有进入执行流程。','agent-trace-placeholder'))}state.busy=false;setBusy(false);refreshThreadList()}}
        function baseOptionLabel(item){const status=String(item.status_label||'').replace(/^可问答 *· */,'');return item.name+(status?'（'+status+'）':'')}
        async function chooseBase(){const preferredThread=new URL(location.href).searchParams.get('thread')||'';++state.loadGeneration;state.stream?.close();state.stream=null;state.runId='';state.threadId='';state.threads=[];state.base=state.bases.find(item=>String(item.id)===projectSelect.value)||null;if(state.base?.id)localStorage.setItem(selectedProjectKey,String(state.base.id));else localStorage.removeItem(selectedProjectKey);const native=selectedBaseIsNative();modeField.hidden=!native;if(!native)modeSelect.value='auto';clearConversation();renderThreadList();query.value='';syncRoute();setBusy(false);if(state.base?.queryable)await loadThreads(preferredThread)}
        async function loadBases(){try{const response=await fetch('/api/agent/knowledge-bases'),data=await response.json();if(!response.ok)throw Error(data.error||'知识库读取失败');state.bases=data.items||[];projectSelect.replaceChildren(new Option(data.queryable_count?'选择一个知识库':'当前没有可用于问答的知识库',''));for(const item of state.bases){const option=new Option(baseOptionLabel(item),String(item.id));option.disabled=!item.queryable;projectSelect.append(option)}const requested=new URL(location.href).searchParams.get('project')||localStorage.getItem(selectedProjectKey)||'',candidate=state.bases.find(item=>String(item.id)===requested&&item.queryable);if(candidate){projectSelect.value=String(candidate.id);await chooseBase()}}catch(error){projectSelect.replaceChildren(new Option(error.message||'知识库读取失败',''))}}
        projectSelect.addEventListener('change',chooseBase);sourceTurn.addEventListener('change',()=>selectTurn(sourceTurn.value));newThread.addEventListener('click',async()=>{++state.loadGeneration;try{await createThread();query.focus()}catch(error){renderThreadList(error.message||'无法创建问答会话')}});query.addEventListener('input',()=>setBusy(state.busy));query.addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();submitQuestion()}});send.addEventListener('click',submitQuestion);stop.addEventListener('click',async()=>{if(!state.runId)return;stop.disabled=true;stop.textContent='正在停止…';try{await fetch('/api/agent/runs/'+state.runId+'/cancel',{method:'POST'});}finally{stop.disabled=false;stop.textContent='停止回答'}});$('#agent-evidence-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',event=>{if(event.target===dialog)dialog.close()});loadBases();renderThreadList();setBusy(false);
      })();
      </script></body></html>"""
    return template.replace("__CSS__", CSS)


def render_status_page(status: int, title: str, message: str) -> str:
    """Render a consistent human-facing status page for browser routes."""

    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <title>{_esc(title)} · Knowledge Forge</title><style>{CSS}</style></head><body>
      <main class="shell home-shell"><header class="home-header"><div><h1>Knowledge Forge</h1>
      <p>通用知识库与知识图谱构建平台</p></div>
      <nav class="home-primary-actions" aria-label="主要功能">
        <a class="button primary" href="/#knowledge-bases">知识库</a>
        <a class="button secondary" href="/chat">智能问答</a>
      </nav></header>
      <section class="panel status-page" aria-labelledby="status-page-title">
        <span class="status-page-code">{int(status)}</span>
        <div class="status-page-copy"><h2 id="status-page-title">{_esc(title)}</h2><p>{_esc(message)}</p>
          <div class="status-page-actions"><a class="button primary" href="/">返回首页</a>
            <a class="button secondary" href="/#knowledge-bases">查看知识库</a></div>
        </div>
      </section></main></body></html>"""


def _embedded_graph_browser(
    strategy_key: str,
    *,
    ready: bool,
    active: bool,
) -> str:
    native = strategy_key == GRAPHRAG_NATIVE_STRATEGY
    title = "GraphRAG Native 知识图谱" if native else "Knowledge Forge 知识图谱"
    description = (
        "浏览实体、关系、社区与文本单元证据；内容来自当前有效的知识图谱成果。"
        if native else
        "浏览实体、关系、社区和原文证据。可筛选实体或社区，点击节点和连线查看详细信息。"
    )
    group_label = "全部社区"
    relationship_label = "GraphRAG 抽取关系" if native else "已验证关系"
    evidence_label = "文本单元依据" if native else "证据"
    detail_intro = (
        "全局地图显示主要社区及其联系。点击社区进入实体关系，点击节点或连线查看关系说明与文本单元依据。"
        if native else
        "全局地图只显示最重要的社区和它们之间的联系。点击社区进入实体关系，点击节点或连线查看详情与原文证据。"
    )
    community_scope_help = (
        "重点关系显示核心实体及其直接关系；更多关系扩展到相邻实体；全部关系显示该社区全部抽取关系。"
        if native else
        "重点关系显示核心实体及其直接关系；更多关系扩展到相邻实体；全部关系显示该社区全部已验证关系。"
    )
    evidence_modal_title = "文本单元资料" if native else "原文资料"
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
    # A ready graph owns its loading state inside the canvas.  A graph that has
    # not been built yet has no canvas to load, so it retains the standalone
    # empty/building state rather than exposing inactive controls.
    outer_initial = "" if ready else initial
    canvas_initial = initial if ready else ""
    viewer = "" if ready else " hidden"
    return f"""<section id="embedded-graph-browser" class="workspace-section embedded-graph-browser"
      data-strategy="{_esc(strategy_key)}" data-ready="{str(ready).lower()}" data-active="{str(active).lower()}">
      <style>
        .embedded-graph-mode-switch{{display:inline-flex;align-items:center;padding:8px 10px;border:1px solid #c7d5e5;border-radius:8px;background:#f4f8fc;color:#1769c2;font-size:13px;font-weight:700}}
        .graph-evidence-quote{{margin-top:7px;padding:9px 10px;border-left:3px solid #72a9df;border-radius:4px;background:#f5f9fd;color:#29445f;white-space:pre-wrap;line-height:1.55}}
        .graph-provenance{{color:#718398;font-size:11px}}
        .kf-gpu-graph-edge-labels{{position:absolute;inset:0;z-index:1;overflow:hidden;pointer-events:none}}
        .kf-gpu-edge-label{{position:absolute;left:0;top:0;max-width:130px;padding:2px 5px;border:1px solid rgba(144,169,193,.7);border-radius:9px;background:rgba(255,255,255,.92);color:#315d89;font-size:10px;font-weight:700;line-height:1.25;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transform:translate(-50%,-50%)}}
        .kf-gpu-edge-label.is-topic-association{{border-style:dashed;border-color:#9aafc2;color:#536f89}}
        .kf-gpu-edge-label.is-selected{{border-color:#e27619;background:#fff3dc;color:#8f4c0c}}
        .kf-gpu-default-edge.is-muted{{opacity:.08}}.kf-gpu-default-node.is-muted{{opacity:.28}}
        .kf-gpu-graph-surface{{z-index:0}}.embedded-graph-canvas .kf-gpu-default-edges,.embedded-graph-canvas .kf-gpu-emphasized-edges{{min-width:0;min-height:0;background:transparent}}.kf-gpu-default-edges{{z-index:10}}.kf-gpu-default-nodes{{z-index:11}}.kf-gpu-emphasized-edges{{z-index:12}}.kf-gpu-emphasized-nodes{{z-index:13}}.kf-gpu-node-rings{{z-index:14}}.kf-gpu-graph-edge-labels{{z-index:15}}.kf-gpu-graph-labels{{z-index:16}}.kf-gpu-graph-tooltip{{z-index:17}}.graph-canvas-scope,.graph-canvas-nav{{z-index:30}}.graph-canvas-overlay{{z-index:40}}
        .embedded-graph-layout{{--graph-sidebar-width:320px;grid-template-columns:minmax(0,1fr) 11px var(--graph-sidebar-width);gap:0;align-items:stretch}}
        .graph-pane-resizer{{position:relative;min-width:11px;border:0;background:transparent;cursor:col-resize;touch-action:none}}
        .graph-pane-resizer::before{{content:"";position:absolute;left:4px;top:14px;bottom:14px;width:3px;border-radius:3px;background:#d8e2ec;transition:background .15s,box-shadow .15s}}
        .graph-pane-resizer:hover::before,.graph-pane-resizer:focus-visible::before,.graph-pane-resizer.is-dragging::before{{background:#6c9dcb;box-shadow:0 0 0 3px rgba(70,130,190,.12)}}
        .graph-pane-resizer:focus-visible{{outline:0}}body.is-resizing-graph-pane,body.is-resizing-graph-pane *{{cursor:col-resize!important;user-select:none!important}}
        .embedded-graph-sidebar{{min-width:0}}
        .graph-community-card-summary,.graph-community-card-report{{display:block;grid-column:2;margin-top:4px;color:#60758d;font-size:11px;line-height:1.5;text-align:left}}
        .graph-community-card-report{{padding-top:6px;border-top:1px solid #edf2f7;color:#405d78}}
        .graph-community-card-report b{{display:block;margin-bottom:2px;color:#315878;font-size:11px}}
        .graph-report-loading{{display:inline-flex;align-items:center;gap:7px;color:#607991}}
        .graph-report-spinner{{display:inline-block;width:12px;height:12px;border:2px solid #d2deea;border-top-color:#397dbd;border-radius:50%;animation:stat-spin .8s linear infinite}}
        .embedded-graph-group-card .graph-report-loading{{grid-column:auto;margin:0}}
        .embedded-graph-group-card .graph-report-spinner{{grid-row:auto;grid-column:auto;width:12px;height:12px;margin:0}}
        .graph-community-detail{{color:#344f69}}
        .graph-community-detail-header{{position:sticky;top:-15px;z-index:2;margin:-15px -15px 0;padding:14px 15px 10px;border-bottom:1px solid #dce6ef;background:rgba(251,253,255,.97);backdrop-filter:blur(5px)}}
        .graph-community-detail-header h3{{margin:0}}
        .graph-community-description{{display:block;margin:0}}
        .graph-community-section{{padding:12px 0;border:0;border-bottom:1px solid #e2eaf2;border-radius:0;background:transparent}}
        .graph-community-section h4{{margin:0 0 5px;color:#315978;font-size:12px}}
        .graph-community-section p{{margin:0;color:#48627b;font-size:12px;line-height:1.65}}
        .graph-community-relations{{background:transparent}}
        .graph-community-report{{margin:0;padding:14px 0;border:0;border-bottom:1px solid #e2eaf2;border-radius:0;background:transparent}}
        .graph-community-report>h4{{margin:0 0 8px;color:#234f76;font-size:14px;line-height:1.45}}
        .graph-community-report-summary{{margin:0 0 7px;color:#405c75;font-size:12px;line-height:1.68}}
        .graph-report-findings-heading{{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin:13px 0 2px}}
        .graph-report-findings-heading h5{{margin:0;color:#315978;font-size:12px}}
        .graph-report-findings-heading small{{color:#71869b;font-size:10px;text-align:right}}
        .graph-report-findings{{counter-reset:report-finding;display:block;margin:0;padding:0;list-style:none}}
        .graph-report-findings li{{position:relative;padding:9px 0 9px 25px;border-top:1px solid #e8eef4;background:transparent}}
        .graph-report-findings li::before{{counter-increment:report-finding;content:counter(report-finding);position:absolute;left:0;top:10px;display:grid;place-items:center;width:17px;height:17px;border-radius:50%;background:#e9f2fa;color:#35668f;font-size:10px;font-weight:800}}
        .graph-report-findings b{{display:block;margin-bottom:3px;color:#2e5374;font-size:12px}}
        .graph-report-findings p{{margin:0;color:#48627b;font-size:12px;line-height:1.62}}
        .graph-finding-evidence{{margin-top:8px;padding:7px 9px;border-radius:5px;background:#f5f8fb}}
        .graph-finding-evidence>summary{{display:flex;align-items:baseline;gap:7px;cursor:pointer;list-style:none;color:#315f86;font-size:11px;font-weight:700}}
        .graph-finding-evidence>summary::-webkit-details-marker{{display:none}}
        .graph-finding-evidence>summary small{{min-width:0;overflow:hidden;color:#6f8296;font-size:10px;font-weight:400;text-overflow:ellipsis;white-space:nowrap}}
        .graph-finding-evidence>summary::after{{content:"展开";margin-left:auto;color:#6b8298;font-size:10px;font-weight:400}}
        .graph-finding-evidence[open]>summary::after{{content:"收起"}}
        .graph-finding-evidence .graph-source-list{{margin-top:6px}}
        .graph-finding-evidence .graph-source-card:first-child{{border-top:0}}
        .graph-finding-evidence-missing{{margin-top:7px!important;color:#8a6b43!important;font-size:10px!important}}
        .graph-report-limitations{{margin:7px 0 0}}
        .graph-report-limitations summary{{display:flex;align-items:center;justify-content:space-between;gap:8px;color:#617890;font-size:11px}}
        .graph-report-limitations p{{margin:7px 0 0;padding:8px 9px;border-radius:5px;background:#f4f7fa;color:#61758a;font-size:11px;line-height:1.55}}
        .graph-community-sources{{padding:12px 0;border-bottom:1px solid #e2eaf2}}
        .graph-community-sources>summary,.graph-report-limitations>summary{{cursor:pointer;list-style:none}}
        .graph-community-sources>summary::-webkit-details-marker,.graph-report-limitations>summary::-webkit-details-marker{{display:none}}
        .graph-community-sources>summary{{display:flex;align-items:baseline;justify-content:space-between;gap:8px;color:#315978;font-size:12px;font-weight:700}}
        .graph-community-sources>summary::after,.graph-report-limitations>summary::after{{content:"展开";margin-left:auto;color:#6b8298;font-size:10px;font-weight:400}}
        .graph-community-sources[open]>summary::after,.graph-report-limitations[open]>summary::after{{content:"收起"}}
        .graph-community-sources>summary small{{margin-right:auto;color:#71869b;font-size:10px;font-weight:400}}
        .graph-community-sources blockquote{{margin:10px 0 0;padding:8px 10px;border-left:2px solid #82acd3;background:#f5f8fb;color:#48637d;font-size:12px;line-height:1.6}}
        .graph-source-list{{display:grid;gap:0;margin-top:8px}}
        .graph-source-card{{padding:9px 0;border:0;border-top:1px solid #e5edf4;border-radius:0;background:transparent}}
        .graph-source-card p{{margin:5px 0 0;line-height:1.55}}
        .graph-source-location{{display:flex;align-items:center;flex-wrap:wrap;gap:5px;color:#60778e;font-size:11px}}
        .graph-source-location a{{color:#246598;font-weight:600;text-decoration:none;overflow-wrap:anywhere}}
        .graph-source-location a:hover{{color:#174f82;text-decoration:underline}}
        .graph-source-relation{{color:#315d85;font-size:11px;font-weight:700}}
        .graph-source-media{{display:flex;gap:7px;flex-wrap:wrap;margin-top:7px}}
        .graph-media-thumb{{display:inline-flex;align-items:center;justify-content:center;max-width:116px;min-height:34px;padding:4px;border:1px solid #cedbe8;border-radius:5px;background:#f7fafc;color:#315d85;cursor:pointer}}
        .graph-media-thumb:hover{{border-color:#6ea4d7;background:#eef6fd}}
        .graph-media-thumb img{{display:block;max-width:104px;max-height:72px;object-fit:contain}}
        .graph-evidence-modal{{width:min(980px,92vw);max-height:88vh;padding:0;border:0;border-radius:10px;box-shadow:0 18px 55px rgba(18,42,67,.24)}}
        .graph-evidence-modal::backdrop{{background:rgba(24,42,61,.42)}}
        .graph-evidence-modal-header{{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 15px;border-bottom:1px solid #dce6ef}}
        .graph-evidence-modal-header h3{{margin:0;font-size:15px}}
        .graph-evidence-modal-close{{width:34px;height:34px;border:0;border-radius:6px;background:#edf3f8;color:#355b7e;font-size:22px;cursor:pointer}}
        .graph-evidence-modal-body{{max-height:calc(88vh - 60px);padding:16px;overflow:auto}}
        .graph-evidence-modal-body img{{display:block;max-width:100%;height:auto;margin:auto}}
        .graph-evidence-modal-body pre{{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}}
        @media(max-width:1100px){{.embedded-graph-layout{{grid-template-columns:1fr}}.graph-pane-resizer{{display:none}}}}
      </style>
      <div class="section-heading"><div><h2>{_esc(title)}</h2><p>{_esc(description)}</p></div>
        <div class="embedded-graph-tabs" role="tablist" aria-label="图谱浏览方式">
          <button id="embedded-graph-network-tab" class="embedded-graph-tab is-active"
            type="button" role="tab" aria-selected="true" aria-controls="embedded-graph-network-panel"
            data-graph-tab="network">图谱连接</button>
          <button id="embedded-graph-audit-tab" class="embedded-graph-tab"
            type="button" role="tab" aria-selected="false" aria-controls="embedded-graph-audit-panel"
            tabindex="-1" data-graph-tab="audit">关系审计表</button>
        </div></div>
      {outer_initial}
      <div class="embedded-graph-viewer"{viewer}>
        <div class="embedded-graph-controls">
          <select data-graph-group-filter aria-label="图谱分组"><option value="">{_esc(group_label)}</option></select>
          <input data-graph-filter placeholder="筛选实体名称、类别、说明或别名" aria-label="图谱筛选条件">
          <button class="button primary" type="button" data-apply-graph-filter>筛选</button>
          <button class="button secondary" type="button" data-reset-graph>重置</button>
        </div>
        <div id="embedded-graph-network-panel" role="tabpanel"
          aria-labelledby="embedded-graph-network-tab" data-graph-panel="network">
          <div class="graph-reading-key" aria-label="图谱阅读说明">
            <span><i class="graph-key-dot"></i>圆点：实体或社区；圆点越大，关联越多</span>
            <span><i class="graph-key-line"></i>连线：{_esc(relationship_label)}</span>
            <span data-graph-topic-key hidden><i class="graph-key-topic-line"></i>虚线：内容相近的社区</span>
          </div>
          <div class="embedded-graph-layout">
            <div class="embedded-graph-canvas">
              <div data-graph-network role="img" aria-label="可拖动、缩放和逐层查看的知识图谱"></div>
              <div class="graph-canvas-overlay" data-graph-status aria-live="polite">{canvas_initial}</div>
              <div class="graph-canvas-scope" role="group" aria-label="全局关系范围" data-graph-overview-scope hidden>
                <button type="button" data-overview-relation-scope="backbone">主要关系</button>
                <button type="button" data-overview-relation-scope="all">全部关系</button>
                <button type="button" data-overview-relation-scope="all_communities">全部关系社区</button>
                <span class="graph-scope-help">
                  <button class="graph-scope-help-trigger" type="button" aria-label="查看全局关系范围说明"
                    aria-describedby="graph-overview-scope-help">?</button>
                  <span class="graph-scope-help-tip" id="graph-overview-scope-help" role="tooltip">主要关系显示社区间较有代表性的事实联系；全部关系显示所有跨社区事实联系；全部关系社区还会显示只有内部关系的社区，并用虚线标出内容相近的导航关联。虚线不代表原文事实。</span>
                </span>
              </div>
              <div class="graph-canvas-scope" role="group" aria-label="社区关系范围" data-graph-community-scope hidden>
                <button type="button" data-community-stage="focus">重点关系</button>
                <button type="button" data-community-stage="expanded">更多关系</button>
                <button type="button" data-community-stage="full">全部关系</button>
                <span class="graph-scope-help">
                  <button class="graph-scope-help-trigger" type="button" aria-label="查看社区关系范围说明"
                    aria-describedby="graph-community-scope-help">?</button>
                  <span class="graph-scope-help-tip" id="graph-community-scope-help" role="tooltip">{_esc(community_scope_help)}</span>
                </span>
              </div>
              <div class="graph-canvas-nav" role="toolbar" aria-label="图谱导航">
                <button type="button" data-graph-history-back aria-label="后退" title="后退" disabled><span aria-hidden="true">‹</span></button>
                <button type="button" data-graph-history-forward aria-label="前进" title="前进" disabled><span aria-hidden="true">›</span></button>
                <button type="button" data-graph-return-overview aria-label="返回全局地图" title="返回全局地图" disabled><span aria-hidden="true">⌂</span></button>
                <span class="graph-canvas-zoom" role="group" aria-label="图谱缩放">
                  <button type="button" data-graph-zoom-out aria-label="缩小图谱" title="缩小图谱"><span aria-hidden="true">−</span></button>
                  <output data-graph-zoom-value aria-live="polite">100%</output>
                  <button type="button" data-graph-zoom-in aria-label="放大图谱" title="放大图谱"><span aria-hidden="true">＋</span></button>
                </span>
                <button type="button" data-graph-fit aria-label="适应画布" title="适应画布"><span aria-hidden="true">⤢</span></button>
              </div>
              <p class="embedded-graph-count" data-graph-count></p>
            </div>
            <div class="graph-pane-resizer" data-graph-pane-resizer role="separator"
              aria-label="调整图谱与详情栏宽度" aria-orientation="vertical"
              aria-valuemin="260" aria-valuenow="320" tabindex="0"
              title="拖动调整两侧宽度；双击恢复默认"></div>
            <aside class="embedded-graph-sidebar">
              <div data-graph-details><h3>从社区开始浏览</h3><p>{_esc(detail_intro)}</p></div>
              <div data-graph-groups class="embedded-graph-groups"></div>
            </aside>
          </div>
          <dialog class="graph-evidence-modal" data-graph-evidence-modal>
            <div class="graph-evidence-modal-header"><h3 data-graph-evidence-modal-title>{_esc(evidence_modal_title)}</h3>
              <button class="graph-evidence-modal-close" type="button" data-graph-evidence-modal-close aria-label="关闭">×</button></div>
            <div class="graph-evidence-modal-body" data-graph-evidence-modal-body></div>
          </dialog>
        </div>
        <div id="embedded-graph-audit-panel" role="tabpanel"
          aria-labelledby="embedded-graph-audit-tab" data-graph-panel="audit" hidden>
          <p class="muted" data-graph-audit-summary></p>
          <div class="table-wrap embedded-graph-audit"><table><thead><tr>
            <th>源实体</th><th>关系</th><th>目标实体</th><th data-graph-audit-metric>关系信息</th><th data-graph-audit-evidence>{_esc(evidence_label)}</th>
          </tr></thead><tbody data-graph-audit-rows></tbody></table></div>
        </div>
      </div>
    </section>"""


def render_page(
    project_id: int | None,
    strategy_key: str = KNOWLEDGE_FORGE_STRATEGY,
    message: tuple[str, str] | None = None,
) -> str:
    try:
        strategy_key = normalize_strategy(strategy_key)
    except ValueError:
        strategy_key = KNOWLEDGE_FORGE_STRATEGY
    visible_strategy_keys = {
        item.key for item in strategy_options(include_hidden=False)
    }
    if strategy_key not in visible_strategy_keys:
        strategy_key = KNOWLEDGE_FORGE_STRATEGY
    ui = _strategy_ui(strategy_key)
    projects, models = SERVICE.db.list_projects(), _available_models()
    current = next((item for item in projects if item["id"] == project_id), None)
    if current:
        binding = _project_binding(int(current["id"]))
        if binding is not None:
            if str(binding.get("association_status") or "") != "bound":
                raise ValueError("知识库尚未绑定知识图谱构建方式")
            if str(binding.get("selected_strategy") or "") != strategy_key:
                raise ValueError("工作区与知识库绑定的构建方式不一致")
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY:
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
    preparation_status_getter = getattr(SERVICE, "project_preparation_status", None)
    preparation_status = (
        preparation_status_getter(int(current["id"]))
        if current and callable(preparation_status_getter)
        else _fallback_preparation_status(documents, strategy_key)
    )
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
    mutating_job_getter = getattr(
        SERVICE, "latest_active_mutating_job", SERVICE.latest_active_job,
    )
    active_job = newest(
        strategy_active_job,
        mutating_job_getter(int(current["id"])) if current else None,
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
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY
        else f"""<div class="workspace-section settings-processing">
          <div class="section-heading"><div><h2>源文档解析</h2>
            <p>解析器应用于下一次上传；文本单元参数由 GraphRAG Native 工作流固定管理。</p></div></div>
          <div class="processing-options settings-processing-options">
            <div id="processing-selects" class="form-grid compact">
              <label>解析器<select name="requested_parser" form="document-upload-form">{_named_options(PARSER_OPTIONS,'auto')}</select></label>
              <input type="hidden" name="requested_strategy" value="auto" form="document-upload-form">
              <div class="native-config-note"><b>文本单元设置</b><span>GraphRAG 3.1.1 固定使用 900 Token、100 Token 重叠和 o200k_base 分词器。</span></div>
            </div>
            <p id="processing-hint" class="muted">这里只生成源结构；不会生成 Knowledge Forge 知识块、向量索引或知识结构定义。</p>
          </div>
        </div>"""
    )
    rows = _document_rows(documents, strategy_key)
    knowledge_structure_section = (
        _knowledge_structure_details(
            schema,
            project_id=int(current["id"]) if current else 0,
            reassessment_available=schema_reassessment_available,
        ) if strategy_key == KNOWLEDGE_FORGE_STRATEGY else ""
    )
    process_artifacts_section = (
        _knowledge_forge_process_artifacts(documents)
        if strategy_key == KNOWLEDGE_FORGE_STRATEGY
        else _graphrag_native_process_artifacts(documents)
        if strategy_key == GRAPHRAG_NATIVE_STRATEGY else ""
    )
    current_block = ""
    graph_query_actions = ""
    embedded_graph_section = ""
    settings_section = ""
    if current:
        build_ready = _documents_ready_for_knowledge_build(
            documents, strategy_key,
        )
        preparation_pending = int(preparation_status.get("pending_count") or 0) > 0
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
              {'disabled' if not native_run or not (native_status or {}).get('fresh') else ''}>查看构建产物</button>"""
            if strategy_key == GRAPHRAG_NATIVE_STRATEGY else ""
        )
        preparation_controls = _construction_controls(
            project_id=int(current["id"]),
            strategy_key=strategy_key,
            latest_job_id=preparation_job_id,
            latest_batch_id=preparation_batch_id,
            active_job_id=active_job_id,
            active_job_kind=active_job_kind,
            active_batch_id=active_batch_id,
            preparation_pending=preparation_pending,
            build_ready=build_ready,
            scope="preparation",
        )
        build_controls = _construction_controls(
            project_id=int(current["id"]),
            strategy_key=strategy_key,
            latest_job_id=build_job_id,
            latest_batch_id=build_batch_id,
            active_job_id=active_job_id,
            active_job_kind=active_job_kind,
            active_batch_id=active_batch_id,
            build_ready=build_ready,
            scope="graph",
            after_build=native_artifact_control,
        )
        graph_update_notice = (
            _knowledge_forge_graph_update_notice(
                documents,
                graph_state,
                build_ready=build_ready,
                active_job_id=active_job_id,
                active_job_kind=active_job_kind,
            )
            if strategy_key == KNOWLEDGE_FORGE_STRATEGY else ""
        )
        preparation_update_notice = _preparation_update_notice(
            preparation_status,
            strategy_key=strategy_key,
            active_job_id=active_job_id,
            active_job_kind=active_job_kind,
        )
        strategy_metadata = ""
        stat_labels = ui["stat_labels"]
        assert isinstance(stat_labels, dict)
        updating_statistics = _updating_statistic_keys(
            strategy_key, active_job_kind,
        )
        current_block = f"""<div class="preparation-heading-side">
          {preparation_update_notice}
          <div class="preparation-heading-actions">{preparation_controls}</div>
        </div>"""
        graph_query_actions = f"""<div class="workspace-section graph-query-actions">
          <div class="section-heading graph-query-heading"><div><h2>{_esc(ui['workspace_title'])}</h2>
            <p>{_esc(ui['workspace_description'])}</p></div>{graph_update_notice}</div>
          <div class="stats"><div><b data-stat="documents">{statistics['documents']}</b><span>{_esc(stat_labels['documents'])}</span></div>
            <div><b data-stat="chunks"{' class="is-updating"' if 'chunks' in updating_statistics else ''}>{statistics['chunks']}</b><span>{_esc(stat_labels['chunks'])}</span></div>
            <div><b data-stat="entities"{' class="is-updating"' if 'entities' in updating_statistics else ''}>{statistics['entities']}</b><span>{_esc(stat_labels['entities'])}</span></div>
            <div><b data-stat="relations"{' class="is-updating"' if 'relations' in updating_statistics else ''}>{statistics['relations']}</b><span>{_esc(stat_labels['relations'])}</span></div>
          </div>
          <div class="stats-actions">{build_controls}{strategy_metadata}
            <div class="graph-actions">{graph_retry}</div>
          </div>
        </div>"""
        if strategy_key in {KNOWLEDGE_FORGE_STRATEGY, GRAPHRAG_NATIVE_STRATEGY}:
            embedded_graph_section = _embedded_graph_browser(
                strategy_key, ready=graph_data_ready, active=bool(active_job_id),
            )
        chunk_setting = (
            f"""<label>单切片 Token 上限
            <input name="chunk_max_tokens" type="number" min="64" step="1" required value="{int(current.get('chunk_max_tokens') or 512)}">
            <small id="chunk-token-limit-hint" class="field-help">正在读取 LLM 与 Embedding 的共同上限...</small></label>"""
            if strategy_key == KNOWLEDGE_FORGE_STRATEGY
            else f"""<input name="chunk_max_tokens" type="hidden" value="{int(current.get('chunk_max_tokens') or 512)}">"""
        )
        settings_section = f"""
        <div class="workspace-settings"><div class="section-heading"><div><h2>知识库设置</h2>
          <p>设置当前知识库使用的模型、知识语言与上下文预算；不会修改 Ollama 全局上下文。</p></div>
          <button id="sync-models" class="icon-button refresh-button" type="button"
            title="从 Ollama 同步可用模型" aria-label="从 Ollama 同步可用模型">↻</button></div>
        <p id="model-sync-status" class="model-sync-status" role="status" hidden></p>
        <form id="project-settings-form" method="post" action="/projects/{current['id']}/models" class="form-grid compact">
          <label>LLM<select name="llm_model">{_model_options(models,'llm',current.get('llm_model') or '')}</select></label>
          <label>Embedding<select name="embedding_model">{_model_options(models,'embedding',current.get('embedding_model') or '')}</select></label>
          <label>知识语言<select name="knowledge_language" required
            data-original-value="{_esc(current.get('knowledge_language') or DOCUMENT_LANGUAGE)}">{_knowledge_language_options(str(current.get('knowledge_language') or DOCUMENT_LANGUAGE))}</select>
            <small class="field-help">更改不会改写当前知识库或图谱，将从下一次知识构建生效。</small></label>
          <label>知识库上下文 Token 上限
            <input name="model_max_tokens" type="number" min="1024" step="1" required value="{int(current.get('model_max_tokens') or 32768)}">
            <small id="model-token-limit-hint" class="field-help">正在读取当前 LLM 的上下文上限...</small></label>
          {chunk_setting}
          <div class="form-actions"><button class="button primary" type="submit">保存知识库设置</button></div></form>
        {processing_options}</div>"""
    job_hidden = "" if active_job_id else " hidden"
    notice = _notice(*message) if message else ""
    task_box = f"""<div id="task-box" class="task-box"{job_hidden} data-job="{_esc(active_job_id)}"
      data-batch="{_esc(active_batch_id)}" data-kind="{_esc(active_job_kind)}">
      <div class="task-head"><div><b id="task-title">{_esc(ui['task_title'])}</b>
        <small id="task-summary" class="task-summary" hidden></small></div><span><span id="task-status">等待</span>
        <span id="task-progress" class="task-progress">总进度 0%</span>
        <span id="task-duration" class="task-duration">用时 00:00</span>
        <button id="active-job-log" class="button small task-log-button" type="button"
          data-view-job-log data-job="{_esc(active_job_id)}" data-batch="{_esc(active_batch_id)}">查看详细日志</button></span></div>
      <div id="task-prerequisites" class="task-prerequisites" hidden>
        <i aria-hidden="true">✓</i><span><b>预处理结果与知识结构已就绪</b>
        <small id="task-prerequisite-detail"></small></span>
      </div>
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
    retrieval_section = graph_query_actions + graph_task_slot + native_runtime_section + """<div class="workspace-section" {visible}><div class="section-heading"><div><h2>知识检索</h2><p>输入一个概念、实体名称或问题，系统会同时查找相关原文、实体和已验证关系。</p></div></div>
      <form id="search-form" class="search-form"><input name="query" placeholder="输入关键词、实体名称或检索描述" required><button class="button primary">检索</button></form><div id="search-results"></div>
      </div>{graph}""".format(visible="" if current else "hidden", graph=embedded_graph_section)
    if strategy_key == GRAPHRAG_NATIVE_STRATEGY:
        retrieval_section = graph_query_actions + graph_task_slot + native_runtime_section + """<div class="workspace-section" {visible}><div class="section-heading"><div><h2>GraphRAG 知识问答</h2><p>选择适合的问题类型并输入问题。查询后会同时展示回答和本次实际使用的原文或图谱依据。</p></div></div>
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
    <body class="workspace-body"><main class="shell workspace-shell">
    <header class="workspace-brand-header"><a class="brand-link" href="/" aria-label="返回知识库首页"><h1>Knowledge Forge</h1>
      <p>通用知识库与知识图谱构建平台</p></a>
      <nav class="workspace-global-nav" aria-label="全局功能">
        <a class="button secondary is-current" href="/#knowledge-bases" aria-current="page">知识库</a>
        <a class="button secondary" href="/chat">智能问答</a>
      </nav></header>{notice}
    <div class="workspace-layout">
      <aside class="workspace-sidebar" aria-label="知识库工作区导航">
        <div class="workspace-project-summary"><div class="workspace-project-title">
          <div><span class="workspace-context-label">当前知识库</span>
          <h2 title="{_esc(current['name'] if current else '知识库不存在')}">{_esc(current['name'] if current else '知识库不存在')}</h2></div>
          <span class="workspace-strategy-badge">{_esc(strategy_label)}</span>
        </div></div>
        <nav class="workspace-nav" aria-label="知识库功能">
          <button class="workspace-nav-item" type="button" data-workspace-target="documents"
            aria-controls="workspace-documents" aria-selected="true"><span>01</span><b>文档管理</b><small>上传与文件列表</small></button>
          <button class="workspace-nav-item" type="button" data-workspace-target="build"
            aria-controls="workspace-build" aria-selected="false"><span>02</span><b>{_esc(ui['preparation_navigation_label'])}</b><small>{_esc(ui['preparation_navigation_hint'])}</small></button>
          <button class="workspace-nav-item" type="button" data-workspace-target="query"
            aria-controls="workspace-query" aria-selected="false"><span>03</span><b>{_esc(query_navigation_label)}</b><small>{_esc(ui['query_navigation_hint'])}</small></button>
          <button class="workspace-nav-item" type="button" data-workspace-target="settings"
            aria-controls="workspace-settings" aria-selected="false"><span>04</span><b>知识库设置</b><small>模型与上下文</small></button>
        </nav>
      </aside>
      <div class="workspace-content">
        <section id="workspace-documents" class="workspace-view" data-workspace-panel="documents"
          role="tabpanel" aria-label="文档管理">
          <div class="section-heading workspace-view-heading"><div>
            <h2>上传与管理知识库资料</h2><p>上传源文件并查看处理状态；解析与分块方式统一在知识库设置中调整。</p></div></div>
          <form id="document-upload-form" action="/upload" method="post" enctype="multipart/form-data" class="upload-form processing-form">
            <input type="hidden" name="project_id" value="{_esc(current['id'] if current else '')}">
            <input type="hidden" name="strategy_key" value="{_esc(strategy_key)}">
            <div class="upload-line"><input id="upload-files" type="file" name="files" multiple
              accept=".docx,.pdf,.md,.markdown,.txt,.xlsx,.xls,.csv" required>
              <button class="button primary" type="submit">上传文件</button></div>
            <p class="upload-help">支持多文件上传，单个文件最大 {Config.upload_max_bytes // (1024 * 1024)} MB。</p>
          </form>
          <div class="workspace-section" {'' if current else 'hidden'}><div class="section-heading"><div>
            <h2>知识库文档</h2><p>源文件、处理状态和来源链接</p></div></div>
            <div class="table-wrap document-table-wrap"><table class="document-table"><thead><tr><th>文件</th><th>大小</th><th>状态</th><th>更新时间</th>
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
          role="tabpanel" aria-label="知识库设置" hidden>
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
    <dialog id="chunk-dialog" class="structure-dialog"><div class="dialog-heading"><div><h2>知识块</h2><p id="chunk-summary">正在读取分块结果...</p></div>
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
    <dialog id="document-media-dialog" class="document-media-dialog">
      <div class="dialog-heading"><div><h2 id="document-media-title">图表预览</h2><p id="document-media-summary"></p></div>
        <button class="icon-button" type="button" data-close-document-media aria-label="关闭">×</button></div>
      <div id="document-media-body" class="document-media-body"></div>
      <div class="dialog-actions"><button class="button secondary" type="button" data-close-document-media>关闭</button></div>
    </dialog>
    <dialog id="native-artifact-dialog" class="structure-dialog"><div class="dialog-heading"><div>
      <h2 id="native-artifact-title">GraphRAG 构建产物</h2>
      <p id="native-artifact-summary">正在读取...</p></div>
      <button class="icon-button" type="button" data-close-native-artifacts aria-label="关闭">×</button></div>
      <div id="native-artifact-error" class="job-log-error" hidden></div>
      <div id="native-artifact-tools" class="structure-tools">
        <select id="native-artifact-kind" aria-label="选择 GraphRAG Native 产物">
          <option value="documents">本次构建使用的结构化文档</option>
          <option value="source-map">原文位置映射</option>
          <option value="text-units">文本单元</option>
          <option value="entities">实体</option>
          <option value="relationships">关系</option>
          <option value="communities">社区</option>
          <option value="community-reports">社区报告</option>
        </select>
      </div>
      <pre id="native-artifact-content" class="artifact-content"></pre>
      <div class="dialog-actions"><button class="button secondary" type="button" data-close-native-artifacts>关闭</button></div>
    </dialog>
    </main><script src="/static/graph-explorer.js?v=20260826-graph-ui-39-topic-links"></script><script>window.KF={{projectId:{int(current['id']) if current else 0},strategyKey:{json.dumps(strategy_key)},taskProfile:{json.dumps(ui['task_profile'])},taskLabels:{json.dumps(ui['task_labels'], ensure_ascii=False)},activeJob:{str(bool(active_job_id)).lower()},activeJobKind:{json.dumps(active_job_kind)},docling:{str(docling_available()).lower()}}};{SCRIPT}</script></body></html>"""


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
const typeLabels={GenericEntity:'通用实体',Person:'人物',Organization:'组织机构',Location:'地点',Product:'产品或材料',Technology:'技术或方法',Event:'事件'};
const displayType=value=>typeLabels[String(value??'').trim()]||String(value??'').trim()||'通用实体';
const state={graph:null,communities:[],reports:[],nodeById:new Map(),titleToNode:new Map(),communityByNode:new Map(),communityMembershipByNode:new Map(),selectedCommunity:'',query:'',showIsolates:true,selectedNodeId:'',selectedEdgeIndex:-1,positions:new Map()};
const $=id=>document.getElementById(id);
const esc=value=>String(value??'').replace(/[&<>'"]/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
const text=value=>String(value??'').replace(/\s+/g,' ').trim();
const short=(value,length=180)=>{const valueText=text(value);return valueText.length>length?valueText.slice(0,length-1)+'…':valueText};
const communityKey=item=>String(item.level??'?')+':'+String(item.community??item.human_readable_id??item.id??'unknown');
const communityLabel=item=>'社区 '+String(item.display_index??item.community??item.human_readable_id??item.id??'—');
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
    const title=group.label+' ('+group.count+')';
    if(group.key==='unassigned'){
      const label=createSvg('text',{x:group.x,y:group.y,class:'community-label',fill:colorFor(group.key)});label.textContent=title;groups.append(label);continue;
    }
    const trigger=createSvg('g',{class:'community-trigger',transform:'translate('+Math.max(0,group.x-5)+' '+Math.max(0,group.y-18)+')',role:'button',tabindex:'0','aria-label':'筛选'+group.label});
    const background=createSvg('rect',{x:0,y:0,width:Math.max(92,title.length*13+18),height:25,rx:6});
    const label=createSvg('text',{x:8,y:17,class:'community-label',fill:colorFor(group.key)});label.textContent=title;
    const selectCommunity=()=>{state.selectedCommunity=state.selectedCommunity===group.key?'':group.key;$('community-filter').value=state.selectedCommunity;state.selectedNodeId='';state.selectedEdgeIndex=-1;renderGraph();renderDetails();renderCommunities()};
    trigger.addEventListener('click',event=>{event.stopPropagation();selectCommunity()});
    trigger.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectCommunity()}});
    trigger.append(background,label);groups.append(trigger);
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
    panel.innerHTML='<h3>关系详情</h3><dl><dt>源实体</dt><dd>'+esc(edge.source)+'</dd><dt>目标实体</dt><dd>'+esc(edge.target)+'</dd><dt>权重</dt><dd>'+esc(edge.weight)+'</dd><dt>关系描述</dt><dd>'+esc(edge.description)+'</dd><dt>文本单元证据</dt><dd>'+evidenceList(edge.text_unit_ids)+'</dd></dl>';
    return;
  }
  const node=state.nodeById.get(state.selectedNodeId);
  if(!node){panel.innerHTML='<h3>浏览说明</h3><p>圆点表示实体，箭头表示关系。颜色按 GraphRAG 社区划分；点击实体或关系可查看描述和文本单元证据。</p>';return;}
  const community=nodeCommunity(node);
  const neighbours=visibleGraph().edges.filter(edge=>String(edge.source)===String(node.title)||String(edge.target)===String(node.title));
  panel.innerHTML='<h3>'+esc(node.title)+'</h3><dl><dt>类型</dt><dd>'+esc(node.type||'—')+'</dd><dt>度数</dt><dd>'+esc(node.degree||0)+'</dd><dt>社区</dt><dd>'+esc(community?.label||'未归入社区')+'</dd><dt>实体描述</dt><dd>'+esc(short(node.description,800)||'—')+'</dd></dl><h4>相邻关系 ('+neighbours.length+')</h4><ol class="neighbour-list">'+neighbours.slice(0,12).map(edge=>'<li><button type="button" data-detail-edge="'+esc(edge.id)+'">'+esc(edge.source)+' → '+esc(edge.target)+'</button></li>').join('')+'</ol>';
  for(const button of panel.querySelectorAll('[data-detail-edge]'))button.addEventListener('click',()=>{
    const index=visibleGraph().edges.findIndex(edge=>String(edge.id)===button.dataset.detailEdge);state.selectedEdgeIndex=index;state.selectedNodeId='';renderGraph();renderDetails();
  });
}
function renderCommunities(){
  const box=$('community-list');
  const visible=state.selectedCommunity?state.communities.filter(community=>communityKey(community)===state.selectedCommunity):state.communities;
  box.innerHTML=visible.map(community=>{
    const report=reportsFor(community)[0]||{};
    const active=communityKey(community)===state.selectedCommunity?' active':'';
    return '<button type="button" class="community-card'+active+'" data-community="'+esc(communityKey(community))+'"><span class="community-dot" style="background:'+colorFor(communityKey(community))+'"></span><b>'+esc(communityLabel(community))+'</b><small>'+esc(community.size??(community.entity_ids||[]).length)+' 个实体 · '+esc((community.relationship_ids||[]).length)+' 条关系'+(Number(community.level||0)>0?' · 聚类层级 '+esc(community.level):'')+'</small><em>'+esc(short(report.summary??report.full_content??report.description??'暂无社区摘要',220))+'</em></button>';
  }).join('')||'<p>当前知识图谱中没有社区记录。</p>';
  for(const button of box.querySelectorAll('[data-community]'))button.addEventListener('click',()=>{
    state.selectedCommunity=state.selectedCommunity===button.dataset.community?'':button.dataset.community;$('community-filter').value=state.selectedCommunity;state.selectedNodeId='';state.selectedEdgeIndex=-1;renderGraph();renderDetails();renderCommunities();
  });
}
function renderAudit(edges){
  $('audit-summary').textContent='显示 '+edges.length+' / '+String(state.graph?.total_relationships??edges.length)+' 条关系；表格用于审计，网络图用于探索连接。';
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
    if(!graphResponse.ok)throw Error(graph.error||'读取知识图谱失败');if(!communitiesResponse.ok)throw Error(communities.error||'读取社区失败');if(!reportsResponse.ok)throw Error(reports.error||'读取社区报告失败');
    for(const node of graph.nodes||[])node.type=displayType(node.type);
    state.graph=graph;state.communities=(communities.items||[]).slice().sort((left,right)=>Number(left.level??0)-Number(right.level??0)||Number(left.community??0)-Number(right.community??0));state.communities.forEach((item,index)=>{item.display_index=index});state.reports=reports.items||[];
    for(const node of graph.nodes||[]){state.nodeById.set(String(node.id),node);if(!state.titleToNode.has(String(node.title??'')))state.titleToNode.set(String(node.title??''),node);}
    for(const community of state.communities){const entry={key:communityKey(community),label:communityLabel(community),raw:community};for(const entityId of community.entity_ids||[]){const nodeId=String(entityId);if(!state.communityMembershipByNode.has(nodeId))state.communityMembershipByNode.set(nodeId,new Map());state.communityMembershipByNode.get(nodeId).set(entry.key,entry);if(!state.communityByNode.has(nodeId))state.communityByNode.set(nodeId,entry);}}
    renderControls();renderGraph();renderDetails();renderCommunities();$('loading').hidden=true;$('viewer').hidden=false;
  }catch(error){$('loading').textContent='无法读取当前知识图谱：'+(error.message||error);}
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
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#172033;background:#f4f7fb}body{margin:0}.page{max-width:1480px;margin:0 auto;padding:28px}.top{display:flex;justify-content:space-between;gap:20px;align-items:start;flex-wrap:wrap}.top h1{margin:0;font-size:25px}.top p{color:#536981;margin:.4rem 0 0}.tabs,.controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.tab,button{font:inherit}.tab,.controls button{border:1px solid #c5d4e5;border-radius:6px;background:#fff;padding:8px 12px;color:#294764;cursor:pointer}.tab.active{background:#2677d8;border-color:#2677d8;color:#fff}.controls{margin:18px 0}.controls input,.controls select{border:1px solid #b9cade;border-radius:6px;padding:9px 10px;font:inherit;background:#fff}.controls input{min-width:260px;flex:1}.controls label{display:flex;align-items:center;gap:6px;color:#435b75}.network-layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:16px}.card{background:#fff;border:1px solid #d7e2ee;border-radius:9px;box-shadow:0 8px 20px rgba(25,53,84,.05)}.graph-card{min-height:740px;overflow:auto}.graph-card svg{display:block;width:100%;min-width:880px;min-height:720px;background:linear-gradient(145deg,#fbfdff,#eef5fb)}.community-label{font-size:13px;font-weight:700}.community-trigger{cursor:pointer}.community-trigger rect{fill:#fff;fill-opacity:.9;stroke:#c6d5e4}.community-trigger:hover rect,.community-trigger:focus rect{fill:#eef6ff;stroke:#2677d8}.community-trigger:focus{outline:none}.native-edge{stroke:#7f94aa;stroke-opacity:.72;cursor:pointer}.native-edge.dim,.native-node.dim{opacity:.12}.native-node{cursor:pointer}.native-node circle{stroke:#fff;stroke-width:2}.native-node.selected circle{stroke:#142b45;stroke-width:4}.node-label{font-size:10px;text-anchor:middle;fill:#273e57;pointer-events:none}.side{padding:17px;min-height:280px}.side h3{margin:0 0 12px}.side h4{margin:18px 0 8px}.side dl{margin:0;display:grid;grid-template-columns:88px 1fr;gap:8px;font-size:13px}.side dt{color:#64778a;font-weight:700}.side dd{margin:0;overflow-wrap:anywhere;line-height:1.45}.side code,.audit code{display:inline-block;max-width:100%;overflow-wrap:anywhere;font-size:11px;color:#465c73}.neighbour-list{padding-left:18px;margin:0}.neighbour-list button{padding:0;border:0;background:transparent;color:#1766b5;text-align:left;cursor:pointer;font-size:12px}.summary{color:#60758b;font-size:13px;margin:0 0 12px}.communities{margin-top:18px;padding:17px}.communities h2,.audit h2{font-size:18px;margin:0 0 6px}.community-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:10px}.community-card{border:1px solid #d6e2ed;background:#fff;border-radius:7px;padding:12px;text-align:left;cursor:pointer;color:#243d57}.community-card.active{border-color:#2677d8;box-shadow:0 0 0 2px #d8eaff}.community-card b,.community-card small,.community-card em{display:block}.community-card small{color:#6a7f94;margin:5px 0}.community-card em{font-size:12px;color:#526a81;font-style:normal;line-height:1.4}.community-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}.audit{margin-top:18px;padding:17px;overflow:auto}.audit table{border-collapse:collapse;width:100%;min-width:980px}.audit th,.audit td{padding:10px;border-bottom:1px solid #e0e8f0;vertical-align:top;text-align:left}.audit th{background:#f5f8fc}.audit td:nth-child(2){max-width:460px;line-height:1.35}@media(max-width:900px){.page{padding:16px}.network-layout{grid-template-columns:1fr}.graph-card{min-height:520px}.controls input{min-width:180px}.top{flex-direction:column}.graph-card svg{min-width:760px}}
</style></head><body><main class="page"><header class="top"><div><h1>GraphRAG Native 知识图谱浏览</h1><p>运行编号：<code>"""
        + _esc(run["run_id"])
    + """</code> · GraphRAG """
        + _esc(run.get("graphrag_version"))
        + """ · 浏览当前有效的知识图谱、社区与证据。</p></div><div class="tabs"><button class="tab active" type="button" data-native-tab="network">图谱连接</button><button class="tab" type="button" data-native-tab="audit">关系审计表</button></div></header><p id="loading">正在加载实体、关系与社区…</p><section id="viewer" hidden><div class="controls"><input id="entity-filter" type="search" placeholder="输入实体、类型或描述"><button id="apply-filter" type="button">检索</button><select id="community-filter" aria-label="按社区筛选"></select><label><input id="toggle-isolates" type="checkbox" checked>显示孤立实体</label><button id="reset-graph" type="button">重置视图</button></div><section id="network-panel"><div class="network-layout"><div class="card graph-card"><svg id="native-network" viewBox="0 0 1200 760" role="img"><defs><marker id="native-arrow" markerWidth="8" markerHeight="8" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#7f94aa"></path></marker></defs><g id="community-groups"></g><g id="edge-layer"></g><g id="node-layer"></g></svg></div><aside id="selection-details" class="card side"></aside></div><p id="network-count" class="summary"></p><section class="card communities"><h2>GraphRAG 社区</h2><p class="summary">社区由 Leiden 聚类和社区报告生成；点击卡片可筛选图中的成员和关系。</p><div id="community-list" class="community-grid"></div></section></section><section id="audit-panel" class="card audit" hidden><h2>关系审计表</h2><p id="audit-summary" class="summary"></p><table><thead><tr><th>源实体</th><th>关系描述</th><th>目标实体</th><th>权重</th><th>文本单元证据</th></tr></thead><tbody id="audit-rows"></tbody></table></section></section></main><script>"""
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

    def _respond(self, status: int, body: str | bytes, content_type: str = "text/html; charset=utf-8", cookie: str = "", cache_control: str = "") -> None:
        payload = body.encode() if isinstance(body, str) else body
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(payload)))
        if cookie: self.send_header("Set-Cookie", cookie)
        if cache_control: self.send_header("Cache-Control", cache_control)
        self.end_headers(); self.wfile.write(payload)

    def _json(self, status: int, data: Any, *, cookie: str = "") -> None:
        self._respond(
            status, json.dumps(data, ensure_ascii=False, default=str),
            "application/json; charset=utf-8", cookie=cookie,
            cache_control="no-store",
        )

    def _agent_session(self, *, create: bool = False) -> tuple[str, str]:
        header = str(self.headers.get("X-KF-Agent-Session") or "").strip()
        cookie = SimpleCookie(self.headers.get("Cookie"))
        item = cookie.get("knowledge_forge_agent_session")
        value = header or (str(item.value) if item else "")
        try:
            return validate_session_key(value), ""
        except PermissionError:
            if not create:
                raise
        value = new_session_key()
        return value, (
            "knowledge_forge_agent_session=" + value
            + "; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000"
        )

    def _request_data(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        if length > Config.reranker_max_body_bytes:
            raise ValueError("请求内容过大")
        raw = self.rfile.read(length)
        content_type = str(self.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("请求 JSON 必须是对象")
            return value
        return {
            key: values[0] for key, values in
            parse_qs(raw.decode("utf-8", "ignore")).items()
        }

    def _agent_sse(self, run_id: str, *, session_key: str, after_id: int) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        cursor = max(0, int(after_id))
        deadline = time.monotonic() + Config.agent_sse_max_seconds
        heartbeat = time.monotonic()
        terminal = {"completed", "failed", "cancelled"}
        try:
            while time.monotonic() < deadline:
                rows = SERVICE.agent_events(
                    run_id, session_key=session_key, after_id=cursor,
                )
                for row in rows:
                    cursor = int(row["id"])
                    event_type = str(row.get("event_type") or "message")
                    payload = json.dumps(
                        row.get("payload") or {}, ensure_ascii=False, default=str,
                    )
                    block = (
                        f"id: {cursor}\n"
                        f"event: {event_type}\n"
                        f"data: {payload}\n\n"
                    ).encode("utf-8")
                    self.wfile.write(block)
                    self.wfile.flush()
                    if event_type in terminal:
                        self.close_connection = True
                        return
                if time.monotonic() - heartbeat >= 15:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    heartbeat = time.monotonic()
                time.sleep(Config.agent_sse_poll_seconds)
            payload = json.dumps({"reconnect": True}, ensure_ascii=False)
            self.wfile.write(f"event: timeout\ndata: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return
        except Exception as exc:
            try:
                payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
                self.wfile.write(f"event: error\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True
            return

    def _redirect(self, target: str, project_id: int | None = None) -> None:
        self.send_response(HTTPStatus.SEE_OTHER); self.send_header("Location", target)
        if project_id: self.send_header("Set-Cookie", f"knowledge_forge_project={project_id}; Path=/; SameSite=Lax")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {
            "/static/graph-explorer.js",
            "/static/graph-explorer.js.map",
        }:
            name = parsed.path.rsplit("/", 1)[-1]
            path = (Path(__file__).resolve().parent / "static" / name).resolve()
            static_root = (Path(__file__).resolve().parent / "static").resolve()
            if path.parent != static_root or not path.is_file():
                self._respond(404, "静态资源不存在", "text/plain; charset=utf-8")
                return
            content_type = (
                "application/javascript; charset=utf-8"
                if name.endswith(".js") else "application/json; charset=utf-8"
            )
            self._respond(
                200, path.read_bytes(), content_type,
                cache_control="no-store, max-age=0",
            )
            return
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
        if parsed.path == "/chat":
            self._respond(200, render_chat_page())
            return
        if parsed.path == "/api/agent/knowledge-bases":
            try:
                self._json(200, SERVICE.agent_knowledge_bases())
            except Exception as exc:
                self._json(503, {"ok": False, "error": str(exc)})
            return
        if parsed.path == "/api/agent/threads":
            try:
                params = parse_qs(parsed.query)
                project_id = int(params.get("project_id", ["0"])[0] or 0)
                limit = int(params.get("limit", ["100"])[0] or 100)
                if project_id <= 0:
                    raise ValueError("请选择知识库")
                session_key, cookie = self._agent_session(create=True)
                self._json(
                    200,
                    SERVICE.list_agent_threads(
                        project_id, session_key=session_key, limit=limit,
                    ),
                    cookie=cookie,
                )
            except (TypeError, ValueError, LookupError) as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._json(503, {"ok": False, "error": str(exc)})
            return
        match = re.fullmatch(r"/api/agent/threads/(ath_[0-9a-f]{32})", parsed.path)
        if match:
            try:
                session_key, _cookie = self._agent_session()
                self._json(200, SERVICE.get_agent_thread(
                    match.group(1), session_key=session_key,
                ))
            except PermissionError as exc:
                self._json(403, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        match = re.fullmatch(
            r"/api/agent/runs/(arun_[0-9a-f]{32})/events", parsed.path,
        )
        if match:
            try:
                session_key, _cookie = self._agent_session()
                SERVICE.get_agent_run(match.group(1), session_key=session_key)
                params = parse_qs(parsed.query)
                last_event = str(self.headers.get("Last-Event-ID") or "")
                after = params.get("after", [last_event or "0"])[0]
                self._agent_sse(
                    match.group(1), session_key=session_key,
                    after_id=int(after or 0),
                )
            except PermissionError as exc:
                self._json(403, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        match = re.fullmatch(
            r"/api/agent/runs/(arun_[0-9a-f]{32})/trace", parsed.path,
        )
        if match:
            try:
                session_key, _cookie = self._agent_session()
                self._json(200, SERVICE.get_agent_run_trace(
                    match.group(1), session_key=session_key,
                ))
            except PermissionError as exc:
                self._json(403, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        match = re.fullmatch(r"/api/agent/runs/(arun_[0-9a-f]{32})", parsed.path)
        if match:
            try:
                session_key, _cookie = self._agent_session()
                self._json(200, SERVICE.get_agent_run(
                    match.group(1), session_key=session_key,
                ))
            except PermissionError as exc:
                self._json(403, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        match = re.fullmatch(r"/api/agent/evidence/(ev_[0-9a-f]{32})", parsed.path)
        if match:
            try:
                session_key, _cookie = self._agent_session()
                self._json(200, SERVICE.get_agent_evidence(
                    match.group(1), session_key=session_key,
                ))
            except PermissionError as exc:
                self._json(403, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._json(400, {"ok": False, "error": str(exc)})
            return
        match = re.fullmatch(r"/api/projects/(\d+)/graph-explorer", parsed.path)
        if match:
            params = parse_qs(parsed.query)
            try:
                explorer_args = dict(
                    strategy_key=params.get("strategy", [""])[0],
                    mode=params.get("mode", ["overview"])[0],
                    community_key=params.get("community", [""])[0],
                    entity_id=params.get("entity", [""])[0],
                    query=params.get("q", [""])[0],
                    limit=int(params.get("limit", ["500"])[0]),
                )
                # Do not require older strategy adapters to accept a new
                # optional parameter when the caller did not request it.
                if "community_level" in params:
                    explorer_args["community_level"] = params["community_level"][0]
                if "overview_relation_scope" in params:
                    explorer_args["overview_relation_scope"] = params[
                        "overview_relation_scope"
                    ][0]
                self._json(200, SERVICE.graph_explorer(
                    int(match.group(1)), **explorer_args,
                ))
            except LookupError as exc:
                self._json(404, {"error": str(exc)})
            except (GraphNotReadyError, GraphRAGNativeStale) as exc:
                self._json(409, {"error": str(exc)})
            except GraphRAGNativeUnavailable as exc:
                self._json(404, {"error": str(exc)})
            except (TypeError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(503, {"error": str(exc)})
            return
        match = re.fullmatch(
            r"/projects/(\d+)/(knowledge-forge|graphrag-native)", parsed.path,
        )
        if match:
            project_id = int(match.group(1))
            strategy_key = (
                GRAPHRAG_NATIVE_STRATEGY
                if match.group(2) == "graphrag-native"
                else KNOWLEDGE_FORGE_STRATEGY
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
                self._respond(
                    404,
                    render_status_page(
                        404, "知识库不存在", str(exc) or "没有找到请求的知识库。",
                    ),
                )
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
                    raise LookupError("知识库不存在")
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
        match = re.fullmatch(
            r"/api/projects/(\d+)/knowledge-forge/community-reports",
            parsed.path,
        )
        if match:
            try:
                self._json(200, SERVICE.knowledge_forge_community_report_status(
                    int(match.group(1)), auto_start=True,
                ))
            except LookupError as exc:
                self._json(404, {"error": str(exc)})
            except GraphNotReadyError as exc:
                self._json(409, {"error": str(exc)})
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
                requested_top_k = int(params.get("top_k", ["0"])[0] or 0)
                self._json(200, SERVICE.search(
                    int(params.get("project", ["0"])[0]),
                    params.get("q", [""])[0],
                    requested_top_k or None,
                ))
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
                    "strategy", [KNOWLEDGE_FORGE_STRATEGY],
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
                    project_id, KNOWLEDGE_FORGE_STRATEGY,
                )
                if strategy_key != KNOWLEDGE_FORGE_STRATEGY:
                    raise ValueError("当前知识库只允许打开 Knowledge Forge 图谱")
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
        self._respond(
            404,
            render_status_page(
                404, "页面不存在", "请求的页面可能已移动，或地址输入有误。",
            ),
        )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/agent/threads":
                data = self._request_data()
                session_key, cookie = self._agent_session(create=True)
                thread = SERVICE.create_agent_thread(
                    int(data.get("project_id") or 0), session_key=session_key,
                )
                self._json(201, {"ok": True, "thread": thread}, cookie=cookie)
                return
            match = re.fullmatch(
                r"/api/agent/threads/(ath_[0-9a-f]{32})/messages", parsed.path,
            )
            if match:
                data = self._request_data()
                session_key, _cookie = self._agent_session()
                result = SERVICE.submit_agent_message(
                    match.group(1), session_key=session_key,
                    query=str(data.get("query") or data.get("content") or ""),
                    requested_mode=str(data.get("mode") or "auto"),
                )
                self._json(202, {"ok": True, **result})
                return
            match = re.fullmatch(
                r"/api/agent/runs/(arun_[0-9a-f]{32})/cancel", parsed.path,
            )
            if match:
                session_key, _cookie = self._agent_session()
                result = SERVICE.cancel_agent_run(
                    match.group(1), session_key=session_key,
                )
                self._json(202, {"ok": True, "run": result})
                return
            match = re.fullmatch(
                r"/api/agent/runs/(arun_[0-9a-f]{32})/feedback", parsed.path,
            )
            if match:
                data = self._request_data()
                session_key, _cookie = self._agent_session()
                result = SERVICE.record_agent_feedback(
                    match.group(1), session_key=session_key,
                    rating=int(data.get("rating") or 0),
                    comment=str(data.get("comment") or ""),
                    action=str(data.get("action") or "answer_rating"),
                )
                self._json(200, {"ok": True, "feedback": result})
                return
            if parsed.path == "/projects":
                data = self._form()
                project = SERVICE.create_project(
                    data.get("name", ""), data.get("description", ""),
                    data.get("llm_model", ""), data.get("embedding_model", ""),
                    strategy_key=data.get("strategy_key", ""),
                    knowledge_language=data.get("knowledge_language", CHINESE),
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
                    int(data.get("model_max_tokens", "0")), int(data.get("chunk_max_tokens", "0")),
                    data.get("knowledge_language"))
                strategy_key = SERVICE._assert_project_strategy(int(project["id"]))
                self._redirect(
                    _workspace_path(int(project["id"]), strategy_key)
                    + f"?kind=ok&message={quote('知识库设置已更新')}#settings",
                    int(project["id"]),
                )
                return
            if parsed.path == "/upload": self._upload(); return
            if parsed.path == "/api/models/sync":
                models = SERVICE.sync_ollama_models(); self._json(200, {"ok": True, "count": len(models), "models": models}); return
            match = re.fullmatch(r"/api/projects/(\d+)/jobs/cancel", parsed.path)
            if match:
                data = self._form()
                try:
                    result = SERVICE.request_project_job_cancellation(
                        int(match.group(1)),
                        job_id=str(data.get("job_id") or ""),
                        batch_id=str(data.get("batch_id") or ""),
                        expected_kind=str(data.get("kind") or ""),
                    )
                    self._json(202, {"ok": True, **result})
                except JobPublicationCommitted as exc:
                    self._json(409, {"ok": False, "error": str(exc)})
                except LookupError as exc:
                    self._json(404, {"ok": False, "error": str(exc)})
                except ValueError as exc:
                    self._json(409, {"ok": False, "error": str(exc)})
                return
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
                    int(match.group(1)), strategy_key=KNOWLEDGE_FORGE_STRATEGY,
                )
                self._json(202, {"ok": True, **result})
                return
            match = re.fullmatch(
                r"/api/projects/(\d+)/knowledge-forge/community-reports/build",
                parsed.path,
            )
            if match:
                data = self._form()
                retry_failed = str(data.get("retry_failed") or "").lower() in {
                    "1", "true", "yes", "on",
                }
                result = SERVICE.submit_knowledge_forge_community_reports(
                    int(match.group(1)), retry_failed=retry_failed,
                )
                self._json(202, {"ok": True, **result})
                return
            match = re.fullmatch(
                r"/api/projects/(\d+)/knowledge-forge/community-reports/(.+)/prioritize",
                parsed.path,
            )
            if match:
                result = SERVICE.prioritize_knowledge_forge_community_report(
                    int(match.group(1)), unquote(match.group(2)),
                )
                self._json(202, result)
                return
            match = re.fullmatch(
                r"/api/projects/(\d+)/(knowledge-forge|graphrag-native)/prepare",
                parsed.path,
            )
            if match:
                expected = (
                    GRAPHRAG_NATIVE_STRATEGY
                    if match.group(2) == "graphrag-native"
                    else KNOWLEDGE_FORGE_STRATEGY
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
        except PermissionError as exc:
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": str(exc)})
        except LookupError as exc:
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
        except (GraphNotReadyError, GraphRAGNativeStale) as exc:
            self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
        except TreatmentMaterializationRequired as exc:
            self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
        except Exception as exc:
            LOGGER.exception("Request failed: %s", self.path)
            if self.path.startswith("/api/"): self._json(400, {"ok": False, "error": str(exc)})
            else: self._redirect(f"/?kind=error&message={quote(str(exc))}", self._project())

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        match = re.fullmatch(r"/api/agent/threads/(ath_[0-9a-f]{32})", parsed.path)
        if not match:
            self._json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "接口不存在"})
            return
        try:
            session_key, _cookie = self._agent_session()
            thread = SERVICE.archive_agent_thread(
                match.group(1), session_key=session_key,
            )
            self._json(HTTPStatus.OK, {"ok": True, "thread": thread})
        except PermissionError as exc:
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "error": str(exc)})
        except RuntimeError as exc:
            self._json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
        except Exception as exc:
            LOGGER.exception("Request failed: %s", self.path)
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

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
            raise LookupError("知识库不存在")
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
.task-head>div{min-width:0}.task-head>span{display:flex;align-items:center;gap:12px;flex:0 0 auto}.task-summary{display:block;margin-top:4px;color:#9eb3cc;font-size:12px;font-weight:400;line-height:1.4;overflow-wrap:anywhere}.task-duration{color:#9eb3cc;font-variant-numeric:tabular-nums}.task-log-button{border:1px solid #50647e;background:#1a2a3f;color:#d9e8fb}.task-log-button:hover{background:#263b55}.button.safe-stop{background:#fff;color:#a33b3f;border:1px solid #d9a6a9}.button.safe-stop:hover{background:#fff4f4}.button.safe-stop:disabled{background:#f4f5f7;color:#8793a2;border-color:#d8e0e9}.project-heading{margin:0 0 18px}.project-heading h2{font-size:24px}.project-heading .project-picker select,.project-heading .project-picker .button{font-size:17px}
.native-runtime-details{background:#fbfdff}.native-runtime-status{display:flex;justify-content:space-between;align-items:center;margin-top:14px;color:#315a83;font-size:14px}.native-runtime-status b{color:#203f63}.native-runtime-progress{margin-top:8px;background:#dce8f5}.native-runtime-progress i{background:#4ca36f}
.stats-actions{justify-content:flex-start;flex-wrap:nowrap;overflow-x:auto;padding-bottom:2px}.schema-actions,.graph-actions{display:flex;align-items:center;gap:8px;flex:0 0 auto}.stats-actions .schema-chip{padding:9px 12px;font-size:16px;font-weight:700}
.dialog-title-actions{display:flex;align-items:center;gap:8px}.model-sync-status{margin:14px 0 0;font-size:14px}.model-sync-status.success{color:#20773b}.model-sync-status.failure{color:#b72d35}.field-help{display:block;min-height:18px;color:#6b7e93;font-size:12px;line-height:1.5}.field-help.failure{color:#b72d35}.construction-controls{display:flex;align-items:center;justify-content:flex-end;gap:8px;flex:1 1 auto;flex-wrap:nowrap;margin:0}.construction-controls form{margin:0}.construction-controls label{flex-direction:row;align-items:center;gap:8px;white-space:nowrap}.construction-controls select{min-width:160px;padding:8px 10px}
.delete-progress-dialog{width:min(520px,calc(100% - 28px))}.delete-progress-body{display:flex;align-items:center;gap:16px;margin:22px 0 16px;padding:16px;border:1px solid #d6e1ec;border-radius:6px;background:#f8fbff}.delete-progress-body p{margin-top:4px}.delete-spinner{flex:0 0 30px;width:30px;height:30px;border:3px solid #cfdae7;border-top-color:#2677d8;border-radius:50%;animation:delete-spin .8s linear infinite}.delete-result{display:grid;place-items:center;flex:0 0 30px;width:30px;height:30px;border-radius:50%;color:#fff;font-size:20px;font-weight:700}.delete-result.success{background:#23a453}.delete-result.failure{background:#d4383f}.delete-progress-bar{margin:0 0 18px}.delete-progress-bar i{background:#2677d8}.deleting-row{opacity:.48;pointer-events:none}@keyframes delete-spin{to{transform:rotate(360deg)}}
.processing-form{display:flex;flex-direction:column;align-items:stretch}.upload-line{display:flex;align-items:center;gap:10px}.upload-line input{flex:1}.upload-help{margin:7px 0 0;color:#6b7e93;font-size:13px}.processing-options{border:1px solid #d7e2ed;border-radius:6px;padding:10px 12px;background:#f8fbff}.processing-options summary{font-size:14px}.processing-options .form-grid{margin-top:12px;max-width:none}.processing-options .muted{margin-top:10px}
.task-prerequisites{display:flex;align-items:center;gap:9px;margin-top:13px;padding:9px 11px;border:1px solid #29425f;border-radius:6px;background:#15253a}.task-prerequisites[hidden]{display:none}.task-prerequisites>i{display:grid;place-items:center;flex:0 0 21px;width:21px;height:21px;border-radius:50%;background:#2f9b62;color:#fff;font:700 12px/1 Arial,sans-serif}.task-prerequisites span,.task-prerequisites b,.task-prerequisites small{display:block}.task-prerequisites b{color:#cfe1f3;font-size:12px}.task-prerequisites small{margin-top:3px;color:#8fa5bd;font-size:11px;line-height:1.4}.task-steps{display:grid;gap:13px;margin-top:15px}.task-step{display:grid;grid-template-columns:minmax(0,390px) minmax(160px,1fr);align-items:center;gap:16px}.task-step[hidden]{display:none}.task-step-head{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:start;gap:12px;font-size:13px}.task-step-head>div{min-width:0}.task-step-head span{display:block;color:#d9e8fb;font-weight:700}.task-step-head small{display:block;margin-top:4px;color:#8fa5bd;font-size:11px;line-height:1.4}.task-step-head .task-step-detail{color:#a9c5e4;overflow-wrap:anywhere}.task-step-head b{min-width:0;color:#8fa5bd;font-weight:400;white-space:nowrap}.task-step .progress{margin:0;height:7px;border-radius:5px;overflow:hidden}.task-step.running .task-step-head b{color:#72b8ff}.task-step.running .task-step-head span{color:#fff}.task-step.completed .task-step-head b{color:#75d69a}.task-step.completed .progress i{background:#38ae68}.task-step.failed .task-step-head b,.task-step.cancelled .task-step-head b{color:#ff9297}.task-step.failed .progress i,.task-step.cancelled .progress i{background:#df5960}.stats b.is-updating::after{content:"";display:inline-block;width:12px;height:12px;margin-left:8px;border:2px solid #c6d5e5;border-top-color:#2677d8;border-radius:50%;animation:stat-spin .8s linear infinite;vertical-align:1px}.stats b.is-updating+span::after{content:" · 更新中";color:#2677d8}@keyframes stat-spin{to{transform:rotate(360deg)}}
.disabled-link{opacity:.45;pointer-events:none;cursor:not-allowed}.status-queued{background:#eef3ff;color:#315a83}.status-cancelled{background:#f1f3f5;color:#667789}.chunk-badges{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:9px}.chunk-badge{padding:3px 7px;border-radius:4px;background:#eef3f8;color:#496079;font-size:12px}.chunk-badge.graph{background:#e8f7ec;color:#20773b}.chunk-badge.skipped{background:#fff3df;color:#8b6400}
.structure-dialog{width:min(980px,calc(100% - 28px))}.structure-tools{display:grid;grid-template-columns:minmax(0,1fr) 210px;gap:10px;margin:16px 0}.structure-tools select{min-width:0}.structure-list{max-height:58vh;overflow:auto;border:1px solid #d6e1ec;border-radius:6px}.structure-row{padding:14px 16px;border-bottom:1px solid #e1e9f1}.structure-row:last-child{border-bottom:0}.structure-row-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:9px}.structure-row-meta{display:flex;align-items:center;flex-wrap:wrap;gap:7px;color:#637891;font-size:12px}.block-type{display:inline-block;padding:3px 7px;border-radius:4px;background:#eaf3fc;color:#245d94;font-weight:700}.structure-text{margin:0;color:#243a55;white-space:pre-wrap;word-break:break-word;font:14px/1.65 Arial,"Microsoft YaHei",sans-serif}.structure-metadata{margin-top:10px}.structure-metadata summary{font-size:12px;font-weight:400}.structure-metadata pre{margin:8px 0 0;padding:10px;background:#f6f9fc;border-radius:4px;overflow:auto;font-size:12px}.structure-pager{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:14px 0}.structure-pager>div{display:flex;gap:8px}
.home-shell{max-width:1240px}.home-header{padding:2px 3px 0}.home-header p{margin-top:4px}.home-primary-actions{display:flex;align-items:center;gap:8px}.home-primary-actions .button{min-width:102px;text-align:center}.chat-entry{min-height:420px;padding:42px;background:linear-gradient(135deg,#fff 0%,#f4f9ff 100%)}.chat-entry h2{margin-top:9px;color:#173a62;font-size:30px}.chat-entry>p{max-width:760px;font-size:15px}.chat-entry-status{display:flex;align-items:center;gap:14px;max-width:760px;margin-top:28px;padding:16px 18px;border:1px solid #cbdbea;border-radius:8px;background:#f8fbff}.chat-entry-status b{white-space:nowrap;color:#245d94}.chat-entry-status span{color:#5d7088;font-size:13px;line-height:1.55}.home-hero{position:relative;overflow:hidden;padding:30px;background:linear-gradient(135deg,#fff 0%,#f4f9ff 58%,#e9f5ff 100%);border-color:#bfd5ec}.home-hero:after{content:"";position:absolute;right:-95px;top:-120px;width:300px;height:300px;border:46px solid rgba(38,119,216,.055);border-radius:50%;pointer-events:none}.hero-layout{position:relative;z-index:1;display:grid;grid-template-columns:minmax(0,1.45fr) minmax(310px,.8fr);gap:34px;align-items:start}.hero-kicker,.section-kicker{display:inline-block;color:#1769c2;font-size:12px;font-weight:800;letter-spacing:.08em}.hero-copy h2{max-width:760px;margin-top:9px;color:#173a62;font-size:30px;line-height:1.25}.hero-copy p{max-width:780px;color:#465e79;font-size:15px}.hero-points{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.hero-points div{min-height:104px;padding:15px;border:1px solid rgba(159,190,222,.75);border-radius:8px;background:rgba(255,255,255,.82);box-shadow:0 8px 20px rgba(31,79,126,.06)}.hero-points b,.hero-points span{display:block}.hero-points b{margin-bottom:7px;color:#245d94;font-size:14px}.hero-points span{color:#60758d;font-size:12px;line-height:1.55}.home-flow{position:relative;z-index:1;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:26px 0 0;padding:17px 0 0;border-top:1px solid #d5e4f2;list-style:none}.home-flow li{display:flex;align-items:center;gap:10px;padding:0 10px}.home-flow li+li{border-left:1px solid #d5e4f2}.home-flow li>b{display:grid;place-items:center;flex:0 0 30px;width:30px;height:30px;border-radius:50%;background:#2677d8;color:#fff;font-size:13px}.home-flow span,.home-flow strong,.home-flow small{display:block}.home-flow strong{color:#263f60;font-size:14px}.home-flow small{margin-top:3px;color:#718398}.strategy-overview{padding:24px;background:#fff}.strategy-heading{align-items:flex-end;margin-bottom:17px}.strategy-heading h2{margin-top:7px;font-size:24px}.strategy-heading p{max-width:760px}.strategy-intro-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:17px}.strategy-explainer{display:flex;flex-direction:column;padding:23px;border:1px solid #ccdaea;border-top:4px solid #2677d8;border-radius:9px;background:#fff;box-shadow:0 8px 20px rgba(24,58,98,.045)}.strategy-native{border-top-color:#159b8a}.strategy-card-head{display:flex;align-items:center;gap:12px}.strategy-card-head h3{margin:3px 0 0;color:#183b63;font-size:22px}.strategy-mark{display:grid;place-items:center;flex:0 0 44px;width:44px;height:44px;border-radius:9px;background:#e7f1fd;color:#1769c2;font-size:14px;font-weight:900}.strategy-native .strategy-mark{background:#e3f5f1;color:#08796c}.strategy-tag{color:#6a7e94;font-size:12px;font-weight:700}.strategy-lead{min-height:68px;margin-top:16px;color:#344e6b}.strategy-path{margin-top:13px;padding:11px 12px;border-radius:6px;background:#eef5fc;color:#245d94;font-size:12px;font-weight:700;line-height:1.55}.strategy-native .strategy-path{background:#eaf7f5;color:#08796c}.strategy-details{display:grid;gap:0;margin:17px 0 0}.strategy-details div{display:grid;grid-template-columns:78px minmax(0,1fr);gap:10px;padding:10px 0;border-top:1px solid #e2e9f1}.strategy-details dt{color:#263f60;font-size:13px;font-weight:800}.strategy-details dd{margin:0;color:#5d7088;font-size:13px;line-height:1.55}.strategy-fit{margin-top:auto;padding:14px 15px;border-radius:7px;background:#f7faff}.strategy-fit>b{color:#274c75;font-size:13px}.strategy-fit ul{margin:8px 0 0;padding-left:18px;color:#4c647f;font-size:13px;line-height:1.7}.strategy-tradeoff{margin-top:13px;padding-left:11px;border-left:3px solid #e0a02b;color:#65768a;font-size:12px}.strategy-tradeoff b{color:#8d6417}.choice-guide{display:grid;grid-template-columns:auto repeat(2,minmax(0,1fr));gap:9px 18px;align-items:start;margin-top:17px;padding:16px 18px;border:1px solid #cbdbea;border-radius:8px;background:#f8fbff;color:#49627e;font-size:13px;line-height:1.55}.choice-guide>b{color:#183b63}.choice-guide strong{color:#1769c2}.choice-guide em{grid-column:2/-1;color:#8a681f;font-style:normal}.project-list-dialog{width:min(1100px,calc(100% - 28px));height:min(82vh,880px);max-height:none;overflow:hidden}.project-list-dialog[open]{display:flex;flex-direction:column}.project-list-dialog .project-grid{flex:1;min-height:0;overflow:auto;padding:2px 4px 4px}.project-list-heading{flex:0 0 auto;align-items:flex-end}.project-list-heading h2{margin-top:6px;font-size:23px}.project-list-heading .button{min-width:96px}.strategy-choice-help{margin:0;padding:10px 12px;border:1px solid #d6e3f0;border-radius:6px;background:#f6faff;color:#49627e;font-size:13px}.project-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:18px}.project-card{display:flex;flex-direction:column;justify-content:space-between;gap:18px;padding:18px;border:1px solid #d6e1ec;border-radius:8px;background:#fbfdff}.project-card h3{margin:0;color:#203b5b}.strategy-badge,.fixed-strategy{display:inline-flex;margin-top:10px;padding:6px 9px;border-radius:5px;background:#eaf3fc;color:#245d94;font-size:13px;font-weight:700}.bind-form{display:flex;align-items:center;gap:8px}.bind-form select{min-width:0}.native-config-note{display:flex;flex-direction:column;gap:6px;padding:10px 12px;border:1px solid #d6e1ec;border-radius:6px;background:#fbfdff;color:#5d7088;font-size:13px}.native-config-note b{color:#284a70}.artifact-content{min-height:260px;max-height:58vh;overflow:auto;padding:14px;border-radius:6px;background:#101c2e;color:#d9e8fb;white-space:pre-wrap;word-break:break-word;font:12px/1.55 monospace}
.workspace-shell{max-width:1440px}.workspace-brand-header{margin-bottom:16px}.workspace-layout{display:grid;grid-template-columns:280px minmax(0,1fr);align-items:stretch;border:1px solid #d2deeb;border-radius:10px;background:#fff;box-shadow:0 10px 24px rgba(26,61,101,.06);overflow:hidden}.workspace-sidebar{min-width:0;border-right:1px solid #dce5ee;background:#f9fbfd}.workspace-project-summary{padding:21px 19px 18px;border-bottom:1px solid #dce5ee;background:linear-gradient(145deg,#f9fcff,#edf5fd)}.workspace-project-title{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:start;gap:9px}.workspace-project-summary h2{margin:0;color:#183b63;font-size:20px;line-height:1.3;overflow-wrap:anywhere}.workspace-strategy-badge{display:inline-flex;padding:5px 8px;border-radius:5px;background:#dcecff;color:#1769c2;font-size:12px;font-weight:800}.workspace-nav{display:grid;gap:5px;padding:10px}.workspace-nav-item{display:grid;grid-template-columns:34px minmax(0,1fr);grid-template-rows:auto auto;column-gap:10px;width:100%;padding:12px 11px;border:1px solid transparent;border-radius:7px;background:transparent;color:#415a75;text-align:left;cursor:pointer}.workspace-nav-item>span{grid-row:1/3;display:grid;place-items:center;align-self:center;width:30px;height:30px;border-radius:7px;background:#edf3f9;color:#657b93;font-size:11px;font-weight:800}.workspace-nav-item>b{align-self:end;color:#284765;font-size:14px}.workspace-nav-item>small{align-self:start;margin-top:3px;color:#7a8da1;font-size:11px}.workspace-nav-item:hover{background:#f5f9fd;border-color:#dbe6f1}.workspace-nav-item[aria-selected=true]{border-color:#bad4f1;background:#eaf4ff}.workspace-nav-item[aria-selected=true]>span{background:#2677d8;color:#fff}.workspace-nav-item[aria-selected=true]>b{color:#1769c2}.workspace-nav-item:focus-visible{outline:3px solid #9cc6f4;outline-offset:1px}.workspace-content{min-width:0;background:#fff}.workspace-view{min-height:560px;margin:0;padding:24px}.workspace-view[hidden]{display:none}.workspace-view-kicker{display:inline-block;margin-bottom:7px;color:#2677d8;font-size:12px;font-weight:800;letter-spacing:.08em}.workspace-view-heading{align-items:flex-end}.preparation-view-heading{align-items:flex-start}.workspace-view-heading h2{font-size:23px}.workspace-view>.workspace-section:first-child{margin-top:0;padding-top:0;border-top:0}.workspace-settings .form-grid{max-width:900px;margin-top:22px}.workspace-settings .form-actions{display:flex;justify-content:flex-end}.workspace-settings .form-actions .button{min-width:140px}.workspace-settings .native-config-note{min-height:82px;justify-content:center}.settings-processing{margin-top:30px}.settings-processing-options{margin-top:16px;padding:16px}.settings-processing-options .form-grid{margin-top:0}.workspace-content .graph-summary{margin-top:18px}.workspace-content .native-runtime-details{margin-top:22px}.workspace-content .task-box{margin-top:20px}.workspace-content .stats-actions{align-items:center;flex-wrap:wrap;overflow:visible}.workspace-content .construction-controls{justify-content:flex-start;flex:1 1 560px;flex-wrap:wrap}.preparation-heading-actions{flex:0 0 auto}.workspace-content .preparation-heading-actions .construction-controls{justify-content:flex-end;flex:0 0 auto;flex-wrap:nowrap}.workspace-content .schema-actions,.workspace-content .graph-actions{flex-wrap:wrap}.knowledge-structure-title{display:flex;align-items:center;gap:9px}.schema-version-badge{display:inline-flex;padding:3px 7px;border-radius:12px;background:#eaf3fc;color:#315a83;font-size:11px;font-weight:800}.definition-group{margin-top:20px}.definition-group-heading{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}.definition-group-heading h3{margin:0;color:#234568;font-size:16px}.definition-group-heading span{color:#6b7e93;font-size:12px}.definition-grid{display:grid;gap:10px}.entity-definition-grid{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}.relation-definition-grid{grid-template-columns:repeat(auto-fit,minmax(290px,1fr))}.definition-card{padding:13px 14px;border:1px solid #d7e2ed;border-radius:7px;background:#fbfdff}.definition-card-head{display:flex;align-items:center;justify-content:space-between;gap:10px}.definition-card-head strong{color:#1f456c;overflow-wrap:anywhere}.definition-card-head span{flex:0 0 auto;padding:3px 6px;border-radius:4px;background:#eaf3fc;color:#315a83;font-size:11px}.definition-card p{margin-top:7px;font-size:12px}.definition-card dl{margin:10px 0 0}.definition-card dl>div{display:grid;grid-template-columns:64px minmax(0,1fr);gap:8px;padding-top:7px;border-top:1px solid #e3eaf1;font-size:11px;line-height:1.45}.definition-card dt{color:#536a83;font-weight:700}.definition-card dd{margin:0;color:#314a65}.empty-state{padding:22px;border:1px dashed #cbd8e6;border-radius:7px;color:#708196;text-align:center}.compact-empty{padding:15px}.artifact-table{min-width:760px}.artifact-table .artifact-count{display:block;color:#1f5f9f;font-size:19px}.artifact-table td>small{margin-top:3px}.artifact-table .artifact-actions{white-space:nowrap}.artifact-table .artifact-actions>*+*{margin-left:6px}.graph-query-heading{align-items:flex-start}.graph-query-heading>div:first-child{min-width:0}.graph-update-notice{display:flex;align-items:center;gap:9px;flex:0 1 480px;max-width:480px;padding:9px 10px;border:1px solid #efd18b;border-radius:7px;background:#fff9ea;color:#714f0b}.graph-update-icon{display:grid;place-items:center;flex:0 0 20px;width:20px;height:20px;border-radius:50%;background:#d99416;color:#fff;font-size:12px;font-weight:800}.graph-update-copy{display:flex;flex:1 1 auto;min-width:0;flex-direction:column;gap:2px;line-height:1.35}.graph-update-copy b{font-size:13px}.graph-update-copy small{color:#886b30;font-size:11px}.graph-update-action{flex:0 0 auto;padding:6px 9px;border:1px solid #d7aa4f;border-radius:5px;background:#fff;color:#825b0d;font:inherit;font-size:12px;font-weight:700;cursor:pointer}.graph-update-action:hover{background:#fff2cb}.graph-update-action:focus-visible{outline:3px solid #f0d28f;outline-offset:1px}.graph-update-state{flex:0 0 auto;padding:4px 7px;border-radius:10px;background:#f5e4ba;color:#795716;font-size:11px;font-weight:700}
.preparation-heading-side{display:grid;justify-items:end;gap:8px;max-width:min(620px,60%)}.preparation-update-notice{width:min(480px,100%);max-width:none}.preparation-heading-side .preparation-heading-actions{width:100%}.preparation-heading-side .construction-controls{justify-content:flex-end}
.embedded-graph-browser{padding-bottom:2px}.embedded-graph-tabs{display:inline-flex;gap:2px;padding:3px;border:1px solid #c7d5e5;border-radius:8px;background:#eaf0f7}.embedded-graph-tab{border:0;border-radius:5px;background:transparent;color:#5b6f86;padding:8px 14px;font:inherit;font-size:14px;cursor:pointer}.embedded-graph-tab:hover{color:#174f8d;background:#f5f9fd}.embedded-graph-tab.is-active{background:#fff;color:#1769c2;font-weight:700;box-shadow:0 1px 4px rgba(31,63,99,.18)}.embedded-graph-tab:focus-visible{outline:2px solid #2677d8;outline-offset:2px}.embedded-graph-loading{display:flex;align-items:center;justify-content:center;gap:14px;min-height:130px;margin-top:16px;border:1px dashed #cbd8e6;border-radius:8px;background:#f8fbff}.embedded-graph-loading p{margin-top:4px}.embedded-graph-spinner{width:28px;height:28px;border:3px solid #cfdae7;border-top-color:#2677d8;border-radius:50%;animation:stat-spin .8s linear infinite}.embedded-graph-empty{margin-top:16px}.embedded-graph-controls{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:16px 0}.embedded-graph-controls>input{min-width:260px;flex:1}.embedded-graph-controls>select{min-width:160px;width:auto}.embedded-graph-controls>label{flex-direction:row;align-items:center;gap:6px;white-space:nowrap}.embedded-graph-controls>label input{width:auto}.embedded-graph-zoom{display:inline-flex;align-items:center;gap:5px;padding:3px;border:1px solid #d0dce8;border-radius:7px;background:#f7faff}.embedded-graph-zoom .button{min-width:34px;padding:6px 9px}.embedded-graph-zoom span{min-width:46px;color:#49627d;font-size:12px;text-align:center}.embedded-graph-layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:13px}.embedded-graph-canvas,.embedded-graph-sidebar{border:1px solid #d5e0ed;border-radius:8px;background:#fbfdff}.embedded-graph-canvas{min-width:0;overflow:auto}.embedded-graph-canvas svg{display:block;width:100%;min-width:760px;min-height:620px;background:linear-gradient(145deg,#fbfdff,#eef5fb)}.embedded-graph-canvas marker path{fill:#8298ae}.embedded-graph-edge{stroke:#8398ad;stroke-opacity:.72;cursor:pointer}.embedded-graph-edge.is-dim,.embedded-graph-node.is-dim{opacity:.1}.embedded-graph-edge.is-selected{stroke:#ec7e20;stroke-opacity:1}.embedded-graph-node{cursor:pointer;transition:opacity .15s}.embedded-graph-node circle{stroke:#fff;stroke-width:2}.embedded-graph-node.is-selected circle{stroke:#ec7e20;stroke-width:4}.embedded-graph-node text{font-size:12px;font-weight:600;fill:#29445f;text-anchor:middle;paint-order:stroke;stroke:#fbfdff;stroke-width:3px;stroke-linejoin:round;pointer-events:none}.embedded-graph-group-label{font-size:14px;font-weight:800;paint-order:stroke;stroke:#fbfdff;stroke-width:4px}.embedded-graph-count{margin:0;padding:8px 12px;border-top:1px solid #d5e0ed;font-size:12px}.embedded-graph-sidebar{padding:15px;max-height:665px;overflow:auto}.embedded-graph-sidebar h3{margin:0 0 9px;color:#234568;font-size:16px}.embedded-graph-sidebar h4{margin:16px 0 8px}.embedded-graph-sidebar p{font-size:13px}.embedded-graph-sidebar dl{margin:0}.embedded-graph-sidebar dl>div{padding:7px 0;border-top:1px solid #e1e9f1}.embedded-graph-sidebar dt{color:#6a7f96;font-size:11px}.embedded-graph-sidebar dd{margin:3px 0 0;color:#29445f;font-size:12px;white-space:pre-wrap;word-break:break-word}.graph-entity-heading{display:flex;align-items:center;gap:9px;margin-bottom:10px}.graph-entity-heading h3{margin:0}.graph-entity-dot{display:inline-block;flex:0 0 auto;width:17px;height:17px;border:2px solid #fff;border-radius:50%;box-shadow:0 0 0 1px #aebfd0}.graph-relation-card{display:grid;grid-template-columns:minmax(0,1fr) 74px minmax(0,1fr);align-items:center;gap:7px;width:100%;margin:8px 0;padding:9px;border:1px solid #d7e2ed;border-radius:7px;background:#fff;color:#29445f;text-align:left}.graph-relation-card.button-card{cursor:pointer}.graph-relation-card.button-card:hover{border-color:#82b4e9;background:#f4f9ff}.graph-mini-node{display:flex;align-items:center;gap:6px;min-width:0;font-size:11px;font-weight:700}.graph-mini-node span:last-child{overflow:hidden;text-overflow:ellipsis}.graph-mini-relation{position:relative;color:#315d89;font-size:10px;text-align:center;overflow:hidden;text-overflow:ellipsis}.graph-mini-relation::before{content:"";position:absolute;left:0;right:0;top:50%;border-top:2px solid #91a6ba;z-index:0}.graph-mini-relation b{position:relative;z-index:1;padding:1px 4px;background:#fff;font-weight:600}.graph-mini-relation::after{content:"›";position:absolute;right:-1px;top:calc(50% - 11px);color:#718aa3;font-size:18px}.embedded-graph-community-status{margin-top:18px;padding-top:15px;border-top:1px solid #dce5ee}.embedded-graph-community-status p{margin-top:5px;font-size:12px}.embedded-graph-groups{display:grid;gap:7px;margin-top:10px}.embedded-graph-group-card{display:grid;grid-template-columns:10px minmax(0,1fr);gap:2px 8px;width:100%;padding:9px;border:1px solid #d7e2ed;border-radius:6px;background:#fff;color:#29445f;text-align:left;cursor:pointer}.embedded-graph-group-card.is-active{border-color:#2677d8;background:#eef6ff}.embedded-graph-group-card i{grid-row:1/3;width:9px;height:9px;margin-top:4px;border-radius:50%}.embedded-graph-group-card b{font-size:12px}.embedded-graph-group-card small{color:#70849a}.embedded-graph-group-card span{grid-column:2;margin-top:3px;color:#60758d;font-size:11px;line-height:1.4}.embedded-graph-audit table{min-width:850px}.embedded-graph-audit td{font-size:12px;vertical-align:top}.embedded-graph-audit td:last-child{max-width:360px;white-space:pre-wrap;word-break:break-word}
.workspace-body{overflow:hidden}.workspace-shell{display:flex;flex-direction:column;width:calc(100% - 16px);max-width:none;height:100dvh;margin:0 auto;padding:10px 0}.workspace-brand-header{flex:0 0 auto;margin-bottom:10px}.workspace-layout{flex:1 1 auto;height:auto;min-height:0}.workspace-sidebar{height:100%;overflow-y:auto}.workspace-content{height:100%;min-height:0;overflow:hidden}.workspace-view{height:100%;min-height:0;overflow-y:auto;overscroll-behavior:contain;scrollbar-gutter:stable}
.embedded-graph-group-trigger{cursor:pointer}.embedded-graph-group-trigger rect{fill:#fff;fill-opacity:.9;stroke:#c6d5e4}.embedded-graph-group-trigger:hover rect,.embedded-graph-group-trigger:focus rect{fill:#eef6ff;stroke:#2677d8}.embedded-graph-group-trigger:focus{outline:none}
.workspace-project-title{display:flex;flex-direction:column;align-items:flex-start;gap:8px}.workspace-project-summary h2{width:100%;word-break:normal}.workspace-strategy-badge{white-space:nowrap}.task-head>div{min-width:0}.task-head>span{flex:0 0 auto}.task-summary{display:block;margin-top:4px;color:#9eb3cc;font-size:12px;font-weight:400;line-height:1.4;overflow-wrap:anywhere}.task-progress{color:#72b8ff;font-variant-numeric:tabular-nums;white-space:nowrap}.task-step-head .task-step-detail[hidden]{display:none}
.project-list-tools{display:flex;align-items:center;gap:10px;flex:0 0 auto;margin-top:16px}.project-list-tools input{min-width:0;flex:1}.project-list-tools span{min-width:88px;color:#60758d;font-size:12px;text-align:right}
.project-card[hidden]{display:none}
.embedded-graph-layout{grid-template-columns:minmax(0,1fr) 320px}.embedded-graph-controls>select,.embedded-graph-controls>input{height:42px;box-sizing:border-box}.embedded-graph-controls>.button{min-height:42px}.embedded-graph-canvas{position:relative;overflow:hidden}.embedded-graph-canvas [data-graph-network]{position:relative;display:block;width:100%;height:clamp(620px,70vh,920px);min-height:620px;background:#f6f8fb}.graph-canvas-overlay{position:absolute;inset:0;z-index:8;display:grid;place-items:center;padding:24px;background:rgba(246,248,251,.88);backdrop-filter:blur(1px)}.graph-canvas-overlay[hidden]{display:none!important}.graph-canvas-overlay .embedded-graph-loading{min-height:0;margin:0;border:0;background:transparent}.graph-canvas-overlay .empty-state{max-width:460px;background:#fff}.kf-gpu-graph-surface,.kf-gpu-default-edges,.kf-gpu-default-nodes,.kf-gpu-emphasized-edges,.kf-gpu-emphasized-nodes,.kf-gpu-node-rings,.kf-gpu-graph-labels{position:absolute;inset:0}.kf-gpu-graph-surface{overflow:hidden;cursor:grab;touch-action:none}.kf-gpu-graph-surface.is-panning{cursor:grabbing}.kf-gpu-graph-surface canvas{display:block;cursor:grab!important;touch-action:none}.kf-gpu-graph-surface.is-panning canvas{cursor:grabbing!important}.kf-gpu-default-edges{z-index:1;overflow:hidden;pointer-events:none}.kf-gpu-default-edge{fill:none;stroke:#7891a7;stroke-width:1.15;stroke-linecap:round;opacity:.32}.kf-gpu-default-edge.is-topic-association{stroke:#8b9fb2;stroke-width:1.05;stroke-dasharray:5 5;opacity:.42}.kf-gpu-default-nodes{z-index:2;overflow:hidden;pointer-events:none}.kf-gpu-default-node{position:absolute;display:block;border:2px solid rgba(255,255,255,.95);border-radius:50%;box-sizing:border-box;box-shadow:0 1px 3px rgba(34,55,77,.09);opacity:.94}.kf-gpu-emphasized-edges{z-index:3;overflow:hidden;pointer-events:none}.kf-gpu-emphasized-edge{fill:none;stroke:#286da8;stroke-width:2.35;stroke-linecap:round;stroke-linejoin:round;opacity:.94}.kf-gpu-emphasized-edge.is-topic-association{stroke:#5f7f9c;stroke-dasharray:6 4;opacity:.88}.kf-gpu-emphasized-nodes{z-index:4;overflow:hidden;pointer-events:none}.kf-gpu-emphasized-node{position:absolute;display:block;border:2px solid rgba(255,255,255,.98);border-radius:50%;box-sizing:border-box;box-shadow:0 1px 3px rgba(34,55,77,.16);opacity:1}.kf-gpu-emphasized-node.is-active{border-color:#fff;box-shadow:0 0 0 2px #df8646,0 1px 5px rgba(194,103,44,.28)}.kf-gpu-node-rings{z-index:5;overflow:hidden;pointer-events:none}.kf-gpu-node-ring{position:absolute;display:block;border:2px solid rgba(255,255,255,.96);border-radius:50%;box-shadow:0 1px 3px rgba(34,55,77,.16);pointer-events:none}.kf-gpu-node-ring.is-active{border-color:#df8646;box-shadow:0 1px 5px rgba(194,103,44,.28)}.kf-gpu-graph-labels{z-index:6;overflow:hidden;pointer-events:none}.kf-gpu-label{position:absolute;left:0;top:0;max-width:190px;padding:1px 4px;border-radius:3px;background:rgba(255,255,255,.68);color:#334b62;font-size:11px;font-weight:600;line-height:1.3;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;box-shadow:0 1px 2px rgba(45,83,120,.05);transform-origin:left top}.kf-gpu-label.is-selected{z-index:3;background:rgba(255,248,239,.94);color:#8f4c0c;font-size:12px}.kf-gpu-label.is-related{z-index:2;background:rgba(246,250,253,.88);color:#315f84;font-size:12px}.kf-gpu-graph-tooltip{position:absolute;z-index:7;max-width:270px;padding:7px 9px;border:0;border-radius:5px;background:rgba(255,255,255,.96);color:#344e66;font-size:12px;box-shadow:0 3px 12px rgba(26,62,99,.12);pointer-events:none}.graph-reading-key{display:flex;align-items:center;flex-wrap:wrap;gap:7px 16px;margin:-6px 0 10px;color:#62778e;font-size:12px}.graph-reading-key span{display:inline-flex;align-items:center;gap:5px}.graph-key-dot{width:10px;height:10px;border-radius:50%;background:#6e91b2}.graph-key-line{width:17px;border-top:1px solid rgba(120,145,167,.7)}.graph-key-topic-line{width:17px;border-top:1px dashed rgba(108,132,154,.8)}.graph-key-size-scale{display:inline-flex!important;align-items:flex-end!important;gap:2px;width:34px;height:17px}.graph-key-size-scale b{display:block;flex:0 0 auto;border-radius:50%;background:#6e91b2}.graph-key-size-scale b:nth-child(1){width:7px;height:7px}.graph-key-size-scale b:nth-child(2){width:11px;height:11px}.graph-key-size-scale b:nth-child(3){width:16px;height:16px}.graph-neighbour-button{margin:9px 0 3px}.graph-scene-note{margin-top:9px!important;padding:8px 9px;border-left:3px solid #7da8d4;background:#f3f8fd;color:#476887!important;font-size:12px!important;line-height:1.5}.graph-scene-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:11px 0}.graph-scene-stats>div{padding:7px;border:1px solid #dce7f1;border-radius:5px;background:#f8fbfe}.graph-scene-stats dt{color:#6a8095;font-size:10px}.graph-scene-stats dd{margin:3px 0 0;color:#294b6e;font-size:11px;font-weight:700}.graph-community-stages>div{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}.graph-community-stages .button{padding:7px 9px;font-size:12px}.graph-directory-heading{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin:2px 0 8px}.graph-directory-heading b{color:#234568;font-size:13px}.graph-directory-heading small,.graph-directory-note{color:#71859a;font-size:11px;line-height:1.45}.graph-directory-note{margin:8px 1px 2px}.embedded-graph-groups[hidden]{display:none!important}
.graph-canvas-scope{position:absolute;top:12px;left:12px;z-index:6;display:inline-flex;align-items:center;padding:3px;border:1px solid rgba(185,205,225,.96);border-radius:7px;background:rgba(255,255,255,.94);box-shadow:0 3px 12px rgba(38,77,116,.1)}.graph-canvas-scope[hidden]{display:none!important}.graph-canvas-scope button{min-height:30px;padding:0 10px;border:0;border-radius:5px;background:transparent;color:#58718b;font:700 12px/1 Arial,"Microsoft YaHei",sans-serif;cursor:pointer}.graph-canvas-scope button:hover{background:#edf6ff;color:#1769c2}.graph-canvas-scope button.is-active{background:#2677d8;color:#fff;box-shadow:0 1px 3px rgba(29,92,158,.18)}.graph-canvas-scope button:focus-visible{outline:2px solid #2677d8;outline-offset:2px}.graph-canvas-nav{position:absolute;top:12px;right:12px;z-index:6;display:flex;align-items:center;gap:5px;padding:4px;border:1px solid rgba(185,205,225,.96);border-radius:7px;background:rgba(255,255,255,.94);box-shadow:0 3px 12px rgba(38,77,116,.14)}.graph-canvas-nav button{display:grid;place-items:center;width:30px;height:30px;padding:0;border:1px solid transparent;border-radius:5px;background:transparent;color:#245a8e;font:700 23px/1 Arial,sans-serif;cursor:pointer}.graph-canvas-nav button:hover:not(:disabled){border-color:#bcd4eb;background:#edf6ff;color:#1769c2}.graph-canvas-nav button:focus-visible{outline:2px solid #2677d8;outline-offset:2px}.graph-canvas-nav button:disabled{color:#aebdca;cursor:not-allowed}.graph-canvas-nav button span{position:relative;top:-1px}.graph-canvas-nav button[data-graph-return-overview]{margin-left:2px;border-left-color:#d7e3ef;border-radius:0}.graph-canvas-nav button[data-graph-return-overview] span{font-size:17px;top:0}.graph-canvas-zoom{display:inline-flex;align-items:center;gap:2px;margin-left:2px;padding-left:4px;border-left:1px solid #d7e3ef}.graph-canvas-zoom output{min-width:44px;color:#49627d;font:12px/1 Arial,"Microsoft YaHei",sans-serif;text-align:center}.graph-community-switch{display:inline-flex;align-items:center;padding:3px;border:1px solid #c7d8e9;border-radius:7px;background:#f8fbfe}.graph-community-switch button{min-height:30px;padding:0 9px;border:0;border-radius:5px;background:transparent;color:#58718b;font:700 12px/1 Arial,"Microsoft YaHei",sans-serif;cursor:pointer}.graph-community-switch button:hover{background:#edf6ff;color:#1769c2}.graph-community-switch button.is-active{background:#2677d8;color:#fff;box-shadow:0 1px 3px rgba(29,92,158,.18)}.graph-community-switch button:focus-visible{outline:2px solid #2677d8;outline-offset:2px}
@media(max-width:1100px){.workspace-body{overflow:auto}.workspace-shell{display:block;width:auto;height:auto;margin:16px auto;padding:0 12px 24px}.workspace-layout{grid-template-columns:1fr;height:auto;min-height:0}.workspace-sidebar{height:auto;overflow:visible;border-right:0;border-bottom:1px solid #dce5ee}.workspace-project-summary{padding:15px 16px}.workspace-nav{grid-template-columns:repeat(4,minmax(160px,1fr));overflow-x:auto}.workspace-nav-item{min-width:160px}.workspace-content,.workspace-view{height:auto;min-height:0;overflow:visible}.embedded-graph-layout{grid-template-columns:1fr}.embedded-graph-sidebar{max-height:none}}
.structure-media-gallery{display:flex;align-items:flex-start;flex-wrap:wrap;gap:10px;margin-top:12px}.structure-image-figure{max-width:260px;margin:0}.structure-image-button{display:block;width:100%;padding:5px;border:1px solid #cfdae6;border-radius:7px;background:#f7fafc;cursor:zoom-in}.structure-image-button:hover{border-color:#6fa5d7;background:#eef6fd}.structure-image-button img{display:block;width:auto;max-width:100%;max-height:180px;margin:auto;object-fit:contain}.structure-image-figure figcaption{margin-top:5px;color:#637991;font-size:11px;line-height:1.45}.structure-table-preview{width:100%;margin-top:12px;overflow:auto;border:1px solid #d6e1ec;border-radius:7px;background:#fff}.structure-table-preview table,.document-media-body table{width:100%;min-width:520px}.structure-table-preview th,.structure-table-preview td,.document-media-body th,.document-media-body td{padding:8px 10px;font-size:12px;white-space:pre-wrap;word-break:break-word}.structure-table-actions{display:flex;align-items:center;justify-content:flex-end;padding:7px 9px;border-top:1px solid #e1e9f1;background:#f8fafc}.document-media-dialog{width:min(1120px,calc(100% - 28px));max-height:92vh}.document-media-body{max-height:72vh;margin:14px 0;overflow:auto}.document-media-body>img{display:block;max-width:100%;height:auto;margin:auto}.document-media-body>p{white-space:pre-wrap}.document-media-table{overflow:auto;border:1px solid #d6e1ec;border-radius:7px}.result-source{display:flex;align-items:center;flex-wrap:wrap;gap:5px;margin-top:8px;color:#657b91;font-size:12px}.result-source a{font-weight:600}.result-media{display:flex;align-items:flex-start;flex-wrap:wrap;gap:8px;width:100%;margin-top:4px}.result-media-thumb{display:inline-flex;padding:4px;border:1px solid #d1deea;border-radius:6px;background:#f7fafc}.result-media-thumb img{display:block;max-width:128px;max-height:86px;object-fit:contain}.result-table{width:100%;margin-top:3px}.result-table summary{font-size:12px}.result-table pre{max-height:260px;overflow:auto;margin:7px 0 0;padding:10px;border-radius:6px;background:#f4f7fa;white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}.result-report-findings{margin:8px 0 0;padding-left:20px;color:#405a74;font-size:13px}.result-report-findings li+li{margin-top:5px}.result-retrieval-note{padding:9px 11px;border-left:3px solid #74a8d7;background:#f4f9fd;color:#506a83;font-size:12px}
.result-report-findings-title{margin:12px 0 0;color:#315978;font-size:13px}.result-report-findings-title+.result-report-findings{margin-top:6px}
@media(max-width:760px){.shell{margin:16px auto;padding:0 10px 24px}header,.section-heading,.project-picker,.upload-form,.search-form{align-items:stretch;flex-direction:column}.home-primary-actions{width:100%}.home-primary-actions .button{flex:1}.chat-entry{min-height:0;padding:24px}.chat-entry-status{align-items:flex-start;flex-direction:column}.project-picker label{min-width:0}.form-grid,.structure-tools{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.construction-controls,.stats-actions,.schema-actions,.graph-actions{align-items:stretch;flex-direction:column}.construction-controls{flex-wrap:wrap}.construction-controls form,.construction-controls .button,.stats-actions .button{width:100%}.construction-controls label{align-items:stretch;flex-direction:column}.construction-controls select{min-width:0}.search-form select{width:100%;min-width:0;flex:auto}.stats-actions .schema-chip{justify-content:center}.panel{padding:16px}.task-step{grid-template-columns:1fr;gap:5px}.home-hero{padding:20px}.hero-layout,.strategy-intro-grid{grid-template-columns:1fr}.hero-copy h2{font-size:25px}.hero-points{margin-top:2px}.home-flow{grid-template-columns:repeat(2,minmax(0,1fr));gap:15px 0}.home-flow li:nth-child(3){border-left:0}.strategy-overview{padding-top:14px}.strategy-heading h2{font-size:22px}.strategy-lead{min-height:0}.choice-guide{grid-template-columns:1fr}.choice-guide em{grid-column:auto}.project-list-heading .button{width:100%}.workspace-nav{padding:8px}.workspace-view-heading h2{font-size:21px}.workspace-settings .form-actions .button{width:100%}}
@media(max-width:440px){.hero-points,.home-flow{grid-template-columns:1fr}.home-flow li+li{border-left:0}.home-flow li{padding:5px 2px}.strategy-details div{grid-template-columns:1fr;gap:4px}.project-grid{grid-template-columns:1fr}}
@media(max-width:760px){.task-head{align-items:stretch;flex-direction:column}.task-head>span{justify-content:space-between;flex-wrap:wrap}.task-step-head{grid-template-columns:minmax(0,1fr) auto}.project-list-tools{align-items:stretch;flex-direction:column}.project-list-tools span{text-align:left}}
@media(max-width:760px){.preparation-heading-side{width:100%;max-width:none;justify-items:stretch}.preparation-update-notice{width:100%}.preparation-heading-side .construction-controls{justify-content:stretch}}
.graph-scope-help{position:relative;display:inline-flex;align-items:center;margin-left:3px;padding-left:6px;border-left:1px solid #d7e3ef}.graph-canvas-scope .graph-scope-help-trigger{display:grid;place-items:center;width:22px;height:22px;min-height:22px;padding:0;border:1px solid #adc3d9;border-radius:50%;background:#fff;color:#486a8a;font:700 12px/1 Arial,"Microsoft YaHei",sans-serif;cursor:help}.graph-canvas-scope .graph-scope-help-trigger:hover,.graph-canvas-scope .graph-scope-help-trigger:focus-visible{border-color:#2677d8;background:#edf6ff;color:#1769c2;outline:none;box-shadow:0 0 0 2px rgba(38,119,216,.16)}.graph-scope-help-tip{position:absolute;top:calc(100% + 9px);left:0;z-index:9;width:290px;max-width:calc(100vw - 80px);padding:9px 11px;border:1px solid #c7d7e7;border-radius:6px;background:rgba(255,255,255,.98);box-shadow:0 5px 16px rgba(30,67,103,.14);color:#405d79;font:12px/1.55 Arial,"Microsoft YaHei",sans-serif;text-align:left;white-space:normal;visibility:hidden;opacity:0;transform:translateY(-3px);transition:opacity .12s ease,transform .12s ease,visibility .12s;pointer-events:none}.graph-scope-help-tip:before{content:"";position:absolute;top:-5px;left:7px;width:8px;height:8px;border-top:1px solid #c7d7e7;border-left:1px solid #c7d7e7;background:#fff;transform:rotate(45deg)}.graph-scope-help:hover .graph-scope-help-tip,.graph-scope-help-trigger:focus-visible+.graph-scope-help-tip{visibility:visible;opacity:1;transform:translateY(0)}
.graph-community-description{display:grid;gap:9px;margin:4px 0 12px}.graph-community-section{padding:10px 11px;border:1px solid #dce6f0;border-radius:6px;background:#fff}.graph-community-section h4{margin:0 0 6px;color:#2a5278;font-size:12px}.graph-community-section p{margin:0;color:#405b76;font-size:12px;line-height:1.6}.graph-community-relations{background:#f8fbfe}.graph-community-evidence{border-left:3px solid #7da8d4;background:#f5f9fd}.graph-community-evidence blockquote{margin:0;color:#3f5c78;font-size:12px;line-height:1.6}

.graph-community-link-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin:10px 0}.graph-community-link-metric{padding:8px 9px;border:1px solid #dce6ef;border-radius:6px;background:#f8fbfe}.graph-community-link-metric small{display:block;color:#74879a;font-size:10px}.graph-community-link-metric b{display:block;margin-top:2px;color:#244b70;font-size:14px}.graph-community-link-section{margin-top:12px}.graph-community-link-section>h4{margin:0 0 7px;color:#2b5278;font-size:12px}.graph-community-link-note{margin:9px 0 0;padding:8px 9px;border-left:3px solid #86a9cb;background:#f4f8fc;color:#4e6880;font-size:11px;line-height:1.55}.graph-predicate-profile{display:grid;gap:6px}.graph-predicate-row{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:2px 8px;align-items:center;color:#365773;font-size:11px}.graph-predicate-row>span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.graph-predicate-row>i{grid-column:1/-1;height:3px;border-radius:3px;background:#e5edf4;overflow:hidden}.graph-predicate-row>i::after{content:"";display:block;width:var(--predicate-share);height:100%;border-radius:inherit;background:#6d9bc6}.graph-community-relation-examples{display:grid;gap:8px}.graph-community-relation-examples .graph-source-card{margin:0}.graph-topic-match-list{display:grid;gap:6px;margin:0;padding:0;list-style:none}.graph-topic-match-list li{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;padding:7px 8px;border:1px solid #dce6ef;border-radius:6px;background:#f8fbfe;color:#365773;font-size:11px}.graph-topic-match-list li span{overflow-wrap:anywhere}.graph-topic-match-list li small{color:#6e8296;white-space:nowrap}

/* 2026 visual refresh: product home and knowledge-base workspace. */
:root{--kf-ink:#172a42;--kf-muted:#667b91;--kf-line:#d8e3ee;--kf-blue:#236fc8;--kf-blue-dark:#185a9f;--kf-soft:#f5f8fc;--kf-card:#fff;--kf-radius:12px}body{background:radial-gradient(circle at 12% -8%,rgba(66,139,216,.10),transparent 28rem),#f3f6fa;color:var(--kf-ink)}.button,.icon-button,input,select,textarea{transition:border-color .16s ease,box-shadow .16s ease,background .16s ease,transform .16s ease}.button{border-radius:8px;font-weight:650}.button.primary{background:linear-gradient(135deg,#2c7bd8,#216bc0);box-shadow:0 4px 12px rgba(35,111,200,.16)}.button.primary:hover{background:linear-gradient(135deg,#236fc8,#185a9f);transform:translateY(-1px)}.button.secondary:hover{border-color:#9db9d5;background:#f6faff;color:#185a9f}.button:focus-visible,.icon-button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible{outline:none;border-color:#4b91dc;box-shadow:0 0 0 3px rgba(44,123,216,.16)}.panel{border-color:var(--kf-line);border-radius:var(--kf-radius);box-shadow:0 12px 32px rgba(31,62,96,.055)}
.home-shell{max-width:1280px;margin-top:20px}.home-header{min-height:58px;margin-bottom:14px;padding:0 4px}.home-header h1{font-size:25px;letter-spacing:-.025em}.home-header p{color:#6f8194;font-size:13px}.home-primary-actions{padding:4px;border:1px solid #d6e1ec;border-radius:10px;background:rgba(255,255,255,.76);box-shadow:0 5px 16px rgba(27,59,94,.05)}.home-primary-actions .button{min-width:112px;border-color:transparent;box-shadow:none}.home-primary-actions .button.secondary{background:transparent}.home-hero{padding:36px;border-color:#c7dced;border-radius:16px;background:linear-gradient(128deg,#fff 0%,#f7fbff 53%,#eaf4fc 100%);box-shadow:0 18px 44px rgba(31,74,119,.075)}.home-hero:before{content:"";position:absolute;right:6%;bottom:-150px;width:330px;height:330px;border-radius:50%;background:radial-gradient(circle,rgba(38,119,216,.09),rgba(38,119,216,0) 66%);pointer-events:none}.hero-layout{grid-template-columns:minmax(0,1.3fr) minmax(330px,.7fr);gap:42px;align-items:center}.hero-copy h2{max-width:720px;margin-top:10px;font-size:36px;line-height:1.18;letter-spacing:-.035em}.hero-copy p{max-width:720px;margin-top:12px;line-height:1.7}.hero-actions{display:flex;align-items:center;gap:10px;margin-top:22px}.hero-actions .button{min-width:132px;text-align:center}.hero-progress-note{display:flex;align-items:center;gap:8px!important;margin-top:15px!important;color:#718398!important;font-size:12px!important}.hero-progress-note i{display:block;flex:0 0 8px;width:8px;height:8px;border-radius:50%;background:#35a66f;box-shadow:0 0 0 4px rgba(53,166,111,.1)}.hero-points{gap:12px}.hero-points div{min-height:112px;padding:17px;border-color:rgba(176,201,225,.8);border-radius:11px;background:rgba(255,255,255,.78);box-shadow:0 10px 25px rgba(36,79,121,.055)}.hero-points b{font-size:15px}.hero-points span{font-size:12.5px}.home-flow{margin-top:30px;padding-top:20px}.home-flow li{position:relative;padding:0 16px}.home-flow li>b{background:#e8f2fc;color:#236fc8;box-shadow:inset 0 0 0 1px #c8dcef}.home-flow strong{font-size:13px}.home-flow small{font-size:11px}.product-overview{padding:28px}.product-overview .section-heading h2{margin-top:7px;font-size:25px;letter-spacing:-.02em}.product-overview .section-heading p{max-width:740px}.product-entry-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px;margin-top:20px}.product-entry-card{display:grid;grid-template-columns:54px minmax(0,1fr);gap:16px;min-height:176px;padding:22px;border:1px solid #d6e2ed;border-radius:12px;background:linear-gradient(145deg,#fff,#f8fbff)}.product-entry-card.chat-product-entry{background:linear-gradient(145deg,#fff,#f5fbfa)}.product-entry-index{display:grid;place-items:center;align-self:start;width:48px;height:48px;border-radius:12px;background:#e8f2fc;color:#2168ad;font-size:13px;font-weight:850}.chat-product-entry .product-entry-index{background:#e4f5f1;color:#14786d}.product-entry-heading{display:flex;align-items:center;justify-content:space-between;gap:10px}.product-entry-heading h3{margin:0;color:#1a3e64;font-size:21px}.product-entry-heading span{padding:4px 8px;border-radius:999px;background:#e6f6ec;color:#257647;font-size:11px;font-weight:750}.chat-product-entry .product-entry-heading span{background:#eef2f6;color:#63768b}.product-entry-card p{min-height:51px;margin-top:12px;font-size:13px}.text-action{display:inline-flex;align-items:center;gap:7px;margin-top:17px;padding:0;border:0;background:none;color:#216bbe;font:inherit;font-size:13px;font-weight:750;text-decoration:none;cursor:pointer}.text-action:hover{color:#164f8d}.text-action b{transition:transform .16s ease}.text-action:hover b{transform:translateX(3px)}.strategy-overview{padding:28px}.strategy-heading{margin-bottom:20px}.strategy-heading h2{font-size:25px;letter-spacing:-.02em}.strategy-explainer{padding:22px;border-top-width:3px;border-radius:12px}.strategy-lead{min-height:52px;line-height:1.65}.strategy-summary-list{display:grid;gap:0;margin:16px 0 0;padding:0;border-top:1px solid #e2eaf2;list-style:none}.strategy-summary-list li{display:grid;grid-template-columns:52px minmax(0,1fr);gap:10px;padding:11px 0;border-bottom:1px solid #e7edf3;color:#566d84;font-size:13px;line-height:1.55}.strategy-summary-list b{color:#274b70}.choice-guide{border-radius:10px;background:#f6f9fd}.project-list-dialog{border-radius:14px;background:#fbfdff}.project-card{border-radius:11px;background:#fff;box-shadow:0 5px 16px rgba(32,66,101,.045)}.project-card:hover{border-color:#a9c7e5;box-shadow:0 10px 24px rgba(32,66,101,.08)}
.workspace-brand-header{align-items:center;min-height:54px;margin:0 4px 10px}.workspace-brand-header .brand-link{display:block}.workspace-brand-header h1{font-size:22px;letter-spacing:-.025em}.workspace-brand-header p{margin-top:2px;font-size:12px}.workspace-global-nav{display:flex;align-items:center;gap:6px;margin-left:auto;padding:4px;border:1px solid #d6e1ec;border-radius:10px;background:rgba(255,255,255,.84)}.workspace-global-nav .button{min-width:94px;padding:7px 12px;border-color:transparent;background:transparent;box-shadow:none;text-align:center}.workspace-global-nav .button.is-current{background:#e8f2fc;color:#185fa8}.workspace-layout{grid-template-columns:264px minmax(0,1fr);border-color:#cfdeeb;border-radius:14px;box-shadow:0 18px 45px rgba(25,57,91,.08)}.workspace-sidebar{background:linear-gradient(180deg,#f8fbfe 0%,#f2f6fa 100%)}.workspace-project-summary{padding:23px 19px 20px;background:linear-gradient(145deg,#f6fbff,#e9f3fc)}.workspace-project-title{grid-template-columns:minmax(0,1fr);gap:12px}.workspace-context-label{display:block;margin-bottom:6px;color:#6e859b;font-size:10px;font-weight:800;letter-spacing:.12em}.workspace-project-summary h2{font-size:19px;line-height:1.35}.workspace-strategy-badge{justify-self:start;border:1px solid #c9ddf1;border-radius:999px;background:rgba(255,255,255,.72);font-size:11px}.workspace-nav{gap:7px;padding:14px 11px}.workspace-nav-item{grid-template-columns:36px minmax(0,1fr);padding:12px;border-radius:9px}.workspace-nav-item>span{width:32px;height:32px;border-radius:9px;background:#e8eef5}.workspace-nav-item:hover{background:#fff;border-color:#d7e3ee;box-shadow:0 4px 12px rgba(31,64,99,.045)}.workspace-nav-item[aria-selected=true]{border-color:#b7d3ee;background:#fff;box-shadow:0 7px 18px rgba(37,91,145,.08)}.workspace-nav-item[aria-selected=true]>span{background:linear-gradient(135deg,#2d7cd8,#216bbe);box-shadow:0 4px 10px rgba(38,119,216,.2)}.workspace-content{background:#fff}.workspace-view{padding:30px 32px}.workspace-view-kicker{margin-bottom:8px}.workspace-view-heading h2{font-size:25px;letter-spacing:-.02em}.workspace-view-heading p{max-width:760px}.upload-form{margin-top:22px;padding:17px 18px;border:1px dashed #bbcee0;border-radius:11px;background:#f7faff}.upload-line input{padding:10px;border-color:#cfdeeb}.upload-help{padding-left:2px}.workspace-section{border-color:#e0e8f0}.workspace-section>.section-heading h2{font-size:19px}.table-wrap{border-color:#d8e3ee;border-radius:10px}.document-table{min-width:760px}.document-table .actions{min-width:145px}.document-table tbody tr{transition:background .14s ease}.document-table tbody tr:hover{background:#f8fbfe}.document-table tbody tr:hover .actions{background:#f8fbfe}.stats{gap:12px;margin:17px 0}.stats div{border:1px solid #dbe6ef;border-left:3px solid #2f7bd4;border-radius:9px;background:#f9fbfd;padding:13px 15px}.stats b{color:#1d4268;font-size:23px}.stats span{margin-top:2px}.task-box{border:1px solid #263c58;border-radius:11px;box-shadow:0 10px 25px rgba(8,24,43,.13)}.definition-card,.processing-options,.native-query-answer,.native-context-card{border-radius:9px}.definition-card{background:#fff;box-shadow:0 3px 11px rgba(31,64,99,.035)}.workspace-settings .form-grid{padding:20px;border:1px solid #dbe5ee;border-radius:11px;background:#fafcff}.settings-processing-options{background:#f7faff}.document-table td:first-child strong{display:block;max-width:430px;overflow-wrap:anywhere}
@media(max-width:1100px){.workspace-brand-header{margin-inline:2px}.workspace-global-nav{margin-left:auto}.workspace-layout{grid-template-columns:1fr;height:auto;min-height:0;border-radius:12px}.workspace-sidebar{height:auto;overflow:visible;border-right:0;border-bottom:1px solid #dce5ee}.workspace-content,.workspace-view{height:auto;min-height:0;overflow:visible}.workspace-project-summary{padding:16px 18px}.workspace-project-title{grid-template-columns:minmax(0,1fr) auto;align-items:center}.workspace-context-label{margin-bottom:3px}.workspace-strategy-badge{justify-self:end}.workspace-nav{grid-template-columns:repeat(4,minmax(155px,1fr));padding:10px}.workspace-view{padding:24px}.workspace-nav-item{min-width:155px}}
@media(max-width:760px){body{background:#f3f6fa}.home-shell{margin-top:10px}.home-header,.workspace-brand-header{align-items:stretch;flex-direction:column;gap:12px}.home-primary-actions,.workspace-global-nav{width:100%;margin:0}.home-primary-actions .button,.workspace-global-nav .button{flex:1;min-width:0}.home-hero{padding:23px;border-radius:13px}.hero-layout,.strategy-intro-grid{grid-template-columns:1fr}.hero-layout{gap:24px}.hero-copy h2{font-size:29px}.hero-copy p{font-size:14px}.hero-actions{align-items:stretch}.hero-actions .button{flex:1;min-width:0}.hero-points{grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.hero-points div{min-height:98px;padding:13px}.home-flow{grid-template-columns:repeat(2,minmax(0,1fr));gap:15px 0}.product-overview,.strategy-overview{padding:20px}.product-entry-grid{grid-template-columns:1fr}.product-entry-card{min-height:0;padding:18px}.strategy-intro-grid{gap:12px}.strategy-explainer{padding:18px}.choice-guide{padding:14px}.workspace-shell{margin-top:10px}.workspace-project-title{grid-template-columns:minmax(0,1fr)}.workspace-strategy-badge{justify-self:start}.workspace-nav{grid-template-columns:repeat(2,minmax(0,1fr));overflow:visible}.workspace-nav-item{min-width:0}.workspace-view{padding:21px 16px}.upload-form{padding:14px}.upload-line{align-items:stretch;flex-direction:column}.upload-line .button{width:100%}.workspace-view-heading h2{font-size:22px}.document-table-wrap{overflow:visible;border:0;border-radius:0;background:transparent}.document-table{display:block;min-width:0}.document-table thead{display:none}.document-table tbody{display:grid;gap:11px}.document-table tbody tr{display:block;overflow:hidden;border:1px solid #d7e3ed;border-radius:10px;background:#fff;box-shadow:0 5px 15px rgba(31,64,99,.045)}.document-table tbody td{display:grid;grid-template-columns:78px minmax(0,1fr);gap:9px;padding:9px 13px;border-bottom:1px solid #edf1f5;text-align:left}.document-table tbody td:before{content:attr(data-label);color:#74879a;font-size:12px;font-weight:700}.document-table tbody td>*{grid-column:2}.document-table tbody td:first-child{padding-top:13px}.document-table tbody td:first-child strong{color:#1d4268}.document-table tbody td .status{justify-self:start}.document-table tbody td small{max-width:none;white-space:normal}.document-table tbody .actions{position:static;display:flex;align-items:center;justify-content:flex-end;min-width:0;padding:10px 13px;background:#f8fbfe}.document-table tbody .actions:before{margin-right:auto}.document-table tbody tr:hover .actions{background:#f8fbfe}.document-table .document-empty-row{box-shadow:none}.document-table .document-empty-row td{display:block;padding:24px}.document-table .document-empty-row td:before{content:none}.workspace-settings .form-grid{padding:15px}.stats{gap:8px}.stats div{padding:11px 12px}}
@media(max-width:440px){.hero-points{grid-template-columns:repeat(2,minmax(0,1fr))}.hero-points div{min-height:104px}.hero-points span{font-size:11.5px}.product-entry-card{grid-template-columns:42px minmax(0,1fr);gap:12px}.product-entry-index{width:40px;height:40px}.product-entry-heading{align-items:flex-start;flex-direction:column}.strategy-summary-list li{grid-template-columns:45px minmax(0,1fr)}}

/* Full-surface refresh: chat, process artifacts, dialogs and result views. */
.workspace-settings #project-settings-form{grid-template-columns:repeat(3,minmax(220px,1fr));align-items:start;gap:17px 18px;max-width:none;margin-top:22px;padding:22px}
.workspace-settings #project-settings-form>label{min-width:0}
.workspace-settings #project-settings-form>.native-config-note{grid-column:span 2;min-height:82px}
.workspace-settings #project-settings-form>.form-actions{grid-column:1/-1;margin-top:3px;padding-top:17px;border-top:1px solid #e0e8f0}
.workspace-settings .settings-processing-options{padding:19px 20px;background:#f7faff}
.workspace-settings .settings-processing-options .form-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;max-width:none;margin:0;padding:0;border:0;border-radius:0;background:transparent}
.workspace-settings .settings-processing-options .native-config-note{min-height:0;justify-content:center;background:rgba(255,255,255,.72)}
.workspace-settings .settings-processing-options>.muted{margin:13px 0 0;padding-top:12px;border-top:1px solid #dfe8f1}
.status-page{display:grid;grid-template-columns:180px minmax(0,1fr);align-items:center;gap:36px;min-height:430px;padding:48px;background:linear-gradient(135deg,#fff 0%,#f7fbff 64%,#edf5fc 100%)}
.status-page-code{display:grid;place-items:center;width:160px;height:160px;border:1px solid #cbddec;border-radius:36px;background:rgba(255,255,255,.82);box-shadow:0 14px 32px rgba(35,85,133,.08);color:#347cc6;font-size:46px;font-weight:850;letter-spacing:-.04em}
.status-page-copy h2{margin-top:9px;color:#1a3e63;font-size:30px;letter-spacing:-.025em}
.status-page-copy p{max-width:620px;font-size:14px;line-height:1.7}
.status-page-actions{display:flex;align-items:center;gap:9px;margin-top:22px}
/* Knowledge Agent: a focused, evidence-first answering workspace. */
.agent-shell{max-width:1480px}
.agent-workbench{padding:0;overflow:hidden;background:#fbfdff}
.agent-heading{align-items:flex-start;margin:0;padding:25px 28px 22px;border-bottom:1px solid #dce6ef;background:linear-gradient(135deg,#fff 0%,#f5faff 100%)}
.agent-heading h2{margin-top:7px;color:#183d63;font-size:26px;letter-spacing:-.025em}
.agent-heading p{max-width:820px;font-size:13px;line-height:1.7}
.agent-scope-bar{display:flex;align-items:flex-end;gap:14px;padding:16px 22px;border-bottom:1px solid #dfe8f1;background:#fff}
.agent-scope-bar>label{display:grid;flex:1 1 520px;gap:6px;min-width:0}
.agent-scope-bar>#agent-mode-field{flex:0 1 310px}
.agent-scope-bar>[hidden]{display:none}
.agent-scope-bar label>span{color:#49657f;font-size:11px;font-weight:800;letter-spacing:.035em}
.agent-scope-bar select{min-width:0;height:42px;border-radius:8px;background:#fbfdff}
.agent-layout{display:grid;grid-template-columns:250px minmax(0,1fr) 350px;height:clamp(650px,calc(100vh - 220px),900px);min-height:0}
.agent-thread-pane{display:flex;min-width:0;min-height:0;flex-direction:column;border-right:1px solid #dce6ef;background:#fbfdff}
.agent-thread-heading{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:17px 14px 13px;border-bottom:1px solid #e1e9f1}
.agent-thread-heading h3{margin:0;color:#1f4569;font-size:16px}
.agent-thread-heading span{display:block;margin-top:3px;color:#7a8da0;font-size:10px}
.agent-thread-heading .button{min-width:auto;padding:7px 10px;border-radius:7px;font-size:11px}
.agent-thread-list{display:flex;flex:1;min-height:0;flex-direction:column;gap:7px;padding:11px;overflow:auto;scrollbar-gutter:stable}
.agent-thread-empty{margin:0;padding:18px 10px;color:#7a8da0;font-size:11px;line-height:1.65;text-align:center}
.agent-thread-row{position:relative;min-width:0}
.agent-thread-item{display:flex;width:100%;flex-direction:column;gap:5px;padding:11px 48px 11px 10px;border:1px solid transparent;border-radius:9px;background:transparent;color:#2a4864;text-align:left;cursor:pointer;transition:border-color .14s ease,background .14s ease,box-shadow .14s ease}
.agent-thread-item:hover{border-color:#d4e1ed;background:#f4f8fc}
.agent-thread-item.active{border-color:#b9d3eb;background:#edf6ff;box-shadow:inset 3px 0 #2b79ca}
.agent-thread-item:disabled{cursor:default;opacity:.72}
.agent-thread-item strong{display:-webkit-box;overflow:hidden;color:#234768;font-size:12px;line-height:1.45;-webkit-box-orient:vertical;-webkit-line-clamp:2}
.agent-thread-preview{display:block;overflow:hidden;color:#6a8094;font-size:10px;line-height:1.45;text-overflow:ellipsis;white-space:nowrap}
.agent-thread-item small{display:flex;align-items:center;gap:7px;color:#8293a4;font-size:9px}
.agent-thread-item small span+span::before{content:'·';margin-right:7px;color:#adbac6}
.agent-thread-running{color:#27835d!important;font-weight:800}
.agent-thread-delete{position:absolute;top:8px;right:7px;padding:4px 6px;border:0;border-radius:5px;background:transparent;color:#8798a8;font-size:9px;cursor:pointer;opacity:.68;transition:opacity .14s ease,background .14s ease,color .14s ease}
.agent-thread-row:hover .agent-thread-delete,.agent-thread-delete:focus-visible{opacity:1}
.agent-thread-delete:hover{background:#fff0f0;color:#b84e54}
.agent-thread-delete:disabled{cursor:default;opacity:.3}
.agent-conversation{display:grid;grid-template-rows:minmax(0,1fr) auto;min-width:0;min-height:0;overflow:hidden;border-right:1px solid #dce6ef;background:#f6f8fb}
.agent-message-list{display:flex;min-height:0;flex-direction:column;gap:18px;padding:27px 32px;overflow:auto;scrollbar-gutter:stable}
.agent-message-list::before{content:"";margin-top:auto}
.agent-empty{display:grid;place-items:center;align-content:center;flex:1;min-height:430px;padding:40px;text-align:center}
.agent-empty[hidden]{display:none}
.agent-empty>span{display:grid;place-items:center;width:64px;height:64px;border:1px solid #c8daec;border-radius:19px;background:linear-gradient(145deg,#e8f3fd,#fff);box-shadow:0 10px 25px rgba(39,87,136,.09);color:#2167ad;font-weight:900;letter-spacing:.06em}
.agent-empty h3{margin:17px 0 0;color:#214568;font-size:21px}
.agent-empty p{max-width:650px;font-size:13px;line-height:1.75}
.agent-message{position:relative;width:min(850px,92%);padding:16px 18px;border:1px solid #d8e3ed;border-radius:12px;background:#fff;box-shadow:0 5px 16px rgba(28,59,91,.045);transition:border-color .16s ease,background-color .16s ease,box-shadow .16s ease}
.agent-message.user{align-self:flex-end;width:min(700px,86%);border-color:#bdd6ee;background:#edf6ff}
.agent-message.assistant{align-self:flex-start}
.agent-message.assistant.agent-turn-selectable{cursor:pointer}
.agent-message.assistant.evidence-selected{border-color:#9fc3e5;box-shadow:inset 3px 0 #2b79ca,0 5px 16px rgba(28,59,91,.055)}
.agent-message.failed{border-color:#efc4c6;background:#fffafa}
.agent-message.cancelled{border-color:#d8dfe7;background:#f8fafc}
.agent-message-role{display:block;margin-bottom:8px;color:#617991;font-size:11px;font-weight:800;letter-spacing:.05em}
.agent-message.user .agent-message-role{color:#2b6da9}
.agent-message-body{color:#203b56;font-size:14px;line-height:1.8;white-space:pre-wrap;overflow-wrap:anywhere}
.agent-execution-trace{margin:0 0 13px;border:1px solid #dbe6ef;border-radius:9px;background:#f8fbfe;color:#395673;overflow:hidden}
.agent-execution-trace>summary{display:flex;align-items:center;gap:10px;min-height:39px;padding:8px 11px;cursor:pointer;list-style:none;user-select:none}
.agent-execution-trace>summary::-webkit-details-marker{display:none}
.agent-execution-trace>summary::before{content:'›';display:grid;place-items:center;width:18px;height:18px;border-radius:50%;background:#e7f1fa;color:#3477b7;font-size:17px;font-weight:800;line-height:1;transition:transform .15s ease}
.agent-execution-trace[open]>summary::before{transform:rotate(90deg)}
.agent-trace-summary-label{color:#2e587e;font-size:12px;font-weight:800}
.agent-trace-summary-status{margin-left:auto;color:#698097;font-size:10px;text-align:right}
.agent-execution-trace.is-running .agent-trace-summary-status{color:#247153}
.agent-execution-trace.is-failed .agent-trace-summary-status{color:#a74a50}
.agent-trace-body{display:grid;gap:13px;max-height:350px;padding:12px 13px 13px;border-top:1px solid #e2eaf1;overflow:auto;background:#fff;scrollbar-gutter:stable}
.agent-trace-current{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:7px;background:#edf8f3;color:#28694f;font-size:11px;font-weight:750}
.agent-trace-current i{width:7px;height:7px;border-radius:50%;background:#35a276;box-shadow:0 0 0 4px rgba(53,162,118,.11);animation:agent-trace-pulse 1.3s ease-in-out infinite}
@keyframes agent-trace-pulse{50%{opacity:.45;transform:scale(.78)}}
.agent-trace-body h4{margin:0 0 7px;color:#315978;font-size:11px;letter-spacing:.025em}
.agent-trace-decisions>div{display:grid;gap:7px}
.agent-trace-decisions>div>div{padding:8px 10px;border-left:2px solid #8db6d9;border-radius:4px;background:#f5f9fc}
.agent-trace-decisions b{display:block;color:#315a7d;font-size:11px}
.agent-trace-decisions p,.agent-trace-item p{margin:3px 0 0;color:#5a7187;font-size:10.5px;line-height:1.55}
.agent-trace-timeline ol{display:grid;gap:0;margin:0;padding:0;list-style:none}
.agent-trace-item{position:relative;display:grid;grid-template-columns:17px minmax(0,1fr);gap:8px;padding:0 0 11px}
.agent-trace-item:not(:last-child)::before{content:'';position:absolute;left:7px;top:14px;bottom:0;width:1px;background:#dce6ee}
.agent-trace-marker{position:relative;z-index:1;display:block;width:9px;height:9px;margin:4px;border:2px solid #fff;border-radius:50%;background:#7da1c2;box-shadow:0 0 0 1px #aec5d9}
.agent-trace-item.is-tool .agent-trace-marker{border-radius:3px;background:#5e8fba}
.agent-trace-item.status-running .agent-trace-marker{background:#31a173;box-shadow:0 0 0 1px #67b799}
.agent-trace-item.status-failed .agent-trace-marker{background:#c75e65;box-shadow:0 0 0 1px #dc9499}
.agent-trace-item.status-degraded .agent-trace-marker{background:#ca8a2c;box-shadow:0 0 0 1px #dfb46f}
.agent-trace-item>div,.agent-trace-item-detail{min-width:0}
.agent-trace-item-detail{border-radius:6px;transition:background-color .14s ease}
.agent-trace-item-detail[open]{margin:-5px -7px -2px;padding:5px 7px 7px;background:#f7fafc}
.agent-trace-item-detail>summary{cursor:pointer;list-style:none}
.agent-trace-item-detail>summary::-webkit-details-marker{display:none}
.agent-trace-item-detail>summary:focus-visible{outline:2px solid #7eb1dd;outline-offset:3px;border-radius:3px}
.agent-trace-item-head{display:flex;align-items:flex-start;justify-content:space-between;gap:9px}
.agent-trace-item-heading{display:block;min-width:0}
.agent-trace-item-head strong{color:#345773;font-size:11px;line-height:1.4}
.agent-trace-item-meta{display:flex;flex:0 0 auto;align-items:center;gap:6px;color:#8091a2}
.agent-trace-item-head small{font-size:9px}
.agent-trace-item code{display:inline-block;margin-top:5px;padding:2px 5px;border-radius:4px;background:#edf3f8;color:#56718a;font:9px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace}
.agent-trace-expand-label{display:inline-flex;align-items:center;gap:3px;color:#47789f;font-size:9px;font-style:normal;font-weight:700}
.agent-trace-expand-label::before{content:'›';font-size:13px;line-height:1;transition:transform .14s ease}
.agent-trace-item-detail[open] .agent-trace-expand-label::before{transform:rotate(90deg)}
.agent-trace-item-detail[open] .agent-trace-expand-label{font-size:0}
.agent-trace-item-detail[open] .agent-trace-expand-label::after{content:'收起';font-size:9px}
.agent-trace-item-panel{margin-top:8px;padding-top:8px;border-top:1px solid #e1e9f0}
.agent-trace-facts{display:grid;grid-template-columns:minmax(86px,auto) minmax(0,1fr);gap:5px 11px;margin:0}
.agent-trace-facts dt{color:#71869a;font-size:9.5px;font-weight:700}
.agent-trace-facts dd{min-width:0;margin:0;color:#334f68;font-size:10px;line-height:1.5;overflow-wrap:anywhere}
.agent-trace-evidence{display:grid;gap:7px;margin-top:10px;padding-top:9px;border-top:1px solid #e1e9f0}
.agent-trace-evidence h5{margin:0;color:#315978;font-size:10px}
.agent-trace-evidence>div{display:grid;gap:6px}
.agent-trace-evidence-card{display:grid;width:100%;gap:4px;padding:8px 9px;border:1px solid #d9e5ef;border-radius:6px;background:#fff;color:#344f68;text-align:left;cursor:pointer;transition:border-color .14s ease,background-color .14s ease,box-shadow .14s ease}
.agent-trace-evidence-card:hover{border-color:#9fc3e3;background:#f8fbfe;box-shadow:0 2px 7px rgba(38,82,123,.07)}
.agent-trace-evidence-card:focus-visible{outline:2px solid #71a7d6;outline-offset:1px}
.agent-trace-evidence-head{display:flex;align-items:flex-start;justify-content:space-between;gap:9px}
.agent-trace-evidence-head b{min-width:0;color:#2f597c;font-size:10px;line-height:1.4;overflow-wrap:anywhere}
.agent-trace-evidence-head small{flex:0 0 auto;color:#668098;font-size:8.5px;line-height:1.4}
.agent-trace-evidence-location{overflow:hidden;color:#75899c;font-size:8.5px;text-overflow:ellipsis;white-space:nowrap}
.agent-trace-evidence-card p{display:-webkit-box;overflow:hidden;margin:0;color:#4f687f;font-size:9.5px;line-height:1.48;-webkit-box-orient:vertical;-webkit-line-clamp:3}
.agent-trace-placeholder{margin:0;color:#71869a;font-size:11px;line-height:1.6}
.agent-trace-disclosure{margin:0;padding-top:9px;border-top:1px solid #e8eef3;color:#8292a1;font-size:9.5px;line-height:1.5}
.agent-message-actions{display:flex;align-items:center;gap:6px;margin-top:13px;padding-top:10px;border-top:1px solid #e7edf3}
.agent-message-actions button{padding:5px 8px;border:0;border-radius:6px;background:transparent;color:#61778c;font:inherit;font-size:11px;cursor:pointer}
.agent-message-actions button:hover{background:#eef4fa;color:#1e67ad}
.agent-message-actions button:disabled{cursor:default;color:#3a805e}
.agent-composer{padding:15px 20px 18px;border-top:1px solid #dbe5ee;background:rgba(255,255,255,.96)}
.agent-composer textarea{min-height:78px;max-height:190px;padding:13px 14px;border-radius:10px;resize:vertical;background:#fff;line-height:1.55}
.agent-composer-footer{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:10px}
.agent-composer-footer>span{color:#6b8196;font-size:11px}
.agent-composer-footer>div{display:flex;align-items:center;gap:8px}
.agent-composer-footer .button{min-width:86px}
.agent-source-pane{min-width:0;min-height:0;padding:20px 17px;background:#fff;overflow:auto}
.agent-source-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;padding:0 3px 14px;border-bottom:1px solid #e0e8f0}
.agent-source-heading h3{margin:5px 0 0;color:#1c4167;font-size:20px}
.agent-source-heading>span{display:grid;place-items:center;min-width:30px;height:30px;padding:0 8px;border-radius:999px;background:#eaf3fc;color:#2168ad;font-size:12px;font-weight:800}
.agent-source-turn{display:grid;gap:6px;margin:13px 2px 3px}
.agent-source-turn[hidden]{display:none}
.agent-source-turn>span{color:#60778d;font-size:10px;font-weight:800}
.agent-source-turn select{width:100%;height:36px;padding:0 30px 0 9px;border:1px solid #cfdeeb;border-radius:7px;background:#f8fbfe;color:#31516f;font-size:11px}
.agent-source-content{min-width:0;transform-origin:top center}
#agent-source-empty{padding:14px 3px;color:#74879a;font-size:12px;line-height:1.65}
.agent-source-list{display:grid;gap:9px;margin-top:13px}
.agent-source-card{display:block;width:100%;padding:12px;border:1px solid #d9e4ee;border-radius:9px;background:#fbfdff;color:#304d69;text-align:left;cursor:pointer;transition:border-color .14s ease,background .14s ease,transform .14s ease}
.agent-source-card:hover{border-color:#a9c7e5;background:#f4f9fe;transform:translateY(-1px)}
.agent-source-card-head{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}
.agent-source-card-head b{color:#24567f;font-size:12px;line-height:1.45}
.agent-source-card-head small{flex:0 0 auto;padding:3px 6px;border-radius:999px;background:#eaf2f9;color:#52718d;font-size:9px;font-weight:700}
.agent-source-card-location{display:block;margin-top:7px;color:#6c8195;font-size:10px;line-height:1.45;overflow-wrap:anywhere}
.agent-source-card-media{display:flex;align-items:center;gap:6px;margin-top:9px;overflow:hidden}.agent-source-card-media img{display:block;width:auto;max-width:calc(50% - 3px);height:82px;border:1px solid #d7e2ec;border-radius:6px;background:#fff;object-fit:contain}.agent-source-card-media small{align-self:flex-end;padding:3px 6px;border-radius:999px;background:#eaf2f9;color:#52718d;font-size:9px;font-weight:700}
.agent-source-card p{display:-webkit-box;margin-top:7px;overflow:hidden;color:#445f78;font-size:11px;line-height:1.55;-webkit-box-orient:vertical;-webkit-line-clamp:4}
.agent-evidence-dialog{width:min(880px,calc(100% - 28px));max-height:88vh}
.agent-evidence-dialog[open]{display:flex;flex-direction:column}
.agent-evidence-dialog>.dialog-heading{flex:0 0 auto}
.agent-evidence-body{display:grid;gap:13px;min-height:0;padding:18px 21px 22px;overflow:auto;background:#f6f8fb}
.agent-evidence-content{margin:0;padding:14px 15px;border:1px solid #d9e4ee;border-left:3px solid #5794cf;border-radius:8px;background:#fff;color:#294660;font-size:13px;line-height:1.75;white-space:pre-wrap}
.agent-evidence-source{padding:14px 15px;border:1px solid #d8e3ed;border-radius:9px;background:#fff}
.agent-evidence-source h3{margin:0;color:#234b70;font-size:14px;overflow-wrap:anywhere}
.agent-evidence-source h4{margin:14px 0 7px;color:#355c7e;font-size:12px}
.agent-evidence-source p{font-size:11px}
.agent-evidence-source blockquote{margin:12px 0;padding:10px 12px;border-left:3px solid #85add2;background:#f5f9fd;color:#36536f;font-size:12px;line-height:1.7;white-space:pre-wrap}
.agent-evidence-source>a:not(.agent-evidence-image){display:inline-flex;margin-top:8px;color:#1769c2;font-size:12px;font-weight:750;text-decoration:none}
.agent-evidence-source pre{max-height:320px;margin:0;padding:11px;border:1px solid #e0e7ee;border-radius:7px;background:#f8fafc;color:#2f4962;font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;overflow:auto;white-space:pre-wrap}
.agent-evidence-image{display:inline-block;margin:12px 10px 0 0;padding:5px;border:1px solid #d7e2ec;border-radius:8px;background:#fff}
.agent-evidence-image img{display:block;width:auto;max-width:240px;max-height:180px;border-radius:5px;object-fit:contain}

.process-artifacts .table-wrap{background:#fff}
.artifact-table tbody tr{transition:background .14s ease}
.artifact-table tbody tr:hover{background:#f8fbfe}
.artifact-table td:first-child strong{display:block;max-width:520px;color:#1f405f;overflow-wrap:anywhere}
.artifact-table .artifact-actions{text-align:right}
.artifact-table .artifact-actions .button{margin-block:2px}

dialog:not(.graph-evidence-modal){padding:0;border-color:#cbd9e6;border-radius:14px;background:#fff;overflow:hidden;box-shadow:0 26px 75px rgba(15,35,60,.26)}
dialog:not(.graph-evidence-modal)::backdrop{background:rgba(24,40,59,.48);backdrop-filter:blur(2px)}
dialog:not(.graph-evidence-modal)>.dialog-heading{position:relative;z-index:2;margin:0;padding:18px 21px 16px;border-bottom:1px solid #dde6ee;background:linear-gradient(135deg,#fff,#f7faff)}
dialog:not(.graph-evidence-modal)>.dialog-heading h2{color:#1a395a;font-size:20px;letter-spacing:-.015em}
dialog:not(.graph-evidence-modal)>.dialog-heading p{font-size:12px}
dialog:not(.graph-evidence-modal)>.dialog-heading .icon-button{border:1px solid #d0dce8;background:#fff;color:#3b5875;box-shadow:none}
dialog:not(.graph-evidence-modal)>.dialog-heading .icon-button:hover{border-color:#a9bfd5;background:#eef5fb}
#create-project-dialog>form{margin:0;padding:20px 22px}
.project-list-dialog .project-list-tools{margin:0;padding:16px 22px 0}
.project-list-dialog .project-grid{margin:0;padding:16px 22px 22px}
dialog:not(.graph-evidence-modal)>.dialog-actions,#create-project-dialog .dialog-actions{margin:0;padding:13px 20px;border-top:1px solid #dde6ee;background:#f8fafc}
.log-dialog>.history-log{margin:16px 20px;min-height:300px;border:1px solid #263a53;border-radius:9px;background:#0f1d2e;box-shadow:inset 0 1px 10px rgba(0,0,0,.16)}
dialog:not(.graph-evidence-modal)>.job-log-error{margin:16px 20px 0}
.delete-progress-dialog>.delete-progress-body{margin:20px}
.delete-progress-dialog>.delete-progress-bar{margin:0 20px 20px}
.structure-dialog>.structure-tools{margin:16px 20px 12px}
.structure-dialog>.structure-list{display:grid;gap:9px;max-height:58vh;margin:0 20px;padding:10px;border-color:#d7e2ec;border-radius:10px;background:#f5f8fb}
.structure-dialog>.structure-pager{margin:12px 20px;color:#687e94;font-size:12px}
.structure-row{padding:14px 15px;border:1px solid #d8e3ed!important;border-radius:9px;background:#fff;box-shadow:0 3px 10px rgba(31,64,99,.035)}
.structure-row-head{align-items:flex-start}
.structure-row-meta{line-height:1.5}
.structure-text{color:#2c4660;font:13px/1.72 system-ui,-apple-system,"Microsoft YaHei",sans-serif}
.structure-metadata{padding-top:8px;border-top:1px solid #e8edf2}
.structure-metadata pre{border:1px solid #e1e8ef;border-radius:7px;background:#f8fafc}
.block-type,.chunk-badge{border-radius:999px}
.structure-image-button,.structure-table-preview{border-radius:9px}
.document-media-dialog>.document-media-body{margin:16px 20px;max-height:72vh;padding:12px;border:1px solid #dce5ed;border-radius:10px;background:#f7f9fc}
.document-media-body>p{padding:0 8px;color:#496178;font-size:13px;line-height:1.65}
.structure-dialog>.artifact-content{margin:16px 20px;min-height:320px;border:1px solid #263a53;border-radius:9px;background:#0f1d2e}
.native-query-answer{box-shadow:0 6px 18px rgba(31,70,109,.05)}
.result-group{display:grid;gap:10px;margin-top:18px}
.result-card{padding:14px 15px;border:1px solid #dbe5ee;border-radius:9px;background:#fbfdff}
.result-card h3{color:#274c70}
.result-card p{line-height:1.65}
.result-source a{color:#216bbe;text-decoration:none}
.result-source a:hover{text-decoration:underline}

@media(max-width:900px){
  .agent-scope-bar{flex-wrap:wrap}
  .agent-scope-bar>label,.agent-scope-bar>#agent-mode-field{flex:1 1 300px}
  .agent-layout{grid-template-columns:1fr;height:auto;min-height:650px}
  .agent-thread-pane{max-height:250px;border-right:0;border-bottom:1px solid #dce6ef}
  .agent-thread-list{max-height:190px}
  .agent-conversation{border-right:0;border-bottom:1px solid #dce6ef}
  .agent-message-list{min-height:500px}
  .agent-source-pane{max-height:440px}
}
@media(min-width:901px) and (max-width:1180px){
  .agent-layout{grid-template-columns:225px minmax(0,1fr)}
  .agent-source-pane{grid-column:1/-1;max-height:390px;border-top:1px solid #dce6ef}
}
@media(max-width:1180px){
  .workspace-settings #project-settings-form{grid-template-columns:repeat(2,minmax(0,1fr))}
  .workspace-settings #project-settings-form>.native-config-note{grid-column:1/-1}
}
@media(max-width:760px){
  .agent-heading{align-items:flex-start;flex-direction:column;padding:21px 18px}
  .agent-heading h2{font-size:23px}
  .agent-scope-bar{align-items:stretch;flex-direction:column;padding:14px}
  .agent-scope-bar>label,.agent-scope-bar>#agent-mode-field{flex:auto;width:100%}
  .agent-thread-heading{padding:14px 13px 11px}
  .agent-thread-list{padding:9px}
  .agent-message-list{gap:13px;min-height:420px;padding:18px 12px}
  .agent-empty{min-height:360px;padding:24px 14px}
  .agent-message,.agent-message.user{width:96%;padding:14px}
  .agent-composer{padding:12px}
  .agent-composer-footer{align-items:flex-start;flex-direction:column}
  .agent-composer-footer>div{align-self:stretch}
  .agent-composer-footer .button{flex:1}
  .agent-source-pane{padding:17px 13px}
  .agent-evidence-body{padding:13px 14px 17px}
  .agent-evidence-image img{max-width:100%}
  .workspace-settings #project-settings-form{grid-template-columns:1fr;padding:15px}
  .workspace-settings #project-settings-form>.native-config-note{grid-column:auto}
  .workspace-settings .settings-processing-options{padding:15px}
  .workspace-settings .settings-processing-options .form-grid{grid-template-columns:1fr}
  .workspace-settings>.section-heading{align-items:flex-start;flex-direction:row}
  .workspace-settings>.section-heading>div{min-width:0}
  .workspace-settings>.section-heading .refresh-button{flex:0 0 34px}
  .workspace-content .stats-actions{align-items:stretch;flex-direction:column;overflow:visible}
  .workspace-content .stats-actions>.construction-controls{flex:0 0 auto;width:100%;align-items:stretch;flex-direction:column;flex-wrap:nowrap}
  .workspace-content .stats-actions>.construction-controls form,
  .workspace-content .stats-actions>.construction-controls .button{width:100%}
  .workspace-content .stats-actions>.schema-actions,
  .workspace-content .stats-actions>.graph-actions{width:100%;align-items:stretch;flex-direction:column}
  .workspace-content .stats-actions>.schema-actions .button,
  .workspace-content .stats-actions>.graph-actions .button{width:100%}
  .embedded-graph-controls{display:grid;grid-template-columns:minmax(0,1fr) auto auto;align-items:center}
  .embedded-graph-controls>select{grid-column:1/-1;width:100%;min-width:0}
  .embedded-graph-controls>input{width:100%;min-width:0}
  .graph-canvas-scope{top:10px;left:10px;max-width:calc(100% - 20px)}
  .graph-canvas-nav{top:55px;right:10px}
  .status-page{grid-template-columns:1fr;gap:24px;min-height:0;padding:30px 22px}
  .status-page-code{width:104px;height:104px;border-radius:27px;font-size:34px}
  .status-page-copy h2{font-size:26px}
  .status-page-actions{align-items:stretch;flex-direction:column}
  .status-page-actions .button{text-align:center}
  dialog:not(.graph-evidence-modal){width:calc(100% - 18px);max-height:94vh}
  dialog:not(.graph-evidence-modal)>.dialog-heading{align-items:flex-start;flex-direction:row;padding:15px 16px}
  #create-project-dialog>form{padding:16px}
  .project-list-dialog .project-list-tools{padding:13px 16px 0}
  .project-list-dialog .project-grid{padding:13px 16px 16px}
  .project-list-dialog .project-card{padding:15px}
  .project-list-dialog .bind-form{align-items:stretch;flex-direction:column}
  .structure-dialog>.structure-tools{margin:13px 14px 9px}
  .structure-dialog>.structure-list{margin:0 14px;padding:7px}
  .structure-dialog>.structure-pager{align-items:stretch;flex-direction:column;margin:10px 14px}
  .structure-dialog>.structure-pager>div .button{flex:1}
  .structure-row{padding:12px}
  .structure-row-head{align-items:flex-start;flex-direction:column}
  .document-media-dialog>.document-media-body{margin:13px 14px;padding:8px}
  .log-dialog>.history-log,.structure-dialog>.artifact-content{margin:13px 14px;min-height:260px}
  dialog:not(.graph-evidence-modal)>.dialog-actions,#create-project-dialog .dialog-actions{padding:11px 14px}
  .process-artifacts .table-wrap{overflow:visible;border:0;background:transparent}
  .artifact-table{display:block;min-width:0}
  .artifact-table thead{display:none}
  .artifact-table tbody{display:grid;gap:10px}
  .artifact-table tbody tr{display:block;overflow:hidden;border:1px solid #d7e3ed;border-radius:10px;background:#fff;box-shadow:0 5px 15px rgba(31,64,99,.045)}
  .artifact-table tbody td{display:grid;grid-template-columns:92px minmax(0,1fr);gap:9px;padding:10px 12px;border-bottom:1px solid #edf1f5;text-align:left}
  .artifact-table tbody td::before{content:attr(data-label);color:#74879a;font-size:11px;font-weight:700}
  .artifact-table tbody td>*{grid-column:2}
  .artifact-table tbody td .status{justify-self:start}
  .artifact-table .artifact-actions{display:flex;align-items:center;flex-wrap:wrap;gap:7px;text-align:left}
  .artifact-table .artifact-actions::before{margin-right:auto}
  .artifact-table .artifact-actions>*{grid-column:auto;margin:0!important}
  .artifact-table .artifact-actions .button{flex:1 1 130px}
  .artifact-table .artifact-empty-row{box-shadow:none}
  .artifact-table .artifact-empty-row td{display:block;padding:22px}
  .artifact-table .artifact-empty-row td::before{content:none}
}
"""

SCRIPT = r"""
const box=document.querySelector('#task-box'),taskStatus=document.querySelector('#task-status'),taskSummary=document.querySelector('#task-summary'),taskProgress=document.querySelector('#task-progress'),taskDuration=document.querySelector('#task-duration'),taskSteps=document.querySelector('#task-steps'),taskPrerequisites=document.querySelector('#task-prerequisites'),taskPrerequisiteDetail=document.querySelector('#task-prerequisite-detail'),cancelJobButtons=[...document.querySelectorAll('[data-cancel-project-job]')];
const esc=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const ingestionLogButton=document.querySelector('#view-ingestion-log'),preparationLogButton=document.querySelector('#view-preparation-log'),taskTitle=document.querySelector('#task-title');
const stageLabels={structure:'结构解析中',structure_ready:'结构解析完成',chunking:'知识分块中',chunk_ready:'知识分块完成',graph:'知识图谱构建中',completed:'全部完成',failed:'处理失败'};
const subtaskLabels={structure:'解析文档结构',chunking:'生成知识块与向量',schema_base:'确认知识结构',extraction:'识别实体与关系',graph_sync:'发布知识图谱'};
// Native is a separate GraphRAG Standard pipeline.  It must never inherit
// Stable's HugeGraph/pgvector/entity-relation stage names in the UI.
const nativeSubtaskLabels=window.KF.taskProfile==='graphrag_native'?window.KF.taskLabels:{structure:'准备构建输入',chunking:'检查 GraphRAG 配置',schema_base:'检查模型服务',extraction:'生成文本单元、知识图谱与社区',graph_sync:'验证并发布知识图谱成果'};
const nativePreparationLabels={structure:'解析并冻结源文档结构',chunking:'',schema_base:'验证结构化文档',extraction:'',graph_sync:''};
const taskStatusLabels={queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',cancelled:'已终止'};
const clampProgress=value=>Math.max(0,Math.min(100,Math.round(Number(value)||0)));
function labelsForJob(job){if(job.kind==='graphrag_native_build')return nativeSubtaskLabels;if(job.kind==='content_preparation'&&window.KF.taskProfile==='graphrag_native')return nativePreparationLabels;return window.KF.taskLabels||subtaskLabels}
function legacySubtasks(job){const p=clampProgress(job.progress),step=(status,progress,detail='')=>({status,progress:clampProgress(progress),detail});const result={structure:step('waiting',0),chunking:step('waiting',0),schema_base:step('waiting',0),extraction:step('waiting',0),graph_sync:step('waiting',0)};if(job.kind==='graph_sync'){result.graph_sync=step(job.status==='completed'?'completed':job.status==='failed'?'failed':'running',p);return result}if(job.kind==='schema_reassessment'){result.schema_base=step(job.status==='completed'?'completed':job.status==='failed'?'failed':'running',p,'重新评估');return result}result.structure=p>=18?step('completed',100):step('running',p/18*100);if(p>=20)result.chunking=p>=42?step('completed',100):step('running',(p-20)/22*100);if(p>=44)result.schema_base=p>=50?step('completed',100):step('running',(p-44)/6*100);if(p>=52)result.extraction=p>=90?step('completed',100):step('running',(p-52)/38*100);if(p>=92)result.graph_sync=p>=96?step('completed',100):step('running',(p-92)/4*100);if(job.status==='completed')for(const key of Object.keys(result))result[key]=step('completed',100,result[key].detail);return result}
function nativeSubtasks(job){
  const p=clampProgress(job.progress),step=(status,progress,detail='')=>({status,progress:clampProgress(progress),detail}),result={structure:step('waiting',0),chunking:step('waiting',0),schema_base:step('waiting',0),extraction:step('waiting',0),graph_sync:step('waiting',0)};
  const nativeProgress=job.metadata?.native_progress||{},numberOr=(value,fallback)=>Number.isFinite(Number(value))?clampProgress(value):clampProgress(fallback);
  if(p<10)result.structure=step('running',p*10);else result.structure=step('completed',100,'已固定文档版本');
  if(p>=10&&p<25)result.chunking=step('running',(p-10)/15*100);else if(p>=25)result.chunking=step('completed',100);
  if(p>=25&&p<30)result.schema_base=step('running',(p-25)/5*100);else if(p>=30)result.schema_base=step('completed',100);
  const executionProgress=numberOr(nativeProgress.execution_progress,(p-30)/42*100),executionDetail=String(nativeProgress.detail||'').trim()||'GraphRAG 标准流程正在运行';
  if(p>=30&&p<72)result.extraction=step('running',executionProgress,executionDetail);else if(p>=72)result.extraction=step('completed',100,'GraphRAG 标准流程已完成');
  const validationProgress=numberOr(nativeProgress.phase_progress,(p-72)/28*100),validationDetail=['validation','publication'].includes(String(nativeProgress.phase||''))&&String(nativeProgress.detail||'').trim()?String(nativeProgress.detail).trim():'正在验证并发布构建成果';
  if(p>=72&&p<100)result.graph_sync=step('running',validationProgress,validationDetail);
  if(job.status==='completed')for(const key of Object.keys(result))result[key]=step('completed',100,result[key].detail);
  if(job.status==='failed'||job.status==='cancelled'){const active=['graph_sync','extraction','schema_base','chunking','structure'].find(key=>result[key].status==='running')||'structure';result[active]=step(job.status,p,result[active].detail)}
  return result
}
function jobSubtasks(job){if(job.kind==='graphrag_native_build')return nativeSubtasks(job);const stored=job.metadata?.subtasks;if(!stored||typeof stored!=='object')return legacySubtasks(job);return {...legacySubtasks({kind:job.kind,status:'queued',progress:0}),...stored}}
let activeMonitorGeneration=0,renderedMonitorKey='',renderedSubtaskProgress={};
function presentationDetail(key,data,job){const detail=String(data?.detail||'').trim();if(window.KF.strategyKey==='knowledge_forge'&&key==='schema_base'&&data?.status==='completed')return '已确认本次构建使用的实体类别、关系规则和发布约束。';return detail}
const isKnowledgeForgeBuild=job=>job.kind==='ingestion'&&window.KF.strategyKey==='knowledge_forge';
function knowledgeForgeBuildProgress(job,subtasks){
  if(job.status==='completed')return 100;
  const schemaDiscovery=Boolean(job.metadata?.schema_discovery_during_build);
  const value=key=>{const step=subtasks[key]||{};return step.status==='completed'?100:clampProgress(step.progress)};
  return clampProgress((schemaDiscovery?value('schema_base')*.10:0)+value('extraction')*(schemaDiscovery?.82:.90)+value('graph_sync')*(schemaDiscovery?.08:.10));
}
function renderBuildPrerequisites(job,subtasks){
  const visible=isKnowledgeForgeBuild(job);
  if(taskPrerequisites)taskPrerequisites.hidden=!visible;
  if(!visible||!taskPrerequisiteDetail)return;
  const metadata=job.metadata||{},parts=[];
  const documentCount=Number(metadata.prepared_document_count||metadata.target_document_count||0),chunkCount=Number(metadata.prepared_chunk_count||0);
  if(documentCount>0)parts.push(documentCount+' 份文档结构');
  if(chunkCount>0)parts.push(chunkCount+' 个知识块与向量');
  const schemaDetail=String(subtasks.schema_base?.detail||'').trim();
  if(!metadata.schema_discovery_during_build&&schemaDetail)parts.push(schemaDetail);
  const checking=metadata.prerequisite_status==='checking';
  taskPrerequisites.querySelector('b').textContent=checking?'正在确认预处理结果':metadata.schema_discovery_during_build?'预处理结果已就绪':'预处理结果与知识结构已就绪';
  taskPrerequisiteDetail.textContent=parts.join(' · ')||(checking?'正在校验文档结构、知识块、向量和当前知识结构':'本次构建直接复用已完成的准备结果');
}
function renderSubtasks(job,monitorKey=''){
  if(monitorKey&&renderedMonitorKey!==monitorKey){renderedMonitorKey=monitorKey;renderedSubtaskProgress={}}
  const subtasks=jobSubtasks(job),labels=labelsForJob(job),metadata=job.metadata||{},knowledgeBuild=isKnowledgeForgeBuild(job),activeKey=Object.keys(subtasks).find(key=>subtasks[key]?.status==='running'),activeDetail=activeKey?presentationDetail(activeKey,subtasks[activeKey],job):'',documentPrefix=Number(metadata.active_document_index)>0&&Number(metadata.target_document_count)>1?'第 '+metadata.active_document_index+'/'+metadata.target_document_count+' 份文档 · ':'';
  if(taskStatus)taskStatus.textContent=metadata.cancel_requested&&['queued','running'].includes(job.status)?'正在终止':taskStatusLabels[job.status]||job.status||'等待';
  if(taskProgress){const progress=knowledgeBuild?knowledgeForgeBuildProgress(job,subtasks):clampProgress(job.progress);taskProgress.textContent=(knowledgeBuild?'构建进度 ':'总进度 ')+progress+'%'}
  renderBuildPrerequisites(job,subtasks);
  if(taskSummary){const cancelling=metadata.cancel_requested&&['queued','running'].includes(job.status);taskSummary.textContent=cancelling?'正在等待当前安全步骤结束，未完成结果不会发布。':activeDetail?documentPrefix+activeDetail:'';taskSummary.hidden=!taskSummary.textContent}
  for(const row of taskSteps?.querySelectorAll('[data-subtask]')||[]){
    const key=row.dataset.subtask,title=row.querySelector('.task-step-head span');if(title)title.textContent=labels[key]||key;
    const preparationHidden=job.kind==='content_preparation'&&(['extraction','graph_sync'].includes(key)||(window.KF.taskProfile==='graphrag_native'&&key==='chunking')),knowledgeBuildHidden=knowledgeBuild&&(['structure','chunking'].includes(key)||(key==='schema_base'&&!metadata.schema_discovery_during_build));
    row.hidden=job.kind==='document_cleanup'||(job.kind==='schema_reassessment'&&key!=='schema_base')||preparationHidden||knowledgeBuildHidden;
    const data=subtasks[key]||{status:'waiting',progress:0,detail:''},status=data.status||'waiting',reportedProgress=clampProgress(data.progress),progress=monitorKey&&['running','completed'].includes(status)?Math.max(renderedSubtaskProgress[key]||0,reportedProgress):reportedProgress,label=row.querySelector('[data-subtask-status]'),detail=row.querySelector('[data-subtask-detail]'),bar=row.querySelector('[data-subtask-bar]'),visibleDetail=presentationDetail(key,data,job);
    if(monitorKey&&['running','completed'].includes(status))renderedSubtaskProgress[key]=progress;row.className='task-step '+status;
    if(detail){detail.textContent=visibleDetail;detail.hidden=!visibleDetail}
    if(label){if(status==='completed')label.textContent='已完成';else if(status==='failed')label.textContent='失败';else if(status==='cancelled')label.textContent='已终止';else if(status==='running')label.textContent=progress+'%';else label.textContent='等待'}
    if(bar)bar.style.width=progress+'%';
  }
}
function syncCancellationControls(job,id,batch){const kind=String(job.kind||box?.dataset.kind||''),active=['queued','running'].includes(job.status),requested=Boolean(job.metadata?.cancel_requested),scope=kind==='content_preparation'?'preparation':['ingestion','graphrag_native_build','graph_sync'].includes(kind)?'graph':'';for(const button of cancelJobButtons){const visible=active&&scope&&button.dataset.cancelScope===scope;button.hidden=!visible;if(!visible)continue;button.dataset.job=id||String(job.id||'');button.dataset.batch=batch||String(job.batch_id||'');button.dataset.kind=kind;button.disabled=requested;button.textContent=requested?'正在终止…':scope==='preparation'?'终止预处理':'终止构建'}}
function formatDuration(seconds){const value=Math.max(0,Math.floor(Number(seconds)||0)),hours=Math.floor(value/3600),minutes=Math.floor(value%3600/60),secs=value%60;return(hours?String(hours).padStart(2,'0')+':':'')+String(minutes).padStart(2,'0')+':'+String(secs).padStart(2,'0')}
let lastStatisticsRefresh=0;
async function refreshStatistics(force=false){if(!window.KF.projectId||(!force&&Date.now()-lastStatisticsRefresh<2000))return;lastStatisticsRefresh=Date.now();try{const response=await fetch('/api/projects/'+window.KF.projectId+'/statistics'),data=await response.json();if(!response.ok)return;for(const key of ['documents','chunks','entities','relations']){const target=document.querySelector('[data-stat="'+key+'"]');if(target)target.textContent=Number(data[key]||0).toLocaleString()}}catch(_error){}}
function updatingStatisticKeys(kind){if(kind==='content_preparation')return window.KF.taskProfile==='graphrag_native'?[]:['chunks'];if(kind==='ingestion')return['entities','relations'];if(['graphrag_native_build','document_cleanup'].includes(kind))return['chunks','entities','relations'];if(kind==='graph_sync')return['entities','relations'];return[]}
function setStatisticsUpdating(updating,kind=''){const active=new Set(updating?updatingStatisticKeys(kind):[]);for(const key of ['chunks','entities','relations']){const target=document.querySelector('[data-stat="'+key+'"]');if(!target)continue;const changing=active.has(key);target.classList.toggle('is-updating',changing);target.setAttribute('aria-label',changing?target.textContent+'，仍在更新':target.textContent)}}
function applyJobReadiness(job){const metadata=job.metadata||{},documentId=job.document_id||metadata.document_id;if(documentId){const structure=document.querySelector('[data-view-structure="'+documentId+'"]'),chunks=document.querySelector('[data-view-chunks="'+documentId+'"]'),label=document.querySelector('[data-stage-label="'+documentId+'"]'),status=document.querySelector('[data-document-status="'+documentId+'"]');if(structure&&metadata.structure_ready)structure.disabled=false;if(chunks&&metadata.chunks_ready)chunks.disabled=false;if(label&&metadata.stage)label.textContent=stageLabels[metadata.stage]||metadata.stage;if(status){let state=job.status==='completed'?'indexed':job.status==='failed'?'failed':job.status==='running'?'processing':'uploaded',text={uploaded:'已上传',processing:'处理中',indexed:'已入库',failed:'失败'}[state];if(job.kind==='content_preparation'){const native=window.KF.taskProfile==='graphrag_native';state=job.status==='failed'?'failed':job.status==='running'?'processing':'uploaded';text=job.status==='completed'?(native?'已准备':'预处理完成'):job.status==='failed'?(native?'准备失败':'预处理失败'):job.status==='running'?(native?'准备中':'预处理中'):(native?'待准备':'等待预处理')}status.className='status status-'+state;status.textContent=text||job.status}}}
function monitor(id,kind='',batch=''){if(!id)return;const generation=++activeMonitorGeneration,monitorKey=(batch?'batch:':'job:')+(batch||id),preparation=kind==='content_preparation',slot=document.querySelector('[data-task-slot="'+(preparation?'preparation':'graph')+'"]');if(slot&&box.parentElement!==slot)slot.append(box);box.hidden=false;box.dataset.job=id;box.dataset.batch=batch;box.dataset.kind=kind;setStatisticsUpdating(true,kind);const target=preparation?preparationLogButton:['ingestion','graphrag_native_build'].includes(kind)?ingestionLogButton:null,activeLogButton=document.querySelector('#active-job-log');for(const button of [target,activeLogButton])if(button){button.dataset.job=id;button.dataset.batch=batch;button.disabled=false}if(taskTitle)taskTitle.textContent=kind==='graphrag_native_build'?'GraphRAG Native 构建':kind==='graph_sync'?'发布知识图谱':kind==='schema_reassessment'?'更新知识结构':kind==='document_cleanup'?'清理文档':kind==='content_preparation'?(window.KF.taskProfile==='graphrag_native'?'准备 GraphRAG 源文档':'文档预处理'):window.KF.strategyKey==='knowledge_forge'?'Knowledge Forge 构建':'知识库构建';const tick=async()=>{if(generation!==activeMonitorGeneration)return;try{const endpoint=batch?'/api/job-batches/'+batch:'/api/jobs/'+id,res=await fetch(endpoint),job=await res.json();if(generation!==activeMonitorGeneration)return;if(!res.ok)throw new Error(job.error||'任务状态读取失败');renderSubtasks(job,monitorKey);syncCancellationControls(job,id,batch);updateNativeRuntime(job);setStatisticsUpdating(!['completed','failed','cancelled'].includes(job.status),job.kind||kind);if(taskDuration)taskDuration.textContent='用时 '+formatDuration(job.elapsed_seconds);for(const item of job.jobs||[job])applyJobReadiness(item);await refreshStatistics();if(generation!==activeMonitorGeneration)return;if(['completed','failed','cancelled'].includes(job.status)){setStatisticsUpdating(false,job.kind||kind);setTimeout(()=>{if(generation===activeMonitorGeneration)location.reload()},900);return}}catch(error){if(generation!==activeMonitorGeneration)return;if(taskStatus)taskStatus.textContent='状态读取失败'}if(generation===activeMonitorGeneration)setTimeout(tick,800)};tick()}
if(box?.dataset.job)monitor(box.dataset.job,box.dataset.kind||'',box.dataset.batch||'');
for(const button of cancelJobButtons)button.addEventListener('click',async()=>{const preparation=button.dataset.cancelScope==='preparation',label=preparation?'预处理':'图谱构建';if(!confirm('确认终止当前'+label+'吗？系统会等待当前安全步骤结束，并且不会发布未完成结果。'))return;button.disabled=true;button.textContent='正在提交…';try{const body=new URLSearchParams({job_id:button.dataset.job||'',batch_id:button.dataset.batch||'',kind:button.dataset.kind||''}),response=await fetch('/api/projects/'+window.KF.projectId+'/jobs/cancel',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8'},body}),data=await response.json();if(!response.ok)throw new Error(data.error||'终止请求提交失败');button.textContent='正在终止…';if(taskStatus)taskStatus.textContent='正在终止';if(taskSummary){taskSummary.textContent='正在等待当前安全步骤结束，未完成结果不会发布。';taskSummary.hidden=false}}catch(error){button.disabled=false;button.textContent=preparation?'终止预处理':'终止构建';alert(error.message||'终止请求提交失败')}});
document.querySelector('#build-project-knowledge')?.addEventListener('click',async event=>{const button=event.currentTarget,native=window.KF.strategyKey==='graphrag_native',operation=native?'知识图谱构建':'知识构建';if(fileInput?.files?.length){alert('请先上传当前已选择的文件，再执行'+operation+'。');return}button.disabled=true;const original=button.textContent;button.textContent='提交中...';try{const endpoint=native?'/api/projects/'+button.dataset.buildProject+'/graphrag-native/build':'/api/projects/'+button.dataset.buildProject+'/knowledge-forge/build',response=await fetch(endpoint,{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||operation+'提交失败');if(!native)for(const status of document.querySelectorAll('[data-document-status]')){if(!status.classList.contains('status-indexed')){status.className='status status-queued';status.textContent='排队中'}}monitor(data.job_id,native?'graphrag_native_build':'ingestion',data.batch_id||'')}catch(error){alert(error.message||operation+'提交失败');button.disabled=false;button.textContent=original}});
document.querySelector('#prepare-project-documents')?.addEventListener('click',async event=>{const button=event.currentTarget;if(fileInput?.files?.length){alert('请先上传当前已选择的文件，再执行预处理。');return}button.disabled=true;const original=button.textContent;button.textContent='提交中...';try{const slug=window.KF.strategyKey==='graphrag_native'?'graphrag-native':'knowledge-forge',response=await fetch('/api/projects/'+button.dataset.prepareProject+'/'+slug+'/prepare',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'文档预处理提交失败');monitor(data.job_id,'content_preparation',data.batch_id||'')}catch(error){alert(error.message||'文档预处理提交失败');button.disabled=false;button.textContent=original}});
const deleteDialog=document.querySelector('#delete-progress-dialog'),deleteSpinner=document.querySelector('#delete-progress-spinner'),deleteResult=document.querySelector('#delete-progress-result'),deleteTitle=document.querySelector('#delete-progress-title'),deleteMessage=document.querySelector('#delete-progress-message'),deleteDetail=document.querySelector('#delete-progress-detail'),deleteBar=document.querySelector('#delete-progress-bar'),deleteClose=document.querySelector('#delete-progress-close');let deletePollTimer=null;
function updateDeleteProgress(progress,title,detail){deleteBar.style.width=clampProgress(progress)+'%';deleteTitle.textContent=title;deleteDetail.textContent=detail||''}
function finishDeleteProgress(success,title,detail){deleteSpinner.hidden=true;deleteResult.hidden=false;deleteResult.className='delete-result '+(success?'success':'failure');deleteResult.textContent=success?'✓':'!';updateDeleteProgress(100,title,detail);deleteClose.disabled=false;if(success)setTimeout(()=>location.reload(),1300)}
async function pollDeleteJob(jobId){try{const response=await fetch('/api/jobs/'+jobId),job=await response.json();if(!response.ok)throw new Error(job.error||'删除任务不存在');const latest=(job.log||[]).at(-1)||'后台清理正在进行';updateDeleteProgress(job.progress||10,job.status==='queued'?'正在等待解析步骤退出':'正在清理派生数据',latest);if(job.status==='completed'){finishDeleteProgress(true,'删除完成','文档、知识库贡献、图谱数据和过程文件已完成清理。');return}if(job.status==='failed'||job.status==='cancelled'){finishDeleteProgress(false,'后台清理未完成',job.error_message||latest);return}deletePollTimer=setTimeout(()=>pollDeleteJob(jobId),700)}catch(error){finishDeleteProgress(false,'删除状态读取失败',error.message||'请刷新页面确认文档状态')}}
document.querySelectorAll('[data-delete]').forEach(button=>button.onclick=async()=>{if(!confirm('确认软删除该文档及其向量、实体提及和关系贡献吗？'))return;button.disabled=true;deleteSpinner.hidden=false;deleteResult.hidden=true;deleteResult.className='delete-result';deleteClose.disabled=true;deleteMessage.textContent='正在提交删除请求...';updateDeleteProgress(3,'正在取消相关任务','请稍候，不要重复点击删除。');deleteDialog?.showModal();try{const response=await fetch('/api/documents/'+button.dataset.delete+'/delete',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'删除失败');button.closest('tr')?.classList.add('deleting-row');deleteMessage.textContent='文档已从当前知识库移除，正在完成后台清理';updateDeleteProgress(8,'文档已软删除','正在等待活动解析步骤安全退出。');pollDeleteJob(data.job_id)}catch(error){button.disabled=false;finishDeleteProgress(false,'删除失败',error.message||'删除请求执行失败')}});
deleteClose?.addEventListener('click',()=>location.reload());deleteDialog?.addEventListener('cancel',event=>{if(deleteClose.disabled)event.preventDefault()});
document.querySelector('[data-reassess-schema]')?.addEventListener('click',async event=>{if(!confirm('将基于当前知识库全部已入库文档，从基础知识结构重新评估实体类别和关系约束。该操作不会自动重跑已有文档的实体抽取，是否继续？'))return;const button=event.currentTarget;button.disabled=true;try{const response=await fetch('/api/projects/'+button.dataset.reassessSchema+'/schema/reassess',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'知识结构重新评估提交失败');monitor(data.job_id,'schema_reassessment')}catch(error){alert(error.message||'知识结构重新评估提交失败');button.disabled=false}});
document.querySelector('[data-rebuild-graph]')?.addEventListener('click',async event=>{const button=event.currentTarget;button.disabled=true;try{const response=await fetch('/api/projects/'+button.dataset.rebuildGraph+'/graph',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'图谱重建提交失败');monitor(data.job_id,'graph_sync')}catch(error){alert(error.message||'图谱重建提交失败');button.disabled=false}});
const createDialog=document.querySelector('#create-project-dialog');
document.querySelector('#open-create-project')?.addEventListener('click',()=>createDialog?.showModal());
document.querySelectorAll('[data-close-create-project]').forEach(button=>button.addEventListener('click',()=>createDialog?.close()));
createDialog?.addEventListener('click',event=>{if(event.target===createDialog)createDialog.close()});
const settingsForm=document.querySelector('#project-settings-form'),llmSelect=settingsForm?.querySelector('select[name="llm_model"]'),embeddingSelect=settingsForm?.querySelector('select[name="embedding_model"]'),knowledgeLanguageSelect=settingsForm?.querySelector('select[name="knowledge_language"]'),modelTokenInput=settingsForm?.querySelector('input[name="model_max_tokens"]'),chunkTokenInput=settingsForm?.querySelector('input[name="chunk_max_tokens"]'),modelTokenHint=document.querySelector('#model-token-limit-hint'),chunkTokenHint=document.querySelector('#chunk-token-limit-hint');
settingsForm?.addEventListener('submit',event=>{{
  if(!knowledgeLanguageSelect||knowledgeLanguageSelect.value===knowledgeLanguageSelect.dataset.originalValue)return;
  if(!confirm('知识语言更改后不会改写当前知识库或图谱，将从下一次知识构建生效。是否保存？'))event.preventDefault();
}});
let tokenLimits=null;
function applyTokenLimits(clamp=false){if(!tokenLimits||!modelTokenInput||!chunkTokenInput)return;const llmMax=Number(tokenLimits.llm_context_max),embeddingMax=Number(tokenLimits.embedding_context_max);modelTokenInput.max=String(llmMax);if(clamp&&Number(modelTokenInput.value)>llmMax)modelTokenInput.value=String(llmMax);const chunkMax=Math.min(Number(modelTokenInput.value)||llmMax,llmMax,embeddingMax);chunkTokenInput.max=String(chunkMax);if(clamp&&Number(chunkTokenInput.value)>chunkMax)chunkTokenInput.value=String(chunkMax);if(modelTokenHint){modelTokenHint.textContent='当前 LLM 支持的最大上下文：'+llmMax.toLocaleString()+' Token';modelTokenHint.classList.remove('failure')}if(chunkTokenHint){chunkTokenHint.textContent='当前模型组合允许的最大切片：'+chunkMax.toLocaleString()+' Token（Embedding 上限 '+embeddingMax.toLocaleString()+'）';chunkTokenHint.classList.remove('failure')}}
async function refreshTokenLimits(clamp=false){if(!llmSelect||!embeddingSelect)return;if(modelTokenHint)modelTokenHint.textContent='正在读取当前 LLM 的上下文上限...';if(chunkTokenHint)chunkTokenHint.textContent='正在读取 LLM 与 Embedding 的共同上限...';try{const query=new URLSearchParams({llm_model:llmSelect.value,embedding_model:embeddingSelect.value});const response=await fetch('/api/models/limits?'+query),data=await response.json();if(!response.ok)throw new Error(data.error||'模型上限读取失败');tokenLimits=data;applyTokenLimits(clamp)}catch(error){if(modelTokenHint){modelTokenHint.textContent=error.message||'模型上限读取失败';modelTokenHint.classList.add('failure')}if(chunkTokenHint){chunkTokenHint.textContent='请确认当前模型提供方可用'}}}
function replaceModelOptions(select,models,kind){if(!select)return;const current=select.value,names=models.filter(model=>model.model_type===kind).map(model=>model.name);if(current&&!names.includes(current))names.unshift(current);select.replaceChildren(...names.map(name=>new Option(name,name,name===current,name===current)))}
document.querySelector('#sync-models')?.addEventListener('click',async e=>{const button=e.currentTarget,status=document.querySelector('#model-sync-status');button.disabled=true;if(status){status.hidden=true;status.className='model-sync-status'}try{const r=await fetch('/api/models/sync',{method:'POST'}),d=await r.json();if(!r.ok)throw new Error(d.error||'同步失败');replaceModelOptions(llmSelect,d.models||[],'llm');replaceModelOptions(embeddingSelect,d.models||[],'embedding');await refreshTokenLimits(true);if(status){status.textContent='已同步 '+d.count+' 个当前提供方模型';status.classList.add('success');status.hidden=false}}catch(error){if(status){status.textContent=error.message||'同步失败';status.classList.add('failure');status.hidden=false}}finally{button.disabled=false}});
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
document.querySelector('[data-graph-update-action="build"]')?.addEventListener('click',()=>{
  const buildButton=document.querySelector('#build-project-knowledge');
  if(buildButton&&!buildButton.disabled)buildButton.click();
});
document.querySelector('[data-graph-update-action="prepare"]')?.addEventListener('click',()=>{
  activateWorkspaceView('build',{updateHistory:true});
  document.querySelector('#prepare-project-documents')?.focus();
});
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
const jobStatus={queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',cancelled:'已终止'};
const nativeRuntimeDetails=document.querySelector('#native-runtime-details'),nativeRuntimeStatus=document.querySelector('#native-runtime-status'),nativeRuntimeProgressText=document.querySelector('#native-runtime-progress-text'),nativeRuntimeProgress=document.querySelector('#native-runtime-progress'),nativeRuntimeLogButton=nativeRuntimeDetails?.querySelector('[data-view-job-log]');
function updateNativeRuntime(job){if(!nativeRuntimeDetails||job.kind!=='graphrag_native_build')return;const progress=clampProgress(job.progress);nativeRuntimeDetails.dataset.nativeJob=job.id||'';if(nativeRuntimeStatus)nativeRuntimeStatus.textContent=jobStatus[job.status]||job.status||'未知状态';if(nativeRuntimeProgressText)nativeRuntimeProgressText.textContent=progress+'%';if(nativeRuntimeProgress)nativeRuntimeProgress.style.width=progress+'%';if(nativeRuntimeLogButton){nativeRuntimeLogButton.dataset.job=job.id||'';nativeRuntimeLogButton.disabled=!job.id}}
function jobStageSummary(job){const key=job.metadata?.active_subtask,step=jobSubtasks(job)[key];if(!key||!step)return'';const suffix=step.status==='completed'?'已完成':step.status==='running'?clampProgress(step.progress)+'%':step.status==='failed'?'失败':step.status==='cancelled'?'已终止':'等待';const labels=labelsForJob(job);return(labels[key]||key)+' · '+suffix+(step.detail?' · '+step.detail:'')}
async function refreshJobLog(id,batch,generation,firstLoad=false){try{const endpoint=batch?'/api/job-batches/'+batch:'/api/jobs/'+id,response=await fetch(endpoint),job=await response.json();if(generation!==jobLogGeneration||!jobDialog?.open)return;if(!response.ok)throw new Error(job.error||'任务不存在');const keepFollowing=firstLoad||jobLogFollow||jobLogNearBottom(),degraded=Number(job.metadata?.extraction_errors||0),stageSummary=jobStageSummary(job),count=Number(job.document_count||0);jobSummary.textContent=(jobStatus[job.status]||job.status)+(count?' · '+count+' 个文档':'')+' · 用时 '+formatDuration(job.elapsed_seconds)+(stageSummary?' · '+stageSummary:'')+(degraded?' · '+degraded+' 个知识块已降级':'');jobHistory.textContent=(job.log||[]).join('\n')||'该任务暂无日志。';jobError.hidden=true;if(job.error_message){jobError.textContent=job.error_message;jobError.hidden=false}if(keepFollowing){jobLogFollow=true;scrollJobLogToBottom();}if(['queued','running'].includes(job.status))jobLogPollTimer=setTimeout(()=>refreshJobLog(id,batch,generation),1000)}catch(error){if(generation!==jobLogGeneration||!jobDialog?.open)return;jobSummary.textContent='日志读取失败';jobError.textContent=error.message||'任务不存在';jobError.hidden=false}}
function showJobLog(id,batch=''){if(!id)return;stopJobLogPolling();const generation=jobLogGeneration;jobLogFollow=true;jobDialog?.showModal();jobSummary.textContent='正在读取任务日志...';jobHistory.textContent='';jobError.hidden=true;refreshJobLog(id,batch,generation,true)}
document.querySelectorAll('[data-view-job-log]').forEach(button=>button.addEventListener('click',()=>showJobLog(button.dataset.job,button.dataset.batch||'')));
jobHistory?.addEventListener('scroll',()=>{jobLogFollow=jobLogNearBottom()});
document.querySelectorAll('[data-close-job-log]').forEach(button=>button.addEventListener('click',()=>{stopJobLogPolling();jobDialog?.close()}));
jobDialog?.addEventListener('click',event=>{if(event.target===jobDialog){stopJobLogPolling();jobDialog.close()}});
const documentMediaDialog=document.querySelector('#document-media-dialog'),documentMediaTitle=document.querySelector('#document-media-title'),documentMediaSummary=document.querySelector('#document-media-summary'),documentMediaBody=document.querySelector('#document-media-body'),documentMediaAssets=new Map();let documentMediaSequence=0;
function safeDocumentMediaUrl(value){try{const url=new URL(String(value||''),location.origin);return ['http:','https:'].includes(url.protocol)?url.href:''}catch(_error){return''}}
function documentTableRows(text,metadata={}){
  const headers=Array.isArray(metadata.headers)?metadata.headers.map(value=>String(value??'')):[],values=Array.isArray(metadata.values)?metadata.values.map(value=>String(value??'')):[];
  if(headers.length||values.length)return [headers.length?headers:values.map((_value,index)=>'列 '+(index+1)),values].filter(row=>row.length);
  const rows=String(text||'').split(/\r?\n/).map(line=>line.trim()).filter(line=>line.startsWith('|')&&line.endsWith('|')&&line.length>2).map(line=>line.slice(1,-1).split('|').map(cell=>cell.trim()));
  return rows.filter(row=>!row.length||!row.every(cell=>/^:?-{3,}:?$/.test(cell.replace(/\s+/g,''))));
}
function documentTableHtml(rows,limit=0){
  const selected=limit>0?rows.slice(0,limit):rows;if(!selected.length)return'';
  const width=Math.max(...selected.map(row=>row.length)),normalized=selected.map(row=>[...row,...Array(Math.max(0,width-row.length)).fill('')]);
  return '<table><thead><tr>'+normalized[0].map(cell=>'<th>'+esc(cell)+'</th>').join('')+'</tr></thead><tbody>'+normalized.slice(1).map(row=>'<tr>'+row.map(cell=>'<td>'+esc(cell)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
}
function registerDocumentMedia(asset){const id='document-media-'+(++documentMediaSequence);documentMediaAssets.set(id,asset);return id}
function documentImagesMarkup(assets){
  const seen=new Set(),items=[];
  for(const raw of Array.isArray(assets)?assets:[]){const url=safeDocumentMediaUrl(raw?.url);if(!url||seen.has(url))continue;seen.add(url);const title=String(raw.title||'原文图片'),description=String(raw.description||''),id=registerDocumentMedia({kind:'image',url,title,description,width:Number(raw.width||0),height:Number(raw.height||0)});items.push('<figure class="structure-image-figure"><button class="structure-image-button" type="button" data-document-media="'+id+'" title="点击放大"><img src="'+esc(url)+'" alt="'+esc(title)+'" loading="lazy"></button><figcaption>'+esc(title)+(description?' · '+esc(description):'')+'</figcaption></figure>')}
  return items.length?'<div class="structure-media-gallery">'+items.join('')+'</div>':'';
}
function documentTableMarkup(text,metadata,title){
  const rows=documentTableRows(text,metadata);if(!rows.length)return'';const id=registerDocumentMedia({kind:'table',title:title||'表格',rows}),remaining=Math.max(0,rows.length-9);
  return '<div class="structure-table-preview">'+documentTableHtml(rows,9)+'<div class="structure-table-actions"><button class="button small secondary" type="button" data-document-media="'+id+'">'+(remaining?'查看完整表格（另有 '+remaining+' 行）':'放大表格')+'</button></div></div>';
}
function structureBlockMediaMarkup(block){const metadata=block?.metadata||{},images=block?.block_type==='image'?[metadata]:(metadata.media_assets||[]),table=['table','table_row'].includes(block?.block_type)?documentTableMarkup(block.text,metadata,block.section_title||'表格'):'';return documentImagesMarkup(images)+table}
function chunkMediaMarkup(chunk){const metadata=chunk?.metadata||{},tableMetadata=metadata.table&&typeof metadata.table==='object'?metadata.table:metadata,hasTabularRows=documentTableRows(chunk.content,tableMetadata).length>=2,isTable=Boolean(metadata.table)||String(metadata.chunk_role||'')==='table_row'||(Array.isArray(metadata.block_types)&&metadata.block_types.includes('table'))||hasTabularRows;return documentImagesMarkup(metadata.media_assets||[])+(isTable?documentTableMarkup(chunk.content,tableMetadata,chunk.section_title||'表格'):'')}
function documentPageLabel(values){const pages=[...new Set((Array.isArray(values)?values:[]).map(Number).filter(value=>Number.isInteger(value)&&value>0))].sort((left,right)=>left-right);if(!pages.length)return'';return pages.length===1?'第 '+pages[0]+' 页':'第 '+pages.slice(0,6).join('、')+(pages.length>6?' 等':'')+'页'}
function openDocumentMedia(id){const asset=documentMediaAssets.get(String(id||''));if(!asset||!documentMediaDialog)return;documentMediaTitle.textContent=asset.title||'图表预览';documentMediaSummary.textContent=asset.kind==='image'&&asset.width&&asset.height?asset.width+' × '+asset.height+' 像素':asset.kind==='table'?asset.rows.length+' 行表格':'';documentMediaBody.innerHTML=asset.kind==='image'?'<img src="'+esc(asset.url)+'" alt="'+esc(asset.title||'原文图片')+'">'+(asset.description?'<p>'+esc(asset.description)+'</p>':''):'<div class="document-media-table">'+documentTableHtml(asset.rows)+'</div>';documentMediaDialog.showModal()}
function resetDocumentMediaAssets(){documentMediaAssets.clear();documentMediaSequence=0}
const structureDialog=document.querySelector('#structure-dialog'),structureSummary=document.querySelector('#structure-summary'),structureError=document.querySelector('#structure-error'),structureTools=document.querySelector('#structure-tools'),structureList=document.querySelector('#structure-list'),structureSearch=document.querySelector('#structure-search'),structureType=document.querySelector('#structure-type'),structurePager=document.querySelector('#structure-pager'),structureRange=document.querySelector('#structure-range'),structurePrev=document.querySelector('#structure-prev'),structureNext=document.querySelector('#structure-next');
const structureTypeNames={heading:'标题',paragraph:'正文段落',table:'表格',table_row:'表格行',image:'图片'};let structureData=null,structurePage=1;const structurePageSize=20;
function resetPagedListScroll(container){
  if(!container)return;
  container.scrollTop=0;container.scrollLeft=0;
  requestAnimationFrame(()=>{container.scrollTop=0;container.scrollLeft=0});
}
function filteredStructureBlocks(){if(!structureData)return[];const query=(structureSearch.value||'').trim().toLowerCase(),type=structureType.value;return structureData.blocks.filter(block=>{if(type&&block.block_type!==type)return false;if(!query)return true;return [block.text,block.section_title,(block.heading_path||[]).join(' / '),block.page].some(value=>String(value??'').toLowerCase().includes(query))})}
function renderStructureBlocks(){resetDocumentMediaAssets();const blocks=filteredStructureBlocks(),pages=Math.max(1,Math.ceil(blocks.length/structurePageSize));structurePage=Math.min(Math.max(1,structurePage),pages);const start=(structurePage-1)*structurePageSize,visible=blocks.slice(start,start+structurePageSize);structureList.innerHTML=visible.map(block=>{const path=(block.heading_path||[]).filter(Boolean).join(' / '),metadata=block.metadata&&Object.keys(block.metadata).length?'<details class="structure-metadata"><summary>查看元数据</summary><pre>'+esc(JSON.stringify(block.metadata,null,2))+'</pre></details>':'';return '<article class="structure-row"><div class="structure-row-head"><div class="structure-row-meta"><strong>#'+esc(block.order)+'</strong><span class="block-type">'+esc(structureTypeNames[block.block_type]||block.block_type||'未知')+'</span>'+(block.page?'<span>第 '+esc(block.page)+' 页</span>':'')+'<span>层级 '+esc(block.level||1)+'</span><span>章节：'+esc(block.section_title||'未命名')+'</span></div></div>'+(path?'<div class="structure-row-meta">标题路径：'+esc(path)+'</div>':'')+'<p class="structure-text">'+esc(block.text||'')+'</p>'+structureBlockMediaMarkup(block)+metadata+'</article>'}).join('')||'<div class="empty">没有符合条件的结构块。</div>';structureRange.textContent=blocks.length?'第 '+(start+1)+'-'+Math.min(start+structurePageSize,blocks.length)+' 条，共 '+blocks.length+' 条':'共 0 条';structurePrev.disabled=structurePage<=1;structureNext.disabled=structurePage>=pages;structurePager.hidden=false;resetPagedListScroll(structureList)}
async function showStructure(documentId){structureDialog?.showModal();structureSummary.textContent='正在读取解析结果...';structureError.hidden=true;structureTools.hidden=true;structurePager.hidden=true;structureList.innerHTML='<div class="empty">正在加载结构块...</div>';structureSearch.value='';structureType.replaceChildren(new Option('全部类型',''));structureData=null;try{const response=await fetch('/api/documents/'+documentId+'/structure'),data=await response.json();if(!response.ok)throw new Error(data.error||'结构块读取失败');structureData=data;const parser=data.plan?.effective_parser||'未知解析器',strategy=data.plan?.effective_strategy||'未知策略';structureSummary.textContent=data.document_name+' · '+parser+' · '+strategy+' · '+data.block_count+' 个结构块';for(const type of data.block_types||[])structureType.add(new Option(structureTypeNames[type]||type,type));structureTools.hidden=false;structurePage=1;renderStructureBlocks()}catch(error){structureSummary.textContent='结构块读取失败';structureError.textContent=error.message||'结构块读取失败';structureError.hidden=false;structureList.innerHTML=''}}
document.querySelectorAll('[data-view-structure]').forEach(button=>button.addEventListener('click',()=>showStructure(button.dataset.viewStructure)));
structureSearch?.addEventListener('input',()=>{structurePage=1;renderStructureBlocks()});structureType?.addEventListener('change',()=>{structurePage=1;renderStructureBlocks()});structurePrev?.addEventListener('click',()=>{structurePage--;renderStructureBlocks()});structureNext?.addEventListener('click',()=>{structurePage++;renderStructureBlocks()});document.querySelectorAll('[data-close-structure]').forEach(button=>button.addEventListener('click',()=>structureDialog?.close()));structureDialog?.addEventListener('click',event=>{if(event.target===structureDialog)structureDialog.close()});
const chunkDialog=document.querySelector('#chunk-dialog'),chunkSummary=document.querySelector('#chunk-summary'),chunkError=document.querySelector('#chunk-error'),chunkTools=document.querySelector('#chunk-tools'),chunkList=document.querySelector('#chunk-list'),chunkSearch=document.querySelector('#chunk-search'),chunkRole=document.querySelector('#chunk-role'),chunkPager=document.querySelector('#chunk-pager'),chunkRange=document.querySelector('#chunk-range'),chunkPrev=document.querySelector('#chunk-prev'),chunkNext=document.querySelector('#chunk-next');let chunkData=null,chunkPage=1;const chunkPageSize=12;
function filteredChunks(){if(!chunkData)return[];const query=(chunkSearch.value||'').trim().toLowerCase(),role=chunkRole.value;return chunkData.chunks.filter(chunk=>{const metadata=chunk.metadata||{},chunkRoleValue=metadata.chunk_role||'child';if(role&&chunkRoleValue!==role)return false;if(!query)return true;return [chunk.chunk_key,chunk.section_title,chunk.content].some(value=>String(value??'').toLowerCase().includes(query))})}
function renderChunks(){
  resetDocumentMediaAssets();const chunks=filteredChunks(),pageCount=Math.max(1,Math.ceil(chunks.length/chunkPageSize));chunkPage=Math.min(Math.max(1,chunkPage),pageCount);const start=(chunkPage-1)*chunkPageSize,visible=chunks.slice(start,start+chunkPageSize);
  chunkList.innerHTML=visible.map((chunk,index)=>{
    const metadata=chunk.metadata||{},role=metadata.chunk_role||'child',enabled=metadata.graph_extraction_enabled!==false,parent=metadata.parent_key||'',pageLabel=documentPageLabel(metadata.pages);
    return '<article class="structure-row"><div class="structure-row-head"><div class="structure-row-meta"><strong>#'+(start+index+1)+'</strong><span class="block-type">'+esc(chunk.section_title||'未命名章节')+'</span><span>'+esc(chunk.chunk_key)+'</span></div></div><div class="chunk-badges"><span class="chunk-badge">'+esc(role==='parent'?'父级摘要块':role==='table_row'?'表格行块':'检索子块')+'</span><span class="chunk-badge">'+esc(chunk.token_count??metadata.token_count??0)+' Token</span>'+(pageLabel?'<span class="chunk-badge">'+esc(pageLabel)+'</span>':'')+'<span class="chunk-badge '+(enabled?'graph':'skipped')+'">'+(enabled?'参与图谱抽取':'不参与图谱抽取')+'</span>'+(parent?'<span class="chunk-badge">父块：'+esc(parent)+'</span>':'')+'</div><p class="structure-text">'+esc(chunk.content||'')+'</p>'+chunkMediaMarkup(chunk)+'<details class="structure-metadata"><summary>查看元数据</summary><pre>'+esc(JSON.stringify(metadata,null,2))+'</pre></details></article>';
  }).join('')||'<div class="empty">没有符合条件的知识块。</div>';
  chunkRange.textContent=chunks.length?'第 '+(start+1)+'-'+Math.min(start+chunkPageSize,chunks.length)+' 条，共 '+chunks.length+' 条':'共 0 条';chunkPrev.disabled=chunkPage<=1;chunkNext.disabled=chunkPage>=pageCount;chunkPager.hidden=false;
  resetPagedListScroll(chunkList);
}
async function showChunks(documentId){chunkDialog?.showModal();chunkSummary.textContent='正在读取分块结果...';chunkError.hidden=true;chunkTools.hidden=true;chunkPager.hidden=true;chunkList.innerHTML='<div class="empty">正在加载知识块...</div>';chunkSearch.value='';chunkRole.value='';chunkData=null;try{const response=await fetch('/api/documents/'+documentId+'/chunks'),data=await response.json();if(!response.ok)throw new Error(data.error||'知识块读取失败');chunkData=data;chunkSummary.textContent=data.document_name+' · '+data.chunk_count+' 个知识块';chunkTools.hidden=false;chunkPage=1;renderChunks()}catch(error){chunkSummary.textContent='知识块读取失败';chunkError.textContent=error.message||'知识块读取失败';chunkError.hidden=false;chunkList.innerHTML=''}}
document.querySelectorAll('[data-view-chunks]').forEach(button=>button.addEventListener('click',()=>showChunks(button.dataset.viewChunks)));chunkSearch?.addEventListener('input',()=>{chunkPage=1;renderChunks()});chunkRole?.addEventListener('change',()=>{chunkPage=1;renderChunks()});chunkPrev?.addEventListener('click',()=>{chunkPage--;renderChunks()});chunkNext?.addEventListener('click',()=>{chunkPage++;renderChunks()});document.querySelectorAll('[data-close-chunks]').forEach(button=>button.addEventListener('click',()=>chunkDialog?.close()));chunkDialog?.addEventListener('click',event=>{if(event.target===chunkDialog)chunkDialog.close()});
[structureList,chunkList].forEach(container=>container?.addEventListener('click',event=>{const trigger=event.target.closest('[data-document-media]');if(trigger&&container.contains(trigger))openDocumentMedia(trigger.dataset.documentMedia)}));
document.querySelectorAll('[data-close-document-media]').forEach(button=>button.addEventListener('click',()=>documentMediaDialog?.close()));documentMediaDialog?.addEventListener('click',event=>{if(event.target===documentMediaDialog)documentMediaDialog.close()});
const nativeArtifactDialog=document.querySelector('#native-artifact-dialog'),nativeArtifactTitle=document.querySelector('#native-artifact-title'),nativeArtifactSummary=document.querySelector('#native-artifact-summary'),nativeArtifactError=document.querySelector('#native-artifact-error'),nativeArtifactKind=document.querySelector('#native-artifact-kind'),nativeArtifactContent=document.querySelector('#native-artifact-content');
async function loadNativeArtifact(kind){nativeArtifactError.hidden=true;nativeArtifactContent.textContent='正在读取...';try{const response=await fetch('/api/projects/'+window.KF.projectId+'/graphrag-native/artifacts/'+encodeURIComponent(kind)+'?limit=100'),data=await response.json();if(!response.ok)throw new Error(data.error||'构建产物读取失败');nativeArtifactTitle.textContent='GraphRAG 构建产物';nativeArtifactSummary.textContent=data.logical_name+' · 显示 '+data.items.length+' / '+data.total+' 条 · 运行编号 '+data.run_id;nativeArtifactContent.textContent=JSON.stringify(data.items,null,2)}catch(error){nativeArtifactSummary.textContent='读取失败';nativeArtifactError.textContent=error.message||'构建产物读取失败';nativeArtifactError.hidden=false;nativeArtifactContent.textContent=''}}
async function showNativeDocuments(documentId){nativeArtifactDialog?.showModal();nativeArtifactKind.closest('.structure-tools').hidden=true;nativeArtifactError.hidden=true;nativeArtifactContent.textContent='正在生成结构化文档预览...';try{const response=await fetch('/api/projects/'+window.KF.projectId+'/graphrag-native/source-documents?document_id='+encodeURIComponent(documentId)+'&limit=100'),data=await response.json();if(!response.ok)throw new Error(data.error||'结构化文档读取失败');nativeArtifactTitle.textContent='结构化文档';nativeArtifactSummary.textContent='显示 '+data.items.length+' / '+data.total+' 条；这是本次 GraphRAG 构建使用的文档内容。';nativeArtifactContent.textContent=JSON.stringify(data.items,null,2)}catch(error){nativeArtifactSummary.textContent='读取失败';nativeArtifactError.textContent=error.message||'结构化文档读取失败';nativeArtifactError.hidden=false;nativeArtifactContent.textContent=''}}
document.querySelectorAll('[data-view-native-documents]').forEach(button=>button.addEventListener('click',()=>showNativeDocuments(button.dataset.viewNativeDocuments)));
document.querySelector('#view-native-artifacts')?.addEventListener('click',()=>{nativeArtifactDialog?.showModal();nativeArtifactKind.closest('.structure-tools').hidden=false;nativeArtifactKind.value='documents';loadNativeArtifact('documents')});
nativeArtifactKind?.addEventListener('change',()=>loadNativeArtifact(nativeArtifactKind.value));
document.querySelectorAll('[data-close-native-artifacts]').forEach(button=>button.addEventListener('click',()=>nativeArtifactDialog?.close()));nativeArtifactDialog?.addEventListener('click',event=>{if(event.target===nativeArtifactDialog)nativeArtifactDialog.close()});
const toast=document.querySelector('.notice');if(toast){setTimeout(()=>toast.classList.add('toast-hide'),4200);setTimeout(()=>toast.remove(),4550);const url=new URL(location.href);url.searchParams.delete('kind');url.searchParams.delete('message');history.replaceState({},'',url.pathname+url.search)}
function searchSourceMarkup(record){
  const info=record?.source_info||{},filename=String(info.filename||record?.original_filename||'').trim(),baseUrl=String(info.document_url||record?.storage_path||'').trim();
  const firstPage=Array.isArray(info.pages)?Number(info.pages[0]||0):0,url=baseUrl&&firstPage>0&&baseUrl.toLowerCase().includes('.pdf')&&!baseUrl.includes('#')?baseUrl+'#page='+firstPage:baseUrl;
  const location=[info.page_label,info.section_title].map(value=>String(value||'').trim()).filter(Boolean).join(' · '),media=[];
  for(const image of Array.isArray(info.media_assets)?info.media_assets:[]){
    const imageUrl=String(image?.url||'').trim();if(!imageUrl)continue;
    media.push('<a class="result-media-thumb" href="'+esc(imageUrl)+'" target="_blank" rel="noopener" title="打开原图"><img src="'+esc(imageUrl)+'" alt="'+esc(image.title||'原文图片')+'" loading="lazy"></a>');
  }
  if(info.table&&String(info.table.text||'').trim())media.push('<details class="result-table"><summary>查看表格</summary><pre>'+esc(info.table.text)+'</pre></details>');
  if(!filename&&!location&&!url&&!media.length)return '';
  const documentMarkup=url?'<a href="'+esc(url)+'" target="_blank" rel="noopener">'+esc(filename||'查看原文')+'</a>':(filename?'<b>'+esc(filename)+'</b>':'');
  return '<div class="result-source">'+documentMarkup+(location?(documentMarkup?' · ':'')+esc(location):'')+(media.length?'<div class="result-media">'+media.join('')+'</div>':'')+'</div>';
}
function resultGroupMarkup(title,items,render){
  const rows=Array.isArray(items)?items:[];
  return '<section class="result-group"><h2>'+esc(title)+'</h2>'+(rows.map(render).join('')||'<p class="muted">没有匹配结果。</p>')+'</section>';
}
function communityReportResultMarkup(report){
  const findings=(Array.isArray(report.findings)?report.findings:[]).slice(0,4).map(item=>'<li><b>'+esc(item.title||'结论')+'</b>：'+esc(item.explanation||'')+'</li>').join('');
  const sources=(Array.isArray(report.sources)?report.sources:[]).slice(0,4).map(source=>'<div class="result-source-item">'+(source.source_title?'<p><b>'+esc(source.source_title)+'</b> '+esc(source.predicate_label||'关系')+' <b>'+esc(source.target_title||'')+'</b></p>':'')+(source.evidence_text?'<p>'+esc(source.evidence_text)+'</p>':'')+searchSourceMarkup(source)+'</div>').join('');
  return '<article class="result-card"><h3>'+esc(report.title||'社区报告')+' <span class="score">'+Number(report.score||0).toFixed(3)+'</span></h3><p>'+esc(report.summary||'')+'</p>'+(findings?'<h4 class="result-report-findings-title">关键结论</h4><ul class="result-report-findings">'+findings+'</ul>':'')+(report.limitations?'<p class="muted">说明：'+esc(report.limitations)+'</p>':'')+sources+'</article>';
}
document.querySelector('#search-form')?.addEventListener('submit',async event=>{
  event.preventDefault();const target=document.querySelector('#search-results'),query=String(new FormData(event.currentTarget).get('query')||'');target.innerHTML='<p class="muted">正在检索...</p>';
  try{
    const response=await fetch('/api/search?project='+window.KF.projectId+'&q='+encodeURIComponent(query)),data=await response.json();if(!response.ok)throw new Error(data.error||'检索失败');
    const reportState=data.community_report_retrieval||{},reportStatus=String(reportState.status||'not_started'),reportNote=['queued','running','not_started'].includes(reportStatus)?'<p class="result-retrieval-note">详细社区报告仍在后台生成；当前检索已使用知识块、实体和关系，已完成的社区报告也会参与排序。</p>':'';
    target.innerHTML=reportNote
      +resultGroupMarkup('知识块',data.chunks,x=>'<article class="result-card"><h3>'+esc(x.section_title||x.object_key)+' <span class="score">'+Number(x.score||0).toFixed(3)+'</span></h3><p>'+esc(x.content||'')+'</p>'+searchSourceMarkup(x)+'</article>')
      +resultGroupMarkup('实体',data.entities,x=>'<article class="result-card"><h3>'+esc(x.name)+' <span class="muted">'+esc(x.entity_type_label||x.entity_type||'通用实体')+'</span></h3><p>'+esc(x.description||'')+'</p></article>')
      +resultGroupMarkup('关系',data.relations,x=>'<article class="result-card"><h3>'+esc(x.source_name)+' <span class="relation">'+esc(x.relation_type_label||'关系')+'</span> '+esc(x.target_name)+'</h3><p>'+esc(x.description||'')+'</p>'+searchSourceMarkup(x)+'</article>')
      +resultGroupMarkup('社区报告',data.community_reports,communityReportResultMarkup);
  }catch(error){target.innerHTML='<p class="failure">'+esc(error.message||'检索失败')+'</p>'}
});
const nativeQueryForm=document.querySelector('#native-search-form'),nativeModeHelp=document.querySelector('[data-native-query-mode-help]');
const nativeModeInfo={
  local:{label:'详细问答',help:'综合相关实体、关系、社区报告和原文片段，适合询问具体对象及其联系。'},
  global:{label:'全局总结',help:'综合各社区报告回答全局性问题，适合总结整个资料集的主题、趋势和格局。'},
  drift:{label:'主题深入分析',help:'从相关社区出发继续追问和扩展证据，适合分析复杂主题及其相互影响。'},
  basic:{label:'原文问答',help:'主要依据最相关的文本单元回答，适合查找具体事实或原文依据。'},
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
if(graphRoot&&graphRoot.dataset.ready==='true'&&false){
  const q=selector=>graphRoot.querySelector(selector);
  const NS='http://www.w3.org/2000/svg';
  const colors=['#2677d8','#b54cda','#d85c49','#119c87','#d18b18','#5974d9','#c45379','#1d9aa7','#7b8b3d','#8b5e3c'];
  const typeLabels={GenericEntity:'通用实体',Person:'人物',Organization:'组织机构',Location:'地点',Product:'产品或材料',Technology:'技术或方法',Event:'事件'};
  const relationLabels={related_to:'相关',part_of:'组成部分',belongs_to:'隶属或归属',located_in:'位于',uses:'使用',produces:'生产或创建',depends_on:'依赖',causes:'导致',executes_task:'执行任务',supply_to:'供应给',supports:'支持',manages:'管理',transports:'运输',member_of:'成员属于',represents:'代表'};
  const graphState={nodes:[],edges:[],groups:[],nodeById:new Map(),query:'',group:'',showIsolates:true,selectedNode:'',selectedEdge:null,positions:new Map(),totalEntities:0,totalRelationships:0,totalCommunities:0,communityReady:false,zoom:1,worldWidth:1200,worldHeight:720,viewCenterX:null,viewCenterY:null};
  const panState={active:false,moved:false,pointerId:null,startClientX:0,startClientY:0,startCenterX:0,startCenterY:0};
  const clean=value=>String(value??'').replace(/\s+/g,' ').trim();
  const short=(value,size=52)=>{const result=clean(value);return result.length>size?result.slice(0,size-1)+'…':result};
  const displayType=value=>typeLabels[clean(value)]||clean(value)||'通用实体';
  const colorFor=value=>colors[Math.abs([...String(value)].reduce((sum,char)=>sum+char.charCodeAt(0),0))%colors.length];
  const svg=(name,attributes={})=>{const element=document.createElementNS(NS,name);for(const [key,value] of Object.entries(attributes))element.setAttribute(key,String(value));return element};
  const groupLabel=group=>clean(group.label||group.key||'未分组');
  const compactCommunityLabels=groups=>{
    const ordered=[...groups].sort((left,right)=>Number(left.level??0)-Number(right.level??0)||Number(left.community??left.display_index??0)-Number(right.community??right.display_index??0)||String(left.key).localeCompare(String(right.key)));
    let displayIndex=0;
    return ordered.map(group=>{
      if(String(group.key)==='未分组'||String(group.key)==='knowledge-forge-unconnected')return group;
      return {...group,display_index:displayIndex,label:'社区 '+displayIndex++};
    });
  };
  const edgeText=edge=>clean(edge.predicate_label||relationLabels[clean(edge.predicate)]||edge.description||'相关');
  const evidenceText=edge=>{
    if(clean(edge.evidence_text))return clean(edge.evidence_text);
    const ids=Array.isArray(edge.text_unit_ids)?edge.text_unit_ids:[];
    return ids.length?'文本单元：'+ids.join(', '):'—';
  };
  function normalizeNative(graph,communities,reports){
    const nodes=(graph.nodes||[]).map((node,index)=>({...node,id:String(node.id||node.title||index),title:clean(node.title||node.name||node.id),type:displayType(node.type),group:'未分组'}));
    const nodeById=new Map(nodes.map(node=>[String(node.id),node]));
    const byTitle=new Map(nodes.map(node=>[String(node.title),node]));
    const reportByKey=new Map((reports||[]).map(item=>[String(item.level??0)+':'+String(item.community??item.id??''),item]));
    const groups=[];
    for(const item of communities||[]){
      const key=String(item.level??0)+':'+String(item.community??item.id??'');
      const report=reportByKey.get(key)||{};
      const group={key,community:Number(item.community??item.id??0),level:Number(item.level??0),label:'',size:Number(item.size||(item.entity_ids||[]).length||0),relationship_count:Number((item.relationship_ids||[]).length||0),summary:clean(report.summary||report.full_content||report.description||'暂无社区摘要')};
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
    if(nodes.some(node=>node.group==='未分组'))groups.push({key:'未分组',label:'未归入社区',size:nodes.filter(node=>node.group==='未分组').length,relationship_count:0,summary:'当前知识图谱没有为这些实体分配社区。'});
    const displayGroups=compactCommunityLabels(groups);
    return {nodes,edges,groups:displayGroups,totalEntities:Number(graph.total_entities||nodes.length),totalRelationships:Number(graph.total_relationships||edges.length),totalCommunities:displayGroups.length,communityReady:true};
  }
  function normalizeKnowledgeForge(graph){
    return {
      nodes:(graph.nodes||[]).map(node=>({...node,id:String(node.id),title:clean(node.title||node.name||node.id),group:String(node.group||node.type||'其他')})),
      edges:(graph.edges||[]).map((edge,index)=>({...edge,id:String(edge.id||index),source_id:String(edge.source),target_id:String(edge.target),source_title:clean(edge.source_title||edge.source),target_title:clean(edge.target_title||edge.target)})),
      groups:compactCommunityLabels(graph.groups||[]),
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
    const network=q('[data-graph-network]'),zoom=Math.max(.25,Math.min(8,graphState.zoom));
    const width=graphState.worldWidth/zoom,height=graphState.worldHeight/zoom;
    let centerX=Number.isFinite(graphState.viewCenterX)?graphState.viewCenterX:graphState.worldWidth/2;
    let centerY=Number.isFinite(graphState.viewCenterY)?graphState.viewCenterY:graphState.worldHeight/2;
    if(!Number.isFinite(graphState.viewCenterX)&&graphState.selectedNode&&graphState.positions.has(graphState.selectedNode)){
      const point=graphState.positions.get(graphState.selectedNode);centerX=point.x;centerY=point.y;
    }else if(!Number.isFinite(graphState.viewCenterX)&&graphState.selectedEdge){
      const source=graphState.positions.get(graphState.selectedEdge.source_id),target=graphState.positions.get(graphState.selectedEdge.target_id);
      if(source&&target){centerX=(source.x+target.x)/2;centerY=(source.y+target.y)/2}
    }
    const minX=Math.min(0,graphState.worldWidth-width),maxX=Math.max(0,graphState.worldWidth-width);
    const minY=Math.min(0,graphState.worldHeight-height),maxY=Math.max(0,graphState.worldHeight-height);
    const x=Math.max(minX,Math.min(maxX,centerX-width/2));
    const y=Math.max(minY,Math.min(maxY,centerY-height/2));
    network.setAttribute('viewBox',[x,y,width,height].join(' '));
    q('[data-graph-zoom-value]').textContent=Math.round(zoom*100)+'%';
  }
  function setZoom(value){graphState.zoom=Math.max(.25,Math.min(8,Number(value)||1));updateViewport()}
  function resetViewport(){graphState.zoom=1;graphState.viewCenterX=null;graphState.viewCenterY=null}
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
      panel.innerHTML='<h3>关系详情</h3>'+relationCard(edge)+'<dl><div><dt>关系说明</dt><dd>'+esc(clean(edge.description)||edgeText(edge))+'</dd></div><div><dt>强度</dt><dd>'+esc(edge.confidence??edge.weight??'—')+'</dd></div><div><dt>原文证据</dt><dd>'+esc(evidenceText(edge))+'</dd></div></dl>';
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
        const selected=graphState.groups.find(group=>String(group.key)===String(graphState.group));
        status.innerHTML=selected
          ?'<h3>'+esc(groupLabel(selected))+'</h3><p>'+esc(selected.summary||'当前社区的实体和关系已在左侧图中筛选显示。')+(Number(selected.level||0)>0?' 聚类层级：'+esc(selected.level)+'。':'')+'</p>'
          :'<h3>社区聚类</h3><p>当前图谱包含 '+esc(graphState.totalCommunities||graphState.groups.length)+' 个社区。颜色、分区和下方卡片均按社区划分；未连接实体会合并显示。</p>';
      }else{
        status.innerHTML='<h3>社区尚未生成</h3><p class="failure">当前图谱版本没有对应的社区聚类结果。下方仅按实体类别临时布局，不代表社区；重新构建后将自动生成社区。</p>';
      }
    }
    const visibleGroups=graphState.group?graphState.groups.filter(group=>String(group.key)===String(graphState.group)):graphState.groups;
    box.innerHTML=visibleGroups.map(group=>'<button type="button" class="embedded-graph-group-card'+(graphState.group===String(group.key)?' is-active':'')+'" data-embedded-group="'+esc(group.key)+'"><i style="background:'+colorFor(group.key)+'"></i><b>'+esc(groupLabel(group))+'</b><small>'+esc(group.size??0)+' 个实体 · '+esc(group.relationship_count??0)+' 条关系'+(Number(group.level||0)>0?' · 聚类层级 '+esc(group.level):'')+'</small><span>'+esc(short(group.summary||'',120))+'</span></button>').join('');
    box.querySelectorAll('[data-embedded-group]').forEach(button=>button.addEventListener('click',()=>selectEmbeddedGroup(button.dataset.embeddedGroup)));
  }
  function selectEmbeddedGroup(key){graphState.group=graphState.group===String(key)?'':String(key);q('[data-graph-group-filter]').value=graphState.group;clearSelection();renderAll()}
  function renderAudit(edges){
    q('[data-graph-audit-summary]').textContent='显示 '+edges.length+' / '+graphState.totalRelationships+' 条关系；点击“图谱连接”可查看网络结构。';
    q('[data-graph-audit-rows]').innerHTML=edges.map(edge=>'<tr><td>'+esc(edge.source_title)+'</td><td>'+esc(edgeText(edge))+'</td><td>'+esc(edge.target_title)+'</td><td>'+esc(edge.confidence??edge.weight??'—')+'</td><td>'+esc(evidenceText(edge))+'</td></tr>').join('');
  }
  function renderGraph(){
    const {nodes,edges}=visibleGraph(),labels=layout(nodes,edges);
    const labelLayer=q('[data-graph-group-labels]'),edgeLayer=q('[data-graph-edges]'),nodeLayer=q('[data-graph-nodes]');
    labelLayer.replaceChildren();edgeLayer.replaceChildren();nodeLayer.replaceChildren();
    for(const item of labels){
      const title=item.label+' ('+item.count+')';
      if(item.key==='__isolates__'){
        const label=svg('text',{x:item.x,y:item.y,class:'embedded-graph-group-label',fill:colorFor(item.key)});label.textContent=title;labelLayer.append(label);continue;
      }
      const trigger=svg('g',{class:'embedded-graph-group-trigger',transform:'translate('+Math.max(0,item.x-5)+' '+Math.max(0,item.y-19)+')',role:'button',tabindex:'0','aria-label':'筛选'+item.label});
      const background=svg('rect',{x:0,y:0,width:Math.max(94,title.length*14+18),height:27,rx:6});
      const label=svg('text',{x:8,y:19,class:'embedded-graph-group-label',fill:colorFor(item.key)});label.textContent=title;
      const activate=()=>selectEmbeddedGroup(item.key);
      trigger.addEventListener('click',event=>{event.stopPropagation();activate()});
      trigger.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();activate()}});
      trigger.append(background,label);labelLayer.append(trigger);
    }
    const selected=graphState.selectedNode,neighbours=new Set();
    if(selected)for(const edge of edges){if(edge.source_id===selected)neighbours.add(edge.target_id);if(edge.target_id===selected)neighbours.add(edge.source_id)}
    const selectedEdgeIds=new Set(graphState.selectedEdge?[graphState.selectedEdge.source_id,graphState.selectedEdge.target_id]:[]);
    for(const edge of edges){
      const source=graphState.positions.get(edge.source_id),target=graphState.positions.get(edge.target_id);if(!source||!target)continue;
      const related=graphState.selectedEdge?graphState.selectedEdge.id===edge.id:(!selected||edge.source_id===selected||edge.target_id===selected);
      const line=svg('line',{x1:source.x,y1:source.y,x2:target.x,y2:target.y,class:'embedded-graph-edge'+(related?'':' is-dim')+(graphState.selectedEdge?.id===edge.id?' is-selected':''),'marker-end':'url(#embedded-graph-arrow)'});
      line.style.strokeWidth=String(Math.min(5,1+Math.log1p(Number(edge.weight||edge.confidence)||1)));const title=svg('title');title.textContent=edge.source_title+' → '+edge.target_title+'\n'+edgeText(edge);line.append(title);
      line.addEventListener('click',event=>{event.stopPropagation();graphState.viewCenterX=null;graphState.viewCenterY=null;graphState.selectedEdge=edge;graphState.selectedNode='';renderGraph();renderDetails()});edgeLayer.append(line);
    }
    for(const node of nodes){
      const point=graphState.positions.get(String(node.id));if(!point)continue;
      const relevant=graphState.selectedEdge?selectedEdgeIds.has(String(node.id)):(!selected||String(node.id)===selected||neighbours.has(String(node.id)));
      const group=svg('g',{class:'embedded-graph-node'+(relevant?'':' is-dim')+(String(node.id)===selected?' is-selected':''),transform:'translate('+point.x+' '+point.y+')'});
      const circle=svg('circle',{r:Math.min(20,9+Math.log1p(Number(node.degree)||0)*3),fill:colorFor(node.group)}),title=svg('title');title.textContent=node.title+'\n'+String(node.type||'')+' · '+String(node.degree||0)+' 个连接';circle.append(title);
      const label=svg('text',{x:0,y:29});label.textContent=short(node.title,20);group.append(circle,label);group.addEventListener('click',event=>{event.stopPropagation();graphState.viewCenterX=null;graphState.viewCenterY=null;graphState.selectedNode=String(node.id);graphState.selectedEdge=null;renderGraph();renderDetails()});nodeLayer.append(group);
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
        const failed=responses.findIndex(response=>!response.ok);if(failed>=0)throw Error(payloads[failed].error||'知识图谱读取失败');
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
  q('[data-apply-graph-filter]').addEventListener('click',()=>{graphState.query=q('[data-graph-filter]').value;graphState.viewCenterX=null;graphState.viewCenterY=null;clearSelection();renderAll()});
  q('[data-graph-filter]').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();q('[data-apply-graph-filter]').click()}});
  q('[data-graph-group-filter]').addEventListener('change',event=>{graphState.group=event.target.value;graphState.viewCenterX=null;graphState.viewCenterY=null;clearSelection();renderAll()});
  q('[data-toggle-graph-isolates]').addEventListener('change',event=>{graphState.showIsolates=event.target.checked;graphState.viewCenterX=null;graphState.viewCenterY=null;clearSelection();renderAll()});
  q('[data-graph-zoom-out]').addEventListener('click',()=>setZoom(graphState.zoom/1.25));
  q('[data-graph-zoom-in]').addEventListener('click',()=>setZoom(graphState.zoom*1.25));
  q('[data-graph-fit]').addEventListener('click',()=>{resetViewport();updateViewport()});
  q('[data-reset-graph]').addEventListener('click',()=>{graphState.query='';graphState.group='';graphState.showIsolates=true;resetViewport();clearSelection();q('[data-graph-filter]').value='';q('[data-graph-group-filter]').value='';q('[data-toggle-graph-isolates]').checked=true;renderAll()});
  const graphNetwork=q('[data-graph-network]');
  graphNetwork.style.cursor='grab';
  graphNetwork.addEventListener('wheel',event=>{
    event.preventDefault();
    const rect=graphNetwork.getBoundingClientRect(),box=graphNetwork.viewBox.baseVal;
    const ratioX=(event.clientX-rect.left)/Math.max(1,rect.width),ratioY=(event.clientY-rect.top)/Math.max(1,rect.height);
    const pointX=box.x+ratioX*box.width,pointY=box.y+ratioY*box.height;
    const next=Math.max(.25,Math.min(8,graphState.zoom*Math.exp(-event.deltaY*.0015)));
    const nextWidth=graphState.worldWidth/next,nextHeight=graphState.worldHeight/next;
    graphState.viewCenterX=pointX+(.5-ratioX)*nextWidth;
    graphState.viewCenterY=pointY+(.5-ratioY)*nextHeight;
    setZoom(next);
  },{passive:false});
  graphNetwork.addEventListener('pointerdown',event=>{
    if(event.button!==0||event.target!==graphNetwork)return;
    const box=graphNetwork.viewBox.baseVal;
    panState.active=true;panState.moved=false;panState.pointerId=event.pointerId;
    panState.startClientX=event.clientX;panState.startClientY=event.clientY;
    panState.startCenterX=box.x+box.width/2;panState.startCenterY=box.y+box.height/2;
    graphNetwork.setPointerCapture(event.pointerId);graphNetwork.style.cursor='grabbing';
  });
  graphNetwork.addEventListener('pointermove',event=>{
    if(!panState.active||event.pointerId!==panState.pointerId)return;
    const rect=graphNetwork.getBoundingClientRect(),box=graphNetwork.viewBox.baseVal;
    const dx=event.clientX-panState.startClientX,dy=event.clientY-panState.startClientY;
    if(Math.abs(dx)+Math.abs(dy)>3)panState.moved=true;
    graphState.viewCenterX=panState.startCenterX-dx*box.width/Math.max(1,rect.width);
    graphState.viewCenterY=panState.startCenterY-dy*box.height/Math.max(1,rect.height);
    updateViewport();
  });
  const finishPan=event=>{
    if(!panState.active||event.pointerId!==panState.pointerId)return;
    panState.active=false;graphNetwork.style.cursor='grab';
    if(graphNetwork.hasPointerCapture(event.pointerId))graphNetwork.releasePointerCapture(event.pointerId);
  };
  graphNetwork.addEventListener('pointerup',finishPan);graphNetwork.addEventListener('pointercancel',finishPan);
  graphNetwork.addEventListener('click',()=>{if(panState.moved){panState.moved=false;return}clearSelection();renderGraph();renderDetails()});
  loadEmbeddedGraph();
}
}

/* Shared, bounded GPU graph browser.  The preceding SVG implementation is
   intentionally not entered while this adapter is being kept for downgrade
   compatibility with already-rendered pages. */
{
const root=document.querySelector('#embedded-graph-browser');
if(root&&root.dataset.ready==='true'){
  const q=selector=>root.querySelector(selector);
  const clean=value=>String(value??'').replace(/\s+/g,' ').trim();
  const short=(value,size=180)=>{const text=clean(value);return text.length>size?text.slice(0,size-1)+'…':text};
  const html=value=>esc(String(value??''));
  const state={view:null,overview:null,renderer:null,community:'',communityLevel:'full',overviewRelationScope:'all_communities',query:'',mode:'overview',loading:false,history:[],historyIndex:-1,entityMetricMode:'evidence',reportPollTimer:0,reportStatus:'not_started',evidenceAssets:new Map(),evidenceAssetSequence:0};
  const endpoint='/api/projects/'+window.KF.projectId+'/graph-explorer';
  const reportEndpoint='/api/projects/'+window.KF.projectId+'/knowledge-forge/community-reports';
  const graphContainer=q('[data-graph-network]');
  const isKnowledgeForge=root.dataset.strategy==='knowledge_forge';
  const relationshipLabel=isKnowledgeForge?'已验证关系':'GraphRAG 抽取关系';
  const evidenceLabel=isKnowledgeForge?'原文证据':'文本单元依据';
  const evidenceHeading=isKnowledgeForge?'原文依据':'文本单元依据';
  const evidenceLocationHeading=isKnowledgeForge?'原文依据与来源':'文本单元依据与来源';
  const evidenceModalTitle=isKnowledgeForge?'原文资料':'文本单元资料';
  const groupName=group=>clean(group?.label||group?.title||group?.key||'未分组');
  function summaryCommunityTopic(group){
    const summary=clean(group?.summary||group?.description);if(!summary)return '';
    const focus=summary.match(/^该社区主要围绕[“"]([^”"]+)[”"]展开/u);
    if(focus)return clean(focus[1]);
    const pair=summary.match(/^该社区主要介绍[“"]([^”"]+)[”"]与[“"]([^”"]+)[”"]之间/u);
    if(pair)return clean(pair[1])+' 与 '+clean(pair[2]);
    const quoted=[...summary.matchAll(/[“"]([^”"]{1,64})[”"]/gu)].map(match=>clean(match[1])).filter(Boolean);
    return [...new Set(quoted)].slice(0,2).join(' 与 ');
  }
  const communityTopic=group=>clean(group?.semantic_title||group?.community_report?.title).replace(/^(?:社区|Community)\s*[0-9A-Za-z_-]+\s*[：:]\s*/iu,'').trim()||summaryCommunityTopic(group);
  const groupNavigationName=group=>{const base=groupName(group),topic=communityTopic(group);return topic&&topic!==base?base+' · '+short(topic,36):base};
  const communityCanvasName=(group,node)=>communityTopic(group)||clean(node?.title||node?.label)||groupName(group);
  const groupForNode=node=>(state.overview?.groups||state.view?.groups||[]).find(group=>String(group?.key||'')===String(node?.community_key||node?.group||''));
  const nodeName=node=>node?.kind==='community'?communityCanvasName(groupForNode(node),node):clean(node?.title||node?.label||node?.id||'未知实体');
  const relationName=edge=>clean(edge?.label||edge?.predicate_label||relationLabels[clean(edge?.predicate)]||edge?.description||'关系');
  const evidence=edge=>clean(edge?.evidence_text)||((edge?.text_unit_ids||[]).length?'文本单元编号：'+edge.text_unit_ids.join('、'):'暂无可展示的'+evidenceLabel);
  const evidenceCount=edge=>{if(edge?.evidence_count!==null&&edge?.evidence_count!==undefined&&edge?.evidence_count!==''){const explicit=Number(edge.evidence_count);if(Number.isFinite(explicit)&&explicit>=0)return Math.round(explicit)}if(Array.isArray(edge?.text_unit_ids))return edge.text_unit_ids.length;return clean(edge?.evidence_text)?1:0};
  const confidenceValue=edge=>{if(edge?.confidence===null||edge?.confidence===undefined||edge?.confidence==='')return null;const value=Number(edge.confidence);return Number.isFinite(value)&&value>=0&&value<=1?value:null};
  const metricNumber=value=>{const number=Number(value);return Number.isFinite(number)?String(Math.round(number*100)/100):'—'};
  function entityMetricMode(edges){
    const confidences=edges.map(confidenceValue).filter(value=>value!==null),confidenceValues=new Set(confidences.map(value=>value.toFixed(4)));
    if(confidences.length===edges.length&&confidenceValues.size>1)return 'confidence';
    const weights=edges.filter(edge=>edge?.weight!==null&&edge?.weight!==undefined&&edge?.weight!=='').map(edge=>Number(edge.weight)).filter(Number.isFinite),weightValues=new Set(weights.map(value=>value.toFixed(4)));
    if(!isKnowledgeForge&&weights.length===edges.length&&weightValues.size>1)return 'weight';
    return 'evidence';
  }
  const entityMetricLabel=mode=>mode==='confidence'?'置信度':mode==='weight'?'关系强度':isKnowledgeForge?'支持证据':'文本依据';
  const entityMetricValue=(edge,mode)=>mode==='confidence'?metricNumber(confidenceValue(edge)):mode==='weight'?metricNumber(edge?.weight):evidenceCount(edge)+' 条';
  const colorFor=value=>['#2878d0','#11a089','#d58b16','#a052cf','#e05d49','#557c19','#3574b8','#c84e91'][Math.abs([...String(value??'')].reduce((sum,char)=>sum+char.charCodeAt(0),0))%8];
  const miniNode=node=>'<span class="graph-mini-node"><span class="graph-entity-dot" style="background:'+colorFor(node?.group||node?.type)+'"></span><span>'+html(nodeName(node))+'</span></span>';
  const successfulReport=report=>['succeeded','reused'].includes(clean(report?.status));
  const pendingReport=group=>['pending','running','queued','not_started'].includes(clean(group?.community_report?.status||group?.report_status||'not_started'));
  function communityReportHeading(report){
    const title=clean(report?.title),topic=title.replace(/^社区\s*[0-9A-Za-z_-]+\s*[：:]\s*/u,'').trim();
    return topic?'详细报告：'+topic:'详细报告';
  }
  function reportPreviewMarkup(group){
    const report=group?.community_report,status=clean(report?.status||group?.report_status||'not_started');
    if(successfulReport(report)){
      const body=report.summary||report.content||'';
      const heading=clean(group?.semantic_title)?'详细报告':communityReportHeading(report);
      return '<span class="graph-community-card-report"><b>'+html(short(heading,52))+'</b>'+html(short(body,180))+'</span>';
    }
    if(['pending','running','queued','not_started'].includes(status))return '<span class="graph-community-card-report"><b>详细报告</b><span class="graph-report-loading"><i class="graph-report-spinner" aria-hidden="true"></i>生成中…</span></span>';
    if(status==='deferred')return '<span class="graph-community-card-report"><b>详细报告</b>打开该社区时按需生成。</span>';
    if(status==='failed')return '<span class="graph-community-card-report"><b>详细报告</b>本次生成未完成，社区简介和'+html(evidenceLabel)+'仍可查看。</span>';
    if(status==='skipped')return '<span class="graph-community-card-report"><b>详细报告</b>该社区事实较少，使用社区简介展示。</span>';
    return '';
  }
  function registerEvidenceAsset(asset){const id='graph-evidence-'+(++state.evidenceAssetSequence);state.evidenceAssets.set(id,asset);return id}
  function sourceDocumentUrl(info){
    const raw=clean(info?.document_url);if(!raw)return '';
    const page=Array.isArray(info?.pages)&&Number(info.pages[0])>0?Number(info.pages[0]):0;
    return page&&raw.toLowerCase().includes('.pdf')?raw+'#page='+page:raw;
  }
  function sourceLocationMarkup(info){
    if(!info||typeof info!=='object')return '<span>来源位置暂不可用</span>';
    const name=clean(info.filename)||((Number(info.document_id)||0)>0?'文档 '+info.document_id:'来源文档'),url=sourceDocumentUrl(info);
    const parts=[info.page_label,info.section_title].map(clean).filter(Boolean);
    return (url?'<a href="'+html(url)+'" target="_blank" rel="noopener">'+html(name)+'</a>':'<b>'+html(name)+'</b>')+(parts.length?' · '+html(parts.join(' · ')):'');
  }
  function sourceMediaMarkup(info){
    if(!info||typeof info!=='object')return '';
    const buttons=[];
    for(const image of Array.isArray(info.media_assets)?info.media_assets:[]){
      if(!clean(image?.url))continue;
      const id=registerEvidenceAsset({kind:'image',title:image.title||'文档图片',url:image.url,description:image.description||''});
      buttons.push('<button type="button" class="graph-media-thumb" data-graph-evidence-asset="'+id+'" title="点击放大"><img src="'+html(image.url)+'" alt="'+html(image.title||'文档图片')+'" loading="lazy"></button>');
    }
    if(info.table&&clean(info.table.text)){
      const id=registerEvidenceAsset({kind:'table',title:info.table.title||'表格内容',text:info.table.text});
      buttons.push('<button type="button" class="graph-media-thumb" data-graph-evidence-asset="'+id+'">查看表格</button>');
    }
    return buttons.length?'<div class="graph-source-media">'+buttons.join('')+'</div>':'';
  }
  function sourceCardMarkup(record,{showEvidence=true}={}){
    const info=record?.source_info||record||{},relation=[record?.source_title,record?.predicate_label,record?.target_title].map(clean).filter(Boolean).join(' → ');
    return '<div class="graph-source-card">'+(relation?'<div class="graph-source-relation">'+html(relation)+'</div>':'')+'<div class="graph-source-location">'+sourceLocationMarkup(info)+'</div>'+(showEvidence&&clean(record?.evidence_text)?'<p>'+html(record.evidence_text)+'</p>':'')+sourceMediaMarkup(info)+'</div>';
  }
  function findingSources(finding,report){
    const direct=Array.isArray(finding?.sources)?finding.sources.filter(Boolean):[];
    if(direct.length)return direct;
    const relationKeys=new Set((Array.isArray(finding?.relation_keys)?finding.relation_keys:[]).map(clean).filter(Boolean));
    return (Array.isArray(report?.sources)?report.sources:[]).filter(source=>relationKeys.has(clean(source?.relation_key)));
  }
  function findingEvidenceMarkup(finding,report){
    const sources=findingSources(finding,report);
    if(!sources.length)return '<p class="graph-finding-evidence-missing">该条结论暂时没有可展示的'+html(evidenceLabel)+'。</p>';
    const firstInfo=sources[0]?.source_info||{},firstLocation=[clean(firstInfo.filename),clean(firstInfo.page_label)].filter(Boolean).join(' · '),location=firstLocation+(sources.length>1?' 等 '+sources.length+' 处':'');
    const sourceSummary=isKnowledgeForge?'依据 '+sources.length+' 条已验证关系':'关联 '+sources.length+' 项关系或文本单元依据';
    return '<details class="graph-finding-evidence"><summary><span>'+html(sourceSummary)+'</span>'+(location?'<small>'+html(short(location,72))+'</small>':'')+'</summary><div class="graph-source-list">'+sources.map(source=>sourceCardMarkup(source,{showEvidence:true})).join('')+'</div></details>';
  }
  function bindEvidenceActions(container=root){
    container.querySelectorAll('[data-graph-evidence-asset]').forEach(button=>button.addEventListener('click',event=>{
      event.preventDefault();event.stopPropagation();
      const asset=state.evidenceAssets.get(button.dataset.graphEvidenceAsset);if(!asset)return;
      const dialog=q('[data-graph-evidence-modal]'),body=q('[data-graph-evidence-modal-body]');
      q('[data-graph-evidence-modal-title]').textContent=asset.title||evidenceModalTitle;
      body.innerHTML=asset.kind==='image'?'<img src="'+html(asset.url)+'" alt="'+html(asset.title||'文档图片')+'">'+(asset.description?'<p>'+html(asset.description)+'</p>':''):'<pre>'+html(asset.text||'')+'</pre>';
      if(typeof dialog.showModal==='function')dialog.showModal();else dialog.setAttribute('open','');
    }));
  }
  function communityReportMarkup(group){
    const report=group?.community_report,status=clean(report?.status||group?.report_status||'not_started');
    if(['pending','running','queued','not_started'].includes(status))return '<section class="graph-community-report"><h4>详细报告</h4><p class="graph-report-loading"><i class="graph-report-spinner" aria-hidden="true"></i>正在根据该社区的实体、关系和'+html(evidenceLabel)+'生成详细报告…</p></section>';
    if(status==='deferred')return '<section class="graph-community-report"><h4>详细报告</h4><p class="graph-report-loading"><i class="graph-report-spinner" aria-hidden="true"></i>已按需加入生成队列…</p></section>';
    if(status==='failed')return '<section class="graph-community-report"><h4>详细报告</h4><p>本次报告生成未完成；社区简介、关系和'+html(evidenceLabel)+'不受影响。</p></section>';
    if(status==='skipped')return '<section class="graph-community-report"><h4>详细报告</h4><p>该社区事实较少，不单独生成详细报告。</p></section>';
    if(!successfulReport(report))return '';
    const findings=(report.findings||[]).map(item=>'<li><b>'+html(item.title||'结论')+'</b><p>'+html(item.explanation||'')+'</p>'+findingEvidenceMarkup(item,report)+'</li>').join('');
    const limitations=clean(report.limitations)?'<details class="graph-report-limitations"><summary>报告范围与限制</summary><p>'+html(report.limitations)+'</p></details>':'';
    const findingsHelp=isKnowledgeForge?'每条结论均可展开查看关系与原文':'每条结论均可展开查看关系与文本单元';
    return '<section class="graph-community-report"><h4>'+html(communityReportHeading(report))+'</h4><p class="graph-community-report-summary">'+html(report.summary||'')+'</p>'+(findings?'<div class="graph-report-findings-heading"><h5>关键结论</h5><small>'+html(findingsHelp)+'</small></div><ul class="graph-report-findings">'+findings+'</ul>':'')+limitations+'</section>';
  }
  function setBusy(message='正在读取图谱...'){
    state.loading=true;
    q('[data-graph-status]').hidden=false;
    q('[data-graph-status]').innerHTML='<div class="embedded-graph-loading"><span class="embedded-graph-spinner" aria-hidden="true"></span><div><b>'+html(message)+'</b><p>大型图谱会按社区和实体邻域分批加载。</p></div></div>';
  }
  function displayPayload(payload){
    const sourceNodes=Array.isArray(payload.nodes)?payload.nodes:[];
    const edges=Array.isArray(payload.edges)?payload.edges:[];
    if(payload.mode==='overview'){
      const groupsByKey=new Map((payload.groups||[]).map(group=>[String(group.key||''),group]));
      const nodes=sourceNodes.map(node=>{
        if(node?.kind!=='community')return node;
        const group=groupsByKey.get(String(node.community_key||node.group||''));
        return {...node,title:communityCanvasName(group,node)};
      });
      return {...payload,nodes,edges};
    }
    const connected=new Set();
    edges.forEach(edge=>{connected.add(String(edge.source));connected.add(String(edge.target))});
    return {...payload,nodes:sourceNodes.filter(node=>connected.has(String(node.id))),edges};
  }
  function normalizedRoute(parameters={}){
    return {
      mode:parameters.mode||'overview',community:parameters.community||'',entity:parameters.entity||'',q:parameters.q||'',
      community_level:parameters.community_level||'',overview_relation_scope:parameters.overview_relation_scope||'',limit:Number(parameters.limit||1200),
    };
  }
  function routeEquals(left,right){return JSON.stringify(left)===JSON.stringify(right)}
  function syncNavigation(){
    const overview=(state.view||{}).mode==='overview';
    q('[data-graph-history-back]').disabled=state.loading||state.historyIndex<=0;
    q('[data-graph-history-forward]').disabled=state.loading||state.historyIndex>=state.history.length-1;
    q('[data-graph-return-overview]').disabled=state.loading||overview;
  }
  function recordRoute(parameters){
    const route=normalizedRoute(parameters),current=state.history[state.historyIndex];
    if(routeEquals(route,current))return;
    state.history.splice(state.historyIndex+1);state.history.push(route);state.historyIndex=state.history.length-1;
  }
  async function requestView(parameters,{history='push'}={}){
    if(state.loading)return false;
    const route=normalizedRoute(parameters);
    if(history==='push')recordRoute(route);
    syncNavigation();
    setBusy(route.mode==='community'?'正在打开社区...':route.mode==='neighbours'?'正在加载实体邻域...':'正在读取图谱...');
    try{
      const url=new URL(endpoint,location.origin);
      url.searchParams.set('strategy',root.dataset.strategy);
      Object.entries(route).forEach(([key,value])=>{if(value!==''&&value!=null)url.searchParams.set(key,String(value))});
      const response=await fetch(url),payload=await response.json();
      if(!response.ok)throw Error(payload.error||'图谱读取失败');
      state.view=payload;if(['overview','relations'].includes(payload.mode))state.overview=payload;
      q('.embedded-graph-viewer').hidden=false;
      await renderView();
      q('[data-graph-status]').hidden=true;
      return true;
    }catch(error){
      q('[data-graph-status]').hidden=false;
      q('[data-graph-status]').innerHTML='<div class="empty-state failure">图谱加载失败：'+html(error.message||error)+'</div>';
      throw error;
    }finally{state.loading=false;syncNavigation()}
  }
  function nodeById(id){return (state.view?.nodes||[]).find(node=>String(node.id)===String(id))}
  function renderAudit(edges){
    const communityOverview=(state.view||{}).mode==='overview';
    const auditableEdges=communityOverview?edges.filter(edge=>edge.kind!=='community_topic_association'):edges;
    const topicCount=communityOverview?edges.length-auditableEdges.length:0;
    q('[data-graph-audit-summary]').textContent=communityOverview
      ?'当前视图共 '+auditableEdges.length+' 条社区间事实关系汇总；进入社区后可查看实体关系及其'+evidenceLabel+'。'+(topicCount?' 另有 '+topicCount+' 条内容导航关联，不属于关系审计。':'')
      :'当前视图共 '+auditableEdges.length+' 条实体关系；可逐条查看其'+evidenceLabel+'。';
    state.entityMetricMode=entityMetricMode(auditableEdges);
    q('[data-graph-audit-metric]').textContent=communityOverview?'汇总关系数':entityMetricLabel(state.entityMetricMode);
    q('[data-graph-audit-evidence]').hidden=communityOverview;
    q('[data-graph-audit-rows]').innerHTML=auditableEdges.map(edge=>{
      const source=nodeById(edge.source),target=nodeById(edge.target);
      const metric=communityOverview?edge.count??edge.weight??'—':entityMetricValue(edge,state.entityMetricMode);
      const evidenceMarkup='<div class="graph-evidence-quote">'+html(evidence(edge))+'</div>'+sourceCardMarkup(edge,{showEvidence:false});
      return '<tr><td>'+html(nodeName(source)||edge.source_title||edge.source)+'</td><td>'+html(relationName(edge))+'</td><td>'+html(nodeName(target)||edge.target_title||edge.target)+'</td><td>'+html(metric)+'</td>'+(communityOverview?'':'<td>'+evidenceMarkup+'</td>')+'</tr>';
    }).join('')||'<tr><td colspan="'+(communityOverview?'4':'5')+'">当前视图没有可展示的关系。</td></tr>';
    bindEvidenceActions(q('[data-graph-audit-rows]'));
  }
  function communityNarrativeParts(value){
    let overview=String(value??'').replace(/\s+/g,' ').trim(),relations='',sourceEvidence='';
    const evidenceMarker='可核对的原文要点：',evidenceIndex=overview.indexOf(evidenceMarker);
    if(evidenceIndex>=0){
      sourceEvidence=overview.slice(evidenceIndex+evidenceMarker.length).trim();
      overview=overview.slice(0,evidenceIndex).trim();
    }
    const relationMarker='图中主要用',relationIndex=overview.indexOf(relationMarker);
    if(relationIndex>=0){
      relations=overview.slice(relationIndex).trim();
      overview=overview.slice(0,relationIndex).trim();
    }
    return {
      overview:overview||'当前社区暂时没有可展示的简介。',
      relations,
      sourceEvidence:sourceEvidence.replace(/[。.!！？?]\s*$/,''),
    };
  }
  function communityNarrativeMarkup(value){
    const parts=communityNarrativeParts(value);
    return '<div class="graph-community-description">'
      +'<section class="graph-community-section"><h4>社区简介</h4><p>'+html(parts.overview)+'</p></section>'
      +(parts.relations?'<section class="graph-community-section graph-community-relations"><h4>关系概览</h4><p>'+html(parts.relations)+'</p></section>':'')
      +'</div>';
  }
  function communityEvidenceMarkup(value,report,evidenceSources){
    const evidenceText=communityNarrativeParts(value).sourceEvidence;
    const allReportSources=Array.isArray(report?.sources)?report.sources:[],assignedRelationKeys=new Set((Array.isArray(report?.findings)?report.findings:[]).flatMap(item=>[...(Array.isArray(item?.relation_keys)?item.relation_keys:[]),...(Array.isArray(item?.sources)?item.sources.map(source=>source?.relation_key):[])]).map(clean).filter(Boolean)),reportSources=allReportSources.filter(item=>!assignedRelationKeys.has(clean(item?.relation_key))),locations=new Set();
    const locationKey=record=>{const info=record?.source_info||record||{};return [info.document_id,info.filename,info.page_label,info.section_title].map(clean).join('|')};
    allReportSources.forEach(item=>locations.add(locationKey(item)));
    const sources=[...reportSources];
    for(const item of Array.isArray(evidenceSources)?evidenceSources:[]){const key=locationKey(item);if(!locations.has(key)){locations.add(key);sources.push(item)}}
    if(!evidenceText&&!sources.length)return '';
    return '<details class="graph-community-sources"><summary><span>'+html(evidenceLocationHeading)+'</span><small>'+(sources.length?sources.length+' 处':'')+'</small></summary>'+(evidenceText?'<blockquote>'+html(evidenceText)+'</blockquote>':'')+(sources.length?'<div class="graph-source-list">'+sources.map(item=>sourceCardMarkup(item,{showEvidence:Boolean(clean(item?.evidence_text))})).join('')+'</div>':'')+'</details>';
  }
  function renderOverviewDetails(){
    const panel=q('[data-graph-details]'),view=state.view||{};
    if(view.mode==='overview'){
      const shown=view.visible_communities??(view.nodes||[]).length,collapsed=view.collapsed_communities??0,internalOnly=view.internal_only_communities??0,isolated=view.isolated_entities??0;
      const scope=view.overview_relation_scope||'all_communities',totalEdges=view.total_cross_community_relationships??(view.edges||[]).length,visibleEdges=view.visible_cross_community_relationships??(view.edges||[]).length;
      const topicEdges=view.visible_topic_associations??0;
      const includesInternal=scope==='all_communities',visibleInternal=view.visible_internal_only_communities??0,hiddenInternal=view.hidden_internal_only_communities??internalOnly;
      panel.innerHTML='<h3>全局社区地图</h3><p>'+(includesInternal?'显示 '+shown+' 个含关系的社区，其中 '+visibleInternal+' 个社区没有跨社区事实联系。':'显示 '+shown+' 个存在跨社区事实联系的社区。')+' 圆点大小代表成员数，实线粗细代表'+html(relationshipLabel)+'数量。'+(topicEdges?'虚线表示内容相近的社区，仅用于浏览，不代表原文事实。':'')+'</p><p class="graph-scene-note">跨社区事实联系：'+visibleEdges+' / '+totalEdges+' 条'+(topicEdges?' · 内容导航关联：'+topicEdges+' 条':'')+'</p>'+(collapsed?'<p class="graph-scene-note">另有 '+collapsed+' 个较小社区折叠在“其他社区”中，可用下方目录或选择器进入。</p>':'')+(!includesInternal&&internalOnly?'<p class="graph-scene-note">另有 '+internalOnly+' 个仅含内部关系的社区未显示；切换到“全部关系社区”即可同时查看。</p>':'')+(includesInternal&&hiddenInternal?'<p class="graph-scene-note">其余 '+hiddenInternal+' 个仅含内部关系的社区仍可从目录或选择器进入。</p>':'')+(isolated?'<p class="graph-scene-note">另有 '+isolated+' 个没有关系的实体未放入地图，仍可通过搜索查看。</p>':'');
    }else if(view.mode==='community'){
      const summary=view.focus?.description||view.focus?.summary||'这里显示该社区内的实体及关系。';
      panel.innerHTML='<article class="graph-community-detail"><header class="graph-community-detail-header"><h3>'+html(groupName(view.focus))+'</h3></header>'+communityNarrativeMarkup(summary)+communityReportMarkup(view.focus)+communityEvidenceMarkup(summary,view.focus?.community_report,view.focus?.evidence_sources)+'</article>';
      bindEvidenceActions(panel);
    }
    else if(view.mode==='search')panel.innerHTML='<h3>筛选结果</h3><p>匹配 '+(view.matched_entities??(view.nodes||[]).length)+' 个实体，并带出它们的直接相邻关系。点击实体可查看完整邻域。</p>';
    else panel.innerHTML='<h3>实体邻域</h3><p>当前仅显示所选实体及其直接相邻实体和关系。</p>';
  }
  function renderNodeDetails(node){
    if(node.kind==='community'&&node.community_key){
      const panel=q('[data-graph-details]');
      panel.innerHTML='<h3>'+html(nodeName(node))+'</h3><p>正在打开该社区的实体与关系…</p>';
      openCommunity(node.community_key).catch(error=>{
        panel.innerHTML='<h3>'+html(nodeName(node))+'</h3><p class="failure">打开社区失败：'+html(error.message||error)+'</p>';
      });
      return;
    }
    if(node.kind==='community_rollup'){
      q('[data-graph-details]').innerHTML='<h3>'+html(nodeName(node))+'</h3><p>'+html(node.description||'该节点汇总了未在当前地图逐一显示的较小社区。')+'</p><button class="button primary" type="button" data-open-community-directory>打开社区目录</button>';
      q('[data-graph-details] [data-open-community-directory]')?.addEventListener('click',()=>q('[data-graph-group-filter]').focus());
      return;
    }
    const edges=(state.view?.edges||[]).filter(edge=>String(edge.source)===String(node.id)||String(edge.target)===String(node.id));
    const cards=edges.slice(0,30).map(edge=>{
      const source=nodeById(edge.source),target=nodeById(edge.target);
      return '<button type="button" class="graph-relation-card button-card" data-common-edge="'+html(edge.id)+'">'+miniNode(source)+'<span class="graph-mini-relation"><b>'+html(short(relationName(edge),24))+'</b></span>'+miniNode(target)+'</button>';
    }).join('');
    const panel=q('[data-graph-details]');
    panel.innerHTML='<div class="graph-entity-heading"><span class="graph-entity-dot" style="background:'+colorFor(node.group||node.type)+'"></span><h3>'+html(nodeName(node))+'</h3></div><dl><div><dt>实体类别</dt><dd>'+html(node.type||'实体')+'</dd></div><div><dt>当前视图连接数</dt><dd>'+edges.length+'</dd></div><div><dt>实体说明</dt><dd>'+html(node.description||'暂无说明')+'</dd></div></dl><button class="button primary graph-neighbour-button" type="button" data-load-neighbours="'+html(node.id)+'">查看完整邻域</button><h4>相邻关系</h4>'+(cards||'<p>当前视图没有相邻关系。</p>');
    panel.querySelector('[data-load-neighbours]')?.addEventListener('click',event=>requestView({mode:'neighbours',entity:event.currentTarget.dataset.loadNeighbours,limit:800}));
    panel.querySelectorAll('[data-common-edge]').forEach(button=>button.addEventListener('click',()=>{
      const edge=(state.view?.edges||[]).find(item=>String(item.id)===button.dataset.commonEdge);if(edge)renderEdgeDetails(edge);
    }));
  }
  function renderEdgeDetails(edge){
    const source=nodeById(edge.source),target=nodeById(edge.target);
    if(edge.kind==='community_topic_association'){
      const similarity=Number(edge.score),similarityText=Number.isFinite(similarity)?Math.round(similarity*100)+'%':'—';
      const threshold=Number(edge.threshold),thresholdText=Number.isFinite(threshold)&&threshold>0?Math.round(threshold*100)+'%':'';
      const reasonLabels={mutual_nearest:'两个社区互为最相近的合格主题',coverage_nearest:'这是该社区最有代表性的合格内容关联',reciprocal_top3:'两个社区都位于对方最相近的三个主题中'};
      const reasons=(Array.isArray(edge.selection_reasons)?edge.selection_reasons:[]).map(value=>reasonLabels[value]).filter(Boolean);
      const signals=[];if(reasons.length)signals.push(reasons.join('；'));if(Number(edge.shared_document_count)>0)signals.push('共同出现在 '+Number(edge.shared_document_count)+' 个来源文档中');if(edge.lexical_support)signals.push('名称中还存在可区分的共同词汇线索');if(!signals.length)signals.push('两个社区的代表实体向量相似度达到保守阈值');
      const matches=(Array.isArray(edge.representative_entity_matches)?edge.representative_entity_matches:[]).slice(0,5);
      const matchMarkup=matches.length?'<section class="graph-community-link-section"><h4>代表实体对应</h4><ul class="graph-topic-match-list">'+matches.map(item=>{const score=Number(item.score),scoreText=Number.isFinite(score)?Math.round(score*100)+'%':'—';return '<li><span>'+html(item.source_entity_name||item.source_entity_key||'实体')+' ↔ '+html(item.target_entity_name||item.target_entity_key||'实体')+'</span><small>'+html(scoreText)+'</small></li>'}).join('')+'</ul></section>':'';
      const panel=q('[data-graph-details]');
      panel.innerHTML='<h3>社区内容关联</h3><div class="graph-relation-card">'+miniNode(source)+'<span class="graph-mini-relation"><b>'+html(short(edge.label||'内容相近',24))+'</b></span>'+miniNode(target)+'</div><div class="graph-community-link-metrics"><div class="graph-community-link-metric"><small>内容相似度</small><b>'+html(similarityText)+'</b></div>'+(thresholdText?'<div class="graph-community-link-metric"><small>最低入选阈值</small><b>'+html(thresholdText)+'</b></div>':'')+'</div><section class="graph-community-link-section"><h4>为什么相连</h4><p>'+html(signals.join('；'))+'。</p></section>'+matchMarkup+'<p class="graph-community-link-note">这条虚线用于发现内容相近的社区，不表示原文事实，也不进入关系审计或知识检索。</p>';
      return;
    }
    const communityAggregate=edge.kind==='community_relation';
    if(communityAggregate){
      const total=Math.max(0,Number(edge.count??edge.weight??0)||0),predicateTotal=Math.max(0,Number(edge.predicate_type_count)||0),pairTotal=Math.max(0,Number(edge.entity_pair_count)||0),documentTotal=Math.max(0,Number(edge.document_count)||0);
      const predicates=(Array.isArray(edge.predicate_counts)?edge.predicate_counts:[]).filter(item=>Number(item?.count)>0),predicateMarkup=predicates.length?'<section class="graph-community-link-section"><h4>关系构成</h4><div class="graph-predicate-profile">'+predicates.map(item=>{const count=Number(item.count)||0,share=total?Math.max(2,Math.min(100,Math.round(count/total*100))):0;return '<div class="graph-predicate-row"><span>'+html(item.label||'关系')+'</span><b>'+count+' 条</b><i style="--predicate-share:'+share+'%"></i></div>'}).join('')+'</div></section>':'';
      const direction=edge.direction_counts||{},forward=Math.max(0,Number(direction.source_to_target)||0),reverse=Math.max(0,Number(direction.target_to_source)||0),directionMarkup=(forward||reverse)?'<section class="graph-community-link-section"><h4>事实方向</h4><p>'+html(nodeName(source))+' → '+html(nodeName(target))+'：'+forward+' 条；'+html(nodeName(target))+' → '+html(nodeName(source))+'：'+reverse+' 条。</p></section>':'';
      const examples=(Array.isArray(edge.relation_examples)?edge.relation_examples:[]).slice(0,5),examplesMarkup=examples.length?'<section class="graph-community-link-section"><h4>可核对关系 <small>展示 '+examples.length+' / '+total+' 条</small></h4><div class="graph-community-relation-examples">'+examples.map(item=>sourceCardMarkup(item,{showEvidence:true})).join('')+'</div></section>':'<p class="graph-community-link-note">当前聚合结果没有可展示的关系样例；关系数量与类型分布仍来自当前图谱中的实体关系。</p>';
      const aggregateNote=isKnowledgeForge?'实线由已经发布的实体关系汇总而成；下方样例保留原文、页码以及可用的图片或表格。':'实线由 GraphRAG 抽取的实体关系汇总而成；下方样例保留可用的文本单元来源。';
      const panel=q('[data-graph-details]');
      panel.innerHTML='<h3>社区间事实联系</h3><div class="graph-relation-card">'+miniNode(source)+'<span class="graph-mini-relation"><b>'+html(short(edge.label||'社区间联系',24))+'</b></span>'+miniNode(target)+'</div><div class="graph-community-link-metrics"><div class="graph-community-link-metric"><small>实体关系</small><b>'+total+' 条</b></div><div class="graph-community-link-metric"><small>关系类型</small><b>'+predicateTotal+' 种</b></div><div class="graph-community-link-metric"><small>实体对</small><b>'+pairTotal+' 组</b></div><div class="graph-community-link-metric"><small>来源文档</small><b>'+(documentTotal||'—')+'</b></div></div>'+predicateMarkup+directionMarkup+examplesMarkup+'<p class="graph-community-link-note">'+html(aggregateNote)+'</p>';
      bindEvidenceActions(panel);
      return;
    }
    const mode=state.entityMetricMode||entityMetricMode(state.view?.edges||[]),metric='<div><dt>'+html(entityMetricLabel(mode))+'</dt><dd>'+html(entityMetricValue(edge,mode))+'</dd></div>',verification=isKnowledgeForge?'<div><dt>验证状态</dt><dd>已验证</dd></div>':'';
    const panel=q('[data-graph-details]');
    panel.innerHTML='<h3>关系详情</h3><div class="graph-relation-card">'+miniNode(source)+'<span class="graph-mini-relation"><b>'+html(short(relationName(edge),24))+'</b></span>'+miniNode(target)+'</div><dl><div><dt>关系说明</dt><dd>'+html(edge.description||relationName(edge))+'</dd></div>'+verification+metric+'<div><dt>'+html(evidenceHeading)+'</dt><dd><div class="graph-evidence-quote">'+html(evidence(edge))+'</div>'+sourceCardMarkup(edge,{showEvidence:false})+'</dd></div></dl>';
    bindEvidenceActions(panel);
  }
  function renderGroups(){
    const groups=state.overview?.groups||[],box=q('[data-graph-groups]');
    if((state.view||{}).mode!=='overview'){box.replaceChildren();box.hidden=true;return}
    box.hidden=false;
    const relationshipCount=group=>Number(group.relationship_count||group.cross_relationship_count||0);
    const ranked=groups.filter(group=>relationshipCount(group)>0).sort((left,right)=>relationshipCount(right)-relationshipCount(left)||groupName(left).localeCompare(groupName(right),'zh-Hans-CN')).slice(0,48);
    box.innerHTML='<div class="graph-directory-heading"><b>社区目录</b><small>按关系数排序 · 共 '+groups.length+' 个社区</small></div>'+ranked.map(group=>'<button type="button" class="embedded-graph-group-card'+(String(group.key)===state.community?' is-active':'')+'" data-common-group="'+html(group.key)+'"><i style="background:'+colorFor(group.key)+'"></i><b>'+html(groupNavigationName(group))+'</b><small>'+html(group.size??0)+' 个实体 · '+html(group.relationship_count??0)+' 条关系</small><span class="graph-community-card-summary">'+html(short(group.summary||'',110))+'</span>'+reportPreviewMarkup(group)+'</button>').join('')+(groups.length>ranked.length?'<p class="graph-directory-note">目录先显示 '+ranked.length+' 个存在关系的社区；其余可通过上方社区选择器访问。</p>':'');
    box.querySelectorAll('[data-common-group]').forEach(button=>button.addEventListener('click',()=>openCommunity(button.dataset.commonGroup)));
  }
  function updateZoomValue(percent){q('[data-graph-zoom-value]').value=Math.max(1,Math.round(percent||100))+'%';q('[data-graph-zoom-value]').textContent=Math.max(1,Math.round(percent||100))+'%'}
  function syncOverviewScope(scope){const control=q('[data-graph-overview-scope]'),visible=(state.view||{}).mode==='overview';control.hidden=!visible;q('[data-graph-topic-key]').hidden=!(visible&&Number((state.view||{}).visible_topic_associations)>0);control.querySelectorAll('[data-overview-relation-scope]').forEach(button=>{const active=button.dataset.overviewRelationScope===scope;button.classList.toggle('is-active',active);button.setAttribute('aria-pressed',String(active));});}
  function syncCommunityScope(level){const control=q('[data-graph-community-scope]'),visible=(state.view||{}).mode==='community';control.hidden=!visible;control.querySelectorAll('[data-community-stage]').forEach(button=>{const active=button.dataset.communityStage===level;button.classList.toggle('is-active',active);button.setAttribute('aria-pressed',String(active));});}
  function graphSelectionActive(){return Number(state.renderer?.selectedNode)>=0||Number(state.renderer?.selectedEdge)>=0}
  function mergeCommunityReportState(payload){
    const byKey=new Map((payload.reports||[]).map(report=>[String(report.community_key||''),report]));
    const mergeGroup=group=>{const incoming=byKey.get(String(group?.key||''));if(!incoming)return group;const previous=group.community_report||{};return {...group,report_status:incoming.status,community_report:{...previous,...incoming,sources:Array.isArray(incoming.sources)?incoming.sources:(previous.sources||[])}}};
    if(state.overview?.groups)state.overview={...state.overview,community_report_status:payload.status,groups:state.overview.groups.map(mergeGroup)};
    if(state.view){
      const focus=state.view.focus?mergeGroup(state.view.focus):state.view.focus;
      const groups=(state.view.groups||[]).map(mergeGroup);
      const nodes=(state.view.nodes||[]).map(node=>{
        const incoming=byKey.get(String(node.community_key||''));
        return incoming?{...node,report_status:incoming.status,community_report:{...(node.community_report||{}),...incoming}}:node;
      });
      state.view={...state.view,community_report_status:payload.status,focus,groups,nodes};
    }
    state.reportStatus=payload.status||state.reportStatus;
    renderGroups();if(!graphSelectionActive())renderOverviewDetails();
  }
  function scheduleReportPolling(delay=3500){
    if(state.reportPollTimer){clearTimeout(state.reportPollTimer);state.reportPollTimer=0}
    if(!isKnowledgeForge||['completed','completed_empty','partial','failed','cancelled','superseded','disabled'].includes(state.reportStatus))return;
    state.reportPollTimer=window.setTimeout(refreshCommunityReports,document.hidden?9000:delay);
  }
  async function refreshCommunityReports(){
    state.reportPollTimer=0;
    try{
      const response=await fetch(reportEndpoint),payload=await response.json();
      if(response.ok)mergeCommunityReportState(payload);
    }catch(_error){}
    scheduleReportPolling();
  }
  async function renderView(){
    const payload=displayPayload(state.view||{});
    state.evidenceAssets.clear();state.evidenceAssetSequence=0;
    if(!state.renderer){
      if(!window.KFGraphUI?.GraphExplorer)throw Error('图谱渲染组件未加载');
      state.renderer=new window.KFGraphUI.GraphExplorer(graphContainer,{
        onNodeSelect:node=>renderNodeDetails(node),onEdgeSelect:edge=>renderEdgeDetails(edge),onClear:renderOverviewDetails,
        onZoomChange:updateZoomValue,
      });
      // Exposed only to the workspace page so browser-level regression checks
      // can exercise the same renderer instance the user sees.
      window.__kfGraphRenderer=state.renderer;
    }
    await state.renderer.setData(payload);
    state.reportStatus=payload.community_report_status||state.reportStatus;
    renderAudit(payload.edges||[]);renderOverviewDetails();renderGroups();syncOverviewScope(payload.overview_relation_scope||state.overviewRelationScope);syncCommunityScope(payload.community_level||state.communityLevel);
    const visibleNodes=payload.visible_entities??(payload.nodes||[]).length,visibleEdges=payload.visible_relationships??(payload.edges||[]).length;
    const countText=payload.mode==='overview'
      ?'全局社区地图 · '+(payload.visible_communities??visibleNodes)+' 个社区 · '+(payload.visible_cross_community_relationships??visibleEdges)+' / '+(payload.total_cross_community_relationships??visibleEdges)+' 条跨社区关系'
      :payload.mode==='community'
        ?'社区图谱 · '+visibleNodes+' 个实体 · '+visibleEdges+' 条关系'
        :payload.mode==='neighbours'
          ?'实体邻域 · '+visibleNodes+' 个实体 · '+visibleEdges+' 条关系'
          :'筛选结果 · '+visibleNodes+' 个实体 · '+visibleEdges+' 条关系';
    q('[data-graph-count]').textContent=countText;
    scheduleReportPolling();
  }
  async function openCommunity(key,level='full'){state.community=String(key||'');state.communityLevel=level;q('[data-graph-group-filter]').value=state.community;if(!state.community)return requestView({mode:'overview',overview_relation_scope:state.overviewRelationScope,limit:1200});if(isKnowledgeForge)fetch(reportEndpoint+'/'+encodeURIComponent(state.community)+'/prioritize',{method:'POST'}).then(async response=>{const payload=await response.json();if(response.ok&&payload.status)state.reportStatus=payload.status;scheduleReportPolling(600)}).catch(()=>{});return requestView({mode:'community',community:state.community,community_level:state.communityLevel,limit:2000})}
  function resetToOverview(){state.community='';state.communityLevel='full';state.overviewRelationScope='all_communities';state.query='';state.mode='overview';q('[data-graph-filter]').value='';q('[data-graph-group-filter]').value='';requestView({mode:'overview',overview_relation_scope:state.overviewRelationScope,limit:1200})}
  function setupGraphPaneResizer(){
    const layout=q('.embedded-graph-layout'),separator=q('[data-graph-pane-resizer]');if(!layout||!separator)return;
    const storageKey='knowledge-forge.graph.sidebar-width';
    const bounds=()=>({min:260,max:Math.max(260,Math.min(720,layout.getBoundingClientRect().width-480))});
    const setWidth=value=>{const range=bounds(),width=Math.round(Math.max(range.min,Math.min(range.max,Number(value)||320)));layout.style.setProperty('--graph-sidebar-width',width+'px');separator.setAttribute('aria-valuenow',String(width));separator.setAttribute('aria-valuemax',String(Math.round(range.max)));return width};
    try{setWidth(Number(localStorage.getItem(storageKey))||320)}catch(_error){setWidth(320)}
    let drag=null,resizeFrame=0;
    const notifyResize=()=>{if(resizeFrame)return;resizeFrame=requestAnimationFrame(()=>{resizeFrame=0;window.dispatchEvent(new Event('resize'))})};
    separator.addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();event.stopPropagation();drag={pointerId:event.pointerId,startX:event.clientX,startWidth:parseFloat(getComputedStyle(layout).getPropertyValue('--graph-sidebar-width'))||320};separator.setPointerCapture(event.pointerId);separator.classList.add('is-dragging');document.body.classList.add('is-resizing-graph-pane')});
    separator.addEventListener('pointermove',event=>{if(!drag||event.pointerId!==drag.pointerId)return;setWidth(drag.startWidth+drag.startX-event.clientX);notifyResize()});
    const finish=event=>{if(!drag||event.pointerId!==drag.pointerId)return;const width=setWidth(parseFloat(getComputedStyle(layout).getPropertyValue('--graph-sidebar-width')));try{localStorage.setItem(storageKey,String(width))}catch(_error){}if(separator.hasPointerCapture(event.pointerId))separator.releasePointerCapture(event.pointerId);drag=null;separator.classList.remove('is-dragging');document.body.classList.remove('is-resizing-graph-pane');notifyResize()};
    separator.addEventListener('pointerup',finish);separator.addEventListener('pointercancel',finish);
    separator.addEventListener('dblclick',event=>{event.preventDefault();const width=setWidth(320);try{localStorage.setItem(storageKey,String(width))}catch(_error){}notifyResize()});
    separator.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home'].includes(event.key))return;event.preventDefault();const current=parseFloat(getComputedStyle(layout).getPropertyValue('--graph-sidebar-width'))||320;const width=setWidth(event.key==='Home'?320:current+(event.key==='ArrowLeft'?20:-20));try{localStorage.setItem(storageKey,String(width))}catch(_error){}notifyResize()});
    window.addEventListener('resize',()=>setWidth(parseFloat(getComputedStyle(layout).getPropertyValue('--graph-sidebar-width'))||320));
  }
  function restoreHistory(index){
    if(index<0||index>=state.history.length||state.loading)return;
    state.historyIndex=index;
    const route=state.history[index];
    state.community=String(route.community||'');state.communityLevel=route.community_level||'full';state.overviewRelationScope=route.overview_relation_scope||'all_communities';state.query=route.q||'';state.mode=route.mode||'overview';
    q('[data-graph-filter]').value=state.query;q('[data-graph-group-filter]').value=state.community;
    requestView(route,{history:'restore'});
  }
  function setTab(name){q('[data-graph-panel="network"]').hidden=name!=='network';q('[data-graph-panel="audit"]').hidden=name!=='audit';root.querySelectorAll('[data-graph-tab]').forEach(button=>{const active=button.dataset.graphTab===name;button.classList.toggle('is-active',active);button.setAttribute('aria-selected',String(active));button.tabIndex=active?0:-1})}
  root.querySelectorAll('[data-graph-tab]').forEach(button=>button.addEventListener('click',()=>setTab(button.dataset.graphTab)));
  q('[data-apply-graph-filter]').addEventListener('click',()=>{const query=q('[data-graph-filter]').value.trim();state.query=query;if(query)requestView({mode:'search',q:query,limit:800});else if(state.community)openCommunity(state.community);else requestView({mode:'overview',overview_relation_scope:state.overviewRelationScope,limit:1200})});
  q('[data-graph-filter]').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();q('[data-apply-graph-filter]').click()}});
  q('[data-graph-group-filter]').addEventListener('change',event=>openCommunity(event.target.value));
  q('[data-graph-overview-scope]').querySelectorAll('[data-overview-relation-scope]').forEach(button=>button.addEventListener('click',()=>{const scope=button.dataset.overviewRelationScope;if(scope===state.overviewRelationScope||state.loading)return;state.overviewRelationScope=scope;requestView({mode:'overview',overview_relation_scope:scope,limit:1200})}));
  q('[data-graph-community-scope]').querySelectorAll('[data-community-stage]').forEach(button=>button.addEventListener('click',()=>{const level=button.dataset.communityStage;if(level===state.communityLevel||state.loading)return;openCommunity(state.community,level)}));
  q('[data-graph-zoom-out]').addEventListener('click',()=>state.renderer?.zoomBy(.72));
  q('[data-graph-zoom-in]').addEventListener('click',()=>state.renderer?.zoomBy(1.4));
  q('[data-graph-fit]').addEventListener('click',()=>state.renderer?.fit());
  q('[data-reset-graph]').addEventListener('click',resetToOverview);
  q('[data-graph-history-back]').addEventListener('click',()=>restoreHistory(state.historyIndex-1));
  q('[data-graph-history-forward]').addEventListener('click',()=>restoreHistory(state.historyIndex+1));
  q('[data-graph-return-overview]').addEventListener('click',resetToOverview);
  q('[data-graph-evidence-modal-close]').addEventListener('click',()=>q('[data-graph-evidence-modal]').close());
  q('[data-graph-evidence-modal]').addEventListener('click',event=>{if(event.target===event.currentTarget)event.currentTarget.close()});
  document.addEventListener('visibilitychange',()=>scheduleReportPolling(document.hidden?9000:500));
  setupGraphPaneResizer();
  requestView({mode:'overview',overview_relation_scope:state.overviewRelationScope,limit:1200}).then(()=>{
    const select=q('[data-graph-group-filter]'),groups=state.overview?.groups||[];
    select.replaceChildren(new Option('全部社区',''),...groups.map(group=>new Option(groupNavigationName(group),String(group.key))));
  });
}
}
"""


if __name__ == "__main__":
    serve()
