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
from .graph_policy import GraphNotReadyError, is_current_graph
from .service import KnowledgeForgeService


LOGGER = logging.getLogger(__name__)
SERVICE = KnowledgeForgeService()


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


STATUS_NAMES = {"uploaded": "已上传", "processing": "处理中", "indexed": "已入库", "failed": "失败", "deleted": "已删除"}
STAGE_NAMES = {
    "uploaded": "等待处理", "structure": "结构解析中", "structure_ready": "结构解析完成",
    "chunking": "知识分块中", "chunk_ready": "知识分块完成", "graph": "知识图谱构建中",
    "completed": "全部完成", "failed": "处理失败",
}


def render_page(project_id: int | None, message: tuple[str, str] | None = None) -> str:
    projects, models = SERVICE.db.list_projects(), SERVICE.db.list_models()
    current = next((item for item in projects if item["id"] == project_id), None) or (projects[0] if projects else None)
    documents = SERVICE.db.list_documents(int(current["id"])) if current else []
    statistics = SERVICE.db.statistics(int(current["id"])) if current else {"documents": 0, "chunks": 0, "entities": 0, "relations": 0}
    schema = SERVICE.db.get_active_schema(int(current["id"])) if current else None
    latest_job = SERVICE.latest_job(int(current["id"]), "ingestion") if current else None
    active_job = SERVICE.latest_active_job(int(current["id"])) if current else None
    latest_job_id = str(latest_job["id"]) if latest_job else ""
    active_job_id = str(active_job["id"]) if active_job else ""
    active_job_kind = str(active_job["kind"]) if active_job_id else ""
    schema_summary = "尚未初始化"
    if schema:
        data = schema["schema_json"]
        schema_summary = f"v{schema['version']} · {len(data.get('entity_types',[]))} 类实体 · {len(data.get('relation_types',[]))} 类关系"
    options = "".join(f'<option value="{item["id"]}"{" selected" if current and item["id"] == current["id"] else ""}>{_esc(item["name"])}</option>' for item in projects) or '<option value="">暂无项目</option>'
    rows = "".join(
        f"""<tr><td><strong>{_esc(item['original_filename'])}</strong></td>
        <td>{item['file_size_bytes']/1024:.1f} KB</td><td><span class="status status-{_esc(item['status'])}">{STATUS_NAMES.get(item['status'],_esc(item['status']))}</span>
        <small data-stage-label="{item['id']}">{_esc(STAGE_NAMES.get(item.get('processing_stage'), item.get('processing_stage') or '等待处理'))}</small>
        {f'<small>{_esc(item.get("effective_parser"))} · {_esc(item.get("effective_chunk_strategy"))}</small>' if item.get('effective_parser') else ''}
        {f'<small class="failure">{_esc(item.get("error_message"))}</small>' if item.get('error_message') else ''}</td><td>{_time(item['updated_at'])}</td>
        <td class="actions"><button class="button small secondary" type="button" data-view-structure="{item['id']}"{' disabled' if item.get('status') in ('uploaded','deleted') or not item.get('structure_ready_at') else ''}>结构块</button>
        <button class="button small secondary" type="button" data-view-chunks="{item['id']}"{' disabled' if item.get('status') in ('uploaded','deleted') or not item.get('chunks_ready_at') else ''}>Chunk 分片</button>
        <a class="button small secondary" href="/api/documents/{item['id']}/download">下载</a><button class="icon-button danger" title="删除文档" data-delete="{item['id']}">×</button></td></tr>"""
        for item in documents
    ) or '<tr><td colspan="5" class="empty">该项目尚未上传文档。</td></tr>'
    current_block = ""
    settings_button = ""
    settings_dialog = ""
    if current:
        graph_enabled = is_current_graph(current) and not active_job_id
        graph_status = str(current.get("graph_status") or "empty")
        graph_retry = "" if graph_enabled or active_job_id else (
            f'<button class="button secondary" type="button" data-rebuild-graph="{_esc(current["id"])}">重建知识图谱</button>'
        )
        current_block = f"""
        <div class="workspace-section graph-summary"><div class="stats"><div><b>{statistics['documents']}</b><span>文档</span></div><div><b>{statistics['chunks']}</b><span>知识块</span></div>
          <div><b>{statistics['entities']}</b><span>实体</span></div><div><b>{statistics['relations']}</b><span>关系</span></div></div>
        <div class="stats-actions"><span class="schema-chip">{_esc(schema_summary)}</span>
          <span class="schema-chip">图谱：{_esc(graph_status)}</span>{graph_retry}
          <button class="button secondary" type="button" data-reassess-schema="{_esc(current['id'])}"{' disabled' if active_job_id else ''}>重新评估 Schema</button>
          <a id="view-knowledge-graph" class="button secondary{' disabled-link' if not graph_enabled else ''}" href="/graph/view?project={_esc(current['id'])}" target="_blank" rel="noopener noreferrer" aria-disabled="{'false' if graph_enabled else 'true'}">查看知识图谱</a></div></div>"""
        settings_button = '<button class="button secondary" type="button" id="open-project-settings">项目设置</button>'
        settings_dialog = f"""
        <dialog id="project-settings-dialog"><div class="dialog-heading"><div><h2>项目设置</h2><p>设置当前项目使用的模型、项目上下文预算和单切片上限；不会修改 Ollama 全局上下文。</p></div>
          <div class="dialog-title-actions"><button id="sync-models" class="icon-button refresh-button" type="button"
            title="从 Ollama 同步可用模型" aria-label="从 Ollama 同步可用模型">↻</button>
            <button class="icon-button" type="button" data-close-project-settings aria-label="关闭">×</button></div></div>
        <p id="model-sync-status" class="model-sync-status" role="status" hidden></p>
        <form id="project-settings-form" method="post" action="/projects/{current['id']}/models" class="form-grid compact">
          <label>LLM<select name="llm_model">{_model_options(models,'llm',current.get('llm_model') or '')}</select></label>
          <label>Embedding<select name="embedding_model">{_model_options(models,'embedding',current.get('embedding_model') or '')}</select></label>
          <label>项目上下文 Token 上限
            <input name="model_max_tokens" type="number" min="1024" step="1" required value="{int(current.get('model_max_tokens') or 32768)}">
            <small id="model-token-limit-hint" class="field-help">正在读取当前 LLM 的上下文上限...</small></label>
          <label>单切片 Token 上限
            <input name="chunk_max_tokens" type="number" min="64" step="1" required value="{int(current.get('chunk_max_tokens') or 512)}">
            <small id="chunk-token-limit-hint" class="field-help">正在读取 LLM 与 Embedding 的共同上限...</small></label>
          <div class="form-actions dialog-actions"><button class="button secondary" type="button" data-close-project-settings>取消</button>
            <button class="button primary" type="submit">保存项目设置</button></div></form></dialog>"""
    job_hidden = "" if active_job_id else " hidden"
    notice = _notice(*message) if message else ""
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Knowledge Forge</title><style>{CSS}</style></head><body><main class="shell">
    <header><div><h1>Knowledge Forge</h1><p>通用知识库与知识图谱构建平台</p></div></header>{notice}
    <section class="panel project-panel"><div class="section-heading project-heading"><h2>项目空间</h2>
      <div class="project-picker"><form method="get"><select name="project" aria-label="选择当前项目" onchange="this.form.submit()">{options}</select></form>
        {settings_button}<button class="button primary" type="button" id="open-create-project">新建项目</button></div></div>
      <div class="workspace-section" {'' if current else 'hidden'}><div class="section-heading"><div><h2>知识库构建</h2><p>上传后执行章节切分、向量索引、Schema 评估、实体归一与图谱写入。</p></div>
      <div class="section-actions"><button class="button secondary" id="view-ingestion-log" data-view-job-log data-job="{_esc(latest_job_id)}"{' disabled' if not latest_job_id else ''}>查看知识构建日志</button></div></div>
      <form action="/upload" method="post" enctype="multipart/form-data" class="upload-form processing-form"><input type="hidden" name="project_id" value="{_esc(current['id'] if current else '')}">
        <div class="upload-line"><input id="upload-files" type="file" name="files" multiple accept=".docx,.pdf,.md,.markdown,.txt,.xlsx,.xls,.csv" required><button class="button primary" type="submit">上传解析</button></div>
        <p class="upload-help">支持多文件上传，单个文件最大 {Config.upload_max_bytes // (1024 * 1024)} MB。</p>
        <details class="processing-options"><summary>解析与分块设置</summary>
          <div id="processing-selects" class="form-grid compact">
            <label>解析器<select name="requested_parser">{_named_options(PARSER_OPTIONS,'auto')}</select></label>
            <label>分块策略<select name="requested_strategy">{_named_options(CHUNK_STRATEGY_OPTIONS,'auto')}</select></label>
          </div>
          <p id="processing-hint" class="muted">选择文件后会检查解析器和分块策略的兼容性。</p>
        </details>
      </form>
      <div id="task-box" class="task-box"{job_hidden} data-job="{_esc(active_job_id)}" data-kind="{_esc(active_job_kind)}">
        <div class="task-head"><b id="task-title">后台任务</b><span id="task-status">等待</span></div>
        <div id="task-steps" class="task-steps">
          <div class="task-step" data-subtask="structure"><div class="task-step-head"><span>结构解析</span><b data-subtask-status>等待</b></div><div class="progress"><i data-subtask-bar></i></div></div>
          <div class="task-step" data-subtask="chunking"><div class="task-step-head"><span>分块及向量索引</span><b data-subtask-status>等待</b></div><div class="progress"><i data-subtask-bar></i></div></div>
          <div class="task-step" data-subtask="schema_base"><div class="task-step-head"><span>Schema 和基础图数据</span><b data-subtask-status>等待</b></div><div class="progress"><i data-subtask-bar></i></div></div>
          <div class="task-step" data-subtask="extraction"><div class="task-step-head"><span>逐块抽取实体与关系</span><b data-subtask-status>等待</b></div><div class="progress"><i data-subtask-bar></i></div></div>
          <div class="task-step" data-subtask="graph_sync"><div class="task-step-head"><span>HugeGraph 同步</span><b data-subtask-status>等待</b></div><div class="progress"><i data-subtask-bar></i></div></div>
        </div>
        <pre id="task-log"></pre>
      </div>
      </div>
      {current_block}
      <div class="workspace-section" {'' if current else 'hidden'}><div class="section-heading"><div><h2>联合检索</h2><p>融合本地向量召回、关键词匹配、规范实体和关系断言。</p></div></div>
      <form id="search-form" class="search-form"><input name="query" placeholder="输入概念、实体、事件或问题" required><button class="button primary">检索</button></form><div id="search-results"></div>
      </div>
      <div class="workspace-section" {'' if current else 'hidden'}><div class="section-heading"><div><h2>项目文档</h2><p>源文件、处理状态和来源链接</p></div></div>
        <div class="table-wrap"><table><thead><tr><th>文件</th><th>大小</th><th>状态</th><th>更新时间</th><th class="actions">操作</th></tr></thead><tbody>{rows}</tbody></table></div>
      </div>
    </section>
    <dialog id="create-project-dialog"><div class="dialog-heading"><div><h2>新建项目</h2><p>创建独立的文档、向量与知识图谱空间。</p></div>
      <button class="icon-button" type="button" data-close-create-project aria-label="关闭">×</button></div>
      <form action="/projects" method="post" class="form-grid">
        <label>项目名称<input name="name" required maxlength="100" placeholder="例如：产业链知识图谱"></label>
        <label>LLM<select name="llm_model">{_model_options(models,'llm')}</select></label>
        <label>Embedding<select name="embedding_model">{_model_options(models,'embedding')}</select></label>
        <label class="wide">项目说明<textarea name="description" rows="2"></textarea></label>
        <div class="form-actions dialog-actions"><button class="button secondary" type="button" data-close-create-project>取消</button>
          <button class="button primary" type="submit">创建项目</button></div>
      </form>
    </dialog>
    {settings_dialog}
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
    <dialog id="structure-dialog" class="structure-dialog"><div class="dialog-heading"><div><h2>文档结构块</h2><p id="structure-summary">正在读取解析结果...</p></div>
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
    </main><script>window.KF={{projectId:{int(current['id']) if current else 0},docling:{str(docling_available()).lower()}}};{SCRIPT}</script></body></html>"""


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
            project_id = self._project(); params = parse_qs(parsed.query); text = params.get("message", [""])[0]
            self._respond(200, render_page(project_id, (params.get("kind", [""])[0], text) if text else None),
                          cookie=f"knowledge_forge_project={project_id}; Path=/; SameSite=Lax" if project_id else ""); return
        if parsed.path.startswith("/api/jobs/"):
            job = SERVICE.get_job(parsed.path.rsplit("/", 1)[-1]); self._json(200 if job else 404, job or {"error": "任务不存在"}); return
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
            try: self._json(200, SERVICE.search(int(params.get("project", ["0"])[0]), params.get("q", [""])[0]))
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
                project_id = int(parse_qs(parsed.query).get("project", ["0"])[0])
                self._redirect(SERVICE.hubble_project_url(project_id), project_id)
            except GraphNotReadyError as exc:
                self._respond(HTTPStatus.CONFLICT, str(exc), "text/plain; charset=utf-8")
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
                data = self._form(); project = SERVICE.create_project(data.get("name", ""), data.get("description", ""), data.get("llm_model", ""), data.get("embedding_model", ""))
                self._redirect(f"/?kind=ok&message={quote('项目已创建，基础 Schema 已初始化')}", int(project["id"])); return
            match = re.fullmatch(r"/projects/(\d+)/models", parsed.path)
            if match:
                data = self._form(); project = SERVICE.update_project_settings(
                    int(match.group(1)), data.get("llm_model", ""), data.get("embedding_model", ""),
                    int(data.get("model_max_tokens", "0")), int(data.get("chunk_max_tokens", "0")))
                self._redirect(f"/?kind=ok&message={quote('项目设置已更新')}", int(project["id"])); return
            if parsed.path == "/upload": self._upload(); return
            if parsed.path == "/api/models/sync":
                models = SERVICE.sync_ollama_models(); self._json(200, {"ok": True, "count": len(models), "models": models}); return
            match = re.fullmatch(r"/api/documents/(\d+)/ingest", parsed.path)
            if match: self._json(202, {"ok": True, "job_id": SERVICE.submit_ingestion(int(match.group(1)))}); return
            match = re.fullmatch(r"/api/documents/(\d+)/delete", parsed.path)
            if match:
                job_id = SERVICE.delete_document(int(match.group(1)))
                self._json(202, {"ok": True, "job_id": job_id})
                return
            match = re.fullmatch(r"/api/projects/(\d+)/graph", parsed.path)
            if match: self._json(202, {"ok": True, "job_id": SERVICE.submit_graph_build(int(match.group(1)))}); return
            match = re.fullmatch(r"/api/projects/(\d+)/schema/reassess", parsed.path)
            if match: self._json(202, {"ok": True, "job_id": SERVICE.submit_schema_reassessment(int(match.group(1)))}); return
            self._json(404, {"error": "接口不存在"})
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
        for document in uploaded:
            SERVICE.submit_ingestion(int(document["id"]))
        names = "、".join(str(document["original_filename"]) for document in uploaded)
        self._redirect(f"/?kind=ok&message={quote('已上传并提交解析：'+names)}", project_id)

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
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
:root{font-family:Arial,"Microsoft YaHei",sans-serif;color:#13243b;background:#f3f7fb}*{box-sizing:border-box}body{margin:0}.shell{max-width:1180px;margin:30px auto;padding:0 20px 40px}header,.section-heading,.project-picker,.header-actions,.section-actions,.task-head,.dialog-heading{display:flex;align-items:center;gap:12px}header,.section-heading,.task-head,.dialog-heading{justify-content:space-between}header{margin-bottom:18px}h1,h2{margin:0}h1{font-size:27px}h2{font-size:19px}p{color:#5d7088;margin:7px 0 0;line-height:1.5}.panel{background:#fff;border:1px solid #d5e0ed;border-radius:8px;padding:22px;margin-bottom:16px;box-shadow:0 10px 24px rgba(26,61,101,.05)}.project-panel{background:#fff}.project-heading{margin:20px 0 4px}.project-heading h2{font-size:23px}.workspace-section{border-top:1px solid #dce5ee;margin-top:22px;padding-top:22px}.button,.icon-button{border:0;background:#30445f;color:#fff;padding:9px 14px;border-radius:6px;font-size:14px;cursor:pointer;text-decoration:none;white-space:nowrap}.button:hover{background:#20324c}.button.primary{background:#2677d8}.button.secondary{background:#fff;color:#263f60;border:1px solid #c4d2e1}.button.small{padding:7px 10px}.button:disabled{cursor:not-allowed;opacity:1}.button.secondary:disabled{background:#eef2f6;color:#98a5b5;border-color:#d8e0e9}.icon-button{width:34px;height:34px;padding:0;font-size:22px}.danger{background:#df3038}.header-actions{justify-content:flex-end}label{color:#354963;font-size:14px;display:flex;flex-direction:column;gap:7px}input,select,textarea{width:100%;border:1px solid #bfd0e2;border-radius:6px;padding:9px 10px;color:#172b47;background:#fff;font:inherit}select{min-width:240px}.project-picker{margin:0}.project-picker form{margin:0}.project-picker select{width:min(430px,100%)}.project-picker select,.project-picker .button{font-size:16px;min-height:42px}.project-meta{display:flex;align-items:center;flex-wrap:wrap;gap:20px;color:#58708c;font-size:13px;padding:8px 0}.project-meta b{color:#263f60}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}.stats div{border-left:3px solid #2f7bd4;background:#f8fbff;padding:10px 14px}.stats b{display:block;font-size:20px}.stats span{font-size:12px;color:#6d7f93}.stats-actions{display:flex;align-items:center;justify-content:flex-end;gap:8px;margin-top:12px}summary{color:#1769c2;font-weight:700;cursor:pointer}.form-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px;margin-top:14px;max-width:850px}.wide,.form-actions{grid-column:1/-1}.compact{max-width:700px}.settings-toolbar{display:flex;justify-content:flex-end;margin-top:14px}.refresh-button{background:#fff;color:#263f60;border:1px solid #c4d2e1;font-size:21px;line-height:1}.refresh-button:hover{background:#edf4fb}.upload-form,.search-form{display:flex;gap:10px;align-items:center;margin-top:17px}.upload-form input,.search-form input{flex:1}.task-box{margin-top:18px;background:#101c2e;color:#d9e8fb;padding:14px;border-radius:6px}.task-box pre{margin:10px 0 0;max-height:260px;overflow:auto;white-space:pre-wrap;font:13px/1.5 monospace}.progress{height:5px;background:#263951;margin-top:10px}.progress i{display:block;height:100%;width:0;background:#4fa3ff;transition:width .25s}.schema-chip{display:inline-flex;align-items:center;color:#315a83;background:#eaf3fc;padding:6px 9px;border-radius:4px;font-size:13px;line-height:1}.notice{position:fixed;top:18px;right:18px;z-index:1100;max-width:min(430px,calc(100vw - 36px));padding:12px 14px;border-radius:6px;display:flex;gap:9px;align-items:center;box-shadow:0 12px 30px rgba(20,42,70,.2);transition:opacity .25s,transform .25s}.notice.toast-hide{opacity:0;transform:translateY(-8px);pointer-events:none}.notice b{display:grid;place-items:center;flex:0 0 20px;width:20px;height:20px;border-radius:50%;color:#fff}.notice.success{background:#effaf2;color:#176a35;border:1px solid #9fdbaf}.notice.success b{background:#23a453}.notice.error{background:#fff3f3;color:#9c262d;border:1px solid #f0b1b4}.notice.error b{background:#d4383f}dialog{width:min(720px,calc(100% - 28px));border:1px solid #cbd8e6;border-radius:8px;padding:22px;color:#13243b;box-shadow:0 22px 60px rgba(15,35,60,.24)}dialog::backdrop{background:rgba(20,36,58,.46)}dialog .form-grid{max-width:none}.dialog-actions{display:flex;justify-content:flex-end;gap:10px}.log-dialog{width:min(900px,calc(100% - 28px))}.history-log{min-height:220px;max-height:60vh;overflow:auto;padding:16px;border-radius:6px;background:#101c2e;color:#d9e8fb;white-space:pre-wrap;font:13px/1.55 monospace}.job-log-error{margin:14px 0;padding:10px 12px;border:1px solid #f0b1b4;border-radius:6px;background:#fff3f3;color:#9c262d}.table-wrap{overflow:auto;border:1px solid #d6e1ec;border-radius:6px;margin-top:16px}table{width:100%;border-collapse:collapse;min-width:760px}th,td{padding:12px;text-align:left;border-bottom:1px solid #e1e9f1;vertical-align:middle}th{background:#f6f9fc;color:#39516e;font-size:14px}td small{display:block;color:#6b7e93;margin-top:5px;max-width:390px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.actions{position:sticky;right:0;background:#fff;white-space:nowrap;text-align:center;min-width:270px}.actions .button,.actions .icon-button{display:inline-flex;align-items:center;justify-content:center;vertical-align:middle;height:34px}.actions>*+*{margin-left:6px}th.actions{background:#f6f9fc;text-align:center}.status{display:inline-block;padding:4px 8px;border-radius:14px;background:#e8edf3;color:#42566f;font-size:13px}.status-indexed{background:#e8f7ec;color:#20773b}.status-failed{background:#fff0f0;color:#b72d35}.status-processing{background:#fff7df;color:#926a00}.failure{color:#b72d35!important}.empty{text-align:center;color:#74869b;padding:30px}.result-group{margin-top:18px}.result-card{border-top:1px solid #dce5ee;padding:12px 2px}.result-card h3{font-size:15px;margin:0 0 7px}.result-card p{font-size:14px;color:#354963;white-space:pre-wrap}.score{font-size:12px;color:#1b6bc1}.relation{font-weight:700;color:#315a83}.muted{color:#718398;font-size:13px}
.project-heading{margin:0 0 18px}.project-heading h2{font-size:24px}.project-heading .project-picker select,.project-heading .project-picker .button{font-size:17px}
.stats-actions{justify-content:flex-start}.stats-actions .schema-chip{margin-right:auto;padding:9px 12px;font-size:16px;font-weight:700}
.dialog-title-actions{display:flex;align-items:center;gap:8px}.model-sync-status{margin:14px 0 0;font-size:14px}.model-sync-status.success{color:#20773b}.model-sync-status.failure{color:#b72d35}.field-help{display:block;min-height:18px;color:#6b7e93;font-size:12px;line-height:1.5}.field-help.failure{color:#b72d35}
.delete-progress-dialog{width:min(520px,calc(100% - 28px))}.delete-progress-body{display:flex;align-items:center;gap:16px;margin:22px 0 16px;padding:16px;border:1px solid #d6e1ec;border-radius:6px;background:#f8fbff}.delete-progress-body p{margin-top:4px}.delete-spinner{flex:0 0 30px;width:30px;height:30px;border:3px solid #cfdae7;border-top-color:#2677d8;border-radius:50%;animation:delete-spin .8s linear infinite}.delete-result{display:grid;place-items:center;flex:0 0 30px;width:30px;height:30px;border-radius:50%;color:#fff;font-size:20px;font-weight:700}.delete-result.success{background:#23a453}.delete-result.failure{background:#d4383f}.delete-progress-bar{margin:0 0 18px}.delete-progress-bar i{background:#2677d8}.deleting-row{opacity:.48;pointer-events:none}@keyframes delete-spin{to{transform:rotate(360deg)}}
.processing-form{display:flex;flex-direction:column;align-items:stretch}.upload-line{display:flex;align-items:center;gap:10px}.upload-line input{flex:1}.upload-help{margin:7px 0 0;color:#6b7e93;font-size:13px}.processing-options{border:1px solid #d7e2ed;border-radius:6px;padding:10px 12px;background:#f8fbff}.processing-options summary{font-size:14px}.processing-options .form-grid{margin-top:12px;max-width:none}.processing-options .muted{margin-top:10px}
.task-steps{display:grid;gap:10px;margin-top:12px}.task-step{display:grid;grid-template-columns:210px minmax(180px,1fr);align-items:center;gap:14px}.task-step-head{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:13px}.task-step-head span{color:#d9e8fb}.task-step-head b{color:#8fa5bd;font-weight:400;white-space:nowrap}.task-step .progress{margin:0;height:7px;border-radius:5px;overflow:hidden}.task-step.running .task-step-head b{color:#72b8ff}.task-step.completed .task-step-head b{color:#75d69a}.task-step.completed .progress i{background:#38ae68}.task-step.failed .task-step-head b,.task-step.cancelled .task-step-head b{color:#ff9297}.task-step.failed .progress i,.task-step.cancelled .progress i{background:#df5960}
.disabled-link{opacity:.45;pointer-events:none;cursor:not-allowed}.chunk-badges{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:9px}.chunk-badge{padding:3px 7px;border-radius:4px;background:#eef3f8;color:#496079;font-size:12px}.chunk-badge.graph{background:#e8f7ec;color:#20773b}.chunk-badge.skipped{background:#fff3df;color:#8b6400}
.structure-dialog{width:min(980px,calc(100% - 28px))}.structure-tools{display:grid;grid-template-columns:minmax(0,1fr) 210px;gap:10px;margin:16px 0}.structure-tools select{min-width:0}.structure-list{max-height:58vh;overflow:auto;border:1px solid #d6e1ec;border-radius:6px}.structure-row{padding:14px 16px;border-bottom:1px solid #e1e9f1}.structure-row:last-child{border-bottom:0}.structure-row-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:9px}.structure-row-meta{display:flex;align-items:center;flex-wrap:wrap;gap:7px;color:#637891;font-size:12px}.block-type{display:inline-block;padding:3px 7px;border-radius:4px;background:#eaf3fc;color:#245d94;font-weight:700}.structure-text{margin:0;color:#243a55;white-space:pre-wrap;word-break:break-word;font:14px/1.65 Arial,"Microsoft YaHei",sans-serif}.structure-metadata{margin-top:10px}.structure-metadata summary{font-size:12px;font-weight:400}.structure-metadata pre{margin:8px 0 0;padding:10px;background:#f6f9fc;border-radius:4px;overflow:auto;font-size:12px}.structure-pager{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:14px 0}.structure-pager>div{display:flex;gap:8px}
@media(max-width:760px){.shell{margin:16px auto;padding:0 10px 24px}header,.section-heading,.project-picker,.upload-form,.search-form{align-items:stretch;flex-direction:column}.project-picker label{min-width:0}.form-grid,.structure-tools{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.stats-actions{align-items:stretch;flex-direction:column}.stats-actions .schema-chip{margin-right:0;justify-content:center}.stats-actions .button{width:100%}.panel{padding:16px}.task-step{grid-template-columns:1fr;gap:5px}}
"""

SCRIPT = r"""
const box=document.querySelector('#task-box'),log=document.querySelector('#task-log'),taskStatus=document.querySelector('#task-status'),taskSteps=document.querySelector('#task-steps');
const esc=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const ingestionLogButton=document.querySelector('#view-ingestion-log'),taskTitle=document.querySelector('#task-title');
const stageLabels={structure:'结构解析中',structure_ready:'结构解析完成',chunking:'知识分块中',chunk_ready:'知识分块完成',graph:'知识图谱构建中',completed:'全部完成',failed:'处理失败'};
const subtaskLabels={structure:'结构解析',chunking:'分块及向量索引',schema_base:'Schema 和基础图数据',extraction:'逐块抽取实体与关系',graph_sync:'HugeGraph 同步'};
const taskStatusLabels={queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',cancelled:'已取消'};
const clampProgress=value=>Math.max(0,Math.min(100,Math.round(Number(value)||0)));
function legacySubtasks(job){const p=clampProgress(job.progress),step=(status,progress,detail='')=>({status,progress:clampProgress(progress),detail});const result={structure:step('waiting',0),chunking:step('waiting',0),schema_base:step('waiting',0),extraction:step('waiting',0),graph_sync:step('waiting',0)};if(job.kind==='graph_sync'){result.graph_sync=step(job.status==='completed'?'completed':job.status==='failed'?'failed':'running',p);return result}if(job.kind==='schema_reassessment'){result.schema_base=step(job.status==='completed'?'completed':job.status==='failed'?'failed':'running',p,'重新评估');return result}result.structure=p>=18?step('completed',100):step('running',p/18*100);if(p>=20)result.chunking=p>=42?step('completed',100):step('running',(p-20)/22*100);if(p>=44)result.schema_base=p>=50?step('completed',100):step('running',(p-44)/6*100);if(p>=52)result.extraction=p>=90?step('completed',100):step('running',(p-52)/38*100);if(p>=92)result.graph_sync=p>=96?step('completed',100):step('running',(p-92)/4*100);if(job.status==='completed')for(const key of Object.keys(result))result[key]=step('completed',100,result[key].detail);return result}
function jobSubtasks(job){const stored=job.metadata?.subtasks;if(!stored||typeof stored!=='object')return legacySubtasks(job);return {...legacySubtasks({kind:job.kind,status:'queued',progress:0}),...stored}}
function renderSubtasks(job){const subtasks=jobSubtasks(job);if(taskStatus)taskStatus.textContent=taskStatusLabels[job.status]||job.status||'等待';for(const row of taskSteps?.querySelectorAll('[data-subtask]')||[]){const key=row.dataset.subtask;row.hidden=job.kind==='document_cleanup'||(job.kind==='schema_reassessment'&&key!=='schema_base');const data=subtasks[key]||{status:'waiting',progress:0,detail:''},progress=clampProgress(data.progress),status=data.status||'waiting',label=row.querySelector('[data-subtask-status]'),bar=row.querySelector('[data-subtask-bar]');row.className='task-step '+status;if(label){if(status==='completed')label.textContent='已完成'+(data.detail?' · '+data.detail:'');else if(status==='failed')label.textContent='失败'+(data.detail?' · '+data.detail:'');else if(status==='cancelled')label.textContent='已取消';else if(status==='running')label.textContent=progress+'%'+(data.detail?' · '+data.detail:'');else label.textContent='等待'}if(bar)bar.style.width=progress+'%'}}
function applyJobReadiness(job){const metadata=job.metadata||{},documentId=job.document_id||metadata.document_id;if(documentId){const structure=document.querySelector('[data-view-structure="'+documentId+'"]'),chunks=document.querySelector('[data-view-chunks="'+documentId+'"]'),label=document.querySelector('[data-stage-label="'+documentId+'"]');if(structure&&metadata.structure_ready)structure.disabled=false;if(chunks&&metadata.chunks_ready)chunks.disabled=false;if(label&&metadata.stage)label.textContent=stageLabels[metadata.stage]||metadata.stage}const graph=document.querySelector('#view-knowledge-graph');if(graph&&metadata.graph_ready){graph.classList.remove('disabled-link');graph.setAttribute('aria-disabled','false')}}
function monitor(id,kind=''){if(!id)return;box.hidden=false;box.dataset.job=id;box.dataset.kind=kind;const target=kind==='ingestion'?ingestionLogButton:null;if(target){target.dataset.job=id;target.disabled=false}if(taskTitle)taskTitle.textContent=kind==='graph_sync'?'知识图谱同步':kind==='schema_reassessment'?'Schema 重新评估':kind==='document_cleanup'?'文档清理':'知识库构建';const tick=async()=>{const res=await fetch('/api/jobs/'+id),job=await res.json();log.textContent=(job.log||[]).join('\n');renderSubtasks(job);applyJobReadiness(job);if(['completed','failed','cancelled'].includes(job.status)){setTimeout(()=>location.reload(),900);return}setTimeout(tick,800)};tick()}
if(box?.dataset.job)monitor(box.dataset.job,box.dataset.kind||'');
const deleteDialog=document.querySelector('#delete-progress-dialog'),deleteSpinner=document.querySelector('#delete-progress-spinner'),deleteResult=document.querySelector('#delete-progress-result'),deleteTitle=document.querySelector('#delete-progress-title'),deleteMessage=document.querySelector('#delete-progress-message'),deleteDetail=document.querySelector('#delete-progress-detail'),deleteBar=document.querySelector('#delete-progress-bar'),deleteClose=document.querySelector('#delete-progress-close');let deletePollTimer=null;
function updateDeleteProgress(progress,title,detail){deleteBar.style.width=clampProgress(progress)+'%';deleteTitle.textContent=title;deleteDetail.textContent=detail||''}
function finishDeleteProgress(success,title,detail){deleteSpinner.hidden=true;deleteResult.hidden=false;deleteResult.className='delete-result '+(success?'success':'failure');deleteResult.textContent=success?'✓':'!';updateDeleteProgress(100,title,detail);deleteClose.disabled=false;if(success)setTimeout(()=>location.reload(),1300)}
async function pollDeleteJob(jobId){try{const response=await fetch('/api/jobs/'+jobId),job=await response.json();if(!response.ok)throw new Error(job.error||'删除任务不存在');const latest=(job.log||[]).at(-1)||'后台清理正在进行';updateDeleteProgress(job.progress||10,job.status==='queued'?'正在等待解析步骤退出':'正在清理派生数据',latest);if(job.status==='completed'){finishDeleteProgress(true,'删除完成','文档、知识库贡献、图谱数据和过程文件已完成清理。');return}if(job.status==='failed'||job.status==='cancelled'){finishDeleteProgress(false,'后台清理未完成',job.error_message||latest);return}deletePollTimer=setTimeout(()=>pollDeleteJob(jobId),700)}catch(error){finishDeleteProgress(false,'删除状态读取失败',error.message||'请刷新页面确认文档状态')}}
document.querySelectorAll('[data-delete]').forEach(button=>button.onclick=async()=>{if(!confirm('确认软删除该文档及其向量、实体提及和关系贡献吗？'))return;button.disabled=true;deleteSpinner.hidden=false;deleteResult.hidden=true;deleteResult.className='delete-result';deleteClose.disabled=true;deleteMessage.textContent='正在提交删除请求...';updateDeleteProgress(3,'正在取消相关任务','请稍候，不要重复点击删除。');deleteDialog?.showModal();try{const response=await fetch('/api/documents/'+button.dataset.delete+'/delete',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'删除失败');button.closest('tr')?.classList.add('deleting-row');deleteMessage.textContent='文档已从当前项目移除，正在完成后台清理';updateDeleteProgress(8,'文档已软删除','正在等待活动解析步骤安全退出。');pollDeleteJob(data.job_id)}catch(error){button.disabled=false;finishDeleteProgress(false,'删除失败',error.message||'删除请求执行失败')}});
deleteClose?.addEventListener('click',()=>location.reload());deleteDialog?.addEventListener('cancel',event=>{if(deleteClose.disabled)event.preventDefault()});
document.querySelector('[data-reassess-schema]')?.addEventListener('click',async event=>{if(!confirm('将基于当前项目全部已入库文档，从基础 Schema 重新评估类型和关系约束。该操作不会自动重跑已有文档的实体抽取，是否继续？'))return;const button=event.currentTarget;button.disabled=true;try{const response=await fetch('/api/projects/'+button.dataset.reassessSchema+'/schema/reassess',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'Schema 重新评估提交失败');monitor(data.job_id,'schema_reassessment')}catch(error){alert(error.message||'Schema 重新评估提交失败');button.disabled=false}});
document.querySelector('[data-rebuild-graph]')?.addEventListener('click',async event=>{const button=event.currentTarget;button.disabled=true;try{const response=await fetch('/api/projects/'+button.dataset.rebuildGraph+'/graph',{method:'POST'}),data=await response.json();if(!response.ok)throw new Error(data.error||'图谱重建提交失败');monitor(data.job_id,'graph_sync')}catch(error){alert(error.message||'图谱重建提交失败');button.disabled=false}});
const createDialog=document.querySelector('#create-project-dialog');
document.querySelector('#open-create-project')?.addEventListener('click',()=>createDialog?.showModal());
document.querySelectorAll('[data-close-create-project]').forEach(button=>button.addEventListener('click',()=>createDialog?.close()));
createDialog?.addEventListener('click',event=>{if(event.target===createDialog)createDialog.close()});
const settingsDialog=document.querySelector('#project-settings-dialog');
const settingsForm=document.querySelector('#project-settings-form'),llmSelect=settingsForm?.querySelector('select[name="llm_model"]'),embeddingSelect=settingsForm?.querySelector('select[name="embedding_model"]'),modelTokenInput=settingsForm?.querySelector('input[name="model_max_tokens"]'),chunkTokenInput=settingsForm?.querySelector('input[name="chunk_max_tokens"]'),modelTokenHint=document.querySelector('#model-token-limit-hint'),chunkTokenHint=document.querySelector('#chunk-token-limit-hint');
let tokenLimits=null;
function applyTokenLimits(clamp=false){if(!tokenLimits||!modelTokenInput||!chunkTokenInput)return;const llmMax=Number(tokenLimits.llm_context_max),embeddingMax=Number(tokenLimits.embedding_context_max);modelTokenInput.max=String(llmMax);if(clamp&&Number(modelTokenInput.value)>llmMax)modelTokenInput.value=String(llmMax);const chunkMax=Math.min(Number(modelTokenInput.value)||llmMax,llmMax,embeddingMax);chunkTokenInput.max=String(chunkMax);if(clamp&&Number(chunkTokenInput.value)>chunkMax)chunkTokenInput.value=String(chunkMax);modelTokenHint.textContent='当前 LLM 支持的最大上下文：'+llmMax.toLocaleString()+' Token';modelTokenHint.classList.remove('failure');chunkTokenHint.textContent='当前模型组合允许的最大切片：'+chunkMax.toLocaleString()+' Token（Embedding 上限 '+embeddingMax.toLocaleString()+'）';chunkTokenHint.classList.remove('failure')}
async function refreshTokenLimits(clamp=false){if(!llmSelect||!embeddingSelect)return;modelTokenHint.textContent='正在读取当前 LLM 的上下文上限...';chunkTokenHint.textContent='正在读取 LLM 与 Embedding 的共同上限...';try{const query=new URLSearchParams({llm_model:llmSelect.value,embedding_model:embeddingSelect.value});const response=await fetch('/api/models/limits?'+query),data=await response.json();if(!response.ok)throw new Error(data.error||'模型上限读取失败');tokenLimits=data;applyTokenLimits(clamp)}catch(error){modelTokenHint.textContent=error.message||'模型上限读取失败';chunkTokenHint.textContent='请先确认 Ollama 模型可用';modelTokenHint.classList.add('failure');chunkTokenHint.classList.add('failure')}}
function replaceModelOptions(select,models,kind){if(!select)return;const current=select.value,names=models.filter(model=>model.model_type===kind).map(model=>model.name);if(current&&!names.includes(current))names.unshift(current);select.replaceChildren(...names.map(name=>new Option(name,name,name===current,name===current)))}
document.querySelector('#sync-models')?.addEventListener('click',async e=>{const button=e.currentTarget,status=document.querySelector('#model-sync-status');button.disabled=true;if(status){status.hidden=true;status.className='model-sync-status'}try{const r=await fetch('/api/models/sync',{method:'POST'}),d=await r.json();if(!r.ok)throw new Error(d.error||'同步失败');replaceModelOptions(llmSelect,d.models||[],'llm');replaceModelOptions(embeddingSelect,d.models||[],'embedding');await refreshTokenLimits(true);if(status){status.textContent='已同步 '+d.count+' 个 Ollama 模型';status.classList.add('success');status.hidden=false}}catch(error){if(status){status.textContent=error.message||'同步失败';status.classList.add('failure');status.hidden=false}}finally{button.disabled=false}});
document.querySelector('#open-project-settings')?.addEventListener('click',()=>{settingsDialog?.showModal();refreshTokenLimits(false)});
llmSelect?.addEventListener('change',()=>refreshTokenLimits(true));embeddingSelect?.addEventListener('change',()=>refreshTokenLimits(true));modelTokenInput?.addEventListener('input',()=>applyTokenLimits(true));
document.querySelectorAll('[data-close-project-settings]').forEach(button=>button.addEventListener('click',()=>settingsDialog?.close()));
settingsDialog?.addEventListener('click',event=>{if(event.target===settingsDialog)settingsDialog.close()});
const processingForm=document.querySelector('.processing-form'),processingSelects=document.querySelector('#processing-selects'),processingHint=document.querySelector('#processing-hint'),fileInput=document.querySelector('#upload-files');
const parserSupport={native_docx:['.docx'],native_text:['.md','.markdown','.txt'],plaintext_pdf:['.pdf'],docling:['.pdf','.docx'],spreadsheet:['.xlsx','.xls','.csv']};
const strategySupport={default:['.docx','.pdf','.md','.markdown','.txt'],general:['.docx','.pdf','.md','.markdown','.txt','.xlsx','.xls','.csv'],paper:['.docx','.pdf','.md','.markdown','.txt'],table:['.docx','.pdf','.xlsx','.xls','.csv']};
function selectedSuffixes(){return [...(fileInput?.files||[])].map(file=>{const name=file.name.toLowerCase();return name.includes('.')?name.slice(name.lastIndexOf('.')):''})}
function refreshSelect(select,support,suffixes){let available=0;for(const option of select?.options||[]){option.disabled=option.value!=='auto'&&(suffixes.some(suffix=>!support[option.value]?.includes(suffix))||(option.value==='docling'&&!window.KF.docling));if(!option.disabled)available++}if(select?.selectedOptions[0]?.disabled)select.value='auto';return available}
function refreshProcessing(){const selects=processingSelects?.querySelectorAll('select')||[];const suffixes=selectedSuffixes();refreshSelect(selects[0],parserSupport,suffixes);refreshSelect(selects[1],strategySupport,suffixes);const parser=selects[0]?.value||'auto',strategy=selects[1]?.value||'auto';if(processingHint)processingHint.textContent=!suffixes.length?'选择文件后会检查兼容性。':parser==='auto'&&strategy==='auto'?'系统将根据文件类型、文本层和文档结构自动选择解析器与分块策略。':'当前选择可用于所选文件。'}
processingSelects?.querySelectorAll('select').forEach(select=>select.addEventListener('change',refreshProcessing));fileInput?.addEventListener('change',refreshProcessing);refreshProcessing();
const jobDialog=document.querySelector('#job-log-dialog'),jobHistory=document.querySelector('#job-history-log'),jobSummary=document.querySelector('#job-log-summary'),jobError=document.querySelector('#job-log-error');
const jobStatus={queued:'排队中',running:'执行中',completed:'已完成',failed:'失败',cancelled:'已取消'};
function jobStageSummary(job){const key=job.metadata?.active_subtask,step=jobSubtasks(job)[key];if(!key||!step)return'';const suffix=step.status==='completed'?'已完成':step.status==='running'?clampProgress(step.progress)+'%':step.status==='failed'?'失败':step.status==='cancelled'?'已取消':'等待';return(subtaskLabels[key]||key)+' · '+suffix+(step.detail?' · '+step.detail:'')}
async function showJobLog(id){if(!id)return;jobDialog?.showModal();jobSummary.textContent='正在读取任务日志...';jobHistory.textContent='';jobError.hidden=true;const response=await fetch('/api/jobs/'+id),job=await response.json();if(!response.ok){jobSummary.textContent='日志读取失败';jobError.textContent=job.error||'任务不存在';jobError.hidden=false;return}const degraded=Number(job.metadata?.extraction_errors||0),stageSummary=jobStageSummary(job);jobSummary.textContent=(jobStatus[job.status]||job.status)+(stageSummary?' · '+stageSummary:'')+(degraded?' · '+degraded+' 个知识块已降级':'');jobHistory.textContent=(job.log||[]).join('\n')||'该任务暂无日志。';if(job.error_message){jobError.textContent=job.error_message;jobError.hidden=false}}
document.querySelectorAll('[data-view-job-log]').forEach(button=>button.addEventListener('click',()=>showJobLog(button.dataset.job)));
document.querySelectorAll('[data-close-job-log]').forEach(button=>button.addEventListener('click',()=>jobDialog?.close()));
jobDialog?.addEventListener('click',event=>{if(event.target===jobDialog)jobDialog.close()});
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
const toast=document.querySelector('.notice');if(toast){setTimeout(()=>toast.classList.add('toast-hide'),4200);setTimeout(()=>toast.remove(),4550);const url=new URL(location.href);url.searchParams.delete('kind');url.searchParams.delete('message');history.replaceState({},'',url.pathname+url.search)}
document.querySelector('#search-form')?.addEventListener('submit',async e=>{e.preventDefault();const target=document.querySelector('#search-results'),q=new FormData(e.currentTarget).get('query');target.innerHTML='<p class="muted">正在检索...</p>';const r=await fetch('/api/search?project='+window.KF.projectId+'&q='+encodeURIComponent(q)),d=await r.json();if(!r.ok){target.innerHTML='<p class="failure">'+esc(d.error)+'</p>';return}let h='<div class="result-group"><h2>知识块</h2>';for(const x of d.chunks)h+='<div class="result-card"><h3>'+esc(x.section_title||x.object_key)+' <span class="score">'+Number(x.score).toFixed(3)+'</span></h3><p>'+esc(x.content)+'</p><a href="'+esc(x.storage_path)+'" target="_blank">'+esc(x.original_filename)+'</a></div>';h+='</div><div class="result-group"><h2>实体</h2>';for(const x of d.entities)h+='<div class="result-card"><h3>'+esc(x.name)+' <span class="muted">'+esc(x.entity_type)+'</span></h3><p>'+esc(x.description)+'</p></div>';h+='</div><div class="result-group"><h2>关系</h2>';for(const x of d.relations)h+='<div class="result-card"><h3>'+esc(x.source_name)+' <span class="relation">'+esc(x.relation_type)+'</span> '+esc(x.target_name)+'</h3><p>'+esc(x.description)+'</p><a href="'+esc(x.storage_path)+'" target="_blank">'+esc(x.original_filename)+'</a></div>';target.innerHTML=h});
"""


if __name__ == "__main__":
    serve()
