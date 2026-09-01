"""Real HTTP contracts for the first knowledge-agent browser API."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from http.cookiejar import CookieJar
from urllib.error import HTTPError
from urllib.request import (
    HTTPCookieProcessor,
    Request,
    build_opener,
    urlopen,
)

import pytest

from knowledge_forge import web


THREAD_ID = "ath_" + "1" * 32
RUN_ID = "arun_" + "2" * 32
EVIDENCE_ID = "ev_" + "3" * 32


class _AgentHttpService:
    def __init__(self) -> None:
        self.session_key = ""
        self.thread_created = False
        self.thread_archived = False
        self.archive_blocked = False
        self.cancelled = False
        self.feedback = None

    def agent_knowledge_bases(self):
        return {
            "items": [{
                "id": 61, "name": "系统手册", "description": "",
                "strategy": "knowledge_forge", "queryable": True,
                "status": "ready_hybrid", "status_label": "可问答 · 原文与图谱",
                "knowledge_language": "zh-CN", "details": {},
            }],
            "queryable_count": 1, "total": 1,
            "scope_policy": "single_project_explicit_v1",
        }

    def create_agent_thread(self, project_id, *, session_key):
        assert project_id == 61
        self.session_key = session_key
        self.thread_created = True
        return {
            "id": THREAD_ID, "project_id": 61,
            "strategy_key": "knowledge_forge", "status": "active",
        }

    def list_agent_threads(self, project_id, *, session_key, limit=100):
        assert project_id == 61
        assert limit == 100
        if not self.session_key:
            self.session_key = session_key
        if session_key != self.session_key or not self.thread_created:
            return {"items": [], "total": 0, "project_id": 61}
        return {
            "items": [{
                "id": THREAD_ID, "project_id": 61,
                "strategy_key": "knowledge_forge", "status": "active",
                "title": "任务管理有什么作用？", "message_count": 2,
                "last_message_preview": "任务管理负责创建和调度任务。",
                "active_run_id": "", "updated_at": "2026-08-26T10:00:00+08:00",
            }],
            "total": 1, "project_id": 61,
        }

    def _authorize(self, session_key):
        if not self.session_key or session_key != self.session_key:
            raise PermissionError("问答会话不存在或无权访问")

    def get_agent_thread(self, thread_id, *, session_key):
        self._authorize(session_key)
        assert thread_id == THREAD_ID
        if not self.thread_created:
            raise PermissionError("问答会话不存在或无权访问")
        return {"id": THREAD_ID, "project_id": 61, "messages": [], "runs": []}

    def archive_agent_thread(self, thread_id, *, session_key):
        self._authorize(session_key)
        assert thread_id == THREAD_ID
        if self.archive_blocked:
            raise RuntimeError("该会话正在生成回答，请先停止回答再删除")
        self.thread_created = False
        self.thread_archived = True
        return {"id": THREAD_ID, "project_id": 61, "status": "archived"}

    def submit_agent_message(
        self, thread_id, *, session_key, query, requested_mode="auto",
    ):
        self._authorize(session_key)
        assert thread_id == THREAD_ID
        assert query == "任务管理有什么作用？"
        assert requested_mode == "auto"
        return {
            "run_id": RUN_ID, "message_id": "amsg_" + "4" * 32,
            "thread_id": THREAD_ID, "status": "queued",
            "events_url": f"/api/agent/runs/{RUN_ID}/events",
        }

    def get_agent_run(self, run_id, *, session_key):
        self._authorize(session_key)
        assert run_id == RUN_ID
        return {
            "id": RUN_ID, "project_id": 61, "status": "completed",
            "answer_text": "任务管理负责创建和调度任务〔1〕。",
            "citations": [{"evidence_id": EVIDENCE_ID}],
        }

    def get_agent_run_trace(self, run_id, *, session_key):
        self._authorize(session_key)
        assert run_id == RUN_ID
        return {
            "protocol": "knowledge-agent-execution-trace-v2",
            "status": "completed", "current_stage": "completed",
            "current_label": "回答完成", "duration_ms": 1250,
            "step_count": 2, "tool_count": 1,
            "decisions": [{
                "title": "检索选择",
                "detail": "按事实查找问题分配检索预算，使用 Knowledge Forge",
            }],
            "items": [{
                "kind": "tool", "key": "tool:one",
                "title": "Knowledge Forge 混合检索",
                "detail": "返回 8 条带来源候选",
                "status": "completed",
                "technical_name": "retrieve.knowledge_forge",
                "duration_ms": 700,
                "details": [
                    {"label": "请求方式", "value": "Knowledge Forge 动态混合检索"},
                    {"label": "返回候选", "value": "8 条"},
                ],
                "evidence_title": "该次召回中最终保留的证据",
                "evidence": [{
                    "evidence_id": EVIDENCE_ID,
                    "evidence_type": "chunk", "type_label": "原文",
                    "title": "任务管理", "excerpt": "任务管理负责创建和调度任务。",
                    "location": "manual.pdf · 第 12 页",
                    "citation_index": 1, "cited": True,
                }],
            }],
            "disclosure": (
                "这里展示可审计的处理步骤、检索决策和工具调用；"
                "不包含模型隐藏推理、系统提示词或未验证草稿。"
            ),
        }

    def agent_events(self, run_id, *, session_key, after_id=0):
        self._authorize(session_key)
        assert run_id == RUN_ID
        events = [
            {"id": 1, "event_type": "status", "payload": {"status": "running"}},
            {"id": 2, "event_type": "answer_start", "payload": {"message_id": "m"}},
            {"id": 3, "event_type": "answer_delta", "payload": {"delta": "任务管理负责创建和调度任务〔1〕。"}},
            {"id": 4, "event_type": "sources", "payload": {"items": [{
                "evidence_id": EVIDENCE_ID, "citation_index": 1,
            }]}},
            {"id": 5, "event_type": "completed", "payload": {"status": "completed"}},
        ]
        return [item for item in events if item["id"] > after_id]

    def get_agent_evidence(self, evidence_id, *, session_key):
        self._authorize(session_key)
        assert evidence_id == EVIDENCE_ID
        return {
            "evidence_id": EVIDENCE_ID, "project_id": 61,
            "evidence_type": "chunk", "title": "任务管理",
            "content": "任务管理负责创建和调度任务。",
            "source_refs": [{
                "document_id": 9, "filename": "system.pdf",
                "document_url": "/api/documents/9/download",
                "page_label": "第 12 页", "section_title": "任务管理",
                "media_assets": [{
                    "url": "http://files.test/media/task.png",
                    "title": "任务管理流程图",
                }],
            }],
        }

    def cancel_agent_run(self, run_id, *, session_key):
        self._authorize(session_key)
        assert run_id == RUN_ID
        self.cancelled = True
        return {"id": RUN_ID, "status": "cancelled"}

    def record_agent_feedback(
        self, run_id, *, session_key, rating, comment="", action="answer_rating",
    ):
        self._authorize(session_key)
        assert run_id == RUN_ID
        self.feedback = (rating, comment, action)
        return {"id": "afb_1", "run_id": RUN_ID, "rating": rating}


def _json_request(opener, url, *, method="GET", payload=None):
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, method=method, headers=headers)
    with opener.open(request, timeout=3) as response:
        return response.status, response.headers, json.loads(response.read().decode("utf-8"))


def test_agent_chat_inline_script_is_syntactically_valid_when_node_is_available():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed")
    page = web.render_chat_page()
    match = re.search(r"<script>(.*?)</script>", page, re.DOTALL)
    assert match is not None
    result = subprocess.run(
        [node, "--check", "-"], input=match.group(1), text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_agent_http_session_sse_evidence_cancel_and_feedback(monkeypatch):
    service = _AgentHttpService()
    monkeypatch.setattr(web, "SERVICE", service)
    server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.AppHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    try:
        with urlopen(base + "/chat", timeout=3) as response:
            page = response.read().decode("utf-8")
        assert response.status == 200
        assert 'id="agent-project"' in page
        assert 'id="agent-thread-list"' in page
        assert 'id="agent-new-thread"' in page
        assert 'id="agent-source-turn"' in page
        assert 'id="agent-source-content"' in page
        assert "displayTurnSources" in page
        source_transition = page.split("function displayTurnSources", 1)[1].split(
            "function turnLabel", 1,
        )[0]
        assert "renderSources(items,emptyMessage);if(!animate||reduced)return" in source_transition
        assert "opacity:.72" in source_transition
        assert "opacity:.18" not in source_transition
        assert "fill:'forwards'" not in source_transition
        assert "empty.hidden=Boolean(state.base?.queryable)" in page
        assert "height:clamp(650px,calc(100vh - 220px),900px)" in page
        assert '.agent-message-list::before{content:"";margin-top:auto}' in page
        assert "历史会话" in page
        assert 'id="agent-evidence-dialog"' in page
        assert "sourceCardMedia" in page
        assert "agent-source-card-media" in page
        assert "agent-execution-trace" in page
        assert "renderExecutionTrace" in page
        assert "appendTraceFacts" in page
        assert "appendTraceEvidence" in page
        assert "agent-trace-item-detail" in page
        assert "agent-trace-facts" in page
        assert "agent-trace-evidence-card" in page
        assert "执行过程" in page
        assert "/trace" in page
        assert "不包含模型隐藏推理" in page
        assert 'id="agent-mode-field" hidden' in page
        assert "selectedBaseIsNative" in page
        assert "modeField.hidden=!native" in page
        assert "mode:selectedBaseIsNative()?modeSelect.value:'auto'" in page
        assert 'id="agent-scope-state"' not in page
        assert "证据约束问答" not in page
        assert "Native Bundle" not in page
        assert "回答依据</h3>" in page

        status, _headers, bases = _json_request(
            opener, base + "/api/agent/knowledge-bases",
        )
        assert status == 200 and bases["scope_policy"] == "single_project_explicit_v1"

        status, headers, history = _json_request(
            opener, base + "/api/agent/threads?project_id=61",
        )
        assert status == 200 and history["items"] == []
        assert "knowledge_forge_agent_session" in headers.get("Set-Cookie", "")

        status, headers, created = _json_request(
            opener, base + "/api/agent/threads", method="POST",
            payload={"project_id": 61},
        )
        assert status == 201 and created["thread"]["id"] == THREAD_ID

        status, _headers, history = _json_request(
            opener, base + "/api/agent/threads?project_id=61",
        )
        assert status == 200 and history["total"] == 1
        assert history["items"][0]["title"] == "任务管理有什么作用？"

        status, _headers, submitted = _json_request(
            opener, base + f"/api/agent/threads/{THREAD_ID}/messages",
            method="POST", payload={"query": "任务管理有什么作用？", "mode": "auto"},
        )
        assert status == 202 and submitted["run_id"] == RUN_ID

        with opener.open(base + submitted["events_url"], timeout=3) as response:
            stream = response.read().decode("utf-8")
        assert response.headers["Content-Type"].startswith("text/event-stream")
        assert "event: answer_delta" in stream
        assert "event: sources" in stream
        assert "event: completed" in stream
        assert "任务管理负责创建和调度任务" in stream

        status, _headers, trace = _json_request(
            opener, base + f"/api/agent/runs/{RUN_ID}/trace",
        )
        assert status == 200
        assert trace["protocol"] == "knowledge-agent-execution-trace-v2"
        assert trace["tool_count"] == 1
        assert trace["items"][0]["technical_name"] == "retrieve.knowledge_forge"

        status, _headers, evidence = _json_request(
            opener, base + f"/api/agent/evidence/{EVIDENCE_ID}",
        )
        assert status == 200
        assert evidence["project_id"] == 61
        assert evidence["source_refs"][0]["document_url"] == "/api/documents/9/download"
        assert evidence["source_refs"][0]["media_assets"][0]["title"] == "任务管理流程图"

        status, _headers, feedback = _json_request(
            opener, base + f"/api/agent/runs/{RUN_ID}/feedback",
            method="POST", payload={"rating": 1, "comment": "引用清楚"},
        )
        assert status == 200 and feedback["feedback"]["rating"] == 1
        assert service.feedback == (1, "引用清楚", "answer_rating")

        status, _headers, cancelled = _json_request(
            opener, base + f"/api/agent/runs/{RUN_ID}/cancel", method="POST",
        )
        assert status == 202 and cancelled["run"]["status"] == "cancelled"
        assert service.cancelled is True

        outsider = build_opener(HTTPCookieProcessor(CookieJar()))
        status, _headers, outsider_history = _json_request(
            outsider, base + "/api/agent/threads?project_id=61",
        )
        assert status == 200 and outsider_history["items"] == []
        try:
            _json_request(outsider, base + f"/api/agent/runs/{RUN_ID}")
        except HTTPError as exc:
            assert exc.code == 403
        else:  # pragma: no cover - protects the capability boundary
            raise AssertionError("another browser session accessed an agent run")
        try:
            _json_request(outsider, base + f"/api/agent/runs/{RUN_ID}/trace")
        except HTTPError as exc:
            assert exc.code == 403
        else:  # pragma: no cover - protects the capability boundary
            raise AssertionError("another browser session accessed an execution trace")

        service.archive_blocked = True
        try:
            _json_request(
                opener, base + f"/api/agent/threads/{THREAD_ID}", method="DELETE",
            )
        except HTTPError as exc:
            assert exc.code == 409
            assert "正在生成回答" in exc.read().decode("utf-8")
        else:  # pragma: no cover - protects safe deletion while an answer runs
            raise AssertionError("an active conversation was archived")
        service.archive_blocked = False

        status, _headers, archived = _json_request(
            opener, base + f"/api/agent/threads/{THREAD_ID}", method="DELETE",
        )
        assert status == 200 and archived["thread"]["status"] == "archived"
        assert service.thread_archived is True
        status, _headers, history = _json_request(
            opener, base + "/api/agent/threads?project_id=61",
        )
        assert status == 200 and history["items"] == []
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
