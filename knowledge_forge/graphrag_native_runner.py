"""Process-isolated GraphRAG 3.1.1 runner and provider doctor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Callable

import requests


class GraphRAGNativeRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class NativeProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    started_at: str
    completed_at: str
    duration_ms: int
    timed_out: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class NativeProcessProgress:
    """A safe, structured progress event emitted by a running GraphRAG CLI."""

    stage: str
    message: str
    elapsed_seconds: int
    workflow: str = ""
    completed: int | None = None
    total: int | None = None
    heartbeat: bool = False
    workflow_state: str = ""


class GraphRAGNativeRunner:
    """Run GraphRAG only in a fresh staging process group, never in the web worker."""

    def __init__(
        self,
        *,
        python: str = sys.executable,
        timeout_seconds: int = 3600,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.python = str(python)
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.popen_factory = popen_factory

    @staticmethod
    def _redact(value: str, environment: dict[str, str]) -> str:
        redacted = str(value or "")
        for key, secret in environment.items():
            if secret and ("key" in key.lower() or "token" in key.lower() or "password" in key.lower()):
                redacted = redacted.replace(secret, "${REDACTED}")
        redacted = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s]+", r"\1${REDACTED}", redacted)
        return redacted

    @staticmethod
    def _append_event(root: Path, result: NativeProcessResult, stage: str) -> None:
        log_root = root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        event = {"stage": stage, **asdict(result)}
        with (log_root / "runner-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _progress_events_from_log(
        root: Path, offset: int, *, stage: str, elapsed_seconds: int,
    ) -> tuple[int, list[NativeProcessProgress]]:
        """Read newly appended, non-sensitive GraphRAG workflow lines.

        GraphRAG's verbose output can contain document payloads and model
        material.  Only its structural workflow and ``n/total`` progress
        records are forwarded to the user-visible job log.
        """

        path = Path(root).resolve() / "logs" / "indexing-engine.log"
        if not path.is_file() or path.is_symlink():
            return offset, []
        size = path.stat().st_size
        start = 0 if offset < 0 or offset > size else offset
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(start)
            chunk = stream.read()
            next_offset = stream.tell()
        events: list[NativeProcessProgress] = []
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            started = re.search(r"Workflow started:\s*([A-Za-z0-9_]+)", line)
            if started:
                workflow = started.group(1)
                events.append(NativeProcessProgress(
                    stage=stage, workflow=workflow,
                    message=f"GraphRAG workflow 开始：{workflow}",
                    elapsed_seconds=elapsed_seconds,
                    workflow_state="started",
                ))
                continue
            completed = re.search(r"Workflow completed:\s*([A-Za-z0-9_]+)", line)
            if completed:
                workflow = completed.group(1)
                events.append(NativeProcessProgress(
                    stage=stage, workflow=workflow,
                    message=f"GraphRAG workflow 完成：{workflow}",
                    elapsed_seconds=elapsed_seconds,
                    workflow_state="completed",
                ))
                continue
            # GraphRAG's logger prefix itself contains separators such as
            # ``INFO - graphrag.logger.progress -``.  Progress semantics begin
            # after the final logger separator, not at the beginning of line.
            progress_text = line.rsplit(" - ", 1)[-1]
            progress = re.search(
                r"(?P<workflow>[A-Za-z0-9_ -]+?)\s+progress:\s*(?P<completed>\d+)\s*/\s*(?P<total>\d+)",
                progress_text,
            )
            if progress:
                workflow = " ".join(progress.group("workflow").split())
                current = int(progress.group("completed")); total = int(progress.group("total"))
                if total > 0 and 0 <= current <= total:
                    events.append(NativeProcessProgress(
                        stage=stage, workflow=workflow,
                        completed=current, total=total,
                        message=f"GraphRAG {workflow}：{current}/{total}",
                        elapsed_seconds=elapsed_seconds,
                        workflow_state="progress",
                    ))
        return next_offset, events

    def _run(
        self,
        *,
        root: Path,
        stage: str,
        argv: list[str],
        environment: dict[str, str] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[NativeProcessProgress], None] | None = None,
        timeout_seconds: int | None = None,
    ) -> NativeProcessResult:
        root = Path(root).resolve()
        if not root.is_dir() or root.is_symlink():
            raise GraphRAGNativeRunnerError("GraphRAG staging root 不存在或不安全")
        environment = dict(environment or {})
        # GraphRAG imports LiteLLM.  Its default cost-map refresh is unrelated
        # to native indexing/query correctness and introduces an unexpected
        # outbound GitHub request, so native isolated workers use LiteLLM's
        # documented local bundled map unless an operator explicitly overrides
        # it for a separate purpose.
        environment.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        merged_environment = {**os.environ, **environment}
        started = datetime.now(timezone.utc)
        tick = time.monotonic()
        process = self.popen_factory(
            list(argv), cwd=str(root), env=merged_environment, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        timed_out = False
        cancelled = False
        limit = max(1, int(timeout_seconds or self.timeout_seconds))
        progress_offset = 0
        last_heartbeat = tick

        def publish_progress() -> None:
            nonlocal progress_offset, last_heartbeat
            if progress_callback is None:
                return
            elapsed = int(time.monotonic() - tick)
            progress_offset, events = self._progress_events_from_log(
                root, progress_offset, stage=stage, elapsed_seconds=elapsed,
            )
            if not events and time.monotonic() - last_heartbeat >= 15:
                events = [NativeProcessProgress(
                    stage=stage,
                    message="GraphRAG Standard workflow 仍在运行，等待下一条工作流进度",
                    elapsed_seconds=elapsed, heartbeat=True,
                )]
            for event in events:
                try:
                    progress_callback(event)
                except Exception:
                    # Observability never gets to terminate the isolated
                    # semantic build; the durable CLI event log remains.
                    pass
            if events:
                last_heartbeat = time.monotonic()
        try:
            while True:
                if cancel_requested and cancel_requested():
                    cancelled = True
                    raise TimeoutError("GraphRAG Native run 已取消")
                if time.monotonic() - tick > limit:
                    timed_out = True
                    raise TimeoutError("GraphRAG Native 子进程超时")
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    publish_progress()
                    break
                except subprocess.TimeoutExpired:
                    publish_progress()
                    continue
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, AttributeError):
                process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, AttributeError):
                    process.kill()
                stdout, stderr = process.communicate()
        completed = datetime.now(timezone.utc)
        result = NativeProcessResult(
            argv=tuple(argv), returncode=int(process.returncode or 0),
            stdout=self._redact(stdout or "", environment),
            stderr=self._redact(stderr or "", environment),
            started_at=started.isoformat(), completed_at=completed.isoformat(),
            duration_ms=int((time.monotonic() - tick) * 1000), timed_out=timed_out,
            cancelled=cancelled,
        )
        self._append_event(root, result, stage)
        if timed_out or cancelled:
            raise GraphRAGNativeRunnerError(
                "GraphRAG Native 子进程已取消" if cancelled else "GraphRAG Native 子进程超时"
            )
        if result.returncode != 0:
            raise GraphRAGNativeRunnerError(
                f"GraphRAG Native {stage} 失败 (exit={result.returncode}): {result.stderr[-1000:]}"
            )
        return result

    @staticmethod
    def _root_is_empty(root: Path) -> bool:
        return root.is_dir() and not any(root.iterdir())

    def initialize(
        self, *, root: Path, completion_model: str, embedding_model: str,
        environment: dict[str, str],
        cancel_requested: Callable[[], bool] | None = None,
    ) -> NativeProcessResult:
        root = Path(root).resolve()
        if not self._root_is_empty(root):
            raise GraphRAGNativeRunnerError("graphrag init 只允许在全新的空 staging root 执行")
        return self._run(
            root=root, stage="init",
            argv=[self.python, "-m", "graphrag", "init", "--root", str(root), "--model", completion_model, "--embedding", embedding_model],
            environment=environment, cancel_requested=cancel_requested,
        )

    def config_dry_run(
        self, *, root: Path, environment: dict[str, str],
        cancel_requested: Callable[[], bool] | None = None,
    ) -> NativeProcessResult:
        return self._run(
            root=root, stage="config_dry_run",
            argv=[self.python, "-m", "graphrag", "index", "--root", str(Path(root).resolve()), "--method", "standard", "--dry-run", "--skip-validation", "--verbose"],
            environment=environment, cancel_requested=cancel_requested,
        )

    def connectivity_dry_run(
        self, *, root: Path, environment: dict[str, str],
        cancel_requested: Callable[[], bool] | None = None,
    ) -> NativeProcessResult:
        return self._run(
            root=root, stage="connectivity_dry_run",
            argv=[self.python, "-m", "graphrag", "index", "--root", str(Path(root).resolve()), "--method", "standard", "--dry-run", "--verbose"],
            environment=environment, cancel_requested=cancel_requested,
        )

    def index(
        self, *, root: Path, environment: dict[str, str],
        cancel_requested: Callable[[], bool] | None = None,
        progress_callback: Callable[[NativeProcessProgress], None] | None = None,
    ) -> NativeProcessResult:
        return self._run(
            root=root, stage="index",
            argv=[self.python, "-m", "graphrag", "index", "--root", str(Path(root).resolve()), "--method", "standard", "--verbose"],
            environment=environment, cancel_requested=cancel_requested,
            progress_callback=progress_callback,
        )


class GraphRAGNativeProviderDoctor:
    """Explicit provider and embedding-dimension probes; never guessed from a model name."""

    @staticmethod
    def config_only(python: str = sys.executable) -> dict[str, str]:
        from importlib.metadata import version
        from graphrag.config.models.graph_rag_config import GraphRagConfig

        installed = version("graphrag")
        if installed != "3.1.1":
            raise GraphRAGNativeRunnerError(f"需要 graphrag==3.1.1，当前为 {installed}")
        return {
            "python": str(python), "graphrag_version": installed,
            "config_fields": ",".join(sorted(GraphRagConfig.model_fields)),
        }

    @staticmethod
    def probe_embedding(*, api_base: str, model: str, api_key: str, timeout: int = 30) -> dict[str, Any]:
        base = str(api_base).rstrip("/")
        response = requests.post(
            base + "/embeddings", json={"model": model, "input": "native dimension probe"},
            headers={"Authorization": "Bearer " + str(api_key or "graphrag-local-placeholder")},
            timeout=max(1, int(timeout)),
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        vector = data[0].get("embedding") if isinstance(data, list) and data and isinstance(data[0], dict) else None
        if not isinstance(vector, list) or not vector or any(not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in vector):
            raise GraphRAGNativeRunnerError("embedding probe 返回空向量或非有限数值")
        return {"model": model, "dimension": len(vector), "endpoint": base}
