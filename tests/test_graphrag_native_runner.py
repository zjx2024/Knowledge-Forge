"""Process isolation and failure protocol tests for GraphRAG Native runner."""

from __future__ import annotations

import json
import subprocess

import pytest

from knowledge_forge.graphrag_native_runner import (
    GraphRAGNativeProviderDoctor,
    GraphRAGNativeRunner,
    GraphRAGNativeRunnerError,
)


class _Process:
    def __init__(self, *, returncode=0, stdout="ok", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.pid = 999999

    def communicate(self, timeout=None):
        return self.stdout, self.stderr

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_runner_uses_argv_fresh_root_and_redacts_process_logs(tmp_path):
    calls = []

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return _Process(stderr="Authorization: Bearer top-secret")

    runner = GraphRAGNativeRunner(python="native-python", popen_factory=popen)
    result = runner.initialize(
        root=tmp_path, completion_model="chat", embedding_model="embed",
        environment={"GRAPHRAG_API_KEY": "top-secret"},
    )

    assert result.returncode == 0
    assert calls[0][0] == ["native-python", "-m", "graphrag", "init", "--root", str(tmp_path.resolve()), "--model", "chat", "--embedding", "embed"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["start_new_session"] is True
    assert calls[0][1]["env"]["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    event = json.loads((tmp_path / "logs/runner-events.jsonl").read_text(encoding="utf-8"))
    assert "top-secret" not in event["stderr"]
    assert "REDACTED" in event["stderr"]
    with pytest.raises(GraphRAGNativeRunnerError, match="空 staging root"):
        runner.initialize(root=tmp_path, completion_model="chat", embedding_model="embed", environment={})


def test_runner_fails_on_nonzero_exit_without_claiming_success(tmp_path):
    runner = GraphRAGNativeRunner(
        popen_factory=lambda *args, **kwargs: _Process(returncode=17, stderr="configuration bad"),
    )
    with pytest.raises(GraphRAGNativeRunnerError, match="exit=17"):
        runner.config_dry_run(root=tmp_path, environment={})


def test_index_forwards_only_structural_workflow_progress(tmp_path):
    class StreamingProcess(_Process):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                log_root = tmp_path / "logs"; log_root.mkdir(exist_ok=True)
                (log_root / "indexing-engine.log").write_text(
                    "INFO - Workflow started: extract_graph\n"
                    "INFO - extract graph progress: 1/9\n"
                    "DEBUG - document payload: secret source text\n"
                    "INFO - Workflow completed: extract_graph\n",
                    encoding="utf-8",
                )
                raise subprocess.TimeoutExpired("graphrag", timeout)
            return super().communicate(timeout)

    events = []
    runner = GraphRAGNativeRunner(popen_factory=lambda *_args, **_kwargs: StreamingProcess())

    runner.index(root=tmp_path, environment={}, progress_callback=events.append)

    assert [event.message for event in events] == [
        "GraphRAG workflow 开始：extract_graph",
        "GraphRAG extract graph：1/9",
        "GraphRAG workflow 完成：extract_graph",
    ]
    assert events[1].completed == 1 and events[1].total == 9
    assert all("secret source text" not in event.message for event in events)


def test_native_doctor_verifies_actual_graphrag_version_and_config_surface():
    report = GraphRAGNativeProviderDoctor.config_only("python")
    assert report["graphrag_version"] == "3.1.1"
    assert "completion_models" in report["config_fields"]
