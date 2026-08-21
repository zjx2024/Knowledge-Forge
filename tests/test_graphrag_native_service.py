"""Black-box executor tests: native calls only its isolated chain."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from knowledge_forge.graphrag_native_artifacts import NativeArtifact
from knowledge_forge.graphrag_native_config import GraphRAGNativePromptProfileBuilder
from knowledge_forge.graphrag_native_executor import GraphRAGNativeExecutor, NativeNoGraphUnsupported
from knowledge_forge.graphrag_native_inputs import NativeInputBuild, NativeInputPreflight
from knowledge_forge.graphrag_native_runner import NativeProcessProgress


class _Tripwire:
    def __getattr__(self, name):
        raise AssertionError(f"legacy entry point was called: {name}")


class _Database:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.job = {"id": "job-1", "status": "queued", "metadata": {}}
        self.fail_renewal = False

    def assert_graphrag_native_ready(self): self.calls.append(("ready", None)); return {}
    def get_project(self, project_id): return {"id": project_id, "llm_model": "llm", "embedding_model": "embed"}
    def latest_active_job(self, *_args): return None
    def update_job(self, _job_id, **values):
        self.calls.append(("job", values))
        metadata = values.pop("metadata", None)
        if metadata is not None:
            self.job["metadata"] = {**self.job.get("metadata", {}), **metadata}
        self.job.update(values)
        return self.job
    def get_job(self, _job_id): return dict(self.job)
    def create_graphrag_native_run(self, values): self.calls.append(("create", values)); return {"run_id": values["run_id"]}
    def acquire_graphrag_native_run_lease(self, run_id, _owner, **_kwargs): self.calls.append(("lease", run_id)); return {"run_id": run_id}
    def renew_graphrag_native_run_lease(self, run_id, _owner, **_kwargs):
        self.calls.append(("renew", run_id))
        if self.fail_renewal:
            raise RuntimeError("lease expired")
        return {"run_id": run_id}
    def replace_graphrag_native_run_documents(self, run_id, docs, **_values): self.calls.append(("documents", (run_id, docs)))
    def update_graphrag_native_run(self, run_id, **values): self.calls.append(("run", (run_id, values))); return {"run_id": run_id}
    def replace_graphrag_native_artifacts(self, run_id, artifacts, **_values): self.calls.append(("artifacts", (run_id, artifacts)))
    def activate_graphrag_native_run(self, run_id, **values):
        self.calls.append(("activate", (run_id, values)))
        self.job["metadata"] = {
            **self.job.get("metadata", {}),
            "publication_committed": True,
        }
        return {"run_id": run_id, "state": "READY"}


class _Runner:
    def initialize(self, *, root, **_kwargs):
        (root / "prompts").mkdir(parents=True, exist_ok=True)
        for name in GraphRAGNativePromptProfileBuilder.required_prompt_names:
            content = (
                "Return output in English with {input_text}."
                if name in {"extract_graph.txt", "extract_claims.txt"}
                else "native prompt"
            )
            (root / "prompts" / name).write_text(content, encoding="utf-8")
        return SimpleNamespace(returncode=0)
    def config_dry_run(self, **_kwargs): return SimpleNamespace(returncode=0)
    def connectivity_dry_run(self, **_kwargs): return SimpleNamespace(returncode=0)
    def index(self, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback:
            callback(NativeProcessProgress(
                stage="index", workflow="extract graph", completed=4, total=9,
                elapsed_seconds=10, message="GraphRAG extract graph：4/9",
            ))
        return SimpleNamespace(returncode=0)


class _InputBuilder:
    def __init__(self, _service): pass
    def preflight(self, *, project_id):
        return NativeInputPreflight(project_id=project_id, source_snapshot_hash="a" * 64, ready=True)
    def build(self, *, project_id, run_root):
        return NativeInputBuild(
            project_id=project_id, source_snapshot_hash="a" * 64, input_manifest_hash="b" * 64,
            documents_path=run_root / "input/documents.parquet", source_map_path=run_root / "kf-manifest/source-map.parquet",
            manifest_path=run_root / "kf-manifest/input-documents-manifest.jsonl", input_documents=(), source_maps=(),
            run_documents=({"project_id": project_id, "document_id": 1, "revision_id": "r1", "source_hash": "s", "structure_hash": "c" * 64, "input_document_count": 1, "input_manifest_hash": "b" * 64},),
        )


def _executor(monkeypatch, tmp_path):
    import knowledge_forge.graphrag_native_executor as module

    database = _Database()
    service = SimpleNamespace(
        db=database, ollama=SimpleNamespace(model_info=lambda _model: {"digest": "d" * 64}),
        vectors=_Tripwire(), resolver=_Tripwire(), graph=_Tripwire(), relation_shadow=_Tripwire(),
    )
    executor = GraphRAGNativeExecutor(service)
    executor.runner = _Runner()
    monkeypatch.setattr(module.Config, "files_root", tmp_path)
    monkeypatch.setattr(module, "GraphRAGNativeInputBuilder", _InputBuilder)
    monkeypatch.setattr(executor, "_candidate_identity", lambda: {"base_sha": "0" * 40, "head": "1" * 40, "worktree_diff_hash": "2" * 64, "code_tree_hash": "3" * 64})
    monkeypatch.setattr(executor, "_write_candidate_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_write_post_index_evidence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor, "_provider_identity", lambda _project: ({"completion_model_id": "llm", "completion_model_digest": "4" * 64, "embedding_model_id": "embed", "embedding_model_digest": "5" * 64}, 2))
    monkeypatch.setattr(executor.settings, "write_and_validate", lambda **_kwargs: {"settings_hash": "6" * 64})
    artifact = NativeArtifact("fake", "kf-manifest/fake.txt", "7" * 64, 1, "text/plain")
    monkeypatch.setattr(executor, "_collect_artifacts", lambda _root: [artifact])
    monkeypatch.setattr(executor, "_require_nonempty_semantic_graph", lambda _root: None)
    monkeypatch.setattr(executor, "_reject_source_metadata_pollution", lambda _root: None)
    monkeypatch.setattr(module, "write_run_manifest", lambda *_args, **_kwargs: tmp_path / "manifest")
    monkeypatch.setattr(executor.validator, "validate", lambda _root: {"ok": True})
    return executor, database


def test_native_executor_runs_only_native_chain_and_persists_control_plane(monkeypatch, tmp_path):
    executor, database = _executor(monkeypatch, tmp_path)

    executor.run(job_id="job-1", project_id=7)

    names = [name for name, _value in database.calls]
    assert names[0] == "ready"
    assert names.index("create") < names.index("documents") < names.index("artifacts") < names.index("activate")
    assert names.count("run") >= 3  # RUNNING → EXECUTION_COMPLETE → VALIDATING
    assert database.job["status"] == "completed"
    assert any(name == "activate" for name in names)
    progress_updates = [int(value["progress"]) for name, value in database.calls if name == "job" and "progress" in value]
    assert [72, 86, 94] == [value for value in progress_updates if value in {72, 86, 94}]
    assert 47 in progress_updates  # 30 + round(38 * 4 / 9), inside the live index band
    assert any(
        value.get("log") == "GraphRAG Native：GraphRAG extract graph：4/9"
        for name, value in database.calls if name == "job"
    )
    assert "renew" in names
    _run_id, activation_values = next(
        value for name, value in database.calls if name == "activate"
    )
    assert activation_values["job_id"] == "job-1"


def test_native_executor_lease_renewal_failure_stops_before_publication(monkeypatch, tmp_path):
    executor, database = _executor(monkeypatch, tmp_path)
    database.fail_renewal = True
    monkeypatch.setattr(executor, "_LEASE_RENEW_INTERVAL_SECONDS", 0)

    executor.run(job_id="job-1", project_id=7)

    names = [name for name, _value in database.calls]
    assert database.job["status"] == "failed"
    assert "lease" in str(database.job.get("error") or "")
    assert "renew" in names
    assert "activate" not in names
    assert "artifacts" not in names


def test_native_executor_failure_never_falls_back_to_a_legacy_chain(monkeypatch, tmp_path):
    executor, database = _executor(monkeypatch, tmp_path)
    monkeypatch.setattr(executor.runner, "connectivity_dry_run", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("provider unavailable")))

    executor.run(job_id="job-1", project_id=7)

    assert database.job["status"] == "failed"
    assert "禁止回退旧流程" in str(database.job.get("error") or "")
    assert "activate" not in [name for name, _value in database.calls]


def test_native_executor_cancellation_stops_before_publication_and_cleans_bundle(
    monkeypatch, tmp_path,
):
    executor, database = _executor(monkeypatch, tmp_path)

    def cancelled_index(**_kwargs):
        database.job.setdefault("metadata", {})["cancel_requested"] = True
        raise RuntimeError("isolated child stopped")

    monkeypatch.setattr(executor.runner, "index", cancelled_index)

    executor.run(job_id="job-1", project_id=7)

    names = [name for name, _value in database.calls]
    assert database.job["status"] == "cancelled"
    assert database.job["metadata"]["stage"] == "cancelled"
    assert database.job["metadata"]["cancel_acknowledged_at"]
    assert "activate" not in names
    assert "artifacts" not in names
    assert not list((tmp_path / "graphrag_native" / ".staging").glob("*"))


def test_candidate_evidence_is_secret_free_and_contains_reproducible_sources(monkeypatch, tmp_path):
    executor, _database = _executor(monkeypatch, tmp_path)
    root = tmp_path / "evidence"; (root / "kf-manifest").mkdir(parents=True)
    monkeypatch.setattr(executor, "_write_candidate_evidence", GraphRAGNativeExecutor._write_candidate_evidence.__get__(executor, GraphRAGNativeExecutor))
    monkeypatch.setattr(
        executor,
        "_git_output",
        lambda *_args: (
            '+fixture = "Authorization: Bearer secret-token-value-123456"\n'
        ),
    )

    executor._write_candidate_evidence(root, identity={"base_sha": "0" * 40, "head": "1" * 40, "worktree_diff_hash": "2" * 64, "code_tree_hash": "3" * 64}, prompt_hash="4" * 64)

    manifest = root / "kf-manifest"
    assert (manifest / "candidate-source-archive.tar.zst").is_file()
    assert (manifest / "candidate-source-manifest.json").is_file()
    assert "api_key" not in (manifest / "runtime-environment.json").read_text(encoding="utf-8").lower()
    assert "secret-token-value" not in (
        manifest / "candidate-worktree.patch"
    ).read_text(encoding="utf-8")
    redaction = json.loads(
        (manifest / "candidate-worktree-redaction.json").read_text(encoding="utf-8")
    )
    assert redaction["redacted_line_count"] == 1


def test_native_executor_reuses_only_compatible_completed_failed_call_cache(
    monkeypatch, tmp_path,
):
    executor, database = _executor(monkeypatch, tmp_path)
    source = tmp_path / "graphrag_native" / ".staging" / "failed-attempt"
    (source / "cache" / "nested").mkdir(parents=True)
    (source / "cache" / "nested" / "entry.json").write_text(
        '{"cached": true}', encoding="utf-8",
    )
    current = {
        "run_id": "current",
        "source_snapshot_hash": "a" * 64,
        "settings_hash": "b" * 64,
        "prompt_profile_hash": "c" * 64,
        "completion_model_id": "llm",
        "completion_model_digest": "d" * 64,
        "embedding_model_id": "embed",
        "embedding_model_digest": "e" * 64,
        "embedding_dimension": 1024,
        "graphrag_version": "3.1.1",
    }
    database.list_graphrag_native_runs = lambda _project_id: [{
        **current,
        "run_id": "failed",
        "state": "FAILED",
        "failure_class": "GraphRAGNativeBundleError",
        "execution_completed_at": "2026-08-21T00:00:00Z",
        "staging_relative_path": ".staging/failed-attempt",
    }]
    destination = tmp_path / "graphrag_native" / ".staging" / "current-attempt"
    destination.mkdir(parents=True)

    result = executor._seed_compatible_graph_cache(
        destination, project_id=7, current_run=current,
    )

    assert result and result["source_run_id"] == "failed"
    assert result["file_count"] == 1
    assert (destination / "cache" / "nested" / "entry.json").read_text(
        encoding="utf-8",
    ) == '{"cached": true}'
    assert (destination / "kf-manifest" / "cache-reuse.json").is_file()


def test_source_provenance_marker_in_graph_output_is_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path; (root / "output").mkdir()
    pq.write_table(pa.Table.from_pylist([{"title": "[KF_SOURCE_ID:x]"}]), root / "output" / "entities.parquet")
    with pytest.raises(RuntimeError, match="provenance metadata"):
        GraphRAGNativeExecutor._reject_source_metadata_pollution(root)


def test_nonempty_input_without_a_native_graph_never_becomes_ready(monkeypatch, tmp_path):
    executor, database = _executor(monkeypatch, tmp_path)
    monkeypatch.setattr(
        executor, "_require_nonempty_semantic_graph",
        lambda _root: (_ for _ in ()).throw(NativeNoGraphUnsupported("no graph")),
    )

    executor.run(job_id="job-1", project_id=7)

    run_updates = [value for name, value in database.calls if name == "run"]
    assert any(values.get("state") == "NO_GRAPH_UNSUPPORTED" for _run_id, values in run_updates)
    assert database.job["status"] == "failed"
    assert "activate" not in [name for name, _value in database.calls]


def test_empty_input_is_persisted_without_starting_graphrag_or_a_model(monkeypatch, tmp_path):
    import knowledge_forge.graphrag_native_executor as module

    executor, database = _executor(monkeypatch, tmp_path)

    class _EmptyInputBuilder(_InputBuilder):
        def preflight(self, *, project_id):
            return NativeInputPreflight(
                project_id=project_id, source_snapshot_hash="a" * 64,
                ready=False, reason="no renderable source blocks",
            )

    monkeypatch.setattr(module, "GraphRAGNativeInputBuilder", _EmptyInputBuilder)
    monkeypatch.setattr(
        executor.runner, "initialize",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("EMPTY_INPUT must not invoke graphrag init")),
    )
    monkeypatch.setattr(
        executor, "_provider_identity",
        lambda _project: (_ for _ in ()).throw(AssertionError("EMPTY_INPUT must not probe a model")),
    )

    executor.run(job_id="job-1", project_id=7)

    assert database.job["status"] == "failed"
    assert any(
        name == "run" and value[1].get("state") == "EMPTY_INPUT"
        for name, value in database.calls
    )
    assert "activate" not in [name for name, _value in database.calls]
