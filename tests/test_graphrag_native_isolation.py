"""Native read path is file-only and remains usable with legacy graph disabled."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from knowledge_forge.graphrag_native_runtime import GraphRAGNativeRuntime, GraphRAGNativeStale


class _Database:
    def __init__(self, run_id: str, bundle_relative_path: str):
        self.run = {"run_id": run_id, "state": "READY", "bundle_relative_path": bundle_relative_path, "generation": 1, "graphrag_version": "3.1.1", "source_snapshot_hash": "snapshot"}
        self.active_revision = "revision-1"
        self.desired_snapshot = "snapshot"
    def get_active_graphrag_native_run(self, _project_id): return dict(self.run)
    def get_graphrag_native_project_state(self, _project_id): return {"project_id": 1, "active_run_id": self.run["run_id"], "desired_source_snapshot_hash": self.desired_snapshot}
    def list_graphrag_native_run_documents(self, _run_id): return [{"document_id": 1, "revision_id": "revision-1", "source_hash": "source"}]
    def list_documents(self, _project_id):
        return [{
            "id": 1, "active_revision_id": self.active_revision,
            "original_filename": "source.pdf",
            "storage_path": "http://files.test/source.pdf",
            "mime_type": "application/pdf",
        }]
    def get_document_revision(self, revision_id): return {"revision_id": revision_id, "source_hash": "source"}


def _runtime(monkeypatch, tmp_path):
    from knowledge_forge.config import Config
    from tests.test_graphrag_native_artifacts import _valid_bundle

    root = tmp_path / "graphrag_native" / "project-1" / "runs" / "fake"
    root.mkdir(parents=True)
    _valid_bundle(root)
    monkeypatch.setattr(Config, "files_root", tmp_path)
    db = _Database("fake", "project-1/runs/fake")
    # Legacy dependencies deliberately explode if a native read accidentally
    # tries to use them.
    service = SimpleNamespace(
        db=db, graph=_Tripwire(), vectors=_Tripwire(), resolver=_Tripwire(),
        document_structure=lambda _document_id: {
            "blocks": [{
                "key": "b1", "block_type": "image", "page": 3,
                "section_title": "插图章节",
                "metadata": {
                    "url": "http://files.test/figure.png",
                    "width": 640, "height": 480,
                    "title": "示意图", "description": "图示说明",
                },
            }],
        },
    )
    return GraphRAGNativeRuntime(service), db


class _Tripwire:
    def __getattr__(self, name): raise AssertionError(f"legacy dependency invoked: {name}")


def test_native_graph_reads_only_current_validated_bundle(monkeypatch, tmp_path):
    runtime, _db = _runtime(monkeypatch, tmp_path)

    graph = runtime.graph(1, limit=20)

    assert graph["run_id"] == "fake"
    assert [(edge["source"], edge["description"], edge["target"]) for edge in graph["edges"]] == [("A", "related", "B")]


def test_native_explorer_projects_reports_and_text_unit_provenance(monkeypatch, tmp_path):
    runtime, _db = _runtime(monkeypatch, tmp_path)

    payload = runtime.explorer_payload(1)

    assert payload["community_report_status"] == "completed"
    assert payload["edges"][0]["evidence_text"] == "input"
    assert payload["edges"][0]["evidence_count"] == 1
    assert payload["edges"][0]["source_info"]["filename"] == "source.pdf"
    assert payload["edges"][0]["source_info"]["page_label"] == "第 3 页"
    assert payload["edges"][0]["source_info"]["media_assets"][0]["title"] == "示意图"
    group = payload["groups"][0]
    assert group["description"] == "summary"
    assert group["report_status"] == "succeeded"
    assert group["community_report"]["title"] == "community"
    assert group["community_report"]["content"] == "report"


def test_native_current_pointer_fails_closed_when_revision_becomes_stale(monkeypatch, tmp_path):
    runtime, db = _runtime(monkeypatch, tmp_path)
    db.active_revision = "revision-new"

    with pytest.raises(GraphRAGNativeStale, match="revision"):
        runtime.graph(1)


def test_native_current_pointer_fails_closed_when_new_generation_changes_desired_snapshot(monkeypatch, tmp_path):
    runtime, db = _runtime(monkeypatch, tmp_path)
    db.desired_snapshot = "new-snapshot"

    with pytest.raises(GraphRAGNativeStale, match="后续 run"):
        runtime.graph(1)
