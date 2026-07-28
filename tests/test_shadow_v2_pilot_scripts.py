from __future__ import annotations

from scripts.run_shadow_v2_native_pilot import (
    _CountingGateway,
    _freeze_snapshot_manifest,
)


class _DummyGateway:
    @staticmethod
    def effective_model_context(_model: str):
        return None

    @staticmethod
    def json_chat(**_kwargs):
        return {}


def test_native_pilot_freeze_manifest_is_parameterized_by_frozen_documents():
    freeze = {
        "freeze_payload_sha256": "freeze-hash",
        "native_project": {
            "project": {"id": 7},
            "schemas": [
                {"version": 1, "status": "retired"},
                {"version": 4, "status": "active"},
            ],
            "documents": [
                {
                    "id": 10,
                    "active_revision_id": "rev-10",
                    "doc_hash": "doc-hash-10",
                },
                {
                    "id": 11,
                    "active_revision_id": "rev-11",
                    "doc_hash": "doc-hash-11",
                },
                {
                    "id": 12,
                    "active_revision_id": "rev-12",
                    "doc_hash": "doc-hash-12",
                },
            ],
            "chunks": [
                {"id": 102, "document_id": 10, "content_hash": "chunk-102"},
                {"id": 101, "document_id": 10, "content_hash": "chunk-101"},
                {"id": 201, "document_id": 11, "content_hash": "chunk-201"},
            ],
        },
    }

    manifest = _freeze_snapshot_manifest(freeze)

    assert manifest["project"] == {"project_id": 7, "schema_version": 4}
    assert [item["document_id"] for item in manifest["documents"]] == [10, 11, 12]
    assert [item["chunk_id"] for item in manifest["documents"][0]["chunks"]] == [
        101,
        102,
    ]
    assert manifest["documents"][2]["chunks"] == []


def test_native_pilot_counts_proposer_classifier_and_verifier_separately():
    gateway = _CountingGateway(_DummyGateway())

    gateway.json_chat(system="You are the Shadow relation v2 direct proposer.")
    gateway.json_chat(system="You are the Shadow relation v2 classifier.")
    gateway.json_chat(system="You are the independent Shadow relation v2 evidence verifier.")

    assert gateway.proposer_calls == 1
    assert gateway.classifier_calls == 1
    assert gateway.verifier_calls == 1
