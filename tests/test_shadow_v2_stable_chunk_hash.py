from __future__ import annotations

from copy import deepcopy

from knowledge_forge.shadow_context import build_shadow_context_manifest


def test_shadow_context_manifest_does_not_mutate_chunk_content_or_hash():
    document = {
        "id": 1,
        "active_revision_id": "rev-1",
        "doc_hash": "doc-hash",
    }
    chunks = [{
        "id": 10,
        "document_id": 1,
        "revision_id": "rev-1",
        "chunk_key": "chunk-10",
        "content": "Stable chunk text must not change.",
        "content_hash": "stable-hash",
    }]
    before = deepcopy(chunks)

    manifest = build_shadow_context_manifest(
        namespace_id="ns",
        documents=[document],
        chunks_by_document={1: chunks},
        payload_loader=lambda _document: {
            "blocks": [{
                "key": "block-1",
                "block_type": "paragraph",
                "text": "Stable chunk text must not change.",
                "order": 1,
            }],
        },
    )

    assert chunks == before
    assert chunks[0]["content_hash"] == "stable-hash"
    assert manifest.summary["context_units"] == 1
    assert manifest.artifact_hash
