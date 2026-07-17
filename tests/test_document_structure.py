import json
from pathlib import Path

import pytest

from knowledge_forge.config import Config
from knowledge_forge.service import KnowledgeForgeService


class StructureDatabase:
    def __init__(self, document):
        self.document = document

    def get_document(self, document_id):
        return self.document if document_id == self.document["id"] else None

    def list_document_chunks(self, document_id):
        return [{
            "id": 1,
            "chunk_key": "s0001_p001",
            "section_title": "第一章",
            "content": "知识块正文",
            "metadata": {"chunk_role": "child", "token_count": 12},
        }] if document_id == self.document["id"] else []


def service_with_document(document):
    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = StructureDatabase(document)
    return service


def test_document_structure_reads_persisted_blocks_without_reparsing(tmp_path, monkeypatch):
    chunks_root = tmp_path / "chunks"
    monkeypatch.setattr(Config, "chunks_root", chunks_root)
    document = {
        "id": 8,
        "project_id": 3,
        "project_name": "测试项目",
        "doc_hash": "a" * 64,
        "original_filename": "manual.pdf",
        "status": "indexed",
        "metadata": {},
    }
    payload_path = chunks_root / Config.project_storage_key(3, "测试项目") / f"{'a' * 16}.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(json.dumps({
        "plan": {"effective_parser": "plaintext_pdf", "effective_strategy": "general"},
        "quality": {"block_count": 2},
        "blocks": [
            {"order": 1, "block_type": "heading", "text": "标题"},
            {"order": 2, "block_type": "paragraph", "text": "正文"},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    result = service_with_document(document).document_structure(8)

    assert result["document_name"] == "manual.pdf"
    assert result["block_count"] == 2
    assert result["block_types"] == ["heading", "paragraph"]
    assert result["blocks"][1]["text"] == "正文"


def test_document_structure_reports_when_processing_payload_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "chunks_root", tmp_path / "chunks")
    document = {
        "id": 9,
        "project_id": 3,
        "project_name": "测试项目",
        "doc_hash": "b" * 64,
        "original_filename": "manual.pdf",
        "status": "processing",
        "metadata": {},
    }

    with pytest.raises(LookupError, match="尚未生成"):
        service_with_document(document).document_structure(9)


def test_document_chunks_are_available_after_stage_two():
    document = {
        "id": 10,
        "project_id": 3,
        "project_name": "测试项目",
        "doc_hash": "c" * 64,
        "original_filename": "manual.pdf",
        "status": "processing",
        "chunks_ready_at": "2026-07-15T10:00:00+00:00",
        "metadata": {},
    }

    result = service_with_document(document).document_chunks(10)

    assert result["chunk_count"] == 1
    assert result["chunks"][0]["chunk_key"] == "s0001_p001"


def test_document_chunks_are_locked_before_stage_two():
    document = {
        "id": 11,
        "project_id": 3,
        "project_name": "测试项目",
        "doc_hash": "d" * 64,
        "original_filename": "manual.pdf",
        "status": "processing",
        "chunks_ready_at": None,
        "metadata": {},
    }

    with pytest.raises(LookupError, match="第二阶段"):
        service_with_document(document).document_chunks(11)
