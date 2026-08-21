from __future__ import annotations

import pytest

from knowledge_forge.knowledge_language import (
    detect_document_language,
    generated_knowledge_instruction,
    normalize_knowledge_language,
    resolve_knowledge_language,
    user_facing_text_matches_language,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("zh", "zh"), ("中文", "zh"), ("en", "en"), ("English", "en"),
     ("auto", "auto"), ("document", "auto")],
)
def test_knowledge_language_normalization(value, expected):
    assert normalize_knowledge_language(value) == expected


def test_knowledge_language_rejects_unknown_values():
    with pytest.raises(ValueError, match="中文、英文或随文档语言"):
        normalize_knowledge_language("fr")


def test_generated_language_policy_preserves_source_identifiers():
    chinese = generated_knowledge_instruction("zh")
    english = generated_knowledge_instruction("en")
    automatic = generated_knowledge_instruction("auto")

    assert "统一使用中文" in chinese
    assert "English" in english
    assert "保留原始写法" in chinese and "保留原始写法" in english
    assert "跟随当前输入文档" in automatic


def test_document_language_resolution_handles_technical_identifiers():
    assert detect_document_language(
        "中文技术正文包含 API Task_ID FC2 等英文标识。" * 30
    ) == "zh"
    assert detect_document_language(
        "English technical prose contains API and Task_ID identifiers. " * 30
    ) == "en"
    assert resolve_knowledge_language("auto", "这是正文。" * 20) == "zh"
    assert resolve_knowledge_language("en", "这是正文。" * 20) == "en"
    assert user_facing_text_matches_language("中文说明 API", "zh")
    assert user_facing_text_matches_language("English explanation 中文专名", "en")
