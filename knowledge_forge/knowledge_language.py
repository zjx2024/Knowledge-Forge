"""Project-level language policy for generated knowledge artifacts."""

from __future__ import annotations

import math
import re
from typing import Final


DOCUMENT_LANGUAGE: Final = "auto"
CHINESE: Final = "zh"
ENGLISH: Final = "en"

KNOWLEDGE_LANGUAGE_OPTIONS: Final[dict[str, str]] = {
    CHINESE: "中文",
    ENGLISH: "英文",
    DOCUMENT_LANGUAGE: "随文档语言",
}


def normalize_knowledge_language(
    value: object,
    *,
    default: str = DOCUMENT_LANGUAGE,
) -> str:
    """Validate one persisted/form language choice without silent fallback."""

    selected = str(value or default).strip().lower()
    aliases = {
        "auto": DOCUMENT_LANGUAGE,
        "document": DOCUMENT_LANGUAGE,
        "source": DOCUMENT_LANGUAGE,
        "zh": CHINESE,
        "zh-cn": CHINESE,
        "chinese": CHINESE,
        "中文": CHINESE,
        "en": ENGLISH,
        "en-us": ENGLISH,
        "english": ENGLISH,
        "英文": ENGLISH,
    }
    normalized = aliases.get(selected)
    if normalized not in KNOWLEDGE_LANGUAGE_OPTIONS:
        raise ValueError("请选择中文、英文或随文档语言")
    return normalized


def knowledge_language_label(value: object) -> str:
    normalized = normalize_knowledge_language(value)
    return KNOWLEDGE_LANGUAGE_OPTIONS[normalized]


def detect_document_language(texts: object) -> str:
    """Classify Chinese/English prose without treating identifiers as prose."""

    if isinstance(texts, str):
        values = [texts]
    else:
        try:
            values = [str(value or "") for value in texts]  # type: ignore[arg-type]
        except TypeError:
            values = [str(texts or "")]
    joined = "\n".join(values)
    han = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", joined))
    latin_words = len(re.findall(r"[A-Za-z]+", joined))
    latin_units = math.ceil(latin_words * 1.3)
    return CHINESE if han and han >= max(1, math.ceil(latin_units * 0.65)) else ENGLISH


def resolve_knowledge_language(value: object, texts: object) -> str:
    selected = normalize_knowledge_language(value)
    return detect_document_language(texts) if selected == DOCUMENT_LANGUAGE else selected


def user_facing_text_matches_language(value: object, language: object) -> bool:
    """Check only that prose contains the selected script, not its semantics."""

    text = str(value or "")
    resolved = normalize_knowledge_language(language)
    if resolved == DOCUMENT_LANGUAGE:
        resolved = detect_document_language(text)
    if resolved == CHINESE:
        return bool(re.search(r"[\u3400-\u9fff]", text))
    return bool(re.search(r"[A-Za-z]", text))


def generated_knowledge_instruction(value: object) -> str:
    """Return a reusable model instruction for natural-language fields.

    Entity names, identifiers, code and quoted evidence remain source-faithful;
    only model-generated explanatory text follows this policy.
    """

    normalized = normalize_knowledge_language(value)
    if normalized == CHINESE:
        target = "中文"
    elif normalized == ENGLISH:
        target = "English"
    else:
        return (
            "生成的自然语言说明应跟随当前输入文档的主要语言；实体专名、缩写、"
            "标识符、代码、公式和原文引文必须保留原始写法。"
        )
    return (
        f"生成的自然语言说明统一使用{target}；实体专名、缩写、标识符、代码、"
        "公式和原文引文必须保留原始写法，不得为了统一语言改写这些内容。"
    )
