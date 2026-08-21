"""Stable identifiers and presentation metadata for knowledge construction strategies."""

from __future__ import annotations

from dataclasses import dataclass


STABLE_STRATEGY = "stable"
KNOWLEDGE_FORGE_STRATEGY = "knowledge_forge"
# This was the original persisted key while the strategy was still being
# prototyped.  It remains an input-only compatibility alias so old bookmarks,
# queued HTTP requests and historical API clients resolve to the same product.
LEGACY_ENHANCED_SHADOW_STRATEGY = "enhanced_shadow"
# Deprecated Python import compatibility.  New application code must use
# KNOWLEDGE_FORGE_STRATEGY; retaining this alias does not preserve the old
# strategy key in newly persisted records.
ENHANCED_SHADOW_STRATEGY = KNOWLEDGE_FORGE_STRATEGY
GRAPHRAG_NATIVE_STRATEGY = "graphrag_native"


@dataclass(frozen=True)
class ConstructionStrategy:
    key: str
    label: str
    description: str
    production_approved: bool
    build_scope: str = "document"
    user_selectable: bool = True


STRATEGIES = {
    STABLE_STRATEGY: ConstructionStrategy(
        key=STABLE_STRATEGY,
        label="稳定基线",
        description="当前正式实体关系抽取、复核与归一流程",
        production_approved=True,
        user_selectable=False,
    ),
    KNOWLEDGE_FORGE_STRATEGY: ConstructionStrategy(
        key=KNOWLEDGE_FORGE_STRATEGY,
        label="Knowledge Forge",
        description="基于冻结的文档证据执行实体归一、关系验证与可审计知识图谱构建",
        production_approved=False,
        build_scope="project",
    ),
    GRAPHRAG_NATIVE_STRATEGY: ConstructionStrategy(
        key=GRAPHRAG_NATIVE_STRATEGY,
        label="GraphRAG Native",
        description="从已准备的源文档构建实体、关系、主题社区、摘要与知识问答索引",
        production_approved=False,
        build_scope="project",
    ),
}


def normalize_strategy(value: str | None) -> str:
    key = str(value or STABLE_STRATEGY).strip().lower()
    key = {
        LEGACY_ENHANCED_SHADOW_STRATEGY: KNOWLEDGE_FORGE_STRATEGY,
    }.get(key, key)
    if key not in STRATEGIES:
        raise ValueError(f"未知知识构建方案: {value}")
    return key


def strategy_options(*, include_hidden: bool = True) -> list[ConstructionStrategy]:
    """Return strategy metadata, optionally omitting temporarily hidden UI choices.

    Hidden strategies remain valid internal strategy keys so existing snapshots,
    audit records and API compatibility are never altered merely by a product
    presentation decision.
    """

    return [
        item for item in STRATEGIES.values()
        if include_hidden or item.user_selectable
    ]
