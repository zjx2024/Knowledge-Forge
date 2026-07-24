"""Stable identifiers and presentation metadata for knowledge construction strategies."""

from __future__ import annotations

from dataclasses import dataclass


STABLE_STRATEGY = "stable"
ENHANCED_SHADOW_STRATEGY = "enhanced_shadow"


@dataclass(frozen=True)
class ConstructionStrategy:
    key: str
    label: str
    description: str
    production_approved: bool


STRATEGIES = {
    STABLE_STRATEGY: ConstructionStrategy(
        key=STABLE_STRATEGY,
        label="稳定基线",
        description="当前正式实体关系抽取、复核与归一流程",
        production_approved=True,
    ),
    ENHANCED_SHADOW_STRATEGY: ConstructionStrategy(
        key=ENHANCED_SHADOW_STRATEGY,
        label="增强实验",
        description="基于文档预处理结果运行 P1/P2 隔离对比流程",
        production_approved=False,
    ),
}


def normalize_strategy(value: str | None) -> str:
    key = str(value or STABLE_STRATEGY).strip().lower()
    if key not in STRATEGIES:
        raise ValueError(f"未知知识构建方案: {value}")
    return key


def strategy_options() -> list[ConstructionStrategy]:
    return list(STRATEGIES.values())
