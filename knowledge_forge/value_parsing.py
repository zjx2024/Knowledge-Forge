"""Tolerant conversion helpers for model-generated scalar values."""

from __future__ import annotations

import math
from typing import Any


_CONFIDENCE_LABELS = {
    "very high": 0.95,
    "very_high": 0.95,
    "high": 0.9,
    "medium": 0.6,
    "moderate": 0.6,
    "mid": 0.6,
    "low": 0.3,
    "very low": 0.1,
    "very_low": 0.1,
    "非常高": 0.95,
    "高": 0.9,
    "中": 0.6,
    "中等": 0.6,
    "低": 0.3,
    "非常低": 0.1,
}


def coerce_confidence(value: Any, default: float = 0.5) -> float:
    """Convert loose LLM confidence output to a finite value in ``[0, 1]``."""
    if value is None or value == "":
        return max(0.0, min(1.0, float(default)))

    text = str(value).strip().lower()
    if text in _CONFIDENCE_LABELS:
        return _CONFIDENCE_LABELS[text]

    percentage = text.endswith("%")
    if percentage:
        text = text[:-1].strip()
    try:
        number = float(text)
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))
    if not math.isfinite(number):
        return max(0.0, min(1.0, float(default)))
    if percentage or 1 < number <= 100:
        number /= 100
    return max(0.0, min(1.0, number))
