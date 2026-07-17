"""Domain-neutral identity features shared by entity recall and resolution."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable


_ACRONYM_STOP_WORDS = {
    "a", "an", "and", "at", "by", "de", "for", "from", "in", "la", "of", "on", "or", "the", "to",
}


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _latin_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value or "").casefold()
    return re.findall(r"[a-z0-9]+", normalized)


def acronym(value: str) -> str:
    """Return a conservative Latin acronym, or an empty string when unavailable."""
    tokens = _latin_tokens(value)
    if len(tokens) == 1:
        token = tokens[0]
        return token if token.isalpha() and 2 <= len(token) <= 12 else ""
    significant = [token for token in tokens if token not in _ACRONYM_STOP_WORDS]
    if len(significant) < 2:
        return ""
    result = "".join(token[0] for token in significant if token)
    return result if 2 <= len(result) <= 12 else ""


def acronym_match(left: str, right: str) -> bool:
    left_norm, right_norm = normalize_name(left), normalize_name(right)
    if not left_norm or not right_norm:
        return False
    left_acronym, right_acronym = acronym(left), acronym(right)
    return bool(
        (left_acronym and left_acronym == right_norm)
        or (right_acronym and right_acronym == left_norm)
        or (left_acronym and right_acronym and left_acronym == right_acronym)
    )


def _ngrams(value: str, size: int = 2) -> set[str]:
    normalized = normalize_name(value)
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def name_pair_features(left: str, right: str) -> dict[str, float]:
    left_name, right_name = normalize_name(left), normalize_name(right)
    if not left_name or not right_name:
        return {"exact": 0.0, "containment": 0.0, "sequence": 0.0, "ngram": 0.0, "acronym": 0.0,
                "lexical": 0.0}
    exact = float(left_name == right_name)
    shorter, longer = sorted((left_name, right_name), key=len)
    containment = len(shorter) / len(longer) if shorter in longer else 0.0
    sequence = SequenceMatcher(None, left_name, right_name).ratio()
    ngram = _jaccard(_ngrams(left), _ngrams(right))
    acronym_score = float(acronym_match(left, right))
    lexical = max(exact, containment, sequence, ngram, 0.88 * acronym_score)
    return {
        "exact": exact,
        "containment": containment,
        "sequence": sequence,
        "ngram": ngram,
        "acronym": acronym_score,
        "lexical": min(1.0, lexical),
    }


def best_name_pair_features(name: str, candidate: dict[str, Any]) -> dict[str, float]:
    variants = [str(candidate.get("name") or ""), *(str(item) for item in candidate.get("aliases") or [])]
    results = [name_pair_features(name, variant) for variant in variants if variant]
    if not results:
        return name_pair_features(name, "")
    return {key: max(item[key] for item in results) for key in results[0]}


def names_are_compatible(left: str, right: str) -> bool:
    features = name_pair_features(left, right)
    return bool(
        features["exact"]
        or features["acronym"]
        or features["containment"] >= 0.55
        or features["sequence"] >= 0.82
        or features["ngram"] >= 0.72
    )


def credible_aliases(name: str, aliases: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(
        alias.strip() for alias in aliases if alias.strip() and names_are_compatible(name, alias)
    ))


def identifiers_conflict(left: str, right: str) -> bool:
    """Do not merge names carrying different explicit identifiers or versions."""
    pattern = r"\d+(?:[._-]\d+)*"
    left_ids = set(re.findall(pattern, unicodedata.normalize("NFKC", left or "")))
    right_ids = set(re.findall(pattern, unicodedata.normalize("NFKC", right or "")))
    return bool(left_ids and right_ids and left_ids != right_ids)


def types_are_compatible(left: str, right: str) -> bool:
    return left == right or "GenericEntity" in {left, right}


def entity_identity_text(
    name: str,
    entity_type: str,
    description: str = "",
    aliases: Iterable[str] = (),
) -> str:
    """Build context-free identity text so query and stored vectors are comparable."""
    alias_text = "、".join(dict.fromkeys(item.strip() for item in aliases if item.strip()))
    return (
        f"实体名称: {name}\n实体类型: {entity_type}\n"
        f"别名: {alias_text}\n实体描述: {description or ''}"
    ).strip()
