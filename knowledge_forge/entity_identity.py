"""Domain-neutral identity features shared by entity recall and resolution."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable


_ACRONYM_STOP_WORDS = {
    "a", "an", "and", "at", "by", "de", "for", "from", "in", "la", "of", "on", "or", "the", "to",
}

# Ordinary entity identity intentionally ignores punctuation (``U-Net`` and
# ``U Net`` are the same useful lexical form). Mathematical operators are
# different: deleting them makes unrelated expressions such as ``A ∩ B`` and
# ``‖a-b‖`` share the key ``ab``, or makes ``c = 1`` collide with the client
# identifier ``C1``. Preserve only an operator signature and only when an
# unmistakable mathematical symbol is present; prose/name normalization keeps
# its historical behavior.
_MATHEMATICAL_OPERATOR_CODES = {
    "=": "eq",
    "+": "plus",
    "-": "minus",
    "−": "minus",
    "×": "times",
    "*": "times",
    "÷": "divide",
    "/": "divide",
    "%": "percent",
    "~": "range",
    "∼": "range",
    "∩": "intersection",
    "∪": "union",
    "∑": "sum",
    "∏": "product",
    "√": "root",
    "∥": "norm",
    "‖": "norm",
    "≤": "le",
    "≥": "ge",
    "<": "lt",
    ">": "gt",
    "≈": "approx",
    "≠": "neq",
    "∈": "in",
    "∉": "notin",
    "⊂": "subset",
    "⊆": "subseteq",
}
_UNAMBIGUOUS_MATHEMATICAL_SYMBOL = re.compile(
    r"[=×÷∩∪∑∏√∥‖≤≥≈≠∈∉⊂⊆]"
)


def _symbolic_operator_signature(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value or "")
    if not _UNAMBIGUOUS_MATHEMATICAL_SYMBOL.search(normalized):
        return ()
    return tuple(
        code
        for character in normalized
        if (code := _MATHEMATICAL_OPERATOR_CODES.get(character))
    )


def _unicode_identity_lexical(value: str) -> str:
    """Return the script-preserving lexical portion of an identity key.

    Entity names are multilingual identifiers, not ASCII identifiers.  The
    previous allow-list retained Latin, digits and CJK ideographs only, which
    silently erased Greek, Cyrillic, Arabic, Kana and Hangul letters.  That
    made distinct formula symbols such as ``θ j`` and ``j`` exact aliases and
    could collapse ordinary non-Latin names as well.  NFKC plus case-folding
    still unifies compatibility/case variants; Unicode letters and numbers
    themselves must remain part of the identity.
    """

    normalized = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return "".join(character for character in normalized if character.isalnum())


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().strip()
    lexical = _unicode_identity_lexical(value)
    signature = _symbolic_operator_signature(value)
    if signature:
        return lexical + "__math_" + "_".join(signature)
    return lexical


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
    left_symbolic = _symbolic_operator_signature(left)
    right_symbolic = _symbolic_operator_signature(right)
    if left_symbolic != right_symbolic and (left_symbolic or right_symbolic):
        left_lexical = _unicode_identity_lexical(left)
        right_lexical = _unicode_identity_lexical(right)
        if left_lexical and left_lexical == right_lexical:
            return True
    pattern = r"\d+(?:[._-]\d+)*"
    left_value = unicodedata.normalize("NFKC", left or "")
    right_value = unicodedata.normalize("NFKC", right or "")
    left_ids = set(re.findall(pattern, left_value))
    right_ids = set(re.findall(pattern, right_value))
    if left_ids and right_ids:
        return left_ids != right_ids
    if bool(left_ids) == bool(right_ids):
        return False
    # A bare family name and the same family plus an explicit numeric suffix
    # are not automatically the same identity (``MIPS`` versus ``mips64``,
    # ``RFC`` versus ``RFC 9110``).  Treat only an exact identifier-stripped
    # base as a conflict; unrelated names that merely contain a number remain
    # available to the ordinary bounded resolver.
    left_base = normalize_name(re.sub(pattern, "", left_value))
    right_base = normalize_name(re.sub(pattern, "", right_value))
    return bool(left_base and left_base == right_base)


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
