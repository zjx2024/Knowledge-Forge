"""Span-first entity normalization for isolated P2 treatment runs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .entity_publishability import (
    REJECTED as PUBLISHABILITY_REJECTED,
    classify_mention_publishability,
)
from .entity_identity import normalize_name
from .sentence_spans import is_soft_wrap_newline
from .value_parsing import coerce_confidence


_PURE_NUMBER = re.compile(r"^[\W_]*\d+(?:[.\-]\d+)*[\W_]*$")
_YEAR_OR_ESTIMATE = re.compile(r"^(?:18|19|20)\d{2}e?$", re.IGNORECASE)
_DATE_EXPRESSION = re.compile(
    r"^(?:(?:18|19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?"
    r"|(?:18|19|20)\d{2}[-/.]\d{1,2}(?:[-/.]\d{1,2})?)$",
    re.IGNORECASE,
)
_PERCENTAGE = re.compile(r"^[+-]?\d+(?:\.\d+)?\s*%$")
_PHONE = re.compile(r"^\+?\d[\d\s().-]{6,}\d$")
_AMOUNT_OR_UNIT = re.compile(
    r"^[+-]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|tons?|tonnes?|metric\s+tons?|mt|kg|g|lbs?|usd|us\$|\\$|"
    r"million|billion|thousand|万吨|吨|千克|美元|亿元?)?$",
    re.IGNORECASE,
)
_NOISE_ENTITY_TYPES = frozenset({
    "Date", "Year", "Time", "Number", "Quantity", "Amount", "Money",
    "Currency", "Percentage", "Percent", "Phone", "PhoneNumber", "Unit",
})
_NON_REFERENTIAL_MODIFIERS = frozenset({
    "domestic", "foreign", "estimated", "estimate", "free", "total", "subtotal",
})
_STRUCTURAL_ROLES = frozenset({
    "table_of_contents", "header", "footer", "page_number", "index_entry",
    "empty_cell", "reference_heading", "table_header", "unit_header",
    "heading", "title", "page_header", "page_footer", "table_container",
    "caption", "footnote",
})
_STRUCTURED_RELATION_LINE = re.compile(
    r"^\s*(?:[-*+]\s+)?(?P<left>.+?)\s+--(?P<label>[^\n<>]+?)-->\s+"
    r"(?P<right>.+?)\s*$"
)
_STRUCTURED_INLINE_LABEL = re.compile(
    r"^[ \t]*(?P<label>[A-Za-z]{1,12}[-_]\d{1,8}|"
    r"(?:§[ \t]*)?\d+(?:\.\d+){1,4})[ \t]+(?=\S)"
)
_RELATION_SENTENCE_BOUNDARY = re.compile(r"[。！？!?；;.\n]")
_RELATION_CONNECTOR = re.compile(
    r"(?:并且|以及|并|和|与|或|向|对|为|是|则)$"
)
_ENTITY_TERMINAL_PUNCTUATION = " \t\r\n.,;:!?，。；：！？、)]}》〉』」』”’"
_GENERIC_NON_REFERENTIAL_SINGLE_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "for", "to", "of", "in",
    "on", "at", "by", "from", "with", "as", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "they", "them", "their", "he", "she", "her", "his", "we", "you", "your",
    "i", "me", "my", "our", "which", "who", "whom", "when", "where", "while",
    "then", "than", "not", "no", "yes", "try", "except", "finally", "else",
    "elif", "return", "shall", "must", "may", "can", "will", "would", "should",
    "do", "does", "did", "has", "have", "had", "use", "used", "contains",
    "includes", "provides", "supports", "sends", "looks", "took", "out", "down",
    "after", "before", "one", "more", "all", "each", "every",
    # Closed-class Chinese personal pronouns are discourse references, not
    # canonical entities.  Relation extraction has an independent bounded
    # coreference pass that may bind their exact spans to a frozen antecedent;
    # keeping them here would instead publish graph nodes literally named
    # “他” or “她”.
    "我", "我们", "你", "你们", "您", "他", "他们", "她", "她们", "它", "它们",
    # Source-author self references are discourse anchors, not stable graph
    # identities. Their assertions remain available from the other endpoint
    # and the exact source evidence.
    "本文", "本研究", "本工作", "本论文", "本报告", "本节", "本章",
})
_GENERIC_FRAGMENT_STARTERS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "when", "while",
    "if", "for",
})
_GENERIC_FRAGMENT_VERBS = frozenset({
    "contains", "include", "includes", "is", "are", "was", "were", "has",
    "have", "had", "forms", "provides", "supports", "uses", "used", "iterates",
    "sends", "makes", "requires", "shall", "must", "can", "may", "will",
})
_ASCII_WORD_CUE = re.compile(r"[A-Za-z0-9_]")
_LEADING_DETERMINER = re.compile(
    r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE,
)
_COORDINATION_SPLIT = re.compile(
    r"\s*(?:,|，|、|\band\b|\bor\b|\bas well as\b|以及|并且|并|和|与|或|及)\s*",
    re.IGNORECASE,
)
_COORDINATION_ITEM_SEPARATOR = re.compile(
    # Single-character ``与``/``及`` are frequently suffixes of ordinary CJK
    # verbs (参与、涉及).  Keep the structural repair deliberately narrower;
    # punctuation, unambiguous multi-character connectors and ``和`` cover the
    # common repeated-sibling form without treating those verb tails as list
    # boundaries.
    r"(?:、|，|,|以及|并且|和|\b(?:as\s+well\s+as|and|or)\b)",
    re.IGNORECASE,
)
_COORDINATION_SCOPE_BOUNDARY = re.compile(r"[。！？!?；;\r\n]")
_COMPACT_ENTITY_ITEM = re.compile(
    r"[\w'’\-\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE,
)
_SOFT_WRAP_ENTITY_ITEM = re.compile(
    r"[\w'’/\\\-&+()（）\[\]\u3400-\u4dbf\u4e00-\u9fff\s]+",
    re.UNICODE,
)
_TAIL_PREPOSITION = re.compile(
    r"\b(?:for|of|to|from|within|inside|into|as|with)\s+",
    re.IGNORECASE,
)


def _is_cjk_layout_character(value: str) -> bool:
    """Return whether ``value`` belongs to a script normally written unspaced."""

    if not value:
        return False
    codepoint = ord(value)
    return bool(
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x30FF
    )


def _layout_surface_parts(value: str) -> tuple[list[str], list[str]]:
    """Split a source surface into visible characters and inner whitespace."""

    characters: list[str] = []
    separators: list[str] = []
    pending = ""
    for character in str(value or ""):
        if character.isspace():
            pending += character
            continue
        if characters:
            separators.append(pending)
        characters.append(character)
        pending = ""
    return characters, separators


def layout_equivalent_surface_occurrences(
    content: str,
    surface: str,
    *,
    max_inner_whitespace: int = 16,
) -> list[tuple[int, int]]:
    """Find exact-name repeats whose only difference is renderer whitespace.

    This is deliberately narrower than ordinary normalization.  Every visible
    source character must be identical and in the same order; only whitespace
    between CJK/Kana characters or a parser-confirmed soft newline may differ.
    Pure Latin word spaces therefore keep their lexical meaning (``New York``
    is not silently equated with ``NewYork``), while PDF forms such as
    ``联邦学习``/``联邦 学习`` and ``federated\nlearning`` remain recoverable.
    The returned spans always refer to the immutable source text.
    """

    source = str(content or "")
    seed = str(surface or "")
    characters, seed_separators = _layout_surface_parts(seed)
    if len(characters) < 2 or not source:
        return []

    ascii_word = all(
        character.isascii() and (character.isalnum() or character == "_")
        for character in characters
    )
    results: list[tuple[int, int]] = []
    search_from = 0
    while True:
        left = source.find(characters[0], search_from)
        if left < 0:
            break
        search_from = left + 1
        cursor = left + 1
        candidate_separators: list[str] = []
        valid = True
        for character in characters[1:]:
            separator_start = cursor
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
            candidate_separators.append(source[separator_start:cursor])
            if cursor >= len(source) or source[cursor] != character:
                valid = False
                break
            cursor += 1
        if not valid:
            continue
        right = cursor
        candidate_surface = source[left:right]
        if candidate_surface == seed:
            continue
        if sum(len(value) for value in candidate_separators) > max_inner_whitespace:
            continue

        changed = False
        for index, (expected, observed) in enumerate(zip(
            seed_separators, candidate_separators,
        )):
            if expected == observed:
                continue
            changed = True
            adjacent_cjk = (
                _is_cjk_layout_character(characters[index])
                or _is_cjk_layout_character(characters[index + 1])
            )
            newline_offsets = [
                left
                + 1
                + sum(
                    1 + len(value)
                    for value in candidate_separators[:index]
                )
                + offset
                for offset, value in enumerate(observed)
                if value == "\n"
            ]
            soft_newline = bool(newline_offsets) and all(
                is_soft_wrap_newline(source, offset)
                for offset in newline_offsets
            )
            if newline_offsets and not soft_newline:
                valid = False
                break
            # A newline present in the seed was already accepted as part of an
            # exact model span.  It may safely map to the same visible sequence
            # elsewhere; newly crossed source newlines must pass the parser's
            # conservative soft-wrap test.
            if not (
                adjacent_cjk
                or soft_newline
                or "\n" in expected
                or "\r" in expected
            ):
                valid = False
                break
            if "\n\n" in observed or "\r\n\r\n" in observed:
                valid = False
                break
        if not valid or not changed:
            continue
        if ascii_word:
            before = source[left - 1] if left else ""
            after = source[right] if right < len(source) else ""
            if (
                (before and (before.isalnum() or before == "_"))
                or (after and (after.isalnum() or after == "_"))
            ):
                continue
        results.append((left, right))
    return results


_RELATIVE_CLAUSE = re.compile(r"\s+(?:that|which|who|where)\b", re.IGNORECASE)
LAYOUT_MENTION_COVERAGE_POLICY_VERSION = "entity-layout-mention-coverage-v2"


def layout_recovery_seed_rejection_reason(
    surface: str, entity_type: str,
) -> str:
    """Reject only short, low-information seeds from automatic expansion."""

    visible = "".join(
        character for character in str(surface or "")
        if not character.isspace()
    )
    lexical_length = sum(character.isalnum() for character in visible)
    if lexical_length >= 4:
        return ""
    kind = str(entity_type or "GenericEntity")
    if lexical_length >= 2 and kind in {"Person", "Organization", "Location"}:
        return ""
    if lexical_length >= 2 and any(
        character.isdigit() or (
            character.isascii() and character.isupper()
        )
        for character in visible
    ):
        return ""
    if (
        lexical_length >= 3
        and visible.isascii()
        and kind in {"Organization", "Product", "Technology"}
    ):
        return ""
    return "short_low_information_layout_seed"


def non_referential_surface_reason(surface: str) -> str:
    """Classify only syntactically context-dependent placeholder surfaces.

    ``standalone_referential`` is a useful model hint, but local models often
    set it to false for ordinary common nouns such as ``client`` or ``service``.
    The deterministic gate therefore re-checks only language-independent
    surface shape: function/pronoun tokens and deictic phrases whose identity
    cannot be recovered without an antecedent.  A lexical noun phrase remains
    source-backed even when the model's hint is over-conservative.
    """

    lexical = re.sub(
        r"[^\w'’\-\u3400-\u4dbf\u4e00-\u9fff]+", " ",
        str(surface or ""), flags=re.UNICODE,
    ).strip().casefold()
    if not lexical:
        return "empty_normalized_surface"
    words = lexical.split()
    if len(words) == 1 and words[0] in _GENERIC_NON_REFERENTIAL_SINGLE_WORDS:
        return "non_referential_function_word"
    english_deictic = {
        "the following", "the above", "the preceding", "the previous",
        "the next", "the latter", "the former", "this", "that", "these",
        "those", "such", "its", "their", "his", "her",
        "this paper", "this study", "this work", "this article",
        "this report", "this section", "this chapter",
    }
    if lexical in english_deictic or any(
        lexical.startswith(prefix + " ")
        for prefix in (
            "the following", "the above", "the preceding", "the previous",
            "the next", "the latter", "the former",
        )
    ):
        return "non_referential_context_placeholder"
    if re.match(
        r"^(?:该|此|这|那|其|上述|以上|以下|前述|前者|后者)", lexical,
    ):
        return "non_referential_context_placeholder"
    if re.fullmatch(
        r"(?:各|两|多|若干)[\u3400-\u9fffA-Za-z0-9_]{1,16}(?:之)?间",
        lexical,
    ):
        return "non_referential_context_placeholder"
    if re.fullmatch(
        r"(?:等|等等)(?:功能|内容|事项|对象|项目|情况|部分|方面|信息|数据|"
        r"服务|流程|过程|任务|活动)?",
        lexical,
    ):
        return "non_referential_enumeration_tail"
    return ""


def normalize_entity_span(content: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Repair harmless model punctuation drift while preserving source offsets.

    Local models frequently include sentence-final punctuation in an entity
    surface.  The source-backed entity is the exact lexical span before that
    punctuation.  This helper only trims
    terminal punctuation when the supplied offsets exactly cover the source;
    it never searches for or invents a different occurrence.
    """

    value = dict(raw)
    surface = str(value.get("surface") or value.get("name") or "")
    try:
        start, end = int(value.get("mention_start")), int(value.get("mention_end"))
    except (TypeError, ValueError):
        return value
    if not (0 <= start < end <= len(content)) or content[start:end] != surface:
        return value
    trimmed_end = end
    while trimmed_end > start and content[trimmed_end - 1] in _ENTITY_TERMINAL_PUNCTUATION:
        closing = content[trimmed_end - 1]
        opening = {")": "(", "]": "[", "}": "{"}.get(closing)
        if opening and content[start:trimmed_end].count(opening) >= content[
            start:trimmed_end
        ].count(closing):
            # Balanced delimiters are part of callable/indexed identifiers
            # such as ``start()``; only unmatched sentence punctuation is
            # removable boundary drift.
            break
        trimmed_end -= 1
    if trimmed_end == end or trimmed_end <= start:
        return value
    trimmed_surface = content[start:trimmed_end]
    if not trimmed_surface.strip():
        return value
    if "surface" in value:
        value["surface"] = trimmed_surface
    if "name" in value:
        value["name"] = trimmed_surface
    value["mention_end"] = trimmed_end
    return value


def _exact_raw_span(content: str, raw: dict[str, Any]) -> tuple[int, int] | None:
    """Return the model span only when it is an exact source slice."""

    surface = str(raw.get("surface") or raw.get("name") or "")
    try:
        start = int(raw.get("mention_start", raw.get("evidence_start")))
        end = int(raw.get("mention_end", raw.get("evidence_end")))
    except (TypeError, ValueError):
        return None
    if not surface or not (0 <= start < end <= len(content)):
        return None
    return (start, end) if content[start:end] == surface else None


def repair_soft_wrapped_entity_span(
    content: str, raw: dict[str, Any],
) -> dict[str, Any]:
    """Restore an exact source surface when a model collapses a PDF wrap.

    The grounded mention model has already identified the complete noun
    phrase and supplied immutable offsets.  Whitespace normalization performed
    by JSON/model layers must not turn that otherwise exact occurrence into an
    invalid span.  Only parser-confirmed soft newlines are repaired; tables,
    lists, blank-line blocks and arbitrary multiline clauses remain rejected.
    """

    value = dict(raw)
    if value.get("grounded_semantic_mention") is not True:
        return value
    try:
        start = int(value.get("mention_start", value.get("evidence_start")))
        end = int(value.get("mention_end", value.get("evidence_end")))
    except (TypeError, ValueError):
        return value
    if not (0 <= start < end <= len(content)):
        return value
    source_surface = content[start:end]
    if "\n" not in source_surface and "\r" not in source_surface:
        return value
    requested = str(value.get("surface") or value.get("name") or "")
    if (
        not requested
        or re.sub(r"\s+", " ", requested).strip()
        != re.sub(r"\s+", " ", source_surface).strip()
    ):
        return value
    newline_offsets = [
        index for index in range(start, end) if content[index] == "\n"
    ]
    if not newline_offsets or not all(
        is_soft_wrap_newline(content, index) for index in newline_offsets
    ):
        return value
    if "surface" in value:
        value["surface"] = source_surface
    value["name"] = source_surface
    value["soft_wrap_surface_repaired"] = True
    return value


def entity_boundary_reason(
    content: str,
    raw: dict[str, Any],
    peers: Iterable[dict[str, Any]] = (),
) -> str:
    """Reject generic extraction fragments that are not standalone entities.

    LLM entity passes occasionally return a clause fragment or a qualified
    phrase containing the actual entity mentions (for example a whole
    subject/object phrase).  Such rows have exact offsets, so ordinary span
    validation alone cannot distinguish them.  This gate is deliberately
    predicate- and domain-independent: it uses only exact nesting, the
    model's proposition marker, and Unicode boundary structure.  Complete
    proposition arguments remain allowed when explicitly marked as such.
    """

    span = _exact_raw_span(content, raw)
    if span is None:
        return ""
    start, end = span
    surface = content[start:end]
    # If a shorter name also occurs as a complete coordinated item, a model
    # can incorrectly ground that name inside a distinct compound sibling
    # (for example the short final item inside a longer middle item).  The
    # syntax-only recovery below marks the complete sibling.  In that narrow
    # case the nested row is a span-boundary error regardless of its model
    # type; the independently occurring short item remains available.
    for peer in peers:
        if peer is raw or not isinstance(peer, dict):
            continue
        if peer.get("coordinated_distinct_candidate") is not True:
            continue
        peer_span = _exact_raw_span(content, peer)
        if peer_span is None:
            continue
        peer_start, peer_end = peer_span
        if peer_start <= start < end <= peer_end and peer_span != span:
            return "nested_coordinated_item_fragment"
    entity_type = str(raw.get("type") or "GenericEntity")
    lexical = re.sub(r"[^\w'’-]+", " ", surface, flags=re.UNICODE).strip().casefold()
    words = lexical.split()
    # Type hallucinations must not turn a pronoun or grammar/function token
    # into a canonical entity.  This check is intentionally limited to exact
    # lowercase standalone tokens, so named products, acronyms and full
    # technical phrases remain eligible regardless of their asserted type.
    if len(words) == 1 and words[0] in _GENERIC_NON_REFERENTIAL_SINGLE_WORDS:
        return "non_referential_function_word"
    non_referential_reason = non_referential_surface_reason(surface)
    if non_referential_reason:
        return non_referential_reason
    after = content[end] if end < len(content) else ""
    if (
        after and after.isascii()
        and (after.isalnum() or after == "_")
        and surface[-1:].isascii() and surface[-1:].isalnum()
    ):
        return "nonterminal_ascii_surface"
    if entity_type != "GenericEntity" or raw.get("proposition_argument") is True:
        return ""
    endpoint_candidate = raw.get("endpoint_candidate") is True
    if endpoint_candidate and raw.get("grounded_semantic_mention") is not True:
        for peer in peers:
            if (
                peer is raw
                or not isinstance(peer, dict)
                or peer.get("grounded_semantic_mention") is not True
            ):
                continue
            peer_span = _exact_raw_span(content, peer)
            if peer_span is None:
                continue
            peer_start, peer_end = peer_span
            if (
                peer_start <= start < end <= peer_end
                and (peer_start, peer_end) != (start, end)
            ):
                return "nested_cue_fragment_of_grounded_mention"
    # Cue-derived endpoint candidates are exact structural projections.  They
    # may be nested inside a broad model-generated clause, so that broad-clause
    # gate must not discard them.  If one cue candidate contains another cue
    # candidate, keep the smaller endpoint projection and reject the outer one.
    if endpoint_candidate:
        # A model may return an entire table row as one endpoint even though
        # parser-owned projections already expose the marker-free cells.
        # Delimiters are source structure, not part of an entity identity.
        soft_wrapped_grounded = bool(
            raw.get("grounded_semantic_mention") is True
            and raw.get("soft_wrap_surface_repaired") is True
        )
        if (
            any(value in surface for value in ("|", "\t"))
            or (
                any(value in surface for value in ("\n", "\r"))
                and not soft_wrapped_grounded
            )
        ):
            return "composite_structured_endpoint"
        if raw.get("coordinated_distinct_candidate") is True:
            return ""
        if len(surface.strip()) == 1:
            before_match = re.search(r"\S\s*$", content[:start])
            after_match = re.match(r"\s*\S", content[end:])
            before = before_match.group(0).strip() if before_match else ""
            after = after_match.group(0).strip() if after_match else ""
            if any("\u3400" <= value <= "\u9fff" for value in (before, after)):
                return "partial_context_fragment"
        for peer in peers:
            if peer is raw or not isinstance(peer, dict):
                continue
            peer_span = _exact_raw_span(content, peer)
            if peer_span is None or peer_span == span:
                continue
            peer_start, peer_end = peer_span
            if not (start <= peer_start < peer_end <= end):
                continue
            peer_type = str(peer.get("type") or "GenericEntity")
            if peer_type != "GenericEntity" or peer.get("proposition_argument") is True:
                return "nested_non_atomic_surface"
            if peer.get("endpoint_candidate") is True:
                # A cue can also match the final character of a compound
                # noun (for example a one-character residual after a longer
                # endpoint).  That residual is rejected independently by the
                # partial-context rule and must not cause the complete
                # endpoint that contains it to be rejected as well.
                peer_surface = content[peer_start:peer_end].strip()
                if len(peer_surface) == 1:
                    peer_before = content[peer_start - 1] if peer_start else ""
                    peer_after = content[peer_end] if peer_end < len(content) else ""
                    if any(
                        "\u3400" <= value <= "\u9fff"
                        for value in (peer_before, peer_after)
                    ):
                        continue
                # The grounded mention inventory is allowed to retain nested
                # identity-bearing noun phrases (for example a named region
                # and the complete data-set phrase containing that region).
                # Legacy cue projections still keep the smaller endpoint, and
                # clause-like grounded spans remain rejected below.
                if raw.get("grounded_semantic_mention") is True:
                    clause_tokens = set(words) & _GENERIC_FRAGMENT_VERBS
                    if not clause_tokens and not re.search(
                        r"[.!?。！？；;]", surface,
                    ):
                        continue
                return "endpoint_contains_endpoint"
        return ""
    for peer in peers:
        if peer is raw or not isinstance(peer, dict):
            continue
        peer_span = _exact_raw_span(content, peer)
        if peer_span is None or peer_span == span:
            continue
        peer_start, peer_end = peer_span
        if not (start <= peer_start < peer_end <= end):
            continue
        if peer_span == span:
            continue
        peer_surface = content[peer_start:peer_end]
        # A generic outer span that contains a typed/named inner span is a
        # qualified clause, not a second canonical entity.  Keep explicitly
        # declared complete proposition arguments above.
        peer_type = str(peer.get("type") or "GenericEntity")
        if peer_type != "GenericEntity" or len(surface) - len(peer_surface) >= 2:
            return "nested_non_atomic_surface"
    # A one-character generic row adjacent to Chinese text is normally the
    # tail/head of a longer phrase (for example a residual suffix).  Typed
    # one-character entities such as chemical symbols are not filtered here.
    if len(surface.strip()) == 1:
        before = content[start - 1] if start else ""
        after = content[end] if end < len(content) else ""
        if any("\u3400" <= value <= "\u9fff" for value in (before, after)):
            return "partial_context_fragment"
    # Generic spans that stop in the middle of an uninterrupted CJK run are
    # structurally incomplete.  Typed mentions are intentionally exempt: the
    # schema/type assertion is the model's evidence that the shorter span is a
    # standalone referent.  This avoids accepting a clause prefix when the
    # missing tail is in another chunk and therefore unavailable as a nested
    # peer during this pass.
    after = content[end] if end < len(content) else ""
    if after and "\u3400" <= after <= "\u9fff":
        return "nonterminal_generic_surface"
    if entity_type == "GenericEntity":
        if (
            len(words) >= 4
            and words[0] in _GENERIC_FRAGMENT_STARTERS
            and any(word in _GENERIC_FRAGMENT_VERBS for word in words[1:])
        ):
            return "clause_fragment"
    return ""


def structured_relation_endpoint_candidates(
    content: str,
    structural_spans: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exact endpoint mentions from parser-declared relation lines.

    This helper understands only the structural ``left --label--> right``
    notation emitted by media/layout parsers.  It deliberately ignores the
    label: no relation type, direction, acceptance or publication decision is
    made here.  The normal pair planner, classifier and independent verifier
    remain mandatory.  Requiring an eligible list/media span prevents an
    arrow-like fragment in ordinary prose from becoming an entity source.
    """

    accepted_roles = {"list_item", "media_description"}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for span in structural_spans:
        if not isinstance(span, dict):
            continue
        if str(span.get("source_role") or "") not in accepted_roles:
            continue
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(content)):
            continue
        source = content[start:end]
        match = _STRUCTURED_RELATION_LINE.fullmatch(source)
        if not match:
            continue
        for side in ("left", "right"):
            raw = match.group(side)
            surface = raw.strip()
            if not surface:
                continue
            local = match.start(side) + (len(raw) - len(raw.lstrip()))
            left = start + local
            right = left + len(surface)
            key = (left, right)
            if key in seen or content[left:right] != surface:
                continue
            seen.add(key)
            rows.append({
                "id": f"structured-endpoint:{left}:{right}",
                "name": surface,
                "surface": surface,
                "type": "GenericEntity",
                "mention_start": left,
                "mention_end": right,
                "evidence_text": source,
                "evidence_start": start,
                "evidence_end": end,
                "confidence": 0.8,
                "evidence": source,
                "structural_role": str(span.get("source_role") or ""),
                "source_anchor": str(span.get("scope_id") or ""),
                "endpoint_candidate": True,
            })
    return rows


def structured_list_item_endpoint_candidates(
    content: str,
    structural_spans: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project one exact endpoint candidate from each parser list item.

    A list item is structural evidence that the complete item text is a
    reusable endpoint candidate, but it is not a relation assertion.  This
    bounded recall projection is intentionally syntax-only; type resolution,
    pair planning, relation direction, verification and publication remain
    unchanged.  It is especially useful when an entity model omits short
    enumerated concepts while extracting longer neighboring items.
    """

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for span in structural_spans:
        if not isinstance(span, dict) or str(span.get("source_role") or "") != "list_item":
            continue
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(content)):
            continue
        source = content[start:end]
        marker = str(span.get("list_marker") or "")
        local_start = len(source) - len(source.lstrip())
        if marker and source[local_start:].startswith(marker):
            local_start += len(marker)
            while local_start < len(source) and source[local_start].isspace():
                local_start += 1
        surface = source[local_start:].strip()
        if not surface:
            continue
        candidate_start = start + local_start
        candidate_end = candidate_start + len(surface)
        key = (candidate_start, candidate_end)
        if key in seen or content[candidate_start:candidate_end] != surface:
            continue
        seen.add(key)
        rows.append({
            "id": f"structured-list-item:{candidate_start}:{candidate_end}",
            "name": surface,
            "surface": surface,
            "type": "GenericEntity",
            "mention_start": candidate_start,
            "mention_end": candidate_end,
            "evidence_text": source,
            "evidence_start": start,
            "evidence_end": end,
            "confidence": 0.75,
            "evidence": source,
            "structural_role": "list_item",
            "source_anchor": str(span.get("scope_id") or ""),
            "endpoint_candidate": True,
        })
    return rows


def structured_section_label_endpoint_candidates(
    content: str,
    structural_spans: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project exact candidates from standalone, one-line body labels.

    Document parsers often retain labels such as a section/field name as a
    paragraph block.  They are easy for an entity model to omit, especially
    when the following child line carries most of the semantics.  This helper
    emits only a bounded endpoint candidate; it does not assert that the label
    relates to the following text.  Relation planning and verification remain
    mandatory.
    """

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for span in structural_spans:
        if not isinstance(span, dict) or str(span.get("source_role") or "") != "body":
            continue
        try:
            start, end = int(span["start"]), int(span["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= start < end <= len(content)):
            continue
        source = content[start:end]

        # A body line may begin with a stable source identifier and then
        # continue with descriptive prose.
        # The identifier is an exact, reusable endpoint occurrence even when
        # the entity model only returns the longer sentence or an occurrence
        # from a nearby table.  This projection is syntax-only: it does not
        # infer a relation or the identifier's type, and the normal resolver,
        # pair planner and verifier remain mandatory.
        for line_match in re.finditer(r"[^\r\n]+", source):
            line = line_match.group(0)
            label_match = _STRUCTURED_INLINE_LABEL.match(line)
            if not label_match:
                continue
            label = label_match.group("label")
            label_offset = line_match.start() + label_match.start("label")
            inline_start = start + label_offset
            inline_end = inline_start + len(label)
            key = (inline_start, inline_end)
            if key in seen or content[inline_start:inline_end] != label:
                continue
            seen.add(key)
            rows.append({
                "id": f"structured-inline-label:{inline_start}:{inline_end}",
                "name": label,
                "surface": label,
                "type": "GenericEntity",
                "mention_start": inline_start,
                "mention_end": inline_end,
                "evidence_text": line,
                "evidence_start": start + line_match.start(),
                "evidence_end": start + line_match.end(),
                "confidence": 0.72,
                "evidence": line,
                "structural_role": "body",
                "source_anchor": str(span.get("scope_id") or ""),
                "endpoint_candidate": True,
                "section_label_candidate": True,
                "inline_label_candidate": True,
            })
        if "\n" in source or "\r" in source:
            continue
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", end)
        if line_end < 0:
            line_end = len(content)
        if content[line_start:start].strip() or content[end:line_end].strip():
            continue
        if re.search(r"[.!?;。！？；]$", source.strip()):
            continue
        surface = source.strip(" \t\r\n:：")
        if not surface or len(surface) > 160:
            continue
        candidate_start = start + len(source) - len(source.lstrip(" \t\r\n"))
        candidate_end = candidate_start + len(surface)
        if candidate_end > end or (candidate_start, candidate_end) in seen:
            continue
        if content[candidate_start:candidate_end] != surface:
            continue
        seen.add((candidate_start, candidate_end))
        rows.append({
            "id": f"structured-section-label:{candidate_start}:{candidate_end}",
            "name": surface,
            "surface": surface,
            "type": "GenericEntity",
            "mention_start": candidate_start,
            "mention_end": candidate_end,
            "evidence_text": source,
            "evidence_start": start,
            "evidence_end": end,
            "confidence": 0.7,
            "evidence": source,
            "structural_role": "body",
            "source_anchor": str(span.get("scope_id") or ""),
            "endpoint_candidate": True,
            "section_label_candidate": True,
        })

    return rows


def policy_relation_endpoint_candidates(
    content: str,
    relation_types: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recover conservative exact endpoints around configured relation cues.

    This is an entity-recall aid only.  It consumes versioned Schema/Policy
    cues, never predicate identifiers, and emits no relation, direction or
    publication decision.  The normal span sanitizer, entity resolver and
    relation verifier remain mandatory.  The extractor intentionally handles
    only short, single-scope cue frames; long or ambiguous phrases are left to
    the model rather than guessed.
    """

    if not content.strip():
        return []
    cues: set[str] = set()
    for relation in relation_types:
        if not isinstance(relation, dict):
            continue
        for key in ("optional_soft_cues", "positive_cues"):
            for raw in relation.get(key) or []:
                cue = str(raw or "").strip()
                # Ellipsis/templated cues cannot identify exact offsets without
                # semantic parsing; keep those in the LLM prompt only.
                if cue and "..." not in cue and "…" not in cue:
                    cues.add(cue)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    ordered_cues = sorted(cues, key=lambda value: (-len(value), value))

    def cue_spans(cue: str) -> Iterable[tuple[int, int]]:
        """Yield cue occurrences without matching an ASCII word fragment.

        English schema cues such as ``is`` are often substrings of ordinary
        words (``implements``).  CJK cues deliberately keep substring matching:
        Chinese relation words commonly occur as morphemes inside a noun phrase
        and do not have a whitespace word boundary.
        """

        offset = 0
        while True:
            start = content.find(cue, offset)
            if start < 0:
                return
            end = start + len(cue)
            ascii_cue = bool(_ASCII_WORD_CUE.search(cue))
            if ascii_cue:
                before = content[start - 1:start] if start else ""
                after = content[end:end + 1]
                if (
                    _ASCII_WORD_CUE.search(before)
                    or _ASCII_WORD_CUE.search(after)
                ):
                    offset = end
                    continue
            yield start, end
            offset = end

    def trim_leading_determiner(
        text: str, start: int, end: int,
    ) -> tuple[int, int, str]:
        match = _LEADING_DETERMINER.match(text)
        if not match:
            return start, end, text
        start += match.end()
        return start, end, text[match.end():]

    def candidate_variants(
        text: str, start: int, end: int,
    ) -> list[tuple[int, int, str]]:
        """Return a bounded full phrase plus exact coordinated subphrases."""

        text = text.strip(" \t\r\n,，:：;；")
        if not text:
            return []
        # Re-find the trimmed text in the frozen source; this avoids offset
        # drift from punctuation and whitespace normalization.
        source_start = content.find(text, start, end)
        if source_start < 0:
            source_start = start
        source_end = source_start + len(text)
        variants: list[tuple[int, int, str]] = []
        variants.append((source_start, source_end, text))
        trimmed_start, trimmed_end, trimmed = trim_leading_determiner(
            text, source_start, source_end,
        )
        if trimmed and trimmed != text:
            variants.append((trimmed_start, trimmed_end, trimmed))

        def add_parts(value: str, value_start: int, value_end: int) -> None:
            """Add exact coordinated and relative-clause-free subphrases."""

            separators = list(_COORDINATION_SPLIT.finditer(value))
            starts = [0, *(match.end() for match in separators)]
            ends = [*(match.start() for match in separators), len(value)]
            for local_start, local_end in zip(starts, ends):
                part = value[local_start:local_end].strip(
                    " \t\r\n,，:：;；"
                )
                if not part:
                    continue
                part_start = content.find(part, value_start, value_end)
                if part_start < 0:
                    continue
                part_end = part_start + len(part)
                variants.append((part_start, part_end, part))
                part_trimmed_start, part_trimmed_end, part_trimmed = (
                    trim_leading_determiner(part, part_start, part_end)
                )
                if part_trimmed and part_trimmed != part:
                    variants.append((
                        part_trimmed_start, part_trimmed_end, part_trimmed,
                    ))

            # A relative clause often follows the actual endpoint, for
            # example ``a system that supports analysis``. Keep the noun
            # phrase before the clause as an additional exact candidate.
            for match in _RELATIVE_CLAUSE.finditer(value):
                part = value[:match.start()].strip(" \t\r\n,，:：;；")
                if not part:
                    continue
                part_start = content.find(part, value_start, value_end)
                if part_start >= 0:
                    part_end = part_start + len(part)
                    variants.append((part_start, part_end, part))
                    part_trimmed_start, part_trimmed_end, part_trimmed = (
                        trim_leading_determiner(part, part_start, part_end)
                    )
                    if part_trimmed and part_trimmed != part:
                        variants.append((
                            part_trimmed_start, part_trimmed_end, part_trimmed,
                        ))

        # A coordinated object phrase can contain several independently
        # reusable endpoints. Preserve the whole phrase for the model-backed
        # path, but also project each exact item for candidate planning.
        add_parts(text, source_start, source_end)

        # In frames such as ``contains data sets for X`` the grammatical
        # object is the exact tail after the preposition, not necessarily the
        # entire descriptive noun phrase before it.
        for match in _TAIL_PREPOSITION.finditer(text):
            tail = text[match.end():].strip(" \t\r\n,，:：;；")
            if not tail:
                continue
            tail_start = content.find(tail, source_start, source_end)
            if tail_start >= 0:
                tail_end = tail_start + len(tail)
                variants.append((tail_start, tail_end, tail))
                tail_trimmed_start, tail_trimmed_end, tail_trimmed = (
                    trim_leading_determiner(tail, tail_start, tail_end)
                )
                if tail_trimmed and tail_trimmed != tail:
                    variants.append((
                        tail_trimmed_start, tail_trimmed_end, tail_trimmed,
                    ))
                add_parts(tail, tail_start, tail_end)
        return variants

    for cue in ordered_cues:
        for cue_start, cue_end in cue_spans(cue):
            boundary_before = 0
            for match in _RELATION_SENTENCE_BOUNDARY.finditer(
                content, 0, cue_start,
            ):
                boundary_before = match.end()
            after_match = _RELATION_SENTENCE_BOUNDARY.search(content, cue_end)
            boundary_after = after_match.start() if after_match else len(content)
            left_text = content[boundary_before:cue_start].strip(" \t,，:：")
            right_text = content[cue_end:boundary_after].strip(" \t,，:：")
            # Coordinated frames are split at generic connectors so a whole
            # clause is never registered as a single entity candidate.
            left_text = _RELATION_CONNECTOR.sub("", left_text).strip()
            if (
                left_text and right_text
                and len(left_text) <= 96 and len(right_text) <= 96
            ):
                # Search the trimmed surfaces back in the frozen source so
                # punctuation/whitespace trimming never shifts Unicode spans.
                left_start = content.rfind(left_text, boundary_before, cue_start)
                right_start = content.find(right_text, cue_end, boundary_after)
                candidates = [
                    *candidate_variants(left_text, left_start, cue_start),
                    *candidate_variants(
                        right_text, right_start, boundary_after,
                    ),
                ]
                for start, end, surface in candidates:
                    if not (0 <= start < end <= len(content)):
                        continue
                    if content[start:end] != surface or (start, end) in seen:
                        continue
                    seen.add((start, end))
                    rows.append({
                        "id": f"policy-endpoint:{start}:{end}",
                        "name": surface,
                        "surface": surface,
                        "type": "GenericEntity",
                        "mention_start": start,
                        "mention_end": end,
                        "evidence_text": content[boundary_before:boundary_after],
                        "evidence_start": boundary_before,
                        "evidence_end": boundary_after,
                        "confidence": 0.55,
                        "evidence": content[boundary_before:boundary_after],
                        "source_anchor": f"policy-cue:{cue_start}:{cue_end}",
                        "endpoint_candidate": True,
                    })
    return rows


@dataclass(frozen=True)
class SpanMention:
    mention_id: str
    occurrence_key: str
    surface: str
    normalized_surface: str
    start: int
    end: int
    evidence: str
    confidence: float
    extraction_policy_version: str
    filter_decision: str = "accepted"
    filter_reason: str = ""
    publishability_score: float = 1.0
    publishability_protected: bool = False
    publishability_features: tuple[str, ...] = ()
    publishability_policy_version: str = ""
    recovery_kind: str = ""


@dataclass(frozen=True)
class TypeAssertion:
    mention_id: str
    entity_type: str
    confidence: float
    evidence: str
    policy_version: str
    source: str = "model"


@dataclass(frozen=True)
class AliasAssertion:
    mention_id: str
    surface: str
    normalized_surface: str
    start: int
    end: int
    confidence: float
    kind: str = "explicit_alias"


@dataclass
class SpanExtractionResult:
    mentions: list[SpanMention] = field(default_factory=list)
    type_assertions: list[TypeAssertion] = field(default_factory=list)
    aliases: list[AliasAssertion] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)


class SpanFirstEntityExtractor:
    """Accept model output only when it points at an exact source span."""

    def __init__(self, policy_version: str):
        self.policy_version = policy_version

    @staticmethod
    def _exact_span(content: str, surface: str, start: Any, end: Any) -> tuple[int, int] | None:
        try:
            left, right = int(start), int(end)
        except (TypeError, ValueError):
            return None
        if left < 0 or right <= left or right > len(content) or content[left:right] != surface:
            return None
        return left, right

    @staticmethod
    def _surface_occurrences(content: str, surface: str) -> list[tuple[int, int]]:
        """Find exact lexical repeats without inventing a new entity surface.

        The model remains responsible for identifying the entity string and
        type.  Once it has supplied one valid exact occurrence, recovering the
        same literal at other source positions is deterministic occurrence
        accounting, not semantic extraction.  ASCII word boundaries prevent a
        short name from being recovered inside a longer identifier.
        """

        if not surface:
            return []
        ascii_word = all(
            character.isascii() and (character.isalnum() or character == "_")
            for character in surface
        )
        values: list[tuple[int, int]] = []
        start = 0
        while True:
            left = content.find(surface, start)
            if left < 0:
                break
            right = left + len(surface)
            start = max(right, left + 1)
            if ascii_word:
                before = content[left - 1] if left else ""
                after = content[right] if right < len(content) else ""
                if (
                    (before and (before.isalnum() or before == "_"))
                    or (after and (after.isalnum() or after == "_"))
                ):
                    continue
            values.append((left, right))
        return values

    def audit_layout_coverage(
        self,
        *,
        content: str,
        mentions: Iterable[SpanMention],
        type_assertions: Iterable[TypeAssertion],
        chunk_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Audit missing layout-equivalent occurrences without changing output.

        The audit starts only from mentions that already passed exact-span and
        publishability validation.  Candidate source positions are then run
        through the same structural and publishability gates as ordinary
        mentions.  Its result is safe to persist as observability metadata and
        can later drive a bounded recovery pass without adding an LLM call.
        """

        source = str(content or "")
        mention_values = sorted(
            list(mentions),
            key=lambda item: (item.start, item.end, item.mention_id),
        )
        recovered_mentions = [
            {
                "mention_id": item.mention_id,
                "surface": item.surface,
                "normalized_surface": item.normalized_surface,
                "start": item.start,
                "end": item.end,
                "recovery_kind": item.recovery_kind,
            }
            for item in mention_values
            if item.recovery_kind
        ]
        type_by_mention = {
            item.mention_id: item for item in type_assertions
        }
        occupied = [(item.start, item.end) for item in mention_values]
        candidates: dict[tuple[int, int, str], dict[str, Any]] = {}
        overlap_skipped = 0
        for mention in mention_values:
            assertion = type_by_mention.get(mention.mention_id)
            entity_type = str(
                (assertion.entity_type if assertion else "")
                or "GenericEntity"
            )
            seed_reason = layout_recovery_seed_rejection_reason(
                mention.surface, entity_type,
            )
            for left, right in layout_equivalent_surface_occurrences(
                source, mention.surface,
            ):
                if any(
                    existing_left < right and left < existing_right
                    for existing_left, existing_right in occupied
                ):
                    overlap_skipped += 1
                    continue
                candidate_surface = source[left:right]
                raw = {
                    "name": candidate_surface,
                    "surface": candidate_surface,
                    "type": entity_type,
                    "mention_start": left,
                    "mention_end": right,
                    "confidence": mention.confidence,
                    "deterministic_layout_coverage": True,
                    "seed_mention_id": mention.mention_id,
                }
                reason = seed_reason or self._reject_reason(
                    candidate_surface, chunk_metadata, raw,
                    start=left, end=right,
                )
                publishability = None
                if not reason:
                    publishability = classify_mention_publishability(
                        surface=candidate_surface,
                        entity_type=entity_type,
                        content=source,
                        start=left,
                        end=right,
                        chunk_metadata=chunk_metadata,
                        raw=raw,
                    )
                    if publishability.decision == PUBLISHABILITY_REJECTED:
                        reason = publishability.reason
                row = {
                    "seed_mention_id": mention.mention_id,
                    "seed_surface": mention.surface,
                    "surface": candidate_surface,
                    "normalized_surface": normalize_name(candidate_surface),
                    "start": left,
                    "end": right,
                    "entity_type": entity_type,
                    "confidence": mention.confidence,
                    "decision": "rejected" if reason else "eligible",
                    "reason": reason,
                    "publishability": (
                        {
                            "decision": publishability.decision,
                            "reason": publishability.reason,
                            "score": publishability.score,
                            "protected": publishability.protected,
                            "features": list(publishability.features),
                            "policy_version": publishability.policy_version,
                        }
                        if publishability is not None else {}
                    ),
                }
                key = (left, right, row["normalized_surface"])
                existing = candidates.get(key)
                if existing is None or (
                    float(row["confidence"]), str(row["seed_mention_id"])
                ) > (
                    float(existing["confidence"]),
                    str(existing["seed_mention_id"]),
                ):
                    candidates[key] = row
        rows = sorted(
            candidates.values(),
            key=lambda item: (
                int(item["start"]), int(item["end"]),
                str(item["normalized_surface"]),
            ),
        )
        eligible = [item for item in rows if item["decision"] == "eligible"]
        rejected = [item for item in rows if item["decision"] == "rejected"]
        reasons: dict[str, int] = {}
        for item in rejected:
            reason = str(item.get("reason") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        return {
            "policy_version": LAYOUT_MENTION_COVERAGE_POLICY_VERSION,
            "candidate_count": len(rows),
            "eligible_count": len(eligible),
            "rejected_count": len(rejected),
            "recovered_count": len(recovered_mentions),
            "overlap_skipped": overlap_skipped,
            "rejection_reasons": dict(sorted(reasons.items())),
            "recovered_mentions": recovered_mentions,
            "candidates": rows,
        }

    @staticmethod
    def _coordinated_item_span(
        content: str, start: int, end: int,
    ) -> tuple[int, int, int, int, bool, bool] | None:
        """Locate the compact coordinated item containing one exact span.

        The returned scope coordinates make it possible to require a complete
        short item and its longer sibling to occur in the same sentence.  A
        recovered longer item must be bounded by explicit separators on both
        sides, which keeps sentence prefixes and trailing prose out of this
        narrow boundary repair.
        """

        def hard_boundary(match: re.Match[str]) -> bool:
            index = match.start()
            value = match.group(0)
            if value == "\n" and is_soft_wrap_newline(content, index):
                return False
            if (
                value == "\r" and index + 1 < len(content)
                and content[index + 1] == "\n"
                and is_soft_wrap_newline(content, index + 1)
            ):
                return False
            return True

        scope_start = 0
        for match in _COORDINATION_SCOPE_BOUNDARY.finditer(content, 0, start):
            if hard_boundary(match):
                scope_start = match.end()
        boundary_after = next(
            (
                match for match in _COORDINATION_SCOPE_BOUNDARY.finditer(
                    content, end,
                )
                if hard_boundary(match)
            ),
            None,
        )
        scope_end = boundary_after.start() if boundary_after else len(content)
        if not (scope_start <= start < end <= scope_end):
            return None
        separators = [
            match
            for match in
            _COORDINATION_ITEM_SEPARATOR.finditer(
                content, scope_start, scope_end,
            )
            if content[scope_start:match.start()].strip()
            and content[match.end():scope_end].strip()
        ]
        left_separator = next(
            (match for match in reversed(separators) if match.end() <= start),
            None,
        )
        right_separator = next(
            (match for match in separators if match.start() >= end),
            None,
        )
        item_start = left_separator.end() if left_separator else scope_start
        item_end = right_separator.start() if right_separator else scope_end
        while item_start < item_end and content[item_start].isspace():
            item_start += 1
        while item_end > item_start and content[item_end - 1].isspace():
            item_end -= 1
        # A compact coordination is commonly embedded in parentheses, for
        # for example ``Subsystem (level A, level B, level C)``.  The final item is
        # followed by the wrapper's closing delimiter, so without removing
        # that structural character it is not recognized as a complete item
        # and a shorter model span can remain incorrectly grounded inside a
        # longer sibling.  Strip only delimiters whose unmatched opener is
        # before this item in the same frozen scope; balanced identifier text
        # such as ``API(v2)`` therefore remains untouched.
        closing_delimiters = {
            ")": "(", "]": "[", "}": "{", "）": "（", "】": "【",
            "〉": "〈", "》": "《", "」": "「", "』": "『",
        }
        while item_end > item_start:
            closing = content[item_end - 1]
            opening = closing_delimiters.get(closing)
            if opening is None:
                break
            scope_prefix = content[scope_start:item_start]
            if scope_prefix.count(opening) <= scope_prefix.count(closing):
                break
            item_end -= 1
            while item_end > item_start and content[item_end - 1].isspace():
                item_end -= 1
        if not (item_start <= start < end <= item_end):
            return None
        return (
            item_start, item_end, scope_start, scope_end,
            left_separator is not None, right_separator is not None,
        )

    @classmethod
    def _recover_soft_wrapped_coordinated_candidates(
        cls,
        *,
        content: str,
        raw_entities: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Restore a coordinated item when a PDF wrap split its noun phrase.

        This path is intentionally narrower than general phrase expansion. A
        model-supplied exact endpoint must sit at an edge of a comma-bounded
        coordinated item, the omitted bytes must cross only parser-confirmed
        soft newlines, and the complete item must remain a compact noun-like
        surface.  It therefore repairs ``中断/异常\n管理`` from a short
        ``管理`` endpoint without joining ordinary adjacent source lines.
        """

        raw_values = [dict(item) for item in raw_entities]
        known_spans = {
            span
            for item in raw_values
            if (span := _exact_raw_span(content, item)) is not None
        }
        recovered: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for raw in raw_values:
            span = _exact_raw_span(content, raw)
            if span is None:
                continue
            start, end = span
            item = cls._coordinated_item_span(content, start, end)
            if item is None:
                continue
            (
                item_start, item_end, scope_start, scope_end,
                left_bounded, right_bounded,
            ) = item
            item_span = (item_start, item_end)
            if (
                item_span == span
                or item_span in known_spans
                or item_span in seen
                or not (left_bounded and right_bounded)
                or not (start == item_start or end == item_end)
                or item_end - item_start > 80
                or item_end - item_start - (end - start) > 32
            ):
                continue
            item_surface = content[item_start:item_end]
            newline_offsets = [
                index for index in range(item_start, item_end)
                if content[index] == "\n"
            ]
            if (
                not newline_offsets
                or not all(
                    is_soft_wrap_newline(content, index)
                    for index in newline_offsets
                )
                or _SOFT_WRAP_ENTITY_ITEM.fullmatch(item_surface) is None
            ):
                continue
            candidate = dict(raw)
            candidate.update({
                "name": item_surface,
                "surface": item_surface,
                "mention_start": item_start,
                "mention_end": item_end,
                "evidence_text": content[scope_start:scope_end],
                "evidence_start": scope_start,
                "evidence_end": scope_end,
                "confidence": min(
                    coerce_confidence(raw.get("confidence"), 0.7), 0.7,
                ),
                "endpoint_candidate": True,
                "grounded_semantic_mention": True,
                "coordinated_distinct_candidate": True,
                "soft_wrap_surface_repaired": True,
            })
            for field_name in (
                "alias_spans", "aliases", "source_start", "source_end",
            ):
                candidate.pop(field_name, None)
            seen.add(item_span)
            recovered.append(candidate)
        return recovered

    @classmethod
    def _recover_distinct_coordinated_candidates(
        cls,
        *,
        content: str,
        raw_entities: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Repair a short mention grounded inside a distinct sibling item.

        This does not enumerate arbitrary list text.  Recovery is allowed only
        when the model already supplied an exact entity surface, that surface
        occurs as a complete coordinated item, and another occurrence lies at
        an edge of a different compact item bounded by separators on both
        sides.  The complete sibling is emitted as an endpoint candidate while
        the nested short occurrence is rejected by ``entity_boundary_reason``.
        """

        raw_values = [dict(item) for item in raw_entities]
        examples: dict[str, dict[str, Any]] = {}
        for raw in raw_values:
            surface = str(raw.get("surface") or raw.get("name") or "")
            if surface and _exact_raw_span(content, raw) is not None:
                examples.setdefault(surface, raw)
        recovered: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for surface, example in examples.items():
            occurrences = cls._surface_occurrences(content, surface)
            if len(occurrences) < 2:
                continue
            occurrence_items: list[
                tuple[int, int, int, int, int, int, bool, bool]
            ] = []
            for start, end in occurrences:
                item = cls._coordinated_item_span(content, start, end)
                if item is not None:
                    occurrence_items.append((start, end, *item))
            standalone_scopes = {
                (scope_start, scope_end)
                for (
                    start, end, item_start, item_end, scope_start, scope_end,
                    _left_bounded, _right_bounded,
                ) in occurrence_items
                if (start, end) == (item_start, item_end)
            }
            if not standalone_scopes:
                continue
            for (
                start, end, item_start, item_end, scope_start, scope_end,
                left_bounded, right_bounded,
            ) in occurrence_items:
                if (scope_start, scope_end) not in standalone_scopes:
                    continue
                if (start, end) == (item_start, item_end):
                    continue
                if not (left_bounded and right_bounded):
                    continue
                item_surface = content[item_start:item_end]
                added = len(item_surface) - len(surface)
                if (
                    added <= 0 or added > 8 or len(item_surface) > 80
                    or _COMPACT_ENTITY_ITEM.fullmatch(item_surface) is None
                    or not (start == item_start or end == item_end)
                    or (item_start, item_end) in seen
                ):
                    continue
                candidate = dict(example)
                candidate.update({
                    "name": item_surface,
                    "surface": item_surface,
                    "type": str(example.get("type") or "GenericEntity"),
                    "mention_start": item_start,
                    "mention_end": item_end,
                    "evidence_text": content[scope_start:scope_end],
                    "evidence_start": scope_start,
                    "evidence_end": scope_end,
                    "confidence": min(
                        coerce_confidence(example.get("confidence"), 0.7),
                        0.7,
                    ),
                    "endpoint_candidate": True,
                    "coordinated_distinct_candidate": True,
                })
                for field_name in (
                    "alias_spans", "aliases", "source_start", "source_end",
                ):
                    candidate.pop(field_name, None)
                seen.add((item_start, item_end))
                recovered.append(candidate)
        return recovered

    def _recover_repeated_occurrences(
        self,
        *,
        content: str,
        raw_entities: Iterable[dict[str, Any]],
        chunk_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        values = [dict(item) for item in raw_entities]
        values.extend(self._recover_soft_wrapped_coordinated_candidates(
            content=content,
            raw_entities=values,
        ))
        values.extend(self._recover_distinct_coordinated_candidates(
            content=content,
            raw_entities=values,
        ))
        expanded = list(values)
        known = {
            (
                str(item.get("surface") or item.get("name") or ""),
                item.get("mention_start", item.get("evidence_start")),
                item.get("mention_end", item.get("evidence_end")),
            )
            for item in values
        }
        recovered: set[tuple[str, int, int]] = set()
        # Model-supplied exact spans define the local mention boundaries.  A
        # deterministic repeat recovery must not manufacture a shorter mention
        # inside one of those longer spans (for example recovering a city name
        # from inside a facility's full proper name).  Explicit nested mentions
        # remain possible when the extractor itself supplied them; only the
        # automatic repeat expansion is constrained here.
        model_spans: list[tuple[int, int]] = []
        for item in values:
            item_surface = str(item.get("surface") or item.get("name") or "")
            item_span = self._exact_span(
                content,
                item_surface,
                item.get("mention_start", item.get("evidence_start")),
                item.get("mention_end", item.get("evidence_end")),
            )
            if item_span is not None:
                model_spans.append(item_span)
        for raw in values:
            surface = str(raw.get("surface") or raw.get("name") or "")
            if entity_boundary_reason(content, raw, values):
                continue
            source_span = self._exact_span(
                content,
                surface,
                raw.get("mention_start", raw.get("evidence_start")),
                raw.get("mention_end", raw.get("evidence_end")),
            )
            if source_span is None or self._reject_reason(
                surface,
                chunk_metadata,
                raw,
                start=source_span[0],
                end=source_span[1],
            ):
                continue
            occurrence_candidates = [
                (left, right, "")
                for left, right in self._surface_occurrences(content, surface)
            ]
            if not layout_recovery_seed_rejection_reason(
                surface, str(raw.get("type") or "GenericEntity"),
            ):
                occurrence_candidates.extend(
                    (left, right, "layout_whitespace_variant")
                    for left, right in layout_equivalent_surface_occurrences(
                        content, surface,
                    )
                )
            for left, right, recovery_kind in occurrence_candidates:
                candidate_surface = content[left:right]
                key = (candidate_surface, left, right)
                if key in known or key in recovered:
                    continue
                if any(
                    outer_left <= left < right <= outer_right
                    and (outer_left, outer_right) != (left, right)
                    for outer_left, outer_right in model_spans
                ):
                    continue
                candidate = dict(raw)
                candidate["name"] = candidate_surface
                candidate["surface"] = candidate_surface
                candidate["mention_start"] = left
                candidate["mention_end"] = right
                candidate["evidence_start"] = left
                candidate["evidence_end"] = right
                candidate["evidence_text"] = candidate_surface
                if recovery_kind:
                    candidate["recovered_layout_occurrence"] = True
                    candidate["recovery_kind"] = recovery_kind
                else:
                    candidate["recovered_exact_occurrence"] = True
                for field_name in (
                    "source_anchor", "block_key", "source_start",
                    "source_end", "source_base",
                ):
                    candidate.pop(field_name, None)
                if self._reject_reason(
                    candidate_surface,
                    chunk_metadata,
                    candidate,
                    start=left,
                    end=right,
                ):
                    continue
                recovered.add(key)
                expanded.append(candidate)
        return expanded

    @staticmethod
    def _source_anchor(
        *,
        document_key: str,
        chunk_key: str,
        surface: str,
        left: int,
        right: int,
        chunk_metadata: dict[str, Any],
        raw: dict[str, Any],
    ) -> str:
        anchor = str(
            raw.get("source_anchor")
            or raw.get("block_key")
            or chunk_metadata.get("source_anchor")
            or chunk_metadata.get("block_key")
            or f"chunk:{chunk_key}"
        )
        try:
            base = int(
                raw.get("source_base")
                if raw.get("source_base") is not None
                else chunk_metadata.get("source_start")
                if chunk_metadata.get("source_start") is not None
                else chunk_metadata.get("chunk_source_start")
                if chunk_metadata.get("chunk_source_start") is not None
                else 0
            )
        except (TypeError, ValueError):
            base = 0
        source_left = int(raw.get("source_start") if raw.get("source_start") is not None else base + left)
        source_right = int(raw.get("source_end") if raw.get("source_end") is not None else base + right)
        material = "\x1f".join((
            document_key, anchor, str(source_left), str(source_right), surface,
        ))
        return "occurrence_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _reject_reason(
        surface: str,
        metadata: dict[str, Any],
        raw: dict[str, Any],
        *,
        start: int,
        end: int,
    ) -> str:
        stripped = surface.strip()
        normalized = normalize_name(stripped)
        # Parser-owned source structure wins over a model-supplied label.  A
        # model must not be able to reclassify a page header/table header as
        # body text simply by emitting a different structural_role.
        role = str(
            metadata.get("structural_role")
            or metadata.get("source_role")
            or metadata.get("chunk_role")
            or metadata.get("cell_role")
            or raw.get("structural_role")
            or ""
        ).lower().replace("-", "_").replace(" ", "_")
        block_types = {
            str(value).strip().lower().replace("-", "_")
            for value in metadata.get("block_types") or []
            if str(value).strip()
        }
        if not role and len(block_types) == 1:
            role = next(iter(block_types))
        entity_type = str(raw.get("type") or "GenericEntity")
        for structural_span in metadata.get("structural_spans") or []:
            if not isinstance(structural_span, dict):
                continue
            try:
                span_start = int(structural_span["start"])
                span_end = int(structural_span["end"])
            except (KeyError, TypeError, ValueError):
                continue
            span_role = str(
                structural_span.get("source_role") or ""
            ).lower().replace("-", "_")
            if span_start <= start < end <= span_end and span_role in _STRUCTURAL_ROLES:
                return "structural_node"
        if role in _STRUCTURAL_ROLES:
            return "structural_node"
        if entity_type in _NOISE_ENTITY_TYPES:
            return "scalar_or_temporal_noise"
        if not normalized:
            return "empty_normalized_surface"
        if normalized in _NON_REFERENTIAL_MODIFIERS:
            return "non_referential_modifier"
        if (
            _PURE_NUMBER.fullmatch(stripped)
            or _YEAR_OR_ESTIMATE.fullmatch(stripped)
            or _DATE_EXPRESSION.fullmatch(stripped)
            or _PERCENTAGE.fullmatch(stripped)
            or _PHONE.fullmatch(stripped)
            or _AMOUNT_OR_UNIT.fullmatch(stripped)
        ):
            return "scalar_or_temporal_noise"
        return ""

    def normalize(
        self,
        *,
        project_scope: str,
        document_key: str,
        chunk_key: str,
        content: str,
        raw_entities: Iterable[dict[str, Any]],
        chunk_metadata: dict[str, Any],
    ) -> SpanExtractionResult:
        result = SpanExtractionResult()
        seen: set[str] = set()
        expanded_entities = [
            repair_soft_wrapped_entity_span(content, raw)
            for raw in self._recover_repeated_occurrences(
                content=content,
                raw_entities=raw_entities,
                chunk_metadata=chunk_metadata,
            )
        ]
        for raw in expanded_entities:
            raw = normalize_entity_span(content, raw)
            surface = str(raw.get("surface") or raw.get("name") or "")
            boundary_reason = entity_boundary_reason(
                content, raw, expanded_entities,
            )
            if boundary_reason:
                result.rejected.append({
                    "raw": dict(raw), "reason": boundary_reason,
                })
                continue
            span = self._exact_span(
                content,
                surface,
                raw.get("mention_start", raw.get("evidence_start")),
                raw.get("mention_end", raw.get("evidence_end")),
            )
            if span is None:
                result.rejected.append({"raw": dict(raw), "reason": "invalid_exact_span"})
                continue
            reject_reason = self._reject_reason(
                surface,
                chunk_metadata,
                raw,
                start=span[0],
                end=span[1],
            )
            if reject_reason:
                result.rejected.append({"raw": dict(raw), "reason": reject_reason})
                continue
            left, right = span
            entity_type = str(raw.get("type") or "GenericEntity")
            publishability = classify_mention_publishability(
                surface=surface,
                entity_type=entity_type,
                content=content,
                start=left,
                end=right,
                chunk_metadata=chunk_metadata,
                raw=raw,
            )
            if publishability.decision == PUBLISHABILITY_REJECTED:
                result.rejected.append({
                    "raw": dict(raw),
                    "reason": publishability.reason,
                    "policy_version": publishability.policy_version,
                    "publishability": {
                        "decision": publishability.decision,
                        "score": publishability.score,
                        "features": list(publishability.features),
                    },
                })
                continue
            material = "\x1f".join((
                project_scope, document_key, chunk_key, str(left), str(right),
                surface,
            ))
            mention_id = "mention_" + hashlib.sha256(material.encode("utf-8")).hexdigest()
            if mention_id in seen:
                continue
            seen.add(mention_id)
            confidence = coerce_confidence(raw.get("confidence"), 0.5)
            occurrence_key = self._source_anchor(
                document_key=document_key,
                chunk_key=chunk_key,
                surface=surface,
                left=left,
                right=right,
                chunk_metadata=chunk_metadata,
                raw=raw,
            )
            mention = SpanMention(
                mention_id=mention_id,
                occurrence_key=occurrence_key,
                surface=surface,
                normalized_surface=normalize_name(surface),
                start=left,
                end=right,
                evidence=str(raw.get("evidence_text") or surface),
                confidence=confidence,
                extraction_policy_version=self.policy_version,
                filter_decision=publishability.decision,
                filter_reason=(
                    "" if publishability.decision == "accepted"
                    else publishability.reason
                ),
                publishability_score=publishability.score,
                publishability_protected=publishability.protected,
                publishability_features=publishability.features,
                publishability_policy_version=publishability.policy_version,
                recovery_kind=str(raw.get("recovery_kind") or ""),
            )
            result.mentions.append(mention)
            result.type_assertions.append(TypeAssertion(
                mention_id=mention_id,
                entity_type=entity_type,
                confidence=confidence,
                evidence=surface,
                policy_version=self.policy_version,
                source=(
                    "deterministic_layout_repeat"
                    if raw.get("recovered_layout_occurrence") is True
                    else "model"
                ),
            ))
            for alias in raw.get("alias_spans") or []:
                alias_surface = str(alias.get("surface") or "")
                alias_span = self._exact_span(
                    content, alias_surface, alias.get("start"), alias.get("end"),
                )
                if alias_span is None:
                    continue
                result.aliases.append(AliasAssertion(
                    mention_id=mention_id,
                    surface=alias_surface,
                    normalized_surface=normalize_name(alias_surface),
                    start=alias_span[0],
                    end=alias_span[1],
                    confidence=coerce_confidence(alias.get("confidence"), confidence),
                    kind=str(alias.get("kind") or "explicit_alias"),
                ))
        return result
