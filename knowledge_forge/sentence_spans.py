"""Shared sentence segmentation with exact half-open source offsets."""

from __future__ import annotations

import re


_CLOSING_PUNCTUATION = frozenset('"\'”’）)]】》')
_ABBREVIATION_RE = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|[A-Z])\.$",
    re.IGNORECASE,
)
_INITIALISM_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")
_STRUCTURAL_LINE_START_RE = re.compile(
    r"^(?:#{1,6}\s|[-*+•▪◦–—♦◆◇]\s*|\d{1,4}[.)]\s+)"
)


def _period_is_boundary(content: str, index: int) -> bool:
    previous = content[index - 1] if index else ""
    following = content[index + 1] if index + 1 < len(content) else ""
    if following == ".":
        return False
    if previous == ".":
        return following != "."
    if previous.isdigit() and following.isdigit():
        return False
    if previous.isalnum() and following.isalnum():
        return False
    token_start = index
    while token_start and not content[token_start - 1].isspace():
        token_start -= 1
    prefix = content[token_start:index + 1]
    return not (_ABBREVIATION_RE.fullmatch(prefix) or _INITIALISM_RE.fullmatch(prefix))


def split_sentences(content: str) -> list[tuple[int, int, str]]:
    """Split Chinese/English prose without losing exact source coordinates."""

    result: list[tuple[int, int, str]] = []
    start = 0
    index = 0
    length = len(content)
    while index < length:
        value = content[index]
        boundary = value in "。！？!?\n"
        if value == ".":
            boundary = _period_is_boundary(content, index)
        elif value == "…":
            boundary = index + 1 >= length or content[index + 1] != "…"
        if not boundary:
            index += 1
            continue
        end = index + 1
        while end < length and content[end] in _CLOSING_PUNCTUATION:
            end += 1
        # Horizontal whitespace after prose punctuation belongs to the
        # completed sentence.  After a newline, however, it is the indentation
        # of the next source line and must remain at that line's start so list
        # hierarchy survives downstream chunking and source coordinates.
        if value != "\n":
            while end < length and content[end].isspace() and content[end] != "\n":
                end += 1
        if end > start:
            result.append((start, end, content[start:end]))
        start = end
        index = end
    if start < length:
        result.append((start, length, content[start:]))
    return result


def is_soft_wrap_newline(content: str, index: int) -> bool:
    """Recognize a conservative PDF/OCR prose line wrap.

    PDF text extraction commonly inserts a newline at the renderer's right
    margin.  Treating every such newline as an atomic relation boundary loses
    coordinated members and can split one noun phrase in two.  This helper is
    deliberately layout-only: list markers, headings, tables, indentation,
    blank lines and sentence punctuation remain hard boundaries.
    """

    if not (0 <= index < len(content)) or content[index] != "\n":
        return False
    line_start = content.rfind("\n", 0, index) + 1
    next_end = content.find("\n", index + 1)
    if next_end < 0:
        next_end = len(content)
    previous_raw = content[line_start:index]
    next_raw = content[index + 1:next_end]
    previous = previous_raw.strip()
    following = next_raw.strip()
    if not previous or not following:
        return False
    if previous.endswith(("。", "！", "？", "!", "?", ";", "；", ":", "：")):
        return False
    if (
        _STRUCTURAL_LINE_START_RE.match(previous)
        or _STRUCTURAL_LINE_START_RE.match(following)
        or "|" in previous
        or "|" in following
    ):
        return False
    # Indentation is meaningful source structure (code, nested lists and
    # definition blocks), not a renderer-inserted prose wrap.
    if next_raw[:1].isspace():
        return False

    if previous.endswith((",", "，", "、", "-", "‐", "‑")):
        return True

    # A lowercase continuation is a strong signal in Latin prose.  Keep a
    # small length floor so short labels such as ``Status\nready`` do not get
    # merged merely because the value happens to be lowercase.
    first = following[0]
    if first.isascii() and first.islower() and len(previous) >= 12:
        return True

    # CJK has no case distinction.  Long adjacent lines without terminal
    # punctuation are the conservative proxy for a renderer wrap.  Short
    # diagram labels and one-item-per-line inventories remain separate.
    previous_cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", previous))
    following_cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", following))
    return previous_cjk >= 14 and following_cjk >= 2


def split_source_sentences(
    content: str,
) -> list[tuple[int, int, str]]:
    """Return sentence spans while preserving confirmed renderer soft wraps.

    ``split_sentences`` deliberately retains the historical hard-newline
    contract used by structure-aware callers. Chunk construction, however,
    must not cut one PDF/OCR sentence merely because the renderer wrapped it
    at the page margin. This projection keeps exact source offsets and only
    rejoins boundaries accepted by :func:`is_soft_wrap_newline`.
    """

    source = split_sentences(content)
    result: list[tuple[int, int, str]] = []
    current_start: int | None = None
    current_end: int | None = None
    for start, end, text in source:
        if not text.strip():
            if current_start is not None:
                result.append((
                    current_start, int(current_end),
                    content[current_start:int(current_end)],
                ))
                current_start = current_end = None
            continue
        if current_start is None:
            current_start, current_end = start, end
            continue
        boundary_index = int(current_end) - 1
        if (
            boundary_index >= current_start
            and content[boundary_index] == "\n"
            and is_soft_wrap_newline(content, boundary_index)
            and start == current_end
        ):
            current_end = end
            continue
        result.append((
            current_start, int(current_end),
            content[current_start:int(current_end)],
        ))
        current_start, current_end = start, end
    if current_start is not None:
        result.append((
            current_start, int(current_end),
            content[current_start:int(current_end)],
        ))
    return result


def split_relation_scopes(content: str) -> list[tuple[int, int, str]]:
    """Split prose into conservative atomic scopes for relation evidence.

    A semicolon normally joins complete or independently predicated clauses.
    Treating both sides as one atomic relation scope lets a model skip an
    intermediate argument and connect two merely adjacent entities.  This
    splitter preserves the ordinary sentence offsets, then refines them at
    Chinese and ASCII semicolons.  It is deliberately structural and has no
    knowledge of entity names, domains, predicates, or cue words.
    """

    result: list[tuple[int, int, str]] = []
    for sentence_start, sentence_end, _sentence in split_source_sentences(
        content
    ):
        start = sentence_start
        index = sentence_start
        while index < sentence_end:
            if content[index] not in ";；":
                index += 1
                continue
            end = index + 1
            if end > start:
                result.append((start, end, content[start:end]))
            start = end
            index = end
        if start < sentence_end:
            result.append((start, sentence_end, content[start:sentence_end]))
    return result


def sentence_index(sentences: list[tuple[int, int, str]], offset: int) -> int | None:
    for index, (start, end, _) in enumerate(sentences):
        if start <= offset < end:
            return index
    return None
