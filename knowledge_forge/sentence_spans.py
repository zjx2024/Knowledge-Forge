"""Shared sentence segmentation with exact half-open source offsets."""

from __future__ import annotations

import re


_CLOSING_PUNCTUATION = frozenset('"\'”’）)]】》')
_ABBREVIATION_RE = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|[A-Z])\.$",
    re.IGNORECASE,
)
_INITIALISM_RE = re.compile(r"(?:[A-Za-z]\.){2,}$")


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
        while end < length and content[end].isspace() and content[end] != "\n":
            end += 1
        if end > start:
            result.append((start, end, content[start:end]))
        start = end
        index = end
    if start < length:
        result.append((start, length, content[start:]))
    return result


def sentence_index(sentences: list[tuple[int, int, str]], offset: int) -> int | None:
    for index, (start, end, _) in enumerate(sentences):
        if start <= offset < end:
            return index
    return None
