"""Enhanced-only structured context units for Shadow v2 relation work."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .quality import canonical_json
from .sentence_spans import split_relation_scopes


SHADOW_CONTEXT_POLICY_VERSION = "shadow-context-source-map-v8-cross-block-lists"
SHADOW_CONTEXT_SOURCE_MAP_POLICY = (
    "block-sentence-list-or-legacy-local-v8-cross-block-lists"
)
KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION = "knowledge-forge-structure-gate-v5"

# These are parser roles, rather than business concepts.  They deliberately
# decide only whether a source is suitable for factual extraction; relation
# semantics, direction and publication still come exclusively from Schema.
_SOURCE_ROLE_ALIASES = {
    "paragraph": "body",
    "body": "body",
    "text": "body",
    "sentence": "body",
    "structured_chunk": "body",
    "legacy_chunk": "body",
    "clause": "body",
    "apposition": "body",
    "list": "list_item",
    "list_item": "list_item",
    "list-item": "list_item",
    "table_row": "table_row",
    "table-row": "table_row",
    "row": "table_row",
    "table_cell": "table_cell",
    "table-cell": "table_cell",
    "cell": "table_cell",
    "heading": "heading",
    "title": "heading",
    "header": "page_header",
    "page_header": "page_header",
    "page-header": "page_header",
    "footer": "page_footer",
    "page_footer": "page_footer",
    "page-footer": "page_footer",
    "page_number": "page_number",
    "page-number": "page_number",
    "table": "table_container",
    "table_header": "table_header",
    "table-header": "table_header",
    "caption": "caption",
    "footnote": "footnote",
    "image": "media_description",
    "figure": "media_description",
    "media_description": "media_description",
    "reference_heading": "reference_heading",
    "table_of_contents": "table_of_contents",
    "index_entry": "index_entry",
}
_RELATION_ELIGIBLE_SOURCE_ROLES = frozenset({
    "body", "list_item", "table_row", "table_cell", "media_description",
})
_LIST_MARKER_EXCLUDED = frozenset("#.,;:!?，。；：！？()[]{}<>《》【】\"'“”‘’")
_PAGE_NAVIGATION_HEADING = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:"
    r"第\s*\d{1,6}\s*页|"
    r"pages?\s*[-:#.]?\s*\d{1,6}|"
    r"p(?:age)?\.?\s*[-:#.]?\s*\d{1,6}"
    r")\s*$",
    re.IGNORECASE,
)
_STANDALONE_PAGE_NUMBER = re.compile(
    r"^\s*(?:page|页码)?\s*[-:#.]?\s*\d{1,6}\s*$",
    re.IGNORECASE,
)


def _list_marker(line: str) -> tuple[str, int] | None:
    """Return a syntax-only list marker and its indentation width.

    PDF text extraction commonly turns bullets into private-use glyphs or
    em-dashes.  Treating only ASCII ``-``/``*`` markers as lists makes an
    otherwise explicit hierarchy look like unrelated prose.  This parser
    recognizes punctuation/symbol markers (including private-use glyphs) only
    when followed by whitespace and excludes ordinary sentence punctuation.
    No text content or domain vocabulary is inspected.
    """

    leading = len(line) - len(line.lstrip())
    value = line[leading:]
    if not value:
        return None
    if value[0].isdigit():
        index = 0
        while index < len(value) and value[index].isdigit():
            index += 1
        if index < len(value) and value[index] in ".)" and index + 1 < len(value):
            if value[index + 1].isspace() and value[index + 2:].strip():
                return value[: index + 1], leading
        return None
    if len(value) >= 2 and value[0].isascii() and value[0].isalpha():
        if value[1] in ".)" and len(value) > 2 and value[2].isspace() and value[3:].strip():
            return value[:2], leading
        return None
    index = 0
    while index < len(value):
        char = value[index]
        category = unicodedata.category(char)
        if char in _LIST_MARKER_EXCLUDED or category[0] not in {"P", "S", "C"}:
            break
        index += 1
    if index == 0 or index >= len(value) or not value[index].isspace():
        return None
    if not value[index + 1:].strip():
        return None
    return value[:index], leading


def _list_marker_style(marker: str) -> str:
    if marker and marker[0].isdigit():
        return "decimal"
    if len(marker) == 2 and marker[0].isascii() and marker[0].isalpha():
        return "alpha"
    return marker


def _page_navigation_gap(lines: Iterable[str]) -> bool:
    """Return true only for parser-inserted page navigation scaffolding.

    Coalesced PDF chunks can place a Markdown page marker and a standalone
    printed page number between two source list items.  Those lines are
    provenance/navigation, not a semantic section boundary, so they may be
    transparent to list depth.  A normal heading, prose line, caption or bare
    number without a page heading still terminates the list.
    """

    values = [str(line).strip() for line in lines if str(line).strip()]
    return bool(
        values
        and _PAGE_NAVIGATION_HEADING.fullmatch(values[0])
        and all(
            _STANDALONE_PAGE_NUMBER.fullmatch(value)
            for value in values[1:]
        )
    )


def _list_item_scopes(
    lines: Iterable[tuple[int, int, int, str]],
    *,
    block_key: str,
) -> list[dict[str, Any]]:
    """Build auditable list-item scopes and syntax-derived parent links."""

    source_lines = list(lines)
    line_text_by_index = {
        int(line_index): str(line)
        for line_index, _start, _end, line in source_lines
    }
    marked: list[tuple[int, int, int, str, str, int]] = []
    for line_index, start, end, line in source_lines:
        marker = _list_marker(line)
        if marker is None:
            continue
        marker_text, indent = marker
        marked.append((line_index, start, end, line, marker_text, indent))
    groups: list[list[tuple[int, int, int, str, str, int]]] = []
    styles = [_list_marker_style(item[4]) for item in marked]
    for index, item in enumerate(marked):
        previous_line_index = marked[index - 1][0] if index else -1
        page_continuation = bool(
            index > 0
            and item[0] > previous_line_index + 1
            and _page_navigation_gap(
                line_text_by_index.get(line_index, "")
                for line_index in range(previous_line_index + 1, item[0])
            )
        )
        starts_new_root = bool(
            index > 0 and index + 2 < len(marked)
            and item[0] == marked[index - 1][0] + 1
            and styles[index] != styles[index - 1]
            and styles[index + 1] == styles[index - 1]
            and styles[index + 2] == styles[index - 1]
        )
        if (
            not groups
            or (
                item[0] != groups[-1][-1][0] + 1
                and not page_continuation
            )
            or starts_new_root
        ):
            groups.append([])
        groups[-1].append(item)

    scopes: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        last_by_depth: dict[int, str] = {}
        style_depth: dict[str, int] = {}
        indent_depth: dict[int, int] = {}
        previous_depth = 0
        previous_indent = 0
        previous_style = ""
        for item_index, (_line_index, start, end, _line, marker, indent) in enumerate(group):
            style = _list_marker_style(marker)
            if not last_by_depth:
                depth = 0
            elif indent > previous_indent:
                depth = previous_depth + 1
            elif indent < previous_indent:
                # Explicit source indentation is stronger evidence than a
                # reused marker glyph.  PDF extraction commonly renders every
                # level with the same em dash; looking up that glyph in
                # ``style_depth`` made only the first indented child a child
                # and incorrectly promoted all of its siblings back to the
                # root.  Prefer the exact indentation level, with a
                # conservative nearest-shallower fallback for irregular
                # whitespace.
                if indent in indent_depth:
                    depth = indent_depth[indent]
                else:
                    shallower = [
                        (known_indent, known_depth)
                        for known_indent, known_depth in indent_depth.items()
                        if known_indent < indent
                    ]
                    depth = (
                        max(shallower, key=lambda item: item[0])[1] + 1
                        if shallower else 0
                    )
            elif style == previous_style:
                # Same indentation and marker means a sibling, including when
                # the marker is also used by a shallower level.
                depth = previous_depth
            elif style in style_depth:
                # With no indentation signal, a recurring marker style can
                # still close a nested PDF-symbol list back to its prior level.
                depth = style_depth[style]
            else:
                depth = previous_depth + 1
            depth = max(0, min(depth, len(last_by_depth) + 1))
            scope_id = f"{block_key}:list-item:{group_index}:{item_index}"
            parent_scope_id = last_by_depth.get(depth - 1, "") if depth else ""
            scopes.append({
                "start": start,
                "end": end,
                "source_role": "list_item",
                "scope_id": scope_id,
                "scope_kind": "list_item",
                "list_group_id": f"{block_key}:list-group:{group_index}",
                "list_depth": depth,
                "list_marker": marker,
                "list_parent_scope_id": parent_scope_id,
            })
            last_by_depth[depth] = scope_id
            for stale_depth in [value for value in last_by_depth if value > depth]:
                del last_by_depth[stale_depth]
            # Indentation is allowed to refine a marker style's depth.  Keep
            # the first style depth only for unindented symbol hierarchies;
            # equal-style siblings above use ``previous_depth`` directly.
            indent_depth[indent] = depth
            style_depth.setdefault(style, depth)
            previous_depth = depth
            previous_indent = indent
            previous_style = style
    return scopes


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def source_role_for_structure(
    structure: Mapping[str, Any] | None = None,
    *,
    context_kind: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Return a conservative parser-role classification without semantic cues."""

    structure = structure if isinstance(structure, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    raw = _first_text(
        structure.get("source_role"),
        structure.get("structural_role"),
        metadata.get("structural_role"),
        metadata.get("source_role"),
        metadata.get("cell_role"),
        context_kind,
        structure.get("block_type"),
        metadata.get("context_kind"),
    ).lower().replace(" ", "_")
    return _SOURCE_ROLE_ALIASES.get(raw, "unknown")


def relation_eligible_source_role(role: str) -> bool:
    return str(role or "") in _RELATION_ELIGIBLE_SOURCE_ROLES


def structural_profile(
    *,
    content: str,
    structure: Mapping[str, Any] | None = None,
    context_kind: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe source eligibility and deterministic atomic evidence scopes.

    The profile has no knowledge of predicates or domains.  A body unit is
    divided into sentences; a list item, table row and table cell stay atomic.
    Older chunk-local source has sentence scopes too, so its lower structural
    fidelity never silently opens a cross-sentence relation path.
    """

    role = source_role_for_structure(
        structure, context_kind=context_kind, metadata=metadata,
    )
    normalized_kind = str(context_kind or "").strip().lower().replace("-", "_")
    atomic_kind = (
        role if role in {"list_item", "table_row", "table_cell"}
        else "sentence"
    )
    scopes: list[dict[str, Any]] = []
    if atomic_kind == "sentence":
        for index, (start, end, _text) in enumerate(
            split_relation_scopes(content)
        ):
            scope_kind = (
                "clause"
                if content[end - 1:end] in {";", "；"}
                or content[start - 1:start] in {";", "；"}
                else "sentence"
            )
            scopes.append({
                "scope_id": f"relation_scope:{index}:{start}:{end}",
                "scope_kind": scope_kind,
                "start": int(start),
                "end": int(end),
            })
    if not scopes and content:
        scopes.append({
            "scope_id": f"{atomic_kind}:0:{len(content)}",
            "scope_kind": atomic_kind,
            "start": 0,
            "end": len(content),
        })
    return {
        "policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
        "source_role": role,
        "relation_eligible": relation_eligible_source_role(role),
        "context_kind": normalized_kind or "legacy_chunk",
        "atomic_scope_kind": atomic_kind,
        "scopes": scopes,
    }


def occurrence_scope(
    *,
    content: str,
    mention_start: int,
    mention_end: int,
    structure: Mapping[str, Any] | None = None,
    context_kind: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the one exact atomic scope containing an occurrence.

    An occurrence that crosses parser/sentence scope is marked ineligible
    rather than guessed into a neighboring scope.
    """

    profile = structural_profile(
        content=content,
        structure=structure,
        context_kind=context_kind,
        metadata=metadata,
    )
    start, end = int(mention_start), int(mention_end)
    structure = structure if isinstance(structure, Mapping) else {}
    structural_spans = structure.get("structural_spans") or []
    for raw_scope in structural_spans:
        if not isinstance(raw_scope, Mapping):
            continue
        try:
            scope_start = int(raw_scope["start"])
            scope_end = int(raw_scope["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if scope_start <= start < end <= scope_end:
            role = str(raw_scope.get("source_role") or "unknown")
            return {
                **profile,
                "source_role": role,
                "relation_eligible": relation_eligible_source_role(role),
                "occurrence_scope": {
                    "scope_id": str(
                        raw_scope.get("scope_id")
                        or f"mapped:{scope_start}:{scope_end}"
                    ),
                    "scope_kind": str(raw_scope.get("scope_kind") or role),
                    "start": scope_start,
                    "end": scope_end,
                    "list_group_id": str(raw_scope.get("list_group_id") or ""),
                    "list_depth": int(raw_scope.get("list_depth") or 0),
                    "list_marker": str(raw_scope.get("list_marker") or ""),
                    "list_parent_scope_id": str(
                        raw_scope.get("list_parent_scope_id") or ""
                    ),
                },
            }
    if any(
        isinstance(raw_scope, Mapping)
        and int(raw_scope.get("start") or -1) < end
        and int(raw_scope.get("end") or -1) > start
        for raw_scope in structural_spans
    ):
        return {
            **profile,
            "relation_eligible": False,
            "occurrence_scope": {
                "scope_id": "invalid:crosses-structural-span",
                "scope_kind": "invalid",
                "start": start,
                "end": end,
            },
        }
    for scope in profile["scopes"]:
        if int(scope["start"]) <= start < end <= int(scope["end"]):
            return {**profile, "occurrence_scope": dict(scope)}
    return {
        **profile,
        "relation_eligible": False,
        "occurrence_scope": {
            "scope_id": "invalid:span",
            "scope_kind": "invalid",
            "start": start,
            "end": end,
        },
    }


def _line_spans(text: str, offset: int) -> Iterable[tuple[int, int, str]]:
    cursor = 0
    for line in text.splitlines(keepends=True):
        line_end = cursor + len(line)
        body_length = len(line.rstrip("\r\n"))
        if body_length:
            stripped_end = cursor + body_length
            yield offset + cursor, offset + stripped_end, line[:body_length]
        cursor = line_end
    if not text:
        return


def _table_or_list_scopes(
    *,
    block: Mapping[str, Any],
    block_start: int,
    block_end: int,
    block_key: str,
) -> list[dict[str, Any]]:
    """Turn layout rows/items into auditable source spans without domain rules."""

    text = str(block.get("text") or "")
    role = source_role_for_structure(
        {"block_type": block.get("block_type")},
        metadata=_metadata(block.get("metadata")),
    )
    spans: list[dict[str, Any]] = []
    lines = list(_line_spans(text, block_start))
    if role == "table_container":
        for index, (start, end, line) in enumerate(lines):
            # The first row carries column labels.  A second structural row is
            # a header only when it is an actual Markdown alignment delimiter
            # (``| --- | :---: |``).  Some parser inputs use pipe-delimited
            # tables without that optional delimiter; treating their first
            # data row as a header removed every endpoint unique to that row.
            delimiter_cells = [
                value.strip()
                for value in line.strip().strip("|").split("|")
            ]
            markdown_delimiter = bool(
                delimiter_cells
                and all(
                    re.fullmatch(r":?-{3,}:?", value) is not None
                    for value in delimiter_cells
                )
            )
            row_role = (
                "table_header"
                if index == 0 or markdown_delimiter
                else "table_row"
            )
            spans.append({
                "start": start,
                "end": end,
                "source_role": row_role,
                "scope_id": f"{block_key}:table-row:{index}",
                "scope_kind": row_role,
            })
        return spans
    list_scopes = _list_item_scopes(
        [(index, start, end, line) for index, (start, end, line) in enumerate(lines)],
        block_key=block_key,
    )
    if list_scopes:
        return list_scopes
    if role == "body":
        for index, (start, end, _sentence) in enumerate(
            split_relation_scopes(text)
        ):
            if start == end:
                continue
            scope_kind = (
                "clause"
                if text[end - 1:end] in {";", "；"}
                or text[start - 1:start] in {";", "；"}
                else "sentence"
            )
            spans.append({
                "start": block_start + start,
                "end": block_start + end,
                "source_role": "body",
                "scope_id": f"{block_key}:relation-scope:{index}",
                "scope_kind": scope_kind,
            })
        if spans:
            return spans
    spans.append({
        "start": block_start,
        "end": block_end,
        "source_role": role,
        "scope_id": f"{block_key}:{role}:0",
        "scope_kind": role,
    })
    return spans


def structural_spans_for_chunk(
    *,
    content: str,
    blocks: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Map parser blocks to a persisted retrieval chunk only when exact.

    The map intentionally does not use fuzzy text matching.  Unmapped content
    stays body/sentence-scoped; an ambiguous parser block cannot be guessed
    into another region and thereby alter factual extraction.
    """

    spans: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for index, block in enumerate(blocks, start=1):
        text = str(block.get("text") or "")
        if not text:
            continue
        start_at = 0
        match_start = -1
        while True:
            candidate = content.find(text, start_at)
            if candidate < 0:
                break
            candidate_end = candidate + len(text)
            if not any(
                candidate < end and candidate_end > start
                for start, end in occupied
            ):
                match_start = candidate
                occupied.append((candidate, candidate_end))
                break
            start_at = candidate + 1
        if match_start < 0:
            continue
        block_key = _block_key(block, index)
        spans.extend(_table_or_list_scopes(
            block=block,
            block_start=match_start,
            block_end=match_start + len(text),
            block_key=block_key,
        ))
    # List hierarchy belongs to the rendered chunk, not to an arbitrary parser
    # block boundary.  A coalesced PDF chunk can contain the tail of one page
    # as locally recovered text and the next page as an exactly mapped block.
    # Parsing those regions independently loses the active parent across a
    # transparent page marker.  Reparse all non-table lines as one immutable
    # syntax stream and replace block-local list scopes with that global view.
    # This remains syntax-only: ordinary prose is a group boundary and no
    # relation meaning or domain vocabulary is consulted.
    #
    # Table rows are excluded because a pipe followed by whitespace can look
    # like a symbol list marker.  Their parser-owned row/header scopes remain
    # authoritative.
    table_ranges = [
        (int(item["start"]), int(item["end"]))
        for item in spans
        if str(item.get("source_role") or "") in {
            "table_header", "table_row", "table_cell",
        }
    ]
    chunk_list_lines = [
        (index, start, end, line)
        for index, (start, end, line) in enumerate(_line_spans(content, 0))
        if not any(
            start < reserved_end and end > reserved_start
            for reserved_start, reserved_end in table_ranges
        )
    ]
    chunk_list_scopes = _list_item_scopes(
        chunk_list_lines,
        block_key="chunk",
    )
    if chunk_list_scopes:
        spans = [
            item for item in spans
            if str(item.get("scope_kind") or "") != "list_item"
        ]
        spans.extend(chunk_list_scopes)
    return sorted(
        spans,
        key=lambda item: (int(item["start"]), int(item["end"]), str(item["scope_id"])),
    )


@dataclass(frozen=True)
class SourceMapEntry:
    rendered_start: int
    rendered_end: int
    anchor_kind: str
    block_key: str
    block_local_start: int
    block_local_end: int
    table_id: str = ""
    row_id: str = ""
    cell_id: str = ""
    page: int | None = None


@dataclass(frozen=True)
class ContextUnit:
    context_id: str
    namespace_id: str
    document_revision_id: str
    context_kind: str
    rendered_text: str
    rendered_hash: str
    source_block_keys: tuple[str, ...]
    source_map: tuple[SourceMapEntry, ...]
    structure: dict[str, Any] = field(default_factory=dict)
    structure_quality: str = "structured"


@dataclass(frozen=True)
class ShadowContextManifest:
    policy_version: str
    source_map_policy: str
    artifact_hash: str
    context_units: tuple[ContextUnit, ...]
    summary: dict[str, Any]

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "source_map_policy": self.source_map_policy,
            "artifact_hash": self.artifact_hash,
            "summary": self.summary,
        }


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _block_key(block: Mapping[str, Any], order: int) -> str:
    return str(block.get("key") or block.get("block_key") or f"block-{order:04d}")


def _metadata(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _context_kind(block: Mapping[str, Any]) -> str:
    metadata = _metadata(block.get("metadata"))
    kind = str(
        metadata.get("context_kind")
        or metadata.get("role")
        or block.get("block_type")
        or "paragraph"
    ).strip().lower()
    if kind in {"table-row", "table row", "row"}:
        return "table_row"
    if kind in {"list-item", "list item"}:
        return "list_item"
    return kind or "paragraph"


def _source_entry(
    *,
    rendered_text: str,
    block: Mapping[str, Any],
    block_key: str,
) -> SourceMapEntry:
    metadata = _metadata(block.get("metadata"))
    return SourceMapEntry(
        rendered_start=0,
        rendered_end=len(rendered_text),
        anchor_kind=str(metadata.get("anchor_kind") or _context_kind(block)),
        block_key=block_key,
        block_local_start=int(metadata.get("block_local_start") or 0),
        block_local_end=int(
            metadata.get("block_local_end")
            if metadata.get("block_local_end") is not None
            else len(str(block.get("text") or ""))
        ),
        table_id=str(metadata.get("table_id") or ""),
        row_id=str(metadata.get("row_id") or ""),
        cell_id=str(metadata.get("cell_id") or ""),
        page=(
            int(block["page"])
            if block.get("page") is not None and str(block.get("page")).isdigit()
            else None
        ),
    )


def _block_structure(block: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _metadata(block.get("metadata"))
    base = {
        "block_key": _block_key(block, int(block.get("order") or 0)),
        "block_type": str(block.get("block_type") or ""),
        "section_key": str(block.get("section_key") or ""),
        "section_title": str(block.get("section_title") or ""),
        "heading_path": list(block.get("heading_path") or []),
        "page": block.get("page"),
        "table_id": str(metadata.get("table_id") or ""),
        "row_id": str(metadata.get("row_id") or ""),
        "cell_id": str(metadata.get("cell_id") or ""),
        "header_path": list(metadata.get("header_path") or metadata.get("headers") or []),
        "cell_role": str(metadata.get("cell_role") or ""),
        "unit": str(metadata.get("unit") or ""),
        "estimated": bool(metadata.get("estimated") or False),
        "parser_version": str(payload.get("parser_version") or ""),
        "chunk_id": int(
            metadata.get("chunk_id")
            or metadata.get("source_chunk_id")
            or block.get("chunk_id")
            or 0
        ),
    }
    role = source_role_for_structure(base, metadata=metadata)
    return {
        **base,
        "source_role": role,
        "relation_eligible": relation_eligible_source_role(role),
        "structure_policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
    }


def _unit(
    *,
    namespace_id: str,
    revision_id: str,
    context_kind: str,
    rendered_text: str,
    source_entry: SourceMapEntry,
    source_block_keys: tuple[str, ...],
    structure: dict[str, Any],
    structure_quality: str,
) -> ContextUnit:
    rendered_hash = _digest(rendered_text)
    # ``context_id`` identifies the frozen physical source context.  It must
    # stay identical when the entity and relation stages independently rebuild
    # the same manifest under different runtime namespace labels.  Isolation
    # is carried by ContextUnit.namespace_id and the owning run/namespace rows,
    # not by changing the physical context identity.
    context_id = "shadowctx_" + _digest({
        "revision_id": revision_id,
        "context_kind": context_kind,
        "rendered_hash": rendered_hash,
        "source_block_keys": source_block_keys,
        "source_map": [asdict(source_entry)],
        "source_role": str(structure.get("source_role") or ""),
        "relation_eligible": bool(structure.get("relation_eligible")),
        "structure_policy_version": str(
            structure.get("structure_policy_version") or ""
        ),
        "policy": SHADOW_CONTEXT_POLICY_VERSION,
    })
    return ContextUnit(
        context_id=context_id,
        namespace_id=namespace_id,
        document_revision_id=revision_id,
        context_kind=context_kind,
        rendered_text=rendered_text,
        rendered_hash=rendered_hash,
        source_block_keys=source_block_keys,
        source_map=(source_entry,),
        structure=structure,
        structure_quality=structure_quality,
    )


def _chunk_mapped_units(
    *,
    namespace_id: str,
    revision_id: str,
    chunks: Iterable[Mapping[str, Any]],
    blocks: Iterable[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> tuple[ContextUnit, ...]:
    block_values = [item for item in blocks if isinstance(item, Mapping)]
    units: list[ContextUnit] = []
    for chunk in chunks:
        rendered_text = str(chunk.get("content") or "")
        if not rendered_text.strip():
            continue
        chunk_id = int(chunk.get("id") or 0)
        chunk_key = str(chunk.get("chunk_key") or chunk_id or "chunk")
        spans = structural_spans_for_chunk(
            content=rendered_text,
            blocks=block_values,
        )
        source_entry = SourceMapEntry(
            rendered_start=0,
            rendered_end=len(rendered_text),
            anchor_kind="structured_chunk",
            block_key=chunk_key,
            block_local_start=0,
            block_local_end=len(rendered_text),
        )
        units.append(_unit(
            namespace_id=namespace_id,
            revision_id=str(chunk.get("revision_id") or revision_id),
            context_kind="structured_chunk",
            rendered_text=rendered_text,
            source_entry=source_entry,
            source_block_keys=tuple(sorted({
                str(item["scope_id"]).split(":", 1)[0]
                for item in spans
            })) or (chunk_key,),
            structure={
                "chunk_id": chunk_id,
                "chunk_key": chunk_key,
                "section_title": str(chunk.get("section_title") or ""),
                "block_type": "chunk",
                "source_role": "body",
                "relation_eligible": True,
                "structural_spans": spans,
                "parser_version": str(payload.get("parser_version") or ""),
                "structure_policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
            },
            structure_quality="structured_chunk_mapped",
        ))
    return tuple(units)


def build_context_units_for_document(
    *,
    namespace_id: str,
    document: Mapping[str, Any],
    chunks: Iterable[Mapping[str, Any]],
    processing_payload: Mapping[str, Any] | None,
) -> tuple[ContextUnit, ...]:
    """Build enhanced-only ContextUnits from persisted structure payloads.

    When historical documents lack parser blocks/source maps, fall back to
    chunk-local units and explicitly mark the degraded quality.
    """

    payload = processing_payload if isinstance(processing_payload, Mapping) else {}
    revision_id = str(document.get("active_revision_id") or document.get("revision_id") or "")
    chunk_values = list(chunks)
    blocks = [
        item for item in payload.get("blocks") or []
        if isinstance(item, Mapping) and str(item.get("text") or "").strip()
    ]
    if blocks:
        mapped = _chunk_mapped_units(
            namespace_id=namespace_id,
            revision_id=revision_id,
            chunks=chunk_values,
            blocks=blocks,
            payload=payload,
        )
        if mapped:
            return mapped
        units = []
        for index, block in enumerate(blocks, start=1):
            rendered_text = str(block.get("text") or "")
            block_key = _block_key(block, index)
            source_entry = _source_entry(
                rendered_text=rendered_text, block=block, block_key=block_key,
            )
            units.append(_unit(
                namespace_id=namespace_id,
                revision_id=revision_id,
                context_kind=_context_kind(block),
                rendered_text=rendered_text,
                source_entry=source_entry,
                source_block_keys=(block_key,),
                structure=_block_structure(block, payload),
                structure_quality="structured",
            ))
        return tuple(units)

    fallback_units = []
    for chunk in chunk_values:
        rendered_text = str(chunk.get("content") or "")
        if not rendered_text.strip():
            continue
        chunk_key = str(chunk.get("chunk_key") or chunk.get("id") or "chunk")
        source_entry = SourceMapEntry(
            rendered_start=0,
            rendered_end=len(rendered_text),
            anchor_kind="legacy_chunk_local",
            block_key=chunk_key,
            block_local_start=0,
            block_local_end=len(rendered_text),
            page=None,
        )
        fallback_units.append(_unit(
            namespace_id=namespace_id,
            revision_id=str(chunk.get("revision_id") or revision_id),
            context_kind="legacy_chunk",
            rendered_text=rendered_text,
            source_entry=source_entry,
            source_block_keys=(chunk_key,),
            structure={
                "chunk_id": int(chunk.get("id") or 0),
                "chunk_key": chunk_key,
                "section_title": str(chunk.get("section_title") or ""),
                "source_map_quality": "legacy_chunk_local",
                "source_role": "body",
                "relation_eligible": True,
                "structure_policy_version": KNOWLEDGE_FORGE_STRUCTURE_POLICY_VERSION,
            },
            structure_quality="legacy_chunk_local",
        ))
    return tuple(fallback_units)


def build_shadow_context_manifest(
    *,
    namespace_id: str,
    documents: Iterable[Mapping[str, Any]],
    chunks_by_document: Mapping[int, Iterable[Mapping[str, Any]]],
    payload_loader: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
) -> ShadowContextManifest:
    """Build a deterministic structure/source-map manifest for enhanced Shadow."""

    all_units: list[ContextUnit] = []
    document_summaries = []
    for document in sorted(documents, key=lambda item: int(item["id"])):
        document_id = int(document["id"])
        chunks = list(chunks_by_document.get(document_id) or [])
        payload = payload_loader(document)
        units = build_context_units_for_document(
            namespace_id=namespace_id,
            document=document,
            chunks=chunks,
            processing_payload=payload,
        )
        all_units.extend(units)
        document_summaries.append({
            "document_id": document_id,
            "revision_id": str(document.get("active_revision_id") or ""),
            "content_hash": str(document.get("doc_hash") or ""),
            "parser_version": str((payload or {}).get("parser_version") or ""),
            "block_count": len((payload or {}).get("blocks") or []),
            "context_unit_count": len(units),
            "structure_quality": sorted({
                unit.structure_quality for unit in units
            }),
        })
    units_tuple = tuple(sorted(all_units, key=lambda item: item.context_id))
    artifact_payload = {
        "policy_version": SHADOW_CONTEXT_POLICY_VERSION,
        "source_map_policy": SHADOW_CONTEXT_SOURCE_MAP_POLICY,
        "documents": document_summaries,
        "contexts": [{
            "context_id": unit.context_id,
            "revision_id": unit.document_revision_id,
            "kind": unit.context_kind,
            "rendered_hash": unit.rendered_hash,
            "source_block_keys": list(unit.source_block_keys),
            "source_map": [asdict(item) for item in unit.source_map],
            "structure_quality": unit.structure_quality,
            "source_role": str(unit.structure.get("source_role") or ""),
            "relation_eligible": bool(unit.structure.get("relation_eligible")),
            "structure_policy_version": str(
                unit.structure.get("structure_policy_version") or ""
            ),
            "structure_hash": _digest(unit.structure),
        } for unit in units_tuple],
    }
    summary = {
        "documents": len(document_summaries),
        "context_units": len(units_tuple),
        "source_map_entries": sum(len(unit.source_map) for unit in units_tuple),
        "legacy_chunk_local_units": sum(
            1 for unit in units_tuple
            if unit.structure_quality == "legacy_chunk_local"
        ),
        "context_kinds": sorted({unit.context_kind for unit in units_tuple}),
        "relation_eligible_contexts": sum(
            int(bool(unit.structure.get("relation_eligible")))
            for unit in units_tuple
        ),
        "relation_ineligible_contexts": sum(
            int(not bool(unit.structure.get("relation_eligible")))
            for unit in units_tuple
        ),
    }
    return ShadowContextManifest(
        policy_version=SHADOW_CONTEXT_POLICY_VERSION,
        source_map_policy=SHADOW_CONTEXT_SOURCE_MAP_POLICY,
        artifact_hash=_digest(artifact_payload),
        context_units=units_tuple,
        summary=summary,
    )
