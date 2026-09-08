"""Low-cost semantic projection for already verified relation facts.

This layer never changes direction, endpoints or the canonical predicate.  It
only records whether a fact is backed by a versioned Schema predicate or by an
open source expression, plus a conservative morphology-normalized family key
for grouping repeated surface variants.  Synonyms are deliberately not
guessed and no additional model call is introduced.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Mapping


RELATION_SEMANTIC_PROJECTION_VERSION = (
    "relation-semantic-projection-source-family-v11-role-marker-gate"
)

_ASCII_TOKEN = re.compile(r"^[a-z]+$")
_ENGLISH_FAMILY_FUNCTION_WORDS = frozenset({
    "a", "an", "the", "be", "have", "do",
})
_ENGLISH_RELATION_FUNCTION_WORDS = frozenset({
    *_ENGLISH_FAMILY_FUNCTION_WORDS,
    "of", "to", "with", "by", "as", "at", "in", "on", "for", "from",
    "into", "onto", "through", "between", "among",
    "and", "or", "but", "if", "then", "than", "that", "which", "who",
    "can", "could", "may", "might", "must", "shall", "should", "will",
    "would", "this", "that", "these", "those", "it", "its", "other",
    "another", "same",
})
_CJK_RELATION_FUNCTION_WORD = re.compile(
    r"^(?:的|地|得|在|于|和|与|及|或|为|是|由|被|对|向|从|至|以|将|"
    r"并|而|但)$"
)
_CJK_INCOMPLETE_RELATION = re.compile(
    r"^(?:的|地|得)|(?:的|得|及|和|与|或|并|且|而|但|则)$"
)
_SENTENCE_PUNCTUATION = re.compile(r"[。！？!?；;，,:：]")
_SYNTHETIC_ELLIPSIS = re.compile(r"(?:\.{3,}|…{1,2})")
_ENGLISH_NOMINAL_RELATION = re.compile(
    r"^[a-z]+(?:ness|ment|ity|tion|sion|ship|ance|ence)$",
    re.IGNORECASE,
)
_ENGLISH_TRAILING_LINK_WORDS = frozenset({
    "of", "to", "with", "by", "as", "at", "in", "on", "for", "from",
    "into", "onto", "through",
})
_CJK_LEADING_ARGUMENT_FRAGMENT = re.compile(
    r"^(?:在|于|向|从|对|由|以)[\u3400-\u9fff]{1,4}$"
)
_CJK_LOCATIVE_BOUNDARY_FRAGMENT = re.compile(
    r"^(?:中|上|下|内|外)(?:来|去|中|上|下|进行|实现)$"
)
_CJK_LOCATIVE_ADVERB_BOUNDARY_FRAGMENT = re.compile(
    r"^(?:中|上|下|内|外)(?:均|都|皆|也|亦|仍|还|分别|共同|普遍|通常|"
    r"往往|持续|显著|明显|更|最|只|仅)[\u3400-\u9fff]{1,8}$"
)
_ENGLISH_TITLECASE_BARE_FRAGMENT = re.compile(r"^[A-Z][a-z]{2,}$")
_ENGLISH_INFLECTED_VERB_SHAPE = re.compile(
    r"(?:ed|ing|s|izes|ises|ifies)$", re.IGNORECASE,
)
_CJK_TRAILING_ARGUMENT_DETERMINER = re.compile(
    r"(?<=[\u3400-\u9fff]{2})(?:各|该|此|其)$"
)
_RENDERED_EDGE_RELATION_EXPRESSION = re.compile(
    r"^\s*[-=.]{1,4}\s*(?P<label>[^\n<>]{1,80}?)\s*[-=.]{1,4}>\s*$"
)
_FORMULA_OPERATOR_RELATION = re.compile(
    r"^(?:sin|cos|tan|cot|log|ln|exp|sqrt|arg\s*min|arg\s*max|min|max)$",
    re.IGNORECASE,
)
_FORMULA_SYNTAX = re.compile(r"[=×÷∩∪∑∏√∥‖≤≥≈≠∈∉]|\btheta\b|[θλΣ]")
_CJK_ENUMERATION_RELATION = re.compile(
    r"^(?:(?:一|二|三|四|五|六|七|八|九|十)(?:是|为|有)?|"
    r"首先|其次|再次|最后)$"
)
_CJK_NON_RELATIONAL_MARKER = re.compile(
    r"^(?:其他|其它|典型|进行|还是|或是|以及|之间|间|情况下|条件下|过程中)$|"
    r"^如(?:包含|包括)$"
)
_CJK_DANGLING_ARGUMENT_MARKER = re.compile(r"(?:因|所|如|其)$")
_CJK_NOMINAL_LOCATIVE_FRAGMENT = re.compile(
    r"(?:方法|系统|模型|算法|过程|机制|策略|结果|数据|信息|功能|"
    r"能力|方案|结构|框架|技术|任务|活动|关系|条件|场景)在$"
)
_CJK_INCOMPLETE_LOCATIVE_STATE = re.compile(
    r"(?:始终|一直|通常|仍然|依然|长期)处$"
)
_CJK_INCOMPLETE_COMPARATIVE = re.compile(
    r"(?:将|会|可|能)?以(?:更|较|最高|最低)?(?:高|低|大|小|多|少)$"
)
_CJK_LOCATIVE_EDGE_PREFIX = re.compile(r"^[中内上下外]")
_CJK_LAYOUT_FUNCTION_PREFIX = re.compile(
    r"^(?:(?:并且|而且|随后|然后|同时|已经|正在|仍然|依然|"
    r"应当|应该|必须|需要|可以|能够|可能)|"
    r"[的地得和与及或并且而将把在于向对给由被可会能应须需已正不未仅只再也都均])+$"
)
_CJK_INCOMPLETE_FUNCTIONAL_PREFIX = re.compile(
    r"^(?:(?:主要)?原因是|通过|由于|因为|最终|首先|其次|然后|同时|"
    r"其中|是在|方法在|过程中在|程中在)(?:在|于|向|从|对|由|以)?$"
)
_CJK_TRUNCATED_PREPOSITIONAL_RELATION = re.compile(
    r"^(?:中|程中|过程中).{1,18}(?:在|于|向|从|对|由|以|通过|基于)$"
)
_FIGURE_CAPTION_RELATION_SCAFFOLD = re.compile(
    r"^(?:(?:figure|fig\.?|table)\s*[A-Za-z]?\d+(?:[.\-]\d+)*\s+)?"
    r"(?:plot|chart)\s+of$",
    re.IGNORECASE,
)
_MEDIA_VISUAL_ENCODING_RELATION = re.compile(
    r"^(?:display(?:s|ed|ing)?|plot(?:s|ted|ting)?(?:\s+using)?|"
    r"is\s+plotted\s+using|represented\s+by|shown\s+(?:as|with)|"
    r"drawn\s+(?:as|using)|colou?red(?:\s+(?:as|with))?)$",
    re.IGNORECASE,
)
_UNKNOWN_SCHEMA_ENTITY_TYPES = frozenset({
    "", "*", "any", "entity", "genericentity", "generic entity",
    "unknown", "unspecified", "任意类型", "通用实体",
})
_SOURCE_PROJECTION_CANDIDATES = frozenset({
    "ATOMIC_FRAME_SCHEMA_PROJECTION",
    "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
})


def _light_english_lemma(token: str) -> str:
    """Normalize common inflection shapes without semantic interpretation."""

    irregular = {
        "has": "have",
        "does": "do",
        "goes": "go",
        "is": "be",
        "are": "be",
        "was": "be",
        "were": "be",
        "sends": "send",
        "sent": "send",
        "authored": "author",
        "associated": "associate",
        "caused": "cause",
        "created": "create",
        "defined": "define",
        "included": "include",
        "provided": "provide",
        "required": "require",
        "supported": "support",
        "used": "use",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith((
        "ches", "shes", "sses", "xes", "zes", "oes",
    )):
        return token[:-2]
    if len(token) > 3 and token.endswith("ses"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalized_relation_family_key(value: Any) -> str:
    """Return a stable lexical family without collapsing different meanings."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = re.sub(r"[_‐‑‒–—-]+", " ", normalized)
    tokens = re.findall(r"[a-z]+|[\u3400-\u9fff]+|[0-9]+", normalized)
    if not tokens:
        return ""
    if all(_ASCII_TOKEN.fullmatch(token) for token in tokens):
        tokens = [_light_english_lemma(token) for token in tokens]
        lexical = [
            token for token in tokens
            if token not in _ENGLISH_FAMILY_FUNCTION_WORDS
        ]
        if lexical:
            tokens = lexical
    return "|".join(tokens)


def source_expression_matches_schema_cues(
    expression: Any,
    cues: Any,
) -> bool:
    """Return a conservative lexical bridge supplied by the active Schema.

    Publication must not invent a synonym between an open source predicate and
    a normalized Schema predicate.  This helper accepts only a cue explicitly
    stored in the versioned Schema, while normalizing case, layout whitespace,
    light English inflection and harmless modal/auxiliary material.  It never
    branches on a predicate identifier or domain vocabulary.
    """

    source_family = normalized_relation_family_key(expression)
    source_lexical_words = [
        _light_english_lemma(value)
        for value in re.findall(
            r"[a-z]+",
            unicodedata.normalize(
                "NFKC", str(expression or ""),
            ).casefold(),
        )
        if _light_english_lemma(value)
        not in _ENGLISH_RELATION_FUNCTION_WORDS
    ]
    source_compact = re.sub(
        r"[^0-9a-z\u3400-\u9fff]+",
        "",
        unicodedata.normalize("NFKC", str(expression or "")).casefold(),
    )
    if not source_family or not source_compact:
        return False
    values = cues if isinstance(cues, (list, tuple, set, frozenset)) else ()
    for raw_cue in values:
        cue_family = normalized_relation_family_key(raw_cue)
        cue_compact = re.sub(
            r"[^0-9a-z\u3400-\u9fff]+",
            "",
            unicodedata.normalize("NFKC", str(raw_cue or "")).casefold(),
        )
        if not cue_family or not cue_compact:
            continue
        if source_family == cue_family:
            return True
        cue_lexical_words = [
            _light_english_lemma(value)
            for value in re.findall(r"[a-z]+", cue_compact)
            if _light_english_lemma(value)
            not in _ENGLISH_RELATION_FUNCTION_WORDS
        ]
        if (
            source_lexical_words
            and cue_lexical_words
            and source_lexical_words == cue_lexical_words
        ):
            return True
        contains_cjk = bool(re.search(r"[\u3400-\u9fff]", cue_compact))
        if (
            contains_cjk
            and len(cue_compact) >= 2
            and (
                cue_compact in source_compact
                or (
                    len(source_compact) >= 2
                    and source_compact in cue_compact
                )
            )
        ):
            return True
    return False


def _compact_layout_with_positions(value: Any) -> tuple[str, list[int], str]:
    source = unicodedata.normalize("NFKC", str(value or ""))
    compact: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(source):
        if character.isspace():
            continue
        compact.append(character.casefold())
        positions.append(index)
    return "".join(compact), positions, source


def _all_compact_occurrences(source: str, value: str) -> list[int]:
    if not source or not value:
        return []
    output: list[int] = []
    start = 0
    while True:
        index = source.find(value, start)
        if index < 0:
            return output
        output.append(index)
        start = index + 1


def _ordered_relation_geometry(
    expression: str, metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Locate subject, predicate and object in one frozen source frame.

    The helper ignores layout whitespace only.  It does not tokenize, infer a
    predicate, or change argument order.  Its sole purpose is to identify when
    a parser-selected CJK token starts in the middle of intervening material.
    """

    evidence = str(metadata.get("source_frame_evidence_text") or "")
    subject = str(metadata.get("source_subject_surface") or "")
    object_ = str(metadata.get("source_object_surface") or "")
    compact, positions, normalized_source = _compact_layout_with_positions(
        evidence
    )
    compact_expression = _compact_layout_with_positions(expression)[0]
    compact_subject = _compact_layout_with_positions(subject)[0]
    compact_object = _compact_layout_with_positions(object_)[0]
    if not all((compact, compact_expression, compact_subject, compact_object)):
        return None
    candidates: list[tuple[int, int, int]] = []
    for subject_start in _all_compact_occurrences(compact, compact_subject):
        subject_end = subject_start + len(compact_subject)
        for relation_start in _all_compact_occurrences(
            compact, compact_expression,
        ):
            relation_end = relation_start + len(compact_expression)
            if relation_start < subject_end:
                continue
            for object_start in _all_compact_occurrences(
                compact, compact_object,
            ):
                if relation_end <= object_start:
                    candidates.append((subject_start, relation_start, object_start))
    if not candidates:
        return None
    subject_start, relation_start, object_start = min(
        candidates,
        key=lambda item: (
            item[2] + len(compact_object) - item[0],
            item[1] - (item[0] + len(compact_subject)),
            item,
        ),
    )
    subject_end = subject_start + len(compact_subject)
    raw_relation_start = positions[relation_start]
    previous_raw = positions[relation_start - 1] if relation_start else -1
    return {
        "compact_source": compact,
        "normalized_source": normalized_source,
        "subject_start": subject_start,
        "subject_end": subject_end,
        "relation_start": relation_start,
        "object_start": object_start,
        "prefix_between_subject_and_relation": compact[
            subject_end:relation_start
        ],
        "prefix_before_subject": compact[:subject_start],
        "raw_relation_start": raw_relation_start,
        "previous_raw": previous_raw,
        "has_layout_gap_before_relation": bool(
            previous_raw >= 0
            and previous_raw + 1 < raw_relation_start
            and normalized_source[previous_raw + 1:raw_relation_start].isspace()
        ),
    }


def _parser_soft_wrap_boundary_fragment(
    expression: str, metadata: Mapping[str, Any],
) -> bool:
    origin = str(metadata.get("source_syntax_origin") or "")
    if not origin or not re.search(r"[\u3400-\u9fff]", expression):
        return False
    geometry = _ordered_relation_geometry(expression, metadata)
    if geometry is None or not geometry["has_layout_gap_before_relation"]:
        return False
    prefix = str(geometry["prefix_between_subject_and_relation"] or "")
    if not prefix:
        return False
    prefix_cjk = "".join(re.findall(r"[\u3400-\u9fff]", prefix))
    if prefix_cjk and _CJK_LAYOUT_FUNCTION_PREFIX.fullmatch(prefix_cjk):
        return False
    previous_raw = int(geometry["previous_raw"])
    source = str(geometry["normalized_source"])
    return bool(
        0 <= previous_raw < len(source)
        and re.fullmatch(r"[\u3400-\u9fff]", source[previous_raw])
    )


def _locative_boundary_display_expression(
    expression: str, metadata: Mapping[str, Any],
) -> str:
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", expression))
    if (
        str(metadata.get("source_syntax_origin") or "")
        != "intervening_predicate_construction"
        or not _CJK_LOCATIVE_EDGE_PREFIX.match(cjk)
        or len(cjk) < 3
    ):
        return expression
    geometry = _ordered_relation_geometry(expression, metadata)
    if geometry is None or int(geometry["relation_start"]) != int(
        geometry["subject_end"]
    ):
        return expression
    clause_prefix = re.split(
        r"[，。！？；：!?;:]",
        str(geometry["prefix_before_subject"] or ""),
    )[-1]
    if not re.search(r"(?:从|在|于|於)[^，。！？；：!?;:]{0,24}$", clause_prefix):
        return expression
    marker = cjk[0]
    marker_index = expression.find(marker)
    if marker_index < 0:
        return expression
    repaired = expression[:marker_index] + expression[marker_index + 1:]
    return repaired.strip()


def _relation_contains_endpoint_surface(expression: str, value: Any) -> bool:
    endpoint = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    if not endpoint:
        return False
    if re.search(r"[\u3400-\u9fff]", endpoint):
        endpoint = re.sub(r"[\W_]+", "", endpoint)
        compact_expression = re.sub(r"[\W_]+", "", expression.casefold())
        return bool(len(endpoint) >= 2 and endpoint in compact_expression)
    endpoint = re.sub(r"\s+", " ", endpoint)
    if len(re.sub(r"[^0-9a-z]+", "", endpoint)) < 2:
        return False
    normalized_expression = re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", expression).casefold(),
    )
    return re.search(
        rf"(?<![0-9a-z]){re.escape(endpoint)}(?![0-9a-z])",
        normalized_expression,
    ) is not None


def open_relation_display_expression(
    value: Any,
    *,
    predicate_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Repair only unambiguous argument-boundary residue for presentation.

    The exact extracted expression remains in source_relation_expression.
    This display projection may remove a trailing determiner that belongs to
    the target argument (for example the equivalent of "each" before a target
    noun), but it never invents a synonym or changes direction.
    """

    expression = unicodedata.normalize("NFKC", str(value or ""))
    expression = re.sub(r"\s+", " ", expression).strip()
    rendered = _RENDERED_EDGE_RELATION_EXPRESSION.fullmatch(expression)
    if rendered is not None:
        expression = rendered.group("label").strip(" _-=:.")
    if re.search(r"[\u3400-\u9fff]", expression):
        expression = _CJK_TRAILING_ARGUMENT_DETERMINER.sub("", expression)
        expression = _locative_boundary_display_expression(
            expression,
            predicate_metadata if isinstance(predicate_metadata, Mapping) else {},
        )
    return expression.strip()


def _source_expression_is_grounded(source: Any, expression: Any) -> bool:
    """Match source wording while ignoring only layout whitespace/case."""

    source_text = unicodedata.normalize("NFKC", str(source or "")).casefold()
    candidate = unicodedata.normalize(
        "NFKC", str(expression or ""),
    ).casefold().strip()
    if not source_text or not candidate:
        return False
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", candidate):
        source_text = re.sub(r"\s+", "", source_text)
        candidate = re.sub(r"\s+", "", candidate)
        return bool(candidate and candidate in source_text)
    source_text = re.sub(r"\s+", " ", source_text).strip()
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return re.search(
        rf"(?<![0-9a-z_]){re.escape(candidate)}(?![0-9a-z_])",
        source_text,
    ) is not None


def _schema_type_match(actual: Any, expected: Any) -> bool | None:
    """Check only explicit type contradictions; unknown types stay auditable.

    Entity typing is deliberately conservative and a generic entity is not
    evidence of incompatibility.  Conversely, a concrete ``Technology``
    endpoint cannot satisfy a Schema range restricted to ``Location``.  This
    mirrors the candidate-stage three-valued contract without importing the
    extraction runtime or guessing a type hierarchy here.
    """

    expected_values = {
        str(value or "").strip().casefold()
        for value in (expected if isinstance(expected, (list, tuple, set)) else ())
        if str(value or "").strip()
    }
    if not expected_values or expected_values.intersection({"*", "any", "任意类型"}):
        return True
    actual_value = str(actual or "").strip().casefold()
    if actual_value in _UNKNOWN_SCHEMA_ENTITY_TYPES:
        return None
    return actual_value in expected_values


def _source_projection_state(metadata: Mapping[str, Any]) -> bool | None:
    """Return an explicit source-projection state when runtime supplied one."""

    if "source_relation_projection" in metadata:
        return bool(metadata.get("source_relation_projection"))
    if "candidate_source" in metadata:
        return str(metadata.get("candidate_source") or "").upper() in (
            _SOURCE_PROJECTION_CANDIDATES
        )
    return None


def _source_media_derived(metadata: Mapping[str, Any]) -> bool:
    block_types = {
        str(value or "").strip().lower()
        for value in metadata.get("source_chunk_block_types") or ()
    }
    return bool(
        metadata.get("source_media_derived")
        or block_types.intersection({"image", "figure"})
    )


def relation_publication_decision(
    projection: Mapping[str, Any],
    *,
    predicate_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a conservative syntax gate to an already verified predicate.

    Independent verification establishes that a source relation exists.  It
    does not make a truncated grammar fragment a useful graph edge label.  A
    suppressed fact remains in the normalized/audit ledger; only its public
    endpoint projection is withheld.
    """

    layer = str(projection.get("layer") or "")
    metadata = (
        predicate_metadata
        if isinstance(predicate_metadata, Mapping)
        else {}
    )
    schema_layer = layer == "schema_predicate"
    source_projection = _source_projection_state(metadata)
    if schema_layer:
        source_match = _schema_type_match(
            metadata.get("source_entity_type"),
            metadata.get("schema_domain_types"),
        )
        target_match = _schema_type_match(
            metadata.get("target_entity_type"),
            metadata.get("schema_range_types"),
        )
        if source_match is False or target_match is False:
            return {
                "eligible": False,
                "reason": "schema_endpoint_type_contract_mismatch",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
        # Production materialization explicitly records whether a Schema fact
        # came from one exact source predicate frame. Pair-level semantic guesses
        # without that immutable bridge remain in the relation audit ledger but
        # are too weak for the public graph. Absence of the marker retains
        # backward compatibility for old snapshots and direct unit consumers.
        if source_projection is False:
            # A direct proposer is an independently generated, evidence-bound
            # relation candidate that subsequently passed the verifier.  It is
            # not the same as a pair classifier projecting an endpoint pair to
            # a Schema predicate without an explicit source relation frame.
            # Preserve that independent path while keeping pair-only semantic
            # guesses in the audit ledger rather than the public graph.
            if str(metadata.get("relation_candidate_origin") or "") == (
                "direct_proposer"
            ):
                return {
                    "eligible": True,
                    "reason": "independently_verified_direct_proposer",
                    "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
                }
            return {
                "eligible": False,
                "reason": "schema_relation_without_grounded_source_expression",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
        if source_projection is None:
            return {
                "eligible": True,
                "reason": "versioned_schema_predicate",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
    elif layer != "open_source_expression":
        return {
            "eligible": False,
            "reason": "unknown_relation_semantic_layer",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    expression = unicodedata.normalize(
        "NFKC", str(projection.get("source_expression") or ""),
    ).strip()
    if not expression:
        return {
            "eligible": False,
            "reason": (
                "missing_schema_source_expression"
                if schema_layer else "missing_open_source_expression"
            ),
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if _RENDERED_EDGE_RELATION_EXPRESSION.fullmatch(expression):
        return {
            "eligible": False,
            "reason": "renderer_transport_relation_expression",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if len(expression) > 96 or _SENTENCE_PUNCTUATION.search(expression):
        return {
            "eligible": False,
            "reason": "sentence_shaped_relation_expression",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if _SYNTHETIC_ELLIPSIS.search(expression):
        return {
            "eligible": False,
            "reason": "synthetic_placeholder_relation_expression",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if _FIGURE_CAPTION_RELATION_SCAFFOLD.fullmatch(expression):
        return {
            "eligible": False,
            "reason": "figure_caption_relation_scaffold",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if (
        _source_media_derived(metadata)
        and _MEDIA_VISUAL_ENCODING_RELATION.fullmatch(expression)
    ):
        return {
            "eligible": False,
            "reason": "media_visual_encoding_relation",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if (
        projection.get("source_frame_atomic_scope_override") is True
        and not str(projection.get("source_structure_kind") or "")
        and not str(projection.get("source_syntax_origin") or "")
        and not projection.get("resolved_reference_grounding")
    ):
        return {
            "eligible": False,
            "reason": (
                "unsupported_cross_atomic_schema_relation"
                if schema_layer else "unsupported_cross_atomic_open_relation"
            ),
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if (
        projection.get("source_pair_same_atomic_scope") is False
        and not str(projection.get("source_structure_kind") or "")
        and not projection.get("resolved_reference_grounding")
    ):
        return {
            "eligible": False,
            "reason": "source_projection_outside_atomic_scope",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if (
        not str(projection.get("source_structure_kind") or "")
        and projection.get("source_expression_grounded") is not True
    ):
        return {
            "eligible": False,
            "reason": (
                "schema_relation_expression_not_source_grounded"
                if schema_layer else "open_relation_expression_not_source_grounded"
            ),
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    endpoint_surfaces = (
        metadata.get("source_subject_surface"),
        metadata.get("source_object_surface"),
    )
    if any(
        _relation_contains_endpoint_surface(expression, value)
        for value in endpoint_surfaces
    ):
        return {
            "eligible": False,
            "reason": "relation_expression_contains_endpoint_surface",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    evidence_text = str(metadata.get("source_frame_evidence_text") or "")
    if (
        _FORMULA_OPERATOR_RELATION.fullmatch(expression)
        and _FORMULA_SYNTAX.search(evidence_text)
    ):
        return {
            "eligible": False,
            "reason": "formula_operator_relation_expression",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    if _parser_soft_wrap_boundary_fragment(expression, metadata):
        return {
            "eligible": False,
            "reason": "soft_wrap_relation_boundary_fragment",
            "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
        }
    cjk = "".join(re.findall(r"[\u3400-\u9fff]", expression))
    if cjk:
        if _CJK_RELATION_FUNCTION_WORD.fullmatch(cjk):
            reason = "function_only_relation_expression"
        elif _CJK_NON_RELATIONAL_MARKER.fullmatch(cjk):
            reason = "non_relational_marker_expression"
        elif _CJK_ENUMERATION_RELATION.fullmatch(cjk):
            reason = "enumeration_marker_relation_expression"
        elif _CJK_DANGLING_ARGUMENT_MARKER.search(cjk):
            reason = "dangling_argument_marker_relation_expression"
        elif _CJK_NOMINAL_LOCATIVE_FRAGMENT.search(cjk):
            reason = "nominal_locative_fragment_relation_expression"
        elif _CJK_INCOMPLETE_LOCATIVE_STATE.fullmatch(cjk):
            reason = "incomplete_locative_state_relation_expression"
        elif _CJK_INCOMPLETE_COMPARATIVE.fullmatch(cjk):
            reason = "incomplete_comparative_relation_expression"
        elif _CJK_INCOMPLETE_RELATION.search(cjk):
            reason = "incomplete_relation_expression"
        elif _CJK_LEADING_ARGUMENT_FRAGMENT.fullmatch(cjk):
            reason = "leading_argument_fragment_relation_expression"
        elif _CJK_LOCATIVE_BOUNDARY_FRAGMENT.fullmatch(cjk):
            reason = "locative_boundary_fragment_relation_expression"
        elif _CJK_LOCATIVE_ADVERB_BOUNDARY_FRAGMENT.fullmatch(cjk):
            reason = "locative_adverb_boundary_fragment_relation_expression"
        elif _CJK_INCOMPLETE_FUNCTIONAL_PREFIX.fullmatch(cjk):
            reason = "incomplete_functional_prefix_relation_expression"
        elif _CJK_TRUNCATED_PREPOSITIONAL_RELATION.fullmatch(cjk):
            reason = "truncated_prepositional_relation_expression"
        elif re.search(r"[\u3400-\u9fff]\s+[\u3400-\u9fff]", expression) and len(cjk) <= 4:
            reason = "broken_relation_expression"
        else:
            reason = ""
        if reason:
            return {
                "eligible": False,
                "reason": reason,
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
    else:
        words = [
            _light_english_lemma(token)
            for token in re.findall(r"[A-Za-z]+", expression.casefold())
        ]
        lexical = [
            token for token in words
            if token not in _ENGLISH_RELATION_FUNCTION_WORDS
        ]
        if not lexical:
            return {
                "eligible": False,
                "reason": "function_only_relation_expression",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
        if (
            len(words) == 1
            and _ENGLISH_TITLECASE_BARE_FRAGMENT.fullmatch(expression)
            and not _ENGLISH_INFLECTED_VERB_SHAPE.search(expression)
        ):
            return {
                "eligible": False,
                "reason": "titlecase_nominal_fragment_relation_expression",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
        if len(words) == 1 and _ENGLISH_NOMINAL_RELATION.fullmatch(words[0]):
            return {
                "eligible": False,
                "reason": "nominal_fragment_relation_expression",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
    return {
        "eligible": True,
        "reason": (
            "grounded_schema_predicate"
            if schema_layer else "source_predicate_expression"
        ),
        "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
    }


def _display_label_score(value: str) -> tuple[int, int, int, int, int, str]:
    normalized = re.sub(r"\s+", " ", str(value or "")).strip()
    words = re.findall(r"[A-Za-z]+", normalized)
    auxiliary_start = bool(
        words and _light_english_lemma(words[0].casefold()) in {"be", "have", "do"}
    )
    trailing_link_word = bool(
        words and words[-1].casefold() in _ENGLISH_TRAILING_LINK_WORDS
    )
    return (
        0 if _SYNTHETIC_ELLIPSIS.search(normalized) else 1,
        0 if trailing_link_word else 1,
        0 if auxiliary_start else 1,
        1 if 1 <= len(words) <= 3 or not words else 0,
        -len(normalized),
        normalized.casefold(),
    )


def suppress_exact_open_schema_duplicates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Withhold only exact source-expression duplicates of a Schema edge.

    The immutable normalizer intentionally preserves independently verified
    facts.  At publication time, however, showing both ``causes`` and the exact
    source word used to project that same directed Schema fact adds no
    information.  The comparison is lexical and direction-sensitive; no
    synonym, translation or inverse relation is guessed.
    """

    def normalized(value: Any) -> str:
        return re.sub(
            r"\s+", " ", unicodedata.normalize("NFKC", str(value or "")),
        ).strip().casefold()

    precise: set[tuple[str, str, str]] = set()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        semantics = (
            metadata.get("relation_semantics")
            if isinstance(metadata.get("relation_semantics"), dict) else {}
        )
        publication = (
            semantics.get("publication")
            if isinstance(semantics.get("publication"), Mapping) else {}
        )
        expression = normalized(metadata.get("source_relation_expression"))
        if (
            semantics.get("layer") == "schema_predicate"
            and publication.get("eligible") is not False
            and expression
        ):
            precise.add((
                str(row.get("source_canonical_key") or ""),
                str(row.get("target_canonical_key") or ""),
                expression,
            ))

    output: list[dict[str, Any]] = []
    suppressed = 0
    for row in rows:
        value = {**row, "metadata": dict(row.get("metadata") or {})}
        metadata = value["metadata"]
        semantics = (
            dict(metadata.get("relation_semantics") or {})
            if isinstance(metadata.get("relation_semantics"), dict) else {}
        )
        expression = normalized(
            metadata.get("source_relation_expression")
            or semantics.get("source_expression")
        )
        key = (
            str(row.get("source_canonical_key") or ""),
            str(row.get("target_canonical_key") or ""),
            expression,
        )
        if semantics.get("layer") == "open_source_expression" and key in precise:
            semantics["publication"] = {
                "eligible": False,
                "reason": "open_relation_duplicate_of_schema_source_expression",
                "policy": RELATION_SEMANTIC_PROJECTION_VERSION,
            }
            metadata["relation_semantics"] = semantics
            suppressed += 1
        output.append(value)
    return output, suppressed


def converge_relation_display_labels(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Converge morphology variants without changing raw facts or direction."""

    grouped: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    for index, row in enumerate(rows):
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        semantics = (
            metadata.get("relation_semantics")
            if isinstance(metadata.get("relation_semantics"), dict) else {}
        )
        label = str(metadata.get("display_relation_label") or "").strip()
        if semantics.get("layer") != "open_source_expression" or not label:
            continue
        integrity = (
            metadata.get("publication_integrity")
            if isinstance(metadata.get("publication_integrity"), dict) else {}
        )
        key = (
            str(semantics.get("family_key") or ""),
            str(integrity.get("source_entity_type") or "").casefold(),
            str(integrity.get("target_entity_type") or "").casefold(),
        )
        if key[0]:
            grouped[key].append((index, label))

    output = [
        {**row, "metadata": dict(row.get("metadata") or {})}
        for row in rows
    ]
    converged_groups = 0
    relabelled = 0
    for key, values in grouped.items():
        counts = Counter(label for _, label in values)
        if len(counts) <= 1:
            continue
        representative = max(
            counts,
            key=lambda label: (counts[label], *_display_label_score(label)),
        )
        converged_groups += 1
        for index, original in values:
            metadata = output[index]["metadata"]
            semantics = dict(metadata.get("relation_semantics") or {})
            semantics["display_family_label"] = representative
            semantics["display_convergence"] = "morphology_and_endpoint_types"
            metadata["relation_semantics"] = semantics
            metadata["display_relation_label"] = representative
            if original != representative:
                relabelled += 1
    return output, {
        "groups_considered": len(grouped),
        "groups_converged": converged_groups,
        "facts_relabelled": relabelled,
        "additional_model_calls": 0,
    }


def relation_semantic_projection(
    *,
    relation_type: Any,
    source_expression: Any = "",
    open_relation_carrier: bool = False,
    predicate_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe one verified predicate without changing its fact semantics."""

    metadata = (
        predicate_metadata if isinstance(predicate_metadata, Mapping) else {}
    )
    canonical = str(relation_type or "").strip()
    source = str(source_expression or "").strip()
    if open_relation_carrier:
        display = open_relation_display_expression(
            source or canonical,
            predicate_metadata=metadata,
        )
        family_source = display or source or canonical
        projection = {
            "version": RELATION_SEMANTIC_PROJECTION_VERSION,
            "layer": "open_source_expression",
            "canonical_predicate": canonical,
            "source_expression": source,
            "display_expression": display,
            "family_key": "open:" + normalized_relation_family_key(
                family_source
            ),
            "strict_inference_eligible": False,
            "grouping_method": "source_expression_morphology",
            "source_frame_atomic_scope_override": bool(
                metadata.get("source_frame_atomic_scope_override")
            ),
            "source_pair_same_atomic_scope": (
                metadata.get("source_pair_same_atomic_scope")
                if isinstance(
                    metadata.get("source_pair_same_atomic_scope"), bool,
                ) else None
            ),
            "source_structure_kind": str(
                metadata.get("source_structure_kind") or ""
            ),
            "source_syntax_origin": str(
                metadata.get("source_syntax_origin") or ""
            ),
            "source_expression_grounded": bool(
                str(metadata.get("source_structure_kind") or "")
                or _source_expression_is_grounded(
                    metadata.get("source_frame_evidence_text"), source,
                )
            ),
            "resolved_reference_grounding": bool(
                metadata.get("resolved_reference_endpoints")
                or metadata.get("source_resolved_reference_endpoints")
            ),
        }
        projection["publication"] = relation_publication_decision(
            projection,
            predicate_metadata=metadata,
        )
        return projection

    projection = {
        "version": RELATION_SEMANTIC_PROJECTION_VERSION,
        "layer": "schema_predicate",
        "canonical_predicate": canonical,
        "source_expression": source,
        "family_key": "schema:" + normalized_relation_family_key(canonical),
        "strict_inference_eligible": True,
        "grouping_method": "versioned_schema",
        "publish_mode": str(metadata.get("publish_mode") or ""),
        "source_frame_atomic_scope_override": bool(
            metadata.get("source_frame_atomic_scope_override")
        ),
        "source_pair_same_atomic_scope": (
            metadata.get("source_pair_same_atomic_scope")
            if isinstance(
                metadata.get("source_pair_same_atomic_scope"), bool,
            ) else None
        ),
        "source_structure_kind": str(
            metadata.get("source_structure_kind") or ""
        ),
        "source_syntax_origin": str(
            metadata.get("source_syntax_origin") or ""
        ),
        "source_expression_grounded": bool(
            str(metadata.get("source_structure_kind") or "")
            or _source_expression_is_grounded(
                metadata.get("source_frame_evidence_text"), source,
            )
        ),
        "resolved_reference_grounding": bool(
            metadata.get("resolved_reference_endpoints")
            or metadata.get("source_resolved_reference_endpoints")
        ),
        "source_relation_projection": _source_projection_state(metadata),
    }
    projection["publication"] = relation_publication_decision(
        projection,
        predicate_metadata=metadata,
    )
    return projection


__all__ = [
    "RELATION_SEMANTIC_PROJECTION_VERSION",
    "converge_relation_display_labels",
    "normalized_relation_family_key",
    "open_relation_display_expression",
    "relation_publication_decision",
    "relation_semantic_projection",
    "source_expression_matches_schema_cues",
    "suppress_exact_open_schema_duplicates",
]
