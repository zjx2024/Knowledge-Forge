"""Pure relation planning, normalization, and trust-boundary primitives."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

from .config import Config
from .sentence_spans import sentence_index, split_relation_scopes


PENDING_CLASSIFICATION = "PENDING_CLASSIFICATION"
DEFERRED_BUDGET = "DEFERRED_BUDGET"
SOURCE_FIRST_BACKSTOP_SKIPPED = "SOURCE_FIRST_BACKSTOP_SKIPPED"
REJECTED_SELF_LOOP = "REJECTED_SELF_LOOP"
TYPE_UNRESOLVED = "TYPE_UNRESOLVED"
TYPE_INCOMPATIBLE = "TYPE_INCOMPATIBLE"
FILTERED_DISTANCE = "FILTERED_DISTANCE"
FILTERED_STRUCTURE_PEER = "FILTERED_STRUCTURE_PEER"
FILTERED_STRUCTURE_INELIGIBLE = "FILTERED_STRUCTURE_INELIGIBLE"
FILTERED_SCOPE_BOUNDARY = "FILTERED_SCOPE_BOUNDARY"
NO_RELATION = "NO_RELATION"
OTHER_UNSUPPORTED = "OTHER_UNSUPPORTED"
HYPOTHESES = "HYPOTHESES"
EXPLICIT_NO_RELATION = "EXPLICIT_NO_RELATION"
SCHEMA_GAP = "SCHEMA_GAP"
HYPOTHESES_EMITTED = "HYPOTHESES_EMITTED"
CLASSIFICATION_PENDING = "CLASSIFICATION_PENDING"
CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"

GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION = (
    "shadow-relation-classifier-prompt-v16-grounded-carrier-reserved"
)
GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION = (
    "shadow-relation-classifier-schema-v8-grounded-carrier-reserved"
)
GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION = (
    "shadow-relation-classifier-parser-v14-clause-local-lexical-recovery"
)
GROUNDED_RELATION_PROPOSER_PROMPT_VERSION = (
    "knowledge-forge-local-subgraph-prompt-v12-grounded-carrier-reserved"
)
GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION = (
    "knowledge-forge-local-subgraph-schema-v8-grounded-carrier-reserved"
)
GROUNDED_RELATION_PROPOSER_PARSER_VERSION = "knowledge-forge-local-subgraph-parser-v6-legacy-normalization"
GROUNDED_RELATION_VERIFIER_PROMPT_VERSION = (
    "shadow-relation-verifier-prompt-v37-entailed-orientation-roles"
)
GROUNDED_RELATION_STRUCTURAL_SCHEMA_VERSION = "shadow-relation-structural-adjudication-schema-v1"
GROUNDED_RELATION_VERIFIER_SCHEMA_VERSION = (
    "shadow-relation-verifier-schema-v10-entailed-orientation-roles"
)
GROUNDED_RELATION_VERIFIER_PARSER_VERSION = (
    "shadow-relation-verifier-parser-v19-single-edit-enum-repair"
)
GROUNDED_RELATION_ROLE_PROBE_PROMPT_VERSION = "shadow-relation-role-probe-prompt-v13-bounded-context"
GROUNDED_RELATION_ROLE_PROBE_SCHEMA_VERSION = "shadow-relation-role-probe-schema-v6"
GROUNDED_RELATION_ROLE_PROBE_PARSER_VERSION = "shadow-relation-role-probe-parser-v6"
GROUNDED_RELATION_ROLE_ALIGNMENT_PROMPT_VERSION = "shadow-relation-role-alignment-prompt-v15-bounded-context"
GROUNDED_RELATION_ROLE_ALIGNMENT_SCHEMA_VERSION = "shadow-relation-role-alignment-schema-v5"
GROUNDED_RELATION_ROLE_ALIGNMENT_PARSER_VERSION = "shadow-relation-role-alignment-parser-v9"
GROUNDED_RELATION_SEMANTIC_CONTRAST_PROMPT_VERSION = "shadow-relation-semantic-contrast-prompt-v3"
GROUNDED_RELATION_SEMANTIC_CONTRAST_SCHEMA_VERSION = "shadow-relation-semantic-contrast-schema-v1"
GROUNDED_RELATION_SEMANTIC_CONTRAST_PARSER_VERSION = "shadow-relation-semantic-contrast-parser-v1"
GROUNDED_RELATION_ATOMIC_FACT_PROMPT_VERSION = "knowledge-forge-atomic-frame-inventory-prompt-v14"
GROUNDED_RELATION_ATOMIC_FACT_SCHEMA_VERSION = "knowledge-forge-atomic-frame-inventory-schema-v6"
GROUNDED_RELATION_ATOMIC_FACT_PARSER_VERSION = "knowledge-forge-atomic-frame-inventory-parser-v8"
GROUNDED_RELATION_OPEN_FACT_PROMPT_VERSION = (
    "knowledge-forge-grounded-open-fact-prompt-v10-exact-source-predicate"
)
GROUNDED_RELATION_OPEN_FACT_PRIMARY_CONTRACT_VERSION = (
    "knowledge-forge-grounded-open-fact-primary-v9-exact-source-predicate"
)
GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION = (
    "knowledge-forge-grounded-open-fact-recovery-v7-exact-source-predicate"
)
GROUNDED_RELATION_OPEN_FACT_SCHEMA_VERSION = (
    "knowledge-forge-grounded-open-fact-schema-v4-prelocated-references"
)
GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION = (
    "knowledge-forge-grounded-open-fact-parser-v29-reference-token-boundary"
)
GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION = (
    "knowledge-forge-reference-resolution-prompt-v2-possessive-role"
)
GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION = (
    "knowledge-forge-reference-resolution-schema-v1"
)
GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION = (
    "knowledge-forge-reference-resolution-parser-v4-reference-token-boundary"
)
GROUNDED_RELATION_OPEN_FACT_SYSTEM_PROMPT = (
    "You are a source-grounded open fact-frame extractor. Cover every directly "
    "stated binary proposition while using only the supplied immutable mention "
    "IDs or explicitly resolved local-reference IDs. Decompose coordinated and "
    "nested semantic roles, preserve negation "
    "and normative obligations, and use parser structure for tables and lists. "
    "Do not map to a logical predicate, infer a fact, or connect items merely "
    "because they co-occur. A predicate must contain a lexical semantic head; "
    "a determiner, quantifier, conjunction, preposition or modal fragment alone "
    "is not a relation. Return strict JSON."
)
GROUNDED_RELATION_FRAME_ALIGNMENT_PROMPT_VERSION = "knowledge-forge-frame-schema-alignment-prompt-v4"
GROUNDED_RELATION_FRAME_ALIGNMENT_SCHEMA_VERSION = "knowledge-forge-frame-schema-alignment-schema-v2"
GROUNDED_RELATION_FRAME_ALIGNMENT_PARSER_VERSION = "knowledge-forge-frame-schema-alignment-parser-v2"
GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION = (
    "knowledge-forge-source-frame-schema-binding-prompt-v14-runtime-review-policy"
)
GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION = "knowledge-forge-source-frame-schema-binding-schema-v5-runtime-review-policy"
GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION = "knowledge-forge-source-frame-schema-binding-parser-v9-surface-role-topology"
GROUNDED_RELATION_SOURCE_RECALL_SHORTLIST_VERSION = (
    "knowledge-forge-source-schema-recall-shortlist-v3-polarity-policy"
)
GROUNDED_RELATION_CONFLICT_PROMPT_VERSION = (
    "knowledge-forge-source-predicate-conflict-prompt-v1-contrastive"
)
GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION = (
    "knowledge-forge-source-predicate-conflict-schema-v1"
)
GROUNDED_RELATION_CONFLICT_PARSER_VERSION = (
    "knowledge-forge-source-predicate-conflict-parser-v1-fail-closed"
)
GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION = (
    "knowledge-forge-source-projection-role-audit-prompt-v7-role-pair-projection"
)
GROUNDED_RELATION_SOURCE_ROLE_AUDIT_SCHEMA_VERSION = (
    "knowledge-forge-source-projection-role-audit-schema-v2-projection-status"
)
GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PARSER_VERSION = (
    "knowledge-forge-source-projection-role-audit-parser-v2-projection-status"
)
GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION = (
    "knowledge-forge-atomic-publication-audit-prompt-v4-complete-role-contract"
)
GROUNDED_RELATION_PUBLICATION_AUDIT_SCHEMA_VERSION = (
    "knowledge-forge-atomic-publication-audit-schema-v1-keyed-statuses"
)
GROUNDED_RELATION_PUBLICATION_AUDIT_PARSER_VERSION = (
    "knowledge-forge-atomic-publication-audit-parser-v1-exact-key-set"
)
GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION = (
    "knowledge-forge-source-projection-group-audit-prompt-v4-keyed-shared-equivalence"
)
GROUNDED_RELATION_SOURCE_GROUP_AUDIT_SCHEMA_VERSION = (
    "knowledge-forge-source-projection-group-audit-schema-v3-keyed-statuses"
)
GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PARSER_VERSION = (
    "knowledge-forge-source-projection-group-audit-parser-v2-keyed-normalization"
)
GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PROMPT_VERSION = (
    "knowledge-forge-source-projection-group-anchor-audit-prompt-v2-exact-single-role"
)
GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_SCHEMA_VERSION = (
    "knowledge-forge-source-projection-group-anchor-audit-schema-v1-projection-status"
)
GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PARSER_VERSION = (
    "knowledge-forge-source-projection-group-anchor-audit-parser-v1-fail-closed"
)
_SOURCE_PROJECTION_EQUIVALENCE_INSTRUCTION = (
    "Decide only whether the exact open source predicate semantically "
    "instantiates the requested Schema relation and directly assigns the "
    "candidate subject and object its requested roles and direction. ALIGNED "
    "requires relation-meaning equivalence and a direct role match; topical "
    "similarity, a shared verb, compatible endpoint types, implication, "
    "plausibility, causally possible interpretation, or useful inference are "
    "insufficient. Compare the open predicate's argument-0 to argument-1 "
    "direction explicitly with the required Schema subject and object. If "
    "source wording gives argument 0 a whole, container, owner, provider or "
    "other governing role while the Schema requires that role at the object "
    "endpoint, the current unswapped candidate is CONFLICT; only a separately "
    "supplied swapped candidate could align. Apply the same rule to any "
    "inverse role pattern and never treat an inverse paraphrase as support for "
    "the same ordered predicate. "
)
GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PROMPT_VERSION = (
    "knowledge-forge-source-structure-binding-audit-prompt-v1"
)
GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_SCHEMA_VERSION = (
    "knowledge-forge-source-structure-binding-audit-schema-v1"
)
GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PARSER_VERSION = (
    "knowledge-forge-source-structure-binding-audit-parser-v1"
)
GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION = (
    "knowledge-forge-deterministic-parser-structure-terminal-v6-contiguous-id-inventory"
)
GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_SCHEMA_VERSION = (
    "knowledge-forge-deterministic-parser-structure-terminal-schema-v2-parenthetical"
)
GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_PARSER_VERSION = (
    "knowledge-forge-deterministic-parser-structure-terminal-parser-v2-parenthetical"
)
PARENTHETICAL_ENUMERATION_STRUCTURE_KIND = "parenthetical_enumeration"
PARENTHETICAL_DETAIL_RELATION = "parenthetical detail of"
STRUCTURE_PARALLEL_OPEN_CARRIER_SUPPRESSION = (
    "suppress_same_pair_overlapping_open_carriers"
)


_RENDERED_EDGE_RELATION_EXPRESSION = re.compile(
    r"^\s*[-=.]{1,4}\s*(?P<label>[^\n<>]{1,80}?)\s*[-=.]{1,4}>\s*$"
)


def _rendered_edge_relation_label(value: Any) -> str:
    """Recover the semantic label from a generated ``A --r--> B`` edge.

    Image analysis and diagram parsers intentionally serialize visible edges
    in this transport form.  The dashes and arrow carry direction but are not
    part of the predicate label.  Returning only the bounded inner label keeps
    useful diagram relations while preventing renderer syntax from becoming a
    user-facing predicate.
    """

    match = _RENDERED_EDGE_RELATION_EXPRESSION.fullmatch(str(value or ""))
    if match is None:
        return ""
    label = re.sub(r"\s+", " ", match.group("label")).strip(" _-=:.")
    return label if label else ""
GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION = (
    "knowledge-forge-coordinated-source-projection-recovery-v6-final-rejection-authority"
)
GROUNDED_RELATION_COORDINATED_RECOVERY_SCHEMA_VERSION = (
    "knowledge-forge-coordinated-source-projection-recovery-schema-v2-grounding"
)
GROUNDED_RELATION_COORDINATED_RECOVERY_PARSER_VERSION = (
    "knowledge-forge-coordinated-source-projection-recovery-parser-v3-clause-boundary"
)
GROUNDED_RELATION_BATCH_VERIFIER_PROMPT_VERSION = (
    "knowledge-forge-relation-batch-verifier-prompt-v12-entailed-orientation-roles"
)
GROUNDED_RELATION_BATCH_VERIFIER_SCHEMA_VERSION = (
    "knowledge-forge-relation-batch-verifier-schema-v4-entailed-orientation-roles"
)
GROUNDED_RELATION_BATCH_VERIFIER_PARSER_VERSION = (
    "knowledge-forge-relation-batch-verifier-parser-v4-keyed-map-normalization"
)
GROUNDED_RELATION_EXACT_ARGUMENT_PROMPT_VERSION = "shadow-relation-exact-argument-prompt-v16"
GROUNDED_RELATION_EXACT_ARGUMENT_SCHEMA_VERSION = "shadow-relation-exact-argument-schema-v10"
GROUNDED_RELATION_EXACT_ARGUMENT_PARSER_VERSION = "shadow-relation-exact-argument-parser-v12"

PASS = "PASS"
REVIEW_ONLY = "REVIEW_ONLY"
SUPPORTED = "SUPPORTED"
REJECTED = "REJECTED"
UNCERTAIN = "UNCERTAIN"
RETRYABLE_FAILED = "RETRYABLE_FAILED"

_UNKNOWN_TYPES = {"", "unknown", "genericentity", "unresolved"}
PUBLISH_AUTO = "auto"
PUBLISH_REVIEW_ONLY = "review_only"
PUBLISH_DISABLED = "disabled"
CANONICAL_FORWARD = "forward"
CANONICAL_INVERSE = "inverse"
CANONICAL_SYMMETRIC = "symmetric"
DIRECT_PROPOSER = "DIRECT_PROPOSER"
RELATION_GLEANING = "RELATION_GLEANING"
PAIR_PLANNER = "PAIR_PLANNER"
TABLE_STRUCTURE = "TABLE_STRUCTURE"
ENDPOINT_COMPLETION = "ENDPOINT_COMPLETION"
GROUNDED_OPEN_FACT = "GROUNDED_OPEN_FACT"


def _digest(parts: Sequence[object]) -> str:
    material = "\x1f".join(str(value) for value in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


_PROMPT_PREDICATE_FIELDS = (
    "name",
    "semantic_definition",
    "domain_types",
    "range_types",
    "direction_description",
    "subject_role_description",
    "object_role_description",
    "symmetric",
    "inverse_predicate",
    "canonical_orientation",
    "allowed_context_kinds",
    "allowed_source_roles",
    "source_role_pairs",
    "scope_policy",
    "required_evidence_roles",
    "positive_examples",
    "negative_examples",
    "positive_cues",
    "negative_cues",
    "anti_patterns",
    "optional_soft_cues",
    "structural_evidence_kinds",
    "structural_role_labels",
    "identity_qualifier_keys",
    "conflicts_with",
    "negation_policy",
    "modality_policy",
    "publish_mode",
    "open_relation_policy",
)
_PROMPT_TEXT_LIMITS = {
    "semantic_definition": 1600,
    "direction_description": 1000,
    "subject_role_description": 1000,
    "object_role_description": 1000,
    "positive_examples": 600,
    "negative_examples": 600,
    "positive_cues": 240,
    "negative_cues": 240,
    "anti_patterns": 320,
    "optional_soft_cues": 240,
}
_PROMPT_LIST_LIMITS = {
    "domain_types": 32,
    "range_types": 32,
    "allowed_context_kinds": 32,
    "allowed_source_roles": 32,
    "source_role_pairs": 16,
    "required_evidence_roles": 16,
    "positive_examples": 4,
    "negative_examples": 4,
    "positive_cues": 12,
    "negative_cues": 12,
    "anti_patterns": 8,
    "optional_soft_cues": 12,
    "structural_evidence_kinds": 16,
    "structural_role_labels": 32,
    "identity_qualifier_keys": 16,
    "conflicts_with": 16,
}

_CORE_SEMANTIC_SOURCE_ROLES = frozenset({
    "grammatical_subject",
    "action_theme",
    "role_complement",
    "direct_argument",
    "collective_participant",
})
_VALID_SOURCE_ROLE_PAIRS = frozenset({
    f"{subject}->{object_}"
    for subject in _CORE_SEMANTIC_SOURCE_ROLES
    for object_ in _CORE_SEMANTIC_SOURCE_ROLES
    if subject != object_ or subject == "collective_participant"
}) | frozenset({
    "possessor->possessed",
    "possessed->possessor",
})

# ``assertion_status=ASSERTED`` means that the source asserts the proposition;
# it does not mean that the proposition itself is affirmative.  Open frames
# such as ``A is not B`` therefore remain asserted while retaining their
# grammatical negation in ``relation``.  A normal positive Schema predicate
# must never be bound to that frame.  A project can opt a genuinely negative
# predicate (for example, an explicit exclusion relation) into that source
# form through its versioned policy instead of relying on a predicate name.
ALLOW_EXPLICIT_NEGATIVE_RELATION = "allow_explicit_negative_relation"
REJECT_NEGATED_RELATION = "reject_negated"
REQUIRE_ASSERTED_MODALITY = "require_asserted"
_SOURCE_NEGATION_PATTERN = re.compile(
    r"(?:\b(?:not|never|neither|nor|without|cannot|can't|couldn't|doesn't|"
    r"don't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"no|nicht|kein|keine|keinen|jamais|sans|pas|non|nunca|sin|não|sem)\b|"
    r"(?:不|未|无|無|非|没有|沒有|并非|並非|不得|不能|不会|不會|从未|從未|"
    r"ない|ません|ではない|ず|않|없|아니|못))",
    re.IGNORECASE,
)


def _source_relation_has_explicit_negation(value: Any) -> bool:
    """Recognize grammatical negation in predicate-bearing source text.

    This is deliberately language-level rather than domain- or
    predicate-specific.  ``not only`` is contrastive focus, not proposition
    negation, so remove that construction before checking the remaining
    relation material.
    """

    relation = re.sub(r"[_‐‑‒–—-]+", " ", str(value or ""))
    relation = re.sub(
        r"\bnot\s+only\b", " ", relation, flags=re.IGNORECASE,
    )
    return _SOURCE_NEGATION_PATTERN.search(relation) is not None


def compact_predicate_prompt_metadata(
    value: Mapping[str, Any],
    *,
    include_name: bool = True,
) -> dict[str, Any]:
    """Project stored predicate metadata into the model-facing contract.

    The full PredicateSpec contains persistence, qualifier and evolution
    details that are valuable for deterministic validation but irrelevant to a
    single semantic decision. Sending those fields for every local scope made
    prompt size grow with the project schema rather than the source evidence.
    This projection keeps all semantic, type, direction and structural policy
    fields needed by the model and leaves the complete metadata server-side.
    """

    result: dict[str, Any] = {}
    for key in _PROMPT_PREDICATE_FIELDS:
        if key == "name" and not include_name:
            continue
        if key not in value:
            continue
        raw = value.get(key)
        if isinstance(raw, str):
            limit = int(_PROMPT_TEXT_LIMITS.get(key, 1200))
            result[key] = raw[:limit]
        elif isinstance(raw, Mapping):
            result[key] = dict(raw)
        elif isinstance(raw, (list, tuple, set, frozenset)):
            limit = int(_PROMPT_LIST_LIMITS.get(key, 32))
            result[key] = [str(item)[:600] for item in list(raw)[:limit]]
        else:
            result[key] = raw
    return result


_CANDIDATE_PREDICATE_FIELDS = (
    "name",
    "semantic_definition",
    "domain_types",
    "range_types",
    "direction_description",
    "subject_role_description",
    "object_role_description",
    "symmetric",
    "inverse_predicate",
    "canonical_orientation",
    "allowed_context_kinds",
    "allowed_source_roles",
    "source_role_pairs",
    "scope_policy",
    "optional_soft_cues",
    "structural_evidence_kinds",
    "structural_role_labels",
    "publish_mode",
)
_CANDIDATE_TEXT_LIMITS = {
    "semantic_definition": 480,
    "direction_description": 320,
    "subject_role_description": 320,
    "object_role_description": 320,
}


def candidate_predicate_prompt_metadata(
    value: Mapping[str, Any],
    *,
    include_name: bool = True,
) -> dict[str, Any]:
    """Return the recall-stage predicate catalog without verifier detail.

    Candidate generation needs the semantic roles, direction, types and hard
    structural policy.  Repeating every contrastive example, conflict rule and
    evolution field for every predicate made each tiny source scope carry tens
    of thousands of unrelated characters.  Full policy remains server-side
    and is supplied to deterministic validation and the independent final
    verifier, so this projection reduces transport cost without weakening the
    publication gate.
    """

    compact = compact_predicate_prompt_metadata(
        value, include_name=include_name,
    )
    result: dict[str, Any] = {}
    for key in _CANDIDATE_PREDICATE_FIELDS:
        if key == "name" and not include_name:
            continue
        if key not in compact:
            continue
        raw = compact[key]
        if isinstance(raw, str):
            result[key] = raw[:int(_CANDIDATE_TEXT_LIMITS.get(key, 240))]
        elif isinstance(raw, list):
            result[key] = [str(item)[:120] for item in raw[:8]]
        elif isinstance(raw, Mapping):
            result[key] = dict(raw)
        else:
            result[key] = raw
    return result


def bounded_relation_context(
    content: str,
    context_span: tuple[int, int],
    *,
    focus_spans: Sequence[tuple[int, int]] = (),
    max_chars: int | None = None,
) -> dict[str, Any]:
    """Return a bounded source display while preserving absolute coordinates.

    Source inventories are intentionally complete and do not use this helper.
    Pair-level prompts, however, must not repeat a whole document for every
    candidate.  When a contiguous window cannot contain both endpoints, two
    endpoint-centered segments are shown with an explicit omission marker. The
    returned ``source_span`` always remains the original immutable scope, so
    callers and parsers continue to validate evidence against the full source
    coordinates rather than silently changing provenance.
    """

    source_start, source_end = map(int, context_span)
    if (
        source_start < 0
        or source_end <= source_start
        or source_end > len(content)
    ):
        raise ValueError("relation context span is invalid")
    limit = int(max_chars or Config.relation_model_context_max_chars)
    limit = max(2048, limit)
    valid_focus: list[tuple[int, int]] = []
    for raw_start, raw_end in focus_spans:
        start, end = int(raw_start), int(raw_end)
        if source_start <= start < end <= source_end:
            valid_focus.append((start, end))
    valid_focus = sorted(set(valid_focus))
    source_length = source_end - source_start
    if source_length <= limit:
        return {
            "source_span": [source_start, source_end],
            "visible_segments": [[source_start, source_end]],
            "text": content[source_start:source_end],
            "truncated": False,
        }

    if not valid_focus:
        segments = [(source_start, min(source_end, source_start + limit))]
    else:
        left = valid_focus[0][0]
        right = valid_focus[-1][1]
        required = right - left
        if required <= limit:
            spare = limit - required
            window_start = max(source_start, left - spare // 2)
            window_end = min(source_end, window_start + limit)
            window_start = max(source_start, window_end - limit)
            segments = [(window_start, window_end)]
        else:
            # The endpoints are too far apart for one local window. Show an
            # equal-budget neighborhood around each endpoint; the omitted
            # source is explicit and cannot be mistaken for contiguous text.
            per_segment = max(1024, (limit - 128) // 2)
            segments = []
            for start, end in (valid_focus[0], valid_focus[-1]):
                spare = max(0, per_segment - (end - start))
                segment_start = max(source_start, start - spare // 2)
                segment_end = min(source_end, segment_start + per_segment)
                segment_start = max(source_start, segment_end - per_segment)
                segments.append((segment_start, segment_end))
            if segments[0][1] >= segments[1][0]:
                segments = [(segments[0][0], segments[1][1])]

    visible_segments = [[int(start), int(end)] for start, end in segments]
    pieces: list[str] = []
    for index, (start, end) in enumerate(segments):
        if index:
            previous_end = segments[index - 1][1]
            if previous_end < start:
                pieces.append(
                    f"\n[OMITTED_SOURCE start={previous_end} end={start}]\n"
                )
        pieces.append(
            f"[SOURCE_SEGMENT start={start} end={end}]\n"
            f"{content[start:end]}"
        )
    return {
        "source_span": [source_start, source_end],
        "visible_segments": visible_segments,
        "text": "\n".join(pieces),
        "truncated": True,
    }


def _pair_model_context(
    opportunity: PairOpportunity,
    *,
    content: str,
    max_chars: int | None = None,
    extra_focus_spans: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    return bounded_relation_context(
        content,
        opportunity.context_span,
        focus_spans=(
            opportunity.left_span,
            opportunity.right_span,
            *tuple(extra_focus_spans),
        ),
        max_chars=max_chars,
    )


def pair_identity(
    document_revision_id: str,
    left_occurrence_key: str,
    right_occurrence_key: str,
) -> str:
    """Return a policy-independent stable unordered occurrence-pair identity."""

    left, right = sorted((str(left_occurrence_key), str(right_occurrence_key)))
    return "relpair_" + _digest((document_revision_id, left, right))


def opportunity_identity(
    run_id: str,
    candidate_policy_version: str,
    stable_pair_identity: str,
) -> str:
    return "relopp_" + _digest((
        run_id, candidate_policy_version, stable_pair_identity,
    ))


@dataclass(frozen=True)
class PairOpportunity:
    opportunity_identity: str
    pair_identity: str
    run_id: str
    project_id: int
    document_revision_id: str
    chunk_key: str
    candidate_policy_version: str
    left_occurrence_key: str
    right_occurrence_key: str
    left_mention_id: str
    right_mention_id: str
    left_span: tuple[int, int]
    right_span: tuple[int, int]
    left_canonical_id: str
    right_canonical_id: str
    left_assignment_version: str
    right_assignment_version: str
    left_type: str
    right_type: str
    context_identity: str
    context_span: tuple[int, int]
    allowed_predicate_metadata: tuple[dict[str, Any], ...]
    planning_reason: str
    cue: str
    type_status: str
    status: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairOpportunityPlan:
    opportunities: tuple[PairOpportunity, ...]
    theoretical_pairs: int
    duplicate_occurrences_collapsed: int

    @property
    def admitted(self) -> tuple[PairOpportunity, ...]:
        return tuple(
            item for item in self.opportunities
            if item.status == PENDING_CLASSIFICATION
        )


class RelationClassificationError(ValueError):
    """A classifier batch cannot be aligned to its frozen pair universe."""


class RelationVerificationRejected(RelationClassificationError):
    """A verifier response is valid enough to reject, but cannot support a fact.

    This is deliberately distinct from a transport or response-protocol failure:
    a claimed ``SUPPORTED`` verdict without admissible evidence is a semantic
    rejection of that hypothesis, not a reason to fail an otherwise independent
    document build.
    """

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        forward_entailment: str = "",
        reverse_entailment: str = "",
        subject_role_supported: bool = False,
        object_role_supported: bool = False,
    ):
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.forward_entailment = str(forward_entailment)
        self.reverse_entailment = str(reverse_entailment)
        self.subject_role_supported = bool(subject_role_supported)
        self.object_role_supported = bool(object_role_supported)


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    semantic_definition: str
    domain_types: tuple[str, ...]
    range_types: tuple[str, ...]
    direction_description: str
    subject_role_description: str
    object_role_description: str
    symmetric: bool
    inverse_predicate: str
    canonical_orientation: str
    publish_mode: str
    reflexive: bool
    reciprocal_allowed: bool
    allowed_context_kinds: tuple[str, ...]
    allowed_source_roles: tuple[str, ...]
    source_role_pairs: tuple[str, ...]
    scope_policy: dict[str, bool]
    required_evidence_roles: tuple[str, ...]
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    positive_cues: tuple[str, ...]
    negative_cues: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    optional_soft_cues: tuple[str, ...]
    structural_evidence_kinds: tuple[str, ...]
    structural_role_labels: tuple[str, ...]
    qualifier_schema: dict[str, Any]
    identity_qualifier_keys: tuple[str, ...]
    conflicts_with: tuple[str, ...]
    negation_policy: str
    modality_policy: str
    open_relation_policy: dict[str, Any]
    schema_version: str

    @property
    def review_only(self) -> bool:
        return self.publish_mode != PUBLISH_AUTO

    @property
    def publishable(self) -> bool:
        return self.publish_mode == PUBLISH_AUTO


def _fallback_only_open_relation_policy(
    value: PredicateSpec | Mapping[str, Any],
) -> bool:
    """Return whether a predicate is reserved for grounded source frames.

    Pair classifiers and generic subgraph proposers must never see a fallback
    carrier.  It is deliberately retained in frozen pair metadata so an exact
    source-frame projection can still pass the ordinary structure and type
    gates.  Runtime behavior follows policy data, not a predicate identifier.
    """

    policy = (
        value.open_relation_policy
        if isinstance(value, PredicateSpec)
        else value.get("open_relation_policy") or {}
    )
    return bool(
        isinstance(policy, Mapping)
        and policy.get("enabled")
        and policy.get("fallback_only")
    )


@dataclass(frozen=True)
class EvidenceSpan:
    start: int
    end: int


@dataclass(frozen=True)
class RelationHypothesis:
    hypothesis_id: str
    pair_identity: str
    subject_occurrence_key: str
    predicate: str
    object_occurrence_key: str
    evidence_spans: tuple[EvidenceSpan, ...]
    classifier_confidence: float
    classifier_policy_version: str
    classifier_model_revision: str
    classifier_prompt_version: str
    classifier_schema_version: str
    classifier_parser_version: str
    predicate_metadata: dict[str, Any]
    status: str = "CLASSIFIED"


@dataclass(frozen=True)
class PairClassificationDecision:
    pair_identity: str
    decision: str
    hypotheses: tuple[RelationHypothesis, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    reason_code: str

    @property
    def decision_status(self) -> str:
        if str(self.reason_code).startswith("CLASSIFIER_TERMINAL_"):
            return CLASSIFICATION_FAILED
        return pair_decision_status(self.decision)


def pair_decision_status(decision: str) -> str:
    """Return the explicit pair-ledger terminal status for a classifier decision."""

    if decision == NO_RELATION:
        return EXPLICIT_NO_RELATION
    if decision == OTHER_UNSUPPORTED:
        return SCHEMA_GAP
    if decision == HYPOTHESES:
        return HYPOTHESES_EMITTED
    return CLASSIFICATION_FAILED


@dataclass(frozen=True)
class RelationProposal:
    proposal_id: str
    context_id: str
    candidate_source: str
    pass_no: int
    subject_occurrence_key: str
    predicate: str
    object_occurrence_key: str
    evidence_spans: tuple[EvidenceSpan, ...]
    qualifiers: tuple[dict[str, Any], ...]
    confidence: float
    proposer_policy_version: str
    proposer_model_revision: str


@dataclass(frozen=True)
class RelationProposalBatch:
    proposals: tuple[RelationProposal, ...]
    missing_endpoint_mentions: tuple[dict[str, Any], ...]
    coverage: dict[str, Any]


@dataclass(frozen=True)
class HypothesisValidation:
    hypothesis_id: str
    status: str
    reason_code: str
    predicate_metadata: dict[str, Any]
    type_orientation: str = "forward"


@dataclass(frozen=True)
class VerificationAttempt:
    attempt_id: str
    hypothesis_id: str
    attempt_no: int
    verdict: str
    reason_code: str
    supported_evidence_spans: tuple[EvidenceSpan, ...]
    independent_call: bool
    verifier_prompt_version: str
    verifier_schema_version: str
    verifier_parser_version: str
    verifier_model_revision: str
    forward_entailment: str = ""
    reverse_entailment: str = ""
    subject_role_supported: bool = False
    object_role_supported: bool = False
    subject_context_role: str = ""
    object_context_role: str = ""
    subject_role_alignment: str = ""
    object_role_alignment: str = ""
    forward_role_alignment: str = ""
    reverse_role_alignment: str = ""
    direction_normalization: str = "FORWARD"
    normalized_subject_occurrence_key: str = ""
    normalized_object_occurrence_key: str = ""
    error_message: str = ""
    model_request_payload: dict[str, Any] = field(default_factory=dict)
    raw_response_text: str = ""
    parsed_response: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedRelationFact:
    fact_id: str
    semantic_fact_key: str
    project_id: int
    subject_canonical_id: str
    predicate: str
    object_canonical_id: str
    symmetric: bool
    reflexive: bool
    review_only: bool
    normalization_eligible: bool
    evidence_count: int
    reason_code: str = "NORMALIZED"


@dataclass(frozen=True)
class RelationFactEvidence:
    evidence_id: str
    evidence_key: str
    fact_id: str
    document_revision_id: str
    subject_occurrence_key: str
    object_occurrence_key: str
    evidence_spans: tuple[EvidenceSpan, ...]
    hypothesis_id: str
    verification_attempt_id: str
    verifier_supported: bool
    provenance: dict[str, Any]


@dataclass(frozen=True)
class RelationFactConflict:
    conflict_id: str
    pair_identity: str
    hypothesis_id: str
    reason_code: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RelationNormalizationResult:
    facts: tuple[NormalizedRelationFact, ...]
    evidence: tuple[RelationFactEvidence, ...]
    conflicts: tuple[RelationFactConflict, ...]
    collapse_audit: tuple[dict[str, Any], ...]
    metrics: dict[str, int]


class StaleRelationSnapshotError(RuntimeError):
    """A recorded relation artifact no longer matches current behavior inputs."""


@dataclass(frozen=True)
class RelationMaterializationFingerprint:
    version: str
    root: str
    components: dict[str, Any]
    component_hashes: dict[str, str]


@dataclass(frozen=True)
class RelationFreshnessDecision:
    checkpoint: str
    current: bool
    status: str
    recorded_root: str
    current_root: str
    changed_components: tuple[str, ...]


@dataclass(frozen=True)
class RelationMaterializationDecision:
    fact_id: str
    eligible: bool
    reasons: tuple[str, ...]
    freshness: RelationFreshnessDecision


def _type_match(actual: str, expected: tuple[str, ...]) -> bool | None:
    normalized_expected = {str(value).strip().lower() for value in expected}
    if not normalized_expected or "*" in normalized_expected:
        return True
    normalized_actual = str(actual or "").strip().lower()
    if normalized_actual in _UNKNOWN_TYPES:
        return None
    return normalized_actual in normalized_expected


def _predicate_metadata(
    left_type: str,
    right_type: str,
    logical_schema: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], str]:
    metadata: list[dict[str, Any]] = []
    observed: list[bool | None] = []
    for name, spec in normalize_predicate_schema(logical_schema).items():
        direct_left = _type_match(left_type, spec.domain_types)
        direct_right = _type_match(right_type, spec.range_types)
        reverse_left = _type_match(right_type, spec.domain_types)
        reverse_right = _type_match(left_type, spec.range_types)

        def combine(first: bool | None, second: bool | None) -> bool | None:
            if first is False or second is False:
                return False
            if first is None or second is None:
                return None
            return True

        left_to_right = combine(direct_left, direct_right)
        right_to_left = combine(reverse_left, reverse_right)
        observed.extend((left_to_right, right_to_left))
        if left_to_right is not False or right_to_left is not False:
            metadata.append({
                **asdict(spec),
                "left_to_right": left_to_right,
                "right_to_left": right_to_left,
            })
    if any(value is True for value in observed):
        return tuple(metadata), "compatible"
    if any(value is None for value in observed):
        return tuple(metadata), "unresolved"
    return tuple(metadata), "incompatible"


def normalize_predicate_schema(
    logical_schema: Mapping[str, Any],
) -> dict[str, PredicateSpec]:
    """Validate the complete versioned predicate policy without name defaults."""

    schema_version = str(
        logical_schema.get("version")
        or logical_schema.get("schema_version")
        or ""
    )
    if not schema_version:
        raise ValueError("Relation Schema requires a version")
    relation_types = logical_schema.get("relation_types")
    if not isinstance(relation_types, list) or not relation_types:
        raise ValueError("Relation Schema requires relation_types")
    result: dict[str, PredicateSpec] = {}
    required = {
        "name",
        "semantic_definition",
        "domain_types",
        "range_types",
        "direction_description",
        "symmetric",
        "inverse_predicate",
        "canonical_orientation",
        "publish_mode",
        "allowed_source_roles",
        "scope_policy",
        "positive_examples",
        "negative_examples",
        "reflexive",
    }
    for index, raw in enumerate(relation_types):
        if not isinstance(raw, Mapping):
            raise ValueError(f"predicate metadata #{index} must be an object")
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                f"predicate metadata #{index} missing required fields: {missing}"
            )
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ValueError(f"predicate metadata #{index} has empty name")
        if name in result:
            raise ValueError(f"duplicate predicate metadata: {name}")
        semantic_definition = str(
            raw.get("semantic_definition") or ""
        ).strip()
        direction_description = str(
            raw.get("direction_description") or ""
        ).strip()
        if not semantic_definition or not direction_description:
            raise ValueError(
                f"predicate {name} requires semantic and direction definitions"
            )
        domain_types = raw.get("domain_types")
        range_types = raw.get("range_types")
        if (
            not isinstance(domain_types, list)
            or not domain_types
            or not all(str(item).strip() for item in domain_types)
            or not isinstance(range_types, list)
            or not range_types
            or not all(str(item).strip() for item in range_types)
        ):
            raise ValueError(
                f"predicate {name} requires non-empty domain_types/range_types"
            )
        for field_name in (
            "positive_examples", "negative_examples", "optional_soft_cues",
            "positive_cues", "negative_cues", "anti_patterns",
            "allowed_context_kinds", "required_evidence_roles",
            "allowed_source_roles", "identity_qualifier_keys", "conflicts_with",
            "structural_evidence_kinds", "structural_role_labels",
            "source_role_pairs",
        ):
            value = raw.get(field_name, [])
            if (
                not isinstance(value, list)
                or not all(isinstance(item, str) for item in value)
            ):
                raise ValueError(
                    f"predicate {name} {field_name} must be a string array"
                )
        allowed_source_roles = [
            str(item).strip() for item in raw.get("allowed_source_roles") or []
        ]
        if not allowed_source_roles or not all(allowed_source_roles):
            raise ValueError(
                f"predicate {name} requires allowed_source_roles"
            )
        source_role_pairs = tuple(dict.fromkeys(
            str(item).strip()
            for item in raw.get("source_role_pairs") or []
            if str(item).strip()
        ))
        invalid_source_role_pairs = sorted(
            set(source_role_pairs) - _VALID_SOURCE_ROLE_PAIRS
        )
        if invalid_source_role_pairs:
            raise ValueError(
                f"predicate {name} source_role_pairs contains unsupported "
                f"contracts: {invalid_source_role_pairs}"
            )
        scope_policy_raw = raw.get("scope_policy")
        if not isinstance(scope_policy_raw, Mapping):
            raise ValueError(f"predicate {name} scope_policy must be an object")
        scope_policy = {
            "require_same_atomic_scope": scope_policy_raw.get(
                "require_same_atomic_scope"
            ),
            "allow_cross_scope_with_explicit_evidence": scope_policy_raw.get(
                "allow_cross_scope_with_explicit_evidence"
            ),
        }
        if not all(isinstance(value, bool) for value in scope_policy.values()):
            raise ValueError(
                f"predicate {name} scope_policy requires boolean boundary fields"
            )
        if (
            scope_policy["require_same_atomic_scope"]
            and scope_policy["allow_cross_scope_with_explicit_evidence"]
        ):
            raise ValueError(
                f"predicate {name} scope_policy has conflicting boundary rules"
            )
        if not isinstance(raw.get("symmetric"), bool):
            raise ValueError(f"predicate {name} symmetric must be boolean")
        if not isinstance(raw.get("reflexive"), bool):
            raise ValueError(f"predicate {name} reflexive must be boolean")
        inverse_value = raw.get("inverse_predicate")
        if inverse_value is not None and not isinstance(inverse_value, str):
            raise ValueError(
                f"predicate {name} inverse_predicate must be string or null"
            )
        inverse_predicate = str(inverse_value or "").strip()
        canonical_orientation = str(
            raw.get("canonical_orientation") or ""
        )
        if canonical_orientation not in {
            CANONICAL_FORWARD, CANONICAL_INVERSE, CANONICAL_SYMMETRIC,
        }:
            raise ValueError(
                f"predicate {name} canonical_orientation is invalid"
            )
        publish_mode = str(raw.get("publish_mode") or "")
        if publish_mode not in {
            PUBLISH_AUTO, PUBLISH_REVIEW_ONLY, PUBLISH_DISABLED,
        }:
            raise ValueError(f"predicate {name} publish_mode is invalid")
        symmetric = bool(raw["symmetric"])
        if symmetric != (canonical_orientation == CANONICAL_SYMMETRIC):
            raise ValueError(
                f"predicate {name} symmetric/orientation metadata conflicts"
            )
        if canonical_orientation == CANONICAL_INVERSE and not inverse_predicate:
            raise ValueError(
                f"predicate {name} inverse orientation requires inverse_predicate"
            )
        if symmetric and inverse_predicate:
            raise ValueError(
                f"predicate {name} symmetric predicate cannot declare inverse"
            )
        qualifier_schema = raw.get("qualifier_schema") or {}
        if not isinstance(qualifier_schema, Mapping):
            raise ValueError(f"predicate {name} qualifier_schema must be an object")
        if not isinstance(raw.get("reciprocal_allowed", False), bool):
            raise ValueError(f"predicate {name} reciprocal_allowed must be boolean")
        negation_policy = str(
            raw.get("negation_policy") or REJECT_NEGATED_RELATION
        )
        if negation_policy not in {
            REJECT_NEGATED_RELATION, ALLOW_EXPLICIT_NEGATIVE_RELATION,
        }:
            raise ValueError(
                f"predicate {name} negation_policy is unsupported"
            )
        modality_policy = str(
            raw.get("modality_policy") or REQUIRE_ASSERTED_MODALITY
        )
        if modality_policy != REQUIRE_ASSERTED_MODALITY:
            raise ValueError(
                f"predicate {name} modality_policy is unsupported"
            )
        open_relation_raw = raw.get("open_relation_policy") or {}
        if not isinstance(open_relation_raw, Mapping):
            raise ValueError(
                f"predicate {name} open_relation_policy must be an object"
            )
        open_relation_policy = {
            "enabled": open_relation_raw.get("enabled", False),
            "fallback_only": open_relation_raw.get("fallback_only", False),
            "display_source_expression": open_relation_raw.get(
                "display_source_expression", False,
            ),
            "identity_source_expression": open_relation_raw.get(
                "identity_source_expression", False,
            ),
            "require_exact_source_expression": open_relation_raw.get(
                "require_exact_source_expression", False,
            ),
            "allowed_candidate_sources": list(
                open_relation_raw.get("allowed_candidate_sources") or []
            ),
        }
        if not all(
            isinstance(open_relation_policy[field], bool)
            for field in (
                "enabled", "fallback_only", "display_source_expression",
                "identity_source_expression",
                "require_exact_source_expression",
            )
        ):
            raise ValueError(
                f"predicate {name} open_relation_policy flags must be boolean"
            )
        if not all(
            isinstance(value, str) and value.strip()
            for value in open_relation_policy["allowed_candidate_sources"]
        ):
            raise ValueError(
                f"predicate {name} open_relation_policy candidate sources "
                "must be non-empty strings"
            )
        open_relation_policy["allowed_candidate_sources"] = list(
            dict.fromkeys(
                str(value).strip().upper()
                for value in open_relation_policy["allowed_candidate_sources"]
            )
        )
        if not open_relation_policy["enabled"] and any(
            open_relation_policy[field]
            for field in (
                "fallback_only", "display_source_expression",
                "identity_source_expression",
                "require_exact_source_expression",
            )
        ):
            raise ValueError(
                f"predicate {name} has disabled open_relation_policy with "
                "enabled-only flags"
            )
        if open_relation_policy["enabled"] and (
            publish_mode != PUBLISH_AUTO
            or not open_relation_policy["allowed_candidate_sources"]
        ):
            raise ValueError(
                f"predicate {name} open relation carrier requires auto publish "
                "and allowed candidate sources"
            )
        result[name] = PredicateSpec(
            name=name,
            semantic_definition=semantic_definition,
            domain_types=tuple(str(item) for item in domain_types),
            range_types=tuple(str(item) for item in range_types),
            direction_description=direction_description,
            subject_role_description=str(
                raw.get("subject_role_description")
                or "The subject endpoint role follows direction_description."
            ),
            object_role_description=str(
                raw.get("object_role_description")
                or "The object endpoint role follows direction_description."
            ),
            symmetric=symmetric,
            inverse_predicate=inverse_predicate,
            canonical_orientation=canonical_orientation,
            publish_mode=publish_mode,
            reflexive=bool(raw["reflexive"]),
            reciprocal_allowed=bool(raw.get("reciprocal_allowed", False)),
            allowed_context_kinds=tuple(
                str(value) for value in raw.get("allowed_context_kinds") or []
            ),
            allowed_source_roles=tuple(allowed_source_roles),
            source_role_pairs=source_role_pairs,
            scope_policy=scope_policy,
            required_evidence_roles=tuple(
                str(value) for value in raw.get("required_evidence_roles") or []
            ),
            positive_examples=tuple(
                str(value) for value in raw["positive_examples"]
            ),
            negative_examples=tuple(
                str(value) for value in raw["negative_examples"]
            ),
            positive_cues=tuple(
                str(value) for value in raw.get("positive_cues") or []
            ),
            negative_cues=tuple(
                str(value) for value in raw.get("negative_cues") or []
            ),
            anti_patterns=tuple(
                str(value) for value in raw.get("anti_patterns") or []
            ),
            optional_soft_cues=tuple(
                str(value) for value in raw.get("optional_soft_cues") or []
            ),
            structural_evidence_kinds=tuple(
                str(value) for value in raw.get("structural_evidence_kinds") or []
            ),
            structural_role_labels=tuple(
                str(value) for value in raw.get("structural_role_labels") or []
            ),
            qualifier_schema=dict(qualifier_schema),
            identity_qualifier_keys=tuple(
                str(value) for value in raw.get("identity_qualifier_keys") or []
            ),
            conflicts_with=tuple(
                str(value) for value in raw.get("conflicts_with") or []
            ),
            negation_policy=negation_policy,
            modality_policy=modality_policy,
            open_relation_policy=open_relation_policy,
            schema_version=schema_version,
        )
    for spec in result.values():
        if spec.inverse_predicate and spec.inverse_predicate not in result:
            raise ValueError(
                f"predicate {spec.name} references unknown inverse predicate"
            )
        if spec.canonical_orientation == CANONICAL_INVERSE:
            inverse = result[spec.inverse_predicate]
            if inverse.canonical_orientation != CANONICAL_FORWARD:
                raise ValueError(
                    f"predicate {spec.name} inverse target must be canonical forward"
                )
    return result


def predicate_policy_fingerprint_payload(
    logical_schema: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return every validated behavior field in deterministic predicate order."""

    return {
        name: asdict(spec)
        for name, spec in sorted(normalize_predicate_schema(logical_schema).items())
    }


def classifier_response_schema(
    opportunities: Sequence[PairOpportunity],
) -> dict[str, Any]:
    """Return the versioned batch schema; endpoint order is the only direction."""

    count = len(opportunities)
    predicate_names = sorted({
        str(metadata.get("name") or "")
        for item in opportunities
        for metadata in item.allowed_predicate_metadata
        if str(metadata.get("name") or "")
        and not _fallback_only_open_relation_policy(metadata)
    })
    predicate_schema: dict[str, Any] = {"type": "string"}
    if predicate_names:
        predicate_schema["enum"] = predicate_names
    decision_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "pair_id": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": [NO_RELATION, OTHER_UNSUPPORTED, HYPOTHESES],
            },
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "subject_occurrence_id": {
                            "type": "string",
                            "enum": ["LEFT", "RIGHT"],
                        },
                        "predicate": predicate_schema,
                        "object_occurrence_id": {
                            "type": "string",
                            "enum": ["LEFT", "RIGHT"],
                        },
                        "evidence_spans": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start": {
                                        "type": "integer",
                                        "minimum": 0,
                                    },
                                    "end": {
                                        "type": "integer",
                                        "minimum": 1,
                                    },
                                },
                                "required": ["start", "end"],
                            },
                        },
                        "classifier_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                    "required": [
                        "subject_occurrence_id",
                        "predicate",
                        "object_occurrence_id",
                        "evidence_spans",
                        "classifier_confidence",
                    ],
                },
            },
            "evidence_spans": {"type": "array"},
            "reason_code": {"type": "string"},
        },
        "required": ["pair_id", "decision", "hypotheses"],
    }
    if count == 1:
        return {
            "$id": GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
            **decision_schema,
        }
    return {
        "$id": GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": decision_schema,
            },
        },
        "required": ["decisions"],
    }


def classifier_prompt(
    opportunities: Sequence[PairOpportunity],
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    protocol_feedback: str = "",
    macro_context: Mapping[str, Any] | None = None,
) -> str:
    """Build a predicate-agnostic classifier request from versioned metadata."""

    pairs = []
    predicate_catalog: dict[str, dict[str, Any]] = {}
    context_catalog: dict[str, dict[str, Any]] = {}
    context_refs: dict[str, str] = {}
    for item in opportunities:
        context_window = _pair_model_context(
            item,
            content=content,
            max_chars=Config.relation_classifier_context_max_chars,
        )
        context_value = {
            "source_span": list(context_window["source_span"]),
            "text": str(context_window["text"]),
            "visible_segments": list(context_window["visible_segments"]),
            "truncated": bool(context_window["truncated"]),
        }
        context_key = json.dumps(
            context_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        context_ref = context_refs.get(context_key)
        if context_ref is None:
            context_ref = f"CTX-{len(context_catalog) + 1}"
            context_refs[context_key] = context_ref
            context_catalog[context_ref] = context_value
        visible_metadata = [
            value for value in item.allowed_predicate_metadata
            if not _fallback_only_open_relation_policy(value)
        ]
        allowed_names = [str(value["name"]) for value in visible_metadata]
        for value in visible_metadata:
            name = str(value["name"])
            projected = candidate_predicate_prompt_metadata(value)
            existing = predicate_catalog.get(name)
            if existing is not None and existing != projected:
                raise ValueError(
                    "classifier predicate metadata differs within one batch"
                )
            predicate_catalog[name] = projected
        pairs.append({
            "pair_id": item.pair_identity,
            "left_occurrence_id": "LEFT",
            "right_occurrence_id": "RIGHT",
            "left_span": list(item.left_span),
            "right_span": list(item.right_span),
            "left_surface": content[item.left_span[0]:item.left_span[1]],
            "right_surface": content[item.right_span[0]:item.right_span[1]],
            "left_type": item.left_type,
            "right_type": item.right_type,
            "context_span": list(item.context_span),
            "context_ref": context_ref,
            "allowed_predicates": allowed_names,
            "soft_cue_matches": list(
                item.provenance.get("soft_cue_matches") or []
            ),
            "structural_evidence_contract": {
                "source_roles": list(item.provenance.get("source_roles") or []),
                "atomic_scopes": list(item.provenance.get("atomic_scopes") or []),
                "required_atomic_scope": list(
                    item.provenance.get("required_atomic_scope") or []
                ),
                "hierarchical_parent_child": bool(
                    item.provenance.get("hierarchical_list_parent_child")
                    and item.provenance.get("structure_admission")
                    == "hierarchical_list_parent_child"
                ),
                "section_label_parent_child": bool(
                    item.provenance.get("hierarchical_section_label_parent_child")
                    and item.provenance.get("structure_admission")
                    == "hierarchical_section_label_parent_child"
                ),
            },
        })
    if len(opportunities) == 1:
        response_shape = (
            "必须只返回一个 JSON 对象，且该对象只含 pair_id、decision、"
            "hypotheses，以及可选的 evidence_spans、reason_code。没有关系时"
            "也必须返回该 pair 的 NO_RELATION 决策对象；不要返回数组、"
            "decisions、items 或 result。"
        )
    else:
        response_shape = (
            f"必须只返回一个 JSON 对象，且只含 decisions 字段；decisions 是长度恰好"
            f"为 {len(opportunities)} 的数组。空 decisions 数组是协议错误；没有关系时"
            "也必须返回该 pair 的 NO_RELATION 决策对象。每个 decisions 元素只含 "
            "pair_id、decision、hypotheses，以及可选的 evidence_spans、reason_code。"
            "不要返回 items、result 或其他包装对象。"
        )
    instruction = (
        f"contract={GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION}; "
        f"schema={GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION}. "
        "对每个 pair 恰好返回 NO_RELATION、OTHER_UNSUPPORTED 或 HYPOTHESES。"
        "subject_occurrence_id 与 object_occurrence_id 的顺序是方向唯一真相，"
        "且这两个字段只能逐字使用 LEFT 或 RIGHT；不要复制内部 occurrence "
        "哈希，不要输出 direction 字段。一个 pair 可返回多个具体 predicate。"
        "predicate_catalog 在批次内只定义一次；每个 pair 的 allowed_predicates 是"
        "该目录的硬子集。每个 hypothesis 的 predicate 必须逐字来自该 pair 的 "
        "allowed_predicates；"
        "如果你想表达的 predicate 不在该 pair 的 allowed_predicates 中，不得输出"
        "该 hypothesis，应返回 NO_RELATION，或在文本明确有关系但当前 Schema 无法"
        "表达时返回 OTHER_UNSUPPORTED。"
        "只按每个 predicate 的 semantic_definition、端点角色、"
        "direction_description、domain_types、range_types 和结构策略判断语义及方向。"
        "optional_soft_cues 只是可配置提示，不能作为关系、方向或发布的证据。"
        "结构共现本身不等于关系；Schema 无法表示但文本明确表达关系时返回 "
        "OTHER_UNSUPPORTED；没有语义关系返回 NO_RELATION。"
        "macro_context 只是项目级主题和实体覆盖先验，不是本地关系证据；不得根据其中的"
        "共现、频次或名称补造关系，关系端点和证据仍必须来自当前冻结 context。"
        "每个 pair 用 context_ref 引用批次内唯一的 context_catalog 项；该项可能是围绕"
        "精确端点的有界显示窗口。pair.context_span 仍是完整冻结来源坐标，catalog 中的 "
        "visible_segments 列出实际可见的绝对区间，省略区间不可用于推理或证据。"
        "每个 pair 的 structural_evidence_contract 是硬边界；不得跨越 required_atomic_scope，"
        "不得因表格、普通列表或版面相邻而假定关系。若 section_label_parent_child=true，"
        "表示源结构确认了一个独立的、紧邻后续内容的章节/字段标签与其直接内容单元；"
        "这只允许在 Schema 明确定义的层级/归属语义和端点角色均匹配时作为候选证据，"
        "不能把普通段落相邻升级为关系，也不能推导兄弟项。"
        "evidence_spans 必须是冻结 context 内的绝对半开区间。"
        f"{response_shape}"
        "pair_id 必须逐字复制输入。decision=HYPOTHESES 时，hypotheses 中每个元素"
        "只含 subject_occurrence_id、predicate、object_occurrence_id、"
        "evidence_spans、classifier_confidence；evidence_spans 中每项必须是"
        "{\"start\":整数,\"end\":整数}，且这些 span 必须共同覆盖所选 subject "
        "与 object 的精确 occurrence span；只覆盖关系词不合格。为避免边界漂移，"
        "每个 hypothesis 的 evidence_spans 必须直接使用该 pair 的完整 "
        "context_span，即 [{\"start\":context_span[0],\"end\":context_span[1]}]。"
        "decision 为 "
        "NO_RELATION 或 "
        "OTHER_UNSUPPORTED 时 hypotheses 必须为空数组。不要返回 predicates、"
        "subject、object、direction、reasoning 或其他包装对象。"
    )
    return json.dumps(
        {
        "instruction": instruction,
            "protocol_feedback": str(protocol_feedback or ""),
            "macro_context": dict(macro_context or {}),
            "predicate_catalog": {
                name: predicate_catalog[name]
                for name in sorted(predicate_catalog)
            },
            "context_catalog": context_catalog,
            "pairs": pairs,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def proposer_response_schema(
    *,
    mentions: Sequence[Mapping[str, Any]] = (),
    logical_schema: Mapping[str, Any] | None = None,
    content_length: int | None = None,
    max_proposals: int | None = None,
    required_evidence_span: tuple[int, int] | None = None,
) -> dict[str, Any]:
    occurrence_ids = sorted({
        str(item.get("occurrence_key") or "")
        for item in mentions
        if str(item.get("occurrence_key") or "")
    })
    predicate_names = sorted(
        name for name, spec in normalize_predicate_schema(
            logical_schema or {},
        ).items()
        if not _fallback_only_open_relation_policy(spec)
    )
    subject_schema: dict[str, Any] = {"type": "string"}
    object_schema: dict[str, Any] = {"type": "string"}
    predicate_schema: dict[str, Any] = {"type": "string"}
    if occurrence_ids:
        subject_schema["enum"] = occurrence_ids
        object_schema["enum"] = occurrence_ids
    if predicate_names:
        predicate_schema["enum"] = predicate_names
    start_schema: dict[str, Any] = {"type": "integer", "minimum": 0}
    end_schema: dict[str, Any] = {"type": "integer", "minimum": 1}
    if content_length is not None:
        max_end = max(1, int(content_length))
        start_schema["maximum"] = max(0, max_end - 1)
        end_schema["maximum"] = max_end
    if required_evidence_span is not None:
        required_start, required_end = map(int, required_evidence_span)
        if not (0 <= required_start < required_end):
            raise ValueError("proposer evidence scope must be a non-empty span")
        if content_length is not None and required_end > int(content_length):
            raise ValueError("proposer evidence scope exceeds content")
        start_schema["minimum"] = required_start
        start_schema["maximum"] = max(required_start, required_end - 1)
        end_schema["minimum"] = required_start + 1
        end_schema["maximum"] = required_end
    proposal_array_schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "subject_occurrence_id": subject_schema,
                "predicate": predicate_schema,
                "object_occurrence_id": object_schema,
                "evidence_spans": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "start": start_schema,
                            "end": end_schema,
                        },
                        "required": ["start", "end"],
                    },
                },
                "qualifiers": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {
                                "type": "string",
                                "maxLength": 64,
                            },
                            "value": {
                                "type": "string",
                                "maxLength": 256,
                            },
                        },
                        "required": ["name", "value"],
                    },
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "subject_occurrence_id",
                "predicate",
                "object_occurrence_id",
                "evidence_spans",
                "qualifiers",
                "confidence",
            ],
        },
    }
    if max_proposals is not None:
        proposal_array_schema["maxItems"] = max(0, int(max_proposals))
    return {
        "$id": GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposals": proposal_array_schema,
            "missing_endpoint_mentions": {
                "type": "array",
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "surface": {
                            "type": "string",
                            "maxLength": 160,
                        },
                        "suggested_type": {
                            "type": "string",
                            "maxLength": 80,
                        },
                        "reason": {
                            "type": "string",
                            "maxLength": 240,
                        },
                    },
                    "required": ["surface", "suggested_type", "reason"],
                },
            },
            "coverage": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "complete": {"type": "boolean"},
                    "uncovered_cues": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {
                            "type": "string",
                            "maxLength": 160,
                        },
                    },
                },
                "required": ["complete", "uncovered_cues"],
            },
        },
        "required": ["proposals", "missing_endpoint_mentions", "coverage"],
    }


def proposer_prompt(
    *,
    context_id: str,
    content: str,
    mentions: Sequence[Mapping[str, Any]],
    logical_schema: Mapping[str, Any],
    already_found_keys: Sequence[str] = (),
    pass_no: int = 0,
    protocol_feedback: str = "",
    max_proposals: int | None = None,
    required_evidence_span: tuple[int, int] | None = None,
    macro_context: Mapping[str, Any] | None = None,
) -> str:
    predicates = {
        name: candidate_predicate_prompt_metadata(asdict(spec))
        for name, spec in normalize_predicate_schema(logical_schema).items()
        if not _fallback_only_open_relation_policy(spec)
    }
    max_proposals_value = (
        max(0, int(max_proposals))
        if max_proposals is not None else None
    )
    occurrences = [{
        "occurrence_id": str(item.get("occurrence_key") or ""),
        "surface": str(item.get("surface") or ""),
        "type": str(item.get("type") or ""),
        "span": [
            int(item.get("mention_start") or 0),
            int(item.get("mention_end") or 0),
        ],
        "canonical_id": str(item.get("canonical_id") or ""),
    } for item in mentions]
    if required_evidence_span is None:
        focus_start, focus_end = 0, len(content)
    else:
        focus_start, focus_end = map(int, required_evidence_span)
        if not (0 <= focus_start < focus_end <= len(content)):
            raise ValueError("proposer evidence scope is outside frozen content")
    focus_text = content[focus_start:focus_end]
    if any(
        not (
            focus_start <= int(item.get("mention_start") or 0)
            < int(item.get("mention_end") or 0) <= focus_end
        )
        for item in mentions
    ):
        raise ValueError("proposer occurrence crosses the evidence scope")
    uses_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", focus_text))
    proposal_limit = (
        f"本次最多返回 {max_proposals_value} 条候选。"
        if uses_cjk and max_proposals_value is not None
        else f"Return at most {max_proposals_value} proposals for this context. "
        if max_proposals_value is not None
        else ""
    )
    instruction = (
        f"contract={GROUNDED_RELATION_PROPOSER_PROMPT_VERSION}; "
        f"schema={GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION}。"
        "只抽取冻结上下文明确表达的完整局部关系子图。subject_occurrence_id 和 "
        "object_occurrence_id 必须逐字复制 occurrences 中的编号，predicate 必须逐字复制 "
        "predicates 中的名称；不得发明端点、谓词、证据、限定条件或规范实体。"
        "逐句识别谓词及其精确论元。一个谓词若支配多个并列主语或并列宾语，应把同一个"
        "关系分别展开到每个并列成员；并列成员只是共享同一支配项，除非原文明说，不得"
        "把并列成员彼此连接。被动句、包含句和列举句必须先找出共同的施事、整体或其他"
        "支配项，再连接到每个成员，不能把某个成员误当成其他成员的支配项。"
        "解析嵌套从句和修饰语后再选端点；修饰语内部的实体不能替代真正承担关系角色的"
        "中心实体。事件、状态、条件端点必须保留触发词、状态变化、否定和情态等构成完整"
        "论元的部分，不能只取内部名词。一个句子有多个关系时，分别输出各自精确论元。"
        "只有原文没有明确关系时 proposals 才能为空。缺少必要端点时只写入 "
        "missing_endpoint_mentions，不能据此生成 proposal。coverage.complete 只有在所有"
        "独立关系（包括并列展开）均已覆盖时才能为 true。限定条件只使用 name/value；"
        "缺失端点只使用 surface/suggested_type/reason，审计文字保持简短。"
        "evidence_spans 优先使用给定 required_evidence_span；不确定偏移时不要自行计算。"
        "输出必须使用扁平字段 subject_occurrence_id、predicate、object_occurrence_id、"
        "evidence_spans、qualifiers、confidence；禁止返回 subject_endpoint、object_endpoint、"
        "evidence_span 或 canonical_id 嵌套结构。"
        f"{proposal_limit}optional_soft_cues 仅是软提示，不能决定关系、方向或发布。"
        "所有候选仍须经过确定性校验和独立核验。"
        if uses_cjk else
        f"contract={GROUNDED_RELATION_PROPOSER_PROMPT_VERSION}; "
        f"schema={GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION}. "
        "Extract the complete local relation subgraph explicitly supported by this "
        "frozen context. Copy endpoint ids from occurrences and predicates from the "
        "Schema. Do not invent endpoints, predicates, evidence, qualifiers, or "
        "canonical entities. Expand a shared predicate across every coordinated "
        "subject or object, but never infer relations among peer list members merely "
        "because they share a governor. In passive, containment, and enumeration "
        "constructions, identify the shared agent, whole, or other governor before "
        "connecting each member. Resolve nested clauses and modifiers before choosing "
        "exact endpoints. Event, state, and condition endpoints must retain the trigger, "
        "state change, negation, modality, and other text needed to form a complete "
        "argument; do not promote an inner noun. Emit every independently supported "
        "relation, including multiple predicates for a pair. Empty proposals is valid "
        "only when no explicit relation is present. Missing endpoints are audit-only. "
        "Always return coverage as an object. Set coverage.complete true only after "
        "covering coordinated members as well as "
        "ordinary clauses. Keep qualifiers and audit text compact. Prefer the supplied "
        f"required_evidence_span. {proposal_limit}optional_soft_cues are hints only. "
        "Use only the flat fields subject_occurrence_id, predicate, object_occurrence_id, "
        "evidence_spans, qualifiers, and confidence; never return nested "
        "subject_endpoint/object_endpoint/canonical_id or singular evidence_span fields. "
        "Every proposal still requires deterministic validation and an independent "
        "verifier. The macro_context is only a project-level topic/entity-coverage "
        "prior, never local relation evidence: do not infer a relation from its "
        "counts, names, or co-occurrences; endpoints and evidence must remain in "
        "the frozen context."
    )
    return json.dumps(
        {
            "instruction": instruction,
            "protocol_feedback": str(protocol_feedback or ""),
            "macro_context": dict(macro_context or {}),
            "max_proposals": max_proposals_value,
            "context": {
                "context_id": context_id,
                "text": focus_text,
                "start": focus_start,
                "end": focus_end,
            },
            "required_evidence_span": {
                "start": focus_start, "end": focus_end,
            },
            "pass_no": int(pass_no),
            "occurrences": occurrences,
            "predicates": predicates,
            "already_found_keys": list(already_found_keys),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_proposer_audit_fields(
    raw: Mapping[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Normalize audit-only proposer fields without accepting unsafe facts."""

    repairs: list[str] = []
    missing_raw = raw.get("missing_endpoint_mentions")
    if missing_raw is None:
        missing: list[Any] = []
        repairs.append("missing_endpoint_mentions_defaulted")
    elif isinstance(missing_raw, list):
        missing = missing_raw
    else:
        missing = []
        repairs.append("missing_endpoint_mentions_type_defaulted")

    coverage_raw = raw.get("coverage")
    if isinstance(coverage_raw, Mapping):
        coverage = dict(coverage_raw)
    else:
        coverage = {}
        repairs.append("coverage_type_defaulted")

    complete_raw = coverage.get("complete")
    if not isinstance(complete_raw, bool):
        coverage["complete"] = False
        repairs.append("coverage_complete_defaulted")

    uncovered_raw = coverage.get("uncovered_cues")
    if not isinstance(uncovered_raw, list):
        coverage["uncovered_cues"] = []
        repairs.append("coverage_uncovered_cues_defaulted")

    if repairs:
        coverage["_parser_repairs"] = repairs
    return missing, coverage


def _parse_proposer_spans(
    raw: Any,
    *,
    content: str,
    endpoint_spans: Sequence[tuple[int, int]],
    repairs: list[str] | None = None,
    proposal_index: int = 0,
    required_evidence_span: tuple[int, int] | None = None,
) -> tuple[EvidenceSpan, ...]:
    if required_evidence_span is None:
        required_start, required_end = 0, len(content)
    else:
        required_start, required_end = map(int, required_evidence_span)
        if not (0 <= required_start < required_end <= len(content)):
            raise RelationClassificationError(
                "proposal evidence scope is outside frozen context"
            )
    fallback = (EvidenceSpan(start=required_start, end=required_end),)
    repair_prefix = f"proposal_{int(proposal_index)}"
    if len(content) <= 0:
        raise RelationClassificationError("proposal context is empty")

    def fallback_to_context(reason: str) -> tuple[EvidenceSpan, ...]:
        if repairs is not None:
            repairs.append(f"{repair_prefix}_{reason}_defaulted_to_context")
        return fallback

    if not isinstance(raw, list):
        return fallback_to_context("evidence_spans")
    spans: list[EvidenceSpan] = []
    for value in raw:
        if not isinstance(value, Mapping):
            return fallback_to_context("evidence_span")
        try:
            start, end = int(value["start"]), int(value["end"])
        except (KeyError, TypeError, ValueError) as exc:
            if repairs is not None:
                repairs.append(f"{repair_prefix}_evidence_span_invalid")
            return fallback
        if (
            start < required_start
            or end <= start
            or end > required_end
        ):
            return fallback_to_context("evidence_span_outside_context")
        spans.append(EvidenceSpan(start=start, end=end))
    if not spans:
        return fallback_to_context("evidence_empty")
    for required_span in endpoint_spans:
        if not _spans_cover(spans, required_span):
            return fallback_to_context("evidence_missing_endpoint")
    return tuple(spans)


def parse_proposer_response(
    *,
    context_id: str,
    content: str,
    mentions: Sequence[Mapping[str, Any]],
    logical_schema: Mapping[str, Any],
    raw: Any,
    candidate_source: str = DIRECT_PROPOSER,
    pass_no: int = 0,
    proposer_policy_version: str = GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    proposer_model_revision: str = "unspecified",
    required_evidence_span: tuple[int, int] | None = None,
) -> RelationProposalBatch:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError("proposer returned invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError("proposer response must be an object")
    proposals_raw = raw.get("proposals")
    missing_raw, coverage = _normalize_proposer_audit_fields(raw)
    repairs = coverage.setdefault("_parser_repairs", [])
    if not isinstance(repairs, list):
        repairs = []
        coverage["_parser_repairs"] = repairs
    if proposals_raw is None:
        proposals_raw = []
        repairs.append("proposals_defaulted")
    elif isinstance(proposals_raw, Mapping):
        proposals_raw = [proposals_raw]
        repairs.append("single_proposal_object_wrapped")
    elif not isinstance(proposals_raw, list):
        proposals_raw = []
        repairs.append("proposals_type_defaulted")
    mentions_by_key = {
        str(item.get("occurrence_key") or ""): dict(item) for item in mentions
    }
    if any(not key for key in mentions_by_key):
        raise RelationClassificationError("proposer input occurrence id is empty")
    if required_evidence_span is not None:
        required_start, required_end = map(int, required_evidence_span)
        if not (0 <= required_start < required_end <= len(content)):
            raise RelationClassificationError(
                "proposer evidence scope is outside frozen context"
            )
        if any(
            not (
                required_start <= int(item.get("mention_start") or 0)
                < int(item.get("mention_end") or 0) <= required_end
            )
            for item in mentions_by_key.values()
        ):
            raise RelationClassificationError(
                "proposer occurrence crosses the evidence scope"
            )
    schema = normalize_predicate_schema(logical_schema)
    proposals: list[RelationProposal] = []
    seen: set[str] = set()
    rejected = coverage.setdefault("_parser_rejected_proposals", [])
    if not isinstance(rejected, list):
        rejected = []
        coverage["_parser_rejected_proposals"] = rejected

    # Some configured local models continue to emit the pre-v5 proposer
    # envelope even when the current JSON schema and prompt require the flat
    # occurrence-id contract.  Normalize only that protocol shape.  The
    # occurrence id still has to belong to the frozen input universe, the
    # predicate still has to belong to the frozen Schema, and the evidence
    # still passes the exact scope checks below.  No canonical id, surface,
    # alias, or model confidence is used to create an endpoint.
    legacy_normalized = False
    normalized_proposals: list[Any] = []
    for raw_proposal in proposals_raw:
        if not isinstance(raw_proposal, Mapping):
            normalized_proposals.append(raw_proposal)
            continue
        if (
            raw_proposal.get("subject_occurrence_id")
            or raw_proposal.get("object_occurrence_id")
        ):
            normalized_proposals.append(raw_proposal)
            continue
        subject_endpoint = raw_proposal.get("subject_endpoint")
        object_endpoint = raw_proposal.get("object_endpoint")
        if not isinstance(subject_endpoint, Mapping) or not isinstance(
            object_endpoint, Mapping
        ):
            normalized_proposals.append(raw_proposal)
            continue
        repaired = dict(raw_proposal)
        repaired["subject_occurrence_id"] = str(
            subject_endpoint.get("occurrence_id") or ""
        )
        repaired["object_occurrence_id"] = str(
            object_endpoint.get("occurrence_id") or ""
        )
        if "evidence_spans" not in repaired and "evidence_span" in repaired:
            evidence_span = repaired.get("evidence_span")
            repaired["evidence_spans"] = (
                [evidence_span] if isinstance(evidence_span, Mapping) else evidence_span
            )
        repaired.pop("subject_endpoint", None)
        repaired.pop("object_endpoint", None)
        repaired.pop("evidence_span", None)
        normalized_proposals.append(repaired)
        legacy_normalized = True
    if legacy_normalized:
        repairs.append("legacy_nested_endpoint_proposals_normalized")
    proposals_raw = normalized_proposals

    def reject_proposal(index: int, reason: str) -> None:
        rejected.append({"index": int(index), "reason": str(reason)})
        coverage["complete"] = False

    for proposal_index, raw_proposal in enumerate(proposals_raw):
        if not isinstance(raw_proposal, Mapping):
            reject_proposal(proposal_index, "proposal_not_object")
            continue
        subject = str(raw_proposal.get("subject_occurrence_id") or "")
        object_ = str(raw_proposal.get("object_occurrence_id") or "")
        if subject not in mentions_by_key or object_ not in mentions_by_key:
            reject_proposal(proposal_index, "endpoint_not_in_occurrence_universe")
            continue
        if subject == object_:
            reject_proposal(proposal_index, "endpoints_not_distinct")
            continue
        predicate = str(raw_proposal.get("predicate") or "")
        if predicate not in schema:
            reject_proposal(proposal_index, "predicate_not_in_schema")
            continue
        subject_mention = mentions_by_key[subject]
        object_mention = mentions_by_key[object_]
        endpoint_spans = [
            (
                int(subject_mention.get("mention_start") or 0),
                int(subject_mention.get("mention_end") or 0),
            ),
            (
                int(object_mention.get("mention_start") or 0),
                int(object_mention.get("mention_end") or 0),
            ),
        ]
        spans = _parse_proposer_spans(
            raw_proposal.get("evidence_spans"),
            content=content,
            endpoint_spans=endpoint_spans,
            repairs=repairs,
            proposal_index=proposal_index,
            required_evidence_span=required_evidence_span,
        )
        qualifiers = raw_proposal.get("qualifiers")
        if not isinstance(qualifiers, list):
            qualifiers = []
            repairs.append(f"proposal_{proposal_index}_qualifiers_defaulted")
        qualifiers = [
            dict(item) for item in qualifiers
            if isinstance(item, Mapping)
        ]
        try:
            confidence = float(raw_proposal.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.5
            repairs.append(f"proposal_{proposal_index}_confidence_defaulted")
        if not 0 <= confidence <= 1:
            confidence = max(0.0, min(1.0, confidence))
            repairs.append(f"proposal_{proposal_index}_confidence_clamped")
        material = _canonical_json({
            "context_id": context_id,
            "candidate_source": candidate_source,
            "pass_no": int(pass_no),
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "spans": [[span.start, span.end] for span in spans],
            "qualifiers": qualifiers,
        })
        proposal_id = "relproposal_" + hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()
        if proposal_id in seen:
            continue
        seen.add(proposal_id)
        proposals.append(RelationProposal(
            proposal_id=proposal_id,
            context_id=context_id,
            candidate_source=str(candidate_source),
            pass_no=int(pass_no),
            subject_occurrence_key=subject,
            predicate=predicate,
            object_occurrence_key=object_,
            evidence_spans=spans,
            qualifiers=tuple(dict(item) for item in qualifiers),
            confidence=confidence,
            proposer_policy_version=str(proposer_policy_version),
            proposer_model_revision=str(proposer_model_revision),
        ))
    return RelationProposalBatch(
        proposals=tuple(proposals),
        missing_endpoint_mentions=tuple(dict(item) for item in missing_raw if isinstance(item, Mapping)),
        coverage=dict(coverage),
    )


def proposal_to_hypothesis(
    proposal: RelationProposal,
    opportunity: PairOpportunity,
    logical_schema: Mapping[str, Any],
) -> RelationHypothesis:
    if {
        proposal.subject_occurrence_key,
        proposal.object_occurrence_key,
    } != {
        opportunity.left_occurrence_key,
        opportunity.right_occurrence_key,
    }:
        raise RelationClassificationError("proposal pair does not match opportunity")
    spec = normalize_predicate_schema(logical_schema).get(proposal.predicate)
    if spec is None:
        raise RelationClassificationError("proposal predicate is no longer in schema")
    if _fallback_only_open_relation_policy(spec):
        raise RelationClassificationError(
            "fallback relation carrier is reserved for grounded source frames"
        )
    allowed = {
        str(item.get("name") or "")
        for item in opportunity.allowed_predicate_metadata
    }
    if proposal.predicate not in allowed:
        raise RelationClassificationError(
            "proposal predicate is not allowed by the frozen structure policy"
        )
    evidence_spans = proposal.evidence_spans
    if not _spans_stay_within_required_atomic_scope(
        evidence_spans, opportunity,
    ):
        required = opportunity.provenance.get("required_atomic_scope") or []
        try:
            required_start, required_end = int(required[0]), int(required[1])
        except (IndexError, TypeError, ValueError):
            raise RelationClassificationError(
                "proposal evidence crosses the required atomic scope"
            )
        # The proposer contract deliberately permits a conservative whole-
        # ContextUnit evidence span when a small model cannot calculate exact
        # offsets.  If the planner has already proven that both exact endpoint
        # occurrences share one source-derived atomic scope, narrow that broad
        # span to the scope.  This does not infer a relation: the candidate must
        # still pass deterministic validation and the independent verifier.
        endpoint_spans = (opportunity.left_span, opportunity.right_span)
        if not (
            0 <= required_start < required_end
            and all(
                required_start <= start < end <= required_end
                for start, end in endpoint_spans
            )
            and all(_spans_cover(evidence_spans, span) for span in endpoint_spans)
        ):
            raise RelationClassificationError(
                "proposal evidence crosses the required atomic scope"
            )
        evidence_spans = (
            EvidenceSpan(start=required_start, end=required_end),
        )
    spans_material = json.dumps(
        [[span.start, span.end] for span in evidence_spans],
        separators=(",", ":"),
    )
    hypothesis_key = "relhyp_" + _digest((
        opportunity.pair_identity,
        proposal.subject_occurrence_key,
        proposal.predicate,
        proposal.object_occurrence_key,
        spans_material,
    ))
    return RelationHypothesis(
        hypothesis_id=hypothesis_key,
        pair_identity=opportunity.pair_identity,
        subject_occurrence_key=proposal.subject_occurrence_key,
        predicate=proposal.predicate,
        object_occurrence_key=proposal.object_occurrence_key,
        evidence_spans=evidence_spans,
        classifier_confidence=proposal.confidence,
        classifier_policy_version=proposal.proposer_policy_version,
        classifier_model_revision=proposal.proposer_model_revision,
        classifier_prompt_version=GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
        classifier_schema_version=GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION,
        classifier_parser_version=GROUNDED_RELATION_PROPOSER_PARSER_VERSION,
        predicate_metadata=asdict(spec),
        status="PROPOSED",
    )


def _parse_spans(
    raw: Any,
    *,
    opportunity: PairOpportunity,
    content: str,
) -> tuple[EvidenceSpan, ...]:
    if isinstance(raw, Mapping) and {"start", "end"} <= set(raw):
        raw = [raw]
    elif (
        isinstance(raw, Sequence)
        and not isinstance(raw, (str, bytes, bytearray, Mapping))
        and len(raw) == 2
        and all(
            not isinstance(value, (Sequence, Mapping))
            or isinstance(value, (str, bytes, bytearray))
            for value in raw
        )
    ):
        # A second narrow transport repair covers a model collapsing the
        # outer array and returning one span directly as ``[start, end]``.
        # All immutable bounds, endpoint coverage, and scope gates below are
        # unchanged, so this cannot manufacture evidence.
        raw = [raw]
    if not isinstance(raw, list):
        raise RelationClassificationError("evidence spans must be an array")
    parsed: list[EvidenceSpan] = []
    for value in raw:
        # JSON-schema-capable local models occasionally serialize a requested
        # ``{start,end}`` span as the equivalent two-item array.  This narrow
        # shape repair is semantics-free: bounds, frozen-context containment,
        # exact endpoint coverage and atomic-scope checks still run below.
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, Mapping))
            and len(value) == 2
        ):
            value = {"start": value[0], "end": value[1]}
        if not isinstance(value, Mapping):
            raise RelationClassificationError("evidence span must be an object")
        try:
            start, end = int(value["start"]), int(value["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RelationClassificationError("evidence span is invalid") from exc
        if (
            start < opportunity.context_span[0]
            or end <= start
            or end > opportunity.context_span[1]
            or end > len(content)
        ):
            raise RelationClassificationError("evidence span is outside frozen context")
        parsed.append(EvidenceSpan(start=start, end=end))
    return tuple(parsed)


def _spans_stay_within_required_atomic_scope(
    spans: Sequence[EvidenceSpan],
    opportunity: PairOpportunity,
) -> bool:
    required = opportunity.provenance.get("required_atomic_scope") or []
    if not required:
        return True
    if not isinstance(required, Sequence) or len(required) != 2:
        return False
    try:
        start, end = int(required[0]), int(required[1])
    except (TypeError, ValueError):
        return False
    if all(start <= span.start < span.end <= end for span in spans):
        return True

    # A resolved pronoun/reference keeps two distinct pieces of provenance:
    # the relation assertion in the required atomic scope and the exact
    # antecedent mention that gives the proxy endpoint its canonical identity.
    # Permit the evidence envelope to include only those frozen antecedent
    # spans.  Ordinary cross-sentence evidence still fails closed, as do old
    # or malformed reference records without immutable coordinates.
    references = opportunity.provenance.get(
        "resolved_reference_endpoints"
    ) or []
    if not isinstance(references, list):
        return False
    antecedent_spans: list[tuple[int, int]] = []
    for reference in references:
        if not isinstance(reference, Mapping):
            continue
        raw_reference = reference.get("reference_span") or []
        raw_antecedent = reference.get("antecedent_span") or []
        if not (
            isinstance(raw_reference, Sequence)
            and not isinstance(raw_reference, (str, bytes, bytearray))
            and len(raw_reference) == 2
            and isinstance(raw_antecedent, Sequence)
            and not isinstance(raw_antecedent, (str, bytes, bytearray))
            and len(raw_antecedent) == 2
        ):
            continue
        try:
            reference_start, reference_end = map(int, raw_reference)
            antecedent_start, antecedent_end = map(int, raw_antecedent)
        except (TypeError, ValueError):
            continue
        if not (
            start <= reference_start < reference_end <= end
            and opportunity.context_span[0]
            <= antecedent_start < antecedent_end
            <= opportunity.context_span[1]
        ):
            continue
        antecedent_spans.append((antecedent_start, antecedent_end))
    if not antecedent_spans:
        return False
    envelope_start = min(start, *(value[0] for value in antecedent_spans))
    envelope_end = max(end, *(value[1] for value in antecedent_spans))
    return all(
        envelope_start <= span.start < span.end <= envelope_end
        for span in spans
    )


def _normalize_occurrence(
    value: Any,
    opportunity: PairOpportunity,
) -> str:
    aliases = {
        "LEFT": opportunity.left_occurrence_key,
        "RIGHT": opportunity.right_occurrence_key,
        opportunity.left_occurrence_key: opportunity.left_occurrence_key,
        opportunity.right_occurrence_key: opportunity.right_occurrence_key,
        opportunity.left_mention_id: opportunity.left_occurrence_key,
        opportunity.right_mention_id: opportunity.right_occurrence_key,
    }
    normalized = aliases.get(str(value or ""))
    if normalized is None:
        raise RelationClassificationError("hypothesis occurrence drifted from pair")
    return normalized


def _strong_soft_cue_matches(
    opportunity: PairOpportunity,
    *,
    content: str = "",
) -> tuple[dict[str, str], ...]:
    """Return only schema cues useful for a bounded lexical recovery.

    Soft cues remain non-binding hints in the normal classifier.  They become
    a recovery trigger only after the classifier has returned NO_RELATION, an
    exact pair carries the cue in its frozen context, and the resulting
    hypothesis still passes the ordinary validator and independent verifier.
    Very short copulas and generic auxiliaries are excluded because they are
    common in unrelated prose and do not identify a relation frame.
    """

    weak = {
        "is", "are", "was", "were", "be", "been", "being", "has", "have",
        "had", "do", "does", "did", "shall", "will", "may", "can",
    }
    # A lexical cue is intentionally weaker than source-frame extraction or a
    # model-positive classifier decision.  It may recover a false negative
    # only inside one parser/sentence atomic scope.  Cross-scope relations are
    # still available through those primary channels, but a lone cue between
    # two unrelated table/list lines must not fan out into quadratic fallback
    # hypotheses.
    if opportunity.provenance.get("same_atomic_scope") is not True:
        return ()
    matches: list[dict[str, str]] = []

    def clause_local(cue: str) -> bool:
        """Keep fallback cues attached to the later endpoint's clause.

        This recovery runs only after a model returned ``NO_RELATION``.  A cue
        before a comma can describe an earlier object while the selected later
        endpoint belongs to a new solution, consequence, or commentary clause.
        Primary source-frame extraction still handles those richer sentences;
        the cheap lexical backstop must not bridge the clause boundary.
        """

        if not content:
            return True
        endpoint_spans = sorted(
            (opportunity.left_span, opportunity.right_span),
            key=lambda value: (int(value[0]), int(value[1])),
        )
        start = int(endpoint_spans[0][1])
        end = int(endpoint_spans[1][0])
        if not (0 <= start < end <= len(content)):
            return False
        bridge = content[start:end]
        occurrences = list(re.finditer(re.escape(cue), bridge, re.IGNORECASE))
        return any(
            not re.search(r"[,，;；。！？!?]", bridge[match.end():])
            for match in occurrences
        )

    # Recovery happens only after the primary classifier has already returned
    # NO_RELATION.  A cue elsewhere in a multi-entity sentence/chunk is not a
    # reason to manufacture candidates for every pair in that scope.  The
    # planner persists the source-copied cues occurring strictly between this
    # pair's two endpoint occurrences; use that pair-local contract here as
    # well.  This is especially important when an inventory call fails on a
    # large chunk: the fallback must remain O(local relation frames), not turn
    # one word such as ``supports`` into O(entity pairs) verifier work.
    for item in (
        opportunity.provenance.get("pair_local_soft_cue_matches") or []
    ):
        if not isinstance(item, Mapping):
            continue
        predicate = str(item.get("predicate") or "").strip()
        cue = str(item.get("cue") or "").strip()
        if not predicate or not cue:
            continue
        folded = cue.casefold()
        if folded in weak:
            continue
        if len(cue) < 4 and not any("\u4e00" <= char <= "\u9fff" for char in cue):
            continue
        if not clause_local(cue):
            continue
        matches.append({"predicate": predicate, "cue": cue})
    # Recompute cues from immutable bytes as a compatibility safeguard for an
    # opportunity produced by an older planner.  The search range is still
    # strictly pair-local: only text between the two exact occurrences may
    # trigger this narrow recovery.  The ordinary validator and independent
    # verifier remain authoritative after candidate generation.
    if content:
        endpoint_spans = sorted(
            (opportunity.left_span, opportunity.right_span),
            key=lambda value: (int(value[0]), int(value[1])),
        )
        start = int(endpoint_spans[0][1])
        end = int(endpoint_spans[1][0])
        if 0 <= start < end <= len(content):
            folded_context = content[start:end].casefold()
            for metadata in opportunity.allowed_predicate_metadata:
                if not isinstance(metadata, Mapping):
                    continue
                predicate = str(metadata.get("name") or "").strip()
                if not predicate:
                    continue
                for raw_cue in metadata.get("optional_soft_cues") or []:
                    cue = str(raw_cue or "").strip()
                    if not cue:
                        continue
                    folded = cue.casefold()
                    if folded in weak:
                        continue
                    if len(cue) < 4 and not any(
                        "\u4e00" <= char <= "\u9fff" for char in cue
                    ):
                        continue
                    if folded in folded_context and clause_local(cue):
                        matches.append({"predicate": predicate, "cue": cue})
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in matches:
        unique[(item["predicate"], item["cue"].casefold())] = item
    return tuple(sorted(unique.values(), key=lambda item: (item["predicate"], item["cue"])))


def _structural_parent_child_keys(opportunity: PairOpportunity) -> tuple[str, str]:
    """Return immutable ``(child, parent)`` occurrence keys for structure."""

    provenance = opportunity.provenance
    child = str(provenance.get("structural_child_occurrence_key") or "")
    parent = str(provenance.get("structural_parent_occurrence_key") or "")
    known = {opportunity.left_occurrence_key, opportunity.right_occurrence_key}
    if child in known and parent in known and child != parent:
        return child, parent
    # Section labels are ordered before their child content.  This fallback is
    # intentionally structural only; list nesting is persisted explicitly by
    # the planner whenever the parser supplies parent scopes.
    if opportunity.left_span[0] <= opportunity.right_span[0]:
        return opportunity.right_occurrence_key, opportunity.left_occurrence_key
    return opportunity.left_occurrence_key, opportunity.right_occurrence_key


def structural_projection_hypotheses(
    opportunity: PairOpportunity,
    *,
    schema: Mapping[str, PredicateSpec],
    classifier_policy_version: str,
    classifier_model_revision: str,
) -> tuple[RelationHypothesis, ...]:
    """Create candidates from an explicit parser-confirmed parent edge.

    This is a source-structure event, not a classifier recovery.  Keeping it
    as a separate candidate source means a model cannot erase a confirmed
    hierarchy by choosing an unrelated predicate or by returning a negative
    decision first.  The candidate still has to pass deterministic validation,
    the structural verifier and the normal publication gate.
    """

    provenance = opportunity.provenance
    structural_kind = ""
    if (
        provenance.get("hierarchical_list_parent_child") is True
        and provenance.get("structure_admission") == "hierarchical_list_parent_child"
    ):
        structural_kind = "list_parent_child"
    elif (
        provenance.get("hierarchical_section_label_parent_child") is True
        and provenance.get("structure_admission") == "hierarchical_section_label_parent_child"
    ):
        structural_kind = "section_label_parent_child"
    if not structural_kind:
        return ()

    structural = [
        spec for spec in schema.values()
        if structural_kind in spec.structural_evidence_kinds
        and spec.publish_mode != PUBLISH_DISABLED
        and any(
            str(item.get("name") or "") == spec.name
            for item in opportunity.allowed_predicate_metadata
        )
    ]
    # An ambiguous Schema must not turn syntax into a guessed predicate.
    if len(structural) != 1:
        return ()
    child, parent = _structural_parent_child_keys(opportunity)
    predicate = structural[0].name
    spans = (EvidenceSpan(
        start=int(opportunity.context_span[0]),
        end=int(opportunity.context_span[1]),
    ),)
    hypothesis_id = "relhyp_" + _digest((
        opportunity.pair_identity, child, predicate, parent,
        [[span.start, span.end] for span in spans], "STRUCTURAL_PROJECTION",
    ))
    return (RelationHypothesis(
        hypothesis_id=hypothesis_id,
        pair_identity=opportunity.pair_identity,
        subject_occurrence_key=child,
        predicate=predicate,
        object_occurrence_key=parent,
        evidence_spans=spans,
        classifier_confidence=0.55,
        classifier_policy_version=classifier_policy_version,
        classifier_model_revision=classifier_model_revision,
        classifier_prompt_version=GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
        classifier_schema_version=GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
        classifier_parser_version=GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
        predicate_metadata={
            **asdict(structural[0]),
            "candidate_source": "STRUCTURAL_PROJECTION",
            "structural_event_projection": True,
        },
        status="STRUCTURAL_PROJECTION",
    ),)


def _recovery_hypotheses_after_no_relation(
    opportunity: PairOpportunity,
    *,
    content: str,
    schema: Mapping[str, PredicateSpec],
    classifier_policy_version: str,
    classifier_model_revision: str,
) -> tuple[RelationHypothesis, ...]:
    """Recover narrowly scoped candidates after a negative primary decision.

    There are two independent, schema-driven triggers: parser-confirmed
    parent/child structure and a strong configured cue.  Neither trigger
    publishes a fact or bypasses role/type/verifier checks.  The function
    deliberately requires one unambiguous predicate for each trigger; if a
    schema offers competing semantics, the model must resolve the competition
    in a normal or secondary pass.
    """

    structural_projections = structural_projection_hypotheses(
        opportunity,
        schema=schema,
        classifier_policy_version=classifier_policy_version,
        classifier_model_revision=classifier_model_revision,
    )
    if structural_projections:
        return structural_projections

    candidates: list[tuple[str, str, str, str]] = []

    if not candidates:
        cue_matches = _strong_soft_cue_matches(opportunity, content=content)
        names = {item["predicate"] for item in cue_matches}
        cue_specs = [
            schema[name] for name in sorted(names)
            if name in schema
            and schema[name].publish_mode != PUBLISH_DISABLED
            and any(
                str(item.get("name") or "") == name
                for item in opportunity.allowed_predicate_metadata
            )
        ]
        if len(cue_specs) == 1:
            spec = cue_specs[0]
            left_to_right = opportunity.left_occurrence_key
            right_to_left = opportunity.right_occurrence_key
            metadata = next(
                (
                    item for item in opportunity.allowed_predicate_metadata
                    if str(item.get("name") or "") == spec.name
                ),
                {},
            )
            # A schema with unresolved wildcard orientation can safely emit
            # both ordered candidates: the independent verifier will retain
            # only the source-grounded direction.  This is bounded to one
            # predicate and two orientations, never an all-pairs expansion.
            if metadata.get("left_to_right") is True and metadata.get("right_to_left") is not True:
                candidates.append((left_to_right, spec.name, right_to_left, "LEXICAL_CUE_RECOVERY"))
            elif metadata.get("right_to_left") is True and metadata.get("left_to_right") is not True:
                candidates.append((right_to_left, spec.name, left_to_right, "LEXICAL_CUE_RECOVERY"))
            else:
                candidates.extend((
                    (left_to_right, spec.name, right_to_left, "LEXICAL_CUE_RECOVERY"),
                    (right_to_left, spec.name, left_to_right, "LEXICAL_CUE_RECOVERY"),
                ))

    projected: list[RelationHypothesis] = []
    for subject, predicate, object_, source in candidates:
        spec = schema.get(predicate)
        if spec is None:
            continue
        spans = (EvidenceSpan(
            start=int(opportunity.context_span[0]),
            end=int(opportunity.context_span[1]),
        ),)
        hypothesis_id = "relhyp_" + _digest((
            opportunity.pair_identity, subject, predicate, object_,
            [[span.start, span.end] for span in spans], source,
        ))
        projected.append(RelationHypothesis(
            hypothesis_id=hypothesis_id,
            pair_identity=opportunity.pair_identity,
            subject_occurrence_key=subject,
            predicate=predicate,
            object_occurrence_key=object_,
            evidence_spans=spans,
            classifier_confidence=0.55 if source == "STRUCTURAL_PROJECTION" else 0.50,
            classifier_policy_version=classifier_policy_version,
            classifier_model_revision=classifier_model_revision,
            classifier_prompt_version=GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
            classifier_schema_version=GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
            classifier_parser_version=GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
            predicate_metadata={
                **asdict(spec),
                "candidate_source": source,
                "recovered_after_primary_no_relation": True,
            },
            status=source,
        ))
    return tuple(projected)


def parse_classifier_response(
    opportunities: Sequence[PairOpportunity],
    raw: Any,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    classifier_policy_version: str = "shadow-relation-classifier-v2",
    classifier_model_revision: str = "unspecified",
    allow_partial: bool = False,
) -> tuple[PairClassificationDecision, ...]:
    """Parse classifier decisions, failing closed per pair.

    Exact mode requires one valid decision for every requested pair.  Partial
    mode is used only after exact parsing fails: it returns independently valid
    members whose IDs are known and unique, while silently withholding every
    omitted, duplicated, unknown, or malformed member for an isolated retry.
    """

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError("classifier returned invalid JSON") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("decisions")
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise RelationClassificationError("classifier decisions must be an array")
    expected = {item.pair_identity: item for item in opportunities}
    if allow_partial:
        returned_ids = [str(item.get("pair_id") or "") for item in raw]
        duplicate_ids = {
            pair_id for pair_id in returned_ids
            if pair_id and returned_ids.count(pair_id) > 1
        }
        recovered: list[PairClassificationDecision] = []
        for value in raw:
            pair_id = str(value.get("pair_id") or "")
            if pair_id not in expected or pair_id in duplicate_ids:
                continue
            try:
                parsed = parse_classifier_response(
                    [expected[pair_id]],
                    {"decisions": [dict(value)]},
                    content=content,
                    logical_schema=logical_schema,
                    classifier_policy_version=classifier_policy_version,
                    classifier_model_revision=classifier_model_revision,
                    allow_partial=False,
                )
            except RelationClassificationError:
                continue
            recovered.extend(parsed)
        return tuple(recovered)
    if len(expected) == 1 and len(raw) == 1:
        only_pair_id = next(iter(expected))
        only_value = raw[0]
        returned_pair_id = str(only_value.get("pair_id") or "")
        returned_decision = str(only_value.get("decision") or "")
        returned_hypotheses = only_value.get("hypotheses") or []
        if (
            returned_pair_id
            and returned_pair_id not in expected
            and _looks_like_abbreviated_pair_id(returned_pair_id, only_pair_id)
            and returned_decision in {NO_RELATION, OTHER_UNSUPPORTED}
            and returned_hypotheses == []
        ):
            raw = [{**dict(only_value), "pair_id": only_pair_id}]
    ids = [str(item.get("pair_id") or "") for item in raw]
    duplicates = sorted(
        pair_id for pair_id, count in {
            value: ids.count(value) for value in ids if value
        }.items() if count > 1
    )
    if duplicates:
        raise RelationClassificationError(f"duplicate pair ids: {duplicates}")
    unknown = sorted(set(ids) - set(expected))
    if unknown:
        raise RelationClassificationError(f"unknown pair ids: {unknown}")
    missing = sorted(set(expected) - set(ids))
    if missing:
        raise RelationClassificationError(f"missing pair ids: {missing}")

    schema = normalize_predicate_schema(logical_schema)
    decisions: list[PairClassificationDecision] = []
    for value in raw:
        pair = expected[str(value["pair_id"])]
        decision = str(value.get("decision") or "")
        raw_hypotheses = value.get("hypotheses")
        if not isinstance(raw_hypotheses, list):
            raise RelationClassificationError("hypotheses must be an array")
        if decision not in {NO_RELATION, OTHER_UNSUPPORTED, HYPOTHESES}:
            raise RelationClassificationError(f"unknown classifier decision: {decision}")
        if decision in {NO_RELATION, OTHER_UNSUPPORTED} and raw_hypotheses:
            raise RelationClassificationError("negative/schema-gap decision cannot contain hypotheses")
        if decision == HYPOTHESES and not raw_hypotheses:
            raise RelationClassificationError("HYPOTHESES decision requires hypotheses")
        raw_decision_spans = value.get("evidence_spans")
        if decision in {NO_RELATION, OTHER_UNSUPPORTED} and not raw_decision_spans:
            raw_decision_spans = [{
                "start": pair.context_span[0],
                "end": pair.context_span[1],
            }]
        decision_spans = _parse_spans(
            raw_decision_spans or [],
            opportunity=pair,
            content=content,
        )
        reason_code = str(value.get("reason_code") or "")
        if decision == NO_RELATION and not reason_code.strip():
            reason_code = "NO_RELATION_UNSPECIFIED"
        if decision == OTHER_UNSUPPORTED:
            reason_code = "SCHEMA_GAP"
        if decision == NO_RELATION and not decision_spans:
            raise RelationClassificationError(
                "NO_RELATION requires evidence spans covering the pair context"
            )
        if decision == OTHER_UNSUPPORTED and (
            not decision_spans or reason_code != "SCHEMA_GAP"
        ):
            raise RelationClassificationError(
                "OTHER_UNSUPPORTED requires evidence and SCHEMA_GAP"
            )
        parsed_hypotheses: list[RelationHypothesis] = []
        pair_metadata = {
            str(item.get("name") or ""): item
            for item in pair.allowed_predicate_metadata
        }
        for hypothesis in raw_hypotheses:
            if not isinstance(hypothesis, Mapping):
                raise RelationClassificationError("hypothesis must be an object")
            subject = _normalize_occurrence(
                hypothesis.get("subject_occurrence_id"), pair,
            )
            object_ = _normalize_occurrence(
                hypothesis.get("object_occurrence_id"), pair,
            )
            if subject == object_:
                raise RelationClassificationError("hypothesis endpoints must be distinct")
            predicate = str(hypothesis.get("predicate") or "")
            spec = schema.get(predicate)
            metadata = pair_metadata.get(predicate)
            if spec is None or metadata is None:
                raise RelationClassificationError(f"predicate is not allowed: {predicate}")
            if _fallback_only_open_relation_policy(spec):
                raise RelationClassificationError(
                    "fallback relation carrier is reserved for grounded source frames"
                )
            left_to_right = (
                subject == pair.left_occurrence_key
                and object_ == pair.right_occurrence_key
            )
            legacy_direction = hypothesis.get("direction")
            if legacy_direction is not None:
                expected_direction = (
                    "symmetric" if spec.symmetric
                    else "as_is" if left_to_right else "reversed"
                )
                if str(legacy_direction) != expected_direction:
                    raise RelationClassificationError(
                        "legacy direction conflicts with ordered endpoint roles"
                    )
            spans = _parse_spans(
                hypothesis.get("evidence_spans"),
                opportunity=pair,
                content=content,
            )
            if not spans:
                raise RelationClassificationError("positive hypothesis requires evidence")
            if not _spans_stay_within_required_atomic_scope(spans, pair):
                raise RelationClassificationError(
                    "positive hypothesis evidence crosses the required atomic scope"
                )
            try:
                confidence = float(hypothesis.get("classifier_confidence"))
            except (TypeError, ValueError) as exc:
                raise RelationClassificationError("classifier confidence is invalid") from exc
            if not 0 <= confidence <= 1:
                raise RelationClassificationError("classifier confidence is outside [0,1]")
            hypothesis_key = "relhyp_" + _digest((
                pair.pair_identity,
                subject,
                predicate,
                object_,
                json.dumps(
                    [[span.start, span.end] for span in spans],
                    separators=(",", ":"),
                ),
            ))
            parsed_hypotheses.append(RelationHypothesis(
                hypothesis_id=hypothesis_key,
                pair_identity=pair.pair_identity,
                subject_occurrence_key=subject,
                predicate=predicate,
                object_occurrence_key=object_,
                evidence_spans=spans,
                classifier_confidence=confidence,
                classifier_policy_version=classifier_policy_version,
                classifier_model_revision=classifier_model_revision,
                classifier_prompt_version=GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
                classifier_schema_version=GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
                classifier_parser_version=GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
                predicate_metadata=asdict(spec),
            ))
        recovery_hypotheses = ()
        decision_value = decision
        if decision == NO_RELATION:
            recovery_hypotheses = _recovery_hypotheses_after_no_relation(
                pair,
                content=content,
                schema=schema,
                classifier_policy_version=classifier_policy_version,
                classifier_model_revision=classifier_model_revision,
            )
            if recovery_hypotheses:
                parsed_hypotheses.extend(recovery_hypotheses)
                decision_value = HYPOTHESES
                reason_code = (
                    f"{reason_code};RECOVERY_PROJECTION"
                )
        decisions.append(PairClassificationDecision(
            pair_identity=pair.pair_identity,
            decision=decision_value,
            hypotheses=tuple(parsed_hypotheses),
            evidence_spans=decision_spans,
            reason_code=reason_code,
        ))
    return tuple(decisions)


def _looks_like_abbreviated_pair_id(returned_pair_id: str, expected_pair_id: str) -> bool:
    returned = str(returned_pair_id)
    expected = str(expected_pair_id)
    if not returned.startswith("relpair_") or not expected.startswith("relpair_"):
        return False
    if len(returned) >= len(expected):
        return False
    prefix_len = min(32, len(returned))
    suffix_len = min(6, len(returned))
    return expected.startswith(returned[:prefix_len]) and expected.endswith(
        returned[-suffix_len:],
    )


def _occurrence_details(
    opportunity: PairOpportunity,
    occurrence_key: str,
    content: str,
) -> dict[str, Any]:
    if occurrence_key == opportunity.left_occurrence_key:
        span = opportunity.left_span
        return {
            "occurrence_key": occurrence_key,
            "span": span,
            "surface": content[span[0]:span[1]],
            "type": opportunity.left_type,
            "canonical_id": opportunity.left_canonical_id,
        }
    if occurrence_key == opportunity.right_occurrence_key:
        span = opportunity.right_span
        return {
            "occurrence_key": occurrence_key,
            "span": span,
            "surface": content[span[0]:span[1]],
            "type": opportunity.right_type,
            "canonical_id": opportunity.right_canonical_id,
        }
    raise ValueError("occurrence does not belong to pair")


def _spans_cover(
    spans: Sequence[EvidenceSpan],
    required_span: tuple[int, int],
) -> bool:
    return any(
        span.start <= required_span[0] and span.end >= required_span[1]
        for span in spans
    )


def validate_hypothesis(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> HypothesisValidation:
    """Apply deterministic identity, span, type and policy checks before model I/O."""

    schema = normalize_predicate_schema(logical_schema)
    spec = schema.get(hypothesis.predicate)
    empty_metadata: dict[str, Any] = {}

    def result(
        status: str,
        reason: str,
        *,
        type_orientation: str = "forward",
    ) -> HypothesisValidation:
        return HypothesisValidation(
            hypothesis_id=hypothesis.hypothesis_id,
            status=status,
            reason_code=reason,
            predicate_metadata=asdict(spec) if spec else empty_metadata,
            type_orientation=type_orientation,
        )

    if hypothesis.pair_identity != opportunity.pair_identity:
        return result(REJECTED, "PAIR_IDENTITY_MISMATCH")
    endpoint_keys = {
        hypothesis.subject_occurrence_key,
        hypothesis.object_occurrence_key,
    }
    if endpoint_keys != {
        opportunity.left_occurrence_key,
        opportunity.right_occurrence_key,
    }:
        return result(REJECTED, "OCCURRENCE_IDENTITY_MISMATCH")
    if not hypothesis.evidence_spans or any(
        span.start < opportunity.context_span[0]
        or span.end <= span.start
        or span.end > opportunity.context_span[1]
        or span.end > len(content)
        for span in hypothesis.evidence_spans
    ):
        return result(REJECTED, "EVIDENCE_BOUNDARY_INVALID")
    subject = _occurrence_details(
        opportunity, hypothesis.subject_occurrence_key, content,
    )
    object_ = _occurrence_details(
        opportunity, hypothesis.object_occurrence_key, content,
    )
    if not (
        _spans_cover(hypothesis.evidence_spans, subject["span"])
        and _spans_cover(hypothesis.evidence_spans, object_["span"])
    ):
        return result(REJECTED, "EVIDENCE_ENDPOINT_MISMATCH")
    if spec is None:
        return result(REJECTED, "UNKNOWN_PREDICATE")
    allowed_predicates = {
        str(item.get("name") or "")
        for item in opportunity.allowed_predicate_metadata
    }
    if hypothesis.predicate not in allowed_predicates:
        return result(REJECTED, "PREDICATE_NOT_ALLOWED_BY_STRUCTURE")
    structural_kind = (
        "list_parent_child"
        if _has_hierarchical_list_parent_child_evidence(opportunity)
        else "section_label_parent_child"
        if _has_hierarchical_section_label_parent_child_evidence(opportunity)
        else ""
    )
    if structural_kind and structural_kind in spec.structural_evidence_kinds:
        # The parser contract for both hierarchy kinds is immutable
        # ``structural_child -> structural_parent``.  A classifier or source
        # binding may still emit the unordered pair in reverse, and a small
        # verifier can then mistake the visible parent/child layout for support
        # of that reversed Schema claim.  Reject the impossible orientation
        # before model I/O so two apparently SUPPORTED inverse facts cannot
        # cancel the valid edge during normalization.
        structural_child, structural_parent = _structural_parent_child_keys(
            opportunity
        )
        if (
            hypothesis.subject_occurrence_key != structural_child
            or hypothesis.object_occurrence_key != structural_parent
        ):
            return result(REJECTED, "STRUCTURAL_ROLE_ORIENTATION_MISMATCH")
    open_policy = spec.open_relation_policy
    if bool(open_policy.get("enabled")):
        source_metadata = (
            hypothesis.predicate_metadata
            if isinstance(hypothesis.predicate_metadata, Mapping) else {}
        )
        candidate_source = str(
            source_metadata.get("candidate_source") or ""
        ).strip().upper()
        allowed_sources = {
            str(value).strip().upper()
            for value in open_policy.get("allowed_candidate_sources") or ()
        }
        if candidate_source not in allowed_sources:
            return result(REJECTED, "OPEN_RELATION_SOURCE_NOT_ALLOWED")
        source_binding_source = str(
            source_metadata.get("source_binding_source") or ""
        ).strip().upper()
        if bool(open_policy.get("fallback_only")) and (
            source_binding_source != "OPEN_EXPRESSION_CARRIER_FALLBACK"
        ):
            # Parser-owned table roles can use the same fallback carrier when
            # an older active Schema has no exact column predicate.  This is
            # not a generic alternate binding path: the full immutable table
            # contract (kind, roles, label, expression, endpoints and bytes)
            # must authorize it before the ordinary validator can accept it.
            if not (
                source_binding_source
                == "DETERMINISTIC_SOURCE_STRUCTURE_OPEN_CARRIER"
                and source_structure_binding_contract_authorized(
                    hypothesis,
                    opportunity,
                    content=content,
                    logical_schema=logical_schema,
                )
            ):
                return result(
                    REJECTED, "OPEN_RELATION_FALLBACK_BINDING_REQUIRED"
                )
        if bool(open_policy.get("require_exact_source_expression")):
            if not source_relation_display_expression(
                source_metadata.get("source_frame_relation")
            ):
                return result(REJECTED, "OPEN_RELATION_EXPRESSION_INVALID")
            if str(
                source_metadata.get("source_frame_assertion_status") or ""
            ).strip().upper() != "ASSERTED":
                return result(REJECTED, "OPEN_RELATION_NOT_ASSERTED")
            if bool(source_metadata.get("source_frame_explicit_negation")):
                return result(REJECTED, "OPEN_RELATION_EXPLICITLY_NEGATED")
            raw_frame_span = source_metadata.get(
                "source_frame_evidence_span"
            )
            if not (
                isinstance(raw_frame_span, (list, tuple))
                and len(raw_frame_span) == 2
                and all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in raw_frame_span
                )
            ):
                return result(REJECTED, "OPEN_RELATION_EVIDENCE_SPAN_INVALID")
            frame_start, frame_end = map(int, raw_frame_span)
            if not (
                opportunity.context_span[0] <= frame_start < frame_end
                <= opportunity.context_span[1]
                and frame_end <= len(content)
                and any(
                    span.start == frame_start and span.end == frame_end
                    for span in hypothesis.evidence_spans
                )
                and frame_start <= subject["span"][0]
                < subject["span"][1] <= frame_end
                and frame_start <= object_["span"][0]
                < object_["span"][1] <= frame_end
            ):
                return result(REJECTED, "OPEN_RELATION_EVIDENCE_SPAN_MISMATCH")
            frame_text = str(
                source_metadata.get("source_frame_evidence_text") or ""
            )
            if not frame_text or frame_text != content[frame_start:frame_end]:
                return result(REJECTED, "OPEN_RELATION_EVIDENCE_TEXT_MISMATCH")
            if (
                not str(source_metadata.get("source_structure_kind") or "")
                and not _source_relation_expression_is_grounded(
                    frame_text,
                    source_metadata.get("source_frame_relation"),
                )
            ):
                return result(
                    REJECTED,
                    "OPEN_RELATION_EXPRESSION_NOT_SOURCE_GROUNDED",
                )
    if not _spans_stay_within_required_atomic_scope(
        hypothesis.evidence_spans, opportunity,
    ) and not source_structure_binding_contract_authorized(
        hypothesis,
        opportunity,
        content=content,
        logical_schema=logical_schema,
    ):
        return result(REJECTED, "EVIDENCE_CROSSES_ATOMIC_SCOPE")
    if (
        subject["canonical_id"]
        and subject["canonical_id"] == object_["canonical_id"]
        and not spec.reflexive
    ):
        return result(REJECTED, "ILLEGAL_SELF_LOOP")
    if spec.inverse_predicate and spec.inverse_predicate not in schema:
        return result(REJECTED, "INVERSE_METADATA_INVALID")
    subject_match = _type_match(
        str(subject["type"]), spec.domain_types,
    )
    object_match = _type_match(
        str(object_["type"]), spec.range_types,
    )
    if subject_match is False or object_match is False:
        reverse_subject_match = _type_match(
            str(object_["type"]), spec.domain_types,
        )
        reverse_object_match = _type_match(
            str(subject["type"]), spec.range_types,
        )
        if reverse_subject_match is True and reverse_object_match is True:
            if spec.publish_mode == PUBLISH_REVIEW_ONLY:
                return result(
                    REVIEW_ONLY,
                    "PUBLISH_MODE_REVIEW_ONLY",
                    type_orientation="reverse",
                )
            if spec.publish_mode == PUBLISH_DISABLED:
                return result(
                    REVIEW_ONLY,
                    "PUBLISH_MODE_DISABLED",
                    type_orientation="reverse",
                )
            return result(
                PASS,
                "REVERSE_TYPE_COMPATIBLE_REQUIRES_ROLE_VERIFICATION",
                type_orientation="reverse",
            )
        if reverse_subject_match is None or reverse_object_match is None:
            if spec.publish_mode == PUBLISH_REVIEW_ONLY:
                return result(
                    REVIEW_ONLY,
                    "PUBLISH_MODE_REVIEW_ONLY",
                    type_orientation="unresolved",
                )
            if spec.publish_mode == PUBLISH_DISABLED:
                return result(
                    REVIEW_ONLY,
                    "PUBLISH_MODE_DISABLED",
                    type_orientation="unresolved",
                )
            return result(
                PASS,
                "TYPE_ORIENTATION_UNRESOLVED_REQUIRES_VERIFICATION",
                type_orientation="unresolved",
            )
        return result(REJECTED, "TYPE_INCOMPATIBLE")
    if subject_match is None or object_match is None:
        if spec.publish_mode == PUBLISH_REVIEW_ONLY:
            return result(REVIEW_ONLY, "PUBLISH_MODE_REVIEW_ONLY")
        if spec.publish_mode == PUBLISH_DISABLED:
            return result(REVIEW_ONLY, "PUBLISH_MODE_DISABLED")
        endpoint_candidates = opportunity.provenance.get(
            "endpoint_candidates"
        ) or {}
        if bool(
            endpoint_candidates.get("left")
            or endpoint_candidates.get("right")
        ):
            return result(
                PASS,
                "TYPE_UNRESOLVED_ENDPOINT_REQUIRES_VERIFICATION",
            )
        return result(PASS, "TYPE_UNRESOLVED_REQUIRES_VERIFICATION")
    if spec.publish_mode == PUBLISH_REVIEW_ONLY:
        return result(REVIEW_ONLY, "PUBLISH_MODE_REVIEW_ONLY")
    if spec.publish_mode == PUBLISH_DISABLED:
        return result(REVIEW_ONLY, "PUBLISH_MODE_DISABLED")
    return result(PASS, "DETERMINISTIC_VALIDATION_PASSED")


def verifier_role_probe_prompt(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
) -> str:
    """Build a predicate-blind endpoint-role probe.

    Omitting the candidate predicate prevents the first semantic pass from
    anchoring its source reading on the classifier's proposed relation.
    """

    subject = _occurrence_details(
        opportunity, hypothesis.subject_occurrence_key, content,
    )
    object_ = _occurrence_details(
        opportunity, hypothesis.object_occurrence_key, content,
    )
    context_window = _pair_model_context(
        opportunity, content=content,
    )
    context_text = str(context_window["text"])
    uses_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", context_text))
    instruction = (
        "只依据冻结原文，先不要猜测任何目标谓词。检查两个精确端点是否各自构成完整的"
        "命题论元：若端点只是更长专名、事件、状态、条件或带限定短语的一部分，则不完整；"
        "不得丢掉事件触发词、状态变化、否定或情态而把内部名词提升成论元。然后判断原文"
        "是否直接表达了这两个完整端点之间的关系。仅共现、版面相邻、同属一个列举，或"
        "分别与第三端点发生关系，都不算直接关系。若一个语法谓词支配多个并列成员，则"
        "谓词与每个成员可分别构成关系；支配项或谓词可以出现在整组并列项之前或之后，"
        "不能因其不紧邻某个成员就漏掉该成员，但并列成员彼此不因此构成关系。普通的施事—动作—"
        "受事、介词角色和被动结构属于直接关系。分别用简短中文描述端点 A、B 在原句中的"
        "实际语义角色，并用一句话复述原文直接表达的有向关系；角色不能只写实体类别，"
        "不能使用‘主语/宾语、A/B、左/右、它/对方’等指代。嵌套从句和修饰语中出现的第三"
        "实体不能替代中心实体。若没有直接关系，两个角色都写‘无直接角色’，并简述原因。"
        "另外返回每个端点在当前关系中充当完整论元时对应的最小原文绝对半开区间；若当前"
        "端点已经完整，该区间必须与端点区间完全相同，若它只是更长论元的一部分，则区间"
        "必须覆盖完整论元。只返回 endpoint_a_complete_argument、"
        "endpoint_b_complete_argument、endpoint_a_argument_start、"
        "endpoint_a_argument_end、endpoint_b_argument_start、endpoint_b_argument_end、"
        "direct_relation_present、subject_role、object_role、expressed_relation。"
        "context 可能是围绕端点的有界窗口；省略的原文不可用于判断直接关系，所有返回坐标"
        "都是完整来源中的绝对坐标。"
        if uses_cjk else
        "Use only the frozen source context and do not guess a target predicate. First "
        "check whether each exact span is a complete proposition argument. A substring "
        "of a longer name, event, state, condition, or qualified phrase is incomplete; "
        "never drop a trigger, state change, negation, or modality to promote an inner "
        "noun. Then decide whether the source directly relates the two complete spans. "
        "Co-occurrence, layout adjacency, peer membership in one coordinated list, and "
        "separate relations through a third endpoint do not count. A shared grammatical "
        "predicate may relate its governor to each coordinated member, and the governor "
        "or predicate can occur before or after the whole list rather than next to every "
        "member; it does not by itself relate peer members to one another. Describe each endpoint's actual "
        "semantic role as a short standalone role phrase and paraphrase only the directed "
        "source relation. Roles must not "
        "be entity categories, endpoint labels, or pronouns. Resolve modifiers and nested "
        "clauses without substituting a third entity. If no direct relation exists, use "
        "'no direct role' for both roles and state why. Also return the absolute half-open "
        "span of the smallest complete source argument corresponding to each endpoint. "
        "The displayed context may be a bounded endpoint-centered window; omitted source "
        "ranges are unavailable and cannot support a direct relation. All returned offsets "
        "are absolute source offsets. An already complete endpoint must return its exact endpoint span; an incomplete "
        "substring must return the containing complete argument span. Return only the ten "
        "required fields."
    )
    return json.dumps(
        {
            "instruction": instruction,
            "contract_version": GROUNDED_RELATION_ROLE_PROBE_PROMPT_VERSION,
            "hypothesis_id": hypothesis.hypothesis_id,
            "context": {
                "start": opportunity.context_span[0],
                "end": opportunity.context_span[1],
                "text": context_text,
                "visible_segments": context_window["visible_segments"],
                "truncated": bool(context_window["truncated"]),
            },
            "endpoint_a": subject,
            "endpoint_b": object_,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def verifier_role_probe_response_schema() -> dict[str, Any]:
    offset = {"type": "integer", "minimum": 0}
    return {
        "$id": GROUNDED_RELATION_ROLE_PROBE_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "endpoint_a_complete_argument": {"type": "boolean"},
            "endpoint_b_complete_argument": {"type": "boolean"},
            "endpoint_a_argument_start": offset,
            "endpoint_a_argument_end": offset,
            "endpoint_b_argument_start": offset,
            "endpoint_b_argument_end": offset,
            "direct_relation_present": {"type": "boolean"},
            "subject_role": {
                "type": "string", "minLength": 1, "maxLength": 160,
            },
            "object_role": {
                "type": "string", "minLength": 1, "maxLength": 160,
            },
            "expressed_relation": {
                "type": "string", "minLength": 1, "maxLength": 320,
            },
        },
        "required": [
            "endpoint_a_complete_argument",
            "endpoint_b_complete_argument",
            "endpoint_a_argument_start", "endpoint_a_argument_end",
            "endpoint_b_argument_start", "endpoint_b_argument_end",
            "direct_relation_present",
            "subject_role", "object_role", "expressed_relation",
        ],
    }


def parse_verifier_role_probe(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "verifier role probe returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "verifier role probe response must be an object"
        )
    subject_role = str(
        raw.get("subject_role")
        or raw.get("endpoint_a_role")
        or raw.get("a_role")
        or raw.get("source_role")
        or ""
    ).strip()
    object_role = str(
        raw.get("object_role")
        or raw.get("endpoint_b_role")
        or raw.get("b_role")
        or raw.get("target_role")
        or ""
    ).strip()
    if not subject_role or not object_role:
        raise RelationClassificationError(
            "verifier role probe requires both endpoint roles"
        )
    expressed_relation = str(
        raw.get("expressed_relation")
        or raw.get("direct_relation")
        or raw.get("relation")
        or ""
    ).strip()
    # Frozen ledgers and scripted fixtures created before schema v3 do not
    # contain the predicate-blind relation paraphrase.  Preserve replay
    # compatibility without inventing semantics: the alignment parser will
    # use the legacy role-only contract when this marker is present.
    if not expressed_relation:
        expressed_relation = "LEGACY_ROLE_ONLY_AUDIT"
    def first_boolean(*names: str) -> bool | None:
        for name in names:
            value = raw.get(name)
            if isinstance(value, bool):
                return value
        return None

    def first_integer(*names: str) -> int | None:
        for name in names:
            value = raw.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
        return None

    def span_integer(span_name: str, index: int, *names: str) -> int | None:
        value = first_integer(*names)
        if value is not None:
            return value
        span = raw.get(span_name)
        if isinstance(span, (list, tuple)) and len(span) == 2:
            value = span[index]
            if isinstance(value, int) and not isinstance(value, bool):
                return int(value)
        return None

    def direct_relation_value() -> bool | None:
        value = first_boolean(
            "direct_relation_present", "direct_relation_exists",
            "relation_present_between_exact_endpoints",
        )
        if value is not None:
            return value
        relation = raw.get("direct_relation")
        if isinstance(relation, str) and relation.strip():
            return relation.strip().upper() not in {
                "NO_DIRECT_RELATION", "NONE", "FALSE", "NO",
            }
        return None

    return {
        "endpoint_a_complete_argument": first_boolean(
            "endpoint_a_complete_argument", "a_complete_argument",
            "endpoint_a_complete", "a_complete", "complete_argument_a",
            "a_is_complete_argument_mention",
        ),
        "endpoint_b_complete_argument": first_boolean(
            "endpoint_b_complete_argument", "b_complete_argument",
            "endpoint_b_complete", "b_complete", "complete_argument_b",
            "b_is_complete_argument_mention",
        ),
        "endpoint_a_argument_start": span_integer(
            "endpoint_a_span", 0,
            "endpoint_a_argument_start", "a_argument_start", "endpoint_a_start",
        ),
        "endpoint_a_argument_end": span_integer(
            "endpoint_a_span", 1,
            "endpoint_a_argument_end", "a_argument_end", "endpoint_a_end",
        ),
        "endpoint_b_argument_start": span_integer(
            "endpoint_b_span", 0,
            "endpoint_b_argument_start", "b_argument_start", "endpoint_b_start",
        ),
        "endpoint_b_argument_end": span_integer(
            "endpoint_b_span", 1,
            "endpoint_b_argument_end", "b_argument_end", "endpoint_b_end",
        ),
        "direct_relation_present": direct_relation_value(),
        "subject_role": subject_role[:500],
        "object_role": object_role[:500],
        "expressed_relation": expressed_relation[:1000],
    }


def _open_fact_occurrence_catalog(
    occurrences: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Build stable short model IDs for immutable relation occurrences."""

    ordered = sorted(
        (
            item for item in occurrences
            if str(item.get("occurrence_key") or "")
            and str(item.get("surface") or "")
        ),
        key=lambda item: (
            int(item.get("mention_start") or 0),
            int(item.get("mention_end") or 0),
            str(item.get("occurrence_key") or ""),
        ),
    )
    canonical_aliases: dict[str, str] = {}
    catalog: list[dict[str, Any]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, occurrence in enumerate(ordered):
        model_id = f"M{index:04d}"
        canonical = str(occurrence.get("canonical_id") or "")
        same_entity_group = ""
        if canonical:
            same_entity_group = canonical_aliases.setdefault(
                canonical, f"C{len(canonical_aliases):04d}",
            )
        catalog.append({
            "id": model_id,
            "surface": str(occurrence.get("surface") or ""),
            "start": int(occurrence.get("mention_start") or 0),
            "end": int(occurrence.get("mention_end") or 0),
            "type": str(occurrence.get("type") or "GenericEntity"),
            "same_entity_group": same_entity_group,
            "scope_id": str(occurrence.get("scope_id") or ""),
            "scope_kind": str(occurrence.get("scope_kind") or ""),
            "table_id": str(occurrence.get("table_id") or ""),
            "row_id": str(occurrence.get("row_id") or ""),
            "cell_id": str(occurrence.get("cell_id") or ""),
            "list_group_id": str(occurrence.get("list_group_id") or ""),
            "list_depth": int(occurrence.get("list_depth") or 0),
            "is_resolved_reference": bool(
                occurrence.get("relation_reference_proxy")
            ),
        })
        by_id[model_id] = occurrence
    return catalog, by_id


_RELATION_REFERENCE_PATTERN = re.compile(
    r"(?<![\w])(?:"
    r"the\s+former|the\s+latter|he|him|his|himself|she|her|hers|herself|"
    r"it|its|itself|they|them|their|theirs|themselves|this|that|these|"
    r"those|former|latter|who|whom|whose|which|"
    r"er|ihn|ihm|sein|seine|sie|ihr|ihre|es|dessen|deren|dieser|diese|"
    r"dieses|jener|jene|jenes|"
    r"il|elle|ils|elles|lui|eux|son|sa|ses|leur|leurs|celui|celle|ceux|"
    r"celles|ceci|cela|"
    r"él|ella|ellos|ellas|lo|la|los|las|le|les|su|sus|este|esta|estos|"
    r"estas|aquel|aquella|aquellos|aquellas|"
    r"ele|ela|eles|elas|dele|dela|deles|delas|seu|sua|seus|suas"
    r")(?![\w])|"
    r"(?:他们|她们|它们|这些|那些|上述|前述|前者|后者|本身|其中|对方|"
    r"他|她|它|其|该|此)|"
    r"(?:それ|これ|あれ|彼女|彼|同氏|同社)|"
    r"(?:그녀|그것|해당)",
    re.IGNORECASE,
)


def _reference_is_lexical_subtoken(
    content: str, start: int, end: int, surface: str,
) -> bool:
    """Reject closed-class reference glyphs embedded in another lexical item."""

    previous = re.sub(r"\s+", "", content[max(0, start - 4):start])[-1:]
    following = re.sub(r"\s+", "", content[end:min(len(content), end + 4)])[:1]
    if surface == "他" and previous == "其":
        return True
    if surface == "它" and previous == "其":
        return True
    if surface == "其":
        # Lexicalized adverbs, quantifiers and compounds: 尤其/与其/极其/何其,
        # 其实/其次/其余/其他/其它/其后/其间/其一.  In contrast, a possessive
        # determiner before an ordinary noun (其模型) remains a candidate.
        return bool(
            previous in {"尤", "与", "與", "极", "極", "何"}
            or following in {
                "实", "實", "次", "余", "餘", "他", "它", "后", "後",
                "间", "間", "一", "二", "三", "四", "五", "六", "七",
                "八", "九",
            }
        )
    if surface == "该" and previous in {"应", "應"}:
        return True
    if surface == "此" and previous in {"因", "彼"}:
        return True
    return False

_EXPLICIT_REFLEXIVE_REFERENCES = {
    "himself", "herself", "itself", "themselves", "myself", "yourself",
    "yourselves", "ourselves", "oneself", "self", "自己", "自身", "本身",
    "自ら", "自分", "자신", "자기",
}


def _is_explicit_reflexive_reference(surface: str) -> bool:
    return str(surface or "").strip().casefold() in (
        _EXPLICIT_REFLEXIVE_REFERENCES
    )


def _relation_reference_candidates(
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Prelocate generic anaphoric forms without assigning an identity.

    Character boundaries are deterministic parser facts; only the model may
    bind a candidate to one frozen antecedent.  The inventory deliberately
    covers closed-class reference expressions across common document
    languages and never contains domain vocabulary.
    """

    frozen_spans = [
        (
            int(item.get("mention_start") or 0),
            int(item.get("mention_end") or 0),
        )
        for item in occurrences
        if not bool(item.get("relation_reference_proxy"))
    ]
    candidates: list[dict[str, Any]] = []
    for match in _RELATION_REFERENCE_PATTERN.finditer(content):
        start, end = match.span()
        surface = content[start:end]
        if _reference_is_lexical_subtoken(content, start, end, surface):
            continue
        if any(left < end and start < right for left, right in frozen_spans):
            continue
        if len(candidates) >= 96:
            break
        candidates.append({
            "id": f"R{len(candidates) + 1:04d}",
            "surface": surface,
            "start": start,
            "end": end,
        })
    return candidates


_REFERENCE_DETERMINER_FORMS = {
    "his", "her", "its", "their", "whose", "sein", "seine", "ihr",
    "ihre", "dessen", "deren", "son", "sa", "ses", "leur", "leurs",
    "su", "sus", "seu", "sua", "seus", "suas", "其", "该", "此",
    "それ", "그것", "해당",
}

# Only this subset of determiner-shaped references carries a grammatical
# possessor role. Demonstratives such as ``this``/``该`` still remain excluded
# as standalone relation arguments, but must not manufacture ownership.
_POSSESSIVE_DETERMINER_FORMS = {
    "his", "her", "its", "their", "whose", "sein", "seine", "ihr",
    "ihre", "dessen", "deren", "son", "sa", "ses", "leur", "leurs",
    "su", "sus", "seu", "sua", "seus", "suas", "其",
}

# Neuter object references can denote animals represented by a coarse Person
# type, so type alone is not a safe resolver.  It is nevertheless unsafe to
# accept a more distant Person antecedent when the same atomic source scope
# contains a closer explicit non-Person candidate.  That combination is a
# strong, language-level disagreement between the model choice, agreement and
# recency; quarantine it instead of manufacturing a relation endpoint.
_NEUTER_REFERENCE_FORMS = {
    "it", "its", "itself", "es", "cela", "ceci", "lo", "isso",
    "它", "它们", "它們", "それ", "그것",
}

_REFERENCE_NON_PREDICATE_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their", "of", "to", "from", "in", "on",
    "at", "by", "with", "for", "and", "or", "nor", "but", "if", "because",
    "although", "though", "while", "when", "where", "which", "who", "whom",
    "whose", "before", "after", "merely", "just", "very", "not", "no",
}


def _reference_has_predicate_material(value: str) -> bool:
    if any(
        marker in value
        for marker in ("\n", "|", ";", "；", ",", "，", ".", "。", ":", "：")
    ):
        return False
    words = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ]+|[\u3400-\u9fff]+|"
        r"[\u3040-\u30ff]+|[\uac00-\ud7af]+",
        str(value or ""),
    )
    return any(
        word.casefold() not in _REFERENCE_NON_PREDICATE_WORDS
        for word in words
    )


def _reference_resolution_catalog(
    *,
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    max_references: int = 24,
    max_antecedents: int = 10,
) -> dict[str, Any]:
    """Build bounded, predicate-blind coreference decisions worth adjudicating."""

    model_catalog, occurrence_by_model_id = _open_fact_occurrence_catalog(
        occurrences
    )
    model_id_by_key = {
        str(item.get("occurrence_key") or ""): model_id
        for model_id, item in occurrence_by_model_id.items()
    }
    explicit = sorted(
        (
            item for item in occurrences
            if not bool(item.get("relation_reference_proxy"))
            and str(item.get("occurrence_key") or "")
            and str(item.get("canonical_id") or "")
        ),
        key=lambda item: (
            int(item.get("mention_start") or 0),
            int(item.get("mention_end") or 0),
            str(item.get("occurrence_key") or ""),
        ),
    )
    resolved_spans = {
        (
            int(item.get("mention_start") or 0),
            int(item.get("mention_end") or 0),
        )
        for item in occurrences
        if bool(item.get("relation_reference_proxy"))
    }
    subject_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    by_key = {
        str(item.get("occurrence_key") or ""): item for item in occurrences
    }
    for frame in inventory.get("frames") or ():
        if not isinstance(frame, Mapping):
            continue
        keys = [str(value) for value in (
            frame.get("argument_occurrence_keys") or ()
        )]
        if len(keys) != 2:
            continue
        for index, counts in ((0, subject_counts), (1, object_counts)):
            occurrence = by_key.get(keys[index])
            canonical = str((occurrence or {}).get("canonical_id") or "")
            if canonical:
                counts[canonical] = counts.get(canonical, 0) + 1
    scopes = split_relation_scopes(content)
    targets: list[dict[str, Any]] = []
    for reference in _relation_reference_candidates(content, occurrences):
        if len(targets) >= max(0, int(max_references)):
            break
        start, end = int(reference["start"]), int(reference["end"])
        if (start, end) in resolved_spans:
            continue
        containing = [
            (scope_start, scope_end)
            for scope_start, scope_end, _text in scopes
            if scope_start <= start < end <= scope_end
        ]
        if len(containing) != 1:
            continue
        scope_start, scope_end = containing[0]
        scope_occurrences = [
            item for item in explicit
            if scope_start <= int(item.get("mention_start") or 0)
            < int(item.get("mention_end") or 0) <= scope_end
        ]
        before = [
            item for item in scope_occurrences
            if int(item.get("mention_end") or 0) <= start
        ]
        after = [
            item for item in scope_occurrences
            if int(item.get("mention_start") or 0) >= end
        ]
        previous = before[-1] if before else None
        following = after[0] if after else None
        surface = str(reference.get("surface") or "")
        immediate_determiner = bool(
            following is not None
            and surface.casefold() in _REFERENCE_DETERMINER_FORMS
            and not content[
                end:int(following.get("mention_start") or 0)
            ].strip()
        )
        possessive_determiner = bool(
            immediate_determiner
            and surface.casefold() in _POSSESSIVE_DETERMINER_FORMS
        )
        if immediate_determiner and not possessive_determiner:
            # Possessive/determiner references contribute to the following
            # noun phrase; non-possessive demonstratives are not standalone
            # relation arguments and do not imply an owner.
            continue
        left_material = (
            content[int(previous.get("mention_end") or 0):start]
            if previous is not None else ""
        )
        right_material = (
            content[end:int(following.get("mention_start") or 0)]
            if following is not None else ""
        )
        reference_role = ""
        other_endpoint = None
        if possessive_determiner:
            reference_role = "possessive_determiner"
            other_endpoint = following
        elif following is not None and _reference_has_predicate_material(
            right_material
        ):
            reference_role = "grammatical_subject"
            other_endpoint = following
        elif previous is not None and _reference_has_predicate_material(
            left_material
        ):
            reference_role = "predicate_complement"
            other_endpoint = previous
            # ``entity with attribute VERB reference``: the attribute is not
            # the relation's grammatical subject. Preserve the earlier entity
            # as the other endpoint and expose that role to the adjudicator.
            if len(before) >= 2:
                governor, modifier = before[-2], before[-1]
                modifier_text = content[
                    int(governor.get("mention_end") or 0):
                    int(modifier.get("mention_start") or 0)
                ]
                if re.fullmatch(
                    r"[\s,()\[\]-]*(?:with|having|bearing|featuring|"
                    r"including|whose|带有|具有|含有|包括)"
                    r"[\s\wÀ-ÖØ-öø-ÿ-]{0,64}",
                    modifier_text,
                    flags=re.IGNORECASE,
                ):
                    other_endpoint = governor
                    reference_role = "predicate_complement_after_modifier"
        if other_endpoint is None:
            continue
        other_canonical = str(other_endpoint.get("canonical_id") or "")
        prior_by_canonical: dict[str, list[Mapping[str, Any]]] = {}
        for occurrence in explicit:
            if int(occurrence.get("mention_end") or 0) > start:
                break
            canonical = str(occurrence.get("canonical_id") or "")
            if not canonical or canonical == other_canonical:
                continue
            prior_by_canonical.setdefault(canonical, []).append(occurrence)
        if not prior_by_canonical:
            continue
        ranked_groups = sorted(
            prior_by_canonical.items(),
            key=lambda item: (
                -subject_counts.get(item[0], 0),
                -len(item[1]),
                -int(item[1][-1].get("mention_end") or 0),
                item[0],
            ),
        )[:max(1, int(max_antecedents))]
        candidates = []
        allowed_ids: list[str] = []
        for canonical, values in ranked_groups:
            representative = values[-1]
            model_id = model_id_by_key.get(
                str(representative.get("occurrence_key") or ""), "",
            )
            if not model_id:
                continue
            allowed_ids.append(model_id)
            candidates.append({
                "antecedent_id": model_id,
                "canonical_group": str(
                    next((
                        row.get("same_entity_group")
                        for row in model_catalog
                        if row.get("id") == model_id
                    ), "")
                ),
                "representative_surface": str(
                    representative.get("surface") or ""
                ),
                "type": str(representative.get("type") or "GenericEntity"),
                "prior_occurrences": [{
                    "surface": str(value.get("surface") or ""),
                    "start": int(value.get("mention_start") or 0),
                    "end": int(value.get("mention_end") or 0),
                } for value in values[-4:]],
                "matrix_subject_fact_count": subject_counts.get(canonical, 0),
                "object_fact_count": object_counts.get(canonical, 0),
            })
        if not candidates:
            continue
        context_start = max(0, start - 2200)
        context_end = min(len(content), end + 500)
        targets.append({
            "reference_id": str(reference["id"]),
            "surface": surface,
            "start": start,
            "end": end,
            "reference_role": reference_role,
            "other_endpoint": {
                "occurrence_key": str(
                    other_endpoint.get("occurrence_key") or ""
                ),
                "surface": str(other_endpoint.get("surface") or ""),
                "start": int(other_endpoint.get("mention_start") or 0),
                "end": int(other_endpoint.get("mention_end") or 0),
                "canonical_id": other_canonical,
            },
            "context": {
                "start": context_start,
                "end": context_end,
                "text": content[context_start:context_end],
            },
            "candidates": candidates,
            "allowed_antecedent_ids": allowed_ids,
        })
    return {
        "targets": targets,
        "occurrence_by_model_id": occurrence_by_model_id,
    }


def relation_reference_resolution_prompt(
    *,
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> str:
    catalog = _reference_resolution_catalog(
        content=content,
        occurrences=occurrences,
        inventory=inventory,
    )
    return json.dumps({
        "instruction": (
            "Resolve only the supplied exact local references. Track the continuing "
            "discourse center and grammatical roles across the frozen context. A "
            "repeated matrix subject or viewpoint participant is normally more "
            "salient than a possessor, embedded noun-phrase participant, or incidental "
            "object, but source meaning is authoritative. other_endpoint is the "
            "reference's non-reflexive relation counterpart and is deliberately absent "
            "from candidates; never recreate a self-loop. A reference_role of "
            "possessive_determiner asks for the possessor antecedent of the immediately "
            "following frozen noun phrase, not for that noun phrase itself. "
            "For a neuter object reference, do not choose a more distant person when "
            "a closer explicit non-person antecedent occurs in the same atomic clause. "
            "For each reference, choose "
            "only one antecedent_id listed in that reference's candidates when the "
            "identity is unambiguous; otherwise choose UNRESOLVED. Set unambiguous true "
            "only for a source-supported choice. Do not resolve by world knowledge, "
            "invent an entity, or reuse a decision from another item. Return exactly "
            "coverage_complete and resolutions, with one unique resolution for every "
            "reference_id."
        ),
        "contract_version": GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION,
        "references": [
            {
                key: value for key, value in target.items()
                if key != "allowed_antecedent_ids"
            }
            for target in catalog["targets"]
        ],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def relation_reference_resolution_response_schema(
    *,
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    catalog = _reference_resolution_catalog(
        content=content,
        occurrences=occurrences,
        inventory=inventory,
    )
    reference_ids = [
        str(item["reference_id"]) for item in catalog["targets"]
    ]
    antecedent_ids = sorted({
        str(value)
        for item in catalog["targets"]
        for value in item["allowed_antecedent_ids"]
    })
    return {
        "$id": GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "coverage_complete": {"type": "boolean"},
            "resolutions": {
                "type": "array",
                "minItems": len(reference_ids),
                "maxItems": len(reference_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "reference_id": {
                            "type": "string", "enum": reference_ids,
                        },
                        "antecedent_id": {
                            "type": "string",
                            "enum": [*antecedent_ids, "UNRESOLVED"],
                        },
                        "unambiguous": {"type": "boolean"},
                    },
                    "required": [
                        "reference_id", "antecedent_id", "unambiguous",
                    ],
                },
            },
        },
        "required": ["coverage_complete", "resolutions"],
    }


def parse_relation_reference_resolutions(
    raw: Any,
    *,
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
    allow_partial: bool = False,
) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "reference resolution returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "reference resolution response must be an object"
        )
    resolutions = raw.get("resolutions")
    if not isinstance(resolutions, list):
        raise RelationClassificationError(
            "reference resolution requires resolutions"
        )
    catalog = _reference_resolution_catalog(
        content=content,
        occurrences=occurrences,
        inventory=inventory,
    )
    target_by_id = {
        str(item["reference_id"]): item for item in catalog["targets"]
    }
    occurrence_by_model_id = catalog["occurrence_by_model_id"]
    seen: set[str] = set()
    proxies: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item_index, item in enumerate(resolutions):
        reason = ""
        if not isinstance(item, Mapping):
            reason = "resolution_not_object"
            reference_id = antecedent_id = ""
        else:
            reference_id = str(item.get("reference_id") or "")
            antecedent_id = str(item.get("antecedent_id") or "")
        target = target_by_id.get(reference_id)
        if not reason and target is None:
            reason = "unknown_reference_id"
        if not reason and reference_id in seen:
            reason = "duplicate_reference_id"
        if not reason:
            seen.add(reference_id)
        if not reason and antecedent_id == "UNRESOLVED":
            continue
        if not reason and item.get("unambiguous") is not True:
            reason = "reference_not_unambiguous"
        if not reason and antecedent_id not in set(
            target["allowed_antecedent_ids"]
        ):
            reason = "antecedent_not_allowed_for_reference"
        antecedent = occurrence_by_model_id.get(antecedent_id)
        if not reason and (
            antecedent is None
            or bool(antecedent.get("relation_reference_proxy"))
        ):
            reason = "antecedent_not_explicit_occurrence"
        if not reason and str(target.get("surface") or "").strip().casefold() in (
            _NEUTER_REFERENCE_FORMS
        ) and str(target.get("reference_role") or "") == "predicate_complement" and str(
            (antecedent or {}).get("type") or ""
        ).strip().casefold() == "person":
            reference_start = int(target.get("start") or 0)
            containing_scopes = [
                (scope_start, scope_end)
                for scope_start, scope_end, _scope_text in split_relation_scopes(
                    content
                )
                if scope_start <= reference_start < scope_end
            ]
            scope_start = max(
                (value[0] for value in containing_scopes), default=0,
            )
            selected_candidate = next((
                candidate for candidate in target.get("candidates") or ()
                if str(candidate.get("antecedent_id") or "") == antecedent_id
            ), {})
            selected_latest_end = max((
                int(value.get("end") or 0)
                for value in selected_candidate.get("prior_occurrences") or ()
                if scope_start <= int(value.get("end") or 0) <= reference_start
            ), default=-1)
            closer_non_person = any(
                str(candidate.get("type") or "").strip().casefold() != "person"
                and any(
                    selected_latest_end < int(value.get("end") or 0)
                    <= reference_start
                    and scope_start <= int(value.get("start") or 0)
                    for value in candidate.get("prior_occurrences") or ()
                )
                for candidate in target.get("candidates") or ()
                if str(candidate.get("antecedent_id") or "") != antecedent_id
            )
            if closer_non_person:
                reason = "neuter_reference_conflicts_with_closer_nonperson"
        if reason:
            if not allow_partial:
                raise RelationClassificationError(
                    f"reference resolution invalid: {reason}"
                )
            dropped.append({"item_index": item_index, "reason": reason})
            continue
        start, end = int(target["start"]), int(target["end"])
        surface = str(target["surface"])
        if content[start:end] != surface:
            if not allow_partial:
                raise RelationClassificationError(
                    "reference resolution source span drifted"
                )
            dropped.append({
                "item_index": item_index,
                "reason": "reference_source_span_drifted",
            })
            continue
        antecedent_key = str(antecedent.get("occurrence_key") or "")
        material = json.dumps(
            [antecedent_key, start, end, surface],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        proxies.append({
            "occurrence_key": "relation-reference-" + hashlib.sha256(
                material.encode("utf-8")
            ).hexdigest(),
            "mention_id": str(
                antecedent.get("mention_id") or antecedent_key
            ),
            "surface": surface,
            "mention_start": start,
            "mention_end": end,
            "canonical_id": str(antecedent.get("canonical_id") or ""),
            "assignment_version": str(
                antecedent.get("assignment_version") or ""
            ),
            "type": str(antecedent.get("type") or "GenericEntity"),
            "source_role": "body",
            "relation_eligible": True,
            "endpoint_candidate": True,
            "relation_reference_proxy": True,
            "reference_id": reference_id,
            "reference_antecedent_occurrence_key": antecedent_key,
            "reference_antecedent_model_id": antecedent_id,
            "reference_resolution_source": "DISCOURSE_ROLE_ADJUDICATOR",
            "reference_role": str(target.get("reference_role") or ""),
            "reference_other_endpoint_occurrence_key": str(
                (target.get("other_endpoint") or {}).get("occurrence_key") or ""
            ),
        })
    missing = sorted(set(target_by_id) - seen)
    if missing and not allow_partial:
        raise RelationClassificationError(
            "reference resolution omitted references"
        )
    return {
        "coverage_complete": (
            bool(raw.get("coverage_complete")) and not missing and not dropped
        ),
        "reference_proxies": proxies,
        "dropped_resolutions": dropped,
        "missing_reference_ids": missing,
        "parser_version": GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION,
    }


def grounded_open_fact_prompt(
    *,
    context_id: str,
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
    context_kind: str = "",
    already_found: Sequence[Mapping[str, Any]] = (),
    failed_items: Sequence[Mapping[str, Any]] = (),
    recovery_pass: bool = False,
) -> str:
    """Build a predicate-open, occurrence-grounded local fact request."""

    catalog, by_id = _open_fact_occurrence_catalog(occurrences)
    reference_candidates = _relation_reference_candidates(
        content, occurrences,
    )
    key_to_id = {
        str(occurrence.get("occurrence_key") or ""): model_id
        for model_id, occurrence in by_id.items()
    }
    found = []
    found_occurrence_keys: set[str] = set()
    found_occurrence_pairs: list[tuple[str, str]] = []
    for item in already_found:
        if not isinstance(item, Mapping):
            continue
        occurrence_keys = list(item.get("argument_occurrence_keys") or [])
        if len(occurrence_keys) != 2:
            continue
        if all(str(value) in key_to_id for value in occurrence_keys):
            left_key, right_key = (str(value) for value in occurrence_keys)
            found_occurrence_keys.update((left_key, right_key))
            found_occurrence_pairs.append((left_key, right_key))
            found.append({
                "subject_id": key_to_id[left_key],
                "predicate": str(item.get("relation") or ""),
                "object_id": key_to_id[right_key],
                "evidence": str(item.get("evidence_text") or ""),
            })
    invalid_attempts: list[dict[str, Any]] = []
    for item in failed_items:
        if not isinstance(item, Mapping):
            continue
        attempt = item.get("attempt")
        if not isinstance(attempt, Mapping):
            continue
        invalid_attempts.append({
            "reason": str(item.get("reason") or "")[:120],
            "subject_id": str(attempt.get("subject_id") or "")[:200],
            "predicate": str(attempt.get("predicate") or "")[:200],
            "object_id": str(attempt.get("object_id") or "")[:200],
            "evidence": str(attempt.get("evidence") or "")[:1200],
        })
        if len(invalid_attempts) >= 64:
            break
    # A first pass can legitimately return zero rows. The bounded omission
    # pass must still be a distinct request with a complete mention/scope
    # checklist; otherwise request-hash caching can replay the empty primary
    # response and falsely certify saturation.
    recovering = bool(recovery_pass or found or invalid_attempts)
    coverage_audit: dict[str, Any] = {}
    if recovering:
        uncovered_ids = [
            model_id for model_id, occurrence in by_id.items()
            if str(occurrence.get("occurrence_key") or "")
            not in found_occurrence_keys
            and (
                occurrence.get("relation_eligible") is None
                or bool(occurrence.get("relation_eligible"))
            )
        ]
        source_units: list[dict[str, Any]] = []
        for unit_index, (start, end, _text) in enumerate(
            split_relation_scopes(content)
        ):
            unit_keys = [
                str(occurrence.get("occurrence_key") or "")
                for occurrence in by_id.values()
                if start <= int(occurrence.get("mention_start") or 0)
                < int(occurrence.get("mention_end") or 0) <= end
            ]
            if len(unit_keys) < 2:
                continue
            accepted_count = sum(
                left in unit_keys and right in unit_keys
                for left, right in found_occurrence_pairs
            )
            # A unit with no accepted fact is an obvious omission target.  A
            # unit with three or more mentions needs a bounded role audit even
            # when one actor/theme edge was already found, because transfers,
            # coordinated objects and nested clauses commonly contain another
            # directly stated role edge.
            if accepted_count == 0 or len(unit_keys) >= 3:
                source_units.append({
                    "unit_id": f"U{unit_index:04d}",
                    "start": start,
                    "end": end,
                    "mention_ids": [key_to_id[key] for key in unit_keys],
                    "accepted_fact_count": accepted_count,
                })
        coverage_audit = {
            "mention_ids_without_accepted_fact": uncovered_ids,
            "multi_mention_source_units": source_units,
            "unresolved_reference_candidates": [
                dict(item) for item in reference_candidates
                if any(
                    int(unit["start"]) <= int(item["start"])
                    < int(item["end"]) <= int(unit["end"])
                    for unit in source_units
                )
            ],
        }
    recovery = (
        "This is one bounded omission-recovery pass. Do not repeat any "
        "already_found item. Re-read every source unit and return only omitted "
        "direct facts. invalid_attempts are rejected prior rows, not accepted "
        "facts. Where an invalid row used a clause or free text instead of a "
        "mention ID, decompose the directly stated clause into every valid "
        "binary role fact and replace both endpoints with frozen mention IDs; "
        "if no valid frozen endpoint expresses it, omit it. Do not echo SOURCE, "
        "MENTIONS, already_found, invalid_attempts, this instruction, "
        "coverage_audit, or any input field. Use coverage_audit "
        "as a checklist: inspect every listed uncovered mention and every "
        "listed multi-mention source unit for omitted predicates or semantic "
        "roles. A checklist item can legitimately have no direct relation; "
        "never invent one merely to cover the item. "
        if recovering else ""
    )
    instruction = (
        recovery
        + "Using only SOURCE and the frozen MENTIONS, enumerate every directly "
        "stated binary semantic fact. Endpoints must be supplied mention IDs; "
        "never invent, concatenate, shorten or replace an endpoint. Use the "
        "exact mention occurrence inside the evidence; evidence must span both "
        "selected occurrences. A MENTIONS row marked "
        "is_resolved_reference=true is an already resolved local reference: "
        "use that exact local occurrence when it participates in a fact. "
        "Otherwise, for unambiguous coreference use an explicit antecedent "
        "mention with the same_entity_group and extend the contiguous evidence "
        "to include that antecedent. Never perform a new uncertain coreference "
        "guess in this stage. When a directly participating pronoun, demonstrative "
        "or anaphoric noun phrase is local to the fact but is absent from MENTIONS, "
        "add it to resolved_references with an R-prefixed reference_id, its exact "
        "SOURCE text, and one explicit non-reference MENTIONS antecedent. Do not "
        "calculate character offsets; the parser locates the text inside the fact's "
        "exact evidence. "
        "REFERENCE_CANDIDATES already supplies deterministic IDs and exact offsets "
        "for likely local references. Prefer one of those IDs and bind it only to "
        "an explicit antecedent MENTION; never change its text or position. "
        "Do this only when the identity is unambiguous from SOURCE; do not resolve "
        "bridging associations, implicit actors or world knowledge. A fact may then "
        "use that reference_id as an endpoint. A non-reflexive reference must never "
        "be bound to the fact's other endpoint when that would turn an ordinary "
        "directed clause into a self-loop; inspect earlier compatible mentions for "
        "the actual antecedent and leave the reference unresolved if it remains "
        "ambiguous. Emit only references actually used by a returned fact. Before "
        "finishing, audit every REFERENCE_CANDIDATE that lies between a lexical "
        "predicate and another mention, or follows a predicate-bearing modified "
        "subject. If its antecedent is unambiguous, emit the corresponding direct "
        "fact; do not consider a different fact involving the same canonical pair "
        "to be coverage of this exact source occurrence. "
        "predicate must copy one exact contiguous relation phrase from evidence, "
        "in the same language and wording as SOURCE; do not translate, paraphrase, "
        "summarize, invent a protocol label, or replace the source wording with a "
        "logical relation name. It must be short and reusable without endpoint "
        "names or surrounding clause text, and contain a lexical semantic "
        "head: articles, determiners, quantifiers, conjunctions, prepositions, "
        "possessive markers or an unfinished modal phrase by themselves are not "
        "predicates. This stage is predicate-open: do not "
        "map to a predefined logical relation. Split coordinated roles and emit "
        "every independently entailed role edge. Decompose explicit nested "
        "predicates, modifiers and relative clauses, action participants, sources, "
        "destinations, recipients, locations, parts, properties and complements. "
        "For transfer, communication and other multi-role actions, emit the "
        "actor-to-theme fact and each explicitly stated theme-to-recipient, "
        "theme-to-destination or theme-to-source fact separately. "
        "For a verbal table cell, use its verb as the relation and its nominal "
        "participant as an endpoint. Interpret every semantic data cell through "
        "its row key and column-header role. A row identifier, reference, category "
        "or requirement cell explicitly assigned by its header must produce a "
        "row-key-to-cell fact; never connect sibling headers or untyped sibling "
        "values merely because they are adjacent. A label or identifier immediately "
        "prefixing a clause explicitly binds that clause: emit its direct binding "
        "to the principal actor or topic. If a map, index or container explicitly "
        "says it connects X to Y, emit both its links to X and Y and the stated "
        "X-to-Y connection. Treat section/list labels as structural binders only "
        "when source hierarchy explicitly binds them to a child. Must/shall statements "
        "are explicit normative facts, not uncertainty; omit the auxiliary from "
        "the relation phrase while retaining its normative meaning. Preserve an "
        "explicit negative relation in the predicate phrase and never emit its "
        "positive opposite. Do not infer commonsense, identity, ownership, purpose, "
        "tool use or any relation from co-occurrence. evidence must be one exact "
        "contiguous SOURCE substring covering the supporting clause, code line, "
        "list item, table header plus row, or hierarchy block. Return only facts "
        "as exactly one JSON object with resolved_references and facts; "
        "resolved_references is an array and facts is an array. Every fact has "
        "exactly subject_id, predicate, object_id and evidence."
    )
    payload = {
        "instruction": instruction,
        # Keep the primary request semantically immutable when only the
        # bounded recovery protocol evolves.  This makes model-backed A/B
        # measurements and cached source inventories genuinely comparable.
        "contract_version": (
            GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION
            if recovering else GROUNDED_RELATION_OPEN_FACT_PRIMARY_CONTRACT_VERSION
        ),
        "context_id": str(context_id),
        "context_kind": str(context_kind),
        "source": str(content),
        "mentions": catalog,
        "reference_candidates": reference_candidates,
        "already_found": found,
        "output_contract": {
            "resolved_references": [{
                "reference_id": "R0001",
                "surface": (
                    "omit for a prelocated REFERENCE_CANDIDATES id; otherwise "
                    "copy the exact local referring expression"
                ),
                "antecedent_id": "one explicit non-reference MENTIONS id",
            }],
            "facts": [{
                "subject_id": "one MENTIONS or resolved reference id",
                "predicate": "short reusable source relation",
                "object_id": "one MENTIONS or resolved reference id",
                "evidence": "exact contiguous SOURCE substring",
            }],
        },
    }
    if coverage_audit:
        payload["coverage_audit"] = coverage_audit
    if invalid_attempts:
        payload["invalid_attempts"] = invalid_attempts
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def grounded_open_fact_response_schema(
    occurrences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    catalog, _by_id = _open_fact_occurrence_catalog(occurrences)
    ids = [str(item["id"]) for item in catalog]
    endpoint_id = {
        "anyOf": [
            {"type": "string", "enum": ids},
            {"type": "string", "pattern": "^R[0-9]{4}$"},
        ],
    }
    return {
        "$id": GROUNDED_RELATION_OPEN_FACT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "resolved_references": {
                "type": "array",
                "maxItems": 48,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "reference_id": {
                            "type": "string", "pattern": "^R[0-9]{4}$",
                        },
                        "surface": {
                            "type": "string", "minLength": 1,
                            "maxLength": 200,
                        },
                        "antecedent_id": {
                            "type": "string", "enum": ids,
                        },
                    },
                    "required": [
                        "reference_id", "antecedent_id",
                    ],
                },
            },
            "facts": {
                "type": "array",
                "maxItems": 128,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "subject_id": endpoint_id,
                        "predicate": {
                            "type": "string", "minLength": 1, "maxLength": 160,
                        },
                        "object_id": endpoint_id,
                        "evidence": {
                            "type": "string", "minLength": 1, "maxLength": 1200,
                        },
                    },
                    "required": [
                        "subject_id", "predicate", "object_id", "evidence",
                    ],
                },
            },
        },
        "required": ["resolved_references", "facts"],
    }


def _locate_open_fact_evidence(
    content: str,
    evidence: str,
    *,
    near: int,
) -> tuple[int, int] | None:
    exact = _locate_all_open_fact_evidence(content, evidence)
    if not exact:
        return None
    return min(exact, key=lambda span: abs(span[0] - int(near)))


def _locate_all_open_fact_evidence(
    content: str,
    evidence: str,
) -> list[tuple[int, int]]:
    copied = str(evidence or "").strip()
    if not copied:
        return []
    exact: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = content.find(copied, cursor)
        if start < 0:
            break
        exact.append((start, start + len(copied)))
        cursor = start + 1
    if not exact:
        parts = [part for part in re.split(r"\s+", copied) if part]
        pattern = r"\s+".join(re.escape(part) for part in parts)
        exact = [
            (match.start(), match.end())
            for match in re.finditer(pattern, content, re.IGNORECASE)
        ] if pattern else []
    if not exact:
        return []
    return exact


_OPEN_PREDICATE_ENGLISH_CLOSED_CLASS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "such",
    "all", "any", "each", "every", "some", "other", "another",
    "and", "or", "nor", "but", "either", "neither", "as",
    "of", "to", "for", "by", "with", "without", "in", "on", "at",
    "from", "into", "onto", "through", "via", "within", "between",
    "its", "their", "his", "her", "our", "your", "whose", "which",
    "who", "whom", "where", "when", "while", "than", "then",
    "can", "could", "may", "might", "must", "shall", "should", "will",
    "would", "not", "no", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "do", "does", "did",
})
_OPEN_PREDICATE_ENGLISH_COPULAS = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
})
_OPEN_PREDICATE_ENGLISH_RELATIONAL_PREPOSITIONS = frozenset({
    "for", "by", "with", "without", "in", "on", "at", "from", "into",
    "onto", "through", "via", "within", "between", "inside", "outside",
    "under", "over", "near", "beside",
})
_OPEN_PREDICATE_ENGLISH_STRONG_PREPOSITIONS = frozenset({
    "between", "from", "into", "onto", "through", "via", "within",
    "inside", "outside", "under", "over", "near", "beside",
})
_OPEN_PREDICATE_ENGLISH_QUANTIFIERS = frozenset({
    "all", "any", "each", "every", "some", "other", "another",
})
_OPEN_PREDICATE_CJK_MULTI_CLOSED_CLASS = tuple(sorted({
    "各种", "各类", "所有", "全部", "每个", "每种", "一些", "其他", "其它",
    "以及", "并且", "或者", "应该", "必须", "可能", "能够", "可以",
    "这些", "那些", "上述", "以下", "前述", "其中", "之中",
}, key=len, reverse=True))
_OPEN_PREDICATE_CJK_SINGLE_CLOSED_CLASS = frozenset({
    "的", "地", "得", "并", "和", "与", "及", "或", "其", "该", "此",
    "之", "所", "被", "把", "由", "向", "对", "于", "在", "从", "到",
    "为", "是", "有", "中", "内", "外",
})
_OPEN_PREDICATE_INCOMPLETE_CJK_MODAL = re.compile(
    r"^(?:该|此|本|上述|以下|其).{1,32}(?:可|可以|会|将|应|应该|必须|"
    r"可能|能够|能)$"
)
_OPEN_PREDICATE_INCOMPLETE_ENGLISH_MODAL = re.compile(
    r"^(?:the|this|that|these|those|such|it|they|he|she|we)\b.+\b"
    r"(?:can|could|may|might|must|shall|should|will|would)$",
    re.IGNORECASE,
)


def _open_predicate_quality_reason(value: Any) -> str:
    """Reject closed-class or visibly unfinished open relation phrases.

    This is intentionally a small language-form gate, not a predicate
    ontology. Any phrase with a lexical relation head remains available to
    Schema binding and independent verification; only rows that cannot name a
    semantic relation at all are removed before they can become a generic
    ``source_assertion`` edge.
    """

    predicate = re.sub(r"\s+", " ", str(value or "")).strip()
    lexical = re.sub(
        r"[^0-9A-Za-z_\u3400-\u4dbf\u4e00-\u9fff]+", " ", predicate,
    ).strip()
    if not lexical:
        return "predicate_missing_semantic_head"
    if (
        _OPEN_PREDICATE_INCOMPLETE_CJK_MODAL.fullmatch(lexical)
        or _OPEN_PREDICATE_INCOMPLETE_ENGLISH_MODAL.fullmatch(lexical)
    ):
        return "predicate_incomplete_modal_fragment"

    latin_tokens = [
        token.casefold() for token in re.findall(r"[A-Za-z]+", lexical)
    ]
    has_latin_head = any(
        token not in _OPEN_PREDICATE_ENGLISH_CLOSED_CLASS
        for token in latin_tokens
    )
    if not has_latin_head and latin_tokens:
        token_set = set(latin_tokens)
        relational_closed_construction = bool(
            tuple(latin_tokens) == ("such", "as")
            or token_set & _OPEN_PREDICATE_ENGLISH_COPULAS
            or token_set & {"has", "have", "had"}
            or (
                not token_set & _OPEN_PREDICATE_ENGLISH_QUANTIFIERS
                and token_set
                & _OPEN_PREDICATE_ENGLISH_STRONG_PREPOSITIONS
            )
        )
        has_latin_head = relational_closed_construction

    cjk_segments = re.findall(
        r"[\u3400-\u4dbf\u4e00-\u9fff]+", lexical,
    )
    single_cjk_relational_carrier = bool(
        len(cjk_segments) == 1
        and (
            cjk_segments[0] in {"是", "有", "在", "为", "从", "向", "到"}
            or re.fullmatch(
                r"(?:中|内|外)(?:的(?:所有)?)?", cjk_segments[0]
            )
        )
    )

    def cjk_segment_has_head(segment: str) -> bool:
        remainder = segment
        previous = None
        while previous != remainder:
            previous = remainder
            for token in _OPEN_PREDICATE_CJK_MULTI_CLOSED_CLASS:
                remainder = remainder.replace(token, "")
        if not remainder:
            return False
        if all(
            character in _OPEN_PREDICATE_CJK_SINGLE_CLOSED_CLASS
            for character in remainder
        ):
            return False
        if re.fullmatch(r"[可会将应能]+", remainder):
            return False
        return True

    has_cjk_head = single_cjk_relational_carrier or any(
        cjk_segment_has_head(segment) for segment in cjk_segments
    )
    # Digits and identifier punctuation do not turn a connector into a
    # relation head. Open predicates in technical prose still pass through
    # their lexical Latin/CJK verb or relational noun.
    if not has_latin_head and not has_cjk_head:
        return "predicate_missing_semantic_head"
    return ""


def _repair_resolved_reference_nominal_prefix(
    relation: str,
    subject: Mapping[str, Any] | None,
    object_: Mapping[str, Any] | None,
    *,
    content: str,
    occurrences_by_key: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """Remove an anaphoric noun repeated at the start of a predicate.

    Small models sometimes split ``该核心服务主要包括 X`` as reference
    endpoint ``该`` plus predicate ``核心服务主要包括``.  The repeated
    nominal is still part of the anaphoric subject, not the relation.  A
    repair is safe only when the raw predicate is copied immediately after
    the frozen reference span and its removed prefix is an exact suffix of
    the frozen antecedent surface.  The original expression is returned for
    immutable audit metadata.
    """

    value = str(relation or "").strip()
    if (
        not value
        or not isinstance(subject, Mapping)
        or not isinstance(object_, Mapping)
        or not bool(subject.get("relation_reference_proxy"))
    ):
        return value, ""
    antecedent = occurrences_by_key.get(str(
        subject.get("reference_antecedent_occurrence_key") or ""
    ))
    antecedent_surface = str(
        (antecedent or {}).get("surface") or ""
    ).strip()
    if not antecedent_surface:
        return value, ""
    try:
        subject_end = int(subject.get("mention_end"))
        object_start = int(object_.get("mention_start"))
    except (TypeError, ValueError):
        return value, ""
    if not 0 <= subject_end < object_start <= len(content):
        return value, ""
    bridge = content[subject_end:object_start]
    copied_at = bridge.find(value)
    if copied_at < 0 or bridge[:copied_at].strip():
        return value, ""

    antecedent_identity = _normalized_atomic_surface(antecedent_surface)
    candidates: list[tuple[int, str]] = []
    for cut in range(1, len(value)):
        prefix = value[:cut].strip()
        remainder = value[cut:].strip()
        if not prefix or not remainder:
            continue
        prefix_identity = _normalized_atomic_surface(prefix)
        if (
            len(prefix_identity) < 2
            or not antecedent_identity.endswith(prefix_identity)
            or _open_predicate_quality_reason(remainder)
        ):
            continue
        # ASCII words must end at a lexical boundary. CJK nominal compounds
        # have no whitespace boundary, so exact antecedent suffix identity is
        # the controlling contract there.
        if (
            prefix[-1:].isascii()
            and prefix[-1:].isalnum()
            and remainder[:1].isascii()
            and remainder[:1].isalnum()
            and value[cut - 1:cut].isalnum()
            and value[cut:cut + 1].isalnum()
        ):
            continue
        candidates.append((len(prefix_identity), remainder))
    if not candidates:
        return value, ""
    _length, repaired = max(candidates, key=lambda item: item[0])
    return repaired, value


def parse_grounded_open_facts(
    raw: Any,
    *,
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate open facts and convert them to the source-frame inventory."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "grounded open fact response is invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "grounded open fact response must be an object"
        )
    raw_items = raw.get("facts")
    if not isinstance(raw_items, list):
        # The bounded recovery instruction calls these rows "omitted direct
        # facts".  Some schema-capable local models preserve that semantic
        # label even when the supplied response schema calls the array
        # ``facts``.  Accept only the list alias; every row still passes the
        # same immutable-ID, exact-evidence and self-loop validation below.
        for alias in (
            "items", "relations", "relationships", "omitted_facts",
        ):
            if isinstance(raw.get(alias), list):
                raw_items = raw[alias]
                break
    if not isinstance(raw_items, list):
        raise RelationClassificationError(
            "grounded open fact response requires a facts array"
        )
    if len(raw_items) > 128:
        raise RelationClassificationError(
            "grounded open fact response exceeded its bounded item count"
        )
    _catalog, frozen_by_id = _open_fact_occurrence_catalog(occurrences)
    prelocated_references = {
        str(item["id"]): item
        for item in _relation_reference_candidates(content, occurrences)
    }
    by_id = dict(frozen_by_id)
    occurrences_by_key = {
        str(item.get("occurrence_key") or ""): item
        for item in frozen_by_id.values()
        if str(item.get("occurrence_key") or "")
    }
    raw_references = raw.get("resolved_references")
    if raw_references is None:
        # Backward-compatible parsing is required for immutable cached calls
        # and scripted fixtures produced before the reference-aware schema.
        raw_references = []
    if not isinstance(raw_references, list):
        raise RelationClassificationError(
            "grounded open fact resolved_references must be an array"
        )
    if len(raw_references) > 48:
        raise RelationClassificationError(
            "grounded open fact response exceeded its reference count"
        )
    reference_proxies: list[dict[str, Any]] = []
    dropped_references: list[dict[str, Any]] = []
    accepted_reference_ids: set[str] = set()
    accepted_reference_spans: dict[
        tuple[int, int], tuple[str, str]
    ] = {}
    for reference_index, raw_reference in enumerate(raw_references):
        reason = ""
        if not isinstance(raw_reference, Mapping):
            reason = "reference_not_object"
            reference_id = surface = antecedent_id = ""
            start = end = -1
        else:
            reference_id = str(raw_reference.get("reference_id") or "")
            prelocated = prelocated_references.get(reference_id)
            surface = str(
                (prelocated or {}).get("surface")
                or raw_reference.get("surface")
                or ""
            )
            antecedent_id = str(raw_reference.get("antecedent_id") or "")
            if prelocated is not None:
                start = int(prelocated["start"])
                end = int(prelocated["end"])
            else:
                try:
                    start = int(raw_reference.get("start"))
                    end = int(raw_reference.get("end"))
                except (TypeError, ValueError):
                # Offsets were present in the first protocol prototype, but
                # local models count them unreliably. They are now only a
                # backward-compatible disambiguation hint; exact fact evidence
                # determines the accepted span.
                    start = end = -1
        antecedent = frozen_by_id.get(antecedent_id)
        if not reason and not re.fullmatch(r"R[0-9]{4}", reference_id):
            reason = "invalid_reference_id"
        if not reason and (
            reference_id in accepted_reference_ids
            or reference_id in frozen_by_id
        ):
            reason = "duplicate_reference_id"
        if not reason and antecedent is None:
            reason = "unknown_reference_antecedent"
        if not reason and bool(antecedent.get("relation_reference_proxy")):
            reason = "reference_antecedent_is_not_explicit_mention"
        if not reason and (not surface or len(surface) > 200):
            reason = "invalid_reference_surface"
        used_by_facts = [
            item for item in raw_items
            if isinstance(item, Mapping)
            and reference_id in {
                str(item.get("subject_id") or item.get("source_id") or ""),
                str(item.get("object_id") or item.get("target_id") or ""),
            }
        ]
        if not reason and not used_by_facts:
            reason = "unused_reference"
        candidate_spans: set[tuple[int, int]] = set()
        if not reason:
            for fact in used_by_facts:
                fact_evidence = str(
                    fact.get("evidence") or fact.get("evidence_text") or ""
                )
                for evidence_start, evidence_end in (
                    _locate_all_open_fact_evidence(content, fact_evidence)
                ):
                    cursor = evidence_start
                    while True:
                        candidate_start = content.find(
                            surface, cursor, evidence_end,
                        )
                        if candidate_start < 0:
                            break
                        candidate_end = candidate_start + len(surface)
                        if candidate_end <= evidence_end:
                            candidate_spans.add(
                                (candidate_start, candidate_end)
                            )
                        cursor = candidate_start + 1
        if not reason:
            exact_hint = (start, end)
            if len(candidate_spans) == 1:
                start, end = next(iter(candidate_spans))
            elif exact_hint in candidate_spans:
                start, end = exact_hint
            else:
                reason = "reference_not_uniquely_grounded_in_evidence"
        if not reason and _reference_is_lexical_subtoken(
            content, start, end, surface,
        ):
            reason = "reference_is_lexical_subtoken"
        span_binding = accepted_reference_spans.get((start, end))
        antecedent_canonical = str(
            (antecedent or {}).get("canonical_id") or ""
        )
        if not reason and span_binding is not None and span_binding != (
            surface, antecedent_canonical,
        ):
            reason = "conflicting_reference_binding"
        if reason:
            if not allow_partial:
                raise RelationClassificationError(
                    f"grounded open fact reference invalid: {reason}"
                )
            dropped_references.append({
                "reference_index": reference_index,
                "reason": reason,
            })
            continue
        existing_occurrence = next((
            occurrence
            for occurrence in frozen_by_id.values()
            if int(occurrence.get("mention_start") or 0) == start
            and int(occurrence.get("mention_end") or 0) == end
            and str(occurrence.get("canonical_id") or "")
            == antecedent_canonical
        ), None)
        if existing_occurrence is not None:
            proxy = dict(existing_occurrence)
        else:
            antecedent_key = str(
                antecedent.get("occurrence_key") or ""
            )
            reference_material = json.dumps(
                [antecedent_key, start, end, surface],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            proxy = {
                "occurrence_key": "relation-reference-" + hashlib.sha256(
                    reference_material.encode("utf-8")
                ).hexdigest(),
                # Keep materialization attached to the frozen antecedent
                # mention while retaining the exact local endpoint below.
                "mention_id": str(
                    antecedent.get("mention_id") or antecedent_key
                ),
                "surface": surface,
                "mention_start": start,
                "mention_end": end,
                "canonical_id": antecedent_canonical,
                "assignment_version": str(
                    antecedent.get("assignment_version") or ""
                ),
                "type": str(
                    antecedent.get("type") or "GenericEntity"
                ),
                "source_role": "body",
                "relation_eligible": True,
                "endpoint_candidate": True,
                "relation_reference_proxy": True,
                "reference_id": reference_id,
                "reference_antecedent_occurrence_key": antecedent_key,
                "reference_antecedent_model_id": antecedent_id,
            }
            reference_proxies.append(proxy)
        by_id[reference_id] = proxy
        accepted_reference_ids.add(reference_id)
        accepted_reference_spans[(start, end)] = (
            surface, antecedent_canonical,
        )
    frames: list[dict[str, Any]] = []
    dropped_items: list[dict[str, Any]] = []
    repair_items: list[dict[str, Any]] = []
    repaired_endpoint_occurrences = 0
    narrowed_evidence_scopes = 0
    reference_nominal_prefix_repairs = 0
    seen: set[tuple[str, str, str, int, int]] = set()
    relation_scopes = split_relation_scopes(content)

    def inside(
        occurrence: Mapping[str, Any], span: tuple[int, int],
    ) -> bool:
        start = int(occurrence.get("mention_start") or 0)
        end = int(occurrence.get("mention_end") or 0)
        return span[0] <= start < end <= span[1]

    def equivalent_occurrences(
        occurrence: Mapping[str, Any],
    ) -> list[tuple[str, Mapping[str, Any]]]:
        canonical_id = str(occurrence.get("canonical_id") or "").strip()
        if not canonical_id:
            return []
        return [
            (model_id, candidate)
            for model_id, candidate in by_id.items()
            if str(candidate.get("canonical_id") or "").strip()
            == canonical_id
        ]

    def realign_occurrence(
        occurrence: Mapping[str, Any],
        evidence_span: tuple[int, int],
    ) -> tuple[str, Mapping[str, Any]] | None:
        candidates = equivalent_occurrences(occurrence)
        copied_candidates = [
            item for item in candidates if inside(item[1], evidence_span)
        ]
        if len(copied_candidates) == 1:
            return copied_candidates[0]
        # Models sometimes copy only the predicate/object fragment. The exact
        # relation scope containing that fragment remains a safe place to
        # realign a repeated canonical mention; ambiguity stays fail-closed.
        overlapping_scopes = [
            (start, end)
            for start, end, _text in relation_scopes
            if start < evidence_span[1] and evidence_span[0] < end
        ]
        scoped_candidates = [
            item for item in candidates
            if any(inside(item[1], scope) for scope in overlapping_scopes)
        ]
        if len(scoped_candidates) == 1:
            return scoped_candidates[0]
        return None

    for item_index, item in enumerate(raw_items):
        reason = ""
        if not isinstance(item, Mapping):
            reason = "item_not_object"
            subject_id = object_id = predicate = evidence = ""
        else:
            subject_id = str(
                item.get("subject_id") or item.get("source_id") or ""
            )
            object_id = str(
                item.get("object_id") or item.get("target_id") or ""
            )
            predicate = str(
                item.get("predicate")
                or item.get("relation_expression")
                or item.get("relation")
                or ""
            ).strip()
            evidence = str(
                item.get("evidence") or item.get("evidence_text") or ""
            ).strip()
        subject = by_id.get(subject_id)
        object_ = by_id.get(object_id)
        if not reason and (subject is None or object_ is None):
            reason = "unknown_endpoint_id"
        if not reason and subject_id == object_id:
            reason = "self_loop"
        if (
            not reason
            and str(subject.get("canonical_id") or "")
            and str(subject.get("canonical_id") or "")
            == str(object_.get("canonical_id") or "")
        ):
            reference_endpoints = [
                endpoint for endpoint in (subject, object_)
                if bool(endpoint.get("relation_reference_proxy"))
            ]
            if reference_endpoints and not any(
                _is_explicit_reflexive_reference(
                    str(endpoint.get("surface") or "")
                )
                for endpoint in reference_endpoints
            ):
                reason = "nonreflexive_reference_self_loop"
        original_predicate = ""
        if not reason:
            predicate, original_predicate = (
                _repair_resolved_reference_nominal_prefix(
                    predicate,
                    subject,
                    object_,
                    content=content,
                    occurrences_by_key=occurrences_by_key,
                )
            )
            reference_nominal_prefix_repairs += int(bool(original_predicate))
        if not reason and (not predicate or len(predicate) > 160):
            reason = "invalid_predicate"
        if not reason:
            reason = _open_predicate_quality_reason(predicate)
        near = min(
            int((subject or {}).get("mention_start") or 0),
            int((object_ or {}).get("mention_start") or 0),
        )
        evidence_span = (
            _locate_open_fact_evidence(content, evidence, near=near)
            if not reason else None
        )
        if not reason and evidence_span is None:
            reason = "evidence_not_exact_source"
        if not reason:
            if not inside(subject, evidence_span):
                repaired = realign_occurrence(subject, evidence_span)
                if repaired is not None:
                    subject_id, subject = repaired
                    repaired_endpoint_occurrences += 1
            if not inside(object_, evidence_span):
                repaired = realign_occurrence(object_, evidence_span)
                if repaired is not None:
                    object_id, object_ = repaired
                    repaired_endpoint_occurrences += 1

            # A copied evidence fragment may omit one endpoint even though both
            # selected occurrences live in the same immutable parser scope.
            # Expand only to that single source scope; never bridge unrelated
            # sentences, rows or sections to make a model answer pass.
            if not (inside(subject, evidence_span) and inside(object_, evidence_span)):
                endpoint_scopes = [
                    (start, end)
                    for start, end, _text in relation_scopes
                    if inside(subject, (start, end))
                    and inside(object_, (start, end))
                ]
                if len(endpoint_scopes) == 1:
                    evidence_span = endpoint_scopes[0]

            # A model can copy an entire semicolon/newline-delimited sentence
            # even when the asserted relation occupies one parser-proved
            # atomic scope.  If that unique scope contains both exact endpoint
            # occurrences and the copied source predicate, narrow the evidence
            # before atomic-scope validation.  This never manufactures text or
            # bridges scopes; ambiguous and paraphrased predicates retain the
            # original fail-closed evidence.
            endpoint_predicate_scopes = [
                (
                    start + len(scope_text) - len(scope_text.lstrip()),
                    end - len(scope_text) + len(scope_text.rstrip()),
                )
                for start, end, scope_text in relation_scopes
                if inside(subject, (start, end))
                and inside(object_, (start, end))
                and evidence_span[0] <= start < end <= evidence_span[1]
                and _atomic_surface_is_source_copied(
                    scope_text, predicate, allow_contained=True,
                )
            ]
            if (
                len(endpoint_predicate_scopes) == 1
                and endpoint_predicate_scopes[0] != evidence_span
            ):
                evidence_span = endpoint_predicate_scopes[0]
                narrowed_evidence_scopes += 1

            subject_surface = str(subject.get("surface") or "")
            object_surface = str(object_.get("surface") or "")
            evidence_text = content[evidence_span[0]:evidence_span[1]]
            endpoint_spans = (
                (
                    int(subject.get("mention_start") or 0),
                    int(subject.get("mention_end") or 0),
                ),
                (
                    int(object_.get("mention_start") or 0),
                    int(object_.get("mention_end") or 0),
                ),
            )
            if (
                max(endpoint_spans[0][0], endpoint_spans[1][0])
                < min(endpoint_spans[0][1], endpoint_spans[1][1])
            ):
                # Overlapping source slices cannot be two independent
                # proposition arguments.  Lexically nested entities remain
                # valid graph nodes, but a model cannot invent a relation
                # between them merely from the containing noun phrase.
                reason = "overlapping_endpoint_occurrences"
            if not reason and not all(
                evidence_span[0] <= start < end <= evidence_span[1]
                for start, end in endpoint_spans
            ):
                reason = "evidence_excludes_endpoint_occurrence"
        if reason:
            if not allow_partial:
                raise RelationClassificationError(
                    f"grounded open fact item invalid: {reason}"
                )
            dropped_items.append({"item_index": item_index, "reason": reason})
            if isinstance(item, Mapping):
                repair_item = {
                    "item_index": item_index,
                    "reason": reason,
                    "attempt": {
                        "subject_id": subject_id,
                        "predicate": predicate,
                        "object_id": object_id,
                        "evidence": evidence,
                    },
                }
                known_argument_occurrence_keys = [
                    str(endpoint.get("occurrence_key") or "")
                    for endpoint in (subject, object_)
                    if isinstance(endpoint, Mapping)
                    and str(endpoint.get("occurrence_key") or "")
                ]
                if (
                    len(known_argument_occurrence_keys) == 2
                    and known_argument_occurrence_keys[0]
                    != known_argument_occurrence_keys[1]
                ):
                    repair_item["known_argument_occurrence_keys"] = (
                        known_argument_occurrence_keys
                    )
                repair_items.append(repair_item)
            continue
        subject_key = str(subject.get("occurrence_key") or "")
        object_key = str(object_.get("occurrence_key") or "")
        dedup = (
            subject_key, _normalized_atomic_surface(predicate), object_key,
            evidence_span[0], evidence_span[1],
        )
        if dedup in seen:
            continue
        seen.add(dedup)
        frame = {
            "relation": predicate,
            "arguments": [subject_surface, object_surface],
            "roles": ["subject", "object"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": [subject_surface, object_surface],
            # A directly stated deontic or negative proposition is still an
            # asserted source claim. Its open predicate retains modality or
            # polarity semantics for schema mapping and final verification.
            "assertion_status": "ASSERTED",
            "argument_occurrence_keys": [subject_key, object_key],
            "evidence_span": [evidence_span[0], evidence_span[1]],
            "evidence_text": content[evidence_span[0]:evidence_span[1]],
            "source_relation_grounding": (
                "EXACT_SOURCE_EXPRESSION"
                if _source_relation_expression_is_grounded(
                    content[evidence_span[0]:evidence_span[1]], predicate,
                )
                else "MODEL_PARAPHRASE"
            ),
            **(
                {
                    "source_relation_original_expression": (
                        original_predicate
                    ),
                    "source_relation_repair": (
                        "resolved_reference_nominal_prefix"
                    ),
                }
                if original_predicate else {}
            ),
        }
        reference_endpoints = [
            {
                "role": role,
                "occurrence_key": str(
                    endpoint.get("occurrence_key") or ""
                ),
                "antecedent_occurrence_key": str(
                    endpoint.get(
                        "reference_antecedent_occurrence_key"
                    ) or ""
                ),
                "reference_span": [
                    int(endpoint.get("mention_start") or 0),
                    int(endpoint.get("mention_end") or 0),
                ],
                "reference_text": str(endpoint.get("surface") or ""),
            }
            for role, endpoint in (
                ("subject", subject), ("object", object_),
            )
            if bool(endpoint.get("relation_reference_proxy"))
        ]
        if reference_endpoints:
            frame["resolved_reference_endpoints"] = reference_endpoints
        frames.append(frame)
    used_occurrence_keys = {
        str(value)
        for frame in frames
        for value in frame.get("argument_occurrence_keys") or ()
    }
    reference_proxies = [
        item for item in reference_proxies
        if str(item.get("occurrence_key") or "") in used_occurrence_keys
    ]
    return {
        "coverage_complete": not dropped_items and not any(
            str(item.get("reason") or "") != "unused_reference"
            for item in dropped_references
        ),
        "frames": frames,
        "dropped_items": dropped_items,
        "reference_proxies": reference_proxies,
        "dropped_references": dropped_references,
        "repair_items": repair_items,
        "repaired_endpoint_occurrences": repaired_endpoint_occurrences,
        "narrowed_evidence_scopes": narrowed_evidence_scopes,
        "reference_nominal_prefix_repairs": (
            reference_nominal_prefix_repairs
        ),
        "parser_version": GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION,
    }


def augment_grounded_open_fact_inventory(
    inventory: Mapping[str, Any],
    *,
    content: str,
    occurrences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add syntax-licensed table, label and multi-role open frames.

    These are still predicate-open candidates.  The routine never chooses a
    logical Schema predicate: table column text, identifier layout and an
    explicit role connector merely recover endpoint pairs that a generative
    inventory commonly omits.  Every generated frame retains exact occurrence
    IDs and contiguous source evidence and therefore follows the ordinary
    Schema-binding, validation and independent-verification path.
    """

    ordered_occurrences = sorted(
        (
            dict(item) for item in occurrences
            if str(item.get("occurrence_key") or "")
            and str(item.get("surface") or "")
            and 0 <= int(item.get("mention_start") or 0)
            < int(item.get("mention_end") or 0) <= len(content)
            and content[
                int(item.get("mention_start") or 0):
                int(item.get("mention_end") or 0)
            ] == str(item.get("surface") or "")
        ),
        key=lambda item: (
            int(item["mention_start"]), int(item["mention_end"]),
            str(item["occurrence_key"]),
        ),
    )
    # A resolved possessive determiner is a discourse proxy for the possessor,
    # not an independent noun-phrase argument at its surface position. Keep it
    # available for the dedicated possessor->possessed projection below, but
    # never let it displace the following noun in table, hierarchy, adjacency,
    # or ordered actor/theme/complement parsing.
    structural_occurrences = [
        item for item in ordered_occurrences
        if not (
            bool(item.get("relation_reference_proxy"))
            and str(item.get("reference_role") or "")
            == "possessive_determiner"
        )
    ]
    structural_occurrence_keys = {
        str(item["occurrence_key"]) for item in structural_occurrences
    }
    by_key = {
        str(item["occurrence_key"]): item for item in ordered_occurrences
    }
    frames = [
        dict(item) for item in inventory.get("frames") or ()
        if isinstance(item, Mapping)
    ]
    generated: list[dict[str, Any]] = []
    reference_nominal_prefix_repairs = 0

    def append_frame(
        subject: Mapping[str, Any],
        relation: str,
        object_: Mapping[str, Any],
        evidence_span: tuple[int, int],
        *,
        roles: tuple[str, str],
        origin: str,
        role_label: str = "",
        strict_structure: bool = True,
        structure_contract: Mapping[str, Any] | None = None,
        parallel_open_carrier_policy: str = "",
    ) -> None:
        nonlocal reference_nominal_prefix_repairs
        subject_key = str(subject.get("occurrence_key") or "")
        object_key = str(object_.get("occurrence_key") or "")
        relation, original_relation = (
            _repair_resolved_reference_nominal_prefix(
                relation,
                subject,
                object_,
                content=content,
                occurrences_by_key=by_key,
            )
        )
        reference_nominal_prefix_repairs += int(bool(original_relation))
        start, end = evidence_span
        if (
            not subject_key or not object_key or subject_key == object_key
            or str(subject.get("canonical_id") or "")
            and str(subject.get("canonical_id") or "")
            == str(object_.get("canonical_id") or "")
            or not str(relation or "").strip()
            or not (0 <= start < end <= len(content))
            or not (
                start <= int(subject["mention_start"])
                < int(subject["mention_end"]) <= end
                and start <= int(object_["mention_start"])
                < int(object_["mention_end"]) <= end
            )
        ):
            return
        generated_frame = {
            "relation": str(relation).strip()[:160],
            "arguments": [
                str(subject.get("surface") or ""),
                str(object_.get("surface") or ""),
            ],
            "roles": list(roles),
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": [
                str(subject.get("surface") or ""),
                str(object_.get("surface") or ""),
            ],
            "assertion_status": "ASSERTED",
            "argument_occurrence_keys": [subject_key, object_key],
            "evidence_span": [start, end],
            "evidence_text": content[start:end],
            "source_relation_grounding": (
                "PARSER_STRUCTURE_EXPRESSION"
                if strict_structure
                else "EXACT_SOURCE_EXPRESSION"
                if _source_relation_expression_is_grounded(
                    content[start:end], relation,
                )
                else "PARSER_SYNTHETIC_EXPRESSION"
            ),
            "deterministic_source_origin": origin,
            **(
                {"deterministic_source_structure": origin}
                if strict_structure
                else {"deterministic_source_syntax": origin}
            ),
            **(
                {"structural_role_label": str(role_label).strip()}
                if str(role_label).strip() else {}
            ),
            **(
                {"deterministic_structure_contract": dict(structure_contract)}
                if structure_contract else {}
            ),
            **(
                {
                    "parallel_open_carrier_policy": str(
                        parallel_open_carrier_policy
                    ).strip()
                }
                if str(parallel_open_carrier_policy).strip() else {}
            ),
            **(
                {
                    "source_relation_original_expression": (
                        original_relation
                    ),
                    "source_relation_repair": (
                        "resolved_reference_nominal_prefix"
                    ),
                }
                if original_relation else {}
            ),
        }
        reference_endpoints = [
            {
                "role": role,
                "occurrence_key": str(
                    endpoint.get("occurrence_key") or ""
                ),
                "antecedent_occurrence_key": str(
                    endpoint.get(
                        "reference_antecedent_occurrence_key"
                    ) or ""
                ),
                "reference_span": [
                    int(endpoint.get("mention_start") or 0),
                    int(endpoint.get("mention_end") or 0),
                ],
                "reference_text": str(endpoint.get("surface") or ""),
            }
            for role, endpoint in (
                ("subject", subject), ("object", object_),
            )
            if bool(endpoint.get("relation_reference_proxy"))
        ]
        if reference_endpoints:
            generated_frame["resolved_reference_endpoints"] = (
                reference_endpoints
            )
        generated.append(generated_frame)

    def line_cells(start: int, end: int) -> list[tuple[int, int, str]]:
        body_end = end
        while body_end > start and content[body_end - 1] in "\r\n":
            body_end -= 1
        body = content[start:body_end]
        pipes = [index for index, char in enumerate(body) if char == "|"]
        if len(pipes) < 2:
            return []
        boundaries = list(pipes)
        if not body.startswith("|"):
            boundaries.insert(0, -1)
        if not body.endswith("|"):
            boundaries.append(len(body))
        cells: list[tuple[int, int, str]] = []
        for left, right in zip(boundaries, boundaries[1:]):
            cell_start = start + left + 1
            cell_end = start + right
            while cell_start < cell_end and content[cell_start].isspace():
                cell_start += 1
            while cell_end > cell_start and content[cell_end - 1].isspace():
                cell_end -= 1
            cells.append((cell_start, cell_end, content[cell_start:cell_end]))
        return cells

    def compact_identifier_surface(surface: str) -> bool:
        return bool(
            len(surface) <= 48
            and not re.search(r"\s", surface)
            and re.search(r"\d", surface)
            and re.fullmatch(r"[\w.:#§/()\[\]-]+", surface)
        )

    line_spans: list[tuple[int, int]] = []
    cursor = 0
    for line in content.splitlines(keepends=True):
        line_spans.append((cursor, cursor + len(line)))
        cursor += len(line)
    if cursor < len(content):
        line_spans.append((cursor, len(content)))

    # Preserve an explicit ``Head (A, B and C)`` inventory as deterministic
    # detail edges.  Parenthetical members are not direct children of the
    # surrounding list root, but dropping them entirely loses useful local
    # structure and makes recall depend on a model understanding every list
    # idiom.  The contract below is syntax-only: one adjacent head, at least
    # two exact entity occurrences inside one matched parenthesis pair, and a
    # connector-only chain with an unambiguous conjunction/list separator.
    bracket_closers = {"(": ")", "（": "）"}
    for line_start, line_end in line_spans:
        stack: list[tuple[str, int]] = []
        bracket_pairs: list[tuple[int, int]] = []
        for position in range(line_start, line_end):
            character = content[position]
            if character in bracket_closers:
                stack.append((character, position))
                continue
            if not stack or character != bracket_closers[stack[-1][0]]:
                continue
            _opening, opening_index = stack.pop()
            bracket_pairs.append((opening_index, position))
        for opening_index, closing_index in sorted(bracket_pairs):
            raw_members = [
                item for item in structural_occurrences
                if opening_index < int(item["mention_start"])
                < int(item["mention_end"]) <= closing_index
            ]
            # Keep one principal endpoint at each exact start. Overlapping
            # aliases at the same position must not fabricate extra members.
            members_by_start: dict[int, Mapping[str, Any]] = {}
            for item in raw_members:
                start = int(item["mention_start"])
                current = members_by_start.get(start)
                if current is None or int(item["mention_end"]) > int(
                    current["mention_end"]
                ):
                    members_by_start[start] = item
            members = sorted(
                members_by_start.values(),
                key=lambda item: (
                    int(item["mention_start"]),
                    int(item["mention_end"]),
                    str(item.get("occurrence_key") or ""),
                ),
            )
            if len(members) < 2 or any(
                int(left["mention_end"]) > int(right["mention_start"])
                for left, right in zip(members, members[1:])
            ):
                continue
            head_candidates = [
                item for item in structural_occurrences
                if line_start <= int(item["mention_start"])
                < int(item["mention_end"]) <= opening_index
                and not (
                    str(item.get("source_role") or "") == "list_item"
                    and item.get("list_scope_head") is False
                )
            ]
            if not head_candidates:
                continue
            head = max(
                head_candidates,
                key=lambda item: (
                    int(item["mention_end"]),
                    -int(item["mention_start"]),
                    str(item.get("occurrence_key") or ""),
                ),
            )
            if content[int(head["mention_end"]):opening_index].strip():
                continue
            if (
                content[opening_index + 1:int(members[0]["mention_start"])].strip()
                or content[int(members[-1]["mention_end"]):closing_index].strip()
            ):
                continue
            explicit_connector = False
            connector_valid = True
            for left, right in zip(members, members[1:]):
                matched = _COORDINATED_RECOVERY_CONNECTOR.fullmatch(
                    content[
                        int(left["mention_end"]):
                        int(right["mention_start"])
                    ]
                )
                if matched is None:
                    connector_valid = False
                    break
                explicit_connector = explicit_connector or bool(
                    matched.group("word")
                    or matched.group("separator") in {"、", ";", "；"}
                )
            if not connector_valid or not explicit_connector:
                continue
            contract = {
                "kind": PARENTHETICAL_ENUMERATION_STRUCTURE_KIND,
                "head_occurrence_key": str(
                    head.get("occurrence_key") or ""
                ),
                "head_span": [
                    int(head["mention_start"]),
                    int(head["mention_end"]),
                ],
                "member_occurrence_keys": [
                    str(item.get("occurrence_key") or "")
                    for item in members
                ],
                "member_spans": [
                    [int(item["mention_start"]), int(item["mention_end"])]
                    for item in members
                ],
                "member_surfaces": [
                    str(item.get("surface") or "") for item in members
                ],
                "bracket_span": [opening_index, closing_index + 1],
            }
            evidence_span = (
                int(head["mention_start"]), closing_index + 1,
            )
            for member in members:
                append_frame(
                    member,
                    PARENTHETICAL_DETAIL_RELATION,
                    head,
                    evidence_span,
                    roles=("parenthetical_detail", "parenthetical_head"),
                    origin=PARENTHETICAL_ENUMERATION_STRUCTURE_KIND,
                    structure_contract=contract,
                    parallel_open_carrier_policy=(
                        STRUCTURE_PARALLEL_OPEN_CARRIER_SUPPRESSION
                    ),
                )
    table_rows: list[tuple[int, int, list[tuple[int, int, str]]]] = []
    table_header_occurrence_keys: set[str] = set()

    def flush_table() -> None:
        nonlocal table_rows
        semantic = [
            row for row in table_rows
            if not row[2] or not all(
                not text or bool(re.fullmatch(r":?-{3,}:?", text))
                for _start, _end, text in row[2]
            )
        ]
        table_rows = []
        if len(semantic) < 2:
            return
        header_start, _header_end, headers = semantic[0]
        width = len(headers)
        if width < 2:
            return

        def normalized_cell(value: Any) -> str:
            return re.sub(
                r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", "",
                unicodedata.normalize("NFKC", str(value or "")).casefold(),
            )

        def numeric_measurement(value: Any) -> bool:
            text = unicodedata.normalize(
                "NFKC", str(value or ""),
            ).strip().replace(" ", "")
            text = text.strip("*`_")
            return bool(re.fullmatch(
                r"[<>≤≥~≈]?[+-]?\d+(?:\.\d+)?"
                r"(?:[±+-]\d+(?:\.\d+)?)?%?"
                r"(?:\([^)]*\))?",
                text,
            ))

        # Markdown exported from PDF/DOCX commonly represents a grouped table
        # header as two consecutive rows (for example a metric family followed
        # by C1/C2/C3 subcolumns).  Treating the second row as data fabricates
        # entities and relations among header labels.  A continuation is
        # admitted only when its row-key cell is blank or repeats the primary
        # header and every populated subcell remains a non-numeric label.
        header_rows = [semantic[0]]
        primary_key = normalized_cell(headers[0][2])
        for candidate in semantic[1:4]:
            _row_start, _row_end, cells = candidate
            if len(cells) != width:
                break
            candidate_key = normalized_cell(cells[0][2])
            sublabels = [
                str(cell[2] or "").strip() for cell in cells[1:]
                if str(cell[2] or "").strip()
            ]
            if (
                candidate_key not in {"", primary_key}
                or not sublabels
                or any(numeric_measurement(value) for value in sublabels)
            ):
                break
            header_rows.append(candidate)

        combined_headers: list[str] = []
        for column_index in range(width):
            parts: list[str] = []
            normalized_parts: set[str] = set()
            for _row_start, _row_end, cells in header_rows:
                value = str(cells[column_index][2] or "").strip()
                normalized = normalized_cell(value)
                if not value or not normalized or normalized in normalized_parts:
                    continue
                normalized_parts.add(normalized)
                parts.append(value)
            combined_headers.append(" / ".join(parts))

        for _row_start, _row_end, header_cells in header_rows:
            for cell_start, cell_end, _header_text in header_cells:
                table_header_occurrence_keys.update(
                    str(item.get("occurrence_key") or "")
                    for item in structural_occurrences
                    if cell_start <= int(item["mention_start"])
                    < int(item["mention_end"]) <= cell_end
                    and str(item.get("occurrence_key") or "")
                )
        for row_start, row_end, cells in semantic[len(header_rows):]:
            if len(cells) != width:
                continue
            cell_occurrences = [[
                item for item in structural_occurrences
                if cell_start <= int(item["mention_start"])
                < int(item["mention_end"]) <= cell_end
            ] for cell_start, cell_end, _text in cells]
            key_index = next((
                index for index, values in enumerate(cell_occurrences)
                if values
            ), None)
            if key_index is None:
                continue
            for subject in cell_occurrences[key_index]:
                for column_index, objects in enumerate(cell_occurrences):
                    if column_index == key_index or not objects:
                        continue
                    header = combined_headers[column_index]
                    if not header:
                        continue
                    for object_ in objects:
                        append_frame(
                            subject,
                            f"column role: {header}",
                            object_,
                            (header_start, row_end),
                            roles=("row_key", "column_value"),
                            origin="table_header_assignment",
                            role_label=header,
                        )

    for start, end in line_spans:
        cells = line_cells(start, end)
        if cells:
            table_rows.append((start, end, cells))
        else:
            flush_table()
    flush_table()

    # Header cells are parser-owned role labels, not row entities. A
    # generative inventory can otherwise hallucinate a relation among labels
    # such as ``Purpose -> Data set``. Keep those mentions available for
    # provenance/UI inspection, but do not let their exact header occurrences
    # become semantic relation endpoints. Data-row frames generated above are
    # unaffected because their endpoints come only from rows after the header.
    frames_before_header_filter = len(frames)
    if table_header_occurrence_keys:
        frames = [
            frame for frame in frames
            if not table_header_occurrence_keys.intersection(
                str(value or "")
                for value in frame.get("argument_occurrence_keys") or ()
            )
        ]
    table_header_frames_dropped = frames_before_header_filter - len(frames)

    # A compact identifier at the start of a source line explicitly binds the
    # following principal topic.  The identifier shape is syntax-only; no
    # domain vocabulary or logical predicate name participates in the rule.
    label_topic_bridge_full = re.compile(
        r"^[\s:：.．()（）\[\]{}\-‐‑‒–—]*"
        r"(?:(?:a|an|the)\s+)?$",
        re.IGNORECASE,
    )
    for line_start, line_end in line_spans:
        values = [
            item for item in structural_occurrences
            if line_start <= int(item["mention_start"])
            < int(item["mention_end"]) <= line_end
        ]
        if len(values) < 2:
            continue
        label = values[0]
        prefix = content[line_start:int(label["mention_start"])]
        surface = str(label.get("surface") or "")
        compact_identifier = compact_identifier_surface(surface)
        topic_bridge = content[
            int(label["mention_end"]):int(values[1]["mention_start"])
        ]
        if (
            prefix.strip()
            or not compact_identifier
            or label_topic_bridge_full.fullmatch(topic_bridge) is None
        ):
            continue
        append_frame(
            label,
            "label binds topic",
            values[1],
            (int(label["mention_start"]), line_end),
            roles=("source_label", "bound_topic"),
            origin="line_identifier_binding",
        )

    # Prefer the source mapper's exact list parent scope.  It combines marker
    # style and indentation and therefore distinguishes direct children from
    # both siblings and grandchildren.  Reconstructing hierarchy by scanning
    # all following bullet lines used to turn a flat list into a transitive
    # chain of false ``part_of`` candidates.
    occurrences_by_scope: dict[str, list[dict[str, Any]]] = {}
    for item in structural_occurrences:
        scope_id = str(item.get("scope_id") or "")
        if scope_id:
            occurrences_by_scope.setdefault(scope_id, []).append(item)
    for child in structural_occurrences:
        parent_scope_id = str(child.get("list_parent_scope_id") or "")
        if (
            not parent_scope_id
            or child.get("list_scope_head") is False
        ):
            continue
        parent_values = [
            item for item in occurrences_by_scope.get(parent_scope_id) or []
            if item.get("list_scope_head") is not False
        ]
        if not parent_values:
            continue
        parent = min(
            parent_values,
            key=lambda item: (
                int(item.get("mention_start") or 0),
                -int(item.get("mention_end") or 0),
                str(item.get("occurrence_key") or ""),
            ),
        )
        evidence_start = min(
            int(parent.get("scope_start") or parent["mention_start"]),
            int(child.get("scope_start") or child["mention_start"]),
        )
        evidence_end = max(
            int(parent.get("scope_end") or parent["mention_end"]),
            int(child.get("scope_end") or child["mention_end"]),
        )
        append_frame(
            child,
            "structural child of",
            parent,
            (evidence_start, evidence_end),
            roles=("structural_child", "structural_parent"),
            origin="list_parent_child",
        )

    # An unmarked standalone label followed by explicit list items is a
    # second syntax-only form (``Components:\n- A``).  Marked parent items are
    # deliberately excluded here; their direct links were handled above from
    # the parser scope contract.  Only peers at the first child's indentation
    # are attached to the label, while nested descendants remain attached to
    # their actual list parent.
    list_marker_pattern = re.compile(
        r"[ \t]*(?:(?:[-*+•▪◦–—♦◆◇]\s*)|(?:\d+[.)]\s*))"
    )
    for parent_index, (parent_start, parent_end) in enumerate(line_spans):
        parent_values = [
            item for item in structural_occurrences
            if parent_start <= int(item["mention_start"])
            < int(item["mention_end"]) <= parent_end
        ]
        if len(parent_values) != 1:
            continue
        parent = parent_values[0]
        parent_prefix = content[
            parent_start:int(parent["mention_start"])
        ]
        parent_suffix = content[
            int(parent["mention_end"]):parent_end
        ].strip()
        parent_is_explicit_label = parent_suffix in {":", "："}
        parent_surface = str(parent.get("surface") or "")
        if (
            parent_suffix not in {"", ":", "："}
            or parent_prefix.strip()
            or len(content[parent_start:parent_end].strip()) > 180
            # A compact identifier/page marker followed by another identifier
            # is a common flattened diagram or index sequence, not a label
            # governing the next line. Explicit punctuation can still make it
            # a real field label (for example ``R-01:``).
            or (
                not parent_is_explicit_label
                and compact_identifier_surface(parent_surface)
            )
        ):
            continue
        child_index = parent_index + 1
        started_children = False
        direct_child_indent: int | None = None
        while child_index < len(line_spans):
            child_start, child_end = line_spans[child_index]
            child_text = content[child_start:child_end]
            if not child_text.strip():
                if started_children:
                    break
                child_index += 1
                continue
            child_values = [
                item for item in structural_occurrences
                if child_start <= int(item["mention_start"])
                < int(item["mention_end"]) <= child_end
            ]
            if not child_values:
                break
            child = child_values[0]
            child_prefix = content[
                child_start:int(child["mention_start"])
            ]
            marker_match = list_marker_pattern.fullmatch(child_prefix)
            is_list_child = marker_match is not None
            is_identifier_child = (
                not child_prefix.strip()
                and compact_identifier_surface(str(child.get("surface") or ""))
            )
            if not (is_list_child or is_identifier_child):
                break
            if is_identifier_child and not started_children:
                binding_gap = content[
                    int(parent["mention_end"]):int(child["mention_start"])
                ]
                blank_line_boundary = bool(re.search(
                    r"\r?\n[ \t]*\r?\n", binding_gap,
                ))
                if not (parent_is_explicit_label or blank_line_boundary):
                    break
            if is_list_child:
                child_indent = len(child_prefix) - len(
                    child_prefix.lstrip(" \t")
                )
                if direct_child_indent is None:
                    direct_child_indent = child_indent
                elif child_indent > direct_child_indent:
                    child_index += 1
                    continue
                elif child_indent < direct_child_indent:
                    break
            append_frame(
                child,
                "structural child of",
                parent,
                (parent_start, child_end),
                roles=("structural_child", "structural_parent"),
                origin=(
                    "list_parent_child"
                    if is_list_child else "section_label_parent_child"
                ),
            )
            started_children = True
            child_index += 1

    # Expand an already grounded argument across an explicit coordinated
    # phrase.  A comma alone is insufficient: the complete local component
    # must contain a conjunction (or an unambiguous list separator).  This
    # recovers coordinated subjects/objects without ever asserting a relation
    # between the peer members themselves.
    coordination_connector = re.compile(
        r"^\s*(?:(?:[,，]\s*)?(?P<word>and|or|nor|plus|as\s+well\s+as|"
        r"和|及|与|以及|或|或者|并|並|ならびに|及び|または|또는|및)\s*|"
        r"(?P<separator>[,，、])\s*)"
        r"(?:(?:the|a|an|le|la|les|un|une|el|los|las|der|die|das)\s*)?$",
        re.IGNORECASE,
    )
    scope_occurrences: list[
        tuple[int, int, list[dict[str, Any]]]
    ] = []
    reference_scope_occurrences: list[
        tuple[int, int, list[dict[str, Any]]]
    ] = []
    coordination_components_by_scope: list[
        tuple[int, int, list[list[dict[str, Any]]]]
    ] = []
    for scope_start, scope_end, _scope_text in split_relation_scopes(content):
        reference_values = [
            item for item in ordered_occurrences
            if scope_start <= int(item["mention_start"])
            < int(item["mention_end"]) <= scope_end
        ]
        if len(reference_values) >= 2:
            reference_scope_occurrences.append((
                scope_start, scope_end, reference_values,
            ))
        values = [
            item for item in reference_values
            if str(item["occurrence_key"]) in structural_occurrence_keys
        ]
        if len(values) >= 2:
            scope_occurrences.append((scope_start, scope_end, values))
    for scope_start, scope_end, values in scope_occurrences:
        edges: list[tuple[int, bool]] = []
        for index in range(len(values) - 1):
            between = content[
                int(values[index]["mention_end"]):
                int(values[index + 1]["mention_start"])
            ]
            matched = coordination_connector.fullmatch(between)
            if matched is not None:
                edges.append((
                    index,
                    bool(matched.group("word") or matched.group("separator") == "、"),
                ))
        if not edges:
            continue
        edge_by_index = {index: explicit for index, explicit in edges}
        components: list[tuple[list[dict[str, Any]], bool]] = []
        cursor_index = 0
        while cursor_index < len(values):
            component = [values[cursor_index]]
            explicit = False
            while cursor_index in edge_by_index:
                edge_explicit = edge_by_index[cursor_index]
                # A weak comma may introduce members before a final explicit
                # conjunction (``A, B, and C``). Once an explicit conjunction
                # has closed that list, a following ordinary comma is a clause
                # boundary (``A and B, C acts ...``), not permission to project
                # A/B's predicate onto C.
                if explicit and not edge_explicit:
                    break
                explicit = explicit or edge_explicit
                cursor_index += 1
                component.append(values[cursor_index])
            if len(component) > 1 and explicit:
                components.append((component, explicit))
            cursor_index += 1
        if components:
            coordination_components_by_scope.append((
                scope_start,
                scope_end,
                [component for component, _explicit in components],
            ))
        for frame in list(frames):
            keys = list(frame.get("argument_occurrence_keys") or ())
            if len(keys) != 2 or not all(str(key) in by_key for key in keys):
                continue
            relation = str(frame.get("relation") or "")
            subject, object_ = by_key[str(keys[0])], by_key[str(keys[1])]
            raw_seed_span = list(frame.get("evidence_span") or ())
            try:
                seed_start, seed_end = map(int, raw_seed_span)
            except (TypeError, ValueError):
                seed_start = min(
                    int(subject["mention_start"]),
                    int(object_["mention_start"]),
                )
                seed_end = max(
                    int(subject["mention_end"]),
                    int(object_["mention_end"]),
                )
            if not (0 <= seed_start < seed_end <= len(content)):
                continue
            coordination_evidence_span = (
                min(scope_start, seed_start),
                max(scope_end, seed_end),
            )
            for component, _explicit in components:
                component_keys = {
                    str(item["occurrence_key"]) for item in component
                }
                # The non-coordinated endpoint may be an explicit heading or
                # a resolved antecedent in the preceding source unit. Requiring
                # both endpoints inside the list sentence used to expand only
                # its first member (for example ``These processes are A, B,
                # C``). The exact seed frame already binds that outside anchor
                # to one occurrence in this component; syntax expands only
                # that bound role to its directly coordinated peers. Every
                # emitted edge still receives ordinary Schema binding and
                # independent verification.
                if str(subject["occurrence_key"]) in component_keys:
                    for peer in component:
                        if str(peer["occurrence_key"]) == str(
                            subject["occurrence_key"]
                        ):
                            continue
                        append_frame(
                            peer, relation, object_, coordination_evidence_span,
                            roles=("coordinated_subject", "object"),
                            origin="coordinated_argument",
                            strict_structure=False,
                        )
                    if str(object_["occurrence_key"]) not in component_keys:
                        for left_peer, right_peer in combinations(component, 2):
                            append_frame(
                                left_peer,
                                f"collective participants in {relation}",
                                right_peer,
                                coordination_evidence_span,
                                roles=(
                                    "collective_participant",
                                    "collective_participant",
                                ),
                                origin=(
                                    "coordinated_collective_relation_candidate"
                                ),
                                strict_structure=False,
                            )
                if str(object_["occurrence_key"]) in component_keys:
                    for peer in component:
                        if str(peer["occurrence_key"]) == str(
                            object_["occurrence_key"]
                        ):
                            continue
                        append_frame(
                            subject, relation, peer, coordination_evidence_span,
                            roles=("subject", "coordinated_object"),
                            origin="coordinated_argument",
                            strict_structure=False,
                        )

    # A resolved possessive determiner contributes an explicit source role to
    # its following noun phrase.  The resolver decides only identity; this
    # parser step records the generic possessor/possessed topology and leaves
    # ownership, part-whole, authorship, affiliation, and every other logical
    # interpretation to Schema binding plus independent verification.
    for scope_start, scope_end, values in reference_scope_occurrences:
        for possessor in values:
            if (
                not bool(possessor.get("relation_reference_proxy"))
                or str(possessor.get("reference_role") or "")
                != "possessive_determiner"
            ):
                continue
            possessed_key = str(
                possessor.get("reference_other_endpoint_occurrence_key") or ""
            )
            possessed = by_key.get(possessed_key)
            if possessed is None or not (
                scope_start <= int(possessor["mention_start"])
                < int(possessor["mention_end"]) <= scope_end
                and scope_start <= int(possessed["mention_start"])
                < int(possessed["mention_end"]) <= scope_end
            ):
                continue
            append_frame(
                possessor,
                "possessive association",
                possessed,
                (scope_start, scope_end),
                roles=("possessor", "possessed"),
                origin="possessive_reference_construction",
                strict_structure=False,
            )

    # A colon-headed indented block is a parser-level parent/child construct.
    # It covers code conditions as well as other indentation grammars; the
    # logical predicate is still selected and independently verified later.
    for line_index, (header_start, header_end) in enumerate(line_spans[:-1]):
        header_text = content[header_start:header_end].rstrip("\r\n")
        if not header_text.rstrip().endswith((":", "：")):
            continue
        header_indent = len(header_text) - len(header_text.lstrip(" \t"))
        header_values = [
            item for item in structural_occurrences
            if header_start <= int(item["mention_start"])
            < int(item["mention_end"]) <= header_end
        ]
        if len(header_values) != 1:
            continue
        child_index = line_index + 1
        while child_index < len(line_spans):
            child_start, child_end = line_spans[child_index]
            child_text = content[child_start:child_end].rstrip("\r\n")
            if not child_text.strip():
                child_index += 1
                continue
            child_indent = len(child_text) - len(child_text.lstrip(" \t"))
            if child_indent <= header_indent:
                break
            child_values = [
                item for item in structural_occurrences
                if child_start <= int(item["mention_start"])
                < int(item["mention_end"]) <= child_end
            ]
            if child_values:
                append_frame(
                    header_values[0],
                    "indented header governs child",
                    child_values[0],
                    (header_start, child_end),
                    roles=("indented_header", "indented_child"),
                    origin="indented_header_child",
                    strict_structure=False,
                )
            break

    # Recover a complete three-role construction from one grounded actor/theme
    # edge plus an explicit prepositional connector to the next frozen entity.
    # The rule copies the model's open relation root and emits candidates only;
    # Schema binding and ordered verification still decide publication.
    role_connector_full = re.compile(
        r"^[\s,;:()\[\]-]*(?P<connector>(?:out\s+of|away\s+from|"
        r"to|from|into|onto|toward|towards|"
        r"through|via|with|at|in|on|under|over|beside|near|within|inside|"
        r"outside|against|of|for|by)\b|(?:向|至|从|由|到|经由|通过|在|于|对|给|"
        r"へ|から|に|で|에게|에서|로))"
        r"[\s]*(?:(?:a|an|the|my|your|his|her|its|our|their|"
        r"le|la|les|un|une|el|los|las|der|die|das)\s*)?$",
        re.IGNORECASE,
    )
    # Some languages place a recipient/source/location before the lexical
    # predicate (for example ``actor -> recipient -> verb -> theme``).  This
    # is the same n-ary role topology as the postposed connector form below;
    # only surface order differs.  Modal carriers may precede the connector,
    # but no arbitrary lexical material is admitted here.
    preposed_role_connector_full = re.compile(
        r"^[\s,;:()\[\]-]*(?:(?:shall|must|should|will|would|can|could|"
        r"may|might|has\s+to|have\s+to|is\s+required\s+to|"
        r"are\s+required\s+to)\s+|(?:应当|應當|应该|應該|必须|必須|须|須|"
        r"需|需要|应|應)\s*)?"
        r"(?P<connector>to|from|into|onto|toward|towards|through|via|"
        r"at|in|on|under|over|within|inside|outside|against|"
        r"向|至|从|從|到|经由|經由|通过|通過|在|于|於|对|對|给|給|"
        r"へ|から|に|で|에게|에서|로)"
        r"[\s]*(?:(?:a|an|the|my|your|his|her|its|our|their|"
        r"le|la|les|un|une|el|los|las|der|die|das)\s*)?$",
        re.IGNORECASE,
    )
    passive_agent_connector_full = re.compile(
        r"^[\s,;:()\[\]-]*(?P<connector>由|被|由於|由于)\s*$",
        re.IGNORECASE,
    )
    endpoint_role_connector_tail = re.compile(
        r"(?P<connector>out\s+of|away\s+from|to|from|into|onto|toward|"
        r"towards|through|via|with|at|in|on|under|over|beside|near|within|"
        r"inside|outside|against|of|for|by|向|至|从|由|到|经由|通过|在|于|"
        r"对|给|へ|から|に|で|에게|에서|로)"
        r"\s*(?:(?:a|an|the|my|your|his|her|its|our|their|"
        r"le|la|les|un|une|el|los|las|der|die|das)\s*)?$",
        re.IGNORECASE,
    )
    relation_auxiliaries = {
        "am", "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had", "can", "could",
        "may", "might", "must", "shall", "should", "will", "would",
        "ought", "seems", "seem", "应", "应当", "必须", "须", "需", "需要",
    }
    relation_function_words = relation_auxiliaries | {
        "a", "an", "the", "this", "that", "these", "those", "my",
        "your", "his", "her", "its", "our", "their", "to", "from",
        "into", "onto", "toward", "towards", "through", "via", "with",
        "at", "in", "on", "under", "over", "beside", "near", "within",
        "inside", "outside", "against", "of", "for", "by", "out", "away",
        "and", "or", "nor", "but", "if", "because", "although", "though",
        "while", "when", "where", "which", "who", "whom", "whose", "before",
        "after", "merely", "just", "very", "not", "no", "nothing", "it",
        "he", "she", "they", "we", "you", "i", "me", "him", "them",
        # Passive/role particles are argument markers, not standalone
        # semantic actions.  A trailing lexical predicate is recovered below
        # with its semantic agent/theme orientation.
        "由", "被", "由於", "由于", "和", "与", "與", "及", "或",
        "并", "並", "且", "而", "将", "將", "把", "在", "于", "於",
        "向", "对", "對", "给", "給",
    }

    def lexical_relation_root(text: str) -> str:
        rendered_label = _rendered_edge_relation_label(text)
        if rendered_label:
            return rendered_label
        tokens = re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ]+|[\u3400-\u9fff]+|[\u3040-\u30ff]+|[\uac00-\ud7af]+",
            str(text or ""),
        )
        if not tokens:
            return ""
        cjk_function_only = re.compile(
            r"^(?:(?:并且|而且|随后|然后|同时|应当|应该|必须|需要|"
            r"和|与|與|及|或|并|並|且|而|又|再|将|將|把|须|需|应|"
            r"在|于|於|向|对|對|给|給|由|被))+$"
        )
        lexical_candidates = [
            (index, token)
            for index, token in enumerate(tokens)
            if token.casefold() not in relation_function_words
            and cjk_function_only.fullmatch(token) is None
        ]
        if not lexical_candidates:
            return ""
        nominal_determiners = {
            "a", "an", "the", "this", "that", "these", "those",
            "my", "your", "his", "her", "its", "our", "their",
            "le", "la", "les", "un", "une", "el", "los", "las",
            "der", "die", "das",
        }
        # In ``verb + determiner + modifier + ENTITY``, the modifier belongs
        # to the frozen object noun phrase and must not replace the governing
        # verb merely because it is the rightmost lexical token.  Keep a lone
        # verb before a trailing determiner (``submitted the ENTITY``), and
        # otherwise preserve the existing nearest-root behavior.
        for boundary_index in range(len(tokens) - 1, -1, -1):
            if tokens[boundary_index].casefold() not in nominal_determiners:
                continue
            before = [
                value for value in lexical_candidates
                if value[0] < boundary_index
            ]
            after = [
                value for value in lexical_candidates
                if value[0] > boundary_index
            ]
            if before and after:
                return before[-1][1]
        return lexical_candidates[-1][1]

    # Recover a predicate whose grammatical subject is separated from it by
    # an explicit attribute/modifier.  A common shape is
    # ``entity with attribute VERB complement``.  The intervening attribute
    # must not become the verb's subject merely because it is the closest
    # mention.  This is a syntax candidate only; Schema binding and the final
    # ordered verifier still decide whether the relation is publishable.
    modifier_connector_full = re.compile(
        r"^[\s,;:()\[\]-]*(?:with|having|bearing|featuring|including|"
        r"whose|带有|带着|携带着?|拿着|持着|具有|含有|包括|其中有|"
        r"を持つ|を備えた|가진|포함한)"
        r"[\s]*(?:(?:a|an|the|my|your|his|her|its|our|their)\s*)?"
        r"(?:[\wÀ-ÖØ-öø-ÿ-]+[\s]+){0,3}$",
        re.IGNORECASE,
    )
    modifier_interrupted_subjects: set[tuple[str, str]] = set()
    for scope_start, scope_end, values in scope_occurrences:
        for first, modifier, complement in zip(values, values[1:], values[2:]):
            modifier_text = content[
                int(first["mention_end"]):int(modifier["mention_start"])
            ]
            if modifier_connector_full.fullmatch(modifier_text) is None:
                continue
            relation_text = content[
                int(modifier["mention_end"]):int(complement["mention_start"])
            ]
            if (
                not relation_text.strip()
                or len(relation_text) > 160
                or any(marker in relation_text for marker in (";", "；", "\n", "|"))
                or not (
                    lexical_relation_root(relation_text)
                    or _source_relation_has_explicit_negation(relation_text)
                )
            ):
                continue
            relation = re.sub(r"\s+", " ", relation_text).strip(
                " \t\r,;:()[]-"
            )
            relation = re.sub(
                r"\s+(?:a|an|the|my|your|his|her|its|our|their)$",
                "",
                relation,
                flags=re.IGNORECASE,
            )
            if not relation:
                continue
            append_frame(
                first, relation, complement, (scope_start, scope_end),
                roles=("grammatical_subject", "role_complement"),
                origin="modifier_interrupted_predicate",
                strict_structure=False,
            )
            modifier_interrupted_subjects.add((
                str(modifier["occurrence_key"]),
                str(complement["occurrence_key"]),
            ))

    # Recover a theme-to-complement projection when a disposal construction
    # marks the theme but its actor is represented only by a pronoun/reference
    # (and therefore may not be a frozen entity occurrence).  The explicit
    # marker before the theme plus a connector-bearing predicate before the
    # complement licenses only this ordered binary role.  Predicate selection
    # and publication remain with Schema binding and the independent verifier.
    disposal_marker_tail = re.compile(r"(?:把|将|將)\s*$")
    disposal_connector_tail = re.compile(
        r"(?:给|給|向|至|到|入|进|進|在|于|於|从|從|由)\s*$"
    )
    disposal_projected_pairs: set[tuple[str, str]] = set()
    for scope_start, scope_end, values in scope_occurrences:
        for theme, complement in zip(values, values[1:]):
            marker_text = content[
                scope_start:int(theme["mention_start"])
            ]
            if disposal_marker_tail.search(marker_text) is None:
                continue
            relation_text = content[
                int(theme["mention_end"]):int(complement["mention_start"])
            ]
            if (
                not relation_text.strip()
                or len(relation_text) > 80
                or any(
                    marker in relation_text
                    for marker in (",", "，", ";", "；", "\n", "|")
                )
                or disposal_connector_tail.search(relation_text) is None
                or not lexical_relation_root(relation_text)
            ):
                continue
            relation = re.sub(r"\s+", " ", relation_text).strip(
                " \t\r,;:()[]-"
            )
            if not relation:
                continue
            append_frame(
                theme,
                relation,
                complement,
                (scope_start, scope_end),
                roles=("action_theme", "role_complement"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )
            disposal_projected_pairs.add((
                str(theme["occurrence_key"]),
                str(complement["occurrence_key"]),
            ))

    # Consecutive frozen mentions with lexical predicate material between
    # them form a bounded binary syntax candidate.  This deterministic pass
    # recovers nested clauses and local-reference subjects that an open model
    # occasionally omits (for example, a pronoun followed by ``saw X``).
    # Punctuation/table boundaries and function-word-only spans are rejected;
    # no predicate is selected here.
    for scope_start, scope_end, values in scope_occurrences:
        for pair_index, (first, second) in enumerate(
            zip(values, values[1:]),
        ):
            pair_key = (
                str(first["occurrence_key"]),
                str(second["occurrence_key"]),
            )
            if (
                pair_key in modifier_interrupted_subjects
                or pair_key in disposal_projected_pairs
            ):
                continue
            relation_text = content[
                int(first["mention_end"]):int(second["mention_start"])
            ]
            if (
                not relation_text.strip()
                or len(relation_text) > 160
                or any(
                    marker in relation_text
                    for marker in (",", "，", "、", ";", "；", "\n", "|")
                )
            ):
                continue
            if (
                pair_index > 0
                and re.match(r"^\s*[,，、]", relation_text) is not None
            ):
                # A predicate introduced after a clause delimiter inherits or
                # introduces a clause subject; the preceding complement is not
                # licensed as that subject merely because it is the nearest
                # frozen mention. The source model may still recover the actual
                # carried subject explicitly.
                continue
            passive_connector = passive_agent_connector_full.fullmatch(
                relation_text
            )
            if passive_connector is not None:
                # In a post-predicate passive construction the first endpoint
                # is the action theme and the second is the semantic agent:
                # ``theme 由 agent VERB``.  The role particle alone must never
                # become a graph predicate.  Recover the trailing lexical
                # action only when no later endpoint competes for its role.
                if pair_index + 2 >= len(values):
                    trailing_text = content[
                        int(second["mention_end"]):scope_end
                    ]
                    if (
                        trailing_text.strip()
                        and len(trailing_text) <= 160
                        and not any(
                            marker in trailing_text
                            for marker in (";", "；", "\n", "|")
                        )
                    ):
                        trailing_root = lexical_relation_root(trailing_text)
                        if trailing_root:
                            append_frame(
                                second,
                                trailing_root,
                                first,
                                (scope_start, scope_end),
                                roles=(
                                    "grammatical_subject", "action_theme",
                                ),
                                origin=(
                                    "passive_agent_postpredicate_construction"
                                ),
                                strict_structure=False,
                            )
                continue
            root = lexical_relation_root(relation_text)
            negated = _source_relation_has_explicit_negation(relation_text)
            if not root and not negated:
                continue
            # Leave the actor/theme edge to the richer three-role projection
            # below when the next exact mention is introduced by a role
            # connector.  Otherwise binary deduplication would keep the same
            # edge but discard its action_theme provenance.
            if pair_index + 2 < len(values):
                third = values[pair_index + 2]
                connector_text = content[
                    int(second["mention_end"]):int(third["mention_start"])
                ]
                if role_connector_full.fullmatch(connector_text) is not None:
                    continue
            append_frame(
                first,
                (
                    re.sub(r"\s+", " ", relation_text).strip(
                        " \t\r,;:()[]-"
                    )
                    if negated
                    else root
                ),
                second,
                (scope_start, scope_end),
                roles=("grammatical_subject", "direct_argument"),
                origin="intervening_predicate_construction",
                strict_structure=False,
            )

    # Recover the surface-order variant ``actor CONNECTOR complement VERB
    # theme``.  It is common in recipient-, source-, destination-, location-
    # and instrument-fronting constructions.  We emit the same three binary
    # role projections used for the postposed form: actor->theme,
    # theme->complement and actor->complement.  This stage chooses no logical
    # predicate; Schema binding and independent verification remain mandatory.
    for scope_start, scope_end, values in scope_occurrences:
        for actor, complement, theme in zip(
            values, values[1:], values[2:],
        ):
            connector_text = content[
                int(actor["mention_end"]):int(complement["mention_start"])
            ]
            connector = preposed_role_connector_full.fullmatch(
                connector_text
            )
            if connector is None:
                continue
            relation_text = content[
                int(complement["mention_end"]):int(theme["mention_start"])
            ]
            if (
                not relation_text.strip()
                or len(relation_text) > 160
                or any(
                    marker in relation_text
                    for marker in (";", "；", "\n", "|")
                )
            ):
                continue
            root = lexical_relation_root(relation_text)
            negated = _source_relation_has_explicit_negation(relation_text)
            if not root and not negated:
                continue
            relation_root = (
                re.sub(r"\s+", " ", relation_text).strip(
                    " \t\r,;:()[]-"
                )
                if negated else root
            )
            role_relation = (
                f"{relation_root} {connector.group('connector')}"
            )
            append_frame(
                actor,
                relation_root,
                theme,
                (scope_start, scope_end),
                roles=("grammatical_subject", "action_theme"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )
            append_frame(
                theme,
                role_relation,
                complement,
                (scope_start, scope_end),
                roles=("action_theme", "role_complement"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )
            append_frame(
                actor,
                role_relation,
                complement,
                (scope_start, scope_end),
                roles=("grammatical_subject", "role_complement"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )

    # The same construction can be recovered without a model seed when three
    # consecutive frozen occurrences expose both the predicate-bearing middle
    # text and an exact role connector. This removes model variance for common
    # "actor verb theme to/from/in destination" clauses while remaining a
    # candidate-only syntax projection.
    for scope_start, scope_end, scope_values in scope_occurrences:
        values = list(scope_values)
        if (
            len(values) >= 4
            and compact_identifier_surface(str(values[0].get("surface") or ""))
        ):
            values = values[1:]
        for first, second, third in zip(values, values[1:], values[2:]):
            connector_text = content[
                int(second["mention_end"]):int(third["mention_start"])
            ]
            connector = role_connector_full.fullmatch(connector_text)
            if connector is None:
                continue
            relation_text = content[
                int(first["mention_end"]):int(second["mention_start"])
            ]
            if len(relation_text) > 200 or any(
                marker in relation_text for marker in (";", "；", "\n")
            ):
                continue
            root = lexical_relation_root(relation_text)
            negated = _source_relation_has_explicit_negation(relation_text)
            if not root and not negated:
                continue
            relation_root = (
                re.sub(r"\s+", " ", relation_text).strip(
                    " \t\r,;:()[]-"
                )
                if negated
                else root
            )
            role_relation = (
                f"{relation_root} {connector.group('connector')}"
            )
            prior_connector = endpoint_role_connector_tail.search(
                relation_text
            )
            if prior_connector is not None:
                # ``actor VERB/PREP entity PREP third`` contains two adjacent
                # complements, not an actor/theme/complement chain. A surface
                # example is ``sitting by X on Y``: X is not the action theme
                # and cannot inherit the second preposition. Preserve only the
                # first directly licensed binary frame; attachment of the
                # third phrase remains a model/Schema adjudication question.
                append_frame(
                    first,
                    f"{relation_root} {prior_connector.group('connector')}",
                    second,
                    (scope_start, scope_end),
                    roles=("grammatical_subject", "role_complement"),
                    origin="ordered_three_role_construction",
                    strict_structure=False,
                )
                continue
            append_frame(
                first, relation_root, second, (scope_start, scope_end),
                roles=("grammatical_subject", "action_theme"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )
            append_frame(
                second, role_relation, third, (scope_start, scope_end),
                roles=("action_theme", "role_complement"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )
            append_frame(
                first, role_relation, third, (scope_start, scope_end),
                roles=("grammatical_subject", "role_complement"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )

    for frame in [*frames, *generated]:
        keys = list(frame.get("argument_occurrence_keys") or ())
        if len(keys) != 2 or not all(str(key) in by_key for key in keys):
            continue
        relation = str(frame.get("relation") or "").strip()
        relation_roots = re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ]+|[\u3400-\u9fff]+|[\u3040-\u30ff]+|[\uac00-\ud7af]+",
            relation,
        )
        if not relation_roots:
            continue
        root = lexical_relation_root(relation) or relation_roots[0]
        first, second = sorted(
            (by_key[str(keys[0])], by_key[str(keys[1])]),
            key=lambda item: int(item["mention_start"]),
        )
        between_endpoints = content[
            int(first["mention_end"]):int(second["mention_start"])
        ]
        if (
            root.casefold() not in between_endpoints.casefold()
            if re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", root)
            else root not in between_endpoints
        ):
            continue
        containing_scopes = [
            (start, end, values)
            for start, end, values in scope_occurrences
            if start <= int(first["mention_start"])
            and int(second["mention_end"]) <= end
        ]
        if len(containing_scopes) != 1:
            continue
        if compact_identifier_surface(str(first.get("surface") or "")):
            # A leading compact identifier labels the following proposition;
            # it is not its grammatical actor. The independent line-binding
            # rule above already preserves the label/topic relationship.
            continue
        scope_start, scope_end, values = containing_scopes[0]
        later_values = [
            item for item in values
            if int(item["mention_start"]) >= int(second["mention_end"])
        ]
        if not later_values:
            continue
        third = later_values[0]
        connector_text = content[
            int(second["mention_end"]):int(third["mention_start"])
        ]
        connector = role_connector_full.fullmatch(connector_text)
        if connector is None:
            continue
        relation_root = (
            relation
            if _source_relation_has_explicit_negation(relation)
            else root
        )
        role_relation = f"{relation_root} {connector.group('connector')}"
        prior_connector = endpoint_role_connector_tail.search(
            between_endpoints
        )
        if prior_connector is not None:
            append_frame(
                first, relation, second, (scope_start, scope_end),
                roles=("grammatical_subject", "role_complement"),
                origin="ordered_three_role_construction",
                strict_structure=False,
            )
            continue
        append_frame(
            first, relation, second, (scope_start, scope_end),
            roles=("grammatical_subject", "action_theme"),
            origin="ordered_three_role_construction",
            strict_structure=False,
        )
        append_frame(
            second, role_relation, third, (scope_start, scope_end),
            roles=("action_theme", "role_complement"),
            origin="ordered_three_role_construction",
            strict_structure=False,
        )
        append_frame(
            first, role_relation, third, (scope_start, scope_end),
            roles=("grammatical_subject", "role_complement"),
            origin="ordered_three_role_construction",
            strict_structure=False,
        )

    # Recover explicit theme-to-recipient/source/destination roles when the
    # model found two actor edges but omitted the ordered relation between the
    # two objects.  Only a source connector between the exact object mentions
    # licenses this projection; ordinary coordination never does.
    connector_pattern = re.compile(
        r"^[\s,;:()\[\]-]*(to|from|into|onto|toward|towards|through|via|with|"
        r"向|至|从|由|到|经由|通过)\b",
        flags=re.IGNORECASE,
    )
    existing_snapshot = [*frames, *generated]
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for frame in existing_snapshot:
        keys = list(frame.get("argument_occurrence_keys") or ())
        if len(keys) != 2 or not all(str(value) in by_key for value in keys):
            continue
        by_subject.setdefault(str(keys[0]), []).append(frame)
    for subject_key, subject_frames in by_subject.items():
        subject = by_key[subject_key]
        for first_index, first in enumerate(subject_frames):
            for second in subject_frames[first_index + 1:]:
                first_relation = re.sub(
                    r"\s+", " ", str(first.get("relation") or "").casefold(),
                ).strip()
                second_relation = re.sub(
                    r"\s+", " ", str(second.get("relation") or "").casefold(),
                ).strip()
                first_root = re.findall(r"[a-z]+|[\u3400-\u9fff]+", first_relation)
                second_root = re.findall(r"[a-z]+|[\u3400-\u9fff]+", second_relation)
                if not first_root or not second_root or first_root[0] != second_root[0]:
                    continue
                first_object = by_key[str(first["argument_occurrence_keys"][1])]
                second_object = by_key[str(second["argument_occurrence_keys"][1])]
                ordered_objects = sorted(
                    (first_object, second_object),
                    key=lambda item: int(item["mention_start"]),
                )
                left, right = ordered_objects
                between = content[
                    int(left["mention_end"]):int(right["mention_start"])
                ]
                connector = connector_pattern.match(between)
                if connector is None:
                    continue
                containing_scopes = [
                    (start, end) for start, end, _text in split_relation_scopes(content)
                    if start <= int(subject["mention_start"])
                    and int(subject["mention_end"]) <= end
                    and start <= int(left["mention_start"])
                    and int(right["mention_end"]) <= end
                ]
                if len(containing_scopes) != 1:
                    continue
                append_frame(
                    left,
                    f"{first_root[0]} {connector.group(1)}",
                    right,
                    containing_scopes[0],
                    roles=("action_theme", "role_complement"),
                    origin="explicit_role_connector",
                    strict_structure=False,
                )

    # Ordered three-role frames are created after the first coordination pass.
    # Propagate a theme or complement slot over every parser-confirmed member
    # of the same coordinated component. This covers both postposed and
    # preposed role orders while preserving the other endpoint and never
    # relating peer members to one another.
    ordered_snapshot = list(generated)
    for scope_start, scope_end, components in coordination_components_by_scope:
        for frame in ordered_snapshot:
            if str(frame.get("deterministic_source_syntax") or "") != (
                "ordered_three_role_construction"
            ):
                continue
            roles = [str(value) for value in frame.get("roles") or ()]
            if len(roles) != 2:
                continue
            keys = [str(value) for value in (
                frame.get("argument_occurrence_keys") or ()
            )]
            if len(keys) != 2 or any(key not in by_key for key in keys):
                continue
            subject, object_ = by_key[keys[0]], by_key[keys[1]]
            if not (
                scope_start <= int(subject["mention_start"])
                and int(subject["mention_end"]) <= scope_end
                and scope_start <= int(object_["mention_start"])
                and int(object_["mention_end"]) <= scope_end
            ):
                continue
            for component in components:
                component_keys = {
                    str(item["occurrence_key"]) for item in component
                }
                for slot_index in (0, 1):
                    if (
                        roles[slot_index]
                        not in {"action_theme", "role_complement"}
                        or keys[slot_index] not in component_keys
                    ):
                        continue
                    for peer in component:
                        peer_key = str(peer["occurrence_key"])
                        if peer_key == keys[slot_index]:
                            continue
                        endpoints = [subject, object_]
                        endpoints[slot_index] = peer
                        if str(endpoints[0]["occurrence_key"]) == str(
                            endpoints[1]["occurrence_key"]
                        ):
                            continue
                        append_frame(
                            endpoints[0],
                            str(frame.get("relation") or ""),
                            endpoints[1],
                            (scope_start, scope_end),
                            roles=(roles[0], roles[1]),
                            origin="ordered_three_role_construction",
                            strict_structure=False,
                        )

    # The generative inventory is recall-oriented and can flatten an explicit
    # actor -> theme -> complement construction into a direct actor ->
    # complement edge.  The deterministic syntax pass above has enough source
    # information to recover the endpoint roles without choosing a logical
    # predicate.  Attach that parser role topology to any coextensive model
    # frame with the same frozen occurrences.  Schema binding can then enforce
    # its machine-readable source_role_pairs contract on model and parser
    # candidates alike.
    parser_role_candidates: dict[
        tuple[str, str], set[tuple[str, str, str]]
    ] = {}
    parser_surface_role_candidates: dict[
        tuple[str, str], set[tuple[str, str, str]]
    ] = {}
    role_topology_origins = {
        "ordered_three_role_construction",
        "explicit_role_connector",
        "intervening_predicate_construction",
        "modifier_interrupted_predicate",
        "coordinated_collective_relation_candidate",
        "possessive_reference_construction",
        "passive_agent_postpredicate_construction",
    }
    for item in generated:
        origin = str(item.get("deterministic_source_syntax") or "")
        roles = tuple(str(value or "") for value in item.get("roles") or ())
        keys = tuple(
            str(value or "")
            for value in item.get("argument_occurrence_keys") or ()
        )
        if (
            origin not in role_topology_origins
            or len(keys) != 2
            or len(roles) != 2
            or f"{roles[0]}->{roles[1]}" not in _VALID_SOURCE_ROLE_PAIRS
        ):
            continue
        parser_role_candidates.setdefault(
            (keys[0], keys[1]), set(),
        ).add((roles[0], roles[1], origin))
        surface_key = (
            _normalized_atomic_surface(
                str(by_key[keys[0]].get("surface") or "")
            ),
            _normalized_atomic_surface(
                str(by_key[keys[1]].get("surface") or "")
            ),
        )
        if all(surface_key):
            parser_surface_role_candidates.setdefault(
                surface_key, set(),
            ).add((roles[0], roles[1], origin))
    inferred_role_frames = 0

    def annotate_role_topology(values: list[dict[str, Any]]) -> None:
        nonlocal inferred_role_frames
        for index, frame in enumerate(values):
            existing_syntax = str(
                frame.get("deterministic_source_syntax") or ""
            )
            if (
                str(frame.get("deterministic_source_structure") or "")
                or existing_syntax not in {"", "coordinated_argument"}
            ):
                continue
            keys = tuple(
                str(value or "")
                for value in frame.get("argument_occurrence_keys") or ()
            )
            candidates = parser_role_candidates.get(keys, set())
            if not candidates:
                arguments = list(frame.get("arguments") or ())
                heads = list(frame.get("entity_heads") or ())
                if len(arguments) == 2 and len(heads) == 2:
                    surface_key = tuple(
                        _normalized_atomic_surface(
                            str(heads[item_index] or arguments[item_index] or "")
                        )
                        for item_index in (0, 1)
                    )
                    candidates = parser_surface_role_candidates.get(
                        surface_key, set(),
                    )
            role_pairs = {
                (left, right) for left, right, _origin in candidates
            }
            if len(role_pairs) != 1:
                continue
            left_role, right_role = next(iter(role_pairs))
            updated = dict(frame)
            updated["parser_inferred_source_roles"] = [
                left_role, right_role,
            ]
            updated["parser_inferred_source_syntax"] = "+".join(sorted({
                origin for _left, _right, origin in candidates
            }))
            values[index] = updated
            inferred_role_frames += 1

    # Table labels discovered before the generic syntax passes remain parser
    # metadata, never semantic endpoints.  Filter generated frames as well as
    # model frames: a later adjacency/coordination rule must not accidentally
    # reintroduce a header occurrence that the table parser already excluded.
    if table_header_occurrence_keys:
        generated_before_header_filter = len(generated)
        generated = [
            frame for frame in generated
            if not table_header_occurrence_keys.intersection(
                str(value or "")
                for value in frame.get("argument_occurrence_keys") or ()
            )
        ]
        table_header_frames_dropped += (
            generated_before_header_filter - len(generated)
        )

    annotate_role_topology(frames)
    annotate_role_topology(generated)

    seen = {
        (
            tuple(item.get("argument_occurrence_keys") or ()),
            _normalized_atomic_surface(str(item.get("relation") or "")),
            tuple(item.get("evidence_span") or ()),
        )
        for item in frames
    }
    added_by_origin: dict[str, int] = {}
    for item in generated:
        key = (
            tuple(item.get("argument_occurrence_keys") or ()),
            _normalized_atomic_surface(str(item.get("relation") or "")),
            tuple(item.get("evidence_span") or ()),
        )
        if key in seen:
            continue
        seen.add(key)
        frames.append(item)
        origin = str(
            item.get("deterministic_source_origin")
            or item.get("deterministic_source_structure")
            or item.get("deterministic_source_syntax")
            or ""
        )
        added_by_origin[origin] = added_by_origin.get(origin, 0) + 1
    return {
        **dict(inventory),
        "frames": frames,
        "deterministic_frames_added": sum(added_by_origin.values()),
        "deterministic_frames_by_origin": added_by_origin,
        "parser_role_frames_annotated": inferred_role_frames,
        "reference_nominal_prefix_repairs": (
            int(inventory.get("reference_nominal_prefix_repairs") or 0)
            + reference_nominal_prefix_repairs
        ),
        "table_header_frames_dropped": table_header_frames_dropped,
        "relation_endpoint_excluded_occurrence_keys": sorted(
            table_header_occurrence_keys
        ),
    }


def atomic_fact_inventory_prompt(
    opportunity: PairOpportunity,
    *,
    content: str,
) -> str:
    """Build a predicate-blind inventory request for one frozen context.

    This stage identifies direct proposition argument pairs without seeing a
    Schema predicate, entity type, candidate direction, or hypothesis.  The
    same request can therefore be cached for every candidate pair in the
    context and cannot encode predicate-name or domain-specific decisions.
    """

    start, end = opportunity.context_span
    if start < 0 or end <= start or end > len(content):
        raise ValueError("atomic fact inventory context span is invalid")
    context_text = content[start:end]
    uses_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", context_text))
    instruction = (
        "只依据给定原文，把每个直接陈述的最小原子谓词表示为一个语义 frame，不得使用"
        "预设关系类别，不得推断原文未说出的事实。每个 frame 只包含 relation、arguments、"
        "roles、argument_kinds、entity_heads、assertion_status：relation 是不含具名参与者的原文谓词或介词短语；arguments 是"
        "该谓词全部完整论元的字符串数组；roles、argument_kinds 和 entity_heads 均与 arguments 等长且按"
        "下标一一对应。argument_kinds 只能是 ENTITY、EVENT、STATE、CONDITION、VALUE、TEXT。"
        "assertion_status 只能是 ASSERTED、NEGATED、MODAL、CONDITIONAL，表示当前 frame 谓词"
        "本身在原文中是已断言、被否定、仅具有情态可能性还是受条件约束；不得丢失否定或把它改成肯定。"
        "argument_kinds 必须依据整个论元的核心指称判断：带有内嵌动词但整体指称人、组织、对象或"
        "地点的名词短语仍标为 ENTITY。对于 ENTITY，entity_heads 必须是逐字复制自该完整论元、"
        "能够单独指明该论元核心指称的最小名词短语；不得选择所有者、来源、修饰者、语法强调/反身标记或内嵌事件参与者。"
        "对于其他 argument_kinds，对应 entity_heads 必须是空字符串。修饰成分表达的谓词必须另建"
        "frame。包含动作、变化、"
        "发生、未发生、否定动作或完整从句的论元标为 EVENT、STATE 或 CONDITION，不能因为内部"
        "含有实体就标为 ENTITY。信息承载型定语、关系从句、补语或嵌套谓词中明确陈述的每个谓词"
        "都必须单独生成 frame；一个表层句子可以产生多个 frame。对每个谓词分别列出直接承担其"
        "语义角色的参与者，不得省略修饰语或从句中的内部谓词。同时，只有真正承担当前 frame "
        "谓词角色的参与者才能进入该 frame；作为另一个事件、状态或条件内部成分但不承担当前"
        "谓词角色的实体不能提升为当前 frame 的论元。同一角色的每个并列参与者必须拆成独立的 "
        "arguments 数组项，禁止把整段并列列表合并成一个 argument；这些成员使用完全相同的 "
        "role，不得给同角色成员追加顺序编号；不同角色必须使用不同 role，因此并列"
        "同级项不会彼此相连。在把 coverage_complete 设为 true 前，必须逐项检查 arguments；"
        "只要某一项仍表示两个或更多并列参与者，就必须先拆成多个独立项并重复同一 role。每个论元"
        "必须逐字复制自原文。展开省略的共用主语、宾语或谓词，但不得把另一个谓词的论元错误继承"
        "到当前 frame。对于使役或结果谓词，原因论元必须是原文实际引发结果的完整事件、动作或条件，结果论元"
        "必须保留完整结果从句；结果从句内部的语法主语只是结果事件的一部分，不能替代原因论元。"
        "抽象示例：若原文表达‘事件甲引发系统乙未能执行动作丙’，使役 frame 的两个论元应为"
        "‘事件甲’和‘系统乙未能执行动作丙’；不得另建肯定的‘系统乙执行动作丙’ frame。"
        "事件、状态、条件、否定、情态、数量和必要限定词属于完整论元时必须保留，"
        "不能用其中的内部名词替代完整论元；否定、未发生或不可能的动作不得改写为肯定事实。"
        "不同谓词仅通过第三论元相接时必须是不同 frame。只有全部直接事实和全部明确角色参与者"
        "均已覆盖时 coverage_complete 才为 true；原文截断、结构不清或覆盖不全时必须为 false。"
        "整个结果只返回 coverage_complete 和 frames。"
        if uses_cjk else
        "Using only the supplied source, represent each minimal directly stated atomic "
        "predicate as one semantic frame. Do not use a preset relation taxonomy or infer "
        "unstated facts. Every frame contains only relation, arguments, roles, "
        "argument_kinds, entity_heads, and assertion_status. "
        "relation is a source predicate or preposition phrase without named participants; "
        "arguments is the array of all complete predicate arguments; roles, "
        "argument_kinds, and entity_heads are equally sized position-aligned arrays. "
        "argument_kinds may only contain ENTITY, EVENT, STATE, CONDITION, VALUE, or TEXT. "
        "assertion_status must be ASSERTED, NEGATED, MODAL, or CONDITIONAL and records "
        "whether this frame's predicate is asserted, negated, merely modal, or conditional. "
        "Never drop negation or turn it into an affirmative frame. "
        "Classify the kind by the core referent of the whole argument. A noun phrase with "
        "an embedded verb is still ENTITY when the whole phrase denotes a person, "
        "organization, object, or place. For each ENTITY argument, entity_heads must be "
        "the smallest verbatim noun phrase inside that complete argument that independently "
        "identifies its core referent. Never choose its owner, source, modifier, grammatical "
        "emphasis/reflexive marker, or a "
        "participant of an embedded event. For every non-ENTITY argument, the aligned "
        "entity_heads value must be an empty string. Emit predicates in modifiers as "
        "separate frames. An action, change, "
        "occurrence, non-occurrence, negated action, or full "
        "clause is EVENT, STATE, or CONDITION even when it contains entity names. "
        "Every explicitly stated predicate inside an information-bearing modifier, "
        "relative clause, complement, or embedded clause must also receive its own "
        "frame; one surface sentence may yield several frames. For each predicate, "
        "list the participants that directly fill that predicate's semantic roles. "
        "Never promote an entity internal to another event, state, or condition into "
        "the current frame when it does not fill a role of the current predicate. "
        "Every coordinated participant that fills the same role must be a separate "
        "arguments array item; never merge an entire coordinated list into one argument. "
        "Those participants must use exactly the same role without ordinal suffixes, while distinct roles "
        "must differ, so peer members "
        "are never directly connected. Before setting coverage_complete true, inspect every "
        "arguments item; if one item still denotes two or more coordinated participants, "
        "split it into separate items and repeat the same role. Copy every argument verbatim from the source. "
        "Expand omitted shared subjects, objects, or predicates, but never inherit an "
        "argument from a different predicate. For a causative or result predicate, the cause argument must be the complete "
        "source event, action, or condition that brings about the result, and the result "
        "argument must preserve the complete result clause. A grammatical subject inside "
        "the result clause is part of that result event and cannot replace the cause. "
        "Abstract example: if Event A causes System B not to perform Action C, the causative "
        "frame arguments are Event A and the complete result 'System B does not perform "
        "Action C'; never emit an affirmative 'System B performs Action C' frame. "
        "Keep event triggers, states, conditions, negation, modality, quantities, and required qualifiers inside a complete "
        "argument; never turn a negated, unrealized, or impossible action into an "
        "affirmative fact. Predicates connected only through a third argument remain "
        "separate frames. Set coverage_complete true only after covering every direct "
        "fact and every explicit role participant; use false for truncation, structural "
        "ambiguity, or incomplete coverage. Return only coverage_complete and frames."
    )
    return json.dumps(
        {
            "instruction": instruction,
            "contract_version": GROUNDED_RELATION_ATOMIC_FACT_PROMPT_VERSION,
            "context": {
                "start": start,
                "end": end,
                "text": context_text,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def atomic_fact_inventory_response_schema() -> dict[str, Any]:
    return {
        "$id": GROUNDED_RELATION_ATOMIC_FACT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "coverage_complete": {"type": "boolean"},
            "frames": {
                "type": "array",
                "maxItems": 64,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "relation": {
                            "type": "string", "minLength": 1, "maxLength": 300,
                        },
                        "arguments": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 16,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 500,
                            },
                        },
                        "roles": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 16,
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 80,
                            },
                        },
                        "argument_kinds": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 16,
                            "items": {
                                "type": "string",
                                "enum": [
                                    "ENTITY",
                                    "EVENT",
                                    "STATE",
                                    "CONDITION",
                                    "VALUE",
                                    "TEXT",
                                ],
                            },
                        },
                        "entity_heads": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 16,
                            "items": {
                                "type": "string",
                                "minLength": 0,
                                "maxLength": 200,
                            },
                        },
                        "assertion_status": {
                            "type": "string",
                            "enum": [
                                "ASSERTED", "NEGATED", "MODAL", "CONDITIONAL",
                            ],
                        },
                    },
                    "required": [
                        "relation", "arguments", "roles", "argument_kinds",
                        "entity_heads", "assertion_status",
                    ],
                },
            },
        },
        "required": ["coverage_complete", "frames"],
    }


def _normalized_atomic_surface(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _source_relation_expression_is_grounded(
    source: Any,
    expression: Any,
) -> bool:
    """Return whether an open relation is an exact source-language phrase.

    PDF extraction may insert whitespace inside CJK words and may wrap any
    language across lines.  Those layout-only differences are ignored, while
    translation, stemming, synonym expansion and reordered tokens are not.
    """

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


def _atomic_surface_is_source_copied(
    source: Any,
    candidate: Any,
    *,
    allow_contained: bool = False,
) -> bool:
    """Accept only a case/whitespace-normalized copy of frozen source text.

    Local models occasionally change capitalization while copying an entity
    head (for example ``HTTP`` -> ``http``).  That is a protocol variation,
    not semantic paraphrase.  The optional contained form is restricted to
    a complete word boundary so a head such as ``if`` cannot match the middle
    of ``gift``.  No fuzzy matching, aliasing, stemming or domain vocabulary
    is used here.
    """

    source_text = _normalized_atomic_surface(source)
    candidate_text = _normalized_atomic_surface(candidate)
    if not source_text or not candidate_text:
        return False
    if source_text == candidate_text:
        return True
    if not allow_contained:
        return False
    return re.search(
        rf"(?<!\w){re.escape(candidate_text)}(?!\w)",
        source_text,
    ) is not None


def _normalized_atomic_role(value: Any) -> str:
    """Normalize a model-local role identifier without reading its semantics."""

    normalized = _normalized_atomic_surface(value)
    # Models sometimes emit ``member1/member2`` or ``role_1/role_2`` even
    # when the prompt requires one shared identifier.  A trailing decimal is
    # only an ordinal protocol decoration; stripping it is independent of the
    # role language, predicate, document domain, and relation decision.
    return re.sub(r"[\s_\-:#]*[0-9０-９]+$", "", normalized)


def parse_atomic_fact_inventory(
    raw: Any,
    *,
    context_text: str,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate an inventory without adding language- or predicate-specific rules."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "atomic fact inventory returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "atomic fact inventory response must be an object"
        )
    coverage_complete = raw.get("coverage_complete")
    if not isinstance(coverage_complete, bool):
        raise RelationClassificationError(
            "atomic fact inventory requires coverage_complete"
        )
    raw_frames = raw.get("frames")
    # Frozen scripted fixtures from the first inventory prototype use a
    # two-argument ``items`` shape. Normalize that shape structurally into
    # frames; production requests advertise only the current frame schema.
    if raw_frames is None and isinstance(raw.get("items"), list):
        raw_frames = []
        for raw_item in raw["items"]:
            if not isinstance(raw_item, Mapping):
                raise RelationClassificationError(
                    "atomic fact inventory item must be an object"
                )
            raw_frames.append({
                "relation": (
                    raw_item.get("relation")
                    or raw_item.get("relation_text")
                ),
                "arguments": [
                    raw_item.get("arg1")
                    or raw_item.get("subject_surface"),
                    raw_item.get("arg2")
                    or raw_item.get("object_surface"),
                ],
                "roles": ["source_argument", "target_argument"],
                "argument_kinds": ["ENTITY", "ENTITY"],
                "entity_heads": [
                    raw_item.get("arg1") or raw_item.get("subject_surface"),
                    raw_item.get("arg2") or raw_item.get("object_surface"),
                ],
                "assertion_status": "ASSERTED",
            })
    if not isinstance(raw_frames, list):
        raise RelationClassificationError(
            "atomic fact inventory frames must be an array"
        )
    if len(raw_frames) > 64:
        raise RelationClassificationError(
            "atomic fact inventory exceeded its bounded frame count"
        )
    if allow_partial:
        # A context-level source inventory is a recall source. One malformed
        # frame (often a pronoun/coreference mistake or one omitted array
        # item) must not erase independently valid frames in the same source
        # response. Keep the strict parser as the default for verifier audits;
        # this opt-in recovery parses each frame under that same contract and
        # quarantines only the malformed frame.
        valid_frames: list[dict[str, Any]] = []
        dropped_frames = 0
        for raw_frame in raw_frames:
            try:
                partial = parse_atomic_fact_inventory(
                    {
                        "coverage_complete": True,
                        "frames": [raw_frame],
                    },
                    context_text=context_text,
                    allow_partial=False,
                )
            except RelationClassificationError:
                dropped_frames += 1
                continue
            valid_frames.extend(partial.get("frames") or ())
        return {
            "coverage_complete": bool(coverage_complete and not dropped_frames),
            "frames": valid_frames,
            "partial_recovery": bool(dropped_frames),
            "dropped_frames": dropped_frames,
        }
    frames: list[dict[str, Any]] = []
    seen: set[tuple[
        str, tuple[str, ...], tuple[str, ...], tuple[str, ...],
        tuple[str, ...], str,
    ]] = set()
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, Mapping):
            raise RelationClassificationError(
                "atomic fact inventory frame must be an object"
            )
        # The relation pipeline audits binary endpoint pairs.  Local models
        # often also return valid unary predicate frames (for example an
        # event's experiencer or a location-only adjunct).  Those frames do
        # not carry a pair that can be compared with a hypothesis and should
        # not invalidate the binary frames that follow.  Ignore only the
        # structurally complete unary case; malformed binary frames remain
        # fail-closed below.
        raw_arguments = raw_frame.get("arguments")
        if isinstance(raw_arguments, list) and len(raw_arguments) < 2:
            raw_roles = raw_frame.get("roles")
            raw_kinds = raw_frame.get("argument_kinds")
            raw_heads = raw_frame.get("entity_heads")
            if (
                isinstance(raw_roles, list)
                and isinstance(raw_kinds, list)
                and isinstance(raw_heads, list)
                and len(raw_roles) == len(raw_arguments)
                and len(raw_kinds) == len(raw_arguments)
                and len(raw_heads) == len(raw_arguments)
            ):
                continue
        relation = str(raw_frame.get("relation") or "").strip()
        arguments = raw_frame.get("arguments")
        roles = raw_frame.get("roles")
        argument_kinds = raw_frame.get("argument_kinds")
        entity_heads = raw_frame.get("entity_heads")
        assertion_status = str(
            raw_frame.get("assertion_status") or ""
        ).strip().upper()
        # Qwen occasionally omits trailing empty ``entity_heads`` values for
        # EVENT/STATE/VALUE arguments even though it returns the complete
        # arguments and kinds arrays.  Those empty positions carry no identity
        # information, so padding them is a lossless protocol repair.  Never
        # infer a missing head for an ENTITY argument; such a frame remains
        # fail-closed.
        if (
            isinstance(arguments, list)
            and isinstance(argument_kinds, list)
            and len(argument_kinds) == len(arguments)
        ):
            if entity_heads is None and all(
                str(kind or "").strip().upper() != "ENTITY"
                for kind in argument_kinds
            ):
                entity_heads = [""] * len(arguments)
            elif isinstance(entity_heads, list) and len(entity_heads) < len(arguments):
                missing_tail = argument_kinds[len(entity_heads):]
                if all(
                    str(kind or "").strip().upper() != "ENTITY"
                    for kind in missing_tail
                ):
                    entity_heads = [*entity_heads, *([""] * len(missing_tail))]
        if (
            not relation
            or not isinstance(arguments, list)
            or not isinstance(roles, list)
            or not isinstance(argument_kinds, list)
            or not isinstance(entity_heads, list)
            or not 2 <= len(arguments) <= 16
            or len(roles) != len(arguments)
            or len(argument_kinds) != len(arguments)
            or len(entity_heads) != len(arguments)
            or assertion_status not in {
                "ASSERTED", "NEGATED", "MODAL", "CONDITIONAL",
            }
        ):
            raise RelationClassificationError(
                "atomic fact inventory frame is incomplete"
            )
        normalized_arguments: list[str] = []
        normalized_roles: list[str] = []
        normalized_kinds: list[str] = []
        normalized_heads: list[str] = []
        allowed_kinds = {
            "ENTITY", "EVENT", "STATE", "CONDITION", "VALUE", "TEXT",
        }
        for argument, role, argument_kind, entity_head in zip(
            arguments, roles, argument_kinds, entity_heads, strict=True,
        ):
            argument_text = str(argument or "").strip()
            role_text = str(role or "").strip()
            kind_text = str(argument_kind or "").strip().upper()
            head_text = str(entity_head or "").strip()
            if (
                not argument_text
                or not role_text
                or kind_text not in allowed_kinds
            ):
                raise RelationClassificationError(
                    "atomic fact inventory argument role is incomplete"
                )
            if not _atomic_surface_is_source_copied(
                context_text, argument_text, allow_contained=True,
            ):
                raise RelationClassificationError(
                    "atomic fact inventory argument is not copied from source"
                )
            if kind_text == "ENTITY":
                if not head_text or not _atomic_surface_is_source_copied(
                    argument_text, head_text, allow_contained=True,
                ):
                    raise RelationClassificationError(
                        "atomic fact inventory entity head is not copied from its argument"
                    )
                if not _atomic_surface_is_source_copied(
                    context_text, head_text, allow_contained=True,
                ):
                    raise RelationClassificationError(
                        "atomic fact inventory entity head is not copied from source"
                    )
            elif head_text:
                # ``entity_heads`` is an identity channel, not a generic
                # noun-head channel.  Models frequently copy a lexical head
                # for EVENT/STATE/VALUE arguments even when the argument kind
                # is correct.  Discard that non-identity decoration while
                # retaining the complete source-copied argument; accepting it
                # as an entity would create false endpoint pairings.
                head_text = ""
            normalized_arguments.append(argument_text[:500])
            normalized_roles.append(role_text[:80])
            normalized_kinds.append(kind_text)
            normalized_heads.append(head_text[:200])
        key = (
            _normalized_atomic_surface(relation),
            tuple(map(_normalized_atomic_surface, normalized_arguments)),
            tuple(map(_normalized_atomic_role, normalized_roles)),
            tuple(normalized_kinds),
            tuple(map(_normalized_atomic_surface, normalized_heads)),
            assertion_status,
        )
        if key in seen:
            continue
        seen.add(key)
        frames.append({
            "relation": relation[:300],
            "arguments": normalized_arguments,
            "roles": normalized_roles,
            "argument_kinds": normalized_kinds,
            "entity_heads": normalized_heads,
            "assertion_status": assertion_status,
        })
    return {
        "coverage_complete": coverage_complete,
        "frames": frames,
    }


def atomic_fact_inventory_contains_pair(
    inventory: Mapping[str, Any],
    endpoint_a_surface: str,
    endpoint_b_surface: str,
) -> bool:
    """Return whether one frame directly pairs the two endpoint referents.

    Exact copied arguments always match. A qualified ENTITY argument may
    project only to its explicit, source-copied core head. Arbitrary substring
    containment is forbidden, so owners, modifiers and embedded-event
    participants cannot replace the complete argument. Event/state/clause
    arguments remain atomic. This is predicate- and domain-independent.
    """

    endpoint_pair = {
        _normalized_atomic_surface(endpoint_a_surface),
        _normalized_atomic_surface(endpoint_b_surface),
    }
    if len(endpoint_pair) != 2:
        return False
    normalized_a = _normalized_atomic_surface(endpoint_a_surface)
    normalized_b = _normalized_atomic_surface(endpoint_b_surface)

    for frame in inventory.get("frames") or []:
        if not isinstance(frame, Mapping):
            continue
        if str(frame.get("assertion_status") or "").upper() != "ASSERTED":
            continue
        arguments = list(frame.get("arguments") or [])
        roles = list(frame.get("roles") or [])
        argument_kinds = list(frame.get("argument_kinds") or [])
        entity_heads = list(frame.get("entity_heads") or [])
        if not (
            len(arguments) == len(roles) == len(argument_kinds)
            == len(entity_heads)
        ):
            continue

        def matches(index: int, endpoint: str) -> bool:
            if _normalized_atomic_surface(arguments[index]) == endpoint:
                return True
            return (
                str(argument_kinds[index] or "").strip().upper() == "ENTITY"
                and _normalized_atomic_surface(entity_heads[index]) == endpoint
            )

        indices_a = [
            index for index, _value in enumerate(arguments)
            if matches(index, normalized_a)
        ]
        indices_b = [
            index for index, _value in enumerate(arguments)
            if matches(index, normalized_b)
        ]
        if any(
            _normalized_atomic_role(roles[index_a])
            != _normalized_atomic_role(roles[index_b])
            for index_a in indices_a
            for index_b in indices_b
            if index_a != index_b
        ):
            return True
    return False


def _atomic_frame_argument_matches(
    frame: Mapping[str, Any], index: int, endpoint: str,
) -> bool:
    arguments = list(frame.get("arguments") or [])
    kinds = list(frame.get("argument_kinds") or [])
    heads = list(frame.get("entity_heads") or [])
    if not (
        0 <= index < len(arguments)
        and len(arguments) == len(kinds) == len(heads)
    ):
        return False
    normalized = _normalized_atomic_surface(endpoint)
    if _normalized_atomic_surface(arguments[index]) == normalized:
        return True
    return (
        str(kinds[index] or "").strip().upper() == "ENTITY"
        and _normalized_atomic_surface(heads[index]) == normalized
    )


def atomic_frame_schema_alignment_prompt(
    inventory: Mapping[str, Any],
    *,
    subject_surface: str,
    object_surface: str,
    subject_type: str = "",
    object_type: str = "",
    predicate: str,
    logical_schema: Mapping[str, Any],
) -> str:
    """Build a candidate-blind frame-to-Schema role-binding request."""

    spec = normalize_predicate_schema(logical_schema).get(predicate)
    if spec is None:
        raise ValueError("frame alignment requires current predicate metadata")
    frames: list[dict[str, Any]] = []
    for source_index, frame in enumerate(inventory.get("frames") or []):
        if not isinstance(frame, Mapping):
            continue
        if str(frame.get("assertion_status") or "").upper() != "ASSERTED":
            continue
        arguments = list(frame.get("arguments") or [])
        roles = list(frame.get("roles") or [])
        kinds = list(frame.get("argument_kinds") or [])
        heads = list(frame.get("entity_heads") or [])
        if not (
            len(arguments) == len(roles) == len(kinds) == len(heads)
        ):
            continue
        subject_indices = [
            index for index in range(len(arguments))
            if _atomic_frame_argument_matches(frame, index, subject_surface)
        ]
        object_indices = [
            index for index in range(len(arguments))
            if _atomic_frame_argument_matches(frame, index, object_surface)
        ]
        if not any(a != b for a in subject_indices for b in object_indices):
            continue
        entity_types: list[str] = []
        for argument_index in range(len(arguments)):
            matched_types = {
                str(endpoint_type).strip()
                for endpoint_surface, endpoint_type in (
                    (subject_surface, subject_type),
                    (object_surface, object_type),
                )
                if str(endpoint_type).strip()
                and _atomic_frame_argument_matches(
                    frame, argument_index, endpoint_surface,
                )
            }
            entity_types.append(
                next(iter(matched_types)) if len(matched_types) == 1 else ""
            )
        frames.append({
            "frame_index": source_index,
            "relation": str(frame.get("relation") or ""),
            "arguments": arguments,
            "roles": roles,
            "argument_kinds": kinds,
            "entity_heads": heads,
            "entity_types": entity_types,
            "assertion_status": "ASSERTED",
        })
    if not frames:
        raise ValueError("frame alignment requires a direct endpoint frame")
    return json.dumps({
        "instruction": (
            "Candidate endpoints are deliberately withheld. For every semantic_frame, "
            "first decide whether it instantiates the same underlying relation as "
            "schema_relation. Active/passive voice, inverse wording, or a whole-first "
            "surface form may reverse argument order. Mere association, interaction, "
            "co-membership, shared context, or a negative example is not equivalence. "
            "Each selected argument must itself instantiate the required Schema role. "
            "An entity that is merely the patient, theme, tool, recipient, location, "
            "owner, modifier, or result of an action is not the action/task/event/goal "
            "itself, and participation in a frame does not make roles interchangeable. "
            "entity_types are source-grounded canonical categories for the matching "
            "frame arguments when available. They are supporting constraints: an empty "
            "value is unknown, while a known category that conflicts with a required "
            "semantic role must not be ignored even when the Schema taxonomy uses '*'. "
            "Enumerate every Schema-valid ordered binding as frame_index, subject_index, "
            "object_index. The indexes must fill required_subject_role and "
            "required_object_role; never copy argument order merely because it is the "
            "source order. If one whole contains several parts and the Schema direction "
            "is part-to-whole, emit one binding per part, each pointing to the whole. "
            "For an asymmetric relation emit only the semantically correct direction. "
            "Return coverage_complete true only after considering every supplied frame "
            "and every valid binding. Return an empty alignments array when no frame is "
            "equivalent."
        ),
        "contract_version": GROUNDED_RELATION_FRAME_ALIGNMENT_PROMPT_VERSION,
        "semantic_frames": frames,
        "schema_relation": {
            "semantic_definition": spec.semantic_definition,
            "domain_types": list(spec.domain_types),
            "range_types": list(spec.range_types),
            "required_subject_role": spec.subject_role_description,
            "required_object_role": spec.object_role_description,
            "direction_description": spec.direction_description,
            "symmetric": bool(spec.symmetric),
            "positive_examples": list(spec.positive_examples),
            "negative_examples": list(spec.negative_examples),
        },
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_inventory_has_direct_endpoint_pair(
    inventory: Mapping[str, Any],
    *,
    subject_surface: str,
    object_surface: str,
) -> bool:
    """Return whether the source-grounded inventory pairs both endpoints.

    Frame-to-Schema alignment is an auxiliary audit.  It is only meaningful
    when the predicate-blind inventory has already supplied one asserted frame
    containing the exact endpoints.  Keeping this check structural avoids
    treating a qualified event or an unrelated frame as an alignment failure.
    """

    return atomic_fact_inventory_contains_pair(
        inventory, subject_surface, object_surface,
    )


def _atomic_inventory_has_multi_argument_endpoint_frame(
    inventory: Mapping[str, Any],
    *,
    subject_surface: str,
    object_surface: str,
) -> bool:
    """Detect a source frame where the pair is one of several arguments.

    Multi-member frames are the one generic structure in which a direct
    proposer can copy source order instead of the Schema direction. This is a
    structural signal only; it does not name a predicate, language, or domain.
    """

    for frame in inventory.get("frames") or []:
        if not isinstance(frame, Mapping):
            continue
        if str(frame.get("assertion_status") or "").upper() != "ASSERTED":
            continue
        arguments = list(frame.get("arguments") or [])
        roles = list(frame.get("roles") or [])
        kinds = list(frame.get("argument_kinds") or [])
        heads = list(frame.get("entity_heads") or [])
        if len(arguments) < 3 or not (
            len(arguments) == len(roles) == len(kinds) == len(heads)
        ):
            continue
        subject_indices = [
            index for index in range(len(arguments))
            if _atomic_frame_argument_matches(frame, index, subject_surface)
        ]
        object_indices = [
            index for index in range(len(arguments))
            if _atomic_frame_argument_matches(frame, index, object_surface)
        ]
        if any(
            subject_index != object_index
            for subject_index in subject_indices
            for object_index in object_indices
        ):
            return True
    return False


def atomic_frame_schema_alignment_response_schema() -> dict[str, Any]:
    return {
        "$id": GROUNDED_RELATION_FRAME_ALIGNMENT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "coverage_complete": {"type": "boolean"},
            "alignments": {
                "type": "array",
                "maxItems": 128,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "frame_index": {
                            "type": "integer", "minimum": 0, "maximum": 63,
                        },
                        "subject_index": {
                            "type": "integer", "minimum": 0, "maximum": 15,
                        },
                        "object_index": {
                            "type": "integer", "minimum": 0, "maximum": 15,
                        },
                    },
                    "required": [
                        "frame_index", "subject_index", "object_index",
                    ],
                },
            },
        },
        "required": ["coverage_complete", "alignments"],
    }


def deterministic_source_structure_schema_bindings(
    inventory: Mapping[str, Any],
    *,
    logical_schema: Mapping[str, Any],
    include_review_only: bool = True,
) -> dict[str, Any]:
    """Bind parser-proved source roles to one exact Schema capability.

    The match is entirely data-driven: a frame supplies a structural kind and
    optional table-header role, while a logical predicate explicitly declares
    the same capability.  Ambiguous or unlabeled table matches stay unbound.
    These bindings are still hypotheses and retain the ordinary deterministic
    validator plus independent ordered verifier before publication.
    """

    schema = normalize_predicate_schema(logical_schema)
    raw_by_name = {
        str(item.get("name") or ""): item
        for item in logical_schema.get("relation_types") or []
        if isinstance(item, Mapping) and str(item.get("name") or "")
    }

    def normalized_label(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    role_contracts = {
        "table_header_assignment": ("row_key", "column_value"),
        "line_identifier_binding": ("source_label", "bound_topic"),
        "list_parent_child": ("structural_child", "structural_parent"),
        "section_label_parent_child": (
            "structural_child", "structural_parent",
        ),
    }
    source_projection_kinds = {
        "ATOMIC_FRAME_SCHEMA_PROJECTION",
        "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
    }
    open_carriers = [
        spec for spec in schema.values()
        if spec.publish_mode == PUBLISH_AUTO
        and bool(spec.open_relation_policy.get("enabled"))
        and bool(spec.open_relation_policy.get("fallback_only"))
        and source_projection_kinds.intersection({
            str(value)
            for value in spec.open_relation_policy.get(
                "allowed_candidate_sources", ()
            )
        })
    ]
    if len(open_carriers) > 1:
        raise ValueError(
            "logical Schema declares multiple open relation fallback carriers"
        )
    bindings: list[dict[str, Any]] = []
    for frame_index, frame in enumerate(inventory.get("frames") or ()):
        if not isinstance(frame, Mapping):
            continue
        structure_kind = str(
            frame.get("deterministic_source_structure") or ""
        )
        candidate_kinds = {structure_kind}
        if structure_kind == "section_or_list_parent_child":
            # Compatibility for immutable inventories produced by the prior
            # parser version; both legacy forms share child-to-parent order.
            candidate_kinds = {
                "list_parent_child", "section_label_parent_child",
            }
        expected_roles = role_contracts.get(structure_kind)
        if expected_roles is None and structure_kind == (
            "section_or_list_parent_child"
        ):
            expected_roles = ("structural_child", "structural_parent")
        arguments = list(frame.get("arguments") or [])
        roles = tuple(str(value) for value in frame.get("roles") or ())
        if (
            expected_roles is None or len(arguments) != 2
            or roles != expected_roles
            or str(frame.get("assertion_status") or "").upper()
            != "ASSERTED"
        ):
            continue
        frame_role_label = normalized_label(
            frame.get("structural_role_label")
        )
        frame_candidate_source = (
            "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION"
            if list(frame.get("argument_occurrence_keys") or ())
            else "ATOMIC_FRAME_SCHEMA_PROJECTION"
        )
        compatible_open_carriers = [
            spec for spec in open_carriers
            if frame_candidate_source in {
                str(value)
                for value in spec.open_relation_policy.get(
                    "allowed_candidate_sources", ()
                )
            }
        ]
        matches = []
        for name, spec in schema.items():
            if (
                spec.publish_mode == PUBLISH_DISABLED
                or (
                    spec.publish_mode == PUBLISH_REVIEW_ONLY
                    and not include_review_only
                )
            ):
                continue
            declared_kinds = set(spec.structural_evidence_kinds)
            if not declared_kinds.intersection(candidate_kinds):
                continue
            raw = raw_by_name.get(name, {})
            declared_labels = {
                normalized_label(value)
                for value in raw.get("structural_role_labels") or []
                if normalized_label(value)
            }
            if structure_kind == "table_header_assignment" and (
                not frame_role_label or frame_role_label not in declared_labels
            ):
                continue
            matches.append(name)
        binding_source = "DETERMINISTIC_SOURCE_STRUCTURE"
        if len(matches) != 1:
            # A table header itself is an exact source-owned role even when an
            # older active Schema has no predicate that declares that label.
            # Preserve the concrete column role through the single open
            # carrier instead of asking an LLM whether row adjacency counts as
            # prose entailment. Other structure kinds remain Schema-gated.
            if (
                not matches
                and structure_kind == "table_header_assignment"
                and frame_role_label
                and len(compatible_open_carriers) == 1
            ):
                matches = [compatible_open_carriers[0].name]
                binding_source = (
                    "DETERMINISTIC_SOURCE_STRUCTURE_OPEN_CARRIER"
                )
            else:
                continue
        bindings.append({
            "frame_index": frame_index,
            "predicate": matches[0],
            "subject_index": 0,
            "object_index": 1,
            "binding_source": binding_source,
            "source_structure_kind": structure_kind,
        })
    return {
        "coverage_complete": False,
        "bindings": bindings,
        "dropped_bindings": [],
        "parser_version": GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
    }


def source_frame_schema_recall_shortlist(
    inventory: Mapping[str, Any],
    existing_bindings: Sequence[Mapping[str, Any]],
    *,
    occurrences: Sequence[Mapping[str, Any]],
    logical_schema: Mapping[str, Any],
    max_additions_per_frame: int = 1,
    minimum_score: float = 0.18,
    include_review_only: bool = True,
) -> dict[str, Any]:
    """Return a bounded semantic-retrieval backstop for Schema binding.

    Enumerative model binding has a characteristic false-negative mode: a
    plausible lexical predicate is returned and a second, role-equivalent
    predicate is never considered.  This helper follows the usual retrieval
    then rerank architecture.  It proposes at most a tiny deterministic
    shortlist from type-compatible Schema predicates; every proposal remains
    an ordinary hypothesis and must pass deterministic validation plus the
    independent ordered verifier before publication.

    The scorer uses only versioned Schema semantics/examples and frozen source
    text.  It never contains domain terms or predicate-name branches.  Strict
    parser structures are excluded because their exact Schema contracts are
    handled by ``deterministic_source_structure_schema_bindings``.
    """

    cap = max(0, min(4, int(max_additions_per_frame)))
    threshold = max(0.0, min(1.0, float(minimum_score)))
    if cap == 0:
        return {
            "bindings": [],
            "frames_considered": 0,
            "candidates_considered": 0,
            "version": GROUNDED_RELATION_SOURCE_RECALL_SHORTLIST_VERSION,
        }
    schema = normalize_predicate_schema(logical_schema)
    occurrence_by_key = {
        str(item.get("occurrence_key") or ""): item
        for item in occurrences
        if str(item.get("occurrence_key") or "")
    }
    existing_keys = {
        (
            int(item.get("frame_index", -1)),
            str(item.get("predicate") or ""),
            int(item.get("subject_index", -1)),
            int(item.get("object_index", -1)),
        )
        for item in existing_bindings
        if isinstance(item, Mapping)
    }
    selected_predicates_by_frame: dict[int, set[str]] = {}
    for frame_index, predicate, _subject, _object in existing_keys:
        selected_predicates_by_frame.setdefault(frame_index, set()).add(
            predicate
        )

    def semantic_units(value: Any) -> set[str]:
        text = str(value or "").casefold().replace("_", " ")
        words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        units = {word for word in words if word}
        compact = "".join(words)
        if compact:
            width = 3 if len(compact) >= 3 else len(compact)
            if width:
                units.update(
                    f"#{compact[index:index + width]}"
                    for index in range(0, len(compact) - width + 1)
                )
        return units

    def similarity(left: Any, right: Any) -> float:
        left_units = semantic_units(left)
        right_units = semantic_units(right)
        if not left_units or not right_units:
            return 0.0
        return len(left_units & right_units) / len(left_units | right_units)

    def best_similarity(query: Any, values: Sequence[Any]) -> float:
        return max(
            (similarity(query, value) for value in values if str(value or "")),
            default=0.0,
        )

    additions: list[dict[str, Any]] = []
    frames_considered = 0
    candidates_considered = 0
    below_threshold = 0
    frames = list(inventory.get("frames") or ())
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        if str(frame.get("assertion_status") or "").upper() != "ASSERTED":
            continue
        if str(frame.get("deterministic_source_structure") or ""):
            continue
        keys = [str(value) for value in (
            frame.get("argument_occurrence_keys") or ()
        )]
        if len(keys) != 2 or any(key not in occurrence_by_key for key in keys):
            continue
        endpoints = [occurrence_by_key[key] for key in keys]
        if (
            not keys[0] or not keys[1] or keys[0] == keys[1]
            or str(endpoints[0].get("canonical_id") or "")
            and str(endpoints[0].get("canonical_id") or "")
            == str(endpoints[1].get("canonical_id") or "")
        ):
            continue
        relation = str(frame.get("relation") or "")
        evidence = str(frame.get("evidence_text") or "")
        roles = [str(value) for value in frame.get("roles") or ()]
        syntax_origin = str(
            frame.get("deterministic_source_syntax") or ""
        )
        collective_frame = (
            roles == [
                "collective_participant", "collective_participant",
            ]
            and syntax_origin == (
                "coordinated_collective_relation_candidate"
            )
        )
        compiled_contract_frame = any(
            spec.source_role_pairs
            and any(
                _source_frame_role_pair(frame, subject_index, object_index)
                in spec.source_role_pairs
                for subject_index, object_index in ((0, 1), (1, 0))
            )
            for spec in schema.values()
        )
        lexical_binary_frame = (
            roles == ["grammatical_subject", "direct_argument"]
            and syntax_origin == "intervening_predicate_construction"
        )
        if not (
            roles == ["action_theme", "role_complement"]
            and syntax_origin in {
                "ordered_three_role_construction",
                "explicit_role_connector",
            }
            or collective_frame
            or compiled_contract_frame
            or lexical_binary_frame
        ):
            continue
        frames_considered += 1
        selected_names = selected_predicates_by_frame.get(frame_index, set())
        ranked: list[tuple[float, str, int, int]] = []
        for predicate, spec in schema.items():
            if (
                spec.publish_mode == PUBLISH_DISABLED
                or (
                    spec.publish_mode == PUBLISH_REVIEW_ONLY
                    and not include_review_only
                )
                or bool(spec.open_relation_policy.get("fallback_only"))
            ):
                continue
            if not _predicate_allows_source_polarity(spec, frame):
                continue
            # This synthetic source frame joins two grammatical co-subjects.
            # It is only a valid retrieval surface for a Schema relation that
            # explicitly permits reciprocal/collective endpoint semantics.
            # Lexical similarity to the shared verb (for example, an action
            # on a third argument) must never manufacture a peer relation.
            if collective_frame and not (
                spec.symmetric or spec.reciprocal_allowed
            ):
                continue
            if predicate in selected_names:
                # The model has already selected this predicate for the
                # frame. This backstop is contrastive predicate retrieval,
                # not permission to manufacture an endpoint-swapped copy of
                # the same claim.
                continue
            for subject_index, object_index in ((0, 1), (1, 0)):
                if spec.symmetric and subject_index == 1:
                    continue
                if not _predicate_allows_source_role_pair(
                    spec, frame, subject_index, object_index,
                ):
                    continue
                key = (
                    frame_index, predicate, subject_index, object_index,
                )
                if key in existing_keys:
                    continue
                subject = endpoints[subject_index]
                object_ = endpoints[object_index]
                subject_match = _type_match(
                    str(subject.get("type") or ""), spec.domain_types,
                )
                object_match = _type_match(
                    str(object_.get("type") or ""), spec.range_types,
                )
                if subject_match is False or object_match is False:
                    continue
                subject_role = (
                    roles[subject_index] if len(roles) == 2 else ""
                )
                object_role = (
                    roles[object_index] if len(roles) == 2 else ""
                )
                cue_text = " ".join((
                    predicate,
                    *spec.positive_cues,
                    *spec.optional_soft_cues,
                ))
                policy_text = " ".join((
                    spec.semantic_definition,
                    spec.direction_description,
                    spec.subject_role_description,
                    spec.object_role_description,
                ))
                examples = tuple(spec.positive_examples)
                endpoint_query = " ".join((
                    str(subject.get("surface") or ""),
                    relation,
                    str(object_.get("surface") or ""),
                ))
                role_query = " ".join((
                    subject_role,
                    object_role,
                    str(subject.get("surface") or ""),
                    str(object_.get("surface") or ""),
                ))
                relation_score = max(
                    similarity(relation, cue_text),
                    similarity(relation, policy_text),
                )
                evidence_score = best_similarity(evidence, examples)
                endpoint_score = best_similarity(endpoint_query, examples)
                role_score = similarity(role_query, policy_text)
                type_score = (
                    1.0 if subject_match is True and object_match is True
                    else 0.5
                )
                score = (
                    0.36 * relation_score
                    + 0.25 * evidence_score
                    + 0.22 * endpoint_score
                    + 0.12 * role_score
                    + 0.05 * type_score
                )
                candidates_considered += 1
                candidate_threshold = (
                    max(threshold, 0.34)
                    if lexical_binary_frame and not compiled_contract_frame
                    else threshold
                )
                if score < candidate_threshold:
                    below_threshold += 1
                    continue
                ranked.append((score, predicate, subject_index, object_index))
        ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        for score, predicate, subject_index, object_index in ranked[:cap]:
            additions.append({
                "frame_index": frame_index,
                "predicate": predicate,
                "subject_index": subject_index,
                "object_index": object_index,
                "binding_source": "DETERMINISTIC_SEMANTIC_SHORTLIST",
                "shortlist_score": round(score, 6),
                "shortlist_version": (
                    GROUNDED_RELATION_SOURCE_RECALL_SHORTLIST_VERSION
                ),
            })
    return {
        "bindings": additions,
        "frames_considered": frames_considered,
        "candidates_considered": candidates_considered,
        "candidates_below_threshold": below_threshold,
        "version": GROUNDED_RELATION_SOURCE_RECALL_SHORTLIST_VERSION,
    }


def source_frame_schema_binding_prompt(
    inventory: Mapping[str, Any],
    *,
    logical_schema: Mapping[str, Any],
    already_found_bindings: Sequence[Mapping[str, Any]] = (),
    include_review_only: bool = True,
) -> str:
    """Ask for Schema bindings after the source-only frame inventory.

    This is deliberately a context-level operation.  The model sees the
    source-derived frames and the active logical Schema, but no preselected
    endpoint pair.  It may only bind arguments that already exist in a frame;
    the runtime later intersects those bindings with frozen entity mentions
    and still runs the ordinary validator and ordered verifier.
    """

    schema = normalize_predicate_schema(logical_schema)
    raw_predicates = {
        str(item.get("name") or ""): item
        for item in logical_schema.get("relation_types") or []
        if isinstance(item, Mapping) and str(item.get("name") or "")
    }
    predicates = []
    for name, spec in sorted(schema.items()):
        if (
            spec.publish_mode == PUBLISH_DISABLED
            or (
                spec.publish_mode == PUBLISH_REVIEW_ONLY
                and not include_review_only
            )
            or bool(spec.open_relation_policy.get("fallback_only"))
        ):
            continue
        compact = candidate_predicate_prompt_metadata(asdict(spec))
        predicates.append({
            "name": name,
            "semantic_definition": compact.get("semantic_definition", ""),
            "subject_role": compact.get("subject_role_description", ""),
            "object_role": compact.get("object_role_description", ""),
            "domain_types": compact.get("domain_types", []),
            "range_types": compact.get("range_types", []),
            "direction": compact.get("direction_description", ""),
            "symmetric": bool(compact.get("symmetric")),
            "publish_mode": compact.get("publish_mode", ""),
            "optional_soft_cues": compact.get("optional_soft_cues", []),
            "structural_evidence_kinds": list(
                spec.structural_evidence_kinds
            ),
            "structural_role_labels": list(
                raw_predicates.get(name, {}).get(
                    "structural_role_labels"
                ) or []
            ),
            "source_role_pairs": list(spec.source_role_pairs),
            "negation_policy": spec.negation_policy,
        })
    # The inventory intentionally contains unary, negated and modal frames as
    # well as publishable binary frames. Only ASSERTED binary frames can
    # produce an asserted pair binding. Presenting the other frames caused
    # local models to count only eligible frames while returning indexes into
    # the mixed list. Give the binding model a compact eligible-frame index
    # space and map it back to immutable inventory indexes in the parser.
    frames = []
    compact_index_by_source: dict[int, int] = {}
    binding_index = 0
    for source_index, frame in enumerate(inventory.get("frames") or ()):
        if not _source_frame_model_binding_eligible(frame):
            continue
        arguments = list(frame.get("arguments") or [])
        source_roles = list(
            frame.get("parser_inferred_source_roles")
            or frame.get("roles") or []
        )
        source_syntax_origin = str(
            frame.get("parser_inferred_source_syntax")
            or frame.get("deterministic_source_syntax")
            or ""
        )
        frames.append({
            "frame_index": binding_index,
            "relation": str(frame.get("relation") or ""),
            "arguments": arguments,
            # The predicate-open extractor's contract is itself directed:
            # argument 0 is the subject of ``relation`` and argument 1 is its
            # object.  A Schema may still express the semantic inverse, but a
            # binding model must make that exceptional inversion consciously
            # instead of treating source order as arbitrary.
            "open_relation_orientation": {
                "subject_index": 0,
                "object_index": 1,
            },
            "roles": source_roles,
            "argument_kinds": list(frame.get("argument_kinds") or []),
            "entity_heads": list(frame.get("entity_heads") or []),
            "assertion_status": str(frame.get("assertion_status") or ""),
            "evidence": str(frame.get("evidence_text") or "")[:1200],
            "source_structure_kind": str(
                frame.get("deterministic_source_structure") or ""
            ),
            "source_structure_role_label": str(
                frame.get("structural_role_label") or ""
            ),
            "source_syntax_origin": source_syntax_origin,
            "source_syntax_role_projection": (
                f"{str(source_roles[0])}->{str(source_roles[1])}"
                if source_syntax_origin and len(source_roles) == 2
                else ""
            ),
            "explicit_source_negation": (
                _source_relation_has_explicit_negation(
                    frame.get("relation")
                )
            ),
        })
        compact_index_by_source[source_index] = binding_index
        binding_index += 1
    already_found = []
    seen_found: set[tuple[int, str, int, int]] = set()
    for item in already_found_bindings:
        if not isinstance(item, Mapping):
            continue
        source_frame_index = item.get("frame_index")
        subject_index = item.get("subject_index")
        object_index = item.get("object_index")
        predicate = str(item.get("predicate") or "")
        if (
            not isinstance(source_frame_index, int)
            or isinstance(source_frame_index, bool)
            or source_frame_index not in compact_index_by_source
            or not isinstance(subject_index, int)
            or isinstance(subject_index, bool)
            or not isinstance(object_index, int)
            or isinstance(object_index, bool)
            or not predicate
        ):
            continue
        key = (
            compact_index_by_source[source_frame_index],
            predicate,
            subject_index,
            object_index,
        )
        if key in seen_found:
            continue
        seen_found.add(key)
        already_found.append({
            "frame_index": key[0],
            "predicate": key[1],
            "subject_index": key[2],
            "object_index": key[3],
        })
    recovery_prefix = (
        "This is an omission-recovery pass. The already_found_bindings below "
        "are accepted and MUST NOT be repeated. Return only additional entailed "
        "bindings. Recheck every frame against every active predicate in both "
        "argument directions, even when another predicate already describes the "
        "same source fact. "
        if already_found else ""
    )
    return json.dumps({
            "instruction": (
                recovery_prefix
                + "For every ASSERTED semantic frame, inspect every active Schema "
                "predicate and enumerate only bindings whose meaning is entailed by "
                "that frame. A binding is frame_index, predicate, subject_index, "
                "object_index. frame_index refers to the compact eligible-frame list "
                "shown below, not the original inventory list; argument indexes refer "
                "to that frame's complete arguments. Unary, NEGATED, MODAL and "
                "CONDITIONAL frames are omitted because they cannot form an asserted "
                "binary publication. Do "
            "not invent an argument, use a modifier-only participant, connect "
            "coordinated peers, or treat mere co-occurrence as a relation. "
            "Require semantic and role equivalence, not topical similarity or a "
            "shared verb. In particular, submitted content is not a recipient; a "
            "document/location in which something is described is not the state "
            "described by an information carrier; and a checked object maps to a "
            "broader monitoring predicate only when its definition genuinely covers "
            "that checking action. If no active predicate preserves every source "
            "role, emit no binding for that frame. "
            "The open_relation_orientation is semantic, not incidental source "
            "order: the open relation is asserted from argument 0 to argument 1. "
            "Preserve that direction whenever the Schema predicate is equivalent "
            "to the open relation. Reverse it only when the complete Schema "
            "definition is the semantic inverse of the open relation; endpoint "
            "types, whole-first wording, or a preferred canonical layout alone do "
            "not authorize reversal. If a frame presents a whole, container, owner, or source "
            "before one or more explicit member, contained-item, owned-item, or "
            "derived-item arguments, also test the reversed argument order for every "
            "Schema predicate whose subject role is that member/item and whose object "
            "role is that whole/container/owner/source. Emit one binding per explicit "
            "member when the Schema direction requires it; this is role-based "
            "orientation, not a lexical rule. Preserve one binding per coordinated member when the "
            "frame explicitly states the same relation for each member. Reject "
            "NEGATED, MODAL, and CONDITIONAL frames for asserted publication. "
            "When an ASSERTED open frame itself expresses an explicit negative "
            "relation, it may bind only to a Schema predicate whose semantic "
            "definition represents that negative or absence meaning; never map it "
            "to the positive predicate being denied. Must/shall obligations encoded "
            "inside an ASSERTED open frame are source assertions and must be mapped "
            "according to the predicate's normative roles, not rejected as mere "
            "possibility. "
            "A source_structure_kind is parser-validated evidence, not ordinary "
            "adjacency. It may bind only to a predicate declaring that exact "
            "structural_evidence_kind. When source_structure_role_label is present, "
            "it must exactly match one of that predicate's structural_role_labels. "
            "The frame argument order already follows the parser contract (row key "
            "to column value, leading label to governed topic, or structural child "
            "to parent); preserve it unless the Schema relation is symmetric. "
            "A source_syntax_origin instead means the parser only recovered an "
            "explicit coordinated, indented, or multi-role semantic candidate. It "
            "does not require a structural_evidence_kind declaration: bind it by "
            "the open relation, exact arguments, evidence, and Schema roles, and "
            "leave it unbound when those semantics do not match. For a recovered "
            "coordinated_collective_relation_candidate, the two peers are only a "
            "recall candidate: bind them solely when the source predicate itself "
            "semantically asserts a reciprocal/collective relation between those "
            "participants and the Schema defines that same relation. Sharing an "
            "ordinary action or object is never sufficient. For a recovered "
            "multi-role construction, source_syntax_role_projection is semantic "
            "provenance, not decoration. grammatical_subject->action_theme carries "
            "the core predicate; grammatical_subject->role_complement carries the "
            "core predicate plus the explicit connector; and "
            "action_theme->role_complement expresses the connector-introduced role "
            "between those two arguments. Never copy the governing verb's Schema "
            "predicate onto every binary projection merely because the open relation "
            "shares its words. Compare each projected role pair independently with "
            "the Schema subject/object definitions and contrast it with other active "
            "predicates. When a predicate declares non-empty source_role_pairs, that "
            "list is a hard machine-readable contract for these parser-derived "
            "projections; never emit a binding outside it. An empty list means no "
            "compiled role contract is available and does not itself prove a match. "
            "When explicit_source_negation is true, bind only to a predicate whose "
            "negation_policy is allow_explicit_negative_relation. The ordinary "
            "reject_negated policy is a hard contract: never map a denied positive "
            "proposition to that predicate. "
            "A frame can have no Schema binding. Return coverage_complete true "
            "only after considering every supplied frame and every predicate; "
            "return only coverage_complete and bindings."
        ),
        "contract_version": GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
        "frames": frames,
        "predicates": predicates,
        "already_found_bindings": already_found,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_frame_schema_binding_response_schema(
    logical_schema: Mapping[str, Any],
    *,
    include_review_only: bool = True,
) -> dict[str, Any]:
    names = sorted(
        name for name, spec in normalize_predicate_schema(logical_schema).items()
        if spec.publish_mode != PUBLISH_DISABLED
        and (
            include_review_only
            or spec.publish_mode != PUBLISH_REVIEW_ONLY
        )
        and not bool(spec.open_relation_policy.get("fallback_only"))
    )
    return {
        "$id": GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "coverage_complete": {"type": "boolean"},
            "bindings": {
                "type": "array",
                "maxItems": 256,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "frame_index": {
                            "type": "integer", "minimum": 0, "maximum": 63,
                        },
                        "predicate": {"type": "string", "enum": names},
                        "subject_index": {
                            "type": "integer", "minimum": 0, "maximum": 15,
                        },
                        "object_index": {
                            "type": "integer", "minimum": 0, "maximum": 15,
                        },
                    },
                    "required": [
                        "frame_index", "predicate", "subject_index", "object_index",
                    ],
                },
            },
        },
        "required": ["coverage_complete", "bindings"],
    }


def _source_frame_role_pair(
    frame: Mapping[str, Any],
    subject_index: int,
    object_index: int,
) -> str:
    """Return a controlled semantic role pair for parser-derived candidates."""

    origin = str(
        frame.get("parser_inferred_source_syntax")
        or frame.get("deterministic_source_syntax")
        or ""
    )
    if origin not in {
        "ordered_three_role_construction",
        "explicit_role_connector",
        "intervening_predicate_construction",
        "modifier_interrupted_predicate",
        "coordinated_collective_relation_candidate",
        "possessive_reference_construction",
        "passive_agent_postpredicate_construction",
    } and not str(frame.get("parser_inferred_source_syntax") or ""):
        return ""
    roles = [str(value or "") for value in (
        frame.get("parser_inferred_source_roles")
        or frame.get("roles") or ()
    )]
    if not (
        0 <= subject_index < len(roles)
        and 0 <= object_index < len(roles)
        and subject_index != object_index
    ):
        return ""
    pair = f"{roles[subject_index]}->{roles[object_index]}"
    return pair if pair in _VALID_SOURCE_ROLE_PAIRS else ""


def _predicate_allows_source_role_pair(
    spec: PredicateSpec,
    frame: Mapping[str, Any],
    subject_index: int,
    object_index: int,
) -> bool:
    pair = _source_frame_role_pair(frame, subject_index, object_index)
    return bool(
        not pair
        or not spec.source_role_pairs
        or pair in spec.source_role_pairs
    )


def _predicate_allows_source_polarity(
    spec: PredicateSpec,
    frame: Mapping[str, Any],
) -> bool:
    """Enforce the Schema's source-negation policy before model verification."""

    return bool(
        not _source_relation_has_explicit_negation(frame.get("relation"))
        or spec.negation_policy == ALLOW_EXPLICIT_NEGATIVE_RELATION
    )


_BARE_OPEN_RELATION_EXPRESSIONS = frozenset({
    "and", "or", "with", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "of", "to", "for",
    "和", "与", "及", "或", "是", "为", "有", "的", "于",
})


def _displayable_open_relation_expression(value: Any) -> bool:
    """Reject transport/connective labels that cannot describe a graph edge."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    rendered_label = _rendered_edge_relation_label(text)
    if rendered_label:
        text = rendered_label
    if not text or len(text) > 160:
        return False
    normalized = re.sub(
        r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", " ", text.casefold(),
    ).strip()
    if not normalized or normalized in _BARE_OPEN_RELATION_EXPRESSIONS:
        return False
    compact = re.sub(r"\s+", "", normalized)
    has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", compact))
    return len(compact) >= (2 if has_cjk else 3)


def _source_frame_model_binding_eligible(frame: Any) -> bool:
    """Keep model-facing and parser-facing source-frame indexes identical.

    Strict parser structures are bound only by their versioned Schema
    contracts.  The binding parser already rejects every model-selected
    alternative for such a frame, so sending it to the model adds tokens and
    malformed outputs without adding a publishable hypothesis.
    """

    if not isinstance(frame, Mapping):
        return False
    return bool(
        len(list(frame.get("arguments") or ())) >= 2
        and str(frame.get("assertion_status") or "").upper() == "ASSERTED"
        and not str(frame.get("deterministic_source_structure") or "")
        and _displayable_open_relation_expression(frame.get("relation"))
    )


def source_relation_display_expression(value: Any) -> str:
    """Return a bounded human-facing source relation or an empty label.

    The source wording is intentionally not translated into a Schema name.
    Whitespace normalization is presentation-only and deterministic; the
    expression remains auditable against the frozen source-frame payload.
    """

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = _rendered_edge_relation_label(text) or text
    return text if _displayable_open_relation_expression(text) else ""


def _source_relation_identity_expression(value: Any) -> str:
    display = source_relation_display_expression(value)
    return _normalized_atomic_surface(display) if display else ""


def source_frame_open_carrier_bindings(
    inventory: Mapping[str, Any],
    existing_bindings: Sequence[Mapping[str, Any]],
    *,
    logical_schema: Mapping[str, Any],
    max_bindings: int,
) -> dict[str, Any]:
    """Bind otherwise-uncovered exact source frames to one policy carrier.

    This is a GraphRAG-style open-relation fallback, not an all-pairs fallback.
    It can consume only ASSERTED binary frames with frozen occurrence IDs and
    exact evidence.  A carrier is retained as a parallel safety hypothesis
    even when an auto-publish semantic binding exists: normalization suppresses
    it only after that precise binding is independently supported. Review-only
    catch-alls never hide a grounded source assertion. Every returned row
    remains an ordinary hypothesis and must pass the existing deterministic,
    batch and independent verification gates.
    """

    limit = max(0, min(128, int(max_bindings)))
    schema = normalize_predicate_schema(logical_schema)
    carriers = [
        spec for spec in schema.values()
        if bool(spec.open_relation_policy.get("enabled"))
        and bool(spec.open_relation_policy.get("fallback_only"))
        and spec.publish_mode == PUBLISH_AUTO
    ]
    if len(carriers) > 1:
        raise ValueError(
            "logical Schema declares multiple open relation fallback carriers"
        )
    if not carriers or limit == 0:
        return {
            "bindings": [],
            "eligible_frames": 0,
            "skipped_precise_binding": 0,
            "parallel_precise_binding": 0,
            "skipped_low_information": 0,
            "skipped_ungrounded_expression": 0,
            "bindings_capped": 0,
            "carrier": carriers[0].name if carriers else "",
        }
    carrier = carriers[0]
    precise_frames: set[int] = set()
    already_carried: set[int] = set()
    for binding in existing_bindings:
        if not isinstance(binding, Mapping):
            continue
        try:
            frame_index = int(binding.get("frame_index", -1))
        except (TypeError, ValueError):
            continue
        predicate = str(binding.get("predicate") or "")
        spec = schema.get(predicate)
        if spec is None:
            continue
        if spec.name == carrier.name:
            already_carried.add(frame_index)
        elif spec.publish_mode == PUBLISH_AUTO:
            precise_frames.add(frame_index)

    bindings: list[dict[str, Any]] = []
    eligible_candidates: list[tuple[tuple[int, int], int]] = []
    eligible_frames = 0
    skipped_precise = 0
    parallel_precise = 0
    skipped_low_information = 0
    skipped_ungrounded_expression = 0
    bindings_capped = 0
    for frame_index, frame in enumerate(inventory.get("frames") or ()):
        if not isinstance(frame, Mapping):
            continue
        arguments = list(frame.get("arguments") or ())
        occurrence_keys = [
            str(value or "")
            for value in frame.get("argument_occurrence_keys") or ()
        ]
        relation = str(frame.get("relation") or "").strip()
        syntax_origin = str(
            frame.get("parser_inferred_source_syntax")
            or frame.get("deterministic_source_syntax")
            or ""
        )
        inferred_possessive_paraphrase = bool(
            "possessive_reference_construction" in syntax_origin.split("+")
            and not str(frame.get("deterministic_source_syntax") or "")
        )
        strict_structure = bool(
            str(frame.get("deterministic_source_structure") or "")
        )
        source_expression_grounded = (
            _source_relation_expression_is_grounded(
                frame.get("evidence_text"), relation,
            )
        )
        structurally_eligible = bool(
            strict_structure
            or syntax_origin != "coordinated_collective_relation_candidate"
        )
        if not (
            str(frame.get("assertion_status") or "").upper() == "ASSERTED"
            and len(arguments) == 2
            and len(occurrence_keys) == 2
            and all(occurrence_keys)
            and occurrence_keys[0] != occurrence_keys[1]
            and structurally_eligible
            and not inferred_possessive_paraphrase
            and not _source_relation_has_explicit_negation(relation)
            and _displayable_open_relation_expression(relation)
        ):
            skipped_low_information += 1
            continue
        if not strict_structure and not source_expression_grounded:
            skipped_ungrounded_expression += 1
            continue
        eligible_frames += 1
        if frame_index in precise_frames:
            parallel_precise += 1
        if frame_index in already_carried:
            continue
        # A dense context can contain model frames before a long, explicit
        # parser hierarchy.  Taking the first N frames caused the final list
        # children to disappear merely because the model happened to emit
        # other observations first.  Preserve the configured hard bound while
        # ranking parser-proved structure ahead of parser role topology and
        # ordinary model frames.  Order remains stable within each evidence
        # class, so the same immutable inventory always yields the same set.
        priority = 0 if strict_structure else (1 if syntax_origin else 2)
        eligible_candidates.append(((priority, frame_index), frame_index))
    eligible_candidates.sort(key=lambda item: item[0])
    for _priority, frame_index in eligible_candidates[:limit]:
        bindings.append({
            "frame_index": frame_index,
            "predicate": carrier.name,
            "subject_index": 0,
            "object_index": 1,
            "binding_source": "OPEN_EXPRESSION_CARRIER_FALLBACK",
        })
    bindings_capped = max(0, len(eligible_candidates) - len(bindings))
    return {
        "bindings": bindings,
        "eligible_frames": eligible_frames,
        "skipped_precise_binding": skipped_precise,
        "parallel_precise_binding": parallel_precise,
        "skipped_low_information": skipped_low_information,
        "skipped_ungrounded_expression": skipped_ungrounded_expression,
        "bindings_capped": bindings_capped,
        "carrier": carrier.name,
    }


def source_batch_focused_review(
    hypothesis: RelationHypothesis,
    batch_decision: Mapping[str, Any],
    *,
    logical_schema: Mapping[str, Any],
    source_competition_risk: bool = False,
) -> dict[str, Any] | None:
    """Plan a bounded independent single-claim review after batch judging.

    Batching is the efficient primary transport, but two source-first cases
    merit an independent focused call: a weak recall backstop that the batch
    accepted (precision confirmation), and a source-grounded direct/role-bound
    claim that the batch rejected (recall recovery).  The policy is expressed
    entirely through generic provenance and versioned Schema roles; predicate
    names and domain vocabulary never participate.
    """

    metadata = hypothesis.predicate_metadata or {}
    candidate_source = str(metadata.get("candidate_source") or "").upper()
    verdict = str(batch_decision.get("verdict") or "").upper()
    if candidate_source not in {
        "ATOMIC_FRAME_SCHEMA_PROJECTION",
        "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
    }:
        # Pairwise classifier positives are the only publication path that did
        # not originate from a frozen source frame. They are rare after the
        # source-first planner, but a correlated batch mistake would otherwise
        # publish directly. Require one independent full verification (including
        # its predicate-blind endpoint and atomic-frame audits) for every such
        # positive; rejected rows need no expensive recall appeal here.
        if (
            verdict == SUPPORTED
            and str(hypothesis.classifier_prompt_version or "")
            == GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION
        ):
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_PAIRWISE_CLASSIFIER",
                "priority": -2,
                "mandatory": True,
            }
        return None
    schema = normalize_predicate_schema(logical_schema)
    spec = schema.get(hypothesis.predicate)
    if spec is None:
        return None
    binding_source = str(
        metadata.get("source_binding_source") or ""
    ).upper()
    roles = list(metadata.get("source_frame_roles") or [])
    subject_index = metadata.get("source_subject_argument_index")
    object_index = metadata.get("source_object_argument_index")
    role_pair = ""
    if (
        isinstance(subject_index, int)
        and not isinstance(subject_index, bool)
        and isinstance(object_index, int)
        and not isinstance(object_index, bool)
    ):
        role_pair = _source_frame_role_pair(
            {"roles": roles}, subject_index, object_index,
        )
    assertion_status = str(
        metadata.get("source_frame_assertion_status") or ""
    ).upper()
    explicit_negation = bool(
        metadata.get("source_frame_explicit_negation")
    )
    polarity_eligible = bool(
        assertion_status == "ASSERTED"
        and (
            not explicit_negation
            or spec.negation_policy == ALLOW_EXPLICIT_NEGATIVE_RELATION
        )
    )
    reordered_open_orientation = bool(
        isinstance(subject_index, int)
        and not isinstance(subject_index, bool)
        and isinstance(object_index, int)
        and not isinstance(object_index, bool)
        and (subject_index, object_index) != (0, 1)
    )
    if reordered_open_orientation and verdict in {
        SUPPORTED, REJECTED, UNCERTAIN,
    }:
        # The predicate-open inventory already defines argument 0 -> 1 as the
        # direction of its open relation.  A reverse Schema projection can be
        # correct only for a genuine semantic inverse, so it is a mandatory
        # safety audit rather than a budgeted quality sample.
        return {
            "kind": (
                "confirmation" if verdict == SUPPORTED else "recovery"
            ),
            "reason": (
                "BATCH_SUPPORTED_REORDERED_OPEN_ORIENTATION"
                if verdict == SUPPORTED
                else "BATCH_REJECTED_REORDERED_OPEN_ORIENTATION"
            ),
            "priority": -1,
            "source_role_audit": True,
            "mandatory": True,
        }
    projection_contract = _source_syntax_projection_contract(
        roles,
        syntax_origin=str(metadata.get("source_syntax_origin") or ""),
    )
    actor_complement_projection = bool(
        projection_contract.get("projection_kind")
        == "actor_to_connector_complement"
    )
    parser_grounded_open_carrier = bool(
        spec.open_relation_policy.get("enabled")
        and binding_source == "OPEN_EXPRESSION_CARRIER_FALLBACK"
        and str(metadata.get("source_syntax_origin") or "").strip()
        and not str(metadata.get("source_structure_kind") or "").strip()
    )
    model_only_nary_binding = bool(
        binding_source == "MODEL_SOURCE_BINDING"
        and metadata.get("source_frame_other_participants")
        and not str(metadata.get("source_syntax_origin") or "").strip()
    )

    if verdict == SUPPORTED:
        if model_only_nary_binding:
            # A model-only projection from an n-ary frame can silently attach
            # the predicate to the recipient/location instead of the action
            # theme. This is a precision gate, not a budgeted quality sample;
            # coordinated siblings are still coalesced by the grouped audit.
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_MODEL_ONLY_NARY_BINDING",
                "priority": -1,
                "source_role_audit": True,
                "role_audit_only": True,
                "mandatory": True,
            }
        if parser_grounded_open_carrier:
            if not source_competition_risk:
                # This is already the exact open source proposition. The batch
                # verifier has checked its frozen endpoints, direction and
                # evidence; a second full NLI call merely repeats that result.
                return None
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_PARSER_GROUNDED_OPEN_RELATION",
                "priority": 0,
                # The focused ordered verifier already checks exact endpoint
                # roles and both directions.  A separate role-only call adds
                # no independent evidence unless another projection actually
                # competes for the same source slot.
                "source_role_audit": True,
                "role_audit_only": True,
            }
        if actor_complement_projection:
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_ACTOR_COMPLEMENT_PROJECTION",
                "priority": 0,
                "source_role_audit": True,
                "role_audit_only": True,
            }
        if source_competition_risk:
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_COMPETING_SOURCE_PROJECTION",
                "priority": 1,
                "source_role_audit": True,
                "role_audit_only": True,
            }
        if (
            binding_source == "MODEL_SOURCE_BINDING"
            and not spec.open_relation_policy.get("enabled")
        ):
            # Schema projection is a semantic interpretation of an already
            # grounded open fact. One compact equivalence/direction audit is a
            # useful independent precision gate; after it aligns, the complete
            # batch verdict remains the publication authority, so no second
            # full verifier call is needed.
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_DIRECT_SOURCE_BINDING",
                "priority": 1,
                "source_role_audit": True,
                "role_audit_only": True,
            }
        if binding_source == "DETERMINISTIC_SEMANTIC_SHORTLIST":
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_RECALL_SHORTLIST",
                "priority": 2,
                "source_role_audit": True,
                "role_audit_only": True,
            }
        if spec.source_role_pairs and not role_pair:
            return {
                "kind": "confirmation",
                "reason": "BATCH_SUPPORTED_WITHOUT_SOURCE_ROLE_PROJECTION",
                "priority": 3,
            }
        return None

    if verdict not in {REJECTED, UNCERTAIN} or not polarity_eligible:
        return None
    if (
        spec.open_relation_policy.get("enabled")
        and binding_source == "OPEN_EXPRESSION_CARRIER_FALLBACK"
    ):
        parser_theme_complement = bool(
            parser_grounded_open_carrier
            and projection_contract.get("projection_kind")
            == "theme_to_connector_complement"
        )
        if parser_theme_complement:
            # Small local models frequently judge the source sentence
            # correctly in their reason while copying the actor-oriented role
            # booleans from another member of a batch.  A parser-proved
            # theme->connector projection is the exact structural case where
            # that correlated rejection destroys a valid n-ary event role.
            # Keep this bounded to the strong projection contract and require
            # one independent single-claim verifier; the parser still cannot
            # publish the fact by itself.
            return {
                "kind": "recovery",
                "reason": (
                    "BATCH_REJECTED_PARSER_THEME_COMPLEMENT_PROJECTION"
                ),
                "priority": -1,
                "source_role_audit": bool(source_competition_risk),
                "mandatory": True,
            }
        return {
            "kind": "recovery",
            "reason": "BATCH_REJECTED_GROUNDED_OPEN_RELATION",
            "priority": 3,
            "source_role_audit": bool(source_competition_risk),
        }
    if actor_complement_projection:
        return {
            "kind": "recovery",
            "reason": "BATCH_REJECTED_ACTOR_COMPLEMENT_PROJECTION",
            "priority": 4,
            "source_role_audit": True,
        }
    if source_competition_risk:
        return {
            "kind": "recovery",
            "reason": "BATCH_REJECTED_COMPETING_SOURCE_PROJECTION",
            "priority": 4,
            "source_role_audit": True,
        }
    if binding_source == "DETERMINISTIC_SEMANTIC_SHORTLIST":
        return {
            "kind": "recovery",
            "reason": "BATCH_REJECTED_RECALL_SHORTLIST",
            "priority": 5,
            "source_role_audit": True,
        }
    if role_pair and (
        not spec.source_role_pairs or role_pair in spec.source_role_pairs
    ):
        return {
            "kind": "recovery",
            "reason": "BATCH_REJECTED_SCHEMA_ROLE_BOUND_SOURCE",
            "priority": 4,
            "source_role_audit": True,
        }
    if binding_source == "MODEL_SOURCE_BINDING":
        return {
            "kind": "recovery",
            "reason": "BATCH_REJECTED_DIRECT_SOURCE_BINDING",
            "priority": 5,
            "source_role_audit": True,
        }
    return None


def _source_projection_role_shape(
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    """Normalize equivalent extractor-local binary role spellings.

    Deterministic syntax recovery and open-model decomposition use different
    labels for the same ordinary actor/argument slots.  This normalization is
    limited to those protocol-owned labels; it does not interpret document
    vocabulary or a logical predicate.
    """

    aliases = {
        "subject": "source_argument",
        "grammatical_subject": "source_argument",
        "source_argument": "source_argument",
        "object": "target_argument",
        "direct_argument": "target_argument",
        "action_theme": "target_argument",
        "target_argument": "target_argument",
    }
    return tuple(
        aliases.get(
            _normalized_atomic_role(value),
            _normalized_atomic_role(value),
        )
        for value in metadata.get("source_frame_roles") or ()
    )


def _source_projection_slot_is_coordinated(
    left: RelationHypothesis,
    right: RelationHypothesis,
    *,
    slot: str,
    left_span: tuple[int, int],
    right_span: tuple[int, int],
) -> bool:
    """Recognize two claims licensed by one explicit coordinated slot.

    ``slot`` names the endpoint that varies between the claims.  The decision
    uses only extractor protocol fields, exact source spans and occurrence
    identity; document vocabulary and logical predicate names never affect it.
    This is deliberately stronger than generic evidence overlap so a nearby
    adjunct cannot masquerade as another coordinated member.
    """

    if slot not in {"subject", "object"}:
        raise ValueError("coordinated source slot must be subject or object")
    left_metadata = left.predicate_metadata or {}
    right_metadata = right.predicate_metadata or {}
    if left.predicate != right.predicate:
        return False
    normalize_relation = lambda value: re.sub(
        r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", "", str(value or "").casefold(),
    )
    if not normalize_relation(left_metadata.get("source_frame_relation")) or (
        normalize_relation(left_metadata.get("source_frame_relation"))
        != normalize_relation(right_metadata.get("source_frame_relation"))
    ):
        return False
    varying_key = (
        left.subject_occurrence_key if slot == "subject"
        else left.object_occurrence_key
    )
    other_varying_key = (
        right.subject_occurrence_key if slot == "subject"
        else right.object_occurrence_key
    )
    if varying_key == other_varying_key:
        return False
    varying_canonical_field = f"source_{slot}_canonical_id"
    left_varying_canonical = str(
        left_metadata.get(varying_canonical_field) or ""
    )
    right_varying_canonical = str(
        right_metadata.get(varying_canonical_field) or ""
    )
    if (
        left_varying_canonical
        and left_varying_canonical == right_varying_canonical
    ):
        return False
    fixed_slot = "object" if slot == "subject" else "subject"
    fixed_key = (
        left.object_occurrence_key if slot == "subject"
        else left.subject_occurrence_key
    )
    other_fixed_key = (
        right.object_occurrence_key if slot == "subject"
        else right.subject_occurrence_key
    )
    fixed_canonical_field = f"source_{fixed_slot}_canonical_id"
    left_fixed_canonical = str(
        left_metadata.get(fixed_canonical_field) or ""
    )
    right_fixed_canonical = str(
        right_metadata.get(fixed_canonical_field) or ""
    )
    if not (
        fixed_key == other_fixed_key
        or (
            left_fixed_canonical
            and left_fixed_canonical == right_fixed_canonical
        )
    ):
        return False
    if _source_projection_role_shape(
        left_metadata
    ) != _source_projection_role_shape(right_metadata):
        return False
    varying_index_field = f"source_{slot}_argument_index"
    fixed_index_field = f"source_{fixed_slot}_argument_index"
    if (
        left_metadata.get(varying_index_field)
        != right_metadata.get(varying_index_field)
        or left_metadata.get(fixed_index_field)
        != right_metadata.get(fixed_index_field)
    ):
        return False
    left_origin = str(left_metadata.get("source_syntax_origin") or "")
    right_origin = str(right_metadata.get("source_syntax_origin") or "")
    same_parser_slot = bool(
        left_origin
        and left_origin == right_origin
        and left_span == right_span
    )
    varying_surface_field = f"source_{slot}_surface"
    left_surface = _normalized_atomic_surface(
        left_metadata.get(varying_surface_field)
    )
    right_surface = _normalized_atomic_surface(
        right_metadata.get(varying_surface_field)
    )
    left_others = {
        _normalized_atomic_surface(value)
        for value in left_metadata.get(
            "source_frame_other_participants"
        ) or ()
        if _normalized_atomic_surface(value)
    }
    right_others = {
        _normalized_atomic_surface(value)
        for value in right_metadata.get(
            "source_frame_other_participants"
        ) or ()
        if _normalized_atomic_surface(value)
    }
    contained_expansion = bool(
        (
            left_span[0] <= right_span[0]
            and right_span[1] <= left_span[1]
            and right_surface in left_others
        )
        or (
            right_span[0] <= left_span[0]
            and left_span[1] <= right_span[1]
            and left_surface in right_others
        )
    )
    explicit_model_coordination = bool(
        left_surface
        and right_surface
        and (
            (
                left_surface in right_others
                and right_surface in left_others
            )
            or contained_expansion
        )
        and (left_span == right_span or contained_expansion)
    )
    return bool(same_parser_slot or explicit_model_coordination)


def source_projection_coordinated_audit_components(
    claims: Sequence[tuple[RelationHypothesis, PairOpportunity]],
) -> tuple[tuple[str, ...], ...]:
    """Group additive coordinated members for one joint semantic audit.

    A group never crosses a document revision or chunk. Components larger
    than the bounded response contract are deterministically split, retaining
    stable replay identities while avoiding one model call per list member.
    """

    by_id = {
        str(hypothesis.hypothesis_id): (hypothesis, opportunity)
        for hypothesis, opportunity in claims
    }
    graph: dict[str, set[str]] = {}
    ordered = sorted(by_id.items())
    for index, (left_id, (left, left_opportunity)) in enumerate(ordered):
        left_raw_span = (left.predicate_metadata or {}).get(
            "source_frame_evidence_span"
        )
        if not (
            isinstance(left_raw_span, (list, tuple))
            and len(left_raw_span) == 2
        ):
            continue
        left_span = (int(left_raw_span[0]), int(left_raw_span[1]))
        for right_id, (right, right_opportunity) in ordered[index + 1:]:
            if (
                left_opportunity.document_revision_id
                != right_opportunity.document_revision_id
                or left_opportunity.chunk_key != right_opportunity.chunk_key
            ):
                continue
            right_raw_span = (right.predicate_metadata or {}).get(
                "source_frame_evidence_span"
            )
            if not (
                isinstance(right_raw_span, (list, tuple))
                and len(right_raw_span) == 2
            ):
                continue
            right_span = (int(right_raw_span[0]), int(right_raw_span[1]))
            if not any(
                _source_projection_slot_is_coordinated(
                    left,
                    right,
                    slot=slot,
                    left_span=left_span,
                    right_span=right_span,
                )
                for slot in ("subject", "object")
            ):
                continue
            graph.setdefault(left_id, set()).add(right_id)
            graph.setdefault(right_id, set()).add(left_id)
    components: list[tuple[str, ...]] = []
    visited: set[str] = set()
    for root in sorted(graph):
        if root in visited:
            continue
        pending = [root]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            pending.extend(sorted(graph.get(current, ()), reverse=True))
        values = sorted(component)
        for index in range(0, len(values), 16):
            bounded_values = values[index:index + 16]
            # A connected coordination component can contain more than the
            # bounded response contract.  Slicing its sorted IDs is stable,
            # but an articulation claim can land in an earlier slice and
            # leave a later slice whose members are not coordinated with one
            # another.  Emitting that disconnected tail as one joint audit is
            # both semantically wrong and rejected by the anchor invariant.
            # Recompute connected components inside each bounded slice.  Any
            # singleton is deliberately left for the ordinary independent
            # focused audit; it must never borrow coordination through a
            # claim that is absent from its model request.
            bounded_set = set(bounded_values)
            bounded_visited: set[str] = set()
            for bounded_root in bounded_values:
                if bounded_root in bounded_visited:
                    continue
                bounded_pending = [bounded_root]
                bounded_component: set[str] = set()
                while bounded_pending:
                    current = bounded_pending.pop()
                    if current in bounded_visited:
                        continue
                    bounded_visited.add(current)
                    bounded_component.add(current)
                    bounded_pending.extend(sorted(
                        graph.get(current, set()) & bounded_set,
                        reverse=True,
                    ))
                bounded = tuple(sorted(bounded_component))
                if len(bounded) >= 2:
                    components.append(bounded)
    return tuple(components)


_COORDINATED_RECOVERY_CONNECTOR = re.compile(
    r"^\s*(?:(?:[,，]\s*)?(?P<word>and|or|nor|plus|as\s+well\s+as|"
    r"和|及|与|以及|或|或者|并|並|ならびに|及び|または|또는|및)\s*|"
    r"(?P<separator>[,，、;；])\s*)"
    r"(?:(?:the|a|an|le|la|les|un|une|el|los|las|der|die|das)\s*)?$",
    re.IGNORECASE,
)


def _source_projection_coordination_grounding(
    group: Sequence[tuple[RelationHypothesis, PairOpportunity]],
    *,
    varying_slot: str,
) -> dict[str, Any] | None:
    """Ground a coordinated slot in parser topology or explicit source text.

    Model ``other_participants`` fields are recall hints, not sufficient proof
    of coordination: a model can place an actor, temporal adjunct and nested
    noun-phrase member in one participant list.  Publication recovery therefore
    additionally requires either the deterministic coordination parser's
    protocol marker or one contiguous source sequence whose adjacent endpoint
    surfaces are joined only by coordination syntax.  This admits parenthetic
    enumerations and constructions such as ``A, B and C`` / ``甲及乙`` while
    rejecting propagation across ordinary words or clause boundaries.
    """

    if varying_slot not in {"subject", "object"} or len(group) < 2:
        return None
    metadata_values = [
        dict(hypothesis.predicate_metadata or {})
        for hypothesis, _opportunity in group
    ]
    origins = [
        {
            part.strip()
            for part in str(metadata.get("source_syntax_origin") or "").split("+")
            if part.strip()
        }
        for metadata in metadata_values
    ]
    if origins and all("coordinated_argument" in value for value in origins):
        return {
            "kind": "deterministic_coordination_parser",
            "varying_slot": varying_slot,
        }

    evidence_rows: list[tuple[tuple[int, int], str]] = []
    for metadata in metadata_values:
        raw_span = tuple(metadata.get("source_frame_evidence_span") or ())
        raw_text = str(metadata.get("source_frame_evidence_text") or "")
        if (
            len(raw_span) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in raw_span
            )
            or int(raw_span[0]) >= int(raw_span[1])
            or not raw_text
        ):
            return None
        evidence_rows.append((
            (int(raw_span[0]), int(raw_span[1])), raw_text,
        ))

    evidence_span: tuple[int, int] | None = None
    evidence_text = ""
    unique_spans = {span for span, _text in evidence_rows}
    normalized_evidence = {
        re.sub(r"\s+", " ", text).strip()
        for _span, text in evidence_rows
        if re.sub(r"\s+", " ", text).strip()
    }
    if len(unique_spans) == 1 and len(normalized_evidence) == 1:
        evidence_span = next(iter(unique_spans))
        evidence_text = next(iter(normalized_evidence))
    else:
        # Model decompositions often retain an exact source prefix ending at
        # each successive list member.  Accept varying spans only when one
        # frozen evidence row byte-for-byte contains every other row at its
        # declared absolute offset.  This recovers a real coordinated list
        # without treating merely overlapping or paraphrased evidence as one
        # proposition.
        for candidate_span, candidate_text in sorted(
            evidence_rows,
            key=lambda value: (
                -(value[0][1] - value[0][0]), value[0], value[1],
            ),
        ):
            if len(candidate_text) != candidate_span[1] - candidate_span[0]:
                continue
            if not all(
                candidate_span[0] <= span[0] < span[1] <= candidate_span[1]
                and len(text) == span[1] - span[0]
                and candidate_text[
                    span[0] - candidate_span[0]:
                    span[1] - candidate_span[0]
                ] == text
                for span, text in evidence_rows
            ):
                continue
            evidence_span = candidate_span
            evidence_text = re.sub(r"\s+", " ", candidate_text).strip()
            break
    if evidence_span is None or not evidence_text:
        return None
    surface_by_key: dict[str, str] = {}
    for metadata in metadata_values:
        surface = re.sub(
            r"\s+", " ",
            str(metadata.get(f"source_{varying_slot}_surface") or ""),
        ).strip()
        normalized = surface.casefold()
        if surface and normalized:
            surface_by_key.setdefault(normalized, surface)
    if len(surface_by_key) < 2:
        return None

    # Build every exact occurrence because a member can be repeated in a
    # long inventory.  A small graph search then looks for one contiguous
    # connector-only chain covering all distinct varying surfaces; this avoids
    # treating a distant earlier occurrence as the coordinated one.
    nodes: list[tuple[int, int, str, str]] = []
    for normalized, surface in surface_by_key.items():
        pattern = re.compile(re.escape(surface), re.IGNORECASE)
        matches = list(pattern.finditer(evidence_text))
        if len(matches) > 16:
            return None
        nodes.extend(
            (match.start(), match.end(), normalized, surface)
            for match in matches
        )
    if len(nodes) > 128:
        return None
    nodes.sort(key=lambda value: (value[0], value[1], value[2]))
    if {value[2] for value in nodes} != set(surface_by_key):
        return None

    target_keys = frozenset(surface_by_key)

    def connector_edge(
        left: tuple[int, int, str, str],
        right: tuple[int, int, str, str],
    ) -> tuple[bool, bool]:
        if left[1] > right[0] or left[2] == right[2]:
            return False, False
        matched = _COORDINATED_RECOVERY_CONNECTOR.fullmatch(
            evidence_text[left[1]:right[0]]
        )
        if matched is None:
            return False, False
        explicit = bool(
            matched.group("word")
            or matched.group("separator") in {"、", ";", "；"}
        )
        return True, explicit

    def walk(
        node_index: int,
        seen: frozenset[str],
        explicit: bool,
        path: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        if seen == target_keys and explicit:
            return path
        current = nodes[node_index]
        for next_index in range(node_index + 1, len(nodes)):
            candidate = nodes[next_index]
            if candidate[2] in seen:
                continue
            connected, edge_explicit = connector_edge(current, candidate)
            if not connected:
                continue
            if explicit and not edge_explicit:
                continue
            result = walk(
                next_index,
                seen | {candidate[2]},
                explicit or edge_explicit,
                path + (next_index,),
            )
            if result is not None:
                return result
        return None

    for index, node in enumerate(nodes):
        result = walk(index, frozenset({node[2]}), False, (index,))
        if result is None:
            continue
        return {
            "kind": "explicit_source_coordination",
            "varying_slot": varying_slot,
            "surface_sequence": [nodes[value][3] for value in result],
            "evidence_span": list(evidence_span),
        }
    return None


def source_projection_coordinated_recovery_plan(
    claims: Sequence[tuple[RelationHypothesis, PairOpportunity]],
    terminal_attempts: Mapping[str, VerificationAttempt],
) -> dict[str, dict[str, Any]]:
    """Plan deterministic recovery for inconsistent coordinated members.

    A source inventory may expose one asserted relation with an explicit
    coordinated argument slot. If the final verifier supports one member but
    remains uncertain on an otherwise identical member, source topology can
    close that uncertainty. A definite semantic rejection remains authoritative.
    Recover only when every member forms a
    pairwise coordinated clique whose exact evidence rows fit one immutable
    source envelope.  Exact-span subcohorts remain available when an unrelated
    narrower frame is connected only by noisy participant hints.  An ordinary
    supported member may also reconcile a contradictory per-member role audit,
    because all members have the same frozen role shape and direction.  Group
    audit conflicts, unavailable audits, negation and protocol failures remain
    fail-closed.
    """

    by_id = {
        str(hypothesis.hypothesis_id): (hypothesis, opportunity)
        for hypothesis, opportunity in claims
    }
    plans: dict[str, dict[str, Any]] = {}
    cohorts: list[tuple[str, ...]] = []
    seen_cohorts: set[tuple[str, ...]] = set()
    for component in source_projection_coordinated_audit_components(claims):
        members_by_span: dict[tuple[int, int], list[str]] = {}
        valid_component: list[str] = []
        for hypothesis_id in component:
            hypothesis = by_id[hypothesis_id][0]
            raw_span = tuple((hypothesis.predicate_metadata or {}).get(
                "source_frame_evidence_span"
            ) or ())
            if (
                len(raw_span) != 2
                or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in raw_span
                )
                or int(raw_span[0]) >= int(raw_span[1])
            ):
                continue
            exact_span = (int(raw_span[0]), int(raw_span[1]))
            valid_component.append(hypothesis_id)
            members_by_span.setdefault(exact_span, []).append(hypothesis_id)
        candidate_cohorts = [tuple(sorted(valid_component))]
        candidate_cohorts.extend(
            tuple(sorted(member_ids))
            for _exact_span, member_ids in sorted(members_by_span.items())
        )
        for cohort in candidate_cohorts:
            if len(cohort) < 2 or cohort in seen_cohorts:
                continue
            seen_cohorts.add(cohort)
            cohorts.append(cohort)

    for component in cohorts:
        group = [by_id[value] for value in component]
        raw_spans = [tuple(
            (hypothesis.predicate_metadata or {}).get(
                "source_frame_evidence_span"
            ) or ()
        ) for hypothesis, _opportunity in group]
        if any(
            str((hypothesis.predicate_metadata or {}).get(
                "candidate_source"
            ) or "").upper()
            not in {
                "ATOMIC_FRAME_SCHEMA_PROJECTION",
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            }
            or str((hypothesis.predicate_metadata or {}).get(
                "source_frame_assertion_status"
            ) or "").upper() != "ASSERTED"
            or bool((hypothesis.predicate_metadata or {}).get(
                "source_frame_explicit_negation"
            ))
            for hypothesis, _opportunity in group
        ):
            continue

        # A connected component is sufficient for a joint model audit, but
        # deterministic propagation is stricter: every pair must independently
        # occupy the same coordinated source slot.
        pairwise_coordinated = True
        common_varying_slots = {"subject", "object"}
        for index, (left, _left_opportunity) in enumerate(group):
            left_span = (int(raw_spans[index][0]), int(raw_spans[index][1]))
            for right_index in range(index + 1, len(group)):
                right = group[right_index][0]
                right_span = (
                    int(raw_spans[right_index][0]),
                    int(raw_spans[right_index][1]),
                )
                coordinated_slots = {
                    slot
                    for slot in ("subject", "object")
                    if
                    _source_projection_slot_is_coordinated(
                        left,
                        right,
                        slot=slot,
                        left_span=left_span,
                        right_span=right_span,
                    )
                }
                common_varying_slots &= coordinated_slots
                if not coordinated_slots or not common_varying_slots:
                    pairwise_coordinated = False
                    break
            if not pairwise_coordinated:
                break
        if not pairwise_coordinated or len(common_varying_slots) != 1:
            continue
        varying_slot = next(iter(common_varying_slots))
        coordination_grounding = _source_projection_coordination_grounding(
            group,
            varying_slot=varying_slot,
        )
        if coordination_grounding is None:
            continue

        attempts = {
            hypothesis.hypothesis_id: terminal_attempts.get(
                hypothesis.hypothesis_id
            )
            for hypothesis, _opportunity in group
        }

        def eligible_attempt_contract(
            attempt: VerificationAttempt | None,
        ) -> bool:
            if attempt is None:
                return False
            if (
                attempt.verifier_prompt_version
                == GROUNDED_RELATION_VERIFIER_PROMPT_VERSION
            ):
                # The ordinary semantic verifier is authoritative when it can
                # make a definite decision. Coordination may close only its
                # uncertainty; it must never overwrite an explicit rejection.
                return attempt.verdict in {SUPPORTED, UNCERTAIN}
            return bool(
                attempt.verifier_prompt_version
                == GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION
                and attempt.reason_code in {
                    "SOURCE_PROJECTION_ROLE_CONFLICT",
                    "SOURCE_PROJECTION_ROLE_UNCERTAIN",
                }
                and attempt.verdict in {REJECTED, UNCERTAIN}
            )

        if any(
            not eligible_attempt_contract(attempt)
            for attempt in attempts.values()
        ):
            continue
        supported = sorted(
            hypothesis_id for hypothesis_id, attempt in attempts.items()
            if attempt is not None and supported_for_normalization(attempt)
        )
        recoverable = sorted(
            hypothesis_id for hypothesis_id, attempt in attempts.items()
            if attempt is not None
            and (
                (
                    attempt.verifier_prompt_version
                    == GROUNDED_RELATION_VERIFIER_PROMPT_VERSION
                    and attempt.verdict == UNCERTAIN
                )
                or (
                    attempt.verifier_prompt_version
                    == GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION
                    and attempt.reason_code in {
                        "SOURCE_PROJECTION_ROLE_CONFLICT",
                        "SOURCE_PROJECTION_ROLE_UNCERTAIN",
                    }
                    and attempt.verdict in {REJECTED, UNCERTAIN}
                )
            )
        )
        if not supported or not recoverable:
            continue
        anchor_id = supported[0]
        grounding_span = coordination_grounding.get("evidence_span")
        source_span = (
            [int(grounding_span[0]), int(grounding_span[1])]
            if isinstance(grounding_span, (list, tuple))
            and len(grounding_span) == 2
            else [
                min(int(span[0]) for span in raw_spans),
                max(int(span[1]) for span in raw_spans),
            ]
        )
        for hypothesis_id in recoverable:
            plans.setdefault(hypothesis_id, {
                "anchor_hypothesis_id": anchor_id,
                "member_hypothesis_ids": list(component),
                "source_frame_evidence_span": source_span,
                "varying_slot": varying_slot,
                "coordination_grounding": coordination_grounding,
                "policy_version": (
                    GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION
                ),
            })
    return plans


def coordinated_source_projection_recovery_attempt(
    hypothesis: RelationHypothesis,
    *,
    anchor_attempt: VerificationAttempt,
    member_hypothesis_ids: Sequence[str],
    varying_slot: str,
    coordination_grounding: Mapping[str, Any] | None = None,
    attempt_no: int,
) -> VerificationAttempt:
    """Create one auditable support result from a verified coordination anchor."""

    members = tuple(sorted({
        str(value) for value in member_hypothesis_ids if str(value)
    }))
    normalized_varying_slot = str(varying_slot or "")
    if (
        hypothesis.hypothesis_id not in members
        or anchor_attempt.hypothesis_id not in members
        or not supported_for_normalization(anchor_attempt)
        or int(attempt_no) <= int(anchor_attempt.attempt_no)
        or not hypothesis.evidence_spans
        or normalized_varying_slot not in {"subject", "object"}
        or not coordination_grounding
    ):
        raise ValueError("coordinated recovery contract is incomplete")
    return VerificationAttempt(
        attempt_id="relverify_" + _digest((
            hypothesis.hypothesis_id,
            int(attempt_no),
            anchor_attempt.hypothesis_id,
            members,
            GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION,
        )),
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=int(attempt_no),
        verdict=SUPPORTED,
        reason_code="SUPPORTED_COORDINATED_SOURCE_PROJECTION",
        supported_evidence_spans=tuple(hypothesis.evidence_spans),
        independent_call=True,
        verifier_prompt_version=(
            GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION
        ),
        verifier_schema_version=(
            GROUNDED_RELATION_COORDINATED_RECOVERY_SCHEMA_VERSION
        ),
        verifier_parser_version=(
            GROUNDED_RELATION_COORDINATED_RECOVERY_PARSER_VERSION
        ),
        verifier_model_revision=(
            "deterministic:"
            + GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION
        ),
        forward_entailment="ENTAILED",
        reverse_entailment="NOT_ENTAILED",
        subject_role_supported=True,
        object_role_supported=True,
        subject_context_role=(
            "coordinated_source_argument"
            if normalized_varying_slot == "subject"
            else "fixed_source_argument"
        ),
        object_context_role=(
            "coordinated_source_argument"
            if normalized_varying_slot == "object"
            else "fixed_source_argument"
        ),
        subject_role_alignment="VERIFIED_COORDINATION_CLOSURE",
        object_role_alignment="VERIFIED_COORDINATION_CLOSURE",
        forward_role_alignment="VERIFIED_COORDINATION_CLOSURE",
        reverse_role_alignment="VERIFIED_COORDINATION_CLOSURE",
        normalized_subject_occurrence_key=(
            hypothesis.subject_occurrence_key
        ),
        normalized_object_occurrence_key=(
            hypothesis.object_occurrence_key
        ),
        model_request_payload={
            "stage": "coordinated_source_projection_recovery",
            "policy_version": (
                GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION
            ),
            "anchor_hypothesis_id": anchor_attempt.hypothesis_id,
            "member_hypothesis_ids": list(members),
            "varying_slot": normalized_varying_slot,
            "coordination_grounding": dict(coordination_grounding),
            "model_call": False,
        },
        parsed_response={
            "coordination_status": "SUPPORTED_BY_VERIFIED_ANCHOR",
            "coordination_grounding": dict(coordination_grounding),
        },
        token_usage={"model_call": False},
    )


def source_projection_group_anchor_claim(
    claims: Sequence[tuple[RelationHypothesis, PairOpportunity]],
) -> tuple[RelationHypothesis, PairOpportunity]:
    """Choose one exact proposition as a coordinated template anchor.

    Explicit coordination expands one source argument slot into several
    claims.  The relation meaning and ordered role template should therefore
    be challenged once on the narrowest exact proposition before a joint
    member audit can approve the expansion.  This avoids the correlated
    all-positive error local models can make when every coordinated member is
    presented at once, while retaining one-call scaling for long lists.
    """

    ordered = sorted(
        claims, key=lambda value: str(value[0].hypothesis_id),
    )
    if not 2 <= len(ordered) <= 16:
        raise ValueError("source projection group anchor requires 2..16 claims")
    expected = tuple(sorted(
        str(value[0].hypothesis_id) for value in ordered
    ))
    if expected not in source_projection_coordinated_audit_components(
        ordered
    ):
        raise ValueError(
            "source projection group anchor requires one coordinated component"
        )

    def anchor_key(
        value: tuple[RelationHypothesis, PairOpportunity],
    ) -> tuple[int, int, int, str]:
        hypothesis = value[0]
        metadata = hypothesis.predicate_metadata or {}
        raw_span = metadata.get("source_frame_evidence_span")
        span_width = 1 << 30
        if (
            isinstance(raw_span, (list, tuple))
            and len(raw_span) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in raw_span
            )
            and int(raw_span[0]) < int(raw_span[1])
        ):
            span_width = int(raw_span[1]) - int(raw_span[0])
        evidence_text = str(
            metadata.get("source_frame_evidence_text") or ""
        )
        participant_count = len({
            _normalized_atomic_surface(item)
            for item in metadata.get("source_frame_other_participants") or ()
            if _normalized_atomic_surface(item)
        })
        return (
            span_width,
            len(evidence_text) if evidence_text else 1 << 30,
            participant_count,
            str(hypothesis.hypothesis_id),
        )

    return min(ordered, key=anchor_key)


def source_projection_group_anchor_audit_prompt(
    claims: Sequence[tuple[RelationHypothesis, PairOpportunity]],
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> str:
    """Render a single-claim semantic gate for coordinated expansion."""

    anchor_hypothesis, anchor_opportunity = (
        source_projection_group_anchor_claim(claims)
    )
    # Keep this transport byte-for-byte equivalent to the proven single-claim
    # role audit. Even harmless group metadata can prime small local models to
    # copy one decision across a coordinated list—the exact failure this gate
    # exists to prevent. Group membership remains in the persisted request
    # envelope, outside the model-visible prompt.
    return source_projection_role_audit_prompt(
        anchor_hypothesis,
        anchor_opportunity,
        content=content,
        logical_schema=logical_schema,
    )


def source_projection_group_anchor_audit_response_schema() -> dict[str, Any]:
    schema = source_projection_role_audit_response_schema()
    schema["$id"] = GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_SCHEMA_VERSION
    schema["additionalProperties"] = False
    return schema


def parse_source_projection_group_anchor_audit(raw: Any) -> str:
    return parse_source_projection_role_audit(raw)


def source_projection_competition_map(
    hypotheses: Sequence[RelationHypothesis],
    *,
    logical_schema: Mapping[str, Any] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Find source projections that compete for one overlapping role slot.

    Open extractors can copy a governing predicate onto a nearby participant,
    especially around adjuncts and coordinated phrases.  Claims sharing the
    same Schema predicate, normalized source relation and one endpoint slot
    but proposing different fillers for the opposite slot from substantially
    overlapping evidence are therefore marked for single-claim confirmation.
    Keeping the source relation in the slot identity avoids treating every
    independent fact in a dense sentence as mutually exclusive.
    This is a risk signal, not a suppression rule: genuinely coordinated
    arguments can still pass the independent verifier.
    """

    object_slot_groups: dict[
        tuple[str, str, str], list[tuple[RelationHypothesis, int, int]]
    ] = {}
    subject_slot_groups: dict[
        tuple[str, str, str], list[tuple[RelationHypothesis, int, int]]
    ] = {}
    for hypothesis in hypotheses:
        metadata = hypothesis.predicate_metadata or {}
        if str(metadata.get("candidate_source") or "").upper() not in {
            "ATOMIC_FRAME_SCHEMA_PROJECTION",
            "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
        }:
            continue
        relation = re.sub(
            r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+",
            "",
            str(metadata.get("source_frame_relation") or "").casefold(),
        )
        raw_span = metadata.get("source_frame_evidence_span")
        if not relation or not (
            isinstance(raw_span, (list, tuple))
            and len(raw_span) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in raw_span
            )
            and int(raw_span[0]) < int(raw_span[1])
        ):
            continue
        object_slot_groups.setdefault((
            str(hypothesis.predicate),
            relation,
            str(
                metadata.get("source_object_canonical_id")
                or hypothesis.object_occurrence_key
            ),
        ), []).append((hypothesis, int(raw_span[0]), int(raw_span[1])))
        subject_slot_groups.setdefault((
            str(hypothesis.predicate),
            relation,
            str(
                metadata.get("source_subject_canonical_id")
                or hypothesis.subject_occurrence_key
            ),
        ), []).append((hypothesis, int(raw_span[0]), int(raw_span[1])))

    competitors: dict[str, set[str]] = {}
    for slot, groups in (
        ("subject", object_slot_groups),
        ("object", subject_slot_groups),
    ):
        for values in groups.values():
            if len(values) < 2:
                continue
            values.sort(
                key=lambda item: (item[1], item[2], item[0].hypothesis_id)
            )
            for index, (left, left_start, left_end) in enumerate(values):
                for right, right_start, right_end in values[index + 1:]:
                    if right_start >= left_end:
                        break
                    left_metadata = left.predicate_metadata or {}
                    right_metadata = right.predicate_metadata or {}
                    if slot == "subject":
                        left_endpoint_key = left.subject_occurrence_key
                        right_endpoint_key = right.subject_occurrence_key
                        left_canonical = str(
                            left_metadata.get(
                                "source_subject_canonical_id"
                            ) or ""
                        )
                        right_canonical = str(
                            right_metadata.get(
                                "source_subject_canonical_id"
                            ) or ""
                        )
                    else:
                        left_endpoint_key = left.object_occurrence_key
                        right_endpoint_key = right.object_occurrence_key
                        left_canonical = str(
                            left_metadata.get(
                                "source_object_canonical_id"
                            ) or ""
                        )
                        right_canonical = str(
                            right_metadata.get(
                                "source_object_canonical_id"
                            ) or ""
                        )
                    if left_endpoint_key == right_endpoint_key:
                        continue
                    if left_canonical and left_canonical == right_canonical:
                        continue
                    left_origin = str(
                        left_metadata.get("source_syntax_origin") or ""
                    )
                    right_origin = str(
                        right_metadata.get("source_syntax_origin") or ""
                    )
                    varying_index_field = (
                        "source_subject_argument_index"
                        if slot == "subject"
                        else "source_object_argument_index"
                    )
                    fixed_index_field = (
                        "source_object_argument_index"
                        if slot == "subject"
                        else "source_subject_argument_index"
                    )
                    same_parser_slot_expansion = bool(
                        left_origin
                        and left_origin == right_origin
                        and _source_projection_role_shape(left_metadata)
                        == _source_projection_role_shape(right_metadata)
                        and left_metadata.get(varying_index_field)
                        == right_metadata.get(varying_index_field)
                        and left_metadata.get(fixed_index_field)
                        == right_metadata.get(fixed_index_field)
                        and left_metadata.get("source_frame_evidence_span")
                        == right_metadata.get("source_frame_evidence_span")
                    )
                    left_projection = _source_syntax_projection_contract(
                        left_metadata.get("source_frame_roles") or (),
                        syntax_origin=left_origin,
                    )
                    right_projection = _source_syntax_projection_contract(
                        right_metadata.get("source_frame_roles") or (),
                        syntax_origin=right_origin,
                    )
                    distinct_parser_role_projections = bool(
                        left_origin
                        and left_origin == right_origin
                        and left_projection.get("projection_kind")
                        and right_projection.get("projection_kind")
                        and left_projection.get("projection_kind")
                        != right_projection.get("projection_kind")
                        and left_metadata.get("source_frame_evidence_span")
                        == right_metadata.get("source_frame_evidence_span")
                    )
                    varying_surface_field = (
                        "source_subject_surface"
                        if slot == "subject" else "source_object_surface"
                    )
                    left_varying_surface = _normalized_atomic_surface(
                        left_metadata.get(varying_surface_field)
                    )
                    right_varying_surface = _normalized_atomic_surface(
                        right_metadata.get(varying_surface_field)
                    )
                    left_other_participants = {
                        _normalized_atomic_surface(value)
                        for value in left_metadata.get(
                            "source_frame_other_participants"
                        ) or ()
                        if _normalized_atomic_surface(value)
                    }
                    right_other_participants = {
                        _normalized_atomic_surface(value)
                        for value in right_metadata.get(
                            "source_frame_other_participants"
                        ) or ()
                        if _normalized_atomic_surface(value)
                    }
                    contained_coordination_expansion = bool(
                        (
                            left_start <= right_start
                            and right_end <= left_end
                            and right_varying_surface
                            in left_other_participants
                        )
                        or (
                            right_start <= left_start
                            and left_end <= right_end
                            and left_varying_surface
                            in right_other_participants
                        )
                    )
                    explicit_coordinated_slot_expansion = bool(
                        left_varying_surface
                        and right_varying_surface
                        and (
                            (
                                left_varying_surface
                                in right_other_participants
                                and right_varying_surface
                                in left_other_participants
                            )
                            or contained_coordination_expansion
                        )
                        and _source_projection_role_shape(left_metadata)
                        == _source_projection_role_shape(right_metadata)
                        and left_metadata.get(varying_index_field)
                        == right_metadata.get(varying_index_field)
                        and left_metadata.get(fixed_index_field)
                        == right_metadata.get(fixed_index_field)
                        and (
                            left_metadata.get(
                                "source_frame_evidence_span"
                            ) == right_metadata.get(
                                "source_frame_evidence_span"
                            )
                            or contained_coordination_expansion
                        )
                    )
                    if (
                        same_parser_slot_expansion
                        or explicit_coordinated_slot_expansion
                        or distinct_parser_role_projections
                    ):
                        # One parser-proved predicate can govern every member
                        # of an explicit coordinated argument slot. The open
                        # inventory also records every other member for model-
                        # decomposed coordination; requiring mutual membership
                        # prevents an unrelated nearby filler from claiming
                        # this exemption. These projections are additive, not
                        # mutually exclusive.
                        continue
                    overlap = min(left_end, right_end) - max(
                        left_start, right_start,
                    )
                    shorter = min(
                        left_end - left_start, right_end - right_start,
                    )
                    if overlap > 0 and overlap / max(1, shorter) >= 0.8:
                        competitors.setdefault(
                            left.hypothesis_id, set(),
                        ).add(right.hypothesis_id)
                        competitors.setdefault(
                            right.hypothesis_id, set(),
                        ).add(left.hypothesis_id)

    return {
        hypothesis_id: tuple(sorted(values))
        for hypothesis_id, values in competitors.items()
    }


def source_projection_competition_risks(
    hypotheses: Sequence[RelationHypothesis],
    *,
    logical_schema: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Return projection IDs requiring an independent attachment audit."""

    return frozenset(source_projection_competition_map(
        hypotheses, logical_schema=logical_schema,
    ))


def source_projection_competition_components(
    competition_map: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic connected components of competing projections."""

    graph: dict[str, set[str]] = {}
    for raw_key, raw_values in competition_map.items():
        key = str(raw_key)
        if not key:
            continue
        graph.setdefault(key, set())
        for raw_value in raw_values:
            value = str(raw_value)
            if not value or value == key:
                continue
            graph.setdefault(key, set()).add(value)
            graph.setdefault(value, set()).add(key)
    components: list[tuple[str, ...]] = []
    visited: set[str] = set()
    for root in sorted(graph):
        if root in visited:
            continue
        pending = [root]
        component: set[str] = set()
        while pending:
            value = pending.pop()
            if value in visited:
                continue
            visited.add(value)
            component.add(value)
            pending.extend(sorted(graph.get(value, ()), reverse=True))
        if len(component) >= 2:
            components.append(tuple(sorted(component)))
    return tuple(components)


def source_projection_group_audit_prompt(
    claims: Sequence[tuple[RelationHypothesis, PairOpportunity]],
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> str:
    """Render one joint role audit for an overlapping competition group."""

    ordered = sorted(claims, key=lambda value: value[0].hypothesis_id)
    if not 2 <= len(ordered) <= 16:
        raise ValueError("source projection group audit requires 2..16 claims")
    hypotheses = [value[0] for value in ordered]
    candidates: list[dict[str, Any]] = []
    for hypothesis, opportunity in ordered:
        payload = json.loads(source_projection_role_audit_prompt(
            hypothesis,
            opportunity,
            content=content,
            logical_schema=logical_schema,
            competing_projections=[
                value for value in hypotheses
                if value.hypothesis_id != hypothesis.hypothesis_id
            ],
        ))
        candidates.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "source": payload["source"],
            "candidate": payload["candidate"],
            "required_roles": payload["required_roles"],
        })
    return json.dumps(
        {
            "instruction": (
                _SOURCE_PROJECTION_EQUIVALENCE_INSTRUCTION
                + "Jointly compare every candidate against its exact source and "
                "Schema roles. Return one status for every hypothesis_id. Mark "
                "CONFLICT when an endpoint fills a different clause or argument "
                "role, a nearby phrase was copied, the open predicate denotes a "
                "different relation, or the supported orientation is another "
                "candidate. Multiple candidates may be ALIGNED only "
                "when the source independently asserts each one (for example an "
                "explicit coordinated slot); co-occurrence is insufficient. For "
                "a non-symmetric Schema relation, opposite orientations of the "
                "same endpoint pair cannot both be ALIGNED unless the source "
                "explicitly asserts both directed facts. Use UNCERTAIN only for "
                "genuine source ambiguity. Do not inherit a prior verifier "
                "decision. Return one top-level key for every exact "
                "hypothesis_id and map it directly to ALIGNED, CONFLICT, or "
                "UNCERTAIN; return no wrapper or explanation."
            ),
            "contract_version": (
                GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION
            ),
            "candidates": candidates,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_projection_group_audit_response_schema(
    hypothesis_ids: Sequence[str],
) -> dict[str, Any]:
    ids = sorted({str(value) for value in hypothesis_ids if str(value)})
    if not 2 <= len(ids) <= 16:
        raise ValueError("source projection group audit requires 2..16 ids")
    return {
        "$id": GROUNDED_RELATION_SOURCE_GROUP_AUDIT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            hypothesis_id: {
                "type": "string",
                "enum": ["ALIGNED", "CONFLICT", "UNCERTAIN"],
            }
            for hypothesis_id in ids
        },
        "required": ids,
    }


def parse_source_projection_group_audit(
    raw: Any,
    *,
    hypothesis_ids: Sequence[str],
) -> dict[str, str]:
    """Parse an exact, duplicate-free joint projection decision."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "source projection group audit returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "source projection group audit requires statuses"
        )
    expected = {str(value) for value in hypothesis_ids if str(value)}
    statuses = raw.get("statuses")
    if statuses is None and set(map(str, raw)) == expected:
        # Some schema-constrained local runners normalize an array of keyed
        # members directly into a top-level map. Accept that representation
        # only when its keys match the frozen claim universe exactly; extra,
        # missing or abbreviated IDs still fail closed below.
        statuses = raw
    if isinstance(statuses, Mapping):
        normalized_statuses = []
        for key, value in statuses.items():
            if isinstance(value, Mapping):
                embedded = str(value.get("hypothesis_id") or "")
                if embedded and embedded != str(key):
                    raise RelationClassificationError(
                        "source projection group keyed status mismatched"
                    )
                normalized_statuses.append({
                    **dict(value), "hypothesis_id": str(key),
                })
            else:
                normalized_statuses.append({
                    "hypothesis_id": str(key),
                    "projection_status": value,
                })
        statuses = normalized_statuses
    if not isinstance(statuses, list):
        raise RelationClassificationError(
            "source projection group audit requires statuses"
        )
    parsed: dict[str, str] = {}
    for item in statuses:
        if not isinstance(item, Mapping):
            raise RelationClassificationError(
                "source projection group status must be an object"
            )
        hypothesis_id = str(item.get("hypothesis_id") or "")
        status = str(item.get("projection_status") or "").upper()
        if (
            hypothesis_id not in expected
            or hypothesis_id in parsed
            or status not in {"ALIGNED", "CONFLICT", "UNCERTAIN"}
        ):
            raise RelationClassificationError(
                "source projection group status is invalid"
            )
        parsed[hypothesis_id] = status
    if set(parsed) != expected:
        raise RelationClassificationError(
            "source projection group audit omitted a hypothesis"
        )
    return parsed


def atomic_publication_audit_prompt(
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    """Render the final, predicate-focused publication role contract.

    Earlier extraction and verification stages optimize recall and inspect
    several concerns at once.  This deliberately tiny final audit answers one
    question only: does the immutable evidence directly assign the proposed
    ordered endpoints the proposed binary relation?  It is generic across
    document formats, languages and domain vocabularies.
    """

    if not 1 <= len(candidates) <= 16:
        raise ValueError("atomic publication audit requires 1..16 candidates")
    rendered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidates:
        candidate_id = str(raw.get("candidate_id") or "").strip()
        if not candidate_id or candidate_id in seen:
            raise ValueError(
                "atomic publication audit candidate ids must be unique"
            )
        seen.add(candidate_id)
        layer = str(raw.get("relation_layer") or "").strip()
        if layer not in {"schema_predicate", "open_source_expression"}:
            raise ValueError("atomic publication audit layer is invalid")
        subject = str(raw.get("subject") or "").strip()
        object_ = str(raw.get("object") or "").strip()
        evidence = str(raw.get("evidence") or "").strip()
        source_expression = str(
            raw.get("source_expression") or ""
        ).strip()
        canonical_predicate = str(
            raw.get("canonical_predicate") or ""
        ).strip()
        if not all((subject, object_, evidence, canonical_predicate)):
            raise ValueError(
                "atomic publication audit candidate is incomplete"
            )
        schema_contract = raw.get("schema_contract")
        if schema_contract is not None and not isinstance(
            schema_contract, Mapping
        ):
            raise ValueError(
                "atomic publication audit schema contract must be an object"
            )
        resolved_references = raw.get("resolved_reference_endpoints") or []
        if not isinstance(resolved_references, list) or not all(
            isinstance(item, Mapping) for item in resolved_references
        ):
            raise ValueError(
                "atomic publication audit references must be an object array"
            )
        rendered.append({
            "candidate_id": candidate_id,
            "evidence": evidence,
            "proposed_fact": {
                "subject": subject,
                "relation": (
                    source_expression
                    if layer == "open_source_expression"
                    else canonical_predicate
                ),
                "object": object_,
                "ordered": True,
            },
            "relation_layer": layer,
            "exact_source_expression": source_expression,
            "schema_contract": dict(schema_contract or {}),
            "source_order_matches_proposed_fact": bool(
                raw.get("source_order_matches_proposed_fact", True)
            ),
            "endpoint_containment_context": {
                "subject_containing_mentions": [
                    str(value)[:240]
                    for value in (
                        raw.get("subject_containing_mentions") or []
                    )[:6]
                ],
                "object_containing_mentions": [
                    str(value)[:240]
                    for value in (
                        raw.get("object_containing_mentions") or []
                    )[:6]
                ],
            },
            "resolved_reference_grounding": bool(
                raw.get("resolved_reference_grounding")
            ),
            "resolved_reference_endpoints": [
                {
                    "role": str(item.get("role") or ""),
                    "reference_text": str(
                        item.get("reference_text") or ""
                    )[:120],
                    "antecedent_surface": str(
                        item.get("antecedent_surface")
                        or item.get("antecedent_text")
                        or ""
                    )[:240],
                    "reference_span": list(
                        item.get("reference_span") or ()
                    )[:2],
                    "antecedent_span": list(
                        item.get("antecedent_span") or ()
                    )[:2],
                    "local_reference_context": str(
                        item.get("local_reference_context") or ""
                    )[:320],
                }
                for item in resolved_references[:4]
            ],
        })
    return json.dumps(
        {
            "instruction": (
                "Audit each candidate independently using only its evidence. "
                "Return ALIGNED only when the evidence directly states the "
                "complete ordered binary proposition subject -- relation --> "
                "object. The subject and object must themselves fill those "
                "semantic roles; an entity nested inside a longer actor, "
                "theme, modifier, example, comparison, list, formula, figure "
                "label, or nearby clause is not the complete endpoint. A "
                "nonempty endpoint_containment_context is explicit parser "
                "evidence that the proposed endpoint is nested inside a "
                "longer frozen mention; return ALIGNED only if the shorter "
                "endpoint is nevertheless independently the complete role. A "
                "determiner, preposition, conjunction, comparison marker, "
                "example marker, discourse marker, or merely co-occurring "
                "phrase is not a binary relation. Return CONFLICT when the "
                "relation belongs to a third party, endpoint roles or direction "
                "are wrong, an endpoint is truncated, or the evidence expresses "
                "a different proposition. For schema_predicate candidates, "
                "also require semantic equivalence to schema_contract.definition "
                "and exact compliance with its subject_role, object_role and "
                "direction; topical similarity, compatible types, implication "
                "and a shared verb are insufficient. For open_source_expression "
                "candidates, require that the exact source expression connects "
                "the supplied endpoints in that order. A resolved local pronoun "
                "is allowed only when resolved_reference_grounding is true, "
                "resolved_reference_endpoints maps that role to the supplied "
                "endpoint, and the evidence directly assigns the pronoun the "
                "required role. A demonstrative or determiner can be only the "
                "first token of a larger local noun phrase; use "
                "local_reference_context and require the antecedent to denote "
                "that complete phrase, never the closed-class token alone. "
                "A short adjective, modifier, or action before a coordinated "
                "longer noun phrase is not a complete endpoint merely because "
                "the omitted head can be inferred from the neighbouring "
                "conjunct. "
                "The antecedent text may be outside the local "
                "evidence; do not reject it merely for that reason when the "
                "explicit mapping is present. "
                "Use UNCERTAIN only for genuine grammatical ambiguity. Do not "
                "inherit any earlier verifier decision. Return one top-level "
                "key for every exact candidate_id, mapped directly to ALIGNED, "
                "CONFLICT, or UNCERTAIN, with no wrapper or explanation."
            ),
            "contract_version": (
                GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
            ),
            "candidates": rendered,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


_ATOMIC_PUBLICATION_COORDINATOR = re.compile(
    r"^(?:\u3001|\u4ee5\u53ca|\u548c|\u4e0e|\u53ca|\u6216|and|or)",
    re.IGNORECASE,
)
_ATOMIC_PUBLICATION_GENERIC_ENDPOINT_TYPES = frozenset({
    "", "*", "entity", "genericentity", "generic", "unknown",
    "unspecified", "\u901a\u7528\u5b9e\u4f53", "\u672a\u77e5", "\u672a\u6307\u5b9a",
})


def _atomic_publication_compact(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", str(value or ""))
        if not character.isspace()
    )


def _atomic_publication_ordered_geometry(
    candidate: Mapping[str, Any],
) -> tuple[str, str, str, str, int, int, int] | None:
    """Locate the shortest ordered subject/relation/object source geometry."""

    evidence = _atomic_publication_compact(candidate.get("evidence"))
    subject = _atomic_publication_compact(candidate.get("subject"))
    relation = _atomic_publication_compact(
        candidate.get("source_expression")
    )
    object_ = _atomic_publication_compact(candidate.get("object"))
    if not all((evidence, subject, relation, object_)):
        return None

    def occurrences(value: str) -> list[int]:
        output: list[int] = []
        start = 0
        while True:
            index = evidence.find(value, start)
            if index < 0:
                return output
            output.append(index)
            start = index + 1

    candidates: list[tuple[int, int, int]] = []
    for subject_start in occurrences(subject):
        subject_end = subject_start + len(subject)
        for relation_start in occurrences(relation):
            relation_end = relation_start + len(relation)
            if relation_start < subject_end:
                continue
            for object_start in occurrences(object_):
                if relation_end <= object_start:
                    candidates.append((
                        subject_start, relation_start, object_start,
                    ))
    if not candidates:
        return None
    subject_start, relation_start, object_start = min(
        candidates,
        key=lambda item: (
            item[2] + len(object_) - item[0], item,
        ),
    )
    return (
        evidence, subject, relation, object_,
        subject_start, relation_start, object_start,
    )


def atomic_publication_deterministic_rejection_reason(
    candidate: Mapping[str, Any],
) -> str:
    """Reject structurally incomplete endpoints before model judgment.

    A frequent cross-domain construction coordinates a short modifier with a
    longer noun phrase while eliding the latter's head in the first conjunct
    (the equivalent of "reliable and low-quality data sources").  A language
    model may correctly recover the intended phrase yet accidentally approve
    the frozen short surface as a standalone graph node.  This gate does not
    synthesize the omitted phrase; it conservatively withholds only a short,
    generic CJK endpoint followed by a substantially longer conjunct.
    """

    if str(candidate.get("relation_layer") or "") != (
        "open_source_expression"
    ):
        return ""
    geometry = _atomic_publication_ordered_geometry(candidate)
    if geometry is None:
        return ""
    (
        evidence, subject, _relation, object_,
        subject_start, relation_start, object_start,
    ) = geometry

    def generic_type(value: Any) -> bool:
        key = "".join(
            character.casefold()
            for character in unicodedata.normalize(
                "NFKC", str(value or ""),
            )
            if character.isalnum() or character == "*"
        )
        return key in _ATOMIC_PUBLICATION_GENERIC_ENDPOINT_TYPES

    def elided_head(
        endpoint: str,
        following: str,
        endpoint_type: Any,
    ) -> bool:
        cjk = "".join(re.findall(r"[\u3400-\u9fff]", endpoint))
        if (
            not 1 <= len(cjk) <= 3
            or cjk != endpoint
            or not generic_type(endpoint_type)
        ):
            return False
        connector = _ATOMIC_PUBLICATION_COORDINATOR.match(following)
        if connector is None:
            return False
        right = following[connector.end():]
        right = re.split(r"[\uff0c,\u3002.!\uff01\uff1f?\uff1b;\uff1a:]", right, maxsplit=1)[0]
        right_cjk = "".join(re.findall(r"[\u3400-\u9fff]", right))
        return len(right_cjk) >= len(cjk) + 2

    subject_following = evidence[
        subject_start + len(subject):relation_start
    ]
    if elided_head(
        subject, subject_following,
        candidate.get("subject_entity_type"),
    ):
        return "coordinated_elliptical_subject_endpoint"
    object_following = evidence[object_start + len(object_):]
    if elided_head(
        object_, object_following,
        candidate.get("object_entity_type"),
    ):
        return "coordinated_elliptical_object_endpoint"
    return ""


def atomic_publication_audit_requirement(
    candidate: Mapping[str, Any],
) -> tuple[bool, str]:
    """Route only semantically risky facts to the final model audit.

    Schema predicates always need semantic equivalence review.  An open source
    assertion, by contrast, already preserves the exact predicate wording and
    does not claim normalized inference semantics.  When its complete subject,
    exact predicate and complete object occur in a short, ordered local span,
    that immutable geometry is a stronger and cheaper publication contract
    than asking the same model to restate the extraction decision.  Everything
    else remains fail-closed behind the focused audit.
    """

    if str(candidate.get("relation_layer") or "") != (
        "open_source_expression"
    ):
        return True, "schema_semantic_mapping"
    if candidate.get("resolved_reference_grounding"):
        return True, "resolved_reference_role_mapping"
    if candidate.get("source_order_matches_proposed_fact") is not True:
        return True, "source_order_not_proven"
    if (
        candidate.get("subject_containing_mentions")
        or candidate.get("object_containing_mentions")
    ):
        return True, "endpoint_nested_in_source_mention"

    def compact(value: Any) -> str:
        return "".join(
            character.casefold()
            for character in unicodedata.normalize("NFKC", str(value or ""))
            if not character.isspace()
        )

    evidence = compact(candidate.get("evidence"))
    subject = compact(candidate.get("subject"))
    relation = compact(candidate.get("source_expression"))
    object_ = compact(candidate.get("object"))
    if not all((evidence, subject, relation, object_)):
        return True, "incomplete_open_source_contract"

    def occurrences(value: str) -> list[int]:
        output: list[int] = []
        start = 0
        while True:
            index = evidence.find(value, start)
            if index < 0:
                return output
            output.append(index)
            start = index + 1

    candidates: list[tuple[int, int, int]] = []
    for subject_start in occurrences(subject):
        subject_end = subject_start + len(subject)
        for relation_start in occurrences(relation):
            relation_end = relation_start + len(relation)
            if relation_start < subject_end:
                continue
            for object_start in occurrences(object_):
                if relation_end <= object_start:
                    candidates.append((
                        subject_start, relation_start, object_start,
                    ))
    if not candidates:
        return True, "ordered_source_geometry_missing"
    subject_start, relation_start, object_start = min(
        candidates,
        key=lambda item: (
            item[2] + len(object_) - item[0], item,
        ),
    )
    subject_gap = evidence[
        subject_start + len(subject):relation_start
    ]
    object_gap = evidence[
        relation_start + len(relation):object_start
    ]
    if (
        len(subject_gap) > 32
        or len(object_gap) > 32
        or len(subject_gap) + len(object_gap) > 48
        or re.search(r"[。！？!?；;]", subject_gap + object_gap)
    ):
        return True, "nonlocal_source_role_geometry"
    lexical_object_length = sum(
        character.isalnum() for character in object_
    )
    object_tail = evidence[object_start + len(object_):]
    if (
        lexical_object_length <= 3
        and re.match(
            r"^(?:、|和|与|及|或|以及|and|or)",
            object_tail,
            flags=re.IGNORECASE,
        )
    ):
        return True, "short_coordinated_endpoint"
    # Ordered lexical geometry is a useful precondition, but it does not prove
    # that each surface is the complete semantic role.  In particular, a
    # nearby subordinate clause, coordinated predicate, or elided noun head
    # can preserve subject--verb--object order while changing the proposition.
    # The final audit is only a handful of batched calls over facts that have
    # already survived the expensive recall pipeline, so bypassing it saves
    # little and measurably lowers publication precision.
    return True, "open_source_semantic_role_confirmation"


def atomic_publication_audit_response_schema(
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    ids = sorted({str(value) for value in candidate_ids if str(value)})
    if not 1 <= len(ids) <= 16 or len(ids) != len(candidate_ids):
        raise ValueError(
            "atomic publication audit requires 1..16 unique candidate ids"
        )
    return {
        "$id": GROUNDED_RELATION_PUBLICATION_AUDIT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            candidate_id: {
                "type": "string",
                "enum": ["ALIGNED", "CONFLICT", "UNCERTAIN"],
            }
            for candidate_id in ids
        },
        "required": ids,
    }


def parse_atomic_publication_audit(
    raw: Any,
    *,
    candidate_ids: Sequence[str],
) -> dict[str, str]:
    """Parse an exact keyed publication decision without verdict aliases."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "atomic publication audit returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "atomic publication audit response must be an object"
        )
    expected = {str(value) for value in candidate_ids if str(value)}
    if len(expected) != len(candidate_ids) or set(map(str, raw)) != expected:
        raise RelationClassificationError(
            "atomic publication audit response key set mismatched"
        )
    parsed: dict[str, str] = {}
    for candidate_id, value in raw.items():
        status = str(value or "").strip().upper()
        if status not in {"ALIGNED", "CONFLICT", "UNCERTAIN"}:
            raise RelationClassificationError(
                "atomic publication audit status is invalid"
            )
        parsed[str(candidate_id)] = status
    return parsed


def source_role_audit_conflict_is_terminal(
    focused_review: Mapping[str, Any],
) -> bool:
    """Keep a selected source-role audit as a fail-closed publication gate."""

    _ = focused_review
    return True


def source_projection_role_audit_response_schema() -> dict[str, Any]:
    """Return the tiny source-role attachment audit contract."""

    return {
        "$id": GROUNDED_RELATION_SOURCE_ROLE_AUDIT_SCHEMA_VERSION,
        "type": "object",
        "properties": {
            "projection_status": {
                "type": "string",
                "enum": ["ALIGNED", "CONFLICT", "UNCERTAIN"],
            },
        },
        "required": ["projection_status"],
    }


def source_projection_role_audit_prompt(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    competing_projections: Sequence[RelationHypothesis] = (),
) -> str:
    """Audit one risky open-frame projection without repeating final NLI."""

    spec = normalize_predicate_schema(logical_schema).get(
        hypothesis.predicate
    )
    if spec is None:
        raise ValueError("source projection role audit requires Schema policy")
    metadata = hypothesis.predicate_metadata or {}
    raw_span = metadata.get("source_frame_evidence_span")
    if not (
        isinstance(raw_span, (list, tuple))
        and len(raw_span) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_span
        )
    ):
        raise ValueError("source projection role audit span is invalid")
    start, end = int(raw_span[0]), int(raw_span[1])
    if not (
        opportunity.context_span[0] <= start < end
        <= opportunity.context_span[1]
        and end <= len(content)
        and start <= opportunity.left_span[0] < opportunity.left_span[1] <= end
        and start <= opportunity.right_span[0] < opportunity.right_span[1] <= end
    ):
        raise ValueError(
            "source projection role audit does not cover exact endpoints"
        )
    frame_text = str(metadata.get("source_frame_evidence_text") or "")
    if not frame_text or content[start:end] != frame_text:
        raise ValueError("source projection role audit text mismatched source")
    subject = _occurrence_details(
        opportunity, hypothesis.subject_occurrence_key, content,
    )
    object_ = _occurrence_details(
        opportunity, hypothesis.object_occurrence_key, content,
    )
    other_participants = [
        str(value).strip()
        for value in (
            metadata.get("source_frame_other_participants") or ()
        )
        if str(value).strip()
    ]
    competing_candidates: list[dict[str, Any]] = []
    for value in competing_projections:
        competitor_metadata = value.predicate_metadata or {}
        for field in (
            "source_subject_surface", "source_object_surface",
        ):
            competitor_surface = str(
                competitor_metadata.get(field) or ""
            ).strip()
            if (
                competitor_surface
                and competitor_surface
                not in {
                    str(metadata.get("source_subject_surface") or "").strip(),
                    str(metadata.get("source_object_surface") or "").strip(),
                }
                and competitor_surface not in other_participants
            ):
                other_participants.append(competitor_surface)
        competitor_roles = list(
            competitor_metadata.get("source_frame_roles") or ()
        )
        competitor_subject = str(
            competitor_metadata.get("source_subject_surface") or ""
        ).strip()
        competitor_object = str(
            competitor_metadata.get("source_object_surface") or ""
        ).strip()
        if competitor_subject and competitor_object:
            competing_candidates.append({
                "subject": competitor_subject,
                "schema_relation": value.predicate,
                "object": competitor_object,
                "open_predicate": str(
                    competitor_metadata.get("source_frame_relation") or ""
                ),
                "source_roles": competitor_roles,
                "subject_argument_index": competitor_metadata.get(
                    "source_subject_argument_index"
                ),
                "object_argument_index": competitor_metadata.get(
                    "source_object_argument_index"
                ),
                "projection_kind": _source_syntax_projection_contract(
                    competitor_roles,
                    syntax_origin=str(
                        competitor_metadata.get("source_syntax_origin") or ""
                    ),
                ).get("projection_kind", ""),
            })
    source_roles = list(metadata.get("source_frame_roles") or [])
    projection_contract = _source_syntax_projection_contract(
        source_roles,
        syntax_origin=str(metadata.get("source_syntax_origin") or ""),
    )
    return json.dumps(
        {
            "instruction": (
                _SOURCE_PROJECTION_EQUIVALENCE_INSTRUCTION
                + "When competing_bindings are supplied, contrast the candidate "
                "against those complete subject-predicate-object assignments. "
                "Apply candidate.projection_contract as the exact binary role "
                "semantics. In particular, an action_theme subject is not required "
                "to be the grammatical actor. Distinct parser-proved projection "
                "kinds from one n-ary frame are additive rather than alternatives. "
                "Do not mark two alternatives ALIGNED merely because all their "
                "entities occur in the same clause: identify which entity fills "
                "the requested source argument slot. "
                "Return CONFLICT if the relation actually belongs to an omitted "
                "third participant, is copied from a nearby "
                "participant, or substitutes possession or association for the "
                "requested role. Return UNCERTAIN only for genuine grammatical "
                "ambiguity. Output projection_status only."
            ),
            "source": frame_text,
            "candidate": {
                "subject": subject["surface"],
                "schema_relation": hypothesis.predicate,
                "object": object_["surface"],
                "open_predicate": str(
                    metadata.get("source_frame_relation") or ""
                ),
                "source_roles": source_roles,
                "subject_argument_index": metadata.get(
                    "source_subject_argument_index"
                ),
                "object_argument_index": metadata.get(
                    "source_object_argument_index"
                ),
                "projection_kind": projection_contract.get(
                    "projection_kind", ""
                ),
                "projection_contract": projection_contract,
                "other_participants": other_participants[:12],
            },
            "competing_bindings": competing_candidates[:8],
            "required_roles": {
                "definition": spec.semantic_definition,
                "subject": spec.subject_role_description,
                "object": spec.object_role_description,
                "direction": spec.direction_description,
                "symmetric": bool(spec.symmetric),
                "positive_examples": list(spec.positive_examples[:3]),
                "negative_examples": list(spec.negative_examples[:3]),
            },
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def parse_source_projection_role_audit(raw: Any) -> str:
    """Parse a source-role audit without semantic verdict aliases."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "source projection role audit returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "source projection role audit response must be an object"
        )
    result = str(
        raw.get("projection_status") or raw.get("result") or ""
    ).strip().upper()
    if result not in {"ALIGNED", "CONFLICT", "UNCERTAIN"}:
        raise RelationClassificationError(
            "source projection role audit result is invalid"
        )
    return result


def _source_structure_binding_audit_payload(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and render one parser-proved source-structure binding."""

    spec = normalize_predicate_schema(logical_schema).get(
        hypothesis.predicate
    )
    if spec is None:
        raise ValueError("source structure audit requires Schema policy")
    metadata = hypothesis.predicate_metadata or {}
    binding_source = str(
        metadata.get("source_binding_source") or ""
    ).upper()
    open_structure_carrier = binding_source == (
        "DETERMINISTIC_SOURCE_STRUCTURE_OPEN_CARRIER"
    )
    if binding_source not in {
        "DETERMINISTIC_SOURCE_STRUCTURE",
        "DETERMINISTIC_SOURCE_STRUCTURE_OPEN_CARRIER",
    }:
        raise ValueError(
            "source structure audit requires deterministic binding"
        )
    structure_kind = str(
        metadata.get("source_structure_kind") or ""
    )
    role_contracts = {
        "table_header_assignment": ("row_key", "column_value"),
        "line_identifier_binding": ("source_label", "bound_topic"),
        "list_parent_child": ("structural_child", "structural_parent"),
        "section_label_parent_child": (
            "structural_child", "structural_parent",
        ),
    }
    expected_roles = role_contracts.get(structure_kind)
    roles = tuple(str(value) for value in (
        metadata.get("source_frame_roles") or ()
    ))
    if expected_roles is None or roles != expected_roles:
        raise ValueError(
            "source structure audit contract is not Schema-authorized"
        )
    role_label = str(
        metadata.get("source_structure_role_label") or ""
    ).strip()
    raw_policy = next((
        value for value in logical_schema.get("relation_types") or ()
        if isinstance(value, Mapping)
        and str(value.get("name") or "") == hypothesis.predicate
    ), {})
    declared_labels = {
        re.sub(r"\s+", " ", str(value or "").strip()).casefold()
        for value in raw_policy.get("structural_role_labels") or ()
        if str(value or "").strip()
    }
    normalized_role_label = re.sub(
        r"\s+", " ", role_label,
    ).strip().casefold()
    if open_structure_carrier:
        allowed_sources = {
            str(value)
            for value in spec.open_relation_policy.get(
                "allowed_candidate_sources", ()
            )
        }
        frame_relation = re.sub(
            r"\s+", " ",
            str(metadata.get("source_frame_relation") or ""),
        ).strip().casefold()
        candidate_source = str(
            metadata.get("candidate_source") or ""
        )
        if not (
            structure_kind == "table_header_assignment"
            and normalized_role_label
            and frame_relation == f"column role: {normalized_role_label}"
            and spec.publish_mode == PUBLISH_AUTO
            and bool(spec.open_relation_policy.get("enabled"))
            and bool(spec.open_relation_policy.get("fallback_only"))
            and candidate_source in {
                "ATOMIC_FRAME_SCHEMA_PROJECTION",
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            }
            and candidate_source in allowed_sources
        ):
            raise ValueError(
                "source table open carrier contract is not authorized"
            )
    elif (
        structure_kind not in spec.structural_evidence_kinds
        or (
            structure_kind == "table_header_assignment"
            and (
                not normalized_role_label
                or normalized_role_label not in declared_labels
            )
        )
    ):
        raise ValueError(
            "source table audit role label is not Schema-authorized"
        )
    raw_span = metadata.get("source_frame_evidence_span")
    if not (
        isinstance(raw_span, (list, tuple))
        and len(raw_span) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_span
        )
    ):
        raise ValueError("source structure audit span is invalid")
    start, end = int(raw_span[0]), int(raw_span[1])
    frame_text = str(metadata.get("source_frame_evidence_text") or "")
    if not (
        tuple(
            (int(span.start), int(span.end))
            for span in hypothesis.evidence_spans
        ) == ((start, end),)
        and
        opportunity.context_span[0] <= start < end
        <= opportunity.context_span[1]
        and end <= len(content)
        and content[start:end] == frame_text
        and start <= opportunity.left_span[0]
        < opportunity.left_span[1] <= end
        and start <= opportunity.right_span[0]
        < opportunity.right_span[1] <= end
    ):
        raise ValueError(
            "source structure audit evidence is not exact"
        )
    subject = _occurrence_details(
        opportunity, hypothesis.subject_occurrence_key, content,
    )
    object_ = _occurrence_details(
        opportunity, hypothesis.object_occurrence_key, content,
    )
    return {
        "instruction": (
            "Audit only the frozen parser-proved structural binding. The parser "
            "has already identified the exact structure kind, endpoint roles, "
            "row/group boundary, and (when present) column or field label. Decide "
            "whether that exact structural role instantiates the requested Schema "
            "meaning and direction. ALIGNED only for this exact subject and object. "
            "Return CONFLICT for a sibling column/item, reversed role, incompatible "
            "label, or different Schema meaning. Do not demand a lexical verb and "
            "do not reinterpret neighboring rows. Return UNCERTAIN only when the "
            "supplied structural contract itself is semantically ambiguous. Output "
            "structure_status only."
        ),
        "contract_version": (
            GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PROMPT_VERSION
        ),
        "source": {
            "span": [start, end],
            "text": frame_text,
            "structure_kind": structure_kind,
            "role_label": role_label,
            "parser_roles": list(roles),
        },
        "candidate": {
            "subject": subject["surface"],
            "object": object_["surface"],
        },
        "required_relation": {
            "definition": spec.semantic_definition,
            "subject_role": spec.subject_role_description,
            "object_role": spec.object_role_description,
            "direction": spec.direction_description,
            "symmetric": bool(spec.symmetric),
            "authorized_structure_kind": structure_kind,
            "authorized_role_label": role_label,
        },
    }


def source_structure_binding_contract_authorized(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> bool:
    """Return whether a cross-scope structure claim has the full contract.

    Table headers and hierarchical labels necessarily span more than one
    lexical sentence/line.  They may bypass only the lexical atomic-scope gate,
    never the exact parser role, Schema capability, endpoint, or evidence-span
    checks enforced by the structural audit payload.
    """

    try:
        _source_structure_binding_audit_payload(
            hypothesis,
            opportunity,
            content=content,
            logical_schema=logical_schema,
        )
    except (TypeError, ValueError):
        return False
    return True


def source_structure_binding_audit_prompt(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> str:
    return json.dumps(
        _source_structure_binding_audit_payload(
            hypothesis,
            opportunity,
            content=content,
            logical_schema=logical_schema,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_structure_binding_audit_response_schema() -> dict[str, Any]:
    return {
        "$id": GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "structure_status": {
                "type": "string",
                "enum": ["ALIGNED", "CONFLICT", "UNCERTAIN"],
            },
        },
        "required": ["structure_status"],
    }


def parse_source_structure_binding_audit(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "source structure audit returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "source structure audit response must be an object"
        )
    result = str(raw.get("structure_status") or "").strip().upper()
    if result not in {"ALIGNED", "CONFLICT", "UNCERTAIN"}:
        raise RelationClassificationError(
            "source structure audit result is invalid"
        )
    return result


_NEGATION_SCOPE_REJECTION_PATTERN = re.compile(
    r"(?:negat(?:e|ed|es|ing|ion|ive)|negative[_ -]?scope|"
    r"polarity|否定|被否定|n[ée]gation|negaci[oó]n|"
    r"verneinung|negazione|nega[cç][aã]o)",
    re.IGNORECASE,
)


def source_frame_polarity_review_eligible(
    hypothesis: RelationHypothesis,
    terminal_attempt: VerificationAttempt,
) -> bool:
    """Return whether a narrow grammatical-polarity appeal is warranted.

    This is intentionally much stricter than ordinary secondary routing.  It
    applies only when the final verifier accepted both endpoint roles but
    rejected a source-grounded frame specifically because of negation or
    polarity scope, while the immutable source inventory recorded that exact
    frame as asserted and not explicitly negated.  The appeal can request an
    independent scope audit; it cannot publish a fact by itself.
    """

    if (
        str(terminal_attempt.hypothesis_id or "")
        != str(hypothesis.hypothesis_id or "")
        or str(terminal_attempt.verdict or "").upper() != REJECTED
        or terminal_attempt.subject_role_supported is not True
        or terminal_attempt.object_role_supported is not True
    ):
        return False
    metadata = hypothesis.predicate_metadata or {}
    if str(metadata.get("candidate_source") or "").upper() not in {
        "ATOMIC_FRAME_SCHEMA_PROJECTION",
        "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
    }:
        return False
    if (
        str(metadata.get("source_frame_assertion_status") or "").upper()
        != "ASSERTED"
        or bool(metadata.get("source_frame_explicit_negation"))
        or not str(metadata.get("source_frame_relation") or "").strip()
        or not str(metadata.get("source_frame_evidence_text") or "").strip()
    ):
        return False
    span = metadata.get("source_frame_evidence_span")
    if not (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
        and int(span[0]) >= 0
        and int(span[0]) < int(span[1])
    ):
        return False
    return bool(
        _NEGATION_SCOPE_REJECTION_PATTERN.search(
            str(terminal_attempt.reason_code or "")
        )
    )


def parse_source_frame_schema_bindings(
    raw: Any,
    *,
    inventory: Mapping[str, Any],
    logical_schema: Mapping[str, Any],
    allow_partial: bool = False,
    include_review_only: bool = True,
) -> dict[str, Any]:
    """Validate source-to-Schema bindings without lexical heuristics.

    Source binding is a recall-oriented auxiliary channel.  A local model can
    occasionally return a useful binding together with one stale frame or
    argument index.  In strict mode that remains a protocol failure.  The
    runtime may opt into partial mode, where each item is independently
    validated and only invalid items are discarded.  The independent relation
    verifier still gates every projected hypothesis before publication.
    """

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "source Schema binding returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "source Schema binding response must be an object"
        )
    coverage_complete = raw.get("coverage_complete") is True
    if not coverage_complete and not allow_partial:
        raise RelationClassificationError(
            "source Schema binding coverage is incomplete"
        )
    raw_bindings = raw.get("bindings")
    if not isinstance(raw_bindings, list):
        raise RelationClassificationError(
            "source Schema binding requires a bindings array"
        )
    schema = normalize_predicate_schema(logical_schema)
    frames = list(inventory.get("frames") or ())
    strict_structure_keys = {
        (
            int(item["frame_index"]),
            str(item["predicate"]),
            int(item["subject_index"]),
            int(item["object_index"]),
        )
        for item in deterministic_source_structure_schema_bindings(
            inventory,
            logical_schema=logical_schema,
            include_review_only=include_review_only,
        ).get("bindings") or ()
    }
    # Keep the parser's external result in immutable inventory coordinates so
    # projection and provenance continue to point at the original frame. The
    # model-facing binding index is the compact binary-frame coordinate from
    # source_frame_schema_binding_prompt().
    eligible_frame_indexes = [
        index for index, frame in enumerate(frames)
        if _source_frame_model_binding_eligible(frame)
    ]
    bindings: list[dict[str, Any]] = []
    dropped_bindings: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int, int]] = set()
    for item_index, item in enumerate(raw_bindings):
        error_message = ""
        if not isinstance(item, Mapping):
            error_message = "source Schema binding item must be an object"
            frame_index = subject_index = object_index = None
            predicate = ""
        else:
            frame_index = item.get("frame_index")
            subject_index = item.get("subject_index")
            object_index = item.get("object_index")
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (frame_index, subject_index, object_index)
            ):
                error_message = "source Schema binding indexes must be integers"
            predicate = str(item.get("predicate") or "")
        if not error_message and not 0 <= frame_index < len(eligible_frame_indexes):
            error_message = "source Schema binding selected unknown frame"
        source_frame_index = (
            eligible_frame_indexes[frame_index]
            if not error_message else None
        )
        frame = (
            frames[source_frame_index]
            if source_frame_index is not None else None
        )
        arguments = (
            list(frame.get("arguments") or [])
            if isinstance(frame, Mapping) else []
        )
        if not error_message and (
            not 0 <= subject_index < len(arguments)
            or not 0 <= object_index < len(arguments)
        ):
            error_message = "source Schema binding selected unknown argument"
        if not error_message and subject_index == object_index:
            error_message = "source Schema binding endpoints must differ"
        if not error_message and predicate not in schema:
            error_message = "source Schema binding predicate is not in Schema"
        if (
            not error_message
            and schema[predicate].publish_mode == PUBLISH_REVIEW_ONLY
            and not include_review_only
        ):
            error_message = "source Schema binding selected disabled review candidate"
        if not error_message and str(frame.get("assertion_status") or "").upper() != "ASSERTED":
            error_message = "source Schema binding selected non-asserted frame"
        if (
            not error_message
            and str(frame.get("deterministic_source_structure") or "")
            and (
                source_frame_index, predicate, subject_index, object_index,
            ) not in strict_structure_keys
        ):
            error_message = (
                "source Schema binding violates deterministic structure contract"
            )
        if (
            not error_message
            and not _predicate_allows_source_role_pair(
                schema[predicate], frame, subject_index, object_index,
            )
        ):
            error_message = (
                "source Schema binding violates semantic source-role contract"
            )
        if (
            not error_message
            and not _predicate_allows_source_polarity(
                schema[predicate], frame,
            )
        ):
            error_message = (
                "source Schema binding violates source-negation policy"
            )
        if error_message:
            if not allow_partial:
                raise RelationClassificationError(error_message)
            dropped_bindings.append({
                "item_index": item_index,
                "reason": error_message,
            })
            continue
        key = (source_frame_index, predicate, subject_index, object_index)
        if key in seen:
            error_message = "source Schema binding contains duplicate binding"
            if not allow_partial:
                raise RelationClassificationError(error_message)
            dropped_bindings.append({
                "item_index": item_index,
                "reason": error_message,
            })
            continue
        seen.add(key)
        bindings.append({
            "frame_index": source_frame_index,
            "predicate": predicate,
            "subject_index": subject_index,
            "object_index": object_index,
        })
    return {
        "coverage_complete": coverage_complete and not dropped_bindings,
        "bindings": bindings,
        "dropped_bindings": dropped_bindings,
        "parser_version": GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
    }


def _source_endpoint_surface_compatible(
    occurrence_surface: str,
    argument_surface: str,
) -> bool:
    """Match a frozen mention to a source argument without semantic fuzzying.

    Exact source text is preferred. The bounded fallback accepts a complete
    word contained in a longer source argument (needed when a relation takes
    a coordinated/event-valued argument) and a conservative final-token
    singular/plural variation. It never uses aliases, embeddings, stemming or
    a domain vocabulary.
    """

    occurrence = _normalized_atomic_surface(occurrence_surface)
    argument = _normalized_atomic_surface(argument_surface)
    if not occurrence or not argument:
        return False
    if occurrence == argument:
        return True
    if len(occurrence) >= 3 and re.search(
        rf"(?<!\w){re.escape(occurrence)}(?!\w)", argument,
    ):
        return True

    def tokens(value: str) -> list[str]:
        return [item for item in re.findall(r"[\w]+", value) if item]

    def singularize(value: str) -> str:
        if value.endswith("ies") and len(value) > 3:
            return value[:-3] + "y"
        if value.endswith(("ches", "shes", "xes", "zes")) and len(value) > 3:
            return value[:-2]
        if value.endswith("s") and not value.endswith("ss") and len(value) > 2:
            return value[:-1]
        return value

    occurrence_tokens = tokens(occurrence)
    argument_tokens = tokens(argument)
    if occurrence_tokens and argument_tokens:
        occurrence_tokens = [singularize(item) for item in occurrence_tokens]
        argument_tokens = [singularize(item) for item in argument_tokens]
        # Articles are not part of an entity surface and are the only prefix
        # ignored here; all remaining source words must match contiguously.
        while argument_tokens and argument_tokens[0] in {"a", "an", "the"}:
            argument_tokens.pop(0)
        for start in range(
            0, max(0, len(argument_tokens) - len(occurrence_tokens) + 1),
        ):
            if argument_tokens[start:start + len(occurrence_tokens)] == occurrence_tokens:
                return True
    return False


def source_frame_projection_hypotheses(
    inventory: Mapping[str, Any],
    bindings: Mapping[str, Any],
    *,
    occurrences: Sequence[Mapping[str, Any]],
    opportunities_by_pair: Mapping[str, PairOpportunity],
    logical_schema: Mapping[str, Any],
    classifier_model_revision: str,
) -> tuple[RelationHypothesis, ...]:
    """Project source frames onto frozen entity occurrences.

    The projection is intentionally exact: a mention must equal the complete
    source argument or its source-copied entity head.  No alias, fuzzy match,
    stemming, type guess, or domain vocabulary can create an endpoint here.
    A projected candidate is still only a hypothesis and is sent through the
    same deterministic validation and independent ordered verifier as every
    other source.
    """

    schema = normalize_predicate_schema(logical_schema)
    by_surface: dict[str, list[Mapping[str, Any]]] = {}
    by_occurrence_key: dict[str, Mapping[str, Any]] = {}
    for occurrence in occurrences:
        key = str(occurrence.get("occurrence_key") or "")
        surface = str(occurrence.get("surface") or "")
        if not key or not surface:
            continue
        by_occurrence_key[key] = occurrence
        by_surface.setdefault(_normalized_atomic_surface(surface), []).append(occurrence)

    frames = list(inventory.get("frames") or ())
    projected: list[RelationHypothesis] = []
    seen: set[tuple[str, str, str, str, int, int, int]] = set()
    for binding in bindings.get("bindings") or ():
        if not isinstance(binding, Mapping):
            continue
        try:
            frame_index = int(binding["frame_index"])
            subject_index = int(binding["subject_index"])
            object_index = int(binding["object_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= frame_index < len(frames):
            continue
        frame = frames[frame_index]
        if not isinstance(frame, Mapping):
            continue
        if str(frame.get("assertion_status") or "").upper() != "ASSERTED":
            continue
        predicate = str(binding.get("predicate") or "")
        spec = schema.get(predicate)
        arguments = list(frame.get("arguments") or [])
        if (
            spec is None
            or not _displayable_open_relation_expression(
                frame.get("relation")
            )
            or not 0 <= subject_index < len(arguments)
            or not 0 <= object_index < len(arguments)
            or subject_index == object_index
        ):
            continue
        if not _predicate_allows_source_role_pair(
            spec, frame, subject_index, object_index,
        ):
            continue
        if not _predicate_allows_source_polarity(spec, frame):
            continue
        kinds = list(frame.get("argument_kinds") or [])
        heads = list(frame.get("entity_heads") or [])
        roles = [str(value) for value in (
            frame.get("parser_inferred_source_roles")
            or frame.get("roles") or ()
        )]
        source_syntax_origin = str(
            frame.get("parser_inferred_source_syntax")
            or frame.get("deterministic_source_syntax")
            or ""
        )
        # A possessive noun phrase fixes only possessor/possessed topology. It
        # does not by itself choose ownership, part-whole, authorship or
        # affiliation. Precise publication therefore requires an explicit
        # machine-readable Schema role contract. The grounded open carrier is
        # still retained as an auditable association when no such contract is
        # available.
        if (
            "possessive_reference_construction"
            in source_syntax_origin.split("+")
            and not spec.open_relation_policy.get("enabled")
            and not spec.source_role_pairs
        ):
            continue
        if not (len(arguments) == len(kinds) == len(heads)):
            continue
        explicit_occurrence_keys = list(
            frame.get("argument_occurrence_keys") or []
        )
        resolved_reference_audit: list[dict[str, Any]] = []
        for reference in frame.get("resolved_reference_endpoints") or ():
            if not isinstance(reference, Mapping):
                continue
            proxy_key = str(reference.get("occurrence_key") or "")
            antecedent_key = str(
                reference.get("antecedent_occurrence_key") or ""
            )
            proxy = by_occurrence_key.get(proxy_key) or {}
            antecedent = by_occurrence_key.get(antecedent_key)
            if not proxy_key or antecedent is None:
                continue
            resolved_reference_audit.append({
                "frame_role": str(reference.get("role") or ""),
                "proxy_occurrence_key": proxy_key,
                "reference_text": str(
                    reference.get("reference_text")
                    or proxy.get("surface") or ""
                ),
                "reference_span": list(
                    reference.get("reference_span") or [
                        int(proxy.get("mention_start") or 0),
                        int(proxy.get("mention_end") or 0),
                    ]
                ),
                "reference_role": str(
                    proxy.get("reference_role") or ""
                ),
                "antecedent_occurrence_key": antecedent_key,
                "antecedent_surface": str(
                    antecedent.get("surface") or ""
                ),
                "antecedent_span": [
                    int(antecedent.get("mention_start") or 0),
                    int(antecedent.get("mention_end") or 0),
                ],
                "antecedent_canonical_id": str(
                    antecedent.get("canonical_id") or ""
                ),
                "resolution_source": str(
                    proxy.get("reference_resolution_source") or ""
                ),
            })

        def matches(index: int) -> list[Mapping[str, Any]]:
            if (
                len(explicit_occurrence_keys) == len(arguments)
                and 0 <= index < len(explicit_occurrence_keys)
            ):
                explicit = by_occurrence_key.get(
                    str(explicit_occurrence_keys[index])
                )
                # An explicit immutable occurrence ID is authoritative. Never
                # broaden it to another same-surface occurrence.
                return [explicit] if explicit is not None else []
            values = {
                _normalized_atomic_surface(arguments[index]),
                _normalized_atomic_surface(heads[index]),
            }
            values.discard("")
            result: list[Mapping[str, Any]] = []
            for value in values:
                result.extend(by_surface.get(value, ()))
            if not result:
                # Keep the exact fast path above, then allow a frozen mention
                # to occupy an explicit source argument without requiring the
                # inventory model to split every coordinated list perfectly.
                for occurrence in occurrences:
                    surface = str(occurrence.get("surface") or "")
                    if any(
                        _source_endpoint_surface_compatible(surface, candidate)
                        for candidate in (arguments[index], heads[index])
                        if str(candidate or "").strip()
                    ):
                        result.append(occurrence)
            unique: dict[str, Mapping[str, Any]] = {
                str(item.get("occurrence_key")): item for item in result
            }
            return list(unique.values())

        subjects = matches(subject_index)
        objects = matches(object_index)
        for subject in subjects:
            for object_ in objects:
                subject_key = str(subject.get("occurrence_key") or "")
                object_key = str(object_.get("occurrence_key") or "")
                if not subject_key or not object_key or subject_key == object_key:
                    continue
                pair_key = frozenset((subject_key, object_key))
                opportunity = next(
                    (
                        value for value in opportunities_by_pair.values()
                        if frozenset((
                            str(value.left_occurrence_key),
                            str(value.right_occurrence_key),
                        )) == pair_key
                    ),
                    None,
                )
                if opportunity is None:
                    continue
                allowed = {
                    str(item.get("name") or "")
                    for item in opportunity.allowed_predicate_metadata
                }
                if predicate not in allowed:
                    continue
                dedup_key = (
                    str(opportunity.pair_identity), subject_key, predicate,
                    object_key, frame_index, subject_index, object_index,
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                raw_evidence_span = list(frame.get("evidence_span") or [])
                if (
                    len(raw_evidence_span) == 2
                    and all(
                        isinstance(value, int)
                        for value in raw_evidence_span
                    )
                    and raw_evidence_span[0] < raw_evidence_span[1]
                ):
                    spans = (EvidenceSpan(
                        start=int(raw_evidence_span[0]),
                        end=int(raw_evidence_span[1]),
                    ),)
                else:
                    spans = (EvidenceSpan(
                        start=int(opportunity.context_span[0]),
                        end=int(opportunity.context_span[1]),
                    ),)
                candidate_source = (
                    "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION"
                    if explicit_occurrence_keys
                    else "ATOMIC_FRAME_SCHEMA_PROJECTION"
                )
                evidence_start = spans[0].start
                evidence_end = spans[0].end
                endpoint_canonicals = {
                    str(subject.get("canonical_id") or ""),
                    str(object_.get("canonical_id") or ""),
                }
                source_frame_other_participants: list[str] = []
                for occurrence in occurrences:
                    occurrence_key = str(
                        occurrence.get("occurrence_key") or ""
                    )
                    occurrence_canonical = str(
                        occurrence.get("canonical_id") or ""
                    )
                    occurrence_surface = str(
                        occurrence.get("surface") or ""
                    ).strip()
                    try:
                        occurrence_start = int(
                            occurrence.get("mention_start")
                        )
                        occurrence_end = int(
                            occurrence.get("mention_end")
                        )
                    except (TypeError, ValueError):
                        continue
                    if (
                        occurrence_key in {subject_key, object_key}
                        or (
                            occurrence_canonical
                            and occurrence_canonical in endpoint_canonicals
                        )
                        or not occurrence_surface
                        or not (
                            evidence_start <= occurrence_start
                            < occurrence_end <= evidence_end
                        )
                        or occurrence_surface
                        in source_frame_other_participants
                    ):
                        continue
                    source_frame_other_participants.append(
                        occurrence_surface
                    )
                    if len(source_frame_other_participants) >= 12:
                        break
                hypothesis_id = "relhyp_" + _digest((
                    opportunity.pair_identity, subject_key, predicate, object_key,
                    [[span.start, span.end] for span in spans],
                    candidate_source, frame_index,
                    subject_index, object_index,
                ))
                projected.append(RelationHypothesis(
                    hypothesis_id=hypothesis_id,
                    pair_identity=opportunity.pair_identity,
                    subject_occurrence_key=subject_key,
                    predicate=predicate,
                    object_occurrence_key=object_key,
                    evidence_spans=spans,
                    classifier_confidence=0.60,
                    classifier_policy_version=(
                        GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION
                    ),
                    classifier_model_revision=classifier_model_revision,
                    classifier_prompt_version=(
                        GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION
                    ),
                    classifier_schema_version=(
                        GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION
                    ),
                    classifier_parser_version=(
                        GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION
                    ),
                    predicate_metadata={
                        **asdict(spec),
                        "candidate_source": candidate_source,
                        "source_frame_index": frame_index,
                        "source_frame_relation": str(frame.get("relation") or ""),
                        "source_relation_grounding": str(
                            frame.get("source_relation_grounding") or ""
                        ),
                        "source_relation_original_expression": str(
                            frame.get(
                                "source_relation_original_expression"
                            ) or ""
                        ),
                        "source_relation_repair": str(
                            frame.get("source_relation_repair") or ""
                        ),
                        "source_frame_assertion_status": str(
                            frame.get("assertion_status") or ""
                        ).upper(),
                        "source_frame_explicit_negation": bool(
                            _source_relation_has_explicit_negation(
                                frame.get("relation")
                            )
                        ),
                        "source_frame_evidence_span": list(
                            frame.get("evidence_span") or []
                        ),
                        "source_frame_evidence_text": str(
                            frame.get("evidence_text") or ""
                        ),
                        "source_structure_kind": str(
                            frame.get("deterministic_source_structure") or ""
                        ),
                        "source_structure_contract": (
                            dict(frame["deterministic_structure_contract"])
                            if isinstance(
                                frame.get("deterministic_structure_contract"),
                                Mapping,
                            ) else {}
                        ),
                        "source_parallel_open_carrier_policy": str(
                            frame.get("parallel_open_carrier_policy") or ""
                        ),
                        "source_syntax_origin": str(
                            source_syntax_origin
                        ),
                        "source_pair_same_atomic_scope": bool(
                            opportunity.provenance.get("same_atomic_scope")
                            is True
                        ),
                        "source_frame_atomic_scope_override": bool(
                            opportunity.provenance.get(
                                "source_frame_atomic_scope_override"
                            )
                        ),
                        "source_structure_role_label": str(
                            frame.get("structural_role_label") or ""
                        ),
                        "source_subject_argument_index": subject_index,
                        "source_object_argument_index": object_index,
                        "source_subject_surface": str(
                            subject.get("surface") or ""
                        ),
                        "source_object_surface": str(
                            object_.get("surface") or ""
                        ),
                        "source_subject_canonical_id": str(
                            subject.get("canonical_id") or ""
                        ),
                        "source_object_canonical_id": str(
                            object_.get("canonical_id") or ""
                        ),
                        "source_frame_other_participants": (
                            source_frame_other_participants
                        ),
                        "source_binding_source": str(
                            binding.get("binding_source") or "MODEL_SOURCE_BINDING"
                        ),
                        "source_binding_shortlist_score": (
                            float(binding.get("shortlist_score"))
                            if isinstance(
                                binding.get("shortlist_score"), (int, float),
                            )
                            and not isinstance(
                                binding.get("shortlist_score"), bool,
                            )
                            else None
                        ),
                        "source_frame_roles": roles,
                        "source_resolved_reference_endpoints": (
                            resolved_reference_audit
                        ),
                    },
                    status=candidate_source,
                ))
    return tuple(projected)


def conflicting_source_hypothesis_groups(
    hypotheses: Sequence[RelationHypothesis],
    *,
    logical_schema: Mapping[str, Any],
) -> tuple[tuple[RelationHypothesis, ...], ...]:
    """Find Schema-declared mutually exclusive bindings of one source frame.

    Relation extraction is multi-label by default.  Candidates are therefore
    contrastively adjudicated only when the versioned logical Schema declares
    them in ``conflicts_with``.  Runtime code never infers exclusivity from a
    predicate name or from lexical similarity.
    """

    schema = normalize_predicate_schema(logical_schema)
    buckets: dict[
        tuple[str, str, str, int], dict[str, RelationHypothesis]
    ] = {}
    for hypothesis in hypotheses:
        metadata = hypothesis.predicate_metadata or {}
        frame_index = metadata.get("source_frame_index")
        if (
            hypothesis.predicate not in schema
            or not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or frame_index < 0
        ):
            continue
        key = (
            str(hypothesis.pair_identity),
            str(hypothesis.subject_occurrence_key),
            str(hypothesis.object_occurrence_key),
            int(frame_index),
        )
        buckets.setdefault(key, {}).setdefault(
            hypothesis.predicate, hypothesis,
        )

    groups: list[tuple[RelationHypothesis, ...]] = []
    for candidates in buckets.values():
        names = set(candidates)
        adjacency = {
            name: {
                other for other in names - {name}
                if (
                    other in schema[name].conflicts_with
                    or name in schema[other].conflicts_with
                )
            }
            for name in names
        }
        unseen = {name for name, peers in adjacency.items() if peers}
        while unseen:
            seed = min(unseen)
            component: set[str] = set()
            pending = [seed]
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.extend(sorted(adjacency[current] - component))
            unseen -= component
            if len(component) > 1:
                groups.append(tuple(
                    candidates[name] for name in sorted(component)
                ))
    return tuple(sorted(
        groups,
        key=lambda group: (
            str(group[0].pair_identity),
            str(group[0].subject_occurrence_key),
            str(group[0].object_occurrence_key),
            int((group[0].predicate_metadata or {})["source_frame_index"]),
            tuple(item.predicate for item in group),
        ),
    ))


def source_predicate_conflict_prompt(
    hypotheses: Sequence[RelationHypothesis],
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> str:
    """Build one bounded contrastive rerank request for conflicting predicates."""

    group = tuple(hypotheses)
    if len(group) < 2:
        raise ValueError("predicate conflict adjudication requires alternatives")
    schema = normalize_predicate_schema(logical_schema)
    first = group[0]
    identity = (
        first.pair_identity,
        first.subject_occurrence_key,
        first.object_occurrence_key,
        (first.predicate_metadata or {}).get("source_frame_index"),
    )
    if any((
        item.pair_identity,
        item.subject_occurrence_key,
        item.object_occurrence_key,
        (item.predicate_metadata or {}).get("source_frame_index"),
    ) != identity for item in group):
        raise ValueError("predicate conflict alternatives must share one source frame")
    names = [item.predicate for item in group]
    if len(set(names)) != len(names) or any(name not in schema for name in names):
        raise ValueError("predicate conflict alternatives must be unique Schema predicates")
    for name in names:
        peers = set(schema[name].conflicts_with)
        if not any(
            other in peers or name in schema[other].conflicts_with
            for other in names if other != name
        ):
            raise ValueError("predicate conflict alternatives lack a Schema conflict")
    context_window = _pair_model_context(opportunity, content=content)
    subject = _occurrence_details(
        opportunity, first.subject_occurrence_key, content,
    )
    object_ = _occurrence_details(
        opportunity, first.object_occurrence_key, content,
    )
    metadata = first.predicate_metadata or {}
    return json.dumps({
        "instruction": (
            "These logical predicates are explicitly mutually exclusive for this "
            "source-frame binding. Contrast every complete semantic definition, "
            "direction, endpoint role and example against the same frozen source. "
            "Return the one relation directly entailed for the exact ordered subject "
            "and object, or an empty relations array when none is entailed. Do not "
            "select by lexical overlap, broad topical association, endpoint type, or "
            "a relation to an omitted third argument. Never return more than one."
        ),
        "contract_version": GROUNDED_RELATION_CONFLICT_PROMPT_VERSION,
        "source_context": {
            "start": opportunity.context_span[0],
            "end": opportunity.context_span[1],
            "text": context_window["text"],
            "visible_segments": context_window["visible_segments"],
            "truncated": bool(context_window["truncated"]),
        },
        "subject": subject,
        "object": object_,
        "source_frame": {
            "relation": str(metadata.get("source_frame_relation") or ""),
            "roles": list(metadata.get("source_frame_roles") or []),
            "syntax_origin": str(metadata.get("source_syntax_origin") or ""),
        },
        "candidates": [{
            "relation": name,
            "semantic_definition": schema[name].semantic_definition,
            "direction_description": schema[name].direction_description,
            "subject_role_description": schema[name].subject_role_description,
            "object_role_description": schema[name].object_role_description,
            "positive_examples": list(schema[name].positive_examples),
            "negative_examples": list(schema[name].negative_examples),
        } for name in sorted(names)],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def source_predicate_conflict_response_schema(
    hypotheses: Sequence[RelationHypothesis],
    opportunity: PairOpportunity,
    *,
    content: str,
) -> dict[str, Any]:
    group = tuple(hypotheses)
    if not group:
        raise ValueError("predicate conflict response requires candidates")
    names = sorted({item.predicate for item in group})
    # Endpoints are immutable inputs to this adjudication and are never
    # selected by the model. Requiring them to be echoed only creates a second
    # copy channel (and local models commonly omit constant fields).
    _occurrence_details(
        opportunity, group[0].subject_occurrence_key, content,
    )
    _occurrence_details(
        opportunity, group[0].object_occurrence_key, content,
    )
    return {
        "$id": GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "relations": {
                "type": "array",
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "relation": {"type": "string", "enum": names},
                    },
                    "required": ["relation"],
                },
            },
        },
        "required": ["relations"],
    }


def parse_source_predicate_conflict(
    raw: Any,
    *,
    hypotheses: Sequence[RelationHypothesis],
    opportunity: PairOpportunity,
    content: str,
) -> str:
    """Return the unique winning predicate, or ``""`` for a valid none result."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "predicate conflict adjudication returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "predicate conflict adjudication response must be an object"
        )
    relations = raw.get("relations")
    if not isinstance(relations, list) or len(relations) > 1:
        raise RelationClassificationError(
            "predicate conflict adjudication requires at most one relation"
        )
    if not relations:
        return ""
    item = relations[0]
    if not isinstance(item, Mapping):
        raise RelationClassificationError(
            "predicate conflict adjudication relation must be an object"
        )
    group = tuple(hypotheses)
    allowed = {candidate.predicate for candidate in group}
    relation = str(item.get("relation") or "")
    if relation not in allowed:
        raise RelationClassificationError(
            "predicate conflict adjudication selected an unknown relation"
        )
    # Resolve both immutable endpoint records here so stale/mismatched groups
    # still fail before a model-selected predicate can be returned.
    _occurrence_details(
        opportunity, group[0].subject_occurrence_key, content,
    )
    _occurrence_details(
        opportunity, group[0].object_occurrence_key, content,
    )
    return relation


def source_predicate_conflict_terminal_attempt(
    hypothesis: RelationHypothesis,
    *,
    verdict: str,
    reason_code: str,
    verifier_model_revision: str,
    request_payload: Mapping[str, Any] | None = None,
    raw_response_text: str = "",
    parsed_response: Mapping[str, Any] | None = None,
    token_usage: Mapping[str, Any] | None = None,
    error_message: str = "",
) -> VerificationAttempt:
    """Create an audited non-publication result for a conflict alternative."""

    if verdict not in {REJECTED, UNCERTAIN, RETRYABLE_FAILED}:
        raise ValueError("source predicate conflict attempt must not publish")
    return VerificationAttempt(
        attempt_id="relverify_" + _digest((
            hypothesis.hypothesis_id,
            1,
            verifier_model_revision,
            GROUNDED_RELATION_CONFLICT_PROMPT_VERSION,
        )),
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=1,
        verdict=verdict,
        reason_code=str(reason_code),
        supported_evidence_spans=(),
        independent_call=True,
        verifier_prompt_version=GROUNDED_RELATION_CONFLICT_PROMPT_VERSION,
        verifier_schema_version=GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION,
        verifier_parser_version=GROUNDED_RELATION_CONFLICT_PARSER_VERSION,
        verifier_model_revision=str(verifier_model_revision),
        forward_entailment=(
            "NOT_ENTAILED" if verdict == REJECTED else "UNCERTAIN"
        ),
        reverse_entailment="UNCERTAIN",
        subject_role_supported=False,
        object_role_supported=False,
        normalized_subject_occurrence_key=(
            hypothesis.subject_occurrence_key
        ),
        normalized_object_occurrence_key=(
            hypothesis.object_occurrence_key
        ),
        error_message=str(error_message)[:2000],
        model_request_payload=dict(request_payload or {}),
        raw_response_text=str(raw_response_text),
        parsed_response=dict(parsed_response or {}),
        token_usage=dict(token_usage or {}),
    )


def source_projection_role_audit_terminal_attempt(
    hypothesis: RelationHypothesis,
    *,
    result: str,
    verifier_model_revision: str,
    request_payload: Mapping[str, Any] | None = None,
    raw_response_text: str = "",
    parsed_response: Mapping[str, Any] | None = None,
    token_usage: Mapping[str, Any] | None = None,
    error_message: str = "",
) -> VerificationAttempt:
    """Create a fail-closed terminal result for a risky source projection."""

    normalized = str(result or "").upper()
    if normalized not in {"CONFLICT", "UNCERTAIN", "UNAVAILABLE"}:
        raise ValueError("source role audit terminal result must not publish")
    verdict = REJECTED if normalized == "CONFLICT" else UNCERTAIN
    reason_code = {
        "CONFLICT": "SOURCE_PROJECTION_ROLE_CONFLICT",
        "UNCERTAIN": "SOURCE_PROJECTION_ROLE_UNCERTAIN",
        "UNAVAILABLE": "SOURCE_PROJECTION_ROLE_AUDIT_UNAVAILABLE",
    }[normalized]
    return VerificationAttempt(
        attempt_id="relverify_" + _digest((
            hypothesis.hypothesis_id,
            1,
            verifier_model_revision,
            GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION,
        )),
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=1,
        verdict=verdict,
        reason_code=reason_code,
        supported_evidence_spans=(),
        independent_call=True,
        verifier_prompt_version=(
            GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION
        ),
        verifier_schema_version=(
            GROUNDED_RELATION_SOURCE_ROLE_AUDIT_SCHEMA_VERSION
        ),
        verifier_parser_version=(
            GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PARSER_VERSION
        ),
        verifier_model_revision=str(verifier_model_revision),
        forward_entailment=(
            "NOT_ENTAILED" if verdict == REJECTED else "UNCERTAIN"
        ),
        reverse_entailment="UNCERTAIN",
        subject_role_supported=False,
        object_role_supported=False,
        normalized_subject_occurrence_key=(
            hypothesis.subject_occurrence_key
        ),
        normalized_object_occurrence_key=(
            hypothesis.object_occurrence_key
        ),
        error_message=str(error_message)[:2000],
        model_request_payload=dict(request_payload or {}),
        raw_response_text=str(raw_response_text),
        parsed_response=dict(parsed_response or {}),
        token_usage=dict(token_usage or {}),
    )


def source_projection_group_audit_terminal_attempt(
    hypothesis: RelationHypothesis,
    *,
    result: str,
    verifier_model_revision: str,
    request_payload: Mapping[str, Any] | None = None,
    raw_response_text: str = "",
    parsed_response: Mapping[str, Any] | None = None,
    token_usage: Mapping[str, Any] | None = None,
    error_message: str = "",
) -> VerificationAttempt:
    """Create a fail-closed terminal result from joint competition audit."""

    normalized = str(result or "").upper()
    if normalized not in {"CONFLICT", "UNCERTAIN", "UNAVAILABLE"}:
        raise ValueError("source group audit terminal result must not publish")
    verdict = REJECTED if normalized == "CONFLICT" else UNCERTAIN
    reason_code = {
        "CONFLICT": "SOURCE_PROJECTION_GROUP_CONFLICT",
        "UNCERTAIN": "SOURCE_PROJECTION_GROUP_UNCERTAIN",
        "UNAVAILABLE": "SOURCE_PROJECTION_GROUP_AUDIT_UNAVAILABLE",
    }[normalized]
    return VerificationAttempt(
        attempt_id="relverify_" + _digest((
            hypothesis.hypothesis_id,
            1,
            verifier_model_revision,
            GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION,
        )),
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=1,
        verdict=verdict,
        reason_code=reason_code,
        supported_evidence_spans=(),
        independent_call=True,
        verifier_prompt_version=(
            GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION
        ),
        verifier_schema_version=(
            GROUNDED_RELATION_SOURCE_GROUP_AUDIT_SCHEMA_VERSION
        ),
        verifier_parser_version=(
            GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PARSER_VERSION
        ),
        verifier_model_revision=str(verifier_model_revision),
        forward_entailment=(
            "NOT_ENTAILED" if verdict == REJECTED else "UNCERTAIN"
        ),
        reverse_entailment="UNCERTAIN",
        subject_role_supported=False,
        object_role_supported=False,
        normalized_subject_occurrence_key=(
            hypothesis.subject_occurrence_key
        ),
        normalized_object_occurrence_key=(
            hypothesis.object_occurrence_key
        ),
        error_message=str(error_message)[:2000],
        model_request_payload=dict(request_payload or {}),
        raw_response_text=str(raw_response_text),
        parsed_response=dict(parsed_response or {}),
        token_usage=dict(token_usage or {}),
    )


def source_projection_group_anchor_audit_terminal_attempt(
    hypothesis: RelationHypothesis,
    *,
    result: str,
    verifier_model_revision: str,
    request_payload: Mapping[str, Any] | None = None,
    raw_response_text: str = "",
    parsed_response: Mapping[str, Any] | None = None,
    token_usage: Mapping[str, Any] | None = None,
    error_message: str = "",
) -> VerificationAttempt:
    """Fail closed when a coordinated relation template does not align."""

    normalized = str(result or "").upper()
    if normalized not in {"CONFLICT", "UNCERTAIN", "UNAVAILABLE"}:
        raise ValueError("source group anchor audit result must not publish")
    verdict = REJECTED if normalized == "CONFLICT" else UNCERTAIN
    reason_code = {
        "CONFLICT": "SOURCE_PROJECTION_GROUP_ANCHOR_CONFLICT",
        "UNCERTAIN": "SOURCE_PROJECTION_GROUP_ANCHOR_UNCERTAIN",
        "UNAVAILABLE": "SOURCE_PROJECTION_GROUP_ANCHOR_AUDIT_UNAVAILABLE",
    }[normalized]
    return VerificationAttempt(
        attempt_id="relverify_" + _digest((
            hypothesis.hypothesis_id,
            1,
            verifier_model_revision,
            GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PROMPT_VERSION,
        )),
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=1,
        verdict=verdict,
        reason_code=reason_code,
        supported_evidence_spans=(),
        independent_call=True,
        verifier_prompt_version=(
            GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PROMPT_VERSION
        ),
        verifier_schema_version=(
            GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_SCHEMA_VERSION
        ),
        verifier_parser_version=(
            GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PARSER_VERSION
        ),
        verifier_model_revision=str(verifier_model_revision),
        forward_entailment=(
            "NOT_ENTAILED" if verdict == REJECTED else "UNCERTAIN"
        ),
        reverse_entailment="UNCERTAIN",
        subject_role_supported=False,
        object_role_supported=False,
        normalized_subject_occurrence_key=(
            hypothesis.subject_occurrence_key
        ),
        normalized_object_occurrence_key=(
            hypothesis.object_occurrence_key
        ),
        error_message=str(error_message)[:2000],
        model_request_payload=dict(request_payload or {}),
        raw_response_text=str(raw_response_text),
        parsed_response=dict(parsed_response or {}),
        token_usage=dict(token_usage or {}),
    )


def source_structure_binding_audit_terminal_attempt(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    result: str,
    verifier_model_revision: str,
    request_payload: Mapping[str, Any] | None = None,
    raw_response_text: str = "",
    parsed_response: Mapping[str, Any] | None = None,
    token_usage: Mapping[str, Any] | None = None,
    error_message: str = "",
) -> VerificationAttempt:
    """Convert one exact structural audit into the common terminal ledger."""

    # Re-run the complete deterministic authorization and source-span checks;
    # callers cannot turn an arbitrary adjacency into structural evidence by
    # invoking this constructor directly.
    _source_structure_binding_audit_payload(
        hypothesis,
        opportunity,
        content=content,
        logical_schema=logical_schema,
    )
    normalized = str(result or "").strip().upper()
    if normalized not in {
        "ALIGNED", "CONFLICT", "UNCERTAIN", "UNAVAILABLE",
    }:
        raise ValueError("source structure audit result is invalid")
    verdict = (
        SUPPORTED if normalized == "ALIGNED"
        else REJECTED if normalized == "CONFLICT"
        else UNCERTAIN
    )
    spec = normalize_predicate_schema(logical_schema)[hypothesis.predicate]
    spans = tuple(hypothesis.evidence_spans) if verdict == SUPPORTED else ()
    if verdict == SUPPORTED:
        parsed_spans = _parse_spans(
            [asdict(value) for value in spans],
            opportunity=opportunity,
            content=content,
        )
        if not (
            parsed_spans
            and _spans_cover(parsed_spans, opportunity.left_span)
            and _spans_cover(parsed_spans, opportunity.right_span)
        ):
            raise RelationVerificationRejected(
                "SUPPORTED_EVIDENCE_MISSING_ENDPOINT",
                "source structure evidence does not cover exact endpoints",
            )
        spans = parsed_spans
    reason_code = {
        "ALIGNED": "SUPPORTED_SOURCE_STRUCTURE_BINDING",
        "CONFLICT": "SOURCE_STRUCTURE_BINDING_CONFLICT",
        "UNCERTAIN": "SOURCE_STRUCTURE_BINDING_UNCERTAIN",
        "UNAVAILABLE": "SOURCE_STRUCTURE_AUDIT_UNAVAILABLE",
    }[normalized]
    return VerificationAttempt(
        attempt_id="relverify_" + _digest((
            hypothesis.hypothesis_id,
            1,
            verifier_model_revision,
            GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PROMPT_VERSION,
        )),
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=1,
        verdict=verdict,
        reason_code=reason_code,
        supported_evidence_spans=spans,
        independent_call=True,
        verifier_prompt_version=(
            GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PROMPT_VERSION
        ),
        verifier_schema_version=(
            GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_SCHEMA_VERSION
        ),
        verifier_parser_version=(
            GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PARSER_VERSION
        ),
        verifier_model_revision=str(verifier_model_revision),
        forward_entailment=(
            "ENTAILED" if verdict == SUPPORTED else
            "NOT_ENTAILED" if verdict == REJECTED else "UNCERTAIN"
        ),
        reverse_entailment=(
            "ENTAILED" if verdict == SUPPORTED and spec.symmetric else
            "NOT_ENTAILED" if verdict in {SUPPORTED, REJECTED}
            else "UNCERTAIN"
        ),
        subject_role_supported=(verdict == SUPPORTED),
        object_role_supported=(verdict == SUPPORTED),
        normalized_subject_occurrence_key=(
            hypothesis.subject_occurrence_key
        ),
        normalized_object_occurrence_key=(
            hypothesis.object_occurrence_key
        ),
        error_message=str(error_message)[:2000],
        model_request_payload=dict(request_payload or {}),
        raw_response_text=str(raw_response_text),
        parsed_response=dict(parsed_response or {}),
        token_usage=dict(token_usage or {}),
    )


def parse_atomic_frame_schema_alignment(
    raw: Any,
    *,
    inventory: Mapping[str, Any],
    subject_surface: str,
    object_surface: str,
    symmetric: bool = False,
) -> str:
    """Match endpoints against candidate-blind, Schema-valid role bindings."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "frame schema alignment returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "frame schema alignment response must be an object"
        )

    if raw.get("coverage_complete") is not True:
        raise RelationClassificationError(
            "frame schema alignment coverage is incomplete"
        )
    raw_alignments = raw.get("alignments")
    if not isinstance(raw_alignments, list):
        raise RelationClassificationError(
            "frame schema alignment requires an alignments array"
        )
    frames = list(inventory.get("frames") or [])
    bindings: set[tuple[int, int, int]] = set()
    aligned = False
    reversed_ = False
    for item in raw_alignments:
        if not isinstance(item, Mapping):
            raise RelationClassificationError(
                "frame schema alignment item must be an object"
            )
        values = (
            item.get("frame_index"),
            item.get("subject_index"),
            item.get("object_index"),
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in values
        ):
            raise RelationClassificationError(
                "frame schema alignment indexes must be integers"
            )
        frame_index, subject_index, object_index = values
        if not 0 <= frame_index < len(frames):
            raise RelationClassificationError(
                "frame schema alignment selected an unknown frame"
            )
        frame = frames[frame_index]
        arguments = list(frame.get("arguments") or [])
        if (
            subject_index == object_index
            or not 0 <= subject_index < len(arguments)
            or not 0 <= object_index < len(arguments)
        ):
            raise RelationClassificationError(
                "frame schema alignment selected invalid role indexes"
            )
        binding = (frame_index, subject_index, object_index)
        if binding in bindings:
            raise RelationClassificationError(
                "frame schema alignment contains a duplicate binding"
            )
        bindings.add(binding)
        aligned = aligned or (
            _atomic_frame_argument_matches(
                frame, subject_index, subject_surface,
            )
            and _atomic_frame_argument_matches(
                frame, object_index, object_surface,
            )
        )
        reversed_ = reversed_ or (
            _atomic_frame_argument_matches(
                frame, subject_index, object_surface,
            )
            and _atomic_frame_argument_matches(
                frame, object_index, subject_surface,
            )
        )
    if aligned and (symmetric or not reversed_):
        return "ALIGNED"
    if reversed_ and (symmetric or not aligned):
        return "REVERSED"
    return "NOT_ALIGNED"


def verifier_role_alignment_prompt(
    hypothesis: RelationHypothesis,
    *,
    logical_schema: Mapping[str, Any],
    blind_role_audit: Mapping[str, Any],
    opportunity: PairOpportunity | None = None,
    content: str = "",
) -> str:
    """Compare both endpoint orientations to versioned Schema roles.

    The predicate identifier is deliberately omitted.  Behavior is driven by
    role metadata, so renaming the predicate cannot change this gate.
    """

    spec = normalize_predicate_schema(logical_schema).get(hypothesis.predicate)
    if spec is None:
        raise ValueError("role alignment requires current predicate metadata")
    payload: dict[str, Any] = {
            "instruction": (
                "Read the supplied frozen source_context and exact endpoint spans "
                "The source_context may be a bounded endpoint-centered display; omitted "
                "ranges are unavailable, and all offsets remain absolute source offsets. "
                "independently. The predicate-blind role audit is source-evidence from a "
                "separate pass, not a conclusion; check it against the source and use it "
                "as the explicit contextual-role claim for each exact endpoint. "
                "Derive each endpoint's contextual role from the source before comparing "
                "it with each required role one component at a time. Endpoint labels are "
                "neutral. A broad entity category is "
                "insufficient; compare the actual role each endpoint plays in the "
                "stated relation. Do not return an overall direction. The caller "
                "derives direction only after all four comparisons are present. MATCH "
                "means the contextual role satisfies that one required role, MISMATCH "
                "means it does not, and UNCERTAIN means the source is insufficient. "
                "A contextual role opposite to a required role must be MISMATCH, not "
                "approximately matched. Relation-level semantic similarity cannot "
                "override an endpoint-role mismatch. Mere "
                "participation, topical association, or interaction is insufficient. "
                "Return one flat JSON object with no nested answer object and no "
                "explanations. Its endpoint_a_vs_source, endpoint_b_vs_target, "
                "endpoint_b_vs_source, and endpoint_a_vs_target fields must each be "
                "exactly MATCH, MISMATCH, or UNCERTAIN. Also compare the "
                "predicate-blind source "
                "relation paraphrase with the required semantic definition in each "
                "direction. A relation with a different meaning must not match merely "
                "because its endpoint roles or topic are similar. Treat the supplied "
                "positive and negative examples as "
                "contrastive guidance, not as new source facts. Return "
                "forward_relation_vs_definition and reverse_relation_vs_definition "
                "using the same three values. Finally return semantic_contrast as "
                "DEFINITION when the source relation is an instance of the semantic "
                "definition, NEGATIVE_EXAMPLE when it fits a supplied negative "
                "example better, NEITHER when it is a different relation not covered "
                "by either set, or UNCERTAIN when the source is insufficient. Endpoint "
                "role similarity alone is not semantic equivalence. The definition "
                "matches only when Endpoint A and Endpoint B themselves fill its two "
                "argument roles. A relation to a third entity provided, owned, "
                "contained, modified, or qualified by an endpoint is not a relation "
                "to that endpoint."
            ),
            "contract_version": GROUNDED_RELATION_ROLE_ALIGNMENT_PROMPT_VERSION,
            "hypothesis_id": hypothesis.hypothesis_id,
            "required_roles": {
                "relation_source_role": spec.subject_role_description,
                "relation_target_role": spec.object_role_description,
            },
            "required_relation_semantics": {
                "semantic_definition": spec.semantic_definition,
                "direction_description": spec.direction_description,
                "positive_examples": list(spec.positive_examples),
                "negative_examples": list(spec.negative_examples),
            },
            "symmetric": bool(spec.symmetric),
            "orientation_contract": {
                "forward": "endpoint A -> relation source; endpoint B -> relation target",
                "reverse": "endpoint B -> relation source; endpoint A -> relation target",
            },
            "predicate_blind_role_audit": {
                "endpoint_a_contextual_role": str(
                    blind_role_audit.get("subject_role") or ""
                ),
                "endpoint_b_contextual_role": str(
                    blind_role_audit.get("object_role") or ""
                ),
                "expressed_relation": str(
                    blind_role_audit.get("expressed_relation") or ""
                ),
                "endpoint_a_complete_argument": blind_role_audit.get(
                    "endpoint_a_complete_argument"
                ),
                "endpoint_b_complete_argument": blind_role_audit.get(
                    "endpoint_b_complete_argument"
                ),
            },
        }
    if opportunity is not None:
        context_window = _pair_model_context(
            opportunity, content=content,
        )
        payload["source_context"] = {
            "start": opportunity.context_span[0],
            "end": opportunity.context_span[1],
            "text": context_window["text"],
            "visible_segments": context_window["visible_segments"],
            "truncated": bool(context_window["truncated"]),
        }
        payload["endpoint_a"] = _occurrence_details(
            opportunity, hypothesis.subject_occurrence_key, content,
        )
        payload["endpoint_b"] = _occurrence_details(
            opportunity, hypothesis.object_occurrence_key, content,
        )
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def verifier_role_alignment_response_schema() -> dict[str, Any]:
    comparison = {
        "type": "string",
        "enum": ["MATCH", "MISMATCH", "UNCERTAIN"],
    }
    return {
        "$id": GROUNDED_RELATION_ROLE_ALIGNMENT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "endpoint_a_vs_source": comparison,
            "endpoint_b_vs_target": comparison,
            "endpoint_b_vs_source": comparison,
            "endpoint_a_vs_target": comparison,
            "forward_relation_vs_definition": comparison,
            "reverse_relation_vs_definition": comparison,
            "semantic_contrast": {
                "type": "string",
                "enum": [
                    "DEFINITION", "NEGATIVE_EXAMPLE", "NEITHER", "UNCERTAIN",
                ],
            },
        },
        "required": [
            "endpoint_a_vs_source", "endpoint_b_vs_target",
            "endpoint_b_vs_source", "endpoint_a_vs_target",
            "forward_relation_vs_definition", "reverse_relation_vs_definition",
            "semantic_contrast",
        ],
    }


def verifier_semantic_contrast_response_schema() -> dict[str, Any]:
    return {
        "$id": GROUNDED_RELATION_SEMANTIC_CONTRAST_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_contrast": {
                "type": "string",
                "enum": [
                    "DEFINITION", "NEGATIVE_EXAMPLE", "NEITHER", "UNCERTAIN",
                ],
            },
        },
        "required": ["semantic_contrast"],
    }


def verifier_semantic_contrast_prompt(alignment_prompt: str) -> str:
    """Reduce the role-alignment request to one predicate-blind meaning check."""

    try:
        payload = json.loads(alignment_prompt)
    except json.JSONDecodeError as exc:
        raise RelationClassificationError(
            "verifier semantic contrast requires a JSON alignment prompt"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RelationClassificationError(
            "verifier semantic contrast requires an object alignment prompt"
        )
    source_context = payload.get("source_context")
    endpoint_a = payload.get("endpoint_a")
    endpoint_b = payload.get("endpoint_b")
    source_relation = str(
        payload.get("source_relation_paraphrase") or ""
    ).strip()
    required_semantics = payload.get("required_relation_semantics")
    grounded_source = all(isinstance(value, Mapping) for value in (
        source_context, endpoint_a, endpoint_b,
    ))
    if (
        not isinstance(required_semantics, Mapping)
        or (not grounded_source and not source_relation)
    ):
        raise RelationClassificationError(
            "verifier semantic contrast requires source and Schema semantics"
        )
    return json.dumps(
        {
            "instruction": (
                "Read the frozen source text and exact endpoint spans independently; "
                "do not rely on a prior model paraphrase. Judge only whether the "
                "direct relation between Endpoint A and Endpoint B has the same "
                "meaning as the versioned Schema definition in A-to-B order. Return exactly "
                "one JSON field named semantic_contrast. Use DEFINITION when it is "
                "an instance of the definition, NEGATIVE_EXAMPLE when it matches a "
                "negative example better, NEITHER for a different relation, and "
                "UNCERTAIN when the source is insufficient. Do not infer equivalence "
                "from endpoint types, role similarity, or topical association. Use "
                "DEFINITION only when the two exact supplied endpoints themselves fill "
                "the definition's source and target roles. If one endpoint merely "
                "provides, owns, contains, modifies, or qualifies a third entity that "
                "is the actual argument, return NEITHER rather than substituting that "
                "endpoint for the third entity."
            ),
            "contract_version": GROUNDED_RELATION_SEMANTIC_CONTRAST_PROMPT_VERSION,
            **({
                "source_context": dict(source_context),
                "endpoint_a": dict(endpoint_a),
                "endpoint_b": dict(endpoint_b),
            } if grounded_source else {
                # Compatibility for frozen legacy audit fixtures. Production
                # v2 always supplies the source and exact endpoint spans.
                "source_relation_paraphrase": source_relation,
            }),
            "required_relation_semantics": {
                "semantic_definition": str(
                    required_semantics.get("semantic_definition") or ""
                ),
                "direction_description": str(
                    required_semantics.get("direction_description") or ""
                ),
                "positive_examples": list(
                    required_semantics.get("positive_examples") or []
                ),
                "negative_examples": list(
                    required_semantics.get("negative_examples") or []
                ),
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_verifier_semantic_contrast(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "verifier semantic contrast returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "verifier semantic contrast response must be an object"
        )
    value = str(
        raw.get("semantic_contrast") or raw.get("contrast_result") or ""
    ).strip().upper().replace("-", "_").replace(" ", "_")
    normalized = {
        "DEFINITION": "DEFINITION",
        "MATCH": "DEFINITION",
        "POSITIVE": "DEFINITION",
        "NEGATIVE": "NEGATIVE_EXAMPLE",
        "NEGATIVE_EXAMPLE": "NEGATIVE_EXAMPLE",
        "NEITHER": "NEITHER",
        "OTHER": "NEITHER",
        "UNCERTAIN": "UNCERTAIN",
        "UNKNOWN": "UNCERTAIN",
    }.get(value, "")
    if not normalized:
        raise RelationClassificationError(
            "verifier semantic contrast requires one terminal decision"
        )
    return normalized


def verifier_exact_argument_response_schema() -> dict[str, Any]:
    # Each direction is judged by a separate bounded NLI call. This avoids
    # enum-first bias and prevents one answer from conditioning the opposite
    # direction. A single conjunction is materially more reliable on local
    # constrained-output models than several booleans (which may omit false
    # fields). Exact occurrence existence and span integrity are deterministic
    # elsewhere; this answer cannot publish without the full verifier.
    return {
        "$id": GROUNDED_RELATION_EXACT_ARGUMENT_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "direction_supported": {"type": "boolean"},
        },
        "required": ["direction_supported"],
    }


def verifier_exact_argument_prompt(
    alignment_prompt: str,
    *,
    direction: str = "FORWARD",
) -> str:
    """Build one compact ordered NLI check over exact source arguments."""

    try:
        payload = json.loads(alignment_prompt)
    except json.JSONDecodeError as exc:
        raise RelationClassificationError(
            "verifier exact-argument check requires a JSON alignment prompt"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RelationClassificationError(
            "verifier exact-argument check requires an object alignment prompt"
        )
    context = payload.get("source_context")
    endpoint_a = payload.get("endpoint_a")
    endpoint_b = payload.get("endpoint_b")
    roles = payload.get("required_roles")
    semantics = payload.get("required_relation_semantics")
    role_audit = payload.get("predicate_blind_role_audit")
    if not isinstance(role_audit, Mapping):
        legacy_roles = payload.get("context_roles")
        if isinstance(legacy_roles, Mapping):
            role_audit = {
                "endpoint_a_contextual_role": legacy_roles.get(
                    "endpoint_a_role", ""
                ),
                "endpoint_b_contextual_role": legacy_roles.get(
                    "endpoint_b_role", ""
                ),
            }
    if not all(isinstance(value, Mapping) for value in (
        context, endpoint_a, endpoint_b, roles, semantics, role_audit,
    )):
        raise RelationClassificationError(
            "verifier exact-argument check requires source, endpoints, and roles"
        )
    source_text = str(context.get("text") or "")
    context_start = int(context.get("start") or 0)
    direction = str(direction).strip().upper()
    if direction not in {"FORWARD", "REVERSE"}:
        raise ValueError("exact-argument direction must be FORWARD or REVERSE")

    def endpoint_payload(endpoint: Mapping[str, Any]) -> dict[str, Any]:
        span = endpoint.get("span") or ()
        try:
            absolute_start, absolute_end = int(span[0]), int(span[1])
        except (IndexError, TypeError, ValueError):
            absolute_start, absolute_end = context_start, context_start
        local_start = max(0, absolute_start - context_start)
        local_end = max(local_start, absolute_end - context_start)
        return {
            "surface": str(endpoint.get("surface") or ""),
            "absolute_span": [absolute_start, absolute_end],
            "context_span": [local_start, local_end],
        }

    endpoint_a_payload = endpoint_payload(endpoint_a)
    endpoint_b_payload = endpoint_payload(endpoint_b)
    if direction == "FORWARD":
        claim_source = endpoint_a_payload
        claim_target = endpoint_b_payload
        source_contextual_role = str(
            role_audit.get("endpoint_a_contextual_role") or ""
        )
        target_contextual_role = str(
            role_audit.get("endpoint_b_contextual_role") or ""
        )
    else:
        claim_source = endpoint_b_payload
        claim_target = endpoint_a_payload
        source_contextual_role = str(
            role_audit.get("endpoint_b_contextual_role") or ""
        )
        target_contextual_role = str(
            role_audit.get("endpoint_a_contextual_role") or ""
        )

    uses_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", source_text))
    instruction = (
        "这是角色映射审计，不是假设确认。分别比较每个端点的 contextual_role 与"
        "required_role。相反角色（例如整体与部分、施事与受事、来源与目标）不匹配。"
        "原文可以用主动、被动或逆向表述；不要按词序或字段位置匹配，要按实际语义角色匹配。"
        "只有两个端点都匹配，而且原文语义支持该角色映射时，direction_supported 才为 true；"
        "否则为 false。候选字段和世界知识都不是证据。只返回一个布尔字段。"
        if uses_cjk else
        "This is a role-mapping audit, not hypothesis confirmation. Compare each "
        "endpoint's contextual_role with its required_role. Opposite roles, such as "
        "whole versus part, agent versus patient, or source versus destination, do not "
        "match. The source may use active, passive, or inverse wording; match actual "
        "semantic roles rather than word order or field position. Set direction_supported "
        "true only when both endpoints match and the source semantics support that role "
        "mapping; otherwise false. Candidate fields and world knowledge are not evidence. "
        "Return only one boolean field."
    )
    return json.dumps(
        {
            "instruction": instruction,
            "contract_version": GROUNDED_RELATION_EXACT_ARGUMENT_PROMPT_VERSION,
            "source_text": source_text,
            "first_endpoint": {
                **claim_source,
                "required_role": str(
                    roles.get("relation_source_role") or ""
                ),
                "contextual_role": source_contextual_role,
            },
            "second_endpoint": {
                **claim_target,
                "required_role": str(
                    roles.get("relation_target_role") or ""
                ),
                "contextual_role": target_contextual_role,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_verifier_directional_entailment(
    raw: Any,
    *,
    exact_prompt: str | None = None,
) -> bool:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "verifier directional entailment returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "verifier directional entailment requires one boolean"
        )
    if isinstance(raw.get("direction_supported"), bool):
        return bool(raw["direction_supported"])
    aliases = {
        "source_role": (
            "source_role", "source_role_check", "source_role_satisfied",
            "subject_role_supported",
        ),
        "target_role": (
            "target_role", "target_role_check", "target_role_satisfied",
            "object_role_supported",
        ),
        "direct_relation": (
            "direct_relation", "direct_relation_check", "relation_entailed",
            "entailed", "entailment",
        ),
    }
    values: dict[str, bool] = {}
    for field, names in aliases.items():
        matches = [raw.get(name) for name in names if isinstance(raw.get(name), bool)]
        if len(matches) != 1:
            raise RelationClassificationError(
                "verifier directional entailment requires three booleans"
            )
        values[field] = bool(matches[0])
    if set(values) != set(aliases):
        raise RelationClassificationError(
            "verifier directional entailment requires three booleans"
        )
    # ``exact_prompt`` remains accepted so callers bind the parsed response to
    # the immutable request that produced it. Complete-argument source spans
    # are deliberately audited by the independent predicate-blind probe: a
    # compact three-boolean schema is materially more reliable on local models
    # than duplicating nine fields in every directional call.
    return all(values.values())


def parse_verifier_exact_argument(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "verifier exact-argument check returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "verifier exact-argument response must be an object"
        )
    value = str(
        raw.get("orientation") or raw.get("result") or raw.get("answer")
        or raw.get("decision") or raw.get("label") or raw.get("relation")
        or raw.get("判定结果") or ""
    ).strip().upper().replace("-", "_").replace(" ", "_")
    if not value:
        flags = {
            "FORWARD": raw.get("A_TO_B"),
            "REVERSE": raw.get("B_TO_A"),
            "NEITHER": raw.get("NOT_APPLICABLE"),
            "UNCERTAIN": raw.get("UNCERTAIN"),
        }
        selected = [key for key, enabled in flags.items() if enabled is True]
        value = selected[0] if len(selected) == 1 else ""
    normalized = {
        "FORWARD": "FORWARD", "A_TO_B": "FORWARD",
        "REVERSE": "REVERSE", "B_TO_A": "REVERSE",
        "NEITHER": "NEITHER", "NOT_APPLICABLE": "NEITHER",
        "NONE": "NEITHER", "UNCERTAIN": "UNCERTAIN",
        "UNKNOWN": "UNCERTAIN",
    }.get(value, "")
    if not normalized:
        raise RelationClassificationError(
            "verifier exact-argument check requires one terminal orientation"
        )
    def first_boolean(*names: str) -> bool | None:
        for name in names:
            value = raw.get(name)
            if isinstance(value, bool):
                return value
        return None

    comparisons = {
        "a_matches_source": first_boolean(
            "a_matches_source", "a_matches_source_role",
            "endpoint_a_matches_source_role",
        ),
        "b_matches_target": first_boolean(
            "b_matches_target", "b_matches_target_role",
            "endpoint_b_matches_target_role",
        ),
        "b_matches_source": first_boolean(
            "b_matches_source", "b_matches_source_role",
            "endpoint_b_matches_source_role",
        ),
        "a_matches_target": first_boolean(
            "a_matches_target", "a_matches_target_role",
            "endpoint_a_matches_target_role",
        ),
    }
    entailments = {
        "forward": raw.get("forward_predicate_entailed"),
        "reverse": raw.get("reverse_predicate_entailed"),
    }
    complete_arguments = {
        "a": first_boolean(
            "a_is_complete_argument_mention", "a_complete_argument",
            "complete_argument_a", "endpoint_a_complete_argument",
        ),
        "b": first_boolean(
            "b_is_complete_argument_mention", "b_complete_argument",
            "complete_argument_b", "endpoint_b_complete_argument",
        ),
    }
    if (
        normalized in {"FORWARD", "REVERSE"}
        and False in set(complete_arguments.values())
    ):
        raise RelationClassificationError(
            "verifier exact-argument orientation lacks two complete mentions"
        )
    if (
        normalized == "FORWARD" and entailments["forward"] is False
        or normalized == "REVERSE" and entailments["reverse"] is False
        or normalized not in {"NEITHER", "UNCERTAIN"}
        and entailments["forward"] is False
        and entailments["reverse"] is False
    ):
        raise RelationClassificationError(
            "verifier exact-argument orientation contradicts predicate entailment"
        )
    if (
        all(isinstance(value, bool) for value in comparisons.values())
        and all(isinstance(value, bool) for value in entailments.values())
    ):
        derived = (
            "FORWARD" if comparisons["a_matches_source"]
            and comparisons["b_matches_target"]
            and entailments["forward"]
            else "REVERSE" if comparisons["b_matches_source"]
            and comparisons["a_matches_target"]
            and entailments["reverse"]
            else "NEITHER"
        )
        if normalized not in {derived, "UNCERTAIN"}:
            raise RelationClassificationError(
                "verifier exact-argument orientation contradicts comparisons"
            )
    return normalized


def parse_verifier_role_alignment(raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "verifier role alignment returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "verifier role alignment response must be an object"
        )

    contrast_value = str(
        raw.get("semantic_contrast") or ""
    ).strip().upper().replace("-", "_").replace(" ", "_")
    # The dedicated semantic-contrast call is an independent exact-endpoint
    # gate.  A negative/different meaning is sufficient to reject both
    # orientations even when the broader role-alignment model returns a
    # malformed or abbreviated comparison object.
    if contrast_value in {"NEGATIVE", "NEGATIVE_EXAMPLE", "NEITHER", "OTHER"}:
        return {
            "forward_alignment": "MISALIGNED",
            "reverse_alignment": "MISALIGNED",
        }
    if contrast_value in {"UNCERTAIN", "UNKNOWN"}:
        return {
            "forward_alignment": "UNCERTAIN",
            "reverse_alignment": "UNCERTAIN",
        }
    def normalized(value: Any) -> str:
        result = str(value or "").strip().upper()
        return result if result in {
            "ALIGNED", "MISALIGNED", "UNCERTAIN",
        } else ""

    shared = normalized(raw.get("alignment"))
    forward = normalized(raw.get("forward_alignment"))
    reverse = normalized(raw.get("reverse_alignment"))
    if not forward and not reverse:
        def comparison(value: Any) -> str:
            result = str(value or "").strip().upper().replace("-", "_")
            if result in {"MATCH", "MATCHED", "YES", "TRUE", "ALIGNED"}:
                return "MATCH"
            if result in {
                "MISMATCH", "MISMATCHED", "NO", "FALSE", "MISALIGNED",
            }:
                return "MISMATCH"
            if result in {"UNCERTAIN", "UNKNOWN", "UNRESOLVED"}:
                return "UNCERTAIN"
            return ""

        a_source = comparison(
            raw.get("endpoint_a_vs_source")
            or raw.get("endpoint_a_role_vs_relation_source_role")
        )
        b_target = comparison(
            raw.get("endpoint_b_vs_target")
            or raw.get("endpoint_b_role_vs_relation_target_role")
        )
        b_source = comparison(
            raw.get("endpoint_b_vs_source")
            or raw.get("endpoint_b_role_vs_relation_source_role")
        )
        a_target = comparison(
            raw.get("endpoint_a_vs_target")
            or raw.get("endpoint_a_role_vs_relation_target_role")
        )
        if all((a_source, b_target, b_source, a_target)):
            def orientation(first: str, second: str) -> str:
                if first == second == "MATCH":
                    return "ALIGNED"
                if "MISMATCH" in {first, second}:
                    return "MISALIGNED"
                return "UNCERTAIN"

            forward = orientation(a_source, b_target)
            reverse = orientation(b_source, a_target)
            semantic_forward = comparison(
                raw.get("forward_relation_vs_definition")
            )
            semantic_reverse = comparison(
                raw.get("reverse_relation_vs_definition")
            )
            contrast_raw = str(
                raw.get("semantic_contrast") or ""
            ).strip().upper().replace("-", "_").replace(" ", "_")
            contrast = {
                "DEFINITION": "DEFINITION",
                "MATCH": "DEFINITION",
                "POSITIVE": "DEFINITION",
                "NEGATIVE": "NEGATIVE_EXAMPLE",
                "NEGATIVE_EXAMPLE": "NEGATIVE_EXAMPLE",
                "NEITHER": "NEITHER",
                "OTHER": "NEITHER",
                "UNCERTAIN": "UNCERTAIN",
                "UNKNOWN": "UNCERTAIN",
            }.get(contrast_raw, "")
            if contrast in {"NEGATIVE_EXAMPLE", "NEITHER"}:
                forward = reverse = "MISALIGNED"
            elif contrast == "UNCERTAIN":
                forward = reverse = "UNCERTAIN"
            elif semantic_forward and semantic_reverse:
                semantic_forward_alignment = orientation(
                    semantic_forward, semantic_forward,
                )
                semantic_reverse_alignment = orientation(
                    semantic_reverse, semantic_reverse,
                )
                # Relation meaning is the primary orientation gate.  Endpoint
                # role comparisons remain an independent conflict check.  The
                # aggregation itself is purely structural; every semantic
                # MATCH/MISMATCH value comes from the independent model call.
                if contrast and (
                    semantic_forward_alignment == "ALIGNED"
                    and semantic_reverse_alignment != "ALIGNED"
                ):
                    if reverse == "ALIGNED" and forward == "MISALIGNED":
                        forward = reverse = "UNCERTAIN"
                    else:
                        forward, reverse = "ALIGNED", "MISALIGNED"
                elif contrast and (
                    semantic_reverse_alignment == "ALIGNED"
                    and semantic_forward_alignment != "ALIGNED"
                ):
                    if forward == "ALIGNED" and reverse == "MISALIGNED":
                        forward = reverse = "UNCERTAIN"
                    else:
                        forward, reverse = "MISALIGNED", "ALIGNED"
                elif contrast and (
                    semantic_forward_alignment == "ALIGNED"
                    and semantic_reverse_alignment == "ALIGNED"
                ):
                    forward = reverse = "ALIGNED"
                elif contrast and "UNCERTAIN" in {
                    semantic_forward_alignment, semantic_reverse_alignment,
                }:
                    forward = reverse = "UNCERTAIN"
                elif contrast:
                    forward = reverse = "MISALIGNED"
    # Compatibility for frozen v1 ledgers and scripted fixtures.  The v1
    # endpoint-wise contract could establish only the forward orientation; it
    # must never be interpreted as evidence for automatic reversal.
    if not forward:
        subject = normalized(
            raw.get("subject_role") or raw.get("subject_alignment")
        ) or shared
        object_ = normalized(
            raw.get("object_role") or raw.get("object_alignment")
        ) or shared
        if subject and object_:
            forward = (
                "ALIGNED" if subject == object_ == "ALIGNED"
                else "UNCERTAIN" if "UNCERTAIN" in {subject, object_}
                else "MISALIGNED"
            )
            reverse = reverse or "UNCERTAIN"
    if not forward or not reverse:
        raise RelationClassificationError(
            "verifier role alignment requires both orientation decisions"
        )
    return {
        "forward_alignment": forward,
        "reverse_alignment": reverse,
    }


def _source_syntax_projection_contract(
    roles: Sequence[Any],
    *,
    syntax_origin: str,
) -> dict[str, Any]:
    """Describe parser-recovered binary roles without choosing a predicate.

    Multi-role clauses license several endpoint pairs, but those pairs do not
    inherit the same semantic object role.  This contract makes the omitted
    third role explicit to both Schema binding and final verification while
    remaining independent of language, domain terms and predicate names.
    """

    pair = tuple(str(value or "") for value in roles)
    if syntax_origin not in {
        "ordered_three_role_construction",
        "explicit_role_connector",
        "possessive_reference_construction",
        "passive_agent_postpredicate_construction",
    }:
        return {}
    contracts: dict[tuple[str, str], dict[str, Any]] = {
        ("grammatical_subject", "action_theme"): {
            "projection_kind": "actor_to_theme",
            "subject_source_role": "grammatical_subject",
            "object_source_role": "action_theme",
            "entailed_role_pair": (
                "The subject performs or governs the source action and the "
                "object is its acted, transferred, or described theme."
            ),
            "excluded_role": (
                "The object is not the connector-introduced recipient, "
                "destination, source, location, instrument, or other complement."
            ),
        },
        ("action_theme", "role_complement"): {
            "projection_kind": "theme_to_connector_complement",
            "subject_source_role": "action_theme",
            "object_source_role": "role_complement",
            "entailed_role_pair": (
                "The subject is the action theme and the object is the exact "
                "recipient, destination, source, location, instrument, or other "
                "role introduced by the explicit connector."
            ),
            "excluded_role": (
                "The grammatical actor is a third role and must not replace "
                "either endpoint."
            ),
        },
        ("grammatical_subject", "role_complement"): {
            "projection_kind": "actor_to_connector_complement",
            "subject_source_role": "grammatical_subject",
            "object_source_role": "role_complement",
            "entailed_role_pair": (
                "The subject is the action actor and the object is the exact "
                "connector-introduced complement."
            ),
            "excluded_role": (
                "The action theme is a third role. This pair cannot satisfy a "
                "Schema predicate whose object role is the acted, transferred, "
                "or described theme merely because the governing verb matches."
            ),
        },
        ("possessor", "possessed"): {
            "projection_kind": "possessive_reference_to_possessed_entity",
            "subject_source_role": "possessor",
            "object_source_role": "possessed",
            "entailed_role_pair": (
                "The subject is the independently resolved antecedent of the "
                "possessive determiner and the object is the exact noun phrase "
                "introduced by that determiner."
            ),
            "excluded_role": (
                "The possessive construction alone does not choose ownership, "
                "part-whole, authorship, affiliation, or another logical "
                "predicate; the current Schema definition must match the full "
                "source context."
            ),
        },
    }
    return contracts.get(pair, {})


def _independently_audited_frame_context(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
) -> dict[str, Any] | None:
    """Return an exact frame display after an independent ASSERTED audit.

    The immutable opportunity remains the parser's trust boundary.  This
    helper only narrows what the second verifier sees, and only when the
    source-frame span is byte-for-byte grounded, lies in that opportunity,
    and covers both exact endpoint occurrences.  Any malformed audit metadata
    falls back to the ordinary context instead of broadening trust.
    """

    metadata = hypothesis.predicate_metadata or {}
    if (
        str(metadata.get("source_frame_polarity_audit") or "").upper()
        != "ASSERTED"
        or not str(
            metadata.get("source_frame_polarity_audit_version") or ""
        ).strip()
    ):
        return None
    raw_span = metadata.get("source_frame_evidence_span")
    if not (
        isinstance(raw_span, (list, tuple))
        and len(raw_span) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw_span
        )
    ):
        return None
    start, end = int(raw_span[0]), int(raw_span[1])
    source_start, source_end = opportunity.context_span
    if not (
        source_start <= start < end <= source_end
        and end <= len(content)
        and start <= opportunity.left_span[0] < opportunity.left_span[1] <= end
        and start <= opportunity.right_span[0] < opportunity.right_span[1] <= end
    ):
        return None
    frame_text = str(metadata.get("source_frame_evidence_text") or "")
    if not frame_text or content[start:end] != frame_text:
        return None
    return {
        "source_span": [int(source_start), int(source_end)],
        "display_span": [start, end],
        "visible_segments": [[start, end]],
        "text": frame_text,
        "truncated": (start, end) != (source_start, source_end),
    }


def verifier_prompt(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    blind_role_audit: Mapping[str, str],
) -> str:
    """Build one independent, directed triple verification request."""

    spec = normalize_predicate_schema(logical_schema).get(hypothesis.predicate)
    if spec is None:
        raise ValueError("verifier requires a current predicate definition")
    subject = _occurrence_details(
        opportunity, hypothesis.subject_occurrence_key, content,
    )
    object_ = _occurrence_details(
        opportunity, hypothesis.object_occurrence_key, content,
    )
    context_window = _pair_model_context(
        opportunity, content=content,
    )
    audited_frame_context = _independently_audited_frame_context(
        hypothesis, opportunity, content=content,
    )
    if audited_frame_context is not None:
        context_window = audited_frame_context
    context_text = str(context_window["text"])
    uses_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", context_text))
    structural_kind = ""
    if _has_hierarchical_list_parent_child_evidence(opportunity):
        structural_kind = "list_parent_child"
    elif _has_hierarchical_section_label_parent_child_evidence(opportunity):
        structural_kind = "section_label_parent_child"
    else:
        structural_kind = str(
            (hypothesis.predicate_metadata or {}).get(
                "source_structure_kind"
            )
            or opportunity.provenance.get("source_structure_kind")
            or ""
        )
    structural_child_key = ""
    structural_parent_key = ""
    if structural_kind:
        structural_child_key, structural_parent_key = _structural_parent_child_keys(
            opportunity
        )
    structural_schema_match = bool(
        structural_kind and structural_kind in spec.structural_evidence_kinds
    )
    structural_binding_note = (
        "当前候选具有 Schema 明确声明、且由源解析器冻结验证的结构证据。该结构绑定本身就是原文断言；"
        "只能按给定 source_structure_kind、角色和当前主体/客体核验，不能扩展到兄弟单元、其他列或普通相邻项。"
        if uses_cjk and structural_schema_match else
        "This candidate has a Schema-declared structural evidence kind validated by the "
        "frozen source parser. The exact structural binding is itself the source assertion. "
        "Verify only the supplied source_structure_kind, roles, subject and object; never "
        "extend it to sibling units, other columns, or ordinary adjacency."
        if structural_schema_match else ""
    )
    source_binding_source = str(
        (hypothesis.predicate_metadata or {}).get("source_binding_source") or ""
    )
    source_frame_relation = str(
        (hypothesis.predicate_metadata or {}).get("source_frame_relation") or ""
    )
    source_frame_roles = list(
        (hypothesis.predicate_metadata or {}).get("source_frame_roles") or []
    )
    source_syntax_origin = str(
        (hypothesis.predicate_metadata or {}).get("source_syntax_origin") or ""
    )
    source_projection_contract = _source_syntax_projection_contract(
        source_frame_roles,
        syntax_origin=source_syntax_origin,
    )
    source_frame_assertion_status = str(
        (hypothesis.predicate_metadata or {}).get(
            "source_frame_assertion_status"
        ) or ""
    ).upper()
    source_frame_explicit_negation = bool(
        (hypothesis.predicate_metadata or {}).get(
            "source_frame_explicit_negation"
        )
    )
    source_resolved_references = [
        dict(item) for item in (
            (hypothesis.predicate_metadata or {}).get(
                "source_resolved_reference_endpoints"
            ) or ()
        )
        if isinstance(item, Mapping)
    ]
    source_frame_evidence_span = list(
        (hypothesis.predicate_metadata or {}).get(
            "source_frame_evidence_span"
        ) or []
    )
    source_frame_evidence_text = str(
        (hypothesis.predicate_metadata or {}).get(
            "source_frame_evidence_text"
        ) or ""
    )
    source_frame_polarity_audit = str(
        (hypothesis.predicate_metadata or {}).get(
            "source_frame_polarity_audit"
        ) or ""
    ).upper()
    source_frame_polarity_audit_version = str(
        (hypothesis.predicate_metadata or {}).get(
            "source_frame_polarity_audit_version"
        ) or ""
    )
    source_frame_note = (
        (
            " source_frame_candidate 只是候选生成审计，不是事实证明。必须逐项比较开放谓词、论元角色与当前"
            "Schema 关系定义；若只是词面相关、共享第三实体或角色不等价，必须拒绝。若提供了"
            "projection_contract，它同时规定当前二元投影包含和排除的角色；不能把省略的第三角色"
            "偷换成当前客体，也不能仅因支配动词相同就通过。"
            if uses_cjk else
            " source_frame_candidate is candidate-generation audit data, not proof. Compare "
            "its open predicate and argument roles with the current Schema definition. Reject "
            "lexical relatedness, a shared third entity, or non-equivalent roles. When a "
            "projection_contract is present, it defines both included and excluded roles for "
            "this exact binary projection. Never substitute the omitted third role or accept "
            "a Schema object role merely because the governing verb matches."
        )
        if source_binding_source or source_frame_relation else ""
    )
    open_relation_carrier = bool(
        spec.open_relation_policy.get("enabled")
    )
    if open_relation_carrier:
        source_frame_note = (
            (
                " 当前关系策略是有证据开放关系载体，而不是可独立解释的通用谓词。"
                "source_frame_candidate.open_relation 给出的原文关系表达就是本次待核验的"
                "具体关系语义；必须按它的原文论元顺序和精确证据核验，不能按载体名称或"
                "宽泛 Schema 描述放行，也不能把它扩展成共现、相邻或相关性。若存在"
                "projection_contract，当前主体/客体角色以该二元投影为准："
                "action_theme 作为主体时不得再要求它同时是 grammatical actor；"
                "被省略且排除的第三角色也不得替换任一端点。"
            )
            if uses_cjk else
            " This relation policy is a grounded open-relation carrier, not an "
            "independently meaningful generic predicate. The exact "
            "source_frame_candidate.open_relation is the relation semantics being "
            "verified. Apply its source argument order and exact evidence; never "
            "approve from the carrier label or broad Schema description, and never "
            "broaden it into co-occurrence, adjacency, or relatedness. When a "
            "projection_contract is present, its binary subject/object roles govern: "
            "an action_theme subject must not also be required to be the grammatical "
            "actor, and the omitted third role identified as excluded must not replace "
            "either endpoint."
        )
    if open_relation_carrier and source_frame_relation:
        if source_projection_contract:
            claim_relation_definition = " ".join((
                f'Source expression: "{source_frame_relation}".',
                "Exact binary projection:",
                str(source_projection_contract.get("projection_kind") or ""),
                str(source_projection_contract.get("entailed_role_pair") or ""),
                "Excluded role:",
                str(source_projection_contract.get("excluded_role") or ""),
            )).strip()
        else:
            claim_relation_definition = source_frame_relation
    else:
        claim_relation_definition = spec.semantic_definition
    instruction = (
        "只依据冻结原文，独立核验完整且有方向的（主体、关系、客体）事实。必须同时核验"
        "关系定义、两端角色、方向、否定、情态、引用语境和精确出现位置。SUPPORTED 只"
        "允许用于原文直接断言、且两个精确端点分别承担 Schema 所定义角色的关系；共现、"
        "否定范围必须落实到当前谓词：同一上下文中的主句或并列谓词被否定，不会仅因共用"
        "上下文就否定嵌入或并列的当前 frame。source_frame_candidate 中的 assertion_status"
        "和 explicit_relation_negation 是需要结合原文复核的 frame 局部证据，不是自动结论；"
        "只有否定词实际支配当前开放谓词时，才可按否定拒绝。frame_evidence 的精确区间和"
        "文本标识源盘点提出的局部子句；应先审计该子句，再只用较宽上下文解析指代与范围。"
        "它仍是候选证据而不是自动事实证明。"
        "若 resolved_reference_endpoints 非空，当前代词或物主限定词端点是所列冻结先行词的"
        "精确证据代理；应在先行词实体上核验 Schema 角色，同时要求关系在该指代位置被原文"
        "断言。不得仅因代理词面不是名词短语而拒绝，也不得把解析扩展到未列出的实体。"
        "标题、表格或列表相邻、外部常识都不够。但若 structural_evidence_contract 明确标记"
        "hierarchical_parent_child=true，则这不是普通的列表相邻，而是源解析器确认的同一"
        "列表组中的直接父子结构；只有当当前 Schema 的定义、上下文类型和端点角色支持该"
        "结构语义时，才可将它作为该候选的证据。若 section_label_parent_child=true，"
        "表示源结构确认独立标签紧邻其直接内容单元；只能用于 Schema 定义的标签/内容层级，"
        "不得把普通段落相邻升级为关系。不得由此推导兄弟项或非直接父子项的关系。"
        "一个谓词支配多个并列成员时，应分别判断"
        "支配项与每个成员；并列同级成员不会仅因共享支配项而彼此产生关系。不得用端点所"
        "提供、拥有、包含、修饰或限定的第三实体替代该端点。事件、状态或条件端点若缺少"
        "触发词、状态变化、否定、情态或中心词，不能视为完整论元。证据不得越过要求的原子"
        "范围。分别判断正向命题和交换端点后的反向命题。反向审计是把同一个 Schema 谓词"
        "应用于交换后的端点角色，不是在问原事实能否改写成一个含义相反的逆谓词；主动/被动或"
        "整体优先的表述只支持实际满足 Schema 主体/客体角色的一个方向，不能因此把同一非对称"
        "谓词判成两向成立。非对称关系只有恰好一个方向为 "
        "ENTAILED、另一方向为 NOT_ENTAILED，且该方向两端角色均成立时才能 SUPPORTED；"
        "对称关系必须两向都 ENTAILED。subject_role_supported 和 "
        "object_role_supported 只核验被判为 ENTAILED 的方向中两个端点是否承担所需角色；"
        "交换端点后的方向为 NOT_ENTAILED 只表示该反向命题不成立，不能因此把这两个角色"
        "布尔值判为 false。名词组合、话题关联或只出现类似关系词都不算证据。"
        "正反例只用于对比语义，不是原文证据。不得读取或沿用分类器、端点探针或角色对齐器"
        "的判断结论。只返回 verdict、reason_code、forward_entailment、"
        "reverse_entailment、subject_role_supported、object_role_supported、"
        "supported_evidence_spans。reason_code 只写不超过 160 个字符的简短诊断标签，"
        "不得输出分析过程或解释性段落。verdict 只能是 SUPPORTED、REJECTED 或 UNCERTAIN；"
        "蕴含字段只能是 ENTAILED、NOT_ENTAILED 或 UNCERTAIN。只有 SUPPORTED 才能给出"
        "绝对半开证据区间，否则证据数组必须为空。"
        if uses_cjk else
        "Use only the frozen source. Independently verify the complete directed "
        "(subject,predicate,object) fact, including Schema meaning, endpoint roles, "
        "direction, negation, modality, quotation context, and exact occurrences. "
        "SUPPORTED requires a direct assertion between those exact arguments; "
        "Negation scope is predicate-local: a negated matrix or sibling predicate "
        "does not negate an embedded or coordinated source frame merely because both "
        "appear in the same context. Treat source_frame_candidate.assertion_status "
        "and explicit_relation_negation as auditable frame-local evidence, not an "
        "automatic verdict; reject a claim for negation only when negation governs "
        "this frame's open predicate. The exact frame_evidence span/text identifies "
        "the local clause proposed by the source inventory; audit that clause first, "
        "then use the wider context only for reference and scope resolution. It remains "
        "candidate evidence rather than an automatic proof. "
        "When resolved_reference_endpoints is non-empty, the endpoint's local "
        "pronoun or possessive determiner is an exact evidence proxy for the listed "
        "frozen antecedent. Evaluate the Schema role on that antecedent entity while "
        "still requiring the relation to be asserted at the reference span; do not "
        "reject solely because the proxy surface is not a noun phrase, and do not "
        "extend the resolution to any unlisted entity. "
        "The displayed context may be a bounded endpoint-centered window; omitted source "
        "ranges are unavailable and cannot support a relation, while all offsets are "
        "absolute coordinates in the full frozen source. "
        "co-occurrence, headings, ordinary table/list adjacency, and external knowledge "
        "are not evidence. If structural_evidence_contract explicitly marks "
        "hierarchical_parent_child=true, it is not ordinary adjacency: the source "
        "mapper has confirmed an immediate parent-child edge in one list group. Use "
        "that structural evidence only when the current Schema definition, context "
        "kind, and endpoint roles support the structural meaning; never infer sibling "
        "or non-immediate-child relations from it. If section_label_parent_child=true, "
        "the source mapper has confirmed a standalone label directly precedes the child "
        "scope; use it only for a Schema-defined label/child hierarchy, never for ordinary "
        "paragraph adjacency. Expand a shared grammatical predicate from its governor to each "
        "coordinated member. Coordinated peers can support a relation between each "
        "other only when the source predicate itself explicitly has reciprocal or "
        "collective participant semantics matching the Schema; sharing an ordinary "
        "action or object is insufficient. Never "
        "substitute a third entity provided, owned, contained, modified, or qualified "
        "by an endpoint. An event, state, or condition missing its trigger, state "
        "change, negation, modality, or phrase head is not a complete argument. Keep "
        "evidence inside the required atomic scope. Judge forward and endpoint-swapped "
        "reverse claims independently by applying the exact same Schema predicate to "
        "the swapped endpoint roles. The reverse audit does not ask whether the source "
        "fact can be paraphrased with an inverse predicate. Active/passive or whole-first "
        "wording supports only the orientation whose endpoints satisfy the declared "
        "Schema subject and object roles; it never makes the same asymmetric predicate "
        "true in both directions. A non-symmetric relation is SUPPORTED only when "
        "exactly one direction is ENTAILED and both endpoint roles hold; a symmetric "
        "relation requires both directions. subject_role_supported and "
        "object_role_supported judge the required endpoint roles in the orientation "
        "marked ENTAILED. A swapped orientation marked NOT_ENTAILED only means that "
        "reverse claim fails; it must not make either role boolean false. Noun "
        "compounds, topical association, and a "
        "predicate-like word alone are insufficient. Examples are contrastive guidance, "
        "not source facts. Do not inherit conclusions from a classifier, endpoint probe, "
        "or role-alignment model. Return only the "
        "seven required fields. reason_code must be one concise diagnostic label of at "
        "most 160 characters, never analysis or an explanatory paragraph. Use SUPPORTED, "
        "REJECTED, or UNCERTAIN for verdict and "
        "ENTAILED, NOT_ENTAILED, or UNCERTAIN for entailment. Evidence spans are absolute "
        "half-open ranges and must be empty unless verdict is SUPPORTED."
    )
    if structural_binding_note:
        instruction += " " + structural_binding_note
    if source_frame_note:
        instruction += source_frame_note
    if (
        source_frame_polarity_audit == "ASSERTED"
        and source_frame_polarity_audit_version
    ):
        instruction += (
            " independent_polarity_audit 只确认否定词没有支配当前开放 frame；它不确认"
            "Schema 谓词、端点角色或方向。仍须独立通过本请求的全部语义、角色、方向和证据门槛。"
            if uses_cjk else
            " independent_polarity_audit only confirms that source negation does not "
            "govern this open frame. It does not confirm the Schema predicate, endpoint "
            "roles, or direction; independently apply every semantic, role, direction, "
            "and evidence gate in this request."
        )
    return json.dumps(
        {
            "instruction": instruction,
            "contract_version": GROUNDED_RELATION_VERIFIER_PROMPT_VERSION,
            "output_contract": {
                "exact_keys": [
                    "verdict",
                    "reason_code",
                    "forward_entailment",
                    "reverse_entailment",
                    "subject_role_supported",
                    "object_role_supported",
                    "supported_evidence_spans",
                ],
                "forbidden_aliases": [
                    "entailment", "evidence_spans", "verification",
                ],
            },
            "context": {
                "start": int(
                    (context_window.get("display_span") or opportunity.context_span)[0]
                ),
                "end": int(
                    (context_window.get("display_span") or opportunity.context_span)[1]
                ),
                "immutable_source_span": list(opportunity.context_span),
                "text": context_text,
                "visible_segments": context_window["visible_segments"],
                "truncated": bool(context_window["truncated"]),
            },
            "subject": subject,
            # The verifier is deliberately blind to the predicate identifier.
            # Runtime behavior must follow versioned semantics and roles, so a
            # metadata-preserving predicate rename cannot bias the model.
            "relation_policy": {
                **compact_predicate_prompt_metadata(
                    asdict(spec), include_name=False,
                ),
            },
            "object": object_,
            "ordered_claims": {
                "forward": {
                    "subject_surface": subject["surface"],
                    "relation_definition": claim_relation_definition,
                    "object_surface": object_["surface"],
                },
                "reverse": {
                    "subject_surface": object_["surface"],
                    "relation_definition": claim_relation_definition,
                    "object_surface": subject["surface"],
                },
                "symmetric": bool(spec.symmetric),
                "orientation_audit_contract": {
                    "apply_same_schema_predicate_after_swap": True,
                    "inverse_paraphrase_is_not_reverse_entailment": True,
                    "forward_subject_required_role": (
                        spec.subject_role_description
                    ),
                    "forward_object_required_role": (
                        spec.object_role_description
                    ),
                    "reverse_subject_required_role": (
                        spec.subject_role_description
                    ),
                    "reverse_object_required_role": (
                        spec.object_role_description
                    ),
                },
            },
            "classifier_evidence_spans": [
                asdict(value) for value in hypothesis.evidence_spans
            ],
            "source_frame_candidate": {
                "binding_source": source_binding_source,
                "open_relation": source_frame_relation,
                "assertion_status": source_frame_assertion_status,
                "explicit_relation_negation": (
                    source_frame_explicit_negation
                ),
                "resolved_reference_endpoints": (
                    source_resolved_references
                ),
                "frame_evidence": {
                    "span": source_frame_evidence_span,
                    "text": source_frame_evidence_text,
                },
                "independent_polarity_audit": {
                    "result": source_frame_polarity_audit,
                    "version": source_frame_polarity_audit_version,
                },
                "argument_roles": source_frame_roles,
                "syntax_origin": source_syntax_origin,
                "projection_contract": source_projection_contract,
                "structure_kind": structural_kind,
                "shortlist_score": (
                    hypothesis.predicate_metadata or {}
                ).get("source_binding_shortlist_score"),
            },
            "structural_evidence_contract": {
                "source_structure_kind": structural_kind,
                "hierarchical_parent_child": _has_hierarchical_list_parent_child_evidence(
                    opportunity
                ),
                "section_label_parent_child": _has_hierarchical_section_label_parent_child_evidence(
                    opportunity
                ),
                "structure_admission": str(
                    opportunity.provenance.get("structure_admission") or ""
                ),
                "source_roles": list(
                    opportunity.provenance.get("source_roles") or []
                ),
                "atomic_scopes": list(
                    opportunity.provenance.get("atomic_scopes") or []
                ),
                "required_atomic_scope": list(
                    opportunity.provenance.get("required_atomic_scope") or []
                ),
                "schema_scope_policy": dict(spec.scope_policy),
                "required_evidence_roles": list(spec.required_evidence_roles),
                "structural_evidence_kinds": list(spec.structural_evidence_kinds),
                "schema_structural_match": structural_schema_match,
                "candidate_binding": {
                    "kind": structural_kind,
                    "child_occurrence_key": structural_child_key,
                    "parent_occurrence_key": structural_parent_key,
                    "subject_is_child": (
                        structural_child_key == hypothesis.subject_occurrence_key
                    ),
                    "object_is_parent": (
                        structural_parent_key == hypothesis.object_occurrence_key
                    ),
                },
            },
            "hypothesis_id": hypothesis.hypothesis_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def structural_verifier_prompt(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> str:
    """Build a narrow independent adjudication request for Schema structure.

    A general verifier can reject a valid label/child edge because no lexical
    predicate appears between the two spans.  This request asks the same
    verifier model to adjudicate the immutable structural assertion explicitly;
    it is only used after the normal verifier disagrees and only for a
    Schema-declared structural kind.  The resulting response still passes the
    ordinary parser, role gate, evidence gate and normalization.
    """

    spec = normalize_predicate_schema(logical_schema).get(hypothesis.predicate)
    if spec is None:
        raise ValueError("structural verifier requires current predicate metadata")
    child, parent = _structural_parent_child_keys(opportunity)
    if _has_hierarchical_list_parent_child_evidence(opportunity):
        kind = "list_parent_child"
    elif _has_hierarchical_section_label_parent_child_evidence(opportunity):
        kind = "section_label_parent_child"
    else:
        raise ValueError("structural verifier requires structural evidence")
    subject = _occurrence_details(
        opportunity, hypothesis.subject_occurrence_key, content,
    )
    object_ = _occurrence_details(
        opportunity, hypothesis.object_occurrence_key, content,
    )
    child_details = _occurrence_details(opportunity, child, content)
    parent_details = _occurrence_details(opportunity, parent, content)
    context_window = _pair_model_context(
        opportunity,
        content=content,
        extra_focus_spans=(child_details["span"], parent_details["span"]),
    )
    context_text = str(context_window["text"])
    uses_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", context_text))
    instruction = (
        "这是一次独立的结构证据复核。冻结源解析器确认了一个结构类型为 "
        f"{kind} 的直接子项→父项绑定：子项是“{child_details['surface']}”，"
        f"父项是“{parent_details['surface']}”。当前 Schema 声明该结构类型可以支持"
        "此谓词。标签与子项之间可能没有词汇谓词；这不是普通相邻或共现，而是源结构"
        "断言。只有当前主体正好是子项、客体正好是父项，且 Schema 定义与端点类型允许时，"
        "正向命题才是 SUPPORTED；反向命题必须拒绝。请只返回结构裁决协议的 decision、"
        "binding、schema_compatible、evidence_spans、reason_code、hypothesis_id 六个字段，"
        "不要返回普通谓词核验协议。SUPPORTED 时 evidence_spans 使用冻结 context_span，"
        "binding 必须为 CHILD_TO_PARENT，schema_compatible 必须为 true。"
        "context 可能是有界窗口；省略的来源区间不可用于判断，返回坐标必须使用完整来源的绝对坐标。"
        if uses_cjk else
        "This is an independent structural-evidence adjudication. The frozen source mapper "
        f"confirmed a direct child-to-parent binding of kind {kind}: child "
        f"'{child_details['surface']}' and parent '{parent_details['surface']}'. The current "
        "Schema declares this structural kind admissible for the predicate. A label and its "
        "child may have no lexical predicate between them; this is a source structural "
        "assertion, not ordinary adjacency or co-occurrence. Support only when the current "
        "subject is exactly the child, the object exactly the parent, and Schema meaning/types "
        "allow it; the reverse claim must be rejected. Return only the structural adjudication "
        "fields decision, binding, schema_compatible, evidence_spans, reason_code and "
        "hypothesis_id, never the ordinary predicate-verifier fields. For SUPPORTED use the "
        "frozen context span, binding CHILD_TO_PARENT and schema_compatible true. The displayed "
        "context may be bounded; omitted source ranges cannot support the decision, and all "
        "offsets remain absolute coordinates."
    )
    return json.dumps(
        {
            "instruction": instruction,
            "contract_version": GROUNDED_RELATION_VERIFIER_PROMPT_VERSION,
            "context": {
                "start": opportunity.context_span[0],
                "end": opportunity.context_span[1],
                "text": context_text,
                "visible_segments": context_window["visible_segments"],
                "truncated": bool(context_window["truncated"]),
            },
            "subject": subject,
            "object": object_,
            "relation_policy": {
                **compact_predicate_prompt_metadata(
                    asdict(spec), include_name=False,
                ),
            },
            "ordered_claims": {
                "forward": {
                    "subject_surface": subject["surface"],
                    "relation_definition": spec.semantic_definition,
                    "object_surface": object_["surface"],
                },
                "reverse": {
                    "subject_surface": object_["surface"],
                    "relation_definition": spec.semantic_definition,
                    "object_surface": subject["surface"],
                },
                "symmetric": bool(spec.symmetric),
            },
            "structural_binding": {
                "kind": kind,
                "child_occurrence_key": child,
                "parent_occurrence_key": parent,
                "subject_is_child": hypothesis.subject_occurrence_key == child,
                "object_is_parent": hypothesis.object_occurrence_key == parent,
            },
            "adjudication_mode": "STRUCTURAL_SOURCE_ASSERTION",
            "hypothesis_id": hypothesis.hypothesis_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def structural_verifier_response_schema(hypothesis_id: str) -> dict[str, Any]:
    """Return the compact response contract for structure adjudication.

    Structural evidence is not a lexical predicate frame. Reusing the full
    ordered-verifier schema makes a model answer the wrong question by
    reporting unresolved verb roles even when the immutable parser has already
    established the child and parent.
    """

    return {
        "$id": GROUNDED_RELATION_STRUCTURAL_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "decision", "binding", "schema_compatible",
            "evidence_spans", "reason_code", "hypothesis_id",
        ],
        "properties": {
            "decision": {
                "type": "string",
                "enum": [SUPPORTED, REJECTED, UNCERTAIN],
            },
            "binding": {
                "type": "string",
                "enum": ["CHILD_TO_PARENT", "REVERSE", "NOT_CONFIRMED"],
            },
            "schema_compatible": {"type": "boolean"},
            "evidence_spans": {
                "type": "array",
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end"],
                    "properties": {
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 1},
                    },
                },
            },
            "reason_code": {
                "type": "string", "minLength": 1, "maxLength": 160,
            },
            "hypothesis_id": {"type": "string"},
        },
    }


def _parse_structural_verifier_response(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    raw: Any,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    blind_role_audit: Mapping[str, str],
    direction_normalization: str,
    attempt_no: int,
    verifier_model_revision: str,
) -> VerificationAttempt:
    """Convert a compact structural decision into the common audit record."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "structural verifier returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "structural verifier response must be an object"
        )
    response_hypothesis_id = raw.get("hypothesis_id")
    if (
        response_hypothesis_id is None
        or str(response_hypothesis_id) != hypothesis.hypothesis_id
    ):
        raise RelationClassificationError(
            "structural verifier hypothesis id mismatched"
        )
    decision = str(raw.get("decision") or "")
    if decision not in {SUPPORTED, REJECTED, UNCERTAIN}:
        raise RelationClassificationError(
            "structural verifier decision is invalid"
        )
    binding = str(raw.get("binding") or "").strip().upper()
    if binding not in {"CHILD_TO_PARENT", "REVERSE", "NOT_CONFIRMED"}:
        raise RelationClassificationError(
            "structural verifier binding is invalid"
        )
    schema_compatible = raw.get("schema_compatible")
    if not isinstance(schema_compatible, bool):
        raise RelationClassificationError(
            "structural verifier Schema compatibility is required"
        )
    reason_code = str(raw.get("reason_code") or "").strip()
    if not reason_code:
        raise RelationClassificationError(
            "structural verifier reason_code is required"
        )
    raw_spans = raw.get("evidence_spans") or []
    if not isinstance(raw_spans, list):
        raise RelationClassificationError(
            "structural verifier evidence spans must be an array"
        )
    # Some schema-constrained local models wrap the requested pair in a
    # descriptive record: {"span": [start, end], "text": "..."}.  The text
    # is not trusted or persisted here; only the exact numeric span is
    # admitted, then it passes the same frozen-context checks as the canonical
    # {"start": ..., "end": ...} form.
    normalized_raw_spans: list[Any] = []
    for value in raw_spans:
        if (
            isinstance(value, Mapping)
            and "start" not in value
            and "end" not in value
            and "span" in value
        ):
            span = value.get("span")
            if (
                not isinstance(span, Sequence)
                or isinstance(span, (str, bytes, bytearray, Mapping))
                or len(span) != 2
            ):
                raise RelationClassificationError(
                    "structural verifier wrapped evidence span is invalid"
                )
            normalized_raw_spans.append({"start": span[0], "end": span[1]})
        else:
            normalized_raw_spans.append(value)
    raw_spans = normalized_raw_spans
    try:
        spans = _parse_spans(
            raw_spans,
            opportunity=opportunity,
            content=content,
        )
    except RelationClassificationError:
        # Structural prompts show a frozen context with its absolute start,
        # but compact local models occasionally return offsets relative to
        # that displayed context.  Accept only this unambiguous coordinate
        # conversion; all endpoint, frozen-context and atomic-scope gates
        # still run on the converted absolute spans below.
        context_start, context_end = opportunity.context_span
        context_length = context_end - context_start
        relative_spans: list[dict[str, int]] = []
        try:
            for value in raw_spans:
                if (
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray, Mapping))
                    and len(value) == 2
                ):
                    start, end = value[0], value[1]
                elif isinstance(value, Mapping):
                    start, end = value["start"], value["end"]
                else:
                    raise ValueError("structural evidence span shape")
                start, end = int(start), int(end)
                if not (0 <= start < end <= context_length):
                    raise ValueError("structural relative span bounds")
                relative_spans.append({
                    "start": context_start + start,
                    "end": context_start + end,
                })
        except (KeyError, TypeError, ValueError) as exc:
            raise RelationClassificationError(
                "structural evidence span is invalid in absolute and context-relative coordinates"
            ) from exc
        spans = _parse_spans(
            relative_spans,
            opportunity=opportunity,
            content=content,
        )
    child, parent = _structural_parent_child_keys(opportunity)
    exact_binding = bool(
        hypothesis.subject_occurrence_key == child
        and hypothesis.object_occurrence_key == parent
    )
    if decision == SUPPORTED:
        if binding != "CHILD_TO_PARENT" or not schema_compatible:
            raise RelationVerificationRejected(
                "STRUCTURAL_BINDING_NOT_CONFIRMED",
                "structural model did not confirm an admissible child-to-parent binding",
            )
        if not exact_binding:
            raise RelationVerificationRejected(
                "STRUCTURAL_ENDPOINT_DIRECTION_MISMATCH",
                "structural model support was requested for a non child-to-parent hypothesis",
            )
        if not spans:
            raise RelationVerificationRejected(
                "SUPPORTED_EVIDENCE_REQUIRED",
                "supported structural decision requires evidence",
            )
        subject = _occurrence_details(
            opportunity, hypothesis.subject_occurrence_key, content,
        )
        object_ = _occurrence_details(
            opportunity, hypothesis.object_occurrence_key, content,
        )
        if not (
            _spans_cover(spans, subject["span"])
            and _spans_cover(spans, object_["span"])
            and _spans_stay_within_required_atomic_scope(spans, opportunity)
        ):
            raise RelationVerificationRejected(
                "SUPPORTED_EVIDENCE_MISSING_ENDPOINT",
                "structural evidence does not cover both exact endpoints",
            )
        normalized = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": SUPPORTED,
            "reason_code": reason_code,
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [
                {"start": span.start, "end": span.end} for span in spans
            ],
        }
    else:
        normalized = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": decision,
            "reason_code": reason_code,
            "forward_entailment": "NOT_ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": False,
            "object_role_supported": False,
            "supported_evidence_spans": [],
        }
    return _parse_verifier_response(
        hypothesis,
        opportunity,
        normalized,
        content=content,
        logical_schema=logical_schema,
        blind_role_audit=blind_role_audit,
        direction_normalization=direction_normalization,
        attempt_no=attempt_no,
        verifier_model_revision=verifier_model_revision,
    )


def verifier_response_schema(hypothesis_id: str) -> dict[str, Any]:
    return {
        "$id": GROUNDED_RELATION_VERIFIER_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [SUPPORTED, REJECTED, UNCERTAIN],
            },
            "reason_code": {
                "type": "string", "minLength": 1, "maxLength": 160,
            },
            "forward_entailment": {
                "type": "string",
                "enum": ["ENTAILED", "NOT_ENTAILED", "UNCERTAIN"],
            },
            "reverse_entailment": {
                "type": "string",
                "enum": ["ENTAILED", "NOT_ENTAILED", "UNCERTAIN"],
            },
            "subject_role_supported": {
                "type": "boolean",
                "description": (
                    "Whether the subject fulfills its required role in the one "
                    "orientation marked ENTAILED. Failure of the swapped "
                    "NOT_ENTAILED orientation does not make this false."
                ),
            },
            "object_role_supported": {
                "type": "boolean",
                "description": (
                    "Whether the object fulfills its required role in the one "
                    "orientation marked ENTAILED. Failure of the swapped "
                    "NOT_ENTAILED orientation does not make this false."
                ),
            },
            "supported_evidence_spans": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                    "required": ["start", "end"],
                },
            },
        },
        "required": [
            "verdict", "reason_code", "forward_entailment",
            "reverse_entailment", "subject_role_supported",
            "object_role_supported", "supported_evidence_spans",
        ],
    }


def batch_verifier_prompt(
    claims: Sequence[
        tuple[RelationHypothesis, PairOpportunity]
        | tuple[RelationHypothesis, PairOpportunity, str]
    ],
    *,
    content: str | None = None,
    logical_schema: Mapping[str, Any],
) -> str:
    """Render claims from one or more contexts for batched adjudication.

    Each claim retains the exact single-claim verifier payload and coordinates;
    only the long common instruction is deduplicated.  The caller validates an
    exact one-to-one hypothesis-id set and falls back to single calls on any
    protocol gap, so batching changes transport cost rather than trust policy.
    """

    if not claims:
        raise ValueError("relation verifier batch requires claims")
    rendered_claims: list[dict[str, Any]] = []
    claim_instructions: list[str] = []
    context_catalog: dict[str, dict[str, Any]] = {}
    context_refs: dict[str, str] = {}
    relation_policy_catalog: dict[str, dict[str, Any]] = {}
    relation_policy_refs: dict[str, str] = {}

    def intern_catalog_value(
        value: Mapping[str, Any],
        *,
        catalog: dict[str, dict[str, Any]],
        refs: dict[str, str],
        prefix: str,
    ) -> str:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        reference = refs.get(canonical)
        if reference is None:
            reference = f"{prefix}-{len(catalog) + 1}"
            refs[canonical] = reference
            catalog[reference] = dict(value)
        return reference

    for claim in claims:
        if len(claim) == 3:
            hypothesis, opportunity, claim_content = claim
        else:
            hypothesis, opportunity = claim
            if content is None:
                raise ValueError(
                    "relation verifier batch claim requires source content"
                )
            claim_content = content
        payload = json.loads(verifier_prompt(
            hypothesis,
            opportunity,
            content=str(claim_content),
            logical_schema=logical_schema,
            blind_role_audit={},
        ))
        instruction = str(payload.pop("instruction", ""))
        payload.pop("contract_version", None)
        payload.pop("output_contract", None)
        claim_context = payload.pop("context")
        payload["context_ref"] = intern_catalog_value(
            claim_context,
            catalog=context_catalog,
            refs=context_refs,
            prefix="CTX",
        )
        relation_policy = payload.pop("relation_policy")
        payload["relation_policy_ref"] = intern_catalog_value(
            relation_policy,
            catalog=relation_policy_catalog,
            refs=relation_policy_refs,
            prefix="RP",
        )
        claim_instructions.append(instruction)
        rendered_claims.append(payload)
    if len(set(claim_instructions)) == 1:
        common_instruction = claim_instructions[0]
    else:
        # Structural notes can make an otherwise common instruction
        # claim-specific. Keep every full contract on its own claim; using a
        # structural first item's instruction as a batch-wide prefix would
        # incorrectly broaden that evidence mode to ordinary claims.
        common_instruction = (
            "Apply each claim's claim_specific_instruction only to that claim."
        )
        for payload, instruction in zip(
            rendered_claims, claim_instructions, strict=True,
        ):
            payload["claim_specific_instruction"] = instruction
    return json.dumps(
        {
            "instruction": (
                common_instruction
                + " Evaluate every claim independently. A decision for one claim "
                "must not influence another. Return exactly one decision for each "
                "supplied hypothesis_id, in any order, and no extra ids. Resolve "
                "each claim's context_ref and relation_policy_ref only from the two "
                "batch catalogs; never apply one claim's referenced values to another. "
                "The response root must be the single key decisions; never use "
                "verifications, items, results, entailment, or evidence_spans as aliases. "
                "Every decisions member must contain all eight exact keys listed in "
                "output_contract.decision_keys. Each reason_code must be one concise "
                "diagnostic label of at most 160 characters; never include analysis or "
                "an explanatory paragraph."
            ),
            "contract_version": (
                GROUNDED_RELATION_BATCH_VERIFIER_PROMPT_VERSION
            ),
            "output_contract": {
                "root_key": "decisions",
                "forbidden_root_keys": [
                    "verifications", "verdicts", "items", "results",
                ],
                "decision_keys": [
                    "hypothesis_id",
                    "verdict",
                    "reason_code",
                    "forward_entailment",
                    "reverse_entailment",
                    "subject_role_supported",
                    "object_role_supported",
                    "supported_evidence_spans",
                ],
                "forbidden_decision_aliases": [
                    "entailment", "evidence_spans",
                ],
            },
            "context_catalog": context_catalog,
            "relation_policy_catalog": relation_policy_catalog,
            "claims": rendered_claims,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def batch_verifier_response_schema(
    hypothesis_ids: Sequence[str],
) -> dict[str, Any]:
    ids = sorted({str(value) for value in hypothesis_ids if str(value)})
    if not ids:
        raise ValueError("relation verifier batch requires hypothesis ids")
    decision = verifier_response_schema(ids[0])
    properties = dict(decision["properties"])
    properties["hypothesis_id"] = {"type": "string", "enum": ids}
    return {
        "$id": GROUNDED_RELATION_BATCH_VERIFIER_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": len(ids),
                "maxItems": len(ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": ["hypothesis_id", *decision["required"]],
                },
            },
        },
        "required": ["decisions"],
    }


def parse_batch_verifier_response(
    raw: Any,
    *,
    hypothesis_ids: Sequence[str],
    allow_partial: bool = False,
) -> dict[str, dict[str, Any]]:
    """Parse a duplicate-free batch decision set.

    The default remains exact.  ``allow_partial`` is used only after that
    strict parse fails: independently valid known decisions are retained and
    every omitted, duplicate or malformed member falls back to its original
    single-claim verifier call.  Partial transport can therefore save work but
    can never publish an absent or ambiguous result.
    """

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError(
                "relation batch verifier returned invalid JSON"
            ) from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError(
            "relation batch verifier response must be an object"
        )
    decisions = raw.get("decisions")
    expected = {str(value) for value in hypothesis_ids if str(value)}
    if (
        allow_partial
        and not isinstance(decisions, list)
        and isinstance(
            raw.get("verifications") or raw.get("verdicts"), list,
        )
    ):
        # Some schema-capable local models use this root alias even after a
        # structured-output retry.  The alias itself carries no semantics, so
        # partial mode may inspect its members; exact mode still records the
        # protocol breach.  Positive members remain subject to all seven
        # ordinary verifier fields below.
        decisions = raw.get("verifications") or raw.get("verdicts")
    if isinstance(decisions, Mapping):
        # Some local schema-capable models render the requested decision array
        # as an object keyed by the same immutable hypothesis IDs.  This is an
        # unambiguous transport shape: normalize it only when every member is
        # an object and the embedded ID is absent or exactly matches its key.
        normalized_decisions: list[dict[str, Any]] = []
        malformed_keyed_decision = False
        for key, item in decisions.items():
            hypothesis_id = str(key)
            if not isinstance(item, Mapping):
                malformed_keyed_decision = True
                if allow_partial:
                    continue
                break
            embedded_id = str(item.get("hypothesis_id") or "")
            if embedded_id and embedded_id != hypothesis_id:
                malformed_keyed_decision = True
                if allow_partial:
                    continue
                break
            normalized_decisions.append({
                **dict(item),
                "hypothesis_id": hypothesis_id,
            })
        if malformed_keyed_decision and not allow_partial:
            raise RelationClassificationError(
                "relation batch verifier keyed decision mismatched"
            )
        decisions = normalized_decisions
    if not isinstance(decisions, list):
        raise RelationClassificationError(
            "relation batch verifier requires decisions"
        )
    parsed: dict[str, dict[str, Any]] = {}
    invalid_ids: set[str] = set()
    for item in decisions:
        if not isinstance(item, Mapping):
            if allow_partial:
                continue
            raise RelationClassificationError(
                "relation batch verifier decision must be an object"
            )
        hypothesis_id = str(item.get("hypothesis_id") or "")
        if hypothesis_id not in expected:
            if allow_partial:
                continue
            raise RelationClassificationError(
                "relation batch verifier returned an unknown hypothesis"
            )
        if hypothesis_id in parsed:
            if allow_partial:
                parsed.pop(hypothesis_id, None)
                invalid_ids.add(hypothesis_id)
                continue
            raise RelationClassificationError(
                "relation batch verifier returned a duplicate hypothesis"
            )
        if hypothesis_id in invalid_ids:
            continue
        value = dict(item)
        value.pop("hypothesis_id", None)
        if allow_partial:
            if (
                "supported_evidence_spans" not in value
                and "evidence_spans" in value
            ):
                value["supported_evidence_spans"] = value.get(
                    "evidence_spans"
                )
            if (
                "forward_entailment" not in value
                and "entailment" in value
            ):
                value["forward_entailment"] = value.get("entailment")
            verdict = str(value.get("verdict") or "")
            if verdict in {REJECTED, UNCERTAIN}:
                # Filling absent audits is safe only for non-publishable
                # outcomes.  A simplified SUPPORTED member is intentionally
                # left incomplete and must receive its own strict verifier
                # call before it can ever publish.
                value.setdefault(
                    "reason_code",
                    f"MODEL_BATCH_{verdict}_LEGACY_FIELDS",
                )
                value.setdefault("forward_entailment", "UNCERTAIN")
                value.setdefault("reverse_entailment", "UNCERTAIN")
                value.setdefault("subject_role_supported", False)
                value.setdefault("object_role_supported", False)
                value["supported_evidence_spans"] = []
        # Validate the seven ordinary fields without adjudicating them here;
        # verify_hypothesis remains the sole evidence/direction gate.
        missing = set(verifier_response_schema(hypothesis_id)["required"]) - set(value)
        if missing:
            if allow_partial:
                invalid_ids.add(hypothesis_id)
                continue
            raise RelationClassificationError(
                "relation batch verifier decision is incomplete"
            )
        parsed[hypothesis_id] = value
    if not allow_partial and set(parsed) != expected:
        raise RelationClassificationError(
            "relation batch verifier omitted a hypothesis"
        )
    return parsed


def batch_verifier_decision_is_source_grounded(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    decision: Mapping[str, Any],
    *,
    content: str,
    logical_schema: Mapping[str, Any] | None = None,
) -> bool:
    """Preflight a positive batch member before it can replace a single call.

    Batch models can occasionally attach one claim's otherwise valid decision
    to another hypothesis ID.  Negative members cannot publish and retain
    their ordinary audit path.  A positive member is reusable only when its
    evidence parses inside this immutable opportunity, covers both exact
    endpoint occurrences, stays inside the required atomic scope, and, when
    the logical Schema is supplied, passes the same ordered-role/direction
    contract as a single response.  A failed preflight is transport
    contamination or an ambiguous batch result, so the caller falls back to
    the original single-claim verifier instead of recording a semantic veto.
    """

    verdict = str(decision.get("verdict") or "").strip().upper()
    verdict = {
        "ENTAILED": SUPPORTED,
        "DIRECTLY_ENTAILED": SUPPORTED,
    }.get(verdict, verdict)
    if verdict != SUPPORTED:
        return True
    try:
        spans = _parse_spans(
            decision.get("supported_evidence_spans") or [],
            opportunity=opportunity,
            content=content,
        )
        subject = _occurrence_details(
            opportunity, hypothesis.subject_occurrence_key, content,
        )
        object_ = _occurrence_details(
            opportunity, hypothesis.object_occurrence_key, content,
        )
    except (KeyError, TypeError, ValueError, RelationClassificationError):
        return False
    grounded = bool(
        spans
        and _spans_cover(spans, subject["span"])
        and _spans_cover(spans, object_["span"])
        and _spans_stay_within_required_atomic_scope(spans, opportunity)
    )
    if not grounded or logical_schema is None:
        return grounded
    try:
        attempt = _parse_verifier_response(
            hypothesis,
            opportunity,
            decision,
            content=content,
            logical_schema=logical_schema,
            blind_role_audit={
                "subject_role": "batch preflight",
                "object_role": "batch preflight",
                "subject_alignment": "BATCH_PREFLIGHT",
                "object_alignment": "BATCH_PREFLIGHT",
                "forward_alignment": "BATCH_PREFLIGHT",
                "reverse_alignment": "BATCH_PREFLIGHT",
            },
            direction_normalization="FORWARD",
            attempt_no=0,
            verifier_model_revision="batch-preflight",
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        RelationClassificationError,
        RelationVerificationRejected,
    ):
        return False
    return bool(
        attempt.verdict == SUPPORTED
        and attempt.normalized_subject_occurrence_key
        == hypothesis.subject_occurrence_key
        and attempt.normalized_object_occurrence_key
        == hypothesis.object_occurrence_key
    )


def _normalize_entailment_audit(value: Any) -> str:
    """Normalize a narrow model-protocol variation without changing meaning.

    Some JSON-schema-capable local models still emit booleans or a one-edit
    misspelling for a requested entailment enum. Treating an unambiguous
    single insertion/deletion/substitution as the intended closed enum is a
    protocol repair, not a semantic inference. More distant or ambiguous
    representations remain invalid and fail closed.
    """

    if value is True:
        return "ENTAILED"
    if value is False:
        return "NOT_ENTAILED"
    normalized = str(value or "").strip().upper()
    if normalized in {"TRUE", "YES"}:
        return "ENTAILED"
    if normalized in {"FALSE", "NO"}:
        return "NOT_ENTAILED"
    if normalized in {"ENTAILED", "NOT_ENTAILED", "UNCERTAIN"}:
        return normalized
    normalized = re.sub(r"[\s-]+", "_", normalized)
    if normalized in {"ENTAILED", "NOT_ENTAILED", "UNCERTAIN"}:
        return normalized

    def one_edit_apart(left: str, right: str) -> bool:
        if abs(len(left) - len(right)) > 1:
            return False
        if len(left) == len(right):
            return sum(a != b for a, b in zip(left, right)) == 1
        if len(left) > len(right):
            left, right = right, left
        # ``right`` is exactly one character longer than ``left``.
        index = 0
        while index < len(left) and left[index] == right[index]:
            index += 1
        return left[index:] == right[index + 1:]

    repairs = [
        candidate
        for candidate in ("ENTAILED", "NOT_ENTAILED", "UNCERTAIN")
        if one_edit_apart(normalized, candidate)
    ]
    if len(repairs) == 1:
        return repairs[0]
    return ""


def _normalize_boolean_audit(value: Any) -> bool | None:
    """Normalize narrow JSON boolean spellings without inferring a role."""

    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    normalized = str(value or "").strip().upper()
    if normalized in {"TRUE", "YES"}:
        return True
    if normalized in {"FALSE", "NO"}:
        return False
    return None


def _attempt(
    hypothesis: RelationHypothesis,
    *,
    attempt_no: int,
    verdict: str,
    reason_code: str,
    supported_spans: tuple[EvidenceSpan, ...],
    independent_call: bool,
    verifier_model_revision: str,
    forward_entailment: str = "",
    reverse_entailment: str = "",
    subject_role_supported: bool = False,
    object_role_supported: bool = False,
    subject_context_role: str = "",
    object_context_role: str = "",
    subject_role_alignment: str = "",
    object_role_alignment: str = "",
    forward_role_alignment: str = "",
    reverse_role_alignment: str = "",
    direction_normalization: str = "FORWARD",
    normalized_subject_occurrence_key: str = "",
    normalized_object_occurrence_key: str = "",
    error_message: str = "",
) -> VerificationAttempt:
    attempt_id = "relverify_" + _digest((
        hypothesis.hypothesis_id,
        attempt_no,
        verifier_model_revision,
        GROUNDED_RELATION_VERIFIER_PROMPT_VERSION,
    ))
    return VerificationAttempt(
        attempt_id=attempt_id,
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=attempt_no,
        verdict=verdict,
        reason_code=reason_code,
        supported_evidence_spans=supported_spans,
        independent_call=independent_call,
        verifier_prompt_version=GROUNDED_RELATION_VERIFIER_PROMPT_VERSION,
        verifier_schema_version=GROUNDED_RELATION_VERIFIER_SCHEMA_VERSION,
        verifier_parser_version=GROUNDED_RELATION_VERIFIER_PARSER_VERSION,
        verifier_model_revision=verifier_model_revision,
        forward_entailment=forward_entailment,
        reverse_entailment=reverse_entailment,
        subject_role_supported=subject_role_supported,
        object_role_supported=object_role_supported,
        subject_context_role=subject_context_role,
        object_context_role=object_context_role,
        subject_role_alignment=subject_role_alignment,
        object_role_alignment=object_role_alignment,
        forward_role_alignment=forward_role_alignment,
        reverse_role_alignment=reverse_role_alignment,
        direction_normalization=direction_normalization,
        normalized_subject_occurrence_key=(
            normalized_subject_occurrence_key
            or hypothesis.subject_occurrence_key
        ),
        normalized_object_occurrence_key=(
            normalized_object_occurrence_key
            or hypothesis.object_occurrence_key
        ),
        error_message=error_message,
    )


def _parse_verifier_response(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    raw: Any,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    blind_role_audit: Mapping[str, str],
    direction_normalization: str,
    attempt_no: int,
    verifier_model_revision: str,
) -> VerificationAttempt:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RelationClassificationError("verifier returned invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise RelationClassificationError("verifier response must be an object")
    response_hypothesis_id = raw.get("hypothesis_id")
    if (
        response_hypothesis_id is not None
        and str(response_hypothesis_id) != hypothesis.hypothesis_id
    ):
        raise RelationClassificationError("verifier hypothesis id mismatched")
    verdict = str(raw.get("verdict") or "").strip().upper()
    verdict = {
        "ENTAILED": SUPPORTED,
        "DIRECTLY_ENTAILED": SUPPORTED,
        "UNSUPPORTED": REJECTED,
        "NOT_SUPPORTED": REJECTED,
        "NOT_ENTAILED": REJECTED,
        "UNKNOWN": UNCERTAIN,
    }.get(verdict, verdict)
    if verdict not in {SUPPORTED, REJECTED, UNCERTAIN}:
        raise RelationClassificationError(
            f"verifier verdict is invalid: {verdict or '<empty>'}"
        )
    reason_code = str(
        raw.get("reason_code")
        or raw.get("reason")
        or raw.get("rationale")
        or ""
    ).strip()
    if not reason_code:
        # The reason is an audit field, not evidence.  Schema-capable local
        # models sometimes omit it while returning a complete verdict and all
        # evidence/role fields.  Preserve fail-closed semantic gates but keep
        # the valid verdict auditable instead of converting it into a
        # technical failure that secondary review cannot distinguish.
        reason_code = f"MODEL_{verdict}_REASON_OMITTED"
    swap_verified_orientation = False
    spans = _parse_spans(
        raw.get("supported_evidence_spans") or [],
        opportunity=opportunity,
        content=content,
    )
    subject_role_supported = _normalize_boolean_audit(
        raw.get("subject_role_supported")
    )
    object_role_supported = _normalize_boolean_audit(
        raw.get("object_role_supported")
    )
    if verdict == SUPPORTED:
        if not spans:
            raise RelationVerificationRejected(
                "SUPPORTED_EVIDENCE_REQUIRED",
                "SUPPORTED verdict requires evidence",
            )
        subject = _occurrence_details(
            opportunity, hypothesis.subject_occurrence_key, content,
        )
        object_ = _occurrence_details(
            opportunity, hypothesis.object_occurrence_key, content,
        )
        if not (
            _spans_cover(spans, subject["span"])
            and _spans_cover(spans, object_["span"])
        ):
            raise RelationVerificationRejected(
                "SUPPORTED_EVIDENCE_MISSING_ENDPOINT",
                "verifier evidence does not cover exact occurrences",
            )
        if not _spans_stay_within_required_atomic_scope(spans, opportunity):
            raise RelationVerificationRejected(
                "SUPPORTED_EVIDENCE_OUTSIDE_ATOMIC_SCOPE",
                "verifier evidence crosses the required atomic scope",
            )
        forward_entailment = _normalize_entailment_audit(
            raw.get("forward_entailment")
        )
        reverse_entailment = _normalize_entailment_audit(
            raw.get("reverse_entailment")
        )
        if forward_entailment not in {
            "ENTAILED", "NOT_ENTAILED", "UNCERTAIN",
        } or reverse_entailment not in {
            "ENTAILED", "NOT_ENTAILED", "UNCERTAIN",
        }:
            raise RelationClassificationError(
                "verifier ordered entailment audit is required"
            )
        if (
            subject_role_supported is not True
            or object_role_supported is not True
        ):
            raise RelationVerificationRejected(
                "ORDERED_ROLE_NOT_ENTAILED",
                "the entailed orientation's predicate roles are not explicit",
                forward_entailment=forward_entailment,
                reverse_entailment=reverse_entailment,
                subject_role_supported=(subject_role_supported is True),
                object_role_supported=(object_role_supported is True),
            )
        spec = normalize_predicate_schema(logical_schema).get(
            hypothesis.predicate
        )
        if spec is None:
            raise RelationVerificationRejected(
                "PREDICATE_METADATA_MISSING",
                "verifier predicate metadata is unavailable",
            )
        if spec.symmetric:
            orientation_supported = (
                forward_entailment == "ENTAILED"
                and reverse_entailment == "ENTAILED"
            )
            if orientation_supported:
                direction_normalization = "SYMMETRIC"
        else:
            forward_supported = (
                forward_entailment == "ENTAILED"
                and reverse_entailment == "NOT_ENTAILED"
            )
            reverse_supported = (
                reverse_entailment == "ENTAILED"
                and forward_entailment == "NOT_ENTAILED"
            )
            orientation_supported = forward_supported or reverse_supported
            if forward_supported:
                if direction_normalization != "REVERSED":
                    direction_normalization = "FORWARD"
            elif reverse_supported:
                swap_verified_orientation = True
                direction_normalization = (
                    "FORWARD"
                    if direction_normalization == "REVERSED"
                    else "REVERSED"
                )
        if not orientation_supported:
            raise RelationVerificationRejected(
                "DIRECTION_NOT_EXCLUSIVE",
                "ordered entailment does not identify one legal orientation",
                forward_entailment=forward_entailment,
                reverse_entailment=reverse_entailment,
                subject_role_supported=True,
                object_role_supported=True,
            )
    else:
        forward_entailment = _normalize_entailment_audit(
            raw.get("forward_entailment")
        )
        reverse_entailment = _normalize_entailment_audit(
            raw.get("reverse_entailment")
        )
    normalized_subject_occurrence_key = hypothesis.subject_occurrence_key
    normalized_object_occurrence_key = hypothesis.object_occurrence_key
    if verdict == SUPPORTED and swap_verified_orientation:
        normalized_subject_occurrence_key = hypothesis.object_occurrence_key
        normalized_object_occurrence_key = hypothesis.subject_occurrence_key
    return _attempt(
        hypothesis,
        attempt_no=attempt_no,
        verdict=verdict,
        reason_code=reason_code,
        supported_spans=spans,
        independent_call=True,
        verifier_model_revision=verifier_model_revision,
        forward_entailment=forward_entailment,
        reverse_entailment=reverse_entailment,
        subject_role_supported=(subject_role_supported is True),
        object_role_supported=(object_role_supported is True),
        subject_context_role=str(blind_role_audit["subject_role"]),
        object_context_role=str(blind_role_audit["object_role"]),
        subject_role_alignment=str(blind_role_audit["subject_alignment"]),
        object_role_alignment=str(blind_role_audit["object_alignment"]),
        forward_role_alignment=str(blind_role_audit["forward_alignment"]),
        reverse_role_alignment=str(blind_role_audit["reverse_alignment"]),
        direction_normalization=direction_normalization,
        normalized_subject_occurrence_key=normalized_subject_occurrence_key,
        normalized_object_occurrence_key=normalized_object_occurrence_key,
    )


def _apply_independent_forward_orientation_audit(
    raw: Any,
    *,
    hypothesis: RelationHypothesis,
    logical_schema: Mapping[str, Any],
    independent_orientation_audit: str,
) -> Any:
    """Combine independent orientation with a fresh positive final verdict.

    Local models frequently interpret the reverse audit as an inverse-verb
    paraphrase (for example, a whole containing a part) instead of applying
    the *same* directed Schema predicate to swapped endpoint roles.  A
    secondary ROLE_RETRY has already hidden the first verdict and selected an
    exact endpoint orientation from source text.  When that independent audit
    selects the current candidate as FORWARD, a fresh final verifier may use
    it only to resolve an ENTAILED/UNCERTAIN reverse-field ambiguity.  It may
    not supply the final verdict, forward entailment, endpoint roles, or
    evidence, and it never applies to a symmetric predicate.
    """

    if str(independent_orientation_audit or "").strip().upper() != "FORWARD":
        return raw
    if not isinstance(raw, Mapping):
        return raw
    spec = normalize_predicate_schema(logical_schema).get(
        hypothesis.predicate
    )
    if spec is None or spec.symmetric:
        return raw
    repaired = dict(raw)
    if (
        str(repaired.get("verdict") or "").strip().upper() == SUPPORTED
        and _normalize_entailment_audit(
            repaired.get("forward_entailment")
        ) == "ENTAILED"
        and _normalize_boolean_audit(
            repaired.get("subject_role_supported")
        ) is True
        and _normalize_boolean_audit(
            repaired.get("object_role_supported")
        ) is True
        and _normalize_entailment_audit(
            repaired.get("reverse_entailment")
        ) in {"ENTAILED", "UNCERTAIN"}
    ):
        repaired["reverse_entailment"] = "NOT_ENTAILED"
    return repaired


def _parser_hierarchy_open_carrier_intent(
    hypothesis: RelationHypothesis,
    spec: PredicateSpec | None,
    *,
    structure_kind: str,
) -> bool:
    """Recognize an open carrier attached to a frozen hierarchy frame.

    The logical Schema's fallback carrier intentionally has no domain-specific
    ``structural_evidence_kinds``.  Its source frame can nevertheless be a
    parser-owned list/section child binding.  Recognizing that provenance as
    *intent* routes malformed variants through the deterministic checker and
    prevents an LLM from reinterpreting ordinary adjacency as structure.
    """

    metadata = hypothesis.predicate_metadata or {}
    policy = spec.open_relation_policy if spec is not None else {}
    return bool(
        spec is not None
        and bool(policy.get("enabled"))
        and bool(policy.get("fallback_only"))
        and str(metadata.get("candidate_source") or "").upper()
        in {
            "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            "ATOMIC_FRAME_SCHEMA_PROJECTION",
        }
        and str(metadata.get("source_binding_source") or "").upper()
        == "OPEN_EXPRESSION_CARRIER_FALLBACK"
        and str(metadata.get("source_structure_kind") or "")
        == str(structure_kind)
    )


def _parser_hierarchy_open_carrier_authorized(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    spec: PredicateSpec | None,
    *,
    structure_kind: str,
) -> bool:
    """Authorize only the exact asserted child-to-parent carrier contract."""

    if not _parser_hierarchy_open_carrier_intent(
        hypothesis, spec, structure_kind=structure_kind,
    ):
        return False
    metadata = hypothesis.predicate_metadata or {}
    if (
        str(metadata.get("source_frame_assertion_status") or "").upper()
        != "ASSERTED"
        or bool(metadata.get("source_frame_explicit_negation"))
        or metadata.get("source_subject_argument_index") != 0
        or metadata.get("source_object_argument_index") != 1
        or not _displayable_open_relation_expression(
            metadata.get("source_frame_relation")
        )
    ):
        return False
    child, parent = _structural_parent_child_keys(opportunity)
    if (
        hypothesis.subject_occurrence_key != child
        or hypothesis.object_occurrence_key != parent
    ):
        return False

    canonical_by_occurrence = {
        opportunity.left_occurrence_key: opportunity.left_canonical_id,
        opportunity.right_occurrence_key: opportunity.right_canonical_id,
    }
    source_subject_canonical = str(
        metadata.get("source_subject_canonical_id") or ""
    )
    source_object_canonical = str(
        metadata.get("source_object_canonical_id") or ""
    )
    return bool(
        (not source_subject_canonical or source_subject_canonical
         == canonical_by_occurrence.get(child, ""))
        and (not source_object_canonical or source_object_canonical
             == canonical_by_occurrence.get(parent, ""))
    )


def _parser_parenthetical_open_carrier_intent(
    hypothesis: RelationHypothesis,
    spec: PredicateSpec | None,
) -> bool:
    """Recognize the fallback carrier for a parser-owned parenthetical list."""

    metadata = hypothesis.predicate_metadata or {}
    policy = spec.open_relation_policy if spec is not None else {}
    return bool(
        spec is not None
        and bool(policy.get("enabled"))
        and bool(policy.get("fallback_only"))
        and str(metadata.get("candidate_source") or "").upper()
        in {
            "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            "ATOMIC_FRAME_SCHEMA_PROJECTION",
        }
        and str(metadata.get("source_binding_source") or "").upper()
        == "OPEN_EXPRESSION_CARRIER_FALLBACK"
        and str(metadata.get("source_structure_kind") or "")
        == PARENTHETICAL_ENUMERATION_STRUCTURE_KIND
    )


def _parser_parenthetical_open_carrier_authorized(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    spec: PredicateSpec | None,
    *,
    content: str,
) -> bool:
    """Reconstruct one exact ``Head (A, B and C)`` detail contract."""

    if not _parser_parenthetical_open_carrier_intent(hypothesis, spec):
        return False
    metadata = hypothesis.predicate_metadata or {}
    contract = metadata.get("source_structure_contract")
    if not isinstance(contract, Mapping):
        return False
    if (
        str(metadata.get("source_frame_assertion_status") or "").upper()
        != "ASSERTED"
        or bool(metadata.get("source_frame_explicit_negation"))
        or metadata.get("source_subject_argument_index") != 0
        or metadata.get("source_object_argument_index") != 1
        or source_relation_display_expression(
            metadata.get("source_frame_relation")
        ) != PARENTHETICAL_DETAIL_RELATION
        or str(contract.get("kind") or "")
        != PARENTHETICAL_ENUMERATION_STRUCTURE_KIND
    ):
        return False
    head_key = str(contract.get("head_occurrence_key") or "")
    member_keys = [
        str(value or "")
        for value in contract.get("member_occurrence_keys") or ()
    ]
    member_spans = list(contract.get("member_spans") or ())
    member_surfaces = [
        str(value or "")
        for value in contract.get("member_surfaces") or ()
    ]
    if (
        hypothesis.object_occurrence_key != head_key
        or hypothesis.subject_occurrence_key not in member_keys
        or not 2 <= len(member_keys) <= 16
        or len(set(member_keys)) != len(member_keys)
        or head_key in member_keys
        or len(member_spans) != len(member_keys)
        or len(member_surfaces) != len(member_keys)
    ):
        return False

    def exact_span(value: Any) -> tuple[int, int] | None:
        if not (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in value
            )
        ):
            return None
        start, end = int(value[0]), int(value[1])
        return (start, end) if 0 <= start < end <= len(content) else None

    head_span = exact_span(contract.get("head_span"))
    bracket_span = exact_span(contract.get("bracket_span"))
    parsed_member_spans = [exact_span(value) for value in member_spans]
    endpoint_spans = {
        opportunity.left_occurrence_key: opportunity.left_span,
        opportunity.right_occurrence_key: opportunity.right_span,
    }
    if (
        head_span is None
        or bracket_span is None
        or any(value is None for value in parsed_member_spans)
        or endpoint_spans.get(head_key) != head_span
        or endpoint_spans.get(hypothesis.subject_occurrence_key)
        != parsed_member_spans[
            member_keys.index(hypothesis.subject_occurrence_key)
        ]
    ):
        return False
    opening, after_closing = bracket_span
    closing = after_closing - 1
    if (
        content[opening:opening + 1] not in {"(", "（"}
        or content[closing:after_closing]
        != {"(": ")", "（": "）"}[content[opening]]
        or head_span[1] > opening
        or content[head_span[1]:opening].strip()
    ):
        return False
    ordered_members = sorted(
        zip(parsed_member_spans, member_surfaces),
        key=lambda value: value[0] or (-1, -1),
    )
    if any(
        span is None
        or not (opening < span[0] < span[1] <= closing)
        or content[span[0]:span[1]] != surface
        for span, surface in ordered_members
    ):
        return False
    if (
        content[opening + 1:ordered_members[0][0][0]].strip()
        or content[ordered_members[-1][0][1]:closing].strip()
    ):
        return False
    explicit_connector = False
    for (left_span, _left_surface), (right_span, _right_surface) in zip(
        ordered_members, ordered_members[1:],
    ):
        if left_span is None or right_span is None or left_span[1] > right_span[0]:
            return False
        matched = _COORDINATED_RECOVERY_CONNECTOR.fullmatch(
            content[left_span[1]:right_span[0]]
        )
        if matched is None:
            return False
        explicit_connector = explicit_connector or bool(
            matched.group("word")
            or matched.group("separator") in {"、", ";", "；"}
        )
    if not explicit_connector:
        return False

    canonical_by_occurrence = {
        opportunity.left_occurrence_key: opportunity.left_canonical_id,
        opportunity.right_occurrence_key: opportunity.right_canonical_id,
    }
    source_subject_canonical = str(
        metadata.get("source_subject_canonical_id") or ""
    )
    source_object_canonical = str(
        metadata.get("source_object_canonical_id") or ""
    )
    return bool(
        (not source_subject_canonical or source_subject_canonical
         == canonical_by_occurrence.get(hypothesis.subject_occurrence_key, ""))
        and (not source_object_canonical or source_object_canonical
             == canonical_by_occurrence.get(head_key, ""))
    )


def parser_structure_claim_intent(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    logical_schema: Mapping[str, Any],
) -> bool:
    """Return whether a claim asks to use an immutable parser contract.

    This is intentionally broader than authorization.  Once a candidate marks
    itself as parser-derived it must fail closed through the deterministic
    contract checker; malformed structure metadata must never fall through to
    a language model that could reinterpret ordinary adjacency as a fact.
    Exact classifier candidates over a parser-confirmed hierarchy are included
    when their Schema predicate explicitly authorizes that structure kind.
    """

    metadata = hypothesis.predicate_metadata or {}
    if (
        bool(metadata.get("structural_event_projection"))
        or str(metadata.get("source_binding_source") or "").upper().startswith(
            "DETERMINISTIC_SOURCE_STRUCTURE"
        )
    ):
        return True
    spec = normalize_predicate_schema(logical_schema).get(
        hypothesis.predicate
    )
    if _parser_parenthetical_open_carrier_intent(hypothesis, spec):
        return True
    structural_kind = (
        "list_parent_child"
        if _has_hierarchical_list_parent_child_evidence(opportunity)
        else "section_label_parent_child"
        if _has_hierarchical_section_label_parent_child_evidence(opportunity)
        else ""
    )
    if not structural_kind:
        return False
    return bool(
        spec is not None
        and (
            structural_kind in spec.structural_evidence_kinds
            or _parser_hierarchy_open_carrier_intent(
                hypothesis, spec, structure_kind=structural_kind,
            )
        )
    )


def deterministic_parser_structure_terminal_attempt(
    hypothesis: RelationHypothesis,
    validation: HypothesisValidation,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    attempt_no: int = 1,
) -> VerificationAttempt:
    """Independently revalidate and terminate one parser-proved relation.

    Lists, section labels, table cells and identifier bindings are source
    syntax, not probabilistic semantic inferences.  Once the source mapper has
    frozen exact endpoint roles, spans and structure kind, asking the same LLM
    to restate that layout adds latency and variance but no new evidence.  This
    checker reconstructs the full contract from immutable source bytes and the
    versioned logical Schema.  Only an exact directed contract can publish.
    """

    if validation.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("validation does not belong to hypothesis")
    deterministic_revision = (
        "deterministic:"
        + GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION
    )

    def terminal(
        *,
        verdict: str,
        reason_code: str,
        structure_kind: str = "",
        spans: tuple[EvidenceSpan, ...] = (),
        independent: bool = True,
        subject_role: str = "",
        object_role: str = "",
        error_message: str = "",
    ) -> VerificationAttempt:
        supported = verdict == SUPPORTED
        rejected = verdict == REJECTED
        return VerificationAttempt(
            attempt_id="relverify_" + _digest((
                hypothesis.hypothesis_id,
                int(attempt_no),
                GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION,
                reason_code,
            )),
            hypothesis_id=hypothesis.hypothesis_id,
            attempt_no=int(attempt_no),
            verdict=verdict,
            reason_code=reason_code,
            supported_evidence_spans=spans if supported else (),
            independent_call=bool(independent),
            verifier_prompt_version=(
                GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION
            ),
            verifier_schema_version=(
                GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_SCHEMA_VERSION
            ),
            verifier_parser_version=(
                GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_PARSER_VERSION
            ),
            verifier_model_revision=deterministic_revision,
            forward_entailment=(
                "ENTAILED" if supported else
                "NOT_ENTAILED" if rejected else "UNCERTAIN"
            ),
            reverse_entailment=(
                "NOT_ENTAILED" if supported or rejected else "UNCERTAIN"
            ),
            subject_role_supported=supported,
            object_role_supported=supported,
            subject_context_role=subject_role,
            object_context_role=object_role,
            subject_role_alignment=(
                "PARSER_CONTRACT" if supported else ""
            ),
            object_role_alignment=(
                "PARSER_CONTRACT" if supported else ""
            ),
            forward_role_alignment=(
                "PARSER_CONTRACT" if supported else ""
            ),
            reverse_role_alignment=(
                "PARSER_CONTRACT" if supported else ""
            ),
            direction_normalization="FORWARD",
            normalized_subject_occurrence_key=(
                hypothesis.subject_occurrence_key
            ),
            normalized_object_occurrence_key=(
                hypothesis.object_occurrence_key
            ),
            error_message=str(error_message)[:2000],
            model_request_payload={
                "stage": "deterministic_parser_structure_contract",
                "policy_version": (
                    GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION
                ),
                "structure_kind": str(structure_kind),
                "model_call": False,
            },
            parsed_response={
                "structure_status": (
                    "ALIGNED" if supported else
                    "CONFLICT" if rejected else "UNCERTAIN"
                ),
            },
        )

    if validation.status != PASS:
        return terminal(
            verdict=(
                UNCERTAIN if validation.status == REVIEW_ONLY else REJECTED
            ),
            reason_code=validation.reason_code,
            independent=False,
        )

    schema = normalize_predicate_schema(logical_schema)
    spec = schema.get(hypothesis.predicate)
    metadata = hypothesis.predicate_metadata or {}
    structure_kind = ""
    subject_role = ""
    object_role = ""
    try:
        if str(
            metadata.get("source_binding_source") or ""
        ).upper().startswith("DETERMINISTIC_SOURCE_STRUCTURE"):
            payload = _source_structure_binding_audit_payload(
                hypothesis,
                opportunity,
                content=content,
                logical_schema=logical_schema,
            )
            source = payload.get("source") or {}
            structure_kind = str(source.get("structure_kind") or "")
            parser_roles = tuple(
                str(value) for value in source.get("parser_roles") or ()
            )
            if len(parser_roles) != 2:
                raise ValueError(
                    "source structure must declare exactly two roles"
                )
            subject_role, object_role = parser_roles
        elif _parser_parenthetical_open_carrier_intent(
            hypothesis, spec,
        ):
            structure_kind = PARENTHETICAL_ENUMERATION_STRUCTURE_KIND
            if not _parser_parenthetical_open_carrier_authorized(
                hypothesis,
                opportunity,
                spec,
                content=content,
            ):
                raise ValueError(
                    "parenthetical enumeration contract is not authorized"
                )
            subject_role = "parenthetical_detail"
            object_role = "parenthetical_head"
        else:
            structure_kind = (
                "list_parent_child"
                if _has_hierarchical_list_parent_child_evidence(opportunity)
                else "section_label_parent_child"
                if _has_hierarchical_section_label_parent_child_evidence(
                    opportunity
                )
                else ""
            )
            if not structure_kind or spec is None:
                raise ValueError("parser hierarchy contract is missing")
            child, parent = _structural_parent_child_keys(opportunity)
            open_carrier_authorized = (
                _parser_hierarchy_open_carrier_authorized(
                    hypothesis,
                    opportunity,
                    spec,
                    structure_kind=structure_kind,
                )
            )
            if (
                (
                    structure_kind not in spec.structural_evidence_kinds
                    and not open_carrier_authorized
                )
                or hypothesis.subject_occurrence_key != child
                or hypothesis.object_occurrence_key != parent
            ):
                raise ValueError(
                    "parser hierarchy Schema or direction is not authorized"
                )
            subject_role = "structural_child"
            object_role = "structural_parent"
        if spec is None or spec.symmetric:
            raise ValueError(
                "directed parser structure requires a non-symmetric Schema"
            )
        spans = _parse_spans(
            [asdict(value) for value in hypothesis.evidence_spans],
            opportunity=opportunity,
            content=content,
        )
        if not (
            spans
            and _spans_cover(spans, opportunity.left_span)
            and _spans_cover(spans, opportunity.right_span)
        ):
            raise ValueError(
                "parser structure evidence does not cover exact endpoints"
            )
    except (TypeError, ValueError, RelationClassificationError) as exc:
        return terminal(
            verdict=REJECTED,
            reason_code="PARSER_STRUCTURE_CONTRACT_INVALID",
            structure_kind=structure_kind,
            subject_role=subject_role,
            object_role=object_role,
            error_message=f"{type(exc).__name__}: {exc}",
        )

    return terminal(
        verdict=SUPPORTED,
        reason_code="SUPPORTED_PARSER_STRUCTURE_CONTRACT",
        structure_kind=structure_kind,
        spans=spans,
        subject_role=subject_role,
        object_role=object_role,
    )


def _verify_structural_only(
    hypothesis: RelationHypothesis,
    validation: HypothesisValidation,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    verifier: Any,
    verifier_model_revision: str,
    attempt_no: int,
) -> VerificationAttempt:
    """Terminate a parser-confirmed structure without model restatement."""

    del verifier, verifier_model_revision
    return deterministic_parser_structure_terminal_attempt(
        hypothesis,
        validation,
        opportunity,
        content=content,
        logical_schema=logical_schema,
        attempt_no=attempt_no,
    )


def verify_hypothesis_lean(
    hypothesis: RelationHypothesis,
    validation: HypothesisValidation,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    role_probe: Any = None,
    role_alignment_probe: Any = None,
    verifier: Any,
    verifier_model_revision: str,
    atomic_fact_probe: Any | None = None,
    atomic_frame_alignment_probe: Any | None = None,
    attempt_no: int = 1,
    structural_adjudication_only: bool = False,
    independent_orientation_audit: str = "",
) -> VerificationAttempt:
    """Run the bounded two-stage publication verifier.

    Candidate extraction and deterministic validation are the first stage;
    this independent ordered verifier is the second.  Earlier revisions ran
    three additional temperature-zero calls to the same model for every
    candidate (atomic inventory, endpoint-role paraphrase and role alignment).
    Those calls were neither independent evidence nor reliable vetoes and
    multiplied long-document cost.  The final contract already requires exact
    endpoint-covering evidence, both ordered entailments and both Schema roles,
    so this path keeps those hard gates in one auditable call.

    The auxiliary callback parameters intentionally match ``verify_hypothesis``
    for runtime interchangeability; they are not invoked here.
    """

    if validation.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("validation does not belong to hypothesis")
    if structural_adjudication_only:
        return _verify_structural_only(
            hypothesis,
            validation,
            opportunity,
            content=content,
            logical_schema=logical_schema,
            verifier=verifier,
            verifier_model_revision=verifier_model_revision,
            attempt_no=attempt_no,
        )
    if validation.status != PASS:
        return _attempt(
            hypothesis,
            attempt_no=attempt_no,
            verdict=UNCERTAIN if validation.status == REVIEW_ONLY else REJECTED,
            reason_code=validation.reason_code,
            supported_spans=(),
            independent_call=False,
            verifier_model_revision=verifier_model_revision,
        )

    verifier_only_audit = {
        "subject_role": "adjudicated by ordered final verifier",
        "object_role": "adjudicated by ordered final verifier",
        "subject_alignment": "FINAL_VERIFIER",
        "object_alignment": "FINAL_VERIFIER",
        "forward_alignment": "FINAL_VERIFIER",
        "reverse_alignment": "FINAL_VERIFIER",
    }
    try:
        raw = verifier(
            verifier_prompt(
                hypothesis,
                opportunity,
                content=content,
                logical_schema=logical_schema,
                blind_role_audit={},
            ),
            verifier_response_schema(hypothesis.hypothesis_id),
        )
        parsed_raw: Any = raw
        if isinstance(parsed_raw, str):
            try:
                parsed_raw = json.loads(parsed_raw)
            except json.JSONDecodeError:
                parsed_raw = raw
        if (
            isinstance(parsed_raw, Mapping)
            and str(parsed_raw.get("verdict") or "") == SUPPORTED
        ):
            repaired = dict(parsed_raw)
            if (
                not repaired.get("supported_evidence_spans")
                and isinstance(repaired.get("evidence_spans"), list)
            ):
                repaired["supported_evidence_spans"] = list(
                    repaired["evidence_spans"]
                )
            if not repaired.get("supported_evidence_spans"):
                fallback_spans = tuple(hypothesis.evidence_spans)
                if (
                    fallback_spans
                    and all(
                        opportunity.context_span[0] <= span.start < span.end
                        <= opportunity.context_span[1]
                        and span.end <= len(content)
                        for span in fallback_spans
                    )
                    and _spans_cover(fallback_spans, opportunity.left_span)
                    and _spans_cover(fallback_spans, opportunity.right_span)
                    and _spans_stay_within_required_atomic_scope(
                        fallback_spans, opportunity,
                    )
                ):
                    repaired["supported_evidence_spans"] = [
                        {"start": span.start, "end": span.end}
                        for span in fallback_spans
                    ]
            parsed_raw = repaired
        parsed_raw = _apply_independent_forward_orientation_audit(
            parsed_raw,
            hypothesis=hypothesis,
            logical_schema=logical_schema,
            independent_orientation_audit=(
                independent_orientation_audit
            ),
        )
        attempt = _parse_verifier_response(
            hypothesis,
            opportunity,
            parsed_raw,
            content=content,
            logical_schema=logical_schema,
            blind_role_audit=verifier_only_audit,
            direction_normalization="FORWARD",
            attempt_no=attempt_no,
            verifier_model_revision=verifier_model_revision,
        )
        if (
            attempt.verdict == SUPPORTED
            and (
                attempt.normalized_subject_occurrence_key
                != hypothesis.subject_occurrence_key
                or attempt.normalized_object_occurrence_key
                != hypothesis.object_occurrence_key
            )
        ):
            # Candidate generation must propose the directed fact it wants to
            # publish.  A reverse-only verifier result is a useful rejection
            # signal, not authority to silently rewrite the immutable claim.
            raise RelationVerificationRejected(
                "REVERSE_ONLY_CLAIM_REQUIRES_OWN_CANDIDATE",
                "the endpoint-swapped claim must be proposed and verified separately",
                forward_entailment=attempt.forward_entailment,
                reverse_entailment=attempt.reverse_entailment,
                subject_role_supported=False,
                object_role_supported=False,
            )
        return attempt
    except RelationVerificationRejected as exc:
        return _attempt(
            hypothesis,
            attempt_no=attempt_no,
            verdict=REJECTED,
            reason_code=exc.reason_code,
            supported_spans=(),
            independent_call=True,
            verifier_model_revision=verifier_model_revision,
            forward_entailment=exc.forward_entailment,
            reverse_entailment=exc.reverse_entailment,
            subject_role_supported=exc.subject_role_supported,
            object_role_supported=exc.object_role_supported,
            subject_context_role=verifier_only_audit["subject_role"],
            object_context_role=verifier_only_audit["object_role"],
            subject_role_alignment="FINAL_VERIFIER",
            object_role_alignment="FINAL_VERIFIER",
            forward_role_alignment="FINAL_VERIFIER",
            reverse_role_alignment="FINAL_VERIFIER",
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )
    except Exception as exc:
        return _attempt(
            hypothesis,
            attempt_no=attempt_no,
            verdict=RETRYABLE_FAILED,
            reason_code="VERIFIER_CALL_OR_RESPONSE_FAILED",
            supported_spans=(),
            independent_call=True,
            verifier_model_revision=verifier_model_revision,
            subject_context_role=verifier_only_audit["subject_role"],
            object_context_role=verifier_only_audit["object_role"],
            subject_role_alignment="FINAL_VERIFIER",
            object_role_alignment="FINAL_VERIFIER",
            forward_role_alignment="FINAL_VERIFIER",
            reverse_role_alignment="FINAL_VERIFIER",
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )


def verify_hypothesis(
    hypothesis: RelationHypothesis,
    validation: HypothesisValidation,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    role_probe: Any,
    role_alignment_probe: Any,
    verifier: Any,
    verifier_model_revision: str,
    atomic_fact_probe: Any | None = None,
    atomic_frame_alignment_probe: Any | None = None,
    attempt_no: int = 1,
    structural_adjudication_only: bool = False,
    independent_orientation_audit: str = "",
) -> VerificationAttempt:
    """Run one independent verifier call, or record why it was not callable."""

    if validation.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("validation does not belong to hypothesis")
    if structural_adjudication_only:
        return _verify_structural_only(
            hypothesis,
            validation,
            opportunity,
            content=content,
            logical_schema=logical_schema,
            verifier=verifier,
            verifier_model_revision=verifier_model_revision,
            attempt_no=attempt_no,
        )
    if validation.status != PASS:
        return _attempt(
            hypothesis,
            attempt_no=attempt_no,
            verdict=UNCERTAIN if validation.status == REVIEW_ONLY else REJECTED,
            reason_code=validation.reason_code,
            supported_spans=(),
            independent_call=False,
            verifier_model_revision=verifier_model_revision,
        )
    blind_role_audit: dict[str, Any] = {}
    effective_hypothesis = hypothesis
    direction_normalization = "UNRESOLVED"
    frame_direction_repaired = False
    try:
        # Generated hypotheses may present the same unordered occurrence pair
        # in either direction.  Keep immutable source order available for the
        # predicate-blind inventory's endpoint bookkeeping, but do not use it
        # as a substitute for the directed claim supplied by the classifier.
        if opportunity.left_span <= opportunity.right_span:
            source_first = opportunity.left_occurrence_key
            source_second = opportunity.right_occurrence_key
        else:
            source_first = opportunity.right_occurrence_key
            source_second = opportunity.left_occurrence_key
        atomic_fact_audit: dict[str, Any] = {
            "status": "NOT_CONFIGURED",
            "direct_pair_present": None,
        }
        structural_parent_child = _has_hierarchical_parent_child_evidence(
            opportunity
        )
        if structural_parent_child:
            # Atomic fact inventory is a predicate-blind audit of sentence or
            # clause propositions.  An explicit parser-confirmed list edge is
            # a different, immutable source of evidence: requiring the model
            # to restate a layout fact as a textual frame turns valid
            # structure into an avoidable UNCERTAIN result.  Keep this audit
            # distinct from COMPLETE so the later atomic-frame vetoes remain
            # disabled; the independent role/schema/final-verifier gates still
            # decide whether the structural claim is publishable.
            atomic_fact_audit = {
                "status": "STRUCTURAL_PARENT_CHILD",
                "coverage_complete": True,
                "direct_pair_present": None,
            }
        elif atomic_fact_probe is not None:
            atomic_prompt = atomic_fact_inventory_prompt(
                opportunity, content=content,
            )
            context_text = content[
                opportunity.context_span[0]:opportunity.context_span[1]
            ]
            try:
                atomic_fact_audit = parse_atomic_fact_inventory(
                    atomic_fact_probe(
                        atomic_prompt,
                        atomic_fact_inventory_response_schema(),
                    ),
                    context_text=context_text,
                )
            except Exception as exc:
                # This is an auxiliary, predicate-blind audit.  Its protocol
                # failure cannot prove that the candidate is false, and it
                # must not suppress the independent ordered verifier.  Keep a
                # visible UNKNOWN state and error, but continue fail-closed:
                # publication still requires the final verifier, exact
                # endpoint evidence and the independent role gates below.
                atomic_fact_audit = {
                    "status": "UNAVAILABLE",
                    "coverage_complete": False,
                    "direct_pair_present": None,
                    "error_message": (
                        f"{type(exc).__name__}: {exc}"[:2000]
                    ),
                }
            if not atomic_fact_audit["coverage_complete"]:
                # Coverage is an auxiliary recall audit.  An incomplete
                # predicate-blind inventory must not suppress the independent
                # predicate-aware verifier: the latter still has to pass
                # exact evidence, ordered roles, schema/type and scope gates.
                # Keep the audit explicitly incomplete so the stricter direct
                # atomic-pair veto is not bypassed by accident.
                if atomic_fact_audit.get("status") != "UNAVAILABLE":
                    atomic_fact_audit["status"] = "INCOMPLETE"
                atomic_fact_audit["direct_pair_present"] = None
            else:
                endpoint_a = _occurrence_details(
                    opportunity, source_first, content,
                )
                endpoint_b = _occurrence_details(
                    opportunity, source_second, content,
                )
                atomic_fact_audit["status"] = "COMPLETE"
                atomic_fact_audit["direct_pair_present"] = (
                    atomic_fact_inventory_contains_pair(
                        atomic_fact_audit,
                        str(endpoint_a["surface"]),
                        str(endpoint_b["surface"]),
                    )
                )
        # The predicate-blind probe is an auditable recall/role hint. It does
        # not know the requested Schema predicate and small models frequently
        # label nested but valid propositions as ``direct_relation=false``.
        # Exact spans and endpoint existence were already checked
        # deterministically; only the final predicate-aware verifier may veto
        # semantic publication. Auxiliary protocol failures therefore degrade
        # to an explicit UNKNOWN audit instead of suppressing the verifier.
        try:
            blind_role_audit = parse_verifier_role_probe(role_probe(
                verifier_role_probe_prompt(
                    hypothesis, opportunity, content=content,
                ),
                verifier_role_probe_response_schema(),
            ))
            blind_role_audit["probe_status"] = "AVAILABLE"
            endpoint_argument_mismatches: list[str] = []
            for label, occurrence_key in (
                ("endpoint_a", source_first),
                ("endpoint_b", source_second),
            ):
                endpoint = _occurrence_details(
                    opportunity, occurrence_key, content,
                )
                start = blind_role_audit.get(f"{label}_argument_start")
                end = blind_role_audit.get(f"{label}_argument_end")
                span = endpoint["span"]
                valid = (
                    isinstance(start, int)
                    and not isinstance(start, bool)
                    and isinstance(end, int)
                    and not isinstance(end, bool)
                    and opportunity.context_span[0] <= start < end
                    <= opportunity.context_span[1]
                    and start <= span[0] < span[1] <= end
                )
                matches = bool(valid and (start, end) == tuple(span))
                blind_role_audit[f"{label}_argument_span_status"] = (
                    "EXACT" if matches else "EXPANDED" if valid else "INVALID"
                )
                # A model assertion that an endpoint is incomplete is only a
                # publication veto when it is grounded by a valid containing
                # source span. Conversely, a valid larger argument span makes
                # a contradictory ``complete=true`` answer auditable rather
                # than silently allowing the model to add omitted words in its
                # prose explanation. No language-, domain-, or predicate-name
                # rule is involved in this comparison.
                if valid and not matches:
                    endpoint_argument_mismatches.append(label)
            blind_role_audit["incomplete_endpoint_arguments"] = tuple(
                endpoint_argument_mismatches
            )
        except Exception as exc:
            blind_role_audit = {
                "endpoint_a_complete_argument": None,
                "endpoint_b_complete_argument": None,
                "direct_relation_present": None,
                "subject_role": "source endpoint A role unresolved",
                "object_role": "source endpoint B role unresolved",
                "expressed_relation": "",
                "incomplete_endpoint_arguments": (),
                "probe_status": "UNAVAILABLE",
                "probe_error": f"{type(exc).__name__}: {exc}"[:500],
            }

        try:
            raw_alignment = role_alignment_probe(
                verifier_role_alignment_prompt(
                    hypothesis,
                    logical_schema=logical_schema,
                    blind_role_audit=blind_role_audit,
                    opportunity=opportunity,
                    content=content,
                ),
                verifier_role_alignment_response_schema(),
            )
            alignment = parse_verifier_role_alignment(raw_alignment)
            semantic_contrast = (
                parse_verifier_semantic_contrast(raw_alignment)
                if isinstance(raw_alignment, Mapping)
                and raw_alignment.get("semantic_contrast")
                else "UNCERTAIN"
            )
            forward_alignment = alignment["forward_alignment"]
            reverse_alignment = alignment["reverse_alignment"]
            exact_argument_orientation = (
                str(raw_alignment.get("exact_argument_orientation") or "")
                .strip().upper()
                if isinstance(raw_alignment, Mapping) else ""
            )
            exact_forward_supported = bool(
                raw_alignment.get("exact_argument_forward_supported")
            ) if isinstance(raw_alignment, Mapping) else False
            exact_reverse_supported = bool(
                raw_alignment.get("exact_argument_reverse_supported")
            ) if isinstance(raw_alignment, Mapping) else False
            # Scripted/frozen contracts before the explicit per-direction
            # audit flags retain their orientation-only representation.
            if not (exact_forward_supported or exact_reverse_supported):
                exact_forward_supported = exact_argument_orientation in {
                    "FORWARD", "SYMMETRIC",
                }
                exact_reverse_supported = exact_argument_orientation in {
                    "REVERSE", "SYMMETRIC",
                }
            blind_role_audit["alignment_status"] = "AVAILABLE"
        except Exception as exc:
            raw_alignment = {}
            forward_alignment = "UNCERTAIN"
            reverse_alignment = "UNCERTAIN"
            exact_argument_orientation = "UNCERTAIN"
            exact_forward_supported = False
            exact_reverse_supported = False
            semantic_contrast = "UNCERTAIN"
            blind_role_audit["alignment_status"] = "UNAVAILABLE"
            blind_role_audit["alignment_error"] = (
                f"{type(exc).__name__}: {exc}"[:500]
            )
        blind_role_audit.update({
            "forward_alignment": forward_alignment,
            "reverse_alignment": reverse_alignment,
            "exact_argument_orientation": exact_argument_orientation,
            "exact_argument_forward_supported": exact_forward_supported,
            "exact_argument_reverse_supported": exact_reverse_supported,
            "semantic_contrast": semantic_contrast,
        })
        spec = normalize_predicate_schema(logical_schema).get(
            hypothesis.predicate
        )
        if spec is None:
            raise RelationVerificationRejected(
                "PREDICATE_METADATA_MISSING",
                "role orientation requires current predicate metadata",
            )
        structural_role_support = False
        structural_adjudication_used = False
        if structural_parent_child:
            structural_kind = (
                "list_parent_child"
                if _has_hierarchical_list_parent_child_evidence(opportunity)
                else "section_label_parent_child"
                if _has_hierarchical_section_label_parent_child_evidence(opportunity)
                else ""
            )
            structural_child, structural_parent = _structural_parent_child_keys(
                opportunity
            )
            structural_role_support = bool(
                structural_kind
                and structural_kind in spec.structural_evidence_kinds
                and hypothesis.subject_occurrence_key == structural_child
                and hypothesis.object_occurrence_key == structural_parent
            )
        # Auxiliary role and compact-orientation passes are audit signals.  A
        # small model can bind them to endpoint position instead of semantic
        # role, so they must never rewrite the claim before the authoritative
        # ordered verifier sees it.  Keep the classifier's directed claim as
        # the authoritative candidate; source order is an audit coordinate,
        # not a relation direction.  This is language-, domain- and
        # predicate-name independent and preserves inverse/list relations such
        # as component-to-whole claims whose whole appears first in the text.
        effective_hypothesis = hypothesis
        direction_normalization = "FORWARD"
        blind_role_audit.update({
            "subject_alignment": forward_alignment,
            "object_alignment": forward_alignment,
            "direction_normalization": direction_normalization,
        })
        raw = verifier(
            verifier_prompt(
                effective_hypothesis, opportunity,
                content=content,
                logical_schema=logical_schema,
                blind_role_audit=blind_role_audit,
            ),
            verifier_response_schema(effective_hypothesis.hypothesis_id),
        )
        # Some constrained local models return a valid SUPPORTED decision but
        # omit the evidence array.  Recover only evidence that was already
        # supplied by the immutable, validated hypothesis (or, for the
        # stricter atomic audit path, the frozen context span).  This repairs a
        # response-protocol omission; it does not invent a new source span.
        # All endpoint, atomic-scope, role, direction and schema gates below
        # still run after the recovery.
        raw_for_parse = raw
        parsed_raw = raw
        if isinstance(parsed_raw, str):
            try:
                parsed_raw = json.loads(parsed_raw)
            except json.JSONDecodeError:
                parsed_raw = None
        if (
            isinstance(parsed_raw, Mapping)
            and str(parsed_raw.get("verdict") or "") == SUPPORTED
        ):
            repaired_raw = dict(parsed_raw)
            raw_repaired = False
            # Older schema-capable local models use one ``entailment`` field
            # and call the evidence array ``evidence_spans``.  Accept those
            # transport aliases only when the independent immutable audits
            # can reconstruct the ordered directions; the aliases themselves
            # never establish a relation.
            if (
                not repaired_raw.get("supported_evidence_spans")
                and isinstance(repaired_raw.get("evidence_spans"), list)
            ):
                repaired_raw["supported_evidence_spans"] = list(
                    repaired_raw["evidence_spans"]
                )
                raw_repaired = True
            fallback_spans = tuple(hypothesis.evidence_spans)
            fallback_is_valid = bool(
                fallback_spans
                and all(
                    opportunity.context_span[0] <= span.start < span.end
                    <= opportunity.context_span[1]
                    and span.end <= len(content)
                    for span in fallback_spans
                )
                and _spans_cover(fallback_spans, opportunity.left_span)
                and _spans_cover(fallback_spans, opportunity.right_span)
                and _spans_stay_within_required_atomic_scope(
                    fallback_spans, opportunity,
                )
            )
            if not repaired_raw.get("supported_evidence_spans") and fallback_is_valid:
                repaired_raw["supported_evidence_spans"] = [
                    {
                        "start": span.start,
                        "end": span.end,
                    }
                    for span in fallback_spans
                ]
                raw_repaired = True

            # The final verifier's ordered entailment fields are required for
            # publication.  Some constrained local models consistently emit
            # the overall SUPPORTED verdict and exact roles but omit these two
            # enum fields.  The independent exact-argument audit has already
            # evaluated the same two directions from frozen source text and
            # Schema roles.  Reuse only its boolean result to repair an
            # omitted field; arbitrary non-empty values remain fail-closed.
            exact_forward = blind_role_audit.get(
                "exact_argument_forward_supported"
            ) is True
            exact_reverse = blind_role_audit.get(
                "exact_argument_reverse_supported"
            ) is True
            alignment_forward = (
                (
                    forward_alignment == "ALIGNED"
                    and reverse_alignment == "ALIGNED"
                ) if spec.symmetric else (
                    forward_alignment == "ALIGNED"
                    and reverse_alignment != "ALIGNED"
                )
            )
            alignment_reverse = (
                (
                    forward_alignment == "ALIGNED"
                    and reverse_alignment == "ALIGNED"
                ) if spec.symmetric else (
                    reverse_alignment == "ALIGNED"
                    and forward_alignment != "ALIGNED"
                )
            )
            if exact_forward or exact_reverse:
                repaired_forward = exact_forward
                repaired_reverse = exact_reverse
            else:
                # The compact exact NLI pass can answer false for both
                # directions when it follows the role-only wording too
                # literally. A positive final verdict plus the independent
                # semantic role comparison is a bounded protocol repair; the
                # evidence, endpoint and publication gates still apply.
                repaired_forward = alignment_forward
                repaired_reverse = alignment_reverse
            for field_name, audit_name in (
                ("forward_entailment", "exact_argument_forward_supported"),
                ("reverse_entailment", "exact_argument_reverse_supported"),
            ):
                field_value = repaired_raw.get(field_name)
                if (
                    field_value is None
                    or (isinstance(field_value, str) and not field_value.strip())
                ) and isinstance(blind_role_audit.get(audit_name), bool):
                    repaired_raw[field_name] = (
                        "ENTAILED"
                        if (
                            repaired_forward
                            if field_name == "forward_entailment"
                            else repaired_reverse
                        )
                        else "NOT_ENTAILED"
                    )
                    raw_repaired = True

            legacy_entailment = _normalize_entailment_audit(
                repaired_raw.get("entailment")
                or repaired_raw.get("relation_entailment")
                or repaired_raw.get("direction_entailment")
            )
            if legacy_entailment == "ENTAILED":
                if spec.symmetric and exact_forward and exact_reverse:
                    repaired_raw["forward_entailment"] = "ENTAILED"
                    repaired_raw["reverse_entailment"] = "ENTAILED"
                    raw_repaired = True
                elif exact_forward and not exact_reverse:
                    repaired_raw["forward_entailment"] = "ENTAILED"
                    repaired_raw["reverse_entailment"] = "NOT_ENTAILED"
                    raw_repaired = True
                elif exact_reverse and not exact_forward:
                    repaired_raw["forward_entailment"] = "NOT_ENTAILED"
                    repaired_raw["reverse_entailment"] = "ENTAILED"
                    raw_repaired = True
                elif alignment_forward and not alignment_reverse:
                    repaired_raw["forward_entailment"] = "ENTAILED"
                    repaired_raw["reverse_entailment"] = "NOT_ENTAILED"
                    raw_repaired = True
                elif alignment_reverse and not alignment_forward:
                    repaired_raw["forward_entailment"] = "NOT_ENTAILED"
                    repaired_raw["reverse_entailment"] = "ENTAILED"
                    raw_repaired = True

            # The same constrained-output behavior can omit the two required
            # endpoint-role booleans while retaining verdict=SUPPORTED.  The
            # exact directional audit is independent and returns true only
            # when both exact endpoints satisfy the required roles and the
            # source supports that orientation.  Fill only an absent/null
            # field, never an explicit false, and only for that confirmed
            # hypothesis direction.
            if blind_role_audit.get(
                "exact_argument_forward_supported"
            ) is True:
                for field_name in (
                    "subject_role_supported", "object_role_supported",
                ):
                    field_value = repaired_raw.get(field_name)
                    if field_value is None:
                        repaired_raw[field_name] = True
                        raw_repaired = True

            role_alignment_confirms_forward = (
                (
                    forward_alignment == "ALIGNED"
                    and reverse_alignment == "ALIGNED"
                ) if spec.symmetric else (
                    forward_alignment == "ALIGNED"
                    and reverse_alignment != "ALIGNED"
                )
            )
            if role_alignment_confirms_forward:
                for field_name in (
                    "subject_role_supported", "object_role_supported",
                ):
                    field_value = repaired_raw.get(field_name)
                    if field_value is None:
                        repaired_raw[field_name] = True
                        raw_repaired = True

            if raw_repaired:
                raw_for_parse = {
                    **repaired_raw,
                }
        raw_for_parse = _apply_independent_forward_orientation_audit(
            raw_for_parse,
            hypothesis=effective_hypothesis,
            logical_schema=logical_schema,
            independent_orientation_audit=(
                independent_orientation_audit
            ),
        )
        attempt = _parse_verifier_response(
            effective_hypothesis,
            opportunity,
            raw_for_parse,
            content=content,
            logical_schema=logical_schema,
            blind_role_audit=blind_role_audit,
            direction_normalization=direction_normalization,
            attempt_no=attempt_no,
            verifier_model_revision=verifier_model_revision,
        )
        # Structural parent-child evidence has a different adjudication
        # contract from ordinary lexical relations.  A general verifier can
        # correctly notice that no verb or predicate phrase links a label to
        # its indented child, then incorrectly use that observation to reject
        # the frozen source-structure assertion.  Give only this narrowly
        # identified, Schema-admissible case one fresh structural review.  It
        # is still a real verifier call and its result goes through the same
        # parser, exact evidence, orientation and publication gates below;
        # the first decision is never rewritten deterministically.
        if (
            structural_role_support
            and attempt.verdict in {REJECTED, UNCERTAIN}
        ):
            try:
                structural_raw = verifier(
                    structural_verifier_prompt(
                        effective_hypothesis,
                        opportunity,
                        content=content,
                        logical_schema=logical_schema,
                    ),
                    structural_verifier_response_schema(
                        effective_hypothesis.hypothesis_id,
                    ),
                )
                structural_attempt = _parse_structural_verifier_response(
                    effective_hypothesis,
                    opportunity,
                    structural_raw,
                    content=content,
                    logical_schema=logical_schema,
                    blind_role_audit=blind_role_audit,
                    direction_normalization=direction_normalization,
                    attempt_no=attempt_no,
                    verifier_model_revision=verifier_model_revision,
                )
                structural_source = str(
                    structural_attempt.normalized_subject_occurrence_key
                    or effective_hypothesis.subject_occurrence_key
                )
                structural_target = str(
                    structural_attempt.normalized_object_occurrence_key
                    or effective_hypothesis.object_occurrence_key
                )
                if (
                    structural_attempt.verdict == SUPPORTED
                    and structural_source
                    == effective_hypothesis.subject_occurrence_key
                    and structural_target
                    == effective_hypothesis.object_occurrence_key
                ):
                    attempt = structural_attempt
                    structural_adjudication_used = True
            except Exception:
                # The narrow review is recall-oriented but never allowed to
                # turn a transport/protocol failure into a publishable fact.
                # Preserve the original terminal decision and let the normal
                # fail-closed path report it.
                pass
        normalized_source = str(
            attempt.normalized_subject_occurrence_key
            or effective_hypothesis.subject_occurrence_key
        )
        normalized_target = str(
            attempt.normalized_object_occurrence_key
            or effective_hypothesis.object_occurrence_key
        )
        # ``candidate_source`` is persisted as hypothesis provenance, not as
        # predicate metadata.  Direct proposer hypotheses do, however, retain
        # the proposer prompt contract in the immutable hypothesis fields.
        # Keep the explicit metadata form for replay fixtures and older
        # ledgers, then use that contract as the runtime fallback.  Without
        # this distinction the direct-proposer evidence fallback and the
        # multi-argument direction confirmation are silently disabled in the
        # production path.
        candidate_source = str(
            hypothesis.predicate_metadata.get("candidate_source") or ""
        ).strip().upper()
        direct_proposer_candidate = (
            candidate_source == DIRECT_PROPOSER
            or (
                not candidate_source
                and hypothesis.classifier_prompt_version
                == GROUNDED_RELATION_PROPOSER_PROMPT_VERSION
            )
        )
        source_frame_candidate = (
            candidate_source == "ATOMIC_FRAME_SCHEMA_PROJECTION"
        )
        direct_proposer_authoritative_fallback = bool(
            attempt.verdict == SUPPORTED
            and direct_proposer_candidate
            and attempt.subject_role_supported
            and attempt.object_role_supported
            and bool(attempt.supported_evidence_spans)
            and (
                atomic_fact_audit.get("status") != "COMPLETE"
                or atomic_fact_audit.get("direct_pair_present") is True
                or blind_role_audit.get("direct_relation_present") is True
            )
        )
        direct_proposer_multi_argument = (
            direct_proposer_candidate
            and atomic_fact_audit.get("status") == "COMPLETE"
            and atomic_fact_audit.get("direct_pair_present") is True
            and _atomic_inventory_has_multi_argument_endpoint_frame(
                atomic_fact_audit,
                subject_surface=str(
                    _occurrence_details(
                        opportunity, normalized_source, content,
                    )["surface"]
                ),
                object_surface=str(
                    _occurrence_details(
                        opportunity, normalized_target, content,
                    )["surface"]
                ),
            )
        )
        direct_proposer_exact_conflict_fallback = bool(
            direct_proposer_authoritative_fallback
            and (
                atomic_fact_audit.get("direct_pair_present") is not True
                and blind_role_audit.get("direct_relation_present") is True
            )
        )
        expected_orientation = (
            "SYMMETRIC" if spec.symmetric else
            "FORWARD" if (
                normalized_source == hypothesis.subject_occurrence_key
                and normalized_target == hypothesis.object_occurrence_key
            ) else "REVERSE" if (
                normalized_source == hypothesis.object_occurrence_key
                and normalized_target == hypothesis.subject_occurrence_key
            ) else "NEITHER"
        )
        if spec.symmetric:
            semantic_role_support = (
                forward_alignment == "ALIGNED"
                and reverse_alignment == "ALIGNED"
            )
            exact_role_support = (
                exact_forward_supported and exact_reverse_supported
            )
        else:
            expected_alignment = (
                forward_alignment
                if expected_orientation == "FORWARD"
                else reverse_alignment
                if expected_orientation == "REVERSE"
                else "MISALIGNED"
            )
            opposite_alignment = (
                reverse_alignment
                if expected_orientation == "FORWARD"
                else forward_alignment
            )
            semantic_role_support = (
                expected_alignment == "ALIGNED"
                and opposite_alignment != "ALIGNED"
            )
            exact_role_support = (
                exact_forward_supported
                if expected_orientation == "FORWARD"
                else exact_reverse_supported
                if expected_orientation == "REVERSE"
                else False
            )
        auxiliary_unavailable = (
            blind_role_audit.get("probe_status") == "UNAVAILABLE"
            or blind_role_audit.get("alignment_status") == "UNAVAILABLE"
        )
        if attempt.verdict == SUPPORTED and auxiliary_unavailable:
            if structural_adjudication_used:
                # The dedicated structural adjudicator is itself the
                # independent role/schema review for this parser-confirmed
                # child-to-parent binding. Do not turn an unrelated auxiliary
                # probe outage into UNCERTAIN after that exact structural
                # claim has passed the ordered verifier and evidence parser.
                return replace(
                    attempt,
                    reason_code=(
                        "SUPPORTED_STRUCTURAL_ADJUDICATION_AUXILIARY_UNAVAILABLE"
                    ),
                )
            if (
                (direct_proposer_candidate or source_frame_candidate)
                and attempt.supported_evidence_spans
            ):
                # A direct proposer is source-scoped and carries exact frozen
                # endpoints.  The ordered verifier has already returned a
                # complete SUPPORTED result with endpoint-covering evidence;
                # only the auxiliary role probe is unavailable.  Preserve
                # that result instead of converting it to UNCERTAIN.  The
                # fallback is limited to proposer candidates and does not
                # bypass the final verifier, evidence, direction, Schema or
                # normalization gates.  This also avoids discarding a valid
                # fact merely because one optional audit channel timed out.
                return replace(
                    attempt,
                    reason_code=(
                        "SUPPORTED_DIRECT_PROPOSER_AUXILIARY_UNAVAILABLE_FALLBACK"
                    ),
                )
            if attempt_no < Config.relation_max_attempts:
                raise RelationClassificationError(
                    "required independent role audit was unavailable; retry"
                )
            return _attempt(
                effective_hypothesis,
                attempt_no=attempt_no,
                verdict=UNCERTAIN,
                reason_code="AUXILIARY_ROLE_AUDIT_UNAVAILABLE",
                supported_spans=(),
                independent_call=True,
                verifier_model_revision=verifier_model_revision,
                forward_entailment=attempt.forward_entailment,
                reverse_entailment=attempt.reverse_entailment,
                subject_role_supported=False,
                object_role_supported=False,
                subject_context_role=attempt.subject_context_role,
                object_context_role=attempt.object_context_role,
                subject_role_alignment=attempt.subject_role_alignment,
                object_role_alignment=attempt.object_role_alignment,
                forward_role_alignment=attempt.forward_role_alignment,
                reverse_role_alignment=attempt.reverse_role_alignment,
                direction_normalization=attempt.direction_normalization,
                normalized_subject_occurrence_key=normalized_source,
                normalized_object_occurrence_key=normalized_target,
            )
        if (
            attempt.verdict in {REJECTED, UNCERTAIN}
            and attempt_no < Config.relation_max_attempts
            and (semantic_role_support or exact_role_support)
        ):
            # A bounded second final-verifier attempt resolves stochastic
            # disagreement without weakening publication: the first negative
            # result never publishes, and the retry still has to return a
            # schema-valid SUPPORTED verdict and pass the same role consensus.
            return _attempt(
                effective_hypothesis,
                attempt_no=attempt_no,
                verdict=RETRYABLE_FAILED,
                reason_code="VERIFIER_AUXILIARY_DISAGREEMENT_RETRY",
                supported_spans=(),
                independent_call=True,
                verifier_model_revision=verifier_model_revision,
                forward_entailment=attempt.forward_entailment,
                reverse_entailment=attempt.reverse_entailment,
                subject_role_supported=False,
                object_role_supported=False,
                subject_context_role=attempt.subject_context_role,
                object_context_role=attempt.object_context_role,
                subject_role_alignment=attempt.subject_role_alignment,
                object_role_alignment=attempt.object_role_alignment,
                forward_role_alignment=attempt.forward_role_alignment,
                reverse_role_alignment=attempt.reverse_role_alignment,
                direction_normalization=attempt.direction_normalization,
                normalized_subject_occurrence_key=normalized_source,
                normalized_object_occurrence_key=normalized_target,
            )
        frame_alignment = "NOT_CONFIGURED"
        if (
            attempt.verdict == SUPPORTED
            and atomic_fact_audit.get("status") == "COMPLETE"
            and atomic_frame_alignment_probe is not None
        ):
            aligned_subject = _occurrence_details(
                opportunity, normalized_source, content,
            )
            aligned_object = _occurrence_details(
                opportunity, normalized_target, content,
            )
            if _atomic_inventory_has_direct_endpoint_pair(
                atomic_fact_audit,
                subject_surface=str(aligned_subject["surface"]),
                object_surface=str(aligned_object["surface"]),
            ):
                try:
                    frame_alignment_prompt = atomic_frame_schema_alignment_prompt(
                        atomic_fact_audit,
                        subject_surface=str(aligned_subject["surface"]),
                        object_surface=str(aligned_object["surface"]),
                        subject_type=str(aligned_subject["type"]),
                        object_type=str(aligned_object["type"]),
                        predicate=hypothesis.predicate,
                        logical_schema=logical_schema,
                    )
                    frame_alignment = parse_atomic_frame_schema_alignment(
                        atomic_frame_alignment_probe(
                            frame_alignment_prompt,
                            atomic_frame_schema_alignment_response_schema(),
                        ),
                        inventory=atomic_fact_audit,
                        subject_surface=str(aligned_subject["surface"]),
                        object_surface=str(aligned_object["surface"]),
                        symmetric=bool(spec.symmetric),
                    )
                except Exception:
                    # The inventory's exact pair and the ordered verifier's
                    # evidence/role checks remain usable when this auxiliary
                    # bounded call has a protocol or coverage failure.  An
                    # unavailable frame audit must not erase a direct fact.
                    frame_alignment = "UNAVAILABLE"
            if frame_alignment == "REVERSED" and not spec.symmetric:
                initial_orientation = (
                    "FORWARD"
                    if (
                        normalized_source
                        == hypothesis.subject_occurrence_key
                        and normalized_target
                        == hypothesis.object_occurrence_key
                    )
                    else "REVERSE"
                    if (
                        normalized_source
                        == hypothesis.object_occurrence_key
                        and normalized_target
                        == hypothesis.subject_occurrence_key
                    )
                    else "NEITHER"
                )
                initial_role_confirms_direction = (
                    (
                        initial_orientation == "FORWARD"
                        and (
                            (
                                forward_alignment == "ALIGNED"
                                and reverse_alignment != "ALIGNED"
                            )
                            or (
                                exact_forward_supported
                                and not exact_reverse_supported
                            )
                        )
                    )
                    or (
                        initial_orientation == "REVERSE"
                        and (
                            (
                                reverse_alignment == "ALIGNED"
                                and forward_alignment != "ALIGNED"
                            )
                            or (
                                exact_reverse_supported
                                and not exact_forward_supported
                            )
                        )
                    )
                )
                if initial_role_confirms_direction and not direct_proposer_candidate:
                    # The candidate-blind frame mapper is an auxiliary
                    # orientation signal.  When the final ordered verifier
                    # and the independent Schema-role comparison already
                    # agree on a classifier direction, a conflicting frame
                    # result is an audit discrepancy, not permission to
                    # rewrite that classifier claim. Direct-proposer claims
                    # are recall-oriented and must instead obtain a fresh
                    # ordered verification for the frame's opposite binding;
                    # the corrected call remains the publication authority.
                    frame_alignment = "UNAVAILABLE"
                else:
                # The candidate-blind frame mapper may discover the valid
                # Schema binding in the direction opposite to the first
                # verifier result.  Treat that result as a repair proposal,
                # never as publication evidence: swap the exact occurrences
                # and ask the independent ordered verifier to adjudicate that
                # corrected claim in a fresh model call.  The corrected
                # orientation is accepted only when this call returns
                # SUPPORTED for its own forward claim.  A rejection,
                # uncertainty, protocol failure, or a model swap back to the
                # original direction remains fail-closed.
                    corrected_hypothesis = replace(
                        effective_hypothesis,
                        subject_occurrence_key=normalized_target,
                        object_occurrence_key=normalized_source,
                    )
                    corrected_direction_normalization = (
                        "FORWARD"
                        if (
                            corrected_hypothesis.subject_occurrence_key
                            == hypothesis.subject_occurrence_key
                            and corrected_hypothesis.object_occurrence_key
                            == hypothesis.object_occurrence_key
                        )
                        else "REVERSED"
                    )
                    corrected_raw = verifier(
                        verifier_prompt(
                            corrected_hypothesis,
                            opportunity,
                            content=content,
                            logical_schema=logical_schema,
                            blind_role_audit=blind_role_audit,
                        ),
                        verifier_response_schema(
                            corrected_hypothesis.hypothesis_id
                        ),
                    )
                    corrected_attempt = _parse_verifier_response(
                        corrected_hypothesis,
                        opportunity,
                        corrected_raw,
                        content=content,
                        logical_schema=logical_schema,
                        blind_role_audit=blind_role_audit,
                        direction_normalization=(
                            corrected_direction_normalization
                        ),
                        attempt_no=attempt_no,
                        verifier_model_revision=verifier_model_revision,
                    )
                    corrected_source = str(
                        corrected_attempt.normalized_subject_occurrence_key
                        or corrected_hypothesis.subject_occurrence_key
                    )
                    corrected_target = str(
                        corrected_attempt.normalized_object_occurrence_key
                        or corrected_hypothesis.object_occurrence_key
                    )
                    if not (
                        corrected_attempt.verdict == SUPPORTED
                        and corrected_source
                        == corrected_hypothesis.subject_occurrence_key
                        and corrected_target
                        == corrected_hypothesis.object_occurrence_key
                    ):
                        raise RelationVerificationRejected(
                            "ATOMIC_FRAME_DIRECTION_CONFLICT",
                            "opposite frame orientation was not independently supported",
                            forward_entailment="NOT_ENTAILED",
                            reverse_entailment="ENTAILED",
                            subject_role_supported=False,
                            object_role_supported=False,
                        )
                    attempt = corrected_attempt
                    effective_hypothesis = corrected_hypothesis
                    normalized_source = corrected_source
                    normalized_target = corrected_target
                    frame_direction_repaired = True
                    direction_normalization = (
                        corrected_attempt.direction_normalization
                    )
                    # The same candidate-blind binding that was REVERSED for
                    # the old orientation is ALIGNED for the exact swapped
                    # endpoints. It still cannot publish alone:
                    # ``corrected_attempt`` above is the required independent
                    # SUPPORTED verification.
                    frame_alignment = "ALIGNED"
        if attempt.verdict == SUPPORTED and not spec.symmetric:
            normalized_source = str(
                attempt.normalized_subject_occurrence_key
                or effective_hypothesis.subject_occurrence_key
            )
            normalized_target = str(
                attempt.normalized_object_occurrence_key
                or effective_hypothesis.object_occurrence_key
            )
            normalized_orientation = (
                "FORWARD"
                if (
                    normalized_source == hypothesis.subject_occurrence_key
                    and normalized_target == hypothesis.object_occurrence_key
                )
                else "REVERSE"
                if (
                    normalized_source == hypothesis.object_occurrence_key
                    and normalized_target == hypothesis.subject_occurrence_key
                )
                else "NEITHER"
            )
            reverse_role_support = (
                reverse_alignment == "ALIGNED"
                and forward_alignment != "ALIGNED"
            )
            if (
                normalized_orientation == "REVERSE"
                and not frame_direction_repaired
                and frame_alignment in {"NOT_CONFIGURED", "UNAVAILABLE"}
                and not reverse_role_support
            ):
                raise RelationVerificationRejected(
                    "DIRECTION_CORROBORATION_MISSING",
                    "reversed endpoint orientation lacks an independent source-role or frame confirmation",
                    forward_entailment="NOT_ENTAILED",
                    reverse_entailment="NOT_ENTAILED",
                    subject_role_supported=False,
                    object_role_supported=False,
                )
            if frame_alignment == "NOT_ALIGNED":
                if direct_proposer_authoritative_fallback:
                    # The final ordered verifier has already bound the exact
                    # endpoint roles and evidence. A predicate-blind frame
                    # mapper can fail to recognize a nested or coordinated
                    # frame; retain the audit discrepancy without vetoing the
                    # direct-proposer candidate.
                    frame_alignment = "UNAVAILABLE"
                else:
                    raise RelationVerificationRejected(
                        "ATOMIC_FRAME_RELATION_MISMATCH",
                        "source frame relation does not instantiate the Schema relation",
                        forward_entailment="NOT_ENTAILED",
                        reverse_entailment="NOT_ENTAILED",
                        subject_role_supported=False,
                        object_role_supported=False,
                    )
        # The final ordered verifier may call a relation SUPPORTED only after
        # the independent predicate-blind role reading has been compared with
        # the versioned Schema semantics.  Previously an unavailable or
        # UNCERTAIN role comparison could still publish solely on the final
        # model's boolean role fields.  That admitted peer-list cross products,
        # third-entity substitution and direction errors.  Require positive
        # agreement for the exact normalized orientation, while keeping every
        # semantic decision model-driven and predicate-name independent.
        if attempt.verdict == SUPPORTED:
            # The compact exact-argument audit and the source-grounded
            # Schema-role comparison are independent corroborating signals.
            # Requiring both compounds local-model false negatives on passive,
            # coordinated and long-distance clauses.  The final independent
            # verifier plus either positive signal is sufficient. Explicit
            # third-argument substitutions remain subject to the final
            # source-grounded verifier, exact argument completeness, Schema
            # type constraints and the independent role signal; the first
            # probe's model conclusion is never itself a publication veto.
            if not (
                semantic_role_support
                or exact_role_support
                or structural_role_support
                or frame_alignment == "ALIGNED"
                or direct_proposer_authoritative_fallback
            ):
                raise RelationVerificationRejected(
                    "ROLE_ALIGNMENT_NOT_CONFIRMED",
                    "independent Schema-role comparison did not confirm the exact orientation",
                    forward_entailment="NOT_ENTAILED",
                    reverse_entailment="NOT_ENTAILED",
                    subject_role_supported=False,
                    object_role_supported=False,
                )
            if (
                atomic_fact_audit.get("status") == "COMPLETE"
                and atomic_fact_audit.get("direct_pair_present") is not True
                and not direct_proposer_authoritative_fallback
            ):
                raise RelationVerificationRejected(
                    "NO_DIRECT_ATOMIC_PAIR",
                    "complete predicate-blind fact inventory does not directly pair the exact endpoints",
                    forward_entailment="NOT_ENTAILED",
                    reverse_entailment="NOT_ENTAILED",
                    subject_role_supported=False,
                    object_role_supported=False,
                )
        # A model must not expand an incomplete endpoint only in its role
        # paraphrase and then publish the shorter stored entity.  The veto is
        # grounded in source offsets returned before predicate metadata is
        # revealed, rather than in any language-specific suffix or cue.
        if (
            attempt.verdict == SUPPORTED
            and blind_role_audit.get("incomplete_endpoint_arguments")
            and not structural_adjudication_used
        ):
            raise RelationVerificationRejected(
                "INCOMPLETE_ENDPOINT_ARGUMENT",
                "predicate-blind complete argument span exceeds the exact endpoint",
                forward_entailment="NOT_ENTAILED",
                reverse_entailment="NOT_ENTAILED",
                subject_role_supported=False,
                object_role_supported=False,
            )

        # The compact exact-direction pass is an independent conflict signal.
        # A single small-model call is not allowed to veto the authoritative
        # ordered verifier when the separately derived Schema-role alignment
        # supports that same direction.  Publication is blocked only when the
        # exact audit and the role/semantic alignment independently concur on
        # the mismatch. This retains a strict two-signal gate without turning
        # passive voice or long-distance coordination into false negatives.
        if attempt.verdict == SUPPORTED and not spec.symmetric:
            normalized_source = str(
                attempt.normalized_subject_occurrence_key
                or effective_hypothesis.subject_occurrence_key
            )
            normalized_target = str(
                attempt.normalized_object_occurrence_key
                or effective_hypothesis.object_occurrence_key
            )
            expected_exact_orientation = (
                "FORWARD"
                if (
                    normalized_source == hypothesis.subject_occurrence_key
                    and normalized_target == hypothesis.object_occurrence_key
                )
                else "REVERSE"
                if (
                    normalized_source == hypothesis.object_occurrence_key
                    and normalized_target == hypothesis.subject_occurrence_key
                )
                else "NEITHER"
            )
            expected_role_alignment = (
                forward_alignment
                if expected_exact_orientation == "FORWARD"
                else reverse_alignment
            )
            opposite_role_alignment = (
                reverse_alignment
                if expected_exact_orientation == "FORWARD"
                else forward_alignment
            )
            exact_conflicts = (
                exact_argument_orientation in {
                    "FORWARD", "REVERSE", "NEITHER", "SYMMETRIC",
                }
                and exact_argument_orientation != expected_exact_orientation
            )
            role_confirms_conflict = (
                expected_role_alignment == "MISALIGNED"
                and (
                    exact_argument_orientation == "NEITHER"
                    or opposite_role_alignment == "ALIGNED"
                    or (
                        forward_alignment == "MISALIGNED"
                        and reverse_alignment == "MISALIGNED"
                    )
                )
            )
            if (
                exact_conflicts
                and role_confirms_conflict
                and frame_alignment != "ALIGNED"
                and not structural_adjudication_used
            ):
                if not direct_proposer_exact_conflict_fallback:
                    raise RelationVerificationRejected(
                        "EXACT_ARGUMENT_DIRECTION_CONFLICT",
                        "exact endpoint audit disagrees with the supported direction",
                        forward_entailment="NOT_ENTAILED",
                        reverse_entailment="NOT_ENTAILED",
                        subject_role_supported=False,
                        object_role_supported=False,
                    )
        elif (
            attempt.verdict == SUPPORTED
            and spec.symmetric
            and exact_argument_orientation in {"FORWARD", "REVERSE", "NEITHER"}
            and forward_alignment != "ALIGNED"
            and reverse_alignment != "ALIGNED"
        ):
            raise RelationVerificationRejected(
                "EXACT_ARGUMENT_DIRECTION_CONFLICT",
                "exact endpoint audit does not support the symmetric relation",
                forward_entailment="NOT_ENTAILED",
                reverse_entailment="NOT_ENTAILED",
                subject_role_supported=False,
                object_role_supported=False,
            )
        return attempt
    except RelationVerificationRejected as exc:
        return _attempt(
            effective_hypothesis,
            attempt_no=attempt_no,
            verdict=REJECTED,
            reason_code=exc.reason_code,
            supported_spans=(),
            independent_call=True,
            verifier_model_revision=verifier_model_revision,
            forward_entailment=exc.forward_entailment,
            reverse_entailment=exc.reverse_entailment,
            subject_role_supported=exc.subject_role_supported,
            object_role_supported=exc.object_role_supported,
            subject_context_role=str(
                blind_role_audit.get("subject_role") or ""
            ),
            object_context_role=str(
                blind_role_audit.get("object_role") or ""
            ),
            subject_role_alignment=str(
                blind_role_audit.get("subject_alignment") or ""
            ),
            object_role_alignment=str(
                blind_role_audit.get("object_alignment") or ""
            ),
            forward_role_alignment=str(
                blind_role_audit.get("forward_alignment") or ""
            ),
            reverse_role_alignment=str(
                blind_role_audit.get("reverse_alignment") or ""
            ),
            direction_normalization=direction_normalization,
            normalized_subject_occurrence_key=(
                effective_hypothesis.subject_occurrence_key
            ),
            normalized_object_occurrence_key=(
                effective_hypothesis.object_occurrence_key
            ),
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )
    except Exception as exc:
        return _attempt(
            effective_hypothesis,
            attempt_no=attempt_no,
            verdict=RETRYABLE_FAILED,
            reason_code="VERIFIER_CALL_OR_RESPONSE_FAILED",
            supported_spans=(),
            independent_call=True,
            verifier_model_revision=verifier_model_revision,
            subject_context_role=str(
                blind_role_audit.get("subject_role") or ""
            ),
            object_context_role=str(
                blind_role_audit.get("object_role") or ""
            ),
            subject_role_alignment=str(
                blind_role_audit.get("subject_alignment") or ""
            ),
            object_role_alignment=str(
                blind_role_audit.get("object_alignment") or ""
            ),
            forward_role_alignment=str(
                blind_role_audit.get("forward_alignment") or ""
            ),
            reverse_role_alignment=str(
                blind_role_audit.get("reverse_alignment") or ""
            ),
            direction_normalization=direction_normalization,
            normalized_subject_occurrence_key=(
                effective_hypothesis.subject_occurrence_key
            ),
            normalized_object_occurrence_key=(
                effective_hypothesis.object_occurrence_key
            ),
            error_message=f"{type(exc).__name__}: {exc}"[:2000],
        )


def supported_for_normalization(attempt: VerificationAttempt) -> bool:
    return bool(
        attempt.verdict == SUPPORTED
        and attempt.independent_call
        and attempt.supported_evidence_spans
    )


def _canonicalized_fact_parts(
    *,
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    spec: PredicateSpec,
) -> tuple[str, str, str]:
    subject = (
        opportunity.left_canonical_id
        if hypothesis.subject_occurrence_key == opportunity.left_occurrence_key
        else opportunity.right_canonical_id
    )
    object_ = (
        opportunity.left_canonical_id
        if hypothesis.object_occurrence_key == opportunity.left_occurrence_key
        else opportunity.right_canonical_id
    )
    predicate = spec.name
    if spec.canonical_orientation == CANONICAL_INVERSE:
        predicate = spec.inverse_predicate
        subject, object_ = object_, subject
    elif spec.canonical_orientation == CANONICAL_SYMMETRIC:
        subject, object_ = sorted((subject, object_))
    elif spec.canonical_orientation != CANONICAL_FORWARD:
        raise ValueError("unsupported canonical orientation")
    return subject, predicate, object_


def normalize_verified_hypotheses(
    *,
    project_id: int,
    opportunities: Mapping[str, PairOpportunity],
    hypotheses: Sequence[RelationHypothesis],
    verification_attempts: Sequence[VerificationAttempt],
    logical_schema: Mapping[str, Any],
) -> RelationNormalizationResult:
    """Normalize only independently supported hypotheses into facts plus evidence."""

    schema = normalize_predicate_schema(logical_schema)
    unique_hypotheses: dict[str, RelationHypothesis] = {}
    exact_input_duplicates = 0
    for hypothesis in hypotheses:
        existing = unique_hypotheses.get(hypothesis.hypothesis_id)
        if existing is not None:
            if existing != hypothesis:
                raise ValueError("immutable hypothesis id collision")
            exact_input_duplicates += 1
        else:
            unique_hypotheses[hypothesis.hypothesis_id] = hypothesis

    attempts_by_hypothesis: dict[str, dict[int, VerificationAttempt]] = {}
    for attempt in verification_attempts:
        values = attempts_by_hypothesis.setdefault(attempt.hypothesis_id, {})
        existing = values.get(int(attempt.attempt_no))
        if existing is not None and existing != attempt:
            raise ValueError("immutable verification attempt collision")
        values[int(attempt.attempt_no)] = attempt

    accepted: dict[str, VerificationAttempt] = {}
    for hypothesis_id, attempts in attempts_by_hypothesis.items():
        supported = [
            item for _, item in sorted(attempts.items())
            if supported_for_normalization(item)
        ]
        if supported:
            accepted[hypothesis_id] = supported[0]

    fact_map: dict[str, NormalizedRelationFact] = {}
    normalized_mode_by_fact: dict[str, bool] = {}
    identity_expression_by_fact: dict[str, str] = {}
    evidence_map: dict[str, RelationFactEvidence] = {}
    collapse_audit: list[dict[str, Any]] = []
    conflicts: dict[str, RelationFactConflict] = {}
    illegal_self_loops = 0
    symmetric_inverse_collapses = 0
    exact_fact_collapses = exact_input_duplicates
    normalized_fact_by_hypothesis: dict[str, str] = {}

    for hypothesis_id, hypothesis in sorted(unique_hypotheses.items()):
        attempt = accepted.get(hypothesis_id)
        if attempt is None:
            continue
        opportunity = opportunities.get(hypothesis.pair_identity)
        if opportunity is None:
            conflict_id = "relconflict_" + _digest((
                hypothesis_id, "PAIR_LEDGER_MISSING",
            ))
            conflicts[conflict_id] = RelationFactConflict(
                conflict_id=conflict_id,
                pair_identity=hypothesis.pair_identity,
                hypothesis_id=hypothesis_id,
                reason_code="PAIR_LEDGER_MISSING",
                metadata={},
            )
            continue
        spec = schema.get(hypothesis.predicate)
        if spec is None:
            conflict_id = "relconflict_" + _digest((
                hypothesis_id, "UNKNOWN_PREDICATE",
            ))
            conflicts[conflict_id] = RelationFactConflict(
                conflict_id=conflict_id,
                pair_identity=hypothesis.pair_identity,
                hypothesis_id=hypothesis_id,
                reason_code="UNKNOWN_PREDICATE",
                metadata={},
            )
            continue
        normalized_hypothesis = replace(
            hypothesis,
            subject_occurrence_key=(
                attempt.normalized_subject_occurrence_key
                or hypothesis.subject_occurrence_key
            ),
            object_occurrence_key=(
                attempt.normalized_object_occurrence_key
                or hypothesis.object_occurrence_key
            ),
        )
        if (
            bool(spec.open_relation_policy.get("enabled"))
            and (
                normalized_hypothesis.subject_occurrence_key
                != hypothesis.subject_occurrence_key
                or normalized_hypothesis.object_occurrence_key
                != hypothesis.object_occurrence_key
            )
        ):
            conflict_id = "relconflict_" + _digest((
                hypothesis_id, "OPEN_RELATION_DIRECTION_CHANGED",
            ))
            conflicts[conflict_id] = RelationFactConflict(
                conflict_id=conflict_id,
                pair_identity=hypothesis.pair_identity,
                hypothesis_id=hypothesis_id,
                reason_code="OPEN_RELATION_DIRECTION_CHANGED",
                metadata={},
            )
            continue
        if {
            normalized_hypothesis.subject_occurrence_key,
            normalized_hypothesis.object_occurrence_key,
        } != {
            opportunity.left_occurrence_key,
            opportunity.right_occurrence_key,
        }:
            conflict_id = "relconflict_" + _digest((
                hypothesis_id, "VERIFIED_DIRECTION_ENDPOINT_MISMATCH",
            ))
            conflicts[conflict_id] = RelationFactConflict(
                conflict_id=conflict_id,
                pair_identity=hypothesis.pair_identity,
                hypothesis_id=hypothesis_id,
                reason_code="VERIFIED_DIRECTION_ENDPOINT_MISMATCH",
                metadata={},
            )
            continue
        try:
            subject, predicate, object_ = _canonicalized_fact_parts(
                hypothesis=normalized_hypothesis,
                opportunity=opportunity,
                spec=spec,
            )
        except ValueError as exc:
            conflict_id = "relconflict_" + _digest((
                hypothesis_id, "NORMALIZATION_METADATA_INVALID",
            ))
            conflicts[conflict_id] = RelationFactConflict(
                conflict_id=conflict_id,
                pair_identity=hypothesis.pair_identity,
                hypothesis_id=hypothesis_id,
                reason_code="NORMALIZATION_METADATA_INVALID",
                metadata={"error": str(exc)},
            )
            continue
        if subject == object_ and not spec.reflexive:
            illegal_self_loops += 1
            conflict_id = "relconflict_" + _digest((
                hypothesis_id, "ILLEGAL_SELF_LOOP",
            ))
            conflicts[conflict_id] = RelationFactConflict(
                conflict_id=conflict_id,
                pair_identity=hypothesis.pair_identity,
                hypothesis_id=hypothesis_id,
                reason_code="ILLEGAL_SELF_LOOP",
                metadata={},
            )
            continue
        source_relation_expression = source_relation_display_expression(
            (hypothesis.predicate_metadata or {}).get(
                "source_frame_relation"
            )
        )
        identity_expression = ""
        if bool(spec.open_relation_policy.get("identity_source_expression")):
            identity_expression = _source_relation_identity_expression(
                source_relation_expression
            )
            if not identity_expression:
                conflict_id = "relconflict_" + _digest((
                    hypothesis_id, "OPEN_RELATION_IDENTITY_MISSING",
                ))
                conflicts[conflict_id] = RelationFactConflict(
                    conflict_id=conflict_id,
                    pair_identity=hypothesis.pair_identity,
                    hypothesis_id=hypothesis_id,
                    reason_code="OPEN_RELATION_IDENTITY_MISSING",
                    metadata={},
                )
                continue
        semantic_material: tuple[Any, ...] = (
            int(project_id), subject, predicate, object_,
        )
        if identity_expression:
            semantic_material += (
                "source_relation_expression", identity_expression,
            )
        semantic_key = "relfact_" + _digest(semantic_material)
        normalized_fact_by_hypothesis[hypothesis_id] = semantic_key
        if semantic_key not in fact_map:
            fact_map[semantic_key] = NormalizedRelationFact(
                fact_id=semantic_key,
                semantic_fact_key=semantic_key,
                project_id=int(project_id),
                subject_canonical_id=subject,
                predicate=predicate,
                object_canonical_id=object_,
                symmetric=spec.symmetric,
                reflexive=spec.reflexive,
                review_only=spec.publish_mode != PUBLISH_AUTO,
                normalization_eligible=spec.publish_mode == PUBLISH_AUTO,
                evidence_count=0,
            )
            normalized_mode_by_fact[semantic_key] = bool(
                spec.symmetric or predicate != hypothesis.predicate
            )
            identity_expression_by_fact[semantic_key] = identity_expression
        else:
            normalized_changed = bool(
                normalized_mode_by_fact.get(semantic_key)
                or spec.symmetric
                or predicate != hypothesis.predicate
            )
            reason = (
                "SYMMETRIC_OR_INVERSE_DUPLICATE"
                if normalized_changed else "EXACT_SEMANTIC_DUPLICATE"
            )
            symmetric_inverse_collapses += int(normalized_changed)
            exact_fact_collapses += int(not normalized_changed)
            collapse_audit.append({
                "fact_id": semantic_key,
                "hypothesis_id": hypothesis_id,
                "reason_code": reason,
            })
        spans_material = json.dumps(
            [[item.start, item.end] for item in attempt.supported_evidence_spans],
            separators=(",", ":"),
        )
        evidence_key = "relevidence_" + _digest((
            semantic_key,
            opportunity.document_revision_id,
            normalized_hypothesis.subject_occurrence_key,
            hypothesis.predicate,
            normalized_hypothesis.object_occurrence_key,
            spans_material,
            hypothesis.hypothesis_id,
        ))
        if evidence_key not in evidence_map:
            candidate_source = str(
                (hypothesis.predicate_metadata or {}).get(
                    "candidate_source"
                ) or ""
            ).upper()
            source_projection = candidate_source in {
                "ATOMIC_FRAME_SCHEMA_PROJECTION",
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            }
            evidence_map[evidence_key] = RelationFactEvidence(
                evidence_id=evidence_key,
                evidence_key=evidence_key,
                fact_id=semantic_key,
                document_revision_id=opportunity.document_revision_id,
                subject_occurrence_key=(
                    normalized_hypothesis.subject_occurrence_key
                ),
                object_occurrence_key=(
                    normalized_hypothesis.object_occurrence_key
                ),
                evidence_spans=attempt.supported_evidence_spans,
                hypothesis_id=hypothesis.hypothesis_id,
                verification_attempt_id=attempt.attempt_id,
                verifier_supported=True,
                provenance={
                    "pair_identity": hypothesis.pair_identity,
                    "classifier_policy_version": hypothesis.classifier_policy_version,
                    "classifier_model_revision": hypothesis.classifier_model_revision,
                    "verifier_model_revision": attempt.verifier_model_revision,
                    "direction_normalization": (
                        attempt.direction_normalization
                    ),
                    "classifier_subject_occurrence_key": (
                        hypothesis.subject_occurrence_key
                    ),
                    "classifier_object_occurrence_key": (
                        hypothesis.object_occurrence_key
                    ),
                    **(
                        {
                            "source_relation_expression": (
                                source_relation_expression
                            ),
                            "source_relation_identity": identity_expression,
                            "source_relation_projection": True,
                            "source_frame_index": (
                                hypothesis.predicate_metadata or {}
                            ).get("source_frame_index"),
                            "source_binding_source": str(
                                (hypothesis.predicate_metadata or {}).get(
                                    "source_binding_source"
                                ) or ""
                            ),
                            "open_relation_carrier": bool(
                                spec.open_relation_policy.get("enabled")
                            ),
                        }
                        if source_projection and source_relation_expression
                        else {}
                    ),
                },
            )

    # Rejected, uncertain, or retryable reverse hypotheses are normal audited
    # negative outcomes.  They never become facts, but they also must not veto
    # an independently SUPPORTED forward fact.  When *both* directions were
    # independently supported, however, the versioned predicate policy must
    # explicitly permit reciprocity.  Otherwise neither direction is safe to
    # publish: choosing one here would invent semantic evidence that the
    # deterministic normalizer does not have.
    unsupported_reverse_conflicts = 0

    reciprocal_pairs: list[tuple[str, str]] = []
    seen_reciprocal_pairs: set[tuple[str, str]] = set()
    for fact in tuple(fact_map.values()):
        if fact.symmetric:
            continue
        reverse_material: tuple[Any, ...] = (
            int(project_id),
            fact.object_canonical_id,
            fact.predicate,
            fact.subject_canonical_id,
        )
        identity_expression = identity_expression_by_fact.get(
            fact.fact_id, ""
        )
        if identity_expression:
            reverse_material += (
                "source_relation_expression", identity_expression,
            )
        reverse_key = "relfact_" + _digest(reverse_material)
        if reverse_key not in fact_map or reverse_key == fact.fact_id:
            continue
        pair_key = tuple(sorted((fact.fact_id, reverse_key)))
        if pair_key in seen_reciprocal_pairs:
            continue
        seen_reciprocal_pairs.add(pair_key)
        reciprocal_pairs.append(pair_key)

    reciprocal_fact_keys: set[str] = set()
    discarded_conflict_collapse_audits = 0
    for left_fact_id, right_fact_id in reciprocal_pairs:
        left_fact = fact_map.get(left_fact_id)
        right_fact = fact_map.get(right_fact_id)
        if left_fact is None or right_fact is None:
            continue
        spec = schema.get(left_fact.predicate)
        if spec is not None and spec.reciprocal_allowed:
            reciprocal_fact_keys.update((left_fact_id, right_fact_id))
            continue

        removed_ids = {left_fact_id, right_fact_id}
        unsupported_reverse_conflicts += len(removed_ids)
        for hypothesis_id, normalized_fact_id in (
            normalized_fact_by_hypothesis.items()
        ):
            if normalized_fact_id not in removed_ids:
                continue
            conflict_id = "relconflict_" + _digest((
                hypothesis_id, "RECIPROCAL_DIRECTION_CONFLICT",
            ))
            conflicts[conflict_id] = RelationFactConflict(
                conflict_id=conflict_id,
                pair_identity=unique_hypotheses[hypothesis_id].pair_identity,
                hypothesis_id=hypothesis_id,
                reason_code="RECIPROCAL_DIRECTION_CONFLICT",
                metadata={
                    "predicate": left_fact.predicate,
                    "conflicting_fact_ids": sorted(removed_ids),
                    "reciprocal_allowed": False,
                },
            )
        for fact_id in removed_ids:
            fact_map.pop(fact_id, None)
        evidence_map = {
            key: value for key, value in evidence_map.items()
            if value.fact_id not in removed_ids
        }
        retained_collapse_audit = [
            item for item in collapse_audit
            if item["fact_id"] not in removed_ids
        ]
        discarded_conflict_collapse_audits += (
            len(collapse_audit) - len(retained_collapse_audit)
        )
        collapse_audit = retained_collapse_audit

    # The open carrier is the source-faithful recall safety net for a semantic
    # Schema projection.  A Schema hypothesis surviving the broad verifier is
    # not enough to remove that net: publication performs a later, narrower
    # role audit and may still reject the Schema mapping.  Preserve both facts
    # through normalization so publication can keep the carrier when that
    # happens, then suppress an exact duplicate only after the Schema fact has
    # passed every publication gate.  The sole early-dominance exception is a
    # parser-owned structure contract that explicitly declares its parallel
    # carrier exhaustive.
    carrier_fact_ids: set[str] = set()
    for hypothesis_id, fact_id in normalized_fact_by_hypothesis.items():
        if fact_id not in fact_map:
            continue
        hypothesis = unique_hypotheses[hypothesis_id]
        hypothesis_spec = schema.get(hypothesis.predicate)
        if (
            hypothesis_spec is not None
            and hypothesis_spec.open_relation_policy.get("enabled")
        ):
            carrier_fact_ids.add(fact_id)

    # Parser structure and the open inventory can describe the same source
    # proposition with different frame indexes and opposite surface
    # orientation (for example, whole->child ``contains`` versus
    # child->whole ``part_of``).  Exact frame-index matching cannot suppress
    # that duplicate even though both hypotheses retain one immutable pair and
    # overlapping evidence.  A parser-contract interpretation may dominate
    # its open carrier on that narrow evidence slice; ordinary model-selected
    # semantic predicates never receive this privilege.
    def parser_structural_interpretation(
        hypothesis: RelationHypothesis,
    ) -> bool:
        metadata = hypothesis.predicate_metadata or {}
        candidate_source = str(
            metadata.get("candidate_source") or ""
        ).upper()
        binding_source = str(
            metadata.get("source_binding_source") or ""
        ).upper()
        return bool(
            candidate_source == "STRUCTURAL_PROJECTION"
            or binding_source.startswith("DETERMINISTIC_SOURCE_STRUCTURE")
        )

    def substantially_overlapping_evidence(
        left: Sequence[EvidenceSpan],
        right: Sequence[EvidenceSpan],
    ) -> bool:
        for left_span in left:
            for right_span in right:
                overlap = min(left_span.end, right_span.end) - max(
                    left_span.start, right_span.start
                )
                if overlap <= 0:
                    continue
                smaller = min(
                    left_span.end - left_span.start,
                    right_span.end - right_span.start,
                )
                if smaller > 0 and overlap / smaller >= 0.8:
                    return True
        return False

    structural_evidence_by_pair: dict[
        str, list[tuple[str, tuple[EvidenceSpan, ...]]]
    ] = {}
    for evidence_value in evidence_map.values():
        if evidence_value.fact_id not in fact_map:
            continue
        hypothesis = unique_hypotheses.get(evidence_value.hypothesis_id)
        hypothesis_spec = (
            schema.get(hypothesis.predicate) if hypothesis else None
        )
        verification_attempt = (
            accepted.get(hypothesis.hypothesis_id) if hypothesis else None
        )
        # A parser-owned contract can itself use the open carrier when no
        # domain predicate safely represents the source structure.  Suppress
        # parallel open-carrier paraphrases only when that parser explicitly
        # declares its interpretation exhaustive for the same endpoint pair.
        # Other list/table/section structures remain additive.  This is keyed
        # only by frozen structure provenance and verifier state; it is
        # independent of document vocabulary and predicate names.
        deterministic_contract_supported = bool(
            hypothesis is not None
            and verification_attempt is not None
            and verification_attempt.reason_code
            == "SUPPORTED_PARSER_STRUCTURE_CONTRACT"
            and str(
                (hypothesis.predicate_metadata or {}).get(
                    "source_parallel_open_carrier_policy"
                )
                or ""
            ).strip()
            == STRUCTURE_PARALLEL_OPEN_CARRIER_SUPPRESSION
        )
        if not (
            hypothesis is not None
            and hypothesis_spec is not None
            and (
                deterministic_contract_supported
                or (
                    hypothesis_spec.publish_mode == PUBLISH_AUTO
                    and not hypothesis_spec.open_relation_policy.get("enabled")
                    and parser_structural_interpretation(hypothesis)
                )
            )
        ):
            continue
        structural_evidence_by_pair.setdefault(
            str(hypothesis.pair_identity), []
        ).append((
            str(hypothesis.hypothesis_id),
            tuple(evidence_value.evidence_spans),
        ))

    suppressed_open_carrier_evidence = 0
    suppressed_open_carrier_structure_evidence = 0
    suppressed_open_carrier_expression_evidence = 0
    if structural_evidence_by_pair:
        retained_evidence: dict[str, RelationFactEvidence] = {}
        for evidence_key, evidence_value in evidence_map.items():
            hypothesis = unique_hypotheses.get(
                evidence_value.hypothesis_id
            )
            hypothesis_spec = (
                schema.get(hypothesis.predicate) if hypothesis else None
            )
            structure_match = bool(
                hypothesis is not None
                and any(
                    structural_hypothesis_id
                    != evidence_value.hypothesis_id
                    and substantially_overlapping_evidence(
                        evidence_value.evidence_spans,
                        precise_spans,
                    )
                    for (
                        structural_hypothesis_id,
                        precise_spans,
                    ) in structural_evidence_by_pair.get(
                        str(hypothesis.pair_identity), ()
                    )
                )
            )
            suppress = bool(
                hypothesis_spec is not None
                and hypothesis_spec.open_relation_policy.get("enabled")
                and structure_match
            )
            if suppress:
                suppressed_open_carrier_evidence += 1
                suppressed_open_carrier_structure_evidence += 1
            else:
                retained_evidence[evidence_key] = evidence_value
        evidence_map = retained_evidence

    evidence_fact_ids = {
        value.fact_id for value in evidence_map.values()
    }
    suppressed_open_carrier_facts = 0
    for fact_id in sorted(carrier_fact_ids - evidence_fact_ids):
        if fact_id in fact_map:
            fact_map.pop(fact_id, None)
            identity_expression_by_fact.pop(fact_id, None)
            suppressed_open_carrier_facts += 1
    if suppressed_open_carrier_facts:
        collapse_audit = [
            item for item in collapse_audit
            if item["fact_id"] in fact_map
        ]

    evidence_counts: dict[str, int] = {}
    for value in evidence_map.values():
        evidence_counts[value.fact_id] = evidence_counts.get(value.fact_id, 0) + 1
    fact_map = {
        key: replace(value, evidence_count=evidence_counts.get(value.fact_id, 0))
        for key, value in fact_map.items()
    }

    facts = list(fact_map.values())
    endpoint_predicates: dict[tuple[str, str], set[str]] = {}
    for fact in facts:
        endpoint_predicates.setdefault(
            tuple(sorted((
                fact.subject_canonical_id,
                fact.object_canonical_id,
            ))),
            set(),
        ).add(fact.predicate)
    legal_multi_predicate_retained = sum(
        len(predicates) for predicates in endpoint_predicates.values()
        if len(predicates) > 1
    )

    return RelationNormalizationResult(
        facts=tuple(sorted(fact_map.values(), key=lambda item: item.semantic_fact_key)),
        evidence=tuple(sorted(
            evidence_map.values(), key=lambda item: item.evidence_key,
        )),
        conflicts=tuple(sorted(
            conflicts.values(), key=lambda item: item.conflict_id,
        )),
        collapse_audit=tuple(sorted(
            collapse_audit,
            key=lambda item: (item["fact_id"], item["hypothesis_id"]),
        )),
        metrics={
            "exact_duplicates_collapsed": exact_fact_collapses,
            "symmetric_inverse_duplicates_collapsed": symmetric_inverse_collapses,
            "unsupported_reverse_conflicts": unsupported_reverse_conflicts,
            "verified_reciprocal_facts": len(reciprocal_fact_keys),
            "discarded_conflict_collapse_audits": (
                discarded_conflict_collapse_audits
            ),
            "illegal_self_loops": illegal_self_loops,
            "open_carrier_facts_suppressed_by_precise": (
                suppressed_open_carrier_facts
            ),
            "open_carrier_evidence_suppressed_by_precise": (
                suppressed_open_carrier_evidence
            ),
            "open_carrier_evidence_suppressed_by_structure": (
                suppressed_open_carrier_structure_evidence
            ),
            "open_carrier_evidence_suppressed_by_source_expression": (
                suppressed_open_carrier_expression_evidence
            ),
            "legal_multi_predicate_retained": legal_multi_predicate_retained,
            "facts_after_normalization": len(fact_map),
            "evidence_after_deduplication": len(evidence_map),
        },
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _component_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def build_relation_materialization_fingerprint(
    *,
    source_generation: int,
    documents: Sequence[Mapping[str, Any]],
    mention_occurrences: Sequence[Mapping[str, Any]],
    entity_assignment: Mapping[str, Any],
    schema: Mapping[str, Any],
    candidate: Mapping[str, Any],
    proposer: Mapping[str, Any],
    classifier: Mapping[str, Any],
    verifier: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    normalization: Mapping[str, Any],
    publish_policy: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
    materialization_root: Mapping[str, Any],
) -> RelationMaterializationFingerprint:
    """Freeze every behavior-controlling input as named, explainable components."""

    if int(source_generation) < 0:
        raise ValueError("source generation must be non-negative")
    required_mappings = {
        "entity_assignment": entity_assignment,
        "schema": schema,
        "candidate": candidate,
        "proposer": proposer,
        "classifier": classifier,
        "verifier": verifier,
        "thresholds": thresholds,
        "normalization": normalization,
        "publish_policy": publish_policy,
        "quality_gate": quality_gate,
        "materialization_root": materialization_root,
    }
    empty = [name for name, value in required_mappings.items() if not dict(value)]
    if empty:
        raise ValueError(f"fingerprint components are empty: {empty}")
    for name, required in {
        "entity_assignment": ("snapshot_id", "version", "hash"),
        "schema": ("version", "hash", "predicate_metadata_hash"),
        "candidate": ("policy_version", "parameters", "budget"),
        "classifier": (
            "prompt_version", "schema_version", "parser_version",
            "model", "model_revision",
        ),
        "proposer": (
            "prompt_version", "schema_version", "parser_version",
            "model", "model_revision", "max_gleaning_rounds",
        ),
        "verifier": (
            "prompt_version", "schema_version", "parser_version",
            "model", "model_revision",
        ),
        "normalization": ("version", "symmetric_rule", "inverse_rule"),
        "publish_policy": ("version", "mode_source", "unsupported"),
        "quality_gate": (
            "version", "relation_precision_noninferiority_delta",
            "confidence_method", "minimum_gold_facts",
        ),
        "materialization_root": ("code_tree", "policy_root"),
    }.items():
        missing = [
            key for key in required
            if required_mappings[name].get(key) is None
            or required_mappings[name].get(key) == ""
        ]
        if missing:
            raise ValueError(f"{name} fingerprint fields missing: {missing}")
    normalized_documents = sorted(
        (
            {
                **dict(item),
                "chunks": sorted(
                    (dict(chunk) for chunk in item.get("chunks") or []),
                    key=lambda chunk: (
                        str(chunk.get("chunk_id") or ""),
                        str(chunk.get("content_hash") or ""),
                    ),
                ),
            }
            for item in documents
        ),
        key=lambda item: (
            str(item.get("document_id") or ""),
            str(item.get("revision_id") or ""),
        ),
    )
    if any(
        not item.get("revision_id")
        or not item.get("content_hash")
        or any(not chunk.get("content_hash") for chunk in item.get("chunks") or [])
        for item in normalized_documents
    ):
        raise ValueError("document/chunk fingerprint universe is incomplete")
    normalized_mentions = sorted(
        (dict(item) for item in mention_occurrences),
        key=lambda item: (
            str(item.get("occurrence_key") or ""),
            _canonical_json(item.get("span") or []),
        ),
    )
    if any(
        not item.get("occurrence_key")
        or not item.get("content_hash")
        or not item.get("span")
        for item in normalized_mentions
    ):
        raise ValueError("mention occurrence fingerprint universe is incomplete")
    components: dict[str, Any] = {
        "source_generation": int(source_generation),
        "documents": normalized_documents,
        "mention_occurrences": normalized_mentions,
        **{name: dict(value) for name, value in required_mappings.items()},
    }
    component_hashes = {
        name: _component_hash(value) for name, value in components.items()
    }
    root = _component_hash({
        "version": "shadow-relation-materialization-fingerprint-v3",
        "component_hashes": component_hashes,
    })
    return RelationMaterializationFingerprint(
        version="shadow-relation-materialization-fingerprint-v3",
        root=root,
        components=components,
        component_hashes=component_hashes,
    )


_FRESHNESS_CHECKPOINTS = frozenset({
    "evaluate",
    "quality_gate",
    "snapshot_completion",
    "activate",
    "materialize",
    "database_cas",
    "graph_rebuild_input",
})


def compare_relation_fingerprints(
    recorded: RelationMaterializationFingerprint,
    current: RelationMaterializationFingerprint,
    *,
    checkpoint: str,
) -> RelationFreshnessDecision:
    if checkpoint not in _FRESHNESS_CHECKPOINTS:
        raise ValueError(f"unknown freshness checkpoint: {checkpoint}")
    component_names = sorted(
        set(recorded.component_hashes) | set(current.component_hashes)
    )
    changed = tuple(
        name for name in component_names
        if recorded.component_hashes.get(name) != current.component_hashes.get(name)
    )
    is_current = bool(
        recorded.version == current.version
        and recorded.root == current.root
        and not changed
    )
    return RelationFreshnessDecision(
        checkpoint=checkpoint,
        current=is_current,
        status="CURRENT" if is_current else "STALE",
        recorded_root=recorded.root,
        current_root=current.root,
        changed_components=changed,
    )


def assert_relation_fresh(
    recorded: RelationMaterializationFingerprint,
    current: RelationMaterializationFingerprint,
    *,
    checkpoint: str,
) -> RelationFreshnessDecision:
    decision = compare_relation_fingerprints(
        recorded, current, checkpoint=checkpoint,
    )
    if not decision.current:
        raise StaleRelationSnapshotError(
            f"{checkpoint} rejected stale relation snapshot; "
            f"changed={','.join(decision.changed_components)}"
        )
    return decision


def relation_materialization_decision(
    fact: NormalizedRelationFact,
    evidence: Sequence[RelationFactEvidence],
    *,
    recorded_fingerprint: RelationMaterializationFingerprint,
    current_fingerprint: RelationMaterializationFingerprint,
    snapshot_complete: bool,
    current_assignment_and_schema: bool,
    conflicts: Sequence[Any],
) -> RelationMaterializationDecision:
    """Apply the final relation publication boundary without product writes."""

    freshness = compare_relation_fingerprints(
        recorded_fingerprint,
        current_fingerprint,
        checkpoint="materialize",
    )
    reasons: set[str] = set()
    if not freshness.current:
        reasons.add("STALE_FINGERPRINT")
    if not snapshot_complete:
        reasons.add("SNAPSHOT_INCOMPLETE")
    if not current_assignment_and_schema:
        reasons.add("ASSIGNMENT_OR_SCHEMA_STALE")
    if conflicts:
        reasons.add("UNRESOLVED_CONFLICT")
    if (
        fact.review_only
        or not fact.normalization_eligible
    ):
        reasons.add("PREDICATE_REVIEW_ONLY")
    if (
        fact.subject_canonical_id == fact.object_canonical_id
        and not fact.reflexive
    ):
        reasons.add("ILLEGAL_SELF_LOOP")
    supported_evidence = [
        item for item in evidence
        if item.fact_id == fact.fact_id
        and item.verifier_supported
        and bool(item.evidence_spans)
    ]
    if not supported_evidence:
        reasons.add("VERIFIER_EVIDENCE_MISSING")
    return RelationMaterializationDecision(
        fact_id=fact.fact_id,
        eligible=not reasons,
        reasons=tuple(sorted(reasons)),
        freshness=freshness,
    )


def _validated_occurrences(
    content: str,
    mentions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_key: dict[str, dict[str, Any]] = {}
    for raw in mentions:
        key = str(
            raw.get("occurrence_key")
            or raw.get("mention_key")
            or raw.get("mention_id")
            or ""
        )
        surface = str(raw.get("surface") or raw.get("name") or "")
        start, end = raw.get("mention_start"), raw.get("mention_end")
        if not key or not surface or start is None or end is None:
            continue
        start, end = int(start), int(end)
        if (
            start < 0
            or end <= start
            or end > len(content)
            or content[start:end] != surface
        ):
            continue
        normalized = {
            **dict(raw),
            "occurrence_key": key,
            "mention_id": str(raw.get("mention_id") or key),
            "surface": surface,
            "mention_start": start,
            "mention_end": end,
            "canonical_id": str(raw.get("canonical_id") or ""),
            "assignment_version": str(raw.get("assignment_version") or ""),
            "type": str(raw.get("type") or ""),
        }
        existing = by_key.get(key)
        if existing is not None:
            comparable_existing = {
                name: value for name, value in existing.items()
                if name != "mention_id"
            }
            comparable_normalized = {
                name: value for name, value in normalized.items()
                if name != "mention_id"
            }
            if comparable_existing != comparable_normalized:
                raise ValueError(f"occurrence key collision: {key}")
            if str(normalized["mention_id"]) < str(existing["mention_id"]):
                by_key[key] = normalized
            continue
        by_key[key] = normalized
    ordered = sorted(
        by_key.values(),
        key=lambda item: (
            str(item["occurrence_key"]),
            int(item["mention_start"]),
            int(item["mention_end"]),
        ),
    )
    return ordered, max(0, len(mentions) - len(ordered))


def _occurrence_scope(
    occurrence: Mapping[str, Any],
    *,
    content: str,
    sentences: Sequence[tuple[int, int, str]],
) -> dict[str, Any]:
    """Read a runtime-provided atomic scope, with sentence-only legacy fallback."""

    start = int(occurrence["mention_start"])
    end = int(occurrence["mention_end"])
    raw_start = occurrence.get("scope_start")
    raw_end = occurrence.get("scope_end")
    raw_id = str(occurrence.get("scope_id") or "").strip()
    raw_kind = str(occurrence.get("scope_kind") or "").strip()
    try:
        scope_start = int(raw_start)
        scope_end = int(raw_end)
    except (TypeError, ValueError):
        scope_start, scope_end = -1, -1
    if (
        raw_id
        and raw_kind
        and 0 <= scope_start <= start < end <= scope_end <= len(content)
    ):
        return {
            "scope_id": raw_id,
            "scope_kind": raw_kind,
            "start": scope_start,
            "end": scope_end,
            "source": "structured",
        }
    sentence = sentence_index(sentences, start)
    if sentence is not None:
        sentence_start, sentence_end, _ = sentences[sentence]
        if sentence_start <= start < end <= sentence_end:
            return {
                "scope_id": f"legacy-sentence:{sentence}:{sentence_start}:{sentence_end}",
                "scope_kind": "sentence",
                "start": int(sentence_start),
                "end": int(sentence_end),
                "source": "legacy_sentence",
            }
    return {
        "scope_id": "invalid:occurrence-scope",
        "scope_kind": "invalid",
        "start": start,
        "end": end,
        "source": "invalid",
    }


def _source_role(occurrence: Mapping[str, Any]) -> str:
    return str(occurrence.get("source_role") or "body").strip() or "body"


def _relation_eligible(occurrence: Mapping[str, Any]) -> bool:
    value = occurrence.get("relation_eligible")
    return True if value is None else bool(value)


def _metadata_allows_structure(
    metadata: Mapping[str, Any],
    *,
    left_role: str,
    right_role: str,
    same_scope: bool,
) -> bool:
    allowed_roles = {
        str(value) for value in metadata.get("allowed_source_roles") or []
    }
    if left_role not in allowed_roles or right_role not in allowed_roles:
        return False
    scope_policy = metadata.get("scope_policy") or {}
    if not isinstance(scope_policy, Mapping):
        return False
    if same_scope:
        return True
    return bool(scope_policy.get("allow_cross_scope_with_explicit_evidence"))


_STABLE_INLINE_IDENTIFIER_TOKEN = re.compile(
    r"(?:[A-Za-z]{1,12}[-_]\d{1,8}|(?:§\s*)?\d+(?:\.\d+){1,4})"
)


def _stable_inline_identifier_family(surface: str) -> str:
    """Return a syntax family for a bounded source identifier."""

    value = str(surface or "").strip()
    if _STABLE_INLINE_IDENTIFIER_TOKEN.fullmatch(value) is None:
        return ""
    alpha = re.fullmatch(
        r"(?P<prefix>[A-Za-z]{1,12})[-_]\d{1,8}", value,
    )
    if alpha:
        return "alpha:" + str(alpha.group("prefix")).casefold()
    numeric = re.sub(r"^§\s*", "", value)
    return f"numeric-depth:{numeric.count('.') + 1}"


def _is_hierarchical_list_parent_child(
    left: Mapping[str, Any], right: Mapping[str, Any],
) -> bool:
    """Recognize only an explicit syntax-derived list parent/child edge.

    This is a recall gate, not a relation assertion.  The source parser must
    have assigned both occurrences to the same list group and one occurrence's
    immediate parent scope must be the other occurrence's scope.  Siblings,
    unrelated list items and ordinary far-apart prose remain distance-bound.
    """

    if _source_role(left) != "list_item" or _source_role(right) != "list_item":
        return False
    # The parser scope covers the complete list-item line.  Only its principal
    # (earliest, longest co-starting) mention represents that hierarchy node;
    # parenthetical details and other mentions inside the same line must not
    # be promoted to siblings of the item label.  ``None`` keeps direct unit
    # fixtures and old replay artifacts backward-compatible, while current
    # runtime occurrences always carry an explicit boolean marker.
    if (
        left.get("list_scope_head") is False
        or right.get("list_scope_head") is False
    ):
        return False
    group_left = str(left.get("list_group_id") or "")
    group_right = str(right.get("list_group_id") or "")
    if not group_left or group_left != group_right:
        return False
    left_scope = str(left.get("scope_id") or "")
    right_scope = str(right.get("scope_id") or "")
    return bool(
        left_scope and right_scope and (
            str(left.get("list_parent_scope_id") or "") == right_scope
            or str(right.get("list_parent_scope_id") or "") == left_scope
        )
    )


def _has_hierarchical_list_parent_child_evidence(
    opportunity: PairOpportunity,
) -> bool:
    """Return whether the immutable opportunity carries parser proof of a direct list edge.

    This is deliberately narrower than a generic structured context.  A list
    item being near another item is not evidence; only the source mapper's
    same-group, immediate-parent relationship can select this evidence path.
    """

    provenance = opportunity.provenance
    return bool(
        provenance.get("hierarchical_list_parent_child") is True
        and str(provenance.get("structure_admission") or "")
        == "hierarchical_list_parent_child"
    )


def _is_hierarchical_section_label_parent_child(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    content: str,
) -> bool:
    """Recognize a standalone structural label immediately followed by content.

    Many document parsers preserve a section/field label as an untyped body
    block rather than as a heading.  This helper uses only layout and
    punctuation boundaries: a complete one-line scope, followed by a new
    line and a second complete scope.  It does not inspect vocabulary or
    predicate names, and it is only a candidate admission signal; Schema,
    classifier, role checks and the independent verifier still decide whether
    a relation exists.
    """

    if _source_role(left) != "body" or _source_role(right) != "body":
        return False
    ordered = sorted((left, right), key=lambda item: (
        int(item.get("scope_start") or item.get("mention_start") or 0),
        int(item.get("scope_end") or item.get("mention_end") or 0),
    ))
    parent, child = ordered
    parent_start = int(parent.get("scope_start") or parent.get("mention_start") or 0)
    parent_end = int(parent.get("scope_end") or parent.get("mention_end") or 0)
    child_start = int(child.get("scope_start") or child.get("mention_start") or 0)
    child_end = int(child.get("scope_end") or child.get("mention_end") or 0)
    if not (
        0 <= parent_start < parent_end <= len(content)
        and 0 <= child_start < child_end <= len(content)
        and parent_end <= child_start
        and str(parent.get("scope_id") or "")
        != str(child.get("scope_id") or "")
    ):
        return False
    label = content[parent_start:parent_end].strip()
    if not label or len(label) > 160 or "\n" in label or "\r" in label:
        return False
    # A colon-terminated line can be either a compact field/section label or
    # a prose lead-in for a following example.  The latter must not acquire a
    # parent edge to an arbitrary entity several lines inside the example
    # block.  Keep this layout-only ambiguity fail-closed: long multi-word
    # lead-ins are not labels, while ordinary short labels (including labels
    # written in CJK without spaces) remain eligible.
    if label.endswith((":", "：")) and len(label) > 48 and len(label.split()) > 4:
        return False
    # A standalone label may end in a colon (e.g. a field/section label),
    # while sentence-final punctuation is a strong signal that this is prose.
    if re.search(r"[.!?;。！？；]$", label):
        return False
    between = content[parent_end:child_start]
    if not between:
        return False
    if "\n" not in between and "\r" not in between:
        return False
    parent_line_start = content.rfind("\n", 0, parent_start) + 1
    parent_line_prefix = content[parent_line_start:parent_start]
    if parent_line_prefix.strip():
        return False
    parent_line_end = content.find("\n", parent_start)
    if parent_line_end < 0:
        parent_line_end = len(content)
    if parent_end <= parent_line_end:
        parent_tail = content[parent_end:parent_line_end]
    else:
        # Legacy sentence scopes may include the line terminator while the
        # structure mapper's line scope excludes it. Treat only terminators
        # between the scope end and line end as layout, never as label text.
        parent_tail = content[parent_line_end:parent_end]
    if parent_tail.strip():
        return False
    child_line_start = content.rfind("\n", 0, child_start) + 1
    if content[child_line_start:child_start].strip():
        return False
    # A section label governs a direct child block, not every entity mentioned
    # inside a later sentence in that block. Require the bound occurrence to
    # be the first meaningful token on its child line (optionally after a
    # generic list/ordinal marker). This keeps a label connected to a source
    # identifier at the start of a requirement line while rejecting embedded
    # actors, objects and prose mentions. The rule is purely layout-based.
    child_prefix = content[
        child_line_start:int(child.get("mention_start") or 0)
    ]
    if child_prefix.strip() and not re.fullmatch(
        r"[ \t]*(?:(?:[-*+]\s+)|(?:\d+[.)]\s+))?",
        child_prefix,
    ):
        return False
    child_surface = content[
        int(child.get("mention_start") or 0):
        int(child.get("mention_end") or 0)
    ].strip()
    stable_inline_identifier = bool(
        child.get("inline_label_candidate") is True
        or _STABLE_INLINE_IDENTIFIER_TOKEN.fullmatch(child_surface)
    )
    explicit_label_punctuation = label.endswith((":", "："))
    # Ordinary adjacent standalone lines are ubiquitous in figures, title
    # pages and OCR output. They become hierarchy only with an explicit label
    # delimiter or a syntax-recognized stable identifier at the start of the
    # child line.  The latter preserves ordinary requirement/specification
    # inventories such as ``SR-01 ...`` without promoting arbitrary compact
    # diagram labels into a parent/child chain. This is a candidate-admission
    # rule and remains independent of domain vocabulary.
    if not (explicit_label_punctuation or stable_inline_identifier):
        return False
    # Keep the child structurally local.  The endpoint itself may be anywhere
    # in the child scope, but an occurrence that is not actually contained in
    # the declared scope must not receive the structural exemption.
    if not (
        int(parent.get("mention_start") or 0) >= parent_start
        and int(parent.get("mention_end") or 0) <= parent_end
        and int(child.get("mention_start") or 0) >= child_start
        and int(child.get("mention_end") or 0) <= child_end
    ):
        return False
    # The next meaningful line is the direct child of the standalone label.
    # A later contiguous line is eligible only when its first endpoint is a
    # parser-marked/stable source identifier (for example SR-02 under a
    # Requirements label). Ordinary nouns several clauses later must not
    # become ``part_of`` the section title. Explicit list hierarchy has its
    # own parser-owned path above.
    if (
        parent_line_end <= child_start
        and content[parent_line_end:child_start].strip() == ""
    ):
        return True
    if not stable_inline_identifier:
        return False
    intermediate = content[parent_line_end + 1:child_line_start]
    if not intermediate or re.search(r"\r?\n[ \t]*\r?\n", intermediate):
        return False
    for raw_line in intermediate.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            return False
        if stripped.startswith("#") or re.match(
            r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped,
        ):
            return False
        if "|" in stripped and stripped.count("|") >= 2:
            return False
        identifier = _STABLE_INLINE_IDENTIFIER_TOKEN.match(stripped)
        if (
            identifier is None
            or (
                identifier.end() < len(stripped)
                and not stripped[identifier.end()].isspace()
            )
            or _stable_inline_identifier_family(
                identifier.group(0)
            ) != _stable_inline_identifier_family(child_surface)
        ):
            return False
    return True


def _structural_parent_child_occurrences(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    list_parent_child: bool,
    section_label_parent_child: bool,
) -> tuple[str, str]:
    """Persist the parser-derived child/parent direction on an opportunity."""

    if list_parent_child:
        left_scope = str(left.get("scope_id") or "")
        right_scope = str(right.get("scope_id") or "")
        if str(left.get("list_parent_scope_id") or "") == right_scope:
            return str(left.get("occurrence_key") or ""), str(right.get("occurrence_key") or "")
        if str(right.get("list_parent_scope_id") or "") == left_scope:
            return str(right.get("occurrence_key") or ""), str(left.get("occurrence_key") or "")
    if section_label_parent_child:
        ordered = sorted(
            (left, right),
            key=lambda item: int(item.get("mention_start") or 0),
        )
        return (
            str(ordered[1].get("occurrence_key") or ""),
            str(ordered[0].get("occurrence_key") or ""),
        )
    return "", ""


def _has_hierarchical_section_label_parent_child_evidence(
    opportunity: PairOpportunity,
) -> bool:
    """Return whether the immutable opportunity carries a label-child proof."""

    provenance = opportunity.provenance
    return bool(
        provenance.get("hierarchical_section_label_parent_child") is True
        and str(provenance.get("structure_admission") or "")
        == "hierarchical_section_label_parent_child"
    )


def _has_hierarchical_parent_child_evidence(
    opportunity: PairOpportunity,
) -> bool:
    return (
        _has_hierarchical_list_parent_child_evidence(opportunity)
        or _has_hierarchical_section_label_parent_child_evidence(opportunity)
    )


def plan_pair_opportunities(
    *,
    project_id: int,
    document_revision_id: str,
    chunk_key: str,
    content: str,
    mentions: Sequence[Mapping[str, Any]],
    logical_schema: Mapping[str, Any],
    run_id: str,
    candidate_policy_version: str,
    max_pair_opportunities: int | None = None,
    sentence_window: int | None = None,
    max_char_distance: int | None = None,
) -> PairOpportunityPlan:
    """Plan each unordered exact occurrence pair exactly once and retain every state."""

    if not document_revision_id or not run_id or not candidate_policy_version:
        raise ValueError("pair planning requires revision, run and candidate policy")
    occurrences, duplicate_count = _validated_occurrences(content, mentions)
    theoretical_pairs = len(occurrences) * max(0, len(occurrences) - 1) // 2
    sentences = split_relation_scopes(content)
    sentence_window = (
        Config.relation_sentence_window
        if sentence_window is None else max(0, int(sentence_window))
    )
    max_char_distance = (
        Config.relation_max_char_distance
        if max_char_distance is None else max(0, int(max_char_distance))
    )
    planned: list[PairOpportunity] = []
    for first, second in combinations(occurrences, 2):
        left, right = sorted(
            (first, second),
            key=lambda item: (
                str(item["occurrence_key"]),
                int(item["mention_start"]),
                int(item["mention_end"]),
            ),
        )
        stable_pair = pair_identity(
            document_revision_id,
            str(left["occurrence_key"]),
            str(right["occurrence_key"]),
        )
        left_sentence = sentence_index(sentences, int(left["mention_start"]))
        right_sentence = sentence_index(sentences, int(right["mention_start"]))
        if left_sentence is None or right_sentence is None:
            context_start = min(int(left["mention_start"]), int(right["mention_start"]))
            context_end = max(int(left["mention_end"]), int(right["mention_end"]))
            sentence_distance = 0
        else:
            first_sentence, last_sentence = sorted((left_sentence, right_sentence))
            context_start = sentences[first_sentence][0]
            context_end = sentences[last_sentence][1]
            sentence_distance = last_sentence - first_sentence
        centre_distance = abs(
            (int(left["mention_start"]) + int(left["mention_end"])) // 2
            - (int(right["mention_start"]) + int(right["mention_end"])) // 2
        )
        context_text = content[context_start:context_end]
        predicate_metadata, type_status = _predicate_metadata(
            str(left.get("type") or ""),
            str(right.get("type") or ""),
            logical_schema,
        )
        left_scope = _occurrence_scope(
            left, content=content, sentences=sentences,
        )
        right_scope = _occurrence_scope(
            right, content=content, sentences=sentences,
        )
        same_scope = str(left_scope["scope_id"]) == str(right_scope["scope_id"])
        left_role = _source_role(left)
        right_role = _source_role(right)
        source_eligible = _relation_eligible(left) and _relation_eligible(right)
        source_role_allowed_metadata = tuple(
            metadata for metadata in predicate_metadata
            if left_role in {
                str(value) for value in metadata.get("allowed_source_roles") or []
            }
            and right_role in {
                str(value) for value in metadata.get("allowed_source_roles") or []
            }
        )
        structurally_allowed_metadata = tuple(
            metadata for metadata in source_role_allowed_metadata
            if _metadata_allows_structure(
                metadata,
                left_role=left_role,
                right_role=right_role,
                same_scope=same_scope,
            )
        )
        if same_scope:
            context_start, context_end = (
                int(left_scope["start"]), int(left_scope["end"]),
            )
        elif structurally_allowed_metadata:
            context_start = min(
                int(left_scope["start"]), int(right_scope["start"]),
            )
            context_end = max(
                int(left_scope["end"]), int(right_scope["end"]),
            )
        context_text = content[context_start:context_end]
        left_table = str(left.get("table_id") or "")
        right_table = str(right.get("table_id") or "")
        left_row = str(left.get("row_id") or "")
        right_row = str(right.get("row_id") or "")
        left_cell = str(left.get("cell_id") or "")
        right_cell = str(right.get("cell_id") or "")
        same_table = bool(left_table and left_table == right_table)
        same_table_row = bool(same_table and left_row and left_row == right_row)
        same_cell = bool(same_table_row and left_cell and left_cell == right_cell)
        different_table_peer_rows = bool(
            same_table and left_row and right_row and left_row != right_row
        )
        same_scope_for_policy = same_scope or same_table_row
        if same_scope_for_policy != same_scope:
            structurally_allowed_metadata = tuple(
                metadata for metadata in source_role_allowed_metadata
                if _metadata_allows_structure(
                    metadata,
                    left_role=left_role,
                    right_role=right_role,
                    same_scope=True,
                )
            )
            context_start, context_end = 0, len(content)
        same_sentence = sentence_distance == 0
        structural_admission_reason = ""
        hierarchical_list_parent_child = _is_hierarchical_list_parent_child(
            left, right,
        )
        hierarchical_section_label_parent_child = (
            _is_hierarchical_section_label_parent_child(
                left, right, content=content,
            )
        )
        structural_child_occurrence_key, structural_parent_occurrence_key = (
            _structural_parent_child_occurrences(
                left,
                right,
                list_parent_child=hierarchical_list_parent_child,
                section_label_parent_child=hierarchical_section_label_parent_child,
            )
        )
        if not source_eligible:
            structural_admission_reason = "ineligible_source_role"
        elif not structurally_allowed_metadata:
            structural_admission_reason = "predicate_structure_policy_denied"
        elif hierarchical_list_parent_child:
            structural_admission_reason = "hierarchical_list_parent_child"
        elif hierarchical_section_label_parent_child:
            structural_admission_reason = "hierarchical_section_label_parent_child"
        elif same_cell:
            structural_admission_reason = "same_cell"
        elif same_table_row:
            structural_admission_reason = "same_table_row"
        elif same_scope_for_policy:
            structural_admission_reason = "same_atomic_scope"
        elif any(
            bool((metadata.get("scope_policy") or {}).get(
                "allow_cross_scope_with_explicit_evidence"
            ))
            for metadata in structurally_allowed_metadata
        ):
            structural_admission_reason = "cross_atomic_scope_schema_policy"
        normalized_context = context_text.casefold()
        soft_cue_matches = sorted(
            (
                {
                    "predicate": str(metadata["name"]),
                    "cue": str(cue),
                }
                for metadata in predicate_metadata
                for cue in metadata.get("optional_soft_cues") or []
                if str(cue).strip()
                and str(cue).casefold() in normalized_context
            ),
            key=lambda item: (item["predicate"], item["cue"]),
        )
        endpoint_order = sorted(
            (
                (int(left["mention_start"]), int(left["mention_end"])),
                (int(right["mention_start"]), int(right["mention_end"])),
            ),
            key=lambda value: (value[0], value[1]),
        )
        pair_local_text = content[
            endpoint_order[0][1]:endpoint_order[1][0]
        ].casefold()
        pair_local_soft_cue_matches = [
            item for item in soft_cue_matches
            if str(item.get("cue") or "").strip().casefold()
            in pair_local_text
        ]
        cue = str(soft_cue_matches[0]["cue"]) if soft_cue_matches else ""
        status = PENDING_CLASSIFICATION
        reason = structural_admission_reason or "local_pair_within_structural_window"
        if (
            str(left.get("canonical_id") or "")
            and str(left.get("canonical_id")) == str(right.get("canonical_id") or "")
        ):
            status, reason = REJECTED_SELF_LOOP, "same_canonical_assignment"
        elif different_table_peer_rows:
            status, reason = FILTERED_STRUCTURE_PEER, "different_table_peer_rows"
        elif type_status == "incompatible":
            # Preserve the most informative terminal ledger state.  The pair
            # remains non-admissible either way, but a known schema type
            # mismatch must not be reported as a generic source-role gate.
            status, reason = TYPE_INCOMPATIBLE, "known_schema_type_incompatibility"
        elif type_status == "unresolved":
            # An exact endpoint-recall projection may be the only source for
            # an endpoint that the primary entity pass did not type. Keep the
            # pair in the classifier/verifier chain rather than discarding it
            # before semantic evidence is read. No type is guessed here; the
            # unresolved state remains visible and the independent verifier
            # still decides whether the exact directed fact is supported.
            endpoint_projection = bool(
                left.get("endpoint_candidate")
                or right.get("endpoint_candidate")
            )
            if endpoint_projection:
                status, reason = (
                    PENDING_CLASSIFICATION,
                    "endpoint_type_requires_verification",
                )
            else:
                status, reason = TYPE_UNRESOLVED, "unknown_endpoint_type"
        elif not source_eligible:
            status, reason = (
                FILTERED_STRUCTURE_INELIGIBLE,
                "source_role_is_not_relation_eligible",
            )
        elif not source_role_allowed_metadata:
            status, reason = (
                FILTERED_STRUCTURE_INELIGIBLE,
                "schema_disallows_source_role",
            )
        elif not structurally_allowed_metadata:
            status, reason = FILTERED_SCOPE_BOUNDARY, "cross_atomic_scope"
        elif (
            structural_admission_reason
            not in {
                "same_cell", "same_table_row", "same_atomic_scope",
                "hierarchical_list_parent_child",
                "hierarchical_section_label_parent_child",
            }
            and (
                sentence_distance > sentence_window
                or centre_distance > max_char_distance
            )
        ):
            status, reason = FILTERED_DISTANCE, "outside_local_context_window"
        structure_rank = {
            "same_cell": 0,
            "same_table_row": 1,
            "same_atomic_scope": 2,
            "cross_atomic_scope_schema_policy": 3,
            "hierarchical_list_parent_child": 0,
            "hierarchical_section_label_parent_child": 0,
        }.get(structural_admission_reason, 9)
        type_rank = {
            "compatible": 0,
            "unresolved": 1,
            "incompatible": 2,
        }.get(type_status, 3)
        priority = (
            3,
            structure_rank,
            type_rank,
            0 if cue else 1,
            sentence_distance,
            centre_distance,
            str(left["occurrence_key"]),
            str(right["occurrence_key"]),
        )
        planned.append(PairOpportunity(
            opportunity_identity=opportunity_identity(
                run_id, candidate_policy_version, stable_pair,
            ),
            pair_identity=stable_pair,
            run_id=run_id,
            project_id=int(project_id),
            document_revision_id=str(document_revision_id),
            chunk_key=str(chunk_key),
            candidate_policy_version=str(candidate_policy_version),
            left_occurrence_key=str(left["occurrence_key"]),
            right_occurrence_key=str(right["occurrence_key"]),
            left_mention_id=str(left["mention_id"]),
            right_mention_id=str(right["mention_id"]),
            left_span=(int(left["mention_start"]), int(left["mention_end"])),
            right_span=(int(right["mention_start"]), int(right["mention_end"])),
            left_canonical_id=str(left.get("canonical_id") or ""),
            right_canonical_id=str(right.get("canonical_id") or ""),
            left_assignment_version=str(left.get("assignment_version") or ""),
            right_assignment_version=str(right.get("assignment_version") or ""),
            left_type=str(left.get("type") or ""),
            right_type=str(right.get("type") or ""),
            context_identity="relctx_" + _digest((
                document_revision_id, chunk_key, context_start, context_end,
            )),
            context_span=(context_start, context_end),
            allowed_predicate_metadata=structurally_allowed_metadata,
            planning_reason=reason,
            cue=cue,
            type_status=type_status,
            status=status,
            provenance={
                "sentence_distance": sentence_distance,
                "character_distance": centre_distance,
                "structure_admission": structural_admission_reason,
                "source_roles": [left_role, right_role],
                "source_relation_eligible": source_eligible,
                "atomic_scopes": [left_scope, right_scope],
                "same_atomic_scope": same_scope_for_policy,
                "hierarchical_list_parent_child": hierarchical_list_parent_child,
                "hierarchical_section_label_parent_child": (
                    hierarchical_section_label_parent_child
                ),
                "structural_child_occurrence_key": (
                    structural_child_occurrence_key
                ),
                "structural_parent_occurrence_key": (
                    structural_parent_occurrence_key
                ),
                "required_atomic_scope": (
                    [0, len(content)]
                    if same_table_row
                    else [int(left_scope["start"]), int(left_scope["end"])]
                    if same_scope
                    else []
                ),
                "allowed_predicates_after_structure": [
                    str(item.get("name") or "")
                    for item in structurally_allowed_metadata
                ],
                "priority": list(priority),
                "soft_cue_matches": soft_cue_matches,
                "pair_local_soft_cue_matches": (
                    pair_local_soft_cue_matches
                ),
                "endpoint_candidates": {
                    "left": bool(left.get("endpoint_candidate")),
                    "right": bool(right.get("endpoint_candidate")),
                },
            },
        ))
    planned.sort(key=lambda item: tuple(item.provenance.get("priority") or ()))
    if max_pair_opportunities is not None:
        budget = max(0, int(max_pair_opportunities))
        admitted = 0
        bounded: list[PairOpportunity] = []
        for item in planned:
            if item.status != PENDING_CLASSIFICATION:
                bounded.append(item)
            elif admitted < budget:
                admitted += 1
                bounded.append(item)
            else:
                bounded.append(replace(
                    item,
                    status=DEFERRED_BUDGET,
                    planning_reason="pair_budget_exhausted",
                    provenance={**item.provenance, "pair_budget": budget},
                ))
        planned = bounded
    identities = [item.opportunity_identity for item in planned]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate opportunity identity")
    return PairOpportunityPlan(
        opportunities=tuple(planned),
        theoretical_pairs=theoretical_pairs,
        duplicate_occurrences_collapsed=duplicate_count,
    )


def opportunity_payload(item: PairOpportunity) -> dict[str, Any]:
    """Serialize one opportunity deterministically for persistence/fingerprints."""

    return json.loads(json.dumps(
        {
            **item.__dict__,
            "left_span": list(item.left_span),
            "right_span": list(item.right_span),
            "context_span": list(item.context_span),
            "allowed_predicate_metadata": list(item.allowed_predicate_metadata),
        },
        ensure_ascii=False,
        sort_keys=True,
    ))
