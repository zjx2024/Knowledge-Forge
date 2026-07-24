"""Pure Shadow v2 relation planning and trust-boundary primitives."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from typing import Any, Mapping, Sequence

from .config import Config
from .sentence_spans import sentence_index, split_sentences


PENDING_CLASSIFICATION = "PENDING_CLASSIFICATION"
DEFERRED_BUDGET = "DEFERRED_BUDGET"
REJECTED_SELF_LOOP = "REJECTED_SELF_LOOP"
TYPE_UNRESOLVED = "TYPE_UNRESOLVED"
TYPE_INCOMPATIBLE = "TYPE_INCOMPATIBLE"
FILTERED_DISTANCE = "FILTERED_DISTANCE"
NO_RELATION = "NO_RELATION"
OTHER_UNSUPPORTED = "OTHER_UNSUPPORTED"
HYPOTHESES = "HYPOTHESES"

SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION = "shadow-relation-classifier-prompt-v2"
SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION = "shadow-relation-classifier-schema-v2"
SHADOW_RELATION_CLASSIFIER_PARSER_VERSION = "shadow-relation-classifier-parser-v2"
SHADOW_RELATION_VERIFIER_PROMPT_VERSION = "shadow-relation-verifier-prompt-v2"
SHADOW_RELATION_VERIFIER_SCHEMA_VERSION = "shadow-relation-verifier-schema-v2"
SHADOW_RELATION_VERIFIER_PARSER_VERSION = "shadow-relation-verifier-parser-v2"

PASS = "PASS"
REVIEW_ONLY = "REVIEW_ONLY"
SUPPORTED = "SUPPORTED"
REJECTED = "REJECTED"
UNCERTAIN = "UNCERTAIN"
RETRYABLE_FAILED = "RETRYABLE_FAILED"

_UNKNOWN_TYPES = {"", "unknown", "genericentity", "unresolved"}
_RELATION_CUE = re.compile(
    r"\b(?:uses?|causes?|contains?|includes?|supports?|owns?|produces?|supplies?|"
    r"located|part of|depends?|works? for|is a)\b|导致|使用|包含|属于|位于|支持|"
    r"依赖|生产|供应",
    re.IGNORECASE,
)


def _digest(parts: Sequence[object]) -> str:
    material = "\x1f".join(str(value) for value in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


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


class ShadowClassificationError(ValueError):
    """A classifier batch cannot be aligned to its frozen pair universe."""


@dataclass(frozen=True)
class PredicateSpec:
    name: str
    domain: tuple[str, ...]
    range: tuple[str, ...]
    symmetric: bool
    inverse_of: str
    canonical_forward: str
    publishable: bool
    review_only: bool
    reflexive: bool
    description: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    schema_version: str


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


@dataclass(frozen=True)
class HypothesisValidation:
    hypothesis_id: str
    status: str
    reason_code: str
    predicate_metadata: dict[str, Any]


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
    error_message: str = ""


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


def _definition_types(definition: Mapping[str, Any], side: str) -> tuple[str, ...]:
    keys = (
        ("domain", "source_types", "source")
        if side == "source"
        else ("range", "target_types", "target")
    )
    for key in keys:
        value = definition.get(key)
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list):
            return tuple(str(item) for item in value)
    return ("*",)


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
    for definition in logical_schema.get("relation_types") or []:
        if not isinstance(definition, Mapping):
            continue
        name = str(definition.get("name") or "").strip()
        if not name:
            continue
        domain = _definition_types(definition, "source")
        range_ = _definition_types(definition, "target")
        direct_left = _type_match(left_type, domain)
        direct_right = _type_match(right_type, range_)
        reverse_left = _type_match(right_type, domain)
        reverse_right = _type_match(left_type, range_)

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
                "name": name,
                "domain": list(domain),
                "range": list(range_),
                "left_to_right": left_to_right,
                "right_to_left": right_to_left,
                "symmetric": bool(definition.get("symmetric")),
                "inverse_of": str(definition.get("inverse_of") or ""),
                "review_only": bool(definition.get("review_only")),
                "reflexive": bool(definition.get("reflexive")),
            })
    if any(value is True for value in observed):
        return tuple(metadata), "compatible"
    if any(value is None for value in observed):
        return tuple(metadata), "unresolved"
    return tuple(metadata), "incompatible"


def normalize_predicate_schema(
    logical_schema: Mapping[str, Any],
) -> dict[str, PredicateSpec]:
    """Normalize relation semantics and apply the Shadow-only related_to policy."""

    schema_version = str(
        logical_schema.get("version")
        or logical_schema.get("schema_version")
        or "unspecified"
    )
    result: dict[str, PredicateSpec] = {}
    for raw in logical_schema.get("relation_types") or []:
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        review_only = bool(raw.get("review_only")) or name == "related_to"
        publishable = bool(raw.get("publishable", True)) and not review_only
        result[name] = PredicateSpec(
            name=name,
            domain=_definition_types(raw, "source"),
            range=_definition_types(raw, "target"),
            symmetric=bool(raw.get("symmetric")),
            inverse_of=str(raw.get("inverse_of") or raw.get("inverse_predicate") or ""),
            canonical_forward=str(raw.get("canonical_forward") or name),
            publishable=publishable,
            review_only=review_only,
            reflexive=bool(raw.get("reflexive")),
            description=str(raw.get("description") or ""),
            positive_examples=tuple(
                str(value) for value in raw.get("positive_examples") or []
            ),
            negative_examples=tuple(
                str(value) for value in raw.get("negative_examples") or []
            ),
            schema_version=schema_version,
        )
    return result


def classifier_response_schema(
    opportunities: Sequence[PairOpportunity],
) -> dict[str, Any]:
    """Return the versioned batch schema; endpoint order is the only direction."""

    count = len(opportunities)
    return {
        "$id": SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
        "type": "array",
        "minItems": count,
        "maxItems": count,
        "items": {
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
                            "subject_occurrence_id": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object_occurrence_id": {"type": "string"},
                            "evidence_spans": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "start": {"type": "integer", "minimum": 0},
                                        "end": {"type": "integer", "minimum": 1},
                                    },
                                    "required": ["start", "end"],
                                },
                            },
                            "classifier_confidence": {
                                "type": "number", "minimum": 0, "maximum": 1,
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
        },
    }


def classifier_prompt(
    opportunities: Sequence[PairOpportunity],
    *,
    content: str,
    logical_schema: Mapping[str, Any],
) -> str:
    """Build the frozen classifier instruction with explicit hard negatives."""

    predicates = {
        name: asdict(spec)
        for name, spec in normalize_predicate_schema(logical_schema).items()
    }
    pairs = [{
        "pair_id": item.pair_identity,
        "left_occurrence_id": item.left_occurrence_key,
        "right_occurrence_id": item.right_occurrence_key,
        "left_span": list(item.left_span),
        "right_span": list(item.right_span),
        "context_span": list(item.context_span),
        "context": content[item.context_span[0]:item.context_span[1]],
        "allowed_predicates": [
            value["name"] for value in item.allowed_predicate_metadata
        ],
    } for item in opportunities]
    instruction = (
        f"contract={SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION}; "
        f"schema={SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION}. "
        "对每个 pair 恰好返回 NO_RELATION、OTHER_UNSUPPORTED 或 HYPOTHESES。"
        "subject_occurrence_id 与 object_occurrence_id 的顺序是方向唯一真相，"
        "不要输出 direction 字段。一个 pair 可返回多个具体 predicate。"
        "同一列表出现本身不等于关系；电话号码与联系人字段的结构共现不等于关系。"
        "获得供应不等于生产。A 使用 B 与 B 使用 A 不同；"
        "A 依赖 B 与 B 依赖 A 不同。Schema 缺少具体词时返回 "
        "OTHER_UNSUPPORTED，不要用 related_to 兜底；没有语义关系返回 NO_RELATION。"
        "evidence_spans 必须是冻结 context 内的绝对半开区间。"
    )
    return json.dumps(
        {
            "instruction": instruction,
            "predicates": predicates,
            "pairs": pairs,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_spans(
    raw: Any,
    *,
    opportunity: PairOpportunity,
    content: str,
) -> tuple[EvidenceSpan, ...]:
    if not isinstance(raw, list):
        raise ShadowClassificationError("evidence spans must be an array")
    parsed: list[EvidenceSpan] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ShadowClassificationError("evidence span must be an object")
        try:
            start, end = int(value["start"]), int(value["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShadowClassificationError("evidence span is invalid") from exc
        if (
            start < opportunity.context_span[0]
            or end <= start
            or end > opportunity.context_span[1]
            or end > len(content)
        ):
            raise ShadowClassificationError("evidence span is outside frozen context")
        parsed.append(EvidenceSpan(start=start, end=end))
    return tuple(parsed)


def _normalize_occurrence(
    value: Any,
    opportunity: PairOpportunity,
) -> str:
    aliases = {
        opportunity.left_occurrence_key: opportunity.left_occurrence_key,
        opportunity.right_occurrence_key: opportunity.right_occurrence_key,
        opportunity.left_mention_id: opportunity.left_occurrence_key,
        opportunity.right_mention_id: opportunity.right_occurrence_key,
    }
    normalized = aliases.get(str(value or ""))
    if normalized is None:
        raise ShadowClassificationError("hypothesis occurrence drifted from pair")
    return normalized


def parse_classifier_response(
    opportunities: Sequence[PairOpportunity],
    raw: Any,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    classifier_policy_version: str = "shadow-relation-classifier-v2",
    classifier_model_revision: str = "unspecified",
) -> tuple[PairClassificationDecision, ...]:
    """Parse a complete classifier batch and fail closed on any ambiguity."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ShadowClassificationError("classifier returned invalid JSON") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("decisions")
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise ShadowClassificationError("classifier decisions must be an array")
    expected = {item.pair_identity: item for item in opportunities}
    ids = [str(item.get("pair_id") or "") for item in raw]
    duplicates = sorted(
        pair_id for pair_id, count in {
            value: ids.count(value) for value in ids if value
        }.items() if count > 1
    )
    if duplicates:
        raise ShadowClassificationError(f"duplicate pair ids: {duplicates}")
    unknown = sorted(set(ids) - set(expected))
    if unknown:
        raise ShadowClassificationError(f"unknown pair ids: {unknown}")
    missing = sorted(set(expected) - set(ids))
    if missing:
        raise ShadowClassificationError(f"missing pair ids: {missing}")

    schema = normalize_predicate_schema(logical_schema)
    decisions: list[PairClassificationDecision] = []
    for value in raw:
        pair = expected[str(value["pair_id"])]
        decision = str(value.get("decision") or "")
        raw_hypotheses = value.get("hypotheses")
        if not isinstance(raw_hypotheses, list):
            raise ShadowClassificationError("hypotheses must be an array")
        if decision not in {NO_RELATION, OTHER_UNSUPPORTED, HYPOTHESES}:
            raise ShadowClassificationError(f"unknown classifier decision: {decision}")
        if decision in {NO_RELATION, OTHER_UNSUPPORTED} and raw_hypotheses:
            raise ShadowClassificationError("negative/schema-gap decision cannot contain hypotheses")
        if decision == HYPOTHESES and not raw_hypotheses:
            raise ShadowClassificationError("HYPOTHESES decision requires hypotheses")
        decision_spans = _parse_spans(
            value.get("evidence_spans") or [],
            opportunity=pair,
            content=content,
        )
        reason_code = str(value.get("reason_code") or "")
        if decision == OTHER_UNSUPPORTED and (
            not decision_spans or reason_code != "SCHEMA_GAP"
        ):
            raise ShadowClassificationError(
                "OTHER_UNSUPPORTED requires evidence and SCHEMA_GAP"
            )
        parsed_hypotheses: list[RelationHypothesis] = []
        pair_metadata = {
            str(item.get("name") or ""): item
            for item in pair.allowed_predicate_metadata
        }
        for hypothesis in raw_hypotheses:
            if not isinstance(hypothesis, Mapping):
                raise ShadowClassificationError("hypothesis must be an object")
            subject = _normalize_occurrence(
                hypothesis.get("subject_occurrence_id"), pair,
            )
            object_ = _normalize_occurrence(
                hypothesis.get("object_occurrence_id"), pair,
            )
            if subject == object_:
                raise ShadowClassificationError("hypothesis endpoints must be distinct")
            predicate = str(hypothesis.get("predicate") or "")
            spec = schema.get(predicate)
            metadata = pair_metadata.get(predicate)
            if spec is None or metadata is None:
                raise ShadowClassificationError(f"predicate is not allowed: {predicate}")
            left_to_right = (
                subject == pair.left_occurrence_key
                and object_ == pair.right_occurrence_key
            )
            orientation_allowed = metadata.get(
                "left_to_right" if left_to_right else "right_to_left"
            )
            if orientation_allowed is not True:
                raise ShadowClassificationError(
                    f"predicate orientation is not type-compatible: {predicate}"
                )
            legacy_direction = hypothesis.get("direction")
            if legacy_direction is not None:
                expected_direction = (
                    "symmetric" if spec.symmetric
                    else "as_is" if left_to_right else "reversed"
                )
                if str(legacy_direction) != expected_direction:
                    raise ShadowClassificationError(
                        "legacy direction conflicts with ordered endpoint roles"
                    )
            spans = _parse_spans(
                hypothesis.get("evidence_spans"),
                opportunity=pair,
                content=content,
            )
            if not spans:
                raise ShadowClassificationError("positive hypothesis requires evidence")
            try:
                confidence = float(hypothesis.get("classifier_confidence"))
            except (TypeError, ValueError) as exc:
                raise ShadowClassificationError("classifier confidence is invalid") from exc
            if not 0 <= confidence <= 1:
                raise ShadowClassificationError("classifier confidence is outside [0,1]")
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
                classifier_prompt_version=SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
                classifier_schema_version=SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
                classifier_parser_version=SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
                predicate_metadata=asdict(spec),
            ))
        decisions.append(PairClassificationDecision(
            pair_identity=pair.pair_identity,
            decision=decision,
            hypotheses=tuple(parsed_hypotheses),
            evidence_spans=decision_spans,
            reason_code=reason_code,
        ))
    return tuple(decisions)


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


def _lexical_validation_reason(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
) -> tuple[str, str] | None:
    evidence = " ".join(content[item.start:item.end] for item in hypothesis.evidence_spans)
    normalized = evidence.casefold()
    if re.search(r"\b(?:does\s+not|do\s+not|did\s+not|not|never|no longer)\b|未|并未|不再", normalized):
        return REJECTED, "NEGATED"
    if re.search(r"\b(?:might|may|could|would|if|hypothetically|plans?\s+to)\b|可能|假设|计划", normalized):
        return REVIEW_ONLY, "HYPOTHETICAL"
    if hypothesis.predicate == "produces" and re.search(
        r"\b(?:supply|supplies|supplied|supplier)\b|供应", normalized,
    ) and not re.search(r"\bproduc(?:e|es|ed|ing|tion)\b|生产", normalized):
        return REJECTED, "SUPPLY_IS_NOT_PRODUCTION"
    cue_patterns = {
        "uses": r"\buses?\b|使用",
        "depends_on": r"\bdepends?(?:_on|\s+on)\b|依赖",
    }
    cue = cue_patterns.get(hypothesis.predicate)
    if cue and re.search(cue, evidence, re.IGNORECASE):
        subject = _occurrence_details(
            opportunity, hypothesis.subject_occurrence_key, content,
        )["surface"]
        object_ = _occurrence_details(
            opportunity, hypothesis.object_occurrence_key, content,
        )["surface"]
        expected = re.compile(
            re.escape(subject) + r".*?" + cue + r".*?" + re.escape(object_),
            re.IGNORECASE | re.DOTALL,
        )
        opposite = re.compile(
            re.escape(object_) + r".*?" + cue + r".*?" + re.escape(subject),
            re.IGNORECASE | re.DOTALL,
        )
        if not expected.search(evidence) and opposite.search(evidence):
            return REJECTED, "DIRECTION_NOT_ENTAILED"
    return None


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

    def result(status: str, reason: str) -> HypothesisValidation:
        return HypothesisValidation(
            hypothesis_id=hypothesis.hypothesis_id,
            status=status,
            reason_code=reason,
            predicate_metadata=asdict(spec) if spec else empty_metadata,
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
    if (
        subject["canonical_id"]
        and subject["canonical_id"] == object_["canonical_id"]
        and not spec.reflexive
    ):
        return result(REJECTED, "ILLEGAL_SELF_LOOP")
    if spec.inverse_of and spec.inverse_of not in schema:
        return result(REJECTED, "INVERSE_METADATA_INVALID")
    if spec.canonical_forward not in schema:
        return result(REJECTED, "CANONICAL_FORWARD_INVALID")
    subject_match = _type_match(str(subject["type"]), spec.domain)
    object_match = _type_match(str(object_["type"]), spec.range)
    if subject_match is False or object_match is False:
        return result(REJECTED, "TYPE_INCOMPATIBLE")
    if subject_match is None or object_match is None:
        return result(REVIEW_ONLY, "TYPE_UNRESOLVED")
    if spec.review_only or not spec.publishable:
        return result(REVIEW_ONLY, "PREDICATE_REVIEW_ONLY")
    lexical = _lexical_validation_reason(
        hypothesis, opportunity, content=content,
    )
    if lexical:
        return result(*lexical)
    return result(PASS, "DETERMINISTIC_VALIDATION_PASSED")


def verifier_prompt(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
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
    return json.dumps(
        {
            "instruction": (
                "Use only the frozen source context. Verify the complete ordered "
                "(subject,predicate,object) fact, including predicate meaning, roles, "
                "direction, negation, modality, quotation and exact occurrence alignment. "
                "Cooccurrence and external knowledge are insufficient."
            ),
            "contract_version": SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
            "context": {
                "start": opportunity.context_span[0],
                "end": opportunity.context_span[1],
                "text": content[
                    opportunity.context_span[0]:opportunity.context_span[1]
                ],
            },
            "subject": subject,
            "predicate": asdict(spec),
            "object": object_,
            "classifier_evidence_spans": [
                asdict(value) for value in hypothesis.evidence_spans
            ],
            "hypothesis_id": hypothesis.hypothesis_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def verifier_response_schema(hypothesis_id: str) -> dict[str, Any]:
    return {
        "$id": SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "hypothesis_id": {"const": hypothesis_id},
            "verdict": {
                "type": "string",
                "enum": [SUPPORTED, REJECTED, UNCERTAIN],
            },
            "reason_code": {"type": "string", "minLength": 1},
            "supported_evidence_spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                    },
                    "required": ["start", "end"],
                },
            },
        },
        "required": [
            "hypothesis_id", "verdict", "reason_code",
            "supported_evidence_spans",
        ],
    }


def _attempt(
    hypothesis: RelationHypothesis,
    *,
    attempt_no: int,
    verdict: str,
    reason_code: str,
    supported_spans: tuple[EvidenceSpan, ...],
    independent_call: bool,
    verifier_model_revision: str,
    error_message: str = "",
) -> VerificationAttempt:
    attempt_id = "relverify_" + _digest((
        hypothesis.hypothesis_id,
        attempt_no,
        verifier_model_revision,
        SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
    ))
    return VerificationAttempt(
        attempt_id=attempt_id,
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=attempt_no,
        verdict=verdict,
        reason_code=reason_code,
        supported_evidence_spans=supported_spans,
        independent_call=independent_call,
        verifier_prompt_version=SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
        verifier_schema_version=SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
        verifier_parser_version=SHADOW_RELATION_VERIFIER_PARSER_VERSION,
        verifier_model_revision=verifier_model_revision,
        error_message=error_message,
    )


def _parse_verifier_response(
    hypothesis: RelationHypothesis,
    opportunity: PairOpportunity,
    raw: Any,
    *,
    content: str,
    attempt_no: int,
    verifier_model_revision: str,
) -> VerificationAttempt:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ShadowClassificationError("verifier returned invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ShadowClassificationError("verifier response must be an object")
    if str(raw.get("hypothesis_id") or "") != hypothesis.hypothesis_id:
        raise ShadowClassificationError("verifier hypothesis id missing or mismatched")
    verdict = str(raw.get("verdict") or "")
    if verdict not in {SUPPORTED, REJECTED, UNCERTAIN}:
        raise ShadowClassificationError("verifier verdict is invalid")
    reason_code = str(raw.get("reason_code") or "").strip()
    if not reason_code:
        raise ShadowClassificationError("verifier reason code is required")
    spans = _parse_spans(
        raw.get("supported_evidence_spans") or [],
        opportunity=opportunity,
        content=content,
    )
    if verdict == SUPPORTED:
        if not spans:
            raise ShadowClassificationError("SUPPORTED verdict requires evidence")
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
            raise ShadowClassificationError(
                "verifier evidence does not cover exact occurrences"
            )
        if any(
            not any(
                source.start <= span.start and source.end >= span.end
                for source in hypothesis.evidence_spans
            )
            for span in spans
        ):
            raise ShadowClassificationError(
                "verifier evidence escaped classifier evidence"
            )
    return _attempt(
        hypothesis,
        attempt_no=attempt_no,
        verdict=verdict,
        reason_code=reason_code,
        supported_spans=spans,
        independent_call=True,
        verifier_model_revision=verifier_model_revision,
    )


def verify_hypothesis(
    hypothesis: RelationHypothesis,
    validation: HypothesisValidation,
    opportunity: PairOpportunity,
    *,
    content: str,
    logical_schema: Mapping[str, Any],
    verifier: Any,
    verifier_model_revision: str,
    attempt_no: int = 1,
) -> VerificationAttempt:
    """Run one independent verifier call, or record why it was not callable."""

    if validation.hypothesis_id != hypothesis.hypothesis_id:
        raise ValueError("validation does not belong to hypothesis")
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
    try:
        raw = verifier(
            verifier_prompt(
                hypothesis, opportunity,
                content=content, logical_schema=logical_schema,
            ),
            verifier_response_schema(hypothesis.hypothesis_id),
        )
        return _parse_verifier_response(
            hypothesis,
            opportunity,
            raw,
            content=content,
            attempt_no=attempt_no,
            verifier_model_revision=verifier_model_revision,
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
    predicate = spec.canonical_forward
    if predicate != spec.name:
        if spec.inverse_of != predicate:
            raise ValueError("canonical-forward requires matching inverse metadata")
        subject, object_ = object_, subject
    if spec.symmetric:
        subject, object_ = sorted((subject, object_))
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
    evidence_map: dict[str, RelationFactEvidence] = {}
    collapse_audit: list[dict[str, Any]] = []
    conflicts: dict[str, RelationFactConflict] = {}
    illegal_self_loops = 0
    symmetric_inverse_collapses = 0
    exact_fact_collapses = exact_input_duplicates
    normalized_by_hypothesis: dict[str, tuple[str, str, str]] = {}

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
        try:
            subject, predicate, object_ = _canonicalized_fact_parts(
                hypothesis=hypothesis,
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
        semantic_key = "relfact_" + _digest((
            int(project_id), subject, predicate, object_,
        ))
        normalized_by_hypothesis[hypothesis_id] = (
            subject, predicate, object_,
        )
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
                review_only=spec.review_only or not spec.publishable,
                normalization_eligible=spec.publishable and not spec.review_only,
                evidence_count=0,
            )
            normalized_mode_by_fact[semantic_key] = bool(
                spec.symmetric or predicate != hypothesis.predicate
            )
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
            hypothesis.subject_occurrence_key,
            hypothesis.predicate,
            hypothesis.object_occurrence_key,
            spans_material,
            hypothesis.hypothesis_id,
        ))
        if evidence_key not in evidence_map:
            evidence_map[evidence_key] = RelationFactEvidence(
                evidence_id=evidence_key,
                evidence_key=evidence_key,
                fact_id=semantic_key,
                document_revision_id=opportunity.document_revision_id,
                subject_occurrence_key=hypothesis.subject_occurrence_key,
                object_occurrence_key=hypothesis.object_occurrence_key,
                evidence_spans=attempt.supported_evidence_spans,
                hypothesis_id=hypothesis.hypothesis_id,
                verification_attempt_id=attempt.attempt_id,
                verifier_supported=True,
                provenance={
                    "pair_identity": hypothesis.pair_identity,
                    "classifier_policy_version": hypothesis.classifier_policy_version,
                    "classifier_model_revision": hypothesis.classifier_model_revision,
                    "verifier_model_revision": attempt.verifier_model_revision,
                },
            )

    # A rejected reverse hypothesis remains visible as a conflict, but never a fact.
    hypothesis_values = list(unique_hypotheses.values())
    unsupported_reverse_conflicts = 0
    for rejected_hypothesis in hypothesis_values:
        if rejected_hypothesis.hypothesis_id in accepted:
            continue
        rejected_attempts = attempts_by_hypothesis.get(
            rejected_hypothesis.hypothesis_id, {},
        ).values()
        if not any(item.verdict in {REJECTED, UNCERTAIN, RETRYABLE_FAILED}
                   for item in rejected_attempts):
            continue
        for supported_hypothesis in hypothesis_values:
            if supported_hypothesis.hypothesis_id not in accepted:
                continue
            if (
                supported_hypothesis.pair_identity
                == rejected_hypothesis.pair_identity
                and supported_hypothesis.predicate == rejected_hypothesis.predicate
                and supported_hypothesis.subject_occurrence_key
                == rejected_hypothesis.object_occurrence_key
                and supported_hypothesis.object_occurrence_key
                == rejected_hypothesis.subject_occurrence_key
            ):
                spec = schema.get(rejected_hypothesis.predicate)
                if spec and not spec.symmetric and not spec.inverse_of:
                    conflict_id = "relconflict_" + _digest((
                        rejected_hypothesis.hypothesis_id,
                        "UNSUPPORTED_REVERSE_CONFLICT",
                    ))
                    conflicts[conflict_id] = RelationFactConflict(
                        conflict_id=conflict_id,
                        pair_identity=rejected_hypothesis.pair_identity,
                        hypothesis_id=rejected_hypothesis.hypothesis_id,
                        reason_code="UNSUPPORTED_REVERSE_CONFLICT",
                        metadata={
                            "supported_hypothesis_id":
                                supported_hypothesis.hypothesis_id,
                        },
                    )
                    unsupported_reverse_conflicts += 1
                break

    evidence_counts: dict[str, int] = {}
    for value in evidence_map.values():
        evidence_counts[value.fact_id] = evidence_counts.get(value.fact_id, 0) + 1
    fact_map = {
        key: replace(value, evidence_count=evidence_counts.get(value.fact_id, 0))
        for key, value in fact_map.items()
    }

    reciprocal_fact_keys: set[str] = set()
    facts = list(fact_map.values())
    for fact in facts:
        if fact.symmetric:
            continue
        reverse = next((
            candidate for candidate in facts
            if candidate.predicate == fact.predicate
            and candidate.subject_canonical_id == fact.object_canonical_id
            and candidate.object_canonical_id == fact.subject_canonical_id
        ), None)
        if reverse:
            reciprocal_fact_keys.update((fact.fact_id, reverse.fact_id))
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
            "illegal_self_loops": illegal_self_loops,
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
        "verifier": (
            "prompt_version", "schema_version", "parser_version",
            "model", "model_revision",
        ),
        "normalization": ("version", "symmetric_rule", "inverse_rule"),
        "publish_policy": ("version", "related_to", "unsupported"),
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
        "version": "shadow-relation-materialization-fingerprint-v2",
        "component_hashes": component_hashes,
    })
    return RelationMaterializationFingerprint(
        version="shadow-relation-materialization-fingerprint-v2",
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
    """Apply the final Shadow boundary without performing any product write."""

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
        or fact.predicate in {
            "related_to", OTHER_UNSUPPORTED, TYPE_UNRESOLVED, UNCERTAIN,
        }
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
        if existing is not None and existing != normalized:
            raise ValueError(f"occurrence key collision: {key}")
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
    sentences = split_sentences(content)
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
        cue_match = _RELATION_CUE.search(context_text)
        cue = cue_match.group(0) if cue_match else ""
        predicate_metadata, type_status = _predicate_metadata(
            str(left.get("type") or ""),
            str(right.get("type") or ""),
            logical_schema,
        )
        status = PENDING_CLASSIFICATION
        reason = "local_pair_with_relation_cue" if cue else "local_pair_without_explicit_cue"
        if (
            str(left.get("canonical_id") or "")
            and str(left.get("canonical_id")) == str(right.get("canonical_id") or "")
        ):
            status, reason = REJECTED_SELF_LOOP, "same_canonical_assignment"
        elif sentence_distance > sentence_window or centre_distance > max_char_distance:
            status, reason = FILTERED_DISTANCE, "outside_local_context_window"
        elif type_status == "incompatible":
            status, reason = TYPE_INCOMPATIBLE, "known_schema_type_incompatibility"
        elif type_status == "unresolved":
            status, reason = TYPE_UNRESOLVED, "unknown_endpoint_type"
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
            allowed_predicate_metadata=predicate_metadata,
            planning_reason=reason,
            cue=cue,
            type_status=type_status,
            status=status,
            provenance={
                "sentence_distance": sentence_distance,
                "character_distance": centre_distance,
            },
        ))
    planned.sort(key=lambda item: item.pair_identity)
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
