"""Pure Shadow v2 relation planning and trust-boundary primitives."""

from __future__ import annotations

import hashlib
import json
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
FILTERED_STRUCTURE_PEER = "FILTERED_STRUCTURE_PEER"
NO_RELATION = "NO_RELATION"
OTHER_UNSUPPORTED = "OTHER_UNSUPPORTED"
HYPOTHESES = "HYPOTHESES"
EXPLICIT_NO_RELATION = "EXPLICIT_NO_RELATION"
SCHEMA_GAP = "SCHEMA_GAP"
HYPOTHESES_EMITTED = "HYPOTHESES_EMITTED"
CLASSIFICATION_PENDING = "CLASSIFICATION_PENDING"
CLASSIFICATION_FAILED = "CLASSIFICATION_FAILED"

SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION = "shadow-relation-classifier-prompt-v9"
SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION = "shadow-relation-classifier-schema-v6"
SHADOW_RELATION_CLASSIFIER_PARSER_VERSION = "shadow-relation-classifier-parser-v9"
SHADOW_RELATION_PROPOSER_PROMPT_VERSION = "shadow-relation-proposer-prompt-v1"
SHADOW_RELATION_PROPOSER_SCHEMA_VERSION = "shadow-relation-proposer-schema-v1"
SHADOW_RELATION_PROPOSER_PARSER_VERSION = "shadow-relation-proposer-parser-v1"
SHADOW_RELATION_VERIFIER_PROMPT_VERSION = "shadow-relation-verifier-prompt-v5"
SHADOW_RELATION_VERIFIER_SCHEMA_VERSION = "shadow-relation-verifier-schema-v4"
SHADOW_RELATION_VERIFIER_PARSER_VERSION = "shadow-relation-verifier-parser-v4"

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
    required_evidence_roles: tuple[str, ...]
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    positive_cues: tuple[str, ...]
    negative_cues: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    optional_soft_cues: tuple[str, ...]
    qualifier_schema: dict[str, Any]
    identity_qualifier_keys: tuple[str, ...]
    conflicts_with: tuple[str, ...]
    negation_policy: str
    modality_policy: str
    schema_version: str

    @property
    def review_only(self) -> bool:
        return self.publish_mode != PUBLISH_AUTO

    @property
    def publishable(self) -> bool:
        return self.publish_mode == PUBLISH_AUTO


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
        raise ValueError("Shadow v2 Schema requires a version")
    relation_types = logical_schema.get("relation_types")
    if not isinstance(relation_types, list) or not relation_types:
        raise ValueError("Shadow v2 Schema requires relation_types")
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
            "identity_qualifier_keys", "conflicts_with",
        ):
            value = raw.get(field_name, [])
            if (
                not isinstance(value, list)
                or not all(isinstance(item, str) for item in value)
            ):
                raise ValueError(
                    f"predicate {name} {field_name} must be a string array"
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
            qualifier_schema=dict(qualifier_schema),
            identity_qualifier_keys=tuple(
                str(value) for value in raw.get("identity_qualifier_keys") or []
            ),
            conflicts_with=tuple(
                str(value) for value in raw.get("conflicts_with") or []
            ),
            negation_policy=str(
                raw.get("negation_policy") or "reject_negated"
            ),
            modality_policy=str(
                raw.get("modality_policy") or "require_asserted"
            ),
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
                        "predicate": {"type": "string"},
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
            "$id": SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
            **decision_schema,
        }
    return {
        "$id": SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
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
) -> str:
    """Build a predicate-agnostic classifier request from versioned metadata."""

    predicates = {
        name: asdict(spec)
        for name, spec in normalize_predicate_schema(logical_schema).items()
    }
    pairs = [{
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
        "context": content[item.context_span[0]:item.context_span[1]],
        "allowed_predicates": [
            value["name"] for value in item.allowed_predicate_metadata
        ],
        "soft_cue_matches": list(
            item.provenance.get("soft_cue_matches") or []
        ),
    } for item in opportunities]
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
        f"contract={SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION}; "
        f"schema={SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION}. "
        "对每个 pair 恰好返回 NO_RELATION、OTHER_UNSUPPORTED 或 HYPOTHESES。"
        "subject_occurrence_id 与 object_occurrence_id 的顺序是方向唯一真相，"
        "且这两个字段只能逐字使用 LEFT 或 RIGHT；不要复制内部 occurrence "
        "哈希，不要输出 direction 字段。一个 pair 可返回多个具体 predicate。"
        "只按每个 predicate 的 semantic_definition、direction_description、"
        "domain_types、range_types 和正反例判断语义及方向。"
        "optional_soft_cues 只是可配置提示，不能作为关系、方向或发布的证据。"
        "结构共现本身不等于关系；Schema 无法表示但文本明确表达关系时返回 "
        "OTHER_UNSUPPORTED；没有语义关系返回 NO_RELATION。"
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
            "predicates": predicates,
            "pairs": pairs,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def proposer_response_schema() -> dict[str, Any]:
    return {
        "$id": SHADOW_RELATION_PROPOSER_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposals": {
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
                        "qualifiers": {"type": "array"},
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
            },
            "missing_endpoint_mentions": {"type": "array"},
            "coverage": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "complete": {"type": "boolean"},
                    "uncovered_cues": {"type": "array"},
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
) -> str:
    predicates = {
        name: asdict(spec)
        for name, spec in normalize_predicate_schema(logical_schema).items()
    }
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
    return json.dumps(
        {
            "instruction": (
                f"contract={SHADOW_RELATION_PROPOSER_PROMPT_VERSION}; "
                f"schema={SHADOW_RELATION_PROPOSER_SCHEMA_VERSION}. "
                "Directly propose only explicit relations supported by this frozen "
                "ContextUnit. subject_occurrence_id and object_occurrence_id must be "
                "copied from occurrences; predicate must be copied from predicates. "
                "Do not invent endpoints, predicates, evidence spans, qualifiers, or "
                "canonical entities. Empty proposals is valid when no explicit relation "
                "is present. If a needed endpoint is missing, report it only in "
                "missing_endpoint_mentions; it must not create a proposal. "
                "optional_soft_cues are hints only and never decide publication. "
                "Every proposal still requires independent verification."
            ),
            "context": {
                "context_id": context_id,
                "text": content,
                "start": 0,
                "end": len(content),
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


def _parse_proposer_spans(
    raw: Any,
    *,
    content: str,
    endpoint_spans: Sequence[tuple[int, int]],
) -> tuple[EvidenceSpan, ...]:
    if not isinstance(raw, list):
        raise ShadowClassificationError("proposal evidence_spans must be an array")
    spans: list[EvidenceSpan] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ShadowClassificationError("proposal evidence span must be an object")
        try:
            start, end = int(value["start"]), int(value["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ShadowClassificationError("proposal evidence span is invalid") from exc
        if start < 0 or end <= start or end > len(content):
            raise ShadowClassificationError("proposal evidence span is outside context")
        spans.append(EvidenceSpan(start=start, end=end))
    if not spans:
        raise ShadowClassificationError("proposal requires evidence")
    for required_span in endpoint_spans:
        if not _spans_cover(spans, required_span):
            raise ShadowClassificationError("proposal evidence does not cover endpoints")
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
    proposer_policy_version: str = SHADOW_RELATION_PROPOSER_PROMPT_VERSION,
    proposer_model_revision: str = "unspecified",
) -> RelationProposalBatch:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ShadowClassificationError("proposer returned invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ShadowClassificationError("proposer response must be an object")
    proposals_raw = raw.get("proposals")
    missing_raw = raw.get("missing_endpoint_mentions")
    coverage = raw.get("coverage")
    if not isinstance(proposals_raw, list):
        raise ShadowClassificationError("proposer proposals must be an array")
    if not isinstance(missing_raw, list):
        raise ShadowClassificationError("missing_endpoint_mentions must be an array")
    if not isinstance(coverage, Mapping):
        raise ShadowClassificationError("proposer coverage must be an object")
    mentions_by_key = {
        str(item.get("occurrence_key") or ""): dict(item) for item in mentions
    }
    if any(not key for key in mentions_by_key):
        raise ShadowClassificationError("proposer input occurrence id is empty")
    schema = normalize_predicate_schema(logical_schema)
    proposals: list[RelationProposal] = []
    seen: set[str] = set()
    for raw_proposal in proposals_raw:
        if not isinstance(raw_proposal, Mapping):
            raise ShadowClassificationError("proposal must be an object")
        subject = str(raw_proposal.get("subject_occurrence_id") or "")
        object_ = str(raw_proposal.get("object_occurrence_id") or "")
        if subject not in mentions_by_key or object_ not in mentions_by_key:
            raise ShadowClassificationError("proposal endpoint is not in occurrence universe")
        if subject == object_:
            raise ShadowClassificationError("proposal endpoints must be distinct")
        predicate = str(raw_proposal.get("predicate") or "")
        if predicate not in schema:
            raise ShadowClassificationError("proposal predicate is not in schema")
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
        )
        qualifiers = raw_proposal.get("qualifiers")
        if not isinstance(qualifiers, list):
            raise ShadowClassificationError("proposal qualifiers must be an array")
        try:
            confidence = float(raw_proposal.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ShadowClassificationError("proposal confidence is invalid") from exc
        if not 0 <= confidence <= 1:
            raise ShadowClassificationError("proposal confidence is outside [0,1]")
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
        raise ShadowClassificationError("proposal pair does not match opportunity")
    spec = normalize_predicate_schema(logical_schema).get(proposal.predicate)
    if spec is None:
        raise ShadowClassificationError("proposal predicate is no longer in schema")
    spans_material = json.dumps(
        [[span.start, span.end] for span in proposal.evidence_spans],
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
        evidence_spans=proposal.evidence_spans,
        classifier_confidence=proposal.confidence,
        classifier_policy_version=proposal.proposer_policy_version,
        classifier_model_revision=proposal.proposer_model_revision,
        classifier_prompt_version=SHADOW_RELATION_PROPOSER_PROMPT_VERSION,
        classifier_schema_version=SHADOW_RELATION_PROPOSER_SCHEMA_VERSION,
        classifier_parser_version=SHADOW_RELATION_PROPOSER_PARSER_VERSION,
        predicate_metadata=asdict(spec),
        status="PROPOSED",
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
        "LEFT": opportunity.left_occurrence_key,
        "RIGHT": opportunity.right_occurrence_key,
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
            raise ShadowClassificationError(
                "NO_RELATION requires evidence spans covering the pair context"
            )
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
        if decision == HYPOTHESES:
            unsupported_schema_predicate = False
            for hypothesis in raw_hypotheses:
                if not isinstance(hypothesis, Mapping):
                    raise ShadowClassificationError("hypothesis must be an object")
                predicate = str(hypothesis.get("predicate") or "")
                if predicate not in schema or predicate not in pair_metadata:
                    unsupported_schema_predicate = True
                    break
            if unsupported_schema_predicate:
                unsupported_spans = decision_spans or _parse_spans(
                    [{
                        "start": pair.context_span[0],
                        "end": pair.context_span[1],
                    }],
                    opportunity=pair,
                    content=content,
                )
                decisions.append(PairClassificationDecision(
                    pair_identity=pair.pair_identity,
                    decision=OTHER_UNSUPPORTED,
                    hypotheses=(),
                    evidence_spans=unsupported_spans,
                    reason_code="SCHEMA_GAP",
                ))
                continue
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
    if spec.inverse_predicate and spec.inverse_predicate not in schema:
        return result(REJECTED, "INVERSE_METADATA_INVALID")
    subject_match = _type_match(
        str(subject["type"]), spec.domain_types,
    )
    object_match = _type_match(
        str(object_["type"]), spec.range_types,
    )
    if subject_match is False or object_match is False:
        return result(REJECTED, "TYPE_INCOMPATIBLE")
    if subject_match is None or object_match is None:
        return result(REVIEW_ONLY, "TYPE_UNRESOLVED")
    if spec.publish_mode == PUBLISH_REVIEW_ONLY:
        return result(REVIEW_ONLY, "PUBLISH_MODE_REVIEW_ONLY")
    if spec.publish_mode == PUBLISH_DISABLED:
        return result(REVIEW_ONLY, "PUBLISH_MODE_DISABLED")
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
                "Cooccurrence and external knowledge are insufficient. The caller binds "
                "this independent response to the supplied hypothesis_id; do not emit "
                "or invent a correlation identifier. Return exactly one JSON object "
                "with only verdict, reason_code, and supported_evidence_spans. verdict "
                "must be SUPPORTED, REJECTED, or UNCERTAIN. reason_code must be a short "
                "non-empty code. supported_evidence_spans must contain absolute "
                "half-open {start,end} spans only when verdict is SUPPORTED, and must "
                "be an empty array otherwise."
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
            "verdict", "reason_code", "supported_evidence_spans",
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
    response_hypothesis_id = raw.get("hypothesis_id")
    if (
        response_hypothesis_id is not None
        and str(response_hypothesis_id) != hypothesis.hypothesis_id
    ):
        raise ShadowClassificationError("verifier hypothesis id mismatched")
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
                review_only=spec.publish_mode != PUBLISH_AUTO,
                normalization_eligible=spec.publish_mode == PUBLISH_AUTO,
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
                if (
                    spec
                    and not spec.symmetric
                    and not spec.inverse_predicate
                ):
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
        predicate_metadata, type_status = _predicate_metadata(
            str(left.get("type") or ""),
            str(right.get("type") or ""),
            logical_schema,
        )
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
        same_sentence = sentence_distance == 0
        same_paragraph_with_schema_cue = False
        structural_admission_reason = ""
        if same_cell:
            structural_admission_reason = "same_cell"
        elif same_table_row:
            structural_admission_reason = "same_table_row"
        elif same_sentence:
            structural_admission_reason = "same_sentence"
        elif same_paragraph_with_schema_cue:
            structural_admission_reason = "same_paragraph_with_schema_cue"
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
        cue = str(soft_cue_matches[0]["cue"]) if soft_cue_matches else ""
        if (
            not structural_admission_reason
            and sentence_distance <= sentence_window
            and cue
        ):
            same_paragraph_with_schema_cue = True
            structural_admission_reason = "same_paragraph_with_schema_cue"
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
            status, reason = TYPE_INCOMPATIBLE, "known_schema_type_incompatibility"
        elif type_status == "unresolved":
            status, reason = TYPE_UNRESOLVED, "unknown_endpoint_type"
        elif (
            not structural_admission_reason
            and (
                sentence_distance > sentence_window
                or centre_distance > max_char_distance
            )
        ):
            status, reason = FILTERED_DISTANCE, "outside_local_context_window"
        structure_rank = {
            "same_cell": 0,
            "same_table_row": 1,
            "same_sentence": 2,
            "same_paragraph_with_schema_cue": 3,
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
            allowed_predicate_metadata=predicate_metadata,
            planning_reason=reason,
            cue=cue,
            type_status=type_status,
            status=status,
            provenance={
                "sentence_distance": sentence_distance,
                "character_distance": centre_distance,
                "structure_admission": structural_admission_reason,
                "priority": list(priority),
                "soft_cue_matches": soft_cue_matches,
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
