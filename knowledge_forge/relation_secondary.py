"""Reason-routed secondary relation recovery for Enhanced Shadow.

The secondary pass is deliberately narrower than the primary classifier.  It
does not re-run the same prompt with a new temperature; it asks a different
question based on the failure axis and returns only a candidate.  The runtime
still sends that candidate through the ordinary deterministic validation and
independent ordered verifier before normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SECONDARY_PASS_VERSION = "knowledge-forge-secondary-relation-review-v11-ordered-direction-repair"
SECONDARY_PROMPT_VERSION = "knowledge-forge-secondary-evidence-recovery-v11-ordered-direction-repair"
SECONDARY_SCHEMA_VERSION = "knowledge-forge-secondary-evidence-recovery-schema-v2"
SECONDARY_PARSER_VERSION = "knowledge-forge-secondary-evidence-recovery-parser-v6-ordered-direction-repair"

SECONDARY_POLARITY_PROMPT_VERSION = (
    "knowledge-forge-source-frame-polarity-prompt-v1"
)
SECONDARY_POLARITY_SCHEMA_VERSION = (
    "knowledge-forge-source-frame-polarity-schema-v1"
)
SECONDARY_POLARITY_PARSER_VERSION = (
    "knowledge-forge-source-frame-polarity-parser-v1"
)

ROUTE_EVIDENCE_RECOVERY = "EVIDENCE_RECOVERY"
ROUTE_POLARITY_REVIEW = "POLARITY_REVIEW"
ROUTE_STRUCTURAL_REVIEW = "STRUCTURAL_REVIEW"
ROUTE_PREDICATE_REFINEMENT = "PREDICATE_REFINEMENT"
ROUTE_TYPE_ROLE_REVIEW = "TYPE_ROLE_REVIEW"
ROUTE_ROLE_RETRY = "ROLE_RETRY"
ROUTE_TECHNICAL_RETRY = "TECHNICAL_RETRY"
ROUTE_NONE = "NO_SECONDARY_ROUTE"

SECONDARY_RELATION = "RELATION"
SECONDARY_NO_RELATION = "NO_RELATION"
SECONDARY_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
NO_SPECIFIC_RELATION = "NO_SPECIFIC_RELATION"


@dataclass(frozen=True)
class SecondaryRoute:
    route: str
    reason_code: str
    rationale: str


@dataclass(frozen=True)
class SecondaryDecision:
    decision: str
    predicate: str
    subject_side: str
    object_side: str
    evidence_spans: tuple[tuple[int, int], ...]
    reason_code: str


def route_secondary_reason(
    reason_code: str,
    *,
    predicate: str = "",
    validation_status: str = "",
) -> SecondaryRoute:
    """Map a terminal uncertainty to one distinct, bounded secondary action.

    The mapping is intentionally based on protocol/semantic state, not on a
    document title, language, domain term, or a concrete predicate name.
    """

    reason = str(reason_code or "").strip().upper()
    if reason in {
        "ATOMIC_FACT_INVENTORY_UNAVAILABLE",
        "ATOMIC_FACT_INVENTORY_INCOMPLETE",
        "NO_DIRECT_ATOMIC_PAIR",
        "INSUFFICIENT_EVIDENCE",
        "INCOMPLETE_ENDPOINT",
    }:
        return SecondaryRoute(
            ROUTE_EVIDENCE_RECOVERY, reason,
            "原子事实盘点失败、覆盖不完整或漏识别端点对，改用端点聚焦的证据恢复",
        )
    if reason == "PUBLISH_MODE_REVIEW_ONLY":
        return SecondaryRoute(
            ROUTE_PREDICATE_REFINEMENT, reason,
            "当前候选只有泛化或人工复核语义，改为从 Schema 允许谓词中细化",
        )
    if reason in {
        "ORDERED_ROLE_NOT_ENTAILED",
        "DIRECTION_NOT_EXCLUSIVE",
        "REVERSE_ONLY_CLAIM_REQUIRES_OWN_CANDIDATE",
        "SUBJECT_ROLE_MISMATCH",
        "OBJECT_ROLE_MISMATCH",
        "DIRECTION_MISMATCH",
    }:
        # A role failure is not evidence that the predicate itself is wrong.
        # The first verifier may have rejected a valid directed fact because
        # the candidate orientation was wrong or because its auxiliary role
        # audit was incomplete.  Secondary review must therefore keep the
        # complete allowed predicate set and independently re-bind both
        # endpoints; excluding the primary predicate would turn a direction
        # error into a guaranteed false negative.
        return SecondaryRoute(
            ROUTE_ROLE_RETRY, reason,
            "一次候选的有序端点角色或方向未通过，保留 Schema 候选集并独立重判端点绑定",
        )
    if reason in {
        "ATOMIC_FRAME_RELATION_MISMATCH",
        "RELATION_SEMANTICS_MISMATCH",
    }:
        return SecondaryRoute(
            ROUTE_PREDICATE_REFINEMENT, reason,
            "一次候选的有序角色或语义框架不成立，排除该候选后竞争其他允许谓词",
        )
    if reason == "TYPE_UNRESOLVED":
        return SecondaryRoute(
            ROUTE_TYPE_ROLE_REVIEW, reason,
            "端点类型不足以通过一次规则，二次只复核端点角色和可表达谓词",
        )
    if reason == "AUXILIARY_ROLE_AUDIT_UNAVAILABLE":
        return SecondaryRoute(
            ROUTE_ROLE_RETRY, reason,
            "角色审计技术不可用，独立重试角色与有序命题核验",
        )
    if reason == "MODEL_REJECTED_REASON_OMITTED":
        return SecondaryRoute(
            ROUTE_ROLE_RETRY, reason,
            "一次模型返回拒绝但缺少可审计原因，重新独立判断端点角色和有序命题",
        )
    if reason in {
        "RETRYABLE_FAILED",
        "VERIFIER_AUXILIARY_DISAGREEMENT_RETRY",
        "CLASSIFIER_TERMINAL_PROTOCOL_FAILURE",
        "CLASSIFIER_TERMINAL_TIMEOUT",
        "CLASSIFIER_TERMINAL_BUDGET_FAILURE",
    }:
        return SecondaryRoute(
            ROUTE_TECHNICAL_RETRY, reason,
            "一次流程发生可重试的协议或模型故障，二次只做独立技术重试",
        )
    # Explicit co-occurrence, scope denial, incompatible types, ordinary
    # list/table adjacency and unknown semantic verdicts are not uncertainty
    # recovery candidates.  They remain negative/audit outcomes.
    return SecondaryRoute(ROUTE_NONE, reason, "没有满足二次恢复条件的终态原因")


def route_secondary_attempt(
    reason_code: str,
    *,
    verdict: str = "",
    forward_entailment: str = "",
    reverse_entailment: str = "",
    predicate: str = "",
    validation_status: str = "",
) -> SecondaryRoute:
    """Route with the verifier's typed ordered-entailment audit available.

    Free-form model reasons are not a reliable way to recognize a direction
    error.  When a terminal rejection explicitly says the frozen forward
    claim is not entailed and the endpoint-swapped claim is entailed, route a
    distinct orientation-only recovery even if the model used an unfamiliar
    reason string.  The signal can only propose a swapped candidate; a fresh
    ordinary validator and independent verifier still gate publication.
    """

    route = route_secondary_reason(
        reason_code,
        predicate=predicate,
        validation_status=validation_status,
    )
    if (
        str(verdict or "").strip().upper() == "REJECTED"
        and str(forward_entailment or "").strip().upper()
        == "NOT_ENTAILED"
        and str(reverse_entailment or "").strip().upper() == "ENTAILED"
    ):
        return SecondaryRoute(
            ROUTE_ROLE_RETRY,
            "REVERSE_ORIENTATION_ENTAILED",
            "一次终审明确支持交换端点后的方向，二次冻结谓词并独立重绑定端点",
        )
    return route


def source_frame_polarity_response_schema() -> dict[str, Any]:
    """Return the deliberately tiny grammatical-scope response contract."""

    return {
        "$id": SECONDARY_POLARITY_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "result": {
                "type": "string",
                "enum": ["ASSERTED", "NEGATED", "UNCERTAIN"],
            },
        },
        "required": ["result"],
    }


def source_frame_polarity_prompt(
    *,
    content: str,
    context_span: tuple[int, int],
    frame_span: tuple[int, int],
    frame_text: str,
    open_relation: str,
) -> str:
    """Ask only whether source negation governs one exact open frame."""

    context_start, context_end = map(int, context_span)
    frame_start, frame_end = map(int, frame_span)
    if not (
        0 <= context_start <= frame_start < frame_end <= context_end
        <= len(content)
    ):
        raise ValueError("source-frame polarity spans are invalid")
    exact_text = str(frame_text or "")
    if not exact_text or content[frame_start:frame_end] != exact_text:
        raise ValueError("source-frame polarity text does not match source")
    relation = str(open_relation or "").strip()
    if not relation:
        raise ValueError("source-frame polarity relation is required")
    return json.dumps(
        {
            "instruction": (
                "Decide only the grammatical assertion polarity of open_relation "
                "inside exact_frame. Return ASSERTED when that frame is positively "
                "asserted, even if a matrix, sibling, or nearby predicate is negated. "
                "Return NEGATED only when source negation actually governs this "
                "open_relation. Return UNCERTAIN for conditional, hypothetical, "
                "quoted-without-commitment, or genuinely ambiguous scope. Do not judge "
                "a logical Schema relation, truth in the world, endpoint roles, or "
                "direction. Return only the one-field JSON object."
            ),
            "contract_version": SECONDARY_POLARITY_PROMPT_VERSION,
            "source_context": {
                "start": context_start,
                "end": context_end,
                "text": content[context_start:context_end],
            },
            "exact_frame": {
                "start": frame_start,
                "end": frame_end,
                "text": exact_text,
                "open_relation": relation,
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_source_frame_polarity(raw: Any) -> str:
    """Parse a polarity audit without accepting prose or semantic aliases."""

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("source-frame polarity returned invalid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("source-frame polarity response must be an object")
    result = str(raw.get("result") or raw.get("polarity") or "").strip().upper()
    if result not in {"ASSERTED", "NEGATED", "UNCERTAIN"}:
        raise ValueError("source-frame polarity result is invalid")
    if raw.get("result") and raw.get("polarity"):
        if str(raw["result"]).strip().upper() != str(raw["polarity"]).strip().upper():
            raise ValueError("source-frame polarity fields conflict")
    return result


def secondary_response_schema(allowed_predicates: Sequence[str]) -> dict[str, Any]:
    names = sorted({str(value).strip() for value in allowed_predicates if str(value).strip()})
    predicate_schema: dict[str, Any] = {
        "type": "string",
        "enum": [*names, NO_SPECIFIC_RELATION],
    }
    return {
        "$id": SECONDARY_SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {
                "type": "string",
                "enum": [SECONDARY_RELATION, SECONDARY_NO_RELATION, SECONDARY_INSUFFICIENT],
            },
            "predicate": predicate_schema,
            "subject_side": {"type": "string", "enum": ["LEFT", "RIGHT", ""]},
            "object_side": {"type": "string", "enum": ["LEFT", "RIGHT", ""]},
            "evidence_spans": {
                "type": "array",
                "maxItems": 4,
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
            "reason_code": {"type": "string", "minLength": 1, "maxLength": 160},
        },
        "required": [
            "decision", "predicate", "subject_side", "object_side",
            "evidence_spans", "reason_code",
        ],
    }


def secondary_recovery_prompt(
    *,
    route: SecondaryRoute,
    content: str,
    context_span: tuple[int, int],
    left_span: tuple[int, int],
    right_span: tuple[int, int],
    left_surface: str,
    right_surface: str,
    left_type: str,
    right_type: str,
    allowed_predicate_metadata: Sequence[Mapping[str, Any]],
    structural_contract: Mapping[str, Any],
    orientation_only: bool = False,
    excluded_predicate: str = "",
    preferred_predicate: str = "",
) -> str:
    """Build a target-focused prompt that withholds the first verdict."""

    context_text = content[context_span[0]:context_span[1]]
    allowed = []
    for raw in allowed_predicate_metadata:
        if not isinstance(raw, Mapping):
            continue
        allowed.append({
            key: raw.get(key)
            for key in (
                "name", "semantic_definition", "direction_description",
                "domain_types", "range_types", "subject_role_description",
                "object_role_description", "publish_mode", "positive_examples",
                "negative_examples", "allowed_source_roles", "scope_policy",
            )
            if key in raw
        })
    excluded_instruction = (
        f"一次候选谓词 {excluded_predicate!r} 已被排除；本次不能重复输出该谓词，"
        "如果其他允许谓词都不成立则返回 NO_RELATION 或 INSUFFICIENT_EVIDENCE。"
        if excluded_predicate else ""
    )
    orientation_instruction = (
        "本次原因是角色/方向审计失败，必须重新判断 subject_side 与 object_side；"
        "不得复制一次候选的端点顺序。先从原文确定谁承担 Schema 的 source role、谁承担 "
        "target role，再按该语义方向输出。若原文只支持交换后的方向，就输出交换后的 "
        "LEFT/RIGHT；若两端角色仍不确定，返回 INSUFFICIENT_EVIDENCE。"
        if orientation_only else ""
    )
    preferred_instruction = (
        f"一次触发二次复核的候选谓词是 {preferred_predicate!r}。当前是角色/方向重试，"
        "谓词已由路由冻结；必须只审计该谓词在两个精确端点上的语义和方向，输出时必须"
        f"仍使用 {preferred_predicate!r}。不得因为上下文出现普通系动词、包含词或其他弱线索"
        "改换谓词；如果该冻结谓词不成立，返回 NO_RELATION 或 INSUFFICIENT_EVIDENCE。"
        if preferred_predicate else ""
    )
    instruction = (
        "这是关系二次复核，不是一次分类器重跑。只依据冻结 source_context，"
        "先判断两个精确端点之间是否存在原文直接断言，再从允许的 Schema 谓词中选择"
        "最具体且方向正确的一个。一次流程的历史判定不会提供给你，不能为确认历史结论"
        "而补造关系。项目级主题、频次、名称共现不是证据。普通表格/列表相邻、标题、"
        "同段共现也不是关系；只有 structural_contract 明确给出直接父子结构，或给出"
        "section_label_parent_child=true 的独立标签/直接内容结构时，才能把该结构作为"
        "候选证据，并且仍须符合谓词定义和端点角色。关系必须覆盖两个精确端点，"
        "否定、情态、条件、引用和不确定表述不得输出 RELATION。若无法选出具体的允许谓词，"
        "返回 NO_RELATION 或 INSUFFICIENT_EVIDENCE。输出对象必须包含 decision、predicate、"
        "subject_side、object_side、evidence_spans、reason_code 六个字段；不要使用 relation "
        "包装字段，不要只输出谓词字符串。只返回 JSON，不要解释。"
        + orientation_instruction
        + preferred_instruction
        + excluded_instruction
    )
    payload = {
        "instruction": instruction,
        "contract_version": SECONDARY_PROMPT_VERSION,
        "secondary_pass_version": SECONDARY_PASS_VERSION,
        "route": {
            "name": route.route,
            "reason_code": route.reason_code,
            "rationale": route.rationale,
        },
        "source_context": {
            "start": context_span[0],
            "end": context_span[1],
            "text": context_text,
        },
        "endpoints": {
            "left": {"side": "LEFT", "span": list(left_span), "surface": left_surface, "type": left_type},
            "right": {"side": "RIGHT", "span": list(right_span), "surface": right_surface, "type": right_type},
        },
        "allowed_predicates": allowed,
        "excluded_predicate": str(excluded_predicate or ""),
        "preferred_predicate": str(preferred_predicate or ""),
        "structural_contract": dict(structural_contract),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _endpoint_span(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Mapping):
        return None
    raw_span = value.get("span")
    if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
        try:
            start, end = int(raw_span[0]), int(raw_span[1])
        except (TypeError, ValueError):
            return None
        return start, end
    if "start" in value and "end" in value:
        try:
            return int(value["start"]), int(value["end"])
        except (TypeError, ValueError):
            return None
    return None


def _normalise_relation_alias(
    raw: Mapping[str, Any],
    *,
    left_span: tuple[int, int],
    right_span: tuple[int, int],
) -> dict[str, Any]:
    """Adapt common Ollama structured-output drift into the strict contract.

    Some locally served models honour the semantic schema but emit the older
    ``{"relation": {"predicate": ..., "subject": ..., "object": ...}}``
    shape.  We accept it only when both endpoint spans are copied exactly;
    a predicate-only response remains fail-closed because its direction and
    evidence cannot be audited.
    """

    if "decision" in raw:
        return dict(raw)
    relation = raw.get("relation")
    if isinstance(relation, Mapping):
        subject_span = _endpoint_span(relation.get("subject"))
        object_span = _endpoint_span(relation.get("object"))
        if subject_span not in {left_span, right_span}:
            raise ValueError("secondary relation alias subject endpoint is not exact")
        if object_span not in {left_span, right_span}:
            raise ValueError("secondary relation alias object endpoint is not exact")
        if subject_span == object_span:
            raise ValueError("secondary relation alias endpoints must differ")
        evidence_spans = relation.get("evidence_spans", raw.get("evidence_spans"))
        if not isinstance(evidence_spans, list) or not evidence_spans:
            start = min(left_span[0], right_span[0])
            end = max(left_span[1], right_span[1])
            evidence_spans = [{"start": start, "end": end}]
        return {
            "decision": SECONDARY_RELATION,
            "predicate": relation.get("predicate") or raw.get("predicate") or "",
            "subject_side": "LEFT" if subject_span == left_span else "RIGHT",
            "object_side": "LEFT" if object_span == left_span else "RIGHT",
            "evidence_spans": evidence_spans,
            "reason_code": raw.get("reason_code") or "LEGACY_RELATION_SHAPE",
        }
    if isinstance(relation, str) and relation.strip():
        # A few responses put the predicate in ``relation`` while retaining
        # the rest of the canonical fields.  Do not infer missing direction.
        if not raw.get("subject_side") or not raw.get("object_side"):
            raise ValueError("secondary relation alias lacks endpoint roles")
        return {
            **dict(raw),
            "decision": SECONDARY_RELATION,
            "predicate": relation.strip(),
        }
    return dict(raw)


def parse_secondary_decision(
    raw: Any,
    *,
    allowed_predicates: Sequence[str],
    context_span: tuple[int, int],
    left_span: tuple[int, int],
    right_span: tuple[int, int],
    content_length: int,
    left_surface: str | None = None,
    right_surface: str | None = None,
    required_predicate: str = "",
) -> SecondaryDecision:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("secondary response is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("secondary response must be an object")
    raw = _normalise_relation_alias(
        raw,
        left_span=left_span,
        right_span=right_span,
    )
    decision = str(raw.get("decision") or "").strip().upper()
    if decision not in {SECONDARY_RELATION, SECONDARY_NO_RELATION, SECONDARY_INSUFFICIENT}:
        raise ValueError("secondary response has unknown decision")
    predicate = str(raw.get("predicate") or "").strip()
    reason_code = str(raw.get("reason_code") or "SECONDARY_UNSPECIFIED").strip()
    spans_raw = raw.get("evidence_spans")
    if not isinstance(spans_raw, list):
        raise ValueError("secondary evidence_spans must be an array")
    spans: list[tuple[int, int]] = []
    context_length = context_span[1] - context_span[0]
    for item in spans_raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        elif isinstance(item, Mapping):
            raw_span = item.get("span")
            if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
                start, end = raw_span
            else:
                start, end = item.get("start"), item.get("end")
        else:
            raise ValueError("secondary evidence span is invalid")
        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError) as exc:
            raise ValueError("secondary evidence span is invalid") from exc
        if context_span[0] <= start < end <= context_span[1] <= content_length:
            pass
        elif 0 <= start < end <= context_length:
            # The model sees source_context.text, so some otherwise valid
            # structured responses return offsets relative to that text.  Map
            # them back to the immutable document coordinates before applying
            # the same endpoint/evidence gates.  This is coordinate recovery,
            # not evidence expansion.
            start += context_span[0]
            end += context_span[0]
        else:
            raise ValueError("secondary evidence span is outside frozen context")
        spans.append((start, end))
    if decision != SECONDARY_RELATION:
        # A few schema-valid local responses echo an allowed predicate even
        # when decision is NO_RELATION/INSUFFICIENT_EVIDENCE.  The decision is
        # the negative gate; no hypothesis is created, so discard the echoed
        # value instead of turning a safe negative into a protocol failure.
        return SecondaryDecision(
            decision, NO_SPECIFIC_RELATION, "", "", tuple(spans), reason_code,
        )
    if predicate in {"", NO_SPECIFIC_RELATION} or predicate not in set(allowed_predicates):
        raise ValueError("secondary relation predicate is not allowed")
    if required_predicate and predicate != str(required_predicate).strip():
        raise ValueError("secondary relation predicate violates the route predicate lock")
    def endpoint_side(
        value: Any,
        surface: str | None,
        expected_span: tuple[int, int],
        opposite_span: tuple[int, int],
    ) -> str:
        # Schema-constrained local models sometimes return the endpoint object
        # itself in a field named ``subject_side``/``object_side``.  Accept
        # only an exact frozen span or an explicit LEFT/RIGHT marker; this is
        # transport normalization, not fuzzy endpoint resolution.
        if isinstance(value, Mapping):
            nested_side = value.get("side") or value.get("position")
            nested_normalized = str(nested_side or "").strip().upper()
            if nested_normalized in {"LEFT", "RIGHT"}:
                return nested_normalized
            nested_span = _endpoint_span(value)
            if nested_span == expected_span:
                return "LEFT" if expected_span == left_span else "RIGHT"
            if nested_span == opposite_span:
                return "RIGHT" if expected_span == left_span else "LEFT"
            value = value.get("surface") or value.get("text") or ""
        normalized = str(value or "").strip()
        upper = normalized.upper()
        if upper in {"LEFT", "RIGHT"}:
            return upper
        # Some local models copy the endpoint surface into the side fields.
        # Accept that only when the caller supplies the exact frozen surface;
        # no semantic or fuzzy endpoint matching is allowed here.
        if surface and normalized.casefold() == str(surface).strip().casefold():
            return "LEFT" if surface == left_surface else "RIGHT"
        return ""

    subject_side = endpoint_side(
        raw.get("subject_side"), left_surface, left_span, right_span,
    )
    if subject_side == "":
        subject_side = endpoint_side(
            raw.get("subject_side"), right_surface, right_span, left_span,
        )
    object_side = endpoint_side(
        raw.get("object_side"), left_surface, left_span, right_span,
    )
    if object_side == "":
        object_side = endpoint_side(
            raw.get("object_side"), right_surface, right_span, left_span,
        )
    if {subject_side, object_side} != {"LEFT", "RIGHT"}:
        raise ValueError("secondary relation endpoints must be LEFT and RIGHT")
    if not spans:
        raise ValueError("secondary relation requires evidence")
    covers_left = any(
        start <= left_span[0] and end >= left_span[1]
        for start, end in spans
    )
    covers_right = any(
        start <= right_span[0] and end >= right_span[1]
        for start, end in spans
    )
    if not covers_left:
        spans.append(left_span)
    if not covers_right:
        spans.append(right_span)
    return SecondaryDecision(
        decision, predicate, subject_side, object_side, tuple(spans), reason_code,
    )


def route_priority(route: str) -> int:
    return {
        ROUTE_EVIDENCE_RECOVERY: 0,
        ROUTE_POLARITY_REVIEW: 1,
        ROUTE_STRUCTURAL_REVIEW: 2,
        ROUTE_PREDICATE_REFINEMENT: 3,
        ROUTE_ROLE_RETRY: 4,
        ROUTE_TYPE_ROLE_REVIEW: 5,
        ROUTE_TECHNICAL_RETRY: 6,
        ROUTE_NONE: 99,
    }.get(str(route), 99)
