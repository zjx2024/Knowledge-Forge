"""Bounded, versioned relation classification used by shadow experiments."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .chunking import estimate_tokens
from .config import Config
from .observability import increment
from .policy import normalize_policy_set
from .sentence_spans import sentence_index as _sentence_index
from .sentence_spans import split_sentences as _sentences


NO_RELATION = "NO_RELATION"
ALLOWED_DIRECTIONS = frozenset({"as_is", "reversed", "symmetric"})
_RELATION_CUE = re.compile(
    r"\b(?:uses?|causes?|contains?|includes?|supports?|owns?|produces?|located|part of|"
    r"depends?|relates?|works? for|is a)\b|导致|使用|包含|属于|位于|支持|依赖|生产|关联",
    re.IGNORECASE,
)
_DENSE_SENTENCE_ENTITY_THRESHOLD = 6


def _looks_like_enumeration(text: str, entity_count: int) -> bool:
    if entity_count < 4:
        return False
    return (
        text.count(";") + text.count("；") >= 2
        or text.count(",") + text.count("，") >= 4
    )


class RelationBatchValidationError(ValueError):
    """Raised when a classifier batch is incomplete or structurally unsafe."""


@dataclass(frozen=True)
class RelationCandidate:
    candidate_id: str
    source_mention_id: str
    target_mention_id: str
    source_mention_key: str
    target_mention_key: str
    source_name: str
    target_name: str
    source_type: str
    target_type: str
    allowed_predicates: tuple[str, ...]
    allowed_directions: tuple[str, ...]
    context_id: str
    sentence_start: int
    sentence_end: int
    context_start_char: int
    context_end_char: int
    source_start_char: int
    source_end_char: int
    target_start_char: int
    target_end_char: int
    source_canonical_id: str = ""
    target_canonical_id: str = ""
    priority_key: tuple[int, int, int, int, float, str] = (0, 0, 0, 0, 0.0, "")


@dataclass(frozen=True)
class RelationDecision:
    candidate_id: str
    predicate: str
    direction: str
    evidence: str
    evidence_start: int | None
    evidence_end: int | None
    evidence_hash: str
    reason: str


@dataclass(frozen=True)
class RelationBatch:
    batch_id: str
    candidate_ids: list[str]
    prompt: str
    estimated_input_tokens: int
    output_token_reserve: int


@dataclass
class RelationPlan:
    policy_set: dict[str, str]
    raw_pairs: int
    blocked_pairs: int
    candidates: list[RelationCandidate]
    overflow: list[dict[str, Any]] = field(default_factory=list)
    qualified_count: int = 0
    deferred: list[dict[str, Any]] = field(default_factory=list)
    classified_count: int = 0
    gleaning_passes: int = 0

    @property
    def admitted_count(self) -> int:
        return len(self.candidates)


@dataclass
class ShadowClassificationResult:
    status: str
    plan: RelationPlan
    decisions: list[dict[str, Any]]
    batches: list[dict[str, Any]]
    metrics: dict[str, int]
    error: str = ""


@dataclass
class PreparedRelationShadow:
    plan: RelationPlan
    batches: list[RelationBatch]
    metrics: dict[str, int]


def _relation_types(logical_schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for item in logical_schema.get("relation_types", []) if isinstance(item, dict)]


def _types(definition: Mapping[str, Any], side: str) -> set[str]:
    aliases = ("source_types", "domain", "source") if side == "source" else (
        "target_types", "range", "target",
    )
    for key in aliases:
        value = definition.get(key)
        if isinstance(value, str):
            return {value}
        if isinstance(value, list):
            return {str(item) for item in value}
    return {"GenericEntity"}


def _type_allowed(actual: str, expected: set[str]) -> bool:
    return (
        not expected
        or "*" in expected
        or "GenericEntity" in expected
        or actual in expected
        or actual == "GenericEntity"
    )


class VersionedRelationPipeline:
    """Plan and validate relation classification without writing product facts."""

    def __init__(self, policy_set: Mapping[str, object]):
        self.policy_set = normalize_policy_set(policy_set)

    @staticmethod
    def _apply_candidate_limits(
        candidates: list[RelationCandidate],
    ) -> tuple[list[RelationCandidate], list[dict[str, Any]]]:
        ordered = sorted(candidates, key=lambda item: item.priority_key)
        admitted: list[RelationCandidate] = []
        overflow: list[dict[str, Any]] = []
        per_source: dict[str, int] = defaultdict(int)
        for candidate in ordered:
            if per_source[candidate.source_mention_id] >= Config.relation_max_targets_per_source:
                overflow.append({
                    "candidate_id": candidate.candidate_id,
                    "source_mention_id": candidate.source_mention_id,
                    "reason": "per_source_limit",
                    "candidate": asdict(candidate),
                })
                continue
            if len(admitted) >= Config.relation_max_candidates_per_chunk:
                overflow.append({
                    "candidate_id": candidate.candidate_id,
                    "source_mention_id": candidate.source_mention_id,
                    "reason": "chunk_limit",
                    "candidate": asdict(candidate),
                })
                continue
            admitted.append(candidate)
            per_source[candidate.source_mention_id] += 1
        return admitted, overflow

    def plan(
        self,
        *,
        project_scope: str,
        chunk_key: str,
        content: str,
        entities: list[dict[str, Any]],
        logical_schema: Mapping[str, Any],
        sentence_window: int | None = None,
    ) -> RelationPlan:
        if sentence_window is None:
            sentence_window = Config.relation_sentence_window
        fixed: list[dict[str, Any]] = []
        for item in entities:
            mention_id = str(item.get("mention_id") or "")
            name = str(item.get("surface") or item.get("name") or "")
            canonical_id = str(item.get("canonical_id") or "")
            start = item.get("mention_start")
            end = item.get("mention_end")
            if (
                not mention_id
                or not name
                or not canonical_id
                or start is None
                or end is None
                or item.get("confidence") is None
            ):
                continue
            start, end = int(start), int(end)
            if start < 0 or end <= start or end > len(content) or content[start:end] != name:
                continue
            fixed.append({
                **item,
                "mention_id": mention_id,
                "mention_key": str(item.get("mention_key") or mention_id),
                "name": name,
                "canonical_id": canonical_id,
                "mention_start": start,
                "mention_end": end,
            })
        raw_pairs = len(fixed) * max(0, len(fixed) - 1)
        sentences = _sentences(content)
        locations = [
            _sentence_index(sentences, int(item["mention_start"]))
            for item in fixed
        ]
        sentence_buckets: dict[int, list[int]] = defaultdict(list)
        for entity_index, sentence_index in enumerate(locations):
            if sentence_index is not None:
                sentence_buckets[sentence_index].append(entity_index)
        definitions = _relation_types(logical_schema)
        candidates: list[RelationCandidate] = []
        pair_checks = 0
        for source_index, source in enumerate(fixed):
            source_sentence = locations[source_index]
            if source_sentence is None:
                continue
            target_indexes = sorted(
                target_index
                for sentence_index in range(
                    max(0, source_sentence - sentence_window),
                    min(len(sentences), source_sentence + sentence_window + 1),
                )
                for target_index in sentence_buckets.get(sentence_index, ())
                if target_index != source_index
            )
            for target_index in target_indexes:
                pair_checks += 1
                target = fixed[target_index]
                if source["canonical_id"] == target["canonical_id"]:
                    continue
                source_id = str(source["mention_id"])
                target_id = str(target["mention_id"])
                left_index, right_index = source_sentence, locations[target_index]
                allowed_directions: dict[str, str] = {}
                source_type = str(source.get("type") or "GenericEntity")
                target_type = str(target.get("type") or "GenericEntity")
                for definition in definitions:
                    name = str(definition.get("name") or "")
                    if not name:
                        continue
                    direct = (
                        _type_allowed(source_type, _types(definition, "source"))
                        and _type_allowed(target_type, _types(definition, "target"))
                    )
                    reverse = (
                        _type_allowed(source_type, _types(definition, "target"))
                        and _type_allowed(target_type, _types(definition, "source"))
                    )
                    if bool(definition.get("symmetric")) and (direct or reverse):
                        # A symmetric relation is represented by one stable unordered candidate.
                        if source_id < target_id:
                            allowed_directions[name] = "symmetric"
                    elif direct:
                        allowed_directions[name] = "as_is"
                    # Reverse traversal is represented by the explicitly declared inverse
                    # predicate (when present in the schema), not by a duplicate candidate
                    # for the forward predicate.
                allowed = tuple(allowed_directions)
                if right_index is None or not allowed:
                    continue
                sentence_distance = abs(left_index - right_index)
                char_distance = abs(
                    (int(source["mention_start"]) + int(source["mention_end"])) // 2
                    - (int(target["mention_start"]) + int(target["mention_end"])) // 2
                )
                if char_distance > Config.relation_max_char_distance:
                    continue
                context_start, context_end = sorted((left_index, right_index))
                context = " ".join(item[2] for item in sentences[context_start:context_end + 1])
                pair_start = min(
                    int(source["mention_start"]),
                    int(target["mention_start"]),
                )
                pair_end = max(
                    int(source["mention_end"]),
                    int(target["mention_end"]),
                )
                has_relation_cue = bool(_RELATION_CUE.search(content[pair_start:pair_end]))
                dense_same_sentence = (
                    sentence_distance == 0
                    and len(sentence_buckets[source_sentence])
                    >= _DENSE_SENTENCE_ENTITY_THRESHOLD
                )
                list_like_same_sentence = (
                    sentence_distance == 0
                    and _looks_like_enumeration(
                        sentences[source_sentence][2],
                        len(sentence_buckets[source_sentence]),
                    )
                )
                if (
                    sentence_distance > sentence_window
                    or (sentence_distance > 0 and not has_relation_cue)
                    or (dense_same_sentence and not has_relation_cue)
                    or (list_like_same_sentence and not has_relation_cue)
                ):
                    continue
                material = "\x1f".join((
                    project_scope, chunk_key, source_id, target_id,
                    ",".join(allowed), self.policy_set["relation_candidate_policy_version"],
                ))
                candidate_id = f"p1cand_{hashlib.sha256(material.encode('utf-8')).hexdigest()}"
                candidates.append(RelationCandidate(
                    candidate_id=candidate_id,
                    source_mention_id=source_id,
                    target_mention_id=target_id,
                    source_mention_key=str(source["mention_key"]),
                    target_mention_key=str(target["mention_key"]),
                    source_name=str(source["name"]),
                    target_name=str(target["name"]),
                    source_type=str(source.get("type") or "GenericEntity"),
                    target_type=str(target.get("type") or "GenericEntity"),
                    allowed_predicates=allowed,
                    allowed_directions=tuple(allowed_directions[name] for name in allowed),
                    context_id=f"ctx-{context_start}-{context_end}",
                    sentence_start=context_start,
                    sentence_end=context_end,
                    context_start_char=sentences[context_start][0],
                    context_end_char=sentences[context_end][1],
                    source_start_char=int(source["mention_start"]),
                    source_end_char=int(source["mention_end"]),
                    target_start_char=int(target["mention_start"]),
                    target_end_char=int(target["mention_end"]),
                    source_canonical_id=str(source["canonical_id"]),
                    target_canonical_id=str(target["canonical_id"]),
                    priority_key=(
                        sentence_distance,
                        int(source_type == "GenericEntity")
                        + int(target_type == "GenericEntity"),
                        0 if has_relation_cue else 1,
                        char_distance,
                        -float(source.get("confidence") or 0.0)
                        - float(target.get("confidence") or 0.0),
                        candidate_id,
                    ),
                ))
        qualified_count = len(candidates)
        candidates, overflow = self._apply_candidate_limits(candidates)
        increment("relation_raw_pairs", raw_pairs)
        increment("relation_pair_checks", pair_checks)
        increment("relation_candidates", len(candidates))
        increment("relation_candidate_overflow", len(overflow))
        return RelationPlan(
            policy_set=dict(self.policy_set), raw_pairs=raw_pairs,
            blocked_pairs=raw_pairs - qualified_count, candidates=candidates,
            overflow=overflow, qualified_count=qualified_count,
        )

    @staticmethod
    def _candidate_payload(candidate: RelationCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "source_id": candidate.source_mention_id,
            "target_id": candidate.target_mention_id,
            "source_mention_key": candidate.source_mention_key,
            "target_mention_key": candidate.target_mention_key,
            "source_canonical_id": candidate.source_canonical_id,
            "target_canonical_id": candidate.target_canonical_id,
            "allowed_predicates": list(candidate.allowed_predicates),
            "allowed_directions": list(candidate.allowed_directions),
            "context_id": candidate.context_id,
            "context_span": [candidate.context_start_char, candidate.context_end_char],
            "source_span": [candidate.source_start_char, candidate.source_end_char],
            "target_span": [candidate.target_start_char, candidate.target_end_char],
        }

    def build_batches(
        self,
        plan: RelationPlan,
        *,
        content: str,
        logical_schema: Mapping[str, Any],
        candidate_budget_tokens: int,
        output_token_reserve: int = 256,
        max_candidates_per_batch: int = 32,
    ) -> list[RelationBatch]:
        max_candidates_per_batch = max(1, min(32, int(max_candidates_per_batch)))
        if candidate_budget_tokens <= 0:
            plan.deferred.extend({"candidate_id": item.candidate_id, "reason": "no_candidate_budget"}
                                 for item in plan.candidates)
            return []
        fixed = {
            "policy": plan.policy_set["relation_classifier_policy_version"],
            "relation_schema": _relation_types(logical_schema),
        }
        batches: list[RelationBatch] = []
        pending: list[RelationCandidate] = []
        pending_signature: tuple[
            tuple[tuple[str, str], ...],
            str,
            int,
            int,
        ] | None = None

        def batch_signature(candidate: RelationCandidate) -> tuple[
            tuple[tuple[str, str], ...],
            str,
            int,
            int,
        ]:
            contract = tuple(zip(
                candidate.allowed_predicates,
                candidate.allowed_directions,
                strict=True,
            ))
            return (
                contract,
                candidate.context_id,
                candidate.context_start_char,
                candidate.context_end_char,
            )

        def render(values: Iterable[RelationCandidate]) -> str:
            selected = list(values)
            entities: dict[str, dict[str, str]] = {}
            contexts: dict[str, dict[str, Any]] = {}
            for item in selected:
                entities.setdefault(item.source_mention_id, {
                    "id": item.source_mention_id,
                    "name": item.source_name,
                    "type": item.source_type,
                    "canonical_id": item.source_canonical_id,
                })
                entities.setdefault(item.target_mention_id, {
                    "id": item.target_mention_id,
                    "name": item.target_name,
                    "type": item.target_type,
                    "canonical_id": item.target_canonical_id,
                })
                contexts.setdefault(item.context_id, {
                    "context_id": item.context_id,
                    "start": item.context_start_char,
                    "end": item.context_end_char,
                    "text": content[item.context_start_char:item.context_end_char],
                })
            return json.dumps(
                {
                    **fixed,
                    "source_contexts": list(contexts.values()),
                    "entities": list(entities.values()),
                    "candidates": [self._candidate_payload(item) for item in selected],
                },
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )

        def flush() -> None:
            nonlocal pending_signature
            if not pending:
                return
            prompt = render(pending)
            batch_id = "p1batch_" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            batches.append(RelationBatch(
                batch_id=batch_id,
                candidate_ids=[item.candidate_id for item in pending],
                prompt=prompt,
                estimated_input_tokens=estimate_tokens(prompt),
                output_token_reserve=output_token_reserve,
            ))
            pending.clear()
            pending_signature = None

        for candidate in plan.candidates:
            candidate_signature = batch_signature(candidate)
            if pending and candidate_signature != pending_signature:
                flush()
            proposed = [*pending, candidate]
            proposed_tokens = estimate_tokens(render(proposed))
            if proposed_tokens > candidate_budget_tokens or len(proposed) > max_candidates_per_batch:
                flush()
                if estimate_tokens(render([candidate])) > candidate_budget_tokens:
                    plan.deferred.append({
                        "candidate_id": candidate.candidate_id,
                        "reason": "candidate_exceeds_budget",
                    })
                    continue
            pending.append(candidate)
            pending_signature = candidate_signature
        flush()
        return batches

    def validate_decisions(
        self,
        candidates: list[RelationCandidate],
        raw_decisions: Any,
        *,
        content: str,
        logical_schema: Mapping[str, Any],
    ) -> list[RelationDecision]:
        if not isinstance(raw_decisions, list) or any(not isinstance(item, dict) for item in raw_decisions):
            raise RelationBatchValidationError("decisions must be an array of objects")
        requested = {item.candidate_id: item for item in candidates}
        ids = [str(item.get("candidate_id") or "") for item in raw_decisions]
        duplicates = sorted({value for value in ids if value and ids.count(value) > 1})
        if duplicates:
            raise RelationBatchValidationError(f"duplicate candidate ids: {duplicates}")
        unknown = sorted(set(ids) - set(requested))
        if unknown:
            raise RelationBatchValidationError(f"unknown candidate ids: {unknown}")
        missing = sorted(set(requested) - set(ids))
        if missing:
            raise RelationBatchValidationError(f"missing candidate ids: {missing}")
        allowed_schema = {str(item.get("name")) for item in _relation_types(logical_schema)}
        result: list[RelationDecision] = []
        for raw in raw_decisions:
            candidate = requested[str(raw["candidate_id"])]
            predicate = str(raw.get("predicate") or "")
            direction = str(raw.get("direction") or "")
            evidence = str(raw.get("evidence") or "")
            reason = str(raw.get("reason") or "").strip()
            if predicate != NO_RELATION and (
                predicate not in allowed_schema or predicate not in candidate.allowed_predicates
            ):
                raise RelationBatchValidationError(f"predicate is not allowed: {predicate}")
            if direction not in ALLOWED_DIRECTIONS or not reason:
                raise RelationBatchValidationError("direction and reason are required")
            if predicate == NO_RELATION and direction != "as_is":
                raise RelationBatchValidationError("NO_RELATION direction must be as_is")
            if predicate != NO_RELATION:
                expected_direction = candidate.allowed_directions[
                    candidate.allowed_predicates.index(predicate)
                ]
                if direction != expected_direction:
                    raise RelationBatchValidationError(
                        f"direction is not allowed for {predicate}: {direction}"
                    )
            if predicate != NO_RELATION:
                if not evidence or evidence not in content:
                    raise RelationBatchValidationError("positive evidence must be an exact source quote")
                if candidate.source_name not in evidence or candidate.target_name not in evidence:
                    raise RelationBatchValidationError("positive evidence must contain both endpoints")
                raw_start, raw_end = raw.get("evidence_start"), raw.get("evidence_end")
                if raw_start is None or raw_end is None:
                    occurrences = [match.start() for match in re.finditer(re.escape(evidence), content)]
                    if len(occurrences) != 1:
                        raise RelationBatchValidationError(
                            "positive evidence occurrence is ambiguous; exact offsets are required"
                        )
                    evidence_start = occurrences[0]
                    evidence_end = evidence_start + len(evidence)
                else:
                    evidence_start, evidence_end = int(raw_start), int(raw_end)
                    if (
                        evidence_start < 0
                        or evidence_end <= evidence_start
                        or evidence_end > len(content)
                        or content[evidence_start:evidence_end] != evidence
                    ):
                        raise RelationBatchValidationError(
                            "positive evidence must match the exact source span"
                        )
                evidence_hash = hashlib.sha256(evidence.encode("utf-8")).hexdigest()
            else:
                evidence_start = evidence_end = None
                evidence_hash = ""
            result.append(RelationDecision(
                candidate_id=candidate.candidate_id, predicate=predicate,
                direction=direction, evidence=evidence,
                evidence_start=evidence_start, evidence_end=evidence_end,
                evidence_hash=evidence_hash, reason=reason,
            ))
        return result

    @staticmethod
    def _decision_response_schema(
        candidates: list[RelationCandidate],
        *,
        content: str,
    ) -> dict[str, Any]:
        candidate_ids = [item.candidate_id for item in candidates]
        return {
            "type": "object",
            "properties": {
                item.candidate_id: {
                    "type": "object",
                    "properties": {
                        "predicate": {
                            "type": "string",
                            "enum": [NO_RELATION, *dict.fromkeys(item.allowed_predicates)],
                        },
                        "direction": {
                            "type": "string",
                            "enum": sorted(ALLOWED_DIRECTIONS),
                        },
                        "evidence": {
                            "type": "string",
                            "enum": [
                                "",
                                content[item.context_start_char:item.context_end_char],
                            ],
                            "description": (
                                "Empty for NO_RELATION; otherwise the complete, exact "
                                "SOURCE_CONTEXT text containing both endpoints."
                            ),
                        },
                        "evidence_start": {
                            "type": ["integer", "null"],
                            "enum": [None, item.context_start_char],
                            "description": (
                                "Absolute zero-based half-open offset in the full chunk "
                                "content; content[evidence_start:evidence_end] must equal "
                                "evidence."
                            ),
                        },
                        "evidence_end": {
                            "type": ["integer", "null"],
                            "enum": [None, item.context_end_char],
                            "description": (
                                "Absolute zero-based half-open offset in the full chunk "
                                "content; content[evidence_start:evidence_end] must equal "
                                "evidence."
                            ),
                        },
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "predicate",
                        "direction",
                        "evidence",
                        "evidence_start",
                        "evidence_end",
                        "reason",
                    ],
                    "additionalProperties": False,
                }
                for item in candidates
            },
            "required": candidate_ids,
            "additionalProperties": False,
        }

    @staticmethod
    def _decision_system_prompt(candidate_count: int) -> str:
        return (
            "Classify every admitted relation candidate from SOURCE_CONTEXTS only. "
            "Return one root JSON object keyed directly by candidate_id, with exactly "
            f"{candidate_count} required keys; do not add a decisions wrapper. "
            "For every key, always return predicate, direction, evidence, evidence_start, "
            "evidence_end, and a concise non-empty reason. "
            "For each candidate, use only one of that candidate's allowed_predicates. "
            "Use an allowed predicate only when an exact quote proves endpoints, meaning "
            "and direction. For a positive relation, copy the candidate's complete "
            "SOURCE_CONTEXT text as evidence and use that context's absolute start/end "
            "offsets. Otherwise return predicate NO_RELATION, direction as_is, empty "
            "evidence, null offsets, and a concise reason."
        )

    @staticmethod
    def _normalize_model_decisions(
        candidates: list[RelationCandidate],
        raw: Any,
    ) -> list[dict[str, Any]]:
        expected_ids = [item.candidate_id for item in candidates]
        decisions = None
        if isinstance(raw, dict):
            decisions = raw.get("decisions")
            if decisions is None and isinstance(raw.get("items"), list):
                decisions = raw["items"]
            if decisions is None and set(raw) == set(expected_ids):
                decisions = raw
            if (
                decisions is None
                and len(expected_ids) == 1
                and {"predicate", "direction", "evidence", "reason"} <= set(raw)
            ):
                decisions = {expected_ids[0]: raw}
            if decisions is None and len(expected_ids) == 1 and len(raw) == 1:
                only_value = next(iter(raw.values()))
                if (
                    isinstance(only_value, dict)
                    and {"predicate", "direction", "evidence", "reason"} <= set(only_value)
                ):
                    decisions = {expected_ids[0]: only_value}
        if isinstance(decisions, list):
            return decisions
        if not isinstance(decisions, dict):
            raise RelationBatchValidationError(
                "relation response must be an array or an object keyed by candidate_id"
            )
        actual_ids = set(decisions)
        if actual_ids != set(expected_ids):
            missing = sorted(set(expected_ids) - actual_ids)
            unknown = sorted(actual_ids - set(expected_ids))
            raise RelationBatchValidationError(
                f"decision candidate keys mismatch: missing={missing}, unknown={unknown}"
            )
        normalized = []
        for candidate_id in expected_ids:
            value = decisions[candidate_id]
            if not isinstance(value, dict):
                raise RelationBatchValidationError(
                    f"decision must be an object: {candidate_id}"
                )
            normalized.append({"candidate_id": candidate_id, **value})
        return normalized

    @staticmethod
    def _canonicalize_context_evidence_offsets(
        candidates: list[RelationCandidate],
        decisions: list[dict[str, Any]],
        *,
        content: str,
    ) -> list[dict[str, Any]]:
        """Repair only offsets that are provable from an exact frozen context quote."""

        candidate_map = {item.candidate_id: item for item in candidates}
        normalized: list[dict[str, Any]] = []
        for raw in decisions:
            decision = dict(raw)
            candidate = candidate_map.get(str(decision.get("candidate_id") or ""))
            if candidate is None or str(decision.get("predicate") or "") == NO_RELATION:
                normalized.append(decision)
                continue
            evidence = str(decision.get("evidence") or "")
            if (
                not evidence
                or candidate.source_name not in evidence
                or candidate.target_name not in evidence
            ):
                normalized.append(decision)
                continue
            context = content[
                candidate.context_start_char:candidate.context_end_char
            ]
            occurrences = [
                match.start() for match in re.finditer(re.escape(evidence), context)
            ]
            if len(occurrences) != 1:
                normalized.append(decision)
                continue
            expected_start = candidate.context_start_char + occurrences[0]
            expected_end = expected_start + len(evidence)
            if (
                decision.get("evidence_start") != expected_start
                or decision.get("evidence_end") != expected_end
            ):
                increment("relation_evidence_offset_repairs")
                decision["evidence_start"] = expected_start
                decision["evidence_end"] = expected_end
            normalized.append(decision)
        return normalized

    def _fail_close_unsafe_model_decisions(
        self,
        candidates: list[RelationCandidate],
        decisions: list[dict[str, Any]],
        *,
        content: str,
        logical_schema: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Turn an unsafe model positive into a conservative explicit negative."""

        candidate_map = {item.candidate_id: item for item in candidates}
        normalized: list[dict[str, Any]] = []
        for raw in decisions:
            decision = dict(raw)
            candidate = candidate_map.get(str(decision.get("candidate_id") or ""))
            if candidate is None:
                normalized.append(decision)
                continue
            if str(decision.get("predicate") or "") == NO_RELATION:
                decision.update({
                    "direction": "as_is",
                    "evidence": "",
                    "evidence_start": None,
                    "evidence_end": None,
                    "reason": str(decision.get("reason") or "no supported relation"),
                })
                normalized.append(decision)
                continue
            try:
                self.validate_decisions(
                    [candidate], [decision],
                    content=content, logical_schema=logical_schema,
                )
            except RelationBatchValidationError as exc:
                increment("relation_unsafe_positive_fail_closed")
                decision.update({
                    "predicate": NO_RELATION,
                    "direction": "as_is",
                    "evidence": "",
                    "evidence_start": None,
                    "evidence_end": None,
                    "reason": f"fail-closed model positive: {exc}",
                })
            normalized.append(decision)
        return normalized

    def classify_shadow(
        self,
        *,
        classifier: Any,
        model: str,
        context_budget_tokens: int,
        project_scope: str,
        chunk_key: str,
        content: str,
        entities: list[dict[str, Any]],
        logical_schema: Mapping[str, Any],
    ) -> ShadowClassificationResult:
        """Run the treatment classifier with strict completeness and bounded budgets."""

        plan = self.plan(
            project_scope=project_scope, chunk_key=chunk_key, content=content,
            entities=entities, logical_schema=logical_schema,
        )
        uncovered = self.uncovered_cue_windows(content, plan.candidates)
        if uncovered and Config.relation_classifier_gleaning_window > 1:
            expanded = self.plan(
                project_scope=project_scope,
                chunk_key=chunk_key,
                content=content,
                entities=entities,
                logical_schema=logical_schema,
                sentence_window=Config.relation_classifier_gleaning_window,
            )
            known = {item.candidate_id for item in plan.candidates}
            uncovered_indexes = {int(item["sentence_index"]) for item in uncovered}
            additions = [
                item for item in expanded.candidates
                if item.candidate_id not in known
                and any(
                    index in uncovered_indexes
                    for index in range(item.sentence_start, item.sentence_end + 1)
                )
            ]
            if additions:
                combined, gleaning_overflow = self._apply_candidate_limits([
                    *plan.candidates, *additions,
                ])
                plan.candidates = combined
                plan.overflow.extend(gleaning_overflow)
                plan.qualified_count += len(additions)
                plan.blocked_pairs = max(0, plan.raw_pairs - plan.qualified_count)
                plan.gleaning_passes = 1
                increment("relation_gleaning_calls")
        output_reserve = min(
            max(128, Config.relation_classifier_output_reserve),
            max(128, int(context_budget_tokens) // 3),
        )
        candidate_budget = max(
            0,
            int(context_budget_tokens) - output_reserve - Config.relation_classifier_safety_margin,
        )
        batches = self.build_batches(
            plan,
            content=content,
            logical_schema=logical_schema,
            candidate_budget_tokens=candidate_budget,
            output_token_reserve=output_reserve,
            max_candidates_per_batch=Config.relation_classifier_max_candidates,
        )
        metrics = {
            "raw_pairs": plan.raw_pairs,
            "blocked_pairs": plan.blocked_pairs,
            "admitted_candidates": plan.admitted_count,
            "qualified_candidates": plan.qualified_count,
            "overflow_candidates": len(plan.overflow),
            "classified": 0,
            "deferred": len(plan.deferred),
            "model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "uncovered_cue_windows": len(self.uncovered_cue_windows(content, plan.candidates)),
            "gleaning_passes": plan.gleaning_passes,
        }
        if plan.deferred:
            return ShadowClassificationResult(
                status="deferred", plan=plan, decisions=[], batches=[], metrics=metrics,
                error="one or more admitted candidates exceeded the bounded request budget",
            )
        decisions: list[dict[str, Any]] = []
        batch_audit: list[dict[str, Any]] = []
        candidate_map = {item.candidate_id: item for item in plan.candidates}
        for batch in batches:
            metrics["model_calls"] += 1
            increment("relation_classifier_batches")
            metrics["input_tokens"] += batch.estimated_input_tokens
            selected = [candidate_map[candidate_id] for candidate_id in batch.candidate_ids]
            try:
                raw = classifier.json_chat(
                    model=model,
                    system=self._decision_system_prompt(len(selected)),
                    prompt=batch.prompt,
                    schema=self._decision_response_schema(selected, content=content),
                    temperature=0,
                    context_budget_tokens=context_budget_tokens,
                    max_output_tokens=batch.output_token_reserve,
                    safety_margin_tokens=Config.relation_classifier_safety_margin,
                )
                normalized = self._canonicalize_context_evidence_offsets(
                    selected,
                    self._normalize_model_decisions(selected, raw),
                    content=content,
                )
                normalized = self._fail_close_unsafe_model_decisions(
                    selected, normalized,
                    content=content, logical_schema=logical_schema,
                )
                metrics["output_tokens"] += estimate_tokens(
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
                )
                validated = self.validate_decisions(
                    selected,
                    normalized,
                    content=content,
                    logical_schema=logical_schema,
                )
            except Exception as exc:
                increment("relation_classifier_failures")
                return ShadowClassificationResult(
                    status="failed", plan=plan, decisions=[], batches=batch_audit,
                    metrics={**metrics, "classified": 0}, error=str(exc),
                )
            batch_audit.append({
                "batch_id": batch.batch_id,
                "candidate_ids": list(batch.candidate_ids),
                "request": json.loads(batch.prompt),
                "response": raw,
            })
            decisions.extend({
                **asdict(item),
                "status": "negative" if item.predicate == NO_RELATION else "classified",
                "request_batch_id": batch.batch_id,
            } for item in validated)
            increment("relation_verified_positive", sum(
                1 for item in validated if item.predicate != NO_RELATION
            ))
            increment("relation_verified_negative", sum(
                1 for item in validated if item.predicate == NO_RELATION
            ))
        plan.classified_count = len(decisions)
        metrics["classified"] = len(decisions)
        return ShadowClassificationResult(
            status="incomplete" if plan.overflow else "completed",
            plan=plan, decisions=decisions, batches=batch_audit, metrics=metrics,
            error=(
                "bounded candidate limits deferred one or more qualified candidates"
                if plan.overflow else ""
            ),
        )

    def prepare_shadow(
        self,
        *,
        context_budget_tokens: int,
        project_scope: str,
        chunk_key: str,
        content: str,
        entities: list[dict[str, Any]],
        logical_schema: Mapping[str, Any],
    ) -> PreparedRelationShadow:
        """Create deterministic, bounded work without performing model I/O."""

        plan = self.plan(
            project_scope=project_scope, chunk_key=chunk_key, content=content,
            entities=entities, logical_schema=logical_schema,
        )
        uncovered = self.uncovered_cue_windows(content, plan.candidates)
        if uncovered and Config.relation_classifier_gleaning_window > 1:
            expanded = self.plan(
                project_scope=project_scope, chunk_key=chunk_key, content=content,
                entities=entities, logical_schema=logical_schema,
                sentence_window=Config.relation_classifier_gleaning_window,
            )
            known = {item.candidate_id for item in plan.candidates}
            uncovered_indexes = {int(item["sentence_index"]) for item in uncovered}
            additions = [
                item for item in expanded.candidates
                if item.candidate_id not in known
                and any(index in uncovered_indexes for index in range(
                    item.sentence_start, item.sentence_end + 1,
                ))
            ]
            if additions:
                combined, gleaning_overflow = self._apply_candidate_limits([
                    *plan.candidates, *additions,
                ])
                plan.candidates = combined
                plan.overflow.extend(gleaning_overflow)
                plan.qualified_count += len(additions)
                plan.blocked_pairs = max(0, plan.raw_pairs - plan.qualified_count)
                plan.gleaning_passes = 1
                increment("relation_gleaning_calls")
        output_reserve = min(
            max(128, Config.relation_classifier_output_reserve),
            max(128, int(context_budget_tokens) // 3),
        )
        candidate_budget = max(
            0,
            int(context_budget_tokens) - output_reserve - Config.relation_classifier_safety_margin,
        )
        batches = self.build_batches(
            plan,
            content=content,
            logical_schema=logical_schema,
            candidate_budget_tokens=candidate_budget,
            output_token_reserve=output_reserve,
            max_candidates_per_batch=Config.relation_classifier_max_candidates,
        )
        return PreparedRelationShadow(
            plan=plan,
            batches=batches,
            metrics={
                "raw_pairs": plan.raw_pairs,
                "blocked_pairs": plan.blocked_pairs,
                "admitted_candidates": plan.admitted_count,
                "qualified_candidates": plan.qualified_count,
                "overflow_candidates": len(plan.overflow),
                "classified": 0,
                "deferred": len(plan.deferred),
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "uncovered_cue_windows": len(
                    self.uncovered_cue_windows(content, plan.candidates)
                ),
                "gleaning_passes": plan.gleaning_passes,
            },
        )

    def classify_prepared_batch(
        self,
        *,
        classifier: Any,
        model: str,
        context_budget_tokens: int,
        batch: RelationBatch,
        candidates: list[RelationCandidate],
        content: str,
        logical_schema: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Classify exactly one persisted work item and strictly verify completeness."""

        response_schema = self._decision_response_schema(candidates, content=content)
        increment("relation_classifier_batches")
        try:
            raw = classifier.json_chat(
                model=model,
                system=self._decision_system_prompt(len(candidates)),
                prompt=batch.prompt,
                schema=response_schema,
                temperature=0,
                context_budget_tokens=context_budget_tokens,
                max_output_tokens=batch.output_token_reserve,
                safety_margin_tokens=Config.relation_classifier_safety_margin,
            )
            normalized = self._canonicalize_context_evidence_offsets(
                candidates,
                self._normalize_model_decisions(candidates, raw),
                content=content,
            )
            normalized = self._fail_close_unsafe_model_decisions(
                candidates, normalized,
                content=content, logical_schema=logical_schema,
            )
            validated = self.validate_decisions(
                candidates,
                normalized,
                content=content,
                logical_schema=logical_schema,
            )
        except Exception:
            increment("relation_classifier_failures")
            raise
        increment("relation_verified_positive", sum(
            1 for item in validated if item.predicate != NO_RELATION
        ))
        increment("relation_verified_negative", sum(
            1 for item in validated if item.predicate == NO_RELATION
        ))
        return [
            {
                **asdict(item),
                "status": "negative" if item.predicate == NO_RELATION else "classified",
                "request_batch_id": batch.batch_id,
            }
            for item in validated
        ], raw

    @staticmethod
    def uncovered_cue_windows(
        content: str,
        candidates: list[RelationCandidate],
    ) -> list[dict[str, Any]]:
        sentences = _sentences(content)
        covered = {
            index
            for candidate in candidates
            for index in range(candidate.sentence_start, candidate.sentence_end + 1)
        }
        return [
            {"sentence_index": index, "start": start, "end": end, "text": text}
            for index, (start, end, text) in enumerate(sentences)
            if index not in covered and _RELATION_CUE.search(text)
        ]
