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


NO_RELATION = "NO_RELATION"
ALLOWED_DIRECTIONS = frozenset({"as_is", "reversed", "symmetric"})
_SENTENCE_RE = re.compile(r"[^。！？!?\.\n]+[。！？!?\.]?|[^\n]+$")
_RELATION_CUE = re.compile(
    r"\b(?:uses?|causes?|contains?|includes?|supports?|owns?|produces?|located|part of|"
    r"depends?|relates?|works? for|is a)\b|导致|使用|包含|属于|位于|支持|依赖|生产|关联",
    re.IGNORECASE,
)


class RelationBatchValidationError(ValueError):
    """Raised when a classifier batch is incomplete or structurally unsafe."""


@dataclass(frozen=True)
class RelationCandidate:
    candidate_id: str
    source_mention_id: str
    target_mention_id: str
    source_name: str
    target_name: str
    source_type: str
    target_type: str
    allowed_predicates: tuple[str, ...]
    allowed_directions: tuple[str, ...]
    context_id: str
    sentence_start: int
    sentence_end: int


@dataclass(frozen=True)
class RelationDecision:
    candidate_id: str
    predicate: str
    direction: str
    evidence: str
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


def _sentences(content: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in _SENTENCE_RE.finditer(content)]


def _sentence_index(sentences: list[tuple[int, int, str]], offset: int) -> int | None:
    for index, (start, end, _) in enumerate(sentences):
        if start <= offset < end or (offset == end and start < end):
            return index
    return None


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
    return not expected or "GenericEntity" in expected or actual in expected or actual == "GenericEntity"


class VersionedRelationPipeline:
    """Plan and validate relation classification without writing product facts."""

    def __init__(self, policy_set: Mapping[str, object]):
        self.policy_set = normalize_policy_set(policy_set)

    def plan(
        self,
        *,
        project_scope: str,
        chunk_key: str,
        content: str,
        entities: list[dict[str, Any]],
        logical_schema: Mapping[str, Any],
        sentence_window: int = 1,
    ) -> RelationPlan:
        fixed: list[dict[str, Any]] = []
        for item in entities:
            name = str(item.get("name") or "")
            start = item.get("evidence_start")
            end = item.get("evidence_end")
            if not name or start is None or end is None:
                continue
            start, end = int(start), int(end)
            evidence = str(item.get("evidence_text") or "")
            if start < 0 or end <= start or end > len(content) or content[start:end] != evidence:
                continue
            material = "\x1f".join((
                project_scope, chunk_key, str(start), str(end), evidence,
                str(item.get("type") or "GenericEntity"),
                self.policy_set["entity_extraction_policy_version"],
            ))
            fixed.append({
                **item,
                "original_id": str(item.get("id") or ""),
                "id": "p1mention_" + hashlib.sha256(material.encode("utf-8")).hexdigest(),
            })
        raw_pairs = len(fixed) * max(0, len(fixed) - 1)
        sentences = _sentences(content)
        locations = [
            _sentence_index(sentences, int(item.get("evidence_start") or 0))
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
                source_id, target_id = str(source["id"]), str(target["id"])
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
                context_start, context_end = sorted((left_index, right_index))
                context = " ".join(item[2] for item in sentences[context_start:context_end + 1])
                if sentence_distance > sentence_window or (
                    sentence_distance > 0 and not _RELATION_CUE.search(context)
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
                    source_name=str(source["name"]),
                    target_name=str(target["name"]),
                    source_type=str(source.get("type") or "GenericEntity"),
                    target_type=str(target.get("type") or "GenericEntity"),
                    allowed_predicates=allowed,
                    allowed_directions=tuple(allowed_directions[name] for name in allowed),
                    context_id=f"ctx-{context_start}-{context_end}",
                    sentence_start=context_start,
                    sentence_end=context_end,
                ))
        increment("relation_raw_pairs", raw_pairs)
        increment("relation_pair_checks", pair_checks)
        increment("relation_candidates", len(candidates))
        return RelationPlan(
            policy_set=dict(self.policy_set), raw_pairs=raw_pairs,
            blocked_pairs=raw_pairs - len(candidates), candidates=candidates,
        )

    @staticmethod
    def _candidate_payload(candidate: RelationCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "source_id": candidate.source_mention_id,
            "target_id": candidate.target_mention_id,
            "allowed_predicates": list(candidate.allowed_predicates),
            "allowed_directions": list(candidate.allowed_directions),
            "context_id": candidate.context_id,
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
        if candidate_budget_tokens <= 0:
            plan.deferred.extend({"candidate_id": item.candidate_id, "reason": "no_candidate_budget"}
                                 for item in plan.candidates)
            return []
        entities: dict[str, dict[str, str]] = {}
        for item in plan.candidates:
            entities.setdefault(item.source_mention_id, {
                "id": item.source_mention_id, "name": item.source_name, "type": item.source_type,
            })
            entities.setdefault(item.target_mention_id, {
                "id": item.target_mention_id, "name": item.target_name, "type": item.target_type,
            })
        fixed = {
            "policy": plan.policy_set["relation_classifier_policy_version"],
            "source_context": {"context_id": "source", "text": content},
            "entities": list(entities.values()),
            "relation_schema": _relation_types(logical_schema),
        }
        batches: list[RelationBatch] = []
        pending: list[RelationCandidate] = []

        def render(values: Iterable[RelationCandidate]) -> str:
            return json.dumps(
                {**fixed, "candidates": [self._candidate_payload(item) for item in values]},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            )

        def flush() -> None:
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

        for candidate in plan.candidates:
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
            result.append(RelationDecision(
                candidate_id=candidate.candidate_id, predicate=predicate,
                direction=direction, evidence=evidence, reason=reason,
            ))
        return result

    def classify_shadow(
        self,
        *,
        classifier: Any,
        model: str,
        model_max_tokens: int,
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
                plan.candidates.extend(additions)
                plan.blocked_pairs = max(0, plan.raw_pairs - len(plan.candidates))
                plan.gleaning_passes = 1
        output_reserve = min(
            max(128, Config.relation_classifier_output_reserve),
            max(128, int(model_max_tokens) // 3),
        )
        candidate_budget = max(
            0,
            int(model_max_tokens) - output_reserve - Config.relation_classifier_safety_margin,
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
        response_schema = {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "predicate": {"type": "string"},
                            "direction": {"type": "string"},
                            "evidence": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["candidate_id", "predicate", "direction", "evidence", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["decisions"],
            "additionalProperties": False,
        }
        for batch in batches:
            metrics["model_calls"] += 1
            metrics["input_tokens"] += batch.estimated_input_tokens
            try:
                raw = classifier.json_chat(
                    model=model,
                    system=(
                        "Classify every admitted relation candidate from SOURCE_CONTEXT only. "
                        "Return exactly one decision per candidate_id. Use an allowed predicate only when "
                        "an exact quote proves endpoints, meaning and direction; otherwise return NO_RELATION."
                    ),
                    prompt=batch.prompt,
                    schema=response_schema,
                    temperature=0,
                    max_tokens=batch.output_token_reserve,
                )
                metrics["output_tokens"] += estimate_tokens(
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
                )
                selected = [candidate_map[candidate_id] for candidate_id in batch.candidate_ids]
                validated = self.validate_decisions(
                    selected,
                    raw.get("decisions") if isinstance(raw, dict) else None,
                    content=content,
                    logical_schema=logical_schema,
                )
            except Exception as exc:
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
        plan.classified_count = len(decisions)
        metrics["classified"] = len(decisions)
        return ShadowClassificationResult(
            status="completed", plan=plan, decisions=decisions,
            batches=batch_audit, metrics=metrics,
        )

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
