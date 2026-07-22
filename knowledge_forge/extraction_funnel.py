"""Conserved, fail-closed state accounting for entity and relation extraction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable


class FunnelInvariantError(ValueError):
    """A funnel item violated registration or terminal-state invariants."""


DEFAULT_TERMINALS: dict[str, frozenset[str]] = {
    "raw_entity_mentions": frozenset({
        "admitted_mentions", "rejected_mentions", "uncertain_mentions",
    }),
    "entity_resolution_candidates": frozenset({
        "resolved", "new_shadow_canonical", "uncertain", "retryable_failed",
    }),
    "bounded_relation_pairs": frozenset({
        "specific_relation", "NO_RELATION", "OTHER_UNSUPPORTED", "uncertain",
        "retryable_failed", "rejected",
    }),
    "sanitized": frozenset({"accepted", "rejected"}),
    "endpoints_resolved": frozenset({"resolved", "unresolved"}),
    "verified": frozenset({"verified", "rejected", "quarantined"}),
    "publishable": frozenset({"published", "not_publishable"}),
}


@dataclass(frozen=True)
class TerminalRecord:
    state: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ExtractionFunnel:
    """Track every stable item ID through exactly one terminal per stage."""

    def __init__(
        self,
        *,
        run_id: str,
        terminal_states: dict[str, Iterable[str]] | None = None,
    ) -> None:
        self.run_id = str(run_id)
        definitions = terminal_states or DEFAULT_TERMINALS
        self._terminal_states = {
            str(stage): frozenset(str(value) for value in values)
            for stage, values in definitions.items()
        }
        self._registered: dict[str, set[str]] = {}
        self._terminals: dict[str, dict[str, TerminalRecord]] = {}

    def _require_stage(self, stage: str) -> str:
        stage = str(stage)
        if stage not in self._terminal_states:
            raise FunnelInvariantError(f"unknown funnel stage: {stage}")
        return stage

    def register(self, stage: str, item_ids: Iterable[object]) -> None:
        stage = self._require_stage(stage)
        registered = self._registered.setdefault(stage, set())
        for raw_item_id in item_ids:
            item_id = str(raw_item_id)
            if not item_id:
                raise FunnelInvariantError(f"{stage} item id cannot be empty")
            registered.add(item_id)

    def terminate(
        self,
        stage: str,
        item_id: object,
        state: str,
        *,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        stage = self._require_stage(stage)
        item_id = str(item_id)
        state = str(state)
        if state not in self._terminal_states[stage]:
            raise FunnelInvariantError(f"terminal state {state} is not allowed for {stage}")
        if item_id not in self._registered.get(stage, set()):
            raise FunnelInvariantError(f"{stage} item {item_id} is not registered")
        terminals = self._terminals.setdefault(stage, {})
        if item_id in terminals:
            raise FunnelInvariantError(
                f"{stage} item {item_id} already has terminal {terminals[item_id].state}"
            )
        terminals[item_id] = TerminalRecord(
            state=state,
            reason=str(reason or ""),
            metadata=dict(metadata or {}),
        )

    def report(self) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        failures: list[str] = []
        for stage in sorted(self._registered):
            registered = sorted(self._registered[stage])
            terminals = self._terminals.get(stage, {})
            missing = sorted(set(registered) - set(terminals))
            terminal_ids: dict[str, list[str]] = {}
            terminal_counts = Counter(record.state for record in terminals.values())
            for item_id, record in sorted(terminals.items()):
                terminal_ids.setdefault(record.state, []).append(item_id)
            for item_id in missing:
                failures.append(f"{stage}:missing_terminal:{item_id}")
            stages[stage] = {
                "input_count": len(registered),
                "input_ids": registered,
                "terminal_count": len(terminals),
                "terminal_counts": dict(sorted(terminal_counts.items())),
                "terminal_ids": dict(sorted(terminal_ids.items())),
                "missing_terminal_ids": missing,
                "failure_reasons": {
                    item_id: record.reason
                    for item_id, record in sorted(terminals.items())
                    if record.reason
                },
                "conserved": len(registered) == len(terminals) and not missing,
            }
        failures.sort()
        return {
            "run_id": self.run_id,
            "coverage_complete": not failures,
            "failures": failures,
            "stages": stages,
        }


def relation_shadow_funnel(
    *,
    run_id: str,
    plan: Any,
    persisted_work: Iterable[dict[str, Any]],
    decisions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Build a conserved terminal report from one persisted relation shadow run."""

    candidate_ids = {
        str(item.candidate_id)
        for item in list(getattr(plan, "candidates", ()) or ())
    }
    for item in [
        *(list(getattr(plan, "overflow", ()) or ())),
        *(list(getattr(plan, "deferred", ()) or ())),
    ]:
        candidate_id = str(item.get("candidate_id") or "")
        if candidate_id:
            candidate_ids.add(candidate_id)

    funnel = ExtractionFunnel(run_id=run_id)
    funnel.register("bounded_relation_pairs", sorted(candidate_ids))

    decision_by_id: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        candidate_id = str(decision.get("candidate_id") or "")
        if not candidate_id or candidate_id not in candidate_ids:
            raise FunnelInvariantError(
                f"relation decision candidate {candidate_id or '<empty>'} is not registered"
            )
        if candidate_id in decision_by_id:
            raise FunnelInvariantError(f"duplicate relation decision: {candidate_id}")
        decision_by_id[candidate_id] = dict(decision)

    work_by_candidate: dict[str, dict[str, Any]] = {}
    for work in persisted_work:
        for raw_candidate_id in list(work.get("candidate_ids") or []):
            candidate_id = str(raw_candidate_id)
            previous = work_by_candidate.get(candidate_id)
            if previous is not None and str(previous.get("id") or "") != str(work.get("id") or ""):
                raise FunnelInvariantError(f"candidate appears in multiple work items: {candidate_id}")
            work_by_candidate[candidate_id] = work

    for candidate_id in sorted(candidate_ids):
        decision = decision_by_id.get(candidate_id)
        if decision is not None:
            status = str(decision.get("status") or "")
            predicate = str(decision.get("predicate") or "")
            if status in {"uncertain", "retryable_failed", "rejected"}:
                terminal = status
            elif predicate == "NO_RELATION":
                terminal = "NO_RELATION"
            elif predicate == "OTHER_UNSUPPORTED":
                terminal = "OTHER_UNSUPPORTED"
            else:
                terminal = "specific_relation"
            funnel.terminate(
                "bounded_relation_pairs",
                candidate_id,
                terminal,
                reason=str(decision.get("reason") or ""),
                metadata={"predicate": predicate},
            )
            continue

        work = work_by_candidate.get(candidate_id) or {}
        work_status = str(work.get("status") or "")
        if work_status == "terminal_failed":
            funnel.terminate(
                "bounded_relation_pairs",
                candidate_id,
                "retryable_failed",
                reason=str(work.get("error_message") or "terminal failure"),
            )
        elif work_status == "quarantined":
            funnel.terminate(
                "bounded_relation_pairs",
                candidate_id,
                "rejected",
                reason=str(work.get("error_message") or "quarantined"),
            )

    return funnel.report()
