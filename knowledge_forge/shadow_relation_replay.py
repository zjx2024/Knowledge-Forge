"""Fail-closed replay of persisted Shadow v2 model calls.

This module is intentionally generic.  It can recover a completed isolated
ledger after an on-disk-only code fingerprint change, but only when every
other frozen input component is unchanged.  It never calls a live completion
model and refuses materialization when the current runtime does not reproduce
the exact persisted request sequence.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from copy import deepcopy
from typing import Any


class ShadowReplayError(RuntimeError):
    """Persisted calls cannot be replayed under the current runtime."""


_REQUEST_FIELDS = (
    "model",
    "system",
    "prompt",
    "schema",
    "temperature",
    "context_budget_tokens",
    "max_output_tokens",
    "safety_margin_tokens",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _request_key(payload: dict[str, Any]) -> str:
    material = {
        field: payload.get(field)
        for field in _REQUEST_FIELDS
    }
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def assert_code_only_replay_compatible(
    source_run: dict[str, Any],
    current_fingerprint: Any,
) -> dict[str, Any]:
    """Require that only ``materialization_root.code_tree`` changed."""

    if str(source_run.get("status") or "") != "stale":
        raise ShadowReplayError("恢复源必须是 stale Shadow v2 run")
    recorded = deepcopy(dict(source_run.get("fingerprint_components") or {}))
    current = deepcopy(dict(current_fingerprint.components))
    recorded_root = recorded.get("materialization_root")
    current_root = current.get("materialization_root")
    if not isinstance(recorded_root, dict) or not isinstance(current_root, dict):
        raise ShadowReplayError("恢复源缺少完整 materialization_root")
    recorded_code = str(recorded_root.pop("code_tree", "") or "")
    current_code = str(current_root.pop("code_tree", "") or "")
    if not recorded_code or not current_code:
        raise ShadowReplayError("恢复源或当前运行缺少代码指纹")
    if recorded_code == current_code:
        raise ShadowReplayError("代码指纹未变化，不需要执行 stale 账本重放")
    if recorded != current:
        changed = sorted(
            key for key in set(recorded) | set(current)
            if recorded.get(key) != current.get(key)
        )
        raise ShadowReplayError(
            "除代码外的冻结输入发生变化，拒绝恢复: "
            + ",".join(changed)
        )
    return {
        "source_run_id": str(source_run["id"]),
        "source_fingerprint": str(source_run["input_fingerprint"]),
        "current_fingerprint": str(current_fingerprint.root),
        "recorded_code_tree": recorded_code,
        "current_code_tree": current_code,
        "changed_components": ["materialization_root.code_tree"],
    }


class PersistedShadowModelReplayGateway:
    """Replay one immutable run's audited model responses without model I/O."""

    _TERMINAL_MODEL_STATUSES = {
        "ACCEPTED", "SUPPORTED", "REJECTED", "UNCERTAIN",
    }

    def __init__(
        self,
        database: Any,
        source_run_id: str,
        *,
        runtime_model_gateway: Any,
    ):
        self.database = database
        self.source_run_id = str(source_run_id)
        self.runtime_model_gateway = runtime_model_gateway
        self.last_raw_response_text = ""
        self.last_call_diagnostics = None
        self.replay_provenance = {
            "mode": "persisted-shadow-model-call-replay-v1",
            "source_run_id": self.source_run_id,
            "live_model_calls": 0,
        }
        self._request_calls: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._verifier_calls: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._request_audits: dict[str, tuple[Any, ...]] = {}
        self._source_audit_rows: Counter[str] = Counter()
        self._source_counts: Counter[str] = Counter()
        self._consumed_counts: Counter[str] = Counter()
        self._misses: list[str] = []
        self._load()

    def _load(self) -> None:
        with self.database.connection() as conn:
            run = conn.execute(
                "SELECT * FROM shadow_relation_v2_runs WHERE id=%s",
                (self.source_run_id,),
            ).fetchone()
            if not run:
                raise ShadowReplayError("恢复源 Shadow v2 run 不存在")
            if str(run.get("status") or "") != "stale":
                raise ShadowReplayError("只能重放 stale Shadow v2 run")
            rows = conn.execute(
                """SELECT id,component,status,attempt_no,request_payload,
                          raw_response_text,parsed_response,error_type,error_message,
                          independent_call,created_at
                   FROM shadow_model_calls WHERE run_id=%s
                   ORDER BY created_at,id""",
                (self.source_run_id,),
            ).fetchall()
        if not rows:
            raise ShadowReplayError("恢复源没有持久化模型调用")
        for raw_row in rows:
            row = dict(raw_row)
            component = str(row.get("component") or "")
            if component not in {"proposer", "classifier", "verifier"}:
                continue
            self._source_audit_rows[component] += 1
            if component == "verifier":
                # Deterministic validation also persists a verification
                # terminal row for rejected/review-only hypotheses, but those
                # rows explicitly have independent_call=false and never
                # crossed the model boundary.
                if not bool(row.get("independent_call", True)):
                    continue
                self._source_counts[component] += 1
                payload = dict(row.get("parsed_response") or {})
                request_payload = dict(row.get("request_payload") or {})
                hypothesis_id = str(
                    payload.get("hypothesis_id")
                    or request_payload.get("hypothesis_id")
                    or ""
                )
                if not hypothesis_id:
                    raise ShadowReplayError(
                        "verifier 审计缺少 hypothesis_id"
                    )
                self._verifier_calls[hypothesis_id].append(row)
                continue
            request_payload = dict(row.get("request_payload") or {})
            if not request_payload.get("prompt"):
                raise ShadowReplayError(
                    f"{component} 审计缺少冻结请求"
                )
            boundary_key = _request_key(request_payload)
            audit_signature = (
                str(row.get("status") or ""),
                str(row.get("raw_response_text") or ""),
                _canonical_json(row.get("parsed_response")),
                str(row.get("error_type") or ""),
                str(row.get("error_message") or ""),
            )
            existing = self._request_audits.get(boundary_key)
            if existing is not None:
                if existing != audit_signature:
                    raise ShadowReplayError(
                        f"{component} 同一模型请求存在冲突审计结果"
                    )
                # A classifier call is audited once per pair in its batch.
                # These are provenance memberships, not additional model I/O.
                continue
            self._request_audits[boundary_key] = audit_signature
            self._source_counts[component] += 1
            self._request_calls[boundary_key].append(row)
        if not self._source_counts["classifier"]:
            raise ShadowReplayError("恢复源没有 classifier 审计调用")

    def effective_model_context(self, model: str) -> Any:
        return self.runtime_model_gateway.effective_model_context(model)

    @staticmethod
    def _error(row: dict[str, Any]) -> Exception:
        error_name = str(row.get("error_type") or "ShadowReplayRecordedError")
        message = str(
            row.get("error_message")
            or f"replayed persisted {row.get('status') or 'model'} failure"
        )
        error_type = type(error_name, (RuntimeError,), {})
        return error_type(message)

    @staticmethod
    def _response(row: dict[str, Any], component: str) -> Any:
        parsed = row.get("parsed_response")
        if component == "proposer" and isinstance(parsed, dict):
            return parsed.get("raw", parsed)
        if component == "classifier":
            return parsed
        if not isinstance(parsed, dict):
            raise ShadowReplayError("verifier 审计响应不是对象")
        spans = []
        for value in parsed.get("supported_evidence_spans") or []:
            if isinstance(value, dict):
                spans.append({
                    "start": int(value["start"]),
                    "end": int(value["end"]),
                })
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                # Verification audit rows persist the canonical domain form
                # as ``[start, end]`` pairs.  The model boundary schema uses
                # objects, so faithfully reconstruct that boundary shape.
                spans.append({
                    "start": int(value[0]),
                    "end": int(value[1]),
                })
            else:
                raise ShadowReplayError("verifier 审计证据 span 无效")
        return {
            "verdict": str(parsed.get("verdict") or ""),
            "reason_code": str(parsed.get("reason_code") or ""),
            "supported_evidence_spans": spans,
        }

    def json_chat(self, **kwargs: Any) -> Any:
        """Return the next exact persisted response; never call Ollama."""

        self.last_raw_response_text = ""
        self.last_call_diagnostics = None
        system = str(kwargs.get("system") or "")
        prompt = str(kwargs.get("prompt") or "")
        if "proposer" in system.casefold():
            component = "proposer"
            lookup_key = _request_key(dict(kwargs))
            queue = self._request_calls.get(lookup_key)
        elif "classifier" in system.casefold():
            component = "classifier"
            lookup_key = _request_key(dict(kwargs))
            queue = self._request_calls.get(lookup_key)
        elif "verifier" in system.casefold():
            try:
                payload = json.loads(prompt)
            except json.JSONDecodeError as exc:
                raise ShadowReplayError("当前 verifier prompt 不是 JSON") from exc
            hypothesis_id = str(payload.get("hypothesis_id") or "")
            queue = self._verifier_calls.get(hypothesis_id)
            component = "verifier"
            lookup_key = hypothesis_id
        else:
            raise ShadowReplayError("恢复期间出现未知模型组件")
        if not queue:
            differences = []
            for candidate_queue in self._request_calls.values():
                if not candidate_queue:
                    continue
                candidate = candidate_queue[0]
                if str(candidate.get("component") or "") != component:
                    continue
                candidate_payload = dict(
                    candidate.get("request_payload") or {}
                )
                changed = [
                    field for field in _REQUEST_FIELDS
                    if _canonical_json(candidate_payload.get(field))
                    != _canonical_json(kwargs.get(field))
                ]
                differences.append(changed)
            closest = min(differences, key=len) if differences else []
            marker = (
                f"{component}:{lookup_key}:"
                f"closest_changed={','.join(closest) or 'none'}"
            )
            self._misses.append(marker)
            raise ShadowReplayError(
                f"当前运行请求无法在冻结账本中精确匹配: {marker}"
            )
        row = queue.popleft()
        self._consumed_counts[component] += 1
        self.last_raw_response_text = str(row.get("raw_response_text") or "")
        if str(row.get("status") or "") not in self._TERMINAL_MODEL_STATUSES:
            raise self._error(row)
        return self._response(row, component)

    def assert_replay_complete(self) -> None:
        remaining = sum(
            len(queue) for queue in self._request_calls.values()
        ) + sum(len(queue) for queue in self._verifier_calls.values())
        if self._misses or remaining:
            raise ShadowReplayError(
                "冻结模型账本未被当前运行精确重放，拒绝物化: "
                f"misses={len(self._misses)},remaining={remaining},"
                f"samples={self._misses[:3]}"
            )

    def replay_summary(self) -> dict[str, Any]:
        return {
            **self.replay_provenance,
            "source_audit_rows": dict(self._source_audit_rows),
            "source_calls": dict(self._source_counts),
            "consumed_calls": dict(self._consumed_counts),
            "misses": len(self._misses),
            "miss_samples": self._misses[:10],
            "remaining_calls": (
                sum(len(queue) for queue in self._request_calls.values())
                + sum(len(queue) for queue in self._verifier_calls.values())
            ),
        }
