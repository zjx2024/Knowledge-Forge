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


def _structured_prompt_key(payload: Any) -> str:
    """Key a JSON prompt whose audit stores the parsed request object."""

    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def assert_code_only_replay_compatible(
    source_run: dict[str, Any],
    current_fingerprint: Any,
) -> dict[str, Any]:
    """Require that only the semantic-code identity changed.

    ``materialization_root.treatment`` historically embedded ``code_tree``.
    It is therefore a derived companion value rather than an independent
    semantic input: candidate/prompt/verifier/policy/threshold components are
    compared independently below.  This compatibility bridge permits a
    persisted-call replay across the old representation without allowing a
    policy or frozen-input change to reuse an old ledger.
    """

    if str(source_run.get("status") or "") not in {"stale", "ready"}:
        raise ShadowReplayError("恢复源必须是 ready 或 stale Shadow v2 run")
    recorded = deepcopy(dict(source_run.get("fingerprint_components") or {}))
    current = deepcopy(dict(current_fingerprint.components))
    recorded_root = recorded.get("materialization_root")
    current_root = current.get("materialization_root")
    if not isinstance(recorded_root, dict) or not isinstance(current_root, dict):
        raise ShadowReplayError("恢复源缺少完整 materialization_root")
    recorded_code = str(recorded_root.pop("code_tree", "") or "")
    current_code = str(current_root.pop("code_tree", "") or "")
    # See the docstring.  The surrounding immutable fingerprint independently
    # carries every non-code constituent of this digest.
    recorded_root.pop("treatment", None)
    current_root.pop("treatment", None)
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
        "ACCEPTED", "COMPLETED", "SUPPORTED", "REJECTED", "UNCERTAIN",
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
        self.last_parsed_json_response = None
        self.replay_provenance = {
            "mode": "persisted-shadow-model-call-replay-v1",
            "source_run_id": self.source_run_id,
            "live_model_calls": 0,
        }
        self._request_calls: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._source_request_calls: dict[
            str, deque[tuple[str, dict[str, Any]]]
        ] = defaultdict(deque)
        self._structured_prompt_calls: dict[
            str, deque[tuple[str, dict[str, Any]]]
        ] = defaultdict(deque)
        self._verifier_calls: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
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
            if str(run.get("status") or "") not in {"stale", "ready"}:
                raise ShadowReplayError("只能重放 ready 或 stale Shadow v2 run")
            rows = conn.execute(
                """SELECT id,component,status,attempt_no,idempotency_key,request_payload,
                          raw_response_text,parsed_response,error_type,error_message,
                          independent_call,created_at
                   FROM shadow_model_calls WHERE run_id=%s
                   ORDER BY created_at,id""",
                (self.source_run_id,),
            ).fetchall()
        if not rows:
            raise ShadowReplayError("恢复源没有持久化模型调用")
        # Classifier rows are persisted once per pair in a batch so each pair
        # keeps provenance.  They are not independent model invocations.  A
        # partial parser failure can legitimately mark one member retryable
        # while the same raw batch response accepts its other members.  Merge
        # those membership rows back to their model boundary before replaying.
        classifier_boundaries: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        classifier_order: list[tuple[str, str, str, str]] = []
        direct_request_rows: list[tuple[str, str, dict[str, Any]]] = []
        for raw_row in rows:
            row = dict(raw_row)
            component = str(row.get("component") or "")
            if not component:
                continue
            self._source_audit_rows[component] += 1
            if component not in {"proposer", "classifier", "verifier"}:
                # Source-first and batched-verifier calls use ordinary JSON
                # model boundaries too.  Several early audit envelopes store
                # the parsed JSON request rather than the rendered prompt;
                # index those by canonical JSON for exact replay.
                request_payload = dict(row.get("request_payload") or {})
                if not request_payload:
                    continue
                self._source_counts[component] += 1
                if request_payload.get("prompt"):
                    self._source_request_calls[
                        _request_key(request_payload)
                    ].append((component, row))
                else:
                    self._structured_prompt_calls[
                        _structured_prompt_key(request_payload)
                    ].append((component, row))
                continue
            if component == "verifier":
                # Deterministic validation also persists a verification
                # terminal row for rejected/review-only hypotheses, but those
                # rows explicitly have independent_call=false and never
                # crossed the model boundary. Batch-verifier decisions are
                # likewise persisted per hypothesis with independent semantic
                # judgment, but their one transport call is already represented
                # by the separately audited ``batch_verifier`` row. Current
                # terminal envelopes are recognizable by their canonical
                # attempt fields and must not become duplicate replay calls.
                if not bool(row.get("independent_call", True)):
                    continue
                payload = dict(row.get("parsed_response") or {})
                request_payload = dict(row.get("request_payload") or {})
                if (
                    not request_payload.get("prompt")
                    and request_payload.get("stable_attempt_id")
                    and request_payload.get("verdict")
                ):
                    continue
                self._source_counts[component] += 1
                hypothesis_id = str(
                    payload.get("hypothesis_id")
                    or request_payload.get("hypothesis_id")
                    or ""
                )
                if not hypothesis_id:
                    # The v2 verifier envelope predates an explicit
                    # hypothesis_id field in its JSON request.  Its immutable
                    # idempotency key nevertheless records exactly the same
                    # identity as ``verifier:<hypothesis-id>:<attempt>``.
                    idempotency_key = str(row.get("idempotency_key") or "")
                    parts = idempotency_key.split(":")
                    if len(parts) >= 3 and parts[0] == "verifier":
                        hypothesis_id = parts[1]
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
            if component != "classifier":
                direct_request_rows.append((component, boundary_key, row))
                continue
            response_key = hashlib.sha256(
                (
                    str(row.get("raw_response_text") or "")
                    + "\0"
                    + _canonical_json(row.get("parsed_response"))
                ).encode("utf-8")
            ).hexdigest()
            key = (component, boundary_key, response_key, str(
                row.get("request_hash") or "",
            ))
            existing = classifier_boundaries.get(key)
            if existing is None:
                classifier_boundaries[key] = row
                classifier_order.append(key)
                continue
            # A successful member proves the raw batch response crossed the
            # model boundary.  Retain it rather than a member-local parser
            # failure so the current parser receives the original batch JSON.
            if (
                str(existing.get("status") or "")
                not in self._TERMINAL_MODEL_STATUSES
                and str(row.get("status") or "")
                in self._TERMINAL_MODEL_STATUSES
            ):
                classifier_boundaries[key] = row
        for component, boundary_key, row in direct_request_rows:
            self._source_counts[component] += 1
            self._request_calls[boundary_key].append(row)
        for key in classifier_order:
            row = classifier_boundaries[key]
            request_payload = dict(row.get("request_payload") or {})
            self._source_counts["classifier"] += 1
            self._request_calls[_request_key(request_payload)].append(row)
        if not sum(self._source_counts.values()):
            raise ShadowReplayError("恢复源没有可重放的模型审计调用")

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
            "forward_entailment": str(
                parsed.get("forward_entailment") or ""
            ),
            "reverse_entailment": str(
                parsed.get("reverse_entailment") or ""
            ),
            "subject_role_supported": bool(
                parsed.get("subject_role_supported", False)
            ),
            "object_role_supported": bool(
                parsed.get("object_role_supported", False)
            ),
            "supported_evidence_spans": spans,
        }

    @staticmethod
    def _source_response(row: dict[str, Any]) -> Any:
        """Reconstruct the response at the source-model transport boundary."""

        raw_text = str(row.get("raw_response_text") or "").strip()
        if raw_text:
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                pass
        return row.get("parsed_response")

    def json_chat(self, **kwargs: Any) -> Any:
        """Return the next exact persisted response; never call Ollama."""

        self.last_raw_response_text = ""
        self.last_call_diagnostics = None
        self.last_parsed_json_response = None
        system = str(kwargs.get("system") or "")
        prompt = str(kwargs.get("prompt") or "")
        source_queue = self._source_request_calls.get(_request_key(dict(kwargs)))
        if source_queue:
            component, row = source_queue.popleft()
            self._consumed_counts[component] += 1
            self.last_raw_response_text = str(
                row.get("raw_response_text") or ""
            )
            if str(row.get("status") or "") not in self._TERMINAL_MODEL_STATUSES:
                raise self._error(row)
            parsed = self._source_response(row)
            self.last_parsed_json_response = parsed
            return parsed
        structured_queue: deque[tuple[str, dict[str, Any]]] | None = None
        try:
            structured_prompt = json.loads(prompt)
        except json.JSONDecodeError:
            structured_prompt = None
        if structured_prompt is not None:
            structured_queue = self._structured_prompt_calls.get(
                _structured_prompt_key(structured_prompt),
            )
        if structured_queue:
            component, row = structured_queue.popleft()
            self._consumed_counts[component] += 1
            self.last_raw_response_text = str(
                row.get("raw_response_text") or ""
            )
            if str(row.get("status") or "") not in self._TERMINAL_MODEL_STATUSES:
                raise self._error(row)
            parsed = self._source_response(row)
            self.last_parsed_json_response = parsed
            return parsed
        if "proposer" in system.casefold():
            component = "proposer"
            lookup_key = _request_key(dict(kwargs))
            queue = self._request_calls.get(lookup_key)
        elif "classifier" in system.casefold():
            component = "classifier"
            lookup_key = _request_key(dict(kwargs))
            queue = self._request_calls.get(lookup_key)
        elif "semantic contrast judge" in system.casefold():
            try:
                payload = json.loads(prompt)
            except json.JSONDecodeError as exc:
                raise ShadowReplayError(
                    "当前 semantic contrast prompt 不是 JSON"
                ) from exc
            hypothesis_id = str(payload.get("hypothesis_id") or "")
            queue = self._verifier_calls.get(hypothesis_id)
            component = "verifier_semantic_contrast"
            lookup_key = hypothesis_id
        elif "role alignment judge" in system.casefold():
            try:
                payload = json.loads(prompt)
            except json.JSONDecodeError as exc:
                raise ShadowReplayError(
                    "当前 role alignment prompt 不是 JSON"
                ) from exc
            hypothesis_id = str(payload.get("hypothesis_id") or "")
            queue = self._verifier_calls.get(hypothesis_id)
            component = "verifier_role_alignment"
            lookup_key = hypothesis_id
        elif "role analyst" in system.casefold():
            try:
                payload = json.loads(prompt)
            except json.JSONDecodeError as exc:
                raise ShadowReplayError(
                    "当前 role probe prompt 不是 JSON"
                ) from exc
            hypothesis_id = str(payload.get("hypothesis_id") or "")
            queue = self._verifier_calls.get(hypothesis_id)
            component = "verifier_role_probe"
            lookup_key = hypothesis_id
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
            marker = "unknown:" + _request_key(dict(kwargs))
            self._misses.append(marker)
            raise ShadowReplayError(
                f"恢复期间出现未记录的模型组件: {marker}"
            )
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
        row = (
            queue[0]
            if component in {
                "verifier_role_probe", "verifier_role_alignment",
                "verifier_semantic_contrast",
            }
            else queue.popleft()
        )
        self._consumed_counts[component] += 1
        self.last_raw_response_text = str(row.get("raw_response_text") or "")
        if str(row.get("status") or "") not in self._TERMINAL_MODEL_STATUSES:
            raise self._error(row)
        if component == "verifier_role_probe":
            parsed = dict(row.get("parsed_response") or {})
            subject_role = str(parsed.get("subject_context_role") or "").strip()
            object_role = str(parsed.get("object_context_role") or "").strip()
            if not subject_role or not object_role:
                raise ShadowReplayError("冻结 verifier 审计缺少盲角色分析")
            response = {
                "subject_role": subject_role,
                "object_role": object_role,
            }
            self.last_parsed_json_response = response
            return response
        if component == "verifier_semantic_contrast":
            parsed = dict(row.get("parsed_response") or {})
            verdict = str(parsed.get("verdict") or row.get("status") or "")
            response = {
                "semantic_contrast": (
                    "DEFINITION" if verdict == "SUPPORTED" else "UNCERTAIN"
                ),
            }
            self.last_parsed_json_response = response
            return response
        if component == "verifier_role_alignment":
            parsed = dict(row.get("parsed_response") or {})
            forward = str(
                parsed.get("forward_role_alignment")
                or parsed.get("subject_role_alignment")
                or ""
            ).strip()
            reverse = str(
                parsed.get("reverse_role_alignment") or ""
            ).strip()
            if not reverse and forward:
                reverse = "UNCERTAIN"
            if not forward or not reverse:
                raise ShadowReplayError("冻结 verifier 审计缺少角色对齐判定")
            try:
                alignment_payload = json.loads(prompt)
            except json.JSONDecodeError:
                alignment_payload = {}
            symmetric = bool(alignment_payload.get("symmetric"))
            will_verify = (
                (symmetric and "ALIGNED" in {forward, reverse})
                or (
                    not symmetric
                    and ((forward == "ALIGNED") ^ (reverse == "ALIGNED"))
                )
            )
            if not will_verify:
                queue.popleft()
            response = {
                "forward_alignment": forward,
                "reverse_alignment": reverse,
            }
            self.last_parsed_json_response = response
            return response
        response = self._response(row, component)
        self.last_parsed_json_response = response
        return response

    def assert_replay_complete(self) -> None:
        remaining = sum(
            len(queue) for queue in self._request_calls.values()
        ) + sum(len(queue) for queue in self._source_request_calls.values()) + sum(
            len(queue) for queue in self._verifier_calls.values()
        ) + sum(
            len(queue) for queue in self._structured_prompt_calls.values()
        )
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
                + sum(
                    len(queue) for queue in self._source_request_calls.values()
                )
                + sum(len(queue) for queue in self._verifier_calls.values())
                + sum(
                    len(queue)
                    for queue in self._structured_prompt_calls.values()
                )
            ),
        }
