"""Fail-closed runtime orchestration for the isolated Shadow relation v2 path."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from . import relation_shadow_v2 as v2
from .config import Config
from .quality import canonical_digest


SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION = "shadow-relation-candidate-v3-schema-metadata"
SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION = "shadow-relation-classifier-v4-compact-json-retry-budget"
SHADOW_RELATION_V2_NORMALIZATION_VERSION = "shadow-relation-normalization-v3-schema-metadata"
SHADOW_RELATION_V2_MATERIALIZATION_VERSION = "shadow-relation-materialization-v3-schema-metadata"
SHADOW_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT = (
    "You are the Shadow relation v2 classifier. Follow the supplied "
    "pair contract exactly and return only one compact schema-valid "
    "JSON value. Do not include <think>, reasoning, markdown, "
    "commentary, or any text outside JSON."
)
SHADOW_RELATION_V2_POLICY_SET = {
    "candidate_policy_version": SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
    "classifier_policy_version": SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION,
    "classifier_prompt_version": v2.SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
    "classifier_schema_version": v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
    "classifier_parser_version": v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
    "verifier_prompt_version": v2.SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
    "verifier_schema_version": v2.SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
    "verifier_parser_version": v2.SHADOW_RELATION_VERIFIER_PARSER_VERSION,
    "normalization_version": SHADOW_RELATION_V2_NORMALIZATION_VERSION,
    "materialization_version": SHADOW_RELATION_V2_MATERIALIZATION_VERSION,
    "fallback": "forbidden",
}


class ShadowRelationV2Runtime:
    """Run pair planning through isolated materialization without legacy fallback."""

    def __init__(
        self,
        database: Any,
        model_gateway: Any,
        *,
        default_context_tokens: int = 32768,
    ):
        self.db = database
        self.models = model_gateway
        self.default_context_tokens = int(default_context_tokens)

    def _model_revision(self, model: str) -> str:
        runtime = self.models.effective_model_context(model)
        revision = str(runtime.digest or "").strip()
        if not revision:
            raise RuntimeError(
                "Shadow relation v2 无法冻结模型 digest，拒绝运行"
            )
        return revision

    def _context(self, project_id: int, namespace_id: str) -> dict[str, Any]:
        project = self.db.get_project(project_id)
        if not project:
            raise LookupError("项目不存在")
        documents = [
            item for item in self.db.list_documents(project_id)
            if item.get("status") == "indexed"
            and item.get("active_revision_id")
            and not item.get("is_deleted")
        ]
        documents.sort(key=lambda item: int(item["id"]))
        chunks = [
            chunk
            for document in documents
            for chunk in self.db.list_document_chunks(int(document["id"]))
        ]
        chunks.sort(key=lambda item: (
            int(item["document_id"]),
            int(item.get("chunk_index") or 0),
            int(item["id"]),
        ))
        schema_record = self.db.get_active_schema(project_id)
        if not schema_record:
            raise RuntimeError("Shadow relation v2 缺少 active Schema")
        schema = dict(schema_record.get("schema_json") or {})
        mentions = self.db.list_entity_shadow_mentions(namespace_id)
        snapshot = self.db.load_entity_shadow_snapshot(namespace_id)
        assignments = list(snapshot.get("assignments") or [])
        assignment_by_mention = {
            str(item["mention_key"]): item for item in assignments
        }
        if len(assignment_by_mention) != len(assignments):
            raise RuntimeError("Shadow relation v2 assignment occurrence 不唯一")
        missing = sorted(
            str(item["mention_key"]) for item in mentions
            if str(item["mention_key"]) not in assignment_by_mention
        )
        if missing:
            raise RuntimeError(
                "Shadow relation v2 mention 缺少冻结 assignment: "
                + ",".join(missing[:10])
            )
        chunk_by_id = {int(item["id"]): item for item in chunks}
        unknown_chunks = sorted({
            int(item["chunk_id"]) for item in mentions
            if int(item["chunk_id"]) not in chunk_by_id
        })
        if unknown_chunks:
            raise RuntimeError(
                "Shadow relation v2 mention 指向非当前 Chunk: "
                + ",".join(str(value) for value in unknown_chunks[:10])
            )
        model = str(project.get("llm_model") or Config.default_llm_model)
        model_revision = self._model_revision(model)
        assignment_payload = [{
            "mention_key": str(item["mention_key"]),
            "canonical_key": str(item["canonical_key"]),
            "entity_type": str(item.get("entity_type") or ""),
            "policy_version": str(item.get("policy_version") or ""),
            "revision_id": str(item.get("revision_id") or ""),
        } for item in sorted(
            assignments, key=lambda value: str(value["mention_key"]),
        )]
        predicate_payload = v2.predicate_policy_fingerprint_payload(schema)
        document_payload = []
        for document in documents:
            document_chunks = [
                item for item in chunks
                if int(item["document_id"]) == int(document["id"])
            ]
            document_payload.append({
                "document_id": int(document["id"]),
                "revision_id": str(document["active_revision_id"]),
                "content_hash": str(document.get("doc_hash") or ""),
                "chunks": [{
                    "chunk_id": int(item["id"]),
                    "content_hash": str(item.get("content_hash") or ""),
                } for item in document_chunks],
            })
        mention_payload = [{
            "occurrence_key": str(item["mention_key"]),
            "span": [int(item["evidence_start"]), int(item["evidence_end"])],
            "content_hash": str(
                chunk_by_id[int(item["chunk_id"])].get("content_hash") or ""
            ),
            "canonical_id": str(
                assignment_by_mention[str(item["mention_key"])]["canonical_key"]
            ),
            "assignment_version": str(
                assignment_by_mention[str(item["mention_key"])].get(
                    "policy_version"
                ) or ""
            ),
        } for item in sorted(
            mentions, key=lambda value: str(value["mention_key"]),
        )]
        code_hasher = hashlib.sha256()
        for path in (Path(__file__), Path(str(v2.__file__))):
            code_hasher.update(path.read_bytes())
        candidate = {
            "policy_version": SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
            "parameters": {
                "sentence_window": Config.relation_sentence_window,
                "max_char_distance": Config.relation_max_char_distance,
                "batch_size": Config.relation_classifier_max_candidates,
                "soft_cues": "schema-configured-nonbinding",
            },
            "budget": Config.relation_max_candidates_per_chunk,
        }
        classifier = {
            "prompt_version": v2.SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
            "schema_version": v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
            "parser_version": v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
            "model": model,
            "model_revision": model_revision,
        }
        verifier = {
            "prompt_version": v2.SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
            "schema_version": v2.SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
            "parser_version": v2.SHADOW_RELATION_VERIFIER_PARSER_VERSION,
            "model": model,
            "model_revision": model_revision,
        }
        policy_root = canonical_digest({
            "candidate": candidate,
            "classifier": classifier,
            "verifier": verifier,
            "relation_policy": SHADOW_RELATION_V2_POLICY_SET,
        })
        fingerprint = v2.build_relation_materialization_fingerprint(
            source_generation=int(project.get("graph_generation") or 0),
            documents=document_payload,
            mention_occurrences=mention_payload,
            entity_assignment={
                "snapshot_id": namespace_id,
                "version": str(
                    (snapshot.get("namespace") or {}).get("config_hash")
                    or "entity-shadow-assignment-v1"
                ),
                "hash": canonical_digest(assignment_payload),
            },
            schema={
                "version": int(schema_record.get("version") or 1),
                "hash": canonical_digest(schema),
                "predicate_metadata_hash": canonical_digest(predicate_payload),
            },
            candidate=candidate,
            classifier=classifier,
            verifier=verifier,
            thresholds={
                "max_attempts": Config.relation_shadow_max_attempts,
                "classifier_output_reserve":
                    Config.relation_classifier_output_reserve,
                "classifier_safety_margin":
                    Config.relation_classifier_safety_margin,
            },
            normalization={
                "version": SHADOW_RELATION_V2_NORMALIZATION_VERSION,
                "symmetric_rule": "sort-canonical-endpoints",
                "inverse_rule": "declared-canonical-forward-only",
            },
            publish_policy={
                "version": SHADOW_RELATION_V2_MATERIALIZATION_VERSION,
                "mode_source": "versioned-schema-publish_mode",
                "unsupported": "never-publish",
            },
            quality_gate={
                "version": "shadow-relation-release-gate-v2",
                "relation_precision_noninferiority_delta": 0.01,
                "confidence_method": "wilson-95",
                "minimum_gold_facts": 200,
            },
            materialization_root={
                "code_tree": code_hasher.hexdigest(),
                "policy_root": policy_root,
            },
        )
        return {
            "project": project,
            "documents": documents,
            "chunks": chunks,
            "chunk_by_id": chunk_by_id,
            "schema": schema,
            "mentions": mentions,
            "assignments": assignments,
            "assignment_by_mention": assignment_by_mention,
            "model": model,
            "model_revision": model_revision,
            "fingerprint": fingerprint,
        }

    def _classify(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        max_output_tokens: int | None = None,
        opportunities: list[v2.PairOpportunity],
        content: str,
        schema: dict[str, Any],
    ) -> Any:
        output_budget = int(
            max_output_tokens or Config.relation_classifier_output_reserve
        )
        raw = self.models.json_chat(
            model=model,
            system=SHADOW_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT,
            prompt=v2.classifier_prompt(
                opportunities, content=content, logical_schema=schema,
            ),
            schema=v2.classifier_response_schema(opportunities),
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=output_budget,
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        def normalize_negative_decision(value: Any) -> Any:
            if (
                isinstance(value, dict)
                and value.get("decision") in {v2.NO_RELATION, v2.OTHER_UNSUPPORTED}
                and value.get("hypotheses") is None
            ):
                return {**value, "hypotheses": []}
            return value

        def bind_singleton_pair(value: Any) -> Any:
            value = normalize_negative_decision(value)
            if (
                len(opportunities) == 1
                and isinstance(value, dict)
                and {"decision", "hypotheses"} <= set(value)
            ):
                return {
                    **value,
                    "pair_id": opportunities[0].pair_identity,
                }
            return value

        def normalize_decision_list(values: Any) -> Any:
            if len(opportunities) == 1 and values == []:
                return [{
                    "pair_id": opportunities[0].pair_identity,
                    "decision": v2.NO_RELATION,
                    "hypotheses": [],
                    "reason_code": "EMPTY_SINGLETON_CLASSIFIER_RESPONSE",
                }]
            return [bind_singleton_pair(value) for value in values]

        if isinstance(raw, dict):
            for key in ("decisions", "items", "result"):
                if isinstance(raw.get(key), list):
                    return normalize_decision_list(raw[key])
        if isinstance(raw, list):
            return normalize_decision_list(raw)
        raw = bind_singleton_pair(raw)
        if (
            len(opportunities) == 1
            and isinstance(raw, dict)
            and "pair_id" not in raw
            and {"decision", "hypotheses"} <= set(raw)
        ):
            return [{
                "pair_id": opportunities[0].pair_identity,
                **raw,
            }]
        if (
            isinstance(raw, dict)
            and {"pair_id", "decision", "hypotheses"} <= set(raw)
        ):
            return [raw]
        return raw

    @staticmethod
    def _classifier_output_budget(
        *, context_budget_tokens: int, attempt_no: int,
    ) -> int:
        base = max(128, int(Config.relation_classifier_output_reserve))
        requested = base * (2 ** max(0, int(attempt_no) - 1))
        context_cap = max(
            base,
            int(context_budget_tokens)
            - int(Config.relation_classifier_safety_margin)
            - 1024,
        )
        return max(128, min(requested, context_cap, 8192))

    def _verify(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> Any:
        return self.models.json_chat(
            model=model,
            system=(
                "You are the independent Shadow relation v2 evidence verifier. "
                "Judge the complete ordered triple only from the frozen context."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=min(
                1024, Config.relation_classifier_output_reserve,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    @staticmethod
    def _occurrence_canonical(
        opportunity: v2.PairOpportunity,
        occurrence_key: str,
    ) -> str:
        if occurrence_key == opportunity.left_occurrence_key:
            return str(opportunity.left_canonical_id)
        if occurrence_key == opportunity.right_occurrence_key:
            return str(opportunity.right_canonical_id)
        raise RuntimeError("v2 evidence occurrence 不属于 pair")

    def _endpoint_rows(
        self,
        *,
        run_id: str,
        normalization: v2.RelationNormalizationResult,
        materialization_decisions: list[v2.RelationMaterializationDecision],
        opportunities: dict[str, v2.PairOpportunity],
        hypotheses: dict[str, v2.RelationHypothesis],
        chunks: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        eligible = {
            str(item.fact_id) for item in materialization_decisions
            if item.eligible
        }
        evidence_by_fact: dict[str, list[v2.RelationFactEvidence]] = {}
        for evidence in normalization.evidence:
            evidence_by_fact.setdefault(str(evidence.fact_id), []).append(evidence)
        rows = []
        for fact in normalization.facts:
            if fact.fact_id not in eligible:
                continue
            evidence_values = sorted(
                evidence_by_fact.get(fact.fact_id) or [],
                key=lambda item: item.evidence_key,
            )
            if not evidence_values or any(
                not item.verifier_supported or not item.evidence_spans
                for item in evidence_values
            ):
                raise RuntimeError(
                    "v2 可物化事实缺少 SUPPORTED verification evidence"
                )
            evidence = evidence_values[0]
            hypothesis = hypotheses.get(str(evidence.hypothesis_id))
            if not hypothesis:
                raise RuntimeError("v2 fact evidence 缺少 hypothesis")
            opportunity = opportunities.get(str(hypothesis.pair_identity))
            if not opportunity:
                raise RuntimeError("v2 fact evidence 缺少 pair opportunity")
            subject_occurrence = str(evidence.subject_occurrence_key)
            object_occurrence = str(evidence.object_occurrence_key)
            subject_canonical = self._occurrence_canonical(
                opportunity, subject_occurrence,
            )
            object_canonical = self._occurrence_canonical(
                opportunity, object_occurrence,
            )
            if (
                subject_canonical == str(fact.subject_canonical_id)
                and object_canonical == str(fact.object_canonical_id)
            ):
                source_mention, target_mention = (
                    subject_occurrence, object_occurrence,
                )
            elif (
                object_canonical == str(fact.subject_canonical_id)
                and subject_canonical == str(fact.object_canonical_id)
            ):
                source_mention, target_mention = (
                    object_occurrence, subject_occurrence,
                )
            else:
                raise RuntimeError(
                    "v2 normalized fact 与 evidence occurrence assignment 不一致"
                )
            chunk_id = int(opportunity.provenance.get("chunk_id") or 0)
            document_id = int(
                opportunity.provenance.get("document_id") or 0
            )
            chunk = chunks.get(chunk_id)
            if (
                not chunk
                or int(chunk["document_id"]) != document_id
                or str(chunk["revision_id"])
                != str(opportunity.document_revision_id)
            ):
                raise RuntimeError("v2 evidence 指向非冻结 Chunk")
            evidence_start = min(
                int(span.start) for span in evidence.evidence_spans
            )
            evidence_end = max(
                int(span.end) for span in evidence.evidence_spans
            )
            content = str(chunk.get("content") or "")
            rows.append({
                "relation_key": str(fact.fact_id),
                "document_id": document_id,
                "revision_id": str(opportunity.document_revision_id),
                "chunk_id": chunk_id,
                "source_mention_key": source_mention,
                "target_mention_key": target_mention,
                "source_canonical_key": str(fact.subject_canonical_id),
                "target_canonical_key": str(fact.object_canonical_id),
                "relation_type": str(fact.predicate),
                "evidence_text": content[evidence_start:evidence_end],
                "evidence_start": evidence_start,
                "evidence_end": evidence_end,
                "policy_version": SHADOW_RELATION_V2_MATERIALIZATION_VERSION,
                "metadata": {
                    "verification_status": "VERIFIER_SUPPORTED",
                    "shadow_relation_v2_run_id": run_id,
                    "semantic_fact_key": str(fact.semantic_fact_key),
                    "hypothesis_id": str(evidence.hypothesis_id),
                    "verification_attempt_id":
                        str(evidence.verification_attempt_id),
                    "evidence_keys": [
                        str(item.evidence_key) for item in evidence_values
                    ],
                    "all_evidence_spans": [
                        [[span.start, span.end] for span in item.evidence_spans]
                        for item in evidence_values
                    ],
                },
            })
        return rows

    def run_namespace(
        self,
        *,
        namespace_id: str,
        project_id: int,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Execute planner, ledgers, models, normalization and Shadow materialization."""

        self.db.assert_shadow_relation_v2_ready()
        context = self._context(project_id, namespace_id)
        fingerprint = context["fingerprint"]
        run_id = "shadow_relation_v2_" + hashlib.sha256(
            f"{namespace_id}|{fingerprint.root}".encode("utf-8")
        ).hexdigest()
        started = False
        terminal_status = ""
        metrics: dict[str, Any] = {}
        try:
            self.db.start_shadow_relation_v2_run({
                "id": run_id,
                "project_id": project_id,
                "snapshot_id": namespace_id,
                "input_fingerprint": fingerprint.root,
                "fingerprint_components": fingerprint.components,
                "source_generation": int(
                    context["project"].get("graph_generation") or 0
                ),
            })
            started = True
            mentions_by_chunk: dict[int, list[dict[str, Any]]] = {}
            for mention in context["mentions"]:
                assignment = context["assignment_by_mention"][
                    str(mention["mention_key"])
                ]
                mentions_by_chunk.setdefault(
                    int(mention["chunk_id"]), [],
                ).append({
                    "occurrence_key": str(mention["mention_key"]),
                    "mention_id": str(mention["mention_key"]),
                    "surface": str(mention["surface"]),
                    "mention_start": int(mention["evidence_start"]),
                    "mention_end": int(mention["evidence_end"]),
                    "canonical_id": str(assignment["canonical_key"]),
                    "assignment_version": str(
                        assignment.get("policy_version") or namespace_id
                    ),
                    "type": str(
                        assignment.get("entity_type") or "GenericEntity"
                    ),
                })
            plans_by_chunk: dict[int, v2.PairOpportunityPlan] = {}
            all_opportunities: list[v2.PairOpportunity] = []
            total_chunks = len(context["chunks"])
            total_progress = max(1, total_chunks * 3)
            for chunk_index, chunk in enumerate(context["chunks"], start=1):
                plan = v2.plan_pair_opportunities(
                    project_id=project_id,
                    document_revision_id=str(chunk["revision_id"]),
                    chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
                    content=str(chunk.get("content") or ""),
                    mentions=mentions_by_chunk.get(int(chunk["id"]), []),
                    logical_schema=context["schema"],
                    run_id=run_id,
                    candidate_policy_version=
                        SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
                    max_pair_opportunities=
                        Config.relation_max_candidates_per_chunk,
                    sentence_window=Config.relation_sentence_window,
                    max_char_distance=Config.relation_max_char_distance,
                )
                annotated = tuple(
                    replace(
                        item,
                        provenance={
                            **item.provenance,
                            "document_id": int(chunk["document_id"]),
                            "chunk_id": int(chunk["id"]),
                            "content_hash": str(
                                chunk.get("content_hash") or ""
                            ),
                        },
                    )
                    for item in plan.opportunities
                )
                plan = replace(plan, opportunities=annotated)
                plans_by_chunk[int(chunk["id"])] = plan
                all_opportunities.extend(plan.opportunities)
                if progress:
                    progress(
                        chunk_index,
                        total_progress,
                        f"Shadow v2 pair ledger {chunk_index}/{total_chunks}",
                    )
            self.db.persist_shadow_relation_v2_opportunities(
                run_id, all_opportunities,
            )
            decisions: list[v2.PairClassificationDecision] = []
            classifier_calls = 0
            classifier_failures = 0
            context_budget = int(
                context["project"].get("model_max_tokens")
                or self.default_context_tokens
            )
            for chunk_index, chunk in enumerate(context["chunks"], start=1):
                admitted = list(plans_by_chunk[int(chunk["id"])].admitted)
                for offset in range(
                    0, len(admitted),
                    Config.relation_classifier_max_candidates,
                ):
                    batch = admitted[
                        offset:offset
                        + Config.relation_classifier_max_candidates
                    ]
                    prompt = v2.classifier_prompt(
                        batch,
                        content=str(chunk.get("content") or ""),
                        logical_schema=context["schema"],
                    )
                    response_contract = v2.classifier_response_schema(batch)
                    batch_decisions = None
                    last_error: Exception | None = None
                    for attempt_no in range(
                        1, Config.relation_shadow_max_attempts + 1,
                    ):
                        classifier_calls += 1
                        output_budget = self._classifier_output_budget(
                            context_budget_tokens=context_budget,
                            attempt_no=attempt_no,
                        )
                        request_hash = canonical_digest({
                            "system": SHADOW_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT,
                            "prompt": prompt,
                            "schema": response_contract,
                            "max_output_tokens": output_budget,
                            "safety_margin_tokens":
                                Config.relation_classifier_safety_margin,
                        })
                        try:
                            raw = self._classify(
                                model=context["model"],
                                context_budget_tokens=context_budget,
                                max_output_tokens=output_budget,
                                opportunities=batch,
                                content=str(chunk.get("content") or ""),
                                schema=context["schema"],
                            )
                            batch_decisions = v2.parse_classifier_response(
                                batch,
                                raw,
                                content=str(chunk.get("content") or ""),
                                logical_schema=context["schema"],
                                classifier_policy_version=
                                    SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION,
                                classifier_model_revision=
                                    context["model_revision"],
                            )
                            response_hash = canonical_digest(raw)
                            for pair in batch:
                                self.db.record_shadow_relation_v2_classification_attempt(
                                    run_id,
                                    pair.pair_identity,
                                    {
                                        "attempt_no": attempt_no,
                                        "status": "accepted",
                                        "request_hash": request_hash,
                                        "response_hash": response_hash,
                                        "classifier_prompt_version":
                                            v2.SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
                                        "classifier_schema_version":
                                            v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
                                        "classifier_parser_version":
                                            v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
                                        "classifier_model_revision":
                                            context["model_revision"],
                                        "metadata": {
                                            "chunk_id": int(chunk["id"]),
                                            "batch_offset": offset,
                                            "max_output_tokens": output_budget,
                                        },
                                    },
                                )
                            break
                        except Exception as exc:
                            last_error = exc
                            classifier_failures += 1
                            attempt_status = (
                                "permanent_failed"
                                if attempt_no
                                == Config.relation_shadow_max_attempts
                                else "retryable_failed"
                            )
                            for pair in batch:
                                self.db.record_shadow_relation_v2_classification_attempt(
                                    run_id,
                                    pair.pair_identity,
                                    {
                                        "attempt_no": attempt_no,
                                        "status": attempt_status,
                                        "request_hash": request_hash,
                                        "response_hash": "",
                                        "classifier_prompt_version":
                                            v2.SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
                                        "classifier_schema_version":
                                            v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
                                        "classifier_parser_version":
                                            v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
                                        "classifier_model_revision":
                                            context["model_revision"],
                                        "error_message": str(exc)[:2000],
                                        "metadata": {
                                            "chunk_id": int(chunk["id"]),
                                            "batch_offset": offset,
                                            "max_output_tokens": output_budget,
                                        },
                                    },
                                )
                    if batch_decisions is None:
                        raise RuntimeError(
                            "Shadow v2 classifier 未收敛且禁止回退旧流程"
                        ) from last_error
                    self.db.persist_shadow_relation_v2_classifications(
                        run_id, batch_decisions,
                    )
                    decisions.extend(batch_decisions)
                if progress:
                    progress(
                        total_chunks + chunk_index,
                        total_progress,
                        f"Shadow v2 classifier {chunk_index}/{total_chunks}",
                    )
            opportunity_by_pair = {
                str(item.pair_identity): item for item in all_opportunities
            }
            hypotheses = [
                hypothesis
                for decision in decisions
                for hypothesis in decision.hypotheses
            ]
            validations = []
            for hypothesis in hypotheses:
                opportunity = opportunity_by_pair[hypothesis.pair_identity]
                chunk = context["chunk_by_id"][
                    int(opportunity.provenance["chunk_id"])
                ]
                validations.append(v2.validate_hypothesis(
                    hypothesis,
                    opportunity,
                    content=str(chunk.get("content") or ""),
                    logical_schema=context["schema"],
                ))
            self.db.persist_shadow_relation_v2_validations(
                run_id, validations,
            )
            validation_by_hypothesis = {
                str(item.hypothesis_id): item for item in validations
            }
            verification_attempts: list[v2.VerificationAttempt] = []
            verifier_calls = 0
            for hypothesis in hypotheses:
                opportunity = opportunity_by_pair[hypothesis.pair_identity]
                content = str(
                    context["chunk_by_id"][
                        int(opportunity.provenance["chunk_id"])
                    ].get("content") or ""
                )
                validation = validation_by_hypothesis[
                    hypothesis.hypothesis_id
                ]
                final_attempt = None
                for attempt_no in range(
                    1, Config.relation_shadow_max_attempts + 1,
                ):
                    if validation.status == v2.PASS:
                        verifier_calls += 1
                    attempt = v2.verify_hypothesis(
                        hypothesis,
                        validation,
                        opportunity,
                        content=content,
                        logical_schema=context["schema"],
                        verifier=lambda prompt, response_schema: self._verify(
                            model=context["model"],
                            context_budget_tokens=context_budget,
                            prompt=prompt,
                            response_schema=response_schema,
                        ),
                        verifier_model_revision=context["model_revision"],
                        attempt_no=attempt_no,
                    )
                    self.db.persist_shadow_relation_v2_verification_attempts(
                        run_id, [attempt],
                    )
                    verification_attempts.append(attempt)
                    final_attempt = attempt
                    if (
                        attempt.verdict != v2.RETRYABLE_FAILED
                        or validation.status != v2.PASS
                    ):
                        break
                if final_attempt is None:
                    raise RuntimeError("Shadow v2 verifier 未生成终态")
            normalization = v2.normalize_verified_hypotheses(
                project_id=project_id,
                opportunities=opportunity_by_pair,
                hypotheses=hypotheses,
                verification_attempts=verification_attempts,
                logical_schema=context["schema"],
            )
            current = self._context(project_id, namespace_id)["fingerprint"]
            hypothesis_by_id = {
                str(item.hypothesis_id): item for item in hypotheses
            }
            evidence_pairs_by_fact: dict[str, set[str]] = {}
            for evidence in normalization.evidence:
                hypothesis = hypothesis_by_id[str(evidence.hypothesis_id)]
                evidence_pairs_by_fact.setdefault(
                    evidence.fact_id, set(),
                ).add(hypothesis.pair_identity)
            materialization_decisions = [
                v2.relation_materialization_decision(
                    fact,
                    normalization.evidence,
                    recorded_fingerprint=fingerprint,
                    current_fingerprint=current,
                    snapshot_complete=True,
                    current_assignment_and_schema=(
                        fingerprint.root == current.root
                    ),
                    conflicts=[
                        conflict for conflict in normalization.conflicts
                        if conflict.pair_identity
                        in evidence_pairs_by_fact.get(fact.fact_id, set())
                    ],
                )
                for fact in normalization.facts
            ]
            self.db.persist_shadow_relation_v2_normalization(
                run_id,
                normalization,
                input_fingerprint=fingerprint.root,
                materialization_decisions=materialization_decisions,
            )
            decision_counts = Counter(
                str(item.decision) for item in decisions
            )
            terminal_attempts = {}
            for item in verification_attempts:
                current_attempt = terminal_attempts.get(str(item.hypothesis_id))
                if (
                    current_attempt is None
                    or int(item.attempt_no) > int(current_attempt.attempt_no)
                ):
                    terminal_attempts[str(item.hypothesis_id)] = item
            verdict_counts = Counter(
                str(item.verdict) for item in terminal_attempts.values()
            )
            validation_counts = Counter(
                str(item.status) for item in validations
            )
            eligible_count = sum(
                int(item.eligible) for item in materialization_decisions
            )
            metrics = {
                "pair_opportunities_total": len(all_opportunities),
                "pair_opportunities_classified": len(decisions),
                "pair_statuses": dict(Counter(
                    str(item.status) for item in all_opportunities
                )),
                "classification_decisions": dict(decision_counts),
                "classified_hypotheses": len(hypotheses),
                "deterministic_validation": dict(validation_counts),
                "verification_attempts": len(verification_attempts),
                "verification_terminal_count": len(terminal_attempts),
                "verification_verdicts": dict(verdict_counts),
                "verifier_supported": int(
                    verdict_counts.get(v2.SUPPORTED, 0)
                ),
                "facts_after_normalization": len(normalization.facts),
                "facts_materialization_eligible": eligible_count,
                "facts_review_only": sum(
                    int(fact.review_only)
                    for fact in normalization.facts
                ),
                "normalization": dict(normalization.metrics),
                "classifier_model_calls": classifier_calls,
                "classifier_failures": classifier_failures,
                "verifier_model_calls": verifier_calls,
                "legacy_pipeline_calls": 0,
                "production_approved": False,
            }
            if fingerprint.root != current.root:
                terminal_status = "stale"
                self.db.finish_shadow_relation_v2_run(
                    run_id,
                    status="stale",
                    metrics=metrics,
                    input_fingerprint=fingerprint.root,
                    error_message="materialization fingerprint became stale",
                )
                raise v2.StaleRelationSnapshotError(
                    "Shadow relation v2 fingerprint stale; "
                    "refusing isolated materialization"
                )
            endpoint_rows = self._endpoint_rows(
                run_id=run_id,
                normalization=normalization,
                materialization_decisions=materialization_decisions,
                opportunities=opportunity_by_pair,
                hypotheses=hypothesis_by_id,
                chunks=context["chunk_by_id"],
            )
            if len(endpoint_rows) != eligible_count:
                raise RuntimeError(
                    "Shadow relation v2 eligible fact 与隔离物化数量不一致"
                )
            self.db.persist_entity_shadow_relation_endpoints(
                namespace_id, endpoint_rows,
            )
            self.db.finish_shadow_relation_v2_run(
                run_id,
                status="ready",
                metrics=metrics,
                input_fingerprint=fingerprint.root,
            )
            terminal_status = "ready"
            if progress:
                progress(
                    total_progress,
                    total_progress,
                    f"Shadow v2 完成：{len(hypotheses)} 个 hypotheses，"
                    f"{len(normalization.facts)} 个 normalized facts，"
                    f"{len(endpoint_rows)} 个隔离物化事实",
                )
            return {
                "run_id": run_id,
                "input_fingerprint": fingerprint.root,
                "metrics": metrics,
                "endpoint_rows": endpoint_rows,
            }
        except Exception as exc:
            if started and not terminal_status:
                try:
                    self.db.finish_shadow_relation_v2_run(
                        run_id,
                        status="failed",
                        metrics={
                            **metrics,
                            "legacy_pipeline_calls": 0,
                            "runtime_failure": type(exc).__name__,
                        },
                        input_fingerprint=fingerprint.root,
                        error_message=str(exc)[:2000],
                    )
                except Exception:
                    pass
            raise
