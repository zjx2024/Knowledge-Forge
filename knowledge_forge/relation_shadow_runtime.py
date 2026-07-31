"""Fail-closed runtime orchestration for the isolated Shadow relation v2 path."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

from . import relation_shadow_v2 as v2
from .config import Config
from .quality import canonical_digest, canonical_json
from .shadow_schema_overlay import (
    SHADOW_RELATION_SCHEMA_OVERLAY_VERSION,
    apply_shadow_relation_schema_overlay,
)
from .shadow_context import build_shadow_context_manifest, occurrence_scope


def _loaded_relation_code_digest() -> str:
    """Fingerprint the source revision loaded by this worker process once.

    Long-running relation jobs must compare against the code that the Python
    worker actually imported.  Re-reading source files at the final freshness
    checkpoint makes an unrelated on-disk deployment look like an in-process
    code change even though the running worker cannot execute those new bytes.
    """

    code_hasher = hashlib.sha256()
    for path in (Path(__file__), Path(str(v2.__file__))):
        code_hasher.update(path.read_bytes())
    return code_hasher.hexdigest()


LOADED_RELATION_CODE_DIGEST = _loaded_relation_code_digest()


CLASSIFIER_ACCEPTED = "ACCEPTED"
CLASSIFIER_RETRYABLE_EMPTY = "RETRYABLE_EMPTY"
CLASSIFIER_RETRYABLE_PROTOCOL = "RETRYABLE_PROTOCOL"
CLASSIFIER_RETRYABLE_TRUNCATED = "RETRYABLE_TRUNCATED"
CLASSIFIER_RETRYABLE_TIMEOUT = "RETRYABLE_TIMEOUT"
CLASSIFIER_RETRYABLE_MODEL_ERROR = "RETRYABLE_MODEL_ERROR"
CLASSIFIER_TERMINAL_FAILED = "TERMINAL_FAILED"
CLASSIFIER_SUPERSEDED = "SUPERSEDED"
CLASSIFIER_ATTEMPT_STATUSES = {
    CLASSIFIER_ACCEPTED,
    CLASSIFIER_RETRYABLE_EMPTY,
    CLASSIFIER_RETRYABLE_PROTOCOL,
    CLASSIFIER_RETRYABLE_TRUNCATED,
    CLASSIFIER_RETRYABLE_TIMEOUT,
    CLASSIFIER_RETRYABLE_MODEL_ERROR,
    CLASSIFIER_TERMINAL_FAILED,
    CLASSIFIER_SUPERSEDED,
}


def shadow_model_payload_hash(kind: str, payload: Any) -> str:
    """Hash audited model payload bytes with a request/response domain separator."""

    normalized_kind = str(kind)
    if normalized_kind not in {"request", "response"}:
        raise ValueError("shadow model payload hash kind must be request or response")
    if isinstance(payload, bytes):
        material = payload
    elif isinstance(payload, str):
        material = payload.encode("utf-8")
    else:
        material = canonical_json(payload).encode("utf-8")
    return hashlib.sha256(
        normalized_kind.encode("utf-8") + b"\0" + material
    ).hexdigest()


def _safe_json_text(value: Any) -> str:
    try:
        return canonical_json(value)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _proposal_found_key(proposal: v2.RelationProposal) -> str:
    """Key used for relation-only gleaning; independent of pass/retry."""

    return canonical_digest({
        "subject_occurrence_id": proposal.subject_occurrence_key,
        "predicate": proposal.predicate,
        "object_occurrence_id": proposal.object_occurrence_key,
        "qualifiers": list(proposal.qualifiers),
    })


def _with_direct_proposer_source(
    opportunity: v2.PairOpportunity,
) -> v2.PairOpportunity:
    provenance = dict(opportunity.provenance)
    provenance.setdefault("planner_status_before_direct_proposer", opportunity.status)
    provenance.setdefault("source_kind", "v2_pair_planner")
    provenance.setdefault("source_record_id", opportunity.context_identity)
    sources = {
        str(value)
        for value in provenance.get("candidate_sources", [])
        if str(value)
    }
    if not sources:
        sources.add(v2.PAIR_PLANNER)
    sources.add(v2.DIRECT_PROPOSER)
    provenance["candidate_sources"] = sorted(sources)
    provenance["direct_proposer_union"] = True
    if opportunity.status != v2.PENDING_CLASSIFICATION:
        provenance["direct_proposer_rejected_by_planner"] = True
        return replace(opportunity, provenance=provenance)
    return replace(
        opportunity,
        provenance=provenance,
    )


def _classifier_failure_status(exc: Exception, *, terminal: bool) -> str:
    if terminal:
        return CLASSIFIER_TERMINAL_FAILED
    message = str(exc).lower()
    name = type(exc).__name__.lower()
    if "truncated" in message or "max_tokens" in message or "output length" in message:
        return CLASSIFIER_RETRYABLE_TRUNCATED
    if "timeout" in name or "timeout" in message:
        return CLASSIFIER_RETRYABLE_TIMEOUT
    if "missing pair ids" in message or "empty" in message:
        return CLASSIFIER_RETRYABLE_EMPTY
    if (
        "protocol" in name
        or "invalid json" in message
        or "must be" in message
        or "unknown pair ids" in message
        or "duplicate pair ids" in message
        or "predicate is not allowed" in message
        or "hypothesis endpoints must be distinct" in message
        or "positive hypothesis requires evidence" in message
        or "evidence span is outside frozen context" in message
        or "classifier confidence is" in message
        or "requires" in message
    ):
        return CLASSIFIER_RETRYABLE_PROTOCOL
    return CLASSIFIER_RETRYABLE_MODEL_ERROR


def _terminal_classifier_fail_closed_decision(
    opportunity: v2.PairOpportunity,
    exc: Exception | None,
) -> v2.PairClassificationDecision:
    """Return a non-publishable terminal decision for one unclassifiable pair.

    The parser remains strict, but the runtime should not let one malformed
    single-pair model response abort the whole enhanced Shadow run. This is a
    fail-closed ledger state: no hypotheses are emitted, the verifier is not
    invoked for this pair, and no normalized fact can be materialized from it.
    """

    reason_code = "CLASSIFIER_TERMINAL_PROTOCOL_FAILURE"
    if exc is not None and "budget" in type(exc).__name__.lower():
        reason_code = "CLASSIFIER_TERMINAL_BUDGET_FAILURE"
    elif exc is not None and "timeout" in type(exc).__name__.lower():
        reason_code = "CLASSIFIER_TERMINAL_TIMEOUT"
    return v2.PairClassificationDecision(
        pair_identity=opportunity.pair_identity,
        decision=v2.OTHER_UNSUPPORTED,
        hypotheses=(),
        evidence_spans=(
            v2.EvidenceSpan(
                start=int(opportunity.context_span[0]),
                end=int(opportunity.context_span[1]),
            ),
        ),
        reason_code=reason_code,
    )


SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION = "knowledge-forge-candidate-v4-structure-gate"
SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION = "knowledge-forge-classifier-v5-structure-gate"
SHADOW_RELATION_V2_NORMALIZATION_VERSION = "knowledge-forge-normalization-v4-structure-gate"
SHADOW_RELATION_V2_MATERIALIZATION_VERSION = "knowledge-forge-materialization-v4-structure-gate"
SHADOW_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT = (
    "You are the Shadow relation v2 classifier. Follow the supplied "
    "pair contract exactly and return only one compact schema-valid "
    "JSON value. Do not include <think>, reasoning, markdown, "
    "commentary, or any text outside JSON."
)
SHADOW_RELATION_V2_PROPOSER_SYSTEM_PROMPT = (
    "You are the Shadow relation v2 direct proposer. Return only compact "
    "schema-valid JSON. Proposals are candidates only and never bypass the "
    "independent verifier."
)
SHADOW_RELATION_V2_POLICY_SET = {
    "candidate_policy_version": SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
    "classifier_policy_version": SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION,
    "classifier_prompt_version": v2.SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
    "classifier_schema_version": v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
    "classifier_parser_version": v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
    "proposer_prompt_version": v2.SHADOW_RELATION_PROPOSER_PROMPT_VERSION,
    "proposer_schema_version": v2.SHADOW_RELATION_PROPOSER_SCHEMA_VERSION,
    "proposer_parser_version": v2.SHADOW_RELATION_PROPOSER_PARSER_VERSION,
    "verifier_prompt_version": v2.SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
    "verifier_schema_version": v2.SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
    "verifier_parser_version": v2.SHADOW_RELATION_VERIFIER_PARSER_VERSION,
    "normalization_version": SHADOW_RELATION_V2_NORMALIZATION_VERSION,
    "materialization_version": SHADOW_RELATION_V2_MATERIALIZATION_VERSION,
    "schema_overlay_version": SHADOW_RELATION_SCHEMA_OVERLAY_VERSION,
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
        chunks_by_document: dict[int, list[dict[str, Any]]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(int(chunk["document_id"]), []).append(chunk)

        def load_processing_payload(document: dict[str, Any]) -> dict[str, Any] | None:
            loader = getattr(
                self.db, "get_shadow_document_processing_payload", None,
            )
            if not callable(loader):
                return None
            payload = loader(document)
            return payload if isinstance(payload, dict) else None

        shadow_context = build_shadow_context_manifest(
            namespace_id=f"relation-shadow:{project_id}:{namespace_id}",
            documents=documents,
            chunks_by_document=chunks_by_document,
            payload_loader=load_processing_payload,
        )
        chunk_by_id = {int(item["id"]): item for item in chunks}
        relation_chunks: list[dict[str, Any]] = []
        source_context_rows: list[dict[str, Any]] = []
        for unit in shadow_context.context_units:
            structure = dict(unit.structure or {})
            chunk_id = int(structure.get("chunk_id") or 0)
            source_chunk = chunk_by_id.get(chunk_id)
            if source_chunk is None:
                if unit.structure_quality != "legacy_chunk_local":
                    raise RuntimeError(
                        "Shadow relation v2 structured context cannot be "
                        "mapped to frozen chunk mentions"
                    )
                continue
            relation_chunks.append({
                **source_chunk,
                "chunk_key": unit.context_id,
                "content": unit.rendered_text,
                "shadow_context_id": unit.context_id,
                "shadow_context_kind": unit.context_kind,
                "shadow_context_quality": unit.structure_quality,
                "shadow_context_rendered_hash": unit.rendered_hash,
                "shadow_source_map": [asdict(item) for item in unit.source_map],
                "shadow_structure": structure,
            })
            source_context_rows.append({
                "context_identity": unit.context_id,
                "project_id": int(project_id),
                "document_id": int(source_chunk.get("document_id") or 0),
                "document_revision_id": str(
                    source_chunk.get("revision_id")
                    or unit.document_revision_id
                ),
                "chunk_id": int(source_chunk["id"]),
                "chunk_key": str(unit.context_id),
                "context_kind": unit.context_kind,
                "context_start": 0,
                "context_end": len(unit.rendered_text),
                "content_hash": unit.rendered_hash,
                "structure_artifact_hash": shadow_context.artifact_hash,
                "parser_version": str(unit.structure.get("parser_version") or ""),
                "payload": {
                    "context_id": unit.context_id,
                    "namespace_id": unit.namespace_id,
                    "rendered_hash": unit.rendered_hash,
                    "source_block_keys": list(unit.source_block_keys),
                    "structure": structure,
                    "structure_quality": unit.structure_quality,
                    "source_map": [asdict(item) for item in unit.source_map],
                },
                "source_maps": [asdict(item) for item in unit.source_map],
            })
        schema_record = self.db.get_active_schema(project_id)
        if not schema_record:
            raise RuntimeError("Shadow relation v2 缺少 active Schema")
        schema = apply_shadow_relation_schema_overlay(
            dict(schema_record.get("schema_json") or {})
        )
        mentions = self.db.list_entity_shadow_mentions(namespace_id)
        snapshot = self.db.load_entity_shadow_snapshot(namespace_id)
        all_assignments = list(snapshot.get("assignments") or [])
        all_assignment_by_mention = {
            str(item["mention_key"]): item for item in all_assignments
        }
        if len(all_assignment_by_mention) != len(all_assignments):
            raise RuntimeError("Shadow relation v2 assignment occurrence 不唯一")
        missing = sorted(
            str(item["mention_key"]) for item in mentions
            if str(item["mention_key"]) not in all_assignment_by_mention
        )
        if missing:
            raise RuntimeError(
                "Shadow relation v2 mention 缺少冻结 assignment: "
                + ",".join(missing[:10])
            )
        quarantined_canonicals = {
            str(item["canonical_key"])
            for item in snapshot.get("canonicals") or []
            if bool((item.get("metadata") or {}).get("quarantined"))
        }
        assignments = all_assignments
        if quarantined_canonicals:
            assignments = [
                item for item in assignments
                if str(item.get("canonical_key") or "") not in quarantined_canonicals
            ]
        assignment_by_mention = {
            str(item["mention_key"]): item for item in assignments
        }
        if len(assignment_by_mention) != len(assignments):
            raise RuntimeError("Shadow relation v2 assignment occurrence 不唯一")
        mentions = [
            item for item in mentions
            if str(item["mention_key"]) in assignment_by_mention
        ]
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
            "occurrence_key": self._physical_occurrence_key(item),
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
        candidate = {
            "policy_version": SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
            "parameters": {
                "sentence_window": Config.relation_sentence_window,
                "max_char_distance": Config.relation_max_char_distance,
                "batch_size": Config.relation_classifier_max_candidates,
                "soft_cues": "schema-configured-nonbinding",
                "source_context_policy": shadow_context.policy_version,
                "source_map_policy": shadow_context.source_map_policy,
            },
            "budget": Config.relation_max_candidates_per_chunk,
            "structure_artifact_hash": shadow_context.artifact_hash,
        }
        classifier = {
            "prompt_version": v2.SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
            "schema_version": v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
            "parser_version": v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
            "model": model,
            "model_revision": model_revision,
        }
        proposer = {
            "prompt_version": v2.SHADOW_RELATION_PROPOSER_PROMPT_VERSION,
            "schema_version": v2.SHADOW_RELATION_PROPOSER_SCHEMA_VERSION,
            "parser_version": v2.SHADOW_RELATION_PROPOSER_PARSER_VERSION,
            "model": model,
            "model_revision": model_revision,
            "max_gleaning_rounds": 2,
            "failure_circuit_breaker":
                Config.relation_proposer_failure_circuit_breaker,
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
            "proposer": proposer,
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
            proposer=proposer,
            classifier=classifier,
            verifier=verifier,
            thresholds={
                "max_attempts": Config.relation_shadow_max_attempts,
                "max_gleaning_rounds": 2,
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
                "code_tree": LOADED_RELATION_CODE_DIGEST,
                "policy_root": policy_root,
            },
        )
        return {
            "project": project,
            "documents": documents,
            "chunks": chunks,
            "relation_chunks": relation_chunks,
            "chunk_by_id": chunk_by_id,
            "shadow_context": shadow_context,
            "source_context_rows": source_context_rows,
            "schema": schema,
            "mentions": mentions,
            "assignments": assignments,
            "assignment_by_mention": assignment_by_mention,
            "model": model,
            "model_revision": model_revision,
            "fingerprint": fingerprint,
        }

    @staticmethod
    def _physical_occurrence_key(mention: dict[str, Any]) -> str:
        provenance = mention.get("provenance") or {}
        if isinstance(provenance, str):
            try:
                provenance = json.loads(provenance)
            except json.JSONDecodeError:
                provenance = {}
        if not isinstance(provenance, dict):
            provenance = {}
        return str(
            provenance.get("physical_occurrence_key")
            or mention.get("occurrence_key")
            or mention["mention_key"]
        )

    @staticmethod
    def _mentions_for_context(
        chunk: dict[str, Any],
        mentions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach parser-role and atomic-scope facts to frozen occurrences.

        This adapter is intentionally structure-only.  It does not inspect a
        predicate, domain word, document title or model output when deciding a
        relation boundary.
        """

        content = str(chunk.get("content") or "")
        structure = chunk.get("shadow_structure")
        if not isinstance(structure, dict):
            structure = {}
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        context_kind = str(chunk.get("shadow_context_kind") or "")
        enriched: list[dict[str, Any]] = []
        for mention in mentions:
            profile = occurrence_scope(
                content=content,
                mention_start=int(mention["mention_start"]),
                mention_end=int(mention["mention_end"]),
                structure=structure,
                context_kind=context_kind,
                metadata=metadata,
            )
            scope = dict(profile["occurrence_scope"])
            enriched.append({
                **mention,
                "source_role": str(profile["source_role"]),
                "relation_eligible": bool(profile["relation_eligible"]),
                "scope_id": str(scope["scope_id"]),
                "scope_kind": str(scope["scope_kind"]),
                "scope_start": int(scope["start"]),
                "scope_end": int(scope["end"]),
                "table_id": str(
                    mention.get("table_id") or structure.get("table_id") or ""
                ),
                "row_id": str(
                    mention.get("row_id") or structure.get("row_id") or ""
                ),
                "cell_id": str(
                    mention.get("cell_id") or structure.get("cell_id") or ""
                ),
            })
        return enriched

    @staticmethod
    def _direct_proposer_opportunity(
        *,
        run_id: str,
        project_id: int,
        chunk: dict[str, Any],
        proposal: v2.RelationProposal,
        mentions: list[dict[str, Any]],
        logical_schema: dict[str, Any],
    ) -> v2.PairOpportunity | None:
        """Attach a proposer source only to a planner-admitted pair.

        The proposer is recall assistance, never a second candidate policy. A
        malformed or structurally inadmissible proposal remains audited by its
        proposal attempt but cannot manufacture a ledger row or bypass a
        source/scope boundary.
        """

        mentions_by_occurrence = {
            str(item["occurrence_key"]): item for item in mentions
        }
        try:
            subject = mentions_by_occurrence[proposal.subject_occurrence_key]
            object_ = mentions_by_occurrence[proposal.object_occurrence_key]
        except KeyError as exc:
            raise RuntimeError(
                "Shadow v2 direct proposer endpoint missing from context"
            ) from exc
        content = str(chunk.get("content") or "")
        if not content:
            raise RuntimeError("Shadow v2 direct proposer context is empty")
        plan = v2.plan_pair_opportunities(
            project_id=int(project_id),
            document_revision_id=str(chunk["revision_id"]),
            chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
            content=content,
            mentions=[subject, object_],
            logical_schema=logical_schema,
            run_id=run_id,
            candidate_policy_version=SHADOW_RELATION_V2_CANDIDATE_POLICY_VERSION,
            max_pair_opportunities=None,
        )
        if len(plan.opportunities) != 1:
            raise RuntimeError("Shadow v2 direct proposer pair planning is not unique")
        planned = plan.opportunities[0]
        allowed = {
            str(item.get("name") or "")
            for item in planned.allowed_predicate_metadata
        }
        if planned.status != v2.PENDING_CLASSIFICATION or proposal.predicate not in allowed:
            return None
        source_context_id = str(chunk.get("shadow_context_id") or chunk["id"])
        return replace(
            planned,
            planning_reason="direct_proposer_pair_admitted_by_planner",
            cue=v2.DIRECT_PROPOSER,
            provenance={
                **planned.provenance,
                "candidate_sources": [v2.DIRECT_PROPOSER],
                "source_kind": "v2_direct_proposer",
                "source_record_id": proposal.proposal_id,
                "source_context_id": source_context_id,
                "source_context_kind": str(
                    chunk.get("shadow_context_kind") or ""
                ),
                "source_context_quality": str(
                    chunk.get("shadow_context_quality") or ""
                ),
                "document_id": int(chunk["document_id"]),
                "chunk_id": int(chunk["id"]),
                "content_hash": str(chunk.get("content_hash") or ""),
                "direct_proposer_union": True,
            },
        )

    def _classify(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        max_output_tokens: int | None = None,
        opportunities: list[v2.PairOpportunity],
        content: str,
        schema: dict[str, Any],
        prompt: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> Any:
        output_budget = int(
            max_output_tokens or Config.relation_classifier_output_reserve
        )
        request_prompt = (
            prompt if prompt is not None
            else v2.classifier_prompt(
                opportunities, content=content, logical_schema=schema,
            )
        )
        request_schema = (
            response_schema if response_schema is not None
            else v2.classifier_response_schema(opportunities)
        )
        raw = self.models.json_chat(
            model=model,
            system=SHADOW_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT,
            prompt=request_prompt,
            schema=request_schema,
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

        def normalize_decision_list(values: Any) -> Any:
            return [normalize_negative_decision(value) for value in values]

        if isinstance(raw, dict):
            for key in ("decisions", "items", "result"):
                if isinstance(raw.get(key), list):
                    return normalize_decision_list(raw[key])
        if isinstance(raw, list):
            return normalize_decision_list(raw)
        raw = normalize_negative_decision(raw)
        if (
            isinstance(raw, dict)
            and {"pair_id", "decision", "hypotheses"} <= set(raw)
        ):
            return [raw]
        return raw

    def _propose(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        max_output_tokens: int | None = None,
        context_id: str,
        content: str,
        mentions: list[dict[str, Any]],
        schema: dict[str, Any],
        pass_no: int = 0,
        already_found_keys: list[str] | None = None,
        prompt: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> Any:
        output_budget = int(
            max_output_tokens
            or self._proposer_output_budget(
                context_budget_tokens=context_budget_tokens,
                attempt_no=pass_no + 1,
            )
        )
        request_prompt = (
            prompt if prompt is not None
            else v2.proposer_prompt(
                context_id=context_id,
                content=content,
                mentions=mentions,
                logical_schema=schema,
                already_found_keys=already_found_keys or [],
                pass_no=pass_no,
                max_proposals=Config.relation_proposer_max_proposals,
            )
        )
        request_schema = (
            response_schema if response_schema is not None
            else v2.proposer_response_schema(
                mentions=mentions,
                logical_schema=schema,
                content_length=len(content),
                max_proposals=Config.relation_proposer_max_proposals,
            )
        )
        return self.models.json_chat(
            model=model,
            system=SHADOW_RELATION_V2_PROPOSER_SYSTEM_PROMPT,
            prompt=request_prompt,
            schema=request_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=output_budget,
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    @staticmethod
    def _proposer_output_budget(
        *, context_budget_tokens: int, attempt_no: int,
    ) -> int:
        base = max(2048, int(Config.relation_classifier_output_reserve))
        requested = base * (2 ** max(0, int(attempt_no) - 1))
        context_cap = max(
            base,
            int(context_budget_tokens)
            - int(Config.relation_classifier_safety_margin)
            - 1024,
        )
        return max(512, min(requested, context_cap, 8192))

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

    @staticmethod
    def _occurrence_mention_key(
        opportunity: v2.PairOpportunity,
        occurrence_key: str,
    ) -> str:
        if occurrence_key == opportunity.left_occurrence_key:
            return str(opportunity.left_mention_id)
        if occurrence_key == opportunity.right_occurrence_key:
            return str(opportunity.right_mention_id)
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
                    self._occurrence_mention_key(opportunity, subject_occurrence),
                    self._occurrence_mention_key(opportunity, object_occurrence),
                )
            elif (
                object_canonical == str(fact.subject_canonical_id)
                and subject_canonical == str(fact.object_canonical_id)
            ):
                source_mention, target_mention = (
                    self._occurrence_mention_key(opportunity, object_occurrence),
                    self._occurrence_mention_key(opportunity, subject_occurrence),
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
                    "source_occurrence_key": subject_occurrence,
                    "target_occurrence_key": object_occurrence,
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
            self.db.persist_shadow_relation_v2_source_contexts(
                run_id,
                context["source_context_rows"],
            )
            mentions_by_chunk: dict[int, list[dict[str, Any]]] = {}
            for mention in context["mentions"]:
                assignment = context["assignment_by_mention"][
                    str(mention["mention_key"])
                ]
                mentions_by_chunk.setdefault(
                    int(mention["chunk_id"]), [],
                ).append({
                    "occurrence_key": self._physical_occurrence_key(mention),
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
            plans_by_context: dict[str, v2.PairOpportunityPlan] = {}
            mentions_by_context: dict[str, list[dict[str, Any]]] = {}
            all_opportunities: list[v2.PairOpportunity] = []
            proposal_batches: list[tuple[str, v2.RelationProposalBatch]] = []
            proposer_calls = 0
            proposer_failures = 0
            proposer_attempt_counts: Counter[str] = Counter()
            proposer_failure_keys: Counter[tuple[str, str, str]] = Counter()
            proposer_circuit_breaker_tripped = False
            proposer_skipped_by_circuit = 0
            proposer_terminal_statuses: Counter[str] = Counter()
            proposer_hypothesis_rejections: Counter[str] = Counter()
            relation_chunks = list(context["relation_chunks"])
            total_chunks = len(relation_chunks)
            total_progress = max(1, total_chunks * 4)
            context_budget = int(
                context["project"].get("model_max_tokens")
                or self.default_context_tokens
            )
            for chunk_index, chunk in enumerate(relation_chunks, start=1):
                context_key = str(chunk.get("shadow_context_id") or chunk["id"])
                chunk_mentions = self._mentions_for_context(
                    chunk,
                    mentions_by_chunk.get(int(chunk["id"]), []),
                )
                mentions_by_context[context_key] = chunk_mentions
                plan = v2.plan_pair_opportunities(
                    project_id=project_id,
                    document_revision_id=str(chunk["revision_id"]),
                    chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
                    content=str(chunk.get("content") or ""),
                    mentions=chunk_mentions,
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
                            "source_context_id": context_key,
                            "source_context_kind": str(
                                chunk.get("shadow_context_kind") or ""
                            ),
                            "source_context_quality": str(
                                chunk.get("shadow_context_quality") or ""
                            ),
                            "content_hash": str(
                                chunk.get("content_hash") or ""
                            ),
                        },
                    )
                    for item in plan.opportunities
                )
                plan = replace(plan, opportunities=annotated)
                plans_by_context[context_key] = plan
                all_opportunities.extend(plan.opportunities)
                if progress:
                    progress(
                        chunk_index,
                        total_progress,
                        f"Knowledge Forge 候选对规划 {chunk_index}/{total_chunks}",
                    )
                proposer_message = (
                    f"Knowledge Forge 候选补充 {chunk_index}/{total_chunks} "
                    "跳过：实体不足"
                )
                if len(chunk_mentions) >= 2 and not plan.admitted:
                    proposer_message = (
                        f"Knowledge Forge 候选补充 {chunk_index}/{total_chunks} "
                        "跳过：没有通过结构与范围门禁的候选对"
                    )
                elif len(chunk_mentions) >= 2 and plan.admitted:
                    context_kind = str(
                        chunk.get("shadow_context_kind") or "legacy_chunk"
                    )
                    blocked = [
                        key for key, count in proposer_failure_keys.items()
                        if (
                            key[0] == str(chunk.get("document_id") or "")
                            and key[1] == context_kind
                            and count >= int(
                                Config.relation_proposer_failure_circuit_breaker
                            ) > 0
                        )
                    ]
                    if blocked:
                        proposer_skipped_by_circuit += 1
                        proposer_circuit_breaker_tripped = True
                        proposer_terminal_statuses[
                            "TERMINAL_FAILED_WITH_PLANNER_FALLBACK"
                        ] += 1
                        proposer_message = (
                            f"Knowledge Forge 候选补充 {chunk_index}/{total_chunks} "
                            "辅助召回已达异常阈值，本轮后续不再调用（主链继续）"
                        )
                    else:
                        already_found_keys: set[str] = set()
                        context_success = False
                        last_proposer_error: Exception | None = None
                        for pass_no in range(0, 3):
                            legal_batch: v2.RelationProposalBatch | None = None
                            for retry_no in range(
                                1, int(Config.relation_shadow_max_attempts) + 1,
                            ):
                                proposer_calls += 1
                                proposer_attempt_counts[context_key] += 1
                                audit_attempt_no = int(
                                    proposer_attempt_counts[context_key]
                                )
                                output_budget = self._proposer_output_budget(
                                    context_budget_tokens=context_budget,
                                    attempt_no=retry_no,
                                )
                                already_found_payload = sorted(
                                    already_found_keys
                                )
                                proposer_protocol_feedback = ""
                                if last_proposer_error is not None:
                                    proposer_protocol_feedback = (
                                        "Previous proposer response violated "
                                        "the v2 contract and was rejected "
                                        "fail-closed: "
                                        f"{type(last_proposer_error).__name__}: "
                                        f"{str(last_proposer_error)[:800]}. "
                                        "Return exactly one object with "
                                        "proposals as an array, "
                                        "missing_endpoint_mentions as an array, "
                                        "and coverage as an object. Keep "
                                        "endpoints copied from occurrences, "
                                        "predicate copied from predicates, and "
                                        "evidence_spans covering both endpoints."
                                    )
                                proposer_prompt = v2.proposer_prompt(
                                    context_id=str(
                                        chunk.get("chunk_key") or chunk["id"]
                                    ),
                                    content=str(chunk.get("content") or ""),
                                    mentions=chunk_mentions,
                                    logical_schema=context["schema"],
                                    already_found_keys=
                                        already_found_payload,
                                    pass_no=pass_no,
                                    protocol_feedback=
                                        proposer_protocol_feedback,
                                    max_proposals=
                                        Config.relation_proposer_max_proposals,
                                )
                                proposer_schema = v2.proposer_response_schema(
                                    mentions=chunk_mentions,
                                    logical_schema=context["schema"],
                                    content_length=len(
                                        str(chunk.get("content") or ""),
                                    ),
                                    max_proposals=
                                        Config.relation_proposer_max_proposals,
                                )
                                request_payload = {
                                    "stage": "proposer",
                                    "model": context["model"],
                                    "model_revision": context["model_revision"],
                                    "system":
                                        SHADOW_RELATION_V2_PROPOSER_SYSTEM_PROMPT,
                                    "prompt": proposer_prompt,
                                    "schema": proposer_schema,
                                    "temperature": 0.0,
                                    "context_budget_tokens": context_budget,
                                    "max_output_tokens": output_budget,
                                    "safety_margin_tokens":
                                        Config.relation_classifier_safety_margin,
                                    "context_identity": context_key,
                                    "pass_no": pass_no,
                                    "retry_no": retry_no,
                                }
                                request_hash = shadow_model_payload_hash(
                                    "request", request_payload,
                                )
                                raw_response_text = ""
                                parsed_response: Any = {}
                                token_usage: dict[str, Any] = {}
                                try:
                                    raw_proposal = self._propose(
                                        model=context["model"],
                                        context_budget_tokens=context_budget,
                                        max_output_tokens=output_budget,
                                        context_id=str(
                                            chunk.get("chunk_key") or chunk["id"]
                                        ),
                                        content=str(chunk.get("content") or ""),
                                        mentions=chunk_mentions,
                                        schema=context["schema"],
                                        pass_no=pass_no,
                                        already_found_keys=already_found_payload,
                                        prompt=proposer_prompt,
                                        response_schema=proposer_schema,
                                    )
                                    parsed_response = raw_proposal
                                    raw_response_text = str(
                                        getattr(
                                            self.models,
                                            "last_raw_response_text",
                                            "",
                                        )
                                        or _safe_json_text(raw_proposal)
                                    )
                                    diagnostics = getattr(
                                        self.models, "last_call_diagnostics", None,
                                    )
                                    if diagnostics is not None:
                                        token_usage = asdict(diagnostics)
                                    legal_batch = v2.parse_proposer_response(
                                        context_id=str(
                                            chunk.get("chunk_key") or chunk["id"]
                                        ),
                                        content=str(chunk.get("content") or ""),
                                        mentions=chunk_mentions,
                                        logical_schema=context["schema"],
                                        raw=raw_proposal,
                                        candidate_source=v2.DIRECT_PROPOSER,
                                        pass_no=pass_no,
                                        proposer_model_revision=
                                            context["model_revision"],
                                    )
                                    parsed_response = {
                                        "raw": raw_proposal,
                                        "proposals": [
                                            {
                                                "proposal_id": item.proposal_id,
                                                "subject_occurrence_id":
                                                    item.subject_occurrence_key,
                                                "predicate": item.predicate,
                                                "object_occurrence_id":
                                                    item.object_occurrence_key,
                                                "evidence_spans": [
                                                    {
                                                        "start": span.start,
                                                        "end": span.end,
                                                    }
                                                    for span in item.evidence_spans
                                                ],
                                                "qualifiers":
                                                    list(item.qualifiers),
                                                "confidence": item.confidence,
                                            }
                                            for item in legal_batch.proposals
                                        ],
                                        "missing_endpoint_mentions": [
                                            dict(item)
                                            for item in (
                                                legal_batch
                                                .missing_endpoint_mentions
                                            )
                                        ],
                                        "coverage": dict(legal_batch.coverage),
                                    }
                                    self.db.record_shadow_relation_v2_proposal_attempt(
                                        run_id,
                                        {
                                            "context_identity": context_key,
                                            "attempt_no": audit_attempt_no,
                                            "status": "ACCEPTED",
                                            "request_hash": request_hash,
                                            "response_hash":
                                                shadow_model_payload_hash(
                                                    "response",
                                                    raw_response_text,
                                                ),
                                            "proposer_prompt_version":
                                                v2.SHADOW_RELATION_PROPOSER_PROMPT_VERSION,
                                            "proposer_schema_version":
                                                v2.SHADOW_RELATION_PROPOSER_SCHEMA_VERSION,
                                            "proposer_parser_version":
                                                v2.SHADOW_RELATION_PROPOSER_PARSER_VERSION,
                                            "proposer_model_revision":
                                                context["model_revision"],
                                            "model_name": context["model"],
                                            "request_payload": request_payload,
                                            "raw_response_text": raw_response_text,
                                            "parsed_response": parsed_response,
                                            "token_usage": token_usage,
                                            "metadata": {
                                                "chunk_id": int(chunk["id"]),
                                                "pass_no": pass_no,
                                                "retry_no": retry_no,
                                                "proposal_count":
                                                    len(legal_batch.proposals),
                                                "missing_endpoint_mentions":
                                                    len(
                                                        legal_batch
                                                        .missing_endpoint_mentions
                                                    ),
                                            },
                                        },
                                    )
                                    last_proposer_error = None
                                    break
                                except Exception as exc:
                                    proposer_failures += 1
                                    last_proposer_error = exc
                                    raw_response_text = str(
                                        getattr(
                                            self.models,
                                            "last_raw_response_text",
                                            "",
                                        )
                                        or raw_response_text
                                    )
                                    diagnostics = getattr(
                                        self.models, "last_call_diagnostics", None,
                                    )
                                    if diagnostics is not None:
                                        token_usage = asdict(diagnostics)
                                    response_hash = (
                                        shadow_model_payload_hash(
                                            "response", raw_response_text,
                                        )
                                        if raw_response_text else ""
                                    )
                                    self.db.record_shadow_relation_v2_proposal_attempt(
                                        run_id,
                                        {
                                            "context_identity": context_key,
                                            "attempt_no": audit_attempt_no,
                                            "status": (
                                                "TERMINAL_FAILED"
                                                if retry_no == int(
                                                    Config
                                                    .relation_shadow_max_attempts
                                                )
                                                else "RETRYABLE_FAILED"
                                            ),
                                            "request_hash": request_hash,
                                            "response_hash": response_hash,
                                            "proposer_prompt_version":
                                                v2.SHADOW_RELATION_PROPOSER_PROMPT_VERSION,
                                            "proposer_schema_version":
                                                v2.SHADOW_RELATION_PROPOSER_SCHEMA_VERSION,
                                            "proposer_parser_version":
                                                v2.SHADOW_RELATION_PROPOSER_PARSER_VERSION,
                                            "proposer_model_revision":
                                                context["model_revision"],
                                            "model_name": context["model"],
                                            "request_payload": request_payload,
                                            "raw_response_text": raw_response_text,
                                            "parsed_response": parsed_response,
                                            "token_usage": token_usage,
                                            "error_type": type(exc).__name__,
                                            "error_message": str(exc)[:2000],
                                            "metadata": {
                                                "chunk_id": int(chunk["id"]),
                                                "pass_no": pass_no,
                                                "retry_no": retry_no,
                                            },
                                        },
                                    )
                                    continue
                            if legal_batch is None:
                                error_class = (
                                    type(last_proposer_error).__name__
                                    if last_proposer_error else "unknown"
                                )
                                proposer_failure_keys[(
                                    str(chunk.get("document_id") or ""),
                                    context_kind,
                                    error_class,
                                )] += 1
                                proposer_terminal_statuses[
                                    "TERMINAL_FAILED_WITH_PLANNER_FALLBACK"
                                ] += 1
                                proposer_message = (
                                    f"Knowledge Forge 候选补充 "
                                    f"{chunk_index}/{total_chunks} "
                                    f"辅助召回未采用：{error_class}（已审计，主链继续）"
                                )
                                break
                            proposal_batches.append((context_key, legal_batch))
                            context_success = True
                            new_found_keys = {
                                _proposal_found_key(item)
                                for item in legal_batch.proposals
                                if _proposal_found_key(item)
                                not in already_found_keys
                            }
                            already_found_keys.update(new_found_keys)
                            if pass_no == 0:
                                continue
                            if not new_found_keys:
                                break
                        if context_success:
                            proposer_terminal_statuses[
                                "SUCCESS_WITH_PROPOSALS"
                                if already_found_keys else "SUCCESS_EMPTY"
                            ] += 1
                            proposer_message = (
                                f"Knowledge Forge 候选补充 {chunk_index}/{total_chunks} "
                                "完成"
                            )
                if progress:
                    progress(
                        total_chunks + chunk_index,
                        total_progress,
                        proposer_message,
                    )
            opportunity_by_pair = {
                str(item.pair_identity): item for item in all_opportunities
            }
            chunks_by_context = {
                str(chunk.get("shadow_context_id") or chunk["id"]): chunk
                for chunk in relation_chunks
            }
            missing_endpoint_mentions = 0

            def replace_opportunity(updated: v2.PairOpportunity) -> None:
                for index, item in enumerate(all_opportunities):
                    if str(item.pair_identity) == str(updated.pair_identity):
                        all_opportunities[index] = updated
                        break
                for plan_context, plan in list(plans_by_context.items()):
                    replaced = tuple(
                        updated
                        if str(item.pair_identity) == str(updated.pair_identity)
                        else item
                        for item in plan.opportunities
                    )
                    if replaced != plan.opportunities:
                        plans_by_context[plan_context] = replace(
                            plan, opportunities=replaced,
                        )
                opportunity_by_pair[str(updated.pair_identity)] = updated

            def add_direct_opportunity(
                context_key: str,
                opportunity: v2.PairOpportunity,
            ) -> None:
                all_opportunities.append(opportunity)
                plan = plans_by_context[context_key]
                plans_by_context[context_key] = replace(
                    plan,
                    opportunities=(*plan.opportunities, opportunity),
                    theoretical_pairs=max(
                        int(plan.theoretical_pairs),
                        len(plan.opportunities) + 1,
                    ),
                )
                opportunity_by_pair[str(opportunity.pair_identity)] = opportunity

            for context_key, batch in proposal_batches:
                chunk = chunks_by_context[context_key]
                missing_endpoint_mentions += len(batch.missing_endpoint_mentions)
                for proposal in batch.proposals:
                    pair_id = v2.pair_identity(
                        str(chunk["revision_id"]),
                        proposal.subject_occurrence_key,
                        proposal.object_occurrence_key,
                    )
                    opportunity = opportunity_by_pair.get(pair_id)
                    if not opportunity:
                        opportunity = self._direct_proposer_opportunity(
                            run_id=run_id,
                            project_id=project_id,
                            chunk=chunk,
                            proposal=proposal,
                            mentions=mentions_by_context[context_key],
                            logical_schema=context["schema"],
                        )
                        if opportunity is not None:
                            add_direct_opportunity(context_key, opportunity)
                    else:
                        updated = _with_direct_proposer_source(opportunity)
                        if updated != opportunity:
                            opportunity = updated
                            replace_opportunity(opportunity)
            self.db.persist_shadow_relation_v2_opportunities(
                run_id, all_opportunities,
            )
            proposer_hypotheses: list[v2.RelationHypothesis] = []
            proposer_seen_keys: set[str] = set()
            for context_key, batch in proposal_batches:
                chunk = chunks_by_context[context_key]
                for proposal in batch.proposals:
                    found_key = _proposal_found_key(proposal)
                    if found_key in proposer_seen_keys:
                        continue
                    proposer_seen_keys.add(found_key)
                    pair_id = v2.pair_identity(
                        str(chunk["revision_id"]),
                        proposal.subject_occurrence_key,
                        proposal.object_occurrence_key,
                    )
                    opportunity = opportunity_by_pair.get(pair_id)
                    if (
                        opportunity is None
                        or opportunity.status != v2.PENDING_CLASSIFICATION
                    ):
                        continue
                    try:
                        hypothesis = v2.proposal_to_hypothesis(
                            proposal,
                            opportunity,
                            context["schema"],
                        )
                    except v2.ShadowClassificationError as exc:
                        # A direct proposer is auxiliary recall.  If its
                        # candidate violates the frozen planner/schema/scope
                        # contract, reject only that candidate; the admitted
                        # planner pair still proceeds through the classifier.
                        proposer_hypothesis_rejections[str(exc)] += 1
                        continue
                    proposer_hypotheses.append(hypothesis)
            decisions: list[v2.PairClassificationDecision] = []
            classifier_calls = 0
            classifier_failures = 0
            classifier_attempt_counts: Counter[str] = Counter()

            def classify_batch_with_retries(
                *,
                chunk: dict[str, Any],
                batch: list[v2.PairOpportunity],
                batch_offset: int,
            ) -> list[v2.PairClassificationDecision]:
                nonlocal classifier_calls, classifier_failures
                content = str(chunk.get("content") or "")
                last_error: Exception | None = None
                max_attempts = int(Config.relation_shadow_max_attempts)

                for call_attempt_no in range(1, max_attempts + 1):
                    classifier_calls += 1
                    output_budget = self._classifier_output_budget(
                        context_budget_tokens=context_budget,
                        attempt_no=call_attempt_no,
                    )
                    protocol_feedback = ""
                    if last_error is not None:
                        protocol_feedback = (
                            "Previous classifier response violated the v2 "
                            "contract and was rejected fail-closed: "
                            f"{type(last_error).__name__}: "
                            f"{str(last_error)[:800]}. "
                            "Return exactly one decision for each pair. For "
                            "every hypothesis, copy predicate only from that "
                            "pair's allowed_predicates. If no allowed predicate "
                            "fits the text, return NO_RELATION, or "
                            "OTHER_UNSUPPORTED when the text has an explicit "
                            "relation that this Schema cannot express."
                        )
                    prompt = v2.classifier_prompt(
                        batch,
                        content=content,
                        logical_schema=context["schema"],
                        protocol_feedback=protocol_feedback,
                    )
                    response_contract = v2.classifier_response_schema(batch)
                    request_payload = {
                        "stage": "classifier",
                        "model": context["model"],
                        "model_revision": context["model_revision"],
                        "system": SHADOW_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT,
                        "prompt": prompt,
                        "schema": response_contract,
                        "temperature": 0.0,
                        "context_budget_tokens": context_budget,
                        "max_output_tokens": output_budget,
                        "safety_margin_tokens":
                            Config.relation_classifier_safety_margin,
                        "pair_ids": [
                            str(pair.pair_identity) for pair in batch
                        ],
                    }
                    request_hash = shadow_model_payload_hash(
                        "request", request_payload,
                    )
                    attempt_numbers: dict[str, int] = {}
                    for pair in batch:
                        pair_id = str(pair.pair_identity)
                        classifier_attempt_counts[pair_id] += 1
                        attempt_numbers[pair_id] = int(
                            classifier_attempt_counts[pair_id]
                        )
                    token_usage: dict[str, Any] = {}
                    raw_response_text = ""
                    parsed_response: Any = {}
                    try:
                        raw = self._classify(
                            model=context["model"],
                            context_budget_tokens=context_budget,
                            max_output_tokens=output_budget,
                            opportunities=batch,
                            content=content,
                            schema=context["schema"],
                            prompt=prompt,
                            response_schema=response_contract,
                        )
                        parsed_response = raw
                        raw_response_text = str(
                            getattr(self.models, "last_raw_response_text", "")
                            or _safe_json_text(raw)
                        )
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        if diagnostics is not None:
                            token_usage = asdict(diagnostics)
                        batch_decisions = list(v2.parse_classifier_response(
                            batch,
                            raw,
                            content=content,
                            logical_schema=context["schema"],
                            classifier_policy_version=
                                SHADOW_RELATION_V2_CLASSIFIER_POLICY_VERSION,
                            classifier_model_revision=
                                context["model_revision"],
                        ))
                        response_hash = shadow_model_payload_hash(
                            "response", raw_response_text,
                        )
                        for pair in batch:
                            self.db.record_shadow_relation_v2_classification_attempt(
                                run_id,
                                pair.pair_identity,
                                {
                                    "attempt_no":
                                        attempt_numbers[str(pair.pair_identity)],
                                    "status": CLASSIFIER_ACCEPTED,
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
                                    "model_name": context["model"],
                                    "request_payload": request_payload,
                                    "raw_response_text": raw_response_text,
                                    "parsed_response": parsed_response,
                                    "token_usage": token_usage,
                                    "error_type": "",
                                    "independent_call": True,
                                    "metadata": {
                                        "chunk_id": int(chunk["id"]),
                                        "batch_offset": batch_offset,
                                        "batch_size": len(batch),
                                        "max_output_tokens": output_budget,
                                    },
                                },
                            )
                        return batch_decisions
                    except Exception as exc:
                        last_error = exc
                        raw_response_text = str(
                            getattr(self.models, "last_raw_response_text", "")
                            or ""
                        )
                        classifier_failures += 1
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        if diagnostics is not None:
                            token_usage = asdict(diagnostics)
                        terminal = (
                            len(batch) == 1 and call_attempt_no == max_attempts
                        )
                        attempt_status = _classifier_failure_status(
                            exc, terminal=terminal,
                        )
                        response_hash = (
                            shadow_model_payload_hash(
                                "response", raw_response_text,
                            )
                            if raw_response_text else ""
                        )
                        for pair in batch:
                            self.db.record_shadow_relation_v2_classification_attempt(
                                run_id,
                                pair.pair_identity,
                                {
                                    "attempt_no":
                                        attempt_numbers[str(pair.pair_identity)],
                                    "status": attempt_status,
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
                                    "model_name": context["model"],
                                    "request_payload": request_payload,
                                    "raw_response_text": raw_response_text,
                                    "parsed_response": parsed_response,
                                    "token_usage": token_usage,
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc)[:2000],
                                    "independent_call": True,
                                    "metadata": {
                                        "chunk_id": int(chunk["id"]),
                                        "batch_offset": batch_offset,
                                        "batch_size": len(batch),
                                        "max_output_tokens": output_budget,
                                    },
                                },
                            )
                if len(batch) > 1:
                    midpoint = max(1, len(batch) // 2)
                    return [
                        *classify_batch_with_retries(
                            chunk=chunk,
                            batch=batch[:midpoint],
                            batch_offset=batch_offset,
                        ),
                        *classify_batch_with_retries(
                            chunk=chunk,
                            batch=batch[midpoint:],
                            batch_offset=batch_offset + midpoint,
                        ),
                    ]
                return [
                    _terminal_classifier_fail_closed_decision(
                        batch[0], last_error,
                    )
                ]

            for chunk_index, chunk in enumerate(relation_chunks, start=1):
                context_key = str(chunk.get("shadow_context_id") or chunk["id"])
                admitted = list(plans_by_context[context_key].admitted)
                for offset in range(
                    0, len(admitted),
                    Config.relation_classifier_max_candidates,
                ):
                    batch = admitted[
                        offset:offset
                        + Config.relation_classifier_max_candidates
                    ]
                    batch_decisions = classify_batch_with_retries(
                        chunk=chunk,
                        batch=batch,
                        batch_offset=offset,
                    )
                    self.db.persist_shadow_relation_v2_classifications(
                        run_id, batch_decisions,
                    )
                    decisions.extend(batch_decisions)
                if progress:
                    progress(
                        (2 * total_chunks) + chunk_index,
                        total_progress,
                        f"Knowledge Forge 关系分类 {chunk_index}/{total_chunks}",
                    )
            classifier_hypotheses = [
                hypothesis
                for decision in decisions
                for hypothesis in decision.hypotheses
            ]
            classifier_hypothesis_ids = {
                str(item.hypothesis_id) for item in classifier_hypotheses
            }
            proposer_hypotheses_to_persist = [
                item for item in proposer_hypotheses
                if str(item.hypothesis_id) not in classifier_hypothesis_ids
            ]
            if proposer_hypotheses_to_persist:
                self.db.persist_shadow_relation_v2_proposed_hypotheses(
                    run_id,
                    proposer_hypotheses_to_persist,
                    candidate_source=v2.DIRECT_PROPOSER,
                )
            hypothesis_by_key: dict[str, v2.RelationHypothesis] = {}
            for hypothesis in [
                *classifier_hypotheses,
                *proposer_hypotheses_to_persist,
            ]:
                existing = hypothesis_by_key.get(str(hypothesis.hypothesis_id))
                if existing is not None and existing != hypothesis:
                    raise RuntimeError("Shadow v2 hypothesis identity collision")
                hypothesis_by_key[str(hypothesis.hypothesis_id)] = hypothesis
            hypotheses = list(hypothesis_by_key.values())
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
            total_hypotheses = len(hypotheses)
            for hypothesis_index, hypothesis in enumerate(
                hypotheses, start=1,
            ):
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
                if progress:
                    verified_progress = (
                        (3 * total_chunks)
                        + math.ceil(
                            hypothesis_index
                            / max(1, total_hypotheses)
                            * total_chunks
                        )
                    )
                    progress(
                        min(total_progress, verified_progress),
                        total_progress,
                        "Knowledge Forge 关系验证 "
                        f"{hypothesis_index}/{total_hypotheses} 个关系假设",
                    )
            terminal_verifier_attempts = {}
            replay_assertion = getattr(
                self.models, "assert_replay_complete", None,
            )
            if callable(replay_assertion):
                replay_assertion()
            replay_summary = getattr(self.models, "replay_summary", None)
            replay_metrics = (
                dict(replay_summary())
                if callable(replay_summary)
                else {}
            )
            for item in verification_attempts:
                current_attempt = terminal_verifier_attempts.get(
                    str(item.hypothesis_id)
                )
                if (
                    current_attempt is None
                    or int(item.attempt_no) > int(current_attempt.attempt_no)
                ):
                    terminal_verifier_attempts[str(item.hypothesis_id)] = item
            verifier_terminal_failures = sum(
                1 for item in terminal_verifier_attempts.values()
                if item.verdict == v2.RETRYABLE_FAILED
            )
            if verifier_terminal_failures:
                metrics = {
                    "pair_opportunities_total": len(all_opportunities),
                    "pair_opportunities_classified": len(decisions),
                    "pair_statuses": dict(Counter(
                        str(item.status) for item in all_opportunities
                    )),
                    "classification_decisions": dict(Counter(
                        str(item.decision) for item in decisions
                    )),
                    "pair_decision_statuses": dict(Counter(
                        str(item.decision_status) for item in decisions
                    )),
                    "classified_hypotheses": len(hypotheses),
                    "deterministic_validation": dict(Counter(
                        str(item.status) for item in validations
                    )),
                    "verification_attempts": len(verification_attempts),
                    "verification_terminal_count":
                        len(terminal_verifier_attempts),
                    "verification_verdicts": dict(Counter(
                        str(item.verdict)
                        for item in terminal_verifier_attempts.values()
                    )),
                    "verifier_terminal_failures": verifier_terminal_failures,
                    "proposer_model_calls": proposer_calls,
                    "proposer_failures": proposer_failures,
                    "proposer_circuit_breaker_tripped":
                        proposer_circuit_breaker_tripped,
                    "proposer_skipped_by_circuit": proposer_skipped_by_circuit,
                    "proposer_terminal_statuses":
                        dict(proposer_terminal_statuses),
                    "proposer_hypothesis_rejections":
                        dict(proposer_hypothesis_rejections),
                    "proposals_total": len(proposer_hypotheses),
                    "proposer_hypotheses_persisted":
                        len(proposer_hypotheses_to_persist),
                    "missing_endpoint_mentions": missing_endpoint_mentions,
                    "classifier_model_calls": classifier_calls,
                    "classifier_failures": classifier_failures,
                    "verifier_model_calls": verifier_calls,
                    "replay": replay_metrics,
                    "legacy_pipeline_calls": 0,
                    "production_approved": False,
                }
                raise RuntimeError(
                    "Shadow v2 verifier 存在终态失败，拒绝标记 ready 或物化"
                )
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
                "pair_decision_statuses": dict(Counter(
                    str(item.decision_status) for item in decisions
                )),
                "classified_hypotheses": len(hypotheses),
                "deterministic_validation": dict(validation_counts),
                "verification_attempts": len(verification_attempts),
                "verification_terminal_count": len(terminal_attempts),
                "verification_verdicts": dict(verdict_counts),
                "verifier_terminal_failures": int(
                    verdict_counts.get(v2.RETRYABLE_FAILED, 0)
                ),
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
                "proposer_model_calls": proposer_calls,
                "proposer_failures": proposer_failures,
                "proposer_circuit_breaker_tripped":
                    proposer_circuit_breaker_tripped,
                "proposer_skipped_by_circuit": proposer_skipped_by_circuit,
                "proposer_terminal_statuses": dict(proposer_terminal_statuses),
                "proposer_hypothesis_rejections":
                    dict(proposer_hypothesis_rejections),
                "proposals_total": len(proposer_hypotheses),
                "proposer_hypotheses_persisted":
                    len(proposer_hypotheses_to_persist),
                "missing_endpoint_mentions": missing_endpoint_mentions,
                "classifier_model_calls": classifier_calls,
                "classifier_failures": classifier_failures,
                "verifier_model_calls": verifier_calls,
                "replay": replay_metrics,
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
                    f"Knowledge Forge 完成：{len(hypotheses)} 个关系假设，"
                    f"{len(normalization.facts)} 个规范事实，"
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
