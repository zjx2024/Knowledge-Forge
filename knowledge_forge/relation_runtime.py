"""Fail-closed runtime orchestration for Knowledge Forge relations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from . import relation_extraction as v2
from .config import Config
from .entity_publishability import (
    ENTITY_PUBLISHABILITY_POLICY_VERSION,
    canonical_relation_candidate_eligible,
)
from .quality import canonical_digest, canonical_json
from .relation_semantics import (
    RELATION_SEMANTIC_PROJECTION_VERSION,
    converge_relation_display_labels,
    relation_semantic_projection,
    source_expression_matches_schema_cues,
    suppress_exact_open_schema_duplicates,
)
from .macro_context import (
    build_project_macro_context,
    select_macro_context,
)
from .relation_schema_overlay import (
    GROUNDED_RELATION_SCHEMA_OVERLAY_VERSION,
    apply_relation_schema_overlay,
)
from .evidence_context import build_evidence_context_manifest, occurrence_scope
from .sentence_spans import split_relation_scopes
from .relation_secondary import (
    SECONDARY_PASS_VERSION,
    SECONDARY_PARSER_VERSION,
    SECONDARY_POLARITY_PARSER_VERSION,
    SECONDARY_POLARITY_PROMPT_VERSION,
    SECONDARY_POLARITY_SCHEMA_VERSION,
    SECONDARY_PROMPT_VERSION,
    SECONDARY_SCHEMA_VERSION,
    SECONDARY_INSUFFICIENT,
    SECONDARY_NO_RELATION,
    SECONDARY_RELATION,
    ROUTE_EVIDENCE_RECOVERY,
    ROUTE_POLARITY_REVIEW,
    ROUTE_STRUCTURAL_REVIEW,
    ROUTE_NONE,
    ROUTE_PREDICATE_REFINEMENT,
    ROUTE_ROLE_RETRY,
    ROUTE_TECHNICAL_RETRY,
    ROUTE_TYPE_ROLE_REVIEW,
    SecondaryRoute,
    SecondaryDecision,
    parse_secondary_decision,
    parse_source_frame_polarity,
    route_priority,
    route_secondary_attempt,
    secondary_recovery_prompt,
    secondary_response_schema,
    source_frame_polarity_prompt,
    source_frame_polarity_response_schema,
)


def _loaded_relation_implementation_digest() -> str:
    """Fingerprint all relation implementation bytes for diagnostics.

    This is intentionally *not* the snapshot-freshness contract.  A relation
    snapshot is defined by its frozen inputs, prompt/parser versions and
    semantic policy.  Operational changes such as batched ledger persistence,
    progress reporting or retries must not make a verified graph disappear.
    Keep the complete loaded implementation digest available for diagnostics,
    while freshness uses the explicit semantic contract below.
    """

    code_hasher = hashlib.sha256()
    # The relation snapshot may be reused independently of entity extraction.
    # Include every source module that can change candidate routing, prompts,
    # parsing, schema overlay, context admission, or normalization.  Hashing
    # only this orchestrator and the extraction module made a deployed change in
    # relation_secondary.py invisible to the freshness check, so an HTTP
    # rebuild could silently return the old relation snapshot.
    from . import (
        evidence_context,
        macro_context,
        relation_schema_overlay,
        relation_semantics,
        relation_secondary,
        sentence_spans,
    )
    paths = (
        Path(__file__),
        Path(str(v2.__file__)),
        Path(str(relation_secondary.__file__)),
        Path(str(relation_schema_overlay.__file__)),
        Path(str(evidence_context.__file__)),
        Path(str(macro_context.__file__)),
        Path(str(relation_semantics.__file__)),
        Path(str(sentence_spans.__file__)),
    )
    for path in paths:
        code_hasher.update(path.read_bytes())
    return code_hasher.hexdigest()


LOADED_RELATION_IMPLEMENTATION_DIGEST = _loaded_relation_implementation_digest()


# This contract is deliberately versioned rather than derived from every
# implementation byte.  Bump it with any unversioned change that can alter
# candidate selection, model requests/responses, verification, normalization
# or publication.  Prompt/schema/parser and threshold changes already live in
# the immutable fingerprint components, so database I/O and UI-only changes do
# not invalidate an otherwise reusable relation snapshot.
RELATION_RUNTIME_SEMANTIC_CONTRACT_VERSION = (
    "grounded-relation-v2-semantic-contract-2026-08-20-unified-publication-v10"
)


def _loaded_relation_semantic_contract_digest() -> str:
    return canonical_digest({
        "semantic_contract": RELATION_RUNTIME_SEMANTIC_CONTRACT_VERSION,
    })


LOADED_RELATION_CODE_DIGEST = _loaded_relation_semantic_contract_digest()

RELATION_ENDPOINT_INTEGRITY_POLICY_VERSION = (
    "relation-endpoint-integrity-publishability-v2"
)


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


def relation_model_payload_hash(kind: str, payload: Any) -> str:
    """Hash audited model payload bytes with a request/response domain separator."""

    normalized_kind = str(kind)
    if normalized_kind not in {"request", "response"}:
        raise ValueError("relation model payload hash kind must be request or response")
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
    single-pair model response abort the whole grounded relation run. This is a
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


def _source_first_pair_backstop_required(
    opportunity: v2.PairOpportunity,
) -> bool:
    """Keep only independently useful pairwise work after source saturation.

    A complete predicate-open inventory plus its bounded omission pass is the
    primary recall channel. The quadratic pair planner remains useful only
    when immutable parser structure or a source-copied Schema cue supplies an
    independent reason to audit a pair that the inventory did not project.
    """

    provenance = opportunity.provenance or {}
    if provenance.get("source_repair_backstop") is True:
        return True
    structure = str(provenance.get("structure_admission") or "")
    if structure in {
        "same_cell",
        "same_table_row",
        "hierarchical_list_parent_child",
        "hierarchical_section_label_parent_child",
    }:
        return True
    # A cue somewhere in a multi-entity sentence is not evidence for every
    # pair in that sentence.  After a successful source inventory, retain the
    # quadratic backstop only when the schema cue occurs in the exact text
    # between these two endpoint occurrences.  This still covers ordinary
    # active/passive clauses and coordinated objects while preventing
    # ``platform supports A and B`` from becoming ``A supports B``.
    return bool(
        provenance.get("same_atomic_scope") is True
        and provenance.get("pair_local_soft_cue_matches")
    )


def _apply_parser_endpoint_exclusions(
    opportunity: v2.PairOpportunity,
    excluded_occurrence_keys: set[str],
) -> v2.PairOpportunity:
    """Fail closed for parser-owned role labels used as entity endpoints.

    A table header can contain a Schema cue such as ``source`` while still
    being only a column label.  Once the deterministic source parser has
    identified that exact occurrence as role metadata, pairwise lexical
    recovery must not reinterpret it as a row entity.  Row-to-cell facts are
    emitted separately from the parser's immutable column assignment.
    """

    excluded = sorted(
        excluded_occurrence_keys.intersection({
            str(opportunity.left_occurrence_key),
            str(opportunity.right_occurrence_key),
        })
    )
    if not excluded or opportunity.status != v2.PENDING_CLASSIFICATION:
        return opportunity
    return replace(
        opportunity,
        status=v2.FILTERED_STRUCTURE_INELIGIBLE,
        planning_reason="parser_role_label_is_not_relation_endpoint",
        provenance={
            **(opportunity.provenance or {}),
            "parser_endpoint_exclusion": "table_header_role_label",
            "parser_excluded_relation_endpoint_keys": excluded,
            "planner_status_before_parser_endpoint_exclusion": (
                opportunity.status
            ),
        },
    )


GROUNDED_RELATION_V2_CANDIDATE_POLICY_VERSION = (
    "knowledge-forge-source-first-v28-source-boundary-quality-gate"
)
GROUNDED_RELATION_V2_CLASSIFIER_POLICY_VERSION = "knowledge-forge-classifier-v6-project-schema"
GROUNDED_RELATION_V2_NORMALIZATION_VERSION = (
    "knowledge-forge-normalization-v11-publication-fallback-preserved"
)
GROUNDED_RELATION_V2_MATERIALIZATION_VERSION = (
    "knowledge-forge-materialization-v16-unified-role-audit"
)
GROUNDED_RELATION_V2_VERIFIER_GATE_VERSION = (
    "knowledge-forge-verifier-gate-v38-mandatory-theme-projection-recovery"
)
GROUNDED_RELATION_V2_COVERAGE_DECISION_VERSION = (
    "knowledge-forge-atomic-scope-coverage-decision-v1"
)
# These prompt strings retain their historical wording because they are part of
# an immutable request/cache identity. Renaming them would silently change model
# behavior and invalidate persisted replay ledgers; public Python names are current.
GROUNDED_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT = (
    "You are the Shadow relation v2 classifier. Follow the supplied "
    "pair contract exactly and return only one compact schema-valid "
    "JSON value. Do not include <think>, reasoning, markdown, "
    "commentary, or any text outside JSON."
)
GROUNDED_RELATION_V2_PROPOSER_SYSTEM_PROMPT = (
    "You are the Shadow relation v2 direct proposer. Return only compact "
    "schema-valid JSON. Proposals are candidates only and never bypass the "
    "independent verifier."
)
GROUNDED_RELATION_POLICY_SET = {
    "candidate_policy_version": GROUNDED_RELATION_V2_CANDIDATE_POLICY_VERSION,
    "classifier_policy_version": GROUNDED_RELATION_V2_CLASSIFIER_POLICY_VERSION,
    "classifier_prompt_version": v2.GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
    "classifier_schema_version": v2.GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
    "classifier_parser_version": v2.GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
    "proposer_prompt_version": v2.GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
    "proposer_schema_version": v2.GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION,
    "proposer_parser_version": v2.GROUNDED_RELATION_PROPOSER_PARSER_VERSION,
    "verifier_prompt_version": v2.GROUNDED_RELATION_VERIFIER_PROMPT_VERSION,
    "verifier_schema_version": v2.GROUNDED_RELATION_VERIFIER_SCHEMA_VERSION,
    "verifier_parser_version": v2.GROUNDED_RELATION_VERIFIER_PARSER_VERSION,
    "verifier_gate_policy_version": GROUNDED_RELATION_V2_VERIFIER_GATE_VERSION,
    "coverage_decision_policy_version":
        GROUNDED_RELATION_V2_COVERAGE_DECISION_VERSION,
    "role_probe_prompt_version": v2.GROUNDED_RELATION_ROLE_PROBE_PROMPT_VERSION,
    "role_probe_schema_version": v2.GROUNDED_RELATION_ROLE_PROBE_SCHEMA_VERSION,
    "role_probe_parser_version": v2.GROUNDED_RELATION_ROLE_PROBE_PARSER_VERSION,
    "role_alignment_prompt_version":
        v2.GROUNDED_RELATION_ROLE_ALIGNMENT_PROMPT_VERSION,
    "role_alignment_schema_version":
        v2.GROUNDED_RELATION_ROLE_ALIGNMENT_SCHEMA_VERSION,
    "role_alignment_parser_version":
        v2.GROUNDED_RELATION_ROLE_ALIGNMENT_PARSER_VERSION,
    "semantic_contrast_prompt_version":
        v2.GROUNDED_RELATION_SEMANTIC_CONTRAST_PROMPT_VERSION,
    "semantic_contrast_schema_version":
        v2.GROUNDED_RELATION_SEMANTIC_CONTRAST_SCHEMA_VERSION,
    "semantic_contrast_parser_version":
        v2.GROUNDED_RELATION_SEMANTIC_CONTRAST_PARSER_VERSION,
    "atomic_fact_prompt_version":
        v2.GROUNDED_RELATION_ATOMIC_FACT_PROMPT_VERSION,
    "atomic_fact_schema_version":
        v2.GROUNDED_RELATION_ATOMIC_FACT_SCHEMA_VERSION,
    "atomic_fact_parser_version":
        v2.GROUNDED_RELATION_ATOMIC_FACT_PARSER_VERSION,
    "open_fact_prompt_version":
        v2.GROUNDED_RELATION_OPEN_FACT_PROMPT_VERSION,
    "open_fact_primary_contract_version":
        v2.GROUNDED_RELATION_OPEN_FACT_PRIMARY_CONTRACT_VERSION,
    "open_fact_recovery_contract_version":
        v2.GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION,
    "open_fact_schema_version":
        v2.GROUNDED_RELATION_OPEN_FACT_SCHEMA_VERSION,
    "open_fact_parser_version":
        v2.GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION,
    "source_binding_prompt_version":
        v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
    "source_binding_schema_version":
        v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
    "source_binding_parser_version":
        v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
    "source_reference_prompt_version":
        v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION,
    "source_reference_schema_version":
        v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION,
    "source_reference_parser_version":
        v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION,
    "source_recall_shortlist_version":
        v2.GROUNDED_RELATION_SOURCE_RECALL_SHORTLIST_VERSION,
    "source_conflict_prompt_version":
        v2.GROUNDED_RELATION_CONFLICT_PROMPT_VERSION,
    "source_conflict_schema_version":
        v2.GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION,
    "source_conflict_parser_version":
        v2.GROUNDED_RELATION_CONFLICT_PARSER_VERSION,
    "source_role_audit_prompt_version":
        v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION,
    "source_role_audit_schema_version":
        v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_SCHEMA_VERSION,
    "source_role_audit_parser_version":
        v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PARSER_VERSION,
    "publication_audit_prompt_version":
        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION,
    "publication_audit_schema_version":
        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_SCHEMA_VERSION,
    "publication_audit_parser_version":
        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PARSER_VERSION,
    "source_group_audit_prompt_version":
        v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION,
    "source_group_audit_schema_version":
        v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_SCHEMA_VERSION,
    "source_group_audit_parser_version":
        v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PARSER_VERSION,
    "source_group_anchor_audit_prompt_version":
        v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PROMPT_VERSION,
    "source_group_anchor_audit_schema_version":
        v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_SCHEMA_VERSION,
    "source_group_anchor_audit_parser_version":
        v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PARSER_VERSION,
    "source_structure_audit_prompt_version":
        v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PROMPT_VERSION,
    "source_structure_audit_schema_version":
        v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_SCHEMA_VERSION,
    "source_structure_audit_parser_version":
        v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PARSER_VERSION,
    "deterministic_structure_policy_version":
        v2.GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION,
    "deterministic_structure_schema_version":
        v2.GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_SCHEMA_VERSION,
    "deterministic_structure_parser_version":
        v2.GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_PARSER_VERSION,
    "coordinated_recovery_policy_version":
        v2.GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION,
    "coordinated_recovery_schema_version":
        v2.GROUNDED_RELATION_COORDINATED_RECOVERY_SCHEMA_VERSION,
    "coordinated_recovery_parser_version":
        v2.GROUNDED_RELATION_COORDINATED_RECOVERY_PARSER_VERSION,
    "frame_alignment_prompt_version":
        v2.GROUNDED_RELATION_FRAME_ALIGNMENT_PROMPT_VERSION,
    "frame_alignment_schema_version":
        v2.GROUNDED_RELATION_FRAME_ALIGNMENT_SCHEMA_VERSION,
    "frame_alignment_parser_version":
        v2.GROUNDED_RELATION_FRAME_ALIGNMENT_PARSER_VERSION,
    "exact_argument_prompt_version":
        v2.GROUNDED_RELATION_EXACT_ARGUMENT_PROMPT_VERSION,
    "exact_argument_schema_version":
        v2.GROUNDED_RELATION_EXACT_ARGUMENT_SCHEMA_VERSION,
    "exact_argument_parser_version":
        v2.GROUNDED_RELATION_EXACT_ARGUMENT_PARSER_VERSION,
    "normalization_version": GROUNDED_RELATION_V2_NORMALIZATION_VERSION,
    "materialization_version": GROUNDED_RELATION_V2_MATERIALIZATION_VERSION,
    "secondary_pass_version": SECONDARY_PASS_VERSION,
    "secondary_prompt_version": SECONDARY_PROMPT_VERSION,
    "secondary_schema_version": SECONDARY_SCHEMA_VERSION,
    "secondary_parser_version": SECONDARY_PARSER_VERSION,
    "schema_overlay_version": GROUNDED_RELATION_SCHEMA_OVERLAY_VERSION,
    "fallback": "forbidden",
}


def relation_runtime_thresholds() -> dict[str, Any]:
    """Return deployment knobs that can change relation materialization."""

    return {
        "max_attempts": Config.relation_max_attempts,
        "max_gleaning_rounds": 2,
        "secondary_enabled": bool(Config.relation_secondary_enabled),
        "secondary_max_candidates": int(
            Config.relation_secondary_max_candidates
        ),
        "classifier_output_reserve": (
            Config.relation_classifier_output_reserve
        ),
        "classifier_safety_margin": (
            Config.relation_classifier_safety_margin
        ),
        "model_context_max_chars": int(
            Config.relation_model_context_max_chars
        ),
        "classifier_context_max_chars": int(
            Config.relation_classifier_context_max_chars
        ),
        "source_binding_gleaning_rounds": int(
            Config.relation_source_binding_gleaning_rounds
        ),
        "reference_adjudication_enabled": bool(
            Config.relation_reference_adjudication_enabled
        ),
        "source_binding_recall_alternatives": int(
            Config.relation_source_binding_recall_alternatives
        ),
        "source_binding_recall_min_score": float(
            Config.relation_source_binding_recall_min_score
        ),
        "open_carrier_max_per_context": int(
            Config.relation_open_carrier_max_per_context
        ),
        "verifier_batch_size": int(
            Config.relation_verifier_batch_size
        ),
        "verifier_batch_max_chars": int(
            Config.relation_verifier_batch_max_chars
        ),
        "verifier_focused_review_max": int(
            Config.relation_verifier_focused_review_max
        ),
        "verification_mode": str(Config.relation_verification_mode),
    }


def relation_runtime_treatment_digest() -> str:
    """Fingerprint loaded code, versioned contracts and runtime thresholds."""

    return canonical_digest({
        "code_tree": LOADED_RELATION_CODE_DIGEST,
        "policy_set": GROUNDED_RELATION_POLICY_SET,
        "thresholds": relation_runtime_thresholds(),
    })


def _graph_extraction_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the same persisted graph-input subset as entity treatment."""

    selected: list[dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        if metadata.get("graph_extraction_enabled", True) is False:
            continue
        selected.append(chunk)
    return selected


class GroundedRelationRuntime:
    """Run pair planning through isolated materialization without legacy fallback."""

    def __init__(
        self,
        database: Any,
        model_gateway: Any,
        *,
        default_context_tokens: int = 32768,
        processing_payload_loader: (
            Callable[[dict[str, Any]], dict[str, Any] | None] | None
        ) = None,
    ):
        self.db = database
        self.models = model_gateway
        self.default_context_tokens = int(default_context_tokens)
        self.processing_payload_loader = processing_payload_loader

    def _reusable_model_call_lookup(self) -> Callable[..., Any] | None:
        """Return cross-run cache lookup unless this is an audited replay.

        A persisted-call replay must consume every frozen model boundary in
        order.  Cross-run cache reuse is correct for a normal live build but
        would silently bypass the replay gateway and make its final coverage
        assertion fail.  The gateway's explicit provenance marker is the
        narrow capability boundary; ordinary model clients retain caching.
        """

        if getattr(self.models, "replay_provenance", None) is not None:
            return None
        lookup = getattr(self.db, "find_reusable_relation_model_call", None)
        return lookup if callable(lookup) else None

    def _record_classifier_attempts(
        self,
        run_id: str,
        attempts: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Persist one classifier response's per-pair audit rows together.

        Every pair retains an independent immutable decision trail, while a
        normal classifier response is shared by a bounded batch.  Database
        implementations that support the batch API can therefore commit that
        shared boundary in one transaction; narrow test doubles and older
        adapters keep the exact single-row fallback.
        """

        batch_record = getattr(
            self.db, "record_grounded_relation_classification_attempts", None,
        )
        if callable(batch_record):
            batch_record(run_id, attempts)
            return
        record_one = getattr(
            self.db, "record_grounded_relation_classification_attempt", None,
        )
        if not callable(record_one):
            raise RuntimeError("缺少关系分类审计持久化接口")
        for pair_identity, payload in attempts:
            record_one(run_id, pair_identity, payload)

    def _model_revision(self, model: str) -> str:
        runtime = self.models.effective_model_context(model)
        revision = str(runtime.digest or "").strip()
        if not revision:
            raise RuntimeError(
                "Knowledge Forge 关系运行无法冻结模型 digest，拒绝运行"
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
        chunks = _graph_extraction_chunks(chunks)
        chunks_by_document: dict[int, list[dict[str, Any]]] = {}
        for chunk in chunks:
            chunks_by_document.setdefault(int(chunk["document_id"]), []).append(chunk)

        def load_processing_payload(document: dict[str, Any]) -> dict[str, Any] | None:
            if self.processing_payload_loader is not None:
                payload = self.processing_payload_loader(document)
                return payload if isinstance(payload, dict) else None
            loader = getattr(
                self.db, "get_document_processing_payload", None,
            )
            if not callable(loader):
                return None
            payload = loader(document)
            return payload if isinstance(payload, dict) else None

        evidence_context = build_evidence_context_manifest(
            namespace_id=f"relation-shadow:{project_id}:{namespace_id}",
            documents=documents,
            chunks_by_document=chunks_by_document,
            payload_loader=load_processing_payload,
        )
        chunk_by_id = {int(item["id"]): item for item in chunks}
        relation_chunks: list[dict[str, Any]] = []
        source_context_rows: list[dict[str, Any]] = []
        for unit in evidence_context.context_units:
            structure = dict(unit.structure or {})
            chunk_id = int(structure.get("chunk_id") or 0)
            source_chunk = chunk_by_id.get(chunk_id)
            if source_chunk is None:
                if unit.structure_quality != "legacy_chunk_local":
                    raise RuntimeError(
                        "Knowledge Forge structured context cannot be "
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
                "structure_artifact_hash": evidence_context.artifact_hash,
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
            raise RuntimeError("Knowledge Forge 关系运行缺少 active Schema")
        schema = apply_relation_schema_overlay(
            dict(schema_record.get("schema_json") or {})
        )
        mentions = self.db.list_grounded_entity_mentions(namespace_id)
        snapshot = self.db.load_grounded_entity_snapshot(namespace_id)
        namespace_policy = (snapshot.get("namespace") or {}).get("policy_set") or {}
        expected_structure_hash = str(
            namespace_policy.get("structure_artifact_hash") or ""
        )
        if (
            (self.processing_payload_loader is not None and not expected_structure_hash)
            or (
                expected_structure_hash
                and expected_structure_hash != evidence_context.artifact_hash
            )
        ):
            raise RuntimeError(
                "Knowledge Forge 实体与关系阶段的结构上下文不一致，"
                "拒绝降级或混用旧 Chunk"
            )
        all_assignments = list(snapshot.get("assignments") or [])
        all_assignment_by_mention = {
            str(item["mention_key"]): item for item in all_assignments
        }
        if len(all_assignment_by_mention) != len(all_assignments):
            raise RuntimeError("Knowledge Forge 关系 assignment occurrence 不唯一")
        missing = sorted(
            str(item["mention_key"]) for item in mentions
            if str(item["mention_key"]) not in all_assignment_by_mention
        )
        if missing:
            raise RuntimeError(
                "Knowledge Forge 关系 mention 缺少冻结 assignment: "
                + ",".join(missing[:10])
            )
        canonical_values = list(snapshot.get("canonicals") or [])
        quarantined_canonicals = {
            str(item["canonical_key"])
            for item in canonical_values
            if bool((item.get("metadata") or {}).get("quarantined"))
        }
        publication_ineligible_canonicals = {
            str(item["canonical_key"])
            for item in canonical_values
            if not canonical_relation_candidate_eligible(item)
        }
        excluded_canonicals = (
            quarantined_canonicals | publication_ineligible_canonicals
        )
        assignments = all_assignments
        if excluded_canonicals:
            assignments = [
                item for item in assignments
                if str(item.get("canonical_key") or "") not in excluded_canonicals
            ]
        assignment_by_mention = {
            str(item["mention_key"]): item for item in assignments
        }
        if len(assignment_by_mention) != len(assignments):
            raise RuntimeError("Knowledge Forge 关系 assignment occurrence 不唯一")
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
                "Knowledge Forge 关系 mention 指向非当前 Chunk: "
                + ",".join(str(value) for value in unknown_chunks[:10])
            )
        macro_context = build_project_macro_context(
            documents=documents,
            chunks=chunks,
            mentions=mentions,
            assignments=assignments,
            canonicals=[
                item for item in canonical_values
                if str(item.get("canonical_key") or "") not in excluded_canonicals
            ],
        )
        model = str(project.get("llm_model") or Config.default_llm_model)
        model_revision = self._model_revision(model)
        assignment_payload = [{
            "mention_key": str(item["mention_key"]),
            "canonical_key": str(item["canonical_key"]),
            "entity_type": str(item.get("entity_type") or ""),
            "policy_version": str(item.get("policy_version") or ""),
            "revision_id": str(item.get("revision_id") or ""),
            "entity_publishability_policy_version": (
                ENTITY_PUBLISHABILITY_POLICY_VERSION
            ),
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
            "publishability_decision": str(
                item.get("filter_decision") or "accepted"
            ),
            "publishability_policy_version": str(
                (item.get("provenance") or {}).get(
                    "publishability_policy_version"
                ) or ENTITY_PUBLISHABILITY_POLICY_VERSION
            ),
        } for item in sorted(
            mentions, key=lambda value: str(value["mention_key"]),
        )]
        candidate = {
            "policy_version": GROUNDED_RELATION_V2_CANDIDATE_POLICY_VERSION,
            "parameters": {
                "sentence_window": Config.relation_sentence_window,
                "max_char_distance": Config.relation_max_char_distance,
                "batch_size": Config.relation_classifier_max_candidates,
                "soft_cues": "schema-configured-nonbinding",
                "source_context_policy": evidence_context.policy_version,
                "source_map_policy": evidence_context.source_map_policy,
                "macro_context_version": str(
                    macro_context.get("version") or ""
                ),
                "macro_context_hash": str(macro_context.get("hash") or ""),
                "entity_publishability_policy_version": (
                    ENTITY_PUBLISHABILITY_POLICY_VERSION
                ),
            },
            "budget": Config.relation_max_candidates_per_chunk,
            "structure_artifact_hash": evidence_context.artifact_hash,
        }
        classifier = {
            "prompt_version": v2.GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
            "schema_version": v2.GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
            "parser_version": v2.GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
            "model": model,
            "model_revision": model_revision,
        }
        proposer = {
            "prompt_version": v2.GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
            "schema_version": v2.GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION,
            "parser_version": v2.GROUNDED_RELATION_PROPOSER_PARSER_VERSION,
            "model": model,
            "model_revision": model_revision,
            "max_gleaning_rounds": 2,
            "failure_circuit_breaker":
                Config.relation_proposer_failure_circuit_breaker,
            "open_fact_prompt_version":
                v2.GROUNDED_RELATION_OPEN_FACT_PROMPT_VERSION,
            "open_fact_primary_contract_version":
                v2.GROUNDED_RELATION_OPEN_FACT_PRIMARY_CONTRACT_VERSION,
            "open_fact_recovery_contract_version":
                v2.GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION,
            "open_fact_schema_version":
                v2.GROUNDED_RELATION_OPEN_FACT_SCHEMA_VERSION,
            "open_fact_parser_version":
                v2.GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION,
        }
        verifier = {
            "prompt_version": v2.GROUNDED_RELATION_VERIFIER_PROMPT_VERSION,
            "schema_version": v2.GROUNDED_RELATION_VERIFIER_SCHEMA_VERSION,
            "parser_version": v2.GROUNDED_RELATION_VERIFIER_PARSER_VERSION,
            "execution_mode": (
                "source-first-plus-direct-candidate-union-plus-deterministic-"
                "validation-plus-parser-structure-terminal-plus-batched-"
                "ordered-final-verifier-v9"
                if Config.relation_verification_mode == "lean"
                else
                "source-first-plus-legacy-multi-audit-consensus-v7"
            ),
            "output_budget_policy": "bounded-4k-to-8k-schema-limited-v1",
            "model": model,
            "model_revision": model_revision,
            "role_probe_prompt_version":
                v2.GROUNDED_RELATION_ROLE_PROBE_PROMPT_VERSION,
            "role_probe_schema_version":
                v2.GROUNDED_RELATION_ROLE_PROBE_SCHEMA_VERSION,
            "role_probe_parser_version":
                v2.GROUNDED_RELATION_ROLE_PROBE_PARSER_VERSION,
            "role_alignment_prompt_version":
                v2.GROUNDED_RELATION_ROLE_ALIGNMENT_PROMPT_VERSION,
            "role_alignment_schema_version":
                v2.GROUNDED_RELATION_ROLE_ALIGNMENT_SCHEMA_VERSION,
            "role_alignment_parser_version":
                v2.GROUNDED_RELATION_ROLE_ALIGNMENT_PARSER_VERSION,
            "semantic_contrast_prompt_version":
                v2.GROUNDED_RELATION_SEMANTIC_CONTRAST_PROMPT_VERSION,
            "semantic_contrast_schema_version":
                v2.GROUNDED_RELATION_SEMANTIC_CONTRAST_SCHEMA_VERSION,
            "semantic_contrast_parser_version":
                v2.GROUNDED_RELATION_SEMANTIC_CONTRAST_PARSER_VERSION,
            "atomic_fact_prompt_version":
                v2.GROUNDED_RELATION_ATOMIC_FACT_PROMPT_VERSION,
            "atomic_fact_schema_version":
                v2.GROUNDED_RELATION_ATOMIC_FACT_SCHEMA_VERSION,
            "atomic_fact_parser_version":
                v2.GROUNDED_RELATION_ATOMIC_FACT_PARSER_VERSION,
            "source_binding_prompt_version":
                v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
            "source_binding_schema_version":
                v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
            "source_binding_parser_version":
                v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
            "source_reference_prompt_version":
                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION,
            "source_reference_schema_version":
                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION,
            "source_reference_parser_version":
                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION,
            "source_recall_shortlist_version":
                v2.GROUNDED_RELATION_SOURCE_RECALL_SHORTLIST_VERSION,
            "source_conflict_prompt_version":
                v2.GROUNDED_RELATION_CONFLICT_PROMPT_VERSION,
            "source_conflict_schema_version":
                v2.GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION,
            "source_conflict_parser_version":
                v2.GROUNDED_RELATION_CONFLICT_PARSER_VERSION,
            "source_role_audit_prompt_version":
                v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION,
            "source_role_audit_schema_version":
                v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_SCHEMA_VERSION,
            "source_role_audit_parser_version":
                v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PARSER_VERSION,
            "source_group_audit_prompt_version":
                v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION,
            "source_group_audit_schema_version":
                v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_SCHEMA_VERSION,
            "source_group_audit_parser_version":
                v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PARSER_VERSION,
            "source_group_anchor_audit_prompt_version":
                v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PROMPT_VERSION,
            "source_group_anchor_audit_schema_version":
                v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_SCHEMA_VERSION,
            "source_group_anchor_audit_parser_version":
                v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PARSER_VERSION,
            "source_structure_audit_prompt_version":
                v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PROMPT_VERSION,
            "source_structure_audit_schema_version":
                v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_SCHEMA_VERSION,
            "source_structure_audit_parser_version":
                v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PARSER_VERSION,
            "deterministic_structure_policy_version":
                v2.GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_POLICY_VERSION,
            "deterministic_structure_schema_version":
                v2.GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_SCHEMA_VERSION,
            "deterministic_structure_parser_version":
                v2.GROUNDED_RELATION_DETERMINISTIC_STRUCTURE_PARSER_VERSION,
            "coordinated_recovery_policy_version":
                v2.GROUNDED_RELATION_COORDINATED_RECOVERY_POLICY_VERSION,
            "coordinated_recovery_schema_version":
                v2.GROUNDED_RELATION_COORDINATED_RECOVERY_SCHEMA_VERSION,
            "coordinated_recovery_parser_version":
                v2.GROUNDED_RELATION_COORDINATED_RECOVERY_PARSER_VERSION,
            "batch_verifier_prompt_version":
                v2.GROUNDED_RELATION_BATCH_VERIFIER_PROMPT_VERSION,
            "batch_verifier_schema_version":
                v2.GROUNDED_RELATION_BATCH_VERIFIER_SCHEMA_VERSION,
            "batch_verifier_parser_version":
                v2.GROUNDED_RELATION_BATCH_VERIFIER_PARSER_VERSION,
            "frame_alignment_prompt_version":
                v2.GROUNDED_RELATION_FRAME_ALIGNMENT_PROMPT_VERSION,
            "frame_alignment_schema_version":
                v2.GROUNDED_RELATION_FRAME_ALIGNMENT_SCHEMA_VERSION,
            "frame_alignment_parser_version":
                v2.GROUNDED_RELATION_FRAME_ALIGNMENT_PARSER_VERSION,
            "exact_argument_prompt_version":
                v2.GROUNDED_RELATION_EXACT_ARGUMENT_PROMPT_VERSION,
            "exact_argument_schema_version":
                v2.GROUNDED_RELATION_EXACT_ARGUMENT_SCHEMA_VERSION,
            "exact_argument_parser_version":
                v2.GROUNDED_RELATION_EXACT_ARGUMENT_PARSER_VERSION,
        }
        policy_root = canonical_digest({
            "candidate": candidate,
            "proposer": proposer,
            "classifier": classifier,
            "verifier": verifier,
            "relation_policy": GROUNDED_RELATION_POLICY_SET,
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
            thresholds=relation_runtime_thresholds(),
            normalization={
                "version": GROUNDED_RELATION_V2_NORMALIZATION_VERSION,
                "symmetric_rule": "sort-canonical-endpoints",
                "inverse_rule": "declared-canonical-forward-only",
            },
            publish_policy={
                "version": GROUNDED_RELATION_V2_MATERIALIZATION_VERSION,
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
                "treatment": relation_runtime_treatment_digest(),
                "policy_root": policy_root,
            },
        )
        return {
            "project": project,
            "documents": documents,
            "chunks": chunks,
            "relation_chunks": relation_chunks,
            "chunk_by_id": chunk_by_id,
            "shadow_context": evidence_context,
            "source_context_rows": source_context_rows,
            "schema": schema,
            "mentions": mentions,
            "assignments": assignments,
            "assignment_by_mention": assignment_by_mention,
            "entity_publishability": {
                "policy_version": ENTITY_PUBLISHABILITY_POLICY_VERSION,
                "canonicals_total": len(canonical_values),
                "canonicals_relation_eligible": (
                    len(canonical_values) - len(excluded_canonicals)
                ),
                "canonicals_excluded": len(excluded_canonicals),
                "resolver_quarantined": len(quarantined_canonicals),
                "publication_ineligible": len(
                    publication_ineligible_canonicals
                ),
            },
            "macro_context": macro_context,
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

    @classmethod
    def _relation_occurrence_key(
        cls,
        mention: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> str:
        """Make an occurrence key stable within the frozen source map.

        Entity extraction may intentionally reuse a physical anchor across
        overlapping contexts.  The same anchor can also appear in unrelated
        structured blocks (for example, a repeated page header or a parser
        scope id).  Relation opportunities must distinguish the latter or
        two different source spans collapse to one immutable pair identity.
        Mapping the mention back to its source block preserves overlap
        deduplication while separating unrelated blocks.  Legacy contexts use
        their chunk-local source block and therefore remain deterministic.
        """

        physical = cls._physical_occurrence_key(dict(mention))
        if str(context.get("shadow_context_quality") or "") == (
            "legacy_chunk_local"
        ):
            return physical
        mention_start = int(mention["mention_start"])
        mention_end = int(mention["mention_end"])
        candidates: list[tuple[int, tuple[str, ...]]] = []
        for raw_entry in context.get("shadow_source_map") or []:
            if not isinstance(raw_entry, Mapping):
                continue
            try:
                rendered_start = int(raw_entry.get("rendered_start") or 0)
                rendered_end = int(raw_entry.get("rendered_end") or 0)
            except (TypeError, ValueError):
                continue
            if not (
                rendered_start <= mention_start
                and mention_end <= rendered_end
                and rendered_end > rendered_start
            ):
                continue
            block_key = str(raw_entry.get("block_key") or "")
            if not block_key:
                continue
            local_start = int(
                raw_entry.get("block_local_start") or 0
            ) + mention_start - rendered_start
            local_end = int(
                raw_entry.get("block_local_start") or 0
            ) + mention_end - rendered_start
            source_identity = (
                str(raw_entry.get("anchor_kind") or ""),
                block_key,
                str(local_start),
                str(local_end),
                str(raw_entry.get("table_id") or ""),
                str(raw_entry.get("row_id") or ""),
                str(raw_entry.get("cell_id") or ""),
            )
            candidates.append((rendered_end - rendered_start, source_identity))
        if candidates:
            _width, source_identity = min(candidates, key=lambda item: item[0])
        else:
            source_identity = (
                "chunk_local",
                str(context.get("revision_id") or ""),
                str(context.get("id") or context.get("chunk_key") or ""),
                str(mention_start),
                str(mention_end),
            )
        material = "\x1f".join((
            "knowledge-forge-relation-occurrence-v2",
            physical,
            *source_identity,
        ))
        return "occurrence_" + hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()

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
                "physical_occurrence_key": (
                    GroundedRelationRuntime._physical_occurrence_key(mention)
                ),
                "occurrence_key": (
                    GroundedRelationRuntime._relation_occurrence_key(
                        mention, chunk,
                    )
                ),
                "source_role": str(profile["source_role"]),
                "relation_eligible": bool(profile["relation_eligible"]),
                "scope_id": str(scope["scope_id"]),
                "scope_kind": str(scope["scope_kind"]),
                "scope_start": int(scope["start"]),
                "scope_end": int(scope["end"]),
                "list_group_id": str(scope.get("list_group_id") or ""),
                "list_depth": int(scope.get("list_depth") or 0),
                "list_parent_scope_id": str(
                    scope.get("list_parent_scope_id") or ""
                ),
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
        # A list-item scope may contain a principal label plus several nested
        # mentions, for example ``Monitoring (core, partition)``.  Scope
        # membership alone proves that the item label is a child of its list
        # parent; it does not promote every mention inside the item to the same
        # hierarchy level.  Mark only the earliest, longest co-starting mention
        # as the item node.  Parenthetical/detail relations remain available
        # through their own atomic source frames.
        by_list_scope: dict[str, list[dict[str, Any]]] = {}
        for item in enriched:
            item["list_scope_head"] = False
            if str(item.get("source_role") or "") != "list_item":
                continue
            scope_id = str(item.get("scope_id") or "")
            if scope_id:
                by_list_scope.setdefault(scope_id, []).append(item)
        for values in by_list_scope.values():
            first_start = min(int(item["mention_start"]) for item in values)
            first_end = max(
                int(item["mention_end"])
                for item in values
                if int(item["mention_start"]) == first_start
            )
            for item in values:
                item["list_scope_head"] = bool(
                    int(item["mention_start"]) == first_start
                    and int(item["mention_end"]) == first_end
                )
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
        """Create a ledger opportunity for one exact local-subgraph proposal.

        The atomic-context, endpoint, type and Schema gates remain mandatory.
        A character-distance rejection may be lifted because the model supplied
        an explicit relation with both exact endpoints inside the same frozen
        ContextUnit; semantic acceptance still belongs to the independent
        verifier.  Structural/scope/type rejections are never bypassed.
        """

        mentions_by_occurrence = {
            str(item["occurrence_key"]): item for item in mentions
        }
        try:
            subject = mentions_by_occurrence[proposal.subject_occurrence_key]
            object_ = mentions_by_occurrence[proposal.object_occurrence_key]
        except KeyError as exc:
            raise RuntimeError(
                "relation proposer endpoint missing from context"
            ) from exc
        content = str(chunk.get("content") or "")
        if not content:
            raise RuntimeError("relation proposer context is empty")
        plan = v2.plan_pair_opportunities(
            project_id=int(project_id),
            document_revision_id=str(chunk["revision_id"]),
            chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
            content=content,
            mentions=[subject, object_],
            logical_schema=logical_schema,
            run_id=run_id,
            candidate_policy_version=GROUNDED_RELATION_V2_CANDIDATE_POLICY_VERSION,
            max_pair_opportunities=None,
        )
        if len(plan.opportunities) != 1:
            return None
        planned = plan.opportunities[0]
        allowed = {
            str(item.get("name") or "")
            for item in planned.allowed_predicate_metadata
        }
        if proposal.predicate not in allowed:
            return None
        if planned.status == v2.FILTERED_DISTANCE:
            planned = replace(
                planned,
                status=v2.PENDING_CLASSIFICATION,
                planning_reason="explicit_local_subgraph_distance_override",
                provenance={
                    **planned.provenance,
                    "planner_status_before_local_subgraph": v2.FILTERED_DISTANCE,
                    "distance_override": True,
                },
            )
        if planned.status != v2.PENDING_CLASSIFICATION:
            return None
        source_context_id = str(chunk.get("shadow_context_id") or chunk["id"])
        return replace(
            planned,
            planning_reason=(
                planned.planning_reason
                if planned.planning_reason == "explicit_local_subgraph_distance_override"
                else "local_subgraph_pair_admitted_by_structure_gate"
            ),
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

    @staticmethod
    def _promote_local_subgraph_opportunity(
        opportunity: v2.PairOpportunity,
        proposal: v2.RelationProposal,
    ) -> v2.PairOpportunity:
        """Admit an explicit local proposal without bypassing hard gates.

        The exhaustive planner deliberately records every pair, including
        pairs deferred only because a per-context processing budget was
        exhausted and otherwise valid pairs outside the heuristic distance
        window.  A local-subgraph proposal is semantic evidence that those two
        exact occurrences deserve verification, so those two *recall-only*
        states may be promoted.  Type, scope, source-role, self-loop and other
        structural terminal states remain non-admissible.
        """

        allowed = {
            str(item.get("name") or "")
            for item in opportunity.allowed_predicate_metadata
        }
        if proposal.predicate not in allowed:
            return opportunity
        previous = str(opportunity.status)
        promotable = {v2.DEFERRED_BUDGET, v2.FILTERED_DISTANCE}
        if previous not in promotable | {v2.PENDING_CLASSIFICATION}:
            return opportunity
        reason = {
            v2.DEFERRED_BUDGET: "explicit_local_subgraph_budget_override",
            v2.FILTERED_DISTANCE: "explicit_local_subgraph_distance_override",
        }.get(previous, opportunity.planning_reason)
        return replace(
            opportunity,
            status=v2.PENDING_CLASSIFICATION,
            planning_reason=reason,
            provenance={
                **opportunity.provenance,
                "planner_status_before_local_subgraph": str(
                    opportunity.provenance.get(
                        "planner_status_before_local_subgraph",
                    ) or previous
                ),
                "candidate_sources": sorted({
                    *(
                        opportunity.provenance.get("candidate_sources")
                        if isinstance(
                            opportunity.provenance.get("candidate_sources"),
                            list,
                        ) else []
                    ),
                    v2.DIRECT_PROPOSER,
                }),
                "source_kind": "v2_direct_proposer",
                "source_record_id": proposal.proposal_id,
                "local_subgraph_recall_override": previous in promotable,
                "direct_proposer_union": True,
            },
        )

    @staticmethod
    def _grounded_open_fact_opportunity(
        *,
        run_id: str,
        project_id: int,
        chunk: Mapping[str, Any],
        frame: Mapping[str, Any],
        mentions: list[dict[str, Any]],
        logical_schema: dict[str, Any],
    ) -> v2.PairOpportunity | None:
        """Create a pair ledger row from one endpoint-grounded open fact.

        The semantic frame may lift heuristic budget, distance, atomic-scope
        and endpoint-type filters. Entity typing is model-derived and can be
        wrong even when the source copied both endpoints and their predicate;
        for this exact source-grounded path it is therefore a verifier prior,
        not a candidate-generation veto. Self loops, peer-table rows and
        source-role exclusions remain hard gates.
        """

        keys = list(frame.get("argument_occurrence_keys") or [])
        if len(keys) != 2 or not all(str(value) for value in keys):
            return None
        by_key = {
            str(item.get("occurrence_key") or ""): item for item in mentions
        }
        subject = by_key.get(str(keys[0]))
        object_ = by_key.get(str(keys[1]))
        if subject is None or object_ is None:
            return None
        content = str(chunk.get("content") or "")
        source_endpoint_types = [
            str(subject.get("type") or ""),
            str(object_.get("type") or ""),
        ]
        plan = v2.plan_pair_opportunities(
            project_id=int(project_id),
            document_revision_id=str(chunk["revision_id"]),
            chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
            content=content,
            mentions=[
                {**subject, "type": ""},
                {**object_, "type": ""},
            ],
            logical_schema=logical_schema,
            run_id=run_id,
            candidate_policy_version=GROUNDED_RELATION_V2_CANDIDATE_POLICY_VERSION,
            max_pair_opportunities=None,
        )
        if len(plan.opportunities) != 1:
            return None
        planned = plan.opportunities[0]
        previous = str(planned.status)
        promotable = {
            v2.PENDING_CLASSIFICATION,
            # Unknown/generic endpoint types are not negative evidence. The
            # projected predicate still passes exact Schema role/type review
            # and the independent verifier, so preserve the source frame's
            # full evidence context instead of stranding it in the planner's
            # endpoint-only window.
            v2.TYPE_UNRESOLVED,
            v2.DEFERRED_BUDGET,
            v2.FILTERED_DISTANCE,
            v2.FILTERED_SCOPE_BOUNDARY,
        }
        if previous not in promotable:
            return None
        planned_sources = (
            planned.provenance.get("candidate_sources")
            if isinstance(
                planned.provenance.get("candidate_sources"), list,
            ) else []
        )
        evidence_span = list(frame.get("evidence_span") or [])
        evidence_span_valid = bool(
            len(evidence_span) == 2
            and all(isinstance(value, int) for value in evidence_span)
            and 0 <= evidence_span[0] < evidence_span[1] <= len(content)
        )
        if evidence_span_valid:
            context_span = (
                min(
                    int(evidence_span[0]),
                    int(subject["mention_start"]),
                    int(object_["mention_start"]),
                ),
                max(
                    int(evidence_span[1]),
                    int(subject["mention_end"]),
                    int(object_["mention_end"]),
                ),
            )
        else:
            context_span = planned.context_span
        source_context_id = str(
            chunk.get("shadow_context_id") or chunk["id"]
        )
        resolved_reference_endpoints: list[dict[str, Any]] = []
        for raw_reference in frame.get("resolved_reference_endpoints") or ():
            if not isinstance(raw_reference, Mapping):
                continue
            reference = dict(raw_reference)
            proxy = by_key.get(str(reference.get("occurrence_key") or ""))
            antecedent = by_key.get(str(
                reference.get("antecedent_occurrence_key") or ""
            ))
            for occurrence, field_name in (
                (proxy, "reference_span"),
                (antecedent, "antecedent_span"),
            ):
                if occurrence is None:
                    continue
                try:
                    span = [
                        int(occurrence["mention_start"]),
                        int(occurrence["mention_end"]),
                    ]
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= span[0] < span[1] <= len(content):
                    reference[field_name] = span
            resolved_reference_endpoints.append(reference)
        antecedent_bounds = [
            item.get("antecedent_span")
            for item in resolved_reference_endpoints
            if isinstance(item.get("antecedent_span"), list)
            and len(item["antecedent_span"]) == 2
        ]
        if antecedent_bounds:
            context_span = (
                min(
                    int(context_span[0]),
                    *(int(value[0]) for value in antecedent_bounds),
                ),
                max(
                    int(context_span[1]),
                    *(int(value[1]) for value in antecedent_bounds),
                ),
            )
        return replace(
            planned,
            status=v2.PENDING_CLASSIFICATION,
            context_span=context_span,
            cue=v2.GROUNDED_OPEN_FACT,
            planning_reason=(
                "endpoint_grounded_open_fact"
                if previous == v2.PENDING_CLASSIFICATION
                else f"endpoint_grounded_open_fact_override:{previous}"
            ),
            provenance={
                **planned.provenance,
                "planner_status_before_open_fact": previous,
                "candidate_sources": sorted({
                    *planned_sources,
                    v2.GROUNDED_OPEN_FACT,
                }),
                "source_kind": "grounded_open_fact",
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
                "open_relation_expression": str(frame.get("relation") or ""),
                "open_fact_evidence_span": evidence_span,
                # A validated source frame is itself the bounded proposition
                # being audited.  Table-header assignments and resolved
                # cross-sentence references may legitimately cover more than
                # the pair planner's lexical sentence scope.  Widen only to
                # the frame's immutable exact evidence; endpoint coverage,
                # source bytes, Schema roles and independent verification all
                # remain mandatory.
                "required_atomic_scope": (
                    list(evidence_span)
                    if evidence_span_valid
                    else list(
                        planned.provenance.get("required_atomic_scope") or []
                    )
                ),
                "source_frame_atomic_scope_override": bool(
                    evidence_span_valid
                    and list(
                        planned.provenance.get("required_atomic_scope") or []
                    ) != list(evidence_span)
                ),
                "source_endpoint_types": source_endpoint_types,
                "source_type_constraint_mode": (
                    "soft_source_fact_requires_verification"
                ),
                "resolved_reference_endpoints": [
                    dict(item) for item in resolved_reference_endpoints
                ],
                "source_structure_kind": str(
                    frame.get("deterministic_source_structure") or ""
                ),
                "source_structure_role_label": str(
                    frame.get("structural_role_label") or ""
                ),
                "source_syntax_origin": str(
                    frame.get("deterministic_source_syntax") or ""
                ),
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
            system=GROUNDED_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT,
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
        required_evidence_span: tuple[int, int] | None = None,
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
                required_evidence_span=required_evidence_span,
            )
        )
        request_schema = (
            response_schema if response_schema is not None
            else v2.proposer_response_schema(
                mentions=mentions,
                logical_schema=schema,
                content_length=len(content),
                max_proposals=Config.relation_proposer_max_proposals,
                required_evidence_span=required_evidence_span,
            )
        )
        raw = self.models.json_chat(
            model=model,
            system=GROUNDED_RELATION_V2_PROPOSER_SYSTEM_PROMPT,
            prompt=request_prompt,
            schema=request_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=output_budget,
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        return raw

    @staticmethod
    def _proposer_output_budget(
        *, context_budget_tokens: int, attempt_no: int,
    ) -> int:
        # A complete local subgraph may contain several relations with long,
        # exact occurrence identifiers.  2K frequently truncates an otherwise
        # valid structured response; start at 4K and then fit the request to
        # the verified runner context immediately before the HTTP call.
        base = max(4096, int(Config.relation_classifier_output_reserve))
        requested = base * (2 ** max(0, int(attempt_no) - 1))
        context_cap = max(
            base,
            int(context_budget_tokens)
            - int(Config.relation_classifier_safety_margin)
            - 1024,
        )
        return max(512, min(requested, context_cap, 12_288))

    def _fit_proposer_output_budget(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        desired_output_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> int:
        """Fit proposer output to the real prompt and runner context.

        This uses the same rendered request components as the Ollama gateway.
        Test doubles without runtime token capabilities retain the desired
        budget.  A prompt that leaves no meaningful output room fails before
        HTTP and is audited as a local proposer failure; the pair classifier
        remains the independent backstop.
        """

        estimator = getattr(self.models, "token_estimator", None)
        estimate = getattr(estimator, "estimate", None)
        effective_context = getattr(self.models, "effective_model_context", None)
        if not callable(estimate) or not callable(effective_context):
            return int(desired_output_tokens)
        runtime = effective_context(model)
        runtime_tokens = int(
            getattr(runtime, "effective_context_tokens", 0) or 0
        )
        if runtime_tokens <= 0:
            return int(desired_output_tokens)
        rendered = json.dumps({
            "messages": [
                {
                    "role": "system",
                    "content": GROUNDED_RELATION_V2_PROPOSER_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            "format": response_schema,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        estimated_input = int(estimate(rendered))
        available = (
            min(int(context_budget_tokens), runtime_tokens)
            - estimated_input
            - int(Config.relation_classifier_safety_margin)
        )
        # Let the gateway produce the audited admission error when even the
        # minimal value cannot fit; do not raise outside the per-attempt audit
        # boundary in the caller.
        return max(1, min(int(desired_output_tokens), available))

    def _propose_atomic_scope(
        self,
        *,
        run_id: str,
        context: Mapping[str, Any],
        context_budget_tokens: int,
        context_key: str,
        chunk: Mapping[str, Any],
        scope_span: tuple[int, int],
        mentions: list[dict[str, Any]],
        attempt_counts: Counter[str],
        macro_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract one source-derived atomic relation scope with full audit.

        The source coordinates remain absolute to the frozen chunk.  Focusing
        the model on one structural scope prevents unrelated sentences and
        coordinated peer lists from competing for a bounded response, while
        the exhaustive pair ledger remains the fallback whenever coverage is
        incomplete or a model call fails.
        """

        content = str(chunk.get("content") or "")
        scope_start, scope_end = map(int, scope_span)
        if not (0 <= scope_start < scope_end <= len(content)):
            raise ValueError("proposer atomic scope is outside frozen content")
        if len(mentions) < 2:
            return {
                "batches": [], "calls": 0, "failures": 0,
                "cache_hits": 0,
                "complete": False, "error_class": "",
                "found_keys": set(),
            }
        chunk_identity = str(chunk.get("chunk_key") or chunk["id"])
        scope_context_id = (
            f"{chunk_identity}:scope:{scope_start}:{scope_end}"
        )
        batches: list[v2.RelationProposalBatch] = []
        already_found_keys: set[str] = set()
        calls = 0
        cache_hits = 0
        failures = 0
        last_error: Exception | None = None
        terminal_complete = False
        max_proposals = int(Config.relation_proposer_max_proposals)

        for pass_no in range(0, 3):
            legal_batch: v2.RelationProposalBatch | None = None
            for retry_no in range(
                1, int(Config.relation_max_attempts) + 1,
            ):
                attempt_counts[context_key] += 1
                audit_attempt_no = int(attempt_counts[context_key])
                output_budget = self._proposer_output_budget(
                    context_budget_tokens=context_budget_tokens,
                    attempt_no=retry_no,
                )
                protocol_feedback = ""
                if last_error is not None:
                    protocol_feedback = (
                        "Previous proposer response violated the contract and "
                        "was rejected fail-closed: "
                        f"{type(last_error).__name__}: "
                        f"{str(last_error)[:800]}. Return exactly one object "
                        "with proposals, missing_endpoint_mentions, and "
                        "coverage. Copy endpoint and predicate identifiers "
                        "from the supplied lists and keep evidence inside the "
                        "required atomic span."
                    )
                prompt = v2.proposer_prompt(
                    context_id=scope_context_id,
                    content=content,
                    mentions=mentions,
                    logical_schema=dict(context["schema"]),
                    already_found_keys=sorted(already_found_keys),
                    pass_no=pass_no,
                    protocol_feedback=protocol_feedback,
                    max_proposals=max_proposals,
                    required_evidence_span=(scope_start, scope_end),
                    macro_context=macro_context,
                )
                response_schema = v2.proposer_response_schema(
                    mentions=mentions,
                    logical_schema=dict(context["schema"]),
                    content_length=len(content),
                    max_proposals=max_proposals,
                    required_evidence_span=(scope_start, scope_end),
                )
                output_budget = self._fit_proposer_output_budget(
                    model=str(context["model"]),
                    context_budget_tokens=context_budget_tokens,
                    desired_output_tokens=output_budget,
                    prompt=prompt,
                    response_schema=response_schema,
                )
                request_payload = {
                    "stage": "proposer",
                    "model": context["model"],
                    "model_revision": context["model_revision"],
                    "system": GROUNDED_RELATION_V2_PROPOSER_SYSTEM_PROMPT,
                    "prompt": prompt,
                    "schema": response_schema,
                    "temperature": 0.0,
                    "context_budget_tokens": context_budget_tokens,
                    "max_output_tokens": output_budget,
                    "safety_margin_tokens":
                        Config.relation_classifier_safety_margin,
                    "context_identity": context_key,
                    "scope_context_id": scope_context_id,
                    "required_evidence_span": [scope_start, scope_end],
                    "macro_context_hash": str(
                        (macro_context or {}).get("hash") or ""
                    ),
                    "pass_no": pass_no,
                    "retry_no": retry_no,
                }
                request_hash = relation_model_payload_hash(
                    "request", request_payload,
                )
                raw_response_text = ""
                parsed_response: Any = {}
                token_usage: dict[str, Any] = {}
                try:
                    cache_lookup = self._reusable_model_call_lookup()
                    cached_call = (
                        cache_lookup(
                            run_id=run_id,
                            component="proposer",
                            model_digest=str(context["model_revision"]),
                            prompt_version=(
                                v2.GROUNDED_RELATION_PROPOSER_PROMPT_VERSION
                            ),
                            schema_version=(
                                v2.GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION
                            ),
                            parser_version=(
                                v2.GROUNDED_RELATION_PROPOSER_PARSER_VERSION
                            ),
                            request_hash=request_hash,
                        )
                        if callable(cache_lookup) else None
                    )
                    if cached_call is not None:
                        cached_parsed = cached_call.get(
                            "parsed_response"
                        ) or {}
                        raw_proposal = (
                            cached_parsed.get("raw")
                            if isinstance(cached_parsed, Mapping)
                            and "raw" in cached_parsed
                            else cached_parsed
                        )
                        raw_response_text = str(
                            cached_call.get("raw_response_text")
                            or _safe_json_text(raw_proposal)
                        )
                        token_usage = {
                            "cache_hit": True,
                            "reused_model_call_id": str(
                                cached_call.get("id") or ""
                            ),
                        }
                        cache_hits += 1
                    else:
                        calls += 1
                        raw_proposal = self._propose(
                            model=str(context["model"]),
                            context_budget_tokens=context_budget_tokens,
                            max_output_tokens=output_budget,
                            context_id=scope_context_id,
                            content=content,
                            mentions=mentions,
                            schema=dict(context["schema"]),
                            pass_no=pass_no,
                            already_found_keys=sorted(already_found_keys),
                            prompt=prompt,
                            response_schema=response_schema,
                            required_evidence_span=(scope_start, scope_end),
                        )
                        raw_response_text = str(
                            getattr(self.models, "last_raw_response_text", "")
                            or _safe_json_text(raw_proposal)
                        )
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        if diagnostics is not None:
                            token_usage = asdict(diagnostics)
                    legal_batch = v2.parse_proposer_response(
                        context_id=scope_context_id,
                        content=content,
                        mentions=mentions,
                        logical_schema=dict(context["schema"]),
                        raw=raw_proposal,
                        candidate_source=v2.DIRECT_PROPOSER,
                        pass_no=pass_no,
                        proposer_model_revision=str(
                            context["model_revision"]
                        ),
                        required_evidence_span=(scope_start, scope_end),
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
                                    {"start": span.start, "end": span.end}
                                    for span in item.evidence_spans
                                ],
                                "qualifiers": list(item.qualifiers),
                                "confidence": item.confidence,
                            }
                            for item in legal_batch.proposals
                        ],
                        "missing_endpoint_mentions": [
                            dict(item)
                            for item in legal_batch.missing_endpoint_mentions
                        ],
                        "coverage": dict(legal_batch.coverage),
                    }
                    self.db.record_grounded_relation_proposal_attempt(
                        run_id,
                        {
                            "context_identity": context_key,
                            "attempt_no": audit_attempt_no,
                            "status": "ACCEPTED",
                            "request_hash": request_hash,
                            "response_hash": relation_model_payload_hash(
                                "response", raw_response_text,
                            ),
                            "proposer_prompt_version":
                                v2.GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
                            "proposer_schema_version":
                                v2.GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION,
                            "proposer_parser_version":
                                v2.GROUNDED_RELATION_PROPOSER_PARSER_VERSION,
                            "proposer_model_revision":
                                context["model_revision"],
                            "model_name": context["model"],
                            "request_payload": request_payload,
                            "raw_response_text": raw_response_text,
                            "parsed_response": parsed_response,
                            "token_usage": token_usage,
                            "metadata": {
                                "chunk_id": int(chunk["id"]),
                                "scope_start": scope_start,
                                "scope_end": scope_end,
                                "pass_no": pass_no,
                                "retry_no": retry_no,
                                "proposal_count": len(
                                    legal_batch.proposals
                                ),
                                "missing_endpoint_mentions": len(
                                    legal_batch.missing_endpoint_mentions
                                ),
                            },
                        },
                    )
                    last_error = None
                    break
                except Exception as exc:
                    failures += 1
                    last_error = exc
                    raw_response_text = str(
                        getattr(self.models, "last_raw_response_text", "")
                        or raw_response_text
                    )
                    diagnostics = getattr(
                        self.models, "last_call_diagnostics", None,
                    )
                    if diagnostics is not None:
                        token_usage = asdict(diagnostics)
                    self.db.record_grounded_relation_proposal_attempt(
                        run_id,
                        {
                            "context_identity": context_key,
                            "attempt_no": audit_attempt_no,
                            "status": (
                                "TERMINAL_FAILED"
                                if retry_no == int(
                                    Config.relation_max_attempts
                                )
                                else "RETRYABLE_FAILED"
                            ),
                            "request_hash": request_hash,
                            "response_hash": (
                                relation_model_payload_hash(
                                    "response", raw_response_text,
                                ) if raw_response_text else ""
                            ),
                            "proposer_prompt_version":
                                v2.GROUNDED_RELATION_PROPOSER_PROMPT_VERSION,
                            "proposer_schema_version":
                                v2.GROUNDED_RELATION_PROPOSER_SCHEMA_VERSION,
                            "proposer_parser_version":
                                v2.GROUNDED_RELATION_PROPOSER_PARSER_VERSION,
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
                                "scope_start": scope_start,
                                "scope_end": scope_end,
                                "pass_no": pass_no,
                                "retry_no": retry_no,
                            },
                        },
                    )
            if legal_batch is None:
                return {
                    "batches": batches,
                    "calls": calls,
                    "cache_hits": cache_hits,
                    "failures": failures,
                    "complete": False,
                    "error_class": (
                        type(last_error).__name__
                        if last_error is not None else "unknown"
                    ),
                    "found_keys": already_found_keys,
                }

            batches.append(legal_batch)
            new_found_keys = {
                _proposal_found_key(item)
                for item in legal_batch.proposals
                if _proposal_found_key(item) not in already_found_keys
            }
            already_found_keys.update(new_found_keys)
            parser_rejections = list(
                legal_batch.coverage.get("_parser_rejected_proposals") or []
            )
            uncovered = list(
                legal_batch.coverage.get("uncovered_cues") or []
            )
            terminal_complete = bool(
                legal_batch.coverage.get("complete")
                and not parser_rejections
                and not uncovered
            )
            hit_output_cap = (
                max_proposals > 0
                and len(legal_batch.proposals) >= max_proposals
            )
            if terminal_complete and not hit_output_cap:
                break
            if pass_no > 0 and not new_found_keys:
                break

        return {
            "batches": batches,
            "calls": calls,
            "cache_hits": cache_hits,
            "failures": failures,
            "complete": terminal_complete,
            "error_class": "",
            "found_keys": already_found_keys,
        }

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

    def _classifier_request_fits(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        max_output_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> bool:
        """Preflight a classifier batch with the gateway's own estimator.

        Oversized batches are a planning condition, not a model failure.  The
        exact same system message, prompt and JSON response schema used by the
        HTTP request are measured here.  Gateways without the production
        capability API retain the existing call path for test-double
        compatibility; the real Ollama gateway always exposes both methods.
        """

        estimator = getattr(self.models, "token_estimator", None)
        estimate = getattr(estimator, "estimate", None)
        effective_context = getattr(self.models, "effective_model_context", None)
        if not callable(estimate) or not callable(effective_context):
            return True
        runtime = effective_context(model)
        runtime_tokens = int(
            getattr(runtime, "effective_context_tokens", 0) or 0
        )
        if runtime_tokens <= 0:
            return True
        rendered = json.dumps({
            "messages": [
                {
                    "role": "system",
                    "content": GROUNDED_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            "format": response_schema,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        estimated_input = int(estimate(rendered))
        verified_context = min(
            int(context_budget_tokens), runtime_tokens,
        )
        return (
            estimated_input
            + int(max_output_tokens)
            + int(Config.relation_classifier_safety_margin)
            <= verified_context
        )

    @staticmethod
    def _verifier_output_budget(
        context_budget_tokens: int, attempt_no: int = 1,
    ) -> int:
        """Reserve enough output for schema-constrained verifier calls.

        Some local models can spend more than 1K tokens completing a small
        constrained object even with thinking disabled.  Repeating the same
        truncated request three times cannot recover it, so every verifier
        sub-call receives a bounded 4K-or-larger reserve fitted to the verified
        runner context.  JSON Schema limits still cap free-form fields.
        """

        base = max(
            4096, int(Config.relation_classifier_output_reserve) * 2,
        )
        desired = min(16384, base * (2 ** max(0, int(attempt_no) - 1)))
        # Leave at least half the verified context for the rendered system,
        # request, JSON Schema and safety margin. The previous retry growth
        # could reserve 6656 output tokens in an 8192-token context before
        # accounting for a ~2700-token request, making every retry fail before
        # HTTP. Verifier responses are compact, so a half-context cap is both
        # sufficient and protocol-safe.
        context_cap = max(512, int(context_budget_tokens) // 2)
        return max(512, min(desired, context_cap, 16384))

    def _verify(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        attempt_no: int = 1,
    ) -> Any:
        if attempt_no > 1:
            retry_instruction = (
                "The prior independent verdict disagreed with another audit or did not "
                "complete. Re-evaluate the frozen ordered claim independently. Return "
                "only the compact schema fields, short enum values, and minimum exact "
                "evidence spans; do not add narrative text."
            )
            try:
                retry_payload = json.loads(prompt)
            except (TypeError, ValueError, json.JSONDecodeError):
                retry_payload = None
            if isinstance(retry_payload, dict):
                # Keep the model request valid JSON on retries.  Appending prose
                # after a serialized object made the structured protocol
                # malformed and caused deterministic retry failures.
                retry_payload["protocol_retry"] = {
                    "attempt": int(attempt_no),
                    "instruction": retry_instruction,
                }
                prompt = json.dumps(retry_payload, ensure_ascii=False)
            else:
                prompt += f"\n\nPROTOCOL RETRY: {retry_instruction}"
        try:
            prompt_payload = json.loads(prompt)
        except (TypeError, ValueError, json.JSONDecodeError):
            prompt_payload = None
        structural_adjudication = bool(
            isinstance(prompt_payload, Mapping)
            and prompt_payload.get("adjudication_mode")
            == "STRUCTURAL_SOURCE_ASSERTION"
        )
        uses_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", prompt))
        if structural_adjudication:
            system_prompt = (
                "你是结构证据独立裁决器。冻结源解析器已经确认精确的子项→父项绑定；"
                "只检查当前主体是否正好是子项、客体是否正好是父项，以及 Schema 和端点类型是否允许。"
                "结构断言不要求标签与子项之间出现词汇谓词，禁止把普通相邻或句中实体当作结构绑定。"
                "严格按给定 JSON Schema 返回。"
                if uses_cjk else
                "You are the independent structural-source adjudicator. The frozen source mapper "
                "has already confirmed an exact child-to-parent binding. Check only whether the "
                "current subject is exactly the child, the object exactly the parent, and Schema "
                "semantics/types allow it. A structural assertion does not require a lexical "
                "predicate between label and child; never treat ordinary adjacency or an embedded "
                "entity as the binding. Return strictly according to the supplied JSON Schema."
            )
        else:
            system_prompt = (
                "你是独立的有向关系证据核验器。只依据冻结原文核验完整的有向三元组，"
                "并严格按给定 JSON Schema 返回。"
                if uses_cjk else
                "You are the independent Shadow relation v2 evidence verifier. "
                "Judge the complete ordered triple only from the frozen context. "
                "Return exactly the supplied JSON Schema with the seven fields "
                "verdict, reason_code, forward_entailment, reverse_entailment, "
                "subject_role_supported, object_role_supported, and "
                "supported_evidence_spans. Never replace them with entailment, "
                "evidence_spans, verification, or other aliases."
            )
        raw = self.models.json_chat(
            model=model,
            system=system_prompt,
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._verifier_output_budget(
                context_budget_tokens, attempt_no,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        return raw

    def _run_source_structure_binding_audit(
        self,
        *,
        run_id: str,
        hypothesis: v2.RelationHypothesis,
        opportunity: v2.PairOpportunity,
        content: str,
        logical_schema: Mapping[str, Any],
        model: str,
        model_revision: str,
        context_budget_tokens: int,
    ) -> tuple[v2.VerificationAttempt | None, str, bool]:
        """Run the narrow final audit for a parser-proved structure binding.

        The pure prompt builder revalidates the complete parser/Schema
        authorization contract before any model call.  An invalid claimed
        structure falls back to the ordinary verifier; a transport or parser
        failure after authorization becomes a terminal UNCERTAIN result so an
        unreviewed structural edge can never be published.
        """

        request: dict[str, Any] = {
            "stage": "source_structure_binding_audit",
            "model": str(model),
            "model_revision": str(model_revision),
            "temperature": 0.0,
            "context_budget_tokens": int(context_budget_tokens),
            "max_output_tokens": 256,
        }
        raw: Any = None
        raw_text = ""
        parsed: dict[str, Any] = {}
        token_usage: dict[str, Any] = {}
        error_type = ""
        error_message = ""
        result = "UNAVAILABLE"
        contract_authorized = False
        model_called = False
        try:
            prompt = v2.source_structure_binding_audit_prompt(
                hypothesis,
                opportunity,
                content=content,
                logical_schema=logical_schema,
            )
            schema = v2.source_structure_binding_audit_response_schema()
            request.update({
                "prompt": prompt,
                "schema": schema,
                "system": (
                    "Audit exact parser-proved source structure. Return "
                    "only JSON."
                ),
            })
            contract_authorized = True
            model_called = True
            raw = self.models.json_chat(
                model=model,
                system=request["system"],
                prompt=prompt,
                schema=schema,
                temperature=0.0,
                context_budget_tokens=context_budget_tokens,
                max_output_tokens=256,
                safety_margin_tokens=(
                    Config.relation_classifier_safety_margin
                ),
            )
            result = v2.parse_source_structure_binding_audit(raw)
            parsed = {"structure_status": result}
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = f"{error_type}: {exc}"[:2000]
            if not contract_authorized:
                result = "INVALID_CONTRACT"
        finally:
            if model_called:
                raw_text = str(
                    getattr(self.models, "last_raw_response_text", "")
                    or (
                        _safe_json_text(raw)
                        if raw is not None else ""
                    )
                )
                diagnostics = getattr(
                    self.models, "last_call_diagnostics", None,
                )
                token_usage = (
                    asdict(diagnostics)
                    if diagnostics is not None else {}
                )

        attempt: v2.VerificationAttempt | None = None
        if contract_authorized:
            try:
                attempt = v2.source_structure_binding_audit_terminal_attempt(
                    hypothesis,
                    opportunity,
                    content=content,
                    logical_schema=logical_schema,
                    result=result,
                    verifier_model_revision=model_revision,
                    request_payload=request,
                    raw_response_text=raw_text,
                    parsed_response=parsed,
                    token_usage=token_usage,
                    error_message=error_message,
                )
            except Exception as exc:
                attempt = None
                result = "INVALID_CONTRACT"
                error_type = type(exc).__name__
                error_message = f"{error_type}: {exc}"[:2000]

        request_hash = relation_model_payload_hash("request", request)
        self.db.record_grounded_relation_source_model_call(
            run_id,
            {
                "component": "source_structure_binding_audit",
                "model_name": str(model),
                "model_digest": str(model_revision),
                "prompt_version": (
                    v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PROMPT_VERSION
                ),
                "schema_version": (
                    v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_SCHEMA_VERSION
                ),
                "parser_version": (
                    v2.GROUNDED_RELATION_SOURCE_STRUCTURE_AUDIT_PARSER_VERSION
                ),
                "idempotency_key": (
                    "source-structure-audit:" + hypothesis.hypothesis_id
                ),
                "attempt_no": 1,
                "status": "COMPLETED" if not error_message else "FAILED",
                "request_hash": request_hash,
                "response_hash": (
                    relation_model_payload_hash("response", raw_text)
                    if raw_text else ""
                ),
                "request_payload": request,
                "raw_response_text": raw_text,
                "parsed_response": parsed,
                "token_usage": token_usage,
                "error_type": error_type,
                "error_message": error_message,
                "terminal_status": (
                    "COMPLETED" if not error_message else "FAILED"
                ),
            },
        )
        return attempt, result, model_called

    def _batch_verifier_model_request(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        claim_count: int,
    ) -> dict[str, Any]:
        """Build the exact transport request persisted for batch replay."""

        desired_output = max(4096, min(16384, int(claim_count) * 768))
        context_cap = max(1024, int(context_budget_tokens) // 2)
        return {
            "model": model,
            "system": (
                "You are an independent directed-relation evidence verifier. "
                "Evaluate every supplied claim separately from its own frozen "
                "context, Schema policy, endpoints, and evidence. Never infer a "
                "claim from another item in the batch. Return exactly the supplied "
                "JSON Schema. The response root is decisions, never verifications, "
                "verdicts, items, or results. Every decision must use the exact fields "
                "hypothesis_id, verdict, reason_code, forward_entailment, "
                "reverse_entailment, subject_role_supported, "
                "object_role_supported, and supported_evidence_spans; never shorten "
                "them to entailment or evidence_spans. reason_code is one concise "
                "diagnostic label of at most 160 characters, never analysis."
            ),
            "prompt": prompt,
            "schema": response_schema,
            "temperature": 0.0,
            "context_budget_tokens": context_budget_tokens,
            "max_output_tokens": min(desired_output, context_cap),
            "safety_margin_tokens": Config.relation_classifier_safety_margin,
        }

    def _verify_relation_batch(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        claim_count: int,
    ) -> Any:
        """Adjudicate independent relation claims in one transport call."""

        return self.models.json_chat(**self._batch_verifier_model_request(
            model=model,
            context_budget_tokens=context_budget_tokens,
            prompt=prompt,
            response_schema=response_schema,
            claim_count=claim_count,
        ))

    def _probe_verifier_roles(
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
                "You are a predicate-blind endpoint role analyst. Use only the "
                "supplied frozen source context and exact endpoints."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._verifier_output_budget(
                context_budget_tokens,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    def _inventory_atomic_facts(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        attempt_no: int = 1,
    ) -> Any:
        """Extract one predicate-blind direct-fact inventory per context."""

        if attempt_no > 1:
            payload = json.loads(prompt)
            payload["protocol_retry"] = (
                "Re-read the complete source and return only coverage_complete "
                "and the bounded frames array. Copy every argument surface exactly "
                "and keep roles, argument_kinds, and entity_heads position-aligned "
                "with arguments. ENTITY heads must be minimal source-copied core "
                "referents; every non-ENTITY head must be an empty string. Preserve "
                "each frame's assertion_status and never convert negation, modality, "
                "or a condition into ASSERTED."
            )
            prompt = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        uses_cjk = bool(re.search(
            r"[\u3400-\u4dbf\u4e00-\u9fff]", prompt,
        ))
        return self.models.json_chat(
            model=model,
            system=(
                "你是无预设关系类别的原子语义框架分析器。逐一覆盖原文每个显式谓词，只依据"
                "原文拆分完整命题；实体论元保留完整短语，并另给出最小核心指称，事件和状态"
                "保持完整。使役或结果结构必须保留外部完整原因与完整结果从句，不能把结果从句"
                "内部主语提升为使役者。每个并列参与者必须是独立 argument，禁止把并列列表"
                "留在一个 argument 内。严格按给定 "
                "JSON Schema 返回，不得推断或连接同角色并列项。"
                if uses_cjk else
                "You are a predicate-blind atomic proposition analyst. Split only "
                "direct source facts, cover every explicit source predicate, preserve "
                "complete arguments and expose minimal entity core referents. In "
                "causative or result structures preserve the complete external cause "
                "and complete result clause; never promote the result clause's inner "
                "subject to causative agent. Every coordinated participant must be a "
                "separate argument; never leave a coordinated list inside one argument. "
                "Return the supplied JSON Schema without inference or peer-list "
                "cross-links."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._verifier_output_budget(
                context_budget_tokens, attempt_no,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    def _extract_grounded_open_facts(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        attempt_no: int = 1,
    ) -> Any:
        """Extract direct open facts whose endpoints are frozen mention IDs."""

        if attempt_no > 1:
            payload = json.loads(prompt)
            payload["protocol_retry"] = (
                "Return only one facts array. Every row must contain exactly "
                "subject_id, predicate, object_id and evidence. IDs must come "
                "from MENTIONS and evidence must be copied contiguously from "
                "SOURCE. Keep endpoints minimal by selecting IDs rather than "
                "copying or rewriting their surfaces."
            )
            prompt = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return self.models.json_chat(
            model=model,
            system=v2.GROUNDED_RELATION_OPEN_FACT_SYSTEM_PROMPT,
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._proposer_output_budget(
                context_budget_tokens=context_budget_tokens,
                attempt_no=attempt_no,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    def _align_atomic_frame_schema(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        attempt_no: int = 1,
    ) -> Any:
        """Map source-derived frame roles to versioned Schema roles."""

        if attempt_no > 1:
            payload = json.loads(prompt)
            payload["protocol_retry"] = (
                "Return exactly coverage_complete and alignments. Candidate endpoints "
                "are absent by design. Enumerate every Schema-valid ordered argument "
                "binding for every equivalent frame; relatedness alone is not "
                "equivalence and source order is not role order."
            )
            prompt = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return self.models.json_chat(
            model=model,
            system=(
                "You are a strict semantic-frame to versioned-Schema role mapper. "
                "Candidate endpoints are withheld. Judge underlying relation "
                "equivalence first, then exhaustively enumerate Schema-valid ordered "
                "argument bindings. Relatedness is not equivalence and source order is "
                "not role order. Return only the supplied JSON Schema."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._verifier_output_budget(
                context_budget_tokens, attempt_no,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    def _bind_source_frames_schema(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        attempt_no: int = 1,
    ) -> Any:
        """Map one complete source inventory to active Schema bindings."""

        if attempt_no > 1:
            payload = json.loads(prompt)
            payload["protocol_retry"] = (
                "Return only coverage_complete and bindings. Recheck every frame "
                "and emit only exact argument indexes and active predicate names."
            )
            prompt = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return self.models.json_chat(
            model=model,
            system=(
                "You are an independent source-frame to Schema binding judge. "
                "Use only the supplied source frames and active Schema; never "
                "invent endpoints or relations. Return only the supplied JSON Schema."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._verifier_output_budget(
                context_budget_tokens, attempt_no,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    def _resolve_relation_references(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        attempt_no: int = 1,
    ) -> Any:
        """Resolve bounded local references independently of fact extraction."""

        if attempt_no > 1:
            payload = json.loads(prompt)
            payload["protocol_retry"] = (
                "Return exactly coverage_complete and one resolution for every "
                "supplied reference_id. Select only an allowed antecedent_id or "
                "UNRESOLVED; never invent, omit, or duplicate an item."
            )
            prompt = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return self.models.json_chat(
            model=model,
            system=(
                "You are a conservative, predicate-blind coreference adjudicator. "
                "Track discourse and grammatical roles in the frozen source and "
                "return exactly the supplied JSON Schema."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=min(
                4096,
                self._verifier_output_budget(
                    context_budget_tokens, attempt_no,
                ),
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    def _adjudicate_source_predicate_conflict(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> Any:
        """Contrast Schema-declared alternatives for one immutable source frame."""

        return self.models.json_chat(
            model=model,
            system=(
                "You are an independent contrastive relation-policy judge. "
                "Choose at most one Schema-declared conflicting relation for "
                "the frozen ordered source frame and return strict JSON."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=min(
                2048, self._verifier_output_budget(context_budget_tokens),
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )

    def _align_verifier_roles(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        response_schema: dict[str, Any],
        attempt_no: int = 1,
    ) -> Any:
        alignment = self.models.json_chat(
            model=model,
            system=(
                "You are an independent role alignment judge and semantic role "
                "matching judge. Compare only the supplied contextual and required "
                "endpoint roles. The request does not reveal a predicate identifier."
            ),
            prompt=prompt,
            schema=response_schema,
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._verifier_output_budget(
                context_budget_tokens,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        contrast = self.models.json_chat(
            model=model,
            system=(
                "You are a strict semantic contrast judge. Decide semantic_contrast "
                "before considering endpoint-role similarity."
            ),
            prompt=v2.verifier_semantic_contrast_prompt(prompt),
            schema=v2.verifier_semantic_contrast_response_schema(),
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=self._verifier_output_budget(
                context_budget_tokens,
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        merged = dict(alignment) if isinstance(alignment, Mapping) else {}
        merged["semantic_contrast"] = v2.parse_verifier_semantic_contrast(
            contrast
        )
        merged.update(self._verify_exact_argument_alignment(
            model=model,
            context_budget_tokens=context_budget_tokens,
            prompt=prompt,
            attempt_no=attempt_no,
        ))
        return merged

    def _verify_exact_argument_alignment(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        attempt_no: int = 1,
    ) -> dict[str, Any]:
        """Judge each ordered direction independently and combine the results.

        Two one-boolean NLI calls avoid the position/enum bias observed when a
        small local model is asked to choose among FORWARD/REVERSE/NEITHER in
        one response.  Neither call can publish a fact; the final independent
        verifier still has to return SUPPORTED.
        """

        alignment_contract = json.loads(prompt)
        symmetric = bool(alignment_contract.get("symmetric"))
        source_text = str(
            (alignment_contract.get("source_context") or {}).get("text") or ""
        )
        uses_cjk = bool(re.search(
            r"[\u3400-\u4dbf\u4e00-\u9fff]", source_text,
        ))
        entailments: dict[str, bool] = {}
        protocol_errors: dict[str, str] = {}
        for direction in ("FORWARD", "REVERSE"):
            exact_prompt = v2.verifier_exact_argument_prompt(
                prompt, direction=direction,
            )
            if attempt_no > 1:
                retry_payload = json.loads(exact_prompt)
                retry_payload["protocol_retry"] = (
                    "Return only one direction_supported boolean and no explanation."
                )
                exact_prompt = json.dumps(
                    retry_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            try:
                raw = self.models.json_chat(
                    model=model,
                    system=(
                        "你是严格的语义角色映射审计器。逐项比较上下文角色与所需角色，"
                        "字段位置不是证据。"
                        if uses_cjk else
                        "You are a strict semantic-role mapping auditor. Compare "
                        "contextual and required roles; field position is not evidence."
                    ),
                    prompt=exact_prompt,
                    schema=v2.verifier_exact_argument_response_schema(),
                    temperature=0.0,
                    context_budget_tokens=context_budget_tokens,
                    max_output_tokens=min(
                        2048 * (2 ** max(0, int(attempt_no) - 1)),
                        self._verifier_output_budget(
                            context_budget_tokens, attempt_no,
                        ),
                    ),
                    safety_margin_tokens=Config.relation_classifier_safety_margin,
                )
                entailments[direction] = (
                    v2.parse_verifier_directional_entailment(
                        raw, exact_prompt=exact_prompt,
                    )
                )
            except Exception as exc:
                # Each direction is an independent fail-closed signal. A
                # malformed opposite-direction response must not erase a
                # fully grounded expected direction and the separately parsed
                # semantic-role audit. It is recorded as false/uncertain for
                # orientation; symmetric publication still requires both.
                entailments[direction] = False
                protocol_errors[direction] = (
                    f"{type(exc).__name__}: {exc}"[:500]
                )

        forward = entailments["FORWARD"]
        reverse = entailments["REVERSE"]
        orientation = (
            "SYMMETRIC" if forward and reverse and symmetric
            else "UNCERTAIN" if forward and reverse
            else "FORWARD" if forward
            else "REVERSE" if reverse
            else "NEITHER"
        )
        return {
            "exact_argument_orientation": orientation,
            "exact_argument_forward_supported": bool(forward),
            "exact_argument_reverse_supported": bool(reverse),
            "exact_argument_protocol_errors": protocol_errors,
        }

    def _verify_predicate_blind_endpoint_probe(
        self,
        *,
        model: str,
        context_budget_tokens: int,
        prompt: str,
        attempt_no: int = 1,
    ) -> dict[str, Any]:
        """Extract exact endpoint roles before exposing predicate metadata."""

        if attempt_no > 1:
            payload = json.loads(prompt)
            payload["protocol_retry"] = (
                "Return only the ten required schema fields and no explanation."
            )
            prompt = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        raw = self.models.json_chat(
            model=model,
            system=(
                "你是看不到目标谓词的精确端点与命题角色分析器。只依据原文，不得猜测"
                "关系分类，并严格按给定 JSON Schema 返回。"
                if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", prompt) else
                "You are a predicate-blind exact endpoint role analyst and proposition analyst. "
                "Do not guess the target relation taxonomy."
            ),
            prompt=prompt,
            schema=v2.verifier_role_probe_response_schema(),
            temperature=0.0,
            context_budget_tokens=context_budget_tokens,
            max_output_tokens=min(
                2048 * (2 ** max(0, int(attempt_no) - 1)),
                self._verifier_output_budget(context_budget_tokens, attempt_no),
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        # Some schema-capable local models interpret a field named
        # ``*_complete_argument`` as a request to repeat the argument text,
        # even though its JSON Schema type is boolean.  Recover only the
        # structural meaning that is independently checkable from immutable
        # offsets: an exact returned argument span is complete; a valid strict
        # containing span is incomplete.  No language, domain, predicate name,
        # or lexical cue participates in this protocol normalization.
        normalized = dict(raw) if isinstance(raw, Mapping) else raw
        try:
            request = json.loads(prompt)
        except (json.JSONDecodeError, TypeError):
            request = {}
        if isinstance(normalized, dict) and isinstance(request, Mapping):
            for label in ("endpoint_a", "endpoint_b"):
                field = f"{label}_complete_argument"
                if isinstance(normalized.get(field), bool):
                    continue
                endpoint = request.get(label)
                if not isinstance(endpoint, Mapping):
                    continue
                span = endpoint.get("span") or ()
                try:
                    expected = (int(span[0]), int(span[1]))
                    returned = (
                        int(normalized[f"{label}_argument_start"]),
                        int(normalized[f"{label}_argument_end"]),
                    )
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                normalized[field] = returned == expected
            if (
                normalized.get("direct_relation_present") is False
                and not str(normalized.get("expressed_relation") or "").strip()
            ):
                normalized["expressed_relation"] = "NO_DIRECT_RELATION"
        parsed = v2.parse_verifier_role_probe(normalized)
        required_audit = (
            parsed.get("endpoint_a_complete_argument"),
            parsed.get("endpoint_b_complete_argument"),
            parsed.get("direct_relation_present"),
        )
        if not all(isinstance(value, bool) for value in required_audit):
            raise v2.RelationClassificationError(
                "predicate-blind endpoint probe omitted required structural decisions"
            )
        argument_offsets = (
            parsed.get("endpoint_a_argument_start"),
            parsed.get("endpoint_a_argument_end"),
            parsed.get("endpoint_b_argument_start"),
            parsed.get("endpoint_b_argument_end"),
        )
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in argument_offsets
        ):
            raise v2.RelationClassificationError(
                "predicate-blind endpoint probe omitted complete argument spans"
            )
        return normalized

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

    @staticmethod
    def _occurrence_span(
        opportunity: v2.PairOpportunity,
        occurrence_key: str,
    ) -> tuple[int, int]:
        if occurrence_key == opportunity.left_occurrence_key:
            return tuple(map(int, opportunity.left_span))
        if occurrence_key == opportunity.right_occurrence_key:
            return tuple(map(int, opportunity.right_span))
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
        eligible_assignments: Mapping[str, Mapping[str, Any]] | None = None,
        logical_schema: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        schema_specs = (
            v2.normalize_predicate_schema(logical_schema)
            if isinstance(logical_schema, Mapping) and logical_schema
            else {}
        )
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
            all_evidence_values = sorted(
                evidence_by_fact.get(fact.fact_id) or [],
                key=lambda item: item.evidence_key,
            )
            evidence_values = [
                item for item in all_evidence_values
                if item.verifier_supported and item.evidence_spans
            ]
            if not evidence_values:
                raise RuntimeError(
                    "v2 可物化事实缺少 SUPPORTED verification evidence"
                )
            dropped_evidence_count = (
                len(all_evidence_values) - len(evidence_values)
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
                source_orientation_matches_fact = True
                source_mention, target_mention = (
                    self._occurrence_mention_key(opportunity, subject_occurrence),
                    self._occurrence_mention_key(opportunity, object_occurrence),
                )
            elif (
                object_canonical == str(fact.subject_canonical_id)
                and subject_canonical == str(fact.object_canonical_id)
            ):
                source_orientation_matches_fact = False
                source_mention, target_mention = (
                    self._occurrence_mention_key(opportunity, object_occurrence),
                    self._occurrence_mention_key(opportunity, subject_occurrence),
                )
            else:
                raise RuntimeError(
                    "v2 normalized fact 与 evidence occurrence assignment 不一致"
                )
            source_assignment: Mapping[str, Any] = {}
            target_assignment: Mapping[str, Any] = {}
            if eligible_assignments is not None:
                source_assignment = eligible_assignments.get(source_mention)
                target_assignment = eligible_assignments.get(target_mention)
                if not source_assignment or not target_assignment:
                    raise RuntimeError(
                        "v2 publication endpoint is not entity-publishability eligible"
                    )
                if (
                    str(source_assignment.get("canonical_key") or "")
                    != str(fact.subject_canonical_id)
                    or str(target_assignment.get("canonical_key") or "")
                    != str(fact.object_canonical_id)
                ):
                    raise RuntimeError(
                        "v2 publication endpoint assignment changed after verification"
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
            if (
                evidence_start < 0
                or evidence_end <= evidence_start
                or evidence_end > len(content)
                or not content[evidence_start:evidence_end].strip()
            ):
                raise RuntimeError(
                    "v2 publication evidence is outside the frozen source chunk"
                )
            source_relation_expressions: list[str] = []
            for evidence_value in evidence_values:
                evidence_hypothesis = hypotheses.get(
                    str(evidence_value.hypothesis_id)
                )
                if evidence_hypothesis is None:
                    raise RuntimeError(
                        "v2 fact evidence 缺少附加 hypothesis"
                    )
                candidate_source = str(
                    (evidence_hypothesis.predicate_metadata or {}).get(
                        "candidate_source"
                    ) or ""
                ).upper()
                expression = v2.source_relation_display_expression(
                    (evidence_hypothesis.predicate_metadata or {}).get(
                        "source_frame_relation"
                    )
                )
                if (
                    candidate_source in {
                        "ATOMIC_FRAME_SCHEMA_PROJECTION",
                        "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
                    }
                    and expression
                    and expression not in source_relation_expressions
                ):
                    source_relation_expressions.append(expression)
            selected_source_expression = v2.source_relation_display_expression(
                (hypothesis.predicate_metadata or {}).get(
                    "source_frame_relation"
                )
            )
            selected_candidate_source = str(
                (hypothesis.predicate_metadata or {}).get(
                    "candidate_source"
                ) or ""
            ).upper()
            selected_is_source_projection = selected_candidate_source in {
                "ATOMIC_FRAME_SCHEMA_PROJECTION",
                "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
            }
            open_relation_policy = (
                (hypothesis.predicate_metadata or {}).get(
                    "open_relation_policy"
                ) or {}
            )
            open_relation_carrier = bool(
                isinstance(open_relation_policy, Mapping)
                and open_relation_policy.get("enabled")
            )
            source_frame_order_preserved = bool(
                (hypothesis.predicate_metadata or {}).get(
                    "source_subject_argument_index"
                ) == 0
                and (hypothesis.predicate_metadata or {}).get(
                    "source_object_argument_index"
                ) == 1
            )
            display_source_expression = bool(
                open_relation_carrier
                and
                selected_is_source_projection
                and selected_source_expression
                and source_orientation_matches_fact
                and source_frame_order_preserved
            )
            chunk_metadata = (
                chunk.get("metadata")
                if isinstance(chunk.get("metadata"), Mapping) else {}
            )
            source_chunk_block_types = sorted({
                str(value or "").strip().lower()
                for value in chunk_metadata.get("block_types") or ()
                if str(value or "").strip()
            })
            schema_spec = schema_specs.get(str(fact.predicate))
            semantic_metadata = {
                **(hypothesis.predicate_metadata or {}),
                "source_relation_projection": selected_is_source_projection,
                "relation_candidate_origin": (
                    "source_projection"
                    if selected_is_source_projection
                    else "direct_proposer"
                    if (
                        selected_candidate_source == v2.DIRECT_PROPOSER
                        or hypothesis.classifier_prompt_version
                        == v2.GROUNDED_RELATION_PROPOSER_PROMPT_VERSION
                    )
                    else "pair_classifier"
                ),
                "source_entity_type": str(
                    (source_assignment or {}).get("entity_type") or ""
                ),
                "target_entity_type": str(
                    (target_assignment or {}).get("entity_type") or ""
                ),
                "schema_domain_types": (
                    list(schema_spec.domain_types) if schema_spec else []
                ),
                "schema_range_types": (
                    list(schema_spec.range_types) if schema_spec else []
                ),
                "source_chunk_block_types": source_chunk_block_types,
                "source_media_derived": bool(
                    {"image", "figure"}.intersection(
                        source_chunk_block_types
                    )
                ),
            }
            semantic_projection = relation_semantic_projection(
                relation_type=str(fact.predicate),
                source_expression=selected_source_expression,
                open_relation_carrier=open_relation_carrier,
                predicate_metadata=semantic_metadata,
            )
            relation_candidate_origin = str(
                semantic_metadata.get("relation_candidate_origin") or ""
            )
            schema_lexical_cues = (
                tuple(schema_spec.positive_cues)
                + tuple(schema_spec.optional_soft_cues)
                if schema_spec is not None and not open_relation_carrier
                else ()
            )
            # A missing match is meaningful only when the active, versioned
            # Schema actually supplies lexical cues.  Schemas without cues
            # must continue through the semantic role audit instead of being
            # rejected merely because no deterministic lexical contract was
            # available to evaluate.
            schema_lexical_anchor_applicable = bool(schema_lexical_cues)
            schema_lexical_anchor = bool(
                schema_lexical_anchor_applicable
                and selected_source_expression
                and source_expression_matches_schema_cues(
                    selected_source_expression,
                    schema_lexical_cues,
                )
            )
            fact_source_occurrence = (
                subject_occurrence
                if source_orientation_matches_fact else object_occurrence
            )
            fact_target_occurrence = (
                object_occurrence
                if source_orientation_matches_fact else subject_occurrence
            )
            fact_source_span = self._occurrence_span(
                opportunity, fact_source_occurrence,
            )
            fact_target_span = self._occurrence_span(
                opportunity, fact_target_occurrence,
            )
            source_frame_other_participants = [
                str(value).strip()
                for value in (
                    (hypothesis.predicate_metadata or {}).get(
                        "source_frame_other_participants"
                    ) or ()
                )
                if str(value).strip()
            ]

            def containing_source_mentions(
                endpoint_span: tuple[int, int],
            ) -> list[str]:
                """Return exact frozen mentions strictly containing an endpoint."""

                endpoint_start, endpoint_end = endpoint_span
                containing: list[str] = []
                for surface in source_frame_other_participants:
                    cursor = evidence_start
                    while cursor < evidence_end:
                        match_start = content.find(
                            surface, cursor, evidence_end,
                        )
                        if match_start < 0:
                            break
                        match_end = match_start + len(surface)
                        if (
                            match_start <= endpoint_start
                            < endpoint_end <= match_end
                            and (match_start, match_end)
                            != (endpoint_start, endpoint_end)
                        ):
                            containing.append(surface)
                            break
                        cursor = match_start + 1
                    if len(containing) >= 6:
                        break
                return list(dict.fromkeys(containing))

            subject_containing_mentions = containing_source_mentions(
                fact_source_span,
            )
            object_containing_mentions = containing_source_mentions(
                fact_target_span,
            )
            schema_contract = (
                {
                    "definition": schema_spec.semantic_definition[:1200],
                    "subject_role": (
                        schema_spec.subject_role_description[:800]
                    ),
                    "object_role": (
                        schema_spec.object_role_description[:800]
                    ),
                    "direction": schema_spec.direction_description[:800],
                    "symmetric": bool(schema_spec.symmetric),
                }
                if schema_spec is not None and not open_relation_carrier
                else {}
            )
            resolved_reference_endpoints = []
            for item in (
                opportunity.provenance.get(
                    "resolved_reference_endpoints"
                ) or ()
            ):
                if not isinstance(item, Mapping):
                    continue
                value = dict(item)
                reference_span = value.get("reference_span") or ()
                if (
                    isinstance(reference_span, (list, tuple))
                    and len(reference_span) == 2
                    and all(
                        isinstance(position, int)
                        and not isinstance(position, bool)
                        for position in reference_span
                    )
                    and evidence_start <= int(reference_span[0])
                    < int(reference_span[1]) <= evidence_end
                ):
                    reference_start, reference_end = map(
                        int, reference_span,
                    )
                    window_start = max(evidence_start, reference_start - 72)
                    window_end = min(evidence_end, reference_end + 120)
                    value["local_reference_context"] = (
                        content[window_start:reference_start]
                        + "【"
                        + content[reference_start:reference_end]
                        + "】"
                        + content[reference_end:window_end]
                    )
                antecedent_span = value.get("antecedent_span") or ()
                if (
                    isinstance(antecedent_span, (list, tuple))
                    and len(antecedent_span) == 2
                    and all(
                        isinstance(position, int)
                        and not isinstance(position, bool)
                        for position in antecedent_span
                    )
                    and 0 <= int(antecedent_span[0])
                    < int(antecedent_span[1]) <= len(content)
                ):
                    value["antecedent_surface"] = content[
                        int(antecedent_span[0]):int(antecedent_span[1])
                    ]
                resolved_reference_endpoints.append(value)
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
                "policy_version": GROUNDED_RELATION_V2_MATERIALIZATION_VERSION,
                "_publication_audit_candidate": {
                    "candidate_id": str(fact.fact_id),
                    "subject": content[
                        fact_source_span[0]:fact_source_span[1]
                    ],
                    "object": content[
                        fact_target_span[0]:fact_target_span[1]
                    ],
                    "canonical_predicate": str(fact.predicate),
                    "source_expression": selected_source_expression,
                    "evidence": content[evidence_start:evidence_end],
                    "relation_layer": str(
                        semantic_projection.get("layer") or ""
                    ),
                    "schema_contract": schema_contract,
                    "relation_candidate_origin": relation_candidate_origin,
                    "schema_lexical_anchor_applicable": (
                        schema_lexical_anchor_applicable
                    ),
                    "schema_lexical_anchor": schema_lexical_anchor,
                    "source_order_matches_proposed_fact": bool(
                        source_orientation_matches_fact
                        and (
                            source_frame_order_preserved
                            if selected_is_source_projection else True
                        )
                    ),
                    "resolved_reference_grounding": bool(
                        semantic_projection.get(
                            "resolved_reference_grounding"
                        )
                    ),
                    "resolved_reference_endpoints": (
                        resolved_reference_endpoints
                    ),
                    "subject_containing_mentions": (
                        subject_containing_mentions
                    ),
                    "object_containing_mentions": (
                        object_containing_mentions
                    ),
                    "subject_entity_type": str(
                        (source_assignment or {}).get("entity_type") or ""
                    ),
                    "object_entity_type": str(
                        (target_assignment or {}).get("entity_type") or ""
                    ),
                },
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
                    "resolved_reference_endpoints": (
                        resolved_reference_endpoints
                    ),
                    "canonical_relation_type": str(fact.predicate),
                    "normalization_status": (
                        "OPEN_GROUNDED"
                        if open_relation_carrier else "NORMALIZED"
                    ),
                    "open_relation_carrier": open_relation_carrier,
                    "semantic_family_key": str(
                        semantic_projection.get("family_key") or ""
                    ),
                    "relation_semantics": semantic_projection,
                    "publication_integrity": {
                        "policy_version": (
                            RELATION_ENDPOINT_INTEGRITY_POLICY_VERSION
                        ),
                        "entity_publishability_policy_version": (
                            ENTITY_PUBLISHABILITY_POLICY_VERSION
                        ),
                        "valid_evidence_supports": len(evidence_values),
                        "dropped_invalid_evidence_supports": (
                            dropped_evidence_count
                        ),
                        "source_endpoint_eligible": True,
                        "target_endpoint_eligible": True,
                        "source_entity_type": str(
                            (source_assignment or {}).get("entity_type") or ""
                        ),
                        "target_entity_type": str(
                            (target_assignment or {}).get("entity_type") or ""
                        ),
                    },
                    **(
                        {
                            "source_relation_expression": (
                                selected_source_expression
                            ),
                            "source_relation_expressions": (
                                source_relation_expressions
                            ),
                            "source_relation_order_preserved": (
                                source_frame_order_preserved
                                and source_orientation_matches_fact
                            ),
                        }
                        if selected_is_source_projection
                        and selected_source_expression else {}
                    ),
                    **(
                        {
                            "display_relation_label": (
                                str(
                                    semantic_projection.get(
                                        "display_expression"
                                    )
                                    or selected_source_expression
                                )
                            ),
                            "display_relation_source": (
                                "exact_source_expression"
                            ),
                        }
                        if display_source_expression else {}
                    ),
                },
            })
        return rows

    def _run_atomic_publication_audit(
        self,
        *,
        run_id: str,
        rows: list[dict[str, Any]],
        context: Mapping[str, Any],
        context_budget_tokens: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply one narrow, batched semantic-role gate before publication.

        Broad extraction verification remains the recall-oriented authority for
        the immutable fact ledger.  This final gate has no candidate-generation
        power and cannot alter endpoints, direction or predicates; it can only
        withhold a public projection whose exact evidence does not directly
        support the proposed ordered roles.  Parser-owned structural contracts
        have already passed deterministic validation and are not re-litigated
        by a language model.
        """

        working: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        row_by_id: dict[str, dict[str, Any]] = {}
        syntax_ineligible = 0
        structural_contracts = 0
        direct_open_contracts = 0
        schema_unanchored = 0
        integrity_missing = 0
        deterministic_role_conflicts: Counter[str] = Counter()
        for raw_row in rows:
            row = {
                **raw_row,
                "metadata": dict(raw_row.get("metadata") or {}),
            }
            candidate = row.pop("_publication_audit_candidate", None)
            metadata = row["metadata"]
            semantics = (
                dict(metadata.get("relation_semantics") or {})
                if isinstance(metadata.get("relation_semantics"), Mapping)
                else {}
            )
            publication = (
                dict(semantics.get("publication") or {})
                if isinstance(semantics.get("publication"), Mapping)
                else {}
            )
            if publication.get("eligible") is False:
                syntax_ineligible += 1
            elif str(semantics.get("source_structure_kind") or ""):
                structural_contracts += 1
                metadata["atomic_publication_audit"] = {
                    "version": (
                        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                    ),
                    "status": "DETERMINISTIC_STRUCTURE_CONTRACT",
                    "model_called": False,
                }
            elif not isinstance(candidate, Mapping):
                # Missing internal audit material is an integrity failure, not
                # permission to bypass the gate.
                semantics["publication"] = {
                    "eligible": False,
                    "reason": "atomic_publication_audit_unavailable",
                    "policy": (
                        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                    ),
                }
                metadata["atomic_publication_audit"] = {
                    "version": (
                        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                    ),
                    "status": "UNAVAILABLE",
                    "model_called": False,
                }
                integrity_missing += 1
            else:
                candidate_id = str(candidate.get("candidate_id") or "")
                if (
                    not candidate_id
                    or candidate_id != str(row.get("relation_key") or "")
                    or candidate_id in row_by_id
                ):
                    raise RuntimeError(
                        "atomic publication audit candidate identity mismatch"
                    )
                if (
                    str(candidate.get("relation_layer") or "")
                    == "schema_predicate"
                    and str(
                        candidate.get("relation_candidate_origin") or ""
                    ) == "source_projection"
                    and candidate.get(
                        "schema_lexical_anchor_applicable"
                    ) is True
                    and candidate.get("schema_lexical_anchor") is not True
                ):
                    schema_unanchored += 1
                    semantics["publication"] = {
                        "eligible": False,
                        "reason": "schema_projection_without_lexical_anchor",
                        "policy": (
                            v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                        ),
                    }
                    metadata["atomic_publication_audit"] = {
                        "version": (
                            v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                        ),
                        "status": "SCHEMA_LEXICAL_ANCHOR_MISSING",
                        "model_called": False,
                    }
                else:
                    deterministic_reason = (
                        v2.atomic_publication_deterministic_rejection_reason(
                            candidate
                        )
                    )
                    if deterministic_reason:
                        deterministic_role_conflicts[
                            deterministic_reason
                        ] += 1
                        semantics["publication"] = {
                            "eligible": False,
                            "reason": deterministic_reason,
                            "policy": (
                                v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                            ),
                        }
                        metadata["atomic_publication_audit"] = {
                            "version": (
                                v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                            ),
                            "status": "DETERMINISTIC_CONFLICT",
                            "reason": deterministic_reason,
                            "model_called": False,
                        }
                    else:
                        audit_required, route_reason = (
                            v2.atomic_publication_audit_requirement(candidate)
                        )
                        if audit_required:
                            candidates.append(dict(candidate))
                            row_by_id[candidate_id] = row
                        else:
                            direct_open_contracts += 1
                            metadata["atomic_publication_audit"] = {
                                "version": (
                                    v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                                ),
                                "status": "DIRECT_OPEN_SOURCE_CONTRACT",
                                "route_reason": route_reason,
                                "model_called": False,
                            }
            metadata["relation_semantics"] = semantics
            working.append(row)

        def compact_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
            value = dict(candidate)
            evidence = str(value.get("evidence") or "")
            limit = 16_000
            if len(evidence) <= limit:
                return value
            anchors: list[tuple[int, int]] = []
            for field in ("subject", "source_expression", "object"):
                token = str(value.get(field) or "").strip()
                if not token:
                    continue
                index = evidence.find(token)
                if index >= 0:
                    anchors.append((index, index + len(token)))
            if anchors:
                left = min(item[0] for item in anchors)
                right = max(item[1] for item in anchors)
                if right - left < limit - 128:
                    spare = limit - (right - left)
                    start = max(0, left - spare // 2)
                    end = min(len(evidence), start + limit)
                    start = max(0, end - limit)
                    value["evidence"] = evidence[start:end]
                    value["evidence_window"] = [start, end, len(evidence)]
                    return value
            half = (limit - 80) // 2
            value["evidence"] = (
                evidence[:half]
                + "\n[OMITTED_MIDDLE_SOURCE]\n"
                + evidence[-half:]
            )
            value["evidence_window"] = [0, half, len(evidence) - half, len(evidence)]
            return value

        candidates = [compact_evidence(value) for value in candidates]
        statuses: dict[str, str] = {}
        calls = 0
        failed_calls = 0
        split_retries = 0
        max_prompt_chars = min(
            48_000, int(Config.relation_verifier_batch_max_chars),
        )

        batches: list[list[dict[str, Any]]] = []
        current_batch: list[dict[str, Any]] = []
        for candidate in sorted(
            candidates, key=lambda value: str(value["candidate_id"]),
        ):
            # The renderer deliberately enforces the wire contract's hard
            # 16-candidate ceiling.  Flush a full batch before constructing a
            # trial payload; otherwise the boundary check itself attempts to
            # render 17 candidates and aborts a valid large publication run.
            if len(current_batch) >= 16:
                batches.append(current_batch)
                current_batch = []
            trial = [*current_batch, candidate]
            trial_prompt = v2.atomic_publication_audit_prompt(trial)
            if (
                current_batch
                and len(trial_prompt) > max_prompt_chars
            ):
                batches.append(current_batch)
                current_batch = [candidate]
            else:
                current_batch = trial
        if current_batch:
            batches.append(current_batch)

        pending = list(batches)
        while pending:
            batch = pending.pop(0)
            relation_keys = [
                str(value["candidate_id"]) for value in batch
            ]
            wire_ids = [
                f"C{index:04d}" for index in range(1, len(batch) + 1)
            ]
            wire_batch = [
                {**value, "candidate_id": wire_id}
                for value, wire_id in zip(batch, wire_ids)
            ]
            prompt = v2.atomic_publication_audit_prompt(wire_batch)
            schema = v2.atomic_publication_audit_response_schema(
                wire_ids
            )
            request = {
                "stage": "atomic_publication_audit",
                "candidate_ids": wire_ids,
                "relation_keys": relation_keys,
                "model": str(context["model"]),
                "model_revision": str(context["model_revision"]),
                "temperature": 0.0,
                "context_budget_tokens": int(context_budget_tokens),
                "max_output_tokens": 2048,
                "safety_margin_tokens": (
                    Config.relation_classifier_safety_margin
                ),
                "system": (
                    "Audit direct ordered semantic roles in immutable source "
                    "evidence. Return only the exact requested JSON map."
                ),
                "prompt": prompt,
                "schema": schema,
            }
            raw: Any = None
            raw_text = ""
            parsed: dict[str, str] = {}
            token_usage: dict[str, Any] = {}
            error_type = ""
            error_message = ""
            calls += 1
            try:
                raw = self.models.json_chat(
                    model=str(context["model"]),
                    system=str(request["system"]),
                    prompt=prompt,
                    schema=schema,
                    temperature=0.0,
                    context_budget_tokens=int(context_budget_tokens),
                    max_output_tokens=2048,
                    safety_margin_tokens=(
                        Config.relation_classifier_safety_margin
                    ),
                )
                wire_statuses = v2.parse_atomic_publication_audit(
                    raw, candidate_ids=wire_ids,
                )
                parsed = {
                    relation_key: wire_statuses[wire_id]
                    for relation_key, wire_id in zip(
                        relation_keys, wire_ids
                    )
                }
                raw_text = str(
                    getattr(self.models, "last_raw_response_text", "")
                    or _safe_json_text(raw)
                )
            except Exception as exc:
                failed_calls += 1
                error_type = type(exc).__name__
                error_message = f"{error_type}: {exc}"[:2000]
                if raw is not None:
                    raw_text = _safe_json_text(raw)
            finally:
                diagnostics = getattr(
                    self.models, "last_call_diagnostics", None,
                )
                token_usage = (
                    asdict(diagnostics) if diagnostics is not None else {}
                )
            model_call_record = {
                    "component": "atomic_publication_audit",
                    "model_name": str(context["model"]),
                    "model_digest": str(context["model_revision"]),
                    "prompt_version": (
                        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                    ),
                    "schema_version": (
                        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_SCHEMA_VERSION
                    ),
                    "parser_version": (
                        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PARSER_VERSION
                    ),
                    "idempotency_key": (
                        "atomic-publication-audit:"
                        + canonical_digest(relation_keys)
                    ),
                    "attempt_no": 1,
                    "status": "COMPLETED" if not error_message else "FAILED",
                    "request_hash": relation_model_payload_hash(
                        "request", request,
                    ),
                    "response_hash": (
                        relation_model_payload_hash("response", raw_text)
                        if raw_text else ""
                    ),
                    "request_payload": request,
                    "raw_response_text": raw_text,
                    "parsed_response": parsed,
                    "token_usage": token_usage,
                    "error_type": error_type,
                    "error_message": error_message,
                    "terminal_status": (
                        "COMPLETED" if not error_message else "FAILED"
                    ),
                }
            record_model_call = getattr(
                self.db, "record_grounded_relation_source_model_call", None,
            )
            if callable(record_model_call):
                record_model_call(run_id, model_call_record)
            if error_message:
                if len(batch) > 1:
                    midpoint = len(batch) // 2
                    pending[0:0] = [batch[:midpoint], batch[midpoint:]]
                    split_retries += 1
                else:
                    statuses[relation_keys[0]] = "UNAVAILABLE"
            else:
                statuses.update(parsed)

        counts = Counter(statuses.values())
        for candidate_id, row in row_by_id.items():
            metadata = row["metadata"]
            semantics = dict(metadata.get("relation_semantics") or {})
            publication = dict(semantics.get("publication") or {})
            status = statuses.get(candidate_id, "UNAVAILABLE")
            if status != "ALIGNED":
                publication = {
                    "eligible": False,
                    "reason": {
                        "CONFLICT": "atomic_publication_role_conflict",
                        "UNCERTAIN": "atomic_publication_role_uncertain",
                    }.get(status, "atomic_publication_audit_unavailable"),
                    "policy": (
                        v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                    ),
                }
                semantics["publication"] = publication
            metadata["relation_semantics"] = semantics
            metadata["atomic_publication_audit"] = {
                "version": (
                    v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION
                ),
                "status": status,
                "model_called": True,
            }

        return working, {
            "version": v2.GROUNDED_RELATION_PUBLICATION_AUDIT_PROMPT_VERSION,
            "candidate_facts": len(candidates),
            "model_calls": calls,
            "failed_model_calls": failed_calls,
            "split_retries": split_retries,
            "aligned": int(counts.get("ALIGNED", 0)),
            "conflicts": int(counts.get("CONFLICT", 0)),
            "uncertain": int(counts.get("UNCERTAIN", 0)),
            "unavailable": int(counts.get("UNAVAILABLE", 0)),
            "integrity_missing": integrity_missing,
            "deterministic_structure_contracts": structural_contracts,
            "direct_open_source_contracts": direct_open_contracts,
            "deterministic_role_conflicts": sum(
                deterministic_role_conflicts.values()
            ),
            "deterministic_role_conflict_reasons": dict(
                sorted(deterministic_role_conflicts.items())
            ),
            "schema_projections_without_lexical_anchor": schema_unanchored,
            "syntax_ineligible_not_audited": syntax_ineligible,
        }

    def _run_secondary_review(
        self,
        *,
        run_id: str,
        project_id: int,
        context: dict[str, Any],
        fingerprint: v2.RelationMaterializationFingerprint,
        opportunities: dict[str, v2.PairOpportunity],
        hypotheses: list[v2.RelationHypothesis],
        validations: dict[str, v2.HypothesisValidation],
        verification_attempts: list[v2.VerificationAttempt],
        context_budget_tokens: int,
    ) -> dict[str, Any]:
        """Run one bounded child pass over routed terminal uncertainties.

        The returned hypotheses and attempts are appended to the parent run's
        immutable ledger. Existing primary rows are never updated or replaced;
        the latest terminal attempt for a secondary hypothesis is considered by
        the same normalization gate as every other hypothesis.
        """

        empty = {
            "enabled": bool(Config.relation_secondary_enabled),
            "pass_id": "",
            "selected": 0,
            "model_calls": 0,
            "supported": 0,
            "rejected": 0,
            "uncertain": 0,
            "failed": 0,
            "route_counts": {},
            "skipped_by_budget": 0,
            "added_hypotheses": [],
            "added_attempts": [],
        }
        if not Config.relation_secondary_enabled:
            empty["disabled_reason"] = "RELATION_SECONDARY_ENABLED=false"
            return empty
        if not all(callable(getattr(self.db, name, None)) for name in (
            "start_relation_secondary_pass",
            "persist_relation_secondary_candidates",
            "record_relation_secondary_model_call",
            "finish_relation_secondary_pass",
        )):
            # Lightweight replay/unit databases intentionally model only the
            # primary ledger. Production Database exposes the complete
            # secondary audit contract; absence here must not alter those
            # isolated primary-path tests.
            empty["disabled_reason"] = "secondary_ledger_unavailable"
            return empty

        terminal_by_hypothesis: dict[str, v2.VerificationAttempt] = {}
        for attempt in verification_attempts:
            current = terminal_by_hypothesis.get(str(attempt.hypothesis_id))
            if current is None or int(attempt.attempt_no) > int(current.attempt_no):
                terminal_by_hypothesis[str(attempt.hypothesis_id)] = attempt

        def structural_competition_route(
            opportunity: v2.PairOpportunity,
            primary: v2.RelationHypothesis,
            terminal: v2.VerificationAttempt,
        ) -> SecondaryRoute | None:
            """Route a rejected structural candidate to schema competition.

            Structure is immutable evidence, but it does not force the first
            predicate selected by the classifier.  If that candidate is
            rejected and the frozen opportunity exposes exactly one other
            predicate whose schema declares the same structural evidence
            kind, give the secondary pass that alternative.  This keeps the
            recovery generic and bounded while still requiring its own
            validator and ordered verifier.
            """

            if str(terminal.verdict or "").upper() != v2.REJECTED:
                return None
            provenance = opportunity.provenance
            structural_kind = ""
            if (
                provenance.get("hierarchical_list_parent_child") is True
                and provenance.get("structure_admission")
                == "hierarchical_list_parent_child"
            ):
                structural_kind = "list_parent_child"
            elif (
                provenance.get("hierarchical_section_label_parent_child") is True
                and provenance.get("structure_admission")
                == "hierarchical_section_label_parent_child"
            ):
                structural_kind = "section_label_parent_child"
            if not structural_kind:
                return None
            # A parser-confirmed parent/child assertion needs a dedicated
            # second adjudication contract. Re-entering generic predicate
            # competition asks the model to find a lexical relation between a
            # label and its child and therefore recreates the primary false
            # negative. The specialized branch still revalidates the exact
            # endpoints and Schema-declared structural kind.
            return SecondaryRoute(
                ROUTE_STRUCTURAL_REVIEW,
                "STRUCTURAL_REVIEW_AFTER_" + str(terminal.reason_code or "UNKNOWN"),
                "结构解析已确认父子绑定，二次独立复核绑定方向与 Schema 兼容性",
            )
            alternatives = []
            for metadata in opportunity.allowed_predicate_metadata:
                if not isinstance(metadata, Mapping):
                    continue
                name = str(metadata.get("name") or "").strip()
                kinds = metadata.get("structural_evidence_kinds") or []
                if (
                    name
                    and name != str(primary.predicate or "")
                    and structural_kind in kinds
                ):
                    alternatives.append(name)
            if len(set(alternatives)) != 1:
                return None
            return SecondaryRoute(
                ROUTE_PREDICATE_REFINEMENT,
                "STRUCTURAL_ALTERNATIVE_AFTER_" + str(terminal.reason_code or "UNKNOWN"),
                "结构证据明确但一次谓词未通过，排除一次谓词后竞争同结构类型的唯一 Schema 候选",
            )

        routed: list[tuple[int, str, v2.RelationHypothesis, v2.HypothesisValidation, v2.VerificationAttempt, Any]] = []
        for hypothesis in hypotheses:
            terminal = terminal_by_hypothesis.get(str(hypothesis.hypothesis_id))
            if terminal is None:
                continue
            route = route_secondary_attempt(
                terminal.reason_code,
                verdict=terminal.verdict,
                forward_entailment=terminal.forward_entailment,
                reverse_entailment=terminal.reverse_entailment,
                predicate=hypothesis.predicate,
                validation_status=validations.get(str(hypothesis.hypothesis_id), v2.HypothesisValidation(
                    hypothesis_id=hypothesis.hypothesis_id,
                    status=v2.REJECTED,
                    reason_code="MISSING_VALIDATION",
                    predicate_metadata={},
                )).status,
            )
            if v2.source_frame_polarity_review_eligible(
                hypothesis, terminal,
            ):
                route = SecondaryRoute(
                    ROUTE_POLARITY_REVIEW,
                    "PREDICATE_LOCAL_NEGATION_SCOPE_REVIEW",
                    "一次终审将否定范围施加到肯定 source frame，先做独立语法极性审计",
                )
            opportunity = opportunities.get(str(hypothesis.pair_identity))
            if opportunity is not None:
                structural_route = structural_competition_route(
                    opportunity, hypothesis, terminal,
                )
                if (
                    structural_route is not None
                ):
                    route = structural_route
            if route.route == ROUTE_NONE:
                continue
            if opportunity is None:
                continue
            routed.append((
                route_priority(route.route),
                str(hypothesis.hypothesis_id),
                hypothesis,
                validations[str(hypothesis.hypothesis_id)],
                terminal,
                route,
            ))
        routed.sort(key=lambda item: (item[0], item[1]))
        # Keep the bounded pass fair across failure axes. A pure priority
        # slice would consume the whole budget on atomic-inventory failures and
        # starve predicate refinement/role recovery on the same document.
        buckets: dict[str, list[tuple[int, str, Any, Any, Any, Any]]] = {}
        for item in routed:
            buckets.setdefault(str(item[5].route), []).append(item)
        selected: list[tuple[int, str, Any, Any, Any, Any]] = []
        route_order = sorted(
            buckets,
            key=lambda name: route_priority(name),
        )
        while len(selected) < int(Config.relation_secondary_max_candidates):
            progressed = False
            for route_name in route_order:
                queue = buckets[route_name]
                if not queue:
                    continue
                selected.append(queue.pop(0))
                progressed = True
                if len(selected) >= int(Config.relation_secondary_max_candidates):
                    break
            if not progressed:
                break
        skipped = max(0, len(routed) - len(selected))
        if not routed:
            return empty

        pass_id = "shadow_relation_secondary_" + hashlib.sha256(
            f"{run_id}|{fingerprint.root}|{SECONDARY_PASS_VERSION}".encode("utf-8")
        ).hexdigest()
        empty["pass_id"] = pass_id
        empty["selected"] = len(selected)
        empty["skipped_by_budget"] = skipped
        pass_row = self.db.start_relation_secondary_pass({
            "id": pass_id,
            "parent_run_id": run_id,
            "project_id": int(project_id),
            "pass_version": SECONDARY_PASS_VERSION,
            "input_fingerprint": fingerprint.root,
        })
        if str(pass_row.get("status") or "") == "ready":
            empty["metrics"] = dict(pass_row.get("metrics") or {})
            empty["route_counts"] = dict(pass_row.get("route_counts") or {})
            return empty

        route_counts = Counter(str(item[5].route) for item in selected)
        empty["route_counts"] = dict(route_counts)
        planned_rows = []
        for _priority, _key, hypothesis, _validation, _terminal, route in selected:
            opportunity = opportunities[str(hypothesis.pair_identity)]
            candidate_id = "shadow_secondary_candidate_" + hashlib.sha256(
                f"{pass_id}|{hypothesis.hypothesis_id}|{route.route}".encode("utf-8")
            ).hexdigest()
            planned_rows.append({
                "id": candidate_id,
                "parent_hypothesis_id": self.db._grounded_relation_db_id(
                    "relhypdb_", run_id, hypothesis.hypothesis_id,
                ),
                "pair_opportunity_id": opportunity.opportunity_identity,
                "route": route.route,
                "source_reason_code": route.reason_code,
                "predicate_before": hypothesis.predicate,
                "status": "PLANNED",
                "reason_code": route.reason_code,
                "metadata": {
                    "secondary_pass_version": SECONDARY_PASS_VERSION,
                    "primary_hypothesis_id": hypothesis.hypothesis_id,
                    "primary_terminal_verdict": _terminal.verdict,
                    "primary_terminal_reason": _terminal.reason_code,
                },
            })
        self.db.persist_relation_secondary_candidates(pass_id, planned_rows)

        added_hypotheses: list[v2.RelationHypothesis] = []
        added_attempts: list[v2.VerificationAttempt] = []
        added_validations: list[v2.HypothesisValidation] = []
        candidate_updates: list[dict[str, Any]] = []
        model_calls = 0
        secondary_verifier_calls = 0
        polarity_results: Counter[str] = Counter()
        supported = rejected = uncertain = failed = 0
        verify_relation_hypothesis = (
            v2.verify_hypothesis_lean
            if Config.relation_verification_mode == "lean"
            else v2.verify_hypothesis
        )

        for index, item in enumerate(selected, start=1):
            _priority, _key, primary, primary_validation, primary_terminal, route = item
            opportunity = opportunities[str(primary.pair_identity)]
            chunk = context["chunk_by_id"].get(int(opportunity.provenance.get("chunk_id") or 0))
            if not chunk:
                failed += 1
                candidate_updates.append({
                    **planned_rows[index - 1],
                    "status": "FAILED",
                    "reason_code": "SECONDARY_SOURCE_CHUNK_MISSING",
                })
                continue
            content = str(chunk.get("content") or "")
            decision = None
            raw_response: Any = None
            request_payload: dict[str, Any] = {}
            request_hash = ""
            model_attempt_no = 1
            polarity_audit_result = ""
            structural_adjudication_only = route.route == ROUTE_STRUCTURAL_REVIEW
            structural_projection = str(route.reason_code or "").startswith(
                "STRUCTURAL_ALTERNATIVE_AFTER_"
            )
            if route.route == ROUTE_POLARITY_REVIEW:
                metadata = primary.predicate_metadata or {}
                raw_frame_span = metadata.get("source_frame_evidence_span")
                if not (
                    isinstance(raw_frame_span, (list, tuple))
                    and len(raw_frame_span) == 2
                    and all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in raw_frame_span
                    )
                ):
                    failed += 1
                    candidate_updates.append({
                        **planned_rows[index - 1],
                        "status": "FAILED",
                        "reason_code": "POLARITY_FRAME_SPAN_INVALID",
                    })
                    continue
                frame_span = (
                    int(raw_frame_span[0]), int(raw_frame_span[1]),
                )
                if not (
                    frame_span[0] <= opportunity.left_span[0]
                    < opportunity.left_span[1] <= frame_span[1]
                    and frame_span[0] <= opportunity.right_span[0]
                    < opportunity.right_span[1] <= frame_span[1]
                ):
                    failed += 1
                    candidate_updates.append({
                        **planned_rows[index - 1],
                        "status": "FAILED",
                        "reason_code": "POLARITY_FRAME_MISSING_ENDPOINT",
                    })
                    continue
                try:
                    prompt = source_frame_polarity_prompt(
                        content=content,
                        context_span=opportunity.context_span,
                        frame_span=frame_span,
                        frame_text=str(
                            metadata.get("source_frame_evidence_text") or ""
                        ),
                        open_relation=str(
                            metadata.get("source_frame_relation") or ""
                        ),
                    )
                    response_schema = source_frame_polarity_response_schema()
                    request_payload = {
                        "stage": "secondary_source_frame_polarity",
                        "route": route.route,
                        "model": context["model"],
                        "model_revision": context["model_revision"],
                        "system": (
                            "You are an independent grammatical negation-scope "
                            "auditor. Return only the supplied one-field JSON schema."
                        ),
                        "prompt": prompt,
                        "schema": response_schema,
                        "temperature": 0.0,
                        "context_budget_tokens": context_budget_tokens,
                        "max_output_tokens": 256,
                        "safety_margin_tokens": (
                            Config.relation_classifier_safety_margin
                        ),
                        "parent_hypothesis_id": primary.hypothesis_id,
                    }
                    request_hash = relation_model_payload_hash(
                        "request", request_payload,
                    )
                    model_calls += 1
                    raw_response = self.models.json_chat(
                        model=context["model"],
                        system=request_payload["system"],
                        prompt=prompt,
                        schema=response_schema,
                        temperature=0.0,
                        context_budget_tokens=context_budget_tokens,
                        max_output_tokens=256,
                        safety_margin_tokens=(
                            Config.relation_classifier_safety_margin
                        ),
                    )
                    polarity_audit_result = parse_source_frame_polarity(
                        raw_response
                    )
                    polarity_results[polarity_audit_result] += 1
                    raw_text = str(
                        getattr(self.models, "last_raw_response_text", "")
                        or _safe_json_text(raw_response)
                    )
                    response_hash = relation_model_payload_hash(
                        "response", raw_text,
                    )
                    diagnostics = getattr(
                        self.models, "last_call_diagnostics", None,
                    )
                    token_usage = (
                        asdict(diagnostics) if diagnostics is not None else {}
                    )
                    self.db.record_relation_secondary_model_call({
                        "id": self.db._grounded_relation_db_id(
                            "modelcall_", run_id, "secondary_polarity",
                            pass_id, primary.hypothesis_id, model_attempt_no,
                        ),
                        "run_id": run_id,
                        "model_name": context["model"],
                        "model_digest": context["model_revision"],
                        "prompt_version": SECONDARY_POLARITY_PROMPT_VERSION,
                        "schema_version": SECONDARY_POLARITY_SCHEMA_VERSION,
                        "parser_version": SECONDARY_POLARITY_PARSER_VERSION,
                        "idempotency_key": (
                            f"secondary-polarity:{pass_id}:"
                            f"{primary.hypothesis_id}"
                        ),
                        "attempt_no": model_attempt_no,
                        "status": "COMPLETED",
                        "request_hash": request_hash,
                        "response_hash": response_hash,
                        "request_payload": request_payload,
                        "raw_response_text": raw_text,
                        "parsed_response": {
                            "result": polarity_audit_result,
                        },
                        "token_usage": token_usage,
                        "terminal_status": "COMPLETED",
                    })
                except Exception as exc:
                    failed += 1
                    self.db.record_relation_secondary_model_call({
                        "id": self.db._grounded_relation_db_id(
                            "modelcall_", run_id, "secondary_polarity",
                            pass_id, primary.hypothesis_id, model_attempt_no,
                        ),
                        "run_id": run_id,
                        "model_name": context["model"],
                        "model_digest": context["model_revision"],
                        "prompt_version": SECONDARY_POLARITY_PROMPT_VERSION,
                        "schema_version": SECONDARY_POLARITY_SCHEMA_VERSION,
                        "parser_version": SECONDARY_POLARITY_PARSER_VERSION,
                        "idempotency_key": (
                            f"secondary-polarity:{pass_id}:"
                            f"{primary.hypothesis_id}"
                        ),
                        "attempt_no": model_attempt_no,
                        "status": "FAILED",
                        "request_hash": request_hash,
                        "response_hash": "",
                        "request_payload": request_payload,
                        "raw_response_text": str(
                            getattr(self.models, "last_raw_response_text", "")
                            or ""
                        ),
                        "parsed_response": {},
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:2000],
                        "terminal_status": "FAILED",
                    })
                    candidate_updates.append({
                        **planned_rows[index - 1],
                        "status": "FAILED",
                        "reason_code": "POLARITY_MODEL_OR_PROTOCOL_FAILURE",
                        "metadata": {
                            "error": f"{type(exc).__name__}: {exc}"[:500],
                        },
                    })
                    continue
                if polarity_audit_result != "ASSERTED":
                    candidate_updates.append({
                        **planned_rows[index - 1],
                        "status": (
                            "MODEL_NO_RELATION"
                            if polarity_audit_result == "NEGATED"
                            else "MODEL_INSUFFICIENT"
                        ),
                        "reason_code": (
                            "POLARITY_" + polarity_audit_result
                        ),
                        "metadata": {
                            "source_frame_polarity_audit": (
                                polarity_audit_result
                            ),
                        },
                    })
                    continue
                decision = SecondaryDecision(
                    SECONDARY_RELATION,
                    primary.predicate,
                    (
                        "LEFT"
                        if primary.subject_occurrence_key
                        == opportunity.left_occurrence_key
                        else "RIGHT"
                    ),
                    (
                        "RIGHT"
                        if primary.object_occurrence_key
                        == opportunity.right_occurrence_key
                        else "LEFT"
                    ),
                    (frame_span,),
                    route.reason_code,
                )
            elif structural_projection:
                # The planner has already frozen a parent/child relation and
                # the route selected the unique Schema predicate declaring
                # that structural evidence kind.  Project exactly that
                # candidate without asking the same model to overrule an
                # immutable structure cue; it still goes through the normal
                # validator and independent verifier below.
                structural_kind = ""
                provenance = opportunity.provenance
                if (
                    provenance.get("hierarchical_list_parent_child") is True
                    and provenance.get("structure_admission")
                    == "hierarchical_list_parent_child"
                ):
                    structural_kind = "list_parent_child"
                elif (
                    provenance.get("hierarchical_section_label_parent_child") is True
                    and provenance.get("structure_admission")
                    == "hierarchical_section_label_parent_child"
                ):
                    structural_kind = "section_label_parent_child"
                alternative_names = [
                    str(value.get("name") or "")
                    for value in opportunity.allowed_predicate_metadata
                    if str(value.get("name") or "")
                    and str(value.get("name") or "") != str(primary.predicate or "")
                    and structural_kind in (value.get("structural_evidence_kinds") or [])
                ]
                if structural_kind and len(set(alternative_names)) == 1:
                    child, parent = v2._structural_parent_child_keys(opportunity)
                    decision = SecondaryDecision(
                        SECONDARY_RELATION,
                        alternative_names[0],
                        "LEFT" if child == opportunity.left_occurrence_key else "RIGHT",
                        "LEFT" if parent == opportunity.left_occurrence_key else "RIGHT",
                        (tuple(map(int, opportunity.context_span)),),
                        route.reason_code,
                    )
                else:
                    failed += 1
                    candidate_updates.append({
                        **planned_rows[index - 1],
                        "status": "FAILED",
                        "reason_code": "STRUCTURAL_ALTERNATIVE_UNRESOLVED",
                    })
                    continue
            elif structural_adjudication_only:
                child, parent = v2._structural_parent_child_keys(opportunity)
                decision = SecondaryDecision(
                    SECONDARY_RELATION,
                    primary.predicate,
                    "LEFT" if child == opportunity.left_occurrence_key else "RIGHT",
                    "LEFT" if parent == opportunity.left_occurrence_key else "RIGHT",
                    (tuple(map(int, opportunity.context_span)),),
                    route.reason_code,
                )
            elif route.route in {
                ROUTE_EVIDENCE_RECOVERY,
                ROUTE_PREDICATE_REFINEMENT,
                ROUTE_TYPE_ROLE_REVIEW,
                ROUTE_ROLE_RETRY,
            }:
                excluded_predicate = (
                    str(primary.predicate or "")
                    if route.route == ROUTE_PREDICATE_REFINEMENT else ""
                )
                allowed_metadata = [
                    value for value in opportunity.allowed_predicate_metadata
                    if not excluded_predicate
                    or str(value.get("name") or "") != excluded_predicate
                ]
                preferred_predicate = (
                    str(primary.predicate or "")
                    if route.route == ROUTE_ROLE_RETRY else ""
                )
                if preferred_predicate:
                    # ROLE_RETRY is an endpoint/方向复核, not predicate
                    # discovery.  Keep the original predicate as the sole
                    # schema option so a weak cue cannot silently turn a
                    # role failure into a different semantic relation.
                    allowed_metadata = [
                        value for value in allowed_metadata
                        if str(value.get("name") or "") == preferred_predicate
                    ]
                prompt = secondary_recovery_prompt(
                    route=route,
                    content=content,
                    context_span=opportunity.context_span,
                    left_span=opportunity.left_span,
                    right_span=opportunity.right_span,
                    left_surface=content[opportunity.left_span[0]:opportunity.left_span[1]],
                    right_surface=content[opportunity.right_span[0]:opportunity.right_span[1]],
                    left_type=opportunity.left_type,
                    right_type=opportunity.right_type,
                    allowed_predicate_metadata=allowed_metadata,
                    structural_contract={
                        "structure_admission": str(
                            opportunity.provenance.get("structure_admission") or ""
                        ),
                        "hierarchical_parent_child": bool(
                            opportunity.provenance.get("hierarchical_list_parent_child")
                        ),
                        "section_label_parent_child": bool(
                            opportunity.provenance.get(
                                "hierarchical_section_label_parent_child"
                            )
                        ),
                        "source_roles": list(opportunity.provenance.get("source_roles") or []),
                        "required_atomic_scope": list(
                            opportunity.provenance.get("required_atomic_scope") or []
                        ),
                    },
                    orientation_only=(route.route == ROUTE_ROLE_RETRY),
                    excluded_predicate=excluded_predicate,
                    preferred_predicate=preferred_predicate,
                )
                response_schema = secondary_response_schema([
                    str(value.get("name") or "")
                    for value in allowed_metadata
                ])
                request_payload = {
                    "stage": "secondary_relation_recovery",
                    "route": route.route,
                    "model": context["model"],
                    "model_revision": context["model_revision"],
                    "system": "You are an independent secondary relation evidence recovery judge.",
                    "prompt": prompt,
                    "schema": response_schema,
                    "temperature": 0.0,
                    "context_budget_tokens": context_budget_tokens,
                    "max_output_tokens": self._verifier_output_budget(context_budget_tokens),
                    "safety_margin_tokens": Config.relation_classifier_safety_margin,
                    "parent_hypothesis_id": primary.hypothesis_id,
                }
                request_hash = relation_model_payload_hash("request", request_payload)
                try:
                    model_calls += 1
                    raw_response = self.models.json_chat(
                        model=context["model"],
                        system=request_payload["system"],
                        prompt=prompt,
                        schema=response_schema,
                        temperature=0.0,
                        context_budget_tokens=context_budget_tokens,
                        max_output_tokens=self._verifier_output_budget(context_budget_tokens),
                        safety_margin_tokens=Config.relation_classifier_safety_margin,
                    )
                    raw_text = str(
                        getattr(self.models, "last_raw_response_text", "")
                        or _safe_json_text(raw_response)
                    )
                    response_hash = relation_model_payload_hash("response", raw_text)
                    diagnostics = getattr(self.models, "last_call_diagnostics", None)
                    token_usage = asdict(diagnostics) if diagnostics is not None else {}
                    self.db.record_relation_secondary_model_call({
                        "id": self.db._grounded_relation_db_id(
                            "modelcall_", run_id, "secondary_recovery",
                            pass_id, primary.hypothesis_id, model_attempt_no,
                        ),
                        "run_id": run_id,
                        "model_name": context["model"],
                        "model_digest": context["model_revision"],
                        "prompt_version": SECONDARY_PROMPT_VERSION,
                        "schema_version": SECONDARY_SCHEMA_VERSION,
                        "parser_version": SECONDARY_PARSER_VERSION,
                        "idempotency_key": f"secondary:{pass_id}:{primary.hypothesis_id}",
                        "attempt_no": model_attempt_no,
                        "status": "COMPLETED",
                        "request_hash": request_hash,
                        "response_hash": response_hash,
                        "request_payload": request_payload,
                        "raw_response_text": raw_text,
                        "parsed_response": raw_response,
                        "token_usage": token_usage,
                        "terminal_status": "COMPLETED",
                    })
                    decision = parse_secondary_decision(
                        raw_response,
                        allowed_predicates=[str(value.get("name") or "") for value in allowed_metadata],
                        context_span=opportunity.context_span,
                        left_span=opportunity.left_span,
                        right_span=opportunity.right_span,
                        content_length=len(content),
                        left_surface=content[
                            opportunity.left_span[0]:opportunity.left_span[1]
                        ],
                        right_surface=content[
                            opportunity.right_span[0]:opportunity.right_span[1]
                        ],
                        required_predicate=preferred_predicate,
                    )
                except Exception as exc:
                    failed += 1
                    self.db.record_relation_secondary_model_call({
                        "id": self.db._grounded_relation_db_id(
                            "modelcall_", run_id, "secondary_recovery",
                            pass_id, primary.hypothesis_id, model_attempt_no,
                        ),
                        "run_id": run_id,
                        "model_name": context["model"],
                        "model_digest": context["model_revision"],
                        "prompt_version": SECONDARY_PROMPT_VERSION,
                        "schema_version": SECONDARY_SCHEMA_VERSION,
                        "parser_version": SECONDARY_PARSER_VERSION,
                        "idempotency_key": f"secondary:{pass_id}:{primary.hypothesis_id}",
                        "attempt_no": model_attempt_no,
                        "status": "FAILED",
                        "request_hash": request_hash,
                        "response_hash": "",
                        "request_payload": request_payload,
                        "raw_response_text": str(
                            getattr(self.models, "last_raw_response_text", "") or ""
                        ),
                        "parsed_response": {},
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:2000],
                        "terminal_status": "FAILED",
                    })
                    candidate_updates.append({
                        **planned_rows[index - 1],
                        "status": "FAILED",
                        "reason_code": "SECONDARY_MODEL_OR_PROTOCOL_FAILURE",
                        "metadata": {"error": f"{type(exc).__name__}: {exc}"[:500]},
                    })
                    continue
                if decision.decision != SECONDARY_RELATION:
                    status = (
                        "MODEL_NO_RELATION"
                        if decision.decision == SECONDARY_NO_RELATION
                        else "MODEL_INSUFFICIENT"
                    )
                    candidate_updates.append({
                        **planned_rows[index - 1],
                        "status": status,
                        "reason_code": decision.reason_code,
                        "metadata": {"secondary_decision": decision.decision},
                    })
                    continue
            else:
                # Role and technical routes intentionally do not ask a second
                # candidate-generation question. They run a fresh ordered
                # verifier on the same immutable endpoints, with the primary
                # verdict withheld from the prompt.
                decision = SecondaryDecision(
                    SECONDARY_RELATION,
                    primary.predicate,
                    "LEFT" if primary.subject_occurrence_key == opportunity.left_occurrence_key else "RIGHT",
                    "RIGHT" if primary.object_occurrence_key == opportunity.right_occurrence_key else "LEFT",
                    tuple((span.start, span.end) for span in primary.evidence_spans),
                    route.reason_code,
                )

            if decision is None:
                failed += 1
                continue
            side_to_occurrence = {
                "LEFT": opportunity.left_occurrence_key,
                "RIGHT": opportunity.right_occurrence_key,
            }
            predicate_metadata = next(
                (
                    dict(value) for value in opportunity.allowed_predicate_metadata
                    if str(value.get("name") or "") == decision.predicate
                ),
                {},
            )
            preserved_source_metadata = (
                dict(primary.predicate_metadata or {})
                if route.route in {
                    ROUTE_POLARITY_REVIEW,
                    ROUTE_ROLE_RETRY,
                } else {}
            )
            candidate_prompt_version = (
                SECONDARY_POLARITY_PROMPT_VERSION
                if route.route == ROUTE_POLARITY_REVIEW
                else SECONDARY_PROMPT_VERSION
            )
            candidate_schema_version = (
                SECONDARY_POLARITY_SCHEMA_VERSION
                if route.route == ROUTE_POLARITY_REVIEW
                else SECONDARY_SCHEMA_VERSION
            )
            candidate_parser_version = (
                SECONDARY_POLARITY_PARSER_VERSION
                if route.route == ROUTE_POLARITY_REVIEW
                else SECONDARY_PARSER_VERSION
            )
            secondary_id = "secondary_hyp_" + hashlib.sha256(
                f"{pass_id}|{primary.hypothesis_id}|{route.route}|"
                f"{decision.predicate}|"
                f"{side_to_occurrence.get(decision.subject_side, '')}|"
                f"{side_to_occurrence.get(decision.object_side, '')}".encode("utf-8")
            ).hexdigest()
            secondary_hypothesis = v2.RelationHypothesis(
                hypothesis_id=secondary_id,
                pair_identity=primary.pair_identity,
                subject_occurrence_key=side_to_occurrence[decision.subject_side],
                predicate=decision.predicate,
                object_occurrence_key=side_to_occurrence[decision.object_side],
                evidence_spans=tuple(
                    v2.EvidenceSpan(start=start, end=end)
                    for start, end in decision.evidence_spans
                ),
                classifier_confidence=0.5,
                classifier_policy_version=SECONDARY_PASS_VERSION,
                classifier_model_revision=context["model_revision"],
                classifier_prompt_version=candidate_prompt_version,
                classifier_schema_version=candidate_schema_version,
                classifier_parser_version=candidate_parser_version,
                predicate_metadata={
                    **preserved_source_metadata,
                    **predicate_metadata,
                    "candidate_source": "SECONDARY_REVIEW",
                    "secondary_pass_id": pass_id,
                    "primary_hypothesis_id": primary.hypothesis_id,
                    "route": route.route,
                    "source_frame_polarity_audit": polarity_audit_result,
                    "source_frame_polarity_audit_version": (
                        SECONDARY_POLARITY_PROMPT_VERSION
                        if polarity_audit_result else ""
                    ),
                },
                status="SECONDARY_PROPOSED",
            )
            self.db.persist_grounded_relation_proposed_hypotheses(
                run_id, [secondary_hypothesis], candidate_source="SECONDARY_REVIEW",
            )
            added_hypotheses.append(secondary_hypothesis)
            secondary_validation = v2.validate_hypothesis(
                secondary_hypothesis,
                opportunity,
                content=content,
                logical_schema=context["schema"],
            )
            self.db.persist_grounded_relation_validations(
                run_id, [secondary_validation],
            )
            added_validations.append(secondary_validation)
            if secondary_validation.status != v2.PASS:
                secondary_attempt = verify_relation_hypothesis(
                    secondary_hypothesis,
                    secondary_validation,
                    opportunity,
                    content=content,
                    logical_schema=context["schema"],
                    role_probe=lambda _prompt, _schema: {},
                    role_alignment_probe=lambda _prompt, _schema: {},
                    verifier=lambda _prompt, _schema: {},
                    verifier_model_revision=context["model_revision"],
                    atomic_fact_probe=None,
                    atomic_frame_alignment_probe=None,
                    attempt_no=1,
                    structural_adjudication_only=structural_adjudication_only,
                    independent_orientation_audit=(
                        "FORWARD"
                        if route.route == ROUTE_ROLE_RETRY else ""
                    ),
                )
            else:
                def secondary_final_verifier(prompt: str, schema: dict[str, Any]) -> Any:
                    nonlocal secondary_verifier_calls
                    secondary_verifier_calls += 1
                    return self._verify(
                        model=context["model"],
                        context_budget_tokens=context_budget_tokens,
                        prompt=prompt,
                        response_schema=schema,
                        attempt_no=1,
                    )

                def secondary_role_probe(prompt: str, _schema: dict[str, Any]) -> Any:
                    return self._verify_predicate_blind_endpoint_probe(
                        model=context["model"],
                        context_budget_tokens=context_budget_tokens,
                        prompt=prompt,
                        attempt_no=1,
                    )

                def secondary_role_alignment(prompt: str, _schema: dict[str, Any]) -> Any:
                    return self._align_verifier_roles(
                        model=context["model"],
                        context_budget_tokens=context_budget_tokens,
                        prompt=prompt,
                        response_schema=v2.verifier_role_alignment_response_schema(),
                        attempt_no=1,
                    )

                secondary_attempt = verify_relation_hypothesis(
                    secondary_hypothesis,
                    secondary_validation,
                    opportunity,
                    content=content,
                    logical_schema=context["schema"],
                    role_probe=secondary_role_probe,
                    role_alignment_probe=secondary_role_alignment,
                    verifier=secondary_final_verifier,
                    verifier_model_revision=context["model_revision"],
                    atomic_fact_probe=None,
                    atomic_frame_alignment_probe=None,
                    attempt_no=1,
                    structural_adjudication_only=structural_adjudication_only,
                    independent_orientation_audit=(
                        "FORWARD"
                        if route.route == ROUTE_ROLE_RETRY else ""
                    ),
                )
            self.db.persist_grounded_relation_verification_attempts(
                run_id, [secondary_attempt], terminal=True,
            )
            added_attempts.append(secondary_attempt)
            status = str(secondary_attempt.verdict)
            if status == v2.SUPPORTED:
                supported += 1
            elif status == v2.REJECTED:
                rejected += 1
            elif status == v2.UNCERTAIN:
                uncertain += 1
            else:
                failed += 1
                # The verification ledger has more detailed retry/protocol
                # verdicts than the secondary candidate table. Keep the
                # detailed verdict in reason_code, but map all non-terminal
                # success/rejection states to the table's legal FAILED state
                # so an audit write can never fail the whole build.
                status = "FAILED"
            candidate_updates.append({
                **planned_rows[index - 1],
                "predicate_after": decision.predicate,
                "subject_occurrence_key": secondary_hypothesis.subject_occurrence_key,
                "object_occurrence_key": secondary_hypothesis.object_occurrence_key,
                "evidence_spans": [list(span) for span in decision.evidence_spans],
                "status": status,
                "reason_code": secondary_attempt.reason_code,
                "metadata": {
                    "secondary_hypothesis_id": secondary_id,
                    "secondary_attempt_id": secondary_attempt.attempt_id,
                    "route": route.route,
                    "source_frame_polarity_audit": polarity_audit_result,
                },
            })
        if candidate_updates:
            self.db.persist_relation_secondary_candidates(
                pass_id, candidate_updates,
            )
        metrics = {
            "selected": len(selected),
            "routed": len(routed),
            "skipped_by_budget": skipped,
            "model_calls": model_calls,
            "verifier_model_calls": secondary_verifier_calls,
            "total_model_calls": model_calls + secondary_verifier_calls,
            "added_hypotheses": len(added_hypotheses),
            "supported": supported,
            "rejected": rejected,
            "uncertain": uncertain,
            "failed": failed,
            "pass_version": SECONDARY_PASS_VERSION,
            "prompt_version": SECONDARY_PROMPT_VERSION,
            "parser_version": SECONDARY_PARSER_VERSION,
            "polarity_results": dict(polarity_results),
            "polarity_prompt_version": SECONDARY_POLARITY_PROMPT_VERSION,
        }
        self.db.finish_relation_secondary_pass(
            pass_id,
            status="ready",
            metrics=metrics,
            route_counts=dict(route_counts),
        )
        empty.update({
            "metrics": metrics,
            "model_calls": model_calls,
            "supported": supported,
            "rejected": rejected,
            "uncertain": uncertain,
            "failed": failed,
            "added_hypotheses": added_hypotheses,
            "added_attempts": added_attempts,
            "added_validations": added_validations,
        })
        return empty

    def run_namespace(
        self,
        *,
        namespace_id: str,
        project_id: int,
        progress: Callable[[int, int, str], None] | None = None,
        run_id_salt: str = "",
    ) -> dict[str, Any]:
        """Execute planning, ledgers, models, normalization and materialization."""

        self.db.assert_grounded_relation_ready()
        context = self._context(project_id, namespace_id)
        fingerprint = context["fingerprint"]
        run_identity = f"{namespace_id}|{fingerprint.root}"
        if run_id_salt:
            run_identity += "|" + str(run_id_salt)
        run_id = "shadow_relation_v2_" + hashlib.sha256(
            run_identity.encode("utf-8")
        ).hexdigest()
        started = False
        terminal_status = ""
        metrics: dict[str, Any] = {}
        try:
            self.db.start_grounded_relation_run({
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
            self.db.persist_grounded_relation_source_contexts(
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
                    "endpoint_candidate": bool(
                        (mention.get("provenance") or {}).get(
                            "endpoint_candidate",
                        )
                    ),
                })
            plans_by_context: dict[str, v2.PairOpportunityPlan] = {}
            mentions_by_context: dict[str, list[dict[str, Any]]] = {}
            all_opportunities: list[v2.PairOpportunity] = []
            proposal_batches: list[tuple[str, v2.RelationProposalBatch]] = []
            completed_proposer_scopes: dict[
                str, set[tuple[int, int]]
            ] = {}
            proposer_scopes_total = 0
            proposer_scopes_complete = 0
            proposer_scopes_fallback = 0
            proposer_calls = 0
            proposer_cache_hits = 0
            proposer_failures = 0
            proposer_attempt_counts: Counter[str] = Counter()
            proposer_failure_keys: Counter[tuple[str, str, str]] = Counter()
            proposer_circuit_breaker_tripped = False
            proposer_skipped_by_circuit = 0
            proposer_terminal_statuses: Counter[str] = Counter()
            proposer_hypothesis_rejections: Counter[str] = Counter()
            relation_chunks = list(context["relation_chunks"])
            total_chunks = len(relation_chunks)
            # The long-lived relation run has six independently observable
            # phases: deterministic pair planning, bounded proposal recall,
            # source-fact inventory, immutable ledger persistence,
            # classification, and final verification.  Keep each phase in a
            # separate interval so a large provenance ledger cannot look like
            # a stalled 50%-complete task simply because it performs no model
            # calls while it is being committed.
            total_progress = max(1, total_chunks * 6)
            source_first_enabled = callable(
                getattr(
                    self.db,
                    "record_grounded_relation_source_model_call",
                    None,
                )
            )
            source_first_primary_contexts = 0
            proposer_scopes_skipped_by_source_first = 0
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
                        GROUNDED_RELATION_V2_CANDIDATE_POLICY_VERSION,
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
                            "macro_context_hash": str(
                                context["macro_context"].get("hash") or ""
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
                content = str(chunk.get("content") or "")
                scope_units: list[
                    tuple[tuple[int, int], list[dict[str, Any]]]
                ] = []
                for scope_start, scope_end, _scope_text in (
                    split_relation_scopes(content)
                ):
                    scope_mentions = [
                        item for item in chunk_mentions
                        if (
                            scope_start <= int(item["mention_start"])
                            < int(item["mention_end"]) <= scope_end
                        )
                    ]
                    if len(scope_mentions) >= 2:
                        scope_units.append(
                            ((scope_start, scope_end), scope_mentions)
                        )
                proposer_scopes_total += len(scope_units)
                if (
                    source_first_enabled
                    and Config.relation_grounded_open_facts_enabled
                    and len(chunk_mentions) >= 2
                ):
                    # The immutable occurrence-grounded source inventory is
                    # now the primary semantic proposal pass.  Running the
                    # older per-scope proposer first duplicated model work and
                    # could not suppress the later inventory/classifier chain.
                    # Pair planning remains available as a fail-safe
                    # classifier ledger if the source pass misses or fails.
                    source_first_primary_contexts += 1
                    proposer_scopes_skipped_by_source_first += len(scope_units)
                    continue
                proposer_message = (
                    f"Knowledge Forge 候选补充 {chunk_index}/{total_chunks} "
                    "跳过：当前范围内实体不足"
                )
                successful_scopes = 0
                fallback_scopes = 0
                context_kind = str(
                    chunk.get("shadow_context_kind") or "legacy_chunk"
                )
                for scope_span, scope_mentions in scope_units:
                    blocked = any(
                        key[0] == str(chunk.get("document_id") or "")
                        and key[1] == context_kind
                        and count >= int(
                            Config.relation_proposer_failure_circuit_breaker
                        ) > 0
                        for key, count in proposer_failure_keys.items()
                    )
                    if blocked:
                        proposer_skipped_by_circuit += 1
                        proposer_circuit_breaker_tripped = True
                        proposer_scopes_fallback += 1
                        fallback_scopes += 1
                        proposer_terminal_statuses[
                            "TERMINAL_FAILED_WITH_PLANNER_FALLBACK"
                        ] += 1
                        continue
                    scope_result = self._propose_atomic_scope(
                        run_id=run_id,
                        context=context,
                        context_budget_tokens=context_budget,
                        context_key=context_key,
                        chunk=chunk,
                        scope_span=scope_span,
                        mentions=scope_mentions,
                        attempt_counts=proposer_attempt_counts,
                        macro_context=select_macro_context(
                            context["macro_context"],
                            [item.get("canonical_id") for item in scope_mentions],
                        ),
                    )
                    proposer_calls += int(scope_result["calls"])
                    proposer_cache_hits += int(scope_result["cache_hits"])
                    proposer_failures += int(scope_result["failures"])
                    proposal_batches.extend(
                        (context_key, batch)
                        for batch in scope_result["batches"]
                    )
                    error_class = str(scope_result["error_class"] or "")
                    if error_class:
                        proposer_failure_keys[(
                            str(chunk.get("document_id") or ""),
                            context_kind,
                            error_class,
                        )] += 1
                        proposer_scopes_fallback += 1
                        fallback_scopes += 1
                        proposer_terminal_statuses[
                            "TERMINAL_FAILED_WITH_PLANNER_FALLBACK"
                        ] += 1
                    else:
                        successful_scopes += 1
                        proposer_terminal_statuses[
                            "SUCCESS_WITH_PROPOSALS"
                            if scope_result["found_keys"] else "SUCCESS_EMPTY"
                        ] += 1
                    if bool(scope_result["complete"]):
                        completed_proposer_scopes.setdefault(
                            context_key, set(),
                        ).add(scope_span)
                        proposer_scopes_complete += 1
                if successful_scopes:
                    proposer_message = (
                        f"Knowledge Forge 候选补充 {chunk_index}/{total_chunks} "
                        f"完成（{successful_scopes} 个语义范围）"
                    )
                elif fallback_scopes:
                    proposer_message = (
                        f"Knowledge Forge 候选补充 {chunk_index}/{total_chunks} "
                        "未采用，已转入候选对分类"
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
            # Source-first recovery can update many candidates after planning.
            # Keep the two immutable ledgers (project-wide and per-context)
            # addressable by pair identity; repeatedly scanning every planned
            # pair turns a dense document from linear work into quadratic work.
            opportunity_index_by_pair = {
                str(item.pair_identity): index
                for index, item in enumerate(all_opportunities)
            }
            opportunity_plan_location_by_pair: dict[str, tuple[str, int]] = {
                str(item.pair_identity): (context_key, index)
                for context_key, plan in plans_by_context.items()
                for index, item in enumerate(plan.opportunities)
            }
            chunks_by_context = {
                str(chunk.get("shadow_context_id") or chunk["id"]): chunk
                for chunk in relation_chunks
            }
            missing_endpoint_mentions = 0
            source_inventory_calls = 0
            source_binding_calls = 0
            source_inventory_cache_hits = 0
            source_binding_cache_hits = 0
            source_inventory_failures = 0
            source_binding_failures = 0
            source_inventory_incomplete = 0
            source_open_fact_gleaning_calls = 0
            source_open_fact_gleaning_cache_hits = 0
            source_open_fact_gleaning_failures = 0
            source_open_fact_gleaning_rounds_completed = 0
            source_open_facts_recovered = 0
            source_open_fact_reverse_recoveries_suppressed = 0
            source_reference_proxies_accepted = 0
            source_reference_bindings_dropped = 0
            source_reference_binding_conflicts = 0
            source_reference_adjudication_targets = 0
            source_reference_adjudication_calls = 0
            source_reference_adjudication_cache_hits = 0
            source_reference_adjudication_failures = 0
            source_reference_adjudication_proxies_accepted = 0
            source_reference_drop_reasons: Counter[str] = Counter()
            reference_conflict_spans: dict[
                str, set[tuple[int, int]]
            ] = {}
            source_deterministic_frames_recovered = 0
            source_deterministic_frames_by_origin: Counter[str] = Counter()
            source_binding_incomplete = 0
            source_binding_items_dropped = 0
            source_bindings_total = 0
            source_binding_model_skipped_empty_contract = 0
            source_binding_review_candidates_enabled = bool(
                Config.relation_secondary_enabled
            )
            source_binding_gleaning_calls = 0
            source_binding_gleaning_cache_hits = 0
            source_binding_gleaning_failures = 0
            source_binding_gleaning_rounds_completed = 0
            source_bindings_recovered = 0
            source_structure_bindings_total = 0
            source_inventory_audited_contexts: set[str] = set()
            source_inventory_complete_contexts: set[str] = set()
            source_repair_pair_ids: set[str] = set()
            source_first_classifier_backstop_pairs = 0
            source_repair_classifier_backstop_pairs = 0
            source_first_classifier_pairs_skipped = 0
            source_parser_endpoint_pairs_filtered = 0
            source_recall_shortlist_bindings = 0
            source_recall_shortlist_frames_considered = 0
            source_recall_shortlist_candidates_considered = 0
            source_recall_shortlist_candidates_below_threshold = 0
            source_open_carrier_bindings = 0
            source_open_carrier_contexts = 0
            source_open_carrier_eligible_frames = 0
            source_open_carrier_skipped_precise = 0
            source_open_carrier_parallel_precise = 0
            source_open_carrier_skipped_low_information = 0
            source_open_carrier_bindings_capped = 0
            source_projection_hypotheses: list[v2.RelationHypothesis] = []

            def replace_opportunity(updated: v2.PairOpportunity) -> None:
                pair_id = str(updated.pair_identity)
                all_index = opportunity_index_by_pair.get(pair_id)
                location = opportunity_plan_location_by_pair.get(pair_id)
                if all_index is None or location is None:
                    raise LookupError(
                        "source projection pair 不存在于冻结候选账本"
                    )
                plan_context, plan_index = location
                plan = plans_by_context[plan_context]
                plan_items = list(plan.opportunities)
                if (
                    plan_index >= len(plan_items)
                    or str(plan_items[plan_index].pair_identity) != pair_id
                ):
                    raise RuntimeError("候选账本索引与上下文计划不一致")
                all_opportunities[all_index] = updated
                plan_items[plan_index] = updated
                plans_by_context[plan_context] = replace(
                    plan, opportunities=tuple(plan_items),
                )
                opportunity_by_pair[pair_id] = updated

            def add_direct_opportunity(
                context_key: str,
                opportunity: v2.PairOpportunity,
            ) -> None:
                all_opportunities.append(opportunity)
                plan = plans_by_context[context_key]
                pair_id = str(opportunity.pair_identity)
                opportunity_index_by_pair[pair_id] = len(all_opportunities) - 1
                opportunity_plan_location_by_pair[pair_id] = (
                    context_key, len(plan.opportunities),
                )
                plans_by_context[context_key] = replace(
                    plan,
                    opportunities=(*plan.opportunities, opportunity),
                    theoretical_pairs=max(
                        int(plan.theoretical_pairs),
                        len(plan.opportunities) + 1,
                    ),
                )
                opportunity_by_pair[pair_id] = opportunity

            def merge_reference_proxies(
                context_key: str,
                parsed_inventory: Mapping[str, Any],
                chunk: Mapping[str, Any],
            ) -> set[str]:
                """Merge relation-only reference endpoints fail-closed.

                Two model rounds that bind the same local expression to
                different canonical entities invalidate both bindings and all
                dependent frames. The frozen entity inventory is untouched.
                """

                nonlocal source_reference_proxies_accepted
                nonlocal source_reference_bindings_dropped
                nonlocal source_reference_binding_conflicts
                for dropped in (
                    parsed_inventory.get("dropped_references") or ()
                ):
                    if not isinstance(dropped, Mapping):
                        continue
                    reason = str(dropped.get("reason") or "unknown")
                    source_reference_drop_reasons[reason] += 1
                    if reason != "unused_reference":
                        source_reference_bindings_dropped += 1
                current = mentions_by_context[context_key]
                existing_keys = {
                    str(item.get("occurrence_key") or "")
                    for item in current
                }
                existing_by_span: dict[
                    tuple[int, int], list[Mapping[str, Any]]
                ] = {}
                for item in current:
                    if not bool(item.get("relation_reference_proxy")):
                        continue
                    span = (
                        int(item.get("mention_start") or 0),
                        int(item.get("mention_end") or 0),
                    )
                    existing_by_span.setdefault(span, []).append(item)
                conflicts = reference_conflict_spans.setdefault(
                    context_key, set(),
                )
                invalidated: set[str] = set()
                incoming = [
                    dict(item)
                    for item in (
                        parsed_inventory.get("reference_proxies") or ()
                    )
                    if isinstance(item, Mapping)
                ]
                content = str(chunk.get("content") or "")
                structure = chunk.get("shadow_structure")
                if not isinstance(structure, dict):
                    structure = {}
                metadata = chunk.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                context_kind = str(
                    chunk.get("shadow_context_kind") or ""
                )
                for raw_proxy in incoming:
                    profile = occurrence_scope(
                        content=content,
                        mention_start=int(
                            raw_proxy.get("mention_start") or 0
                        ),
                        mention_end=int(raw_proxy.get("mention_end") or 0),
                        structure=structure,
                        context_kind=context_kind,
                        metadata=metadata,
                    )
                    scope = dict(profile["occurrence_scope"])
                    proxy = {
                        **raw_proxy,
                        "physical_occurrence_key": str(
                            raw_proxy.get("occurrence_key") or ""
                        ),
                        "source_role": str(profile["source_role"]),
                        "relation_eligible": bool(
                            profile["relation_eligible"]
                        ),
                        "scope_id": str(scope["scope_id"]),
                        "scope_kind": str(scope["scope_kind"]),
                        "scope_start": int(scope["start"]),
                        "scope_end": int(scope["end"]),
                        "list_group_id": str(
                            scope.get("list_group_id") or ""
                        ),
                        "list_depth": int(scope.get("list_depth") or 0),
                        "list_parent_scope_id": str(
                            scope.get("list_parent_scope_id") or ""
                        ),
                        # Reference proxies are introduced after the frozen
                        # mention inventory has selected each list item's
                        # principal node.  They may participate in semantic
                        # frames, but cannot retroactively become a structural
                        # list head.
                        "list_scope_head": False,
                        "table_id": str(
                            raw_proxy.get("table_id")
                            or structure.get("table_id")
                            or ""
                        ),
                        "row_id": str(
                            raw_proxy.get("row_id")
                            or structure.get("row_id")
                            or ""
                        ),
                        "cell_id": str(
                            raw_proxy.get("cell_id")
                            or structure.get("cell_id")
                            or ""
                        ),
                    }
                    key = str(proxy.get("occurrence_key") or "")
                    span = (
                        int(proxy.get("mention_start") or 0),
                        int(proxy.get("mention_end") or 0),
                    )
                    if not key or key in existing_keys:
                        continue
                    if span in conflicts:
                        invalidated.add(key)
                        continue
                    existing = existing_by_span.get(span) or []
                    canonical_ids = {
                        str(item.get("canonical_id") or "")
                        for item in existing
                    }
                    canonical_ids.add(
                        str(proxy.get("canonical_id") or "")
                    )
                    if len(canonical_ids) > 1:
                        conflicts.add(span)
                        invalidated.add(key)
                        invalidated.update(
                            str(item.get("occurrence_key") or "")
                            for item in existing
                        )
                        source_reference_binding_conflicts += 1
                        continue
                    current.append(proxy)
                    existing_keys.add(key)
                    existing_by_span.setdefault(span, []).append(proxy)
                    source_reference_proxies_accepted += 1
                if invalidated:
                    mentions_by_context[context_key] = [
                        item for item in current
                        if str(item.get("occurrence_key") or "")
                        not in invalidated
                    ]
                return invalidated

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
                        updated = self._promote_local_subgraph_opportunity(
                            opportunity, proposal,
                        )
                        if updated != opportunity:
                            opportunity = updated
                            replace_opportunity(opportunity)

            # Source-first recovery is intentionally a separate candidate
            # source.  The existing atomic inventory used to run only after a
            # pair had already been proposed, which meant it could audit a
            # missed fact but could never recover it.  Build one complete
            # source inventory and one Schema-binding pass per frozen context,
            # then intersect the result with frozen entity occurrences.
            if source_first_enabled:
                for source_index, chunk in enumerate(
                    relation_chunks, start=1,
                ):
                    context_key = str(
                        chunk.get("shadow_context_id") or chunk["id"]
                    )
                    context_opportunities = list(
                        plans_by_context[context_key].opportunities
                    )
                    content = str(chunk.get("content") or "")
                    grounded_open_facts = bool(
                        Config.relation_grounded_open_facts_enabled
                        and len(mentions_by_context[context_key]) >= 2
                    )
                    if not context_opportunities and not grounded_open_facts:
                        if progress:
                            progress(
                                (2 * total_chunks) + source_index,
                                total_progress,
                                "Knowledge Forge 源事实清单 "
                                f"{source_index}/{total_chunks}（无需模型）",
                            )
                        continue
                    if grounded_open_facts:
                        inventory_prompt = v2.grounded_open_fact_prompt(
                            context_id=context_key,
                            content=content,
                            occurrences=mentions_by_context[context_key],
                            context_kind=str(
                                chunk.get("shadow_context_kind") or ""
                            ),
                        )
                        inventory_prompt_version = (
                            v2.GROUNDED_RELATION_OPEN_FACT_PROMPT_VERSION
                        )
                        inventory_schema_version = (
                            v2.GROUNDED_RELATION_OPEN_FACT_SCHEMA_VERSION
                        )
                        inventory_parser_version = (
                            v2.GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION
                        )
                        inventory_response_schema = (
                            v2.grounded_open_fact_response_schema(
                                mentions_by_context[context_key]
                            )
                        )

                        def parse_inventory(value: Any) -> dict[str, Any]:
                            return v2.parse_grounded_open_facts(
                                value,
                                content=content,
                                occurrences=mentions_by_context[context_key],
                                allow_partial=True,
                            )
                    else:
                        source_opportunity = replace(
                            context_opportunities[0],
                            context_span=(0, len(content)),
                        )
                        inventory_prompt = v2.atomic_fact_inventory_prompt(
                            source_opportunity,
                            content=content,
                        )
                        inventory_prompt_version = (
                            v2.GROUNDED_RELATION_ATOMIC_FACT_PROMPT_VERSION
                        )
                        inventory_schema_version = (
                            v2.GROUNDED_RELATION_ATOMIC_FACT_SCHEMA_VERSION
                        )
                        inventory_parser_version = (
                            v2.GROUNDED_RELATION_ATOMIC_FACT_PARSER_VERSION
                        )
                        inventory_response_schema = (
                            v2.atomic_fact_inventory_response_schema()
                        )

                        def parse_inventory(value: Any) -> dict[str, Any]:
                            return v2.parse_atomic_fact_inventory(
                                value,
                                context_text=content,
                                allow_partial=True,
                            )
                    inventory_payload = json.loads(inventory_prompt)
                    inventory = None
                    inventory_error = None
                    cache_lookup = self._reusable_model_call_lookup()
                    inventory_request_hash = relation_model_payload_hash(
                        "request", inventory_prompt,
                    )
                    cached_inventory_call = (
                        cache_lookup(
                            run_id=run_id,
                            component="source_inventory",
                            model_digest=str(context["model_revision"]),
                            prompt_version=inventory_prompt_version,
                            schema_version=inventory_schema_version,
                            parser_version=inventory_parser_version,
                            request_hash=inventory_request_hash,
                        )
                        if callable(cache_lookup) else None
                    )
                    if cached_inventory_call is not None:
                        try:
                            cached_raw_inventory = (
                                cached_inventory_call.get("parsed_response")
                                or {}
                            )
                            inventory = parse_inventory(cached_raw_inventory)
                            source_inventory_cache_hits += 1
                            self.db.record_grounded_relation_source_model_call(
                                run_id,
                                {
                                    "component": "source_inventory",
                                    "model_name": context["model"],
                                    "model_digest": context["model_revision"],
                                    "prompt_version": inventory_prompt_version,
                                    "schema_version": inventory_schema_version,
                                    "parser_version": inventory_parser_version,
                                    "idempotency_key": f"source-inventory:{context_key}",
                                    "attempt_no": 1,
                                    "status": "COMPLETED",
                                    "request_hash": inventory_request_hash,
                                    "response_hash": str(
                                        cached_inventory_call.get(
                                            "response_hash"
                                        )
                                        or ""
                                    ),
                                    "request_payload": inventory_payload,
                                    "raw_response_text": str(
                                        cached_inventory_call.get(
                                            "raw_response_text"
                                        )
                                        or _safe_json_text(
                                            cached_raw_inventory
                                        )
                                    ),
                                    "parsed_response": cached_raw_inventory,
                                    "token_usage": {
                                        "cache_hit": True,
                                        "reused_model_call_id": str(
                                            cached_inventory_call.get("id")
                                            or ""
                                        ),
                                    },
                                    "terminal_status": "COMPLETED",
                                },
                            )
                        except Exception:
                            inventory = None
                    for attempt_no in (1, 2):
                        if inventory is not None:
                            break
                        source_inventory_calls += 1
                        raw_inventory = None
                        try:
                            inventory_request = (
                                self._extract_grounded_open_facts
                                if grounded_open_facts
                                else self._inventory_atomic_facts
                            )
                            raw_inventory = inventory_request(
                                model=context["model"],
                                context_budget_tokens=context_budget,
                                prompt=inventory_prompt,
                                response_schema=inventory_response_schema,
                                attempt_no=attempt_no,
                            )
                            inventory = parse_inventory(raw_inventory)
                            diagnostics = getattr(
                                self.models, "last_call_diagnostics", None,
                            )
                            self.db.record_grounded_relation_source_model_call(
                                run_id,
                                {
                                    "component": "source_inventory",
                                    "model_name": context["model"],
                                    "model_digest": context["model_revision"],
                                    "prompt_version": inventory_prompt_version,
                                    "schema_version": inventory_schema_version,
                                    "parser_version": inventory_parser_version,
                                    "idempotency_key": f"source-inventory:{context_key}",
                                    "attempt_no": attempt_no,
                                    "status": "COMPLETED",
                                    "request_hash": relation_model_payload_hash(
                                        "request", inventory_prompt,
                                    ),
                                    "response_hash": relation_model_payload_hash(
                                        "response", getattr(
                                            self.models, "last_raw_response_text", ""
                                        ) or _safe_json_text(raw_inventory),
                                    ),
                                    "request_payload": inventory_payload,
                                    "raw_response_text": getattr(
                                        self.models, "last_raw_response_text", ""
                                    ) or _safe_json_text(raw_inventory),
                                    "parsed_response": raw_inventory,
                                    "token_usage": asdict(diagnostics) if diagnostics else {},
                                    "terminal_status": "COMPLETED",
                                },
                            )
                            break
                        except Exception as exc:
                            inventory_error = exc
                            source_inventory_failures += 1
                            diagnostics = getattr(
                                self.models, "last_call_diagnostics", None,
                            )
                            raw_text = getattr(
                                self.models, "last_raw_response_text", ""
                            ) or ""
                            self.db.record_grounded_relation_source_model_call(
                                run_id,
                                {
                                    "component": "source_inventory",
                                    "model_name": context["model"],
                                    "model_digest": context["model_revision"],
                                    "prompt_version": inventory_prompt_version,
                                    "schema_version": inventory_schema_version,
                                    "parser_version": inventory_parser_version,
                                    "idempotency_key": f"source-inventory:{context_key}",
                                    "attempt_no": attempt_no,
                                    "status": "FAILED",
                                    "request_hash": relation_model_payload_hash(
                                        "request", inventory_prompt,
                                    ),
                                    "response_hash": (
                                        relation_model_payload_hash("response", raw_text)
                                        if raw_text else ""
                                    ),
                                    "request_payload": inventory_payload,
                                    "raw_response_text": raw_text,
                                    "parsed_response": raw_inventory or {},
                                    "token_usage": asdict(diagnostics) if diagnostics else {},
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc)[:2000],
                                    "terminal_status": "FAILED" if attempt_no == 2 else "",
                                },
                            )
                    if inventory is None:
                        continue
                    omission_pass_required = bool(
                        grounded_open_facts
                        and Config.relation_open_fact_gleaning_rounds > 0
                    )
                    omission_pass_succeeded = not omission_pass_required
                    if grounded_open_facts:
                        invalidated_reference_keys = merge_reference_proxies(
                            context_key, inventory, chunk,
                        )
                        if invalidated_reference_keys:
                            inventory = {
                                **inventory,
                                "coverage_complete": False,
                                "frames": [
                                    item
                                    for item in inventory.get("frames") or ()
                                    if not invalidated_reference_keys.intersection(
                                        str(value)
                                        for value in item.get(
                                            "argument_occurrence_keys"
                                        ) or ()
                                    )
                                ],
                            }
                        for glean_round in range(
                            1,
                            Config.relation_open_fact_gleaning_rounds + 1,
                        ):
                            glean_prompt = v2.grounded_open_fact_prompt(
                                context_id=context_key,
                                content=content,
                                occurrences=mentions_by_context[context_key],
                                context_kind=str(
                                    chunk.get("shadow_context_kind") or ""
                                ),
                                already_found=inventory.get("frames") or (),
                                failed_items=inventory.get("repair_items") or (),
                                recovery_pass=True,
                            )
                            glean_payload = json.loads(glean_prompt)
                            glean_response_schema = (
                                v2.grounded_open_fact_response_schema(
                                    mentions_by_context[context_key]
                                )
                            )
                            glean_hash = relation_model_payload_hash(
                                "request", glean_prompt,
                            )
                            gleaned_inventory = None
                            cached_glean_call = (
                                cache_lookup(
                                    run_id=run_id,
                                    component="source_inventory",
                                    model_digest=str(
                                        context["model_revision"]
                                    ),
                                    prompt_version=inventory_prompt_version,
                                    schema_version=inventory_schema_version,
                                    parser_version=inventory_parser_version,
                                    request_hash=glean_hash,
                                )
                                if callable(cache_lookup) else None
                            )
                            if cached_glean_call is not None:
                                try:
                                    cached_raw_glean = (
                                        cached_glean_call.get(
                                            "parsed_response"
                                        ) or {}
                                    )
                                    gleaned_inventory = parse_inventory(
                                        cached_raw_glean
                                    )
                                    source_inventory_cache_hits += 1
                                    source_open_fact_gleaning_cache_hits += 1
                                    self.db.record_grounded_relation_source_model_call(
                                        run_id,
                                        {
                                            "component": "source_inventory",
                                            "model_name": context["model"],
                                            "model_digest": context[
                                                "model_revision"
                                            ],
                                            "prompt_version": inventory_prompt_version,
                                            "schema_version": inventory_schema_version,
                                            "parser_version": inventory_parser_version,
                                            "idempotency_key": (
                                                f"source-open-fact-glean:"
                                                f"{context_key}:{glean_round}"
                                            ),
                                            "attempt_no": 1,
                                            "status": "COMPLETED",
                                            "request_hash": glean_hash,
                                            "response_hash": str(
                                                cached_glean_call.get(
                                                    "response_hash"
                                                ) or ""
                                            ),
                                            "request_payload": glean_payload,
                                            "raw_response_text": str(
                                                cached_glean_call.get(
                                                    "raw_response_text"
                                                )
                                                or _safe_json_text(
                                                    cached_raw_glean
                                                )
                                            ),
                                            "parsed_response": cached_raw_glean,
                                            "token_usage": {
                                                "cache_hit": True,
                                                "reused_model_call_id": str(
                                                    cached_glean_call.get("id")
                                                    or ""
                                                ),
                                            },
                                            "terminal_status": "COMPLETED",
                                        },
                                    )
                                except Exception:
                                    gleaned_inventory = None
                            for attempt_no in (1, 2):
                                if gleaned_inventory is not None:
                                    break
                                source_inventory_calls += 1
                                source_open_fact_gleaning_calls += 1
                                raw_glean = None
                                try:
                                    raw_glean = self._extract_grounded_open_facts(
                                        model=context["model"],
                                        context_budget_tokens=context_budget,
                                        prompt=glean_prompt,
                                        response_schema=glean_response_schema,
                                        attempt_no=attempt_no,
                                    )
                                    gleaned_inventory = parse_inventory(
                                        raw_glean
                                    )
                                    diagnostics = getattr(
                                        self.models,
                                        "last_call_diagnostics",
                                        None,
                                    )
                                    self.db.record_grounded_relation_source_model_call(
                                        run_id,
                                        {
                                            "component": "source_inventory",
                                            "model_name": context["model"],
                                            "model_digest": context[
                                                "model_revision"
                                            ],
                                            "prompt_version": inventory_prompt_version,
                                            "schema_version": inventory_schema_version,
                                            "parser_version": inventory_parser_version,
                                            "idempotency_key": (
                                                f"source-open-fact-glean:"
                                                f"{context_key}:{glean_round}"
                                            ),
                                            "attempt_no": attempt_no,
                                            "status": "COMPLETED",
                                            "request_hash": glean_hash,
                                            "response_hash": relation_model_payload_hash(
                                                "response",
                                                getattr(
                                                    self.models,
                                                    "last_raw_response_text",
                                                    "",
                                                )
                                                or _safe_json_text(raw_glean),
                                            ),
                                            "request_payload": glean_payload,
                                            "raw_response_text": getattr(
                                                self.models,
                                                "last_raw_response_text",
                                                "",
                                            )
                                            or _safe_json_text(raw_glean),
                                            "parsed_response": raw_glean,
                                            "token_usage": (
                                                asdict(diagnostics)
                                                if diagnostics else {}
                                            ),
                                            "terminal_status": "COMPLETED",
                                        },
                                    )
                                    break
                                except Exception as exc:
                                    source_inventory_failures += 1
                                    source_open_fact_gleaning_failures += 1
                                    diagnostics = getattr(
                                        self.models,
                                        "last_call_diagnostics",
                                        None,
                                    )
                                    raw_text = getattr(
                                        self.models,
                                        "last_raw_response_text",
                                        "",
                                    ) or ""
                                    self.db.record_grounded_relation_source_model_call(
                                        run_id,
                                        {
                                            "component": "source_inventory",
                                            "model_name": context["model"],
                                            "model_digest": context[
                                                "model_revision"
                                            ],
                                            "prompt_version": inventory_prompt_version,
                                            "schema_version": inventory_schema_version,
                                            "parser_version": inventory_parser_version,
                                            "idempotency_key": (
                                                f"source-open-fact-glean:"
                                                f"{context_key}:{glean_round}"
                                            ),
                                            "attempt_no": attempt_no,
                                            "status": "FAILED",
                                            "request_hash": glean_hash,
                                            "response_hash": (
                                                relation_model_payload_hash(
                                                    "response", raw_text,
                                                )
                                                if raw_text else ""
                                            ),
                                            "request_payload": glean_payload,
                                            "raw_response_text": raw_text,
                                            "parsed_response": raw_glean or {},
                                            "token_usage": (
                                                asdict(diagnostics)
                                                if diagnostics else {}
                                            ),
                                            "error_type": type(exc).__name__,
                                            "error_message": str(exc)[:2000],
                                            "terminal_status": (
                                                "FAILED"
                                                if attempt_no == 2 else ""
                                            ),
                                        },
                                    )
                            if gleaned_inventory is None:
                                break
                            omission_pass_succeeded = True
                            invalidated_reference_keys = merge_reference_proxies(
                                context_key, gleaned_inventory, chunk,
                            )
                            if invalidated_reference_keys:
                                inventory = {
                                    **inventory,
                                    "coverage_complete": False,
                                    "frames": [
                                        item
                                        for item in inventory.get("frames") or ()
                                        if not invalidated_reference_keys.intersection(
                                            str(value)
                                            for value in item.get(
                                                "argument_occurrence_keys"
                                            ) or ()
                                        )
                                    ],
                                }
                                gleaned_inventory = {
                                    **gleaned_inventory,
                                    "coverage_complete": False,
                                    "frames": [
                                        item
                                        for item in gleaned_inventory.get(
                                            "frames"
                                        ) or ()
                                        if not invalidated_reference_keys.intersection(
                                            str(value)
                                            for value in item.get(
                                                "argument_occurrence_keys"
                                            ) or ()
                                        )
                                    ],
                                }
                            source_open_fact_gleaning_rounds_completed += 1
                            existing_frame_keys = {
                                (
                                    tuple(item.get(
                                        "argument_occurrence_keys"
                                    ) or ()),
                                    str(item.get("relation") or "").casefold(),
                                    tuple(item.get("evidence_span") or ()),
                                )
                                for item in inventory.get("frames") or ()
                                if isinstance(item, Mapping)
                            }
                            reverse_primary_frame_keys = {
                                (
                                    tuple(reversed(tuple(item.get(
                                        "argument_occurrence_keys"
                                    ) or ()))),
                                    str(item.get("relation") or "").casefold(),
                                    tuple(item.get("evidence_span") or ()),
                                )
                                for item in inventory.get("frames") or ()
                                if isinstance(item, Mapping)
                                and len(tuple(item.get(
                                    "argument_occurrence_keys"
                                ) or ())) == 2
                            }
                            suppressed_reverse_frames = [
                                item
                                for item in gleaned_inventory.get("frames") or ()
                                if isinstance(item, Mapping)
                                and (
                                    tuple(item.get(
                                        "argument_occurrence_keys"
                                    ) or ()),
                                    str(item.get("relation") or "").casefold(),
                                    tuple(item.get("evidence_span") or ()),
                                ) in reverse_primary_frame_keys
                            ]
                            source_open_fact_reverse_recoveries_suppressed += len(
                                suppressed_reverse_frames
                            )
                            new_frames = [
                                item
                                for item in gleaned_inventory.get("frames") or ()
                                if isinstance(item, Mapping)
                                and (
                                    tuple(item.get(
                                        "argument_occurrence_keys"
                                    ) or ()),
                                    str(item.get("relation") or "").casefold(),
                                    tuple(item.get("evidence_span") or ()),
                                ) not in existing_frame_keys
                                and (
                                    tuple(item.get(
                                        "argument_occurrence_keys"
                                    ) or ()),
                                    str(item.get("relation") or "").casefold(),
                                    tuple(item.get("evidence_span") or ()),
                                ) not in reverse_primary_frame_keys
                            ]
                            # The recovery request receives every accepted
                            # frame, every unresolved repair and a complete
                            # mention/scope checklist. Its parser result
                            # therefore certifies the final union, rather than
                            # inheriting an initial malformed-row flag forever.
                            # A failed or partially parsed recovery remains
                            # false and keeps the pairwise fallback active.
                            combined_coverage_complete = bool(
                                gleaned_inventory.get("coverage_complete")
                            )
                            if not new_frames:
                                inventory = {
                                    **inventory,
                                    "coverage_complete": (
                                        combined_coverage_complete
                                    ),
                                    "repair_items": list(
                                        gleaned_inventory.get(
                                            "repair_items"
                                        ) or ()
                                    ),
                                }
                                if combined_coverage_complete:
                                    break
                                continue
                            source_open_facts_recovered += len(new_frames)
                            inventory = {
                                **inventory,
                                "coverage_complete": (
                                    combined_coverage_complete
                                ),
                                "frames": [
                                    *(inventory.get("frames") or ()),
                                    *new_frames,
                                ],
                                "dropped_items": [
                                    *(inventory.get("dropped_items") or ()),
                                    *(gleaned_inventory.get(
                                        "dropped_items"
                                    ) or ()),
                                ],
                                "repair_items": [
                                    *(gleaned_inventory.get(
                                        "repair_items"
                                    ) or ()),
                                ],
                            }
                        if not omission_pass_succeeded:
                            # A failed bounded omission pass cannot certify
                            # source saturation. Keep the ordinary pairwise
                            # backstop active for this context instead of
                            # silently trading recall for fewer model calls.
                            inventory = {
                                **inventory,
                                "coverage_complete": False,
                                "omission_pass_status": "FAILED",
                            }
                        else:
                            source_inventory_audited_contexts.add(
                                context_key
                            )
                    if (
                        grounded_open_facts
                        and Config.relation_reference_adjudication_enabled
                    ):
                        reference_prompt = (
                            v2.relation_reference_resolution_prompt(
                                content=content,
                                occurrences=mentions_by_context[context_key],
                                inventory=inventory,
                            )
                        )
                        reference_payload = json.loads(reference_prompt)
                        reference_count = len(
                            reference_payload.get("references") or ()
                        )
                        source_reference_adjudication_targets += reference_count
                        if reference_count:
                            reference_schema = (
                                v2.relation_reference_resolution_response_schema(
                                    content=content,
                                    occurrences=mentions_by_context[context_key],
                                    inventory=inventory,
                                )
                            )
                            reference_hash = relation_model_payload_hash(
                                "request", reference_prompt,
                            )
                            parsed_references = None
                            cached_reference_call = (
                                cache_lookup(
                                    run_id=run_id,
                                    component="source_reference",
                                    model_digest=str(
                                        context["model_revision"]
                                    ),
                                    prompt_version=(
                                        v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION
                                    ),
                                    schema_version=(
                                        v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION
                                    ),
                                    parser_version=(
                                        v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION
                                    ),
                                    request_hash=reference_hash,
                                )
                                if callable(cache_lookup) else None
                            )
                            if cached_reference_call is not None:
                                try:
                                    cached_raw_references = (
                                        cached_reference_call.get(
                                            "parsed_response"
                                        ) or {}
                                    )
                                    parsed_references = (
                                        v2.parse_relation_reference_resolutions(
                                            cached_raw_references,
                                            content=content,
                                            occurrences=(
                                                mentions_by_context[context_key]
                                            ),
                                            inventory=inventory,
                                            allow_partial=True,
                                        )
                                    )
                                    source_reference_adjudication_cache_hits += 1
                                    self.db.record_grounded_relation_source_model_call(
                                        run_id,
                                        {
                                            "component": "source_reference",
                                            "model_name": context["model"],
                                            "model_digest": context[
                                                "model_revision"
                                            ],
                                            "prompt_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION
                                            ),
                                            "schema_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION
                                            ),
                                            "parser_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION
                                            ),
                                            "idempotency_key": (
                                                f"source-reference:{context_key}"
                                            ),
                                            "attempt_no": 1,
                                            "status": "COMPLETED",
                                            "request_hash": reference_hash,
                                            "response_hash": str(
                                                cached_reference_call.get(
                                                    "response_hash"
                                                ) or ""
                                            ),
                                            "request_payload": reference_payload,
                                            "raw_response_text": str(
                                                cached_reference_call.get(
                                                    "raw_response_text"
                                                )
                                                or _safe_json_text(
                                                    cached_raw_references
                                                )
                                            ),
                                            "parsed_response": (
                                                cached_raw_references
                                            ),
                                            "token_usage": {
                                                "cache_hit": True,
                                                "reused_model_call_id": str(
                                                    cached_reference_call.get(
                                                        "id"
                                                    ) or ""
                                                ),
                                            },
                                            "terminal_status": "COMPLETED",
                                        },
                                    )
                                except Exception:
                                    parsed_references = None
                            for attempt_no in (1, 2):
                                if parsed_references is not None:
                                    break
                                source_reference_adjudication_calls += 1
                                raw_references = None
                                try:
                                    raw_references = (
                                        self._resolve_relation_references(
                                            model=context["model"],
                                            context_budget_tokens=(
                                                context_budget
                                            ),
                                            prompt=reference_prompt,
                                            response_schema=reference_schema,
                                            attempt_no=attempt_no,
                                        )
                                    )
                                    parsed_references = (
                                        v2.parse_relation_reference_resolutions(
                                            raw_references,
                                            content=content,
                                            occurrences=(
                                                mentions_by_context[context_key]
                                            ),
                                            inventory=inventory,
                                            allow_partial=True,
                                        )
                                    )
                                    diagnostics = getattr(
                                        self.models,
                                        "last_call_diagnostics",
                                        None,
                                    )
                                    self.db.record_grounded_relation_source_model_call(
                                        run_id,
                                        {
                                            "component": "source_reference",
                                            "model_name": context["model"],
                                            "model_digest": context[
                                                "model_revision"
                                            ],
                                            "prompt_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION
                                            ),
                                            "schema_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION
                                            ),
                                            "parser_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION
                                            ),
                                            "idempotency_key": (
                                                f"source-reference:{context_key}"
                                            ),
                                            "attempt_no": attempt_no,
                                            "status": "COMPLETED",
                                            "request_hash": reference_hash,
                                            "response_hash": (
                                                relation_model_payload_hash(
                                                    "response",
                                                    getattr(
                                                        self.models,
                                                        "last_raw_response_text",
                                                        "",
                                                    )
                                                    or _safe_json_text(
                                                        raw_references
                                                    ),
                                                )
                                            ),
                                            "request_payload": reference_payload,
                                            "raw_response_text": getattr(
                                                self.models,
                                                "last_raw_response_text",
                                                "",
                                            )
                                            or _safe_json_text(raw_references),
                                            "parsed_response": raw_references,
                                            "token_usage": (
                                                asdict(diagnostics)
                                                if diagnostics else {}
                                            ),
                                            "terminal_status": "COMPLETED",
                                        },
                                    )
                                    break
                                except Exception as exc:
                                    source_reference_adjudication_failures += 1
                                    diagnostics = getattr(
                                        self.models,
                                        "last_call_diagnostics",
                                        None,
                                    )
                                    raw_text = getattr(
                                        self.models,
                                        "last_raw_response_text",
                                        "",
                                    ) or ""
                                    self.db.record_grounded_relation_source_model_call(
                                        run_id,
                                        {
                                            "component": "source_reference",
                                            "model_name": context["model"],
                                            "model_digest": context[
                                                "model_revision"
                                            ],
                                            "prompt_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PROMPT_VERSION
                                            ),
                                            "schema_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_SCHEMA_VERSION
                                            ),
                                            "parser_version": (
                                                v2.GROUNDED_RELATION_REFERENCE_RESOLUTION_PARSER_VERSION
                                            ),
                                            "idempotency_key": (
                                                f"source-reference:{context_key}"
                                            ),
                                            "attempt_no": attempt_no,
                                            "status": "FAILED",
                                            "request_hash": reference_hash,
                                            "response_hash": (
                                                relation_model_payload_hash(
                                                    "response", raw_text,
                                                )
                                                if raw_text else ""
                                            ),
                                            "request_payload": reference_payload,
                                            "raw_response_text": raw_text,
                                            "parsed_response": (
                                                raw_references or {}
                                            ),
                                            "token_usage": (
                                                asdict(diagnostics)
                                                if diagnostics else {}
                                            ),
                                            "error_type": type(exc).__name__,
                                            "error_message": str(exc)[:2000],
                                            "terminal_status": (
                                                "FAILED"
                                                if attempt_no == 2 else ""
                                            ),
                                        },
                                    )
                            if parsed_references is not None:
                                incoming_reference_keys = {
                                    str(item.get("occurrence_key") or "")
                                    for item in parsed_references.get(
                                        "reference_proxies"
                                    ) or ()
                                    if str(item.get("occurrence_key") or "")
                                }
                                invalidated_reference_keys = (
                                    merge_reference_proxies(
                                        context_key,
                                        {
                                            **parsed_references,
                                            "dropped_references": list(
                                                parsed_references.get(
                                                    "dropped_resolutions"
                                                ) or ()
                                            ),
                                        },
                                        chunk,
                                    )
                                )
                                active_reference_keys = {
                                    str(item.get("occurrence_key") or "")
                                    for item in mentions_by_context[context_key]
                                    if bool(
                                        item.get(
                                            "relation_reference_proxy"
                                        )
                                    )
                                }
                                source_reference_adjudication_proxies_accepted += len(
                                    incoming_reference_keys
                                    & active_reference_keys
                                )
                                if invalidated_reference_keys:
                                    inventory = {
                                        **inventory,
                                        "coverage_complete": False,
                                        "frames": [
                                            item
                                            for item in inventory.get(
                                                "frames"
                                            ) or ()
                                            if not invalidated_reference_keys.intersection(
                                                str(value)
                                                for value in item.get(
                                                    "argument_occurrence_keys"
                                                ) or ()
                                            )
                                        ],
                                    }
                    if grounded_open_facts:
                        inventory = v2.augment_grounded_open_fact_inventory(
                            inventory,
                            content=content,
                            occurrences=mentions_by_context[context_key],
                        )
                        excluded_endpoint_keys = {
                            str(value)
                            for value in inventory.get(
                                "relation_endpoint_excluded_occurrence_keys"
                            ) or ()
                            if str(value)
                        }
                        if excluded_endpoint_keys:
                            for planned in tuple(
                                plans_by_context[context_key].opportunities
                            ):
                                filtered = _apply_parser_endpoint_exclusions(
                                    planned, excluded_endpoint_keys,
                                )
                                if filtered is planned:
                                    continue
                                source_parser_endpoint_pairs_filtered += 1
                                replace_opportunity(filtered)
                        source_deterministic_frames_recovered += int(
                            inventory.get("deterministic_frames_added") or 0
                        )
                        source_deterministic_frames_by_origin.update(
                            {
                                str(key): int(value)
                                for key, value in (
                                    inventory.get(
                                        "deterministic_frames_by_origin"
                                    ) or {}
                                ).items()
                            }
                        )
                        if bool(inventory.get("coverage_complete")):
                            source_inventory_complete_contexts.add(
                                context_key
                            )
                        for repair in inventory.get("repair_items") or ():
                            if not isinstance(repair, Mapping):
                                continue
                            repair_keys = [
                                str(value or "")
                                for value in repair.get(
                                    "known_argument_occurrence_keys"
                                ) or ()
                            ]
                            if (
                                len(repair_keys) == 2
                                and all(repair_keys)
                                and repair_keys[0] != repair_keys[1]
                            ):
                                source_repair_pair_ids.add(v2.pair_identity(
                                    str(chunk["revision_id"]),
                                    repair_keys[0],
                                    repair_keys[1],
                                ))
                    if grounded_open_facts:
                        for frame in inventory.get("frames") or ():
                            if not isinstance(frame, Mapping):
                                continue
                            opportunity = self._grounded_open_fact_opportunity(
                                run_id=run_id,
                                project_id=project_id,
                                chunk=chunk,
                                frame=frame,
                                mentions=mentions_by_context[context_key],
                                logical_schema=context["schema"],
                            )
                            if opportunity is None:
                                continue
                            existing = opportunity_by_pair.get(
                                str(opportunity.pair_identity)
                            )
                            if existing is None:
                                add_direct_opportunity(
                                    context_key, opportunity,
                                )
                                continue
                            existing_sources = (
                                existing.provenance.get("candidate_sources")
                                if isinstance(
                                    existing.provenance.get(
                                        "candidate_sources"
                                    ),
                                    list,
                                ) else []
                            )
                            new_sources = (
                                opportunity.provenance.get(
                                    "candidate_sources"
                                )
                                if isinstance(
                                    opportunity.provenance.get(
                                        "candidate_sources"
                                    ),
                                    list,
                                ) else []
                            )
                            opportunity = replace(
                                opportunity,
                                # Multiple source frames may support the same
                                # immutable occurrence pair. Never let a later
                                # narrower frame erase an earlier table header,
                                # section label, or clause range needed by one
                                # of those hypotheses.
                                context_span=(
                                    min(
                                        int(existing.context_span[0]),
                                        int(opportunity.context_span[0]),
                                    ),
                                    max(
                                        int(existing.context_span[1]),
                                        int(opportunity.context_span[1]),
                                    ),
                                ),
                                provenance={
                                    **existing.provenance,
                                    **opportunity.provenance,
                                    "candidate_sources": sorted({
                                        *existing_sources,
                                        *new_sources,
                                    }),
                                },
                            )
                            replace_opportunity(opportunity)
                    if not bool(inventory.get("coverage_complete")):
                        source_inventory_incomplete += 1
                    structural_bindings = (
                        v2.deterministic_source_structure_schema_bindings(
                            inventory,
                            logical_schema=context["schema"],
                            include_review_only=(
                                source_binding_review_candidates_enabled
                            ),
                        )
                    )
                    source_structure_bindings_total += len(
                        structural_bindings.get("bindings") or ()
                    )
                    binding_prompt = v2.source_frame_schema_binding_prompt(
                        inventory,
                        logical_schema=context["schema"],
                        already_found_bindings=(
                            structural_bindings.get("bindings") or ()
                        ),
                        include_review_only=(
                            source_binding_review_candidates_enabled
                        ),
                    )
                    binding_payload = json.loads(binding_prompt)
                    model_binding_required = bool(
                        binding_payload.get("frames")
                        and binding_payload.get("predicates")
                    )
                    bindings = (
                        None
                        if model_binding_required
                        else {
                            "coverage_complete": True,
                            "bindings": [],
                            "dropped_bindings": [],
                            "parser_version": (
                                v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION
                            ),
                        }
                    )
                    if not model_binding_required:
                        source_binding_model_skipped_empty_contract += 1
                    binding_request_hash = relation_model_payload_hash(
                        "request", binding_prompt,
                    )
                    cached_binding_call = (
                        cache_lookup(
                            run_id=run_id,
                            component="source_binding",
                            model_digest=str(context["model_revision"]),
                            prompt_version=(
                                v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION
                            ),
                            schema_version=(
                                v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION
                            ),
                            parser_version=(
                                v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION
                            ),
                            request_hash=binding_request_hash,
                        )
                        if model_binding_required and callable(cache_lookup)
                        else None
                    )
                    if cached_binding_call is not None:
                        try:
                            cached_raw_bindings = (
                                cached_binding_call.get("parsed_response")
                                or {}
                            )
                            bindings = v2.parse_source_frame_schema_bindings(
                                cached_raw_bindings,
                                inventory=inventory,
                                logical_schema=context["schema"],
                                allow_partial=True,
                                include_review_only=(
                                    source_binding_review_candidates_enabled
                                ),
                            )
                            source_binding_cache_hits += 1
                            if not bool(bindings.get("coverage_complete")):
                                source_binding_incomplete += 1
                            source_binding_items_dropped += len(
                                bindings.get("dropped_bindings") or ()
                            )
                            self.db.record_grounded_relation_source_model_call(
                                run_id,
                                {
                                    "component": "source_binding",
                                    "model_name": context["model"],
                                    "model_digest": context["model_revision"],
                                    "prompt_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
                                    "schema_version": v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
                                    "parser_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
                                    "idempotency_key": f"source-binding:{context_key}",
                                    "attempt_no": 1,
                                    "status": "COMPLETED",
                                    "request_hash": binding_request_hash,
                                    "response_hash": str(
                                        cached_binding_call.get("response_hash")
                                        or ""
                                    ),
                                    "request_payload": binding_payload,
                                    "raw_response_text": str(
                                        cached_binding_call.get(
                                            "raw_response_text"
                                        )
                                        or _safe_json_text(cached_raw_bindings)
                                    ),
                                    "parsed_response": cached_raw_bindings,
                                    "token_usage": {
                                        "cache_hit": True,
                                        "reused_model_call_id": str(
                                            cached_binding_call.get("id") or ""
                                        ),
                                    },
                                    "terminal_status": "COMPLETED",
                                },
                            )
                        except Exception:
                            bindings = None
                    for attempt_no in (1, 2):
                        if bindings is not None:
                            break
                        source_binding_calls += 1
                        raw_bindings = None
                        try:
                            raw_bindings = self._bind_source_frames_schema(
                                model=context["model"],
                                context_budget_tokens=context_budget,
                                prompt=binding_prompt,
                                response_schema=(
                                    v2.source_frame_schema_binding_response_schema(
                                        context["schema"],
                                        include_review_only=(
                                            source_binding_review_candidates_enabled
                                        ),
                                    )
                                ),
                                attempt_no=attempt_no,
                            )
                            bindings = v2.parse_source_frame_schema_bindings(
                                raw_bindings,
                                inventory=inventory,
                                logical_schema=context["schema"],
                                allow_partial=True,
                                include_review_only=(
                                    source_binding_review_candidates_enabled
                                ),
                            )
                            if not bool(bindings.get("coverage_complete")):
                                source_binding_incomplete += 1
                            source_binding_items_dropped += len(
                                bindings.get("dropped_bindings") or ()
                            )
                            diagnostics = getattr(
                                self.models, "last_call_diagnostics", None,
                            )
                            self.db.record_grounded_relation_source_model_call(
                                run_id,
                                {
                                    "component": "source_binding",
                                    "model_name": context["model"],
                                    "model_digest": context["model_revision"],
                                    "prompt_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
                                    "schema_version": v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
                                    "parser_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
                                    "idempotency_key": f"source-binding:{context_key}",
                                    "attempt_no": attempt_no,
                                    "status": "COMPLETED",
                                    "request_hash": relation_model_payload_hash(
                                        "request", binding_prompt,
                                    ),
                                    "response_hash": relation_model_payload_hash(
                                        "response", getattr(
                                            self.models, "last_raw_response_text", ""
                                        ) or _safe_json_text(raw_bindings),
                                    ),
                                    "request_payload": binding_payload,
                                    "raw_response_text": getattr(
                                        self.models, "last_raw_response_text", ""
                                    ) or _safe_json_text(raw_bindings),
                                    "parsed_response": raw_bindings,
                                    "token_usage": asdict(diagnostics) if diagnostics else {},
                                    "terminal_status": "COMPLETED",
                                },
                            )
                            break
                        except Exception as exc:
                            source_binding_failures += 1
                            diagnostics = getattr(
                                self.models, "last_call_diagnostics", None,
                            )
                            raw_text = getattr(
                                self.models, "last_raw_response_text", ""
                            ) or ""
                            self.db.record_grounded_relation_source_model_call(
                                run_id,
                                {
                                    "component": "source_binding",
                                    "model_name": context["model"],
                                    "model_digest": context["model_revision"],
                                    "prompt_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
                                    "schema_version": v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
                                    "parser_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
                                    "idempotency_key": f"source-binding:{context_key}",
                                    "attempt_no": attempt_no,
                                    "status": "FAILED",
                                    "request_hash": relation_model_payload_hash(
                                        "request", binding_prompt,
                                    ),
                                    "response_hash": (
                                        relation_model_payload_hash("response", raw_text)
                                        if raw_text else ""
                                    ),
                                    "request_payload": binding_payload,
                                    "raw_response_text": raw_text,
                                    "parsed_response": raw_bindings or {},
                                    "token_usage": asdict(diagnostics) if diagnostics else {},
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc)[:2000],
                                    "terminal_status": "FAILED" if attempt_no == 2 else "",
                                },
                            )
                    structural_rows = list(
                        structural_bindings.get("bindings") or ()
                    )
                    if bindings is None and structural_rows:
                        bindings = {
                            **structural_bindings,
                            "coverage_complete": False,
                        }
                    elif bindings is not None and structural_rows:
                        structural_keys = {
                            (
                                int(item["frame_index"]),
                                str(item["predicate"]),
                                int(item["subject_index"]),
                                int(item["object_index"]),
                            )
                            for item in structural_rows
                        }
                        bindings = {
                            **bindings,
                            "bindings": [
                                *structural_rows,
                                *(item for item in bindings.get("bindings") or ()
                                  if (
                                      int(item["frame_index"]),
                                      str(item["predicate"]),
                                      int(item["subject_index"]),
                                      int(item["object_index"]),
                                  ) not in structural_keys)
                            ],
                        }
                    if bindings is None:
                        continue
                    for glean_round in range(
                        1,
                        (
                            Config.relation_source_binding_gleaning_rounds
                            if model_binding_required else 0
                        ) + 1,
                    ):
                        glean_prompt = v2.source_frame_schema_binding_prompt(
                            inventory,
                            logical_schema=context["schema"],
                            already_found_bindings=(
                                bindings.get("bindings") or ()
                            ),
                            include_review_only=(
                                source_binding_review_candidates_enabled
                            ),
                        )
                        glean_payload = json.loads(glean_prompt)
                        gleaned = None
                        glean_request_hash = relation_model_payload_hash(
                            "request", glean_prompt,
                        )
                        cached_glean_call = (
                            cache_lookup(
                                run_id=run_id,
                                component="source_binding",
                                model_digest=str(context["model_revision"]),
                                prompt_version=(
                                    v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION
                                ),
                                schema_version=(
                                    v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION
                                ),
                                parser_version=(
                                    v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION
                                ),
                                request_hash=glean_request_hash,
                            )
                            if callable(cache_lookup) else None
                        )
                        if cached_glean_call is not None:
                            try:
                                cached_raw_gleaned = (
                                    cached_glean_call.get("parsed_response")
                                    or {}
                                )
                                gleaned = v2.parse_source_frame_schema_bindings(
                                    cached_raw_gleaned,
                                    inventory=inventory,
                                    logical_schema=context["schema"],
                                    allow_partial=True,
                                    include_review_only=(
                                        source_binding_review_candidates_enabled
                                    ),
                                )
                                source_binding_cache_hits += 1
                                source_binding_gleaning_cache_hits += 1
                                source_binding_items_dropped += len(
                                    gleaned.get("dropped_bindings") or ()
                                )
                                self.db.record_grounded_relation_source_model_call(
                                    run_id,
                                    {
                                        "component": "source_binding",
                                        "model_name": context["model"],
                                        "model_digest": context["model_revision"],
                                        "prompt_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
                                        "schema_version": v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
                                        "parser_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
                                        "idempotency_key": (
                                            f"source-binding-glean:{context_key}:"
                                            f"{glean_round}"
                                        ),
                                        "attempt_no": 1,
                                        "status": "COMPLETED",
                                        "request_hash": glean_request_hash,
                                        "response_hash": str(
                                            cached_glean_call.get("response_hash")
                                            or ""
                                        ),
                                        "request_payload": glean_payload,
                                        "raw_response_text": str(
                                            cached_glean_call.get(
                                                "raw_response_text"
                                            )
                                            or _safe_json_text(
                                                cached_raw_gleaned
                                            )
                                        ),
                                        "parsed_response": cached_raw_gleaned,
                                        "token_usage": {
                                            "cache_hit": True,
                                            "reused_model_call_id": str(
                                                cached_glean_call.get("id")
                                                or ""
                                            ),
                                        },
                                        "terminal_status": "COMPLETED",
                                    },
                                )
                                source_binding_gleaning_rounds_completed += 1
                            except Exception:
                                gleaned = None
                        for attempt_no in (1, 2):
                            if gleaned is not None:
                                break
                            source_binding_calls += 1
                            source_binding_gleaning_calls += 1
                            raw_gleaned = None
                            try:
                                raw_gleaned = self._bind_source_frames_schema(
                                    model=context["model"],
                                    context_budget_tokens=context_budget,
                                    prompt=glean_prompt,
                                    response_schema=(
                                        v2.source_frame_schema_binding_response_schema(
                                            context["schema"],
                                            include_review_only=(
                                                source_binding_review_candidates_enabled
                                            ),
                                        )
                                    ),
                                    attempt_no=attempt_no,
                                )
                                gleaned = v2.parse_source_frame_schema_bindings(
                                    raw_gleaned,
                                    inventory=inventory,
                                    logical_schema=context["schema"],
                                    allow_partial=True,
                                    include_review_only=(
                                        source_binding_review_candidates_enabled
                                    ),
                                )
                                source_binding_items_dropped += len(
                                    gleaned.get("dropped_bindings") or ()
                                )
                                diagnostics = getattr(
                                    self.models, "last_call_diagnostics", None,
                                )
                                self.db.record_grounded_relation_source_model_call(
                                    run_id,
                                    {
                                        "component": "source_binding",
                                        "model_name": context["model"],
                                        "model_digest": context["model_revision"],
                                        "prompt_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
                                        "schema_version": v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
                                        "parser_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
                                        "idempotency_key": (
                                            f"source-binding-glean:{context_key}:"
                                            f"{glean_round}"
                                        ),
                                        "attempt_no": attempt_no,
                                        "status": "COMPLETED",
                                        "request_hash": relation_model_payload_hash(
                                            "request", glean_prompt,
                                        ),
                                        "response_hash": relation_model_payload_hash(
                                            "response",
                                            getattr(
                                                self.models,
                                                "last_raw_response_text",
                                                "",
                                            )
                                            or _safe_json_text(raw_gleaned),
                                        ),
                                        "request_payload": glean_payload,
                                        "raw_response_text": getattr(
                                            self.models,
                                            "last_raw_response_text",
                                            "",
                                        )
                                        or _safe_json_text(raw_gleaned),
                                        "parsed_response": raw_gleaned,
                                        "token_usage": (
                                            asdict(diagnostics)
                                            if diagnostics else {}
                                        ),
                                        "terminal_status": "COMPLETED",
                                    },
                                )
                                source_binding_gleaning_rounds_completed += 1
                                break
                            except Exception as exc:
                                source_binding_failures += 1
                                source_binding_gleaning_failures += 1
                                diagnostics = getattr(
                                    self.models, "last_call_diagnostics", None,
                                )
                                raw_text = getattr(
                                    self.models,
                                    "last_raw_response_text",
                                    "",
                                ) or ""
                                self.db.record_grounded_relation_source_model_call(
                                    run_id,
                                    {
                                        "component": "source_binding",
                                        "model_name": context["model"],
                                        "model_digest": context["model_revision"],
                                        "prompt_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PROMPT_VERSION,
                                        "schema_version": v2.GROUNDED_RELATION_SOURCE_BINDING_SCHEMA_VERSION,
                                        "parser_version": v2.GROUNDED_RELATION_SOURCE_BINDING_PARSER_VERSION,
                                        "idempotency_key": (
                                            f"source-binding-glean:{context_key}:"
                                            f"{glean_round}"
                                        ),
                                        "attempt_no": attempt_no,
                                        "status": "FAILED",
                                        "request_hash": relation_model_payload_hash(
                                            "request", glean_prompt,
                                        ),
                                        "response_hash": (
                                            relation_model_payload_hash(
                                                "response", raw_text,
                                            )
                                            if raw_text else ""
                                        ),
                                        "request_payload": glean_payload,
                                        "raw_response_text": raw_text,
                                        "parsed_response": raw_gleaned or {},
                                        "token_usage": (
                                            asdict(diagnostics)
                                            if diagnostics else {}
                                        ),
                                        "error_type": type(exc).__name__,
                                        "error_message": str(exc)[:2000],
                                        "terminal_status": (
                                            "FAILED" if attempt_no == 2 else ""
                                        ),
                                    },
                                )
                        if gleaned is None:
                            break
                        existing_keys = {
                            (
                                int(item["frame_index"]),
                                str(item["predicate"]),
                                int(item["subject_index"]),
                                int(item["object_index"]),
                            )
                            for item in bindings.get("bindings") or ()
                        }
                        new_bindings = [
                            item
                            for item in gleaned.get("bindings") or ()
                            if (
                                int(item["frame_index"]),
                                str(item["predicate"]),
                                int(item["subject_index"]),
                                int(item["object_index"]),
                            ) not in existing_keys
                        ]
                        if not new_bindings:
                            break
                        source_bindings_recovered += len(new_bindings)
                        bindings = {
                            **bindings,
                            "coverage_complete": bool(
                                bindings.get("coverage_complete")
                                and gleaned.get("coverage_complete")
                            ),
                            "bindings": [
                                *(bindings.get("bindings") or ()),
                                *new_bindings,
                            ],
                            "dropped_bindings": [
                                *(bindings.get("dropped_bindings") or ()),
                                *(gleaned.get("dropped_bindings") or ()),
                            ],
                        }
                    shortlist = v2.source_frame_schema_recall_shortlist(
                        inventory,
                        bindings.get("bindings") or (),
                        occurrences=mentions_by_context[context_key],
                        logical_schema=context["schema"],
                        max_additions_per_frame=(
                            Config.relation_source_binding_recall_alternatives
                        ),
                        minimum_score=(
                            Config.relation_source_binding_recall_min_score
                        ),
                        include_review_only=(
                            source_binding_review_candidates_enabled
                        ),
                    )
                    source_recall_shortlist_frames_considered += int(
                        shortlist.get("frames_considered") or 0
                    )
                    source_recall_shortlist_candidates_considered += int(
                        shortlist.get("candidates_considered") or 0
                    )
                    source_recall_shortlist_candidates_below_threshold += int(
                        shortlist.get("candidates_below_threshold") or 0
                    )
                    existing_keys = {
                        (
                            int(item["frame_index"]),
                            str(item["predicate"]),
                            int(item["subject_index"]),
                            int(item["object_index"]),
                        )
                        for item in bindings.get("bindings") or ()
                    }
                    shortlist_bindings = [
                        item
                        for item in shortlist.get("bindings") or ()
                        if (
                            int(item["frame_index"]),
                            str(item["predicate"]),
                            int(item["subject_index"]),
                            int(item["object_index"]),
                        ) not in existing_keys
                    ]
                    if shortlist_bindings:
                        source_recall_shortlist_bindings += len(
                            shortlist_bindings
                        )
                        bindings = {
                            **bindings,
                            "bindings": [
                                *(bindings.get("bindings") or ()),
                                *shortlist_bindings,
                            ],
                        }
                    open_carrier = v2.source_frame_open_carrier_bindings(
                        inventory,
                        bindings.get("bindings") or (),
                        logical_schema=context["schema"],
                        max_bindings=(
                            Config.relation_open_carrier_max_per_context
                        ),
                    )
                    carrier_bindings = list(
                        open_carrier.get("bindings") or ()
                    )
                    source_open_carrier_bindings += len(carrier_bindings)
                    source_open_carrier_contexts += int(bool(carrier_bindings))
                    source_open_carrier_eligible_frames += int(
                        open_carrier.get("eligible_frames") or 0
                    )
                    source_open_carrier_skipped_precise += int(
                        open_carrier.get("skipped_precise_binding") or 0
                    )
                    source_open_carrier_parallel_precise += int(
                        open_carrier.get("parallel_precise_binding") or 0
                    )
                    source_open_carrier_skipped_low_information += int(
                        open_carrier.get("skipped_low_information") or 0
                    )
                    source_open_carrier_bindings_capped += int(
                        open_carrier.get("bindings_capped") or 0
                    )
                    if carrier_bindings:
                        bindings = {
                            **bindings,
                            "bindings": [
                                *(bindings.get("bindings") or ()),
                                *carrier_bindings,
                            ],
                        }
                    source_bindings_total += len(bindings.get("bindings") or ())
                    source_projection_hypotheses.extend(
                        v2.source_frame_projection_hypotheses(
                            inventory,
                            bindings,
                            occurrences=mentions_by_context[context_key],
                            opportunities_by_pair=opportunity_by_pair,
                            logical_schema=context["schema"],
                            classifier_model_revision=context["model_revision"],
                        )
                    )
                    if progress:
                        progress(
                            (2 * total_chunks) + source_index,
                            total_progress,
                            "Knowledge Forge 源事实清单 "
                            f"{source_index}/{total_chunks}",
                        )
            source_projection_pair_ids = {
                str(item.pair_identity)
                for item in source_projection_hypotheses
            }
            # Once the predicate-open inventory has received a successful
            # bounded omission audit, do not reclassify every remaining
            # co-occurring entity pair merely because the model also emitted
            # one quarantined bad row. Route known-endpoint repair rows to
            # their exact pair and preserve parser structure / source-copied
            # Schema cues as independent backstops. Unknown-endpoint rows are
            # entity-recall diagnostics; an all-pairs relation pass cannot
            # create the missing endpoint. Every skipped pair remains explicit
            # in the immutable ledger and is not mislabeled NO_RELATION.
            for context_key in sorted(source_inventory_audited_contexts):
                plan = plans_by_context.get(context_key)
                if plan is None:
                    continue
                for opportunity in tuple(plan.opportunities):
                    if (
                        opportunity.status != v2.PENDING_CLASSIFICATION
                        or str(opportunity.pair_identity)
                        in source_projection_pair_ids
                    ):
                        continue
                    is_source_repair = (
                        str(opportunity.pair_identity)
                        in source_repair_pair_ids
                    )
                    if is_source_repair:
                        opportunity = replace(
                            opportunity,
                            provenance={
                                **opportunity.provenance,
                                "source_repair_backstop": True,
                            },
                        )
                        replace_opportunity(opportunity)
                    if _source_first_pair_backstop_required(opportunity):
                        source_first_classifier_backstop_pairs += 1
                        if is_source_repair:
                            source_repair_classifier_backstop_pairs += 1
                        continue
                    source_first_classifier_pairs_skipped += 1
                    replace_opportunity(replace(
                        opportunity,
                        status=v2.SOURCE_FIRST_BACKSTOP_SKIPPED,
                        planning_reason=(
                            "source_inventory_audited_without_independent_"
                            "pair_backstop"
                        ),
                        provenance={
                            **opportunity.provenance,
                            "planner_status_before_source_first_skip": (
                                opportunity.status
                            ),
                            "source_inventory_audited": True,
                            "source_inventory_complete": (
                                context_key
                                in source_inventory_complete_contexts
                            ),
                        },
                    ))
            # Persisting the immutable candidate ledger performs provenance and
            # identity checks for every candidate source.  On a dense corpus it
            # can outlast an individual model request, so keep it in bounded,
            # committed batches.  A partially persisted run remains
            # non-publishable and is safely replaced on a retry; committing
            # batches gives the user durable progress instead of a long silent
            # transaction.
            opportunity_total = len(all_opportunities)
            opportunity_batch_size = Config.relation_ledger_persist_batch_size
            if not opportunity_total:
                if progress:
                    progress(
                        4 * total_chunks,
                        total_progress,
                        "Knowledge Forge 候选关系审计账本 0/0（无需写入）",
                    )
            else:
                for start in range(
                    0, opportunity_total, opportunity_batch_size,
                ):
                    end = min(
                        opportunity_total,
                        start + opportunity_batch_size,
                    )
                    self.db.persist_grounded_relation_opportunities(
                        run_id,
                        all_opportunities[start:end],
                        mark_classifying=end == opportunity_total,
                        include_persisted_rows=False,
                    )
                    if progress:
                        progress(
                            (3 * total_chunks) + math.ceil(
                                end / opportunity_total * total_chunks
                            ),
                            total_progress,
                            "Knowledge Forge 候选关系审计账本 "
                            f"{end}/{opportunity_total}",
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
                    except v2.RelationClassificationError as exc:
                        # Invalid local-subgraph proposals remain fully
                        # isolated.  A separate planner pair, if one exists,
                        # may still proceed through the classifier backstop.
                        proposer_hypothesis_rejections[str(exc)] += 1
                        continue
                    proposer_hypotheses.append(hypothesis)
            proposer_pair_ids = {
                str(item.pair_identity) for item in proposer_hypotheses
            }
            decisions: list[v2.PairClassificationDecision] = []
            classifier_calls = 0
            classifier_cache_hits = 0
            classifier_failures = 0
            classifier_preflight_splits = 0
            classifier_partial_batches = 0
            classifier_partial_decisions = 0
            classifier_partial_fallbacks = 0
            classifier_attempt_counts: Counter[str] = Counter()
            # A proposer coverage response is an observation, not an
            # independent negative classification.  Keep the metric for
            # backwards-compatible run reports, but never let it suppress the
            # classifier backstop.
            proposer_coverage_no_relation = 0

            def classify_batch_with_retries(
                *,
                chunk: dict[str, Any],
                batch: list[v2.PairOpportunity],
                batch_offset: int,
            ) -> list[v2.PairClassificationDecision]:
                nonlocal classifier_calls, classifier_failures
                nonlocal classifier_cache_hits
                nonlocal classifier_preflight_splits
                nonlocal classifier_partial_batches
                nonlocal classifier_partial_decisions
                nonlocal classifier_partial_fallbacks
                content = str(chunk.get("content") or "")
                batch_macro_context = select_macro_context(
                    context["macro_context"],
                    [
                        value
                        for pair in batch
                        for value in (
                            pair.left_canonical_id,
                            pair.right_canonical_id,
                        )
                    ],
                )
                last_error: Exception | None = None
                max_attempts = int(Config.relation_max_attempts)

                initial_output_budget = self._classifier_output_budget(
                    context_budget_tokens=context_budget,
                    attempt_no=1,
                )
                initial_prompt = v2.classifier_prompt(
                    batch,
                    content=content,
                    logical_schema=context["schema"],
                    macro_context=batch_macro_context,
                )
                initial_contract = v2.classifier_response_schema(batch)
                if len(batch) > 1 and not self._classifier_request_fits(
                    model=context["model"],
                    context_budget_tokens=context_budget,
                    max_output_tokens=initial_output_budget,
                    prompt=initial_prompt,
                    response_schema=initial_contract,
                ):
                    classifier_preflight_splits += 1
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

                for call_attempt_no in range(1, max_attempts + 1):
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
                        macro_context=batch_macro_context,
                    )
                    response_contract = v2.classifier_response_schema(batch)
                    request_payload = {
                        "stage": "classifier",
                        "model": context["model"],
                        "model_revision": context["model_revision"],
                        "system": GROUNDED_RELATION_V2_CLASSIFIER_SYSTEM_PROMPT,
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
                        "macro_context_hash": str(
                            batch_macro_context.get("hash") or ""
                        ),
                    }
                    request_hash = relation_model_payload_hash(
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
                        cache_lookup = self._reusable_model_call_lookup()
                        cached_call = (
                            cache_lookup(
                                run_id=run_id,
                                component="classifier",
                                model_digest=str(context["model_revision"]),
                                prompt_version=(
                                    v2.GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION
                                ),
                                schema_version=(
                                    v2.GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION
                                ),
                                parser_version=(
                                    v2.GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION
                                ),
                                request_hash=request_hash,
                            )
                            if callable(cache_lookup) else None
                        )
                        if cached_call is not None:
                            raw = cached_call.get("parsed_response") or {}
                            raw_response_text = str(
                                cached_call.get("raw_response_text")
                                or _safe_json_text(raw)
                            )
                            token_usage = {
                                "cache_hit": True,
                                "reused_model_call_id": str(
                                    cached_call.get("id") or ""
                                ),
                            }
                            classifier_cache_hits += 1
                        else:
                            classifier_calls += 1
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
                            raw_response_text = str(
                                getattr(
                                    self.models, "last_raw_response_text", "",
                                )
                                or _safe_json_text(raw)
                            )
                            diagnostics = getattr(
                                self.models, "last_call_diagnostics", None,
                            )
                            if diagnostics is not None:
                                token_usage = asdict(diagnostics)
                        parsed_response = raw
                        batch_decisions = list(v2.parse_classifier_response(
                            batch,
                            raw,
                            content=content,
                            logical_schema=context["schema"],
                            classifier_policy_version=
                                GROUNDED_RELATION_V2_CLASSIFIER_POLICY_VERSION,
                            classifier_model_revision=
                                context["model_revision"],
                        ))
                        response_hash = relation_model_payload_hash(
                            "response", raw_response_text,
                        )
                        self._record_classifier_attempts(
                            run_id,
                            [
                                (pair.pair_identity, {
                                    "attempt_no":
                                        attempt_numbers[str(pair.pair_identity)],
                                    "status": CLASSIFIER_ACCEPTED,
                                    "request_hash": request_hash,
                                    "response_hash": response_hash,
                                    "classifier_prompt_version":
                                        v2.GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
                                    "classifier_schema_version":
                                        v2.GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
                                    "classifier_parser_version":
                                        v2.GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
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
                                })
                                for pair in batch
                            ],
                        )
                        return batch_decisions
                    except Exception as exc:
                        last_error = exc
                        latest_raw_response_text = str(
                            getattr(self.models, "last_raw_response_text", "")
                            or ""
                        )
                        if latest_raw_response_text:
                            raw_response_text = latest_raw_response_text
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
                            relation_model_payload_hash(
                                "response", raw_response_text,
                            )
                            if raw_response_text else ""
                        )
                        partial_decisions: list[
                            v2.PairClassificationDecision
                        ] = []
                        if (
                            len(batch) > 1
                            and isinstance(exc, v2.RelationClassificationError)
                            and parsed_response
                        ):
                            try:
                                partial_decisions = list(
                                    v2.parse_classifier_response(
                                        batch,
                                        parsed_response,
                                        content=content,
                                        logical_schema=context["schema"],
                                        classifier_policy_version=(
                                            GROUNDED_RELATION_V2_CLASSIFIER_POLICY_VERSION
                                        ),
                                        classifier_model_revision=(
                                            context["model_revision"]
                                        ),
                                        allow_partial=True,
                                    )
                                )
                            except v2.RelationClassificationError:
                                partial_decisions = []
                        partial_by_pair = {
                            str(item.pair_identity): item
                            for item in partial_decisions
                        }
                        if partial_by_pair:
                            classifier_partial_batches += 1
                            classifier_partial_decisions += len(
                                partial_by_pair
                            )
                            unresolved = [
                                pair for pair in batch
                                if str(pair.pair_identity) not in partial_by_pair
                            ]
                            classifier_partial_fallbacks += len(unresolved)
                            self._record_classifier_attempts(
                                run_id,
                                [
                                    (pair.pair_identity, {
                                        **({
                                            "attempt_no": attempt_numbers[
                                                str(pair.pair_identity)
                                            ],
                                            "status": (
                                                CLASSIFIER_ACCEPTED
                                                if str(pair.pair_identity)
                                                in partial_by_pair
                                                else attempt_status
                                            ),
                                            "request_hash": request_hash,
                                            "response_hash": response_hash,
                                            "classifier_prompt_version":
                                                v2.GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
                                            "classifier_schema_version":
                                                v2.GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
                                            "classifier_parser_version":
                                                v2.GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
                                            "classifier_model_revision":
                                                context["model_revision"],
                                            "model_name": context["model"],
                                            "request_payload": request_payload,
                                            "raw_response_text": raw_response_text,
                                            "parsed_response": parsed_response,
                                            "token_usage": token_usage,
                                        }),
                                        "error_type": (
                                            "" if str(pair.pair_identity)
                                            in partial_by_pair
                                            else type(exc).__name__
                                        ),
                                        "error_message": (
                                            "" if str(pair.pair_identity)
                                            in partial_by_pair
                                            else str(exc)[:2000]
                                        ),
                                        "independent_call": True,
                                        "metadata": {
                                            "chunk_id": int(chunk["id"]),
                                            "batch_offset": batch_offset,
                                            "batch_size": len(batch),
                                            "max_output_tokens": output_budget,
                                            "partial_batch_recovery": True,
                                            "partial_member_accepted": (
                                                str(pair.pair_identity)
                                                in partial_by_pair
                                            ),
                                        },
                                    })
                                    for pair in batch
                                ],
                            )
                            if unresolved:
                                first_unresolved_index = next(
                                    index for index, pair in enumerate(batch)
                                    if str(pair.pair_identity)
                                    not in partial_by_pair
                                )
                                retried = classify_batch_with_retries(
                                    chunk=chunk,
                                    batch=unresolved,
                                    batch_offset=(
                                        batch_offset + first_unresolved_index
                                    ),
                                )
                                combined = {
                                    **partial_by_pair,
                                    **{
                                        str(item.pair_identity): item
                                        for item in retried
                                    },
                                }
                                return [
                                    combined[str(pair.pair_identity)]
                                    for pair in batch
                                ]
                            return [
                                partial_by_pair[str(pair.pair_identity)]
                                for pair in batch
                            ]
                        self._record_classifier_attempts(
                            run_id,
                            [
                                (pair.pair_identity, {
                                    "attempt_no":
                                        attempt_numbers[str(pair.pair_identity)],
                                    "status": attempt_status,
                                    "request_hash": request_hash,
                                    "response_hash": response_hash,
                                    "classifier_prompt_version":
                                        v2.GROUNDED_RELATION_CLASSIFIER_PROMPT_VERSION,
                                    "classifier_schema_version":
                                        v2.GROUNDED_RELATION_CLASSIFIER_SCHEMA_VERSION,
                                    "classifier_parser_version":
                                        v2.GROUNDED_RELATION_CLASSIFIER_PARSER_VERSION,
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
                                })
                                for pair in batch
                            ],
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
                # Local-subgraph extraction is the primary semantic entry.
                # The pair classifier is only a recall backstop for structural
                # candidates not already covered by a valid proposal.
                admitted = [
                    item for item in plans_by_context[context_key].admitted
                    if str(item.pair_identity) not in proposer_pair_ids
                    and str(item.pair_identity)
                    not in source_projection_pair_ids
                ]
                # The proposer is a high-recall proposal source, not a
                # classifier.  Even a response claiming complete coverage can
                # miss an implicit, cross-scope or differently worded fact.
                # Every admitted pair not represented by a valid proposal must
                # therefore reach the independent pair classifier.
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
                    self.db.persist_grounded_relation_classifications(
                        run_id, batch_decisions,
                    )
                    decisions.extend(batch_decisions)
                if progress:
                    progress(
                        (4 * total_chunks) + chunk_index,
                        total_progress,
                        f"Knowledge Forge 关系分类 {chunk_index}/{total_chunks}",
                    )
            classifier_hypotheses = [
                hypothesis
                for decision in decisions
                for hypothesis in decision.hypotheses
            ]
            # Parser-confirmed hierarchy is an independent source event.  It
            # must remain visible even when the classifier guessed a different
            # predicate first or rejected the pair after emitting one.  The
            # Schema still has to name exactly one compatible structural
            # predicate; validation and the dedicated structural verifier are
            # still mandatory before publication.
            structural_hypotheses: list[v2.RelationHypothesis] = []
            classifier_semantic_keys = {
                (
                    str(item.pair_identity),
                    str(item.subject_occurrence_key),
                    str(item.predicate),
                    str(item.object_occurrence_key),
                )
                for item in classifier_hypotheses
            }
            normalized_schema = v2.normalize_predicate_schema(
                context["schema"]
            )
            for opportunity in all_opportunities:
                projected = v2.structural_projection_hypotheses(
                    opportunity,
                    schema=normalized_schema,
                    classifier_policy_version=(
                        GROUNDED_RELATION_V2_CLASSIFIER_POLICY_VERSION
                    ),
                    classifier_model_revision=str(context["model_revision"]),
                )
                for item in projected:
                    semantic_key = (
                        str(item.pair_identity),
                        str(item.subject_occurrence_key),
                        str(item.predicate),
                        str(item.object_occurrence_key),
                    )
                    if semantic_key in classifier_semantic_keys:
                        continue
                    classifier_semantic_keys.add(semantic_key)
                    structural_hypotheses.append(item)
            classifier_hypothesis_ids = {
                str(item.hypothesis_id) for item in classifier_hypotheses
            }
            proposer_hypotheses_to_persist = [
                item for item in proposer_hypotheses
                if str(item.hypothesis_id) not in classifier_hypothesis_ids
            ]
            occupied_semantic_keys = {
                (
                    str(item.pair_identity),
                    str(item.subject_occurrence_key),
                    str(item.predicate),
                    str(item.object_occurrence_key),
                )
                for item in [
                    *classifier_hypotheses,
                    *proposer_hypotheses_to_persist,
                    *structural_hypotheses,
                ]
            }
            source_hypotheses_to_persist = []
            for item in source_projection_hypotheses:
                semantic_key = (
                    str(item.pair_identity),
                    str(item.subject_occurrence_key),
                    str(item.predicate),
                    str(item.object_occurrence_key),
                )
                deterministic_source_structure = bool(
                    str(
                        (item.predicate_metadata or {}).get(
                            "source_binding_source"
                        ) or ""
                    ).upper().startswith(
                        "DETERMINISTIC_SOURCE_STRUCTURE"
                    )
                )
                if (
                    semantic_key in occupied_semantic_keys
                    and not deterministic_source_structure
                ):
                    continue
                # Keep parser-frame provenance even when an equivalent
                # structural/classifier candidate already exists.  Exact fact
                # normalization collapses exact Schema duplicates, while the
                # shared frame identity lets publication compare the surviving
                # Schema projection with its source-faithful open carrier only
                # after all role gates. The terminal structural audit is
                # deterministic, so this does not add model traffic.
                occupied_semantic_keys.add(semantic_key)
                source_hypotheses_to_persist.append(item)
            if proposer_hypotheses_to_persist:
                self.db.persist_grounded_relation_proposed_hypotheses(
                    run_id,
                    proposer_hypotheses_to_persist,
                    candidate_source=v2.DIRECT_PROPOSER,
                )
            if source_hypotheses_to_persist:
                self.db.persist_grounded_relation_proposed_hypotheses(
                    run_id,
                    source_hypotheses_to_persist,
                    candidate_source="ATOMIC_FRAME_SCHEMA_PROJECTION",
                )
            if structural_hypotheses:
                self.db.persist_grounded_relation_proposed_hypotheses(
                    run_id,
                    structural_hypotheses,
                    candidate_source="STRUCTURAL_PROJECTION",
                )
            hypothesis_by_key: dict[str, v2.RelationHypothesis] = {}
            for hypothesis in [
                *classifier_hypotheses,
                *proposer_hypotheses_to_persist,
                *source_hypotheses_to_persist,
                *structural_hypotheses,
            ]:
                existing = hypothesis_by_key.get(str(hypothesis.hypothesis_id))
                if existing is not None and existing != hypothesis:
                    raise RuntimeError("relation hypothesis identity collision")
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
            self.db.persist_grounded_relation_validations(
                run_id, validations,
            )
            validation_by_hypothesis = {
                str(item.hypothesis_id): item for item in validations
            }
            verification_attempts: list[v2.VerificationAttempt] = []
            verifier_calls = 0
            role_probe_calls = 0
            role_probe_cache_hits = 0
            role_alignment_calls = 0
            role_alignment_cache_hits = 0
            atomic_fact_inventory_calls = 0
            atomic_fact_inventory_cache_hits = 0
            atomic_fact_inventory_incomplete = 0
            atomic_fact_inventory_failures = 0
            atomic_fact_inventory_cache: dict[str, dict[str, Any]] = {}
            atomic_frame_alignment_calls = 0
            atomic_frame_alignment_cache_hits = 0
            atomic_frame_alignment_failures = 0
            atomic_frame_alignment_cache: dict[str, dict[str, Any]] = {}
            role_probe_cache: dict[str, Any] = {}
            role_alignment_cache: dict[str, Any] = {}
            source_frame_fast_path_count = 0
            batch_verifier_calls = 0
            batch_verifier_cache_hits = 0
            batch_verifier_failures = 0
            batch_verifier_partial_batches = 0
            batch_verifier_partial_decisions = 0
            batch_verifier_fallbacks = 0
            batch_verifier_ungrounded_decisions = 0
            batch_verifier_structure_bypassed = 0
            batch_verifier_decisions: dict[str, dict[str, Any]] = {}
            focused_review_eligible = 0
            focused_review_selected = 0
            focused_review_mandatory = 0
            focused_review_model_calls = 0
            focused_verifier_model_calls = 0
            focused_review_confirmations = 0
            focused_review_recoveries = 0
            focused_review_batch_positive_confirmed = 0
            focused_review_batch_positive_rejected = 0
            focused_review_batch_negative_recovered = 0
            focused_review_batch_negative_retained = 0
            source_role_audit_calls = 0
            source_role_audit_claims = 0
            source_role_audit_aligned = 0
            source_role_audit_conflicts = 0
            source_role_audit_uncertain = 0
            source_role_audit_failures = 0
            source_group_audit_calls = 0
            source_group_audit_cache_hits = 0
            source_group_audit_claims = 0
            source_group_audit_aligned = 0
            source_group_audit_conflicts = 0
            source_group_audit_uncertain = 0
            source_group_audit_failures = 0
            source_group_anchor_audit_calls = 0
            source_group_anchor_audit_cache_hits = 0
            source_group_anchor_audit_claims = 0
            source_group_anchor_audit_aligned = 0
            source_group_anchor_audit_conflicts = 0
            source_group_anchor_audit_uncertain = 0
            source_group_anchor_audit_failures = 0
            source_structure_audit_calls = 0
            source_structure_audit_aligned = 0
            source_structure_audit_conflicts = 0
            source_structure_audit_uncertain = 0
            source_structure_audit_failures = 0
            deterministic_structure_contract_checks = 0
            deterministic_structure_contract_supported = 0
            deterministic_structure_contract_rejected = 0
            deterministic_structure_contract_uncertain = 0
            coordinated_source_recovery_candidates = 0
            coordinated_source_recoveries = 0
            focused_review_plans: dict[str, dict[str, Any]] = {}
            source_conflict_groups_count = 0
            source_conflict_calls = 0
            source_conflict_cache_hits = 0
            source_conflict_failures = 0
            source_conflict_none = 0
            source_conflict_suppressed = 0
            source_conflict_terminal_attempts: dict[
                str, v2.VerificationAttempt
            ] = {}
            verify_relation_hypothesis = (
                v2.verify_hypothesis_lean
                if Config.relation_verification_mode == "lean"
                else v2.verify_hypothesis
            )

            conflict_groups = v2.conflicting_source_hypothesis_groups(
                [
                    item for item in hypotheses
                    if validation_by_hypothesis[item.hypothesis_id].status
                    == v2.PASS
                ],
                logical_schema=context["schema"],
            )
            source_conflict_groups_count = len(conflict_groups)
            reusable_call_lookup = self._reusable_model_call_lookup()
            for conflict_group in conflict_groups:
                group_opportunity = opportunity_by_pair[
                    conflict_group[0].pair_identity
                ]
                group_chunk = context["chunk_by_id"][
                    int(group_opportunity.provenance["chunk_id"])
                ]
                group_content = str(group_chunk.get("content") or "")
                conflict_prompt = v2.source_predicate_conflict_prompt(
                    conflict_group,
                    group_opportunity,
                    content=group_content,
                    logical_schema=context["schema"],
                )
                conflict_schema = (
                    v2.source_predicate_conflict_response_schema(
                        conflict_group,
                        group_opportunity,
                        content=group_content,
                    )
                )
                conflict_ids = sorted(
                    item.hypothesis_id for item in conflict_group
                )
                conflict_key = hashlib.sha256(
                    "|".join(conflict_ids).encode("utf-8")
                ).hexdigest()
                request_payload = {
                    "stage": "source_conflict",
                    "model": context["model"],
                    "model_revision": context["model_revision"],
                    "prompt": conflict_prompt,
                    "schema": conflict_schema,
                    "temperature": 0.0,
                    "context_budget_tokens": context_budget,
                }
                request_hash = relation_model_payload_hash(
                    "request", request_payload,
                )
                raw_conflict: Any = None
                raw_conflict_text = ""
                conflict_token_usage: dict[str, Any] = {}
                winner: str | None = None
                try:
                    cached_conflict = (
                        reusable_call_lookup(
                            run_id=run_id,
                            component="source_conflict",
                            model_digest=str(context["model_revision"]),
                            prompt_version=(
                                v2.GROUNDED_RELATION_CONFLICT_PROMPT_VERSION
                            ),
                            schema_version=(
                                v2.GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION
                            ),
                            parser_version=(
                                v2.GROUNDED_RELATION_CONFLICT_PARSER_VERSION
                            ),
                            request_hash=request_hash,
                        )
                        if callable(reusable_call_lookup) else None
                    )
                    if cached_conflict is not None:
                        raw_conflict = (
                            cached_conflict.get("parsed_response") or {}
                        )
                        raw_conflict_text = str(
                            cached_conflict.get("raw_response_text")
                            or _safe_json_text(raw_conflict)
                        )
                        conflict_token_usage = {
                            "cache_hit": True,
                            "reused_model_call_id": str(
                                cached_conflict.get("id") or ""
                            ),
                        }
                        source_conflict_cache_hits += 1
                    else:
                        source_conflict_calls += 1
                        raw_conflict = (
                            self._adjudicate_source_predicate_conflict(
                                model=context["model"],
                                context_budget_tokens=context_budget,
                                prompt=conflict_prompt,
                                response_schema=conflict_schema,
                            )
                        )
                        raw_conflict_text = str(
                            getattr(
                                self.models, "last_raw_response_text", "",
                            )
                            or _safe_json_text(raw_conflict)
                        )
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        conflict_token_usage = (
                            asdict(diagnostics)
                            if diagnostics is not None else {}
                        )
                    winner = v2.parse_source_predicate_conflict(
                        raw_conflict,
                        hypotheses=conflict_group,
                        opportunity=group_opportunity,
                        content=group_content,
                    )
                    self.db.record_grounded_relation_source_model_call(
                        run_id,
                        {
                            "component": "source_conflict",
                            "model_name": context["model"],
                            "model_digest": context["model_revision"],
                            "prompt_version": v2.GROUNDED_RELATION_CONFLICT_PROMPT_VERSION,
                            "schema_version": v2.GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION,
                            "parser_version": v2.GROUNDED_RELATION_CONFLICT_PARSER_VERSION,
                            "idempotency_key": f"source-conflict:{conflict_key}",
                            "attempt_no": 1,
                            "status": "COMPLETED",
                            "request_hash": request_hash,
                            "response_hash": relation_model_payload_hash(
                                "response", raw_conflict_text,
                            ),
                            "request_payload": request_payload,
                            "raw_response_text": raw_conflict_text,
                            "parsed_response": raw_conflict,
                            "token_usage": conflict_token_usage,
                            "terminal_status": "COMPLETED",
                        },
                    )
                except Exception as exc:
                    winner = None
                    source_conflict_failures += 1
                    raw_conflict_text = str(
                        getattr(self.models, "last_raw_response_text", "")
                        or raw_conflict_text
                    )
                    self.db.record_grounded_relation_source_model_call(
                        run_id,
                        {
                            "component": "source_conflict",
                            "model_name": context["model"],
                            "model_digest": context["model_revision"],
                            "prompt_version": v2.GROUNDED_RELATION_CONFLICT_PROMPT_VERSION,
                            "schema_version": v2.GROUNDED_RELATION_CONFLICT_SCHEMA_VERSION,
                            "parser_version": v2.GROUNDED_RELATION_CONFLICT_PARSER_VERSION,
                            "idempotency_key": f"source-conflict:{conflict_key}",
                            "attempt_no": 1,
                            "status": "FAILED",
                            "request_hash": request_hash,
                            "response_hash": (
                                relation_model_payload_hash(
                                    "response", raw_conflict_text,
                                ) if raw_conflict_text else ""
                            ),
                            "request_payload": request_payload,
                            "raw_response_text": raw_conflict_text,
                            "parsed_response": (
                                raw_conflict
                                if isinstance(raw_conflict, Mapping) else {}
                            ),
                            "token_usage": conflict_token_usage,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc)[:2000],
                            "terminal_status": "FAILED",
                        },
                    )
                    conflict_error = f"{type(exc).__name__}: {exc}"
                else:
                    conflict_error = ""
                if winner == "":
                    source_conflict_none += 1
                for candidate in conflict_group:
                    if winner and candidate.predicate == winner:
                        continue
                    source_conflict_suppressed += 1
                    source_conflict_terminal_attempts[
                        candidate.hypothesis_id
                    ] = v2.source_predicate_conflict_terminal_attempt(
                        candidate,
                        verdict=(
                            v2.UNCERTAIN if winner is None else v2.REJECTED
                        ),
                        reason_code=(
                            "SCHEMA_CONFLICT_ADJUDICATION_FAILED"
                            if winner is None else
                            "SCHEMA_CONFLICT_NONE_ENTAILED"
                            if winner == "" else
                            "SCHEMA_CONFLICT_ALTERNATIVE_SELECTED"
                        ),
                        verifier_model_revision=context["model_revision"],
                        request_payload=request_payload,
                        raw_response_text=raw_conflict_text,
                        parsed_response=(
                            raw_conflict
                            if isinstance(raw_conflict, Mapping) else {}
                        ),
                        token_usage=conflict_token_usage,
                        error_message=conflict_error,
                    )

            def is_source_frame_hypothesis(
                value: v2.RelationHypothesis,
            ) -> bool:
                return (
                    str(
                        (value.predicate_metadata or {}).get(
                            "candidate_source"
                        )
                        or ""
                    ).upper()
                    in {
                        "ATOMIC_FRAME_SCHEMA_PROJECTION",
                        "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
                    }
                )

            # Deduplicate only verifier transport: every claim keeps its own
            # payload and result. Classifier/proposer conclusions are not
            # included in the verifier prompt, and an incomplete/malformed
            # batch falls back to the original single-claim call below.
            verifier_claims: list[
                tuple[
                    v2.RelationHypothesis,
                    v2.PairOpportunity,
                    str,
                ]
            ] = []
            for candidate in hypotheses:
                candidate_validation = validation_by_hypothesis[
                    candidate.hypothesis_id
                ]
                candidate_opportunity = opportunity_by_pair[
                    candidate.pair_identity
                ]
                candidate_chunk_id = int(
                    candidate_opportunity.provenance["chunk_id"]
                )
                candidate_content = str(
                    context["chunk_by_id"][candidate_chunk_id].get(
                        "content"
                    ) or ""
                )
                deterministic_structure = (
                    v2.parser_structure_claim_intent(
                        candidate,
                        candidate_opportunity,
                        logical_schema=context["schema"],
                    )
                )
                if (
                    candidate.hypothesis_id
                    in source_conflict_terminal_attempts
                    or
                    candidate_validation.status != v2.PASS
                    or deterministic_structure
                ):
                    if (
                        deterministic_structure
                        and candidate_validation.status == v2.PASS
                        and candidate.hypothesis_id
                        not in source_conflict_terminal_attempts
                    ):
                        batch_verifier_structure_bypassed += 1
                    continue
                verifier_claims.append((
                    candidate,
                    candidate_opportunity,
                    candidate_content,
                ))

            batch_verifier_claims_total = len(verifier_claims)
            batch_verifier_claims_processed = 0
            batch_size = int(Config.relation_verifier_batch_size)
            pending_batches = [
                verifier_claims[index:index + batch_size]
                for index in range(0, len(verifier_claims), batch_size)
            ]

            def retain_grounded_batch_decisions(
                decisions: Mapping[str, Mapping[str, Any]],
                claims: list[tuple[
                    v2.RelationHypothesis, v2.PairOpportunity, str,
                ]],
            ) -> dict[str, dict[str, Any]]:
                nonlocal batch_verifier_fallbacks
                nonlocal batch_verifier_ungrounded_decisions
                claim_by_id = {
                    claim[0].hypothesis_id: claim for claim in claims
                }
                retained: dict[str, dict[str, Any]] = {}
                for hypothesis_id, decision in decisions.items():
                    claim = claim_by_id.get(str(hypothesis_id))
                    if claim is None:
                        continue
                    hypothesis, opportunity, claim_content = claim
                    if not v2.batch_verifier_decision_is_source_grounded(
                        hypothesis,
                        opportunity,
                        decision,
                        content=claim_content,
                        logical_schema=context["schema"],
                    ):
                        batch_verifier_ungrounded_decisions += 1
                        batch_verifier_fallbacks += 1
                        continue
                    retained[str(hypothesis_id)] = dict(decision)
                return retained

            while pending_batches:
                claim_batch = pending_batches.pop(0)
                batch_prompt = v2.batch_verifier_prompt(
                    claim_batch,
                    logical_schema=context["schema"],
                )
                if (
                    len(batch_prompt)
                    > int(Config.relation_verifier_batch_max_chars)
                    and len(claim_batch) > 1
                ):
                    midpoint = max(1, len(claim_batch) // 2)
                    pending_batches[0:0] = [
                        claim_batch[:midpoint],
                        claim_batch[midpoint:],
                    ]
                    continue
                if len(batch_prompt) > int(
                    Config.relation_verifier_batch_max_chars
                ):
                    batch_verifier_fallbacks += len(claim_batch)
                    batch_verifier_claims_processed += len(claim_batch)
                    if progress:
                        progress(
                            min(total_progress, 5 * total_chunks),
                            total_progress,
                            "Knowledge Forge 关系批量终审 "
                            f"{batch_verifier_claims_processed}/"
                            f"{batch_verifier_claims_total}",
                        )
                    continue
                hypothesis_ids = [
                    item[0].hypothesis_id for item in claim_batch
                ]
                batch_schema = v2.batch_verifier_response_schema(
                    hypothesis_ids
                )
                batch_request_payload = {
                    "stage": "batch_verifier",
                    "model_revision": context["model_revision"],
                    "claim_count": len(claim_batch),
                    **self._batch_verifier_model_request(
                        model=context["model"],
                        context_budget_tokens=context_budget,
                        prompt=batch_prompt,
                        response_schema=batch_schema,
                        claim_count=len(claim_batch),
                    ),
                }
                batch_request_hash = relation_model_payload_hash(
                    "request", batch_request_payload,
                )
                batch_key = hashlib.sha256(
                    "|".join(hypothesis_ids).encode("utf-8")
                ).hexdigest()
                raw_batch: Any = None
                raw_batch_text = ""
                token_usage: dict[str, Any] = {}
                record_batch_call = getattr(
                    self.db,
                    "record_grounded_relation_source_model_call",
                    None,
                )
                try:
                    cache_lookup = self._reusable_model_call_lookup()
                    cached_batch_call = (
                        cache_lookup(
                            run_id=run_id,
                            component="batch_verifier",
                            model_digest=str(context["model_revision"]),
                            prompt_version=(
                                v2.GROUNDED_RELATION_BATCH_VERIFIER_PROMPT_VERSION
                            ),
                            schema_version=(
                                v2.GROUNDED_RELATION_BATCH_VERIFIER_SCHEMA_VERSION
                            ),
                            parser_version=(
                                v2.GROUNDED_RELATION_BATCH_VERIFIER_PARSER_VERSION
                            ),
                            request_hash=batch_request_hash,
                        )
                        if callable(cache_lookup) else None
                    )
                    if cached_batch_call is not None:
                        raw_batch = cached_batch_call.get(
                            "parsed_response"
                        ) or {}
                        batch_verifier_cache_hits += 1
                        raw_batch_text = str(
                            cached_batch_call.get("raw_response_text")
                            or _safe_json_text(raw_batch)
                        )
                        token_usage = {
                            "cache_hit": True,
                            "reused_model_call_id": str(
                                cached_batch_call.get("id") or ""
                            ),
                        }
                    else:
                        batch_verifier_calls += 1
                        verifier_calls += 1
                        raw_batch = self._verify_relation_batch(
                            model=context["model"],
                            context_budget_tokens=context_budget,
                            prompt=batch_prompt,
                            response_schema=batch_schema,
                            claim_count=len(claim_batch),
                        )
                        raw_batch_text = str(
                            getattr(
                                self.models, "last_raw_response_text", "",
                            )
                            or _safe_json_text(raw_batch)
                        )
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        token_usage = (
                            asdict(diagnostics)
                            if diagnostics is not None else {}
                        )
                    parsed_batch_decisions = (
                        v2.parse_batch_verifier_response(
                            raw_batch,
                            hypothesis_ids=hypothesis_ids,
                        )
                    )
                    parsed_batch_decisions = retain_grounded_batch_decisions(
                        parsed_batch_decisions, claim_batch,
                    )
                    if callable(record_batch_call):
                        record_batch_call(run_id, {
                            "component": "batch_verifier",
                            "model_name": context["model"],
                            "model_digest": context["model_revision"],
                            "prompt_version": v2.GROUNDED_RELATION_BATCH_VERIFIER_PROMPT_VERSION,
                            "schema_version": v2.GROUNDED_RELATION_BATCH_VERIFIER_SCHEMA_VERSION,
                            "parser_version": v2.GROUNDED_RELATION_BATCH_VERIFIER_PARSER_VERSION,
                            "idempotency_key": f"batch-verifier:{batch_key}",
                            "attempt_no": 1,
                            "status": "COMPLETED",
                            "request_hash": batch_request_hash,
                            "response_hash": relation_model_payload_hash(
                                "response", raw_batch_text,
                            ),
                            "request_payload": batch_request_payload,
                            "raw_response_text": raw_batch_text,
                            "parsed_response": raw_batch,
                            "token_usage": token_usage,
                            "terminal_status": "COMPLETED",
                        })
                    batch_verifier_decisions.update(parsed_batch_decisions)
                except Exception as batch_exc:
                    diagnostics = getattr(
                        self.models, "last_call_diagnostics", None,
                    )
                    latest_raw_batch_text = str(
                        getattr(
                            self.models, "last_raw_response_text", "",
                        )
                        or ""
                    )
                    if latest_raw_batch_text:
                        raw_batch_text = latest_raw_batch_text
                    if raw_batch is None:
                        raw_batch = getattr(
                            self.models, "last_parsed_json_response", None,
                        )
                    if raw_batch is None and raw_batch_text:
                        try:
                            raw_batch = json.loads(raw_batch_text)
                        except (TypeError, json.JSONDecodeError):
                            raw_batch = None
                    try:
                        partial = v2.parse_batch_verifier_response(
                            raw_batch,
                            hypothesis_ids=hypothesis_ids,
                            allow_partial=True,
                        )
                    except Exception:
                        partial = {}
                    partial_member_count = len(partial)
                    partial = retain_grounded_batch_decisions(
                        partial, claim_batch,
                    )
                    batch_verifier_fallbacks += (
                        len(claim_batch) - partial_member_count
                    )
                    if callable(record_batch_call):
                        batch_audit_status = (
                            "PARTIAL" if partial else "FAILED"
                        )
                        record_batch_call(run_id, {
                            "component": "batch_verifier",
                            "model_name": context["model"],
                            "model_digest": context["model_revision"],
                            "prompt_version": v2.GROUNDED_RELATION_BATCH_VERIFIER_PROMPT_VERSION,
                            "schema_version": v2.GROUNDED_RELATION_BATCH_VERIFIER_SCHEMA_VERSION,
                            "parser_version": v2.GROUNDED_RELATION_BATCH_VERIFIER_PARSER_VERSION,
                            "idempotency_key": f"batch-verifier:{batch_key}",
                            "attempt_no": 1,
                            "status": batch_audit_status,
                            "request_hash": batch_request_hash,
                            "response_hash": (
                                relation_model_payload_hash(
                                    "response", raw_batch_text,
                                ) if raw_batch_text else ""
                            ),
                            "request_payload": batch_request_payload,
                            "raw_response_text": raw_batch_text,
                            "parsed_response": raw_batch or {},
                            "token_usage": (
                                asdict(diagnostics)
                                if diagnostics is not None else {}
                            ),
                            "error_type": type(batch_exc).__name__,
                            "error_message": str(batch_exc)[:2000],
                            "terminal_status": batch_audit_status,
                        })
                    if partial:
                        batch_verifier_decisions.update(partial)
                        batch_verifier_partial_batches += 1
                        batch_verifier_partial_decisions += len(partial)
                    else:
                        if partial_member_count == 0:
                            batch_verifier_failures += 1
                batch_verifier_claims_processed += len(claim_batch)
                if progress:
                    progress(
                        min(total_progress, 5 * total_chunks),
                        total_progress,
                        "Knowledge Forge 关系批量终审 "
                        f"{batch_verifier_claims_processed}/"
                        f"{batch_verifier_claims_total}",
                    )

            focused_candidates: list[
                tuple[int, str, str, dict[str, Any]]
            ] = []
            source_competition_map = (
                v2.source_projection_competition_map(
                    hypotheses, logical_schema=context["schema"],
                )
            )
            for candidate in hypotheses:
                decision = batch_verifier_decisions.get(
                    candidate.hypothesis_id
                )
                if (
                    decision is None
                    or candidate.hypothesis_id
                    in source_conflict_terminal_attempts
                    or validation_by_hypothesis[
                        candidate.hypothesis_id
                    ].status != v2.PASS
                ):
                    continue
                review = v2.source_batch_focused_review(
                    candidate,
                    decision,
                    logical_schema=context["schema"],
                    source_competition_risk=(
                        candidate.hypothesis_id
                        in source_competition_map
                    ),
                )
                if review is None:
                    continue
                review = {
                    **review,
                    "competing_hypothesis_ids": list(
                        source_competition_map.get(
                            candidate.hypothesis_id, (),
                        )
                    ),
                }
                focused_candidates.append((
                    int(review["priority"]),
                    str(candidate.pair_identity),
                    str(candidate.hypothesis_id),
                    dict(review),
                ))
            focused_review_eligible = len(focused_candidates)
            focused_limit = int(
                Config.relation_verifier_focused_review_max
            )
            ordered_focused_candidates = sorted(
                focused_candidates,
                key=lambda item: (item[0], item[1], item[2]),
            )
            mandatory_focused_candidates = [
                item for item in ordered_focused_candidates
                if item[3].get("mandatory") is True
            ]
            optional_focused_candidates = [
                item for item in ordered_focused_candidates
                if item[3].get("mandatory") is not True
            ]
            focused_review_mandatory = len(
                mandatory_focused_candidates
            )
            remaining_focused_budget = max(
                0, focused_limit - focused_review_mandatory,
            )
            for _priority, _pair_id, hypothesis_id, review in [
                *mandatory_focused_candidates,
                *optional_focused_candidates[:remaining_focused_budget],
            ]:
                focused_review_plans[hypothesis_id] = review
            focused_review_selected = len(focused_review_plans)
            focused_review_confirmations = sum(
                1 for item in focused_review_plans.values()
                if item.get("kind") == "confirmation"
            )
            focused_review_recoveries = sum(
                1 for item in focused_review_plans.values()
                if item.get("kind") == "recovery"
            )
            total_hypotheses = len(hypotheses)
            primary_hypothesis_by_id = {
                str(item.hypothesis_id): item for item in hypotheses
            }
            grouped_role_audits: dict[str, dict[str, Any]] = {}
            coordinated_claims = [
                (
                    primary_hypothesis_by_id[hypothesis_id],
                    opportunity_by_pair[
                        primary_hypothesis_by_id[
                            hypothesis_id
                        ].pair_identity
                    ],
                )
                for hypothesis_id, plan in focused_review_plans.items()
                if (
                    plan.get("kind") == "confirmation"
                    and plan.get("source_role_audit") is True
                    and plan.get("role_audit_only") is True
                    and hypothesis_id in primary_hypothesis_by_id
                )
            ]
            coordinated_components = (
                v2.source_projection_coordinated_audit_components(
                    coordinated_claims
                )
            )
            for component in coordinated_components:
                group_claims = [
                    (
                        primary_hypothesis_by_id[hypothesis_id],
                        opportunity_by_pair[
                            primary_hypothesis_by_id[
                                hypothesis_id
                            ].pair_identity
                        ],
                    )
                    for hypothesis_id in component
                ]
                first_opportunity = group_claims[0][1]
                group_content = str(
                    context["chunk_by_id"][
                        int(first_opportunity.provenance["chunk_id"])
                    ].get("content") or ""
                )
                group_key = hashlib.sha256(
                    "|".join(component).encode("utf-8")
                ).hexdigest()

                # First challenge one narrow proposition from the shared
                # coordinated slot. Local models can otherwise make a
                # correlated all-positive decision when several members are
                # shown together. A non-aligned template is terminal for the
                # whole parser-proved expansion; an aligned template still
                # proceeds to the joint member audit below.
                anchor_hypothesis, _anchor_opportunity = (
                    v2.source_projection_group_anchor_claim(group_claims)
                )
                anchor_prompt = (
                    v2.source_projection_group_anchor_audit_prompt(
                        group_claims,
                        content=group_content,
                        logical_schema=context["schema"],
                    )
                )
                anchor_schema = (
                    v2.source_projection_group_anchor_audit_response_schema()
                )
                anchor_request: dict[str, Any] = {
                    "stage": "source_projection_group_anchor_audit",
                    "representative_hypothesis_id": (
                        anchor_hypothesis.hypothesis_id
                    ),
                    "member_hypothesis_ids": list(component),
                    "focused_reviews": {
                        hypothesis_id: dict(
                            focused_review_plans[hypothesis_id]
                        )
                        for hypothesis_id in component
                    },
                    "model": context["model"],
                    "model_revision": context["model_revision"],
                    "system": (
                        "Audit exact source roles. Return only JSON."
                    ),
                    "prompt": anchor_prompt,
                    "schema": anchor_schema,
                    "temperature": 0.0,
                    "context_budget_tokens": context_budget,
                    "max_output_tokens": 256,
                }
                anchor_request_hash = relation_model_payload_hash(
                    "request", anchor_request,
                )
                raw_anchor: Any = None
                raw_anchor_text = ""
                parsed_anchor: dict[str, Any] = {}
                anchor_token_usage: dict[str, Any] = {}
                anchor_error = ""
                anchor_result = "UNAVAILABLE"
                anchor_transport_called = False
                try:
                    cache_lookup = self._reusable_model_call_lookup()
                    cached_anchor = (
                        cache_lookup(
                            run_id=run_id,
                            component=(
                                "source_projection_group_anchor_audit"
                            ),
                            model_digest=str(context["model_revision"]),
                            prompt_version=(
                                v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PROMPT_VERSION
                            ),
                            schema_version=(
                                v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_SCHEMA_VERSION
                            ),
                            parser_version=(
                                v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PARSER_VERSION
                            ),
                            request_hash=anchor_request_hash,
                        )
                        if callable(cache_lookup) else None
                    )
                    if cached_anchor is not None:
                        raw_anchor = cached_anchor.get(
                            "parsed_response"
                        ) or {}
                        raw_anchor_text = str(
                            cached_anchor.get("raw_response_text")
                            or _safe_json_text(raw_anchor)
                        )
                        anchor_token_usage = {
                            "cache_hit": True,
                            "reused_model_call_id": str(
                                cached_anchor.get("id") or ""
                            ),
                        }
                        source_group_anchor_audit_cache_hits += 1
                    else:
                        source_group_anchor_audit_calls += 1
                        source_role_audit_calls += 1
                        focused_review_model_calls += 1
                        anchor_transport_called = True
                        raw_anchor = self.models.json_chat(
                            model=context["model"],
                            system=anchor_request["system"],
                            prompt=anchor_prompt,
                            schema=anchor_schema,
                            temperature=0.0,
                            context_budget_tokens=context_budget,
                            max_output_tokens=256,
                            safety_margin_tokens=(
                                Config.relation_classifier_safety_margin
                            ),
                        )
                        raw_anchor_text = str(
                            getattr(
                                self.models, "last_raw_response_text", "",
                            )
                            or _safe_json_text(raw_anchor)
                        )
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        anchor_token_usage = (
                            asdict(diagnostics)
                            if diagnostics is not None else {}
                        )
                    anchor_result = (
                        v2.parse_source_projection_group_anchor_audit(
                            raw_anchor
                        )
                    )
                    parsed_anchor = {
                        "projection_status": anchor_result,
                    }
                except Exception as exc:
                    source_group_anchor_audit_failures += 1
                    source_role_audit_failures += 1
                    anchor_error = f"{type(exc).__name__}: {exc}"[:2000]
                    if anchor_transport_called:
                        latest_raw = str(
                            getattr(
                                self.models, "last_raw_response_text", "",
                            ) or ""
                        )
                        if latest_raw:
                            raw_anchor_text = latest_raw
                        if raw_anchor is None:
                            candidate = getattr(
                                self.models,
                                "last_parsed_json_response",
                                None,
                            )
                            if isinstance(candidate, Mapping):
                                raw_anchor = dict(candidate)
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        if diagnostics is not None:
                            anchor_token_usage = asdict(diagnostics)
                source_role_audit_claims += 1
                source_group_anchor_audit_claims += len(component)
                if anchor_result == "ALIGNED":
                    source_group_anchor_audit_aligned += len(component)
                elif anchor_result == "CONFLICT":
                    source_group_anchor_audit_conflicts += len(component)
                elif anchor_result == "UNCERTAIN":
                    source_group_anchor_audit_uncertain += len(component)
                record_anchor_call = getattr(
                    self.db,
                    "record_grounded_relation_source_model_call",
                    None,
                )
                if callable(record_anchor_call):
                    record_anchor_call(run_id, {
                        "component": (
                            "source_projection_group_anchor_audit"
                        ),
                        "model_name": context["model"],
                        "model_digest": context["model_revision"],
                        "prompt_version": (
                            v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PROMPT_VERSION
                        ),
                        "schema_version": (
                            v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_SCHEMA_VERSION
                        ),
                        "parser_version": (
                            v2.GROUNDED_RELATION_SOURCE_GROUP_ANCHOR_AUDIT_PARSER_VERSION
                        ),
                        "idempotency_key": (
                            f"source-group-anchor-audit:{group_key}"
                        ),
                        "attempt_no": 1,
                        "status": (
                            "COMPLETED" if not anchor_error else "FAILED"
                        ),
                        "request_hash": anchor_request_hash,
                        "response_hash": (
                            relation_model_payload_hash(
                                "response", raw_anchor_text,
                            ) if raw_anchor_text else ""
                        ),
                        "request_payload": anchor_request,
                        "raw_response_text": raw_anchor_text,
                        "parsed_response": raw_anchor or {},
                        "token_usage": anchor_token_usage,
                        "error_type": (
                            anchor_error.split(":", 1)[0]
                            if anchor_error else ""
                        ),
                        "error_message": anchor_error,
                        "terminal_status": (
                            "COMPLETED" if not anchor_error else "FAILED"
                        ),
                    })
                if anchor_result != "ALIGNED":
                    anchor_response = {
                        **parsed_anchor,
                        "representative_hypothesis_id": (
                            anchor_hypothesis.hypothesis_id
                        ),
                        "member_hypothesis_ids": list(component),
                    }
                    for hypothesis_id in component:
                        grouped_role_audits[hypothesis_id] = {
                            "result": anchor_result,
                            "audit_kind": "anchor",
                            "request_payload": anchor_request,
                            "raw_response_text": raw_anchor_text,
                            "parsed_response": anchor_response,
                            "token_usage": anchor_token_usage,
                            "error_message": anchor_error,
                        }
                    continue

                group_prompt = v2.source_projection_group_audit_prompt(
                    group_claims,
                    content=group_content,
                    logical_schema=context["schema"],
                )
                group_schema = (
                    v2.source_projection_group_audit_response_schema(
                        component
                    )
                )
                group_request: dict[str, Any] = {
                    "stage": "source_projection_group_audit",
                    "focused_reviews": {
                        hypothesis_id: dict(
                            focused_review_plans[hypothesis_id]
                        )
                        for hypothesis_id in component
                    },
                    "model": context["model"],
                    "model_revision": context["model_revision"],
                    "system": (
                        "Audit exact source roles jointly. Return only JSON."
                    ),
                    "prompt": group_prompt,
                    "schema": group_schema,
                    "temperature": 0.0,
                    "context_budget_tokens": context_budget,
                    "max_output_tokens": min(
                        2048, max(384, 128 * len(component))
                    ),
                }
                request_hash = relation_model_payload_hash(
                    "request", group_request,
                )
                raw_group: Any = None
                raw_group_text = ""
                parsed_group: dict[str, str] = {}
                group_token_usage: dict[str, Any] = {}
                group_error = ""
                group_transport_called = False
                try:
                    cache_lookup = self._reusable_model_call_lookup()
                    cached_group = (
                        cache_lookup(
                            run_id=run_id,
                            component="source_projection_group_audit",
                            model_digest=str(context["model_revision"]),
                            prompt_version=(
                                v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION
                            ),
                            schema_version=(
                                v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_SCHEMA_VERSION
                            ),
                            parser_version=(
                                v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PARSER_VERSION
                            ),
                            request_hash=request_hash,
                        )
                        if callable(cache_lookup) else None
                    )
                    if cached_group is not None:
                        raw_group = cached_group.get(
                            "parsed_response"
                        ) or {}
                        raw_group_text = str(
                            cached_group.get("raw_response_text")
                            or _safe_json_text(raw_group)
                        )
                        group_token_usage = {
                            "cache_hit": True,
                            "reused_model_call_id": str(
                                cached_group.get("id") or ""
                            ),
                        }
                        source_group_audit_cache_hits += 1
                    else:
                        source_group_audit_calls += 1
                        source_role_audit_calls += 1
                        focused_review_model_calls += 1
                        group_transport_called = True
                        raw_group = self.models.json_chat(
                            model=context["model"],
                            system=group_request["system"],
                            prompt=group_prompt,
                            schema=group_schema,
                            temperature=0.0,
                            context_budget_tokens=context_budget,
                            max_output_tokens=int(
                                group_request["max_output_tokens"]
                            ),
                            safety_margin_tokens=(
                                Config.relation_classifier_safety_margin
                            ),
                        )
                        raw_group_text = str(
                            getattr(
                                self.models, "last_raw_response_text", "",
                            )
                            or _safe_json_text(raw_group)
                        )
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        group_token_usage = (
                            asdict(diagnostics)
                            if diagnostics is not None else {}
                        )
                    parsed_group = (
                        v2.parse_source_projection_group_audit(
                            raw_group, hypothesis_ids=component,
                        )
                    )
                except Exception as exc:
                    source_group_audit_failures += 1
                    source_role_audit_failures += 1
                    group_error = f"{type(exc).__name__}: {exc}"[:2000]
                    if group_transport_called:
                        latest_raw = str(
                            getattr(
                                self.models, "last_raw_response_text", "",
                            ) or ""
                        )
                        if latest_raw:
                            raw_group_text = latest_raw
                        if raw_group is None:
                            candidate = getattr(
                                self.models,
                                "last_parsed_json_response",
                                None,
                            )
                            if isinstance(candidate, Mapping):
                                raw_group = dict(candidate)
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        if diagnostics is not None:
                            group_token_usage = asdict(diagnostics)
                source_role_audit_claims += len(component)
                source_group_audit_claims += len(component)
                source_group_audit_aligned += sum(
                    value == "ALIGNED" for value in parsed_group.values()
                )
                source_group_audit_conflicts += sum(
                    value == "CONFLICT" for value in parsed_group.values()
                )
                source_group_audit_uncertain += sum(
                    value == "UNCERTAIN" for value in parsed_group.values()
                )
                record_group_call = getattr(
                    self.db,
                    "record_grounded_relation_source_model_call",
                    None,
                )
                if callable(record_group_call):
                    record_group_call(run_id, {
                        "component": "source_projection_group_audit",
                        "model_name": context["model"],
                        "model_digest": context["model_revision"],
                        "prompt_version": (
                            v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PROMPT_VERSION
                        ),
                        "schema_version": (
                            v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_SCHEMA_VERSION
                        ),
                        "parser_version": (
                            v2.GROUNDED_RELATION_SOURCE_GROUP_AUDIT_PARSER_VERSION
                        ),
                        "idempotency_key": (
                            f"source-group-audit:{group_key}"
                        ),
                        "attempt_no": 1,
                        "status": (
                            "COMPLETED" if not group_error else "FAILED"
                        ),
                        "request_hash": request_hash,
                        "response_hash": (
                            relation_model_payload_hash(
                                "response", raw_group_text,
                            ) if raw_group_text else ""
                        ),
                        "request_payload": group_request,
                        "raw_response_text": raw_group_text,
                        "parsed_response": raw_group or {},
                        "token_usage": group_token_usage,
                        "error_type": (
                            group_error.split(":", 1)[0]
                            if group_error else ""
                        ),
                        "error_message": group_error,
                        "terminal_status": (
                            "COMPLETED" if not group_error else "FAILED"
                        ),
                    })
                normalized_group = {
                    hypothesis_id: parsed_group.get(
                        hypothesis_id, "UNAVAILABLE",
                    )
                    for hypothesis_id in component
                }
                normalized_response = {
                    "statuses": [
                        {
                            "hypothesis_id": hypothesis_id,
                            "projection_status": normalized_group[
                                hypothesis_id
                            ],
                        }
                        for hypothesis_id in component
                    ],
                }
                for hypothesis_id in component:
                    grouped_role_audits[hypothesis_id] = {
                        "result": normalized_group[hypothesis_id],
                        "request_payload": group_request,
                        "raw_response_text": raw_group_text,
                        "parsed_response": normalized_response,
                        "token_usage": group_token_usage,
                        "error_message": group_error,
                    }
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
                conflict_terminal = source_conflict_terminal_attempts.get(
                    hypothesis.hypothesis_id
                )
                if conflict_terminal is not None:
                    self.db.persist_grounded_relation_verification_attempts(
                        run_id, [conflict_terminal], terminal=True,
                    )
                    verification_attempts.append(conflict_terminal)
                    if progress:
                        verified_progress = (
                            (5 * total_chunks)
                            + math.ceil(
                                hypothesis_index
                                / max(1, len(hypotheses))
                                * total_chunks
                            )
                        )
                        progress(
                            min(total_progress, verified_progress),
                            total_progress,
                            "Knowledge Forge 关系冲突复核 "
                            f"{hypothesis_index}/{len(hypotheses)} 个关系假设",
                        )
                    continue
                source_projection_candidate = is_source_frame_hypothesis(
                    hypothesis
                )
                focused_plan = focused_review_plans.get(
                    hypothesis.hypothesis_id
                )
                hypothesis_metadata = hypothesis.predicate_metadata or {}
                if v2.parser_structure_claim_intent(
                    hypothesis,
                    opportunity,
                    logical_schema=context["schema"],
                ):
                    deterministic_structure_contract_checks += 1
                    structure_terminal = (
                        v2.deterministic_parser_structure_terminal_attempt(
                            hypothesis,
                            validation,
                            opportunity,
                            content=content,
                            logical_schema=context["schema"],
                        )
                    )
                    is_source_structure = bool(
                        str(
                            hypothesis_metadata.get(
                                "source_binding_source"
                            ) or ""
                        ).upper().startswith(
                            "DETERMINISTIC_SOURCE_STRUCTURE"
                        )
                    )
                    if structure_terminal.verdict == v2.SUPPORTED:
                        deterministic_structure_contract_supported += 1
                        if is_source_structure:
                            source_structure_audit_aligned += 1
                    elif structure_terminal.verdict == v2.REJECTED:
                        deterministic_structure_contract_rejected += 1
                        if is_source_structure:
                            source_structure_audit_conflicts += 1
                    else:
                        deterministic_structure_contract_uncertain += 1
                        if is_source_structure:
                            source_structure_audit_uncertain += 1
                    self.db.persist_grounded_relation_verification_attempts(
                        run_id, [structure_terminal], terminal=True,
                    )
                    verification_attempts.append(structure_terminal)
                    if focused_plan is not None:
                        if focused_plan.get("kind") == "confirmation":
                            if structure_terminal.verdict == v2.SUPPORTED:
                                focused_review_batch_positive_confirmed += 1
                            else:
                                focused_review_batch_positive_rejected += 1
                        elif focused_plan.get("kind") == "recovery":
                            if structure_terminal.verdict == v2.SUPPORTED:
                                focused_review_batch_negative_recovered += 1
                            else:
                                focused_review_batch_negative_retained += 1
                    if progress:
                        verified_progress = (
                            (5 * total_chunks)
                            + math.ceil(
                                hypothesis_index
                                / max(1, total_hypotheses)
                                * total_chunks
                            )
                        )
                        progress(
                            min(total_progress, verified_progress),
                            total_progress,
                            "Knowledge Forge 关系结构终审 "
                            f"{hypothesis_index}/{total_hypotheses} "
                            "个关系假设",
                        )
                    continue
                reuse_batch_after_role_audit = False
                grouped_role_audit = grouped_role_audits.get(
                    hypothesis.hypothesis_id
                )
                if grouped_role_audit is not None:
                    grouped_result = str(
                        grouped_role_audit.get("result") or "UNAVAILABLE"
                    ).upper()
                    if grouped_result == "ALIGNED":
                        source_role_audit_aligned += 1
                        reuse_batch_after_role_audit = True
                    else:
                        if grouped_result == "CONFLICT":
                            source_role_audit_conflicts += 1
                        elif grouped_result == "UNCERTAIN":
                            source_role_audit_uncertain += 1
                        terminal_factory = (
                            v2.source_projection_group_anchor_audit_terminal_attempt
                            if grouped_role_audit.get("audit_kind") == "anchor"
                            else v2.source_projection_group_audit_terminal_attempt
                        )
                        group_terminal = (
                            terminal_factory(
                                hypothesis,
                                result=grouped_result,
                                verifier_model_revision=(
                                    context["model_revision"]
                                ),
                                request_payload=grouped_role_audit.get(
                                    "request_payload"
                                ) or {},
                                raw_response_text=str(
                                    grouped_role_audit.get(
                                        "raw_response_text"
                                    ) or ""
                                ),
                                parsed_response=grouped_role_audit.get(
                                    "parsed_response"
                                ) or {},
                                token_usage=grouped_role_audit.get(
                                    "token_usage"
                                ) or {},
                                error_message=str(
                                    grouped_role_audit.get(
                                        "error_message"
                                    ) or ""
                                ),
                            )
                        )
                        self.db.persist_grounded_relation_verification_attempts(
                            run_id, [group_terminal], terminal=True,
                        )
                        verification_attempts.append(group_terminal)
                        if focused_plan is not None:
                            if focused_plan.get("kind") == "confirmation":
                                focused_review_batch_positive_rejected += 1
                            elif focused_plan.get("kind") == "recovery":
                                focused_review_batch_negative_retained += 1
                        continue
                elif (
                    focused_plan is not None
                    and focused_plan.get("source_role_audit") is True
                ):
                    competitor_hypotheses = tuple(
                        primary_hypothesis_by_id[value]
                        for value in focused_plan.get(
                            "competing_hypothesis_ids", ()
                        )
                        if value in primary_hypothesis_by_id
                    )
                    role_audit_raw: Any = None
                    role_audit_raw_text = ""
                    role_audit_parsed: dict[str, Any] = {}
                    role_audit_token_usage: dict[str, Any] = {}
                    role_audit_error = ""
                    role_audit_result = "UNAVAILABLE"
                    role_audit_request: dict[str, Any] = {
                        "stage": "source_projection_role_audit",
                        "focused_review": dict(focused_plan),
                        "model": context["model"],
                        "model_revision": context["model_revision"],
                        "temperature": 0.0,
                        "context_budget_tokens": context_budget,
                        "max_output_tokens": 256,
                        "safety_margin_tokens": (
                            Config.relation_classifier_safety_margin
                        ),
                    }
                    try:
                        role_audit_prompt = (
                            v2.source_projection_role_audit_prompt(
                                hypothesis,
                                opportunity,
                                content=content,
                                logical_schema=context["schema"],
                                competing_projections=(
                                    competitor_hypotheses
                                ),
                            )
                        )
                        role_audit_schema = (
                            v2.source_projection_role_audit_response_schema()
                        )
                        role_audit_request.update({
                            "prompt": role_audit_prompt,
                            "schema": role_audit_schema,
                            "system": (
                                "Audit exact source roles. Return only JSON."
                            ),
                        })
                        source_role_audit_calls += 1
                        source_role_audit_claims += 1
                        focused_review_model_calls += 1
                        role_audit_raw = self.models.json_chat(
                            model=context["model"],
                            system=role_audit_request["system"],
                            prompt=role_audit_prompt,
                            schema=role_audit_schema,
                            temperature=0.0,
                            context_budget_tokens=context_budget,
                            max_output_tokens=256,
                            safety_margin_tokens=(
                                Config.relation_classifier_safety_margin
                            ),
                        )
                        role_audit_result = (
                            v2.parse_source_projection_role_audit(
                                role_audit_raw
                            )
                        )
                        role_audit_parsed = {
                            "projection_status": role_audit_result,
                        }
                    except Exception as exc:
                        source_role_audit_failures += 1
                        role_audit_error = (
                            f"{type(exc).__name__}: {exc}"[:2000]
                        )
                    finally:
                        role_audit_raw_text = str(
                            getattr(
                                self.models, "last_raw_response_text", "",
                            )
                            or (
                                _safe_json_text(role_audit_raw)
                                if role_audit_raw is not None else ""
                            )
                        )
                        diagnostics = getattr(
                            self.models, "last_call_diagnostics", None,
                        )
                        role_audit_token_usage = (
                            asdict(diagnostics)
                            if diagnostics is not None else {}
                        )
                    role_audit_request_hash = (
                        relation_model_payload_hash(
                            "request", role_audit_request,
                        )
                    )
                    self.db.record_grounded_relation_source_model_call(
                        run_id,
                        {
                            "component": "source_projection_role_audit",
                            "model_name": context["model"],
                            "model_digest": context["model_revision"],
                            "prompt_version": (
                                v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PROMPT_VERSION
                            ),
                            "schema_version": (
                                v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_SCHEMA_VERSION
                            ),
                            "parser_version": (
                                v2.GROUNDED_RELATION_SOURCE_ROLE_AUDIT_PARSER_VERSION
                            ),
                            "idempotency_key": (
                                "source-role-audit:"
                                + hypothesis.hypothesis_id
                            ),
                            "attempt_no": 1,
                            "status": (
                                "COMPLETED"
                                if not role_audit_error else "FAILED"
                            ),
                            "request_hash": role_audit_request_hash,
                            "response_hash": (
                                relation_model_payload_hash(
                                    "response", role_audit_raw_text,
                                ) if role_audit_raw_text else ""
                            ),
                            "request_payload": role_audit_request,
                            "raw_response_text": role_audit_raw_text,
                            "parsed_response": role_audit_parsed,
                            "token_usage": role_audit_token_usage,
                            "error_type": (
                                role_audit_error.split(":", 1)[0]
                                if role_audit_error else ""
                            ),
                            "error_message": role_audit_error,
                            "terminal_status": (
                                "COMPLETED"
                                if not role_audit_error else "FAILED"
                            ),
                        },
                    )
                    if role_audit_result == "ALIGNED":
                        source_role_audit_aligned += 1
                        reuse_batch_after_role_audit = bool(
                            focused_plan.get("role_audit_only") is True
                            and focused_plan.get("kind") == "confirmation"
                        )
                    elif v2.source_role_audit_conflict_is_terminal(
                        focused_plan
                    ):
                        if role_audit_result == "CONFLICT":
                            source_role_audit_conflicts += 1
                        elif role_audit_result == "UNCERTAIN":
                            source_role_audit_uncertain += 1
                        role_terminal = (
                            v2.source_projection_role_audit_terminal_attempt(
                                hypothesis,
                                result=role_audit_result,
                                verifier_model_revision=(
                                    context["model_revision"]
                                ),
                                request_payload=role_audit_request,
                                raw_response_text=role_audit_raw_text,
                                parsed_response=role_audit_parsed,
                                token_usage=role_audit_token_usage,
                                error_message=role_audit_error,
                            )
                        )
                        self.db.persist_grounded_relation_verification_attempts(
                            run_id, [role_terminal], terminal=True,
                        )
                        verification_attempts.append(role_terminal)
                        if focused_plan.get("kind") == "confirmation":
                            focused_review_batch_positive_rejected += 1
                        elif focused_plan.get("kind") == "recovery":
                            focused_review_batch_negative_retained += 1
                        continue
                    else:
                        if role_audit_result == "CONFLICT":
                            source_role_audit_conflicts += 1
                        elif role_audit_result == "UNCERTAIN":
                            source_role_audit_uncertain += 1
                final_attempt = None
                for attempt_no in range(
                    1, Config.relation_max_attempts + 1,
                ):
                    verifier_transport_called = False
                    verifier_transport_request: dict[str, Any] = {}
                    verifier_transport_raw_text = ""
                    verifier_transport_parsed: dict[str, Any] = {}
                    verifier_transport_token_usage: dict[str, Any] = {}

                    def final_verifier(
                        prompt: str, response_schema: dict[str, Any],
                    ) -> Any:
                        nonlocal verifier_calls
                        nonlocal focused_review_model_calls
                        nonlocal focused_verifier_model_calls
                        nonlocal verifier_transport_called
                        nonlocal verifier_transport_request
                        nonlocal verifier_transport_raw_text
                        nonlocal verifier_transport_parsed
                        nonlocal verifier_transport_token_usage
                        if (
                            attempt_no == 1
                            and hypothesis.hypothesis_id
                            in batch_verifier_decisions
                            and (
                                hypothesis.hypothesis_id
                                not in focused_review_plans
                                or reuse_batch_after_role_audit
                            )
                        ):
                            return batch_verifier_decisions[
                                hypothesis.hypothesis_id
                            ]
                        verifier_calls += 1
                        focused_plan = focused_review_plans.get(
                            hypothesis.hypothesis_id
                        )
                        if focused_plan is not None:
                            focused_review_model_calls += 1
                            focused_verifier_model_calls += 1
                        verifier_transport_called = True
                        verifier_transport_request = {
                            "stage": (
                                "focused_verifier"
                                if focused_plan is not None else "verifier"
                            ),
                            "focused_review": dict(focused_plan or {}),
                            "model": context["model"],
                            "model_revision": context["model_revision"],
                            "prompt": prompt,
                            "schema": response_schema,
                            "temperature": 0.0,
                            "context_budget_tokens": context_budget,
                            "max_output_tokens": self._verifier_output_budget(
                                context_budget, attempt_no,
                            ),
                            "attempt_no": attempt_no,
                        }
                        raw_result: Any = None
                        try:
                            raw_result = self._verify(
                                model=context["model"],
                                context_budget_tokens=context_budget,
                                prompt=prompt,
                                response_schema=response_schema,
                                attempt_no=attempt_no,
                            )
                            return raw_result
                        finally:
                            verifier_transport_raw_text = str(
                                getattr(
                                    self.models,
                                    "last_raw_response_text",
                                    "",
                                )
                                or (
                                    _safe_json_text(raw_result)
                                    if raw_result is not None else ""
                                )
                            )
                            parsed_transport = (
                                raw_result
                                if isinstance(raw_result, Mapping)
                                else getattr(
                                    self.models,
                                    "last_parsed_json_response",
                                    None,
                                )
                            )
                            verifier_transport_parsed = (
                                dict(parsed_transport)
                                if isinstance(parsed_transport, Mapping)
                                else {}
                            )
                            diagnostics = getattr(
                                self.models, "last_call_diagnostics", None,
                            )
                            verifier_transport_token_usage = (
                                asdict(diagnostics)
                                if diagnostics is not None else {}
                            )

                    def predicate_blind_endpoint_probe(
                        prompt: str, _response_schema: dict[str, Any],
                    ) -> dict[str, Any]:
                        nonlocal role_probe_calls
                        nonlocal role_probe_cache_hits
                        if source_projection_candidate:
                            # Source-frame candidates already passed a complete
                            # predicate-blind inventory plus Schema binding.
                            # Re-running the same endpoint-role question here
                            # only repeats the full context. Keep the audit
                            # shape and exact endpoint coordinates, while the
                            # final directed verifier remains mandatory.
                            payload = json.loads(prompt)
                            endpoint_a = payload.get("endpoint_a") or {}
                            endpoint_b = payload.get("endpoint_b") or {}
                            return {
                                "endpoint_a_complete_argument": True,
                                "endpoint_b_complete_argument": True,
                                "endpoint_a_argument_start": int(
                                    endpoint_a["span"][0]
                                ),
                                "endpoint_a_argument_end": int(
                                    endpoint_a["span"][1]
                                ),
                                "endpoint_b_argument_start": int(
                                    endpoint_b["span"][0]
                                ),
                                "endpoint_b_argument_end": int(
                                    endpoint_b["span"][1]
                                ),
                                "direct_relation_present": True,
                                "subject_role": "source-frame bound subject role",
                                "object_role": "source-frame bound object role",
                                "expressed_relation": "source-frame bound assertion",
                            }
                        try:
                            cache_payload = json.loads(prompt)
                            cache_payload.pop("hypothesis_id", None)
                            cache_key = hashlib.sha256(
                                json.dumps(
                                    cache_payload,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()
                        except Exception:
                            cache_key = ""
                        if cache_key:
                            cached = role_probe_cache.get(cache_key)
                            if cached is not None:
                                role_probe_cache_hits += 1
                                return cached
                        role_probe_calls += 1
                        result = self._verify_predicate_blind_endpoint_probe(
                            model=context["model"],
                            context_budget_tokens=context_budget,
                            prompt=prompt,
                            attempt_no=attempt_no,
                        )
                        # Validate before caching. A malformed auxiliary result
                        # must remain retryable and must not poison every
                        # sibling hypothesis sharing this endpoint pair.
                        v2.parse_verifier_role_probe(result)
                        if cache_key:
                            role_probe_cache[cache_key] = result
                        return result

                    def predicate_blind_atomic_fact_inventory(
                        prompt: str, response_schema: dict[str, Any],
                    ) -> dict[str, Any]:
                        nonlocal atomic_fact_inventory_calls
                        nonlocal atomic_fact_inventory_cache_hits
                        nonlocal atomic_fact_inventory_incomplete
                        nonlocal atomic_fact_inventory_failures
                        cache_key = hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest()
                        cached = atomic_fact_inventory_cache.get(cache_key)
                        if cached is not None:
                            atomic_fact_inventory_cache_hits += 1
                            return cached
                        atomic_fact_inventory_calls += 1
                        try:
                            raw_inventory = self._inventory_atomic_facts(
                                model=context["model"],
                                context_budget_tokens=context_budget,
                                prompt=prompt,
                                response_schema=response_schema,
                                attempt_no=attempt_no,
                            )
                            prompt_payload = json.loads(prompt)
                            parsed_inventory = v2.parse_atomic_fact_inventory(
                                raw_inventory,
                                context_text=str(
                                    (prompt_payload.get("context") or {}).get(
                                        "text"
                                    ) or ""
                                ),
                            )
                        except Exception:
                            atomic_fact_inventory_failures += 1
                            raise
                        # INCOMPLETE is an explicit audit result, not a model
                        # transport failure. Cache it as well: repeating the
                        # same incomplete inventory for every predicate does
                        # not improve recall and can multiply long-document
                        # cost. The final predicate-aware verifier remains the
                        # publication authority in either case.
                        atomic_fact_inventory_cache[cache_key] = parsed_inventory
                        if not parsed_inventory["coverage_complete"]:
                            atomic_fact_inventory_incomplete += 1
                        return parsed_inventory

                    def atomic_frame_schema_alignment(
                        prompt: str, response_schema: dict[str, Any],
                    ) -> dict[str, Any]:
                        nonlocal atomic_frame_alignment_calls
                        nonlocal atomic_frame_alignment_cache_hits
                        nonlocal atomic_frame_alignment_failures
                        cache_key = hashlib.sha256(
                            prompt.encode("utf-8")
                        ).hexdigest()
                        cached = atomic_frame_alignment_cache.get(cache_key)
                        if cached is not None:
                            atomic_frame_alignment_cache_hits += 1
                            return cached
                        atomic_frame_alignment_calls += 1
                        try:
                            raw_alignment = self._align_atomic_frame_schema(
                                model=context["model"],
                                context_budget_tokens=context_budget,
                                prompt=prompt,
                                response_schema=response_schema,
                                attempt_no=attempt_no,
                            )
                        except Exception:
                            atomic_frame_alignment_failures += 1
                            raise
                        if not isinstance(raw_alignment, Mapping):
                            atomic_frame_alignment_failures += 1
                            raise v2.RelationClassificationError(
                                "frame schema alignment response must be an object"
                            )
                        result = dict(raw_alignment)
                        atomic_frame_alignment_cache[cache_key] = result
                        return result

                    def complete_role_alignment(
                        prompt: str, _response_schema: dict[str, Any],
                    ) -> dict[str, str]:
                        nonlocal role_alignment_calls
                        nonlocal role_alignment_cache_hits
                        if source_projection_candidate:
                            # The source-binding model has already selected an
                            # ordered Schema binding. Represent that immutable
                            # direction in the same four-way audit contract;
                            # explicit false fields from the final verifier
                            # still remain authoritative and fail closed.
                            return {
                                "endpoint_a_vs_source": "MATCH",
                                "endpoint_b_vs_target": "MATCH",
                                "endpoint_b_vs_source": "MISMATCH",
                                "endpoint_a_vs_target": "MISMATCH",
                                "forward_relation_vs_definition": "MATCH",
                                "reverse_relation_vs_definition": "MISMATCH",
                                "semantic_contrast": "DEFINITION",
                            }
                        try:
                            cache_payload = json.loads(prompt)
                            cache_payload.pop("hypothesis_id", None)
                            cache_key = hashlib.sha256(
                                json.dumps(
                                    cache_payload,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest()
                        except Exception:
                            cache_key = ""
                        if cache_key:
                            cached = role_alignment_cache.get(cache_key)
                            if cached is not None:
                                role_alignment_cache_hits += 1
                                return cached
                        role_alignment_calls += 1
                        result = self._align_verifier_roles(
                            model=context["model"],
                            context_budget_tokens=context_budget,
                            prompt=prompt,
                            response_schema=(
                                v2.verifier_role_alignment_response_schema()
                            ),
                            attempt_no=attempt_no,
                        )
                        v2.parse_verifier_role_alignment(result)
                        if cache_key:
                            role_alignment_cache[cache_key] = result
                        return result

                    attempt = verify_relation_hypothesis(
                        hypothesis,
                        validation,
                        opportunity,
                        content=content,
                        logical_schema=context["schema"],
                        # Stage one reads the source without seeing a target
                        # predicate. Stage two compares those exact endpoint
                        # roles to the versioned predicate policy in both
                        # directions. The final verifier then adjudicates the
                        # resulting ordered claim. No stage can publish alone.
                        role_probe=predicate_blind_endpoint_probe,
                        role_alignment_probe=complete_role_alignment,
                        atomic_fact_probe=(
                            None
                            if source_projection_candidate
                            else predicate_blind_atomic_fact_inventory
                        ),
                        atomic_frame_alignment_probe=(
                            None
                            if source_projection_candidate
                            else atomic_frame_schema_alignment
                        ),
                        verifier=final_verifier,
                        verifier_model_revision=context["model_revision"],
                        attempt_no=attempt_no,
                        structural_adjudication_only=bool(
                            (hypothesis.predicate_metadata or {}).get(
                                "structural_event_projection"
                            )
                        ),
                    )
                    if verifier_transport_called:
                        attempt = replace(
                            attempt,
                            model_request_payload=verifier_transport_request,
                            raw_response_text=verifier_transport_raw_text,
                            parsed_response=verifier_transport_parsed,
                            token_usage=verifier_transport_token_usage,
                        )
                    if source_projection_candidate:
                        source_frame_fast_path_count += 1
                    is_terminal_attempt = (
                        attempt.verdict != v2.RETRYABLE_FAILED
                        or validation.status != v2.PASS
                        or attempt_no == Config.relation_max_attempts
                    )
                    self.db.persist_grounded_relation_verification_attempts(
                        run_id, [attempt], terminal=is_terminal_attempt,
                    )
                    verification_attempts.append(attempt)
                    final_attempt = attempt
                    if (
                        attempt.verdict != v2.RETRYABLE_FAILED
                        or validation.status != v2.PASS
                    ):
                        break
                if final_attempt is None:
                    raise RuntimeError("关系 verifier 未生成终态")
                focused_plan = focused_review_plans.get(
                    hypothesis.hypothesis_id
                )
                if focused_plan is not None:
                    if focused_plan.get("kind") == "confirmation":
                        if final_attempt.verdict == v2.SUPPORTED:
                            focused_review_batch_positive_confirmed += 1
                        else:
                            focused_review_batch_positive_rejected += 1
                    elif focused_plan.get("kind") == "recovery":
                        if final_attempt.verdict == v2.SUPPORTED:
                            focused_review_batch_negative_recovered += 1
                        else:
                            focused_review_batch_negative_retained += 1
                if progress:
                    verified_progress = (
                        (5 * total_chunks)
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
            secondary_metrics = self._run_secondary_review(
                run_id=run_id,
                project_id=project_id,
                context=context,
                fingerprint=fingerprint,
                opportunities=opportunity_by_pair,
                hypotheses=hypotheses,
                validations=validation_by_hypothesis,
                verification_attempts=verification_attempts,
                context_budget_tokens=context_budget,
            )
            hypotheses.extend(secondary_metrics.get("added_hypotheses") or [])
            verification_attempts.extend(secondary_metrics.get("added_attempts") or [])
            for validation in secondary_metrics.get("added_validations") or []:
                validation_by_hypothesis[str(validation.hypothesis_id)] = validation

            # A verified relation over one member of an explicit coordinated
            # source slot licenses the same ordered relation for its equivalent
            # members.  This closes a recurrent local-model inconsistency where
            # one item in "A, B and C" is supported while another is rejected
            # as mere enumeration.  The v2 helper requires a pairwise clique,
            # one common varying slot, identical evidence, asserted polarity,
            # ordinary full-verifier outcomes and one independently supported
            # anchor; model-only transitive groups and audit conflicts remain
            # fail-closed.
            pre_recovery_terminal_attempts: dict[
                str, v2.VerificationAttempt
            ] = {}
            for item in verification_attempts:
                current_attempt = pre_recovery_terminal_attempts.get(
                    str(item.hypothesis_id)
                )
                if (
                    current_attempt is None
                    or int(item.attempt_no) > int(current_attempt.attempt_no)
                ):
                    pre_recovery_terminal_attempts[
                        str(item.hypothesis_id)
                    ] = item
            coordinated_recovery_claims = [
                (
                    hypothesis,
                    opportunity_by_pair[hypothesis.pair_identity],
                )
                for hypothesis in hypotheses
                if validation_by_hypothesis.get(
                    str(hypothesis.hypothesis_id)
                ) is not None
                and validation_by_hypothesis[
                    str(hypothesis.hypothesis_id)
                ].status == v2.PASS
            ]
            coordinated_recovery_plans = (
                v2.source_projection_coordinated_recovery_plan(
                    coordinated_recovery_claims,
                    pre_recovery_terminal_attempts,
                )
            )
            coordinated_source_recovery_candidates = len(
                coordinated_recovery_plans
            )
            hypothesis_by_stable_id = {
                str(item.hypothesis_id): item for item in hypotheses
            }
            for hypothesis_id, recovery_plan in sorted(
                coordinated_recovery_plans.items()
            ):
                hypothesis = hypothesis_by_stable_id[hypothesis_id]
                current_attempt = pre_recovery_terminal_attempts[
                    hypothesis_id
                ]
                anchor_attempt = pre_recovery_terminal_attempts[
                    str(recovery_plan["anchor_hypothesis_id"])
                ]
                recovery_attempt = (
                    v2.coordinated_source_projection_recovery_attempt(
                        hypothesis,
                        anchor_attempt=anchor_attempt,
                        member_hypothesis_ids=recovery_plan[
                            "member_hypothesis_ids"
                        ],
                        varying_slot=str(
                            recovery_plan.get("varying_slot") or ""
                        ),
                        coordination_grounding=dict(
                            recovery_plan.get("coordination_grounding") or {}
                        ),
                        attempt_no=int(current_attempt.attempt_no) + 1,
                    )
                )
                self.db.persist_grounded_relation_verification_attempts(
                    run_id, [recovery_attempt], terminal=True,
                )
                verification_attempts.append(recovery_attempt)
                pre_recovery_terminal_attempts[
                    hypothesis_id
                ] = recovery_attempt
                coordinated_source_recoveries += 1

            terminal_verifier_attempts = {}
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
            # A terminal protocol failure is local to one hypothesis.  It is a
            # fully audited, non-publishable terminal state, just like REJECTED
            # or UNCERTAIN for materialization purposes.  Quarantine it and
            # continue normalizing other independently SUPPORTED hypotheses.
            # Global integrity failures (migration gaps, stale fingerprints,
            # missing source context) still abort the whole run below.
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
            verified_eligible_count = sum(
                int(item.eligible) for item in materialization_decisions
            )
            publication_candidate_rows = self._endpoint_rows(
                run_id=run_id,
                normalization=normalization,
                materialization_decisions=materialization_decisions,
                opportunities=opportunity_by_pair,
                hypotheses=hypothesis_by_id,
                chunks=context["chunk_by_id"],
                eligible_assignments=context["assignment_by_mention"],
                logical_schema=context["schema"],
            )
            (
                publication_candidate_rows,
                atomic_publication_audit_metrics,
            ) = self._run_atomic_publication_audit(
                run_id=run_id,
                rows=publication_candidate_rows,
                context=context,
                context_budget_tokens=context_budget,
            )
            # Publication auditing is itself model-backed. A replay is complete
            # only after that final model boundary has consumed its persisted
            # response; checking immediately after relation verification would
            # reject every valid ledger that contains publication-audit calls.
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
            publication_candidate_rows, _ = (
                suppress_exact_open_schema_duplicates(
                    publication_candidate_rows
                )
            )
            publication_suppressed_reasons = Counter()
            suppressed_fact_ids: set[str] = set()
            for row in publication_candidate_rows:
                semantics = (
                    (row.get("metadata") or {}).get("relation_semantics") or {}
                )
                publication = (
                    semantics.get("publication")
                    if isinstance(semantics.get("publication"), Mapping)
                    else {}
                )
                if publication.get("eligible") is False:
                    fact_id = str(row.get("relation_key") or "")
                    if fact_id:
                        suppressed_fact_ids.add(fact_id)
                    publication_suppressed_reasons[str(
                        publication.get("reason") or "unspecified"
                    )] += 1
            if suppressed_fact_ids:
                materialization_decisions = [
                    replace(
                        item,
                        eligible=False,
                        reasons=tuple(sorted({
                            *item.reasons,
                            "RELATION_PUBLICATION_GATE_REJECTED",
                        })),
                    )
                    if str(item.fact_id) in suppressed_fact_ids else item
                    for item in materialization_decisions
                ]
            endpoint_rows = [
                item for item in publication_candidate_rows
                if str(item.get("relation_key") or "")
                not in suppressed_fact_ids
            ]
            endpoint_rows, predicate_convergence = (
                converge_relation_display_labels(endpoint_rows)
            )
            self.db.persist_grounded_relation_normalization(
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
            verification_reason_counts = Counter(
                str(item.reason_code) for item in terminal_attempts.values()
            )
            validation_counts = Counter(
                str(item.status) for item in validations
            )
            pair_terminal_status = {
                str(item.pair_identity): str(item.status)
                for item in all_opportunities
            }
            for decision in decisions:
                pair_terminal_status[str(decision.pair_identity)] = str(
                    decision.decision_status
                )
            for pair_id in proposer_pair_ids:
                pair_terminal_status[pair_id] = v2.HYPOTHESES_EMITTED
            for pair_id in source_projection_pair_ids:
                pair_terminal_status[pair_id] = v2.HYPOTHESES_EMITTED
            eligible_count = sum(
                int(item.eligible) for item in materialization_decisions
            )
            metrics = {
                "entity_publishability": dict(
                    context.get("entity_publishability") or {}
                ),
                "macro_context": {
                    "version": str(
                        context["macro_context"].get("version") or ""
                    ),
                    "hash": str(context["macro_context"].get("hash") or ""),
                    "project_entity_count": int(
                        context["macro_context"].get("entity_count") or 0
                    ),
                    "project_document_count": int(
                        context["macro_context"].get("document_count") or 0
                    ),
                    "project_chunk_count": int(
                        context["macro_context"].get("chunk_count") or 0
                    ),
                },
                "pair_opportunities_total": len(all_opportunities),
                "pair_opportunities_semantically_processed": (
                    len(decisions)
                    + len(proposer_pair_ids | source_projection_pair_ids)
                ),
                "pair_opportunities_classified": len(decisions),
                "pair_opportunities_classifier_model": (
                    len(decisions) - proposer_coverage_no_relation
                ),
                "pair_opportunities_local_subgraph": len(proposer_pair_ids),
                "pair_statuses": dict(Counter(pair_terminal_status.values())),
                "classification_decisions": dict(decision_counts),
                "pair_decision_statuses": dict(Counter(
                    str(item.decision_status) for item in decisions
                )),
                "hypotheses_total": len(hypotheses),
                "classified_hypotheses": len(classifier_hypotheses),
                "local_subgraph_hypotheses": len(
                    proposer_hypotheses_to_persist
                ),
                "source_first_enabled": bool(source_first_enabled),
                "source_first_primary_contexts": (
                    source_first_primary_contexts
                ),
                "proposer_scopes_skipped_by_source_first": (
                    proposer_scopes_skipped_by_source_first
                ),
                "source_inventory_model_calls": source_inventory_calls,
                "source_inventory_cache_hits": source_inventory_cache_hits,
                "source_inventory_failures": source_inventory_failures,
                "source_inventory_incomplete": source_inventory_incomplete,
                "source_inventory_audited_contexts": len(
                    source_inventory_audited_contexts
                ),
                "source_inventory_complete_contexts": len(
                    source_inventory_complete_contexts
                ),
                "source_first_classifier_backstop_pairs": (
                    source_first_classifier_backstop_pairs
                ),
                "source_repair_classifier_backstop_pairs": (
                    source_repair_classifier_backstop_pairs
                ),
                "source_first_classifier_pairs_skipped": (
                    source_first_classifier_pairs_skipped
                ),
                "source_parser_endpoint_pairs_filtered": (
                    source_parser_endpoint_pairs_filtered
                ),
                "grounded_open_facts_enabled": bool(
                    Config.relation_grounded_open_facts_enabled
                ),
                "source_open_fact_gleaning_calls": (
                    source_open_fact_gleaning_calls
                ),
                "source_open_fact_gleaning_cache_hits": (
                    source_open_fact_gleaning_cache_hits
                ),
                "source_open_fact_gleaning_failures": (
                    source_open_fact_gleaning_failures
                ),
                "source_open_fact_gleaning_rounds_completed": (
                    source_open_fact_gleaning_rounds_completed
                ),
                "source_open_facts_recovered": source_open_facts_recovered,
                "source_open_fact_reverse_recoveries_suppressed": (
                    source_open_fact_reverse_recoveries_suppressed
                ),
                "source_reference_proxies_accepted": (
                    source_reference_proxies_accepted
                ),
                "source_reference_proxies_active": sum(
                    1
                    for context_mentions in mentions_by_context.values()
                    for item in context_mentions
                    if bool(item.get("relation_reference_proxy"))
                ),
                "source_reference_bindings_dropped": (
                    source_reference_bindings_dropped
                ),
                "source_reference_adjudication_enabled": bool(
                    Config.relation_reference_adjudication_enabled
                ),
                "source_reference_adjudication_targets": (
                    source_reference_adjudication_targets
                ),
                "source_reference_adjudication_calls": (
                    source_reference_adjudication_calls
                ),
                "source_reference_adjudication_cache_hits": (
                    source_reference_adjudication_cache_hits
                ),
                "source_reference_adjudication_failures": (
                    source_reference_adjudication_failures
                ),
                "source_reference_adjudication_proxies_accepted": (
                    source_reference_adjudication_proxies_accepted
                ),
                "source_reference_binding_conflicts": (
                    source_reference_binding_conflicts
                ),
                "source_reference_drop_reasons": dict(
                    source_reference_drop_reasons
                ),
                "source_deterministic_frames_recovered": (
                    source_deterministic_frames_recovered
                ),
                "source_deterministic_frames_by_origin": dict(
                    source_deterministic_frames_by_origin
                ),
                "source_binding_model_calls": source_binding_calls,
                "source_binding_cache_hits": source_binding_cache_hits,
                "source_binding_failures": source_binding_failures,
                "source_binding_incomplete": source_binding_incomplete,
                "source_binding_items_dropped": source_binding_items_dropped,
                "source_bindings_total": source_bindings_total,
                "source_binding_model_skipped_empty_contract": (
                    source_binding_model_skipped_empty_contract
                ),
                "source_binding_review_candidates_enabled": (
                    source_binding_review_candidates_enabled
                ),
                "source_structure_bindings_total": (
                    source_structure_bindings_total
                ),
                "source_binding_gleaning_calls": (
                    source_binding_gleaning_calls
                ),
                "source_binding_gleaning_cache_hits": (
                    source_binding_gleaning_cache_hits
                ),
                "source_binding_gleaning_failures": (
                    source_binding_gleaning_failures
                ),
                "source_binding_gleaning_rounds_completed": (
                    source_binding_gleaning_rounds_completed
                ),
                "source_bindings_recovered": source_bindings_recovered,
                "source_recall_shortlist_bindings": (
                    source_recall_shortlist_bindings
                ),
                "source_recall_shortlist_frames_considered": (
                    source_recall_shortlist_frames_considered
                ),
                "source_recall_shortlist_candidates_considered": (
                    source_recall_shortlist_candidates_considered
                ),
                "source_recall_shortlist_candidates_below_threshold": (
                    source_recall_shortlist_candidates_below_threshold
                ),
                "source_open_carrier_bindings": (
                    source_open_carrier_bindings
                ),
                "source_open_carrier_contexts": (
                    source_open_carrier_contexts
                ),
                "source_open_carrier_eligible_frames": (
                    source_open_carrier_eligible_frames
                ),
                "source_open_carrier_skipped_precise": (
                    source_open_carrier_skipped_precise
                ),
                "source_open_carrier_parallel_precise": (
                    source_open_carrier_parallel_precise
                ),
                "source_open_carrier_skipped_low_information": (
                    source_open_carrier_skipped_low_information
                ),
                "source_open_carrier_bindings_capped": (
                    source_open_carrier_bindings_capped
                ),
                "source_frame_projection_hypotheses": len(
                    source_hypotheses_to_persist
                ),
                "source_conflict_groups": source_conflict_groups_count,
                "source_conflict_model_calls": source_conflict_calls,
                "source_conflict_cache_hits": source_conflict_cache_hits,
                "source_conflict_failures": source_conflict_failures,
                "source_conflict_none": source_conflict_none,
                "source_conflict_suppressed": source_conflict_suppressed,
                "deterministic_validation": dict(validation_counts),
                "verification_attempts": len(verification_attempts),
                "verification_terminal_count": len(terminal_attempts),
                "verification_verdicts": dict(verdict_counts),
                "verification_reason_codes": dict(
                    verification_reason_counts
                ),
                "verifier_terminal_failures": int(
                    verdict_counts.get(v2.RETRYABLE_FAILED, 0)
                ),
                "verifier_hypotheses_quarantined": int(
                    verdict_counts.get(v2.RETRYABLE_FAILED, 0)
                ),
                "verifier_supported": int(
                    verdict_counts.get(v2.SUPPORTED, 0)
                ),
                "facts_after_normalization": len(normalization.facts),
                "facts_verified_eligible": verified_eligible_count,
                "facts_materialization_eligible": eligible_count,
                "facts_predicate_publication_suppressed": len(
                    suppressed_fact_ids
                ),
                "atomic_publication_audit": dict(
                    atomic_publication_audit_metrics
                ),
                "facts_review_only": sum(
                    int(fact.review_only)
                    for fact in normalization.facts
                ),
                "normalization": dict(normalization.metrics),
                "proposer_model_calls": proposer_calls,
                "proposer_cache_hits": proposer_cache_hits,
                "proposer_failures": proposer_failures,
                "proposer_circuit_breaker_tripped":
                    proposer_circuit_breaker_tripped,
                "proposer_skipped_by_circuit": proposer_skipped_by_circuit,
                "proposer_terminal_statuses": dict(proposer_terminal_statuses),
                "proposer_scopes_total": proposer_scopes_total,
                "proposer_scopes_complete": proposer_scopes_complete,
                "proposer_scopes_fallback": proposer_scopes_fallback,
                "proposer_coverage_no_relation":
                    proposer_coverage_no_relation,
                "proposer_hypothesis_rejections":
                    dict(proposer_hypothesis_rejections),
                "proposals_total": len(proposer_hypotheses),
                "proposer_hypotheses_persisted":
                    len(proposer_hypotheses_to_persist),
                "missing_endpoint_mentions": missing_endpoint_mentions,
                "classifier_model_calls": classifier_calls,
                "classifier_cache_hits": classifier_cache_hits,
                "classifier_failures": classifier_failures,
                "classifier_preflight_splits": classifier_preflight_splits,
                "classifier_partial_batches": classifier_partial_batches,
                "classifier_partial_decisions": classifier_partial_decisions,
                "classifier_partial_fallbacks": classifier_partial_fallbacks,
                "verification_mode": str(
                    Config.relation_verification_mode
                ),
                "verifier_model_calls": verifier_calls,
                "role_probe_model_calls": role_probe_calls,
                "role_probe_cache_hits": role_probe_cache_hits,
                "role_alignment_model_calls": role_alignment_calls,
                "role_alignment_cache_hits": role_alignment_cache_hits,
                "source_frame_fast_path_count": source_frame_fast_path_count,
                "batch_verifier_calls": (
                    batch_verifier_calls
                ),
                "batch_verifier_cache_hits": (
                    batch_verifier_cache_hits
                ),
                "batch_verifier_failures": (
                    batch_verifier_failures
                ),
                "batch_verifier_partial_batches": (
                    batch_verifier_partial_batches
                ),
                "batch_verifier_partial_decisions": (
                    batch_verifier_partial_decisions
                ),
                "batch_verifier_decisions": len(
                    batch_verifier_decisions
                ),
                "batch_verifier_fallbacks": (
                    batch_verifier_fallbacks
                ),
                "batch_verifier_ungrounded_decisions": (
                    batch_verifier_ungrounded_decisions
                ),
                "batch_verifier_structure_bypassed": (
                    batch_verifier_structure_bypassed
                ),
                "focused_review_eligible": focused_review_eligible,
                "focused_review_selected": focused_review_selected,
                "focused_review_mandatory": focused_review_mandatory,
                "focused_review_model_calls": focused_review_model_calls,
                "focused_verifier_model_calls": focused_verifier_model_calls,
                "focused_review_confirmations": (
                    focused_review_confirmations
                ),
                "focused_review_recoveries": focused_review_recoveries,
                "focused_review_batch_positive_confirmed": (
                    focused_review_batch_positive_confirmed
                ),
                "focused_review_batch_positive_rejected": (
                    focused_review_batch_positive_rejected
                ),
                "focused_review_batch_negative_recovered": (
                    focused_review_batch_negative_recovered
                ),
                "focused_review_batch_negative_retained": (
                    focused_review_batch_negative_retained
                ),
                "source_role_audit_calls": source_role_audit_calls,
                "source_role_audit_claims": source_role_audit_claims,
                "source_role_audit_aligned": source_role_audit_aligned,
                "source_role_audit_conflicts": source_role_audit_conflicts,
                "source_role_audit_uncertain": source_role_audit_uncertain,
                "source_role_audit_failures": source_role_audit_failures,
                "source_group_audit_calls": source_group_audit_calls,
                "source_group_audit_cache_hits": (
                    source_group_audit_cache_hits
                ),
                "source_group_audit_claims": source_group_audit_claims,
                "source_group_audit_aligned": source_group_audit_aligned,
                "source_group_audit_conflicts": (
                    source_group_audit_conflicts
                ),
                "source_group_audit_uncertain": (
                    source_group_audit_uncertain
                ),
                "source_group_audit_failures": source_group_audit_failures,
                "source_group_anchor_audit_calls": (
                    source_group_anchor_audit_calls
                ),
                "source_group_anchor_audit_cache_hits": (
                    source_group_anchor_audit_cache_hits
                ),
                "source_group_anchor_audit_claims": (
                    source_group_anchor_audit_claims
                ),
                "source_group_anchor_audit_aligned": (
                    source_group_anchor_audit_aligned
                ),
                "source_group_anchor_audit_conflicts": (
                    source_group_anchor_audit_conflicts
                ),
                "source_group_anchor_audit_uncertain": (
                    source_group_anchor_audit_uncertain
                ),
                "source_group_anchor_audit_failures": (
                    source_group_anchor_audit_failures
                ),
                "source_structure_audit_calls": (
                    source_structure_audit_calls
                ),
                "source_structure_audit_aligned": (
                    source_structure_audit_aligned
                ),
                "source_structure_audit_conflicts": (
                    source_structure_audit_conflicts
                ),
                "source_structure_audit_uncertain": (
                    source_structure_audit_uncertain
                ),
                "source_structure_audit_failures": (
                    source_structure_audit_failures
                ),
                "deterministic_structure_contract_checks": (
                    deterministic_structure_contract_checks
                ),
                "deterministic_structure_contract_supported": (
                    deterministic_structure_contract_supported
                ),
                "deterministic_structure_contract_rejected": (
                    deterministic_structure_contract_rejected
                ),
                "deterministic_structure_contract_uncertain": (
                    deterministic_structure_contract_uncertain
                ),
                "coordinated_source_recovery_candidates": (
                    coordinated_source_recovery_candidates
                ),
                "coordinated_source_recoveries": (
                    coordinated_source_recoveries
                ),
                "atomic_fact_inventory_model_calls": (
                    atomic_fact_inventory_calls
                ),
                "atomic_fact_inventory_cache_hits": (
                    atomic_fact_inventory_cache_hits
                ),
                "atomic_fact_inventory_cached_contexts": len(
                    atomic_fact_inventory_cache
                ),
                "atomic_fact_inventory_incomplete": (
                    atomic_fact_inventory_incomplete
                ),
                "atomic_fact_inventory_failures": (
                    atomic_fact_inventory_failures
                ),
                "atomic_fact_inventory_vetoes": int(
                    verification_reason_counts.get(
                        "NO_DIRECT_ATOMIC_PAIR", 0,
                    )
                ),
                "atomic_frame_alignment_model_calls": (
                    atomic_frame_alignment_calls
                ),
                "atomic_frame_alignment_cache_hits": (
                    atomic_frame_alignment_cache_hits
                ),
                "atomic_frame_alignment_cached_claims": len(
                    atomic_frame_alignment_cache
                ),
                "atomic_frame_alignment_failures": (
                    atomic_frame_alignment_failures
                ),
                "atomic_frame_alignment_relation_vetoes": int(
                    verification_reason_counts.get(
                        "ATOMIC_FRAME_RELATION_MISMATCH", 0,
                    )
                ),
                "atomic_frame_alignment_direction_vetoes": int(
                    verification_reason_counts.get(
                        "ATOMIC_FRAME_DIRECTION_CONFLICT", 0,
                    )
                ),
                "exact_argument_orientation_calls": role_alignment_calls,
                "exact_argument_model_calls": role_alignment_calls * 2,
                "replay": replay_metrics,
                "secondary_review": {
                    key: value for key, value in secondary_metrics.items()
                    if key not in {"added_hypotheses", "added_attempts", "added_validations"}
                },
                "legacy_pipeline_calls": 0,
                "production_approved": False,
            }
            # The publication audit is model-backed and may run after the
            # earlier materialization-decision snapshot. Re-read every frozen
            # input at the final commit boundary so a document, Schema, policy
            # or semantic-code change during that audit cannot publish a stale
            # graph.
            current = self._context(
                project_id, namespace_id,
            )["fingerprint"]
            if fingerprint.root != current.root:
                terminal_status = "stale"
                self.db.finish_grounded_relation_run(
                    run_id,
                    status="stale",
                    metrics=metrics,
                    input_fingerprint=fingerprint.root,
                    error_message="materialization fingerprint became stale",
                )
                raise v2.StaleRelationSnapshotError(
                    "Knowledge Forge relation fingerprint stale; "
                    "refusing isolated materialization"
                )
            if len(endpoint_rows) != eligible_count:
                raise RuntimeError(
                    "Knowledge Forge eligible fact 与隔离物化数量不一致"
                )
            semantic_layers = Counter(
                str(
                    ((item.get("metadata") or {}).get("relation_semantics") or {})
                    .get("layer") or "unknown"
                )
                for item in endpoint_rows
            )
            semantic_families = {
                str((item.get("metadata") or {}).get("semantic_family_key") or "")
                for item in endpoint_rows
                if str((item.get("metadata") or {}).get("semantic_family_key") or "")
            }
            metrics["predicate_projection"] = {
                "version": RELATION_SEMANTIC_PROJECTION_VERSION,
                "layers": dict(semantic_layers),
                "family_count": len(semantic_families),
                "facts": len(endpoint_rows),
                "suppressed_facts": len(suppressed_fact_ids),
                "suppression_reasons": dict(
                    sorted(publication_suppressed_reasons.items())
                ),
                "display_convergence": predicate_convergence,
                "additional_model_calls": 0,
            }
            metrics["publication_integrity"] = {
                "policy_version": RELATION_ENDPOINT_INTEGRITY_POLICY_VERSION,
                "facts_checked": len(endpoint_rows),
                "valid_evidence_supports": sum(
                    int(
                        ((item.get("metadata") or {}).get(
                            "publication_integrity"
                        ) or {}).get("valid_evidence_supports") or 0
                    )
                    for item in endpoint_rows
                ),
                "dropped_invalid_evidence_supports": sum(
                    int(
                        ((item.get("metadata") or {}).get(
                            "publication_integrity"
                        ) or {}).get(
                            "dropped_invalid_evidence_supports"
                        ) or 0
                    )
                    for item in endpoint_rows
                ),
                "additional_model_calls": 0,
            }
            self.db.persist_grounded_entity_relation_endpoints(
                namespace_id, endpoint_rows,
            )
            self.db.finish_grounded_relation_run(
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
                    f"{len(endpoint_rows)} 个隔离物化事实，"
                    f"{verifier_terminal_failures} 个不可判定假设已隔离",
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
                    self.db.finish_grounded_relation_run(
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
