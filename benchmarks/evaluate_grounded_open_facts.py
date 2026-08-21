#!/usr/bin/env python3
"""Measure the source-first relation inventory against the frozen Gold corpus.

This intentionally evaluates the predicate-open inventory before Schema
binding and publication verification.  Directed and undirected endpoint recall
therefore diagnose candidate/role coverage without pretending that open model
phrases already equal canonical logical predicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_forge.config import Config
from knowledge_forge.ollama import OllamaClient
from knowledge_forge import relation_shadow_v2 as relation_v2
from benchmarks.calibration_policy import (
    partition_assertions,
    unique_fact_rows,
)
from knowledge_forge.relation_shadow_runtime import ShadowRelationV2Runtime
from knowledge_forge.relation_secondary import (
    SECONDARY_POLARITY_PROMPT_VERSION,
    SECONDARY_PASS_VERSION,
    SECONDARY_PROMPT_VERSION,
    ROUTE_ROLE_RETRY,
    SECONDARY_RELATION,
    parse_secondary_decision,
    parse_source_frame_polarity,
    route_secondary_attempt,
    secondary_recovery_prompt,
    secondary_response_schema,
    source_frame_polarity_prompt,
    source_frame_polarity_response_schema,
)
from knowledge_forge.relation_shadow_v2 import (
    SHADOW_RELATION_OPEN_FACT_SYSTEM_PROMPT,
    augment_grounded_open_fact_inventory,
    deterministic_source_structure_schema_bindings,
    grounded_open_fact_prompt,
    grounded_open_fact_response_schema,
    parse_grounded_open_facts,
    parse_relation_reference_resolutions,
    parse_source_frame_schema_bindings,
    source_frame_schema_recall_shortlist,
    source_frame_schema_binding_prompt,
    source_frame_schema_binding_response_schema,
    relation_reference_resolution_prompt,
    relation_reference_resolution_response_schema,
)


DEFAULT_ROOT = Path(__file__).resolve().parent / "knowledge_forge_cross_domain_v1"
ROOT = DEFAULT_ROOT

# Kept only as a reproducible ablation.  The four-domain experiment was slower
# and less accurate than the bounded per-candidate role audit, so it must not
# silently alter the default benchmark or production policy.
ENABLE_SOURCE_GROUP_AUDIT_EXPERIMENT = False


def _read(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _occurrences(
    document_id: str,
    *,
    coreference_links: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = _read("annotations/entities.json")["documents"][document_id]
    occurrences = [{
        "occurrence_key": str(row["mention_id"]),
        "mention_id": str(row["mention_id"]),
        "surface": str(row["surface"]),
        "mention_start": int(row["span"][0]),
        "mention_end": int(row["span"][1]),
        "canonical_id": str(row["canonical_id"]),
        "type": str(row["type"]),
    } for row in rows]
    occurrence_by_mention_id = {
        str(item["mention_id"]): item for item in occurrences
    }
    for link in coreference_links or ():
        if str(link.get("document_id") or "") != document_id:
            continue
        antecedent = occurrence_by_mention_id.get(
            str(link.get("target_mention_id") or "")
        )
        if antecedent is None:
            continue
        start, end = (int(value) for value in link["source_span"])
        occurrences.append({
            "occurrence_key": f"coreference:{link['link_id']}",
            # A relation-only proxy deliberately resolves back to the frozen
            # antecedent mention. It is not a new entity annotation.
            "mention_id": str(antecedent["mention_id"]),
            "surface": str(link["source_text"]),
            "mention_start": start,
            "mention_end": end,
            "canonical_id": str(antecedent["canonical_id"]),
            "type": str(antecedent["type"]),
            "relation_reference_proxy": True,
            "reference_antecedent_occurrence_key": str(
                antecedent["occurrence_key"]
            ),
        })
    return occurrences


def _enrich_source_structure(
    content: str,
    occurrences: list[dict[str, Any]],
) -> None:
    """Attach layout fields available to the production source mapper."""

    line_spans: list[tuple[int, int]] = []
    cursor = 0
    for line in content.splitlines(keepends=True):
        line_spans.append((cursor, cursor + len(line)))
        cursor += len(line)
    if cursor < len(content) or not line_spans:
        line_spans.append((cursor, len(content)))
    table_block_by_line: dict[int, tuple[str, str]] = {}
    table_index = -1
    in_table = False
    row_index = 0
    for line_index, (start, end) in enumerate(line_spans):
        text = content[start:end]
        is_table = text.count("|") >= 2
        if is_table and not in_table:
            table_index += 1
            row_index = 0
        if is_table:
            table_block_by_line[line_index] = (
                f"table-{table_index}", f"table-{table_index}-row-{row_index}",
            )
            row_index += 1
        in_table = is_table
    for occurrence in occurrences:
        mention_start = int(occurrence["mention_start"])
        mention_end = int(occurrence["mention_end"])
        line_match = next((
            (line_index, start, end)
            for line_index, (start, end) in enumerate(line_spans)
            if start <= mention_start < mention_end <= end
        ), None)
        if line_match is None:
            continue
        line_index, start, end = line_match
        occurrence.update({
            "scope_id": f"line:{start}:{end}",
            "scope_kind": "body",
            "scope_start": start,
            "scope_end": end,
            "source_role": "body",
            "relation_eligible": True,
        })
        table = table_block_by_line.get(line_index)
        if table is None:
            continue
        table_id, row_id = table
        line_text = content[start:end]
        pipe_offsets = [
            offset for offset, value in enumerate(line_text) if value == "|"
        ]
        cell_index = sum(
            start + offset < mention_start for offset in pipe_offsets
        )
        occurrence.update({
            "table_id": table_id,
            "row_id": row_id,
            "cell_id": f"{row_id}-cell-{cell_index}",
            "scope_kind": "table_cell",
            "source_role": "table_cell",
        })


def _deduplicate_frames(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    known = {
        (
            tuple(item["argument_occurrence_keys"]),
            re.sub(r"\W+", "", str(item["relation"]).casefold()),
            tuple(item["evidence_span"]),
        )
        for item in existing
    }
    result = list(existing)
    for item in incoming:
        key = (
            tuple(item["argument_occurrence_keys"]),
            re.sub(r"\W+", "", str(item["relation"]).casefold()),
            tuple(item["evidence_span"]),
        )
        if key not in known:
            known.add(key)
            result.append(item)
    return result


def _oracle_schema(
    document_id: str,
    assertions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build an explicitly labelled Gold Schema for an upper-bound audit."""

    entity_rows = _read("annotations/entities.json")["documents"][document_id]
    type_by_canonical: dict[str, set[str]] = {}
    for row in entity_rows:
        type_by_canonical.setdefault(
            str(row["canonical_id"]), set(),
        ).add(str(row["type"]))
    document_assertions = [
        item for item in assertions
        if str(item["document_id"]) == document_id
    ]
    by_predicate: dict[str, list[dict[str, Any]]] = {}
    for assertion in document_assertions:
        by_predicate.setdefault(
            str(assertion["predicate"]), [],
        ).append(assertion)
    structural_contracts = {
        "mapped_to": (
            ["table_header_assignment"], ["Requirement"],
        ),
        "derived_from": (
            ["table_header_assignment"], ["Source"],
        ),
        "used_for": (
            ["table_header_assignment"], ["Purpose"],
        ),
        "checks": (
            ["table_header_assignment"], ["Responsibility"],
        ),
        "submits": (
            ["table_header_assignment"], ["Responsibility"],
        ),
        "applies_to": (["line_identifier_binding"], []),
        "part_of": (
            ["list_parent_child", "section_label_parent_child"], [],
        ),
    }
    semantic_overrides = {
        "mapped_to": (
            "The subject domain item is mapped or associated to the object "
            "requirement/identifier. In a mapping artifact that links X to Y, "
            "X is the subject and Y is the object; the artifact is not either role."
        ),
        "links_to": (
            "The subject is the map, index, connector, or linking artifact and "
            "the object is one endpoint that artifact explicitly links."
        ),
        "submitted_to": (
            "The subject is submitted content or the action theme; the object is "
            "its explicit recipient/destination. The submitting actor is never the "
            "object of submitted_to."
        ),
        "part_of": (
            "The subject is a part/member/child and the object is its containing "
            "whole, even when source wording states that the whole contains the part."
        ),
        "listed_in": (
            "The subject is the listed item and the object is the list or table "
            "that explicitly lists it."
        ),
    }
    # Expert Gold relation contracts live in the benchmark ontology. These are
    # fixture annotations, not production predicate branches. The runtime
    # consumes the same generic policy fields from any logical Schema.
    relation_policies = dict(
        _read("ontology.json").get("relation_policies") or {}
    )
    semantic_conflicts = {
        "about": ["used_for"],
        "used_for": ["about"],
        "provides": ["supports"],
        "supports": ["provides"],
    }
    relation_types: list[dict[str, Any]] = []
    for predicate, rows in sorted(by_predicate.items()):
        domains = sorted({
            entity_type
            for row in rows
            for entity_type in type_by_canonical.get(
                str(row["subject_canonical_id"]), {"*"},
            )
        })
        ranges = sorted({
            entity_type
            for row in rows
            for entity_type in type_by_canonical.get(
                str(row["object_canonical_id"]), {"*"},
            )
        })
        examples = []
        for row in rows[:3]:
            subject_text = str(row["evidence_text"])[
                int(row["subject_span"][0]) - int(row["evidence_span"][0]):
                int(row["subject_span"][1]) - int(row["evidence_span"][0])
            ]
            object_text = str(row["evidence_text"])[
                int(row["object_span"][0]) - int(row["evidence_span"][0]):
                int(row["object_span"][1]) - int(row["evidence_span"][0])
            ]
            examples.append(
                f"{subject_text} -> {object_text}: {row['evidence_text']}"
            )
        structure_kinds, role_labels = structural_contracts.get(
            predicate, ([], []),
        )
        phrase = predicate.replace("_", " ")
        semantic_core = semantic_overrides.get(
            predicate, f"The subject {phrase} the object.",
        )
        relation_policy = dict(relation_policies.get(predicate) or {})
        relation_types.append({
            "name": predicate,
            "description": (
                f"The subject {phrase} the object in the exact direction "
                f"shown by these source-grounded examples: "
                + " | ".join(examples)
            ),
            "semantic_definition": (
                f"{semantic_core} Required direction and "
                f"role examples: " + " | ".join(examples)
            ),
            "source_types": domains or ["*"],
            "target_types": ranges or ["*"],
            "domain_types": domains or ["*"],
            "range_types": ranges or ["*"],
            "direction_description": (
                "Use the ordered source-to-target direction shown by the "
                "examples; reverse whole-first wording when necessary."
            ),
            "subject_role_description": (
                f"The source entity that is the subject of {phrase}."
            ),
            "object_role_description": (
                f"The target entity that is the object of {phrase}."
            ),
            "symmetric": predicate in {"differs_from", "agrees_with"},
            "inverse_predicate": "",
            "canonical_orientation": (
                "symmetric"
                if predicate in {"differs_from", "agrees_with"}
                else "forward"
            ),
            "publish_mode": "auto",
            "reflexive": False,
            "reciprocal_allowed": False,
            "allowed_context_kinds": [
                "paragraph", "sentence", "table_row", "table_cell",
                "list_item", "code",
            ],
            "allowed_source_roles": [
                "body", "list_item", "table_row", "table_cell",
            ],
            "scope_policy": {
                "require_same_atomic_scope": False,
                "allow_cross_scope_with_explicit_evidence": True,
            },
            "required_evidence_roles": [
                "subject", "predicate_cue", "object",
            ],
            "positive_examples": examples,
            "negative_examples": [],
            "positive_cues": [phrase],
            "negative_cues": [],
            "anti_patterns": [],
            "optional_soft_cues": [phrase],
            "structural_evidence_kinds": structure_kinds,
            "structural_role_labels": role_labels,
            "source_role_pairs": list(
                relation_policy.get("source_role_pairs") or []
            ),
            "qualifier_schema": {},
            "identity_qualifier_keys": [],
            "conflicts_with": list(semantic_conflicts.get(predicate, [])),
            "negation_policy": str(
                relation_policy.get("negation_policy") or "reject_negated"
            ),
            "modality_policy": str(
                relation_policy.get("modality_policy") or "require_asserted"
            ),
        })
    return {
        "version": f"gold-oracle-schema:{document_id}:v1",
        "relation_types": relation_types,
    }


def _oracle_final_verify(
    *,
    client: OllamaClient,
    model: str,
    content: str,
    document_id: str,
    inventory: dict[str, Any],
    occurrences: list[dict[str, Any]],
    logical_schema: dict[str, Any],
    bindings: list[dict[str, Any]],
    gold: set[tuple[str, str, str]],
    negative_claims: list[dict[str, Any]],
    force_single_verifier: bool = False,
) -> dict[str, Any]:
    """Exercise the production validator and ordered verifier on oracle bindings."""

    frames = list(inventory.get("frames") or ())
    occurrence_by_key = {
        str(item.get("occurrence_key") or ""): item for item in occurrences
    }
    schema = relation_v2.normalize_predicate_schema(logical_schema)
    chunk = {
        "id": 1,
        "document_id": 1,
        "revision_id": f"gold-benchmark:{document_id}",
        "chunk_key": document_id,
        "content": content,
        "shadow_context_id": document_id,
        "shadow_context_kind": "gold_benchmark",
        "shadow_context_quality": "expert_gold",
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "shadow_structure": {},
        "metadata": {},
    }
    plan = relation_v2.plan_pair_opportunities(
        project_id=1,
        document_revision_id=str(chunk["revision_id"]),
        chunk_key=str(chunk["chunk_key"]),
        content=content,
        mentions=occurrences,
        logical_schema=logical_schema,
        run_id=f"gold-run:{document_id}",
        candidate_policy_version="gold-source-projection-v1",
        max_pair_opportunities=Config.relation_max_candidates_per_chunk,
        sentence_window=Config.relation_sentence_window,
        max_char_distance=Config.relation_max_char_distance,
    )
    opportunities_by_pair = {
        str(item.pair_identity): item for item in plan.opportunities
    }
    # Mirror the production source-first admission path. Binary grounded
    # frames can promote heuristic distance/scope rejections and widen their
    # immutable context to the exact source evidence. N-ary projections use
    # the ordinary all-pair ledger created above.
    direct_opportunities = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        opportunity = ShadowRelationV2Runtime._grounded_open_fact_opportunity(
            run_id=f"gold-run:{document_id}",
            project_id=1,
            chunk=chunk,
            frame=frame,
            mentions=occurrences,
            logical_schema=logical_schema,
        )
        if opportunity is None:
            continue
        existing = opportunities_by_pair.get(str(opportunity.pair_identity))
        if existing is not None:
            opportunity = replace(
                opportunity,
                context_span=(
                    min(existing.context_span[0], opportunity.context_span[0]),
                    max(existing.context_span[1], opportunity.context_span[1]),
                ),
            )
        opportunities_by_pair[str(opportunity.pair_identity)] = opportunity
        direct_opportunities += 1

    projected = relation_v2.source_frame_projection_hypotheses(
        inventory,
        {"coverage_complete": True, "bindings": bindings},
        occurrences=occurrences,
        opportunities_by_pair=opportunities_by_pair,
        logical_schema=logical_schema,
        classifier_model_revision=model,
    )
    conflict_groups = relation_v2.conflicting_source_hypothesis_groups(
        projected, logical_schema=logical_schema,
    )
    conflict_members: set[str] = set()
    conflict_winners: set[str] = set()
    conflict_calls = 0
    conflict_failures = 0
    conflict_none = 0
    conflict_diagnostics: list[dict[str, Any]] = []
    for group in conflict_groups:
        opportunity = opportunities_by_pair.get(str(group[0].pair_identity))
        if opportunity is None:
            conflict_members.update(item.hypothesis_id for item in group)
            conflict_failures += 1
            continue
        conflict_members.update(item.hypothesis_id for item in group)
        try:
            conflict_calls += 1
            raw_conflict = client.json_chat(
                model=model,
                system=(
                    "You are an independent contrastive relation-policy judge. "
                    "Choose at most one Schema-declared conflicting relation for "
                    "the frozen ordered source frame and return strict JSON."
                ),
                prompt=relation_v2.source_predicate_conflict_prompt(
                    group,
                    opportunity,
                    content=content,
                    logical_schema=logical_schema,
                ),
                schema=relation_v2.source_predicate_conflict_response_schema(
                    group, opportunity, content=content,
                ),
                temperature=0.0,
                context_budget_tokens=Config.ollama_default_context_budget,
                max_output_tokens=1024,
                safety_margin_tokens=Config.relation_classifier_safety_margin,
            )
            winner = relation_v2.parse_source_predicate_conflict(
                raw_conflict,
                hypotheses=group,
                opportunity=opportunity,
                content=content,
            )
            if winner:
                conflict_winners.add(next(
                    item.hypothesis_id
                    for item in group if item.predicate == winner
                ))
            else:
                conflict_none += 1
        except Exception as exc:
            # A Schema-declared contradiction cannot safely publish when its
            # contrastive adjudication is malformed or unavailable.
            conflict_failures += 1
            conflict_diagnostics.append({
                "predicates": [item.predicate for item in group],
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "raw_response": str(
                    getattr(client, "last_raw_response_text", "") or ""
                )[:4000],
            })
    projected_before_conflict = len(projected)
    projected = tuple(
        item for item in projected
        if item.hypothesis_id not in conflict_members
        or item.hypothesis_id in conflict_winners
    )
    claims: list[dict[str, Any]] = []
    seen_hypotheses: set[str] = set()
    for hypothesis in projected:
        frame_index = int(
            (hypothesis.predicate_metadata or {}).get(
                "source_frame_index", -1,
            )
        )
        if not 0 <= frame_index < len(frames):
            continue
        subject = occurrence_by_key.get(
            hypothesis.subject_occurrence_key
        )
        object_ = occurrence_by_key.get(hypothesis.object_occurrence_key)
        opportunity = opportunities_by_pair.get(
            str(hypothesis.pair_identity)
        )
        if subject is None or object_ is None or opportunity is None:
            continue
        if hypothesis.hypothesis_id in seen_hypotheses:
            continue
        seen_hypotheses.add(hypothesis.hypothesis_id)
        validation = relation_v2.validate_hypothesis(
            hypothesis,
            opportunity,
            content=content,
            logical_schema=logical_schema,
        )
        claims.append({
            "hypothesis": hypothesis,
            "opportunity": opportunity,
            "validation": validation,
            "triple": (
                str(subject.get("canonical_id") or ""),
                hypothesis.predicate,
                str(object_.get("canonical_id") or ""),
            ),
            "frame_index": frame_index,
        })

    batch_decisions: dict[str, dict[str, Any]] = {}
    verifier_calls = 0
    batch_failures = 0
    batch_failure_diagnostics: list[dict[str, Any]] = []
    batch_ungrounded_decisions = 0
    batch_eligible_claims = [
        item for item in claims
        if not (
            str(
                (item["hypothesis"].predicate_metadata or {}).get(
                    "source_binding_source"
                ) or ""
            ).upper() == "DETERMINISTIC_SOURCE_STRUCTURE"
            and str(
                (item["hypothesis"].predicate_metadata or {}).get(
                    "source_structure_kind"
                ) or ""
            )
        )
    ]
    batch_verifier_structure_bypassed = (
        len(claims) - len(batch_eligible_claims)
    )
    pending = [] if force_single_verifier else [
        batch_eligible_claims[
            index:index + int(Config.relation_verifier_batch_size)
        ]
        for index in range(
            0, len(batch_eligible_claims),
            int(Config.relation_verifier_batch_size),
        )
    ]
    while pending:
        batch = pending.pop(0)
        valid_batch = [
            item for item in batch
            if item["validation"].status == relation_v2.PASS
        ]
        if not valid_batch:
            continue
        rendered = relation_v2.batch_verifier_prompt(
            [
                (item["hypothesis"], item["opportunity"], content)
                for item in valid_batch
            ],
            logical_schema=logical_schema,
        )
        if (
            len(rendered) > int(Config.relation_verifier_batch_max_chars)
            and len(valid_batch) > 1
        ):
            midpoint = max(1, len(batch) // 2)
            pending[0:0] = [batch[:midpoint], batch[midpoint:]]
            continue
        ids = [item["hypothesis"].hypothesis_id for item in valid_batch]
        try:
            verifier_calls += 1
            raw = client.json_chat(
                model=model,
                system=(
                    "You are an independent directed-relation evidence verifier. "
                    "Evaluate every supplied claim separately from its own frozen "
                    "context, Schema policy, endpoints, and evidence. Return exactly "
                    "the supplied JSON Schema with root decisions and all required "
                    "ordered-entailment fields."
                ),
                prompt=rendered,
                schema=relation_v2.batch_verifier_response_schema(ids),
                temperature=0.0,
                context_budget_tokens=Config.ollama_default_context_budget,
                max_output_tokens=max(4096, min(16384, len(ids) * 768)),
                safety_margin_tokens=Config.relation_classifier_safety_margin,
            )
            batch_decisions.update(
                relation_v2.parse_batch_verifier_response(
                    raw, hypothesis_ids=ids,
                )
            )
        except Exception as exc:
            batch_failures += 1
            batch_failure_diagnostics.append({
                "hypothesis_ids": ids,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "raw_response": str(
                    getattr(client, "last_raw_response_text", "") or ""
                )[:4000],
                "parsed_response": getattr(
                    client, "last_parsed_json_response", None,
                ),
            })
            # Preserve independently valid negative members, but never trust
            # a partial positive that omitted the full verifier contract.
            try:
                partial = relation_v2.parse_batch_verifier_response(
                    getattr(client, "last_parsed_json_response", None)
                    or getattr(client, "last_raw_response_text", ""),
                    hypothesis_ids=ids,
                    allow_partial=True,
                )
            except Exception:
                partial = {}
            batch_decisions.update(partial)

    claim_by_hypothesis_id = {
        item["hypothesis"].hypothesis_id: item for item in claims
    }
    for hypothesis_id in list(batch_decisions):
        item = claim_by_hypothesis_id.get(hypothesis_id)
        if item is None:
            batch_decisions.pop(hypothesis_id, None)
            continue
        if not relation_v2.batch_verifier_decision_is_source_grounded(
            item["hypothesis"],
            item["opportunity"],
            batch_decisions[hypothesis_id],
            content=content,
            logical_schema=logical_schema,
        ):
            batch_ungrounded_decisions += 1
            batch_decisions.pop(hypothesis_id, None)

    focused_candidates: list[
        tuple[int, str, dict[str, Any]]
    ] = []
    source_competition_map = relation_v2.source_projection_competition_map(
        [item["hypothesis"] for item in claims],
        logical_schema=logical_schema,
    )
    source_group_audit_calls = 0
    source_group_audit_failures = 0
    source_group_audit_statuses: dict[str, str] = {}
    source_group_audit_diagnostics: list[dict[str, Any]] = []
    for component in (
        relation_v2.source_projection_competition_components(
            source_competition_map
        )
        if ENABLE_SOURCE_GROUP_AUDIT_EXPERIMENT else ()
    ):
        component_items = [
            claim_by_hypothesis_id[hypothesis_id]
            for hypothesis_id in component
            if (
                hypothesis_id in claim_by_hypothesis_id
                and claim_by_hypothesis_id[hypothesis_id][
                    "validation"
                ].status == relation_v2.PASS
            )
        ]
        if len(component_items) < 2:
            continue
        component_ids = [
            item["hypothesis"].hypothesis_id for item in component_items
        ]
        raw_group: Any = None
        group_error = ""
        try:
            source_group_audit_calls += 1
            raw_group = client.json_chat(
                model=model,
                system=(
                    "Jointly audit competing source-role bindings. Return "
                    "only JSON."
                ),
                prompt=relation_v2.source_projection_group_audit_prompt(
                    [
                        (item["hypothesis"], item["opportunity"])
                        for item in component_items
                    ],
                    content=content,
                    logical_schema=logical_schema,
                ),
                schema=(
                    relation_v2.source_projection_group_audit_response_schema(
                        component_ids
                    )
                ),
                temperature=0.0,
                context_budget_tokens=Config.ollama_default_context_budget,
                max_output_tokens=1024,
                safety_margin_tokens=(
                    Config.relation_classifier_safety_margin
                ),
            )
            parsed_group = relation_v2.parse_source_projection_group_audit(
                raw_group, hypothesis_ids=component_ids,
            )
            source_group_audit_statuses.update(parsed_group)
        except Exception as exc:
            source_group_audit_failures += 1
            group_error = f"{type(exc).__name__}: {exc}"[:1000]
            parsed_group = {}
        source_group_audit_diagnostics.append({
            "hypothesis_ids": component_ids,
            "statuses": parsed_group,
            "error": group_error,
            "raw_response": str(
                getattr(client, "last_raw_response_text", "") or ""
            )[:4000],
        })
    for item in claims:
        hypothesis = item["hypothesis"]
        decision = batch_decisions.get(hypothesis.hypothesis_id)
        if decision is None or item["validation"].status != relation_v2.PASS:
            continue
        review = relation_v2.source_batch_focused_review(
            hypothesis,
            decision,
            logical_schema=logical_schema,
            source_competition_risk=(
                hypothesis.hypothesis_id in source_competition_map
            ),
        )
        if review is not None:
            review = {
                **review,
                "competing_hypothesis_ids": list(
                    source_competition_map.get(
                        hypothesis.hypothesis_id, (),
                    )
                ),
            }
            focused_candidates.append((
                int(review["priority"]),
                str(hypothesis.hypothesis_id),
                dict(review),
            ))
    focused_review_plans = {
        hypothesis_id: review
        for _priority, hypothesis_id, review in sorted(
            focused_candidates, key=lambda value: (value[0], value[1]),
        )[:int(Config.relation_verifier_focused_review_max)]
    }
    for hypothesis_id, status in source_group_audit_statuses.items():
        if hypothesis_id in focused_review_plans:
            focused_review_plans[hypothesis_id] = {
                **focused_review_plans[hypothesis_id],
                "source_group_audit": True,
                "source_group_status": status,
            }
            continue
        decision = batch_decisions.get(hypothesis_id) or {}
        focused_review_plans[hypothesis_id] = {
            "kind": (
                "recovery"
                if str(decision.get("verdict") or "").upper()
                in {relation_v2.REJECTED, relation_v2.UNCERTAIN}
                else "confirmation"
            ),
            "reason": "SOURCE_GROUP_AUDIT",
            "priority": 1,
            "source_group_audit": True,
            "source_group_status": status,
            "competing_hypothesis_ids": list(
                source_competition_map.get(hypothesis_id, ())
            ),
        }
    focused_review_model_calls = 0
    focused_positive_confirmed = 0
    focused_positive_rejected = 0
    focused_negative_recovered = 0
    focused_negative_retained = 0
    source_role_audit_calls = 0
    source_role_audit_aligned = 0
    source_role_audit_conflicts = 0
    source_role_audit_uncertain = 0
    source_role_audit_failures = 0
    source_structure_audit_calls = 0
    source_structure_audit_aligned = 0
    source_structure_audit_conflicts = 0
    source_structure_audit_uncertain = 0
    source_structure_audit_failures = 0
    polarity_review_eligible = 0
    polarity_audit_calls = 0
    polarity_audit_failures = 0
    polarity_asserted = 0
    polarity_negated = 0
    polarity_uncertain = 0
    polarity_final_verifier_calls = 0
    polarity_recovered = 0
    polarity_diagnostics: list[dict[str, Any]] = []
    orientation_review_eligible = 0
    orientation_model_calls = 0
    orientation_model_failures = 0
    orientation_final_verifier_calls = 0
    orientation_recovered = 0
    orientation_diagnostics: list[dict[str, Any]] = []

    attempts: list[dict[str, Any]] = []
    supported: set[tuple[str, str, str]] = set()
    single_fallbacks = 0
    for item in claims:
        hypothesis = item["hypothesis"]
        attempt = None
        focused_plan = focused_review_plans.get(hypothesis.hypothesis_id)
        metadata = hypothesis.predicate_metadata or {}
        group_status = source_group_audit_statuses.get(
            hypothesis.hypothesis_id
        )
        if group_status in {"CONFLICT", "UNCERTAIN"}:
            attempt = (
                relation_v2.source_projection_group_audit_terminal_attempt(
                    hypothesis,
                    result=group_status,
                    verifier_model_revision=model,
                    parsed_response={
                        "projection_status": group_status,
                    },
                )
            )
        if (
            attempt is None
            and str(metadata.get("source_binding_source") or "").upper()
            == "DETERMINISTIC_SOURCE_STRUCTURE"
            and str(metadata.get("source_structure_kind") or "")
        ):
            structure_raw: Any = None
            structure_error = ""
            structure_result = "UNAVAILABLE"
            try:
                source_structure_audit_calls += 1
                structure_raw = client.json_chat(
                    model=model,
                    system=(
                        "Audit exact parser-proved source structure. Return "
                        "only JSON."
                    ),
                    prompt=relation_v2.source_structure_binding_audit_prompt(
                        hypothesis,
                        item["opportunity"],
                        content=content,
                        logical_schema=logical_schema,
                    ),
                    schema=(
                        relation_v2.source_structure_binding_audit_response_schema()
                    ),
                    temperature=0.0,
                    context_budget_tokens=(
                        Config.ollama_default_context_budget
                    ),
                    max_output_tokens=256,
                    safety_margin_tokens=(
                        Config.relation_classifier_safety_margin
                    ),
                )
                structure_result = (
                    relation_v2.parse_source_structure_binding_audit(
                        structure_raw
                    )
                )
            except Exception as exc:
                source_structure_audit_failures += 1
                structure_error = f"{type(exc).__name__}: {exc}"[:2000]
            if structure_result == "ALIGNED":
                source_structure_audit_aligned += 1
            elif structure_result == "CONFLICT":
                source_structure_audit_conflicts += 1
            elif structure_result == "UNCERTAIN":
                source_structure_audit_uncertain += 1
            attempt = (
                relation_v2.source_structure_binding_audit_terminal_attempt(
                    hypothesis,
                    item["opportunity"],
                    content=content,
                    logical_schema=logical_schema,
                    result=structure_result,
                    verifier_model_revision=model,
                    parsed_response=(
                        {"structure_status": structure_result}
                        if structure_result != "UNAVAILABLE" else {}
                    ),
                    raw_response_text=str(
                        getattr(client, "last_raw_response_text", "") or ""
                    ),
                    error_message=structure_error,
                )
            )
        if (
            attempt is None
            and
            focused_plan is not None
            and focused_plan.get("source_role_audit") is True
            and group_status is None
        ):
            competitor_hypotheses = tuple(
                claim_by_hypothesis_id[value]["hypothesis"]
                for value in focused_plan.get(
                    "competing_hypothesis_ids", ()
                )
                if value in claim_by_hypothesis_id
            )
            role_audit_raw: Any = None
            role_audit_error = ""
            role_audit_result = "UNAVAILABLE"
            try:
                source_role_audit_calls += 1
                role_audit_raw = client.json_chat(
                    model=model,
                    system=(
                        "Audit exact source roles. Return only JSON."
                    ),
                    prompt=relation_v2.source_projection_role_audit_prompt(
                        hypothesis,
                        item["opportunity"],
                        content=content,
                        logical_schema=logical_schema,
                        competing_projections=competitor_hypotheses,
                    ),
                    schema=(
                        relation_v2.source_projection_role_audit_response_schema()
                    ),
                    temperature=0.0,
                    context_budget_tokens=Config.ollama_default_context_budget,
                    max_output_tokens=256,
                    safety_margin_tokens=(
                        Config.relation_classifier_safety_margin
                    ),
                )
                role_audit_result = (
                    relation_v2.parse_source_projection_role_audit(
                        role_audit_raw
                    )
                )
            except Exception as exc:
                source_role_audit_failures += 1
                role_audit_error = f"{type(exc).__name__}: {exc}"[:2000]
            if role_audit_result == "ALIGNED":
                source_role_audit_aligned += 1
            elif relation_v2.source_role_audit_conflict_is_terminal(
                focused_plan
            ):
                if role_audit_result == "CONFLICT":
                    source_role_audit_conflicts += 1
                elif role_audit_result == "UNCERTAIN":
                    source_role_audit_uncertain += 1
                attempt = relation_v2.source_projection_role_audit_terminal_attempt(
                    hypothesis,
                    result=role_audit_result,
                    verifier_model_revision=model,
                    parsed_response=(
                        {"projection_status": role_audit_result}
                        if role_audit_result != "UNAVAILABLE" else {}
                    ),
                    raw_response_text=str(
                        getattr(client, "last_raw_response_text", "") or ""
                    ),
                    error_message=role_audit_error,
                )
            else:
                if role_audit_result == "CONFLICT":
                    source_role_audit_conflicts += 1
                elif role_audit_result == "UNCERTAIN":
                    source_role_audit_uncertain += 1
        if attempt is None:
            for attempt_no in range(
                1, Config.relation_shadow_max_attempts + 1,
            ):

                def verifier(
                    prompt: str, response_schema: dict[str, Any],
                ) -> Any:
                    nonlocal verifier_calls, single_fallbacks
                    nonlocal focused_review_model_calls
                    cached = (
                        batch_decisions.get(hypothesis.hypothesis_id)
                        if (
                            attempt_no == 1
                            and hypothesis.hypothesis_id
                            not in focused_review_plans
                        ) else None
                    )
                    if cached is not None:
                        return cached
                    verifier_calls += 1
                    single_fallbacks += 1
                    if hypothesis.hypothesis_id in focused_review_plans:
                        focused_review_model_calls += 1
                    return client.json_chat(
                        model=model,
                        system=(
                            "You are an independent directed-relation evidence "
                            "verifier. Judge only the frozen ordered claim and "
                            "return exactly the supplied JSON Schema."
                        ),
                        prompt=prompt,
                        schema=response_schema,
                        temperature=0.0,
                        context_budget_tokens=(
                            Config.ollama_default_context_budget
                        ),
                        max_output_tokens=4096,
                        safety_margin_tokens=(
                            Config.relation_classifier_safety_margin
                        ),
                    )

                attempt = relation_v2.verify_hypothesis_lean(
                    hypothesis,
                    item["validation"],
                    item["opportunity"],
                    content=content,
                    logical_schema=logical_schema,
                    verifier=verifier,
                    verifier_model_revision=model,
                    attempt_no=attempt_no,
                )
                if attempt.verdict != relation_v2.RETRYABLE_FAILED:
                    break
        assert attempt is not None
        triple = item["triple"]
        if attempt.verdict == relation_v2.SUPPORTED:
            supported.add(triple)
        if focused_plan is not None:
            if focused_plan["kind"] == "confirmation":
                if attempt.verdict == relation_v2.SUPPORTED:
                    focused_positive_confirmed += 1
                else:
                    focused_positive_rejected += 1
            elif focused_plan["kind"] == "recovery":
                if attempt.verdict == relation_v2.SUPPORTED:
                    focused_negative_recovered += 1
                else:
                    focused_negative_retained += 1
        attempts.append({
            "triple": {
                "subject": triple[0], "predicate": triple[1], "object": triple[2],
            },
            "frame_index": item["frame_index"],
            "validation": item["validation"].status,
            "validation_reason": item["validation"].reason_code,
            "verdict": attempt.verdict,
            "reason_code": attempt.reason_code,
            "forward_entailment": attempt.forward_entailment,
            "reverse_entailment": attempt.reverse_entailment,
            "subject_role_supported": attempt.subject_role_supported,
            "object_role_supported": attempt.object_role_supported,
            "error_message": attempt.error_message,
            "raw_response_text": (
                str(getattr(client, "last_raw_response_text", "") or "")
                if attempt.error_message else ""
            ),
        })
        orientation_route = route_secondary_attempt(
            attempt.reason_code,
            verdict=attempt.verdict,
            forward_entailment=attempt.forward_entailment,
            reverse_entailment=attempt.reverse_entailment,
            predicate=hypothesis.predicate,
            validation_status=item["validation"].status,
        )
        if (
            orientation_route.route == ROUTE_ROLE_RETRY
            and orientation_review_eligible
            < int(Config.relation_secondary_max_candidates)
        ):
            orientation_review_eligible += 1
            left_surface = content[
                item["opportunity"].left_span[0]:
                item["opportunity"].left_span[1]
            ]
            right_surface = content[
                item["opportunity"].right_span[0]:
                item["opportunity"].right_span[1]
            ]
            allowed_metadata = [
                value
                for value in item["opportunity"].allowed_predicate_metadata
                if str(value.get("name") or "") == hypothesis.predicate
            ]
            try:
                orientation_model_calls += 1
                orientation_raw = client.json_chat(
                    model=model,
                    system=(
                        "You are an independent secondary relation evidence "
                        "recovery judge."
                    ),
                    prompt=secondary_recovery_prompt(
                        route=orientation_route,
                        content=content,
                        context_span=item["opportunity"].context_span,
                        left_span=item["opportunity"].left_span,
                        right_span=item["opportunity"].right_span,
                        left_surface=left_surface,
                        right_surface=right_surface,
                        left_type=item["opportunity"].left_type,
                        right_type=item["opportunity"].right_type,
                        allowed_predicate_metadata=allowed_metadata,
                        structural_contract={
                            "structure_admission": str(
                                item["opportunity"].provenance.get(
                                    "structure_admission"
                                ) or ""
                            ),
                            "hierarchical_parent_child": bool(
                                item["opportunity"].provenance.get(
                                    "hierarchical_list_parent_child"
                                )
                            ),
                            "section_label_parent_child": bool(
                                item["opportunity"].provenance.get(
                                    "hierarchical_section_label_parent_child"
                                )
                            ),
                            "source_roles": list(
                                item["opportunity"].provenance.get(
                                    "source_roles"
                                ) or []
                            ),
                            "required_atomic_scope": list(
                                item["opportunity"].provenance.get(
                                    "required_atomic_scope"
                                ) or []
                            ),
                        },
                        orientation_only=True,
                        preferred_predicate=hypothesis.predicate,
                    ),
                    schema=secondary_response_schema([
                        hypothesis.predicate
                    ]),
                    temperature=0.0,
                    context_budget_tokens=(
                        Config.ollama_default_context_budget
                    ),
                    max_output_tokens=1024,
                    safety_margin_tokens=(
                        Config.relation_classifier_safety_margin
                    ),
                )
                orientation_decision = parse_secondary_decision(
                    orientation_raw,
                    allowed_predicates=[hypothesis.predicate],
                    context_span=item["opportunity"].context_span,
                    left_span=item["opportunity"].left_span,
                    right_span=item["opportunity"].right_span,
                    content_length=len(content),
                    left_surface=left_surface,
                    right_surface=right_surface,
                    required_predicate=hypothesis.predicate,
                )
            except Exception as exc:
                orientation_model_failures += 1
                orientation_diagnostics.append({
                    "triple": {
                        "subject": triple[0],
                        "predicate": triple[1],
                        "object": triple[2],
                    },
                    "stage": "secondary_orientation_candidate",
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                    "raw_response": str(
                        getattr(client, "last_raw_response_text", "") or ""
                    )[:4000],
                })
                orientation_decision = None
            if (
                orientation_decision is not None
                and orientation_decision.decision == SECONDARY_RELATION
            ):
                side_to_occurrence = {
                    "LEFT": item["opportunity"].left_occurrence_key,
                    "RIGHT": item["opportunity"].right_occurrence_key,
                }
                secondary_hypothesis = replace(
                    hypothesis,
                    hypothesis_id="benchmark_orientation_" + hashlib.sha256(
                        (
                            f"{hypothesis.hypothesis_id}|"
                            f"{orientation_decision.subject_side}|"
                            f"{orientation_decision.object_side}"
                        ).encode("utf-8")
                    ).hexdigest(),
                    subject_occurrence_key=side_to_occurrence[
                        orientation_decision.subject_side
                    ],
                    object_occurrence_key=side_to_occurrence[
                        orientation_decision.object_side
                    ],
                    evidence_spans=tuple(
                        relation_v2.EvidenceSpan(start=start, end=end)
                        for start, end in orientation_decision.evidence_spans
                    ),
                    classifier_policy_version=SECONDARY_PASS_VERSION,
                    classifier_prompt_version=SECONDARY_PROMPT_VERSION,
                    predicate_metadata={
                        **(hypothesis.predicate_metadata or {}),
                        "candidate_source": "SECONDARY_REVIEW",
                        "route": ROUTE_ROLE_RETRY,
                        "primary_hypothesis_id": hypothesis.hypothesis_id,
                    },
                    status="SECONDARY_PROPOSED",
                )
                secondary_validation = relation_v2.validate_hypothesis(
                    secondary_hypothesis,
                    item["opportunity"],
                    content=content,
                    logical_schema=logical_schema,
                )

                def orientation_final_verifier(
                    prompt: str, response_schema: dict[str, Any],
                ) -> Any:
                    nonlocal verifier_calls, single_fallbacks
                    nonlocal orientation_final_verifier_calls
                    verifier_calls += 1
                    single_fallbacks += 1
                    orientation_final_verifier_calls += 1
                    return client.json_chat(
                        model=model,
                        system=(
                            "You are an independent directed-relation evidence "
                            "verifier. Judge only the frozen ordered claim and "
                            "return exactly the supplied JSON Schema."
                        ),
                        prompt=prompt,
                        schema=response_schema,
                        temperature=0.0,
                        context_budget_tokens=(
                            Config.ollama_default_context_budget
                        ),
                        max_output_tokens=4096,
                        safety_margin_tokens=(
                            Config.relation_classifier_safety_margin
                        ),
                    )

                secondary_attempt = relation_v2.verify_hypothesis_lean(
                    secondary_hypothesis,
                    secondary_validation,
                    item["opportunity"],
                    content=content,
                    logical_schema=logical_schema,
                    verifier=orientation_final_verifier,
                    verifier_model_revision=model,
                    attempt_no=1,
                    independent_orientation_audit="FORWARD",
                )
                secondary_subject = occurrence_by_key.get(
                    secondary_hypothesis.subject_occurrence_key, {}
                )
                secondary_object = occurrence_by_key.get(
                    secondary_hypothesis.object_occurrence_key, {}
                )
                secondary_triple = (
                    str(secondary_subject.get("canonical_id") or ""),
                    secondary_hypothesis.predicate,
                    str(secondary_object.get("canonical_id") or ""),
                )
                if secondary_attempt.verdict == relation_v2.SUPPORTED:
                    already_supported = secondary_triple in supported
                    supported.add(secondary_triple)
                    if secondary_triple in gold and not already_supported:
                        orientation_recovered += 1
                attempts.append({
                    "stage": "secondary_orientation_final_verifier",
                    "triple": {
                        "subject": secondary_triple[0],
                        "predicate": secondary_triple[1],
                        "object": secondary_triple[2],
                    },
                    "primary_triple": {
                        "subject": triple[0],
                        "predicate": triple[1],
                        "object": triple[2],
                    },
                    "frame_index": item["frame_index"],
                    "validation": secondary_validation.status,
                    "validation_reason": secondary_validation.reason_code,
                    "verdict": secondary_attempt.verdict,
                    "reason_code": secondary_attempt.reason_code,
                    "forward_entailment": (
                        secondary_attempt.forward_entailment
                    ),
                    "reverse_entailment": (
                        secondary_attempt.reverse_entailment
                    ),
                    "subject_role_supported": (
                        secondary_attempt.subject_role_supported
                    ),
                    "object_role_supported": (
                        secondary_attempt.object_role_supported
                    ),
                    "error_message": secondary_attempt.error_message,
                })
        if not relation_v2.source_frame_polarity_review_eligible(
            hypothesis, attempt,
        ):
            continue
        polarity_review_eligible += 1
        metadata = hypothesis.predicate_metadata or {}
        try:
            raw_frame_span = metadata.get("source_frame_evidence_span")
            if not (
                isinstance(raw_frame_span, (list, tuple))
                and len(raw_frame_span) == 2
            ):
                raise ValueError("polarity frame span missing")
            frame_span = (
                int(raw_frame_span[0]), int(raw_frame_span[1]),
            )
            if not (
                frame_span[0] <= item["opportunity"].left_span[0]
                < item["opportunity"].left_span[1] <= frame_span[1]
                and frame_span[0] <= item["opportunity"].right_span[0]
                < item["opportunity"].right_span[1] <= frame_span[1]
            ):
                raise ValueError("polarity frame does not cover both endpoints")
            polarity_audit_calls += 1
            polarity_raw = client.json_chat(
                model=model,
                system=(
                    "You are an independent grammatical negation-scope auditor. "
                    "Return only the supplied one-field JSON schema."
                ),
                prompt=source_frame_polarity_prompt(
                    content=content,
                    context_span=item["opportunity"].context_span,
                    frame_span=frame_span,
                    frame_text=str(
                        metadata.get("source_frame_evidence_text") or ""
                    ),
                    open_relation=str(
                        metadata.get("source_frame_relation") or ""
                    ),
                ),
                schema=source_frame_polarity_response_schema(),
                temperature=0.0,
                context_budget_tokens=Config.ollama_default_context_budget,
                max_output_tokens=256,
                safety_margin_tokens=Config.relation_classifier_safety_margin,
            )
            polarity = parse_source_frame_polarity(polarity_raw)
        except Exception as exc:
            polarity_audit_failures += 1
            polarity_diagnostics.append({
                "triple": {
                    "subject": triple[0],
                    "predicate": triple[1],
                    "object": triple[2],
                },
                "stage": "polarity_audit",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "raw_response": str(
                    getattr(client, "last_raw_response_text", "") or ""
                )[:4000],
            })
            continue
        if polarity == "NEGATED":
            polarity_negated += 1
            continue
        if polarity == "UNCERTAIN":
            polarity_uncertain += 1
            continue
        polarity_asserted += 1
        secondary_hypothesis = replace(
            hypothesis,
            hypothesis_id="benchmark_polarity_" + hashlib.sha256(
                f"{hypothesis.hypothesis_id}|{frame_span}".encode("utf-8")
            ).hexdigest(),
            evidence_spans=(relation_v2.EvidenceSpan(
                start=frame_span[0], end=frame_span[1],
            ),),
            classifier_policy_version=SECONDARY_POLARITY_PROMPT_VERSION,
            classifier_prompt_version=SECONDARY_POLARITY_PROMPT_VERSION,
            predicate_metadata={
                **metadata,
                "candidate_source": "SECONDARY_REVIEW",
                "route": "POLARITY_REVIEW",
                "source_frame_polarity_audit": "ASSERTED",
                "source_frame_polarity_audit_version": (
                    SECONDARY_POLARITY_PROMPT_VERSION
                ),
            },
            status="SECONDARY_PROPOSED",
        )
        secondary_validation = relation_v2.validate_hypothesis(
            secondary_hypothesis,
            item["opportunity"],
            content=content,
            logical_schema=logical_schema,
        )

        def polarity_final_verifier(
            prompt: str, response_schema: dict[str, Any],
        ) -> Any:
            nonlocal verifier_calls, single_fallbacks
            nonlocal polarity_final_verifier_calls
            verifier_calls += 1
            single_fallbacks += 1
            polarity_final_verifier_calls += 1
            return client.json_chat(
                model=model,
                system=(
                    "You are an independent directed-relation evidence verifier. "
                    "Judge only the frozen ordered claim and return exactly the "
                    "supplied JSON Schema."
                ),
                prompt=prompt,
                schema=response_schema,
                temperature=0.0,
                context_budget_tokens=Config.ollama_default_context_budget,
                max_output_tokens=4096,
                safety_margin_tokens=Config.relation_classifier_safety_margin,
            )

        secondary_attempt = relation_v2.verify_hypothesis_lean(
            secondary_hypothesis,
            secondary_validation,
            item["opportunity"],
            content=content,
            logical_schema=logical_schema,
            verifier=polarity_final_verifier,
            verifier_model_revision=model,
            attempt_no=1,
        )
        if secondary_attempt.verdict == relation_v2.SUPPORTED:
            supported.add(triple)
            polarity_recovered += 1
        attempts.append({
            "stage": "secondary_polarity_final_verifier",
            "triple": {
                "subject": triple[0], "predicate": triple[1],
                "object": triple[2],
            },
            "frame_index": item["frame_index"],
            "validation": secondary_validation.status,
            "validation_reason": secondary_validation.reason_code,
            "verdict": secondary_attempt.verdict,
            "reason_code": secondary_attempt.reason_code,
            "forward_entailment": secondary_attempt.forward_entailment,
            "reverse_entailment": secondary_attempt.reverse_entailment,
            "subject_role_supported": (
                secondary_attempt.subject_role_supported
            ),
            "object_role_supported": secondary_attempt.object_role_supported,
            "error_message": secondary_attempt.error_message,
        })
    negative = {
        (
            str(item["subject_canonical_id"]),
            str(item["predicate"]),
            str(item["object_canonical_id"]),
        )
        for item in negative_claims
        if str(item["document_id"]) == document_id
    }
    hits = supported & gold
    return {
        "planned_opportunities": len(opportunities_by_pair),
        "direct_source_opportunities": direct_opportunities,
        "projected_hypotheses": len(projected),
        "projected_hypotheses_before_conflict": projected_before_conflict,
        "predicate_conflict_groups": len(conflict_groups),
        "predicate_conflict_calls": conflict_calls,
        "predicate_conflict_failures": conflict_failures,
        "predicate_conflict_none": conflict_none,
        "predicate_conflict_diagnostics": conflict_diagnostics,
        "predicate_conflict_suppressed": (
            projected_before_conflict - len(projected)
        ),
        "force_single_verifier": bool(force_single_verifier),
        "claims": len(claims),
        "verifier_calls": verifier_calls,
        "batch_failures": batch_failures,
        "batch_failure_diagnostics": batch_failure_diagnostics,
        "batch_ungrounded_decisions": batch_ungrounded_decisions,
        "batch_verifier_structure_bypassed": (
            batch_verifier_structure_bypassed
        ),
        "single_fallbacks": single_fallbacks,
        "focused_review_eligible": len(focused_candidates),
        "focused_review_selected": len(focused_review_plans),
        "focused_review_model_calls": focused_review_model_calls,
        "focused_review_confirmations": sum(
            item["kind"] == "confirmation"
            for item in focused_review_plans.values()
        ),
        "focused_review_recoveries": sum(
            item["kind"] == "recovery"
            for item in focused_review_plans.values()
        ),
        "focused_review_batch_positive_confirmed": focused_positive_confirmed,
        "focused_review_batch_positive_rejected": focused_positive_rejected,
        "focused_review_batch_negative_recovered": focused_negative_recovered,
        "focused_review_batch_negative_retained": focused_negative_retained,
        "source_role_audit_calls": source_role_audit_calls,
        "source_role_audit_aligned": source_role_audit_aligned,
        "source_role_audit_conflicts": source_role_audit_conflicts,
        "source_role_audit_uncertain": source_role_audit_uncertain,
        "source_role_audit_failures": source_role_audit_failures,
        "source_group_audit_calls": source_group_audit_calls,
        "source_group_audit_failures": source_group_audit_failures,
        "source_group_audit_statuses": source_group_audit_statuses,
        "source_group_audit_diagnostics": source_group_audit_diagnostics,
        "source_structure_audit_calls": source_structure_audit_calls,
        "source_structure_audit_aligned": source_structure_audit_aligned,
        "source_structure_audit_conflicts": (
            source_structure_audit_conflicts
        ),
        "source_structure_audit_uncertain": (
            source_structure_audit_uncertain
        ),
        "source_structure_audit_failures": (
            source_structure_audit_failures
        ),
        "polarity_review_eligible": polarity_review_eligible,
        "polarity_audit_calls": polarity_audit_calls,
        "polarity_audit_failures": polarity_audit_failures,
        "polarity_asserted": polarity_asserted,
        "polarity_negated": polarity_negated,
        "polarity_uncertain": polarity_uncertain,
        "polarity_final_verifier_calls": polarity_final_verifier_calls,
        "polarity_recovered": polarity_recovered,
        "polarity_diagnostics": polarity_diagnostics,
        "orientation_review_eligible": orientation_review_eligible,
        "orientation_model_calls": orientation_model_calls,
        "orientation_model_failures": orientation_model_failures,
        "orientation_final_verifier_calls": (
            orientation_final_verifier_calls
        ),
        "orientation_recovered": orientation_recovered,
        "orientation_diagnostics": orientation_diagnostics,
        "focused_review_plans": [
            {
                "hypothesis_id": item["hypothesis"].hypothesis_id,
                "triple": {
                    "subject": item["triple"][0],
                    "predicate": item["triple"][1],
                    "object": item["triple"][2],
                },
                "frame_index": item["frame_index"],
                **focused_review_plans[item["hypothesis"].hypothesis_id],
            }
            for item in claims
            if item["hypothesis"].hypothesis_id in focused_review_plans
        ],
        "supported_unique_facts": len(supported),
        "exact_hits": len(hits),
        "exact_recall": len(hits) / len(gold) if gold else None,
        "exact_precision": len(hits) / len(supported) if supported else None,
        "negative_claim_leaks": len(supported & negative),
        "misses": [
            {"subject": key[0], "predicate": key[1], "object": key[2]}
            for key in sorted(gold - supported)
        ],
        "non_gold_supported": [
            {"subject": key[0], "predicate": key[1], "object": key[2]}
            for key in sorted(supported - gold)
        ],
        "attempts": attempts,
    }


def _oracle_bind_frames(
    *,
    client: OllamaClient,
    model: str,
    content: str,
    inventory: dict[str, Any],
    occurrences: list[dict[str, Any]],
    logical_schema: dict[str, Any],
    gold: set[tuple[str, str, str]],
    negative_claims: list[dict[str, Any]],
    document_id: str,
    binding_rounds: int,
    recall_shortlist: bool,
    final_verifier: bool,
    force_single_verifier: bool,
) -> dict[str, Any]:
    deterministic = deterministic_source_structure_schema_bindings(
        inventory, logical_schema=logical_schema,
    )
    bindings = list(deterministic.get("bindings") or ())
    calls: list[dict[str, Any]] = []
    for round_index in range(1 + max(0, int(binding_rounds))):
        started = time.perf_counter()
        prompt = source_frame_schema_binding_prompt(
            inventory,
            logical_schema=logical_schema,
            already_found_bindings=bindings,
        )
        raw = client.json_chat(
            model=model,
            system=(
                "You are an independent source-frame to Schema binding judge. "
                "Use only the supplied source frames and active Schema; never "
                "invent endpoints or relations. Return only the supplied JSON Schema."
            ),
            prompt=prompt,
            schema=source_frame_schema_binding_response_schema(logical_schema),
            temperature=0.0,
            context_budget_tokens=Config.ollama_default_context_budget,
            max_output_tokens=max(
                4096, int(Config.relation_classifier_output_reserve),
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        parsed = parse_source_frame_schema_bindings(
            raw,
            inventory=inventory,
            logical_schema=logical_schema,
            allow_partial=True,
        )
        known = {
            (
                int(item["frame_index"]), str(item["predicate"]),
                int(item["subject_index"]), int(item["object_index"]),
            )
            for item in bindings
        }
        additions = [
            item for item in parsed.get("bindings") or ()
            if (
                int(item["frame_index"]), str(item["predicate"]),
                int(item["subject_index"]), int(item["object_index"]),
            ) not in known
        ]
        bindings.extend(additions)
        diagnostics = client.last_call_diagnostics
        calls.append({
            "round": round_index,
            "elapsed_seconds": round(
                time.perf_counter() - started, 3,
            ),
            "new_bindings": len(additions),
            "dropped_bindings": list(
                parsed.get("dropped_bindings") or ()
            ),
            "coverage_complete": bool(parsed.get("coverage_complete")),
            "prompt_tokens": (
                int(diagnostics.prompt_eval_count) if diagnostics else None
            ),
            "output_tokens": (
                int(diagnostics.output_eval_count) if diagnostics else None
            ),
        })
        if not additions:
            break
    shortlist = {
        "bindings": [],
        "frames_considered": 0,
        "candidates_considered": 0,
    }
    if recall_shortlist:
        shortlist = source_frame_schema_recall_shortlist(
            inventory,
            bindings,
            occurrences=occurrences,
            logical_schema=logical_schema,
            max_additions_per_frame=1,
        )
        bindings.extend(shortlist.get("bindings") or ())
    occurrence_by_key = {
        str(item["occurrence_key"]): item for item in occurrences
    }
    frames = list(inventory.get("frames") or ())
    predictions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for binding in bindings:
        frame_index = int(binding["frame_index"])
        if not 0 <= frame_index < len(frames):
            continue
        frame = frames[frame_index]
        keys = list(frame.get("argument_occurrence_keys") or ())
        subject_index = int(binding["subject_index"])
        object_index = int(binding["object_index"])
        if not (
            0 <= subject_index < len(keys)
            and 0 <= object_index < len(keys)
        ):
            continue
        subject = occurrence_by_key.get(str(keys[subject_index]))
        object_ = occurrence_by_key.get(str(keys[object_index]))
        if subject is None or object_ is None:
            continue
        triple = (
            str(subject["canonical_id"]),
            str(binding["predicate"]),
            str(object_["canonical_id"]),
        )
        if triple[0] == triple[2]:
            continue
        predictions.setdefault(triple, {
            "subject_canonical_id": triple[0],
            "predicate": triple[1],
            "object_canonical_id": triple[2],
            "frame_index": frame_index,
            "open_relation": str(frame.get("relation") or ""),
            "evidence": str(frame.get("evidence_text") or ""),
            "source_origin": str(
                frame.get("deterministic_source_origin") or "model"
            ),
            "binding_source": str(
                binding.get("binding_source") or "MODEL_SOURCE_BINDING"
            ),
            "shortlist_score": binding.get("shortlist_score"),
        })
    predicted = set(predictions)
    negative = {
        (
            str(item["subject_canonical_id"]),
            str(item["predicate"]),
            str(item["object_canonical_id"]),
        )
        for item in negative_claims
        if str(item["document_id"]) == document_id
    }
    exact_hits = gold & predicted
    final_verifier_result = None
    if final_verifier:
        final_verifier_result = _oracle_final_verify(
            client=client,
            model=model,
            content=content,
            document_id=document_id,
            inventory=inventory,
            occurrences=occurrences,
            logical_schema=logical_schema,
            bindings=bindings,
            gold=gold,
            negative_claims=negative_claims,
            force_single_verifier=force_single_verifier,
        )
    return {
        "schema_predicates": len(logical_schema["relation_types"]),
        "deterministic_bindings": len(
            deterministic.get("bindings") or ()
        ),
        "binding_calls": calls,
        "bindings": len(bindings),
        "recall_shortlist": {
            "enabled": bool(recall_shortlist),
            "bindings_added": len(shortlist.get("bindings") or ()),
            "frames_considered": int(
                shortlist.get("frames_considered") or 0
            ),
            "candidates_considered": int(
                shortlist.get("candidates_considered") or 0
            ),
        },
        "predicted_unique_facts": len(predicted),
        "exact_hits": len(exact_hits),
        "exact_recall": len(exact_hits) / len(gold) if gold else None,
        "exact_precision_before_final_verifier": (
            len(exact_hits) / len(predicted) if predicted else None
        ),
        "negative_claim_leaks_before_final_verifier": len(
            predicted & negative
        ),
        "misses": [
            {"subject": key[0], "predicate": key[1], "object": key[2]}
            for key in sorted(gold - predicted)
        ],
        "non_gold_predictions": [
            predictions[key] for key in sorted(predicted - gold)
        ],
        "predictions": list(predictions.values()),
        "final_verifier": final_verifier_result,
    }


def _evaluate_document(
    *,
    client: OllamaClient,
    model: str,
    document: dict[str, Any],
    gold_assertions: list[dict[str, Any]],
    schema_assertions: list[dict[str, Any]],
    negative_claims: list[dict[str, Any]],
    coreference_links: list[dict[str, Any]],
    gleaning_rounds: int,
    oracle_bindings: bool,
    binding_rounds: int,
    recall_shortlist: bool,
    reference_adjudication: bool,
    oracle_final_verifier: bool,
    oracle_final_force_single: bool,
) -> dict[str, Any]:
    document_id = str(document["id"])
    content = (ROOT / str(document["path"])).read_text(encoding="utf-8")
    occurrences = _occurrences(
        document_id,
        coreference_links=coreference_links,
    )
    _enrich_source_structure(content, occurrences)
    occurrence_by_key = {
        str(item["occurrence_key"]): item for item in occurrences
    }
    frames: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    resolved_reference_keys: set[str] = set()
    reference_adjudication_result: dict[str, Any] | None = None
    for round_index in range(1 + max(0, int(gleaning_rounds))):
        started = time.perf_counter()
        raw = client.json_chat(
            model=model,
            system=SHADOW_RELATION_OPEN_FACT_SYSTEM_PROMPT,
            prompt=grounded_open_fact_prompt(
                context_id=document_id,
                content=content,
                occurrences=occurrences,
                context_kind="gold_benchmark",
                already_found=frames,
                failed_items=failed_items,
            ),
            schema=grounded_open_fact_response_schema(occurrences),
            temperature=0.0,
            context_budget_tokens=Config.ollama_default_context_budget,
            max_output_tokens=max(
                4096, int(Config.relation_classifier_output_reserve),
            ),
            safety_margin_tokens=Config.relation_classifier_safety_margin,
        )
        parsed = parse_grounded_open_facts(
            raw,
            content=content,
            occurrences=occurrences,
            allow_partial=True,
        )
        for proxy in parsed.get("reference_proxies") or ():
            key = str(proxy.get("occurrence_key") or "")
            if not key or key in occurrence_by_key:
                continue
            occurrences.append(proxy)
            occurrence_by_key[key] = proxy
            resolved_reference_keys.add(key)
        before = len(frames)
        frames = _deduplicate_frames(frames, list(parsed["frames"]))
        failed_items = list(parsed.get("repair_items") or ())
        diagnostics = client.last_call_diagnostics
        calls.append({
            "round": round_index,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "root_fields": sorted(str(key) for key in raw),
            "raw_resolved_references": list(
                raw.get("resolved_references") or ()
            ),
            "accepted_frames": len(frames) - before,
            "dropped_items": list(parsed["dropped_items"]),
            "resolved_references": len(
                parsed.get("reference_proxies") or ()
            ),
            "dropped_references": list(
                parsed.get("dropped_references") or ()
            ),
            "prompt_tokens": (
                int(diagnostics.prompt_eval_count) if diagnostics else None
            ),
            "output_tokens": (
                int(diagnostics.output_eval_count) if diagnostics else None
            ),
        })

    if reference_adjudication:
        reference_prompt = relation_reference_resolution_prompt(
            content=content,
            occurrences=occurrences,
            inventory={"frames": frames},
        )
        reference_payload = json.loads(reference_prompt)
        reference_count = len(reference_payload.get("references") or ())
        if reference_count:
            reference_started = time.perf_counter()
            raw_references = client.json_chat(
                model=model,
                system=(
                    "You are a conservative, predicate-blind coreference "
                    "adjudicator. Track discourse and grammar and return "
                    "exactly the supplied JSON Schema."
                ),
                prompt=reference_prompt,
                schema=relation_reference_resolution_response_schema(
                    content=content,
                    occurrences=occurrences,
                    inventory={"frames": frames},
                ),
                temperature=0.0,
                context_budget_tokens=Config.ollama_default_context_budget,
                max_output_tokens=max(
                    4096, 256 * reference_count,
                ),
                safety_margin_tokens=Config.relation_classifier_safety_margin,
            )
            parsed_references = parse_relation_reference_resolutions(
                raw_references,
                content=content,
                occurrences=occurrences,
                inventory={"frames": frames},
                allow_partial=True,
            )
            for proxy in parsed_references.get("reference_proxies") or ():
                key = str(proxy.get("occurrence_key") or "")
                if not key or key in occurrence_by_key:
                    continue
                occurrences.append(proxy)
                occurrence_by_key[key] = proxy
                resolved_reference_keys.add(key)
            diagnostics = client.last_call_diagnostics
            reference_adjudication_result = {
                "target_count": reference_count,
                "accepted_proxies": len(
                    parsed_references.get("reference_proxies") or ()
                ),
                "dropped_resolutions": list(
                    parsed_references.get("dropped_resolutions") or ()
                ),
                "coverage_complete": bool(
                    parsed_references.get("coverage_complete")
                ),
                "elapsed_seconds": round(
                    time.perf_counter() - reference_started, 3,
                ),
                "prompt_tokens": (
                    int(diagnostics.prompt_eval_count)
                    if diagnostics else None
                ),
                "output_tokens": (
                    int(diagnostics.output_eval_count)
                    if diagnostics else None
                ),
            }
    augmented = augment_grounded_open_fact_inventory(
        {"frames": frames},
        content=content,
        occurrences=occurrences,
    )
    frames = list(augmented["frames"])

    frame_rows: list[dict[str, Any]] = []
    directed_pairs: set[tuple[str, str]] = set()
    undirected_pairs: set[frozenset[str]] = set()
    for frame in frames:
        subject_key, object_key = frame["argument_occurrence_keys"]
        subject = str(occurrence_by_key[subject_key]["canonical_id"])
        object_ = str(occurrence_by_key[object_key]["canonical_id"])
        directed_pairs.add((subject, object_))
        undirected_pairs.add(frozenset((subject, object_)))
        frame_rows.append({
            "subject_canonical_id": subject,
            "predicate": str(frame["relation"]),
            "object_canonical_id": object_,
            "evidence_span": list(frame["evidence_span"]),
            "roles": list(
                frame.get("parser_inferred_source_roles")
                or frame.get("roles") or []
            ),
            "source_syntax_origin": str(
                frame.get("parser_inferred_source_syntax")
                or frame.get("deterministic_source_syntax")
                or ""
            ),
        })

    unique_gold: dict[tuple[str, str, str], dict[str, Any]] = {}
    for assertion in gold_assertions:
        if str(assertion["document_id"]) != document_id:
            continue
        key = (
            str(assertion["subject_canonical_id"]),
            str(assertion["predicate"]),
            str(assertion["object_canonical_id"]),
        )
        unique_gold.setdefault(key, assertion)
    oracle_result = None
    if oracle_bindings:
        oracle_result = _oracle_bind_frames(
            client=client,
            model=model,
            content=content,
            inventory={**augmented, "frames": frames},
            occurrences=occurrences,
            logical_schema=_oracle_schema(document_id, schema_assertions),
            gold=set(unique_gold),
            negative_claims=negative_claims,
            document_id=document_id,
            binding_rounds=binding_rounds,
            recall_shortlist=recall_shortlist,
            final_verifier=oracle_final_verifier,
            force_single_verifier=oracle_final_force_single,
        )
    rows: list[dict[str, Any]] = []
    for key, assertion in unique_gold.items():
        directed = (key[0], key[2]) in directed_pairs
        undirected = frozenset((key[0], key[2])) in undirected_pairs
        rows.append({
            "assertion_id": str(assertion["assertion_id"]),
            "subject_canonical_id": key[0],
            "predicate": key[1],
            "object_canonical_id": key[2],
            "directed_endpoint_hit": directed,
            "undirected_endpoint_hit": undirected,
        })
    total = len(rows)
    directed_hits = sum(bool(item["directed_endpoint_hit"]) for item in rows)
    undirected_hits = sum(bool(item["undirected_endpoint_hit"]) for item in rows)
    return {
        "document_id": document_id,
        "domain": str(document["domain"]),
        "gold_unique_facts": total,
        "model_calls": calls,
        "frame_count": len(frame_rows),
        "resolved_reference_proxy_count": len(resolved_reference_keys),
        "reference_adjudication": reference_adjudication_result,
        "deterministic_frames_added": int(
            augmented.get("deterministic_frames_added") or 0
        ),
        "deterministic_frames_by_origin": dict(
            augmented.get("deterministic_frames_by_origin") or {}
        ),
        "parser_role_frames_annotated": int(
            augmented.get("parser_role_frames_annotated") or 0
        ),
        "directed_endpoint_hits": directed_hits,
        "directed_endpoint_recall": directed_hits / total if total else None,
        "undirected_endpoint_hits": undirected_hits,
        "undirected_endpoint_recall": undirected_hits / total if total else None,
        "directed_misses": [
            item for item in rows if not item["directed_endpoint_hit"]
        ],
        "oracle_binding": oracle_result,
        "frames": frame_rows,
    }


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Frozen calibration dataset root (defaults to English v1).",
    )
    parser.add_argument("--documents", nargs="*", default=[])
    parser.add_argument(
        "--gold-tier",
        choices=("core_direct", "all_labels"),
        default="core_direct",
        help=(
            "Primary scoring tier. Interpretation-sensitive labels are "
            "excluded by default but remain available as an all-label "
            "sensitivity run."
        ),
    )
    parser.add_argument("--gleaning-rounds", type=int, default=1)
    parser.add_argument("--model", default=Config.default_llm_model)
    parser.add_argument(
        "--oracle-bindings",
        action="store_true",
        help=(
            "Also map source frames to a Gold-labelled per-document Schema. "
            "This is an upper-bound binding audit, not a production score."
        ),
    )
    parser.add_argument(
        "--binding-rounds",
        type=int,
        default=0,
        help=(
            "Additional universal Schema-binding recovery rounds; defaults "
            "to the production setting of zero."
        ),
    )
    parser.add_argument(
        "--recall-shortlist",
        action="store_true",
        help=(
            "Add at most one deterministic semantic-retrieval alternative "
            "per source frame before final-verifier evaluation."
        ),
    )
    parser.add_argument(
        "--reference-adjudication",
        action="store_true",
        help=(
            "Resolve predicate-adjacent unresolved references in an independent "
            "discourse/role pass before deterministic syntax completion."
        ),
    )
    parser.add_argument(
        "--oracle-final-verifier",
        action="store_true",
        help=(
            "Run the production deterministic validator and ordered verifier "
            "over oracle-Schema candidate bindings."
        ),
    )
    parser.add_argument(
        "--oracle-final-force-single",
        action="store_true",
        help=(
            "Benchmark ablation: bypass verifier batching while keeping the "
            "candidate set and verifier contract unchanged."
        ),
    )
    parser.add_argument(
        "--gold-coreference-proxies",
        action="store_true",
        help=(
            "Ablation upper bound: expose expert Gold reference spans as "
            "relation-only endpoint proxies."
        ),
    )
    args = parser.parse_args()

    ROOT = args.dataset_root.expanduser().resolve()
    if not (ROOT / "manifest.json").is_file():
        raise SystemExit(f"dataset manifest not found: {ROOT / 'manifest.json'}")

    manifest = _read("manifest.json")
    selected = set(args.documents)
    documents = [
        item for item in manifest["documents"]
        if not selected or str(item["id"]) in selected
    ]
    missing = selected - {str(item["id"]) for item in documents}
    if missing:
        raise SystemExit("unknown document ids: " + ", ".join(sorted(missing)))
    assertion_payload = _read("annotations/assertions.json")
    assertions = list(assertion_payload["relations"])
    assertion_bands = partition_assertions(ROOT, assertions)
    gold_assertions = assertion_bands[str(args.gold_tier)]
    negative_claims = list(assertion_payload.get("negative_claims") or ())
    coreference_links = (
        list(_read("annotations/coreference_links.json"))
        if args.gold_coreference_proxies else []
    )
    client = OllamaClient(timeout=Config.ollama_timeout)
    started = time.perf_counter()
    results = [
        _evaluate_document(
            client=client,
            model=str(args.model),
            document=document,
            gold_assertions=gold_assertions,
            schema_assertions=assertions,
            negative_claims=negative_claims,
            coreference_links=coreference_links,
            gleaning_rounds=max(0, min(1, int(args.gleaning_rounds))),
            oracle_bindings=bool(args.oracle_bindings),
            binding_rounds=max(0, min(1, int(args.binding_rounds))),
            recall_shortlist=bool(args.recall_shortlist),
            reference_adjudication=bool(args.reference_adjudication),
            oracle_final_verifier=bool(args.oracle_final_verifier),
            oracle_final_force_single=bool(
                args.oracle_final_force_single
            ),
        )
        for document in documents
    ]
    total = sum(int(item["gold_unique_facts"]) for item in results)
    selected_document_ids = {str(item["id"]) for item in documents}
    all_label_unique_facts = len(unique_fact_rows([
        item for item in assertions
        if str(item["document_id"]) in selected_document_ids
    ]))
    core_direct_unique_facts = len(unique_fact_rows([
        item for item in assertion_bands["core_direct"]
        if str(item["document_id"]) in selected_document_ids
    ]))
    interpretation_unique_facts = len(unique_fact_rows([
        item for item in assertion_bands["interpretation_band"]
        if str(item["document_id"]) in selected_document_ids
    ]))
    directed = sum(int(item["directed_endpoint_hits"]) for item in results)
    undirected = sum(int(item["undirected_endpoint_hits"]) for item in results)
    oracle_predictions = sum(
        int((item.get("oracle_binding") or {}).get(
            "predicted_unique_facts"
        ) or 0)
        for item in results
    )
    oracle_hits = sum(
        int((item.get("oracle_binding") or {}).get("exact_hits") or 0)
        for item in results
    )
    print(json.dumps({
        "dataset_root": str(ROOT),
        "model": str(args.model),
        "gold_tier": str(args.gold_tier),
        "available_gold_facts": {
            "core_direct": core_direct_unique_facts,
            "interpretation_band": interpretation_unique_facts,
            "all_labels": all_label_unique_facts,
        },
        "gleaning_rounds": max(0, min(1, int(args.gleaning_rounds))),
        "gold_coreference_proxies": bool(args.gold_coreference_proxies),
        "oracle_bindings": bool(args.oracle_bindings),
        "binding_rounds": max(0, min(1, int(args.binding_rounds))),
        "recall_shortlist": bool(args.recall_shortlist),
        "reference_adjudication": bool(args.reference_adjudication),
        "oracle_final_verifier": bool(args.oracle_final_verifier),
        "oracle_final_force_single": bool(
            args.oracle_final_force_single
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gold_unique_facts": total,
        "directed_endpoint_hits": directed,
        "directed_endpoint_recall": directed / total if total else None,
        "undirected_endpoint_hits": undirected,
        "undirected_endpoint_recall": undirected / total if total else None,
        "oracle_exact_hits": oracle_hits if args.oracle_bindings else None,
        "oracle_exact_recall": (
            oracle_hits / total
            if args.oracle_bindings and total else None
        ),
        "oracle_exact_precision_before_final_verifier": (
            oracle_hits / oracle_predictions
            if args.oracle_bindings and oracle_predictions else None
        ),
        "documents": results,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
