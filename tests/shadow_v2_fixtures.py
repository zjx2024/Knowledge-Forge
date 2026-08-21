from __future__ import annotations

from typing import Any


def predicate_policy(
    name: str,
    *,
    domain: list[str] | None = None,
    range_: list[str] | None = None,
    semantic_definition: str | None = None,
    direction_description: str | None = None,
    symmetric: bool = False,
    inverse_predicate: str | None = None,
    canonical_orientation: str | None = None,
    publish_mode: str = "auto",
    reflexive: bool = False,
    reciprocal_allowed: bool = False,
    positive_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
    positive_cues: list[str] | None = None,
    negative_cues: list[str] | None = None,
    anti_patterns: list[str] | None = None,
    optional_soft_cues: list[str] | None = None,
    structural_evidence_kinds: list[str] | None = None,
    structural_role_labels: list[str] | None = None,
    qualifier_schema: dict[str, Any] | None = None,
    identity_qualifier_keys: list[str] | None = None,
    conflicts_with: list[str] | None = None,
    allowed_context_kinds: list[str] | None = None,
    allowed_source_roles: list[str] | None = None,
    source_role_pairs: list[str] | None = None,
    scope_policy: dict[str, Any] | None = None,
    required_evidence_roles: list[str] | None = None,
    negation_policy: str = "reject_negated",
    modality_policy: str = "require_asserted",
    open_relation_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build explicit test policy metadata without predicate-name defaults."""

    domains = list(domain or ["*"])
    ranges = list(range_ or ["*"])
    orientation = canonical_orientation or (
        "symmetric" if symmetric else "forward"
    )
    definition = semantic_definition or "Fixture relation semantics."
    direction = direction_description or (
        "The first occurrence is the subject and the second is the object."
    )
    return {
        "name": name,
        "description": definition,
        "source_types": domains,
        "target_types": ranges,
        "semantic_definition": definition,
        "domain_types": domains,
        "range_types": ranges,
        "direction_description": direction,
        "subject_role_description": "The subject endpoint follows the predicate direction.",
        "object_role_description": "The object endpoint follows the predicate direction.",
        "symmetric": bool(symmetric),
        "inverse_predicate": inverse_predicate,
        "canonical_orientation": orientation,
        "publish_mode": publish_mode,
        "reflexive": bool(reflexive),
        "reciprocal_allowed": bool(reciprocal_allowed),
        "allowed_context_kinds": list(allowed_context_kinds or [
            "paragraph", "sentence", "table_row", "table_cell", "list_item",
        ]),
        "allowed_source_roles": list(allowed_source_roles or [
            "body", "list_item", "table_row", "table_cell",
        ]),
        "source_role_pairs": list(source_role_pairs or []),
        "scope_policy": dict(scope_policy or {
            "require_same_atomic_scope": True,
            "allow_cross_scope_with_explicit_evidence": False,
        }),
        "required_evidence_roles": list(required_evidence_roles or [
            "subject", "predicate_cue", "object",
        ]),
        "positive_examples": list(positive_examples or []),
        "negative_examples": list(negative_examples or []),
        "positive_cues": list(positive_cues or []),
        "negative_cues": list(negative_cues or []),
        "anti_patterns": list(anti_patterns or []),
        "optional_soft_cues": list(optional_soft_cues or []),
        "structural_evidence_kinds": list(structural_evidence_kinds or []),
        "structural_role_labels": list(structural_role_labels or []),
        "qualifier_schema": dict(qualifier_schema or {}),
        "identity_qualifier_keys": list(identity_qualifier_keys or []),
        "conflicts_with": list(conflicts_with or []),
        "negation_policy": negation_policy,
        "modality_policy": modality_policy,
        "open_relation_policy": dict(open_relation_policy or {}),
    }


def logical_schema(
    *predicates: dict[str, Any],
    version: str = "test-predicate-policy-v1",
) -> dict[str, Any]:
    return {
        "version": version,
        "relation_types": list(predicates),
    }
