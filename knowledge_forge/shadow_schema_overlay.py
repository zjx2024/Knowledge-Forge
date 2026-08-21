"""Versioned, predicate-agnostic policy completion for Knowledge Forge.

The active project schema is the only source of relation semantics.  This
module never injects domain predicates and never infers behavior from a
predicate name.  It only completes non-semantic runtime contract fields with
generic defaults after checking that every behavior-controlling semantic field
is explicitly present in the active schema.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from . import relation_shadow_v2 as v2


SHADOW_RELATION_SCHEMA_OVERLAY_VERSION = (
    "relation-policy-completion-v5-grounded-open-policy"
)

SHADOW_RELATION_SCHEMA_OVERLAY_REQUIRED_FIELDS = frozenset({
    "name",
    "semantic_definition",
    "domain_types",
    "range_types",
    "direction_description",
    "subject_role_description",
    "object_role_description",
    "symmetric",
    "inverse_predicate",
    "canonical_orientation",
    "reflexive",
    "reciprocal_allowed",
    "publish_mode",
    "allowed_context_kinds",
    "allowed_source_roles",
    "scope_policy",
    "required_evidence_roles",
    "positive_examples",
    "negative_examples",
    "positive_cues",
    "negative_cues",
    "anti_patterns",
    "optional_soft_cues",
    "structural_evidence_kinds",
    "structural_role_labels",
    "qualifier_schema",
    "identity_qualifier_keys",
    "conflicts_with",
    "negation_policy",
    "modality_policy",
    "open_relation_policy",
})

# These fields change meaning, direction or publication.  Missing values must
# fail closed rather than be guessed from the relation identifier.
_REQUIRED_SEMANTIC_FIELDS = frozenset({
    "name",
    "semantic_definition",
    "domain_types",
    "range_types",
    "direction_description",
    "symmetric",
    "inverse_predicate",
    "canonical_orientation",
    "publish_mode",
    "positive_examples",
    "negative_examples",
    "optional_soft_cues",
})

_GENERIC_COMPLETION_DEFAULTS: dict[str, Any] = {
    "subject_role_description": "源端点承担该关系定义中的主语角色。",
    "object_role_description": "目标端点承担该关系定义中的宾语角色。",
    "reflexive": False,
    "reciprocal_allowed": False,
    "allowed_context_kinds": [
        "structured_chunk",
        "legacy_chunk",
        "paragraph",
        "sentence",
        "clause",
        "apposition",
        "table_row",
        "table_cell",
        "list_item",
        "media_description",
    ],
    "allowed_source_roles": [
        "body", "list_item", "table_row", "table_cell", "media_description",
    ],
    "scope_policy": {
        # Candidate generation is a recall stage.  A relation may be stated
        # across adjacent sentence/clause scopes, so the generic completion
        # policy must not discard that pair before the classifier/verifier see
        # the frozen context.  The independent verifier still requires exact
        # endpoint evidence and applies the predicate policy before publish.
        "require_same_atomic_scope": False,
        "allow_cross_scope_with_explicit_evidence": True,
    },
    "required_evidence_roles": ["subject", "predicate_cue", "object"],
    "positive_cues": [],
    "negative_cues": [],
    "anti_patterns": [],
    "qualifier_schema": {},
    "identity_qualifier_keys": [],
    "conflicts_with": [],
    "structural_evidence_kinds": [],
    "structural_role_labels": [],
    "negation_policy": "reject_negated",
    "modality_policy": "require_asserted",
    "open_relation_policy": {},
}

# Kept as an empty compatibility export.  Earlier versions exposed a tuple of
# hard-coded generic/industry predicates.  An empty value is itself an
# invariant: the completion layer cannot silently expand the project schema.
SHADOW_RELATION_SCHEMA_OVERLAY: tuple[dict[str, Any], ...] = ()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def shadow_relation_schema_overlay_hash() -> str:
    return hashlib.sha256(_canonical_json({
        "version": SHADOW_RELATION_SCHEMA_OVERLAY_VERSION,
        "semantic_fields": sorted(_REQUIRED_SEMANTIC_FIELDS),
        "completion_defaults": _GENERIC_COMPLETION_DEFAULTS,
        "injected_predicates": [],
    }).encode("utf-8")).hexdigest()


def _normalize_compatibility_aliases(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(raw))
    if "semantic_definition" not in value and "description" in value:
        value["semantic_definition"] = value["description"]
    if "domain_types" not in value and "source_types" in value:
        value["domain_types"] = list(value.get("source_types") or [])
    if "range_types" not in value and "target_types" in value:
        value["range_types"] = list(value.get("target_types") or [])
    if "source_types" not in value and "domain_types" in value:
        value["source_types"] = list(value.get("domain_types") or [])
    if "target_types" not in value and "range_types" in value:
        value["target_types"] = list(value.get("range_types") or [])
    if "description" not in value and "semantic_definition" in value:
        value["description"] = value["semantic_definition"]
    return value


def _complete_policy(raw: Mapping[str, Any]) -> dict[str, Any]:
    value = _normalize_compatibility_aliases(raw)
    name = str(value.get("name") or "").strip()
    if not name:
        raise ValueError("relation policy predicate name is empty")
    missing_semantic = sorted(
        field for field in _REQUIRED_SEMANTIC_FIELDS if field not in value
    )
    if missing_semantic:
        raise ValueError(
            f"predicate {name} missing behavior metadata: {missing_semantic}"
        )
    if not str(value.get("semantic_definition") or "").strip():
        raise ValueError(f"predicate {name} has empty semantic_definition")
    if not str(value.get("direction_description") or "").strip():
        raise ValueError(f"predicate {name} has empty direction_description")
    if not isinstance(value.get("domain_types"), list) or not value["domain_types"]:
        raise ValueError(f"predicate {name} has invalid domain_types")
    if not isinstance(value.get("range_types"), list) or not value["range_types"]:
        raise ValueError(f"predicate {name} has invalid range_types")
    for field in ("positive_examples", "negative_examples", "optional_soft_cues"):
        if not isinstance(value.get(field), list):
            raise ValueError(f"predicate {name} has invalid {field}")
    completed = {
        **deepcopy(_GENERIC_COMPLETION_DEFAULTS),
        **value,
    }
    missing = sorted(
        field for field in SHADOW_RELATION_SCHEMA_OVERLAY_REQUIRED_FIELDS
        if field not in completed
    )
    if missing:
        raise ValueError(f"predicate {name} missing runtime fields: {missing}")
    return completed


def apply_shadow_relation_schema_overlay(
    active_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Complete the active schema without adding or renaming predicates."""

    source_schema = deepcopy(dict(active_schema))
    source_version = str(
        source_schema.get("version") or source_schema.get("schema_version") or ""
    ).strip()
    if not source_version:
        raise ValueError("relation policy completion requires active schema version")
    raw_relations = source_schema.get("relation_types")
    if not isinstance(raw_relations, list) or not raw_relations:
        raise ValueError("relation policy completion requires relation_types")
    completed = []
    seen: set[str] = set()
    for raw in raw_relations:
        if not isinstance(raw, Mapping):
            raise ValueError("relation policy metadata must be objects")
        policy = _complete_policy(raw)
        name = str(policy["name"])
        if name in seen:
            raise ValueError(f"duplicate predicate metadata: {name}")
        seen.add(name)
        completed.append(policy)
    enhanced_schema = {
        **source_schema,
        "version": f"{source_version}+{SHADOW_RELATION_SCHEMA_OVERLAY_VERSION}",
        "shadow_relation_schema_overlay_version": SHADOW_RELATION_SCHEMA_OVERLAY_VERSION,
        "shadow_relation_schema_overlay_hash": shadow_relation_schema_overlay_hash(),
        "relation_types": completed,
    }
    v2.normalize_predicate_schema(enhanced_schema)
    return enhanced_schema
