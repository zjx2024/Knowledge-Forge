"""Enhanced-only Shadow relation schema overlay.

The overlay is data, not control flow.  It supplies complete behavior metadata
for the Shadow v2 relation engine and validates user/project predicates before
the runtime can call proposer/classifier/verifier code.  Stable ingestion keeps
using the active project schema directly.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from . import relation_shadow_v2 as v2


SHADOW_RELATION_SCHEMA_OVERLAY_VERSION = "shadow-relation-schema-overlay-v2"

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
    "qualifier_schema",
    "identity_qualifier_keys",
    "conflicts_with",
    "negation_policy",
    "modality_policy",
})

_BEHAVIOR_CONFLICT_FIELDS = frozenset({
    "symmetric",
    "inverse_predicate",
    "canonical_orientation",
    "reflexive",
    "reciprocal_allowed",
    "publish_mode",
})

_CONTEXT_KINDS = [
    "paragraph",
    "sentence",
    "clause",
    "apposition",
    "table_row",
    "table_cell",
    "list_item",
]
_FACT_SOURCE_ROLES = ["body", "list_item", "table_row", "table_cell"]
_DEFAULT_SCOPE_POLICY = {
    "require_same_atomic_scope": True,
    "allow_cross_scope_with_explicit_evidence": False,
}

_RELATION_EVIDENCE_ROLES = ["subject", "predicate_cue", "object"]


def _policy(
    name: str,
    domain: list[str],
    range_: list[str],
    semantic_definition: str,
    direction_description: str,
    *,
    subject_role: str = "The source endpoint plays the subject role.",
    object_role: str = "The target endpoint plays the object role.",
    symmetric: bool = False,
    inverse_predicate: str | None = None,
    canonical_orientation: str | None = None,
    publish_mode: str = v2.PUBLISH_AUTO,
    reflexive: bool = False,
    reciprocal_allowed: bool = False,
    positive_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
    positive_cues: list[str] | None = None,
    negative_cues: list[str] | None = None,
    anti_patterns: list[str] | None = None,
    optional_soft_cues: list[str] | None = None,
    qualifier_schema: dict[str, Any] | None = None,
    identity_qualifier_keys: list[str] | None = None,
    conflicts_with: list[str] | None = None,
    allowed_context_kinds: list[str] | None = None,
    allowed_source_roles: list[str] | None = None,
    scope_policy: dict[str, Any] | None = None,
    required_evidence_roles: list[str] | None = None,
    negation_policy: str = "reject_negated",
    modality_policy: str = "require_asserted",
) -> dict[str, Any]:
    orientation = canonical_orientation or (
        v2.CANONICAL_SYMMETRIC if symmetric else v2.CANONICAL_FORWARD
    )
    return {
        "name": name,
        "description": semantic_definition,
        "source_types": list(domain),
        "target_types": list(range_),
        "semantic_definition": semantic_definition,
        "domain_types": list(domain),
        "range_types": list(range_),
        "direction_description": direction_description,
        "subject_role_description": subject_role,
        "object_role_description": object_role,
        "symmetric": bool(symmetric),
        "inverse_predicate": inverse_predicate,
        "canonical_orientation": orientation,
        "publish_mode": publish_mode,
        "reflexive": bool(reflexive),
        "reciprocal_allowed": bool(reciprocal_allowed),
        "allowed_context_kinds": list(allowed_context_kinds or _CONTEXT_KINDS),
        "allowed_source_roles": list(
            allowed_source_roles or _FACT_SOURCE_ROLES
        ),
        "scope_policy": dict(scope_policy or _DEFAULT_SCOPE_POLICY),
        "required_evidence_roles": list(
            required_evidence_roles or _RELATION_EVIDENCE_ROLES
        ),
        "positive_examples": list(positive_examples or []),
        "negative_examples": list(negative_examples or []),
        "positive_cues": list(positive_cues or []),
        "negative_cues": list(negative_cues or []),
        "anti_patterns": list(anti_patterns or []),
        "optional_soft_cues": list(optional_soft_cues or []),
        "qualifier_schema": dict(qualifier_schema or {}),
        "identity_qualifier_keys": list(identity_qualifier_keys or []),
        "conflicts_with": list(conflicts_with or []),
        "negation_policy": negation_policy,
        "modality_policy": modality_policy,
    }


SHADOW_RELATION_SCHEMA_OVERLAY: tuple[dict[str, Any], ...] = (
    _policy(
        "related_to",
        ["*"],
        ["*"],
        "A broad semantic association that is not safe to auto-publish.",
        "No directed inverse role is implied.",
        publish_mode=v2.PUBLISH_REVIEW_ONLY,
        positive_cues=["related", "associated"],
        optional_soft_cues=["related", "关联"],
        required_evidence_roles=["subject", "object", "association_cue"],
    ),
    _policy(
        "is_a",
        ["*"],
        ["*"],
        "The subject is an instance, class member, or appositive type of the object.",
        "Orient from the specific instance/object to the class or type.",
        positive_cues=["is a", "is an", "classified as"],
        anti_patterns=["produces", "uses"],
    ),
    _policy(
        "part_of",
        ["*"],
        ["*"],
        "The subject is a component, portion, section, or member part of the object.",
        "Orient from the part as subject to the whole as object.",
        optional_soft_cues=["part of", "组成"],
    ),
    _policy(
        "belongs_to",
        ["*"],
        ["*"],
        "The subject is owned by, controlled by, or belongs to the object.",
        "Orient from the member or owned item to the owning object.",
        optional_soft_cues=["belongs to", "属于"],
    ),
    _policy(
        "located_in",
        ["Facility", "Project", "Organization", "Resource", "GenericEntity"],
        ["Location"],
        "The subject is physically or administratively located in the object.",
        "Orient from the located subject to the containing location.",
        optional_soft_cues=["located in", "位于"],
        negative_examples=["Peer list items in neighboring rows are not locations of one another."],
    ),
    _policy(
        "produces",
        ["Organization", "Facility", "Location", "Project", "GenericEntity"],
        ["Material", "Product", "GenericEntity"],
        "The subject creates, mines, manufactures, or outputs the object.",
        "Orient from the producer as subject to the produced output.",
        optional_soft_cues=["produces", "生产"],
        negative_examples=["A material used as an input is not necessarily produced by the user."],
    ),
    _policy(
        "uses",
        ["Organization", "Industry", "Process", "Application", "Technology", "GenericEntity"],
        ["Material", "Product", "Technology", "GenericEntity"],
        "The subject applies, consumes, or depends on the object as an input.",
        "Orient from the user/process/application to the used object.",
        optional_soft_cues=["uses", "使用"],
        negative_examples=["A material being used in an application should not be reversed."],
    ),
    _policy(
        "used_in",
        ["Material", "Product", "Technology", "GenericEntity"],
        ["Application", "Industry", "Process", "Product", "GenericEntity"],
        "The subject is used in the object application, process, or industry.",
        "Orient from the used material/product/technology to the application.",
        inverse_predicate="uses",
        canonical_orientation=v2.CANONICAL_FORWARD,
        optional_soft_cues=["used in", "用于"],
    ),
    _policy(
        "used_as",
        ["Material", "Product", "Technology", "GenericEntity"],
        ["Application", "Process", "GenericEntity"],
        "The subject serves as the object role or application.",
        "Orient from the material/product to the use role.",
        optional_soft_cues=["used as", "as a"],
    ),
    _policy(
        "imports_from",
        ["Location", "Organization", "GenericEntity"],
        ["Location", "Organization", "GenericEntity"],
        "The subject imports goods or materials from the object.",
        "Orient from importer to source.",
        optional_soft_cues=["imports from", "imported from"],
    ),
    _policy(
        "exports_to",
        ["Location", "Organization", "GenericEntity"],
        ["Location", "Organization", "GenericEntity"],
        "The subject exports goods or materials to the object.",
        "Orient from exporter to destination.",
        optional_soft_cues=["exports to", "exported to"],
    ),
    _policy(
        "processed_in",
        ["Material", "Product", "GenericEntity"],
        ["Facility", "Location", "GenericEntity"],
        "The subject is processed, refined, or converted in the object facility or place.",
        "Orient from processed material/product to processing facility or location.",
        optional_soft_cues=["processed in", "refined in"],
    ),
    _policy(
        "has_production",
        ["Location", "Organization", "Facility", "GenericEntity"],
        ["Material", "Product", "GenericEntity"],
        "The subject has reported production of the object commodity.",
        "Orient from reporting region/entity to commodity; quantities are qualifiers.",
        qualifier_schema={"amount": "number", "unit": "string", "period": "string", "estimated": "boolean"},
        identity_qualifier_keys=["period", "unit"],
        allowed_context_kinds=["table_row", "table_cell", "paragraph"],
    ),
    _policy(
        "has_reserve",
        ["Location", "Organization", "GenericEntity"],
        ["Material", "Product", "GenericEntity"],
        "The subject has reported reserves of the object commodity.",
        "Orient from reporting region/entity to commodity; reserve amount is a qualifier.",
        qualifier_schema={"amount": "number", "unit": "string", "period": "string", "estimated": "boolean"},
        identity_qualifier_keys=["period", "unit"],
        allowed_context_kinds=["table_row", "table_cell", "paragraph"],
    ),
    _policy(
        "has_resource",
        ["Location", "Organization", "GenericEntity"],
        ["Material", "Product", "GenericEntity"],
        "The subject has reported resources of the object commodity.",
        "Orient from reporting region/entity to commodity; resource amount is a qualifier.",
        qualifier_schema={"amount": "number", "unit": "string", "period": "string", "estimated": "boolean"},
        identity_qualifier_keys=["period", "unit"],
        allowed_context_kinds=["table_row", "table_cell", "paragraph"],
    ),
    _policy(
        "funded_by",
        ["Project", "ProgramOrPolicy", "Organization", "GenericEntity"],
        ["Organization", "ProgramOrPolicy", "GenericEntity"],
        "The subject project, program, or recipient is funded by the object.",
        "Orient from funded recipient/activity to funding source.",
        qualifier_schema={"amount": "number", "currency": "string", "period": "string"},
        optional_soft_cues=["funded by", "awarded by"],
    ),
    _policy(
        "operated_by",
        ["Facility", "Project", "ProgramOrPolicy", "GenericEntity"],
        ["Organization", "GenericEntity"],
        "The subject facility or project is operated by the object organization.",
        "Orient from operated asset to operator.",
        optional_soft_cues=["operated by", "operator"],
    ),
    _policy(
        "published_by",
        ["Publication", "GenericEntity"],
        ["Organization", "GenericEntity"],
        "The subject publication or report is published by the object organization.",
        "Orient from publication/report to publisher.",
        optional_soft_cues=["published by", "publisher"],
        negative_examples=["Publisher should not be the subject when the report is the published item."],
    ),
    _policy(
        "substitutes_for",
        ["Material", "Product", "Technology", "GenericEntity"],
        ["Material", "Product", "Technology", "GenericEntity"],
        "The subject can substitute for, replace, or reduce use of the object.",
        "Orient from substitute to item being substituted.",
        optional_soft_cues=["substitute for", "replacement for"],
    ),
    _policy(
        "depends_on",
        ["*"],
        ["*"],
        "The subject requires the object for operation, existence, supply, or function.",
        "Orient from the dependent subject to the dependency object.",
        optional_soft_cues=["depends on", "依赖"],
    ),
    _policy(
        "causes",
        ["EventOrTrend", "Process", "GenericEntity", "*"],
        ["EventOrTrend", "Process", "GenericEntity", "*"],
        "The subject directly causes or leads to the object event, state, or trend.",
        "Orient from cause to effect.",
        optional_soft_cues=["causes", "导致"],
        modality_policy="require_asserted_or_historical",
    ),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def shadow_relation_schema_overlay_hash() -> str:
    return hashlib.sha256(
        _canonical_json({
            "version": SHADOW_RELATION_SCHEMA_OVERLAY_VERSION,
            "predicates": SHADOW_RELATION_SCHEMA_OVERLAY,
        }).encode("utf-8")
    ).hexdigest()


def _relation_name(raw: Mapping[str, Any]) -> str:
    return str(raw.get("name") or "").strip()


def _missing_required_fields(raw: Mapping[str, Any]) -> list[str]:
    return sorted(
        field for field in SHADOW_RELATION_SCHEMA_OVERLAY_REQUIRED_FIELDS
        if field not in raw
    )


def apply_shadow_relation_schema_overlay(
    active_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an enhanced-only relation schema with complete predicate policy.

    Known generic/industry predicates receive missing metadata from the overlay.
    Unknown project predicates are accepted only if they already carry every
    behavior-controlling field; otherwise the enhanced Shadow path fails closed.
    """

    source_schema = deepcopy(dict(active_schema))
    source_version = str(
        source_schema.get("version")
        or source_schema.get("schema_version")
        or ""
    )
    if not source_version:
        raise ValueError("Shadow relation overlay requires active schema version")
    raw_relations = source_schema.get("relation_types")
    if not isinstance(raw_relations, list):
        raise ValueError("Shadow relation overlay requires relation_types")
    overlay_by_name = {
        _relation_name(item): deepcopy(item)
        for item in SHADOW_RELATION_SCHEMA_OVERLAY
    }
    output_by_name: dict[str, dict[str, Any]] = {}
    for raw in raw_relations:
        if not isinstance(raw, Mapping):
            raise ValueError("Shadow relation overlay relation metadata must be objects")
        name = _relation_name(raw)
        if not name:
            raise ValueError("Shadow relation overlay predicate name is empty")
        active = deepcopy(dict(raw))
        overlay = overlay_by_name.get(name)
        if overlay is None:
            missing = _missing_required_fields(active)
            if missing:
                raise ValueError(
                    f"predicate {name} missing Shadow overlay fields: {missing}"
                )
            output_by_name[name] = active
            continue
        conflicts = []
        for field in _BEHAVIOR_CONFLICT_FIELDS:
            if field not in active:
                continue
            active_value = active.get(field)
            if active_value in ("", None):
                continue
            if active_value != overlay.get(field):
                conflicts.append(field)
        if conflicts:
            raise ValueError(
                f"predicate {name} conflicts with Shadow overlay fields: {conflicts}"
            )
        output_by_name[name] = {**overlay, **active}
    for name, overlay in overlay_by_name.items():
        output_by_name.setdefault(name, deepcopy(overlay))
    enhanced_schema = {
        **source_schema,
        "version": f"{source_version}+{SHADOW_RELATION_SCHEMA_OVERLAY_VERSION}",
        "shadow_relation_schema_overlay_version": SHADOW_RELATION_SCHEMA_OVERLAY_VERSION,
        "shadow_relation_schema_overlay_hash": shadow_relation_schema_overlay_hash(),
        "relation_types": [
            output_by_name[name] for name in sorted(output_by_name)
        ],
    }
    for relation in enhanced_schema["relation_types"]:
        missing = _missing_required_fields(relation)
        if missing:
            raise ValueError(
                f"predicate {_relation_name(relation)} missing Shadow overlay fields: {missing}"
            )
    v2.normalize_predicate_schema(enhanced_schema)
    return enhanced_schema
