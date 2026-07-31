"""Versioned construction policy sets shared by evaluation and publication."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from .relation_verification import RELATION_VERIFICATION_POLICY_VERSION


POLICY_AXES = (
    "entity_extraction_policy_version",
    "entity_resolution_policy_version",
    "relation_candidate_policy_version",
    "relation_classifier_policy_version",
    "verification_policy_version",
    "schema_policy_version",
)

CONTROL_POLICY_SET: Mapping[str, str] = MappingProxyType({
    "entity_extraction_policy_version": "entity-extraction-p0-v1",
    "entity_resolution_policy_version": "entity-resolution-p0-v1",
    "relation_candidate_policy_version": "relation-candidate-p0-v1",
    "relation_classifier_policy_version": "relation-classifier-p0-v1",
    "verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION,
    "schema_policy_version": "schema-policy-p0-v1",
})

P1_RELATION_SHADOW_POLICY_SET: Mapping[str, str] = MappingProxyType({
    **CONTROL_POLICY_SET,
    "relation_candidate_policy_version": "relation-candidate-p1-shadow-v4",
    "relation_classifier_policy_version": "relation-classifier-p1-shadow-v1",
    "verification_policy_version": "relation-classifier-p1-shadow-v1",
})

P2_ENTITY_SHADOW_POLICY_SET: Mapping[str, str] = MappingProxyType({
    **P1_RELATION_SHADOW_POLICY_SET,
    "entity_extraction_policy_version": "entity-extraction-p2-structure-gate-v2",
    "entity_resolution_policy_version": "entity-resolution-p2-evidence-identity-v2",
    "entity_identity_policy_version": "entity-identity-exact-canonical-name-v1",
    "entity_identity_profile": "exact_canonical_name_only",
})

PUBLISH_PREDICATE_VERSION = "publish-predicate-p0-v1"


def normalize_policy_set(values: Mapping[str, object]) -> dict[str, str]:
    """Return a complete, ordered policy set and reject partial snapshots."""

    missing = [axis for axis in POLICY_AXES if not str(values.get(axis) or "").strip()]
    if missing:
        raise ValueError(f"策略集合缺少版本轴: {', '.join(missing)}")
    return {axis: str(values[axis]) for axis in POLICY_AXES}
