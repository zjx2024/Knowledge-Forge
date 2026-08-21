from __future__ import annotations

from copy import deepcopy

import pytest

from knowledge_forge.schema_evolution import BASE_SCHEMA
from knowledge_forge.shadow_schema_overlay import (
    SHADOW_RELATION_SCHEMA_OVERLAY,
    SHADOW_RELATION_SCHEMA_OVERLAY_REQUIRED_FIELDS,
    SHADOW_RELATION_SCHEMA_OVERLAY_VERSION,
    apply_shadow_relation_schema_overlay,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


def test_policy_completion_injects_no_hidden_predicates():
    original = deepcopy(BASE_SCHEMA)

    enhanced = apply_shadow_relation_schema_overlay(BASE_SCHEMA)

    assert SHADOW_RELATION_SCHEMA_OVERLAY == ()
    assert BASE_SCHEMA == original
    assert enhanced["shadow_relation_schema_overlay_version"] == (
        SHADOW_RELATION_SCHEMA_OVERLAY_VERSION
    )
    assert [item["name"] for item in enhanced["relation_types"]] == [
        item["name"] for item in BASE_SCHEMA["relation_types"]
    ]
    for item in enhanced["relation_types"]:
        assert SHADOW_RELATION_SCHEMA_OVERLAY_REQUIRED_FIELDS <= set(item)
        # Generic production completion keeps cross-scope candidates visible
        # to the classifier/verifier instead of deleting them at planning.
        assert item["scope_policy"] == {
            "require_same_atomic_scope": False,
            "allow_cross_scope_with_explicit_evidence": True,
        }


def test_unknown_custom_predicate_missing_behavior_fields_fails_closed():
    with pytest.raises(ValueError, match="missing behavior metadata"):
        apply_shadow_relation_schema_overlay({
            "version": "custom-schema-v1",
            "relation_types": [{
                "name": "custom_rel",
                "semantic_definition": "Custom relationship.",
                "domain_types": ["*"],
                "range_types": ["*"],
                "direction_description": "Custom direction.",
                "symmetric": False,
                "inverse_predicate": None,
                "canonical_orientation": "forward",
                "publish_mode": "auto",
                "positive_examples": [],
                "negative_examples": [],
                "reflexive": False,
            }],
        })


def test_complete_arbitrary_predicate_is_preserved_without_name_lookup():
    custom = predicate_policy(
        "alpha_rel",
        domain=["SourceType"],
        range_=["TargetType"],
        positive_cues=["alpha cue"],
    )

    enhanced = apply_shadow_relation_schema_overlay(logical_schema(
        custom, version="custom-complete-v1",
    ))

    assert len(enhanced["relation_types"]) == 1
    result = enhanced["relation_types"][0]
    assert result["name"] == "alpha_rel"
    assert result["positive_cues"] == ["alpha cue"]
    assert result["domain_types"] == ["SourceType"]


def test_schema_metadata_controls_publication_without_name_special_case():
    custom = predicate_policy("beta_rel", publish_mode="disabled")

    enhanced = apply_shadow_relation_schema_overlay(logical_schema(custom))

    assert enhanced["relation_types"][0]["publish_mode"] == "disabled"
