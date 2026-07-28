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


def test_industry_overlay_contains_complete_versioned_policy_fields():
    by_name = {item["name"]: item for item in SHADOW_RELATION_SCHEMA_OVERLAY}

    for predicate in (
        "is_a",
        "used_in",
        "used_as",
        "has_production",
        "has_reserve",
        "has_resource",
        "funded_by",
        "published_by",
        "substitutes_for",
        "related_to",
    ):
        assert predicate in by_name
        assert SHADOW_RELATION_SCHEMA_OVERLAY_REQUIRED_FIELDS <= set(by_name[predicate])
        assert by_name[predicate]["semantic_definition"]
        assert by_name[predicate]["direction_description"]
        assert by_name[predicate]["publish_mode"] in {"auto", "review_only", "disabled"}


def test_overlay_fills_known_base_schema_without_mutating_stable_schema():
    original = deepcopy(BASE_SCHEMA)

    enhanced = apply_shadow_relation_schema_overlay(BASE_SCHEMA)

    assert BASE_SCHEMA == original
    assert enhanced is not BASE_SCHEMA
    assert enhanced["shadow_relation_schema_overlay_version"] == (
        SHADOW_RELATION_SCHEMA_OVERLAY_VERSION
    )
    by_name = {item["name"]: item for item in enhanced["relation_types"]}
    assert SHADOW_RELATION_SCHEMA_OVERLAY_REQUIRED_FIELDS <= set(by_name["produces"])
    assert by_name["produces"]["domain_types"] == ["*"]
    assert by_name["has_production"]["qualifier_schema"]["amount"] == "number"
    assert by_name["related_to"]["publish_mode"] == "review_only"


def test_unknown_custom_predicate_missing_overlay_fields_fails_closed():
    with pytest.raises(ValueError, match="missing Shadow overlay fields"):
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


def test_unknown_custom_predicate_with_complete_metadata_is_preserved():
    custom = predicate_policy(
        "alpha_rel",
        domain=["SourceType"],
        range_=["TargetType"],
        positive_cues=["alpha cue"],
    )

    enhanced = apply_shadow_relation_schema_overlay(logical_schema(
        custom, version="custom-complete-v1",
    ))

    by_name = {item["name"]: item for item in enhanced["relation_types"]}
    assert by_name["alpha_rel"]["positive_cues"] == ["alpha cue"]
    assert by_name["alpha_rel"]["domain_types"] == ["SourceType"]
    assert "has_production" in by_name


def test_known_predicate_structural_policy_conflict_fails_closed():
    broken = deepcopy(BASE_SCHEMA)
    for item in broken["relation_types"]:
        if item["name"] == "produces":
            item["publish_mode"] = "disabled"
            break

    with pytest.raises(ValueError, match="conflicts with Shadow overlay"):
        apply_shadow_relation_schema_overlay(broken)
