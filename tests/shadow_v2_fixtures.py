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
    positive_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
    optional_soft_cues: list[str] | None = None,
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
        "symmetric": bool(symmetric),
        "inverse_predicate": inverse_predicate,
        "canonical_orientation": orientation,
        "publish_mode": publish_mode,
        "reflexive": bool(reflexive),
        "positive_examples": list(positive_examples or []),
        "negative_examples": list(negative_examples or []),
        "optional_soft_cues": list(optional_soft_cues or []),
    }


def logical_schema(
    *predicates: dict[str, Any],
    version: str = "test-predicate-policy-v1",
) -> dict[str, Any]:
    return {
        "version": version,
        "relation_types": list(predicates),
    }
