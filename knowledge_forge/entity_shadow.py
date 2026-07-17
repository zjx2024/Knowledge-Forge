"""Deterministic stateful namespace for entity-resolution shadow replay."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Iterable

from .entity_identity import normalize_name
from .entity_identity import identifiers_conflict, types_are_compatible


class EntityShadowState:
    """Small pure state machine used before persisting an isolated shadow snapshot."""

    def __init__(self, *, namespace: str, policy_version: str):
        self.namespace = namespace
        self.policy_version = policy_version
        self._canonicals: dict[str, dict[str, Any]] = {}
        self._assignments: list[dict[str, Any]] = []
        self._cannot_links: dict[frozenset[str], str] = {}
        self._mention_to_canonical: dict[str, str] = {}
        self._review_candidates: list[dict[str, str]] = []
        self._hard_negatives: list[dict[str, str]] = []

    def add_cannot_link(self, left_mention_id: str, right_mention_id: str, reason: str) -> None:
        self._cannot_links[frozenset((left_mention_id, right_mention_id))] = reason

    def _blocked(self, mention_id: str, canonical: dict[str, Any]) -> bool:
        return any(
            frozenset((mention_id, member)) in self._cannot_links
            for member in canonical["mention_ids"]
        )

    def _new_canonical(self, mention: dict[str, Any]) -> dict[str, Any]:
        material = "\x1f".join((
            self.namespace,
            self.policy_version,
            str(mention["mention_id"]),
        ))
        canonical_id = "shadow_entity_" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        canonical = {
            "id": canonical_id,
            "namespace": self.namespace,
            "name": str(mention["surface"]),
            "normalized_name": str(mention.get("normalized_surface") or normalize_name(mention["surface"])),
            "entity_type": str(mention.get("entity_type") or "GenericEntity"),
            "aliases": sorted({
                str(mention.get("normalized_surface") or normalize_name(mention["surface"])),
                *(str(value) for value in mention.get("aliases") or [] if value),
            }),
            "mention_ids": [str(mention["mention_id"])],
        }
        self._canonicals[canonical_id] = canonical
        return canonical

    def replay(self, mentions: Iterable[dict[str, Any]]) -> None:
        for mention in mentions:
            mention_id = str(mention["mention_id"])
            if mention_id in self._mention_to_canonical:
                continue
            normalized = str(mention.get("normalized_surface") or normalize_name(mention["surface"]))
            aliases = {normalized, *(str(value) for value in mention.get("aliases") or [] if value)}
            exact_candidates = [
                item for item in self._canonicals.values()
                if not self._blocked(mention_id, item)
                and aliases.intersection(item["aliases"])
            ]
            compatible: list[dict[str, Any]] = []
            for item in exact_candidates:
                if identifiers_conflict(str(mention["surface"]), str(item["name"])):
                    self._hard_negatives.append({
                        "mention_id": mention_id, "canonical_id": item["id"],
                        "reason": "identifier_conflict",
                    })
                    continue
                if not types_are_compatible(
                    str(mention.get("entity_type") or "GenericEntity"),
                    str(item.get("entity_type") or "GenericEntity"),
                ):
                    self._review_candidates.append({
                        "mention_id": mention_id, "canonical_id": item["id"],
                        "reason": "cross_type_exact_alias",
                    })
                    continue
                compatible.append(item)
            candidates = compatible
            candidates.sort(key=lambda item: item["id"])
            canonical = candidates[0] if candidates else self._new_canonical(mention)
            if mention_id not in canonical["mention_ids"]:
                canonical["mention_ids"].append(mention_id)
                canonical["mention_ids"].sort()
                canonical["aliases"] = sorted(set(canonical["aliases"]).union(aliases))
            self._mention_to_canonical[mention_id] = canonical["id"]
            self._assignments.append({
                "namespace": self.namespace,
                "mention_id": mention_id,
                "canonical_id": canonical["id"],
                "policy_version": self.policy_version,
            })

    def snapshot(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "policy_version": self.policy_version,
            "canonicals": sorted(deepcopy(list(self._canonicals.values())), key=lambda item: item["id"]),
            "assignments": sorted(deepcopy(self._assignments), key=lambda item: item["mention_id"]),
            "cannot_links": sorted(
                ({"mentions": sorted(key), "reason": reason} for key, reason in self._cannot_links.items()),
                key=lambda item: item["mentions"],
            ),
            "review_candidates": sorted(
                deepcopy(self._review_candidates),
                key=lambda item: (item["mention_id"], item["canonical_id"]),
            ),
            "hard_negatives": sorted(
                deepcopy(self._hard_negatives),
                key=lambda item: (item["mention_id"], item["canonical_id"]),
            ),
        }
