"""Independent, deterministic entity-resolution state for P2 shadow runs."""

from __future__ import annotations

import hashlib
import json
import logging
from copy import deepcopy
from typing import Any, Iterable

from .entity_identity import (
    best_name_pair_features,
    identifiers_conflict,
    normalize_name,
    types_are_compatible,
)

logger = logging.getLogger(__name__)


_UNRESOLVED_TYPES = {"", "GenericEntity", "Unknown", "Unresolved"}
_HARD_INCOMPATIBLE_TYPE_PAIRS = {
    frozenset(("Location", "Material")),
    frozenset(("Location", "Product")),
    frozenset(("Material", "Product")),
    frozenset(("Person", "Material")),
    frozenset(("Person", "Product")),
    frozenset(("Person", "Location")),
}


def shadow_types_are_compatible(left: str, right: str) -> bool:
    """Enhanced-only hard type gate for shadow canonical assignment."""

    left_type = str(left or "GenericEntity")
    right_type = str(right or "GenericEntity")
    if left_type in _UNRESOLVED_TYPES or right_type in _UNRESOLVED_TYPES:
        return left_type == right_type
    if frozenset((left_type, right_type)) in _HARD_INCOMPATIBLE_TYPE_PAIRS:
        return False
    return left_type == right_type


class DatabaseShadowStore:
    """Narrow adapter that can only access entity shadow persistence APIs."""

    def __init__(self, database: Any):
        self.database = database

    def find_exact(self, namespace: str, normalized: str, limit: int) -> list[dict[str, Any]]:
        return self.database.find_exact_entity_shadow_candidates(namespace, normalized, limit)

    def search_lexical(
        self, namespace: str, normalized: str, limit: int,
    ) -> list[dict[str, Any]]:
        return self.database.search_entity_shadow_name_candidates(namespace, normalized, limit)

    def search_vector(
        self,
        namespace: str,
        model_name: str,
        vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        return self.database.search_entity_shadow_vector_candidates(
            namespace, model_name, vector, limit,
        )

    def persist_candidates(self, rows: list[dict[str, Any]]) -> None:
        self.database.persist_entity_shadow_candidates(rows)

    def find_canonical(self, namespace: str, canonical_key: str) -> dict[str, Any] | None:
        return self.database.find_entity_shadow_canonical(namespace, canonical_key)


class ShadowModelGateway:
    """Bounded model calls for non-exact shadow resolution only."""

    def __init__(
        self,
        ollama: Any,
        *,
        llm_model: str,
        context_budget_tokens: int,
        max_output_tokens: int = 512,
    ):
        self.ollama = ollama
        self.llm_model = llm_model
        self.context_budget_tokens = context_budget_tokens
        self.max_output_tokens = max_output_tokens

    def embed(self, texts: list[str], model_name: str) -> list[list[float]]:
        return self.ollama.embed(texts, model_name)

    def review(
        self, mention: dict[str, Any], candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not candidates:
            return None
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decision": {"type": "string", "enum": ["merge", "separate"]},
                "canonical_key": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["decision", "canonical_key", "reason"],
        }
        payload = {
            "mention": {
                "surface": mention["surface"],
                "type_assertion": mention.get("entity_type") or "GenericEntity",
                "evidence": str(mention.get("evidence") or "")[:1200],
            },
            "candidates": [{
                "canonical_key": item["canonical_key"],
                "name": item["name"],
                "type_assertion": item.get("primary_type") or "GenericEntity",
                "aliases": item.get("aliases") or [],
                "scores": item.get("resolution_scores") or {},
            } for item in candidates],
        }
        try:
            result = self.ollama.json_chat(
                model=self.llm_model,
                system=(
                    "Conservatively decide whether an exact source mention and one bounded "
                    "candidate refer to the same real-world entity. Related entities, "
                    "conflicting identifiers, versions and homonyms must remain separate. "
                    "Return only schema-valid JSON."
                ),
                prompt=json.dumps(payload, ensure_ascii=False),
                schema=schema,
                temperature=0,
                context_budget_tokens=self.context_budget_tokens,
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            # Review is an optional merge accelerator. Fail closed so a malformed,
            # truncated or unavailable review can never create an unsafe merge or
            # abort an otherwise complete entity snapshot.
            logger.warning("entity shadow LLM review skipped: %s", exc)
            return None
        if result.get("decision") != "merge":
            return None
        return result


class ShadowEntityResolver:
    """Resolve one exact source mention without writing production entity state.

    ``store`` is deliberately a shadow-only port.  It exposes bounded exact,
    lexical and vector retrieval plus candidate audit persistence.  Keeping the
    resolver free of :class:`Database` makes its no-model exact path and safety
    constraints straightforward to characterize.
    """

    def __init__(
        self,
        *,
        namespace: str,
        policy_version: str,
        store: Any,
        models: Any,
        reranker: Any,
        must_links: dict[str, str] | None = None,
        cannot_links: set[frozenset[str]] | None = None,
        lexical_cap: int = 20,
        vector_cap: int = 30,
        reranker_cap: int = 10,
        auto_merge_threshold: float = 0.94,
        review_threshold: float = 0.72,
        identity_profile: str = "bounded_similarity",
    ):
        self.namespace = namespace
        self.policy_version = policy_version
        self.store = store
        self.models = models
        self.reranker = reranker
        self.must_links = must_links or {}
        self.cannot_links = cannot_links or set()
        self.lexical_cap = max(1, min(50, int(lexical_cap)))
        self.vector_cap = max(1, min(100, int(vector_cap)))
        self.reranker_cap = max(1, min(30, int(reranker_cap)))
        self.auto_merge_threshold = float(auto_merge_threshold)
        self.review_threshold = float(review_threshold)
        self.identity_profile = str(identity_profile or "").strip()
        if self.identity_profile not in {
            "bounded_similarity", "exact_canonical_name_only",
        }:
            raise ValueError("unknown entity shadow identity profile")

    def _new_key(self, mention: dict[str, Any]) -> str:
        material = "\x1f".join((
            self.namespace,
            self.policy_version,
            str(mention["mention_id"]),
        ))
        return "shadow_entity_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _blocked_reason(self, mention: dict[str, Any], candidate: dict[str, Any]) -> str:
        canonical_key = str(candidate["canonical_key"])
        reasons = []
        if frozenset((str(mention["mention_id"]), canonical_key)) in self.cannot_links:
            reasons.append("cannot_link")
        if identifiers_conflict(str(mention["surface"]), str(candidate["name"])):
            reasons.append("identifier_conflict")
        if not shadow_types_are_compatible(
            str(mention.get("entity_type") or "GenericEntity"),
            str(candidate.get("primary_type") or "GenericEntity"),
        ):
            reasons.append("type_incompatible")
        return ",".join(reasons)

    @staticmethod
    def _scores(mention: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
        features = best_name_pair_features(str(mention["surface"]), candidate)
        lexical = max(float(candidate.get("lexical_score") or 0.0), features["lexical"])
        vector = max(0.0, float(candidate.get("vector_score") or 0.0))
        type_score = 1.0 if shadow_types_are_compatible(
            str(mention.get("entity_type") or "GenericEntity"),
            str(candidate.get("primary_type") or "GenericEntity"),
        ) else 0.0
        combined = min(1.0, 0.52 * lexical + 0.38 * vector + 0.10 * type_score)
        return {
            "lexical": lexical,
            "vector": vector,
            "type": type_score,
            "exact": features["exact"],
            "acronym": features["acronym"],
            "combined": combined,
            "reranker": 0.0,
        }

    def _audit(
        self,
        mention: dict[str, Any],
        candidates: list[dict[str, Any]],
        *,
        stage: str,
    ) -> None:
        rows = []
        for rank, candidate in enumerate(candidates, 1):
            rows.append({
                "namespace_id": self.namespace,
                "mention_key": str(mention["mention_id"]),
                "canonical_key": str(candidate["canonical_key"]),
                "stage": stage,
                "rank": rank,
                "cap": {
                    "exact": 10,
                    "must_link": 1,
                    "lexical": self.lexical_cap,
                    "vector": self.vector_cap,
                    "retrieval": self.lexical_cap + self.vector_cap,
                    "reranker": self.reranker_cap,
                }.get(stage, self.reranker_cap),
                "scores": candidate.get("resolution_scores") or {},
                "filter_reason": candidate.get("filter_reason") or "",
                "policy_version": self.policy_version,
            })
        if rows:
            self.store.persist_candidates(rows)

    @staticmethod
    def identity_text(mention: dict[str, Any]) -> str:
        return " | ".join((
            str(mention["surface"]),
            str(mention.get("entity_type") or "GenericEntity"),
            str(mention.get("evidence") or ""),
        ))

    def try_exact(self, mention: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve indexed exact aliases without invoking any model."""

        normalized = str(
            mention.get("normalized_surface") or normalize_name(str(mention["surface"]))
        )
        required_key = self.must_links.get(str(mention["mention_id"]))
        if required_key:
            candidate = self.store.find_canonical(self.namespace, required_key)
            if not candidate:
                return {
                    "canonical_key": self._new_key(mention),
                    "decision": "uncertain_missing_must_link",
                    "reason": "hard must-link 指向不存在的 shadow canonical，按策略保持隔离",
                    "scores": {}, "embedding": None, "candidates": [],
                }
            candidate = dict(candidate)
            reason = self._blocked_reason(mention, candidate)
            candidate["resolution_scores"] = {
                **self._scores(mention, candidate), "combined": 1.0,
            }
            candidate["filter_reason"] = reason
            self._audit(mention, [candidate], stage="must_link")
            if reason:
                return {
                    "canonical_key": self._new_key(mention),
                    "decision": "uncertain_constraint_conflict",
                    "reason": f"hard must-link 与硬约束冲突: {reason}",
                    "scores": candidate["resolution_scores"],
                    "embedding": None, "candidates": [candidate],
                }
            return {
                "canonical_key": required_key,
                "decision": "hard_must_link",
                "reason": "命中显式 hard must-link",
                "scores": candidate["resolution_scores"],
                "embedding": None, "candidates": [candidate],
            }
        exact = list(self.store.find_exact(self.namespace, normalized, 10))[:10]
        exact.sort(key=lambda item: str(item["canonical_key"]))
        allowed_exact = []
        audited_exact = []
        for candidate in exact:
            reason = self._blocked_reason(mention, candidate)
            candidate = dict(candidate)
            if (
                self.identity_profile == "exact_canonical_name_only"
                and str(candidate.get("normalized_name") or "") != normalized
            ):
                reason = ",".join(filter(None, [reason, "not_exact_canonical_name"]))
            candidate["resolution_scores"] = {
                **self._scores(mention, candidate),
                "combined": 1.0,
            }
            candidate["filter_reason"] = reason
            if not reason:
                allowed_exact.append(candidate)
            audited_exact.append(candidate)
        exact = audited_exact
        self._audit(mention, exact, stage="exact")
        if len(allowed_exact) == 1:
            return {
                "canonical_key": str(allowed_exact[0]["canonical_key"]),
                "decision": "exact_alias",
                "reason": "shadow 规范名称或可信别名精确命中",
                "scores": allowed_exact[0]["resolution_scores"],
                "embedding": None,
                "candidates": exact,
            }
        if len(allowed_exact) > 1:
            return {
                "canonical_key": self._new_key(mention),
                "decision": "quarantined_ambiguous_exact",
                "reason": "精确别名对应多个 canonical，隔离为 quarantined occurrence 等待人工裁决",
                "scores": {},
                "embedding": None,
                "candidates": exact,
                "quarantined": True,
            }
        return None

    def resolve_with_embedding(
        self,
        mention: dict[str, Any],
        *,
        embedding_model: str,
        embedding: list[float],
    ) -> dict[str, Any]:
        """Resolve a non-exact mention using one caller-supplied embedding."""

        normalized = str(
            mention.get("normalized_surface") or normalize_name(str(mention["surface"]))
        )

        if self.identity_profile == "exact_canonical_name_only":
            return {
                "canonical_key": self._new_key(mention),
                "decision": "new_canonical_exact_name_policy",
                "reason": (
                    "当前归一策略只允许规范名称精确匹配；"
                    "相似名称与向量近邻保留为独立 canonical"
                ),
                "scores": {},
                "embedding": embedding,
                "candidates": [],
            }

        lexical = list(self.store.search_lexical(
            self.namespace, normalized, self.lexical_cap,
        ))[: self.lexical_cap]
        vector = list(self.store.search_vector(
            self.namespace, embedding_model, embedding, self.vector_cap,
        ))[: self.vector_cap]
        combined: dict[str, dict[str, Any]] = {}
        for candidate in [*lexical, *vector]:
            key = str(candidate["canonical_key"])
            current = combined.setdefault(key, dict(candidate))
            current["lexical_score"] = max(
                float(current.get("lexical_score") or 0.0),
                float(candidate.get("lexical_score") or 0.0),
            )
            current["vector_score"] = max(
                float(current.get("vector_score") or 0.0),
                float(candidate.get("vector_score") or 0.0),
            )
        ranked = []
        for candidate in combined.values():
            candidate["resolution_scores"] = self._scores(mention, candidate)
            candidate["filter_reason"] = self._blocked_reason(mention, candidate)
            ranked.append(candidate)
        ranked.sort(
            key=lambda item: (
                -float(item["resolution_scores"]["combined"]),
                str(item["canonical_key"]),
            )
        )
        self._audit(mention, ranked, stage="retrieval")

        eligible = [item for item in ranked if not item["filter_reason"]]
        selected = eligible[: self.reranker_cap]
        reranker_scores = list(self.reranker(mention, selected)) if selected else []
        for candidate, score in zip(selected, reranker_scores):
            candidate["resolution_scores"]["reranker"] = max(0.0, min(1.0, float(score)))
            candidate["resolution_scores"]["combined"] = min(
                1.0,
                0.34 * candidate["resolution_scores"]["lexical"]
                + 0.30 * candidate["resolution_scores"]["vector"]
                + 0.26 * candidate["resolution_scores"]["reranker"]
                + 0.10 * candidate["resolution_scores"]["type"],
            )
        selected.sort(
            key=lambda item: (
                -float(item["resolution_scores"]["combined"]),
                str(item["canonical_key"]),
            )
        )
        self._audit(mention, selected, stage="reranker")
        best = selected[0] if selected else None
        if best and float(best["resolution_scores"]["combined"]) >= self.auto_merge_threshold:
            return {
                "canonical_key": str(best["canonical_key"]),
                "decision": "bounded_similarity_merge",
                "reason": "有界检索与 reranker 达到 shadow 自动归一阈值",
                "scores": best["resolution_scores"],
                "embedding": embedding,
                "candidates": ranked,
            }
        if best and float(best["resolution_scores"]["combined"]) >= self.review_threshold:
            try:
                reviewed = self.models.review(mention, selected)
            except Exception as exc:
                logger.warning("entity shadow candidate review skipped: %s", exc)
                reviewed = None
            reviewed_key = str((reviewed or {}).get("canonical_key") or "")
            reviewed_candidate = next(
                (item for item in selected if str(item["canonical_key"]) == reviewed_key),
                None,
            )
            if reviewed_candidate and not reviewed_candidate["filter_reason"]:
                return {
                    "canonical_key": reviewed_key,
                    "decision": "bounded_llm_merge",
                    "reason": str((reviewed or {}).get("reason") or "LLM shadow 裁决"),
                    "scores": reviewed_candidate["resolution_scores"],
                    "embedding": embedding,
                    "candidates": ranked,
                }
        return {
            "canonical_key": self._new_key(mention),
            "decision": "new_canonical",
            "reason": "没有通过约束和阈值的已有 canonical",
            "scores": {},
            "embedding": embedding,
            "candidates": ranked,
        }

    def resolve(self, mention: dict[str, Any], *, embedding_model: str = "") -> dict[str, Any]:
        """Resolve one mention; callers with multiple mentions should batch embeddings."""

        exact = self.try_exact(mention)
        if exact is not None:
            return exact
        embedding = self.models.embed([self.identity_text(mention)], embedding_model)[0]
        return self.resolve_with_embedding(
            mention,
            embedding_model=embedding_model,
            embedding=embedding,
        )


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
        self._events: list[dict[str, Any]] = []
        self._event_before: dict[str, dict[str, Any]] = {}

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

    def _state(self) -> dict[str, Any]:
        return {
            "canonicals": deepcopy(self._canonicals),
            "assignments": deepcopy(self._assignments),
            "mention_to_canonical": deepcopy(self._mention_to_canonical),
        }

    def _event(
        self,
        event_type: str,
        *,
        source: str = "",
        target: str = "",
        reason: str,
        before: dict[str, Any],
    ) -> str:
        material = "\x1f".join((
            self.namespace, self.policy_version, event_type, source, target,
            reason, str(len(self._events)),
        ))
        event_id = "shadow_event_" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        self._event_before[event_id] = before
        self._events.append({
            "id": event_id,
            "event_type": event_type,
            "source_canonical_id": source,
            "target_canonical_id": target,
            "reason": reason,
            "reversible": event_type != "undo",
        })
        return event_id

    def merge(self, source_id: str, target_id: str, reason: str) -> str:
        if source_id == target_id:
            raise ValueError("不能合并同一个 shadow canonical")
        source = self._canonicals.get(source_id)
        target = self._canonicals.get(target_id)
        if not source or not target:
            raise LookupError("shadow canonical 不存在")
        if any(
            frozenset((left, right)) in self._cannot_links
            for left in source["mention_ids"] for right in target["mention_ids"]
        ):
            raise ValueError("cannot-link 禁止该 shadow merge")
        before = self._state()
        target["mention_ids"] = sorted(set(target["mention_ids"] + source["mention_ids"]))
        target["aliases"] = sorted(set(target["aliases"] + source["aliases"]))
        for mention_id in source["mention_ids"]:
            self._mention_to_canonical[mention_id] = target_id
        for assignment in self._assignments:
            if assignment["canonical_id"] == source_id:
                assignment["canonical_id"] = target_id
        del self._canonicals[source_id]
        return self._event(
            "merge", source=source_id, target=target_id, reason=reason, before=before,
        )

    def split(self, canonical_id: str, mention_ids: Iterable[str], reason: str) -> str:
        canonical = self._canonicals.get(canonical_id)
        selected = sorted(set(str(item) for item in mention_ids))
        if not canonical or not selected or not set(selected).issubset(canonical["mention_ids"]):
            raise ValueError("split mentions 必须属于目标 shadow canonical")
        if len(selected) == len(canonical["mention_ids"]):
            raise ValueError("split 必须保留至少一个原 canonical mention")
        before = self._state()
        material = "\x1f".join((self.namespace, self.policy_version, canonical_id, *selected))
        target_id = "shadow_entity_" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        first = selected[0]
        target = {
            **deepcopy(canonical),
            "id": target_id,
            "mention_ids": selected,
        }
        canonical["mention_ids"] = sorted(set(canonical["mention_ids"]) - set(selected))
        self._canonicals[target_id] = target
        for mention_id in selected:
            self._mention_to_canonical[mention_id] = target_id
        for assignment in self._assignments:
            if assignment["mention_id"] in selected:
                assignment["canonical_id"] = target_id
        return self._event(
            "split", source=canonical_id, target=target_id,
            reason=f"{reason}; seed={first}", before=before,
        )

    def undo(self, event_id: str, reason: str) -> str:
        before = self._event_before.get(event_id)
        if before is None:
            raise LookupError("可逆 shadow event 不存在")
        current = self._state()
        self._canonicals = deepcopy(before["canonicals"])
        self._assignments = deepcopy(before["assignments"])
        self._mention_to_canonical = deepcopy(before["mention_to_canonical"])
        return self._event(
            "undo", source=event_id, reason=reason, before=current,
        )

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
            "events": deepcopy(self._events),
        }
