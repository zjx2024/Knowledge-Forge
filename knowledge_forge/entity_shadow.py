"""Independent, deterministic entity-resolution state for P2 shadow runs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from copy import deepcopy
from typing import Any, Iterable

from .entity_identity import (
    best_name_pair_features,
    identifiers_conflict,
    normalize_name,
    types_are_compatible,
)
from .sentence_spans import split_sentences

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


def bounded_identity_evidence(
    content: str,
    start: int,
    end: int,
    *,
    sentence_radius: int = 1,
    max_chars: int = 1200,
) -> str:
    """Return exact neighboring source sentences for identity/coreference review."""

    source = str(content or "")
    if not source:
        return ""
    left = max(0, min(len(source), int(start)))
    right = max(left, min(len(source), int(end)))
    # Newlines immediately following terminal punctuation may be emitted as
    # their own exact span by the shared splitter.  They carry no identity
    # evidence and must not consume the bounded neighboring-sentence radius.
    sentences = [item for item in split_sentences(source) if item[2].strip()]
    overlapping = [
        index for index, (sentence_start, sentence_end, _text) in enumerate(sentences)
        if sentence_start < max(right, left + 1) and sentence_end > left
    ]
    if overlapping:
        first = max(0, overlapping[0] - max(0, int(sentence_radius)))
        last = min(
            len(sentences) - 1,
            overlapping[-1] + max(0, int(sentence_radius)),
        )
        window_start = sentences[first][0]
        window_end = sentences[last][1]
    else:
        window_start = max(0, left - max_chars // 2)
        window_end = min(len(source), max(right, left + 1) + max_chars // 2)
    if window_end - window_start > max_chars:
        half = max(1, max_chars // 2)
        center = (left + right) // 2
        window_start = max(window_start, center - half)
        window_end = min(window_end, window_start + max_chars)
        window_start = max(0, window_end - max_chars)
    return source[window_start:window_end].strip()


def shadow_types_are_compatible(left: str, right: str) -> bool:
    """Enhanced-only hard type gate for shadow canonical assignment."""

    left_type = str(left or "GenericEntity")
    right_type = str(right or "GenericEntity")
    if left_type in _UNRESOLVED_TYPES or right_type in _UNRESOLVED_TYPES:
        # An unresolved extraction label is missing information, not negative
        # identity evidence.  Treating it as a hard contradiction permanently
        # split exact names such as ``Service``/``service`` and prevented the
        # bounded context-review path from seeing plausible aliases.  Explicit
        # resolved-type contradictions remain fail-closed below.
        return True
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
        self,
        namespace: str,
        normalized: str,
        limit: int,
        *,
        document_id: int | None = None,
        chunk_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.database.search_entity_shadow_name_candidates(
            namespace,
            normalized,
            limit,
            preferred_document_id=document_id,
            preferred_chunk_id=chunk_id,
        )

    def search_vector(
        self,
        namespace: str,
        model_name: str,
        vector: list[float],
        limit: int,
        *,
        document_id: int | None = None,
        chunk_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.database.search_entity_shadow_vector_candidates(
            namespace,
            model_name,
            vector,
            limit,
            preferred_document_id=document_id,
            preferred_chunk_id=chunk_id,
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
            # ``canonical_key`` and ``reason`` are optional at the transport
            # boundary because small schema-capable models frequently emit the
            # correct terminal singleton decision without them.  A merge is
            # repaired only for one bounded candidate below; a missing key
            # with multiple candidates remains fail-closed.
            "required": ["decision"],
        }
        payload = {
            "mention": {
                "surface": mention["surface"],
                "normalized_surface": normalize_name(str(mention["surface"])),
                "type_assertion": mention.get("entity_type") or "GenericEntity",
                "evidence": str(mention.get("evidence") or "")[:1200],
                "source": {
                    "document_id": mention.get("document_id"),
                    "chunk_id": mention.get("chunk_id"),
                    "start": mention.get("start"),
                    "end": mention.get("end"),
                },
            },
            "candidates": [{
                "canonical_key": item["canonical_key"],
                "name": item["name"],
                "normalized_name": str(
                    item.get("normalized_name")
                    or normalize_name(str(item["name"]))
                ),
                "exact_normalized_name_match": (
                    normalize_name(str(mention["surface"]))
                    == str(
                        item.get("normalized_name")
                        or normalize_name(str(item["name"]))
                    )
                ),
                "type_assertion": item.get("primary_type") or "GenericEntity",
                "aliases": item.get("aliases") or [],
                "evidence": str(item.get("mention_evidence") or "")[:1200],
                "source": {
                    "document_id": item.get("mention_document_id"),
                    "chunk_id": item.get("mention_chunk_id"),
                    "start": item.get("mention_evidence_start"),
                    "end": item.get("mention_evidence_end"),
                },
                "same_document": bool(
                    mention.get("document_id") is not None
                    and item.get("mention_document_id") is not None
                    and int(mention["document_id"])
                    == int(item["mention_document_id"])
                ),
                "same_chunk": bool(
                    mention.get("chunk_id") is not None
                    and item.get("mention_chunk_id") is not None
                    and int(mention["chunk_id"])
                    == int(item["mention_chunk_id"])
                ),
                "contained_name_form": bool(
                    normalize_name(str(mention["surface"]))
                    in normalize_name(str(item.get("name") or ""))
                    or normalize_name(str(item.get("name") or ""))
                    in normalize_name(str(mention["surface"]))
                ),
                "scores": item.get("resolution_scores") or {},
            } for item in candidates],
        }
        try:
            result = self.ollama.json_chat(
                model=self.llm_model,
                system=(
                    "Conservatively decide whether a source mention and one bounded candidate "
                    "refer to the same real-world entity. Entity type assertions may be noisy, "
                    "so compare both names and the exact source contexts. Related entities, "
                    "conflicting identifiers, versions, namesakes and homonyms must remain "
                    "separate. An entity-type label difference alone is never evidence that two "
                    "exactly identical normalized names are distinct. For an exact normalized "
                    "name, merge when the contexts are compatible continuations of the same "
                    "entity and there is no explicit distinguishing identifier, version, locale, "
                    "namesake or contradictory identity evidence; choose separate only when such "
                    "source evidence exists. For different names, merge only when the contexts "
                    "support identity, not merely a semantic relationship. In one source chunk, "
                    "an introduced full name followed by a shorter definite name, a person's "
                    "role/title plus the same personal name followed by the bare personal name, "
                    "or a product name followed by the same name plus a generic designator can "
                    "be positive identity evidence when their discourse roles are continuous. "
                    "In particular, within one chunk, an indefinite introduced modified noun "
                    "phrase followed later by the definite shorter head is ordinary local "
                    "coreference: choose merge unless the source explicitly introduces, "
                    "contrasts, counts or assigns different roles to two entities. Do not "
                    "choose separate merely because the shorter wording could be generic in "
                    "isolation. Containment by itself is still insufficient, and a generic head "
                    "that the source identifies as a part, category, host or related object must "
                    "stay separate. Return only "
                    "schema-valid JSON."
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
        raw_decision = str(
            result.get("decision")
            or result.get("result")
            or result.get("status") or ""
        ).strip().lower()
        if raw_decision != "merge":
            return None
        if result.get("decision") != "merge":
            result = {**result, "decision": "merge"}
        # Some schema-capable local models return the correct terminal decision
        # but omit the candidate key.  It is safe to repair only the singleton
        # case because there is no candidate choice to infer; multiple bounded
        # candidates remain fail-closed.
        if not str(result.get("canonical_key") or "") and len(candidates) == 1:
            result = {
                **result,
                "canonical_key": str(candidates[0]["canonical_key"]),
                "reason": str(result.get("reason") or "唯一候选的同一实体证据成立"),
            }
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
        reviewer_cap: int = 5,
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
        self.reviewer_cap = max(1, min(10, int(reviewer_cap)))
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
        if self._cjk_near_edit_identity_ambiguous(
            str(mention["surface"]), str(candidate["name"]),
        ):
            reasons.append("cjk_near_edit_ambiguous")
        if not shadow_types_are_compatible(
            str(mention.get("entity_type") or "GenericEntity"),
            str(candidate.get("primary_type") or "GenericEntity"),
        ):
            reasons.append("type_incompatible")
        return ",".join(reasons)

    @staticmethod
    def _cjk_near_edit_identity_ambiguous(left: str, right: str) -> bool:
        """Keep short CJK near-neighbours separate without explicit alias proof.

        One-character contrasts often carry the entire domain distinction
        (for example intra- versus inter-partition), while embeddings and a
        reranker score the surrounding shared phrase almost identically.  A
        small edit distance is therefore evidence of ambiguity, not identity.
        Exact and contained names remain on their existing exact/nested-review
        paths; this guard covers only non-contained near edits and never tries
        to infer the meaning of the changed characters.
        """

        left_name = normalize_name(str(left or ""))
        right_name = normalize_name(str(right or ""))
        if (
            not left_name
            or not right_name
            or left_name == right_name
            or left_name in right_name
            or right_name in left_name
            or min(len(left_name), len(right_name)) < 3
            or max(len(left_name), len(right_name)) > 48
            or not any("\u3400" <= value <= "\u9fff" for value in left_name)
            or not any("\u3400" <= value <= "\u9fff" for value in right_name)
        ):
            return False
        maximum_distance = 1 if max(len(left_name), len(right_name)) <= 5 else 2
        previous = list(range(len(right_name) + 1))
        for left_index, left_character in enumerate(left_name, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right_name, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                ))
            if min(current) > maximum_distance:
                return False
            previous = current
        return previous[-1] <= maximum_distance

    @staticmethod
    def _nested_surface_identity_ambiguous(
        mention_surface: str, candidate_name: str, *, strict: bool = False,
    ) -> bool:
        """Route strict nested source spans through explicit identity review."""

        mention = re.sub(r"\s+", " ", str(mention_surface or "")).strip().casefold()
        candidate = re.sub(r"\s+", " ", str(candidate_name or "")).strip().casefold()
        if not mention or not candidate or mention == candidate:
            return False
        shorter, longer = (
            (mention, candidate)
            if len(mention) < len(candidate)
            else (candidate, mention)
        )
        if any("\u3400" <= value <= "\u9fff" for value in shorter + longer):
            return shorter in longer
        # Ordinary short-name resolution still needs the reranker/LLM review
        # path (for example ``Acme`` versus ``Acme Holdings``).  Compound or
        # structurally marked source spans are different: their inner token is
        # often a generic head, not an identity continuation.  The explicit
        # strict flag is supplied by endpoint/span recovery; punctuation is a
        # conservative fallback for callers that do not carry that metadata.
        if not strict and not any(mark in shorter + longer for mark in ("-", "/")):
            return False
        pattern = rf"(?<![\w]){re.escape(shorter)}(?![\w])"
        return re.search(pattern, longer, flags=re.IGNORECASE) is not None

    @staticmethod
    def _nested_identity_review_allowed(
        mention: dict[str, Any], candidate: dict[str, Any],
    ) -> bool:
        """Allow model identity review only for bounded alias-shaped nests.

        A shared common-noun head is normally a class/component relation, not
        an alias (``管理`` versus ``设备管理`` or ``platform`` versus
        ``research platform``).  Local models over-merge these singleton
        candidates even when both contexts are shown.  Person names with a
        title/role and parser-numbered section labels are the two generic,
        syntax-supported nested forms retained for review.  Everything else
        stays separate unless an exact alias, must-link or inflectional rule
        already established identity.
        """

        mention_type = str(mention.get("entity_type") or "GenericEntity")
        candidate_type = str(candidate.get("primary_type") or "GenericEntity")
        if "Person" in {mention_type, candidate_type}:
            return True
        mention_surface = re.sub(
            r"\s+", " ", str(mention.get("surface") or ""),
        ).strip()
        candidate_surface = re.sub(
            r"\s+", " ", str(candidate.get("name") or ""),
        ).strip()
        shorter, longer = (
            (mention_surface, candidate_surface)
            if len(mention_surface) <= len(candidate_surface)
            else (candidate_surface, mention_surface)
        )
        if not shorter or not longer.casefold().endswith(shorter.casefold()):
            return False
        prefix = longer[: len(longer) - len(shorter)].strip()
        return re.fullmatch(
            r"(?:§\s*)?\d+(?:\.\d+){1,5}[.):：]?",
            prefix,
        ) is not None

    @staticmethod
    def _inflectional_surface_equivalent(left: str, right: str) -> bool:
        """Recognize conservative Latin singular/plural name variants.

        This is intentionally narrower than stemming: token count and every
        non-inflected token must match, and only ordinary terminal plural
        forms are considered.  It recovers headings such as ``Compound
        statements`` versus a singular occurrence without equating merely
        related nested names.
        """

        def tokens(value: str) -> list[str]:
            normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
            return re.findall(r"[a-z0-9]+", normalized)

        def forms(token: str) -> set[str]:
            values = {token}
            if len(token) <= 3:
                return values
            if token.endswith("ies") and len(token) > 4:
                values.add(token[:-3] + "y")
            if token.endswith("es") and len(token) > 4:
                values.add(token[:-1])
                values.add(token[:-2])
            if token.endswith("s") and not token.endswith(("ss", "us", "is")):
                values.add(token[:-1])
            return values

        left_tokens, right_tokens = tokens(left), tokens(right)
        if (
            not left_tokens
            or len(left_tokens) != len(right_tokens)
            or left_tokens == right_tokens
        ):
            return False
        return all(
            bool(forms(left_token) & forms(right_token))
            for left_token, right_token in zip(left_tokens, right_tokens)
        )

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
                "metadata": {
                    "identity_review_required": str(
                        candidate.get("identity_review_required") or ""
                    ),
                    "inflectional_identity_equivalent": bool(
                        candidate.get("inflectional_identity_equivalent")
                    ),
                },
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
        exact_rows = list(self.store.find_exact(self.namespace, normalized, 10))[:10]
        exact = list({
            str(item["canonical_key"]): item for item in exact_rows
        }.values())
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
            # Duplicate canonicals can be created by earlier parallel or
            # interrupted runs.  If every candidate's canonical name exactly
            # matches this occurrence and every candidate has the same hard
            # type, they represent the same name/type identity under this
            # policy.  Deterministically converge on one key without a model.
            # Alias-only homonyms (whose canonical names differ) remain
            # quarantined, as do type disagreements and explicit cannot-links.
            signatures = {
                (
                    str(item.get("normalized_name") or ""),
                    str(item.get("primary_type") or "GenericEntity"),
                )
                for item in allowed_exact
            }
            if signatures == {(
                normalized,
                str(mention.get("entity_type") or "GenericEntity"),
            )}:
                selected = allowed_exact[0]
                return {
                    "canonical_key": str(selected["canonical_key"]),
                    "decision": "exact_name_type_convergence",
                    "reason": "相同规范名称和硬类型的重复 canonical 已确定性收敛",
                    "scores": selected["resolution_scores"],
                    "embedding": None,
                    "candidates": exact,
                }
            return {
                "canonical_key": self._new_key(mention),
                "decision": "quarantined_ambiguous_exact",
                "reason": "精确别名对应多个 canonical，隔离为 quarantined occurrence 等待人工裁决",
                "scores": {},
                "embedding": None,
                "candidates": exact,
                "quarantined": True,
            }
        # Extraction type labels are assertions rather than identity keys.  An
        # exact canonical name can therefore be blocked solely because the two
        # occurrences received different type labels.  Resolve that narrow
        # case from both exact contexts instead of either splitting forever or
        # weakening the type gate for every similarity candidate.
        cross_type_exact = [
            item for item in exact
            if str(item.get("normalized_name") or "") == normalized
            and str(item.get("filter_reason") or "") == "type_incompatible"
        ]
        if cross_type_exact:
            try:
                reviewed = self.models.review(
                    mention, cross_type_exact[: self.reviewer_cap],
                )
            except Exception as exc:
                logger.warning("entity shadow exact-name type review skipped: %s", exc)
                reviewed = None
            reviewed_key = str((reviewed or {}).get("canonical_key") or "")
            reviewed_candidate = next(
                (
                    item for item in cross_type_exact
                    if str(item["canonical_key"]) == reviewed_key
                ),
                None,
            )
            if reviewed_candidate:
                return {
                    "canonical_key": reviewed_key,
                    "decision": "exact_name_cross_type_review",
                    "reason": str(
                        (reviewed or {}).get("reason")
                        or "双方精确原文支持同一实体，类型标签冲突保留在审计记录中"
                    ),
                    "scores": reviewed_candidate["resolution_scores"],
                    "embedding": None,
                    "candidates": exact,
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
            self.namespace,
            normalized,
            self.lexical_cap,
            document_id=mention.get("document_id"),
            chunk_id=mention.get("chunk_id"),
        ))[: self.lexical_cap]
        vector = list(self.store.search_vector(
            self.namespace,
            embedding_model,
            embedding,
            self.vector_cap,
            document_id=mention.get("document_id"),
            chunk_id=mention.get("chunk_id"),
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
            candidate["inflectional_identity_equivalent"] = bool(
                not candidate["filter_reason"]
                and self._inflectional_surface_equivalent(
                    str(mention.get("surface") or ""),
                    str(candidate.get("name") or ""),
                )
            )
            if (
                not candidate["filter_reason"]
                and not candidate["inflectional_identity_equivalent"]
                and self._nested_surface_identity_ambiguous(
                    str(mention.get("surface") or ""),
                    str(candidate.get("name") or ""),
                    strict=bool(
                        mention.get("endpoint_candidate")
                        or candidate.get("endpoint_candidate")
                    ),
                )
            ):
                # A contained surface can be either a true short-name alias
                # (a qualified role -> its short role / a titled person ->
                # the personal name) or a distinct part/head.  Similarity
                # alone must never merge it, but a
                # hard filter also makes genuine aliases unrecoverable.  Keep
                # it eligible for the bounded context review and record that
                # the automatic path is forbidden.
                if self._nested_identity_review_allowed(mention, candidate):
                    candidate["identity_review_required"] = (
                        "nested_surface_identity_ambiguity"
                    )
                else:
                    candidate["filter_reason"] = ",".join(filter(None, (
                        str(candidate.get("filter_reason") or ""),
                        "nested_identity_ambiguous",
                    )))
            ranked.append(candidate)
        inflectional_candidates = [
            item for item in ranked
            if item.get("inflectional_identity_equivalent")
        ]
        if len(inflectional_candidates) > 1:
            for candidate in inflectional_candidates:
                candidate["identity_review_required"] = (
                    "ambiguous_inflectional_alias"
                )
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
        selected_inflectional = [
            item for item in selected
            if item.get("inflectional_identity_equivalent")
        ]
        if len(selected_inflectional) == 1 and len(inflectional_candidates) == 1:
            equivalent = selected_inflectional[0]
            return {
                "canonical_key": str(equivalent["canonical_key"]),
                "decision": "bounded_inflectional_alias",
                "reason": "名称仅存在保守的单复数词形差异，确定性归一",
                "scores": equivalent["resolution_scores"],
                "embedding": embedding,
                "candidates": ranked,
            }
        if (
            best
            and not best.get("identity_review_required")
            and float(best["resolution_scores"]["combined"])
            >= self.auto_merge_threshold
        ):
            # Similarity is candidate-retrieval evidence, not identity proof.
            # Even a very high embedding/reranker score can collapse nearby
            # concepts.  Keep the high-scoring pair eligible for one bounded
            # context review, but never merge it automatically.
            best["identity_review_required"] = (
                "non_exact_similarity_requires_review"
            )
        if best and float(best["resolution_scores"]["combined"]) >= self.review_threshold:
            # Nested/contained names pose one explicit pairwise coreference
            # question.  Passing unrelated lower-ranked entities into that
            # review makes local models compare categories instead of judging
            # the best pair.  Ordinary ambiguity can still inspect a small,
            # configured shortlist.
            review_candidates = (
                [best]
                if best.get("identity_review_required")
                else selected[: self.reviewer_cap]
            )
            try:
                reviewed = self.models.review(mention, review_candidates)
            except Exception as exc:
                logger.warning("entity shadow candidate review skipped: %s", exc)
                reviewed = None
            reviewed_key = str((reviewed or {}).get("canonical_key") or "")
            reviewed_candidate = next(
                (
                    item for item in review_candidates
                    if str(item["canonical_key"]) == reviewed_key
                ),
                None,
            )
            if reviewed_candidate and not reviewed_candidate["filter_reason"]:
                return {
                    "canonical_key": reviewed_key,
                    "decision": (
                        "bounded_nested_identity_review_merge"
                        if reviewed_candidate.get("identity_review_required")
                        in {
                            "nested_surface_identity_ambiguity",
                            "ambiguous_inflectional_alias",
                        }
                        else "bounded_llm_merge"
                    ),
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
