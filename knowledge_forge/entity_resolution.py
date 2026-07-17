"""Conservative project-scoped entity resolution with provenance retention."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .database import Database
from .entity_identity import (
    best_name_pair_features,
    credible_aliases,
    entity_identity_text,
    identifiers_conflict,
    name_pair_features,
    names_are_compatible,
    normalize_name,
    types_are_compatible,
)
from .entity_reranker import EntityReranker
from .relation_verification import (
    RELATION_VERIFICATION_POLICY_VERSION,
    UNCERTAIN,
    is_publishable_relation,
)
from .value_parsing import coerce_confidence
from .vector_store import VectorStore


LOGGER = logging.getLogger(__name__)


def lexical_name_similarity(left: str, right: str) -> float:
    return name_pair_features(left, right)["lexical"]


@dataclass
class EntityResolutionSession:
    project_id: int
    embedding_model: str
    llm_model: str
    model_max_tokens: int
    exact_cache: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    exact_candidates_cache: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    query_vector_cache: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    population_cache: dict[str, int] = field(default_factory=dict)


class EntityResolver:
    def __init__(
        self,
        database: Database,
        vectors: VectorStore,
        reranker: EntityReranker | None = None,
    ):
        self.db = database
        self.vectors = vectors
        self.reranker = reranker or EntityReranker()

    def start_session(self, project: dict[str, Any]) -> EntityResolutionSession:
        return EntityResolutionSession(
            project_id=int(project["id"]),
            embedding_model=project.get("embedding_model") or Config.default_embedding_model,
            llm_model=project.get("llm_model") or Config.default_llm_model,
            model_max_tokens=int(project.get("model_max_tokens") or 32768),
        )

    @staticmethod
    def _candidate_score(raw: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
        name_features = best_name_pair_features(raw["name"], candidate)
        lexical = max(float(candidate.get("lexical_score") or 0.0), name_features["lexical"])
        vector = max(0.0, float(candidate.get("vector_score") or 0.0))
        type_score = 1.0 if candidate["entity_type"] == raw["type"] else 0.8
        normalized = normalize_name(raw["name"])
        alias_score = 1.0 if normalized in set(candidate.get("normalized_aliases") or []) else 0.0
        combined = 0.50 * vector + 0.40 * lexical + 0.10 * type_score
        if alias_score:
            combined = max(combined, 0.99)
        return {
            "combined": min(1.0, combined),
            "lexical": lexical,
            "vector": vector,
            "type": type_score,
            "alias": alias_score,
            "exact": name_features["exact"],
            "containment": name_features["containment"],
            "sequence": name_features["sequence"],
            "ngram": name_features["ngram"],
            "acronym": name_features["acronym"],
            "reranker": 0.0,
        }

    def _rerank_candidates(self, raw: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
        adaptive_top_k = min(
            Config.entity_reranker_max_k,
            max(Config.entity_reranker_top_k, math.ceil(math.sqrt(max(1, len(candidates))) * 2)),
        )
        selected = candidates[:adaptive_top_k]
        scores = self.reranker.score(raw, selected)
        if not self.reranker.available:
            return
        for candidate, reranker_score in zip(selected, scores):
            values = candidate["resolution_scores"]
            values["reranker"] = max(0.0, min(1.0, reranker_score))
            values["combined"] = min(
                1.0,
                0.32 * values["vector"]
                + 0.28 * values["lexical"]
                + 0.30 * values["reranker"]
                + 0.10 * values["type"],
            )
            if values["alias"]:
                values["combined"] = max(values["combined"], 0.99)

    def _candidate_limits(
        self,
        session: EntityResolutionSession,
        raw: dict[str, Any],
    ) -> tuple[int, int]:
        entity_type = str(raw.get("type") or "GenericEntity")
        if entity_type not in session.population_cache:
            try:
                session.population_cache[entity_type] = self.db.count_canonical_entities(
                    session.project_id,
                    entity_type,
                )
            except (AttributeError, TypeError):
                session.population_cache[entity_type] = 0
        population = session.population_cache[entity_type]
        growth = max(0, math.ceil(math.log2(max(1, population) + 1)) - 5)
        lexical_top_k = Config.entity_lexical_top_k + growth * 2
        vector_top_k = Config.entity_vector_top_k + growth * 5

        normalized = normalize_name(str(raw.get("name") or ""))
        if entity_type == "GenericEntity" or len(normalized) <= 4:
            lexical_top_k += max(1, Config.entity_lexical_top_k // 2)
            vector_top_k += Config.entity_vector_top_k
        return (
            max(1, min(Config.entity_lexical_max_k, lexical_top_k)),
            max(1, min(Config.entity_vector_max_k, vector_top_k)),
        )

    def _retrieve_candidates(
        self,
        session: EntityResolutionSession,
        raw: dict[str, Any],
        chunk: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        normalized = normalize_name(raw["name"])
        lexical_top_k, vector_top_k = self._candidate_limits(session, raw)
        combined: dict[int, dict[str, Any]] = {}
        lexical_rows = self.db.search_entity_name_candidates(
            session.project_id,
            raw["type"],
            normalized,
            lexical_top_k,
            Config.entity_lexical_recall_threshold,
        )
        for row in lexical_rows:
            combined[int(row["id"])] = dict(row)

        content = entity_identity_text(
            raw["name"], raw["type"], raw.get("description") or "", raw.get("aliases") or [],
        )
        cache_key = (raw["type"], normalized)
        query_vector = session.query_vector_cache.get(cache_key)
        if query_vector is None:
            query_vector = self.vectors.ollama.embed([content], session.embedding_model)[0]
            session.query_vector_cache[cache_key] = query_vector
        vector_rows = self.db.search_entity_vector_candidates(
            session.project_id,
            raw["type"],
            session.embedding_model,
            query_vector,
            vector_top_k,
        )
        for row in vector_rows:
            if float(row.get("vector_score") or 0.0) < Config.entity_vector_recall_threshold:
                continue
            entity_id = int(row["id"])
            current = combined.setdefault(entity_id, dict(row))
            current["vector_score"] = max(
                float(current.get("vector_score") or 0.0),
                float(row.get("vector_score") or 0.0),
            )

        ranked = []
        for candidate in combined.values():
            if not types_are_compatible(raw["type"], candidate["entity_type"]):
                continue
            if identifiers_conflict(raw["name"], candidate["name"]):
                continue
            item = dict(candidate)
            item["resolution_scores"] = self._candidate_score(raw, candidate)
            ranked.append(item)
        ranked.sort(key=lambda item: item["resolution_scores"]["combined"], reverse=True)
        self._rerank_candidates(raw, ranked)
        ranked.sort(
            key=lambda item: (
                item["resolution_scores"]["combined"],
                item["resolution_scores"]["vector"],
                item.get("source_count") or 0,
            ),
            reverse=True,
        )
        return ranked, query_vector

    def _llm_adjudicate(
        self,
        session: EntityResolutionSession,
        raw: dict[str, Any],
        chunk: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str, float]:
        if not Config.entity_llm_review_enabled or not candidates:
            return None, "LLM review disabled or no candidates", 0.0
        selected = candidates[: max(1, Config.entity_llm_max_candidates)]
        schema = {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["merge", "separate"]},
                "candidate_id": {"type": "integer"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": ["decision", "candidate_id", "confidence", "reason"],
        }
        payload = {
            "mention": {
                "name": raw["name"],
                "type": raw["type"],
                "description": raw.get("description") or "",
                "context": str(chunk.get("content") or "")[:3000],
            },
            "candidates": [
                {
                    "id": int(item["id"]),
                    "name": item["name"],
                    "type": item["entity_type"],
                    "aliases": item.get("aliases") or [],
                    "description": item.get("description") or "",
                    "scores": item["resolution_scores"],
                }
                for item in selected
            ],
        }
        try:
            result = self.vectors.ollama.json_chat(
                model=session.llm_model,
                system=(
                    "You perform conservative entity resolution. Merge only when the mention and one candidate "
                    "refer to the same real-world entity, not merely related or similar entities. Conflicting "
                    "model numbers, versions, people, organizations, products, places, or events must remain "
                    "separate. Return only schema-valid JSON."
                ),
                prompt=json.dumps(payload, ensure_ascii=False),
                schema=schema,
                temperature=0,
                max_tokens=session.model_max_tokens,
            )
        except Exception as exc:
            return None, f"LLM review failed: {exc}", 0.0
        confidence = coerce_confidence(result.get("confidence"), 0.0)
        candidate_id = int(result.get("candidate_id") or 0)
        match = next((item for item in selected if int(item["id"]) == candidate_id), None)
        if result.get("decision") != "merge" or confidence < Config.entity_llm_accept_threshold:
            return None, str(result.get("reason") or "LLM kept entities separate"), confidence
        if (
            not match
            or not types_are_compatible(raw["type"], match["entity_type"])
            or identifiers_conflict(raw["name"], match["name"])
        ):
            return None, "LLM selected an invalid or conflicting candidate", confidence
        return match, str(result.get("reason") or "LLM confirmed the same entity"), confidence

    def _resolve_one(
        self,
        session: EntityResolutionSession,
        raw: dict[str, Any],
        chunk: dict[str, Any],
        schema_version: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized = normalize_name(raw["name"])
        cache_key = (raw["type"], normalized)
        cached = session.exact_cache.get(cache_key)
        if cached:
            entity = cached
            exact = [cached]
            decision: dict[str, Any] = {
                "decision": "session_exact", "candidate_ids": [int(cached["id"])],
                "scores": {str(cached["id"]): {"combined": 1.0}}, "reason": "同一文档会话内精确命中",
            }
        else:
            if cache_key in session.exact_candidates_cache:
                exact = session.exact_candidates_cache[cache_key]
            else:
                exact = self.db.find_exact_entity_candidates(
                    session.project_id, raw["type"], normalized,
                )
                session.exact_candidates_cache[cache_key] = exact
            entity = exact[0] if exact else None
        if entity and not cached:
            decision = {
                "decision": "database_exact", "candidate_ids": [int(item["id"]) for item in exact],
                "scores": {str(entity["id"]): {"combined": 1.0}}, "reason": "规范名称或别名精确命中",
            }
        elif not entity:
            candidates, _ = self._retrieve_candidates(session, raw, chunk)
            score_map = {
                str(item["id"]): item["resolution_scores"]
                for item in candidates[: max(Config.entity_lexical_max_k, Config.entity_vector_max_k)]
            }
            best = candidates[0] if candidates else None
            best_scores = best["resolution_scores"] if best else {}
            strong_identity_form = bool(
                best
                and (
                    best_scores.get("alias", 0.0) >= 1.0
                    or best_scores.get("lexical", 0.0) >= Config.entity_auto_merge_lexical_threshold
                    or (
                        best_scores.get("acronym", 0.0) >= 1.0
                        and best_scores.get("vector", 0.0) >= Config.entity_auto_merge_vector_threshold
                        and best_scores.get("reranker", 0.0) >= Config.entity_reranker_auto_threshold
                    )
                )
            )
            if (
                best
                and best_scores["combined"] >= Config.entity_merge_threshold
                and strong_identity_form
            ):
                entity = best
                decision = {
                    "decision": "rule_merge", "candidate_ids": [int(item["id"]) for item in candidates],
                    "scores": score_map, "reason": "名称、类型和向量综合得分达到自动合并阈值",
                }
            elif best and (
                best_scores["combined"] >= Config.entity_llm_review_threshold
                or best_scores["vector"] >= Config.entity_vector_recall_threshold
                or best_scores["reranker"] >= Config.entity_reranker_review_threshold
            ):
                entity, reason, confidence = self._llm_adjudicate(session, raw, chunk, candidates)
                decision = {
                    "decision": "llm_merge" if entity else "llm_separate",
                    "candidate_ids": [int(item["id"]) for item in candidates],
                    "scores": {**score_map, "llm_confidence": confidence}, "reason": reason,
                }
            else:
                decision = {
                    "decision": "new_no_candidate", "candidate_ids": [], "scores": score_map,
                    "reason": "未召回达到复核阈值的候选实体",
                }

        aliases = credible_aliases(raw["name"], raw.get("aliases") or [])
        normalized_aliases = [value for value in (normalize_name(alias) for alias in aliases) if value]
        if not entity:
            key = hashlib.sha256(f"{session.project_id}|{raw['type']}|{normalized}".encode()).hexdigest()
            entity = self.db.upsert_canonical_entity({
                "project_id": session.project_id, "entity_key": key, "name": raw["name"],
                "normalized_name": normalized, "entity_type": raw["type"], "aliases": aliases,
                "normalized_aliases": normalized_aliases, "description": raw.get("description") or "",
                "schema_version": schema_version, "metadata": {"resolution": "new"},
            })
            query_vector = session.query_vector_cache.get(cache_key)
            if query_vector is not None and hasattr(self.vectors, "upsert_entity_embedding_with_vector"):
                self.vectors.upsert_entity_embedding_with_vector(
                    session.project_id, entity, session.embedding_model, query_vector,
                )
            else:
                self.vectors.upsert_entity_embedding(
                    session.project_id, entity, session.embedding_model,
                )
            if decision["decision"] == "llm_separate":
                decision["decision"] = "new_after_llm"
        else:
            type_changed = False
            if entity["entity_type"] == "GenericEntity" and raw["type"] != "GenericEntity":
                entity = self.db.update_canonical_entity_type(int(entity["id"]), raw["type"])
                type_changed = True
            if normalize_name(entity["name"]) != normalized and (
                decision["decision"] == "llm_merge" or names_are_compatible(entity["name"], raw["name"])
            ):
                aliases = list(dict.fromkeys([*aliases, raw["name"]]))
                normalized_aliases = list(dict.fromkeys([*normalized_aliases, normalized]))
            old_aliases = set(entity.get("aliases") or [])
            old_description = entity.get("description") or ""
            evidence_changed = bool(set(aliases) - old_aliases) or len(raw.get("description") or "") > len(old_description)
            entity = self.db.merge_canonical_entity_evidence(
                int(entity["id"]), aliases, normalized_aliases, raw.get("description") or "",
            )
            if type_changed or evidence_changed:
                self.vectors.upsert_entity_embedding(session.project_id, entity, session.embedding_model)

        session.exact_cache[cache_key] = entity
        return entity, decision

    def resolve_chunk(
        self,
        project: dict[str, Any],
        document_id: int,
        chunk: dict[str, Any],
        extraction: dict[str, Any],
        schema_version: int,
        session: EntityResolutionSession | None = None,
        ingestion_run_id: str = "",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        project_id = int(project["id"])
        session = session or self.start_session(project)
        raw_entities = [
            raw for raw in (extraction.get("entities") or [])
            if normalize_name(raw.get("name") or "")
        ]
        unique_raws: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in raw_entities:
            key = (str(raw["type"]), normalize_name(raw["name"]))
            unique_raws.setdefault(key, raw)
        exact_requests = [
            key for key in unique_raws
            if key not in session.exact_cache and key not in session.exact_candidates_cache
        ]
        if exact_requests:
            if hasattr(self.db, "find_exact_entity_candidates_batch"):
                session.exact_candidates_cache.update(
                    self.db.find_exact_entity_candidates_batch(project_id, exact_requests)
                )
            else:
                for entity_type, normalized in exact_requests:
                    session.exact_candidates_cache[(entity_type, normalized)] = (
                        self.db.find_exact_entity_candidates(project_id, entity_type, normalized)
                    )
        embedding_requests = [
            (key, raw)
            for key, raw in unique_raws.items()
            if key not in session.exact_cache
            and not session.exact_candidates_cache.get(key)
            and key not in session.query_vector_cache
        ]
        batch_size = max(1, Config.entity_embedding_batch_size)
        for start in range(0, len(embedding_requests), batch_size):
            batch = embedding_requests[start : start + batch_size]
            contents = [
                entity_identity_text(
                    raw["name"], raw["type"], raw.get("description") or "",
                    raw.get("aliases") or [],
                )
                for _, raw in batch
            ]
            vectors = self.vectors.ollama.embed(contents, session.embedding_model)
            if len(vectors) != len(batch):
                raise RuntimeError("实体 embedding 批次返回数量不一致")
            for (key, _), vector in zip(batch, vectors):
                session.query_vector_cache[key] = vector
        local_map: dict[str, dict[str, Any]] = {}
        mention_values: list[dict[str, Any]] = []
        decision_values: list[dict[str, Any]] = []
        for raw in raw_entities:
            normalized = normalize_name(raw["name"])
            if not normalized: continue
            aliases = credible_aliases(raw["name"], raw.get("aliases") or [])
            entity, resolution = self._resolve_one(session, raw, chunk, schema_version)
            local_map[raw["id"]] = entity
            mention_values.append({"project_id": project_id, "document_id": document_id, "chunk_id": int(chunk["id"]),
                "canonical_entity_id": int(entity["id"]), "mention_text": raw["name"], "normalized_name": normalized,
                "entity_type": raw["type"], "description": raw.get("description") or "",
                "confidence": 0.5 if raw.get("confidence") is None else raw.get("confidence"),
                "extraction_version": "v2", "evidence_text": raw.get("evidence_text") or "",
                "evidence_start": raw.get("evidence_start"), "evidence_end": raw.get("evidence_end"),
                "metadata": {"raw_aliases": raw.get("raw_aliases") or [], "accepted_aliases": aliases,
                    "verified_surface_forms": raw.get("verified_surface_forms") or [],
                    "resolution": resolution}})
            decision_values.append({
                "project_id": project_id, "document_id": document_id, "chunk_id": int(chunk["id"]),
                "canonical_entity_id": int(entity["id"]), "mention_text": raw["name"],
                "normalized_name": normalized, "entity_type": raw["type"], "decision": resolution["decision"],
                "candidate_entity_ids": resolution.get("candidate_ids") or [], "scores": resolution.get("scores") or {},
                "reason": resolution.get("reason") or "", "model_name": session.llm_model,
                "metadata": {"schema_version": schema_version},
            })
        if hasattr(self.db, "persist_entity_resolution_batch"):
            mentions = self.db.persist_entity_resolution_batch(mention_values, decision_values)
        else:
            mentions = []
            for mention_values_item, decision_values_item in zip(mention_values, decision_values):
                mentions.append(self.db.add_entity_mention(mention_values_item))
                self.db.add_entity_resolution_decision(decision_values_item)
        raw_entity_map = {str(item["id"]): item for item in extraction.get("entities") or []}
        candidate_values: list[dict[str, Any]] = []
        for raw in extraction.get("relations") or []:
            original = raw.get("_original_candidate") or {}
            original_source_id = str(original.get("source") or raw.get("source") or "")
            original_target_id = str(original.get("target") or raw.get("target") or "")
            original_source = raw_entity_map.get(original_source_id, {})
            original_target = raw_entity_map.get(original_target_id, {})
            candidate_values.append({
                "candidate_id": raw["candidate_id"], "project_id": project_id,
                "document_id": document_id, "chunk_id": int(chunk["id"]),
                "candidate_ordinal": int(raw.get("candidate_ordinal") or 0),
                "source_mention_id": original_source_id,
                "source_name": original.get("source_name") or original_source.get("name") or "",
                "target_mention_id": original_target_id,
                "target_name": original.get("target_name") or original_target.get("name") or "",
                "proposed_relation_type": original.get("type") or raw.get("type") or "",
                "proposed_evidence": original.get("evidence_text") or "",
                "evidence_start": original.get("evidence_start"),
                "evidence_end": original.get("evidence_end"),
                "model_confidence": (
                    0.5 if original.get("confidence") is None else original.get("confidence")
                ),
                "status": raw.get("verification_status") or UNCERTAIN,
                "ingestion_run_id": ingestion_run_id,
                "reason": raw.get("verification_reason") or "",
                "corrected_relation_type": (
                    raw.get("type") if raw.get("type") != original.get("type") else None
                ),
                "orientation": raw.get("orientation") or "as_is",
                "metadata": {
                    "schema_version": schema_version,
                    "verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION,
                    "original_candidate": original,
                },
            })
        event_values = [
            {
                **event, "project_id": project_id, "document_id": document_id,
                "chunk_id": int(chunk["id"]),
                "ingestion_run_id": ingestion_run_id,
            }
            for event in extraction.get("_verification_events") or []
        ]
        assertion_values: list[dict[str, Any]] = []
        assertion_context: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for raw in extraction.get("relations") or []:
            source, target = local_map.get(raw["source"]), local_map.get(raw["target"])
            if not source or not target or source["id"] == target["id"]: continue
            relation_key = hashlib.sha256(
                f"{project_id}|{document_id}|{chunk['id']}|{raw.get('candidate_id')}|"
                f"{source['id']}|{raw['type']}|{target['id']}".encode()
            ).hexdigest()
            assertion_values.append({"project_id": project_id, "document_id": document_id,
                "chunk_id": int(chunk["id"]), "source_entity_id": int(source["id"]), "target_entity_id": int(target["id"]),
                "relation_key": relation_key, "relation_type": raw["type"], "description": raw.get("description") or "",
                "confidence": 0.5 if raw.get("confidence") is None else raw.get("confidence"),
                "extraction_version": "v2", "ingestion_run_id": ingestion_run_id,
                "evidence_text": raw.get("evidence_text") or "", "evidence_start": raw.get("evidence_start"),
                "evidence_end": raw.get("evidence_end"),
                "verification_status": raw.get("verification_status") or "unverified",
                "metadata": {"schema_version": schema_version,
                             "verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION,
                             "candidate_id": raw.get("candidate_id"),
                             "verification_reason": raw.get("verification_reason") or "",
                             "verification_status": raw.get("verification_status") or "unverified"}})
            assertion_context.append((source, target))

        relations = self.db.persist_relation_audit_chain(
            candidate_values,
            event_values,
            assertion_values,
        )
        for relation, (source, target) in zip(relations, assertion_context):
            if is_publishable_relation(
                relation.get("verification_status"), relation.get("metadata"),
            ):
                try:
                    self.vectors.upsert_relation_embedding(
                        project_id, document_id, int(chunk["id"]), relation,
                        source["name"], target["name"], session.embedding_model,
                    )
                    self.db.complete_indexing_failure(
                        project_id, "relation", str(relation["id"]), session.embedding_model,
                    )
                except Exception as exc:
                    self.db.record_indexing_failure({
                        "project_id": project_id,
                        "document_id": document_id,
                        "chunk_id": int(chunk["id"]),
                        "object_type": "relation",
                        "object_key": str(relation["id"]),
                        "model_name": session.embedding_model,
                        "last_error": str(exc),
                        "payload": {
                            "source_name": source["name"],
                            "target_name": target["name"],
                            "relation_type": relation["relation_type"],
                            "relation_key": relation["relation_key"],
                        },
                    })
                    relation["indexing_pending"] = True
                    LOGGER.warning(
                        "Relation embedding deferred for assertion %s: %s",
                        relation.get("id"), exc,
                    )
        return mentions, relations
