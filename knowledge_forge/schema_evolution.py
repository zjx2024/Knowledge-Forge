"""Versioned logical schema evaluation and structured graph extraction."""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from .config import Config
from .database import Database
from .ollama import OllamaClient
from .relation_verification import (
    LLM_VERIFIED,
    REJECTED,
    RELATION_ORIENTATIONS,
    RELATION_VERDICTS,
    RULES_VERIFIED,
    UNCERTAIN,
    VERIFICATION_FAILED,
    VERIFICATION_MISSING,
    is_publishable_relation,
    relation_candidate_id,
)
from .value_parsing import coerce_confidence


LOGGER = logging.getLogger(__name__)


BASE_SCHEMA: dict[str, Any] = {
    "name": "Knowledge Forge General Schema",
    "description": "A conservative cross-domain starting schema.",
    "entity_types": [
        {"name": "GenericEntity", "description": "Fallback concept or named object"},
        {"name": "Person", "description": "Natural person or role"},
        {"name": "Organization", "description": "Organization, institution or enterprise"},
        {"name": "Location", "description": "Physical or administrative location"},
        {"name": "Product", "description": "Product, device, component or material"},
        {"name": "Technology", "description": "Technology, method, process or standard"},
        {"name": "Event", "description": "Event, activity, state change or incident"},
    ],
    "relation_types": [
        {"name": "related_to", "source_types": ["*"], "target_types": ["*"], "description": "General semantic relation"},
        {"name": "part_of", "source_types": ["*"], "target_types": ["*"], "description": "Part-whole relation"},
        {"name": "belongs_to", "source_types": ["*"], "target_types": ["*"], "description": "Membership or ownership"},
        {"name": "located_in", "source_types": ["*"], "target_types": ["Location"], "description": "Location relation"},
        {"name": "uses", "source_types": ["*"], "target_types": ["*"], "description": "Use or application"},
        {"name": "produces", "source_types": ["*"], "target_types": ["*"], "description": "Production or output"},
        {"name": "depends_on", "source_types": ["*"], "target_types": ["*"], "description": "Dependency"},
        {"name": "causes", "source_types": ["*"], "target_types": ["*"], "description": "Causal relation"},
    ],
    "rules": {"directed_relations": True, "preserve_provenance": True, "allow_generic_fallback": True},
}


EXTRACTION_FORMAT = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "name": {"type": "string"}, "type": {"type": "string"},
            "description": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["id", "name", "type", "description", "aliases", "evidence", "confidence"]}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string"},
            "description": {"type": "string"}, "evidence": {"type": "string"},
            "confidence": {"type": "number"}},
            "required": ["source", "target", "type", "description", "evidence", "confidence"]}},
    },
    "required": ["entities", "relations"],
}


RELATION_VERIFICATION_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "insufficient_evidence"],
                    },
                    "corrected_type": {"type": "string"},
                    "orientation": {"type": "string", "enum": ["as_is", "reversed"]},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "candidate_id", "verdict", "orientation", "evidence", "reason"
                ],
            },
        },
    },
    "required": ["decisions"],
}


SCHEMA_REVIEW_FORMAT = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entity_type_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence_chunks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description", "confidence", "evidence_chunks"],
            },
        },
        "relation_type_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "source_types": {"type": "array", "items": {"type": "string"}},
                    "target_types": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "evidence_chunks": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "name", "description", "source_types", "target_types", "confidence", "evidence_chunks"
                ],
            },
        },
    },
    "required": ["summary", "entity_type_candidates", "relation_type_candidates"],
}


_LOW_INFORMATION_TITLES = re.compile(
    r"^(目录|目次|contents?|版权(声明)?|修订记录|版本记录|封面|索引|table\s+of\s+contents)$",
    re.IGNORECASE,
)
_RELATION_CUES = re.compile(
    r"(属于|包括|包含|组成|位于|使用|采用|依赖|导致|引起|产生|提供|供应|负责|连接|支持|关联|"
    r"is\s+a|part\s+of|belongs\s+to|located\s+in|uses?|depends\s+on|causes?|produces?|supplies)",
    re.IGNORECASE,
)


def _chunk_key(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_key") or chunk.get("object_key") or chunk.get("id") or "")


def _chunk_section(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return str(
        metadata.get("section_key")
        or metadata.get("section_id")
        or chunk.get("section_id")
        or _chunk_key(chunk).rsplit("_p", 1)[0]
    )


def _chunk_title(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    if metadata.get("section_title"):
        return str(metadata["section_title"]).strip()
    for line in str(chunk.get("content") or "").splitlines():
        value = line.strip().lstrip("#").strip()
        if value:
            return value
    return ""


def _chunk_body(chunk: dict[str, Any]) -> str:
    content = str(chunk.get("content") or "")
    framed = re.match(
        r"(?s)^\s*(?:# [^\n]*\n+)?章节ID\s*:[^\n]*\n+来源文档\s*:[^\n]*\n+"
        r"分块\s*:\s*\d+\s*/\s*\d+\s*\n+(.*)$",
        content,
    )
    if framed:
        return framed.group(1).strip()
    marker = re.search(r"(?:^|\n)分块\s*:[^\n]*\n+", content)
    return content[marker.end():].strip() if marker else content.strip()


def _looks_like_table_of_contents(title: str, body: str) -> bool:
    if _LOW_INFORMATION_TITLES.fullmatch(re.sub(r"\s+", " ", title).strip()):
        return True
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    toc_lines = sum(
        bool(
            re.search(r"\.{3,}\s*\d+\s*$", line)
            or re.fullmatch(r"(?:\d+(?:\.\d+)*)?\s*[^。！？]{2,40}\s+\d+", line)
        )
        for line in lines
    )
    return toc_lines / len(lines) >= 0.5


def _information_score(chunk: dict[str, Any]) -> float:
    body = _chunk_body(chunk)
    meaningful = re.findall(r"[0-9A-Za-z\u4e00-\u9fff]", body)
    if len(meaningful) < Config.schema_sample_min_chars or _looks_like_table_of_contents(_chunk_title(chunk), body):
        return 0.0
    length_score = min(1.0, len(meaningful) / 900)
    sentence_score = min(1.0, len(re.findall(r"[。！？；.!?;]", body)) / 8)
    relation_score = min(1.0, len(_RELATION_CUES.findall(body)) / 6)
    return 0.55 * length_score + 0.20 * sentence_score + 0.25 * relation_score


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _type_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _type_name_exists(name: str, existing: set[str]) -> bool:
    candidate = _type_key(name)
    return any(
        candidate == _type_key(value)
        or SequenceMatcher(None, candidate, _type_key(value)).ratio() >= 0.92
        for value in existing
    )


def _valid_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,63}", value or ""))


class SchemaEvolutionService:
    def __init__(self, database: Database, ollama: OllamaClient | None = None):
        self.db = database
        self.ollama = ollama or OllamaClient()

    def ensure_base(self, project_id: int) -> dict[str, Any]:
        return self.db.ensure_base_schema(project_id, BASE_SCHEMA)

    def activate_base(self, project_id: int) -> dict[str, Any]:
        return self.db.activate_base_schema(project_id, BASE_SCHEMA)

    @staticmethod
    def _evaluated(record: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        value = dict(record)
        value["_evaluation"] = evaluation
        return value

    @staticmethod
    def _adaptive_sample_limits(eligible_chunks: int, section_count: int) -> tuple[int, int]:
        """Scale schema evidence sublinearly while retaining hard resource ceilings."""
        volume_target = math.ceil(math.sqrt(max(1, eligible_chunks)) * 1.5)
        section_target = math.ceil(math.log2(max(1, section_count) + 1) * 4)
        chunk_target = min(
            Config.schema_sample_ceiling_chunks,
            max(Config.schema_sample_base_chunks, volume_target, section_target),
            max(1, eligible_chunks),
        )
        char_target = min(
            Config.schema_sample_ceiling_chars,
            max(Config.schema_sample_base_chars, chunk_target * Config.schema_sample_chars_per_chunk),
        )
        return chunk_target, char_target

    def select_representative_chunks(
        self,
        project: dict[str, Any],
        document_id: int,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Select information-rich, section-spanning and semantically diverse chunks."""
        model = project.get("embedding_model") or Config.default_embedding_model
        vector_rows = self.db.list_document_embeddings(document_id, "chunk", model)
        vectors = {
            str(row["object_key"]): list(row.get("embedding") or [])
            for row in vector_rows
        }
        eligible: list[dict[str, Any]] = []
        for order, raw in enumerate(chunks):
            item = dict(raw)
            item["_schema_order"] = order
            item["_schema_section"] = _chunk_section(item)
            item["_schema_information"] = _information_score(item)
            item["_schema_vector"] = vectors.get(_chunk_key(item), [])
            if item["_schema_information"] > 0:
                eligible.append(item)

        if not eligible:
            fallback = sorted(
                (dict(item, _schema_order=order, _schema_section=_chunk_section(item),
                      _schema_information=0.01, _schema_vector=vectors.get(_chunk_key(item), []))
                 for order, item in enumerate(chunks) if _chunk_body(item).strip()),
                key=lambda item: len(_chunk_body(item)),
                reverse=True,
            )
            eligible = fallback

        by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in eligible:
            by_section[item["_schema_section"]].append(item)
        target_chunks, target_chars = self._adaptive_sample_limits(len(eligible), len(by_section))
        pool: list[dict[str, Any]] = []
        for section_items in by_section.values():
            pool.extend(sorted(
                section_items,
                key=lambda item: (item["_schema_information"], len(_chunk_body(item))),
                reverse=True,
            )[: max(1, Config.schema_sample_per_section)])
        if len(pool) < target_chunks:
            pooled_keys = {_chunk_key(item) for item in pool}
            pool.extend(item for item in eligible if _chunk_key(item) not in pooled_keys)

        max_information = max((item["_schema_information"] for item in pool), default=1.0)
        selected: list[dict[str, Any]] = []
        remaining = list(pool)
        used_chars = 0
        while remaining and len(selected) < target_chunks:
            selected_sections = {item["_schema_section"] for item in selected}

            def selection_score(item: dict[str, Any]) -> float:
                information = item["_schema_information"] / max_information
                vector = item["_schema_vector"]
                similarities = [
                    _cosine(vector, chosen["_schema_vector"])
                    for chosen in selected
                    if vector and chosen["_schema_vector"]
                ]
                diversity = 1.0 - max(similarities) if similarities else (1.0 if not selected else 0.5)
                section_bonus = 1.0 if item["_schema_section"] not in selected_sections else 0.0
                return 0.55 * diversity + 0.30 * information + 0.15 * section_bonus

            choice = max(remaining, key=selection_score)
            remaining.remove(choice)
            remaining_chars = target_chars - used_chars
            if remaining_chars < Config.schema_sample_min_chars and selected:
                break
            sample_text = str(choice.get("content") or "")[: min(
                remaining_chars,
                Config.schema_sample_chars_per_chunk,
            )]
            if not sample_text.strip():
                continue
            choice["_schema_text"] = sample_text
            choice["_schema_score"] = round(selection_score(choice), 4)
            selected.append(choice)
            used_chars += len(sample_text)

        selected.sort(key=lambda item: item["_schema_order"])
        diagnostics = {
            "strategy": "low-information-filter+section-stratification+embedding-diversity",
            "total_chunks": len(chunks),
            "eligible_chunks": len(eligible),
            "filtered_chunks": max(0, len(chunks) - len(eligible)),
            "selected_chunks": [_chunk_key(item) for item in selected],
            "selected_sections": len({item["_schema_section"] for item in selected}),
            "selected_chars": sum(len(item.get("_schema_text") or "") for item in selected),
            "embedding_coverage": sum(bool(item["_schema_vector"]) for item in selected),
            "target_chunks": target_chunks,
            "target_chars": target_chars,
            "chunk_ceiling": Config.schema_sample_ceiling_chunks,
            "char_ceiling": Config.schema_sample_ceiling_chars,
        }
        return selected, diagnostics

    def _should_review(
        self,
        project: dict[str, Any],
        document_id: int,
        samples: list[dict[str, Any]],
    ) -> tuple[bool, dict[str, Any]]:
        project_id = int(project["id"])
        prior_documents = self.db.count_prior_indexed_documents(project_id, document_id)
        if prior_documents == 0:
            return True, {"reason": "initial_document", "prior_documents": 0, "max_novelty": 1.0}

        model = project.get("embedding_model") or Config.default_embedding_model
        similarities: list[float] = []
        for item in samples:
            vector = item.get("_schema_vector") or []
            if not vector:
                continue
            score = self.db.nearest_embedding_score(
                project_id,
                "chunk",
                model,
                vector,
                exclude_document_id=document_id,
            )
            if score is not None:
                similarities.append(max(0.0, min(1.0, score)))
        if not similarities:
            return True, {
                "reason": "comparison_unavailable",
                "prior_documents": prior_documents,
                "max_novelty": None,
            }
        max_novelty = max(1.0 - score for score in similarities)
        should_review = max_novelty >= Config.schema_review_novelty_threshold
        return should_review, {
            "reason": "novel_content" if should_review else "covered_by_active_schema",
            "prior_documents": prior_documents,
            "compared_chunks": len(similarities),
            "max_novelty": round(max_novelty, 4),
            "threshold": Config.schema_review_novelty_threshold,
        }

    @staticmethod
    def _candidate_payload(value: dict[str, Any]) -> dict[str, Any]:
        entities = value.get("entity_type_candidates")
        relations = value.get("relation_type_candidates")
        if entities is None:
            entities = value.get("entity_type_additions") or []
        if relations is None:
            relations = value.get("relation_type_additions") or []
        return {
            "summary": str(value.get("summary") or ""),
            "entity_type_candidates": [item for item in entities or [] if isinstance(item, dict)],
            "relation_type_candidates": [item for item in relations or [] if isinstance(item, dict)],
        }

    @staticmethod
    def _format_samples(samples: list[dict[str, Any]]) -> str:
        blocks = []
        for item in samples:
            blocks.append(
                f"[{_chunk_key(item)}] section={item['_schema_section']} title={_chunk_title(item)}\n"
                f"{item.get('_schema_text') or item.get('content') or ''}"
            )
        return "\n\n---\n\n".join(blocks)

    def _discover_candidates(
        self,
        project: dict[str, Any],
        current: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        model = project.get("llm_model") or Config.default_llm_model
        batches, errors = [], []
        batch_size = max(1, Config.schema_review_batch_size)
        system = (
            "You discover reusable logical knowledge graph types from representative document excerpts. "
            "Propose only domain types that are repeatedly useful and missing from the current schema. "
            "Do not model individual names as types. Never remove, rename or redefine existing types. "
            "Use short English identifiers. Every candidate must cite evidence_chunks using the exact bracketed "
            "chunk keys supplied. Return empty candidate arrays when the general schema is sufficient."
        )
        current_json = json.dumps(current, ensure_ascii=False)
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            try:
                value = self.ollama.json_chat(
                    model=model,
                    system=system,
                    prompt=f"Current schema:\n{current_json}\n\nRepresentative excerpts:\n{self._format_samples(batch)}",
                    schema=SCHEMA_REVIEW_FORMAT,
                    max_tokens=int(project.get("model_max_tokens") or 32768),
                )
                batches.append(self._candidate_payload(value))
            except Exception as exc:
                message = f"batch {start // batch_size + 1}: {exc}"
                errors.append(message)
                LOGGER.warning("Schema candidate discovery failed: %s", message)
        return batches, errors

    def _consolidate_candidates(
        self,
        project: dict[str, Any],
        current: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not candidates:
            return {"summary": "", "entity_type_candidates": [], "relation_type_candidates": []}
        if len(candidates) == 1:
            return candidates[0]
        try:
            return self._candidate_payload(self.ollama.json_chat(
                model=project.get("llm_model") or Config.default_llm_model,
                system=(
                    "You consolidate candidate additions to an existing logical knowledge graph schema. "
                    "Merge synonyms, reject instance names and overly narrow types, preserve all valid evidence chunk keys, "
                    "and keep only additive reusable types supported by the supplied candidates. Use short English identifiers."
                ),
                prompt=(
                    f"Current schema:\n{json.dumps(current, ensure_ascii=False)}\n\n"
                    f"Candidate batches:\n{json.dumps(candidates, ensure_ascii=False)}"
                ),
                schema=SCHEMA_REVIEW_FORMAT,
                max_tokens=int(project.get("model_max_tokens") or 32768),
            ))
        except Exception as exc:
            LOGGER.warning("Schema candidate consolidation failed; using batch candidates: %s", exc)
            return {
                "summary": "；".join(item["summary"] for item in candidates if item["summary"]),
                "entity_type_candidates": [
                    value for item in candidates for value in item["entity_type_candidates"]
                ],
                "relation_type_candidates": [
                    value for item in candidates for value in item["relation_type_candidates"]
                ],
            }

    @staticmethod
    def _evidence_supported(item: dict[str, Any], sample_map: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
        evidence = list(dict.fromkeys(
            str(value).strip() for value in item.get("evidence_chunks") or []
            if str(value).strip() in sample_map
        ))
        confidence = coerce_confidence(item.get("confidence"), 0.0)
        if len(evidence) == 1 and confidence >= Config.schema_review_single_evidence_confidence:
            return True, evidence
        if len(evidence) < Config.schema_review_min_evidence:
            return False, evidence
        sections = {sample_map[key]["_schema_section"] for key in evidence}
        orders = sorted(sample_map[key]["_schema_order"] for key in evidence)
        diverse = len(sections) > 1 or (orders[-1] - orders[0] > 1)
        return diverse, evidence

    def _apply_candidates(
        self,
        current: dict[str, Any],
        review: dict[str, Any],
        samples: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
        updated = copy.deepcopy(current)
        entity_names = {str(item["name"]) for item in updated.get("entity_types", [])}
        relation_names = {str(item["name"]) for item in updated.get("relation_types", [])}
        sample_map = {_chunk_key(item): item for item in samples}
        accepted_entities: list[dict[str, Any]] = []
        accepted_relations: list[dict[str, Any]] = []
        rejected = {"entity_types": 0, "relation_types": 0}

        for item in review.get("entity_type_candidates") or []:
            name = str(item.get("name") or "").strip()
            confidence = coerce_confidence(item.get("confidence"), 0.0)
            supported, evidence = self._evidence_supported(item, sample_map)
            if (
                len(accepted_entities) >= Config.schema_max_entity_additions
                or confidence < Config.schema_review_min_confidence
                or not _valid_identifier(name)
                or _type_name_exists(name, entity_names)
                or not supported
            ):
                rejected["entity_types"] += 1
                continue
            value = {
                "name": name,
                "description": str(item.get("description") or "")[:500],
                "evidence_chunks": evidence,
                "confidence": confidence,
            }
            updated["entity_types"].append({"name": name, "description": value["description"]})
            entity_names.add(name)
            accepted_entities.append(value)

        allowed_types = entity_names | {"*"}
        for item in review.get("relation_type_candidates") or []:
            name = str(item.get("name") or "").strip()
            confidence = coerce_confidence(item.get("confidence"), 0.0)
            sources = list(dict.fromkeys(str(value) for value in item.get("source_types") or ["*"] if str(value) in allowed_types))
            targets = list(dict.fromkeys(str(value) for value in item.get("target_types") or ["*"] if str(value) in allowed_types))
            supported, evidence = self._evidence_supported(item, sample_map)
            if (
                len(accepted_relations) >= Config.schema_max_relation_additions
                or confidence < Config.schema_review_min_confidence
                or not _valid_identifier(name)
                or _type_name_exists(name, relation_names)
                or not sources
                or not targets
                or not supported
            ):
                rejected["relation_types"] += 1
                continue
            value = {
                "name": name,
                "description": str(item.get("description") or "")[:500],
                "source_types": sources,
                "target_types": targets,
                "evidence_chunks": evidence,
                "confidence": confidence,
            }
            updated["relation_types"].append({
                "name": name,
                "description": value["description"],
                "source_types": sources,
                "target_types": targets,
            })
            relation_names.add(name)
            accepted_relations.append(value)
        return updated, accepted_entities, accepted_relations, rejected

    def evaluate_document(
        self,
        project: dict[str, Any],
        document_id: int,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        active = self.ensure_base(int(project["id"]))
        if not Config.schema_auto_apply or not chunks:
            return self._evaluated(active, {"reviewed": False, "reason": "disabled_or_empty"})

        samples, sampling = self.select_representative_chunks(project, document_id, chunks)
        if not samples:
            return self._evaluated(active, {"reviewed": False, "reason": "no_representative_chunks", "sampling": sampling})
        should_review, trigger = self._should_review(project, document_id, samples)
        if not should_review:
            return self._evaluated(active, {
                "reviewed": False,
                "reason": trigger["reason"],
                "sampling": sampling,
                "trigger": trigger,
            })

        current = active["schema_json"]
        batches, errors = self._discover_candidates(project, current, samples)
        review = self._consolidate_candidates(project, current, batches)
        updated, accepted_entities, accepted_relations, rejected = self._apply_candidates(current, review, samples)
        evaluation = {
            "reviewed": True,
            "reason": trigger["reason"],
            "sampling": sampling,
            "trigger": trigger,
            "batch_count": len(batches),
            "batch_errors": errors,
            "accepted_entity_types": accepted_entities,
            "accepted_relation_types": accepted_relations,
            "rejected_candidates": rejected,
        }
        if not accepted_entities and not accepted_relations:
            return self._evaluated(active, evaluation)

        report = {
            "summary": review.get("summary") or "",
            **evaluation,
            "policy": "representative-sampling+evidence-gated+additive-only",
        }
        created = self.db.create_schema_version(int(project["id"]), updated, report, [document_id])
        return self._evaluated(created, evaluation)

    def reassess_project(
        self,
        project: dict[str, Any],
        documents: list[dict[str, Any]],
        progress_callback: Any | None = None,
        completion_guard: Any | None = None,
    ) -> dict[str, Any]:
        """Build a fresh logical schema from the baseline and all active document evidence."""
        project_id = int(project["id"])
        if not documents:
            return self.activate_base(project_id)

        current = copy.deepcopy(BASE_SCHEMA)
        reports: list[dict[str, Any]] = []
        all_document_ids: list[int] = []
        total = len(documents)
        for index, document in enumerate(documents, start=1):
            if completion_guard:
                completion_guard()
            document_id = int(document["id"])
            all_document_ids.append(document_id)
            chunks = [
                item for item in self.db.list_document_chunks(document_id)
                if (item.get("metadata") or {}).get("chunk_role") != "parent"
                and (item.get("metadata") or {}).get("graph_extraction_enabled", True)
            ]
            samples, sampling = self.select_representative_chunks(project, document_id, chunks)
            batches, errors = self._discover_candidates(project, current, samples) if samples else ([], [])
            review = self._consolidate_candidates(project, current, batches)
            updated, entities, relations, rejected = self._apply_candidates(current, review, samples)
            current = updated
            report = {
                "document_id": document_id,
                "document_name": document.get("original_filename") or "",
                "sampling": sampling,
                "accepted_entity_types": entities,
                "accepted_relation_types": relations,
                "rejected_candidates": rejected,
                "batch_errors": errors,
            }
            reports.append(report)
            if progress_callback:
                progress_callback(index, total, report)

        compatibility_report = {
            "summary": "Manual full reassessment from the immutable base schema",
            "policy": "manual-reassessment+representative-sampling+evidence-gated",
            "document_reports": reports,
            "previous_active_version": int(project.get("active_schema_version") or 1),
        }
        if completion_guard:
            completion_guard()
        return self.db.create_schema_version(
            project_id,
            current,
            compatibility_report,
            all_document_ids,
            created_by="manual",
        )

    @staticmethod
    def _flat_extraction(result: dict[str, Any]) -> dict[str, Any]:
        if "items" not in result:
            return result
        items = result.get("items") or []
        return {
            "entities": [item for item in items if isinstance(item, dict) and not item.get("relation")],
            "relations": [
                {
                    "source": item.get("source") or item.get("source_id") or item.get("id"),
                    "target": item.get("target") or item.get("target_id"),
                    "type": item.get("type") or item.get("relation"),
                    "description": item.get("description") or "",
                    "evidence": item.get("evidence") or "",
                    "confidence": 0.5 if item.get("confidence") is None else item.get("confidence"),
                }
                for item in items if isinstance(item, dict) and item.get("relation")
            ],
        }

    @staticmethod
    def _locate_evidence(content: str, evidence: str, alternatives: list[str] | None = None) -> tuple[str, int | None, int | None]:
        candidates = [evidence, *(alternatives or [])]
        for value in candidates:
            raw_candidate = str(value or "").strip()
            if not raw_candidate:
                continue
            candidate = re.sub(r"\s+", " ", raw_candidate)
            position = content.find(candidate)
            if position < 0:
                position = content.casefold().find(candidate.casefold())
            if position >= 0:
                return content[position:position + len(candidate)], position, position + len(candidate)
            pattern = r"\s+".join(re.escape(part) for part in re.split(r"\s+", raw_candidate) if part)
            match = re.search(pattern, content, re.IGNORECASE) if pattern else None
            if match:
                return match.group(0), match.start(), match.end()
        return "", None, None

    @staticmethod
    def _sentence_evidence(content: str, source_names: list[str], target_names: list[str]) -> tuple[str, int | None, int | None]:
        for match in re.finditer(r"[^\n。！？.!?]+(?:[。！？.!?]+|$)", content):
            sentence = match.group(0).strip()
            folded = sentence.casefold()
            if any(name.casefold() in folded for name in source_names if name) and any(
                name.casefold() in folded for name in target_names if name
            ):
                start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
                return sentence, start, start + len(sentence)
        return "", None, None

    @staticmethod
    def _compatible_relation(
        definition: dict[str, Any], source_type: str, target_type: str
    ) -> bool:
        sources = set(definition.get("source_types") or ["*"])
        targets = set(definition.get("target_types") or ["*"])
        return ("*" in sources or source_type in sources) and ("*" in targets or target_type in targets)

    def _sanitize_extraction(
        self,
        result: dict[str, Any],
        content: str,
        logical: dict[str, Any],
    ) -> dict[str, Any]:
        result = self._flat_extraction(result)
        allowed_entities = {item["name"] for item in logical.get("entity_types", [])}
        relation_definitions = {
            item["name"]: item for item in logical.get("relation_types", [])
        }
        entities: list[dict[str, Any]] = []
        local_entities: dict[str, dict[str, Any]] = {}
        rejected_entities = 0
        for raw in result.get("entities") or []:
            if not isinstance(raw, dict):
                continue
            local_id = str(raw.get("id") or "").strip()
            name = re.sub(r"\s+", " ", str(raw.get("name") or "")).strip()
            if not local_id or not name or local_id in local_entities:
                rejected_entities += 1
                continue
            aliases = list(dict.fromkeys(
                re.sub(r"\s+", " ", str(value)).strip()
                for value in raw.get("aliases") or [] if str(value).strip()
            ))
            evidence, start, end = self._locate_evidence(
                content, str(raw.get("evidence") or ""), [name, *aliases]
            )
            if not evidence:
                rejected_entities += 1
                continue
            entity = {
                "id": local_id,
                "name": name,
                "type": raw.get("type") if raw.get("type") in allowed_entities else "GenericEntity",
                "description": str(raw.get("description") or "")[:1000],
                "aliases": aliases,
                "confidence": coerce_confidence(raw.get("confidence"), 0.5),
                "evidence_text": evidence,
                "evidence_start": start,
                "evidence_end": end,
            }
            entities.append(entity)
            local_entities[local_id] = entity

        relations: list[dict[str, Any]] = []
        rejected_relations = 0
        for raw in result.get("relations") or []:
            if not isinstance(raw, dict):
                continue
            source_id = str(raw.get("source") or "").strip()
            target_id = str(raw.get("target") or "").strip()
            source, target = local_entities.get(source_id), local_entities.get(target_id)
            if not source or not target or source_id == target_id:
                rejected_relations += 1
                continue
            relation_type = str(raw.get("type") or "").strip()
            definition = relation_definitions.get(relation_type)
            compatible = bool(
                definition and self._compatible_relation(definition, source["type"], target["type"])
            )
            source_names = [source["name"], *source.get("aliases", [])]
            target_names = [target["name"], *target.get("aliases", [])]
            evidence, start, end = self._locate_evidence(content, str(raw.get("evidence") or ""))
            confidence = coerce_confidence(raw.get("confidence"), 0.5)
            explicit_endpoints = bool(
                evidence
                and any(name.casefold() in evidence.casefold() for name in source_names if name)
                and any(name.casefold() in evidence.casefold() for name in target_names if name)
            )
            relation = {
                "source": source_id,
                "target": target_id,
                "type": relation_type,
                "description": str(raw.get("description") or "")[:1000],
                "confidence": confidence,
                "evidence_text": evidence,
                "evidence_start": start,
                "evidence_end": end,
                "verification_status": UNCERTAIN,
                "verification_reason": (
                    "candidate requires semantic verification"
                    if compatible
                    else "relation type is unknown or incompatible with endpoint types"
                ),
            }
            # Co-occurrence, endpoint presence and model confidence are not
            # sufficient proof of predicate meaning or direction.
            relation["_needs_verification"] = True
            relation["_explicit_endpoints"] = explicit_endpoints
            relation["_original_candidate"] = {
                "source": source_id,
                "source_name": source["name"],
                "target": target_id,
                "target_name": target["name"],
                "type": relation_type,
                "evidence_text": evidence,
                "evidence_start": start,
                "evidence_end": end,
                "confidence": confidence,
                "compatible": compatible,
            }
            relations.append(relation)
        return {
            "entities": entities,
            "relations": relations,
            "_diagnostics": {
                "rejected_entities": rejected_entities,
                "rejected_relations": rejected_relations,
            },
        }

    @staticmethod
    def _merge_extractions(primary: dict[str, Any], supplement: dict[str, Any]) -> dict[str, Any]:
        entities = [dict(item) for item in primary.get("entities") or []]
        relations = [dict(item) for item in primary.get("relations") or []]
        by_identity = {
            (item["type"], re.sub(r"\s+", "", item["name"]).casefold()): item["id"]
            for item in entities
        }
        used_ids = {item["id"] for item in entities}
        id_map: dict[str, str] = {}
        next_id = 1
        for item in supplement.get("entities") or []:
            identity = (item["type"], re.sub(r"\s+", "", item["name"]).casefold())
            if identity in by_identity:
                id_map[item["id"]] = by_identity[identity]
                continue
            local_id = item["id"]
            while local_id in used_ids:
                local_id = f"g{next_id}"
                next_id += 1
            value = {**item, "id": local_id}
            entities.append(value)
            used_ids.add(local_id)
            by_identity[identity] = local_id
            id_map[item["id"]] = local_id
        relation_keys = {
            (item["source"], item["type"], item["target"], item.get("evidence_text") or "")
            for item in relations
        }
        for item in supplement.get("relations") or []:
            source = id_map.get(item["source"], item["source"])
            target = id_map.get(item["target"], item["target"])
            if source not in used_ids or target not in used_ids or source == target:
                continue
            value = {**item, "source": source, "target": target}
            key = (source, value["type"], target, value.get("evidence_text") or "")
            if key not in relation_keys:
                relation_keys.add(key)
                relations.append(value)
        return {"entities": entities, "relations": relations}

    def _relation_candidates(
        self,
        content: str,
        entities: list[dict[str, Any]],
        logical: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Generate local candidates without the quadratic all-project entity comparison."""
        sentences = [
            (match.start(), match.end(), match.group(0).strip())
            for match in re.finditer(r"[^\n。！？.!?]+(?:[。！？.!?]+|$)", content)
            if match.group(0).strip()
        ]
        locations: dict[str, int] = {}
        for entity in entities:
            start = entity.get("evidence_start")
            if start is None:
                continue
            sentence_index = next(
                (index for index, (left, right, _) in enumerate(sentences) if left <= int(start) < right),
                -1,
            )
            if sentence_index >= 0:
                locations[entity["id"]] = sentence_index
        definitions = logical.get("relation_types") or []
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for source in entities:
            source_sentence = locations.get(source["id"])
            if source_sentence is None:
                continue
            for target in entities:
                if source["id"] == target["id"]:
                    continue
                target_sentence = locations.get(target["id"])
                if target_sentence is None or abs(source_sentence - target_sentence) > 1:
                    continue
                pair = (source["id"], target["id"])
                if pair in seen:
                    continue
                left, right = sorted((source_sentence, target_sentence))
                context = " ".join(sentences[index][2] for index in range(left, right + 1))
                if source_sentence != target_sentence and (
                    len(context) > 500 or not _RELATION_CUES.search(context)
                ):
                    continue
                allowed = [
                    definition["name"] for definition in definitions
                    if self._compatible_relation(definition, source["type"], target["type"])
                ]
                if not allowed:
                    continue
                seen.add(pair)
                candidates.append({
                    "source": source["id"],
                    "source_name": source["name"],
                    "target": target["id"],
                    "target_name": target["name"],
                    "allowed_relation_types": allowed,
                    "context": context,
                })
        return candidates

    def _needs_gleaning(
        self,
        chunk: dict[str, Any],
        extraction: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> bool:
        if not Config.extraction_gleaning_enabled:
            return False
        entities = extraction.get("entities") or []
        relations = extraction.get("relations") or []
        body = _chunk_body(chunk)
        if not entities:
            return _information_score(chunk) >= Config.extraction_gleaning_information_score
        return len(entities) >= 2 and not relations and bool(candidates) and bool(_RELATION_CUES.search(body))

    def _verify_relations(
        self,
        project: dict[str, Any],
        logical: dict[str, Any],
        content: str,
        extraction: dict[str, Any],
        schema_version: int,
        chunk_key: str,
    ) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]]]:
        relations = [dict(item) for item in extraction.get("relations") or []]
        entity_map = {item["id"]: item for item in extraction.get("entities") or []}
        relation_definitions = {
            item["name"]: item for item in logical.get("relation_types") or []
        }
        for ordinal, relation in enumerate(relations):
            original = relation.get("_original_candidate") or {}
            relation["candidate_id"] = relation_candidate_id(
                project_scope=project.get("project_key") or project.get("id") or "",
                chunk_key=chunk_key,
                source_mention=original.get("source", relation.get("source")),
                target_mention=original.get("target", relation.get("target")),
                proposed_type=original.get("type", relation.get("type")),
                ordinal=ordinal,
                evidence_start=original.get("evidence_start", relation.get("evidence_start")),
                evidence_end=original.get("evidence_end", relation.get("evidence_end")),
                evidence=original.get("evidence_text", relation.get("evidence_text")),
            )
            relation["candidate_ordinal"] = ordinal

        pending = [item for item in relations if item.get("_needs_verification")]
        diagnostics = {
            "verification_requested": len(pending),
            "verification_candidates": len(pending),
            "rules_verified": sum(
                item.get("verification_status") == RULES_VERIFIED for item in relations
            ),
            "llm_verified": 0,
            "verified": 0,
            "rejected": 0,
            "verification_rejected": 0,
            "uncertain": 0,
            "missing_decisions": 0,
            "verifier_failures": 0,
            "retry_count": 0,
            "quarantined": 0,
            "published": 0,
            "verification_llm_calls": 0,
        }
        events: list[dict[str, Any]] = []
        if pending and Config.extraction_verification_enabled:
            model = project.get("llm_model") or Config.default_llm_model
            batch_size = 8
            for offset in range(0, len(pending), batch_size):
                batch = pending[offset:offset + batch_size]
                missing = self._review_relation_batch(
                    project, logical, content, entity_map, relation_definitions,
                    batch, events, diagnostics, model, schema_version, attempt=1,
                )
                if missing:
                    diagnostics["retry_count"] += 1
                    retry_missing = self._review_relation_batch(
                        project, logical, content, entity_map, relation_definitions,
                        missing, events, diagnostics, model, schema_version, attempt=2,
                    )
                    for relation in retry_missing:
                        relation["verification_status"] = VERIFICATION_MISSING
                        relation["verification_reason"] = "verifier omitted candidate after one retry"
                        diagnostics["missing_decisions"] += 1
        elif pending:
            for relation in pending:
                relation["verification_status"] = UNCERTAIN
                relation["verification_reason"] = "relation verification is disabled"

        for relation in relations:
            relation.pop("_needs_verification", None)
            relation.pop("_explicit_endpoints", None)
            status = relation.get("verification_status") or UNCERTAIN
            if status == LLM_VERIFIED:
                diagnostics["llm_verified"] += 1
            elif status == REJECTED:
                diagnostics["rejected"] += 1
            elif status == UNCERTAIN:
                diagnostics["uncertain"] += 1
            elif status == VERIFICATION_FAILED:
                diagnostics["verifier_failures"] += 1
            if is_publishable_relation(status):
                diagnostics["published"] += 1
            else:
                diagnostics["quarantined"] += 1
        diagnostics["verified"] = diagnostics["llm_verified"] + diagnostics["rules_verified"]
        diagnostics["verification_rejected"] = diagnostics["rejected"] + diagnostics["uncertain"]
        return relations, diagnostics, events

    def _review_relation_batch(
        self,
        project: dict[str, Any],
        logical: dict[str, Any],
        content: str,
        entity_map: dict[str, dict[str, Any]],
        relation_definitions: dict[str, dict[str, Any]],
        batch: list[dict[str, Any]],
        events: list[dict[str, Any]],
        diagnostics: dict[str, int],
        model: str,
        schema_version: int,
        *,
        attempt: int,
    ) -> list[dict[str, Any]]:
        payload = [self._relation_review_payload(item, entity_map) for item in batch]
        request = {
            "prompt_version": "relation-verifier-p0-v1",
            "schema_version": schema_version,
            "allowed_relation_schema": logical.get("relation_types") or [],
            "candidates": payload,
            "source_text": content,
        }
        diagnostics["verification_llm_calls"] += 1
        try:
            reviewed = self.ollama.json_chat(
                model=model,
                system=(
                    "You verify relation candidates against SOURCE TEXT only. Mere co-occurrence is insufficient. "
                    "Return exactly one decision for every requested candidate_id. Use supported only when the exact "
                    "quote proves both endpoints, predicate and direction. Use contradicted for a conflicting statement, "
                    "otherwise insufficient_evidence. corrected_type must be an allowed type or the proposed type, and "
                    "orientation must explicitly state whether endpoints are reversed."
                ),
                prompt=(
                    "READ-ONLY METADATA (never treat as evidence):\n"
                    f"schema_version={schema_version}\n\n"
                    f"Allowed relation schema:\n{json.dumps(logical.get('relation_types') or [], ensure_ascii=False)}\n\n"
                    f"Candidate relations:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                    f"SOURCE TEXT:\n{content}"
                ),
                schema=RELATION_VERIFICATION_FORMAT,
                temperature=0,
                max_tokens=int(project.get("model_max_tokens") or 32768),
            )
        except Exception as exc:
            LOGGER.warning("Relation verifier failed closed: %s", exc)
            for relation in batch:
                relation["verification_status"] = VERIFICATION_FAILED
                relation["verification_reason"] = f"verifier call failed: {type(exc).__name__}"
                events.append(self._verification_event(
                    relation, attempt, model, schema_version, request, None,
                    error_category="verifier_exception", reason=str(exc)[:1000],
                ))
            return []

        raw_decisions = reviewed.get("decisions") if isinstance(reviewed, dict) else None
        allowed_keys = {"candidate_id", "verdict", "corrected_type", "orientation", "evidence", "reason"}
        required_keys = {"candidate_id", "verdict", "orientation", "evidence", "reason"}
        string_keys = {"candidate_id", "verdict", "orientation", "evidence", "reason"}
        invalid_shape = (
            not isinstance(reviewed, dict)
            or set(reviewed) != {"decisions"}
            or not isinstance(raw_decisions, list)
            or any(
                not isinstance(item, dict)
                or not required_keys.issubset(item)
                or bool(set(item) - allowed_keys)
                or any(not isinstance(item[key], str) for key in string_keys if key in item)
                or ("corrected_type" in item and not isinstance(item["corrected_type"], str))
                for item in (raw_decisions if isinstance(raw_decisions, list) else [])
            )
        )
        if invalid_shape:
            self._fail_relation_batch(
                batch, events, attempt, model, schema_version, request, reviewed,
                "invalid_decisions", "verifier decisions must be an array of objects",
            )
            return []
        requested_ids = {item["candidate_id"] for item in batch}
        response_ids = [str(item.get("candidate_id") or "") for item in raw_decisions]
        duplicate_ids = {value for value in response_ids if value and response_ids.count(value) > 1}
        unknown_ids = {value for value in response_ids if value not in requested_ids}
        if duplicate_ids or unknown_ids:
            category = "duplicate_candidate_id" if duplicate_ids else "unknown_candidate_id"
            detail = sorted(duplicate_ids or unknown_ids)
            self._fail_relation_batch(
                batch, events, attempt, model, schema_version, request, reviewed,
                category, f"invalid verifier candidate ids: {detail}",
            )
            return []

        decisions = {str(item.get("candidate_id")): item for item in raw_decisions}
        missing: list[dict[str, Any]] = []
        for relation in batch:
            decision = decisions.get(relation["candidate_id"])
            if decision is None:
                missing.append(relation)
                events.append(self._verification_event(
                    relation, attempt, model, schema_version, request, reviewed,
                    error_category="missing_decision", reason="candidate omitted by verifier",
                ))
                continue
            error = self._apply_relation_decision(
                relation, decision, content, entity_map, relation_definitions,
            )
            if error:
                relation["verification_status"] = VERIFICATION_FAILED
                relation["verification_reason"] = error
            events.append(self._verification_event(
                relation, attempt, model, schema_version, request, reviewed,
                verdict=str(decision.get("verdict") or ""),
                reason=str(error or decision.get("reason") or "")[:1000],
                error_category="invalid_decision" if error else "",
            ))
        return missing

    @staticmethod
    def _relation_review_payload(
        relation: dict[str, Any], entity_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        source = entity_map.get(str(relation.get("source")), {})
        target = entity_map.get(str(relation.get("target")), {})
        return {
            "candidate_id": relation["candidate_id"],
            "source_id": relation.get("source"),
            "source_name": source.get("name") or "",
            "source_type": source.get("type") or "GenericEntity",
            "proposed_type": relation.get("type") or "",
            "target_id": relation.get("target"),
            "target_name": target.get("name") or "",
            "target_type": target.get("type") or "GenericEntity",
            "proposed_evidence": relation.get("evidence_text") or "",
        }

    def _apply_relation_decision(
        self,
        relation: dict[str, Any],
        decision: dict[str, Any],
        content: str,
        entity_map: dict[str, dict[str, Any]],
        relation_definitions: dict[str, dict[str, Any]],
    ) -> str:
        verdict = str(decision.get("verdict") or "")
        orientation = str(decision.get("orientation") or "")
        reason = str(decision.get("reason") or "").strip()
        if verdict not in RELATION_VERDICTS or orientation not in RELATION_ORIENTATIONS or not reason:
            return "verdict, orientation or reason is invalid"
        corrected_type = str(decision.get("corrected_type") or relation.get("type") or "").strip()
        if corrected_type not in relation_definitions:
            return "corrected relation type is not in the allowed schema"
        if verdict == "contradicted":
            relation["verification_status"] = REJECTED
            relation["verification_reason"] = reason
            return ""
        if verdict == "insufficient_evidence":
            relation["verification_status"] = UNCERTAIN
            relation["verification_reason"] = reason
            return ""

        source_id, target_id = str(relation.get("source")), str(relation.get("target"))
        if orientation == "reversed":
            source_id, target_id = target_id, source_id
        source, target = entity_map.get(source_id), entity_map.get(target_id)
        definition = relation_definitions.get(corrected_type)
        if not source or not target or not definition:
            return "corrected endpoints or relation type are not in the allowed schema"
        if not self._compatible_relation(definition, source["type"], target["type"]):
            return "corrected relation type is incompatible with endpoint types"
        evidence = str(decision.get("evidence") or "").strip()
        start = content.find(evidence) if evidence else -1
        if start < 0:
            return "supported decision evidence is not an exact source quote"
        folded = evidence.casefold()
        source_names = [source["name"], *(source.get("aliases") or [])]
        target_names = [target["name"], *(target.get("aliases") or [])]
        if not any(str(name).casefold() in folded for name in source_names if name) or not any(
            str(name).casefold() in folded for name in target_names if name
        ):
            return "supported evidence does not contain both corrected endpoints"
        relation.update({
            "source": source_id,
            "target": target_id,
            "type": corrected_type,
            "evidence_text": evidence,
            "evidence_start": start,
            "evidence_end": start + len(evidence),
            "verification_status": LLM_VERIFIED,
            "verification_reason": reason,
            "orientation": orientation,
        })
        return ""

    def _fail_relation_batch(
        self,
        batch: list[dict[str, Any]],
        events: list[dict[str, Any]],
        attempt: int,
        model: str,
        schema_version: int,
        request: dict[str, Any],
        response: Any,
        category: str,
        reason: str,
    ) -> None:
        for relation in batch:
            relation["verification_status"] = VERIFICATION_FAILED
            relation["verification_reason"] = reason
            events.append(self._verification_event(
                relation, attempt, model, schema_version, request, response,
                error_category=category, reason=reason,
            ))

    @staticmethod
    def _verification_event(
        relation: dict[str, Any],
        attempt: int,
        model: str,
        schema_version: int,
        request: dict[str, Any],
        response: Any,
        *,
        verdict: str = "",
        reason: str = "",
        error_category: str = "",
    ) -> dict[str, Any]:
        return {
            "candidate_id": relation["candidate_id"],
            "attempt": attempt,
            "model_name": model,
            "prompt_version": "relation-verifier-p0-v1",
            "schema_version": schema_version,
            "request_json": request,
            "raw_response": response,
            "verdict": verdict,
            "reason": reason,
            "error_category": error_category,
        }

    def extract_chunk(self, project: dict[str, Any], schema_record: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
        logical = schema_record["schema_json"]
        content = _chunk_body(chunk)
        model = project.get("llm_model") or Config.default_llm_model
        entity_schema = logical.get("entity_types") or []
        relation_schema = logical.get("relation_types") or []
        metadata_context = {
            "title": _chunk_title(chunk),
            "section": _chunk_section(chunk),
            "chunk_key": _chunk_key(chunk),
        }
        primary = self.ollama.json_chat(
            model=model,
            system=(
                "Extract a concise knowledge graph using only facts explicitly stated in the source text. "
                "Every entity and relation must include an exact evidence quote copied from the source. "
                "Entity ids are local unique ids and every relation endpoint must reference one of them. "
                "Do not create entities absent from the text, and do not infer relations from co-occurrence. "
                "Prefer specific allowed types; use GenericEntity or related_to only when the text supports the fact but not a specific type."
            ),
            prompt=(
                f"Entity schema:\n{json.dumps(entity_schema, ensure_ascii=False)}\n\n"
                f"Relation schema:\n{json.dumps(relation_schema, ensure_ascii=False)}\n\n"
                "Read-only context metadata (never extract entities or evidence from it):\n"
                f"{json.dumps(metadata_context, ensure_ascii=False)}\n\nSource text:\n{content}"
            ),
            schema=EXTRACTION_FORMAT,
            max_tokens=int(project.get("model_max_tokens") or 32768),
        )
        extraction = self._sanitize_extraction(primary, content, logical)
        diagnostics = {
            **(extraction.pop("_diagnostics", {})),
            "llm_calls": 1,
            "gleaning_used": False,
        }
        relation_candidates = self._relation_candidates(
            content, extraction.get("entities") or [], logical
        )
        diagnostics["relation_candidates"] = len(relation_candidates)
        if self._needs_gleaning(chunk, extraction, relation_candidates):
            try:
                supplement_raw = self.ollama.json_chat(
                    model=model,
                    system=(
                        "Perform one targeted second-pass knowledge graph extraction. Return only facts missed by the first pass. "
                        "Repeat endpoint entities needed by a new relation. Every item requires an exact source quote. "
                        "Do not add facts based only on co-occurrence."
                    ),
                    prompt=(
                        f"Schema:\n{json.dumps(logical, ensure_ascii=False)}\n\n"
                        f"First-pass result:\n{json.dumps(extraction, ensure_ascii=False)}\n\nSource text:\n{content}"
                        f"\n\nLocally generated candidate entity pairs:\n"
                        f"{json.dumps(relation_candidates, ensure_ascii=False)}"
                    ),
                    schema=EXTRACTION_FORMAT,
                    max_tokens=int(project.get("model_max_tokens") or 32768),
                )
                supplement = self._sanitize_extraction(supplement_raw, content, logical)
                supplement_diagnostics = supplement.pop("_diagnostics", {})
                extraction = self._merge_extractions(extraction, supplement)
                diagnostics.update({
                    "llm_calls": diagnostics["llm_calls"] + 1,
                    "gleaning_used": True,
                    "gleaning_rejected_entities": supplement_diagnostics.get("rejected_entities", 0),
                    "gleaning_rejected_relations": supplement_diagnostics.get("rejected_relations", 0),
                })
            except Exception as exc:
                diagnostics["gleaning_error"] = str(exc)
                LOGGER.warning("Targeted extraction pass failed: %s", exc)

        chunk_scope = (
            f"doc={chunk.get('document_id') or ''}|id={chunk.get('id') or ''}|key={_chunk_key(chunk)}"
        )
        relations, verification, verification_events = self._verify_relations(
            project, logical, content, extraction,
            int(schema_record.get("version") or 1), chunk_scope,
        )
        diagnostics["llm_calls"] += verification.get("verification_llm_calls", 0)
        diagnostics.update(verification)
        extraction["relations"] = relations
        extraction["_verification_events"] = verification_events
        extraction["_diagnostics"] = diagnostics
        return extraction
