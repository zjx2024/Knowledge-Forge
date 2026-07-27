"""Versioned logical schema evaluation and structured graph extraction."""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from bisect import bisect_right
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any

from .config import Config
from .database import Database
from .ollama import OllamaClient, OllamaProtocolError
from .observability import gauge, increment
from .relation_verification import (
    LLM_VERIFIED,
    REJECTED,
    RELATION_VERIFICATION_POLICY_VERSION,
    RELATION_ORIENTATIONS,
    RELATION_VERDICTS,
    RULES_VERIFIED,
    UNCERTAIN,
    VERIFICATION_FAILED,
    VERIFICATION_MISSING,
    is_publishable_relation,
    relation_candidate_id,
)
from .sentence_spans import split_sentences
from .value_parsing import coerce_confidence


LOGGER = logging.getLogger(__name__)


def _versioned_relation_policy(
    name: str,
    source_types: list[str],
    target_types: list[str],
    semantic_definition: str,
    direction_description: str,
    *,
    publish_mode: str = "auto",
    optional_soft_cues: list[str] | None = None,
) -> dict[str, Any]:
    """Describe predicate behavior as data; v2 never branches on ``name``."""

    return {
        "name": name,
        "source_types": list(source_types),
        "target_types": list(target_types),
        "description": semantic_definition,
        "semantic_definition": semantic_definition,
        "domain_types": list(source_types),
        "range_types": list(target_types),
        "direction_description": direction_description,
        "symmetric": False,
        "inverse_predicate": None,
        "canonical_orientation": "forward",
        "publish_mode": publish_mode,
        "reflexive": False,
        "positive_examples": [],
        "negative_examples": [],
        "optional_soft_cues": list(optional_soft_cues or []),
    }


BASE_SCHEMA: dict[str, Any] = {
    "version": "general-schema-v2-predicate-policy-v1",
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
        _versioned_relation_policy(
            "related_to", ["*"], ["*"], "General semantic relation.",
            "The subject is semantically related to the object; no inverse role is implied.",
            publish_mode="review_only",
            optional_soft_cues=["related", "关联"],
        ),
        _versioned_relation_policy(
            "part_of", ["*"], ["*"], "The subject is a component or part of the object.",
            "Orient from the part as subject to the whole as object.",
            optional_soft_cues=["part of", "组成"],
        ),
        _versioned_relation_policy(
            "belongs_to", ["*"], ["*"], "The subject is a member of or owned by the object.",
            "Orient from the member or owned item to the owning object.",
            optional_soft_cues=["belongs to", "属于"],
        ),
        _versioned_relation_policy(
            "located_in", ["*"], ["Location"], "The subject is physically or administratively located in the object.",
            "Orient from the located subject to the containing location.",
            optional_soft_cues=["located in", "位于"],
        ),
        _versioned_relation_policy(
            "uses", ["*"], ["*"], "The subject applies or consumes the object.",
            "Orient from the user as subject to the used object.",
            optional_soft_cues=["uses", "使用"],
        ),
        _versioned_relation_policy(
            "produces", ["*"], ["*"], "The subject creates or outputs the object.",
            "Orient from the producer as subject to the produced output.",
            optional_soft_cues=["produces", "生产"],
        ),
        _versioned_relation_policy(
            "depends_on", ["*"], ["*"], "The subject requires the object for operation or existence.",
            "Orient from the dependent subject to the dependency object.",
            optional_soft_cues=["depends on", "依赖"],
        ),
        _versioned_relation_policy(
            "causes", ["*"], ["*"], "The subject brings about the object.",
            "Orient from the cause as subject to the resulting object.",
            optional_soft_cues=["causes", "导致"],
        ),
    ],
    "rules": {"directed_relations": True, "preserve_provenance": True, "allow_generic_fallback": True},
}


EXTRACTION_FORMAT = {
    "type": "object",
    "properties": {
        "entities": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "name": {"type": "string"}, "type": {"type": "string"},
            "description": {"type": "string"}, "aliases": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "string"},
            "evidence_start": {"type": ["integer", "null"]},
            "evidence_end": {"type": ["integer", "null"]},
            "mention_start": {"type": ["integer", "null"]},
            "mention_end": {"type": ["integer", "null"]},
            "confidence": {"type": "number"}},
            "required": ["id", "name", "type", "description", "aliases", "evidence", "confidence"]}},
        "relations": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string"},
            "description": {"type": "string"}, "evidence": {"type": "string"},
            "evidence_start": {"type": ["integer", "null"]},
            "evidence_end": {"type": ["integer", "null"]},
            "confidence": {"type": "number"}},
            "required": ["source", "target", "type", "description", "evidence", "confidence"]}},
    },
    "required": ["entities", "relations"],
}

_ENTITY_ONLY_ITEM_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "type": {"type": "string"},
        "mention_start": {"type": "integer"},
        "mention_end": {"type": "integer"},
        "confidence": {"type": "number"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
    },
    "required": [
        "id", "name", "type", "mention_start", "mention_end", "confidence",
    ],
}

ENTITY_ONLY_EXTRACTION_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            **EXTRACTION_FORMAT["properties"]["entities"],
            "items": _ENTITY_ONLY_ITEM_FORMAT,
            "maxItems": Config.entity_only_max_items_per_segment,
        },
    },
    "required": ["entities"],
}

RELATION_ONLY_EXTRACTION_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "relations": copy.deepcopy(EXTRACTION_FORMAT["properties"]["relations"]),
    },
    "required": ["relations"],
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
                    "semantic_definition": {"type": "string"},
                    "domain_types": {"type": "array", "items": {"type": "string"}},
                    "range_types": {"type": "array", "items": {"type": "string"}},
                    "direction_description": {"type": "string"},
                    "symmetric": {"type": "boolean"},
                    "inverse_predicate": {"type": ["string", "null"]},
                    "canonical_orientation": {
                        "type": "string",
                        "enum": ["forward", "inverse", "symmetric"],
                    },
                    "publish_mode": {
                        "type": "string",
                        "enum": ["auto", "review_only", "disabled"],
                    },
                    "reflexive": {"type": "boolean"},
                    "positive_examples": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "negative_examples": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "optional_soft_cues": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "confidence": {"type": "number"},
                    "evidence_chunks": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "name", "description", "source_types", "target_types",
                    "semantic_definition", "domain_types", "range_types",
                    "direction_description", "symmetric", "inverse_predicate",
                    "canonical_orientation", "publish_mode", "reflexive",
                    "positive_examples", "negative_examples",
                    "optional_soft_cues", "confidence", "evidence_chunks",
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


def _chunk_body_with_offset(chunk: dict[str, Any]) -> tuple[str, int]:
    content = str(chunk.get("content") or "")
    framed = re.match(
        r"(?s)^# [^\n]*\n\n章节ID: [^\n]*\n来源文档: [^\n]*\n\n"
        r"分块:\s*\d+/\d+\n\n(.*)$",
        content,
    )
    if framed:
        raw_body = framed.group(1)
        leading = len(raw_body) - len(raw_body.lstrip())
        return raw_body.strip(), framed.start(1) + leading
    leading = len(content) - len(content.lstrip())
    return content.strip(), leading


def _chunk_body(chunk: dict[str, Any]) -> str:
    return _chunk_body_with_offset(chunk)[0]


def _surface_spans(content: str, surface: str) -> list[tuple[int, int]]:
    """Locate source-backed surface forms without ASCII substring collisions."""

    value = str(surface or "").strip()
    if not value:
        return []
    parts = [part for part in re.split(r"\s+", value) if part]
    core = r"\s+".join(re.escape(part) for part in parts)
    if not core:
        return []
    if value.isascii() and re.match(r"[A-Za-z0-9]", value) and re.search(r"[A-Za-z0-9]$", value):
        core = rf"(?<![A-Za-z0-9]){core}(?![A-Za-z0-9])"
    return [(match.start(), match.end()) for match in re.finditer(core, content, re.IGNORECASE)]


def _contains_surface(content: str, surface: str) -> bool:
    return bool(_surface_spans(content, surface))


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
            "Every relation candidate must explicitly define its complete versioned semantic policy: "
            "semantic definition, domain/range types, direction, symmetry, inverse predicate or null, "
            "canonical orientation, publish mode, reflexivity, positive/negative examples, and optional "
            "soft cues. Soft cues are non-binding model hints, never deterministic rules. "
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
                    context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                    max_output_tokens=Config.schema_review_max_output_tokens,
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
                    "and keep only additive reusable types supported by the supplied candidates. Preserve every explicit "
                    "relation policy field; never infer behavior from a predicate name. Use short English identifiers."
                ),
                prompt=(
                    f"Current schema:\n{json.dumps(current, ensure_ascii=False)}\n\n"
                    f"Candidate batches:\n{json.dumps(candidates, ensure_ascii=False)}"
                ),
                schema=SCHEMA_REVIEW_FORMAT,
                context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                max_output_tokens=Config.schema_review_max_output_tokens,
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
            sources = list(dict.fromkeys(
                str(value) for value in item.get("source_types") or []
                if str(value) in allowed_types
            ))
            targets = list(dict.fromkeys(
                str(value) for value in item.get("target_types") or []
                if str(value) in allowed_types
            ))
            domains = list(dict.fromkeys(
                str(value) for value in item.get("domain_types") or []
                if str(value) in allowed_types
            ))
            ranges = list(dict.fromkeys(
                str(value) for value in item.get("range_types") or []
                if str(value) in allowed_types
            ))
            semantic_definition = str(
                item.get("semantic_definition") or ""
            ).strip()[:1000]
            direction_description = str(
                item.get("direction_description") or ""
            ).strip()[:1000]
            canonical_orientation = str(
                item.get("canonical_orientation") or ""
            ).strip()
            publish_mode = str(item.get("publish_mode") or "").strip()
            symmetric = item.get("symmetric")
            reflexive = item.get("reflexive")
            inverse_value = item.get("inverse_predicate")
            inverse_predicate = (
                str(inverse_value).strip()
                if isinstance(inverse_value, str) else ""
            )
            positive_examples = item.get("positive_examples")
            negative_examples = item.get("negative_examples")
            optional_soft_cues = item.get("optional_soft_cues")
            supported, evidence = self._evidence_supported(item, sample_map)
            if (
                len(accepted_relations) >= Config.schema_max_relation_additions
                or confidence < Config.schema_review_min_confidence
                or not _valid_identifier(name)
                or _type_name_exists(name, relation_names)
                or not sources
                or not targets
                or sources != domains
                or targets != ranges
                or not semantic_definition
                or not direction_description
                or not isinstance(symmetric, bool)
                or not isinstance(reflexive, bool)
                or canonical_orientation not in {"forward", "inverse", "symmetric"}
                or publish_mode not in {"auto", "review_only", "disabled"}
                or symmetric != (canonical_orientation == "symmetric")
                or (
                    canonical_orientation == "inverse"
                    and (
                        not inverse_predicate
                        or inverse_predicate not in relation_names
                    )
                )
                or (symmetric and bool(inverse_predicate))
                or (
                    inverse_predicate
                    and inverse_predicate not in relation_names
                )
                or not isinstance(positive_examples, list)
                or not all(isinstance(value, str) for value in positive_examples)
                or not isinstance(negative_examples, list)
                or not all(isinstance(value, str) for value in negative_examples)
                or not isinstance(optional_soft_cues, list)
                or not all(isinstance(value, str) for value in optional_soft_cues)
                or not supported
            ):
                rejected["relation_types"] += 1
                continue
            value = {
                "name": name,
                "description": str(item.get("description") or semantic_definition)[:500],
                "source_types": sources,
                "target_types": targets,
                "semantic_definition": semantic_definition,
                "domain_types": domains,
                "range_types": ranges,
                "direction_description": direction_description,
                "symmetric": symmetric,
                "inverse_predicate": inverse_predicate or None,
                "canonical_orientation": canonical_orientation,
                "publish_mode": publish_mode,
                "reflexive": reflexive,
                "positive_examples": list(positive_examples),
                "negative_examples": list(negative_examples),
                "optional_soft_cues": list(optional_soft_cues),
                "evidence_chunks": evidence,
                "confidence": confidence,
            }
            updated["relation_types"].append({
                key: candidate_value
                for key, candidate_value in value.items()
                if key not in {"evidence_chunks", "confidence"}
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
    def _locate_evidence(
        content: str,
        evidence: str,
        alternatives: list[str] | None = None,
        *,
        near: int | None = None,
    ) -> tuple[str, int | None, int | None]:
        candidates = [evidence, *(alternatives or [])]
        for value in candidates:
            raw_candidate = str(value or "").strip()
            if not raw_candidate:
                continue
            pattern = r"\s+".join(re.escape(part) for part in re.split(r"\s+", raw_candidate) if part)
            matches = list(re.finditer(pattern, content, re.IGNORECASE)) if pattern else []
            if matches:
                match = min(matches, key=lambda item: abs(item.start() - near)) if near is not None else matches[0]
                return match.group(0), match.start(), match.end()
        return "", None, None

    @staticmethod
    def _sentence_evidence(content: str, source_names: list[str], target_names: list[str]) -> tuple[str, int | None, int | None]:
        for start, end, sentence in split_sentences(content):
            folded = sentence.casefold()
            if any(name.casefold() in folded for name in source_names if name) and any(
                name.casefold() in folded for name in target_names if name
            ):
                return sentence, start, end
        return "", None, None

    @staticmethod
    def _exact_source_span(
        content: str,
        start: Any,
        end: Any,
        surfaces: list[str],
    ) -> tuple[int, int] | None:
        try:
            left, right = int(start), int(end)
        except (TypeError, ValueError):
            return None
        if left < 0 or right <= left or right > len(content):
            return None
        actual = re.sub(r"\s+", " ", content[left:right]).strip().casefold()
        expected = {
            re.sub(r"\s+", " ", value).strip().casefold()
            for value in surfaces if str(value).strip()
        }
        return (left, right) if actual in expected else None

    @staticmethod
    def _sentence_for_span(content: str, start: int, end: int) -> tuple[str, int, int]:
        for sentence_start, sentence_end, sentence in split_sentences(content):
            if sentence_start <= start and end <= sentence_end:
                return sentence, sentence_start, sentence_end
        return content[start:end], start, end

    @staticmethod
    def _sentence_for_two_spans(
        content: str,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> tuple[str, int | None, int | None]:
        for sentence_start, sentence_end, sentence in split_sentences(content):
            if (
                sentence_start <= first[0] < first[1] <= sentence_end
                and sentence_start <= second[0] < second[1] <= sentence_end
            ):
                return sentence, sentence_start, sentence_end
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
            raw_aliases = list(dict.fromkeys(
                re.sub(r"\s+", " ", str(value)).strip()
                for value in raw.get("aliases") or [] if str(value).strip()
            ))
            name_spans = _surface_spans(content, name)
            if not name_spans:
                rejected_entities += 1
                continue
            aliases = [alias for alias in raw_aliases if _surface_spans(content, alias)]
            verified_surface_forms = list(dict.fromkeys([name, *aliases]))
            mention_span = self._exact_source_span(
                content, raw.get("mention_start"), raw.get("mention_end"),
                verified_surface_forms,
            )
            if mention_span is None:
                if len(name_spans) != 1:
                    rejected_entities += 1
                    continue
                mention_span = name_spans[0]
            mention_start, mention_end = mention_span
            raw_evidence = str(raw.get("evidence") or "")
            evidence_span = self._exact_source_span(
                content, raw.get("evidence_start"), raw.get("evidence_end"),
                [raw_evidence],
            )
            if evidence_span is not None and (
                evidence_span[0] <= mention_start < mention_end <= evidence_span[1]
            ):
                start, end = evidence_span
                evidence = content[start:end]
            else:
                inferred_evidence, inferred_start, inferred_end = self._locate_evidence(
                    content, raw_evidence, near=mention_start,
                )
                if (
                    inferred_start is not None
                    and inferred_end is not None
                    and inferred_start <= mention_start < mention_end <= inferred_end
                ):
                    evidence, start, end = inferred_evidence, inferred_start, inferred_end
                else:
                    evidence, start, end = self._sentence_for_span(
                        content, mention_start, mention_end,
                    )
            entity = {
                "id": local_id,
                "name": name,
                "type": raw.get("type") if raw.get("type") in allowed_entities else "GenericEntity",
                "description": str(raw.get("description") or "")[:1000],
                "aliases": aliases,
                "raw_aliases": raw_aliases,
                "verified_surface_forms": verified_surface_forms,
                "confidence": coerce_confidence(raw.get("confidence"), 0.5),
                "evidence_text": evidence,
                "evidence_start": start,
                "evidence_end": end,
                "mention_start": mention_start,
                "mention_end": mention_end,
                "alias_spans": [
                    {
                        "surface": alias,
                        "start": alias_span[0],
                        "end": alias_span[1],
                        "confidence": coerce_confidence(raw.get("confidence"), 0.5),
                    }
                    for alias in aliases
                    for alias_span in _surface_spans(content, alias)[:1]
                ],
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
            source_names = source.get("verified_surface_forms") or [source["name"]]
            target_names = target.get("verified_surface_forms") or [target["name"]]
            raw_evidence = str(raw.get("evidence") or "")
            evidence_span = self._exact_source_span(
                content, raw.get("evidence_start"), raw.get("evidence_end"),
                [raw_evidence],
            )
            if evidence_span is not None:
                start, end = evidence_span
                evidence = content[start:end]
            elif raw_evidence:
                evidence, start, end = self._sentence_for_two_spans(
                    content,
                    (int(source["mention_start"]), int(source["mention_end"])),
                    (int(target["mention_start"]), int(target["mention_end"])),
                )
            else:
                evidence, start, end = "", None, None
            confidence = coerce_confidence(raw.get("confidence"), 0.5)
            explicit_endpoints = bool(
                evidence
                and any(_contains_surface(evidence, name) for name in source_names if name)
                and any(_contains_surface(evidence, name) for name in target_names if name)
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
        sentence_starts = [left for left, _, _ in sentences]
        locations: dict[str, int] = {}
        for entity in entities:
            start = entity.get("evidence_start")
            if start is None:
                continue
            sentence_index = bisect_right(sentence_starts, int(start)) - 1
            if (
                sentence_index >= 0
                and sentence_index < len(sentences)
                and sentences[sentence_index][0] <= int(start) < sentences[sentence_index][1]
            ):
                locations[entity["id"]] = sentence_index
        sentence_buckets: dict[int, list[int]] = defaultdict(list)
        for entity_index, entity in enumerate(entities):
            sentence_index = locations.get(entity["id"])
            if sentence_index is not None:
                sentence_buckets[sentence_index].append(entity_index)
        definitions = logical.get("relation_types") or []
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for source in entities:
            source_sentence = locations.get(source["id"])
            if source_sentence is None:
                continue
            target_indexes = sorted(
                index
                for sentence_index in range(source_sentence - 1, source_sentence + 2)
                for index in sentence_buckets.get(sentence_index, ())
            )
            for target_index in target_indexes:
                increment("relation_pair_checks")
                target = entities[target_index]
                if source["id"] == target["id"]:
                    continue
                target_sentence = locations.get(target["id"])
                if target_sentence is None:
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
        increment("relation_candidates", len(candidates))
        gauge(
            "candidate_payload_bytes",
            len(json.dumps(candidates, ensure_ascii=False).encode("utf-8")),
        )
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
            relation["metadata"] = {
                **(relation.get("metadata") if isinstance(relation.get("metadata"), dict) else {}),
                "verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION,
            }
            status = relation.get("verification_status") or UNCERTAIN
            if status == LLM_VERIFIED:
                diagnostics["llm_verified"] += 1
            elif status == REJECTED:
                diagnostics["rejected"] += 1
            elif status == UNCERTAIN:
                diagnostics["uncertain"] += 1
            elif status == VERIFICATION_FAILED:
                diagnostics["verifier_failures"] += 1
            if is_publishable_relation(
                status,
                {"verification_policy_version": RELATION_VERIFICATION_POLICY_VERSION},
            ):
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
                context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                max_output_tokens=Config.relation_review_max_output_tokens,
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

        reviewed = self._normalize_relation_verifier_response(reviewed, batch)
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
    def _normalize_relation_verifier_response(
        reviewed: Any,
        batch: list[dict[str, Any]],
    ) -> Any:
        """Normalize bounded Ollama schema variants without weakening evidence checks.

        Some runners return a root array as ``items`` or key decisions by candidate
        ID even when given the strict JSON schema.  Only known field/orientation
        aliases are repaired here.  A supported verdict may reuse the candidate's
        already source-backed evidence, which still has to pass the exact-quote and
        endpoint checks in ``_apply_relation_decision``.
        """

        if not isinstance(reviewed, dict):
            return reviewed
        expected = {
            str(relation.get("candidate_id") or ""): relation for relation in batch
        }
        raw = reviewed.get("decisions")
        keyed = False
        if raw is not None and set(reviewed) != {"decisions"}:
            return reviewed
        if raw is None and isinstance(reviewed.get("items"), list):
            if set(reviewed) != {"items"}:
                return reviewed
            raw = reviewed["items"]
        elif raw is None and reviewed and set(reviewed).issubset(expected):
            if all(isinstance(value, dict) for value in reviewed.values()):
                raw = reviewed
                keyed = True
        if raw is None:
            return reviewed

        if keyed:
            values = [
                {"candidate_id": candidate_id, **dict(decision)}
                for candidate_id, decision in raw.items()
            ]
        elif isinstance(raw, list):
            values = raw
        else:
            return reviewed

        allowed_input_keys = {
            "candidate_id", "verdict", "decision", "corrected_type",
            "orientation", "evidence", "reason",
        }
        orientation_aliases = {
            "as_is": "as_is",
            "not_reversed": "as_is",
            "endpoints are not reversed": "as_is",
            "reversed": "reversed",
            "endpoints are reversed": "reversed",
        }
        normalized: list[dict[str, Any]] = []
        for value in values:
            if not isinstance(value, dict) or bool(set(value) - allowed_input_keys):
                return reviewed
            candidate_id = str(value.get("candidate_id") or "")
            relation = expected.get(candidate_id)
            verdict = str(value.get("verdict") or value.get("decision") or "")
            orientation_raw = str(value.get("orientation") or "")
            orientation_key = " ".join(
                orientation_raw.strip().lower().replace("-", "_").split()
            ).rstrip(".")
            orientation = orientation_aliases.get(orientation_key, orientation_raw)
            evidence = value.get("evidence")
            if evidence is None and relation is not None:
                evidence = (
                    str(relation.get("evidence_text") or "")
                    if verdict == "supported" else ""
                )
            reason = value.get("reason")
            if reason is None and verdict in RELATION_VERDICTS:
                reason = (
                    "verifier returned supported; reused source-backed candidate evidence"
                    if verdict == "supported"
                    else f"verifier returned {verdict} without an explicit reason"
                )
            item = {
                "candidate_id": candidate_id,
                "verdict": verdict,
                "orientation": orientation,
                "evidence": evidence,
                "reason": reason,
            }
            if "corrected_type" in value:
                item["corrected_type"] = value["corrected_type"]
            normalized.append(item)
        return {"decisions": normalized}

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
        if verdict == "contradicted":
            relation["verification_status"] = REJECTED
            relation["verification_reason"] = reason
            return ""
        if verdict == "insufficient_evidence":
            relation["verification_status"] = UNCERTAIN
            relation["verification_reason"] = reason
            return ""

        corrected_type = str(decision.get("corrected_type") or relation.get("type") or "").strip()
        if corrected_type not in relation_definitions:
            return "corrected relation type is not in the allowed schema"
        source_id, target_id = str(relation.get("source")), str(relation.get("target"))
        if orientation == "reversed":
            source_id, target_id = target_id, source_id
        source, target = entity_map.get(source_id), entity_map.get(target_id)
        definition = relation_definitions.get(corrected_type)
        if not source or not target or not definition:
            return "corrected endpoints or relation type are not in the allowed schema"
        if not self._compatible_relation(definition, source["type"], target["type"]):
            return "corrected relation type is incompatible with endpoint types"
        evidence, start, end = self._locate_evidence(
            content,
            str(decision.get("evidence") or ""),
            near=relation.get("evidence_start"),
        )
        if start is None or end is None:
            return "supported decision evidence is not an exact source quote"
        source_names = source.get("verified_surface_forms") or [source["name"]]
        target_names = target.get("verified_surface_forms") or [target["name"]]
        if not any(_contains_surface(evidence, str(name)) for name in source_names if name) or not any(
            _contains_surface(evidence, str(name)) for name in target_names if name
        ):
            return "supported evidence does not contain both corrected endpoints"
        relation.update({
            "source": source_id,
            "target": target_id,
            "type": corrected_type,
            "evidence_text": evidence,
            "evidence_start": start,
            "evidence_end": end,
            "verification_status": LLM_VERIFIED,
            "verification_reason": reason,
            "orientation": orientation,
            "description": f"{source['name']} {corrected_type} {target['name']}",
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
        content, body_start = _chunk_body_with_offset(chunk)
        model = project.get("llm_model") or Config.default_llm_model
        entity_schema = logical.get("entity_types") or []
        relation_schema = logical.get("relation_types") or []
        metadata_context = {
            "title": _chunk_title(chunk),
            "section": _chunk_section(chunk),
            "chunk_key": _chunk_key(chunk),
        }
        primary_calls = 1
        primary_fallback = False
        try:
            primary = self.ollama.json_chat(
                model=model,
                system=(
                    "Extract a concise knowledge graph using only facts explicitly stated in the source text. "
                    "Every entity and relation must include an exact evidence quote copied from the source. "
                    "Every entity must include exact half-open mention_start/mention_end offsets; every relation "
                    "must include exact half-open evidence_start/evidence_end offsets. "
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
                context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                max_output_tokens=Config.extraction_max_output_tokens,
            )
        except OllamaProtocolError as exc:
            if "truncated by output length" not in str(exc):
                raise
            primary_calls += 1
            primary_fallback = True
            entity_only = self.ollama.json_chat(
                model=model,
                system=(
                    "The previous combined graph response exceeded the output budget. Extract only "
                    "source-backed entity mentions now. Preserve distinct occurrences and return exact "
                    "half-open mention/evidence offsets. Do not output relations."
                ),
                prompt=(
                    f"Entity schema:\n{json.dumps(entity_schema, ensure_ascii=False)}\n\n"
                    f"Source text:\n{content}"
                ),
                schema=ENTITY_ONLY_EXTRACTION_FORMAT,
                context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                max_output_tokens=Config.extraction_max_output_tokens,
            )
            primary = {"entities": entity_only.get("entities") or [], "relations": []}
        extraction = self._sanitize_extraction(primary, content, logical)
        diagnostics = {
            **(extraction.pop("_diagnostics", {})),
            "llm_calls": primary_calls,
            "gleaning_used": False,
            "primary_truncation_fallback": primary_fallback,
        }
        extracted_entities = extraction.get("entities") or []
        extracted_relations = extraction.get("relations") or []
        relation_candidates: list[dict[str, Any]] = []
        if (
            Config.extraction_gleaning_enabled
            and len(extracted_entities) >= 2
            and not extracted_relations
            and bool(_RELATION_CUES.search(content))
        ):
            relation_candidates = self._relation_candidates(content, extracted_entities, logical)
        diagnostics["relation_candidates"] = len(relation_candidates)
        if self._needs_gleaning(chunk, extraction, relation_candidates):
            try:
                relation_only = bool(extracted_entities and relation_candidates)
                supplement_raw = self.ollama.json_chat(
                    model=model,
                    system=(
                        "Extract only explicit relations between the supplied entity ids. Return no entities. "
                        "Use only allowed relation types and candidate pairs. Each relation requires an exact "
                        "source quote and exact half-open evidence offsets. Do not infer from co-occurrence."
                        if relation_only else
                        "Perform one targeted second-pass knowledge graph extraction. Return only facts missed by the first pass. "
                        "Repeat endpoint entities needed by a new relation. Every item requires an exact source quote. "
                        "Do not add facts based only on co-occurrence."
                    ),
                    prompt=(
                        f"Relation schema:\n{json.dumps(relation_schema, ensure_ascii=False)}\n\n"
                        f"Known entities:\n{json.dumps([{'id': item['id'], 'name': item['name'], 'type': item['type'], 'mention_start': item.get('mention_start'), 'mention_end': item.get('mention_end')} for item in extracted_entities], ensure_ascii=False)}\n\n"
                        f"Candidate entity pairs:\n{json.dumps(relation_candidates, ensure_ascii=False)}\n\n"
                        f"Source text:\n{content}"
                        if relation_only else
                        f"Schema:\n{json.dumps(logical, ensure_ascii=False)}\n\n"
                        f"First-pass result:\n{json.dumps(extraction, ensure_ascii=False)}\n\nSource text:\n{content}"
                        f"\n\nLocally generated candidate entity pairs:\n"
                        f"{json.dumps(relation_candidates, ensure_ascii=False)}"
                    ),
                    schema=RELATION_ONLY_EXTRACTION_FORMAT if relation_only else EXTRACTION_FORMAT,
                    context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                    max_output_tokens=Config.extraction_max_output_tokens,
                )
                supplement_payload = (
                    {"entities": extracted_entities, "relations": supplement_raw.get("relations") or []}
                    if relation_only else supplement_raw
                )
                supplement = self._sanitize_extraction(supplement_payload, content, logical)
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
        for entity in extraction.get("entities") or []:
            for key in ("evidence_start", "evidence_end", "mention_start", "mention_end"):
                if entity.get(key) is not None:
                    entity[key] = int(entity[key]) + body_start
            for alias in entity.get("alias_spans") or []:
                for key in ("start", "end"):
                    if alias.get(key) is not None:
                        alias[key] = int(alias[key]) + body_start
        for relation in extraction.get("relations") or []:
            for record in (relation, relation.get("_original_candidate") or {}):
                for key in ("evidence_start", "evidence_end"):
                    if record.get(key) is not None:
                        record[key] = int(record[key]) + body_start
        return extraction

    def extract_entities_only(
        self,
        project: dict[str, Any],
        schema_record: dict[str, Any],
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the isolated P2 entity treatment without relation work or gleaning."""

        logical = schema_record["schema_json"]
        content, body_start = _chunk_body_with_offset(chunk)
        model = project.get("llm_model") or Config.default_llm_model
        system = (
            "Extract every explicit, source-backed entity mention: named people, organizations, "
            "locations, products, materials, technologies, processes, standards, events, and "
            "other reusable concrete concepts allowed by the entity schema. Preserve repeated "
            "occurrences. The name must be the exact source substring and mention_start/end must "
            "be its exact Unicode half-open offsets in Source text. Keep the JSON compact; omit "
            "optional aliases and evidence unless they add source-backed information. Do not "
            "extract or infer relations. Return an empty entities array only when the Source text "
            "truly contains no concrete entity mention."
        )
        entity_schema = json.dumps(logical.get("entity_types") or [], ensure_ascii=False)
        llm_calls = 0
        segmented_fallback = False
        segment_max_chars = Config.entity_only_segment_max_chars
        segment_max_items = Config.entity_only_max_items_per_segment

        def split_index(value: str) -> int:
            midpoint = len(value) // 2
            minimum = min(512, max(1, len(value) // 4))
            boundaries = [
                end for _start, end, _sentence in split_sentences(value)
                if minimum <= end <= len(value) - minimum
            ]
            if boundaries:
                return min(boundaries, key=lambda end: abs(end - midpoint))
            for distance in range(0, max(midpoint, len(value) - midpoint)):
                for candidate in (midpoint + distance, midpoint - distance):
                    if minimum <= candidate <= len(value) - minimum and value[candidate].isspace():
                        return candidate
            return midpoint

        def request_segment(
            segment: str,
            offset: int = 0,
            depth: int = 0,
        ) -> list[dict[str, Any]]:
            nonlocal llm_calls, segmented_fallback
            if len(segment) > segment_max_chars:
                boundary = split_index(segment)
                if 0 < boundary < len(segment):
                    segmented_fallback = True
                    return [
                        *request_segment(segment[:boundary], offset, depth + 1),
                        *request_segment(segment[boundary:], offset + boundary, depth + 1),
                    ]
            llm_calls += 1
            try:
                raw = self.ollama.json_chat(
                    model=model,
                    system=system,
                    prompt=f"Entity schema:\n{entity_schema}\n\nSource text:\n{segment}",
                    schema=ENTITY_ONLY_EXTRACTION_FORMAT,
                    context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                    max_output_tokens=Config.entity_only_max_output_tokens,
                )
            except OllamaProtocolError as exc:
                recoverable = (
                    "truncated by output length" in str(exc)
                    or "prompt token estimate underestimated" in str(exc)
                )
                if not recoverable:
                    raise
                boundary = split_index(segment)
                if (
                    depth < 8
                    and len(segment) > 64
                    and 0 < boundary < len(segment)
                ):
                    segmented_fallback = True
                    return [
                        *request_segment(segment[:boundary], offset, depth + 1),
                        *request_segment(segment[boundary:], offset + boundary, depth + 1),
                    ]
                else:
                    llm_calls += 1
                    raw = self.ollama.json_chat(
                        model=model,
                        system=system + " Return one compact JSON array of entity objects.",
                        prompt=f"Entity schema:\n{entity_schema}\n\nSource text:\n{segment}",
                        schema=None,
                        temperature=0,
                        context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                        max_output_tokens=min(Config.entity_only_max_output_tokens, 4096),
                    )
            raw_rows = self._flat_extraction(raw).get("entities") or []
            if (
                len(raw_rows) >= segment_max_items
                and len(segment) >= 256
                and depth < 6
            ):
                boundary = split_index(segment)
                if 0 < boundary < len(segment):
                    segmented_fallback = True
                    return [
                        *request_segment(segment[:boundary], offset, depth + 1),
                        *request_segment(segment[boundary:], offset + boundary, depth + 1),
                    ]
            rows: list[dict[str, Any]] = []
            for item in raw_rows:
                if not isinstance(item, dict):
                    continue
                normalized = dict(item)
                for start_key, end_key in (
                    ("mention_start", "mention_end"),
                    ("evidence_start", "evidence_end"),
                ):
                    if normalized.get(start_key) is not None:
                        normalized[start_key] = int(normalized[start_key]) + offset
                    if normalized.get(end_key) is not None:
                        normalized[end_key] = int(normalized[end_key]) + offset
                rows.append(normalized)
            return rows

        raw_entities = request_segment(content)
        entity_rows: list[dict[str, Any]] = []
        for index, item in enumerate(raw_entities, start=1):
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            normalized["id"] = f"entity_{index:04d}"
            if normalized.get("confidence") is None:
                normalized["confidence"] = 0.5
            entity_rows.append(normalized)
        extraction = self._sanitize_extraction(
            {"entities": entity_rows, "relations": []}, content, logical,
        )
        extraction["relations"] = []
        extraction["_verification_events"] = []
        extraction["_diagnostics"] = {
            **extraction.get("_diagnostics", {}),
            "llm_calls": llm_calls,
            "entity_only": True,
            "gleaning_used": False,
            "verification_llm_calls": 0,
            "segmented_fallback": segmented_fallback,
        }
        for entity in extraction.get("entities") or []:
            for key in ("evidence_start", "evidence_end", "mention_start", "mention_end"):
                if entity.get(key) is not None:
                    entity[key] = int(entity[key]) + body_start
            for alias in entity.get("alias_spans") or []:
                for key in ("start", "end"):
                    if alias.get(key) is not None:
                        alias[key] = int(alias[key]) + body_start
        return extraction
