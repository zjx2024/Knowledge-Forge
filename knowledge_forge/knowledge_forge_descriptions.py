"""Grounded project-language descriptions for graph presentation.

Descriptions are derived only after entity assignment and verified fact
materialization.  They never participate in extraction, normalization,
verification, direction selection, or publication decisions.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import re
from typing import Any, Iterable, Mapping

from .knowledge_language import (
    CHINESE,
    ENGLISH,
    normalize_knowledge_language,
    resolve_knowledge_language,
)


DESCRIPTION_VERSION = "knowledge-forge-grounded-description-v2-language-policy"

ENTITY_TYPE_LABELS = {
    "GenericEntity": "通用实体",
    "Person": "人物",
    "Organization": "组织机构",
    "Location": "地点",
    "Product": "产品或材料",
    "Technology": "技术或方法",
    "Event": "事件",
}

ENTITY_TYPE_LABELS_EN = {
    "GenericEntity": "generic entity",
    "Person": "person",
    "Organization": "organization",
    "Location": "location",
    "Product": "product or material",
    "Technology": "technology or method",
    "Event": "event",
}

RELATION_TYPE_LABELS = {
    "related_to": "相关",
    "source_assertion": "原文直接关系",
    "part_of": "组成部分",
    "belongs_to": "隶属或归属",
    "located_in": "位于",
    "uses": "使用",
    "produces": "生产或创建",
    "depends_on": "依赖",
    "causes": "导致",
    "executes_task": "执行任务",
    "supply_to": "供应给",
    "supports": "支持",
    "manages": "管理",
    "transports": "运输",
    "member_of": "成员属于",
    "represents": "代表",
}

RELATION_TYPE_LABELS_EN = {
    "related_to": "related to",
    "source_assertion": "source-stated relation",
    "part_of": "part of",
    "belongs_to": "belongs to",
    "located_in": "located in",
    "uses": "uses",
    "produces": "produces or creates",
    "depends_on": "depends on",
    "causes": "causes",
    "executes_task": "executes task",
    "supply_to": "supplies to",
    "supports": "supports",
    "manages": "manages",
    "transports": "transports",
    "member_of": "member of",
    "represents": "represents",
}


def entity_type_label(
    value: Any, knowledge_language: object = CHINESE,
) -> str:
    key = str(value or "").strip()
    if normalize_knowledge_language(knowledge_language) == ENGLISH:
        return ENTITY_TYPE_LABELS_EN.get(key, key or "generic entity")
    return ENTITY_TYPE_LABELS.get(key, key or "通用实体")


def relation_type_label(
    value: Any, knowledge_language: object = CHINESE,
) -> str:
    key = str(value or "").strip()
    language = normalize_knowledge_language(knowledge_language)
    if language == ENGLISH and key in RELATION_TYPE_LABELS_EN:
        return RELATION_TYPE_LABELS_EN[key]
    if key in RELATION_TYPE_LABELS:
        return RELATION_TYPE_LABELS[key]
    if not key:
        return "related to" if language == ENGLISH else "相关"
    # The logical Schema is extensible, so a finite translation table cannot
    # cover every valid predicate.  A generic readable fallback is preferable
    # to the old catch-all ``关系``: it preserves the predicate distinction
    # without exposing the storage identifier verbatim.
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", key):
        return key
    readable = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", key)
    readable = re.sub(r"[_\-]+", " ", readable)
    readable = re.sub(r"\s+", " ", readable).strip()
    return readable[:1].upper() + readable[1:] if readable else "相关"


def is_open_relation_carrier(metadata: Mapping[str, Any] | None) -> bool:
    """Whether metadata describes the generic exact-source fallback.

    The fallback is identified by its versioned policy/state, never by a
    concrete predicate name. This keeps logical Schema IDs stable while every
    presentation surface avoids exposing their implementation detail.
    """

    values = dict(metadata or {})
    policy = values.get("open_relation_policy")
    return bool(
        values.get("open_relation_carrier")
        or (
            isinstance(policy, Mapping)
            and bool(policy.get("enabled"))
        )
    )


def relation_display_label(
    relation_type: Any,
    metadata: Mapping[str, Any] | None = None,
    knowledge_language: object = CHINESE,
) -> str:
    """Return a readable relation label without leaking storage vocabulary."""

    values = dict(metadata or {})
    explicit_display = _clean(values.get("display_relation_label"))
    if explicit_display:
        return explicit_display
    if is_open_relation_carrier(values):
        return _clean(values.get("source_relation_expression")) or (
            "source-stated relation"
            if normalize_knowledge_language(knowledge_language) == ENGLISH
            else "原文直接关系"
        )
    return relation_type_label(relation_type, knowledge_language)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique(values: Iterable[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def _short(value: Any, limit: int = 260) -> str:
    text = _clean(value)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _sentence(value: Any, knowledge_language: object = CHINESE) -> str:
    text = _clean(value)
    if not text:
        return ""
    if text[-1] in "。！？.!?…":
        return text
    return text + (
        "."
        if normalize_knowledge_language(knowledge_language) == ENGLISH
        else "。"
    )


def enrich_knowledge_forge_snapshot(
    snapshot: dict[str, Any],
    *,
    mentions: list[dict[str, Any]],
    knowledge_language: object = CHINESE,
) -> dict[str, Any]:
    """Attach deterministic, evidence-grounded display descriptions.

    The returned structure is a presentation/build adapter copy.  The immutable
    entity and relation ledgers are not modified.
    """

    language_texts = [
        str(item.get("evidence_text") or item.get("surface") or "")
        for item in mentions
    ] + [
        str(item.get("evidence_text") or "")
        for item in snapshot.get("relation_endpoints") or []
        if isinstance(item, Mapping)
    ]
    language = resolve_knowledge_language(
        knowledge_language, language_texts,
    )
    english = language == ENGLISH
    result = deepcopy(snapshot)
    canonicals = list(result.get("canonicals") or [])
    assignments = list(result.get("assignments") or [])
    relations = list(result.get("relation_endpoints") or [])
    canonical_by_key = {
        str(item.get("canonical_key") or ""): item
        for item in canonicals if str(item.get("canonical_key") or "")
    }
    canonical_for_mention = {
        str(item.get("mention_key") or ""): str(item.get("canonical_key") or "")
        for item in assignments
        if str(item.get("mention_key") or "")
        and str(item.get("canonical_key") or "")
    }
    evidence_by_canonical: dict[str, list[str]] = defaultdict(list)
    for mention in sorted(
        mentions,
        key=lambda item: (
            int(item.get("document_id") or 0),
            int(item.get("chunk_id") or 0),
            int(item.get("evidence_start") or 0),
            str(item.get("mention_key") or ""),
        ),
    ):
        key = canonical_for_mention.get(str(mention.get("mention_key") or ""))
        if key:
            evidence_by_canonical[key].append(
                _short(mention.get("evidence_text"), 220)
            )

    incident_by_canonical: dict[str, list[str]] = defaultdict(list)
    enriched_relations: list[dict[str, Any]] = []
    for relation in relations:
        item = deepcopy(relation)
        source_key = str(item.get("source_canonical_key") or "")
        target_key = str(item.get("target_canonical_key") or "")
        source_name = str(
            canonical_by_key.get(source_key, {}).get("name") or source_key
        )
        target_name = str(
            canonical_by_key.get(target_key, {}).get("name") or target_key
        )
        metadata = (
            dict(item.get("metadata") or {})
            if isinstance(item.get("metadata"), dict) else {}
        )
        label = relation_display_label(
            item.get("relation_type"), metadata, language,
        )
        evidence = _short(item.get("evidence_text"), 280)
        parts = ([
            f'The verified relation from “{source_name}” to “{target_name}” is “{label}”.'
        ] if english else [
            f"“{source_name}”与“{target_name}”之间的已验证关系为“{label}”。"
        ])
        if evidence:
            parts.append(
                (" Source evidence: " if english else "原文依据：")
                + _sentence(evidence, language)
            )
        existing_description = _clean(metadata.get("description"))
        if existing_description and existing_description not in "".join(parts):
            parts.append(
                (" Existing description: " if english else "已有说明：")
                + _sentence(existing_description, language)
            )
        description = "".join(parts)
        metadata.update({
            "display_description": description,
            "display_relation_label": label,
            "display_relation_id": str(item.get("relation_type") or ""),
            "description_version": DESCRIPTION_VERSION,
            "description_language": language,
            "description_source": "verified_fact_and_exact_evidence",
        })
        item["metadata"] = metadata
        enriched_relations.append(item)
        if english:
            incident_by_canonical[source_key].append(
                f'has a “{label}” relation to “{target_name}”'
            )
            incident_by_canonical[target_key].append(
                f'has a “{label}” relation with “{source_name}”'
            )
        else:
            incident_by_canonical[source_key].append(
                f"与“{target_name}”存在“{label}”关系"
            )
            incident_by_canonical[target_key].append(
                f"与“{source_name}”存在“{label}”关系"
            )

    enriched_canonicals: list[dict[str, Any]] = []
    for canonical in canonicals:
        item = deepcopy(canonical)
        key = str(item.get("canonical_key") or "")
        name = str(item.get("name") or key)
        label = entity_type_label(item.get("primary_type"), language)
        evidence = _unique(
            (
                value for value in evidence_by_canonical.get(key, [])
                if _clean(value).strip("“”\"'。！？.!? ")
                != _clean(name).strip("“”\"'。！？.!? ")
            ),
            limit=2,
        )
        relations_for_entity = _unique(
            incident_by_canonical.get(key, []), limit=4,
        )
        parts = (
            [f'“{name}” is identified in the source documents as {label}.']
            if english else [f"“{name}”是文档中识别的{label}。"]
        )
        if evidence:
            if english:
                parts.append(
                    " Source information: " + "; ".join(
                        _sentence(value, language).rstrip(".")
                        for value in evidence
                    ) + "."
                )
            else:
                parts.append(
                    "原文信息：" + "；".join(
                        _sentence(value, language).rstrip("。")
                        for value in evidence
                    ) + "。"
                )
        if relations_for_entity:
            parts.append(
                (" Verified connections: " if english else "已验证联系：")
                + ("; " if english else "；").join(relations_for_entity)
                + ("." if english else "。")
            )
        metadata = (
            dict(item.get("metadata") or {})
            if isinstance(item.get("metadata"), dict) else {}
        )
        metadata.update({
            "description": "".join(parts),
            "display_type_label": label,
            "description_version": DESCRIPTION_VERSION,
            "description_language": language,
            "description_source": "mentions_and_verified_facts",
        })
        item["metadata"] = metadata
        enriched_canonicals.append(item)

    result["canonicals"] = enriched_canonicals
    result["relation_endpoints"] = enriched_relations
    result["description_version"] = DESCRIPTION_VERSION
    result["description_language"] = language
    return result
