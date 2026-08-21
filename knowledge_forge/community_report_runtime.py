"""Asynchronous, snapshot-bound reports for Knowledge Forge communities.

Community detection remains deterministic and graph publication never waits for
this module.  A report is an optional, source-grounded interpretation of one
already-published community.  Every input and citation is bound to immutable
canonical/relation keys so stale prose can never be attached to a newer graph.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import json
import math
import re
from typing import Any

from .config import Config
from .knowledge_forge_descriptions import relation_display_label
from .knowledge_language import (
    DOCUMENT_LANGUAGE,
    ENGLISH,
    generated_knowledge_instruction,
    normalize_knowledge_language,
)
from .ollama import ConservativeTokenEstimator


COMMUNITY_REPORT_PROMPT_VERSION = (
    "kf-community-report-grounded-v5-diverse-hierarchy-quality"
)
COMMUNITY_REPORT_SCHEMA_VERSION = "kf-community-report-schema-v3"
COMMUNITY_REPORT_CONTEXT_VERSION = "kf-community-report-context-v2-diverse-hierarchy"


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _text(value: object, *, limit: int = 0) -> str:
    result = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit and len(result) > limit:
        return result[:limit].rstrip("，,；;：:。.!！？? ") + "…"
    return result


def _relation_key(item: dict[str, Any]) -> str:
    key = _text(item.get("relation_key"))
    if key:
        return key
    return "relation:" + _digest({
        "source": item.get("source_canonical_key"),
        "target": item.get("target_canonical_key"),
        "type": item.get("relation_type"),
        "document_id": item.get("document_id"),
        "revision_id": item.get("revision_id"),
        "chunk_id": item.get("chunk_id"),
        "evidence": item.get("evidence_text"),
    })[:40]


@dataclass(frozen=True)
class CommunityReportPolicy:
    min_entities: int = 2
    min_relations: int = 1
    max_entities: int = 24
    max_relations: int = 36
    context_tokens: int = 8192
    max_output_tokens: int = 1800
    max_attempts: int = 2

    @classmethod
    def from_config(cls) -> "CommunityReportPolicy":
        return cls(
            min_entities=Config.knowledge_forge_community_report_min_entities,
            min_relations=Config.knowledge_forge_community_report_min_relations,
            max_entities=Config.knowledge_forge_community_report_max_entities,
            max_relations=Config.knowledge_forge_community_report_max_relations,
            context_tokens=Config.knowledge_forge_community_report_context_tokens,
            max_output_tokens=(
                Config.knowledge_forge_community_report_max_output_tokens
            ),
            max_attempts=Config.knowledge_forge_community_report_max_attempts,
        )

    def validate(self) -> None:
        if self.min_entities < 2 or self.min_relations < 1:
            raise ValueError("社区报告资格阈值无效")
        if self.max_entities < self.min_entities:
            raise ValueError("社区报告实体上下文小于资格阈值")
        if self.max_relations < self.min_relations:
            raise ValueError("社区报告关系上下文小于资格阈值")
        if min(self.context_tokens, self.max_output_tokens, self.max_attempts) < 1:
            raise ValueError("社区报告模型预算无效")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


REPORT_SYSTEM_PROMPT = """你是证据约束的知识图谱社区分析员。请只根据输入 JSON 中的实体和已验证关系撰写社区报告，不得补充常识、猜测或输入中没有的事实。报告语言应跟随输入内容的主要语言。

findings 是面向用户的“关键结论”。每项 finding 必须引用至少一条 relationships 中的关系编号，也可以同时引用实体编号，以便逐条回到原文核查。标题应概括引用关系直接支持的事实；explanation 应先用自然语言说明关系的主体、含义和客体，再在证据确实支持时给出有限解释。不要写“输入数据表明”“该关系揭示”等无信息量套话。不得把共现、相关、包含或一般关联扩写成因果、保证、目的、重要性、核心作用、完整流程或价值判断，除非 relation_label、description 或 evidence 明确表达了这些语义。多条关系无法共同支持更高层结论时，应拆开陈述，不要强行归纳。编号只用于引用，不要写入自然语言。

title 必须是能帮助用户识别主题的简洁语义标题，不得只写“社区 N”“社区报告”或编号。summary 应概括该社区直接由已验证关系支持的主题、主要组成和组织方式，不得比 findings 作出更强的推断。findings 应优先覆盖不同关系语义、来源或结构分支，避免多项结论反复改写同一条关系。

若输入提供 subcommunities，它们只是父社区的主题导航和组织线索，不是独立证据；任何采用其主题的结论仍必须引用 relationships 中可回到原文的关系编号。若输入只覆盖社区的一部分，必须在 limitations 中明确说明。输出对象必须严格包含 title、summary、findings、limitations；每个 finding 必须严格包含 title、explanation、entity_refs、relation_refs，不得用 finding、description 或 evidence_refs 等其他字段替代。只返回符合给定 JSON Schema 的对象。"""


def report_system_prompt(
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> str:
    return (
        REPORT_SYSTEM_PROMPT
        + "\n\n项目知识语言规则："
        + generated_knowledge_instruction(knowledge_language)
    )


REPORT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "summary", "findings", "limitations"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "summary": {"type": "string", "minLength": 1},
        "findings": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "title", "explanation", "entity_refs", "relation_refs",
                ],
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "explanation": {"type": "string", "minLength": 1},
                    "entity_refs": {
                        "type": "array", "items": {"type": "string"},
                        "uniqueItems": True,
                    },
                    "relation_refs": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "string"}, "uniqueItems": True,
                    },
                },
            },
        },
        "limitations": {"type": "string"},
    },
}


def _refresh_context_coverage(context: dict[str, Any]) -> None:
    """Recompute prompt coverage after any deterministic budget pruning."""

    relationships = context.get("relationships") or []
    entities = context.get("entities") or []
    known_relation_refs = {
        _text(item.get("ref")) for item in relationships
        if isinstance(item, dict) and _text(item.get("ref"))
    }
    subcommunities = context.setdefault("subcommunities", [])
    retained_subcommunities: list[dict[str, Any]] = []
    for item in subcommunities:
        if not isinstance(item, dict):
            continue
        support_refs = [
            _text(value) for value in item.get("support_relation_refs") or []
            if _text(value) in known_relation_refs
        ]
        # A subcommunity synopsis is navigation rather than evidence.  Without
        # at least one retained raw relationship it cannot safely influence the
        # generated parent report, so omit it from the model context.
        if not support_refs:
            continue
        retained = dict(item)
        retained["support_relation_refs"] = sorted(set(support_refs))
        retained_subcommunities.append(retained)
    subcommunities[:] = retained_subcommunities

    predicates = {
        _text(item.get("predicate")) for item in relationships
        if isinstance(item, dict) and _text(item.get("predicate"))
    }
    documents = {
        int(item.get("document_id") or 0) for item in relationships
        if isinstance(item, dict) and int(item.get("document_id") or 0) > 0
    }
    source_locations = {
        _source_location_key(item) for item in relationships
        if isinstance(item, dict) and _source_location_key(item)
    }
    coverage = context.setdefault("coverage", {})
    coverage["entities_in_prompt"] = len(entities)
    coverage["relationships_in_prompt"] = len(relationships)
    coverage["predicates_in_prompt"] = len(predicates)
    coverage["documents_in_prompt"] = len(documents)
    coverage["source_locations_in_prompt"] = len(source_locations)
    coverage["subcommunities_in_prompt"] = len(retained_subcommunities)
    coverage["complete"] = (
        len(entities) == int(coverage.get("entities_total") or 0)
        and len(relationships) == int(coverage.get("relationships_total") or 0)
        and len(retained_subcommunities)
        == int(coverage.get("subcommunities_total") or 0)
    )


def _fit_context_budget(
    context: dict[str, Any], policy: CommunityReportPolicy,
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> dict[str, Any]:
    """Shrink representative context before Ollama's fail-closed admission.

    The immutable content hash still covers every community fact.  Only the
    model prompt is bounded here, with the lowest-ranked context records
    removed first and explicit coverage counters retained for the report.
    """

    estimator = ConservativeTokenEstimator()
    input_budget = max(
        512,
        policy.context_tokens - policy.max_output_tokens
        - int(Config.ollama_default_safety_margin) - 900,
    )

    def estimate() -> int:
        return estimator.estimate(
            report_system_prompt(knowledge_language)
            + _json(REPORT_JSON_SCHEMA)
            + report_prompt(context, knowledge_language)
        )

    relationships = context["relationships"]
    entities = context["entities"]
    subcommunities = context.setdefault("subcommunities", [])

    def prune_unreferenced_entities() -> None:
        referenced = {
            str(value)
            for relation in relationships
            for value in (
                relation.get("source_ref"), relation.get("target_ref"),
            )
            if str(value or "")
        }
        entities[:] = [
            entity for entity in entities
            if str(entity.get("ref") or "") in referenced
        ]

    def compact_prose(*, entity_limit: int, relation_limit: int,
                      evidence_limit: int, alias_limit: int) -> None:
        for entity in entities:
            entity["description"] = _text(
                entity.get("description"), limit=entity_limit,
            )
            entity["aliases"] = list(entity.get("aliases") or [])[:alias_limit]
        for relation in relationships:
            relation["description"] = _text(
                relation.get("description"), limit=relation_limit,
            )
            relation["evidence"] = _text(
                relation.get("evidence"), limit=evidence_limit,
            )
        for subcommunity in subcommunities:
            subcommunity["summary"] = _text(
                subcommunity.get("summary"), limit=max(120, relation_limit * 2),
            )
            subcommunity["topic_entities"] = list(
                subcommunity.get("topic_entities") or []
            )[:alias_limit]

    # Findings must cite relations.  Standalone entity profiles therefore
    # have lower information value than topology and source evidence and are
    # removed before any relationship is sacrificed.
    prune_unreferenced_entities()
    _refresh_context_coverage(context)
    if estimate() > input_budget:
        compact_prose(
            entity_limit=160, relation_limit=120,
            evidence_limit=300, alias_limit=5,
        )

    # Child-community synopses improve macro organization but remain secondary
    # to auditable entity/relation facts.  Keep several where possible, then
    # release synopsis budget before sacrificing the relational core.
    while estimate() > input_budget and len(subcommunities) > 4:
        subcommunities.pop()
        _refresh_context_coverage(context)

    # Retain a useful relational core where possible.  Every time a low-ranked
    # edge is removed, discard endpoints no longer used by the remaining
    # topology instead of carrying dozens of unrelated entity descriptions.
    relation_floor = min(len(relationships), max(policy.min_relations, 4))
    while estimate() > input_budget and len(relationships) > relation_floor:
        relationships.pop()
        prune_unreferenced_entities()
        _refresh_context_coverage(context)
    if estimate() > input_budget:
        compact_prose(
            entity_limit=80, relation_limit=60,
            evidence_limit=160, alias_limit=3,
        )
    while estimate() > input_budget and len(relationships) > policy.min_relations:
        relationships.pop()
        prune_unreferenced_entities()
        _refresh_context_coverage(context)
    while estimate() > input_budget and subcommunities:
        subcommunities.pop()
        _refresh_context_coverage(context)
    if estimate() > input_budget:
        raise ValueError("社区报告上下文无法放入配置的模型预算")
    _refresh_context_coverage(context)
    return context


def _canonical_payload(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "key": _text(item.get("canonical_key")),
        "name": _text(item.get("name")),
        "type": _text(item.get("primary_type") or "GenericEntity"),
        "aliases": sorted({_text(value) for value in item.get("aliases") or [] if _text(value)}),
        "description": _text(metadata.get("description")),
        "source_count": max(0, int(item.get("source_count") or 0)),
    }


def _relation_payload(
    item: dict[str, Any],
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "key": _relation_key(item),
        "source": _text(item.get("source_canonical_key")),
        "target": _text(item.get("target_canonical_key")),
        "predicate": _text(item.get("relation_type") or "related_to"),
        "label": _text(relation_display_label(
            item.get("relation_type") or "related_to", metadata,
            knowledge_language,
        )),
        "description": _text(metadata.get("display_description")),
        "evidence": _text(item.get("evidence_text")),
        "document_id": int(item.get("document_id") or 0),
        "chunk_id": int(item.get("chunk_id") or 0),
    }


def _source_document_key(item: dict[str, Any]) -> str:
    document_id = int(item.get("document_id") or 0)
    return f"document:{document_id}" if document_id > 0 else ""


def _source_location_key(item: dict[str, Any]) -> str:
    document = _source_document_key(item)
    chunk_id = int(item.get("chunk_id") or 0)
    return f"{document}:chunk:{chunk_id}" if document and chunk_id > 0 else document


def _select_diverse_relations(
    relations: list[dict[str, Any]],
    *,
    degrees: Counter[str],
    max_relations: int,
    max_entities: int,
    relation_subcommunities: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Select a coherent but source/semantic-diverse auditable subgraph.

    This is a deterministic maximum-marginal-relevance style selector.  Pure
    degree ordering repeatedly selected the same hub, predicate and document;
    pure diversity produced disconnected anecdotes.  The combined objective
    keeps a structural core, rewards frontier expansion, and covers new
    predicates, documents and child-community branches without model calls.
    """

    if not relations:
        return []
    relation_subcommunities = relation_subcommunities or {}
    remaining = {
        _relation_key(item): item for item in relations
    }
    selected: list[dict[str, Any]] = []
    selected_entities: set[str] = set()
    selected_predicates: set[str] = set()
    selected_documents: set[str] = set()
    selected_locations: set[str] = set()
    selected_subcommunities: set[str] = set()
    degree_scale = max(1, max(degrees.values(), default=1))

    while remaining and len(selected) < max_relations:
        candidates: list[tuple[float, float, str, dict[str, Any]]] = []
        for key, item in remaining.items():
            source = _text(item.get("source_canonical_key"))
            target = _text(item.get("target_canonical_key"))
            endpoints = {source, target}
            additions = endpoints.difference(selected_entities)
            if len(selected_entities) + len(additions) > max_entities:
                continue
            source_degree = int(degrees[source])
            target_degree = int(degrees[target])
            structural = (
                1.35 * max(source_degree, target_degree) / degree_scale
                + 0.65 * (source_degree + target_degree) / (2 * degree_scale)
            )
            predicate = _text(item.get("relation_type") or "related_to")
            document = _source_document_key(item)
            location = _source_location_key(item)
            child_keys = relation_subcommunities.get(key, set())
            new_children = child_keys.difference(selected_subcommunities)

            if not selected:
                frontier = 0.0
            elif len(additions) == 1:
                frontier = 1.65
            elif not additions:
                frontier = 0.35
            else:
                frontier = -0.35
            peripheral = 0.45 * sum(
                1.0 / max(1, int(degrees[value])) for value in additions
            )
            score = (
                structural
                + frontier
                + 0.40 * len(additions)
                + peripheral
                + (2.50 if predicate not in selected_predicates else 0.0)
                + (1.80 if document and document not in selected_documents else 0.0)
                + (0.55 if location and location not in selected_locations else 0.0)
                + min(3.20, 1.60 * len(new_children))
                + (0.35 if _text(item.get("evidence_text")) else 0.0)
            )
            candidates.append((score, structural, key, item))
        if not candidates:
            break
        _, _, selected_key, chosen = min(
            candidates,
            key=lambda value: (-value[0], -value[1], value[2]),
        )
        selected.append(chosen)
        source = _text(chosen.get("source_canonical_key"))
        target = _text(chosen.get("target_canonical_key"))
        selected_entities.update((source, target))
        selected_predicates.add(
            _text(chosen.get("relation_type") or "related_to")
        )
        document = _source_document_key(chosen)
        if document:
            selected_documents.add(document)
        location = _source_location_key(chosen)
        if location:
            selected_locations.add(location)
        selected_subcommunities.update(
            relation_subcommunities.get(selected_key, set())
        )
        remaining.pop(selected_key)
    return selected


def _context_for_community(
    community: dict[str, Any],
    *,
    canonical_by_key: dict[str, dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    child_communities: list[dict[str, Any]] | None,
    policy: CommunityReportPolicy,
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> dict[str, Any]:
    member_keys = sorted({
        _text(value) for value in community.get("entity_keys") or []
        if _text(value) in canonical_by_key
    })
    member_set = set(member_keys)
    child_communities = [
        dict(item) for item in child_communities or []
        if isinstance(item, dict)
    ]
    internal = [
        item for item in relation_rows
        if _text(item.get("source_canonical_key")) in member_set
        and _text(item.get("target_canonical_key")) in member_set
        and _text(item.get("source_canonical_key"))
        != _text(item.get("target_canonical_key"))
    ]
    relation_by_key = {
        _relation_key(item): item for item in internal
    }
    internal = [relation_by_key[key] for key in sorted(relation_by_key)]
    degrees: Counter[str] = Counter()
    for relation in internal:
        degrees[_text(relation.get("source_canonical_key"))] += 1
        degrees[_text(relation.get("target_canonical_key"))] += 1

    child_members: dict[str, set[str]] = {}
    child_keys_by_entity: dict[str, set[str]] = defaultdict(set)
    for child in child_communities:
        child_key = _text(child.get("key"))
        child_set = {
            _text(value) for value in child.get("entity_keys") or []
            if _text(value) in member_set
        }
        if not child_key or len(child_set) < 2:
            continue
        child_members[child_key] = child_set
        for entity_key in child_set:
            child_keys_by_entity[entity_key].add(child_key)
    relation_subcommunities: dict[str, set[str]] = {}
    for relation in internal:
        source = _text(relation.get("source_canonical_key"))
        target = _text(relation.get("target_canonical_key"))
        memberships = child_keys_by_entity.get(source, set()).intersection(
            child_keys_by_entity.get(target, set())
        )
        if memberships:
            relation_subcommunities[_relation_key(relation)] = memberships

    selected_relations = _select_diverse_relations(
        internal,
        degrees=degrees,
        max_relations=policy.max_relations,
        max_entities=policy.max_entities,
        relation_subcommunities=relation_subcommunities,
    )
    selected_keys = {
        _text(value)
        for relation in selected_relations
        for value in (
            relation.get("source_canonical_key"),
            relation.get("target_canonical_key"),
        )
        if _text(value)
    }
    ranked_entities = sorted(
        member_keys,
        key=lambda key: (
            -degrees[key],
            -int(canonical_by_key[key].get("source_count") or 0),
            _text(canonical_by_key[key].get("name")).casefold(),
            key,
        ),
    )
    selected_entity_keys = [key for key in ranked_entities if key in selected_keys]

    entity_refs = {
        key: f"E{index:03d}" for index, key in enumerate(selected_entity_keys, 1)
    }
    entities = []
    for key in selected_entity_keys:
        item = _canonical_payload(canonical_by_key[key])
        entities.append({
            "ref": entity_refs[key],
            "key": key,
            "name": item["name"],
            "type": item["type"],
            "aliases": item["aliases"][:8],
            "description": _text(item["description"], limit=360),
            "degree_in_community": int(degrees[key]),
            "source_count": item["source_count"],
        })
    relations = []
    for index, relation in enumerate(selected_relations, 1):
        item = _relation_payload(relation, knowledge_language)
        relations.append({
            "ref": f"R{index:03d}",
            "key": item["key"],
            "source_ref": entity_refs[item["source"]],
            "source_name": _text(canonical_by_key[item["source"]].get("name")),
            "predicate": item["predicate"],
            "relation_label": item["label"],
            "target_ref": entity_refs[item["target"]],
            "target_name": _text(canonical_by_key[item["target"]].get("name")),
            "description": _text(item["description"], limit=280),
            "evidence": _text(item["evidence"], limit=520),
            "document_id": item["document_id"],
            "chunk_id": item["chunk_id"],
        })

    selected_relation_ref = {
        _text(item.get("key")): _text(item.get("ref")) for item in relations
    }
    ranked_children = sorted(
        (
            child for child in child_communities
            if _text(child.get("key")) in child_members
        ),
        key=lambda item: (
            -int(item.get("relationship_count") or 0),
            -len(child_members[_text(item.get("key"))]),
            _text(item.get("key")),
        ),
    )
    subcommunity_limit = min(16, max(4, policy.max_entities // 2))
    subcommunities: list[dict[str, Any]] = []
    for child in ranked_children:
        child_key = _text(child.get("key"))
        support_refs = sorted({
            selected_relation_ref[relation_key]
            for relation_key, memberships in relation_subcommunities.items()
            if child_key in memberships and relation_key in selected_relation_ref
        })
        if not support_refs:
            continue
        subcommunities.append({
            "key": child_key,
            "label": _text(child.get("label")),
            "summary": _text(
                child.get("summary") or child.get("description"), limit=520,
            ),
            "topic_entities": [
                _text(value) for value in child.get("topic_entities") or []
                if _text(value)
            ][:8],
            "entity_count": len(child_members[child_key]),
            "relationship_count": max(
                0, int(child.get("relationship_count") or 0),
            ),
            "support_relation_refs": support_refs,
        })
        if len(subcommunities) >= subcommunity_limit:
            break

    full_entities = [_canonical_payload(canonical_by_key[key]) for key in member_keys]
    full_relations = [
        _relation_payload(item, knowledge_language) for item in internal
    ]
    content_hash = _digest({
        "entities": full_entities,
        "relationships": full_relations,
    })
    context = {
        "community": {
            "key": _text(community.get("key")),
            "label": _text(community.get("label")),
            "level": max(0, int(community.get("level") or 0)),
            "entity_count": len(member_keys),
            "relationship_count": len(internal),
            "parent": _text(community.get("parent")),
        },
        "coverage": {
            "entities_in_prompt": len(entities),
            "entities_total": len(member_keys),
            "relationships_in_prompt": len(relations),
            "relationships_total": len(internal),
            "predicates_in_prompt": len({
                _text(item.get("relation_type") or "related_to")
                for item in selected_relations
            }),
            "predicates_total": len({
                _text(item.get("relation_type") or "related_to")
                for item in internal
            }),
            "documents_in_prompt": len({
                _source_document_key(item) for item in selected_relations
                if _source_document_key(item)
            }),
            "documents_total": len({
                _source_document_key(item) for item in internal
                if _source_document_key(item)
            }),
            "source_locations_in_prompt": len({
                _source_location_key(item) for item in selected_relations
                if _source_location_key(item)
            }),
            "source_locations_total": len({
                _source_location_key(item) for item in internal
                if _source_location_key(item)
            }),
            "subcommunities_in_prompt": len(subcommunities),
            "subcommunities_total": len({
                child_key
                for memberships in relation_subcommunities.values()
                for child_key in memberships
            }),
            "complete": (
                len(entities) == len(member_keys)
                and len(relations) == len(internal)
                and len(subcommunities) == len({
                    child_key
                    for memberships in relation_subcommunities.values()
                    for child_key in memberships
                })
            ),
        },
        "selection": {
            "version": COMMUNITY_REPORT_CONTEXT_VERSION,
            "method": "structural_diversity_mmr",
        },
        "entities": entities,
        "relationships": relations,
        "subcommunities": subcommunities,
    }
    context = _fit_context_budget(
        context, policy, knowledge_language,
    )
    eligible = (
        len(member_keys) >= policy.min_entities
        and len(internal) >= policy.min_relations
        and bool(relations)
    )
    # Root/macro communities and structurally rich communities are processed
    # first.  Parent reports now receive citation-closed child-community
    # navigation, so the bounded initial batch provides useful global coverage;
    # the value is only queue ordering and never changes graph facts.
    priority = (
        1_000_000
        + (500_000 if not _text(community.get("parent")) else 0)
        + max(0, int(community.get("level") or 0)) * 10_000
        + int(math.log2(len(internal) + 1) * 1_000)
        + min(len(internal), 999)
        + min(len(member_keys), 999)
    )
    return {
        "community_key": _text(community.get("key")),
        "level": max(0, int(community.get("level") or 0)),
        "status": "pending" if eligible else "skipped",
        "priority": priority,
        "community_content_hash": content_hash,
        "context_hash": _digest(context),
        "entity_keys": member_keys,
        "relation_keys": sorted(relation_by_key),
        "input_context": context,
        "skip_reason": "" if eligible else (
            "社区没有足够的实体和已验证内部关系"
        ),
    }


def build_community_report_plan(
    *,
    community_run: dict[str, Any],
    snapshot: dict[str, Any],
    policy: CommunityReportPolicy,
    initial_limit: int | None = None,
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> dict[str, Any]:
    """Build immutable report work without invoking a model or writing state."""

    policy.validate()
    knowledge_language = normalize_knowledge_language(knowledge_language)
    canonical_by_key = {
        _text(item.get("canonical_key")): dict(item)
        for item in snapshot.get("canonicals") or []
        if _text(item.get("canonical_key"))
    }
    relations = [
        dict(item) for item in snapshot.get("relation_endpoints") or []
        if _text(item.get("source_canonical_key")) in canonical_by_key
        and _text(item.get("target_canonical_key")) in canonical_by_key
    ]
    communities = [
        dict(item) for item in community_run.get("communities") or []
        if isinstance(item, dict) and _text(item.get("key"))
    ]
    children_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for community in communities:
        parent_key = _text(community.get("parent"))
        if parent_key:
            children_by_parent[parent_key].append(community)
    items = [
        _context_for_community(
            community,
            canonical_by_key=canonical_by_key,
            relation_rows=relations,
            child_communities=children_by_parent.get(
                _text(community.get("key")), []
            ),
            policy=policy,
            knowledge_language=knowledge_language,
        )
        for community in communities
    ]
    items.sort(key=lambda item: (-int(item["priority"]), item["community_key"]))
    fact_eligible_count = sum(item["status"] == "pending" for item in items)
    if initial_limit is not None:
        remaining = max(0, int(initial_limit))
        for item in items:
            if item["status"] != "pending":
                continue
            if remaining:
                remaining -= 1
                continue
            # ``skipped`` is the persisted terminal state supported by the v1
            # report table.  ``deferred`` and its error category distinguish a
            # fact-rich, on-demand item from a genuinely fact-poor community;
            # it can later be atomically moved back to pending.
            item["status"] = "skipped"
            item["deferred"] = True
            item["skip_reason"] = "详细报告将在打开该社区或检索命中时按需生成"
    eligible_count = sum(item["status"] == "pending" for item in items)
    deferred_count = sum(bool(item.get("deferred")) for item in items)
    return {
        "items": items,
        "expected_count": len(items),
        "fact_eligible_count": fact_eligible_count,
        "eligible_count": eligible_count,
        "skipped_count": sum(item["status"] == "skipped" for item in items),
        "deferred_count": deferred_count,
        "input_hash": _digest({
            "community_run_id": community_run.get("run_id"),
            "community_input_hash": community_run.get("input_hash"),
            "initial_limit": initial_limit,
            "knowledge_language": knowledge_language,
            "context_version": COMMUNITY_REPORT_CONTEXT_VERSION,
            "policy": policy.as_dict(),
            "items": [{
                "community_key": item["community_key"],
                "community_content_hash": item["community_content_hash"],
                "context_hash": item["context_hash"],
                "status": item["status"],
                "deferred": bool(item.get("deferred")),
            } for item in items],
        }),
    }


def report_prompt(
    input_context: dict[str, Any],
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> str:
    instruction = (
        "Generate a detailed community report from the context below. "
        "Citations may use only the refs provided in entities/relationships."
        if normalize_knowledge_language(knowledge_language) == ENGLISH
        else
        "请根据下面的社区上下文生成详细社区报告。"
        "引用只能使用 entities/relationships 中提供的 ref。"
    )
    return instruction + "\n\n" + json.dumps(
        input_context, ensure_ascii=False, indent=2, sort_keys=True,
    )


def community_report_request_hash(
    model_id: str,
    input_context: dict[str, Any],
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> str:
    return _digest({
        "model": str(model_id),
        "prompt_version": COMMUNITY_REPORT_PROMPT_VERSION,
        "schema_version": COMMUNITY_REPORT_SCHEMA_VERSION,
        "knowledge_language": normalize_knowledge_language(
            knowledge_language,
        ),
        "context": input_context,
    })


_REPORT_REFERENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([ER]\d{3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def _report_reference_tokens(*values: object) -> set[str]:
    """Read only syntactically valid model-facing E/R references.

    Models occasionally return the right references under a generic
    ``evidence_refs`` field or inline in prose despite receiving a strict JSON
    schema.  This parser deliberately does not resolve names, relation labels
    or fuzzy identifiers: every recovered token must still match an exact
    reference present in the immutable input context.
    """

    pending = list(values)
    references: set[str] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, (list, tuple, set)):
            pending.extend(value)
            continue
        if not isinstance(value, str):
            continue
        references.update(
            match.upper() for match in _REPORT_REFERENCE_PATTERN.findall(value)
        )
    return references


def clean_report_prose(
    value: object, input_context: dict[str, Any] | None = None,
) -> str:
    """Remove model-facing E/R references from user-facing report prose.

    The references remain represented by ``entity_keys`` and ``relation_keys``
    in the validated report.  Only exact synthetic references present in this
    report's immutable input are removed, so domain identifiers which merely
    resemble an E/R code are not changed.
    """

    result = _text(value)
    context = input_context if isinstance(input_context, dict) else {}
    known_refs = {
        _text(item.get("ref")).upper()
        for collection in (context.get("entities") or [], context.get("relationships") or [])
        for item in collection
        if isinstance(item, dict) and _text(item.get("ref"))
    }
    for reference in sorted(known_refs, key=len, reverse=True):
        escaped = re.escape(reference)
        result = re.sub(
            rf"[（(]\s*{escaped}\s*[）)]", "", result,
            flags=re.IGNORECASE,
        )
        result = re.sub(
            rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", "", result,
            flags=re.IGNORECASE,
        )
    result = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", result)
    result = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=（)", "", result)
    result = re.sub(r"\s+([，。；：！？、,.!?;:])", r"\1", result)
    return _text(result)


def _report_limitations(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "；".join(
            part for part in (_text(item) for item in value) if part
        )
    return _text(value)


def _fallback_report_title(
    input_context: dict[str, Any],
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> str:
    """Create a factual, deterministic title without asking the model again."""

    for relation in input_context.get("relationships") or []:
        if not isinstance(relation, dict):
            continue
        source = _text(relation.get("source_name"))
        target = _text(relation.get("target_name"))
        label = _text(
            relation.get("relation_label") or relation.get("predicate")
        )
        if source and target and label:
            separator = " — " if normalize_knowledge_language(
                knowledge_language,
            ) == ENGLISH else "："
            return _text(
                f"{source}{separator}{label}{separator}{target}", limit=140,
            )
    names: list[str] = []
    for item in input_context.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"))
        if name and name not in names:
            names.append(name)
        if len(names) >= 3:
            break
    if names:
        return "、".join(names)
    community = (
        input_context.get("community")
        if isinstance(input_context.get("community"), dict) else {}
    )
    return _text(community.get("label")) or (
        "Community report"
        if normalize_knowledge_language(knowledge_language) == ENGLISH
        else "社区报告"
    )


def _normalized_quality_text(value: object) -> str:
    return re.sub(
        r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff]+", "",
        _text(value).casefold(),
    )


def _is_generic_report_title(
    title: str, input_context: dict[str, Any],
) -> bool:
    normalized = _normalized_quality_text(title)
    community = (
        input_context.get("community")
        if isinstance(input_context.get("community"), dict) else {}
    )
    label = _normalized_quality_text(community.get("label"))
    generic = {
        "报告", "详细报告", "社区", "社区报告", "详细社区报告",
        "report", "community", "communityreport", "detailedreport",
        "detailedcommunityreport",
    }
    if label:
        generic.update({label, label + "报告", label + "report"})
    return not normalized or normalized in generic or bool(re.fullmatch(
        r"(?:社区|community)\d+(?:报告|report)?", normalized,
        flags=re.IGNORECASE,
    ))


def _near_duplicate_findings(
    left: dict[str, Any], right: dict[str, Any],
) -> bool:
    if set(left.get("relation_keys") or []) != set(
        right.get("relation_keys") or []
    ):
        return False
    title_similarity = SequenceMatcher(
        None,
        _normalized_quality_text(left.get("title")),
        _normalized_quality_text(right.get("title")),
    ).ratio()
    explanation_similarity = SequenceMatcher(
        None,
        _normalized_quality_text(left.get("explanation")),
        _normalized_quality_text(right.get("explanation")),
    ).ratio()
    return title_similarity >= 0.88 or explanation_similarity >= 0.92


def validate_report_payload(
    payload: dict[str, Any],
    input_context: dict[str, Any],
    knowledge_language: object = DOCUMENT_LANGUAGE,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("社区报告响应不是 JSON object")
    raw_title = clean_report_prose(payload.get("title"), input_context)
    requested_title = re.sub(
        r"^(?:社区|community)\s*[0-9A-Za-z_-]+\s*[：:\-—]\s*",
        "", raw_title, flags=re.IGNORECASE,
    ).strip()
    generic_title = _is_generic_report_title(requested_title, input_context)
    title_repaired = generic_title or requested_title != raw_title
    title = (
        _fallback_report_title(input_context, knowledge_language)
        if generic_title else requested_title
    )
    summary = clean_report_prose(payload.get("summary"), input_context)
    limitations = clean_report_prose(
        _report_limitations(payload.get("limitations")), input_context,
    )
    raw_findings = payload.get("findings")
    if not summary or not isinstance(raw_findings, list) or not raw_findings:
        raise ValueError("社区报告缺少 summary 或 findings")
    entity_by_ref = {
        _text(item.get("ref")).upper(): _text(item.get("key"))
        for item in input_context.get("entities") or []
        if isinstance(item, dict) and _text(item.get("ref")) and _text(item.get("key"))
    }
    relation_by_ref = {
        _text(item.get("ref")).upper(): _text(item.get("key"))
        for item in input_context.get("relationships") or []
        if isinstance(item, dict) and _text(item.get("ref")) and _text(item.get("key"))
    }
    findings: list[dict[str, Any]] = []
    duplicate_findings_removed = 0
    for index, raw in enumerate(raw_findings[:6]):
        if isinstance(raw, str):
            raw = {"explanation": raw}
        if not isinstance(raw, dict):
            raise ValueError(f"社区报告 finding {index + 1} 不是 object 或 string")
        explanation = _text(
            raw.get("explanation")
            or raw.get("description")
            or raw.get("finding")
            or raw.get("claim")
            or raw.get("detail")
        )
        finding_title = _text(
            raw.get("title") or raw.get("summary") or raw.get("heading")
        ) or _text(explanation, limit=64)
        references = _report_reference_tokens(
            raw.get("entity_refs"),
            raw.get("relation_refs"),
            raw.get("evidence_refs"),
            raw.get("references"),
            raw.get("refs"),
            raw.get("citations"),
            finding_title,
            explanation,
        )
        unknown = references.difference(entity_by_ref).difference(
            relation_by_ref
        )
        if unknown:
            raise ValueError("社区报告引用了输入中不存在的实体或关系编号")
        entity_refs = sorted(references.intersection(entity_by_ref))
        relation_refs = sorted(references.intersection(relation_by_ref))
        if not finding_title or not explanation:
            raise ValueError("社区报告 finding 缺少标题或说明")
        if not relation_refs:
            raise ValueError(
                "社区报告每项关键结论必须引用至少一条已验证关系"
            )
        entity_keys = sorted({entity_by_ref[value] for value in entity_refs})
        relation_keys = sorted({relation_by_ref[value] for value in relation_refs})
        finding = {
            "title": clean_report_prose(finding_title, input_context),
            "explanation": clean_report_prose(explanation, input_context),
            "entity_keys": entity_keys,
            "relation_keys": relation_keys,
        }
        if any(
            _near_duplicate_findings(finding, prior) for prior in findings
        ):
            duplicate_findings_removed += 1
            continue
        findings.append(finding)
    cited_entities = {
        key for item in findings for key in item.get("entity_keys") or []
    }
    cited_relations = {
        key for item in findings for key in item.get("relation_keys") or []
    }
    if not cited_relations:
        raise ValueError("社区报告至少需要引用一条可追溯原文的关系")
    relationship_context = [
        item for item in input_context.get("relationships") or []
        if isinstance(item, dict)
    ]
    relationship_by_key = {
        _text(item.get("key")): item for item in relationship_context
        if _text(item.get("key"))
    }
    available_predicates = {
        _text(item.get("predicate")) for item in relationship_context
        if _text(item.get("predicate"))
    }
    cited_predicates = {
        _text(relationship_by_key[key].get("predicate"))
        for key in cited_relations if key in relationship_by_key
        and _text(relationship_by_key[key].get("predicate"))
    }
    available_documents = {
        int(item.get("document_id") or 0) for item in relationship_context
        if int(item.get("document_id") or 0) > 0
    }
    cited_documents = {
        int(relationship_by_key[key].get("document_id") or 0)
        for key in cited_relations if key in relationship_by_key
        and int(relationship_by_key[key].get("document_id") or 0) > 0
    }
    # The schema gate already guarantees per-finding provenance.  These two
    # modest breadth checks reject reports that ignore nearly all of a rich,
    # deliberately diversified prompt, while allowing genuinely small or
    # single-predicate communities to publish normally.
    if len(relationship_context) >= 6 and len(cited_relations) < 2:
        raise ValueError("社区报告对代表性关系的覆盖过窄")
    if len(available_predicates) >= 3 and len(cited_predicates) < 2:
        raise ValueError("社区报告未覆盖输入中的多种关系语义")
    if (
        duplicate_findings_removed
        and len(findings) == 1
        and len(relationship_context) >= 4
    ):
        raise ValueError("社区报告关键结论重复且未覆盖其他代表性事实")
    coverage = input_context.get("coverage") if isinstance(input_context.get("coverage"), dict) else {}
    if not bool(coverage.get("complete")) and not limitations:
        limitations = (
                "This report uses representative entities and relationships "
                "selected for structural importance and semantic/source diversity; "
                "it does not cover every fact in the community."
            if normalize_knowledge_language(knowledge_language) == ENGLISH
            else (
                "本报告基于该社区中兼顾结构重要性、关系语义与来源多样性的"
                "代表性实体和关系，"
                "未覆盖社区内全部事实。"
            )
        )
    content_parts = [title, summary]
    content_parts.extend(
        f"{item['title']}：{item['explanation']}" for item in findings
    )
    if limitations:
        content_parts.append(
            ("Limitations: " if normalize_knowledge_language(
                knowledge_language,
            ) == ENGLISH else "限制：") + limitations
        )
    content = "\n".join(content_parts)
    subcommunities = [
        item for item in input_context.get("subcommunities") or []
        if isinstance(item, dict)
    ]
    cited_refs = {
        _text(item.get("ref")) for item in relationship_context
        if _text(item.get("key")) in cited_relations
    }
    covered_subcommunities = sum(
        bool(cited_refs.intersection({
            _text(value) for value in item.get("support_relation_refs") or []
            if _text(value)
        }))
        for item in subcommunities
    )
    quality = {
        "passed": True,
        "title_repaired": title_repaired,
        "duplicate_findings_removed": duplicate_findings_removed,
        "finding_count": len(findings),
        "available_relation_count": len(relationship_context),
        "cited_relation_count": len(cited_relations),
        "relation_citation_coverage": round(
            len(cited_relations) / max(1, len(relationship_context)), 4,
        ),
        "available_predicate_count": len(available_predicates),
        "cited_predicate_count": len(cited_predicates),
        "available_document_count": len(available_documents),
        "cited_document_count": len(cited_documents),
        "subcommunity_count": len(subcommunities),
        "covered_subcommunity_count": covered_subcommunities,
    }
    return {
        "title": title,
        "summary": summary,
        "findings": findings,
        "limitations": limitations,
        "content": content,
        "cited_entity_keys": sorted(cited_entities),
        "cited_relation_keys": sorted(cited_relations),
        "quality": quality,
        "output_hash": _digest({
            "title": title,
            "summary": summary,
            "findings": findings,
            "limitations": limitations,
        }),
    }


class KnowledgeForgeCommunityReportGenerator:
    """Generate and validate one bounded community report with Ollama."""

    def __init__(self, ollama, policy: CommunityReportPolicy | None = None):
        self.ollama = ollama
        self.policy = policy or CommunityReportPolicy.from_config()

    @property
    def prompt_hash(self) -> str:
        return self.prompt_hash_for(DOCUMENT_LANGUAGE)

    def prompt_hash_for(
        self, knowledge_language: object = DOCUMENT_LANGUAGE,
    ) -> str:
        return _digest({
            "version": COMMUNITY_REPORT_PROMPT_VERSION,
            "knowledge_language": normalize_knowledge_language(
                knowledge_language,
            ),
            "system": report_system_prompt(knowledge_language),
            "schema": REPORT_JSON_SCHEMA,
        })

    @property
    def config_hash(self) -> str:
        return _digest(self.policy.as_dict())

    def generate(
        self,
        *,
        model_id: str,
        input_context: dict[str, Any],
        knowledge_language: object = DOCUMENT_LANGUAGE,
    ) -> dict[str, Any]:
        knowledge_language = normalize_knowledge_language(knowledge_language)
        prompt = report_prompt(input_context, knowledge_language)
        payload = self.ollama.json_chat(
            model=model_id,
            system=report_system_prompt(knowledge_language),
            prompt=prompt,
            schema=REPORT_JSON_SCHEMA,
            temperature=0.1,
            context_budget_tokens=self.policy.context_tokens,
            max_output_tokens=self.policy.max_output_tokens,
        )
        normalized = validate_report_payload(
            payload, input_context, knowledge_language,
        )
        diagnostics = self.ollama.last_call_diagnostics
        diagnostics_payload = asdict(diagnostics) if diagnostics else {}
        diagnostics_payload["report_quality"] = normalized.get("quality") or {}
        usage = ({
            "prompt_eval_count": diagnostics.prompt_eval_count,
            "output_eval_count": diagnostics.output_eval_count,
            "estimated_input_tokens": diagnostics.estimated_input_tokens,
        } if diagnostics else {})
        usage["report_quality"] = normalized.get("quality") or {}
        return {
            **normalized,
            "request_hash": community_report_request_hash(
                model_id, input_context, knowledge_language,
            ),
            "raw_response_text": str(self.ollama.last_raw_response_text or ""),
            "parsed_response": payload,
            "diagnostics": diagnostics_payload,
            "usage": usage,
        }
