"""Versioned logical schema evaluation and structured graph extraction."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Mapping

from .config import Config
from .database import Database
from .knowledge_language import (
    CHINESE,
    ENGLISH,
    normalize_knowledge_language,
    resolve_knowledge_language,
    user_facing_text_matches_language,
)
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
from .sentence_spans import split_relation_scopes, split_sentences
from .span_entities import (
    entity_boundary_reason,
    non_referential_surface_reason,
    normalize_entity_span,
    repair_soft_wrapped_entity_span,
)
from .value_parsing import coerce_confidence


LOGGER = logging.getLogger(__name__)


SCHEMA_DISCOVERY_PROTOCOL_VERSION = (
    "knowledge-forge-schema-discovery-v16-project-language-policy"
)


def _versioned_relation_policy(
    name: str,
    source_types: list[str],
    target_types: list[str],
    semantic_definition: str,
    direction_description: str,
    *,
    subject_role_description: str,
    object_role_description: str,
    publish_mode: str = "auto",
    positive_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
    optional_soft_cues: list[str] | None = None,
    structural_evidence_kinds: list[str] | None = None,
    structural_role_labels: list[str] | None = None,
    source_role_pairs: list[str] | None = None,
    conflicts_with: list[str] | None = None,
    negation_policy: str = "reject_negated",
    modality_policy: str = "require_asserted",
    open_relation_policy: Mapping[str, Any] | None = None,
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
        "subject_role_description": subject_role_description,
        "object_role_description": object_role_description,
        "symmetric": False,
        "inverse_predicate": None,
        "canonical_orientation": "forward",
        "publish_mode": publish_mode,
        "reflexive": False,
        "positive_examples": list(positive_examples or []),
        "negative_examples": list(negative_examples or []),
        "optional_soft_cues": list(optional_soft_cues or []),
        # Structural evidence is an explicit, versioned policy capability.
        # Runtime code may consume these generic kinds, but never derives
        # behavior from a concrete predicate identifier.
        "structural_evidence_kinds": list(structural_evidence_kinds or []),
        "structural_role_labels": list(structural_role_labels or []),
        "source_role_pairs": list(source_role_pairs or []),
        # These predicates are competing interpretations only when they are
        # projected from the same immutable source frame and ordered endpoint
        # pair.  Distinct source assertions remain multi-label.
        "conflicts_with": list(conflicts_with or []),
        "negation_policy": str(negation_policy),
        "modality_policy": str(modality_policy),
        "open_relation_policy": dict(open_relation_policy or {}),
    }


BASE_SCHEMA: dict[str, Any] = {
    "version": "general-schema-v13-grounded-open-carrier-v1",
    "name": "Knowledge Forge General Schema",
    "description": "面向跨领域文档的保守型通用知识结构。",
    "entity_types": [
        {"name": "GenericEntity", "description": "无法归入更具体类别的命名对象或概念"},
        {"name": "Person", "description": "自然人、人物身份或角色"},
        {"name": "Organization", "description": "组织、机构、企业或团队"},
        {"name": "Location", "description": "物理地点或行政区域"},
        {"name": "Product", "description": "产品、设备、组件或材料"},
        {"name": "Technology", "description": "技术、方法、流程或标准"},
        {"name": "Event", "description": "完整的事件、活动、状态变化或事故"},
    ],
    "relation_types": [
        _versioned_relation_policy(
            "related_to", ["*"], ["*"], "主体与客体之间存在原文明确表达的通用语义联系。",
            "方向仅用于保存原文中的端点顺序，不暗示可交换的反向角色。",
            subject_role_description="原文明确联系中的一个参与方。",
            object_role_description="原文明确联系中的另一个参与方。",
            publish_mode="review_only",
            positive_examples=["原文明确说明实体甲与实体乙存在某种语义联系。"],
            negative_examples=["实体甲与实体乙只是在同一段落中出现。"],
            optional_soft_cues=["related", "关联"],
        ),
        _versioned_relation_policy(
            "source_assertion", ["*"], ["*"],
            (
                "主体通过原文中明确表达的实际谓词，"
                "直接且有方向地作用于客体。该开放谓词本身就是本关系的具体语义；"
                "仅共现、并列同级、普通相邻或共享第三实体不属于此关系。"
            ),
            (
                "严格保持开放事实清单中的有向论元顺序：从该原文谓词的完整语义主体"
                "指向其完整语义客体。"
            ),
            subject_role_description=(
                "原文实际谓词在精确证据中直接支配的完整语义主体。"
            ),
            object_role_description=(
                "同一开放谓词在精确证据中直接支配的完整语义客体。"
            ),
            positive_examples=[
                "开放事实清单以精确端点和连续原文证据记录一个明确的有向谓词。",
            ],
            negative_examples=[
                "两个实体只因共现、同列、同级并列或共同参与第三个事实而被连接。",
                "开放谓词为空、只是连接词、被否定，或端点方向与原文角色相反。",
            ],
            open_relation_policy={
                "enabled": True,
                "fallback_only": True,
                "display_source_expression": True,
                "identity_source_expression": True,
                "require_exact_source_expression": True,
                "allowed_candidate_sources": [
                    "GROUNDED_OPEN_FACT_SCHEMA_PROJECTION",
                    "ATOMIC_FRAME_SCHEMA_PROJECTION",
                ],
            },
        ),
        _versioned_relation_policy(
            "part_of", ["*"], ["*"], "主体是客体的组成部分、子集或结构成员。",
            "从局部或组件指向包含它的整体。",
            subject_role_description="被整体包含的组件、子集或组成成员。",
            object_role_description="包含该组件或由多个并列成员构成的整体。",
            positive_examples=["系统整体列出多个并列组件时，每个组件分别属于该整体。"],
            negative_examples=[
                "两个对象发生交互，但彼此不存在组成或包含关系。",
                "某组织的代表加入委员会，不表示该组织本身是委员会的组成部分。",
                "组件甲和组件乙同属整体丙，不表示组件甲是组件乙的一部分，也不表示组件乙是组件甲的一部分。",
                "某网络、流程或系统连接多个领域或对象，不表示这些领域或对象是该网络、流程或系统的组成部分。",
            ],
            optional_soft_cues=["part of", "组成"],
            structural_evidence_kinds=[
                "list_parent_child",
                "section_label_parent_child",
            ],
        ),
        _versioned_relation_policy(
            "scopes_topic", ["*"], ["*"],
            "行首编号或标签明确标注并管辖其所在条款的主要主题或行为主体。",
            "从行首编号或标签指向其直接管辖条款的主要主题或行为主体。",
            subject_role_description=(
                "直接位于条款行首并管辖该条款内容的编号、引用标识或标签。"
            ),
            object_role_description=(
                "该编号或标签所管辖条款中原文明示的主要主题或行为主体。"
            ),
            positive_examples=[
                "一个独立编号直接前缀于条款正文时，该编号标注该条款的主要主题。"
            ],
            negative_examples=[
                "编号与实体只在同一段落中出现，但编号并未直接前缀并管辖该条款。",
                "编号标注某条款，不表示编号与条款主题是同一实体，也不表示主题创建了编号。",
            ],
            optional_soft_cues=["leading identifier", "条款编号"],
            structural_evidence_kinds=["line_identifier_binding"],
        ),
        _versioned_relation_policy(
            "belongs_to", ["*"], ["*"], "主体是客体的成员、所属项或被拥有对象。",
            "从成员或所属项指向其群体、所有者或归属对象。",
            subject_role_description="成员、被拥有项目或具有明确隶属关系的实体。",
            object_role_description="该主体所属的群体、所有者、组织或上位对象。",
            positive_examples=["部门甲明确隶属于组织乙。"],
            negative_examples=[
                "组织甲向组织乙供货，不表示组织甲属于组织乙。",
                "组织甲的代表属于委员会乙，不表示组织甲本身属于委员会乙。",
            ],
            optional_soft_cues=["belongs to", "属于"],
        ),
        _versioned_relation_policy(
            "located_in", ["*"], ["Location"], "主体在物理位置或行政归属上位于客体之内。",
            "从被定位的实体指向容纳它的物理地点或行政区域。",
            subject_role_description="原文明确陈述其所在位置的实体。",
            object_role_description="容纳主体的物理地点或行政区域。",
            positive_examples=["机构甲明确位于城市乙。"],
            negative_examples=[
                "产品甲只是在设施乙中被使用，原文没有陈述产品甲归属于该地点。",
                "产品甲被运输或交付到目的地乙；临时目的地不等于稳定的位置或行政归属。",
            ],
            optional_soft_cues=["located in", "位于"],
        ),
        _versioned_relation_policy(
            "uses", ["*"], ["*"], "主体把客体作为工具、材料、方法、技术或运行资源主动使用或消耗。",
            "从主动使用者或运行系统指向被其使用的工具、材料、方法、技术或资源。",
            subject_role_description=(
                "主动运用、操作或消耗其他对象来完成自身任务的人员、组织、设备或系统；"
                "不是仅执行检查、观察、识别、运输等其他动作的参与方。"
            ),
            object_role_description=(
                "被主体当作手段或运行资源实际运用或消耗的工具、材料、方法、技术或资源；"
                "不能只是被检查、观察、识别、监控或处理的对象，也不能只是地点、接收方、"
                "动作结果、记录的所有者或上下文中的相关系统。"
            ),
            positive_examples=["组织甲明确采用技术乙完成任务。"],
            negative_examples=[
                "产品甲被组织乙使用；产品甲是被使用对象而不是使用者。",
                "设备甲被部署到环境乙，不表示设备甲使用环境乙。",
                "设备甲识别缺陷乙；缺陷乙是被识别对象而不是设备使用的工具或资源。",
                "团队甲检查系统乙的日志；被检查的系统乙不是团队使用的工具。",
            ],
            optional_soft_cues=["uses", "使用"],
        ),
        _versioned_relation_policy(
            "produces", ["*"], ["*"], "主体制造、生成或输出客体。",
            "无论原文使用主动句还是被动句，都从生产者指向产出物。",
            subject_role_description="实际制造、生成或输出结果的人员、组织、过程或系统。",
            object_role_description="被制造、生成或输出的产品、材料、事件或其他结果。",
            positive_examples=["一个生产者列出多个并列产出物时，每个产出物分别由该生产者产生。"],
            negative_examples=[
                "供应商甲把组件交付给客户乙；客户乙是接收方而不是产出物。",
                "组织甲使用产品乙，不能据此认定组织甲生产了产品乙。",
            ],
            optional_soft_cues=["produces", "生产"],
        ),
        _versioned_relation_policy(
            "depends_on", ["*"], ["*"], "主体的运行、实现或存在明确需要客体作为依赖或前提。",
            "从依赖方指向被依赖的对象或必要前提。",
            subject_role_description="其运行、实现或存在需要另一个对象的实体。",
            object_role_description="主体所需的依赖项、基础条件或先决条件。",
            positive_examples=["系统甲明确依赖服务乙才能运行。"],
            negative_examples=[
                "甲与乙有关联，但原文没有陈述任何必要性或依赖关系。",
                "事件甲导致结果乙，不表示事件甲依赖结果乙，也不表示结果乙依赖事件甲。",
                "系统甲生成产物乙，不表示系统甲依赖产物乙。",
            ],
            optional_soft_cues=["depends on", "依赖"],
        ),
        _versioned_relation_policy(
            "causes", ["Event"], ["Event"], "主体所表示的完整事件、动作或条件直接引发客体所表示的完整结果。",
            "从完整的原因事件、动作或条件指向完整的结果事件或状态。",
            subject_role_description="能够产生后果的完整事件、动作或条件，必须保留触发词和状态变化。",
            object_role_description="由原因导致的完整事件、状态或后果，不能只保留其中的内部名词。",
            positive_examples=["完整条件甲被原文明确陈述为直接导致完整结果乙。"],
            negative_examples=[
                "甲与乙只是在同一上下文中出现，没有明确因果陈述。",
                "候选端点只截取了完整原因或结果事件中的内部对象名。",
            ],
            optional_soft_cues=["causes", "导致"],
        ),
        _versioned_relation_policy(
            "supports", ["*"], ["*"],
            "主体提供能力、框架、资源或保障，使客体所表示的功能、行动、分析或目标得以实现。",
            "从提供支持的主体指向被其支持的功能、行动、分析或目标。",
            subject_role_description=(
                "明确提供能力、框架、资源、保障或实现条件的系统、数据集、平台、组织或过程。"
            ),
            object_role_description=(
                "被主体直接支持、促进、保障或作为实现框架的功能、行动、分析或目标；"
                "仅与主体同段出现、被主体检查或由主体产生的对象不属于此角色。"
            ),
            positive_examples=[
                "数据集为区域分析提供框架。",
                "平台支持某项运行或跟踪活动。",
            ],
            negative_examples=[
                "两个实体只是在同一列表或表格中相邻。",
                "系统检查、识别或监控对象，不等于系统支持对象。",
                "主体生成客体，不等于主体支持客体。",
            ],
            optional_soft_cues=["supports", "provides a framework for", "支持", "提供框架"],
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
        "argument_complete": {"type": "boolean"},
        "standalone_referential": {"type": "boolean"},
        "complete_mention_start": {"type": "integer"},
        "complete_mention_end": {"type": "integer"},
        "confidence": {"type": "number"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
    },
    "required": [
        "id", "name", "type", "mention_start", "mention_end", "confidence",
        "argument_complete", "standalone_referential",
        "complete_mention_start", "complete_mention_end",
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

# Compact contract for the enhanced entity path. Exact offsets are grounded
# deterministically from the copied surface and evidence, leaving the model to
# focus on coverage and minimal mention boundaries.
GROUNDED_MENTION_INVENTORY_FORMAT = {
    "$id": "knowledge-forge-grounded-mention-inventory-v1",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "maxItems": Config.entity_grounded_max_items_per_segment,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "surface": {
                        "type": "string", "minLength": 1, "maxLength": 200,
                    },
                    "evidence": {
                        "type": "string", "minLength": 1, "maxLength": 800,
                    },
                    "kind": {"type": "string", "maxLength": 100},
                },
                "required": ["surface", "evidence"],
            },
        },
    },
    "required": ["items"],
}

PROPOSITION_ARGUMENT_EXTRACTION_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": Config.entity_only_max_items_per_segment,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "mention_start": {"type": "integer"},
                    "mention_end": {"type": "integer"},
                    "argument_complete": {"type": "boolean"},
                    "proposition_argument": {"type": "boolean", "const": True},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "name", "type", "mention_start", "mention_end",
                    "argument_complete", "proposition_argument",
                ],
            },
        },
    },
    "required": ["entities"],
}

OPEN_RELATION_OBSERVATION_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "maxItems": 128,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type": "string", "maxLength": 240},
                    "relation_expression": {
                        "type": "string", "maxLength": 160,
                    },
                    "object": {"type": "string", "maxLength": 240},
                    "evidence": {"type": "string", "maxLength": 1200},
                    "assertion_status": {
                        "type": "string", "maxLength": 40,
                    },
                },
                "required": [
                    "subject", "relation_expression", "object", "evidence",
                    "assertion_status",
                ],
            },
        },
    },
}
# Local structured-output models often choose a semantically equivalent root
# noun. The semantic collector below applies the same exact-source and status
# gates to every alias; declaring transport aliases avoids losing a valid
# observation inventory before that parser can inspect it.
OPEN_RELATION_OBSERVATION_FORMAT["properties"]["observations"] = copy.deepcopy(
    OPEN_RELATION_OBSERVATION_FORMAT["properties"]["items"]
)
OPEN_RELATION_OBSERVATION_FORMAT["properties"]["relations"] = copy.deepcopy(
    OPEN_RELATION_OBSERVATION_FORMAT["properties"]["items"]
)

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
                    "evidence_quotes": {
                        "type": "array", "items": {"type": "string"},
                    },
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
                    "subject_role_description": {"type": "string"},
                    "object_role_description": {"type": "string"},
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
                    "structural_evidence_kinds": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "structural_role_labels": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "source_role_pairs": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "grammatical_subject->action_theme",
                                "action_theme->role_complement",
                                "grammatical_subject->role_complement",
                                "grammatical_subject->direct_argument",
                                "direct_argument->grammatical_subject",
                                "action_theme->grammatical_subject",
                                "role_complement->grammatical_subject",
                                "collective_participant->collective_participant",
                                "possessor->possessed",
                                "possessed->possessor",
                            ],
                        },
                    },
                    "conflicts_with": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "negation_policy": {
                        "type": "string",
                        "enum": [
                            "reject_negated",
                            "allow_explicit_negative_relation",
                        ],
                    },
                    "modality_policy": {
                        "type": "string", "enum": ["require_asserted"],
                    },
                    "confidence": {"type": "number"},
                    "evidence_chunks": {"type": "array", "items": {"type": "string"}},
                    "evidence_quotes": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": [
                    "name", "description", "source_types", "target_types",
                    "semantic_definition", "domain_types", "range_types",
                    "direction_description", "subject_role_description",
                    "object_role_description", "symmetric", "inverse_predicate",
                    "canonical_orientation", "publish_mode", "reflexive",
                    "positive_examples", "negative_examples",
                    "optional_soft_cues", "confidence", "evidence_chunks",
                    "evidence_quotes",
                ],
            },
        },
    },
}


# Open-observation induction is intentionally relation-only and must never ask
# a local model to regenerate an unbounded copy of the complete logical
# schema.  Keeping this transport contract small prevents one verbose response
# from discarding an otherwise source-grounded observation group.  The
# candidate still passes the ordinary evidence, metadata, novelty and policy
# gates before it can enter a schema version.
OPEN_RELATION_INDUCTION_FORMAT = copy.deepcopy(SCHEMA_REVIEW_FORMAT)
OPEN_RELATION_INDUCTION_FORMAT["additionalProperties"] = False
OPEN_RELATION_INDUCTION_FORMAT["properties"]["summary"]["maxLength"] = 320
OPEN_RELATION_INDUCTION_FORMAT["properties"]["entity_type_candidates"][
    "maxItems"
] = 0
OPEN_RELATION_INDUCTION_FORMAT["properties"]["relation_type_candidates"][
    "maxItems"
] = 1
_OPEN_RELATION_POLICY_FORMAT = OPEN_RELATION_INDUCTION_FORMAT["properties"][
    "relation_type_candidates"
]["items"]
_OPEN_RELATION_POLICY_FORMAT["additionalProperties"] = False
for _field_name in (
    "name", "description", "semantic_definition", "direction_description",
    "subject_role_description", "object_role_description",
):
    _OPEN_RELATION_POLICY_FORMAT["properties"][_field_name]["maxLength"] = 1200
for _field_name, _max_items, _max_length in (
    ("source_types", 8, 128),
    ("target_types", 8, 128),
    ("domain_types", 8, 128),
    ("range_types", 8, 128),
    ("positive_examples", 4, 512),
    ("negative_examples", 4, 512),
    ("optional_soft_cues", 8, 128),
    ("source_role_pairs", 4, 80),
    ("conflicts_with", 8, 128),
    ("evidence_chunks", 3, 256),
    ("evidence_quotes", 3, 1024),
):
    _array_format = _OPEN_RELATION_POLICY_FORMAT["properties"][_field_name]
    _array_format["maxItems"] = _max_items
    _array_format["items"] = {"type": "string", "maxLength": _max_length}
OPEN_RELATION_INDUCTION_FORMAT["properties"][
    "induced_relation_family"
] = copy.deepcopy(_OPEN_RELATION_POLICY_FORMAT)
OPEN_RELATION_INDUCTION_FORMAT["properties"]["relation_types"] = {
    "type": "array",
    "maxItems": 1,
    "items": copy.deepcopy(_OPEN_RELATION_POLICY_FORMAT),
}

# Project-wide induction assigns a stable reusable identifier and concise role
# hints to several frozen observation groups in one bounded request.  It does
# not regenerate the full relation-policy object: doing so made local models
# spend most of their output on boilerplate and frequently lose the immutable
# group ID.  The complete additive policy is compiled deterministically from
# this compact naming result plus the exact observations below.
_OPEN_RELATION_BATCH_MAX = max(
    1, min(16, Config.schema_max_relation_additions)
)
_OPEN_RELATION_BATCH_POLICY_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "observation_group_id": {"type": "string", "maxLength": 32},
        "observation_group_ids": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": {"type": "string", "maxLength": 32},
        },
        "name": {"type": "string", "maxLength": 128},
        "label": {"type": "string", "maxLength": 128},
        "label_en": {"type": "string", "maxLength": 128},
        "short_identifier": {"type": "string", "maxLength": 128},
        "predicate": {"type": "string", "maxLength": 128},
        "description": {"type": "string", "maxLength": 800},
        "description_zh": {"type": "string", "maxLength": 800},
        "description_cn": {"type": "string", "maxLength": 800},
        "semantic_definition": {"type": "string", "maxLength": 1200},
        "source_types": {
            "type": "array", "maxItems": 8,
            "items": {"type": "string", "maxLength": 128},
        },
        "target_types": {
            "type": "array", "maxItems": 8,
            "items": {"type": "string", "maxLength": 128},
        },
        "domain_types": {
            "type": "array", "maxItems": 8,
            "items": {"type": "string", "maxLength": 128},
        },
        "range_types": {
            "type": "array", "maxItems": 8,
            "items": {"type": "string", "maxLength": 128},
        },
        "subject_role_description": {"type": "string", "maxLength": 800},
        "object_role_description": {"type": "string", "maxLength": 800},
        "direction_description": {"type": "string", "maxLength": 800},
    },
}
_OPEN_RELATION_BATCH_ARRAY_FORMAT = {
    "type": "array",
    "maxItems": _OPEN_RELATION_BATCH_MAX,
    "items": copy.deepcopy(_OPEN_RELATION_BATCH_POLICY_FORMAT),
}
_OPEN_RELATION_NESTED_BATCH_ITEM_FORMAT = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "observation_group_id": {"type": "string", "maxLength": 32},
        "relation_contract": copy.deepcopy(
            _OPEN_RELATION_BATCH_POLICY_FORMAT
        ),
    },
}
OPEN_RELATION_BATCH_INDUCTION_FORMAT = {
    "$id": "knowledge-forge-open-relation-batch-naming-v2",
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "relation_type_candidates": copy.deepcopy(
            _OPEN_RELATION_BATCH_ARRAY_FORMAT
        ),
        "relation_types": copy.deepcopy(_OPEN_RELATION_BATCH_ARRAY_FORMAT),
        "families": copy.deepcopy(_OPEN_RELATION_BATCH_ARRAY_FORMAT),
        "candidates": copy.deepcopy(_OPEN_RELATION_BATCH_ARRAY_FORMAT),
        "rows": copy.deepcopy(_OPEN_RELATION_BATCH_ARRAY_FORMAT),
        "candidate_relations": {
            "type": "array",
            "maxItems": _OPEN_RELATION_BATCH_MAX,
            "items": copy.deepcopy(_OPEN_RELATION_NESTED_BATCH_ITEM_FORMAT),
        },
    },
}


SCHEMA_EVIDENCE_GROUNDING_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "evidence_quotes": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string"},
        },
        "confidence": {"type": "number"},
    },
    "required": ["evidence_quotes", "confidence"],
}


SCHEMA_POLICY_VERIFICATION_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTED", "REJECTED", "UNCERTAIN"],
        },
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}

SCHEMA_POLICY_NOVELTY_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["NOVEL", "REDUNDANT", "UNCERTAIN"],
        },
        "reason": {"type": "string"},
        "overlapping_relation": {"type": ["string", "null"]},
        "exact_role_alignment": {"type": "boolean"},
        "core_contract_equivalent": {"type": "boolean"},
        "orientation_alignment": {
            "type": "string",
            "enum": ["same", "inverse", "none"],
        },
    },
    "required": [
        "verdict", "reason", "overlapping_relation",
        "exact_role_alignment", "core_contract_equivalent",
        "orientation_alignment",
    ],
}


SCHEMA_POLICY_REPAIR_FORMAT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "repairable": {"type": "boolean"},
        "reason": {"type": "string"},
        "candidate": {
            "type": "object",
            "additionalProperties": False,
            "properties": copy.deepcopy(
                SCHEMA_REVIEW_FORMAT["properties"]
                ["relation_type_candidates"]["items"]["properties"]
            ),
            "required": copy.deepcopy(
                SCHEMA_REVIEW_FORMAT["properties"]
                ["relation_type_candidates"]["items"]["required"]
            ),
        },
    },
    "required": ["repairable", "reason"],
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
_STRUCTURED_RELATION_MARKER = re.compile(
    r"(?m)^\s*(?:[-*+]\s+)?\S.*?\s+--[^\n<>]+?-->\s+\S"
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


_GROUNDING_TOKEN = re.compile(
    r"[\w]+(?:['’\-][\w]+)*|[\u3400-\u4dbf\u4e00-\u9fff]",
    re.UNICODE,
)
_LEADING_MENTION_DETERMINER = re.compile(
    r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE,
)


def _ground_copied_mention_surface(
    content: str,
    surface: str,
    evidence: str,
) -> tuple[int, int] | None:
    """Ground a compact model mention to one immutable source occurrence.

    Exact source matching is authoritative.  The only fallback ignores common
    inline formatting marks (for example Markdown backticks around a code
    identifier) while requiring the complete normalized token sequence.  It
    never performs fuzzy, semantic, alias or embedding matching.
    """

    requested = re.sub(r"\s+", " ", str(surface or "")).strip()
    copied_evidence = str(evidence or "").strip()
    if not requested:
        return None
    evidence_spans = _surface_spans(content, copied_evidence)
    exact_spans = _surface_spans(content, requested)
    if evidence_spans:
        contained = [
            span for span in exact_spans
            if any(left <= span[0] < span[1] <= right for left, right in evidence_spans)
        ]
        if len(contained) == 1:
            selected = contained[0]
        elif contained:
            selected = contained[0]
        elif len(exact_spans) == 1:
            selected = exact_spans[0]
        else:
            selected = None
    else:
        selected = exact_spans[0] if len(exact_spans) == 1 else None

    if selected is None:
        wanted = [
            token.group(0).casefold()
            for token in _GROUNDING_TOKEN.finditer(
                requested.replace("`", "").replace("*", "")
            )
        ]
        source_tokens = list(_GROUNDING_TOKEN.finditer(content))
        candidates: list[tuple[int, int]] = []
        if wanted:
            for index in range(0, len(source_tokens) - len(wanted) + 1):
                window = source_tokens[index:index + len(wanted)]
                if [item.group(0).casefold() for item in window] != wanted:
                    continue
                start, end = window[0].start(), window[-1].end()
                while start > 0 and content[start - 1] in "`*_~":
                    start -= 1
                while end < len(content) and content[end] in "`*_~":
                    end += 1
                candidates.append((start, end))
        if evidence_spans:
            candidates = [
                span for span in candidates
                if any(left <= span[0] < span[1] <= right for left, right in evidence_spans)
            ]
        if len(candidates) != 1:
            return None
        selected = candidates[0]

    start, end = selected
    # A local model occasionally copies a callable token through its opening
    # delimiter only.  Completing an immediately adjacent balanced empty pair
    # is a source-character repair, not a semantic boundary guess.
    if (
        end < len(content)
        and content[start:end].endswith(("(", "[", "{"))
        and content[end] == {"(": ")", "[": "]", "{": "}"}[content[end - 1]]
    ):
        end += 1
    source_surface = content[start:end]
    determiner = _LEADING_MENTION_DETERMINER.match(source_surface)
    if determiner and determiner.end() < len(source_surface):
        start += determiner.end()
        while start < end and content[start].isspace():
            start += 1
    return (start, end) if 0 <= start < end <= len(content) else None


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
    # Schema evidence is often expressed by a short, explicit paragraph, list
    # item or table row.  Treating ``SCHEMA_SAMPLE_MIN_CHARS`` as a per-chunk
    # admission threshold discarded most concise Chinese sections before the
    # reviewer could see them.  Keep that setting as the *batch* floor in the
    # sampler and reject here only truly tiny fragments and parser-recognised
    # navigation content.
    if len(meaningful) < 24 or _looks_like_table_of_contents(_chunk_title(chunk), body):
        return 0.0
    length_score = min(1.0, len(meaningful) / 900)
    sentence_score = min(1.0, len(re.findall(r"[。！？；.!?;]", body)) / 8)
    lexical_diversity = min(
        1.0,
        len(set(character.casefold() for character in meaningful))
        / max(12.0, math.sqrt(len(meaningful)) * 3.0),
    )
    structural_relation_score = min(
        1.0, len(_STRUCTURED_RELATION_MARKER.findall(body)) / 3,
    )
    # Parser-declared relationship notation is a domain-neutral signal that a
    # chunk is valuable for *schema discovery*.  It never creates a relation;
    # candidate semantics, direction and publication remain model-reviewed and
    # evidence-gated below.
    return min(
        1.0,
        0.45 * length_score
        + 0.20 * sentence_score
        + 0.20 * lexical_diversity
        + 0.15 * structural_relation_score,
    )


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


_OPEN_RELATION_MODAL_PREFIX = re.compile(
    r"^(?:(?:shall|must|should|is\s+required\s+to|"
    r"are\s+required\s+to|has\s+to|have\s+to)\s+|"
    r"(?:应当|必须|须|需要)\s*)",
    flags=re.IGNORECASE,
)
_OPEN_RELATION_ROLE_CARRIER_PREFIX = re.compile(
    r"^(?:"
    r"(?:has|have|had)\s+(?:the\s+)?"
    r"(?:responsibility|duty|obligation)\s+(?:to|for)\s+|"
    r"(?:is|are|was|were)\s+"
    r"(?:responsible|obligated)\s+(?:to|for)\s+|"
    r"(?:responsibility|duty|obligation)\s+(?:to|for)\s+|"
    r"(?:负责|有责任|有义务)\s*"
    r")",
    flags=re.IGNORECASE,
)
_OPEN_RELATION_AUXILIARY_TOKENS = frozenset({
    "am", "are", "be", "been", "being", "can", "could", "did", "do",
    "does", "had", "has", "have", "is", "may", "might", "must", "shall",
    "should", "was", "were", "will", "would",
})


def _normalize_open_relation_expression(value: str) -> str:
    """Remove grammatical carriers while preserving the semantic predicate.

    Requirement prose and table headers often express the same action as
    ``shall check``, ``has responsibility to check`` and ``check``.  The
    auxiliary or responsibility carrier is not an independent endpoint role;
    retaining it creates duplicate Schema families and crowds out genuinely
    different relations.  This normalization is language/morphology based and
    never branches on a logical predicate identifier.
    """

    normalized = re.sub(r"\s+", " ", str(value or "").strip())
    normalized = _OPEN_RELATION_MODAL_PREFIX.sub("", normalized).strip()
    normalized = _OPEN_RELATION_ROLE_CARRIER_PREFIX.sub("", normalized).strip()
    return normalized


def _is_bare_open_relation_carrier(value: str) -> bool:
    """Return whether an English expression contains grammar but no role."""

    tokens = re.findall(r"[a-z]+", str(value or "").strip().casefold())
    return bool(tokens) and all(
        token in _OPEN_RELATION_AUXILIARY_TOKENS for token in tokens
    )


def _has_chinese_user_text(value: Any) -> bool:
    """Require Chinese presentation text without interpreting its semantics."""

    return bool(re.search(r"[\u3400-\u9fff]", str(value or "")))


def _schema_knowledge_language(
    project: Mapping[str, Any], samples: list[dict[str, Any]],
) -> str:
    texts = [
        str(item.get("_schema_text") or item.get("content") or "")
        for item in samples
    ]
    return resolve_knowledge_language(
        project.get("knowledge_language") or "auto", texts,
    )


def _schema_language_instruction(language: str) -> str:
    if language == ENGLISH:
        return (
            "Use short English identifiers and write every user-facing summary, "
            "description, semantic definition, role, direction, example and "
            "explanation in clear English. Preserve proper nouns, identifiers, "
            "code and verbatim evidence in their source spelling."
        )
    return (
        "Use short English identifiers, but write every user-facing summary, "
        "description, semantic definition, role, direction, example and "
        "explanation in clear Simplified Chinese. Preserve proper nouns, "
        "identifiers, code and verbatim evidence in their source spelling."
    )


def _schema_user_text_matches(value: Any, language: str) -> bool:
    return user_facing_text_matches_language(value, language)


def _relation_expression_family_key(value: str) -> str:
    """Return a conservative inflection-insensitive grouping key.

    Open observations copy short source predicates.  English table prose often
    supplies third-person singular forms while a requirement below uses the
    base form (``checks``/``check``).  Treating those as unrelated families
    wastes schema slots and invites conflicting policies.  This intentionally
    normalizes only the first ASCII verb token; particles such as ``to`` stay
    in the key so active actor-to-theme and passive theme-to-recipient roles do
    not collapse.
    """

    normalized = _normalize_open_relation_expression(value).casefold()
    tokens = re.findall(r"[a-z]+|[\u3400-\u9fff]+|[0-9]+", normalized)
    if not tokens:
        return ""
    irregular = {
        "has": "have", "does": "do", "goes": "go",
        "is": "be", "are": "be", "was": "be", "were": "be",
        "sends": "send", "sent": "send",
    }
    first = tokens[0]
    if first in irregular:
        tokens[0] = irregular[first]
    elif re.fullmatch(r"[a-z]+", first):
        if len(first) > 4 and first.endswith("ies"):
            tokens[0] = first[:-3] + "y"
        elif len(first) > 4 and first.endswith((
            "ches", "shes", "sses", "xes", "zes", "oes",
        )):
            tokens[0] = first[:-2]
        elif len(first) > 3 and first.endswith("ses"):
            # uses -> use, causes -> cause; passes is covered above.
            tokens[0] = first[:-1]
        elif len(first) > 3 and first.endswith("s") and not first.endswith("ss"):
            tokens[0] = first[:-1]
    return "|".join(tokens)


def _source_relation_identifier(value: str) -> str:
    """Prefer a reusable source-stable ASCII identifier when one is available."""

    normalized = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    passive = re.match(
        r"^(?:is|are|was|were|be|been|being)\s+([a-z]+(?:\s+.+)?)$",
        normalized,
    )
    if passive and re.match(r"^[a-z]+(?:ed|en)\b", passive.group(1)):
        normalized = passive.group(1)
    if not normalized or re.search(r"[^a-z0-9_\s-]", normalized):
        return ""
    # A bare grammatical carrier does not name a reusable semantic relation.
    # Let the bounded induction model use the frozen complement/evidence to
    # choose a role-specific identifier instead of publishing ``is``/``has``.
    if _is_bare_open_relation_carrier(normalized):
        return ""
    identifier = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return identifier if _valid_identifier(identifier) else ""


def _strip_endpoint_suffix_from_identifier(
    value: str,
    observations: list[dict[str, Any]],
    endpoint_type_names: list[str] | tuple[str, ...] = (),
) -> str:
    """Remove a model-added endpoint or declared-type suffix from an ID."""

    identifier = str(value or "").strip()
    if not _valid_identifier(identifier):
        return identifier
    endpoint_tokens = {
        token
        for item in observations
        for field in ("subject_surface", "object_surface")
        for token in re.findall(
            r"[a-z][a-z0-9]*",
            str(item.get(field) or "").casefold(),
        )
        if len(token) > 1 and token not in {"the", "an", "a"}
    }
    endpoint_tokens.update(
        token
        for type_name in endpoint_type_names
        for token in re.findall(
            r"[a-z][a-z0-9]*", str(type_name or "").casefold(),
        )
        if len(token) > 1 and token not in {"generic", "entity"}
    )
    parts = identifier.split("_")
    while len(parts) > 1 and parts[-1].casefold() in endpoint_tokens:
        parts.pop()
    reusable = "_".join(parts)
    return reusable if _valid_identifier(reusable) else identifier


class SchemaEvolutionService:
    def __init__(self, database: Database, ollama: OllamaClient | None = None):
        self.db = database
        self.ollama = ollama or OllamaClient()

    def ensure_base(self, project_id: int) -> dict[str, Any]:
        active = self.db.ensure_base_schema(project_id, BASE_SCHEMA)
        return self._ensure_runtime_schema_capabilities(project_id, active)

    def _ensure_runtime_schema_capabilities(
        self,
        project_id: int,
        active: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Add newly introduced generic runtime policies without redefining any.

        Long-lived projects retain their active logical Schema across service
        upgrades.  A new evidence carrier would otherwise be available only
        to newly created projects, making extraction behavior depend on project
        age.  Runtime capabilities are recognized through versioned policy
        metadata and appended exactly once. Existing predicates, meanings and
        publication modes are never overwritten or renamed.
        """

        active_schema = copy.deepcopy(dict(active.get("schema_json") or {}))
        active_relations = active_schema.get("relation_types")
        if not isinstance(active_relations, list):
            raise ValueError("active Schema requires relation_types")

        def capability(value: Any) -> tuple[bool, bool]:
            if not isinstance(value, Mapping):
                return False, False
            policy = value.get("open_relation_policy") or {}
            if not isinstance(policy, Mapping):
                return False, False
            return bool(policy.get("enabled")), bool(
                policy.get("fallback_only")
            )

        active_capabilities = {
            capability(value)
            for value in active_relations
            if capability(value)[0]
        }
        active_names = {
            str(value.get("name") or "")
            for value in active_relations
            if isinstance(value, Mapping)
        }
        additions: list[dict[str, Any]] = []
        for base_relation in BASE_SCHEMA.get("relation_types") or ():
            signature = capability(base_relation)
            if not signature[0] or signature in active_capabilities:
                continue
            name = str(base_relation.get("name") or "")
            if not name or name in active_names:
                raise ValueError(
                    "active Schema conflicts with a required additive runtime "
                    "relation capability"
                )
            additions.append(copy.deepcopy(dict(base_relation)))
            active_names.add(name)
            active_capabilities.add(signature)
        if not additions:
            return dict(active)

        previous_schema_version = str(
            active_schema.get("version")
            or active_schema.get("schema_version")
            or f"schema-{int(active.get('version') or 1)}"
        )
        active_schema["version"] = (
            f"{previous_schema_version}+runtime-capabilities-"
            f"{BASE_SCHEMA.get('version') or 'current'}"
        )
        active_schema["relation_types"] = [
            *active_relations,
            *additions,
        ]
        report = {
            "summary": "Added generic evidence-preserving runtime capabilities.",
            "policy": "additive-runtime-capability-upgrade",
            "previous_active_version": int(active.get("version") or 1),
            "base_schema_profile": str(BASE_SCHEMA.get("version") or ""),
            "added_relation_capabilities": [
                str(item.get("name") or "") for item in additions
            ],
        }
        created = self.db.create_schema_version(
            int(project_id),
            active_schema,
            report,
            [
                int(value)
                for value in (active.get("source_document_ids") or ())
            ],
            created_by="system",
        )
        return dict(created)

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
        revision_ids = {
            str(item.get("revision_id") or "").strip()
            for item in chunks
            if str(item.get("revision_id") or "").strip()
        }
        # Ingestion evaluates Schema before activating the new document
        # revision. Read vectors from that explicitly frozen revision instead
        # of silently falling back to the previously active contribution.
        revision_id = next(iter(revision_ids)) if len(revision_ids) == 1 else None
        vector_rows = self.db.list_document_embeddings(
            document_id,
            "chunk",
            model,
            revision_id=revision_id,
        )
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
            "embedding_revision_scope": revision_id or "active",
            "target_chunks": target_chunks,
            "target_chars": target_chars,
            "chunk_ceiling": Config.schema_sample_ceiling_chunks,
            "char_ceiling": Config.schema_sample_ceiling_chars,
        }
        return selected, diagnostics

    def select_representative_corpus_chunks(
        self,
        project: dict[str, Any],
        documents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Select one bounded, document-balanced sample for a project build.

        Per-document schema evolution made the active vocabulary depend on
        upload order and reset the addition limit for every file.  This sampler
        first gathers each document's ordinary representative candidates, then
        applies one corpus-wide chunk/character ceiling with document and
        section diversity.  Text/vector indexing remains complete; only the
        comparatively expensive logical-Schema review is sampled.
        """

        candidates: list[dict[str, Any]] = []
        document_diagnostics: list[dict[str, Any]] = []
        eligible_chunk_count = 0
        input_document_count = max(1, len(documents))
        per_document_candidate_cap = (
            Config.schema_sample_ceiling_chunks
            if input_document_count == 1 else max(
                2,
                math.ceil(
                    Config.schema_sample_ceiling_chunks
                    / math.sqrt(input_document_count)
                ),
            )
        )
        for document_order, document in enumerate(
            sorted(documents, key=lambda item: int(item["id"])), start=1,
        ):
            document_id = int(document["id"])
            chunks = [
                item for item in self.db.list_document_chunks(document_id)
                if (
                    item.get("metadata")
                    if isinstance(item.get("metadata"), Mapping) else {}
                ).get("chunk_role") != "parent"
                and (
                    item.get("metadata")
                    if isinstance(item.get("metadata"), Mapping) else {}
                ).get("graph_extraction_enabled", True) is not False
            ]
            samples, sampling = self.select_representative_chunks(
                project, document_id, chunks,
            )
            if len(samples) > per_document_candidate_cap:
                samples = sorted(
                    samples,
                    key=lambda item: (
                        float(item.get("_schema_score") or 0.0),
                        float(item.get("_schema_information") or 0.0),
                        len(str(item.get("_schema_text") or "")),
                    ),
                    reverse=True,
                )[:per_document_candidate_cap]
            eligible_chunk_count += int(sampling.get("eligible_chunks") or 0)
            document_diagnostics.append({
                "document_id": document_id,
                "document_name": str(
                    document.get("original_filename") or ""
                ),
                "sampling": sampling,
            })
            for item in samples:
                candidate = dict(item)
                candidate["_schema_document_id"] = document_id
                candidate["_schema_document_order"] = document_order
                candidate["_schema_local_order"] = int(
                    item.get("_schema_order") or 0
                )
                # Section identifiers are document-local in several parsers.
                # Prefixing them prevents two files' first sections from being
                # mistaken for repeated evidence from one section.
                candidate["_schema_section"] = (
                    f"d{document_id}:"
                    f"{str(item.get('_schema_section') or _chunk_section(item))}"
                )
                candidates.append(candidate)

        if not candidates:
            return [], {
                "strategy": "project-batch-document-balanced-mmr",
                "document_count": len(documents),
                "eligible_chunks": eligible_chunk_count,
                "candidate_chunks": 0,
                "selected_chunks": [],
                "selected_chars": 0,
                "document_diagnostics": document_diagnostics,
            }

        document_count = len({
            int(item["_schema_document_id"]) for item in candidates
        })
        volume_target = math.ceil(
            math.sqrt(max(1, eligible_chunk_count)) * 1.75
        )
        target_chunks = min(
            Config.schema_sample_ceiling_chunks,
            len(candidates),
            max(
                Config.schema_sample_base_chunks,
                min(document_count, Config.schema_sample_ceiling_chunks),
                volume_target,
            ),
        )
        target_chars = min(
            Config.schema_sample_ceiling_chars,
            max(
                Config.schema_sample_base_chars,
                target_chunks * Config.schema_sample_chars_per_chunk,
            ),
        )
        max_information = max(
            (float(item.get("_schema_information") or 0.0)
             for item in candidates),
            default=1.0,
        ) or 1.0
        selected: list[dict[str, Any]] = []
        remaining = list(candidates)
        used_chars = 0

        while remaining and len(selected) < target_chunks:
            selected_documents = {
                int(item["_schema_document_id"]) for item in selected
            }
            selected_sections = {
                str(item["_schema_section"]) for item in selected
            }

            def corpus_score(item: dict[str, Any]) -> tuple[float, int, int]:
                vector = list(item.get("_schema_vector") or [])
                similarities = [
                    _cosine(vector, list(chosen.get("_schema_vector") or []))
                    for chosen in selected
                    if vector and chosen.get("_schema_vector")
                ]
                diversity = (
                    1.0 - max(similarities)
                    if similarities else (1.0 if not selected else 0.5)
                )
                information = float(
                    item.get("_schema_information") or 0.0
                ) / max_information
                document_bonus = (
                    1.0
                    if int(item["_schema_document_id"])
                    not in selected_documents else 0.0
                )
                section_bonus = (
                    1.0
                    if str(item["_schema_section"])
                    not in selected_sections else 0.0
                )
                score = (
                    0.42 * diversity
                    + 0.28 * information
                    + 0.22 * document_bonus
                    + 0.08 * section_bonus
                )
                # Stable tie breakers keep an identical source snapshot
                # reproducible without biasing the semantic score.
                return (
                    score,
                    -int(item["_schema_document_order"]),
                    -int(item["_schema_local_order"]),
                )

            choice = max(remaining, key=corpus_score)
            remaining.remove(choice)
            remaining_chars = target_chars - used_chars
            if remaining_chars < Config.schema_sample_min_chars and selected:
                break
            text = str(
                choice.get("_schema_text")
                or choice.get("content") or ""
            )[:remaining_chars]
            if not text.strip():
                continue
            selected_choice = dict(choice)
            selected_choice["_schema_text"] = text
            selected.append(selected_choice)
            used_chars += len(text)

        selected.sort(key=lambda item: (
            int(item["_schema_document_order"]),
            int(item["_schema_local_order"]),
            _chunk_key(item),
        ))
        for order, item in enumerate(selected):
            item["_schema_order"] = order
        diagnostics = {
            "strategy": "project-batch-document-balanced-mmr",
            "document_count": document_count,
            "eligible_chunks": eligible_chunk_count,
            "candidate_chunks": len(candidates),
            "selected_chunks": [_chunk_key(item) for item in selected],
            "selected_documents": len({
                int(item["_schema_document_id"]) for item in selected
            }),
            "selected_sections": len({
                str(item["_schema_section"]) for item in selected
            }),
            "selected_chars": used_chars,
            "target_chunks": target_chunks,
            "target_chars": target_chars,
            "chunk_ceiling": Config.schema_sample_ceiling_chunks,
            "char_ceiling": Config.schema_sample_ceiling_chars,
            "review_batch_size": Config.schema_corpus_review_batch_size,
            "per_document_candidate_cap": per_document_candidate_cap,
            "document_diagnostics": document_diagnostics,
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
    def _candidate_payload(
        value: dict[str, Any],
        *,
        fallback_evidence_chunks: list[str] | None = None,
    ) -> dict[str, Any]:
        """Normalize bounded schema-review response shapes.

        Local structured-output models occasionally unwrap a one-item array
        and return the relation candidate itself.  Treat that as a transport
        shape variant, not an empty review.  Fallback evidence is permitted
        only from the exact chunks in that model call; the default runtime
        sends one representative chunk per call so provenance cannot be
        broadened to unrelated text.
        """

        if not isinstance(value, dict):
            return {
                "summary": "", "entity_type_candidates": [],
                "relation_type_candidates": [],
            }
        entities = value.get("entity_type_candidates")
        relations = value.get("relation_type_candidates")
        if entities is None:
            entities = value.get("entity_type_additions") or []
        if relations is None:
            relations = value.get("relation_type_additions") or []
        # Structured-output models may use a semantically equivalent transport
        # key even when the supplied schema asks for relation_type_candidates.
        # Normalize the shape here; the downstream evidence, novelty and policy
        # gates remain authoritative and no predicate name is used as a rule.
        if relations is None or not relations:
            relations = value.get("relation_candidates") or []
        if relations is None or not relations:
            relations = value.get("relation_type_additions") or []
        family = (
            value.get("candidate_relation_family")
            or value.get("induced_relation_family")
            or value.get("candidate")
            or value.get("relation")
        )
        if (
            (relations is None or not relations)
            and isinstance(family, Mapping)
            and (
                family.get("name") or family.get("predicate")
            )
        ):
            relations = [family]
        if isinstance(relations, Mapping):
            relations = [relations]
        if not relations and (
            value.get("semantic_definition")
            or value.get("domain_types")
            or value.get("direction_description")
        ):
            relations = [value]
        if not entities and not relations and isinstance(value.get("items"), list):
            items = [item for item in value["items"] if isinstance(item, dict)]
            relations = [
                item for item in items
                if item.get("semantic_definition") or item.get("domain_types")
            ]
            entities = [item for item in items if item not in relations]
        fallback = list(dict.fromkeys(
            str(item) for item in fallback_evidence_chunks or [] if str(item)
        ))
        normalized_relations = []
        for item in relations or []:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            if not candidate.get("name"):
                candidate["name"] = str(
                    candidate.get("predicate")
                    or candidate.get("relation_id")
                    or candidate.get("label")
                    or candidate.get("label_en")
                    or candidate.get("short_identifier")
                    or ""
                ).strip()
            if not candidate.get("description"):
                description_alias = str(
                    candidate.get("description_zh")
                    or candidate.get("description_cn")
                    or candidate.get("user_facing_description_cn")
                    or candidate.get("description_en")
                    or ""
                ).strip()
                if description_alias:
                    candidate["description"] = description_alias
            if not candidate.get("semantic_definition"):
                semantic_alias = str(
                    candidate.get("description")
                    or candidate.get("description_zh")
                    or candidate.get("user_facing_description_cn")
                    or ""
                ).strip()
                if semantic_alias:
                    candidate["semantic_definition"] = semantic_alias
            # Keep the versioned policy field names stable when a model uses
            # compact aliases.  These are structural aliases only; missing
            # semantics still fail closed in the policy verifier.
            if not candidate.get("direction_description") and candidate.get("direction"):
                candidate["direction_description"] = candidate.get("direction")
            if not candidate.get("domain_types") and candidate.get("source_types"):
                candidate["domain_types"] = list(candidate.get("source_types") or [])
            if not candidate.get("range_types") and candidate.get("target_types"):
                candidate["range_types"] = list(candidate.get("target_types") or [])
            cited = [
                str(value).strip().strip("[]")
                for value in candidate.get("evidence_chunks") or []
                if str(value).strip()
            ]
            candidate["evidence_chunks"] = (
                [value for value in cited if not fallback or value in fallback]
                or fallback
            )
            normalized_relations.append(candidate)
        normalized_entities = []
        for item in entities or []:
            if not isinstance(item, dict):
                continue
            candidate = dict(item)
            if not candidate.get("evidence_chunks") and fallback:
                candidate["evidence_chunks"] = fallback
            normalized_entities.append(candidate)
        return {
            "summary": str(value.get("summary") or value.get("description") or ""),
            "entity_type_candidates": normalized_entities,
            "relation_type_candidates": normalized_relations,
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
        *,
        batch_size: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        model = project.get("llm_model") or Config.default_llm_model
        knowledge_language = _schema_knowledge_language(project, samples)
        language_instruction = _schema_language_instruction(
            knowledge_language,
        )
        batches, errors = [], []
        open_observations: list[dict[str, Any]] = []
        observation_keys: set[tuple[str, str, str, str, str]] = set()
        batch_size = max(1, int(
            Config.schema_review_batch_size
            if batch_size is None else batch_size
        ))
        system = (
            "You discover reusable logical knowledge graph types from representative document excerpts. "
            "Propose only reusable semantic types that are missing from the current schema. "
            "Do not model individual names as types. Never remove, rename or redefine existing types. "
            "Interpret every existing relation as its exact semantic contract, not as a broad umbrella. "
            "A generic or review-only relation does not cover a more precise recurring meaning. Compare "
            "the explicit actions and states in the excerpts against semantic_definition, direction and "
            "domain/range metadata. Propose a missing reusable relation when the supplied evidence supports "
            "it; do not return an empty result merely because both endpoints fit existing entity types. "
            "Every relation candidate must explicitly define its complete versioned semantic policy: "
            "semantic definition, domain/range types, direction, symmetry, inverse predicate or null, "
            "distinct subject/object role descriptions, canonical orientation, publish mode, "
            "reflexivity, positive/negative examples, and optional "
            "soft cues. Soft cues are non-binding model hints, never deterministic rules. "
            "conflicts_with may list only known relation identifiers that are competing "
            "semantic interpretations when projected from the exact same source assertion "
            "and ordered endpoint pair. Do not list predicates merely because they are "
            "related, hierarchical, inverse, broader/narrower, or may independently co-occur. "
            "Use an empty list when no such source-binding competitor is established. "
            "Set negation_policy to reject_negated for ordinary positive relations. "
            "Use allow_explicit_negative_relation only when the candidate's own stable "
            "semantic definition denotes absence, exclusion, prohibition, or another "
            "intrinsically negative state and an explicitly negated source construction "
            "directly asserts that same relation. This policy never turns the negation of "
            "an ordinary positive predicate into a positive fact. Set modality_policy to "
            "require_asserted; modal, hypothetical, conditional, quoted, or proposed "
            "claims are not published as asserted facts. "
            "Each relation is binary: source and target must each represent one stable semantic role; "
            "never mix a transferred item and its recipient into one target role. When a clause has "
            "three or more semantic arguments, define one explicit ordered endpoint pair per candidate "
            "and treat every other argument only as evidence context; never describe alternative endpoint "
            "choices inside one policy. Set inverse_predicate "
            "to null unless that inverse identifier already exists in the current schema. For every "
            "relation candidate, evidence_quotes must contain one to three short verbatim excerpts "
            "copied from the supplied text that explicitly support that exact relation meaning. "
            "Never put invented or generalized examples in evidence_quotes. "
            + language_instruction + " "
            "Return only finalized policy text: never include analysis, self-correction, alternative "
            "definitions, or phrases such as 'let us', 'correction', or 'refinement'. "
            "Every candidate must cite evidence_chunks using the exact bracketed "
            "chunk keys supplied. Return empty candidate arrays when the general schema is sufficient."
        )
        current_json = json.dumps(current, ensure_ascii=False)

        def ground_relation_evidence(
            payload: dict[str, Any], batch: list[dict[str, Any]], label: str,
        ) -> dict[str, Any]:
            def normalize_grounding(value: Any) -> tuple[list[str], Any]:
                """Accept bounded transport variants without weakening provenance.

                Some local structured-output models return the requested string
                array as a top-level JSON array. ``OllamaClient`` preserves that
                shape as ``{"items": [...]}``.  Only strings that are later found
                verbatim in the cited frozen chunks can satisfy the deterministic
                evidence gate, so accepting this transport variant does not make
                the semantic acceptance rule more permissive.
                """

                if not isinstance(value, dict):
                    return [], None
                raw_quotes = value.get("evidence_quotes")
                if raw_quotes is None:
                    raw_quotes = value.get("items")
                if isinstance(raw_quotes, str):
                    raw_quotes = [raw_quotes]
                quotes = [
                    str(item).strip()
                    for item in raw_quotes or []
                    if isinstance(item, str) and str(item).strip()
                ][:3]
                return quotes, value.get("confidence")

            sample_by_key = {_chunk_key(item): item for item in batch}
            for index, candidate in enumerate(
                payload.get("relation_type_candidates") or [], start=1,
            ):
                cited = [
                    key for key in candidate.get("evidence_chunks") or []
                    if key in sample_by_key
                ]
                if not cited:
                    continue
                if self._source_backed_example_count(candidate, sample_by_key):
                    if not candidate.get("evidence_quotes"):
                        source_texts = [
                            str(
                                sample_by_key[key].get("_schema_text")
                                or sample_by_key[key].get("content") or ""
                            )
                            for key in cited
                        ]
                        candidate["evidence_quotes"] = [
                            str(example).strip()
                            for example in candidate.get("positive_examples") or []
                            if str(example).strip()
                            and any(
                                str(example).strip() in text
                                for text in source_texts
                            )
                        ][:3]
                    continue
                excerpts = {
                    key: str(
                        sample_by_key[key].get("_schema_text")
                        or sample_by_key[key].get("content") or ""
                    )
                    for key in cited
                }
                try:
                    grounding = self.ollama.json_chat(
                        model=model,
                        system=(
                            "You ground one proposed relation schema candidate in frozen "
                            "source text. You cannot create, rename, broaden, narrow, accept, "
                            "or reject the relation. Copy one to three short verbatim source "
                            "excerpts that explicitly express the candidate's exact semantic "
                            "definition and ordered endpoint roles. A source excerpt may also "
                            "mention an intervening item, event, location, or other context; "
                            "that does not invalidate an explicitly stated ordered relation "
                            "between the candidate endpoint roles. Return an empty array only "
                            "when the source does not explicitly support that exact meaning."
                        ),
                        prompt=json.dumps({
                            "candidate": candidate,
                            "source_excerpts": excerpts,
                        }, ensure_ascii=False),
                        schema=SCHEMA_EVIDENCE_GROUNDING_FORMAT,
                        temperature=0,
                        context_budget_tokens=int(
                            project.get("model_max_tokens") or 32768
                        ),
                        max_output_tokens=1024,
                    )
                    quotes, confidence = normalize_grounding(grounding)
                    candidate["evidence_quotes"] = quotes
                    if confidence is not None:
                        candidate["confidence"] = confidence
                except Exception as exc:
                    message = f"batch {label} candidate {index} grounding: {exc}"
                    errors.append(message)
                    LOGGER.warning("Schema evidence grounding failed: %s", message)
            return payload

        def collect_open_observations(
            observation_value: Any, batch: list[dict[str, Any]],
            *, origin: str = "coverage",
        ) -> int:
            """Persist only exact, affirmative, source-backed observations."""

            raw_observations = (
                observation_value.get("items")
                or observation_value.get("observations")
                or observation_value.get("relations")
                if isinstance(observation_value, Mapping) else []
            ) or []
            sources = {
                _chunk_key(item): str(
                    item.get("_schema_text") or item.get("content") or ""
                )
                for item in batch
            }
            added = 0
            for raw in raw_observations:
                if not isinstance(raw, Mapping):
                    continue
                subject = str(
                    raw.get("subject") or raw.get("subject_surface") or ""
                ).strip()
                relation_expression = str(
                    raw.get("relation_expression") or ""
                ).strip()
                raw_relation_expression = relation_expression
                modality_stripped = _OPEN_RELATION_MODAL_PREFIX.sub(
                    "", relation_expression,
                ).strip()
                relation_expression = _normalize_open_relation_expression(
                    relation_expression,
                )
                object_surface = str(
                    raw.get("object") or raw.get("object_surface") or ""
                ).strip()
                evidence = str(
                    raw.get("evidence") or raw.get("evidence_quote") or ""
                ).strip()
                status = str(
                    raw.get("assertion_status") or ""
                ).strip().upper()
                if status in {
                    "EXPLICIT", "ASSERTED", "POSITIVE", "DIRECT",
                    "FACT", "TRUE",
                }:
                    status = "AFFIRMED"
                if (
                    status != "AFFIRMED" or not subject
                    or not relation_expression or not object_surface
                    or not evidence
                ):
                    continue
                chunk_key = next((
                    key for key, source in sources.items()
                    if evidence in source
                ), "")
                if not chunk_key:
                    continue
                if subject not in evidence or object_surface not in evidence:
                    containing_scopes = [
                        scope_text.strip()
                        for _start, _end, scope_text in split_relation_scopes(
                            sources[chunk_key]
                        )
                        if subject in scope_text and object_surface in scope_text
                    ]
                    containing_scopes = list(dict.fromkeys(
                        value for value in containing_scopes if value
                    ))
                    if len(containing_scopes) != 1:
                        continue
                    evidence = containing_scopes[0]
                key = (
                    chunk_key, subject, relation_expression,
                    object_surface, evidence,
                )
                if key in observation_keys:
                    continue
                observation_keys.add(key)
                observation = {
                    "chunk_key": chunk_key,
                    "subject_surface": subject,
                    "relation_expression": relation_expression,
                    "source_relation_expression": raw_relation_expression,
                    "source_modality": (
                        "NORMATIVE"
                        if modality_stripped != raw_relation_expression
                        else "ASSERTED"
                    ),
                    "object_surface": object_surface,
                    "evidence_quote": evidence,
                    "assertion_status": "AFFIRMED",
                    "observation_origin": origin,
                }
                # Parser-derived observations carry only generic structure
                # metadata.  It informs semantic naming and Schema capability
                # matching; it never selects a concrete logical predicate.
                for metadata_key in (
                    "semantic_naming_required",
                    "structural_evidence_kind",
                    "structural_role_label",
                    "structural_role_contract",
                    "ordered_role_projection",
                ):
                    metadata_value = raw.get(metadata_key)
                    if metadata_value not in (None, "", False):
                        observation[metadata_key] = copy.deepcopy(
                            metadata_value
                        )
                open_observations.append(observation)
                added += 1
            return added

        def collect_structural_observations(
            batch: list[dict[str, Any]],
        ) -> int:
            """Recover explicit table, identifier and hierarchy role frames.

            This stage is deliberately predicate-open.  It reads only source
            syntax and emits exact endpoint/evidence spans as observations;
            the bounded induction and independent policy gates still decide
            whether a reusable logical relation exists.  In particular, a
            column heading is a role description, not a predicate identifier.
            """

            added = 0

            def compact_identifier(surface: str) -> bool:
                return bool(
                    len(surface) <= 48
                    and not re.search(r"\s", surface)
                    and re.search(r"\d", surface)
                    and re.fullmatch(r"[\w.:#§/()\[\]-]+", surface)
                )

            for source_item in batch:
                source = str(
                    source_item.get("_schema_text")
                    or source_item.get("content") or ""
                )
                if not source:
                    continue
                chunk_key = _chunk_key(source_item)
                line_spans: list[tuple[int, int]] = []
                cursor = 0
                for line in source.splitlines(keepends=True):
                    line_spans.append((cursor, cursor + len(line)))
                    cursor += len(line)
                if cursor < len(source):
                    line_spans.append((cursor, len(source)))

                def trimmed_span(start: int, end: int) -> tuple[int, int]:
                    while start < end and source[start].isspace():
                        start += 1
                    while end > start and source[end - 1].isspace():
                        end -= 1
                    return start, end

                def line_cells(
                    start: int, end: int,
                ) -> list[tuple[int, int, str]]:
                    body_end = end
                    while body_end > start and source[body_end - 1] in "\r\n":
                        body_end -= 1
                    body = source[start:body_end]
                    pipes = [
                        index for index, character in enumerate(body)
                        if character == "|"
                    ]
                    if len(pipes) < 2:
                        return []
                    boundaries = list(pipes)
                    if not body.startswith("|"):
                        boundaries.insert(0, -1)
                    if not body.endswith("|"):
                        boundaries.append(len(body))
                    cells: list[tuple[int, int, str]] = []
                    for left, right in zip(
                        boundaries, boundaries[1:], strict=False,
                    ):
                        cell_start, cell_end = trimmed_span(
                            start + left + 1, start + right,
                        )
                        cells.append((
                            cell_start, cell_end,
                            source[cell_start:cell_end],
                        ))
                    return cells

                table_rows: list[
                    tuple[int, int, list[tuple[int, int, str]]]
                ] = []

                def flush_table() -> None:
                    nonlocal table_rows, added
                    semantic_rows = [
                        row for row in table_rows
                        if row[2] and not all(
                            not cell_text or bool(re.fullmatch(
                                r":?-{3,}:?", cell_text,
                            ))
                            for _cell_start, _cell_end, cell_text in row[2]
                        )
                    ]
                    table_rows = []
                    if len(semantic_rows) < 2:
                        return
                    header_start, _header_end, headers = semantic_rows[0]
                    width = len(headers)
                    if width < 2:
                        return
                    structural_items: list[dict[str, Any]] = []
                    for _row_start, row_end, cells in semantic_rows[1:]:
                        if len(cells) != width:
                            continue
                        key_index = next((
                            index for index, (_start, _end, value)
                            in enumerate(cells) if value.strip()
                        ), None)
                        if key_index is None:
                            continue
                        subject = cells[key_index][2].strip()
                        evidence = source[header_start:row_end].strip()
                        for column_index, (_start, _end, value) in enumerate(
                            cells
                        ):
                            if column_index == key_index:
                                continue
                            object_surface = value.strip()
                            header = headers[column_index][2].strip()
                            if (
                                not subject or not object_surface or not header
                                or subject == object_surface
                            ):
                                continue
                            # If the source model already decomposed this cell
                            # into a precise predicate/object pair, a second
                            # row-to-whole-cell role would be less atomic.  A
                            # still-uncovered nominal/reference cell remains a
                            # useful parser-grounded observation.
                            row_text = source[
                                semantic_rows[0][0]:row_end
                            ]
                            already_decomposed = any(
                                str(item.get("chunk_key") or "") == chunk_key
                                and str(item.get("subject_surface") or "")
                                == subject
                                and str(item.get("evidence_quote") or "")
                                in row_text
                                and (
                                    str(item.get("object_surface") or "")
                                    in object_surface
                                    or str(item.get("relation_expression") or "")
                                    in object_surface
                                )
                                for item in open_observations
                            )
                            if already_decomposed:
                                continue
                            structural_items.append({
                                "subject": subject,
                                "relation_expression": (
                                    f"column role: {header}"
                                ),
                                "object": object_surface,
                                "evidence": evidence,
                                "assertion_status": "AFFIRMED",
                                "semantic_naming_required": True,
                                "structural_evidence_kind": (
                                    "table_header_assignment"
                                ),
                                "structural_role_label": header,
                                "structural_role_contract": (
                                    "The table header assigns the row key to "
                                    "this column value. It proves only that "
                                    "ordered role or mapping; it does not by "
                                    "itself prove fulfillment, compliance, "
                                    "causation, identity or ownership."
                                ),
                            })
                    if structural_items:
                        added += collect_open_observations(
                            {"items": structural_items}, [source_item],
                            origin="source_structure",
                        )

                for line_start, line_end in line_spans:
                    cells = line_cells(line_start, line_end)
                    if cells:
                        table_rows.append((line_start, line_end, cells))
                    else:
                        flush_table()
                flush_table()

                # A compact identifier at line start governs the principal
                # actor/topic already recovered from that same line.  Reusing
                # a source-grounded observation avoids guessing a noun phrase.
                identifier_items: list[dict[str, Any]] = []
                for line_start, line_end in line_spans:
                    body_start, body_end = trimmed_span(line_start, line_end)
                    body = source[body_start:body_end]
                    match = re.match(r"([^\s]+)", body)
                    if match is None or not compact_identifier(match.group(1)):
                        continue
                    label = match.group(1)
                    suffix = body[match.end():]
                    candidates: list[tuple[int, str]] = []
                    for observation in open_observations:
                        if str(observation.get("chunk_key") or "") != chunk_key:
                            continue
                        evidence = str(
                            observation.get("evidence_quote") or ""
                        )
                        if not evidence or not (
                            evidence in body or body in evidence
                        ):
                            continue
                        subject = str(
                            observation.get("subject_surface") or ""
                        ).strip()
                        position = suffix.find(subject) if subject else -1
                        if position >= 0 and subject != label:
                            candidates.append((position, subject))
                    if not candidates:
                        continue
                    earliest_position = min(position for position, _ in candidates)
                    earliest = list(dict.fromkeys(
                        subject for position, subject in candidates
                        if position == earliest_position
                    ))
                    if len(earliest) != 1:
                        continue
                    identifier_items.append({
                        "subject": label,
                        "relation_expression": "label binds topic",
                        "object": earliest[0],
                        "evidence": body,
                        "assertion_status": "AFFIRMED",
                        "semantic_naming_required": True,
                        "structural_evidence_kind": (
                            "line_identifier_binding"
                        ),
                        "structural_role_contract": (
                            "The leading identifier scopes or labels this "
                            "clause and its principal actor/topic. It does "
                            "not identify the label and topic as the same "
                            "entity and does not imply authorship."
                        ),
                    })
                if identifier_items:
                    added += collect_open_observations(
                        {"items": identifier_items}, [source_item],
                        origin="source_structure",
                    )

                # A standalone heading followed by compact identifiers is an
                # explicit parent/child structure.  Existing Schema policies
                # that declare this capability can consume it directly.
                hierarchy_items: list[dict[str, Any]] = []
                for parent_index, (parent_start, parent_end) in enumerate(
                    line_spans
                ):
                    clean_start, clean_end = trimmed_span(
                        parent_start, parent_end,
                    )
                    parent = source[clean_start:clean_end].rstrip(":：")
                    if (
                        not parent or len(parent) > 180
                        or "|" in parent or len(parent.splitlines()) != 1
                        or re.search(r"[.!?;。！？；]$", parent)
                        or len(re.findall(
                            r"[A-Za-z0-9]+|[\u3400-\u9fff]", parent,
                        )) > 24
                    ):
                        continue
                    child_index = parent_index + 1
                    started = False
                    while child_index < len(line_spans):
                        child_start, child_end = line_spans[child_index]
                        body_start, body_end = trimmed_span(
                            child_start, child_end,
                        )
                        child_body = source[body_start:body_end]
                        if not child_body:
                            if started:
                                break
                            child_index += 1
                            continue
                        match = re.match(r"([^\s]+)", child_body)
                        if (
                            match is None
                            or not compact_identifier(match.group(1))
                        ):
                            break
                        label = match.group(1)
                        hierarchy_items.append({
                            "subject": label,
                            "relation_expression": "structural child of",
                            "object": parent,
                            "evidence": source[clean_start:body_end],
                            "assertion_status": "AFFIRMED",
                            "semantic_naming_required": True,
                            "structural_evidence_kind": (
                                "section_label_parent_child"
                            ),
                            "structural_role_contract": (
                                "The compact child identifier is directly "
                                "nested under the standalone section label."
                            ),
                        })
                        started = True
                        child_index += 1
                if hierarchy_items:
                    added += collect_open_observations(
                        {"items": hierarchy_items}, [source_item],
                        origin="source_structure",
                    )
            return added

        def prune_duplicate_actor_to_third_party_projection() -> int:
            """Prefer actor→theme plus theme→third-party decomposition.

            An omission-audit model can emit both ``actor→recipient`` and
            ``theme→recipient`` for one transfer clause even after correctly
            recovering the latter.  The frozen observations make that error
            detectable without interpreting a predicate name: the same actor
            already points to a compound theme phrase containing both the
            shorter theme and third-party surfaces.  Remove only the redundant
            actor projection when its correct theme sibling is also present.
            """

            remove_indexes: set[int] = set()

            def first_role_token(item: Mapping[str, Any]) -> str:
                key = _relation_expression_family_key(str(
                    item.get("relation_expression") or ""
                ))
                return key.split("|", 1)[0] if key else ""

            # Recover the canonical theme→third-party projection when the
            # model returned only actor→theme and actor→third-party rows.  The
            # recovery is licensed only by one exact compound object in the
            # same evidence plus one independently repeated minimal theme
            # surface.  No predicate identifier is consulted.
            synthetic: list[dict[str, Any]] = []
            role_particles = {
                "to", "from", "into", "onto", "toward", "towards",
                "through", "via", "for",
            }
            snapshot = list(open_observations)
            for projected_index, projected in enumerate(snapshot):
                projected_key = _relation_expression_family_key(str(
                    projected.get("relation_expression") or ""
                ))
                projected_tokens = projected_key.split("|")
                if (
                    len(projected_tokens) < 2
                    or not any(
                        token in role_particles
                        for token in projected_tokens[1:]
                    )
                ):
                    continue
                actor = str(projected.get("subject_surface") or "")
                third_party = str(projected.get("object_surface") or "")
                evidence = str(projected.get("evidence_quote") or "")
                first_token = projected_tokens[0]
                if not actor or not third_party or not evidence:
                    continue
                compounds = [
                    item for item in snapshot
                    if str(item.get("subject_surface") or "") == actor
                    and str(item.get("evidence_quote") or "") == evidence
                    and first_role_token(item) == first_token
                    and third_party.casefold() in str(
                        item.get("object_surface") or ""
                    ).casefold()
                    and len(str(item.get("object_surface") or ""))
                    > len(third_party)
                ]
                theme_candidates: set[str] = set()
                for compound in compounds:
                    compound_object = str(
                        compound.get("object_surface") or ""
                    )
                    for possible_theme in snapshot:
                        theme = str(
                            possible_theme.get("object_surface") or ""
                        )
                        theme_key = _relation_expression_family_key(str(
                            possible_theme.get("relation_expression") or ""
                        ))
                        if (
                            str(possible_theme.get("subject_surface") or "")
                            == actor
                            and theme
                            and theme.casefold() != third_party.casefold()
                            and theme.casefold() in compound_object.casefold()
                            and len(theme) < len(compound_object)
                            and third_party.casefold()
                            not in theme.casefold()
                            and theme_key == first_token
                        ):
                            theme_candidates.add(theme)
                if len(theme_candidates) != 1:
                    continue
                theme = next(iter(theme_candidates))
                synthetic.append({
                    **projected,
                    "subject_surface": theme,
                    "object_surface": third_party,
                    "ordered_role_projection": "theme_to_third_party",
                    "source_role_pair": (
                        "action_theme->role_complement"
                    ),
                })
                for actor_edge in snapshot:
                    if (
                        str(actor_edge.get("subject_surface") or "") == actor
                        and str(actor_edge.get("evidence_quote") or "") == evidence
                        and str(actor_edge.get("object_surface") or "").casefold()
                        == theme.casefold()
                    ):
                        actor_edge["source_role_pair"] = (
                            "grammatical_subject->action_theme"
                        )
                for index, item in enumerate(snapshot):
                    item_key = _relation_expression_family_key(str(
                        item.get("relation_expression") or ""
                    ))
                    item_tokens = item_key.split("|")
                    if (
                        str(item.get("subject_surface") or "") == actor
                        and str(item.get("evidence_quote") or "") == evidence
                        and item_tokens[:1] == [first_token]
                        and any(
                            token in role_particles
                            for token in item_tokens[1:]
                        )
                    ):
                        remove_indexes.add(index)
                    # The omission pass can also relabel a repeated
                    # actor→theme row with a recipient particle in another
                    # table or sentence. Once the exact compound clause has
                    # licensed theme→third-party, that relabelled duplicate is
                    # not a second ordered role and must not contaminate the
                    # projected family.
                    if (
                        str(item.get("subject_surface") or "") == actor
                        and str(item.get("object_surface") or "").casefold()
                        == theme.casefold()
                        and item_tokens[:1] == [first_token]
                        and any(
                            token in role_particles
                            for token in item_tokens[1:]
                        )
                    ):
                        remove_indexes.add(index)
                for compound in compounds:
                    compound["object_surface"] = theme

            existing_projection_keys = {
                (
                    str(item.get("chunk_key") or ""),
                    str(item.get("subject_surface") or ""),
                    str(item.get("relation_expression") or ""),
                    str(item.get("object_surface") or ""),
                    str(item.get("evidence_quote") or ""),
                )
                for item in open_observations
            }
            for item in synthetic:
                key = (
                    str(item.get("chunk_key") or ""),
                    str(item.get("subject_surface") or ""),
                    str(item.get("relation_expression") or ""),
                    str(item.get("object_surface") or ""),
                    str(item.get("evidence_quote") or ""),
                )
                if key not in existing_projection_keys:
                    open_observations.append(item)
                    existing_projection_keys.add(key)

            for index, candidate in enumerate(open_observations):
                candidate_role = first_role_token(candidate)
                if not candidate_role:
                    continue
                siblings = [
                    item for item in open_observations
                    if item is not candidate
                    and item.get("evidence_quote")
                    == candidate.get("evidence_quote")
                    and item.get("object_surface")
                    == candidate.get("object_surface")
                    and item.get("subject_surface")
                    != candidate.get("subject_surface")
                    and first_role_token(item) == candidate_role
                ]
                if not siblings:
                    continue
                for sibling in siblings:
                    actor = str(candidate.get("subject_surface") or "")
                    theme = str(sibling.get("subject_surface") or "")
                    third_party = str(
                        candidate.get("object_surface") or ""
                    )
                    if not actor or not theme or not third_party:
                        continue
                    supporting = any(
                        item is not candidate
                        and item is not sibling
                        and str(item.get("subject_surface") or "") == actor
                        and item.get("evidence_quote")
                        == candidate.get("evidence_quote")
                        and first_role_token(item) == candidate_role
                        and theme.casefold() in str(
                            item.get("object_surface") or ""
                        ).casefold()
                        and third_party.casefold() in str(
                            item.get("object_surface") or ""
                        ).casefold()
                        for item in open_observations
                    )
                    if supporting:
                        remove_indexes.add(index)
                        break
            if not remove_indexes:
                return 0
            open_observations[:] = [
                item for index, item in enumerate(open_observations)
                if index not in remove_indexes
            ]
            observation_keys.clear()
            observation_keys.update(
                (
                    str(item["chunk_key"]),
                    str(item["subject_surface"]),
                    str(item["relation_expression"]),
                    str(item["object_surface"]),
                    str(item["evidence_quote"]),
                )
                for item in open_observations
            )
            return len(remove_indexes)

        def annotate_direct_ordered_role_projections() -> int:
            """Classify already-minimal multi-participant observations.

            A coverage model may emit the correct theme-to-third-party edge
            directly rather than through the compound repair path.  Detect its
            role from the same frozen actor-to-theme frame and connector.  If
            that actor independently links both participants, retain the more
            general container-participant contract instead.  No domain term or
            logical predicate identifier participates in this distinction.
            """

            connector = re.compile(
                r"^[\s,;:()\[\]-]*(?:to|from|into|onto|toward|towards|"
                r"through|via|with|向|至|从|由|到|经由|通过)\b",
                flags=re.IGNORECASE,
            )
            annotated = 0
            for candidate in open_observations:
                existing_projection = str(
                    candidate.get("ordered_role_projection") or ""
                )
                if existing_projection in {
                    "theme_to_third_party",
                    "container_participant_projection",
                }:
                    candidate["source_role_pair"] = (
                        "action_theme->role_complement"
                    )
                evidence = str(candidate.get("evidence_quote") or "")
                theme = str(candidate.get("subject_surface") or "")
                third_party = str(candidate.get("object_surface") or "")
                if not evidence or not theme or not third_party:
                    continue
                theme_position = evidence.casefold().find(theme.casefold())
                third_position = evidence.casefold().find(
                    third_party.casefold(), theme_position + len(theme),
                )
                if theme_position < 0 or third_position < 0:
                    continue
                between = evidence[
                    theme_position + len(theme):third_position
                ]
                if connector.match(between) is None:
                    continue
                candidate_root = _relation_expression_family_key(str(
                    candidate.get("relation_expression") or ""
                )).split("|")[:1]
                if not candidate_root:
                    continue
                actor_edges = []
                for item in open_observations:
                    if item is candidate or str(
                        item.get("evidence_quote") or ""
                    ) != evidence:
                        continue
                    item_root = _relation_expression_family_key(str(
                        item.get("relation_expression") or ""
                    )).split("|")[:1]
                    broad_object = str(
                        item.get("object_surface") or ""
                    ).casefold()
                    if (
                        item_root == candidate_root
                        and theme.casefold() in broad_object
                        and third_party.casefold() not in broad_object
                        and str(item.get("subject_surface") or "")
                        not in {theme, third_party}
                    ):
                        actor_edges.append(item)
                actors = list(dict.fromkeys(
                    str(item.get("subject_surface") or "")
                    for item in actor_edges
                    if str(item.get("subject_surface") or "")
                ))
                if len(actors) != 1:
                    continue
                for actor_edge in actor_edges:
                    actor_edge["source_role_pair"] = (
                        "grammatical_subject->action_theme"
                    )
                actor = actors[0]
                actor_third_party_edges = [
                    item for item in open_observations
                    if item is not candidate
                    and str(item.get("evidence_quote") or "") == evidence
                    and str(item.get("subject_surface") or "") == actor
                    and third_party.casefold() in str(
                        item.get("object_surface") or ""
                    ).casefold()
                    and theme.casefold() not in str(
                        item.get("object_surface") or ""
                    ).casefold()
                ]
                actor_also_links_third_party = bool(
                    actor_third_party_edges
                )
                for actor_edge in actor_third_party_edges:
                    actor_edge["source_role_pair"] = (
                        "grammatical_subject->role_complement"
                    )
                candidate["semantic_naming_required"] = True
                candidate["source_role_pair"] = (
                    "action_theme->role_complement"
                )
                if actor_also_links_third_party:
                    candidate["ordered_role_projection"] = (
                        "container_participant_projection"
                    )
                    candidate["structural_role_contract"] = (
                        "One source container/action independently links both "
                        "participants and explicitly connects this ordered pair."
                    )
                else:
                    candidate["ordered_role_projection"] = (
                        "theme_to_third_party"
                    )
                    candidate["structural_role_contract"] = (
                        "The same source frame explicitly directs the acted-on "
                        "theme to this recipient, source or destination."
                    )
                annotated += 1
            return annotated

        def discover_open_batch(
            batch: list[dict[str, Any]], label: str,
        ) -> None:
            """Run predicate-open coverage independently of broad review.

            The broad schema transport is intentionally rich and can fail on
            a local model without making the smaller source-observation
            contract unusable. Keeping these calls independent prevents one
            malformed broad response from erasing every precise relation
            family in the document.
            """

            for coverage_round in range(
                Config.schema_relation_coverage_rounds
            ):
                prior = [
                    item for item in open_observations
                    if item["chunk_key"] in {
                        _chunk_key(source) for source in batch
                    }
                ]
                try:
                    omission_audit = (
                        " This is the bounded omission-audit round. Audit each "
                        "previous observation rather than paraphrasing it. If "
                        "one source clause states actor, acted-on theme and an "
                        "explicit recipient/source/destination, an existing "
                        "actor-to-theme item is incomplete: additionally emit "
                        "the theme-to-third-party projection, with the theme "
                        "and third party copied as separate minimal source "
                        "spans. Name that projection from the theme's ordered "
                        "role (for example, 'sent to'), never with an ellipsis "
                        "placeholder and never as actor-to-third-party. If an "
                        "identifier or label prefixes a clause, "
                        "emit its binding to that clause's principal actor or "
                        "topic. For tables, check every row-key/header role and "
                        "do not treat a previously emitted verbal cell as "
                        "covering an identifier/reference/category cell. If a "
                        "map or container connects two named participants, "
                        "emit both container-to-participant projections and "
                        "the explicitly stated participant-to-participant "
                        "projection. Return only genuinely missing items."
                        if coverage_round else ""
                    )
                    observation_value = self.ollama.json_chat(
                        model=model,
                        system=(
                            "Read the frozen source unit by unit and record every explicit binary "
                            "semantic observation, enumerating every directly stated ordered role. "
                            "Each item must use exactly subject, relation_expression, object, "
                            "evidence and assertion_status. The subject, object and evidence must "
                            "be verbatim source spans. Use a short reusable base relation expression "
                            "with no endpoint names; omit tense and must/shall auxiliaries while "
                            "preserving the underlying normative role. assertion_status must be "
                            "AFFIRMED for a directly stated positive fact or obligation, NEGATED "
                            "only for a denied positive fact, and UNCERTAIN only for genuine "
                            "possibility or uncertainty. Never turn negation or uncertainty into a "
                            "positive observation. Split coordination and nested clauses into all "
                            "independently entailed ordered binary roles. For transfer, communication "
                            "and other multi-participant actions, emit actor-to-theme plus every "
                            "explicit theme-to-recipient, theme-to-destination and theme-to-source "
                            "projection; never replace the acted-on theme with a recipient merely "
                            "because both occur in one clause. In a table, interpret verbal cells and "
                            "every identifier/reference/category cell through its row key and column "
                            "heading. A label immediately prefixing a labeled, normative or list "
                            "clause explicitly binds it to its principal actor/topic. When a map or "
                            "container explicitly connects X to Y, record the stated X-to-Y connection "
                            "as well as explicit links from the map/container. Preserve hierarchy, "
                            "membership, source, purpose, location, property and complement roles when "
                            "directly stated. Do not repeat a previously recorded observation; on later "
                            "rounds inspect only for omissions. Do not map to a predefined predicate, "
                            "infer missing facts, or treat co-occurrence/adjacency as a relation. This "
                            "stage records open observations only and cannot publish a fact. Return "
                            "strict JSON."
                            + omission_audit
                        ),
                        prompt=(
                            "Representative excerpts:\n"
                            + self._format_samples(batch)
                            + "\n\nPreviously recorded observations:\n"
                            + json.dumps(prior, ensure_ascii=False)
                            + f"\n\nCoverage round: {coverage_round + 1}"
                        ),
                        schema=OPEN_RELATION_OBSERVATION_FORMAT,
                        temperature=0,
                        context_budget_tokens=int(
                            project.get("model_max_tokens") or 32768
                        ),
                        max_output_tokens=min(
                            Config.schema_review_max_output_tokens, 4096,
                        ),
                    )
                    collect_open_observations(
                        observation_value, batch, origin="coverage",
                    )
                    prune_duplicate_actor_to_third_party_projection()
                    annotate_direct_ordered_role_projections()
                except Exception as exc:
                    message = (
                        f"batch {label} open observation round "
                        f"{coverage_round + 1}: {exc}"
                    )
                    errors.append(message)
                    LOGGER.warning(
                        "Schema open observation failed without invalidating "
                        "the broad review: %s", message,
                    )

            collect_structural_observations(batch)
            prune_duplicate_actor_to_third_party_projection()
            annotate_direct_ordered_role_projections()

            if not Config.schema_relation_compound_audit_enabled:
                return
            batch_keys = {_chunk_key(item) for item in batch}
            connector_pattern = re.compile(
                r"\b(?:to|from|into|onto|toward|towards|through|via|between)\b"
                r"|[向至从由到]",
                flags=re.IGNORECASE,
            )
            unresolved_compounds = [
                item for item in open_observations
                if item.get("chunk_key") in batch_keys
                and connector_pattern.search(str(
                    item.get("object_surface") or ""
                ))
                and len(re.findall(
                    r"[A-Za-z0-9]+|[\u3400-\u9fff]",
                    str(item.get("object_surface") or ""),
                )) >= 3
            ]
            if not unresolved_compounds:
                return
            try:
                compound_value = self.ollama.json_chat(
                    model=model,
                    system=(
                        "Read only the supplied unresolved compound-endpoint "
                        "observations and their frozen source. Decompose each "
                        "one into every directly entailed ordered binary role "
                        "whose subject and object are separate minimal verbatim "
                        "source spans. A compound object containing a theme plus "
                        "recipient/source/destination must yield the theme-to-"
                        "third-party role. When a map, index, relation or "
                        "container explicitly connects X with/to Y, emit its "
                        "separate links to X and Y plus the explicitly stated "
                        "X-to-Y role. For a nested complement, emit both the "
                        "outer actor-to-theme role and the inner theme-to-"
                        "complement role. Never return the original compound "
                        "phrase as one endpoint, never infer from adjacency, and "
                        "never repeat a previously valid minimal observation. "
                        "Each item must use subject, relation_expression, object, "
                        "evidence and assertion_status=AFFIRMED. Return strict JSON."
                    ),
                    prompt=json.dumps({
                        "frozen_source": self._format_samples(batch),
                        "unresolved_compound_observations": (
                            unresolved_compounds
                        ),
                        "already_recorded_observations": [
                            item for item in open_observations
                            if item.get("chunk_key") in batch_keys
                            and item not in unresolved_compounds
                        ],
                    }, ensure_ascii=False),
                    schema=OPEN_RELATION_OBSERVATION_FORMAT,
                    temperature=0,
                    context_budget_tokens=int(
                        project.get("model_max_tokens") or 32768
                    ),
                    max_output_tokens=min(
                        Config.schema_review_max_output_tokens, 3072,
                    ),
                )
                collect_open_observations(
                    compound_value, batch, origin="compound_audit",
                )
                prune_duplicate_actor_to_third_party_projection()
                annotate_direct_ordered_role_projections()

                # The focused audit can restate one active source edge as an
                # automatically voiced inverse. Keep the orientation whose
                # subject is the original compound observation's subject;
                # inverse predicates require their own explicit source role,
                # not merely a grammatical rewrite by this audit.
                original_subjects = {
                    (
                        str(item.get("evidence_quote") or ""),
                        str(item.get("subject_surface") or ""),
                    )
                    for item in unresolved_compounds
                }
                inverse_audit_ids: set[int] = set()
                audit_items = [
                    item for item in open_observations
                    if item.get("observation_origin") == "compound_audit"
                ]
                for item in audit_items:
                    evidence = str(item.get("evidence_quote") or "")
                    source = str(item.get("subject_surface") or "")
                    target = str(item.get("object_surface") or "")
                    if (evidence, target) not in original_subjects:
                        continue
                    if any(
                        other is not item
                        and str(other.get("evidence_quote") or "") == evidence
                        and str(other.get("subject_surface") or "") == target
                        and str(other.get("object_surface") or "") == source
                        for other in audit_items
                    ):
                        inverse_audit_ids.add(id(item))
                if inverse_audit_ids:
                    open_observations[:] = [
                        item for item in open_observations
                        if id(item) not in inverse_audit_ids
                    ]

                normalized_audit: dict[
                    tuple[str, str, str], tuple[tuple[int, int], dict[str, Any]]
                ] = {}
                for item in open_observations:
                    if item.get("observation_origin") != "compound_audit":
                        continue
                    evidence = str(item.get("evidence_quote") or "")
                    source = str(item.get("subject_surface") or "")
                    target = str(item.get("object_surface") or "")
                    matching_compounds = [
                        compound for compound in unresolved_compounds
                        if str(compound.get("evidence_quote") or "") == evidence
                        and (
                            source == str(
                                compound.get("subject_surface") or ""
                            )
                            or source.casefold() in str(
                                compound.get("object_surface") or ""
                            ).casefold()
                        )
                        and (
                            target == str(
                                compound.get("subject_surface") or ""
                            )
                            or target.casefold() in str(
                                compound.get("object_surface") or ""
                            ).casefold()
                        )
                    ]
                    if len(matching_compounds) != 1:
                        continue
                    compound = matching_compounds[0]
                    original_subject = str(
                        compound.get("subject_surface") or ""
                    )
                    broad_object = str(
                        compound.get("object_surface") or ""
                    ).casefold()
                    if target == original_subject and source != original_subject:
                        continue
                    if source == original_subject and target != original_subject:
                        target_position = broad_object.find(
                            target.casefold()
                        )
                        connector_match = re.search(
                            r"\b(?:to|from|into|onto|toward|towards|through|"
                            r"via|with|between)\b|[向至从由到]",
                            broad_object,
                            flags=re.IGNORECASE,
                        )
                        if target_position >= 0 and connector_match is not None:
                            item["source_role_pair"] = (
                                "grammatical_subject->action_theme"
                                if target_position < connector_match.start()
                                else "grammatical_subject->role_complement"
                            )
                    if source != original_subject and target != original_subject:
                        source_position = broad_object.find(source.casefold())
                        target_position = broad_object.find(target.casefold())
                        if (
                            source_position < 0 or target_position < 0
                            or source_position >= target_position
                        ):
                            continue
                        # This is either an actor/theme/third-party projection
                        # or the participant pair licensed by a container/map
                        # clause.  Distinguish them from the already recorded
                        # actor-to-theme edge, never from domain vocabulary.
                        original_relation_key = (
                            _relation_expression_family_key(str(
                                compound.get("relation_expression") or ""
                            ))
                        )
                        actor_to_source = any(
                            other is not item
                            and other.get("observation_origin")
                            != "compound_audit"
                            and str(other.get("evidence_quote") or "")
                            == evidence
                            and str(other.get("subject_surface") or "")
                            == original_subject
                            and str(other.get("object_surface") or "").casefold()
                            == source.casefold()
                            and _relation_expression_family_key(str(
                                other.get("relation_expression") or ""
                            )) == original_relation_key
                            for other in open_observations
                        )
                        item["semantic_naming_required"] = True
                        if actor_to_source:
                            item["ordered_role_projection"] = (
                                "theme_to_third_party"
                            )
                            item["structural_role_contract"] = (
                                "The source explicitly states an actor-to-theme "
                                "action and directs that theme to this third "
                                "party. Preserve theme-to-third-party order."
                            )
                        else:
                            item["ordered_role_projection"] = (
                                "container_participant_projection"
                            )
                            item["structural_role_contract"] = (
                                "The source explicitly states that a container "
                                "or map connects the ordered participant pair. "
                                "Name that participant relation without "
                                "reversing it or claiming identity, containment "
                                "or ownership."
                            )
                    relation_key = _relation_expression_family_key(str(
                        item.get("relation_expression") or ""
                    ))
                    original_key = _relation_expression_family_key(str(
                        compound.get("relation_expression") or ""
                    ))
                    rank = (
                        0 if relation_key == original_key else 1,
                        len(str(item.get("relation_expression") or "")),
                    )
                    pair_key = (evidence, source.casefold(), target.casefold())
                    previous = normalized_audit.get(pair_key)
                    if previous is None or rank < previous[0]:
                        normalized_audit[pair_key] = (rank, item)
                selected_audit_ids = {
                    id(item) for _rank, item in normalized_audit.values()
                }
                open_observations[:] = [
                    item for item in open_observations
                    if item.get("observation_origin") != "compound_audit"
                    or id(item) in selected_audit_ids
                ]

                resolved_ids: set[int] = set()
                for compound in unresolved_compounds:
                    broad_object = str(
                        compound.get("object_surface") or ""
                    ).casefold()
                    evidence = str(compound.get("evidence_quote") or "")
                    narrower_surfaces = {
                        surface.casefold()
                        for item in open_observations
                        if item is not compound
                        and str(item.get("evidence_quote") or "") == evidence
                        for surface in (
                            str(item.get("subject_surface") or ""),
                            str(item.get("object_surface") or ""),
                        )
                        if surface
                        and surface.casefold() in broad_object
                        and len(surface) < len(broad_object)
                    }
                    if len(narrower_surfaces) >= 2:
                        resolved_ids.add(id(compound))
                if resolved_ids:
                    open_observations[:] = [
                        item for item in open_observations
                        if id(item) not in resolved_ids
                    ]
                    observation_keys.clear()
                    observation_keys.update(
                        (
                            str(item["chunk_key"]),
                            str(item["subject_surface"]),
                            str(item["relation_expression"]),
                            str(item["object_surface"]),
                            str(item["evidence_quote"]),
                        )
                        for item in open_observations
                    )
                annotate_direct_ordered_role_projections()
            except Exception as exc:
                message = f"batch {label} compound endpoint audit: {exc}"
                errors.append(message)
                LOGGER.warning(
                    "Schema compound endpoint audit failed without erasing "
                    "other observations: %s", message,
                )

        def discover_batch(
            batch: list[dict[str, Any]], label: str, depth: int = 0,
        ) -> None:
            try:
                value = self.ollama.json_chat(
                    model=model,
                    system=system,
                    prompt=f"Current schema:\n{current_json}\n\nRepresentative excerpts:\n{self._format_samples(batch)}",
                    schema=SCHEMA_REVIEW_FORMAT,
                    temperature=0,
                    context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                    max_output_tokens=Config.schema_review_max_output_tokens,
                )
                payload = self._candidate_payload(
                    value,
                    fallback_evidence_chunks=[_chunk_key(item) for item in batch],
                )
                for candidate in payload.get(
                    "relation_type_candidates"
                ) or []:
                    candidate["_discovery_origin"] = "broad_review"
                batches.append(ground_relation_evidence(payload, batch, label))
            except Exception as exc:
                recoverable = (
                    isinstance(exc, OllamaProtocolError)
                    and (
                        "truncated by output length" in str(exc)
                        or "prompt token estimate underestimated" in str(exc)
                    )
                )
                if recoverable and len(batch) > 1 and depth < 8:
                    midpoint = max(1, len(batch) // 2)
                    discover_batch(batch[:midpoint], f"{label}.1", depth + 1)
                    discover_batch(batch[midpoint:], f"{label}.2", depth + 1)
                    return
                message = f"batch {label}: {exc}"
                errors.append(message)
                LOGGER.warning("Schema candidate discovery failed: %s", message)
            discover_open_batch(batch, label)

        for start in range(0, len(samples), batch_size):
            discover_batch(
                samples[start:start + batch_size],
                str(start // batch_size + 1),
            )
        if open_observations:
            grouped_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for observation in open_observations:
                expression_key = _relation_expression_family_key(
                    str(observation["relation_expression"]),
                )
                if expression_key:
                    grouped_observations[expression_key].append(observation)
            minimum_observation_count = (
                1 if Config.schema_single_observation_relation_induction else
                max(2, Config.schema_review_min_evidence)
            )
            active_relation_names = {
                str(item.get("name") or "")
                for item in current.get("relation_types") or []
                if isinstance(item, Mapping)
            }
            active_structural_kinds = {
                str(kind)
                for item in current.get("relation_types") or []
                if isinstance(item, Mapping)
                for kind in item.get("structural_evidence_kinds") or []
                if str(kind)
            }

            def group_needs_induction(
                observations: list[dict[str, Any]],
            ) -> bool:
                if not observations:
                    return False
                if (
                    len(observations) < max(
                        2, Config.schema_review_min_evidence,
                    )
                    and _is_bare_open_relation_carrier(str(
                        observations[0].get("relation_expression") or ""
                    ))
                ):
                    # One copular/auxiliary observation cannot establish
                    # whether the reusable role is identity, classification,
                    # state or property attribution.  Repetition may justify
                    # bounded induction; a singleton stays fail-closed.
                    return False
                structural_kinds = {
                    str(item.get("structural_evidence_kind") or "")
                    for item in observations
                    if str(item.get("structural_evidence_kind") or "")
                }
                if (
                    structural_kinds
                    and all(
                        kind in active_structural_kinds
                        for kind in structural_kinds
                    )
                ):
                    return False
                requires_semantic_naming = any(
                    item.get("semantic_naming_required") is True
                    for item in observations
                )
                if not requires_semantic_naming:
                    source_identifier = _source_relation_identifier(str(
                        observations[0].get("relation_expression") or ""
                    ))
                    if source_identifier in active_relation_names:
                        return False
                return True

            recurring_groups = sorted(
                (
                    values for values in grouped_observations.values()
                    if len(values) >= minimum_observation_count
                    and group_needs_induction(values)
                ),
                key=lambda values: (
                    -len(values),
                    -sum(
                        1 for item in values
                        if item.get("observation_origin") == "source_structure"
                    ),
                    values[0]["relation_expression"],
                ),
            )[: max(1, Config.schema_max_relation_additions)]

            relation_groups: list[dict[str, Any]] = []
            observations_by_group: dict[str, list[dict[str, Any]]] = {}
            for group_index, observations in enumerate(
                recurring_groups, start=1,
            ):
                group_id = f"G{group_index:04d}"
                observations_by_group[group_id] = observations
                relation_groups.append({
                    "observation_group_id": group_id,
                    "relation_expression": str(
                        observations[0]["relation_expression"]
                    ),
                    "observations": observations,
                })

            compiled_summaries: list[dict[str, Any]] = []

            def align_induced_candidate_groups(
                candidates: list[dict[str, Any]],
                group_records: list[dict[str, Any]],
            ) -> dict[int, str]:
                """Bind batch-compiled policies back to one frozen role group."""

                if not candidates or not group_records:
                    return {}
                candidate_indexes = list(range(len(candidates)))
                group_ids = [
                    str(item["observation_group_id"])
                    for item in group_records
                ]
                response_schema = {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "bindings": {
                            "type": "array",
                            "maxItems": min(
                                len(candidate_indexes), len(group_ids),
                            ),
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "candidate_index": {
                                        "type": "integer",
                                        "enum": candidate_indexes,
                                    },
                                    "observation_group_id": {
                                        "type": "string", "enum": group_ids,
                                    },
                                },
                                "required": [
                                    "candidate_index",
                                    "observation_group_id",
                                ],
                            },
                        },
                    },
                    "required": ["bindings"],
                }
                last_error: Exception | None = None
                for attempt in range(2):
                    try:
                        value = self.ollama.json_chat(
                            model=model,
                            system=(
                                "You align already-compiled binary relation policies to frozen "
                                "open-observation groups. Judge exact relation meaning, ordered "
                                "subject role and ordered object role. Each candidate can bind to "
                                "at most one group and each group to at most one candidate. Omit a "
                                "candidate when it merges roles, changes direction, is merely a "
                                "generic catch-all, or has no exact group. You cannot create or "
                                "modify a policy. Return only bindings."
                                + (
                                    " Protocol retry: return exactly one bindings array using only "
                                    "the supplied candidate indexes and group IDs."
                                    if attempt else ""
                                )
                            ),
                            prompt=json.dumps({
                                "candidates": [{
                                    "candidate_index": index,
                                    "name": str(item.get("name") or ""),
                                    "semantic_definition": str(
                                        item.get("semantic_definition") or ""
                                    ),
                                    "subject_role_description": str(
                                        item.get("subject_role_description") or ""
                                    ),
                                    "object_role_description": str(
                                        item.get("object_role_description") or ""
                                    ),
                                    "evidence_quotes": list(
                                        item.get("evidence_quotes") or []
                                    ),
                                } for index, item in enumerate(candidates)],
                                "observation_groups": group_records,
                            }, ensure_ascii=False),
                            schema=response_schema,
                            temperature=0,
                            context_budget_tokens=int(
                                project.get("model_max_tokens") or 32768
                            ),
                            max_output_tokens=2048,
                        )
                        raw_bindings = (
                            value.get("bindings")
                            if isinstance(value, Mapping) else None
                        )
                        if not isinstance(raw_bindings, list):
                            raise OllamaProtocolError(
                                "candidate-group alignment requires bindings"
                            )
                        aligned: dict[int, str] = {}
                        used_groups: set[str] = set()
                        for item in raw_bindings:
                            if not isinstance(item, Mapping):
                                continue
                            candidate_index = item.get("candidate_index")
                            group_id = str(
                                item.get("observation_group_id") or ""
                            )
                            if (
                                not isinstance(candidate_index, int)
                                or isinstance(candidate_index, bool)
                                or candidate_index not in candidate_indexes
                                or group_id not in group_ids
                                or candidate_index in aligned
                                or group_id in used_groups
                            ):
                                continue
                            aligned[candidate_index] = group_id
                            used_groups.add(group_id)
                        return aligned
                    except Exception as exc:
                        last_error = exc
                raise OllamaProtocolError(
                    "candidate-group alignment failed: "
                    f"{last_error}"
                )

            def append_induced_batch(
                induced_value: Any,
                group_records: list[dict[str, Any]],
                label: str,
            ) -> None:
                active_group_ids = {
                    str(item["observation_group_id"])
                    for item in group_records
                }
                active_observations = [
                    observation
                    for group_id in active_group_ids
                    for observation in observations_by_group[group_id]
                ]
                group_keys = list(dict.fromkeys(
                    str(item["chunk_key"]) for item in active_observations
                ))
                induced = {
                    "summary": str(
                        induced_value.get("summary") or ""
                    ) if isinstance(induced_value, Mapping) else "",
                    "entity_type_candidates": [],
                    "relation_type_candidates": [],
                }
                raw_candidates: list[dict[str, Any]] = []
                if isinstance(induced_value, Mapping):
                    for root_key in (
                        "relation_type_candidates", "relation_types",
                        "relation_candidates", "candidate_relations",
                        "candidate_families", "families", "candidates",
                        "rows",
                    ):
                        root_items = induced_value.get(root_key)
                        if not isinstance(root_items, list):
                            continue
                        for raw_item in root_items:
                            if not isinstance(raw_item, Mapping):
                                continue
                            nested = (
                                raw_item.get("relation_contract")
                                or raw_item.get("policy")
                                or raw_item.get("candidate")
                            )
                            merged = (
                                dict(nested)
                                if isinstance(nested, Mapping)
                                else dict(raw_item)
                            )
                            for provenance_key in (
                                "observation_group_id",
                                "observation_group_ids",
                                "observation_groups", "group_ids",
                            ):
                                if (
                                    provenance_key not in merged
                                    and provenance_key in raw_item
                                ):
                                    merged[provenance_key] = copy.deepcopy(
                                        raw_item[provenance_key]
                                    )
                            normalized = self._candidate_payload(
                                {"relation_type_candidates": [merged]},
                                fallback_evidence_chunks=group_keys,
                            ).get("relation_type_candidates") or []
                            raw_candidates.extend(
                                dict(item) for item in normalized
                                if isinstance(item, Mapping)
                            )
                        if raw_candidates:
                            break
                    if (
                        not raw_candidates
                        and isinstance(
                            induced_value.get("induced_relation_family"),
                            Mapping,
                        )
                    ):
                        raw_candidates = list(self._candidate_payload(
                            {"relation_type_candidates": [dict(
                                induced_value["induced_relation_family"]
                            )]},
                            fallback_evidence_chunks=group_keys,
                        ).get("relation_type_candidates") or [])
                if group_records and not raw_candidates:
                    raise OllamaProtocolError(
                        "batch induction returned no compact naming rows"
                    )

                candidate_rows: list[tuple[dict[str, Any], list[str]]] = []
                compact_policy_fields = {
                    *SCHEMA_REVIEW_FORMAT["properties"]
                    ["relation_type_candidates"]["items"]
                    ["properties"].keys(),
                    "observation_group_ids", "observation_group_id",
                    "observation_groups", "group_ids",
                }
                for raw_candidate in raw_candidates:
                    if not isinstance(raw_candidate, Mapping):
                        continue
                    candidate = {
                        key: copy.deepcopy(value)
                        for key, value in raw_candidate.items()
                        if key in compact_policy_fields
                    }
                    raw_group_ids = (
                        candidate.get("observation_group_ids")
                        or candidate.get("observation_group_id")
                        or candidate.get("observation_groups")
                        or candidate.get("group_ids")
                        or []
                    )
                    if isinstance(raw_group_ids, str):
                        raw_group_ids = [raw_group_ids]
                    group_ids = list(dict.fromkeys(
                        str(value) for value in raw_group_ids
                        if str(value) in active_group_ids
                    ))
                    if not group_ids:
                        exact_quotes = {
                            str(value).strip()
                            for value in candidate.get("evidence_quotes") or []
                            if str(value).strip()
                        }
                        matching_groups = [
                            group_id for group_id in sorted(active_group_ids)
                            if any(
                                str(item["evidence_quote"]) in exact_quotes
                                for item in observations_by_group[group_id]
                            )
                        ]
                        # One exact group can safely recover an omitted
                        # transport ID. Ambiguous evidence shared by several
                        # ordered projections remains fail-closed.
                        if len(matching_groups) == 1:
                            group_ids = matching_groups
                    candidate_rows.append((candidate, group_ids))

                used_explicit_groups = {
                    group_ids[0]
                    for _candidate, group_ids in candidate_rows
                    if len(group_ids) == 1
                }
                remaining_group_records = [
                    item for item in group_records
                    if str(item["observation_group_id"])
                    not in used_explicit_groups
                ]
                # A compact model may omit the transport ID while preserving
                # the source-stable relation identifier. Bind only an exact,
                # unique match before considering a model alignment fallback.
                for row_index, (candidate, group_ids) in enumerate(
                    candidate_rows
                ):
                    if len(group_ids) == 1:
                        continue
                    candidate_name = _type_key(
                        str(candidate.get("name") or "")
                    )
                    matches = [
                        item for item in remaining_group_records
                        if candidate_name
                        and candidate_name == _type_key(
                            _source_relation_identifier(str(
                                item.get("relation_expression") or ""
                            ))
                        )
                    ]
                    if len(matches) != 1:
                        continue
                    group_id = str(matches[0]["observation_group_id"])
                    candidate_rows[row_index] = (candidate, [group_id])
                    remaining_group_records = [
                        item for item in remaining_group_records
                        if str(item["observation_group_id"]) != group_id
                    ]

                unresolved_indexes = [
                    index for index, (_candidate, group_ids)
                    in enumerate(candidate_rows) if len(group_ids) != 1
                ]
                if (
                    unresolved_indexes
                    and len(candidate_rows) == len(group_records)
                    and len(unresolved_indexes) == len(remaining_group_records)
                ):
                    # The compact contract asks for exactly one row per input
                    # group in input order. Positional recovery is safe only
                    # when cardinality is complete and all explicit bindings
                    # leave an equally sized ordered remainder.
                    for row_index, group_record in zip(
                        unresolved_indexes, remaining_group_records,
                        strict=True,
                    ):
                        candidate, _group_ids = candidate_rows[row_index]
                        candidate_rows[row_index] = (
                            candidate,
                            [str(group_record["observation_group_id"])],
                        )

                unresolved = [
                    candidate
                    for candidate, group_ids in candidate_rows
                    if len(group_ids) != 1
                ]
                if unresolved:
                    aligned = align_induced_candidate_groups(
                        unresolved, group_records,
                    )
                    alignment_cursor = 0
                    repaired_rows: list[
                        tuple[dict[str, Any], list[str]]
                    ] = []
                    for candidate, group_ids in candidate_rows:
                        if len(group_ids) == 1:
                            repaired_rows.append((candidate, group_ids))
                            continue
                        aligned_group = aligned.get(alignment_cursor)
                        alignment_cursor += 1
                        repaired_rows.append((
                            candidate,
                            [aligned_group] if aligned_group else [],
                        ))
                    candidate_rows = repaired_rows

                compiled: list[dict[str, Any]] = []
                used_group_ids: set[str] = set()
                for candidate, group_ids in candidate_rows:
                    if len(group_ids) != 1 or group_ids[0] in used_group_ids:
                        continue
                    used_group_ids.add(group_ids[0])
                    group_record = next(
                        item for item in group_records
                        if str(item["observation_group_id"]) == group_ids[0]
                    )
                    candidate_observations = [
                        item
                        for group_id in group_ids
                        for item in observations_by_group[group_id]
                    ]
                    for key in (
                        "observation_group_ids", "observation_groups",
                        "observation_group_id", "group_ids",
                    ):
                        candidate.pop(key, None)
                    candidate["_induction_observations"] = copy.deepcopy(
                        candidate_observations
                    )
                    candidate["source_role_pairs"] = sorted({
                        str(item.get("source_role_pair") or "")
                        for item in candidate_observations
                        if str(item.get("source_role_pair") or "") in {
                            "grammatical_subject->action_theme",
                            "action_theme->role_complement",
                            "grammatical_subject->role_complement",
                            "grammatical_subject->direct_argument",
                            "direct_argument->grammatical_subject",
                            "action_theme->grammatical_subject",
                            "role_complement->grammatical_subject",
                            "collective_participant->collective_participant",
                            "possessor->possessed",
                            "possessed->possessor",
                        }
                    })
                    candidate["evidence_chunks"] = list(dict.fromkeys(
                        str(item["chunk_key"])
                        for item in candidate_observations
                    ))
                    candidate["evidence_quotes"] = list(dict.fromkeys(
                        str(item["evidence_quote"])
                        for item in candidate_observations
                    ))[:3]
                    source_identifier = _source_relation_identifier(str(
                        group_record.get("relation_expression") or ""
                    ))
                    has_reprojected_role = any(
                        str(item.get("ordered_role_projection") or "")
                        for item in candidate_observations
                    )
                    requires_semantic_naming = any(
                        item.get("semantic_naming_required") is True
                        for item in candidate_observations
                    )
                    if has_reprojected_role or requires_semantic_naming:
                        candidate["name"] = (
                            _strip_endpoint_suffix_from_identifier(
                                str(candidate.get("name") or ""),
                                candidate_observations,
                                tuple(
                                    str(value)
                                    for field in (
                                        "source_types", "target_types",
                                        "domain_types", "range_types",
                                    )
                                    for value in candidate.get(field) or []
                                ),
                            )
                        )
                    if (
                        source_identifier
                        and not has_reprojected_role
                        and not requires_semantic_naming
                    ):
                        suggested_name = str(candidate.get("name") or "")
                        if suggested_name and suggested_name != source_identifier:
                            candidate["_model_suggested_name"] = suggested_name
                        candidate["name"] = source_identifier
                    compiled.append(candidate)

                name_counts = Counter(
                    _type_key(str(item.get("name") or ""))
                    for item in compiled
                    if _valid_identifier(str(item.get("name") or ""))
                )
                compiled = [
                    item for item in compiled
                    if name_counts.get(
                        _type_key(str(item.get("name") or "")), 0,
                    ) == 1
                ]
                if raw_candidates and not compiled:
                    raise OllamaProtocolError(
                        "batch induction candidates lacked unambiguous "
                        "observation-group provenance"
                    )
                if not compiled:
                    return
                induced["entity_type_candidates"] = []
                induced["relation_type_candidates"] = compiled
                batches.append(induced)
                compiled_summaries.extend({
                    "name": str(item.get("name") or ""),
                    "semantic_definition": str(
                        item.get("semantic_definition") or ""
                    ),
                    "subject_role_description": str(
                        item.get("subject_role_description") or ""
                    ),
                    "object_role_description": str(
                        item.get("object_role_description") or ""
                    ),
                } for item in compiled)

            def induce_relation_groups(
                group_records: list[dict[str, Any]],
                label: str,
                depth: int = 0,
            ) -> None:
                last_error: Exception | None = None
                for attempt in range(2):
                    try:
                        induced_value = self.ollama.json_chat(
                            model=model,
                            system=(
                                "Induce reusable logical knowledge graph relation families by "
                                "assigning one compact reusable relation-family name to every supplied "
                                "validated open-observation group. Return exactly one row per group "
                                "in input order and copy its observation_group_id exactly; do not omit "
                                "a row merely because its name already exists. Prefer the normalized "
                                "source relation expression as the short English identifier when it is "
                                "already reusable and role-specific. Do not replace a precise source "
                                "action with a broader synonym or add endpoint/domain words to the "
                                "identifier. Groups with different ordered subject/object roles must "
                                "remain separate, including actor-to-theme versus theme-to-recipient, "
                                "destination or source projections. For an observation marked "
                                "ordered_role_projection=theme_to_third_party, name the relation from "
                                "the theme's passive/directed role toward the third party, never from "
                                "the actor's active action. For observations marked "
                                "semantic_naming_required, the parser expression describes source "
                                "layout rather than the logical predicate: infer a reusable semantic "
                                "role from the frozen header, endpoint order and evidence. Treat each "
                                "structural_role_contract as an upper bound on meaning. A table-header "
                                "assignment establishes only its ordered column role or mapping, not "
                                "fulfillment, compliance, causation, identity or ownership. A leading "
                                "identifier binding establishes that the identifier scopes or labels "
                                "the clause's principal topic/actor, not that both endpoints are "
                                "identical. A participant projection from an explicit container/map "
                                "clause preserves the stated participant order and must not be voiced "
                                "as the inverse. Reuse an "
                                "existing identifier only when its complete role contract is an exact "
                                "fit. Never use a bare auxiliary or a generic placeholder that merely "
                                "states that an unspecified relation, action or property exists. If "
                                "the source does not support a more specific stable role, keep the "
                                "row but give it a fail-closed, evidence-specific naming decision for "
                                "the later policy verifier; do not disguise uncertainty as a generic "
                                "catch-all. Do not create instance-, document- or surface-name-specific "
                                "identifiers. Each row is a compact naming "
                                "decision, not a full Schema policy: return only observation_group_id, "
                                "name, one concise user-facing description in the configured project "
                                "knowledge language, and optional concise "
                                "source/target type and role hints. Never return analysis, protocol "
                                "metadata, examples, evidence, nested relation_contract objects or a "
                                "complete policy. This stage cannot publish facts. Return strict JSON. "
                                + language_instruction
                                + (
                                    " Protocol retry: return only a root "
                                    "relation_type_candidates array with one compact row for every "
                                    "input group; every row must include observation_group_id and name."
                                    if attempt else ""
                                )
                            ),
                            prompt=json.dumps({
                                "available_entity_types": [
                                    str(item.get("name") or "")
                                    for item in current.get("entity_types") or []
                                    if isinstance(item, Mapping)
                                ],
                                "existing_relation_identifiers": [
                                    str(item.get("name") or "")
                                    for item in current.get("relation_types") or []
                                    if isinstance(item, Mapping)
                                ],
                                "identifiers_already_compiled": [
                                    str(item.get("name") or "")
                                    for item in compiled_summaries
                                ],
                                "validated_observation_groups": group_records,
                                "protocol_attempt": attempt + 1,
                            }, ensure_ascii=False),
                            schema=OPEN_RELATION_BATCH_INDUCTION_FORMAT,
                            temperature=0,
                            context_budget_tokens=int(
                                project.get("model_max_tokens") or 32768
                            ),
                            max_output_tokens=min(
                                Config.schema_review_max_output_tokens, 4096,
                            ),
                        )
                        append_induced_batch(
                            induced_value, group_records, label,
                        )
                        return
                    except Exception as exc:
                        last_error = exc
                split_worthy = any(
                    marker in str(last_error or "").lower()
                    for marker in (
                        "truncated", "output length", "underestimated",
                        "omitted required fields",
                    )
                )
                if len(group_records) > 1 and depth < 1 and split_worthy:
                    midpoint = max(1, len(group_records) // 2)
                    induce_relation_groups(
                        group_records[:midpoint], f"{label}.1", depth + 1,
                    )
                    induce_relation_groups(
                        group_records[midpoint:], f"{label}.2", depth + 1,
                    )
                    return
                message = (
                    f"open relation schema induction batch {label}: "
                    f"{last_error}"
                )
                errors.append(message)
                LOGGER.warning("Open relation schema induction failed: %s", message)

            induction_batch_size = int(
                Config.schema_relation_induction_batch_size
            )
            for start in range(0, len(relation_groups), induction_batch_size):
                induce_relation_groups(
                    relation_groups[start:start + induction_batch_size],
                    str(start // induction_batch_size + 1),
                )
        induced_evidence = {
            (
                str(observation.get("chunk_key") or ""),
                str(observation.get("evidence_quote") or ""),
            )
            for batch in batches
            for candidate in batch.get("relation_type_candidates") or []
            if candidate.get("_induction_observations")
            for observation in candidate.get("_induction_observations") or []
            if str(observation.get("evidence_quote") or "")
        }
        if induced_evidence:
            for batch in batches:
                retained_relations = []
                for candidate in batch.get(
                    "relation_type_candidates"
                ) or []:
                    if candidate.get("_discovery_origin") != "broad_review":
                        retained_relations.append(candidate)
                        continue
                    citations = {
                        (str(chunk_key), str(quote))
                        for chunk_key in candidate.get("evidence_chunks") or []
                        for quote in candidate.get("evidence_quotes") or []
                        if str(chunk_key) and str(quote)
                    }
                    if citations and citations <= induced_evidence:
                        continue
                    retained_relations.append(candidate)
                batch["relation_type_candidates"] = retained_relations
        return batches, errors

    def _consolidate_candidates(
        self,
        project: dict[str, Any],
        current: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not candidates:
            return {"summary": "", "entity_type_candidates": [], "relation_type_candidates": []}
        knowledge_language = resolve_knowledge_language(
            project.get("knowledge_language") or "auto",
            json.dumps(candidates, ensure_ascii=False),
        )
        language_instruction = _schema_language_instruction(
            knowledge_language,
        )

        def deterministic() -> dict[str, Any]:
            """Merge only identical candidate identifiers and source evidence.

            This is a protocol-recovery path, not a semantic synonym merger.
            The later evidence and metadata gates remain authoritative.
            """

            entity_values: dict[str, dict[str, Any]] = {}
            relation_values: dict[str, dict[str, Any]] = {}
            for batch in candidates:
                for source, target in (
                    (batch.get("entity_type_candidates") or [], entity_values),
                    (batch.get("relation_type_candidates") or [], relation_values),
                ):
                    for raw in source:
                        if not isinstance(raw, dict):
                            continue
                        name = str(raw.get("name") or "").strip()
                        if not _valid_identifier(name):
                            continue
                        key = _type_key(name)
                        current = target.get(key)
                        if current is None:
                            current = dict(raw)
                            target[key] = current
                        elif (
                            bool(raw.get("_induction_observations"))
                            and not bool(current.get("_induction_observations"))
                        ) or (
                            bool(raw.get("_induction_observations"))
                            == bool(current.get("_induction_observations"))
                            and coerce_confidence(
                                raw.get("confidence"), 0.0,
                            ) > coerce_confidence(
                                current.get("confidence"), 0.0,
                            )
                        ):
                            replacement = dict(raw)
                            replacement["evidence_chunks"] = list(dict.fromkeys([
                                *(current.get("evidence_chunks") or []),
                                *(raw.get("evidence_chunks") or []),
                            ]))
                            current = replacement
                            target[key] = current
                        else:
                            current["evidence_chunks"] = list(dict.fromkeys([
                                *(current.get("evidence_chunks") or []),
                                *(raw.get("evidence_chunks") or []),
                            ]))
                        for field in (
                            "positive_examples", "negative_examples",
                            "optional_soft_cues", "evidence_quotes",
                        ):
                            if field in current or field in raw:
                                current[field] = list(dict.fromkeys([
                                    *(current.get(field) or []),
                                    *(raw.get(field) or []),
                                ]))
            for candidate in relation_values.values():
                if not candidate.get("source_types") and candidate.get("domain_types"):
                    candidate["source_types"] = list(candidate["domain_types"])
                if not candidate.get("target_types") and candidate.get("range_types"):
                    candidate["target_types"] = list(candidate["range_types"])

            # Collapse only a narrow, source-provable class of semantic
            # duplicates: two induced policies over the exact same ordered
            # endpoint/evidence occurrence where one open expression is merely
            # the other plus an evidence-specific suffix.  This catches names
            # such as a reusable base role versus the same role with an endpoint
            # or contrastive comment appended, without merging distinct
            # multi-label facts stated about the same pair.
            collapsed_relations = 0

            def observation_signatures(
                candidate: Mapping[str, Any],
            ) -> set[tuple[str, str, str, str]]:
                return {
                    (
                        str(item.get("chunk_key") or ""),
                        str(item.get("evidence_quote") or ""),
                        str(item.get("subject_surface") or "").casefold(),
                        str(item.get("object_surface") or "").casefold(),
                    )
                    for item in candidate.get("_induction_observations") or []
                    if isinstance(item, Mapping)
                    and str(item.get("evidence_quote") or "")
                    and str(item.get("subject_surface") or "")
                    and str(item.get("object_surface") or "")
                }

            def expression_keys(
                candidate: Mapping[str, Any],
            ) -> set[tuple[str, ...]]:
                return {
                    tuple(key.split("|"))
                    for item in candidate.get("_induction_observations") or []
                    if isinstance(item, Mapping)
                    for key in [_relation_expression_family_key(str(
                        item.get("source_relation_expression")
                        or item.get("relation_expression") or ""
                    ))]
                    if key
                }

            def strict_prefix_pair(
                left: set[tuple[str, ...]], right: set[tuple[str, ...]],
            ) -> bool:
                return any(
                    len(shorter) < len(longer)
                    and longer[:len(shorter)] == shorter
                    for shorter in left
                    for longer in right
                ) or any(
                    len(shorter) < len(longer)
                    and longer[:len(shorter)] == shorter
                    for shorter in right
                    for longer in left
                )

            relation_keys = list(relation_values)
            removed_keys: set[str] = set()
            for left_index, left_key in enumerate(relation_keys):
                if left_key in removed_keys:
                    continue
                for right_key in relation_keys[left_index + 1:]:
                    if right_key in removed_keys:
                        continue
                    left = relation_values[left_key]
                    right = relation_values[right_key]
                    left_signatures = observation_signatures(left)
                    right_signatures = observation_signatures(right)
                    left_expressions = expression_keys(left)
                    right_expressions = expression_keys(right)
                    if (
                        not left_signatures
                        or not right_signatures
                        or not left_signatures.intersection(right_signatures)
                        or not strict_prefix_pair(
                            left_expressions, right_expressions,
                        )
                    ):
                        continue

                    def reuse_rank(
                        item: Mapping[str, Any],
                    ) -> tuple[int, int, float, str]:
                        lengths = [
                            len(value) for value in expression_keys(item)
                        ]
                        return (
                            min(lengths) if lengths else 10_000,
                            -len(item.get("_induction_observations") or []),
                            -coerce_confidence(item.get("confidence"), 0.0),
                            str(item.get("name") or ""),
                        )

                    winner_key, loser_key = (
                        (left_key, right_key)
                        if reuse_rank(left) <= reuse_rank(right)
                        else (right_key, left_key)
                    )
                    winner = relation_values[winner_key]
                    loser = relation_values[loser_key]
                    for field in (
                        "evidence_chunks", "evidence_quotes",
                        "positive_examples", "negative_examples",
                        "optional_soft_cues",
                    ):
                        winner[field] = list(dict.fromkeys([
                            *(winner.get(field) or []),
                            *(loser.get(field) or []),
                        ]))
                    merged_observations: dict[str, dict[str, Any]] = {}
                    for observation in [
                        *(winner.get("_induction_observations") or []),
                        *(loser.get("_induction_observations") or []),
                    ]:
                        if not isinstance(observation, Mapping):
                            continue
                        key = json.dumps(
                            observation, ensure_ascii=False,
                            sort_keys=True, separators=(",", ":"),
                        )
                        merged_observations.setdefault(
                            key, copy.deepcopy(dict(observation)),
                        )
                    winner["_induction_observations"] = list(
                        merged_observations.values()
                    )
                    removed_keys.add(loser_key)
                    collapsed_relations += 1
                    if loser_key == left_key:
                        break
            for key in removed_keys:
                relation_values.pop(key, None)
            return {
                "summary": (
                    "The candidate set was merged by identical identifiers while preserving source evidence."
                    if knowledge_language == ENGLISH else
                    "模型汇总未保留候选；按相同标识符合并来源证据"
                ),
                "entity_type_candidates": list(entity_values.values()),
                "relation_type_candidates": list(relation_values.values()),
                "consolidation_diagnostics": {
                    "source_overqualified_duplicates_collapsed": (
                        collapsed_relations
                    ),
                },
            }

        # Discovery batches already carry immutable evidence.  Generative
        # synonym consolidation added another full-schema call, frequently
        # returned the active schema instead of candidates, and could replace
        # an exact open-observation policy with a broad prose draft.  Exact-ID
        # consolidation is complete here; entity existence checks and the
        # independent relation novelty gate handle semantic duplication later.
        consolidated = deterministic()
        consolidated["summary"] = (
            "Candidates were merged by identical identifiers while preserving ordered roles and source evidence."
            if knowledge_language == ENGLISH else
            "候选按相同标识符合并；保留开放观察的有序角色与来源证据"
        )
        return consolidated

        entity_candidates = [
            item
            for batch in candidates
            for item in (batch.get("entity_type_candidates") or [])
            if isinstance(item, dict)
        ]
        relation_candidates = [
            item
            for batch in candidates
            for item in (batch.get("relation_type_candidates") or [])
            if isinstance(item, dict)
        ]
        if (
            not entity_candidates
            and relation_candidates
            and all(
                item.get("_induction_observations")
                for item in relation_candidates
            )
        ):
            # Open relation induction already compares every observation group
            # with the active schema and with the other groups in its bounded
            # batch.  A second generative synonym-consolidation pass cannot add
            # evidence, but can erase ordered-role provenance or turn two
            # inverse projections into one broad predicate.  Merge only exact
            # identifiers here; independent policy and novelty gates still run
            # below before any additive schema change is accepted.
            consolidated = deterministic()
            consolidated["summary"] = (
                "开放关系候选按相同标识符合并；保留有序角色与来源证据"
            )
            return consolidated

        try:
            consolidated = self._candidate_payload(self.ollama.json_chat(
                model=project.get("llm_model") or Config.default_llm_model,
                system=(
                    "You consolidate candidate additions to an existing logical knowledge graph schema. "
                    "Merge synonyms, reject instance names and overly narrow types, preserve all valid evidence chunk keys, "
                    "preserve only verbatim evidence_quotes already present in candidate batches, "
                    "and keep only additive reusable types supported by the supplied candidates. Preserve every explicit "
                    "relation policy field; repair structural inconsistencies so source_types equals domain_types and "
                    "target_types equals range_types, without broadening the observed semantics. Never infer behavior "
                    "from a predicate name. " + language_instruction + " "
                    "Never preserve analysis, self-correction, unresolved alternatives, or draft commentary "
                    "inside a finalized policy; discard such a candidate instead of guessing its meaning."
                ),
                prompt=(
                    f"Current schema:\n{json.dumps(current, ensure_ascii=False)}\n\n"
                    f"Candidate batches:\n{json.dumps(candidates, ensure_ascii=False)}"
                ),
                schema=SCHEMA_REVIEW_FORMAT,
                temperature=0,
                context_budget_tokens=int(project.get("model_max_tokens") or 32768),
                max_output_tokens=Config.schema_review_max_output_tokens,
            ), fallback_evidence_chunks=[
                str(key)
                for item in candidates
                for candidate in (
                    *(item.get("entity_type_candidates") or []),
                    *(item.get("relation_type_candidates") or []),
                )
                for key in candidate.get("evidence_chunks") or []
            ])
            had_candidates = any(
                item.get("entity_type_candidates")
                or item.get("relation_type_candidates")
                for item in candidates
            )
            exact_merge = deterministic()
            if had_candidates and not (
                consolidated.get("entity_type_candidates")
                or consolidated.get("relation_type_candidates")
            ):
                LOGGER.warning(
                    "Schema candidate consolidation returned an empty transport shape; "
                    "using exact-identifier evidence merge"
                )
                return exact_merge
            # Consolidation may merge synonyms, but it has no rejection/audit
            # contract and therefore cannot silently erase only part of the
            # source-grounded inventory.  Restore any missing exact identifier
            # from the deterministic evidence merge.  Independent policy and
            # evidence gates below still decide whether it is activated.
            for field in (
                "entity_type_candidates", "relation_type_candidates",
            ):
                output = list(consolidated.get(field) or [])
                present = {
                    _type_key(str(item.get("name") or ""))
                    for item in output if isinstance(item, dict)
                }
                output.extend(
                    copy.deepcopy(item)
                    for item in exact_merge.get(field) or []
                    if _type_key(str(item.get("name") or "")) not in present
                )
                consolidated[field] = output
            return consolidated
        except Exception as exc:
            LOGGER.warning("Schema candidate consolidation failed; using batch candidates: %s", exc)
            return deterministic()

    def _historical_relation_candidate_batch(
        self,
        project_id: int,
        document_id: int,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Replay only historical drafts whose provenance is still exact.

        Candidate discovery is generative and may omit a previously observed
        reusable relation on a retry.  Replaying a draft is safe only when it
        belongs to the same document and its quoted evidence still occurs
        verbatim in the currently selected frozen chunks.  The replayed draft
        receives no acceptance credit: consolidation, policy verification,
        evidence gates and additive application all run again.
        """

        loader = getattr(self.db, "list_schema_evaluations", None)
        if not callable(loader) or not samples:
            return None
        sample_by_key = {_chunk_key(item): item for item in samples}
        selected: dict[str, dict[str, Any]] = {}
        try:
            rows = loader(project_id, document_id=document_id)
        except Exception as exc:
            LOGGER.warning("Schema candidate audit replay unavailable: %s", exc)
            return None
        for row in rows[:8]:
            evaluation = row.get("evaluation") if isinstance(row, dict) else {}
            if not isinstance(evaluation, dict):
                continue
            reports = evaluation.get("document_reports") or [evaluation]
            for report in reports:
                if not isinstance(report, dict):
                    continue
                if str(
                    report.get("schema_discovery_protocol_version") or ""
                ) != SCHEMA_DISCOVERY_PROTOCOL_VERSION:
                    continue
                for raw in report.get("proposed_relation_types") or []:
                    if not isinstance(raw, dict):
                        continue
                    name = str(raw.get("name") or "").strip()
                    key = _type_key(name)
                    if not _valid_identifier(name) or key in selected:
                        continue
                    cited = [
                        str(value) for value in raw.get("evidence_chunks") or []
                        if str(value) in sample_by_key
                    ]
                    if not cited:
                        continue
                    exact_quotes = []
                    for quote in raw.get("evidence_quotes") or []:
                        value = str(quote).strip()
                        if value and any(
                            value in str(
                                sample_by_key[chunk_key].get("_schema_text")
                                or sample_by_key[chunk_key].get("content") or ""
                            )
                            for chunk_key in cited
                        ):
                            exact_quotes.append(value)
                    if not exact_quotes:
                        continue
                    candidate = {
                        field: copy.deepcopy(value)
                        for field, value in raw.items()
                        if not str(field).startswith("_")
                    }
                    candidate["evidence_chunks"] = list(dict.fromkeys(cited))
                    candidate["evidence_quotes"] = list(dict.fromkeys(exact_quotes))
                    selected[key] = candidate
        if not selected:
            return None
        return {
            "summary": "重放当前原文仍逐字支持的历史候选草案",
            "entity_type_candidates": [],
            "relation_type_candidates": list(selected.values()),
        }

    def _verify_relation_policies(
        self,
        project: dict[str, Any],
        review: dict[str, Any],
        samples: list[dict[str, Any]],
        current_schema: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Verify that every proposed predicate is one coherent binary policy."""

        errors: list[str] = []
        knowledge_language = _schema_knowledge_language(project, samples)
        language_instruction = _schema_language_instruction(
            knowledge_language,
        )
        sample_by_key = {_chunk_key(item): item for item in samples}
        model = project.get("llm_model") or Config.default_llm_model
        system = (
            "你是知识图谱知识结构的独立一致性核验器，只审查给定候选，不修改候选，"
            "也不得根据谓词名称猜测。只有同时满足以下条件才返回 SUPPORTED：该定义是一个"
            "二元关系；主语和宾语各自只有一个稳定、互斥的语义角色；domain/range 与角色"
            "一致；方向唯一；正反例与定义一致；逐字证据明确支持该有序关系。若同一端点"
            "混合接收者、物品、地点、事件等不同论元角色，或候选文本含未解决的备选定义、"
            "自我修正、分析过程，返回 REJECTED。若定义或端点角色只说明存在某种未指明的"
            "关系、动作、属性或系词连接，没有给出可跨实例复用且可与其他关系区分的语义，"
            "也必须返回 REJECTED；不得用通用占位语义掩盖证据不足。"
            "available_entity_types 是当前可用的实体类型及定义。通配符 * 只表示经过证据"
            "核验后确实允许任意实体类型，不表示类型未知，也不得为省事而代替能够由稳定"
            "端点角色确定的具体类型集合。若角色定义要求动作、事件、地点、组织、产品等"
            "特定语义类别，而 domain/range 却使用 * 或包含与该角色明显不相容的类别，"
            "必须返回 REJECTED。端点只是参与某动作，不等于端点本身就是该动作、任务、"
            "结果、工具、地点或接收方。"
            "如果候选包含 _induction_observations，主语和宾语角色必须分别只对应每条观察"
            "中的 subject_surface 和 object_surface；证据中的未选第三参与者不得成为备选"
            "端点。核验的是原文直接蕴含的语义角色，而不是表面主动/被动语态；明确的"
            "多参与者动作直接支持 actor→theme 以及 theme→recipient/source/destination"
            "投影，明确连接 X 与 Y 的容器也直接支持其分别指向 X、Y。不得仅因同一证据"
            "还支持主动关系，或投影标识采用被动措辞，就拒绝一个端点正确的直接角色投影。"
            "若观察含 structural_role_contract，必须把它当作解析器已验证的语义上界而非"
            "普通相邻：table_header_assignment 直接支持行键到对应表头列值的有序角色/映射，"
            "但不支持满足、合规、因果、同一或所有权等更强含义；line_identifier_binding "
            "直接支持前置编号对该条款主要主题/行为主体的管辖或标注，但不支持同一或作者"
            "关系。候选超出该上界应 REJECTED，候选精确落在该上界内不得仅因缺少句内动词"
            "而拒绝。"
            + language_instruction + " 信息不足返回 UNCERTAIN。"
            "只返回结构化结果。"
        )
        candidates = review.get("relation_type_candidates") or []
        for candidate in candidates:
            observations = list(
                candidate.get("_induction_observations") or []
            )
            if not observations:
                continue
            relation_expressions = list(dict.fromkeys(
                str(item.get("source_relation_expression")
                    or item.get("relation_expression") or "").strip()
                for item in observations
                if str(item.get("source_relation_expression")
                    or item.get("relation_expression") or "").strip()
            ))
            primary_expression = (
                relation_expressions[0][:160]
                if relation_expressions else "明确关系"
            )
            available_names = {
                str(item.get("name") or "")
                for item in (current_schema or {}).get("entity_types") or []
                if isinstance(item, Mapping)
            }

            def compatible_types(values: Any) -> list[str]:
                return list(dict.fromkeys(
                    str(value) for value in values or []
                    if (
                        not available_names
                        or str(value) == "*"
                        or str(value) in available_names
                    )
                ))

            source_types = compatible_types(
                candidate.get("source_types")
                or candidate.get("domain_types") or []
            )
            target_types = compatible_types(
                candidate.get("target_types")
                or candidate.get("range_types") or []
            )
            if not source_types and "GenericEntity" in available_names:
                source_types = ["GenericEntity"]
            if not target_types and "GenericEntity" in available_names:
                target_types = ["GenericEntity"]
            if source_types:
                candidate["source_types"] = list(source_types)
                candidate["domain_types"] = list(source_types)
            if target_types:
                candidate["target_types"] = list(target_types)
                candidate["range_types"] = list(target_types)

            ordered_role_projections = {
                str(item.get("ordered_role_projection") or "")
                for item in observations
                if str(item.get("ordered_role_projection") or "")
            }
            source_role_pairs = sorted({
                str(item.get("source_role_pair") or "")
                for item in observations
                if str(item.get("source_role_pair") or "") in {
                    "grammatical_subject->action_theme",
                    "action_theme->role_complement",
                    "grammatical_subject->role_complement",
                    "grammatical_subject->direct_argument",
                    "direct_argument->grammatical_subject",
                    "action_theme->grammatical_subject",
                    "role_complement->grammatical_subject",
                    "collective_participant->collective_participant",
                    "possessor->possessed",
                    "possessed->possessor",
                }
            })
            if source_role_pairs:
                candidate["source_role_pairs"] = source_role_pairs
            structural_evidence_kinds = {
                str(item.get("structural_evidence_kind") or "")
                for item in observations
                if str(item.get("structural_evidence_kind") or "")
            }
            if ordered_role_projections == {"theme_to_third_party"}:
                if knowledge_language == ENGLISH:
                    candidate.update({
                        "description": "The acted-on theme is explicitly directed to its recipient, source, or destination.",
                        "semantic_definition": "A directed binary role relation from the acted-on theme in a multi-participant action to the recipient, source, or destination explicitly stated by the source.",
                        "subject_role_description": "The entity acting as the theme in the explicitly stated multi-participant action.",
                        "object_role_description": "The recipient, source, or destination entity to which the source explicitly directs that theme.",
                        "direction_description": "From the acted-on theme to its explicitly stated recipient, source, or destination.",
                    })
                else:
                    candidate.update({
                        "description": "被作用主题明确指向其接收方、来源或目的地。",
                        "semantic_definition": "表示多参与者动作中，被作用主题到原文明示的接收方、来源或目的地之间的有向二元角色关系。",
                        "subject_role_description": "在原文明示的多参与者动作中作为被作用主题的实体。",
                        "object_role_description": "该主题在原文中明确指向的接收方、来源或目的地实体。",
                        "direction_description": "从被作用主题指向其原文明示的接收方、来源或目的地。",
                    })
            elif ordered_role_projections == {
                "container_participant_projection"
            }:
                if knowledge_language == ENGLISH:
                    candidate.update({
                        "description": "A source-stated container, mapping, or index connects two ordered participants.",
                        "semantic_definition": "A directed binary relation from the first participant to the second participant directly connected by a container, mapping, or index; it does not assert identity, containment, or ownership.",
                        "subject_role_description": "The first participant in the source-stated connection order.",
                        "object_role_description": "The second participant explicitly reached from the first in the same connection expression.",
                        "direction_description": "From the first participant to the second in the source-stated connection.",
                    })
                else:
                    candidate.update({
                        "description": "原文明示的容器、映射或索引连接两个有序参与方。",
                        "semantic_definition": "表示容器、映射或索引在原文中直接连接的第一个参与方到第二个参与方之间的有向二元关系，不表示同一、包含或所有权。",
                        "subject_role_description": "原文连接表达中按当前顺序出现的第一个参与方。",
                        "object_role_description": "同一连接表达中由前一参与方明确指向的第二个参与方。",
                        "direction_description": "从原文明示连接中的第一个参与方指向第二个参与方。",
                    })

            if structural_evidence_kinds == {"table_header_assignment"}:
                role_labels = list(dict.fromkeys(
                    str(item.get("structural_role_label") or "").strip()
                    for item in observations
                    if str(item.get("structural_role_label") or "").strip()
                ))
                role_text = " / ".join(role_labels) or (
                    "corresponding column"
                    if knowledge_language == ENGLISH else "对应列"
                )
                if knowledge_language == ENGLISH:
                    candidate.update({
                        "description": f'The table row key is explicitly mapped to the value under the "{role_text}" column header.',
                        "semantic_definition": f'A directed role assignment or mapping from a row-key entity to the value in the "{role_text}" column of the same row; it preserves only the role directly stated by the table and does not imply satisfaction, compliance, causation, identity, or ownership.',
                        "subject_role_description": "The entity serving as the row key around which the remaining columns provide information.",
                        "object_role_description": f'The value in the "{role_text}" column of the same row, whose role is defined by that header.',
                        "direction_description": f'From the table row key to the value in the "{role_text}" column of the same row.',
                    })
                else:
                    candidate.update({
                        "description": f"表格行键按“{role_text}”列表头明确映射到该列值。",
                        "semantic_definition": f"表示同一表格行中，行键实体到“{role_text}”表头所定义列值之间的有向角色赋值或映射；仅保留表格直接陈述的角色，不扩展为满足、合规、因果、同一或所有权。",
                        "subject_role_description": "表格数据行中承担行键作用、其余列围绕其描述的实体。",
                        "object_role_description": f"同一数据行中位于“{role_text}”列、由该表头定义角色的值。",
                        "direction_description": f"从表格行键指向同一行的“{role_text}”列值。",
                    })
                candidate["structural_evidence_kinds"] = sorted(
                    structural_evidence_kinds
                )
                candidate["structural_role_labels"] = role_labels
            elif structural_evidence_kinds == {
                "line_identifier_binding"
            }:
                if knowledge_language == ENGLISH:
                    candidate.update({
                        "description": "A leading identifier or label explicitly marks the principal topic or actor of its clause.",
                        "semantic_definition": "A directed labeling relation from a line-leading identifier or label to the principal topic or actor in the clause it directly governs; it does not assert identity or authorship.",
                        "subject_role_description": "The identifier or label appearing at the start of and governing the clause.",
                        "object_role_description": "The principal topic or actor explicitly stated in the clause governed by that identifier or label.",
                        "direction_description": "From the line-leading identifier or label to the principal topic or actor of its governed clause.",
                    })
                else:
                    candidate.update({
                        "description": "前置编号或标签明确标注其所在条款的主要主题或行为主体。",
                        "semantic_definition": "表示行首编号或标签到其直接管辖条款中主要主题或行为主体之间的有向标注关系，不表示两个端点同一或互为作者。",
                        "subject_role_description": "直接位于条款行首并管辖该条款内容的编号或标签。",
                        "object_role_description": "该编号或标签所管辖条款中原文明示的主要主题或行为主体。",
                        "direction_description": "从行首编号或标签指向其直接管辖条款的主要主题或行为主体。",
                    })
                candidate["structural_evidence_kinds"] = sorted(
                    structural_evidence_kinds
                )

            localized_fallbacks = (
                {
                    "description": f'A directed binary relation explicitly expressed by "{primary_expression}" in the source.',
                    "semantic_definition": f'The subject acts on or relates to the object according to the source-stated meaning of "{primary_expression}".',
                    "subject_role_description": f'The entity serving as the subject in the "{primary_expression}" relation.',
                    "object_role_description": f'The entity serving as the object in the "{primary_expression}" relation.',
                }
                if knowledge_language == ENGLISH else
                {
                    "description": f"表示原文明示的“{primary_expression}”有向二元关系。",
                    "semantic_definition": f"主体按原文明示的“{primary_expression}”语义作用于客体。",
                    "subject_role_description": f"在“{primary_expression}”关系中充当主语的实体。",
                    "object_role_description": f"在“{primary_expression}”关系中充当客体的实体。",
                }
            )
            for field, fallback in localized_fallbacks.items():
                if not _schema_user_text_matches(
                    candidate.get(field), knowledge_language,
                ):
                    candidate[field] = fallback
            if not _schema_user_text_matches(
                candidate.get("direction_description"), knowledge_language,
            ):
                candidate["direction_description"] = (
                    "From the subject role to the object role."
                    if knowledge_language == ENGLISH else
                    f"从{candidate['subject_role_description']}指向"
                    f"{candidate['object_role_description']}"
                )
            subject_role = str(
                candidate.get("subject_role_description") or ""
            ).strip()
            object_role = str(
                candidate.get("object_role_description") or ""
            ).strip()
            semantic_definition = str(
                candidate.get("semantic_definition")
                or candidate.get("description") or ""
            ).strip()
            description = str(
                candidate.get("description") or semantic_definition
            ).strip()
            if source_types:
                candidate["source_types"] = source_types
                candidate["domain_types"] = list(source_types)
            if target_types:
                candidate["target_types"] = target_types
                candidate["range_types"] = list(target_types)
            if description:
                candidate.setdefault("description", description)
            if semantic_definition:
                candidate.setdefault(
                    "semantic_definition", semantic_definition,
                )
            if subject_role and object_role:
                candidate.setdefault(
                    "direction_description",
                    f"从{subject_role}指向{object_role}。",
                )
            candidate.setdefault("symmetric", False)
            candidate.setdefault("inverse_predicate", None)
            candidate.setdefault("canonical_orientation", "forward")
            candidate.setdefault("publish_mode", "auto")
            candidate.setdefault("reflexive", False)
            candidate.setdefault(
                "confidence",
                Config.schema_review_single_evidence_confidence,
            )
            candidate.setdefault("positive_examples", [
                "The source explicitly states that the subject relates to the object in this way."
                if knowledge_language == ENGLISH else
                "原文明示主体按该关系作用于客体。",
            ])
            candidate.setdefault("negative_examples", [
                "The two entities merely co-occur; the source does not state this relation."
                if knowledge_language == ENGLISH else
                "两个实体仅共同出现，原文未陈述该关系。",
            ])
            candidate.setdefault("optional_soft_cues", list(dict.fromkeys(
                relation_expressions
            )))
            candidate.setdefault("source_role_pairs", [])
            candidate.setdefault("conflicts_with", [])
            candidate.setdefault("negation_policy", "reject_negated")
            candidate.setdefault("modality_policy", "require_asserted")
            candidate["_deterministic_policy_completion"] = {
                "version": SCHEMA_DISCOVERY_PROTOCOL_VERSION,
                "source": "validated_open_observations",
                "observation_count": len(observations),
            }
        # Prefer candidates compiled from explicit open observations over
        # broad-review prose.  This is a provenance/role-quality ordering, not
        # a predicate-name preference.  It also makes within-review semantic
        # deduplication retain the candidate with exact ordered endpoints.
        candidates.sort(key=lambda item: (
            0 if item.get("_induction_observations") else 1,
            -len(item.get("_induction_observations") or []),
            -self._source_backed_example_count(item, sample_by_key),
            str(item.get("name") or ""),
        ))
        required_policy_fields = set(
            SCHEMA_REVIEW_FORMAT["properties"]
            ["relation_type_candidates"]["items"]["required"]
        )

        def structural_issues(candidate: dict[str, Any]) -> list[str]:
            """Return transport/metadata defects without interpreting semantics."""

            issues = [
                f"missing:{field}"
                for field in sorted(required_policy_fields)
                if field not in candidate
            ]
            for field in (
                "description", "semantic_definition", "direction_description",
                "subject_role_description", "object_role_description",
            ):
                if field in candidate and not _schema_user_text_matches(
                    candidate.get(field), knowledge_language,
                ):
                    issues.append(f"wrong_language:{field}")
            if "symmetric" in candidate and not isinstance(
                candidate.get("symmetric"), bool,
            ):
                issues.append("invalid:symmetric")
            if "reflexive" in candidate and not isinstance(
                candidate.get("reflexive"), bool,
            ):
                issues.append("invalid:reflexive")
            return issues

        def verify(
            candidate: dict[str, Any], excerpts: dict[str, str], index: int,
            *, phase: str,
        ) -> dict[str, str]:
            try:
                request = {
                    "candidate": candidate,
                    "source_excerpts": excerpts,
                    "available_entity_types": list(
                        (current_schema or {}).get("entity_types") or []
                    ),
                }
                for protocol_attempt in range(2):
                    value = self.ollama.json_chat(
                        model=model,
                        system=(
                            system
                            + (
                                " 协议重试：必须同时返回有效 verdict 和非空 reason；"
                                "reason 应简要说明端点角色、方向和逐字证据为何通过或不通过。"
                                if protocol_attempt else ""
                            )
                        ),
                        prompt=json.dumps({
                            **request,
                            "protocol_attempt": protocol_attempt + 1,
                        }, ensure_ascii=False),
                        schema=SCHEMA_POLICY_VERIFICATION_FORMAT,
                        temperature=0,
                        context_budget_tokens=int(
                            project.get("model_max_tokens") or 32768
                        ),
                        max_output_tokens=2048,
                    )
                    raw_verdict = str(
                        value.get("verdict")
                        or value.get("decision")
                        or value.get("result")
                        or value.get("status") or ""
                    ).strip().upper() if isinstance(value, dict) else ""
                    verdict = (
                        raw_verdict
                        if raw_verdict in {
                            "SUPPORTED", "REJECTED", "UNCERTAIN",
                        }
                        else "UNCERTAIN"
                    )
                    reason = str(
                        value.get("reason")
                        or value.get("reasoning")
                        or value.get("explanation") or ""
                    ).strip() if isinstance(value, dict) else ""
                    if reason:
                        return {"verdict": verdict, "reason": reason[:2000]}
                return {
                    "verdict": "UNCERTAIN",
                    "reason": "verifier_missing_reason_after_retry",
                }
            except Exception as exc:
                message = f"candidate {index} policy {phase}: {exc}"
                errors.append(message)
                LOGGER.warning("Schema relation policy %s failed: %s", phase, message)
                return {
                    "verdict": "UNCERTAIN",
                    "reason": f"{phase}_error:{type(exc).__name__}",
                }

        def batch_verify_induced_policies() -> dict[int, dict[str, str]]:
            eligible = [
                (index, candidate)
                for index, candidate in enumerate(candidates)
                if candidate.get("_induction_observations")
                and not structural_issues(candidate)
            ]
            if len(eligible) < 2:
                return {}
            eligible_indexes = [index for index, _candidate in eligible]
            response_schema = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "decisions": {
                        "type": "array",
                        "maxItems": len(eligible),
                        "items": {
                            "type": "string",
                            "enum": [
                                "SUPPORTED", "REJECTED", "UNCERTAIN",
                            ],
                        },
                    },
                },
                "required": ["decisions"],
            }
            best_decisions: dict[int, dict[str, str]] = {}

            def normalize_verdict(value: Any) -> str:
                verdict = str(value or "").strip().upper()
                return {
                    "ACCEPTED": "SUPPORTED",
                    "ACCEPT": "SUPPORTED",
                    "PASS": "SUPPORTED",
                    "VALID": "SUPPORTED",
                    "DENIED": "REJECTED",
                    "DENY": "REJECTED",
                    "INVALID": "REJECTED",
                    "INSUFFICIENT": "UNCERTAIN",
                    "INSUFFICIENT_EVIDENCE": "UNCERTAIN",
                }.get(verdict, verdict)

            def add_compact_decision(
                decisions: dict[int, dict[str, str]],
                candidate_index: Any,
                raw_verdict: Any,
                reason: Any = "",
                *, mode: str,
            ) -> None:
                verdict = normalize_verdict(raw_verdict)
                if (
                    not isinstance(candidate_index, int)
                    or isinstance(candidate_index, bool)
                    or candidate_index not in eligible_indexes
                    or candidate_index in decisions
                    or verdict not in {
                        "SUPPORTED", "REJECTED", "UNCERTAIN",
                    }
                ):
                    return
                explanation = str(reason or "").strip()
                if not explanation:
                    explanation = "model_compact_batch_policy_decision"
                decisions[candidate_index] = {
                    "verdict": verdict,
                    "reason": explanation[:2000],
                    "verification_mode": mode,
                }

            def compact_decision_fields(
                item: Mapping[str, Any],
            ) -> tuple[Any, Any]:
                reason = item.get("reason") or item.get("explanation") or ""
                raw_verdict = (
                    item.get("verdict")
                    or item.get("decision")
                    or item.get("status")
                )
                if not raw_verdict and reason:
                    # Qwen occasionally places the constrained enum at the
                    # start of the reason while omitting its sibling field.
                    # Recover only that explicit leading protocol token; no
                    # semantic judgment is inferred from free-form prose.
                    match = re.match(
                        r"^\s*(SUPPORTED|REJECTED|UNCERTAIN|ACCEPTED|"
                        r"ACCEPT|PASS|VALID|DENIED|DENY|INVALID)\b",
                        str(reason), flags=re.IGNORECASE,
                    )
                    if match:
                        raw_verdict = match.group(1)
                return raw_verdict, reason

            policy_verification_fields = (
                "name", "semantic_definition", "domain_types", "range_types",
                "direction_description", "subject_role_description",
                "object_role_description", "symmetric",
                "canonical_orientation", "reflexive", "positive_examples",
                "negative_examples", "structural_evidence_kinds",
                "structural_role_labels", "source_role_pairs",
                "conflicts_with", "negation_policy", "modality_policy",
            )
            observation_verification_fields = (
                "subject_surface", "relation_expression",
                "source_relation_expression", "object_surface",
                "evidence_quote", "ordered_role_projection",
                "source_role_pair",
                "structural_evidence_kind", "structural_role_label",
                "structural_role_contract",
            )

            for attempt in range(3):
                try:
                    requested_eligible = [
                        item for item in eligible
                        if item[0] not in best_decisions
                    ]
                    if not requested_eligible:
                        return best_decisions
                    requested_indexes = [
                        index for index, _candidate in requested_eligible
                    ]
                    attempt_schema = copy.deepcopy(response_schema)
                    attempt_decisions = attempt_schema["properties"][
                        "decisions"
                    ]
                    attempt_decisions["minItems"] = len(requested_indexes)
                    attempt_decisions["maxItems"] = len(requested_indexes)
                    value = self.ollama.json_chat(
                        model=model,
                        system=(
                            "你是知识图谱关系策略的批量独立核验器。逐个候选检查其二元"
                            "语义、固定主宾角色、方向、类型、正反例与已冻结开放观察是否"
                            "完全一致。不同候选互不借用证据。端点角色混合、方向错误、"
                            "把提交内容当接收方、把地点当状态、或定义只是泛化关联时必须"
                            "REJECTED；信息不足为 UNCERTAIN。核验语义蕴含而非表面语态："
                            "原文明示 actor、theme 与 recipient/source/destination 时，"
                            "该主动句直接支持 theme 到第三方的有序投影；不得因另有"
                            "actor-to-theme 候选或关系标识采用被动措辞而拒绝它。"
                            "validated_observations 中若有 structural_role_contract，"
                            "按其语义上界核验而非按普通相邻核验：表头赋值可直接支持行键"
                            "到列值的有序角色/映射，但不支持满足、合规、因果、同一或所有权；"
                            "前置编号绑定可直接支持编号对条款主要主题/行为主体的标注或管辖。"
                            "候选标识符也必须以主语到宾语方向准确概括策略，不得使用方向相反、"
                            "语义更强或只表示未知关系/动作/属性的名称。不得修改"
                            "候选。按 candidates 输入顺序返回等长 decisions 数组；数组每项"
                            "只能是 SUPPORTED、REJECTED 或 UNCERTAIN。不要返回理由、索引、"
                            "对象、分析或输入复述。"
                            + (
                                " 协议重试：只返回一个等长的 decisions 字符串数组。"
                                if attempt else ""
                            )
                        ),
                        prompt=json.dumps({
                            "available_entity_types": list(
                                (current_schema or {}).get("entity_types") or []
                            ),
                            "candidates": [{
                                "candidate_index": index,
                                "policy": {
                                    key: copy.deepcopy(candidate.get(key))
                                    for key in policy_verification_fields
                                    if candidate.get(key) is not None
                                },
                                "validated_observations": [{
                                    key: copy.deepcopy(observation.get(key))
                                    for key in observation_verification_fields
                                    if observation.get(key) not in (None, "")
                                } for observation in (
                                    candidate.get("_induction_observations") or []
                                )],
                            } for index, candidate in requested_eligible],
                        }, ensure_ascii=False),
                        schema=attempt_schema,
                        temperature=0,
                        context_budget_tokens=int(
                            project.get("model_max_tokens") or 32768
                        ),
                        max_output_tokens=2048,
                    )
                    raw_decisions = (
                        value.get("decisions")
                        or value.get("items")
                        or value.get("verdicts")
                        if isinstance(value, Mapping) else None
                    )
                    decisions: dict[int, dict[str, str]] = {}
                    reasons = (
                        value.get("reasons")
                        if isinstance(value, Mapping) else None
                    )
                    # Some constrained models unwrap a one-item batch into the
                    # item itself.  This is still an unambiguous transport as
                    # long as the immutable candidate index is present.
                    if (
                        isinstance(value, Mapping)
                        and value.get("candidate_index") is not None
                    ):
                        add_compact_decision(
                            decisions,
                            value.get("candidate_index"),
                            value.get("verdict")
                            or value.get("decision")
                            or value.get("status"),
                            value.get("reason")
                            or value.get("explanation"),
                            mode="batch_unwrapped",
                        )
                    if isinstance(raw_decisions, list):
                        positional = (
                            len(raw_decisions) == len(requested_indexes)
                        )
                        for position, item in enumerate(raw_decisions):
                            if isinstance(item, Mapping):
                                candidate_index = item.get("candidate_index")
                                if candidate_index is None and positional:
                                    candidate_index = requested_indexes[position]
                                raw_verdict, reason = compact_decision_fields(
                                    item
                                )
                                add_compact_decision(
                                    decisions,
                                    candidate_index,
                                    raw_verdict,
                                    reason,
                                    mode="batch",
                                )
                            elif positional:
                                reason = (
                                    reasons[position]
                                    if isinstance(reasons, list)
                                    and position < len(reasons)
                                    else ""
                                )
                                add_compact_decision(
                                    decisions,
                                    requested_indexes[position], item, reason,
                                    mode="batch_positional",
                                )
                    elif isinstance(raw_decisions, Mapping):
                        for raw_index, item in raw_decisions.items():
                            match = re.search(r"\d+", str(raw_index))
                            candidate_index = (
                                int(match.group()) if match else None
                            )
                            if isinstance(item, Mapping):
                                raw_verdict, reason = compact_decision_fields(
                                    item
                                )
                                add_compact_decision(
                                    decisions, candidate_index,
                                    raw_verdict,
                                    reason,
                                    mode="batch_mapping",
                                )
                            else:
                                reason = (
                                    reasons.get(raw_index)
                                    if isinstance(reasons, Mapping) else ""
                                )
                                add_compact_decision(
                                    decisions, candidate_index, item, reason,
                                    mode="batch_mapping",
                                )
                    if isinstance(value, Mapping):
                        for raw_key, raw_value in value.items():
                            if re.fullmatch(r"\d+", str(raw_key)):
                                candidate_index = int(str(raw_key))
                                if isinstance(raw_value, Mapping):
                                    raw_verdict, reason = (
                                        compact_decision_fields(raw_value)
                                    )
                                else:
                                    raw_verdict = raw_value
                                    reason = (
                                        reasons.get(raw_key)
                                        if isinstance(reasons, Mapping) else ""
                                    )
                                add_compact_decision(
                                    decisions, candidate_index, raw_verdict,
                                    reason, mode="batch_numeric_mapping",
                                )
                                continue
                            match = re.fullmatch(
                                r"candidate[_-]?(\d+)[_-](?:decision|verdict|status)",
                                str(raw_key), flags=re.IGNORECASE,
                            )
                            if not match:
                                continue
                            add_compact_decision(
                                decisions, int(match.group(1)), raw_value,
                                value.get(
                                    f"candidate_{match.group(1)}_reason"
                                ),
                                mode="batch_flattened",
                            )
                    best_decisions.update(decisions)
                    if set(best_decisions) == set(eligible_indexes):
                        return best_decisions
                except Exception as exc:
                    LOGGER.warning(
                        "Schema batch policy verification attempt failed: %s",
                        exc,
                    )
            for candidate_index in eligible_indexes:
                best_decisions.setdefault(candidate_index, {
                    "verdict": "UNCERTAIN",
                    "reason": "batch_verifier_missing_decision_after_retry",
                    "verification_mode": "batch_missing",
                })
            return best_decisions

        def repair(
            candidate: dict[str, Any], excerpts: dict[str, str], index: int,
            *, verifier_feedback: str = "",
        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            """Compile one ambiguous draft into one evidence-bounded binary policy.

            The repair model may clarify metadata, but it cannot change the
            candidate identifier, confidence, or provenance.  A second,
            independent verifier remains the only acceptance authority.
            """

            # Structured-output models sometimes return a semantically useful,
            # source-grounded candidate in a compact alias shape.  Missing
            # policy metadata is not allowed into an active Schema, but it is
            # exactly what this bounded compiler is meant to fill before a
            # second independent verification.  Require immutable identity,
            # source evidence, and an explicit semantic draft; otherwise stay
            # fail-closed without asking the model to guess from a name.
            if (
                not _valid_identifier(str(candidate.get("name") or ""))
                or not excerpts
                or (
                    not str(
                        candidate.get("semantic_definition")
                        or candidate.get("description") or ""
                    ).strip()
                    and not candidate.get("_induction_observations")
                )
            ):
                return None, {
                    "attempted": False, "repairable": False,
                    "reason": "insufficient_identity_semantics_or_evidence",
                }
            try:
                repair_input = candidate
                if verifier_feedback and candidate.get("_induction_observations"):
                    # Avoid anchoring the feedback pass on prose that the
                    # independent verifier already found role-ambiguous.  The
                    # immutable identifier, source evidence and selected open
                    # observation projections are sufficient to recompile a
                    # clean policy from scratch.
                    repair_input = {
                        key: copy.deepcopy(candidate.get(key))
                        for key in (
                            "name", "confidence", "evidence_chunks",
                            "evidence_quotes", "_induction_observations",
                        )
                    }
                value = self.ollama.json_chat(
                    model=model,
                    system=(
                        "你是知识图谱关系策略的受限编译器。输入是一个未通过独立核验的"
                        "候选关系及其冻结原文。你只能把它整理成一个语义明确的二元关系，"
                        "不能放宽、虚构或根据关系名称猜测含义。主语和宾语必须各自只有"
                        "一个稳定角色；domain/source 与主语角色一致，range/target 与宾语"
                        "角色一致；方向必须唯一。若原文包含三个或更多参与者，只选择证据"
                        "明确支持的一组固定有序端点，其他参与者只作为上下文，不得写成备选"
                        "端点。如果多组二元投影都成立，必须选择在全部 evidence_quotes 中"
                        "角色最稳定、动作参与关系最直接的一组；存在其他可建模投影本身不是"
                        "不可修复的理由。原草稿把多个宾语角色混在一起、包含解释性分析或有"
                        "明显的角色文字矛盾，正是本编译步骤应当清理的问题；只要全部引用原文"
                        "共同支持至少一组固定有序端点，就必须选择该组并返回 repairable=true，"
                        "不得仅因草稿写坏而拒绝修复。只有不存在任何一组被引用原文逐条明确支持的固定"
                        "端点时，才返回 repairable=false。保留原候选标识符；必须重新编写"
                        "domain/range 时必须参考 available_entity_types，并选择与稳定角色"
                        "相容的最小现有类型集合。* 只允许用于证据确实支持任意实体类型的"
                        "关系，不能用来掩盖未知或混合角色。端点参与某动作不等于端点本身"
                        "就是该动作、任务、结果、工具、地点或接收方。"
                        "candidate 中的 description、semantic_definition、direction_description、"
                        "subject_role_description、object_role_description、positive_examples 和"
                        "negative_examples，必须按项目知识语言重写这些面向用户的字段；"
                        "禁止输出分析过程、自我修正、互斥备选"
                        "或思维链。无法在不猜测"
                        "的情况下形成单一策略时返回 repairable=false。修复结果仍须经过独立"
                        "核验，不能自行决定接受或发布。如果 verifier_feedback 非空，必须"
                        "针对其中指出的结构缺陷重新编译，不能原样返回同一个多角色草稿。"
                        + language_instruction
                    ),
                    prompt=json.dumps({
                        "candidate": repair_input,
                        "source_excerpts": excerpts,
                        "available_entity_types": list(
                            (current_schema or {}).get("entity_types") or []
                        ),
                        "verifier_feedback": verifier_feedback,
                    }, ensure_ascii=False),
                    schema=SCHEMA_POLICY_REPAIR_FORMAT,
                    temperature=0,
                    context_budget_tokens=int(
                        project.get("model_max_tokens") or 32768
                    ),
                    max_output_tokens=3072,
                )
            except Exception as exc:
                message = f"candidate {index} policy repair: {exc}"
                errors.append(message)
                LOGGER.warning("Schema relation policy repair failed: %s", message)
                return None, {
                    "attempted": True,
                    "repairable": False,
                    "reason": f"repair_error:{type(exc).__name__}",
                }

            if not isinstance(value, dict):
                return None, {
                    "attempted": True, "repairable": False,
                    "reason": "invalid_repair_transport",
                }
            raw_candidate = value.get("candidate")
            repairable = value.get("repairable") is True
            if not isinstance(raw_candidate, dict) and value.get("semantic_definition"):
                # Bounded transport recovery for models that unwrap the single
                # requested candidate.  Semantic gates below remain unchanged.
                raw_candidate = value
                repairable = True
            if not repairable or not isinstance(raw_candidate, dict):
                return None, {
                    "attempted": True,
                    "repairable": False,
                    "reason": str(value.get("reason") or "unrepairable")[:2000],
                }

            repaired = dict(raw_candidate)
            # Provenance and identity are immutable across semantic repair.
            for field in (
                "name", "confidence", "evidence_chunks", "evidence_quotes",
                "_induction_observations", "source_role_pairs",
                "conflicts_with", "negation_policy", "modality_policy",
            ):
                repaired[field] = copy.deepcopy(candidate.get(field))
            # A repair may narrow a role type, but a compact transport must
            # not erase a previously valid endpoint contract.  Restore the
            # original aligned type pair when either repaired side is empty.
            repaired_sources = list(
                repaired.get("source_types")
                or repaired.get("domain_types") or []
            )
            repaired_targets = list(
                repaired.get("target_types")
                or repaired.get("range_types") or []
            )
            if not repaired_sources:
                repaired_sources = list(
                    candidate.get("source_types")
                    or candidate.get("domain_types") or []
                )
            if not repaired_targets:
                repaired_targets = list(
                    candidate.get("target_types")
                    or candidate.get("range_types") or []
                )
            if repaired_sources:
                repaired["source_types"] = repaired_sources
                repaired["domain_types"] = list(repaired_sources)
            if repaired_targets:
                repaired["target_types"] = repaired_targets
                repaired["range_types"] = list(repaired_targets)
            for field in ("positive_examples", "negative_examples"):
                repaired[field] = [
                    str(value).strip()
                    for value in repaired.get(field) or []
                    if _schema_user_text_matches(
                        value, knowledge_language,
                    )
                ]
            missing = sorted(
                field for field in required_policy_fields
                if field not in repaired
            )
            if missing:
                return None, {
                    "attempted": True, "repairable": False,
                    "reason": f"repair_missing_fields:{','.join(missing)}",
                }
            wrong_language = [
                field for field in (
                    "description", "semantic_definition",
                    "direction_description", "subject_role_description",
                    "object_role_description",
                )
                if not _schema_user_text_matches(
                    repaired.get(field), knowledge_language,
                )
            ]
            if wrong_language:
                return None, {
                    "attempted": True, "repairable": False,
                    "reason": (
                        "repair_non_chinese_fields:"
                        if knowledge_language == CHINESE else
                        "repair_wrong_language_fields:"
                    ) + ",".join(wrong_language),
                }
            return repaired, {
                "attempted": True,
                "repairable": True,
                "reason": str(value.get("reason") or "")[:2000],
            }

        batch_policy_decisions = batch_verify_induced_policies()
        for candidate_index, decision in batch_policy_decisions.items():
            candidates[candidate_index]["_policy_verification"] = decision

        for index, candidate in enumerate(list(candidates), start=1):
            excerpts = {
                key: str(
                    sample_by_key[key].get("_schema_text")
                    or sample_by_key[key].get("content") or ""
                )
                for key in candidate.get("evidence_chunks") or []
                if key in sample_by_key
            }
            exact_examples = self._source_backed_example_count(
                candidate, sample_by_key,
            )
            if exact_examples:
                evidence_confidence = Config.schema_review_single_evidence_confidence
                # Model confidence is advisory.  Exact frozen evidence is a
                # stronger deterministic signal, but still cannot skip the
                # independent policy and novelty verifiers below.
                candidate["confidence"] = max(
                    coerce_confidence(candidate.get("confidence"), 0.0),
                    evidence_confidence,
                )
            first = candidate.get("_policy_verification") or verify(
                candidate, excerpts, index, phase="verification",
            )
            candidate["_policy_verification"] = first
            candidate_issues = structural_issues(candidate)
            if first["verdict"] == "SUPPORTED" and not candidate_issues:
                continue
            if (
                first["verdict"] != "SUPPORTED"
                and str(first.get("verification_mode") or "").startswith(
                    "batch"
                )
                and not candidate_issues
            ):
                # The compact batch is a screening pass, not a final rejection
                # authority.  A positional REJECTED decision carries no reason
                # and one malformed member can distort the entire list. Route
                # only non-supported members to a reasoned single-candidate
                # adjudication.  The corpus-level addition cap bounds this to
                # at most a handful of one-time calls, while preserving the
                # batch fast path for supported candidates.
                candidate["_policy_batch_verification"] = copy.deepcopy(first)
                first = verify(
                    candidate, excerpts, index,
                    phase="targeted_batch_recheck",
                )
                candidate["_policy_verification"] = first
                if first["verdict"] == "SUPPORTED":
                    continue
            repaired, repair_audit = repair(candidate, excerpts, index)
            if candidate_issues:
                repair_audit["input_structure_issues"] = candidate_issues
            candidate["_policy_repair"] = repair_audit
            if repaired is None:
                continue
            second = verify(repaired, excerpts, index, phase="reverification")
            repair_audit["reverification"] = second
            if second["verdict"] != "SUPPORTED":
                # One bounded feedback repair is allowed.  It still cannot
                # rename the predicate or alter provenance, and a third
                # independent verification remains mandatory.  This is useful
                # when the compiler preserved two endpoint-role alternatives
                # even though the frozen evidence supports one stable binary
                # projection.
                feedback_repaired, feedback_audit = repair(
                    repaired, excerpts, index,
                    verifier_feedback=second.get("reason") or "",
                )
                repair_audit["feedback_repair"] = feedback_audit
                if feedback_repaired is None:
                    continue
                third = verify(
                    feedback_repaired, excerpts, index,
                    phase="feedback_reverification",
                )
                feedback_audit["reverification"] = third
                if third["verdict"] != "SUPPORTED":
                    continue
                repaired = feedback_repaired
                second = third
            repaired["_policy_verification"] = second
            repaired["_policy_repair"] = repair_audit
            candidates[index - 1] = repaired

        current_relations = [
            item for item in (current_schema or {}).get("relation_types") or []
            if isinstance(item, Mapping)
        ]
        if current_relations:
            current_relation_names = {
                str(item.get("name") or "") for item in current_relations
            }
            for candidate in candidates:
                candidate_name = str(candidate.get("name") or "")
                if candidate_name not in current_relation_names:
                    continue
                candidate["_policy_novelty"] = {
                    "verdict": "REDUNDANT",
                    "reason": "active_schema_already_contains_identifier",
                    "overlapping_relation": candidate_name,
                    "exact_role_alignment": False,
                    "core_contract_equivalent": False,
                    "orientation_alignment": "none",
                    "verification_mode": "deterministic_identifier_guard",
                }

            def eligible_comparison_policy(candidate: dict[str, Any]) -> bool:
                """Admit only structurally viable candidates to this-run dedup."""

                source_types = list(candidate.get("source_types") or [])
                target_types = list(candidate.get("target_types") or [])
                domain_types = list(candidate.get("domain_types") or [])
                range_types = list(candidate.get("range_types") or [])
                evidence_supported, _ = self._evidence_supported(
                    candidate, sample_by_key,
                )
                return bool(
                    _valid_identifier(str(candidate.get("name") or ""))
                    and not structural_issues(candidate)
                    and source_types and target_types
                    and source_types == domain_types
                    and target_types == range_types
                    and coerce_confidence(candidate.get("confidence"), 0.0)
                    >= Config.schema_review_min_confidence
                    and evidence_supported
                )

            def batch_verify_novelty() -> dict[int, dict[str, Any]]:
                current_names = {
                    str(item.get("name") or "")
                    for item in current_relations
                }
                eligible = [
                    (index, candidate)
                    for index, candidate in enumerate(candidates)
                    if str(
                        (candidate.get("_policy_verification") or {}).get(
                            "verdict"
                        ) or ""
                    ).upper() == "SUPPORTED"
                    and eligible_comparison_policy(candidate)
                    and str(candidate.get("name") or "")
                    not in current_names
                ]
                if len(eligible) < 2:
                    return {}
                indexes = [index for index, _candidate in eligible]
                response_schema = {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "decisions": {
                            "type": "array",
                            "maxItems": len(eligible),
                            "items": {
                                "type": "string",
                                "enum": [
                                    "NOVEL", "REDUNDANT", "UNCERTAIN",
                                ],
                            },
                        },
                    },
                    "required": ["decisions"],
                }
                response_schema["properties"]["decisions"][
                    "minItems"
                ] = len(indexes)
                compact_fields = (
                    "name", "semantic_definition", "domain_types",
                    "range_types", "direction_description",
                    "subject_role_description", "object_role_description",
                    "symmetric", "canonical_orientation", "positive_examples",
                    "negative_examples",
                )
                comparison_policies_by_index: dict[
                    int, list[dict[str, Any]]
                ] = {}

                def compact_policy(item: Mapping[str, Any]) -> dict[str, Any]:
                    return {
                        key: copy.deepcopy(item.get(key))
                        for key in compact_fields
                    }

                def policy_signature(item: Mapping[str, Any]) -> str:
                    return "\n".join(
                        str(item.get(key) or "")
                        for key in (
                            "name", "semantic_definition",
                            "subject_role_description",
                            "object_role_description",
                            "direction_description",
                        )
                    )

                embed = getattr(self.ollama, "embed", None)
                if callable(embed):
                    try:
                        comparison_items = [
                            *current_relations,
                            *(candidate for _index, candidate in eligible),
                        ]
                        vectors = embed(
                            [policy_signature(item) for item in comparison_items],
                            project.get("embedding_model")
                            or Config.default_embedding_model,
                        )
                        if len(vectors) != len(comparison_items):
                            raise RuntimeError(
                                "schema novelty embedding count mismatch"
                            )
                        current_count = len(current_relations)
                        current_vectors = vectors[:current_count]
                        candidate_vectors = vectors[current_count:]
                        for position, (candidate_index, _candidate) in enumerate(
                            eligible
                        ):
                            candidate_vector = candidate_vectors[position]
                            active_ranked = sorted(
                                range(current_count),
                                key=lambda item_index: _cosine(
                                    candidate_vector,
                                    current_vectors[item_index],
                                ),
                                reverse=True,
                            )[: min(6, current_count)]
                            prior_ranked = sorted(
                                range(position),
                                key=lambda prior_position: _cosine(
                                    candidate_vector,
                                    candidate_vectors[prior_position],
                                ),
                                reverse=True,
                            )[:3]
                            comparison_policies_by_index[candidate_index] = [
                                *(compact_policy(current_relations[item_index])
                                  for item_index in active_ranked),
                                *(compact_policy(eligible[prior_position][1])
                                  for prior_position in prior_ranked),
                            ]
                    except Exception as exc:
                        LOGGER.warning(
                            "Schema novelty shortlist embeddings unavailable: %s",
                            exc,
                        )
                if not comparison_policies_by_index:
                    prior_candidates: list[dict[str, Any]] = []
                    for candidate_index, candidate in eligible:
                        comparison_policies_by_index[candidate_index] = [
                            *(compact_policy(item) for item in current_relations),
                            *(compact_policy(item) for item in prior_candidates),
                        ]
                        prior_candidates.append(candidate)
                for candidate_index, comparison_policies in (
                    comparison_policies_by_index.items()
                ):
                    candidates[candidate_index][
                        "_novelty_comparison_policies"
                    ] = copy.deepcopy(comparison_policies)
                best_decisions: dict[int, dict[str, Any]] = {}

                def add_novelty_decision(
                    decisions: dict[int, dict[str, Any]],
                    candidate_index: Any,
                    raw_verdict: Any,
                    item: Mapping[str, Any] | None = None,
                    *, mode: str,
                ) -> None:
                    if (
                        not isinstance(candidate_index, int)
                        or isinstance(candidate_index, bool)
                        or candidate_index not in indexes
                        or candidate_index in decisions
                    ):
                        return
                    verdict = str(raw_verdict or "").strip().upper()
                    if not verdict and item:
                        reason_prefix = str(
                            item.get("reason")
                            or item.get("explanation") or ""
                        )
                        match = re.match(
                            r"^\s*(NOVEL|REDUNDANT|UNCERTAIN|NEW|UNIQUE|"
                            r"DUPLICATE|EQUIVALENT)\b",
                            reason_prefix, flags=re.IGNORECASE,
                        )
                        if match:
                            verdict = match.group(1).upper()
                    verdict = {
                        "NEW": "NOVEL", "UNIQUE": "NOVEL",
                        "DUPLICATE": "REDUNDANT",
                        "EQUIVALENT": "REDUNDANT",
                    }.get(verdict, verdict)
                    if verdict not in {"NOVEL", "REDUNDANT", "UNCERTAIN"}:
                        return
                    payload = item or {}
                    reason = str(
                        payload.get("reason")
                        or payload.get("explanation") or ""
                    ).strip()
                    overlap = str(
                        payload.get("overlapping_relation") or ""
                    )
                    exact_roles = payload.get(
                        "exact_role_alignment"
                    ) is True
                    equivalent = payload.get(
                        "core_contract_equivalent"
                    ) is True
                    orientation = str(
                        payload.get("orientation_alignment") or "none"
                    ).lower()
                    if verdict == "REDUNDANT" and not (
                        overlap in current_names
                        and exact_roles and equivalent
                        and orientation in {"same", "inverse"}
                    ):
                        # A compact duplicate label without the existing
                        # aligned contract is not safe enough to suppress a
                        # source-grounded new predicate.
                        return
                    if verdict in {"NOVEL", "UNCERTAIN"}:
                        overlap = ""
                        exact_roles = False
                        equivalent = False
                        orientation = "none"
                    if not reason:
                        reason = f"model_compact_batch_novelty_{verdict.lower()}"
                    decisions[candidate_index] = {
                        "verdict": verdict,
                        "reason": reason[:2000],
                        "overlapping_relation": overlap,
                        "exact_role_alignment": exact_roles,
                        "core_contract_equivalent": equivalent,
                        "orientation_alignment": orientation,
                        "verification_mode": mode,
                    }

                for attempt in range(2):
                    try:
                        value = self.ollama.json_chat(
                            model=model,
                            system=(
                                "你是关系 Schema 的批量独立语义去重核验器。只比较核心"
                                "语义、固定主宾角色、方向、类型及正反例。泛化关系不能替代"
                                "具有明确动作或职责语义的具体关系；类型相容也不等于角色"
                                "等价。只有核心契约、角色及方向全部等价时才可 REDUNDANT，"
                                "每个候选只与其 comparison_policies 检索短名单比较；短名单"
                                "中没有完整等价策略时必须为 NOVEL，而不是因为存在相关策略"
                                "就返回 UNCERTAIN。只有候选或短名单缺少核心契约字段时才用"
                                "UNCERTAIN。不得修改候选。按"
                                "candidates 输入顺序返回等长 decisions 字符串数组，每项"
                                "只能是 NOVEL、REDUNDANT 或 UNCERTAIN；不要返回理由、"
                                "索引、对象或分析。REDUNDANT 会由后续单项核验补齐重叠证明。"
                                + (
                                    " 协议重试：只返回一个等长的 decisions 字符串数组。"
                                    if attempt else ""
                                )
                            ),
                            prompt=json.dumps({
                                "candidates": [{
                                    "candidate_index": index,
                                    **{
                                        key: copy.deepcopy(candidate.get(key))
                                        for key in compact_fields
                                    },
                                    "comparison_policies": (
                                        comparison_policies_by_index.get(
                                            index, []
                                        )
                                    ),
                                } for index, candidate in eligible],
                            }, ensure_ascii=False),
                            schema=response_schema,
                            temperature=0,
                            context_budget_tokens=int(
                                project.get("model_max_tokens") or 32768
                            ),
                            max_output_tokens=4096,
                        )
                        raw_decisions = (
                            value.get("decisions")
                            or value.get("items")
                            or value.get("verdicts")
                            if isinstance(value, Mapping) else None
                        )
                        decisions: dict[int, dict[str, Any]] = {}
                        if isinstance(raw_decisions, list):
                            positional = len(raw_decisions) == len(indexes)
                            for position, item in enumerate(raw_decisions):
                                if isinstance(item, Mapping):
                                    candidate_index = item.get(
                                        "candidate_index"
                                    )
                                    if candidate_index is None and positional:
                                        candidate_index = indexes[position]
                                    add_novelty_decision(
                                        decisions, candidate_index,
                                        item.get("verdict")
                                        or item.get("decision")
                                        or item.get("status"),
                                        item, mode="batch",
                                    )
                                elif positional:
                                    add_novelty_decision(
                                        decisions, indexes[position], item,
                                        mode="batch_positional",
                                    )
                        elif isinstance(raw_decisions, Mapping):
                            for raw_index, item in raw_decisions.items():
                                match = re.search(r"\d+", str(raw_index))
                                candidate_index = (
                                    int(match.group()) if match else None
                                )
                                if isinstance(item, Mapping):
                                    add_novelty_decision(
                                        decisions, candidate_index,
                                        item.get("verdict")
                                        or item.get("decision")
                                        or item.get("status"),
                                        item, mode="batch_mapping",
                                    )
                                else:
                                    add_novelty_decision(
                                        decisions, candidate_index, item,
                                        mode="batch_mapping",
                                    )
                        if isinstance(value, Mapping):
                            for raw_key, raw_value in value.items():
                                if re.fullmatch(r"\d+", str(raw_key)):
                                    candidate_index = int(str(raw_key))
                                    if isinstance(raw_value, Mapping):
                                        raw_verdict = (
                                            raw_value.get("verdict")
                                            or raw_value.get("decision")
                                            or raw_value.get("status")
                                        )
                                        payload = raw_value
                                    else:
                                        raw_verdict = raw_value
                                        payload = None
                                    add_novelty_decision(
                                        decisions, candidate_index,
                                        raw_verdict, payload,
                                        mode="batch_numeric_mapping",
                                    )
                                    continue
                                match = re.fullmatch(
                                    r"candidate[_-]?(\d+)[_-]"
                                    r"(?:decision|verdict|status)",
                                    str(raw_key), flags=re.IGNORECASE,
                                )
                                if not match:
                                    continue
                                payload = {
                                    "reason": value.get(
                                        f"candidate_{match.group(1)}_reason"
                                    ) or "",
                                }
                                add_novelty_decision(
                                    decisions, int(match.group(1)), raw_value,
                                    payload, mode="batch_flattened",
                                )
                        best_decisions.update(decisions)
                        if set(best_decisions) == set(indexes):
                            return best_decisions
                    except Exception as exc:
                        LOGGER.warning(
                            "Schema batch novelty attempt failed: %s", exc,
                        )
                return best_decisions

            for candidate_index, novelty in batch_verify_novelty().items():
                candidates[candidate_index]["_policy_novelty"] = novelty

            for index, candidate in enumerate(candidates, start=1):
                policy = candidate.get("_policy_verification") or {}
                if str(policy.get("verdict") or "").upper() != "SUPPORTED":
                    continue
                precomputed_novelty = candidate.get("_policy_novelty")
                if isinstance(precomputed_novelty, Mapping) and str(
                    precomputed_novelty.get("verdict") or ""
                ) in {"NOVEL", "REDUNDANT", "UNCERTAIN"}:
                    if (
                        precomputed_novelty.get("verdict") == "NOVEL"
                        and eligible_comparison_policy(candidate)
                    ):
                        current_relations.append({
                            key: copy.deepcopy(value)
                            for key, value in candidate.items()
                            if not str(key).startswith("_")
                            and key not in {
                                "evidence_chunks", "evidence_quotes",
                                "confidence",
                            }
                        })
                    continue
                novelty: dict[str, str] = {
                    "verdict": "UNCERTAIN",
                    "reason": "novelty_verifier_missing_result",
                    "overlapping_relation": "",
                }
                try:
                    for protocol_attempt in range(2):
                        try:
                            value = self.ollama.json_chat(
                                model=model,
                                system=(
                                    "你是知识图谱关系 Schema 的独立语义去重核验器。不得根据"
                                    "谓词名称判断，只比较 semantic_definition、主宾语角色、"
                                    "domain/range、方向、正反例及逐字证据。若候选只是改名、"
                                    "同义改写，或仅收窄现有关系而没有改变稳定端点角色和核心"
                                    "关系契约，返回 REDUNDANT，并指出现有关系。通用参与关系"
                                    "不能替代一个具有明确动作、状态或职责语义的具体关系；类型"
                                    "范围相容也不能证明角色相同。尤其不得把动作的接收方、动作"
                                    "产物、被使用资源、地点或结果状态视为同一种宾语角色。"
                                    "REDUNDANT 必须给出当前 Schema 中确切存在的 overlapping_relation，"
                                    "并同时满足 core_contract_equivalent=true、"
                                    "exact_role_alignment=true；orientation_alignment 必须说明端点"
                                    "按相同方向对齐还是按逆向对齐。只要稳定角色或核心动作不同，"
                                    "exact_role_alignment 或 core_contract_equivalent 必须为 false。"
                                    "只有现有 Schema"
                                    "无法用相同方向和端点角色准确表示该证据语义时才返回 NOVEL。"
                                    "无法判断返回 UNCERTAIN。必须给出非空 reason；不修改候选。"
                                    + (
                                        " 协议重试：必须返回有效 verdict、非空 reason 和"
                                        " overlapping_relation（无重叠时为 null）。"
                                        if protocol_attempt else ""
                                    )
                                ),
                                prompt=json.dumps({
                                    "candidate": {
                                        **{
                                            key: copy.deepcopy(candidate.get(key))
                                            for key in (
                                                "name", "semantic_definition",
                                                "domain_types", "range_types",
                                                "direction_description",
                                                "subject_role_description",
                                                "object_role_description", "symmetric",
                                                "canonical_orientation",
                                                "positive_examples",
                                                "negative_examples",
                                            )
                                        },
                                        "validated_observations": [{
                                            key: copy.deepcopy(
                                                observation.get(key)
                                            )
                                            for key in (
                                                "subject_surface",
                                                "relation_expression",
                                                "object_surface",
                                                "evidence_quote",
                                                "ordered_role_projection",
                                                "structural_evidence_kind",
                                            )
                                            if observation.get(key)
                                            not in (None, "")
                                        } for observation in (
                                            candidate.get(
                                                "_induction_observations"
                                            ) or []
                                        )[:3]],
                                    },
                                    "current_relation_policies": (
                                        candidate.get(
                                            "_novelty_comparison_policies"
                                        ) or current_relations
                                    ),
                                    "protocol_attempt": protocol_attempt + 1,
                                }, ensure_ascii=False),
                                schema=SCHEMA_POLICY_NOVELTY_FORMAT,
                                temperature=0,
                                context_budget_tokens=int(
                                    project.get("model_max_tokens") or 32768
                                ),
                                max_output_tokens=(
                                    1024 if protocol_attempt == 0 else 2048
                                ),
                            )
                        except Exception:
                            if protocol_attempt == 0:
                                continue
                            raise
                        if not isinstance(value, Mapping):
                            continue
                        verdict = str(
                            value.get("verdict") or value.get("decision") or ""
                        ).strip().upper()
                        reason = str(
                            value.get("reason") or value.get("explanation") or ""
                        ).strip()
                        overlap = value.get("overlapping_relation")
                        exact_role_alignment = value.get(
                            "exact_role_alignment"
                        ) is True
                        core_contract_equivalent = value.get(
                            "core_contract_equivalent"
                        ) is True
                        orientation_alignment = str(
                            value.get("orientation_alignment")
                            or ("none" if verdict != "REDUNDANT" else "")
                        ).strip().lower()
                        current_names = {
                            str(item.get("name") or "")
                            for item in current_relations
                        }
                        redundant_is_consistent = verdict != "REDUNDANT" or (
                                bool(overlap)
                                and str(overlap) in current_names
                                and exact_role_alignment
                                and core_contract_equivalent
                                and orientation_alignment in {"same", "inverse"}
                        )
                        if (
                            verdict in {"NOVEL", "REDUNDANT", "UNCERTAIN"}
                            and reason and redundant_is_consistent
                            and orientation_alignment in {"same", "inverse", "none"}
                        ):
                            novelty = {
                                "verdict": verdict,
                                "reason": reason[:2000],
                                "overlapping_relation": str(overlap or ""),
                                "exact_role_alignment": exact_role_alignment,
                                "core_contract_equivalent": core_contract_equivalent,
                                "orientation_alignment": orientation_alignment,
                                "verification_mode": "single_shortlist",
                            }
                            break
                except Exception as exc:
                    message = f"candidate {index} novelty verification: {exc}"
                    errors.append(message)
                    LOGGER.warning("Schema novelty verifier failed: %s", message)
                    novelty = {
                        "verdict": "UNCERTAIN",
                        "reason": f"novelty_error:{type(exc).__name__}",
                        "overlapping_relation": "",
                    }
                candidate["_policy_novelty"] = novelty
                if (
                    novelty.get("verdict") == "NOVEL"
                    and eligible_comparison_policy(candidate)
                ):
                    current_relations.append({
                        key: copy.deepcopy(value)
                        for key, value in candidate.items()
                        if not str(key).startswith("_")
                        and key not in {"evidence_chunks", "evidence_quotes", "confidence"}
                    })
        return review, errors

    @staticmethod
    def _source_backed_example_count(
        item: dict[str, Any], sample_map: dict[str, dict[str, Any]],
    ) -> int:
        texts = [
            str(sample_map[key].get("_schema_text") or sample_map[key].get("content") or "")
            for key in item.get("evidence_chunks") or []
            if key in sample_map
        ]
        examples = {
            re.sub(r"\s+", " ", str(value)).strip()
            for value in (
                item.get("evidence_quotes")
                or item.get("positive_examples")
                or []
            )
            if str(value).strip()
        }
        return sum(
            1 for example in examples
            if any(example in re.sub(r"\s+", " ", text) for text in texts)
        )

    @staticmethod
    def _evidence_supported(item: dict[str, Any], sample_map: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
        evidence = list(dict.fromkeys(
            str(value).strip() for value in item.get("evidence_chunks") or []
            if str(value).strip() in sample_map
        ))
        confidence = coerce_confidence(item.get("confidence"), 0.0)
        exact_examples = SchemaEvolutionService._source_backed_example_count(
            item, sample_map,
        )
        if exact_examples:
            # LLM self-confidence is optional transport metadata.  An exact
            # frozen quote meets the configured single-evidence gate; additional
            # quotes improve provenance breadth but do not inflate confidence.
            evidence_confidence = Config.schema_review_single_evidence_confidence
            confidence = max(confidence, evidence_confidence)
            item["confidence"] = confidence
        if (
            len(evidence) == 1
            and exact_examples >= 1
            and confidence >= Config.schema_review_single_evidence_confidence
        ):
            return True, evidence
        if (
            len(evidence) == 1
            and exact_examples >= 2
            and confidence >= Config.schema_review_min_confidence
        ):
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
        *,
        knowledge_language: str = CHINESE,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        updated = copy.deepcopy(current)
        entity_names = {str(item["name"]) for item in updated.get("entity_types", [])}
        relation_names = {str(item["name"]) for item in updated.get("relation_types", [])}
        sample_map = {_chunk_key(item): item for item in samples}
        accepted_entities: list[dict[str, Any]] = []
        accepted_relations: list[dict[str, Any]] = []
        rejected: dict[str, Any] = {
            "entity_types": 0, "relation_types": 0, "details": [],
        }

        for item in review.get("entity_type_candidates") or []:
            name = str(item.get("name") or "").strip()
            supported, evidence = self._evidence_supported(item, sample_map)
            confidence = coerce_confidence(item.get("confidence"), 0.0)
            reasons = []
            if len(accepted_entities) >= Config.schema_max_entity_additions:
                reasons.append("addition_limit")
            if confidence < Config.schema_review_min_confidence:
                reasons.append("confidence_below_threshold")
            if not _valid_identifier(name):
                reasons.append("invalid_identifier")
            if _type_name_exists(name, entity_names):
                reasons.append("already_exists")
            if not supported:
                reasons.append("source_evidence_insufficient")
            if reasons:
                rejected["entity_types"] += 1
                rejected["details"].append({
                    "kind": "entity_type", "name": name,
                    "reasons": reasons, "confidence": confidence,
                    "evidence_chunks": evidence,
                })
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
        active_relation_names = set(relation_names)
        proposed_relation_names = {
            str(item.get("name") or "").strip()
            for item in review.get("relation_type_candidates") or []
            if isinstance(item, Mapping)
            and _valid_identifier(str(item.get("name") or "").strip())
        }
        for item in review.get("relation_type_candidates") or []:
            name = str(item.get("name") or "").strip()
            raw_sources = list(dict.fromkeys(
                str(value) for value in item.get("source_types") or [] if str(value)
            ))
            raw_targets = list(dict.fromkeys(
                str(value) for value in item.get("target_types") or [] if str(value)
            ))
            raw_domains = list(dict.fromkeys(
                str(value) for value in item.get("domain_types") or [] if str(value)
            ))
            raw_ranges = list(dict.fromkeys(
                str(value) for value in item.get("range_types") or [] if str(value)
            ))
            sources = list(dict.fromkeys(
                str(value) for value in raw_sources
                if str(value) in allowed_types
            ))
            targets = list(dict.fromkeys(
                str(value) for value in raw_targets
                if str(value) in allowed_types
            ))
            domains = list(dict.fromkeys(
                str(value) for value in raw_domains
                if str(value) in allowed_types
            ))
            ranges = list(dict.fromkeys(
                str(value) for value in raw_ranges
                if str(value) in allowed_types
            ))
            semantic_definition = str(
                item.get("semantic_definition") or ""
            ).strip()[:1000]
            direction_description = str(
                item.get("direction_description") or ""
            ).strip()[:1000]
            subject_role_description = str(
                item.get("subject_role_description") or ""
            ).strip()[:1000]
            object_role_description = str(
                item.get("object_role_description") or ""
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
            structural_evidence_kinds = item.get(
                "structural_evidence_kinds", []
            )
            structural_role_labels = item.get(
                "structural_role_labels", []
            )
            source_role_pairs = item.get("source_role_pairs", [])
            conflicts_with = item.get("conflicts_with", [])
            negation_policy = str(
                item.get("negation_policy") or "reject_negated"
            ).strip()
            modality_policy = str(
                item.get("modality_policy") or "require_asserted"
            ).strip()
            policy_verification = (
                item.get("_policy_verification")
                if isinstance(item.get("_policy_verification"), dict) else {}
            )
            policy_verdict = str(
                policy_verification.get("verdict") or ""
            ).strip().upper()
            policy_novelty = (
                item.get("_policy_novelty")
                if isinstance(item.get("_policy_novelty"), dict) else {}
            )
            novelty_verdict = str(
                policy_novelty.get("verdict") or ""
            ).strip().upper()
            supported, evidence = self._evidence_supported(item, sample_map)
            confidence = coerce_confidence(item.get("confidence"), 0.0)
            reasons = []
            if len(accepted_relations) >= Config.schema_max_relation_additions:
                reasons.append("addition_limit")
            if confidence < Config.schema_review_min_confidence:
                reasons.append("confidence_below_threshold")
            if not _valid_identifier(name):
                reasons.append("invalid_identifier")
            if _type_name_exists(name, relation_names):
                reasons.append("already_exists")
            if not sources:
                reasons.append("missing_source_types")
            if not targets:
                reasons.append("missing_target_types")
            if any(value not in allowed_types for value in [
                *raw_sources, *raw_targets, *raw_domains, *raw_ranges,
            ]):
                reasons.append("unknown_declared_type")
            if sources != domains:
                reasons.append("source_domain_mismatch")
            if targets != ranges:
                reasons.append("target_range_mismatch")
            for missing, value in (
                ("missing_semantic_definition", semantic_definition),
                ("missing_direction_description", direction_description),
                ("missing_subject_role_description", subject_role_description),
                ("missing_object_role_description", object_role_description),
            ):
                if not value:
                    reasons.append(missing)
            if any(not _schema_user_text_matches(
                value, knowledge_language,
            ) for value in (
                str(item.get("description") or semantic_definition),
                semantic_definition,
                direction_description,
                subject_role_description,
                object_role_description,
            )):
                reasons.append("invalid_user_facing_language")
            if not isinstance(symmetric, bool):
                reasons.append("invalid_symmetric")
            if not isinstance(reflexive, bool):
                reasons.append("invalid_reflexive")
            if canonical_orientation not in {"forward", "inverse", "symmetric"}:
                reasons.append("invalid_canonical_orientation")
            if publish_mode not in {"auto", "review_only", "disabled"}:
                reasons.append("invalid_publish_mode")
            if negation_policy not in {
                "reject_negated", "allow_explicit_negative_relation",
            }:
                reasons.append("invalid_negation_policy")
            if modality_policy != "require_asserted":
                reasons.append("invalid_modality_policy")
            if isinstance(symmetric, bool) and symmetric != (
                canonical_orientation == "symmetric"
            ):
                reasons.append("symmetric_orientation_mismatch")
            if canonical_orientation == "inverse" and (
                not inverse_predicate or inverse_predicate not in relation_names
            ):
                reasons.append("invalid_inverse_orientation")
            if symmetric is True and bool(inverse_predicate):
                reasons.append("symmetric_inverse_conflict")
            if inverse_predicate and inverse_predicate not in relation_names:
                reasons.append("unknown_inverse_predicate")
            for invalid, field in (
                (
                    not isinstance(positive_examples, list)
                    or not all(isinstance(value, str) for value in positive_examples or []),
                    "invalid_positive_examples",
                ),
                (
                    not isinstance(negative_examples, list)
                    or not all(isinstance(value, str) for value in negative_examples or []),
                    "invalid_negative_examples",
                ),
                (
                    not isinstance(optional_soft_cues, list)
                    or not all(isinstance(value, str) for value in optional_soft_cues or []),
                    "invalid_optional_soft_cues",
                ),
                (
                    not isinstance(structural_evidence_kinds, list)
                    or not all(
                        isinstance(value, str)
                        for value in structural_evidence_kinds or []
                    ),
                    "invalid_structural_evidence_kinds",
                ),
                (
                    not isinstance(structural_role_labels, list)
                    or not all(
                        isinstance(value, str)
                        for value in structural_role_labels or []
                    ),
                    "invalid_structural_role_labels",
                ),
                (
                    not isinstance(source_role_pairs, list)
                    or not all(
                        isinstance(value, str)
                        and value in {
                            "grammatical_subject->action_theme",
                            "action_theme->role_complement",
                            "grammatical_subject->role_complement",
                            "grammatical_subject->direct_argument",
                            "direct_argument->grammatical_subject",
                            "action_theme->grammatical_subject",
                            "role_complement->grammatical_subject",
                            "collective_participant->collective_participant",
                            "possessor->possessed",
                            "possessed->possessor",
                        }
                        for value in source_role_pairs or []
                    ),
                    "invalid_source_role_pairs",
                ),
                (
                    not isinstance(conflicts_with, list)
                    or not all(
                        isinstance(value, str)
                        and _valid_identifier(value.strip())
                        for value in conflicts_with or []
                    ),
                    "invalid_conflicts_with",
                ),
            ):
                if invalid:
                    reasons.append(field)
            normalized_conflicts = list(dict.fromkeys(
                value.strip() for value in conflicts_with or []
                if isinstance(value, str) and value.strip()
            )) if isinstance(conflicts_with, list) else []
            if name in normalized_conflicts:
                reasons.append("self_conflicting_predicate")
            if any(
                value not in active_relation_names | proposed_relation_names
                for value in normalized_conflicts
            ):
                reasons.append("unknown_conflicting_predicate")
            if not supported:
                reasons.append("source_evidence_insufficient")
            if policy_verdict != "SUPPORTED":
                reasons.append(
                    "policy_verification_rejected"
                    if policy_verdict == "REJECTED"
                    else "policy_verification_uncertain"
                )
            # Production review against a non-empty active Schema always
            # attaches this independent audit.  Candidate acceptance remains
            # fail-closed if the comparison did not converge; direct legacy
            # callers without a novelty audit keep their historical contract.
            if policy_novelty and novelty_verdict != "NOVEL":
                reasons.append(
                    "semantic_redundancy"
                    if novelty_verdict == "REDUNDANT"
                    else "novelty_verification_uncertain"
                )
            if reasons:
                rejected["relation_types"] += 1
                rejected["details"].append({
                    "kind": "relation_type", "name": name,
                    "reasons": list(dict.fromkeys(reasons)),
                    "confidence": confidence,
                    "evidence_chunks": evidence,
                    "policy_verification": policy_verification,
                    "policy_novelty": policy_novelty,
                })
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
                "subject_role_description": subject_role_description,
                "object_role_description": object_role_description,
                "symmetric": symmetric,
                "inverse_predicate": inverse_predicate or None,
                "canonical_orientation": canonical_orientation,
                "publish_mode": publish_mode,
                "reflexive": reflexive,
                "positive_examples": list(positive_examples),
                "negative_examples": list(negative_examples),
                "optional_soft_cues": list(optional_soft_cues),
                "structural_evidence_kinds": list(
                    structural_evidence_kinds
                ),
                "structural_role_labels": list(structural_role_labels),
                "source_role_pairs": list(dict.fromkeys(source_role_pairs)),
                "conflicts_with": normalized_conflicts,
                "negation_policy": negation_policy,
                "modality_policy": modality_policy,
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

        # A candidate may reference another candidate from the same review,
        # but an active Schema must never retain a dangling conflict link when
        # that peer failed another acceptance gate.  Resolve this transitively
        # after all candidates have been judged, then remove only this review's
        # newly appended policies.
        while True:
            accepted_names = {
                str(item.get("name") or "") for item in accepted_relations
            }
            available_conflict_names = active_relation_names | accepted_names
            dangling = [
                item for item in accepted_relations
                if any(
                    peer not in available_conflict_names
                    for peer in item.get("conflicts_with") or []
                )
            ]
            if not dangling:
                break
            dangling_names = {
                str(item.get("name") or "") for item in dangling
            }
            accepted_relations[:] = [
                item for item in accepted_relations
                if str(item.get("name") or "") not in dangling_names
            ]
            updated["relation_types"] = [
                item for item in updated.get("relation_types") or []
                if not (
                    str(item.get("name") or "") in dangling_names
                    and str(item.get("name") or "") not in active_relation_names
                )
            ]
            for item in dangling:
                name = str(item.get("name") or "")
                relation_names.discard(name)
                rejected["relation_types"] += 1
                rejected["details"].append({
                    "kind": "relation_type",
                    "name": name,
                    "reasons": ["dangling_conflicting_predicate"],
                    "confidence": item.get("confidence"),
                    "evidence_chunks": list(item.get("evidence_chunks") or []),
                    "conflicts_with": list(item.get("conflicts_with") or []),
                })
        return updated, accepted_entities, accepted_relations, rejected

    def evaluate_document(
        self,
        project: dict[str, Any],
        document_id: int,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        active = self.ensure_base(int(project["id"]))

        def finish(
            record: dict[str, Any], evaluation: dict[str, Any], *, outcome: str,
        ) -> dict[str, Any]:
            value = self._evaluated(record, evaluation)
            recorder = getattr(self.db, "record_schema_evaluation", None)
            if callable(recorder):
                recorder(
                    int(project["id"]), int(document_id),
                    prior_schema_version=int(active["version"]),
                    resulting_schema_version=int(record["version"]),
                    outcome=outcome,
                    evaluation=evaluation,
                )
            return value

        if not Config.schema_auto_apply or not chunks:
            return finish(
                active, {"reviewed": False, "reason": "disabled_or_empty"},
                outcome="skipped",
            )

        samples, sampling = self.select_representative_chunks(project, document_id, chunks)
        if not samples:
            return finish(
                active,
                {"reviewed": False, "reason": "no_representative_chunks", "sampling": sampling},
                outcome="skipped",
            )
        should_review, trigger = self._should_review(project, document_id, samples)
        if not should_review:
            return finish(active, {
                "reviewed": False,
                "reason": trigger["reason"],
                "sampling": sampling,
                "trigger": trigger,
            }, outcome="skipped")

        current = active["schema_json"]
        batches, errors = self._discover_candidates(project, current, samples)
        historical = self._historical_relation_candidate_batch(
            int(project["id"]), document_id, samples,
        )
        if historical:
            batches.append(historical)
        review = self._consolidate_candidates(project, current, batches)
        review, verification_errors = self._verify_relation_policies(
            project, review, samples, current,
        )
        errors.extend(verification_errors)
        updated, accepted_entities, accepted_relations, rejected = self._apply_candidates(
            current, review, samples,
            knowledge_language=_schema_knowledge_language(project, samples),
        )
        evaluation = {
            "schema_discovery_protocol_version": (
                SCHEMA_DISCOVERY_PROTOCOL_VERSION
            ),
            "reviewed": True,
            "reason": trigger["reason"],
            "sampling": sampling,
            "trigger": trigger,
            "batch_count": len(batches),
            "batch_errors": errors,
            "proposed_entity_types": list(review.get("entity_type_candidates") or []),
            "proposed_relation_types": list(review.get("relation_type_candidates") or []),
            "accepted_entity_types": accepted_entities,
            "accepted_relation_types": accepted_relations,
            "rejected_candidates": rejected,
        }
        if not accepted_entities and not accepted_relations:
            return finish(
                active, evaluation,
                outcome="failed" if errors and not batches else "no_change",
            )

        report = {
            "summary": review.get("summary") or "",
            **evaluation,
            "policy": "representative-sampling+evidence-gated+additive-only",
        }
        created = self.db.create_schema_version(int(project["id"]), updated, report, [document_id])
        return finish(created, evaluation, outcome="applied")

    def evaluate_corpus(
        self,
        project: dict[str, Any],
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate one immutable document batch and publish at most one Schema.

        The active Schema is read once, all document evidence is sampled under
        one global budget, and the additive limits are applied once.  This
        removes upload-order dependence and prevents every document from
        independently consuming another full set of narrow predicates.
        """

        project_id = int(project["id"])
        active = self.ensure_base(project_id)
        ordered_documents = sorted(
            {
                int(item["id"]): dict(item)
                for item in documents
                if item.get("id") is not None
            }.values(),
            key=lambda item: int(item["id"]),
        )
        document_ids = [int(item["id"]) for item in ordered_documents]
        fingerprint_payload = {
            "protocol": SCHEMA_DISCOVERY_PROTOCOL_VERSION,
            "knowledge_language_setting": normalize_knowledge_language(
                project.get("knowledge_language") or "auto",
            ),
            "base_schema_profile": str(BASE_SCHEMA.get("version") or ""),
            "review_batch_size": int(
                Config.schema_corpus_review_batch_size
            ),
            "sample_ceiling_chunks": int(
                Config.schema_sample_ceiling_chunks
            ),
            "sample_ceiling_chars": int(
                Config.schema_sample_ceiling_chars
            ),
            "documents": [{
                "document_id": int(item["id"]),
                "revision_id": str(item.get("active_revision_id") or ""),
                "content_hash": str(item.get("doc_hash") or ""),
            } for item in ordered_documents],
        }
        input_fingerprint = hashlib.sha256(json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

        def record(
            result: dict[str, Any],
            evaluation: dict[str, Any],
            *,
            outcome: str,
        ) -> dict[str, Any]:
            value = self._evaluated(result, evaluation)
            recorder = getattr(self.db, "record_schema_evaluation", None)
            if callable(recorder):
                for document_id in document_ids:
                    recorder(
                        project_id, document_id,
                        prior_schema_version=int(active["version"]),
                        resulting_schema_version=int(result["version"]),
                        outcome=outcome,
                        evaluation=evaluation,
                    )
            return value

        if not Config.schema_auto_apply or not ordered_documents:
            return record(active, {
                "schema_discovery_protocol_version": (
                    SCHEMA_DISCOVERY_PROTOCOL_VERSION
                ),
                "scope": "project_document_batch",
                "reviewed": False,
                "reason": "disabled_or_empty",
                "input_fingerprint": input_fingerprint,
                "document_ids": document_ids,
            }, outcome="skipped")

        compatibility = (
            active.get("compatibility_report")
            if isinstance(active.get("compatibility_report"), dict) else {}
        )
        if str(compatibility.get("input_fingerprint") or "") == input_fingerprint:
            return self._evaluated(active, {
                "schema_discovery_protocol_version": (
                    SCHEMA_DISCOVERY_PROTOCOL_VERSION
                ),
                "scope": "project_document_batch",
                "reviewed": False,
                "reason": "identical_batch_already_evaluated",
                "input_fingerprint": input_fingerprint,
                "document_ids": document_ids,
            })

        # No-change evaluations do not create a graph_schemas row.  Reuse the
        # latest per-document audit only when every member was evaluated
        # together against the still-active version and immutable revisions.
        evaluator = getattr(self.db, "list_schema_evaluations", None)
        if callable(evaluator):
            latest_by_document: dict[int, dict[str, Any]] = {}
            for row in evaluator(project_id):
                document_id = int(row.get("document_id") or 0)
                if document_id in document_ids and document_id not in latest_by_document:
                    latest_by_document[document_id] = row
            reusable_audit = bool(document_ids) and all(
                document_id in latest_by_document
                and str((
                    latest_by_document[document_id].get("evaluation")
                    if isinstance(
                        latest_by_document[document_id].get("evaluation"), dict,
                    ) else {}
                ).get("input_fingerprint") or "") == input_fingerprint
                for document_id in document_ids
            )
            if reusable_audit:
                return self._evaluated(active, {
                    "schema_discovery_protocol_version": (
                        SCHEMA_DISCOVERY_PROTOCOL_VERSION
                    ),
                    "scope": "project_document_batch",
                    "reviewed": False,
                    "reason": "identical_no_change_batch_already_evaluated",
                    "input_fingerprint": input_fingerprint,
                    "document_ids": document_ids,
                })

        samples, sampling = self.select_representative_corpus_chunks(
            project, ordered_documents,
        )
        if not samples:
            return record(active, {
                "schema_discovery_protocol_version": (
                    SCHEMA_DISCOVERY_PROTOCOL_VERSION
                ),
                "scope": "project_document_batch",
                "reviewed": False,
                "reason": "no_representative_chunks",
                "input_fingerprint": input_fingerprint,
                "document_ids": document_ids,
                "sampling": sampling,
            }, outcome="skipped")

        current = active["schema_json"]
        batches, errors = self._discover_candidates(
            project,
            current,
            samples,
            batch_size=Config.schema_corpus_review_batch_size,
        )
        for document_id in document_ids:
            document_samples = [
                item for item in samples
                if int(item.get("_schema_document_id") or 0)
                == document_id
            ]
            historical = self._historical_relation_candidate_batch(
                project_id, document_id, document_samples,
            )
            if historical:
                batches.append(historical)
        review = self._consolidate_candidates(project, current, batches)
        review, verification_errors = self._verify_relation_policies(
            project, review, samples, current,
        )
        errors.extend(verification_errors)
        updated, accepted_entities, accepted_relations, rejected = (
            self._apply_candidates(
                current, review, samples,
                knowledge_language=_schema_knowledge_language(
                    project, samples,
                ),
            )
        )
        evaluation = {
            "schema_discovery_protocol_version": (
                SCHEMA_DISCOVERY_PROTOCOL_VERSION
            ),
            "scope": "project_document_batch",
            "reviewed": True,
            "reason": "new_or_changed_document_batch",
            "input_fingerprint": input_fingerprint,
            "document_ids": document_ids,
            "document_revisions": fingerprint_payload["documents"],
            "sampling": sampling,
            "batch_count": len(batches),
            "batch_errors": errors,
            "proposed_entity_types": list(
                review.get("entity_type_candidates") or []
            ),
            "proposed_relation_types": list(
                review.get("relation_type_candidates") or []
            ),
            "accepted_entity_types": accepted_entities,
            "accepted_relation_types": accepted_relations,
            "rejected_candidates": rejected,
        }
        if not accepted_entities and not accepted_relations:
            return record(
                active, evaluation,
                outcome="failed" if errors and not batches else "no_change",
            )

        report = {
            "summary": review.get("summary") or "",
            **evaluation,
            "policy": (
                "project-batch-representative-sampling+"
                "evidence-gated+additive-only"
            ),
        }
        created = self.db.create_schema_version(
            project_id, updated, report, document_ids,
        )
        return record(created, evaluation, outcome="applied")

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
            historical = self._historical_relation_candidate_batch(
                project_id, document_id, samples,
            )
            if historical:
                batches.append(historical)
            review = self._consolidate_candidates(project, current, batches)
            review, verification_errors = self._verify_relation_policies(
                project, review, samples, current,
            )
            errors.extend(verification_errors)
            updated, entities, relations, rejected = self._apply_candidates(
                current, review, samples,
                knowledge_language=_schema_knowledge_language(
                    project, samples,
                ),
            )
            current = updated
            report = {
                "schema_discovery_protocol_version": (
                    SCHEMA_DISCOVERY_PROTOCOL_VERSION
                ),
                "document_id": document_id,
                "document_name": document.get("original_filename") or "",
                "sampling": sampling,
                "proposed_entity_types": list(
                    review.get("entity_type_candidates") or []
                ),
                "proposed_relation_types": list(
                    review.get("relation_type_candidates") or []
                ),
                "accepted_entity_types": entities,
                "accepted_relation_types": relations,
                "rejected_candidates": rejected,
                "batch_errors": errors,
            }
            reports.append(report)
            if progress_callback:
                progress_callback(index, total, report)

        compatibility_report = {
            "schema_discovery_protocol_version": (
                SCHEMA_DISCOVERY_PROTOCOL_VERSION
            ),
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
    def _nearest_unique_surface_span(
        spans: list[tuple[int, int]],
        start: Any,
        end: Any,
    ) -> tuple[int, int] | None:
        """Align an imprecise generated offset to one exact source occurrence.

        Local models sometimes count Unicode offsets one or two characters
        early while still returning the exact source surface.  Rejecting every
        repeated surface in that case destroys endpoint recall.  This repair is
        deliberately lexical and provenance preserving: it never changes the
        returned surface and accepts only a uniquely nearest exact occurrence.
        A tie remains ambiguous and therefore fail-closed.
        """

        try:
            requested_start, requested_end = int(start), int(end)
        except (TypeError, ValueError):
            return None
        if requested_end <= requested_start or not spans:
            return None
        ranked = sorted(
            (
                abs(left - requested_start) + abs(right - requested_end),
                left,
                right,
            )
            for left, right in spans
        )
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            return None
        _, left, right = ranked[0]
        return left, right

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
        rejected_entity_rows: list[dict[str, Any]] = []
        raw_entity_values = list(result.get("entities") or [])
        for raw in raw_entity_values:
            if not isinstance(raw, dict):
                rejected_entity_rows.append({
                    "reason": "invalid_entity_object", "raw": {"value": str(raw)[:1000]},
                })
                continue
            raw = repair_soft_wrapped_entity_span(content, raw)
            raw = normalize_entity_span(content, raw)
            local_id = str(raw.get("id") or "").strip()
            name = re.sub(r"\s+", " ", str(raw.get("name") or "")).strip()
            if not local_id or not name:
                rejected_entities += 1
                rejected_entity_rows.append({
                    "reason": "missing_entity_identity", "raw": dict(raw),
                })
                continue
            if raw.get("standalone_referential") is False:
                placeholder_reason = non_referential_surface_reason(name)
                if placeholder_reason:
                    rejected_entities += 1
                    rejected_entity_rows.append({
                        # Keep the stable persisted category while the helper
                        # supplies the more precise deterministic decision.
                        "reason": "non_referential_placeholder",
                        "placeholder_reason": placeholder_reason,
                        "raw": dict(raw),
                    })
                    continue
                # The model hint was over-conservative for a source-backed
                # lexical noun phrase. Preserve the source span and make the
                # repair explicit in the normalized record so downstream
                # resolution can audit it without treating the phrase as a
                # pronoun or deictic placeholder.
                raw["standalone_referential"] = True
            boundary_reason = entity_boundary_reason(
                content, raw, raw_entity_values,
            )
            if boundary_reason:
                rejected_entities += 1
                rejected_entity_rows.append({
                    "reason": boundary_reason, "raw": dict(raw),
                })
                continue
            if local_id in local_entities:
                rejected_entities += 1
                rejected_entity_rows.append({
                    "reason": "duplicate_local_entity_id", "raw": dict(raw),
                })
                continue
            raw_aliases = list(dict.fromkeys(
                re.sub(r"\s+", " ", str(value)).strip()
                for value in raw.get("aliases") or [] if str(value).strip()
            ))
            name_spans = _surface_spans(content, name)
            if not name_spans:
                rejected_entities += 1
                rejected_entity_rows.append({
                    "reason": "surface_not_found", "raw": dict(raw),
                })
                continue
            aliases = [alias for alias in raw_aliases if _surface_spans(content, alias)]
            verified_surface_forms = list(dict.fromkeys([name, *aliases]))
            mention_span = self._exact_source_span(
                content, raw.get("mention_start"), raw.get("mention_end"),
                verified_surface_forms,
            )
            if mention_span is None:
                raw_evidence = str(
                    raw.get("evidence")
                    or raw.get("evidence_text")
                    or raw.get("evidence_quote") or ""
                )
                located_evidence = self._locate_evidence(content, raw_evidence)
                if located_evidence[1] is not None and located_evidence[2] is not None:
                    inside = [
                        span for span in name_spans
                        if located_evidence[1] <= span[0] < span[1] <= located_evidence[2]
                    ]
                    if len(inside) == 1:
                        mention_span = inside[0]
                if mention_span is None:
                    mention_span = self._nearest_unique_surface_span(
                        name_spans,
                        raw.get("mention_start"), raw.get("mention_end"),
                    )
                if mention_span is None and len(name_spans) != 1:
                    rejected_entities += 1
                    rejected_entity_rows.append({
                        "reason": "ambiguous_surface_without_exact_span",
                        "raw": dict(raw),
                    })
                    continue
                if mention_span is None:
                    mention_span = name_spans[0]
            mention_start, mention_end = mention_span
            raw_evidence = str(
                raw.get("evidence")
                or raw.get("evidence_text")
                or raw.get("evidence_quote") or ""
            )
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
                "argument_complete": bool(
                    raw.get("argument_complete", True)
                ),
                "standalone_referential": bool(
                    raw.get("standalone_referential", True)
                ),
                "argument_span_repaired": bool(
                    raw.get("argument_span_repaired", False)
                ),
                "endpoint_candidate": bool(
                    raw.get("endpoint_candidate", False)
                ),
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
                **({
                    "grounded_semantic_mention": True,
                } if raw.get("grounded_semantic_mention") is True else {}),
                **({
                    "soft_wrap_surface_repaired": True,
                } if raw.get("soft_wrap_surface_repaired") is True else {}),
                **({
                    "coordinated_distinct_candidate": True,
                } if raw.get("coordinated_distinct_candidate") is True else {}),
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
            "_rejected_entities": rejected_entity_rows,
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

    def extract_grounded_mentions(
        self,
        project: dict[str, Any],
        schema_record: dict[str, Any],
        chunk: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract one complete, minimal, source-grounded mention inventory.

        This is the entity half of the GraphRAG-style local fact path.  A
        compact semantic request replaces the former primary entity request,
        proposition-argument request and relation-endpoint gleaning request.
        The model never controls provenance offsets: copied surfaces are
        grounded against copied evidence before the ordinary sanitizer runs.
        """

        logical = schema_record["schema_json"]
        content, body_start = _chunk_body_with_offset(chunk)
        if not content.strip():
            return {
                "entities": [], "relations": [],
                "_diagnostics": {
                    "llm_calls": 0,
                    "grounded_mention_inventory": True,
                    "coverage_complete": True,
                },
            }
        model = project.get("llm_model") or Config.default_llm_model
        entity_types = [
            str(item.get("name") or "")
            for item in logical.get("entity_types") or []
            if isinstance(item, Mapping) and str(item.get("name") or "")
        ]
        type_by_key = {value.casefold(): value for value in entity_types}
        fallback_type = type_by_key.get("genericentity") or "GenericEntity"
        max_chars = int(Config.entity_grounded_segment_max_chars)
        max_sentences = int(
            Config.entity_grounded_max_sentences_per_segment
        )
        max_items = int(Config.entity_grounded_max_items_per_segment)
        calls = 0
        split_retries = 0
        dropped_items = 0

        system = (
            "Extract every occurrence of a minimal reusable referential mention "
            "from SOURCE. Return only an items array whose rows contain surface, "
            "evidence and an optional kind. Include named and common noun phrases, "
            "identifiers, actors, organizations, objects, concepts, locations, "
            "nominal actions, events and states, plus an adjective state when code "
            "or a condition independently refers to it. Include short lowercase "
            "relation arguments and preserve repeated occurrences. Include meaningful "
            "section labels and table column headings as concepts, as well as every "
            "semantic data cell. Include callable/indexed source tokens with their "
            "balanced delimiters. Each surface must be a complete identity-bearing "
            "source noun phrase: omit "
            "determiners, numbering markers, auxiliaries, modality, negation, "
            "relation verbs, prepositions, relative clauses and surrounding clause "
            "text, but retain a following common-noun head when it completes a named "
            "identifier (for example a product name followed by its object class). "
            "A nested noun phrase and its complete containing noun phrase may both be "
            "emitted when each is independently referential; never replace either with "
            "a clause. Preserve a coordinated noun phrase when the source treats the "
            "whole phrase as one argument. Split a modified noun phrase only when the "
            "modifier itself states another fact. In a verbal table cell, emit its "
            "nominal participants, not the complete verb phrase. Never emit "
            "bare pronouns and deictic placeholders, complete propositions, bare verbs, "
            "punctuation or fragments. Evidence must be one exact contiguous SOURCE "
            "substring containing the occurrence. When a determiner or demonstrative "
            "precedes an ordinary lexical noun, emit that noun span rather than omitting "
            "the referent. "
            "Re-read every sentence, code line, "
            "list item and table row before returning. Do not extract relations."
        )

        def choose_type(raw_hint: Any) -> str:
            hint = str(raw_hint or "").strip().casefold()
            if hint in type_by_key:
                return type_by_key[hint]
            semantic_tokens = set(re.findall(r"[a-z]+", hint))
            preferred: tuple[str, ...] = ()
            if semantic_tokens & {"event", "state", "process", "action"}:
                preferred = ("event", "process", "genericentity")
            elif semantic_tokens & {"location", "place", "geo"}:
                preferred = ("location", "genericentity")
            elif semantic_tokens & {"person", "actor"}:
                preferred = ("person", "genericentity")
            elif semantic_tokens & {"organization", "organisation"}:
                preferred = ("organization", "organisation", "genericentity")
            for key in preferred:
                if key in type_by_key:
                    return type_by_key[key]
            return fallback_type

        def split_at_boundary(segment: str) -> int:
            midpoint = len(segment) // 2
            minimum = min(256, max(1, len(segment) // 4))
            boundaries = [
                end for _start, end, _text in split_sentences(segment)
                if minimum <= end <= len(segment) - minimum
            ]
            if boundaries:
                return min(boundaries, key=lambda value: abs(value - midpoint))
            newline = segment.rfind("\n", minimum, max(minimum, midpoint + 1))
            if newline > 0:
                return newline + 1
            return midpoint

        def request_segment(
            segment: str,
            offset: int,
            depth: int = 0,
            *,
            already_found: list[dict[str, Any]] | None = None,
        ) -> list[dict[str, Any]]:
            nonlocal calls, split_retries, dropped_items
            if len(segment) > max_chars and depth < 10:
                boundary = split_at_boundary(segment)
                if 0 < boundary < len(segment):
                    split_retries += 1
                    return [
                        *request_segment(
                            segment[:boundary], offset, depth + 1,
                            already_found=already_found,
                        ),
                        *request_segment(
                            segment[boundary:], offset + boundary, depth + 1,
                            already_found=already_found,
                        ),
                    ]
            calls += 1
            try:
                raw = self.ollama.json_chat(
                    model=model,
                    system=(
                        system
                        + (
                            " This is one bounded omission-recovery pass. "
                            "ALREADY_FOUND rows are immutable and must not be "
                            "repeated. Re-read SOURCE and return only missing "
                            "mention occurrences; return an empty items array "
                            "when none are missing."
                            if already_found is not None else ""
                        )
                    ),
                    prompt=(
                        "Current logical entity types (kind hint only):\n"
                        + json.dumps(entity_types, ensure_ascii=False)
                        + (
                            "\n\nALREADY_FOUND:\n"
                            + json.dumps(
                                already_found, ensure_ascii=False,
                            )
                            if already_found is not None else ""
                        )
                        + "\n\nSOURCE:\n" + segment
                    ),
                    schema=GROUNDED_MENTION_INVENTORY_FORMAT,
                    temperature=0,
                    context_budget_tokens=int(
                        project.get("model_max_tokens") or 32768
                    ),
                    max_output_tokens=Config.entity_only_max_output_tokens,
                )
            except OllamaProtocolError as exc:
                recoverable = (
                    "truncated by output length" in str(exc)
                    or "prompt token estimate underestimated" in str(exc)
                )
                if recoverable and len(segment) > 64 and depth < 10:
                    boundary = split_at_boundary(segment)
                    if 0 < boundary < len(segment):
                        split_retries += 1
                        return [
                            *request_segment(
                                segment[:boundary], offset, depth + 1,
                                already_found=already_found,
                            ),
                            *request_segment(
                                segment[boundary:], offset + boundary,
                                depth + 1,
                                already_found=already_found,
                            ),
                        ]
                raise
            raw_items: Any = []
            if isinstance(raw, Mapping):
                raw_items = raw.get("items")
                if not isinstance(raw_items, list):
                    raw_items = raw.get("entities") or []
            if not isinstance(raw_items, list):
                raise RuntimeError(
                    "grounded mention inventory requires an items array"
                )
            if len(raw_items) >= max_items and len(segment) > 128 and depth < 10:
                boundary = split_at_boundary(segment)
                if 0 < boundary < len(segment):
                    split_retries += 1
                    return [
                        *request_segment(
                            segment[:boundary], offset, depth + 1,
                            already_found=already_found,
                        ),
                        *request_segment(
                            segment[boundary:], offset + boundary, depth + 1,
                            already_found=already_found,
                        ),
                    ]
            rows: list[dict[str, Any]] = []
            seen_local: set[tuple[int, int]] = set()
            for item in raw_items:
                if not isinstance(item, Mapping):
                    dropped_items += 1
                    continue
                surface = str(
                    item.get("surface") or item.get("name") or ""
                ).strip()
                evidence = str(
                    item.get("evidence")
                    or item.get("evidence_text")
                    or item.get("evidence_quote")
                    or ""
                ).strip()
                span = _ground_copied_mention_surface(
                    segment, surface, evidence,
                )
                if span is None or span in seen_local:
                    dropped_items += 1
                    continue
                seen_local.add(span)
                start, end = span
                evidence_candidates = _surface_spans(segment, evidence)
                evidence_span = next((
                    value for value in evidence_candidates
                    if value[0] <= start < end <= value[1]
                ), None)
                if evidence_span is None:
                    evidence_text, evidence_start, evidence_end = (
                        self._sentence_for_span(segment, start, end)
                    )
                else:
                    evidence_start, evidence_end = evidence_span
                    evidence_text = segment[evidence_start:evidence_end]
                rows.append({
                    "id": f"grounded_{offset + start}_{offset + end}",
                    "name": segment[start:end],
                    "type": choose_type(
                        item.get("kind") or item.get("type")
                    ),
                    "mention_start": offset + start,
                    "mention_end": offset + end,
                    "complete_mention_start": offset + start,
                    "complete_mention_end": offset + end,
                    "evidence": evidence_text,
                    "evidence_start": offset + evidence_start,
                    "evidence_end": offset + evidence_end,
                    "argument_complete": True,
                    "standalone_referential": True,
                    "endpoint_candidate": True,
                    "grounded_semantic_mention": True,
                    "confidence": coerce_confidence(
                        item.get("confidence"), 0.80,
                    ),
                })
            return rows

        sentence_rows = split_sentences(content)
        segments: list[tuple[int, int]] = []
        if len(content) <= max_chars:
            segments = [(0, len(content))]
        elif not sentence_rows:
            segments = [(0, len(content))]
        else:
            group_start = sentence_rows[0][0]
            group_end = group_start
            group_count = 0
            for sentence_start, sentence_end, _text in sentence_rows:
                would_overflow = (
                    group_count > 0
                    and (
                        group_count >= max_sentences
                        or sentence_end - group_start > max_chars
                    )
                )
                if would_overflow:
                    segments.append((group_start, group_end))
                    group_start = sentence_start
                    group_count = 0
                group_end = sentence_end
                group_count += 1
            if group_end > group_start:
                segments.append((group_start, group_end))
        raw_entities: list[dict[str, Any]] = []
        for start, end in segments:
            raw_entities.extend(request_segment(content[start:end], start))
        primary_calls = calls
        known_spans = {
            (
                int(item.get("mention_start") or 0),
                int(item.get("mention_end") or 0),
            )
            for item in raw_entities
        }
        gleaned_entities: list[dict[str, Any]] = []
        for _round in range(Config.entity_grounded_gleaning_rounds):
            round_rows: list[dict[str, Any]] = []
            for start, end in segments:
                found = [
                    {
                        "surface": str(item.get("name") or ""),
                        "start": int(item.get("mention_start") or 0) - start,
                        "end": int(item.get("mention_end") or 0) - start,
                    }
                    for item in [*raw_entities, *gleaned_entities]
                    if start <= int(item.get("mention_start") or -1)
                    and int(item.get("mention_end") or -1) <= end
                ]
                round_rows.extend(request_segment(
                    content[start:end], start, already_found=found,
                ))
            new_rows = []
            for item in round_rows:
                span = (
                    int(item.get("mention_start") or 0),
                    int(item.get("mention_end") or 0),
                )
                if span in known_spans:
                    continue
                known_spans.add(span)
                new_rows.append(item)
            if not new_rows:
                break
            gleaned_entities.extend(new_rows)
        raw_entities.extend(gleaned_entities)

        # Local models commonly return one representative row for a repeated
        # surface even when the contract asks for every occurrence.  Relation
        # provenance, however, is occurrence based: borrowing the first
        # occurrence for a later clause makes an otherwise correct fact point
        # at the wrong source span.  Recover exact, case-preserving repeats
        # deterministically after the model has established that the surface
        # is a valid minimal mention.  Suppress a repeat nested in a longer
        # accepted mention when it is only a single lexical token (for example
        # ``platform`` inside ``research platform``). A repeated multi-token
        # phrase can itself be referential inside a larger data-set/product
        # name, so the grounded nested-mention policy retains it.
        modeled_spans = {
            (
                int(item.get("mention_start") or 0),
                int(item.get("mention_end") or 0),
            )
            for item in raw_entities
        }
        accepted_spans = sorted(modeled_spans)
        recovered_entities: list[dict[str, Any]] = []
        for template in list(raw_entities):
            surface = str(template.get("name") or "")
            if not surface:
                continue
            for repeat_start, repeat_end in _surface_spans(content, surface):
                repeat_span = (repeat_start, repeat_end)
                if repeat_span in modeled_spans:
                    continue
                # _surface_spans is intentionally case-insensitive for legacy
                # extraction repair. Repeat recovery is stricter so a lowercase
                # data value does not manufacture an entity from a capitalized
                # parser-owned table heading.
                repeated_source = content[repeat_start:repeat_end]
                if repeated_source != surface:
                    # Permit capitalization-only variants for multi-token
                    # mentions (for example a heading repeated in prose). Keep
                    # single-token matching case-exact so a lowercase data
                    # value does not become a capitalized table heading.
                    line_start = content.rfind("\n", 0, repeat_start) + 1
                    line_end = content.find("\n", repeat_end)
                    if line_end < 0:
                        line_end = len(content)
                    standalone_line = (
                        content[line_start:line_end].strip() == repeated_source
                    )
                    if repeated_source.casefold() != surface.casefold() or (
                        len(re.findall(r"[\w]+", surface)) < 2
                        and not standalone_line
                    ):
                        continue
                nested_in_accepted = any(
                    outer_start <= repeat_start
                    and repeat_end <= outer_end
                    and (outer_start, outer_end) != repeat_span
                    for outer_start, outer_end in accepted_spans
                )
                if (
                    nested_in_accepted
                    and len(re.findall(r"[\w]+", surface)) < 2
                ):
                    continue
                evidence_text, evidence_start, evidence_end = (
                    self._sentence_for_span(
                        content, repeat_start, repeat_end,
                    )
                )
                recovered = {
                    **template,
                    "name": repeated_source,
                    "mention_start": repeat_start,
                    "mention_end": repeat_end,
                    "complete_mention_start": repeat_start,
                    "complete_mention_end": repeat_end,
                    "evidence": evidence_text,
                    "evidence_start": evidence_start,
                    "evidence_end": evidence_end,
                    "deterministic_repeat_recovery": True,
                }
                recovered_entities.append(recovered)
                modeled_spans.add(repeat_span)
        raw_entities.extend(recovered_entities)
        raw_entities.sort(key=lambda item: (
            int(item.get("mention_start") or 0),
            int(item.get("mention_end") or 0),
            str(item.get("name") or ""),
        ))
        # Stable local IDs are assigned after segmentation because a recoverable
        # split can otherwise create duplicate model-local identifiers.
        for index, entity in enumerate(raw_entities, start=1):
            entity["id"] = f"grounded_entity_{index:05d}"
        extraction = self._sanitize_extraction(
            {"entities": raw_entities, "relations": []}, content, logical,
        )
        grounded_spans = {
            (int(item["mention_start"]), int(item["mention_end"]))
            for item in raw_entities
        }
        for entity in extraction.get("entities") or []:
            entity["endpoint_candidate"] = (
                int(entity.get("mention_start") or -1),
                int(entity.get("mention_end") or -1),
            ) in grounded_spans
            for key in (
                "evidence_start", "evidence_end",
                "mention_start", "mention_end",
            ):
                if entity.get(key) is not None:
                    entity[key] = int(entity[key]) + body_start
        extraction["relations"] = []
        extraction["_verification_events"] = []
        extraction["_diagnostics"] = {
            **extraction.get("_diagnostics", {}),
            "llm_calls": calls,
            "primary_llm_calls": primary_calls,
            "grounded_gleaning_calls": calls - primary_calls,
            "grounded_gleaning_mentions_recovered": len(gleaned_entities),
            "grounded_mention_inventory": True,
            "coverage_complete": dropped_items == 0,
            "dropped_items": dropped_items,
            "segments": len(segments),
            "split_retries": split_retries,
            "deterministic_occurrences_recovered": len(recovered_entities),
            "legacy_proposition_calls_avoided": primary_calls,
            "legacy_endpoint_calls_avoided": primary_calls,
        }
        return extraction

    def extract_relation_endpoint_mentions(
        self,
        project: dict[str, Any],
        schema_record: dict[str, Any],
        chunk: dict[str, Any],
        *,
        existing_entities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Recover exact mentions required by explicit local relations.

        This recall stage returns entities only.  It cannot emit a predicate,
        direction, verification result, or publish decision.  Every output is
        passed through the ordinary exact-span sanitizer before the entity
        resolver sees it.
        """

        logical = schema_record["schema_json"]
        relation_types = list(logical.get("relation_types") or [])
        content, body_start = _chunk_body_with_offset(chunk)
        if not relation_types or not content.strip():
            return {"entities": [], "_diagnostics": {"llm_calls": 0}}
        model = project.get("llm_model") or Config.default_llm_model
        entity_schema = json.dumps(
            logical.get("entity_types") or [], ensure_ascii=False,
        )
        max_chars = Config.entity_only_segment_max_chars
        calls = 0
        segmented = False

        def boundary(value: str) -> int:
            midpoint = len(value) // 2
            candidates = [
                end for _start, end, _sentence in split_sentences(value)
                if len(value) // 4 <= end <= (len(value) * 3) // 4
            ]
            return min(candidates, key=lambda item: abs(item - midpoint)) \
                if candidates else midpoint

        def request(
            segment: str,
            offset: int = 0,
            depth: int = 0,
            known_entities: list[dict[str, Any]] | None = None,
            gleaning_round: int = 0,
        ) -> list[dict[str, Any]]:
            nonlocal calls, segmented
            if len(segment) > max_chars and depth < 8:
                split_at = boundary(segment)
                if 0 < split_at < len(segment):
                    segmented = True
                    return [
                        *request(
                            segment[:split_at], offset, depth + 1,
                            known_entities, gleaning_round,
                        ),
                        *request(
                            segment[split_at:], offset + split_at, depth + 1,
                            known_entities, gleaning_round,
                        ),
                    ]
            calls += 1
            try:
                raw = self.ollama.json_chat(
                    model=model,
                    system=(
                        "Audit the source sentence by sentence for every explicit "
                        "binary semantic relation statement, including statements "
                        "whose endpoint is a recipient, destination, applied setting, "
                        "result or transferred item. Return both exact source entity "
                        "occurrences needed as the statement's subject or object, "
                        "especially endpoints missing from the existing entity pass. "
                        "Read each versioned relation definition in its canonical "
                        "direction even when the source uses inverse wording, passive "
                        "voice, a containing object before its members, or a service "
                        "before the capability/action it enables. Preserve every exact "
                        "member in a coordinated or enumerated endpoint list. Short or "
                        "lowercase common-noun concepts are still valid endpoints when "
                        "the source assigns them a role; do not omit them merely because "
                        "they are generic words. "
                        "Do not return relations, predicates, "
                        "inferred entities, headings, units, bare values, or mere "
                        "co-occurrences. Names and Unicode half-open offsets must match "
                        "the source exactly. Return only occurrences absent from the "
                        "existing pass; return an empty array when coverage is complete. "
                        "Do not emit a pronoun, deictic phrase, or context-dependent "
                        "placeholder as a standalone entity. When its explicit named "
                        "antecedent occurs in this source segment, return that exact named "
                        "occurrence instead. "
                        "Relation definitions are guidance only and never authorize a "
                        "fact. If a sentence contains more than one participant, select "
                        "the two endpoints explicitly connected by the sentence and "
                        "keep any transferred item or contextual participant separate; "
                        "do not collapse an entire clause into one entity."
                    ),
                    prompt=(
                        "Entity schema:\n" + entity_schema
                        + "\n\nVersioned relation definitions:\n"
                        + json.dumps(relation_types, ensure_ascii=False)
                        + "\n\nExisting entity pass:\n"
                        + json.dumps([{
                            "name": str(item.get("name") or ""),
                            "type": str(item.get("type") or "GenericEntity"),
                        } for item in [
                            *existing_entities, *(known_entities or []),
                        ]], ensure_ascii=False)
                        + f"\n\nCoverage audit round: {gleaning_round}"
                        + "\n\nSource text:\n" + segment
                    ),
                    schema=ENTITY_ONLY_EXTRACTION_FORMAT,
                    temperature=0,
                    context_budget_tokens=int(
                        project.get("model_max_tokens") or 32768
                    ),
                    max_output_tokens=Config.entity_only_max_output_tokens,
                )
            except OllamaProtocolError as exc:
                if (
                    depth < 8 and len(segment) > 64
                    and (
                        "truncated by output length" in str(exc)
                        or "prompt token estimate underestimated" in str(exc)
                    )
                ):
                    split_at = boundary(segment)
                    if 0 < split_at < len(segment):
                        segmented = True
                        return [
                            *request(
                                segment[:split_at], offset, depth + 1,
                                known_entities, gleaning_round,
                            ),
                            *request(
                                segment[split_at:], offset + split_at, depth + 1,
                                known_entities, gleaning_round,
                            ),
                        ]
                raise
            rows = []
            for item in self._flat_extraction(raw).get("entities") or []:
                if not isinstance(item, dict):
                    continue
                value = dict(item)
                # This pass emits exact endpoint projections only.  Mark them
                # so the generic clause-boundary gate does not discard a valid
                # endpoint merely because the primary model also returned a
                # broader clause span; the ordinary exact-span and type gates
                # still apply.
                value["endpoint_candidate"] = True
                for start_key, end_key in (
                    ("mention_start", "mention_end"),
                    ("evidence_start", "evidence_end"),
                ):
                    if value.get(start_key) is not None:
                        value[start_key] = int(value[start_key]) + offset
                    if value.get(end_key) is not None:
                        value[end_key] = int(value[end_key]) + offset
                rows.append(value)
            return rows

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        gleaning_calls = 0
        seen: set[tuple[str, str, int, int]] = set()

        # Endpoint recovery is a coverage audit, not a document-level
        # extraction pass.  Send sentence-aligned groups so a long chunk cannot
        # cause a valid relation in a later sentence to be omitted.  Groups are
        # bounded by the same character budget used by the normal request
        # splitter, preserving exact offsets and keeping model calls bounded.
        coverage_segments: list[tuple[int, int, str]] = []
        sentence_rows = split_sentences(content)
        if not sentence_rows:
            sentence_rows = [(0, len(content), content)]
        group_start = sentence_rows[0][0]
        group_end = group_start
        for sentence_start, sentence_end, _sentence in sentence_rows:
            if (
                group_end > group_start
                and sentence_end - group_start > max_chars
            ):
                coverage_segments.append(
                    (group_start, group_end, content[group_start:group_end])
                )
                group_start = sentence_start
            group_end = sentence_end
        if group_end > group_start:
            coverage_segments.append(
                (group_start, group_end, content[group_start:group_end])
            )
        if not coverage_segments:
            coverage_segments = [(0, len(content), content)]
        for round_no in range(Config.entity_endpoint_gleaning_rounds + 1):
            raw_rows: list[dict[str, Any]] = []
            for segment_start, _segment_end, segment in coverage_segments:
                raw_rows.extend(request(
                    segment,
                    offset=segment_start,
                    known_entities=accepted,
                    gleaning_round=round_no,
                ))
            if round_no:
                gleaning_calls += 1
            # Endpoint coverage requests exact source occurrences but a local
            # model may omit offsets when the same surface occurs repeatedly.
            # Expand that bounded, source-searchable ambiguity into one row
            # per exact occurrence. This preserves mention provenance and lets
            # later pair planning choose the contextual occurrence; it does
            # not infer an edge or a predicate.
            expanded_rows: list[dict[str, Any]] = []
            for item in raw_rows:
                if not isinstance(item, dict):
                    continue
                value = dict(item)
                surface = str(value.get("name") or value.get("surface") or "")
                spans = _surface_spans(content, surface)
                try:
                    start = int(value.get("mention_start"))
                    end = int(value.get("mention_end"))
                except (TypeError, ValueError):
                    start = end = -1
                exact = (
                    0 <= start < end <= len(content)
                    and content[start:end] == surface
                )
                if value.get("endpoint_candidate") is True and not exact and spans:
                    for index, (span_start, span_end) in enumerate(spans):
                        occurrence = dict(value)
                        occurrence["id"] = (
                            f"{value.get('id') or 'endpoint'}:{index}"
                        )
                        occurrence["mention_start"] = span_start
                        occurrence["mention_end"] = span_end
                        occurrence["recovered_exact_occurrence"] = True
                        expanded_rows.append(occurrence)
                else:
                    expanded_rows.append(value)
            raw_rows = expanded_rows
            prepared = []
            for index, item in enumerate(raw_rows, start=1):
                value = dict(item)
                value["id"] = f"endpoint_{round_no:02d}_{index:04d}"
                value["confidence"] = float(value.get("confidence") or 0.5)
                prepared.append(value)
            current = self._sanitize_extraction(
                {"entities": prepared, "relations": []}, content, logical,
            )
            rejected.extend(current.get("_rejected_entities") or [])
            added = 0
            for entity in current.get("entities") or []:
                key = (
                    str(entity.get("name") or ""), str(entity.get("type") or ""),
                    int(entity.get("mention_start") or 0),
                    int(entity.get("mention_end") or 0),
                )
                if key in seen:
                    continue
                seen.add(key)
                accepted.append(entity)
                added += 1
            if round_no and added == 0:
                break
        extraction = {
            "entities": accepted,
            "relations": [],
            "_rejected_entities": rejected,
            "_diagnostics": {"rejected_entities": len(rejected)},
        }
        for entity in extraction.get("entities") or []:
            for key in (
                "evidence_start", "evidence_end", "mention_start", "mention_end",
            ):
                if entity.get(key) is not None:
                    entity[key] = int(entity[key]) + body_start
            entity["endpoint_coverage_pass"] = True
        extraction["relations"] = []
        extraction["_diagnostics"] = {
            **extraction.get("_diagnostics", {}),
            "llm_calls": calls,
            "segmented_fallback": segmented,
            "endpoint_coverage": True,
            "endpoint_gleaning_calls": gleaning_calls,
        }
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
            "occurrences. Short or lowercase common-noun concepts remain valid reusable "
            "concepts when "
            "the source uses them as an argument; do not omit them merely because they are "
            "generic words. "
            "The name must be the exact source substring and mention_start/end must "
            "be its exact Unicode half-open offsets in Source text. Keep the JSON compact; omit "
            "optional aliases and evidence unless they add source-backed information. Do not "
            "extract a pronoun, deictic phrase, or context-dependent placeholder as a standalone "
            "entity; emit an exact named antecedent occurrence when it is present in this source "
            "segment. For an event, state, condition, failure, change, or other proposition "
            "argument, preserve the complete exact source span needed to identify that argument, "
            "including its trigger or state head and any attached negation or modality. Do not "
            "reduce a complete event/state phrase to an inner product, organization, or process "
            "noun. It is valid to retain a separately mentioned concrete inner entity only when "
            "the source also uses that exact span as its own argument. Enumerated peer items must "
            "remain separate occurrences and must not be merged into one mention. For every item, "
            "standalone_referential must be true only when the exact span independently names a "
            "stable entity or reusable concept; set it false for pronouns, deictic phrases, "
            "context-dependent placeholders, bare dates, times, quantities, or list labels. "
            "A bare date or time is context metadata, not an Event entity. "
            "argument_complete says whether its exact mention is already a complete entity or "
            "proposition argument. complete_mention_start/end must equal mention_start/end when "
            "complete; when an event, state, condition, failure, change, or process mention is "
            "shortened, they must cover the smallest complete exact source span containing it. "
            "These fields describe spans only and must not infer a relation. Do not extract or "
            "infer relations. Return an empty entities array only when the Source text "
            "truly contains no concrete entity mention."
        )
        entity_schema = json.dumps(logical.get("entity_types") or [], ensure_ascii=False)
        llm_calls = 0
        proposition_gleaning_calls = 0
        proposition_gleaning_failures = 0
        segmented_fallback = False
        segment_max_chars = Config.entity_only_segment_max_chars
        segment_max_items = Config.entity_only_max_items_per_segment
        segment_max_sentences = Config.entity_only_max_sentences_per_segment

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
            nonlocal llm_calls, proposition_gleaning_calls
            nonlocal proposition_gleaning_failures, segmented_fallback
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
            # A second, deliberately narrow pass extracts complete proposition
            # arguments. General entity extraction models often emit only an
            # inner process/product noun (for example the object inside a
            # failure or state change), which leaves relation endpoints
            # structurally incomplete. This pass is domain- and predicate-
            # independent: it asks only for exact source spans of explicit
            # events, states, conditions, failures and changes, and it
            # never creates a relation.
            uses_cjk = bool(re.search(
                r"[\u3400-\u4dbf\u4e00-\u9fff]", segment,
            ))
            proposition_system = (
                "只抽取原文中本身被另一命题当作论元的完整事件、状态、条件、故障或变化，"
                "包括明确充当原因、结果或条件的完整事件/状态片段。"
                "每个 name 必须是最小但完整的原文连续片段，必须保留触发词、状态变化、"
                "否定和情态。普通主谓宾事实、实体之间的使用/生产/位置/隶属等关系句，"
                "如果该动作本身没有被原文当作事件、状态、条件或另一关系的端点，就不要"
                "把整句建成事件实体；这些事实应由关系阶段处理。不要把其中的产品、组织"
                "或过程内部名词单独作为本次输出。"
                "当一个命题明确连接原因与结果状态/变化时，原因端和结果端都必须各自输出"
                "为两个互相分离的完整论元，绝不能把同时包含原因和结果的整句作为其中一端；"
                "这条要求不适用于没有因果、条件或状态变化语义的普通关系句。"
                "不要推断关系。type 必须从实体结构定义中选择。argument_complete 必须为 "
                "true，proposition_argument 必须为 true。严格返回 JSON。"
                if uses_cjk else
                "Extract only complete explicit proposition arguments: events, states, "
                "conditions, failures, or changes that the source itself uses as an argument "
                "of another proposition, including complete event/state spans explicitly "
                "serving as a cause, result, or condition. Each name must be the smallest "
                "complete contiguous source span and retain its trigger, state change, "
                "negation, and modality. Do not turn an ordinary subject-verb-object fact "
                "into an Event when its action is not itself referenced as an event, state, "
                "condition, or endpoint; the relation stage handles that fact. Do not return "
                "inner product, organization, or "
                "process nouns in this focused pass. Do not infer relations. "
                "When one proposition explicitly connects a cause to a resulting state or "
                "change, return both complete endpoints as separate spans; never return the "
                "whole clause containing both cause and result as either endpoint. This "
                "requirement does not "
                "apply to ordinary relation statements without causal, conditional, or state-"
                "change semantics. "
                "Choose type from the entity schema, set argument_complete true, and make the complete "
                "span equal the mention span. Set proposition_argument true. Return strict JSON."
            )
            proposition_prompt = (
                f"Entity schema:\n{entity_schema}\n\nSource text:\n{segment}"
            )
            proposition_raw: Any = {"entities": []}
            for proposition_attempt in range(2):
                proposition_gleaning_calls += 1
                llm_calls += 1
                try:
                    proposition_raw = self.ollama.json_chat(
                        model=model,
                        system=proposition_system,
                        prompt=proposition_prompt,
                        schema=PROPOSITION_ARGUMENT_EXTRACTION_FORMAT,
                        temperature=0,
                        context_budget_tokens=int(
                            project.get("model_max_tokens") or 32768
                        ),
                        max_output_tokens=min(
                            Config.entity_only_max_output_tokens, 4096,
                        ),
                    )
                    break
                except (OllamaProtocolError, RuntimeError):
                    # Proposition gleaning is an optional recall pass.  Some
                    # local models answer this narrower prompt with a valid
                    # but legacy ``arguments`` object (or another
                    # schema-shape error) instead of the requested
                    # ``entities`` envelope.  The primary entity pass is
                    # still authoritative; a malformed optional pass must
                    # not invalidate the complete entity snapshot.
                    proposition_gleaning_failures += 1
                    if proposition_attempt:
                        proposition_raw = {"entities": []}
            raw_rows = list(raw_rows)
            proposition_items: Any = []
            if isinstance(proposition_raw, Mapping):
                proposition_items = (
                    proposition_raw.get("entities")
                    or proposition_raw.get("arguments")
                    or proposition_raw.get("items")
                    or []
                )
            if isinstance(proposition_items, list):
                for item in proposition_items:
                    if (
                        not isinstance(item, Mapping)
                        or item.get("proposition_argument") is not True
                    ):
                        continue
                    name = str(item.get("name") or "")
                    if not name or len(name) > 320:
                        continue
                    starts: list[int] = []
                    try:
                        proposed_start = int(item.get("mention_start"))
                        proposed_end = int(item.get("mention_end"))
                    except (TypeError, ValueError):
                        proposed_start = proposed_end = -1
                    if (
                        0 <= proposed_start < proposed_end <= len(segment)
                        and segment[proposed_start:proposed_end] == name
                    ):
                        starts = [proposed_start]
                    else:
                        cursor = 0
                        while True:
                            found = segment.find(name, cursor)
                            if found < 0:
                                break
                            starts.append(found)
                            cursor = found + max(1, len(name))
                    for found in starts:
                        raw_rows.append({
                            "name": name,
                            "type": str(item.get("type") or ""),
                            "mention_start": found,
                            "mention_end": found + len(name),
                            "argument_complete": True,
                            "proposition_argument": True,
                            "confidence": float(item.get("confidence") or 0.7),
                        })
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
                try:
                    mention_start = int(normalized.get("mention_start"))
                    mention_end = int(normalized.get("mention_end"))
                    complete_start = int(
                        normalized.get("complete_mention_start")
                    )
                    complete_end = int(
                        normalized.get("complete_mention_end")
                    )
                except (TypeError, ValueError):
                    mention_start = mention_end = complete_start = complete_end = -1
                valid_expansion = (
                    0 <= complete_start <= mention_start < mention_end
                    <= complete_end <= len(segment)
                    and (complete_start, complete_end)
                    != (mention_start, mention_end)
                    and complete_end - complete_start <= 320
                )
                if valid_expansion:
                    normalized["name"] = segment[complete_start:complete_end]
                    normalized["mention_start"] = complete_start
                    normalized["mention_end"] = complete_end
                    normalized["argument_complete"] = True
                    normalized["argument_span_repaired"] = True
                normalized.pop("complete_mention_start", None)
                normalized.pop("complete_mention_end", None)
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

        # Entity recall degrades sharply when a single model response must cover
        # many independent statements.  Split at source sentence boundaries even
        # when the character budget is not exhausted.  This is a structural
        # completeness measure: it does not inspect domain words, entity names or
        # predicates, and every returned item still has to pass exact-span
        # validation against the original chunk.
        sentences = split_sentences(content)
        top_level_segments: list[tuple[int, int]] = []
        if len(sentences) > segment_max_sentences:
            for index in range(0, len(sentences), segment_max_sentences):
                group = sentences[index:index + segment_max_sentences]
                top_level_segments.append((group[0][0], group[-1][1]))
            segmented_fallback = True
        else:
            top_level_segments = [(0, len(content))]

        raw_entities: list[dict[str, Any]] = []
        for start, end in top_level_segments:
            raw_entities.extend(request_segment(content[start:end], start))
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
            "proposition_gleaning_calls": proposition_gleaning_calls,
            "proposition_gleaning_failures": proposition_gleaning_failures,
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
