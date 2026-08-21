from __future__ import annotations

import copy

import pytest

from knowledge_forge.config import Config
from knowledge_forge.community_report_runtime import (
    COMMUNITY_REPORT_PROMPT_VERSION,
    REPORT_JSON_SCHEMA,
    REPORT_SYSTEM_PROMPT,
    CommunityReportPolicy,
    KnowledgeForgeCommunityReportGenerator,
    build_community_report_plan,
    clean_report_prose,
    report_system_prompt,
    validate_report_payload,
)
from knowledge_forge.service import _attach_public_community_report_sources


def _snapshot() -> dict:
    return {
        "canonicals": [
            {
                "canonical_key": key,
                "name": name,
                "primary_type": "Concept",
                "aliases": [],
                "source_count": 1,
                "metadata": {"description": f"{name} 的证据约束说明"},
            }
            for key, name in (("a", "甲"), ("b", "乙"), ("c", "丙"), ("d", "丁"))
        ],
        "relation_endpoints": [
            {
                "relation_key": key,
                "source_canonical_key": source,
                "target_canonical_key": target,
                "relation_type": "related_to",
                "document_id": 1,
                "chunk_id": index,
                "evidence_text": evidence,
                "metadata": {},
            }
            for index, (key, source, target, evidence) in enumerate((
                ("r1", "a", "b", "甲与乙存在已验证联系。"),
                ("r2", "b", "c", "乙与丙存在已验证联系。"),
                ("r3", "c", "d", "丙与丁存在已验证联系。"),
            ), 1)
        ],
    }


def _community_run() -> dict:
    return {
        "run_id": "community-run-1",
        "input_hash": "community-input-1",
        "communities": [
            {
                "key": "community:main",
                "label": "社区 1",
                "level": 0,
                "entity_keys": ["a", "b", "c", "d"],
            },
            {
                "key": "community:single",
                "label": "社区 2",
                "level": 0,
                "entity_keys": ["a"],
            },
        ],
    }


def test_report_plan_is_deterministic_and_skips_fact_poor_communities():
    policy = CommunityReportPolicy(context_tokens=16_000)

    left = build_community_report_plan(
        community_run=_community_run(), snapshot=_snapshot(), policy=policy,
    )
    right = build_community_report_plan(
        community_run=_community_run(), snapshot=_snapshot(), policy=policy,
    )

    assert left == right
    assert left["expected_count"] == 2
    assert left["eligible_count"] == 1
    assert left["skipped_count"] == 1
    main = next(item for item in left["items"] if item["community_key"] == "community:main")
    assert main["status"] == "pending"
    assert main["relation_keys"] == ["r1", "r2", "r3"]
    assert main["input_context"]["coverage"]["complete"] is True


def test_report_context_budget_prioritizes_relations_over_unused_entities(
    monkeypatch,
):
    monkeypatch.setattr(Config, "ollama_default_safety_margin", 256)
    entity_count = 30
    snapshot = {
        "canonicals": [{
            "canonical_key": f"entity-{index}",
            "name": f"实体 {index}",
            "primary_type": "Concept",
            "aliases": [f"别名 {index}-{alias}" for alias in range(8)],
            "source_count": 1,
            "metadata": {"description": "实体说明" * 80},
        } for index in range(entity_count)],
        "relation_endpoints": [{
            "relation_key": f"relation-{index}",
            "source_canonical_key": f"entity-{index}",
            "target_canonical_key": f"entity-{index + 1}",
            "relation_type": "related_to",
            "document_id": 1,
            "chunk_id": index + 1,
            "evidence_text": "可核对的关系原文" * 80,
            "metadata": {"display_description": "关系说明" * 50},
        } for index in range(entity_count - 1)],
    }
    community_run = {
        "run_id": "community-budget-run",
        "input_hash": "community-budget-input",
        "communities": [{
            "key": "community:budget",
            "label": "预算测试社区",
            "level": 0,
            "entity_keys": [f"entity-{index}" for index in range(entity_count)],
        }],
    }

    plan = build_community_report_plan(
        community_run=community_run,
        snapshot=snapshot,
        policy=CommunityReportPolicy(
            max_entities=24,
            max_relations=36,
            context_tokens=8192,
            max_output_tokens=512,
        ),
    )
    context = plan["items"][0]["input_context"]
    referenced = {
        value
        for relation in context["relationships"]
        for value in (relation["source_ref"], relation["target_ref"])
    }

    assert len(context["relationships"]) >= 4
    assert len(context["entities"]) < 24
    assert all(item["ref"] in referenced for item in context["entities"])
    assert context["coverage"]["relationships_in_prompt"] == len(
        context["relationships"]
    )


def test_report_plan_bounds_initial_generation_and_defers_fact_rich_tail():
    community_run = _community_run()
    community_run["communities"].append({
        "key": "community:secondary",
        "label": "社区 3",
        "level": 0,
        "entity_keys": ["a", "b"],
    })

    plan = build_community_report_plan(
        community_run=community_run,
        snapshot=_snapshot(),
        policy=CommunityReportPolicy(context_tokens=16_000),
        initial_limit=1,
    )

    assert plan["fact_eligible_count"] == 2
    assert plan["eligible_count"] == 1
    assert plan["deferred_count"] == 1
    deferred = next(item for item in plan["items"] if item.get("deferred"))
    assert deferred["status"] == "skipped"
    assert "按需生成" in deferred["skip_reason"]


def test_reuse_hash_covers_facts_omitted_from_bounded_prompt():
    policy = CommunityReportPolicy(
        max_entities=3, max_relations=2, context_tokens=16_000,
    )
    original = build_community_report_plan(
        community_run=_community_run(), snapshot=_snapshot(), policy=policy,
    )
    changed_snapshot = copy.deepcopy(_snapshot())
    changed_snapshot["relation_endpoints"][2]["evidence_text"] = "丁与丙的证据已更新。"
    changed = build_community_report_plan(
        community_run=_community_run(), snapshot=changed_snapshot, policy=policy,
    )

    original_item = next(item for item in original["items"] if item["status"] == "pending")
    changed_item = next(item for item in changed["items"] if item["status"] == "pending")
    assert len(original_item["input_context"]["relationships"]) == 2
    assert original_item["community_content_hash"] != changed_item["community_content_hash"]


def test_report_validation_resolves_only_known_citations_and_marks_partial_context():
    policy = CommunityReportPolicy(
        max_entities=3, max_relations=2, context_tokens=16_000,
    )
    plan = build_community_report_plan(
        community_run=_community_run(), snapshot=_snapshot(), policy=policy,
    )
    context = next(item for item in plan["items"] if item["status"] == "pending")[
        "input_context"
    ]
    entity_ref = context["entities"][0]["ref"]
    relation_ref = context["relationships"][0]["ref"]
    result = validate_report_payload({
        "title": "社区主题",
        "summary": "该社区围绕若干已验证联系组织。",
        "findings": [{
            "title": "主要联系",
            "explanation": "输入证据支持该联系。",
            "entity_refs": [entity_ref],
            "relation_refs": [relation_ref],
        }],
        "limitations": "",
    }, context)

    assert result["cited_entity_keys"]
    assert result["cited_relation_keys"]
    assert "代表性实体和关系" in result["limitations"]

    with pytest.raises(ValueError, match="不存在"):
        validate_report_payload({
            "title": "错误引用",
            "summary": "错误引用。",
            "findings": [{
                "title": "错误",
                "explanation": "引用不存在。",
                "entity_refs": ["E999"],
                "relation_refs": [],
            }],
            "limitations": "",
        }, context)


def test_report_validation_repairs_only_exact_grounded_model_variants():
    plan = build_community_report_plan(
        community_run=_community_run(),
        snapshot=_snapshot(),
        policy=CommunityReportPolicy(context_tokens=16_000),
    )
    context = next(item for item in plan["items"] if item["status"] == "pending")[
        "input_context"
    ]
    entity_ref = context["entities"][0]["ref"]
    relation_ref = context["relationships"][0]["ref"]

    repaired = validate_report_payload({
        # qwen-family models can omit a top-level title even when the response
        # schema requires one.  The title is presentation metadata, so it can
        # be deterministically derived from immutable input entity names.
        "summary": "该社区由输入中的已验证事实组织而成。",
        "findings": [{
            "finding": f"输入关系支持该结论 [{relation_ref}, {entity_ref}]。",
            "evidence_refs": [relation_ref, entity_ref],
        }, {
            "title": "内嵌引用",
            "description": f"另一个说明直接携带引用 {relation_ref}。",
        }],
        "limitations": [],
    }, context)

    assert repaired["title"] != "社区报告"
    assert context["relationships"][0]["relation_label"] in repaired["title"]
    assert repaired["findings"][0]["explanation"].startswith("输入关系")
    assert repaired["cited_entity_keys"]
    assert repaired["cited_relation_keys"]

    with pytest.raises(ValueError, match="不存在"):
        validate_report_payload({
            "summary": "包含未知引用。",
            "findings": [{
                "description": "该说明引用了 R999。",
            }],
            "limitations": "",
        }, context)

    with pytest.raises(ValueError, match="每项关键结论"):
        validate_report_payload({
            "title": "只有实体引用",
            "summary": "缺少可回到原文的关系引用。",
            "findings": [{
                "title": "实体说明",
                "explanation": "只引用了实体。",
                "entity_refs": [entity_ref],
                "relation_refs": [],
            }],
            "limitations": "",
        }, context)


def test_every_report_finding_requires_its_own_verified_relation():
    plan = build_community_report_plan(
        community_run=_community_run(),
        snapshot=_snapshot(),
        policy=CommunityReportPolicy(context_tokens=16_000),
    )
    context = next(item for item in plan["items"] if item["status"] == "pending")[
        "input_context"
    ]
    entity_ref = context["entities"][0]["ref"]
    relation_ref = context["relationships"][0]["ref"]

    with pytest.raises(ValueError, match="每项关键结论"):
        validate_report_payload({
            "title": "混合引用报告",
            "summary": "其中一项没有独立关系依据。",
            "findings": [{
                "title": "有关系依据",
                "explanation": "甲与乙存在已验证联系。",
                "entity_refs": [entity_ref],
                "relation_refs": [relation_ref],
            }, {
                "title": "只有实体依据",
                "explanation": "该项没有引用关系。",
                "entity_refs": [entity_ref],
                "relation_refs": [],
            }],
            "limitations": "",
        }, context)

    relation_schema = REPORT_JSON_SCHEMA["properties"]["findings"]["items"][
        "properties"
    ]["relation_refs"]
    assert relation_schema["minItems"] == 1
    assert relation_schema["uniqueItems"] is True
    assert "-v5-" in COMMUNITY_REPORT_PROMPT_VERSION
    assert "不得把共现、相关、包含或一般关联扩写成因果" in REPORT_SYSTEM_PROMPT


def test_report_prompt_hash_and_instruction_follow_project_knowledge_language():
    generator = KnowledgeForgeCommunityReportGenerator(
        object(), CommunityReportPolicy(context_tokens=16_000),
    )

    assert generator.prompt_hash_for("zh") != generator.prompt_hash_for("en")
    assert "统一使用中文" in report_system_prompt("zh")
    assert "English" in report_system_prompt("en")
    assert "跟随当前输入文档" in report_system_prompt("auto")


def test_public_report_sources_are_attached_to_the_citing_finding():
    report = {
        "sources": [],
        "findings": [
            {"title": "结论一", "relation_keys": ["r2"], "sources": []},
            {"title": "结论二", "relation_keys": ["r1", "r3"], "sources": []},
        ],
    }
    sources = [
        {"relation_key": "r1", "evidence_text": "依据一"},
        {"relation_key": "r2", "evidence_text": "依据二"},
        {"relation_key": "unused", "evidence_text": "报告级补充依据"},
    ]

    attached = _attach_public_community_report_sources(report, sources)

    assert [item["relation_key"] for item in attached["sources"]] == [
        "r1", "r2", "unused",
    ]
    assert [
        item["relation_key"] for item in attached["findings"][0]["sources"]
    ] == ["r2"]
    assert [
        item["relation_key"] for item in attached["findings"][1]["sources"]
    ] == ["r1"]


def test_report_prose_hides_only_model_facing_references():
    context = {
        "entities": [{"ref": "E001", "key": "entity-1"}],
        "relationships": [{"ref": "R001", "key": "relation-1"}],
    }

    assert clean_report_prose(
        "中心实体 E001 通过关系 R001 连接 E123 道路。", context,
    ) == "中心实体通过关系连接 E123 道路。"
    assert clean_report_prose("节点（E001）已验证。", context) == "节点已验证。"
    assert clean_report_prose(
        "核心实体 E001（示例节点）已验证。", context,
    ) == "核心实体（示例节点）已验证。"


def test_report_context_balances_structure_predicates_and_source_documents():
    snapshot = {
        "canonicals": [{
            "canonical_key": f"e{index}",
            "name": f"实体 {index}",
            "primary_type": "Concept",
            "aliases": [],
            "source_count": 1,
            "metadata": {},
        } for index in range(8)],
        "relation_endpoints": [
            {
                "relation_key": f"hub-{index}",
                "source_canonical_key": "e0",
                "target_canonical_key": f"e{index}",
                "relation_type": "related_to",
                "document_id": 1,
                "chunk_id": index,
                "evidence_text": f"中心关系 {index}",
                "metadata": {},
            }
            for index in range(1, 5)
        ] + [
            {
                "relation_key": "depends",
                "source_canonical_key": "e4",
                "target_canonical_key": "e5",
                "relation_type": "depends_on",
                "document_id": 2,
                "chunk_id": 5,
                "evidence_text": "实体 4 依赖实体 5。",
                "metadata": {},
            },
            {
                "relation_key": "owner",
                "source_canonical_key": "e5",
                "target_canonical_key": "e6",
                "relation_type": "owned_by",
                "document_id": 3,
                "chunk_id": 6,
                "evidence_text": "实体 5 归实体 6 所有。",
                "metadata": {},
            },
            {
                "relation_key": "output",
                "source_canonical_key": "e6",
                "target_canonical_key": "e7",
                "relation_type": "produces",
                "document_id": 4,
                "chunk_id": 7,
                "evidence_text": "实体 6 产生实体 7。",
                "metadata": {},
            },
        ],
    }
    plan = build_community_report_plan(
        community_run={
            "run_id": "diverse-run",
            "input_hash": "diverse-input",
            "communities": [{
                "key": "community:diverse",
                "label": "社区 1",
                "level": 0,
                "entity_keys": [f"e{index}" for index in range(8)],
            }],
        },
        snapshot=snapshot,
        policy=CommunityReportPolicy(
            max_entities=8, max_relations=4, context_tokens=16_000,
        ),
    )
    context = plan["items"][0]["input_context"]

    assert len(context["relationships"]) == 4
    assert context["coverage"]["predicates_in_prompt"] >= 3
    assert context["coverage"]["documents_in_prompt"] >= 3
    assert any(
        item["predicate"] != "related_to"
        for item in context["relationships"]
    )


def test_parent_report_context_uses_child_topics_only_with_raw_relation_closure():
    community_run = {
        "run_id": "hierarchy-run",
        "input_hash": "hierarchy-input",
        "communities": [{
            "key": "community:parent",
            "label": "社区 0",
            "level": 1,
            "entity_keys": ["a", "b", "c", "d"],
        }, {
            "key": "community:left",
            "label": "社区 1",
            "level": 0,
            "parent": "community:parent",
            "entity_keys": ["a", "b"],
            "relationship_count": 1,
            "summary": "左侧子主题围绕甲与乙。",
            "topic_entities": ["甲", "乙"],
        }, {
            "key": "community:right",
            "label": "社区 2",
            "level": 0,
            "parent": "community:parent",
            "entity_keys": ["c", "d"],
            "relationship_count": 1,
            "summary": "右侧子主题围绕丙与丁。",
            "topic_entities": ["丙", "丁"],
        }],
    }
    plan = build_community_report_plan(
        community_run=community_run,
        snapshot=_snapshot(),
        policy=CommunityReportPolicy(context_tokens=16_000),
    )
    parent = next(
        item for item in plan["items"]
        if item["community_key"] == "community:parent"
    )["input_context"]
    known_refs = {item["ref"] for item in parent["relationships"]}

    assert {item["key"] for item in parent["subcommunities"]} == {
        "community:left", "community:right",
    }
    assert all(
        set(item["support_relation_refs"]).issubset(known_refs)
        and item["support_relation_refs"]
        for item in parent["subcommunities"]
    )
    assert parent["coverage"]["subcommunities_in_prompt"] == 2


def test_report_quality_gate_rejects_single_fact_view_of_rich_context():
    snapshot = _snapshot()
    snapshot["canonicals"].extend([{
        "canonical_key": f"x{index}",
        "name": f"扩展实体 {index}",
        "primary_type": "Concept",
        "aliases": [],
        "source_count": 1,
        "metadata": {},
    } for index in range(4)])
    snapshot["relation_endpoints"].extend([{
        "relation_key": f"extra-{index}",
        "source_canonical_key": "a",
        "target_canonical_key": f"x{index}",
        "relation_type": "related_to",
        "document_id": index + 2,
        "chunk_id": index + 4,
        "evidence_text": f"补充关系 {index}",
        "metadata": {},
    } for index in range(4)])
    plan = build_community_report_plan(
        community_run={
            "run_id": "quality-run",
            "input_hash": "quality-input",
            "communities": [{
                "key": "community:quality",
                "label": "社区 1",
                "level": 0,
                "entity_keys": [
                    "a", "b", "c", "d", "x0", "x1", "x2", "x3",
                ],
            }],
        },
        snapshot=snapshot,
        policy=CommunityReportPolicy(context_tokens=16_000),
    )
    context = plan["items"][0]["input_context"]
    relation_ref = context["relationships"][0]["ref"]

    with pytest.raises(ValueError, match="覆盖过窄"):
        validate_report_payload({
            "title": "扩展实体关系",
            "summary": "报告只选择了一项事实。",
            "findings": [{
                "title": "单一事实",
                "explanation": "仅说明一条已验证关系。",
                "entity_refs": [],
                "relation_refs": [relation_ref],
            }],
            "limitations": "",
        }, context)
