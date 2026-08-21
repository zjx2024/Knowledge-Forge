#!/usr/bin/env python3
"""Build and validate the frozen Chinese calibration supplement.

The declarations below are the manually reviewed annotation source.  The
script only computes exact character offsets, hashes and deterministic IDs; it
does not infer entities or relations with a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


DOCUMENTS: dict[str, dict[str, Any]] = {
    "zh-software-hierarchy": {
        "split": "dev",
        "domain": "embedded_software",
        "structure": ["numbered_heading", "hierarchical_list", "prose"],
        "text": """3 核心操作系统

ACore平台提供分区管理、任务管理、时间管理、任务间通信管理、健康监控和虚中断管理。

— 分区管理
  — 空间隔离
  — 分区间通信
— 任务管理
  — 调度表管理
  — 定时器管理
— 任务间通信管理
  — 消息队列
  — 互斥信号量
  — 事件

健康监控分为核心级、多分区级和分区级。日志模块输出故障记录，但不负责调度任务。
配置数据库保存调度表和分区参数。虚中断管理由核心操作系统提供。

列表中的同级项目是独立能力；相邻出现并不表示一个功能包含另一个功能。
""",
    },
    "zh-manufacturing-table": {
        "split": "dev",
        "domain": "manufacturing",
        "structure": ["prose", "pipe_table", "batch_record", "negation"],
        "text": """## 供应与检测记录

华宁材料厂位于江苏，生产电池隔膜和复合涂层。远航物流公司向华宁材料厂供应包装材料。电池隔膜应用于储能电池装配线。

| 产品 | 原料来源 | 检测方法 | 用途 |
| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |
| 复合涂层 | 陶瓷粉体 | 附着力试验 | 耐热保护 |

批次A-17由质量实验室检验，结果为合格。仓库保存批次A-17。
江苏不是供应商，检测方法也不是产品；表格中的相邻单元格不能自动建立额外关系。
""",
    },
    "zh-regulation-clauses": {
        "split": "holdout",
        "domain": "regulation",
        "structure": ["article_heading", "numbered_clauses", "obligation", "negation"],
        "text": """第十二条 申请与审查

（一）申请人应当向受理机构提交申请材料和安全报告。
（二）受理机构应在五个工作日内审查申请材料，并将审查结果通知申请人。
（三）申请材料缺失的，受理机构应要求申请人补正；补正通知应列明缺失项目。

安全报告是审批决定的依据。申请人并不因提交材料而自动成为许可持有人。
第（一）项与第（二）项是并列义务，不是相互包含的条款。
""",
    },
    "zh-narrative-coreference": {
        "split": "holdout",
        "domain": "maintenance_narrative",
        "structure": ["narrative", "coreference", "event_sequence", "explicit_negation"],
        "text": """清晨，林岚带着测温仪来到北站。她把仪器交给值班员周启，周启用它检查冷却泵。
检查结束后，他将测温仪放入蓝色工具箱，并把检测报告交还给林岚。
随后警报响起，但冷却泵没有停机。林岚阅读报告后通知维修组。
维修组赶到北站并更换了传感器。周启没有离开北站；“警报响起”也不等于“冷却泵停机”。
""",
    },
}


ENTITY_SPECS: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "zh-software-hierarchy": [
        ("zh-os:platform", "Technology", ("核心操作系统", "ACore平台")),
        ("zh-os:partition_management", "Technology", ("分区管理",)),
        ("zh-os:task_management", "Technology", ("任务管理",)),
        ("zh-os:time_management", "Technology", ("时间管理",)),
        ("zh-os:intertask_communication", "Technology", ("任务间通信管理",)),
        ("zh-os:health_monitoring", "Technology", ("健康监控",)),
        ("zh-os:virtual_interrupt_management", "Technology", ("虚中断管理",)),
        ("zh-os:space_isolation", "Technology", ("空间隔离",)),
        ("zh-os:interpartition_communication", "Technology", ("分区间通信",)),
        ("zh-os:schedule_management", "Technology", ("调度表管理",)),
        ("zh-os:timer_management", "Technology", ("定时器管理",)),
        ("zh-os:message_queue", "Technology", ("消息队列",)),
        ("zh-os:mutex_semaphore", "Technology", ("互斥信号量",)),
        ("zh-os:event", "Event", ("事件",)),
        ("zh-os:core_level", "GenericEntity", ("核心级",)),
        ("zh-os:multi_partition_level", "GenericEntity", ("多分区级",)),
        ("zh-os:partition_level", "GenericEntity", ("分区级",)),
        ("zh-os:log_module", "Technology", ("日志模块",)),
        ("zh-os:fault_record", "Product", ("故障记录",)),
        ("zh-os:scheduling_task", "Event", ("调度任务",)),
        ("zh-os:configuration_database", "Technology", ("配置数据库",)),
        ("zh-os:schedule_table", "Product", ("调度表",)),
        ("zh-os:partition_parameter", "Product", ("分区参数",)),
    ],
    "zh-manufacturing-table": [
        ("zh-mfg:factory", "Organization", ("华宁材料厂",)),
        ("zh-mfg:jiangsu", "Location", ("江苏",)),
        ("zh-mfg:separator", "Product", ("电池隔膜",)),
        ("zh-mfg:coating", "Product", ("复合涂层",)),
        ("zh-mfg:logistics", "Organization", ("远航物流公司",)),
        ("zh-mfg:packaging", "Product", ("包装材料",)),
        ("zh-mfg:assembly_line", "Technology", ("储能电池装配线",)),
        ("zh-mfg:resin", "Product", ("聚乙烯树脂",)),
        ("zh-mfg:thickness_measurement", "Event", ("厚度测量",)),
        ("zh-mfg:ceramic_powder", "Product", ("陶瓷粉体",)),
        ("zh-mfg:adhesion_test", "Event", ("附着力试验",)),
        ("zh-mfg:heat_protection", "GenericEntity", ("耐热保护",)),
        ("zh-mfg:batch_a17", "Product", ("批次A-17",)),
        ("zh-mfg:quality_lab", "Organization", ("质量实验室",)),
        ("zh-mfg:qualified", "GenericEntity", ("合格",)),
        ("zh-mfg:warehouse", "Organization", ("仓库",)),
        ("zh-mfg:supplier", "GenericEntity", ("供应商",)),
        ("zh-mfg:inspection_method", "GenericEntity", ("检测方法",)),
        ("zh-mfg:product", "GenericEntity", ("产品",)),
    ],
    "zh-regulation-clauses": [
        ("zh-law:article_12", "GenericEntity", ("第十二条", "申请与审查")),
        ("zh-law:clause_1", "GenericEntity", ("（一）", "第（一）项")),
        ("zh-law:clause_2", "GenericEntity", ("（二）", "第（二）项")),
        ("zh-law:clause_3", "GenericEntity", ("（三）",)),
        ("zh-law:applicant", "Person", ("申请人",)),
        ("zh-law:receiving_authority", "Organization", ("受理机构",)),
        ("zh-law:application_materials", "Product", ("申请材料", "提交材料")),
        ("zh-law:safety_report", "Product", ("安全报告",)),
        ("zh-law:five_workdays", "GenericEntity", ("五个工作日",)),
        ("zh-law:review", "Event", ("审查",)),
        ("zh-law:review_result", "Product", ("审查结果",)),
        ("zh-law:supplement", "Event", ("补正",)),
        ("zh-law:supplement_notice", "Product", ("补正通知",)),
        ("zh-law:missing_items", "Product", ("缺失项目",)),
        ("zh-law:approval_decision", "Event", ("审批决定",)),
        ("zh-law:permit_holder", "Person", ("许可持有人",)),
        ("zh-law:parallel_obligations", "GenericEntity", ("并列义务",)),
    ],
    "zh-narrative-coreference": [
        ("zh-story:lin_lan", "Person", ("林岚",)),
        ("zh-story:thermometer", "Product", ("测温仪", "仪器")),
        ("zh-story:north_station", "Location", ("北站",)),
        ("zh-story:zhou_qi", "Person", ("周启",)),
        ("zh-story:cooling_pump", "Technology", ("冷却泵",)),
        ("zh-story:inspection", "Event", ("检查",)),
        ("zh-story:blue_toolbox", "Product", ("蓝色工具箱",)),
        ("zh-story:inspection_report", "Product", ("检测报告", "报告")),
        ("zh-story:alarm", "Event", ("警报", "警报响起")),
        ("zh-story:shutdown", "Event", ("停机", "冷却泵停机")),
        ("zh-story:maintenance_team", "Organization", ("维修组",)),
        ("zh-story:sensor", "Technology", ("传感器",)),
    ],
}


def relation(
    subject: str,
    predicate: str,
    object_: str,
    evidence: str,
    subject_surface: str,
    object_surface: str,
    structure: str,
    *,
    tier: str = "core_direct",
    subject_occurrence: int = 0,
    object_occurrence: int = 0,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "evidence": evidence,
        "subject_surface": subject_surface,
        "object_surface": object_surface,
        "structure": structure,
        "adjudication_tier": tier,
        "subject_occurrence": subject_occurrence,
        "object_occurrence": object_occurrence,
    }


RELATIONS: dict[str, list[dict[str, Any]]] = {
    "zh-software-hierarchy": [
        relation("zh-os:platform", "provides", "zh-os:partition_management", "ACore平台提供分区管理、任务管理、时间管理、任务间通信管理、健康监控和虚中断管理。", "ACore平台", "分区管理", "prose"),
        relation("zh-os:platform", "provides", "zh-os:task_management", "ACore平台提供分区管理、任务管理、时间管理、任务间通信管理、健康监控和虚中断管理。", "ACore平台", "任务管理", "prose"),
        relation("zh-os:platform", "provides", "zh-os:time_management", "ACore平台提供分区管理、任务管理、时间管理、任务间通信管理、健康监控和虚中断管理。", "ACore平台", "时间管理", "prose"),
        relation("zh-os:platform", "provides", "zh-os:intertask_communication", "ACore平台提供分区管理、任务管理、时间管理、任务间通信管理、健康监控和虚中断管理。", "ACore平台", "任务间通信管理", "prose"),
        relation("zh-os:platform", "provides", "zh-os:health_monitoring", "ACore平台提供分区管理、任务管理、时间管理、任务间通信管理、健康监控和虚中断管理。", "ACore平台", "健康监控", "prose"),
        relation("zh-os:platform", "provides", "zh-os:virtual_interrupt_management", "ACore平台提供分区管理、任务管理、时间管理、任务间通信管理、健康监控和虚中断管理。", "ACore平台", "虚中断管理", "prose"),
        relation("zh-os:space_isolation", "part_of", "zh-os:partition_management", "— 分区管理\n  — 空间隔离", "空间隔离", "分区管理", "hierarchical_list"),
        relation("zh-os:interpartition_communication", "part_of", "zh-os:partition_management", "— 分区管理\n  — 空间隔离\n  — 分区间通信", "分区间通信", "分区管理", "hierarchical_list"),
        relation("zh-os:schedule_management", "part_of", "zh-os:task_management", "— 任务管理\n  — 调度表管理", "调度表管理", "任务管理", "hierarchical_list"),
        relation("zh-os:timer_management", "part_of", "zh-os:task_management", "— 任务管理\n  — 调度表管理\n  — 定时器管理", "定时器管理", "任务管理", "hierarchical_list"),
        relation("zh-os:message_queue", "part_of", "zh-os:intertask_communication", "— 任务间通信管理\n  — 消息队列", "消息队列", "任务间通信管理", "hierarchical_list"),
        relation("zh-os:mutex_semaphore", "part_of", "zh-os:intertask_communication", "— 任务间通信管理\n  — 消息队列\n  — 互斥信号量", "互斥信号量", "任务间通信管理", "hierarchical_list"),
        relation("zh-os:event", "part_of", "zh-os:intertask_communication", "— 任务间通信管理\n  — 消息队列\n  — 互斥信号量\n  — 事件", "事件", "任务间通信管理", "hierarchical_list"),
        relation("zh-os:health_monitoring", "has_level", "zh-os:core_level", "健康监控分为核心级、多分区级和分区级。", "健康监控", "核心级", "prose"),
        relation("zh-os:health_monitoring", "has_level", "zh-os:multi_partition_level", "健康监控分为核心级、多分区级和分区级。", "健康监控", "多分区级", "prose"),
        relation("zh-os:health_monitoring", "has_level", "zh-os:partition_level", "健康监控分为核心级、多分区级和分区级。", "健康监控", "分区级", "prose", object_occurrence=1),
        relation("zh-os:log_module", "outputs", "zh-os:fault_record", "日志模块输出故障记录，但不负责调度任务。", "日志模块", "故障记录", "prose"),
        relation("zh-os:configuration_database", "stores", "zh-os:schedule_table", "配置数据库保存调度表和分区参数。", "配置数据库", "调度表", "prose"),
        relation("zh-os:configuration_database", "stores", "zh-os:partition_parameter", "配置数据库保存调度表和分区参数。", "配置数据库", "分区参数", "prose"),
    ],
    "zh-manufacturing-table": [
        relation("zh-mfg:factory", "located_in", "zh-mfg:jiangsu", "华宁材料厂位于江苏，生产电池隔膜和复合涂层。", "华宁材料厂", "江苏", "prose"),
        relation("zh-mfg:factory", "produces", "zh-mfg:separator", "华宁材料厂位于江苏，生产电池隔膜和复合涂层。", "华宁材料厂", "电池隔膜", "prose"),
        relation("zh-mfg:factory", "produces", "zh-mfg:coating", "华宁材料厂位于江苏，生产电池隔膜和复合涂层。", "华宁材料厂", "复合涂层", "prose"),
        relation("zh-mfg:logistics", "supplies", "zh-mfg:packaging", "远航物流公司向华宁材料厂供应包装材料。", "远航物流公司", "包装材料", "prose"),
        relation("zh-mfg:packaging", "supplied_to", "zh-mfg:factory", "远航物流公司向华宁材料厂供应包装材料。", "包装材料", "华宁材料厂", "prose"),
        relation("zh-mfg:separator", "used_for", "zh-mfg:assembly_line", "电池隔膜应用于储能电池装配线。", "电池隔膜", "储能电池装配线", "prose"),
        relation("zh-mfg:separator", "derived_from", "zh-mfg:resin", "| 产品 | 原料来源 | 检测方法 | 用途 |\n| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |", "电池隔膜", "聚乙烯树脂", "table_header_bound"),
        relation("zh-mfg:separator", "checked_by", "zh-mfg:thickness_measurement", "| 产品 | 原料来源 | 检测方法 | 用途 |\n| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |", "电池隔膜", "厚度测量", "table_header_bound"),
        relation("zh-mfg:separator", "used_for", "zh-mfg:assembly_line", "| 产品 | 原料来源 | 检测方法 | 用途 |\n| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |", "电池隔膜", "储能电池装配线", "table_header_bound"),
        relation("zh-mfg:coating", "derived_from", "zh-mfg:ceramic_powder", "| 产品 | 原料来源 | 检测方法 | 用途 |\n| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |\n| 复合涂层 | 陶瓷粉体 | 附着力试验 | 耐热保护 |", "复合涂层", "陶瓷粉体", "table_header_bound"),
        relation("zh-mfg:coating", "checked_by", "zh-mfg:adhesion_test", "| 产品 | 原料来源 | 检测方法 | 用途 |\n| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |\n| 复合涂层 | 陶瓷粉体 | 附着力试验 | 耐热保护 |", "复合涂层", "附着力试验", "table_header_bound"),
        relation("zh-mfg:coating", "used_for", "zh-mfg:heat_protection", "| 产品 | 原料来源 | 检测方法 | 用途 |\n| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |\n| 复合涂层 | 陶瓷粉体 | 附着力试验 | 耐热保护 |", "复合涂层", "耐热保护", "table_header_bound"),
        relation("zh-mfg:batch_a17", "inspected_by", "zh-mfg:quality_lab", "批次A-17由质量实验室检验，结果为合格。", "批次A-17", "质量实验室", "batch_record"),
        relation("zh-mfg:batch_a17", "has_status", "zh-mfg:qualified", "批次A-17由质量实验室检验，结果为合格。", "批次A-17", "合格", "batch_record"),
        relation("zh-mfg:warehouse", "stores", "zh-mfg:batch_a17", "仓库保存批次A-17。", "仓库", "批次A-17", "prose"),
    ],
    "zh-regulation-clauses": [
        relation("zh-law:clause_1", "part_of", "zh-law:article_12", "第十二条 申请与审查\n\n（一）", "（一）", "第十二条", "section_hierarchy"),
        relation("zh-law:clause_2", "part_of", "zh-law:article_12", "第十二条 申请与审查\n\n（一）申请人应当向受理机构提交申请材料和安全报告。\n（二）", "（二）", "第十二条", "section_hierarchy"),
        relation("zh-law:clause_3", "part_of", "zh-law:article_12", "第十二条 申请与审查\n\n（一）申请人应当向受理机构提交申请材料和安全报告。\n（二）受理机构应在五个工作日内审查申请材料，并将审查结果通知申请人。\n（三）", "（三）", "第十二条", "section_hierarchy"),
        relation("zh-law:applicant", "submits", "zh-law:application_materials", "（一）申请人应当向受理机构提交申请材料和安全报告。", "申请人", "申请材料", "numbered_clause"),
        relation("zh-law:applicant", "submits", "zh-law:safety_report", "（一）申请人应当向受理机构提交申请材料和安全报告。", "申请人", "安全报告", "numbered_clause"),
        relation("zh-law:application_materials", "submitted_to", "zh-law:receiving_authority", "（一）申请人应当向受理机构提交申请材料和安全报告。", "申请材料", "受理机构", "numbered_clause"),
        relation("zh-law:safety_report", "submitted_to", "zh-law:receiving_authority", "（一）申请人应当向受理机构提交申请材料和安全报告。", "安全报告", "受理机构", "numbered_clause"),
        relation("zh-law:receiving_authority", "reviews", "zh-law:application_materials", "（二）受理机构应在五个工作日内审查申请材料，并将审查结果通知申请人。", "受理机构", "申请材料", "numbered_clause"),
        relation("zh-law:review", "has_deadline", "zh-law:five_workdays", "（二）受理机构应在五个工作日内审查申请材料，并将审查结果通知申请人。", "审查", "五个工作日", "numbered_clause"),
        relation("zh-law:receiving_authority", "notifies", "zh-law:review_result", "（二）受理机构应在五个工作日内审查申请材料，并将审查结果通知申请人。", "受理机构", "审查结果", "numbered_clause"),
        relation("zh-law:review_result", "notified_to", "zh-law:applicant", "（二）受理机构应在五个工作日内审查申请材料，并将审查结果通知申请人。", "审查结果", "申请人", "numbered_clause"),
        relation("zh-law:receiving_authority", "requires_action_from", "zh-law:applicant", "（三）申请材料缺失的，受理机构应要求申请人补正；补正通知应列明缺失项目。", "受理机构", "申请人", "numbered_clause"),
        relation("zh-law:applicant", "supplements", "zh-law:application_materials", "（三）申请材料缺失的，受理机构应要求申请人补正；补正通知应列明缺失项目。", "申请人", "申请材料", "numbered_clause", tier="obligation_inference"),
        relation("zh-law:supplement_notice", "lists", "zh-law:missing_items", "（三）申请材料缺失的，受理机构应要求申请人补正；补正通知应列明缺失项目。", "补正通知", "缺失项目", "numbered_clause"),
        relation("zh-law:safety_report", "basis_for", "zh-law:approval_decision", "安全报告是审批决定的依据。", "安全报告", "审批决定", "prose"),
    ],
    "zh-narrative-coreference": [
        relation("zh-story:lin_lan", "carries", "zh-story:thermometer", "清晨，林岚带着测温仪来到北站。", "林岚", "测温仪", "narrative"),
        relation("zh-story:lin_lan", "arrives_at", "zh-story:north_station", "清晨，林岚带着测温仪来到北站。", "林岚", "北站", "narrative"),
        relation("zh-story:lin_lan", "gives", "zh-story:thermometer", "她把仪器交给值班员周启，周启用它检查冷却泵。", "她", "仪器", "narrative_coreference"),
        relation("zh-story:thermometer", "given_to", "zh-story:zhou_qi", "她把仪器交给值班员周启，周启用它检查冷却泵。", "仪器", "周启", "narrative_coreference"),
        relation("zh-story:zhou_qi", "uses", "zh-story:thermometer", "她把仪器交给值班员周启，周启用它检查冷却泵。", "周启", "它", "narrative_coreference"),
        relation("zh-story:zhou_qi", "checks", "zh-story:cooling_pump", "她把仪器交给值班员周启，周启用它检查冷却泵。", "周启", "冷却泵", "narrative"),
        relation("zh-story:zhou_qi", "places", "zh-story:thermometer", "检查结束后，他将测温仪放入蓝色工具箱，并把检测报告交还给林岚。", "他", "测温仪", "narrative_coreference"),
        relation("zh-story:thermometer", "located_in", "zh-story:blue_toolbox", "检查结束后，他将测温仪放入蓝色工具箱，并把检测报告交还给林岚。", "测温仪", "蓝色工具箱", "narrative"),
        relation("zh-story:zhou_qi", "returns", "zh-story:inspection_report", "检查结束后，他将测温仪放入蓝色工具箱，并把检测报告交还给林岚。", "他", "检测报告", "narrative_coreference"),
        relation("zh-story:inspection_report", "returned_to", "zh-story:lin_lan", "检查结束后，他将测温仪放入蓝色工具箱，并把检测报告交还给林岚。", "检测报告", "林岚", "narrative"),
        relation("zh-story:lin_lan", "reads", "zh-story:inspection_report", "林岚阅读报告后通知维修组。", "林岚", "报告", "narrative"),
        relation("zh-story:lin_lan", "notifies", "zh-story:maintenance_team", "林岚阅读报告后通知维修组。", "林岚", "维修组", "narrative"),
        relation("zh-story:maintenance_team", "arrives_at", "zh-story:north_station", "维修组赶到北站并更换了传感器。", "维修组", "北站", "narrative"),
        relation("zh-story:maintenance_team", "replaces", "zh-story:sensor", "维修组赶到北站并更换了传感器。", "维修组", "传感器", "narrative"),
        relation("zh-story:zhou_qi", "remains_at", "zh-story:north_station", "周启没有离开北站；“警报响起”也不等于“冷却泵停机”。", "周启", "北站", "explicit_negation", tier="negation_paraphrase"),
    ],
}


NEGATIVES: dict[str, list[dict[str, str]]] = {
    "zh-software-hierarchy": [
        {"subject": "zh-os:task_management", "predicate": "part_of", "object": "zh-os:partition_management", "evidence": "— 分区管理\n  — 空间隔离\n  — 分区间通信\n— 任务管理", "subject_surface": "任务管理", "object_surface": "分区管理", "reason_code": "SIBLING_LIST_ITEMS"},
        {"subject": "zh-os:log_module", "predicate": "schedules", "object": "zh-os:scheduling_task", "evidence": "日志模块输出故障记录，但不负责调度任务。", "subject_surface": "日志模块", "object_surface": "调度任务", "reason_code": "EXPLICIT_NEGATION"},
        {"subject": "zh-os:message_queue", "predicate": "contains", "object": "zh-os:mutex_semaphore", "evidence": "— 任务间通信管理\n  — 消息队列\n  — 互斥信号量\n  — 事件", "subject_surface": "消息队列", "object_surface": "互斥信号量", "reason_code": "SIBLING_LIST_ITEMS"},
        {"subject": "zh-os:core_level", "predicate": "same_as", "object": "zh-os:partition_level", "evidence": "健康监控分为核心级、多分区级和分区级。", "subject_surface": "核心级", "object_surface": "分区级", "object_occurrence": 1, "reason_code": "COORDINATED_DISTINCT_ITEMS"},
    ],
    "zh-manufacturing-table": [
        {"subject": "zh-mfg:jiangsu", "predicate": "instance_of", "object": "zh-mfg:supplier", "evidence": "江苏不是供应商，检测方法也不是产品；表格中的相邻单元格不能自动建立额外关系。", "subject_surface": "江苏", "object_surface": "供应商", "reason_code": "EXPLICIT_NEGATION"},
        {"subject": "zh-mfg:inspection_method", "predicate": "instance_of", "object": "zh-mfg:product", "evidence": "江苏不是供应商，检测方法也不是产品；表格中的相邻单元格不能自动建立额外关系。", "subject_surface": "检测方法", "object_surface": "产品", "reason_code": "EXPLICIT_NEGATION"},
        {"subject": "zh-mfg:resin", "predicate": "used_for", "object": "zh-mfg:assembly_line", "evidence": "| 产品 | 原料来源 | 检测方法 | 用途 |\n| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |", "subject_surface": "聚乙烯树脂", "object_surface": "储能电池装配线", "reason_code": "SIBLING_COLUMNS"},
        {"subject": "zh-mfg:separator", "predicate": "part_of", "object": "zh-mfg:coating", "evidence": "| 电池隔膜 | 聚乙烯树脂 | 厚度测量 | 储能电池装配线 |\n| 复合涂层 | 陶瓷粉体 | 附着力试验 | 耐热保护 |", "subject_surface": "电池隔膜", "object_surface": "复合涂层", "reason_code": "SIBLING_ROWS"},
    ],
    "zh-regulation-clauses": [
        {"subject": "zh-law:applicant", "predicate": "same_as", "object": "zh-law:permit_holder", "evidence": "申请人并不因提交材料而自动成为许可持有人。", "subject_surface": "申请人", "object_surface": "许可持有人", "reason_code": "EXPLICIT_NEGATION"},
        {"subject": "zh-law:clause_1", "predicate": "part_of", "object": "zh-law:clause_2", "evidence": "第（一）项与第（二）项是并列义务，不是相互包含的条款。", "subject_surface": "第（一）项", "object_surface": "第（二）项", "reason_code": "SIBLING_CLAUSES"},
        {"subject": "zh-law:review_result", "predicate": "part_of", "object": "zh-law:application_materials", "evidence": "（二）受理机构应在五个工作日内审查申请材料，并将审查结果通知申请人。", "subject_surface": "审查结果", "object_surface": "申请材料", "reason_code": "SEPARATE_CLAUSE_ROLES"},
        {"subject": "zh-law:application_materials", "predicate": "grants_status", "object": "zh-law:permit_holder", "evidence": "申请人并不因提交材料而自动成为许可持有人。", "subject_surface": "提交材料", "object_surface": "许可持有人", "reason_code": "EXPLICIT_NEGATION"},
    ],
    "zh-narrative-coreference": [
        {"subject": "zh-story:alarm", "predicate": "causes", "object": "zh-story:shutdown", "evidence": "随后警报响起，但冷却泵没有停机。", "subject_surface": "警报", "object_surface": "停机", "reason_code": "EXPLICIT_NEGATION"},
        {"subject": "zh-story:zhou_qi", "predicate": "leaves", "object": "zh-story:north_station", "evidence": "周启没有离开北站；“警报响起”也不等于“冷却泵停机”。", "subject_surface": "周启", "object_surface": "北站", "reason_code": "EXPLICIT_NEGATION"},
        {"subject": "zh-story:alarm", "predicate": "same_as", "object": "zh-story:shutdown", "evidence": "周启没有离开北站；“警报响起”也不等于“冷却泵停机”。", "subject_surface": "警报响起", "object_surface": "冷却泵停机", "reason_code": "EXPLICIT_NEGATION"},
        {"subject": "zh-story:lin_lan", "predicate": "gives", "object": "zh-story:inspection_report", "evidence": "检查结束后，他将测温仪放入蓝色工具箱，并把检测报告交还给林岚。", "subject_surface": "林岚", "object_surface": "检测报告", "reason_code": "REVERSED_ROLE"},
    ],
}


COREFERENCE_SPECS = {
    "zh-narrative-coreference": [
        ("她", 0, "zh-story:lin_lan"),
        ("它", 0, "zh-story:thermometer"),
        ("他", 0, "zh-story:zhou_qi"),
    ],
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _all_spans(text: str, surface: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(surface, cursor)
        if start < 0:
            return spans
        spans.append((start, start + len(surface)))
        cursor = start + max(1, len(surface))


def _nth_span(text: str, surface: str, occurrence: int = 0) -> tuple[int, int]:
    spans = _all_spans(text, surface)
    if not 0 <= occurrence < len(spans):
        raise ValueError(f"surface not found: {surface!r} occurrence={occurrence}")
    return spans[occurrence]


def _build_entities() -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    documents: dict[str, list[dict[str, Any]]] = {}
    first_mentions: dict[str, dict[str, str]] = {}
    for document_id, specs in ENTITY_SPECS.items():
        text = DOCUMENTS[document_id]["text"]
        candidates: list[tuple[int, int, str, str, str]] = []
        for canonical_id, type_, aliases in specs:
            for alias in aliases:
                matches = _all_spans(text, alias)
                if not matches:
                    raise ValueError(
                        f"{document_id}: entity alias absent: {alias!r}"
                    )
                candidates.extend(
                    (start, end, alias, canonical_id, type_)
                    for start, end in matches
                )
        # Prefer a manually declared longer surface over its nested alias.
        selected: list[tuple[int, int, str, str, str]] = []
        for candidate in sorted(
            candidates, key=lambda row: (row[0], -(row[1] - row[0]), row[3])
        ):
            start, end, _surface, canonical_id, _type = candidate
            containing = next((
                row for row in selected
                if row[0] <= start and end <= row[1]
            ), None)
            if containing is not None:
                # Chinese domain terms frequently contain another valid term
                # (for example, “多分区级” contains “分区级”).  The frozen
                # mention policy is longest explicit term at one location;
                # the shorter term is still annotated at standalone uses.
                continue
            if any(start < row[1] and row[0] < end for row in selected):
                raise ValueError(f"{document_id}: overlapping entity mentions")
            selected.append(candidate)
        rows: list[dict[str, Any]] = []
        first_mentions[document_id] = {}
        for index, (start, end, surface, canonical_id, type_) in enumerate(
            sorted(selected), 1
        ):
            mention_id = f"{document_id}:m{index:03d}"
            rows.append({
                "mention_id": mention_id,
                "surface": surface,
                "span": [start, end],
                "canonical_id": canonical_id,
                "type": type_,
                "label_source": "assistant_expert_annotator",
                "review_status": "single_expert_reviewed",
            })
            first_mentions[document_id].setdefault(canonical_id, mention_id)
        documents[document_id] = rows
    return {"documents": documents}, first_mentions


def _evidence_span(text: str, evidence: str) -> tuple[int, int]:
    spans = _all_spans(text, evidence)
    if len(spans) != 1:
        raise ValueError(
            f"evidence must occur exactly once ({len(spans)}): {evidence!r}"
        )
    return spans[0]


def _surface_in_evidence(
    text: str, evidence: str, surface: str, occurrence: int = 0,
) -> tuple[int, int]:
    evidence_start, _ = _evidence_span(text, evidence)
    local_spans = _all_spans(evidence, surface)
    if not 0 <= occurrence < len(local_spans):
        raise ValueError(f"endpoint {surface!r} absent from evidence")
    local_start, local_end = local_spans[occurrence]
    return evidence_start + local_start, evidence_start + local_end


def _build_assertions() -> dict[str, Any]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for document_id in DOCUMENTS:
        text = DOCUMENTS[document_id]["text"]
        for index, spec in enumerate(RELATIONS[document_id], 1):
            evidence_start, evidence_end = _evidence_span(
                text, spec["evidence"]
            )
            positives.append({
                "assertion_id": f"{document_id}:a{index:03d}",
                "document_id": document_id,
                "subject_canonical_id": spec["subject"],
                "subject_span": list(_surface_in_evidence(
                    text, spec["evidence"], spec["subject_surface"],
                    int(spec.get("subject_occurrence") or 0),
                )),
                "predicate": spec["predicate"],
                "object_canonical_id": spec["object"],
                "object_span": list(_surface_in_evidence(
                    text, spec["evidence"], spec["object_surface"],
                    int(spec.get("object_occurrence") or 0),
                )),
                "evidence_span": [evidence_start, evidence_end],
                "evidence_text": spec["evidence"],
                "structure": spec["structure"],
                "schema_status": "extension",
                "adjudication_tier": spec["adjudication_tier"],
                "label_source": "assistant_expert_annotator",
                "review_status": "single_expert_reviewed",
            })
        for index, spec in enumerate(NEGATIVES[document_id], 1):
            evidence_start, evidence_end = _evidence_span(
                text, spec["evidence"]
            )
            negatives.append({
                "claim_id": f"{document_id}:n{index:03d}",
                "document_id": document_id,
                "subject_canonical_id": spec["subject"],
                "subject_span": list(_surface_in_evidence(
                    text, spec["evidence"], spec["subject_surface"],
                    int(spec.get("subject_occurrence") or 0),
                )),
                "predicate": spec["predicate"],
                "object_canonical_id": spec["object"],
                "object_span": list(_surface_in_evidence(
                    text, spec["evidence"], spec["object_surface"],
                    int(spec.get("object_occurrence") or 0),
                )),
                "evidence_span": [evidence_start, evidence_end],
                "evidence_text": spec["evidence"],
                "reason_code": spec["reason_code"],
                "label_source": "assistant_expert_annotator",
                "review_status": "single_expert_reviewed",
            })
    return {
        "annotation_policy": (
            "Complete binary source assertions under the declared entity and "
            "relation policy; optional inference tiers are reported separately."
        ),
        "relations": positives,
        "negative_claims": negatives,
    }


def _build_coreference(
    first_mentions: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document_id, specs in COREFERENCE_SPECS.items():
        text = DOCUMENTS[document_id]["text"]
        for index, (surface, occurrence, canonical_id) in enumerate(specs, 1):
            rows.append({
                "link_id": f"{document_id}:coref{index:02d}",
                "document_id": document_id,
                "source_span": list(_nth_span(text, surface, occurrence)),
                "source_text": surface,
                "target_mention_id": first_mentions[document_id][canonical_id],
                "relation": "refers_to",
                "label_source": "assistant_expert_annotator",
                "review_status": "single_expert_reviewed",
            })
    return rows


def build_payloads() -> dict[Path, Any]:
    entities, first_mentions = _build_entities()
    assertions = _build_assertions()
    coreference = _build_coreference(first_mentions)
    relation_types = sorted({
        item["predicate"] for item in assertions["relations"]
    })
    entity_types = sorted({
        item["type"]
        for rows in entities["documents"].values() for item in rows
    })
    documents = []
    for document_id, spec in DOCUMENTS.items():
        documents.append({
            "id": document_id,
            "path": f"documents/{document_id}.txt",
            "split": spec["split"],
            "content_sha256": _sha(spec["text"]),
            "language": "zh-CN",
            "domain": spec["domain"],
            "structure": spec["structure"],
        })
    unique_facts = {
        (
            item["subject_canonical_id"], item["predicate"],
            item["object_canonical_id"],
        )
        for item in assertions["relations"]
    }
    manifest = {
        "dataset": "knowledge-forge-cross-domain-zh",
        "version": "v1",
        "level": "calibration_supplement",
        "status": "single_expert_reviewed_calibration",
        "label_source": "assistant_expert_annotator",
        "language": "zh-CN",
        "frozen_at": "2026-08-11T00:00:00Z",
        "recall_universe_complete": True,
        "precision_adjudication_universe": (
            "targeted hard negatives; not the entity Cartesian product"
        ),
        "production_gate_eligible": False,
        "trusted_approval": None,
        "ontology_path": "ontology.json",
        "sources_path": "SOURCES.json",
        "annotations": {
            "entities": "annotations/entities.json",
            "coreference_links": "annotations/coreference_links.json",
            "assertions": "annotations/assertions.json",
            "ambiguous_claims": "annotations/ambiguous_claims.json",
        },
        "documents": documents,
        "expected_counts": {
            "documents": len(documents),
            "entity_mentions": sum(
                len(rows) for rows in entities["documents"].values()
            ),
            "canonical_entities": sum(
                len({item["canonical_id"] for item in rows})
                for rows in entities["documents"].values()
            ),
            "coreference_links": len(coreference),
            "complete_positive_assertions": len(assertions["relations"]),
            "unique_positive_facts": len(unique_facts),
            "predicate_specific_negative_claims": len(
                assertions["negative_claims"]
            ),
        },
    }
    sources = {
        "notes": (
            "These Chinese documents are expert-authored synthetic calibration "
            "fixtures. They contain no uploaded or proprietary source text."
        ),
        "documents": [{
            "id": document_id,
            "origin": "expert-authored synthetic calibration fixture",
            "license_note": "repository test fixture",
            "content_sha256": _sha(spec["text"]),
        } for document_id, spec in DOCUMENTS.items()],
    }
    ontology = {
        "version": "kf-cross-domain-zh-predicates-v1",
        "entity_types": entity_types,
        "relation_types": relation_types,
        "notes": {
            "direction": "All relation types use declared subject-to-object direction.",
            "tiering": (
                "core_direct is the primary score; obligation_inference and "
                "negation_paraphrase form an optional interpretation band."
            ),
        },
    }
    payloads: dict[Path, Any] = {
        ROOT / "manifest.json": manifest,
        ROOT / "SOURCES.json": sources,
        ROOT / "ontology.json": ontology,
        ROOT / "annotations" / "entities.json": entities,
        ROOT / "annotations" / "assertions.json": assertions,
        ROOT / "annotations" / "coreference_links.json": coreference,
        ROOT / "annotations" / "ambiguous_claims.json": {
            "annotation_policy": "No unresolved cases in this supplement.",
            "claims": [],
        },
    }
    for document_id, spec in DOCUMENTS.items():
        payloads[ROOT / "documents" / f"{document_id}.txt"] = spec["text"]
    return payloads


def _render(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payloads = build_payloads()
    mismatches: list[str] = []
    for path, value in payloads.items():
        rendered = _render(value)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                mismatches.append(str(path.relative_to(ROOT)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    if mismatches:
        raise SystemExit("out-of-date generated files: " + ", ".join(mismatches))
    manifest = payloads[ROOT / "manifest.json"]
    print(json.dumps({
        "status": "valid" if args.check else "built",
        **manifest["expected_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
