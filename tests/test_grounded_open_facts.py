import json

import pytest

from knowledge_forge.relation_extraction import (
    GROUNDED_RELATION_OPEN_FACT_PRIMARY_CONTRACT_VERSION,
    GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION,
    GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION,
    RelationClassificationError,
    augment_grounded_open_fact_inventory,
    deterministic_source_structure_schema_bindings,
    grounded_open_fact_prompt,
    grounded_open_fact_response_schema,
    parse_grounded_open_facts,
    parse_relation_reference_resolutions,
    relation_reference_resolution_prompt,
    relation_reference_resolution_response_schema,
)
from tests.grounded_relation_fixtures import logical_schema, predicate_policy


def _occurrence(key: str, surface: str, start: int, end: int, canonical: str):
    return {
        "occurrence_key": key,
        "surface": surface,
        "mention_start": start,
        "mention_end": end,
        "canonical_id": canonical,
        "type": "GenericEntity",
        "scope_id": "sentence-1",
        "scope_kind": "sentence",
    }


def test_grounded_open_fact_contract_uses_short_frozen_mention_ids():
    content = "A controller sends a message to a receiver."
    occurrences = [
        _occurrence("occ-controller", "controller", 2, 12, "c-controller"),
        _occurrence("occ-message", "message", 21, 28, "c-message"),
        _occurrence("occ-receiver", "receiver", 34, 42, "c-receiver"),
    ]

    payload = json.loads(grounded_open_fact_prompt(
        context_id="ctx-1",
        content=content,
        occurrences=occurrences,
        context_kind="prose",
    ))
    schema = grounded_open_fact_response_schema(occurrences)

    assert [item["id"] for item in payload["mentions"]] == [
        "M0000", "M0001", "M0002",
    ]
    assert "occ-controller" not in json.dumps(payload)
    assert schema["properties"]["facts"]["items"]["properties"][
        "subject_id"
    ]["anyOf"][0]["enum"] == ["M0000", "M0001", "M0002"]
    assert schema["required"] == ["resolved_references", "facts"]
    assert "predefined logical relation" in payload["instruction"]
    assert "exact contiguous relation phrase" in payload["instruction"]
    assert payload["contract_version"] == (
        GROUNDED_RELATION_OPEN_FACT_PRIMARY_CONTRACT_VERSION
    )


def test_soft_wrapped_lexicalized_qi_is_not_a_reference_candidate():
    content = "尤 其在数据量较大、临床价值更高的客户端上。"
    occurrences = []
    for key, surface in (
        ("amount", "数据量"),
        ("value", "临床价值"),
        ("client", "客户端"),
    ):
        start = content.index(surface)
        occurrences.append(
            _occurrence(key, surface, start, start + len(surface), key)
        )

    payload = json.loads(grounded_open_fact_prompt(
        context_id="ctx-soft-wrap-adverb",
        content=content,
        occurrences=occurrences,
        context_kind="prose",
    ))

    assert not any(
        item["surface"] == "其" for item in payload["reference_candidates"]
    )


def test_chinese_reference_glyphs_inside_lexical_items_are_not_candidates():
    content = "其他客户端因此应该更新模型，其模型随后保持可用。"
    model_start = content.index("模型", content.index("其模型"))
    occurrences = [
        _occurrence("model", "模型", model_start, model_start + 2, "model"),
    ]

    payload = json.loads(grounded_open_fact_prompt(
        context_id="ctx-reference-token-boundary",
        content=content,
        occurrences=occurrences,
        context_kind="prose",
    ))

    candidates = payload["reference_candidates"]
    assert [(item["surface"], item["start"]) for item in candidates] == [
        ("其", content.index("其模型")),
    ]


def test_grounded_open_fact_recovery_targets_uncovered_mentions_and_roles():
    content = "A controller sends a message to a receiver."
    occurrences = [
        _occurrence("occ-controller", "controller", 2, 12, "c-controller"),
        _occurrence("occ-message", "message", 21, 28, "c-message"),
        _occurrence("occ-receiver", "receiver", 34, 42, "c-receiver"),
    ]

    payload = json.loads(grounded_open_fact_prompt(
        context_id="ctx-1",
        content=content,
        occurrences=occurrences,
        already_found=[{
            "argument_occurrence_keys": ["occ-controller", "occ-message"],
            "relation": "sends",
            "evidence_text": content,
        }],
    ))

    assert payload["coverage_audit"][
        "mention_ids_without_accepted_fact"
    ] == ["M0002"]
    assert payload["coverage_audit"]["multi_mention_source_units"] == [{
        "unit_id": "U0000",
        "start": 0,
        "end": len(content),
        "mention_ids": ["M0000", "M0001", "M0002"],
        "accepted_fact_count": 1,
    }]
    assert payload["contract_version"] == (
        GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION
    )


def test_grounded_open_fact_empty_recovery_is_distinct_from_empty_primary():
    content = "A controller sends a message."
    occurrences = [
        _occurrence("occ-controller", "controller", 2, 12, "c-controller"),
        _occurrence("occ-message", "message", 21, 28, "c-message"),
    ]

    primary = json.loads(grounded_open_fact_prompt(
        context_id="ctx-empty-recovery",
        content=content,
        occurrences=occurrences,
    ))
    recovery = json.loads(grounded_open_fact_prompt(
        context_id="ctx-empty-recovery",
        content=content,
        occurrences=occurrences,
        already_found=[],
        recovery_pass=True,
    ))

    assert primary["contract_version"] == (
        GROUNDED_RELATION_OPEN_FACT_PRIMARY_CONTRACT_VERSION
    )
    assert recovery["contract_version"] == (
        GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION
    )
    assert recovery["coverage_audit"][
        "mention_ids_without_accepted_fact"
    ] == ["M0000", "M0001"]
    assert recovery != primary


def test_reference_adjudication_excludes_nonreflexive_other_endpoint():
    content = (
        "Alpha remained the continuing subject. "
        "Beta with Markers moved close by her."
    )
    alpha_start = content.index("Alpha")
    beta_start = content.index("Beta")
    markers_start = content.index("Markers")
    occurrences = [
        _occurrence(
            "alpha", "Alpha", alpha_start, alpha_start + 5, "alpha",
        ),
        _occurrence("beta", "Beta", beta_start, beta_start + 4, "beta"),
        _occurrence(
            "markers", "Markers", markers_start, markers_start + 7,
            "markers",
        ),
    ]
    inventory = {"frames": [{
        "argument_occurrence_keys": ["alpha", "beta"],
    }]}
    payload = json.loads(relation_reference_resolution_prompt(
        content=content, occurrences=occurrences, inventory=inventory,
    ))
    schema = relation_reference_resolution_response_schema(
        content=content, occurrences=occurrences, inventory=inventory,
    )

    assert len(payload["references"]) == 1
    reference = payload["references"][0]
    assert reference["reference_role"] == (
        "predicate_complement_after_modifier"
    )
    assert reference["other_endpoint"]["surface"] == "Beta"
    candidate_by_surface = {
        item["representative_surface"]: item["antecedent_id"]
        for item in reference["candidates"]
    }
    assert "Beta" not in candidate_by_surface
    assert schema["additionalProperties"] is False

    result = parse_relation_reference_resolutions(
        {
            "coverage_complete": True,
            "resolutions": [{
                "reference_id": reference["reference_id"],
                "antecedent_id": candidate_by_surface["Alpha"],
                "unambiguous": True,
            }],
        },
        content=content,
        occurrences=occurrences,
        inventory=inventory,
    )
    assert len(result["reference_proxies"]) == 1
    assert result["reference_proxies"][0]["canonical_id"] == "alpha"
    assert result["reference_proxies"][0][
        "reference_resolution_source"
    ] == "DISCOURSE_ROLE_ADJUDICATOR"


def test_reference_adjudication_quarantines_neuter_person_over_closer_object():
    content = "Alice did not enter the hole because she saw the Rabbit enter it."

    def occurrence(key, surface, entity_type, *, start=None):
        offset = content.index(surface) if start is None else start
        return {
            **_occurrence(key, surface, offset, offset + len(surface), key),
            "type": entity_type,
        }

    occurrences = [
        occurrence("alice", "Alice", "Person"),
        occurrence("hole", "hole", "Location"),
        occurrence("rabbit", "Rabbit", "Person"),
    ]
    payload = json.loads(relation_reference_resolution_prompt(
        content=content, occurrences=occurrences, inventory={"frames": []},
    ))
    target = next(
        item for item in payload["references"] if item["surface"] == "it"
    )
    assert target["reference_role"] == "predicate_complement"
    alice_id = next(
        item["antecedent_id"] for item in target["candidates"]
        if item["representative_surface"] == "Alice"
    )

    result = parse_relation_reference_resolutions(
        {
            "coverage_complete": True,
            "resolutions": [{
                "reference_id": target["reference_id"],
                "antecedent_id": alice_id,
                "unambiguous": True,
            }],
        },
        content=content,
        occurrences=occurrences,
        inventory={"frames": []},
        allow_partial=True,
    )

    assert result["reference_proxies"] == []
    assert result["coverage_complete"] is False
    assert result["dropped_resolutions"] == [{
        "item_index": 0,
        "reason": "neuter_reference_conflicts_with_closer_nonperson",
    }]


def test_reference_adjudication_resolves_filtered_chinese_personal_pronoun():
    content = "林岚来到北站。她把仪器交给周启。"

    def occurrence(key, surface, entity_type):
        start = content.index(surface)
        return {
            **_occurrence(key, surface, start, start + len(surface), key),
            "type": entity_type,
        }

    occurrences = [
        occurrence("lin", "林岚", "Person"),
        occurrence("station", "北站", "Location"),
        occurrence("instrument", "仪器", "Product"),
        occurrence("zhou", "周启", "Person"),
    ]
    payload = json.loads(relation_reference_resolution_prompt(
        content=content, occurrences=occurrences, inventory={"frames": []},
    ))
    target = next(
        item for item in payload["references"] if item["surface"] == "她"
    )
    assert target["reference_role"] == "grammatical_subject"
    lin_id = next(
        item["antecedent_id"] for item in target["candidates"]
        if item["representative_surface"] == "林岚"
    )

    result = parse_relation_reference_resolutions(
        {
            "coverage_complete": True,
            "resolutions": [{
                "reference_id": target["reference_id"],
                "antecedent_id": lin_id,
                "unambiguous": True,
            }],
        },
        content=content,
        occurrences=occurrences,
        inventory={"frames": []},
    )

    assert result["reference_proxies"][0]["canonical_id"] == "lin"
    assert result["reference_proxies"][0]["surface"] == "她"


def test_reference_adjudication_resolves_possessive_determiner_and_preserves_role():
    content = "A Device used its Module."
    device_start = content.index("Device")
    module_start = content.index("Module")
    occurrences = [
        _occurrence(
            "device", "Device", device_start, device_start + 6,
            "device",
        ),
        _occurrence(
            "module", "Module", module_start, module_start + 6,
            "module",
        ),
    ]
    payload = json.loads(relation_reference_resolution_prompt(
        content=content,
        occurrences=occurrences,
        inventory={"frames": []},
    ))
    assert len(payload["references"]) == 1
    reference = payload["references"][0]
    assert reference["surface"] == "its"
    assert reference["reference_role"] == "possessive_determiner"
    assert reference["other_endpoint"]["occurrence_key"] == "module"
    antecedent_id = next(
        item["antecedent_id"] for item in reference["candidates"]
        if item["representative_surface"] == "Device"
    )
    resolved = parse_relation_reference_resolutions(
        {
            "coverage_complete": True,
            "resolutions": [{
                "reference_id": reference["reference_id"],
                "antecedent_id": antecedent_id,
                "unambiguous": True,
            }],
        },
        content=content,
        occurrences=occurrences,
        inventory={"frames": []},
    )
    proxy = resolved["reference_proxies"][0]
    assert proxy["canonical_id"] == "device"
    assert proxy["reference_role"] == "possessive_determiner"
    assert proxy["reference_other_endpoint_occurrence_key"] == "module"

    augmented = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[*occurrences, proxy],
    )
    frame = next(
        item for item in augmented["frames"]
        if item.get("deterministic_source_syntax")
        == "possessive_reference_construction"
    )
    assert frame["argument_occurrence_keys"] == [
        proxy["occurrence_key"], "module",
    ]
    assert frame["roles"] == ["possessor", "possessed"]
    assert frame["resolved_reference_endpoints"][0][
        "antecedent_occurrence_key"
    ] == "device"


def test_possessive_topology_annotates_unkeyed_atomic_model_frame_by_surface():
    content = "A Device used its Module."
    device_start = content.index("Device")
    module_start = content.index("Module")
    its_start = content.index("its")
    device = {
        **_occurrence(
            "device", "Device", device_start, device_start + 6, "device",
        ),
        "type": "Product",
    }
    module = {
        **_occurrence(
            "module", "Module", module_start, module_start + 6, "module",
        ),
        "type": "Product",
    }
    proxy = {
        **_occurrence(
            "reference:its", "its", its_start, its_start + 3, "device",
        ),
        "type": "Product",
        "relation_reference_proxy": True,
        "reference_role": "possessive_determiner",
        "reference_antecedent_occurrence_key": "device",
        "reference_other_endpoint_occurrence_key": "module",
    }
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "contained",
            "arguments": ["its", "Module"],
            "roles": ["subject", "object"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["its", "Module"],
            "assertion_status": "ASSERTED",
        }]},
        content=content,
        occurrences=[device, module, proxy],
    )

    atomic = result["frames"][0]
    assert atomic["parser_inferred_source_roles"] == [
        "possessor", "possessed",
    ]
    assert atomic["parser_inferred_source_syntax"] == (
        "possessive_reference_construction"
    )


def test_reference_adjudication_does_not_treat_demonstrative_as_possession():
    content = "A Device used this Module."
    device_start = content.index("Device")
    module_start = content.index("Module")
    payload = json.loads(relation_reference_resolution_prompt(
        content=content,
        occurrences=[
            _occurrence(
                "device", "Device", device_start, device_start + 6,
                "device",
            ),
            _occurrence(
                "module", "Module", module_start, module_start + 6,
                "module",
            ),
        ],
        inventory={"frames": []},
    ))
    assert payload["references"] == []


def test_parse_grounded_open_facts_preserves_exact_endpoints_and_evidence():
    content = "A controller sends a message to a receiver."
    occurrences = [
        _occurrence("occ-controller", "controller", 2, 12, "c-controller"),
        _occurrence("occ-message", "message", 21, 28, "c-message"),
        _occurrence("occ-receiver", "receiver", 34, 42, "c-receiver"),
    ]

    parsed = parse_grounded_open_facts(
        {
            "facts": [{
                "subject_id": "M0000",
                "predicate": "sends",
                "object_id": "M0001",
                "evidence": content,
            }],
        },
        content=content,
        occurrences=occurrences,
    )

    assert parsed["coverage_complete"] is True
    assert parsed["parser_version"] == GROUNDED_RELATION_OPEN_FACT_PARSER_VERSION
    assert parsed["frames"] == [{
        "relation": "sends",
        "arguments": ["controller", "message"],
        "roles": ["subject", "object"],
        "argument_kinds": ["ENTITY", "ENTITY"],
        "entity_heads": ["controller", "message"],
        "assertion_status": "ASSERTED",
        "argument_occurrence_keys": ["occ-controller", "occ-message"],
        "evidence_span": [0, len(content)],
        "evidence_text": content,
        "source_relation_grounding": "EXACT_SOURCE_EXPRESSION",
    }]


def test_parse_grounded_open_facts_rejects_overlapping_endpoint_slices():
    content = "requirements capture"
    occurrences = [
        _occurrence(
            "requirements", "requirements", 0,
            len("requirements"), "c-requirements",
        ),
        _occurrence(
            "requirements-capture", "requirements capture", 0,
            len(content), "c-requirements-capture",
        ),
    ]

    parsed = parse_grounded_open_facts(
        {
            "resolved_references": [],
            "facts": [{
                "subject_id": "M0000",
                "predicate": "is part of",
                "object_id": "M0001",
                "evidence": content,
            }],
        },
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert parsed["frames"] == []
    assert parsed["dropped_items"] == [{
        "item_index": 0,
        "reason": "overlapping_endpoint_occurrences",
    }]


def test_parse_grounded_open_facts_narrows_copied_sentence_to_atomic_scope():
    content = (
        "A client sends a request to a server; "
        "HTTP messages travel over a connection."
    )
    messages_start = content.index("HTTP messages")
    connection_start = content.index("connection")
    occurrences = [
        _occurrence(
            "occ-messages", "HTTP messages", messages_start,
            messages_start + len("HTTP messages"), "c-messages",
        ),
        _occurrence(
            "occ-connection", "connection", connection_start,
            connection_start + len("connection"), "c-connection",
        ),
    ]

    parsed = parse_grounded_open_facts(
        {"facts": [{
            "subject_id": "M0000",
            "predicate": "travel over",
            "object_id": "M0001",
            "evidence": content,
        }]},
        content=content,
        occurrences=occurrences,
    )

    frame = parsed["frames"][0]
    assert frame["evidence_text"] == (
        "HTTP messages travel over a connection."
    )
    assert frame["evidence_span"] == [messages_start, len(content)]
    assert parsed["narrowed_evidence_scopes"] == 1


def test_parse_grounded_open_facts_adds_relation_only_reference_proxy():
    content = "Alice waited. Then she opened the book."
    alice_start = content.index("Alice")
    she_start = content.index("she")
    book_start = content.index("book")
    occurrences = [
        {
            **_occurrence(
                "occ-alice", "Alice", alice_start, alice_start + 5,
                "c-alice",
            ),
            "mention_id": "mention-alice",
            "assignment_version": "assign-v1",
            "type": "Person",
        },
        {
            **_occurrence(
                "occ-book", "book", book_start, book_start + 4,
                "c-book",
            ),
            "mention_id": "mention-book",
            "type": "Product",
        },
    ]

    parsed = parse_grounded_open_facts(
        {
            "resolved_references": [{
                "reference_id": "R0001",
                "surface": "she",
                "start": she_start,
                "end": she_start + 3,
                "antecedent_id": "M0000",
            }],
            "facts": [{
                "subject_id": "R0001",
                "predicate": "opened",
                "object_id": "M0001",
                "evidence": "Then she opened the book.",
            }],
        },
        content=content,
        occurrences=occurrences,
    )

    assert parsed["coverage_complete"] is True
    assert parsed["dropped_references"] == []
    assert len(parsed["reference_proxies"]) == 1
    proxy = parsed["reference_proxies"][0]
    assert proxy["mention_id"] == "mention-alice"
    assert proxy["canonical_id"] == "c-alice"
    assert proxy["assignment_version"] == "assign-v1"
    assert proxy["relation_reference_proxy"] is True
    assert proxy["reference_antecedent_occurrence_key"] == "occ-alice"
    assert parsed["frames"][0]["argument_occurrence_keys"] == [
        proxy["occurrence_key"], "occ-book",
    ]
    assert parsed["frames"][0]["resolved_reference_endpoints"] == [{
        "role": "subject",
        "occurrence_key": proxy["occurrence_key"],
        "antecedent_occurrence_key": "occ-alice",
        "reference_span": [she_start, she_start + 3],
        "reference_text": "she",
    }]


def test_reference_nominal_prefix_is_removed_from_open_predicate():
    content = "实时操作系统核心服务\n该核心服务主要包括任务管理。"
    antecedent_start = content.index("实时操作系统核心服务")
    reference_start = content.index("该")
    task_start = content.index("任务管理")
    occurrences = [
        _occurrence(
            "core-service", "实时操作系统核心服务",
            antecedent_start, antecedent_start + len("实时操作系统核心服务"),
            "c-core-service",
        ),
        _occurrence(
            "task-management", "任务管理", task_start,
            task_start + len("任务管理"), "c-task-management",
        ),
    ]

    parsed = parse_grounded_open_facts(
        {
            "resolved_references": [{
                "reference_id": "R0001",
                "surface": "该",
                "start": reference_start,
                "end": reference_start + 1,
                "antecedent_id": "M0000",
            }],
            "facts": [{
                "subject_id": "R0001",
                "predicate": "核心服务主要包括",
                "object_id": "M0001",
                "evidence": "该核心服务主要包括任务管理。",
            }],
        },
        content=content,
        occurrences=occurrences,
    )

    assert parsed["reference_nominal_prefix_repairs"] == 1
    assert parsed["frames"][0]["relation"] == "主要包括"
    assert parsed["frames"][0]["source_relation_original_expression"] == (
        "核心服务主要包括"
    )
    assert parsed["frames"][0]["source_relation_repair"] == (
        "resolved_reference_nominal_prefix"
    )


def test_augmented_reference_frame_removes_nominal_predicate_prefix():
    content = "实时操作系统核心服务\n该核心服务主要包括任务管理。"
    antecedent_start = content.index("实时操作系统核心服务")
    reference_start = content.index("该")
    task_start = content.index("任务管理")
    antecedent = _occurrence(
        "core-service", "实时操作系统核心服务",
        antecedent_start, antecedent_start + len("实时操作系统核心服务"),
        "c-core-service",
    )
    task = _occurrence(
        "task-management", "任务管理", task_start,
        task_start + len("任务管理"), "c-task-management",
    )
    proxy = {
        **_occurrence(
            "core-service-reference", "该", reference_start,
            reference_start + 1, "c-core-service",
        ),
        "relation_reference_proxy": True,
        "reference_role": "grammatical_subject",
        "reference_antecedent_occurrence_key": "core-service",
    }

    augmented = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[antecedent, task, proxy],
    )

    repaired = next(
        frame for frame in augmented["frames"]
        if frame.get("source_relation_repair")
        == "resolved_reference_nominal_prefix"
    )
    assert repaired["relation"] == "主要包括"
    assert repaired["source_relation_original_expression"] == (
        "核心服务主要包括"
    )
    assert augmented["reference_nominal_prefix_repairs"] >= 1


def test_parse_grounded_open_facts_rejects_untraceable_reference_only():
    content = "Alice waited. Then she opened the book."
    alice_start = content.index("Alice")
    book_start = content.index("book")
    occurrences = [
        _occurrence(
            "occ-alice", "Alice", alice_start, alice_start + 5, "c-alice",
        ),
        _occurrence(
            "occ-book", "book", book_start, book_start + 4, "c-book",
        ),
    ]
    parsed = parse_grounded_open_facts(
        {
            "resolved_references": [{
                "reference_id": "R0001",
                "surface": "she",
                "start": 0,
                "end": 3,
                "antecedent_id": "M9999",
            }],
            "facts": [{
                "subject_id": "R0001",
                "predicate": "opened",
                "object_id": "M0001",
                "evidence": "Then she opened the book.",
            }],
        },
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert parsed["frames"] == []
    assert parsed["dropped_references"] == [{
        "reference_index": 0,
        "reason": "unknown_reference_antecedent",
    }]
    assert parsed["dropped_items"] == [{
        "item_index": 0,
        "reason": "unknown_endpoint_id",
    }]


def test_parse_grounded_open_facts_rejects_nonreflexive_reference_self_loop():
    content = "Alice waited. Rabbit ran close by her."
    alice = content.index("Alice")
    rabbit = content.index("Rabbit")
    her = content.index("her")
    occurrences = [
        _occurrence("alice", "Alice", alice, alice + 5, "alice"),
        _occurrence("rabbit", "Rabbit", rabbit, rabbit + 6, "rabbit"),
    ]

    parsed = parse_grounded_open_facts(
        {
            "resolved_references": [{
                "reference_id": "R0001",
                "antecedent_id": "M0001",
            }],
            "facts": [{
                "subject_id": "M0001",
                "predicate": "ran close by",
                "object_id": "R0001",
                "evidence": "Rabbit ran close by her.",
            }],
        },
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert parsed["frames"] == []
    assert parsed["reference_proxies"] == []
    assert parsed["dropped_items"] == [{
        "item_index": 0,
        "reason": "nonreflexive_reference_self_loop",
    }]


def test_parse_grounded_open_facts_quarantines_only_invalid_partial_item():
    content = "A controller sends a message."
    occurrences = [
        _occurrence("occ-controller", "controller", 2, 12, "c-controller"),
        _occurrence("occ-message", "message", 21, 28, "c-message"),
    ]
    raw = {"facts": [
        {
            "source_id": "M0000",
            "relation_expression": "sends",
            "target_id": "M0001",
            "evidence_text": content,
        },
        {
            "source_id": "M9999",
            "relation_expression": "sends",
            "target_id": "M0001",
            "evidence_text": content,
        },
    ]}

    with pytest.raises(RelationClassificationError):
        parse_grounded_open_facts(
            raw, content=content, occurrences=occurrences,
        )

    partial = parse_grounded_open_facts(
        raw,
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )
    assert len(partial["frames"]) == 1
    assert partial["coverage_complete"] is False
    assert partial["dropped_items"] == [{
        "item_index": 1,
        "reason": "unknown_endpoint_id",
    }]
    assert partial["repair_items"] == [{
        "item_index": 1,
        "reason": "unknown_endpoint_id",
        "attempt": {
            "subject_id": "M9999",
            "predicate": "sends",
            "object_id": "M0001",
            "evidence": content,
        },
    }]


def test_parse_grounded_open_facts_rejects_closed_class_predicates_only():
    content = "The client connects to the server."
    client_start = content.index("client")
    server_start = content.index("server")
    occurrences = [
        _occurrence(
            "client", "client", client_start,
            client_start + len("client"), "client",
        ),
        _occurrence(
            "server", "server", server_start,
            server_start + len("server"), "server",
        ),
    ]

    parsed = parse_grounded_open_facts(
        {
            "resolved_references": [],
            "facts": [
                {
                    "subject_id": "M0000", "predicate": "connects to",
                    "object_id": "M0001", "evidence": content,
                },
                {
                    "subject_id": "M0000", "predicate": "All of",
                    "object_id": "M0001", "evidence": content,
                },
                {
                    "subject_id": "M0000", "predicate": "the client may",
                    "object_id": "M0001", "evidence": content,
                },
            ],
        },
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert [item["relation"] for item in parsed["frames"]] == [
        "connects to",
    ]
    assert {item["reason"] for item in parsed["dropped_items"]} == {
        "predicate_missing_semantic_head",
        "predicate_incomplete_modal_fragment",
    }


def test_parse_grounded_open_facts_rejects_cjk_connector_fragments():
    content = "客户端连接服务器。"
    server_start = content.index("服务器")
    occurrences = [
        _occurrence("client", "客户端", 0, 3, "client"),
        _occurrence(
            "server", "服务器", server_start,
            server_start + len("服务器"), "server",
        ),
    ]

    parsed = parse_grounded_open_facts(
        {
            "resolved_references": [],
            "facts": [
                {
                    "subject_id": "M0000", "predicate": "连接",
                    "object_id": "M0001", "evidence": content,
                },
                {
                    "subject_id": "M0000", "predicate": "的各种",
                    "object_id": "M0001", "evidence": content,
                },
                {
                    "subject_id": "M0000", "predicate": "以及",
                    "object_id": "M0001", "evidence": content,
                },
                {
                    "subject_id": "M0000", "predicate": "该调用可",
                    "object_id": "M0001", "evidence": content,
                },
            ],
        },
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert [item["relation"] for item in parsed["frames"]] == ["连接"]
    assert len(parsed["dropped_items"]) == 3


def test_grounded_open_fact_recovery_exposes_rejected_clause_endpoint():
    content = "A controller sends a message to a receiver."
    occurrences = [
        _occurrence("occ-controller", "controller", 2, 12, "c-controller"),
        _occurrence("occ-message", "message", 21, 28, "c-message"),
        _occurrence("occ-receiver", "receiver", 34, 42, "c-receiver"),
    ]

    payload = json.loads(grounded_open_fact_prompt(
        context_id="ctx-1",
        content=content,
        occurrences=occurrences,
        failed_items=[{
            "reason": "unknown_endpoint_id",
            "attempt": {
                "subject_id": "M0000",
                "predicate": "performed action",
                "object_id": "sends a message to a receiver",
                "evidence": content,
            },
        }],
    ))

    assert payload["invalid_attempts"] == [{
        "reason": "unknown_endpoint_id",
        "subject_id": "M0000",
        "predicate": "performed action",
        "object_id": "sends a message to a receiver",
        "evidence": content,
    }]
    assert payload["coverage_audit"][
        "mention_ids_without_accepted_fact"
    ] == ["M0000", "M0001", "M0002"]
    assert payload["contract_version"] == (
        GROUNDED_RELATION_OPEN_FACT_RECOVERY_CONTRACT_VERSION
    )


def test_parse_grounded_open_facts_accepts_bounded_recovery_array_alias():
    content = "A controller sends a message to a receiver."
    occurrences = [
        _occurrence("occ-controller", "controller", 2, 12, "c-controller"),
        _occurrence("occ-message", "message", 21, 28, "c-message"),
        _occurrence("occ-receiver", "receiver", 34, 42, "c-receiver"),
    ]

    parsed = parse_grounded_open_facts(
        {"omitted_facts": [{
            "subject_id": "M0001",
            "predicate": "sent to",
            "object_id": "M0002",
            "evidence": content,
        }]},
        content=content,
        occurrences=occurrences,
    )

    assert parsed["coverage_complete"] is True
    assert parsed["frames"][0]["argument_occurrence_keys"] == [
        "occ-message", "occ-receiver",
    ]


def test_parse_grounded_open_facts_realigns_unique_repeated_occurrences():
    content = "Alpha links Beta. Alpha later links Beta."
    second_alpha = content.rindex("Alpha")
    second_beta = content.rindex("Beta")
    occurrences = [
        _occurrence("alpha-first", "Alpha", 0, 5, "alpha"),
        _occurrence("beta-first", "Beta", 12, 16, "beta"),
        _occurrence(
            "alpha-second", "Alpha", second_alpha, second_alpha + 5, "alpha",
        ),
        _occurrence(
            "beta-second", "Beta", second_beta, second_beta + 4, "beta",
        ),
    ]

    partial = parse_grounded_open_facts(
        {"facts": [{
            # Select the first occurrences while quoting only the second
            # clause. Surface matching alone must not validate provenance.
            "subject_id": "M0000",
            "predicate": "links",
            "object_id": "M0001",
            "evidence": "Alpha later links Beta.",
        }]},
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert partial["dropped_items"] == []
    assert partial["repaired_endpoint_occurrences"] == 2
    assert partial["frames"][0]["argument_occurrence_keys"] == [
        "alpha-second", "beta-second",
    ]
    assert partial["frames"][0]["evidence_text"] == (
        "Alpha later links Beta."
    )


def test_parse_grounded_open_facts_does_not_guess_ambiguous_repeat():
    content = "Alpha waits. Alpha links Beta and Alpha."
    second_alpha = content.index("Alpha", 1)
    last_alpha = content.rindex("Alpha")
    beta = content.index("Beta")
    occurrences = [
        _occurrence("alpha-first", "Alpha", 0, 5, "alpha"),
        _occurrence(
            "alpha-second", "Alpha", second_alpha, second_alpha + 5, "alpha",
        ),
        _occurrence("beta", "Beta", beta, beta + 4, "beta"),
        _occurrence(
            "alpha-third", "Alpha", last_alpha, last_alpha + 5, "alpha",
        ),
    ]

    partial = parse_grounded_open_facts(
        {"facts": [{
            "subject_id": "M0000",
            "predicate": "links",
            "object_id": "M0002",
            "evidence": "Beta",
        }]},
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert partial["frames"] == []
    assert partial["dropped_items"] == [{
        "item_index": 0,
        "reason": "evidence_excludes_endpoint_occurrence",
    }]


def test_grounded_open_fact_repair_retains_exact_known_endpoint_pair():
    content = "Alpha links Beta."
    occurrences = [
        _occurrence("alpha", "Alpha", 0, 5, "alpha"),
        _occurrence("beta", "Beta", 12, 16, "beta"),
    ]

    partial = parse_grounded_open_facts(
        {"facts": [{
            "subject_id": "M0000",
            "predicate": "links",
            "object_id": "M0001",
            "evidence": "Alpha connects Beta",
        }]},
        content=content,
        occurrences=occurrences,
        allow_partial=True,
    )

    assert partial["frames"] == []
    assert partial["repair_items"][0]["reason"] == (
        "evidence_not_exact_source"
    )
    assert partial["repair_items"][0][
        "known_argument_occurrence_keys"
    ] == ["alpha", "beta"]


def test_structural_open_facts_bind_table_headers_without_peer_links():
    content = (
        "Item | Source | Purpose\n"
        "Alpha | Atlas | mapping\n"
        "Beta | Survey | analysis\n"
    )

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("alpha", "Alpha"),
        occurrence("atlas", "Atlas"),
        occurrence("mapping", "mapping"),
        occurrence("beta", "Beta"),
        occurrence("survey", "Survey"),
        occurrence("analysis", "analysis"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=occurrences,
    )

    pairs = {
        tuple(item["argument_occurrence_keys"]): item["relation"]
        for item in result["frames"]
    }
    assert pairs == {
        ("alpha", "atlas"): "column role: Source",
        ("alpha", "mapping"): "column role: Purpose",
        ("beta", "survey"): "column role: Source",
        ("beta", "analysis"): "column role: Purpose",
    }
    assert result["deterministic_frames_by_origin"] == {
        "table_header_assignment": 4,
    }


def test_structural_open_facts_treat_repeated_key_row_as_multilevel_header():
    content = (
        "| Methods | Dice/% (↑) | Dice/% (↑) | HD95(↓) | HD95(↓) |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| Methods | C1 | C2 | C1 | C2 |\n"
        "| Baseline | Atlas | Mapping | Survey | Analysis |\n"
    )

    def occurrence(key: str, surface: str, *, start_at: int = 0):
        start = content.index(surface, start_at)
        return _occurrence(key, surface, start, start + len(surface), key)

    second_header_start = content.index("| Methods", 1)
    occurrences = [
        occurrence("header-methods", "Methods", start_at=second_header_start),
        occurrence("header-c1-a", "C1", start_at=second_header_start),
        occurrence(
            "header-c2-a", "C2",
            start_at=content.index("C1", second_header_start) + 2,
        ),
        occurrence(
            "header-c1-b", "C1",
            start_at=content.index("C2", second_header_start) + 2,
        ),
        occurrence(
            "header-c2-b", "C2",
            start_at=content.index("C1", content.index("C2", second_header_start) + 2) + 2,
        ),
        occurrence("baseline", "Baseline"),
        occurrence("atlas", "Atlas"),
        occurrence("mapping", "Mapping"),
        occurrence("survey", "Survey"),
        occurrence("analysis", "Analysis"),
    ]

    result = augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=occurrences,
    )
    frames = {
        tuple(item["argument_occurrence_keys"]): item["relation"]
        for item in result["frames"]
    }

    assert frames == {
        ("baseline", "atlas"): "column role: Dice/% (↑) / C1",
        ("baseline", "mapping"): "column role: Dice/% (↑) / C2",
        ("baseline", "survey"): "column role: HD95(↓) / C1",
        ("baseline", "analysis"): "column role: HD95(↓) / C2",
    }
    assert {
        "header-methods", "header-c1-a", "header-c2-a",
        "header-c1-b", "header-c2-b",
    } <= set(result["relation_endpoint_excluded_occurrence_keys"])


def test_structural_open_facts_drop_model_relations_between_header_cells():
    content = "Item | Source | Purpose\nAlpha | Atlas | mapping\n"

    def occurrence(key, surface, *, offset=0):
        start = content.index(surface, offset)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("header-item", "Item"),
        occurrence("header-source", "Source"),
        occurrence("header-purpose", "Purpose"),
        occurrence("alpha", "Alpha"),
        occurrence("atlas", "Atlas"),
        occurrence("mapping", "mapping"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "has source",
            "argument_occurrence_keys": ["header-purpose", "header-item"],
            "evidence_span": [0, content.index("\n")],
        }]},
        content=content,
        occurrences=occurrences,
    )

    assert result["table_header_frames_dropped"] == 1
    assert result["relation_endpoint_excluded_occurrence_keys"] == [
        "header-item", "header-purpose", "header-source",
    ]
    assert not any(
        {"header-purpose", "header-item"}.intersection(
            frame.get("argument_occurrence_keys") or []
        )
        for frame in result["frames"]
    )
    assert {
        tuple(frame["argument_occurrence_keys"])
        for frame in result["frames"]
    } == {("alpha", "atlas"), ("alpha", "mapping")}


def test_structural_open_facts_recover_identifier_and_explicit_role_connector():
    content = "R-02 The sender submits a message to the service."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("label", "R-02"),
        occurrence("sender", "sender"),
        occurrence("message", "message"),
        occurrence("service", "service"),
    ]
    inventory = {"frames": [
        {
            "relation": "submits",
            "argument_occurrence_keys": ["sender", "message"],
            "evidence_span": [0, len(content)],
        },
        {
            "relation": "submits to",
            "argument_occurrence_keys": ["sender", "service"],
            "evidence_span": [0, len(content)],
        },
    ]}
    result = augment_grounded_open_fact_inventory(
        inventory, content=content, occurrences=occurrences,
    )
    generated = {
        tuple(item["argument_occurrence_keys"]): item["relation"]
        for item in result["frames"]
        if item.get("deterministic_source_origin")
    }

    assert generated[("label", "sender")] == "label binds topic"
    assert generated[("message", "service")] == "submits to"


def test_structural_open_facts_do_not_treat_grammatical_batch_as_line_label():
    content = "批次A-17由质量实验室检验，结果为合格。"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("batch", "批次A-17"),
            occurrence("laboratory", "质量实验室"),
            occurrence("qualified", "合格"),
        ],
    )

    assert not any(
        item.get("deterministic_source_structure") == "line_identifier_binding"
        for item in result["frames"]
    )


def test_structural_open_facts_expand_explicit_coordinated_arguments():
    content = "The report contains Alpha, Beta, and Gamma."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("report", "report"),
        occurrence("alpha", "Alpha"),
        occurrence("beta", "Beta"),
        occurrence("gamma", "Gamma"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "contains",
            "argument_occurrence_keys": ["report", "alpha"],
            "evidence_span": [0, content.index("Alpha") + len("Alpha")],
        }]},
        content=content,
        occurrences=occurrences,
    )

    coordinated = {
        tuple(item["argument_occurrence_keys"])
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "coordinated_argument"
    }
    assert coordinated == {
        ("report", "beta"),
        ("report", "gamma"),
    }


def test_structural_open_facts_use_direct_list_parent_and_never_siblings():
    content = "— Parent\n  — Child\n— Sibling"

    def scoped_occurrence(
        key, surface, *, scope_id, depth, parent_scope_id,
    ):
        start = content.index(surface)
        line_start = content.rfind("\n", 0, start) + 1
        line_end = content.find("\n", start)
        if line_end < 0:
            line_end = len(content)
        return {
            **_occurrence(
                key, surface, start, start + len(surface), key,
            ),
            "scope_id": scope_id,
            "scope_start": line_start,
            "scope_end": line_end,
            "list_group_id": "group-1",
            "list_depth": depth,
            "list_parent_scope_id": parent_scope_id,
        }

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            scoped_occurrence(
                "parent", "Parent", scope_id="scope-parent",
                depth=0, parent_scope_id="",
            ),
            scoped_occurrence(
                "child", "Child", scope_id="scope-child",
                depth=1, parent_scope_id="scope-parent",
            ),
            scoped_occurrence(
                "sibling", "Sibling", scope_id="scope-sibling",
                depth=0, parent_scope_id="",
            ),
        ],
    )
    structural_pairs = {
        tuple(item["argument_occurrence_keys"])
        for item in result["frames"]
        if item.get("deterministic_source_structure")
        == "list_parent_child"
    }

    assert structural_pairs == {("child", "parent")}


def test_flattened_figure_lines_do_not_become_structural_hierarchy():
    content = "Page8\nCoreModule\nTaskR1\nTaskR2\nTaskR3"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("page", "Page8"),
            occurrence("core", "CoreModule"),
            occurrence("r1", "TaskR1"),
            occurrence("r2", "TaskR2"),
            occurrence("r3", "TaskR3"),
        ],
    )

    assert not any(
        item.get("deterministic_source_structure")
        in {"list_parent_child", "section_label_parent_child"}
        for item in result["frames"]
    )


def test_coordinated_recovery_covers_every_member_across_pdf_soft_wrap():
    content = (
        "Design processes\n"
        "These processes are capture, conceptual\n"
        "design, implementation and transition."
    )

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("processes", "Design processes"),
        occurrence("capture", "capture"),
        occurrence("conceptual", "conceptual\ndesign"),
        occurrence("implementation", "implementation"),
        occurrence("transition", "transition"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "include",
            "argument_occurrence_keys": ["processes", "capture"],
            "evidence_span": [0, content.index("conceptual")],
        }]},
        content=content,
        occurrences=occurrences,
    )

    recovered = {
        tuple(item["argument_occurrence_keys"])
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "coordinated_argument"
    }
    assert recovered == {
        ("processes", "conceptual"),
        ("processes", "implementation"),
        ("processes", "transition"),
    }


def test_coordinated_recovery_covers_cjk_soft_wrapped_member():
    content = (
        "核心流程\n"
        "主要包括任务管理、任务间通信、中断/异常\n"
        "管理、时钟管理和多核支持。"
    )

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("core", "核心流程"),
        occurrence("task", "任务管理"),
        occurrence("communication", "任务间通信"),
        occurrence("interrupt", "中断/异常\n管理"),
        occurrence("clock", "时钟管理"),
        occurrence("multicore", "多核支持"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "包括",
            "argument_occurrence_keys": ["core", "task"],
            "evidence_span": [0, content.index("任务间通信")],
        }]},
        content=content,
        occurrences=occurrences,
    )

    recovered = {
        tuple(item["argument_occurrence_keys"])
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "coordinated_argument"
    }
    assert recovered == {
        ("core", "communication"),
        ("core", "interrupt"),
        ("core", "clock"),
        ("core", "multicore"),
    }


def test_coordination_stops_at_ordinary_comma_after_explicit_conjunction():
    content = (
        "初始化本地模型和全局模型和协作图，客户端基于本地数据计算不确定性。"
    )

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("local", "本地模型"),
        occurrence("global", "全局模型"),
        occurrence("graph", "协作图"),
        occurrence("client", "客户端"),
        occurrence("uncertainty", "不确定性"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "计算",
            "argument_occurrence_keys": ["client", "uncertainty"],
            "evidence_span": [content.index("客户端"), len(content)],
        }]},
        content=content,
        occurrences=occurrences,
    )

    calculated_subjects = {
        item["argument_occurrence_keys"][0]
        for item in result["frames"]
        if item.get("relation") == "计算"
        and item.get("argument_occurrence_keys", [None, None])[1]
        == "uncertainty"
    }
    assert calculated_subjects == {"client"}


def test_structural_open_facts_recover_three_role_construction_from_reverse_seed():
    content = "The map links a service to a requirement."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("map", "map"),
        occurrence("service", "service"),
        occurrence("requirement", "requirement"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "links",
            # A model may discover the pair but reverse its open orientation.
            "argument_occurrence_keys": ["service", "map"],
            "evidence_span": [0, content.index("service") + len("service")],
        }]},
        content=content,
        occurrences=occurrences,
    )

    generated = {
        tuple(item["argument_occurrence_keys"]): item["relation"]
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "ordered_three_role_construction"
    }
    assert generated == {
        ("map", "service"): "links",
        ("service", "requirement"): "links to",
        ("map", "requirement"): "links to",
    }

    seedless = augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=occurrences,
    )
    assert {
        tuple(item["argument_occurrence_keys"])
        for item in seedless["frames"]
        if item.get("deterministic_source_syntax")
        == "ordered_three_role_construction"
    } == {
        ("map", "service"),
        ("service", "requirement"),
        ("map", "requirement"),
    }


def test_intervening_predicate_recovery_does_not_cross_list_delimiter():
    content = "数据量较大、临床价值更高的客户端。"

    def occurrence(key: str, surface: str):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("amount", "数据量"),
            occurrence("value", "临床价值"),
            occurrence("client", "客户端"),
        ],
    )

    assert not any(
        set(frame.get("argument_occurrence_keys") or ())
        == {"amount", "value"}
        for frame in result["frames"]
    )


def test_three_role_projection_does_not_promote_prepositional_object_to_theme():
    content = "Alice was sitting by her sister on the bank."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("alice", "Alice"),
        occurrence("sister", "sister"),
        occurrence("bank", "bank"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "was sitting by",
            "argument_occurrence_keys": ["alice", "sister"],
            "evidence_span": [0, content.index("sister") + len("sister")],
        }]},
        content=content,
        occurrences=occurrences,
    )
    generated = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "ordered_three_role_construction"
    ]
    assert {
        tuple(item["argument_occurrence_keys"])
        for item in generated
    } == {("alice", "sister")}
    assert generated[0]["roles"] == [
        "grammatical_subject", "role_complement",
    ]
    assert all(
        "sister" not in item["argument_occurrence_keys"]
        or "bank" not in item["argument_occurrence_keys"]
        for item in result["frames"]
    )


def test_three_role_projection_expands_every_coordinated_complement():
    content = "The catalog links a service to alpha, beta, and gamma."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("catalog", "catalog"),
        occurrence("service", "service"),
        occurrence("alpha", "alpha"),
        occurrence("beta", "beta"),
        occurrence("gamma", "gamma"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=occurrences,
    )
    generated = {
        tuple(item["argument_occurrence_keys"])
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "ordered_three_role_construction"
    }
    assert {
        ("service", "alpha"), ("service", "beta"),
        ("service", "gamma"), ("catalog", "alpha"),
        ("catalog", "beta"), ("catalog", "gamma"),
    } <= generated


def test_source_syntax_recovers_theme_source_and_direct_object():
    content = "The Rabbit took a watch out of its pocket."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("rabbit", "Rabbit"),
        occurrence("watch", "watch"),
        occurrence("pocket", "pocket"),
    ]
    its_start = content.index("its")
    possessive_proxy = {
        **_occurrence(
            "reference:its", "its", its_start, its_start + 3, "rabbit",
        ),
        "relation_reference_proxy": True,
        "reference_role": "possessive_determiner",
        "reference_antecedent_occurrence_key": "rabbit",
        "reference_other_endpoint_occurrence_key": "pocket",
    }
    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[*occurrences, possessive_proxy],
    )
    generated = {
        tuple(item["argument_occurrence_keys"]): item["relation"]
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
    }

    assert generated[("rabbit", "watch")] == "took"
    assert generated[("watch", "pocket")] == "took out of"
    assert generated[("rabbit", "pocket")] == "took out of"
    assert ("watch", "reference:its") not in generated
    assert ("reference:its", "pocket") in generated


def test_source_syntax_keeps_modifier_from_becoming_predicate_subject():
    content = "A device with red indicators moved close by the operator."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("device", "device"),
            occurrence("indicators", "indicators"),
            occurrence("operator", "operator"),
        ],
    )
    modifier_frames = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "modifier_interrupted_predicate"
    ]
    assert len(modifier_frames) == 1
    assert modifier_frames[0]["argument_occurrence_keys"] == [
        "device", "operator",
    ]
    assert modifier_frames[0]["relation"] == "moved close by"
    assert not any(
        item["argument_occurrence_keys"] == ["indicators", "operator"]
        for item in result["frames"]
    )


def test_source_syntax_recovers_local_binary_nested_predicate():
    content = "She saw the Rabbit."
    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            _occurrence("she", "She", 0, 3, "alice"),
            _occurrence("rabbit", "Rabbit", 12, 18, "rabbit"),
        ],
    )
    generated = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "intervening_predicate_construction"
    ]
    assert len(generated) == 1
    assert generated[0]["argument_occurrence_keys"] == ["she", "rabbit"]
    assert generated[0]["relation"] == "saw"


def test_source_syntax_removes_renderer_transport_from_visible_edge_label():
    content = "Service-A --calls--> Service-B"
    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            _occurrence("a", "Service-A", 0, 9, "a"),
            _occurrence("b", "Service-B", 21, 30, "b"),
        ],
    )
    generated = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "intervening_predicate_construction"
    ]

    assert len(generated) == 1
    assert generated[0]["relation"] == "calls"


def test_source_syntax_keeps_object_modifier_out_of_binary_predicate():
    content = "The service shall validate a submitted operation intent."
    service_start = content.index("service")
    intent_start = content.index("operation intent")
    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            _occurrence(
                "service", "service", service_start,
                service_start + len("service"), "service",
            ),
            _occurrence(
                "intent", "operation intent", intent_start,
                intent_start + len("operation intent"), "intent",
            ),
        ],
    )
    generated = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "intervening_predicate_construction"
    ]
    assert len(generated) == 1
    assert generated[0]["relation"] == "validate"

    direct_content = "The service submitted an operation intent."
    direct_intent_start = direct_content.index("operation intent")
    direct = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=direct_content,
        occurrences=[
            _occurrence("service", "service", 4, 11, "service"),
            _occurrence(
                "intent", "operation intent", direct_intent_start,
                direct_intent_start + len("operation intent"), "intent",
            ),
        ],
    )
    assert next(
        item for item in direct["frames"]
        if item.get("deterministic_source_syntax")
        == "intervening_predicate_construction"
    )["relation"] == "submitted"


def test_source_syntax_does_not_promote_prior_clause_complement_to_subject():
    content = "华宁材料厂位于江苏，生产电池隔膜和复合涂层。"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("factory", "华宁材料厂"),
            occurrence("jiangsu", "江苏"),
            occurrence("separator", "电池隔膜"),
            occurrence("coating", "复合涂层"),
        ],
    )
    generated_pairs = {
        tuple(item.get("argument_occurrence_keys") or ())
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
    }

    assert ("factory", "jiangsu") in generated_pairs
    assert ("jiangsu", "separator") not in generated_pairs
    assert ("separator", "coating") not in generated_pairs


def test_source_syntax_keeps_accompaniment_from_becoming_motion_subject():
    content = "清晨，林岚带着测温仪来到北站。"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("lin", "林岚"),
            occurrence("instrument", "测温仪"),
            occurrence("station", "北站"),
        ],
    )
    generated = {
        tuple(item.get("argument_occurrence_keys") or ()): item
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
    }

    assert generated[("lin", "station")]["relation"] == "来到"
    assert generated[("lin", "station")][
        "deterministic_source_syntax"
    ] == "modifier_interrupted_predicate"
    assert ("instrument", "station") not in generated


def test_source_syntax_rejects_connector_only_cross_clause_relation():
    content = "受理机构审查申请材料，并将审查结果通知申请人。"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("authority", "受理机构"),
            occurrence("materials", "申请材料"),
            occurrence("result", "审查结果"),
            occurrence("applicant", "申请人"),
        ],
    )

    assert not any(
        item.get("argument_occurrence_keys") == ["materials", "result"]
        for item in result["frames"]
    )


def test_source_syntax_recovers_postpredicate_passive_agent_not_particle():
    content = "虚中断管理由核心操作系统提供。"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("capability", "虚中断管理"),
            occurrence("provider", "核心操作系统"),
        ],
    )
    generated = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "passive_agent_postpredicate_construction"
    ]

    assert len(generated) == 1
    assert generated[0]["argument_occurrence_keys"] == [
        "provider", "capability",
    ]
    assert generated[0]["roles"] == [
        "grammatical_subject", "action_theme",
    ]
    assert generated[0]["relation"] == "提供"
    assert not any(
        item.get("argument_occurrence_keys") == [
            "capability", "provider",
        ]
        and item.get("relation") == "由"
        for item in result["frames"]
    )


def test_source_syntax_recovers_preposed_recipient_and_coordinated_themes():
    content = "申请人应当向受理机构提交申请材料和安全报告。"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("applicant", "申请人"),
            occurrence("authority", "受理机构"),
            occurrence("materials", "申请材料"),
            occurrence("report", "安全报告"),
        ],
    )
    generated = {
        tuple(item["argument_occurrence_keys"]): item
        for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "ordered_three_role_construction"
    }

    assert generated[("applicant", "materials")]["relation"] == "提交"
    assert generated[("applicant", "report")]["relation"] == "提交"
    assert generated[("materials", "authority")]["roles"] == [
        "action_theme", "role_complement",
    ]
    assert generated[("report", "authority")]["roles"] == [
        "action_theme", "role_complement",
    ]
    assert generated[("applicant", "authority")]["roles"] == [
        "grammatical_subject", "role_complement",
    ]


def test_source_syntax_recovers_disposal_theme_without_actor_occurrence():
    content = "她把仪器交给值班员周启。"

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            occurrence("instrument", "仪器"),
            occurrence("recipient", "值班员周启"),
        ],
    )
    matching = [
        item for item in result["frames"]
        if item.get("argument_occurrence_keys") == [
            "instrument", "recipient",
        ]
    ]

    assert len(matching) == 1
    assert matching[0]["relation"] == "交给"
    assert matching[0]["roles"] == [
        "action_theme", "role_complement",
    ]
    assert matching[0]["deterministic_source_syntax"] == (
        "ordered_three_role_construction"
    )


def test_source_syntax_preserves_grammatical_negation_in_binary_predicate():
    content = "A neighboring cell is not a semantic relation."
    cell_start = content.index("neighboring cell")
    relation_start = content.index("semantic relation")
    result = augment_grounded_open_fact_inventory(
        {"frames": []},
        content=content,
        occurrences=[
            _occurrence(
                "cell", "neighboring cell", cell_start,
                cell_start + len("neighboring cell"), "cell",
            ),
            _occurrence(
                "relation", "semantic relation", relation_start,
                relation_start + len("semantic relation"), "relation",
            ),
        ],
    )
    generated = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "intervening_predicate_construction"
    ]
    assert len(generated) == 1
    assert generated[0]["relation"] == "is not a"
    # The source asserts a negative proposition. Publication polarity is
    # enforced separately by each logical predicate's negation policy.
    assert generated[0]["assertion_status"] == "ASSERTED"


def test_source_syntax_annotates_flattened_model_frame_with_role_topology():
    content = "The catalog contains records for engines and pumps."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    catalog = occurrence("catalog", "catalog")
    records = occurrence("records", "records")
    engines = occurrence("engines", "engines")
    pumps = occurrence("pumps", "pumps")
    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "contains",
            "arguments": ["catalog", "engines"],
            "roles": ["subject", "object"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["catalog", "engines"],
            "assertion_status": "ASSERTED",
            "argument_occurrence_keys": ["catalog", "engines"],
            "evidence_span": [0, len(content)],
            "evidence_text": content,
        }]},
        content=content,
        occurrences=[catalog, records, engines, pumps],
    )
    assert result["parser_role_frames_annotated"] == 2
    assert result["frames"][0]["parser_inferred_source_roles"] == [
        "grammatical_subject", "role_complement",
    ]
    assert result["frames"][0]["parser_inferred_source_syntax"] == (
        "ordered_three_role_construction"
    )
    coordinated = next(
        item for item in result["frames"]
        if item.get("deterministic_source_syntax") == "coordinated_argument"
        and item.get("argument_occurrence_keys") == ["catalog", "pumps"]
    )
    assert coordinated["parser_inferred_source_roles"] == [
        "grammatical_subject", "role_complement",
    ]


def test_structural_open_facts_recover_indented_header_child():
    content = "if ready:\n    start()\n"
    ready = content.index("ready")
    start = content.index("start")
    occurrences = [
        _occurrence("ready", "ready", ready, ready + 5, "ready"),
        _occurrence("start", "start", start, start + 5, "start"),
    ]

    result = augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=occurrences,
    )

    generated = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "indented_header_child"
    ]
    assert len(generated) == 1
    assert generated[0]["argument_occurrence_keys"] == ["ready", "start"]


def test_structural_open_facts_do_not_connect_coordinated_peer_objects():
    content = "The platform supports tracking and monitoring."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("platform", "platform"),
        occurrence("tracking", "tracking"),
        occurrence("monitoring", "monitoring"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": [
            {
                "relation": "supports",
                "argument_occurrence_keys": ["platform", "tracking"],
                "evidence_span": [0, len(content)],
            },
            {
                "relation": "supports",
                "argument_occurrence_keys": ["platform", "monitoring"],
                "evidence_span": [0, len(content)],
            },
        ]},
        content=content,
        occurrences=occurrences,
    )

    assert not any(
        set(item["argument_occurrence_keys"]) == {"tracking", "monitoring"}
        for item in result["frames"]
    )


def test_source_syntax_marks_collective_subject_pair_as_candidate_only():
    content = "The recipient and the agency agree on the terms."

    def occurrence(key, surface):
        start = content.index(surface)
        return _occurrence(key, surface, start, start + len(surface), key)

    result = augment_grounded_open_fact_inventory(
        {"frames": [{
            "relation": "agree on",
            "argument_occurrence_keys": ["recipient", "terms"],
            "evidence_span": [0, len(content)],
        }]},
        content=content,
        occurrences=[
            occurrence("recipient", "recipient"),
            occurrence("agency", "agency"),
            occurrence("terms", "terms"),
        ],
    )
    collective = [
        item for item in result["frames"]
        if item.get("deterministic_source_syntax")
        == "coordinated_collective_relation_candidate"
    ]
    assert len(collective) == 1
    assert collective[0]["argument_occurrence_keys"] == [
        "recipient", "agency",
    ]
    assert collective[0]["roles"] == [
        "collective_participant", "collective_participant",
    ]


def test_structural_open_facts_recover_section_identifier_children():
    content = "Requirements\n\nR-01 First rule.\nR-02 Second rule.\n\nProse."

    def occurrence(key, surface, *, start_at=0):
        start = content.index(surface, start_at)
        return _occurrence(key, surface, start, start + len(surface), key)

    occurrences = [
        occurrence("requirements", "Requirements"),
        occurrence("r1", "R-01"),
        occurrence("r2", "R-02"),
    ]
    result = augment_grounded_open_fact_inventory(
        {"frames": []}, content=content, occurrences=occurrences,
    )

    hierarchy = {
        tuple(item["argument_occurrence_keys"])
        for item in result["frames"]
        if item.get("deterministic_source_structure")
        == "section_label_parent_child"
    }
    assert hierarchy == {
        ("r1", "requirements"),
        ("r2", "requirements"),
    }


def test_source_structure_binding_requires_exact_kind_and_table_role_label():
    requirement_policy = predicate_policy(
        "maps_requirement",
        structural_evidence_kinds=["table_header_assignment"],
    )
    requirement_policy["structural_role_labels"] = ["Requirement"]
    source_policy = predicate_policy(
        "has_source",
        structural_evidence_kinds=["table_header_assignment"],
    )
    source_policy["structural_role_labels"] = ["Source"]
    topic_policy = predicate_policy(
        "scopes_topic",
        structural_evidence_kinds=["line_identifier_binding"],
    )
    schema = logical_schema(
        requirement_policy, source_policy, topic_policy,
        version="source-structure-schema-v1",
    )
    inventory = {"frames": [
        {
            "arguments": ["Service A", "R-01"],
            "roles": ["row_key", "column_value"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Service A", "R-01"],
            "assertion_status": "ASSERTED",
            "deterministic_source_structure": "table_header_assignment",
            "structural_role_label": "Requirement",
        },
        {
            "arguments": ["R-01", "Service A"],
            "roles": ["source_label", "bound_topic"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["R-01", "Service A"],
            "assertion_status": "ASSERTED",
            "deterministic_source_structure": "line_identifier_binding",
        },
        {
            "arguments": ["Service A", "unknown"],
            "roles": ["row_key", "column_value"],
            "argument_kinds": ["ENTITY", "ENTITY"],
            "entity_heads": ["Service A", "unknown"],
            "assertion_status": "ASSERTED",
            "deterministic_source_structure": "table_header_assignment",
            "structural_role_label": "Unregistered column",
        },
    ]}

    result = deterministic_source_structure_schema_bindings(
        inventory, logical_schema=schema,
    )

    assert [
        (item["frame_index"], item["predicate"])
        for item in result["bindings"]
    ] == [(0, "maps_requirement"), (1, "scopes_topic")]
    assert all(
        item["subject_index"] == 0 and item["object_index"] == 1
        for item in result["bindings"]
    )
