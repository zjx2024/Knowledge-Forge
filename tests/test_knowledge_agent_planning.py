"""Generic query planning remains bounded, auditable and fail-safe."""

from types import SimpleNamespace

from knowledge_forge.agent.planning import (
    RETRIEVAL_FAMILIES,
    RETRIEVAL_PLAN_PROTOCOL,
    apply_retrieval_plan,
    diagnose_primary_payload,
    plan_retrieval,
    plan_supplemental_retrieval,
    supplemental_keyword_query,
)


def test_planner_allocates_relation_questions_without_disabling_any_family():
    plan = plan_retrieval(
        "对象 A 与对象 B 为什么存在这种联系？",
        strategy="knowledge_forge",
        candidate_limit=40,
    )

    assert plan.protocol == RETRIEVAL_PLAN_PROTOCOL
    assert plan.activation == "observe"
    assert plan.intent == "relation_explanation"
    assert plan.scope == "local"
    assert sum(plan.family_limits.values()) == 40
    assert set(plan.family_limits) == set(RETRIEVAL_FAMILIES)
    assert all(value >= 1 for value in plan.family_limits.values())
    assert plan.family_limits["relation"] > plan.family_limits["entity"]


def test_planner_prefers_community_context_for_global_summary():
    plan = plan_retrieval(
        "请总结这些文档的整体主题与共同趋势。",
        strategy="knowledge_forge",
        candidate_limit=32,
    )

    assert plan.intent == "global_summary"
    assert plan.scope == "global"
    assert plan.native_mode == "global"
    assert plan.family_limits["community"] == max(plan.family_limits.values())


def test_planner_preserves_explicit_native_mode_over_automatic_analysis():
    plan = plan_retrieval(
        "请给出整体概览。",
        strategy="graphrag_native",
        candidate_limit=20,
        requested_mode="basic",
    )

    assert plan.native_mode == "basic"
    assert "explicit_native_mode" in plan.reason_codes


def test_planner_uses_local_for_single_document_native_summary():
    plan = plan_retrieval(
        "请总结全文的主要挑战。",
        strategy="graphrag_native",
        candidate_limit=20,
        knowledge_scope={
            "document_count": 1,
            "document_names": ["research.pdf"],
        },
    )

    assert plan.intent == "global_summary"
    assert plan.native_mode == "local"
    assert "single_document_summary_uses_local" in plan.reason_codes


def test_planner_keeps_global_for_multi_document_native_summary():
    plan = plan_retrieval(
        "请总结这些文档的整体主题。",
        strategy="graphrag_native",
        candidate_limit=20,
        knowledge_scope={
            "document_count": 3,
            "document_names": ["a.pdf", "b.pdf", "c.pdf"],
        },
    )

    assert plan.native_mode == "global"


def test_planner_uses_balanced_safe_floor_for_unclassified_query():
    plan = plan_retrieval(
        "ZX-103 的阈值是多少？",
        strategy="knowledge_forge",
        candidate_limit=10,
        referenced_terms=["ZX-103", "ZX-103"],
    )

    assert plan.intent == "fact_lookup"
    assert plan.candidate_limit == 10
    assert all(value >= 1 for value in plan.family_limits.values())
    assert plan.referenced_terms == ("ZX-103",)


def test_active_plan_backfills_missing_families_without_shrinking_recall():
    plan = plan_retrieval(
        "请总结整体主题。", strategy="knowledge_forge",
        candidate_limit=8, activation="active",
    )
    items = [
        SimpleNamespace(evidence_id=f"chunk-{index}", evidence_type="chunk", metadata={})
        for index in range(8)
    ]

    selected, diagnostics = apply_retrieval_plan(
        items, plan=plan, candidate_limit=8,
    )

    assert selected == items
    assert diagnostics["candidate_count"] == 8
    assert diagnostics["selected_by_family"]["text"] == 8
    assert diagnostics["backfilled"] > 0


def test_active_plan_reserves_relation_slots_before_global_backfill():
    plan = plan_retrieval(
        "A 和 B 有什么关系？", strategy="knowledge_forge",
        candidate_limit=10, activation="active",
    )
    items = [
        *[
            SimpleNamespace(evidence_id=f"chunk-{index}", evidence_type="chunk", metadata={})
            for index in range(8)
        ],
        *[
            SimpleNamespace(evidence_id=f"relation-{index}", evidence_type="relation", metadata={})
            for index in range(8)
        ],
    ]

    selected, diagnostics = apply_retrieval_plan(
        items, plan=plan, candidate_limit=10,
    )

    assert len(selected) == 10
    assert diagnostics["selected_by_family"]["relation"] >= plan.family_limits["relation"]


def test_active_plan_scales_family_reservations_for_small_answer_context():
    plan = plan_retrieval(
        "图 2 展示了什么？", strategy="knowledge_forge",
        candidate_limit=40, activation="active",
    )
    items = [
        *[
            SimpleNamespace(
                evidence_id=f"relation-{index}", evidence_type="relation",
                metadata={},
            )
            for index in range(12)
        ],
        *[
            SimpleNamespace(
                evidence_id=f"chunk-{index}", evidence_type="chunk",
                metadata={},
            )
            for index in range(12)
        ],
        SimpleNamespace(
            evidence_id="entity-1", evidence_type="entity", metadata={},
        ),
    ]

    selected, diagnostics = apply_retrieval_plan(
        items, plan=plan, candidate_limit=6,
    )

    assert len(selected) == 6
    assert sum(diagnostics["reserved_by_family"].values()) == 6
    assert diagnostics["selected_by_family"]["text"] >= 2
    assert diagnostics["selected_by_family"]["relation"] >= 1
    assert diagnostics["selected_by_family"]["entity"] == 1


def test_primary_probe_recommends_only_bounded_expansion_from_existing_rows():
    plan = plan_retrieval(
        "A 与 B 有什么关系？", strategy="knowledge_forge",
        candidate_limit=20, activation="active",
    )
    diagnostics = diagnose_primary_payload({
        "chunks": [{"id": 1}],
        "entities": [{"id": "a"}, {"id": "b"}],
        "relations": [],
        "community_reports": [],
        "graph_retrieval": {"seed_entities": [
            {"canonical_key": "a", "score": 1.0, "query_surface_match": True},
            {"canonical_key": "b", "score": 1.0, "query_surface_match": True},
        ]},
    }, plan=plan, graph_available=True)

    assert diagnostics["expansion_recommended"] is True
    assert diagnostics["model_calls"] == 0
    assert diagnostics["embedding_calls"] == 0
    assert "sparse_primary_relations" in diagnostics["expansion_reasons"]


def test_primary_probe_does_not_expand_a_supported_single_seed_query():
    plan = plan_retrieval(
        "A 的定义是什么？", strategy="knowledge_forge",
        candidate_limit=20, activation="active",
    )
    diagnostics = diagnose_primary_payload({
        "chunks": [], "entities": [],
        "relations": [{"id": index} for index in range(4)],
        "community_reports": [],
        "graph_retrieval": {"seed_entities": [
            {"canonical_key": "a", "score": 1.0, "query_surface_match": True},
        ]},
    }, plan=plan, graph_available=True)

    assert diagnostics["expansion_recommended"] is False


def test_supplemental_plan_is_bounded_and_cannot_retry_itself():
    primary = plan_retrieval(
        "对象 A 为什么影响对象 B？",
        strategy="knowledge_forge",
        candidate_limit=40,
        activation="active",
    )
    decision = plan_supplemental_retrieval(
        "对象 A 为什么影响对象 B？",
        strategy="knowledge_forge",
        primary_plan=primary,
        sufficiency=SimpleNamespace(
            state="retryable", aspect_coverage=0.2,
            graph_connectivity=0.3, independent_support=0.5,
        ),
        maximum_candidates=16,
    )

    assert decision.execute is True
    assert decision.candidate_limit == 16
    assert decision.plan is not None
    assert decision.plan.retry_allowed is False
    assert sum(decision.plan.family_limits.values()) == 16
    assert "supplemental_once" in decision.plan.reason_codes


def test_supplemental_query_only_removes_generic_question_words():
    assert supplemental_keyword_query(
        "请问对象 ZX-103 的校准上限是多少？"
    ) == "对象 ZX-103 的校准上限"


def test_supplemental_plan_does_not_run_after_sufficient_evidence():
    primary = plan_retrieval(
        "对象 A 是什么？", strategy="knowledge_forge",
        candidate_limit=20, activation="active",
    )
    decision = plan_supplemental_retrieval(
        "对象 A 是什么？", strategy="knowledge_forge",
        primary_plan=primary,
        sufficiency=SimpleNamespace(state="sufficient"),
        maximum_candidates=10,
    )

    assert decision.execute is False
    assert decision.plan is None
    apply_retrieval_plan,
