"""Deterministic contracts and policy checks for Knowledge Agent V1."""

from __future__ import annotations

from knowledge_forge.agent.answer import (
    _schema_for_sufficiency,
    generate_answer,
    normalize_answer_payload,
)
from knowledge_forge.agent.citations import validate_and_render
from knowledge_forge.agent.contracts import EvidenceItem, SnapshotRef, SourceRef
from knowledge_forge.agent.fusion import (
    apply_final_ranking,
    deduplicate_evidence,
    reciprocal_rank_fuse,
    relevance_filtered_context_pool,
    select_context,
)
from knowledge_forge.agent.sufficiency import assess_sufficiency


def _snapshot(*, source_hash: str = "source-a") -> SnapshotRef:
    return SnapshotRef(
        strategy="knowledge_forge",
        identifier="kf:4:namespace:relations",
        source_hash=source_hash,
        generation=4,
        graph_available=True,
    )


def _source(document_id: int = 7, *, page: int = 3) -> SourceRef:
    return SourceRef.from_dict({
        "document_id": document_id,
        "filename": f"manual-{document_id}.pdf",
        "revision_id": f"revision-{document_id}",
        "pages": [page],
        "page_label": f"第 {page} 页",
        "section_title": "系统组成",
        "document_url": "/must-not-leak/local/path.pdf",
    }, quote="系统包含任务管理模块。")


def _evidence(
    run_id: str,
    record_key: str,
    *,
    content: str = "系统包含任务管理模块，并由调度器执行周期任务。",
    evidence_type: str = "chunk",
    channel: str = "chunk_dense",
    score: float = 0.8,
    source: SourceRef | None = None,
) -> EvidenceItem:
    return EvidenceItem.create(
        run_id=run_id,
        evidence_type=evidence_type,
        project_id=11,
        strategy="knowledge_forge",
        snapshot_ref=_snapshot(),
        record_key=record_key,
        title="任务管理",
        content=content,
        source_refs=[source or _source()],
        retrieval_channels=[channel],
        raw_scores={channel: score},
        entity_keys=["task-management"],
        relation_keys=[record_key] if evidence_type == "relation" else [],
    )


def test_source_reference_is_browser_safe_and_evidence_identity_is_run_scoped():
    source = _source()
    first = _evidence("run-a", "chunk-7", source=source)
    repeated = _evidence("run-a", "chunk-7", source=source)
    other_run = _evidence("run-b", "chunk-7", source=source)

    assert source.document_url == "/api/documents/7/download"
    assert source.valid is True
    assert first.provenance_valid is True
    assert first.evidence_id == repeated.evidence_id
    assert first.evidence_id != other_run.evidence_id
    assert first.metadata["stable_key"] == other_run.metadata["stable_key"]
    assert first.snapshot_ref.source_hash == "source-a"


def test_source_reference_without_revision_is_not_publishable_provenance():
    source = SourceRef.from_dict({
        "document_id": 7,
        "filename": "manual.pdf",
        "pages": [3],
    })

    assert source.valid is False


def test_rrf_merges_duplicate_channels_then_ranks_and_selects_within_budget():
    dense = _evidence("run-a", "chunk-7", channel="chunk_dense", score=0.9)
    lexical = _evidence("run-a", "chunk-7", channel="chunk_lexical", score=0.7)
    relation = _evidence(
        "run-a", "relation-2", evidence_type="relation",
        channel="relation_graph", score=0.85, source=_source(8),
    )
    merged = deduplicate_evidence([dense, lexical, relation])

    assert len(merged) == 2
    chunk = next(item for item in merged if item.evidence_type == "chunk")
    assert set(chunk.retrieval_channels) == {"chunk_dense", "chunk_lexical"}

    fused = reciprocal_rank_fuse(merged)
    for item in fused:
        item.rerank_score = item.fused_score
    ranked = apply_final_ranking(fused)
    selected = select_context(ranked, token_budget=900, max_items=2)

    assert [item.rank for item in ranked] == [1, 2]
    assert 1 <= len(selected) <= 2
    assert all(item.selected for item in selected)
    assert all(0 <= item.metadata["final_score"] <= 1 for item in ranked)


def test_untraceable_evidence_is_not_provenance_complete():
    item = EvidenceItem.create(
        run_id="run-a", evidence_type="chunk", project_id=11,
        strategy="knowledge_forge", snapshot_ref=_snapshot(),
        record_key="orphan", title="无来源", content="无法定位的文本",
        source_refs=[], retrieval_channels=["chunk_dense"],
        raw_scores={"chunk_dense": 1.0},
    )

    assert item.provenance_valid is False
    assert item.provenance_score == 0


def test_oversized_evidence_is_bounded_for_prompt_without_losing_audit_text():
    full_content = "系统能力与约束。" * 3000
    item = _evidence("run-a", "large-report", content=full_content)
    item.rerank_score = item.fused_score = item.provenance_score = 1.0
    item.metadata["final_score"] = 1.0

    selected = select_context([item], token_budget=700, max_items=4)

    assert selected == [item]
    assert item.content == full_content
    assert len(item.metadata["context_content"]) < len(full_content)
    assert item.metadata["context_content"].endswith("…")
    assert int(item.metadata["context_token_cost"]) <= 700


def test_context_selection_does_not_prefer_short_weak_text_over_long_strong_evidence():
    strong = _evidence(
        "run-a", "strong", content="任务调度器负责周期任务。" * 500,
    )
    weak = _evidence(
        "run-a", "weak-short", content="配置记录。",
        source=_source(8),
    )
    strong.metadata["final_score"] = 0.96
    weak.metadata["final_score"] = 0.48

    selected = select_context([strong, weak], token_budget=1200, max_items=1)

    assert selected == [strong]
    assert int(strong.metadata["context_token_cost"]) <= 600


def test_context_pool_removes_only_clear_cross_encoder_noise():
    strong = _evidence("run-a", "strong")
    secondary = _evidence("run-a", "secondary", source=_source(8))
    noise = _evidence("run-a", "noise", source=_source(9))
    strong.rerank_score = 0.98
    secondary.rerank_score = 0.18
    noise.rerank_score = 0.0004

    filtered, floor = relevance_filtered_context_pool(
        [strong, secondary, noise], reranker_used_model=True,
    )

    assert filtered == [strong, secondary]
    assert 0.009 < floor < 0.010

    degraded, degraded_floor = relevance_filtered_context_pool(
        [strong, secondary, noise], reranker_used_model=False,
    )
    assert degraded == [strong, secondary, noise]
    assert degraded_floor == 0


def test_sufficiency_refuses_empty_results_without_calling_a_model():
    result = assess_sufficiency("系统支持哪些能力？", [])

    class NeverCalled:
        def json_chat(self, **_kwargs):  # pragma: no cover - failure sentinel
            raise AssertionError("insufficient evidence must not call the model")

    payload = generate_answer(
        ollama=NeverCalled(),
        project={"knowledge_language": "zh-CN"},
        query="系统支持哪些能力？",
        selected=[], sufficiency=result,
    )

    assert result.state == "insufficient"
    assert payload["claims"] == []
    assert "没有找到足够" in payload["answer"]


def test_sufficiency_rejects_only_when_all_absolute_relevance_signals_are_weak():
    unrelated = _evidence(
        "run-a", "weak", content="存储模块保存配置记录。", score=0.40,
    )
    unrelated.rerank_score = 0.0001

    rejected = assess_sufficiency(
        "火星基地管理员的私人手机号码是什么？",
        [unrelated],
        reranker_used_model=True,
    )

    assert rejected.state == "insufficient"
    assert rejected.reranker_used_model is True
    assert rejected.semantic_relevance == 0.40
    assert "多路检索" in "；".join(rejected.reasons)


def test_sufficiency_preserves_semantic_paraphrase_when_dense_signal_is_strong():
    paraphrase = _evidence(
        "run-a", "semantic", content="控制器在异常后切换备用通道并恢复服务。",
        score=0.68,
    )
    paraphrase.rerank_score = 0.001

    result = assess_sufficiency(
        "设备发生故障时采用何种恢复方案？",
        [paraphrase],
        reranker_used_model=True,
    )

    assert result.state != "insufficient"
    assert result.semantic_relevance == 0.68
    assert "覆盖不足" not in "；".join(result.reasons)


def test_english_refusal_follows_auto_query_language():
    result = assess_sufficiency("What is the private phone number?", [])

    class NeverCalled:
        def json_chat(self, **_kwargs):  # pragma: no cover - failure sentinel
            raise AssertionError("insufficient evidence must not call the model")

    payload = generate_answer(
        ollama=NeverCalled(), project={"knowledge_language": "auto"},
        query="What is the private phone number?", selected=[], sufficiency=result,
    )

    assert payload["answerability"] == "insufficient"
    assert payload["answer"].startswith("The selected knowledge base")


def test_sufficient_answer_schema_requires_a_citable_claim():
    schema = _schema_for_sufficiency("sufficient")

    assert "insufficient" not in schema["properties"]["answerability"]["enum"]
    assert schema["properties"]["claims"]["minItems"] == 1
    assert (
        schema["properties"]["claims"]["items"]["properties"]
        ["citations"]["minItems"]
    ) == 1
    assert "insufficient" in (
        _schema_for_sufficiency("retryable")["properties"]
        ["answerability"]["enum"]
    )


def test_flat_local_model_answer_is_normalized_into_a_citable_claim():
    item = _evidence("run-a", "chunk-7")
    result = normalize_answer_payload({
        "answer": "系统包含任务管理模块。",
        "evidence_ids": [item.evidence_id],
        "limitations": None,
        "status": "sufficient",
    }, sufficiency_state="sufficient")

    assert result["claims"] == [{
        "text": "系统包含任务管理模块。",
        "evidence_ids": [item.evidence_id],
        "status": "supported",
    }]
    assert result["answerability"] == "supported"


def test_short_model_citation_aliases_are_mapped_to_server_evidence_ids():
    item = _evidence("run-a", "chunk-7")

    result = normalize_answer_payload({
        "answerability": "sufficient",
        "claims": [{
            "text": "系统包含任务管理模块。",
            "citations": ["E1"],
        }],
        "limitations": [],
    }, sufficiency_state="sufficient", evidence_aliases={"E1": item.evidence_id})

    assert result["answerability"] == "supported"
    assert result["claims"] == [{
        "text": "系统包含任务管理模块。",
        "evidence_ids": [item.evidence_id],
        "status": "supported",
    }]
    assert result["limitations"] == []


def test_citation_validation_drops_unknown_claims_and_maps_every_kept_claim():
    first = _evidence("run-a", "chunk-7")
    second = _evidence(
        "run-a", "relation-2", evidence_type="relation",
        channel="relation_graph", source=_source(8),
    )
    first.rank, second.rank = 1, 2
    validated = validate_and_render({
        "answer": "模型自由文本不会越过逐项声明校验。",
        "claims": [
            {
                "text": "系统包含任务管理模块。",
                "evidence_ids": [first.evidence_id],
                "status": "supported",
            },
            {
                "text": "这是没有来源的额外结论。",
                "evidence_ids": ["ev_unknown"],
                "status": "supported",
            },
        ],
        "limitations": [],
        "follow_up_questions": [],
    }, [first, second], sufficiency_state="sufficient")

    assert len(validated.answer_json["claims"]) == 1
    assert len(validated.citations) == 1
    assert validated.citations[0]["evidence_id"] == first.evidence_id
    assert "〔1〕" in validated.answer_text
    assert "额外结论" not in validated.answer_text
    assert validated.answer_json["citation_validation"][
        "unknown_or_uncited_claims_removed"
    ] == 1


def test_citation_validator_never_turns_top_evidence_into_an_answer_fallback():
    item = _evidence("run-a", "chunk-7")
    item.rank = 1

    validated = validate_and_render(
        {"claims": [{"text": "", "evidence_ids": [], "status": "supported"}]},
        [item], sufficiency_state="retryable",
    )

    assert validated.answer_json["claims"] == []
    assert validated.citations == []
    assert validated.answer_text == ""
