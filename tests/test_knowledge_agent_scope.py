from __future__ import annotations

from knowledge_forge.agent.contracts import EvidenceItem, SnapshotRef, SourceRef
from knowledge_forge.agent.scope import (
    assess_input_query_scope,
    assess_query_scope,
    is_short_bare_topic,
    preflight_query_scope,
    scope_boundary_payload,
)
from knowledge_forge.agent.sufficiency import SufficiencyResult


def _evidence() -> EvidenceItem:
    snapshot = SnapshotRef(
        strategy="knowledge_forge", identifier="kf:scope",
        source_hash="source-scope", generation=1, graph_available=True,
    )
    return EvidenceItem.create(
        run_id="scope-run", evidence_type="chunk", project_id=7,
        strategy="knowledge_forge", snapshot_ref=snapshot,
        record_key="scope-chunk", title="联邦学习的隐私保护",
        content="联邦学习让参与方在不交换原始数据的情况下协同训练模型。",
        source_refs=[SourceRef.from_dict({
            "document_id": 3, "revision_id": "revision-1",
            "filename": "federated-learning.pdf", "pages": [4],
        })],
        retrieval_channels=["chunk_dense"],
        raw_scores={"chunk_dense": 0.72},
    )


def _sufficiency(
    *, relevance: float, semantic: float, aspect: float, state: str = "retryable",
) -> SufficiencyResult:
    return SufficiencyResult(
        state=state, score=0.4, aspect_coverage=aspect,
        relevance=relevance, semantic_relevance=semantic,
        reranker_used_model=True, provenance=1.0,
        independent_support=0.5, graph_connectivity=0.45,
        conflict_penalty=0.0, reasons=(),
    )


def test_agent_meta_preflight_is_high_precision_and_domain_requests_stay_in_scope():
    for query in (
        "你能做什么", "你有什么能力？", "如何使用这个问答智能体？",
        "介绍一下你自己", "What can you do?", "你好",
    ):
        assert preflight_query_scope(query).route == "agent_meta"

    for query in (
        "你能介绍一下联邦学习吗？",
        "这个系统能做什么任务？",
        "知识库中的任务管理有什么能力？",
    ):
        assert preflight_query_scope(query).route == "knowledge"


def test_semantic_input_scope_skips_retrieval_only_for_confident_out_of_scope():
    class InputScopeModel:
        def __init__(self, *, route: str, confidence: float) -> None:
            self.route = route
            self.confidence = confidence
            self.calls = []

        def json_chat(self, **values):
            self.calls.append(values)
            assert "检索前" in values["prompt"]
            assert "联邦学习" in values["prompt"]
            return {
                "route": self.route,
                "request_kind": (
                    "external_or_general"
                    if self.route == "out_of_scope" else "knowledge_lookup"
                ),
                "profile_alignment": (
                    "none" if self.route == "out_of_scope" else "direct"
                ),
                "confidence": self.confidence,
            }

    profile = {
        "documents": [{"filename": "联邦学习在医学领域中的应用.pdf"}],
        "document_count": 1,
    }
    confident = InputScopeModel(route="out_of_scope", confidence=0.98)
    decision = assess_input_query_scope(
        ollama=confident,
        project={
            "name": "联邦学习资料",
            "llm_model": "qwen3.6:latest",
            "model_max_tokens": 8192,
        },
        query="今天是几月几日？",
        knowledge_scope=profile,
    )
    uncertain = InputScopeModel(route="out_of_scope", confidence=0.42)
    fallback = assess_input_query_scope(
        ollama=uncertain,
        project={
            "name": "联邦学习资料",
            "llm_model": "qwen3.6:latest",
            "model_max_tokens": 8192,
        },
        query="这项研究采用了什么日期划分？",
        knowledge_scope=profile,
    )

    assert decision.route == "out_of_scope"
    assert decision.model_used is True
    assert fallback.route == "knowledge"
    assert fallback.reason_code == "input_scope_not_confident_to_skip_retrieval"


def test_semantic_input_scope_failure_preserves_recall_by_continuing_to_retrieval():
    class BrokenModel:
        def json_chat(self, **_values):
            raise RuntimeError("router unavailable")

    decision = assess_input_query_scope(
        ollama=BrokenModel(),
        project={
            "name": "系统手册",
            "llm_model": "qwen3.6:latest",
            "model_max_tokens": 8192,
        },
        query="该系统如何调度任务？",
        knowledge_scope={"documents": [{"filename": "系统手册.pdf"}]},
    )

    assert decision.route == "knowledge"
    assert decision.method == "input_scope_fail_open_to_retrieval"
    assert decision.model_error


def test_short_bare_topic_always_uses_retrieval_first_for_scope():
    class NeverCalled:
        def json_chat(self, **_values):  # pragma: no cover - failure sentinel
            raise AssertionError("short topics must not be rejected from profile metadata")

    assert is_short_bare_topic("草莓") is True
    assert is_short_bare_topic("越野跑") is True
    assert is_short_bare_topic("什么是草莓？") is True
    assert is_short_bare_topic("介绍草莓") is True
    assert is_short_bare_topic("其他方法有哪些？") is False
    assert is_short_bare_topic("它的作用") is False

    decision = assess_input_query_scope(
        ollama=NeverCalled(),
        project={
            "name": "联邦学习资料",
            "description": "医学领域中的联邦学习研究资料",
            "llm_model": "qwen3.6:latest",
            "model_max_tokens": 8192,
        },
        query="草莓",
        knowledge_scope={
            "documents": [{"filename": "联邦学习在医学领域中的应用.pdf"}],
        },
    )

    assert decision.route == "knowledge"
    assert decision.confidence == 0.0
    assert decision.method == "short_topic_retrieval_first"
    assert decision.reason_code == "short_topic_requires_evidence_check"
    assert decision.model_used is False


def test_strong_cross_encoder_alignment_skips_scope_model():
    class NeverCalled:
        def json_chat(self, **_kwargs):  # pragma: no cover - failure sentinel
            raise AssertionError("strong evidence must not call the scope model")

    evidence = _evidence()
    decision = assess_query_scope(
        ollama=NeverCalled(),
        project={"llm_model": "qwen3.6:latest", "model_max_tokens": 8192},
        query="联邦学习如何保护原始数据？",
        evidence=[evidence],
        sufficiency=_sufficiency(relevance=0.94, semantic=0.72, aspect=0.5),
    )

    assert decision.route == "knowledge"
    assert decision.method == "strong_retrieval_signals"
    assert decision.model_used is False
    assert decision.matched_evidence_ids == (evidence.evidence_id,)


def test_short_abbreviation_with_exact_compound_token_skips_scope_model():
    class NeverCalled:
        def json_chat(self, **_kwargs):  # pragma: no cover - failure sentinel
            raise AssertionError("exact short-topic evidence must skip scope model")

    evidence = EvidenceItem.create(
        run_id="scope-run", evidence_type="chunk", project_id=7,
        strategy="knowledge_forge",
        snapshot_ref=SnapshotRef(
            strategy="knowledge_forge", identifier="kf:scope",
            source_hash="source-scope", generation=1, graph_available=True,
        ),
        record_key="fl-chunk", title="FL-QSAR 架构",
        content="该架构基于联邦学习，用于药物发现。",
        source_refs=[SourceRef.from_dict({
            "document_id": 3, "revision_id": "revision-1",
            "filename": "federated-learning.pdf", "pages": [4],
        })],
        retrieval_channels=["chunk_lexical"],
        raw_scores={"chunk_lexical": 0.4},
    )
    decision = assess_query_scope(
        ollama=NeverCalled(),
        project={"llm_model": "qwen3.6:latest", "model_max_tokens": 8192},
        query="FL是什么", evidence=[evidence],
        sufficiency=_sufficiency(
            relevance=0.28, semantic=0.42, aspect=1.0, state="sufficient",
        ),
    )

    assert decision.route == "knowledge"
    assert decision.method == "short_topic_exact_lexical_alignment"
    assert decision.matched_evidence_ids == (evidence.evidence_id,)


def test_borderline_alignment_uses_structured_scope_model_and_does_not_answer():
    class ScopeModel:
        def __init__(self) -> None:
            self.calls = []

        def json_chat(self, **values):
            self.calls.append(values)
            assert "只判断" in values["prompt"]
            assert values["temperature"] == 0.0
            return {
                "route": "out_of_scope",
                "confidence": 0.98,
                "matched_evidence_indices": [],
            }

    model = ScopeModel()
    decision = assess_query_scope(
        ollama=model,
        project={
            "name": "联邦学习论文",
            "llm_model": "qwen3.6:latest",
            "model_max_tokens": 8192,
        },
        query="法国的首都是什么？",
        evidence=[_evidence()],
        sufficiency=_sufficiency(relevance=0.01, semantic=0.50, aspect=0.0),
        knowledge_scope={
            "documents": [{"filename": "federated-learning.pdf"}],
        },
    )

    assert len(model.calls) == 1
    assert decision.route == "out_of_scope"
    assert decision.model_used is True
    assert decision.matched_evidence_ids == ()


def test_scope_model_failure_is_reported_as_retryable_service_degradation():
    class BrokenModel:
        def json_chat(self, **_values):
            raise RuntimeError("model unavailable")

    decision = assess_query_scope(
        ollama=BrokenModel(),
        project={"llm_model": "qwen3.6:latest", "model_max_tokens": 8192},
        query="一个弱匹配问题",
        evidence=[_evidence()],
        sufficiency=_sufficiency(relevance=0.01, semantic=0.50, aspect=0.0),
    )

    assert decision.route == "service_degraded"
    assert decision.method == "service_degraded"
    assert decision.reason_code == "scope_service_unavailable"
    assert decision.retryable is True
    assert decision.degraded_components == ("ollama",)
    assert decision.model_error


def test_post_retrieval_scope_does_not_reclassify_unrelated_question_as_agent_meta():
    class InvalidMetaModel:
        def json_chat(self, **_values):
            return {
                "route": "agent_meta",
                "confidence": 0.0,
                "matched_evidence_indices": [],
            }

    decision = assess_query_scope(
        ollama=InvalidMetaModel(),
        project={"llm_model": "qwen3.6:latest", "model_max_tokens": 8192},
        query="今天是几月几日？",
        evidence=[_evidence()],
        sufficiency=_sufficiency(relevance=0.01, semantic=0.1, aspect=0.0),
    )

    assert decision.route == "service_degraded"
    assert decision.method == "service_degraded"


def test_low_confidence_scope_result_is_uncertain_instead_of_a_hard_rejection():
    class ScopeModel:
        def json_chat(self, **_values):
            return {
                "route": "out_of_scope",
                "confidence": 0.42,
                "matched_evidence_indices": [],
            }

    decision = assess_query_scope(
        ollama=ScopeModel(),
        project={"llm_model": "qwen3.6:latest", "model_max_tokens": 8192},
        query="一个边界不清的问题",
        evidence=[_evidence()],
        sufficiency=_sufficiency(relevance=0.01, semantic=0.1, aspect=0.0),
    )

    assert decision.route == "uncertain"
    assert decision.reason_code == "scope_model_low_confidence"
    assert decision.model_used is True


def test_model_can_explicitly_report_an_ambiguous_scope():
    class ScopeModel:
        def json_chat(self, **_values):
            return {
                "route": "uncertain",
                "confidence": 0.91,
                "matched_evidence_indices": [],
            }

    decision = assess_query_scope(
        ollama=ScopeModel(),
        project={"llm_model": "qwen3.6:latest", "model_max_tokens": 8192},
        query="这个问题可能有多个解释",
        evidence=[_evidence()],
        sufficiency=_sufficiency(relevance=0.01, semantic=0.1, aspect=0.0),
    )

    assert decision.route == "uncertain"
    assert decision.reason_code == "scope_model_uncertain"


def test_scope_boundary_payload_explains_each_non_answer_route():
    project = {"name": "联邦学习资料", "knowledge_language": "zh-CN"}

    assert "无关" in scope_boundary_payload(
        project, "草莓", route="out_of_scope",
    )["answer"]
    assert "没有找到能直接支持" in scope_boundary_payload(
        project, "未覆盖的问题", route="unsupported",
    )["answer"]
    assert "无法可靠判断" in scope_boundary_payload(
        project, "边界问题", route="uncertain",
    )["answer"]
    degraded = scope_boundary_payload(
        project, "当前问题", route="service_degraded",
    )
    assert degraded["answerability"] == "degraded"
    assert "模型服务暂时不可用" in degraded["answer"]
