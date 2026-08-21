from benchmarks.knowledge_forge_hybrid_rag_v1 import (
    evaluate_retrieval as evaluator,
)
from benchmarks.knowledge_forge_hybrid_rag_v1.evaluate_retrieval import (
    load_benchmark,
    score_question,
    summarize,
)


def test_retrieval_scorer_combines_graph_fact_with_vector_negative_evidence():
    benchmark = load_benchmark()
    question = next(
        item for item in benchmark["questions"]
        if item["question_id"] == "rag-q019"
    )
    assertion = benchmark["assertions"][
        question["required_assertion_ids"][0]
    ]
    negative = benchmark["negatives"][
        question["required_negative_claim_ids"][0]
    ]
    response = {
        "chunks": [{"content": negative["evidence_text"]}],
        "relations": [{
            "source_name": "华宁材料厂",
            "relation_type": assertion["predicate"],
            "target_name": "江苏",
            "evidence_text": assertion["evidence_text"],
        }],
    }

    scored = score_question(
        question, response, benchmark, top_k=5,
    )

    assert scored["vector_success"] is False
    assert scored["graph_positive_hits"] == 1
    assert scored["vector_negative_evidence_hits"] == 1
    assert scored["hybrid_success"] is True
    assert scored["negative_relation_leaks"] == 0


def test_retrieval_scorer_separates_grounded_fact_value_from_predicate_name():
    benchmark = load_benchmark()
    question = next(
        item for item in benchmark["questions"]
        if item["question_id"] == "rag-q001"
    )
    assertion = benchmark["assertions"][
        question["required_assertion_ids"][0]
    ]
    response = {
        "chunks": [],
        "relations": [{
            "source_name": "`if` statement",
            "relation_type": "enables_purpose",
            "target_name": "conditional execution",
            "evidence_text": assertion["evidence_text"],
        }],
    }

    scored = score_question(question, response, benchmark, top_k=5)

    assert scored["graph_positive_hits"] == 1
    assert scored["graph_exact_predicate_positive_hits"] == 0
    assert scored["graph_success"] is True
    assert scored["hybrid_success"] is True


def test_retrieval_scorer_fails_closed_on_generic_relation_publication():
    benchmark = load_benchmark()
    question = next(
        item for item in benchmark["questions"]
        if item["question_id"] == "rag-q001"
    )
    assertion = benchmark["assertions"][
        question["required_assertion_ids"][0]
    ]
    response = {
        "chunks": [{"content": assertion["evidence_text"]}],
        "relations": [{
            "source_name": "`if` statement",
            "relation_type": "related_to",
            "target_name": "conditional execution",
            "evidence_text": assertion["evidence_text"],
        }],
    }

    scored = score_question(question, response, benchmark, top_k=5)

    assert scored["generic_relation_publications"] == 1
    assert scored["graph_success"] is False
    assert scored["hybrid_success"] is False


def test_retrieval_summary_rejects_a_published_hard_negative():
    benchmark = load_benchmark()
    rows = [{
        "question_id": f"q{index}",
        "language": "zh-CN" if index % 2 else "en",
        "task_type": "single_fact",
        "retrieval_expectation": "graph_preferred",
        "vector_success": True,
        "graph_success": True,
        "hybrid_success": True,
        "negative_relation_leaks": 1 if index == 0 else 0,
        "vector_context_chars": 1000,
        "graph_context_chars": 100,
    } for index in range(24)]

    report = summarize(rows, benchmark["manifest"])

    assert report["checks"]["negative_relation_safety"] is False
    assert report["status"] == "fail"


def test_retrieval_summary_accepts_compressed_grounded_graph_context():
    benchmark = load_benchmark()
    rows = [{
        "question_id": f"q{index}",
        "language": "zh-CN" if index % 2 else "en",
        "task_type": "aggregation",
        "retrieval_expectation": "graph_preferred",
        "vector_success": True,
        "graph_success": False,
        "hybrid_success": True,
        "required_positive_facts": 4,
        "graph_positive_hits": 3,
        "negative_relation_leaks": 0,
        "generic_relation_publications": 0,
        "vector_context_chars": 1000,
        "graph_context_chars": 200,
    } for index in range(24)]

    report = summarize(rows, benchmark["manifest"])

    assert report["rates"]["graph_suitable_fact_coverage"] == 0.75
    assert report["graph_suitable_facts"]["context_ratio"] == 0.2
    assert report["checks"]["graph_adds_value"] is True
    assert report["status"] == "pass"


def test_real_route_evaluator_bypasses_environment_proxies(monkeypatch):
    captured = {"calls": 0}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def read():
            return b'{"chunks": [], "relations": []}'

    class Opener:
        @staticmethod
        def open(_url, *, timeout):
            assert timeout == 180
            captured["calls"] += 1
            return Response()

    def build_opener(handler):
        captured["proxies"] = handler.proxies
        return Opener()

    monkeypatch.setattr(evaluator, "build_opener", build_opener)

    report = evaluator.evaluate_project("http://127.0.0.1:8011", 78)

    assert captured["proxies"] == {}
    assert captured["calls"] == report["questions"] == 24
