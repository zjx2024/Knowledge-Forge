from __future__ import annotations

from contextlib import contextmanager

import pytest

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.config import Config
from knowledge_forge.graph_policy import graph_policy_hash
from knowledge_forge.policy import CONTROL_POLICY_SET, RELATION_CANDIDATE_POLICY_SET
from knowledge_forge.relation_pipeline import (
    NO_RELATION,
    RelationCandidate,
    RelationPlan,
    RelationBatchValidationError,
    RelationCandidatePipeline,
)
from knowledge_forge.schema_evolution import BASE_SCHEMA


def _entities():
    return [
        {
            "mention_id": "m1", "mention_key": "mention-key-1",
            "surface": "Alpha", "name": "Alpha", "type": "Organization",
            "mention_start": 0, "mention_end": 5, "confidence": 0.95,
            "canonical_id": "canonical-1", "evidence": "Alpha uses Beta.",
        },
        {
            "mention_id": "m2", "mention_key": "mention-key-2",
            "surface": "Beta", "name": "Beta", "type": "Product",
            "mention_start": 11, "mention_end": 15, "confidence": 0.94,
            "canonical_id": "canonical-2", "evidence": "Alpha uses Beta.",
        },
    ]


def _schema():
    return {
        "relation_types": [{
            "name": "uses", "source_types": ["Organization"],
            "target_types": ["Product"], "inverse": "used_by", "symmetric": False,
        }]
    }


def test_graph_policy_hash_covers_all_six_policy_axes():
    baseline = graph_policy_hash(CONTROL_POLICY_SET, schema_version=3, config_summary={"window": 1})
    assert baseline == graph_policy_hash(
        dict(CONTROL_POLICY_SET), schema_version=3, config_summary={"window": 1},
    )
    for axis in CONTROL_POLICY_SET:
        changed = dict(CONTROL_POLICY_SET)
        changed[axis] += "-changed"
        assert graph_policy_hash(changed, schema_version=3, config_summary={"window": 1}) != baseline


def test_stage_one_mentions_produce_stable_blocked_candidates_and_counts():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    first = pipeline.plan(
        project_scope="project-1", chunk_key="chunk-1",
        content="Alpha uses Beta.", entities=_entities(), logical_schema=_schema(),
    )
    second = pipeline.plan(
        project_scope="project-1", chunk_key="chunk-1",
        content="Alpha uses Beta.", entities=_entities(), logical_schema=_schema(),
    )

    assert first.raw_pairs == 2
    assert first.blocked_pairs == 1
    assert first.admitted_count == 1
    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id
    assert first.candidates[0].source_mention_id == "m1"
    assert first.candidates[0].target_mention_id == "m2"
    renamed_ids = pipeline.plan(
        project_scope="project-1", chunk_key="chunk-1", content="Alpha uses Beta.",
        entities=[{
            **item, "mention_id": f"other-{index}",
        } for index, item in enumerate(_entities())],
        logical_schema=_schema(),
    )
    assert renamed_ids.candidates[0].candidate_id != first.candidates[0].candidate_id


def test_base_schema_wildcard_matches_multiple_concrete_entity_types() -> None:
    plan = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET).plan(
        project_scope="project-1",
        chunk_key="wildcard",
        content="Alpha uses Beta.",
        entities=_entities(),
        logical_schema=BASE_SCHEMA,
    )

    uses = [item for item in plan.candidates if "uses" in item.allowed_predicates]
    assert uses
    assert {(item.source_type, item.target_type) for item in uses} == {
        ("Organization", "Product"),
        ("Product", "Organization"),
    }


def test_relation_planner_rejects_pairs_already_resolved_to_same_entity() -> None:
    entities = [
        {**_entities()[0], "canonical_id": "canonical-shared"},
        {**_entities()[1], "canonical_id": "canonical-shared"},
    ]

    plan = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET).plan(
        project_scope="project-1",
        chunk_key="same-canonical",
        content="Alpha uses Beta.",
        entities=entities,
        logical_schema=_schema(),
    )

    assert plan.raw_pairs == 2
    assert plan.qualified_count == 0
    assert plan.blocked_pairs == 2
    assert plan.candidates == []
    assert plan.overflow == []


def test_dense_entity_list_without_relation_cue_does_not_create_cartesian_candidates() -> None:
    names = ["Brazil", "France", "Germany", "India", "Japan", "Mexico", "Peru", "Spain"]
    content = ", ".join(names) + "."
    entities = []
    cursor = 0
    for index, name in enumerate(names):
        start = content.index(name, cursor)
        cursor = start + len(name)
        entities.append({
            "mention_id": f"m{index}",
            "mention_key": f"key-{index}",
            "surface": name,
            "name": name,
            "type": "Location",
            "mention_start": start,
            "mention_end": start + len(name),
            "confidence": 0.95,
            "canonical_id": f"canonical-{index}",
            "evidence": content,
        })

    plan = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET).plan(
        project_scope="project-1",
        chunk_key="dense-list",
        content=content,
        entities=entities,
        logical_schema={
            "relation_types": [{
                "name": "related_to",
                "source_types": ["Location"],
                "target_types": ["Location"],
                "symmetric": True,
            }],
        },
    )

    assert plan.raw_pairs == len(names) * (len(names) - 1)
    assert plan.qualified_count == 0
    assert plan.blocked_pairs == plan.raw_pairs
    assert plan.candidates == []
    assert plan.overflow == []


def test_relation_cue_outside_endpoint_span_does_not_qualify_list_pairs() -> None:
    content = (
        "Global end uses were estimated as follows: batteries, 87%; ceramics and glass, "
        "5%; lubricating greases, 2%; air treatment, 1%; medical, 1%."
    )
    names = ["ceramics", "lubricating greases", "air treatment", "medical"]
    entities = []
    for index, name in enumerate(names):
        start = content.index(name)
        entities.append({
            "mention_id": f"m{index}",
            "mention_key": f"key-{index}",
            "surface": name,
            "name": name,
            "type": "Product",
            "mention_start": start,
            "mention_end": start + len(name),
            "confidence": 0.95,
            "canonical_id": f"canonical-{index}",
            "evidence": content,
        })

    plan = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET).plan(
        project_scope="project-1",
        chunk_key="misleading-context-cue",
        content=content,
        entities=entities,
        logical_schema={
            "relation_types": [{
                "name": "uses",
                "source_types": ["Product"],
                "target_types": ["Product"],
            }],
        },
    )

    assert plan.qualified_count == 0
    assert plan.candidates == []


def test_dense_sentence_with_explicit_relation_cue_remains_eligible() -> None:
    content = "A relates B relates C relates D relates E relates F relates G."
    entities = []
    cursor = 0
    for index, name in enumerate("ABCDEFG"):
        start = content.index(name, cursor)
        cursor = start + len(name)
        entities.append({
            "mention_id": f"m{index}",
            "mention_key": f"key-{index}",
            "surface": name,
            "name": name,
            "type": "GenericEntity",
            "mention_start": start,
            "mention_end": start + len(name),
            "confidence": 0.95,
            "canonical_id": f"canonical-{index}",
            "evidence": content,
        })

    plan = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET).plan(
        project_scope="project-1",
        chunk_key="dense-cue",
        content=content,
        entities=entities,
        logical_schema={
            "relation_types": [{
                "name": "related_to",
                "source_types": ["GenericEntity"],
                "target_types": ["GenericEntity"],
                "symmetric": True,
            }],
        },
    )

    assert plan.qualified_count == len(entities) * (len(entities) - 1) // 2
    assert plan.candidates


def test_relation_batches_do_not_mix_different_predicate_contracts() -> None:
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha uses Beta in Paris."
    common = {
        "source_mention_id": "m1",
        "target_mention_id": "m2",
        "source_mention_key": "k1",
        "target_mention_key": "k2",
        "source_name": "Alpha",
        "target_name": "Beta",
        "source_type": "Organization",
        "target_type": "Product",
        "context_id": "ctx",
        "sentence_start": 0,
        "sentence_end": 0,
        "context_start_char": 0,
        "context_end_char": len(content),
        "source_start_char": 0,
        "source_end_char": 5,
        "target_start_char": 11,
        "target_end_char": 15,
    }
    candidates = [
        RelationCandidate(
            candidate_id="candidate-general",
            allowed_predicates=("related_to", "uses"),
            allowed_directions=("as_is", "as_is"),
            **common,
        ),
        RelationCandidate(
            candidate_id="candidate-location",
            allowed_predicates=("related_to", "located_in"),
            allowed_directions=("as_is", "as_is"),
            **common,
        ),
    ]
    plan = RelationPlan(
        policy_set=dict(RELATION_CANDIDATE_POLICY_SET),
        raw_pairs=2,
        blocked_pairs=0,
        candidates=candidates,
        qualified_count=2,
    )

    batches = pipeline.build_batches(
        plan,
        content=content,
        logical_schema={
            "relation_types": [
                {"name": "related_to", "source_types": ["*"], "target_types": ["*"]},
                {"name": "uses", "source_types": ["*"], "target_types": ["*"]},
                {"name": "located_in", "source_types": ["*"], "target_types": ["Location"]},
            ],
        },
        candidate_budget_tokens=2048,
        output_token_reserve=512,
        max_candidates_per_batch=8,
    )

    assert [batch.candidate_ids for batch in batches] == [
        ["candidate-general"],
        ["candidate-location"],
    ]


def test_relation_batches_do_not_mix_different_evidence_contexts() -> None:
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha uses Beta. Gamma uses Delta."
    common = {
        "allowed_predicates": ("related_to", "uses"),
        "allowed_directions": ("as_is", "as_is"),
        "sentence_start": 0,
        "sentence_end": 0,
    }
    candidates = [
        RelationCandidate(
            candidate_id="candidate-first",
            source_mention_id="m1",
            target_mention_id="m2",
            source_mention_key="k1",
            target_mention_key="k2",
            source_name="Alpha",
            target_name="Beta",
            source_type="Organization",
            target_type="Product",
            context_id="ctx-first",
            context_start_char=0,
            context_end_char=16,
            source_start_char=0,
            source_end_char=5,
            target_start_char=11,
            target_end_char=15,
            **common,
        ),
        RelationCandidate(
            candidate_id="candidate-second",
            source_mention_id="m3",
            target_mention_id="m4",
            source_mention_key="k3",
            target_mention_key="k4",
            source_name="Gamma",
            target_name="Delta",
            source_type="Organization",
            target_type="Product",
            context_id="ctx-second",
            context_start_char=17,
            context_end_char=len(content),
            source_start_char=17,
            source_end_char=22,
            target_start_char=28,
            target_end_char=33,
            **common,
        ),
    ]
    plan = RelationPlan(
        policy_set=dict(RELATION_CANDIDATE_POLICY_SET),
        raw_pairs=2,
        blocked_pairs=0,
        candidates=candidates,
        qualified_count=2,
    )

    batches = pipeline.build_batches(
        plan,
        content=content,
        logical_schema={
            "relation_types": [
                {"name": "related_to", "source_types": ["*"], "target_types": ["*"]},
                {"name": "uses", "source_types": ["*"], "target_types": ["*"]},
            ],
        },
        candidate_budget_tokens=2048,
        output_token_reserve=512,
        max_candidates_per_batch=8,
    )

    assert [batch.candidate_ids for batch in batches] == [
        ["candidate-first"],
        ["candidate-second"],
    ]


def test_relation_planner_checks_only_sentence_local_pairs_without_changing_denominator(
    monkeypatch,
):
    import knowledge_forge.relation_pipeline as relation_pipeline

    entity_count = 100
    sentences = [f"Entity{index} is present." for index in range(entity_count)]
    content = " ".join(sentences)
    entities = []
    cursor = 0
    for index, sentence in enumerate(sentences):
        name = f"Entity{index}"
        start = content.index(name, cursor)
        cursor = start + len(sentence)
        entities.append({
            "mention_id": f"m{index}", "mention_key": f"key-{index}",
            "surface": name, "name": name, "type": "GenericEntity",
            "mention_start": start, "mention_end": start + len(name),
            "confidence": 0.9, "canonical_id": f"canonical-{index}",
            "evidence": sentence,
        })
    checks = 0
    original = relation_pipeline._type_allowed

    def counted(actual, expected):
        nonlocal checks
        checks += 1
        return original(actual, expected)

    monkeypatch.setattr(relation_pipeline, "_type_allowed", counted)
    plan = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET).plan(
        project_scope="project-1", chunk_key="chunk-dense", content=content,
        entities=entities,
        logical_schema={"relation_types": [{
            "name": "related_to", "source_types": ["GenericEntity"],
            "target_types": ["GenericEntity"], "symmetric": False,
        }]},
        sentence_window=1,
    )

    assert plan.raw_pairs == entity_count * (entity_count - 1)
    assert plan.blocked_pairs + plan.admitted_count == plan.raw_pairs
    assert checks < entity_count * 10


def test_compact_batches_repeat_context_once_and_defer_without_truncation():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    plan = pipeline.plan(
        project_scope="project-1", chunk_key="chunk-1",
        content="Alpha uses Beta.", entities=_entities(), logical_schema=_schema(),
    )
    batches = pipeline.build_batches(
        plan, content="Alpha uses Beta.", logical_schema=_schema(), candidate_budget_tokens=200,
    )
    assert len(batches) == 1
    assert batches[0].prompt.count("Alpha uses Beta.") == 1
    assert batches[0].candidate_ids == [plan.candidates[0].candidate_id]
    assert plan.deferred == []

    deferred_plan = pipeline.plan(
        project_scope="project-1", chunk_key="chunk-1",
        content="Alpha uses Beta.", entities=_entities(), logical_schema=_schema(),
    )
    deferred = pipeline.build_batches(
        deferred_plan,
        content="Alpha uses Beta.", logical_schema=_schema(), candidate_budget_tokens=1,
    )
    assert deferred == []
    assert deferred_plan.deferred == [{
        "candidate_id": deferred_plan.candidates[0].candidate_id,
        "reason": "candidate_exceeds_budget",
    }]


def test_classifier_requires_one_explicit_decision_per_admitted_candidate():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    plan = pipeline.plan(
        project_scope="project-1", chunk_key="chunk-1",
        content="Alpha uses Beta.", entities=_entities(), logical_schema=_schema(),
    )
    candidate_id = plan.candidates[0].candidate_id

    decisions = pipeline.validate_decisions(
        plan.candidates,
        [{
            "candidate_id": candidate_id, "predicate": NO_RELATION,
            "direction": "as_is", "evidence": "", "reason": "co-occurrence only",
        }],
        content="Alpha uses Beta.", logical_schema=_schema(),
    )
    assert decisions[0].predicate == NO_RELATION

    with pytest.raises(RelationBatchValidationError, match="missing"):
        pipeline.validate_decisions(plan.candidates, [], content="Alpha uses Beta.", logical_schema=_schema())
    with pytest.raises(RelationBatchValidationError, match="duplicate"):
        duplicate = [{
            "candidate_id": candidate_id, "predicate": NO_RELATION,
            "direction": "as_is", "evidence": "", "reason": "none",
        }] * 2
        pipeline.validate_decisions(
            plan.candidates, duplicate, content="Alpha uses Beta.", logical_schema=_schema(),
        )


def test_candidate_direction_constraints_cover_inverse_and_symmetric_relations():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    direct = pipeline.plan(
        project_scope="p", chunk_key="c", content="Alpha uses Beta.",
        entities=_entities(), logical_schema=_schema(),
    )
    assert direct.candidates[0].allowed_directions == ("as_is",)
    with pytest.raises(RelationBatchValidationError, match="direction"):
        pipeline.validate_decisions(
            direct.candidates,
            [{
                "candidate_id": direct.candidates[0].candidate_id,
                "predicate": "uses", "direction": "reversed",
                "evidence": "Alpha uses Beta", "reason": "wrong direction",
            }],
            content="Alpha uses Beta.", logical_schema=_schema(),
        )

    symmetric_schema = {
        "relation_types": [{
            "name": "related_to", "source_types": ["GenericEntity"],
            "target_types": ["GenericEntity"], "symmetric": True,
        }]
    }
    symmetric_content = "Alpha relates to Beta."
    beta_start = symmetric_content.index("Beta")
    symmetric_entities = [
        {**_entities()[0], "type": "GenericEntity"},
        {
            **_entities()[1], "type": "GenericEntity", "mention_start": beta_start,
            "mention_end": beta_start + 4, "evidence": symmetric_content,
        },
    ]
    symmetric = pipeline.plan(
        project_scope="p", chunk_key="c", content=symmetric_content,
        entities=symmetric_entities, logical_schema=symmetric_schema,
    )
    assert symmetric.raw_pairs == 2
    assert symmetric.admitted_count == 1
    assert symmetric.candidates[0].allowed_directions == ("symmetric",)


def test_coverage_gleaning_expands_only_uncovered_cue_windows_once():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha is present. The system depends on context. Beta is present."
    entities = [
        {**_entities()[0], "mention_start": 0, "mention_end": 5, "evidence": content},
        {
            **_entities()[1], "mention_start": content.index("Beta"),
            "mention_end": content.index("Beta") + 4, "evidence": content,
        },
    ]
    result = pipeline.classify_relation_batch(
        classifier=_Classifier(), model="fixture", context_budget_tokens=4096,
        project_scope="p", chunk_key="c", content=content,
        entities=entities, logical_schema=_schema(),
    )
    assert result.metrics["gleaning_passes"] <= 1
    assert result.metrics["raw_pairs"] == 2


def test_grounded_schema_is_additive_and_shadow_write_never_updates_product_projection():
    assert "CREATE TABLE IF NOT EXISTS policy_snapshots" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS policy_snapshot_documents" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS relation_shadow_runs" in SCHEMA_SQL
    assert "CREATE TABLE IF NOT EXISTS relation_shadow_decisions" in SCHEMA_SQL
    for axis in CONTROL_POLICY_SET:
        assert axis in SCHEMA_SQL

    class Result:
        def fetchone(self):
            return {"id": "shadow-1"}

    class Connection:
        def __init__(self):
            self.calls = []

        def execute(self, statement, params=None):
            self.calls.append((str(statement), params))
            return Result()

    connection = Connection()
    database = Database()

    @contextmanager
    def connect():
        yield connection

    database.connection = connect
    database.start_relation_pipeline_run({
        "id": "shadow-1", "project_id": 3, "document_id": 5, "chunk_id": 9,
        "control_policy_hash": "control", "treatment_policy_hash": "treatment",
        "policy_set": RELATION_CANDIDATE_POLICY_SET,
    })
    statements = "\n".join(statement for statement, _ in connection.calls)
    assert "INSERT INTO relation_shadow_runs" in statements
    assert "UPDATE projects" not in statements
    assert "relation_assertions" not in statements
    assert "vector_embeddings" not in statements
    assert "jobs" not in statements


class _Classifier:
    def __init__(self, omit=False):
        self.calls = []
        self.omit = omit

    def json_chat(self, **kwargs):
        self.calls.append(kwargs)
        payload = __import__("json").loads(kwargs["prompt"])
        decisions = [{
            "candidate_id": item["candidate_id"], "predicate": NO_RELATION,
            "direction": "as_is", "evidence": "", "reason": "not explicitly related",
        } for item in payload["candidates"]]
        if self.omit:
            decisions = decisions[:-1]
        return {"decisions": decisions}


def test_relation_classifier_is_bounded_and_missing_decision_fails_closed():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    classifier = _Classifier()
    result = pipeline.classify_relation_batch(
        classifier=classifier, model="fixture", context_budget_tokens=4096,
        project_scope="project-1", chunk_key="chunk-1", content="Alpha uses Beta.",
        entities=_entities(), logical_schema=_schema(),
    )
    assert result.status == "completed"
    assert result.metrics["classified"] == 1
    assert result.metrics["model_calls"] == 1
    assert classifier.calls[0]["context_budget_tokens"] == 4096
    assert classifier.calls[0]["max_output_tokens"] < 4096

    failed = pipeline.classify_relation_batch(
        classifier=_Classifier(omit=True), model="fixture", context_budget_tokens=4096,
        project_scope="project-1", chunk_key="chunk-1", content="Alpha uses Beta.",
        entities=_entities(), logical_schema=_schema(),
    )
    assert failed.status == "failed"
    assert failed.metrics["classified"] == 0
    assert "missing" in failed.error
