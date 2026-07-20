from __future__ import annotations

from contextlib import contextmanager

import pytest

from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.config import Config
from knowledge_forge.graph_policy import graph_policy_hash
from knowledge_forge.policy import CONTROL_POLICY_SET, P1_RELATION_SHADOW_POLICY_SET
from knowledge_forge.relation_pipeline import (
    NO_RELATION,
    RelationBatchValidationError,
    VersionedRelationPipeline,
)


def _entities():
    return [
        {
            "id": "m1", "name": "Alpha", "type": "Organization",
            "evidence_text": "Alpha", "evidence_start": 0, "evidence_end": 5,
        },
        {
            "id": "m2", "name": "Beta", "type": "Product",
            "evidence_text": "Beta", "evidence_start": 11, "evidence_end": 15,
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
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
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
    assert first.candidates[0].source_mention_id.startswith("p1mention_")
    assert first.candidates[0].target_mention_id.startswith("p1mention_")
    renamed_ids = pipeline.plan(
        project_scope="project-1", chunk_key="chunk-1", content="Alpha uses Beta.",
        entities=[{**item, "id": f"other-{index}"} for index, item in enumerate(_entities())],
        logical_schema=_schema(),
    )
    assert renamed_ids.candidates[0].candidate_id == first.candidates[0].candidate_id


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
            "id": f"m{index}", "name": name, "type": "GenericEntity",
            "evidence_text": name, "evidence_start": start,
            "evidence_end": start + len(name),
        })
    checks = 0
    original = relation_pipeline._type_allowed

    def counted(actual, expected):
        nonlocal checks
        checks += 1
        return original(actual, expected)

    monkeypatch.setattr(relation_pipeline, "_type_allowed", counted)
    plan = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET).plan(
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
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
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
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
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
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
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
            **_entities()[1], "type": "GenericEntity", "evidence_start": beta_start,
            "evidence_end": beta_start + 4,
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
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    content = "Alpha is present. The system depends on context. Beta is present."
    entities = [
        {**_entities()[0], "evidence_start": 0, "evidence_end": 5, "evidence_text": "Alpha"},
        {
            **_entities()[1], "evidence_start": content.index("Beta"),
            "evidence_end": content.index("Beta") + 4, "evidence_text": "Beta",
        },
    ]
    result = pipeline.classify_shadow(
        classifier=_Classifier(), model="fixture", context_budget_tokens=4096,
        project_scope="p", chunk_key="c", content=content,
        entities=entities, logical_schema=_schema(),
    )
    assert result.metrics["gleaning_passes"] <= 1
    assert result.metrics["raw_pairs"] == 2


def test_shadow_schema_is_additive_and_shadow_write_never_updates_product_projection():
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
    database.start_relation_shadow_run({
        "id": "shadow-1", "project_id": 3, "document_id": 5, "chunk_id": 9,
        "control_policy_hash": "control", "treatment_policy_hash": "treatment",
        "policy_set": P1_RELATION_SHADOW_POLICY_SET,
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


def test_shadow_classifier_is_bounded_and_missing_decision_fails_closed():
    assert Config.relation_shadow_enabled is False
    pipeline = VersionedRelationPipeline(P1_RELATION_SHADOW_POLICY_SET)
    classifier = _Classifier()
    result = pipeline.classify_shadow(
        classifier=classifier, model="fixture", context_budget_tokens=4096,
        project_scope="project-1", chunk_key="chunk-1", content="Alpha uses Beta.",
        entities=_entities(), logical_schema=_schema(),
    )
    assert result.status == "completed"
    assert result.metrics["classified"] == 1
    assert result.metrics["model_calls"] == 1
    assert classifier.calls[0]["context_budget_tokens"] == 4096
    assert classifier.calls[0]["max_output_tokens"] < 4096

    failed = pipeline.classify_shadow(
        classifier=_Classifier(omit=True), model="fixture", context_budget_tokens=4096,
        project_scope="project-1", chunk_key="chunk-1", content="Alpha uses Beta.",
        entities=_entities(), logical_schema=_schema(),
    )
    assert failed.status == "failed"
    assert failed.metrics["classified"] == 0
    assert "missing" in failed.error
