from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
import uuid

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge.config import Config
from knowledge_forge.database import Database, SCHEMA_SQL
from knowledge_forge.migrations import MIGRATIONS, MigrationRunner
from knowledge_forge.policy import RELATION_CANDIDATE_POLICY_SET
from knowledge_forge.relation_pipeline import (
    RelationBatchValidationError,
    RelationCandidate,
    RelationPlan,
    RelationCandidatePipeline,
    _sentences,
)
from knowledge_forge.service import KnowledgeForgeService


def candidate() -> RelationCandidate:
    return RelationCandidate(
        candidate_id="candidate-1",
        source_mention_id="mention-a",
        target_mention_id="mention-b",
        source_mention_key="mention-key-a",
        target_mention_key="mention-key-b",
        source_name="Alpha",
        target_name="Beta",
        source_type="Organization",
        target_type="Product",
        allowed_predicates=("produces",),
        allowed_directions=("as_is",),
        context_id="ctx-0-0",
        sentence_start=0,
        sentence_end=0,
        context_start_char=0,
        context_end_char=38,
        source_start_char=0,
        source_end_char=5,
        target_start_char=15,
        target_end_char=19,
    )


@pytest.mark.parametrize(
    ("overflow", "deferred", "work", "decision_count", "expected"),
    [
        ([{"candidate_id": "overflow"}], [], [], 0, {"overflow"}),
        ([], [{"candidate_id": "deferred"}], [], 0, {"deferred"}),
        ([], [], [{"status": "terminal_failed", "error_message": "missing response"}], 1,
         {"missing_response", "terminal_failure"}),
        ([], [], [{"status": "terminal_failed", "error_message": "model unavailable"}], 1,
         {"terminal_failure"}),
        ([], [], [{"status": "quarantined", "error_message": "unsafe"}], 1,
         {"quarantined"}),
    ],
)
def test_relation_coverage_failures_propagate_all_terminal_states(
    overflow, deferred, work, decision_count, expected,
):
    plan = RelationPlan(
        policy_set=RELATION_CANDIDATE_POLICY_SET,
        raw_pairs=1,
        blocked_pairs=0,
        candidates=[candidate()],
        deferred=list(deferred),
        overflow=list(overflow),
        qualified_count=1,
    )
    failures = KnowledgeForgeService._relation_coverage_failures(
        plan, work, decision_count,
    )
    assert expected <= set(failures)


def _database_for_dsn(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    database = Database(DisposableConfig)

    @contextmanager
    def connection():
        with psycopg.connect(dsn, row_factory=dict_row) as scoped:
            revision_context = database._revision_context.get()
            if revision_context:
                scoped.execute(
                    "SELECT set_config('knowledge_forge.revision_id',%s,TRUE),"
                    "set_config('knowledge_forge.ingestion_run_id',%s,TRUE),"
                    "set_config('knowledge_forge.revision_lease_token',%s,TRUE)",
                    (
                        revision_context["revision_id"],
                        revision_context["ingestion_run_id"],
                        revision_context["lease_token"],
                    ),
                )
            yield scoped

    database.connection = connection
    return database


def test_relation_pipeline_has_durable_leased_work_queue_and_attempt_audit():
    migration_sql = "\n".join(item.sql for item in MIGRATIONS)
    combined = SCHEMA_SQL + migration_sql
    assert "CREATE TABLE IF NOT EXISTS relation_shadow_work_items" in combined
    assert "CREATE TABLE IF NOT EXISTS relation_shadow_attempts" in combined
    for state in (
        "pending", "leased", "completed", "retryable_failed", "terminal_failed", "quarantined",
    ):
        assert state in combined
    for column in (
        "lease_owner", "lease_token", "lease_expires_at", "attempt", "next_attempt_at",
        "accepted_output_hash", "input_fingerprint",
    ):
        assert column in combined


def test_sentence_offsets_preserve_decimal_versions_urls_email_abbreviations_and_ellipsis():
    content = (
        'Version v1.2.3 costs 3.14 USD. Visit https://example.com/a.b or e-mail a.b@example.com. '
        'Dr. Smith said “ready...” Then继续……完成！分块: 只是正文。'
    )
    values = _sentences(content)
    assert "".join(text for _, _, text in values) == content
    assert [text for _, _, text in values] == [
        "Version v1.2.3 costs 3.14 USD. ",
        "Visit https://example.com/a.b or e-mail a.b@example.com. ",
        "Dr. Smith said “ready...” ",
        "Then继续……",
        "完成！",
        "分块: 只是正文。",
    ]
    assert all(content[start:end] == text for start, end, text in values)


def test_duplicate_evidence_requires_exact_offsets_and_hash():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha produces Beta. Alpha produces Beta."
    evidence = "Alpha produces Beta."
    second_start = content.rindex(evidence)
    decision = pipeline.validate_decisions(
        [candidate()],
        [{
            "candidate_id": "candidate-1",
            "predicate": "produces",
            "direction": "as_is",
            "evidence": evidence,
            "evidence_start": second_start,
            "evidence_end": second_start + len(evidence),
            "reason": "explicit",
        }],
        content=content,
        logical_schema={"relation_types": [{"name": "produces"}]},
    )[0]
    assert decision.evidence_start == second_start
    assert decision.evidence_end == second_start + len(evidence)
    assert decision.evidence_hash == hashlib.sha256(evidence.encode("utf-8")).hexdigest()

    with pytest.raises(RelationBatchValidationError, match="ambiguous"):
        pipeline.validate_decisions(
            [candidate()],
            [{
                "candidate_id": "candidate-1", "predicate": "produces",
                "direction": "as_is", "evidence": evidence, "reason": "explicit",
            }],
            content=content,
            logical_schema={"relation_types": [{"name": "produces"}]},
        )


def test_evidence_offsets_are_half_open_and_must_match_source_exactly():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha produces Beta."
    raw = [{
        "candidate_id": "candidate-1", "predicate": "produces", "direction": "as_is",
        "evidence": content, "evidence_start": 0, "evidence_end": len(content) - 1,
        "reason": "bad end",
    }]
    with pytest.raises(RelationBatchValidationError, match="exact source span"):
        pipeline.validate_decisions(
            [candidate()], raw, content=content,
            logical_schema={"relation_types": [{"name": "produces"}]},
        )


def test_classifier_repairs_offsets_only_for_exact_frozen_context_quote():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha produces Beta." + (" " * 18)
    relation_candidate = candidate()
    raw = [{
        "candidate_id": relation_candidate.candidate_id,
        "predicate": "produces",
        "direction": "as_is",
        "evidence": content,
        "evidence_start": relation_candidate.source_start_char,
        "evidence_end": relation_candidate.target_end_char,
        "reason": "explicit",
    }]

    repaired = pipeline._canonicalize_context_evidence_offsets(
        [relation_candidate], raw, content=content,
    )

    assert repaired[0]["evidence_start"] == relation_candidate.context_start_char
    assert repaired[0]["evidence_end"] == relation_candidate.context_end_char
    assert raw[0]["evidence_start"] == relation_candidate.source_start_char
    decision = pipeline.validate_decisions(
        [relation_candidate], repaired, content=content,
        logical_schema={"relation_types": [{"name": "produces"}]},
    )[0]
    assert decision.evidence_start == relation_candidate.context_start_char
    assert decision.evidence_end == relation_candidate.context_end_char


def test_classifier_repairs_uniquely_locatable_partial_context_quote():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Prefix. Alpha produces Beta. Suffix."
    evidence = "Alpha produces Beta."
    relation_candidate = candidate()
    relation_candidate = RelationCandidate(
        **{
            **relation_candidate.__dict__,
            "context_start_char": 0,
            "context_end_char": len(content),
            "source_start_char": content.index("Alpha"),
            "source_end_char": content.index("Alpha") + len("Alpha"),
            "target_start_char": content.index("Beta"),
            "target_end_char": content.index("Beta") + len("Beta"),
        },
    )
    repaired = pipeline._canonicalize_context_evidence_offsets(
        [relation_candidate],
        [{
            "candidate_id": relation_candidate.candidate_id,
            "predicate": "produces",
            "direction": "as_is",
            "evidence": evidence,
            "evidence_start": 0,
            "evidence_end": len(content),
            "reason": "explicit",
        }],
        content=content,
    )

    assert repaired[0]["evidence_start"] == content.index(evidence)
    assert repaired[0]["evidence_end"] == content.index(evidence) + len(evidence)


def test_flat_singleton_relation_response_is_unambiguous_but_multi_candidate_is_not():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    raw = {
        "predicate": "produces",
        "direction": "as_is",
        "evidence": "Alpha produces Beta.",
        "evidence_start": 0,
        "evidence_end": len("Alpha produces Beta."),
        "reason": "explicit",
    }

    normalized = pipeline._normalize_model_decisions([candidate()], raw)

    assert normalized == [{"candidate_id": "candidate-1", **raw}]
    with pytest.raises(RelationBatchValidationError, match="array or an object"):
        pipeline._normalize_model_decisions(
            [
                candidate(),
                RelationCandidate(**{**candidate().__dict__, "candidate_id": "candidate-2"}),
            ],
            raw,
        )


def test_singleton_relation_response_repairs_one_mangled_dynamic_key():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    decision = {
        "predicate": "produces",
        "direction": "as_is",
        "evidence": "Alpha produces Beta.",
        "evidence_start": 0,
        "evidence_end": len("Alpha produces Beta."),
        "reason": "explicit",
    }

    normalized = pipeline._normalize_model_decisions(
        [candidate()], {"candidate-1-with-missing-characters": decision},
    )

    assert normalized == [{"candidate_id": "candidate-1", **decision}]
    with pytest.raises(RelationBatchValidationError, match="array or an object"):
        pipeline._normalize_model_decisions(
            [
                candidate(),
                RelationCandidate(**{**candidate().__dict__, "candidate_id": "candidate-2"}),
            ],
            {"candidate-1-with-missing-characters": decision},
        )


def test_unsafe_model_positive_is_conservatively_classified_as_no_relation():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha produces Beta." + (" " * 18)
    relation_candidate = candidate()
    unsafe = [{
        "candidate_id": relation_candidate.candidate_id,
        "predicate": "located_in",
        "direction": "as_is",
        "evidence": content,
        "evidence_start": 0,
        "evidence_end": len(content),
        "reason": "unsupported model assertion",
    }]

    normalized = pipeline._fail_close_unsafe_model_decisions(
        [relation_candidate], unsafe,
        content=content,
        logical_schema={"relation_types": [{"name": "produces"}]},
    )

    assert normalized[0]["predicate"] == "NO_RELATION"
    assert normalized[0]["direction"] == "as_is"
    assert normalized[0]["evidence"] == ""
    assert normalized[0]["evidence_start"] is None
    assert "predicate is not allowed" in normalized[0]["reason"]
    assert unsafe[0]["predicate"] == "located_in"
    decision = pipeline.validate_decisions(
        [relation_candidate], normalized, content=content,
        logical_schema={"relation_types": [{"name": "produces"}]},
    )[0]
    assert decision.predicate == "NO_RELATION"


def test_p2_control_protocol_preserves_mention_identity_and_exact_half_open_positions():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha produces Beta. Alpha is archived."
    beta_start = content.index("Beta")
    plan = pipeline.plan(
        project_scope="project:revision",
        chunk_key="protocol",
        content=content,
        entities=[
            {
                "mention_id": "mention-alpha",
                "mention_key": "original-alpha-key",
                "surface": "Alpha",
                "type": "Organization",
                "mention_start": 0,
                "mention_end": 5,
                "confidence": 0.91,
                "canonical_id": "canonical-alpha",
                "evidence": "Alpha produces Beta.",
            },
            {
                "mention_id": "mention-beta",
                "mention_key": "original-beta-key",
                "name": "Beta",
                "type": "Product",
                "mention_start": beta_start,
                "mention_end": beta_start + 4,
                "confidence": 0.88,
                "canonical_id": "canonical-beta",
                "evidence": "Alpha produces Beta.",
            },
        ],
        logical_schema={
            "relation_types": [{
                "name": "produces",
                "source_types": ["Organization"],
                "target_types": ["Product"],
            }],
        },
    )

    assert plan.admitted_count == 1
    relation = plan.candidates[0]
    assert relation.source_mention_id == "mention-alpha"
    assert relation.target_mention_id == "mention-beta"
    assert relation.source_mention_key == "original-alpha-key"
    assert relation.target_mention_key == "original-beta-key"
    assert relation.source_canonical_id == "canonical-alpha"
    assert relation.target_canonical_id == "canonical-beta"
    assert (relation.source_start_char, relation.source_end_char) == (0, 5)
    assert (relation.target_start_char, relation.target_end_char) == (
        beta_start,
        beta_start + 4,
    )

    invalid = pipeline.plan(
        project_scope="project:revision",
        chunk_key="invalid-protocol",
        content=content,
        entities=[{
            "mention_id": "mention-alpha",
            "surface": "Alpha",
            "type": "Organization",
            "mention_start": content.rindex("Alpha"),
            "mention_end": content.rindex("Alpha") + 4,
            "confidence": 1.0,
            "canonical_id": "canonical-alpha",
            "evidence": "Alpha produces Beta.",
        }],
        logical_schema={"relation_types": []},
    )
    assert invalid.candidates == []
    assert invalid.raw_pairs == 0


def test_grounded_entity_assignments_are_attached_before_relation_planning():
    raw_entities = [
        {
            "mention_id": "raw-alpha",
            "mention_key": "original-alpha-key",
            "surface": "Alpha",
            "type": "Organization",
            "mention_start": 0,
            "mention_end": 5,
            "confidence": 0.9,
        },
        {
            "mention_id": "raw-beta",
            "surface": "Beta",
            "type": "Product",
            "mention_start": 15,
            "mention_end": 19,
            "confidence": 0.9,
        },
        {
            "mention_id": "raw-rejected",
            "surface": "Rejected",
            "type": "GenericEntity",
            "mention_start": 20,
            "mention_end": 28,
            "confidence": 0.2,
        },
    ]

    resolved = KnowledgeForgeService._relation_entities_with_canonicals(
        raw_entities,
        {
            "raw-alpha": "mention-alpha",
            "raw-beta": "mention-beta",
        },
        {
            "mention-alpha": "canonical-alpha",
            "mention-beta": "canonical-beta",
        },
    )

    assert resolved == [
        {
            **raw_entities[0],
            "mention_id": "mention-alpha",
            "mention_key": "original-alpha-key",
            "canonical_id": "canonical-alpha",
        },
        {
            **raw_entities[1],
            "mention_id": "mention-beta",
            "mention_key": "mention-beta",
            "canonical_id": "canonical-beta",
        },
    ]


def test_grounded_relation_endpoint_accepts_raw_or_normalized_mention_id():
    raw_to_mention = {"raw-alpha": "mention-alpha"}
    canonical_by_mention = {"mention-alpha": "canonical-alpha"}

    assert KnowledgeForgeService._relation_endpoint_mention_id(
        "raw-alpha", raw_to_mention, canonical_by_mention,
    ) == "mention-alpha"
    assert KnowledgeForgeService._relation_endpoint_mention_id(
        "mention-alpha", raw_to_mention, canonical_by_mention,
    ) == "mention-alpha"
    assert KnowledgeForgeService._relation_endpoint_mention_id(
        "missing", raw_to_mention, canonical_by_mention,
    ) == ""


def test_relation_pipeline_run_identity_changes_with_candidate_input():
    first = KnowledgeForgeService._relation_run_id(
        3, 5, "revision-1", 7, "policy-hash", "input-a",
    )
    repeated = KnowledgeForgeService._relation_run_id(
        3, 5, "revision-1", 7, "policy-hash", "input-a",
    )
    changed = KnowledgeForgeService._relation_run_id(
        3, 5, "revision-1", 7, "policy-hash", "input-b",
    )

    assert first == repeated
    assert first != changed
    assert first.startswith("relshadow_")


def test_relation_pipeline_input_fingerprint_includes_canonical_endpoints():
    candidate = {
        "candidate_id": "candidate-1",
        "source_mention_id": "mention-a",
        "target_mention_id": "mention-b",
        "source_canonical_id": "canonical-a",
        "target_canonical_id": "canonical-b",
    }
    first = KnowledgeForgeService._relation_input_fingerprint(
        revision_id="revision-1",
        chunk_id=7,
        content="Alpha produces Beta.",
        schema_version=1,
        treatment_hash="policy-hash",
        candidate_map={"candidate-1": candidate},
    )
    changed = KnowledgeForgeService._relation_input_fingerprint(
        revision_id="revision-1",
        chunk_id=7,
        content="Alpha produces Beta.",
        schema_version=1,
        treatment_hash="policy-hash",
        candidate_map={
            "candidate-1": {
                **candidate,
                "source_canonical_id": "canonical-a-v2",
            },
        },
    )

    assert changed != first


def test_relation_classifier_defaults_leave_room_for_complete_json(monkeypatch):
    monkeypatch.delenv("RELATION_CLASSIFIER_OUTPUT_RESERVE", raising=False)
    monkeypatch.delenv("RELATION_CLASSIFIER_MAX_CANDIDATES", raising=False)

    assert Config.relation_classifier_output_reserve >= 2048
    assert Config.relation_classifier_max_candidates <= 8


def test_relation_decision_schema_requires_exact_candidate_mapping():
    relation_candidate = candidate()
    content = "Alpha produces Beta." + ("x" * 18)
    schema = RelationCandidatePipeline._decision_response_schema(
        [relation_candidate],
        content=content,
    )

    assert schema["type"] == "object"
    assert schema["required"] == [
        relation_candidate.candidate_id,
    ]
    assert list(schema["properties"]) == [relation_candidate.candidate_id]
    assert schema["additionalProperties"] is False
    decision_schema = schema["properties"][relation_candidate.candidate_id]
    assert set(decision_schema["required"]) == {
        "predicate",
        "direction",
        "evidence",
        "evidence_start",
        "evidence_end",
        "reason",
    }
    assert "NO_RELATION" in decision_schema["properties"]["predicate"]["enum"]
    assert decision_schema["properties"]["direction"]["enum"] == [
        "as_is",
        "reversed",
        "symmetric",
    ]
    assert "absolute zero-based half-open offset" in (
        decision_schema["properties"]["evidence_start"]["description"].lower()
    )
    assert "full chunk content" in (
        decision_schema["properties"]["evidence_end"]["description"].lower()
    )
    assert decision_schema["properties"]["evidence"]["enum"] == [
        "",
        content,
    ]
    assert decision_schema["properties"]["evidence_start"]["enum"] == [
        None,
        0,
    ]
    assert decision_schema["properties"]["evidence_end"]["enum"] == [
        None,
        len(content),
    ]


def test_relation_model_decision_mapping_normalizes_to_strict_array():
    relation_candidate = candidate()
    normalized = RelationCandidatePipeline._normalize_model_decisions(
        [relation_candidate],
        {
            "decisions": {
                relation_candidate.candidate_id: {
                    "predicate": "NO_RELATION",
                    "direction": "as_is",
                    "evidence": "",
                    "reason": "not stated",
                },
            },
        },
    )

    assert normalized == [{
        "candidate_id": relation_candidate.candidate_id,
        "predicate": "NO_RELATION",
        "direction": "as_is",
        "evidence": "",
        "reason": "not stated",
    }]


def test_relation_model_decision_array_remains_compatible_with_strict_verifier():
    relation_candidate = candidate()
    decisions = [{
        "candidate_id": relation_candidate.candidate_id,
        "predicate": "NO_RELATION",
        "direction": "as_is",
        "evidence": "",
        "reason": "not stated",
    }]

    assert RelationCandidatePipeline._normalize_model_decisions(
        [relation_candidate], {"decisions": decisions},
    ) == decisions


def test_relation_root_candidate_mapping_matches_ollama_structured_output():
    relation_candidate = candidate()
    raw = {
        relation_candidate.candidate_id: {
            "predicate": "NO_RELATION",
            "direction": "as_is",
            "evidence": "",
            "reason": "not stated",
        },
    }

    assert RelationCandidatePipeline._normalize_model_decisions(
        [relation_candidate], raw,
    )[0]["candidate_id"] == relation_candidate.candidate_id


def test_relation_only_snapshot_passes_strict_control_protocol_and_maps_canonical_by_key():
    content = "Alpha produces Beta."

    class SnapshotDb:
        @staticmethod
        def get_revision_chunk(chunk_id, revision_id):
            assert (chunk_id, revision_id) == (7, "revision-1")
            return {
                "id": 7, "document_id": 5, "revision_id": "revision-1",
                "chunk_key": "chunk-1", "content": content, "content_hash": "chunk-hash",
            }

        @staticmethod
        def get_project(project_id):
            assert project_id == 3
            return {"id": 3, "project_key": "project-3", "llm_model": "llm",
                    "model_max_tokens": 4096, "active_policy_hash": "control"}

        @staticmethod
        def get_schema_version(project_id, version):
            assert (project_id, version) == (3, 1)
            return {
                "version": 1,
                "schema_json": {"relation_types": [{
                    "name": "produces", "source_types": ["Organization"],
                    "target_types": ["Product"],
                }]},
            }

        @staticmethod
        def list_control_entities_for_chunk(chunk_id, revision_id):
            assert (chunk_id, revision_id) == (7, "revision-1")
            return [
                {"id": 101, "mention_key": "key-alpha", "name": "Alpha",
                 "entity_type": "Organization", "canonical_entity_id": 201,
                 "document_id": 5, "revision_id": "revision-1", "chunk_id": 7,
                 "confidence": 0.91, "evidence_text": content,
                 "metadata": {"mention_start": 0, "mention_end": 5}},
                {"id": 102, "mention_key": "key-beta", "name": "Beta",
                 "entity_type": "Product", "canonical_entity_id": 202,
                 "document_id": 5, "revision_id": "revision-1", "chunk_id": 7,
                 "confidence": 0.88, "evidence_text": content,
                 "metadata": {"mention_start": 15, "mention_end": 19}},
            ]

        @staticmethod
        def list_control_relations_for_chunk(_chunk_id, _revision_id):
            return []

    service = KnowledgeForgeService.__new__(KnowledgeForgeService)
    service.db = SnapshotDb()
    captured = {}

    def relation_pipeline(_run_id, _project, _document_id, _chunk, extraction, _schema, **_):
        captured["entities"] = extraction["entities"]
        return {
            "coverage_complete": True,
            "coverage_failures": [],
            "metrics": {"input_tokens": 10, "output_tokens": 5, "model_calls": 1},
            "candidates": [{
                "candidate_id": "candidate-1",
                "source_mention_id": "101", "target_mention_id": "102",
                "source_mention_key": "key-alpha", "target_mention_key": "key-beta",
            }],
            "decisions": [{
                "candidate_id": "candidate-1", "predicate": "produces",
                "direction": "as_is", "evidence": content,
                "evidence_start": 0, "evidence_end": len(content), "confidence": 0.99,
            }],
        }

    service._run_versioned_relation_pipeline = relation_pipeline
    manifest = {
        "manifest_hash": "manifest-hash",
        "project": {"schema_version": 1},
        "documents": [{
            "document_id": 5, "revision_id": "revision-1",
            "chunks": [{"chunk_id": 7, "content_hash": "chunk-hash"}],
        }],
    }
    result = service.process_policy_snapshot_work(
        {"id": "snapshot-1", "project_id": 3, "mode": "relation_only",
         "snapshot_manifest": manifest, "manifest_hash": "manifest-hash",
         "policy_hash": "treatment"},
        {"phase": "relation", "document_id": 5, "revision_id": "revision-1", "chunk_id": 7},
    )

    assert captured["entities"] == [
        {"mention_id": "101", "mention_key": "key-alpha", "surface": "Alpha",
         "name": "Alpha", "type": "Organization", "evidence": content,
         "mention_start": 0, "mention_end": 5, "confidence": 0.91,
         "canonical_id": "201"},
        {"mention_id": "102", "mention_key": "key-beta", "surface": "Beta",
         "name": "Beta", "type": "Product", "evidence": content,
         "mention_start": 15, "mention_end": 19, "confidence": 0.88,
         "canonical_id": "202"},
    ]
    treatment_relation = result["evaluation_item"]["treatment"]["relations"][0]
    assert treatment_relation["subject_mention_id"] == "101"
    assert treatment_relation["subject_canonical_id"] == "201"
    assert treatment_relation["object_mention_id"] == "102"
    assert treatment_relation["object_canonical_id"] == "202"


def test_long_unrelated_chunk_prefix_does_not_consume_candidate_batch_budget():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = ("Unrelated background sentence. " * 300) + "Alpha produces Beta."
    alpha_start = content.index("Alpha")
    beta_start = content.index("Beta")
    prepared = pipeline.prepare_relation_batch(
        context_budget_tokens=2048, project_scope="project:revision",
        chunk_key="long-chunk", content=content,
        entities=[
            {
                "mention_id": "alpha", "mention_key": "alpha-key",
                "surface": "Alpha", "name": "Alpha", "type": "Organization",
                "mention_start": alpha_start, "mention_end": alpha_start + 5,
                "confidence": 0.9, "canonical_id": "canonical-alpha",
                "evidence": "Alpha produces Beta.",
            },
            {
                "mention_id": "beta", "mention_key": "beta-key",
                "surface": "Beta", "name": "Beta", "type": "Product",
                "mention_start": beta_start, "mention_end": beta_start + 4,
                "confidence": 0.9, "canonical_id": "canonical-beta",
                "evidence": "Alpha produces Beta.",
            },
        ],
        logical_schema={
            "relation_types": [{
                "name": "produces", "source_types": ["Organization"],
                "target_types": ["Product"],
            }],
        },
    )
    assert prepared.plan.admitted_count == 1
    assert prepared.plan.deferred == []
    assert len(prepared.batches) == 1
    payload = __import__("json").loads(prepared.batches[0].prompt)
    assert payload["source_contexts"] == [{
        "context_id": prepared.plan.candidates[0].context_id,
        "start": prepared.plan.candidates[0].context_start_char,
        "end": prepared.plan.candidates[0].context_end_char,
        "text": "Alpha produces Beta.",
    }]


def test_dense_candidates_are_streamed_across_batches_without_loss():
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "A relates B relates C relates D relates E relates F."
    names = ["A", "B", "C", "D", "E", "F"]
    entities = []
    cursor = 0
    for name in names:
        start = content.index(name, cursor)
        cursor = start + 1
        entities.append({
            "mention_id": name, "mention_key": f"key-{name}", "surface": name,
            "name": name, "type": "GenericEntity", "mention_start": start,
            "mention_end": start + 1, "confidence": 0.9,
            "canonical_id": f"canonical-{name}", "evidence": content,
        })
    plan = pipeline.plan(
        project_scope="project:revision", chunk_key="dense", content=content,
        entities=entities,
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
    )
    batches = pipeline.build_batches(
        plan, content=content,
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
        candidate_budget_tokens=2048, max_candidates_per_batch=3,
    )
    planned = [item.candidate_id for item in plan.candidates]
    batched = [candidate_id for batch in batches for candidate_id in batch.candidate_ids]
    assert len(planned) == 15
    assert batched == planned
    assert plan.deferred == []


def test_relation_candidate_limits_are_deterministic_and_conserve_overflow(monkeypatch):
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_targets_per_source", 2)
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_candidates_per_chunk", 5)
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_char_distance", 384)
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "A relates B relates C relates D relates E relates F."
    entities = []
    cursor = 0
    for name in "ABCDEF":
        start = content.index(name, cursor)
        cursor = start + 1
        entities.append({
            "mention_id": name, "mention_key": f"key-{name}", "surface": name,
            "name": name, "type": "GenericEntity", "mention_start": start,
            "mention_end": start + 1, "confidence": 0.9,
            "canonical_id": f"canonical-{name}", "evidence": content,
        })
    plan = pipeline.plan(
        project_scope="project:revision", chunk_key="bounded", content=content,
        entities=entities,
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
    )

    assert len(plan.candidates) == 5
    assert plan.qualified_count == len(plan.candidates) + len(plan.overflow)
    assert all(item["reason"] in {"per_source_limit", "chunk_limit"} for item in plan.overflow)
    assert [item.candidate_id for item in plan.candidates] == sorted(
        (item.candidate_id for item in plan.candidates),
        key=lambda candidate_id: next(
            candidate.priority_key for candidate in plan.candidates
            if candidate.candidate_id == candidate_id
        ),
    )


@pytest.mark.parametrize("count", [20, 50, 100])
def test_relation_plan_and_batches_bound_real_20_50_100_positioned_entities(
    monkeypatch, count,
):
    monkeypatch.setattr(
        "knowledge_forge.relation_pipeline.Config.relation_max_char_distance", 4096,
    )
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    names = [f"Entity{index:03d}" for index in range(count)]
    content = " relates ".join(names) + "."
    entities = []
    cursor = 0
    for index, name in enumerate(names):
        start = content.index(name, cursor)
        cursor = start + len(name)
        entities.append({
            "mention_id": f"mention-{index:03d}",
            "mention_key": f"original-{index:03d}",
            "surface": name,
            "type": "Organization" if index % 3 else "GenericEntity",
            "mention_start": start,
            "mention_end": start + len(name),
            "confidence": 0.99 - index / 1000,
            "canonical_id": f"canonical-{index:03d}",
            "evidence": content,
        })
    plan = pipeline.plan(
        project_scope="project:revision",
        chunk_key=f"real-{count}",
        content=content,
        entities=entities,
        logical_schema={
            "relation_types": [{"name": "related_to", "symmetric": True}],
        },
    )
    batches = pipeline.build_batches(
        plan, content=content,
        logical_schema={"relation_types": [{"name": "related_to"}]},
        candidate_budget_tokens=100_000, max_candidates_per_batch=100,
    )

    assert plan.qualified_count == count * (count - 1) // 2
    assert plan.admitted_count <= 256
    assert plan.qualified_count == plan.admitted_count + len(plan.overflow)
    per_source = {}
    for item in plan.candidates:
        per_source[item.source_mention_id] = per_source.get(item.source_mention_id, 0) + 1
    assert max(per_source.values(), default=0) <= 12
    assert all(len(batch.candidate_ids) <= 32 for batch in batches)
    assert [item for batch in batches for item in batch.candidate_ids] == [
        item.candidate_id for item in plan.candidates
    ]
    assert len(batches) == (plan.admitted_count + 31) // 32
    assert [item.priority_key for item in plan.candidates] == sorted(
        item.priority_key for item in plan.candidates
    )


def test_relation_char_distance_blocks_far_pairs_before_model_calls(monkeypatch):
    monkeypatch.setattr("knowledge_forge.relation_pipeline.Config.relation_max_char_distance", 32)
    pipeline = RelationCandidatePipeline(RELATION_CANDIDATE_POLICY_SET)
    content = "Alpha " + ("x" * 100) + " Beta."
    beta = content.index("Beta")
    plan = pipeline.plan(
        project_scope="project:revision", chunk_key="far", content=content,
        entities=[
            {"mention_id": "a", "mention_key": "key-a", "surface": "Alpha",
             "name": "Alpha", "type": "GenericEntity", "mention_start": 0,
             "mention_end": 5, "confidence": 0.9, "canonical_id": "canonical-a",
             "evidence": content},
            {"mention_id": "b", "mention_key": "key-b", "surface": "Beta",
             "name": "Beta", "type": "GenericEntity", "mention_start": beta,
             "mention_end": beta + 4, "confidence": 0.9, "canonical_id": "canonical-b",
             "evidence": content},
        ],
        logical_schema={"relation_types": [{"name": "related_to", "symmetric": True}]},
    )
    assert plan.candidates == []
    assert plan.qualified_count == 0


@pytest.mark.integration
def test_relation_pipeline_queue_recovers_leases_retries_and_late_workers_without_product_writes():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database_for_dsn(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"relation-shadow-{suffix}", f"Relation Shadow {suffix}"),
        ).fetchone()["id"])
        document_id = int(conn.execute(
            """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
               storage_path,local_path,file_size_bytes)
               VALUES(%s,%s,'shadow.txt','shadow.txt','/shadow','/shadow',1) RETURNING id""",
            (project_id, f"shadow-{suffix}"),
        ).fetchone()["id"])
    run_id = f"ingestion-{suffix}"
    manifest = {
        "content_hash": f"content-{suffix}", "source_hash": f"source-{suffix}",
        "config_hash": "config", "policy_hash": "policy",
    }
    revision = database.begin_document_revision(document_id, run_id, manifest)
    chunks = database.replace_document_content(
        document_id,
        [{
            "key": "section", "title": "Section", "level": 1,
            "content": "Alpha produces Beta.", "content_hash": "section-hash",
        }],
        [{
            "key": "chunk", "section_key": "section", "part_index": 1,
            "content": "Alpha produces Beta.", "content_hash": "chunk-hash",
        }],
        revision_id=revision["revision_id"], ingestion_run_id=run_id,
        lease_token=revision["lease_token"], manifest=manifest,
    )
    chunk_id = int(chunks[0]["id"])
    relation_run_id = f"shadow-run-{suffix}"
    work = [
        {
            "id": f"work-{suffix}-{index}", "revision_id": revision["revision_id"],
            "project_id": project_id, "document_id": document_id, "chunk_id": chunk_id,
            "batch_id": f"batch-{index}", "input_fingerprint": f"fingerprint-{index}",
            "candidate_ids": [f"candidate-{index}"], "request_json": {"batch": index},
            "estimated_input_tokens": 10, "max_output_tokens": 10, "max_attempts": 3,
        }
        for index in (1, 2)
    ]

    with database.content_revision_context(
        revision["revision_id"], run_id, revision["lease_token"],
    ):
        database.start_relation_pipeline_run({
            "id": relation_run_id, "project_id": project_id, "document_id": document_id,
            "chunk_id": chunk_id, "control_policy_hash": "control",
            "treatment_policy_hash": "treatment", "policy_set": RELATION_CANDIDATE_POLICY_SET,
            "input_fingerprint": f"run-fingerprint-{suffix}",
        })
        database.prepare_relation_pipeline_work(relation_run_id, work)
        with database.connection() as conn:
            product_before = {
                table: int(conn.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE project_id=%s", (project_id,),
                ).fetchone()["count"])
                for table in (
                    "relation_assertions", "relation_candidates",
                    "mention_relation_assertions", "canonical_relation_projections",
                    "vector_embeddings", "jobs",
                )
            }

        expired = database.lease_relation_pipeline_work(relation_run_id, "worker-expired")
        assert expired and expired["batch_id"] == "batch-1"
        concurrent = database.lease_relation_pipeline_work(relation_run_id, "worker-concurrent")
        assert concurrent and concurrent["batch_id"] == "batch-2"
        assert concurrent["id"] != expired["id"]
        with database.connection() as conn:
            conn.execute(
                "UPDATE relation_shadow_work_items SET lease_expires_at=NOW()-INTERVAL '1 second' WHERE id=%s",
                (expired["id"],),
            )
        recovered = database.lease_relation_pipeline_work(relation_run_id, "worker-recovery")
        assert recovered and recovered["id"] == expired["id"]
        accepted = database.settle_relation_pipeline_work(
            recovered["id"], recovered["lease_token"],
            output={"decisions": [{"candidate_id": "candidate-1"}]},
        )
        assert accepted["status"] == "completed"
        late = database.settle_relation_pipeline_work(
            expired["id"], expired["lease_token"],
            output={"decisions": [{"candidate_id": "candidate-1", "late": True}]},
        )
        assert late["accepted_output_hash"] == accepted["accepted_output_hash"]

        failed = database.settle_relation_pipeline_work(
            concurrent["id"], concurrent["lease_token"], error_message="temporary model outage",
        )
        assert failed["status"] == "retryable_failed"
        retried = database.lease_relation_pipeline_work(relation_run_id, "worker-retry")
        assert retried and retried["attempt"] == 2
        database.settle_relation_pipeline_work(
            retried["id"], retried["lease_token"],
            output={"decisions": [{"candidate_id": "candidate-2"}]},
        )

        rows = database.list_relation_pipeline_work(relation_run_id)
        assert [row["status"] for row in rows] == ["completed", "completed"]
        assert all(float(row["accepted_latency_ms"] or 0) >= 0 for row in rows)
        persisted_candidate = {
            **candidate().__dict__,
            "chunk_id": chunk_id,
        }
        persisted_decision = {
            "candidate_id": "candidate-1",
            "predicate": "produces",
            "direction": "as_is",
            "evidence": "Alpha produces Beta.",
            "evidence_start": 0,
            "evidence_end": 20,
            "evidence_hash": hashlib.sha256(
                b"Alpha produces Beta."
            ).hexdigest(),
            "reason": "explicit statement",
            "status": "classified",
        }
        result = database.finish_relation_pipeline_run(
            relation_run_id,
            candidates=[persisted_candidate],
            decisions=[persisted_decision],
            metrics={"classified": 2, "model_calls": 4},
        )
        repeated = database.finish_relation_pipeline_run(
            relation_run_id,
            candidates=[persisted_candidate],
            decisions=[persisted_decision],
            metrics={"classified": 2, "model_calls": 4},
        )
        assert result["artifact_hash"] == repeated["artifact_hash"]
        assert result["snapshot_complete"] is True
        with database.connection() as conn:
            persisted = conn.execute(
                """SELECT allowed_predicates FROM relation_shadow_decisions
                   WHERE shadow_run_id=%s AND candidate_id='candidate-1'""",
                (relation_run_id,),
            ).fetchone()
        assert persisted["allowed_predicates"] == ["produces"]

        terminal_run_id = f"terminal-shadow-run-{suffix}"
        database.start_relation_pipeline_run({
            "id": terminal_run_id, "project_id": project_id, "document_id": document_id,
            "chunk_id": chunk_id, "control_policy_hash": "control",
            "treatment_policy_hash": "treatment", "policy_set": RELATION_CANDIDATE_POLICY_SET,
            "input_fingerprint": f"terminal-fingerprint-{suffix}",
        })
        terminal_work = {
            **work[0], "id": f"terminal-work-{suffix}", "batch_id": "terminal-batch",
            "input_fingerprint": f"terminal-work-fingerprint-{suffix}", "max_attempts": 1,
        }
        database.prepare_relation_pipeline_work(terminal_run_id, [terminal_work])
        terminal_lease = database.lease_relation_pipeline_work(terminal_run_id, "terminal-worker")
        terminal = database.settle_relation_pipeline_work(
            terminal_lease["id"], terminal_lease["lease_token"], error_message="permanent failure",
        )
        assert terminal["status"] == "terminal_failed"
        with pytest.raises(ValueError, match="failed work item"):
            database.finish_relation_pipeline_run(
                terminal_run_id, candidates=[], decisions=[], metrics={}, status="completed",
            )
        terminal_result = database.finish_relation_pipeline_run(
            terminal_run_id, candidates=[], decisions=[], metrics={}, status="terminal_failed",
        )
        assert terminal_result["snapshot_complete"] is True
        assert terminal_result["failed_work_items"] == 1

        with database.connection() as conn:
            attempts = conn.execute(
                """SELECT attempt_no,status FROM relation_shadow_attempts
                   WHERE work_item_id=%s ORDER BY attempt_no""",
                (expired["id"],),
            ).fetchall()
            assert attempts == [
                {"attempt_no": 1, "status": "superseded"},
                {"attempt_no": 2, "status": "accepted"},
            ]
            product_after = {
                table: int(conn.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE project_id=%s", (project_id,),
                ).fetchone()["count"])
                for table in (
                    "relation_assertions", "relation_candidates",
                    "mention_relation_assertions", "canonical_relation_projections",
                    "vector_embeddings", "jobs",
                )
            }
        assert product_after == product_before


@pytest.mark.integration
def test_postgres_same_name_retry_preserves_two_occurrence_assignments():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database_for_dsn(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"same-name-{suffix}", f"Same Name {suffix}"),
        ).fetchone()["id"])
        document_id = int(conn.execute(
            """INSERT INTO documents(project_id,doc_hash,original_filename,stored_filename,
               storage_path,local_path,file_size_bytes)
               VALUES(%s,%s,'same.txt','same.txt','/same','/same',1) RETURNING id""",
            (project_id, f"same-{suffix}"),
        ).fetchone()["id"])
        entities = conn.execute(
            """INSERT INTO canonical_entities(
               project_id,entity_key,name,normalized_name,entity_type)
               VALUES(%s,%s,'张伟','张伟','Person'),(%s,%s,'张伟','张伟','Person')
               RETURNING id""",
            (project_id, f"zhang-a-{suffix}", project_id, f"zhang-b-{suffix}"),
        ).fetchall()
    first_entity, second_entity = sorted(int(item["id"]) for item in entities)
    run_id = f"same-name-run-{suffix}"
    manifest = {
        "content_hash": f"same-content-{suffix}", "source_hash": f"same-source-{suffix}",
        "config_hash": "config", "policy_hash": "policy",
    }
    revision = database.begin_document_revision(document_id, run_id, manifest)
    chunks = database.replace_document_content(
        document_id,
        [{"key": "section", "title": "Section", "level": 1,
          "content": "张伟会见张伟。", "content_hash": "section-hash"}],
        [{"key": "chunk", "section_key": "section", "part_index": 1,
          "content": "张伟会见张伟。", "content_hash": "chunk-hash"}],
        revision_id=revision["revision_id"], ingestion_run_id=run_id,
        lease_token=revision["lease_token"], manifest=manifest,
    )
    chunk_id = int(chunks[0]["id"])
    base = {
        "project_id": project_id, "document_id": document_id, "chunk_id": chunk_id,
        "revision_id": revision["revision_id"], "mention_text": "张伟",
        "normalized_name": "张伟", "entity_type": "Person", "confidence": 0.9,
        "extraction_version": "v2", "evidence_text": "张伟会见张伟。",
        "evidence_start": 0, "evidence_end": 7,
    }
    mentions = [
        {**base, "canonical_entity_id": first_entity, "mention_start": 0, "mention_end": 2},
        {**base, "canonical_entity_id": second_entity, "mention_start": 4, "mention_end": 6},
    ]
    for item in mentions:
        item["mention_key"] = Database._stable_mention_key(item)

    def decisions(values):
        return [{
            **item, "decision": "llm_merge", "candidate_entity_ids": [item["canonical_entity_id"]],
            "scores": {"llm": 0.99}, "reason": "test", "model_name": "test-llm",
        } for item in values]

    with database.content_revision_context(
        revision["revision_id"], run_id, revision["lease_token"],
    ):
        database.persist_entity_resolution_batch(mentions, decisions(mentions))
        reversed_retry = [
            {**mentions[0], "canonical_entity_id": second_entity},
            {**mentions[1], "canonical_entity_id": first_entity},
        ]
        database.persist_entity_resolution_batch(
            reversed_retry, decisions(reversed_retry),
        )

    assignments = database.find_active_entity_assignments_by_mention_keys(
        project_id, [item["mention_key"] for item in mentions],
    )
    assert assignments[mentions[0]["mention_key"]]["id"] == first_entity
    assert assignments[mentions[1]["mention_key"]]["id"] == second_entity
    assert first_entity != second_entity
