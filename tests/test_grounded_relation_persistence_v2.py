from __future__ import annotations

import os
import inspect
import uuid

import pytest
from psycopg.conninfo import conninfo_to_dict

from knowledge_forge.database import Database
from knowledge_forge.relation_extraction import (
    HYPOTHESES,
    PASS,
    SUPPORTED,
    PairClassificationDecision,
    RelationMaterializationDecision,
    RelationFreshnessDecision,
    normalize_verified_hypotheses,
    parse_classifier_response,
    plan_pair_opportunities,
    validate_hypothesis,
    verify_hypothesis,
)
from tests.grounded_relation_fixtures import (
    logical_schema,
    persist_pair_source_context,
    predicate_policy,
)


def _role_probe(*_args):
    return {
        "subject_role": "source endpoint role",
        "object_role": "target endpoint role",
    }


def _role_alignment_probe(*_args):
    return {"subject_role": "ALIGNED", "object_role": "ALIGNED"}


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _schema() -> dict:
    return logical_schema(
        predicate_policy(
            "uses",
            domain=["Organization"],
            range_=["Product"],
        ),
        version="schema-v1",
    )


@pytest.mark.integration
def test_grounded_model_cache_is_component_scoped_across_code_tree_changes() -> None:
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    project_id = int(database.create_project(
        f"Shadow component cache {suffix}", "", "test-model", "test-embed",
    )["id"])
    prior_run_id = f"shadow-cache-prior-{suffix}"
    current_run_id = f"shadow-cache-current-{suffix}"
    prior_fingerprint = f"prior-fingerprint-{suffix}"
    current_fingerprint = f"current-fingerprint-{suffix}"
    database.start_grounded_relation_run({
        "id": prior_run_id,
        "project_id": project_id,
        "input_fingerprint": prior_fingerprint,
        "fingerprint_components": {
            "materialization_root": {"code_tree": "old-tree"},
        },
        "source_generation": 1,
    })
    persisted = database.record_grounded_relation_source_model_call(
        prior_run_id,
        {
            "component": "source_inventory",
            "model_name": "test-model",
            "model_digest": "model-digest",
            "prompt_version": "prompt-v1",
            "schema_version": "schema-v1",
            "parser_version": "parser-v1",
            "idempotency_key": "inventory:one",
            "attempt_no": 1,
            "status": "COMPLETED",
            "request_hash": "exact-request-hash",
            "response_hash": "exact-response-hash",
            "request_payload": {"prompt": "exact rendered request"},
            "raw_response_text": '{"facts":[]}',
            "parsed_response": {"facts": []},
            "token_usage": {},
            "terminal_status": "COMPLETED",
        },
    )
    database.finish_grounded_relation_run(
        prior_run_id,
        status="ready",
        metrics={},
        input_fingerprint=prior_fingerprint,
    )
    database.start_grounded_relation_run({
        "id": current_run_id,
        "project_id": project_id,
        "input_fingerprint": current_fingerprint,
        "fingerprint_components": {
            "materialization_root": {"code_tree": "new-unrelated-tree"},
        },
        "source_generation": 1,
    })

    reused = database.find_reusable_relation_model_call(
        run_id=current_run_id,
        component="source_inventory",
        model_digest="model-digest",
        prompt_version="prompt-v1",
        schema_version="schema-v1",
        parser_version="parser-v1",
        request_hash="exact-request-hash",
    )
    assert reused is not None
    assert reused["id"] == persisted["id"]
    reparsed = database.find_reusable_relation_model_call(
        run_id=current_run_id,
        component="source_inventory",
        model_digest="model-digest",
        prompt_version="prompt-v1",
        schema_version="schema-v1",
        parser_version="parser-v2-stricter-local-parser",
        request_hash="exact-request-hash",
    )
    assert reparsed is not None
    assert reparsed["id"] == persisted["id"]
    assert database.find_reusable_relation_model_call(
        run_id=current_run_id,
        component="source_inventory",
        model_digest="model-digest",
        prompt_version="prompt-v2",
        schema_version="schema-v1",
        parser_version="parser-v1",
        request_hash="exact-request-hash",
    ) is None
    database.finish_grounded_relation_run(
        current_run_id,
        status="cancelled",
        metrics={},
        input_fingerprint=current_fingerprint,
    )


@pytest.mark.integration
def test_grounded_relation_v2_persistence_is_idempotent_and_product_isolated() -> None:
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"shadow-v2-{suffix}", f"Shadow v2 {suffix}"),
        ).fetchone()["id"])
        product_before = {
            table: int(conn.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE project_id=%s",
                (project_id,),
            ).fetchone()["count"])
            for table in (
                "relation_assertions",
                "canonical_relation_projections",
                "vector_embeddings",
                "entity_mentions",
                "entity_assignments",
                "canonical_entity_aliases",
            )
        }

    content = "Alpha uses Beta."
    run_id = f"shadow-v2-run-{suffix}"
    fingerprint = f"shadow-v2-fingerprint-{suffix}"
    plan = plan_pair_opportunities(
        project_id=project_id,
        document_revision_id=f"revision-{suffix}",
        chunk_key=f"chunk-{suffix}",
        content=content,
        mentions=[
            {
                "occurrence_key": "occ-alpha",
                "mention_id": "mention-alpha",
                "surface": "Alpha",
                "mention_start": 0,
                "mention_end": 5,
                "canonical_id": "entity-alpha",
                "assignment_version": "assignment-v1",
                "type": "Organization",
            },
            {
                "occurrence_key": "occ-beta",
                "mention_id": "mention-beta",
                "surface": "Beta",
                "mention_start": 11,
                "mention_end": 15,
                "canonical_id": "entity-beta",
                "assignment_version": "assignment-v1",
                "type": "Product",
            },
        ],
        logical_schema=_schema(),
        run_id=run_id,
        candidate_policy_version="candidate-v2",
    )
    pair = plan.admitted[0]
    decisions = parse_classifier_response(
        plan.admitted,
        [{
            "pair_id": pair.pair_identity,
            "decision": HYPOTHESES,
            "hypotheses": [{
                "subject_occurrence_id": "occ-alpha",
                "predicate": "uses",
                "object_occurrence_id": "occ-beta",
                "evidence_spans": [{"start": 0, "end": len(content)}],
                "classifier_confidence": 0.9,
            }],
        }],
        content=content,
        logical_schema=_schema(),
        classifier_model_revision="classifier-r1",
    )
    assert isinstance(decisions[0], PairClassificationDecision)
    hypothesis = decisions[0].hypotheses[0]
    validation = validate_hypothesis(
        hypothesis, pair, content=content, logical_schema=_schema(),
    )
    assert validation.status == PASS
    attempt = verify_hypothesis(
        hypothesis,
        validation,
        pair,
        content=content,
        logical_schema=_schema(),
        role_probe=_role_probe,
        role_alignment_probe=_role_alignment_probe,
        verifier=lambda _prompt, _schema: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": SUPPORTED,
            "reason_code": "EXACT_TRIPLE_SUPPORTED",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": [{"start": 0, "end": len(content)}],
        },
        verifier_model_revision="verifier-r1",
    )
    normalization = normalize_verified_hypotheses(
        project_id=project_id,
        opportunities={pair.pair_identity: pair},
        hypotheses=[hypothesis],
        verification_attempts=[attempt],
        logical_schema=_schema(),
    )
    materialization = [
        RelationMaterializationDecision(
            fact_id=normalization.facts[0].fact_id,
            eligible=True,
            reasons=(),
            freshness=RelationFreshnessDecision(
                checkpoint="materialize",
                current=True,
                status="CURRENT",
                recorded_root=fingerprint,
                current_root=fingerprint,
                changed_components=(),
            ),
        )
    ]

    database.start_grounded_relation_run({
        "id": run_id,
        "project_id": project_id,
        "input_fingerprint": fingerprint,
        "fingerprint_components": {"test": suffix},
        "source_generation": 1,
    })
    persist_pair_source_context(database, run_id, pair, content)
    database.persist_grounded_relation_opportunities(run_id, plan.opportunities)
    database.record_grounded_relation_classification_attempt(
        run_id,
        pair.pair_identity,
        {
            "attempt_no": 1,
            "status": "ACCEPTED",
            "request_hash": "request-hash",
            "response_hash": "response-hash",
            "classifier_prompt_version": hypothesis.classifier_prompt_version,
            "classifier_schema_version": hypothesis.classifier_schema_version,
            "classifier_parser_version": hypothesis.classifier_parser_version,
            "classifier_model_revision": hypothesis.classifier_model_revision,
            "model_name": "test-model",
            "request_payload": {"stage": "classifier"},
            "raw_response_text": "{\"decisions\": []}",
            "parsed_response": {"decisions": []},
            "token_usage": {},
            "error_type": "",
            "independent_call": True,
        },
    )
    database.persist_grounded_relation_classifications(run_id, decisions)
    database.persist_grounded_relation_validations(run_id, [validation])
    database.persist_grounded_relation_verification_attempts(run_id, [attempt])
    database.persist_grounded_relation_normalization(
        run_id,
        normalization,
        input_fingerprint=fingerprint,
        materialization_decisions=materialization,
    )
    database.finish_grounded_relation_run(
        run_id,
        status="ready",
        metrics={"pair_opportunities_total": 1, "facts_materialization_eligible": 1},
        input_fingerprint=fingerprint,
    )

    # Every persistence stage is retry-safe and may be replayed after a worker restart.
    database.start_grounded_relation_run({
        "id": run_id,
        "project_id": project_id,
        "input_fingerprint": fingerprint,
        "fingerprint_components": {"test": suffix},
        "source_generation": 1,
    })
    database.persist_grounded_relation_opportunities(run_id, plan.opportunities)
    database.persist_grounded_relation_classifications(run_id, decisions)
    database.persist_grounded_relation_validations(run_id, [validation])
    database.persist_grounded_relation_verification_attempts(run_id, [attempt])
    database.persist_grounded_relation_normalization(
        run_id,
        normalization,
        input_fingerprint=fingerprint,
        materialization_decisions=materialization,
    )

    with database.connection() as conn:
        assert conn.execute(
            "SELECT status FROM shadow_relation_v2_runs WHERE id=%s", (run_id,),
        ).fetchone()["status"] == "ready"
        expected_counts = {
            "shadow_relation_pair_opportunities": 1,
            "shadow_relation_classification_attempts": 1,
            "shadow_relation_hypotheses": 1,
            "shadow_relation_hypothesis_validations": 1,
            "shadow_relation_verification_attempts": 1,
            "shadow_relation_normalized_facts": 1,
            "shadow_relation_fact_memberships": 1,
            "shadow_relation_fact_evidence": 1,
        }
        for table, expected in expected_counts.items():
            assert int(conn.execute(
                f"SELECT COUNT(*) AS count FROM {table}",
            ).fetchone()["count"]) == expected
        assert conn.execute(
            """SELECT status FROM shadow_relation_fact_memberships
               WHERE run_id=%s""",
            (run_id,),
        ).fetchone()["status"] == "ELIGIBLE"
        product_after = {
            table: int(conn.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE project_id=%s",
                (project_id,),
            ).fetchone()["count"])
            for table in product_before
        }
    assert product_after == product_before


def test_grounded_relation_v2_database_api_is_explicitly_isolated() -> None:
    source_names = {
        "start_grounded_relation_run",
        "persist_grounded_relation_opportunities",
        "record_grounded_relation_classification_attempt",
        "persist_grounded_relation_classifications",
        "persist_grounded_relation_proposed_hypotheses",
        "persist_grounded_relation_validations",
        "persist_grounded_relation_verification_attempts",
        "persist_grounded_relation_normalization",
        "finish_grounded_relation_run",
    }

    assert source_names <= set(dir(Database))


def test_grounded_relation_v2_persistence_writes_v2_audit_source_labels() -> None:
    """v2 rows must not depend on compatibility defaults that look like P1."""

    persistence_source = "\n".join(
        inspect.getsource(method)
        for method in (
            Database.persist_grounded_relation_opportunities,
            Database.persist_grounded_relation_classifications,
            Database.persist_grounded_relation_proposed_hypotheses,
            Database.persist_grounded_relation_normalization,
        )
    )

    assert '"v2_pair_planner"' in persistence_source
    assert '"v2_classifier"' in persistence_source
    assert '"v2_verifier"' in persistence_source
    assert "legacy_pair" not in persistence_source
    assert "legacy_classifier" not in persistence_source
    assert "legacy_evidence" not in persistence_source


@pytest.mark.integration
def test_grounded_relation_v2_failed_run_is_reset_for_same_input_retry() -> None:
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex[:12]
    with database.connection() as conn:
        project_id = int(conn.execute(
            "INSERT INTO projects(project_key,name) VALUES(%s,%s) RETURNING id",
            (f"shadow-v2-retry-{suffix}", f"Shadow v2 retry {suffix}"),
        ).fetchone()["id"])

    content = "Alpha uses Beta."
    run_id = f"shadow-v2-retry-run-{suffix}"
    fingerprint = f"shadow-v2-retry-fingerprint-{suffix}"
    plan = plan_pair_opportunities(
        project_id=project_id,
        document_revision_id=f"revision-{suffix}",
        chunk_key=f"chunk-{suffix}",
        content=content,
        mentions=[
            {
                "occurrence_key": "occ-alpha",
                "mention_id": "mention-alpha",
                "surface": "Alpha",
                "mention_start": 0,
                "mention_end": 5,
                "canonical_id": "entity-alpha",
                "assignment_version": "assignment-v1",
                "type": "Organization",
            },
            {
                "occurrence_key": "occ-beta",
                "mention_id": "mention-beta",
                "surface": "Beta",
                "mention_start": 11,
                "mention_end": 15,
                "canonical_id": "entity-beta",
                "assignment_version": "assignment-v1",
                "type": "Product",
            },
        ],
        logical_schema=_schema(),
        run_id=run_id,
        candidate_policy_version="candidate-v2",
    )
    pair = plan.admitted[0]
    decisions = parse_classifier_response(
        plan.admitted,
        [{
            "pair_id": pair.pair_identity,
            "decision": HYPOTHESES,
            "hypotheses": [{
                "subject_occurrence_id": "occ-alpha",
                "predicate": "uses",
                "object_occurrence_id": "occ-beta",
                "evidence_spans": [{"start": 0, "end": len(content)}],
                "classifier_confidence": 0.9,
            }],
        }],
        content=content,
        logical_schema=_schema(),
        classifier_model_revision="classifier-r1",
    )
    hypothesis = decisions[0].hypotheses[0]

    database.start_grounded_relation_run({
        "id": run_id,
        "project_id": project_id,
        "input_fingerprint": fingerprint,
        "fingerprint_components": {"test": suffix},
        "source_generation": 1,
    })
    database.persist_grounded_relation_opportunities(run_id, plan.opportunities)
    database.record_grounded_relation_classification_attempt(
        run_id,
        pair.pair_identity,
        {
            "attempt_no": 1,
            "status": "ACCEPTED",
            "request_hash": "request-hash",
            "response_hash": "response-hash",
            "classifier_prompt_version": hypothesis.classifier_prompt_version,
            "classifier_schema_version": hypothesis.classifier_schema_version,
            "classifier_parser_version": hypothesis.classifier_parser_version,
            "classifier_model_revision": hypothesis.classifier_model_revision,
            "model_name": "test-model",
            "request_payload": {"stage": "classifier"},
            "raw_response_text": "{\"decisions\": []}",
            "parsed_response": {"decisions": []},
            "token_usage": {},
            "error_type": "",
            "independent_call": True,
        },
    )
    database.persist_grounded_relation_classifications(run_id, decisions)
    database.finish_grounded_relation_run(
        run_id,
        status="failed",
        metrics={"runtime_failure": "SyntheticFailure"},
        input_fingerprint=fingerprint,
        error_message="synthetic failure",
    )

    with database.connection() as conn:
        assert conn.execute(
            """SELECT COUNT(*) AS count FROM shadow_relation_pair_opportunities
               WHERE run_id=%s""",
            (run_id,),
        ).fetchone()["count"] == 1
        assert conn.execute(
            """SELECT COUNT(*) AS count
               FROM shadow_relation_classification_attempts attempt
               JOIN shadow_relation_pair_opportunities pair
                 ON pair.id=attempt.pair_opportunity_id
               WHERE pair.run_id=%s""",
            (run_id,),
        ).fetchone()["count"] == 1
        assert conn.execute(
            """SELECT COUNT(*) AS count
               FROM shadow_relation_hypotheses hypothesis
               JOIN shadow_relation_pair_opportunities pair
                 ON pair.id=hypothesis.pair_opportunity_id
               WHERE pair.run_id=%s""",
            (run_id,),
        ).fetchone()["count"] == 1

    restarted = database.start_grounded_relation_run({
        "id": run_id,
        "project_id": project_id,
        "input_fingerprint": fingerprint,
        "fingerprint_components": {"test": suffix},
        "source_generation": 1,
    })
    assert restarted["status"] == "planning"
    assert restarted["completed_at"] is None
    assert restarted["error_message"] == ""

    with database.connection() as conn:
        for table in (
            "shadow_relation_pair_opportunities",
            "shadow_relation_classification_attempts",
            "shadow_relation_hypotheses",
        ):
            if table == "shadow_relation_pair_opportunities":
                count = conn.execute(
                    f"SELECT COUNT(*) AS count FROM {table} WHERE run_id=%s",
                    (run_id,),
                ).fetchone()["count"]
            else:
                count = conn.execute(
                    f"""SELECT COUNT(*) AS count FROM {table} item
                        JOIN shadow_relation_pair_opportunities pair
                          ON pair.id=item.pair_opportunity_id
                        WHERE pair.run_id=%s""",
                    (run_id,),
                ).fetchone()["count"]
            assert count == 0
