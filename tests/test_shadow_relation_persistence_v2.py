from __future__ import annotations

import os
import uuid

import pytest
from psycopg.conninfo import conninfo_to_dict

from knowledge_forge.database import Database
from knowledge_forge.relation_shadow_v2 import (
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


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _schema() -> dict:
    return {
        "version": "schema-v1",
        "relation_types": [{
            "name": "uses",
            "domain": ["Organization"],
            "range": ["Product"],
            "canonical_forward": "uses",
            "publishable": True,
        }],
    }


@pytest.mark.integration
def test_shadow_relation_v2_persistence_is_idempotent_and_product_isolated() -> None:
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
        verifier=lambda _prompt, _schema: {
            "hypothesis_id": hypothesis.hypothesis_id,
            "verdict": SUPPORTED,
            "reason_code": "EXACT_TRIPLE_SUPPORTED",
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

    database.start_shadow_relation_v2_run({
        "id": run_id,
        "project_id": project_id,
        "input_fingerprint": fingerprint,
        "fingerprint_components": {"test": suffix},
        "source_generation": 1,
    })
    database.persist_shadow_relation_v2_opportunities(run_id, plan.opportunities)
    database.record_shadow_relation_v2_classification_attempt(
        run_id,
        pair.pair_identity,
        {
            "attempt_no": 1,
            "status": "accepted",
            "request_hash": "request-hash",
            "response_hash": "response-hash",
            "classifier_prompt_version": hypothesis.classifier_prompt_version,
            "classifier_schema_version": hypothesis.classifier_schema_version,
            "classifier_parser_version": hypothesis.classifier_parser_version,
            "classifier_model_revision": hypothesis.classifier_model_revision,
        },
    )
    database.persist_shadow_relation_v2_classifications(run_id, decisions)
    database.persist_shadow_relation_v2_validations(run_id, [validation])
    database.persist_shadow_relation_v2_verification_attempts(run_id, [attempt])
    database.persist_shadow_relation_v2_normalization(
        run_id,
        normalization,
        input_fingerprint=fingerprint,
        materialization_decisions=materialization,
    )
    database.finish_shadow_relation_v2_run(
        run_id,
        status="ready",
        metrics={"pair_opportunities_total": 1, "facts_materialization_eligible": 1},
        input_fingerprint=fingerprint,
    )

    # Every persistence stage is retry-safe and may be replayed after a worker restart.
    database.start_shadow_relation_v2_run({
        "id": run_id,
        "project_id": project_id,
        "input_fingerprint": fingerprint,
        "fingerprint_components": {"test": suffix},
        "source_generation": 1,
    })
    database.persist_shadow_relation_v2_opportunities(run_id, plan.opportunities)
    database.persist_shadow_relation_v2_classifications(run_id, decisions)
    database.persist_shadow_relation_v2_validations(run_id, [validation])
    database.persist_shadow_relation_v2_verification_attempts(run_id, [attempt])
    database.persist_shadow_relation_v2_normalization(
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


def test_shadow_relation_v2_database_api_is_explicitly_isolated() -> None:
    source_names = {
        "start_shadow_relation_v2_run",
        "persist_shadow_relation_v2_opportunities",
        "record_shadow_relation_v2_classification_attempt",
        "persist_shadow_relation_v2_classifications",
        "persist_shadow_relation_v2_validations",
        "persist_shadow_relation_v2_verification_attempts",
        "persist_shadow_relation_v2_normalization",
        "finish_shadow_relation_v2_run",
    }

    assert source_names <= set(dir(Database))
