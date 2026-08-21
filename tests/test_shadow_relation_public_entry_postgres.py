from __future__ import annotations

from concurrent.futures import Future
import hashlib
import json
import os
from types import SimpleNamespace
import uuid

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row

from knowledge_forge import relation_shadow_v2 as v2
from knowledge_forge import relation_shadow_runtime as relation_runtime
from knowledge_forge.construction_strategies import KNOWLEDGE_FORGE_STRATEGY
from knowledge_forge.database import Database
from knowledge_forge.relation_shadow_runtime import ShadowRelationV2Runtime
from knowledge_forge.service import KnowledgeForgeService
from knowledge_forge.shadow_relation_replay import (
    PersistedShadowModelReplayGateway,
    assert_code_only_replay_compatible,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


class _ImmediateExecutor:
    def submit(self, callback, *args):
        future = Future()
        try:
            future.set_result(callback(*args))
        except BaseException as exc:  # pragma: no cover - worker owns failures
            future.set_exception(exc)
        return future

    @staticmethod
    def shutdown(*_args, **_kwargs):
        return None


class _ScriptedV2Gateway:
    def __init__(self):
        self.source_inventory_calls = 0
        self.source_binding_calls = 0
        self.source_role_audit_calls = 0
        self.publication_audit_calls = 0
        self.classifier_calls = 0
        self.role_probe_calls = 0
        self.role_alignment_calls = 0
        self.semantic_contrast_calls = 0
        self.exact_argument_calls = 0
        self.verifier_calls = 0

    @staticmethod
    def effective_model_context(_model):
        return SimpleNamespace(digest="sha256:public-entry-v2-fixture")

    def json_chat(self, **kwargs):
        payload = json.loads(kwargs["prompt"])
        if (
            isinstance(payload.get("mentions"), list)
            and isinstance(payload.get("source"), str)
            and payload.get("context_id")
        ):
            self.source_inventory_calls += 1
            mentions = list(payload["mentions"])
            return {
                "resolved_references": [],
                "facts": [{
                    "subject_id": str(mentions[0]["id"]),
                    "predicate": "acts on",
                    "object_id": str(mentions[1]["id"]),
                    "evidence": str(payload["source"]),
                }],
            }
        if (
            isinstance(payload.get("frames"), list)
            and isinstance(payload.get("predicates"), list)
        ):
            self.source_binding_calls += 1
            return {
                "coverage_complete": True,
                "bindings": [{
                    "frame_index": 0,
                    "predicate": "acts_on",
                    "subject_index": 0,
                    "object_index": 1,
                }],
            }
        if (
            isinstance(payload.get("candidate"), dict)
            and isinstance(payload.get("required_roles"), dict)
            and isinstance(payload.get("source"), str)
        ):
            self.source_role_audit_calls += 1
            return {"projection_status": "ALIGNED"}
        if (
            isinstance(payload.get("candidates"), list)
            and "atomic-publication-audit" in str(
                payload.get("contract_version") or ""
            )
        ):
            self.publication_audit_calls += 1
            return {
                str(candidate["candidate_id"]): "ALIGNED"
                for candidate in payload["candidates"]
            }
        if "proposer" in str(kwargs["system"]):
            return {
                "proposals": [],
                "missing_endpoint_mentions": [],
                "coverage": {"complete": True, "uncovered_cues": []},
            }
        if "classifier" in str(kwargs["system"]):
            self.classifier_calls += 1
            return {
                "items": [{
                    "pair_id": pair["pair_id"],
                    "decision": v2.HYPOTHESES,
                    "hypotheses": [{
                        "subject_occurrence_id": pair["left_occurrence_id"],
                        "predicate": "acts_on",
                        "object_occurrence_id": pair["right_occurrence_id"],
                        "evidence_spans": [{
                            "start": pair["context_span"][0],
                            "end": pair["context_span"][1],
                        }],
                        "classifier_confidence": 0.9,
                    }],
                } for pair in payload["pairs"]]
            }
        if "semantic contrast judge" in str(kwargs["system"]):
            self.semantic_contrast_calls += 1
            return {"semantic_contrast": "DEFINITION"}
        if "directional textual-entailment verifier" in str(kwargs["system"]):
            self.exact_argument_calls += 1
            entailed = self.exact_argument_calls % 2 == 1
            return {
                "source_role": entailed,
                "target_role": entailed,
                "direct_relation": entailed,
            }
        if "role alignment judge" in str(kwargs["system"]):
            self.role_alignment_calls += 1
            return {
                "subject_role": "ALIGNED",
                "object_role": "ALIGNED",
            }
        if "role analyst" in str(kwargs["system"]):
            self.role_probe_calls += 1
            return {
                "endpoint_a_complete_argument": True,
                "endpoint_b_complete_argument": True,
                "direct_relation_present": True,
                "subject_role": "scripted source endpoint role",
                "object_role": "scripted target endpoint role",
                "expressed_relation": "Alpha directly acts on Beta.",
            }
        if isinstance(payload.get("claims"), list):
            self.verifier_calls += 1
            return {
                "decisions": [{
                    "hypothesis_id": claim["hypothesis_id"],
                    "verdict": v2.SUPPORTED,
                    "reason_code": "SCRIPTED_EXACT_EVIDENCE",
                    "forward_entailment": "ENTAILED",
                    "reverse_entailment": "NOT_ENTAILED",
                    "subject_role_supported": True,
                    "object_role_supported": True,
                    "supported_evidence_spans": claim[
                        "classifier_evidence_spans"
                    ],
                } for claim in payload["claims"]],
            }
        self.verifier_calls += 1
        return {
            "hypothesis_id": payload["hypothesis_id"],
            "verdict": v2.SUPPORTED,
            "reason_code": "SCRIPTED_EXACT_EVIDENCE",
            "forward_entailment": "ENTAILED",
            "reverse_entailment": "NOT_ENTAILED",
            "subject_role_supported": True,
            "object_role_supported": True,
            "supported_evidence_spans": payload["classifier_evidence_spans"],
        }


class _NoopStrategyGraph:
    def __init__(self):
        self.calls = 0

    def replace_strategy_snapshot(
        self,
        _project,
        documents,
        chunks_by_document,
        shadow,
        *,
        strategy_key,
        mutation,
        **_kwargs,
    ):
        assert strategy_key == KNOWLEDGE_FORGE_STRATEGY
        assert mutation.namespace_id
        self.calls += 1
        return {
            "documents": len(documents),
            "chunks": sum(len(value) for value in chunks_by_document.values()),
            "entities": len(shadow.get("canonicals") or []),
            "mentions": len(shadow.get("mentions") or []),
            "relations": len(shadow.get("relation_endpoints") or []),
        }


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)

    class DisposableConfig:
        postgres = params
        postgres_maintenance_db = params["dbname"]

    return Database(DisposableConfig)


def _seed_runtime(database: Database, suffix: str) -> dict[str, object]:
    project = database.create_project(
        f"Shadow v2 public {suffix}", "", "scripted-v2", "scripted-embed",
    )
    project_id = int(project["id"])
    database.ensure_project_strategy_graphs(project_id)
    content = "Alpha acts on Beta."
    with database.connection() as conn:
        conn.execute(
            """INSERT INTO graph_schemas(
               project_id,version,status,schema_json,source_document_ids,
               created_by,applied_at)
               VALUES(%s,1,'active',%s::jsonb,'{}','test',NOW())""",
            (
                project_id,
                json.dumps({
                    "version": "public-entry-schema-v1",
                    "entity_types": [
                        {"name": "Organization"},
                        {"name": "Product"},
                    ],
                    "relation_types": logical_schema(
                        predicate_policy(
                            "acts_on",
                            domain=["Organization"],
                            range_=["Product"],
                        ),
                    )["relation_types"],
                }),
            ),
        )
        document_id = int(conn.execute(
            """INSERT INTO documents(
               project_id,doc_hash,original_filename,stored_filename,storage_path,
               local_path,file_size_bytes,status,indexed_at)
               VALUES(%s,%s,%s,%s,%s,%s,%s,'indexed',NOW()) RETURNING id""",
            (
                project_id,
                f"document-hash-{suffix}",
                f"{suffix}.txt",
                f"{suffix}.txt",
                f"/isolated/{suffix}",
                f"/isolated/{suffix}",
                len(content),
            ),
        ).fetchone()["id"])
        revision_id = f"revision-{suffix}"
        conn.execute(
            """INSERT INTO document_content_revisions(
               revision_id,project_id,document_id,ingestion_run_id,content_version,
               content_hash,source_hash,config_hash,policy_hash,manifest_hash,
               manifest,status,activated_at)
               VALUES(%s,%s,%s,%s,1,%s,%s,'config','policy',%s,
                      '{}'::jsonb,'active',NOW())""",
            (
                revision_id,
                project_id,
                document_id,
                f"ingestion-{suffix}",
                f"content-{suffix}",
                f"source-{suffix}",
                f"manifest-{suffix}",
            ),
        )
        conn.execute(
            "UPDATE documents SET active_revision_id=%s WHERE id=%s",
            (revision_id, document_id),
        )
        section_id = int(conn.execute(
            """INSERT INTO sections(
               project_id,document_id,revision_id,section_key,title,level,
               content,content_hash)
               VALUES(%s,%s,%s,'section','Section',1,%s,%s) RETURNING id""",
            (
                project_id,
                document_id,
                revision_id,
                content,
                f"section-hash-{suffix}",
            ),
        ).fetchone()["id"])
        chunk_id = int(conn.execute(
            """INSERT INTO chunks(
               project_id,document_id,revision_id,section_id,chunk_key,chunk_index,
               content,content_hash,metadata)
               VALUES(%s,%s,%s,%s,'chunk',0,%s,%s,'{}'::jsonb) RETURNING id""",
            (
                project_id,
                document_id,
                revision_id,
                section_id,
                content,
                f"chunk-hash-{suffix}",
            ),
        ).fetchone()["id"])

    database.ensure_document_strategy_builds(document_id)
    namespace_id = f"entity-shadow-public-{suffix}"
    database.create_entity_shadow_namespace(
        project_id,
        namespace_id,
        {"mode": "scripted-v2-public-entry"},
        [document_id],
        input_fingerprint=f"entity-input-{suffix}",
        model_name="scripted-v2",
        embedding_model="scripted-embed",
        config_hash="entity-config-v1",
        code_version="entity-fixture-v1",
    )
    alpha_start = content.index("Alpha")
    beta_start = content.index("Beta")
    with database.connection() as conn:
        conn.execute(
            """INSERT INTO entity_shadow_mentions(
               namespace_id,mention_key,project_id,document_id,revision_id,chunk_id,
               surface,normalized_surface,evidence_text,evidence_start,evidence_end,
               confidence,extraction_policy_version)
               VALUES
               (%s,%s,%s,%s,%s,%s,'Alpha','alpha','Alpha',%s,%s,0.99,'fixture'),
               (%s,%s,%s,%s,%s,%s,'Beta','beta','Beta',%s,%s,0.99,'fixture')""",
            (
                namespace_id,
                f"alpha-{suffix}",
                project_id,
                document_id,
                revision_id,
                chunk_id,
                alpha_start,
                alpha_start + len("Alpha"),
                namespace_id,
                f"beta-{suffix}",
                project_id,
                document_id,
                revision_id,
                chunk_id,
                beta_start,
                beta_start + len("Beta"),
            ),
        )
        conn.execute(
            """INSERT INTO entity_shadow_canonicals(
               namespace_id,canonical_key,name,normalized_name,primary_type,source_count)
               VALUES
               (%s,%s,'Alpha','alpha','Organization',1),
               (%s,%s,'Beta','beta','Product',1)""",
            (
                namespace_id,
                f"canonical-alpha-{suffix}",
                namespace_id,
                f"canonical-beta-{suffix}",
            ),
        )
        conn.execute(
            """INSERT INTO entity_shadow_assignments(
               namespace_id,mention_key,canonical_key,document_id,chunk_id,
               revision_id,entity_type,policy_version)
               VALUES
               (%s,%s,%s,%s,%s,%s,'Organization','assignment-v1'),
               (%s,%s,%s,%s,%s,%s,'Product','assignment-v1')""",
            (
                namespace_id,
                f"alpha-{suffix}",
                f"canonical-alpha-{suffix}",
                document_id,
                chunk_id,
                revision_id,
                namespace_id,
                f"beta-{suffix}",
                f"canonical-beta-{suffix}",
                document_id,
                chunk_id,
                revision_id,
            ),
        )
        conn.execute(
            """UPDATE entity_shadow_namespaces
               SET status='ready',snapshot_complete=TRUE,
                   expected_work_items=0,completed_work_items=0,failed_work_items=0,
                   artifact_hash=%s,metrics='{}'::jsonb,completed_at=NOW()
               WHERE id=%s""",
            (f"entity-artifact-{suffix}", namespace_id),
        )
    return {
        "project_id": project_id,
        "document_id": document_id,
        "namespace_id": namespace_id,
    }


def _formal_state(database: Database, project_id: int) -> dict[str, object]:
    with database.connection() as conn:
        project = conn.execute(
            """SELECT graph_generation,graph_ready_generation,graph_status,
                      graph_policy_hash,active_policy_hash
               FROM projects WHERE id=%s""",
            (project_id,),
        ).fetchone()
        stable = conn.execute(
            """SELECT generation,ready_generation,status,policy_summary,stats
               FROM project_strategy_graphs
               WHERE project_id=%s AND strategy_key='stable'""",
            (project_id,),
        ).fetchone()
        counts = conn.execute(
            """SELECT
               (SELECT COUNT(*) FROM relation_assertions
                 WHERE project_id=%s AND is_deleted=FALSE) AS assertions,
               (SELECT COUNT(*) FROM canonical_relation_projections
                 WHERE project_id=%s) AS projections,
               (SELECT COUNT(*) FROM policy_snapshots
                 WHERE project_id=%s AND status='active') AS activations""",
            (project_id, project_id, project_id),
        ).fetchone()
    payload = {"project": project, "stable": stable, "counts": counts}
    payload["checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return payload


def _wire_public_service(
    database: Database,
    seeded: dict[str, object],
    gateway: _ScriptedV2Gateway,
):
    service = KnowledgeForgeService(database)
    service.executor.shutdown(wait=True)
    service.cleanup_executor.shutdown(wait=True)
    service.executor = _ImmediateExecutor()
    service.cleanup_executor = _ImmediateExecutor()
    service.ollama = gateway
    graph = _NoopStrategyGraph()
    service.graph = graph
    legacy_calls = {"service": 0, "pipeline": 0}

    def legacy_service_tripwire(*_args, **_kwargs):
        legacy_calls["service"] += 1
        raise AssertionError("legacy _run_relation_shadow was called")

    def legacy_pipeline_tripwire(*_args, **_kwargs):
        legacy_calls["pipeline"] += 1
        raise AssertionError("legacy relation pipeline was called")

    service._run_relation_shadow = legacy_service_tripwire
    service.relation_shadow.prepare_shadow = legacy_pipeline_tripwire
    service._prepare_shared_content = lambda _job_id, document_id, **_kwargs: (
        database.get_document(document_id),
        database.get_project(int(seeded["project_id"])),
        database.list_document_chunks(document_id),
    )
    service.schemas.evaluate_corpus = lambda _project, _documents: (
        database.get_active_schema(int(seeded["project_id"]))
    )

    def build_runtime(project_id, *, progress=None, **_kwargs):
        result = ShadowRelationV2Runtime(database, gateway).run_namespace(
            namespace_id=str(seeded["namespace_id"]),
            project_id=project_id,
            progress=progress,
        )
        with database.connection() as conn:
            conn.execute(
                """UPDATE entity_shadow_namespaces SET
                   metrics=metrics||%s::jsonb,
                   status='ready',snapshot_complete=TRUE,completed_at=NOW()
                   WHERE id=%s""",
                (
                    json.dumps({
                        "relation_v2_run_id": result["run_id"],
                        "relation_v2_fingerprint": result["input_fingerprint"],
                        "relation_v2": result["metrics"],
                    }),
                    seeded["namespace_id"],
                ),
            )
        snapshot = database.load_entity_shadow_snapshot(
            str(seeded["namespace_id"])
        )
        return {"namespace_id": seeded["namespace_id"], **snapshot}

    service.build_grounded_entity_namespace = build_runtime
    return service, graph, legacy_calls


@pytest.mark.integration
def test_public_knowledge_forge_entry_executes_v2_with_zero_legacy_calls(
    monkeypatch,
):
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex
    seeded = _seed_runtime(database, suffix)
    gateway = _ScriptedV2Gateway()
    service, graph, legacy_calls = _wire_public_service(
        database, seeded, gateway,
    )
    stage_calls = {
        "planner": 0,
        "classifier_parser": 0,
        "verifier": 0,
        "normalizer": 0,
    }
    for function_name, counter_name in (
        ("plan_pair_opportunities", "planner"),
        ("parse_classifier_response", "classifier_parser"),
        ("verify_hypothesis_lean", "verifier"),
        ("normalize_verified_hypotheses", "normalizer"),
    ):
        original = getattr(v2, function_name)

        def wrapper(*args, _original=original, _counter=counter_name, **kwargs):
            stage_calls[_counter] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(v2, function_name, wrapper)

    before = _formal_state(database, int(seeded["project_id"]))
    job_id = service.submit_ingestion(
        int(seeded["document_id"]),
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        target_document_ids=[int(seeded["document_id"])],
    )
    after = _formal_state(database, int(seeded["project_id"]))

    assert database.get_job(job_id)["status"] == "completed"
    assert legacy_calls == {"service": 0, "pipeline": 0}
    assert stage_calls["planner"] >= 1
    assert stage_calls["classifier_parser"] == 0
    assert stage_calls["verifier"] >= 1
    assert stage_calls["normalizer"] == 1
    assert gateway.source_inventory_calls >= 1
    assert gateway.source_binding_calls == 1
    assert gateway.classifier_calls == 0
    assert gateway.role_probe_calls == 0
    assert gateway.role_alignment_calls == 0
    assert gateway.exact_argument_calls == 0
    assert gateway.verifier_calls == 1
    assert graph.calls == 1
    assert after == before

    with database.connection() as conn:
        run = conn.execute(
            """SELECT * FROM shadow_relation_v2_runs
               WHERE project_id=%s ORDER BY created_at DESC LIMIT 1""",
            (seeded["project_id"],),
        ).fetchone()
        pair_count = int(conn.execute(
            """SELECT COUNT(*) AS count FROM shadow_relation_pair_opportunities
               WHERE run_id=%s""",
            (run["id"],),
        ).fetchone()["count"])
        hypothesis_count = int(conn.execute(
            """SELECT COUNT(*) AS count
               FROM shadow_relation_hypotheses hypothesis
               JOIN shadow_relation_pair_opportunities pair
                 ON pair.id=hypothesis.pair_opportunity_id
               WHERE pair.run_id=%s""",
            (run["id"],),
        ).fetchone()["count"])
        verifier_counts = conn.execute(
            """WITH terminal AS (
                 SELECT DISTINCT ON (attempt.hypothesis_id)
                        attempt.hypothesis_id,attempt.verdict,
                        attempt.independent_call
                 FROM shadow_relation_verification_attempts attempt
                 JOIN shadow_relation_hypotheses hypothesis
                   ON hypothesis.id=attempt.hypothesis_id
                 JOIN shadow_relation_pair_opportunities pair
                   ON pair.id=hypothesis.pair_opportunity_id
                 WHERE pair.run_id=%s
                 ORDER BY attempt.hypothesis_id,attempt.attempt_no DESC
               )
               SELECT COUNT(*) AS total,
                      COUNT(*) FILTER(
                        WHERE verdict='SUPPORTED' AND independent_call
                      ) AS supported
               FROM terminal""",
            (run["id"],),
        ).fetchone()
        facts = conn.execute(
            """SELECT COUNT(DISTINCT membership.fact_id) AS total,
                      COUNT(DISTINCT membership.fact_id) FILTER(
                        WHERE attempt.verdict='SUPPORTED'
                          AND attempt.independent_call
                      ) AS supported
               FROM shadow_relation_fact_memberships membership
               LEFT JOIN shadow_relation_fact_evidence evidence
                 ON evidence.run_id=membership.run_id
                AND evidence.fact_id=membership.fact_id
               LEFT JOIN shadow_relation_verification_attempts attempt
                 ON attempt.id=evidence.verification_attempt_id
               WHERE membership.run_id=%s""",
            (run["id"],),
        ).fetchone()
        endpoints = int(conn.execute(
            """SELECT COUNT(*) AS count
               FROM entity_shadow_relation_endpoints WHERE namespace_id=%s""",
            (seeded["namespace_id"],),
        ).fetchone()["count"])
    assert run["status"] == "ready"
    assert pair_count == 1
    assert hypothesis_count == 2
    assert verifier_counts == {"total": 2, "supported": 2}
    assert facts == {"total": 2, "supported": 2}
    assert endpoints == 1
    assert run["metrics"]["facts_predicate_publication_suppressed"] == 1

    before_retry_counts = (
        pair_count,
        hypothesis_count,
        int(verifier_counts["total"]),
        int(facts["total"]),
        endpoints,
    )
    retry_job_id = service.submit_ingestion(
        int(seeded["document_id"]),
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
        target_document_ids=[int(seeded["document_id"])],
    )
    assert database.get_job(retry_job_id)["status"] == "completed"
    with database.connection() as conn:
        after_retry_counts = (
            int(conn.execute(
                "SELECT COUNT(*) AS count FROM shadow_relation_pair_opportunities "
                "WHERE run_id=%s",
                (run["id"],),
            ).fetchone()["count"]),
            int(conn.execute(
                """SELECT COUNT(*) AS count
                   FROM shadow_relation_hypotheses hypothesis
                   JOIN shadow_relation_pair_opportunities pair
                     ON pair.id=hypothesis.pair_opportunity_id
                   WHERE pair.run_id=%s""",
                (run["id"],),
            ).fetchone()["count"]),
            int(conn.execute(
                """SELECT COUNT(*) AS count
                   FROM shadow_relation_verification_attempts attempt
                   JOIN shadow_relation_hypotheses hypothesis
                     ON hypothesis.id=attempt.hypothesis_id
                   JOIN shadow_relation_pair_opportunities pair
                     ON pair.id=hypothesis.pair_opportunity_id
                   WHERE pair.run_id=%s""",
                (run["id"],),
            ).fetchone()["count"]),
            int(conn.execute(
                """SELECT COUNT(*) AS count
                   FROM shadow_relation_fact_memberships WHERE run_id=%s""",
                (run["id"],),
            ).fetchone()["count"]),
            int(conn.execute(
                """SELECT COUNT(*) AS count
                   FROM entity_shadow_relation_endpoints WHERE namespace_id=%s""",
                (seeded["namespace_id"],),
            ).fetchone()["count"]),
        )
    assert after_retry_counts == before_retry_counts
    assert legacy_calls == {"service": 0, "pipeline": 0}
    assert _formal_state(database, int(seeded["project_id"])) == before


@pytest.mark.integration
def test_public_knowledge_forge_v2_failure_never_falls_back_or_materializes(
    monkeypatch,
):
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex
    seeded = _seed_runtime(database, suffix)
    gateway = _ScriptedV2Gateway()
    service, graph, legacy_calls = _wire_public_service(
        database, seeded, gateway,
    )

    def fail_normalization(*_args, **_kwargs):
        raise RuntimeError("scripted mandatory normalization persistence failure")

    monkeypatch.setattr(
        database,
        "persist_shadow_relation_v2_normalization",
        fail_normalization,
    )
    before = _formal_state(database, int(seeded["project_id"]))

    job_id = service.submit_ingestion(
        int(seeded["document_id"]),
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
    )

    job = database.get_job(job_id)
    assert job["status"] == "failed"
    assert "禁止回退旧流程" in str(job["error_message"])
    assert legacy_calls == {"service": 0, "pipeline": 0}
    assert graph.calls == 0
    assert _formal_state(database, int(seeded["project_id"])) == before
    with database.connection() as conn:
        run = conn.execute(
            """SELECT status FROM shadow_relation_v2_runs
               WHERE project_id=%s ORDER BY created_at DESC LIMIT 1""",
            (seeded["project_id"],),
        ).fetchone()
        endpoints = int(conn.execute(
            """SELECT COUNT(*) AS count
               FROM entity_shadow_relation_endpoints WHERE namespace_id=%s""",
            (seeded["namespace_id"],),
        ).fetchone()["count"])
    assert run["status"] == "failed"
    assert endpoints == 0


@pytest.mark.integration
def test_public_knowledge_forge_stale_fingerprint_never_materializes_or_falls_back():
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex
    seeded = _seed_runtime(database, suffix)
    gateway = _ScriptedV2Gateway()
    service, graph, legacy_calls = _wire_public_service(
        database, seeded, gateway,
    )
    normal_build = service.build_grounded_entity_namespace

    def stale_build(project_id, **kwargs):
        original_get_project = database.get_project
        reads = {"count": 0}

        def stale_get_project(value):
            project = original_get_project(value)
            reads["count"] += 1
            if project and reads["count"] > 1:
                return {
                    **project,
                    "graph_generation": int(
                        project.get("graph_generation") or 0
                    ) + 1,
                }
            return project

        database.get_project = stale_get_project
        try:
            return normal_build(project_id, **kwargs)
        finally:
            database.get_project = original_get_project

    service.build_grounded_entity_namespace = stale_build
    before = _formal_state(database, int(seeded["project_id"]))

    job_id = service.submit_ingestion(
        int(seeded["document_id"]),
        strategy_key=KNOWLEDGE_FORGE_STRATEGY,
    )

    job = database.get_job(job_id)
    assert job["status"] == "failed"
    assert "fingerprint stale" in str(job["error_message"])
    assert legacy_calls == {"service": 0, "pipeline": 0}
    assert graph.calls == 0
    assert _formal_state(database, int(seeded["project_id"])) == before
    with database.connection() as conn:
        run = conn.execute(
            """SELECT status FROM shadow_relation_v2_runs
               WHERE project_id=%s ORDER BY created_at DESC LIMIT 1""",
            (seeded["project_id"],),
        ).fetchone()
        endpoints = int(conn.execute(
            """SELECT COUNT(*) AS count
               FROM entity_shadow_relation_endpoints WHERE namespace_id=%s""",
            (seeded["namespace_id"],),
        ).fetchone()["count"])
    assert run["status"] == "stale"
    assert endpoints == 0


@pytest.mark.integration
def test_code_only_stale_run_replays_every_persisted_call_without_model_io(
    monkeypatch,
):
    dsn = os.getenv("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("KNOWLEDGE_FORGE_TEST_POSTGRES_DSN is not configured")
    database = _database(dsn)
    database.bootstrap()
    suffix = uuid.uuid4().hex
    seeded = _seed_runtime(database, suffix)
    source_code = "source-process-code"
    current_code = "current-process-code"
    monkeypatch.setattr(
        relation_runtime, "LOADED_RELATION_CODE_DIGEST", source_code,
    )

    class DriftAfterPublicationAudit(_ScriptedV2Gateway):
        def json_chat(self, **kwargs):
            response = super().json_chat(**kwargs)
            payload = json.loads(kwargs["prompt"])
            if (
                isinstance(payload.get("candidates"), list)
                and "atomic-publication-audit" in str(
                    payload.get("contract_version") or ""
                )
            ):
                monkeypatch.setattr(
                    relation_runtime,
                    "LOADED_RELATION_CODE_DIGEST",
                    current_code,
                )
            return response

    source_gateway = DriftAfterPublicationAudit()
    with pytest.raises(v2.StaleRelationSnapshotError):
        ShadowRelationV2Runtime(database, source_gateway).run_namespace(
            namespace_id=str(seeded["namespace_id"]),
            project_id=int(seeded["project_id"]),
        )
    with database.connection() as conn:
        source_run = conn.execute(
            """SELECT * FROM shadow_relation_v2_runs
               WHERE project_id=%s ORDER BY created_at DESC LIMIT 1""",
            (seeded["project_id"],),
        ).fetchone()
        assert source_run["status"] == "stale"
        assert conn.execute(
            """SELECT COUNT(*) AS count
               FROM entity_shadow_relation_endpoints WHERE namespace_id=%s""",
            (seeded["namespace_id"],),
        ).fetchone()["count"] == 0

    replay = PersistedShadowModelReplayGateway(
        database,
        str(source_run["id"]),
        runtime_model_gateway=_ScriptedV2Gateway(),
    )
    runtime = ShadowRelationV2Runtime(database, replay)
    current = runtime._context(
        int(seeded["project_id"]), str(seeded["namespace_id"]),
    )["fingerprint"]
    compatibility = assert_code_only_replay_compatible(
        source_run, current,
    )
    assert compatibility["changed_components"] == [
        "materialization_root.code_tree",
    ]

    recovered = runtime.run_namespace(
        namespace_id=str(seeded["namespace_id"]),
        project_id=int(seeded["project_id"]),
    )

    assert recovered["metrics"]["replay"]["live_model_calls"] == 0
    assert recovered["metrics"]["replay"]["misses"] == 0
    assert recovered["metrics"]["replay"]["remaining_calls"] == 0
    assert recovered["metrics"]["verifier_supported"] == int(
        source_run["metrics"]["verifier_supported"]
    )
    assert len(recovered["endpoint_rows"]) == int(
        source_run["metrics"]["facts_materialization_eligible"]
    ) == 1
    with database.connection() as conn:
        assert conn.execute(
            """SELECT COUNT(*) AS count
               FROM entity_shadow_relation_endpoints WHERE namespace_id=%s""",
            (seeded["namespace_id"],),
        ).fetchone()["count"] == 1
        current_run = conn.execute(
            "SELECT status,metrics FROM shadow_relation_v2_runs WHERE id=%s",
            (recovered["run_id"],),
        ).fetchone()
    assert current_run["status"] == "ready"
    assert current_run["metrics"]["replay"]["source_run_id"] == source_run["id"]
