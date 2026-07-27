#!/usr/bin/env python3
"""Run native relation v2 through the public enhanced entry in an isolated clone."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge import relation_shadow_v2 as v2
from knowledge_forge.config import Config
from knowledge_forge.construction_strategies import ENHANCED_SHADOW_STRATEGY
from knowledge_forge.database import Database
from knowledge_forge.quality import canonical_digest
from knowledge_forge.relation_shadow_runtime import ShadowRelationV2Runtime
from knowledge_forge.service import KnowledgeForgeService


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)
    database_name = str(params.get("dbname") or "")
    if not database_name.startswith("knowledge_forge_shadow_v2_pilot_"):
        raise RuntimeError(
            "native pilot requires knowledge_forge_shadow_v2_pilot_*"
        )

    class PilotConfig:
        postgres = params
        postgres_maintenance_db = database_name

    return Database(PilotConfig)


class _ImmediateExecutor:
    def submit(self, callback, *args):
        future = Future()
        try:
            future.set_result(callback(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    @staticmethod
    def shutdown(*_args, **_kwargs):
        return None


class _CountingGateway:
    def __init__(self, gateway: Any):
        self.gateway = gateway
        self.classifier_calls = 0
        self.verifier_calls = 0

    def effective_model_context(self, model: str):
        return self.gateway.effective_model_context(model)

    def json_chat(self, **kwargs):
        system = str(kwargs.get("system") or "")
        if "classifier" in system:
            self.classifier_calls += 1
        elif "verifier" in system:
            self.verifier_calls += 1
        return self.gateway.json_chat(**kwargs)


class _NoWriteGraph:
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
    ):
        if strategy_key != ENHANCED_SHADOW_STRATEGY:
            raise AssertionError("native pilot escaped enhanced strategy")
        if not mutation.namespace_id:
            raise AssertionError("strategy namespace is missing")
        self.calls += 1
        return {
            "documents": len(documents),
            "chunks": sum(len(value) for value in chunks_by_document.values()),
            "entities": len(shadow.get("canonicals") or []),
            "mentions": len(shadow.get("mentions") or []),
            "relations": len(shadow.get("relation_endpoints") or []),
        }


def _copy_namespace_table(
    conn: Any,
    table: str,
    source_namespace: str,
    target_namespace: str,
) -> int:
    columns = [
        str(row["column_name"])
        for row in conn.execute(
            """SELECT column_name
               FROM information_schema.columns
               WHERE table_schema=current_schema() AND table_name=%s
               ORDER BY ordinal_position""",
            (table,),
        ).fetchall()
    ]
    if not columns or "namespace_id" not in columns:
        raise RuntimeError(f"namespace table contract missing: {table}")
    select_values = [
        sql.Placeholder() if column == "namespace_id"
        else sql.Identifier(column)
        for column in columns
    ]
    query = sql.SQL(
        "INSERT INTO {table} ({columns}) "
        "SELECT {values} FROM {table} WHERE namespace_id=%s"
    ).format(
        table=sql.Identifier(table),
        columns=sql.SQL(",").join(map(sql.Identifier, columns)),
        values=sql.SQL(",").join(select_values),
    )
    params = [
        target_namespace for column in columns if column == "namespace_id"
    ]
    params.append(source_namespace)
    cursor = conn.execute(query, params)
    return max(0, int(cursor.rowcount))


def _clone_entity_snapshot(
    database: Database,
    source_namespace: str,
    target_namespace: str,
    freeze_hash: str,
) -> None:
    source = database.get_entity_shadow_namespace(source_namespace)
    if not source:
        raise RuntimeError("frozen native entity namespace is absent")
    existing = database.get_entity_shadow_namespace(target_namespace)
    if existing:
        if (
            int(existing["project_id"]) != int(source["project_id"])
            or str(existing.get("input_fingerprint") or "")
            != f"native-v2-pilot:{freeze_hash}"
        ):
            raise RuntimeError(
                "native pilot namespace exists with different frozen inputs"
            )
        return
    database.create_entity_shadow_namespace(
        int(source["project_id"]),
        target_namespace,
        dict(source.get("policy_set") or {}),
        [int(value) for value in source.get("document_order") or []],
        input_fingerprint=f"native-v2-pilot:{freeze_hash}",
        model_name=str(source.get("model_name") or ""),
        embedding_model=str(source.get("embedding_model") or ""),
        config_hash=str(source.get("config_hash") or ""),
        code_version="native-v2-public-entry-pilot-v1",
    )
    with database.connection() as conn:
        for table in (
            "entity_shadow_mentions",
            "entity_shadow_canonicals",
            "entity_shadow_assignments",
            "entity_shadow_aliases",
        ):
            _copy_namespace_table(
                conn, table, source_namespace, target_namespace,
            )
        conn.execute(
            """UPDATE entity_shadow_namespaces
               SET status='ready',snapshot_complete=TRUE,
                   expected_work_items=0,completed_work_items=0,
                   failed_work_items=0,error_message='',
                   artifact_hash=%s,metrics=%s::jsonb,completed_at=NOW()
               WHERE id=%s""",
            (
                f"native-v2-pilot:{freeze_hash}",
                json.dumps({
                    "pilot_clone_of": source_namespace,
                    "freeze_payload_sha256": freeze_hash,
                }),
                target_namespace,
            ),
        )


def _formal_state(database: Database, project_id: int) -> dict[str, Any]:
    with database.connection() as conn:
        project = dict(conn.execute(
            """SELECT graph_generation,graph_ready_generation,graph_status,
                      graph_policy_hash,active_policy_hash
               FROM projects WHERE id=%s""",
            (project_id,),
        ).fetchone())
        stable = conn.execute(
            """SELECT generation,ready_generation,status,policy_summary,stats
               FROM project_strategy_graphs
               WHERE project_id=%s AND strategy_key='stable'""",
            (project_id,),
        ).fetchone()
        counts = dict(conn.execute(
            """SELECT
               (SELECT COUNT(*) FROM relation_assertions
                 WHERE project_id=%s AND is_deleted=FALSE) AS assertions,
               (SELECT COUNT(*) FROM canonical_relation_projections
                 WHERE project_id=%s) AS projections,
               (SELECT COUNT(*) FROM policy_snapshots
                 WHERE project_id=%s AND status='active') AS activations""",
            (project_id, project_id, project_id),
        ).fetchone())
    payload = {
        "project": project,
        "stable": dict(stable or {}),
        "counts": counts,
    }
    payload["checksum"] = canonical_digest(payload)
    return payload


def _preflight(runtime: ShadowRelationV2Runtime, project_id: int, namespace_id: str):
    context = runtime._context(project_id, namespace_id)
    mentions_by_chunk: dict[int, list[dict[str, Any]]] = {}
    for mention in context["mentions"]:
        assignment = context["assignment_by_mention"][
            str(mention["mention_key"])
        ]
        mentions_by_chunk.setdefault(int(mention["chunk_id"]), []).append({
            "occurrence_key": str(mention["mention_key"]),
            "mention_id": str(mention["mention_key"]),
            "surface": str(mention["surface"]),
            "mention_start": int(mention["evidence_start"]),
            "mention_end": int(mention["evidence_end"]),
            "canonical_id": str(assignment["canonical_key"]),
            "assignment_version": str(
                assignment.get("policy_version") or namespace_id
            ),
            "type": str(assignment.get("entity_type") or "GenericEntity"),
        })
    statuses: Counter[str] = Counter()
    theoretical = 0
    for chunk in context["chunks"]:
        plan = v2.plan_pair_opportunities(
            project_id=project_id,
            document_revision_id=str(chunk["revision_id"]),
            chunk_key=str(chunk.get("chunk_key") or chunk["id"]),
            content=str(chunk.get("content") or ""),
            mentions=mentions_by_chunk.get(int(chunk["id"]), []),
            logical_schema=context["schema"],
            run_id="native-v2-preflight",
            candidate_policy_version="native-v2-preflight",
            max_pair_opportunities=Config.relation_max_candidates_per_chunk,
            sentence_window=Config.relation_sentence_window,
            max_char_distance=Config.relation_max_char_distance,
        )
        theoretical += plan.theoretical_pairs
        statuses.update(item.status for item in plan.opportunities)
    return {
        "chunks": len(context["chunks"]),
        "mentions": len(context["mentions"]),
        "theoretical_pairs": theoretical,
        "pair_statuses": dict(statuses),
        "classifier_calls_at_frozen_batch_size": int(
            statuses.get(v2.PENDING_CLASSIFICATION, 0)
        ),
    }


def _reconcile(database: Database, run_id: str) -> dict[str, Any]:
    with database.connection() as conn:
        pair_statuses = {
            str(row["status"]): int(row["count"])
            for row in conn.execute(
                """SELECT status,COUNT(*) AS count
                   FROM shadow_relation_pair_opportunities
                   WHERE run_id=%s GROUP BY status ORDER BY status""",
                (run_id,),
            ).fetchall()
        }
        classification = {
            str(row["status"]): int(row["count"])
            for row in conn.execute(
                """SELECT status,COUNT(*) AS count
                   FROM shadow_relation_pair_opportunities
                   WHERE run_id=%s AND status=ANY(%s)
                   GROUP BY status ORDER BY status""",
                (
                    run_id,
                    [v2.NO_RELATION, v2.OTHER_UNSUPPORTED, v2.HYPOTHESES],
                ),
            ).fetchall()
        }
        verifier = {
            str(row["verdict"]): int(row["count"])
            for row in conn.execute(
                """WITH terminal AS (
                     SELECT DISTINCT ON (attempt.hypothesis_id)
                            attempt.hypothesis_id,attempt.verdict
                     FROM shadow_relation_verification_attempts attempt
                     JOIN shadow_relation_hypotheses hypothesis
                       ON hypothesis.id=attempt.hypothesis_id
                     JOIN shadow_relation_pair_opportunities pair
                       ON pair.id=hypothesis.pair_opportunity_id
                     WHERE pair.run_id=%s
                     ORDER BY attempt.hypothesis_id,attempt.attempt_no DESC
                   )
                   SELECT verdict,COUNT(*) AS count FROM terminal
                   GROUP BY verdict ORDER BY verdict""",
                (run_id,),
            ).fetchall()
        }
        facts = dict(conn.execute(
            """SELECT
                 COUNT(DISTINCT membership.fact_id) AS total,
                 COUNT(DISTINCT membership.fact_id) FILTER(
                   WHERE attempt.verdict='SUPPORTED'
                     AND attempt.independent_call=TRUE
                 ) AS supported
               FROM shadow_relation_fact_memberships membership
               LEFT JOIN shadow_relation_fact_evidence evidence
                 ON evidence.run_id=membership.run_id
                AND evidence.fact_id=membership.fact_id
               LEFT JOIN shadow_relation_verification_attempts attempt
                 ON attempt.id=evidence.verification_attempt_id
               WHERE membership.run_id=%s""",
            (run_id,),
        ).fetchone())
    total = int(facts["total"])
    supported = int(facts["supported"])
    return {
        "pair_statuses": pair_statuses,
        "classification_decisions": classification,
        "verifier_terminal": verifier,
        "normalized_facts": total,
        "normalized_facts_with_supported_verification": supported,
        "supported_verification_ratio": (
            1.0 if total == 0 else supported / total
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if conninfo_to_dict(args.dsn)["dbname"] != freeze["source_database"]:
        raise RuntimeError("native pilot DSN does not match frozen isolated database")
    project_id = int(freeze["native_project"]["project"]["id"])
    source_namespace = str(
        freeze["native_entity_snapshot"]["namespace"]["id"]
    )
    freeze_hash = str(freeze["freeze_payload_sha256"])
    namespace_id = "entity_shadow_native_v2_" + hashlib.sha256(
        freeze_hash.encode("utf-8")
    ).hexdigest()
    database = _database(args.dsn)
    database.assert_shadow_relation_v2_ready()
    service = KnowledgeForgeService(database)
    service.executor.shutdown(wait=True)
    service.cleanup_executor.shutdown(wait=True)
    gateway = _CountingGateway(service.ollama)
    service.ollama = gateway
    runtime = ShadowRelationV2Runtime(database, gateway)
    preflight = _preflight(runtime, project_id, source_namespace)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
        return 0

    _clone_entity_snapshot(
        database, source_namespace, namespace_id, freeze_hash,
    )
    runtime = ShadowRelationV2Runtime(database, gateway)
    service.executor = _ImmediateExecutor()
    service.cleanup_executor = _ImmediateExecutor()
    graph = _NoWriteGraph()
    service.graph = graph
    legacy_calls = {"service": 0, "pipeline": 0}

    def legacy_service_tripwire(*_args, **_kwargs):
        legacy_calls["service"] += 1
        raise AssertionError("legacy _run_relation_shadow called")

    def legacy_pipeline_tripwire(*_args, **_kwargs):
        legacy_calls["pipeline"] += 1
        raise AssertionError("legacy relation pipeline called")

    service._run_relation_shadow = legacy_service_tripwire
    service.relation_shadow.prepare_shadow = legacy_pipeline_tripwire
    service._prepare_shared_content = lambda _job_id, document_id: (
        database.get_document(document_id),
        database.get_project(project_id),
        database.list_document_chunks(document_id),
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
        ("verify_hypothesis", "verifier"),
        ("normalize_verified_hypotheses", "normalizer"),
    ):
        original = getattr(v2, function_name)

        def wrapper(*values, _original=original, _counter=counter_name, **kwargs):
            stage_calls[_counter] += 1
            return _original(*values, **kwargs)

        setattr(v2, function_name, wrapper)

    def build_runtime(target_project_id: int, *, progress=None, **_kwargs):
        result = runtime.run_namespace(
            namespace_id=namespace_id,
            project_id=target_project_id,
            progress=progress,
        )
        with database.connection() as conn:
            conn.execute(
                """UPDATE entity_shadow_namespaces
                   SET metrics=metrics||%s::jsonb,status='ready',
                       snapshot_complete=TRUE,completed_at=NOW()
                   WHERE id=%s""",
                (
                    json.dumps({
                        "relation_v2_run_id": result["run_id"],
                        "relation_v2_fingerprint": result["input_fingerprint"],
                        "relation_v2": result["metrics"],
                    }),
                    namespace_id,
                ),
            )
        return {
            "namespace_id": namespace_id,
            **database.load_entity_shadow_snapshot(namespace_id),
        }

    service.build_entity_shadow_namespace = build_runtime
    before = _formal_state(database, project_id)
    document_ids = sorted(
        int(item["id"]) for item in freeze["native_project"]["documents"]
    )
    job_id = service.submit_ingestion(
        document_ids[0],
        strategy_key=ENHANCED_SHADOW_STRATEGY,
        target_document_ids=document_ids,
    )
    job = database.get_job(job_id)
    after = _formal_state(database, project_id)
    with database.connection() as conn:
        run = conn.execute(
            """SELECT * FROM shadow_relation_v2_runs
               WHERE project_id=%s AND snapshot_id=%s
               ORDER BY created_at DESC LIMIT 1""",
            (project_id, namespace_id),
        ).fetchone()
    if not run:
        raise RuntimeError("native public entry produced no v2 run")
    reconciliation = _reconcile(database, str(run["id"]))
    result = {
        "database": conninfo_to_dict(args.dsn)["dbname"],
        "job_id": job_id,
        "job_status": str(job["status"]),
        "run_id": str(run["id"]),
        "run_status": str(run["status"]),
        "namespace_id": namespace_id,
        "preflight": preflight,
        "stage_calls": stage_calls,
        "model_calls": {
            "classifier": gateway.classifier_calls,
            "verifier": gateway.verifier_calls,
        },
        "legacy_calls": legacy_calls,
        "reconciliation": reconciliation,
        "formal_graph_unchanged": before == after,
        "formal_graph_before": before,
        "formal_graph_after": after,
        "formal_graph_writes": 0,
        "activation_count": (
            int(after["counts"]["activations"])
            - int(before["counts"]["activations"])
        ),
        "external_strategy_graph_writes": 0,
        "no_write_graph_calls": graph.calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if (
        job["status"] == "completed"
        and run["status"] == "ready"
        and legacy_calls == {"service": 0, "pipeline": 0}
        and reconciliation["supported_verification_ratio"] == 1.0
        and before == after
        and result["activation_count"] == 0
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
