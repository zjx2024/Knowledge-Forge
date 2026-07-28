#!/usr/bin/env python3
"""Freeze the isolated Shadow v2 pilot inputs without mutating product data."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_forge.config import Config
from knowledge_forge.database import Database
from knowledge_forge.ollama import OllamaClient
from knowledge_forge import relation_shadow_v2 as v2
from knowledge_forge.relation_shadow_runtime import (
    SHADOW_RELATION_V2_POLICY_SET,
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _rows(conn, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [dict(item) for item in conn.execute(query, params).fetchall()]


def _database(dsn: str) -> Database:
    params = conninfo_to_dict(dsn)
    database_name = str(params.get("dbname") or "")
    if not database_name.startswith("knowledge_forge_shadow_v2_pilot_"):
        raise RuntimeError(
            "freeze requires an explicit knowledge_forge_shadow_v2_pilot_* database"
        )

    class PilotConfig:
        postgres = params
        postgres_maintenance_db = database_name

    return Database(PilotConfig)


def _project_freeze(conn, project_id: int) -> dict[str, Any]:
    project = dict(conn.execute(
        "SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE",
        (project_id,),
    ).fetchone())
    documents = _rows(
        conn,
        """SELECT id,project_id,original_filename,doc_hash,active_revision_id,status
           FROM documents
           WHERE project_id=%s AND is_deleted=FALSE ORDER BY id""",
        (project_id,),
    )
    chunks = _rows(
        conn,
        """SELECT id,project_id,document_id,revision_id,chunk_key,chunk_index,
                  content_hash
           FROM chunks
           WHERE project_id=%s AND is_deleted=FALSE
           ORDER BY document_id,chunk_index,id""",
        (project_id,),
    )
    schemas = _rows(
        conn,
        """SELECT id,project_id,version,status,schema_json,source_document_ids,
                  created_by,created_at,applied_at
           FROM graph_schemas WHERE project_id=%s ORDER BY version""",
        (project_id,),
    )
    return {
        "project": project,
        "documents": documents,
        "chunks": chunks,
        "schemas": schemas,
    }


def _namespace_freeze(conn, namespace_id: str) -> dict[str, Any]:
    namespace = dict(conn.execute(
        "SELECT * FROM entity_shadow_namespaces WHERE id=%s",
        (namespace_id,),
    ).fetchone())
    mentions = _rows(
        conn,
        """SELECT namespace_id,mention_key,project_id,document_id,revision_id,
                  chunk_id,surface,normalized_surface,evidence_start,evidence_end,
                  confidence,extraction_policy_version
           FROM entity_shadow_mentions
           WHERE namespace_id=%s ORDER BY mention_key""",
        (namespace_id,),
    )
    canonicals = _rows(
        conn,
        """SELECT namespace_id,canonical_key,name,normalized_name,primary_type,
                  source_count,generation,metadata
           FROM entity_shadow_canonicals
           WHERE namespace_id=%s ORDER BY canonical_key""",
        (namespace_id,),
    )
    assignments = _rows(
        conn,
        """SELECT namespace_id,mention_key,canonical_key,document_id,chunk_id,
                  revision_id,entity_type,policy_version,metadata
           FROM entity_shadow_assignments
           WHERE namespace_id=%s ORDER BY mention_key""",
        (namespace_id,),
    )
    endpoints = _rows(
        conn,
        """SELECT namespace_id,relation_key,document_id,revision_id,chunk_id,
                  source_mention_key,target_mention_key,source_canonical_key,
                  target_canonical_key,relation_type,evidence_text,evidence_start,
                  evidence_end,policy_version,metadata
           FROM entity_shadow_relation_endpoints
           WHERE namespace_id=%s ORDER BY relation_key""",
        (namespace_id,),
    )
    source_revisions = _rows(
        conn,
        """SELECT DISTINCT revision.revision_id,revision.project_id,
                  revision.document_id,revision.content_version,
                  revision.content_hash,revision.source_hash,revision.config_hash,
                  revision.policy_hash,revision.manifest_hash,revision.status
           FROM document_content_revisions revision
           JOIN entity_shadow_mentions mention
             ON mention.revision_id=revision.revision_id
           WHERE mention.namespace_id=%s
           ORDER BY revision.document_id,revision.revision_id""",
        (namespace_id,),
    )
    source_chunks = _rows(
        conn,
        """SELECT DISTINCT chunk.id,chunk.project_id,chunk.document_id,
                  chunk.revision_id,chunk.chunk_key,chunk.chunk_index,
                  chunk.content,chunk.content_hash
           FROM chunks chunk
           JOIN entity_shadow_mentions mention ON mention.chunk_id=chunk.id
           WHERE mention.namespace_id=%s
           ORDER BY chunk.document_id,chunk.chunk_index,chunk.id""",
        (namespace_id,),
    )
    return {
        "namespace": namespace,
        "source_revisions": source_revisions,
        "source_revisions_sha256": _digest(source_revisions),
        "source_chunks": source_chunks,
        "source_chunks_sha256": _digest(source_chunks),
        "mentions": mentions,
        "mentions_sha256": _digest(mentions),
        "canonicals": canonicals,
        "canonicals_sha256": _digest(canonicals),
        "assignments": assignments,
        "assignments_sha256": _digest(assignments),
        "endpoints": endpoints,
        "endpoints_sha256": _digest(endpoints),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-project-id", type=int, default=13)
    parser.add_argument("--frozen-legacy-project-id", type=int, default=12)
    parser.add_argument(
        "--native-entity-namespace-id",
        default="",
        help=(
            "Explicit enhanced_shadow entity namespace to freeze. If omitted, "
            "the latest ready complete namespace for --native-project-id is used."
        ),
    )
    parser.add_argument(
        "--legacy-entity-namespace-id",
        default="",
        help=(
            "Explicit legacy comparison namespace to freeze. If omitted, the "
            "latest ready complete namespace for --frozen-legacy-project-id is used."
        ),
    )
    args = parser.parse_args()

    database = _database(args.dsn)
    with database.connection() as conn:
        native = _project_freeze(conn, args.native_project_id)
        if args.native_entity_namespace_id:
            native_namespace_rows = _rows(
                conn,
                """SELECT id FROM entity_shadow_namespaces
                   WHERE id=%s AND project_id=%s
                     AND status='ready' AND snapshot_complete=TRUE""",
                (args.native_entity_namespace_id, args.native_project_id),
            )
        else:
            native_namespace_rows = _rows(
                conn,
                """SELECT id FROM entity_shadow_namespaces
                   WHERE project_id=%s
                     AND status='ready'
                     AND snapshot_complete=TRUE
                   ORDER BY created_at DESC LIMIT 1""",
                (args.native_project_id,),
            )
        if args.legacy_entity_namespace_id:
            legacy_namespace_rows = _rows(
                conn,
                """SELECT id FROM entity_shadow_namespaces
                   WHERE id=%s AND project_id=%s
                     AND status='ready' AND snapshot_complete=TRUE""",
                (args.legacy_entity_namespace_id, args.frozen_legacy_project_id),
            )
        else:
            legacy_namespace_rows = _rows(
                conn,
                """SELECT id FROM entity_shadow_namespaces
                   WHERE project_id=%s
                     AND status='ready'
                     AND snapshot_complete=TRUE
                   ORDER BY created_at DESC LIMIT 1""",
                (args.frozen_legacy_project_id,),
            )
        if len(native_namespace_rows) != 1:
            raise RuntimeError("native pilot namespace is not explicit/unique")
        if len(legacy_namespace_rows) != 1:
            raise RuntimeError("legacy pilot namespace is not explicit/unique")
        native_namespace = _namespace_freeze(
            conn, str(native_namespace_rows[0]["id"])
        )
        frozen_legacy = _namespace_freeze(
            conn, str(legacy_namespace_rows[0]["id"])
        )
        frozen_legacy_project = _project_freeze(
            conn, args.frozen_legacy_project_id
        )
        formal_state = dict(conn.execute(
            """SELECT graph_generation,graph_ready_generation,graph_status,
                      graph_policy_hash,active_policy_hash
               FROM projects WHERE id=%s""",
            (args.native_project_id,),
        ).fetchone())

    model = str(native["project"].get("llm_model") or Config.default_llm_model)
    model_context = OllamaClient().effective_model_context(model)
    code_paths = (
        Path("knowledge_forge/relation_shadow_v2.py"),
        Path("knowledge_forge/relation_shadow_runtime.py"),
        Path("knowledge_forge/service.py"),
        Path("knowledge_forge/database.py"),
        Path("scripts/replay_shadow_v2_legacy_freeze.py"),
    )
    code_digests = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in code_paths
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_diff = subprocess.run(
        ["git", "diff", "--binary", "--", *map(str, code_paths)],
        check=True,
        capture_output=True,
    ).stdout
    freeze = {
        "freeze_version": "shadow-v2-pilot-freeze-v1",
        "source_database": conninfo_to_dict(args.dsn)["dbname"],
        "git_head": git_head,
        "working_tree_diff_sha256": hashlib.sha256(
            git_diff
        ).hexdigest(),
        "code_sha256": code_digests,
        "native_project": native,
        "native_entity_snapshot": native_namespace,
        "native_formal_graph_state": formal_state,
        "frozen_legacy_project": frozen_legacy_project,
        "frozen_legacy_176": frozen_legacy,
        "legacy_secondary_only": {
            "namespace_id": native_namespace["namespace"]["id"],
            "count": len(native_namespace["endpoints"]),
            "sha256": native_namespace["endpoints_sha256"],
        },
        "relation_v2_policy": dict(SHADOW_RELATION_V2_POLICY_SET),
        "prompt_contract": {
            "classifier_prompt_version":
                v2.SHADOW_RELATION_CLASSIFIER_PROMPT_VERSION,
            "classifier_schema_version":
                v2.SHADOW_RELATION_CLASSIFIER_SCHEMA_VERSION,
            "classifier_parser_version":
                v2.SHADOW_RELATION_CLASSIFIER_PARSER_VERSION,
            "verifier_prompt_version":
                v2.SHADOW_RELATION_VERIFIER_PROMPT_VERSION,
            "verifier_schema_version":
                v2.SHADOW_RELATION_VERIFIER_SCHEMA_VERSION,
            "verifier_parser_version":
                v2.SHADOW_RELATION_VERIFIER_PARSER_VERSION,
            "classifier_prompt_source_sha256": hashlib.sha256(
                inspect.getsource(v2.classifier_prompt).encode("utf-8")
            ).hexdigest(),
            "verifier_prompt_source_sha256": hashlib.sha256(
                inspect.getsource(v2.verifier_prompt).encode("utf-8")
            ).hexdigest(),
        },
        "model": {
            "name": model,
            "digest": str(model_context.digest),
            "project_model_max_tokens": int(
                native["project"].get("model_max_tokens")
                or Config.default_model_context_tokens
            ),
        },
        "thresholds_and_candidate_parameters": {
            "relation_sentence_window": Config.relation_sentence_window,
            "relation_max_char_distance": Config.relation_max_char_distance,
            "relation_max_candidates_per_chunk":
                Config.relation_max_candidates_per_chunk,
            "relation_classifier_max_candidates":
                Config.relation_classifier_max_candidates,
            "relation_classifier_output_reserve":
                Config.relation_classifier_output_reserve,
            "relation_classifier_safety_margin":
                Config.relation_classifier_safety_margin,
            "relation_shadow_max_attempts":
                Config.relation_shadow_max_attempts,
        },
    }
    freeze["freeze_payload_sha256"] = _digest(freeze)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            freeze,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "freeze_payload_sha256": freeze["freeze_payload_sha256"],
        "documents": len(native["documents"]),
        "chunks": len(native["chunks"]),
        "mentions": len(native_namespace["mentions"]),
        "assignments": len(native_namespace["assignments"]),
        "frozen_legacy_relations": len(frozen_legacy["endpoints"]),
        "native_secondary_relations": len(native_namespace["endpoints"]),
        "model_digest": str(model_context.digest),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
