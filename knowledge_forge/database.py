"""PostgreSQL persistence for projects, retrieval, graph provenance and jobs."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import re
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .config import Config
from .graph_policy import (
    GRAPH_EMPTY,
    GRAPH_FAILED,
    GRAPH_POLICY_HASH,
    GRAPH_POLICY_VERSION,
    GRAPH_READY,
    GRAPH_REBUILDING,
    GRAPH_STALE,
)
from .relation_verification import (
    LEGACY_UNVERIFIED,
    LLM_VERIFIED,
    MANUAL_VERIFIED,
    RELATION_VERIFICATION_POLICY_VERSION,
    RULES_VERIFIED,
    UNCERTAIN,
    is_publishable_relation,
)
from .observability import InstrumentedConnection, current_metrics, increment


LOGGER = logging.getLogger(__name__)


SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS operation_metrics (
    id BIGSERIAL PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    project_id BIGINT, document_id BIGINT, job_id TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id)
);
CREATE INDEX IF NOT EXISTS idx_operation_metrics_scope
    ON operation_metrics(project_id,document_id,created_at DESC);
CREATE TABLE IF NOT EXISTS models (
    id BIGSERIAL PRIMARY KEY, name TEXT NOT NULL, model_type TEXT NOT NULL CHECK (model_type IN ('llm','embedding')),
    provider TEXT NOT NULL DEFAULT 'ollama', base_url TEXT, is_active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(name, provider)
);
CREATE TABLE IF NOT EXISTS projects (
    id BIGSERIAL PRIMARY KEY, project_key TEXT NOT NULL UNIQUE, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    llm_model TEXT, embedding_model TEXT,
    model_max_tokens INTEGER NOT NULL DEFAULT 32768 CHECK (model_max_tokens BETWEEN 1024 AND 1048576),
    chunk_max_tokens INTEGER NOT NULL DEFAULT 512 CHECK (chunk_max_tokens BETWEEN 64 AND 262144),
    graph_generation BIGINT NOT NULL DEFAULT 0,
    graph_ready_generation BIGINT NOT NULL DEFAULT 0,
    graph_policy_hash TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','archived','failed')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), is_deleted BOOLEAN NOT NULL DEFAULT FALSE, deleted_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_projects_name_active ON projects(name) WHERE is_deleted=FALSE;
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), doc_hash TEXT NOT NULL,
    original_filename TEXT NOT NULL, stored_filename TEXT NOT NULL, storage_path TEXT NOT NULL, local_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL, mime_type TEXT,
    status TEXT NOT NULL DEFAULT 'uploaded' CHECK (status IN ('uploaded','processing','indexed','failed','deleted')),
    error_message TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    indexed_at TIMESTAMPTZ, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMPTZ, UNIQUE(project_id, doc_hash)
);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project_id,is_deleted,updated_at DESC);
CREATE TABLE IF NOT EXISTS sections (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), document_id BIGINT NOT NULL REFERENCES documents(id),
    section_key TEXT NOT NULL, title TEXT NOT NULL, level INTEGER NOT NULL DEFAULT 1, content TEXT NOT NULL,
    content_hash TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(document_id,section_key)
);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id,is_deleted);
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), document_id BIGINT NOT NULL REFERENCES documents(id),
    section_id BIGINT REFERENCES sections(id), chunk_key TEXT NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 1, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE, UNIQUE(document_id,chunk_key)
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id,is_deleted);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, project_id BIGINT REFERENCES projects(id), document_id BIGINT REFERENCES documents(id),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','running','completed','failed','cancelled')),
    progress INTEGER NOT NULL DEFAULT 0, log JSONB NOT NULL DEFAULT '[]'::jsonb, error_message TEXT,
    target_generation BIGINT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id,updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_active_project ON jobs(project_id,updated_at DESC)
    WHERE status IN ('queued','running');

CREATE TABLE IF NOT EXISTS vector_embeddings (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), document_id BIGINT REFERENCES documents(id),
    chunk_id BIGINT REFERENCES chunks(id), object_type TEXT NOT NULL CHECK (object_type IN ('chunk','entity','relation','schema')),
    object_key TEXT NOT NULL, content TEXT NOT NULL, content_hash TEXT NOT NULL, model_name TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL, embedding_vector vector NOT NULL, dimension INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE, UNIQUE(project_id,object_type,object_key,model_name)
);
CREATE INDEX IF NOT EXISTS idx_vectors_lookup ON vector_embeddings(project_id,object_type,model_name,is_deleted);

CREATE TABLE IF NOT EXISTS indexing_failures (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT REFERENCES documents(id), chunk_id BIGINT REFERENCES chunks(id),
    object_type TEXT NOT NULL, object_key TEXT NOT NULL, model_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','completed','permanent_failed')),
    attempts INTEGER NOT NULL DEFAULT 1, max_attempts INTEGER NOT NULL DEFAULT 8,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), last_error TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ,
    UNIQUE(project_id,object_type,object_key,model_name)
);
CREATE INDEX IF NOT EXISTS idx_indexing_failures_pending
    ON indexing_failures(project_id,status,updated_at) WHERE status='pending';

CREATE TABLE IF NOT EXISTS graph_schemas (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft','active','retired','failed')), schema_json JSONB NOT NULL,
    compatibility_report JSONB NOT NULL DEFAULT '{}'::jsonb, source_document_ids BIGINT[] NOT NULL DEFAULT '{}',
    created_by TEXT NOT NULL DEFAULT 'system', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), applied_at TIMESTAMPTZ,
    UNIQUE(project_id,version)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_graph_schema ON graph_schemas(project_id) WHERE status='active';

CREATE TABLE IF NOT EXISTS canonical_entities (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), entity_key TEXT NOT NULL,
    name TEXT NOT NULL, normalized_name TEXT NOT NULL, entity_type TEXT NOT NULL, aliases TEXT[] NOT NULL DEFAULT '{}',
    normalized_aliases TEXT[] NOT NULL DEFAULT '{}',
    description TEXT NOT NULL DEFAULT '', source_count INTEGER NOT NULL DEFAULT 0, graph_vertex_id TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE, UNIQUE(project_id,entity_key)
);
CREATE INDEX IF NOT EXISTS idx_entities_project_type ON canonical_entities(project_id,entity_type,is_deleted);
CREATE INDEX IF NOT EXISTS idx_entities_normalized ON canonical_entities(project_id,normalized_name,is_deleted);
CREATE INDEX IF NOT EXISTS idx_entities_aliases ON canonical_entities USING gin(normalized_aliases);
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm ON canonical_entities
    USING gin(normalized_name gin_trgm_ops) WHERE is_deleted=FALSE;

CREATE TABLE IF NOT EXISTS entity_mentions (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), document_id BIGINT NOT NULL REFERENCES documents(id),
    chunk_id BIGINT NOT NULL REFERENCES chunks(id), canonical_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    mention_text TEXT NOT NULL, normalized_name TEXT NOT NULL, entity_type TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5, extraction_version TEXT NOT NULL DEFAULT 'v1',
    graph_edge_id TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(document_id,chunk_id,canonical_entity_id,mention_text)
);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON entity_mentions(canonical_entity_id,is_deleted);
CREATE INDEX IF NOT EXISTS idx_mentions_document ON entity_mentions(document_id,is_deleted);

CREATE TABLE IF NOT EXISTS entity_resolution_decisions (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT NOT NULL REFERENCES documents(id), chunk_id BIGINT NOT NULL REFERENCES chunks(id),
    canonical_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id), mention_text TEXT NOT NULL,
    normalized_name TEXT NOT NULL, entity_type TEXT NOT NULL, decision TEXT NOT NULL,
    candidate_entity_ids BIGINT[] NOT NULL DEFAULT '{}', scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason TEXT NOT NULL DEFAULT '', model_name TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(document_id,chunk_id,mention_text,entity_type)
);
CREATE INDEX IF NOT EXISTS idx_resolution_document ON entity_resolution_decisions(document_id,is_deleted);
CREATE INDEX IF NOT EXISTS idx_resolution_entity ON entity_resolution_decisions(canonical_entity_id,is_deleted);

CREATE TABLE IF NOT EXISTS relation_assertions (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id), document_id BIGINT NOT NULL REFERENCES documents(id),
    chunk_id BIGINT NOT NULL REFERENCES chunks(id), source_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    target_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id), relation_key TEXT NOT NULL, relation_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '', confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    extraction_version TEXT NOT NULL DEFAULT 'v1', ingestion_run_id TEXT NOT NULL DEFAULT '', graph_edge_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(project_id,relation_key)
);
CREATE INDEX IF NOT EXISTS idx_relations_project ON relation_assertions(project_id,relation_type,is_deleted);
CREATE INDEX IF NOT EXISTS idx_relations_document ON relation_assertions(document_id,is_deleted);

CREATE TABLE IF NOT EXISTS relation_candidates (
    candidate_id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT NOT NULL REFERENCES documents(id), chunk_id BIGINT NOT NULL,
    candidate_ordinal INTEGER NOT NULL, source_mention_id TEXT NOT NULL, source_name TEXT NOT NULL,
    target_mention_id TEXT NOT NULL, target_name TEXT NOT NULL, proposed_relation_type TEXT NOT NULL,
    proposed_evidence TEXT NOT NULL DEFAULT '', evidence_start INTEGER, evidence_end INTEGER,
    model_confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5, status TEXT NOT NULL,
    ingestion_run_id TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '', corrected_relation_type TEXT, orientation TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), is_deleted BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_relation_candidates_project_status
    ON relation_candidates(project_id,status,is_deleted,updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_relation_candidates_document
    ON relation_candidates(document_id,is_deleted,chunk_id);
CREATE INDEX IF NOT EXISTS idx_relation_candidates_chunk
    ON relation_candidates(project_id,chunk_id,is_deleted);

CREATE TABLE IF NOT EXISTS relation_verification_events (
    id BIGSERIAL PRIMARY KEY, candidate_id TEXT NOT NULL REFERENCES relation_candidates(candidate_id),
    project_id BIGINT NOT NULL REFERENCES projects(id), document_id BIGINT NOT NULL REFERENCES documents(id),
    chunk_id BIGINT NOT NULL, attempt INTEGER NOT NULL, model_name TEXT NOT NULL,
    prompt_version TEXT NOT NULL, schema_version INTEGER NOT NULL, ingestion_run_id TEXT NOT NULL DEFAULT '',
    request_json JSONB NOT NULL DEFAULT '{}'::jsonb, raw_response JSONB,
    verdict TEXT, reason TEXT NOT NULL DEFAULT '', error_category TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_relation_verification_candidate
    ON relation_verification_events(candidate_id,attempt,created_at);
CREATE INDEX IF NOT EXISTS idx_relation_verification_scope
    ON relation_verification_events(project_id,document_id,chunk_id,created_at);

ALTER TABLE projects ADD COLUMN IF NOT EXISTS schema_status TEXT NOT NULL DEFAULT 'base';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS active_schema_version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS graph_status TEXT NOT NULL DEFAULT 'empty';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS graph_policy_version TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS graph_error TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS graph_generation BIGINT NOT NULL DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS graph_ready_generation BIGINT NOT NULL DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS graph_policy_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS model_max_tokens INTEGER NOT NULL DEFAULT 32768;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS chunk_max_tokens INTEGER NOT NULL DEFAULT 512;
ALTER TABLE projects DROP COLUMN IF EXISTS ragflow_dataset_id;
ALTER TABLE projects DROP COLUMN IF EXISTS ragflow_dataset_name;
ALTER TABLE projects DROP COLUMN IF EXISTS default_parser;
ALTER TABLE projects DROP COLUMN IF EXISTS default_chunk_strategy;
ALTER TABLE documents DROP COLUMN IF EXISTS ragflow_document_id;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS requested_parser TEXT NOT NULL DEFAULT 'auto';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS requested_chunk_strategy TEXT NOT NULL DEFAULT 'auto';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS effective_parser TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS effective_chunk_strategy TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parser_version TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunk_config JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parse_quality JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS routing_reason TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_stage TEXT NOT NULL DEFAULT 'uploaded';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS structure_ready_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunks_ready_at TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS graph_ready_at TIMESTAMPTZ;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS target_generation BIGINT;
ALTER TABLE indexing_failures ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL DEFAULT 8;
ALTER TABLE indexing_failures ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE indexing_failures DROP CONSTRAINT IF EXISTS indexing_failures_status_check;
ALTER TABLE indexing_failures ADD CONSTRAINT indexing_failures_status_check
    CHECK (status IN ('pending','completed','permanent_failed'));
WITH ranked_graph_jobs AS (
    SELECT id,ROW_NUMBER() OVER (PARTITION BY project_id ORDER BY created_at,id) AS ordinal
    FROM jobs WHERE kind='graph_sync' AND status IN ('queued','running')
)
UPDATE jobs SET status='cancelled',error_message='由迁移合并的重复图重建任务',completed_at=NOW(),updated_at=NOW()
WHERE id IN (SELECT id FROM ranked_graph_jobs WHERE ordinal>1);
CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_active_graph_sync
    ON jobs(project_id) WHERE kind='graph_sync' AND status IN ('queued','running');
ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS evidence_text TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS evidence_start INTEGER;
ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS evidence_end INTEGER;
ALTER TABLE relation_assertions ADD COLUMN IF NOT EXISTS evidence_text TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_assertions ADD COLUMN IF NOT EXISTS evidence_start INTEGER;
ALTER TABLE relation_assertions ADD COLUMN IF NOT EXISTS evidence_end INTEGER;
ALTER TABLE relation_assertions ADD COLUMN IF NOT EXISTS verification_status TEXT NOT NULL DEFAULT 'unverified';
ALTER TABLE relation_assertions ADD COLUMN IF NOT EXISTS ingestion_run_id TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_candidates ADD COLUMN IF NOT EXISTS ingestion_run_id TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_verification_events ADD COLUMN IF NOT EXISTS ingestion_run_id TEXT NOT NULL DEFAULT '';
UPDATE documents SET structure_ready_at=COALESCE(structure_ready_at,updated_at)
WHERE structure_ready_at IS NULL
  AND CASE WHEN COALESCE(parse_quality->>'block_count','') ~ '^[0-9]+$'
           THEN (parse_quality->>'block_count')::INTEGER ELSE 0 END > 0;
UPDATE documents d SET chunks_ready_at=COALESCE(d.chunks_ready_at,d.updated_at)
WHERE d.chunks_ready_at IS NULL AND EXISTS(SELECT 1 FROM chunks c WHERE c.document_id=d.id AND c.is_deleted=FALSE);
UPDATE documents SET graph_ready_at=COALESCE(graph_ready_at,indexed_at,updated_at)
WHERE graph_ready_at IS NULL AND status='indexed';
UPDATE documents d SET processing_stage=CASE
    WHEN d.status='indexed' THEN 'completed'
    WHEN d.status='failed' THEN 'failed'
    WHEN d.status='processing' AND EXISTS(SELECT 1 FROM chunks c WHERE c.document_id=d.id AND c.is_deleted=FALSE) THEN 'graph'
    WHEN d.structure_ready_at IS NOT NULL THEN 'structure_ready'
    ELSE 'uploaded' END
WHERE d.processing_stage='uploaded' OR d.processing_stage IS NULL;
ALTER TABLE chunks DROP COLUMN IF EXISTS ragflow_chunk_id;
ALTER TABLE IF EXISTS graph_jobs DROP COLUMN IF EXISTS ragflow_task_id;
"""


class Database:
    def __init__(self, config: type[Config] = Config):
        self.config = config

    def bootstrap(self) -> None:
        try:
            with psycopg.connect(**{**self.config.postgres, "dbname": self.config.postgres_maintenance_db}, autocommit=True) as conn:
                exists = conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (self.config.postgres["dbname"],)).fetchone()
                if not exists:
                    conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.config.postgres["dbname"])))
        except psycopg.errors.InsufficientPrivilege:
            pass
        with self.connection() as conn:
            self._enable_vector_extensions(conn)
            conn.execute(
                "ALTER TABLE IF EXISTS canonical_entities "
                "ADD COLUMN IF NOT EXISTS normalized_aliases TEXT[] NOT NULL DEFAULT '{}'"
            )
            conn.execute(SCHEMA_SQL)
            self._migrate_relation_verification_policy(conn)
            self._migrate_graph_policy(conn)
            self._migrate_vector_storage(conn)
            self._migrate_entity_aliases(conn)
            self._seed_settings(conn)

    @staticmethod
    def _enable_vector_extensions(conn: psycopg.Connection) -> None:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        except psycopg.Error as exc:
            raise RuntimeError(
                "PostgreSQL 未安装 pgvector 扩展；请使用项目 deploy/postgres 构建的镜像，"
                "或在当前 PostgreSQL 16 中安装 postgresql-16-pgvector"
            ) from exc

    def _migrate_vector_storage(self, conn: psycopg.Connection) -> None:
        conn.execute("ALTER TABLE vector_embeddings ADD COLUMN IF NOT EXISTS embedding_vector vector")
        conn.execute(
            """UPDATE vector_embeddings
            SET dimension=cardinality(embedding), embedding_vector=embedding::vector
            WHERE embedding_vector IS NULL OR dimension<>cardinality(embedding)"""
        )
        missing = conn.execute(
            "SELECT COUNT(*) AS count FROM vector_embeddings WHERE embedding_vector IS NULL"
        ).fetchone()
        if int(missing["count"] or 0):
            raise RuntimeError("部分历史向量无法迁移到 pgvector")
        conn.execute("ALTER TABLE vector_embeddings ALTER COLUMN embedding_vector SET NOT NULL")
        conn.execute(
            """CREATE INDEX IF NOT EXISTS idx_vectors_content_trgm
            ON vector_embeddings USING gin (content gin_trgm_ops) WHERE is_deleted=FALSE"""
        )
        configurations = conn.execute(
            "SELECT DISTINCT model_name,dimension FROM vector_embeddings WHERE dimension>0"
        ).fetchall()
        for item in configurations:
            self._ensure_hnsw_index(conn, item["model_name"], int(item["dimension"]))

    @staticmethod
    def _normalize_entity_name(value: str) -> str:
        value = unicodedata.normalize("NFKC", value or "").lower().strip()
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)

    def _migrate_entity_aliases(self, conn: psycopg.Connection) -> None:
        rows = conn.execute(
            """SELECT id,aliases FROM canonical_entities
               WHERE cardinality(aliases)>0 AND cardinality(normalized_aliases)=0"""
        ).fetchall()
        for item in rows:
            normalized = list(dict.fromkeys(
                value for value in (
                    self._normalize_entity_name(alias) for alias in (item["aliases"] or [])
                ) if value
            ))
            conn.execute(
                "UPDATE canonical_entities SET normalized_aliases=%s WHERE id=%s",
                (normalized, item["id"]),
            )

    @staticmethod
    def _vector_literal(values: list[float]) -> str:
        vector = [float(value) for value in values]
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("Embedding 向量不能为空且必须全部为有限数值")
        return "[" + ",".join(format(value, ".12g") for value in vector) + "]"

    @staticmethod
    def _ensure_hnsw_index(conn: psycopg.Connection, model_name: str, dimension: int) -> None:
        if dimension <= 0 or dimension > Config.pgvector_hnsw_max_dimensions:
            LOGGER.info("向量维度 %s 不适合 HNSW vector 索引，将使用精确 pgvector 检索", dimension)
            return
        suffix = hashlib.sha256(f"{model_name}|{dimension}".encode()).hexdigest()[:12]
        statement = sql.SQL(
            """CREATE INDEX IF NOT EXISTS {} ON vector_embeddings
            USING hnsw ((embedding_vector::vector({})) vector_cosine_ops)
            WITH (m={}, ef_construction={})
            WHERE model_name={} AND dimension={} AND is_deleted=FALSE"""
        ).format(
            sql.Identifier(f"idx_vectors_hnsw_{suffix}"),
            sql.Literal(dimension),
            sql.Literal(Config.pgvector_hnsw_m),
            sql.Literal(Config.pgvector_hnsw_ef_construction),
            sql.Literal(model_name),
            sql.Literal(dimension),
        )
        conn.execute(statement)

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        increment("db_connection_count")
        with psycopg.connect(**self.config.postgres, row_factory=dict_row) as conn:
            if current_metrics() is None:
                yield conn
            else:
                yield InstrumentedConnection(conn)

    def record_operation_metrics(
        self,
        *,
        run_id: str,
        kind: str,
        metrics: dict[str, Any],
        project_id: int | None = None,
        document_id: int | None = None,
        job_id: str | None = None,
    ) -> None:
        """Best-effort callers persist aggregate metrics after the measured scope."""

        with self.connection() as conn:
            conn.execute(
                """INSERT INTO operation_metrics(
                       run_id,kind,project_id,document_id,job_id,metrics
                   ) VALUES(%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT(run_id) DO UPDATE SET metrics=EXCLUDED.metrics""",
                (run_id, kind, project_id, document_id, job_id, self._json(metrics)),
            )

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps({} if value is None else value, ensure_ascii=False)

    @staticmethod
    def _publishable_relation_clause(alias: str = "r") -> tuple[str, list[Any]]:
        """Return the fail-closed product predicate and its parameters."""

        return (
            f"({alias}.verification_status=%s OR "
            f"({alias}.verification_status=ANY(%s) AND "
            f"COALESCE({alias}.metadata->>'verification_policy_version','')=%s))",
            [
                MANUAL_VERIFIED,
                sorted({RULES_VERIFIED, LLM_VERIFIED}),
                RELATION_VERIFICATION_POLICY_VERSION,
            ],
        )

    @classmethod
    def _migrate_relation_verification_policy(
        cls, conn: psycopg.Connection,
    ) -> dict[str, int]:
        """Quarantine pre-policy automatic assertions/candidates and bad vectors."""

        assertion_result = conn.execute(
            """UPDATE relation_assertions
               SET verification_status=%s,
                   metadata=(CASE WHEN jsonb_typeof(metadata)='object' THEN metadata
                                  ELSE jsonb_build_object('legacy_metadata',metadata) END)
                            || jsonb_build_object(
                                'legacy_verification_status',verification_status,
                                'migration_reason','missing current strict verifier policy'),
                   graph_edge_id=NULL,
                   updated_at=NOW()
               WHERE verification_status=ANY(%s)
                 AND COALESCE(metadata->>'verification_policy_version','')<>%s""",
            (
                LEGACY_UNVERIFIED,
                sorted({RULES_VERIFIED, LLM_VERIFIED}),
                RELATION_VERIFICATION_POLICY_VERSION,
            ),
        )
        candidate_result = conn.execute(
            """UPDATE relation_candidates
               SET status=%s,
                   metadata=(CASE WHEN jsonb_typeof(metadata)='object' THEN metadata
                                  ELSE jsonb_build_object('legacy_metadata',metadata) END)
                            || jsonb_build_object(
                                'legacy_verification_status',status,
                                'migration_reason','missing current strict verifier policy'),
                   updated_at=NOW()
               WHERE status=ANY(%s)
                 AND COALESCE(metadata->>'verification_policy_version','')<>%s""",
            (
                LEGACY_UNVERIFIED,
                sorted({RULES_VERIFIED, LLM_VERIFIED}),
                RELATION_VERIFICATION_POLICY_VERSION,
            ),
        )
        clause, policy_params = cls._publishable_relation_clause("r")
        vector_result = conn.execute(
            """UPDATE vector_embeddings v
               SET is_deleted=TRUE,updated_at=NOW()
               WHERE v.object_type='relation' AND v.is_deleted=FALSE
                 AND NOT EXISTS (
                     SELECT 1 FROM relation_assertions r
                     WHERE r.project_id=v.project_id AND r.id::text=v.object_key
                       AND r.is_deleted=FALSE AND """ + clause + ")",
            tuple(policy_params),
        )
        return {
            "assertions_quarantined": max(0, assertion_result.rowcount or 0),
            "candidates_quarantined": max(0, candidate_result.rowcount or 0),
            "vectors_retired": max(0, vector_result.rowcount or 0),
        }

    @staticmethod
    def _migrate_graph_policy(conn: psycopg.Connection) -> int:
        """Mark every non-empty graph built under an old/unknown policy stale."""

        result = conn.execute(
            """UPDATE projects
               SET graph_status=%s,graph_policy_version='',graph_policy_hash='',
                   graph_error=%s,updated_at=NOW()
               WHERE is_deleted=FALSE AND graph_status<>%s
                 AND (graph_policy_version<>%s OR graph_policy_hash<>%s
                      OR graph_generation<>graph_ready_generation)""",
            (
                GRAPH_STALE,
                "图谱需要按当前策略集合和代次重建",
                GRAPH_EMPTY,
                GRAPH_POLICY_VERSION,
                GRAPH_POLICY_HASH,
            ),
        )
        return max(0, result.rowcount or 0)

    def migrate_relation_verification_policy(self) -> dict[str, int]:
        """Run the idempotent database-only part of the P0 verification upgrade."""

        with self.connection() as conn:
            return self._migrate_relation_verification_policy(conn)

    @staticmethod
    def _seed_settings(conn: psycopg.Connection) -> None:
        defaults = {"ollama_base_url": Config.ollama_base_url, "hugegraph_graph": Config.hugegraph_graph}
        for key, value in defaults.items():
            conn.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO NOTHING", (key, value))
        conn.execute(
            """INSERT INTO settings(key,value) VALUES('vector_backend','pgvector-hnsw')
            ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()"""
        )

    def list_models(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM models WHERE is_active=TRUE ORDER BY model_type,name").fetchall()

    def upsert_model(
        self,
        name: str,
        model_type: str,
        provider: str = "ollama",
        base_url: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO models(name,model_type,provider,base_url,metadata) VALUES(%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(name,provider) DO UPDATE SET model_type=EXCLUDED.model_type,base_url=EXCLUDED.base_url,
                metadata=EXCLUDED.metadata,is_active=TRUE,updated_at=NOW() RETURNING *""",
                (name, model_type, provider, base_url or None, self._json(metadata)),
            ).fetchone()

    @staticmethod
    def project_key(project_id: int, name: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "-", name or "project").strip("-").lower()[:32] or "project"
        return f"{normalized}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{project_id}"

    def create_project(self, name: str, description: str, llm_model: str | None, embedding_model: str | None) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """INSERT INTO projects(project_key,name,description,llm_model,embedding_model)
                VALUES(%s,%s,%s,%s,%s) RETURNING *""",
                (f"pending-{uuid.uuid4().hex}", name, description, llm_model, embedding_model),
            ).fetchone()
            key = self.project_key(int(row["id"]), name)
            return conn.execute("UPDATE projects SET project_key=%s,updated_at=NOW() WHERE id=%s RETURNING *", (key, row["id"])).fetchone()

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM projects WHERE is_deleted=FALSE ORDER BY updated_at DESC,id DESC").fetchall()

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE", (project_id,)).fetchone()

    def update_project(self, project_id: int, **values: Any) -> dict[str, Any]:
        allowed = {"llm_model", "embedding_model", "model_max_tokens", "chunk_max_tokens",
                   "schema_status", "active_schema_version", "graph_status",
                   "graph_policy_version", "graph_policy_hash", "graph_error", "metadata", "status"}
        fields = [key for key in values if key in allowed]
        if not fields:
            return self.get_project(project_id) or {}
        assignments = ",".join(f"{key}={'%s::jsonb' if key == 'metadata' else '%s'}" for key in fields)
        params = [self._json(values[key]) if key == "metadata" and not isinstance(values[key], str) else values[key] for key in fields] + [project_id]
        with self.connection() as conn:
            return conn.execute(f"UPDATE projects SET {assignments},updated_at=NOW() WHERE id=%s RETURNING *", params).fetchone()

    def set_graph_state(
        self,
        project_id: int,
        status: str,
        *,
        error: str = "",
        current_policy: bool = False,
    ) -> dict[str, Any]:
        """Persist one fail-closed graph lifecycle transition."""

        if status not in {GRAPH_EMPTY, GRAPH_STALE, GRAPH_REBUILDING, GRAPH_READY, GRAPH_FAILED}:
            raise ValueError(f"未知图谱状态: {status}")
        policy_version = GRAPH_POLICY_VERSION if current_policy else ""
        policy_hash = GRAPH_POLICY_HASH if current_policy else ""
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE projects SET graph_status=%s,graph_policy_version=%s,
                   graph_policy_hash=%s,graph_error=%s,
                   graph_ready_generation=CASE WHEN %s THEN graph_generation ELSE graph_ready_generation END,
                   updated_at=NOW()
                   WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                (status, policy_version, policy_hash, error or "", current_policy, project_id),
            ).fetchone()
            if not row:
                raise LookupError("项目不存在")
            return row

    def mark_graph_stale(self, project_id: int, reason: str) -> dict[str, Any]:
        """Atomically advance graph facts and close product graph reads."""

        with self.connection() as conn:
            row = conn.execute(
                """UPDATE projects
                   SET graph_generation=graph_generation+1,graph_status=%s,
                       graph_policy_version='',graph_policy_hash='',graph_error=%s,updated_at=NOW()
                   WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                (GRAPH_STALE, reason or "图谱事实已更新，等待重建", project_id),
            ).fetchone()
            if not row:
                raise LookupError("项目不存在")
            return row

    def complete_graph_rebuild(
        self,
        project_id: int,
        target_generation: int,
        *,
        policy_hash: str,
        empty: bool = False,
    ) -> dict[str, Any] | None:
        """Publish a rebuild only if its immutable generation is still current."""

        status = GRAPH_EMPTY if empty else GRAPH_READY
        with self.connection() as conn:
            return conn.execute(
                """UPDATE projects
                   SET graph_status=%s,graph_policy_version=%s,graph_policy_hash=%s,
                       graph_ready_generation=%s,graph_error='',updated_at=NOW()
                   WHERE id=%s AND is_deleted=FALSE AND graph_generation=%s
                   RETURNING *""",
                (
                    status,
                    GRAPH_POLICY_VERSION,
                    policy_hash,
                    target_generation,
                    project_id,
                    target_generation,
                ),
            ).fetchone()

    def has_pending_ingestions(self, project_id: int, *, exclude_job_id: str = "") -> bool:
        """Return whether more queued ingestion work can change this project generation."""

        with self.connection() as conn:
            return bool(conn.execute(
                """SELECT 1 FROM jobs WHERE project_id=%s AND kind='ingestion'
                   AND status='queued' AND id<>%s LIMIT 1""",
                (project_id, exclude_job_id),
            ).fetchone())

    def list_orphaned_graph_projects(self) -> list[dict[str, Any]]:
        """Find graph states that need a durable recovery job after restart."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT p.* FROM projects p
                   WHERE p.is_deleted=FALSE AND p.graph_status IN (%s,%s,%s)
                     AND NOT EXISTS (
                         SELECT 1 FROM jobs j WHERE j.project_id=p.id
                           AND j.kind=ANY(%s) AND j.status IN ('queued','running')
                     )
                   ORDER BY p.id""",
                (
                    GRAPH_STALE,
                    GRAPH_REBUILDING,
                    GRAPH_FAILED,
                    ["ingestion", "graph_sync", "schema_reassessment", "document_cleanup"],
                ),
            ).fetchall()

    def create_document(self, values: dict[str, Any]) -> dict[str, Any]:
        columns = ("project_id","doc_hash","original_filename","stored_filename","storage_path","local_path",
                   "file_size_bytes","mime_type","requested_parser","requested_chunk_strategy","chunk_config","metadata")
        defaults = {
            "requested_parser": "auto",
            "requested_chunk_strategy": "auto",
            "chunk_config": {},
            "metadata": {},
        }
        params = []
        for key in columns:
            value = values.get(key, defaults.get(key))
            if key in {"chunk_config", "metadata"} and not isinstance(value, str):
                value = self._json(value)
            params.append(value)
        with self.connection() as conn:
            return conn.execute(
                f"""INSERT INTO documents({','.join(columns)})
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb) RETURNING *""", params
            ).fetchone()

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT d.*,p.name AS project_name,p.project_key,p.llm_model,p.embedding_model
                FROM documents d JOIN projects p ON p.id=d.project_id WHERE d.id=%s AND d.is_deleted=FALSE""", (document_id,)
            ).fetchone()

    def get_document_including_deleted(self, document_id: int) -> dict[str, Any] | None:
        """Return a document snapshot for durable cleanup and job recovery."""
        with self.connection() as conn:
            return conn.execute(
                """SELECT d.*,p.name AS project_name,p.project_key,p.llm_model,p.embedding_model
                FROM documents d JOIN projects p ON p.id=d.project_id WHERE d.id=%s""",
                (document_id,),
            ).fetchone()

    def find_document_by_hash(self, project_id: int, doc_hash: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM documents WHERE project_id=%s AND doc_hash=%s", (project_id, doc_hash)).fetchone()

    def list_documents(self, project_id: int | None = None) -> list[dict[str, Any]]:
        condition, params = "d.is_deleted=FALSE", []
        if project_id:
            condition += " AND d.project_id=%s"; params.append(project_id)
        with self.connection() as conn:
            return conn.execute(
                f"""SELECT d.*,p.name AS project_name FROM documents d JOIN projects p ON p.id=d.project_id
                WHERE {condition} ORDER BY d.updated_at DESC,d.id DESC""", params
            ).fetchall()

    def count_documents(self, project_id: int) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM documents WHERE project_id=%s AND is_deleted=FALSE",
                (project_id,),
            ).fetchone()
            return int(row["count"] if row else 0)

    def update_document(self, document_id: int, **values: Any) -> dict[str, Any]:
        if not values:
            return self.get_document(document_id) or {}
        fields = list(values)
        json_fields = {"metadata", "chunk_config", "parse_quality"}
        assignments = ",".join(f"{key}={'%s::jsonb' if key in json_fields else '%s'}" for key in fields)
        params = [self._json(values[key]) if key in json_fields and not isinstance(values[key], str) else values[key] for key in fields] + [document_id]
        with self.connection() as conn:
            row = conn.execute(f"UPDATE documents SET {assignments},updated_at=NOW() WHERE id=%s RETURNING *", params).fetchone()
            if not row: raise LookupError("文档不存在")
            return row

    def finalize_document_graph_facts(
        self,
        document_id: int,
        *,
        reason: str,
        **values: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Commit the indexed document projection and generation advance together."""

        fields = list(values)
        json_fields = {"metadata", "chunk_config", "parse_quality"}
        assignments = ",".join(
            f"{key}={'%s::jsonb' if key in json_fields else '%s'}" for key in fields
        )
        params = [
            self._json(values[key])
            if key in json_fields and not isinstance(values[key], str)
            else values[key]
            for key in fields
        ]
        with self.connection() as conn:
            document = conn.execute(
                f"""UPDATE documents SET {assignments},updated_at=NOW()
                    WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                [*params, document_id],
            ).fetchone()
            if not document:
                raise LookupError("文档不存在或已删除")
            project = conn.execute(
                """UPDATE projects
                   SET graph_generation=graph_generation+1,graph_status=%s,
                       graph_policy_version='',graph_policy_hash='',graph_error=%s,updated_at=NOW()
                   WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                (GRAPH_STALE, reason, document["project_id"]),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            return document, project

    def soft_delete_document(self, document_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE documents SET is_deleted=TRUE,status='deleted',deleted_at=NOW(),updated_at=NOW()
                WHERE id=%s AND is_deleted=FALSE RETURNING *""", (document_id,)
            ).fetchone()
            conn.execute("UPDATE sections SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            conn.execute("UPDATE chunks SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            conn.execute("UPDATE vector_embeddings SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            conn.execute("UPDATE entity_mentions SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            conn.execute("UPDATE entity_resolution_decisions SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            conn.execute("UPDATE relation_assertions SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            conn.execute("UPDATE relation_candidates SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            self._retire_orphan_entities(conn)
            if row:
                conn.execute(
                    """UPDATE projects
                       SET graph_generation=graph_generation+1,graph_status=%s,
                           graph_policy_version='',graph_policy_hash='',graph_error=%s,updated_at=NOW()
                       WHERE id=%s AND is_deleted=FALSE""",
                    (GRAPH_STALE, "项目文档已删除，图谱正在重新收敛", row["project_id"]),
                )
            return row

    def replace_document_content(self, document_id: int, sections: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self.connection() as conn:
            document = conn.execute(
                "SELECT project_id FROM documents WHERE id=%s AND is_deleted=FALSE",
                (document_id,),
            ).fetchone()
            if not document: raise LookupError("文档不存在或已删除")
            conn.execute("UPDATE relation_candidates SET is_deleted=TRUE,updated_at=NOW() WHERE document_id=%s", (document_id,))
            conn.execute("DELETE FROM indexing_failures WHERE document_id=%s", (document_id,))
            conn.execute("DELETE FROM vector_embeddings WHERE document_id=%s", (document_id,))
            conn.execute("DELETE FROM relation_assertions WHERE document_id=%s", (document_id,))
            conn.execute("DELETE FROM entity_resolution_decisions WHERE document_id=%s", (document_id,))
            conn.execute("DELETE FROM entity_mentions WHERE document_id=%s", (document_id,))
            conn.execute("DELETE FROM chunks WHERE document_id=%s", (document_id,))
            conn.execute("DELETE FROM sections WHERE document_id=%s", (document_id,))
            section_ids: dict[str, int] = {}
            for section in sections:
                row = conn.execute(
                    """INSERT INTO sections(project_id,document_id,section_key,title,level,content,content_hash,metadata)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb) RETURNING id""",
                    (document["project_id"],document_id,section["key"],section["title"],section["level"],section["content"],section["content_hash"],self._json(section.get("metadata"))),
                ).fetchone(); section_ids[section["key"]] = int(row["id"])
            inserted = []
            for chunk in chunks:
                row = conn.execute(
                    """INSERT INTO chunks(project_id,document_id,section_id,chunk_key,content,content_hash,chunk_index,metadata)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb) RETURNING *""",
                    (document["project_id"],document_id,section_ids.get(chunk["section_key"]),chunk["key"],chunk["content"],chunk["content_hash"],chunk["part_index"],self._json(chunk.get("metadata"))),
                ).fetchone(); inserted.append(row)
            return inserted

    def list_document_chunks(self, document_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT c.*,s.title AS section_title FROM chunks c LEFT JOIN sections s ON s.id=c.section_id
                WHERE c.document_id=%s AND c.is_deleted=FALSE ORDER BY c.chunk_index,c.id""", (document_id,)
            ).fetchall()

    def replace_document_embeddings(self, document_id: int, records: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            active = conn.execute(
                "SELECT 1 FROM documents WHERE id=%s AND is_deleted=FALSE",
                (document_id,),
            ).fetchone()
            if not active:
                raise LookupError("文档不存在或已删除")
            conn.execute("DELETE FROM vector_embeddings WHERE document_id=%s AND object_type='chunk'", (document_id,))
            for record in records: self._upsert_embedding(conn, record)

    def upsert_embedding(self, record: dict[str, Any]) -> None:
        with self.connection() as conn: self._upsert_embedding(conn, record)

    def _upsert_embedding(self, conn: psycopg.Connection, record: dict[str, Any]) -> None:
        vector = list(record["embedding"])
        vector_literal = self._vector_literal(vector)
        self._ensure_hnsw_index(conn, record["model_name"], len(vector))
        conn.execute(
            """INSERT INTO vector_embeddings(project_id,document_id,chunk_id,object_type,object_key,content,content_hash,
            model_name,embedding,embedding_vector,dimension,metadata)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,%s::jsonb)
            ON CONFLICT(project_id,object_type,object_key,model_name) DO UPDATE SET document_id=EXCLUDED.document_id,
            chunk_id=EXCLUDED.chunk_id,content=EXCLUDED.content,content_hash=EXCLUDED.content_hash,embedding=EXCLUDED.embedding,
            embedding_vector=EXCLUDED.embedding_vector,dimension=EXCLUDED.dimension,metadata=EXCLUDED.metadata,
            is_deleted=FALSE,updated_at=NOW()""",
            (record["project_id"],record.get("document_id"),record.get("chunk_id"),record["object_type"],
             record["object_key"],record["content"],record["content_hash"],record["model_name"],vector,
             vector_literal,len(vector),self._json(record.get("metadata"))),
        )

    def search_embeddings(
        self,
        project_id: int,
        object_type: str,
        model_name: str,
        query_vector: list[float],
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        vector_literal = self._vector_literal(query_vector)
        dimension = len(query_vector)
        candidate_limit = max(limit, min(Config.pgvector_max_candidates, limit * Config.pgvector_candidate_multiplier))
        pattern = f"%{query}%"
        vector_type = sql.SQL("vector({})").format(sql.Literal(dimension))
        relation_filter = ""
        relation_policy_params: list[Any] = []
        if object_type == "relation":
            relation_clause, relation_policy_params = self._publishable_relation_clause("r")
            relation_filter = (
                " AND EXISTS (SELECT 1 FROM relation_assertions r "
                "WHERE r.project_id=v0.project_id AND r.id::text=v0.object_key "
                "AND r.is_deleted=FALSE AND " + relation_clause + ")"
            )
        statement = sql.SQL(
            """WITH vector_hits AS (
                SELECT id FROM vector_embeddings v0
                WHERE project_id=%s AND object_type=%s AND model_name=%s AND dimension=%s AND is_deleted=FALSE {}
                ORDER BY embedding_vector::{} <=> %s::{}
                LIMIT %s
            ), lexical_hits AS (
                SELECT id FROM vector_embeddings v0
                WHERE project_id=%s AND object_type=%s AND model_name=%s AND dimension=%s AND is_deleted=FALSE {}
                  AND content ILIKE %s
                ORDER BY length(content),id LIMIT %s
            ), candidates AS (
                SELECT id FROM vector_hits UNION SELECT id FROM lexical_hits
            )
            SELECT v.id,v.project_id,v.document_id,v.chunk_id,v.object_type,v.object_key,v.content,
                   v.content_hash,v.model_name,v.dimension,v.metadata,v.created_at,v.updated_at,
                   d.original_filename,d.storage_path,s.title AS section_title,
                   GREATEST(0.0,1.0-(v.embedding_vector::{} <=> %s::{})) AS dense_score
            FROM candidates hit JOIN vector_embeddings v ON v.id=hit.id
            LEFT JOIN documents d ON d.id=v.document_id
            LEFT JOIN chunks c ON c.id=v.chunk_id LEFT JOIN sections s ON s.id=c.section_id
            WHERE (d.id IS NULL OR d.is_deleted=FALSE)
            ORDER BY v.embedding_vector::{} <=> %s::{}
            LIMIT %s"""
        ).format(
            sql.SQL(relation_filter), vector_type, vector_type,
            sql.SQL(relation_filter), vector_type, vector_type, vector_type, vector_type,
        )
        params = (
            project_id,object_type,model_name,dimension,*relation_policy_params,
            vector_literal,candidate_limit,
            project_id,object_type,model_name,dimension,*relation_policy_params,
            pattern,candidate_limit,
            vector_literal,vector_literal,candidate_limit,
        )
        with self.connection() as conn:
            conn.execute("SELECT set_config('hnsw.ef_search',%s,TRUE)", (str(Config.pgvector_hnsw_ef_search),))
            return conn.execute(statement, params).fetchall()

    def record_indexing_failure(self, values: dict[str, Any]) -> dict[str, Any]:
        """Persist a retryable derived-index failure without changing source truth."""

        with self.connection() as conn:
            existing = conn.execute(
                """SELECT attempts,max_attempts FROM indexing_failures
                   WHERE project_id=%s AND object_type=%s AND object_key=%s AND model_name=%s
                   FOR UPDATE""",
                (
                    values["project_id"], values["object_type"], str(values["object_key"]),
                    values["model_name"],
                ),
            ).fetchone()
            max_attempts = int(
                values.get("max_attempts")
                or (existing or {}).get("max_attempts")
                or Config.indexing_retry_max_attempts
            )
            attempts = int((existing or {}).get("attempts") or 0) + 1
            status = "permanent_failed" if attempts >= max_attempts else "pending"
            delay_seconds = min(
                86400,
                Config.indexing_retry_base_seconds * (2 ** max(0, attempts - 1)),
            )
            next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
            return conn.execute(
                """INSERT INTO indexing_failures(project_id,document_id,chunk_id,object_type,
                   object_key,model_name,status,attempts,max_attempts,next_attempt_at,last_error,payload)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT(project_id,object_type,object_key,model_name) DO UPDATE SET
                   document_id=EXCLUDED.document_id,chunk_id=EXCLUDED.chunk_id,status=EXCLUDED.status,
                   attempts=EXCLUDED.attempts,max_attempts=EXCLUDED.max_attempts,
                   next_attempt_at=EXCLUDED.next_attempt_at,last_error=EXCLUDED.last_error,
                   payload=EXCLUDED.payload,completed_at=NULL,updated_at=NOW() RETURNING *""",
                (
                    values["project_id"], values.get("document_id"), values.get("chunk_id"),
                    values["object_type"], str(values["object_key"]), values["model_name"],
                    status, attempts, max_attempts, next_attempt_at,
                    values.get("last_error") or "", self._json(values.get("payload")),
                ),
            ).fetchone()

    def complete_indexing_failure(
        self, project_id: int, object_type: str, object_key: str, model_name: str,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """UPDATE indexing_failures SET status='completed',last_error='',
                   completed_at=NOW(),updated_at=NOW()
                   WHERE project_id=%s AND object_type=%s AND object_key=%s AND model_name=%s""",
                (project_id, object_type, str(object_key), model_name),
            )

    def list_pending_indexing_failures(
        self,
        project_id: int,
        object_type: str = "relation",
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        page_size = max(1, min(limit or Config.indexing_retry_batch_size, 1000))
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM indexing_failures WHERE project_id=%s AND object_type=%s
                   AND status='pending' AND next_attempt_at<=NOW()
                   ORDER BY next_attempt_at,id LIMIT %s""",
                (project_id, object_type, page_size),
            ).fetchall()

    def cleanup_indexing_failures(self) -> int:
        """Bound completed/permanent outbox history by the configured retention."""

        cutoff = datetime.now(timezone.utc) - timedelta(days=Config.indexing_failure_retention_days)
        with self.connection() as conn:
            result = conn.execute(
                """DELETE FROM indexing_failures
                   WHERE status IN ('completed','permanent_failed') AND updated_at<%s""",
                (cutoff,),
            )
            return max(0, result.rowcount or 0)

    def list_embeddings(self, project_id: int, object_type: str, model_name: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT v.*,d.original_filename,d.storage_path,s.title AS section_title
                FROM vector_embeddings v LEFT JOIN documents d ON d.id=v.document_id
                LEFT JOIN chunks c ON c.id=v.chunk_id LEFT JOIN sections s ON s.id=c.section_id
                WHERE v.project_id=%s AND v.object_type=%s AND v.model_name=%s AND v.is_deleted=FALSE
                AND (d.id IS NULL OR d.is_deleted=FALSE)""", (project_id,object_type,model_name)
            ).fetchall()

    def list_document_embeddings(
        self, document_id: int, object_type: str, model_name: str
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT id,project_id,document_id,chunk_id,object_type,object_key,content,
                          model_name,embedding,dimension,metadata
                   FROM vector_embeddings
                   WHERE document_id=%s AND object_type=%s AND model_name=%s AND is_deleted=FALSE
                   ORDER BY chunk_id,id""",
                (document_id, object_type, model_name),
            ).fetchall()

    def count_prior_indexed_documents(self, project_id: int, document_id: int) -> int:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS count FROM documents
                   WHERE project_id=%s AND id<>%s AND status='indexed' AND is_deleted=FALSE""",
                (project_id, document_id),
            ).fetchone()
            return int(row["count"] if row else 0)

    def nearest_embedding_score(
        self,
        project_id: int,
        object_type: str,
        model_name: str,
        query_vector: list[float],
        *,
        exclude_document_id: int | None = None,
    ) -> float | None:
        vector_literal = self._vector_literal(query_vector)
        dimension = len(query_vector)
        vector_type = sql.SQL("vector({})").format(sql.Literal(dimension))
        statement = sql.SQL(
            """SELECT GREATEST(0.0,1.0-(embedding_vector::{} <=> %s::{})) AS score
               FROM vector_embeddings
               WHERE project_id=%s AND object_type=%s AND model_name=%s AND dimension=%s
                 AND is_deleted=FALSE AND (%s IS NULL OR document_id<>%s)
               ORDER BY embedding_vector::{} <=> %s::{}
               LIMIT 1"""
        ).format(vector_type, vector_type, vector_type, vector_type)
        with self.connection() as conn:
            conn.execute(
                "SELECT set_config('hnsw.ef_search',%s,TRUE)",
                (str(Config.pgvector_hnsw_ef_search),),
            )
            row = conn.execute(
                statement,
                (
                    vector_literal,
                    project_id,
                    object_type,
                    model_name,
                    dimension,
                    exclude_document_id,
                    exclude_document_id,
                    vector_literal,
                ),
            ).fetchone()
            return float(row["score"]) if row and row.get("score") is not None else None

    def create_job(self, kind: str, project_id: int | None = None, document_id: int | None = None) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        with self.connection() as conn:
            return conn.execute("INSERT INTO jobs(id,kind,project_id,document_id) VALUES(%s,%s,%s,%s) RETURNING *", (job_id,kind,project_id,document_id)).fetchone()

    def create_graph_rebuild_job(self, project_id: int) -> dict[str, Any]:
        """Queue or coalesce one rebuild at the project's current generation."""

        job_id = uuid.uuid4().hex
        with self.connection() as conn:
            project = conn.execute(
                "SELECT id,graph_generation FROM projects WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            target_generation = int(project.get("graph_generation") or 0)
            active_rebuild = conn.execute(
                """SELECT id,status,target_generation FROM jobs WHERE project_id=%s AND kind='graph_sync'
                   AND status IN ('queued','running') ORDER BY created_at LIMIT 1 FOR UPDATE""",
                (project_id,),
            ).fetchone()
            if active_rebuild:
                if active_rebuild.get("status") == "running":
                    return {**active_rebuild, "_coalesced": True}
                row = conn.execute(
                    """UPDATE jobs SET target_generation=%s,updated_at=NOW()
                       WHERE id=%s RETURNING *""",
                    (target_generation, active_rebuild["id"]),
                ).fetchone()
                return {**row, "_coalesced": True}
            active = conn.execute(
                """SELECT id FROM jobs WHERE project_id=%s
                   AND kind=ANY(%s) AND status IN ('queued','running') LIMIT 1""",
                (project_id, ["ingestion", "schema_reassessment", "document_cleanup"]),
            ).fetchone()
            if active:
                raise ValueError("当前项目已有会修改图谱的任务正在执行")
            job = conn.execute(
                """INSERT INTO jobs(id,kind,project_id,target_generation)
                   VALUES(%s,'graph_sync',%s,%s)
                   RETURNING *""",
                (job_id, project_id, target_generation),
            ).fetchone()
            conn.execute(
                """UPDATE projects SET graph_status=%s,graph_policy_version='',graph_error='',
                   updated_at=NOW() WHERE id=%s""",
                (GRAPH_REBUILDING, project_id),
            )
            return job

    def begin_graph_mutation(self, job_id: str) -> dict[str, Any]:
        """Start one project graph writer while excluding other running writers."""

        with self.connection() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (job_id,)).fetchone()
            if not job:
                raise LookupError("任务不存在")
            if not job.get("project_id"):
                raise ValueError("图谱写任务缺少项目")
            conn.execute(
                "SELECT id FROM projects WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (job["project_id"],),
            )
            other = conn.execute(
                """SELECT id FROM jobs WHERE project_id=%s AND id<>%s
                   AND kind=ANY(%s) AND status='running' LIMIT 1""",
                (
                    job["project_id"], job_id,
                    ["ingestion", "graph_sync", "schema_reassessment"],
                ),
            ).fetchone()
            if other:
                raise RuntimeError("当前项目已有图谱写任务正在运行")
            conn.execute(
                "UPDATE jobs SET status='running',updated_at=NOW() WHERE id=%s",
                (job_id,),
            )
            conn.execute(
                """UPDATE projects SET graph_status=%s,graph_policy_version='',graph_error='',
                   updated_at=NOW() WHERE id=%s""",
                (GRAPH_REBUILDING, job["project_id"]),
            )
            return {**job, "status": "running"}

    def update_job(self, job_id: str, *, status: str | None = None, progress: int | None = None, log: str = "", error: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (job_id,)).fetchone()
            if not row: raise LookupError("任务不存在")
            logs = list(row.get("log") or [])
            if log: logs.append(log)
            next_status = status or row["status"]
            existing_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            merged_metadata = {**existing_metadata, **metadata} if metadata is not None else existing_metadata
            return conn.execute(
                """UPDATE jobs SET status=%s,progress=%s,log=%s::jsonb,error_message=NULLIF(%s,''),
                metadata=%s::jsonb,updated_at=NOW(),completed_at=CASE WHEN %s IN ('completed','failed','cancelled') THEN NOW() ELSE completed_at END
                WHERE id=%s RETURNING *""",
                (next_status,progress if progress is not None else row["progress"],self._json(logs),error,self._json(merged_metadata),next_status,job_id),
            ).fetchone()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connection() as conn: return conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()

    def cancel_document_jobs(self, document_id: int, reason: str = "文档已删除，任务已取消") -> int:
        """Cancel queued/running ingestion work before document cleanup starts."""
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT id,log,metadata FROM jobs
                WHERE document_id=%s AND status IN ('queued','running') FOR UPDATE""",
                (document_id,),
            ).fetchall()
            for row in rows:
                logs = [*list(row.get("log") or []), reason]
                metadata = {
                    **(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}),
                    "stage": "cancelled",
                    "cancelled_by_document_delete": True,
                }
                conn.execute(
                    """UPDATE jobs SET status='cancelled',log=%s::jsonb,metadata=%s::jsonb,
                    error_message=NULL,updated_at=NOW(),completed_at=NOW() WHERE id=%s""",
                    (self._json(logs), self._json(metadata), row["id"]),
                )
            return len(rows)

    def cancel_project_jobs(self, project_id: int, kinds: list[str], reason: str) -> int:
        """Cancel selected project-wide work whose input snapshot is no longer valid."""
        if not kinds:
            return 0
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT id,log,metadata FROM jobs
                WHERE project_id=%s AND kind=ANY(%s) AND status IN ('queued','running') FOR UPDATE""",
                (project_id, kinds),
            ).fetchall()
            for row in rows:
                logs = [*list(row.get("log") or []), reason]
                metadata = {
                    **(row.get("metadata") if isinstance(row.get("metadata"), dict) else {}),
                    "stage": "cancelled",
                    "cancelled_by_document_change": True,
                }
                conn.execute(
                    """UPDATE jobs SET status='cancelled',log=%s::jsonb,metadata=%s::jsonb,
                    error_message=NULL,updated_at=NOW(),completed_at=NOW() WHERE id=%s""",
                    (self._json(logs), self._json(metadata), row["id"]),
                )
            return len(rows)

    def list_recoverable_jobs(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM jobs WHERE status IN ('queued','running')
                ORDER BY created_at,id"""
            ).fetchall()

    def latest_job(self, project_id: int, kind: str = "") -> dict[str, Any] | None:
        query = "SELECT * FROM jobs WHERE project_id=%s"
        params: list[Any] = [project_id]
        if kind:
            query += " AND kind=%s"
            params.append(kind)
        with self.connection() as conn:
            return conn.execute(query + " ORDER BY updated_at DESC LIMIT 1", params).fetchone()

    def latest_active_job(self, project_id: int, kind: str = "") -> dict[str, Any] | None:
        query = "SELECT * FROM jobs WHERE project_id=%s AND status IN ('queued','running')"
        params: list[Any] = [project_id]
        if kind:
            query += " AND kind=%s"
            params.append(kind)
        query += " ORDER BY CASE WHEN status='running' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1"
        with self.connection() as conn:
            return conn.execute(query, params).fetchone()

    def ensure_base_schema(self, project_id: int, schema_json: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM graph_schemas WHERE project_id=%s AND status='active'", (project_id,)).fetchone()
            if row: return row
            row = conn.execute(
                "INSERT INTO graph_schemas(project_id,version,status,schema_json,created_by,applied_at) VALUES(%s,1,'active',%s::jsonb,'system',NOW()) RETURNING *",
                (project_id,self._json(schema_json)),
            ).fetchone()
            conn.execute("UPDATE projects SET active_schema_version=1,schema_status='base',updated_at=NOW() WHERE id=%s", (project_id,))
            return row

    def get_active_schema(self, project_id: int) -> dict[str, Any] | None:
        with self.connection() as conn: return conn.execute("SELECT * FROM graph_schemas WHERE project_id=%s AND status='active'", (project_id,)).fetchone()

    def create_schema_version(self, project_id: int, schema_json: dict[str, Any], report: dict[str, Any], document_ids: list[int], created_by: str = "llm") -> dict[str, Any]:
        with self.connection() as conn:
            project = conn.execute(
                "SELECT id FROM projects WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            if created_by == "manual":
                active_rows = conn.execute(
                    """SELECT id FROM documents
                    WHERE project_id=%s AND is_deleted=FALSE AND status='indexed'
                    ORDER BY id""",
                    (project_id,),
                ).fetchall()
                active_document_ids = [int(item["id"]) for item in active_rows]
                if sorted(int(item) for item in document_ids) != active_document_ids:
                    raise RuntimeError("项目文档已变化，请重新发起 Schema 评估")
            version = int(conn.execute("SELECT COALESCE(MAX(version),0)+1 AS version FROM graph_schemas WHERE project_id=%s", (project_id,)).fetchone()["version"])
            conn.execute("UPDATE graph_schemas SET status='retired' WHERE project_id=%s AND status='active'", (project_id,))
            row = conn.execute(
                """INSERT INTO graph_schemas(project_id,version,status,schema_json,compatibility_report,source_document_ids,created_by,applied_at)
                VALUES(%s,%s,'active',%s::jsonb,%s::jsonb,%s,%s,NOW()) RETURNING *""",
                (project_id,version,self._json(schema_json),self._json(report),document_ids,created_by),
            ).fetchone()
            conn.execute("UPDATE projects SET active_schema_version=%s,schema_status='adapted',updated_at=NOW() WHERE id=%s", (version,project_id))
            return row

    def activate_base_schema(self, project_id: int, schema_json: dict[str, Any]) -> dict[str, Any]:
        """Activate the immutable project baseline after all document evidence disappears."""
        with self.connection() as conn:
            project = conn.execute(
                "SELECT id FROM projects WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            conn.execute("UPDATE graph_schemas SET status='retired' WHERE project_id=%s", (project_id,))
            row = conn.execute(
                """INSERT INTO graph_schemas(project_id,version,status,schema_json,compatibility_report,
                source_document_ids,created_by,applied_at)
                VALUES(%s,1,'active',%s::jsonb,'{}'::jsonb,'{}','system',NOW())
                ON CONFLICT(project_id,version) DO UPDATE SET status='active',schema_json=EXCLUDED.schema_json,
                compatibility_report='{}'::jsonb,source_document_ids='{}',applied_at=NOW()
                RETURNING *""",
                (project_id, self._json(schema_json)),
            ).fetchone()
            conn.execute(
                """UPDATE projects SET active_schema_version=1,schema_status='base',graph_status='empty',
                updated_at=NOW() WHERE id=%s""",
                (project_id,),
            )
            return row

    def list_schema_versions(self, project_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn: return conn.execute("SELECT * FROM graph_schemas WHERE project_id=%s ORDER BY version DESC", (project_id,)).fetchall()

    def list_canonical_entities(self, project_id: int, entity_type: str = "") -> list[dict[str, Any]]:
        query, params = "SELECT * FROM canonical_entities WHERE project_id=%s AND is_deleted=FALSE", [project_id]
        if entity_type: query += " AND entity_type=%s"; params.append(entity_type)
        with self.connection() as conn: return conn.execute(query + " ORDER BY source_count DESC,name", params).fetchall()

    def count_canonical_entities(self, project_id: int, entity_type: str = "") -> int:
        query = "SELECT COUNT(*) AS count FROM canonical_entities WHERE project_id=%s AND is_deleted=FALSE"
        params: list[Any] = [project_id]
        if entity_type:
            query += " AND (%s='GenericEntity' OR entity_type=%s OR entity_type='GenericEntity')"
            params.extend([entity_type, entity_type])
        with self.connection() as conn:
            return int(conn.execute(query, params).fetchone()["count"])

    def find_exact_entity_candidates(
        self, project_id: int, entity_type: str, normalized_name: str
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT *,
                          CASE WHEN normalized_name=%s THEN 1.0 ELSE 0.99 END AS lexical_score
                   FROM canonical_entities
                   WHERE project_id=%s AND is_deleted=FALSE
                     AND (%s='GenericEntity' OR entity_type=%s OR entity_type='GenericEntity')
                     AND (normalized_name=%s OR %s=ANY(normalized_aliases))
                   ORDER BY (entity_type=%s) DESC,(normalized_name=%s) DESC,source_count DESC,id
                   LIMIT 10""",
                (
                    normalized_name, project_id, entity_type, entity_type,
                    normalized_name, normalized_name, entity_type, normalized_name,
                ),
            ).fetchall()

    def search_entity_name_candidates(
        self,
        project_id: int,
        entity_type: str,
        normalized_name: str,
        limit: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            conn.execute(
                "SELECT set_config('pg_trgm.similarity_threshold',%s,TRUE)",
                (str(max(0.0, min(1.0, threshold))),),
            )
            return conn.execute(
                """SELECT c.*,
                          GREATEST(
                            similarity(c.normalized_name,%s),
                            COALESCE((SELECT MAX(similarity(alias,%s))
                                      FROM unnest(c.normalized_aliases) AS alias),0.0)
                          ) AS lexical_score
                   FROM canonical_entities c
                   WHERE project_id=%s AND is_deleted=FALSE
                     AND (%s='GenericEntity' OR entity_type=%s OR entity_type='GenericEntity')
                     AND (
                       normalized_name %% %s
                       OR EXISTS (SELECT 1 FROM unnest(c.normalized_aliases) AS alias
                                  WHERE similarity(alias,%s)>=%s)
                     )
                   ORDER BY lexical_score DESC,source_count DESC,id
                   LIMIT %s""",
                (
                    normalized_name, normalized_name, project_id, entity_type, entity_type,
                    normalized_name, normalized_name, threshold, limit,
                ),
            ).fetchall()

    def search_entity_vector_candidates(
        self,
        project_id: int,
        entity_type: str,
        model_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        vector_literal = self._vector_literal(query_vector)
        dimension = len(query_vector)
        vector_type = sql.SQL("vector({})").format(sql.Literal(dimension))
        statement = sql.SQL(
            """SELECT c.*,
                      GREATEST(0.0,1.0-(v.embedding_vector::{} <=> %s::{})) AS vector_score
               FROM vector_embeddings v
               JOIN canonical_entities c ON c.id::text=v.object_key
               WHERE v.project_id=%s AND v.object_type='entity' AND v.model_name=%s
                 AND v.dimension=%s AND v.is_deleted=FALSE
                 AND c.project_id=%s AND c.is_deleted=FALSE
                 AND (%s='GenericEntity' OR c.entity_type=%s OR c.entity_type='GenericEntity')
               ORDER BY v.embedding_vector::{} <=> %s::{}
               LIMIT %s"""
        ).format(vector_type, vector_type, vector_type, vector_type)
        with self.connection() as conn:
            conn.execute("SELECT set_config('hnsw.ef_search',%s,TRUE)", (str(Config.pgvector_hnsw_ef_search),))
            conn.execute("SELECT set_config('hnsw.iterative_scan','relaxed_order',TRUE)")
            return conn.execute(
                statement,
                (
                    vector_literal, project_id, model_name, dimension, project_id,
                    entity_type, entity_type, vector_literal, limit,
                ),
            ).fetchall()

    def upsert_canonical_entity(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO canonical_entities(project_id,entity_key,name,normalized_name,entity_type,aliases,
                normalized_aliases,description,schema_version,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(project_id,entity_key) DO UPDATE SET name=EXCLUDED.name,normalized_name=EXCLUDED.normalized_name,
                entity_type=EXCLUDED.entity_type,aliases=(SELECT ARRAY(SELECT DISTINCT unnest(canonical_entities.aliases || EXCLUDED.aliases))),
                normalized_aliases=(SELECT ARRAY(SELECT DISTINCT unnest(canonical_entities.normalized_aliases || EXCLUDED.normalized_aliases))),
                description=CASE WHEN length(EXCLUDED.description)>length(canonical_entities.description) THEN EXCLUDED.description ELSE canonical_entities.description END,
                schema_version=GREATEST(canonical_entities.schema_version,EXCLUDED.schema_version),metadata=canonical_entities.metadata||EXCLUDED.metadata,
                is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (values["project_id"],values["entity_key"],values["name"],values["normalized_name"],values["entity_type"],
                 values.get("aliases") or [],values.get("normalized_aliases") or [],values.get("description") or "",
                 values.get("schema_version") or 1,self._json(values.get("metadata"))),
            ).fetchone()

    def merge_canonical_entity_evidence(
        self,
        entity_id: int,
        aliases: list[str],
        normalized_aliases: list[str],
        description: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """UPDATE canonical_entities SET
                   aliases=(SELECT ARRAY(SELECT DISTINCT unnest(aliases || %s::text[]))),
                   normalized_aliases=(SELECT ARRAY(SELECT DISTINCT unnest(normalized_aliases || %s::text[]))),
                   description=CASE WHEN length(%s)>length(description) THEN %s ELSE description END,
                   updated_at=NOW()
                   WHERE id=%s RETURNING *""",
                (aliases, normalized_aliases, description, description, entity_id),
            ).fetchone()

    def update_canonical_entity_type(self, entity_id: int, entity_type: str) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                "UPDATE canonical_entities SET entity_type=%s,updated_at=NOW() WHERE id=%s RETURNING *",
                (entity_type, entity_id),
            ).fetchone()

    def add_entity_mention(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """INSERT INTO entity_mentions(project_id,document_id,chunk_id,canonical_entity_id,mention_text,normalized_name,entity_type,description,confidence,extraction_version,evidence_text,evidence_start,evidence_end,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(document_id,chunk_id,canonical_entity_id,mention_text) DO UPDATE SET description=EXCLUDED.description,
                confidence=EXCLUDED.confidence,evidence_text=EXCLUDED.evidence_text,evidence_start=EXCLUDED.evidence_start,
                evidence_end=EXCLUDED.evidence_end,metadata=EXCLUDED.metadata,is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (values["project_id"],values["document_id"],values["chunk_id"],values["canonical_entity_id"],values["mention_text"],values["normalized_name"],values["entity_type"],values.get("description") or "",0.5 if values.get("confidence") is None else values.get("confidence"),values.get("extraction_version") or "v2",values.get("evidence_text") or "",values.get("evidence_start"),values.get("evidence_end"),self._json(values.get("metadata"))),
            ).fetchone()
            conn.execute("UPDATE canonical_entities SET source_count=(SELECT COUNT(DISTINCT document_id) FROM entity_mentions WHERE canonical_entity_id=%s AND is_deleted=FALSE),updated_at=NOW() WHERE id=%s", (values["canonical_entity_id"],values["canonical_entity_id"]))
            return row

    def add_entity_resolution_decision(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO entity_resolution_decisions(project_id,document_id,chunk_id,canonical_entity_id,
                mention_text,normalized_name,entity_type,decision,candidate_entity_ids,scores,reason,model_name,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT(document_id,chunk_id,mention_text,entity_type) DO UPDATE SET
                canonical_entity_id=EXCLUDED.canonical_entity_id,decision=EXCLUDED.decision,
                candidate_entity_ids=EXCLUDED.candidate_entity_ids,scores=EXCLUDED.scores,
                reason=EXCLUDED.reason,model_name=EXCLUDED.model_name,metadata=EXCLUDED.metadata,
                is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (
                    values["project_id"], values["document_id"], values["chunk_id"],
                    values["canonical_entity_id"], values["mention_text"], values["normalized_name"],
                    values["entity_type"], values["decision"], values.get("candidate_entity_ids") or [],
                    self._json(values.get("scores")), values.get("reason") or "", values.get("model_name"),
                    self._json(values.get("metadata")),
                ),
            ).fetchone()

    def upsert_relation_candidate(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            return self._upsert_relation_candidate(conn, values)

    def _upsert_relation_candidate(
        self, conn: psycopg.Connection, values: dict[str, Any],
    ) -> dict[str, Any]:
        return conn.execute(
                """INSERT INTO relation_candidates(candidate_id,project_id,document_id,chunk_id,candidate_ordinal,
                source_mention_id,source_name,target_mention_id,target_name,proposed_relation_type,
                proposed_evidence,evidence_start,evidence_end,model_confidence,status,ingestion_run_id,reason,
                corrected_relation_type,orientation,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(candidate_id) DO UPDATE SET project_id=EXCLUDED.project_id,
                document_id=EXCLUDED.document_id,chunk_id=EXCLUDED.chunk_id,
                candidate_ordinal=EXCLUDED.candidate_ordinal,source_mention_id=EXCLUDED.source_mention_id,
                source_name=EXCLUDED.source_name,target_mention_id=EXCLUDED.target_mention_id,
                target_name=EXCLUDED.target_name,proposed_relation_type=EXCLUDED.proposed_relation_type,
                proposed_evidence=EXCLUDED.proposed_evidence,evidence_start=EXCLUDED.evidence_start,
                evidence_end=EXCLUDED.evidence_end,model_confidence=EXCLUDED.model_confidence,
                status=EXCLUDED.status,ingestion_run_id=EXCLUDED.ingestion_run_id,reason=EXCLUDED.reason,
                corrected_relation_type=EXCLUDED.corrected_relation_type,orientation=EXCLUDED.orientation,
                metadata=EXCLUDED.metadata,is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (
                    values["candidate_id"], values["project_id"], values["document_id"], values["chunk_id"],
                    values["candidate_ordinal"], values["source_mention_id"], values.get("source_name") or "",
                    values["target_mention_id"], values.get("target_name") or "",
                    values.get("proposed_relation_type") or "", values.get("proposed_evidence") or "",
                    values.get("evidence_start"), values.get("evidence_end"),
                    0.5 if values.get("model_confidence") is None else values.get("model_confidence"),
                    values.get("status") or UNCERTAIN, values.get("ingestion_run_id") or "",
                    values.get("reason") or "",
                    values.get("corrected_relation_type"), values.get("orientation"),
                    self._json(values.get("metadata")),
                ),
            ).fetchone()

    def add_relation_verification_event(self, values: dict[str, Any]) -> dict[str, Any]:
        """Append one immutable verifier attempt record."""

        with self.connection() as conn:
            return self._add_relation_verification_event(conn, values)

    def _add_relation_verification_event(
        self, conn: psycopg.Connection, values: dict[str, Any],
    ) -> dict[str, Any]:
        return conn.execute(
                """INSERT INTO relation_verification_events(candidate_id,project_id,document_id,chunk_id,
                attempt,model_name,prompt_version,schema_version,ingestion_run_id,request_json,raw_response,verdict,reason,error_category)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s) RETURNING *""",
                (
                    values["candidate_id"], values["project_id"], values["document_id"], values["chunk_id"],
                    values["attempt"], values.get("model_name") or "", values.get("prompt_version") or "",
                    int(values.get("schema_version") or 1), values.get("ingestion_run_id") or "",
                    self._json(values.get("request_json")),
                    self._json(values.get("raw_response")) if values.get("raw_response") is not None else None,
                    values.get("verdict") or None, values.get("reason") or "",
                    values.get("error_category") or "",
                ),
            ).fetchone()

    def add_relation_assertion(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            return self._add_relation_assertion(conn, values)

    def _add_relation_assertion(
        self, conn: psycopg.Connection, values: dict[str, Any],
    ) -> dict[str, Any]:
        row = conn.execute(
                """INSERT INTO relation_assertions(project_id,document_id,chunk_id,source_entity_id,target_entity_id,relation_key,relation_type,description,confidence,extraction_version,ingestion_run_id,evidence_text,evidence_start,evidence_end,verification_status,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(project_id,relation_key) DO UPDATE SET document_id=EXCLUDED.document_id,
                chunk_id=EXCLUDED.chunk_id,source_entity_id=EXCLUDED.source_entity_id,
                target_entity_id=EXCLUDED.target_entity_id,relation_type=EXCLUDED.relation_type,description=EXCLUDED.description,
                confidence=EXCLUDED.confidence,evidence_text=EXCLUDED.evidence_text,evidence_start=EXCLUDED.evidence_start,
                evidence_end=EXCLUDED.evidence_end,verification_status=EXCLUDED.verification_status,
                ingestion_run_id=EXCLUDED.ingestion_run_id,metadata=EXCLUDED.metadata,
                is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (values["project_id"],values["document_id"],values["chunk_id"],values["source_entity_id"],values["target_entity_id"],values["relation_key"],values["relation_type"],values.get("description") or "",0.5 if values.get("confidence") is None else values.get("confidence"),values.get("extraction_version") or "v2",values.get("ingestion_run_id") or "",values.get("evidence_text") or "",values.get("evidence_start"),values.get("evidence_end"),values.get("verification_status") or "unverified",self._json(values.get("metadata"))),
            ).fetchone()
        if not is_publishable_relation(row.get("verification_status"), row.get("metadata")):
            conn.execute(
                """UPDATE vector_embeddings SET is_deleted=TRUE,updated_at=NOW()
                   WHERE project_id=%s AND object_type='relation' AND object_key=%s""",
                (row["project_id"], str(row["id"])),
            )
        return row

    def persist_relation_audit_chain(
        self,
        candidates: list[dict[str, Any]],
        events: list[dict[str, Any]],
        assertions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist one Chunk's relation audit chain in a single PostgreSQL transaction."""

        with self.connection() as conn:
            for values in candidates:
                self._upsert_relation_candidate(conn, values)
            for values in events:
                self._add_relation_verification_event(conn, values)
            return [self._add_relation_assertion(conn, values) for values in assertions]

    def list_project_relations(self, project_id: int) -> list[dict[str, Any]]:
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            return conn.execute(
                """SELECT r.*,s.name AS source_name,s.entity_type AS source_type,t.name AS target_name,t.entity_type AS target_type,
                d.original_filename,c.content AS chunk_content FROM relation_assertions r
                JOIN canonical_entities s ON s.id=r.source_entity_id JOIN canonical_entities t ON t.id=r.target_entity_id
                JOIN documents d ON d.id=r.document_id JOIN chunks c ON c.id=r.chunk_id
                WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                AND """ + clause + " ORDER BY r.id", (project_id, *policy_params)
            ).fetchall()

    def count_publishable_relations(
        self, project_id: int, document_ids: list[int] | None = None,
    ) -> int:
        clause, policy_params = self._publishable_relation_clause("r")
        document_filter = ""
        document_params: list[Any] = []
        if document_ids is not None:
            document_filter = " AND r.document_id=ANY(%s)"
            document_params.append(document_ids)
        with self.connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS count FROM relation_assertions r
                   JOIN documents d ON d.id=r.document_id
                   WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                   """ + document_filter + " AND " + clause,
                (project_id, *document_params, *policy_params),
            ).fetchone()
            return int(row["count"] if row else 0)

    def list_all_project_relations(self, project_id: int) -> list[dict[str, Any]]:
        """Audit-only relation listing. Callers explicitly opt into quarantined states."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT r.*,s.name AS source_name,s.entity_type AS source_type,
                t.name AS target_name,t.entity_type AS target_type,d.original_filename,
                c.content AS chunk_content FROM relation_assertions r
                JOIN canonical_entities s ON s.id=r.source_entity_id
                JOIN canonical_entities t ON t.id=r.target_entity_id
                JOIN documents d ON d.id=r.document_id JOIN chunks c ON c.id=r.chunk_id
                WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                ORDER BY r.id""",
                (project_id,),
            ).fetchall()

    def list_document_mentions(self, document_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT m.*,e.name,e.entity_type,e.graph_vertex_id FROM entity_mentions m
                JOIN canonical_entities e ON e.id=m.canonical_entity_id WHERE m.document_id=%s AND m.is_deleted=FALSE""", (document_id,)
            ).fetchall()

    def list_document_relations(self, document_id: int) -> list[dict[str, Any]]:
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            return conn.execute(
                """SELECT r.*,s.graph_vertex_id AS source_vertex_id,t.graph_vertex_id AS target_vertex_id
                FROM relation_assertions r JOIN canonical_entities s ON s.id=r.source_entity_id
                JOIN canonical_entities t ON t.id=r.target_entity_id
                WHERE r.document_id=%s AND r.is_deleted=FALSE AND """ + clause,
                (document_id, *policy_params),
            ).fetchall()

    def list_all_document_relations(self, document_id: int) -> list[dict[str, Any]]:
        """Audit-only document relation listing including quarantined assertions."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT r.*,s.graph_vertex_id AS source_vertex_id,
                t.graph_vertex_id AS target_vertex_id FROM relation_assertions r
                JOIN canonical_entities s ON s.id=r.source_entity_id
                JOIN canonical_entities t ON t.id=r.target_entity_id
                WHERE r.document_id=%s AND r.is_deleted=FALSE""",
                (document_id,),
            ).fetchall()

    def list_relation_candidates(
        self, project_id: int, document_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Audit-only candidate listing; never use this as a product relation feed."""

        statement = """SELECT * FROM relation_candidates
            WHERE project_id=%s AND is_deleted=FALSE"""
        params: list[Any] = [project_id]
        if document_id is not None:
            statement += " AND document_id=%s"
            params.append(document_id)
        statement += " ORDER BY updated_at DESC,candidate_ordinal"
        with self.connection() as conn:
            return conn.execute(statement, params).fetchall()

    def set_entity_graph_id(self, entity_id: int, graph_vertex_id: str) -> None:
        with self.connection() as conn: conn.execute("UPDATE canonical_entities SET graph_vertex_id=%s,updated_at=NOW() WHERE id=%s", (graph_vertex_id,entity_id))

    def set_mention_graph_id(self, mention_id: int, edge_id: str) -> None:
        with self.connection() as conn: conn.execute("UPDATE entity_mentions SET graph_edge_id=%s,updated_at=NOW() WHERE id=%s", (edge_id,mention_id))

    def set_relation_graph_id(self, relation_id: int, edge_id: str) -> None:
        with self.connection() as conn: conn.execute("UPDATE relation_assertions SET graph_edge_id=%s,updated_at=NOW() WHERE id=%s", (edge_id,relation_id))

    @staticmethod
    def _retire_orphan_entities(conn: psycopg.Connection) -> None:
        conn.execute(
            """UPDATE canonical_entities e SET source_count=COALESCE(x.cnt,0),is_deleted=COALESCE(x.cnt,0)=0,updated_at=NOW()
            FROM (SELECT e2.id,COUNT(DISTINCT m.document_id) FILTER (WHERE m.is_deleted=FALSE) AS cnt
            FROM canonical_entities e2 LEFT JOIN entity_mentions m ON m.canonical_entity_id=e2.id GROUP BY e2.id) x WHERE e.id=x.id"""
        )
        conn.execute(
            """UPDATE vector_embeddings v SET is_deleted=e.is_deleted,updated_at=NOW()
            FROM canonical_entities e WHERE v.object_type='entity' AND v.object_key=e.id::text"""
        )

    def retire_orphan_entities(self) -> None:
        with self.connection() as conn:
            self._retire_orphan_entities(conn)

    def statistics(self, project_id: int) -> dict[str, int]:
        clause, policy_params = self._publishable_relation_clause("r")
        statement = f"""SELECT
            (SELECT COUNT(*) FROM documents WHERE project_id=%s AND is_deleted=FALSE) documents,
            (SELECT COUNT(*) FROM chunks WHERE project_id=%s AND is_deleted=FALSE) chunks,
            (SELECT COUNT(*) FROM canonical_entities WHERE project_id=%s AND is_deleted=FALSE) entities,
            (SELECT COUNT(*) FROM relation_assertions r WHERE r.project_id=%s
                AND r.is_deleted=FALSE AND {clause}) relations,
            (SELECT COUNT(*) FROM relation_candidates rc WHERE rc.project_id=%s
                AND rc.is_deleted=FALSE) relation_candidates"""
        with self.connection() as conn:
            row = conn.execute(
                statement,
                (project_id,project_id,project_id,project_id,*policy_params,project_id),
            ).fetchone(); return {key:int(value or 0) for key,value in row.items()}

    def search_graph_text(self, project_id: int, query: str, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
        pattern = f"%{query}%"
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            entities = conn.execute(
                """SELECT id,name,entity_type,description,aliases,source_count FROM canonical_entities
                WHERE project_id=%s AND is_deleted=FALSE AND (name ILIKE %s OR description ILIKE %s OR %s=ANY(aliases))
                ORDER BY source_count DESC,name LIMIT %s""", (project_id,pattern,pattern,query,limit)
            ).fetchall()
            relations = conn.execute(
                """SELECT r.id,s.name AS source_name,r.relation_type,t.name AS target_name,r.description,r.confidence,
                d.original_filename,d.storage_path,c.chunk_key FROM relation_assertions r
                JOIN canonical_entities s ON s.id=r.source_entity_id JOIN canonical_entities t ON t.id=r.target_entity_id
                JOIN documents d ON d.id=r.document_id JOIN chunks c ON c.id=r.chunk_id
                WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                AND """ + clause + """
                AND (s.name ILIKE %s OR t.name ILIKE %s OR r.relation_type ILIKE %s OR r.description ILIKE %s)
                ORDER BY r.confidence DESC,r.id LIMIT %s""",
                (project_id,*policy_params,pattern,pattern,pattern,pattern,limit)
            ).fetchall()
        return {"entities": entities, "relations": relations}

    def get_entities_by_ids(self, project_id: int, ids: list[int]) -> list[dict[str, Any]]:
        if not ids: return []
        with self.connection() as conn:
            return conn.execute(
                """SELECT id,name,entity_type,description,aliases,source_count FROM canonical_entities
                WHERE project_id=%s AND id=ANY(%s) AND is_deleted=FALSE""", (project_id,ids)
            ).fetchall()

    def get_relations_by_ids(self, project_id: int, ids: list[int]) -> list[dict[str, Any]]:
        if not ids: return []
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            return conn.execute(
                """SELECT r.id,s.name AS source_name,r.relation_type,t.name AS target_name,r.description,r.confidence,
                r.verification_status,d.original_filename,d.storage_path,c.chunk_key FROM relation_assertions r
                JOIN canonical_entities s ON s.id=r.source_entity_id JOIN canonical_entities t ON t.id=r.target_entity_id
                JOIN documents d ON d.id=r.document_id JOIN chunks c ON c.id=r.chunk_id
                WHERE r.project_id=%s AND r.id=ANY(%s) AND r.is_deleted=FALSE
                AND d.is_deleted=FALSE AND """ + clause,
                (project_id,ids,*policy_params)
            ).fetchall()
