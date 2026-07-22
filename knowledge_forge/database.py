"""PostgreSQL persistence for projects, retrieval, graph provenance and jobs."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import re
import uuid
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .config import Config
from .entity_identity import normalize_name
from .graph_manifest import GRAPH_MANIFEST_PROTOCOL
from .graph_policy import (
    GRAPH_EMPTY,
    GRAPH_FAILED,
    GRAPH_POLICY_HASH,
    GRAPH_POLICY_VERSION,
    GRAPH_READY,
    GRAPH_REBUILDING,
    GRAPH_STALE,
    graph_policy_hash,
    is_current_graph,
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
from .policy import POLICY_AXES, normalize_policy_set
from .migrations import MigrationRunner
from .audit import redact_audit_payload


LOGGER = logging.getLogger(__name__)


class ContentRevisionError(RuntimeError):
    """Base error for immutable document content revisions."""


class RevisionManifestMismatch(ContentRevisionError):
    """The same ingestion run was retried with different immutable inputs."""


class StaleContentRevision(ContentRevisionError):
    """A late worker attempted to mutate or activate an obsolete revision."""


class RevisionLeaseLost(StaleContentRevision):
    """The worker no longer owns the revision lease."""


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
    active_policy_hash TEXT NOT NULL DEFAULT '',
    entity_extraction_policy_version TEXT NOT NULL DEFAULT 'entity-extraction-p0-v1',
    entity_resolution_policy_version TEXT NOT NULL DEFAULT 'entity-resolution-p0-v1',
    relation_candidate_policy_version TEXT NOT NULL DEFAULT 'relation-candidate-p0-v1',
    relation_classifier_policy_version TEXT NOT NULL DEFAULT 'relation-classifier-p0-v1',
    verification_policy_version TEXT NOT NULL DEFAULT 'relation-verifier-p0-v1',
    schema_policy_version TEXT NOT NULL DEFAULT 'schema-policy-p0-v1',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','archived','failed')),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), is_deleted BOOLEAN NOT NULL DEFAULT FALSE, deleted_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_projects_name_active ON projects(name) WHERE is_deleted=FALSE;
CREATE TABLE IF NOT EXISTS policy_snapshots (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    policy_set JSONB NOT NULL, policy_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('building','ready','active','failed','rolled_back')),
    source_generation BIGINT NOT NULL, expected_documents INTEGER NOT NULL DEFAULT 0,
    completed_documents INTEGER NOT NULL DEFAULT 0, snapshot_manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb, error_message TEXT NOT NULL DEFAULT '',
    coverage_complete BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ, activated_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_policy_snapshots_project
    ON policy_snapshots(project_id,status,created_at DESC);
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
CREATE TABLE IF NOT EXISTS policy_snapshot_documents (
    snapshot_id TEXT NOT NULL REFERENCES policy_snapshots(id),
    document_id BIGINT NOT NULL REFERENCES documents(id),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','ready','failed')),
    shadow_run_count INTEGER NOT NULL DEFAULT 0, error_message TEXT NOT NULL DEFAULT '',
    completed_at TIMESTAMPTZ, PRIMARY KEY(snapshot_id,document_id)
);
CREATE INDEX IF NOT EXISTS idx_policy_snapshot_documents_status
    ON policy_snapshot_documents(snapshot_id,status,document_id);
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

CREATE TABLE IF NOT EXISTS canonical_entity_documents (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    canonical_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    document_id BIGINT NOT NULL REFERENCES documents(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE(canonical_entity_id,document_id)
);
CREATE INDEX IF NOT EXISTS idx_canonical_entity_documents_project
    ON canonical_entity_documents(project_id,canonical_entity_id,is_deleted);
CREATE INDEX IF NOT EXISTS idx_canonical_entity_documents_document
    ON canonical_entity_documents(document_id,is_deleted);

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

CREATE TABLE IF NOT EXISTS canonical_entity_aliases (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    canonical_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    entity_type TEXT NOT NULL DEFAULT 'GenericEntity', alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL, alias_kind TEXT NOT NULL DEFAULT 'observed',
    active BOOLEAN NOT NULL DEFAULT TRUE, evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    policy_version TEXT NOT NULL DEFAULT 'entity-resolution-p0-v1',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id,canonical_entity_id,normalized_alias,alias_kind)
);
CREATE INDEX IF NOT EXISTS idx_entity_alias_exact
    ON canonical_entity_aliases(project_id,normalized_alias,entity_type,canonical_entity_id)
    WHERE active=TRUE;
CREATE INDEX IF NOT EXISTS idx_entity_alias_trgm ON canonical_entity_aliases
    USING gin(normalized_alias gin_trgm_ops) WHERE active=TRUE;

ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS mention_key TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_mentions ADD COLUMN IF NOT EXISTS extraction_policy_version TEXT NOT NULL DEFAULT 'entity-extraction-p0-v1';
CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_mentions_stable_key
    ON entity_mentions(project_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE;

CREATE TABLE IF NOT EXISTS entity_type_assertions (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT NOT NULL REFERENCES documents(id), chunk_id BIGINT NOT NULL REFERENCES chunks(id),
    mention_id BIGINT REFERENCES entity_mentions(id), mention_key TEXT NOT NULL,
    entity_type TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    evidence_text TEXT NOT NULL, evidence_start INTEGER, evidence_end INTEGER,
    policy_version TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'model',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id,mention_key,entity_type,policy_version,source)
);
CREATE INDEX IF NOT EXISTS idx_entity_type_assertions_mention
    ON entity_type_assertions(project_id,mention_key,created_at);

CREATE TABLE IF NOT EXISTS entity_resolution_events (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    event_type TEXT NOT NULL CHECK (event_type IN ('assign','merge','split','undo','manual','cannot_link')),
    source_entity_id BIGINT REFERENCES canonical_entities(id),
    target_entity_id BIGINT REFERENCES canonical_entities(id),
    policy_version TEXT NOT NULL, model_name TEXT, operator TEXT NOT NULL DEFAULT 'system',
    reason TEXT NOT NULL, scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb, before_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    after_state JSONB NOT NULL DEFAULT '{}'::jsonb, reversible BOOLEAN NOT NULL DEFAULT TRUE,
    reversed_by_event_id TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_entity_resolution_events_project
    ON entity_resolution_events(project_id,created_at DESC);

CREATE TABLE IF NOT EXISTS entity_assignments (
    id BIGSERIAL PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    mention_id BIGINT NOT NULL REFERENCES entity_mentions(id),
    canonical_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    assignment_generation BIGINT NOT NULL, policy_version TEXT NOT NULL,
    event_id TEXT REFERENCES entity_resolution_events(id), reason TEXT NOT NULL DEFAULT '',
    active BOOLEAN NOT NULL DEFAULT TRUE, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_assignment_active
    ON entity_assignments(project_id,mention_id) WHERE active=TRUE;
CREATE INDEX IF NOT EXISTS idx_entity_assignments_canonical
    ON entity_assignments(project_id,canonical_entity_id,active);

CREATE TABLE IF NOT EXISTS entity_cannot_links (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    left_entity_id BIGINT REFERENCES canonical_entities(id),
    right_entity_id BIGINT REFERENCES canonical_entities(id),
    left_mention_key TEXT NOT NULL DEFAULT '', right_mention_key TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL, evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    policy_version TEXT NOT NULL, operator TEXT NOT NULL DEFAULT 'system',
    active BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (left_entity_id IS NOT NULL OR left_mention_key<>''),
    CHECK (right_entity_id IS NOT NULL OR right_mention_key<>'')
);
CREATE INDEX IF NOT EXISTS idx_entity_cannot_links_project
    ON entity_cannot_links(project_id,active,left_entity_id,right_entity_id);

CREATE TABLE IF NOT EXISTS canonical_entity_redirects (
    project_id BIGINT NOT NULL REFERENCES projects(id),
    source_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    target_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    event_id TEXT NOT NULL REFERENCES entity_resolution_events(id),
    active BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(project_id,source_entity_id), CHECK(source_entity_id<>target_entity_id)
);

CREATE TABLE IF NOT EXISTS entity_shadow_namespaces (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    snapshot_id TEXT REFERENCES policy_snapshots(id), policy_set JSONB NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('building','ready','failed','deleted')),
    document_order BIGINT[] NOT NULL DEFAULT '{}', metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS entity_shadow_canonicals (
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id), canonical_key TEXT NOT NULL,
    name TEXT NOT NULL, normalized_name TEXT NOT NULL, primary_type TEXT NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT '{}', metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY(namespace_id,canonical_key)
);
CREATE TABLE IF NOT EXISTS entity_shadow_assignments (
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id), mention_key TEXT NOT NULL,
    canonical_key TEXT NOT NULL, document_id BIGINT NOT NULL, chunk_id BIGINT NOT NULL,
    entity_type TEXT NOT NULL, policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, PRIMARY KEY(namespace_id,mention_key)
);

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

CREATE TABLE IF NOT EXISTS canonical_relation_projections (
    project_id BIGINT NOT NULL REFERENCES projects(id),
    assertion_id BIGINT NOT NULL REFERENCES relation_assertions(id),
    source_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    target_entity_id BIGINT NOT NULL REFERENCES canonical_entities(id),
    assignment_generation BIGINT NOT NULL, policy_hash TEXT NOT NULL,
    publishable BOOLEAN NOT NULL DEFAULT TRUE, quarantine_reason TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(project_id,assertion_id)
);
CREATE INDEX IF NOT EXISTS idx_relation_projection_endpoints
    ON canonical_relation_projections(project_id,source_entity_id,target_entity_id,publishable);

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

CREATE TABLE IF NOT EXISTS mention_relation_assertions (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT NOT NULL REFERENCES documents(id), chunk_id BIGINT NOT NULL REFERENCES chunks(id),
    candidate_id TEXT NOT NULL REFERENCES relation_candidates(candidate_id),
    canonical_assertion_id BIGINT REFERENCES relation_assertions(id),
    source_mention_key TEXT NOT NULL, target_mention_key TEXT NOT NULL,
    predicate TEXT NOT NULL, direction TEXT NOT NULL DEFAULT 'forward',
    evidence_text TEXT NOT NULL, evidence_start INTEGER, evidence_end INTEGER,
    extraction_policy_version TEXT NOT NULL, verification_policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_mention_relation_assertion_scope
    ON mention_relation_assertions(project_id,document_id,chunk_id);
CREATE INDEX IF NOT EXISTS idx_mention_relation_assertion_endpoints
    ON mention_relation_assertions(project_id,source_mention_key,target_mention_key);
CREATE INDEX IF NOT EXISTS idx_mention_relation_assertion_canonical
    ON mention_relation_assertions(project_id,canonical_assertion_id);

CREATE TABLE IF NOT EXISTS relation_verification_batches (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT NOT NULL REFERENCES documents(id), chunk_id BIGINT NOT NULL,
    attempt INTEGER NOT NULL, model_name TEXT NOT NULL, prompt_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL, ingestion_run_id TEXT NOT NULL DEFAULT '',
    request_json JSONB NOT NULL DEFAULT '{}'::jsonb, raw_response JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_relation_verification_batch_scope
    ON relation_verification_batches(project_id,document_id,chunk_id,created_at);

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
ALTER TABLE projects ADD COLUMN IF NOT EXISTS active_policy_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS model_max_tokens INTEGER NOT NULL DEFAULT 32768;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS chunk_max_tokens INTEGER NOT NULL DEFAULT 512;
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
ALTER TABLE relation_verification_events ADD COLUMN IF NOT EXISTS batch_id TEXT
    REFERENCES relation_verification_batches(id);
CREATE INDEX IF NOT EXISTS idx_relation_verification_event_batch
    ON relation_verification_events(batch_id) WHERE batch_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS relation_shadow_runs (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT NOT NULL REFERENCES documents(id), chunk_id BIGINT NOT NULL REFERENCES chunks(id),
    control_policy_hash TEXT NOT NULL, treatment_policy_hash TEXT NOT NULL,
    policy_set JSONB NOT NULL, status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running','completed','failed','deferred')),
    raw_pairs INTEGER NOT NULL DEFAULT 0, blocked_pairs INTEGER NOT NULL DEFAULT 0,
    admitted_candidates INTEGER NOT NULL DEFAULT 0, classified INTEGER NOT NULL DEFAULT 0,
    deferred INTEGER NOT NULL DEFAULT 0, model_calls INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
    snapshot_id TEXT REFERENCES policy_snapshots(id),
    error_message TEXT NOT NULL DEFAULT '', metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_relation_shadow_runs_scope
    ON relation_shadow_runs(project_id,document_id,chunk_id,created_at DESC);
CREATE TABLE IF NOT EXISTS relation_shadow_decisions (
    id BIGSERIAL PRIMARY KEY, shadow_run_id TEXT NOT NULL REFERENCES relation_shadow_runs(id),
    candidate_id TEXT NOT NULL, source_mention_id TEXT NOT NULL, target_mention_id TEXT NOT NULL,
    allowed_predicates TEXT[] NOT NULL DEFAULT '{}', predicate TEXT NOT NULL,
    direction TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL,
    status TEXT NOT NULL, request_batch_id TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), UNIQUE(shadow_run_id,candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_relation_shadow_decisions_run
    ON relation_shadow_decisions(shadow_run_id,status,candidate_id);

CREATE TABLE IF NOT EXISTS quality_evaluation_runs (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    snapshot_id TEXT REFERENCES policy_snapshots(id), control_policy_hash TEXT NOT NULL,
    treatment_policy_hash TEXT NOT NULL, dataset TEXT NOT NULL, dataset_version TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('dev','holdout','combined')),
    manifest JSONB NOT NULL, manifest_valid BOOLEAN NOT NULL DEFAULT FALSE,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb, denominators JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running','completed','failed')),
    error_message TEXT NOT NULL DEFAULT '', created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_quality_evaluation_project
    ON quality_evaluation_runs(project_id,created_at DESC);

CREATE TABLE IF NOT EXISTS policy_activation_gates (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    snapshot_id TEXT NOT NULL REFERENCES policy_snapshots(id), evaluation_run_id TEXT NOT NULL
        REFERENCES quality_evaluation_runs(id), manifest_valid BOOLEAN NOT NULL DEFAULT FALSE,
    engineering_complete BOOLEAN NOT NULL DEFAULT FALSE,
    production_approved BOOLEAN NOT NULL DEFAULT FALSE,
    failed_gates TEXT[] NOT NULL DEFAULT '{}', gate_report JSONB NOT NULL DEFAULT '{}'::jsonb,
    approved_by TEXT, approved_at TIMESTAMPTZ, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id,snapshot_id)
);

CREATE TABLE IF NOT EXISTS policy_rollouts (
    id TEXT PRIMARY KEY, project_id BIGINT NOT NULL REFERENCES projects(id),
    snapshot_id TEXT NOT NULL REFERENCES policy_snapshots(id),
    rollout_stage TEXT NOT NULL CHECK (rollout_stage IN
        ('control','shadow','canary','limited','expanded','new_project_default','migrated','rolled_back')),
    previous_snapshot_id TEXT REFERENCES policy_snapshots(id),
    rollback_snapshot_id TEXT REFERENCES policy_snapshots(id),
    percentage DOUBLE PRECISION NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'planned' CHECK (status IN ('planned','active','completed','failed','rolled_back')),
    reason TEXT NOT NULL DEFAULT '', metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_policy_rollout_project
    ON policy_rollouts(project_id,created_at DESC);

ALTER TABLE projects ADD COLUMN IF NOT EXISTS entity_extraction_policy_version TEXT NOT NULL DEFAULT 'entity-extraction-p0-v1';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS entity_resolution_policy_version TEXT NOT NULL DEFAULT 'entity-resolution-p0-v1';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS relation_candidate_policy_version TEXT NOT NULL DEFAULT 'relation-candidate-p0-v1';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS relation_classifier_policy_version TEXT NOT NULL DEFAULT 'relation-classifier-p0-v1';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS verification_policy_version TEXT NOT NULL DEFAULT 'relation-verifier-p0-v1';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS schema_policy_version TEXT NOT NULL DEFAULT 'schema-policy-p0-v1';
INSERT INTO canonical_entity_documents(project_id,canonical_entity_id,document_id)
SELECT m.project_id,m.canonical_entity_id,m.document_id
FROM entity_mentions m
WHERE m.is_deleted=FALSE
GROUP BY m.project_id,m.canonical_entity_id,m.document_id
ON CONFLICT(canonical_entity_id,document_id) DO UPDATE
SET project_id=EXCLUDED.project_id,is_deleted=FALSE,updated_at=NOW();
INSERT INTO canonical_entity_aliases(
    project_id,canonical_entity_id,entity_type,alias,normalized_alias,alias_kind,evidence,policy_version)
SELECT e.project_id,e.id,e.entity_type,e.name,e.normalized_name,'canonical_name','{}'::jsonb,
       'entity-resolution-p0-v1'
FROM canonical_entities e
WHERE e.normalized_name<>''
ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind) DO UPDATE
SET alias=EXCLUDED.alias,entity_type=EXCLUDED.entity_type,active=TRUE,updated_at=NOW();
INSERT INTO canonical_entity_aliases(
    project_id,canonical_entity_id,entity_type,alias,normalized_alias,alias_kind,evidence,policy_version)
SELECT e.project_id,e.id,e.entity_type,pair.alias,pair.normalized,'observed','{}'::jsonb,
       'entity-resolution-p0-v1'
FROM canonical_entities e
CROSS JOIN LATERAL unnest(e.aliases,e.normalized_aliases) AS pair(alias,normalized)
WHERE pair.normalized<>''
ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind) DO UPDATE
SET alias=EXCLUDED.alias,entity_type=EXCLUDED.entity_type,active=TRUE,updated_at=NOW();
INSERT INTO entity_assignments(
    project_id,mention_id,canonical_entity_id,assignment_generation,policy_version,reason,active)
SELECT m.project_id,m.id,m.canonical_entity_id,1,'entity-resolution-p0-v1','legacy backfill',TRUE
FROM entity_mentions m
WHERE m.is_deleted=FALSE
ON CONFLICT(project_id,mention_id) WHERE active=TRUE DO NOTHING;
WITH source_keys AS (
    SELECT DISTINCT ON (fact.id) fact.id,mention.mention_key
    FROM mention_relation_assertions fact
    JOIN relation_assertions assertion ON assertion.id=fact.canonical_assertion_id
    JOIN entity_mentions mention
      ON mention.project_id=fact.project_id AND mention.document_id=fact.document_id
     AND mention.chunk_id=fact.chunk_id
     AND mention.canonical_entity_id=assertion.source_entity_id
     AND mention.is_deleted=FALSE AND mention.mention_key<>''
    WHERE fact.source_mention_key !~ '^mention_'
    ORDER BY fact.id,mention.id
)
UPDATE mention_relation_assertions fact SET source_mention_key=source_keys.mention_key
FROM source_keys WHERE fact.id=source_keys.id;
WITH target_keys AS (
    SELECT DISTINCT ON (fact.id) fact.id,mention.mention_key
    FROM mention_relation_assertions fact
    JOIN relation_assertions assertion ON assertion.id=fact.canonical_assertion_id
    JOIN entity_mentions mention
      ON mention.project_id=fact.project_id AND mention.document_id=fact.document_id
     AND mention.chunk_id=fact.chunk_id
     AND mention.canonical_entity_id=assertion.target_entity_id
     AND mention.is_deleted=FALSE AND mention.mention_key<>''
    WHERE fact.target_mention_key !~ '^mention_'
    ORDER BY fact.id,mention.id
)
UPDATE mention_relation_assertions fact SET target_mention_key=target_keys.mention_key
FROM target_keys WHERE fact.id=target_keys.id;
UPDATE canonical_entities e SET source_count=s.cnt,updated_at=NOW()
FROM (
    SELECT canonical_entity_id,COUNT(*)::INTEGER AS cnt
    FROM canonical_entity_documents WHERE is_deleted=FALSE GROUP BY canonical_entity_id
) s WHERE e.id=s.canonical_entity_id AND e.source_count<>s.cnt;
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
"""


class Database:
    _hnsw_cache: OrderedDict[tuple[str, str, str, int], None] = OrderedDict()
    _hnsw_cache_lock = threading.RLock()
    _hnsw_cache_limit = 256
    _revision_context: ContextVar[dict[str, str] | None] = ContextVar(
        "knowledge_forge_revision_context", default=None,
    )

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
            migrations = MigrationRunner()
            migrations.ensure_table(conn)
            fresh_install = not bool(conn.execute(
                "SELECT to_regclass('projects') AS table_name"
            ).fetchone()["table_name"])
            revision_schema_applied = migrations.is_applied(
                conn, "0001_document_content_revisions",
            )
            if not revision_schema_applied:
                conn.execute(SCHEMA_SQL)
                conn.commit()
            migrations.apply(conn, startup_safe_only=not fresh_install)
            if not migrations.is_applied(conn, "0001_document_content_revisions"):
                raise RuntimeError(
                    "检测到待执行的重迁移；请先运行 "
                    "`.venv/bin/python scripts/manage_migrations.py apply`，再启动服务"
                )
            conn.execute(
                "ALTER TABLE IF EXISTS canonical_entities "
                "ADD COLUMN IF NOT EXISTS normalized_aliases TEXT[] NOT NULL DEFAULT '{}'"
            )
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
        return normalize_name(value)

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
    def _column_exists(conn: psycopg.Connection, table: str, column: str) -> bool:
        if not isinstance(conn, psycopg.Connection):
            return False
        row = conn.execute(
            """SELECT EXISTS(
                 SELECT 1 FROM pg_attribute
                 WHERE attrelid=to_regclass(%s) AND attname=%s
                   AND attnum>0 AND NOT attisdropped
               ) AS present""",
            (table, column),
        ).fetchone()
        return bool(row and row.get("present"))

    @classmethod
    def _reset_hnsw_cache_for_tests(cls) -> None:
        with cls._hnsw_cache_lock:
            cls._hnsw_cache.clear()

    def _ensure_hnsw_index(self, conn: psycopg.Connection, model_name: str, dimension: int) -> None:
        if dimension <= 0 or dimension > Config.pgvector_hnsw_max_dimensions:
            increment("hnsw_degraded_exact_scan")
            LOGGER.info("向量维度 %s 不适合 HNSW vector 索引，将使用精确 pgvector 检索", dimension)
            return
        schema_row = conn.execute("SELECT current_schema() AS schema").fetchone()
        schema_name = str((schema_row or {}).get("schema") or "public")
        database_name = str(self.config.postgres.get("dbname") or "")
        cache_key = (database_name, schema_name, str(model_name), int(dimension))
        with self._hnsw_cache_lock:
            if cache_key in self._hnsw_cache:
                self._hnsw_cache.move_to_end(cache_key)
                increment("hnsw_cache_hit")
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
        with self._hnsw_cache_lock:
            if cache_key in self._hnsw_cache:
                self._hnsw_cache.move_to_end(cache_key)
                increment("hnsw_cache_hit")
                return
            conn.execute(statement)
            self._hnsw_cache[cache_key] = None
            self._hnsw_cache.move_to_end(cache_key)
            while len(self._hnsw_cache) > self._hnsw_cache_limit:
                self._hnsw_cache.popitem(last=False)
            increment("hnsw_index_ensured")

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        increment("db_connection_count")
        with psycopg.connect(**self.config.postgres, row_factory=dict_row) as conn:
            revision = self._revision_context.get()
            if revision:
                conn.execute(
                    "SELECT set_config('knowledge_forge.revision_id',%s,TRUE),"
                    "set_config('knowledge_forge.ingestion_run_id',%s,TRUE),"
                    "set_config('knowledge_forge.revision_lease_token',%s,TRUE)",
                    (
                        revision["revision_id"], revision["ingestion_run_id"],
                        revision["lease_token"],
                    ),
                )
            if current_metrics() is None:
                yield conn
            else:
                yield InstrumentedConnection(conn)

    @contextmanager
    def content_revision_context(
        self, revision_id: str, ingestion_run_id: str, lease_token: str,
    ) -> Iterator[None]:
        """Attach a revision lease to every short transaction in this worker."""

        token = self._revision_context.set({
            "revision_id": revision_id,
            "ingestion_run_id": ingestion_run_id,
            "lease_token": lease_token,
        })
        try:
            yield
        finally:
            self._revision_context.reset(token)

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
    def _audit_json(value: Any) -> str:
        payload = {} if value is None else value
        if Config.audit_redaction_enabled:
            payload = redact_audit_payload(payload)
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(
            {} if value is None else value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

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
                 AND (graph_policy_version<>%s
                      OR graph_policy_hash<>COALESCE(NULLIF(active_policy_hash,''),%s)
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
                """INSERT INTO projects(
                   project_key,name,description,llm_model,embedding_model,graph_namespace_id)
                VALUES(%s,%s,%s,%s,%s,%s) RETURNING *""",
                (
                    f"pending-{uuid.uuid4().hex}", name, description, llm_model,
                    embedding_model, str(uuid.uuid4()),
                ),
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
        namespace_id: str,
        lease_token: str,
        manifest_digest: str,
        empty: bool = False,
    ) -> dict[str, Any] | None:
        """Publish a rebuild only if its immutable generation is still current."""

        status = GRAPH_EMPTY if empty else GRAPH_READY
        with self.connection() as conn:
            lease = conn.execute(
                """SELECT id FROM graph_leases
                   WHERE project_id=%s AND namespace_id=%s AND graph_generation=%s
                     AND lease_token=%s AND status='active' AND expires_at>NOW()
                   FOR UPDATE""",
                (project_id, namespace_id, target_generation, lease_token),
            ).fetchone()
            if not lease:
                return None
            manifests = conn.execute(
                """SELECT role,content_digest,protocol,policy_hash FROM graph_content_manifests
                   WHERE project_id=%s AND namespace_id=%s AND graph_generation=%s
                     AND lease_token=%s ORDER BY role""",
                (project_id, namespace_id, target_generation, lease_token),
            ).fetchall()
            if (
                {row["role"] for row in manifests} != {"expected", "writer", "actual"}
                or any(row["content_digest"] != manifest_digest for row in manifests)
                or any(row["policy_hash"] != policy_hash for row in manifests)
                or any(row["protocol"] != GRAPH_MANIFEST_PROTOCOL for row in manifests)
            ):
                return None
            return conn.execute(
                """UPDATE projects
                   SET graph_status=%s,graph_policy_version=%s,graph_policy_hash=%s,
                       active_policy_hash=%s,
                       graph_ready_generation=%s,graph_error='',updated_at=NOW()
                   WHERE id=%s AND is_deleted=FALSE AND graph_generation=%s
                     AND graph_namespace_id=%s
                   RETURNING *""",
                (
                    status,
                    GRAPH_POLICY_VERSION,
                    policy_hash,
                    policy_hash,
                    target_generation,
                    project_id,
                    target_generation,
                    namespace_id,
                ),
            ).fetchone()

    def acquire_graph_lease(
        self, project_id: int, namespace_id: str, graph_generation: int,
        owner: str, *, ttl_seconds: int = 300,
    ) -> dict[str, Any]:
        """Acquire one cross-process generation lease in a short transaction."""

        lease_token = uuid.uuid4().hex
        lease_id = uuid.uuid4().hex
        with self.connection() as conn:
            project = conn.execute(
                """SELECT graph_namespace_id,graph_generation FROM projects
                   WHERE id=%s AND is_deleted=FALSE FOR UPDATE""", (project_id,),
            ).fetchone()
            if not project or str(project["graph_namespace_id"]) != str(namespace_id):
                raise RuntimeError("graph lease namespace does not own project")
            if int(project["graph_generation"]) != int(graph_generation):
                raise RuntimeError("graph lease generation is stale")
            conn.execute(
                """UPDATE graph_leases SET status='superseded',released_at=NOW()
                   WHERE project_id=%s AND namespace_id=%s AND status='active'
                     AND graph_generation<%s""",
                (project_id, namespace_id, graph_generation),
            )
            active = conn.execute(
                """SELECT id,expires_at FROM graph_leases
                   WHERE project_id=%s AND namespace_id=%s AND graph_generation=%s
                     AND status='active' FOR UPDATE""",
                (project_id, namespace_id, graph_generation),
            ).fetchone()
            if active and active["expires_at"] > datetime.now(timezone.utc):
                raise RuntimeError("graph lease is already owned by another worker")
            if active:
                conn.execute(
                    "UPDATE graph_leases SET status='expired',released_at=NOW() WHERE id=%s",
                    (active["id"],),
                )
            attempt = conn.execute(
                """SELECT COALESCE(MAX(attempt),0)+1 AS attempt FROM graph_leases
                   WHERE project_id=%s AND namespace_id=%s AND graph_generation=%s""",
                (project_id, namespace_id, graph_generation),
            ).fetchone()["attempt"]
            return conn.execute(
                """INSERT INTO graph_leases(
                   id,project_id,namespace_id,graph_generation,owner,lease_token,status,
                   attempt,expires_at)
                   VALUES(%s,%s,%s,%s,%s,%s,'active',%s,NOW()+(%s*INTERVAL '1 second'))
                   RETURNING *""",
                (
                    lease_id, project_id, namespace_id, graph_generation, owner,
                    lease_token, attempt, max(10, int(ttl_seconds)),
                ),
            ).fetchone()

    def assert_graph_lease(
        self, project_id: int, namespace_id: str, graph_generation: int, lease_token: str,
    ) -> bool:
        with self.connection() as conn:
            return bool(conn.execute(
                """SELECT 1 FROM graph_leases
                   WHERE project_id=%s AND namespace_id=%s AND graph_generation=%s
                     AND lease_token=%s AND status='active' AND expires_at>NOW()""",
                (project_id, namespace_id, graph_generation, lease_token),
            ).fetchone())

    def heartbeat_graph_lease(self, lease_id: str, lease_token: str, *, ttl_seconds: int = 300) -> bool:
        with self.connection() as conn:
            return bool(conn.execute(
                """UPDATE graph_leases SET heartbeat_at=NOW(),
                   expires_at=NOW()+(%s*INTERVAL '1 second')
                   WHERE id=%s AND lease_token=%s AND status='active' AND expires_at>NOW()
                   RETURNING id""",
                (max(10, int(ttl_seconds)), lease_id, lease_token),
            ).fetchone())

    def release_graph_lease(self, lease_id: str, lease_token: str) -> bool:
        with self.connection() as conn:
            return bool(conn.execute(
                """UPDATE graph_leases SET status='released',released_at=NOW()
                   WHERE id=%s AND lease_token=%s AND status='active' RETURNING id""",
                (lease_id, lease_token),
            ).fetchone())

    def persist_graph_manifest(
        self, project_id: int, namespace_id: str, graph_generation: int,
        role: str, lease_token: str, manifest: dict[str, Any],
    ) -> dict[str, Any]:
        if role not in {"expected", "writer", "actual"}:
            raise ValueError("unknown graph manifest role")
        if (
            manifest.get("protocol") != GRAPH_MANIFEST_PROTOCOL
            or str(manifest.get("namespace_id") or "") != str(namespace_id)
            or int(manifest.get("project_id") or 0) != int(project_id)
            or int(manifest.get("graph_generation", -1)) != int(graph_generation)
            or not str(manifest.get("policy_hash") or "")
        ):
            raise ValueError("graph manifest scope or protocol mismatch")
        with self.connection() as conn:
            lease = conn.execute(
                """SELECT 1 FROM graph_leases
                   WHERE project_id=%s AND namespace_id=%s AND graph_generation=%s
                     AND lease_token=%s AND status='active' AND expires_at>NOW()""",
                (project_id, namespace_id, graph_generation, lease_token),
            ).fetchone()
            if not lease:
                raise RuntimeError("graph manifest requires the active generation lease")
            existing = conn.execute(
                """SELECT * FROM graph_content_manifests
                   WHERE project_id=%s AND namespace_id=%s AND graph_generation=%s
                     AND lease_token=%s AND role=%s""",
                (project_id, namespace_id, graph_generation, lease_token, role),
            ).fetchone()
            if existing:
                if existing["content_digest"] != manifest["content_digest"]:
                    raise RuntimeError("immutable graph manifest conflict")
                return existing
            return conn.execute(
                """INSERT INTO graph_content_manifests(
                   id,project_id,namespace_id,graph_generation,role,protocol,policy_hash,
                   lease_token,fact_count,content_digest,manifest)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) RETURNING *""",
                (
                    uuid.uuid4().hex, project_id, namespace_id, graph_generation, role,
                    manifest["protocol"], manifest["policy_hash"], lease_token,
                    manifest["fact_count"], manifest["content_digest"], self._json(manifest),
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

    @staticmethod
    def _revision_manifest_hash(manifest: dict[str, Any]) -> str:
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def begin_document_revision(
        self,
        document_id: int,
        ingestion_run_id: str,
        manifest: dict[str, Any],
        *,
        lease_seconds: int = 86400,
    ) -> dict[str, Any]:
        """Create or resume one immutable building revision in a short transaction."""

        if not ingestion_run_id:
            raise ValueError("ingestion_run_id 不能为空")
        manifest = dict(manifest or {})
        manifest_hash = self._revision_manifest_hash(manifest)
        lease_token = uuid.uuid4().hex
        lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(60, lease_seconds))
        with self.connection() as conn:
            document = conn.execute(
                "SELECT * FROM documents WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (document_id,),
            ).fetchone()
            if not document:
                raise LookupError("文档不存在或已删除")
            existing = conn.execute(
                """SELECT * FROM document_content_revisions
                   WHERE document_id=%s AND ingestion_run_id=%s FOR UPDATE""",
                (document_id, ingestion_run_id),
            ).fetchone()
            if existing:
                if existing["manifest_hash"] != manifest_hash:
                    raise RevisionManifestMismatch(
                        "同一 ingestion_run_id 的 manifest 已改变，拒绝非幂等重试"
                    )
                if existing["status"] == "active":
                    return {**existing, "idempotent": True, "already_active": True}
                if existing["status"] in {"retired", "legacy_orphan"}:
                    raise StaleContentRevision("历史修订不能重新进入构建状态")
                row = conn.execute(
                    """UPDATE document_content_revisions
                       SET status='building',lease_token=%s,lease_expires_at=%s,
                           failure='',failed_at=NULL,updated_at=NOW()
                       WHERE revision_id=%s RETURNING *""",
                    (lease_token, lease_expires_at, existing["revision_id"]),
                ).fetchone()
                return {**row, "idempotent": True, "already_active": False}

            next_version = int(conn.execute(
                """SELECT COALESCE(MAX(content_version),0)+1 AS version
                   FROM document_content_revisions WHERE document_id=%s""",
                (document_id,),
            ).fetchone()["version"])
            revision_material = f"{document['project_id']}|{document_id}|{ingestion_run_id}"
            revision_id = "revision_" + hashlib.sha256(revision_material.encode("utf-8")).hexdigest()
            row = conn.execute(
                """INSERT INTO document_content_revisions(
                       revision_id,project_id,document_id,ingestion_run_id,content_version,
                       content_hash,source_hash,config_hash,policy_hash,manifest_hash,manifest,
                       status,lease_token,lease_expires_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,'building',%s,%s)
                   RETURNING *""",
                (
                    revision_id, document["project_id"], document_id, ingestion_run_id,
                    next_version, str(manifest.get("content_hash") or document["doc_hash"]),
                    str(manifest.get("source_hash") or document["doc_hash"]),
                    str(manifest.get("config_hash") or ""),
                    str(manifest.get("policy_hash") or ""), manifest_hash,
                    self._json(manifest), lease_token, lease_expires_at,
                ),
            ).fetchone()
            return {**row, "idempotent": False, "already_active": False}

    def get_document_revision(self, revision_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM document_content_revisions WHERE revision_id=%s",
                (revision_id,),
            ).fetchone()

    def list_document_revisions(self, document_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM document_content_revisions WHERE document_id=%s
                   ORDER BY content_version DESC""",
                (document_id,),
            ).fetchall()

    def renew_document_revision_lease(
        self, revision_id: str, ingestion_run_id: str, lease_token: str, *, lease_seconds: int = 86400,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE document_content_revisions SET lease_expires_at=%s,updated_at=NOW()
                   WHERE revision_id=%s AND ingestion_run_id=%s AND lease_token=%s
                     AND status='building' RETURNING *""",
                (
                    datetime.now(timezone.utc) + timedelta(seconds=max(60, lease_seconds)),
                    revision_id, ingestion_run_id, lease_token,
                ),
            ).fetchone()
            if not row:
                raise RevisionLeaseLost("修订 lease 已失效或不再属于当前 worker")
            return row

    def fail_document_revision(
        self, revision_id: str, ingestion_run_id: str, lease_token: str, failure: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """UPDATE document_content_revisions
                   SET status='failed',failure=%s,failed_at=NOW(),lease_token=NULL,
                       lease_expires_at=NULL,updated_at=NOW()
                   WHERE revision_id=%s AND ingestion_run_id=%s AND lease_token=%s
                     AND status='building' RETURNING *""",
                (failure, revision_id, ingestion_run_id, lease_token),
            ).fetchone()

    def activate_document_revision(
        self,
        document_id: int,
        revision_id: str,
        ingestion_run_id: str,
        lease_token: str,
        *,
        reason: str,
        document_values: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Atomically publish a complete revision and advance the graph generation."""

        document_values = dict(document_values or {})
        json_fields = {"metadata", "chunk_config", "parse_quality"}
        with self.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (document_id,))
            document = conn.execute(
                "SELECT * FROM documents WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (document_id,),
            ).fetchone()
            if not document:
                raise LookupError("文档不存在或已删除")
            revision = conn.execute(
                """SELECT * FROM document_content_revisions
                   WHERE revision_id=%s AND document_id=%s FOR UPDATE""",
                (revision_id, document_id),
            ).fetchone()
            if not revision:
                raise LookupError("内容修订不存在")
            if revision["status"] == "active" and document["active_revision_id"] == revision_id:
                project = conn.execute(
                    "SELECT * FROM projects WHERE id=%s", (document["project_id"],)
                ).fetchone()
                return document, project
            if revision["status"] != "building":
                raise StaleContentRevision(f"修订状态不允许激活: {revision['status']}")
            if revision["ingestion_run_id"] != ingestion_run_id:
                raise StaleContentRevision("ingestion_run_id 与修订不匹配")
            if revision["lease_token"] != lease_token:
                raise RevisionLeaseLost("激活前 revision lease 已丢失")
            if not revision["lease_expires_at"] or revision["lease_expires_at"] <= datetime.now(timezone.utc):
                raise RevisionLeaseLost("激活前 revision lease 已过期")
            newer = conn.execute(
                """SELECT revision_id,status FROM document_content_revisions
                   WHERE document_id=%s AND content_version>%s
                     AND status IN ('building','active') ORDER BY content_version DESC LIMIT 1""",
                (document_id, revision["content_version"]),
            ).fetchone()
            if newer:
                raise StaleContentRevision(
                    f"存在更新的修订 {newer['revision_id']}，旧 worker 不得覆盖"
                )
            conn.execute(
                """UPDATE document_content_revisions
                   SET status='retired',retired_at=NOW(),lease_token=NULL,
                       lease_expires_at=NULL,updated_at=NOW()
                   WHERE document_id=%s AND status='active' AND revision_id<>%s""",
                (document_id, revision_id),
            )
            conn.execute(
                """UPDATE document_content_revisions
                   SET status='active',activated_at=NOW(),lease_token=NULL,
                       lease_expires_at=NULL,updated_at=NOW()
                   WHERE revision_id=%s""",
                (revision_id,),
            )
            fields = list(document_values)
            assignments = ["active_revision_id=%s", "content_version=%s"]
            params: list[Any] = [revision_id, revision["content_version"]]
            for key in fields:
                assignments.append(f"{key}={'%s::jsonb' if key in json_fields else '%s'}")
                value = document_values[key]
                params.append(
                    self._json(value) if key in json_fields and not isinstance(value, str) else value
                )
            document = conn.execute(
                f"""UPDATE documents SET {','.join(assignments)},updated_at=NOW()
                    WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                [*params, document_id],
            ).fetchone()
            self._retire_orphan_entities(conn, int(document["project_id"]))
            project = conn.execute(
                """UPDATE projects SET graph_generation=graph_generation+1,graph_status=%s,
                       graph_policy_version='',graph_policy_hash='',graph_error=%s,updated_at=NOW()
                   WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                (GRAPH_STALE, reason, document["project_id"]),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            return document, project

    def finalize_document_graph_facts(
        self,
        document_id: int,
        *,
        reason: str,
        revision_id: str | None = None,
        ingestion_run_id: str | None = None,
        lease_token: str | None = None,
        **values: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Commit the indexed document projection and generation advance together."""

        if revision_id:
            if not ingestion_run_id or not lease_token:
                raise ValueError("激活修订必须提供 ingestion_run_id 和 lease_token")
            return self.activate_document_revision(
                document_id,
                revision_id,
                ingestion_run_id,
                lease_token,
                reason=reason,
                document_values=values,
            )

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
            scope = conn.execute(
                "SELECT project_id,active_revision_id FROM documents WHERE id=%s", (document_id,)
            ).fetchone()
            if not scope:
                return None
            affected_rows = conn.execute(
                "SELECT DISTINCT canonical_entity_id AS id FROM entity_mentions "
                "WHERE document_id=%s AND revision_id=%s AND is_deleted=FALSE",
                (document_id, scope.get("active_revision_id")),
            ).fetchall()
            affected_ids = [int(item["id"]) for item in affected_rows]
            row = conn.execute(
                """UPDATE documents SET is_deleted=TRUE,status='deleted',deleted_at=NOW(),updated_at=NOW()
                WHERE id=%s AND is_deleted=FALSE RETURNING *""", (document_id,)
            ).fetchone()
            if row:
                self._retire_orphan_entities(
                    conn, project_id=int(scope["project_id"]), entity_ids=affected_ids,
                )
                conn.execute(
                    """UPDATE projects
                       SET graph_generation=graph_generation+1,graph_status=%s,
                           graph_policy_version='',graph_policy_hash='',graph_error=%s,updated_at=NOW()
                       WHERE id=%s AND is_deleted=FALSE""",
                    (GRAPH_STALE, "项目文档已删除，图谱正在重新收敛", row["project_id"]),
                )
            return row

    def replace_document_content(
        self,
        document_id: int,
        sections: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        *,
        revision_id: str | None = None,
        ingestion_run_id: str | None = None,
        lease_token: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Write one immutable content revision without deleting prior audit history."""

        implicit_revision = revision_id is None
        if implicit_revision:
            ingestion_run_id = ingestion_run_id or f"compat:{uuid.uuid4().hex}"
            manifest = manifest or {
                "content_hash": hashlib.sha256(
                    self._json({"sections": sections, "chunks": chunks}).encode("utf-8")
                ).hexdigest(),
                "source_hash": "compatibility-api",
                "config_hash": "compatibility-api",
                "policy_hash": "compatibility-api",
            }
            revision = self.begin_document_revision(document_id, ingestion_run_id, manifest)
            revision_id = str(revision["revision_id"])
            lease_token = str(revision["lease_token"])
        if not revision_id or not ingestion_run_id or not lease_token:
            raise ValueError("写入内容必须绑定 revision_id、ingestion_run_id 和 lease_token")

        with self.content_revision_context(revision_id, ingestion_run_id, lease_token):
            with self.connection() as conn:
                document = conn.execute(
                    "SELECT project_id FROM documents WHERE id=%s AND is_deleted=FALSE",
                    (document_id,),
                ).fetchone()
                if not document:
                    raise LookupError("文档不存在或已删除")
                section_ids: dict[str, int] = {}
                for section in sections:
                    row = conn.execute(
                        """INSERT INTO sections(
                               project_id,document_id,revision_id,section_key,title,level,
                               content,content_hash,metadata)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                           ON CONFLICT(revision_id,section_key) DO UPDATE SET
                               title=EXCLUDED.title,level=EXCLUDED.level,content=EXCLUDED.content,
                               content_hash=EXCLUDED.content_hash,metadata=EXCLUDED.metadata,
                               is_deleted=FALSE,updated_at=NOW() RETURNING id""",
                        (
                            document["project_id"], document_id, revision_id, section["key"],
                            section["title"], section["level"], section["content"],
                            section["content_hash"], self._json(section.get("metadata")),
                        ),
                    ).fetchone()
                    section_ids[section["key"]] = int(row["id"])
                inserted = []
                for chunk in chunks:
                    row = conn.execute(
                        """INSERT INTO chunks(
                               project_id,document_id,revision_id,section_id,chunk_key,content,
                               content_hash,chunk_index,metadata)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                           ON CONFLICT(revision_id,chunk_key) DO UPDATE SET
                               section_id=EXCLUDED.section_id,content=EXCLUDED.content,
                               content_hash=EXCLUDED.content_hash,chunk_index=EXCLUDED.chunk_index,
                               metadata=EXCLUDED.metadata,is_deleted=FALSE,updated_at=NOW()
                           RETURNING *""",
                        (
                            document["project_id"], document_id, revision_id,
                            section_ids.get(chunk["section_key"]), chunk["key"], chunk["content"],
                            chunk["content_hash"], chunk["part_index"],
                            self._json(chunk.get("metadata")),
                        ),
                    ).fetchone()
                    inserted.append(row)
        if implicit_revision:
            self.activate_document_revision(
                document_id,
                revision_id,
                ingestion_run_id,
                lease_token,
                reason="兼容写入接口激活内容修订",
                document_values={},
            )
        return inserted

    def list_document_chunks(self, document_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT c.*,s.title AS section_title FROM chunks c LEFT JOIN sections s ON s.id=c.section_id
                JOIN documents d ON d.id=c.document_id AND d.active_revision_id=c.revision_id
                WHERE c.document_id=%s AND c.is_deleted=FALSE AND d.is_deleted=FALSE
                ORDER BY c.chunk_index,c.id""", (document_id,)
            ).fetchall()

    def replace_document_embeddings(self, document_id: int, records: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            active = conn.execute(
                "SELECT 1 FROM documents WHERE id=%s AND is_deleted=FALSE",
                (document_id,),
            ).fetchone()
            if not active:
                raise LookupError("文档不存在或已删除")
            for record in records: self._upsert_embedding(conn, record)

    def upsert_embedding(self, record: dict[str, Any]) -> None:
        with self.connection() as conn: self._upsert_embedding(conn, record)

    def _upsert_embedding(self, conn: psycopg.Connection, record: dict[str, Any]) -> None:
        vector = list(record["embedding"])
        vector_literal = self._vector_literal(vector)
        self._ensure_hnsw_index(conn, record["model_name"], len(vector))
        revision = self._revision_context.get()
        revision_id = (
            revision["revision_id"] if record.get("document_id") is not None and revision
            else record.get("revision_id")
        )
        if record.get("document_id") is not None and not revision_id:
            raise ValueError("文档向量写入必须绑定内容修订")
        conflict = (
            "ON CONFLICT(revision_id,object_type,object_key,model_name) "
            "WHERE revision_id IS NOT NULL"
            if record.get("document_id") is not None
            else "ON CONFLICT(project_id,object_type,object_key,model_name) "
                 "WHERE revision_id IS NULL"
        )
        conn.execute(
            f"""INSERT INTO vector_embeddings(project_id,document_id,revision_id,chunk_id,object_type,object_key,content,content_hash,
            model_name,embedding,embedding_vector,dimension,metadata)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s,%s::jsonb)
            {conflict} DO UPDATE SET document_id=EXCLUDED.document_id,
            chunk_id=EXCLUDED.chunk_id,content=EXCLUDED.content,content_hash=EXCLUDED.content_hash,embedding=EXCLUDED.embedding,
            embedding_vector=EXCLUDED.embedding_vector,dimension=EXCLUDED.dimension,metadata=EXCLUDED.metadata,
            is_deleted=FALSE,updated_at=NOW()""",
            (record["project_id"],record.get("document_id"),revision_id,record.get("chunk_id"),record["object_type"],
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
        with self.connection() as conn:
            revisioned = all((
                self._column_exists(conn, "vector_embeddings", "revision_id"),
                self._column_exists(conn, "relation_assertions", "revision_id"),
                self._column_exists(conn, "documents", "active_revision_id"),
            ))
            relation_filter = ""
            relation_policy_params: list[Any] = []
            if object_type == "relation":
                relation_clause, relation_policy_params = self._publishable_relation_clause("r")
                revision_join = (
                    "AND r.revision_id IS NOT DISTINCT FROM v0.revision_id "
                    if revisioned else ""
                )
                relation_filter = (
                    " AND EXISTS (SELECT 1 FROM relation_assertions r "
                    "WHERE r.project_id=v0.project_id AND r.id::text=v0.object_key "
                    + revision_join + "AND r.is_deleted=FALSE AND " + relation_clause + ")"
                )
            active_filter = (
                " AND (v0.document_id IS NULL OR EXISTS (SELECT 1 FROM documents active_document "
                "WHERE active_document.id=v0.document_id AND active_document.is_deleted=FALSE "
                "AND active_document.active_revision_id=v0.revision_id))"
                if revisioned else
                " AND (v0.document_id IS NULL OR EXISTS (SELECT 1 FROM documents active_document "
                "WHERE active_document.id=v0.document_id AND active_document.is_deleted=FALSE))"
            )
            final_active_filter = (
                "(d.id IS NULL OR (d.is_deleted=FALSE AND d.active_revision_id=v.revision_id))"
                if revisioned else "(d.id IS NULL OR d.is_deleted=FALSE)"
            )
            statement = sql.SQL(
                """WITH vector_hits AS (
                    SELECT id FROM vector_embeddings v0
                    WHERE project_id=%s AND object_type=%s AND model_name=%s AND dimension=%s AND is_deleted=FALSE {} {}
                    ORDER BY embedding_vector::{} <=> %s::{}
                    LIMIT %s
                ), lexical_hits AS (
                    SELECT id FROM vector_embeddings v0
                    WHERE project_id=%s AND object_type=%s AND model_name=%s AND dimension=%s AND is_deleted=FALSE {} {}
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
                WHERE {}
                ORDER BY v.embedding_vector::{} <=> %s::{}
                LIMIT %s"""
            ).format(
                sql.SQL(relation_filter), sql.SQL(active_filter), vector_type, vector_type,
                sql.SQL(relation_filter), sql.SQL(active_filter), vector_type, vector_type,
                sql.SQL(final_active_filter), vector_type, vector_type,
            )
            params = (
                project_id,object_type,model_name,dimension,*relation_policy_params,
                vector_literal,candidate_limit,
                project_id,object_type,model_name,dimension,*relation_policy_params,
                pattern,candidate_limit,
                vector_literal,vector_literal,candidate_limit,
            )
            conn.execute("SELECT set_config('hnsw.ef_search',%s,TRUE)", (str(Config.pgvector_hnsw_ef_search),))
            return conn.execute(statement, params).fetchall()

    def record_indexing_failure(self, values: dict[str, Any]) -> dict[str, Any]:
        """Persist a retryable derived-index failure without changing source truth."""

        revision = self._revision_context.get()
        revision_id = (
            revision["revision_id"] if values.get("document_id") is not None and revision
            else values.get("revision_id")
        )
        if values.get("document_id") is not None and not revision_id:
            raise ValueError("文档索引失败记录必须绑定内容修订")
        with self.connection() as conn:
            existing = conn.execute(
                """SELECT attempts,max_attempts FROM indexing_failures
                   WHERE project_id=%s AND revision_id IS NOT DISTINCT FROM %s
                     AND object_type=%s AND object_key=%s AND model_name=%s
                   FOR UPDATE""",
                (
                    values["project_id"], revision_id, values["object_type"], str(values["object_key"]),
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
            conflict = (
                "ON CONFLICT(revision_id,object_type,object_key,model_name) "
                "WHERE revision_id IS NOT NULL"
                if values.get("document_id") is not None
                else "ON CONFLICT(project_id,object_type,object_key,model_name) "
                     "WHERE revision_id IS NULL"
            )
            return conn.execute(
                f"""INSERT INTO indexing_failures(project_id,document_id,revision_id,chunk_id,object_type,
                   object_key,model_name,status,attempts,max_attempts,next_attempt_at,last_error,payload)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                   {conflict} DO UPDATE SET
                   document_id=EXCLUDED.document_id,chunk_id=EXCLUDED.chunk_id,status=EXCLUDED.status,
                   attempts=EXCLUDED.attempts,max_attempts=EXCLUDED.max_attempts,
                   next_attempt_at=EXCLUDED.next_attempt_at,last_error=EXCLUDED.last_error,
                   payload=EXCLUDED.payload,completed_at=NULL,updated_at=NOW() RETURNING *""",
                (
                    values["project_id"], values.get("document_id"), revision_id, values.get("chunk_id"),
                    values["object_type"], str(values["object_key"]), values["model_name"],
                    status, attempts, max_attempts, next_attempt_at,
                    values.get("last_error") or "", self._json(values.get("payload")),
                ),
            ).fetchone()

    def complete_indexing_failure(
        self, project_id: int, object_type: str, object_key: str, model_name: str,
    ) -> None:
        revision = self._revision_context.get()
        revision_id = revision["revision_id"] if revision else None
        with self.connection() as conn:
            if revision_id:
                conn.execute(
                    """UPDATE indexing_failures SET status='completed',last_error='',
                       completed_at=NOW(),updated_at=NOW()
                       WHERE project_id=%s AND revision_id=%s
                         AND object_type=%s AND object_key=%s AND model_name=%s""",
                    (project_id, revision_id, object_type, str(object_key), model_name),
                )
            else:
                conn.execute(
                    """UPDATE indexing_failures failure
                       SET status='completed',last_error='',completed_at=NOW(),updated_at=NOW()
                       WHERE failure.project_id=%s AND failure.object_type=%s
                         AND failure.object_key=%s AND failure.model_name=%s
                         AND (failure.revision_id IS NULL OR EXISTS(
                             SELECT 1 FROM documents document
                             WHERE document.id=failure.document_id
                               AND document.is_deleted=FALSE
                               AND document.active_revision_id=failure.revision_id
                         ))""",
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
                """SELECT failure.* FROM indexing_failures failure
                   LEFT JOIN documents document ON document.id=failure.document_id
                   WHERE failure.project_id=%s AND failure.object_type=%s
                   AND (failure.document_id IS NULL OR (
                       document.is_deleted=FALSE
                       AND document.active_revision_id=failure.revision_id
                   ))
                   AND status='pending' AND next_attempt_at<=NOW()
                   ORDER BY failure.next_attempt_at,failure.id LIMIT %s""",
                (project_id, object_type, page_size),
            ).fetchall()

    def list_blocking_indexing_failures(
        self, project_id: int, graph_generation: int,
    ) -> list[dict[str, Any]]:
        """Return unresolved derived indexes that explicitly gate one graph generation."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM indexing_failures
                   WHERE project_id=%s AND status IN ('pending','permanent_failed')
                     AND COALESCE(payload->>'blocks_graph_generation','') ~ '^[0-9]+$'
                     AND (payload->>'blocks_graph_generation')::BIGINT=%s
                   ORDER BY object_type,object_key""",
                (project_id, graph_generation),
            ).fetchall()

    def cleanup_indexing_failures(self) -> int:
        """Bound completed/permanent outbox history by the configured retention."""

        if not Config.indexing_failure_cleanup_enabled:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=Config.indexing_failure_retention_days)
        with self.connection() as conn:
            result = conn.execute(
                """DELETE FROM indexing_failures
                   WHERE status IN ('completed','permanent_failed') AND updated_at<%s""",
                (cutoff,),
            )
            return max(0, result.rowcount or 0)

    def cleanup_audit_metrics(self) -> int:
        """Apply configured time and capacity bounds to redacted metric audits."""

        if not Config.audit_cleanup_enabled:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=Config.audit_retention_days)
        with self.connection() as conn:
            expired = conn.execute(
                "DELETE FROM operation_metrics WHERE created_at<%s", (cutoff,),
            ).rowcount or 0
            overflow = conn.execute(
                """DELETE FROM operation_metrics WHERE id IN (
                       SELECT id FROM operation_metrics ORDER BY created_at DESC,id DESC
                       OFFSET %s
                   )""",
                (Config.audit_max_rows,),
            ).rowcount or 0
            return max(0, int(expired)) + max(0, int(overflow))

    def list_embeddings(self, project_id: int, object_type: str, model_name: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT v.*,d.original_filename,d.storage_path,s.title AS section_title
                FROM vector_embeddings v LEFT JOIN documents d ON d.id=v.document_id
                LEFT JOIN chunks c ON c.id=v.chunk_id LEFT JOIN sections s ON s.id=c.section_id
                WHERE v.project_id=%s AND v.object_type=%s AND v.model_name=%s AND v.is_deleted=FALSE
                AND (d.id IS NULL OR (d.is_deleted=FALSE AND d.active_revision_id=v.revision_id))""",
                (project_id,object_type,model_name)
            ).fetchall()

    def list_document_embeddings(
        self, document_id: int, object_type: str, model_name: str
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT id,project_id,document_id,chunk_id,object_type,object_key,content,
                          model_name,embedding,dimension,metadata
                   FROM vector_embeddings v JOIN documents d
                     ON d.id=v.document_id AND d.active_revision_id=v.revision_id
                   WHERE v.document_id=%s AND v.object_type=%s AND v.model_name=%s
                     AND v.is_deleted=FALSE AND d.is_deleted=FALSE
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
               FROM vector_embeddings v LEFT JOIN documents d ON d.id=v.document_id
               WHERE v.project_id=%s AND v.object_type=%s AND v.model_name=%s AND v.dimension=%s
                 AND v.is_deleted=FALSE
                 AND (d.id IS NULL OR (d.is_deleted=FALSE AND d.active_revision_id=v.revision_id))
                 AND (%s IS NULL OR v.document_id<>%s)
               ORDER BY v.embedding_vector::{} <=> %s::{}
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

    def list_revision_chunks(self, document_id: int, revision_id: str) -> list[dict[str, Any]]:
        """Read an immutable historical revision without consulting the active pointer."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT c.*,s.title AS section_title
                   FROM chunks c LEFT JOIN sections s
                     ON s.id=c.section_id AND s.revision_id=c.revision_id
                   WHERE c.document_id=%s AND c.revision_id=%s AND c.is_deleted=FALSE
                   ORDER BY c.chunk_index,c.id""",
                (document_id, revision_id),
            ).fetchall()

    def get_revision_chunk(self, chunk_id: int, revision_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT c.*,s.title AS section_title
                   FROM chunks c LEFT JOIN sections s
                     ON s.id=c.section_id AND s.revision_id=c.revision_id
                   WHERE c.id=%s AND c.revision_id=%s AND c.is_deleted=FALSE""",
                (chunk_id, revision_id),
            ).fetchone()

    def list_control_entities_for_chunk(
        self, chunk_id: int, revision_id: str,
    ) -> list[dict[str, Any]]:
        """Load frozen control mentions for relation-only treatment endpoints."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT id,canonical_entity_id,document_id,revision_id,chunk_id,
                          mention_text AS name,entity_type,description AS evidence,
                          mention_key,evidence_text,evidence_start,evidence_end,
                          metadata,confidence
                   FROM entity_mentions
                   WHERE chunk_id=%s AND revision_id=%s AND is_deleted=FALSE
                   ORDER BY id""",
                (chunk_id, revision_id),
            ).fetchall()

    def list_control_relations_for_chunk(
        self, chunk_id: int, revision_id: str,
    ) -> list[dict[str, Any]]:
        """Load frozen publishable control relations without consulting an active pointer."""

        clause, policy_params = self._publishable_relation_clause("relation")
        with self.connection() as conn:
            return conn.execute(
                """SELECT relation.* FROM relation_assertions relation
                   LEFT JOIN canonical_relation_projections projection
                     ON projection.project_id=relation.project_id
                    AND projection.assertion_id=relation.id
                   WHERE relation.chunk_id=%s AND relation.revision_id=%s
                     AND relation.is_deleted=FALSE
                     AND COALESCE(projection.publishable,TRUE)=TRUE
                     AND """ + clause + " ORDER BY relation.id",
                (chunk_id, revision_id, *policy_params),
            ).fetchall()

    def get_entity_shadow_namespace_for_snapshot(
        self, snapshot_id: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM entity_shadow_namespaces
                   WHERE snapshot_id=%s ORDER BY created_at DESC LIMIT 1""",
                (snapshot_id,),
            ).fetchone()
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

    def get_schema_version(self, project_id: int, version: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM graph_schemas WHERE project_id=%s AND version=%s",
                (project_id, version),
            ).fetchone()

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

    @staticmethod
    def build_exact_entity_candidates_query(
        project_id: int,
        entity_type: str,
        normalized_name: str,
        limit: int = 10,
    ) -> tuple[str, tuple[Any, ...]]:
        """Build the production exact-alias query used by runtime and EXPLAIN tests."""

        bounded = max(1, min(50, int(limit)))
        statement = """WITH matched AS (
               SELECT entity.*,
                      MAX(CASE WHEN alias.alias_kind='canonical_name' THEN 1.0 ELSE 0.99 END)
                          AS lexical_score,
                      BOOL_OR(alias.alias_kind='canonical_name') AS canonical_alias_match
               FROM canonical_entity_aliases alias
               JOIN canonical_entities entity
                 ON entity.id=alias.canonical_entity_id
                AND entity.project_id=alias.project_id
               LEFT JOIN canonical_entity_redirects redirect
                 ON redirect.project_id=entity.project_id
                AND redirect.source_entity_id=entity.id AND redirect.active=TRUE
               WHERE alias.project_id=%s AND alias.active=TRUE
                 AND alias.normalized_alias=%s
                 AND entity.is_deleted=FALSE AND redirect.source_entity_id IS NULL
                 AND (%s='GenericEntity' OR alias.entity_type=%s
                      OR alias.entity_type='GenericEntity')
               GROUP BY entity.id
           )
           SELECT * FROM matched
           ORDER BY (entity_type=%s) DESC,canonical_alias_match DESC,source_count DESC,id
           LIMIT %s"""
        return statement, (
            int(project_id), str(normalized_name), str(entity_type), str(entity_type),
            str(entity_type), bounded,
        )

    @staticmethod
    def build_lexical_entity_candidates_query(
        project_id: int,
        entity_type: str,
        normalized_name: str,
        limit: int,
        threshold: float,
    ) -> tuple[str, tuple[Any, ...]]:
        """Build a bounded trigram query over the normalized alias source of truth."""

        bounded = max(1, min(100, int(limit)))
        score_floor = max(0.0, min(1.0, float(threshold)))
        statement = """WITH matched AS (
               SELECT entity.*,MAX(similarity(alias.normalized_alias,%s)) AS lexical_score
               FROM canonical_entity_aliases alias
               JOIN canonical_entities entity
                 ON entity.id=alias.canonical_entity_id
                AND entity.project_id=alias.project_id
               LEFT JOIN canonical_entity_redirects redirect
                 ON redirect.project_id=entity.project_id
                AND redirect.source_entity_id=entity.id AND redirect.active=TRUE
               WHERE alias.project_id=%s AND alias.active=TRUE
                 AND entity.is_deleted=FALSE AND redirect.source_entity_id IS NULL
                 AND (%s='GenericEntity' OR alias.entity_type=%s
                      OR alias.entity_type='GenericEntity')
                 AND alias.normalized_alias %% %s
                 AND similarity(alias.normalized_alias,%s)>=%s
               GROUP BY entity.id
           )
           SELECT * FROM matched ORDER BY lexical_score DESC,source_count DESC,id LIMIT %s"""
        return statement, (
            str(normalized_name), int(project_id), str(entity_type), str(entity_type),
            str(normalized_name), str(normalized_name), score_floor, bounded,
        )

    def find_exact_entity_candidates(
        self, project_id: int, entity_type: str, normalized_name: str
    ) -> list[dict[str, Any]]:
        statement, params = self.build_exact_entity_candidates_query(
            project_id, entity_type, normalized_name, 10,
        )
        with self.connection() as conn:
            return conn.execute(statement, params).fetchall()

    def find_exact_entity_candidates_batch(
        self,
        project_id: int,
        requests: list[tuple[str, str]],
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        unique_requests = list(dict.fromkeys(
            (str(entity_type), str(normalized_name))
            for entity_type, normalized_name in requests
            if normalized_name
        ))
        result = {request: [] for request in unique_requests}
        if not unique_requests:
            return result
        statement, params = self.build_exact_entity_candidates_batch_query(
            project_id, unique_requests,
        )
        with self.connection() as conn:
            rows = conn.execute(statement, params).fetchall()
        for row in rows:
            key = (
                str(row.pop("requested_entity_type")),
                str(row.pop("requested_normalized_name")),
            )
            row.pop("request_rank", None)
            row.pop("ordinality", None)
            row.pop("exact_type", None)
            row.pop("canonical_alias", None)
            result[key].append(row)
        return result

    @staticmethod
    def build_exact_entity_candidates_batch_query(
        project_id: int,
        requests: list[tuple[str, str]],
        per_request_limit: int = 10,
    ) -> tuple[str, tuple[Any, ...]]:
        """Build the production and EXPLAIN query with a hard per-request SQL cap."""

        unique_requests = list(dict.fromkeys(
            (str(entity_type), str(normalized_name))
            for entity_type, normalized_name in requests
            if normalized_name
        ))
        entity_types = [item[0] for item in unique_requests]
        normalized_names = [item[1] for item in unique_requests]
        bounded = max(1, min(10, int(per_request_limit)))
        statement = """WITH requested AS (
               SELECT entity_type,normalized_name,ordinality
               FROM unnest(%s::text[],%s::text[]) WITH ORDINALITY
                    AS request(entity_type,normalized_name,ordinality)
           ), deduplicated AS (
               SELECT DISTINCT ON (request.ordinality,entity.id)
                      request.ordinality,request.entity_type AS requested_entity_type,
                      request.normalized_name AS requested_normalized_name,
                      entity.*,
                      CASE WHEN entity.normalized_name=request.normalized_name
                           THEN 1.0 ELSE 0.99 END AS lexical_score,
                      (entity.entity_type=request.entity_type) AS exact_type,
                      (alias.alias_kind='canonical_name') AS canonical_alias
               FROM requested request
               JOIN canonical_entity_aliases alias
                 ON alias.project_id=%s AND alias.active=TRUE
                AND alias.normalized_alias=request.normalized_name
               JOIN canonical_entities entity
                 ON entity.id=alias.canonical_entity_id
                AND entity.project_id=alias.project_id AND entity.is_deleted=FALSE
               LEFT JOIN canonical_entity_redirects redirect
                 ON redirect.project_id=entity.project_id
                AND redirect.source_entity_id=entity.id AND redirect.active=TRUE
               WHERE redirect.source_entity_id IS NULL
                 AND (request.entity_type='GenericEntity'
                      OR alias.entity_type=request.entity_type
                      OR alias.entity_type='GenericEntity')
               ORDER BY request.ordinality,entity.id,
                        (entity.entity_type=request.entity_type) DESC,
                        (alias.alias_kind='canonical_name') DESC,
                        entity.source_count DESC,entity.id
           ), ranked AS (
               SELECT deduplicated.*,
                      ROW_NUMBER() OVER (
                        PARTITION BY ordinality
                        ORDER BY exact_type DESC,canonical_alias DESC,source_count DESC,id
                      ) AS request_rank
               FROM deduplicated
           )
           SELECT * FROM ranked WHERE ranked.request_rank<=%s
           ORDER BY ordinality,request_rank"""
        return statement, (entity_types, normalized_names, int(project_id), bounded)

    def search_entity_name_candidates(
        self,
        project_id: int,
        entity_type: str,
        normalized_name: str,
        limit: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        statement, params = self.build_lexical_entity_candidates_query(
            project_id, entity_type, normalized_name, limit, threshold,
        )
        with self.connection() as conn:
            conn.execute(
                "SELECT set_config('pg_trgm.similarity_threshold',%s,TRUE)",
                (str(max(0.0, min(1.0, threshold))),),
            )
            return conn.execute(statement, params).fetchall()

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

    def blocked_entity_candidate_ids(
        self,
        project_id: int,
        mention_key: str,
        candidate_ids: list[int],
    ) -> list[int]:
        """Return cannot-linked current roots in one bounded candidate query."""

        ids = sorted({int(value) for value in candidate_ids})
        if not ids:
            return []
        with self.connection() as conn:
            redirects = conn.execute(
                """SELECT source_entity_id,target_entity_id FROM canonical_entity_redirects
                   WHERE project_id=%s AND active=TRUE""",
                (project_id,),
            ).fetchall()
            redirect_map = {
                int(item["source_entity_id"]): int(item["target_entity_id"])
                for item in redirects
            }

            def root(entity_id: int) -> int:
                seen: set[int] = set()
                while entity_id in redirect_map and entity_id not in seen:
                    seen.add(entity_id)
                    entity_id = redirect_map[entity_id]
                return entity_id

            roots = {root(value) for value in ids}
            mention_row = conn.execute(
                """SELECT COALESCE(assignment.canonical_entity_id,
                                   mention.canonical_entity_id) AS canonical_entity_id
                   FROM entity_mentions mention
                   LEFT JOIN entity_assignments assignment
                     ON assignment.project_id=mention.project_id
                    AND assignment.mention_id=mention.id AND assignment.active=TRUE
                   WHERE mention.project_id=%s AND mention.mention_key=%s
                     AND mention.is_deleted=FALSE
                   ORDER BY mention.id DESC LIMIT 1""",
                (project_id, mention_key),
            ).fetchone()
            mention_root = (
                root(int(mention_row["canonical_entity_id"]))
                if mention_row and mention_row.get("canonical_entity_id") is not None
                else None
            )
            relevant_roots = roots | ({mention_root} if mention_root is not None else set())
            relevant_ids = sorted({
                entity_id for entity_id in {
                    *ids, *redirect_map.keys(), *redirect_map.values(), *relevant_roots,
                }
                if root(entity_id) in relevant_roots
            })
            links = conn.execute(
                """SELECT left_entity_id,right_entity_id,left_mention_key,right_mention_key
                   FROM entity_cannot_links WHERE project_id=%s AND active=TRUE
                     AND (left_entity_id=ANY(%s) OR right_entity_id=ANY(%s)
                          OR left_mention_key=%s OR right_mention_key=%s)""",
                (project_id, relevant_ids, relevant_ids, mention_key, mention_key),
            ).fetchall()
        blocked_roots: set[int] = set()
        for link in links:
            left_id = root(int(link["left_entity_id"])) if link.get("left_entity_id") else None
            right_id = root(int(link["right_entity_id"])) if link.get("right_entity_id") else None
            if link.get("left_mention_key") == mention_key and right_id is not None:
                blocked_roots.add(right_id)
            if link.get("right_mention_key") == mention_key and left_id is not None:
                blocked_roots.add(left_id)
            if mention_root is not None and left_id == mention_root and right_id is not None:
                blocked_roots.add(right_id)
            if mention_root is not None and right_id == mention_root and left_id is not None:
                blocked_roots.add(left_id)
        return [value for value in ids if root(value) in blocked_roots]

    def upsert_canonical_entity(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
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
            self._sync_entity_alias_rows(
                conn, row,
                [row["name"], *(row.get("aliases") or [])],
                policy_version=str((values.get("metadata") or {}).get("policy_version")
                                   or "entity-resolution-p0-v1"),
            )
            return row

    @classmethod
    def _sync_entity_alias_rows(
        cls,
        conn: psycopg.Connection,
        entity: dict[str, Any],
        aliases: list[str],
        *,
        policy_version: str,
    ) -> None:
        canonical = cls._normalize_entity_name(str(entity.get("name") or ""))
        for alias in dict.fromkeys(str(item).strip() for item in aliases if str(item).strip()):
            normalized = cls._normalize_entity_name(alias)
            if not normalized:
                continue
            alias_kind = "canonical_name" if normalized == canonical else "observed"
            conn.execute(
                """INSERT INTO canonical_entity_aliases(
                   project_id,canonical_entity_id,entity_type,alias,normalized_alias,
                   alias_kind,policy_version)
                   VALUES(%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind)
                   DO UPDATE SET alias=EXCLUDED.alias,entity_type=EXCLUDED.entity_type,
                       active=TRUE,policy_version=EXCLUDED.policy_version,updated_at=NOW()""",
                (
                    entity["project_id"], entity["id"], entity["entity_type"], alias,
                    normalized, alias_kind, policy_version,
                ),
            )

    def merge_canonical_entity_evidence(
        self,
        entity_id: int,
        aliases: list[str],
        normalized_aliases: list[str],
        description: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE canonical_entities SET
                   aliases=(SELECT ARRAY(SELECT DISTINCT unnest(aliases || %s::text[]))),
                   normalized_aliases=(SELECT ARRAY(SELECT DISTINCT unnest(normalized_aliases || %s::text[]))),
                   description=CASE WHEN length(%s)>length(description) THEN %s ELSE description END,
                   updated_at=NOW()
                   WHERE id=%s RETURNING *""",
                (aliases, normalized_aliases, description, description, entity_id),
            ).fetchone()
            self._sync_entity_alias_rows(
                conn, row, [row["name"], *aliases],
                policy_version="entity-resolution-p0-v1",
            )
            return row

    def update_canonical_entity_type(self, entity_id: int, entity_type: str) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                "UPDATE canonical_entities SET entity_type=%s,updated_at=NOW() WHERE id=%s RETURNING *",
                (entity_type, entity_id),
            ).fetchone()

    def add_entity_mention(self, values: dict[str, Any]) -> dict[str, Any]:
        mention_key = self._mention_conflict_key(values)
        with self.connection() as conn:
            row = conn.execute(
                """INSERT INTO entity_mentions(project_id,document_id,chunk_id,
                canonical_entity_id,mention_key,mention_text,normalized_name,entity_type,
                description,confidence,extraction_version,evidence_text,evidence_start,evidence_end,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(revision_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE
                DO UPDATE SET canonical_entity_id=EXCLUDED.canonical_entity_id,
                description=EXCLUDED.description,
                confidence=EXCLUDED.confidence,evidence_text=EXCLUDED.evidence_text,evidence_start=EXCLUDED.evidence_start,
                evidence_end=EXCLUDED.evidence_end,metadata=EXCLUDED.metadata,is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (values["project_id"],values["document_id"],values["chunk_id"],
                 values["canonical_entity_id"],mention_key,
                 values["mention_text"],values["normalized_name"],values["entity_type"],
                 values.get("description") or "",0.5 if values.get("confidence") is None
                 else values.get("confidence"),values.get("extraction_version") or "v2",
                 values.get("evidence_text") or "",values.get("evidence_start"),
                 values.get("evidence_end"),self._json(values.get("metadata"))),
            ).fetchone()
            conn.execute(
                """INSERT INTO canonical_entity_documents(project_id,canonical_entity_id,document_id)
                   VALUES(%s,%s,%s)
                   ON CONFLICT(revision_id,canonical_entity_id) DO UPDATE
                   SET project_id=EXCLUDED.project_id,is_deleted=FALSE,updated_at=NOW()""",
                (values["project_id"], values["canonical_entity_id"], values["document_id"]),
            )
            self._refresh_entity_source_counts(
                conn, int(values["project_id"]), [int(values["canonical_entity_id"])],
            )
            return row

    def add_entity_resolution_decision(self, values: dict[str, Any]) -> dict[str, Any]:
        mention_key = self._mention_conflict_key(values)
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO entity_resolution_decisions(project_id,document_id,chunk_id,
                canonical_entity_id,mention_key,mention_text,normalized_name,entity_type,
                decision,candidate_entity_ids,scores,reason,model_name,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT(revision_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE
                DO UPDATE SET
                canonical_entity_id=EXCLUDED.canonical_entity_id,decision=EXCLUDED.decision,
                candidate_entity_ids=EXCLUDED.candidate_entity_ids,scores=EXCLUDED.scores,
                reason=EXCLUDED.reason,model_name=EXCLUDED.model_name,metadata=EXCLUDED.metadata,
                is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (
                    values["project_id"], values["document_id"], values["chunk_id"],
                    values["canonical_entity_id"], mention_key,
                    values["mention_text"], values["normalized_name"], values["entity_type"],
                    values["decision"], values.get("candidate_entity_ids") or [],
                    self._json(values.get("scores")), values.get("reason") or "", values.get("model_name"),
                    self._json(values.get("metadata")),
                ),
            ).fetchone()

    @staticmethod
    def _mention_conflict_key(values: dict[str, Any]) -> str:
        return str(values.get("mention_key") or Database._stable_mention_key(values))

    @staticmethod
    def _decision_conflict_key(values: dict[str, Any]) -> str:
        return str(values.get("mention_key") or Database._stable_mention_key(values))

    @staticmethod
    def _stable_mention_key(values: dict[str, Any]) -> str:
        mention_start = values.get("mention_start")
        mention_end = values.get("mention_end")
        if mention_start is None:
            mention_start = values.get("evidence_start")
        if mention_end is None:
            mention_end = values.get("evidence_end")
        material = "\x1f".join((
            str(values["project_id"]), str(values["document_id"]),
            str(values.get("revision_id") or ""), str(values["chunk_id"]),
            str(mention_start if mention_start is not None else ""),
            str(mention_end if mention_end is not None else ""),
            str(values.get("entity_type") or ""), str(values.get("mention_text") or ""),
        ))
        return "mention_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    def persist_entity_resolution_batch(
        self,
        mentions: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist one Chunk's mention decisions and source counts atomically."""

        if not mentions:
            return []
        project_id = int(mentions[0]["project_id"])
        document_id = int(mentions[0]["document_id"])
        if any(
            int(item["project_id"]) != project_id or int(item["document_id"]) != document_id
            for item in [*mentions, *decisions]
        ):
            raise ValueError("实体批次必须属于同一项目和文档")
        unique_mentions: dict[str, dict[str, Any]] = {}
        for item in mentions:
            unique_mentions[self._mention_conflict_key(item)] = item
        unique_decisions: dict[str, dict[str, Any]] = {}
        for item in decisions:
            unique_decisions[self._decision_conflict_key(item)] = item

        with self.connection() as conn:
            persisted_by_key: dict[str, dict[str, Any]] = {}
            for key, values in unique_mentions.items():
                mention_key = str(values.get("mention_key") or self._stable_mention_key(values))
                extraction_policy = str(
                    values.get("extraction_policy_version") or "entity-extraction-p0-v1"
                )
                row = conn.execute(
                    """INSERT INTO entity_mentions(project_id,document_id,chunk_id,canonical_entity_id,
                    mention_text,normalized_name,entity_type,description,confidence,extraction_version,
                    evidence_text,evidence_start,evidence_end,mention_key,extraction_policy_version,metadata)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT(revision_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE
                    DO UPDATE SET canonical_entity_id=entity_mentions.canonical_entity_id,
                    description=EXCLUDED.description,confidence=EXCLUDED.confidence,
                    evidence_text=EXCLUDED.evidence_text,evidence_start=EXCLUDED.evidence_start,
                    evidence_end=EXCLUDED.evidence_end,
                    mention_key=CASE WHEN entity_mentions.mention_key='' THEN EXCLUDED.mention_key
                                     ELSE entity_mentions.mention_key END,
                    extraction_policy_version=EXCLUDED.extraction_policy_version,
                    metadata=EXCLUDED.metadata,
                    is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                    (
                        values["project_id"], values["document_id"], values["chunk_id"],
                        values["canonical_entity_id"], values["mention_text"], values["normalized_name"],
                        values["entity_type"], values.get("description") or "",
                        0.5 if values.get("confidence") is None else values.get("confidence"),
                        values.get("extraction_version") or "v2", values.get("evidence_text") or "",
                        values.get("evidence_start"), values.get("evidence_end"),
                        mention_key, extraction_policy,
                        self._json({
                            **(values.get("metadata") or {}),
                            "mention_start": values.get("mention_start"),
                            "mention_end": values.get("mention_end"),
                        }),
                    ),
                ).fetchone()
                persisted_by_key[key] = row
                conn.execute(
                    """INSERT INTO entity_assignments(
                       project_id,mention_id,canonical_entity_id,assignment_generation,
                       policy_version,reason,active,metadata)
                       VALUES(%s,%s,%s,1,%s,'ingestion assignment',TRUE,%s::jsonb)
                       ON CONFLICT(project_id,mention_id) WHERE active=TRUE DO NOTHING""",
                    (
                        project_id, row["id"], values["canonical_entity_id"],
                        str(values.get("resolution_policy_version") or "entity-resolution-p0-v1"),
                        self._json({"document_id": document_id, "chunk_id": values["chunk_id"]}),
                    ),
                )
                conn.execute(
                    """INSERT INTO entity_type_assertions(
                       project_id,document_id,chunk_id,mention_id,mention_key,entity_type,
                       confidence,evidence_text,evidence_start,evidence_end,policy_version,source,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'model',%s::jsonb)
                       ON CONFLICT(revision_id,mention_key,entity_type,policy_version,source)
                       DO UPDATE SET confidence=EXCLUDED.confidence,evidence_text=EXCLUDED.evidence_text,
                       evidence_start=EXCLUDED.evidence_start,evidence_end=EXCLUDED.evidence_end,
                       metadata=EXCLUDED.metadata""",
                    (
                        project_id, document_id, values["chunk_id"], row["id"], mention_key,
                        values["entity_type"],
                        0.5 if values.get("confidence") is None else values.get("confidence"),
                        values.get("evidence_text") or values.get("mention_text") or "",
                        values.get("evidence_start"), values.get("evidence_end"),
                        extraction_policy, self._json({"description": values.get("description") or ""}),
                    ),
                )
                aliases = [values.get("mention_text") or ""]
                aliases.extend((values.get("metadata") or {}).get("accepted_aliases") or [])
                for alias in dict.fromkeys(str(item) for item in aliases if item):
                    normalized_alias = self._normalize_entity_name(alias)
                    if not normalized_alias:
                        continue
                    conn.execute(
                        """INSERT INTO canonical_entity_aliases(
                           project_id,canonical_entity_id,entity_type,alias,normalized_alias,
                           alias_kind,evidence,policy_version)
                           VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                           ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind)
                           DO UPDATE SET alias=EXCLUDED.alias,entity_type=EXCLUDED.entity_type,
                           active=TRUE,evidence=EXCLUDED.evidence,updated_at=NOW()""",
                        (
                            project_id, values["canonical_entity_id"], values["entity_type"], alias,
                            normalized_alias,
                            "observed",
                            self._json({"mention_key": mention_key}),
                            str(values.get("resolution_policy_version") or "entity-resolution-p0-v1"),
                        ),
                    )
            for values in unique_decisions.values():
                conn.execute(
                    """INSERT INTO entity_resolution_decisions(project_id,document_id,chunk_id,
                    canonical_entity_id,mention_text,normalized_name,entity_type,mention_key,decision,
                    candidate_entity_ids,scores,reason,model_name,metadata)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT(revision_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE
                    DO UPDATE SET
                    canonical_entity_id=EXCLUDED.canonical_entity_id,decision=EXCLUDED.decision,
                    candidate_entity_ids=EXCLUDED.candidate_entity_ids,scores=EXCLUDED.scores,
                    reason=EXCLUDED.reason,model_name=EXCLUDED.model_name,metadata=EXCLUDED.metadata,
                    is_deleted=FALSE,updated_at=NOW()""",
                    (
                        values["project_id"], values["document_id"], values["chunk_id"],
                        values["canonical_entity_id"], values["mention_text"], values["normalized_name"],
                        values["entity_type"], values.get("mention_key") or self._stable_mention_key(values),
                        values["decision"], values.get("candidate_entity_ids") or [],
                        self._json(values.get("scores")), values.get("reason") or "",
                        values.get("model_name"), self._json(values.get("metadata")),
                    ),
                )
            affected_ids = sorted({int(item["canonical_entity_id"]) for item in mentions})
            conn.execute(
                """INSERT INTO canonical_entity_documents(project_id,canonical_entity_id,document_id)
                   SELECT %s,entity_id,%s FROM unnest(%s::bigint[]) AS entity_id
                   ON CONFLICT(revision_id,canonical_entity_id) DO UPDATE
                   SET project_id=EXCLUDED.project_id,is_deleted=FALSE,updated_at=NOW()""",
                (project_id, document_id, affected_ids),
            )
            self._refresh_entity_source_counts(conn, project_id, affected_ids)
        return [persisted_by_key[self._mention_conflict_key(item)] for item in mentions]

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
                attempt,model_name,prompt_version,schema_version,ingestion_run_id,request_json,raw_response,
                verdict,reason,error_category,batch_id)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s) RETURNING *""",
                (
                    values["candidate_id"], values["project_id"], values["document_id"], values["chunk_id"],
                    values["attempt"], values.get("model_name") or "", values.get("prompt_version") or "",
                    int(values.get("schema_version") or 1), values.get("ingestion_run_id") or "",
                    self._audit_json(values.get("request_json")),
                    self._audit_json(values.get("raw_response")) if values.get("raw_response") is not None else None,
                    values.get("verdict") or None, values.get("reason") or "",
                    values.get("error_category") or "",
                    values.get("batch_id"),
                ),
            ).fetchone()

    @staticmethod
    def _relation_verification_batch_key(values: dict[str, Any]) -> tuple[Any, ...]:
        return (
            int(values["project_id"]), int(values["document_id"]), int(values["chunk_id"]),
            int(values["attempt"]), values.get("model_name") or "",
            values.get("prompt_version") or "", int(values.get("schema_version") or 1),
            values.get("ingestion_run_id") or "",
            json.dumps(values.get("request_json") or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(values.get("raw_response"), ensure_ascii=False, sort_keys=True)
            if values.get("raw_response") is not None else None,
        )

    def _upsert_relation_verification_batch(
        self, conn: psycopg.Connection, values: dict[str, Any],
    ) -> str:
        key = self._relation_verification_batch_key(values)
        digest = hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        conn.execute(
            """INSERT INTO relation_verification_batches(id,project_id,document_id,chunk_id,attempt,
               model_name,prompt_version,schema_version,ingestion_run_id,request_json,raw_response)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
               ON CONFLICT(id) DO NOTHING""",
            (
                digest, values["project_id"], values["document_id"], values["chunk_id"],
                values["attempt"], values.get("model_name") or "",
                values.get("prompt_version") or "", int(values.get("schema_version") or 1),
                values.get("ingestion_run_id") or "", self._audit_json(values.get("request_json")),
                self._audit_json(values.get("raw_response"))
                if values.get("raw_response") is not None else None,
            ),
        )
        return digest

    def add_relation_assertion(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            return self._add_relation_assertion(conn, values)

    def _add_relation_assertion(
        self, conn: psycopg.Connection, values: dict[str, Any],
    ) -> dict[str, Any]:
        conflict_target = (
            "(revision_id,relation_key)"
            if self._column_exists(conn, "relation_assertions", "revision_id")
            else "(project_id,relation_key)"
        )
        row = conn.execute(
                f"""INSERT INTO relation_assertions(project_id,document_id,chunk_id,source_entity_id,target_entity_id,relation_key,relation_type,description,confidence,extraction_version,ingestion_run_id,evidence_text,evidence_start,evidence_end,verification_status,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT{conflict_target} DO UPDATE SET document_id=EXCLUDED.document_id,
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
            batches: dict[tuple[Any, ...], str] = {}
            for values in events:
                if not values.get("request_json") and values.get("raw_response") is None:
                    self._add_relation_verification_event(conn, values)
                    continue
                key = self._relation_verification_batch_key(values)
                batch_id = batches.get(key)
                if batch_id is None:
                    batch_id = self._upsert_relation_verification_batch(conn, values)
                    batches[key] = batch_id
                event_values = {
                    **values,
                    "batch_id": batch_id,
                    "request_json": {},
                    "raw_response": None,
                }
                self._add_relation_verification_event(conn, event_values)
            persisted = [self._add_relation_assertion(conn, values) for values in assertions]
            candidate_map = {str(item["candidate_id"]): item for item in candidates}
            for values, assertion in zip(assertions, persisted):
                metadata = values.get("metadata") if isinstance(values.get("metadata"), dict) else {}
                candidate_id = str(metadata.get("candidate_id") or "")
                candidate = candidate_map.get(candidate_id)
                if not candidate:
                    continue
                fact_id = "mention_relation_" + hashlib.sha256(
                    f"{values['project_id']}\x1f{candidate_id}\x1f{values['relation_key']}".encode("utf-8")
                ).hexdigest()
                conn.execute(
                    """INSERT INTO mention_relation_assertions(
                       id,project_id,document_id,chunk_id,candidate_id,canonical_assertion_id,
                       source_mention_key,target_mention_key,predicate,direction,evidence_text,
                       evidence_start,evidence_end,extraction_policy_version,
                       verification_policy_version,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        fact_id, values["project_id"], values["document_id"], values["chunk_id"],
                        candidate_id, assertion["id"], candidate["source_mention_id"],
                        candidate["target_mention_id"], values["relation_type"],
                        metadata.get("orientation") or candidate.get("orientation") or "forward",
                        values.get("evidence_text") or candidate.get("proposed_evidence") or "",
                        values.get("evidence_start"), values.get("evidence_end"),
                        metadata.get("entity_extraction_policy_version") or "entity-extraction-p0-v1",
                        values.get("extraction_version") or "relation-verifier-p0-v1",
                        self._json({"relation_key": values["relation_key"]}),
                    ),
                )
        return persisted

    def find_active_entity_assignments_by_mention_keys(
        self, project_id: int, mention_keys: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Return immutable occurrence assignments before any new resolution work."""

        keys = sorted({str(value) for value in mention_keys if value})
        if not keys:
            return {}
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT DISTINCT ON (mention.mention_key)
                          mention.mention_key,entity.*
                   FROM entity_mentions mention
                   JOIN entity_assignments assignment
                     ON assignment.project_id=mention.project_id
                    AND assignment.mention_id=mention.id AND assignment.active=TRUE
                   JOIN canonical_entities entity
                     ON entity.project_id=assignment.project_id
                    AND entity.id=assignment.canonical_entity_id
                   WHERE mention.project_id=%s AND mention.mention_key=ANY(%s)
                     AND mention.is_deleted=FALSE
                   ORDER BY mention.mention_key,assignment.assignment_generation DESC,
                            assignment.id DESC""",
                (project_id, keys),
            ).fetchall()
        return {str(row.pop("mention_key")): row for row in rows}

    def start_relation_shadow_run(self, values: dict[str, Any]) -> dict[str, Any]:
        """Start an isolated treatment run without mutating product projection state."""

        policy_set = normalize_policy_set(values["policy_set"])
        with self.connection() as conn:
            row = conn.execute(
                """INSERT INTO relation_shadow_runs(
                   id,project_id,document_id,chunk_id,control_policy_hash,treatment_policy_hash,
                   policy_set,input_fingerprint)
                   VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s)
                   ON CONFLICT(id) DO UPDATE SET policy_set=EXCLUDED.policy_set
                   WHERE relation_shadow_runs.input_fingerprint IN ('',EXCLUDED.input_fingerprint)
                   RETURNING *""",
                (
                    values["id"], values["project_id"], values["document_id"], values["chunk_id"],
                    values["control_policy_hash"], values["treatment_policy_hash"],
                    self._json(policy_set), values.get("input_fingerprint") or "",
                ),
            ).fetchone()
            if row:
                return row
            existing = conn.execute(
                "SELECT * FROM relation_shadow_runs WHERE id=%s", (values["id"],),
            ).fetchone()
            if existing and existing.get("input_fingerprint") != values.get("input_fingerprint", ""):
                raise ValueError("relation shadow run input fingerprint mismatch")
            if not existing:
                raise RuntimeError("relation shadow run could not be started")
            return existing

    def prepare_relation_shadow_work(
        self,
        run_id: str,
        work_items: list[dict[str, Any]],
        *,
        quarantined: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Idempotently persist deterministic batches before any model call."""

        values = [*work_items, *(quarantined or [])]
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM relation_shadow_runs WHERE id=%s FOR UPDATE", (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("relation shadow run 不存在")
            for item in values:
                conn.execute(
                    """INSERT INTO relation_shadow_work_items(
                       id,shadow_run_id,revision_id,project_id,document_id,chunk_id,batch_id,
                       input_fingerprint,candidate_ids,request_json,estimated_input_tokens,
                       max_output_tokens,status,max_attempts,error_message)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s)
                       ON CONFLICT(shadow_run_id,batch_id) DO NOTHING""",
                    (
                        item["id"], run_id, item["revision_id"], item["project_id"],
                        item["document_id"], item["chunk_id"], item["batch_id"],
                        item["input_fingerprint"], self._json(item.get("candidate_ids") or []),
                        self._json(item.get("request_json") or {}),
                        int(item.get("estimated_input_tokens") or 0),
                        int(item.get("max_output_tokens") or 0),
                        item.get("status") or "pending", int(item.get("max_attempts") or 3),
                        item.get("error_message") or "",
                    ),
                )
                existing = conn.execute(
                    """SELECT input_fingerprint FROM relation_shadow_work_items
                       WHERE shadow_run_id=%s AND batch_id=%s""",
                    (run_id, item["batch_id"]),
                ).fetchone()
                if not existing or existing["input_fingerprint"] != item["input_fingerprint"]:
                    raise ValueError("relation shadow work item fingerprint mismatch")
            count = conn.execute(
                "SELECT COUNT(*) AS count FROM relation_shadow_work_items WHERE shadow_run_id=%s",
                (run_id,),
            ).fetchone()["count"]
            conn.execute(
                """UPDATE relation_shadow_runs SET expected_work_items=%s,
                   snapshot_complete=FALSE WHERE id=%s""",
                (count, run_id),
            )
            return conn.execute(
                """SELECT * FROM relation_shadow_work_items
                   WHERE shadow_run_id=%s ORDER BY batch_id""",
                (run_id,),
            ).fetchall()

    def lease_relation_shadow_work(
        self,
        run_id: str,
        owner: str,
        *,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        """Lease one recoverable item without holding a transaction during model I/O."""

        token = uuid.uuid4().hex
        expires = datetime.now(timezone.utc) + timedelta(seconds=max(10, lease_seconds))
        with self.connection() as conn:
            leased = conn.execute(
                """WITH candidate AS (
                       SELECT id FROM relation_shadow_work_items
                       WHERE shadow_run_id=%s AND (
                           (status IN ('pending','retryable_failed') AND next_attempt_at<=NOW())
                           OR (status='leased' AND lease_expires_at<NOW())
                       )
                       ORDER BY created_at,batch_id FOR UPDATE SKIP LOCKED LIMIT 1
                   )
                   UPDATE relation_shadow_work_items item SET status='leased',lease_owner=%s,
                       lease_token=%s,lease_expires_at=%s,attempt=item.attempt+1,updated_at=NOW()
                   FROM candidate WHERE item.id=candidate.id RETURNING item.*""",
                (run_id, owner, token, expires),
            ).fetchone()
            if not leased:
                return None
            attempt_id = "relattempt_" + hashlib.sha256(
                f"{leased['id']}|{leased['attempt']}|{token}".encode("utf-8")
            ).hexdigest()
            conn.execute(
                """INSERT INTO relation_shadow_attempts(
                   id,work_item_id,attempt_no,lease_token,status,started_at)
                   VALUES(%s,%s,%s,%s,'leased',NOW())
                   ON CONFLICT(work_item_id,attempt_no) DO NOTHING""",
                (attempt_id, leased["id"], int(leased["attempt"]), token),
            )
            return leased

    def settle_relation_shadow_work(
        self,
        work_item_id: str,
        lease_token: str,
        *,
        output: dict[str, Any] | None = None,
        error_message: str = "",
        retry_delay_seconds: int = 0,
    ) -> dict[str, Any]:
        """Persist every attempt; first valid terminal CAS owns the work item."""

        output_json = output if output is not None else None
        output_text = self._canonical_json(output_json) if output_json is not None else ""
        output_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest() if output_text else ""
        with self.connection() as conn:
            item = conn.execute(
                "SELECT * FROM relation_shadow_work_items WHERE id=%s FOR UPDATE",
                (work_item_id,),
            ).fetchone()
            if not item:
                raise LookupError("relation shadow work item 不存在")
            attempt = conn.execute(
                """SELECT * FROM relation_shadow_attempts
                   WHERE work_item_id=%s AND lease_token=%s FOR UPDATE""",
                (work_item_id, lease_token),
            ).fetchone()
            if not attempt:
                raise ValueError("relation shadow attempt lease token 不存在")
            attempt_no = int(attempt["attempt_no"])
            attempt_id = str(attempt["id"])
            owns_lease = item["status"] == "leased" and item["lease_token"] == lease_token
            if output_json is not None and owns_lease:
                attempt_status = "accepted"
                terminal_status = "completed"
            elif output_json is not None:
                attempt_status = "superseded"
                terminal_status = None
            elif owns_lease and attempt_no < int(item["max_attempts"]):
                attempt_status = "retryable_failed"
                terminal_status = "retryable_failed"
            elif owns_lease:
                attempt_status = "terminal_failed"
                terminal_status = "terminal_failed"
            else:
                attempt_status = "superseded"
                terminal_status = None
            if attempt["status"] == "leased":
                conn.execute(
                    """UPDATE relation_shadow_attempts SET status=%s,output_json=%s::jsonb,
                       output_hash=%s,error_message=%s,completed_at=NOW()
                       WHERE id=%s AND status='leased'""",
                    (
                        attempt_status, output_text or None, output_hash,
                        error_message or "", attempt_id,
                    ),
                )
            if terminal_status == "completed":
                conn.execute(
                    """UPDATE relation_shadow_work_items SET status='completed',
                       accepted_attempt_id=%s,accepted_output_hash=%s,error_message='',
                       lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
                       completed_at=NOW(),updated_at=NOW() WHERE id=%s""",
                    (attempt_id, output_hash, work_item_id),
                )
            elif terminal_status:
                next_attempt = datetime.now(timezone.utc) + timedelta(
                    seconds=max(0, retry_delay_seconds)
                )
                conn.execute(
                    """UPDATE relation_shadow_work_items SET status=%s,error_message=%s,
                       next_attempt_at=%s,lease_owner=NULL,lease_token=NULL,
                       lease_expires_at=NULL,completed_at=CASE WHEN %s='terminal_failed'
                       THEN NOW() ELSE NULL END,updated_at=NOW() WHERE id=%s""",
                    (terminal_status, error_message or "", next_attempt, terminal_status, work_item_id),
                )
            return conn.execute(
                "SELECT * FROM relation_shadow_work_items WHERE id=%s", (work_item_id,),
            ).fetchone()

    def list_relation_shadow_work(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT item.*,attempt.output_json,attempt.output_hash,
                   GREATEST(0,EXTRACT(EPOCH FROM
                       (attempt.completed_at-attempt.started_at))*1000) AS accepted_latency_ms
                   FROM relation_shadow_work_items item
                   LEFT JOIN relation_shadow_attempts attempt ON attempt.id=item.accepted_attempt_id
                   WHERE item.shadow_run_id=%s ORDER BY item.batch_id""",
                (run_id,),
            ).fetchall()

    def finish_relation_shadow_run(
        self,
        run_id: str,
        *,
        candidates: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        metrics: dict[str, Any],
        status: str = "completed",
        error_message: str = "",
    ) -> dict[str, Any]:
        """Atomically persist treatment-only decisions and final run counters."""

        if status not in {"completed", "failed", "deferred", "terminal_failed", "quarantined"}:
            raise ValueError(f"非法 shadow 状态: {status}")
        candidate_map = {str(item["candidate_id"]): item for item in candidates}
        if any(str(item.get("candidate_id") or "") not in candidate_map for item in decisions):
            raise ValueError("shadow decision 引用了未知 candidate")
        with self.connection() as conn:
            for decision in decisions:
                candidate = candidate_map[str(decision["candidate_id"])]
                conn.execute(
                    """INSERT INTO relation_shadow_decisions(
                       shadow_run_id,candidate_id,source_mention_id,target_mention_id,allowed_predicates,
                       predicate,direction,evidence,reason,status,request_batch_id,metadata,
                       chunk_id,evidence_start,evidence_end,evidence_hash,source_mention_start,
                       source_mention_end,target_mention_start,target_mention_end,policy_version,
                       model_name,config_hash,accepted_attempt_id,output_hash)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,
                              %s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(shadow_run_id,candidate_id) DO UPDATE SET
                       predicate=EXCLUDED.predicate,direction=EXCLUDED.direction,evidence=EXCLUDED.evidence,
                       reason=EXCLUDED.reason,status=EXCLUDED.status,
                       request_batch_id=EXCLUDED.request_batch_id,metadata=EXCLUDED.metadata,
                       evidence_start=EXCLUDED.evidence_start,evidence_end=EXCLUDED.evidence_end,
                       evidence_hash=EXCLUDED.evidence_hash,accepted_attempt_id=EXCLUDED.accepted_attempt_id,
                       output_hash=EXCLUDED.output_hash""",
                    (
                        run_id, decision["candidate_id"], candidate["source_mention_id"],
                        candidate["target_mention_id"], candidate.get("allowed_predicates") or [],
                        decision["predicate"], decision["direction"], decision.get("evidence") or "",
                        decision.get("reason") or "", decision.get("status") or "classified",
                        decision.get("request_batch_id"), self._json(decision.get("metadata")),
                        candidate.get("chunk_id"), decision.get("evidence_start"),
                        decision.get("evidence_end"), decision.get("evidence_hash") or "",
                        candidate.get("source_start_char"), candidate.get("source_end_char"),
                        candidate.get("target_start_char"), candidate.get("target_end_char"),
                        decision.get("policy_version") or "", decision.get("model_name") or "",
                        decision.get("config_hash") or "", decision.get("accepted_attempt_id"),
                        decision.get("output_hash") or "",
                    ),
                )
            work_counts = conn.execute(
                """SELECT COUNT(*) AS total,
                   COUNT(*) FILTER(WHERE status='completed') AS completed,
                   COUNT(*) FILTER(WHERE status IN ('terminal_failed','quarantined')) AS failed,
                   COUNT(*) FILTER(WHERE status IN ('pending','leased','retryable_failed')) AS open
                   FROM relation_shadow_work_items WHERE shadow_run_id=%s""",
                (run_id,),
            ).fetchone()
            if status == "completed" and int(work_counts["open"] or 0) > 0:
                raise ValueError("relation shadow 仍有未完成 work item，不能标记 completed")
            if status == "completed" and int(work_counts["failed"] or 0) > 0:
                raise ValueError("relation shadow 存在 failed work item，不能标记 completed")
            terminal_hashes = [
                row["accepted_output_hash"] or f"{row['status']}:{row['id']}"
                for row in conn.execute(
                    """SELECT id,status,accepted_output_hash FROM relation_shadow_work_items
                       WHERE shadow_run_id=%s ORDER BY id""",
                    (run_id,),
                ).fetchall()
            ]
            artifact_hash = hashlib.sha256(
                self._canonical_json(terminal_hashes).encode("utf-8")
            ).hexdigest()
            row = conn.execute(
                """UPDATE relation_shadow_runs SET status=%s,raw_pairs=%s,blocked_pairs=%s,
                   admitted_candidates=%s,classified=%s,deferred=%s,model_calls=%s,
                   input_tokens=%s,output_tokens=%s,error_message=%s,metadata=%s::jsonb,
                   completed_work_items=%s,failed_work_items=%s,snapshot_complete=%s,
                   artifact_hash=%s,completed_at=NOW() WHERE id=%s RETURNING *""",
                (
                    status, int(metrics.get("raw_pairs") or 0), int(metrics.get("blocked_pairs") or 0),
                    int(metrics.get("admitted_candidates") or 0), int(metrics.get("classified") or 0),
                    int(metrics.get("deferred") or 0), int(metrics.get("model_calls") or 0),
                    int(metrics.get("input_tokens") or 0), int(metrics.get("output_tokens") or 0),
                    error_message or "", self._json(metrics), int(work_counts["completed"] or 0),
                    int(work_counts["failed"] or 0), int(work_counts["open"] or 0) == 0,
                    artifact_hash, run_id,
                ),
            ).fetchone()
            if not row:
                raise LookupError("shadow run 不存在")
            return row

    def create_policy_snapshot(
        self,
        project_id: int,
        policy_set: dict[str, Any],
        *,
        expected_documents: int,
        snapshot_manifest: dict[str, Any] | None = None,
        snapshot_id: str | None = None,
        mode: str = "relation_only",
        manifest_hash: str = "",
        expected_items: int = 0,
        pilot: bool = True,
        created_by: str = "system",
    ) -> dict[str, Any]:
        normalized = normalize_policy_set(policy_set)
        snapshot_id = snapshot_id or uuid.uuid4().hex
        if mode not in {"relation_only", "end_to_end"}:
            raise ValueError("未知 policy snapshot mode")
        with self.connection() as conn:
            project = conn.execute(
                """SELECT graph_generation,active_schema_version
                   FROM projects WHERE id=%s AND is_deleted=FALSE""",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            policy_hash = graph_policy_hash(
                normalized,
                schema_version=int(project.get("active_schema_version") or 1),
            )
            documents = conn.execute(
                """SELECT id,active_revision_id FROM documents
                   WHERE project_id=%s AND status='indexed' AND is_deleted=FALSE
                     AND active_revision_id IS NOT NULL ORDER BY id""",
                (project_id,),
            ).fetchall()
            actual_documents = len(documents)
            if int(expected_documents) != actual_documents:
                raise ValueError(
                    f"策略快照文档数与当前 indexed 文档不一致: expected={expected_documents}, "
                    f"actual={actual_documents}"
                )
            existing = conn.execute(
                "SELECT * FROM policy_snapshots WHERE id=%s FOR UPDATE", (snapshot_id,),
            ).fetchone()
            if existing:
                if (
                    existing.get("manifest_hash") != manifest_hash
                    or existing.get("snapshot_manifest") != (snapshot_manifest or {})
                ):
                    raise ValueError("同一 snapshot_id 对应的不可变 manifest 已改变")
                return existing
            row = conn.execute(
                """INSERT INTO policy_snapshots(
                   id,project_id,policy_set,policy_hash,status,source_generation,
                   expected_documents,snapshot_manifest,mode,manifest_hash,expected_items,
                   pilot,created_by)
                   VALUES(%s,%s,%s::jsonb,%s,'building',%s,%s,%s::jsonb,%s,%s,%s,%s,%s)
                   RETURNING *""",
                (
                    snapshot_id, project_id, self._json(normalized), policy_hash,
                    int(project.get("graph_generation") or 0), actual_documents,
                    self._json(snapshot_manifest), mode, manifest_hash, int(expected_items),
                    bool(pilot), created_by,
                ),
            ).fetchone()
            for document in documents:
                conn.execute(
                    """INSERT INTO policy_snapshot_documents(snapshot_id,document_id,revision_id)
                       VALUES(%s,%s,%s) ON CONFLICT(snapshot_id,document_id) DO NOTHING""",
                    (snapshot_id, document["id"], document["active_revision_id"]),
                )
            return row

    def list_policy_snapshot_inputs(self, project_id: int) -> list[dict[str, Any]]:
        """Return the ordered active revision universe used to freeze a snapshot."""

        with self.connection() as conn:
            documents = conn.execute(
                """SELECT d.id AS document_id,d.active_revision_id AS revision_id,
                          r.content_hash
                   FROM documents d
                   JOIN document_content_revisions r
                     ON r.revision_id=d.active_revision_id AND r.document_id=d.id
                   WHERE d.project_id=%s AND d.status='indexed' AND d.is_deleted=FALSE
                   ORDER BY d.id""",
                (project_id,),
            ).fetchall()
            for document in documents:
                document["chunks"] = conn.execute(
                    """SELECT id AS chunk_id,content_hash FROM chunks
                       WHERE document_id=%s AND revision_id=%s AND is_deleted=FALSE
                       ORDER BY id""",
                    (document["document_id"], document["revision_id"]),
                ).fetchall()
            return documents

    def seed_policy_snapshot_work_items(
        self, snapshot_id: str, items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert the exact deterministic work universe; conflicting input fails closed."""

        with self.connection() as conn:
            snapshot = conn.execute(
                "SELECT project_id,expected_items FROM policy_snapshots WHERE id=%s FOR UPDATE",
                (snapshot_id,),
            ).fetchone()
            if not snapshot:
                raise LookupError("策略快照不存在")
            if int(snapshot["expected_items"]) != len(items):
                raise ValueError("策略快照 expected_items 与待写 work universe 不一致")
            for item in items:
                if int(item["project_id"]) != int(snapshot["project_id"]):
                    raise ValueError("策略快照 work item 项目不一致")
                existing = conn.execute(
                    "SELECT input_hash FROM policy_snapshot_work_items WHERE id=%s",
                    (item["id"],),
                ).fetchone()
                if existing and existing["input_hash"] != item["input_hash"]:
                    raise ValueError("policy snapshot work item input hash 冲突")
                conn.execute(
                    """INSERT INTO policy_snapshot_work_items(
                       id,snapshot_id,phase,phase_order,project_id,document_id,revision_id,
                       chunk_id,input_hash,max_attempts)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        item["id"], snapshot_id, item["phase"], item["phase_order"],
                        item["project_id"], item.get("document_id"), item.get("revision_id"),
                        item.get("chunk_id"), item["input_hash"],
                        int(item.get("max_attempts") or 3),
                    ),
                )
            rows = conn.execute(
                "SELECT * FROM policy_snapshot_work_items WHERE snapshot_id=%s ORDER BY phase_order,id",
                (snapshot_id,),
            ).fetchall()
            if len(rows) != len(items):
                raise ValueError("策略快照 work universe 不完整或包含冲突")
            return rows

    def claim_policy_snapshot_work(
        self, snapshot_id: str, owner: str, *, lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        lease_token = uuid.uuid4().hex
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))
        with self.connection() as conn:
            row = conn.execute(
                """WITH candidate AS (
                       SELECT work.id FROM policy_snapshot_work_items work
                       JOIN policy_snapshots snapshot ON snapshot.id=work.snapshot_id
                       WHERE work.snapshot_id=%s AND snapshot.status='building'
                         AND snapshot.cancel_requested=FALSE
                         AND work.attempt<work.max_attempts
                         AND (
                           (work.status IN ('pending','retryable_failed') AND work.next_attempt_at<=NOW())
                           OR (work.status='leased' AND work.lease_expires_at<NOW())
                         )
                         AND NOT EXISTS (
                           SELECT 1 FROM policy_snapshot_work_items previous
                           WHERE previous.snapshot_id=work.snapshot_id
                             AND previous.phase_order<work.phase_order
                             AND previous.status<>'completed'
                         )
                       ORDER BY work.phase_order,work.id
                       FOR UPDATE OF work SKIP LOCKED LIMIT 1
                   )
                   UPDATE policy_snapshot_work_items work
                   SET status='leased',attempt=attempt+1,lease_owner=%s,lease_token=%s,
                       lease_expires_at=%s,error_message='',updated_at=NOW()
                   FROM candidate WHERE work.id=candidate.id RETURNING work.*""",
                (snapshot_id, owner, lease_token, expires_at),
            ).fetchone()
            return row

    def heartbeat_policy_snapshot_work(
        self, work_id: str, lease_token: str, *, lease_seconds: int = 300,
    ) -> bool:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE policy_snapshot_work_items SET lease_expires_at=%s,updated_at=NOW()
                   WHERE id=%s AND lease_token=%s AND status='leased'
                     AND lease_expires_at>NOW() RETURNING id""",
                (expires_at, work_id, lease_token),
            ).fetchone()
            return bool(row)

    def settle_policy_snapshot_work(
        self,
        work_id: str,
        lease_token: str,
        *,
        artifact: dict[str, Any] | None,
        error_message: str = "",
        retry_delay_seconds: int = 0,
    ) -> dict[str, Any]:
        """CAS one leased item and append exactly one immutable artifact."""

        artifact_hash = (
            hashlib.sha256(self._canonical_json(artifact).encode("utf-8")).hexdigest()
            if artifact is not None else ""
        )
        with self.connection() as conn:
            work = conn.execute(
                "SELECT * FROM policy_snapshot_work_items WHERE id=%s FOR UPDATE", (work_id,),
            ).fetchone()
            if not work:
                raise LookupError("策略快照 work item 不存在")
            existing = conn.execute(
                "SELECT * FROM policy_snapshot_artifacts WHERE work_id=%s", (work_id,),
            ).fetchone()
            if existing:
                if existing["content_hash"] != artifact_hash:
                    raise RuntimeError("immutable artifact hash conflict")
                return work
            if work["status"] != "leased" or work["lease_token"] != lease_token:
                raise RuntimeError("stale policy snapshot lease")
            if not work["lease_expires_at"] or work["lease_expires_at"] <= datetime.now(timezone.utc):
                raise RuntimeError("expired policy snapshot lease")
            if error_message:
                terminal = int(work["attempt"]) >= int(work["max_attempts"])
                status = "terminal_failed" if terminal else "retryable_failed"
                return conn.execute(
                    """UPDATE policy_snapshot_work_items
                       SET status=%s,error_message=%s,lease_owner=NULL,lease_token=NULL,
                           lease_expires_at=NULL,next_attempt_at=NOW()+(%s*INTERVAL '1 second'),
                           updated_at=NOW(),completed_at=CASE WHEN %s THEN NOW() ELSE NULL END
                       WHERE id=%s RETURNING *""",
                    (status, error_message[:2000], max(0, retry_delay_seconds), terminal, work_id),
                ).fetchone()
            if artifact is None:
                raise ValueError("成功 work item 必须提供 artifact")
            conn.execute(
                """INSERT INTO policy_snapshot_artifacts(
                   snapshot_id,artifact_key,work_id,kind,project_id,document_id,revision_id,
                   chunk_id,content_hash,payload)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                (
                    work["snapshot_id"], work_id, work_id, work["phase"], work["project_id"],
                    work.get("document_id"), work.get("revision_id"), work.get("chunk_id"),
                    artifact_hash, self._json(artifact),
                ),
            )
            return conn.execute(
                """UPDATE policy_snapshot_work_items
                   SET status='completed',accepted_artifact_hash=%s,lease_owner=NULL,
                       lease_token=NULL,lease_expires_at=NULL,completed_at=NOW(),updated_at=NOW()
                   WHERE id=%s RETURNING *""",
                (artifact_hash, work_id),
            ).fetchone()

    def finalize_policy_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        coverage_error = ""
        with self.connection() as conn:
            snapshot = conn.execute(
                "SELECT * FROM policy_snapshots WHERE id=%s FOR UPDATE", (snapshot_id,),
            ).fetchone()
            if not snapshot:
                raise LookupError("策略快照不存在")
            counts = conn.execute(
                """SELECT COUNT(*) AS total,
                          COUNT(*) FILTER(WHERE status='completed') AS completed,
                          COUNT(*) FILTER(WHERE status='terminal_failed') AS failed,
                          COUNT(*) FILTER(WHERE status IN ('pending','leased','retryable_failed')) AS open
                   FROM policy_snapshot_work_items WHERE snapshot_id=%s""",
                (snapshot_id,),
            ).fetchone()
            artifacts = conn.execute(
                """SELECT artifact_key,content_hash,kind,payload
                   FROM policy_snapshot_artifacts
                   WHERE snapshot_id=%s ORDER BY artifact_key""",
                (snapshot_id,),
            ).fetchall()
            expected = int(snapshot["expected_items"])
            if (
                int(counts["total"]) != expected
                or int(counts["completed"]) != expected
                or int(counts["failed"]) or int(counts["open"])
                or len(artifacts) != expected
            ):
                raise ValueError("策略快照 item/artifact universe 尚未精确完成")
            incomplete = [
                item for item in artifacts
                if item.get("kind") == "relation"
                and not bool((item.get("payload") or {}).get("coverage_complete"))
            ]
            if incomplete:
                coverage_error = "relation coverage incomplete; snapshot finalization blocked"
                conn.execute(
                    """UPDATE policy_snapshots SET status='failed',coverage_complete=FALSE,
                       failed_items=GREATEST(failed_items,%s),error_message=%s,
                       completed_at=NOW() WHERE id=%s""",
                    (len(incomplete), coverage_error, snapshot_id),
                )
            else:
                root_hash = hashlib.sha256(
                    self._canonical_json([
                        [item["artifact_key"], item["content_hash"]] for item in artifacts
                    ]).encode("utf-8")
                ).hexdigest()
                return conn.execute(
                    """UPDATE policy_snapshots SET status='ready',coverage_complete=TRUE,
                       completed_items=%s,failed_items=0,artifact_root_hash=%s,completed_at=NOW()
                       WHERE id=%s AND status='building' RETURNING *""",
                    (expected, root_hash, snapshot_id),
                ).fetchone() or snapshot
        raise ValueError(coverage_error)

    def request_policy_snapshot_cancel(self, snapshot_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            snapshot = conn.execute(
                """UPDATE policy_snapshots SET cancel_requested=TRUE,status='cancelled',
                   error_message='cancelled by operator',completed_at=NOW()
                   WHERE id=%s AND status='building' RETURNING *""",
                (snapshot_id,),
            ).fetchone()
            if not snapshot:
                snapshot = conn.execute(
                    "SELECT * FROM policy_snapshots WHERE id=%s", (snapshot_id,),
                ).fetchone()
            if not snapshot:
                raise LookupError("策略快照不存在")
            conn.execute(
                """UPDATE policy_snapshot_work_items SET status='cancelled',lease_owner=NULL,
                   lease_token=NULL,lease_expires_at=NULL,completed_at=NOW(),updated_at=NOW()
                   WHERE snapshot_id=%s AND status IN ('pending','leased','retryable_failed')""",
                (snapshot_id,),
            )
            return snapshot

    def policy_snapshot_status(self, snapshot_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            snapshot = conn.execute(
                "SELECT * FROM policy_snapshots WHERE id=%s", (snapshot_id,),
            ).fetchone()
            if not snapshot:
                raise LookupError("策略快照不存在")
            return {
                "snapshot": snapshot,
                "work_items": conn.execute(
                    """SELECT * FROM policy_snapshot_work_items
                       WHERE snapshot_id=%s ORDER BY phase_order,id""",
                    (snapshot_id,),
                ).fetchall(),
                "artifacts": conn.execute(
                    """SELECT * FROM policy_snapshot_artifacts
                       WHERE snapshot_id=%s ORDER BY artifact_key""",
                    (snapshot_id,),
                ).fetchall(),
            }

    def complete_policy_snapshot_document(
        self,
        snapshot_id: str,
        document_id: int,
        *,
        shadow_run_count: int,
        error_message: str = "",
    ) -> dict[str, Any]:
        status = "failed" if error_message else "ready"
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE policy_snapshot_documents
                   SET status=%s,shadow_run_count=%s,error_message=%s,completed_at=NOW()
                   WHERE snapshot_id=%s AND document_id=%s RETURNING *""",
                (status, shadow_run_count, error_message or "", snapshot_id, document_id),
            ).fetchone()
            if not row:
                raise LookupError("策略快照文档不存在")
            return row

    def complete_policy_snapshot(
        self,
        snapshot_id: str,
        *,
        completed_documents: int,
        metrics: dict[str, Any],
        error_message: str = "",
    ) -> dict[str, Any]:
        with self.connection() as conn:
            snapshot = conn.execute(
                "SELECT * FROM policy_snapshots WHERE id=%s FOR UPDATE", (snapshot_id,),
            ).fetchone()
            if not snapshot:
                raise LookupError("策略快照不存在")
            coverage = conn.execute(
                """SELECT COUNT(*) AS total,
                          COUNT(*) FILTER (WHERE status='ready') AS ready,
                          COUNT(*) FILTER (WHERE status='failed') AS failed
                   FROM policy_snapshot_documents WHERE snapshot_id=%s""",
                (snapshot_id,),
            ).fetchone() or {}
            ready_documents = int(coverage.get("ready") or 0)
            if int(completed_documents) != ready_documents:
                raise ValueError("策略快照完成数与数据库文档清单不一致")
            complete = bool(
                not error_message
                and int(coverage.get("total") or 0) == int(snapshot["expected_documents"])
                and ready_documents == int(snapshot["expected_documents"])
                and not int(coverage.get("failed") or 0)
            )
            status = "ready" if complete else "failed"
            return conn.execute(
                """UPDATE policy_snapshots SET status=%s,completed_documents=%s,metrics=%s::jsonb,
                   error_message=%s,completed_at=NOW() WHERE id=%s RETURNING *""",
                (status, ready_documents, self._json(metrics), error_message or "", snapshot_id),
            ).fetchone()

    def activate_policy_snapshot(self, project_id: int, snapshot_id: str) -> dict[str, Any]:
        """Atomically switch a complete snapshot and invalidate the current graph projection."""

        with self.connection() as conn:
            snapshot = conn.execute(
                """SELECT policy_set,coverage_complete FROM policy_snapshots
                   WHERE id=%s AND project_id=%s FOR UPDATE""",
                (snapshot_id, project_id),
            ).fetchone()
            if not snapshot:
                raise LookupError("策略快照不存在")
            if not bool(snapshot.get("coverage_complete")):
                raise ValueError("策略快照 coverage 不完整，禁止激活")
            from .policy import CONTROL_POLICY_SET
            if normalize_policy_set(snapshot["policy_set"]) != dict(CONTROL_POLICY_SET):
                gate = conn.execute(
                    """SELECT id FROM policy_activation_gates
                       WHERE project_id=%s AND snapshot_id=%s
                         AND production_approved=TRUE AND manifest_valid=TRUE
                       FOR UPDATE""",
                    (project_id, snapshot_id),
                ).fetchone()
                if not gate:
                    raise ValueError("非 control 策略缺少有效人工 Gold 门禁，禁止激活")
            return self._activate_policy_snapshot_in_connection(conn, project_id, snapshot_id)

    @staticmethod
    def _activate_policy_snapshot_in_connection(
        conn: psycopg.Connection, project_id: int, snapshot_id: str,
    ) -> dict[str, Any]:
            snapshot = conn.execute(
                """SELECT * FROM policy_snapshots
                   WHERE id=%s AND project_id=%s FOR UPDATE""",
                (snapshot_id, project_id),
            ).fetchone()
            if not snapshot or snapshot.get("status") not in {"ready", "rolled_back"}:
                raise ValueError("策略快照未完整就绪，禁止激活")
            if not bool(snapshot.get("coverage_complete")):
                raise ValueError("策略快照 coverage 不完整，禁止激活")
            coverage = conn.execute(
                """SELECT COUNT(*) AS total,
                          COUNT(*) FILTER (WHERE status='ready') AS ready,
                          COUNT(*) FILTER (WHERE status='failed') AS failed
                   FROM policy_snapshot_documents WHERE snapshot_id=%s""",
                (snapshot_id,),
            ).fetchone()
            if (
                int((coverage or {}).get("total") or 0) != int(snapshot.get("expected_documents") or 0)
                or int((coverage or {}).get("ready") or 0) != int(snapshot.get("expected_documents") or 0)
                or int((coverage or {}).get("failed") or 0)
            ):
                raise ValueError("策略快照文档覆盖不完整，禁止激活")
            policy_set = normalize_policy_set(snapshot["policy_set"])
            conn.execute(
                """UPDATE policy_snapshots SET status='rolled_back'
                   WHERE project_id=%s AND status='active'""",
                (project_id,),
            )
            assignments = ",".join(f"{axis}=%s" for axis in POLICY_AXES)
            row = conn.execute(
                f"""UPDATE projects SET {assignments},graph_generation=graph_generation+1,
                    graph_status=%s,graph_policy_version='',graph_policy_hash='',
                    active_policy_hash=%s,
                    graph_error=%s,updated_at=NOW()
                    WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                (
                    *(policy_set[axis] for axis in POLICY_AXES), GRAPH_STALE,
                    snapshot["policy_hash"],
                    "策略集合已切换，等待全量图重建", project_id,
                ),
            ).fetchone()
            if not row:
                raise LookupError("项目不存在")
            conn.execute(
                """UPDATE policy_snapshots SET status='active',activated_at=NOW()
                   WHERE id=%s""",
                (snapshot_id,),
            )
            return row

    def persist_gold_manifest(
        self,
        project_id: int,
        manifest: dict[str, Any],
        annotations: list[dict[str, Any]],
        validation_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one frozen Gold bundle; an idempotent retry must be byte-equivalent."""

        manifest_hash = hashlib.sha256(self._canonical_json(manifest).encode("utf-8")).hexdigest()
        annotations_sorted = sorted(annotations, key=lambda item: str(item.get("item_id") or ""))
        annotations_hash = hashlib.sha256(
            self._canonical_json(annotations_sorted).encode("utf-8")
        ).hexdigest()
        manifest_id = f"gold_{manifest_hash[:40]}"
        manifest_items = {str(item["id"]): item for item in manifest.get("items") or []}
        if set(manifest_items) != {str(item.get("item_id") or "") for item in annotations_sorted}:
            raise ValueError("Gold manifest 与 annotation item universe 不一致")
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT * FROM gold_manifests WHERE id=%s", (manifest_id,),
            ).fetchone()
            if existing:
                if (
                    existing["manifest_hash"] != manifest_hash
                    or existing["annotations_hash"] != annotations_hash
                ):
                    raise RuntimeError("immutable Gold hash conflict")
                return existing
            row = conn.execute(
                """INSERT INTO gold_manifests(
                   id,project_id,dataset,dataset_version,level,manifest_hash,
                   annotations_hash,manifest,validation_report,trusted_approval)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                   RETURNING *""",
                (
                    manifest_id, project_id, manifest["dataset"], manifest["version"],
                    validation_report.get("level") or "pilot", manifest_hash,
                    annotations_hash, self._json(manifest), self._json(validation_report),
                    self._json(manifest.get("trusted_approval")),
                ),
            ).fetchone()
            for annotation in annotations_sorted:
                item_id = str(annotation["item_id"])
                item = manifest_items[item_id]
                item_hash = hashlib.sha256(
                    self._canonical_json(annotation).encode("utf-8")
                ).hexdigest()
                conn.execute(
                    """INSERT INTO gold_manifest_items(
                       gold_manifest_id,item_id,split,content_hash,item_hash,payload)
                       VALUES(%s,%s,%s,%s,%s,%s::jsonb)""",
                    (
                        manifest_id, item_id, item["split"], item["content_sha256"],
                        item_hash, self._json(annotation),
                    ),
                )
            return row

    def create_quality_evaluation_run(self, values: dict[str, Any]) -> dict[str, Any]:
        """Append a trusted evaluator result and its item-level content hashes."""

        run_id = str(values.get("id") or uuid.uuid4().hex)
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT * FROM quality_evaluation_runs WHERE id=%s", (run_id,),
            ).fetchone()
            if existing:
                if existing.get("report_hash") != values.get("report_hash"):
                    raise RuntimeError("immutable evaluation hash conflict")
                return existing
            row = conn.execute(
                """INSERT INTO quality_evaluation_runs(
                   id,project_id,snapshot_id,control_policy_hash,treatment_policy_hash,
                   dataset,dataset_version,split,manifest,manifest_valid,metrics,denominators,
                   cost_metrics,status,error_message,completed_at,gold_manifest_id,
                   evaluator_version,artifact_root_hash,gold_manifest_hash,gold_items_hash,
                   report_hash,item_hashes,release_level,evaluation_attempt)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,
                          %s::jsonb,%s,%s,CASE WHEN %s='running' THEN NULL ELSE NOW() END,
                          %s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
                   RETURNING *""",
                (
                    run_id, values["project_id"], values.get("snapshot_id"),
                    values["control_policy_hash"], values["treatment_policy_hash"],
                    values["dataset"], values["dataset_version"], values.get("split") or "combined",
                    self._json(values.get("manifest")), bool(values.get("manifest_valid")),
                    self._json(values.get("metrics")), self._json(values.get("denominators")),
                    self._json(values.get("cost_metrics")), values.get("status") or "completed",
                    values.get("error_message") or "", values.get("status") or "completed",
                    values.get("gold_manifest_id"), values.get("evaluator_version") or "quality-v2",
                    values.get("artifact_root_hash") or "", values.get("gold_manifest_hash") or "",
                    values.get("gold_items_hash") or "", values.get("report_hash") or "",
                    self._json(values.get("item_hashes") or []),
                    values.get("release_level") or "pilot",
                    int(values.get("evaluation_attempt") or 1),
                ),
            ).fetchone()
            for item in values.get("evaluation_items") or []:
                result = item.get("result") or {}
                result_hash = hashlib.sha256(
                    self._canonical_json(result).encode("utf-8")
                ).hexdigest()
                conn.execute(
                    """INSERT INTO quality_evaluation_items(
                       evaluation_run_id,item_id,artifact_hash,result_hash,result)
                       VALUES(%s,%s,%s,%s,%s::jsonb)""",
                    (
                        run_id, item["item_id"], item["artifact_hash"], result_hash,
                        self._json(result),
                    ),
                )
            return row

    def get_quality_evaluation_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT evaluation.*,snapshot.status AS snapshot_status,
                          snapshot.manifest_hash AS current_snapshot_manifest_hash,
                          snapshot.artifact_root_hash AS current_artifact_root_hash,
                          snapshot.expected_items AS snapshot_expected_items,
                          snapshot.completed_items AS snapshot_completed_items,
                          snapshot.failed_items AS snapshot_failed_items,
                          snapshot.coverage_complete AS snapshot_coverage_complete,
                          gold.manifest_hash AS current_gold_manifest_hash,
                          gold.annotations_hash AS current_gold_items_hash,
                          gold.manifest AS current_gold_manifest,
                          gold.level AS current_gold_level,
                          gold.validation_report AS current_gold_validation_report
                   FROM quality_evaluation_runs evaluation
                   JOIN policy_snapshots snapshot ON snapshot.id=evaluation.snapshot_id
                   JOIN gold_manifests gold ON gold.id=evaluation.gold_manifest_id
                   WHERE evaluation.id=%s""",
                (run_id,),
            ).fetchone()
            if row:
                row["evaluation_items"] = conn.execute(
                    """SELECT * FROM quality_evaluation_items
                       WHERE evaluation_run_id=%s ORDER BY item_id""",
                    (run_id,),
                ).fetchall()
                gold_items = conn.execute(
                    """SELECT payload FROM gold_manifest_items
                       WHERE gold_manifest_id=%s ORDER BY item_id""",
                    (row["gold_manifest_id"],),
                ).fetchall()
                row["recomputed_gold_manifest_hash"] = hashlib.sha256(
                    self._canonical_json(row["current_gold_manifest"]).encode("utf-8")
                ).hexdigest()
                row["recomputed_gold_items_hash"] = hashlib.sha256(
                    self._canonical_json([item["payload"] for item in gold_items]).encode("utf-8")
                ).hexdigest()
                row["current_evaluation_item_hashes"] = [
                    str(item["artifact_hash"]) for item in row["evaluation_items"]
                ]
                row["evaluation_item_results_valid"] = all(
                    hashlib.sha256(
                        self._canonical_json(item["result"]).encode("utf-8")
                    ).hexdigest() == item["result_hash"]
                    for item in row["evaluation_items"]
                )
            return row

    def create_policy_activation_gate(self, values: dict[str, Any]) -> dict[str, Any]:
        """Append one gate decision. Existing gate facts can never be overwritten."""

        gate_id = str(values.get("id") or uuid.uuid4().hex)
        with self.connection() as conn:
            snapshot = conn.execute(
                """SELECT coverage_complete FROM policy_snapshots
                   WHERE id=%s AND project_id=%s""",
                (values["snapshot_id"], values["project_id"]),
            ).fetchone()
            if not snapshot or not bool(snapshot.get("coverage_complete")):
                raise ValueError("策略快照 coverage 不完整，禁止创建激活门禁")
            existing = conn.execute(
                "SELECT * FROM policy_activation_gates WHERE evaluation_run_id=%s",
                (values["evaluation_run_id"],),
            ).fetchone()
            if existing:
                return existing
            return conn.execute(
                """INSERT INTO policy_activation_gates(
                   id,project_id,snapshot_id,evaluation_run_id,manifest_valid,
                   engineering_complete,production_approved,failed_gates,gate_report,
                   approved_by,approved_at)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,
                          CASE WHEN %s THEN NOW() ELSE NULL END)
                   RETURNING *""",
                (
                    gate_id, values["project_id"], values["snapshot_id"],
                    values["evaluation_run_id"], bool(values.get("manifest_valid")),
                    bool(values.get("engineering_complete")),
                    bool(values.get("production_approved")), values.get("failed_gates") or [],
                    self._json(values.get("gate_report")), values.get("approved_by"),
                    bool(values.get("production_approved")),
                ),
            ).fetchone()

    def activate_policy_snapshot_with_gate(
        self, project_id: int, snapshot_id: str,
    ) -> dict[str, Any]:
        """Activate treatment only after an immutable production approval gate."""

        with self.connection() as conn:
            gate = conn.execute(
                """SELECT * FROM policy_activation_gates
                   WHERE project_id=%s AND snapshot_id=%s
                     AND production_approved=TRUE AND manifest_valid=TRUE
                   FOR UPDATE""",
                (project_id, snapshot_id),
            ).fetchone()
            if not gate:
                raise ValueError("缺少有效人工 Gold 和 production_approved 门禁，禁止激活")
            return self._activate_policy_snapshot_in_connection(conn, project_id, snapshot_id)

    def record_policy_rollout(
        self, project_id: int, snapshot_id: str, rollout_stage: str, *,
        previous_snapshot_id: str | None = None,
        rollback_snapshot_id: str | None = None,
        percentage: float = 0.0, status: str = "planned", reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stages = {
            "control", "shadow", "canary", "limited", "expanded",
            "new_project_default", "migrated", "rolled_back",
        }
        if rollout_stage not in stages:
            raise ValueError("未知 rollout stage")
        rollout_id = uuid.uuid4().hex
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO policy_rollouts(
                   id,project_id,snapshot_id,rollout_stage,previous_snapshot_id,
                   rollback_snapshot_id,percentage,status,reason,metadata)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) RETURNING *""",
                (
                    rollout_id, project_id, snapshot_id, rollout_stage,
                    previous_snapshot_id, rollback_snapshot_id, min(100.0, max(0.0, percentage)),
                    status, reason, self._json(metadata),
                ),
            ).fetchone()

    @staticmethod
    def _resolution_event_id(project_id: int, event_type: str, material: str) -> str:
        digest = hashlib.sha256(
            f"{project_id}\x1f{event_type}\x1f{material}\x1f{uuid.uuid4().hex}".encode("utf-8")
        ).hexdigest()
        return f"entity_event_{digest}"

    @staticmethod
    def _mark_graph_stale_in_transaction(
        conn: psycopg.Connection,
        project_id: int,
        reason: str,
    ) -> dict[str, Any]:
        row = conn.execute(
            """UPDATE projects SET graph_generation=graph_generation+1,graph_status=%s,
               graph_policy_version='',graph_policy_hash='',graph_error=%s,updated_at=NOW()
               WHERE id=%s AND is_deleted=FALSE RETURNING *""",
            (GRAPH_STALE, reason, project_id),
        ).fetchone()
        if not row:
            raise LookupError("项目不存在")
        return row

    def add_entity_cannot_link(
        self,
        project_id: int,
        *,
        left_entity_id: int | None = None,
        right_entity_id: int | None = None,
        left_mention_key: str = "",
        right_mention_key: str = "",
        reason: str,
        policy_version: str,
        operator: str = "system",
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("cannot-link 必须说明原因")
        left = str(left_entity_id or left_mention_key)
        right = str(right_entity_id or right_mention_key)
        if not left or not right or left == right:
            raise ValueError("cannot-link 两端必须存在且不同")
        link_id = "cannot_link_" + hashlib.sha256(
            f"{project_id}\x1f{min(left,right)}\x1f{max(left,right)}\x1f{policy_version}".encode()
        ).hexdigest()
        event_id = self._resolution_event_id(project_id, "cannot_link", link_id)
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO entity_resolution_events(
                   id,project_id,event_type,source_entity_id,target_entity_id,policy_version,
                   operator,reason,evidence,before_state,after_state,reversible)
                   VALUES(%s,%s,'cannot_link',%s,%s,%s,%s,%s,%s::jsonb,'{}'::jsonb,
                          %s::jsonb,TRUE)""",
                (
                    event_id, project_id, left_entity_id, right_entity_id, policy_version,
                    operator, reason, self._json(evidence), self._json({"link_id": link_id}),
                ),
            )
            return conn.execute(
                """INSERT INTO entity_cannot_links(
                   id,project_id,left_entity_id,right_entity_id,left_mention_key,right_mention_key,
                   reason,evidence,policy_version,operator,active)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,TRUE)
                   ON CONFLICT(id) DO UPDATE SET reason=EXCLUDED.reason,evidence=EXCLUDED.evidence,
                   operator=EXCLUDED.operator,active=TRUE RETURNING *""",
                (
                    link_id, project_id, left_entity_id, right_entity_id,
                    left_mention_key, right_mention_key, reason, self._json(evidence),
                    policy_version, operator,
                ),
            ).fetchone()

    def list_entity_resolution_events(self, project_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM entity_resolution_events
                   WHERE project_id=%s ORDER BY created_at DESC,id""",
                (project_id,),
            ).fetchall()

    @staticmethod
    def _refresh_relation_projections(
        conn: psycopg.Connection,
        project_id: int,
        policy_hash: str,
        generation: int,
    ) -> None:
        revisioned = all((
            Database._column_exists(conn, "documents", "active_revision_id"),
            Database._column_exists(conn, "relation_assertions", "revision_id"),
            Database._column_exists(conn, "mention_relation_assertions", "revision_id"),
            Database._column_exists(conn, "entity_mentions", "revision_id"),
        ))
        active_revision_join = (
            "AND active_document.active_revision_id=r.revision_id" if revisioned else ""
        )
        fact_revision_filter = "AND fact.revision_id=r.revision_id" if revisioned else ""
        source_revision_filter = (
            "AND source_mention.revision_id=r.revision_id" if revisioned else ""
        )
        target_revision_filter = (
            "AND target_mention.revision_id=r.revision_id" if revisioned else ""
        )
        conn.execute(
            f"""INSERT INTO canonical_relation_projections(
               project_id,assertion_id,source_entity_id,target_entity_id,
               assignment_generation,policy_hash,publishable,quarantine_reason)
               SELECT r.project_id,r.id,
                      COALESCE(source_assignment.canonical_entity_id,
                               source_redirect.target_entity_id,r.source_entity_id),
                      COALESCE(target_assignment.canonical_entity_id,
                               target_redirect.target_entity_id,r.target_entity_id),
                      %s,%s,
                      COALESCE(source_assignment.canonical_entity_id,
                               source_redirect.target_entity_id,r.source_entity_id)
                        <>COALESCE(target_assignment.canonical_entity_id,
                                   target_redirect.target_entity_id,r.target_entity_id),
                      CASE WHEN COALESCE(source_assignment.canonical_entity_id,
                                         source_redirect.target_entity_id,r.source_entity_id)
                                     =COALESCE(target_assignment.canonical_entity_id,
                                               target_redirect.target_entity_id,r.target_entity_id)
                           THEN 'self relation after entity assignment change' ELSE '' END
               FROM relation_assertions r
               JOIN documents active_document
                 ON active_document.id=r.document_id
                {active_revision_join}
                AND active_document.is_deleted=FALSE
               LEFT JOIN LATERAL (
                   SELECT fact.source_mention_key,fact.target_mention_key
                   FROM mention_relation_assertions fact
                   WHERE fact.project_id=r.project_id AND fact.canonical_assertion_id=r.id
                     {fact_revision_filter}
                   ORDER BY fact.created_at DESC,fact.id DESC LIMIT 1
               ) mention_fact ON TRUE
               LEFT JOIN entity_mentions source_mention
                ON source_mention.project_id=r.project_id
                AND source_mention.mention_key=mention_fact.source_mention_key
                {source_revision_filter}
                AND source_mention.is_deleted=FALSE
               LEFT JOIN entity_assignments source_assignment
                 ON source_assignment.project_id=r.project_id
                AND source_assignment.mention_id=source_mention.id
                AND source_assignment.active=TRUE
               LEFT JOIN entity_mentions target_mention
                ON target_mention.project_id=r.project_id
                AND target_mention.mention_key=mention_fact.target_mention_key
                {target_revision_filter}
                AND target_mention.is_deleted=FALSE
               LEFT JOIN entity_assignments target_assignment
                 ON target_assignment.project_id=r.project_id
                AND target_assignment.mention_id=target_mention.id
                AND target_assignment.active=TRUE
               LEFT JOIN canonical_entity_redirects source_redirect
                 ON source_redirect.project_id=r.project_id
                AND source_redirect.source_entity_id=r.source_entity_id
                AND source_redirect.active=TRUE
               LEFT JOIN canonical_entity_redirects target_redirect
                 ON target_redirect.project_id=r.project_id
                AND target_redirect.source_entity_id=r.target_entity_id
                AND target_redirect.active=TRUE
               WHERE r.project_id=%s AND r.is_deleted=FALSE
               ON CONFLICT(project_id,assertion_id) DO UPDATE SET
               source_entity_id=EXCLUDED.source_entity_id,
               target_entity_id=EXCLUDED.target_entity_id,
               assignment_generation=EXCLUDED.assignment_generation,
               policy_hash=EXCLUDED.policy_hash,publishable=EXCLUDED.publishable,
               quarantine_reason=EXCLUDED.quarantine_reason,updated_at=NOW()""",
            (generation, policy_hash, project_id),
        )
        schema_row = conn.execute(
            """SELECT schema_json FROM graph_schemas
               WHERE project_id=%s AND status='active' ORDER BY version DESC LIMIT 1""",
            (project_id,),
        ).fetchone()
        definitions = {
            str(item.get("name") or ""): item
            for item in ((schema_row or {}).get("schema_json") or {}).get("relation_types", [])
        }
        if not definitions:
            return
        rows = conn.execute(
            f"""SELECT projection.assertion_id,r.relation_type,
                      source.entity_type AS source_type,target.entity_type AS target_type
               FROM canonical_relation_projections projection
               JOIN relation_assertions r ON r.id=projection.assertion_id
               JOIN documents active_document
                 ON active_document.id=r.document_id
                {active_revision_join}
                AND active_document.is_deleted=FALSE
               JOIN canonical_entities source ON source.id=projection.source_entity_id
               JOIN canonical_entities target ON target.id=projection.target_entity_id
               WHERE projection.project_id=%s AND projection.publishable=TRUE""",
            (project_id,),
        ).fetchall()
        for row in rows:
            definition = definitions.get(str(row["relation_type"]))
            if not definition:
                conn.execute(
                    """UPDATE canonical_relation_projections SET publishable=FALSE,
                       quarantine_reason='relation type missing from active schema',updated_at=NOW()
                       WHERE project_id=%s AND assertion_id=%s""",
                    (project_id, row["assertion_id"]),
                )
                continue
            sources = set(definition.get("source_types") or ["*"])
            targets = set(definition.get("target_types") or ["*"])
            if ("*" not in sources and row["source_type"] not in sources) or (
                "*" not in targets and row["target_type"] not in targets
            ):
                conn.execute(
                    """UPDATE canonical_relation_projections SET publishable=FALSE,
                       quarantine_reason='domain/range invalid after entity assignment change',
                       updated_at=NOW() WHERE project_id=%s AND assertion_id=%s""",
                    (project_id, row["assertion_id"]),
                )

    @staticmethod
    def _refresh_entity_document_projections(
        conn: psycopg.Connection,
        project_id: int,
        entity_ids: list[int],
    ) -> None:
        """Rebuild bounded canonical-to-document provenance from active assignments."""

        ids = sorted({int(value) for value in entity_ids})
        if not ids:
            return
        revisioned = all((
            Database._column_exists(conn, "documents", "active_revision_id"),
            Database._column_exists(conn, "canonical_entity_documents", "revision_id"),
            Database._column_exists(conn, "entity_mentions", "revision_id"),
        ))
        active_projection_filter = (
            "AND document.active_revision_id=source.revision_id" if revisioned else ""
        )
        active_mention_filter = (
            "AND document.active_revision_id=mention.revision_id" if revisioned else ""
        )
        conflict_target = (
            "(revision_id,canonical_entity_id)"
            if revisioned else "(canonical_entity_id,document_id)"
        )
        conn.execute(
            f"""UPDATE canonical_entity_documents source
               SET is_deleted=TRUE,updated_at=NOW()
               FROM documents document
               WHERE source.project_id=%s AND source.canonical_entity_id=ANY(%s)
                 AND document.id=source.document_id
                 {active_projection_filter}
                 AND document.is_deleted=FALSE""",
            (project_id, ids),
        )
        conn.execute(
            f"""INSERT INTO canonical_entity_documents(
                   project_id,canonical_entity_id,document_id)
               SELECT assignment.project_id,assignment.canonical_entity_id,mention.document_id
               FROM entity_assignments assignment
               JOIN entity_mentions mention ON mention.id=assignment.mention_id
               JOIN documents document ON document.id=mention.document_id
                 {active_mention_filter}
               WHERE assignment.project_id=%s AND assignment.active=TRUE
                 AND assignment.canonical_entity_id=ANY(%s)
                 AND mention.is_deleted=FALSE AND document.is_deleted=FALSE
               GROUP BY assignment.project_id,assignment.canonical_entity_id,mention.document_id
               ON CONFLICT{conflict_target} DO UPDATE SET
               project_id=EXCLUDED.project_id,is_deleted=FALSE,updated_at=NOW()""",
            (project_id, ids),
        )

    @staticmethod
    def _refresh_assignment_aliases(
        conn: psycopg.Connection,
        project_id: int,
        entity_ids: list[int],
        policy_version: str,
    ) -> None:
        """Refresh assignment-derived aliases while retaining audited manual aliases."""

        ids = sorted({int(value) for value in entity_ids})
        if not ids:
            return
        revisioned = all((
            Database._column_exists(conn, "documents", "active_revision_id"),
            Database._column_exists(conn, "entity_mentions", "revision_id"),
        ))
        active_mention_filter = (
            "AND document.active_revision_id=mention.revision_id" if revisioned else ""
        )
        conn.execute(
            """UPDATE canonical_entity_aliases SET active=FALSE,updated_at=NOW()
               WHERE project_id=%s AND canonical_entity_id=ANY(%s)
                 AND alias_kind='assignment'""",
            (project_id, ids),
        )
        conn.execute(
            f"""INSERT INTO canonical_entity_aliases(
                   project_id,canonical_entity_id,entity_type,alias,normalized_alias,
                   alias_kind,active,evidence,policy_version)
               SELECT DISTINCT ON (assignment.canonical_entity_id,mention.normalized_name)
                   assignment.project_id,assignment.canonical_entity_id,entity.entity_type,
                   mention.mention_text,mention.normalized_name,'assignment',TRUE,
                   jsonb_build_object('mention_key',mention.mention_key),%s
               FROM entity_assignments assignment
               JOIN entity_mentions mention ON mention.id=assignment.mention_id
               JOIN documents document ON document.id=mention.document_id
                 {active_mention_filter}
                 AND document.is_deleted=FALSE
               JOIN canonical_entities entity ON entity.id=assignment.canonical_entity_id
               WHERE assignment.project_id=%s AND assignment.active=TRUE
                 AND assignment.canonical_entity_id=ANY(%s)
                 AND mention.is_deleted=FALSE AND mention.normalized_name<>''
               ORDER BY assignment.canonical_entity_id,mention.normalized_name,mention.id
               ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind)
               DO UPDATE SET alias=EXCLUDED.alias,entity_type=EXCLUDED.entity_type,
               active=TRUE,evidence=EXCLUDED.evidence,policy_version=EXCLUDED.policy_version,
               updated_at=NOW()""",
            (policy_version, project_id, ids),
        )
        conn.execute(
            """UPDATE canonical_entities entity SET
                   aliases=COALESCE(alias_set.aliases,'{}'::TEXT[]),
                   normalized_aliases=COALESCE(alias_set.normalized_aliases,'{}'::TEXT[]),
                   updated_at=NOW()
               FROM (
                   SELECT canonical_entity_id,
                          ARRAY_AGG(DISTINCT alias ORDER BY alias) AS aliases,
                          ARRAY_AGG(DISTINCT normalized_alias ORDER BY normalized_alias)
                              AS normalized_aliases
                   FROM canonical_entity_aliases
                   WHERE project_id=%s AND canonical_entity_id=ANY(%s) AND active=TRUE
                     AND alias_kind<>'canonical_name'
                   GROUP BY canonical_entity_id
               ) alias_set
               WHERE entity.project_id=%s AND entity.id=alias_set.canonical_entity_id""",
            (project_id, ids, project_id),
        )
        conn.execute(
            """UPDATE canonical_entities entity SET aliases='{}',normalized_aliases='{}',
                   updated_at=NOW()
               WHERE entity.project_id=%s AND entity.id=ANY(%s)
                 AND NOT EXISTS(
                     SELECT 1 FROM canonical_entity_aliases alias
                     WHERE alias.project_id=entity.project_id
                       AND alias.canonical_entity_id=entity.id AND alias.active=TRUE
                       AND alias.alias_kind<>'canonical_name'
                 )""",
            (project_id, ids),
        )
    def manual_merge_entities(
        self,
        project_id: int,
        source_entity_id: int,
        target_entity_id: int,
        *,
        reason: str,
        policy_version: str,
        operator: str = "manual",
        scores: dict[str, Any] | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a reversible merge assignment without rewriting assertion facts."""

        if source_entity_id == target_entity_id:
            raise ValueError("不能合并同一实体")
        if not reason.strip():
            raise ValueError("人工合并必须说明原因")
        with self.connection() as conn:
            redirect_rows = conn.execute(
                """SELECT source_entity_id,target_entity_id
                   FROM canonical_entity_redirects
                   WHERE project_id=%s AND active=TRUE FOR SHARE""",
                (project_id,),
            ).fetchall()
            redirect_map = {
                int(item["source_entity_id"]): int(item["target_entity_id"])
                for item in redirect_rows
            }

            def root(entity_id: int) -> int:
                seen: set[int] = set()
                while entity_id in redirect_map and entity_id not in seen:
                    seen.add(entity_id)
                    entity_id = redirect_map[entity_id]
                return entity_id

            source_entity_id = root(int(source_entity_id))
            target_entity_id = root(int(target_entity_id))
            if source_entity_id == target_entity_id:
                raise ValueError("两个实体已经指向同一当前根实体")
            event_id = self._resolution_event_id(
                project_id, "merge", f"{source_entity_id}:{target_entity_id}",
            )
            revisioned = all((
                self._column_exists(conn, "documents", "active_revision_id"),
                self._column_exists(conn, "entity_mentions", "revision_id"),
                self._column_exists(conn, "canonical_entity_documents", "revision_id"),
            ))
            active_mention_filter = (
                "AND d.active_revision_id=m.revision_id" if revisioned else ""
            )
            active_assignment_filter = (
                "AND document.active_revision_id=mention.revision_id" if revisioned else ""
            )
            active_projection_filter = (
                "AND document.active_revision_id=source.revision_id" if revisioned else ""
            )
            projection_conflict_target = (
                "(revision_id,canonical_entity_id)"
                if revisioned else "(canonical_entity_id,document_id)"
            )
            entities = conn.execute(
                """SELECT * FROM canonical_entities
                   WHERE project_id=%s AND id=ANY(%s) FOR UPDATE""",
                (project_id, [source_entity_id, target_entity_id]),
            ).fetchall()
            if {int(item["id"]) for item in entities} != {source_entity_id, target_entity_id}:
                raise LookupError("待合并实体不存在或不属于该项目")
            mention_roots = {
                str(item["mention_key"]): root(int(item["canonical_entity_id"]))
                for item in conn.execute(
                    """SELECT mention.mention_key,
                              COALESCE(assignment.canonical_entity_id,
                                       mention.canonical_entity_id) AS canonical_entity_id
                       FROM entity_mentions mention
                       LEFT JOIN entity_assignments assignment
                         ON assignment.project_id=mention.project_id
                        AND assignment.mention_id=mention.id AND assignment.active=TRUE
                       WHERE mention.project_id=%s AND mention.is_deleted=FALSE
                         AND COALESCE(assignment.canonical_entity_id,
                                      mention.canonical_entity_id)=ANY(%s)""",
                    (project_id, [source_entity_id, target_entity_id]),
                ).fetchall()
            }
            cannot_links = conn.execute(
                """SELECT left_entity_id,right_entity_id,left_mention_key,right_mention_key
                   FROM entity_cannot_links WHERE project_id=%s AND active=TRUE""",
                (project_id,),
            ).fetchall()
            blocked = False
            for link in cannot_links:
                left_root = (
                    root(int(link["left_entity_id"]))
                    if link.get("left_entity_id") is not None
                    else mention_roots.get(str(link.get("left_mention_key") or ""))
                )
                right_root = (
                    root(int(link["right_entity_id"]))
                    if link.get("right_entity_id") is not None
                    else mention_roots.get(str(link.get("right_mention_key") or ""))
                )
                if {left_root, right_root} == {source_entity_id, target_entity_id}:
                    blocked = True
                    break
            if blocked:
                raise ValueError("实体存在有效 cannot-link，禁止合并")
            conn.execute(
                f"""INSERT INTO entity_assignments(
                   project_id,mention_id,canonical_entity_id,assignment_generation,
                   policy_version,reason,active)
                   SELECT m.project_id,m.id,m.canonical_entity_id,1,%s,'legacy assignment',TRUE
                   FROM entity_mentions m
                   JOIN documents d ON d.id=m.document_id
                     {active_mention_filter} AND d.is_deleted=FALSE
                   WHERE m.project_id=%s AND m.is_deleted=FALSE
                   ON CONFLICT(project_id,mention_id) WHERE active=TRUE DO NOTHING""",
                (policy_version, project_id),
            )
            assignments = conn.execute(
                f"""SELECT assignment.id,assignment.mention_id,assignment.canonical_entity_id,
                          assignment.assignment_generation
                   FROM entity_assignments assignment
                   JOIN entity_mentions mention ON mention.id=assignment.mention_id
                   JOIN documents document ON document.id=mention.document_id
                     {active_assignment_filter}
                     AND document.is_deleted=FALSE
                   WHERE assignment.project_id=%s AND assignment.canonical_entity_id=%s
                     AND assignment.active=TRUE ORDER BY assignment.mention_id FOR UPDATE""",
                (project_id, source_entity_id),
            ).fetchall()
            source_documents = conn.execute(
                f"""SELECT source.document_id FROM canonical_entity_documents source
                   JOIN documents document ON document.id=source.document_id
                     {active_projection_filter}
                     AND document.is_deleted=FALSE
                   WHERE source.project_id=%s AND source.canonical_entity_id=%s
                     AND source.is_deleted=FALSE ORDER BY source.document_id""",
                (project_id, source_entity_id),
            ).fetchall()
            before_state = {
                "assignments": assignments,
                "source_document_ids": [int(item["document_id"]) for item in source_documents],
            }
            conn.execute(
                """INSERT INTO entity_resolution_events(
                   id,project_id,event_type,source_entity_id,target_entity_id,policy_version,
                   model_name,operator,reason,scores,evidence,before_state,after_state,reversible)
                   VALUES(%s,%s,'merge',%s,%s,%s,NULL,%s,%s,%s::jsonb,%s::jsonb,
                          %s::jsonb,%s::jsonb,TRUE)""",
                (
                    event_id, project_id, source_entity_id, target_entity_id, policy_version,
                    operator, reason, self._json(scores), self._json(evidence),
                    self._json(before_state), self._json({"redirect": [source_entity_id, target_entity_id]}),
                ),
            )
            assignment_ids = [int(item["id"]) for item in assignments]
            if assignment_ids:
                conn.execute(
                    "UPDATE entity_assignments SET active=FALSE WHERE id=ANY(%s)",
                    (assignment_ids,),
                )
                conn.execute(
                    """INSERT INTO entity_assignments(
                       project_id,mention_id,canonical_entity_id,assignment_generation,
                       policy_version,event_id,reason,active)
                       SELECT project_id,mention_id,%s,assignment_generation+1,%s,%s,%s,TRUE
                       FROM entity_assignments WHERE id=ANY(%s)""",
                    (target_entity_id, policy_version, event_id, reason, assignment_ids),
                )
            conn.execute(
                """INSERT INTO canonical_entity_redirects(
                   project_id,source_entity_id,target_entity_id,event_id,active)
                   VALUES(%s,%s,%s,%s,TRUE)
                   ON CONFLICT(project_id,source_entity_id) DO UPDATE SET
                   target_entity_id=EXCLUDED.target_entity_id,event_id=EXCLUDED.event_id,active=TRUE""",
                (project_id, source_entity_id, target_entity_id, event_id),
            )
            conn.execute(
                f"""INSERT INTO canonical_entity_documents(project_id,canonical_entity_id,document_id)
                   SELECT project_id,%s,document_id FROM canonical_entity_documents
                   WHERE project_id=%s AND canonical_entity_id=%s AND is_deleted=FALSE
                     AND EXISTS(
                         SELECT 1 FROM documents document
                         WHERE document.id=canonical_entity_documents.document_id
                           {active_projection_filter.replace('source.', 'canonical_entity_documents.')}
                           AND document.is_deleted=FALSE
                     )
                   ON CONFLICT{projection_conflict_target} DO UPDATE
                   SET project_id=EXCLUDED.project_id,is_deleted=FALSE,updated_at=NOW()""",
                (target_entity_id, project_id, source_entity_id),
            )
            conn.execute(
                f"""UPDATE canonical_entity_documents source
                   SET is_deleted=TRUE,updated_at=NOW()
                   FROM documents document
                   WHERE source.project_id=%s AND source.canonical_entity_id=%s
                     AND document.id=source.document_id
                     {active_projection_filter}
                     AND document.is_deleted=FALSE""",
                (project_id, source_entity_id),
            )
            conn.execute(
                """INSERT INTO canonical_entity_aliases(
                   project_id,canonical_entity_id,entity_type,alias,normalized_alias,
                   alias_kind,active,evidence,policy_version)
                   SELECT project_id,%s,entity_type,alias,normalized_alias,'merge',TRUE,
                          jsonb_build_object(
                              'source_entity_id',%s::BIGINT,'event_id',%s::TEXT
                          ),%s
                   FROM canonical_entity_aliases
                   WHERE project_id=%s AND canonical_entity_id=%s AND active=TRUE
                   ON CONFLICT(project_id,canonical_entity_id,normalized_alias,alias_kind)
                   DO UPDATE SET active=TRUE,evidence=EXCLUDED.evidence,updated_at=NOW()""",
                (
                    target_entity_id, source_entity_id, event_id, policy_version,
                    project_id, source_entity_id,
                ),
            )
            conn.execute(
                "UPDATE canonical_entities SET is_deleted=TRUE,updated_at=NOW() WHERE id=%s",
                (source_entity_id,),
            )
            project = self._mark_graph_stale_in_transaction(
                conn, project_id, f"实体合并 {source_entity_id}->{target_entity_id}，等待重建",
            )
            affected_entity_ids = [source_entity_id, target_entity_id]
            self._refresh_entity_document_projections(
                conn, project_id, affected_entity_ids,
            )
            self._refresh_assignment_aliases(
                conn, project_id, affected_entity_ids, policy_version,
            )
            policy_hash = str(project.get("active_policy_hash") or GRAPH_POLICY_HASH)
            self._refresh_relation_projections(
                conn, project_id, policy_hash, int(project["graph_generation"]),
            )
            self._refresh_entity_source_counts(conn, project_id, affected_entity_ids)
            return {
                "event_id": event_id,
                "project": project,
                "assignments": len(assignments),
                "affected_entity_ids": affected_entity_ids,
            }

    def undo_entity_resolution_event(
        self,
        project_id: int,
        event_id: str,
        *,
        operator: str = "manual",
        reason: str = "undo",
    ) -> dict[str, Any]:
        """Undo a reversible merge by appending a compensating event."""

        undo_id = self._resolution_event_id(project_id, "undo", event_id)
        with self.connection() as conn:
            revisioned = all((
                self._column_exists(conn, "documents", "active_revision_id"),
                self._column_exists(conn, "entity_mentions", "revision_id"),
                self._column_exists(conn, "canonical_entity_documents", "revision_id"),
            ))
            active_mention_filter = (
                "AND document.active_revision_id=mention.revision_id" if revisioned else ""
            )
            active_projection_filter = (
                "AND document.active_revision_id=source.revision_id" if revisioned else ""
            )
            event = conn.execute(
                """SELECT * FROM entity_resolution_events
                   WHERE id=%s AND project_id=%s FOR UPDATE""",
                (event_id, project_id),
            ).fetchone()
            if not event or not event.get("reversible") or event.get("reversed_by_event_id"):
                raise ValueError("事件不存在、不可逆或已回滚")
            if event.get("event_type") != "merge":
                raise ValueError("当前只支持回滚 merge 事件")
            source_id = int(event["source_entity_id"])
            assignment_ids = [int(item["id"]) for item in (event.get("before_state") or {}).get("assignments", [])]
            if assignment_ids:
                current_count = int(conn.execute(
                    f"""SELECT COUNT(*) AS count FROM entity_assignments assignment
                       JOIN entity_mentions mention ON mention.id=assignment.mention_id
                       JOIN documents document ON document.id=mention.document_id
                         {active_mention_filter}
                         AND document.is_deleted=FALSE
                       WHERE assignment.id=ANY(%s)""",
                    (assignment_ids,),
                ).fetchone()["count"])
                if current_count != len(assignment_ids):
                    raise ValueError("该 merge 事件属于历史内容修订，不能改写退休事实")
            conn.execute(
                "UPDATE entity_assignments SET active=FALSE WHERE event_id=%s AND active=TRUE",
                (event_id,),
            )
            if assignment_ids:
                conn.execute(
                    "UPDATE entity_assignments SET active=TRUE WHERE id=ANY(%s)",
                    (assignment_ids,),
                )
            conn.execute(
                """UPDATE canonical_entity_redirects SET active=FALSE
                   WHERE project_id=%s AND source_entity_id=%s AND event_id=%s""",
                (project_id, source_id, event_id),
            )
            conn.execute(
                "UPDATE canonical_entities SET is_deleted=FALSE,updated_at=NOW() WHERE id=%s",
                (source_id,),
            )
            document_ids = [
                int(value) for value in (event.get("before_state") or {}).get("source_document_ids", [])
            ]
            if document_ids:
                conn.execute(
                    f"""UPDATE canonical_entity_documents source
                       SET is_deleted=FALSE,updated_at=NOW()
                       FROM documents document
                       WHERE source.project_id=%s AND source.canonical_entity_id=%s
                         AND source.document_id=ANY(%s)
                         AND document.id=source.document_id
                         {active_projection_filter}
                         AND document.is_deleted=FALSE""",
                    (project_id, source_id, document_ids),
                )
            conn.execute(
                """INSERT INTO entity_resolution_events(
                   id,project_id,event_type,source_entity_id,target_entity_id,policy_version,
                   operator,reason,before_state,after_state,reversible)
                   VALUES(%s,%s,'undo',%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,FALSE)""",
                (
                    undo_id, project_id, source_id, event["target_entity_id"],
                    event["policy_version"], operator, reason,
                    self._json({"reversed_event_id": event_id}),
                    self._json({"restored_assignments": assignment_ids}),
                ),
            )
            conn.execute(
                "UPDATE entity_resolution_events SET reversed_by_event_id=%s WHERE id=%s",
                (undo_id, event_id),
            )
            project = self._mark_graph_stale_in_transaction(
                conn, project_id, f"实体合并事件 {event_id} 已回滚，等待重建",
            )
            affected_entity_ids = [source_id, int(event["target_entity_id"])]
            self._refresh_entity_document_projections(
                conn, project_id, affected_entity_ids,
            )
            self._refresh_assignment_aliases(
                conn, project_id, affected_entity_ids, str(event["policy_version"]),
            )
            policy_hash = str(project.get("active_policy_hash") or GRAPH_POLICY_HASH)
            self._refresh_relation_projections(
                conn, project_id, policy_hash, int(project["graph_generation"]),
            )
            self._refresh_entity_source_counts(conn, project_id, affected_entity_ids)
            return {
                "event_id": undo_id,
                "project": project,
                "reversed_event_id": event_id,
                "affected_entity_ids": affected_entity_ids,
            }

    def split_entity_mentions(
        self,
        project_id: int,
        source_entity_id: int,
        target_entity_id: int,
        mention_ids: list[int],
        *,
        reason: str,
        policy_version: str,
        operator: str = "manual",
    ) -> dict[str, Any]:
        """Reassign selected mentions while retaining immutable mention/assertion facts."""

        selected = sorted({int(value) for value in mention_ids})
        if not selected or source_entity_id == target_entity_id:
            raise ValueError("split 必须提供 mention 且目标实体不同")
        event_id = self._resolution_event_id(
            project_id, "split", f"{source_entity_id}:{target_entity_id}:{selected}",
        )
        with self.connection() as conn:
            revisioned = all((
                self._column_exists(conn, "documents", "active_revision_id"),
                self._column_exists(conn, "entity_mentions", "revision_id"),
            ))
            active_mention_filter = (
                "AND document.active_revision_id=mention.revision_id" if revisioned else ""
            )
            rows = conn.execute(
                f"""SELECT a.id,a.project_id,a.mention_id,a.canonical_entity_id,
                          a.assignment_generation,a.policy_version,a.event_id,a.reason,
                          a.active,a.metadata
                   FROM entity_assignments a
                   JOIN entity_mentions mention ON mention.id=a.mention_id
                   JOIN documents document ON document.id=mention.document_id
                     {active_mention_filter}
                     AND document.is_deleted=FALSE
                   WHERE a.project_id=%s AND a.mention_id=ANY(%s)
                     AND a.canonical_entity_id=%s AND a.active=TRUE
                   ORDER BY a.mention_id FOR UPDATE""",
                (project_id, selected, source_entity_id),
            ).fetchall()
            if len(rows) != len(selected):
                raise ValueError("部分 mention 不属于源实体或没有 active assignment")
            target = conn.execute(
                """SELECT id FROM canonical_entities
                   WHERE project_id=%s AND id=%s AND is_deleted=FALSE FOR UPDATE""",
                (project_id, target_entity_id),
            ).fetchone()
            if not target:
                raise LookupError("split 目标实体不存在")
            conn.execute(
                """INSERT INTO entity_resolution_events(
                   id,project_id,event_type,source_entity_id,target_entity_id,policy_version,
                   operator,reason,before_state,after_state,reversible)
                   VALUES(%s,%s,'split',%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,TRUE)""",
                (
                    event_id, project_id, source_entity_id, target_entity_id, policy_version,
                    operator, reason, self._json({"assignments": rows}),
                    self._json({"mention_ids": selected}),
                ),
            )
            old_ids = [int(item["id"]) for item in rows]
            conn.execute("UPDATE entity_assignments SET active=FALSE WHERE id=ANY(%s)", (old_ids,))
            conn.execute(
                """INSERT INTO entity_assignments(
                   project_id,mention_id,canonical_entity_id,assignment_generation,
                   policy_version,event_id,reason,active)
                   SELECT project_id,mention_id,%s,assignment_generation+1,%s,%s,%s,TRUE
                   FROM entity_assignments WHERE id=ANY(%s)""",
                (target_entity_id, policy_version, event_id, reason, old_ids),
            )
            project = self._mark_graph_stale_in_transaction(
                conn, project_id, f"实体 mention split {source_entity_id}->{target_entity_id}，等待重建",
            )
            affected_entity_ids = [source_entity_id, target_entity_id]
            self._refresh_entity_document_projections(
                conn, project_id, affected_entity_ids,
            )
            self._refresh_assignment_aliases(
                conn, project_id, affected_entity_ids, policy_version,
            )
            self._refresh_relation_projections(
                conn, project_id, str(project.get("active_policy_hash") or GRAPH_POLICY_HASH),
                int(project["graph_generation"]),
            )
            self._refresh_entity_source_counts(conn, project_id, affected_entity_ids)
            return {
                "event_id": event_id,
                "project": project,
                "mentions": len(selected),
                "affected_entity_ids": affected_entity_ids,
            }

    def create_entity_shadow_namespace(
        self,
        project_id: int,
        namespace_id: str,
        policy_set: dict[str, Any],
        document_order: list[int],
        snapshot_id: str | None = None,
        *,
        input_fingerprint: str = "",
        model_name: str = "",
        embedding_model: str = "",
        config_hash: str = "",
        code_version: str = "",
    ) -> dict[str, Any]:
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT * FROM entity_shadow_namespaces WHERE id=%s FOR UPDATE",
                (namespace_id,),
            ).fetchone()
            if existing:
                if (
                    input_fingerprint
                    and existing.get("input_fingerprint")
                    and existing["input_fingerprint"] != input_fingerprint
                ):
                    raise RuntimeError("entity shadow namespace 输入指纹冲突")
                return existing
            return conn.execute(
                """INSERT INTO entity_shadow_namespaces(
                   id,project_id,snapshot_id,policy_set,status,document_order,
                   input_fingerprint,model_name,embedding_model,config_hash,code_version)
                   VALUES(%s,%s,%s,%s::jsonb,'building',%s,%s,%s,%s,%s,%s)
                   RETURNING *""",
                (
                    namespace_id, project_id, snapshot_id, self._json(policy_set),
                    document_order, input_fingerprint, model_name, embedding_model,
                    config_hash, code_version,
                ),
            ).fetchone()

    def get_entity_shadow_namespace(self, namespace_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM entity_shadow_namespaces WHERE id=%s", (namespace_id,),
            ).fetchone()

    def seed_entity_shadow_work_items(
        self,
        namespace_id: str,
        items: list[dict[str, Any]],
        *,
        max_attempts: int = 3,
    ) -> int:
        with self.connection() as conn:
            for item in items:
                conn.execute(
                    """INSERT INTO entity_shadow_work_items(
                       id,namespace_id,revision_id,project_id,document_id,chunk_id,
                       input_fingerprint,status,max_attempts)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,'pending',%s)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        item["id"], namespace_id, item["revision_id"],
                        item["project_id"], item["document_id"], item["chunk_id"],
                        item["input_fingerprint"], max(1, min(20, max_attempts)),
                    ),
                )
            count = int(conn.execute(
                "SELECT COUNT(*) AS count FROM entity_shadow_work_items WHERE namespace_id=%s",
                (namespace_id,),
            ).fetchone()["count"])
            conn.execute(
                """UPDATE entity_shadow_namespaces SET expected_work_items=%s
                   WHERE id=%s""",
                (count, namespace_id),
            )
            return count

    def claim_entity_shadow_work(
        self,
        namespace_id: str,
        worker_id: str,
        *,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None:
        lease_token = uuid.uuid4().hex
        with self.connection() as conn:
            return conn.execute(
                """WITH candidate AS (
                       SELECT id FROM entity_shadow_work_items
                       WHERE namespace_id=%s AND attempt<max_attempts AND (
                         status IN ('pending','retryable_failed')
                         OR (status='leased' AND lease_expires_at<NOW())
                       )
                       ORDER BY document_id,chunk_id,id FOR UPDATE SKIP LOCKED LIMIT 1
                   )
                   UPDATE entity_shadow_work_items work SET
                     status='leased',attempt=work.attempt+1,lease_owner=%s,lease_token=%s,
                     lease_expires_at=NOW()+(%s*INTERVAL '1 second'),updated_at=NOW(),
                     error_message=''
                   FROM candidate WHERE work.id=candidate.id RETURNING work.*""",
                (namespace_id, worker_id, lease_token, max(30, int(lease_seconds))),
            ).fetchone()

    def complete_entity_shadow_work(
        self,
        work_id: str,
        lease_token: str,
        output: dict[str, Any],
        output_hash: str,
    ) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE entity_shadow_work_items SET status='completed',output_json=%s::jsonb,
                   output_hash=%s,lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
                   completed_at=NOW(),updated_at=NOW()
                   WHERE id=%s AND status='leased' AND lease_token=%s RETURNING id""",
                (self._json(output), output_hash, work_id, lease_token),
            ).fetchone()
            return bool(row)

    def fail_entity_shadow_work(
        self,
        work_id: str,
        lease_token: str,
        error: str,
    ) -> str:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE entity_shadow_work_items SET
                   status=CASE WHEN attempt>=max_attempts THEN 'terminal_failed'
                               ELSE 'retryable_failed' END,
                   error_message=%s,lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
                   updated_at=NOW()
                   WHERE id=%s AND status='leased' AND lease_token=%s
                   RETURNING status""",
                (error[:4000], work_id, lease_token),
            ).fetchone()
            return str((row or {}).get("status") or "stale")

    def persist_entity_shadow_extraction(
        self,
        namespace_id: str,
        *,
        project_id: int,
        document_id: int,
        revision_id: str,
        chunk_id: int,
        mentions: list[dict[str, Any]],
        type_assertions: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
    ) -> None:
        with self.connection() as conn:
            for item in mentions:
                conn.execute(
                    """INSERT INTO entity_shadow_mentions(
                       namespace_id,mention_key,project_id,document_id,revision_id,chunk_id,
                       surface,normalized_surface,evidence_text,evidence_start,evidence_end,
                       confidence,extraction_policy_version,filter_decision,filter_reason,provenance)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'accepted','',%s::jsonb)
                       ON CONFLICT(namespace_id,mention_key) DO NOTHING""",
                    (
                        namespace_id, item["mention_id"], project_id, document_id,
                        revision_id, chunk_id, item["surface"], item["normalized_surface"],
                        item.get("evidence") or item["surface"], int(item["start"]),
                        int(item["end"]), float(item["confidence"]),
                        item["extraction_policy_version"], self._json(item.get("provenance")),
                    ),
                )
            for item in type_assertions:
                conn.execute(
                    """INSERT INTO entity_shadow_type_assertions(
                       namespace_id,mention_key,entity_type,confidence,evidence_text,
                       policy_version,source,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT DO NOTHING""",
                    (
                        namespace_id, item["mention_id"], item["entity_type"],
                        float(item["confidence"]), item["evidence"],
                        item["policy_version"], item.get("source") or "model",
                        self._json(item.get("metadata")),
                    ),
                )
            for item in rejected:
                conn.execute(
                    """INSERT INTO entity_shadow_filter_decisions(
                       namespace_id,revision_id,document_id,chunk_id,decision,reason,
                       raw_entity,policy_version)
                       VALUES(%s,%s,%s,%s,'rejected',%s,%s::jsonb,%s)
                       ON CONFLICT DO NOTHING""",
                    (
                        namespace_id, revision_id, document_id, chunk_id,
                        item["reason"], self._json(item.get("raw") or {}),
                        str(item.get("policy_version") or "entity-extraction-p2-shadow-v1"),
                    ),
                )

    def list_entity_shadow_mentions(
        self, namespace_id: str, *, chunk_id: int | None = None,
    ) -> list[dict[str, Any]]:
        clause = "" if chunk_id is None else " AND mention.chunk_id=%s"
        params: tuple[Any, ...] = (namespace_id,) if chunk_id is None else (namespace_id, chunk_id)
        with self.connection() as conn:
            return conn.execute(
                """SELECT mention.*,assertion.entity_type,assertion.confidence AS type_confidence,
                   assertion.source AS type_source
                   FROM entity_shadow_mentions mention
                   LEFT JOIN LATERAL (
                     SELECT * FROM entity_shadow_type_assertions item
                     WHERE item.namespace_id=mention.namespace_id
                       AND item.mention_key=mention.mention_key
                     ORDER BY item.confidence DESC,item.entity_type LIMIT 1
                   ) assertion ON TRUE
                   WHERE mention.namespace_id=%s""" + clause
                + " ORDER BY mention.document_id,mention.chunk_id,mention.evidence_start,mention.mention_key",
                params,
            ).fetchall()

    def find_exact_entity_shadow_candidates(
        self, namespace_id: str, normalized_alias: str, limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT canonical.*,1.0 AS lexical_score
                   FROM entity_shadow_aliases alias
                   JOIN entity_shadow_canonicals canonical
                     ON canonical.namespace_id=alias.namespace_id
                    AND canonical.canonical_key=alias.canonical_key
                   WHERE alias.namespace_id=%s AND alias.normalized_alias=%s
                   ORDER BY canonical.canonical_key LIMIT %s""",
                (namespace_id, normalized_alias, max(1, min(10, limit))),
            ).fetchall()

    def find_entity_shadow_canonical(
        self, namespace_id: str, canonical_key: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM entity_shadow_canonicals
                   WHERE namespace_id=%s AND canonical_key=%s""",
                (namespace_id, canonical_key),
            ).fetchone()

    def search_entity_shadow_name_candidates(
        self, namespace_id: str, normalized_alias: str, limit: int,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT canonical.*,MAX(similarity(alias.normalized_alias,%s)) AS lexical_score
                   FROM entity_shadow_aliases alias
                   JOIN entity_shadow_canonicals canonical
                     ON canonical.namespace_id=alias.namespace_id
                    AND canonical.canonical_key=alias.canonical_key
                   WHERE alias.namespace_id=%s AND alias.normalized_alias %% %s
                   GROUP BY canonical.namespace_id,canonical.canonical_key
                   ORDER BY lexical_score DESC,canonical.canonical_key LIMIT %s""",
                (normalized_alias, namespace_id, normalized_alias, max(1, min(50, limit))),
            ).fetchall()

    def _ensure_entity_shadow_hnsw_index(
        self, conn: psycopg.Connection, model_name: str, dimension: int,
    ) -> None:
        if dimension <= 0 or dimension > Config.pgvector_hnsw_max_dimensions:
            raise RuntimeError(
                f"shadow embedding 维度 {dimension} 无可用 HNSW，按策略拒绝退化为全表扫描"
            )
        suffix = hashlib.sha256(f"shadow|{model_name}|{dimension}".encode()).hexdigest()[:12]
        statement = sql.SQL(
            """CREATE INDEX IF NOT EXISTS {} ON entity_shadow_embeddings
               USING hnsw ((embedding_vector::vector({})) vector_cosine_ops)
               WITH (m={},ef_construction={}) WHERE model_name={} AND dimension={}"""
        ).format(
            sql.Identifier(f"idx_entity_shadow_hnsw_{suffix}"),
            sql.Literal(dimension), sql.Literal(Config.pgvector_hnsw_m),
            sql.Literal(Config.pgvector_hnsw_ef_construction),
            sql.Literal(model_name), sql.Literal(dimension),
        )
        conn.execute(statement)

    def search_entity_shadow_vector_candidates(
        self,
        namespace_id: str,
        model_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        vector_literal = self._vector_literal(query_vector)
        dimension = len(query_vector)
        vector_type = sql.SQL("vector({})").format(sql.Literal(dimension))
        statement = sql.SQL(
            """SELECT canonical.*,
               GREATEST(0.0,1.0-(embedding.embedding_vector::{} <=> %s::{})) AS vector_score
               FROM entity_shadow_embeddings embedding
               JOIN entity_shadow_canonicals canonical
                 ON canonical.namespace_id=embedding.namespace_id
                AND canonical.canonical_key=embedding.canonical_key
               WHERE embedding.namespace_id=%s AND embedding.model_name=%s
                 AND embedding.dimension=%s
               ORDER BY embedding.embedding_vector::{} <=> %s::{} LIMIT %s"""
        ).format(vector_type, vector_type, vector_type, vector_type)
        with self.connection() as conn:
            self._ensure_entity_shadow_hnsw_index(conn, model_name, dimension)
            conn.execute("SELECT set_config('hnsw.iterative_scan','relaxed_order',TRUE)")
            return conn.execute(
                statement,
                (
                    vector_literal, namespace_id, model_name, dimension,
                    vector_literal, max(1, min(100, limit)),
                ),
            ).fetchall()

    def persist_entity_shadow_candidates(self, rows: list[dict[str, Any]]) -> None:
        with self.connection() as conn:
            for item in rows:
                scores = item.get("scores") or {}
                conn.execute(
                    """INSERT INTO entity_shadow_candidates(
                       namespace_id,mention_key,canonical_key,stage,rank,cap,
                       lexical_score,vector_score,reranker_score,combined_score,
                       filter_reason,decision,reason,policy_version,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(namespace_id,mention_key,canonical_key,stage) DO UPDATE SET
                       rank=EXCLUDED.rank,cap=EXCLUDED.cap,lexical_score=EXCLUDED.lexical_score,
                       vector_score=EXCLUDED.vector_score,reranker_score=EXCLUDED.reranker_score,
                       combined_score=EXCLUDED.combined_score,filter_reason=EXCLUDED.filter_reason,
                       decision=EXCLUDED.decision,reason=EXCLUDED.reason,metadata=EXCLUDED.metadata""",
                    (
                        item["namespace_id"], item["mention_key"], item["canonical_key"],
                        item["stage"], int(item["rank"]), int(item["cap"]),
                        float(scores.get("lexical") or 0), float(scores.get("vector") or 0),
                        float(scores.get("reranker") or 0), float(scores.get("combined") or 0),
                        item.get("filter_reason") or "", item.get("decision") or "",
                        item.get("reason") or "", item["policy_version"],
                        self._json(item.get("metadata")),
                    ),
                )

    def upsert_entity_shadow_canonical(
        self,
        namespace_id: str,
        canonical: dict[str, Any],
        mention: dict[str, Any],
        *,
        aliases: list[dict[str, Any]],
        embedding_model: str,
        embedding: list[float] | None,
        decision: str,
        reason: str,
        scores: dict[str, Any],
        config_hash: str,
    ) -> dict[str, Any]:
        canonical_key = str(canonical["canonical_key"])
        with self.connection() as conn:
            existing_assignment = conn.execute(
                """SELECT canonical_key FROM entity_shadow_assignments
                   WHERE namespace_id=%s AND mention_key=%s""",
                (namespace_id, mention["mention_id"]),
            ).fetchone()
            row = conn.execute(
                """INSERT INTO entity_shadow_canonicals(
                   namespace_id,canonical_key,name,normalized_name,primary_type,aliases,metadata)
                   VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT(namespace_id,canonical_key) DO UPDATE SET
                   aliases=(SELECT ARRAY(SELECT DISTINCT unnest(
                     entity_shadow_canonicals.aliases||EXCLUDED.aliases))),
                   metadata=entity_shadow_canonicals.metadata||EXCLUDED.metadata
                   RETURNING *""",
                (
                    namespace_id, canonical_key, canonical["name"], canonical["normalized_name"],
                    canonical.get("primary_type") or "GenericEntity",
                    canonical.get("aliases") or [], self._json(canonical.get("metadata")),
                ),
            ).fetchone()
            alias_rows = [{
                "surface": mention["surface"],
                "normalized_surface": mention["normalized_surface"],
                "start": mention["start"], "end": mention["end"],
                "confidence": mention.get("confidence") or 0.5,
                "kind": "mention",
            }, *aliases]
            for alias in alias_rows:
                if not alias.get("normalized_surface"):
                    continue
                conn.execute(
                    """INSERT INTO entity_shadow_aliases(
                       namespace_id,canonical_key,mention_key,alias,normalized_alias,
                       alias_kind,evidence_start,evidence_end,confidence,policy_version,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT DO NOTHING""",
                    (
                        namespace_id, canonical_key, mention["mention_id"], alias["surface"],
                        alias["normalized_surface"], alias.get("kind") or "observed",
                        alias.get("start"), alias.get("end"), float(alias.get("confidence") or 0.5),
                        mention["policy_version"], self._json(alias.get("metadata")),
                    ),
                )
            if embedding is not None:
                self._ensure_entity_shadow_hnsw_index(conn, embedding_model, len(embedding))
                vector_literal = self._vector_literal(embedding)
                content_hash = hashlib.sha256(
                    f"{canonical['name']}|{canonical.get('primary_type') or ''}".encode()
                ).hexdigest()
                conn.execute(
                    """INSERT INTO entity_shadow_embeddings(
                       namespace_id,canonical_key,model_name,embedding,embedding_vector,
                       dimension,content_hash,policy_version,metadata)
                       VALUES(%s,%s,%s,%s,%s::vector,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(namespace_id,canonical_key,model_name) DO UPDATE SET
                       embedding=EXCLUDED.embedding,embedding_vector=EXCLUDED.embedding_vector,
                       dimension=EXCLUDED.dimension,content_hash=EXCLUDED.content_hash,
                       metadata=EXCLUDED.metadata""",
                    (
                        namespace_id, canonical_key, embedding_model, embedding, vector_literal,
                        len(embedding), content_hash, mention["policy_version"],
                        self._json({"config_hash": config_hash}),
                    ),
                )
            conn.execute(
                """INSERT INTO entity_shadow_assignments(
                   namespace_id,mention_key,canonical_key,document_id,chunk_id,revision_id,
                   entity_type,policy_version,decision,reason,scores,metadata)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                   ON CONFLICT(namespace_id,mention_key) DO UPDATE SET
                   canonical_key=EXCLUDED.canonical_key,entity_type=EXCLUDED.entity_type,
                   policy_version=EXCLUDED.policy_version,decision=EXCLUDED.decision,
                   reason=EXCLUDED.reason,scores=EXCLUDED.scores,metadata=EXCLUDED.metadata""",
                (
                    namespace_id, mention["mention_id"], canonical_key,
                    int(mention["document_id"]), int(mention["chunk_id"]), mention["revision_id"],
                    mention.get("entity_type") or "GenericEntity", mention["policy_version"],
                    decision, reason, self._json(scores),
                    self._json({
                        "evidence_start": mention["start"], "evidence_end": mention["end"],
                        "type_source": mention.get("type_source") or "model",
                    }),
                ),
            )
            old_canonical_key = (
                str(existing_assignment["canonical_key"]) if existing_assignment else ""
            )
            if not old_canonical_key:
                row = conn.execute(
                    """UPDATE entity_shadow_canonicals SET source_count=source_count+1
                       WHERE namespace_id=%s AND canonical_key=%s RETURNING *""",
                    (namespace_id, canonical_key),
                ).fetchone()
            elif old_canonical_key != canonical_key:
                conn.execute(
                    """UPDATE entity_shadow_canonicals
                       SET source_count=GREATEST(0,source_count-1)
                       WHERE namespace_id=%s AND canonical_key=%s""",
                    (namespace_id, old_canonical_key),
                )
                row = conn.execute(
                    """UPDATE entity_shadow_canonicals SET source_count=source_count+1
                       WHERE namespace_id=%s AND canonical_key=%s RETURNING *""",
                    (namespace_id, canonical_key),
                ).fetchone()
            event_id = "shadow_event_" + hashlib.sha256(
                f"{namespace_id}|{mention['mention_id']}|{decision}|{canonical_key}".encode()
            ).hexdigest()
            conn.execute(
                """INSERT INTO entity_shadow_events(
                   id,namespace_id,event_type,mention_key,target_canonical_key,after_state,
                   reason,scores,model_name,config_hash,policy_version,provenance)
                   VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s,%s,%s,%s::jsonb)
                   ON CONFLICT(id) DO NOTHING""",
                (
                    event_id, namespace_id,
                    (
                        "new" if decision == "new_canonical"
                        else "uncertain" if decision.startswith("uncertain_")
                        else "assign"
                    ),
                    mention["mention_id"], canonical_key,
                    self._json({"canonical_key": canonical_key}), reason, self._json(scores),
                    mention.get("model_name") or "", config_hash, mention["policy_version"],
                    self._json({
                        "document_id": mention["document_id"], "revision_id": mention["revision_id"],
                        "chunk_id": mention["chunk_id"], "span": [mention["start"], mention["end"]],
                    }),
                ),
            )
            return row

    def list_entity_shadow_cannot_links(self, namespace_id: str) -> set[frozenset[str]]:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT left_key,right_key FROM entity_shadow_cannot_links
                   WHERE namespace_id=%s AND hard=TRUE""",
                (namespace_id,),
            ).fetchall()
            return {frozenset((str(item["left_key"]), str(item["right_key"]))) for item in rows}

    def list_entity_shadow_must_links(self, namespace_id: str) -> dict[str, str]:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT mention_key,canonical_key FROM entity_shadow_must_links
                   WHERE namespace_id=%s AND hard=TRUE
                   ORDER BY mention_key,canonical_key""",
                (namespace_id,),
            ).fetchall()
            result: dict[str, str] = {}
            for item in rows:
                mention_key = str(item["mention_key"])
                canonical_key = str(item["canonical_key"])
                if mention_key in result and result[mention_key] != canonical_key:
                    raise RuntimeError("同一 mention 存在冲突的 hard must-link")
                result[mention_key] = canonical_key
            return result

    def finalize_entity_shadow_namespace(
        self, namespace_id: str, *, metrics: dict[str, Any], artifact_hash: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            counts = conn.execute(
                """SELECT COUNT(*) AS expected,
                   COUNT(*) FILTER(WHERE status='completed') AS completed,
                   COUNT(*) FILTER(WHERE status='terminal_failed') AS failed,
                   COUNT(*) FILTER(WHERE status NOT IN ('completed','terminal_failed')) AS pending
                   FROM entity_shadow_work_items WHERE namespace_id=%s""",
                (namespace_id,),
            ).fetchone()
            status = "ready" if not int(counts["failed"]) and not int(counts["pending"]) else "failed"
            row = conn.execute(
                """UPDATE entity_shadow_namespaces SET status=%s,metrics=%s::jsonb,
                   artifact_hash=%s,expected_work_items=%s,completed_work_items=%s,
                   failed_work_items=%s,snapshot_complete=%s,error_message=%s,
                   completed_at=CASE WHEN %s THEN NOW() ELSE NULL END
                   WHERE id=%s RETURNING *""",
                (
                    status, self._json(metrics), artifact_hash, int(counts["expected"]),
                    int(counts["completed"]), int(counts["failed"]), status == "ready",
                    "" if status == "ready" else "entity shadow work 未全部成功",
                    status == "ready", namespace_id,
                ),
            ).fetchone()
            if status != "ready":
                raise RuntimeError("entity shadow snapshot 不完整，不能标记 ready")
            return row

    def persist_entity_shadow_relation_endpoints(
        self, namespace_id: str, rows: list[dict[str, Any]],
    ) -> None:
        with self.connection() as conn:
            for item in rows:
                conn.execute(
                    """INSERT INTO entity_shadow_relation_endpoints(
                       namespace_id,relation_key,document_id,revision_id,chunk_id,
                       source_mention_key,target_mention_key,source_canonical_key,
                       target_canonical_key,relation_type,evidence_text,evidence_start,
                       evidence_end,policy_version,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(namespace_id,relation_key) DO UPDATE SET
                       source_mention_key=EXCLUDED.source_mention_key,
                       target_mention_key=EXCLUDED.target_mention_key,
                       source_canonical_key=EXCLUDED.source_canonical_key,
                       target_canonical_key=EXCLUDED.target_canonical_key,
                       relation_type=EXCLUDED.relation_type,
                       evidence_text=EXCLUDED.evidence_text,
                       evidence_start=EXCLUDED.evidence_start,
                       evidence_end=EXCLUDED.evidence_end,
                       policy_version=EXCLUDED.policy_version,
                       metadata=EXCLUDED.metadata""",
                    (
                        namespace_id, item["relation_key"], item["document_id"],
                        item["revision_id"], item["chunk_id"], item["source_mention_key"],
                        item["target_mention_key"], item["source_canonical_key"],
                        item["target_canonical_key"], item["relation_type"],
                        item.get("evidence_text") or "", item.get("evidence_start"),
                        item.get("evidence_end"), item["policy_version"],
                        self._json(item.get("metadata")),
                    ),
                )

    def load_entity_shadow_snapshot(self, namespace_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            namespace = conn.execute(
                "SELECT * FROM entity_shadow_namespaces WHERE id=%s", (namespace_id,),
            ).fetchone()
            if not namespace:
                raise LookupError("entity shadow namespace 不存在")
            canonicals = conn.execute(
                """SELECT * FROM entity_shadow_canonicals WHERE namespace_id=%s
                   ORDER BY canonical_key""", (namespace_id,),
            ).fetchall()
            assignments = conn.execute(
                """SELECT * FROM entity_shadow_assignments WHERE namespace_id=%s
                   ORDER BY mention_key""", (namespace_id,),
            ).fetchall()
            endpoints = conn.execute(
                """SELECT * FROM entity_shadow_relation_endpoints WHERE namespace_id=%s
                   ORDER BY relation_key""", (namespace_id,),
            ).fetchall()
            return {
                "namespace": namespace,
                "canonicals": canonicals,
                "assignments": assignments,
                "relation_endpoints": endpoints,
            }

    def persist_entity_shadow_snapshot(
        self,
        namespace_id: str,
        canonicals: list[dict[str, Any]],
        assignments: list[dict[str, Any]],
        *,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a treatment namespace without touching any product entity table."""

        with self.connection() as conn:
            for item in canonicals:
                conn.execute(
                    """INSERT INTO entity_shadow_canonicals(
                       namespace_id,canonical_key,name,normalized_name,primary_type,aliases,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(namespace_id,canonical_key) DO UPDATE SET
                       name=EXCLUDED.name,normalized_name=EXCLUDED.normalized_name,
                       primary_type=EXCLUDED.primary_type,aliases=EXCLUDED.aliases,
                       metadata=EXCLUDED.metadata""",
                    (
                        namespace_id, item["id"], item["name"], item["normalized_name"],
                        item.get("entity_type") or "GenericEntity", item.get("aliases") or [],
                        self._json(item),
                    ),
                )
            for item in assignments:
                conn.execute(
                    """INSERT INTO entity_shadow_assignments(
                       namespace_id,mention_key,canonical_key,document_id,chunk_id,
                       entity_type,policy_version,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(namespace_id,mention_key) DO UPDATE SET
                       canonical_key=EXCLUDED.canonical_key,entity_type=EXCLUDED.entity_type,
                       policy_version=EXCLUDED.policy_version,metadata=EXCLUDED.metadata""",
                    (
                        namespace_id, item["mention_id"], item["canonical_id"],
                        int(item.get("document_id") or 0), int(item.get("chunk_id") or 0),
                        item.get("entity_type") or "GenericEntity", item["policy_version"],
                        self._json(item),
                    ),
                )
            row = conn.execute(
                """UPDATE entity_shadow_namespaces SET status='ready',metrics=%s::jsonb,
                   completed_at=NOW() WHERE id=%s RETURNING *""",
                (self._json(metrics), namespace_id),
            ).fetchone()
            if not row:
                raise LookupError("entity shadow namespace 不存在")
            return row

    def list_relation_verification_events(
        self, project_id: int, document_id: int | None = None,
    ) -> list[dict[str, Any]]:
        document_clause = "" if document_id is None else " AND event.document_id=%s"
        params: tuple[Any, ...] = (project_id,) if document_id is None else (project_id, document_id)
        with self.connection() as conn:
            return conn.execute(
                """SELECT event.*,
                   CASE WHEN event.batch_id IS NULL THEN event.request_json ELSE batch.request_json END
                       AS effective_request_json,
                   CASE WHEN event.batch_id IS NULL THEN event.raw_response ELSE batch.raw_response END
                       AS effective_raw_response
                   FROM relation_verification_events event
                   LEFT JOIN relation_verification_batches batch ON batch.id=event.batch_id
                   WHERE event.project_id=%s""" + document_clause + " ORDER BY event.id",
                params,
            ).fetchall()

    def list_project_relations(self, project_id: int) -> list[dict[str, Any]]:
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            return conn.execute(
                """SELECT r.*,
                COALESCE(projection.source_entity_id,r.source_entity_id) AS source_entity_id,
                COALESCE(projection.target_entity_id,r.target_entity_id) AS target_entity_id,
                s.name AS source_name,s.entity_type AS source_type,t.name AS target_name,t.entity_type AS target_type,
                d.original_filename,c.content AS chunk_content FROM relation_assertions r
                LEFT JOIN canonical_relation_projections projection
                  ON projection.project_id=r.project_id AND projection.assertion_id=r.id
                JOIN canonical_entities s ON s.id=COALESCE(projection.source_entity_id,r.source_entity_id)
                JOIN canonical_entities t ON t.id=COALESCE(projection.target_entity_id,r.target_entity_id)
                JOIN documents d ON d.id=r.document_id AND d.active_revision_id=r.revision_id
                JOIN chunks c ON c.id=r.chunk_id
                WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                AND COALESCE(projection.publishable,TRUE)=TRUE
                AND """ + clause + " ORDER BY r.id", (project_id, *policy_params)
            ).fetchall()

    def export_graphrag_control_snapshot(self, project_id: int):
        """Export one consistent fail-closed snapshot of the formal control graph."""

        from .graphrag_community import ControlGraphSnapshot

        clause, policy_params = self._publishable_relation_clause("r")
        with self.connection() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            project = conn.execute(
                "SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            if not is_current_graph(project):
                raise RuntimeError("项目正式图不是 current/ready 状态")
            relations = conn.execute(
                """SELECT
                   r.id AS assertion_id,r.document_id,r.chunk_id,r.revision_id,
                   r.relation_type,r.description AS relation_description,
                   COALESCE(projection.source_entity_id,r.source_entity_id)
                       AS source_entity_id,
                   COALESCE(projection.target_entity_id,r.target_entity_id)
                       AS target_entity_id,
                   source.entity_key AS source_entity_key,
                   source.name AS source_name,source.entity_type AS source_type,
                   source.description AS source_description,
                   target.entity_key AS target_entity_key,
                   target.name AS target_name,target.entity_type AS target_type,
                   target.description AS target_description,
                   c.chunk_key
                   FROM relation_assertions r
                   LEFT JOIN canonical_relation_projections projection
                     ON projection.project_id=r.project_id
                    AND projection.assertion_id=r.id
                   JOIN canonical_entities source
                     ON source.id=COALESCE(projection.source_entity_id,r.source_entity_id)
                    AND source.project_id=r.project_id AND source.is_deleted=FALSE
                   JOIN canonical_entities target
                     ON target.id=COALESCE(projection.target_entity_id,r.target_entity_id)
                    AND target.project_id=r.project_id AND target.is_deleted=FALSE
                   JOIN documents d
                     ON d.id=r.document_id AND d.active_revision_id=r.revision_id
                   JOIN chunks c
                     ON c.id=r.chunk_id AND c.revision_id=r.revision_id
                    AND c.is_deleted=FALSE
                   WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                     AND COALESCE(projection.publishable,TRUE)=TRUE
                     AND """ + clause + " ORDER BY r.id",
                (project_id, *policy_params),
            ).fetchall()
            return ControlGraphSnapshot(
                project_id=int(project["id"]),
                graph_generation=int(project["graph_generation"]),
                graph_policy_hash=str(project.get("graph_policy_hash") or ""),
                active_policy_hash=str(project.get("active_policy_hash") or ""),
                relations=tuple(dict(item) for item in relations),
            )

    def create_graphrag_community_run(
        self,
        *,
        run_id: str,
        snapshot,
        input_hash: str,
        config: dict[str, Any],
        metrics: dict[str, int],
        graphrag_version: str,
        adapter_version: str,
        weighting_strategy: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO graphrag_community_runs(
                   run_id,project_id,graph_generation,graph_policy_hash,
                   active_policy_hash,input_hash,graphrag_version,adapter_version,
                   weighting_strategy,seed,max_cluster_size,use_lcc,status,
                   node_count,aggregated_edge_count,excluded_self_loops,
                   excluded_isolated_entities,config)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'running',
                          %s,%s,%s,%s,%s::jsonb) RETURNING *""",
                (
                    run_id, snapshot.project_id, snapshot.graph_generation,
                    snapshot.graph_policy_hash, snapshot.active_policy_hash,
                    input_hash, graphrag_version, adapter_version, weighting_strategy,
                    config["seed"], config["max_cluster_size"], config["use_lcc"],
                    metrics["node_count"], metrics["aggregated_edge_count"],
                    metrics["excluded_self_loops"],
                    metrics["excluded_isolated_entities"], self._json(config),
                ),
            ).fetchone()

    def supersede_graphrag_community_run(
        self, run_id: str, reason: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE graphrag_community_runs
                   SET status='superseded',error_message=%s,completed_at=NOW()
                   WHERE run_id=%s AND status='running' RETURNING *""",
                (reason[-4000:], run_id),
            ).fetchone()
            if not row:
                raise LookupError("GraphRAG community run 不存在或已终止")
            return row

    def fail_graphrag_community_run(self, run_id: str, error: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE graphrag_community_runs
                   SET status='failed',error_message=%s,completed_at=NOW()
                   WHERE run_id=%s AND status='running' RETURNING *""",
                (error[-4000:], run_id),
            ).fetchone()
            return row or {}

    def finalize_graphrag_community_run(
        self,
        *,
        run_id: str,
        snapshot,
        input_hash: str,
        status: str,
        communities: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if status not in {"completed", "completed_empty"}:
            raise ValueError("非法 GraphRAG community 完成状态")
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM graphrag_community_runs WHERE run_id=%s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if not run:
                raise RuntimeError("GraphRAG community run 不存在")
            if run["status"] in {"completed", "completed_empty"}:
                same_completion = bool(
                    run["status"] == status
                    and run["input_hash"] == input_hash
                    and int(run["graph_generation"]) == int(snapshot.graph_generation)
                    and run["graph_policy_hash"] == snapshot.graph_policy_hash
                    and run["active_policy_hash"] == snapshot.active_policy_hash
                )
                if same_completion:
                    return run
                raise RuntimeError("GraphRAG community run 已完成且重试参数不一致")
            if run["status"] != "running":
                raise RuntimeError("GraphRAG community run 不是 running")
            project = conn.execute(
                "SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE FOR SHARE",
                (run["project_id"],),
            ).fetchone()
            unchanged = bool(
                project
                and is_current_graph(project)
                and int(project["graph_generation"]) == int(snapshot.graph_generation)
                and str(project.get("graph_policy_hash") or "") == snapshot.graph_policy_hash
                and str(project.get("active_policy_hash") or "") == snapshot.active_policy_hash
                and run["input_hash"] == input_hash
            )
            if not unchanged:
                return conn.execute(
                    """UPDATE graphrag_community_runs
                       SET status='superseded',error_message=%s,completed_at=NOW()
                       WHERE run_id=%s RETURNING *""",
                    ("正式图 generation/policy/input_hash 已变化", run_id),
                ).fetchone()

            for community in communities:
                community_metadata = {
                    "relationship_ids": community["relationship_ids"],
                    "text_unit_ids": community["text_unit_ids"],
                    "children": community["children"],
                }
                conn.execute(
                    """INSERT INTO graphrag_communities(
                       run_id,project_id,stable_community_id,graphrag_output_id,
                       graphrag_community,level,parent_stable_community_id,title,size,
                       period,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(run_id,stable_community_id) DO NOTHING""",
                    (
                        run_id, run["project_id"], community["stable_community_id"],
                        community["graphrag_output_id"],
                        community["graphrag_community"], community["level"],
                        community["parent_stable_community_id"], community["title"],
                        community["size"], community["period"],
                        self._json(community_metadata),
                    ),
                )
                for entity_id in community["canonical_entity_ids"]:
                    conn.execute(
                        """INSERT INTO graphrag_community_members(
                           run_id,project_id,stable_community_id,canonical_entity_id)
                           VALUES(%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (
                            run_id, run["project_id"],
                            community["stable_community_id"], entity_id,
                        ),
                    )
            for edge in edges:
                conn.execute(
                    """INSERT INTO graphrag_community_edges(
                       run_id,project_id,stable_edge_id,source_entity_id,target_entity_id,
                       weight,assertion_ids,chunk_ids,relation_types,text_unit_ids,
                       provenance_hash,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(run_id,stable_edge_id) DO NOTHING""",
                    (
                        run_id, run["project_id"], edge["stable_edge_id"],
                        edge["source_entity_id"], edge["target_entity_id"],
                        edge["weight"], edge["assertion_ids"], edge["chunk_ids"],
                        edge["relation_types"], edge["text_unit_ids"],
                        edge["provenance_hash"], self._json({}),
                    ),
                )
            hierarchy_levels = len({int(item["level"]) for item in communities})
            return conn.execute(
                """UPDATE graphrag_community_runs SET status=%s,
                   community_count=%s,hierarchy_levels=%s,metadata=%s::jsonb,
                   error_message='',completed_at=NOW()
                   WHERE run_id=%s RETURNING *""",
                (status, len(communities), hierarchy_levels, self._json(metadata), run_id),
            ).fetchone()

    def get_graphrag_community_run(self, run_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM graphrag_community_runs WHERE run_id=%s", (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("GraphRAG community run 不存在")
            project = conn.execute(
                "SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE",
                (run["project_id"],),
            ).fetchone()
            result = dict(run)
            result["current"] = bool(
                run["status"] in {"completed", "completed_empty"}
                and project
                and is_current_graph(project)
                and int(project["graph_generation"]) == int(run["graph_generation"])
                and str(project.get("graph_policy_hash") or "") == run["graph_policy_hash"]
                and str(project.get("active_policy_hash") or "") == run["active_policy_hash"]
            )
            return result

    def show_graphrag_community_run(self, run_id: str) -> dict[str, Any]:
        run = self.get_graphrag_community_run(run_id)
        with self.connection() as conn:
            communities = conn.execute(
                """SELECT * FROM graphrag_communities
                   WHERE run_id=%s ORDER BY level,graphrag_community,stable_community_id""",
                (run_id,),
            ).fetchall()
            members = conn.execute(
                """SELECT member.*,entity.entity_key,entity.name,entity.entity_type
                   FROM graphrag_community_members member
                   JOIN canonical_entities entity ON entity.id=member.canonical_entity_id
                   WHERE member.run_id=%s
                   ORDER BY member.stable_community_id,member.canonical_entity_id""",
                (run_id,),
            ).fetchall()
            edges = conn.execute(
                """SELECT * FROM graphrag_community_edges
                   WHERE run_id=%s ORDER BY stable_edge_id""",
                (run_id,),
            ).fetchall()
        return {
            "run": run,
            "current": run["current"],
            "communities": communities,
            "members": members,
            "edges": edges,
        }

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
                   JOIN documents d ON d.id=r.document_id AND d.active_revision_id=r.revision_id
                   LEFT JOIN canonical_relation_projections projection
                     ON projection.project_id=r.project_id AND projection.assertion_id=r.id
                   WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                   AND COALESCE(projection.publishable,TRUE)=TRUE
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
                WHERE r.project_id=%s AND r.is_deleted=FALSE
                ORDER BY r.id""",
                (project_id,),
            ).fetchall()

    def list_document_mentions(self, document_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT m.*,e.name,e.entity_type,e.graph_vertex_id FROM entity_mentions m
                JOIN canonical_entities e ON e.id=m.canonical_entity_id
                JOIN documents d ON d.id=m.document_id AND d.active_revision_id=m.revision_id
                WHERE m.document_id=%s AND m.is_deleted=FALSE AND d.is_deleted=FALSE""", (document_id,)
            ).fetchall()

    def list_document_relations(self, document_id: int) -> list[dict[str, Any]]:
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            return conn.execute(
                """SELECT r.*,
                COALESCE(projection.source_entity_id,r.source_entity_id) AS source_entity_id,
                COALESCE(projection.target_entity_id,r.target_entity_id) AS target_entity_id,
                s.graph_vertex_id AS source_vertex_id,t.graph_vertex_id AS target_vertex_id
                FROM relation_assertions r
                LEFT JOIN canonical_relation_projections projection
                  ON projection.project_id=r.project_id AND projection.assertion_id=r.id
                JOIN canonical_entities s ON s.id=COALESCE(projection.source_entity_id,r.source_entity_id)
                JOIN canonical_entities t ON t.id=COALESCE(projection.target_entity_id,r.target_entity_id)
                JOIN documents d ON d.id=r.document_id AND d.active_revision_id=r.revision_id
                WHERE r.document_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                AND COALESCE(projection.publishable,TRUE)=TRUE AND """ + clause,
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
    def _refresh_entity_source_counts(
        conn: psycopg.Connection,
        project_id: int,
        entity_ids: list[int] | None = None,
    ) -> None:
        ids = sorted({int(value) for value in (entity_ids or [])})
        id_filter = " AND e.id=ANY(%s)" if ids else ""
        revisioned = all((
            Database._column_exists(conn, "documents", "active_revision_id"),
            Database._column_exists(conn, "canonical_entity_documents", "revision_id"),
        ))
        active_projection_filter = (
            "AND document.active_revision_id=source.revision_id" if revisioned else ""
        )
        params: list[Any] = [project_id]
        if ids:
            params.append(ids)
        conn.execute(
            f"""UPDATE canonical_entities e
               SET source_count=(
                       SELECT COUNT(*) FROM canonical_entity_documents source
                       JOIN documents document ON document.id=source.document_id
                         {active_projection_filter}
                       WHERE source.project_id=e.project_id AND document.is_deleted=FALSE
                         AND source.canonical_entity_id=e.id AND source.is_deleted=FALSE
                   ),
                   is_deleted=NOT EXISTS(
                       SELECT 1 FROM canonical_entity_documents source
                       JOIN documents document ON document.id=source.document_id
                         {active_projection_filter}
                       WHERE source.project_id=e.project_id AND document.is_deleted=FALSE
                         AND source.canonical_entity_id=e.id AND source.is_deleted=FALSE
                   ),
                   updated_at=NOW()
               WHERE e.project_id=%s""" + id_filter,
            params,
        )

    @classmethod
    def _retire_orphan_entities(
        cls,
        conn: psycopg.Connection,
        project_id: int,
        entity_ids: list[int] | None = None,
    ) -> None:
        cls._refresh_entity_source_counts(conn, project_id, entity_ids)
        ids = sorted({int(value) for value in (entity_ids or [])})
        id_filter = " AND e.id=ANY(%s)" if ids else ""
        params: list[Any] = [project_id]
        if ids:
            params.append(ids)
        conn.execute(
            """UPDATE vector_embeddings v SET is_deleted=e.is_deleted,updated_at=NOW()
            FROM canonical_entities e
            WHERE e.project_id=%s""" + id_filter + """
              AND v.project_id=e.project_id AND v.object_type='entity'
              AND v.object_key=e.id::text""",
            params,
        )

    def retire_orphan_entities(
        self, project_id: int, entity_ids: list[int] | None = None,
    ) -> None:
        with self.connection() as conn:
            self._retire_orphan_entities(conn, project_id, entity_ids)

    def statistics(self, project_id: int) -> dict[str, int]:
        clause, policy_params = self._publishable_relation_clause("r")
        statement = f"""SELECT
            (SELECT COUNT(*) FROM documents WHERE project_id=%s AND is_deleted=FALSE) documents,
            (SELECT COUNT(*) FROM chunks c JOIN documents d
                ON d.id=c.document_id AND d.active_revision_id=c.revision_id
                WHERE c.project_id=%s AND c.is_deleted=FALSE AND d.is_deleted=FALSE) chunks,
            (SELECT COUNT(*) FROM canonical_entities WHERE project_id=%s AND is_deleted=FALSE) entities,
            (SELECT COUNT(*) FROM relation_assertions r JOIN documents d
                ON d.id=r.document_id AND d.active_revision_id=r.revision_id
                WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                AND {clause}) relations,
            (SELECT COUNT(*) FROM relation_candidates rc JOIN documents d
                ON d.id=rc.document_id AND d.active_revision_id=rc.revision_id
                WHERE rc.project_id=%s AND rc.is_deleted=FALSE AND d.is_deleted=FALSE) relation_candidates"""
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
                LEFT JOIN canonical_relation_projections projection
                  ON projection.project_id=r.project_id AND projection.assertion_id=r.id
                JOIN canonical_entities s ON s.id=COALESCE(projection.source_entity_id,r.source_entity_id)
                JOIN canonical_entities t ON t.id=COALESCE(projection.target_entity_id,r.target_entity_id)
                JOIN documents d ON d.id=r.document_id AND d.active_revision_id=r.revision_id
                JOIN chunks c ON c.id=r.chunk_id
                WHERE r.project_id=%s AND r.is_deleted=FALSE AND d.is_deleted=FALSE
                AND COALESCE(projection.publishable,TRUE)=TRUE
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
                """SELECT * FROM canonical_entities
                WHERE project_id=%s AND id=ANY(%s) AND is_deleted=FALSE""", (project_id,ids)
            ).fetchall()

    def list_project_relations_for_entities(
        self, project_id: int, entity_ids: list[int],
    ) -> list[dict[str, Any]]:
        """List publishable projected relations touching a bounded entity set."""

        ids = sorted({int(value) for value in entity_ids})
        if not ids:
            return []
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            return conn.execute(
                """SELECT r.*,
                   COALESCE(projection.source_entity_id,r.source_entity_id) AS source_entity_id,
                   COALESCE(projection.target_entity_id,r.target_entity_id) AS target_entity_id,
                   source.name AS source_name,target.name AS target_name
                   FROM relation_assertions r
                   LEFT JOIN canonical_relation_projections projection
                     ON projection.project_id=r.project_id AND projection.assertion_id=r.id
                   JOIN canonical_entities source
                     ON source.id=COALESCE(projection.source_entity_id,r.source_entity_id)
                   JOIN canonical_entities target
                     ON target.id=COALESCE(projection.target_entity_id,r.target_entity_id)
                   JOIN documents document ON document.id=r.document_id
                     AND document.active_revision_id=r.revision_id
                   WHERE r.project_id=%s AND r.is_deleted=FALSE AND document.is_deleted=FALSE
                     AND COALESCE(projection.publishable,TRUE)=TRUE
                     AND (COALESCE(projection.source_entity_id,r.source_entity_id)=ANY(%s)
                       OR COALESCE(projection.target_entity_id,r.target_entity_id)=ANY(%s))
                     AND """ + clause + " ORDER BY r.id",
                (project_id, ids, ids, *policy_params),
            ).fetchall()

    def retire_derived_embeddings_for_entity_change(
        self, project_id: int, entity_ids: list[int], model_name: str,
    ) -> None:
        """Retire only entity/relation vectors invalidated by a canonical assignment change."""

        ids = sorted({int(value) for value in entity_ids})
        if not ids:
            return
        keys = [str(value) for value in ids]
        with self.connection() as conn:
            conn.execute(
                """UPDATE vector_embeddings SET is_deleted=TRUE,updated_at=NOW()
                   WHERE project_id=%s AND object_type='entity' AND model_name=%s
                     AND object_key=ANY(%s)""",
                (project_id, model_name, keys),
            )
            conn.execute(
                """UPDATE vector_embeddings vector SET is_deleted=TRUE,updated_at=NOW()
                   FROM relation_assertions relation
                   LEFT JOIN canonical_relation_projections projection
                     ON projection.project_id=relation.project_id
                    AND projection.assertion_id=relation.id
                   WHERE vector.project_id=%s AND vector.object_type='relation'
                     AND vector.model_name=%s AND vector.object_key=relation.id::TEXT
                     AND relation.project_id=%s
                     AND (COALESCE(projection.source_entity_id,relation.source_entity_id)=ANY(%s)
                       OR COALESCE(projection.target_entity_id,relation.target_entity_id)=ANY(%s))""",
                (project_id, model_name, project_id, ids, ids),
            )

    def get_relations_by_ids(self, project_id: int, ids: list[int]) -> list[dict[str, Any]]:
        if not ids: return []
        clause, policy_params = self._publishable_relation_clause()
        with self.connection() as conn:
            return conn.execute(
                """SELECT r.id,s.name AS source_name,r.relation_type,t.name AS target_name,r.description,r.confidence,
                r.verification_status,d.original_filename,d.storage_path,c.chunk_key FROM relation_assertions r
                LEFT JOIN canonical_relation_projections projection
                  ON projection.project_id=r.project_id AND projection.assertion_id=r.id
                JOIN canonical_entities s ON s.id=COALESCE(projection.source_entity_id,r.source_entity_id)
                JOIN canonical_entities t ON t.id=COALESCE(projection.target_entity_id,r.target_entity_id)
                JOIN documents d ON d.id=r.document_id JOIN chunks c ON c.id=r.chunk_id
                WHERE r.project_id=%s AND r.id=ANY(%s) AND r.is_deleted=FALSE
                AND d.is_deleted=FALSE AND COALESCE(projection.publishable,TRUE)=TRUE AND """ + clause,
                (project_id,ids,*policy_params)
            ).fetchall()
