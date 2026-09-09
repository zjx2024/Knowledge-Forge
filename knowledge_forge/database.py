"""PostgreSQL persistence for projects, retrieval, graph provenance and jobs."""

from __future__ import annotations

import json
import hashlib
import logging
import math
import re
import uuid
import threading
from collections import Counter, OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Mapping

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .config import Config
from .construction_strategies import (
    KNOWLEDGE_FORGE_STRATEGY,
    GRAPHRAG_NATIVE_STRATEGY,
    STABLE_STRATEGY,
    normalize_strategy,
)
from .knowledge_language import CHINESE, normalize_knowledge_language
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
from .unicode_safety import sanitize_unicode_data


LOGGER = logging.getLogger(__name__)


class ContentRevisionError(RuntimeError):
    """Base error for immutable document content revisions."""


class RevisionManifestMismatch(ContentRevisionError):
    """The same ingestion run was retried with different immutable inputs."""


class StaleContentRevision(ContentRevisionError):
    """A late worker attempted to mutate or activate an obsolete revision."""


class RevisionLeaseLost(StaleContentRevision):
    """The worker no longer owns the revision lease."""


class JobCancellationRequested(RuntimeError):
    """A publication CAS observed a durable user cancellation request."""


class JobPublicationCommitted(RuntimeError):
    """A cancellation arrived after the task's atomic publication boundary."""


GROUNDED_RELATION_V2_REQUIRED_MIGRATIONS = (
    "0014_shadow_relation_v2_ledger",
    "0015_shadow_relation_v2_verification",
    "0016_shadow_relation_v2_normalization",
    "0017_shadow_relation_v2_freshness",
    "0018_shadow_relation_v2_gold",
    "0019_shadow_relation_v2_execution",
    "0020_shadow_relation_v2_model_audit",
    "0021_shadow_relation_v2_artifact_audit",
    "0027_shadow_relation_secondary_review",
    "0028_shadow_relation_secondary_idempotency",
    "0029_shadow_relation_secondary_idempotency_cleanup",
)

KNOWLEDGE_FORGE_NAMING_REQUIRED_MIGRATIONS = (
    "0030_knowledge_forge_strategy_naming",
)

GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS = (
    "0022_graphrag_native_control_plane",
)

GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS = {
    "graphrag_native_runs": {
        "run_id", "project_id", "generation", "state", "base_sha",
        "worktree_diff_hash", "code_tree_hash", "source_snapshot_hash",
        "settings_hash", "prompt_profile_hash", "completion_model_id",
        "completion_model_digest", "embedding_model_id", "embedding_model_digest",
        "embedding_dimension", "graphrag_version", "python_version",
        "staging_relative_path", "bundle_relative_path", "validator_version",
        "validation_report_hash", "lease_owner", "lease_expires_at",
    },
    "graphrag_native_run_documents": {
        "run_id", "project_id", "document_id", "revision_id", "source_hash",
        "structure_hash", "input_document_count", "input_manifest_hash", "ordinal",
    },
    "graphrag_native_artifacts": {
        "run_id", "logical_name", "relative_path", "sha256", "byte_size",
        "media_type", "schema_hash", "row_count",
    },
    "graphrag_native_project_state": {
        "project_id", "active_run_id", "active_generation",
        "desired_source_snapshot_hash", "updated_at",
    },
    "graphrag_native_query_runs": {
        "query_run_id", "project_id", "run_id", "query_mode", "question_hash",
        "status", "model_id", "latency_ms", "token_usage", "context_reference_count",
        "failure_class",
    },
}

GROUNDED_RELATION_V2_REQUIRED_TABLE_COLUMNS = {
    "shadow_relation_v2_runs": {
        "id", "project_id", "input_fingerprint", "fingerprint_components",
        "source_generation", "status", "pipeline_generation", "audit_quality",
    },
    "shadow_relation_pair_opportunities": {
        "id", "run_id", "pair_identity", "left_occurrence_key",
        "right_occurrence_key", "status", "pipeline_generation", "context_id",
        "source_kind", "source_record_id",
    },
    "shadow_relation_classification_attempts": {
        "id", "pair_opportunity_id", "attempt_no", "status",
        "model_name", "request_payload", "raw_response_text",
        "parsed_response", "token_usage", "error_type", "independent_call",
        "model_call_id", "audit_quality", "terminal_status",
    },
    "shadow_relation_hypotheses": {
        "id", "pair_opportunity_id", "hypothesis_key", "predicate", "status",
        "qualifiers", "source_kind", "source_record_id",
    },
    "shadow_relation_hypothesis_validations": {
        "hypothesis_id", "status", "reason_code",
    },
    "shadow_relation_verification_attempts": {
        "id", "hypothesis_id", "attempt_no", "verdict", "independent_call",
        "model_call_id", "audit_quality", "terminal_status",
    },
    "shadow_relation_normalized_facts": {
        "id", "project_id", "semantic_fact_key", "predicate",
        "qualifiers", "audit_quality",
    },
    "shadow_relation_fact_memberships": {
        "run_id", "fact_id", "input_fingerprint", "status",
    },
    "shadow_relation_fact_evidence": {
        "id", "fact_id", "run_id", "hypothesis_id",
        "verification_attempt_id", "evidence_spans", "context_id", "source_kind",
        "source_record_id", "qualifiers", "audit_quality",
    },
    "shadow_relation_current_fingerprints": {
        "project_id", "current_fingerprint", "components", "source_generation",
    },
    "shadow_source_contexts": {
        "id", "run_id", "project_id", "document_revision_id", "chunk_id",
        "context_kind", "context_identity", "content_hash",
        "structure_artifact_hash", "parser_version", "payload",
    },
    "shadow_context_source_maps": {
        "id", "context_id", "source_kind", "source_record_id",
        "source_start", "source_end", "payload_hash", "payload",
    },
    "entity_shadow_occurrences": {
        "id", "namespace_id", "project_id", "document_revision_id",
        "chunk_id", "occurrence_key", "physical_occurrence_key",
        "surface", "normalized_surface", "source_context_id",
    },
    "entity_shadow_mention_occurrence_map": {
        "namespace_id", "mention_key", "occurrence_id", "mapping_policy_version",
    },
    "entity_shadow_occurrence_assignments": {
        "occurrence_id", "namespace_id", "canonical_key", "assignment_version",
        "status", "confidence",
    },
    "entity_shadow_canonical_redirects": {
        "id", "namespace_id", "source_canonical_key", "target_canonical_key",
        "redirect_policy_version", "reason_code",
    },
    "shadow_model_calls": {
        "id", "run_id", "component", "model_name", "model_digest",
        "prompt_version", "schema_version", "parser_version", "idempotency_key",
        "attempt_no", "status", "request_hash", "response_hash",
        "request_payload", "raw_response_text", "parsed_response",
        "token_usage", "independent_call", "audit_quality", "terminal_status",
    },
    "shadow_model_call_members": {
        "id", "model_call_id", "run_id", "source_kind", "source_record_id",
        "pair_opportunity_id", "hypothesis_id",
    },
    "shadow_relation_pair_contexts": {
        "pair_opportunity_id", "context_id", "context_role", "payload",
    },
    "shadow_relation_candidate_sources": {
        "id", "run_id", "pair_opportunity_id", "candidate_source",
        "source_kind", "source_record_id", "candidate_rank", "admitted",
        "terminal_status",
    },
    "shadow_relation_proposal_attempts": {
        "id", "run_id", "context_id", "model_call_id", "attempt_no", "status",
        "proposer_prompt_version", "proposer_schema_version",
        "proposer_parser_version", "proposer_model_revision",
        "request_hash", "response_hash", "token_usage",
    },
    "shadow_relation_proposals": {
        "id", "proposal_attempt_id", "pair_opportunity_id", "hypothesis_id",
        "proposal_key", "predicate", "subject_occurrence_key",
        "object_occurrence_key", "evidence_spans", "qualifiers",
        "confidence", "status",
    },
    "shadow_relation_evidence_refs": {
        "id", "run_id", "fact_evidence_id", "verification_attempt_id",
        "context_id", "source_kind", "source_record_id", "evidence_hash",
        "qualifiers",
    },
    "shadow_relation_secondary_passes": {
        "id", "parent_run_id", "project_id", "pass_version",
        "input_fingerprint", "status", "route_counts", "metrics",
    },
    "shadow_relation_secondary_candidates": {
        "id", "secondary_pass_id", "pair_opportunity_id", "route",
        "source_reason_code", "predicate_after", "status", "reason_code",
    },
}


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
    knowledge_language TEXT NOT NULL DEFAULT 'zh'
        CHECK (knowledge_language IN ('auto','zh','en')),
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
ALTER TABLE projects ADD COLUMN IF NOT EXISTS llm_provider TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS embedding_provider TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS ollama_base_url TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS ollama_embedding_base_url TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS llm_api_base_url TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS llm_api_key TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS llm_api_models TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS llm_api_context_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS embedding_api_base_url TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS embedding_api_key TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS embedding_api_models TEXT NOT NULL DEFAULT '';
ALTER TABLE projects ADD COLUMN IF NOT EXISTS embedding_api_context_tokens INTEGER NOT NULL DEFAULT 0;
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


GROUNDED_RELATION_SOURCE_MODEL_COMPONENTS = frozenset({
    "source_inventory",
    "source_reference",
    "source_binding",
    "source_conflict",
    "batch_verifier",
    "source_structure_binding_audit",
    "source_projection_role_audit",
    "source_projection_group_audit",
    "source_projection_group_anchor_audit",
    "atomic_publication_audit",
})
GROUNDED_RELATION_REUSABLE_MODEL_COMPONENTS = frozenset({
    *GROUNDED_RELATION_SOURCE_MODEL_COMPONENTS,
    "proposer",
    "classifier",
})


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
            missing_naming = [
                version for version in KNOWLEDGE_FORGE_NAMING_REQUIRED_MIGRATIONS
                if not migrations.is_applied(conn, version)
            ]
            if missing_naming:
                raise RuntimeError(
                    "Knowledge Forge 命名迁移尚未完成；请先运行 "
                    "`.venv/bin/python scripts/manage_migrations.py apply`"
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

    def assert_grounded_relation_ready(self) -> dict[str, Any]:
        """Fail closed unless every immutable v2 migration and runtime table is present."""

        runner = MigrationRunner()
        with self.connection() as conn:
            missing_migrations = [
                version for version in GROUNDED_RELATION_V2_REQUIRED_MIGRATIONS
                if not runner.is_applied(conn, version)
            ]
            missing_tables: list[str] = []
            missing_columns: dict[str, list[str]] = {}
            for table, required_columns in GROUNDED_RELATION_V2_REQUIRED_TABLE_COLUMNS.items():
                relation = conn.execute(
                    "SELECT to_regclass(%s) AS table_name", (table,),
                ).fetchone()
                if not relation or not relation["table_name"]:
                    missing_tables.append(table)
                    continue
                columns = {
                    str(row["column_name"])
                    for row in conn.execute(
                        """SELECT column_name FROM information_schema.columns
                           WHERE table_schema=current_schema() AND table_name=%s""",
                        (table,),
                    ).fetchall()
                }
                absent = sorted(required_columns - columns)
                if absent:
                    missing_columns[table] = absent
            if missing_migrations or missing_tables or missing_columns:
                details = {
                    "missing_migrations": missing_migrations,
                    "missing_tables": sorted(missing_tables),
                    "missing_columns": missing_columns,
                }
                raise RuntimeError(
                    "Knowledge Forge 数据库结构不完整，"
                    "拒绝运行且不会回退旧流程: "
                    + self._canonical_json(details)
                )
            return {
                "migrations": list(GROUNDED_RELATION_V2_REQUIRED_MIGRATIONS),
                "tables": sorted(GROUNDED_RELATION_V2_REQUIRED_TABLE_COLUMNS),
            }

    def assert_graphrag_native_ready(self) -> dict[str, Any]:
        """Fail closed if the isolated native control-plane migration is incomplete."""

        runner = MigrationRunner()
        with self.connection() as conn:
            missing_migrations = [
                version for version in GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS
                if not runner.is_applied(conn, version)
            ]
            missing_tables: list[str] = []
            missing_columns: dict[str, list[str]] = {}
            for table, required_columns in GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS.items():
                relation = conn.execute(
                    "SELECT to_regclass(%s) AS table_name", (table,),
                ).fetchone()
                if not relation or not relation["table_name"]:
                    missing_tables.append(table)
                    continue
                columns = {
                    str(row["column_name"])
                    for row in conn.execute(
                        """SELECT column_name FROM information_schema.columns
                           WHERE table_schema=current_schema() AND table_name=%s""",
                        (table,),
                    ).fetchall()
                }
                absent = sorted(required_columns - columns)
                if absent:
                    missing_columns[table] = absent
            if missing_migrations or missing_tables or missing_columns:
                raise RuntimeError(
                    "GraphRAG Native 数据库结构不完整，拒绝运行且不会回退旧流程: "
                    + self._json({
                        "missing_migrations": missing_migrations,
                        "missing_tables": missing_tables,
                        "missing_columns": missing_columns,
                    })
                )
            return {
                "migrations": list(GRAPHRAG_NATIVE_REQUIRED_MIGRATIONS),
                "tables": sorted(GRAPHRAG_NATIVE_REQUIRED_TABLE_COLUMNS),
            }

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
        raw_connection = (
            conn._connection if isinstance(conn, InstrumentedConnection) else conn
        )
        if not isinstance(raw_connection, psycopg.Connection):
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
        return json.dumps(
            sanitize_unicode_data({} if value is None else value),
            ensure_ascii=False,
        )

    @staticmethod
    def _audit_json(value: Any) -> str:
        payload = {} if value is None else value
        if Config.audit_redaction_enabled:
            payload = redact_audit_payload(payload)
        return json.dumps(sanitize_unicode_data(payload), ensure_ascii=False)

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(
            sanitize_unicode_data({} if value is None else value),
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
                   graph_error=%s
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
        defaults = {
            "ollama_base_url": Config.ollama_base_url,
            "hugegraph_graph": Config.hugegraph_graph,
            # These are persisted defaults, not project overrides.  INSERT
            # ... DO NOTHING keeps an operator's saved values when the
            # service restarts or the deployment environment changes.
            "global_default_llm_model": Config.default_llm_model,
            "global_default_embedding_model": Config.default_embedding_model,
            "global_llm_provider": Config.llm_provider,
            "global_embedding_provider": Config.embedding_provider,
            "global_ollama_base_url": Config.ollama_base_url,
            "global_ollama_embedding_base_url": Config.ollama_embedding_base_url,
            "global_llm_api_base_url": Config.llm_api_base_url,
            "global_llm_api_key": Config.llm_api_key,
            "global_llm_api_models": ",".join(Config.llm_api_models),
            "global_llm_api_context_tokens": str(Config.llm_api_context_tokens),
            "global_embedding_api_base_url": Config.embedding_api_base_url,
            "global_embedding_api_key": Config.embedding_api_key,
            "global_embedding_api_models": ",".join(Config.embedding_api_models),
            "global_embedding_api_context_tokens": str(Config.embedding_api_context_tokens),
        }
        for key, value in defaults.items():
            conn.execute("INSERT INTO settings(key,value) VALUES(%s,%s) ON CONFLICT(key) DO NOTHING", (key, value))
        conn.execute(
            """INSERT INTO settings(key,value) VALUES('vector_backend','pgvector-hnsw')
            ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()"""
        )

    def get_settings(self, prefix: str = "") -> dict[str, str]:
        """Read persisted runtime settings, optionally limited by key prefix."""

        with self.connection() as conn:
            if prefix:
                rows = conn.execute(
                    "SELECT key,value FROM settings WHERE key LIKE %s ORDER BY key",
                    (f"{prefix}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT key,value FROM settings ORDER BY key",
                ).fetchall()
        return {
            str(row["key"]): str(row.get("value") or "")
            for row in rows
        }

    def set_settings(self, values: Mapping[str, object]) -> dict[str, str]:
        """Persist a bounded settings update and return the stored values."""

        normalized = {
            str(key): str(value if value is not None else "")
            for key, value in values.items()
            if str(key).strip()
        }
        if not normalized:
            return {}
        with self.connection() as conn:
            for key, value in normalized.items():
                conn.execute(
                    """INSERT INTO settings(key,value) VALUES(%s,%s)
                    ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value,updated_at=NOW()""",
                    (key, value),
                )
        return normalized

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

    def create_project(
        self,
        name: str,
        description: str,
        llm_model: str | None,
        embedding_model: str | None,
        *,
        strategy_key: str | None = None,
        knowledge_language: str = "zh",
    ) -> dict[str, Any]:
        selected_strategy = (
            normalize_strategy(strategy_key) if strategy_key is not None else STABLE_STRATEGY
        )
        if strategy_key is not None and selected_strategy not in {
            KNOWLEDGE_FORGE_STRATEGY,
            GRAPHRAG_NATIVE_STRATEGY,
        }:
            raise ValueError("新项目只能绑定 Knowledge Forge 或 GraphRAG Native")
        knowledge_language = normalize_knowledge_language(
            knowledge_language, default=CHINESE,
        )
        with self.connection() as conn:
            language_column = conn.execute(
                """SELECT 1 FROM information_schema.columns
                   WHERE table_schema=current_schema()
                     AND table_name='projects'
                     AND column_name='knowledge_language'""",
            ).fetchone()
            if language_column:
                row = conn.execute(
                    """INSERT INTO projects(
                       project_key,name,description,llm_model,embedding_model,
                       knowledge_language,graph_namespace_id)
                    VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        f"pending-{uuid.uuid4().hex}", name, description,
                        llm_model, embedding_model, knowledge_language,
                        str(uuid.uuid4()),
                    ),
                ).fetchone()
            else:
                # Migration and upgrade tests intentionally exercise historic
                # schemas.  Keep project creation usable until migration 0034
                # adds the forward-only language preference.
                row = conn.execute(
                    """INSERT INTO projects(
                       project_key,name,description,llm_model,embedding_model,
                       graph_namespace_id)
                    VALUES(%s,%s,%s,%s,%s,%s) RETURNING *""",
                    (
                        f"pending-{uuid.uuid4().hex}", name, description,
                        llm_model, embedding_model, str(uuid.uuid4()),
                    ),
                ).fetchone()
            key = self.project_key(int(row["id"]), name)
            result = conn.execute(
                "UPDATE projects SET project_key=%s,updated_at=NOW() WHERE id=%s RETURNING *",
                (key, row["id"]),
            ).fetchone()
            selection_table = conn.execute(
                "SELECT to_regclass('project_construction_strategy_state') AS table_name",
            ).fetchone()
            if selection_table and selection_table["table_name"]:
                binding_column = conn.execute(
                    """SELECT 1 FROM information_schema.columns
                       WHERE table_schema=current_schema()
                         AND table_name='project_construction_strategy_state'
                         AND column_name='association_status'""",
                ).fetchone()
                if binding_column:
                    association_status = "bound" if strategy_key is not None else "unassigned"
                    conn.execute(
                        """INSERT INTO project_construction_strategy_state(
                               project_id,selected_strategy,association_status,bound_at)
                           VALUES(%s,%s,%s,CASE WHEN %s='bound' THEN NOW() ELSE NULL END)
                           ON CONFLICT(project_id) DO NOTHING""",
                        (
                            int(row["id"]), selected_strategy, association_status,
                            association_status,
                        ),
                    )
                else:
                    conn.execute(
                        """INSERT INTO project_construction_strategy_state(
                               project_id,selected_strategy)
                           VALUES(%s,%s) ON CONFLICT(project_id) DO NOTHING""",
                        (int(row["id"]), selected_strategy),
                    )
            return result

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM projects WHERE is_deleted=FALSE ORDER BY updated_at DESC,id DESC").fetchall()

    def get_project(self, project_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE", (project_id,)).fetchone()

    def update_project(self, project_id: int, **values: Any) -> dict[str, Any]:
        allowed = {"llm_model", "embedding_model", "model_max_tokens", "chunk_max_tokens",
                   "llm_provider", "embedding_provider", "ollama_base_url",
                   "ollama_embedding_base_url", "llm_api_base_url", "llm_api_key",
                   "llm_api_models", "llm_api_context_tokens", "embedding_api_base_url",
                   "embedding_api_key", "embedding_api_models", "embedding_api_context_tokens",
                   "knowledge_language",
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
        stats: dict[str, int] | None = None,
        job_id: str = "",
    ) -> dict[str, Any] | None:
        """Publish a rebuild only if its immutable generation is still current."""

        status = GRAPH_EMPTY if empty else GRAPH_READY
        with self.connection() as conn:
            if job_id:
                job = conn.execute(
                    "SELECT status,metadata FROM jobs WHERE id=%s FOR UPDATE",
                    (str(job_id),),
                ).fetchone()
                metadata = (
                    job.get("metadata")
                    if job and isinstance(job.get("metadata"), dict)
                    else {}
                )
                if (
                    not job
                    or str(job.get("status") or "") not in {"queued", "running"}
                    or bool(metadata.get("cancel_requested"))
                ):
                    raise JobCancellationRequested(
                        "任务已终止，拒绝发布项目图谱"
                    )
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
            project = conn.execute(
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
            if not project:
                return None
            strategy_status = "empty" if empty else "ready"
            conn.execute(
                """UPDATE project_strategy_graphs
                   SET namespace_id=%s,generation=%s,ready_generation=%s,status=%s,
                       error_message='',stats=COALESCE(%s::jsonb,stats),
                       policy_summary=policy_summary ||
                           jsonb_build_object(
                               'policy_version',%s::text,
                               'policy_hash',%s::text,
                               'production_approved',TRUE
                           ),
                       updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s""",
                (
                    namespace_id,
                    target_generation,
                    target_generation,
                    strategy_status,
                    self._json(stats) if stats is not None else None,
                    GRAPH_POLICY_VERSION,
                    policy_hash,
                    project_id,
                    STABLE_STRATEGY,
                ),
            )
            conn.execute(
                """UPDATE documents SET graph_ready_at=NOW(),updated_at=NOW()
                   WHERE project_id=%s AND status='indexed' AND is_deleted=FALSE""",
                (project_id,),
            )
            if job_id:
                conn.execute(
                    """UPDATE jobs SET metadata=metadata || %s::jsonb,updated_at=NOW()
                       WHERE id=%s""",
                    (
                        self._json({
                            "publication_committed": True,
                            "publication_kind": "project_graph",
                            "published_generation": int(target_generation),
                        }),
                        str(job_id),
                    ),
                )
            return project

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
            if not project:
                raise RuntimeError("graph lease project does not exist")
            owner_generation: int | None = None
            if str(project["graph_namespace_id"]) == str(namespace_id):
                owner_generation = int(project["graph_generation"])
            else:
                strategy_graph = conn.execute(
                    """SELECT generation FROM project_strategy_graphs
                       WHERE project_id=%s AND namespace_id=%s FOR UPDATE""",
                    (project_id, namespace_id),
                ).fetchone()
                if strategy_graph:
                    owner_generation = int(strategy_graph["generation"])
            if owner_generation is None:
                raise RuntimeError("graph lease namespace does not own project")
            if owner_generation != int(graph_generation):
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

    def release_job_graph_leases(self, job_id: str) -> int:
        """Release graph leases owned by one interrupted durable job.

        Strategy writers and ordinary project graph rebuilds include the job
        id in their owner token.  After a process restart no old worker can
        still be using those leases, so cancellation recovery may release only
        those exact owners instead of waiting for the normal TTL.
        """

        owners = [
            f"strategy:{KNOWLEDGE_FORGE_STRATEGY}:{str(job_id)}",
            f"graph-rebuild:{str(job_id)}",
        ]
        with self.connection() as conn:
            result = conn.execute(
                """UPDATE graph_leases SET status='released',released_at=NOW()
                   WHERE owner=ANY(%s) AND status='active'""",
                (owners,),
            )
            return max(0, int(result.rowcount or 0))

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

    def list_documents(
        self,
        project_id: int | None = None,
        strategy_key: str = "",
    ) -> list[dict[str, Any]]:
        condition, params = "d.is_deleted=FALSE", []
        if project_id:
            condition += " AND d.project_id=%s"; params.append(project_id)
        strategy = normalize_strategy(strategy_key) if strategy_key else STABLE_STRATEGY
        params.insert(0, strategy)
        with self.connection() as conn:
            return conn.execute(
                f"""SELECT d.*,p.name AS project_name,
                    COALESCE(build.status,'uploaded') AS strategy_status,
                    COALESCE(build.error_message,'') AS strategy_error_message,
                    COALESCE(build.stats,'{{}}'::jsonb) AS strategy_stats,
                    active_revision.status AS active_revision_status,
                    active_revision.content_hash AS active_revision_content_hash,
                    active_revision.source_hash AS active_revision_source_hash,
                    active_revision.config_hash AS active_revision_config_hash,
                    (
                        SELECT COUNT(*) FROM chunks prepared_chunk
                        WHERE prepared_chunk.document_id=d.id
                          AND prepared_chunk.is_deleted=FALSE
                          AND (
                              d.active_revision_id IS NULL
                              OR prepared_chunk.revision_id=d.active_revision_id
                          )
                    ) AS prepared_chunk_count,
                    build.latest_job_id AS strategy_job_id,
                    build.content_revision_id AS strategy_revision_id,
                    build.started_at AS strategy_started_at,
                    build.completed_at AS strategy_completed_at,
                    build.updated_at AS strategy_updated_at
                FROM documents d JOIN projects p ON p.id=d.project_id
                LEFT JOIN document_strategy_builds build
                  ON build.document_id=d.id AND build.strategy_key=%s
                LEFT JOIN document_content_revisions active_revision
                  ON active_revision.revision_id=d.active_revision_id
                WHERE {condition} ORDER BY d.updated_at DESC,d.id DESC""", params
            ).fetchall()

    def ensure_project_strategy_graphs(self, project_id: int) -> list[dict[str, Any]]:
        """Create the two durable graph scopes for a newly created project."""

        with self.connection() as conn:
            project = conn.execute(
                "SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            conn.execute(
                """INSERT INTO project_strategy_graphs(
                       project_id,strategy_key,namespace_id,generation,ready_generation,
                       status,error_message,policy_summary,updated_at)
                   VALUES(%s,'stable',%s,%s,%s,%s,%s,%s::jsonb,NOW())
                   ON CONFLICT(project_id,strategy_key) DO UPDATE SET
                     namespace_id=EXCLUDED.namespace_id,generation=EXCLUDED.generation,
                     ready_generation=EXCLUDED.ready_generation,status=EXCLUDED.status,
                     error_message=EXCLUDED.error_message,
                     policy_summary=EXCLUDED.policy_summary,updated_at=NOW()""",
                (
                    project_id, project["graph_namespace_id"], project["graph_generation"],
                    project["graph_ready_generation"], project["graph_status"],
                    project.get("graph_error") or "",
                    self._json({
                        "policy_version": project.get("graph_policy_version") or "",
                        "policy_hash": project.get("graph_policy_hash") or "",
                        "production_approved": True,
                    }),
                ),
            )
            conn.execute(
                """INSERT INTO project_strategy_graphs(
                       project_id,strategy_key,namespace_id,status,policy_summary)
                   VALUES(%s,%s,%s,'empty',%s::jsonb)
                   ON CONFLICT(project_id,strategy_key) DO NOTHING""",
                (
                    project_id,
                    KNOWLEDGE_FORGE_STRATEGY,
                    str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"knowledge-forge:{project['graph_namespace_id']}:grounded",
                    )),
                    self._json({"mode": "grounded"}),
                ),
            )
            conn.execute(
                """UPDATE project_strategy_graphs graph
                   SET stats='{}'::jsonb,updated_at=NOW()
                   WHERE graph.project_id=%s
                     AND NOT EXISTS(
                         SELECT 1 FROM documents active_document
                         WHERE active_document.project_id=graph.project_id
                           AND active_document.is_deleted=FALSE
                     )
                     AND graph.stats<>'{}'::jsonb""",
                (project_id,),
            )
            return conn.execute(
                """SELECT * FROM project_strategy_graphs
                   WHERE project_id=%s ORDER BY strategy_key""",
                (project_id,),
            ).fetchall()

    def ensure_document_strategy_builds(
        self, document_id: int, *, reset: bool = False,
    ) -> list[dict[str, Any]]:
        """Ensure each strategy has an independent state row for the source file."""

        with self.connection() as conn:
            document = conn.execute(
                "SELECT project_id FROM documents WHERE id=%s FOR UPDATE", (document_id,),
            ).fetchone()
            if not document:
                raise LookupError("文档不存在")
            for strategy in (STABLE_STRATEGY, KNOWLEDGE_FORGE_STRATEGY):
                conn.execute(
                    """INSERT INTO document_strategy_builds(
                           project_id,document_id,strategy_key,status,error_message,stats)
                       VALUES(%s,%s,%s,'uploaded','','{}'::jsonb)
                       ON CONFLICT(document_id,strategy_key) DO UPDATE SET
                         status=CASE WHEN %s THEN 'uploaded'
                                     ELSE document_strategy_builds.status END,
                         content_revision_id=CASE WHEN %s THEN NULL
                                                  ELSE document_strategy_builds.content_revision_id END,
                         latest_job_id=CASE WHEN %s THEN NULL
                                            ELSE document_strategy_builds.latest_job_id END,
                         stats=CASE WHEN %s THEN '{}'::jsonb ELSE document_strategy_builds.stats END,
                         error_message=CASE WHEN %s THEN ''
                                            ELSE document_strategy_builds.error_message END,
                         started_at=CASE WHEN %s THEN NULL ELSE document_strategy_builds.started_at END,
                         completed_at=CASE WHEN %s THEN NULL ELSE document_strategy_builds.completed_at END,
                         updated_at=NOW()""",
                    (
                        document["project_id"], document_id, strategy,
                        reset, reset, reset, reset, reset, reset, reset,
                    ),
                )
            return conn.execute(
                """SELECT * FROM document_strategy_builds
                   WHERE document_id=%s ORDER BY strategy_key""",
                (document_id,),
            ).fetchall()

    @staticmethod
    def _validate_graphrag_native_relative_path(value: str, *, field: str) -> str:
        """Reject absolute, traversal, and empty critical bundle paths."""

        candidate = str(value or "").replace("\\", "/")
        if not candidate or candidate.startswith("/"):
            raise ValueError(f"{field} 必须是非空相对路径")
        parts = [part for part in candidate.split("/") if part]
        if not parts or any(part in {".", ".."} for part in parts):
            raise ValueError(f"{field} 包含不安全路径")
        return "/".join(parts)

    def ensure_graphrag_native_project_state(self, project_id: int) -> dict[str, Any]:
        """Create the native control-plane pointer without touching HugeGraph state."""

        with self.connection() as conn:
            project = conn.execute(
                "SELECT id FROM projects WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            conn.execute(
                """INSERT INTO graphrag_native_project_state(project_id)
                   VALUES(%s) ON CONFLICT(project_id) DO NOTHING""",
                (project_id,),
            )
            return conn.execute(
                "SELECT * FROM graphrag_native_project_state WHERE project_id=%s",
                (project_id,),
            ).fetchone()

    def get_graphrag_native_project_state(self, project_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM graphrag_native_project_state WHERE project_id=%s", (project_id,),
            ).fetchone()

    def get_graphrag_native_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM graphrag_native_runs WHERE run_id=%s", (run_id,),
            ).fetchone()

    def cancel_unpublished_graphrag_native_runs(
        self, project_id: int, *, reason: str,
    ) -> list[dict[str, Any]]:
        """Close restart-orphaned native attempts without touching READY runs."""

        with self.connection() as conn:
            return conn.execute(
                """UPDATE graphrag_native_runs
                   SET state='CANCELLED',failure_class='service_restart_cancel',
                       failure_message=%s,failed_at=COALESCE(failed_at,NOW()),
                       heartbeat_at=NOW(),lease_owner='',lease_expires_at=NULL
                   WHERE project_id=%s
                     AND state IN (
                       'PENDING','PREPARING','RUNNING','EXECUTION_COMPLETE','VALIDATING'
                     )
                   RETURNING *""",
                (str(reason or "服务恢复时确认构建已终止")[:4000], int(project_id)),
            ).fetchall()

    def list_graphrag_native_runs(self, project_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM graphrag_native_runs WHERE project_id=%s
                   ORDER BY generation DESC,created_at DESC""",
                (project_id,),
            ).fetchall()

    def create_graphrag_native_run(self, values: dict[str, Any]) -> dict[str, Any]:
        """Allocate a fenced generation for one immutable native bundle attempt."""

        required = {
            "run_id", "project_id", "base_sha", "worktree_diff_hash", "code_tree_hash",
            "source_snapshot_hash", "settings_hash", "prompt_profile_hash",
            "completion_model_id", "completion_model_digest", "embedding_model_id",
            "embedding_model_digest", "graphrag_version", "python_version",
        }
        missing = sorted(key for key in required if not str(values.get(key) or ""))
        if missing:
            raise ValueError("GraphRAG Native run 缺少不可变身份字段: " + ", ".join(missing))
        project_id = int(values["project_id"])
        run_id = str(values["run_id"])
        state = str(values.get("state") or "PENDING")
        if state != "PENDING":
            raise ValueError("GraphRAG Native run 必须从 PENDING 创建")
        staging_path = self._validate_graphrag_native_relative_path(
            str(values.get("staging_relative_path") or ""), field="staging_relative_path",
        )
        with self.connection() as conn:
            project = conn.execute(
                "SELECT id FROM projects WHERE id=%s AND is_deleted=FALSE FOR UPDATE",
                (project_id,),
            ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            pointer = conn.execute(
                """INSERT INTO graphrag_native_project_state(project_id)
                   VALUES(%s) ON CONFLICT(project_id) DO NOTHING
                   RETURNING project_id""",
                (project_id,),
            ).fetchone()
            if not pointer:
                pointer = conn.execute(
                    """SELECT * FROM graphrag_native_project_state
                       WHERE project_id=%s FOR UPDATE""",
                    (project_id,),
                ).fetchone()
            else:
                pointer = conn.execute(
                    """SELECT * FROM graphrag_native_project_state
                       WHERE project_id=%s FOR UPDATE""",
                    (project_id,),
                ).fetchone()
            generation = int(pointer.get("active_generation") or 0) + 1
            row = conn.execute(
                """INSERT INTO graphrag_native_runs(
                       run_id,project_id,generation,state,base_sha,worktree_diff_hash,
                       code_tree_hash,source_snapshot_hash,settings_hash,prompt_profile_hash,
                       completion_model_id,completion_model_digest,embedding_model_id,
                       embedding_model_digest,embedding_dimension,graphrag_version,python_version,
                       staging_relative_path,bundle_relative_path,validator_version
                   ) VALUES(%s,%s,%s,'PENDING',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'','')
                   RETURNING *""",
                (
                    run_id, project_id, generation, values["base_sha"],
                    values["worktree_diff_hash"], values["code_tree_hash"],
                    values["source_snapshot_hash"], values["settings_hash"],
                    values["prompt_profile_hash"], values["completion_model_id"],
                    values["completion_model_digest"], values["embedding_model_id"],
                    values["embedding_model_digest"], int(values.get("embedding_dimension") or 0),
                    values["graphrag_version"], values["python_version"], staging_path,
                ),
            ).fetchone()
            conn.execute(
                """UPDATE graphrag_native_project_state
                   SET active_generation=%s,desired_source_snapshot_hash=%s,updated_at=NOW()
                   WHERE project_id=%s""",
                (generation, str(values["source_snapshot_hash"]), project_id),
            )
            return row

    def acquire_graphrag_native_run_lease(
        self, run_id: str, owner: str, *, ttl_seconds: int = 900,
    ) -> dict[str, Any]:
        if not owner:
            raise ValueError("native lease owner 不能为空")
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE graphrag_native_runs SET lease_owner=%s,
                       lease_expires_at=NOW()+(%s*INTERVAL '1 second'),
                       heartbeat_at=NOW(),started_at=COALESCE(started_at,NOW()),
                       state=CASE WHEN state='PENDING' THEN 'PREPARING' ELSE state END
                   WHERE run_id=%s AND state IN ('PENDING','PREPARING','RUNNING','VALIDATING')
                     AND (lease_expires_at IS NULL OR lease_expires_at<NOW() OR lease_owner=%s)
                   RETURNING *""",
                (owner, max(30, int(ttl_seconds)), run_id, owner),
            ).fetchone()
            if not row:
                raise RuntimeError("GraphRAG Native run lease 已失效或被其他 worker 持有")
            return row

    def renew_graphrag_native_run_lease(
        self, run_id: str, owner: str, *, ttl_seconds: int = 900,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE graphrag_native_runs SET heartbeat_at=NOW(),
                       lease_expires_at=NOW()+(%s*INTERVAL '1 second')
                   WHERE run_id=%s AND lease_owner=%s
                     AND state IN ('PREPARING','RUNNING','VALIDATING')
                     AND lease_expires_at>NOW() RETURNING *""",
                (max(30, int(ttl_seconds)), run_id, owner),
            ).fetchone()
            if not row:
                raise RuntimeError("GraphRAG Native run lease 已失效")
            return row

    def release_graphrag_native_run_lease(self, run_id: str, owner: str) -> dict[str, Any]:
        """Release a completed/terminal native worker lease without deleting its run."""

        with self.connection() as conn:
            row = conn.execute(
                """UPDATE graphrag_native_runs SET lease_owner='',lease_expires_at=NULL,
                       heartbeat_at=NOW()
                   WHERE run_id=%s AND lease_owner=%s RETURNING *""",
                (str(run_id), str(owner)),
            ).fetchone()
            if not row:
                raise RuntimeError("GraphRAG Native run lease 无法释放：owner 不匹配或 run 不存在")
            return row

    def fail_graphrag_native_run_after_lease_loss(
        self,
        run_id: str,
        *,
        owner: str,
        state: str,
        failure_class: str,
        failure_message: str,
    ) -> dict[str, Any]:
        """Close the owning worker's run after its lease has already expired.

        Normal state changes deliberately require an unexpired lease.  That is
        essential for publishing.  A worker which *used* to own a lease still
        needs one safe, non-publishing operation though: recording that it
        stopped after losing that lease.  Requiring the same owner and an
        active (non-terminal) state prevents an old worker from modifying a
        run reclaimed by a different owner or from changing a published run.
        """

        if state not in {"FAILED", "CANCELLED", "NO_GRAPH_UNSUPPORTED"}:
            raise ValueError("GraphRAG Native lease-loss terminal state 非法")
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE graphrag_native_runs
                   SET state=%s,failure_class=%s,failure_message=%s,
                       failed_at=COALESCE(failed_at,NOW()),heartbeat_at=NOW(),
                       lease_owner='',lease_expires_at=NULL
                   WHERE run_id=%s AND lease_owner=%s
                     AND state IN ('PREPARING','RUNNING','VALIDATING','EXECUTION_COMPLETE')
                   RETURNING *""",
                (
                    state, str(failure_class)[:255], str(failure_message)[:4000],
                    str(run_id), str(owner),
                ),
            ).fetchone()
            if not row:
                raise RuntimeError("GraphRAG Native run 无法以 lease-loss 方式收尾")
            return row

    def update_graphrag_native_run(
        self, run_id: str, *, owner: str = "", **values: Any,
    ) -> dict[str, Any]:
        allowed = {
            "state", "bundle_relative_path", "validator_version", "validation_report_hash",
            "embedding_dimension", "failure_class", "failure_message", "started_at",
            "execution_completed_at", "validated_at", "ready_at", "failed_at",
        }
        illegal = sorted(set(values) - allowed)
        if illegal:
            raise ValueError("不允许更新 GraphRAG Native 不可变字段: " + ", ".join(illegal))
        if not values:
            existing = self.get_graphrag_native_run(run_id)
            if not existing:
                raise LookupError("GraphRAG Native run 不存在")
            return existing
        if "bundle_relative_path" in values and values["bundle_relative_path"]:
            values["bundle_relative_path"] = self._validate_graphrag_native_relative_path(
                str(values["bundle_relative_path"]), field="bundle_relative_path",
            )
        assignments = [f"{key}=%s" for key in values]
        params: list[Any] = [values[key] for key in values]
        # State timestamps are database timestamps, not caller-provided clock
        # values.  This preserves a trustworthy execution/validation trail.
        if values.get("state") == "EXECUTION_COMPLETE" and "execution_completed_at" not in values:
            assignments.append("execution_completed_at=COALESCE(execution_completed_at,NOW())")
        if values.get("state") == "VALIDATING" and "validated_at" not in values:
            assignments.append("validated_at=COALESCE(validated_at,NOW())")
        if values.get("state") in {"FAILED", "CANCELLED", "STALE", "SUPERSEDED"} and "failed_at" not in values:
            assignments.append("failed_at=COALESCE(failed_at,NOW())")
        clause = "run_id=%s"
        params.append(run_id)
        if owner:
            # A source mutation fences the run by moving it to STALE or
            # SUPERSEDED.  A late worker must never turn that terminal state
            # back into EXECUTION_COMPLETE/READY merely because it still owns
            # an old process lease.
            clause += " AND lease_owner=%s AND lease_expires_at>NOW() AND state NOT IN ('READY','EMPTY_INPUT','NO_GRAPH_UNSUPPORTED','FAILED','CANCELLED','STALE','SUPERSEDED')"
            params.append(owner)
        with self.connection() as conn:
            row = conn.execute(
                f"""UPDATE graphrag_native_runs SET {','.join(assignments)},
                       heartbeat_at=NOW() WHERE {clause} RETURNING *""",
                params,
            ).fetchone()
            if not row:
                raise RuntimeError("GraphRAG Native run 已过期、丢失 lease 或不存在")
            return row

    def replace_graphrag_native_run_documents(
        self, run_id: str, documents: list[dict[str, Any]], *, owner: str,
    ) -> None:
        if not documents:
            raise ValueError("GraphRAG Native run 至少需要一个输入文档")
        with self.connection() as conn:
            run = conn.execute(
                """SELECT * FROM graphrag_native_runs WHERE run_id=%s FOR UPDATE""", (run_id,),
            ).fetchone()
            if not run or run["lease_owner"] != owner or not run.get("lease_expires_at") or run["lease_expires_at"] <= datetime.now(timezone.utc):
                raise RuntimeError("GraphRAG Native run document 写入需要有效 lease")
            conn.execute("DELETE FROM graphrag_native_run_documents WHERE run_id=%s", (run_id,))
            for ordinal, item in enumerate(documents):
                if int(item["project_id"]) != int(run["project_id"]):
                    raise ValueError("GraphRAG Native run document 跨项目")
                conn.execute(
                    """INSERT INTO graphrag_native_run_documents(
                           run_id,project_id,document_id,revision_id,source_hash,structure_hash,
                           input_document_count,input_manifest_hash,ordinal
                       ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        run_id, int(item["project_id"]), int(item["document_id"]),
                        str(item["revision_id"]), str(item["source_hash"]),
                        str(item["structure_hash"]), int(item["input_document_count"]),
                        str(item["input_manifest_hash"]), ordinal,
                    ),
                )

    def replace_graphrag_native_artifacts(
        self, run_id: str, artifacts: list[dict[str, Any]], *, owner: str,
    ) -> None:
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM graphrag_native_runs WHERE run_id=%s FOR UPDATE", (run_id,),
            ).fetchone()
            if not run or run["lease_owner"] != owner or not run.get("lease_expires_at") or run["lease_expires_at"] <= datetime.now(timezone.utc):
                raise RuntimeError("GraphRAG Native artifact 写入需要有效 lease")
            conn.execute("DELETE FROM graphrag_native_artifacts WHERE run_id=%s", (run_id,))
            seen: set[str] = set()
            for item in artifacts:
                logical_name = str(item.get("logical_name") or "")
                if not logical_name or logical_name in seen:
                    raise ValueError("GraphRAG Native artifact logical_name 必须唯一")
                seen.add(logical_name)
                relative_path = self._validate_graphrag_native_relative_path(
                    str(item.get("relative_path") or ""), field="artifact.relative_path",
                )
                digest = str(item.get("sha256") or "")
                if not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("GraphRAG Native artifact SHA-256 非法")
                conn.execute(
                    """INSERT INTO graphrag_native_artifacts(
                           run_id,logical_name,relative_path,sha256,byte_size,media_type,
                           schema_hash,row_count
                       ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        run_id, logical_name, relative_path, digest,
                        int(item.get("byte_size") or 0), str(item.get("media_type") or ""),
                        str(item.get("schema_hash") or ""), int(item.get("row_count", -1)),
                    ),
                )

    def mark_graphrag_native_stale(self, project_id: int, reason: str) -> int:
        """Fence all native attempts after a source mutation; never touch old graphs."""

        with self.connection() as conn:
            self.ensure_graphrag_native_project_state(project_id)
            result = conn.execute(
                """UPDATE graphrag_native_runs SET state=CASE
                           WHEN state IN ('PENDING','PREPARING','RUNNING','VALIDATING') THEN 'SUPERSEDED'
                           WHEN state='READY' THEN 'STALE' ELSE state END,
                       failure_class=CASE WHEN state IN ('PENDING','PREPARING','RUNNING','VALIDATING')
                                          THEN 'source_changed' ELSE failure_class END,
                       failure_message=CASE WHEN state IN ('PENDING','PREPARING','RUNNING','VALIDATING')
                                            THEN %s ELSE failure_message END,
                       failed_at=CASE WHEN state IN ('PENDING','PREPARING','RUNNING','VALIDATING')
                                      THEN NOW() ELSE failed_at END
                   WHERE project_id=%s AND state IN ('PENDING','PREPARING','RUNNING','VALIDATING','READY')""",
                (str(reason)[:2000], project_id),
            )
            conn.execute(
                """UPDATE graphrag_native_project_state
                   SET desired_source_snapshot_hash='',updated_at=NOW() WHERE project_id=%s""",
                (project_id,),
            )
            return max(0, int(result.rowcount or 0))

    def activate_graphrag_native_run(
        self, run_id: str, *, owner: str, validation_report_hash: str,
        expected_source_snapshot_hash: str, job_id: str = "",
    ) -> dict[str, Any]:
        """CAS the native pointer only after validation and unchanged source fingerprint."""

        if not re.fullmatch(r"[0-9a-f]{64}", str(validation_report_hash)):
            raise ValueError("validation_report_hash 必须是 SHA-256")
        with self.connection() as conn:
            if job_id:
                job = conn.execute(
                    "SELECT status,metadata FROM jobs WHERE id=%s FOR UPDATE",
                    (str(job_id),),
                ).fetchone()
                metadata = (
                    job.get("metadata") if job and isinstance(job.get("metadata"), dict)
                    else {}
                )
                if (
                    not job
                    or str(job.get("status") or "") not in {"queued", "running"}
                    or bool(metadata.get("cancel_requested"))
                ):
                    raise JobCancellationRequested(
                        "任务已终止，拒绝发布 GraphRAG Native 成果"
                    )
            run = conn.execute(
                "SELECT * FROM graphrag_native_runs WHERE run_id=%s FOR UPDATE", (run_id,),
            ).fetchone()
            if not run or run["lease_owner"] != owner or not run.get("lease_expires_at") or run["lease_expires_at"] <= datetime.now(timezone.utc):
                raise RuntimeError("GraphRAG Native 激活需要有效 lease")
            if run["state"] != "VALIDATING":
                raise RuntimeError("只有 VALIDATING 的 GraphRAG Native run 可以激活")
            if run["source_snapshot_hash"] != expected_source_snapshot_hash:
                raise RuntimeError("GraphRAG Native source snapshot 已变化，拒绝激活")
            if not self._graphrag_native_run_source_is_current(conn, int(run["project_id"]), run_id):
                raise RuntimeError("GraphRAG Native source revision 已变化，拒绝激活旧 snapshot")
            pointer = conn.execute(
                """SELECT * FROM graphrag_native_project_state WHERE project_id=%s FOR UPDATE""",
                (run["project_id"],),
            ).fetchone()
            if not pointer or pointer["desired_source_snapshot_hash"] != expected_source_snapshot_hash or int(pointer["active_generation"] or 0) != int(run["generation"]):
                raise RuntimeError("GraphRAG Native generation 已被更新 source 或后续 run 取代")
            ready = conn.execute(
                """UPDATE graphrag_native_runs SET state='READY',validation_report_hash=%s,
                       validated_at=COALESCE(validated_at,NOW()),ready_at=NOW(),
                       lease_owner='',lease_expires_at=NULL WHERE run_id=%s RETURNING *""",
                (validation_report_hash, run_id),
            ).fetchone()
            conn.execute(
                """UPDATE graphrag_native_project_state SET active_run_id=%s,updated_at=NOW()
                   WHERE project_id=%s""",
                (run_id, run["project_id"]),
            )
            # Publishing a user-requested build is project activity. Keep the
            # recent-project order aligned with the Knowledge Forge path while
            # the Native graph storage itself remains completely independent.
            conn.execute(
                "UPDATE projects SET updated_at=NOW() WHERE id=%s AND is_deleted=FALSE",
                (run["project_id"],),
            )
            if job_id:
                conn.execute(
                    """UPDATE jobs SET metadata=metadata || %s::jsonb,updated_at=NOW()
                       WHERE id=%s""",
                    (
                        self._json({
                            "publication_committed": True,
                            "publication_kind": "graphrag_native_bundle",
                            "published_run_id": str(run_id),
                        }),
                        str(job_id),
                    ),
                )
            return ready

    @staticmethod
    def _graphrag_native_run_source_is_current(
        conn: psycopg.Connection, project_id: int, run_id: str,
    ) -> bool:
        """Compare a frozen run's document/revision set with current sources.

        This is intentionally database-local rather than relying only on a
        service invalidation callback: a direct revision mutation, restart, or
        late worker cannot activate an obsolete immutable bundle.
        """

        row = conn.execute(
            """SELECT NOT EXISTS(
                     SELECT 1 FROM graphrag_native_run_documents frozen
                     LEFT JOIN documents document ON document.id=frozen.document_id
                     LEFT JOIN document_content_revisions revision
                       ON revision.revision_id=document.active_revision_id
                     WHERE frozen.run_id=%s AND (
                       document.is_deleted=TRUE OR revision.revision_id IS NULL
                       OR revision.revision_id<>frozen.revision_id
                       OR revision.source_hash<>frozen.source_hash
                     )
                   ) AND NOT EXISTS(
                     SELECT 1 FROM documents current_document
                     WHERE current_document.project_id=%s AND current_document.is_deleted=FALSE
                       AND NOT EXISTS(
                         SELECT 1 FROM graphrag_native_run_documents frozen
                         WHERE frozen.run_id=%s AND frozen.document_id=current_document.id
                       )
                   ) AS current""",
            (str(run_id), int(project_id), str(run_id)),
        ).fetchone()
        return bool(row and row["current"])

    def rollback_graphrag_native_active_run(
        self, project_id: int, target_run_id: str, *, expected_active_run_id: str,
    ) -> dict[str, Any]:
        """CAS only the native pointer; no bundle, legacy graph, or strategy is changed."""

        with self.connection() as conn:
            pointer = conn.execute(
                """SELECT * FROM graphrag_native_project_state WHERE project_id=%s FOR UPDATE""",
                (int(project_id),),
            ).fetchone()
            if not pointer or str(pointer.get("active_run_id") or "") != str(expected_active_run_id):
                raise RuntimeError("GraphRAG Native rollback CAS 失败：active run 已变化")
            target = conn.execute(
                """SELECT * FROM graphrag_native_runs
                   WHERE run_id=%s AND project_id=%s FOR KEY SHARE""",
                (str(target_run_id), int(project_id)),
            ).fetchone()
            if not target or target.get("state") != "READY":
                raise RuntimeError("GraphRAG Native rollback target 必须是同项目 READY run")
            if not self._graphrag_native_run_source_is_current(conn, int(project_id), str(target_run_id)):
                raise RuntimeError("GraphRAG Native rollback target 已不对应当前 source")
            row = conn.execute(
                """UPDATE graphrag_native_project_state SET active_run_id=%s,
                       desired_source_snapshot_hash=%s,updated_at=NOW()
                   WHERE project_id=%s AND active_run_id=%s RETURNING *""",
                (str(target_run_id), str(target["source_snapshot_hash"]), int(project_id), str(expected_active_run_id)),
            ).fetchone()
            if not row:
                raise RuntimeError("GraphRAG Native rollback CAS 失败")
            return row

    def get_project_construction_strategy_selection(self, project_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM project_construction_strategy_state WHERE project_id=%s", (int(project_id),),
            ).fetchone()

    def bind_project_construction_strategy(
        self, project_id: int, strategy_key: str, *, expected_generation: int,
    ) -> dict[str, Any]:
        """Bind an unassigned project exactly once using a generation CAS."""

        strategy_key = normalize_strategy(strategy_key)
        if strategy_key not in {KNOWLEDGE_FORGE_STRATEGY, GRAPHRAG_NATIVE_STRATEGY}:
            raise ValueError("项目只能绑定 Knowledge Forge 或 GraphRAG Native")
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE project_construction_strategy_state
                   SET selected_strategy=%s,association_status='bound',bound_at=NOW(),
                       generation=generation+1,updated_at=NOW()
                   WHERE project_id=%s AND generation=%s
                     AND association_status='unassigned'
                     AND selected_strategy='stable'
                   RETURNING *""",
                (strategy_key, int(project_id), int(expected_generation)),
            ).fetchone()
            if not row:
                raise RuntimeError("项目策略绑定失败：项目已绑定或页面状态已过期")
            return row

    def assert_project_construction_strategy(
        self, project_id: int, expected_strategy: str | None = None,
    ) -> str:
        """Return the immutable project strategy or fail closed."""

        selection = self.get_project_construction_strategy_selection(project_id)
        if not selection:
            raise RuntimeError("项目缺少知识图谱策略状态，请先完成数据库迁移")
        if str(selection.get("association_status") or "") != "bound":
            raise ValueError("项目尚未绑定知识图谱策略，请先在项目首页完成一次性选择")
        strategy_key = normalize_strategy(str(selection.get("selected_strategy") or ""))
        if strategy_key not in {KNOWLEDGE_FORGE_STRATEGY, GRAPHRAG_NATIVE_STRATEGY}:
            raise RuntimeError("项目绑定了不可用于当前界面的知识图谱策略")
        if expected_strategy is not None and strategy_key != normalize_strategy(expected_strategy):
            raise ValueError("请求的知识图谱策略与项目绑定策略不一致")
        return strategy_key

    def select_project_construction_strategy(
        self, project_id: int, strategy_key: str, *, expected_generation: int,
    ) -> dict[str, Any]:
        """Compatibility name for the one-time binding operation."""

        return self.bind_project_construction_strategy(
            project_id, strategy_key, expected_generation=expected_generation,
        )

    def rollback_project_construction_strategy(
        self, project_id: int, target_strategy: str, *, expected_generation: int,
    ) -> dict[str, Any]:
        """Normal product workflows never rebind an existing project."""

        raise RuntimeError("项目知识图谱策略已固定，不能回滚或切换")

    def get_active_graphrag_native_run(self, project_id: int) -> dict[str, Any] | None:
        """Return the native-only active pointer, never a legacy strategy graph."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT run.* FROM graphrag_native_project_state state
                   JOIN graphrag_native_runs run ON run.run_id=state.active_run_id
                   WHERE state.project_id=%s""",
                (int(project_id),),
            ).fetchone()

    def list_graphrag_native_runs(self, project_id: int) -> list[dict[str, Any]]:
        """List native control-plane rows for operator retention planning only."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM graphrag_native_runs WHERE project_id=%s
                   ORDER BY created_at DESC,run_id DESC""",
                (int(project_id),),
            ).fetchall()

    def list_graphrag_native_run_documents(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM graphrag_native_run_documents
                   WHERE run_id=%s ORDER BY ordinal,document_id""",
                (str(run_id),),
            ).fetchall()

    def list_graphrag_native_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM graphrag_native_artifacts
                   WHERE run_id=%s ORDER BY logical_name""",
                (str(run_id),),
            ).fetchall()

    def create_graphrag_native_query_run(
        self, *, query_run_id: str, project_id: int, run_id: str,
        query_mode: str, question_hash: str, model_id: str,
    ) -> dict[str, Any]:
        if query_mode not in {"local", "global", "drift", "basic"}:
            raise ValueError("不支持的 GraphRAG Native query mode")
        if not re.fullmatch(r"[0-9a-f]{64}", str(question_hash)):
            raise ValueError("GraphRAG Native question_hash 必须是 SHA-256")
        with self.connection() as conn:
            run = conn.execute(
                """SELECT * FROM graphrag_native_runs WHERE run_id=%s AND project_id=%s
                   FOR KEY SHARE""",
                (str(run_id), int(project_id)),
            ).fetchone()
            if not run:
                raise LookupError("GraphRAG Native run 不属于当前项目")
            return conn.execute(
                """INSERT INTO graphrag_native_query_runs(
                       query_run_id,project_id,run_id,query_mode,question_hash,status,model_id,started_at
                   ) VALUES(%s,%s,%s,%s,%s,'RUNNING',%s,NOW()) RETURNING *""",
                (str(query_run_id), int(project_id), str(run_id), query_mode,
                 str(question_hash), str(model_id or "")),
            ).fetchone()

    def complete_graphrag_native_query_run(
        self, query_run_id: str, *, status: str, latency_ms: int,
        context_reference_count: int = 0, token_usage: dict[str, Any] | None = None,
        failure_class: str = "",
    ) -> dict[str, Any]:
        if status not in {"READY", "FAILED", "CANCELLED"}:
            raise ValueError("GraphRAG Native query 终态非法")
        payload = json.dumps(token_usage or {}, ensure_ascii=False, sort_keys=True)
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE graphrag_native_query_runs SET status=%s,completed_at=NOW(),
                       latency_ms=%s,context_reference_count=%s,token_usage=%s::jsonb,
                       failure_class=%s
                   WHERE query_run_id=%s AND status='RUNNING' RETURNING *""",
                (status, max(0, int(latency_ms)), max(0, int(context_reference_count)),
                 payload, str(failure_class or ""), str(query_run_id)),
            ).fetchone()
            if not row:
                raise RuntimeError("GraphRAG Native query run 不存在或已终态")
            return row

    def get_document_strategy_build(
        self, document_id: int, strategy_key: str,
    ) -> dict[str, Any] | None:
        strategy_key = normalize_strategy(strategy_key)
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM document_strategy_builds
                   WHERE document_id=%s AND strategy_key=%s""",
                (document_id, strategy_key),
            ).fetchone()

    def update_document_strategy_build(
        self, document_id: int, strategy_key: str, **values: Any,
    ) -> dict[str, Any]:
        strategy_key = normalize_strategy(strategy_key)
        allowed = {
            "status", "content_revision_id", "latest_job_id", "stats",
            "error_message", "started_at", "completed_at",
        }
        fields = [name for name in values if name in allowed]
        if not fields:
            row = self.get_document_strategy_build(document_id, strategy_key)
            if not row:
                raise LookupError("文档策略状态不存在")
            return row
        assignments = ",".join(
            f"{name}={'%s::jsonb' if name == 'stats' else '%s'}" for name in fields
        )
        params = [
            self._json(values[name]) if name == "stats" and not isinstance(values[name], str)
            else values[name]
            for name in fields
        ]
        with self.connection() as conn:
            row = conn.execute(
                f"""UPDATE document_strategy_builds SET {assignments},updated_at=NOW()
                    WHERE document_id=%s AND strategy_key=%s RETURNING *""",
                (*params, document_id, strategy_key),
            ).fetchone()
            if not row:
                raise LookupError("文档策略状态不存在")
            return row

    def get_project_strategy_graph(
        self, project_id: int, strategy_key: str,
    ) -> dict[str, Any] | None:
        strategy_key = normalize_strategy(strategy_key)
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM project_strategy_graphs
                   WHERE project_id=%s AND strategy_key=%s""",
                (project_id, strategy_key),
            ).fetchone()

    def reconcile_knowledge_forge_schema_freshness(
        self, project_id: int,
    ) -> dict[str, Any] | None:
        """Fail closed when a ready Knowledge Forge snapshot predates its active Schema.

        Older graph rows did not persist the logical Schema version used by the
        extraction run.  Such a row cannot prove that its normalized facts are
        compatible with the currently active Schema, so it must be rebuilt.
        The transition is idempotent: only a graph still advertised as ready
        advances its CAS generation.
        """

        with self.connection() as conn:
            return conn.execute(
                """UPDATE project_strategy_graphs graph
                   SET generation=graph.generation+1,status='stale',
                       error_message='逻辑 Schema 已变化，Knowledge Forge 需要重新构建',
                       lease_token='',lease_expires_at=NULL,
                       policy_summary=graph.policy_summary || jsonb_build_object(
                           'invalidated_by','logical_schema',
                           'invalidated_schema_version',project.active_schema_version
                       ),
                       updated_at=NOW()
                   FROM projects project
                   WHERE graph.project_id=project.id
                     AND graph.project_id=%s
                     AND graph.strategy_key=%s
                     AND graph.status='ready'
                     AND CASE
                         WHEN COALESCE(
                             graph.policy_summary->>'schema_version',''
                         ) ~ '^[0-9]+$'
                         THEN (graph.policy_summary->>'schema_version')::BIGINT
                             <> project.active_schema_version
                         ELSE TRUE
                     END
                   RETURNING graph.*""",
                (project_id, KNOWLEDGE_FORGE_STRATEGY),
            ).fetchone()

    def reconcile_knowledge_forge_entity_freshness(
        self,
        project_id: int,
        *,
        code_version: str,
        policy_set: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Invalidate a ready graph backed by an older entity treatment.

        Document strategy rows describe source preparation, so they remain
        ``indexed`` when only extraction or identity-resolution code changes.
        The project graph must nevertheless be rebuilt unless its referenced,
        complete entity namespace proves both the loaded code version and all
        required policy axes. Missing legacy metadata deliberately fails
        closed once.
        """

        expected_code = str(code_version or "").strip()
        expected_policy = {
            str(key): str(value)
            for key, value in dict(policy_set or {}).items()
            if str(key).strip() and str(value).strip()
        }
        if not expected_code or not expected_policy:
            raise ValueError(
                "entity freshness requires a code version and policy set"
            )
        with self.connection() as conn:
            return conn.execute(
                """UPDATE project_strategy_graphs graph
                   SET generation=graph.generation+1,status='stale',
                       error_message=(
                         '实体抽取或归一策略已升级，Knowledge Forge 需要重新构建'
                       ),
                       lease_token='',lease_expires_at=NULL,
                       policy_summary=graph.policy_summary || jsonb_build_object(
                           'invalidated_by','entity_treatment',
                           'expected_entity_code_version',%s::text,
                           'expected_entity_policy_set',%s::jsonb
                       ),
                       updated_at=NOW()
                   WHERE graph.project_id=%s
                     AND graph.strategy_key=%s
                     AND graph.status='ready'
                     AND COALESCE(
                       graph.policy_summary #>>
                         '{relation_artifact_preservation,mode}',
                       ''
                     )<>'frozen_artifact_reprojection_v1'
                     AND NOT EXISTS (
                       SELECT 1
                       FROM entity_shadow_namespaces namespace
                       WHERE namespace.id=(
                           graph.policy_summary->>'entity_namespace_id'
                       )
                         AND namespace.project_id=graph.project_id
                         AND namespace.status='ready'
                         AND namespace.snapshot_complete=TRUE
                         AND namespace.code_version=%s
                         AND namespace.policy_set @> %s::jsonb
                     )
                   RETURNING graph.*""",
                (
                    expected_code,
                    self._json(expected_policy),
                    int(project_id),
                    KNOWLEDGE_FORGE_STRATEGY,
                    expected_code,
                    self._json(expected_policy),
                ),
            ).fetchone()

    def reconcile_knowledge_forge_relation_freshness(
        self,
        project_id: int,
        *,
        code_digest: str,
        treatment_digest: str,
    ) -> dict[str, Any] | None:
        """Invalidate a ready graph whose relation treatment is not loaded.

        Source bytes and entity assignments can remain reusable while relation
        candidate, verifier, normalization or deployment-threshold policy has
        changed. A ready strategy graph must therefore prove that its exact
        referenced relation run used both the currently loaded code tree and
        runtime treatment digest. Missing legacy metadata fails closed once.
        """

        expected_code = str(code_digest or "").strip()
        expected_treatment = str(treatment_digest or "").strip()
        if not expected_code or not expected_treatment:
            raise ValueError(
                "relation freshness requires code and treatment digests"
            )
        with self.connection() as conn:
            return conn.execute(
                """UPDATE project_strategy_graphs graph
                   SET generation=graph.generation+1,status='stale',
                       error_message=(
                         '关系抽取策略已升级，Knowledge Forge 需要重新构建'
                       ),
                       lease_token='',lease_expires_at=NULL,
                       policy_summary=graph.policy_summary || jsonb_build_object(
                           'invalidated_by','relation_treatment',
                           'expected_relation_code_tree',%s::text,
                           'expected_relation_treatment',%s::text
                       ),
                       updated_at=NOW()
                   WHERE graph.project_id=%s
                     AND graph.strategy_key=%s
                     AND graph.status='ready'
                     -- A recovered graph may intentionally retain a frozen,
                     -- audited relation run after an implementation-only
                     -- deployment.  Its public read path independently
                     -- proves the published endpoint set still equals that
                     -- run's eligible fact set before exposing it.  Do not
                     -- invalidate it here merely because the current runtime
                     -- treatment digest differs; source/schema changes still
                     -- advance the normal graph generation elsewhere.
                     AND COALESCE(
                       graph.policy_summary #>>
                         '{relation_artifact_preservation,mode}',
                       ''
                     )<>'frozen_artifact_reprojection_v1'
                     AND NOT EXISTS (
                       SELECT 1
                       FROM shadow_relation_v2_runs run
                       WHERE run.id=(
                           graph.policy_summary #>>
                           '{entity_metrics,relation_v2_run_id}'
                       )
                         AND run.project_id=graph.project_id
                         AND run.status='ready'
                         AND run.fingerprint_components #>>
                               '{materialization_root,code_tree}'=%s
                         AND run.fingerprint_components #>>
                               '{materialization_root,treatment}'=%s
                     )
                   RETURNING graph.*""",
                (
                    expected_code,
                    expected_treatment,
                    int(project_id),
                    KNOWLEDGE_FORGE_STRATEGY,
                    expected_code,
                    expected_treatment,
                ),
            ).fetchone()

    def mark_strategy_graph_stale(
        self, project_id: int, strategy_key: str, reason: str,
    ) -> dict[str, Any]:
        strategy_key = normalize_strategy(strategy_key)
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE project_strategy_graphs
                   SET generation=generation+1,status='stale',error_message=%s,
                       lease_token='',lease_expires_at=NULL,updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s RETURNING *""",
                (reason or "策略事实已更新，等待重建", project_id, strategy_key),
            ).fetchone()
            if not row:
                raise LookupError("项目策略图状态不存在")
            return row

    def invalidate_knowledge_forge_for_document(
        self, document_id: int, *, reason: str,
    ) -> None:
        """Invalidate derived Knowledge Forge results for an explicit source mutation.

        The Knowledge Forge strategy is a project-wide snapshot.  Callers use this at
        document/source mutation boundaries, not merely because a stable
        strategy rebuild starts.  Empty graphs stay empty so a first baseline
        build is not presented as a failed or stale experiment.
        """

        with self.connection() as conn:
            document = conn.execute(
                "SELECT project_id FROM documents WHERE id=%s AND is_deleted=FALSE",
                (document_id,),
            ).fetchone()
            if not document:
                raise LookupError("文档不存在或已删除")
            conn.execute(
                """UPDATE document_strategy_builds
                   SET status='uploaded',content_revision_id=NULL,
                       stats='{}'::jsonb,error_message='',started_at=NULL,
                       completed_at=NULL,updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s AND status<>'deleted'""",
                (document["project_id"], KNOWLEDGE_FORGE_STRATEGY),
            )
            conn.execute(
                """UPDATE project_strategy_graphs
                   SET generation=generation+1,status='stale',error_message=%s,
                       lease_token='',lease_expires_at=NULL,updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s AND status<>'empty'""",
                (
                    reason or "稳定基线已更新，Knowledge Forge 需要重新构建",
                    document["project_id"], KNOWLEDGE_FORGE_STRATEGY,
                ),
            )

    def start_strategy_graph_rebuild(
        self, project_id: int, strategy_key: str,
    ) -> dict[str, Any]:
        strategy_key = normalize_strategy(strategy_key)
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE project_strategy_graphs SET status='rebuilding',error_message='',
                       updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s RETURNING *""",
                (project_id, strategy_key),
            ).fetchone()
            if not row:
                raise LookupError("项目策略图状态不存在")
            return row

    def complete_strategy_graph_rebuild(
        self, project_id: int, strategy_key: str, generation: int,
        *, stats: dict[str, Any], policy_summary: dict[str, Any], empty: bool = False,
        job_id: str = "",
    ) -> dict[str, Any] | None:
        strategy_key = normalize_strategy(strategy_key)
        with self.connection() as conn:
            if job_id:
                job = conn.execute(
                    "SELECT status,metadata FROM jobs WHERE id=%s FOR UPDATE",
                    (str(job_id),),
                ).fetchone()
                metadata = (
                    job.get("metadata") if job and isinstance(job.get("metadata"), dict)
                    else {}
                )
                if (
                    not job
                    or str(job.get("status") or "") not in {"queued", "running"}
                    or bool(metadata.get("cancel_requested"))
                ):
                    raise JobCancellationRequested(
                        "任务已终止，拒绝发布策略图谱"
                    )
            row = conn.execute(
                """UPDATE project_strategy_graphs SET status=%s,ready_generation=%s,
                       error_message='',stats=%s::jsonb,policy_summary=%s::jsonb,
                       updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s AND generation=%s
                   RETURNING *""",
                (
                    "empty" if empty else "ready", generation, self._json(stats),
                    self._json(policy_summary), project_id, strategy_key, generation,
                ),
            ).fetchone()
            if row and job_id:
                conn.execute(
                    """UPDATE jobs SET metadata=metadata || %s::jsonb,updated_at=NOW()
                       WHERE id=%s""",
                    (
                        self._json({
                            "publication_committed": True,
                            "publication_kind": "strategy_graph",
                            "published_generation": int(generation),
                        }),
                        str(job_id),
                    ),
                )
            return row

    def fail_strategy_graph_rebuild(
        self, project_id: int, strategy_key: str, generation: int, error: str,
    ) -> dict[str, Any] | None:
        strategy_key = normalize_strategy(strategy_key)
        with self.connection() as conn:
            return conn.execute(
                """UPDATE project_strategy_graphs SET status='failed',error_message=%s,
                       updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s AND generation=%s
                   RETURNING *""",
                (error[:2000], project_id, strategy_key, generation),
            ).fetchone()

    def cancel_strategy_graph_rebuild(
        self, project_id: int, strategy_key: str, generation: int, reason: str,
    ) -> dict[str, Any] | None:
        """Close an unpublished strategy generation after cooperative cancel.

        A previous ready generation remains a valid historical artifact, but
        the current generation must not be advertised as ready.  First builds
        return to ``empty``; rebuilds remain ``stale`` and can be retried.
        """

        strategy_key = normalize_strategy(strategy_key)
        with self.connection() as conn:
            return conn.execute(
                """UPDATE project_strategy_graphs
                   SET status=CASE WHEN ready_generation>0 THEN 'stale' ELSE 'empty' END,
                       error_message=%s,lease_token='',lease_expires_at=NULL,
                       updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s AND generation=%s
                     AND status='rebuilding'
                   RETURNING *""",
                (
                    str(reason or "构建已安全终止，未发布未完成图谱")[:2000],
                    project_id, strategy_key, generation,
                ),
            ).fetchone()

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

    def abandon_job_document_revision(
        self, ingestion_run_id: str, *, reason: str,
    ) -> dict[str, Any] | None:
        """Close only an unpublished revision owned by an interrupted job."""

        with self.connection() as conn:
            return conn.execute(
                """UPDATE document_content_revisions
                   SET status='failed',failure=%s,failed_at=NOW(),lease_token=NULL,
                       lease_expires_at=NULL,updated_at=NOW()
                   WHERE ingestion_run_id=%s AND status='building'
                   RETURNING *""",
                (str(reason or "任务已安全终止")[:2000], str(ingestion_run_id)),
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
        mutate_legacy_graph: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Atomically publish a complete revision and advance the graph generation."""

        document_values = dict(document_values or {})
        # Revision publication owns these fields. Callers may still send the
        # legacy content_version value, but it must never produce a duplicate
        # assignment in the UPDATE statement.
        for reserved in ("active_revision_id", "content_version", "updated_at"):
            document_values.pop(reserved, None)
        json_fields = {"metadata", "chunk_config", "parse_quality"}
        with self.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (document_id,))
            job = conn.execute(
                "SELECT kind,status,metadata FROM jobs WHERE id=%s FOR UPDATE",
                (str(ingestion_run_id),),
            ).fetchone()
            job_metadata = (
                job.get("metadata") if job and isinstance(job.get("metadata"), dict)
                else {}
            )
            if job and (
                str(job.get("status") or "") not in {"queued", "running"}
                or bool(job_metadata.get("cancel_requested"))
            ):
                raise JobCancellationRequested(
                    "任务已终止，拒绝发布文档预处理结果"
                )
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
            if mutate_legacy_graph:
                self._retire_orphan_entities(conn, int(document["project_id"]))
                project = conn.execute(
                    """UPDATE projects SET graph_generation=graph_generation+1,graph_status=%s,
                           graph_policy_version='',graph_policy_hash='',graph_error=%s,updated_at=NOW()
                       WHERE id=%s AND is_deleted=FALSE RETURNING *""",
                    (GRAPH_STALE, reason, document["project_id"]),
                ).fetchone()
            else:
                project = conn.execute(
                    "SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE",
                    (document["project_id"],),
                ).fetchone()
            if not project:
                raise LookupError("项目不存在")
            # A content-preparation job owns exactly one document revision.
            # Mark its publication in the same transaction so a cancellation
            # cannot be accepted in the tiny tail before the worker writes its
            # terminal job status.  Ingestion jobs continue into graph
            # construction and therefore must remain cancellable here.
            if job and str(job.get("kind") or "") == "content_preparation":
                conn.execute(
                    """UPDATE jobs SET metadata=metadata || %s::jsonb,updated_at=NOW()
                       WHERE id=%s""",
                    (
                        self._json({
                            "publication_committed": True,
                            "publication_kind": "document_revision",
                            "published_revision_id": str(revision_id),
                        }),
                        str(ingestion_run_id),
                    ),
                )
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
                conn.execute(
                    """UPDATE document_strategy_builds
                       SET status='deleted',error_message='',completed_at=NOW(),updated_at=NOW()
                       WHERE document_id=%s""",
                    (document_id,),
                )
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
                conn.execute(
                    """UPDATE project_strategy_graphs
                       SET generation=generation+1,status='stale',
                           stats=CASE WHEN NOT EXISTS(
                               SELECT 1 FROM documents active_document
                               WHERE active_document.project_id=%s
                                 AND active_document.is_deleted=FALSE
                           ) THEN '{}'::jsonb ELSE stats END,
                           error_message=%s,lease_token='',lease_expires_at=NULL,updated_at=NOW()
                       WHERE project_id=%s""",
                    (
                        row["project_id"],
                        "项目文档已删除，当前方案图谱需要重新构建",
                        row["project_id"],
                    ),
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

        sections = sanitize_unicode_data(sections)
        chunks = sanitize_unicode_data(chunks)
        for item in [*sections, *chunks]:
            if "content" in item:
                item["content_hash"] = hashlib.sha256(
                    str(item.get("content") or "").encode("utf-8")
                ).hexdigest()

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
                JOIN documents d ON d.id=c.document_id
                WHERE c.document_id=%s AND c.is_deleted=FALSE AND d.is_deleted=FALSE
                AND c.revision_id=COALESCE(
                    (SELECT revision.revision_id FROM document_content_revisions revision
                     WHERE revision.document_id=d.id AND revision.status='building'
                     ORDER BY revision.content_version DESC,revision.created_at DESC LIMIT 1),
                    d.active_revision_id
                )
                ORDER BY c.chunk_index,c.id""", (document_id,)
            ).fetchall()

    def get_chunk_provenance(
        self, project_id: int, chunk_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        """Load source presentation metadata for exact current chunks."""

        ids = sorted({int(value) for value in chunk_ids if int(value) > 0})
        if not ids:
            return {}
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT chunk.id,chunk.document_id,chunk.revision_id,
                          chunk.content,chunk.metadata,
                          section.title AS section_title,
                          document.original_filename,document.storage_path,
                          document.mime_type
                   FROM chunks chunk
                   JOIN documents document ON document.id=chunk.document_id
                   LEFT JOIN sections section ON section.id=chunk.section_id
                   WHERE chunk.project_id=%s AND chunk.id=ANY(%s)
                     AND chunk.is_deleted=FALSE AND document.is_deleted=FALSE
                     AND document.active_revision_id=chunk.revision_id""",
                (int(project_id), ids),
            ).fetchall()
            return {int(item["id"]): item for item in rows}

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
                   AND failure.status='pending' AND failure.next_attempt_at<=NOW()
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

    def list_active_embeddings_for_runtime_reindex(
        self,
        runtime_profile: str,
        *,
        project_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return only current, rebuildable vectors from another runtime."""

        project_filter = "AND v.project_id=%s" if project_id is not None else ""
        params: list[Any] = [str(runtime_profile)]
        if project_id is not None:
            params.append(int(project_id))
        with self.connection() as conn:
            return conn.execute(
                f"""SELECT v.id,v.project_id,v.object_type,v.object_key,v.content,
                           v.model_name,v.dimension
                    FROM vector_embeddings v
                    JOIN projects p ON p.id=v.project_id
                    LEFT JOIN documents d ON d.id=v.document_id
                    WHERE v.is_deleted=FALSE AND p.is_deleted=FALSE
                      AND COALESCE(v.metadata->>'embedding_runtime_profile','')<>%s
                      AND (v.document_id IS NULL OR (
                        d.is_deleted=FALSE AND d.active_revision_id=v.revision_id
                      ))
                      {project_filter}
                    ORDER BY v.model_name,v.project_id,v.object_type,v.id""",
                tuple(params),
            ).fetchall()

    def replace_embedding_runtime_batch(
        self,
        records: list[dict[str, Any]],
        runtime_profile: str,
    ) -> int:
        """Atomically replace a bounded vector batch and stamp its backend."""

        if not records:
            return 0
        updated = 0
        with self.connection() as conn:
            for record in records:
                vector = [float(value) for value in record.get("embedding") or []]
                if not vector or not all(math.isfinite(value) for value in vector):
                    raise ValueError("Embedding 重建返回了空向量或非有限数值")
                model_name = str(record["model_name"])
                self._ensure_hnsw_index(conn, model_name, len(vector))
                result = conn.execute(
                    """UPDATE vector_embeddings
                       SET embedding=%s,embedding_vector=%s::vector,dimension=%s,
                           metadata=metadata||%s::jsonb,updated_at=NOW()
                       WHERE id=%s AND is_deleted=FALSE AND model_name=%s""",
                    (
                        vector, self._vector_literal(vector), len(vector),
                        self._json({
                            "embedding_runtime_profile": str(runtime_profile),
                        }),
                        int(record["id"]), model_name,
                    ),
                )
                updated += max(0, int(result.rowcount or 0))
        return updated

    def list_document_embeddings(
        self,
        document_id: int,
        object_type: str,
        model_name: str,
        *,
        revision_id: str | None = None,
    ) -> list[dict[str, Any]]:
        revision_filter = (
            "v.revision_id=%s"
            if revision_id
            else "d.active_revision_id=v.revision_id"
        )
        params: tuple[Any, ...] = (
            (document_id, object_type, model_name, revision_id)
            if revision_id
            else (document_id, object_type, model_name)
        )
        with self.connection() as conn:
            return conn.execute(
                f"""SELECT v.id,v.project_id,v.document_id,v.chunk_id,v.revision_id,v.object_type,
                          v.object_key,v.content,v.content_hash,v.model_name,
                          v.embedding,v.dimension,v.metadata
                   FROM vector_embeddings v JOIN documents d ON d.id=v.document_id
                   WHERE v.document_id=%s AND v.object_type=%s AND v.model_name=%s
                     AND v.is_deleted=FALSE AND d.is_deleted=FALSE
                     AND {revision_filter}
                   ORDER BY v.chunk_id,v.id""",
                params,
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

    def create_job(
        self,
        kind: str,
        project_id: int | None = None,
        document_id: int | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        strategy_key: str = STABLE_STRATEGY,
    ) -> dict[str, Any]:
        strategy_key = normalize_strategy(strategy_key)
        job_id = uuid.uuid4().hex
        with self.connection() as conn:
            job = conn.execute(
                """INSERT INTO jobs(id,kind,project_id,document_id,metadata,strategy_key)
                   VALUES(%s,%s,%s,%s,%s::jsonb,%s) RETURNING *""",
                (job_id, kind, project_id, document_id, self._json(metadata), strategy_key),
            ).fetchone()
            # The project chooser is ordered by project activity. Creating a
            # durable project job is user-visible activity even when the job
            # itself does not mutate project metadata, so keep the project in
            # the recent-project list while work is queued or resumed.
            if project_id is not None:
                conn.execute(
                    "UPDATE projects SET updated_at=NOW() WHERE id=%s AND is_deleted=FALSE",
                    (project_id,),
                )
            return job

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
            metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
            if (
                str(job.get("status") or "") not in {"queued", "running"}
                or bool(metadata.get("cancel_requested"))
            ):
                raise RuntimeError("任务已取消或结束，拒绝开始图谱写入")
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
            next_status = status or row["status"]
            existing_metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            merged_metadata = {**existing_metadata, **metadata} if metadata is not None else existing_metadata
            terminal = {"completed", "failed", "cancelled"}
            # A late worker update must never revive a terminal job.  While a
            # running job is cooperatively stopping, suppress ordinary progress
            # writes so the durable cancellation request remains authoritative;
            # the worker may still publish an explicit terminal result.
            if str(row.get("status") or "") in terminal and next_status != row["status"]:
                return row
            if (
                bool(existing_metadata.get("cancel_requested"))
                and next_status == "completed"
                and not bool(existing_metadata.get("publication_committed"))
            ):
                raise JobCancellationRequested(
                    "任务已请求终止，拒绝记录未发布的完成状态"
                )
            if (
                bool(existing_metadata.get("cancel_requested"))
                and next_status not in terminal
            ):
                return row
            # Cancellation acknowledgement is a terminal worker write, not an
            # ordinary progress update.  It may close the currently active UI
            # subtask, but it must not be able to alter any other durable
            # metadata that was frozen when the cancellation was requested.
            if (
                bool(existing_metadata.get("cancel_requested"))
                and next_status == "cancelled"
                and metadata is not None
            ):
                safe_metadata: dict[str, Any] = {}
                if metadata.get("stage") == "cancelled":
                    safe_metadata["stage"] = "cancelled"
                if metadata.get("cancel_acknowledged_at"):
                    safe_metadata["cancel_acknowledged_at"] = str(
                        metadata["cancel_acknowledged_at"]
                    )
                candidate_subtasks = metadata.get("subtasks")
                if isinstance(candidate_subtasks, dict):
                    existing_subtasks = (
                        existing_metadata.get("subtasks")
                        if isinstance(existing_metadata.get("subtasks"), dict)
                        else {}
                    )
                    closed_subtasks = {
                        key: value
                        for key, value in candidate_subtasks.items()
                        if (
                            key in existing_subtasks
                            and isinstance(value, dict)
                            and str(value.get("status") or "") == "cancelled"
                        )
                    }
                    if closed_subtasks:
                        safe_metadata["subtasks"] = {
                            **existing_subtasks,
                            **closed_subtasks,
                        }
                merged_metadata = {**existing_metadata, **safe_metadata}
            logs = list(row.get("log") or [])
            if log: logs.append(log)
            return conn.execute(
                """UPDATE jobs SET status=%s,progress=%s,log=%s::jsonb,error_message=NULLIF(%s,''),
                metadata=%s::jsonb,updated_at=NOW(),completed_at=CASE WHEN %s IN ('completed','failed','cancelled') THEN NOW() ELSE completed_at END
                WHERE id=%s RETURNING *""",
                (next_status,progress if progress is not None else row["progress"],self._json(logs),error,self._json(merged_metadata),next_status,job_id),
            ).fetchone()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connection() as conn: return conn.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()

    def request_job_cancellation(
        self,
        job_id: str,
        *,
        project_id: int,
        allowed_kinds: list[str],
        reason: str,
    ) -> dict[str, Any]:
        """Persist a cooperative cancellation without lying about worker exit.

        Queued work has not entered a side-effecting worker and can become
        terminal immediately.  Running work remains active until its worker
        reaches a checkpoint, rolls back or abandons staging output, and
        acknowledges cancellation.
        """

        if not allowed_kinds:
            raise ValueError("未指定可取消的任务类型")
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id=%s FOR UPDATE", (str(job_id),),
            ).fetchone()
            if not row or int(row.get("project_id") or 0) != int(project_id):
                raise LookupError("任务不存在或不属于当前项目")
            if str(row.get("kind") or "") not in set(allowed_kinds):
                raise ValueError("该任务不支持从当前页面终止")
            if str(row.get("status") or "") not in {"queued", "running"}:
                raise ValueError("任务已经结束，无需终止")
            existing_metadata = (
                row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            )
            if existing_metadata.get("publication_committed"):
                raise JobPublicationCommitted(
                    "任务成果已经发布，正在完成收尾，无需终止"
                )
            already_requested = bool(existing_metadata.get("cancel_requested"))
            requested_at = str(
                existing_metadata.get("cancel_requested_at")
                or datetime.now(timezone.utc).isoformat()
            )
            cancellation_reason = str(reason or "用户请求安全终止任务")
            next_status = (
                "cancelled" if str(row.get("status") or "") == "queued"
                else "running"
            )
            next_metadata = {
                **existing_metadata,
                "cancel_requested": True,
                "cancel_requested_at": requested_at,
                "cancel_reason": cancellation_reason,
                "stage": (
                    "cancelled" if next_status == "cancelled"
                    else "cancelling"
                ),
            }
            if next_status == "cancelled":
                next_metadata["cancel_acknowledged_at"] = datetime.now(
                    timezone.utc,
                ).isoformat()
            logs = list(row.get("log") or [])
            if not already_requested:
                logs.append(
                    cancellation_reason
                    + (
                        "；排队任务已取消"
                        if next_status == "cancelled"
                        else "；正在等待当前安全步骤结束"
                    )
                )
            updated = conn.execute(
                """UPDATE jobs SET status=%s,progress=%s,log=%s::jsonb,
                       metadata=%s::jsonb,error_message=NULL,updated_at=NOW(),
                       completed_at=CASE WHEN %s='cancelled' THEN NOW() ELSE completed_at END
                   WHERE id=%s RETURNING *""",
                (
                    next_status,
                    100 if next_status == "cancelled" else row.get("progress"),
                    self._json(logs), self._json(next_metadata), next_status,
                    str(job_id),
                ),
            ).fetchone()
            if next_status == "cancelled" and str(row.get("kind") or "") == "graph_sync":
                conn.execute(
                    """UPDATE projects SET graph_status=%s,graph_policy_version='',
                       graph_policy_hash='',graph_error=%s,updated_at=NOW()
                       WHERE id=%s AND graph_status=%s""",
                    (
                        GRAPH_STALE,
                        "图谱构建已终止，未完成结果未发布",
                        project_id,
                        GRAPH_REBUILDING,
                    ),
                )
            if next_status == "cancelled" and str(row.get("kind") or "") == "ingestion":
                # Submission marks every target strategy row queued before
                # handing the job to the executor.  A queued cancellation has
                # no worker cleanup path, so close those exact rows here.  The
                # latest-job guard prevents an old request from touching a
                # later retry.
                target_ids = [
                    int(value)
                    for value in (
                        existing_metadata.get("target_document_ids")
                        or [row.get("document_id")]
                    )
                    if value is not None
                ]
                if target_ids:
                    conn.execute(
                        """UPDATE document_strategy_builds
                           SET status='cancelled',error_message=%s,
                               completed_at=NOW(),updated_at=NOW()
                           WHERE project_id=%s AND strategy_key=%s
                             AND document_id=ANY(%s) AND latest_job_id=%s
                             AND status='queued'""",
                        (
                            cancellation_reason,
                            int(project_id),
                            str(row.get("strategy_key") or STABLE_STRATEGY),
                            target_ids,
                            str(job_id),
                        ),
                    )
            return updated

    def list_ingestion_batch_jobs(
        self, batch_id: str, strategy_key: str = "",
    ) -> list[dict[str, Any]]:
        """Return every document job submitted by one multi-file upload."""

        params: list[Any] = [batch_id]
        condition = (
            "j.kind IN ('ingestion','content_preparation') "
            "AND j.metadata->>'batch_id'=%s"
        )
        if strategy_key:
            condition += " AND j.strategy_key=%s"
            params.append(normalize_strategy(strategy_key))
        with self.connection() as conn:
            return conn.execute(
                """SELECT j.*,d.original_filename AS document_name
                   FROM jobs j LEFT JOIN documents d ON d.id=j.document_id
                   WHERE """ + condition + " ORDER BY j.created_at,j.id",
                params,
            ).fetchall()

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
            report_job_ids = [
                str(row["id"]) for row in rows
                if str((row.get("metadata") or {}).get("report_run_id") or "")
            ]
            if report_job_ids:
                conn.execute(
                    """UPDATE knowledge_forge_community_reports report
                       SET status='cancelled',error_message=%s,updated_at=NOW(),
                           completed_at=NOW()
                       FROM knowledge_forge_community_report_runs run
                       WHERE report.run_id=run.run_id AND run.job_id=ANY(%s)
                         AND report.status IN ('pending','running')""",
                    (str(reason)[-4000:], report_job_ids),
                )
                conn.execute(
                    """UPDATE knowledge_forge_community_report_runs
                       SET status='cancelled',error_message=%s,completed_at=NOW()
                       WHERE job_id=ANY(%s) AND status IN ('queued','running')""",
                    (str(reason)[-4000:], report_job_ids),
                )
            return len(rows)

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

    def get_grounded_entity_namespace_for_snapshot(
        self, snapshot_id: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM entity_shadow_namespaces
                   WHERE snapshot_id=%s ORDER BY created_at DESC LIMIT 1""",
                (snapshot_id,),
            ).fetchone()

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

    def latest_job(
        self, project_id: int, kind: str = "", strategy_key: str = "",
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM jobs WHERE project_id=%s"
        params: list[Any] = [project_id]
        if kind:
            query += " AND kind=%s"
            params.append(kind)
        if strategy_key:
            query += " AND strategy_key=%s"
            params.append(normalize_strategy(strategy_key))
        with self.connection() as conn:
            return conn.execute(query + " ORDER BY updated_at DESC LIMIT 1", params).fetchone()

    def latest_active_job(
        self, project_id: int, kind: str = "", strategy_key: str = "",
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM jobs WHERE project_id=%s AND status IN ('queued','running')"
        params: list[Any] = [project_id]
        if kind:
            query += " AND kind=%s"
            params.append(kind)
        if strategy_key:
            query += " AND strategy_key=%s"
            params.append(normalize_strategy(strategy_key))
        query += " ORDER BY CASE WHEN status='running' THEN 0 ELSE 1 END, updated_at DESC LIMIT 1"
        with self.connection() as conn:
            return conn.execute(query, params).fetchone()

    def latest_active_mutating_job(self, project_id: int) -> dict[str, Any] | None:
        """Return active project work other than read-only enhancement jobs."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM jobs
                   WHERE project_id=%s AND status IN ('queued','running')
                     AND kind<>'community_report_build'
                   ORDER BY CASE WHEN status='running' THEN 0 ELSE 1 END,
                            updated_at DESC LIMIT 1""",
                (int(project_id),),
            ).fetchone()

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
            # Logical Schema is an optional normalization layer, but a graph
            # that claims to be ready must still identify the exact version it
            # used.  Invalidate the Knowledge Forge snapshot atomically with Schema
            # publication so no request can observe a new Schema paired with
            # an old ready graph.  A graph already stale is not advanced again;
            # an empty graph has no derived facts to invalidate.
            conn.execute(
                """UPDATE project_strategy_graphs
                   SET generation=CASE
                           WHEN status IN ('ready','rebuilding') THEN generation+1
                           ELSE generation
                       END,
                       status=CASE WHEN status='empty' THEN 'empty' ELSE 'stale' END,
                       error_message=CASE
                           WHEN status='empty' THEN error_message
                           ELSE '逻辑 Schema 已变化，Knowledge Forge 需要重新构建'
                       END,
                       lease_token=CASE WHEN status='empty' THEN lease_token ELSE '' END,
                       lease_expires_at=CASE
                           WHEN status='empty' THEN lease_expires_at ELSE NULL
                       END,
                       policy_summary=policy_summary || jsonb_build_object(
                           'invalidated_by','logical_schema',
                           'invalidated_schema_version',%s::BIGINT
                       ),
                       updated_at=NOW()
                   WHERE project_id=%s AND strategy_key=%s""",
                (version, project_id, KNOWLEDGE_FORGE_STRATEGY),
            )
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

    def record_schema_evaluation(
        self,
        project_id: int,
        document_id: int,
        *,
        prior_schema_version: int,
        resulting_schema_version: int,
        outcome: str,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one adaptive-schema decision even when it creates no version."""

        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO schema_evaluation_runs(
                   project_id,document_id,prior_schema_version,
                   resulting_schema_version,outcome,evaluation)
                   VALUES(%s,%s,%s,%s,%s,%s::jsonb) RETURNING *""",
                (
                    project_id, document_id, prior_schema_version,
                    resulting_schema_version, outcome, self._json(evaluation),
                ),
            ).fetchone()

    def list_schema_evaluations(
        self, project_id: int, *, document_id: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM schema_evaluation_runs WHERE project_id=%s"
        params: list[Any] = [project_id]
        if document_id is not None:
            query += " AND document_id=%s"
            params.append(document_id)
        with self.connection() as conn:
            return conn.execute(query + " ORDER BY created_at DESC,id DESC", params).fetchall()

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
                """INSERT INTO entity_mentions(project_id,document_id,revision_id,chunk_id,
                canonical_entity_id,mention_key,mention_text,normalized_name,entity_type,
                description,confidence,extraction_version,evidence_text,evidence_start,evidence_end,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(revision_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE
                DO UPDATE SET canonical_entity_id=EXCLUDED.canonical_entity_id,
                description=EXCLUDED.description,
                confidence=EXCLUDED.confidence,evidence_text=EXCLUDED.evidence_text,evidence_start=EXCLUDED.evidence_start,
                evidence_end=EXCLUDED.evidence_end,metadata=EXCLUDED.metadata,is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (values["project_id"],values["document_id"],values["revision_id"],values["chunk_id"],
                 values["canonical_entity_id"],mention_key,
                 values["mention_text"],values["normalized_name"],values["entity_type"],
                 values.get("description") or "",0.5 if values.get("confidence") is None
                 else values.get("confidence"),values.get("extraction_version") or "v2",
                 values.get("evidence_text") or "",values.get("evidence_start"),
                 values.get("evidence_end"),self._json(values.get("metadata"))),
            ).fetchone()
            conn.execute(
                """INSERT INTO canonical_entity_documents(
                       project_id,canonical_entity_id,document_id,revision_id)
                   VALUES(%s,%s,%s,%s)
                   ON CONFLICT(revision_id,canonical_entity_id) DO UPDATE
                   SET project_id=EXCLUDED.project_id,is_deleted=FALSE,updated_at=NOW()""",
                (
                    values["project_id"], values["canonical_entity_id"],
                    values["document_id"], values["revision_id"],
                ),
            )
            self._refresh_entity_source_counts(
                conn, int(values["project_id"]), [int(values["canonical_entity_id"])],
            )
            return row

    def add_entity_resolution_decision(self, values: dict[str, Any]) -> dict[str, Any]:
        mention_key = self._mention_conflict_key(values)
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO entity_resolution_decisions(
                project_id,document_id,revision_id,chunk_id,
                canonical_entity_id,mention_key,mention_text,normalized_name,entity_type,
                decision,candidate_entity_ids,scores,reason,model_name,metadata)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                ON CONFLICT(revision_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE
                DO UPDATE SET
                canonical_entity_id=EXCLUDED.canonical_entity_id,decision=EXCLUDED.decision,
                candidate_entity_ids=EXCLUDED.candidate_entity_ids,scores=EXCLUDED.scores,
                reason=EXCLUDED.reason,model_name=EXCLUDED.model_name,metadata=EXCLUDED.metadata,
                is_deleted=FALSE,updated_at=NOW() RETURNING *""",
                (
                    values["project_id"], values["document_id"], values["revision_id"],
                    values["chunk_id"],
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
        revision_context = self._revision_context.get() or {}
        context_revision_id = str(revision_context.get("revision_id") or "")
        mentions = [
            {
                **item,
                "revision_id": str(item.get("revision_id") or context_revision_id),
            }
            for item in mentions
        ]
        decisions = [
            {
                **item,
                "revision_id": str(item.get("revision_id") or context_revision_id),
            }
            for item in decisions
        ]
        revision_ids = {
            str(item.get("revision_id") or "") for item in [*mentions, *decisions]
        }
        if "" in revision_ids or len(revision_ids) != 1:
            raise ValueError("实体批次必须绑定同一内容修订")
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
                    """INSERT INTO entity_mentions(
                    project_id,document_id,revision_id,chunk_id,canonical_entity_id,
                    mention_text,normalized_name,entity_type,description,confidence,extraction_version,
                    evidence_text,evidence_start,evidence_end,mention_key,extraction_policy_version,metadata)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
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
                        values["project_id"], values["document_id"], values["revision_id"],
                        values["chunk_id"],
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
                        project_id,document_id,revision_id,chunk_id,mention_id,mention_key,entity_type,
                       confidence,evidence_text,evidence_start,evidence_end,policy_version,source,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'model',%s::jsonb)
                       ON CONFLICT(revision_id,mention_key,entity_type,policy_version,source)
                       DO UPDATE SET confidence=EXCLUDED.confidence,evidence_text=EXCLUDED.evidence_text,
                       evidence_start=EXCLUDED.evidence_start,evidence_end=EXCLUDED.evidence_end,
                       metadata=EXCLUDED.metadata""",
                    (
                        project_id, document_id, values["revision_id"], values["chunk_id"],
                        row["id"], mention_key,
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
                    """INSERT INTO entity_resolution_decisions(
                    project_id,document_id,revision_id,chunk_id,
                    canonical_entity_id,mention_text,normalized_name,entity_type,mention_key,decision,
                    candidate_entity_ids,scores,reason,model_name,metadata)
                    VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT(revision_id,mention_key) WHERE mention_key<>'' AND is_deleted=FALSE
                    DO UPDATE SET
                    canonical_entity_id=EXCLUDED.canonical_entity_id,decision=EXCLUDED.decision,
                    candidate_entity_ids=EXCLUDED.candidate_entity_ids,scores=EXCLUDED.scores,
                    reason=EXCLUDED.reason,model_name=EXCLUDED.model_name,metadata=EXCLUDED.metadata,
                    is_deleted=FALSE,updated_at=NOW()""",
                    (
                        values["project_id"], values["document_id"], values["revision_id"],
                        values["chunk_id"],
                        values["canonical_entity_id"], values["mention_text"], values["normalized_name"],
                        values["entity_type"], values.get("mention_key") or self._stable_mention_key(values),
                        values["decision"], values.get("candidate_entity_ids") or [],
                        self._json(values.get("scores")), values.get("reason") or "",
                        values.get("model_name"), self._json(values.get("metadata")),
                    ),
                )
            affected_ids = sorted({int(item["canonical_entity_id"]) for item in mentions})
            conn.execute(
                """INSERT INTO canonical_entity_documents(
                       project_id,canonical_entity_id,document_id,revision_id)
                   SELECT %s,entity_id,%s,%s FROM unnest(%s::bigint[]) AS entity_id
                   ON CONFLICT(revision_id,canonical_entity_id) DO UPDATE
                   SET project_id=EXCLUDED.project_id,is_deleted=FALSE,updated_at=NOW()""",
                (project_id, document_id, mentions[0]["revision_id"], affected_ids),
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

    def start_relation_pipeline_run(self, values: dict[str, Any]) -> dict[str, Any]:
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
                raise ValueError("relation pipeline run input fingerprint mismatch")
            if not existing:
                raise RuntimeError("relation pipeline run could not be started")
            return existing

    def prepare_relation_pipeline_work(
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
                raise LookupError("关系流水线运行不存在")
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
                    raise ValueError("relation work item fingerprint mismatch")
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

    def lease_relation_pipeline_work(
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

    def settle_relation_pipeline_work(
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
                raise LookupError("关系工作项不存在")
            attempt = conn.execute(
                """SELECT * FROM relation_shadow_attempts
                   WHERE work_item_id=%s AND lease_token=%s FOR UPDATE""",
                (work_item_id, lease_token),
            ).fetchone()
            if not attempt:
                raise ValueError("关系处理 attempt lease token 不存在")
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

    def list_relation_pipeline_work(self, run_id: str) -> list[dict[str, Any]]:
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

    def finish_relation_pipeline_run(
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
            raise ValueError(f"非法关系运行状态: {status}")
        candidate_map = {str(item["candidate_id"]): item for item in candidates}
        if any(str(item.get("candidate_id") or "") not in candidate_map for item in decisions):
            raise ValueError("关系 decision 引用了未知 candidate")
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
                        candidate["target_mention_id"],
                        list(candidate.get("allowed_predicates") or []),
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
                raise ValueError("关系流水线仍有未完成 work item，不能标记 completed")
            if status == "completed" and int(work_counts["failed"] or 0) > 0:
                raise ValueError("关系流水线存在 failed work item，不能标记 completed")
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
                raise LookupError("关系运行不存在")
            return row

    @staticmethod
    def _grounded_relation_db_id(prefix: str, *parts: object) -> str:
        material = "\x1f".join(str(part) for part in parts)
        return prefix + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _set_grounded_relation_session(
        conn: psycopg.Connection,
        run_id: str,
    ) -> None:
        # The database triggers use this transaction-local marker to turn an
        # accidental product-table write into a hard error.
        conn.execute(
            "SELECT set_config('knowledge_forge.shadow_relation_v2_run_id',%s,TRUE)",
            (run_id,),
        )

    def start_grounded_relation_run(
        self,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Create one isolated v2 run without changing any active product state."""

        run_id = str(values.get("id") or "")
        fingerprint = str(values.get("input_fingerprint") or "")
        components = dict(values.get("fingerprint_components") or {})
        if not run_id or not fingerprint or not components:
            raise ValueError("Relation run requires an id and complete fingerprint")
        project_id = int(values["project_id"])
        source_generation = int(values.get("source_generation") or 0)
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            conn.execute(
                """INSERT INTO shadow_relation_v2_runs(
                   id,project_id,snapshot_id,input_fingerprint,fingerprint_components,
                   source_generation,status,audit_quality)
                   VALUES(%s,%s,%s,%s,%s::jsonb,%s,'planning','complete')
                   ON CONFLICT(id) DO NOTHING""",
                (
                    run_id,
                    project_id,
                    values.get("snapshot_id"),
                    fingerprint,
                    self._canonical_json(components),
                    source_generation,
                ),
            )
            row = conn.execute(
                "SELECT * FROM shadow_relation_v2_runs WHERE id=%s",
                (run_id,),
            ).fetchone()
            if not row:
                raise RuntimeError("Relation run could not be created")
            if (
                int(row["project_id"]) != project_id
                or str(row["input_fingerprint"]) != fingerprint
                or int(row["source_generation"]) != source_generation
                or dict(row.get("fingerprint_components") or {}) != components
            ):
                raise ValueError("Relation run immutable input mismatch")
            if str(row.get("status") or "") in {"failed", "cancelled", "stale"}:
                self._reset_grounded_relation_run_attempt(conn, run_id)
                row = conn.execute(
                    """UPDATE shadow_relation_v2_runs
                       SET status='planning',metrics='{}'::jsonb,error_message='',
                           completed_at=NULL
                       WHERE id=%s RETURNING *""",
                    (run_id,),
                ).fetchone()
            return row

    def persist_grounded_relation_source_contexts(
        self,
        run_id: str,
        contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist relation SourceUnit/ProposalWindow context audit rows.

        This is an isolated relation ledger. It records the rendered
        relation context and source-map rows used by the v2 runtime; it never
        mutates product chunks, stable entities or formal graph state.
        """

        values = list(contexts)
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            run = conn.execute(
                "SELECT * FROM shadow_relation_v2_runs WHERE id=%s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("Relation run does not exist")
            persisted: list[dict[str, Any]] = []
            for item in values:
                context_identity = str(item.get("context_identity") or "")
                if not context_identity:
                    raise ValueError("Shadow source context requires identity")
                project_id = int(item.get("project_id") or run["project_id"])
                if project_id != int(run["project_id"]):
                    raise ValueError("Shadow source context project mismatch")
                context_id = self._grounded_relation_db_id(
                    "srcctx_", run_id, context_identity,
                )
                payload = dict(item.get("payload") or {})
                source_maps = list(item.get("source_maps") or [])
                conn.execute(
                    """INSERT INTO shadow_source_contexts(
                       id,run_id,project_id,document_id,document_revision_id,
                       chunk_id,chunk_key,context_kind,context_identity,
                       context_start,context_end,content_hash,
                       structure_artifact_hash,parser_version,payload)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(run_id,context_identity) DO UPDATE SET
                         payload=EXCLUDED.payload,
                         context_kind=EXCLUDED.context_kind,
                         context_start=EXCLUDED.context_start,
                         context_end=EXCLUDED.context_end,
                         content_hash=EXCLUDED.content_hash,
                         structure_artifact_hash=EXCLUDED.structure_artifact_hash,
                         parser_version=EXCLUDED.parser_version
                       RETURNING *""",
                    (
                        context_id,
                        run_id,
                        project_id,
                        item.get("document_id") or None,
                        str(item.get("document_revision_id") or ""),
                        item.get("chunk_id") or None,
                        str(item.get("chunk_key") or context_identity),
                        str(item.get("context_kind") or "legacy_chunk"),
                        context_identity,
                        int(item.get("context_start") or 0),
                        int(item.get("context_end") or 0),
                        str(item.get("content_hash") or ""),
                        str(item.get("structure_artifact_hash") or ""),
                        str(item.get("parser_version") or ""),
                        self._canonical_json(payload),
                    ),
                )
                for source_map in source_maps:
                    source_kind = str(
                        source_map.get("anchor_kind")
                        or source_map.get("source_kind")
                        or "legacy_chunk_local"
                    )
                    source_record_id = str(
                        source_map.get("block_key")
                        or source_map.get("source_record_id")
                        or context_identity
                    )
                    source_start = int(
                        source_map.get("block_local_start")
                        if source_map.get("block_local_start") is not None
                        else source_map.get("source_start") or 0
                    )
                    source_end = int(
                        source_map.get("block_local_end")
                        if source_map.get("block_local_end") is not None
                        else source_map.get("source_end") or 0
                    )
                    payload_json = self._canonical_json(source_map)
                    payload_hash = hashlib.sha256(
                        payload_json.encode("utf-8")
                    ).hexdigest()
                    source_map_id = self._grounded_relation_db_id(
                        "srcmap_",
                        run_id,
                        context_identity,
                        source_kind,
                        source_record_id,
                        source_start,
                        source_end,
                    )
                    conn.execute(
                        """INSERT INTO shadow_context_source_maps(
                           id,context_id,source_kind,source_record_id,
                           source_start,source_end,payload_hash,payload)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                           ON CONFLICT(
                             context_id,source_kind,source_record_id,
                             source_start,source_end
                           ) DO UPDATE SET
                             payload_hash=EXCLUDED.payload_hash,
                             payload=EXCLUDED.payload""",
                        (
                            source_map_id,
                            context_id,
                            source_kind,
                            source_record_id,
                            source_start,
                            source_end,
                            payload_hash,
                            payload_json,
                        ),
                    )
                persisted.append({
                    "id": context_id,
                    "context_identity": context_identity,
                    "source_maps": len(source_maps),
                })
            return persisted

    def _reset_grounded_relation_run_attempt(self, conn: Any, run_id: str) -> None:
        """Discard one failed relation attempt before a same-input retry.

        The run id is deterministic over the immutable input fingerprint.  A
        crashed or failed worker can therefore be retried with exactly the same
        id.  Before doing that, remove half-written isolated ledgers for this
        run only; never touch product tables or other runs' normalized facts.
        """

        pair_ids = (
            "SELECT id FROM shadow_relation_pair_opportunities WHERE run_id=%s"
        )
        hypothesis_ids = (
            "SELECT hypothesis.id FROM shadow_relation_hypotheses hypothesis "
            "JOIN shadow_relation_pair_opportunities pair "
            "ON pair.id=hypothesis.pair_opportunity_id WHERE pair.run_id=%s"
        )
        context_ids = "SELECT id FROM shadow_source_contexts WHERE run_id=%s"
        proposal_attempt_ids = (
            "SELECT id FROM shadow_relation_proposal_attempts WHERE run_id=%s"
        )
        delete_statements = [
            ("DELETE FROM shadow_relation_evidence_refs WHERE run_id=%s", (run_id,)),
            (
                "DELETE FROM shadow_relation_fact_collapse_audit WHERE run_id=%s",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_fact_conflicts WHERE run_id=%s",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_fact_evidence WHERE run_id=%s",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_fact_memberships WHERE run_id=%s",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_proposals "
                f"WHERE proposal_attempt_id IN ({proposal_attempt_ids}) "
                f"OR pair_opportunity_id IN ({pair_ids}) "
                f"OR hypothesis_id IN ({hypothesis_ids})",
                (run_id, run_id, run_id),
            ),
            (
                "DELETE FROM shadow_relation_proposal_attempts WHERE run_id=%s",
                (run_id,),
            ),
            ("DELETE FROM shadow_model_call_members WHERE run_id=%s", (run_id,)),
            (
                "DELETE FROM shadow_relation_verification_attempts "
                f"WHERE hypothesis_id IN ({hypothesis_ids})",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_hypothesis_validations "
                f"WHERE hypothesis_id IN ({hypothesis_ids})",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_classification_attempts "
                f"WHERE pair_opportunity_id IN ({pair_ids})",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_candidate_sources WHERE run_id=%s",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_pair_contexts "
                f"WHERE pair_opportunity_id IN ({pair_ids})",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_hypotheses "
                f"WHERE pair_opportunity_id IN ({pair_ids})",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_relation_pair_opportunities WHERE run_id=%s",
                (run_id,),
            ),
            (
                "DELETE FROM shadow_context_source_maps "
                f"WHERE context_id IN ({context_ids})",
                (run_id,),
            ),
            ("DELETE FROM shadow_source_contexts WHERE run_id=%s", (run_id,)),
            ("DELETE FROM shadow_model_calls WHERE run_id=%s", (run_id,)),
        ]
        for sql, params in delete_statements:
            conn.execute(sql, params)

    def persist_grounded_relation_opportunities(
        self,
        run_id: str,
        opportunities: Any,
        *,
        mark_classifying: bool = True,
        include_persisted_rows: bool = True,
    ) -> list[dict[str, Any]]:
        """Persist an immutable opportunity ledger batch before model I/O.

        The first version of this ledger performed five or more client/server
        round trips for every candidate.  Dense documents can legitimately
        have thousands of candidates, turning provenance persistence into a
        much longer step than relation inference.  Build each independent
        table payload in memory, insert it set-wise, then retain the same
        immutable pair-payload audit in one bounded read.

        ``mark_classifying`` is deliberately controlled by the caller when a
        full ledger is submitted in several batches.  A partly written ledger
        is never a publishable run and must not appear classified merely
        because its first batch committed.
        """

        values = list(opportunities)
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            run = conn.execute(
                "SELECT * FROM shadow_relation_v2_runs WHERE id=%s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("Relation run does not exist")
            source_context_identities = sorted({
                str(item.provenance.get("source_context_id") or "")
                for item in values
                if (
                    isinstance(item.provenance, dict)
                    and str(item.provenance.get("source_context_id") or "")
                )
            })
            context_id_by_identity: dict[str, str] = {}
            if source_context_identities:
                source_context_rows = conn.execute(
                    """SELECT id,context_identity FROM shadow_source_contexts
                       WHERE run_id=%s AND context_identity=ANY(%s)""",
                    (run_id, source_context_identities),
                ).fetchall()
                context_id_by_identity = {
                    str(row["context_identity"]): str(row["id"])
                    for row in source_context_rows
                }
            pair_rows: list[dict[str, Any]] = []
            pair_context_rows: list[dict[str, Any]] = []
            candidate_source_rows: list[dict[str, Any]] = []
            expected_by_id: dict[str, tuple[Any, ...]] = {}
            for item in values:
                if str(item.run_id) != run_id:
                    raise ValueError("pair opportunity belongs to another run")
                if int(item.project_id) != int(run["project_id"]):
                    raise ValueError("pair opportunity belongs to another project")
                provenance = (
                    item.provenance if isinstance(item.provenance, dict) else {}
                )
                source_kind = str(
                    provenance.get("source_kind") or "v2_pair_planner"
                )
                source_record_id = str(
                    provenance.get("source_record_id")
                    or item.context_identity
                )
                source_context_identity = str(
                    provenance.get("source_context_id") or ""
                )
                source_context_id = context_id_by_identity.get(
                    source_context_identity, "",
                )
                expected = (
                    str(item.pair_identity),
                    str(item.candidate_policy_version),
                    str(item.left_occurrence_key),
                    str(item.right_occurrence_key),
                    int(item.left_span[0]),
                    int(item.left_span[1]),
                    int(item.right_span[0]),
                    int(item.right_span[1]),
                    str(item.context_identity),
                    int(item.context_span[0]),
                    int(item.context_span[1]),
                )
                previous_expected = expected_by_id.get(
                    str(item.opportunity_identity),
                )
                if previous_expected is not None and previous_expected != expected:
                    raise ValueError("pair opportunity immutable input mismatch")
                expected_by_id[str(item.opportunity_identity)] = expected
                pair_rows.append({
                    "id": str(item.opportunity_identity),
                    "run_id": run_id,
                    "project_id": int(item.project_id),
                    "document_revision_id": str(item.document_revision_id),
                    "document_id": provenance.get("document_id"),
                    "chunk_id": provenance.get("chunk_id"),
                    "chunk_key": str(item.chunk_key),
                    "pair_identity": str(item.pair_identity),
                    "candidate_policy_version": str(
                        item.candidate_policy_version,
                    ),
                    "left_occurrence_key": str(item.left_occurrence_key),
                    "right_occurrence_key": str(item.right_occurrence_key),
                    "left_mention_id": str(item.left_mention_id),
                    "right_mention_id": str(item.right_mention_id),
                    "left_start": int(item.left_span[0]),
                    "left_end": int(item.left_span[1]),
                    "right_start": int(item.right_span[0]),
                    "right_end": int(item.right_span[1]),
                    "left_canonical_id": str(item.left_canonical_id),
                    "right_canonical_id": str(item.right_canonical_id),
                    "left_assignment_version": str(item.left_assignment_version),
                    "right_assignment_version": str(item.right_assignment_version),
                    "context_identity": str(item.context_identity),
                    "context_start": int(item.context_span[0]),
                    "context_end": int(item.context_span[1]),
                    "allowed_predicate_metadata": list(
                        item.allowed_predicate_metadata,
                    ),
                    "planning_reason": str(item.planning_reason),
                    "cue": str(item.cue),
                    "type_status": str(item.type_status),
                    "status": str(item.status),
                    "provenance": provenance,
                    "context_id": source_context_id,
                    "source_kind": source_kind,
                    "source_record_id": source_record_id,
                })
                if source_context_identity and source_context_id:
                    pair_context_rows.append({
                        "pair_opportunity_id": str(item.opportunity_identity),
                        "context_id": source_context_id,
                        "context_role": "classification_context",
                        "payload": {
                            "context_identity": str(item.context_identity),
                            "context_span": [
                                int(item.context_span[0]),
                                int(item.context_span[1]),
                            ],
                            "candidate_sources": list(
                                provenance.get("candidate_sources") or [],
                            ),
                            "planning_reason": str(item.planning_reason),
                        },
                    })
                candidate_sources = [
                    str(value)
                    for value in provenance.get("candidate_sources", [])
                    if str(value)
                ]
                if not candidate_sources:
                    candidate_sources = ["PAIR_PLANNER"]
                for rank, candidate_source in enumerate(
                    sorted(set(candidate_sources)), start=1,
                ):
                    candidate_source_rows.append({
                        "id": self._grounded_relation_db_id(
                            "relcandsrc_",
                            run_id,
                            item.opportunity_identity,
                            candidate_source,
                            source_kind,
                            source_record_id,
                        ),
                        "run_id": run_id,
                        "pair_opportunity_id": str(item.opportunity_identity),
                        "candidate_source": candidate_source,
                        "source_kind": source_kind,
                        "source_record_id": source_record_id,
                        "candidate_rank": rank,
                        "admitted": bool(
                            item.status == "PENDING_CLASSIFICATION",
                        ),
                        "terminal_status": str(item.status),
                        "metadata": {
                            "planning_reason": str(item.planning_reason),
                            "cue": str(item.cue),
                            "type_status": str(item.type_status),
                        },
                    })
            if pair_rows:
                conn.execute(
                    """INSERT INTO shadow_relation_pair_opportunities(
                       id,run_id,project_id,document_revision_id,document_id,chunk_id,
                       chunk_key,pair_identity,candidate_policy_version,
                       left_occurrence_key,right_occurrence_key,left_mention_id,right_mention_id,
                       left_start,left_end,right_start,right_end,left_canonical_id,
                       right_canonical_id,left_assignment_version,right_assignment_version,
                       context_identity,context_start,context_end,allowed_predicate_metadata,
                       planning_reason,cue,type_status,status,provenance,
                       context_id,source_kind,source_record_id)
                       SELECT item.id,item.run_id,item.project_id,
                              item.document_revision_id,item.document_id,item.chunk_id,
                              item.chunk_key,item.pair_identity,
                              item.candidate_policy_version,
                              item.left_occurrence_key,item.right_occurrence_key,
                              item.left_mention_id,item.right_mention_id,
                              item.left_start,item.left_end,item.right_start,item.right_end,
                              item.left_canonical_id,item.right_canonical_id,
                              item.left_assignment_version,item.right_assignment_version,
                              item.context_identity,item.context_start,item.context_end,
                              item.allowed_predicate_metadata,item.planning_reason,
                              item.cue,item.type_status,item.status,item.provenance,
                              item.context_id,item.source_kind,item.source_record_id
                         FROM jsonb_to_recordset(%s::jsonb) AS item(
                              id TEXT,run_id TEXT,project_id BIGINT,
                              document_revision_id TEXT,document_id BIGINT,chunk_id BIGINT,
                              chunk_key TEXT,pair_identity TEXT,
                              candidate_policy_version TEXT,
                              left_occurrence_key TEXT,right_occurrence_key TEXT,
                              left_mention_id TEXT,right_mention_id TEXT,
                              left_start INTEGER,left_end INTEGER,
                              right_start INTEGER,right_end INTEGER,
                              left_canonical_id TEXT,right_canonical_id TEXT,
                              left_assignment_version TEXT,right_assignment_version TEXT,
                              context_identity TEXT,context_start INTEGER,context_end INTEGER,
                              allowed_predicate_metadata JSONB,planning_reason TEXT,
                              cue TEXT,type_status TEXT,status TEXT,provenance JSONB,
                              context_id TEXT,source_kind TEXT,source_record_id TEXT
                         )
                       ON CONFLICT(id) DO NOTHING""",
                    (self._canonical_json(pair_rows),),
                )
                persisted_rows = conn.execute(
                    """SELECT id,pair_identity,candidate_policy_version,
                              left_occurrence_key,right_occurrence_key,
                              left_start,left_end,right_start,right_end,
                              context_identity,context_start,context_end
                         FROM shadow_relation_pair_opportunities
                        WHERE run_id=%s AND id=ANY(%s)""",
                    (run_id, list(expected_by_id)),
                ).fetchall()
                persisted_by_id = {
                    str(row["id"]): row for row in persisted_rows
                }
                if len(persisted_by_id) != len(expected_by_id):
                    raise LookupError("persisted pair opportunity is missing")
                for opportunity_id, expected in expected_by_id.items():
                    persisted = persisted_by_id[opportunity_id]
                    actual = (
                        str(persisted["pair_identity"]),
                        str(persisted["candidate_policy_version"]),
                        str(persisted["left_occurrence_key"]),
                        str(persisted["right_occurrence_key"]),
                        int(persisted["left_start"]),
                        int(persisted["left_end"]),
                        int(persisted["right_start"]),
                        int(persisted["right_end"]),
                        str(persisted["context_identity"]),
                        int(persisted["context_start"]),
                        int(persisted["context_end"]),
                    )
                    if actual != expected:
                        raise ValueError("pair opportunity immutable input mismatch")
            if pair_context_rows:
                conn.execute(
                    """INSERT INTO shadow_relation_pair_contexts(
                           pair_opportunity_id,context_id,context_role,payload)
                       SELECT item.pair_opportunity_id,item.context_id,
                              item.context_role,item.payload
                         FROM jsonb_to_recordset(%s::jsonb) AS item(
                              pair_opportunity_id TEXT,context_id TEXT,
                              context_role TEXT,payload JSONB
                         )
                       ON CONFLICT(pair_opportunity_id,context_id,context_role)
                       DO NOTHING""",
                    (self._canonical_json(pair_context_rows),),
                )
            if candidate_source_rows:
                conn.execute(
                    """INSERT INTO shadow_relation_candidate_sources(
                           id,run_id,pair_opportunity_id,candidate_source,
                           source_kind,source_record_id,candidate_rank,admitted,
                           terminal_status,metadata)
                       SELECT item.id,item.run_id,item.pair_opportunity_id,
                              item.candidate_source,item.source_kind,
                              item.source_record_id,item.candidate_rank,
                              item.admitted,item.terminal_status,item.metadata
                         FROM jsonb_to_recordset(%s::jsonb) AS item(
                              id TEXT,run_id TEXT,pair_opportunity_id TEXT,
                              candidate_source TEXT,source_kind TEXT,
                              source_record_id TEXT,candidate_rank INTEGER,
                              admitted BOOLEAN,terminal_status TEXT,metadata JSONB
                         )
                       ON CONFLICT(
                           pair_opportunity_id,candidate_source,
                           source_kind,source_record_id)
                       DO NOTHING""",
                    (self._canonical_json(candidate_source_rows),),
                )
            if mark_classifying and run["status"] == "planning":
                conn.execute(
                    """UPDATE shadow_relation_v2_runs SET status='classifying'
                       WHERE id=%s AND status='planning'""",
                    (run_id,),
                )
            if not include_persisted_rows:
                return []
            return conn.execute(
                """SELECT * FROM shadow_relation_pair_opportunities
                   WHERE run_id=%s ORDER BY pair_identity""",
                (run_id,),
            ).fetchall()

    def record_grounded_relation_proposal_attempt(
        self,
        run_id: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one direct proposer attempt and its model-call envelope."""

        context_identity = str(values.get("context_identity") or "")
        attempt_no = int(values.get("attempt_no") or 0)
        status = str(values.get("status") or "")
        if not context_identity or attempt_no <= 0 or not status:
            raise ValueError("proposal attempt requires context, attempt and status")
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            context = conn.execute(
                """SELECT * FROM shadow_source_contexts
                   WHERE run_id=%s AND context_identity=%s""",
                (run_id, context_identity),
            ).fetchone()
            if not context:
                raise LookupError("proposal attempt context does not exist")
            model_call_id = str(
                values.get("model_call_id")
                or self._grounded_relation_db_id(
                    "modelcall_", run_id, "proposer", context["id"], attempt_no,
                )
            )
            idempotency_key = f"proposer:{context['id']}:{attempt_no}"
            conn.execute(
                """INSERT INTO shadow_model_calls(
                   id,run_id,component,model_name,model_digest,
                   prompt_version,schema_version,parser_version,
                   idempotency_key,attempt_no,status,request_hash,response_hash,
                   request_payload,raw_response_text,parsed_response,
                   token_usage,error_type,error_message,independent_call)
                   VALUES(%s,%s,'proposer',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                          %s::jsonb,%s,%s::jsonb,%s::jsonb,%s,%s,TRUE)
                   ON CONFLICT(run_id,component,idempotency_key,attempt_no)
                   DO NOTHING""",
                (
                    model_call_id,
                    run_id,
                    str(values.get("model_name") or ""),
                    str(values.get("proposer_model_revision") or ""),
                    str(values.get("proposer_prompt_version") or ""),
                    str(values.get("proposer_schema_version") or ""),
                    str(values.get("proposer_parser_version") or ""),
                    idempotency_key,
                    attempt_no,
                    status,
                    str(values.get("request_hash") or ""),
                    str(values.get("response_hash") or ""),
                    self._canonical_json(values.get("request_payload") or {}),
                    str(values.get("raw_response_text") or ""),
                    self._canonical_json(values.get("parsed_response") or {}),
                    self._canonical_json(values.get("token_usage") or {}),
                    str(values.get("error_type") or ""),
                    str(values.get("error_message") or ""),
                ),
            )
            attempt_id = self._grounded_relation_db_id(
                "relpropattdb_",
                run_id,
                context["id"],
                attempt_no,
                str(values.get("proposer_prompt_version") or ""),
            )
            conn.execute(
                """INSERT INTO shadow_relation_proposal_attempts(
                   id,run_id,context_id,model_call_id,attempt_no,status,
                   proposer_prompt_version,proposer_schema_version,
                   proposer_parser_version,proposer_model_revision,
                   request_hash,response_hash,token_usage,error_type,
                   error_message,metadata)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,
                          %s::jsonb)
                   ON CONFLICT(run_id,context_id,attempt_no,proposer_prompt_version)
                   DO NOTHING""",
                (
                    attempt_id,
                    run_id,
                    context["id"],
                    model_call_id,
                    attempt_no,
                    status,
                    str(values.get("proposer_prompt_version") or ""),
                    str(values.get("proposer_schema_version") or ""),
                    str(values.get("proposer_parser_version") or ""),
                    str(values.get("proposer_model_revision") or ""),
                    str(values.get("request_hash") or ""),
                    str(values.get("response_hash") or ""),
                    self._canonical_json(values.get("token_usage") or {}),
                    str(values.get("error_type") or ""),
                    str(values.get("error_message") or ""),
                    self._canonical_json(values.get("metadata") or {}),
                ),
            )
            return conn.execute(
                "SELECT * FROM shadow_relation_proposal_attempts WHERE id=%s",
                (attempt_id,),
            ).fetchone()

    def record_grounded_relation_classification_attempts(
        self,
        run_id: str,
        attempts: Any,
    ) -> list[dict[str, Any]]:
        """Append a classifier-attempt batch without weakening immutability.

        A classifier response normally covers several pairs.  Each member
        keeps its own durable audit row, but the common model boundary should
        not require one database connection and four statements per member.
        This method validates every member first and then writes both audit
        tables set-wise in one transaction.
        """

        values = [
            (str(pair_identity), dict(payload or {}))
            for pair_identity, payload in attempts
        ]
        supported_statuses = {
            "ACCEPTED",
            "RETRYABLE_EMPTY",
            "RETRYABLE_PROTOCOL",
            "RETRYABLE_TRUNCATED",
            "RETRYABLE_TIMEOUT",
            "RETRYABLE_MODEL_ERROR",
            "TERMINAL_FAILED",
            "SUPERSEDED",
            # Historical audit rows remain readable during compatibility upgrades.
            "accepted",
            "retryable_failed",
            "permanent_failed",
            "superseded",
        }
        pair_attempt_keys: set[tuple[str, int]] = set()
        for pair_identity, payload in values:
            attempt_no = int(payload.get("attempt_no") or 0)
            status = str(payload.get("status") or "")
            if attempt_no <= 0:
                raise ValueError("classification attempt number must be positive")
            if status not in supported_statuses:
                raise ValueError("invalid classification attempt status")
            key = (pair_identity, attempt_no)
            if key in pair_attempt_keys:
                raise ValueError("classification batch repeats pair attempt")
            pair_attempt_keys.add(key)
        if not values:
            return []
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            pair_identities = sorted({pair_identity for pair_identity, _ in values})
            pair_rows = conn.execute(
                """SELECT * FROM shadow_relation_pair_opportunities
                   WHERE run_id=%s AND pair_identity=ANY(%s)""",
                (run_id, pair_identities),
            ).fetchall()
            pairs_by_identity = {
                str(item["pair_identity"]): item for item in pair_rows
            }
            if len(pairs_by_identity) != len(pair_identities):
                raise LookupError("classification attempt pair does not exist")
            model_call_rows: list[dict[str, Any]] = []
            attempt_rows: list[dict[str, Any]] = []
            expected_by_id: dict[str, tuple[Any, ...]] = {}
            output_ids: list[str] = []
            for pair_identity, payload in values:
                pair = pairs_by_identity[pair_identity]
                attempt_no = int(payload["attempt_no"])
                status = str(payload["status"])
                attempt_id = self._grounded_relation_db_id(
                    "relclassdb_", pair["id"], attempt_no,
                )
                model_call_id = str(
                    payload.get("model_call_id")
                    or self._grounded_relation_db_id(
                        "modelcall_", run_id, "classifier", pair["id"],
                        attempt_no,
                    )
                )
                idempotency_key = f"classifier:{pair['id']}:{attempt_no}"
                request_payload = payload.get("request_payload") or {}
                parsed_response = payload.get("parsed_response") or {}
                token_usage = payload.get("token_usage") or {}
                model_call_rows.append({
                    "id": model_call_id,
                    "run_id": str(run_id),
                    "component": "classifier",
                    "model_name": str(payload.get("model_name") or ""),
                    "model_digest": str(
                        payload.get("classifier_model_revision") or ""
                    ),
                    "prompt_version": str(
                        payload.get("classifier_prompt_version") or ""
                    ),
                    "schema_version": str(
                        payload.get("classifier_schema_version") or ""
                    ),
                    "parser_version": str(
                        payload.get("classifier_parser_version") or ""
                    ),
                    "idempotency_key": idempotency_key,
                    "attempt_no": attempt_no,
                    "status": status,
                    "request_hash": str(payload.get("request_hash") or ""),
                    "response_hash": str(payload.get("response_hash") or ""),
                    "request_payload": request_payload,
                    "raw_response_text": str(
                        payload.get("raw_response_text") or ""
                    ),
                    "parsed_response": parsed_response,
                    "token_usage": token_usage,
                    "error_type": str(payload.get("error_type") or ""),
                    "error_message": str(payload.get("error_message") or ""),
                    "independent_call": bool(
                        payload.get("independent_call", True)
                    ),
                })
                attempt_rows.append({
                    "id": attempt_id,
                    "pair_opportunity_id": str(pair["id"]),
                    "attempt_no": attempt_no,
                    "status": status,
                    "request_hash": str(payload.get("request_hash") or ""),
                    "response_hash": str(payload.get("response_hash") or ""),
                    "classifier_prompt_version": str(
                        payload.get("classifier_prompt_version") or ""
                    ),
                    "classifier_schema_version": str(
                        payload.get("classifier_schema_version") or ""
                    ),
                    "classifier_parser_version": str(
                        payload.get("classifier_parser_version") or ""
                    ),
                    "classifier_model_revision": str(
                        payload.get("classifier_model_revision") or ""
                    ),
                    "error_message": str(payload.get("error_message") or ""),
                    "metadata": payload.get("metadata") or {},
                    "model_name": str(payload.get("model_name") or ""),
                    "request_payload": request_payload,
                    "raw_response_text": str(
                        payload.get("raw_response_text") or ""
                    ),
                    "parsed_response": parsed_response,
                    "token_usage": token_usage,
                    "error_type": str(payload.get("error_type") or ""),
                    "independent_call": bool(
                        payload.get("independent_call", True)
                    ),
                    "model_call_id": model_call_id,
                    "audit_quality": "complete",
                    "terminal_status": (
                        status if status in {
                            "ACCEPTED", "TERMINAL_FAILED", "SUPERSEDED",
                            "accepted", "permanent_failed", "superseded",
                        } else ""
                    ),
                })
                expected_by_id[attempt_id] = (
                    attempt_no,
                    status,
                    str(payload.get("request_hash") or ""),
                    str(payload.get("response_hash") or ""),
                    str(payload.get("classifier_prompt_version") or ""),
                    str(payload.get("classifier_schema_version") or ""),
                    str(payload.get("classifier_parser_version") or ""),
                    str(payload.get("classifier_model_revision") or ""),
                    str(payload.get("model_name") or ""),
                    request_payload,
                    str(payload.get("raw_response_text") or ""),
                    parsed_response,
                    token_usage,
                    str(payload.get("error_type") or ""),
                    bool(payload.get("independent_call", True)),
                    model_call_id,
                )
                output_ids.append(attempt_id)
            conn.execute(
                """INSERT INTO shadow_model_calls(
                       id,run_id,component,model_name,model_digest,
                       prompt_version,schema_version,parser_version,
                       idempotency_key,attempt_no,status,request_hash,response_hash,
                       request_payload,raw_response_text,parsed_response,
                       token_usage,error_type,error_message,independent_call)
                   SELECT item.id,item.run_id,item.component,item.model_name,
                          item.model_digest,item.prompt_version,item.schema_version,
                          item.parser_version,item.idempotency_key,item.attempt_no,
                          item.status,item.request_hash,item.response_hash,
                          item.request_payload,item.raw_response_text,
                          item.parsed_response,item.token_usage,item.error_type,
                          item.error_message,item.independent_call
                   FROM jsonb_to_recordset(%s::jsonb) AS item(
                       id TEXT,run_id TEXT,component TEXT,model_name TEXT,
                       model_digest TEXT,prompt_version TEXT,schema_version TEXT,
                       parser_version TEXT,idempotency_key TEXT,attempt_no INTEGER,
                       status TEXT,request_hash TEXT,response_hash TEXT,
                       request_payload JSONB,raw_response_text TEXT,
                       parsed_response JSONB,token_usage JSONB,error_type TEXT,
                       error_message TEXT,independent_call BOOLEAN)
                   ON CONFLICT(run_id,component,idempotency_key,attempt_no)
                   DO NOTHING""",
                (self._json(model_call_rows),),
            )
            conn.execute(
                """INSERT INTO shadow_relation_classification_attempts(
                       id,pair_opportunity_id,attempt_no,status,request_hash,response_hash,
                       classifier_prompt_version,classifier_schema_version,
                       classifier_parser_version,classifier_model_revision,error_message,
                       metadata,model_name,request_payload,raw_response_text,
                       parsed_response,token_usage,error_type,independent_call,
                       model_call_id,audit_quality,terminal_status)
                   SELECT item.id,item.pair_opportunity_id,item.attempt_no,item.status,
                          item.request_hash,item.response_hash,
                          item.classifier_prompt_version,item.classifier_schema_version,
                          item.classifier_parser_version,
                          item.classifier_model_revision,item.error_message,
                          item.metadata,item.model_name,item.request_payload,
                          item.raw_response_text,item.parsed_response,item.token_usage,
                          item.error_type,item.independent_call,item.model_call_id,
                          item.audit_quality,item.terminal_status
                   FROM jsonb_to_recordset(%s::jsonb) AS item(
                       id TEXT,pair_opportunity_id TEXT,attempt_no INTEGER,status TEXT,
                       request_hash TEXT,response_hash TEXT,
                       classifier_prompt_version TEXT,classifier_schema_version TEXT,
                       classifier_parser_version TEXT,classifier_model_revision TEXT,
                       error_message TEXT,metadata JSONB,model_name TEXT,
                       request_payload JSONB,raw_response_text TEXT,
                       parsed_response JSONB,token_usage JSONB,error_type TEXT,
                       independent_call BOOLEAN,model_call_id TEXT,audit_quality TEXT,
                       terminal_status TEXT)
                   ON CONFLICT(id) DO NOTHING""",
                (self._json(attempt_rows),),
            )
            persisted_rows = conn.execute(
                """SELECT * FROM shadow_relation_classification_attempts
                   WHERE id=ANY(%s)""",
                (sorted(expected_by_id),),
            ).fetchall()
            persisted_by_id = {
                str(item["id"]): item for item in persisted_rows
            }
            if len(persisted_by_id) != len(expected_by_id):
                raise ValueError("classification attempt immutable payload mismatch")
            for attempt_id, expected in expected_by_id.items():
                row = persisted_by_id[attempt_id]
                immutable = (
                    int(row["attempt_no"]),
                    str(row["status"]),
                    str(row["request_hash"]),
                    str(row["response_hash"]),
                    str(row["classifier_prompt_version"]),
                    str(row["classifier_schema_version"]),
                    str(row["classifier_parser_version"]),
                    str(row["classifier_model_revision"]),
                    str(row.get("model_name") or ""),
                    row.get("request_payload") or {},
                    str(row.get("raw_response_text") or ""),
                    row.get("parsed_response") or {},
                    row.get("token_usage") or {},
                    str(row.get("error_type") or ""),
                    bool(row.get("independent_call", True)),
                    str(row.get("model_call_id") or ""),
                )
                if immutable != expected:
                    raise ValueError("classification attempt immutable payload mismatch")
            return [persisted_by_id[attempt_id] for attempt_id in output_ids]

    def record_grounded_relation_classification_attempt(
        self,
        run_id: str,
        pair_identity: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one classifier attempt without overwriting an earlier attempt."""
        return self.record_grounded_relation_classification_attempts(
            run_id, [(pair_identity, values)],
        )[0]

    def record_grounded_relation_source_model_call(
        self,
        run_id: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one context/batch-level relation model call."""

        component = str(values.get("component") or "")
        if component not in GROUNDED_RELATION_SOURCE_MODEL_COMPONENTS:
            raise ValueError("invalid source model-call component")
        attempt_no = int(values.get("attempt_no") or 1)
        if attempt_no <= 0:
            raise ValueError("source model-call attempt number must be positive")
        request_hash = str(values.get("request_hash") or "")
        idempotency_key = str(values.get("idempotency_key") or "")
        if not request_hash or not idempotency_key:
            raise ValueError("source model call requires request hash and idempotency key")
        model_call_id = str(
            values.get("id") or self._grounded_relation_db_id(
                "modelcall_", run_id, component, idempotency_key, attempt_no,
            )
        )
        status = str(values.get("status") or "COMPLETED")
        with self.connection() as conn:
            row = conn.execute(
                """INSERT INTO shadow_model_calls(
                   id,run_id,component,model_name,model_digest,prompt_version,
                   schema_version,parser_version,idempotency_key,attempt_no,status,
                   request_hash,response_hash,request_payload,raw_response_text,
                   parsed_response,token_usage,error_type,error_message,
                   independent_call,audit_quality,terminal_status)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                          %s::jsonb,%s,%s::jsonb,%s::jsonb,%s,%s,TRUE,'complete',%s)
                   ON CONFLICT(run_id,component,idempotency_key,attempt_no)
                   DO UPDATE SET terminal_status=EXCLUDED.terminal_status
                   WHERE shadow_model_calls.request_hash=EXCLUDED.request_hash
                   RETURNING *""",
                (
                    model_call_id,
                    str(run_id),
                    component,
                    str(values.get("model_name") or ""),
                    str(values.get("model_digest") or ""),
                    str(values.get("prompt_version") or ""),
                    str(values.get("schema_version") or ""),
                    str(values.get("parser_version") or ""),
                    idempotency_key,
                    attempt_no,
                    status,
                    request_hash,
                    str(values.get("response_hash") or ""),
                    self._canonical_json(values.get("request_payload") or {}),
                    str(values.get("raw_response_text") or ""),
                    self._canonical_json(values.get("parsed_response") or {}),
                    self._canonical_json(values.get("token_usage") or {}),
                    str(values.get("error_type") or ""),
                    str(values.get("error_message") or ""),
                    str(values.get("terminal_status") or status),
                ),
            ).fetchone()
            if not row:
                raise ValueError("source model-call immutable payload mismatch")
            return row

    def find_reusable_relation_model_call(
        self,
        *,
        run_id: str,
        component: str,
        model_digest: str,
        prompt_version: str,
        schema_version: str,
        parser_version: str,
        request_hash: str,
    ) -> dict[str, Any] | None:
        """Return an immutable accepted response from an equivalent prior run.

        Reuse is deliberately same-project and content-addressed at the model
        component boundary.  The model digest, prompt/schema contract
        versions and full rendered request hash must all match.  Parser-only
        upgrades deliberately may reuse the immutable raw JSON: the current
        runtime reparses it under the new parser and falls back to a real model
        call if that stricter parse fails.  A run-wide
        code-tree match is intentionally *not* required: unrelated downstream
        or sibling-component edits must not force every expensive source call
        to run again.  Changed prompts, schemas, parser contracts, generation
        options or source payloads already change one of the exact keys below.
        Failed, incomplete-audit, or non-terminal calls are never candidates.
        """

        component_value = str(component or "")
        if component_value not in GROUNDED_RELATION_REUSABLE_MODEL_COMPONENTS:
            raise ValueError("invalid reusable model-call component")
        if not all((
            str(run_id or ""), str(model_digest or ""),
            str(prompt_version or ""), str(schema_version or ""),
            str(parser_version or ""), str(request_hash or ""),
        )):
            return None
        with self.connection() as conn:
            return conn.execute(
                """SELECT prior.*
                   FROM shadow_model_calls prior
                   JOIN shadow_relation_v2_runs prior_run
                     ON prior_run.id=prior.run_id
                   JOIN shadow_relation_v2_runs current_run
                     ON current_run.id=%s
                   WHERE prior.run_id<>current_run.id
                     AND prior_run.project_id=current_run.project_id
                     AND prior.component=%s
                     AND prior.model_digest=%s
                     AND prior.prompt_version=%s
                     AND prior.schema_version=%s
                     AND prior.request_hash=%s
                     AND prior.error_type=''
                     AND prior.audit_quality='complete'
                     AND prior.terminal_status<>''
                     AND prior.status IN ('COMPLETED','ACCEPTED')
                     AND prior_run.status='ready'
                   ORDER BY prior.created_at DESC, prior.id DESC
                   LIMIT 1""",
                (
                    str(run_id), component_value, str(model_digest),
                    str(prompt_version), str(schema_version),
                    str(request_hash),
                ),
            ).fetchone()

    def persist_grounded_relation_classifications(
        self,
        run_id: str,
        decisions: Any,
    ) -> list[dict[str, Any]]:
        """Persist pair decisions and zero-to-many positive hypotheses.

        Classifier prompts are deliberately small, so a dense context is
        committed through many small decision batches.  Doing a read, attempt
        lookup, update and hypothesis round-trip for *each* member made this
        purely local persistence phase slower than model inference.  Lock and
        validate the whole submitted batch first, then apply decisions and
        hypotheses set-wise while retaining the same immutable retry checks.
        """

        values = list(decisions)
        pair_identity_set: set[str] = set()
        for decision in values:
            pair_identity = str(decision.pair_identity)
            if pair_identity in pair_identity_set:
                raise ValueError("classification batch repeats pair identity")
            pair_identity_set.add(pair_identity)
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            pairs_by_identity: dict[str, dict[str, Any]] = {}
            attempt_outcomes: dict[str, dict[str, bool]] = {}
            if pair_identity_set:
                pair_rows = conn.execute(
                    """SELECT * FROM shadow_relation_pair_opportunities
                       WHERE run_id=%s AND pair_identity=ANY(%s)
                       FOR UPDATE""",
                    (run_id, sorted(pair_identity_set)),
                ).fetchall()
                pairs_by_identity = {
                    str(item["pair_identity"]): item for item in pair_rows
                }
                if len(pairs_by_identity) != len(pair_identity_set):
                    raise LookupError("classification decision pair does not exist")
                pair_ids = [str(item["id"]) for item in pair_rows]
                attempt_rows = conn.execute(
                    """SELECT pair_opportunity_id,
                              bool_or(lower(status)='accepted') AS accepted,
                              bool_or(lower(status)='terminal_failed') AS terminal_failed
                       FROM shadow_relation_classification_attempts
                       WHERE pair_opportunity_id=ANY(%s)
                       GROUP BY pair_opportunity_id""",
                    (pair_ids,),
                ).fetchall()
                attempt_outcomes = {
                    str(item["pair_opportunity_id"]): {
                        "accepted": bool(item.get("accepted")),
                        "terminal_failed": bool(item.get("terminal_failed")),
                    }
                    for item in attempt_rows
                }

            decision_rows: list[dict[str, Any]] = []
            hypothesis_rows: list[dict[str, Any]] = []
            expected_hypotheses: dict[str, tuple[str, str, str, str, list[list[int]]]] = {}
            for decision in values:
                pair_identity = str(decision.pair_identity)
                pair = pairs_by_identity[pair_identity]
                outcome = attempt_outcomes.get(str(pair["id"]), {})
                accepted_attempt = bool(outcome.get("accepted"))
                terminal_fail_closed = bool(
                    str(getattr(decision, "decision", ""))
                    == "OTHER_UNSUPPORTED"
                    and not tuple(getattr(decision, "hypotheses", ()) or ())
                    and str(getattr(decision, "reason_code", "")).startswith(
                        "CLASSIFIER_TERMINAL_"
                    )
                )
                terminal_attempt = bool(
                    terminal_fail_closed
                    and not accepted_attempt
                    and outcome.get("terminal_failed")
                )
                if not accepted_attempt and not terminal_attempt:
                    raise ValueError("classification decision has no accepted attempt")
                decision_status = str(
                    getattr(decision, "decision_status", None)
                    or {
                        "NO_RELATION": "EXPLICIT_NO_RELATION",
                        "OTHER_UNSUPPORTED": "SCHEMA_GAP",
                        "HYPOTHESES": "HYPOTHESES_EMITTED",
                    }.get(str(decision.decision), str(decision.decision))
                )
                current_status = str(pair["status"])
                if current_status not in {
                    "PENDING_CLASSIFICATION",
                    decision_status,
                    str(decision.decision),
                }:
                    raise ValueError("pair already has a different terminal decision")
                provenance = dict(pair.get("provenance") or {})
                provenance["classification"] = {
                    "decision": decision.decision,
                    "decision_status": decision_status,
                    "reason_code": decision.reason_code,
                    "attempt_outcome": (
                        "accepted" if accepted_attempt else "terminal_fail_closed"
                    ),
                    "evidence_spans": [
                        [span.start, span.end] for span in decision.evidence_spans
                    ],
                }
                decision_rows.append({
                    "id": str(pair["id"]),
                    "status": decision_status,
                    "provenance": provenance,
                })
                for hypothesis in decision.hypotheses:
                    if hypothesis.pair_identity != decision.pair_identity:
                        raise ValueError("hypothesis belongs to another pair")
                    database_id = self._grounded_relation_db_id(
                        "relhypdb_", run_id, hypothesis.hypothesis_id,
                    )
                    spans = [
                        [span.start, span.end]
                        for span in hypothesis.evidence_spans
                    ]
                    metadata = {
                        "stable_hypothesis_id": hypothesis.hypothesis_id,
                        "evidence_spans": spans,
                        "classifier_confidence": hypothesis.classifier_confidence,
                        "classifier_prompt_version": hypothesis.classifier_prompt_version,
                        "classifier_schema_version": hypothesis.classifier_schema_version,
                        "classifier_parser_version": hypothesis.classifier_parser_version,
                        "predicate_metadata": hypothesis.predicate_metadata,
                    }
                    expected = (
                        hypothesis.hypothesis_id,
                        hypothesis.predicate,
                        hypothesis.subject_occurrence_key,
                        hypothesis.object_occurrence_key,
                        spans,
                    )
                    previous_expected = expected_hypotheses.get(database_id)
                    if previous_expected is not None and previous_expected != expected:
                        raise ValueError("hypothesis immutable payload mismatch")
                    expected_hypotheses[database_id] = expected
                    hypothesis_rows.append({
                        "id": database_id,
                        "pair_opportunity_id": str(pair["id"]),
                        "hypothesis_key": hypothesis.hypothesis_id,
                        "predicate": hypothesis.predicate,
                        "subject_occurrence_key": hypothesis.subject_occurrence_key,
                        "object_occurrence_key": hypothesis.object_occurrence_key,
                        "evidence_start": min(
                            span.start for span in hypothesis.evidence_spans
                        ),
                        "evidence_end": max(
                            span.end for span in hypothesis.evidence_spans
                        ),
                        "classifier_policy_version": (
                            hypothesis.classifier_policy_version
                        ),
                        "classifier_model_revision": (
                            hypothesis.classifier_model_revision
                        ),
                        "metadata": metadata,
                        "source_kind": "v2_classifier",
                        "source_record_id": hypothesis.pair_identity,
                    })
            if decision_rows:
                conn.execute(
                    """UPDATE shadow_relation_pair_opportunities AS pair
                       SET status=item.status,provenance=item.provenance,
                           updated_at=NOW()
                       FROM jsonb_to_recordset(%s::jsonb) AS item(
                           id TEXT,status TEXT,provenance JSONB)
                       WHERE pair.id=item.id""",
                    (self._json(decision_rows),),
                )
            if hypothesis_rows:
                conn.execute(
                    """INSERT INTO shadow_relation_hypotheses(
                           id,pair_opportunity_id,hypothesis_key,predicate,
                           subject_occurrence_key,object_occurrence_key,evidence_start,
                           evidence_end,classifier_policy_version,
                           classifier_model_revision,status,metadata,
                           source_kind,source_record_id)
                       SELECT item.id,item.pair_opportunity_id,item.hypothesis_key,
                              item.predicate,item.subject_occurrence_key,
                              item.object_occurrence_key,item.evidence_start,
                              item.evidence_end,item.classifier_policy_version,
                              item.classifier_model_revision,'CLASSIFIED_POSITIVE',
                              item.metadata,item.source_kind,item.source_record_id
                       FROM jsonb_to_recordset(%s::jsonb) AS item(
                           id TEXT,pair_opportunity_id TEXT,hypothesis_key TEXT,
                           predicate TEXT,subject_occurrence_key TEXT,
                           object_occurrence_key TEXT,evidence_start INTEGER,
                           evidence_end INTEGER,classifier_policy_version TEXT,
                           classifier_model_revision TEXT,metadata JSONB,
                           source_kind TEXT,source_record_id TEXT)
                       ON CONFLICT(id) DO NOTHING""",
                    (self._json(hypothesis_rows),),
                )
                persisted_rows = conn.execute(
                    """SELECT id,hypothesis_key,predicate,subject_occurrence_key,
                              object_occurrence_key,metadata
                       FROM shadow_relation_hypotheses WHERE id=ANY(%s)""",
                    (sorted(expected_hypotheses),),
                ).fetchall()
                persisted_by_id = {
                    str(item["id"]): item for item in persisted_rows
                }
                if len(persisted_by_id) != len(expected_hypotheses):
                    raise ValueError("hypothesis immutable payload mismatch")
                for database_id, expected in expected_hypotheses.items():
                    persisted = persisted_by_id[database_id]
                    immutable = (
                        str(persisted["hypothesis_key"]),
                        str(persisted["predicate"]),
                        str(persisted["subject_occurrence_key"]),
                        str(persisted["object_occurrence_key"]),
                        dict(persisted.get("metadata") or {}).get(
                            "evidence_spans"
                        ),
                    )
                    if immutable != expected:
                        raise ValueError("hypothesis immutable payload mismatch")
            conn.execute(
                """UPDATE shadow_relation_v2_runs SET status='verifying'
                   WHERE id=%s AND status IN ('planning','classifying')""",
                (run_id,),
            )
            return conn.execute(
                """SELECT hypothesis.* FROM shadow_relation_hypotheses hypothesis
                   JOIN shadow_relation_pair_opportunities pair
                     ON pair.id=hypothesis.pair_opportunity_id
                   WHERE pair.run_id=%s ORDER BY hypothesis.id""",
                (run_id,),
            ).fetchall()

    def persist_grounded_relation_validations(
        self,
        run_id: str,
        validations: Any,
    ) -> list[dict[str, Any]]:
        """Persist deterministic hard-rule outcomes before independent verification."""

        values = list(validations)
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            if not values:
                return []
            hypothesis_ids = [
                str(item.hypothesis_id) for item in values
            ]
            if len(set(hypothesis_ids)) != len(hypothesis_ids):
                raise ValueError("validation batch repeats hypothesis")
            hypothesis_rows = conn.execute(
                """SELECT hypothesis.* FROM shadow_relation_hypotheses hypothesis
                   JOIN shadow_relation_pair_opportunities pair
                     ON pair.id=hypothesis.pair_opportunity_id
                   WHERE pair.run_id=%s AND hypothesis.hypothesis_key=ANY(%s)""",
                (run_id, hypothesis_ids),
            ).fetchall()
            hypotheses_by_key = {
                str(item["hypothesis_key"]): item for item in hypothesis_rows
            }
            if len(hypotheses_by_key) != len(hypothesis_ids):
                raise LookupError("validation hypothesis does not exist")
            rows = []
            for validation in values:
                hypothesis = hypotheses_by_key[str(validation.hypothesis_id)]
                rows.append({
                    "hypothesis_id": str(hypothesis["id"]),
                    "status": str(validation.status),
                    "reason_code": str(validation.reason_code),
                    "predicate_metadata": validation.predicate_metadata,
                    "hypothesis_status": {
                    "PASS": "RULE_VALID",
                    "REVIEW_ONLY": "RULE_REVIEW_ONLY",
                    "REJECTED": "RULE_REJECTED",
                    }[validation.status],
                })
            conn.execute(
                """INSERT INTO shadow_relation_hypothesis_validations(
                       hypothesis_id,status,reason_code,predicate_metadata)
                   SELECT item.hypothesis_id,item.status,item.reason_code,
                          item.predicate_metadata
                   FROM jsonb_to_recordset(%s::jsonb) AS item(
                       hypothesis_id TEXT,status TEXT,reason_code TEXT,
                       predicate_metadata JSONB)
                   ON CONFLICT(hypothesis_id) DO NOTHING""",
                (self._json(rows),),
            )
            persisted_rows = conn.execute(
                """SELECT * FROM shadow_relation_hypothesis_validations
                   WHERE hypothesis_id=ANY(%s)""",
                ([str(item["hypothesis_id"]) for item in rows],),
            ).fetchall()
            persisted_by_id = {
                str(item["hypothesis_id"]): item for item in persisted_rows
            }
            if len(persisted_by_id) != len(rows):
                raise ValueError("hypothesis validation immutable payload mismatch")
            for item in rows:
                row = persisted_by_id[str(item["hypothesis_id"])]
                if (
                    str(row["status"]) != str(item["status"])
                    or str(row["reason_code"]) != str(item["reason_code"])
                    or self._canonical_json(row.get("predicate_metadata") or {})
                    != self._canonical_json(item["predicate_metadata"])
                ):
                    raise ValueError("hypothesis validation immutable payload mismatch")
            conn.execute(
                """UPDATE shadow_relation_hypotheses AS hypothesis
                   SET status=item.hypothesis_status
                   FROM jsonb_to_recordset(%s::jsonb) AS item(
                       hypothesis_id TEXT,hypothesis_status TEXT)
                   WHERE hypothesis.id=item.hypothesis_id""",
                (self._json(rows),),
            )
            return conn.execute(
                """SELECT validation.* FROM shadow_relation_hypothesis_validations validation
                   JOIN shadow_relation_hypotheses hypothesis
                     ON hypothesis.id=validation.hypothesis_id
                   JOIN shadow_relation_pair_opportunities pair
                     ON pair.id=hypothesis.pair_opportunity_id
                   WHERE pair.run_id=%s ORDER BY validation.hypothesis_id""",
                (run_id,),
            ).fetchall()

    def persist_grounded_relation_proposed_hypotheses(
        self,
        run_id: str,
        hypotheses: Any,
        *,
        candidate_source: str,
    ) -> list[dict[str, Any]]:
        """Persist non-classifier candidate hypotheses in the relation ledger."""

        values = list(hypotheses)
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            for hypothesis in values:
                pair = conn.execute(
                    """SELECT * FROM shadow_relation_pair_opportunities
                       WHERE run_id=%s AND pair_identity=%s FOR UPDATE""",
                    (run_id, hypothesis.pair_identity),
                ).fetchone()
                if not pair:
                    raise LookupError("proposed hypothesis pair does not exist")
                database_id = self._grounded_relation_db_id(
                    "relhypdb_", run_id, hypothesis.hypothesis_id,
                )
                spans = [
                    [span.start, span.end] for span in hypothesis.evidence_spans
                ]
                metadata = {
                    "stable_hypothesis_id": hypothesis.hypothesis_id,
                    "candidate_source": str(candidate_source),
                    "evidence_spans": spans,
                    "proposal_confidence": hypothesis.classifier_confidence,
                    "proposer_prompt_version": hypothesis.classifier_prompt_version,
                    "proposer_schema_version": hypothesis.classifier_schema_version,
                    "proposer_parser_version": hypothesis.classifier_parser_version,
                    "predicate_metadata": hypothesis.predicate_metadata,
                }
                conn.execute(
                    """INSERT INTO shadow_relation_hypotheses(
                       id,pair_opportunity_id,hypothesis_key,predicate,
                       subject_occurrence_key,object_occurrence_key,evidence_start,
                       evidence_end,classifier_policy_version,
                       classifier_model_revision,status,metadata,
                       source_kind,source_record_id)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'PROPOSED_POSITIVE',
                              %s::jsonb,%s,%s)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        database_id,
                        pair["id"],
                        hypothesis.hypothesis_id,
                        hypothesis.predicate,
                        hypothesis.subject_occurrence_key,
                        hypothesis.object_occurrence_key,
                        min(span.start for span in hypothesis.evidence_spans),
                        max(span.end for span in hypothesis.evidence_spans),
                        hypothesis.classifier_policy_version,
                        hypothesis.classifier_model_revision,
                        self._canonical_json(metadata),
                        f"v2_{candidate_source}",
                        hypothesis.pair_identity,
                    ),
                )
                persisted = conn.execute(
                    "SELECT * FROM shadow_relation_hypotheses WHERE id=%s",
                    (database_id,),
                ).fetchone()
                persisted_metadata = dict(persisted.get("metadata") or {})
                immutable = (
                    str(persisted["hypothesis_key"]),
                    str(persisted["predicate"]),
                    str(persisted["subject_occurrence_key"]),
                    str(persisted["object_occurrence_key"]),
                    persisted_metadata.get("evidence_spans"),
                )
                expected = (
                    hypothesis.hypothesis_id,
                    hypothesis.predicate,
                    hypothesis.subject_occurrence_key,
                    hypothesis.object_occurrence_key,
                    spans,
                )
                if immutable != expected:
                    raise ValueError("proposed hypothesis immutable payload mismatch")
                provenance = dict(pair.get("provenance") or {})
                sources = set(provenance.get("candidate_sources") or [])
                sources.add(str(candidate_source))
                provenance["candidate_sources"] = sorted(sources)
                provenance["classification"] = {
                    "decision": "HYPOTHESES",
                    "decision_status": "HYPOTHESES_EMITTED",
                    "reason_code": "LOCAL_SUBGRAPH_PROPOSAL",
                    "entry": "local_subgraph",
                }
                conn.execute(
                    """UPDATE shadow_relation_pair_opportunities
                       SET status='HYPOTHESES_EMITTED',
                           provenance=%s::jsonb,updated_at=NOW()
                       WHERE id=%s""",
                    (self._canonical_json(provenance), pair["id"]),
                )
            return conn.execute(
                """SELECT hypothesis.* FROM shadow_relation_hypotheses hypothesis
                   JOIN shadow_relation_pair_opportunities pair
                     ON pair.id=hypothesis.pair_opportunity_id
                   WHERE pair.run_id=%s ORDER BY hypothesis.hypothesis_key""",
                (run_id,),
            ).fetchall()

    def persist_grounded_relation_verification_attempts(
        self,
        run_id: str,
        attempts: Any,
        *,
        terminal: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Append verifier attempts; retries never overwrite earlier verdicts."""

        values = list(attempts)
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            for attempt in values:
                hypothesis = conn.execute(
                    """SELECT hypothesis.* FROM shadow_relation_hypotheses hypothesis
                       JOIN shadow_relation_pair_opportunities pair
                         ON pair.id=hypothesis.pair_opportunity_id
                       WHERE pair.run_id=%s AND hypothesis.hypothesis_key=%s""",
                    (run_id, attempt.hypothesis_id),
                ).fetchone()
                if not hypothesis:
                    raise LookupError("verification hypothesis does not exist")
                validation = conn.execute(
                    """SELECT * FROM shadow_relation_hypothesis_validations
                       WHERE hypothesis_id=%s""",
                    (hypothesis["id"],),
                ).fetchone()
                if not validation:
                    raise ValueError("verification requires deterministic validation")
                if attempt.independent_call and validation["status"] != "PASS":
                    raise ValueError("independent verifier cannot bypass failed rules")
                database_id = self._grounded_relation_db_id(
                    "relverifydb_", run_id, attempt.attempt_id,
                )
                canonical_attempt = {
                    "stable_attempt_id": attempt.attempt_id,
                    "hypothesis_id": attempt.hypothesis_id,
                    "attempt_no": int(attempt.attempt_no),
                    "verdict": attempt.verdict,
                    "reason_code": attempt.reason_code,
                    "forward_entailment": attempt.forward_entailment,
                    "reverse_entailment": attempt.reverse_entailment,
                    "subject_role_supported": bool(
                        attempt.subject_role_supported
                    ),
                    "object_role_supported": bool(
                        attempt.object_role_supported
                    ),
                    "subject_context_role": attempt.subject_context_role,
                    "object_context_role": attempt.object_context_role,
                    "subject_role_alignment": attempt.subject_role_alignment,
                    "object_role_alignment": attempt.object_role_alignment,
                    "forward_role_alignment": attempt.forward_role_alignment,
                    "reverse_role_alignment": attempt.reverse_role_alignment,
                    "direction_normalization": attempt.direction_normalization,
                    "normalized_subject_occurrence_key": (
                        attempt.normalized_subject_occurrence_key
                    ),
                    "normalized_object_occurrence_key": (
                        attempt.normalized_object_occurrence_key
                    ),
                    "supported_evidence_spans": [
                        [span.start, span.end]
                        for span in attempt.supported_evidence_spans
                    ],
                    "independent_call": bool(attempt.independent_call),
                    "verifier_prompt_version": attempt.verifier_prompt_version,
                    "verifier_schema_version": attempt.verifier_schema_version,
                    "verifier_parser_version": attempt.verifier_parser_version,
                    "verifier_model_revision": attempt.verifier_model_revision,
                    "error_message": attempt.error_message,
                }
                model_request_payload = (
                    dict(attempt.model_request_payload)
                    if attempt.model_request_payload else canonical_attempt
                )
                raw_response_text = str(
                    attempt.raw_response_text
                    or self._canonical_json(canonical_attempt)
                )
                parsed_response = (
                    dict(attempt.parsed_response)
                    if attempt.parsed_response else canonical_attempt
                )
                token_usage = dict(attempt.token_usage or {})
                request_hash = hashlib.sha256(
                    self._canonical_json(model_request_payload).encode("utf-8")
                ).hexdigest()
                response_hash = hashlib.sha256(
                    raw_response_text.encode("utf-8")
                ).hexdigest()
                model_call_id = self._grounded_relation_db_id(
                    "modelcall_",
                    run_id,
                    "verifier",
                    hypothesis["id"],
                    int(attempt.attempt_no),
                )
                idempotency_key = f"verifier:{hypothesis['id']}:{int(attempt.attempt_no)}"
                attempt_is_terminal = (
                    attempt.verdict != "RETRYABLE_FAILED"
                    if terminal is None else bool(terminal)
                )
                terminal_status = (
                    str(attempt.verdict) if attempt_is_terminal else ""
                )
                conn.execute(
                    """INSERT INTO shadow_model_calls(
                       id,run_id,component,model_name,model_digest,
                       prompt_version,schema_version,parser_version,
                       idempotency_key,attempt_no,status,request_hash,response_hash,
                       request_payload,raw_response_text,parsed_response,
                       token_usage,error_type,error_message,independent_call,
                       audit_quality,terminal_status)
                       VALUES(%s,%s,'verifier',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                              %s::jsonb,%s,%s::jsonb,%s::jsonb,%s,%s,%s,
                              'complete',%s)
                       ON CONFLICT(run_id,component,idempotency_key,attempt_no)
                       DO UPDATE SET terminal_status=EXCLUDED.terminal_status
                       WHERE shadow_model_calls.request_hash=EXCLUDED.request_hash
                         AND shadow_model_calls.terminal_status=''""",
                    (
                        model_call_id,
                        run_id,
                        str(
                            model_request_payload.get("model")
                            or attempt.verifier_model_revision
                            or ""
                        ),
                        str(attempt.verifier_model_revision or ""),
                        attempt.verifier_prompt_version,
                        attempt.verifier_schema_version,
                        attempt.verifier_parser_version,
                        idempotency_key,
                        int(attempt.attempt_no),
                        attempt.verdict,
                        request_hash,
                        response_hash,
                        self._canonical_json(model_request_payload),
                        raw_response_text,
                        self._canonical_json(parsed_response),
                        self._canonical_json(token_usage),
                        "RelationClassificationError"
                        if attempt.verdict == "RETRYABLE_FAILED" else "",
                        str(attempt.error_message or ""),
                        bool(attempt.independent_call),
                        terminal_status,
                    ),
                )
                conn.execute(
                    """INSERT INTO shadow_relation_verification_attempts(
                       id,hypothesis_id,attempt_no,verdict,reason_code,
                       supported_evidence_spans,independent_call,
                       verifier_prompt_version,verifier_schema_version,
                       verifier_parser_version,verifier_model_revision,
                       request_hash,response_hash,error_message,model_call_id,
                       audit_quality,terminal_status)
                       VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                              'complete',%s)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        database_id,
                        hypothesis["id"],
                        int(attempt.attempt_no),
                        attempt.verdict,
                        attempt.reason_code,
                        self._canonical_json(
                            canonical_attempt["supported_evidence_spans"]
                        ),
                        bool(attempt.independent_call),
                        attempt.verifier_prompt_version,
                        attempt.verifier_schema_version,
                        attempt.verifier_parser_version,
                        attempt.verifier_model_revision,
                        request_hash,
                        response_hash,
                        attempt.error_message,
                        model_call_id,
                        terminal_status,
                    ),
                )
                if attempt_is_terminal:
                    conn.execute(
                        """UPDATE shadow_relation_verification_attempts
                           SET terminal_status=%s
                           WHERE id=%s AND request_hash=%s
                             AND terminal_status=''""",
                        (terminal_status, database_id, request_hash),
                    )
                row = conn.execute(
                    "SELECT * FROM shadow_relation_verification_attempts WHERE id=%s",
                    (database_id,),
                ).fetchone()
                if (
                    int(row["attempt_no"]) != int(attempt.attempt_no)
                    or str(row["verdict"]) != str(attempt.verdict)
                    or str(row["request_hash"]) != request_hash
                ):
                    raise ValueError("verification attempt immutable payload mismatch")
                conn.execute(
                    "UPDATE shadow_relation_hypotheses SET status=%s WHERE id=%s",
                    (f"VERIFIER_{attempt.verdict}", hypothesis["id"]),
                )
            return conn.execute(
                """SELECT attempt.* FROM shadow_relation_verification_attempts attempt
                   JOIN shadow_relation_hypotheses hypothesis
                     ON hypothesis.id=attempt.hypothesis_id
                   JOIN shadow_relation_pair_opportunities pair
                     ON pair.id=hypothesis.pair_opportunity_id
                   WHERE pair.run_id=%s
                   ORDER BY hypothesis.hypothesis_key,attempt.attempt_no""",
                (run_id,),
            ).fetchall()

    def persist_grounded_relation_normalization(
        self,
        run_id: str,
        result: Any,
        *,
        input_fingerprint: str,
        materialization_decisions: Any = (),
    ) -> dict[str, int]:
        """Persist normalized relation facts and provenance, not product assertions."""

        decisions = {
            str(item.fact_id): item for item in materialization_decisions
        }
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            run = conn.execute(
                "SELECT * FROM shadow_relation_v2_runs WHERE id=%s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("Relation run does not exist")
            if str(run["input_fingerprint"]) != str(input_fingerprint):
                raise ValueError("normalization fingerprint does not match run")
            for fact in result.facts:
                if int(fact.project_id) != int(run["project_id"]):
                    raise ValueError("normalized fact belongs to another project")
                conn.execute(
                    """INSERT INTO shadow_relation_normalized_facts(
                       id,project_id,semantic_fact_key,subject_canonical_id,predicate,
                       object_canonical_id,is_symmetric,is_reflexive,is_review_only,
                       qualifiers,audit_quality)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'{}'::jsonb,'complete')
                       ON CONFLICT(id) DO UPDATE SET audit_quality='complete'""",
                    (
                        fact.fact_id,
                        int(fact.project_id),
                        fact.semantic_fact_key,
                        fact.subject_canonical_id,
                        fact.predicate,
                        fact.object_canonical_id,
                        bool(fact.symmetric),
                        bool(fact.reflexive),
                        bool(fact.review_only),
                    ),
                )
                persisted = conn.execute(
                    "SELECT * FROM shadow_relation_normalized_facts WHERE id=%s",
                    (fact.fact_id,),
                ).fetchone()
                immutable = (
                    int(persisted["project_id"]),
                    str(persisted["semantic_fact_key"]),
                    str(persisted["subject_canonical_id"]),
                    str(persisted["predicate"]),
                    str(persisted["object_canonical_id"]),
                    bool(persisted["is_symmetric"]),
                    bool(persisted["is_reflexive"]),
                    bool(persisted["is_review_only"]),
                )
                expected = (
                    int(fact.project_id),
                    fact.semantic_fact_key,
                    fact.subject_canonical_id,
                    fact.predicate,
                    fact.object_canonical_id,
                    bool(fact.symmetric),
                    bool(fact.reflexive),
                    bool(fact.review_only),
                )
                if immutable != expected:
                    raise ValueError("normalized fact immutable payload mismatch")
                decision = decisions.get(fact.fact_id)
                if decision and decision.eligible:
                    membership_status = "ELIGIBLE"
                elif fact.review_only:
                    membership_status = "REVIEW_ONLY"
                else:
                    membership_status = "NORMALIZED"
                conn.execute(
                    """INSERT INTO shadow_relation_fact_memberships(
                       run_id,fact_id,input_fingerprint,status)
                       VALUES(%s,%s,%s,%s)
                       ON CONFLICT(run_id,fact_id) DO UPDATE SET
                           status=EXCLUDED.status
                       WHERE shadow_relation_fact_memberships.input_fingerprint
                             =EXCLUDED.input_fingerprint""",
                    (
                        run_id,
                        fact.fact_id,
                        input_fingerprint,
                        membership_status,
                    ),
                )
            for evidence in result.evidence:
                hypothesis = conn.execute(
                    """SELECT hypothesis.* FROM shadow_relation_hypotheses hypothesis
                       JOIN shadow_relation_pair_opportunities pair
                         ON pair.id=hypothesis.pair_opportunity_id
                       WHERE pair.run_id=%s AND hypothesis.hypothesis_key=%s""",
                    (run_id, evidence.hypothesis_id),
                ).fetchone()
                if not hypothesis:
                    raise LookupError("fact evidence hypothesis does not exist")
                verification_id = self._grounded_relation_db_id(
                    "relverifydb_", run_id, evidence.verification_attempt_id,
                )
                verification = conn.execute(
                    """SELECT * FROM shadow_relation_verification_attempts
                       WHERE id=%s AND hypothesis_id=%s""",
                    (verification_id, hypothesis["id"]),
                ).fetchone()
                if (
                    not verification
                    or verification["verdict"] != "SUPPORTED"
                    or not verification["independent_call"]
                ):
                    raise ValueError("fact evidence lacks independent verifier support")
                pair = conn.execute(
                    """SELECT * FROM shadow_relation_pair_opportunities
                       WHERE id=%s AND run_id=%s""",
                    (hypothesis["pair_opportunity_id"], run_id),
                ).fetchone()
                if not pair:
                    raise LookupError("fact evidence pair does not exist")
                if str(pair.get("context_id") or ""):
                    source_context = conn.execute(
                        """SELECT id FROM shadow_source_contexts
                           WHERE run_id=%s AND id=%s""",
                        (run_id, str(pair["context_id"])),
                    ).fetchone()
                else:
                    pair_provenance = dict(pair.get("provenance") or {})
                    source_context = conn.execute(
                        """SELECT id FROM shadow_source_contexts
                           WHERE run_id=%s AND context_identity=%s""",
                        (
                            run_id,
                            str(
                                pair_provenance.get("source_context_id")
                                or pair.get("context_identity")
                                or ""
                            ),
                        ),
                    ).fetchone()
                if not source_context:
                    raise ValueError("fact evidence lacks frozen source context")
                evidence_id = self._grounded_relation_db_id(
                    "relevidencedb_", run_id, evidence.evidence_id,
                )
                conn.execute(
                    """INSERT INTO shadow_relation_fact_evidence(
                       id,fact_id,run_id,evidence_key,document_revision_id,
                       subject_occurrence_key,object_occurrence_key,evidence_spans,
                       hypothesis_id,verification_attempt_id,provenance,
                       context_id,source_kind,source_record_id,qualifiers,
                       audit_quality)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb,
                              %s,%s,%s,'{}'::jsonb,'complete')
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        evidence_id,
                        evidence.fact_id,
                        run_id,
                        evidence.evidence_key,
                        evidence.document_revision_id,
                        evidence.subject_occurrence_key,
                        evidence.object_occurrence_key,
                        self._canonical_json([
                            [span.start, span.end] for span in evidence.evidence_spans
                        ]),
                        hypothesis["id"],
                        verification_id,
                        self._canonical_json(evidence.provenance),
                        source_context["id"],
                        "v2_verifier",
                        evidence.hypothesis_id,
                    ),
                )
            for collapse in result.collapse_audit:
                hypothesis = conn.execute(
                    """SELECT hypothesis.id FROM shadow_relation_hypotheses hypothesis
                       JOIN shadow_relation_pair_opportunities pair
                         ON pair.id=hypothesis.pair_opportunity_id
                       WHERE pair.run_id=%s AND hypothesis.hypothesis_key=%s""",
                    (run_id, collapse["hypothesis_id"]),
                ).fetchone()
                if not hypothesis:
                    raise LookupError("collapse audit hypothesis does not exist")
                audit_id = self._grounded_relation_db_id(
                    "relcollapsedb_",
                    run_id,
                    collapse["fact_id"],
                    collapse["hypothesis_id"],
                    collapse["reason_code"],
                )
                conn.execute(
                    """INSERT INTO shadow_relation_fact_collapse_audit(
                       id,run_id,fact_id,hypothesis_id,reason_code,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        audit_id,
                        run_id,
                        collapse["fact_id"],
                        hypothesis["id"],
                        collapse["reason_code"],
                        self._canonical_json(collapse),
                    ),
                )
            for conflict in result.conflicts:
                conn.execute(
                    """INSERT INTO shadow_relation_fact_conflicts(
                       id,run_id,pair_identity,hypothesis_id,reason_code,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        self._grounded_relation_db_id(
                            "relconflictdb_", run_id, conflict.conflict_id,
                        ),
                        run_id,
                        conflict.pair_identity,
                        conflict.hypothesis_id,
                        conflict.reason_code,
                        self._canonical_json(conflict.metadata),
                    ),
                )
            conn.execute(
                """UPDATE shadow_relation_v2_runs
                   SET status='normalizing',metrics=%s::jsonb
                   WHERE id=%s AND status IN ('classifying','verifying','normalizing')""",
                (self._canonical_json(result.metrics), run_id),
            )
            return {
                "facts": len(result.facts),
                "evidence": len(result.evidence),
                "conflicts": len(result.conflicts),
                "collapsed": len(result.collapse_audit),
            }

    def finish_grounded_relation_run(
        self,
        run_id: str,
        *,
        status: str,
        metrics: dict[str, Any],
        input_fingerprint: str,
        error_message: str = "",
    ) -> dict[str, Any]:
        """Close the isolated run only after its pair and verifier ledgers settle."""

        if status not in {"ready", "stale", "failed", "cancelled"}:
            raise ValueError("invalid relation run terminal status")
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, run_id)
            run = conn.execute(
                "SELECT * FROM shadow_relation_v2_runs WHERE id=%s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("Relation run does not exist")
            if str(run["input_fingerprint"]) != str(input_fingerprint):
                raise ValueError("completion fingerprint does not match run")
            if status == "ready":
                pending_pairs = int(conn.execute(
                    """SELECT COUNT(*) AS count
                       FROM shadow_relation_pair_opportunities
                       WHERE run_id=%s AND status='PENDING_CLASSIFICATION'""",
                    (run_id,),
                ).fetchone()["count"])
                incomplete_hypotheses = int(conn.execute(
                    """SELECT COUNT(*) AS count
                       FROM shadow_relation_hypotheses hypothesis
                       JOIN shadow_relation_pair_opportunities pair
                         ON pair.id=hypothesis.pair_opportunity_id
                       LEFT JOIN shadow_relation_hypothesis_validations validation
                         ON validation.hypothesis_id=hypothesis.id
                       LEFT JOIN shadow_relation_verification_attempts attempt
                         ON attempt.hypothesis_id=hypothesis.id
                       WHERE pair.run_id=%s
                         AND (validation.hypothesis_id IS NULL OR attempt.id IS NULL)""",
                    (run_id,),
                ).fetchone()["count"])
                unsupported_eligible = int(conn.execute(
                    """SELECT COUNT(*) AS count
                       FROM shadow_relation_fact_memberships membership
                       WHERE membership.run_id=%s AND membership.status='ELIGIBLE'
                         AND NOT EXISTS (
                             SELECT 1 FROM shadow_relation_fact_evidence evidence
                             JOIN shadow_relation_verification_attempts attempt
                               ON attempt.id=evidence.verification_attempt_id
                             WHERE evidence.run_id=membership.run_id
                               AND evidence.fact_id=membership.fact_id
                               AND attempt.verdict='SUPPORTED'
                               AND attempt.independent_call=TRUE
                         )""",
                    (run_id,),
                ).fetchone()["count"])
                if pending_pairs or incomplete_hypotheses or unsupported_eligible:
                    raise ValueError(
                        "Relation run is incomplete and cannot become ready"
                    )
            merged_metrics = {
                **dict(run.get("metrics") or {}),
                **dict(metrics),
            }
            row = conn.execute(
                """UPDATE shadow_relation_v2_runs
                   SET status=%s,metrics=%s::jsonb,error_message=%s,
                       audit_quality='complete',completed_at=NOW()
                   WHERE id=%s RETURNING *""",
                (
                    status,
                    self._canonical_json(merged_metrics),
                    error_message or "",
                    run_id,
                ),
            ).fetchone()
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
            policy_set = normalize_policy_set(snapshot["policy_set"])
            from .policy import CONTROL_POLICY_SET
            if policy_set != dict(CONTROL_POLICY_SET):
                materialization_fingerprint = str(
                    snapshot.get("materialization_fingerprint") or ""
                )
                current = conn.execute(
                    """SELECT current_fingerprint
                       FROM shadow_relation_current_fingerprints
                       WHERE project_id=%s FOR UPDATE""",
                    (project_id,),
                ).fetchone()
                current_fingerprint = str(
                    (current or {}).get("current_fingerprint") or ""
                )
                if (
                    not materialization_fingerprint
                    or not current_fingerprint
                    or materialization_fingerprint != current_fingerprint
                ):
                    raise ValueError(
                        "treatment materialization_fingerprint 已过期，禁止激活"
                    )
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

    def create_grounded_entity_namespace(
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
                    raise RuntimeError("实体快照 namespace 输入指纹冲突")
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

    def get_grounded_entity_namespace(self, namespace_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM entity_shadow_namespaces WHERE id=%s", (namespace_id,),
            ).fetchone()

    def list_grounded_entity_embeddings(
        self, namespace_id: str, canonical_keys: list[str],
    ) -> list[dict[str, Any]]:
        """Read a bounded set of vectors from one immutable entity snapshot.

        Community navigation reuses the exact embedding model recorded on the
        namespace.  It must not silently mix models or generate fresh vectors
        during graph publication.
        """

        keys = sorted({str(key) for key in canonical_keys if str(key)})
        if not keys:
            return []
        if len(keys) > 10_000:
            raise ValueError("社区主题关联请求的实体向量超过安全上限")
        with self.connection() as conn:
            namespace = conn.execute(
                """SELECT status,snapshot_complete,embedding_model
                   FROM entity_shadow_namespaces WHERE id=%s""",
                (namespace_id,),
            ).fetchone()
            if not namespace:
                raise LookupError("实体快照 namespace 不存在")
            if (
                str(namespace.get("status") or "") != "ready"
                or not bool(namespace.get("snapshot_complete"))
            ):
                raise RuntimeError("实体快照尚未就绪，不能读取社区主题向量")
            model_name = str(namespace.get("embedding_model") or "")
            if not model_name:
                return []
            return conn.execute(
                """SELECT embedding.canonical_key,embedding.model_name,
                          embedding.embedding,embedding.dimension,
                          canonical.name,canonical.primary_type
                   FROM entity_shadow_embeddings embedding
                   JOIN entity_shadow_canonicals canonical
                     ON canonical.namespace_id=embedding.namespace_id
                    AND canonical.canonical_key=embedding.canonical_key
                   WHERE embedding.namespace_id=%s
                     AND embedding.model_name=%s
                     AND embedding.canonical_key=ANY(%s)
                   ORDER BY embedding.canonical_key""",
                (namespace_id, model_name, keys),
            ).fetchall()

    def seed_grounded_entity_work_items(
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

    def requeue_terminal_grounded_entity_work(self, namespace_id: str) -> int:
        """Requeue a failed snapshot when a new build explicitly retries it."""

        with self.connection() as conn:
            rows = conn.execute(
                """SELECT chunk_id FROM entity_shadow_work_items
                   WHERE namespace_id=%s AND status='terminal_failed'
                   ORDER BY chunk_id FOR UPDATE""",
                (namespace_id,),
            ).fetchall()
            if not rows:
                return 0
            for row in rows:
                self._clear_grounded_entity_chunk_artifacts(
                    conn, namespace_id, int(row["chunk_id"]),
                )
            reset = conn.execute(
                """UPDATE entity_shadow_work_items SET
                   status='pending',attempt=0,lease_owner=NULL,lease_token=NULL,
                   lease_expires_at=NULL,output_json='{}'::jsonb,output_hash='',
                   error_message='',completed_at=NULL,updated_at=NOW()
                   WHERE namespace_id=%s AND status='terminal_failed'
                   RETURNING id""",
                (namespace_id,),
            ).fetchall()
            counts = conn.execute(
                """SELECT COUNT(*) AS expected,
                   COUNT(*) FILTER(WHERE status='completed') AS completed
                   FROM entity_shadow_work_items WHERE namespace_id=%s""",
                (namespace_id,),
            ).fetchone()
            conn.execute(
                """UPDATE entity_shadow_namespaces SET status='building',
                   metrics='{}'::jsonb,error_message='',artifact_hash='',
                   expected_work_items=%s,completed_work_items=%s,failed_work_items=0,
                   snapshot_complete=FALSE,completed_at=NULL
                   WHERE id=%s""",
                (
                    int(counts["expected"]), int(counts["completed"]), namespace_id,
                ),
            )
            return len(reset)

    def claim_grounded_entity_work(
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
                       WHERE namespace_id=%s AND (
                         (status IN ('pending','retryable_failed') AND attempt<max_attempts)
                         OR (status='leased' AND lease_expires_at<NOW())
                       )
                       ORDER BY document_id,chunk_id,id FOR UPDATE SKIP LOCKED LIMIT 1
                   )
                   UPDATE entity_shadow_work_items work SET
                     status='leased',
                     attempt=CASE WHEN work.status='leased' THEN work.attempt
                                  ELSE work.attempt+1 END,
                     lease_owner=%s,lease_token=%s,
                     lease_expires_at=NOW()+(%s*INTERVAL '1 second'),updated_at=NOW(),
                     error_message=''
                   FROM candidate WHERE work.id=candidate.id RETURNING work.*""",
                (namespace_id, worker_id, lease_token, max(30, int(lease_seconds))),
            ).fetchone()

    def grounded_entity_work_summary(self, namespace_id: str) -> dict[str, Any]:
        """Return durable progress and the first terminal error for one snapshot."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT COUNT(*) AS expected,
                   COUNT(*) FILTER(WHERE status='completed') AS completed,
                   COUNT(*) FILTER(WHERE status='terminal_failed') AS failed,
                   COUNT(*) FILTER(WHERE status NOT IN ('completed','terminal_failed')) AS pending,
                   MIN(NULLIF(error_message,'')) FILTER(WHERE status='terminal_failed') AS first_error
                   FROM entity_shadow_work_items WHERE namespace_id=%s""",
                (namespace_id,),
            ).fetchone()

    def grounded_entity_layout_coverage_summary(
        self, namespace_id: str,
    ) -> dict[str, Any]:
        """Aggregate the persisted, read-only mention coverage observations."""

        with self.connection() as conn:
            rows = conn.execute(
                """SELECT chunk_id,output_json->'layout_mention_coverage' AS audit
                   FROM entity_shadow_work_items
                   WHERE namespace_id=%s AND status='completed'
                     AND output_json ? 'layout_mention_coverage'
                   ORDER BY document_id,chunk_id""",
                (namespace_id,),
            ).fetchall()
        candidate_count = eligible_count = rejected_count = overlap_skipped = 0
        recovered_count = 0
        rejection_reasons: Counter[str] = Counter()
        chunks_with_candidates = chunks_with_eligible = 0
        policy_versions: set[str] = set()
        for row in rows:
            audit = row.get("audit") if isinstance(row.get("audit"), dict) else {}
            candidates = int(audit.get("candidate_count") or 0)
            eligible = int(audit.get("eligible_count") or 0)
            candidate_count += candidates
            eligible_count += eligible
            rejected_count += int(audit.get("rejected_count") or 0)
            recovered_count += int(audit.get("recovered_count") or 0)
            overlap_skipped += int(audit.get("overlap_skipped") or 0)
            chunks_with_candidates += int(candidates > 0)
            chunks_with_eligible += int(eligible > 0)
            version = str(audit.get("policy_version") or "").strip()
            if version:
                policy_versions.add(version)
            rejection_reasons.update({
                str(reason): int(count or 0)
                for reason, count in dict(
                    audit.get("rejection_reasons") or {}
                ).items()
            })
        return {
            "audited_chunks": len(rows),
            "chunks_with_candidates": chunks_with_candidates,
            "chunks_with_eligible": chunks_with_eligible,
            "candidate_count": candidate_count,
            "eligible_count": eligible_count,
            "rejected_count": rejected_count,
            "recovered_count": recovered_count,
            "overlap_skipped": overlap_skipped,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "policy_versions": sorted(policy_versions),
        }

    def complete_grounded_entity_work(
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

    def fail_grounded_entity_work(
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
                   RETURNING status,namespace_id,chunk_id""",
                (error[:4000], work_id, lease_token),
            ).fetchone()
            if row:
                self._clear_grounded_entity_chunk_artifacts(
                    conn, str(row["namespace_id"]), int(row["chunk_id"]),
                )
            return str((row or {}).get("status") or "stale")

    def reset_grounded_entity_chunk_attempt(
        self,
        namespace_id: str,
        chunk_id: int,
    ) -> dict[str, int]:
        """Remove partial artifacts before a leased chunk attempt is retried."""

        with self.connection() as conn:
            return self._clear_grounded_entity_chunk_artifacts(
                conn, namespace_id, chunk_id,
            )

    def _clear_grounded_entity_chunk_artifacts(
        self,
        conn: Any,
        namespace_id: str,
        chunk_id: int,
    ) -> dict[str, int]:
        assignment_rows = conn.execute(
            """SELECT mention_key,canonical_key
               FROM entity_shadow_assignments
               WHERE namespace_id=%s AND chunk_id=%s""",
            (namespace_id, chunk_id),
        ).fetchall()
        mention_rows = conn.execute(
            """SELECT mention_key FROM entity_shadow_mentions
               WHERE namespace_id=%s AND chunk_id=%s""",
            (namespace_id, chunk_id),
        ).fetchall()
        mention_keys = sorted({
            str(item["mention_key"]) for item in [*assignment_rows, *mention_rows]
        })
        affected_canonicals = sorted({
            str(item["canonical_key"]) for item in assignment_rows
        })

        removed_relations = int(conn.execute(
            """WITH removed AS (
                 DELETE FROM entity_shadow_relation_endpoints
                 WHERE namespace_id=%s AND (
                   chunk_id=%s
                   OR source_mention_key=ANY(%s)
                   OR target_mention_key=ANY(%s)
                 )
                 RETURNING 1
               ) SELECT COUNT(*) AS count FROM removed""",
            (namespace_id, chunk_id, mention_keys, mention_keys),
        ).fetchone()["count"])
        if mention_keys:
            conn.execute(
                """DELETE FROM entity_shadow_must_links
                   WHERE namespace_id=%s AND mention_key=ANY(%s)""",
                (namespace_id, mention_keys),
            )
            conn.execute(
                """DELETE FROM entity_shadow_candidates
                   WHERE namespace_id=%s AND mention_key=ANY(%s)""",
                (namespace_id, mention_keys),
            )
            conn.execute(
                """DELETE FROM entity_shadow_aliases
                   WHERE namespace_id=%s AND mention_key=ANY(%s)""",
                (namespace_id, mention_keys),
            )
            conn.execute(
                """DELETE FROM entity_shadow_type_assertions
                   WHERE namespace_id=%s AND mention_key=ANY(%s)""",
                (namespace_id, mention_keys),
            )
            conn.execute(
                """DELETE FROM entity_shadow_events
                   WHERE namespace_id=%s AND mention_key=ANY(%s)""",
                (namespace_id, mention_keys),
            )
            conn.execute(
                """DELETE FROM entity_shadow_assignments
                   WHERE namespace_id=%s AND mention_key=ANY(%s)""",
                (namespace_id, mention_keys),
            )
        removed_mentions = int(conn.execute(
            """WITH removed AS (
                 DELETE FROM entity_shadow_mentions
                 WHERE namespace_id=%s AND chunk_id=%s
                 RETURNING 1
               ) SELECT COUNT(*) AS count FROM removed""",
            (namespace_id, chunk_id),
        ).fetchone()["count"])
        conn.execute(
            """DELETE FROM entity_shadow_filter_decisions
               WHERE namespace_id=%s AND chunk_id=%s""",
            (namespace_id, chunk_id),
        )

        removed_canonicals = 0
        if affected_canonicals:
            orphan_rows = conn.execute(
                """SELECT canonical.canonical_key
                   FROM entity_shadow_canonicals canonical
                   WHERE canonical.namespace_id=%s
                     AND canonical.canonical_key=ANY(%s)
                     AND NOT EXISTS (
                       SELECT 1 FROM entity_shadow_assignments assignment
                       WHERE assignment.namespace_id=canonical.namespace_id
                         AND assignment.canonical_key=canonical.canonical_key
                     )
                     AND NOT EXISTS (
                       SELECT 1 FROM entity_shadow_must_links link
                       WHERE link.namespace_id=canonical.namespace_id
                         AND link.canonical_key=canonical.canonical_key
                     )""",
                (namespace_id, affected_canonicals),
            ).fetchall()
            orphan_keys = [str(item["canonical_key"]) for item in orphan_rows]
            if orphan_keys:
                conn.execute(
                    """DELETE FROM entity_shadow_relation_endpoints
                       WHERE namespace_id=%s AND (
                         source_canonical_key=ANY(%s)
                         OR target_canonical_key=ANY(%s)
                       )""",
                    (namespace_id, orphan_keys, orphan_keys),
                )
                conn.execute(
                    """DELETE FROM entity_shadow_candidates
                       WHERE namespace_id=%s AND canonical_key=ANY(%s)""",
                    (namespace_id, orphan_keys),
                )
                conn.execute(
                    """DELETE FROM entity_shadow_aliases
                       WHERE namespace_id=%s AND canonical_key=ANY(%s)""",
                    (namespace_id, orphan_keys),
                )
                conn.execute(
                    """DELETE FROM entity_shadow_embeddings
                       WHERE namespace_id=%s AND canonical_key=ANY(%s)""",
                    (namespace_id, orphan_keys),
                )
                removed_canonicals = int(conn.execute(
                    """WITH removed AS (
                         DELETE FROM entity_shadow_canonicals
                         WHERE namespace_id=%s AND canonical_key=ANY(%s)
                         RETURNING 1
                       ) SELECT COUNT(*) AS count FROM removed""",
                    (namespace_id, orphan_keys),
                ).fetchone()["count"])

        conn.execute(
            """UPDATE entity_shadow_canonicals canonical
               SET source_count=(
                 SELECT COUNT(*) FROM entity_shadow_assignments assignment
                 WHERE assignment.namespace_id=canonical.namespace_id
                   AND assignment.canonical_key=canonical.canonical_key
               )
               WHERE canonical.namespace_id=%s""",
            (namespace_id,),
        )
        return {
            "mentions": removed_mentions,
            "relations": removed_relations,
            "canonicals": removed_canonicals,
        }

    def persist_grounded_entity_extraction(
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
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                       ON CONFLICT(namespace_id,mention_key) DO NOTHING""",
                    (
                        namespace_id, item["mention_id"], project_id, document_id,
                        revision_id, chunk_id, item["surface"], item["normalized_surface"],
                        item.get("evidence") or item["surface"], int(item["start"]),
                        int(item["end"]), float(item["confidence"]),
                        item["extraction_policy_version"],
                        str(item.get("filter_decision") or "accepted"),
                        str(item.get("filter_reason") or ""),
                        self._json(item.get("provenance")),
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
                        str(
                            item.get("policy_version")
                            or "entity-extraction-p2-structure-gate-v2"
                        ),
                    ),
                )

    def get_knowledge_forge_entity_extraction_cache(
        self,
        project_id: int,
        cache_key: str,
    ) -> dict[str, Any] | None:
        """Return and account for one immutable, exact-input extraction result."""

        with self.connection() as conn:
            return conn.execute(
                """UPDATE knowledge_forge_entity_extraction_cache
                   SET hit_count=hit_count+1,last_used_at=NOW()
                   WHERE cache_key=%s AND project_id=%s
                   RETURNING *""",
                (cache_key, int(project_id)),
            ).fetchone()

    def persist_knowledge_forge_entity_extraction_cache(
        self,
        *,
        cache_key: str,
        project_id: int,
        document_id: int,
        revision_id: str,
        chunk_id: int,
        content_hash: str,
        schema_fingerprint: str,
        model_name: str,
        model_digest: str,
        extraction_policy_version: str,
        code_version: str,
        request_fingerprint: str,
        request_metadata: dict[str, Any],
        output: dict[str, Any],
        output_hash: str,
    ) -> dict[str, Any]:
        """Persist a successful extraction without replacing prior evidence."""

        with self.connection() as conn:
            row = conn.execute(
                """INSERT INTO knowledge_forge_entity_extraction_cache(
                   cache_key,project_id,document_id,revision_id,chunk_id,
                   content_hash,schema_fingerprint,model_name,model_digest,
                   extraction_policy_version,code_version,request_fingerprint,
                   request_metadata,output_json,output_hash)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
                   ON CONFLICT(cache_key) DO NOTHING
                   RETURNING *""",
                (
                    cache_key, int(project_id), int(document_id), revision_id,
                    int(chunk_id), content_hash, schema_fingerprint, model_name,
                    model_digest, extraction_policy_version, code_version,
                    request_fingerprint, self._json(request_metadata),
                    self._json(output), output_hash,
                ),
            ).fetchone()
            if row:
                return row
            existing = conn.execute(
                """SELECT * FROM knowledge_forge_entity_extraction_cache
                   WHERE cache_key=%s AND project_id=%s FOR UPDATE""",
                (cache_key, int(project_id)),
            ).fetchone()
            if not existing:
                raise RuntimeError("Knowledge Forge 实体抽取缓存写入冲突")
            if (
                str(existing.get("request_fingerprint") or "")
                != request_fingerprint
                or str(existing.get("output_hash") or "") != output_hash
            ):
                raise RuntimeError("Knowledge Forge 实体抽取缓存不可变内容不一致")
            return existing

    def delete_knowledge_forge_entity_extraction_cache(
        self,
        project_id: int,
        cache_key: str,
    ) -> bool:
        """Remove only a corrupt cache row; source and graph facts are untouched."""

        with self.connection() as conn:
            row = conn.execute(
                """DELETE FROM knowledge_forge_entity_extraction_cache
                   WHERE cache_key=%s AND project_id=%s RETURNING cache_key""",
                (cache_key, int(project_id)),
            ).fetchone()
            return bool(row)

    def list_grounded_entity_mentions(
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

    def find_exact_grounded_entity_candidates(
        self, namespace_id: str, normalized_alias: str, limit: int = 10,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT DISTINCT ON (canonical.canonical_key)
                          canonical.*,1.0 AS lexical_score,
                          occurrence.evidence_text AS mention_evidence
                   FROM entity_shadow_aliases alias
                   JOIN entity_shadow_canonicals canonical
                     ON canonical.namespace_id=alias.namespace_id
                    AND canonical.canonical_key=alias.canonical_key
                   LEFT JOIN LATERAL (
                     SELECT mention.evidence_text
                     FROM entity_shadow_assignments assignment
                     JOIN entity_shadow_mentions mention
                       ON mention.namespace_id=assignment.namespace_id
                      AND mention.mention_key=assignment.mention_key
                     WHERE assignment.namespace_id=canonical.namespace_id
                       AND assignment.canonical_key=canonical.canonical_key
                     ORDER BY mention.document_id,mention.chunk_id,
                              mention.evidence_start,mention.mention_key
                     LIMIT 1
                   ) occurrence ON TRUE
                   WHERE alias.namespace_id=%s AND alias.normalized_alias=%s
                   ORDER BY canonical.canonical_key LIMIT %s""",
                (namespace_id, normalized_alias, max(1, min(10, limit))),
            ).fetchall()

    def find_grounded_entity_canonical(
        self, namespace_id: str, canonical_key: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM entity_shadow_canonicals
                   WHERE namespace_id=%s AND canonical_key=%s""",
                (namespace_id, canonical_key),
            ).fetchone()

    def search_grounded_entity_name_candidates(
        self,
        namespace_id: str,
        normalized_alias: str,
        limit: int,
        *,
        preferred_document_id: int | None = None,
        preferred_chunk_id: int | None = None,
    ) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(50, limit))
        preferred_document = int(preferred_document_id or -1)
        preferred_chunk = int(preferred_chunk_id or -1)
        with self.connection() as conn:
            return conn.execute(
                """WITH ranked AS MATERIALIZED (
                     SELECT canonical.*,
                            MAX(similarity(alias.normalized_alias,%s)) AS lexical_score
                     FROM entity_shadow_aliases alias
                     JOIN entity_shadow_canonicals canonical
                       ON canonical.namespace_id=alias.namespace_id
                      AND canonical.canonical_key=alias.canonical_key
                     WHERE alias.namespace_id=%s AND alias.normalized_alias %% %s
                     GROUP BY canonical.namespace_id,canonical.canonical_key
                     ORDER BY lexical_score DESC,canonical.canonical_key LIMIT %s
                   )
                   SELECT ranked.*,occurrence.evidence_text AS mention_evidence,
                          occurrence.document_id AS mention_document_id,
                          occurrence.chunk_id AS mention_chunk_id,
                          occurrence.evidence_start AS mention_evidence_start,
                          occurrence.evidence_end AS mention_evidence_end,
                          occurrence.surface AS mention_surface
                   FROM ranked
                   LEFT JOIN LATERAL (
                     SELECT mention.evidence_text,mention.document_id,mention.chunk_id,
                            mention.evidence_start,mention.evidence_end,mention.surface
                     FROM entity_shadow_assignments assignment
                     JOIN entity_shadow_mentions mention
                       ON mention.namespace_id=assignment.namespace_id
                      AND mention.mention_key=assignment.mention_key
                     WHERE assignment.namespace_id=ranked.namespace_id
                       AND assignment.canonical_key=ranked.canonical_key
                     ORDER BY (mention.chunk_id=%s) DESC,
                              (mention.document_id=%s) DESC,
                              mention.document_id,mention.chunk_id,
                              mention.evidence_start,mention.mention_key
                     LIMIT 1
                   ) occurrence ON TRUE
                   ORDER BY ranked.lexical_score DESC,ranked.canonical_key""",
                (
                    normalized_alias,
                    namespace_id,
                    normalized_alias,
                    bounded_limit,
                    preferred_chunk,
                    preferred_document,
                ),
            ).fetchall()

    def _ensure_grounded_entity_hnsw_index(
        self, conn: psycopg.Connection, model_name: str, dimension: int,
    ) -> None:
        if dimension <= 0 or dimension > Config.pgvector_hnsw_max_dimensions:
            raise RuntimeError(
                f"实体 embedding 维度 {dimension} 无可用 HNSW，按策略拒绝退化为全表扫描"
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

    def search_grounded_entity_vector_candidates(
        self,
        namespace_id: str,
        model_name: str,
        query_vector: list[float],
        limit: int,
        *,
        preferred_document_id: int | None = None,
        preferred_chunk_id: int | None = None,
    ) -> list[dict[str, Any]]:
        vector_literal = self._vector_literal(query_vector)
        dimension = len(query_vector)
        vector_type = sql.SQL("vector({})").format(sql.Literal(dimension))
        statement = sql.SQL(
            """WITH ranked AS MATERIALIZED (
                 SELECT embedding.namespace_id,embedding.canonical_key,
                        embedding.embedding_vector::{} <=> %s::{} AS distance
                 FROM entity_shadow_embeddings embedding
                 JOIN entity_shadow_canonicals eligible
                   ON eligible.namespace_id=embedding.namespace_id
                  AND eligible.canonical_key=embedding.canonical_key
                 WHERE embedding.namespace_id=%s AND embedding.model_name=%s
                   AND embedding.dimension=%s
                   AND COALESCE(
                     (eligible.metadata#>>'{{entity_publishability,retrieval_eligible}}')::boolean,
                     TRUE
                   )
                 ORDER BY embedding.embedding_vector::{} <=> %s::{} LIMIT %s
               )
               SELECT canonical.*,GREATEST(0.0,1.0-ranked.distance) AS vector_score,
                      occurrence.evidence_text AS mention_evidence,
                      occurrence.document_id AS mention_document_id,
                      occurrence.chunk_id AS mention_chunk_id,
                      occurrence.evidence_start AS mention_evidence_start,
                      occurrence.evidence_end AS mention_evidence_end,
                      occurrence.surface AS mention_surface
               FROM ranked
               JOIN entity_shadow_canonicals canonical
                 ON canonical.namespace_id=ranked.namespace_id
                AND canonical.canonical_key=ranked.canonical_key
               LEFT JOIN LATERAL (
                 SELECT mention.evidence_text,mention.document_id,mention.chunk_id,
                        mention.evidence_start,mention.evidence_end,mention.surface
                 FROM entity_shadow_assignments assignment
                 JOIN entity_shadow_mentions mention
                   ON mention.namespace_id=assignment.namespace_id
                  AND mention.mention_key=assignment.mention_key
                 WHERE assignment.namespace_id=ranked.namespace_id
                   AND assignment.canonical_key=ranked.canonical_key
                 ORDER BY (mention.chunk_id=%s) DESC,
                          (mention.document_id=%s) DESC,
                          mention.document_id,mention.chunk_id,
                          mention.evidence_start,mention.mention_key
                 LIMIT 1
               ) occurrence ON TRUE
               ORDER BY ranked.distance,ranked.canonical_key"""
        ).format(vector_type, vector_type, vector_type, vector_type)
        with self.connection() as conn:
            self._ensure_grounded_entity_hnsw_index(conn, model_name, dimension)
            conn.execute("SELECT set_config('hnsw.iterative_scan','relaxed_order',TRUE)")
            return conn.execute(
                statement,
                (
                    vector_literal, namespace_id, model_name, dimension,
                    vector_literal, max(1, min(100, limit)),
                    int(preferred_chunk_id or -1),
                    int(preferred_document_id or -1),
                ),
            ).fetchall()

    def persist_grounded_entity_candidates(self, rows: list[dict[str, Any]]) -> None:
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

    @staticmethod
    def _grounded_entity_display_aliases(
        canonical: dict[str, Any],
        mention: dict[str, Any],
        aliases: list[dict[str, Any]],
    ) -> list[str]:
        return sorted({
            str(value).strip()
            for value in (
                canonical.get("name"),
                mention.get("surface"),
                *(canonical.get("aliases") or []),
                *(item.get("surface") for item in aliases),
            )
            if str(value or "").strip()
        })

    def upsert_grounded_entity_canonical(
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
        # Canonical aliases are presentation/search surfaces, while normalized
        # keys live in ``entity_shadow_aliases.normalized_alias``.  Keeping
        # only compact normalized strings here made merged names disappear
        # from graph display and surface-based retrieval/evaluation.
        display_aliases = self._grounded_entity_display_aliases(
            canonical, mention, aliases,
        )
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
                    display_aliases, self._json(canonical.get("metadata")),
                ),
            ).fetchone()
            alias_rows = [{
                "surface": mention["surface"],
                "normalized_surface": mention["normalized_surface"],
                "start": mention["start"], "end": mention["end"],
                "confidence": (
                    0.5 if mention.get("confidence") is None
                    else mention.get("confidence")
                ),
                "kind": "surface_observation",
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
                        alias.get("start"), alias.get("end"),
                        float(
                            0.5 if alias.get("confidence") is None
                            else alias.get("confidence")
                        ),
                        mention["policy_version"], self._json(alias.get("metadata")),
                    ),
                )
            if embedding is not None:
                self._ensure_grounded_entity_hnsw_index(conn, embedding_model, len(embedding))
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
                        self._json({
                            "config_hash": config_hash,
                            "embedding_runtime_profile": (
                                Config.ollama_embedding_runtime_profile
                            ),
                        }),
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

    def list_grounded_entity_cannot_links(self, namespace_id: str) -> set[frozenset[str]]:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT left_key,right_key FROM entity_shadow_cannot_links
                   WHERE namespace_id=%s AND hard=TRUE""",
                (namespace_id,),
            ).fetchall()
            return {frozenset((str(item["left_key"]), str(item["right_key"]))) for item in rows}

    def list_grounded_entity_must_links(self, namespace_id: str) -> dict[str, str]:
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

    def finalize_grounded_entity_namespace(
        self, namespace_id: str, *, metrics: dict[str, Any], artifact_hash: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            counts = conn.execute(
                """SELECT COUNT(*) AS expected,
                   COUNT(*) FILTER(WHERE status='completed') AS completed,
                   COUNT(*) FILTER(WHERE status='terminal_failed') AS failed,
                   COUNT(*) FILTER(WHERE status NOT IN ('completed','terminal_failed')) AS pending,
                   MIN(NULLIF(error_message,'')) FILTER(WHERE status='terminal_failed') AS first_error
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
                    "" if status == "ready" else str(
                        counts.get("first_error") or "实体快照 work 未全部成功"
                    ),
                    status == "ready", namespace_id,
                ),
            ).fetchone()
            if status != "ready":
                detail = str(counts.get("first_error") or "存在未完成知识块")
                raise RuntimeError(
                    "实体快照不完整，不能标记 ready："
                    f"已完成 {int(counts['completed'])}/{int(counts['expected'])}，"
                    f"失败 {int(counts['failed'])}；{detail}"
                )
            return row

    def persist_grounded_entity_relation_endpoints(
        self, namespace_id: str, rows: list[dict[str, Any]],
    ) -> None:
        with self.connection() as conn:
            # A namespace can be reused for an idempotent source snapshot
            # while each relation run independently re-evaluates publication.
            # Replace the namespace's derived endpoint contribution atomically
            # so a rejected fact from an older run cannot remain navigable in
            # the current graph.
            conn.execute(
                "DELETE FROM entity_shadow_relation_endpoints WHERE namespace_id=%s",
                (namespace_id,),
            )
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

    def validate_grounded_entity_relation_artifact(
        self,
        namespace_id: str,
        run_id: str,
    ) -> dict[str, int | bool]:
        """Prove a namespace endpoint snapshot is exactly one ready run's facts.

        A graph projection is an immutable derived view, not a new inference.
        This check permits a code-only deployment to reproject an already
        audited source run without claiming that a new runtime reproduced its
        historical model-request batching.  Both sides are compared as sets:
        every endpoint must be an eligible fact in the referenced run, and
        every eligible fact must have exactly one endpoint in the namespace.
        """

        with self.connection() as conn:
            run = conn.execute(
                """SELECT id,project_id,status FROM shadow_relation_v2_runs
                   WHERE id=%s""",
                (run_id,),
            ).fetchone()
            namespace = conn.execute(
                """SELECT id,project_id,status,snapshot_complete
                   FROM entity_shadow_namespaces WHERE id=%s""",
                (namespace_id,),
            ).fetchone()
            if (
                not run or not namespace
                or int(run["project_id"]) != int(namespace["project_id"])
                or str(run.get("status") or "") != "ready"
                or str(namespace.get("status") or "") != "ready"
                or not bool(namespace.get("snapshot_complete"))
            ):
                return {
                    "valid": False,
                    "endpoints": 0,
                    "eligible_facts": 0,
                    "endpoint_only": 0,
                    "fact_only": 0,
                }
            counts = conn.execute(
                """WITH endpoint_keys AS (
                       SELECT relation_key
                       FROM entity_shadow_relation_endpoints
                       WHERE namespace_id=%s
                     ), eligible_fact_keys AS (
                       SELECT DISTINCT fact_id AS relation_key
                       FROM shadow_relation_fact_memberships
                       WHERE run_id=%s AND status='ELIGIBLE'
                     )
                   SELECT
                     (SELECT COUNT(*) FROM endpoint_keys) AS endpoints,
                     (SELECT COUNT(*) FROM eligible_fact_keys) AS eligible_facts,
                     (SELECT COUNT(*) FROM endpoint_keys endpoint
                       LEFT JOIN eligible_fact_keys fact
                         ON fact.relation_key=endpoint.relation_key
                       WHERE fact.relation_key IS NULL) AS endpoint_only,
                     (SELECT COUNT(*) FROM eligible_fact_keys fact
                       LEFT JOIN endpoint_keys endpoint
                         ON endpoint.relation_key=fact.relation_key
                       WHERE endpoint.relation_key IS NULL) AS fact_only""",
                (namespace_id, run_id),
            ).fetchone()
            endpoint_only = int(counts["endpoint_only"] or 0)
            fact_only = int(counts["fact_only"] or 0)
            return {
                "valid": endpoint_only == 0 and fact_only == 0,
                "endpoints": int(counts["endpoints"] or 0),
                "eligible_facts": int(counts["eligible_facts"] or 0),
                "endpoint_only": endpoint_only,
                "fact_only": fact_only,
            }

    def load_grounded_entity_snapshot(self, namespace_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            namespace = conn.execute(
                "SELECT * FROM entity_shadow_namespaces WHERE id=%s", (namespace_id,),
            ).fetchone()
            if not namespace:
                raise LookupError("实体快照 namespace 不存在")
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

    def persist_grounded_entity_publishability_audit(
        self,
        namespace_id: str,
        rows: list[dict[str, Any]],
    ) -> int:
        """Attach a deterministic publication projection to canonicals.

        Raw mentions, assignments and embeddings remain immutable. Re-running
        this policy therefore costs bounded JSON metadata updates instead of a
        second extraction or identity-resolution pass.
        """

        updated = 0
        with self.connection() as conn:
            for item in rows:
                row = conn.execute(
                    """UPDATE entity_shadow_canonicals
                       SET metadata=jsonb_set(
                         COALESCE(metadata,'{}'::jsonb),
                         '{entity_publishability}',%s::jsonb,TRUE
                       )
                       WHERE namespace_id=%s AND canonical_key=%s
                       RETURNING canonical_key""",
                    (
                        self._json(item.get("audit") or {}),
                        namespace_id,
                        str(item["canonical_key"]),
                    ),
                ).fetchone()
                if not row:
                    raise RuntimeError(
                        "entity publishability audit references missing canonical: "
                        + str(item.get("canonical_key") or "")
                    )
                updated += 1
        return updated

    def grounded_entity_publishability_summary(
        self, namespace_id: str,
    ) -> dict[str, Any]:
        """Return persisted mention and rejection decision counters."""

        with self.connection() as conn:
            mention_rows = conn.execute(
                """SELECT filter_decision AS decision,filter_reason AS reason,
                          COUNT(*) AS count
                   FROM entity_shadow_mentions
                   WHERE namespace_id=%s
                   GROUP BY filter_decision,filter_reason
                   ORDER BY filter_decision,filter_reason""",
                (namespace_id,),
            ).fetchall()
            rejected_rows = conn.execute(
                """SELECT decision,reason,COUNT(*) AS count
                   FROM entity_shadow_filter_decisions
                   WHERE namespace_id=%s
                   GROUP BY decision,reason
                   ORDER BY decision,reason""",
                (namespace_id,),
            ).fetchall()
        decisions: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        for item in [*mention_rows, *rejected_rows]:
            count = int(item.get("count") or 0)
            decisions[str(item.get("decision") or "unknown")] += count
            reason = str(item.get("reason") or "")
            if reason:
                reasons[reason] += count
        return {
            "decisions": dict(sorted(decisions.items())),
            "reasons": dict(sorted(reasons.items())),
        }

    def search_knowledge_forge_entities(
        self, namespace_id: str, query: str, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search canonicals inside one immutable Knowledge Forge namespace."""

        query = str(query or "").strip()
        if not query:
            return []
        normalized = normalize_name(query)
        pattern = f"%{query}%"
        normalized_pattern = f"%{normalized}%"
        with self.connection() as conn:
            return conn.execute(
                """SELECT canonical.*,
                   GREATEST(
                     CASE WHEN canonical.name ILIKE %s THEN 1.0 ELSE 0.0 END,
                     CASE WHEN canonical.normalized_name ILIKE %s THEN 1.0 ELSE 0.0 END,
                     CASE WHEN COALESCE(canonical.metadata->>'description','') ILIKE %s
                          THEN 0.8 ELSE 0.0 END,
                     similarity(canonical.normalized_name,%s),
                     COALESCE(alias_score.score,0.0)
                   ) AS lexical_score
                   FROM entity_shadow_canonicals canonical
                   LEFT JOIN LATERAL (
                     SELECT MAX(GREATEST(
                       CASE WHEN alias.alias ILIKE %s THEN 1.0 ELSE 0.0 END,
                       CASE WHEN alias.normalized_alias ILIKE %s THEN 1.0 ELSE 0.0 END,
                       similarity(alias.normalized_alias,%s)
                     )) AS score
                     FROM entity_shadow_aliases alias
                     WHERE alias.namespace_id=canonical.namespace_id
                       AND alias.canonical_key=canonical.canonical_key
                   ) alias_score ON TRUE
                   WHERE canonical.namespace_id=%s
                     AND COALESCE(
                       (canonical.metadata#>>'{entity_publishability,retrieval_eligible}')::boolean,
                       TRUE
                     ) AND (
                     canonical.name ILIKE %s
                     OR canonical.normalized_name ILIKE %s
                     OR COALESCE(canonical.metadata->>'description','') ILIKE %s
                     OR COALESCE(alias_score.score,0.0)>0.15
                   )
                   ORDER BY lexical_score DESC,canonical.source_count DESC,
                            canonical.canonical_key
                   LIMIT %s""",
                (
                    pattern, normalized_pattern, pattern, normalized,
                    pattern, normalized_pattern, normalized,
                    namespace_id, pattern, normalized_pattern, pattern,
                    max(1, min(100, int(limit))),
                ),
            ).fetchall()

    def get_grounded_relation_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the immutable relation run referenced by a Knowledge Forge graph."""

        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM shadow_relation_v2_runs WHERE id=%s",
                (str(run_id),),
            ).fetchone()

    def start_relation_secondary_pass(
        self, values: dict[str, Any],
    ) -> dict[str, Any]:
        """Create one append-only, reason-routed child pass for a v2 run."""

        pass_id = str(values.get("id") or "")
        parent_run_id = str(values.get("parent_run_id") or "")
        fingerprint = str(values.get("input_fingerprint") or "")
        if not pass_id or not parent_run_id or not fingerprint:
            raise ValueError("secondary pass requires immutable parent and fingerprint")
        with self.connection() as conn:
            self._set_grounded_relation_session(conn, parent_run_id)
            parent = conn.execute(
                "SELECT project_id FROM shadow_relation_v2_runs WHERE id=%s",
                (parent_run_id,),
            ).fetchone()
            if not parent:
                raise LookupError("secondary pass parent relation run does not exist")
            project_id = int(values.get("project_id") or parent["project_id"])
            if project_id != int(parent["project_id"]):
                raise ValueError("secondary pass project mismatch")
            conn.execute(
                """INSERT INTO shadow_relation_secondary_passes(
                   id,parent_run_id,project_id,pass_version,input_fingerprint,status)
                   VALUES(%s,%s,%s,%s,%s,'planning')
                   ON CONFLICT(id) DO NOTHING""",
                (
                    pass_id, parent_run_id, project_id,
                    str(values.get("pass_version") or ""), fingerprint,
                ),
            )
            row = conn.execute(
                "SELECT * FROM shadow_relation_secondary_passes WHERE id=%s",
                (pass_id,),
            ).fetchone()
            if not row:
                raise RuntimeError("secondary pass could not be created")
            if (
                str(row["parent_run_id"]) != parent_run_id
                or int(row["project_id"]) != project_id
                or str(row["input_fingerprint"]) != fingerprint
            ):
                raise ValueError("secondary pass immutable input mismatch")
            if str(row.get("status") or "") in {"failed", "cancelled"}:
                row = conn.execute(
                    """UPDATE shadow_relation_secondary_passes
                       SET status='planning',route_counts='{}'::jsonb,
                           metrics='{}'::jsonb,error_message='',completed_at=NULL
                       WHERE id=%s RETURNING *""",
                    (pass_id,),
                ).fetchone()
            return row

    def persist_relation_secondary_candidates(
        self, pass_id: str, candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist secondary routing decisions without changing primary rows."""

        values = list(candidates)
        with self.connection() as conn:
            pass_row = conn.execute(
                "SELECT * FROM shadow_relation_secondary_passes WHERE id=%s FOR UPDATE",
                (str(pass_id),),
            ).fetchone()
            if not pass_row:
                raise LookupError("secondary pass does not exist")
            for item in values:
                pair = conn.execute(
                    """SELECT id,run_id FROM shadow_relation_pair_opportunities
                       WHERE id=%s""",
                    (str(item["pair_opportunity_id"]),),
                ).fetchone()
                if not pair or str(pair["run_id"]) != str(pass_row["parent_run_id"]):
                    raise ValueError("secondary candidate pair belongs to another run")
                parent_hypothesis_id = item.get("parent_hypothesis_id")
                if parent_hypothesis_id:
                    parent = conn.execute(
                        """SELECT hypothesis.id FROM shadow_relation_hypotheses hypothesis
                           JOIN shadow_relation_pair_opportunities pair
                             ON pair.id=hypothesis.pair_opportunity_id
                           WHERE hypothesis.id=%s AND pair.run_id=%s""",
                        (str(parent_hypothesis_id), str(pass_row["parent_run_id"])),
                    ).fetchone()
                    if not parent:
                        raise ValueError("secondary candidate parent hypothesis mismatch")
                conn.execute(
                    """INSERT INTO shadow_relation_secondary_candidates(
                       id,secondary_pass_id,parent_hypothesis_id,pair_opportunity_id,
                       route,source_reason_code,predicate_before,predicate_after,
                       subject_occurrence_key,object_occurrence_key,evidence_spans,
                       status,reason_code,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                       ON CONFLICT(id) DO UPDATE SET
                           predicate_after=EXCLUDED.predicate_after,
                           subject_occurrence_key=EXCLUDED.subject_occurrence_key,
                           object_occurrence_key=EXCLUDED.object_occurrence_key,
                           evidence_spans=EXCLUDED.evidence_spans,
                           status=EXCLUDED.status,reason_code=EXCLUDED.reason_code,
                           metadata=EXCLUDED.metadata""",
                    (
                        str(item["id"]), str(pass_id),
                        str(parent_hypothesis_id) if parent_hypothesis_id else None,
                        str(item["pair_opportunity_id"]), str(item["route"]),
                        str(item.get("source_reason_code") or ""),
                        str(item.get("predicate_before") or ""),
                        str(item.get("predicate_after") or ""),
                        str(item.get("subject_occurrence_key") or ""),
                        str(item.get("object_occurrence_key") or ""),
                        self._canonical_json(item.get("evidence_spans") or []),
                        str(item.get("status") or "PLANNED"),
                        str(item.get("reason_code") or ""),
                        self._canonical_json(item.get("metadata") or {}),
                    ),
                )
            return conn.execute(
                """SELECT * FROM shadow_relation_secondary_candidates
                   WHERE secondary_pass_id=%s ORDER BY id""",
                (str(pass_id),),
            ).fetchall()

    def record_relation_secondary_model_call(
        self, values: dict[str, Any],
    ) -> dict[str, Any]:
        """Audit one secondary recovery request in the parent run's call ledger."""

        with self.connection() as conn:
            self._set_grounded_relation_session(conn, str(values["run_id"]))
            request_payload = values.get("request_payload") or {}
            response_payload = values.get("parsed_response") or {}
            request_hash = str(values.get("request_hash") or "")
            response_hash = str(values.get("response_hash") or "")
            row = conn.execute(
                """INSERT INTO shadow_model_calls(
                   id,run_id,component,model_name,model_digest,prompt_version,
                   schema_version,parser_version,idempotency_key,attempt_no,status,
                   request_hash,response_hash,request_payload,raw_response_text,
                   parsed_response,token_usage,error_type,error_message,
                   independent_call,audit_quality,terminal_status)
                   VALUES(%s,%s,'secondary_recovery',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                          %s::jsonb,%s,%s::jsonb,%s::jsonb,%s,%s,TRUE,'complete',%s)
                   ON CONFLICT(run_id,component,idempotency_key,attempt_no)
                   DO UPDATE SET terminal_status=EXCLUDED.terminal_status
                   WHERE shadow_model_calls.request_hash=EXCLUDED.request_hash
                   RETURNING *""",
                (
                    str(values["id"]), str(values["run_id"]),
                    str(values.get("model_name") or ""), str(values.get("model_digest") or ""),
                    str(values.get("prompt_version") or ""),
                    str(values.get("schema_version") or ""),
                    str(values.get("parser_version") or ""),
                    str(values.get("idempotency_key") or ""),
                    int(values.get("attempt_no") or 1), str(values.get("status") or ""),
                    request_hash, response_hash,
                    self._canonical_json(request_payload),
                    str(values.get("raw_response_text") or ""),
                    self._canonical_json(response_payload),
                    self._canonical_json(values.get("token_usage") or {}),
                    str(values.get("error_type") or ""),
                    str(values.get("error_message") or ""),
                    str(values.get("terminal_status") or str(values.get("status") or "")),
                ),
            ).fetchone()
            if not row:
                raise ValueError("secondary model call immutable payload mismatch")
            return row

    def finish_relation_secondary_pass(
        self, pass_id: str, *, status: str, metrics: dict[str, Any],
        route_counts: dict[str, Any], error_message: str = "",
    ) -> dict[str, Any]:
        if status not in {"ready", "failed", "cancelled"}:
            raise ValueError("invalid secondary pass terminal status")
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE shadow_relation_secondary_passes
                   SET status=%s,metrics=%s::jsonb,route_counts=%s::jsonb,
                       error_message=%s,completed_at=NOW()
                   WHERE id=%s RETURNING *""",
                (
                    status, self._canonical_json(metrics),
                    self._canonical_json(route_counts), error_message or "",
                    str(pass_id),
                ),
            ).fetchone()
            if not row:
                raise LookupError("secondary pass does not exist")
            return row

    def get_relation_secondary_pass(self, pass_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM shadow_relation_secondary_passes WHERE id=%s",
                (str(pass_id),),
            ).fetchone()

    def list_agent_knowledge_forge_community_relation_candidates(
        self,
        namespace_id: str,
        relation_run_id: str,
        entity_keys: list[str],
        *,
        scope: str,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        """Read a bounded, lightweight relation index for community routing.

        This is not an evidence endpoint.  It only helps the Knowledge Forge
        agent choose relation IDs from the current immutable snapshot.  Every
        selected ID is subsequently re-read through
        :meth:`search_knowledge_forge_relations`, whose evidence joins and
        source checks remain the publication gate.
        """

        if scope not in {"cross", "internal"}:
            raise ValueError("community relation scope must be cross or internal")
        keys = sorted({str(value) for value in entity_keys if str(value)})[:5_000]
        if not keys:
            return []
        membership_clause = (
            "((endpoint.source_canonical_key=ANY(%s::text[])) <> "
            "(endpoint.target_canonical_key=ANY(%s::text[])))"
            if scope == "cross"
            else "(endpoint.source_canonical_key=ANY(%s::text[]) "
            "AND endpoint.target_canonical_key=ANY(%s::text[]))"
        )
        with self.connection() as conn:
            return conn.execute(
                """SELECT endpoint.relation_key AS id,
                          endpoint.source_canonical_key,source.name AS source_name,
                          endpoint.relation_type,
                          endpoint.target_canonical_key,target.name AS target_name,
                          endpoint.document_id,endpoint.chunk_id,
                          endpoint.evidence_text,endpoint.metadata
                   FROM entity_shadow_relation_endpoints endpoint
                   JOIN entity_shadow_canonicals source
                     ON source.namespace_id=endpoint.namespace_id
                    AND source.canonical_key=endpoint.source_canonical_key
                   JOIN entity_shadow_canonicals target
                     ON target.namespace_id=endpoint.namespace_id
                    AND target.canonical_key=endpoint.target_canonical_key
                   JOIN shadow_relation_normalized_facts fact
                     ON fact.id=endpoint.relation_key
                    AND fact.subject_canonical_id=endpoint.source_canonical_key
                    AND fact.predicate=endpoint.relation_type
                    AND fact.object_canonical_id=endpoint.target_canonical_key
                    AND fact.is_review_only=FALSE
                   JOIN shadow_relation_fact_memberships membership
                     ON membership.fact_id=fact.id AND membership.run_id=%s
                    AND membership.status='ELIGIBLE'
                   WHERE endpoint.namespace_id=%s
                     AND endpoint.metadata->>'shadow_relation_v2_run_id'=%s
                     AND """ + membership_clause + """
                   ORDER BY endpoint.relation_key
                   LIMIT %s""",
                (
                    str(relation_run_id), str(namespace_id),
                    str(relation_run_id), keys, keys,
                    max(1, min(500, int(limit))),
                ),
            ).fetchall()

    def search_knowledge_forge_relations(
        self,
        namespace_id: str,
        relation_run_id: str,
        query: str,
        *,
        chunk_ids: list[int] | None = None,
        canonical_keys: list[str] | None = None,
        relation_keys: list[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search verified facts, optionally fanning out from entity seeds.

        ``canonical_keys`` and ``relation_keys`` are deliberately bounded and
        only match the immutable namespace.  The latter lets an upstream
        community navigator name candidate facts, but those IDs still pass
        this method's eligible-membership, independent-verification, active
        document and active chunk joins before they can become evidence.
        """

        query = str(query or "").strip()
        if not query:
            return []
        pattern = f"%{query}%"
        chunk_ids = sorted({int(value) for value in (chunk_ids or [])})
        canonical_keys = sorted({
            str(value) for value in (canonical_keys or []) if str(value)
        })[:50]
        relation_keys = sorted({
            str(value) for value in (relation_keys or []) if str(value)
        })[:50]
        chunk_clause = (
            " OR endpoint.chunk_id=ANY(%s)"
            if chunk_ids else ""
        )
        params: list[Any] = [
            canonical_keys, canonical_keys,
            relation_keys,
            pattern, pattern, pattern, pattern, pattern, pattern,
            query, query, query, query, query, query,
            relation_run_id, relation_run_id,
            namespace_id, relation_run_id,
            pattern, pattern, pattern, pattern, pattern, pattern,
            canonical_keys, canonical_keys,
            relation_keys,
        ]
        if chunk_ids:
            params.append(chunk_ids)
        params.append(max(1, min(100, int(limit))))
        with self.connection() as conn:
            return conn.execute(
                """SELECT endpoint.relation_key AS id,endpoint.namespace_id,
                   endpoint.document_id,endpoint.revision_id,endpoint.chunk_id,
                   endpoint.source_canonical_key,source.name AS source_name,
                   endpoint.relation_type,endpoint.target_canonical_key,
                   target.name AS target_name,endpoint.evidence_text,
                   endpoint.evidence_start,endpoint.evidence_end,endpoint.metadata,
                   fact.semantic_fact_key,fact.qualifiers,fact.audit_quality,
                   evidence.evidence_count,evidence.evidence_records,
                   document.original_filename,document.storage_path,chunk.chunk_key,
                   CASE WHEN endpoint.source_canonical_key=ANY(%s::text[])
                          OR endpoint.target_canonical_key=ANY(%s::text[])
                          OR endpoint.relation_key=ANY(%s::text[])
                        THEN 1 ELSE 0 END AS seed_match,
                   GREATEST(
                     CASE WHEN source.name ILIKE %s THEN 1.0 ELSE 0.0 END,
                     CASE WHEN target.name ILIKE %s THEN 1.0 ELSE 0.0 END,
                     CASE WHEN endpoint.relation_type ILIKE %s THEN 1.0 ELSE 0.0 END,
                     CASE WHEN endpoint.evidence_text ILIKE %s THEN 0.9 ELSE 0.0 END,
                     CASE WHEN endpoint.metadata->>'display_relation_label' ILIKE %s
                          THEN 1.0 ELSE 0.0 END,
                     CASE WHEN endpoint.metadata->>'source_relation_expression' ILIKE %s
                          THEN 1.0 ELSE 0.0 END,
                     similarity(LOWER(source.name),LOWER(%s)),
                     similarity(LOWER(target.name),LOWER(%s)),
                     similarity(LOWER(endpoint.relation_type),LOWER(%s)),
                     similarity(LOWER(endpoint.evidence_text),LOWER(%s)),
                     similarity(
                       LOWER(COALESCE(endpoint.metadata->>'display_relation_label','')),
                       LOWER(%s)
                     ),
                     similarity(
                       LOWER(COALESCE(endpoint.metadata->>'source_relation_expression','')),
                       LOWER(%s)
                     )
                   ) AS lexical_score
                   FROM entity_shadow_relation_endpoints endpoint
                   JOIN entity_shadow_canonicals source
                     ON source.namespace_id=endpoint.namespace_id
                    AND source.canonical_key=endpoint.source_canonical_key
                   JOIN entity_shadow_canonicals target
                     ON target.namespace_id=endpoint.namespace_id
                    AND target.canonical_key=endpoint.target_canonical_key
                   JOIN shadow_relation_normalized_facts fact
                     ON fact.id=endpoint.relation_key
                    AND fact.subject_canonical_id=endpoint.source_canonical_key
                    AND fact.predicate=endpoint.relation_type
                    AND fact.object_canonical_id=endpoint.target_canonical_key
                    AND fact.is_review_only=FALSE
                   JOIN shadow_relation_fact_memberships membership
                     ON membership.fact_id=fact.id AND membership.run_id=%s
                    AND membership.status='ELIGIBLE'
                   JOIN LATERAL (
                     SELECT COUNT(*) AS evidence_count,
                       JSONB_AGG(JSONB_BUILD_OBJECT(
                         'id',item.id,'evidence_key',item.evidence_key,
                         'evidence_spans',item.evidence_spans,
                         'document_revision_id',item.document_revision_id,
                         'verification_attempt_id',item.verification_attempt_id,
                         'provenance',item.provenance
                       ) ORDER BY item.evidence_key) AS evidence_records,
                       BOOL_AND(verification.verdict='SUPPORTED'
                         AND verification.independent_call=TRUE) AS all_supported
                     FROM shadow_relation_fact_evidence item
                     JOIN shadow_relation_verification_attempts verification
                       ON verification.id=item.verification_attempt_id
                     WHERE item.run_id=%s AND item.fact_id=fact.id
                   ) evidence ON evidence.evidence_count>0
                              AND evidence.all_supported=TRUE
                   JOIN documents document
                     ON document.id=endpoint.document_id
                    AND document.project_id=fact.project_id
                    AND document.active_revision_id=endpoint.revision_id
                    AND document.is_deleted=FALSE
                   JOIN chunks chunk
                     ON chunk.id=endpoint.chunk_id
                    AND chunk.revision_id=endpoint.revision_id
                    AND chunk.is_deleted=FALSE
                   WHERE endpoint.namespace_id=%s
                     AND endpoint.metadata->>'shadow_relation_v2_run_id'=%s
                     AND (
                       source.name ILIKE %s OR target.name ILIKE %s
                       OR endpoint.relation_type ILIKE %s
                       OR endpoint.evidence_text ILIKE %s
                       OR endpoint.metadata->>'display_relation_label' ILIKE %s
                       OR endpoint.metadata->>'source_relation_expression' ILIKE %s
                       OR endpoint.source_canonical_key=ANY(%s::text[])
                       OR endpoint.target_canonical_key=ANY(%s::text[])
                       OR endpoint.relation_key=ANY(%s::text[])"""
                + chunk_clause
                + """)
                   ORDER BY seed_match DESC,lexical_score DESC,
                            evidence.evidence_count DESC,
                            endpoint.relation_key
                   LIMIT %s""",
                tuple(params),
            ).fetchall()

    def persist_grounded_entity_snapshot(
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
                raise LookupError("实体快照 namespace 不存在")
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
            entities = conn.execute(
                """SELECT entity.id,entity.entity_key,entity.name,entity.entity_type,
                   entity.description,
                   ARRAY_AGG(DISTINCT 'chunk:' || mention.revision_id || ':' || chunk.chunk_key)
                       AS text_unit_ids
                   FROM canonical_entities entity
                   JOIN entity_mentions mention
                     ON mention.canonical_entity_id=entity.id
                    AND mention.project_id=entity.project_id AND mention.is_deleted=FALSE
                   JOIN documents document
                     ON document.id=mention.document_id
                    AND document.project_id=entity.project_id
                    AND document.active_revision_id=mention.revision_id
                    AND document.is_deleted=FALSE
                   JOIN chunks chunk
                     ON chunk.id=mention.chunk_id AND chunk.revision_id=mention.revision_id
                    AND chunk.is_deleted=FALSE
                   WHERE entity.project_id=%s AND entity.is_deleted=FALSE
                   GROUP BY entity.id,entity.entity_key,entity.name,entity.entity_type,
                            entity.description
                   ORDER BY entity.id""",
                (project_id,),
            ).fetchall()
            return ControlGraphSnapshot(
                project_id=int(project["id"]),
                graph_generation=int(project["graph_generation"]),
                graph_policy_hash=str(project.get("graph_policy_hash") or ""),
                active_policy_hash=str(project.get("active_policy_hash") or ""),
                relations=tuple(dict(item) for item in relations),
                entities=tuple(dict(item) for item in entities),
            )

    def get_knowledge_forge_community_run(
        self,
        project_id: int,
        graph_generation: int,
        entity_namespace_id: str,
        *,
        input_hash: str = "",
    ) -> dict[str, Any] | None:
        clauses = [
            "project_id=%s", "strategy_key=%s", "graph_generation=%s",
            "entity_namespace_id=%s",
        ]
        params: list[Any] = [
            int(project_id), KNOWLEDGE_FORGE_STRATEGY,
            int(graph_generation), str(entity_namespace_id),
        ]
        if input_hash:
            clauses.append("input_hash=%s")
            params.append(str(input_hash))
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM knowledge_forge_community_runs
                   WHERE """ + " AND ".join(clauses) + """
                     AND status IN ('completed','completed_empty')
                   ORDER BY created_at DESC LIMIT 1""",
                tuple(params),
            ).fetchone()

    def create_knowledge_forge_community_run(
        self,
        *,
        run_id: str,
        project_id: int,
        graph_generation: int,
        entity_namespace_id: str,
        input_hash: str,
        node_count: int,
        edge_count: int,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """INSERT INTO knowledge_forge_community_runs(
                   run_id,project_id,strategy_key,graph_generation,
                   entity_namespace_id,input_hash,status,node_count,edge_count,config)
                   VALUES(%s,%s,%s,%s,%s,%s,'running',%s,%s,%s::jsonb)
                   ON CONFLICT(
                     project_id,strategy_key,graph_generation,entity_namespace_id,input_hash
                   ) DO NOTHING RETURNING *""",
                (
                    str(run_id), int(project_id), KNOWLEDGE_FORGE_STRATEGY,
                    int(graph_generation), str(entity_namespace_id), str(input_hash),
                    int(node_count), int(edge_count), self._json(config),
                ),
            ).fetchone()
            if row:
                return row
            existing = conn.execute(
                """SELECT * FROM knowledge_forge_community_runs
                   WHERE project_id=%s AND strategy_key=%s AND graph_generation=%s
                     AND entity_namespace_id=%s AND input_hash=%s
                   ORDER BY created_at DESC LIMIT 1""",
                (
                    int(project_id), KNOWLEDGE_FORGE_STRATEGY,
                    int(graph_generation), str(entity_namespace_id), str(input_hash),
                ),
            ).fetchone()
            if not existing:
                raise RuntimeError("Knowledge Forge 社区运行幂等创建失败")
            return existing

    def complete_knowledge_forge_community_run(
        self,
        run_id: str,
        *,
        status: str,
        communities: list[dict[str, Any]],
        hierarchy_levels: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if status not in {"completed", "completed_empty"}:
            raise ValueError("非法 Knowledge Forge 社区完成状态")
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE knowledge_forge_community_runs
                   SET status=%s,community_count=%s,hierarchy_levels=%s,
                       communities=%s::jsonb,metadata=%s::jsonb,
                       error_message='',completed_at=NOW()
                   WHERE run_id=%s AND status='running' RETURNING *""",
                (
                    status, len(communities), int(hierarchy_levels),
                    self._json(communities), self._json(metadata), str(run_id),
                ),
            ).fetchone()
            if row:
                return row
            existing = conn.execute(
                "SELECT * FROM knowledge_forge_community_runs WHERE run_id=%s",
                (str(run_id),),
            ).fetchone()
            if (
                existing
                and existing["status"] == status
                and int(existing["community_count"]) == len(communities)
            ):
                return existing
            raise RuntimeError("Knowledge Forge 社区运行已终止或完成内容不一致")

    def fail_knowledge_forge_community_run(
        self, run_id: str, error: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """UPDATE knowledge_forge_community_runs
                   SET status='failed',error_message=%s,completed_at=NOW()
                   WHERE run_id=%s AND status='running' RETURNING *""",
                (str(error)[-4000:], str(run_id)),
            ).fetchone() or {}

    def get_matching_knowledge_forge_community_report_run(
        self,
        *,
        project_id: int,
        community_run_id: str,
        model_id: str,
        prompt_hash: str,
        config_hash: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE project_id=%s AND community_run_id=%s AND model_id=%s
                     AND prompt_hash=%s AND config_hash=%s
                   ORDER BY retry_round DESC,created_at DESC LIMIT 1""",
                (
                    int(project_id), str(community_run_id), str(model_id),
                    str(prompt_hash), str(config_hash),
                ),
            ).fetchone()

    def create_knowledge_forge_community_report_build(
        self,
        values: dict[str, Any],
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Atomically persist one immutable report plan and its durable job."""

        with self.connection() as conn:
            community_run = conn.execute(
                """SELECT * FROM knowledge_forge_community_runs
                   WHERE run_id=%s FOR SHARE""",
                (str(values["community_run_id"]),),
            ).fetchone()
            if (
                not community_run
                or int(community_run["project_id"]) != int(values["project_id"])
                or str(community_run.get("status") or "")
                not in {"completed", "completed_empty"}
                or int(community_run["graph_generation"])
                != int(values["graph_generation"])
                or str(community_run["entity_namespace_id"])
                != str(values["entity_namespace_id"])
                or str(community_run["input_hash"])
                != str(values["community_input_hash"])
            ):
                raise RuntimeError("社区报告输入不是完整且匹配的社区快照")
            existing = conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE project_id=%s AND community_run_id=%s AND model_id=%s
                     AND prompt_hash=%s AND config_hash=%s AND retry_round=%s
                   FOR SHARE""",
                (
                    int(values["project_id"]), str(values["community_run_id"]),
                    str(values["model_id"]), str(values["prompt_hash"]),
                    str(values["config_hash"]), int(values.get("retry_round") or 0),
                ),
            ).fetchone()
            if existing:
                immutable = (
                    "graph_generation", "entity_namespace_id",
                    "community_input_hash", "input_hash", "embedding_model_id",
                    "prompt_version",
                )
                if any(str(existing[key]) != str(values[key]) for key in immutable):
                    raise RuntimeError("社区报告运行幂等键对应的不可变输入不一致")
                job = (
                    conn.execute(
                        "SELECT * FROM jobs WHERE id=%s",
                        (str(existing.get("job_id") or ""),),
                    ).fetchone()
                    if existing.get("job_id") else None
                )
                return {"run": existing, "job": job, "_coalesced": True}

            job_id = str(values["job_id"])
            job = conn.execute(
                """INSERT INTO jobs(
                       id,kind,project_id,metadata,strategy_key
                   ) VALUES(%s,'community_report_build',%s,%s::jsonb,%s)
                   RETURNING *""",
                (
                    job_id, int(values["project_id"]),
                    self._json({
                        "stage": "queued",
                        "report_run_id": str(values["run_id"]),
                        "community_run_id": str(values["community_run_id"]),
                        "graph_generation": int(values["graph_generation"]),
                        "automatic": bool(values.get("automatic")),
                    }),
                    KNOWLEDGE_FORGE_STRATEGY,
                ),
            ).fetchone()
            run = conn.execute(
                """INSERT INTO knowledge_forge_community_report_runs(
                       run_id,project_id,community_run_id,graph_generation,
                       entity_namespace_id,community_input_hash,input_hash,
                       model_id,embedding_model_id,prompt_version,prompt_hash,
                       config_hash,retry_round,status,expected_count,eligible_count,
                       skipped_count,job_id,config)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'queued',
                          %s,%s,%s,%s,%s::jsonb)
                   RETURNING *""",
                (
                    str(values["run_id"]), int(values["project_id"]),
                    str(values["community_run_id"]), int(values["graph_generation"]),
                    str(values["entity_namespace_id"]),
                    str(values["community_input_hash"]), str(values["input_hash"]),
                    str(values["model_id"]), str(values["embedding_model_id"]),
                    str(values["prompt_version"]), str(values["prompt_hash"]),
                    str(values["config_hash"]), int(values.get("retry_round") or 0),
                    int(values.get("expected_count") or 0),
                    int(values.get("eligible_count") or 0),
                    int(values.get("skipped_count") or 0), job_id,
                    self._json(values.get("config") or {}),
                ),
            ).fetchone()
            rows = []
            for item in items:
                status = str(item["status"])
                rows.append({
                    "report_id": str(item["report_id"]),
                    "run_id": str(values["run_id"]),
                    "project_id": int(values["project_id"]),
                    "community_run_id": str(values["community_run_id"]),
                    "community_key": str(item["community_key"]),
                    "community_content_hash": str(item["community_content_hash"]),
                    "context_hash": str(item["context_hash"]),
                    "level": int(item.get("level") or 0),
                    "status": status,
                    "priority": int(item.get("priority") or 0),
                    "max_attempts": int(item.get("max_attempts") or 2),
                    "entity_keys": list(item.get("entity_keys") or []),
                    "relation_keys": list(item.get("relation_keys") or []),
                    "input_context": item.get("input_context") or {},
                    "embedding_status": (
                        "not_requested" if status == "skipped" else "pending"
                    ),
                    "error_category": (
                        "Deferred" if bool(item.get("deferred")) else ""
                    ),
                    "error_message": str(item.get("skip_reason") or ""),
                })
            if rows:
                conn.execute(
                    """INSERT INTO knowledge_forge_community_reports(
                           report_id,run_id,project_id,community_run_id,
                           community_key,community_content_hash,context_hash,level,
                           status,priority,max_attempts,entity_keys,relation_keys,
                           input_context,embedding_status,error_category,
                           error_message,completed_at)
                       SELECT item.report_id,item.run_id,item.project_id,
                              item.community_run_id,item.community_key,
                              item.community_content_hash,item.context_hash,item.level,
                              item.status,item.priority,item.max_attempts,
                              item.entity_keys,item.relation_keys,item.input_context,
                              item.embedding_status,item.error_category,
                              item.error_message,
                              CASE WHEN item.status='skipped' THEN NOW() ELSE NULL END
                       FROM jsonb_to_recordset(%s::jsonb) AS item(
                           report_id TEXT,run_id TEXT,project_id BIGINT,
                           community_run_id TEXT,community_key TEXT,
                           community_content_hash TEXT,context_hash TEXT,level INTEGER,
                           status TEXT,priority INTEGER,max_attempts INTEGER,
                           entity_keys TEXT[],relation_keys TEXT[],input_context JSONB,
                           embedding_status TEXT,error_category TEXT,
                           error_message TEXT
                       )""",
                    (self._json(rows),),
                )
            return {"run": run, "job": job, "_coalesced": False}

    def start_knowledge_forge_community_report_run(
        self, run_id: str,
    ) -> dict[str, Any]:
        """Resume a report run and close attempts interrupted by a restart."""

        with self.connection() as conn:
            run = conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE run_id=%s FOR UPDATE""",
                (str(run_id),),
            ).fetchone()
            if not run:
                raise LookupError("社区报告运行不存在")
            if str(run["status"]) not in {"queued", "running"}:
                return run
            conn.execute(
                """UPDATE knowledge_forge_community_report_attempts
                   SET status='failed',error_category='ServiceRestart',
                       error_message='服务重启中断了本次模型调用',completed_at=NOW()
                   WHERE run_id=%s AND status='running'""",
                (str(run_id),),
            )
            conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET status=CASE WHEN attempt_count<max_attempts
                                   THEN 'pending' ELSE 'failed' END,
                       error_category=CASE WHEN attempt_count<max_attempts
                                           THEN '' ELSE 'ServiceRestart' END,
                       error_message=CASE WHEN attempt_count<max_attempts
                                         THEN '' ELSE '重启后重试预算已耗尽' END,
                       updated_at=NOW(),
                       completed_at=CASE WHEN attempt_count<max_attempts
                                         THEN NULL ELSE NOW() END
                   WHERE run_id=%s AND status='running'""",
                (str(run_id),),
            )
            return conn.execute(
                """UPDATE knowledge_forge_community_report_runs
                   SET status='running',started_at=COALESCE(started_at,NOW()),
                       error_message=''
                   WHERE run_id=%s RETURNING *""",
                (str(run_id),),
            ).fetchone()

    def claim_knowledge_forge_community_report(
        self, run_id: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT report_id FROM knowledge_forge_community_reports
                   WHERE run_id=%s AND status='pending'
                     AND attempt_count<max_attempts
                   ORDER BY priority DESC,level,community_key
                   LIMIT 1 FOR UPDATE SKIP LOCKED""",
                (str(run_id),),
            ).fetchone()
            if not row:
                return None
            return conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET status='running',
                       error_category='',error_message='',updated_at=NOW(),
                       completed_at=NULL
                   WHERE report_id=%s RETURNING *""",
                (str(row["report_id"]),),
            ).fetchone()

    def begin_knowledge_forge_community_report_model_attempt(
        self, report_id: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET attempt_count=attempt_count+1,updated_at=NOW()
                   WHERE report_id=%s AND status='running'
                     AND attempt_count<max_attempts RETURNING *""",
                (str(report_id),),
            ).fetchone()
            if not row:
                raise RuntimeError("社区报告模型重试预算已耗尽")
            return row

    def find_reusable_knowledge_forge_community_report(
        self, run_id: str, report_id: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            current = conn.execute(
                """SELECT report.*,run.model_id,run.prompt_hash,run.config_hash
                   FROM knowledge_forge_community_reports report
                   JOIN knowledge_forge_community_report_runs run
                     ON run.run_id=report.run_id
                   WHERE report.report_id=%s AND report.run_id=%s""",
                (str(report_id), str(run_id)),
            ).fetchone()
            if not current:
                raise LookupError("社区报告工作项不存在")
            return conn.execute(
                """SELECT prior.* FROM knowledge_forge_community_reports prior
                   JOIN knowledge_forge_community_report_runs prior_run
                     ON prior_run.run_id=prior.run_id
                   WHERE prior.project_id=%s AND prior.report_id<>%s
                     AND prior.community_content_hash=%s
                     AND prior.context_hash=%s
                     AND prior.status IN ('succeeded','reused')
                     AND prior.content<>''
                     AND prior_run.model_id=%s
                     AND prior_run.prompt_hash=%s
                     AND prior_run.config_hash=%s
                   ORDER BY prior.completed_at DESC NULLS LAST,prior.updated_at DESC
                   LIMIT 1""",
                (
                    int(current["project_id"]), str(report_id),
                    str(current["community_content_hash"]),
                    str(current["context_hash"]), str(current["model_id"]),
                    str(current["prompt_hash"]), str(current["config_hash"]),
                ),
            ).fetchone()

    def reuse_knowledge_forge_community_report(
        self, report_id: str, source_report_id: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            source = conn.execute(
                """SELECT * FROM knowledge_forge_community_reports
                   WHERE report_id=%s AND status IN ('succeeded','reused')""",
                (str(source_report_id),),
            ).fetchone()
            current = conn.execute(
                """SELECT * FROM knowledge_forge_community_reports
                   WHERE report_id=%s FOR UPDATE""",
                (str(report_id),),
            ).fetchone()
            if not source or not current:
                raise LookupError("可复用社区报告或当前工作项不存在")
            if (
                str(source["community_content_hash"])
                != str(current["community_content_hash"])
                or str(source["context_hash"]) != str(current["context_hash"])
            ):
                raise RuntimeError("社区报告复用内容哈希不一致")
            return conn.execute(
                """UPDATE knowledge_forge_community_reports SET
                       status='reused',title=%s,summary=%s,findings=%s::jsonb,
                       limitations=%s,content=%s,output_hash=%s,
                       cited_entity_keys=%s,cited_relation_keys=%s,
                       source_report_id=%s,usage=%s::jsonb,latency_ms=0,
                       embedding_status='pending',embedding_error='',
                       error_category='',error_message='',updated_at=NOW(),
                       completed_at=NOW()
                   WHERE report_id=%s AND status='running' RETURNING *""",
                (
                    source["title"], source["summary"], self._json(source["findings"]),
                    source["limitations"], source["content"], source["output_hash"],
                    source["cited_entity_keys"], source["cited_relation_keys"],
                    str(source_report_id), self._json(source.get("usage") or {}),
                    str(report_id),
                ),
            ).fetchone()

    def record_knowledge_forge_community_report_attempt_start(
        self,
        *,
        attempt_id: str,
        report_id: str,
        run_id: str,
        attempt_no: int,
        model_id: str,
        prompt_version: str,
        request_hash: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO knowledge_forge_community_report_attempts(
                       attempt_id,report_id,run_id,attempt_no,model_id,
                       prompt_version,request_hash,status)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,'running')
                   ON CONFLICT(report_id,attempt_no) DO UPDATE SET
                       status='running',request_hash=EXCLUDED.request_hash,
                       raw_response_text='',parsed_response='{}'::jsonb,
                       diagnostics='{}'::jsonb,latency_ms=0,error_category='',
                       error_message='',completed_at=NULL
                   RETURNING *""",
                (
                    str(attempt_id), str(report_id), str(run_id), int(attempt_no),
                    str(model_id), str(prompt_version), str(request_hash),
                ),
            ).fetchone()

    def complete_knowledge_forge_community_report_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        raw_response_text: str = "",
        parsed_response: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
        latency_ms: int = 0,
        error_category: str = "",
        error_message: str = "",
    ) -> dict[str, Any]:
        if status not in {"succeeded", "failed"}:
            raise ValueError("非法社区报告尝试状态")
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE knowledge_forge_community_report_attempts
                   SET status=%s,raw_response_text=%s,parsed_response=%s::jsonb,
                       diagnostics=%s::jsonb,latency_ms=%s,error_category=%s,
                       error_message=%s,completed_at=NOW()
                   WHERE attempt_id=%s AND status='running' RETURNING *""",
                (
                    status, str(raw_response_text)[-100_000:],
                    self._json(parsed_response or {}),
                    self._json(diagnostics or {}), max(0, int(latency_ms)),
                    str(error_category), str(error_message)[-4000:],
                    str(attempt_id),
                ),
            ).fetchone()
            if not row:
                raise RuntimeError("社区报告尝试不存在或已完成")
            return row

    def complete_knowledge_forge_community_report(
        self, report_id: str, values: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE knowledge_forge_community_reports SET
                       status='succeeded',title=%s,summary=%s,findings=%s::jsonb,
                       limitations=%s,content=%s,output_hash=%s,
                       cited_entity_keys=%s,cited_relation_keys=%s,
                       usage=%s::jsonb,latency_ms=%s,error_category='',
                       error_message='',embedding_status='pending',
                       embedding_error='',updated_at=NOW(),completed_at=NOW()
                   WHERE report_id=%s AND status='running' RETURNING *""",
                (
                    str(values.get("title") or ""),
                    str(values.get("summary") or ""),
                    self._json(values.get("findings") or []),
                    str(values.get("limitations") or ""),
                    str(values.get("content") or ""),
                    str(values.get("output_hash") or ""),
                    list(values.get("cited_entity_keys") or []),
                    list(values.get("cited_relation_keys") or []),
                    self._json(values.get("usage") or {}),
                    max(0, int(values.get("latency_ms") or 0)), str(report_id),
                ),
            ).fetchone()
            if not row:
                raise RuntimeError("社区报告工作项已结束，拒绝覆盖")
            return row

    def fail_or_retry_knowledge_forge_community_report(
        self,
        report_id: str,
        *,
        error_category: str,
        error_message: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            current = conn.execute(
                """SELECT * FROM knowledge_forge_community_reports
                   WHERE report_id=%s FOR UPDATE""",
                (str(report_id),),
            ).fetchone()
            if not current:
                raise LookupError("社区报告工作项不存在")
            retry = int(current["attempt_count"]) < int(current["max_attempts"])
            return conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET status=%s,error_category=%s,error_message=%s,
                       updated_at=NOW(),completed_at=%s
                   WHERE report_id=%s AND status='running' RETURNING *""",
                (
                    "pending" if retry else "failed", str(error_category),
                    str(error_message)[-4000:], None if retry else datetime.now(timezone.utc),
                    str(report_id),
                ),
            ).fetchone()

    def set_knowledge_forge_community_report_embedding_status(
        self, report_id: str, *, status: str, error: str = "",
    ) -> dict[str, Any]:
        if status not in {"succeeded", "failed", "not_requested"}:
            raise ValueError("非法社区报告向量状态")
        with self.connection() as conn:
            return conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET embedding_status=%s,embedding_error=%s,updated_at=NOW()
                   WHERE report_id=%s RETURNING *""",
                (status, str(error)[-2000:], str(report_id)),
            ).fetchone()

    def finalize_knowledge_forge_community_report_run(
        self, run_id: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE run_id=%s FOR UPDATE""",
                (str(run_id),),
            ).fetchone()
            if not run:
                raise LookupError("社区报告运行不存在")
            if str(run["status"]) in {"superseded", "cancelled"}:
                return run
            counts = conn.execute(
                """SELECT
                     COUNT(*) FILTER(WHERE status IN ('succeeded','reused')) AS completed,
                     COUNT(*) FILTER(WHERE status='reused') AS reused,
                     COUNT(*) FILTER(WHERE status='failed') AS failed,
                     COUNT(*) FILTER(WHERE status='skipped') AS skipped,
                     COUNT(*) FILTER(WHERE status IN ('pending','running')) AS pending
                   FROM knowledge_forge_community_reports WHERE run_id=%s""",
                (str(run_id),),
            ).fetchone()
            completed = int(counts["completed"] or 0)
            reused = int(counts["reused"] or 0)
            failed = int(counts["failed"] or 0)
            skipped = int(counts["skipped"] or 0)
            pending = int(counts["pending"] or 0)
            if pending:
                # An on-demand activation can win the run-row lock immediately
                # after the worker observed an empty queue.  Leave the run
                # active so the worker can claim the newly pending item rather
                # than failing a healthy, serialized race.
                return {**run, "_pending": pending}
            eligible = int(run["eligible_count"] or 0)
            if eligible == 0:
                status = "completed_empty"
            elif failed == 0 and completed == eligible:
                status = "completed"
            elif completed:
                status = "partial"
            else:
                status = "failed"
            error = (
                f"{failed} 个社区报告生成失败" if failed else ""
            )
            return conn.execute(
                """UPDATE knowledge_forge_community_report_runs SET
                       status=%s,completed_count=%s,reused_count=%s,
                       failed_count=%s,skipped_count=%s,
                       metrics=%s::jsonb,error_message=%s,completed_at=NOW()
                   WHERE run_id=%s RETURNING *""",
                (
                    status, completed, reused, failed, skipped,
                    self._json({
                        "coverage": (
                            completed / eligible if eligible else 1.0
                        ),
                        "pending": pending,
                    }),
                    error, str(run_id),
                ),
            ).fetchone()

    def terminate_knowledge_forge_community_report_run(
        self, run_id: str, *, status: str, reason: str,
    ) -> dict[str, Any]:
        if status not in {"cancelled", "superseded", "failed"}:
            raise ValueError("非法社区报告终止状态")
        with self.connection() as conn:
            conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET status=%s,error_message=%s,updated_at=NOW(),completed_at=NOW()
                   WHERE run_id=%s AND status IN ('pending','running')""",
                (status, str(reason)[-4000:], str(run_id)),
            )
            return conn.execute(
                """UPDATE knowledge_forge_community_report_runs
                   SET status=%s,error_message=%s,completed_at=NOW()
                   WHERE run_id=%s AND status IN ('queued','running') RETURNING *""",
                (status, str(reason)[-4000:], str(run_id)),
            ).fetchone() or conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE run_id=%s""", (str(run_id),),
            ).fetchone()

    def get_knowledge_forge_community_report_state(
        self,
        *,
        project_id: int,
        community_run_id: str,
        graph_generation: int,
        entity_namespace_id: str,
        community_input_hash: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE project_id=%s AND community_run_id=%s
                     AND graph_generation=%s AND entity_namespace_id=%s
                     AND community_input_hash=%s
                   ORDER BY retry_round DESC,created_at DESC LIMIT 1""",
                (
                    int(project_id), str(community_run_id), int(graph_generation),
                    str(entity_namespace_id), str(community_input_hash),
                ),
            ).fetchone()
            reports = (
                conn.execute(
                    """SELECT * FROM knowledge_forge_community_reports
                       WHERE run_id=%s ORDER BY level,community_key""",
                    (str(run["run_id"]),),
                ).fetchall()
                if run else []
            )
            return {"run": run, "reports": reports}

    def get_knowledge_forge_community_report_run(
        self, run_id: str,
    ) -> dict[str, Any] | None:
        """Read current counters for one durable report run."""

        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE run_id=%s""",
                (str(run_id),),
            ).fetchone()

    def list_searchable_knowledge_forge_community_reports(
        self, run_id: str, embedding_model_id: str,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT report.*,vector.embedding
                   FROM knowledge_forge_community_reports report
                   LEFT JOIN vector_embeddings vector
                     ON vector.project_id=report.project_id
                    AND vector.object_type='community_report'
                    AND vector.object_key=report.report_id
                    AND vector.model_name=%s AND vector.is_deleted=FALSE
                   WHERE report.run_id=%s
                     AND report.status IN ('succeeded','reused')
                   ORDER BY report.level,report.community_key""",
                (str(embedding_model_id), str(run_id)),
            ).fetchall()

    def bump_knowledge_forge_community_report_priority(
        self,
        *,
        run_id: str,
        community_key: str = "",
        entity_keys: list[str] | None = None,
    ) -> int:
        if not community_key and not entity_keys:
            return 0
        clauses = []
        params: list[Any] = [str(run_id)]
        if community_key:
            clauses.append("community_key=%s")
            params.append(str(community_key))
        if entity_keys:
            clauses.append("entity_keys && %s::text[]")
            params.append(sorted({str(value) for value in entity_keys if str(value)}))
        with self.connection() as conn:
            result = conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET priority=GREATEST(priority,2000000000),updated_at=NOW()
                   WHERE run_id=%s AND status='pending' AND ("""
                + " OR ".join(clauses) + ")",
                tuple(params),
            )
            return max(0, int(result.rowcount or 0))

    def queue_deferred_knowledge_forge_community_reports(
        self,
        *,
        project_id: int,
        run_id: str,
        job_id: str,
        community_key: str = "",
        entity_keys: list[str] | None = None,
        limit: int = 1,
    ) -> dict[str, Any]:
        """Activate a bounded set of fact-rich deferred reports.

        The report run row serializes this transition with finalization.  If
        the initial batch has already ended, the same immutable run is resumed
        under a fresh durable job; completed reports are never regenerated.
        """

        keys = sorted({str(value) for value in entity_keys or [] if str(value)})
        if not community_key and not keys:
            return {"activated": 0, "prioritized": 0, "dispatch": False}
        clauses: list[str] = []
        selector_params: list[Any] = []
        if community_key:
            clauses.append("community_key=%s")
            selector_params.append(str(community_key))
        if keys:
            clauses.append("entity_keys && %s::text[]")
            selector_params.append(keys)
        selector = " OR ".join(clauses)
        with self.connection() as conn:
            run = conn.execute(
                """SELECT * FROM knowledge_forge_community_report_runs
                   WHERE run_id=%s AND project_id=%s FOR UPDATE""",
                (str(run_id), int(project_id)),
            ).fetchone()
            if not run:
                raise LookupError("社区报告运行不存在")
            if str(run.get("status") or "") in {"cancelled", "superseded"}:
                return {
                    "activated": 0, "prioritized": 0, "dispatch": False,
                    "run": run,
                }
            prioritized = conn.execute(
                """UPDATE knowledge_forge_community_reports
                   SET priority=GREATEST(priority,2000000000),updated_at=NOW()
                   WHERE run_id=%s AND status='pending' AND ("""
                + selector + ")",
                (str(run_id), *selector_params),
            ).rowcount
            deferred = conn.execute(
                """SELECT report_id FROM knowledge_forge_community_reports
                   WHERE run_id=%s AND status='skipped'
                     AND error_category='Deferred' AND ("""
                + selector
                + ") ORDER BY priority DESC,level,community_key "
                  "LIMIT %s FOR UPDATE",
                (
                    str(run_id), *selector_params,
                    max(1, min(20, int(limit))),
                ),
            ).fetchall()
            report_ids = [str(item["report_id"]) for item in deferred]
            activated = len(report_ids)
            if report_ids:
                conn.execute(
                    """UPDATE knowledge_forge_community_reports SET
                           status='pending',priority=2000000000,
                           embedding_status='pending',embedding_error='',
                           error_category='',error_message='',updated_at=NOW(),
                           completed_at=NULL
                       WHERE report_id=ANY(%s)""",
                    (report_ids,),
                )
                run = conn.execute(
                    """UPDATE knowledge_forge_community_report_runs SET
                           eligible_count=eligible_count+%s,
                           skipped_count=GREATEST(0,skipped_count-%s)
                       WHERE run_id=%s RETURNING *""",
                    (activated, activated, str(run_id)),
                ).fetchone()
            dispatch = False
            job = None
            if activated and str(run.get("status") or "") in {
                "completed", "completed_empty", "partial", "failed",
            }:
                job = conn.execute(
                    """INSERT INTO jobs(
                           id,kind,project_id,metadata,strategy_key
                       ) VALUES(%s,'community_report_build',%s,%s::jsonb,%s)
                       RETURNING *""",
                    (
                        str(job_id), int(project_id), self._json({
                            "stage": "queued",
                            "report_run_id": str(run_id),
                            "community_run_id": str(run["community_run_id"]),
                            "graph_generation": int(run["graph_generation"]),
                            "automatic": False,
                            "on_demand": True,
                        }),
                        KNOWLEDGE_FORGE_STRATEGY,
                    ),
                ).fetchone()
                run = conn.execute(
                    """UPDATE knowledge_forge_community_report_runs SET
                           status='queued',job_id=%s,error_message='',
                           completed_at=NULL
                       WHERE run_id=%s RETURNING *""",
                    (str(job_id), str(run_id)),
                ).fetchone()
                dispatch = True
            elif run.get("job_id"):
                job = conn.execute(
                    "SELECT * FROM jobs WHERE id=%s",
                    (str(run["job_id"]),),
                ).fetchone()
            return {
                "activated": activated,
                "prioritized": max(0, int(prioritized or 0)),
                "dispatch": dispatch,
                "run": run,
                "job": job,
            }

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

    def start_community_report_run(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            detection = conn.execute(
                "SELECT * FROM graphrag_community_runs WHERE run_id=%s FOR SHARE",
                (values["detection_run_id"],),
            ).fetchone()
            if not detection or int(detection["project_id"]) != int(values["project_id"]):
                raise LookupError("community detection run 不存在或跨项目")
            if detection["status"] not in {"completed", "completed_empty"}:
                raise RuntimeError("community detection run 尚未完整完成")
            row = conn.execute(
                """INSERT INTO graphrag_community_report_runs(
                   run_id,project_id,graph_generation,detection_run_id,
                   community_input_hash,input_hash,prompt_hash,model_config_hash,model_id,
                   graphrag_version,status,expected,config)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'running',%s,%s::jsonb)
                   ON CONFLICT(run_id) DO NOTHING RETURNING *""",
                (
                    values["run_id"], values["project_id"], values["graph_generation"],
                    values["detection_run_id"], values["community_input_hash"],
                    values["input_hash"], values["prompt_hash"],
                    values["model_config_hash"], values["model_id"],
                    values["graphrag_version"],
                    values["expected"], self._json(values.get("config") or {}),
                ),
            ).fetchone()
            if row:
                return row
            existing = conn.execute(
                "SELECT * FROM graphrag_community_report_runs WHERE run_id=%s",
                (values["run_id"],),
            ).fetchone()
            immutable = (
                "project_id", "graph_generation", "detection_run_id", "input_hash",
                "prompt_hash", "model_config_hash", "model_id", "graphrag_version",
            )
            if not existing or any(str(existing[key]) != str(values[key]) for key in immutable):
                raise RuntimeError("community report run 幂等键与不可变输入不一致")
            return existing

    def list_community_report_items(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM graphrag_community_reports
                   WHERE run_id=%s ORDER BY level,community_id""",
                (run_id,),
            ).fetchall()

    def upsert_community_report_item(
        self, run_id: str, item: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM graphrag_community_report_runs WHERE run_id=%s FOR SHARE",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("community report run 不存在")
            community = conn.execute(
                """SELECT * FROM graphrag_communities
                   WHERE run_id=%s AND project_id=%s AND stable_community_id=%s""",
                (run["detection_run_id"], run["project_id"], item["community_id"]),
            ).fetchone()
            if not community:
                raise ValueError("community report 引用了跨项目或不存在的 community")
            entity_ids = sorted({int(value) for value in item.get("entity_ids") or []})
            if entity_ids:
                matched = conn.execute(
                    """SELECT COUNT(*) AS count FROM graphrag_community_members
                       WHERE run_id=%s AND project_id=%s AND stable_community_id=%s
                         AND canonical_entity_id=ANY(%s)""",
                    (
                        run["detection_run_id"], run["project_id"],
                        item["community_id"], entity_ids,
                    ),
                ).fetchone()
                if int(matched["count"]) != len(entity_ids):
                    raise ValueError("community report 引用了跨社区或不存在的 entity")
            relation_ids = sorted({str(value) for value in item.get("relation_ids") or []})
            if relation_ids:
                matched = conn.execute(
                    """SELECT COUNT(*) AS count FROM graphrag_community_edges edge
                       WHERE edge.run_id=%s AND edge.project_id=%s
                         AND edge.stable_edge_id=ANY(%s)
                         AND EXISTS(
                           SELECT 1 FROM graphrag_community_members source_member
                           WHERE source_member.run_id=edge.run_id
                             AND source_member.stable_community_id=%s
                             AND source_member.canonical_entity_id=edge.source_entity_id)
                         AND EXISTS(
                           SELECT 1 FROM graphrag_community_members target_member
                           WHERE target_member.run_id=edge.run_id
                             AND target_member.stable_community_id=%s
                             AND target_member.canonical_entity_id=edge.target_entity_id)""",
                    (
                        run["detection_run_id"], run["project_id"], relation_ids,
                        item["community_id"], item["community_id"],
                    ),
                ).fetchone()
                if int(matched["count"]) != len(relation_ids):
                    raise ValueError("community report 引用了跨社区或不存在的 relation")
            return conn.execute(
                """INSERT INTO graphrag_community_reports(
                   report_id,run_id,project_id,detection_run_id,community_id,level,status,
                   attempt,title,summary,findings,entity_ids,relation_ids,input_hash,
                   output_hash,usage,latency_ms,error_category,error_message)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,%s,%s,%s)
                   ON CONFLICT(run_id,community_id) DO UPDATE SET
                     status=EXCLUDED.status,attempt=EXCLUDED.attempt,
                     title=EXCLUDED.title,summary=EXCLUDED.summary,findings=EXCLUDED.findings,
                     entity_ids=EXCLUDED.entity_ids,relation_ids=EXCLUDED.relation_ids,
                     input_hash=EXCLUDED.input_hash,output_hash=EXCLUDED.output_hash,
                     usage=EXCLUDED.usage,latency_ms=EXCLUDED.latency_ms,
                     error_category=EXCLUDED.error_category,
                     error_message=EXCLUDED.error_message,
                     updated_at=NOW()
                   WHERE graphrag_community_reports.status<>'succeeded'
                   RETURNING *""",
                (
                    item["report_id"], run_id, run["project_id"], run["detection_run_id"],
                    item["community_id"], item.get("level", 0), item["status"],
                    item.get("attempt", 1), item.get("title", ""), item.get("summary", ""),
                    self._json(item.get("findings") or []), entity_ids,
                    relation_ids, item["input_hash"],
                    item.get("output_hash", ""), self._json(item.get("usage") or {}),
                    item.get("latency_ms", 0), item.get("error_category", ""),
                    item.get("error", ""),
                ),
            ).fetchone() or conn.execute(
                "SELECT * FROM graphrag_community_reports WHERE run_id=%s AND community_id=%s",
                (run_id, item["community_id"]),
            ).fetchone()

    def finalize_community_report_run(
        self, run_id: str, *, status: str, metrics: dict[str, Any], error: str = "",
    ) -> dict[str, Any]:
        if status not in {"succeeded", "retryable_failed", "terminal_failed", "superseded"}:
            raise ValueError("非法 community report run 状态")
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM graphrag_community_report_runs WHERE run_id=%s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("community report run 不存在")
            counts = conn.execute(
                """SELECT COUNT(*) FILTER(WHERE status='succeeded') AS completed,
                   COUNT(*) FILTER(WHERE status<>'succeeded') AS failed
                   FROM graphrag_community_reports WHERE run_id=%s""",
                (run_id,),
            ).fetchone()
            completed = int(counts["completed"])
            failed = int(counts["failed"])
            if status == "succeeded":
                if completed != int(run["expected"]) or failed:
                    raise RuntimeError("community reports 不完整，禁止 active")
                detection = conn.execute(
                    """SELECT detection.*,project.graph_status,
                       project.graph_generation AS current_graph_generation,
                       project.graph_ready_generation,project.graph_policy_version,
                       project.graph_policy_hash AS current_graph_policy_hash,
                       project.active_policy_hash AS current_active_policy_hash,
                       project.active_schema_version,
                       project.entity_extraction_policy_version,
                       project.entity_resolution_policy_version,
                       project.relation_candidate_policy_version,
                       project.relation_classifier_policy_version,
                       project.verification_policy_version,
                       project.schema_policy_version,
                       project.is_deleted AS project_is_deleted
                       FROM graphrag_community_runs detection
                       JOIN projects project ON project.id=detection.project_id
                       WHERE detection.run_id=%s FOR SHARE OF detection,project""",
                    (run["detection_run_id"],),
                ).fetchone()
                project_view = dict(detection or {})
                if detection:
                    project_view["graph_generation"] = detection["current_graph_generation"]
                    project_view["graph_policy_hash"] = detection["current_graph_policy_hash"]
                    project_view["active_policy_hash"] = detection["current_active_policy_hash"]
                detection_current = bool(
                    detection
                    and not detection["project_is_deleted"]
                    and detection["status"] in {"completed", "completed_empty"}
                    and is_current_graph(project_view)
                    and int(detection["graph_generation"])
                    == int(detection["current_graph_generation"])
                    and detection["graph_policy_hash"]
                    == detection["current_graph_policy_hash"]
                    and detection["active_policy_hash"]
                    == detection["current_active_policy_hash"]
                )
                if not detection_current:
                    raise RuntimeError("community detection 已 stale，禁止 active reports")
                conn.execute(
                    """UPDATE graphrag_community_report_runs SET active=FALSE,status='superseded'
                       WHERE project_id=%s AND active=TRUE AND run_id<>%s""",
                    (run["project_id"], run_id),
                )
            row = conn.execute(
                """UPDATE graphrag_community_report_runs SET status=%s,active=%s,
                   completed=%s,failed=%s,metrics=%s::jsonb,error_message=%s,
                   completed_at=NOW() WHERE run_id=%s RETURNING *""",
                (
                    status, status == "succeeded", completed,
                    failed, self._json({**metrics, "completed": completed, "failed": failed}),
                    error, run_id,
                ),
            ).fetchone()
            return row

    def show_community_report_run(self, run_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM graphrag_community_report_runs WHERE run_id=%s",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("community report run 不存在")
            reports = conn.execute(
                """SELECT * FROM graphrag_community_reports
                   WHERE run_id=%s ORDER BY level,community_id""",
                (run_id,),
            ).fetchall()
        detection = self.get_graphrag_community_run(run["detection_run_id"])
        current = bool(
            run["status"] == "succeeded" and run["active"] and detection.get("current")
        )
        return {"run": {**run, "current": current}, "reports": reports, "current": current}

    def get_active_community_report_run(self, project_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT * FROM graphrag_community_report_runs
                   WHERE project_id=%s AND active=TRUE AND status='succeeded'
                   ORDER BY completed_at DESC LIMIT 1""",
                (project_id,),
            ).fetchone()

    def _require_current_community_report_run(
        self, conn, report_run_id: str, project_id: int,
    ) -> dict[str, Any]:
        report = conn.execute(
            "SELECT * FROM graphrag_community_report_runs WHERE run_id=%s FOR SHARE",
            (report_run_id,),
        ).fetchone()
        if not report:
            raise LookupError("community report run 不存在")
        if int(report["project_id"]) != int(project_id):
            raise RuntimeError("community report run 跨项目")
        detection = conn.execute(
            "SELECT * FROM graphrag_community_runs WHERE run_id=%s FOR SHARE",
            (report["detection_run_id"],),
        ).fetchone()
        project = conn.execute(
            "SELECT * FROM projects WHERE id=%s FOR SHARE",
            (project_id,),
        ).fetchone()
        current = bool(
            report["active"]
            and report["status"] == "succeeded"
            and detection
            and detection["status"] in {"completed", "completed_empty"}
            and project
            and not project["is_deleted"]
            and is_current_graph(project)
            and int(report["graph_generation"]) == int(detection["graph_generation"])
            and int(detection["graph_generation"]) == int(project["graph_generation"])
            and report["community_input_hash"] == detection["input_hash"]
            and detection["graph_policy_hash"] == project["graph_policy_hash"]
            and detection["active_policy_hash"] == project["active_policy_hash"]
        )
        if not current:
            raise RuntimeError("community report snapshot stale or incomplete")
        return report

    def start_global_search_run(self, values: dict[str, Any]) -> dict[str, Any]:
        with self.connection() as conn:
            self._require_current_community_report_run(
                conn, values["report_run_id"], int(values["project_id"]),
            )
            inserted = conn.execute(
                """INSERT INTO graphrag_global_search_runs(
                   run_id,project_id,report_run_id,query_hash,report_input_hash,
                   prompt_hash,model_config_hash,model_id,graphrag_version,status,config)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'running',%s::jsonb)
                   ON CONFLICT(run_id) DO NOTHING
                   RETURNING *""",
                (
                    values["run_id"], values["project_id"], values["report_run_id"],
                    values["query_hash"], values["report_input_hash"],
                    values["prompt_hash"], values["model_config_hash"],
                    values["model_id"], values["graphrag_version"],
                    self._json(values.get("config") or {}),
                ),
            ).fetchone()
            if inserted:
                return inserted
            existing = conn.execute(
                "SELECT * FROM graphrag_global_search_runs WHERE run_id=%s",
                (values["run_id"],),
            ).fetchone()
            immutable = (
                "project_id", "report_run_id", "query_hash", "report_input_hash",
                "prompt_hash", "model_config_hash", "model_id", "graphrag_version",
            )
            if not existing or any(
                str(existing[key]) != str(values[key]) for key in immutable
            ):
                raise RuntimeError("global search run 幂等键与不可变输入不一致")
            if existing["status"] in {"succeeded", "terminal_failed"}:
                return existing
            return conn.execute(
                """UPDATE graphrag_global_search_runs
                   SET status='running',error_category='',error_message='',completed_at=NULL
                   WHERE run_id=%s RETURNING *""",
                (values["run_id"],),
            ).fetchone()

    def finalize_global_search_run(
        self, run_id: str, *, status: str, result: dict[str, Any], error: str = "",
        error_category: str = "", latency_ms: int = 0,
    ) -> dict[str, Any]:
        if status not in {"succeeded", "retryable_failed", "terminal_failed", "superseded"}:
            raise ValueError("非法 global search 状态")
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM graphrag_global_search_runs WHERE run_id=%s FOR UPDATE",
                (run_id,),
            ).fetchone()
            if not run:
                raise LookupError("global search run 不存在")
            if status == "succeeded":
                self._require_current_community_report_run(
                    conn, run["report_run_id"], int(run["project_id"]),
                )
                report_ids = sorted({
                    str(value) for value in result.get("report_ids") or []
                })
                if not report_ids:
                    raise RuntimeError("global search 缺少 report 引用")
                matched = conn.execute(
                    """SELECT COUNT(*) AS count FROM graphrag_community_reports
                       WHERE run_id=%s AND status='succeeded' AND report_id=ANY(%s)""",
                    (run["report_run_id"], report_ids),
                ).fetchone()
                if int(matched["count"]) != len(report_ids):
                    raise RuntimeError("global search 引用了跨快照或不存在的 report")
            return conn.execute(
                """UPDATE graphrag_global_search_runs SET status=%s,result=%s::jsonb,
                   selected_report_ids=%s,usage=%s::jsonb,latency_ms=%s,
                   error_category=%s,error_message=%s,completed_at=NOW()
                   WHERE run_id=%s RETURNING *""",
                (
                    status, self._json(result), result.get("report_ids") or [],
                    self._json(result.get("usage") or {}), latency_ms,
                    error_category, error, run_id,
                ),
            ).fetchone()

    def show_global_search_run(self, run_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM graphrag_global_search_runs WHERE run_id=%s",
                (run_id,),
            ).fetchone()
            if not row:
                raise LookupError("global search run 不存在")
            return row

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

    # ------------------------------------------------------------------
    # Knowledge-agent v1 control plane.  These methods intentionally live in
    # the PostgreSQL boundary: the agent consumes current facts but never
    # mutates a graph, schema, document revision or embedding.

    def agent_project_source_snapshot(self, project_id: int) -> dict[str, Any]:
        """Return a deterministic digest of current active document revisions."""

        with self.connection() as conn:
            project = conn.execute(
                "SELECT * FROM projects WHERE id=%s AND is_deleted=FALSE",
                (int(project_id),),
            ).fetchone()
            if not project:
                raise LookupError("知识库不存在")
            documents = conn.execute(
                """SELECT document.id,document.active_revision_id,
                          revision.source_hash,revision.manifest_hash,
                          document.original_filename
                   FROM documents document
                   LEFT JOIN document_content_revisions revision
                     ON revision.revision_id=document.active_revision_id
                   WHERE document.project_id=%s AND document.is_deleted=FALSE
                   ORDER BY document.id""",
                (int(project_id),),
            ).fetchall()
            counts = conn.execute(
                """SELECT
                     (SELECT COUNT(*) FROM chunks chunk
                       JOIN documents document ON document.id=chunk.document_id
                        AND document.active_revision_id=chunk.revision_id
                      WHERE chunk.project_id=%s AND chunk.is_deleted=FALSE
                        AND document.is_deleted=FALSE) AS chunks,
                     (SELECT COUNT(*) FROM vector_embeddings vector
                       JOIN documents document ON document.id=vector.document_id
                        AND document.active_revision_id=vector.revision_id
                      WHERE vector.project_id=%s AND vector.object_type='chunk'
                        AND vector.model_name=%s AND vector.is_deleted=FALSE
                        AND document.is_deleted=FALSE) AS embeddings""",
                (
                    int(project_id), int(project_id),
                    str(project.get("embedding_model") or self.config.default_embedding_model),
                ),
            ).fetchone()
        material = [{
            "document_id": int(row["id"]),
            "revision_id": str(row.get("active_revision_id") or ""),
            "source_hash": str(row.get("source_hash") or ""),
            "manifest_hash": str(row.get("manifest_hash") or ""),
        } for row in documents]
        digest = hashlib.sha256(
            self._canonical_json(material).encode("utf-8")
        ).hexdigest()
        return {
            "project_id": int(project_id),
            "source_hash": digest,
            "documents": material,
            "document_count": len(material),
            "chunk_count": int((counts or {}).get("chunks") or 0),
            "embedding_count": int((counts or {}).get("embeddings") or 0),
            "embedding_model": str(
                project.get("embedding_model") or self.config.default_embedding_model
            ),
        }

    def agent_entity_sources(
        self, namespace_id: str, canonical_keys: list[str], *, limit_per_entity: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Resolve grounded canonical entities back to exact current mentions."""

        keys = list(dict.fromkeys(str(value) for value in canonical_keys if str(value)))[:100]
        if not keys:
            return {}
        with self.connection() as conn:
            rows = conn.execute(
                """WITH ranked AS (
                     SELECT assignment.canonical_key,mention.evidence_text,
                            mention.evidence_start,mention.evidence_end,
                            mention.revision_id,mention.document_id,mention.chunk_id,
                            chunk.content,chunk.metadata,section.title AS section_title,
                            document.original_filename,document.storage_path,
                            document.mime_type,
                            ROW_NUMBER() OVER (
                              PARTITION BY assignment.canonical_key
                              ORDER BY mention.confidence DESC,mention.document_id,
                                       mention.chunk_id,mention.evidence_start
                            ) AS source_rank
                       FROM entity_shadow_assignments assignment
                       JOIN entity_shadow_mentions mention
                         ON mention.namespace_id=assignment.namespace_id
                        AND mention.mention_key=assignment.mention_key
                       JOIN chunks chunk ON chunk.id=mention.chunk_id
                        AND chunk.revision_id=mention.revision_id
                       LEFT JOIN sections section ON section.id=chunk.section_id
                       JOIN documents document ON document.id=mention.document_id
                        AND document.active_revision_id=mention.revision_id
                      WHERE assignment.namespace_id=%s
                        AND assignment.canonical_key=ANY(%s::text[])
                        AND document.is_deleted=FALSE AND chunk.is_deleted=FALSE
                   ) SELECT * FROM ranked WHERE source_rank<=%s
                     ORDER BY canonical_key,source_rank""",
                (str(namespace_id), keys, max(1, min(10, int(limit_per_entity)))),
            ).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
        for row in rows:
            grouped.setdefault(str(row["canonical_key"]), []).append(row)
        return grouped

    def agent_page_media_assets(
        self, project_id: int, document_ids: list[int],
    ) -> list[dict[str, Any]]:
        """Load current media-bearing chunks for exact-page source enrichment.

        This is a presentation/provenance lookup only.  It never changes
        retrieval scores or treats a nearby image as an additional graph fact.
        The agent adapter is responsible for attaching an asset only when an
        evidence item explicitly refers to a visual on the same source page.
        """

        ids = sorted({int(value) for value in document_ids if int(value) > 0})[:200]
        if not ids:
            return []
        with self.connection() as conn:
            return conn.execute(
                """SELECT chunk.document_id,chunk.revision_id,chunk.chunk_index,
                          chunk.metadata
                     FROM chunks chunk
                     JOIN documents document ON document.id=chunk.document_id
                      AND document.active_revision_id=chunk.revision_id
                    WHERE chunk.project_id=%s AND chunk.document_id=ANY(%s)
                      AND chunk.is_deleted=FALSE AND document.is_deleted=FALSE
                      AND jsonb_typeof(chunk.metadata->'media_assets')='array'
                      AND jsonb_array_length(chunk.metadata->'media_assets')>0
                    ORDER BY chunk.document_id,chunk.chunk_index,chunk.id""",
                (int(project_id), ids),
            ).fetchall()

    def agent_visual_reference_chunks(
        self,
        project_id: int,
        patterns: list[str],
        *,
        query: str = "",
        limit: int = 48,
    ) -> list[dict[str, Any]]:
        """Load current chunks that may contain a numbered visual reference.

        ``patterns`` are deliberately coarse recall surfaces produced by the
        agent adapter.  The adapter performs the exact figure/table identifier
        comparison before admitting a row as evidence.
        """

        surfaces = list(dict.fromkeys(
            str(value)[:120] for value in patterns if str(value).strip()
        ))[:24]
        if not surfaces:
            return []
        with self.connection() as conn:
            return conn.execute(
                """SELECT chunk.id,chunk.document_id,chunk.revision_id,
                          chunk.content,chunk.metadata,
                          section.title AS section_title,
                          document.original_filename,document.storage_path,
                          document.mime_type
                     FROM chunks chunk
                     JOIN documents document ON document.id=chunk.document_id
                      AND document.active_revision_id=chunk.revision_id
                     LEFT JOIN sections section ON section.id=chunk.section_id
                    WHERE chunk.project_id=%s AND chunk.is_deleted=FALSE
                      AND document.is_deleted=FALSE
                      AND (
                        chunk.content ILIKE ANY(%s::text[])
                        OR chunk.metadata::text ILIKE ANY(%s::text[])
                      )
                    ORDER BY
                      CASE WHEN strpos(
                        lower(%s), lower(document.original_filename)
                      )>0 THEN 0 ELSE 1 END,
                      document.id,chunk.chunk_index,chunk.id
                    LIMIT %s""",
                (
                    int(project_id), surfaces, surfaces, str(query or ""),
                    max(1, min(64, int(limit))),
                ),
            ).fetchall()

    def agent_conversation_context(
        self, run_id: str, *, limit: int = 24,
    ) -> dict[str, Any]:
        """Read prior completed messages for one project-isolated agent turn.

        The current user message is already persisted when a run is queued, so
        it is excluded by run ID.  Failed or cancelled turns are excluded as
        well: their user text remains auditable, but it must not silently enter
        the semantic conversation context without a completed assistant turn.
        """

        # The caller performs token-aware admission.  This is only a bounded
        # database bootstrap ceiling so a newly restored long thread can feed
        # several incremental compaction batches without loading unbounded
        # rows into memory.
        bounded_limit = max(2, min(500, int(limit)))
        with self.connection() as conn:
            run = conn.execute(
                "SELECT id,thread_id,project_id FROM agent_runs WHERE id=%s",
                (str(run_id),),
            ).fetchone()
            if not run:
                raise LookupError("问答任务不存在")
            total = conn.execute(
                """SELECT COUNT(*)::integer AS count
                     FROM agent_messages message
                     JOIN agent_runs prior ON prior.id=message.run_id
                    WHERE message.thread_id=%s AND message.project_id=%s
                      AND message.run_id<>%s AND message.status='completed'
                      AND prior.status='completed'""",
                (
                    str(run["thread_id"]), int(run["project_id"]),
                    str(run_id),
                ),
            ).fetchone()
            rows = conn.execute(
                """SELECT message.id,message.run_id,message.role,
                          message.content,message.created_at
                     FROM agent_messages message
                     JOIN agent_runs prior ON prior.id=message.run_id
                    WHERE message.thread_id=%s AND message.project_id=%s
                      AND message.run_id<>%s AND message.status='completed'
                      AND prior.status='completed'
                    ORDER BY message.created_at DESC,message.id DESC
                    LIMIT %s""",
                (
                    str(run["thread_id"]), int(run["project_id"]),
                    str(run_id), bounded_limit,
                ),
            ).fetchall()
            documents = conn.execute(
                """SELECT id,original_filename
                     FROM documents
                    WHERE project_id=%s AND is_deleted=FALSE
                    ORDER BY id LIMIT 50""",
                (int(run["project_id"]),),
            ).fetchall()
        rows.reverse()
        count = int((total or {}).get("count") or 0)
        return {
            "thread_id": str(run["thread_id"]),
            "project_id": int(run["project_id"]),
            "items": rows,
            "total": count,
            "truncated": count > len(rows),
            "knowledge_scope": {
                "document_count": len(documents),
                "documents": [
                    {
                        "document_id": int(item["id"]),
                        "filename": str(item.get("original_filename") or ""),
                    }
                    for item in documents
                ],
                "document_list_truncated": len(documents) >= 50,
            },
        }

    def create_agent_thread(
        self, *, thread_id: str, project_id: int, strategy_key: str,
        session_key_hash: str, title: str = "", metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            project = conn.execute(
                "SELECT 1 FROM projects WHERE id=%s AND is_deleted=FALSE",
                (int(project_id),),
            ).fetchone()
            if not project:
                raise LookupError("知识库不存在")
            return conn.execute(
                """INSERT INTO agent_threads(
                       id,project_id,strategy_key,session_key_hash,title,metadata)
                   VALUES(%s,%s,%s,%s,%s,%s::jsonb) RETURNING *""",
                (
                    str(thread_id), int(project_id), str(strategy_key),
                    str(session_key_hash), str(title or "")[:200],
                    self._json(metadata),
                ),
            ).fetchone()

    def list_agent_threads(
        self, *, project_id: int, session_key_hash: str, limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List one browser capability's conversations without loading transcripts."""

        bounded_limit = max(1, min(100, int(limit)))
        with self.connection() as conn:
            return conn.execute(
                """SELECT thread.id,thread.project_id,thread.strategy_key,
                          thread.title,thread.status,thread.metadata,
                          thread.created_at,thread.updated_at,
                          (SELECT COUNT(*)::integer
                             FROM agent_messages message
                            WHERE message.thread_id=thread.id) AS message_count,
                          COALESCE((
                            SELECT LEFT(message.content,240)
                              FROM agent_messages message
                             WHERE message.thread_id=thread.id
                             ORDER BY message.created_at DESC,message.id DESC
                             LIMIT 1
                          ),'') AS last_message_preview,
                          COALESCE((
                            SELECT run.id
                              FROM agent_runs run
                             WHERE run.thread_id=thread.id
                               AND run.status IN ('queued','running')
                             ORDER BY run.created_at DESC,run.id DESC
                             LIMIT 1
                          ),'') AS active_run_id
                     FROM agent_threads thread
                    WHERE thread.project_id=%s
                      AND thread.session_key_hash=%s
                      AND thread.status='active'
                    ORDER BY thread.updated_at DESC,thread.id DESC
                    LIMIT %s""",
                (int(project_id), str(session_key_hash), bounded_limit),
            ).fetchall()

    def get_agent_thread(
        self, thread_id: str, *, session_key_hash: str | None = None,
    ) -> dict[str, Any] | None:
        params: list[Any] = [str(thread_id)]
        clause = ""
        if session_key_hash is not None:
            clause = " AND session_key_hash=%s"
            params.append(str(session_key_hash))
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM agent_threads WHERE id=%s" + clause,
                tuple(params),
            ).fetchone()

    def get_agent_thread_payload(
        self, thread_id: str, *, session_key_hash: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            thread = conn.execute(
                """SELECT * FROM agent_threads
                   WHERE id=%s AND session_key_hash=%s AND status='active'""",
                (str(thread_id), str(session_key_hash)),
            ).fetchone()
            if not thread:
                return None
            messages = conn.execute(
                """SELECT * FROM agent_messages WHERE thread_id=%s
                   ORDER BY created_at,id""",
                (str(thread_id),),
            ).fetchall()
            runs = conn.execute(
                """SELECT * FROM agent_runs WHERE thread_id=%s
                   ORDER BY created_at,id""",
                (str(thread_id),),
            ).fetchall()
            return {**dict(thread), "messages": messages, "runs": runs}

    def archive_agent_thread(
        self, thread_id: str, *, session_key_hash: str,
    ) -> dict[str, Any]:
        """Soft-delete an owned conversation while preserving its audit ledger."""

        with self.connection() as conn:
            thread = conn.execute(
                """SELECT * FROM agent_threads
                   WHERE id=%s AND session_key_hash=%s AND status='active'
                   FOR UPDATE""",
                (str(thread_id), str(session_key_hash)),
            ).fetchone()
            if not thread:
                raise PermissionError("问答会话不存在或无权访问")
            active = conn.execute(
                """SELECT 1 FROM agent_runs
                   WHERE thread_id=%s AND status IN ('queued','running')
                   LIMIT 1""",
                (str(thread_id),),
            ).fetchone()
            if active:
                raise RuntimeError("该会话正在生成回答，请先停止回答再删除")
            return conn.execute(
                """UPDATE agent_threads SET status='archived',updated_at=NOW()
                   WHERE id=%s RETURNING *""",
                (str(thread_id),),
            ).fetchone()

    def create_agent_run(
        self, *, run_id: str, message_id: str, thread_id: str,
        session_key_hash: str, query: str, requested_mode: str,
        budget: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            thread = conn.execute(
                """SELECT * FROM agent_threads
                   WHERE id=%s AND session_key_hash=%s AND status='active'
                   FOR UPDATE""",
                (str(thread_id), str(session_key_hash)),
            ).fetchone()
            if not thread:
                raise PermissionError("问答会话不存在或无权访问")
            active = conn.execute(
                """SELECT id FROM agent_runs WHERE thread_id=%s
                   AND status IN ('queued','running') FOR UPDATE""",
                (str(thread_id),),
            ).fetchone()
            if active:
                raise RuntimeError("当前会话已有问题正在处理")
            run = conn.execute(
                """INSERT INTO agent_runs(
                       id,thread_id,project_id,strategy_key,requested_mode,query,budget)
                   VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb) RETURNING *""",
                (
                    str(run_id), str(thread_id), int(thread["project_id"]),
                    str(thread["strategy_key"]), str(requested_mode), str(query),
                    self._json(budget),
                ),
            ).fetchone()
            conn.execute(
                """INSERT INTO agent_messages(
                       id,thread_id,run_id,project_id,role,status,content)
                   VALUES(%s,%s,%s,%s,'user','completed',%s)""",
                (
                    str(message_id), str(thread_id), str(run_id),
                    int(thread["project_id"]), str(query),
                ),
            )
            conn.execute(
                """INSERT INTO agent_run_events(run_id,project_id,event_type,payload)
                   VALUES(%s,%s,'queued',%s::jsonb)""",
                (str(run_id), int(thread["project_id"]), self._json({"status": "queued"})),
            )
            conn.execute(
                """UPDATE agent_threads
                      SET title=CASE WHEN btrim(title)='' THEN %s ELSE title END,
                          updated_at=NOW()
                    WHERE id=%s""",
                (str(query).strip()[:80], str(thread_id)),
            )
            return run

    def get_agent_run(
        self, run_id: str, *, session_key_hash: str | None = None,
    ) -> dict[str, Any] | None:
        params: list[Any] = [str(run_id)]
        clause = ""
        if session_key_hash is not None:
            clause = " AND thread.session_key_hash=%s AND thread.status='active'"
            params.append(str(session_key_hash))
        with self.connection() as conn:
            return conn.execute(
                """SELECT run.* FROM agent_runs run
                   JOIN agent_threads thread ON thread.id=run.thread_id
                   WHERE run.id=%s""" + clause,
                tuple(params),
            ).fetchone()

    def get_agent_run_payload(
        self, run_id: str, *, session_key_hash: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            run = conn.execute(
                """SELECT run.* FROM agent_runs run
                   JOIN agent_threads thread ON thread.id=run.thread_id
                   WHERE run.id=%s AND thread.session_key_hash=%s
                     AND thread.status='active'""",
                (str(run_id), str(session_key_hash)),
            ).fetchone()
            if not run:
                return None
            steps = conn.execute(
                "SELECT * FROM agent_run_steps WHERE run_id=%s ORDER BY sequence",
                (str(run_id),),
            ).fetchall()
            tools = conn.execute(
                "SELECT * FROM agent_tool_calls WHERE run_id=%s ORDER BY started_at,id",
                (str(run_id),),
            ).fetchall()
            evidence = conn.execute(
                """SELECT evidence_id,project_id,strategy_key,snapshot_ref,
                          evidence_type,title,content,source_refs,
                          fused_score,rerank_score,provenance_valid,rank,selected
                   FROM agent_evidence WHERE run_id=%s
                   ORDER BY selected DESC,rank,evidence_id""",
                (str(run_id),),
            ).fetchall()
            citations = conn.execute(
                """SELECT * FROM agent_claim_citations WHERE run_id=%s
                   ORDER BY claim_index,citation_index,evidence_id""",
                (str(run_id),),
            ).fetchall()
            messages = conn.execute(
                """SELECT * FROM agent_messages WHERE run_id=%s
                   ORDER BY created_at,id""",
                (str(run_id),),
            ).fetchall()
            return {
                **dict(run), "steps": steps, "tool_calls": tools,
                "evidence": evidence, "citations": citations,
                "messages": messages,
            }

    def get_agent_run_trace_payload(
        self, run_id: str, *, session_key_hash: str,
    ) -> dict[str, Any] | None:
        """Read only the lightweight execution ledger for one owned run.

        The browser polls this endpoint while an answer is running.  It reads
        only bounded summaries for selected evidence; full evidence bodies,
        source payloads and message transcripts remain behind their dedicated
        authorized endpoints.
        """

        with self.connection() as conn:
            run = conn.execute(
                """SELECT run.* FROM agent_runs run
                   JOIN agent_threads thread ON thread.id=run.thread_id
                   WHERE run.id=%s AND thread.session_key_hash=%s
                     AND thread.status='active'""",
                (str(run_id), str(session_key_hash)),
            ).fetchone()
            if not run:
                return None
            steps = conn.execute(
                "SELECT * FROM agent_run_steps WHERE run_id=%s ORDER BY sequence",
                (str(run_id),),
            ).fetchall()
            tools = conn.execute(
                "SELECT * FROM agent_tool_calls WHERE run_id=%s ORDER BY started_at,id",
                (str(run_id),),
            ).fetchall()
            evidence_summaries = conn.execute(
                """SELECT evidence.evidence_id,evidence.evidence_type,
                          evidence.title,LEFT(evidence.content,500) AS excerpt,
                          evidence.rank,evidence.rerank_score,
                          COALESCE(evidence.source_refs->0->>'filename','') AS filename,
                          COALESCE(evidence.source_refs->0->>'page_label','') AS page_label,
                          COALESCE(evidence.source_refs->0->>'section_title','') AS section_title,
                          (SELECT MIN(citation.citation_index)
                             FROM agent_claim_citations citation
                            WHERE citation.run_id=evidence.run_id
                              AND citation.evidence_id=evidence.evidence_id)
                            AS citation_index
                     FROM agent_evidence evidence
                    WHERE evidence.run_id=%s AND evidence.selected IS TRUE
                      AND evidence.provenance_valid IS TRUE
                    ORDER BY evidence.rank,evidence.evidence_id
                    LIMIT 24""",
                (str(run_id),),
            ).fetchall()
            return {
                **dict(run), "steps": steps, "tool_calls": tools,
                "evidence_summaries": evidence_summaries,
            }

    def start_agent_run(self, run_id: str) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE agent_runs SET status='running',stage='authorizing',
                          started_at=NOW(),updated_at=NOW()
                   WHERE id=%s AND status='queued' AND cancellation_requested=FALSE
                   RETURNING *""",
                (str(run_id),),
            ).fetchone()
            if row:
                conn.execute(
                    """INSERT INTO agent_run_events(run_id,project_id,event_type,payload)
                       VALUES(%s,%s,'status',%s::jsonb)""",
                    (str(run_id), int(row["project_id"]), self._json({"status": "running", "stage": "authorizing"})),
                )
                return row
            current = conn.execute(
                "SELECT * FROM agent_runs WHERE id=%s", (str(run_id),),
            ).fetchone()
            if not current:
                raise LookupError("问答任务不存在")
            return current

    def update_agent_run(
        self, run_id: str, *, stage: str | None = None,
        snapshot_ref: dict[str, Any] | None = None,
        retrieval_summary: dict[str, Any] | None = None,
        sufficiency: dict[str, Any] | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assignments = ["updated_at=NOW()"]
        params: list[Any] = []
        for key, value in (
            ("stage", stage), ("model_name", model_name),
        ):
            if value is not None:
                assignments.append(f"{key}=%s")
                params.append(str(value))
        for key, value in (
            ("snapshot_ref", snapshot_ref),
            ("retrieval_summary", retrieval_summary),
            ("sufficiency", sufficiency), ("metadata", metadata),
        ):
            if value is not None:
                assignments.append(f"{key}=%s::jsonb")
                params.append(self._json(value))
        params.append(str(run_id))
        with self.connection() as conn:
            row = conn.execute(
                f"UPDATE agent_runs SET {','.join(assignments)} WHERE id=%s RETURNING *",
                tuple(params),
            ).fetchone()
            if not row:
                raise LookupError("问答任务不存在")
            return row

    def record_agent_step(
        self, *, run_id: str, project_id: int, sequence: int,
        step_type: str, status: str, input_summary: dict[str, Any] | None = None,
        output_summary: dict[str, Any] | None = None, error_message: str = "",
    ) -> dict[str, Any]:
        completed = status in {"completed", "failed", "cancelled", "degraded"}
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO agent_run_steps(
                       run_id,project_id,sequence,step_type,status,input_summary,
                       output_summary,error_message,completed_at)
                   VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,
                          CASE WHEN %s THEN NOW() ELSE NULL END)
                   ON CONFLICT(run_id,sequence) DO UPDATE SET
                     step_type=EXCLUDED.step_type,status=EXCLUDED.status,
                     input_summary=EXCLUDED.input_summary,
                     output_summary=EXCLUDED.output_summary,
                     error_message=EXCLUDED.error_message,
                     completed_at=EXCLUDED.completed_at
                   RETURNING *""",
                (
                    str(run_id), int(project_id), int(sequence), str(step_type),
                    str(status), self._json(input_summary), self._json(output_summary),
                    str(error_message)[:2000], completed,
                ),
            ).fetchone()

    def start_agent_tool_call(
        self, *, call_id: str, run_id: str, project_id: int,
        tool_name: str, request_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            return conn.execute(
                """INSERT INTO agent_tool_calls(
                       id,run_id,project_id,tool_name,status,request_summary)
                   VALUES(%s,%s,%s,%s,'running',%s::jsonb) RETURNING *""",
                (
                    str(call_id), str(run_id), int(project_id), str(tool_name),
                    self._json(request_summary),
                ),
            ).fetchone()

    def finish_agent_tool_call(
        self, call_id: str, *, status: str,
        response_summary: dict[str, Any] | None = None, error_message: str = "",
    ) -> dict[str, Any]:
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE agent_tool_calls SET status=%s,response_summary=%s::jsonb,
                          error_message=%s,completed_at=NOW()
                   WHERE id=%s RETURNING *""",
                (
                    str(status), self._json(response_summary),
                    str(error_message)[:2000], str(call_id),
                ),
            ).fetchone()
            if not row:
                raise LookupError("问答工具调用不存在")
            return row

    def persist_agent_evidence(self, run_id: str, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        with self.connection() as conn:
            run = conn.execute(
                "SELECT project_id,status FROM agent_runs WHERE id=%s FOR UPDATE",
                (str(run_id),),
            ).fetchone()
            if not run or run["status"] != "running":
                raise RuntimeError("问答任务不接受证据写入")
            count = 0
            for row in rows:
                if int(row.get("project_id") or 0) != int(run["project_id"]):
                    raise ValueError("问答证据跨知识库")
                conn.execute(
                    """INSERT INTO agent_evidence(
                         evidence_id,run_id,project_id,evidence_type,strategy_key,
                         snapshot_ref,title,content,entity_keys,relation_keys,
                         source_refs,retrieval_channels,raw_scores,fused_score,
                         rerank_score,provenance_score,provenance_valid,rank,
                         selected,metadata)
                       VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s::jsonb,
                              %s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                    (
                        str(row["evidence_id"]), str(run_id), int(row["project_id"]),
                        str(row["evidence_type"]), str(row["strategy_key"]),
                        self._json(row["snapshot_ref"]), str(row.get("title") or ""),
                        str(row.get("content") or ""), list(row.get("entity_keys") or []),
                        list(row.get("relation_keys") or []), self._json(row.get("source_refs") or []),
                        list(row.get("retrieval_channels") or []), self._json(row.get("raw_scores") or {}),
                        float(row.get("fused_score") or 0), float(row.get("rerank_score") or 0),
                        float(row.get("provenance_score") or 0), bool(row.get("provenance_valid")),
                        int(row.get("rank") or 0), bool(row.get("selected")),
                        self._json(row.get("metadata") or {}),
                    ),
                )
                count += 1
            return count

    def append_agent_event(
        self, run_id: str, event_type: str, payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                "SELECT project_id FROM agent_runs WHERE id=%s", (str(run_id),),
            ).fetchone()
            if not run:
                raise LookupError("问答任务不存在")
            return conn.execute(
                """INSERT INTO agent_run_events(run_id,project_id,event_type,payload)
                   VALUES(%s,%s,%s,%s::jsonb) RETURNING *""",
                (
                    str(run_id), int(run["project_id"]), str(event_type),
                    self._json(payload),
                ),
            ).fetchone()

    def list_agent_events(
        self, run_id: str, *, session_key_hash: str, after_id: int = 0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            owned = conn.execute(
                """SELECT 1 FROM agent_runs run JOIN agent_threads thread
                     ON thread.id=run.thread_id
                   WHERE run.id=%s AND thread.session_key_hash=%s
                     AND thread.status='active'""",
                (str(run_id), str(session_key_hash)),
            ).fetchone()
            if not owned:
                raise PermissionError("问答任务不存在或无权访问")
            return conn.execute(
                """SELECT * FROM agent_run_events
                   WHERE run_id=%s AND id>%s ORDER BY id LIMIT %s""",
                (str(run_id), max(0, int(after_id)), max(1, min(500, int(limit)))),
            ).fetchall()

    def agent_run_cancel_requested(self, run_id: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT cancellation_requested,status FROM agent_runs WHERE id=%s",
                (str(run_id),),
            ).fetchone()
            return bool(
                not row or row.get("cancellation_requested")
                or row.get("status") in {"cancelled", "failed"}
            )

    def request_agent_run_cancellation(
        self, run_id: str, *, session_key_hash: str,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            owned = conn.execute(
                """SELECT run.* FROM agent_runs run
                   JOIN agent_threads thread ON thread.id=run.thread_id
                   WHERE run.id=%s AND thread.session_key_hash=%s
                     AND thread.status='active'
                   FOR UPDATE OF run""",
                (str(run_id), str(session_key_hash)),
            ).fetchone()
            if not owned:
                raise PermissionError("问答任务不存在或无权访问")
            if owned["status"] in {"completed", "failed", "cancelled"}:
                return owned
            row = conn.execute(
                """UPDATE agent_runs run SET cancellation_requested=TRUE,
                          status=CASE WHEN run.status='queued' THEN 'cancelled' ELSE run.status END,
                          stage=CASE WHEN run.status='queued' THEN 'cancelled' ELSE run.stage END,
                          completed_at=CASE WHEN run.status='queued' THEN NOW() ELSE run.completed_at END,
                          updated_at=NOW()
                   WHERE run.id=%s AND run.status IN ('queued','running')
                   RETURNING run.*""",
                (str(run_id),),
            ).fetchone()
            if not row:
                return owned
            conn.execute(
                """INSERT INTO agent_run_events(run_id,project_id,event_type,payload)
                   VALUES(%s,%s,%s,%s::jsonb)""",
                (
                    str(run_id), int(row["project_id"]),
                    "cancelled" if row["status"] == "cancelled" else "cancellation_requested",
                    self._json({"status": str(row["status"])}),
                ),
            )
            return row

    def complete_agent_run(
        self, *, run_id: str, message_id: str, answer_text: str,
        answer_json: dict[str, Any], citations: list[dict[str, Any]],
        effective_mode: str, stream_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                "SELECT * FROM agent_runs WHERE id=%s FOR UPDATE", (str(run_id),),
            ).fetchone()
            if not run:
                raise LookupError("问答任务不存在")
            if run.get("cancellation_requested"):
                raise JobCancellationRequested("问答任务已请求终止")
            if run["status"] != "running":
                raise RuntimeError("问答任务不在运行状态")
            conn.execute(
                """INSERT INTO agent_messages(
                       id,thread_id,run_id,project_id,role,status,content,metadata)
                   VALUES(%s,%s,%s,%s,'assistant','completed',%s,%s::jsonb)""",
                (
                    str(message_id), str(run["thread_id"]), str(run_id),
                    int(run["project_id"]), str(answer_text),
                    self._json({"answer": answer_json}),
                ),
            )
            for citation in citations:
                conn.execute(
                    """INSERT INTO agent_claim_citations(
                         run_id,project_id,message_id,claim_index,claim_text,
                         evidence_id,citation_index,status)
                       VALUES(%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        str(run_id), int(run["project_id"]), str(message_id),
                        int(citation["claim_index"]), str(citation["claim_text"]),
                        str(citation["evidence_id"]), int(citation["citation_index"]),
                        str(citation["status"]),
                    ),
                )
            completed = conn.execute(
                """UPDATE agent_runs SET status='completed',stage='completed',
                          effective_mode=%s,answer_json=%s::jsonb,answer_text=%s,
                          completed_at=NOW(),updated_at=NOW()
                   WHERE id=%s RETURNING *""",
                (
                    str(effective_mode), self._json(answer_json),
                    str(answer_text), str(run_id),
                ),
            ).fetchone()
            for event in stream_events or []:
                conn.execute(
                    """INSERT INTO agent_run_events(
                         run_id,project_id,event_type,payload)
                       VALUES(%s,%s,%s,%s::jsonb)""",
                    (
                        str(run_id), int(run["project_id"]),
                        str(event.get("event_type") or "answer_delta"),
                        self._json(event.get("payload") or {}),
                    ),
                )
            conn.execute(
                """INSERT INTO agent_run_events(run_id,project_id,event_type,payload)
                   VALUES(%s,%s,'completed',%s::jsonb)""",
                (
                    str(run_id), int(run["project_id"]),
                    self._json({"status": "completed", "message_id": str(message_id)}),
                ),
            )
            conn.execute(
                "UPDATE agent_threads SET updated_at=NOW() WHERE id=%s",
                (str(run["thread_id"]),),
            )
            return completed

    def terminate_agent_run(
        self, run_id: str, *, status: str, error_message: str = "",
    ) -> dict[str, Any]:
        if status not in {"failed", "cancelled"}:
            raise ValueError("非法问答终止状态")
        with self.connection() as conn:
            row = conn.execute(
                """UPDATE agent_runs SET status=%s,stage=%s,error_message=%s,
                          completed_at=NOW(),updated_at=NOW()
                   WHERE id=%s AND status IN ('queued','running') RETURNING *""",
                (str(status), str(status), str(error_message)[:4000], str(run_id)),
            ).fetchone()
            changed = bool(row)
            if not row:
                row = conn.execute(
                    "SELECT * FROM agent_runs WHERE id=%s", (str(run_id),),
                ).fetchone()
            if not row:
                raise LookupError("问答任务不存在")
            if changed and row["status"] == status:
                conn.execute(
                    """INSERT INTO agent_run_events(run_id,project_id,event_type,payload)
                       VALUES(%s,%s,%s,%s::jsonb)""",
                    (
                        str(run_id), int(row["project_id"]), str(status),
                        self._json({"status": status, "error": str(error_message)[:1000]}),
                    ),
                )
            return row

    def get_agent_evidence(
        self, evidence_id: str, *, session_key_hash: str,
    ) -> dict[str, Any] | None:
        with self.connection() as conn:
            return conn.execute(
                """SELECT evidence.* FROM agent_evidence evidence
                   JOIN agent_runs run ON run.id=evidence.run_id
                   JOIN agent_threads thread ON thread.id=run.thread_id
                   WHERE evidence.evidence_id=%s
                     AND thread.session_key_hash=%s
                     AND thread.status='active'""",
                (str(evidence_id), str(session_key_hash)),
            ).fetchone()

    def record_agent_feedback(
        self, *, feedback_id: str, run_id: str, session_key_hash: str,
        rating: int, comment: str, action: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.connection() as conn:
            run = conn.execute(
                """SELECT run.* FROM agent_runs run JOIN agent_threads thread
                     ON thread.id=run.thread_id
                   WHERE run.id=%s AND thread.session_key_hash=%s
                     AND thread.status='active'""",
                (str(run_id), str(session_key_hash)),
            ).fetchone()
            if not run:
                raise PermissionError("问答任务不存在或无权访问")
            return conn.execute(
                """INSERT INTO agent_feedback(
                       id,run_id,project_id,session_key_hash,rating,comment,action,metadata)
                   VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                   ON CONFLICT(run_id,session_key_hash,action) DO UPDATE SET
                     rating=EXCLUDED.rating,comment=EXCLUDED.comment,
                     metadata=EXCLUDED.metadata,updated_at=NOW()
                   RETURNING *""",
                (
                    str(feedback_id), str(run_id), int(run["project_id"]),
                    str(session_key_hash), int(rating), str(comment)[:2000],
                    str(action)[:100], self._json(metadata),
                ),
            ).fetchone()

    def recover_interrupted_agent_runs(self) -> int:
        """Fail closed after process loss; a queued/running answer is never resumed blindly."""

        with self.connection() as conn:
            rows = conn.execute(
                """UPDATE agent_runs SET status='failed',stage='failed',
                          error_message='服务重启中断了本次问答，请重新检索',
                          completed_at=NOW(),updated_at=NOW()
                   WHERE status IN ('queued','running') RETURNING id,project_id"""
            ).fetchall()
            for row in rows:
                conn.execute(
                    """INSERT INTO agent_run_events(run_id,project_id,event_type,payload)
                       VALUES(%s,%s,'failed',%s::jsonb)""",
                    (
                        str(row["id"]), int(row["project_id"]),
                        self._json({
                            "status": "failed",
                            "error": "服务重启中断了本次问答，请重新检索",
                        }),
                    ),
                )
            return len(rows)
