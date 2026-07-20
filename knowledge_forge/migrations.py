"""Versioned PostgreSQL migrations with checksum and advisory-lock protection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Iterable

import psycopg


MIGRATION_LOCK_ID = 4_919_726_401


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


REVISION_SCHEMA_SQL = r"""
ALTER TABLE documents ADD CONSTRAINT uq_documents_id_project UNIQUE (id,project_id);

CREATE TABLE document_content_revisions (
    revision_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL,
    document_id BIGINT NOT NULL,
    ingestion_run_id TEXT NOT NULL,
    content_version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    manifest JSONB NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('building','active','retired','failed','legacy_orphan')
    ),
    lease_token TEXT,
    lease_expires_at TIMESTAMPTZ,
    provenance_ambiguous BOOLEAN NOT NULL DEFAULT FALSE,
    failure TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    retired_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    CONSTRAINT fk_content_revision_document
        FOREIGN KEY(document_id,project_id) REFERENCES documents(id,project_id),
    UNIQUE(document_id,content_version),
    UNIQUE(document_id,ingestion_run_id),
    UNIQUE(revision_id,document_id),
    UNIQUE(revision_id,project_id,document_id)
);
CREATE UNIQUE INDEX uq_document_active_revision
    ON document_content_revisions(document_id) WHERE status='active';
CREATE INDEX idx_content_revisions_document
    ON document_content_revisions(document_id,status,content_version DESC);

ALTER TABLE documents ADD COLUMN active_revision_id TEXT;

INSERT INTO document_content_revisions(
    revision_id,project_id,document_id,ingestion_run_id,content_version,
    content_hash,source_hash,config_hash,policy_hash,manifest_hash,manifest,
    status,provenance_ambiguous,activated_at,retired_at
)
SELECT 'legacy:'||d.id,d.project_id,d.id,'legacy:'||d.id,
       GREATEST(d.content_version,1),d.doc_hash,d.doc_hash,'legacy','legacy',
       md5('legacy:'||d.id),
       jsonb_build_object('legacy',TRUE,'document_id',d.id),
       CASE WHEN d.is_deleted THEN 'retired' ELSE 'active' END,
       TRUE,
       CASE WHEN d.is_deleted THEN NULL ELSE COALESCE(d.indexed_at,d.updated_at) END,
       CASE WHEN d.is_deleted THEN COALESCE(d.deleted_at,d.updated_at) ELSE NULL END
FROM documents d
ON CONFLICT(document_id,ingestion_run_id) DO NOTHING;

UPDATE documents d SET active_revision_id=r.revision_id,
    content_version=GREATEST(d.content_version,r.content_version)
FROM document_content_revisions r
WHERE r.document_id=d.id AND r.status='active' AND d.active_revision_id IS NULL;

ALTER TABLE documents ADD CONSTRAINT fk_documents_active_revision
    FOREIGN KEY(active_revision_id) REFERENCES document_content_revisions(revision_id)
    DEFERRABLE INITIALLY DEFERRED;

ALTER TABLE sections ADD COLUMN revision_id TEXT;
ALTER TABLE chunks ADD COLUMN revision_id TEXT;
ALTER TABLE vector_embeddings ADD COLUMN revision_id TEXT;
ALTER TABLE indexing_failures ADD COLUMN revision_id TEXT;
ALTER TABLE entity_mentions ADD COLUMN revision_id TEXT;
ALTER TABLE canonical_entity_documents ADD COLUMN revision_id TEXT;
ALTER TABLE entity_resolution_decisions ADD COLUMN revision_id TEXT;
ALTER TABLE entity_type_assertions ADD COLUMN revision_id TEXT;
ALTER TABLE entity_assignments ADD COLUMN revision_id TEXT;
ALTER TABLE relation_assertions ADD COLUMN revision_id TEXT;
ALTER TABLE canonical_relation_projections ADD COLUMN revision_id TEXT;
ALTER TABLE relation_candidates ADD COLUMN revision_id TEXT;
ALTER TABLE mention_relation_assertions ADD COLUMN revision_id TEXT;
ALTER TABLE relation_verification_batches ADD COLUMN revision_id TEXT;
ALTER TABLE relation_verification_events ADD COLUMN revision_id TEXT;
ALTER TABLE relation_shadow_runs ADD COLUMN revision_id TEXT;
ALTER TABLE relation_shadow_decisions ADD COLUMN revision_id TEXT;
ALTER TABLE entity_shadow_assignments ADD COLUMN revision_id TEXT;
ALTER TABLE policy_snapshot_documents ADD COLUMN revision_id TEXT;

UPDATE sections s SET revision_id=d.active_revision_id
FROM documents d WHERE d.id=s.document_id AND s.revision_id IS NULL;
UPDATE chunks c SET revision_id=d.active_revision_id
FROM documents d WHERE d.id=c.document_id AND c.revision_id IS NULL;

INSERT INTO document_content_revisions(
    revision_id,project_id,document_id,ingestion_run_id,content_version,
    content_hash,source_hash,config_hash,policy_hash,manifest_hash,manifest,
    status,provenance_ambiguous,failure
)
SELECT 'legacy-orphan:'||d.id,d.project_id,d.id,'legacy-orphan:'||d.id,
       COALESCE((SELECT MAX(r.content_version)+1 FROM document_content_revisions r
                 WHERE r.document_id=d.id),2),
       d.doc_hash,d.doc_hash,'legacy','legacy',
       md5('legacy-orphan:'||d.id),
       jsonb_build_object('legacy_orphan',TRUE,'document_id',d.id),
       'legacy_orphan',TRUE,'历史事实引用的 chunk 已失联'
FROM documents d
WHERE EXISTS (
    SELECT 1 FROM relation_candidates c
    WHERE c.document_id=d.id AND NOT EXISTS(SELECT 1 FROM chunks k WHERE k.id=c.chunk_id)
) OR EXISTS (
    SELECT 1 FROM relation_verification_events e
    WHERE e.document_id=d.id AND NOT EXISTS(SELECT 1 FROM chunks k WHERE k.id=e.chunk_id)
)
ON CONFLICT(document_id,ingestion_run_id) DO NOTHING;

UPDATE vector_embeddings x SET revision_id=COALESCE(c.revision_id,d.active_revision_id)
FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
WHERE x.document_id=d.id AND (x.chunk_id IS NULL OR c.id=x.chunk_id) AND x.revision_id IS NULL;
UPDATE indexing_failures x SET revision_id=COALESCE(c.revision_id,d.active_revision_id)
FROM documents d LEFT JOIN chunks c ON c.document_id=d.id
WHERE x.document_id=d.id AND (x.chunk_id IS NULL OR c.id=x.chunk_id) AND x.revision_id IS NULL;
UPDATE entity_mentions x SET revision_id=c.revision_id FROM chunks c
WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE canonical_entity_documents x SET revision_id=d.active_revision_id FROM documents d
WHERE d.id=x.document_id AND x.revision_id IS NULL;
UPDATE entity_resolution_decisions x SET revision_id=c.revision_id FROM chunks c
WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE entity_type_assertions x SET revision_id=c.revision_id FROM chunks c
WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE entity_assignments x SET revision_id=m.revision_id FROM entity_mentions m
WHERE m.id=x.mention_id AND x.revision_id IS NULL;
UPDATE relation_assertions x SET revision_id=c.revision_id FROM chunks c
WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE canonical_relation_projections x SET revision_id=r.revision_id FROM relation_assertions r
WHERE r.id=x.assertion_id AND x.revision_id IS NULL;
UPDATE relation_candidates x SET revision_id=COALESCE(
    c.revision_id,
    (SELECT r.revision_id FROM document_content_revisions r
     WHERE r.document_id=x.document_id AND r.status='legacy_orphan' LIMIT 1)
) FROM chunks c WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE relation_candidates x SET revision_id=r.revision_id
FROM document_content_revisions r
WHERE r.document_id=x.document_id AND r.status='legacy_orphan' AND x.revision_id IS NULL;
UPDATE mention_relation_assertions x SET revision_id=c.revision_id FROM chunks c
WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE relation_verification_batches x SET revision_id=COALESCE(
    c.revision_id,
    (SELECT r.revision_id FROM document_content_revisions r
     WHERE r.document_id=x.document_id AND r.status='legacy_orphan' LIMIT 1)
) FROM chunks c WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE relation_verification_events x SET revision_id=COALESCE(
    c.revision_id,
    (SELECT r.revision_id FROM document_content_revisions r
     WHERE r.document_id=x.document_id AND r.status='legacy_orphan' LIMIT 1)
) FROM chunks c WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE relation_shadow_runs x SET revision_id=c.revision_id FROM chunks c
WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE relation_shadow_decisions x SET revision_id=r.revision_id FROM relation_shadow_runs r
WHERE r.id=x.shadow_run_id AND x.revision_id IS NULL;
UPDATE entity_shadow_assignments x SET revision_id=c.revision_id FROM chunks c
WHERE c.id=x.chunk_id AND x.revision_id IS NULL;
UPDATE policy_snapshot_documents x SET revision_id=d.active_revision_id FROM documents d
WHERE d.id=x.document_id AND x.revision_id IS NULL;

DO $$
DECLARE item RECORD;
BEGIN
    FOR item IN
        SELECT conrelid::regclass AS table_name,conname
        FROM pg_constraint
        WHERE contype='u' AND (
            (conrelid='sections'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (document_id, section_key)%') OR
            (conrelid='chunks'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (document_id, chunk_key)%') OR
            (conrelid='canonical_entity_documents'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (canonical_entity_id, document_id)%') OR
            (conrelid='entity_resolution_decisions'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (document_id, chunk_id, mention_text, entity_type)%') OR
            (conrelid='relation_assertions'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (project_id, relation_key)%') OR
            (conrelid='vector_embeddings'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (project_id, object_type, object_key, model_name)%') OR
            (conrelid='indexing_failures'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (project_id, object_type, object_key, model_name)%')
            OR (conrelid='entity_mentions'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (document_id, chunk_id, canonical_entity_id, mention_text)%')
            OR (conrelid='entity_type_assertions'::regclass AND pg_get_constraintdef(oid) LIKE 'UNIQUE (project_id, mention_key, entity_type, policy_version, source)%')
        )
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I',item.table_name,item.conname);
    END LOOP;
END;
$$;
DROP INDEX uq_entity_mentions_stable_key;

CREATE UNIQUE INDEX uq_sections_revision_key ON sections(revision_id,section_key);
CREATE UNIQUE INDEX uq_chunks_revision_key ON chunks(revision_id,chunk_key);
CREATE UNIQUE INDEX uq_canonical_entity_documents_revision
    ON canonical_entity_documents(revision_id,canonical_entity_id);
CREATE UNIQUE INDEX uq_resolution_revision_key
    ON entity_resolution_decisions(revision_id,chunk_id,mention_text,entity_type);
CREATE UNIQUE INDEX uq_relation_assertion_revision_key
    ON relation_assertions(revision_id,relation_key);
CREATE UNIQUE INDEX uq_entity_mentions_revision_key
    ON entity_mentions(revision_id,mention_key)
    WHERE mention_key<>'' AND is_deleted=FALSE;
CREATE UNIQUE INDEX uq_entity_mentions_revision_payload
    ON entity_mentions(revision_id,chunk_id,canonical_entity_id,mention_text);
CREATE UNIQUE INDEX uq_entity_type_assertions_revision
    ON entity_type_assertions(revision_id,mention_key,entity_type,policy_version,source);
CREATE UNIQUE INDEX uq_vectors_revision_object
    ON vector_embeddings(revision_id,object_type,object_key,model_name)
    WHERE revision_id IS NOT NULL;
CREATE UNIQUE INDEX uq_vectors_project_object
    ON vector_embeddings(project_id,object_type,object_key,model_name)
    WHERE revision_id IS NULL;
CREATE UNIQUE INDEX uq_indexing_failures_revision_object
    ON indexing_failures(revision_id,object_type,object_key,model_name)
    WHERE revision_id IS NOT NULL;
CREATE UNIQUE INDEX uq_indexing_failures_project_object
    ON indexing_failures(project_id,object_type,object_key,model_name)
    WHERE revision_id IS NULL;

ALTER TABLE chunks ADD CONSTRAINT uq_chunks_id_revision UNIQUE(id,revision_id);
ALTER TABLE entity_mentions ADD CONSTRAINT uq_mentions_id_revision UNIQUE(id,revision_id);
ALTER TABLE relation_assertions ADD CONSTRAINT uq_assertions_id_revision UNIQUE(id,revision_id);
ALTER TABLE relation_candidates ADD CONSTRAINT uq_candidates_id_revision UNIQUE(candidate_id,revision_id);
ALTER TABLE relation_shadow_runs ADD CONSTRAINT uq_shadow_runs_id_revision UNIQUE(id,revision_id);

ALTER TABLE sections ADD CONSTRAINT fk_sections_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE chunks ADD CONSTRAINT fk_chunks_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);

ALTER TABLE entity_mentions ADD CONSTRAINT fk_mentions_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE entity_mentions ADD CONSTRAINT fk_mentions_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id);
ALTER TABLE entity_resolution_decisions ADD CONSTRAINT fk_resolution_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE entity_resolution_decisions ADD CONSTRAINT fk_resolution_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id);
ALTER TABLE entity_type_assertions ADD CONSTRAINT fk_type_assertion_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE entity_type_assertions ADD CONSTRAINT fk_type_assertion_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id);
ALTER TABLE relation_assertions ADD CONSTRAINT fk_relation_assertion_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE relation_assertions ADD CONSTRAINT fk_relation_assertion_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id);
ALTER TABLE relation_candidates ADD CONSTRAINT fk_relation_candidate_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE relation_candidates ADD CONSTRAINT fk_relation_candidate_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id) NOT VALID;
ALTER TABLE mention_relation_assertions ADD CONSTRAINT fk_mention_relation_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE mention_relation_assertions ADD CONSTRAINT fk_mention_relation_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id);
ALTER TABLE relation_verification_batches ADD CONSTRAINT fk_verification_batch_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE relation_verification_batches ADD CONSTRAINT fk_verification_batch_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id) NOT VALID;
ALTER TABLE relation_verification_events ADD CONSTRAINT fk_verification_event_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE relation_verification_events ADD CONSTRAINT fk_verification_event_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id) NOT VALID;
ALTER TABLE relation_shadow_runs ADD CONSTRAINT fk_relation_shadow_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE relation_shadow_runs ADD CONSTRAINT fk_relation_shadow_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id);

ALTER TABLE vector_embeddings ADD CONSTRAINT fk_vector_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE vector_embeddings ADD CONSTRAINT ck_vector_revision_scope
    CHECK ((document_id IS NULL AND revision_id IS NULL) OR
           (document_id IS NOT NULL AND revision_id IS NOT NULL));
ALTER TABLE indexing_failures ADD CONSTRAINT fk_indexing_failure_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE indexing_failures ADD CONSTRAINT ck_indexing_failure_revision_scope
    CHECK ((document_id IS NULL AND revision_id IS NULL) OR
           (document_id IS NOT NULL AND revision_id IS NOT NULL));
ALTER TABLE canonical_entity_documents ADD CONSTRAINT fk_entity_document_revision
    FOREIGN KEY(revision_id,project_id,document_id)
    REFERENCES document_content_revisions(revision_id,project_id,document_id);
ALTER TABLE entity_assignments ADD CONSTRAINT fk_assignment_mention_revision
    FOREIGN KEY(mention_id,revision_id) REFERENCES entity_mentions(id,revision_id);
ALTER TABLE canonical_relation_projections ADD CONSTRAINT fk_projection_assertion_revision
    FOREIGN KEY(assertion_id,revision_id) REFERENCES relation_assertions(id,revision_id);
ALTER TABLE relation_shadow_decisions ADD CONSTRAINT fk_shadow_decision_run_revision
    FOREIGN KEY(shadow_run_id,revision_id) REFERENCES relation_shadow_runs(id,revision_id);
ALTER TABLE entity_shadow_assignments ADD CONSTRAINT fk_entity_shadow_chunk_revision
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id);
ALTER TABLE policy_snapshot_documents ADD CONSTRAINT fk_snapshot_document_revision
    FOREIGN KEY(revision_id,document_id)
    REFERENCES document_content_revisions(revision_id,document_id);

CREATE OR REPLACE FUNCTION kf_guard_revision_write() RETURNS trigger AS $$
DECLARE
    ctx_revision TEXT := NULLIF(current_setting('knowledge_forge.revision_id',TRUE),'');
    ctx_run TEXT := NULLIF(current_setting('knowledge_forge.ingestion_run_id',TRUE),'');
    ctx_lease TEXT := NULLIF(current_setting('knowledge_forge.revision_lease_token',TRUE),'');
    revision_row document_content_revisions%ROWTYPE;
BEGIN
    IF NEW.revision_id IS NULL THEN
        NEW.revision_id := ctx_revision;
    END IF;
    IF NEW.revision_id IS NULL AND TG_ARGV[0]='chunk' THEN
        EXECUTE 'SELECT revision_id FROM chunks WHERE id=$1'
            INTO NEW.revision_id USING NEW.chunk_id;
    ELSIF NEW.revision_id IS NULL AND TG_ARGV[0]='mention' THEN
        EXECUTE 'SELECT revision_id FROM entity_mentions WHERE id=$1'
            INTO NEW.revision_id USING NEW.mention_id;
    ELSIF NEW.revision_id IS NULL AND TG_ARGV[0]='assertion' THEN
        EXECUTE 'SELECT revision_id FROM relation_assertions WHERE id=$1'
            INTO NEW.revision_id USING NEW.assertion_id;
    ELSIF NEW.revision_id IS NULL AND TG_ARGV[0]='candidate' THEN
        EXECUTE 'SELECT revision_id FROM relation_candidates WHERE candidate_id=$1'
            INTO NEW.revision_id USING NEW.candidate_id;
    ELSIF NEW.revision_id IS NULL AND TG_ARGV[0]='shadow_run' THEN
        EXECUTE 'SELECT revision_id FROM relation_shadow_runs WHERE id=$1'
            INTO NEW.revision_id USING NEW.shadow_run_id;
    ELSIF NEW.revision_id IS NULL AND TG_ARGV[0]='document' THEN
        EXECUTE 'SELECT active_revision_id FROM documents WHERE id=$1'
            INTO NEW.revision_id USING NEW.document_id;
    END IF;
    IF NEW.revision_id IS NULL THEN
        RAISE EXCEPTION 'document-scoped write requires content revision'
            USING ERRCODE='55000';
    END IF;
    SELECT * INTO revision_row FROM document_content_revisions
    WHERE revision_id=NEW.revision_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown content revision %',NEW.revision_id USING ERRCODE='23503';
    END IF;
    IF ctx_revision IS NOT NULL THEN
        IF ctx_revision<>NEW.revision_id OR revision_row.status<>'building'
           OR revision_row.ingestion_run_id IS DISTINCT FROM ctx_run
           OR revision_row.lease_token IS DISTINCT FROM ctx_lease
           OR revision_row.lease_expires_at IS NULL OR revision_row.lease_expires_at<=NOW() THEN
            RAISE EXCEPTION 'stale content revision or lease lost: %',NEW.revision_id
                USING ERRCODE='55000';
        END IF;
    ELSIF revision_row.status NOT IN ('active','legacy_orphan') THEN
        RAISE EXCEPTION 'content revision is not writable: % (%)',
            NEW.revision_id,revision_row.status USING ERRCODE='55000';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sections_revision BEFORE INSERT OR UPDATE ON sections
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('document');
CREATE TRIGGER trg_chunks_revision BEFORE INSERT OR UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('document');
CREATE TRIGGER trg_vectors_revision BEFORE INSERT OR UPDATE ON vector_embeddings
    FOR EACH ROW WHEN (NEW.document_id IS NOT NULL)
    EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_indexing_failures_revision BEFORE INSERT OR UPDATE ON indexing_failures
    FOR EACH ROW WHEN (NEW.document_id IS NOT NULL)
    EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_mentions_revision BEFORE INSERT OR UPDATE ON entity_mentions
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_entity_documents_revision BEFORE INSERT OR UPDATE ON canonical_entity_documents
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('document');
CREATE TRIGGER trg_resolution_revision BEFORE INSERT OR UPDATE ON entity_resolution_decisions
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_type_assertion_revision BEFORE INSERT OR UPDATE ON entity_type_assertions
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_assignment_revision BEFORE INSERT OR UPDATE ON entity_assignments
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('mention');
CREATE TRIGGER trg_relation_assertion_revision BEFORE INSERT OR UPDATE ON relation_assertions
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_relation_projection_revision BEFORE INSERT OR UPDATE ON canonical_relation_projections
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('assertion');
CREATE TRIGGER trg_relation_candidate_revision BEFORE INSERT OR UPDATE ON relation_candidates
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_mention_relation_revision BEFORE INSERT OR UPDATE ON mention_relation_assertions
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_verification_batch_revision BEFORE INSERT OR UPDATE ON relation_verification_batches
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_verification_event_revision BEFORE INSERT OR UPDATE ON relation_verification_events
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_shadow_run_revision BEFORE INSERT OR UPDATE ON relation_shadow_runs
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
CREATE TRIGGER trg_shadow_decision_revision BEFORE INSERT OR UPDATE ON relation_shadow_decisions
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('shadow_run');
CREATE TRIGGER trg_entity_shadow_assignment_revision BEFORE INSERT OR UPDATE ON entity_shadow_assignments
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
"""


RELATION_SHADOW_QUEUE_SQL = r"""
ALTER TABLE relation_shadow_runs DROP CONSTRAINT IF EXISTS relation_shadow_runs_status_check;
ALTER TABLE relation_shadow_runs ADD CONSTRAINT relation_shadow_runs_status_check
    CHECK(status IN ('running','completed','failed','deferred','terminal_failed','quarantined'));
ALTER TABLE relation_shadow_runs ADD COLUMN input_fingerprint TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_shadow_runs ADD COLUMN artifact_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_shadow_runs ADD COLUMN expected_work_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE relation_shadow_runs ADD COLUMN completed_work_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE relation_shadow_runs ADD COLUMN failed_work_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE relation_shadow_runs ADD COLUMN snapshot_complete BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE relation_shadow_decisions ADD COLUMN chunk_id BIGINT;
ALTER TABLE relation_shadow_decisions ADD COLUMN evidence_start INTEGER;
ALTER TABLE relation_shadow_decisions ADD COLUMN evidence_end INTEGER;
ALTER TABLE relation_shadow_decisions ADD COLUMN evidence_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_shadow_decisions ADD COLUMN source_mention_start INTEGER;
ALTER TABLE relation_shadow_decisions ADD COLUMN source_mention_end INTEGER;
ALTER TABLE relation_shadow_decisions ADD COLUMN target_mention_start INTEGER;
ALTER TABLE relation_shadow_decisions ADD COLUMN target_mention_end INTEGER;
ALTER TABLE relation_shadow_decisions ADD COLUMN policy_version TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_shadow_decisions ADD COLUMN model_name TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_shadow_decisions ADD COLUMN config_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE relation_shadow_decisions ADD COLUMN accepted_attempt_id TEXT;
ALTER TABLE relation_shadow_decisions ADD COLUMN output_hash TEXT NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS relation_shadow_work_items (
    id TEXT PRIMARY KEY,
    shadow_run_id TEXT NOT NULL,
    revision_id TEXT NOT NULL,
    project_id BIGINT NOT NULL,
    document_id BIGINT NOT NULL,
    chunk_id BIGINT NOT NULL,
    batch_id TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    candidate_ids JSONB NOT NULL,
    request_json JSONB NOT NULL,
    estimated_input_tokens INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'pending','leased','completed','retryable_failed','terminal_failed','quarantined'
    )),
    lease_owner TEXT,
    lease_token TEXT,
    lease_expires_at TIMESTAMPTZ,
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(max_attempts BETWEEN 1 AND 20),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accepted_attempt_id TEXT,
    accepted_output_hash TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(shadow_run_id,batch_id),
    FOREIGN KEY(shadow_run_id,revision_id) REFERENCES relation_shadow_runs(id,revision_id),
    FOREIGN KEY(revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id),
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
);
CREATE INDEX IF NOT EXISTS idx_relation_shadow_work_lease
    ON relation_shadow_work_items(status,next_attempt_at,lease_expires_at,created_at);
CREATE INDEX IF NOT EXISTS idx_relation_shadow_work_run
    ON relation_shadow_work_items(shadow_run_id,status,batch_id);

CREATE TABLE IF NOT EXISTS relation_shadow_attempts (
    id TEXT PRIMARY KEY,
    work_item_id TEXT NOT NULL REFERENCES relation_shadow_work_items(id),
    attempt_no INTEGER NOT NULL,
    lease_token TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'leased','accepted','superseded','retryable_failed','terminal_failed'
    )),
    output_json JSONB,
    output_hash TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(work_item_id,attempt_no)
);
CREATE INDEX IF NOT EXISTS idx_relation_shadow_attempt_work
    ON relation_shadow_attempts(work_item_id,attempt_no);

DROP TRIGGER IF EXISTS trg_shadow_work_revision ON relation_shadow_work_items;
CREATE TRIGGER trg_shadow_work_revision BEFORE INSERT OR UPDATE ON relation_shadow_work_items
    FOR EACH ROW EXECUTE FUNCTION kf_guard_revision_write('chunk');
"""


ENTITY_SHADOW_PIPELINE_SQL = r"""
ALTER TABLE entity_shadow_namespaces ADD COLUMN input_fingerprint TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_shadow_namespaces ADD COLUMN artifact_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_shadow_namespaces ADD COLUMN expected_work_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE entity_shadow_namespaces ADD COLUMN completed_work_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE entity_shadow_namespaces ADD COLUMN failed_work_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE entity_shadow_namespaces ADD COLUMN snapshot_complete BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE entity_shadow_namespaces ADD COLUMN model_name TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_shadow_namespaces ADD COLUMN embedding_model TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_shadow_namespaces ADD COLUMN config_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_shadow_namespaces ADD COLUMN code_version TEXT NOT NULL DEFAULT '';

ALTER TABLE entity_shadow_assignments ADD COLUMN decision TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_shadow_assignments ADD COLUMN reason TEXT NOT NULL DEFAULT '';
ALTER TABLE entity_shadow_assignments ADD COLUMN scores JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE entity_shadow_canonicals ADD COLUMN source_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE entity_shadow_canonicals ADD COLUMN generation BIGINT NOT NULL DEFAULT 1;

CREATE TABLE IF NOT EXISTS entity_shadow_work_items (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    revision_id TEXT NOT NULL,
    project_id BIGINT NOT NULL,
    document_id BIGINT NOT NULL,
    chunk_id BIGINT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'pending','leased','completed','retryable_failed','terminal_failed'
    )),
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(max_attempts BETWEEN 1 AND 20),
    lease_owner TEXT,
    lease_token TEXT,
    lease_expires_at TIMESTAMPTZ,
    output_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_hash TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(namespace_id,chunk_id),
    FOREIGN KEY(revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id),
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_work_claim
    ON entity_shadow_work_items(namespace_id,status,updated_at,id);

CREATE TABLE IF NOT EXISTS entity_shadow_mentions (
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    mention_key TEXT NOT NULL,
    project_id BIGINT NOT NULL,
    document_id BIGINT NOT NULL,
    revision_id TEXT NOT NULL,
    chunk_id BIGINT NOT NULL,
    surface TEXT NOT NULL,
    normalized_surface TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    evidence_start INTEGER NOT NULL,
    evidence_end INTEGER NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    extraction_policy_version TEXT NOT NULL,
    filter_decision TEXT NOT NULL DEFAULT 'accepted',
    filter_reason TEXT NOT NULL DEFAULT '',
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,mention_key),
    CHECK(evidence_start>=0 AND evidence_end>evidence_start),
    FOREIGN KEY(revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id),
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_mentions_chunk
    ON entity_shadow_mentions(namespace_id,chunk_id,evidence_start,mention_key);

CREATE TABLE IF NOT EXISTS entity_shadow_filter_decisions (
    id BIGSERIAL PRIMARY KEY,
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    revision_id TEXT NOT NULL,
    document_id BIGINT NOT NULL,
    chunk_id BIGINT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    raw_entity JSONB NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(namespace_id,chunk_id,reason,raw_entity),
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
);

CREATE TABLE IF NOT EXISTS entity_shadow_type_assertions (
    namespace_id TEXT NOT NULL,
    mention_key TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    evidence_text TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    source TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,mention_key,entity_type,policy_version,source),
    FOREIGN KEY(namespace_id,mention_key)
        REFERENCES entity_shadow_mentions(namespace_id,mention_key)
);

CREATE TABLE IF NOT EXISTS entity_shadow_aliases (
    namespace_id TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    mention_key TEXT NOT NULL,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_kind TEXT NOT NULL,
    evidence_start INTEGER,
    evidence_end INTEGER,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,canonical_key,mention_key,normalized_alias,alias_kind),
    FOREIGN KEY(namespace_id,canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key),
    FOREIGN KEY(namespace_id,mention_key)
        REFERENCES entity_shadow_mentions(namespace_id,mention_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_alias_exact
    ON entity_shadow_aliases(namespace_id,normalized_alias);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_alias_trgm ON entity_shadow_aliases
    USING gin(normalized_alias gin_trgm_ops);

CREATE TABLE IF NOT EXISTS entity_shadow_candidates (
    namespace_id TEXT NOT NULL,
    mention_key TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    stage TEXT NOT NULL,
    rank INTEGER NOT NULL,
    cap INTEGER NOT NULL,
    lexical_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    vector_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    reranker_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    combined_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    filter_reason TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,mention_key,canonical_key,stage),
    FOREIGN KEY(namespace_id,mention_key)
        REFERENCES entity_shadow_mentions(namespace_id,mention_key),
    FOREIGN KEY(namespace_id,canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_candidates_rank
    ON entity_shadow_candidates(namespace_id,mention_key,stage,rank);

CREATE TABLE IF NOT EXISTS entity_shadow_embeddings (
    namespace_id TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    model_name TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    embedding_vector vector NOT NULL,
    dimension INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,canonical_key,model_name),
    FOREIGN KEY(namespace_id,canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_embedding_scope
    ON entity_shadow_embeddings(namespace_id,model_name,dimension);

CREATE TABLE IF NOT EXISTS entity_shadow_cannot_links (
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    left_key TEXT NOT NULL,
    right_key TEXT NOT NULL,
    reason TEXT NOT NULL,
    hard BOOLEAN NOT NULL DEFAULT TRUE,
    policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,left_key,right_key),
    CHECK(left_key<right_key)
);

CREATE TABLE IF NOT EXISTS entity_shadow_must_links (
    namespace_id TEXT NOT NULL,
    mention_key TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    reason TEXT NOT NULL,
    hard BOOLEAN NOT NULL DEFAULT TRUE,
    policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,mention_key,canonical_key),
    FOREIGN KEY(namespace_id,mention_key)
        REFERENCES entity_shadow_mentions(namespace_id,mention_key),
    FOREIGN KEY(namespace_id,canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_must_link_mention
    ON entity_shadow_must_links(namespace_id,mention_key) WHERE hard=TRUE;

CREATE TABLE IF NOT EXISTS entity_shadow_events (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    event_type TEXT NOT NULL CHECK(event_type IN (
        'assign','new','merge','split','undo','cannot_link','uncertain'
    )),
    mention_key TEXT,
    source_canonical_key TEXT,
    target_canonical_key TEXT,
    before_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    after_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    reason TEXT NOT NULL,
    scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    reversible BOOLEAN NOT NULL DEFAULT TRUE,
    reversed_by_event_id TEXT,
    model_name TEXT NOT NULL DEFAULT '',
    config_hash TEXT NOT NULL DEFAULT '',
    policy_version TEXT NOT NULL,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_events_namespace
    ON entity_shadow_events(namespace_id,created_at,id);

CREATE TABLE IF NOT EXISTS entity_shadow_relation_endpoints (
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    relation_key TEXT NOT NULL,
    document_id BIGINT NOT NULL,
    revision_id TEXT NOT NULL,
    chunk_id BIGINT NOT NULL,
    source_mention_key TEXT NOT NULL,
    target_mention_key TEXT NOT NULL,
    source_canonical_key TEXT NOT NULL,
    target_canonical_key TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    evidence_text TEXT NOT NULL DEFAULT '',
    evidence_start INTEGER,
    evidence_end INTEGER,
    policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,relation_key),
    FOREIGN KEY(namespace_id,source_mention_key)
        REFERENCES entity_shadow_mentions(namespace_id,mention_key),
    FOREIGN KEY(namespace_id,target_mention_key)
        REFERENCES entity_shadow_mentions(namespace_id,mention_key),
    FOREIGN KEY(namespace_id,source_canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key),
    FOREIGN KEY(namespace_id,target_canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key),
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_relation_chunk
    ON entity_shadow_relation_endpoints(namespace_id,chunk_id,relation_key);
"""


POLICY_SNAPSHOT_ORCHESTRATOR_SQL = r"""
ALTER TABLE policy_snapshots ADD COLUMN mode TEXT NOT NULL DEFAULT 'relation_only'
    CHECK(mode IN ('relation_only','end_to_end'));
ALTER TABLE policy_snapshots ADD COLUMN manifest_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE policy_snapshots ADD COLUMN artifact_root_hash TEXT NOT NULL DEFAULT '';
ALTER TABLE policy_snapshots ADD COLUMN expected_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE policy_snapshots ADD COLUMN completed_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE policy_snapshots ADD COLUMN failed_items INTEGER NOT NULL DEFAULT 0;
ALTER TABLE policy_snapshots ADD COLUMN cancel_requested BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE policy_snapshots ADD COLUMN pilot BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE policy_snapshots ADD COLUMN created_by TEXT NOT NULL DEFAULT 'system';

DO $$
DECLARE constraint_name TEXT;
BEGIN
    SELECT conname INTO constraint_name
    FROM pg_constraint
    WHERE conrelid='policy_snapshots'::regclass
      AND contype='c'
      AND pg_get_constraintdef(oid) LIKE '%status%building%ready%';
    IF constraint_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE policy_snapshots DROP CONSTRAINT %I',constraint_name);
    END IF;
END $$;
ALTER TABLE policy_snapshots ADD CONSTRAINT ck_policy_snapshot_status
    CHECK(status IN ('building','ready','active','failed','rolled_back','cancelled'));
CREATE UNIQUE INDEX uq_policy_snapshot_manifest
    ON policy_snapshots(project_id,manifest_hash) WHERE manifest_hash<>'';

CREATE TABLE policy_snapshot_work_items (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES policy_snapshots(id),
    phase TEXT NOT NULL CHECK(phase IN ('control','entity','relation')),
    phase_order INTEGER NOT NULL,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT,
    revision_id TEXT,
    chunk_id BIGINT,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(
        status IN ('pending','leased','retryable_failed','completed','terminal_failed','cancelled')
    ),
    attempt INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3 CHECK(max_attempts BETWEEN 1 AND 20),
    lease_owner TEXT,
    lease_token TEXT,
    lease_expires_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accepted_artifact_hash TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(snapshot_id,phase,document_id,revision_id,chunk_id),
    FOREIGN KEY(revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id),
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
);
CREATE INDEX idx_policy_snapshot_work_claim
    ON policy_snapshot_work_items(snapshot_id,phase_order,status,next_attempt_at,lease_expires_at,id);

CREATE TABLE policy_snapshot_artifacts (
    snapshot_id TEXT NOT NULL REFERENCES policy_snapshots(id),
    artifact_key TEXT NOT NULL,
    work_id TEXT NOT NULL UNIQUE REFERENCES policy_snapshot_work_items(id),
    kind TEXT NOT NULL,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT,
    revision_id TEXT,
    chunk_id BIGINT,
    content_hash TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(snapshot_id,artifact_key),
    FOREIGN KEY(revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id),
    FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
);
CREATE INDEX idx_policy_snapshot_artifact_scope
    ON policy_snapshot_artifacts(snapshot_id,kind,document_id,chunk_id);

CREATE FUNCTION kf_reject_policy_snapshot_artifact_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'policy snapshot artifacts are immutable' USING ERRCODE='55000';
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trg_policy_snapshot_artifacts_immutable
    BEFORE UPDATE OR DELETE ON policy_snapshot_artifacts
    FOR EACH ROW EXECUTE FUNCTION kf_reject_policy_snapshot_artifact_mutation();
"""


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version="0001_document_content_revisions",
        description="Version document content and preserve immutable ingestion lineage",
        sql=REVISION_SCHEMA_SQL,
    ),
    Migration(
        version="0002_relation_shadow_work_queue",
        description="Persist exact resumable relation shadow work and attempt lineage",
        sql=RELATION_SHADOW_QUEUE_SQL,
    ),
    Migration(
        version="0003_entity_shadow_pipeline",
        description="Persist independent span-first entity shadow treatment and audit lineage",
        sql=ENTITY_SHADOW_PIPELINE_SQL,
    ),
    Migration(
        version="0004_policy_snapshot_orchestrator",
        description="Persist immutable resumable policy snapshot work and artifacts",
        sql=POLICY_SNAPSHOT_ORCHESTRATOR_SQL,
    ),
)


class MigrationRunner:
    """Apply immutable migrations exactly once under a PostgreSQL advisory lock."""

    def __init__(self, migrations: Iterable[Migration] = MIGRATIONS):
        self.migrations = tuple(migrations)

    @staticmethod
    def ensure_table(conn: psycopg.Connection) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations(
               version TEXT PRIMARY KEY,description TEXT NOT NULL,checksum TEXT NOT NULL,
               status TEXT NOT NULL CHECK(status IN ('applying','applied','failed')),
               duration_ms BIGINT NOT NULL DEFAULT 0,error_message TEXT NOT NULL DEFAULT '',
               applied_at TIMESTAMPTZ,updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"""
        )

    def status(self, conn: psycopg.Connection) -> list[dict[str, str]]:
        self.ensure_table(conn)
        applied = {
            row["version"]: row
            for row in conn.execute("SELECT * FROM schema_migrations ORDER BY version").fetchall()
        }
        return [
            {
                "version": item.version,
                "description": item.description,
                "checksum": item.checksum,
                "status": str((applied.get(item.version) or {}).get("status") or "pending"),
            }
            for item in self.migrations
        ]

    def is_applied(self, conn: psycopg.Connection, version: str) -> bool:
        """Return whether an exact immutable migration is already committed."""

        self.ensure_table(conn)
        row = conn.execute(
            "SELECT checksum,status FROM schema_migrations WHERE version=%s",
            (version,),
        ).fetchone()
        if not row or row["status"] != "applied":
            return False
        migration = next((item for item in self.migrations if item.version == version), None)
        if migration is None:
            return True
        if row["checksum"] != migration.checksum:
            raise RuntimeError(
                f"Migration checksum mismatch for {version}: "
                f"database={row['checksum']} code={migration.checksum}"
            )
        return True

    def apply(self, conn: psycopg.Connection) -> list[str]:
        self.ensure_table(conn)
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        applied: list[str] = []
        for migration in self.migrations:
            row = conn.execute(
                "SELECT checksum,status FROM schema_migrations WHERE version=%s FOR UPDATE",
                (migration.version,),
            ).fetchone()
            if row:
                if row["checksum"] != migration.checksum:
                    raise RuntimeError(
                        f"Migration checksum mismatch for {migration.version}: "
                        f"database={row['checksum']} code={migration.checksum}"
                    )
                if row["status"] == "applied":
                    continue
            started = time.monotonic()
            conn.execute(
                """INSERT INTO schema_migrations(version,description,checksum,status)
                   VALUES(%s,%s,%s,'applying')
                   ON CONFLICT(version) DO UPDATE SET status='applying',error_message='',updated_at=NOW()""",
                (migration.version, migration.description, migration.checksum),
            )
            try:
                conn.execute(migration.sql)
            except Exception as exc:
                # PostgreSQL aborts the surrounding transaction; the caller receives the
                # original error and no partial migration can be committed.
                raise RuntimeError(f"Migration {migration.version} failed: {exc}") from exc
            duration_ms = int((time.monotonic() - started) * 1000)
            conn.execute(
                """UPDATE schema_migrations SET status='applied',duration_ms=%s,
                   applied_at=NOW(),updated_at=NOW() WHERE version=%s""",
                (duration_ms, migration.version),
            )
            applied.append(migration.version)
        return applied
