"""Versioned PostgreSQL migrations with checksum and advisory-lock protection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Callable, Iterable

import psycopg


MIGRATION_LOCK_ID = 4_919_726_401


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    sql: str
    startup_safe: bool = False
    transactional: bool = True

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


RELATION_WORK_QUEUE_SQL = r"""
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


GROUNDED_ENTITY_PIPELINE_SQL = r"""
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


RELEASE_GATE_ARTIFACTS_SQL = r"""
CREATE TABLE gold_manifests (
    id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    dataset TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    level TEXT NOT NULL CHECK(level IN ('pilot','release')),
    manifest_hash TEXT NOT NULL UNIQUE,
    annotations_hash TEXT NOT NULL,
    manifest JSONB NOT NULL,
    validation_report JSONB NOT NULL,
    trusted_approval JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id,dataset,dataset_version)
);

CREATE TABLE gold_manifest_items (
    gold_manifest_id TEXT NOT NULL REFERENCES gold_manifests(id),
    item_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('dev','holdout')),
    content_hash TEXT NOT NULL,
    item_hash TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(gold_manifest_id,item_id)
);

ALTER TABLE quality_evaluation_runs
    ADD COLUMN gold_manifest_id TEXT REFERENCES gold_manifests(id),
    ADD COLUMN evaluator_version TEXT NOT NULL DEFAULT 'legacy-untrusted',
    ADD COLUMN artifact_root_hash TEXT NOT NULL DEFAULT '',
    ADD COLUMN gold_manifest_hash TEXT NOT NULL DEFAULT '',
    ADD COLUMN gold_items_hash TEXT NOT NULL DEFAULT '',
    ADD COLUMN report_hash TEXT NOT NULL DEFAULT '',
    ADD COLUMN item_hashes JSONB NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN release_level TEXT NOT NULL DEFAULT 'pilot'
        CHECK(release_level IN ('pilot','release')),
    ADD COLUMN evaluation_attempt INTEGER NOT NULL DEFAULT 1 CHECK(evaluation_attempt>0);

CREATE TABLE quality_evaluation_items (
    evaluation_run_id TEXT NOT NULL REFERENCES quality_evaluation_runs(id),
    item_id TEXT NOT NULL,
    artifact_hash TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(evaluation_run_id,item_id)
);

ALTER TABLE policy_activation_gates DROP CONSTRAINT IF EXISTS policy_activation_gates_project_id_snapshot_id_key;
CREATE UNIQUE INDEX uq_policy_activation_gate_evaluation
    ON policy_activation_gates(evaluation_run_id);

CREATE FUNCTION kf_reject_quality_fact_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'quality and Gold facts are append-only' USING ERRCODE='55000';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_gold_manifests_immutable BEFORE UPDATE OR DELETE ON gold_manifests
    FOR EACH ROW EXECUTE FUNCTION kf_reject_quality_fact_mutation();
CREATE TRIGGER trg_gold_manifest_items_immutable BEFORE UPDATE OR DELETE ON gold_manifest_items
    FOR EACH ROW EXECUTE FUNCTION kf_reject_quality_fact_mutation();
CREATE TRIGGER trg_quality_evaluation_runs_immutable BEFORE UPDATE OR DELETE ON quality_evaluation_runs
    FOR EACH ROW EXECUTE FUNCTION kf_reject_quality_fact_mutation();
CREATE TRIGGER trg_quality_evaluation_items_immutable BEFORE UPDATE OR DELETE ON quality_evaluation_items
    FOR EACH ROW EXECUTE FUNCTION kf_reject_quality_fact_mutation();
CREATE TRIGGER trg_policy_activation_gates_immutable BEFORE UPDATE OR DELETE ON policy_activation_gates
    FOR EACH ROW EXECUTE FUNCTION kf_reject_quality_fact_mutation();
"""


GRAPH_NAMESPACE_MANIFEST_SQL = r"""
ALTER TABLE projects ADD COLUMN graph_namespace_id TEXT DEFAULT gen_random_uuid()::text;
UPDATE projects
SET graph_namespace_id=(
    substr(md5(random()::text||clock_timestamp()::text||id::text),1,8)||'-'||
    substr(md5(random()::text||clock_timestamp()::text||id::text),1,4)||'-4'||
    substr(md5(random()::text||clock_timestamp()::text||id::text),1,3)||'-a'||
    substr(md5(random()::text||clock_timestamp()::text||id::text),1,3)||'-'||
    substr(md5(random()::text||clock_timestamp()::text||id::text),1,12)
)
WHERE graph_namespace_id IS NULL;
ALTER TABLE projects ALTER COLUMN graph_namespace_id SET NOT NULL;
ALTER TABLE projects ADD CONSTRAINT uq_projects_graph_namespace UNIQUE(graph_namespace_id);
ALTER TABLE projects ADD CONSTRAINT uq_projects_id_graph_namespace
    UNIQUE(id,graph_namespace_id);

UPDATE projects SET graph_generation=graph_generation+1,graph_status='stale',
    graph_policy_version='',graph_policy_hash='',graph_error='graph namespace migration requires rebuild'
WHERE graph_status<>'empty';

CREATE TABLE graph_leases (
    id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    namespace_id TEXT NOT NULL,
    graph_generation BIGINT NOT NULL,
    owner TEXT NOT NULL,
    lease_token TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK(status IN ('active','released','expired','superseded')),
    attempt INTEGER NOT NULL CHECK(attempt>0),
    heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    released_at TIMESTAMPTZ,
    FOREIGN KEY(project_id,namespace_id) REFERENCES projects(id,graph_namespace_id)
);
CREATE UNIQUE INDEX uq_graph_lease_active_generation
    ON graph_leases(project_id,namespace_id,graph_generation) WHERE status='active';
CREATE INDEX idx_graph_lease_recovery
    ON graph_leases(status,expires_at,project_id,graph_generation);

CREATE TABLE graph_content_manifests (
    id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    namespace_id TEXT NOT NULL,
    graph_generation BIGINT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('expected','writer','actual')),
    protocol TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    lease_token TEXT NOT NULL REFERENCES graph_leases(lease_token),
    fact_count BIGINT NOT NULL CHECK(fact_count>=0),
    content_digest TEXT NOT NULL,
    manifest JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY(project_id,namespace_id) REFERENCES projects(id,graph_namespace_id),
    UNIQUE(project_id,namespace_id,graph_generation,lease_token,role)
);
CREATE INDEX idx_graph_manifests_generation
    ON graph_content_manifests(project_id,namespace_id,graph_generation,lease_token,role);

CREATE FUNCTION kf_reject_graph_manifest_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'graph content manifests are immutable' USING ERRCODE='55000';
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trg_graph_content_manifests_immutable
    BEFORE UPDATE OR DELETE ON graph_content_manifests
    FOR EACH ROW EXECUTE FUNCTION kf_reject_graph_manifest_mutation();
"""


INDEXING_FAILURE_STATUS_SQL = r"""
DO $$
DECLARE definition TEXT;
BEGIN
    SELECT pg_get_constraintdef(oid) INTO definition
    FROM pg_constraint
    WHERE conrelid='indexing_failures'::regclass
      AND conname='indexing_failures_status_check';
    IF definition IS NOT NULL AND position('permanent_failed' IN definition)=0 THEN
        ALTER TABLE indexing_failures DROP CONSTRAINT indexing_failures_status_check;
        ALTER TABLE indexing_failures ADD CONSTRAINT indexing_failures_status_check
            CHECK (status IN ('pending','completed','permanent_failed'));
    END IF;
END;
$$;
"""


MENTION_OCCURRENCE_IDENTITY_SQL = r"""
ALTER TABLE entity_resolution_decisions
    ADD COLUMN IF NOT EXISTS mention_key TEXT NOT NULL DEFAULT '';

UPDATE entity_resolution_decisions decision
SET mention_key='mention_' || md5(
    concat_ws(E'\x1f',decision.project_id::text,decision.document_id::text,
              decision.revision_id,decision.chunk_id::text,decision.entity_type,
              decision.mention_text))
WHERE mention_key='';

DROP INDEX IF EXISTS uq_entity_mentions_revision_payload;
DROP INDEX IF EXISTS uq_resolution_revision_key;
DROP INDEX IF EXISTS idx_entity_alias_exact;

CREATE UNIQUE INDEX IF NOT EXISTS uq_resolution_revision_mention
    ON entity_resolution_decisions(revision_id,mention_key)
    WHERE mention_key<>'' AND is_deleted=FALSE;

CREATE INDEX IF NOT EXISTS idx_entity_alias_exact
    ON canonical_entity_aliases(project_id,normalized_alias,entity_type,canonical_entity_id)
    WHERE active=TRUE;
"""


POLICY_SNAPSHOT_COVERAGE_SQL = r"""
ALTER TABLE policy_snapshots
    ADD COLUMN IF NOT EXISTS coverage_complete BOOLEAN NOT NULL DEFAULT TRUE;
"""


GRAPHRAG_COMMUNITIES_SQL = r"""
CREATE UNIQUE INDEX uq_canonical_entities_id_project
    ON canonical_entities(id,project_id);

CREATE TABLE graphrag_community_runs (
    run_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    graph_generation BIGINT NOT NULL,
    graph_policy_hash TEXT NOT NULL,
    active_policy_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    graphrag_version TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    weighting_strategy TEXT NOT NULL,
    seed INTEGER NOT NULL,
    max_cluster_size INTEGER NOT NULL CHECK(max_cluster_size>0),
    use_lcc BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL CHECK(
        status IN ('running','completed','completed_empty','failed','superseded')
    ),
    node_count INTEGER NOT NULL DEFAULT 0 CHECK(node_count>=0),
    aggregated_edge_count INTEGER NOT NULL DEFAULT 0 CHECK(aggregated_edge_count>=0),
    excluded_self_loops INTEGER NOT NULL DEFAULT 0 CHECK(excluded_self_loops>=0),
    excluded_isolated_entities INTEGER NOT NULL DEFAULT 0
        CHECK(excluded_isolated_entities>=0),
    community_count INTEGER NOT NULL DEFAULT 0 CHECK(community_count>=0),
    hierarchy_levels INTEGER NOT NULL DEFAULT 0 CHECK(hierarchy_levels>=0),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(run_id,project_id)
);
CREATE INDEX idx_graphrag_community_runs_project
    ON graphrag_community_runs(project_id,status,created_at DESC);
CREATE INDEX idx_graphrag_community_runs_current
    ON graphrag_community_runs(project_id,input_hash,created_at DESC)
    WHERE status IN ('completed','completed_empty');

CREATE TABLE graphrag_communities (
    run_id TEXT NOT NULL,
    project_id BIGINT NOT NULL,
    stable_community_id TEXT NOT NULL,
    graphrag_output_id TEXT NOT NULL,
    graphrag_community INTEGER NOT NULL,
    level INTEGER NOT NULL CHECK(level>=0),
    parent_stable_community_id TEXT,
    title TEXT NOT NULL,
    size INTEGER NOT NULL CHECK(size>=0),
    period TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(run_id,stable_community_id),
    UNIQUE(run_id,project_id,stable_community_id),
    FOREIGN KEY(run_id,project_id)
        REFERENCES graphrag_community_runs(run_id,project_id) ON DELETE CASCADE,
    FOREIGN KEY(run_id,parent_stable_community_id)
        REFERENCES graphrag_communities(run_id,stable_community_id)
        DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX idx_graphrag_communities_project
    ON graphrag_communities(project_id,run_id,level,graphrag_community);

CREATE TABLE graphrag_community_members (
    run_id TEXT NOT NULL,
    project_id BIGINT NOT NULL,
    stable_community_id TEXT NOT NULL,
    canonical_entity_id BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(run_id,stable_community_id,canonical_entity_id),
    FOREIGN KEY(run_id,project_id,stable_community_id)
        REFERENCES graphrag_communities(run_id,project_id,stable_community_id)
        ON DELETE CASCADE,
    FOREIGN KEY(canonical_entity_id,project_id)
        REFERENCES canonical_entities(id,project_id)
);
CREATE INDEX idx_graphrag_community_members_entity
    ON graphrag_community_members(project_id,canonical_entity_id,run_id);

CREATE TABLE graphrag_community_edges (
    run_id TEXT NOT NULL,
    project_id BIGINT NOT NULL,
    stable_edge_id TEXT NOT NULL,
    source_entity_id BIGINT NOT NULL,
    target_entity_id BIGINT NOT NULL,
    weight DOUBLE PRECISION NOT NULL CHECK(
        weight>0 AND weight NOT IN (
            'NaN'::DOUBLE PRECISION,
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION
        )
    ),
    assertion_ids BIGINT[] NOT NULL DEFAULT '{}',
    chunk_ids BIGINT[] NOT NULL DEFAULT '{}',
    relation_types TEXT[] NOT NULL DEFAULT '{}',
    text_unit_ids TEXT[] NOT NULL DEFAULT '{}',
    provenance_hash TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(run_id,stable_edge_id),
    FOREIGN KEY(run_id,project_id)
        REFERENCES graphrag_community_runs(run_id,project_id) ON DELETE CASCADE,
    FOREIGN KEY(source_entity_id,project_id)
        REFERENCES canonical_entities(id,project_id),
    FOREIGN KEY(target_entity_id,project_id)
        REFERENCES canonical_entities(id,project_id),
    CHECK(source_entity_id<>target_entity_id)
);
CREATE INDEX idx_graphrag_community_edges_project
    ON graphrag_community_edges(project_id,run_id,source_entity_id,target_entity_id);
"""


GRAPHRAG_REPORTS_SEARCH_SQL = r"""
CREATE TABLE graphrag_community_report_runs (
    run_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    graph_generation BIGINT NOT NULL,
    detection_run_id TEXT NOT NULL REFERENCES graphrag_community_runs(run_id),
    community_input_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    model_config_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    graphrag_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'pending','running','succeeded','retryable_failed','terminal_failed','superseded'
    )),
    active BOOLEAN NOT NULL DEFAULT FALSE,
    expected INTEGER NOT NULL DEFAULT 0 CHECK(expected>=0),
    completed INTEGER NOT NULL DEFAULT 0 CHECK(completed>=0),
    failed INTEGER NOT NULL DEFAULT 0 CHECK(failed>=0),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(project_id,graph_generation,detection_run_id,input_hash,prompt_hash,model_config_hash)
);
CREATE UNIQUE INDEX uq_graphrag_report_run_active
    ON graphrag_community_report_runs(project_id) WHERE active=TRUE;
CREATE INDEX idx_graphrag_report_runs_project
    ON graphrag_community_report_runs(project_id,status,created_at DESC);

CREATE TABLE graphrag_community_reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES graphrag_community_report_runs(run_id) ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    detection_run_id TEXT NOT NULL,
    community_id TEXT NOT NULL,
    level INTEGER NOT NULL CHECK(level>=0),
    status TEXT NOT NULL CHECK(status IN (
        'pending','running','succeeded','retryable_failed','terminal_failed','superseded'
    )),
    attempt INTEGER NOT NULL DEFAULT 1 CHECK(attempt>0),
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    findings JSONB NOT NULL DEFAULT '[]'::jsonb,
    entity_ids BIGINT[] NOT NULL DEFAULT '{}',
    relation_ids TEXT[] NOT NULL DEFAULT '{}',
    input_hash TEXT NOT NULL,
    output_hash TEXT NOT NULL DEFAULT '',
    usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms BIGINT NOT NULL DEFAULT 0 CHECK(latency_ms>=0),
    error_category TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id,community_id),
    FOREIGN KEY(detection_run_id,community_id)
        REFERENCES graphrag_communities(run_id,stable_community_id)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE graphrag_global_search_runs (
    run_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    report_run_id TEXT NOT NULL REFERENCES graphrag_community_report_runs(run_id),
    query_hash TEXT NOT NULL,
    report_input_hash TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    model_config_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    graphrag_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'pending','running','succeeded','retryable_failed','terminal_failed','superseded'
    )),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    selected_report_ids TEXT[] NOT NULL DEFAULT '{}',
    usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms BIGINT NOT NULL DEFAULT 0 CHECK(latency_ms>=0),
    error_category TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX idx_graphrag_global_search_project
    ON graphrag_global_search_runs(project_id,created_at DESC);
"""


MULTI_STRATEGY_CONSTRUCTION_SQL = r"""
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS strategy_key TEXT NOT NULL DEFAULT 'stable';
CREATE INDEX IF NOT EXISTS idx_jobs_project_strategy
    ON jobs(project_id,strategy_key,updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_document_strategy
    ON jobs(document_id,strategy_key,updated_at DESC)
    WHERE document_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS document_strategy_builds (
    project_id BIGINT NOT NULL,
    document_id BIGINT NOT NULL,
    strategy_key TEXT NOT NULL CHECK (strategy_key IN ('stable','enhanced_shadow')),
    status TEXT NOT NULL DEFAULT 'uploaded' CHECK (
        status IN ('uploaded','queued','processing','indexed','failed','cancelled','deleted')
    ),
    content_revision_id TEXT REFERENCES document_content_revisions(revision_id),
    latest_job_id TEXT REFERENCES jobs(id),
    stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(document_id,strategy_key),
    CONSTRAINT fk_document_strategy_document
        FOREIGN KEY(document_id,project_id) REFERENCES documents(id,project_id)
);
CREATE INDEX IF NOT EXISTS idx_document_strategy_project
    ON document_strategy_builds(project_id,strategy_key,status,updated_at DESC);

CREATE TABLE IF NOT EXISTS project_strategy_graphs (
    project_id BIGINT NOT NULL REFERENCES projects(id),
    strategy_key TEXT NOT NULL CHECK (strategy_key IN ('stable','enhanced_shadow')),
    namespace_id TEXT NOT NULL UNIQUE,
    generation BIGINT NOT NULL DEFAULT 0,
    ready_generation BIGINT NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'empty' CHECK (
        status IN ('empty','stale','rebuilding','ready','failed')
    ),
    error_message TEXT NOT NULL DEFAULT '',
    policy_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    lease_token TEXT NOT NULL DEFAULT '',
    lease_expires_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(project_id,strategy_key)
);
CREATE INDEX IF NOT EXISTS idx_project_strategy_graph_status
    ON project_strategy_graphs(project_id,status,updated_at DESC);

INSERT INTO document_strategy_builds(
    project_id,document_id,strategy_key,status,content_revision_id,stats,error_message,
    started_at,completed_at,updated_at
)
SELECT d.project_id,d.id,'stable',
       CASE
           WHEN d.is_deleted THEN 'deleted'
           WHEN d.status='indexed' THEN 'indexed'
           WHEN d.status='processing' THEN 'processing'
           WHEN d.status='failed' THEN 'failed'
           ELSE 'uploaded'
       END,
       d.active_revision_id,
       jsonb_build_object(
           'section_count',COALESCE((d.metadata->>'section_count')::integer,0),
           'chunk_count',COALESCE((d.metadata->>'chunk_count')::integer,0),
           'entity_mentions',COALESCE((d.metadata->>'entity_mentions')::integer,0),
           'relations',COALESCE((d.metadata->>'relations')::integer,0)
       ),
       COALESCE(d.error_message,''),
       d.uploaded_at,
       CASE WHEN d.status='indexed' THEN COALESCE(d.indexed_at,d.updated_at) ELSE NULL END,
       d.updated_at
FROM documents d
ON CONFLICT(document_id,strategy_key) DO NOTHING;

INSERT INTO project_strategy_graphs(
    project_id,strategy_key,namespace_id,generation,ready_generation,status,
    error_message,policy_summary,updated_at
)
SELECT p.id,'stable',p.graph_namespace_id,p.graph_generation,p.graph_ready_generation,
       p.graph_status,COALESCE(p.graph_error,''),
       jsonb_build_object(
           'policy_version',COALESCE(p.graph_policy_version,''),
           'policy_hash',COALESCE(p.graph_policy_hash,'')
       ),p.updated_at
FROM projects p
ON CONFLICT(project_id,strategy_key) DO UPDATE SET
    namespace_id=EXCLUDED.namespace_id,
    generation=EXCLUDED.generation,
    ready_generation=EXCLUDED.ready_generation,
    status=EXCLUDED.status,
    error_message=EXCLUDED.error_message,
    policy_summary=EXCLUDED.policy_summary,
    updated_at=EXCLUDED.updated_at;

INSERT INTO project_strategy_graphs(project_id,strategy_key,namespace_id,status,policy_summary)
SELECT p.id,'enhanced_shadow',
       (SUBSTRING(md5('knowledge-forge:'||p.graph_namespace_id||':enhanced-shadow'),1,8)||'-'||
        SUBSTRING(md5('knowledge-forge:'||p.graph_namespace_id||':enhanced-shadow'),9,4)||'-'||
        SUBSTRING(md5('knowledge-forge:'||p.graph_namespace_id||':enhanced-shadow'),13,4)||'-'||
        SUBSTRING(md5('knowledge-forge:'||p.graph_namespace_id||':enhanced-shadow'),17,4)||'-'||
        SUBSTRING(md5('knowledge-forge:'||p.graph_namespace_id||':enhanced-shadow'),21,12)),
       'empty',
       jsonb_build_object('mode','shadow','production_approved',FALSE)
FROM projects p
ON CONFLICT(project_id,strategy_key) DO NOTHING;
"""


STRATEGY_GRAPH_LEASE_NAMESPACE_SQL = r"""
-- A strategy graph owns a namespace through project_strategy_graphs rather than
-- projects.graph_namespace_id. Project ownership is still enforced by the
-- existing project_id foreign keys and by Database.acquire_graph_lease().
ALTER TABLE graph_content_manifests
    DROP CONSTRAINT IF EXISTS graph_content_manifests_project_id_namespace_id_fkey;
ALTER TABLE graph_leases
    DROP CONSTRAINT IF EXISTS graph_leases_project_id_namespace_id_fkey;
"""


GROUNDED_RELATION_V2_LEDGER_SQL = r"""
CREATE TABLE IF NOT EXISTS shadow_relation_v2_runs (
    id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    snapshot_id TEXT,
    input_fingerprint TEXT NOT NULL,
    source_generation BIGINT NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN (
        'planning','classifying','verifying','normalizing','ready','stale','failed','cancelled'
    )),
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_v2_runs_project
    ON shadow_relation_v2_runs(project_id,status,created_at DESC);

CREATE TABLE IF NOT EXISTS shadow_relation_pair_opportunities (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id),
    project_id BIGINT NOT NULL REFERENCES projects(id),
    document_revision_id TEXT NOT NULL,
    document_id BIGINT,
    chunk_id BIGINT,
    chunk_key TEXT NOT NULL,
    pair_identity TEXT NOT NULL,
    candidate_policy_version TEXT NOT NULL,
    left_occurrence_key TEXT NOT NULL,
    right_occurrence_key TEXT NOT NULL,
    left_mention_id TEXT NOT NULL,
    right_mention_id TEXT NOT NULL,
    left_start INTEGER NOT NULL,
    left_end INTEGER NOT NULL,
    right_start INTEGER NOT NULL,
    right_end INTEGER NOT NULL,
    left_canonical_id TEXT NOT NULL DEFAULT '',
    right_canonical_id TEXT NOT NULL DEFAULT '',
    left_assignment_version TEXT NOT NULL DEFAULT '',
    right_assignment_version TEXT NOT NULL DEFAULT '',
    context_identity TEXT NOT NULL,
    context_start INTEGER NOT NULL,
    context_end INTEGER NOT NULL,
    allowed_predicate_metadata JSONB NOT NULL DEFAULT '[]'::jsonb,
    planning_reason TEXT NOT NULL,
    cue TEXT NOT NULL DEFAULT '',
    type_status TEXT NOT NULL,
    status TEXT NOT NULL,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id,candidate_policy_version,pair_identity),
    CHECK(left_occurrence_key < right_occurrence_key),
    CHECK(left_start >= 0 AND left_end > left_start),
    CHECK(right_start >= 0 AND right_end > right_start),
    CHECK(context_start >= 0 AND context_end > context_start)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_pairs_run_status
    ON shadow_relation_pair_opportunities(run_id,status,pair_identity);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_pairs_cross_policy
    ON shadow_relation_pair_opportunities(project_id,document_revision_id,pair_identity);

CREATE TABLE IF NOT EXISTS shadow_relation_hypotheses (
    id TEXT PRIMARY KEY,
    pair_opportunity_id TEXT NOT NULL REFERENCES shadow_relation_pair_opportunities(id),
    hypothesis_key TEXT NOT NULL,
    predicate TEXT NOT NULL,
    subject_occurrence_key TEXT NOT NULL,
    object_occurrence_key TEXT NOT NULL,
    evidence_text TEXT NOT NULL DEFAULT '',
    evidence_start INTEGER,
    evidence_end INTEGER,
    classifier_policy_version TEXT NOT NULL,
    classifier_model_revision TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(
        pair_opportunity_id,subject_occurrence_key,predicate,
        object_occurrence_key,hypothesis_key
    )
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_hypotheses_pair
    ON shadow_relation_hypotheses(pair_opportunity_id,status,predicate);
"""


GROUNDED_RELATION_V2_VERIFICATION_SQL = r"""
CREATE TABLE IF NOT EXISTS shadow_relation_verification_attempts (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT NOT NULL REFERENCES shadow_relation_hypotheses(id),
    attempt_no INTEGER NOT NULL CHECK(attempt_no > 0),
    verdict TEXT NOT NULL CHECK(verdict IN (
        'SUPPORTED','REJECTED','UNCERTAIN','RETRYABLE_FAILED'
    )),
    reason_code TEXT NOT NULL,
    supported_evidence_spans JSONB NOT NULL DEFAULT '[]'::jsonb,
    independent_call BOOLEAN NOT NULL,
    verifier_prompt_version TEXT NOT NULL,
    verifier_schema_version TEXT NOT NULL,
    verifier_parser_version TEXT NOT NULL,
    verifier_model_revision TEXT NOT NULL,
    request_hash TEXT NOT NULL DEFAULT '',
    response_hash TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(hypothesis_id,attempt_no)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_verification_verdict
    ON shadow_relation_verification_attempts(hypothesis_id,verdict,attempt_no);
"""


GROUNDED_RELATION_V2_NORMALIZATION_SQL = r"""
CREATE TABLE IF NOT EXISTS shadow_relation_normalized_facts (
    id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    semantic_fact_key TEXT NOT NULL,
    subject_canonical_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_canonical_id TEXT NOT NULL,
    is_symmetric BOOLEAN NOT NULL DEFAULT FALSE,
    is_reflexive BOOLEAN NOT NULL DEFAULT FALSE,
    is_review_only BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(project_id,semantic_fact_key)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_facts_endpoints
    ON shadow_relation_normalized_facts(
        project_id,subject_canonical_id,predicate,object_canonical_id
    );

CREATE TABLE IF NOT EXISTS shadow_relation_fact_memberships (
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id),
    fact_id TEXT NOT NULL REFERENCES shadow_relation_normalized_facts(id),
    input_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN (
        'NORMALIZED','REVIEW_ONLY','ELIGIBLE','STALE','CONFLICT'
    )),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(run_id,fact_id)
);

CREATE TABLE IF NOT EXISTS shadow_relation_fact_evidence (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES shadow_relation_normalized_facts(id),
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id),
    evidence_key TEXT NOT NULL,
    document_revision_id TEXT NOT NULL,
    subject_occurrence_key TEXT NOT NULL,
    object_occurrence_key TEXT NOT NULL,
    evidence_spans JSONB NOT NULL,
    hypothesis_id TEXT NOT NULL REFERENCES shadow_relation_hypotheses(id),
    verification_attempt_id TEXT NOT NULL
        REFERENCES shadow_relation_verification_attempts(id),
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(fact_id,evidence_key)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_evidence_run
    ON shadow_relation_fact_evidence(run_id,fact_id);

CREATE TABLE IF NOT EXISTS shadow_relation_fact_collapse_audit (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id),
    fact_id TEXT NOT NULL REFERENCES shadow_relation_normalized_facts(id),
    hypothesis_id TEXT NOT NULL REFERENCES shadow_relation_hypotheses(id),
    reason_code TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id,fact_id,hypothesis_id,reason_code)
);

CREATE TABLE IF NOT EXISTS shadow_relation_fact_conflicts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id),
    pair_identity TEXT NOT NULL,
    hypothesis_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id,hypothesis_id,reason_code)
);

CREATE TABLE IF NOT EXISTS shadow_relation_legacy_duplicate_audit (
    project_id BIGINT NOT NULL REFERENCES projects(id),
    legacy_group_key TEXT NOT NULL,
    duplicate_count INTEGER NOT NULL,
    assertion_ids JSONB NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(project_id,legacy_group_key)
);
INSERT INTO shadow_relation_legacy_duplicate_audit(
    project_id,legacy_group_key,duplicate_count,assertion_ids
)
SELECT project_id,
       md5(
           project_id::text||':'||source_entity_id::text||':'||
           relation_type||':'||target_entity_id::text
       ),
       COUNT(*)::integer,
       jsonb_agg(id ORDER BY id)
FROM relation_assertions
WHERE is_deleted=FALSE
GROUP BY project_id,source_entity_id,relation_type,target_entity_id
HAVING COUNT(*)>1
ON CONFLICT(project_id,legacy_group_key) DO NOTHING;
"""


GROUNDED_RELATION_V2_FRESHNESS_SQL = r"""
ALTER TABLE policy_snapshots
    ADD COLUMN IF NOT EXISTS materialization_fingerprint TEXT NOT NULL DEFAULT '';
ALTER TABLE policy_snapshots
    ADD COLUMN IF NOT EXISTS freshness_components JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE policy_snapshots
    ADD COLUMN IF NOT EXISTS stale_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS shadow_relation_current_fingerprints (
    project_id BIGINT PRIMARY KEY REFERENCES projects(id),
    current_fingerprint TEXT NOT NULL,
    components JSONB NOT NULL,
    source_generation BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION kf_block_shadow_relation_product_write()
RETURNS trigger AS $$
BEGIN
    IF NULLIF(
        current_setting('knowledge_forge.shadow_relation_v2_run_id',TRUE),
        ''
    ) IS NOT NULL THEN
        RAISE EXCEPTION 'Shadow relation v2 cannot write product table %',TG_TABLE_NAME
            USING ERRCODE='object_not_in_prerequisite_state';
    END IF;
    RETURN COALESCE(NEW,OLD);
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_shadow_v2_block_relation_assertions ON relation_assertions;
CREATE TRIGGER trg_shadow_v2_block_relation_assertions
    BEFORE INSERT OR UPDATE OR DELETE ON relation_assertions
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
DROP TRIGGER IF EXISTS trg_shadow_v2_block_relation_projections ON canonical_relation_projections;
CREATE TRIGGER trg_shadow_v2_block_relation_projections
    BEFORE INSERT OR UPDATE OR DELETE ON canonical_relation_projections
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
DROP TRIGGER IF EXISTS trg_shadow_v2_block_vectors ON vector_embeddings;
CREATE TRIGGER trg_shadow_v2_block_vectors
    BEFORE INSERT OR UPDATE OR DELETE ON vector_embeddings
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
DROP TRIGGER IF EXISTS trg_shadow_v2_block_indexing_outbox ON indexing_failures;
CREATE TRIGGER trg_shadow_v2_block_indexing_outbox
    BEFORE INSERT OR UPDATE OR DELETE ON indexing_failures
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
DROP TRIGGER IF EXISTS trg_shadow_v2_block_mentions ON entity_mentions;
CREATE TRIGGER trg_shadow_v2_block_mentions
    BEFORE INSERT OR UPDATE OR DELETE ON entity_mentions
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
DROP TRIGGER IF EXISTS trg_shadow_v2_block_assignments ON entity_assignments;
CREATE TRIGGER trg_shadow_v2_block_assignments
    BEFORE INSERT OR UPDATE OR DELETE ON entity_assignments
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
DROP TRIGGER IF EXISTS trg_shadow_v2_block_aliases ON canonical_entity_aliases;
CREATE TRIGGER trg_shadow_v2_block_aliases
    BEFORE INSERT OR UPDATE OR DELETE ON canonical_entity_aliases
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
DROP TRIGGER IF EXISTS trg_shadow_v2_block_project_state ON projects;
CREATE TRIGGER trg_shadow_v2_block_project_state
    BEFORE INSERT OR UPDATE OR DELETE ON projects
    FOR EACH ROW EXECUTE FUNCTION kf_block_shadow_relation_product_write();
"""


GROUNDED_RELATION_V2_GOLD_SQL = r"""
CREATE TABLE IF NOT EXISTS shadow_relation_gold_pair_decisions (
    gold_manifest_id TEXT NOT NULL REFERENCES gold_manifests(id),
    pair_id TEXT NOT NULL,
    split TEXT NOT NULL CHECK(split IN ('dev','holdout')),
    document_revision_id TEXT NOT NULL,
    document_content_hash TEXT NOT NULL,
    left_occurrence_id TEXT NOT NULL,
    left_span JSONB NOT NULL,
    left_canonical_id TEXT NOT NULL,
    right_occurrence_id TEXT NOT NULL,
    right_span JSONB NOT NULL,
    right_canonical_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN (
        'NO_RELATION','OTHER_UNSUPPORTED','HAS_RELATION'
    )),
    evidence_spans JSONB NOT NULL DEFAULT '[]'::jsonb,
    unsupported_description TEXT NOT NULL DEFAULT '',
    critical BOOLEAN NOT NULL DEFAULT FALSE,
    annotator TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reviewed_at TIMESTAMPTZ NOT NULL,
    adjudication_status TEXT NOT NULL CHECK(adjudication_status='adjudicated'),
    payload_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(gold_manifest_id,pair_id)
);

CREATE TABLE IF NOT EXISTS shadow_relation_gold_facts (
    id TEXT PRIMARY KEY,
    gold_manifest_id TEXT NOT NULL REFERENCES gold_manifests(id),
    pair_id TEXT NOT NULL,
    subject_occurrence_id TEXT NOT NULL,
    subject_canonical_id TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_occurrence_id TEXT NOT NULL,
    object_canonical_id TEXT NOT NULL,
    evidence_spans JSONB NOT NULL,
    critical BOOLEAN NOT NULL DEFAULT FALSE,
    adjudication_status TEXT NOT NULL CHECK(adjudication_status='adjudicated'),
    payload_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(
        gold_manifest_id,pair_id,subject_occurrence_id,predicate,
        object_occurrence_id,payload_hash
    ),
    FOREIGN KEY(gold_manifest_id,pair_id)
        REFERENCES shadow_relation_gold_pair_decisions(gold_manifest_id,pair_id)
);

CREATE TABLE IF NOT EXISTS shadow_relation_prediction_adjudications (
    id TEXT PRIMARY KEY,
    gold_manifest_id TEXT NOT NULL REFERENCES gold_manifests(id),
    snapshot_id TEXT NOT NULL REFERENCES policy_snapshots(id),
    prediction_id TEXT NOT NULL,
    semantic_fact_key TEXT NOT NULL,
    correct BOOLEAN NOT NULL,
    critical_error BOOLEAN NOT NULL DEFAULT FALSE,
    annotator TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reviewed_at TIMESTAMPTZ NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    payload_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(gold_manifest_id,snapshot_id,prediction_id)
);
"""


GROUNDED_RELATION_V2_EXECUTION_SQL = r"""
ALTER TABLE shadow_relation_v2_runs
    ADD COLUMN IF NOT EXISTS fingerprint_components JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS shadow_relation_classification_attempts (
    id TEXT PRIMARY KEY,
    pair_opportunity_id TEXT NOT NULL REFERENCES shadow_relation_pair_opportunities(id),
    attempt_no INTEGER NOT NULL CHECK(attempt_no > 0),
    status TEXT NOT NULL CHECK(status IN (
        'accepted','retryable_failed','permanent_failed','superseded'
    )),
    request_hash TEXT NOT NULL,
    response_hash TEXT NOT NULL DEFAULT '',
    classifier_prompt_version TEXT NOT NULL,
    classifier_schema_version TEXT NOT NULL,
    classifier_parser_version TEXT NOT NULL,
    classifier_model_revision TEXT NOT NULL,
    error_message TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(pair_opportunity_id,attempt_no)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_classification_attempts_pair
    ON shadow_relation_classification_attempts(pair_opportunity_id,status,attempt_no);

CREATE TABLE IF NOT EXISTS shadow_relation_hypothesis_validations (
    hypothesis_id TEXT PRIMARY KEY REFERENCES shadow_relation_hypotheses(id),
    status TEXT NOT NULL CHECK(status IN ('PASS','REVIEW_ONLY','REJECTED')),
    reason_code TEXT NOT NULL,
    predicate_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE shadow_relation_fact_evidence
    DROP CONSTRAINT IF EXISTS shadow_relation_fact_evidence_fact_id_evidence_key_key;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname='uq_shadow_relation_fact_evidence_run'
          AND conrelid='shadow_relation_fact_evidence'::regclass
    ) THEN
        ALTER TABLE shadow_relation_fact_evidence
            ADD CONSTRAINT uq_shadow_relation_fact_evidence_run
            UNIQUE(run_id,fact_id,evidence_key);
    END IF;
END
$$;
"""


GROUNDED_RELATION_V2_MODEL_AUDIT_SQL = r"""
ALTER TABLE shadow_relation_classification_attempts
    ADD COLUMN IF NOT EXISTS model_name TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS raw_response_text TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS parsed_response JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS error_type TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS independent_call BOOLEAN NOT NULL DEFAULT TRUE;

ALTER TABLE shadow_relation_classification_attempts
    DROP CONSTRAINT IF EXISTS shadow_relation_classification_attempts_status_check;

ALTER TABLE shadow_relation_classification_attempts
    ADD CONSTRAINT shadow_relation_classification_attempts_status_check
    CHECK(status IN (
        'ACCEPTED',
        'RETRYABLE_EMPTY',
        'RETRYABLE_PROTOCOL',
        'RETRYABLE_TRUNCATED',
        'RETRYABLE_TIMEOUT',
        'RETRYABLE_MODEL_ERROR',
        'TERMINAL_FAILED',
        'SUPERSEDED',
        'accepted',
        'retryable_failed',
        'permanent_failed',
        'superseded'
    ));

CREATE INDEX IF NOT EXISTS idx_shadow_relation_classification_attempts_status_v2
    ON shadow_relation_classification_attempts(status,attempt_no);
"""


GROUNDED_RELATION_V2_ARTIFACT_AUDIT_SQL = r"""
ALTER TABLE shadow_relation_v2_runs
    ADD COLUMN IF NOT EXISTS pipeline_generation BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS audit_quality TEXT NOT NULL DEFAULT 'legacy_audit_incomplete';

ALTER TABLE shadow_relation_pair_opportunities
    ADD COLUMN IF NOT EXISTS pipeline_generation BIGINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS context_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS source_kind TEXT NOT NULL DEFAULT 'v2_pair_planner',
    ADD COLUMN IF NOT EXISTS source_record_id TEXT NOT NULL DEFAULT '';

ALTER TABLE shadow_relation_classification_attempts
    ADD COLUMN IF NOT EXISTS model_call_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS audit_quality TEXT NOT NULL DEFAULT 'legacy_audit_incomplete',
    ADD COLUMN IF NOT EXISTS terminal_status TEXT NOT NULL DEFAULT '';

ALTER TABLE shadow_relation_hypotheses
    ADD COLUMN IF NOT EXISTS qualifiers JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS source_kind TEXT NOT NULL DEFAULT 'v2_classifier',
    ADD COLUMN IF NOT EXISTS source_record_id TEXT NOT NULL DEFAULT '';

ALTER TABLE shadow_relation_verification_attempts
    ADD COLUMN IF NOT EXISTS model_call_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS audit_quality TEXT NOT NULL DEFAULT 'legacy_audit_incomplete',
    ADD COLUMN IF NOT EXISTS terminal_status TEXT NOT NULL DEFAULT '';

ALTER TABLE shadow_relation_normalized_facts
    ADD COLUMN IF NOT EXISTS qualifiers JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS audit_quality TEXT NOT NULL DEFAULT 'legacy_audit_incomplete';

ALTER TABLE shadow_relation_fact_evidence
    ADD COLUMN IF NOT EXISTS context_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS source_kind TEXT NOT NULL DEFAULT 'v2_verifier',
    ADD COLUMN IF NOT EXISTS source_record_id TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS qualifiers JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS audit_quality TEXT NOT NULL DEFAULT 'legacy_audit_incomplete';

CREATE TABLE IF NOT EXISTS shadow_source_contexts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id) ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT,
    document_revision_id TEXT,
    chunk_id BIGINT,
    chunk_key TEXT NOT NULL DEFAULT '',
    context_kind TEXT NOT NULL,
    context_identity TEXT NOT NULL,
    context_start INTEGER NOT NULL DEFAULT 0 CHECK(context_start>=0),
    context_end INTEGER NOT NULL DEFAULT 0 CHECK(context_end>=context_start),
    content_hash TEXT NOT NULL,
    structure_artifact_hash TEXT NOT NULL DEFAULT '',
    parser_version TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id,context_identity),
    FOREIGN KEY(document_revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id),
    FOREIGN KEY(chunk_id,document_revision_id)
        REFERENCES chunks(id,revision_id)
);
CREATE INDEX IF NOT EXISTS idx_shadow_source_contexts_run
    ON shadow_source_contexts(run_id,document_revision_id,chunk_id,context_identity);

CREATE TABLE IF NOT EXISTS shadow_context_source_maps (
    id TEXT PRIMARY KEY,
    context_id TEXT NOT NULL REFERENCES shadow_source_contexts(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    source_start INTEGER NOT NULL DEFAULT 0 CHECK(source_start>=0),
    source_end INTEGER NOT NULL DEFAULT 0 CHECK(source_end>=source_start),
    payload_hash TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(context_id,source_kind,source_record_id,source_start,source_end)
);
CREATE INDEX IF NOT EXISTS idx_shadow_context_source_maps_context
    ON shadow_context_source_maps(context_id,source_kind,source_record_id);

CREATE TABLE IF NOT EXISTS entity_shadow_occurrences (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id) ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT,
    document_revision_id TEXT,
    chunk_id BIGINT,
    occurrence_key TEXT NOT NULL,
    physical_occurrence_key TEXT NOT NULL DEFAULT '',
    surface TEXT NOT NULL,
    normalized_surface TEXT NOT NULL,
    evidence_start INTEGER NOT NULL DEFAULT 0 CHECK(evidence_start>=0),
    evidence_end INTEGER NOT NULL DEFAULT 0 CHECK(evidence_end>=evidence_start),
    source_context_id TEXT REFERENCES shadow_source_contexts(id),
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(namespace_id,occurrence_key),
    FOREIGN KEY(document_revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id),
    FOREIGN KEY(chunk_id,document_revision_id)
        REFERENCES chunks(id,revision_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_occurrences_namespace
    ON entity_shadow_occurrences(namespace_id,document_revision_id,chunk_id,occurrence_key);

CREATE TABLE IF NOT EXISTS entity_shadow_mention_occurrence_map (
    namespace_id TEXT NOT NULL,
    mention_key TEXT NOT NULL,
    occurrence_id TEXT NOT NULL REFERENCES entity_shadow_occurrences(id) ON DELETE CASCADE,
    mapping_policy_version TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(namespace_id,mention_key),
    FOREIGN KEY(namespace_id,mention_key)
        REFERENCES entity_shadow_mentions(namespace_id,mention_key)
);

CREATE TABLE IF NOT EXISTS entity_shadow_occurrence_assignments (
    occurrence_id TEXT NOT NULL REFERENCES entity_shadow_occurrences(id) ON DELETE CASCADE,
    namespace_id TEXT NOT NULL,
    canonical_key TEXT NOT NULL,
    assignment_version TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    decision_reason TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(occurrence_id,assignment_version),
    FOREIGN KEY(namespace_id,canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key)
);
CREATE INDEX IF NOT EXISTS idx_entity_shadow_occurrence_assignments_canonical
    ON entity_shadow_occurrence_assignments(namespace_id,canonical_key,status);

CREATE TABLE IF NOT EXISTS entity_shadow_canonical_redirects (
    id TEXT PRIMARY KEY,
    namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id) ON DELETE CASCADE,
    source_canonical_key TEXT NOT NULL,
    target_canonical_key TEXT NOT NULL,
    redirect_policy_version TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(namespace_id,source_canonical_key),
    CHECK(source_canonical_key<>target_canonical_key),
    FOREIGN KEY(namespace_id,source_canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key),
    FOREIGN KEY(namespace_id,target_canonical_key)
        REFERENCES entity_shadow_canonicals(namespace_id,canonical_key)
);

CREATE TABLE IF NOT EXISTS shadow_model_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id) ON DELETE CASCADE,
    component TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_digest TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    attempt_no INTEGER NOT NULL CHECK(attempt_no>0),
    status TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_hash TEXT NOT NULL DEFAULT '',
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_response_text TEXT NOT NULL DEFAULT '',
    parsed_response JSONB NOT NULL DEFAULT '{}'::jsonb,
    token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms BIGINT NOT NULL DEFAULT 0 CHECK(latency_ms>=0),
    error_type TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    independent_call BOOLEAN NOT NULL DEFAULT TRUE,
    audit_quality TEXT NOT NULL DEFAULT 'complete',
    terminal_status TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(run_id,component,idempotency_key,attempt_no)
);
CREATE INDEX IF NOT EXISTS idx_shadow_model_calls_run_component
    ON shadow_model_calls(run_id,component,status,attempt_no);

CREATE TABLE IF NOT EXISTS shadow_model_call_members (
    id TEXT PRIMARY KEY,
    model_call_id TEXT NOT NULL REFERENCES shadow_model_calls(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    pair_opportunity_id TEXT REFERENCES shadow_relation_pair_opportunities(id),
    hypothesis_id TEXT REFERENCES shadow_relation_hypotheses(id),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(model_call_id,source_kind,source_record_id)
);

CREATE TABLE IF NOT EXISTS shadow_relation_pair_contexts (
    pair_opportunity_id TEXT NOT NULL REFERENCES shadow_relation_pair_opportunities(id)
        ON DELETE CASCADE,
    context_id TEXT NOT NULL REFERENCES shadow_source_contexts(id) ON DELETE CASCADE,
    context_role TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(pair_opportunity_id,context_id,context_role)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_pair_contexts_context
    ON shadow_relation_pair_contexts(context_id,context_role);

CREATE TABLE IF NOT EXISTS shadow_relation_candidate_sources (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id) ON DELETE CASCADE,
    pair_opportunity_id TEXT NOT NULL REFERENCES shadow_relation_pair_opportunities(id)
        ON DELETE CASCADE,
    candidate_source TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    candidate_rank INTEGER NOT NULL DEFAULT 0 CHECK(candidate_rank>=0),
    admitted BOOLEAN NOT NULL DEFAULT FALSE,
    terminal_status TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(pair_opportunity_id,candidate_source,source_kind,source_record_id)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_candidate_sources_pair
    ON shadow_relation_candidate_sources(pair_opportunity_id,candidate_source,admitted);

CREATE TABLE IF NOT EXISTS shadow_relation_proposal_attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id) ON DELETE CASCADE,
    context_id TEXT REFERENCES shadow_source_contexts(id),
    model_call_id TEXT REFERENCES shadow_model_calls(id),
    attempt_no INTEGER NOT NULL CHECK(attempt_no>0),
    status TEXT NOT NULL,
    proposer_prompt_version TEXT NOT NULL,
    proposer_schema_version TEXT NOT NULL,
    proposer_parser_version TEXT NOT NULL,
    proposer_model_revision TEXT NOT NULL DEFAULT '',
    request_hash TEXT NOT NULL,
    response_hash TEXT NOT NULL DEFAULT '',
    token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_type TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(run_id,context_id,attempt_no,proposer_prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_proposal_attempts_run
    ON shadow_relation_proposal_attempts(run_id,status,attempt_no);

CREATE TABLE IF NOT EXISTS shadow_relation_proposals (
    id TEXT PRIMARY KEY,
    proposal_attempt_id TEXT NOT NULL REFERENCES shadow_relation_proposal_attempts(id)
        ON DELETE CASCADE,
    pair_opportunity_id TEXT REFERENCES shadow_relation_pair_opportunities(id),
    hypothesis_id TEXT REFERENCES shadow_relation_hypotheses(id),
    proposal_key TEXT NOT NULL,
    predicate TEXT NOT NULL,
    subject_occurrence_key TEXT NOT NULL,
    object_occurrence_key TEXT NOT NULL,
    evidence_spans JSONB NOT NULL DEFAULT '[]'::jsonb,
    qualifiers JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    reason_code TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(proposal_attempt_id,proposal_key)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_proposals_attempt
    ON shadow_relation_proposals(proposal_attempt_id,status,predicate);

CREATE TABLE IF NOT EXISTS shadow_relation_evidence_refs (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id) ON DELETE CASCADE,
    fact_evidence_id TEXT NOT NULL REFERENCES shadow_relation_fact_evidence(id)
        ON DELETE CASCADE,
    verification_attempt_id TEXT REFERENCES shadow_relation_verification_attempts(id),
    context_id TEXT REFERENCES shadow_source_contexts(id),
    source_kind TEXT NOT NULL,
    source_record_id TEXT NOT NULL,
    evidence_start INTEGER NOT NULL DEFAULT 0 CHECK(evidence_start>=0),
    evidence_end INTEGER NOT NULL DEFAULT 0 CHECK(evidence_end>=evidence_start),
    evidence_hash TEXT NOT NULL,
    qualifiers JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(run_id,fact_evidence_id,context_id,evidence_hash)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_evidence_refs_run
    ON shadow_relation_evidence_refs(run_id,fact_evidence_id,context_id);
"""


GROUNDED_RELATION_SECONDARY_REVIEW_SQL = r"""
-- The secondary pass is append-only review lineage. It never overwrites the
-- primary hypothesis or verification attempt and does not own product graph
-- state. Successful candidates re-enter the existing Shadow normalization and
-- materialization gates in the same isolated run.
CREATE TABLE IF NOT EXISTS shadow_relation_secondary_passes (
    id TEXT PRIMARY KEY,
    parent_run_id TEXT NOT NULL REFERENCES shadow_relation_v2_runs(id) ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    pass_version TEXT NOT NULL,
    input_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('planning','running','ready','failed','cancelled')),
    route_counts JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(parent_run_id,pass_version,input_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_secondary_passes_parent
    ON shadow_relation_secondary_passes(parent_run_id,status,created_at DESC);

CREATE TABLE IF NOT EXISTS shadow_relation_secondary_candidates (
    id TEXT PRIMARY KEY,
    secondary_pass_id TEXT NOT NULL REFERENCES shadow_relation_secondary_passes(id) ON DELETE CASCADE,
    parent_hypothesis_id TEXT REFERENCES shadow_relation_hypotheses(id),
    pair_opportunity_id TEXT NOT NULL REFERENCES shadow_relation_pair_opportunities(id),
    route TEXT NOT NULL,
    source_reason_code TEXT NOT NULL,
    predicate_before TEXT NOT NULL DEFAULT '',
    predicate_after TEXT NOT NULL DEFAULT '',
    subject_occurrence_key TEXT NOT NULL DEFAULT '',
    object_occurrence_key TEXT NOT NULL DEFAULT '',
    evidence_spans JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL CHECK(status IN ('PLANNED','MODEL_NO_RELATION','MODEL_INSUFFICIENT','CANDIDATE','SUPPORTED','REJECTED','UNCERTAIN','FAILED','SKIPPED')),
    reason_code TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(secondary_pass_id,pair_opportunity_id,route,predicate_after,subject_occurrence_key,object_occurrence_key)
);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_secondary_candidates_pass
    ON shadow_relation_secondary_candidates(secondary_pass_id,status,route);
CREATE INDEX IF NOT EXISTS idx_shadow_relation_secondary_candidates_parent
    ON shadow_relation_secondary_candidates(parent_hypothesis_id,created_at DESC);
"""


GROUNDED_RELATION_SECONDARY_IDEMPOTENCY_SQL = r"""
-- A pair may have multiple primary hypotheses (for example, competing
-- predicates or directions).  Secondary review is hypothesis-scoped, so the
-- old pair/route-only key incorrectly collapsed distinct recovery candidates.
-- Keep NULL parent ids deterministic for legacy/audit rows as well.
ALTER TABLE IF EXISTS shadow_relation_secondary_candidates
    DROP CONSTRAINT IF EXISTS
    shadow_relation_secondary_candidates_secondary_pass_id_pair_opport_key;
CREATE UNIQUE INDEX IF NOT EXISTS
    uq_shadow_relation_secondary_candidates_idempotency
    ON shadow_relation_secondary_candidates(
        secondary_pass_id,
        pair_opportunity_id,
        route,
        COALESCE(parent_hypothesis_id, ''::text),
        predicate_after,
        subject_occurrence_key,
        object_occurrence_key
    );
"""


GROUNDED_RELATION_SECONDARY_IDEMPOTENCY_CLEANUP_SQL = r"""
-- PostgreSQL truncates long autogenerated constraint names.  Remove both the
-- intended and truncated spellings so upgrades from every 0027 installation
-- converge on the expression index introduced by 0028.
ALTER TABLE IF EXISTS shadow_relation_secondary_candidates
    DROP CONSTRAINT IF EXISTS
    shadow_relation_secondary_candidates_secondary_pass_id_pair_opport_key;
ALTER TABLE IF EXISTS shadow_relation_secondary_candidates
    DROP CONSTRAINT IF EXISTS
    shadow_relation_secondary_can_secondary_pass_id_pair_opport_key;
"""


# GraphRAG Native intentionally has its own control plane.  Final GraphRAG
# entities, semantic relationships, communities, reports, and LanceDB vectors
# remain immutable files in the per-run bundle; none are copied into the
# formal stable/enhanced entity or relation tables.
GRAPHRAG_NATIVE_CONTROL_PLANE_SQL = r"""
CREATE TABLE IF NOT EXISTS graphrag_native_runs (
    run_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    generation BIGINT NOT NULL CHECK(generation>0),
    state TEXT NOT NULL CHECK(state IN (
        'PENDING','PREPARING','RUNNING','EXECUTION_COMPLETE',
        'COMPLETED_WITH_FAILURES','VALIDATING','READY','EMPTY_INPUT',
        'NO_GRAPH_UNSUPPORTED','FAILED','CANCELLED','STALE','SUPERSEDED'
    )),
    base_sha TEXT NOT NULL,
    worktree_diff_hash TEXT NOT NULL,
    code_tree_hash TEXT NOT NULL,
    source_snapshot_hash TEXT NOT NULL,
    settings_hash TEXT NOT NULL,
    prompt_profile_hash TEXT NOT NULL,
    completion_model_id TEXT NOT NULL,
    completion_model_digest TEXT NOT NULL,
    embedding_model_id TEXT NOT NULL,
    embedding_model_digest TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL DEFAULT 0 CHECK(embedding_dimension>=0),
    graphrag_version TEXT NOT NULL,
    python_version TEXT NOT NULL,
    staging_relative_path TEXT NOT NULL DEFAULT '' CHECK(
        staging_relative_path='' OR (
            left(staging_relative_path,1)<>'/'
            AND staging_relative_path !~ '(^|/)\\.\\.(/|$)'
        )
    ),
    bundle_relative_path TEXT NOT NULL DEFAULT '' CHECK(
        bundle_relative_path='' OR (
            left(bundle_relative_path,1)<>'/'
            AND bundle_relative_path !~ '(^|/)\\.\\.(/|$)'
        )
    ),
    validator_version TEXT NOT NULL DEFAULT '',
    validation_report_hash TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    execution_completed_at TIMESTAMPTZ,
    validated_at TIMESTAMPTZ,
    ready_at TIMESTAMPTZ,
    failed_at TIMESTAMPTZ,
    failure_class TEXT NOT NULL DEFAULT '',
    failure_message TEXT NOT NULL DEFAULT '',
    lease_owner TEXT NOT NULL DEFAULT '',
    lease_expires_at TIMESTAMPTZ,
    superseded_by TEXT REFERENCES graphrag_native_runs(run_id),
    UNIQUE(project_id,generation)
);
CREATE INDEX IF NOT EXISTS idx_graphrag_native_runs_project_state
    ON graphrag_native_runs(project_id,state,generation DESC);
CREATE INDEX IF NOT EXISTS idx_graphrag_native_runs_lease
    ON graphrag_native_runs(state,lease_expires_at)
    WHERE state IN ('PENDING','PREPARING','RUNNING','VALIDATING');

CREATE TABLE IF NOT EXISTS graphrag_native_run_documents (
    run_id TEXT NOT NULL REFERENCES graphrag_native_runs(run_id) ON DELETE RESTRICT,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    document_id BIGINT NOT NULL,
    revision_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    structure_hash TEXT NOT NULL,
    input_document_count INTEGER NOT NULL CHECK(input_document_count>0),
    input_manifest_hash TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal>=0),
    PRIMARY KEY(run_id,document_id),
    UNIQUE(run_id,ordinal),
    FOREIGN KEY(revision_id,project_id,document_id)
        REFERENCES document_content_revisions(revision_id,project_id,document_id)
);
CREATE INDEX IF NOT EXISTS idx_graphrag_native_run_documents_project
    ON graphrag_native_run_documents(project_id,document_id,revision_id);

CREATE TABLE IF NOT EXISTS graphrag_native_artifacts (
    run_id TEXT NOT NULL REFERENCES graphrag_native_runs(run_id) ON DELETE RESTRICT,
    logical_name TEXT NOT NULL,
    relative_path TEXT NOT NULL CHECK(
        left(relative_path,1)<>'/'
        AND relative_path !~ '(^|/)\\.\\.(/|$)'
    ),
    sha256 TEXT NOT NULL CHECK(sha256 ~ '^[0-9a-f]{64}$'),
    byte_size BIGINT NOT NULL CHECK(byte_size>=0),
    media_type TEXT NOT NULL,
    schema_hash TEXT NOT NULL DEFAULT '',
    row_count BIGINT NOT NULL DEFAULT -1 CHECK(row_count>=-1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(run_id,logical_name),
    UNIQUE(run_id,relative_path)
);
CREATE INDEX IF NOT EXISTS idx_graphrag_native_artifacts_run
    ON graphrag_native_artifacts(run_id,logical_name);

CREATE TABLE IF NOT EXISTS graphrag_native_project_state (
    project_id BIGINT PRIMARY KEY REFERENCES projects(id),
    active_run_id TEXT REFERENCES graphrag_native_runs(run_id),
    active_generation BIGINT NOT NULL DEFAULT 0 CHECK(active_generation>=0),
    desired_source_snapshot_hash TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_graphrag_native_project_state_active
    ON graphrag_native_project_state(active_run_id) WHERE active_run_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS graphrag_native_query_runs (
    query_run_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id),
    run_id TEXT NOT NULL REFERENCES graphrag_native_runs(run_id) ON DELETE RESTRICT,
    query_mode TEXT NOT NULL CHECK(query_mode IN ('local','global','drift','basic')),
    question_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','READY','FAILED','CANCELLED')),
    model_id TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    latency_ms BIGINT NOT NULL DEFAULT 0 CHECK(latency_ms>=0),
    token_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    context_reference_count INTEGER NOT NULL DEFAULT 0 CHECK(context_reference_count>=0),
    failure_class TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_graphrag_native_queries_project_run
    ON graphrag_native_query_runs(project_id,run_id,created_at DESC);

INSERT INTO graphrag_native_project_state(project_id)
SELECT id FROM projects
ON CONFLICT(project_id) DO NOTHING;
"""


# This pointer is intentionally independent from both the native active-run
# pointer and existing stable/enhanced graph state.  No migration selects
# GraphRAG Native automatically; existing projects remain Stable by default.
PROJECT_CONSTRUCTION_SELECTION_SQL = r"""
CREATE TABLE IF NOT EXISTS project_construction_strategy_state (
    project_id BIGINT PRIMARY KEY REFERENCES projects(id),
    selected_strategy TEXT NOT NULL DEFAULT 'stable' CHECK(
        selected_strategy IN ('stable','enhanced_shadow','graphrag_native')
    ),
    generation BIGINT NOT NULL DEFAULT 0 CHECK(generation>=0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO project_construction_strategy_state(project_id,selected_strategy)
SELECT id,'stable' FROM projects
ON CONFLICT(project_id) DO NOTHING;
"""

PROJECT_STRATEGY_BINDING_SQL = r"""
ALTER TABLE project_construction_strategy_state
    ADD COLUMN IF NOT EXISTS association_status TEXT NOT NULL DEFAULT 'unassigned',
    ADD COLUMN IF NOT EXISTS bound_at TIMESTAMPTZ;

UPDATE project_construction_strategy_state
SET association_status=CASE
        WHEN selected_strategy IN ('enhanced_shadow','graphrag_native') THEN 'bound'
        ELSE 'unassigned'
    END,
    bound_at=CASE
        WHEN selected_strategy IN ('enhanced_shadow','graphrag_native')
            THEN COALESCE(bound_at,updated_at,NOW())
        ELSE NULL
    END;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname='project_construction_strategy_association_status_check'
          AND conrelid='project_construction_strategy_state'::regclass
    ) THEN
        ALTER TABLE project_construction_strategy_state
            ADD CONSTRAINT project_construction_strategy_association_status_check
            CHECK(association_status IN ('unassigned','bound'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname='project_construction_strategy_binding_check'
          AND conrelid='project_construction_strategy_state'::regclass
    ) THEN
        ALTER TABLE project_construction_strategy_state
            ADD CONSTRAINT project_construction_strategy_binding_check
            CHECK(
                (association_status='unassigned' AND selected_strategy='stable' AND bound_at IS NULL)
                OR
                (association_status='bound'
                 AND selected_strategy IN ('enhanced_shadow','graphrag_native')
                 AND bound_at IS NOT NULL)
            );
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_project_construction_strategy_binding
    ON project_construction_strategy_state(association_status,selected_strategy,project_id);

CREATE OR REPLACE FUNCTION kf_guard_project_strategy_binding()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.association_status='bound' AND (
        NEW.association_status IS DISTINCT FROM OLD.association_status
        OR NEW.selected_strategy IS DISTINCT FROM OLD.selected_strategy
        OR NEW.bound_at IS DISTINCT FROM OLD.bound_at
    ) THEN
        RAISE EXCEPTION 'project construction strategy is immutable after binding';
    END IF;
    RETURN NEW;
END
$$;
DROP TRIGGER IF EXISTS trg_guard_project_strategy_binding
    ON project_construction_strategy_state;
CREATE TRIGGER trg_guard_project_strategy_binding
BEFORE UPDATE ON project_construction_strategy_state
FOR EACH ROW EXECUTE FUNCTION kf_guard_project_strategy_binding();
"""

KNOWLEDGE_FORGE_COMMUNITIES_SQL = r"""
CREATE TABLE knowledge_forge_community_runs (
    run_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    strategy_key TEXT NOT NULL DEFAULT 'enhanced_shadow'
        CHECK(strategy_key='enhanced_shadow'),
    graph_generation BIGINT NOT NULL CHECK(graph_generation>0),
    entity_namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(
        status IN ('running','completed','completed_empty','failed','superseded')
    ),
    node_count INTEGER NOT NULL DEFAULT 0 CHECK(node_count>=0),
    edge_count INTEGER NOT NULL DEFAULT 0 CHECK(edge_count>=0),
    community_count INTEGER NOT NULL DEFAULT 0 CHECK(community_count>=0),
    hierarchy_levels INTEGER NOT NULL DEFAULT 0 CHECK(hierarchy_levels>=0),
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    communities JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(project_id,strategy_key,graph_generation,entity_namespace_id,input_hash)
);
CREATE INDEX idx_knowledge_forge_community_current
    ON knowledge_forge_community_runs(
        project_id,strategy_key,graph_generation,entity_namespace_id,created_at DESC
    ) WHERE status IN ('completed','completed_empty');
"""


SCHEMA_EVALUATION_AUDIT_SQL = r"""
CREATE TABLE IF NOT EXISTS schema_evaluation_runs (
    id BIGSERIAL PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    prior_schema_version INTEGER NOT NULL CHECK(prior_schema_version>0),
    resulting_schema_version INTEGER NOT NULL CHECK(resulting_schema_version>0),
    outcome TEXT NOT NULL CHECK(
        outcome IN ('skipped','no_change','applied','failed')
    ),
    evaluation JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_schema_evaluation_project_document
    ON schema_evaluation_runs(project_id,document_id,created_at DESC);
"""


# ``enhanced_shadow`` was a pre-release product identifier.  The physical
# entity/relation ledger deliberately retains its historical table names: they
# are part of immutable provenance and cannot be rewritten safely.  This
# migration changes the user-facing strategy identity and its control-plane
# records only, while normalize_strategy still accepts the legacy input key.
KNOWLEDGE_FORGE_STRATEGY_NAMING_SQL = r"""
ALTER TABLE document_strategy_builds
    DROP CONSTRAINT IF EXISTS document_strategy_builds_strategy_key_check;
ALTER TABLE project_strategy_graphs
    DROP CONSTRAINT IF EXISTS project_strategy_graphs_strategy_key_check;
ALTER TABLE knowledge_forge_community_runs
    DROP CONSTRAINT IF EXISTS knowledge_forge_community_runs_strategy_key_check;
ALTER TABLE project_construction_strategy_state
    DROP CONSTRAINT IF EXISTS project_construction_strategy_binding_check;
ALTER TABLE project_construction_strategy_state
    DROP CONSTRAINT IF EXISTS project_construction_strategy_state_selected_strategy_check;
ALTER TABLE project_construction_strategy_state
    DISABLE TRIGGER trg_guard_project_strategy_binding;

UPDATE document_strategy_builds
SET strategy_key='knowledge_forge'
WHERE strategy_key='enhanced_shadow';
UPDATE project_strategy_graphs
SET strategy_key='knowledge_forge'
WHERE strategy_key='enhanced_shadow';
UPDATE knowledge_forge_community_runs
SET strategy_key='knowledge_forge'
WHERE strategy_key='enhanced_shadow';
UPDATE project_construction_strategy_state
SET selected_strategy='knowledge_forge'
WHERE selected_strategy='enhanced_shadow';
UPDATE jobs
SET strategy_key='knowledge_forge'
WHERE strategy_key='enhanced_shadow';
UPDATE project_strategy_graphs
SET policy_summary=jsonb_set(
        COALESCE(policy_summary,'{}'::jsonb),
        '{mode}',to_jsonb('grounded'::text),TRUE
    )
WHERE strategy_key='knowledge_forge'
  AND COALESCE(policy_summary->>'mode','')='shadow';

ALTER TABLE knowledge_forge_community_runs
    ALTER COLUMN strategy_key SET DEFAULT 'knowledge_forge';
ALTER TABLE document_strategy_builds
    ADD CONSTRAINT document_strategy_builds_strategy_key_check
    CHECK(strategy_key IN ('stable','knowledge_forge'));
ALTER TABLE project_strategy_graphs
    ADD CONSTRAINT project_strategy_graphs_strategy_key_check
    CHECK(strategy_key IN ('stable','knowledge_forge'));
ALTER TABLE knowledge_forge_community_runs
    ADD CONSTRAINT knowledge_forge_community_runs_strategy_key_check
    CHECK(strategy_key='knowledge_forge');
ALTER TABLE project_construction_strategy_state
    ADD CONSTRAINT project_construction_strategy_state_selected_strategy_check
    CHECK(selected_strategy IN ('stable','knowledge_forge','graphrag_native'));
ALTER TABLE project_construction_strategy_state
    ADD CONSTRAINT project_construction_strategy_binding_check
    CHECK(
        (association_status='unassigned' AND selected_strategy='stable' AND bound_at IS NULL)
        OR
        (association_status='bound'
         AND selected_strategy IN ('knowledge_forge','graphrag_native')
         AND bound_at IS NOT NULL)
    );
ALTER TABLE project_construction_strategy_state
    ENABLE TRIGGER trg_guard_project_strategy_binding;
"""


KNOWLEDGE_FORGE_COMMUNITY_REPORTS_SQL = r"""
CREATE TABLE IF NOT EXISTS knowledge_forge_community_report_runs (
    run_id TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    community_run_id TEXT NOT NULL
        REFERENCES knowledge_forge_community_runs(run_id) ON DELETE CASCADE,
    graph_generation BIGINT NOT NULL CHECK(graph_generation>0),
    entity_namespace_id TEXT NOT NULL REFERENCES entity_shadow_namespaces(id),
    community_input_hash TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    embedding_model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    retry_round INTEGER NOT NULL DEFAULT 0 CHECK(retry_round>=0),
    status TEXT NOT NULL CHECK(status IN (
        'queued','running','completed','completed_empty','partial','failed',
        'superseded','cancelled'
    )),
    expected_count INTEGER NOT NULL DEFAULT 0 CHECK(expected_count>=0),
    eligible_count INTEGER NOT NULL DEFAULT 0 CHECK(eligible_count>=0),
    completed_count INTEGER NOT NULL DEFAULT 0 CHECK(completed_count>=0),
    reused_count INTEGER NOT NULL DEFAULT 0 CHECK(reused_count>=0),
    failed_count INTEGER NOT NULL DEFAULT 0 CHECK(failed_count>=0),
    skipped_count INTEGER NOT NULL DEFAULT 0 CHECK(skipped_count>=0),
    job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE(
        project_id,community_run_id,model_id,prompt_hash,config_hash,retry_round
    )
);
CREATE INDEX IF NOT EXISTS idx_kf_community_report_runs_current
    ON knowledge_forge_community_report_runs(
        project_id,community_run_id,created_at DESC
    );
CREATE INDEX IF NOT EXISTS idx_kf_community_report_runs_active
    ON knowledge_forge_community_report_runs(project_id,status,created_at)
    WHERE status IN ('queued','running');

CREATE TABLE IF NOT EXISTS knowledge_forge_community_reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES knowledge_forge_community_report_runs(run_id)
        ON DELETE CASCADE,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    community_run_id TEXT NOT NULL,
    community_key TEXT NOT NULL,
    community_content_hash TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    level INTEGER NOT NULL DEFAULT 0 CHECK(level>=0),
    status TEXT NOT NULL CHECK(status IN (
        'pending','running','succeeded','reused','failed','skipped',
        'cancelled','superseded'
    )),
    priority INTEGER NOT NULL DEFAULT 0,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count>=0),
    max_attempts INTEGER NOT NULL DEFAULT 2 CHECK(max_attempts>0),
    entity_keys TEXT[] NOT NULL DEFAULT '{}',
    relation_keys TEXT[] NOT NULL DEFAULT '{}',
    cited_entity_keys TEXT[] NOT NULL DEFAULT '{}',
    cited_relation_keys TEXT[] NOT NULL DEFAULT '{}',
    input_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    title TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    findings JSONB NOT NULL DEFAULT '[]'::jsonb,
    limitations TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    output_hash TEXT NOT NULL DEFAULT '',
    source_report_id TEXT,
    usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms BIGINT NOT NULL DEFAULT 0 CHECK(latency_ms>=0),
    embedding_status TEXT NOT NULL DEFAULT 'pending' CHECK(
        embedding_status IN ('pending','succeeded','failed','not_requested')
    ),
    embedding_error TEXT NOT NULL DEFAULT '',
    error_category TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(run_id,community_key)
);
CREATE INDEX IF NOT EXISTS idx_kf_community_reports_queue
    ON knowledge_forge_community_reports(run_id,status,priority DESC,level,community_key);
CREATE INDEX IF NOT EXISTS idx_kf_community_reports_reuse
    ON knowledge_forge_community_reports(
        project_id,community_content_hash,status,completed_at DESC
    ) WHERE status IN ('succeeded','reused');
CREATE INDEX IF NOT EXISTS idx_kf_community_reports_entities
    ON knowledge_forge_community_reports USING gin(entity_keys);
CREATE INDEX IF NOT EXISTS idx_kf_community_reports_content_trgm
    ON knowledge_forge_community_reports USING gin(content gin_trgm_ops)
    WHERE status IN ('succeeded','reused');

CREATE TABLE IF NOT EXISTS knowledge_forge_community_report_attempts (
    attempt_id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL REFERENCES knowledge_forge_community_reports(report_id)
        ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES knowledge_forge_community_report_runs(run_id)
        ON DELETE CASCADE,
    attempt_no INTEGER NOT NULL CHECK(attempt_no>0),
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running','succeeded','failed')),
    raw_response_text TEXT NOT NULL DEFAULT '',
    parsed_response JSONB NOT NULL DEFAULT '{}'::jsonb,
    diagnostics JSONB NOT NULL DEFAULT '{}'::jsonb,
    latency_ms BIGINT NOT NULL DEFAULT 0 CHECK(latency_ms>=0),
    error_category TEXT NOT NULL DEFAULT '',
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    UNIQUE(report_id,attempt_no)
);

ALTER TABLE vector_embeddings
    DROP CONSTRAINT IF EXISTS vector_embeddings_object_type_check;
ALTER TABLE vector_embeddings
    ADD CONSTRAINT vector_embeddings_object_type_check CHECK(
        object_type IN ('chunk','entity','relation','schema','community_report')
    );
"""


KNOWLEDGE_FORGE_ENTITY_EXTRACTION_CACHE_SQL = r"""
CREATE TABLE IF NOT EXISTS knowledge_forge_entity_extraction_cache (
    cache_key TEXT PRIMARY KEY,
    project_id BIGINT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL,
    revision_id TEXT NOT NULL,
    chunk_id BIGINT NOT NULL,
    content_hash TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_digest TEXT NOT NULL CHECK(model_digest<>''),
    extraction_policy_version TEXT NOT NULL,
    code_version TEXT NOT NULL,
    request_fingerprint TEXT NOT NULL,
    request_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_json JSONB NOT NULL,
    output_hash TEXT NOT NULL,
    hit_count BIGINT NOT NULL DEFAULT 0 CHECK(hit_count>=0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT fk_kf_entity_extraction_cache_document
        FOREIGN KEY(document_id,project_id) REFERENCES documents(id,project_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_kf_entity_extraction_cache_chunk_revision
        FOREIGN KEY(chunk_id,revision_id) REFERENCES chunks(id,revision_id)
        ON DELETE CASCADE,
    UNIQUE(project_id,request_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_kf_entity_extraction_cache_source
    ON knowledge_forge_entity_extraction_cache(
        project_id,document_id,revision_id,chunk_id
    );
"""


PROJECT_ACTIVITY_TIMESTAMP_REPAIR_SQL = r"""
-- A graph-policy compatibility pass historically marked every affected graph
-- stale and also replaced projects.updated_at in one statement. That field
-- drives the recent-project chooser, so a maintenance batch made unrelated
-- old projects look like simultaneous user activity. Recover only an
-- identifiable multi-project compatibility batch and only when durable
-- project records prove an earlier last activity time.
WITH recoverable_activity AS (
    SELECT p.id,
           GREATEST(
               p.created_at,
               COALESCE((
                   SELECT MAX(j.updated_at) FROM jobs j
                   WHERE j.project_id=p.id
               ),p.created_at),
               COALESCE((
                   SELECT MAX(d.updated_at) FROM documents d
                   WHERE d.project_id=p.id
               ),p.created_at),
               COALESCE((
                   SELECT MAX(s.updated_at)
                   FROM project_construction_strategy_state s
                   WHERE s.project_id=p.id
               ),p.created_at)
           ) AS recovered_at
    FROM projects p
    WHERE p.is_deleted=FALSE
      AND p.graph_status='stale'
      AND p.graph_error='图谱需要按当前策略集合和代次重建'
), maintenance_batches AS (
    SELECT p.updated_at
    FROM projects p
    JOIN recoverable_activity activity ON activity.id=p.id
    WHERE p.updated_at>activity.recovered_at
    GROUP BY p.updated_at
    HAVING COUNT(*)>1
)
UPDATE projects p
SET updated_at=activity.recovered_at
FROM recoverable_activity activity,maintenance_batches batch
WHERE p.id=activity.id
  AND p.updated_at=batch.updated_at
  AND p.updated_at>activity.recovered_at;
"""


PROJECT_KNOWLEDGE_LANGUAGE_SQL = r"""
-- Existing projects predate this preference and therefore retain the historic
-- source-language behaviour.  Only projects inserted after this migration use
-- Chinese as the database default.
ALTER TABLE projects
    ADD COLUMN IF NOT EXISTS knowledge_language TEXT NOT NULL DEFAULT 'auto';

ALTER TABLE projects
    DROP CONSTRAINT IF EXISTS projects_knowledge_language_check;
ALTER TABLE projects
    ADD CONSTRAINT projects_knowledge_language_check
    CHECK (knowledge_language IN ('auto','zh','en'));

ALTER TABLE projects
    ALTER COLUMN knowledge_language SET DEFAULT 'zh';
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
        sql=RELATION_WORK_QUEUE_SQL,
    ),
    Migration(
        version="0003_entity_shadow_pipeline",
        description="Persist independent span-first entity shadow treatment and audit lineage",
        sql=GROUNDED_ENTITY_PIPELINE_SQL,
    ),
    Migration(
        version="0004_policy_snapshot_orchestrator",
        description="Persist immutable resumable policy snapshot work and artifacts",
        sql=POLICY_SNAPSHOT_ORCHESTRATOR_SQL,
    ),
    Migration(
        version="0005_release_gate_artifacts",
        description="Bind append-only Gold and evaluation facts to frozen snapshot artifacts",
        sql=RELEASE_GATE_ARTIFACTS_SQL,
    ),
    Migration(
        version="0006_graph_namespace_manifest",
        description="Isolate graph ownership and persist leased generation manifests",
        sql=GRAPH_NAMESPACE_MANIFEST_SQL,
    ),
    Migration(
        version="0007_indexing_failure_status",
        description="Safely extend the indexing failure terminal-state constraint",
        sql=INDEXING_FAILURE_STATUS_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0008_mention_occurrence_identity",
        description="Preserve repeated mention occurrences with stable revision-scoped identity",
        sql=MENTION_OCCURRENCE_IDENTITY_SQL,
    ),
    Migration(
        version="0009_policy_snapshot_coverage",
        description="Fail policy promotion closed on incomplete relation coverage",
        sql=POLICY_SNAPSHOT_COVERAGE_SQL,
    ),
    Migration(
        version="0010_graphrag_communities",
        description="Persist shadow GraphRAG BYOG community detection results",
        sql=GRAPHRAG_COMMUNITIES_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0011_graphrag_reports_global_search",
        description="Persist snapshot-bound community reports and opt-in global search shadow",
        sql=GRAPHRAG_REPORTS_SEARCH_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0012_multi_strategy_construction",
        description="Isolate per-strategy document builds, jobs, and graph namespaces",
        sql=MULTI_STRATEGY_CONSTRUCTION_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0013_strategy_graph_lease_namespace",
        description="Permit leased graph generations for project strategy namespaces",
        sql=STRATEGY_GRAPH_LEASE_NAMESPACE_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0014_shadow_relation_v2_ledger",
        description="Persist unordered mention-pair opportunities and multi-hypothesis lineage",
        sql=GROUNDED_RELATION_V2_LEDGER_SQL,
    ),
    Migration(
        version="0015_shadow_relation_v2_verification",
        description="Persist independent append-only relation verification attempts",
        sql=GROUNDED_RELATION_V2_VERIFICATION_SQL,
    ),
    Migration(
        version="0016_shadow_relation_v2_normalization",
        description="Normalize verified facts, retain evidence, and audit legacy duplicates",
        sql=GROUNDED_RELATION_V2_NORMALIZATION_SQL,
    ),
    Migration(
        version="0017_shadow_relation_v2_freshness",
        description="Bind relation materialization freshness and block Shadow product writes",
        sql=GROUNDED_RELATION_V2_FRESHNESS_SQL,
    ),
    Migration(
        version="0018_shadow_relation_v2_gold",
        description="Persist pair-level Gold, fact positives, and prediction adjudication",
        sql=GROUNDED_RELATION_V2_GOLD_SQL,
    ),
    Migration(
        version="0019_shadow_relation_v2_execution",
        description="Persist classifier attempts, rule validation, and retry-safe run evidence",
        sql=GROUNDED_RELATION_V2_EXECUTION_SQL,
    ),
    Migration(
        version="0020_shadow_relation_v2_model_audit",
        description="Audit Shadow v2 classifier request and response payloads with fail-closed statuses",
        sql=GROUNDED_RELATION_V2_MODEL_AUDIT_SQL,
    ),
    Migration(
        version="0021_shadow_relation_v2_artifact_audit",
        description="Persist enhanced Shadow v2 source contexts, model calls, proposals, and export audit refs",
        sql=GROUNDED_RELATION_V2_ARTIFACT_AUDIT_SQL,
    ),
    Migration(
        version="0022_graphrag_native_control_plane",
        description="Create isolated GraphRAG Native run control plane and audit metadata",
        sql=GRAPHRAG_NATIVE_CONTROL_PLANE_SQL,
    ),
    Migration(
        version="0023_project_construction_strategy_selection",
        description="Persist a separately authorized project construction-strategy pointer",
        sql=PROJECT_CONSTRUCTION_SELECTION_SQL,
    ),
    Migration(
        version="0024_project_strategy_binding",
        description="Bind each project once to exactly one user-visible construction strategy",
        sql=PROJECT_STRATEGY_BINDING_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0025_knowledge_forge_communities",
        description="Persist strategy-owned Knowledge Forge community clustering snapshots",
        sql=KNOWLEDGE_FORGE_COMMUNITIES_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0026_schema_evaluation_audit",
        description="Persist every adaptive schema review outcome and rejection audit",
        sql=SCHEMA_EVALUATION_AUDIT_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0027_shadow_relation_secondary_review",
        description="Persist reason-routed secondary relation recovery lineage",
        sql=GROUNDED_RELATION_SECONDARY_REVIEW_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0028_shadow_relation_secondary_idempotency",
        description="Scope secondary candidate idempotency by parent hypothesis",
        sql=GROUNDED_RELATION_SECONDARY_IDEMPOTENCY_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0029_shadow_relation_secondary_idempotency_cleanup",
        description="Remove the truncated legacy secondary candidate constraint",
        sql=GROUNDED_RELATION_SECONDARY_IDEMPOTENCY_CLEANUP_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0030_knowledge_forge_strategy_naming",
        description="Rename the enhanced strategy control plane to Knowledge Forge",
        sql=KNOWLEDGE_FORGE_STRATEGY_NAMING_SQL,
    ),
    Migration(
        version="0031_knowledge_forge_community_reports",
        description="Persist version-bound asynchronous Knowledge Forge community reports",
        sql=KNOWLEDGE_FORGE_COMMUNITY_REPORTS_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0032_knowledge_forge_entity_extraction_cache",
        description="Reuse byte-identical grounded entity extraction results across graph builds",
        sql=KNOWLEDGE_FORGE_ENTITY_EXTRACTION_CACHE_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0033_project_activity_timestamp_repair",
        description="Keep recent-project activity ordering independent from policy maintenance",
        sql=PROJECT_ACTIVITY_TIMESTAMP_REPAIR_SQL,
        startup_safe=True,
    ),
    Migration(
        version="0034_project_knowledge_language",
        description="Persist a forward-only project language policy for generated knowledge",
        sql=PROJECT_KNOWLEDGE_LANGUAGE_SQL,
        startup_safe=True,
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
               execution_mode TEXT NOT NULL DEFAULT 'transactional',
               progress JSONB NOT NULL DEFAULT '{}'::jsonb,
               applied_at TIMESTAMPTZ,updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"""
        )
        conn.execute(
            "ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS "
            "execution_mode TEXT NOT NULL DEFAULT 'transactional'"
        )
        conn.execute(
            "ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS "
            "progress JSONB NOT NULL DEFAULT '{}'::jsonb"
        )

    @staticmethod
    def _table_exists(conn: psycopg.Connection) -> bool:
        row = conn.execute(
            "SELECT to_regclass('schema_migrations') AS table_name"
        ).fetchone()
        return bool(row and row.get("table_name"))

    def plan(self, conn: psycopg.Connection) -> list[dict[str, str]]:
        """Return the immutable migration plan without issuing any write."""

        rows = []
        if self._table_exists(conn):
            rows = conn.execute(
                "SELECT version,checksum,status FROM schema_migrations ORDER BY version"
            ).fetchall()
        recorded = {str(row["version"]): row for row in rows}
        plan: list[dict[str, str]] = []
        for migration in self.migrations:
            row = recorded.get(migration.version)
            if row and row["checksum"] != migration.checksum:
                raise RuntimeError(
                    f"Migration checksum mismatch for {migration.version}: "
                    f"database={row['checksum']} code={migration.checksum}"
                )
            plan.append({
                "version": migration.version,
                "description": migration.description,
                "checksum": migration.checksum,
                "status": str((row or {}).get("status") or "pending"),
                "startup_safe": "true" if migration.startup_safe else "false",
                "execution_mode": "transactional" if migration.transactional else "nontransactional",
            })
        return plan

    def status(self, conn: psycopg.Connection) -> list[dict[str, str]]:
        return self.plan(conn)

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

    def apply(
        self,
        conn: psycopg.Connection,
        *,
        startup_safe_only: bool = False,
    ) -> list[str]:
        """Apply migrations with a session lock and durable per-step status.

        Every transactional migration is committed independently.  A failed
        migration is rolled back, recorded as failed in a fresh transaction,
        and may be resumed by a later invocation.
        """

        conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_ID,))
        conn.commit()
        applied: list[str] = []
        try:
            self.ensure_table(conn)
            conn.commit()
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
                        conn.commit()
                        continue
                if startup_safe_only and not migration.startup_safe:
                    conn.rollback()
                    continue
                started = time.monotonic()
                mode = "transactional" if migration.transactional else "nontransactional"
                conn.execute(
                    """INSERT INTO schema_migrations(
                           version,description,checksum,status,execution_mode,progress)
                       VALUES(%s,%s,%s,'applying',%s,'{}'::jsonb)
                       ON CONFLICT(version) DO UPDATE SET
                           description=EXCLUDED.description,status='applying',
                           error_message='',execution_mode=EXCLUDED.execution_mode,
                           updated_at=NOW()""",
                    (migration.version, migration.description, migration.checksum, mode),
                )
                conn.commit()
                try:
                    if migration.transactional:
                        conn.execute(migration.sql)
                    else:
                        previous_autocommit = conn.autocommit
                        conn.autocommit = True
                        try:
                            conn.execute(migration.sql)
                        finally:
                            conn.autocommit = previous_autocommit
                    duration_ms = int((time.monotonic() - started) * 1000)
                    conn.execute(
                        """UPDATE schema_migrations SET status='applied',duration_ms=%s,
                           error_message='',progress='{"complete":true}'::jsonb,
                           applied_at=NOW(),updated_at=NOW() WHERE version=%s""",
                        (duration_ms, migration.version),
                    )
                    conn.commit()
                    applied.append(migration.version)
                except Exception as exc:
                    conn.rollback()
                    duration_ms = int((time.monotonic() - started) * 1000)
                    conn.execute(
                        """UPDATE schema_migrations SET status='failed',duration_ms=%s,
                           error_message=%s,updated_at=NOW() WHERE version=%s""",
                        (duration_ms, str(exc)[:2000], migration.version),
                    )
                    conn.commit()
                    raise RuntimeError(f"Migration {migration.version} failed: {exc}") from exc
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_ID,))
            conn.commit()
        return applied

    @staticmethod
    def run_batched_backfill(
        conn: psycopg.Connection,
        *,
        version: str,
        fetch_batch: Callable[[psycopg.Connection, int], list[object]],
        apply_batch: Callable[[psycopg.Connection, list[object]], None],
        batch_size: int = 500,
    ) -> int:
        """Run a bounded, resumable backfill and persist monotonic progress."""

        row = conn.execute(
            "SELECT progress FROM schema_migrations WHERE version=%s FOR UPDATE",
            (version,),
        ).fetchone()
        if not row:
            raise RuntimeError(f"Backfill migration state is missing: {version}")
        progress = row.get("progress") or {}
        total = max(0, int(progress.get("processed") or 0))
        conn.commit()
        while True:
            rows = list(fetch_batch(conn, max(1, int(batch_size))))
            if not rows:
                break
            apply_batch(conn, rows)
            total += len(rows)
            conn.execute(
                """UPDATE schema_migrations
                   SET progress=jsonb_build_object('processed',%s),updated_at=NOW()
                   WHERE version=%s""",
                (total, version),
            )
            conn.commit()
        conn.execute(
            """UPDATE schema_migrations
               SET progress=jsonb_build_object('processed',%s,'complete',TRUE),updated_at=NOW()
               WHERE version=%s""",
            (total, version),
        )
        conn.commit()
        return total
