# Knowledge Forge contributor guide

## Product boundary

Knowledge Forge is a generic, multi-project knowledge base and knowledge graph
builder. It is independent from `icp-fault-detection-agent`. Do not edit, import
runtime code from, or couple deployments to that sibling project.

RAGFlow source under `external/` is reference material only. The application
must not require a running RAGFlow service or call its API.

## Runtime ownership

- PostgreSQL: projects, documents, sections, chunks, embeddings, logical schema
  versions, canonical entities, mentions, relation assertions and jobs.
- HugeGraph physical graph `knowledge_forge`: navigable project-isolated graph.
- Ollama: model discovery, embeddings, schema review and structured extraction.
- `KnowledgeFiles/`: source documents, derived chunk JSON and soft-delete trash.

## Non-negotiable invariants

1. Every derived record carries `project_id`; graph records also carry
   `project_key`.
2. Every entity mention and relation assertion retains document and chunk
   provenance.
3. Physical HugeGraph labels are the stable `KF*` schema. Domain-specific
   changes belong in versioned logical schemas, not new physical labels.
4. LLM schema evolution is additive-only. Never automatically remove, rename,
   or redefine an active type.
5. Entity resolution is conservative. Ambiguous candidates remain separate.
6. Re-ingestion replaces only the selected document's derived contribution.
7. Document deletion is soft in PostgreSQL and must not remove facts still
   supported by another document.
8. Long-running work uses the PostgreSQL `jobs` table; do not rely on process
   memory for user-visible status.
9. Fixes to extraction, normalization, graph construction and UI state must be
   generic system fixes. Production control flow must not branch on the current
   task, a specific uploaded document, a concrete domain term, or a hard-coded
   predicate name. Such examples may appear only in tests, Gold fixtures or
   documentation.
10. Before reporting implementation work as complete, run and report relevant
    self-checks. Prefer targeted regression tests plus the standard compile/test
    commands below; do not rely on the user to discover first-order failures
    through the UI.
11. A change that affects a user-visible HTTP route, browser interaction,
    background job, or model-backed workflow is not self-checked by unit tests
    alone. Before reporting it complete, exercise the real public HTTP/job
    entrypoint against isolated data, wait for its terminal state, and inspect
    the persisted result or failure record. For Ollama-backed workflows this
    must use the configured runtime model service, not only scripted fixtures.
    A fail-closed error, stale lease, malformed model response, migration gap,
    or missing UI state means the change is incomplete and must be fixed before
    asking the user to test it.
12. `graphrag_native` is an independent, project-wide GraphRAG 3.1.1 strategy.
    Its algorithm facts stay in immutable per-run Parquet/LanceDB bundles and
    its PostgreSQL tables are control-plane metadata only.  Never route it
    through Stable/Enhanced extraction, canonical entities, pgvector or
    HugeGraph, and never silently fall back to those paths.
13. A native run may become queryable only after independent bundle validation
    and current-source freshness checks.  Native active-run and project
    selected-strategy pointers are separate CAS-controlled concepts; do not
    auto-select, activate, delete, commit, push or deploy native work without
    explicit user authority.
14. Before restarting `knowledge-forge.service`, query every active
    `graphrag_native_build`.  Do not restart while one is running: systemd
    terminates its isolated GraphRAG child and recovery restarts the entire
    run.  Finish it, cancel it with user authority, or use a non-disruptive
    observation path instead.

## Module boundaries

- `chunking.py`: deterministic extraction and chunking only.
- `ollama.py`: Ollama transport and JSON parsing only.
- `vector_store.py`: embedding persistence and retrieval scoring.
- `schema_evolution.py`: logical schema policy and structured extraction.
- `entity_resolution.py`: mention-to-canonical resolution.
- `hugegraph_store.py`: physical schema and graph synchronization.
- `database.py`: PostgreSQL schema/migrations and persistence operations.
- `service.py`: orchestration and background jobs.
- `web.py`: HTTP routing and presentation; keep domain logic out.

## Change workflow

Before editing, inspect existing migrations and avoid dropping user data. Prefer
`CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.
Update `.env.example`, Docker files and README when runtime configuration changes.

Run before finishing:

```bash
python -m py_compile knowledge_forge/*.py server.py
python -m pytest -q
```

For integration changes, also verify:

```bash
curl http://127.0.0.1:8011/health
curl http://127.0.0.1:8080/graphs
```

Do not commit secrets, uploaded documents, generated vectors, model files or
runtime logs.
