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
