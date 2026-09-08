# Knowledge Agent V1 engineering acceptance set

This package freezes the first Knowledge Forge answering baseline. It reuses
the 24 manually authored bilingual questions in
`knowledge_forge_hybrid_rag_v1` and adds explicit no-answer and source-safety
probes. It is an engineering calibration set, not an independent public
leaderboard.

The retrieval-only comparison against plain Chunk RAG remains owned by the
existing hybrid-RAG evaluator. This evaluator exercises the real public agent
HTTP API and scores the additional V1 contracts:

- every retained claim maps to persisted evidence;
- every cited evidence row belongs to the selected project and current run;
- browser source references remain resolvable;
- answer atoms are covered for answerable calibration questions;
- deliberately unanswerable questions are refused;
- failed or cancelled runs never count as answers.

Validate the frozen files:

```bash
.venv/bin/python benchmarks/knowledge_agent_v1/validate.py
```

Run the real HTTP baseline against a prepared Knowledge Forge calibration
project (this invokes the configured reranker and Ollama model):

```bash
.venv/bin/python benchmarks/knowledge_agent_v1/evaluate.py \
  --project-id PROJECT_ID --assert-acceptance
```

Use `--limit 2` for a cheap transport smoke test. A full run makes one bounded
answer call per question and is therefore intentionally not part of the unit
test suite.
