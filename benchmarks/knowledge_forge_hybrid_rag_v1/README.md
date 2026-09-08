# Knowledge Forge hybrid-RAG calibration v1

This is a small, manually authored engineering acceptance set for the next
stage after graph construction. It contains 24 questions over all ten frozen
English and Chinese calibration documents. The questions cover direct facts,
multi-hop paths, set aggregation, and cases where explicit negation must come
from source text rather than from a published positive graph edge.

The set is deliberately not a production-grade independent Gold benchmark.
Its source facts share the single-expert limitations of the two extraction
corpora, and it must not be used to claim general benchmark leadership. Its
purpose is to answer the practical product question: does graph retrieval add
useful grounded context to direct knowledge-base retrieval without worsening
answers or citations?

Hard-gated questions may reference only `core_direct` assertions. Plausible
interpretation-band abstractions remain useful qualitative probes, but are not
allowed to decide whether hybrid retrieval passes. The validator enforces this
contract.

Run the contract check with:

```bash
python benchmarks/knowledge_forge_hybrid_rag_v1/validate.py
```

The eventual runtime gate compares vector-only and hybrid retrieval on the
same frozen questions. Vector-required questions must retain source passages;
negative claims are audit labels and must never be materialized as positive
graph relations.

After the ten documents have been ingested into one enhanced Knowledge Forge
project, score the real public search route with:

```bash
.venv/bin/python benchmarks/knowledge_forge_hybrid_rag_v1/evaluate_retrieval.py \
  --project-id PROJECT_ID
```

The runner measures citation/evidence retrieval at top 10 for vector, graph and
their union. Ten follows the bounded local-context budget and is large enough
for the widest six-fact aggregation question in this set. A graph hit requires
directed endpoints plus matching source evidence; exact predicate-name
agreement is reported separately because each project has an additive,
induced logical Schema.

The graph is one source in a hybrid retriever, not a required replacement for
source chunks. Consequently, the practical graph-value gate uses fact-level
coverage on graph-preferred questions plus context compression; an all-or-none
question score remains diagnostic. This avoids an impossible "beat vector"
gate when vector retrieval already saturates the small corpus. A published
`related_to` or a declared hard-negative edge still fails closed. The runner is
not an answer-generation judge: unsupported generation and final citation
rendering remain a later RAG-layer evaluation.
