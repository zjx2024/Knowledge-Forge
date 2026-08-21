# Knowledge Forge engineering acceptance profile (2026-08-11)

## Decision

Knowledge Forge does not need perfect open-domain entity and relation extraction
before the graph can serve as one retrieval source in a hybrid RAG system. The
release target for the current work is **engineering-usable**, not
independently human-certified production Gold.

The extraction benchmark is deliberately split into:

1. a small, evidence-complete bilingual calibration corpus;
2. a separate long-document cost and termination corpus;
3. downstream retrieval questions comparing vector-only, graph-only and hybrid
   retrieval.

This prevents a short extraction score from standing in for long-document
operability or actual RAG value.

## Corpus audit

`benchmarks/audit_calibration_corpora.py` currently reports:

| Measure | English v1 | Chinese supplement | Combined |
| --- | ---: | ---: | ---: |
| Documents | 6 | 4 | 10 |
| Entity mentions | 144 | 113 | 257 |
| Canonical entities | 90 | 71 | 161 |
| Positive assertion occurrences | 80 | 64 | 144 |
| Core-direct positive assertions | 71 | 62 | 133 |
| Predicate-specific hard negatives | 16 | 16 | 32 |

The combined set covers nine domains and 25 declared structure tags in English
and Chinese. It is adequate for engineering calibration because no supervised
model is trained on it and because the holdout documents remain frozen.

Important limitations remain:

- every current calibration document is only 421–708 UTF-8 bytes;
- 46 of 73 relation types occur once, so relation macro-F1 is unstable;
- labels are a single assistant expert pass, not independent human Gold;
- the corpus is below an optional local target of 200 independently
  adjudicated production-Gold facts; that number is not an external benchmark
  standard and is not a blocker for the current hybrid-RAG engineering release;
- the deterministic 1x/10x stress set and the seven real PDF manuals measure
  cost/termination only; neither is relation Gold.

The English second-pass audit places nine plausible but interpretation-sensitive
facts in a separate band (temporal state, metonymy, mediator decomposition or
ontology abstraction). The Chinese set contains 62 core-direct facts and two
explicit interpretation-band facts. Hard negatives are never softened by this
tiering.

The downstream supplement now contains 24 manually authored questions over all
ten documents: 12 English and 12 Chinese, with five single-fact, six multi-hop,
eight aggregation and five vector-fallback cases. It references 55
core-direct positive facts and eight explicit hard negatives. The validator
rejects any hard-gated question that references an interpretation-band fact;
the two such dependencies found by the audit were replaced with direct facts.
The validator is
`benchmarks/knowledge_forge_hybrid_rag_v1/validate.py`.

The Chinese supplement has an independent executable validator at
`benchmarks/knowledge_forge_cross_domain_zh_v1/validate.py`. It checks the
generated-file freeze as well as document hashes, mention/coreference spans,
relation evidence, endpoint grounding, tier counts and hard-negative coverage.
A separate semantic reread confirmed that all 62 core facts are directly
supported by prose, table headers or hierarchy indentation; the obligation
inference and negation paraphrase remain the two non-gating interpretation
facts. This is still assistant review, not inter-annotator human agreement.
Because all four Chinese fixtures are short and synthetic, their result must be
paired with the real-PDF workflow and downstream Chinese questions rather than
treated as standalone proof of field performance.

No bulk corpus expansion is recommended now. Ten bilingual documents, 133
core-direct assertion occurrences, 32 targeted negatives, nine domains and 25
structure tags are enough to detect engineering regressions without training a
model on the labels. Adding more labels from the same assistant merely to reach
200 would create false confidence. Add a small independently reviewed document
family only when a real run exposes an uncovered structure/language bucket, a
bucket falls below 0.70 recall, a hard-safety error repeats, or downstream
hybrid retrieval loses accuracy because of a graph extraction omission.

## Current measured baseline

The source-first relation experiment with frozen Gold mentions and an oracle
per-document Schema recovered 76 of 78 all-label unique English facts after
final verification across the saved six-domain runs. Recomputing the primary
score over the second-pass assistant-reviewed tiers yields 68/69 core-direct
facts,
98.6% recall and 97.1% uncertainty-aware precision; interpretation-only outputs
are ignored rather than rewarded or penalized. Minimum document core recall is
87.5%, and no declared hard negative leaked. This comfortably clears the
engineering relation threshold, but it is an upper-bound component test:
entity extraction and project Schema induction are deliberately held out and
must still pass the real persisted workflow. The compact, hash-addressed
result manifest is
`benchmarks/results/knowledge_forge_relation_upper_bound_2026-08-11.json`.
That run used one source-binding recovery round; the current default uses zero,
so the result is not presented as the current end-to-end score.

The graph scorer and source-first runner now use `core_direct` as the primary
tier and report the interpretation band separately. The source-first runner's
default Schema-binding recovery count also matches production (zero); an
explicit one-round run remains available only as an upper-bound ablation.

The bilingual grounded-mention runner is now available at
`benchmarks/evaluate_grounded_mentions.py`. It reports exact and overlapping
span precision/recall plus type accuracy. Canonical B3 is intentionally scored
only from a persisted runtime namespace; the benchmark does not substitute a
surface-string clustering heuristic for the production resolver.

For seven selected Chinese PDF manuals below 1 MiB, generic title/table-row
classification, bounded micro-section coalescing and high-confidence TOC
classification reduced graph-eligible chunks from 453 to 187 compared with the
immediately preceding parser state (59%). The same documents retain 251 vector
retrieval chunks: 35 duplicate parent overviews and 29 TOC children are
retrieval-only. Representative original-to-current graph observations are
59→19, 185→31, 103→23 and 125→46. All paragraph/table source blocks remain in
the section payload; page markers and internal section titles remain in raw
processing provenance, while heading-only TOC markers need not become graph
facts. The document-revision fingerprint includes parser/chunk profile versions
and all relevant chunk settings, so a rebuild cannot silently reuse content
produced by the previous layout algorithm.

An offline preflight over those seven PDFs measured 62,385 graph tokens and
935 scheduled base model calls under the current defaults: 374 grounded-entity
calls and 561 open-fact/Schema-binding calls. The largest single document plans
230 base calls. Every graph chunk is below 1,486 characters, so entity
segmentation does not multiply the chunk count in this sample. These figures
exclude data-dependent Schema induction, reference adjudication, batched final
verification, focused review and retries; they are a workload bound, not an
elapsed-time claim. Schema sampling now consumes the same graph-eligible,
non-parent chunk set, so TOC/reference-only chunks no longer spend Schema model
calls or influence logical types. The new Schema sampling profile is included
in the document-revision fingerprint.

The checked-in deterministic stress runner composes all ten bilingual fixtures
at 1x and 10x. Source bytes grew 9.93x, while graph chunks and the scheduled
entity/open-fact/binding base passes grew 9.09x; the offline scaling gate passes.
This is deliberately not a model-quality score. Enhanced extraction now also
honors `graph_extraction_enabled=false`, so retrieval-only parents, references
and navigation blocks no longer trigger hidden entity/relation model work.

The six-domain binding experiment found only two additional bindings from six
universal second binding calls. The default therefore keeps open-fact gleaning
(which recovered substantial frames) but disables universal Schema-binding
gleaning; the deterministic semantic shortlist remains enabled and deployments
can opt back into one binding recovery round. Both extra bindings occurred in
the Python document. Even pessimistically treating both as distinct required
core hits would leave aggregate core recall at 95.7% and that document at
81.8%, above the current gates. This is a candidate-set bound rather than a
replacement for the pending default-runtime run. Final-verifier `reason_code` is
now bounded to 160 characters so a local model cannot turn a four-claim batch
into thousands of tokens of duplicated prose and force expensive single-call
fallbacks.

### Completed end-to-end runtime gate

The real public build route was exercised repeatedly on isolated enhanced
project 78 with all ten frozen bilingual documents and the configured
`qwen3.6:latest` Ollama runtime. The final persisted public job
`48d55b170daf44fdbb12f20e442ec406` reached `completed` after 103 seconds; its
immutable relation run is
`shadow_relation_v2_74c8a702242d3d394f4ac10ac1aabaecd85bae4fb221af2618b7464f08776d85`.
The graph contains 169 entities and 157 published relations. It retains the
coordinated actor/theme/recipient projections and the parser-grounded
open-carrier projections that were previously lost when an inconsistent batch
verifier rejected them. The concrete persisted checks include
`仪器 -> 值班员周启`, `申请材料 -> 受理机构`, and
`安全报告 -> 受理机构`. The observed false `Purpose -> data set` edge remains
absent while all annotated data-row assignments remain present.

The end-to-end extraction score is intentionally reported as an interval rather
than as perfect Gold precision. The frozen labels are evidence-complete for the
listed assertions but do not declare every other source-supported open fact to
be false. Consequently, treating every annotation-unlisted edge as false would
produce only a conservative lower bound, not a trustworthy precision estimate.
Direct inspection found additional grounded open facts among unmatched rows,
while the explicit hard-negative checks are reliable and had zero leaks.

| Runtime measure | English | Chinese | Combined |
| --- | ---: | ---: | ---: |
| Mention surface recall | 92.4% | 91.2% | 91.8% |
| Canonical surface coverage | 91.1% | 91.5% | 91.3% |
| Core directed endpoint recall | 48/69 (69.6%) | 55/61 (90.2%) | 103/130 (79.2%) |
| Grounded endpoint + evidence recall | 48/69 (69.6%) | 55/61 (90.2%) | 103/130 (79.2%) |
| Declared hard-negative leaks | 0 | 0 | 0 |

The preceding persisted run scored 84.6% combined grounded recall; the latest
result is 5.4 percentage points lower and remains inside the declared 0.08
run-to-run stability band. Comparison of both immutable ledgers attributes most
of the difference to conservative entity under-merging and local-model frame
variation, not to the declared structural open-carrier suppression. Ambiguous
short/full forms remain separate by policy because a false canonical merge is
more damaging than a missed graph edge in the hybrid retrieval product.

The strongest structural results are English table 10/10, Chinese
hierarchy/list 7/7, Chinese table 6/6, English numbered clauses 6/8,
requirements 3/3 and English section hierarchy 2/2.
The original ACore-style capability example is therefore represented as
separate navigable platform-to-capability and parent-to-child edges rather than
one generic co-occurrence edge. The public graph and search routes both return
all six representative ACore platform capabilities and all three health-monitor
levels (`核心级`, `多分区级`, `分区级`). Remaining weak diagnostic buckets are
English free prose, cross-sentence/coreference cases and a few singleton
code/section labels. Those omissions are material limitations, but they are
covered by direct chunk retrieval and do not create unsupported graph answers
in the current downstream gate.

A separate real-PDF regression used public job
`6341b4b08b59484abb44eeb325b33620` and immutable relation run
`shadow_relation_v2_7a0a7b3a3a608756748ca16427f42b4dc4d4064ab8bad4d9cd032be460d44d1a`.
It completed with 508 entities and 288 published relations. The audited
capability slice retained 17/17 core-OS children, 3/3 task-communication
members, 3/3 health-monitor levels and 4/4 multicore children. No nested child
was attached directly to the core-OS root, and there were no generic-predicate
edges or self-loops. The parser's explicit exhaustive-structure declaration
removed nine overlapping open-carrier evidence rows; ordinary list, table and
section structures do not declare that policy and remain additive.

The final public top-10 hybrid retrieval evaluator passed all release checks on
24 frozen questions. Vector and hybrid retrieval each succeeded on 22/24
questions and graph-only evidence satisfied 13/24; the graph supplied 36/43
(83.7%) required facts on graph-preferred questions while returning only 7,638
characters versus 60,858 vector-context characters (12.6%). There were zero
negative-relation leaks and zero published
generic `related_to`/`relation` edges. Exact predicate-name agreement remains a
diagnostic for an independently induced open Schema; directed endpoints plus
matching source evidence are the product fact contract.

Recent persisted post-fix builds retained zero hard-negative leaks despite
ordinary local-model output variation. The final compilation and regression
suite completed with 1,104 tests passed and 45 intentionally skipped.

## Why this follows mature benchmark practice

- [DocRED](https://aclanthology.org/P19-1074/) jointly annotates entities and
  document-level relations, including facts requiring evidence across
  sentences.
- [Re-DocRED](https://aclanthology.org/2022.emnlp-main.580/) showed that a
  recommend-and-revise candidate workflow still leaves many false negatives.
  Our recall universe therefore includes a source-first manual pass and does
  not treat every unlisted pair as a negative.
- [HacRED](https://aclanthology.org/2021.findings-acl.249/) separates relation,
  entity and triple annotation and uses three annotators per item. We adopt the
  stage separation and hard-case buckets, while explicitly retaining
  single-expert/non-production status.
- [CBLUE/CMeIE](https://github.com/CBLUEbenchmark/CBLUE) reports micro-F1 for
  schema-constrained Chinese relation extraction. Our Chinese supplement adds
  general-domain hierarchy, table, regulation and coreference cases rather
  than importing a medical training distribution.
- [Text2KGBench](https://arxiv.org/abs/2308.02357) measures fact extraction,
  ontology conformance and subject/relation/object hallucination separately.
  Knowledge Forge likewise gates evidence, normalization and unsupported-edge
  safety independently of exact triple F1.
- [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
  evaluates fact retrieval, complex reasoning, contextual summarization and
  generation. This is why the final gate includes downstream questions rather
  than assuming a denser graph is automatically better.

## Engineering acceptance thresholds

These thresholds apply to the core-direct calibration tier. The all-label tier
is reported as a sensitivity interval, not a hard gate.

### Entity extraction and normalization

- one-to-one overlap mention precision at least 0.90 and recall at least 0.85;
- exact-span F1 remains a diagnostic because single-expert boundary choices
  are not absolute; strict span-plus-type micro-F1, macro-F1 and per-type
  counts are reported separately and may not hide a failed mention gate;
- canonical entity B3 F1 at least 0.85;
- zero declared cannot-link violations and zero known false automatic merges;
- every published entity keeps document/chunk provenance.

Conservative under-merging is acceptable because vector retrieval can still
reach separate mentions. A false merge is more damaging than leaving an
ambiguous candidate separate.

### Relation extraction and normalization

- oracle-Schema component precision at least 0.88;
- oracle-Schema component recall at least 0.85;
- candidate endpoint-pair recall at least 0.90;
- direction accuracy at least 0.90 for non-symmetric relations;
- exact evidence-span validity 1.00;
- no explicit-negation leak, `related_to` publication, illegal self-loop or
  unsupported reverse conflict;
- no repeated, adequately populated document/language/structure bucket below
  0.70 core recall; singleton buckets and cross-sentence coreference remain
  diagnostic for the hybrid-RAG release.

For the oracle-Schema component test, canonical precision/recall includes exact
predicate identity. For a real project with an independently induced additive
Schema, directed endpoint plus exact evidence is the primary fact/retrieval
match; exact Gold predicate-string agreement is diagnostic unless a semantic
alignment has been adjudicated. Every published predicate must still belong to
the active Schema, repeated source meanings must normalize consistently, and a
generic `related_to`/`relation` edge is always a failure. This prevents harmless
names such as `depends_on` versus `condition_for` from hiding whether the graph
actually retrieved the source fact, without weakening evidence or safety.

A small number of optional temporal or ontology abstractions may remain
`UNCERTAIN` without blocking use. They must not be silently published as facts.

### Stability and efficiency

- run the bilingual calibration twice; aggregate core F1 may vary by at most
  0.08 and no repeated hard-safety error is allowed;
- increasing relation-bearing chunks by 10x must increase model calls and
  elapsed extraction work by no more than 15x, excluding one-time model load;
- pair opportunities and focused reviews remain configuration-bounded; no
  all-entity Cartesian model pass is allowed;
- every selected document must reach a persisted terminal job state without a
  stale lease or process-memory-only status;
- wall-clock results for the selected sub-1MB real documents are recorded as a
  hardware baseline rather than inventing a platform-independent SLA.

### Downstream hybrid RAG

Use at least 24 frozen questions divided among single-fact retrieval,
multi-hop/relationship retrieval, aggregation/context questions, and details
that should fall back to vector chunks.

- hybrid retrieval must not trail vector-only citation-supported answer
  accuracy by more than two percentage points overall;
- at top 10, graph-preferred questions must recover at least 70% of their
  required facts while graph context uses no more than 35% of vector context;
- unsupported-answer and wrong-citation rates must not increase;
- graph-only is diagnostic, not a required default: direct knowledge-base
  retrieval remains available for facts the graph omits.

The retrieval runner counts a graph contribution only when the directed
endpoints and source evidence match. It reports exact predicate normalization
on those grounded hits separately and fails the run on hard-negative or generic
relation publication.

## Stop rule

The current engineering extraction work can be closed when the bilingual
aggregate endpoint/evidence score is at least 0.70, hard-safety checks pass,
long-document work scales within the configured bound, and downstream hybrid
retrieval passes. Oracle component targets and weak structure buckets remain
regression diagnostics; they do not justify indefinite prompt tuning when the
graph is only one evidence source beside direct chunks. Reaching 200 labels is
not part of this stop rule.

All current closure gates are complete. The deterministic 10x workload gate
passed, the public persisted Ollama build completed, bilingual aggregate
grounded recall is 79.2%, and hybrid retrieval passed without safety leaks.
Re-open extraction for a reproducible hard-safety error, superlinear cost
regression, repeated populated structure bucket below 0.70, or a downstream
retrieval deficit attributable to missing/incorrect graph facts. A future claim
of production-grade extraction quality would still require independent human
adjudication and a genuinely held-out corpus.

Before restarting the service, the active `graphrag_native_build` query returned
no queued or running jobs. After restart, the public health route reported
PostgreSQL reachable, two Ollama models and HugeGraph ready. This host uses the
configured HugeGraph port 8081 (port 8080 belongs to another local service), and
`http://127.0.0.1:8081/graphs` returned `knowledge_forge`.
