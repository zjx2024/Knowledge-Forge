# Knowledge Forge cross-domain benchmark v1

This directory is a frozen **expert-annotated calibration corpus**, not production Gold. It is
intended to make relation-extraction changes measurable across domains and
document structures before any model or policy change is accepted.

The six snapshots deliberately cover different failure modes:

| ID | Domain | Structure | What it stresses |
| --- | --- | --- | --- |
| `software-python-reference` | programming language | numbered heading, prose, code block | hierarchy, definitions, code adjacency |
| `network-rfc9110-excerpt` | protocol standard | numbered normative sections | direction, roles, request/response symmetry |
| `legal-ecfr-grants-excerpt` | regulation | subsection and numbered clauses | obligations, roles, list adjacency |
| `knowledge-usgs-aquifer-report-excerpt` | earth science | narrative plus pipe table | table cells, provenance, source/purpose roles |
| `engineering-nasa-utm-requirements-excerpt` | aerospace software | service map plus requirement IDs | mapping tables, `shall` obligations, identifier traps |
| `narrative-alice-excerpt` | public-domain fiction | prose, dialogue-like action, coreference | event order, pronouns, non-entailment |

The local documents are short, curated source excerpts or faithful benchmark
snapshots, not a redistribution of the full source documents. `SOURCES.json`
records the authoritative URL, access date, usage note, and content hash. The
source pages should be consulted before expanding an excerpt or redistributing
larger text.

## Annotation contract

`annotations/assertions.json` is the primary, complete expert assertion
universe. It contains 80 positive evidence occurrences (78 unique semantic
facts) and 16 predicate-specific hard negatives. `annotations/ambiguous_claims.json`
quarantines one syntactically plausible but not directly entailed attachment
instead of forcing it into either class. The benchmark explicitly distinguishes
header-bound table semantics from raw cell adjacency, and preserves code,
requirement, section-hierarchy and narrative-coreference cases.

`annotations/pairs.json` is the older 43-case calibration slice retained for
historical A/B comparisons. Every pair in that slice is assigned one of:

* `HAS_RELATION`: a fact is asserted by the excerpt and has evidence;
* `NO_RELATION`: the endpoints co-occur or are structurally adjacent, but the
  excerpt does not assert a relation;
* `OTHER_UNSUPPORTED`: the excerpt directly asserts a relation for this pair,
  but that relation is outside the benchmark's frozen base predicate
  vocabulary. These are Schema-extension positives, not false-positive or
  `NO_RELATION` cases; their directed labels live in
  `annotations/unsupported_facts.json`.

`annotations/entities.json` is the expert entity-mention Gold for the
explicit domain-bearing terms in these six snapshots. It includes all such
mentions under the stated coverage policy, canonical IDs, and coarse entity
types. `annotations/coreference_links.json` records the pronoun/antecedent
links that are needed by the narrative case; pronouns are deliberately not
published as independent entities.

`annotations/facts.json` and `annotations/unsupported_facts.json` are the
historical 31-fact anchor subset. `annotations/assertions.json` supersedes them
for completeness and precision measurement. Offsets
are byte-independent Python string offsets into the corresponding UTF-8 text
file. `evidence_text` must equal the exact half-open span in the document.

The current labels are `assistant_expert_annotator`: the assistant completed the
entity spans, coarse types, coreference links, 80 directed assertion
occurrences, one ambiguous attachment, and 16 predicate-specific hard negatives
as the expert annotator.
The assertion universe is complete under the stated policy: it covers all
source-asserted binary frames in these frozen excerpts, while the negative set
targets high-risk structural/co-occurrence claims rather than the mathematically
exhaustive cartesian product of every entity pair. This is a single-expert calibration
Gold, so `production_gate_eligible` remains false until an independent
inter-annotator audit is performed. That audit is a release-quality check, not
a prerequisite for using this corpus to tune and regression-test the system.

## Suggested tuning sequence

1. Run extraction without changing policy and record endpoint recall, pair
   coverage, fact recall, direction accuracy, exact evidence coverage, and
   false-positive reasons per document structure.
2. Fix candidate generation and evidence recovery first; do not tune predicate
   names while the endpoint or pair universe is incomplete.
3. Tune direction and role validation on the RFC, legal, and requirements
   snapshots; tune table adjacency rejection on the USGS and NASA snapshots.
4. Tune coreference and event non-entailment on Alice, then rerun every other
   snapshot to catch regressions.
5. After human adjudication, keep one document or section family as holdout and
   require a release gate over the complete corpus.

`score_graph.py` reports `core_direct` as the primary uncertainty-aware score,
keeps interpretation-only outputs out of the primary precision denominator,
and reports all labels as a separate sensitivity band. It accepts
`--dataset-root` so the same contract can score the Chinese supplement. The
report also includes semantic-fact endpoint recall, exact-predicate recall,
evidence-grounded variants of both metrics, per-structure recall and
predicate-specific negative leakage. Precision credit and hard-negative leaks
require matching source evidence, not merely a graph-global endpoint pair. The
optional `--document-ids 101,102,...` slice must be supplied when one runtime
project contains both language corpora; it prevents the other corpus's edges
from being counted as false positives. Entity surface coverage remains a
project-wide diagnostic because the public graph node payload is canonical and
does not claim a single source document.
historical
8-base/23-extension slice is still included under `historical_anchor`, but its
occurrence-local `NO_RELATION` cases are explicitly non-gating when another
occurrence supports a graph-global fact for the same canonical pair.

## Files

* `SOURCES.json` — source provenance and usage notes.
* `documents/` — frozen text inputs.
* `annotations/entities.json` — 144 explicit entity mentions and coarse types.
* `annotations/coreference_links.json` — 11 narrative pronoun/antecedent links.
* `annotations/assertions.json` — complete 78-unique-positive/16-negative expert universe.
* `annotations/ambiguous_claims.json` — quarantined claims whose evidence does not determine one reading.
* `annotations/pairs.json` — 43 expert-labeled pair cases.
* `annotations/facts.json` — historical 11-fact base-schema anchor.
* `annotations/unsupported_facts.json` — historical 20-fact extension anchor.
* `manifest.json` — dataset metadata and hashes.
