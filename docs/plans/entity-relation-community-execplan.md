# Entity, Relation, and Community Continuous Execution Plan

## Objective

Improve entity admission and resolution measurability, recover bounded relation
recall without weakening publishable-graph safety, and complete a snapshot-bound
GraphRAG community detection, report, and opt-in global-search shadow path.

The current production/control projection remains unchanged unless an independent
human Gold gate and explicit product approval are both available. P1/P2 treatment
data stays isolated and shadow by default.

## Baseline

- Historical anchor: `74a12308a92d16ff93969f28e76c82fb90ab5643`
- Start HEAD: `74a12308a92d16ff93969f28e76c82fb90ab5643`
- Start `origin/main`: `cb0d833944e474bbae963b3c14bf4364b01aeb9f`
- Working branch: `feat/entity-relation-recall-community`
- GraphRAG dependency: `3.1.1`
- Existing community implementation: explicit BYOG `create_communities` shadow
- Existing entity/relation implementation: control plus immutable P1/P2 shadow

## Invariants

1. Candidate relations originate only from current active-revision mentions and
   a bounded local context window.
2. Cross-document canonical reuse never creates relation evidence by itself.
3. Shadow work cannot mutate product facts, graph generation, manifests,
   communities, reports, or default query results.
4. Incomplete candidate coverage fails closed.
5. Unknown, uncertain, retryable, and unsupported relation states are never
   converted to `related_to` for publication.
6. No treatment is activated without human Gold, isolation proof, and explicit
   product approval.

## Phases

### P0 - Baseline and conserved funnel

- Freeze corpus, revision, policy, prompt, model, evaluator, and graph hashes.
- Add mutually exclusive entity/relation funnel terminal states and conservation
  checks.
- Separate old runtime statistics, current publishable control, and treatment
  shadow statistics.

### P1 - Evaluator correctness

- Cover wildcard type rules and exact evidence/alias behavior.
- Consume must-link, cannot-link, hard-negative, entity candidate Recall@K, and
  relation-pair candidate Recall@budget using correctly named metrics.
- Emit confusion matrices, denominators, sample IDs, confidence intervals, and a
  human annotation queue when Gold is insufficient.

### P2 - Entity shadow v2

- Separate stable occurrences, versioned assertions, admission, candidate
  generation, and canonical decisions.
- Preserve punctuation-bearing identifiers and explicit retryable states.
- Prove formal-table immutability with a restricted disposable PostgreSQL role.

### P3 - Relation shadow v2

- Generate bounded same-sentence, adjacent-sentence, and chunk-boundary pairs.
- Classify only versioned predicates, `NO_RELATION`, or `OTHER_UNSUPPORTED`.
- Make repair coverage-driven and resumable without rescanning arbitrary pairs.

### P4 - Frozen A/B and release gate

- Compare control and treatment on one immutable input/Gold universe.
- Keep treatment shadow when human Gold is absent or any quality/isolation gate
  fails.

### P5 - Community detection verification

- Re-run deterministic GraphRAG BYOG detection against current publishable
  control only, including real disposable PostgreSQL persistence.
- Preserve active-revision canonical entities without publishable edges as
  deterministic singleton communities.

### P6 - Community reports shadow

- Add snapshot-bound, resumable, atomically activated report runs using the
  verified GraphRAG 3.1.1 public workflow boundary.
- Validate citations back to active revision evidence.
- Keep incomplete runs non-active; support retryable resume and zero-call empty
  graph completion.

### P7 - Global Search opt-in shadow

- Add an explicit non-default map/reduce query path over current complete report
  runs with fail-closed generation and citation checks.
- Persist query/report/model hashes and map/reduce provenance without changing
  the product's default search path.

### P8 - Release evidence

- Run no-external tests, disposable PostgreSQL, compilation, dependency, lint,
  migration, Compose, and CPU Docker gates.
- Document all false/blocked production states and push only the feature branch.

## Phase Evidence

Evidence and machine-resume state live under
`.codex/work/entity-relation-community-evidence/` and
`.codex/work/entity-relation-community-state.yaml`. These files are local and
must not be committed.

## Completion Semantics

Engineering completion, human Gold approval, shadow isolation, community
detection, live report verification, Global Search release readiness, and
production approval are independent booleans. Passing one never implies another.
