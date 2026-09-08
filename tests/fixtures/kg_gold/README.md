# Knowledge graph gold annotation bundle

This directory is a schema-valid engineering fixture, not human-confirmed Gold.
It exercises the evaluator and must never authorize a production policy gate.

For a production bundle, copy the manifest and annotation shapes, expand to the
required representative corpus, set every label source to `human_reviewed`, and
record independent reviewer IDs, review times, frozen dev/holdout splits, source
content hashes, and compatible policy/schema ranges.
