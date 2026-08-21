# Knowledge Forge Chinese calibration supplement v1

This is a small, frozen Chinese-language supplement to
`knowledge_forge_cross_domain_v1`. It targets four source shapes that are
important to Knowledge Forge: hierarchical feature lists, header-bound tables,
numbered obligations, and narrative coreference/negation.

The source declarations and relation decisions in `build.py` were reviewed as
an expert annotation task. The script computes offsets, hashes, and generated
JSON only; it does not ask a model to infer labels. Validate both frozen output
and the independent hash/span/endpoint contracts with:

```bash
.venv/bin/python benchmarks/knowledge_forge_cross_domain_zh_v1/build.py --check
.venv/bin/python benchmarks/knowledge_forge_cross_domain_zh_v1/validate.py
```

This remains a single-expert calibration corpus. It is useful for regression,
language and structure checks, but it is not independent human Gold and is not
eligible to approve a production release. Primary acceptance reports should
score `core_direct` facts separately from the small optional interpretation
band (`obligation_inference` and `negation_paraphrase`).

The shared graph scorer can evaluate this corpus with
`benchmarks/knowledge_forge_cross_domain_v1/score_graph.py --dataset-root`
and enforces that tier separation. When English and Chinese files share one
runtime project, also pass the four Chinese runtime IDs through
`--document-ids` so relation precision is corpus-local.

The documents are original synthetic fixtures, so no uploaded, proprietary, or
third-party source text is stored here.
