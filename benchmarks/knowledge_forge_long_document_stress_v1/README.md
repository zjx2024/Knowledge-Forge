# Knowledge Forge long-document stress v1

This fixture deliberately does **not** add relation or entity labels. It keeps
quality calibration separate from scalability measurement, so repeated source
facts cannot inflate recall or precision.

The runner composes the ten frozen English/Chinese calibration documents at
1x and 10x scale, then invokes the production parser and general chunker. It
checks that graph-eligible chunks and the deterministic base model-pass
schedule grow no faster than 1.5 times source growth. Model execution,
verification calls, retry behavior and wall-clock acceptance remain part of
the real HTTP/Ollama benchmark; this fixture catches parser/chunker
combinatorial regressions quickly and offline.

Run:

```bash
.venv/bin/python benchmarks/knowledge_forge_long_document_stress_v1/evaluate.py
```

The seven locally available ACoreOSMP PDF manuals are a separate, non-checked-
in real-document stress sample. They are not treated as Gold and their source
text is never copied into this repository.
