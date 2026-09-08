#!/usr/bin/env python3
"""Validate the frozen Knowledge Agent V1 engineering question set."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> dict[str, object]:
    manifest = _load(ROOT / "manifest.json")
    base = _load((ROOT / manifest["base_questions_path"]).resolve())
    probes = _load(ROOT / manifest["probes_path"])
    rows = [*base, *probes]
    ids = [str(row.get("question_id") or "") for row in rows]
    assert all(ids) and len(ids) == len(set(ids)), "question IDs must be non-empty and unique"
    for row in rows:
        assert str(row.get("language") or "") in {"zh-CN", "en"}
        assert str(row.get("question") or "").strip()
        assert str(row.get("reference_answer") or "").strip()
        assert isinstance(row.get("answer_atoms"), list)
        if row.get("expected_outcome") == "refusal":
            assert row["task_type"] == "no_answer"
            assert not row["answer_atoms"]
    expected = manifest["expected_counts"]
    assert len(base) == int(expected["base_questions"])
    assert len(probes) == int(expected["probes"])
    languages = Counter(str(row["language"]) for row in rows)
    assert dict(languages) == expected["languages"]
    unanswerable = sum(row.get("expected_outcome") == "refusal" for row in rows)
    assert unanswerable == int(expected["unanswerable"])
    return {
        "status": "valid",
        "questions": len(rows),
        "base_questions": len(base),
        "probes": len(probes),
        "languages": dict(sorted(languages.items())),
        "unanswerable": unanswerable,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
