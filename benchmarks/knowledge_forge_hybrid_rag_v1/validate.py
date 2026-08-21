#!/usr/bin/env python3
"""Validate the manually authored hybrid-RAG engineering question set."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.calibration_policy import partition_assertions  # noqa: E402


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> dict[str, object]:
    manifest = _load(ROOT / "manifest.json")
    questions = _load(ROOT / str(manifest["questions_path"]))
    documents: dict[str, dict[str, object]] = {}
    assertions: dict[str, dict[str, object]] = {}
    negative_claims: dict[str, dict[str, object]] = {}
    interpretation_assertion_ids: set[str] = set()
    for relative in manifest["source_corpora"]:
        corpus_root = (ROOT / str(relative)).resolve()
        corpus_manifest = _load(corpus_root / "manifest.json")
        for document in corpus_manifest["documents"]:
            document_id = str(document["id"])
            if document_id in documents:
                raise AssertionError(f"duplicate document id: {document_id}")
            documents[document_id] = document
        annotation_path = corpus_root / str(
            corpus_manifest["annotations"]["assertions"]
        )
        annotation = _load(annotation_path)
        bands = partition_assertions(
            corpus_root, list(annotation["relations"]),
        )
        interpretation_assertion_ids.update(
            str(row["assertion_id"])
            for row in bands["interpretation_band"]
        )
        for row in annotation["relations"]:
            assertion_id = str(row["assertion_id"])
            if assertion_id in assertions:
                raise AssertionError(f"duplicate assertion id: {assertion_id}")
            assertions[assertion_id] = row
        for row in annotation["negative_claims"]:
            claim_id = str(row["claim_id"])
            if claim_id in negative_claims:
                raise AssertionError(f"duplicate negative claim id: {claim_id}")
            negative_claims[claim_id] = row

    question_ids: set[str] = set()
    top_k = int(manifest["retrieval_acceptance"]["top_k"])
    for question in questions:
        question_id = str(question["question_id"])
        if question_id in question_ids:
            raise AssertionError(f"duplicate question id: {question_id}")
        question_ids.add(question_id)
        document_id = str(question["document_id"])
        if document_id not in documents:
            raise AssertionError(f"unknown document: {document_id}")
        if not str(question.get("question") or "").strip():
            raise AssertionError(f"empty question: {question_id}")
        if not str(question.get("reference_answer") or "").strip():
            raise AssertionError(f"empty reference answer: {question_id}")
        if not question.get("answer_atoms"):
            raise AssertionError(f"empty answer atoms: {question_id}")
        positive_ids = list(question.get("required_assertion_ids") or ())
        negative_ids = list(question.get("required_negative_claim_ids") or ())
        if len(positive_ids) > top_k:
            raise AssertionError(
                f"question requires {len(positive_ids)} facts but top_k is "
                f"{top_k}: {question_id}"
            )
        if not positive_ids and not negative_ids:
            raise AssertionError(f"question has no grounded labels: {question_id}")
        for assertion_id in positive_ids:
            row = assertions.get(str(assertion_id))
            if row is None or str(row["document_id"]) != document_id:
                raise AssertionError(
                    f"invalid assertion reference {assertion_id}: {question_id}"
                )
            if str(assertion_id) in interpretation_assertion_ids:
                raise AssertionError(
                    "hard-gated question references interpretation-band "
                    f"assertion {assertion_id}: {question_id}"
                )
        for claim_id in negative_ids:
            row = negative_claims.get(str(claim_id))
            if row is None or str(row["document_id"]) != document_id:
                raise AssertionError(
                    f"invalid negative reference {claim_id}: {question_id}"
                )
        inherited_language = str(documents[document_id].get("language") or "en")
        if str(question["language"]) != inherited_language:
            raise AssertionError(f"language mismatch: {question_id}")
        if (
            question["task_type"] == "vector_fallback"
            and not negative_ids
        ):
            raise AssertionError(
                f"vector fallback lacks negative evidence: {question_id}"
            )

    expected = manifest["expected_counts"]
    languages = Counter(str(item["language"]) for item in questions)
    task_types = Counter(str(item["task_type"]) for item in questions)
    covered = {str(item["document_id"]) for item in questions}
    assert len(questions) == int(expected["questions"])
    assert dict(sorted(languages.items())) == dict(
        sorted(expected["languages"].items())
    )
    assert dict(sorted(task_types.items())) == dict(
        sorted(expected["task_types"].items())
    )
    assert len(covered) == int(expected["documents_covered"])
    return {
        "status": "valid",
        "questions": len(questions),
        "languages": dict(sorted(languages.items())),
        "task_types": dict(sorted(task_types.items())),
        "documents_covered": len(covered),
        "positive_labels_referenced": len({
            str(value)
            for item in questions
            for value in item.get("required_assertion_ids") or ()
        }),
        "interpretation_labels_referenced": len({
            str(value)
            for item in questions
            for value in item.get("required_assertion_ids") or ()
            if str(value) in interpretation_assertion_ids
        }),
        "negative_labels_referenced": len({
            str(value)
            for item in questions
            for value in item.get("required_negative_claim_ids") or ()
        }),
    }


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
