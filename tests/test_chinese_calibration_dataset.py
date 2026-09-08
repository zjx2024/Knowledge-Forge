from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.knowledge_forge_cross_domain_zh_v1 import build, validate
from benchmarks.audit_calibration_corpora import audit


ROOT = Path(build.__file__).resolve().parent


def _load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_chinese_calibration_generated_files_are_frozen_and_complete():
    for path, value in build.build_payloads().items():
        rendered = build._render(value)
        assert path.read_text(encoding="utf-8") == rendered

    manifest = _load("manifest.json")
    assert manifest["production_gate_eligible"] is False
    assert manifest["expected_counts"] == {
        "documents": 4,
        "entity_mentions": 113,
        "canonical_entities": 71,
        "coreference_links": 3,
        "complete_positive_assertions": 64,
        "unique_positive_facts": 63,
        "predicate_specific_negative_claims": 16,
    }
    assert {item["split"] for item in manifest["documents"]} == {
        "dev", "holdout",
    }
    for document in manifest["documents"]:
        content = (ROOT / document["path"]).read_text(encoding="utf-8")
        assert any("\u4e00" <= value <= "\u9fff" for value in content)
        assert hashlib.sha256(content.encode("utf-8")).hexdigest() == (
            document["content_sha256"]
        )


def test_chinese_calibration_relation_endpoints_are_grounded_mentions_or_coref():
    entities = _load("annotations/entities.json")["documents"]
    assertions = _load("annotations/assertions.json")
    coreference = _load("annotations/coreference_links.json")
    mention_by_id = {
        item["mention_id"]: item
        for rows in entities.values() for item in rows
    }
    coref_by_document_span = {
        (item["document_id"], *item["source_span"]): mention_by_id[
            item["target_mention_id"]
        ]["canonical_id"]
        for item in coreference
    }
    by_document_canonical = {
        document_id: {item["canonical_id"] for item in rows}
        for document_id, rows in entities.items()
    }

    def endpoint_is_grounded(
        document_id: str, canonical_id: str, span: list[int],
    ) -> bool:
        start, end = span
        if coref_by_document_span.get((document_id, start, end)) == canonical_id:
            return True
        return any(
            item["canonical_id"] == canonical_id
            and item["span"][0] <= start < end <= item["span"][1]
            for item in entities[document_id]
        )

    rows = [*assertions["relations"], *assertions["negative_claims"]]
    for item in rows:
        document_id = item["document_id"]
        content = (ROOT / f"documents/{document_id}.txt").read_text(
            encoding="utf-8"
        )
        evidence_start, evidence_end = item["evidence_span"]
        assert content[evidence_start:evidence_end] == item["evidence_text"]
        assert item["subject_canonical_id"] in by_document_canonical[document_id]
        assert item["object_canonical_id"] in by_document_canonical[document_id]
        assert item["subject_canonical_id"] != item["object_canonical_id"]
        for endpoint in ("subject", "object"):
            start, end = item[f"{endpoint}_span"]
            assert evidence_start <= start < end <= evidence_end
            assert endpoint_is_grounded(
                document_id,
                item[f"{endpoint}_canonical_id"],
                item[f"{endpoint}_span"],
            )

    core = [
        item for item in assertions["relations"]
        if item["adjudication_tier"] == "core_direct"
    ]
    assert len(core) == 62
    assert {
        item["document_id"] for item in assertions["negative_claims"]
    } == set(entities)
    assert all(
        sum(
            item["document_id"] == document_id
            for item in assertions["negative_claims"]
        ) >= 4
        for document_id in entities
    )


def test_combined_calibration_is_bilingual_but_not_production_gold():
    report = audit()
    assert report["status"] == "engineering_calibration_adequate"
    assert report["totals"]["documents"] == 10
    assert report["totals"]["languages"] == {"en": 6, "zh-CN": 4}
    assert report["totals"]["core_positive_assertions"] == 133
    assert report["totals"]["hard_negatives"] == 32
    assert report["adequacy"] == {
        "engineering_calibration": True,
        "production_release_gold": False,
        "downstream_question_set": True,
        "long_document_scaling": True,
        "dataset_expansion_required_now": False,
    }
    assert "no_long_document_performance_corpus" not in report["gaps"]
    assert "no_downstream_hybrid_rag_question_set" not in report["gaps"]
    assert report["long_document"]["status"] == "pass"
    assert report["long_document"]["quality_gold"] is False
    assert report["downstream"]["positive_labels"] == 55
    assert report["downstream"][
        "interpretation_labels_in_hard_gate"
    ] == []
    assert report["expansion_policy"]["bulk_expansion_recommended_now"] is False


def test_chinese_calibration_has_a_standalone_semantic_validator():
    report = validate.validate()
    assert report["status"] == "valid_calibration_corpus"
    assert report["assertions_by_tier"] == {
        "core_direct": 62,
        "negation_paraphrase": 1,
        "obligation_inference": 1,
    }
    assert report["predicate_specific_negative_claims"] == 16
