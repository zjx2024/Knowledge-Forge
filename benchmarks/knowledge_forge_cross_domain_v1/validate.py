#!/usr/bin/env python3
"""Validate the local cross-domain calibration corpus without model calls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.calibration_policy import partition_assertions  # noqa: E402


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    manifest = read_json("manifest.json")
    entity_payload = read_json(manifest["annotations"]["entities"])
    coreference_links = read_json(manifest["annotations"]["coreference_links"])
    assertions = read_json(manifest["annotations"]["assertions"])
    assertion_bands = partition_assertions(ROOT, assertions["relations"])
    ambiguous_payload = read_json(manifest["annotations"]["ambiguous_claims"])
    pairs = read_json(manifest["annotations"]["pairs"])
    facts = read_json(manifest["annotations"]["facts"])
    unsupported = read_json(manifest["annotations"]["unsupported_facts"])
    assert manifest["production_gate_eligible"] is False
    assert manifest["status"] == "expert_annotated_calibration"

    documents = {}
    for item in manifest["documents"]:
        path = ROOT / item["path"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == item["content_sha256"], (item["id"], digest)
        documents[item["id"]] = (item, raw.decode("utf-8"))

    mention_ids = set()
    mention_canonical_by_span = {}
    canonical_by_mention_id = {}
    mention_count = 0
    for document_id, rows in entity_payload["documents"].items():
        _, text = documents[document_id]
        for mention in rows:
            mention_id = str(mention["mention_id"])
            assert mention_id not in mention_ids, mention_id
            mention_ids.add(mention_id)
            canonical_by_mention_id[mention_id] = str(mention["canonical_id"])
            mention_count += 1
            start, end = mention["span"]
            assert 0 <= start < end <= len(text), mention_id
            assert text[start:end] == mention["surface"], mention_id
            mention_canonical_by_span[(document_id, start, end)] = str(
                mention["canonical_id"]
            )
            assert mention["type"] in {"GenericEntity", "Person", "Organization", "Location", "Product", "Technology", "Event"}

    assert mention_count == manifest["expected_counts"]["entity_mentions"]
    coreference_canonical_by_span = {}
    for link in coreference_links:
        _, text = documents[link["document_id"]]
        start, end = link["source_span"]
        assert text[start:end] == link["source_text"], link["link_id"]
        assert link["target_mention_id"] in mention_ids, link["link_id"]
        coreference_canonical_by_span[(
            link["document_id"], start, end,
        )] = canonical_by_mention_id[link["target_mention_id"]]
    assert len(coreference_links) == manifest["expected_counts"]["coreference_links"]

    assertion_ids = set()
    positive_by_structure = Counter()
    positive_by_schema = Counter()
    for assertion in assertions["relations"]:
        assertion_id = str(assertion["assertion_id"])
        assert assertion_id not in assertion_ids, assertion_id
        assertion_ids.add(assertion_id)
        document_id = str(assertion["document_id"])
        _, text = documents[document_id]
        evidence_start, evidence_end = assertion["evidence_span"]
        assert text[evidence_start:evidence_end] == assertion["evidence_text"], assertion_id
        for role in ("subject", "object"):
            start, end = assertion[f"{role}_span"]
            canonical_id = str(assertion[f"{role}_canonical_id"])
            resolved = mention_canonical_by_span.get((document_id, start, end))
            if resolved is None:
                resolved = coreference_canonical_by_span.get(
                    (document_id, start, end)
                )
            assert resolved == canonical_id, (assertion_id, role, resolved, canonical_id)
            assert evidence_start <= start < end <= evidence_end, (assertion_id, role)
        assert assertion["schema_status"] in {"base", "extension"}, assertion_id
        positive_by_structure[str(assertion["structure"])] += 1
        positive_by_schema[str(assertion["schema_status"])] += 1

    negative_ids = set()
    negative_by_reason = Counter()
    for claim in assertions["negative_claims"]:
        claim_id = str(claim["claim_id"])
        assert claim_id not in negative_ids, claim_id
        negative_ids.add(claim_id)
        document_id = str(claim["document_id"])
        _, text = documents[document_id]
        evidence_start, evidence_end = claim["evidence_span"]
        assert 0 <= evidence_start < evidence_end <= len(text), claim_id
        for role in ("subject", "object"):
            start, end = claim[f"{role}_span"]
            canonical_id = str(claim[f"{role}_canonical_id"])
            resolved = mention_canonical_by_span.get((document_id, start, end))
            if resolved is None:
                resolved = coreference_canonical_by_span.get(
                    (document_id, start, end)
                )
            assert resolved == canonical_id, (claim_id, role, resolved, canonical_id)
            assert evidence_start <= start < end <= evidence_end, (claim_id, role)
        negative_by_reason[str(claim["reason_code"])] += 1

    assert len(assertion_ids) == manifest["expected_counts"]["complete_positive_assertions"]
    assert len(assertion_bands["core_direct"]) == 71
    assert len(assertion_bands["interpretation_band"]) == 9
    assert len(negative_ids) == manifest["expected_counts"]["predicate_specific_negative_claims"]

    ambiguous_ids = set()
    for claim in ambiguous_payload["claims"]:
        claim_id = str(claim["claim_id"])
        assert claim_id not in ambiguous_ids, claim_id
        assert claim_id not in negative_ids, claim_id
        assert claim_id not in assertion_ids, claim_id
        ambiguous_ids.add(claim_id)
        document_id = str(claim["document_id"])
        _, text = documents[document_id]
        evidence_start, evidence_end = claim["evidence_span"]
        assert text[evidence_start:evidence_end] == claim["evidence_text"], claim_id
        for role in ("subject", "object"):
            start, end = claim[f"{role}_span"]
            canonical_id = str(claim[f"{role}_canonical_id"])
            resolved = mention_canonical_by_span.get((document_id, start, end))
            if resolved is None:
                resolved = coreference_canonical_by_span.get(
                    (document_id, start, end)
                )
            assert resolved == canonical_id, (claim_id, role, resolved, canonical_id)
            assert evidence_start <= start < end <= evidence_end, (claim_id, role)
        assert claim["reason_code"] == "PP_ATTACHMENT_AMBIGUOUS", claim_id
    assert len(ambiguous_ids) == manifest["expected_counts"]["ambiguous_claims"]

    pair_map = {}
    for pair in pairs:
        pair_id = pair["pair_id"]
        assert pair_id not in pair_map
        pair_map[pair_id] = pair
        assert pair["dataset"] == manifest["dataset"]
        assert pair["dataset_version"] == manifest["version"]
        assert pair["decision"] in {"HAS_RELATION", "NO_RELATION", "OTHER_UNSUPPORTED"}
        document_id = pair["document_revision_id"].removesuffix("@v1")
        _, text = documents[document_id]
        assert pair["document_content_hash"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        for side in ("left", "right"):
            start, end = pair[f"{side}_span"]
            assert 0 <= start < end <= len(text), (pair["pair_id"], side)
            assert pair[f"{side}_occurrence_id"] in mention_ids, pair["pair_id"]
        for start, end in pair["evidence_spans"]:
            assert 0 <= start < end <= len(text), (pair["pair_id"], "evidence")

    fact_pairs = set()
    for fact in facts:
        pair = pair_map[fact["pair_id"]]
        assert pair["decision"] == "HAS_RELATION"
        assert fact["pair_id"] not in fact_pairs
        fact_pairs.add(fact["pair_id"])
        _, text = documents[fact["document_id"]]
        assert text[fact["evidence_start"]:fact["evidence_end"]] == fact["evidence_text"]
        assert text[fact["subject_start"]:fact["subject_end"]]
        assert text[fact["object_start"]:fact["object_end"]]

    for item in unsupported:
        assert pair_map[item["pair_id"]]["decision"] == "OTHER_UNSUPPORTED"
        _, text = documents[pair_map[item["pair_id"]]["document_revision_id"].removesuffix("@v1")]
        assert text[item["evidence_start"]:item["evidence_end"]] == item["evidence_text"]

    decisions = Counter(pair["decision"] for pair in pairs)
    by_document = Counter(
        pair["document_revision_id"].removesuffix("@v1") for pair in pairs
    )
    print(json.dumps({
        "status": "valid_calibration_corpus",
        "documents": len(documents),
        "pairs": len(pairs),
        "decisions": decisions,
        "current_schema_facts": len(facts),
        "extension_facts": len(unsupported),
        "entity_mentions": mention_count,
        "coreference_links": len(coreference_links),
        "complete_positive_assertions": len(assertion_ids),
        "positive_assertions_by_tier": {
            "core_direct": len(assertion_bands["core_direct"]),
            "interpretation_band": len(
                assertion_bands["interpretation_band"]
            ),
        },
        "positive_assertions_by_schema": positive_by_schema,
        "positive_assertions_by_structure": positive_by_structure,
        "predicate_specific_negative_claims": len(negative_ids),
        "ambiguous_claims": len(ambiguous_ids),
        "negative_claims_by_reason": negative_by_reason,
        "pairs_by_document": by_document,
        "production_gate_eligible": manifest["production_gate_eligible"],
    }, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
