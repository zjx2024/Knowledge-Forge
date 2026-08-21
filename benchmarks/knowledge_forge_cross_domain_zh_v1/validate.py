#!/usr/bin/env python3
"""Validate the frozen Chinese calibration corpus without model calls."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.knowledge_forge_cross_domain_zh_v1 import build


ROOT = Path(__file__).resolve().parent


def _read(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def validate() -> dict[str, Any]:
    """Return a compact report after checking hashes, spans and label contracts."""

    for path, payload in build.build_payloads().items():
        assert path.read_text(encoding="utf-8") == build._render(payload), path

    manifest = _read("manifest.json")
    ontology = _read(manifest["ontology_path"])
    entity_payload = _read(manifest["annotations"]["entities"])
    coreference = _read(manifest["annotations"]["coreference_links"])
    assertions = _read(manifest["annotations"]["assertions"])
    ambiguous = _read(manifest["annotations"]["ambiguous_claims"])

    assert manifest["production_gate_eligible"] is False
    assert manifest["recall_universe_complete"] is True
    allowed_types = set(map(str, ontology["entity_types"]))
    allowed_predicates = set(map(str, ontology["relation_types"]))

    documents: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    for item in manifest["documents"]:
        raw = (ROOT / item["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["content_sha256"]
        text = raw.decode("utf-8")
        assert any("\u4e00" <= char <= "\u9fff" for char in text)
        documents[str(item["id"])] = text
        split_counts[str(item["split"])] += 1

    mentions: dict[str, dict[str, Any]] = {}
    mention_by_span: dict[tuple[str, int, int], str] = {}
    mention_spans_by_document: dict[str, list[tuple[int, int, str]]] = {}
    canonical_counts = 0
    for document_id, rows in entity_payload["documents"].items():
        text = documents[document_id]
        canonical_counts += len({str(item["canonical_id"]) for item in rows})
        for mention in rows:
            mention_id = str(mention["mention_id"])
            assert mention_id not in mentions
            start, end = map(int, mention["span"])
            assert 0 <= start < end <= len(text)
            assert text[start:end] == mention["surface"]
            assert mention["type"] in allowed_types
            mentions[mention_id] = {**mention, "document_id": document_id}
            mention_by_span[(document_id, start, end)] = str(
                mention["canonical_id"]
            )
            mention_spans_by_document.setdefault(document_id, []).append((
                start, end, str(mention["canonical_id"]),
            ))

    coreference_by_span: dict[tuple[str, int, int], str] = {}
    link_ids: set[str] = set()
    for link in coreference:
        link_id = str(link["link_id"])
        assert link_id not in link_ids
        link_ids.add(link_id)
        document_id = str(link["document_id"])
        start, end = map(int, link["source_span"])
        assert documents[document_id][start:end] == link["source_text"]
        target = mentions[str(link["target_mention_id"])]
        assert target["document_id"] == document_id
        coreference_by_span[(document_id, start, end)] = str(
            target["canonical_id"]
        )

    def _validate_claim(
        item: dict[str, Any], *, require_declared_predicate: bool,
    ) -> None:
        document_id = str(item["document_id"])
        text = documents[document_id]
        evidence_start, evidence_end = map(int, item["evidence_span"])
        assert text[evidence_start:evidence_end] == item["evidence_text"]
        assert str(item["predicate"]).strip()
        if require_declared_predicate:
            assert item["predicate"] in allowed_predicates
        for role in ("subject", "object"):
            start, end = map(int, item[f"{role}_span"])
            canonical_id = str(item[f"{role}_canonical_id"])
            resolved = mention_by_span.get((document_id, start, end))
            if resolved is None:
                resolved = coreference_by_span.get((document_id, start, end))
            if resolved is None:
                enclosing = {
                    candidate_id
                    for mention_start, mention_end, candidate_id
                    in mention_spans_by_document[document_id]
                    if mention_start <= start < end <= mention_end
                }
                resolved = next(iter(enclosing)) if len(enclosing) == 1 else None
            assert resolved == canonical_id, (item, role, resolved)
            assert evidence_start <= start < end <= evidence_end
        assert item["subject_canonical_id"] != item["object_canonical_id"]

    relation_ids: set[str] = set()
    tier_counts: Counter[str] = Counter()
    structure_counts: Counter[str] = Counter()
    unique_facts: set[tuple[str, str, str, str]] = set()
    for relation in assertions["relations"]:
        assertion_id = str(relation["assertion_id"])
        assert assertion_id not in relation_ids
        relation_ids.add(assertion_id)
        _validate_claim(relation, require_declared_predicate=True)
        tier_counts[str(relation["adjudication_tier"])] += 1
        structure_counts[str(relation["structure"])] += 1
        unique_facts.add((
            str(relation["document_id"]),
            str(relation["subject_canonical_id"]),
            str(relation["predicate"]),
            str(relation["object_canonical_id"]),
        ))

    negative_ids: set[str] = set()
    negative_by_document: Counter[str] = Counter()
    negative_by_reason: Counter[str] = Counter()
    for claim in assertions["negative_claims"]:
        claim_id = str(claim["claim_id"])
        assert claim_id not in negative_ids
        negative_ids.add(claim_id)
        _validate_claim(claim, require_declared_predicate=False)
        negative_by_document[str(claim["document_id"])] += 1
        negative_by_reason[str(claim["reason_code"])] += 1

    expected = manifest["expected_counts"]
    assert len(documents) == expected["documents"]
    assert len(mentions) == expected["entity_mentions"]
    assert canonical_counts == expected["canonical_entities"]
    assert len(link_ids) == expected["coreference_links"]
    assert len(relation_ids) == expected["complete_positive_assertions"]
    assert len(unique_facts) == expected["unique_positive_facts"]
    assert len(negative_ids) == expected["predicate_specific_negative_claims"]
    assert tier_counts["core_direct"] == 62
    assert set(split_counts) == {"dev", "holdout"}
    assert set(negative_by_document) == set(documents)
    assert all(count >= 4 for count in negative_by_document.values())
    assert not ambiguous.get("claims")

    return {
        "status": "valid_calibration_corpus",
        "documents": len(documents),
        "splits": dict(sorted(split_counts.items())),
        "entity_mentions": len(mentions),
        "canonical_entities": canonical_counts,
        "coreference_links": len(link_ids),
        "positive_assertions": len(relation_ids),
        "unique_positive_facts": len(unique_facts),
        "assertions_by_tier": dict(sorted(tier_counts.items())),
        "assertions_by_structure": dict(sorted(structure_counts.items())),
        "predicate_specific_negative_claims": len(negative_ids),
        "negative_claims_by_reason": dict(sorted(negative_by_reason.items())),
        "production_gate_eligible": False,
    }


def main() -> int:
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
