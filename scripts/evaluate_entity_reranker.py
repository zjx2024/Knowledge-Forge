"""Small offline smoke benchmark for the optional entity reranker."""

from __future__ import annotations

import json

from knowledge_forge.entity_reranker import EntityReranker


PAIRS = [
    ("NIST", "National Institute of Standards and Technology", "Organization", True),
    ("USGS", "U.S. Geological Survey", "Organization", True),
    ("IBM", "International Business Machines", "Organization", True),
    ("CVE-2024-1234", "CVE 2024 1234", "Standard", True),
    ("聚对苯二甲酸乙二醇酯", "PET", "Material", True),
    ("Microsoft", "Microsoft Azure", "Organization", False),
    ("碳酸锂", "氢氧化锂", "Material", False),
    ("CVE-2024-1234", "CVE-2024-1235", "Standard", False),
    ("上海", "上海浦东新区", "Place", False),
    ("供应商", "客户", "Organization", False),
]


def main() -> None:
    reranker = EntityReranker()
    results = []
    for left, right, entity_type, expected_same in PAIRS:
        score = reranker.score(
            {"name": left, "type": entity_type, "description": "", "aliases": []},
            [{"name": right, "entity_type": entity_type, "description": "", "aliases": []}],
        )[0]
        results.append({
            "left": left,
            "right": right,
            "expected_same": expected_same,
            "reranker_score": round(score, 4),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
