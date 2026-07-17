"""Build a review-only annotation queue without claiming human gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TEMPLATES = (
    ("zh", "{left}导致{right}。", "causes"),
    ("zh", "{left}采用{right}完成处理。", "uses"),
    ("en", "{left} causes {right}.", "causes"),
    ("en", "{left} uses {right} for processing.", "uses"),
)
PAIRS = (
    ("轴承磨损", "泵体振动"),
    ("控制器A", "协议B"),
    ("Service Alpha", "Protocol Beta"),
    ("Model X version 1", "Module Y"),
)


def build_review_queue(size: int = 64) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(size):
        language, template, predicate = TEMPLATES[index % len(TEMPLATES)]
        left, right = PAIRS[(index // len(TEMPLATES)) % len(PAIRS)]
        suffix = index // (len(TEMPLATES) * len(PAIRS))
        source = left if suffix == 0 else f"{left} {suffix}"
        target = right if suffix == 0 else f"{right} {suffix}"
        text = template.format(left=source, right=target)
        rows.append({
            "id": f"review-{index + 1:03d}",
            "split": "dev" if index < int(size * 0.75) else "holdout_candidate",
            "language": language,
            "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "mention_candidates": [
                {"surface": source, "type": "GenericEntity"},
                {"surface": target, "type": "GenericEntity"},
            ],
            "relation_candidates": [{
                "source": source,
                "predicate": predicate,
                "target": target,
                "evidence": text.rstrip("。."),
            }],
            "must_link_candidates": [],
            "cannot_link_candidates": [],
            "label_source": "synthetic_candidate_unreviewed",
            "review_status": "pending_human_confirmation",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    if not 50 <= args.size <= 300:
        raise SystemExit("size must be between 50 and 300")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in build_review_queue(args.size):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
