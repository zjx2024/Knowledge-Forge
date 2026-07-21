#!/usr/bin/env python3
"""Regenerate the Python 3.12 CPU and CUDA constraints from a pip report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = """# Fully resolved for Python 3.12 on Linux x86_64.\n# Regenerate from one resolver run after changing any requirements file:\n#   python3.12 -m pip install --dry-run -c constraints.txt --report /tmp/knowledge-forge-py312-resolver-report.json -r requirements.txt -r requirements-docling.txt -r requirements-entity-resolution.txt\n#   python3.12 scripts/generate_python312_constraints.py --report /tmp/knowledge-forge-py312-resolver-report.json\n# Then run pip check plus the unit, PostgreSQL, and Docker verification suites.\n\n"""


def _locked_lines(report: Path, *, torch_variant: str) -> list[str]:
    payload = json.loads(report.read_text(encoding="utf-8"))
    resolved = {
        item["metadata"]["name"].lower(): (
            item["metadata"]["name"], item["metadata"]["version"]
        )
        for item in payload["install"]
    }
    if resolved.get("torch", (None, None))[1] != "2.7.1+cpu":
        raise ValueError("resolver report must contain torch==2.7.1+cpu")
    if resolved.get("torchvision", (None, None))[1] != "0.22.1+cpu":
        raise ValueError("resolver report must contain torchvision==0.22.1+cpu")
    resolved["torch"] = ("torch", f"2.7.1+{torch_variant}")
    resolved["torchvision"] = ("torchvision", f"0.22.1+{torch_variant}")
    return [
        f"{name}=={version}"
        for _, (name, version) in sorted(resolved.items(), key=lambda item: item[0])
    ]


def _write(path: Path, lines: list[str]) -> None:
    path.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    _write(ROOT / "constraints.txt", _locked_lines(args.report, torch_variant="cpu"))
    _write(ROOT / "constraints-gpu.txt", _locked_lines(args.report, torch_variant="cu128"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
