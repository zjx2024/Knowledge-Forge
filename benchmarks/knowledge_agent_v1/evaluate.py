#!/usr/bin/env python3
"""Exercise and score the real public Knowledge Agent V1 HTTP workflow."""

from __future__ import annotations

import argparse
from http.cookiejar import CookieJar
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parent


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load(ROOT / "manifest.json")
    return manifest, [
        *_load((ROOT / manifest["base_questions_path"]).resolve()),
        *_load(ROOT / manifest["probes_path"]),
    ]


def _normal(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


class AgentClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        # A developer machine may export an HTTP proxy without a correct
        # localhost exclusion.  The acceptance client must exercise the local
        # public service directly instead of silently leaving the host.
        self.opener = build_opener(
            ProxyHandler({}), HTTPCookieProcessor(CookieJar()),
        )

    def json(self, path: str, *, method: str = "GET", payload=None) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, method=method, headers=headers)
        with self.opener.open(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("agent API response is not an object")
        return value

    def create_thread(self, project_id: int) -> str:
        value = self.json(
            "/api/agent/threads", method="POST", payload={"project_id": project_id},
        )
        return str((value.get("thread") or {}).get("id") or "")

    def ask(
        self, thread_id: str, question: str, *, mode: str, timeout: float,
    ) -> dict[str, Any]:
        submitted = self.json(
            f"/api/agent/threads/{thread_id}/messages", method="POST",
            payload={"query": question, "mode": mode},
        )
        run_id = str(submitted.get("run_id") or "")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.json(f"/api/agent/runs/{run_id}")
            if str(run.get("status") or "") in {"completed", "failed", "cancelled"}:
                return run
            time.sleep(0.25)
        self.json(f"/api/agent/runs/{run_id}/cancel", method="POST")
        raise TimeoutError(f"agent run exceeded {timeout:.0f}s: {run_id}")


def score_run(question: dict[str, Any], run: dict[str, Any], project_id: int) -> dict[str, Any]:
    answer = str(run.get("answer_text") or "")
    answer_json = run.get("answer_json") if isinstance(run.get("answer_json"), dict) else {}
    claims = answer_json.get("claims") if isinstance(answer_json.get("claims"), list) else []
    evidence = run.get("evidence") if isinstance(run.get("evidence"), list) else []
    citations = run.get("citations") if isinstance(run.get("citations"), list) else []
    evidence_by_id = {str(item.get("evidence_id") or ""): item for item in evidence}
    cited_ids = [str(item.get("evidence_id") or "") for item in citations]
    parseable = all(value and value in evidence_by_id for value in cited_ids)
    complete_claims = [
        claim for claim in claims
        if isinstance(claim, dict)
        and claim.get("evidence_ids")
        and all(str(value) in evidence_by_id for value in claim["evidence_ids"])
    ]
    claim_completeness = len(complete_claims) / len(claims) if claims else (
        1.0 if answer_json.get("sufficiency_state") == "insufficient" else 0.0
    )
    atoms = list(question.get("answer_atoms") or [])
    atom_recall = (
        sum(_normal(atom) in _normal(answer) for atom in atoms) / len(atoms)
        if atoms else None
    )
    refused = bool(
        answer_json.get("sufficiency_state") == "insufficient"
        or (not claims and any(marker in answer for marker in (
            "没有找到足够", "不能可靠回答", "does not contain", "insufficient",
        )))
    )
    cross_project = sum(
        int(item.get("project_id") or 0) != int(project_id) for item in evidence
    )
    source_parseable = all(
        isinstance(item.get("source_refs"), list)
        and all(int(ref.get("document_id") or 0) > 0 for ref in item["source_refs"])
        for item in evidence if item.get("selected")
    )
    expected_refusal = question.get("expected_outcome") == "refusal"
    return {
        "question_id": question["question_id"],
        "status": str(run.get("status") or ""),
        "terminal_success": run.get("status") == "completed",
        "citation_parseable": parseable and source_parseable,
        "claim_citation_completeness": claim_completeness,
        "answer_atom_recall": atom_recall,
        "expected_refusal": expected_refusal,
        "refused": refused,
        "cross_project_evidence": cross_project,
        "answer_chars": len(answer),
        "citation_count": len(set(cited_ids)),
    }


def summarize(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    total = max(1, len(rows))
    answerable = [row for row in rows if row["answer_atom_recall"] is not None]
    no_answer = [row for row in rows if row["expected_refusal"]]
    metrics = {
        "terminal_success_rate": sum(row["terminal_success"] for row in rows) / total,
        "citation_parse_rate": sum(row["citation_parseable"] for row in rows) / total,
        "claim_citation_completeness": sum(
            row["claim_citation_completeness"] for row in rows
        ) / total,
        "answer_atom_recall": sum(
            row["answer_atom_recall"] for row in answerable
        ) / max(1, len(answerable)),
        "no_answer_refusal_rate": sum(row["refused"] for row in no_answer) / max(1, len(no_answer)),
        "cross_project_evidence": sum(row["cross_project_evidence"] for row in rows),
    }
    expected = manifest["acceptance"]
    gates = {
        "terminal_success": metrics["terminal_success_rate"] >= expected["minimum_terminal_success_rate"],
        "citations_parse": metrics["citation_parse_rate"] >= expected["minimum_citation_parse_rate"],
        "claim_citations": metrics["claim_citation_completeness"] >= expected["minimum_claim_citation_completeness"],
        "answer_atoms": metrics["answer_atom_recall"] >= expected["minimum_answer_atom_recall"],
        "no_answer": metrics["no_answer_refusal_rate"] >= expected["minimum_no_answer_refusal_rate"],
        "project_isolation": metrics["cross_project_evidence"] <= expected["maximum_cross_project_evidence"],
    }
    return {"questions": len(rows), "metrics": metrics, "gates": gates, "passed": all(gates.values()), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--mode", choices=("auto", "basic", "local", "global"), default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output")
    parser.add_argument("--assert-acceptance", action="store_true")
    args = parser.parse_args()
    manifest, questions = load_questions()
    if args.limit > 0:
        questions = questions[:args.limit]
    client = AgentClient(args.base_url)
    thread_id = client.create_thread(args.project_id)
    rows = []
    for index, question in enumerate(questions, 1):
        print(f"[{index}/{len(questions)}] {question['question_id']}", flush=True)
        run = client.ask(
            thread_id, str(question["question"]), mode=args.mode,
            timeout=args.timeout,
        )
        rows.append(score_run(question, run, args.project_id))
    report = summarize(rows, manifest)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if args.assert_acceptance and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
