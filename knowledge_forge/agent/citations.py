"""Deterministic claim/citation validation and user-facing rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..unicode_safety import sanitize_unicode_text
from .contracts import EvidenceItem


ALLOWED_CLAIM_STATUSES = frozenset({"supported", "qualified", "conflict"})


@dataclass(frozen=True)
class ValidatedAnswer:
    answer_text: str
    answer_json: dict[str, Any]
    citations: list[dict[str, Any]]
    source_cards: list[dict[str, Any]]


def validate_and_render(
    raw: dict[str, Any], selected: list[EvidenceItem], *,
    sufficiency_state: str,
) -> ValidatedAnswer:
    evidence = {item.evidence_id: item for item in selected if item.provenance_valid}
    claims = raw.get("claims") if isinstance(raw, dict) else []
    claims = claims if isinstance(claims, list) else []
    valid_claims: list[dict[str, Any]] = []
    for value in claims:
        if not isinstance(value, dict):
            continue
        text = sanitize_unicode_text(str(value.get("text") or "")).strip()[:4000]
        ids = list(dict.fromkeys(
            str(item) for item in (value.get("evidence_ids") or [])
            if str(item) in evidence
        ))
        status = str(value.get("status") or "supported").lower()
        if not text or not ids or status not in ALLOWED_CLAIM_STATUSES:
            continue
        if sufficiency_state in {"retryable", "ambiguous"} and status == "supported":
            status = "qualified"
        elif sufficiency_state == "conflict":
            status = "conflict"
        valid_claims.append({"text": text, "evidence_ids": ids, "status": status})
    citation_index: dict[str, int] = {}
    citation_rows: list[dict[str, Any]] = []
    rendered_claims: list[str] = []
    for claim_index, claim in enumerate(valid_claims):
        indices: list[int] = []
        for evidence_id in claim["evidence_ids"]:
            if evidence_id not in citation_index:
                citation_index[evidence_id] = len(citation_index) + 1
            index = citation_index[evidence_id]
            indices.append(index)
            citation_rows.append({
                "claim_index": claim_index,
                "claim_text": claim["text"],
                "evidence_id": evidence_id,
                "citation_index": index,
                "status": claim["status"],
            })
        prefix = "需要注意：" if claim["status"] == "qualified" else (
            "证据存在分歧：" if claim["status"] == "conflict" else ""
        )
        markers = "".join(f"〔{value}〕" for value in sorted(set(indices)))
        rendered_claims.append(prefix + claim["text"] + markers)

    limitations = [
        sanitize_unicode_text(str(value)).strip()[:1000]
        for value in (raw.get("limitations") or [])
        if str(value or "").strip()
    ] if isinstance(raw, dict) else []
    state_limitations = {
        "retryable": "当前证据只覆盖了问题的一部分，结论按现有来源限定。",
        "ambiguous": "问题范围仍有歧义，回答采用检索证据能够支持的解释。",
        "conflict": "检索到的来源存在冲突，以下内容不代表系统已裁定其中一方。",
    }
    if sufficiency_state in state_limitations:
        limitation = state_limitations[sufficiency_state]
        if limitation not in limitations:
            limitations.append(limitation)
    followups = [
        sanitize_unicode_text(str(value)).strip()[:500]
        for value in (raw.get("follow_up_questions") or [])
        if str(value or "").strip()
    ] if isinstance(raw, dict) else []
    if sufficiency_state == "insufficient":
        answer_text = sanitize_unicode_text(str(raw.get("answer") or "")).strip()
        if not answer_text:
            answer_text = "当前知识库中没有找到足够且可核验的依据，因此不能可靠回答这个问题。"
    else:
        answer_text = "\n\n".join(rendered_claims)
        if answer_text and limitations:
            answer_text += "\n\n说明：" + "；".join(limitations)
    source_cards = []
    for evidence_id, index in sorted(citation_index.items(), key=lambda pair: pair[1]):
        item = evidence[evidence_id]
        source_cards.append({
            "citation_index": index,
            "evidence_id": evidence_id,
            "evidence_type": item.evidence_type,
            "title": item.title,
            "excerpt": item.content[:600],
            "source_refs": [ref.to_dict() for ref in item.source_refs],
        })
    answer_json = {
        "answer": answer_text,
        "answerability": str(raw.get("answerability") or ""),
        "claims": valid_claims,
        "limitations": limitations,
        "follow_up_questions": followups,
        "sufficiency_state": sufficiency_state,
        "source_cards": source_cards,
        "citation_validation": {
            "accepted_claims": len(valid_claims),
            "citation_count": len(citation_index),
            "unknown_or_uncited_claims_removed": max(0, len(claims) - len(valid_claims)),
        },
    }
    return ValidatedAnswer(answer_text, answer_json, citation_rows, source_cards)
