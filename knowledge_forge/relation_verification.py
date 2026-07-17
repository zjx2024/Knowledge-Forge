"""Shared relation verification states and deterministic candidate identity."""

from __future__ import annotations

import hashlib


RULES_VERIFIED = "rules_verified"
LLM_VERIFIED = "llm_verified"
MANUAL_VERIFIED = "manual_verified"

REJECTED = "rejected"
UNCERTAIN = "uncertain"
VERIFICATION_FAILED = "verification_failed"
VERIFICATION_MISSING = "verification_missing"
EVIDENCE_UNVERIFIED = "evidence_unverified"

PUBLISHABLE_RELATION_STATUSES = frozenset({
    RULES_VERIFIED,
    LLM_VERIFIED,
    MANUAL_VERIFIED,
})
NON_PUBLISHABLE_RELATION_STATUSES = frozenset({
    REJECTED,
    UNCERTAIN,
    VERIFICATION_FAILED,
    VERIFICATION_MISSING,
    EVIDENCE_UNVERIFIED,
    "unverified",
})

RELATION_VERDICTS = frozenset({
    "supported",
    "contradicted",
    "insufficient_evidence",
})
RELATION_ORIENTATIONS = frozenset({"as_is", "reversed"})


def is_publishable_relation(status: object) -> bool:
    """Return true only for explicitly approved relation states."""

    return str(status or "") in PUBLISHABLE_RELATION_STATUSES


def relation_candidate_id(
    *,
    project_scope: object,
    chunk_key: object,
    source_mention: object,
    target_mention: object,
    proposed_type: object,
    ordinal: int,
    evidence_start: object = None,
    evidence_end: object = None,
    evidence: object = "",
) -> str:
    """Build a stable ID from candidate provenance, not response position."""

    material = "\x1f".join((
        str(project_scope or ""),
        str(chunk_key or ""),
        str(source_mention or ""),
        str(target_mention or ""),
        str(proposed_type or ""),
        str(int(ordinal)),
        "" if evidence_start is None else str(evidence_start),
        "" if evidence_end is None else str(evidence_end),
        str(evidence or ""),
    ))
    return f"relcand_{hashlib.sha256(material.encode('utf-8')).hexdigest()}"
