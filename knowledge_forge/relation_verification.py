"""Shared relation verification states and deterministic candidate identity."""

from __future__ import annotations

import hashlib


RULES_VERIFIED = "rules_verified"
LLM_VERIFIED = "llm_verified"
MANUAL_VERIFIED = "manual_verified"

RELATION_VERIFICATION_POLICY_VERSION = "relation-verifier-p0-v1"

REJECTED = "rejected"
UNCERTAIN = "uncertain"
VERIFICATION_FAILED = "verification_failed"
VERIFICATION_MISSING = "verification_missing"
EVIDENCE_UNVERIFIED = "evidence_unverified"
LEGACY_UNVERIFIED = "legacy_unverified"

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
    LEGACY_UNVERIFIED,
    "unverified",
})

RELATION_VERDICTS = frozenset({
    "supported",
    "contradicted",
    "insufficient_evidence",
})
RELATION_ORIENTATIONS = frozenset({"as_is", "reversed"})


def is_publishable_relation(
    status: object,
    metadata: object = None,
) -> bool:
    """Return true only for manual review or current-policy automatic review."""

    normalized_status = str(status or "")
    if normalized_status == MANUAL_VERIFIED:
        return True
    if normalized_status not in {RULES_VERIFIED, LLM_VERIFIED}:
        return False
    values = metadata if isinstance(metadata, dict) else {}
    return values.get("verification_policy_version") == RELATION_VERIFICATION_POLICY_VERSION


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
