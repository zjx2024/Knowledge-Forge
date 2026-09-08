"""Current name for construction-context assembly.

The persisted source-context tables deliberately retain historical names so
existing evidence references never change.
"""

from .evidence_context import (
    EvidenceContextManifest,
    build_evidence_context_manifest,
    occurrence_scope,
)


build_grounded_context_manifest = build_evidence_context_manifest

__all__ = [
    "EvidenceContextManifest",
    "build_grounded_context_manifest",
    "occurrence_scope",
]
