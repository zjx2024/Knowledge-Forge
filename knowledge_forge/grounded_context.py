"""Current name for construction-context assembly.

The persisted source-context tables deliberately retain historical names so
existing evidence references never change.
"""

from .shadow_context import build_shadow_context_manifest, occurrence_scope


build_grounded_context_manifest = build_shadow_context_manifest

__all__ = ["build_grounded_context_manifest", "occurrence_scope"]
