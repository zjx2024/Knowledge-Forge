"""Current names for the evidence-grounded relation runtime.

Relation ledgers keep their original physical table names for audit
compatibility.  This facade prevents those storage names from becoming the
application's product vocabulary.
"""

from .relation_runtime import (
    LOADED_RELATION_CODE_DIGEST,
    GROUNDED_RELATION_POLICY_SET,
    GroundedRelationRuntime,
    relation_runtime_treatment_digest,
)


grounded_relation_treatment_digest = relation_runtime_treatment_digest

__all__ = [
    "GROUNDED_RELATION_POLICY_SET",
    "GroundedRelationRuntime",
    "LOADED_RELATION_CODE_DIGEST",
    "grounded_relation_treatment_digest",
]
