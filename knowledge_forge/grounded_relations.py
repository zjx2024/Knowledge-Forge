"""Current names for the evidence-grounded relation runtime.

Relation ledgers keep their original physical table names for audit
compatibility.  This facade prevents those storage names from becoming the
application's product vocabulary.
"""

from .relation_shadow_runtime import (
    LOADED_RELATION_CODE_DIGEST,
    SHADOW_RELATION_V2_POLICY_SET,
    ShadowRelationV2Runtime,
    relation_runtime_treatment_digest,
)


GROUNDED_RELATION_POLICY_SET = SHADOW_RELATION_V2_POLICY_SET
GroundedRelationRuntime = ShadowRelationV2Runtime
grounded_relation_treatment_digest = relation_runtime_treatment_digest

__all__ = [
    "GROUNDED_RELATION_POLICY_SET",
    "GroundedRelationRuntime",
    "LOADED_RELATION_CODE_DIGEST",
    "grounded_relation_treatment_digest",
]
