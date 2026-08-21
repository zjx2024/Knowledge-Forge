"""Current names for the evidence-grounded entity treatment.

The underlying ledger retains its historical ``entity_shadow_*`` table names
because its records are immutable provenance.  New runtime code should import
these names rather than propagating the discontinued prototype terminology.
"""

from .entity_shadow import (
    DatabaseShadowStore,
    ShadowEntityResolver,
    ShadowModelGateway,
    bounded_identity_evidence,
)


DatabaseGroundedStore = DatabaseShadowStore
GroundedEntityResolver = ShadowEntityResolver
GroundedModelGateway = ShadowModelGateway

__all__ = [
    "DatabaseGroundedStore",
    "GroundedEntityResolver",
    "GroundedModelGateway",
    "bounded_identity_evidence",
]
