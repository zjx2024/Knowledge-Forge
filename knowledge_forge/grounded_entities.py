"""Current names for the evidence-grounded entity treatment.

The underlying ledger retains its historical ``entity_shadow_*`` table names
because its records are immutable provenance.  New runtime code should import
these names rather than propagating the discontinued prototype terminology.
"""

from .entity_runtime import (
    DatabaseGroundedStore,
    GroundedEntityResolver,
    GroundedModelGateway,
    bounded_identity_evidence,
)


__all__ = [
    "DatabaseGroundedStore",
    "GroundedEntityResolver",
    "GroundedModelGateway",
    "bounded_identity_evidence",
]
