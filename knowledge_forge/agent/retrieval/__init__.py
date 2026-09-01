"""Read-only strategy adapters for the unified evidence protocol."""

from .graphrag_native import GraphRAGNativeRetriever, choose_native_mode
from .knowledge_forge import KnowledgeForgeRetriever

__all__ = ["GraphRAGNativeRetriever", "KnowledgeForgeRetriever", "choose_native_mode"]
