import inspect

from knowledge_forge.service import KnowledgeForgeService


def test_current_entity_shadow_replays_formal_mentions_characterization():
    source = inspect.getsource(KnowledgeForgeService.build_entity_shadow_namespace)
    assert "list_document_mentions" in source
    assert "SpanFirstEntityExtractor" not in source
