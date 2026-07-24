from urllib.parse import parse_qs, quote, urlparse
import uuid

from knowledge_forge.service import KnowledgeForgeService


def test_project_graph_query_includes_all_entities_and_entity_relationships():
    namespace = str(uuid.UUID(int=9))
    query = KnowledgeForgeService._project_entity_relations_gremlin(9, namespace)

    scope = f'.has("kf_namespace_id","{namespace}").has("kf_project_id",9)'
    assert query.startswith(f'g.V(){scope}.hasLabel("KFEntity").union(')
    assert "__.identity()" in query
    assert f'__.bothE("kf_related"){scope}.dedup().otherV()' in query
    assert query.endswith(").path().limit(1000)")
    assert query.count(scope) == 2
    assert "kf_mentions" not in query
    assert "kf_contains" not in query


def test_project_entity_relations_query_is_safe_to_embed_in_hubble_url():
    query = KnowledgeForgeService._project_entity_relations_gremlin(
        9, str(uuid.UUID(int=9)),
    )
    url = f"http://example/graph-management/1/data-analyze?gremlin={quote(query, safe='')}"

    assert parse_qs(urlparse(url).query)["gremlin"] == [query]
