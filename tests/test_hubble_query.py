from urllib.parse import parse_qs, quote, urlparse

from knowledge_forge.service import KnowledgeForgeService


def test_project_provenance_query_requires_multiple_source_documents():
    query = KnowledgeForgeService._project_provenance_gremlin(9)

    assert query.startswith('g.V().has("kf_project_id",9).union(')
    assert '__.bothE().has("kf_project_id",9).dedup().otherV().limit(300)' in query
    assert '__.hasLabel("KFEntity")' in query
    assert 'inE("kf_mentions")' in query
    assert 'values("kf_doc_id").dedup().count().is(gte(2))' in query
    assert 'inE("kf_contains")' in query
    assert 'inE("kf_contains").has("kf_project_id",9).outV().limit(200)' in query
    assert query.endswith(').path().limit(500)')


def test_project_provenance_query_is_safe_to_embed_in_hubble_url():
    query = KnowledgeForgeService._project_provenance_gremlin(9)
    url = f"http://example/graph-management/1/data-analyze?gremlin={quote(query, safe='')}"

    assert parse_qs(urlparse(url).query)["gremlin"] == [query]
