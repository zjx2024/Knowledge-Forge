from knowledge_forge.graph_retrieval import (
    GRAPH_RETRIEVAL_POLICY_VERSION,
    build_graph_entity_hops,
    graph_neighbor_keys,
    rank_graph_relations,
    select_graph_seed_entities,
)


def test_query_surface_entities_are_stable_graph_seeds():
    entities = [
        {
            "canonical_key": "noise", "name": "unrelated framework",
            "score": 0.95, "lexical_score": 0.0, "dense_score": 0.95,
        },
        {
            "canonical_key": "instrument", "name": "测温仪",
            "score": 0.55, "lexical_score": 0.2, "dense_score": 0.6,
        },
    ]

    selected = select_graph_seed_entities(
        "测温仪最后被放在哪里？", entities, limit=2,
    )

    assert selected[0]["canonical_key"] == "instrument"
    assert selected[0]["query_surface_match"] is True


def test_shared_neighbors_are_preferred_for_bounded_second_hop():
    rows = [
        {
            "source_canonical_key": "seed-a",
            "target_canonical_key": "bridge",
            "lexical_score": 0.2,
        },
        {
            "source_canonical_key": "seed-b",
            "target_canonical_key": "bridge",
            "lexical_score": 0.1,
        },
        {
            "source_canonical_key": "seed-a",
            "target_canonical_key": "leaf",
            "lexical_score": 0.9,
        },
    ]

    assert graph_neighbor_keys(rows, ["seed-a", "seed-b"]) == [
        "bridge", "leaf",
    ]


def test_relation_context_prefers_in_network_edges_and_diversifies_facts():
    rows = [
        {
            "id": "duplicate-high", "source_canonical_key": "a",
            "target_canonical_key": "b", "score": 0.99,
            "lexical_score": 0.9, "evidence_count": 1,
            "evidence_text": "first supporting passage",
        },
        {
            "id": "duplicate-low", "source_canonical_key": "a",
            "target_canonical_key": "b", "score": 0.8,
            "lexical_score": 0.7, "evidence_count": 1,
            "evidence_text": "first supporting passage",
        },
        {
            "id": "second-pair", "source_canonical_key": "a",
            "target_canonical_key": "c", "score": 0.6,
            "lexical_score": 0.5, "evidence_count": 1,
        },
        {
            "id": "text-only", "source_canonical_key": "x",
            "target_canonical_key": "y", "score": 1.0,
            "lexical_score": 1.0, "evidence_count": 2,
        },
    ]
    hops = build_graph_entity_hops(["a", "b", "c"], rows)

    ranked = rank_graph_relations(
        rows,
        seed_scores={"a": 1.0, "b": 0.8, "c": 0.7},
        entity_hops=hops,
        limit=4,
    )

    assert [item["id"] for item in ranked[:2]] == [
        "duplicate-high", "second-pair",
    ]
    assert ranked[0]["network_scope"] == "in_network"
    assert ranked[0]["graph_hop"] == 1
    assert ranked[0]["retrieval_policy_version"] == (
        GRAPH_RETRIEVAL_POLICY_VERSION
    )
    assert ranked[2]["id"] == "text-only"
    assert ranked[-1]["id"] == "duplicate-low"


def test_distinct_relations_for_one_entity_pair_remain_primary_context():
    rows = [
        {
            "id": "placement", "source_canonical_key": "person",
            "target_canonical_key": "instrument", "score": 0.9,
            "relation_semantic_score": 0.9,
            "relation_type": "places", "evidence_text": "placed the instrument",
        },
        {
            "id": "usage", "source_canonical_key": "person",
            "target_canonical_key": "instrument", "score": 0.8,
            "relation_semantic_score": 0.8,
            "relation_type": "uses", "evidence_text": "used the instrument",
        },
        {
            "id": "unrelated", "source_canonical_key": "person",
            "target_canonical_key": "site", "score": 0.1,
            "relation_semantic_score": 0.1,
            "relation_type": "visits", "evidence_text": "visited the site",
        },
    ]

    ranked = rank_graph_relations(
        rows,
        seed_scores={"person": 1.0, "instrument": 0.9, "site": 0.2},
        entity_hops={"person": 0, "instrument": 0, "site": 0},
        limit=2,
    )

    assert [item["id"] for item in ranked] == ["placement", "usage"]


def test_semantically_strong_boundary_edge_can_outrank_weak_seed_internal_edge():
    rows = [
        {
            "id": "weak-internal",
            "source_canonical_key": "seed-a",
            "target_canonical_key": "seed-b",
            "relation_semantic_score": 0.05,
            "score": 0.0,
        },
        {
            "id": "strong-boundary",
            "source_canonical_key": "seed-a",
            "target_canonical_key": "answer",
            "relation_semantic_score": 0.95,
            "score": 0.0,
        },
    ]

    ranked = rank_graph_relations(
        rows,
        seed_scores={"seed-a": 1.0, "seed-b": 0.7},
        entity_hops={"seed-a": 0, "seed-b": 0, "answer": 1},
        limit=2,
    )

    assert [item["id"] for item in ranked] == [
        "strong-boundary", "weak-internal",
    ]
