"""Community clustering for the immutable Knowledge Forge strategy snapshot."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re
from typing import Any, Callable
import unicodedata
import uuid

from .construction_strategies import KNOWLEDGE_FORGE_STRATEGY
from .graphrag_community import (
    ADAPTER_VERSION,
    BYOGSnapshot,
    CommunityConfig,
    CommunityDetectionCancelled,
    GraphRAGCommunityRunner,
)
from .knowledge_forge_descriptions import (
    relation_display_label,
    relation_type_label,
)
from .knowledge_language import (
    CHINESE,
    ENGLISH,
    normalize_knowledge_language,
)


COMMUNITY_NARRATIVE_VERSION = "verified-graph-theme-narrative-v3-language-policy"
COMMUNITY_TOPIC_ASSOCIATION_VERSION = "community-topic-association-v2-explainable"
STRATEGY_COMMUNITY_ADAPTER_VERSION = (
    ADAPTER_VERSION + "-strategy-snapshot-v4-explainable-community-links"
)

# Community topic associations are a bounded navigation aid, not extracted
# facts.  The conservative cross-document threshold prevents a generic entity
# vocabulary from making unrelated documents look connected, while the
# same-document threshold can recover section-level thematic continuity.
_TOPIC_ASSOCIATION_COMMUNITY_CAP = 240
_TOPIC_ASSOCIATION_REPRESENTATIVE_CAP = 32
_TOPIC_ASSOCIATION_MATCH_CANDIDATE_CAP = 6
_TOPIC_ASSOCIATION_MATCH_CAP = 4
_TOPIC_ASSOCIATION_MAX_DEGREE = 3
_TOPIC_ASSOCIATION_SAME_DOCUMENT_THRESHOLD = 0.82
_TOPIC_ASSOCIATION_CROSS_DOCUMENT_THRESHOLD = 0.92
_TOPIC_ASSOCIATION_SAME_DOCUMENT_SEMANTIC_OVERRIDE = 0.94
_TOPIC_ASSOCIATION_CROSS_DOCUMENT_SEMANTIC_OVERRIDE = 0.96
_TOPIC_ASSOCIATION_METADATA_ONLY_TYPES = frozenset({
    "Person", "Organization", "Location",
})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _inline_text(value: object, *, limit: int = 0) -> str:
    """Normalize extracted prose without changing the underlying graph data."""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit > 0 and len(text) > limit:
        shortened = text[:limit].rstrip("，,；;：:。.!！？? ")
        return shortened + "…"
    return text


def _quoted_names(
    values: list[str], *, limit: int = 6,
    knowledge_language: object = CHINESE,
) -> str:
    selected = [
        f"“{_inline_text(value, limit=72)}”"
        for value in values[:limit] if _inline_text(value)
    ]
    separator = (
        ", "
        if normalize_knowledge_language(knowledge_language) == ENGLISH
        else "、"
    )
    return separator.join(selected)


def _relation_identity(relation: dict[str, Any]) -> str:
    explicit = str(relation.get("relation_key") or "").strip()
    if explicit:
        return explicit
    return _digest([
        relation.get("source_canonical_key"),
        relation.get("target_canonical_key"),
        relation.get("relation_type"),
        relation.get("document_id"),
        relation.get("revision_id"),
        relation.get("chunk_id"),
        relation.get("evidence_text"),
    ])


def _topic_candidate_communities(
    communities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a deterministic, bounded set of relationship-bearing leaves."""

    candidates = [
        item for item in communities
        if int(item.get("level") or 0) == 0
        and int(item.get("relationship_count") or 0) > 0
        and len({str(key) for key in item.get("entity_keys") or [] if str(key)}) >= 2
        and str(item.get("key") or "")
    ]
    return sorted(
        candidates,
        key=lambda item: (
            -int(item.get("relationship_count") or 0),
            -int(item.get("size") or len(item.get("entity_keys") or [])),
            str(item.get("key") or ""),
        ),
    )[:_TOPIC_ASSOCIATION_COMMUNITY_CAP]


def _topic_community_context(
    communities: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]], dict[str, str], dict[str, Counter[str]],
    dict[str, set[int]], set[tuple[str, str]],
]:
    candidates = _topic_candidate_communities(communities)
    community_by_entity: dict[str, str] = {}
    degree_by_community: dict[str, Counter[str]] = {
        str(item["key"]): Counter() for item in candidates
    }
    documents_by_community: dict[str, set[int]] = {
        str(item["key"]): set() for item in candidates
    }
    for item in candidates:
        community_key = str(item["key"])
        for entity_key in item.get("entity_keys") or []:
            community_by_entity.setdefault(str(entity_key), community_key)

    factual_pairs: set[tuple[str, str]] = set()
    for relation in relation_rows:
        source = str(relation.get("source_canonical_key") or "")
        target = str(relation.get("target_canonical_key") or "")
        source_community = community_by_entity.get(source)
        target_community = community_by_entity.get(target)
        if not source_community or not target_community:
            continue
        if source_community != target_community:
            factual_pairs.add(tuple(sorted((source_community, target_community))))
            continue
        if source == target:
            continue
        degree_by_community[source_community][source] += 1
        degree_by_community[source_community][target] += 1
        document_id = int(relation.get("document_id") or 0)
        if document_id > 0:
            documents_by_community[source_community].add(document_id)
    return (
        candidates, community_by_entity, degree_by_community,
        documents_by_community, factual_pairs,
    )


def community_topic_embedding_keys(
    communities: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
) -> list[str]:
    """Select only bounded central members whose existing vectors are needed."""

    candidates, _membership, degree, _documents, _factual = (
        _topic_community_context(communities, relation_rows)
    )
    selected: list[str] = []
    seen: set[str] = set()
    for community in candidates:
        community_key = str(community["key"])
        members = sorted(
            {str(key) for key in community.get("entity_keys") or [] if str(key)},
            key=lambda key: (-int(degree[community_key].get(key, 0)), key),
        )[:_TOPIC_ASSOCIATION_REPRESENTATIVE_CAP]
        for key in members:
            if key not in seen:
                seen.add(key)
                selected.append(key)
    return selected


def _unit_vector(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in vector):
        return None
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 1e-12:
        return None
    return tuple(item / norm for item in vector)


def _topic_lexical_features(value: object) -> set[str]:
    """Create language-neutral lexical anchors for semantic-link auditing."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    features = {
        "latin:" + token
        for token in re.findall(r"[a-z0-9][a-z0-9_+.-]{2,}", text)
    }
    for sequence in re.findall(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]+", text):
        if len(sequence) == 1:
            continue
        features.update(
            "cjk:" + sequence[index:index + 2]
            for index in range(len(sequence) - 1)
        )
    return features


def _vector_similarity(
    left: tuple[float, ...], right: tuple[float, ...],
) -> float:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )


def _representative_topic_matches(
    left: str, right: str, *,
    members_by_community: dict[str, list[str]],
    embedding_by_key: dict[str, tuple[float, ...]],
    name_by_key: dict[str, str],
    centroids: dict[str, tuple[float, ...]],
) -> list[dict[str, Any]]:
    """Explain a selected centroid link with a bounded set of entity pairs.

    Ranking every 32-by-32 member pair for every association is unnecessary.
    First retain the six members nearest to the opposite centroid, then inspect
    only their Cartesian product.  This keeps explanation work bounded while
    still selecting examples representative of the association itself.
    """

    left_centroid = centroids.get(left)
    right_centroid = centroids.get(right)
    if left_centroid is None or right_centroid is None:
        return []
    left_members = sorted(
        members_by_community.get(left, []),
        key=lambda key: (
            -_vector_similarity(embedding_by_key[key], right_centroid), key,
        ),
    )[:_TOPIC_ASSOCIATION_MATCH_CANDIDATE_CAP]
    right_members = sorted(
        members_by_community.get(right, []),
        key=lambda key: (
            -_vector_similarity(embedding_by_key[key], left_centroid), key,
        ),
    )[:_TOPIC_ASSOCIATION_MATCH_CANDIDATE_CAP]
    candidates = sorted((
        {
            "source_entity_key": left_key,
            "source_entity_name": (
                _inline_text(name_by_key.get(left_key), limit=96) or left_key
            ),
            "target_entity_key": right_key,
            "target_entity_name": (
                _inline_text(name_by_key.get(right_key), limit=96) or right_key
            ),
            "score": round(_vector_similarity(
                embedding_by_key[left_key], embedding_by_key[right_key],
            ), 6),
        }
        for left_key in left_members
        for right_key in right_members
    ), key=lambda item: (
        -float(item["score"]), item["source_entity_key"], item["target_entity_key"],
    ))
    selected: list[dict[str, Any]] = []
    used_left: set[str] = set()
    used_right: set[str] = set()
    for item in candidates:
        source_key = str(item["source_entity_key"])
        target_key = str(item["target_entity_key"])
        if source_key in used_left or target_key in used_right:
            continue
        selected.append(item)
        used_left.add(source_key)
        used_right.add(target_key)
        if len(selected) >= _TOPIC_ASSOCIATION_MATCH_CAP:
            break
    return selected


def build_community_topic_associations(
    communities: list[dict[str, Any]],
    *,
    relation_rows: list[dict[str, Any]],
    embedding_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build sparse semantic navigation links without creating graph facts.

    Existing cross-community facts always take precedence.  Candidate links
    are selected with mutual-nearest preference and a small maximum degree so
    the overview gains useful structure without becoming another dense graph.
    """

    (
        candidates, _membership, degree_by_community,
        documents_by_community, factual_pairs,
    ) = _topic_community_context(communities, relation_rows)
    requested_keys = set(community_topic_embedding_keys(communities, relation_rows))
    embedding_by_key: dict[str, tuple[float, ...]] = {}
    model_by_key: dict[str, str] = {}
    name_by_key: dict[str, str] = {}
    type_by_key: dict[str, str] = {}
    dimension_counts: Counter[int] = Counter()
    for row in embedding_rows:
        key = str(row.get("canonical_key") or "")
        if key not in requested_keys:
            continue
        vector = _unit_vector(row.get("embedding"))
        if vector is None:
            continue
        embedding_by_key[key] = vector
        model_by_key[key] = str(row.get("model_name") or "")
        name_by_key[key] = str(row.get("name") or "")
        type_by_key[key] = str(row.get("primary_type") or "")
        dimension_counts[len(vector)] += 1

    metrics: dict[str, Any] = {
        "policy_version": COMMUNITY_TOPIC_ASSOCIATION_VERSION,
        "candidate_communities": len(candidates),
        "requested_embeddings": len(requested_keys),
        "available_embeddings": len(embedding_by_key),
        "embedded_communities": 0,
        "qualified_pairs": 0,
        "selected_associations": 0,
        "skipped_existing_fact_pairs": len(factual_pairs),
        "same_document_threshold": _TOPIC_ASSOCIATION_SAME_DOCUMENT_THRESHOLD,
        "cross_document_threshold": _TOPIC_ASSOCIATION_CROSS_DOCUMENT_THRESHOLD,
        "same_document_semantic_override": (
            _TOPIC_ASSOCIATION_SAME_DOCUMENT_SEMANTIC_OVERRIDE
        ),
        "cross_document_semantic_override": (
            _TOPIC_ASSOCIATION_CROSS_DOCUMENT_SEMANTIC_OVERRIDE
        ),
        "max_degree": _TOPIC_ASSOCIATION_MAX_DEGREE,
        "metadata_only_communities": 0,
        "lexically_supported_pairs": 0,
    }
    if not candidates or not dimension_counts:
        return [], metrics

    dominant_dimension = sorted(
        dimension_counts.items(), key=lambda item: (-item[1], item[0]),
    )[0][0]
    centroids: dict[str, tuple[float, ...]] = {}
    members_by_community: dict[str, list[str]] = {}
    lexical_features: dict[str, set[str]] = {}
    representative_types: dict[str, set[str]] = {}
    model_names: Counter[str] = Counter()
    for community in candidates:
        community_key = str(community["key"])
        members = sorted(
            {str(key) for key in community.get("entity_keys") or [] if str(key)},
            key=lambda key: (
                -int(degree_by_community[community_key].get(key, 0)), key,
            ),
        )[:_TOPIC_ASSOCIATION_REPRESENTATIVE_CAP]
        embedded_members = [
            key for key in members
            if key in embedding_by_key
            and len(embedding_by_key[key]) == dominant_dimension
        ]
        vectors = [embedding_by_key[key] for key in embedded_members]
        if len(vectors) < 2:
            continue
        mean = [0.0] * dominant_dimension
        for vector in vectors:
            for index, value in enumerate(vector):
                mean[index] += value
        centroid = _unit_vector([value / len(vectors) for value in mean])
        if centroid is None:
            continue
        centroids[community_key] = centroid
        members_by_community[community_key] = embedded_members
        lexical_features[community_key] = set().union(*(
            _topic_lexical_features(name_by_key.get(key)) for key in members
        ))
        representative_types[community_key] = {
            type_by_key[key] for key in members if type_by_key.get(key)
        }
        model_names.update(
            model_by_key[key] for key in members
            if key in model_by_key and model_by_key[key]
        )
    metrics["embedded_communities"] = len(centroids)
    metrics["embedding_dimension"] = dominant_dimension
    embedding_model = (
        sorted(model_names.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if model_names else ""
    )
    metrics["embedding_model"] = embedding_model

    metadata_only = {
        key for key, types in representative_types.items()
        if types and types.issubset(_TOPIC_ASSOCIATION_METADATA_ONLY_TYPES)
    }
    metrics["metadata_only_communities"] = len(metadata_only)
    feature_frequency: Counter[str] = Counter(
        feature
        for features in lexical_features.values()
        for feature in features
    )
    maximum_feature_frequency = max(2, math.ceil(len(centroids) * 0.25))
    distinctive_features = {
        key: {
            feature for feature in features
            if feature_frequency[feature] <= maximum_feature_frequency
        }
        for key, features in lexical_features.items()
    }

    pair_candidates: list[dict[str, Any]] = []
    community_keys = sorted(centroids)
    for left_index, left in enumerate(community_keys):
        for right in community_keys[left_index + 1:]:
            pair = (left, right)
            if pair in factual_pairs:
                continue
            if left in metadata_only or right in metadata_only:
                continue
            shared_documents = (
                documents_by_community.get(left, set())
                & documents_by_community.get(right, set())
            )
            threshold = (
                _TOPIC_ASSOCIATION_SAME_DOCUMENT_THRESHOLD
                if shared_documents
                else _TOPIC_ASSOCIATION_CROSS_DOCUMENT_THRESHOLD
            )
            semantic_override = (
                _TOPIC_ASSOCIATION_SAME_DOCUMENT_SEMANTIC_OVERRIDE
                if shared_documents
                else _TOPIC_ASSOCIATION_CROSS_DOCUMENT_SEMANTIC_OVERRIDE
            )
            score = _vector_similarity(centroids[left], centroids[right])
            if score + 1e-12 < threshold:
                continue
            lexical_support = bool(
                distinctive_features.get(left, set())
                & distinctive_features.get(right, set())
            )
            if not lexical_support and score + 1e-12 < semantic_override:
                continue
            pair_candidates.append({
                "left": left,
                "right": right,
                "score": score,
                "threshold": threshold,
                "shared_document_count": len(shared_documents),
                "lexical_support": lexical_support,
            })
    pair_candidates.sort(
        key=lambda item: (-float(item["score"]), item["left"], item["right"]),
    )
    metrics["qualified_pairs"] = len(pair_candidates)
    metrics["lexically_supported_pairs"] = sum(
        bool(item["lexical_support"]) for item in pair_candidates
    )

    neighbours: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pair_candidates:
        neighbours[item["left"]].append(item)
        neighbours[item["right"]].append(item)
    nearest: dict[str, tuple[str, str]] = {}
    neighbour_rank: dict[tuple[str, tuple[str, str]], int] = {}
    for community_key, values in neighbours.items():
        ordered = sorted(
            values,
            key=lambda item: (
                -float(item["score"]), item["left"], item["right"],
            ),
        )
        for rank, item in enumerate(ordered):
            pair = (item["left"], item["right"])
            neighbour_rank[(community_key, pair)] = rank
        if ordered:
            nearest[community_key] = (ordered[0]["left"], ordered[0]["right"])

    selected: list[dict[str, Any]] = []
    selected_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    selected_degree: Counter[str] = Counter()

    def add(item: dict[str, Any], reason: str) -> bool:
        pair = (str(item["left"]), str(item["right"]))
        if pair in selected_by_pair:
            reasons = selected_by_pair[pair]["selection_reasons"]
            if reason not in reasons:
                reasons.append(reason)
            return False
        if (
            selected_degree[pair[0]] >= _TOPIC_ASSOCIATION_MAX_DEGREE
            or selected_degree[pair[1]] >= _TOPIC_ASSOCIATION_MAX_DEGREE
        ):
            return False
        selected_item = dict(item)
        selected_item["selection_reasons"] = [reason]
        selected.append(selected_item)
        selected_by_pair[pair] = selected_item
        selected_degree.update(pair)
        return True

    # Mutual nearest neighbours are the most defensible semantic links.
    for item in pair_candidates:
        pair = (str(item["left"]), str(item["right"]))
        if nearest.get(pair[0]) == pair and nearest.get(pair[1]) == pair:
            add(item, "mutual_nearest")

    # Give every community with a qualifying neighbour one opportunity to
    # appear, without forcing below-threshold links into the map.
    represented = set(selected_degree)
    for item in pair_candidates:
        left, right = str(item["left"]), str(item["right"])
        if left in represented and right in represented:
            continue
        if add(item, "coverage_nearest"):
            represented.update((left, right))

    # Add only reciprocal top-three neighbours as optional secondary context.
    for item in pair_candidates:
        pair = (str(item["left"]), str(item["right"]))
        if (
            neighbour_rank.get((pair[0], pair), 99) < 3
            and neighbour_rank.get((pair[1], pair), 99) < 3
        ):
            add(item, "reciprocal_top3")

    reason_order = {
        "mutual_nearest": 0,
        "coverage_nearest": 1,
        "reciprocal_top3": 2,
    }
    associations: list[dict[str, Any]] = []
    for item in selected:
        left = str(item["left"])
        right = str(item["right"])
        reasons = sorted(
            set(item.get("selection_reasons") or []),
            key=lambda value: (reason_order.get(str(value), 99), str(value)),
        )
        matches = _representative_topic_matches(
            left, right,
            members_by_community=members_by_community,
            embedding_by_key=embedding_by_key,
            name_by_key=name_by_key,
            centroids=centroids,
        )
        associations.append({
            "id": "community-topic:" + _digest([
                COMMUNITY_TOPIC_ASSOCIATION_VERSION, left, right,
            ])[:24],
            "source_community_key": left,
            "target_community_key": right,
            "score": round(float(item["score"]), 6),
            "threshold": float(item["threshold"]),
            "shared_document_count": int(item["shared_document_count"]),
            "lexical_support": bool(item["lexical_support"]),
            "selection_reasons": reasons,
            "representative_entity_matches": matches,
            "embedding_model": embedding_model,
            "policy_version": COMMUNITY_TOPIC_ASSOCIATION_VERSION,
            "explanation_version": COMMUNITY_TOPIC_ASSOCIATION_VERSION,
            "kind": "community_topic_association",
        })
    associations.sort(key=lambda item: (-float(item["score"]), item["id"]))
    metrics["selected_associations"] = len(associations)
    metrics["explained_associations"] = sum(
        bool(item.get("representative_entity_matches")) for item in associations
    )
    return associations, metrics


def _community_narrative(
    entity_keys: list[str],
    *,
    canonical_by_key: dict[str, dict[str, Any]],
    relations: list[dict[str, Any]],
    knowledge_language: object = CHINESE,
) -> tuple[str, list[str]]:
    """Describe a community as a topic while staying inside verified facts.

    Community reports are displayed hundreds at a time on larger projects.  A
    model call per cluster would make graph publication needlessly slow and
    would introduce unsupported prose.  This extractive narrator instead uses
    the internal topology to identify the topic anchor, names its immediate
    subject matter, and retains one source-backed fact as the explanation.
    """

    language = normalize_knowledge_language(knowledge_language)
    english = language == ENGLISH
    members = [key for key in dict.fromkeys(entity_keys) if key in canonical_by_key]
    if not members:
        return (
            "This community currently has no canonical entities available for a thematic description."
            if english else "该社区暂时没有可用于说明主题的规范实体。",
            [],
        )

    name_by_key = {
        key: _inline_text(canonical_by_key[key].get("name") or key)
        for key in members
    }
    member_set = set(members)
    internal = [
        relation for relation in relations
        if str(relation.get("source_canonical_key") or "") in member_set
        and str(relation.get("target_canonical_key") or "") in member_set
        and str(relation.get("source_canonical_key") or "")
        != str(relation.get("target_canonical_key") or "")
    ]
    if not internal:
        ordered_names = sorted(
            (name_by_key[key] for key in members), key=lambda value: value.casefold(),
        )
        if len(ordered_names) == 1:
            return (
                (
                    f'This community centers on “{_inline_text(ordered_names[0], limit=96)}”. '
                    "The graph currently has no verified internal relations, so its connections to other entities cannot yet be described."
                    if english else
                    f"该社区主要收录“{_inline_text(ordered_names[0], limit=96)}”这一主题。"
                    "当前图谱尚无已验证的内部关系，因此暂不能进一步说明它与其他实体的联系。"
                ),
                ordered_names,
            )
        return (
            (
                "This community includes "
                + _quoted_names(
                    ordered_names, knowledge_language=language,
                )
                + ". The graph currently has no verified internal relations, so the connections among these items cannot yet be described."
                if english else
                "该社区主要收录" + _quoted_names(ordered_names)
                + "等内容。当前图谱尚无已验证的内部关系，"
                  "因此暂不能进一步说明这些内容之间的联系。"
            ),
            ordered_names,
        )

    neighbours: dict[str, set[str]] = defaultdict(set)
    incident_count: Counter[str] = Counter()
    for relation in internal:
        source = str(relation.get("source_canonical_key") or "")
        target = str(relation.get("target_canonical_key") or "")
        neighbours[source].add(target)
        neighbours[target].add(source)
        incident_count[source] += 1
        incident_count[target] += 1

    def entity_rank(key: str) -> tuple[int, int, int, str, str]:
        source_count = int(canonical_by_key[key].get("source_count") or 0)
        return (
            -len(neighbours.get(key, set())),
            -int(incident_count.get(key, 0)),
            -source_count,
            name_by_key[key].casefold(),
            key,
        )

    focal_key = min(members, key=entity_rank)
    focal_name = name_by_key[focal_key]
    direct_neighbours = sorted(
        neighbours.get(focal_key, set()),
        key=lambda key: (
            -sum(
                1 for relation in internal
                if {str(relation.get("source_canonical_key") or ""),
                    str(relation.get("target_canonical_key") or "")}
                == {focal_key, key}
            ),
            entity_rank(key),
        ),
    )
    remaining = sorted(
        member_set.difference({focal_key, *direct_neighbours}), key=entity_rank,
    )
    topic_keys = [focal_key, *direct_neighbours, *remaining]
    topic_names = [name_by_key[key] for key in topic_keys]

    if len(members) == 2 and direct_neighbours:
        intro = (
            f'This community describes the knowledge connection between “{_inline_text(focal_name, limit=96)}” and '
            f'“{_inline_text(name_by_key[direct_neighbours[0]], limit=96)}”.'
            if english else
            f"该社区主要介绍“{_inline_text(focal_name, limit=96)}”与"
            f"“{_inline_text(name_by_key[direct_neighbours[0]], limit=96)}”之间的知识联系。"
        )
    elif direct_neighbours:
        neighbour_names = [name_by_key[key] for key in direct_neighbours]
        shown = _quoted_names(
            neighbour_names, knowledge_language=language,
        )
        suffix = " and related topics" if english else (
            "等内容" if len(neighbour_names) > 6 else "相关内容"
        )
        intro = (
            f'This community centers on “{_inline_text(focal_name, limit=96)}” and highlights {shown}{suffix}.'
            if english else
            f"该社区主要围绕“{_inline_text(focal_name, limit=96)}”展开，"
            f"重点介绍{shown}{suffix}。"
        )
    else:
        intro = (
            f'This community centers on “{_inline_text(focal_name, limit=96)}” and also covers '
            f'{_quoted_names(topic_names[1:], knowledge_language=language)}.'
            if english else
            f"该社区主要围绕“{_inline_text(focal_name, limit=96)}”展开，"
            f"并介绍{_quoted_names(topic_names[1:])}等相关内容。"
        )

    label_counts: Counter[str] = Counter()
    for relation in internal:
        metadata = (
            relation.get("metadata")
            if isinstance(relation.get("metadata"), dict) else {}
        )
        label = _inline_text(relation_display_label(
            str(relation.get("relation_type") or "related_to"), metadata,
            language,
        ))
        if label:
            label_counts[label] += 1
    labels = [
        value for value, _count in sorted(
            label_counts.items(), key=lambda item: (-item[1], item[0].casefold()),
        )[:3]
    ]
    relation_sentence = ""
    if labels:
        relation_sentence = (
            " The graph primarily uses "
            + _quoted_names(
                labels, limit=3, knowledge_language=language,
            )
            + " to express the verified connections among these entities."
            if english else
            "图中主要用" + _quoted_names(labels, limit=3)
            + "来表述这些实体之间已验证的联系。"
        )

    evidence_candidates: list[tuple[tuple[int, int, int, int, str], str]] = []
    for relation in internal:
        evidence = _inline_text(relation.get("evidence_text"))
        if not evidence:
            continue
        source = str(relation.get("source_canonical_key") or "")
        target = str(relation.get("target_canonical_key") or "")
        covered_names = sum(
            bool(name and name.casefold() in evidence.casefold())
            for name in name_by_key.values()
        )
        evidence_candidates.append((
            (
                -covered_names,
                -int(focal_key in {source, target}),
                -min(len(evidence), 320),
                abs(180 - min(len(evidence), 360)),
                _relation_identity(relation),
            ),
            evidence,
        ))
    evidence_sentence = ""
    if evidence_candidates:
        evidence = min(evidence_candidates, key=lambda item: item[0])[1]
        evidence = _inline_text(evidence, limit=320).rstrip("。.!！？? ")
        evidence_sentence = (
            f" Verifiable source point: {evidence}."
            if english else f"可核对的原文要点：{evidence}。"
        )

    return intro + relation_sentence + evidence_sentence, topic_names


def enrich_community_descriptions(
    communities: list[dict[str, Any]],
    *,
    snapshot: dict[str, Any],
    knowledge_language: object = CHINESE,
) -> list[dict[str, Any]]:
    """Attach current narrative descriptions to new or historical communities."""

    canonical_by_key = {
        str(item.get("canonical_key") or ""): item
        for item in snapshot.get("canonicals") or []
        if str(item.get("canonical_key") or "")
    }
    relation_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for relation in snapshot.get("relation_endpoints") or []:
        source = str(relation.get("source_canonical_key") or "")
        target = str(relation.get("target_canonical_key") or "")
        if source not in canonical_by_key or target not in canonical_by_key:
            continue
        relation_index[source].append(relation)
        if target != source:
            relation_index[target].append(relation)

    result: list[dict[str, Any]] = []
    for original in communities:
        item = dict(original)
        entity_keys = [
            str(key) for key in item.get("entity_keys") or []
            if str(key) in canonical_by_key
        ]
        relation_by_identity: dict[str, dict[str, Any]] = {}
        for key in entity_keys:
            for relation in relation_index.get(key, []):
                relation_by_identity.setdefault(
                    _relation_identity(relation), relation,
                )
        narrative, topic_names = _community_narrative(
            entity_keys,
            canonical_by_key=canonical_by_key,
            relations=list(relation_by_identity.values()),
            knowledge_language=knowledge_language,
        )
        item["summary"] = narrative
        item["description"] = narrative
        item["description_source"] = "verified_graph_theme_narrative"
        item["description_version"] = COMMUNITY_NARRATIVE_VERSION
        item["description_language"] = normalize_knowledge_language(
            knowledge_language,
        )
        item["topic_entities"] = topic_names[:8]
        result.append(item)
    return result


def build_knowledge_forge_byog(
    *,
    project_id: int,
    graph_generation: int,
    entity_namespace_id: str,
    snapshot: dict[str, Any],
    config: CommunityConfig,
    knowledge_language: object = CHINESE,
) -> tuple[BYOGSnapshot, dict[int, str], dict[str, dict[str, Any]]]:
    """Adapt text-keyed strategy canonicals to GraphRAG's supported BYOG input."""

    config.validate()
    canonicals = sorted(
        list(snapshot.get("canonicals") or []),
        key=lambda item: str(item.get("canonical_key") or ""),
    )
    canonical_by_key = {
        str(item.get("canonical_key") or ""): item
        for item in canonicals if str(item.get("canonical_key") or "")
    }
    id_by_key = {
        key: index for index, key in enumerate(canonical_by_key, start=1)
    }
    key_by_id = {value: key for key, value in id_by_key.items()}

    relation_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    excluded_self_loops = 0
    for relation in sorted(
        list(snapshot.get("relation_endpoints") or []),
        key=lambda item: str(item.get("relation_key") or ""),
    ):
        source = str(relation.get("source_canonical_key") or "")
        target = str(relation.get("target_canonical_key") or "")
        if source not in canonical_by_key or target not in canonical_by_key:
            raise ValueError("Knowledge Forge 社区输入存在悬空关系端点")
        if source == target:
            excluded_self_loops += 1
            continue
        relation_groups.setdefault(tuple(sorted((source, target))), []).append(relation)

    degree = {key: 0 for key in canonical_by_key}
    incident_units: dict[str, set[str]] = {key: set() for key in canonical_by_key}
    relationships: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for human_id, ((source, target), rows) in enumerate(sorted(relation_groups.items())):
        degree[source] += 1
        degree[target] += 1
        relationship_id = "kf_strategy_edge:" + _digest([
            project_id, entity_namespace_id, source, target,
            sorted(str(row.get("relation_key") or "") for row in rows),
        ])[:32]
        relation_types = sorted({
            str(row.get("relation_type") or "") for row in rows
            if str(row.get("relation_type") or "")
        })
        text_unit_ids = sorted({
            f"chunk:{row.get('revision_id')}:{row.get('chunk_id')}"
            for row in rows
        })
        incident_units[source].update(text_unit_ids)
        incident_units[target].update(text_unit_ids)
        evidence = sorted({
            str(row.get("evidence_text") or "").strip()
            for row in rows if str(row.get("evidence_text") or "").strip()
        })
        display_descriptions = sorted({
            str((row.get("metadata") or {}).get("display_description") or "").strip()
            for row in rows if isinstance(row.get("metadata"), dict)
            and str((row.get("metadata") or {}).get("display_description") or "").strip()
        })
        language = normalize_knowledge_language(knowledge_language)
        description = ("; " if language == ENGLISH else "；").join(
            display_descriptions
            or [
                ("Relation types: " if language == ENGLISH else "关系类别：")
                + (", " if language == ENGLISH else "、").join(
                    relation_type_label(value, language)
                    for value in relation_types
                ),
                *evidence,
            ]
        )[:4000]
        weight = float(len(rows))
        relationships.append({
            "id": relationship_id,
            "human_readable_id": human_id,
            "source": source,
            "target": target,
            "description": description,
            "weight": weight,
            "combined_degree": degree[source] + degree[target],
            "text_unit_ids": text_unit_ids,
        })
        edges.append({
            "stable_edge_id": relationship_id,
            "source_entity_id": id_by_key[source],
            "target_entity_id": id_by_key[target],
            "weight": weight,
            "relation_keys": sorted(
                str(row.get("relation_key") or "") for row in rows
            ),
            "relation_types": relation_types,
            "text_unit_ids": text_unit_ids,
        })

    entities: list[dict[str, Any]] = []
    for human_id, (key, item) in enumerate(canonical_by_key.items()):
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        description = "\n".join(filter(None, [
            f"Name: {item.get('name') or key}",
            f"Type: {item.get('primary_type') or 'GenericEntity'}",
            str(metadata.get("description") or ""),
        ]))[:4000]
        entities.append({
            "id": f"kf_entity:{id_by_key[key]}",
            "human_readable_id": human_id,
            "title": key,
            "type": str(item.get("primary_type") or "GenericEntity"),
            "description": description,
            "text_unit_ids": sorted(incident_units[key]),
            "frequency": int(item.get("source_count") or 0),
            "degree": int(degree[key]),
        })

    input_hash = _digest({
        "project_id": int(project_id),
        "strategy": KNOWLEDGE_FORGE_STRATEGY,
        "strategy_adapter_version": STRATEGY_COMMUNITY_ADAPTER_VERSION,
        "community_narrative_version": COMMUNITY_NARRATIVE_VERSION,
        "knowledge_language": normalize_knowledge_language(
            knowledge_language,
        ),
        "graph_generation": int(graph_generation),
        "entity_namespace_id": str(entity_namespace_id),
        "config": config.as_dict(),
        "canonicals": [
            {
                "key": key,
                "name": canonical_by_key[key].get("name"),
                "type": canonical_by_key[key].get("primary_type"),
                "description": (
                    (canonical_by_key[key].get("metadata") or {}).get("description")
                    if isinstance(canonical_by_key[key].get("metadata"), dict)
                    else ""
                ),
            }
            for key in canonical_by_key
        ],
        "relationships": relationships,
    })
    result = BYOGSnapshot(
        project_id=int(project_id),
        input_hash=input_hash,
        entities=entities,
        relationships=relationships,
        edges=edges,
        metrics={
            "node_count": len(entities),
            "aggregated_edge_count": len(relationships),
            "excluded_self_loops": excluded_self_loops,
            "excluded_isolated_entities": 0,
            "isolated_entity_count": sum(1 for value in degree.values() if value == 0),
        },
    )
    return result, key_by_id, canonical_by_key


def _browser_communities(
    communities: list[dict[str, Any]],
    *,
    key_by_id: dict[int, str],
    canonical_by_key: dict[str, dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    knowledge_language: object = CHINESE,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in communities:
        entity_keys = [
            key_by_id[int(entity_id)]
            for entity_id in item.get("canonical_entity_ids") or []
            if int(entity_id) in key_by_id
        ]
        relationship_keys = list(item.get("relationship_ids") or [])
        level = int(item.get("level") or 0)
        number = int(item.get("graphrag_community") or 0)
        result.append({
            "key": str(item["stable_community_id"]),
            "label": f"社区 {number}",
            "level": level,
            "community": number,
            "parent": item.get("parent_stable_community_id"),
            "size": len(entity_keys),
            "relationship_count": len(relationship_keys),
            "entity_keys": entity_keys,
            "relationship_keys": relationship_keys,
        })
    ordered = sorted(
        result, key=lambda item: (
            int(item["level"]), int(item["community"]), str(item["key"]),
        ),
    )
    # Leiden/GraphRAG community identifiers are internal clustering labels and
    # may be sparse.  User-facing numbering is deliberately compact and
    # deterministic; the stable key and original number remain unchanged for
    # persistence and membership lookups.
    for display_index, item in enumerate(ordered):
        item["display_index"] = display_index
        item["label"] = f"社区 {display_index}"
    return enrich_community_descriptions(
        ordered,
        snapshot={
            "canonicals": list(canonical_by_key.values()),
            "relation_endpoints": relation_rows,
        },
        knowledge_language=knowledge_language,
    )


class KnowledgeForgeCommunityService:
    """Run and persist clustering bound to one ready strategy generation."""

    def __init__(self, database, runner: GraphRAGCommunityRunner | None = None):
        self.database = database
        self.runner = runner or GraphRAGCommunityRunner()

    def detect(
        self,
        *,
        project_id: int,
        graph_generation: int,
        entity_namespace_id: str,
        snapshot: dict[str, Any],
        config: CommunityConfig | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        knowledge_language: object = CHINESE,
    ) -> dict[str, Any]:
        config = config or CommunityConfig()
        knowledge_language = normalize_knowledge_language(
            knowledge_language,
        )
        byog, key_by_id, canonical_by_key = build_knowledge_forge_byog(
            project_id=project_id,
            graph_generation=graph_generation,
            entity_namespace_id=entity_namespace_id,
            snapshot=snapshot,
            config=config,
            knowledge_language=knowledge_language,
        )
        existing = self.database.get_knowledge_forge_community_run(
            project_id, graph_generation, entity_namespace_id,
            input_hash=byog.input_hash,
        )
        if existing:
            return existing
        created = self.database.create_knowledge_forge_community_run(
            run_id="kfcommunity_" + uuid.uuid4().hex,
            project_id=project_id,
            graph_generation=graph_generation,
            entity_namespace_id=entity_namespace_id,
            input_hash=byog.input_hash,
            node_count=byog.metrics["node_count"],
            edge_count=byog.metrics["aggregated_edge_count"],
            config={
                **config.as_dict(),
                "adapter_version": STRATEGY_COMMUNITY_ADAPTER_VERSION,
                "knowledge_language": knowledge_language,
            },
        )
        run_id = str(created["run_id"])
        if str(created.get("status") or "") in {"completed", "completed_empty"}:
            return created
        try:
            if cancel_requested is None:
                raw = self.runner.run(byog, config)
            else:
                raw = self.runner.run(
                    byog, config, cancel_requested=cancel_requested,
                )
            communities = _browser_communities(
                raw.communities,
                key_by_id=key_by_id,
                canonical_by_key=canonical_by_key,
                relation_rows=list(snapshot.get("relation_endpoints") or []),
                knowledge_language=knowledge_language,
            )
            if cancel_requested is not None and cancel_requested():
                raise CommunityDetectionCancelled(
                    "cancelled before community topic association"
                )
            association_error = ""
            try:
                embedding_keys = community_topic_embedding_keys(
                    communities, list(snapshot.get("relation_endpoints") or []),
                )
                embedding_loader = getattr(
                    self.database, "list_grounded_entity_embeddings", None,
                )
                embedding_rows = (
                    embedding_loader(entity_namespace_id, embedding_keys)
                    if callable(embedding_loader) else []
                )
                topic_associations, association_metrics = (
                    build_community_topic_associations(
                        communities,
                        relation_rows=list(
                            snapshot.get("relation_endpoints") or []
                        ),
                        embedding_rows=list(embedding_rows or []),
                    )
                )
            except Exception as exc:
                # This layer is an optional map-navigation enhancement.  A
                # vector read failure must not invalidate the verified graph or
                # its communities; persist the failure for diagnosis instead.
                topic_associations = []
                association_error = str(exc)
                association_metrics = {
                    "policy_version": COMMUNITY_TOPIC_ASSOCIATION_VERSION,
                    "status": "unavailable",
                    "error": association_error,
                }
            status = "completed" if communities else "completed_empty"
            return self.database.complete_knowledge_forge_community_run(
                run_id,
                status=status,
                communities=communities,
                hierarchy_levels=len({
                    int(item["level"]) for item in communities
                }),
                metadata={
                    "graphrag_log_tail": raw.log[-2000:],
                    "input_metrics": byog.metrics,
                    "community_topic_associations": topic_associations,
                    "community_topic_association_metrics": association_metrics,
                    "community_topic_association_error": association_error,
                },
            )
        except Exception as exc:
            self.database.fail_knowledge_forge_community_run(run_id, str(exc))
            raise
