"""Knowledge Forge adapter with a current-source, text-only degradation path."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any

from ...config import Config
from ...construction_strategies import KNOWLEDGE_FORGE_STRATEGY
from ...graph_policy import GraphNotReadyError
from ...knowledge_forge_descriptions import relation_display_label
from ..community_navigation import (
    COMMUNITY_NAVIGATION_POLICY_VERSION,
    KnowledgeForgeCommunityNavigator,
)
from ..contracts import EvidenceItem, RetrievalResult, SnapshotRef, SourceRef
from ..planning import (
    RetrievalPlan,
    diagnose_primary_payload,
    plan_retrieval,
)


LOGGER = logging.getLogger(__name__)


_VISUAL_REFERENCE_RE = re.compile(
    r"(?:图|表)\s*(?:[A-Za-z]?\s*\d+|[一二三四五六七八九十]+|中|所示|示意)"
    r"|(?:如|见|参见|如下|上|下)(?:图|表)"
    r"|\b(?:figure|fig\.?|table|chart|diagram)\s*"
    r"(?:[A-Z]?\s*\d+|above|below)",
    re.IGNORECASE,
)

_NUMBERED_VISUAL_ANCHOR_RE = re.compile(
    r"(?P<kind>图|表|figure|fig(?:ure)?\.?|table)\s*"
    r"(?:(?:编号|number|no\.?)\s*)?"
    r"(?P<label>[A-Za-z]?\s*(?:\d+(?:\s*[.\-]\s*\d+)*|"
    r"[IVXLCDM]+|[一二三四五六七八九十百]+))",
    re.IGNORECASE,
)


def _numbered_visual_anchors(value: object) -> set[tuple[str, str]]:
    """Extract language-neutral figure/table identifiers from free text."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    anchors: set[tuple[str, str]] = set()
    for match in _NUMBERED_VISUAL_ANCHOR_RE.finditer(text):
        raw_kind = str(match.group("kind") or "").casefold()
        kind = "table" if raw_kind in {"表", "table"} else "figure"
        label = re.sub(r"\s+", "", str(match.group("label") or "")).casefold()
        label = re.sub(r"[.\-]+", ".", label).strip(".")
        if label:
            anchors.add((kind, label))
    return anchors


def _visual_anchor_patterns(
    anchors: set[tuple[str, str]],
) -> list[str]:
    """Create bounded coarse SQL patterns; exact matching stays in Python."""

    patterns: list[str] = []
    for kind, label in sorted(anchors):
        surface = label.replace(".", "%")
        prefixes = (
            ("图", "figure", "fig.", "fig")
            if kind == "figure" else ("表", "table")
        )
        patterns.extend(f"%{prefix}%{surface}%" for prefix in prefixes)
    return list(dict.fromkeys(patterns))[:24]


def _row_matches_visual_anchors(
    row: dict[str, Any], anchors: set[tuple[str, str]],
) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    surface = "\n".join((
        str(row.get("content") or ""),
        str(row.get("section_title") or ""),
        json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    ))
    return bool(anchors & _numbered_visual_anchors(surface))


def _enrich_numbered_visual_recall(
    service: Any,
    project_id: int,
    query: str,
    payload: dict[str, Any],
    *,
    limit: int = 48,
) -> dict[str, Any]:
    """Add a bounded exact-reference channel for numbered figures and tables.

    Dense/lexical retrieval remains the broad recall path.  This channel only
    reconnects a user-supplied visual identifier to current source chunks and
    therefore cannot invent a topic, document, page or graph fact.
    """

    anchors = _numbered_visual_anchors(query)
    if not anchors:
        return {"status": "not_needed", "matched_chunks": 0, "added_chunks": 0}

    chunks = payload.setdefault("chunks", [])
    by_id: dict[int, dict[str, Any]] = {}
    matched_ids: set[int] = set()
    for raw in chunks:
        if not isinstance(raw, dict):
            continue
        chunk_id = int(raw.get("chunk_id") or raw.get("id") or 0)
        if chunk_id > 0:
            by_id[chunk_id] = raw
        if _row_matches_visual_anchors(raw, anchors):
            raw["visual_anchor_score"] = 1.0
            raw["visual_anchor_labels"] = [
                f"{kind}:{label}" for kind, label in sorted(anchors)
            ]
            if chunk_id > 0:
                matched_ids.add(chunk_id)

    loader = getattr(service.db, "agent_visual_reference_chunks", None)
    if not callable(loader):
        return {
            "status": "existing_only" if matched_ids else "unavailable",
            "matched_chunks": len(matched_ids),
            "added_chunks": 0,
        }
    try:
        rows = loader(
            project_id,
            _visual_anchor_patterns(anchors),
            query=query,
            limit=max(1, min(64, int(limit))),
        )
    except Exception as exc:
        LOGGER.warning(
            "Agent numbered-visual recall degraded for project %s: %s",
            project_id, exc,
        )
        return {
            "status": "degraded",
            "matched_chunks": len(matched_ids),
            "added_chunks": 0,
            "reason": str(exc)[:300],
        }

    added = 0
    for value in rows or []:
        row = dict(value)
        if not _row_matches_visual_anchors(row, anchors):
            continue
        chunk_id = int(row.get("id") or row.get("chunk_id") or 0)
        if chunk_id <= 0:
            continue
        existing = by_id.get(chunk_id)
        labels = [f"{kind}:{label}" for kind, label in sorted(anchors)]
        if existing is not None:
            existing["visual_anchor_score"] = 1.0
            existing["visual_anchor_labels"] = labels
            if not existing.get("source_info"):
                existing["source_info"] = _chunk_source_payload(row)
        else:
            candidate = {
                "chunk_id": chunk_id,
                "content": str(row.get("content") or ""),
                "section_title": str(row.get("section_title") or ""),
                "original_filename": str(row.get("original_filename") or ""),
                "dense_score": 0.0,
                "lexical_score": 0.0,
                "visual_anchor_score": 1.0,
                "visual_anchor_labels": labels,
                "source_info": _chunk_source_payload(row),
            }
            chunks.append(candidate)
            by_id[chunk_id] = candidate
            added += 1
        matched_ids.add(chunk_id)
    return {
        "status": "matched" if matched_ids else "no_match",
        "matched_chunks": len(matched_ids),
        "added_chunks": added,
        "anchors": [f"{kind}:{label}" for kind, label in sorted(anchors)],
    }


def _media_asset_payload(value: Any) -> dict[str, Any] | None:
    """Return the browser-safe projection of one persisted image asset."""

    if not isinstance(value, dict) or not str(value.get("url") or "").strip():
        return None

    def dimension(key: str) -> int:
        try:
            return max(0, int(value.get(key) or 0))
        except (TypeError, ValueError):
            return 0

    sources = []
    for source in value.get("sources") or []:
        if not isinstance(source, dict):
            continue
        sources.append({
            "page": source.get("page"),
            "location": str(source.get("location") or ""),
        })
    return {
        "type": "image",
        "url": str(value.get("url") or ""),
        "width": dimension("width"),
        "height": dimension("height"),
        "title": str(value.get("title") or "文档图片"),
        "description": str(value.get("description") or ""),
        "sources": sources,
    }


def _sources(values: Any, *, quote: str = "") -> list[SourceRef]:
    if isinstance(values, dict):
        values = [values]
    result = []
    for value in values or []:
        if not isinstance(value, dict):
            continue
        source = SourceRef.from_dict(value, quote=quote)
        if source.valid:
            result.append(source)
    return result


def _page_label(values: Any) -> str:
    pages = sorted({
        int(value) for value in (values or [])
        if str(value).isdigit() and int(value) > 0
    })
    if not pages:
        return ""
    if len(pages) == 1:
        return f"第 {pages[0]} 页"
    return "第 " + "、".join(str(value) for value in pages[:6]) + (" 等页" if len(pages) > 6 else " 页")


def _chunk_source_payload(chunk: dict[str, Any] | None) -> dict[str, Any]:
    if not chunk:
        return {}
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    content = str(chunk.get("content") or "")
    lines = [line for line in content.splitlines() if line.strip()]
    looks_tabular = bool(
        metadata.get("table")
        or str(metadata.get("chunk_role") or "") == "table_row"
        or "table" in set(metadata.get("block_types") or [])
        or sum(line.count("|") >= 2 for line in lines) >= 2
    )
    return {
        "document_id": int(chunk.get("document_id") or 0),
        "filename": str(chunk.get("original_filename") or ""),
        "document_url": str(chunk.get("storage_path") or ""),
        "revision_id": str(chunk.get("revision_id") or ""),
        "section_title": str(chunk.get("section_title") or ""),
        "pages": list(metadata.get("pages") or []),
        "page_label": _page_label(metadata.get("pages") or []),
        "media_assets": [
            asset for value in (metadata.get("media_assets") or [])
            if (asset := _media_asset_payload(value)) is not None
        ][:12],
        "table": ({
            "title": str(chunk.get("section_title") or "表格内容"),
            "text": content[:12_000],
        } if looks_tabular else None),
    }


def _positive_pages(values: Any) -> set[int]:
    result: set[int] = set()
    for value in values or []:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            result.add(page)
    return result


def _asset_pages(raw: dict[str, Any], metadata: dict[str, Any]) -> set[int]:
    pages = _positive_pages(
        source.get("page")
        for source in (raw.get("sources") or [])
        if isinstance(source, dict)
    )
    if not pages:
        pages = _positive_pages(metadata.get("pages") or [])
    return pages


def _attach_exact_page_media(
    service: Any, project_id: int, items: list[EvidenceItem],
) -> dict[str, Any]:
    """Attach a referenced visual from the same immutable source page.

    Parser image blocks intentionally remain independent retrieval regions.
    A prose relation such as "图 1 展示……" can therefore cite the correct
    sentence while its image lives in another section block on that PDF page.
    This bounded lookup reconnects only that exact-page presentation asset; it
    does not affect ranking, answer generation, or graph facts.
    """

    loader = getattr(service.db, "agent_page_media_assets", None)
    if not callable(loader):
        return {"status": "unavailable", "attached_references": 0}
    document_ids = sorted({
        ref.document_id for item in items for ref in item.source_refs
        if ref.document_id > 0 and ref.pages
    })
    if not document_ids:
        return {"status": "not_needed", "attached_references": 0}
    try:
        rows = loader(project_id, document_ids)
    except Exception as exc:
        LOGGER.warning(
            "Agent exact-page media enrichment degraded for project %s: %s",
            project_id, exc,
        )
        return {
            "status": "degraded", "attached_references": 0,
            "reason": str(exc)[:300],
        }

    page_assets: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for row in rows or []:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        document_id = int(row.get("document_id") or 0)
        revision_id = str(row.get("revision_id") or "")
        if document_id <= 0 or not revision_id:
            continue
        for raw in metadata.get("media_assets") or []:
            if not isinstance(raw, dict):
                continue
            # Decorative/context-only media must not be promoted merely because
            # prose on the same page mentions another figure.
            relevance = str(raw.get("knowledge_relevance") or "").strip()
            if relevance in {"decorative", "context_only"}:
                continue
            asset = _media_asset_payload(raw)
            if asset is None:
                continue
            for page in _asset_pages(raw, metadata):
                page_assets.setdefault((document_id, revision_id, page), []).append(asset)

    attached = 0
    attached_assets = 0
    for item in items:
        updated_refs: list[SourceRef] = []
        for ref in item.source_refs:
            evidence_text = f"{item.content}\n{ref.quote}"
            if not _VISUAL_REFERENCE_RE.search(evidence_text):
                updated_refs.append(ref)
                continue
            candidates: list[dict[str, Any]] = []
            for page in ref.pages:
                candidates.extend(page_assets.get(
                    (ref.document_id, ref.revision_id, int(page)), [],
                ))
            if not candidates:
                updated_refs.append(ref)
                continue
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for asset in [*ref.media_assets, *candidates]:
                payload = dict(asset)
                identity = str(payload.get("url") or "")
                if not identity or identity in seen:
                    continue
                seen.add(identity)
                merged.append(payload)
                if len(merged) >= 12:
                    break
            if len(merged) == len(ref.media_assets):
                updated_refs.append(ref)
                continue
            payload = ref.to_dict()
            payload["media_assets"] = merged
            updated_refs.append(SourceRef.from_dict(payload, quote=ref.quote))
            attached += 1
            attached_assets += len(merged) - len(ref.media_assets)
        item.source_refs = updated_refs
    return {
        "status": "attached" if attached else "no_match",
        "attached_references": attached,
        "attached_assets": attached_assets,
    }


class KnowledgeForgeRetriever:
    """Wrap current KF retrieval without ever reading a stale graph snapshot."""

    strategy = KNOWLEDGE_FORGE_STRATEGY

    def __init__(self, service: Any) -> None:
        self.service = service
        self.community_navigator = KnowledgeForgeCommunityNavigator(service.db)

    def _snapshot(self, project_id: int) -> tuple[SnapshotRef, dict[str, Any] | None, dict[str, Any]]:
        source = self.service.db.agent_project_source_snapshot(project_id)
        reference: dict[str, Any] | None = None
        try:
            reference = self.service._current_knowledge_forge_snapshot_reference(project_id)
        except GraphNotReadyError:
            reference = None
        if reference:
            graph = reference["graph"]
            generation = int(graph.get("ready_generation") or 0)
            namespace_id = str(reference["entity_namespace_id"])
            relation_run_id = str(reference["relation_run_id"])
            identifier = f"kf:{generation}:{namespace_id}:{relation_run_id}"
            details = {
                "entity_namespace_id": namespace_id,
                "relation_run_id": relation_run_id,
                "embedding_model": source["embedding_model"],
                "document_count": source["document_count"],
            }
        else:
            generation = 0
            identifier = f"kf-source:{source['source_hash']}:{source['embedding_model']}"
            details = {
                "embedding_model": source["embedding_model"],
                "document_count": source["document_count"],
                "degraded_reason": "graph_not_current",
            }
        return SnapshotRef(
            strategy=self.strategy, identifier=identifier,
            source_hash=str(source["source_hash"]), generation=generation,
            graph_available=bool(reference), details=details,
        ), reference, source

    def retrieve(
        self, *, run_id: str, project: dict[str, Any], query: str,
        candidate_limit: int, plan: RetrievalPlan | None = None,
    ) -> RetrievalResult:
        project_id = int(project["id"])
        snapshot, reference, source_snapshot = self._snapshot(project_id)
        if source_snapshot["chunk_count"] <= 0 or source_snapshot["embedding_count"] <= 0:
            raise GraphNotReadyError("当前知识库尚无可检索的知识块向量，请先完成文档预处理")
        if reference:
            payload = self.service.search(
                project_id, query, top_k=min(50, max(1, candidate_limit)),
                allow_report_scheduling=False,
            )
        else:
            chunks = self.service.vectors.search(
                project, query, top_k=min(50, max(1, candidate_limit)),
            )
            provenance = self.service.db.get_chunk_provenance(
                project_id,
                [int(item.get("chunk_id") or 0) for item in chunks],
            )
            payload = {
                "chunks": [{
                    **item,
                    "source_info": _chunk_source_payload(
                        provenance.get(int(item.get("chunk_id") or 0)),
                    ),
                } for item in chunks],
                "entities": [], "relations": [], "community_reports": [],
            }

        effective_plan = plan or plan_retrieval(
            query,
            strategy=self.strategy,
            candidate_limit=candidate_limit,
            activation="observe",
        )
        visual_recall = _enrich_numbered_visual_recall(
            self.service,
            project_id,
            query,
            payload,
            limit=max(12, min(48, candidate_limit)),
        )
        primary_probe = diagnose_primary_payload(
            payload,
            plan=effective_plan,
            graph_available=bool(reference),
        )
        community_navigation: dict[str, Any] = {
            "policy_version": COMMUNITY_NAVIGATION_POLICY_VERSION,
            "status": "not_applicable" if not reference else "disabled",
            "synthetic_evidence_count": 0,
        }
        navigation_allowed = bool(
            reference
            and Config.agent_community_navigation_enabled
            and (
                effective_plan.activation != "active"
                or primary_probe["expansion_recommended"]
            )
        )
        if navigation_allowed:
            policy = reference.get("policy_summary") or {}
            try:
                extra_relations, community_navigation = (
                    self.community_navigator.expand(
                        project_id=project_id,
                        graph_generation=int(
                            reference["graph"].get("ready_generation") or 0
                        ),
                        namespace_id=str(reference["entity_namespace_id"]),
                        relation_run_id=str(reference["relation_run_id"]),
                        community_run_id=str(
                            policy.get("community_run_id") or ""
                        ),
                        community_input_hash=str(
                            policy.get("community_input_hash") or ""
                        ),
                        query=query,
                        payload=payload,
                        max_communities=(
                            Config.agent_community_navigation_max_communities
                        ),
                        max_relations=(
                            min(
                                Config.agent_community_navigation_max_relations,
                                max(
                                    1,
                                    int(effective_plan.family_limits["relation"]),
                                ),
                            )
                        ),
                    )
                )
                if extra_relations:
                    provenance = self.service.db.get_chunk_provenance(
                        project_id,
                        sorted({
                            int(item.get("chunk_id") or 0)
                            for item in extra_relations
                            if int(item.get("chunk_id") or 0) > 0
                        }),
                    )
                    knowledge_language = policy.get("knowledge_language")
                    for item in extra_relations:
                        item["source_info"] = _chunk_source_payload(
                            provenance.get(int(item.get("chunk_id") or 0)),
                        )
                        metadata = (
                            item.get("metadata")
                            if isinstance(item.get("metadata"), dict) else {}
                        )
                        item["relation_type_label"] = relation_display_label(
                            item.get("relation_type"), metadata,
                            knowledge_language,
                        )
                        item["description"] = str(
                            item.get("evidence_text") or ""
                        )

                    merged: dict[str, dict[str, Any]] = {}
                    order: list[str] = []
                    for raw in [
                        *(payload.get("relations") or []), *extra_relations,
                    ]:
                        item = dict(raw)
                        key = str(item.get("id") or "")
                        if not key:
                            continue
                        if key not in merged:
                            merged[key] = item
                            order.append(key)
                            continue
                        existing = merged[key]
                        navigation_score = float(
                            item.get("community_navigation_score") or 0.0
                        )
                        if navigation_score > float(
                            existing.get("community_navigation_score") or 0.0
                        ):
                            existing["community_navigation_score"] = (
                                navigation_score
                            )
                            existing["community_navigation"] = dict(
                                item.get("community_navigation") or {}
                            )
                        if not existing.get("source_info") and item.get(
                            "source_info"
                        ):
                            existing["source_info"] = item["source_info"]
                    payload["relations"] = [merged[key] for key in order]
            except Exception as exc:
                # Community navigation is an optional recall layer.  Existing
                # chunk/entity/relation/report retrieval remains valid and
                # must not be turned into an agent outage by an index problem.
                LOGGER.exception(
                    "Knowledge Forge community navigation degraded for project %s",
                    project_id,
                )
                community_navigation = {
                    "policy_version": COMMUNITY_NAVIGATION_POLICY_VERSION,
                    "status": "degraded",
                    "reason": str(exc),
                    "synthetic_evidence_count": 0,
                }
        elif reference and Config.agent_community_navigation_enabled:
            community_navigation = {
                "policy_version": COMMUNITY_NAVIGATION_POLICY_VERSION,
                "status": "not_triggered",
                "reason": "primary_probe_did_not_recommend_expansion",
                "trigger_reasons": list(
                    primary_probe.get("expansion_reasons") or []
                ),
                "synthetic_evidence_count": 0,
            }

        items: list[EvidenceItem] = []
        dropped_without_source = 0
        for row in payload.get("chunks") or []:
            content = str(row.get("content") or "").strip()
            refs = _sources(row.get("source_info"), quote=content)
            if not content or not refs:
                dropped_without_source += 1
                continue
            channels = ["chunk_dense", "chunk_lexical"]
            raw_scores = {
                "chunk_dense": float(row.get("dense_score") or 0.0),
                "chunk_lexical": float(row.get("lexical_score") or 0.0),
            }
            visual_anchor_score = float(row.get("visual_anchor_score") or 0.0)
            if visual_anchor_score > 0:
                channels.append("visual_anchor")
                raw_scores["visual_anchor"] = visual_anchor_score
            items.append(EvidenceItem.create(
                run_id=run_id, evidence_type="chunk", project_id=project_id,
                strategy=self.strategy, snapshot_ref=snapshot,
                record_key=str(row.get("chunk_id") or row.get("object_key") or ""),
                title=str(
                    (row.get("source_info") or {}).get("section_title")
                    or row.get("section_title") or row.get("original_filename") or "原文知识块"
                ),
                content=content, source_refs=refs, retrieval_channels=channels,
                raw_scores=raw_scores,
                metadata={
                    "chunk_id": int(row.get("chunk_id") or 0),
                    "visual_anchor_labels": list(
                        row.get("visual_anchor_labels") or []
                    ),
                },
            ))

        entity_rows = list(payload.get("entities") or [])
        source_rows = self.service.db.agent_entity_sources(
            str((reference or {}).get("entity_namespace_id") or ""),
            [str(row.get("canonical_key") or "") for row in entity_rows],
        ) if reference and entity_rows else {}
        for row in entity_rows:
            key = str(row.get("canonical_key") or "")
            exact_sources = source_rows.get(key) or []
            refs = [
                SourceRef.from_dict(
                    _chunk_source_payload(source),
                    quote=str(source.get("evidence_text") or ""),
                )
                for source in exact_sources
            ]
            refs = [ref for ref in refs if ref.valid]
            if not refs:
                dropped_without_source += 1
                continue
            mention_evidence = "\n".join(
                str(source.get("evidence_text") or "").strip()
                for source in exact_sources if str(source.get("evidence_text") or "").strip()
            )
            content = "\n".join(filter(None, (
                f"实体：{row.get('name') or key}",
                f"类别：{row.get('entity_type_label') or row.get('entity_type') or ''}",
                str(row.get("description") or ""), mention_evidence,
            )))
            items.append(EvidenceItem.create(
                run_id=run_id, evidence_type="entity", project_id=project_id,
                strategy=self.strategy, snapshot_ref=snapshot,
                record_key=key, title=str(row.get("name") or key), content=content,
                entity_keys=[key], source_refs=refs,
                retrieval_channels=["entity_dense", "entity_lexical"],
                raw_scores={
                    "entity_dense": float(row.get("dense_score") or 0.0),
                    "entity_lexical": float(row.get("lexical_score") or 0.0),
                },
            ))

        for row in payload.get("relations") or []:
            relation_id = str(row.get("id") or "")
            content = "\n".join(filter(None, (
                f"{row.get('source_name') or ''} — {row.get('relation_type_label') or row.get('relation_type') or ''} → {row.get('target_name') or ''}",
                str(row.get("evidence_text") or row.get("description") or ""),
            )))
            refs = _sources(row.get("source_info"), quote=str(row.get("evidence_text") or ""))
            if not content or not refs:
                dropped_without_source += 1
                continue
            channels = [
                "relation_graph", "relation_lexical", "relation_semantic",
            ]
            raw_scores = {
                "relation_graph": float(row.get("score") or 0.0),
                "relation_lexical": float(row.get("lexical_score") or 0.0),
                "relation_semantic": float(
                    row.get("relation_semantic_score") or 0.0
                ),
            }
            navigation = row.get("community_navigation")
            navigation = navigation if isinstance(navigation, dict) else {}
            navigation_score = float(
                row.get("community_navigation_score") or 0.0
            )
            if navigation and navigation_score > 0:
                channels.append("community_navigation")
                raw_scores["community_navigation"] = navigation_score
            items.append(EvidenceItem.create(
                run_id=run_id, evidence_type="relation", project_id=project_id,
                strategy=self.strategy, snapshot_ref=snapshot,
                record_key=relation_id, title=(
                    f"{row.get('source_name') or ''} · "
                    f"{row.get('relation_type_label') or row.get('relation_type') or ''} · "
                    f"{row.get('target_name') or ''}"
                ), content=content,
                entity_keys=[
                    str(row.get("source_canonical_key") or ""),
                    str(row.get("target_canonical_key") or ""),
                ], relation_keys=[relation_id], source_refs=refs,
                retrieval_channels=channels,
                raw_scores=raw_scores,
                metadata=(
                    {"community_navigation": navigation}
                    if navigation else {}
                ),
            ))

        for row in payload.get("community_reports") or []:
            report_id = str(row.get("report_id") or "")
            findings = "\n".join(
                str(value.get("summary") or value.get("description") or value)
                for value in (row.get("findings") or [])
                if isinstance(value, (dict, str))
            )
            content = "\n".join(filter(None, (
                str(row.get("summary") or ""), findings,
                str(row.get("limitations") or ""),
            )))
            refs = _sources(row.get("sources"), quote=content[:2000])
            if not content or not refs:
                dropped_without_source += 1
                continue
            items.append(EvidenceItem.create(
                run_id=run_id, evidence_type="community_report", project_id=project_id,
                strategy=self.strategy, snapshot_ref=snapshot,
                record_key=report_id, title=str(row.get("title") or "详细社区报告"),
                content=content,
                entity_keys=list(row.get("cited_entity_keys") or []),
                relation_keys=list(row.get("cited_relation_keys") or []),
                source_refs=refs,
                retrieval_channels=["community_dense", "community_lexical"],
                raw_scores={
                    "community_dense": float(row.get("dense_score") or 0.0),
                    "community_lexical": float(row.get("lexical_score") or 0.0),
                },
                metadata={"community_key": str(row.get("community_key") or "")},
            ))
        media_enrichment = _attach_exact_page_media(
            self.service, project_id, items,
        )
        return RetrievalResult(
            strategy=self.strategy, mode="hybrid" if reference else "text_only",
            snapshot_ref=snapshot, evidence=items,
            diagnostics={
                "graph_used": bool(reference),
                "candidate_count": len(items),
                "dropped_without_source": dropped_without_source,
                "source_snapshot": source_snapshot,
                "community_navigation": community_navigation,
                "page_media_enrichment": media_enrichment,
                "numbered_visual_recall": visual_recall,
                "retrieval_plan_received": bool(plan),
                "primary_probe": primary_probe,
            },
        )

    def assert_current(self, project_id: int, snapshot: SnapshotRef) -> None:
        current, _reference, _source = self._snapshot(project_id)
        if current.to_dict() != snapshot.to_dict():
            raise RuntimeError("问答期间知识库内容发生变化，未发布基于旧快照的回答")
