from __future__ import annotations

from knowledge_forge.migrations import MIGRATIONS
from knowledge_forge.relation_shadow_v2 import (
    HYPOTHESES,
    EvidenceSpan,
    VerificationAttempt,
    normalize_verified_hypotheses,
    parse_classifier_response,
    plan_pair_opportunities,
)
from tests.shadow_v2_fixtures import logical_schema, predicate_policy


def _mention(key, surface, start, canonical):
    return {
        "mention_id": f"mention-{key}",
        "occurrence_key": f"occ-{key}",
        "surface": surface,
        "mention_start": start,
        "mention_end": start + len(surface),
        "canonical_id": canonical,
        "assignment_version": "assignment-v1",
        "type": "GenericEntity",
    }


def _predicate(name, **metadata):
    return predicate_policy(name, **metadata)


def _classified(
    *,
    content="Alpha links Beta.",
    revision="revision-1",
    schema=None,
    hypotheses=None,
    canonicals=("entity-alpha", "entity-beta"),
    run_id="run-1",
):
    schema = schema or logical_schema(_predicate("links"))
    beta_start = content.index("Beta")
    mentions = [
        _mention("alpha", "Alpha", content.index("Alpha"), canonicals[0]),
        _mention("beta", "Beta", beta_start, canonicals[1]),
    ]
    pair = plan_pair_opportunities(
        project_id=12,
        document_revision_id=revision,
        chunk_key=f"chunk-{revision}",
        content=content,
        mentions=mentions,
        logical_schema=schema,
        run_id=run_id,
        candidate_policy_version="candidate-v2",
    ).opportunities[0]
    raw_hypotheses = hypotheses or [{
        "subject_occurrence_id": "occ-alpha",
        "predicate": "links",
        "object_occurrence_id": "occ-beta",
        "evidence_spans": [{"start": 0, "end": len(content)}],
        "classifier_confidence": 0.9,
    }]
    parsed = parse_classifier_response(
        [pair],
        {"decisions": [{
            "pair_id": pair.pair_identity,
            "decision": HYPOTHESES,
            "hypotheses": raw_hypotheses,
        }]},
        content=content,
        logical_schema=schema,
        classifier_policy_version="classifier-v2",
        classifier_model_revision="classifier-model-r1",
    )
    return schema, content, pair, list(parsed[0].hypotheses)


def _attempt(hypothesis, *, verdict="SUPPORTED", attempt_no=1):
    return VerificationAttempt(
        attempt_id=f"attempt-{hypothesis.hypothesis_id}-{attempt_no}",
        hypothesis_id=hypothesis.hypothesis_id,
        attempt_no=attempt_no,
        verdict=verdict,
        reason_code="DIRECTLY_ENTAILED" if verdict == "SUPPORTED" else "NOT_ENTAILED",
        supported_evidence_spans=(
            hypothesis.evidence_spans if verdict == "SUPPORTED" else ()
        ),
        independent_call=True,
        verifier_prompt_version="verifier-prompt-v2",
        verifier_schema_version="verifier-schema-v2",
        verifier_parser_version="verifier-parser-v2",
        verifier_model_revision="verifier-model-r1",
    )


def _normalize(schema, pairs, hypotheses, attempts):
    return normalize_verified_hypotheses(
        project_id=12,
        opportunities={item.pair_identity: item for item in pairs},
        hypotheses=hypotheses,
        verification_attempts=attempts,
        logical_schema=schema,
    )


def test_exact_duplicates_collapse_to_one_fact_and_one_evidence() -> None:
    schema, _, pair, hypotheses = _classified()
    result = _normalize(
        schema,
        [pair],
        [hypotheses[0], hypotheses[0]],
        [_attempt(hypotheses[0]), _attempt(hypotheses[0])],
    )

    assert len(result.facts) == 1
    assert len(result.evidence) == 1
    assert result.metrics["exact_duplicates_collapsed"] == 1


def test_symmetric_reverse_hypotheses_collapse_after_endpoint_normalization() -> None:
    schema = logical_schema(_predicate("partners_with", symmetric=True))
    schema, _, pair, hypotheses = _classified(
        schema=schema,
        hypotheses=[
            {
                "subject_occurrence_id": "occ-alpha",
                "predicate": "partners_with",
                "object_occurrence_id": "occ-beta",
                "evidence_spans": [{"start": 0, "end": 17}],
                "classifier_confidence": 0.9,
            },
            {
                "subject_occurrence_id": "occ-beta",
                "predicate": "partners_with",
                "object_occurrence_id": "occ-alpha",
                "evidence_spans": [{"start": 0, "end": 17}],
                "classifier_confidence": 0.9,
            },
        ],
    )
    result = _normalize(
        schema, [pair], hypotheses, [_attempt(item) for item in hypotheses],
    )

    assert len(result.facts) == 1
    assert len(result.evidence) == 2
    assert result.metrics["symmetric_inverse_duplicates_collapsed"] == 1


def test_inverse_predicates_collapse_to_declared_canonical_forward_fact() -> None:
    schema = logical_schema(
        _predicate(
            "supplies",
            inverse_predicate="is_supplied_by",
        ),
        _predicate(
            "is_supplied_by",
            inverse_predicate="supplies",
            canonical_orientation="inverse",
        ),
    )
    schema, _, pair, hypotheses = _classified(
        schema=schema,
        hypotheses=[
            {
                "subject_occurrence_id": "occ-alpha",
                "predicate": "supplies",
                "object_occurrence_id": "occ-beta",
                "evidence_spans": [{"start": 0, "end": 17}],
                "classifier_confidence": 0.9,
            },
            {
                "subject_occurrence_id": "occ-beta",
                "predicate": "is_supplied_by",
                "object_occurrence_id": "occ-alpha",
                "evidence_spans": [{"start": 0, "end": 17}],
                "classifier_confidence": 0.9,
            },
        ],
    )
    result = _normalize(
        schema, [pair], hypotheses, [_attempt(item) for item in hypotheses],
    )

    assert len(result.facts) == 1
    assert result.facts[0].predicate == "supplies"
    assert result.facts[0].subject_canonical_id == "entity-alpha"
    assert result.facts[0].object_canonical_id == "entity-beta"
    assert result.metrics["symmetric_inverse_duplicates_collapsed"] == 1


def test_verified_non_symmetric_reciprocal_facts_are_both_preserved() -> None:
    schema, _, pair, hypotheses = _classified(hypotheses=[
        {
            "subject_occurrence_id": "occ-alpha",
            "predicate": "links",
            "object_occurrence_id": "occ-beta",
            "evidence_spans": [{"start": 0, "end": 17}],
            "classifier_confidence": 0.9,
        },
        {
            "subject_occurrence_id": "occ-beta",
            "predicate": "links",
            "object_occurrence_id": "occ-alpha",
            "evidence_spans": [{"start": 0, "end": 17}],
            "classifier_confidence": 0.9,
        },
    ])
    result = _normalize(
        schema, [pair], hypotheses, [_attempt(item) for item in hypotheses],
    )

    assert len(result.facts) == 2
    assert result.metrics["verified_reciprocal_facts"] == 2
    assert result.metrics["unsupported_reverse_conflicts"] == 0


def test_unsupported_reverse_hypothesis_never_creates_a_second_fact() -> None:
    schema, _, pair, hypotheses = _classified(hypotheses=[
        {
            "subject_occurrence_id": "occ-alpha",
            "predicate": "links",
            "object_occurrence_id": "occ-beta",
            "evidence_spans": [{"start": 0, "end": 17}],
            "classifier_confidence": 0.9,
        },
        {
            "subject_occurrence_id": "occ-beta",
            "predicate": "links",
            "object_occurrence_id": "occ-alpha",
            "evidence_spans": [{"start": 0, "end": 17}],
            "classifier_confidence": 0.9,
        },
    ])
    result = _normalize(
        schema,
        [pair],
        hypotheses,
        [_attempt(hypotheses[0]), _attempt(hypotheses[1], verdict="REJECTED")],
    )

    assert len(result.facts) == 1
    assert result.metrics["unsupported_reverse_conflicts"] == 1
    assert all(item.reason_code != "UNSUPPORTED_REVERSE_CONFLICT" for item in result.facts)


def test_multiple_occurrence_evidence_bind_to_one_semantic_fact() -> None:
    schema, _, first_pair, first_hypotheses = _classified(revision="revision-1")
    _, _, second_pair, second_hypotheses = _classified(
        revision="revision-2", run_id="run-2",
    )
    hypotheses = [first_hypotheses[0], second_hypotheses[0]]
    result = _normalize(
        schema,
        [first_pair, second_pair],
        hypotheses,
        [_attempt(item) for item in hypotheses],
    )

    assert len(result.facts) == 1
    assert len(result.evidence) == 2
    assert {item.document_revision_id for item in result.evidence} == {
        "revision-1", "revision-2",
    }


def test_different_supported_predicates_on_same_pair_are_preserved() -> None:
    schema = logical_schema(_predicate("uses"), _predicate("supplies"))
    schema, _, pair, hypotheses = _classified(
        schema=schema,
        hypotheses=[
            {
                "subject_occurrence_id": "occ-alpha",
                "predicate": predicate,
                "object_occurrence_id": "occ-beta",
                "evidence_spans": [{"start": 0, "end": 17}],
                "classifier_confidence": 0.9,
            }
            for predicate in ("uses", "supplies")
        ],
    )
    result = _normalize(
        schema, [pair], hypotheses, [_attempt(item) for item in hypotheses],
    )

    assert {item.predicate for item in result.facts} == {"uses", "supplies"}
    assert result.metrics["legal_multi_predicate_retained"] == 2


def test_non_reflexive_self_loop_is_rejected_even_if_attempt_is_forged_supported() -> None:
    schema, _, pair, hypotheses = _classified(
        canonicals=("entity-same", "entity-same"),
    )
    result = _normalize(
        schema, [pair], hypotheses, [_attempt(hypotheses[0])],
    )

    assert result.facts == ()
    assert result.metrics["illegal_self_loops"] == 1


def test_batch_replay_and_verifier_retry_are_idempotent() -> None:
    schema, _, pair, hypotheses = _classified()
    first = _attempt(hypotheses[0], attempt_no=1)
    retry = _attempt(hypotheses[0], attempt_no=2)

    once = _normalize(schema, [pair], hypotheses, [first])
    replayed = _normalize(
        schema,
        [pair, pair],
        [hypotheses[0], hypotheses[0]],
        [first, retry, first],
    )

    assert [item.semantic_fact_key for item in once.facts] == [
        item.semantic_fact_key for item in replayed.facts
    ]
    assert [item.evidence_key for item in once.evidence] == [
        item.evidence_key for item in replayed.evidence
    ]


def test_normalization_migration_audits_legacy_duplicates_without_deleting_them() -> None:
    migration = next(
        item for item in MIGRATIONS
        if item.version == "0016_shadow_relation_v2_normalization"
    )

    assert "shadow_relation_normalized_facts" in migration.sql
    assert "shadow_relation_fact_evidence" in migration.sql
    assert "shadow_relation_fact_collapse_audit" in migration.sql
    assert "shadow_relation_legacy_duplicate_audit" in migration.sql
    assert "INSERT INTO shadow_relation_legacy_duplicate_audit" in migration.sql
    assert "HAVING COUNT(*)>1" in migration.sql
    assert "DELETE FROM relation_assertions" not in migration.sql
    assert "ALTER TABLE relation_assertions" not in migration.sql
